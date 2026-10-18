"""Readers for the files that come with Photoshop actions.

    .pat  patterns   -> {name/id: RGBA uint8 array}
    .abr  brushes    -> {name: BrushPreset(tip, spacing, dynamics...)}

Only what the action player needs is decoded. Big pattern files (Envato
packs are often 100+ MB) are indexed first and a pattern is decoded only
when an action asks for it.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

from .atn import _Reader, Obj, UnitFloat, AtnError


# --------------------------------------------------------------------------
# PackBits (RLE) - Photoshop's compression for channels
# --------------------------------------------------------------------------
def unpack_bits(src: bytes, size: int) -> bytes:
    out = bytearray()
    i, n = 0, len(src)
    while i < n and len(out) < size:
        h = src[i]
        i += 1
        if h < 128:
            out += src[i:i + h + 1]
            i += h + 1
        elif h > 128:
            out += src[i:i + 1] * (257 - h)
            i += 1
    if len(out) < size:
        out += bytes(size - len(out))
    return bytes(out[:size])


def _rle_plane(data: bytes, h: int, w: int, depth: int) -> np.ndarray:
    """RLE plane: h big-endian u16 row lengths, then the PackBits rows."""
    bpr = w * (depth // 8)
    counts = np.frombuffer(data[:2 * h], ">u2").astype(np.int64)
    off = 2 * h
    rows = []
    for c in counts:
        rows.append(unpack_bits(data[off:off + c], bpr))
        off += c
    buf = b"".join(rows)
    return _to_u8(buf, h, w, depth)


def _to_u8(buf: bytes, h: int, w: int, depth: int) -> np.ndarray:
    if depth == 16:
        a = np.frombuffer(buf[:h * w * 2], ">u2").reshape(h, w)
        return (a >> 8).astype(np.uint8)
    if depth == 1:
        bits = np.unpackbits(np.frombuffer(buf, np.uint8)).reshape(h, -1)[:, :w]
        return (bits * 255).astype(np.uint8)
    return np.frombuffer(buf[:h * w], np.uint8).reshape(h, w).copy()


# --------------------------------------------------------------------------
# .pat
# --------------------------------------------------------------------------
@dataclass
class PatternInfo:
    name: str
    ident: str
    mode: int
    width: int
    height: int
    offset: int          # where the virtual memory array list starts
    path: str
    palette: bytes | None = None


def index_patterns(path: str) -> list[PatternInfo]:
    """Names, ids and sizes of every pattern, without decoding pixels."""
    out = []
    with open(path, "rb") as fh:
        head = fh.read(10)
        if head[:4] != b"8BPT":
            raise AtnError("not a Photoshop pattern file")
        count = struct.unpack(">I", head[6:10])[0]
        for _ in range(count):
            ver, mode = struct.unpack(">II", fh.read(8))
            h, w = struct.unpack(">HH", fh.read(4))
            n = struct.unpack(">I", fh.read(4))[0]
            name = fh.read(2 * n).decode("utf-16-be", "replace").rstrip("\x00")
            idl = fh.read(1)[0]
            ident = fh.read(idl).decode("latin-1")
            palette = None
            if mode == 2:                       # indexed colour
                palette = fh.read(768)
            start = fh.tell()
            _, length = struct.unpack(">II", fh.read(8))
            out.append(PatternInfo(name, ident, mode, w, h, start, path, palette))
            fh.seek(start + 8 + length)
    return out


def decode_pattern(info: PatternInfo) -> np.ndarray:
    """RGBA uint8 (h, w, 4)."""
    with open(info.path, "rb") as fh:
        fh.seek(info.offset)
        _, length = struct.unpack(">II", fh.read(8))
        blob = fh.read(length)
    o = 0
    top, left, bottom, right, nch = struct.unpack(">5I", blob[o:o + 20])
    o += 20
    planes = []
    for _ in range(nch + 2):
        if o + 4 > len(blob):
            break
        written = struct.unpack(">I", blob[o:o + 4])[0]
        o += 4
        if not written:
            continue
        clen = struct.unpack(">I", blob[o:o + 4])[0]
        o += 4
        if clen == 0:
            continue
        end = o + clen
        depth = struct.unpack(">I", blob[o:o + 4])[0]
        t, l, b, r = struct.unpack(">4I", blob[o + 4:o + 20])
        d2 = struct.unpack(">H", blob[o + 20:o + 22])[0]
        comp = blob[o + 22]
        data = blob[o + 23:end]
        h, w = b - t, r - l
        depth = d2 or depth
        if h <= 0 or w <= 0:
            o = end
            continue
        if comp == 1:
            planes.append(_rle_plane(data, h, w, depth))
        else:
            planes.append(_to_u8(data, h, w, depth))
        o = end
    if not planes:
        raise AtnError(f"pattern {info.name!r} has no pixels")
    h, w = planes[0].shape
    planes = [p for p in planes if p.shape == (h, w)]
    if info.mode == 3:                          # RGB (+ alpha)
        rgb = planes[:3]
        alpha = planes[3] if len(planes) > 3 else np.full((h, w), 255, np.uint8)
        return np.dstack(rgb + [alpha])
    if info.mode == 2 and info.palette:         # indexed
        pal = np.frombuffer(info.palette, np.uint8).reshape(3, 256).T
        rgb = pal[planes[0]]
        alpha = planes[1] if len(planes) > 1 else np.full((h, w), 255, np.uint8)
        return np.dstack([rgb, alpha])
    g = planes[0]                               # grayscale / bitmap
    alpha = planes[1] if len(planes) > 1 else np.full((h, w), 255, np.uint8)
    return np.dstack([g, g, g, alpha])


class PatternStore:
    """All patterns from the .pat files of a pack, decoded on demand."""

    def __init__(self, paths: list[str], cache_dir: str | None = None):
        self.items: list[PatternInfo] = []
        for p in paths:
            try:
                self.items += index_patterns(p)
            except (OSError, AtnError, struct.error):
                pass
        self.cache_dir = cache_dir
        self._mem: dict[str, np.ndarray] = {}

    def find(self, name: str = "", ident: str = "") -> PatternInfo | None:
        for it in self.items:
            if ident and it.ident == ident:
                return it
        for it in self.items:
            if name and it.name == name:
                return it
        low = (name or "").strip().lower()
        for it in self.items:
            if low and it.name.strip().lower() == low:
                return it
        return None

    def get(self, name: str = "", ident: str = "") -> np.ndarray | None:
        info = self.find(name, ident)
        if info is None:
            return None
        key = info.ident or info.name
        if key in self._mem:
            return self._mem[key]
        arr = None
        cache = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            safe = "".join(c if c.isalnum() else "_" for c in key)[:80]
            cache = os.path.join(self.cache_dir, f"pat_{safe}.npy")
            if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(info.path):
                try:
                    arr = np.load(cache)
                except (OSError, ValueError):
                    arr = None
        if arr is None:
            arr = decode_pattern(info)
            if cache:
                try:
                    np.save(cache, arr)
                except OSError:
                    pass
        self._mem[key] = arr
        return arr


# --------------------------------------------------------------------------
# .abr (version 6+)
# --------------------------------------------------------------------------
@dataclass
class BrushPreset:
    name: str
    tip: np.ndarray | None          # uint8 alpha (h, w), 255 = paint
    diameter: float = 30.0          # px
    spacing: float = 0.25           # fraction of diameter
    angle: float = 0.0              # degrees
    roundness: float = 1.0
    hardness: float = 1.0
    flip_x: bool = False
    flip_y: bool = False
    size_jitter: float = 0.0        # 0..1
    min_size: float = 0.0
    angle_jitter: float = 0.0       # 0..1 (of 360)
    angle_dir: bool = False         # follow stroke direction
    round_jitter: float = 0.0
    scatter: float = 0.0            # fraction of diameter
    scatter_both: bool = False
    count: int = 1
    count_jitter: float = 0.0
    opacity_jitter: float = 0.0
    flow_jitter: float = 0.0
    fg_bg_jitter: float = 0.0
    hue_jitter: float = 0.0
    sat_jitter: float = 0.0
    bri_jitter: float = 0.0
    extra: dict = field(default_factory=dict)


def _pct(v, default=0.0):
    if isinstance(v, UnitFloat):
        return v.value / 100.0 if v.unit == "#Prc" else v.value
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _num(v, default=0.0):
    if isinstance(v, UnitFloat):
        return v.value
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _jitter(obj, key):
    o = obj.get(key) if obj else None
    if isinstance(o, Obj):
        return _pct(o.get("jitter"), 0.0), _pct(o.get("Mnm "), 0.0), o.get("bVTy")
    return 0.0, 0.0, None


def _read_samples(data: bytes, sub: int) -> dict[str, np.ndarray]:
    tips = {}
    o = 0
    while o + 4 <= len(data):
        size = struct.unpack(">I", data[o:o + 4])[0]
        o += 4
        start = o
        end = start + size
        end += (4 - size % 4) % 4
        try:
            n = data[o]
            uid = data[o + 1:o + 1 + n].decode("latin-1")
            o2 = start + (47 if sub == 1 else 301)
            t, l, b, r = struct.unpack(">4i", data[o2:o2 + 16])
            depth = struct.unpack(">H", data[o2 + 16:o2 + 18])[0]
            comp = data[o2 + 18]
            h, w = b - t, r - l
            raw = data[o2 + 19:start + size]
            if 0 < h < 10000 and 0 < w < 10000:
                if comp == 1:
                    tip = _rle_plane(raw, h, w, depth)
                else:
                    tip = _to_u8(raw, h, w, depth)
                tips[uid] = tip
        except (struct.error, IndexError, ValueError):
            pass
        o = end
    return tips


def read_abr(path: str) -> dict[str, BrushPreset]:
    with open(path, "rb") as fh:
        data = fh.read()
    ver, sub = struct.unpack(">HH", data[:4])
    if ver < 6:
        return _read_abr_old(data, ver)
    tips: dict[str, np.ndarray] = {}
    desc: Obj | None = None
    o = 4
    while o + 12 <= len(data):
        if data[o:o + 4] != b"8BIM":
            break
        key = data[o + 4:o + 8]
        n = struct.unpack(">I", data[o + 8:o + 12])[0]
        body = data[o + 12:o + 12 + n]
        if key == b"samp":
            tips.update(_read_samples(body, sub))
        elif key == b"desc":
            r = _Reader(body)
            r.u32()                                     # descriptor version
            try:
                desc = r.descriptor()
            except AtnError:
                desc = None
        o += 12 + n
        o += (4 - n % 4) % 4 if n % 4 and data[o:o + 4] != b"8BIM" else 0

    presets: dict[str, BrushPreset] = {}
    items = desc.get("Brsh") if desc else None
    if not isinstance(items, list):
        for i, (uid, tip) in enumerate(tips.items()):
            presets[f"brush {i + 1}"] = BrushPreset(f"brush {i + 1}", tip, float(max(tip.shape)))
        return presets
    for it in items:
        if not isinstance(it, Obj):
            continue
        name = it.get("Nm  ") or ""
        b = it.get("Brsh")
        tip = None
        bp = BrushPreset(name, None)
        if isinstance(b, Obj):
            uid = b.get("sampledData")
            if uid and uid in tips:
                tip = tips[uid]
            bp.diameter = _num(b.get("Dmtr"), 30.0)
            bp.spacing = _pct(b.get("Spcn"), 0.25)
            bp.angle = _num(b.get("Angl"), 0.0)
            bp.roundness = _pct(b.get("Rndn"), 1.0) or 1.0
            bp.hardness = _pct(b.get("Hrdn"), 1.0)
            bp.flip_x = bool(b.get("flipX", False))
            bp.flip_y = bool(b.get("flipY", False))
            if b.get("Intr") is False:
                bp.spacing = 0.25
        bp.tip = tip
        if it.get("useTipDynamics"):
            bp.size_jitter, _, _ = _jitter(it, "szVr")
            bp.min_size = _pct(it.get("minimumDiameter"), 0.0)
            bp.angle_jitter, _, ctl = _jitter(it, "angleDynamics")
            bp.angle_dir = ctl in (6, 7)                # direction / initial direction
            bp.round_jitter, _, _ = _jitter(it, "roundnessDynamics")
        if it.get("useScatter"):
            bp.scatter = _pct(it.get("Scat") or (it.get("scatterDynamics") or Obj("", {})).get("jitter"), 0.0)
            bp.scatter_both = bool(it.get("bothAxes", False))
            bp.count = int(_num(it.get("Cnt "), 1)) or 1
            bp.count_jitter, _, _ = _jitter(it, "countDynamics")
        if it.get("usePaintDynamics"):
            bp.opacity_jitter, _, _ = _jitter(it, "opVr")
            bp.flow_jitter, _, _ = _jitter(it, "prVr")
        if it.get("useColorDynamics"):
            bp.fg_bg_jitter, _, _ = _jitter(it, "clVr")
            bp.hue_jitter = _pct(it.get("H   "), 0.0)
            bp.sat_jitter = _pct(it.get("Strt"), 0.0)
            bp.bri_jitter = _pct(it.get("Brgh"), 0.0)
        bp.extra = {k: v for k, v in it.items.items() if k not in ("Brsh",)}
        presets[name] = bp
    return presets


def _read_abr_old(data: bytes, ver: int) -> dict[str, BrushPreset]:
    """ABR v1/v2: a list of brushes, sampled ones carry their own pixels."""
    out = {}
    count = struct.unpack(">H", data[2:4])[0]
    o = 4
    for i in range(count):
        try:
            btype, size = struct.unpack(">HI", data[o:o + 6])
            o += 6
            body = data[o:o + size]
            o += size
            if btype != 2:
                continue
            spacing = struct.unpack(">H", body[4:6])[0] / 100.0
            p = 6
            name = f"brush {i + 1}"
            if ver == 2:
                n = struct.unpack(">I", body[p:p + 4])[0]
                name = body[p + 4:p + 4 + 2 * n].decode("utf-16-be", "replace").rstrip("\x00")
                p += 4 + 2 * n
            p += 1                                      # antialias
            p += 8                                      # short bounds
            t, l, b, r = struct.unpack(">4i", body[p:p + 16])
            depth = struct.unpack(">H", body[p + 16:p + 18])[0]
            comp = body[p + 18]
            h, w = b - t, r - l
            raw = body[p + 19:]
            tip = _rle_plane(raw, h, w, depth) if comp else _to_u8(raw, h, w, depth)
            out[name] = BrushPreset(name, tip, float(max(h, w)), spacing or 0.25)
        except (struct.error, ValueError, IndexError):
            break
    return out
