"""
Read a Photoshop file and print/export everything it contains.

    python tools/dump_psd.py refs/source.psd [--out refs] [--target refs/target.jpg]

Writes
    <out>/psd_tree.json        layer tree (bottom -> top), all settings
    <out>/layers/NN_name.png   every layer's pixels (RGBA), NN = order
    <out>/layers/NN_name_mask.png   layer masks
    <out>/embedded/...         files embedded in smart objects
    <out>/displace_map.png     Displace maps embedded in smart filters
    outputs/psd_composite.png  this tool's own composite of the visible layers
and prints the error of that composite against the PSD's flattened image
and against the target picture.

psd-tools is not needed (the reader is in tools/psdread.py, pure NumPy).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import psdread                                  # noqa: E402
from actions import filters as F                # noqa: E402
from actions.atn import Enum, Obj, UnitFloat, _Reader   # noqa: E402

MODES = {"norm": "Nrml", "mul ": "Mltp", "scrn": "Scrn", "over": "Ovrl", "sLit": "SftL",
         "hLit": "HrdL", "lLit": "linearLight", "vLit": "vividLight", "pLit": "pinLight",
         "hMix": "hardMix", "dark": "Drkn", "lite": "Lghn", "idiv": "CBrn", "div ": "CDdg",
         "lbrn": "linearBurn", "lddg": "linearDodge", "diff": "Dfrn", "smud": "Xclu",
         "fsub": "blendSubtraction", "fdiv": "blendDivide", "hue ": "H   ", "sat ": "Strt",
         "colr": "Clr ", "lum ": "Lmns", "dkCl": "darkerColor", "lgCl": "lighterColor",
         "pass": "passThrough", "diss": "Dslv"}
MODE_NAMES = {"norm": "normal", "mul ": "multiply", "scrn": "screen", "over": "overlay",
              "sLit": "soft light", "hLit": "hard light", "lLit": "linear light",
              "vLit": "vivid light", "pLit": "pin light", "hMix": "hard mix", "dark": "darken",
              "lite": "lighten", "idiv": "color burn", "div ": "color dodge",
              "lbrn": "linear burn", "lddg": "linear dodge", "diff": "difference",
              "smud": "exclusion", "fsub": "subtract", "fdiv": "divide", "hue ": "hue",
              "sat ": "saturation", "colr": "color", "lum ": "luminosity", "pass": "pass through",
              "diss": "dissolve", "dkCl": "darker color", "lgCl": "lighter color"}
ADJUST_KEYS = {"levl": "levels", "curv": "curves", "vibA": "vibrance", "hue2": "hue/saturation",
               "blnc": "color balance", "grdm": "gradient map", "brit": "brightness/contrast",
               "mixr": "channel mixer", "post": "posterize", "thrs": "threshold",
               "selc": "selective color", "expA": "exposure", "phfl": "photo filter",
               "nvrt": "invert", "blwh": "black & white", "clrL": "color lookup"}
FILL_KEYS = {"SoCo": "solid color", "GdFl": "gradient fill", "PtFl": "pattern fill"}


# --------------------------------------------------------------------------
# descriptors -> plain JSON
# --------------------------------------------------------------------------
def plain(v):
    if isinstance(v, Obj):
        d = {"_class": v.cls}
        for k, x in v.items.items():
            d[k.strip()] = plain(x)
        return d
    if isinstance(v, list):
        return [plain(x) for x in v]
    if isinstance(v, UnitFloat):
        return {"value": v.value, "unit": v.unit}
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, tuple) and v and v[0] == "data":
        return {"_data": v[1], "bytes": len(v[2])}
    if isinstance(v, tuple) and v and v[0] == "class":
        return {"_class": v[1]}
    if isinstance(v, bytes):
        return {"bytes": len(v)}
    return v


def descriptor(data, offset):
    r = _Reader(data)
    r.o = offset
    return r.descriptor()


# --------------------------------------------------------------------------
# adjustment layers
# --------------------------------------------------------------------------
def read_levels(d):
    ver = struct.unpack_from(">H", d, 0)[0]
    out = []
    names = ["RGB", "R", "G", "B"]
    for i in range(4):
        lo, hi, olo, ohi, g = struct.unpack_from(">5H", d, 2 + i * 10)
        out.append({"channel": names[i], "in_black": lo, "in_white": hi, "out_black": olo,
                    "out_white": ohi, "gamma": g / 100.0})
    return {"version": ver, "channels": out}


def read_curves(d):
    o = 1
    ver = struct.unpack_from(">H", d, o)[0]; o += 2
    bm = struct.unpack_from(">I", d, o)[0]; o += 4
    names = ["RGB", "R", "G", "B"]
    chans = []
    for c in range(32):
        if bm >> c & 1:
            n = struct.unpack_from(">H", d, o)[0]; o += 2
            pts = [struct.unpack_from(">HH", d, o + 4 * k) for k in range(n)]
            o += 4 * n
            chans.append({"channel": names[c] if c < 4 else c,
                          "points": [{"in": p[1], "out": p[0]} for p in pts]})
    return {"version": ver, "channels": chans}


def read_adjustment(key, d):
    try:
        if key == "levl":
            return read_levels(d)
        if key == "curv":
            return read_curves(d)
        if key in ("vibA", "blwh", "expA", "phfl", "clrL", "mixr", "selc"):
            return plain(descriptor(d, 4))
        if key == "hue2":
            ver, colz = struct.unpack_from(">HB", d, 0)
            mh, ms, ml = struct.unpack_from(">hhh", d, 4 + (6 if colz else 6))
            return {"colorize": bool(colz), "master": {"hue": mh, "saturation": ms, "lightness": ml},
                    "note": "per-range values not decoded"}
        if key == "blnc":
            v = struct.unpack_from(">9hB", d, 0)
            return {"shadows": v[0:3], "midtones": v[3:6], "highlights": v[6:9],
                    "preserve_luminosity": bool(v[9])}
        if key == "brit":
            b, c = struct.unpack_from(">hh", d, 0)
            return {"brightness": b, "contrast": c}
        if key == "post":
            return {"levels": struct.unpack_from(">H", d, 0)[0]}
        if key == "thrs":
            return {"level": struct.unpack_from(">H", d, 0)[0]}
        if key == "nvrt":
            return {}
        if key == "grdm":
            return {"readable": False, "note": "gradient map stops not decoded", "bytes": len(d)}
    except Exception as exc:                   # noqa: BLE001
        return {"readable": False, "error": str(exc)}
    return {"readable": False, "bytes": len(d)}


# --------------------------------------------------------------------------
# smart objects and smart filters
# --------------------------------------------------------------------------
def read_smart(L, out_dir, idx):
    raw = L.info.get("SoLd") or L.info.get("SoLE")
    if raw is None:
        return None
    try:
        d = descriptor(raw, 12)
    except Exception as exc:                   # noqa: BLE001
        return {"readable": False, "error": str(exc)}
    info = {"source_id": d.items.get("Idnt"), "placed_id": d.items.get("placed"),
            "size": plain(d.items.get("Sz  ")), "transform": plain(d.items.get("Trnf"))}
    fx = d.items.get("filterFX")
    filters = []
    if isinstance(fx, Obj):
        info["filter_mask"] = {"enabled": fx.items.get("filterMaskEnable"),
                               "linked": fx.items.get("filterMaskLinked")}
        for k, f in enumerate(fx.items.get("filterFXList") or []):
            flt = f.items.get("Fltr")
            bo = f.items.get("blendOptions")
            entry = {"order": k + 1, "name": f.items.get("Nm  "), "enabled": f.items.get("enab"),
                     "blend": plain(bo), "filter_class": flt.cls if isinstance(flt, Obj) else None,
                     "settings": plain(flt), "readable": isinstance(flt, Obj)}
            # embedded displacement map
            if isinstance(flt, Obj) and "DspD" in flt.items:
                data = flt.items["DspD"][2]
                mp = decode_displace_map(data)
                if mp is not None:
                    path = os.path.join(out_dir, f"displace_map_L{idx:02d}.png")
                    cv2.imwrite(path, mp)
                    entry["embedded_map_png"] = os.path.relpath(path)
                    entry["embedded_map_size"] = [int(mp.shape[1]), int(mp.shape[0])]
            if isinstance(flt, Obj) and "DspF" in flt.items:
                p = flt.items["DspF"][2].decode("utf-16-be", "ignore")
                entry["original_map_path"] = re.sub(r"[^\x20-\x7e]", "", p)
            filters.append(entry)
    info["smart_filters"] = filters
    return info


def decode_displace_map(data):
    """Embedded Displace map: <w,h,channels> little-endian header, then
    planar 8-bit channels."""
    try:
        w, h, c = struct.unpack_from("<IIH", data, 0)
        # header is (1000, 750) = (rows, cols) in practice; check both
        body = np.frombuffer(data, np.uint8, offset=10)
        for (hh, ww) in ((w, h), (h, w)):
            if body.size >= hh * ww:
                plane = body[:hh * ww].reshape(hh, ww)
                if np.abs(np.diff(plane.astype(int), axis=0)).mean() < 40:
                    return plane
    except Exception:                          # noqa: BLE001
        return None
    return None


def global_blocks(info):
    b = info["buf"]
    o = 26
    o += 4 + struct.unpack_from(">I", b, o)[0]
    o += 4 + struct.unpack_from(">I", b, o)[0]
    big = info["ver"] == 2
    lm_len = struct.unpack_from(">Q" if big else ">I", b, o)[0]
    lm_start = o + (8 if big else 4)
    lm_end = lm_start + lm_len
    li = struct.unpack_from(">Q" if big else ">I", b, lm_start)[0]
    p = lm_start + (8 if big else 4) + li
    p += 4 + struct.unpack_from(">I", b, p)[0]
    out = {}
    while p + 12 <= lm_end:
        sig = b[p:p + 4]
        if sig not in (b"8BIM", b"8B64"):
            break
        key = b[p + 4:p + 8].decode("latin1")
        ln = struct.unpack_from(">I", b, p + 8)[0]
        out[key] = (p + 12, ln)
        p += 12 + ln
        p += (-ln) % 4
    return out


def extract_embedded(info, out_dir):
    b = info["buf"]
    blocks = global_blocks(info)
    files = []
    for key in ("lnk2", "lnkD", "lnk3"):
        if key not in blocks:
            continue
        s, ln = blocks[key]
        q, end = s, s + ln
        while q + 8 < end:
            L = struct.unpack_from(">Q", b, q)[0]
            st = r = q + 8
            typ = b[r:r + 4].decode("latin1"); r += 4
            ver = struct.unpack_from(">I", b, r)[0]; r += 4
            pl = b[r]; uid = b[r + 1:r + 1 + pl].decode(); r += 1 + pl
            nn = struct.unpack_from(">I", b, r)[0]; r += 4
            fname = b[r:r + 2 * nn].decode("utf-16-be").rstrip("\x00"); r += 2 * nn
            r += 8
            dlen = struct.unpack_from(">Q", b, r)[0]; r += 8
            hasdesc = b[r]; r += 1
            if hasdesc:
                r += 4
                rd = _Reader(b[r:])
                rd.descriptor()
                r += rd.o
            data = b[r:r + dlen]
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, re.sub(r'[\\/:*?"<>|]', "_", fname))
            with open(path, "wb") as fh:
                fh.write(data)
            files.append({"block": key, "type": typ, "id": uid, "name": fname, "bytes": dlen,
                          "saved": os.path.relpath(path)})
            q = st + L
            q += (-L) % 4
    # Photoshop's cached input of smart filters (FEid)
    if "FEid" in blocks:
        files.append({"block": "FEid", "note": "cached smart-filter input image (decoded by "
                      "tools/psd_to_recipe.py)", "bytes": blocks["FEid"][1]})
    return files


# --------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------
def layer_kind(L):
    sec = L.info.get("lsct")
    if sec:
        t = struct.unpack_from(">I", sec, 0)[0]
        if t in (1, 2):
            return "group"
        if t == 3:
            return "group_end"
    if "SoLd" in L.info or "SoLE" in L.info or "PlLd" in L.info:
        return "smart_object"
    for k in ADJUST_KEYS:
        if k in L.info:
            return "adjustment"
    for k in FILL_KEYS:
        if k in L.info:
            return "fill"
    if "TySh" in L.info:
        return "text"
    if "vmsk" in L.info or "vsms" in L.info:
        return "shape"
    return "pixel"


def build_tree(info, out_dir, export=True):
    layers = info["layers"]
    root = {"name": "(document)", "type": "group", "children": [], "mode": "pass through"}
    stack = [root]
    lay_dir = os.path.join(out_dir, "layers")
    if export:
        os.makedirs(lay_dir, exist_ok=True)
    for i, L in enumerate(layers):
        kind = layer_kind(L)
        if kind == "group_end":
            node = {"name": "(group start)", "type": "group", "children": []}
            stack[-1]["children"].append(node)
            stack.append(node)
            continue
        node = {"index": i, "name": L.name, "type": kind,
                "visible": not bool(L.flags & 2), "opacity": round(L.opacity / 255 * 100, 1),
                "opacity_255": L.opacity, "blend_mode": MODE_NAMES.get(L.mode, L.mode),
                "blend_key": L.mode, "clipping": bool(L.clip),
                "bbox": {"top": L.top, "left": L.left, "bottom": L.bottom, "right": L.right}}
        if "iOpa" in L.info:
            node["fill_opacity"] = round(L.info["iOpa"][0] / 255 * 100, 1)
        if L.mask:
            t, l, bt, r, defc, flags = L.mask
            node["mask"] = {"bbox": [t, l, bt, r], "default_color": defc,
                            "disabled": bool(flags & 2), "empty": (bt - t) * (r - l) == 0}
        if L.blendif and any(L.blendif[k] not in (0, 255) for k in range(len(L.blendif))):
            node["blend_if"] = L.blendif.hex()
        fx = L.info.get("lfx2") or L.info.get("lmfx")
        if fx:
            try:
                node["effects"] = plain(descriptor(fx, 8))
            except Exception as exc:          # noqa: BLE001
                node["effects"] = {"readable": False, "error": str(exc)}
        else:
            node["effects"] = None
        for k, nm in ADJUST_KEYS.items():
            if k in L.info:
                node["adjustment"] = {"kind": nm, "key": k, "settings": read_adjustment(k, L.info[k])}
        for k, nm in FILL_KEYS.items():
            if k in L.info:
                try:
                    node["fill"] = {"kind": nm, "settings": plain(descriptor(L.info[k], 4))}
                except Exception as exc:      # noqa: BLE001
                    node["fill"] = {"kind": nm, "readable": False, "error": str(exc)}
        if kind == "smart_object":
            node["smart_object"] = read_smart(L, out_dir, i)
        if kind == "group":
            sec = L.info["lsct"]
            gm = sec[8:12].decode("latin1") if len(sec) >= 12 else "pass"
            node["blend_mode"] = MODE_NAMES.get(gm, gm)
            node["blend_key"] = gm
            node["collapsed"] = struct.unpack_from(">I", sec, 0)[0] == 2
            cur = stack.pop()
            node["children"] = cur["children"]
            stack[-1]["children"].remove(cur)
            stack[-1]["children"].append(node)
        if export and kind != "group":
            px = psdread.rgba(info, L)
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", L.name).strip("_") or "layer"
            if px is not None and px.size:
                path = os.path.join(lay_dir, f"{i:02d}_{safe}.png")
                cv2.imwrite(path, cv2.cvtColor(px, cv2.COLOR_RGBA2BGRA))
                node["png"] = os.path.relpath(path)
            if -2 in L.pix:
                path = os.path.join(lay_dir, f"{i:02d}_{safe}_mask.png")
                cv2.imwrite(path, L.pix[-2])
                node["mask_png"] = os.path.relpath(path)
        if kind != "group":
            stack[-1]["children"].append(node)
    return root


# --------------------------------------------------------------------------
# compositing (visible layers, bottom -> top)
# --------------------------------------------------------------------------
def _curve_lut(points):
    xs = np.array([p["in"] for p in points], float)
    ys = np.array([p["out"] for p in points], float)
    n = len(xs)
    if n == 2:
        return np.interp(np.arange(256), xs, ys) / 255.0
    hh = np.diff(xs)
    a = np.zeros((n, n)); r = np.zeros(n)
    a[0, 0] = a[-1, -1] = 1
    for i in range(1, n - 1):
        a[i, i - 1], a[i, i], a[i, i + 1] = hh[i - 1], 2 * (hh[i - 1] + hh[i]), hh[i]
        r[i] = 3 * ((ys[i + 1] - ys[i]) / hh[i] - (ys[i] - ys[i - 1]) / hh[i - 1])
    c = np.linalg.solve(a, r)
    x = np.arange(256, dtype=float)
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 2)
    dx = x - xs[i]
    b = (ys[i + 1] - ys[i]) / hh[i] - hh[i] * (2 * c[i] + c[i + 1]) / 3
    d = (c[i + 1] - c[i]) / (3 * hh[i])
    return np.clip(ys[i] + b * dx + c[i] * dx ** 2 + d * dx ** 3, 0, 255) / 255.0


def apply_adjustment(img, adj):
    kind, st = adj["kind"], adj["settings"]
    if kind == "levels":
        c0 = st["channels"][0]
        out = F.levels(img, c0["in_black"], c0["in_white"], c0["gamma"], c0["out_black"], c0["out_white"])
        for k, ch in enumerate(st["channels"][1:4]):
            if (ch["in_black"], ch["in_white"], ch["gamma"], ch["out_black"], ch["out_white"]) != (0, 255, 1.0, 0, 255):
                out[..., k] = F.levels(out[..., k], ch["in_black"], ch["in_white"], ch["gamma"],
                                       ch["out_black"], ch["out_white"])
        return out
    if kind == "curves":
        out = img
        for ch in st["channels"]:
            lut = _curve_lut(ch["points"]).astype(np.float32)
            if ch["channel"] == "RGB":
                out = np.interp(np.clip(out, 0, 1), np.linspace(0, 1, 256), lut).astype(np.float32)
            elif ch["channel"] in ("R", "G", "B"):
                k = "RGB".index(ch["channel"])
                out[..., k] = np.interp(np.clip(out[..., k], 0, 1), np.linspace(0, 1, 256), lut)
        return out
    if kind == "vibrance":
        return F.vibrance(img, st.get("vibrance", 0), st.get("Strt", 0))
    if kind == "hue/saturation":
        m = st["master"]
        return F.hue_saturation(img, m["hue"], m["saturation"], m["lightness"])
    if kind == "brightness/contrast":
        return F.brightness_contrast(img, st["brightness"], st["contrast"])
    if kind == "posterize":
        return F.posterize(img, st["levels"])
    if kind == "invert":
        return 1 - img
    return img


def composite(info, tree):
    h, w = info["h"], info["w"]
    layers = info["layers"]

    def layer_rgba(i):
        L = layers[i]
        px = psdread.rgba(info, L)
        full = np.zeros((h, w, 4), np.float32)
        if px is None:
            return full
        t, l = L.top, L.left
        y0, x0 = max(0, t), max(0, l)
        y1, x1 = min(h, L.bottom), min(w, L.right)
        if y1 > y0 and x1 > x0:
            full[y0:y1, x0:x1] = px[y0 - t:y1 - t, x0 - l:x1 - l].astype(np.float32) / 255
        return full

    def mask_of(i):
        L = layers[i]
        if not L.mask:
            return None
        t, l, bt, r, defc, flags = L.mask
        if flags & 2:
            return None
        m = np.full((h, w), defc / 255.0, np.float32)
        if -2 in L.pix and (bt - t) * (r - l) > 0:
            y0, x0, y1, x1 = max(0, t), max(0, l), min(h, bt), min(w, r)
            m[y0:y1, x0:x1] = L.pix[-2][y0 - t:y1 - t, x0 - l:x1 - l] / 255.0
        return m

    def render(children, base_rgb, base_a):
        rgb, a = base_rgb, base_a
        for node in children:
            if not node.get("visible", True):
                continue
            op = node.get("opacity_255", 255) / 255.0
            if node["type"] == "group":
                if node.get("blend_key", "pass") == "pass":
                    rgb, a = render(node["children"], rgb, a)
                    if op < 1:
                        pass                  # pass-through group opacity: rare, ignored
                else:
                    g_rgb, g_a = render(node["children"], np.zeros_like(rgb), np.zeros_like(a))
                    rgb, a2 = F.composite(rgb, a, g_rgb, g_a * op, MODES.get(node["blend_key"], "Nrml"))
                    a = a2 if a2 is not None else np.ones_like(a)
                continue
            i = node["index"]
            m = mask_of(i)
            fill = node.get("fill_opacity", 100) / 100.0
            if node["type"] == "adjustment":
                adj = apply_adjustment(np.clip(rgb, 0, 1), node["adjustment"])
                k = op * (m if m is not None else 1.0)
                k = k[..., None] if isinstance(k, np.ndarray) else k
                rgb = rgb + (adj - rgb) * k
                continue
            if node["type"] == "fill" and node.get("fill", {}).get("kind") == "solid color":
                c = node["fill"]["settings"].get("Clr", {})
                col = np.array([c.get("Rd", 0), c.get("Grn", 0), c.get("Bl", 0)], np.float32) / 255
                src = np.broadcast_to(col, rgb.shape).astype(np.float32)
                sa = np.ones(a.shape, np.float32)
            else:
                px = layer_rgba(i)
                src, sa = px[..., :3], px[..., 3:4]
            sa = sa * op * fill
            if m is not None:
                sa = sa * m[..., None]
            mode = MODES.get(node.get("blend_key", "norm"), "Nrml")
            rgb, a2 = F.composite(rgb, a, src, sa, mode)
            a = a2 if a2 is not None else np.ones_like(a)
        return rgb, a

    rgb, a = render(tree["children"], np.zeros((h, w, 3), np.float32), np.zeros((h, w, 1), np.float32))
    white = np.ones_like(rgb)
    return np.clip(rgb * a + white * (1 - a), 0, 1)


def scores(a, b, size=1000):
    """PSNR and SSIM (grey, short side ``size``) of two uint8 RGB pictures."""
    from skimage.metrics import structural_similarity
    k = size / min(a.shape[:2])
    dims = (int(a.shape[1] * k), int(a.shape[0] * k))
    x = cv2.resize(a, dims, interpolation=cv2.INTER_AREA).astype(np.float64)
    y = cv2.resize(b, dims, interpolation=cv2.INTER_AREA).astype(np.float64)
    mse = ((x - y) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
    gx = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gy = cv2.cvtColor(y.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ssim = structural_similarity(gx, gy, data_range=255)
    return {"psnr": round(float(psnr), 2), "ssim": round(float(ssim), 4),
            "mae": round(float(np.abs(x - y).mean()), 2)}


def summary(tree, depth=0, lines=None):
    lines = [] if lines is None else lines
    for n in tree["children"]:
        pad = "  " * depth
        vis = "" if n.get("visible", True) else "  [HIDDEN]"
        extra = ""
        if n.get("adjustment"):
            extra = f"  {n['adjustment']['kind']}: {json.dumps(n['adjustment']['settings'])[:160]}"
        if n.get("fill"):
            extra = f"  {n['fill']['kind']}"
        if n.get("smart_object"):
            fl = n["smart_object"].get("smart_filters") or []
            extra = "  smart filters: " + ", ".join(f"{f['name']}" for f in fl)
        fo = f", fill {n['fill_opacity']}%" if "fill_opacity" in n else ""
        lines.append(f"{pad}- [{n['type']}] {n['name']!r}  {n.get('blend_mode')} "
                     f"{n.get('opacity', 100)}%{fo}{vis}{extra}")
        if n["type"] == "group":
            summary(n, depth + 1, lines)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("psd")
    ap.add_argument("--out", default="refs")
    ap.add_argument("--target", default="refs/target.jpg")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    info = psdread.parse(a.psd, want_pixels=True)
    print(f"{a.psd}: {info['w']} x {info['h']}, {info['ch']} channels, {info['depth']} bit, "
          f"{'PSB' if info['ver'] == 2 else 'PSD'}, {len(info['layers'])} layer records")
    tree = build_tree(info, a.out, export=not a.no_export)
    emb = extract_embedded(info, os.path.join(a.out, "embedded"))
    tree["embedded_files"] = emb
    with open(os.path.join(a.out, "psd_tree.json"), "w") as fh:
        json.dump(tree, fh, indent=2, default=str)
    print("\nLayer tree, TOP first (bottom-to-top order is in psd_tree.json):")

    def rev(t):
        t = dict(t)
        if "children" in t:
            t["children"] = [rev(c) for c in reversed(t["children"])]
        return t
    for ln in summary(rev(tree)):
        print(ln)
    print("\nEmbedded files:")
    for e in emb:
        print("  ", e)

    comp = (composite(info, tree) * 255 + 0.5).astype(np.uint8)
    # templates often have a label group meant to be hidden before saving
    hide = [n for n in tree["children"] if re.search(r"hide", n["name"], re.I)]
    for n in hide:
        n["visible"] = False
    comp_clean = (composite(info, tree) * 255 + 0.5).astype(np.uint8) if hide else comp
    for n in hide:
        n["visible"] = True
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/psd_composite.png", comp[..., ::-1])
    merged = psdread.merged(info)
    print("\nOwn composite vs the PSD's flattened image:", scores(comp, merged))
    if hide:
        cv2.imwrite("outputs/psd_composite_clean.png", comp_clean[..., ::-1])
        print("Hidden for the clean composite:", [n["name"] for n in hide])
    if os.path.exists(a.target):
        t = cv2.imread(a.target)[..., ::-1]
        if t.shape[:2] == comp.shape[:2]:
            print("Own composite vs target:", scores(comp, t))
            print("PSD flattened image vs target:", scores(merged, t))
            if hide:
                print("Own composite with those groups hidden vs target:", scores(comp_clean, t))
        else:
            print("target size differs:", t.shape, comp.shape)


if __name__ == "__main__":
    main()
