"""Play Photoshop actions (.atn) inside OilStudio - no Photoshop needed.

    from actions import player
    result = player.run(atn_path, image_rgb_u8, resources=[...], focus=mask)
    result.image      # RGB uint8
    result.skipped    # steps that could not be done, with the reason

The player keeps a small Photoshop-like document: a tree of layers (pixel
layers, adjustment layers, fill layers, groups), each with opacity, fill,
blend mode, layer mask, "blend if" and a pattern overlay style; plus the
active selection, the work path, the foreground / background colours and
the current brush. Each recorded step changes that document the way
Photoshop would, and at the end the visible layers are composited.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import atn as A
from . import filters as F
from .resources import BrushPreset, PatternStore, read_abr

REFERENCE_SIDE = 3500          # the size most Envato actions are made for


# --------------------------------------------------------------------------
# small value helpers
# --------------------------------------------------------------------------
def num(v, default=0.0):
    if isinstance(v, A.UnitFloat):
        return v.value
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return default


def pct(v, default=1.0):
    """Percent values -> 0..1 ("Opct": 66% -> 0.66)."""
    if isinstance(v, A.UnitFloat):
        return v.value / 100.0
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) / 100.0
    return default


def enum(v, default=""):
    if isinstance(v, A.Enum):
        return v.value
    if isinstance(v, str):
        return v
    return default


def colour(o) -> np.ndarray | None:
    """Colour descriptor -> RGB 0..1."""
    if not isinstance(o, A.Obj):
        return None
    c = o.cls
    if c == "RGBC":
        return np.array([num(o.get("Rd  ")), num(o.get("Grn ")), num(o.get("Bl  "))], np.float32) / 255
    if c == "HSBC":
        h = num(o.get("H   ")) % 360
        s = num(o.get("Strt")) / 100
        b = num(o.get("Brgh")) / 100
        hsv = np.array([[[h, s, b]]], np.float32)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    if c == "Grsc":
        g = 1 - num(o.get("Gry ")) / 100
        return np.array([g, g, g], np.float32)
    if c == "CMYC":
        cc, m, y, k = (num(o.get(k)) / 100 for k in ("Cyn ", "Mgnt", "Ylw ", "Blck"))
        return np.array([(1 - cc) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)], np.float32)
    if c == "LbCl":
        lab = np.array([[[num(o.get("Lmnc")), num(o.get("A   ")), num(o.get("B   "))]]], np.float32)
        return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)[0, 0], 0, 1)
    return None


# --------------------------------------------------------------------------
# document model
# --------------------------------------------------------------------------
@dataclass(eq=False)
class Layer:
    name: str
    kind: str = "pixel"            # pixel | adjust | fill | group
    visible: bool = True
    opacity: float = 1.0
    fill: float = 1.0
    mode: str = "Nrml"
    rgb: np.ndarray | None = None  # uint8 (h, w, 3); None = empty layer
    alpha: np.ndarray | None = None  # uint8 (h, w); None = opaque (if rgb)
    mask: np.ndarray | None = None   # uint8 (h, w); 255 = shows
    mask_feather: float = 0.0
    mask_enabled: bool = True
    effects: A.Obj | None = None
    blend_if: list = field(default_factory=list)
    adjust: tuple | None = None    # (type, Obj)
    content: A.Obj | None = None   # fill layer description
    children: list = field(default_factory=list)
    parent: "Layer | None" = None
    background: bool = False
    label: str = ""

    def empty(self):
        if self.kind != "pixel":
            return False
        if self.rgb is None:
            return True
        return self.alpha is not None and not self.alpha.any()

    def __repr__(self):
        return f"<{self.kind} {self.name!r}{'' if self.visible else ' hidden'}>"


@dataclass
class Skip:
    index: int
    title: str
    reason: str


@dataclass
class Result:
    image: np.ndarray                  # RGB uint8
    skipped: list
    approximated: list
    steps: int
    seconds: float
    layers: list


class StepError(Exception):
    pass


class Doc:
    def __init__(self, img_rgb: np.ndarray, scale: float, patterns: PatternStore,
                 brushes: dict[str, BrushPreset], seed: int = 1):
        self.h, self.w = img_rgb.shape[:2]
        self.s = scale
        self.root = Layer("", "group")
        bg = Layer("Background", rgb=img_rgb.copy(), background=True)
        self.add(bg, self.root)
        self.targets: list[Layer] = [bg]
        self.channel = "rgb"
        self.selection: np.ndarray | None = None      # float32 0..1
        self.path: list[np.ndarray] | None = None     # list of (n, 2) float arrays
        self.fg = np.zeros(3, np.float32)
        self.bg = np.ones(3, np.float32)
        self.patterns = patterns
        self.brushes = brushes
        self.brush: BrushPreset | None = None
        self.brush_diameter: float | None = None
        self.style_clip: A.Obj | None = None
        self._mask_cache: dict = {}
        self.rng = np.random.default_rng(seed)

    # -- tree helpers ------------------------------------------------------
    @staticmethod
    def add(layer, parent, index=None):
        layer.parent = parent
        if index is None:
            parent.children.append(layer)
        else:
            parent.children.insert(index, layer)

    def walk(self, node=None):
        node = node or self.root
        for c in node.children:
            yield c
            if c.kind == "group":
                yield from self.walk(c)

    def top_down(self, node=None):
        node = node or self.root
        for c in reversed(node.children):
            yield c
            if c.kind == "group":
                yield from self.top_down(c)

    def find(self, name):
        for l in self.top_down():
            if l.name == name:
                return l
        low = name.strip().lower()
        for l in self.top_down():
            if l.name.strip().lower() == low:
                return l
        return None

    def flat(self):
        """Layers bottom -> top, groups listed after their content."""
        out = []

        def rec(node):
            for c in node.children:
                if c.kind == "group":
                    rec(c)
                out.append(c)
        rec(self.root)
        return out

    def place_above(self, new, ref):
        if ref is None or ref.parent is None:
            self.add(new, self.root)
            return
        p = ref.parent
        self.add(new, p, p.children.index(ref) + 1)

    def remove(self, layer):
        if layer.parent is not None and layer in layer.parent.children:
            layer.parent.children.remove(layer)
        layer.parent = None

    def topmost(self, layers):
        order = {id(l): i for i, l in enumerate(self.flat())}
        return max(layers, key=lambda l: order.get(id(l), -1))

    def stack_sorted(self, layers):
        order = {id(l): i for i, l in enumerate(self.flat())}
        return sorted(layers, key=lambda l: order.get(id(l), -1))

    def names(self):
        return {l.name for l in self.walk()}

    def next_name(self, base):
        n = 0
        rx = re.compile(r"^" + re.escape(base) + r" (\d+)$")
        for name in self.names():
            m = rx.match(name)
            if m:
                n = max(n, int(m.group(1)))
        return f"{base} {n + 1}"

    def copy_name(self, src):
        base = re.sub(r" copy( \d+)?$", "", src.name)
        if not src.background and re.match(r"^Layer \d+$", base):
            return self.next_name("Layer")
        n = 0
        rx = re.compile(r"^" + re.escape(base) + r" copy(?: (\d+))?$")
        for name in self.names():
            m = rx.match(name)
            if m:
                n = max(n, int(m.group(1) or 1))
        return f"{base} copy" if n == 0 else f"{base} copy {n + 1}"

    @property
    def target(self):
        return self.targets[0] if self.targets else None

    # -- pixels --------------------------------------------------------------
    def rgba(self, l):
        """float32 rgb (h, w, 3) and alpha (h, w, 1) of a pixel layer."""
        if l.rgb is None:
            return (np.zeros((self.h, self.w, 3), np.float32),
                    np.zeros((self.h, self.w, 1), np.float32))
        rgb = np.multiply(l.rgb, np.float32(1 / 255), dtype=np.float32)
        a = (np.ones((self.h, self.w, 1), np.float32) if l.alpha is None
             else np.multiply(l.alpha, np.float32(1 / 255), dtype=np.float32)[..., None])
        return rgb, a

    @staticmethod
    def u8(x):
        x = np.asarray(x, np.float32)
        return cv2.convertScaleAbs(np.maximum(x, 0), alpha=255.0)

    def set_rgba(self, l, rgb, a=None):
        l.rgb = self.u8(rgb)
        if a is None or (np.isscalar(a)):
            l.alpha = None
        else:
            a = a[..., 0] if a.ndim == 3 else a
            l.alpha = None if a.min() >= 0.999 else self.u8(a)

    def mask_f(self, l):
        if l.mask is None or not l.mask_enabled:
            return None
        hit = self._mask_cache.get(id(l))
        if hit is not None and hit[0] is l.mask and hit[1] == l.mask_feather:
            return hit[2]
        m = np.multiply(l.mask, np.float32(1 / 255), dtype=np.float32)
        if l.mask_feather > 0.05:
            m = F.gauss(m, l.mask_feather * self.s)
        m = m[..., None]
        self._mask_cache[id(l)] = (l.mask, l.mask_feather, m)
        return m

    # -- layer rendering -----------------------------------------------------
    def pattern_image(self, pobj, scale):
        name = pobj.get("Nm  ") if isinstance(pobj, A.Obj) else ""
        ident = pobj.get("Idnt") if isinstance(pobj, A.Obj) else ""
        pat = self.patterns.get(name or "", ident or "")
        if pat is None:
            raise StepError(f"pattern {name!r} is not in the action's .pat file")
        k = max(scale * self.s, 1e-3)
        ph, pw = pat.shape[:2]
        th, tw = max(1, int(round(ph * k))), max(1, int(round(pw * k)))
        if th >= self.h and tw >= self.w:            # one tile covers the canvas
            sh = min(ph, int(math.ceil(self.h / k)) + 1)
            sw = min(pw, int(math.ceil(self.w / k)) + 1)
            part = pat[:sh, :sw]
            out = cv2.resize(part, (max(1, int(round(sw * k))), max(1, int(round(sh * k)))),
                             interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            out = out[:self.h, :self.w]
            if out.shape[0] < self.h or out.shape[1] < self.w:
                out = np.pad(out, ((0, self.h - out.shape[0]), (0, self.w - out.shape[1]), (0, 0)), mode="wrap")
        else:
            tile = cv2.resize(pat, (tw, th), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            ny, nx = -(-self.h // th), -(-self.w // tw)
            out = np.tile(tile, (ny, nx, 1))[:self.h, :self.w]
        out = out.astype(np.float32) / 255
        return out[..., :3], out[..., 3:4]

    def render(self, l, approx=None):
        """A pixel or fill layer as (rgb, alpha) with fill, style, mask, opacity."""
        if l.kind == "fill":
            rgb = self.fill_content(l)
            a = np.full((self.h, self.w, 1), l.fill, np.float32)
        else:
            rgb, a = self.rgba(l)
            a = a * l.fill
        base_a = self.rgba(l)[1] if l.kind == "pixel" else np.ones((self.h, self.w, 1), np.float32)
        fx = l.effects
        if isinstance(fx, A.Obj) and fx.get("masterFXSwitch", True) is not False:
            po = fx.get("patternFill")
            if isinstance(po, A.Obj) and po.get("enab", True):
                prgb, pa = self.pattern_image(po.get("Ptrn"), pct(po.get("Scl "), 1.0))
                pa = pa * pct(po.get("Opct"), 1.0) * base_a
                mode = enum(po.get("Md  "), "Nrml")
                rgb, a = F.composite(rgb, a, prgb, pa, mode)
            so = fx.get("SoFi")
            if isinstance(so, A.Obj) and so.get("enab", True):
                c = colour(so.get("Clr ")) if so.get("Clr ") is not None else np.zeros(3, np.float32)
                sa = base_a * pct(so.get("Opct"), 1.0)
                rgb, a = F.composite(rgb, a, np.broadcast_to(c, rgb.shape).astype(np.float32), sa,
                                     enum(so.get("Md  "), "Nrml"))
            go = fx.get("GrFl")
            if isinstance(go, A.Obj) and go.get("enab", True):
                g = self.gradient(go)
                ga = base_a * pct(go.get("Opct"), 1.0)
                rgb, a = F.composite(rgb, a, g, ga, enum(go.get("Md  "), "Nrml"))
            for k in ("DrSh", "IrSh", "OrGl", "IrGl", "ebbl", "FrFX", "ChFX"):
                v = fx.get(k)
                if isinstance(v, A.Obj) and v.get("enab", True) and approx is not None:
                    approx.add(f"layer style {k.strip()} (not drawn)")
        m = self.mask_f(l)
        if m is not None:
            a = a * m
        return rgb, a * l.opacity

    def gradient(self, o, angle=None):
        g = o.get("Grad") if o.cls != "Grdn" else o
        stops = []
        if isinstance(g, A.Obj):
            for st in g.get("Clrs") or []:
                c = colour(st.get("Clr "))
                t = enum(st.get("Type"), "UsrS")
                if t == "FrgC":
                    c = self.fg
                elif t == "BckC":
                    c = self.bg
                if c is None:
                    c = np.zeros(3, np.float32)
                stops.append((num(st.get("Lctn")) / 4096.0, c))
        if not stops:
            stops = [(0.0, self.fg), (1.0, self.bg)]
        stops.sort(key=lambda s: s[0])
        ang = math.radians(num(o.get("Angl"), 90.0) if angle is None else angle)
        sc = pct(o.get("Scl "), 1.0) or 1.0
        kind = enum(o.get("Type"), "Lnr ")
        y, x = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        cx, cy = self.w / 2, self.h / 2
        if kind in ("Rdl ", "radial"):
            r = math.hypot(self.w, self.h) / 2 * sc
            t = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max(r, 1)
        else:
            dx, dy = math.cos(ang), -math.sin(ang)
            half = (abs(self.w * dx) + abs(self.h * dy)) / 2 * sc
            t = ((x - cx) * dx + (y - cy) * dy) / max(half, 1) * 0.5 + 0.5
        if o.get("Rvrs"):
            t = 1 - t
        t = np.clip(t, 0, 1)
        locs = np.array([p for p, _ in stops], np.float32)
        cols = np.array([c for _, c in stops], np.float32)
        out = np.stack([np.interp(t, locs, cols[:, i]) for i in range(3)], axis=-1)
        return out.astype(np.float32)

    def fill_content(self, l):
        o = l.content
        t = o.get("Type") if isinstance(o, A.Obj) else None
        if isinstance(t, A.Obj) and t.cls == "gradientLayer":
            return self.gradient(t)
        if isinstance(t, A.Obj) and t.cls == "solidColorLayer":
            c = colour(t.get("Clr ")) if t.get("Clr ") is not None else self.fg
            return np.broadcast_to(c, (self.h, self.w, 3)).astype(np.float32).copy()
        if isinstance(t, A.Obj) and t.cls == "patternLayer":
            rgb, _ = self.pattern_image(t.get("Ptrn"), pct(t.get("Scl "), 1.0))
            return rgb
        return np.broadcast_to(self.fg, (self.h, self.w, 3)).astype(np.float32).copy()

    def apply_adjust(self, rgb, adj):
        kind, o = adj
        if kind == "Lvls":
            return levels_obj(rgb, o)
        if kind == "vibrance":
            return F.vibrance(rgb, num(o.get("vibrance")), num(o.get("Strt")))
        if kind == "BrgC":
            return F.brightness_contrast(rgb, num(o.get("Brgh")), num(o.get("Cntr")),
                                         bool(o.get("useLegacy", False)))
        if kind == "HStr":
            return hue_sat_obj(rgb, o)
        if kind == "Crvs":
            return curves_obj(rgb, o)
        if kind == "Invr":
            return F.invert(rgb)
        if kind == "Thrs":
            return F.threshold(rgb, num(o.get("Lvl "), 128))
        if kind == "Pstr":
            return F.posterize(rgb, num(o.get("Lvls"), 4))
        if kind == "exposure":
            e = 2.0 ** num(o.get("Exps"))
            off = num(o.get("Ofst"))
            g = num(o.get("gammaCorrection"), 1.0) or 1.0
            return np.clip(np.power(np.clip(rgb * e + off, 0, 1), 1 / g), 0, 1)
        if kind == "GdMp":
            l = F.lum(rgb)
            g = self.gradient(o if o.get("Grad") is not None else A.Obj("x", {"Grad": o}), angle=0)
            ramp = g[self.h // 2, :, :]
            idx = np.clip((l * (ramp.shape[0] - 1)).astype(np.int32), 0, ramp.shape[0] - 1)
            return ramp[idx]
        raise StepError(f"adjustment {kind!r} is not supported yet")

    def blend_if_alpha(self, l, src_rgb, dst_rgb):
        f = None
        for b in l.blend_if:
            if not isinstance(b, A.Obj):
                continue
            ch = b.get("Chnl")
            which = ch.first().value.value if isinstance(ch, A.Ref) and ch.items and isinstance(ch.first().value, A.Enum) else "Gry "
            def chan(img):
                if which == "Rd  ":
                    return img[..., 0]
                if which == "Grn ":
                    return img[..., 1]
                if which == "Bl  ":
                    return img[..., 2]
                return F.lum(img)
            v = chan(src_rgb) * 255
            d = chan(dst_rgb) * 255
            k = ramp(v, num(b.get("SrcB"), 0), num(b.get("Srcl"), 0), num(b.get("SrcW"), 255), num(b.get("Srcm"), 255))
            k = k * ramp(d, num(b.get("DstB"), 0), num(b.get("Dstl"), 0), num(b.get("DstW"), 255), num(b.get("Dstt"), 255))
            f = k if f is None else f * k
        return None if f is None else f[..., None]

    def composite_into(self, node, rgb, a, approx=None, only=None):
        for c in node.children:
            if not c.visible:
                continue
            if only is not None and c not in only and c.kind != "group":
                continue
            if c.kind == "group":
                if c.opacity >= 0.999 and c.mask is None and c.mode in ("passThrough", "Nrml"):
                    rgb, a = self.composite_into(c, rgb, a, approx, only)
                else:
                    g_rgb, g_a = self.composite_into(c, np.zeros_like(rgb),
                                                     np.zeros((self.h, self.w, 1), np.float32), approx, only)
                    if g_a is None:
                        g_a = np.ones((self.h, self.w, 1), np.float32)
                    m = self.mask_f(c)
                    if m is not None:
                        g_a = g_a * m
                    rgb, a = F.composite(rgb, a, g_rgb, g_a * c.opacity,
                                         "Nrml" if c.mode == "passThrough" else c.mode)
                continue
            if c.kind == "adjust":
                try:
                    adj = self.apply_adjust(rgb, c.adjust)
                except StepError:
                    if approx is not None:
                        approx.add(f"adjustment layer {c.name!r} (type not supported)")
                    continue
                k = np.float32(c.opacity)
                m = self.mask_f(c)
                if m is not None:
                    k = k * m
                if c.mode not in ("Nrml", "normal"):
                    adj = F.blend(c.mode, rgb, adj)
                rgb = rgb * (1 - k) + adj * k
                continue
            if c.kind == "pixel" and c.rgb is None and c.effects is None:
                continue
            if (c.kind == "pixel" and c.alpha is None and c.fill >= 0.999 and c.opacity >= 0.999
                    and c.effects is None and c.mode in ("Nrml", "normal") and not c.blend_if
                    and (c.mask is None or not c.mask_enabled)):
                rgb = np.multiply(c.rgb, np.float32(1 / 255), dtype=np.float32)   # covers all below
                a = None
                continue
            s_rgb, s_a = self.render(c, approx)
            if c.blend_if:
                k = self.blend_if_alpha(c, s_rgb, rgb)
                if k is not None:
                    s_a = s_a * k
            rgb, a = F.composite(rgb, a, s_rgb, s_a, c.mode)
        return rgb, a

    def flatten(self, approx=None):
        rgb = np.zeros((self.h, self.w, 3), np.float32)
        a = np.zeros((self.h, self.w, 1), np.float32)
        return self.composite_into(self.root, rgb, a, approx)


def ramp(v, b0, b1, w0, w1):
    """Blend-if factor: 0 below b0, fades in to b1, fades out from w0 to w1."""
    lo = np.clip((v - b0) / max(b1 - b0, 1e-3), 0, 1) if b1 > b0 else (v >= b0).astype(np.float32)
    hi = np.clip((w1 - v) / max(w1 - w0, 1e-3), 0, 1) if w1 > w0 else (v <= w1).astype(np.float32)
    return lo * hi


def _chan_index(ref):
    if isinstance(ref, A.Ref) and ref.items:
        v = ref.first().value
        e = v.value if isinstance(v, A.Enum) else ""
        return {"Rd  ": 0, "Grn ": 1, "Bl  ": 2}.get(e, -1)
    return -1


def levels_obj(rgb, o):
    out = rgb.copy()
    for adj in o.get("Adjs") or []:
        if not isinstance(adj, A.Obj):
            continue
        inp = adj.get("Inpt") or [0, 255]
        outp = adj.get("Otpt") or [0, 255]
        g = num(adj.get("Gmm "), 1.0)
        ci = _chan_index(adj.get("Chnl"))
        args = (num(inp[0]), num(inp[1]), g, num(outp[0]), num(outp[1]))
        if ci < 0:
            out = F.levels(out, *args)
        else:
            out[..., ci] = F.levels(out[..., ci], *args)
    return np.clip(out, 0, 1)


def hue_sat_obj(rgb, o):
    out = rgb
    for adj in o.get("Adjs") or []:
        if isinstance(adj, A.Obj) and adj.get("LclR") is None:
            out = F.hue_saturation(out, num(adj.get("H   ")), num(adj.get("Strt")), num(adj.get("Lght")))
    if o.get("Clrz"):
        l = F.lum(out)
        h = num(o.get("H   ", 0)) if o.get("H   ") is not None else 0
        hsv = np.dstack([np.full_like(l, h % 360), np.full_like(l, 0.25), l])
        out = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
    return out


def curves_obj(rgb, o):
    out = rgb.copy()
    for adj in o.get("Adjs") or []:
        if not isinstance(adj, A.Obj):
            continue
        pts = adj.get("Crv ")
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        xs = np.array([num(p.get("Hrzn")) for p in pts], np.float32) / 255
        ys = np.array([num(p.get("Vrtc")) for p in pts], np.float32) / 255
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        grid = np.linspace(0, 1, 256, dtype=np.float32)
        lut = np.interp(grid, xs, ys)
        if len(xs) > 2:                     # smooth like Photoshop's spline
            lut = F.gauss(lut.reshape(1, -1).astype(np.float32), 3.0).ravel()
            lut = np.interp(grid, grid, lut)
        ci = _chan_index(adj.get("Chnl"))
        idx = np.clip(out * 255, 0, 255).astype(np.int32)
        if ci < 0:
            out = lut[idx].astype(np.float32)
        else:
            out[..., ci] = lut[idx[..., ci]]
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------
# brush strokes along the work path
# --------------------------------------------------------------------------
def _dab_tip(tip: np.ndarray, size: float, angle: float, roundness: float, flip_x: bool, flip_y: bool):
    t = tip
    if flip_x:
        t = t[:, ::-1]
    if flip_y:
        t = t[::-1]
    th, tw = t.shape
    k = size / max(th, tw)
    nw, nh = max(1, int(round(tw * k))), max(1, int(round(th * k * roundness)))
    t = cv2.resize(t, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    if abs(angle) > 0.5:
        d = int(math.ceil(math.hypot(nw, nh))) + 2
        m = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
        m[0, 2] += (d - nw) / 2
        m[1, 2] += (d - nh) / 2
        t = cv2.warpAffine(t, m, (d, d), flags=cv2.INTER_LINEAR, borderValue=0)
    return t.astype(np.float32) / 255


def _resample(path: np.ndarray, closed: bool, step: float):
    pts = np.vstack([path, path[:1]]) if closed else path
    seg = np.diff(pts, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    total = ln.sum()
    if total < 1e-3:
        return pts[:1], np.zeros(1)
    cum = np.concatenate([[0], np.cumsum(ln)])
    d = np.arange(0, total, max(step, 1.0))
    x = np.interp(d, cum, pts[:, 0])
    y = np.interp(d, cum, pts[:, 1])
    i = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    ang = np.degrees(np.arctan2(-seg[i, 1], seg[i, 0]))
    return np.stack([x, y], 1), ang


def paint_path(doc: Doc, target: np.ndarray, colour_value, brush: BrushPreset, diameter: float,
               rng: np.random.Generator, channels: int):
    """Stamp the brush along every sub-path. ``target`` is float32 (h, w, channels + 1):
    colour channels followed by alpha (for a mask: value channel, then a dummy)."""
    if brush is None or brush.tip is None:
        tip = np.zeros((65, 65), np.uint8)
        cv2.circle(tip, (32, 32), 31, 255, -1)
        tip = cv2.GaussianBlur(tip, (0, 0), 4)
        brush = BrushPreset("round", tip, diameter, 0.25)
    d0 = max(1.0, diameter)
    step = max(1.0, brush.spacing * d0)
    base = brush.tip
    if max(base.shape) > d0 * 1.05:              # shrink the big sampled tip once
        k0 = d0 / max(base.shape)
        base = cv2.resize(base, (max(1, int(round(base.shape[1] * k0))), max(1, int(round(base.shape[0] * k0)))),
                          interpolation=cv2.INTER_AREA)
    h, w = target.shape[:2]
    col = np.asarray(colour_value, np.float32).reshape(-1)[:channels]
    count = 0
    for sub in doc.path or []:
        pts, angs = _resample(sub, True, step)
        for (x, y), a_dir in zip(pts, angs):
            n = brush.count
            if brush.count_jitter:
                n = max(1, int(round(n * (1 - brush.count_jitter * rng.random()))))
            for _ in range(n):
                size = d0
                if brush.size_jitter:
                    size *= 1 - brush.size_jitter * rng.random()
                size = max(size, d0 * brush.min_size, 1.0)
                ang = brush.angle
                if brush.angle_dir:
                    ang += a_dir
                if brush.angle_jitter:
                    ang += (rng.random() - 0.5) * 360 * brush.angle_jitter
                rnd = brush.roundness
                if brush.round_jitter:
                    rnd *= 1 - brush.round_jitter * rng.random() * 0.75
                px, py = x, y
                if brush.scatter:
                    off = (rng.random() * 2 - 1) * brush.scatter * d0
                    ra = math.radians(a_dir + 90)
                    px += math.cos(ra) * off
                    py -= math.sin(ra) * off
                    if brush.scatter_both:
                        off2 = (rng.random() * 2 - 1) * brush.scatter * d0
                        ra2 = math.radians(a_dir)
                        px += math.cos(ra2) * off2
                        py -= math.sin(ra2) * off2
                op = 1.0
                if brush.opacity_jitter:
                    op *= 1 - brush.opacity_jitter * rng.random()
                if brush.flow_jitter:
                    op *= 1 - brush.flow_jitter * rng.random()
                t = _dab_tip(base, size, ang, rnd, brush.flip_x, brush.flip_y) * op
                th, tw = t.shape
                x0, y0 = int(round(px - tw / 2)), int(round(py - th / 2))
                xa, ya = max(0, x0), max(0, y0)
                xb, yb = min(w, x0 + tw), min(h, y0 + th)
                if xa >= xb or ya >= yb:
                    continue
                k = t[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
                reg = target[ya:yb, xa:xb]
                c = col
                if brush.fg_bg_jitter:
                    c = col + (np.asarray(doc.bg[:channels]) - col) * rng.random() * brush.fg_bg_jitter
                if channels == 1:                       # a mask: paint the grey value
                    reg[..., 0:1] = reg[..., 0:1] * (1 - k) + c[0] * k
                else:                                   # a pixel layer: paint over
                    ra = reg[..., 3:4]
                    na = k + ra * (1 - k)
                    reg[..., :3] = (c * k + reg[..., :3] * ra * (1 - k)) / np.maximum(na, 1e-6)
                    reg[..., 3:4] = na
                count += 1
    return count


# --------------------------------------------------------------------------
# the player
# --------------------------------------------------------------------------
ALIASES = {
    "slct": "select", "setd": "set", "Hd  ": "hide", "Shw ": "show", "Mk  ": "make",
    "Rset": "reset", "CpTL": "copyToLayer", "Lvls": "levels", "Strk": "stroke",
    "Dlt ": "delete", "move": "move", "Invr": "invert", "Mrg2": "mergeLayersNew",
    "MrgL": "mergeLayersNew", "Exch": "exchange", "Fl  ": "fill", "Dslc": "deselect", "Dstt": "desaturate",
    "MrgV": "mergeVisible", "HghP": "highPass", "Prge": "purge", "Trnf": "transform",
    "CpFX": "copyEffects", "PaFX": "pasteEffects", "BrgC": "brightnessEvent",
    "GsnB": "gaussianBlur", "UnsM": "unsharpMask", "AdNs": "addNoise", "Mdn ": "median",
    "HStr": "hueSaturation", "Crvs": "curves", "Pstr": "posterization", "Thrs": "thresholdClassEvent",
    "FltI": "flattenImage", "Dplc": "duplicate", "Embs": "emboss", "Shrp": "sharpen",
    "ShrM": "sharpenMore", "Blr ": "blur", "BlrM": "blurMore", "MtnB": "motionBlur",
    "CnvM": "convertMode", "ImgS": "imageSize", "CnvS": "canvasSize", "Clse": "close",
    "save": "save", "Opn ": "open", "rasterizeLayer": "rasterizeLayer", "RstL": "rasterizeLayer",
    "Exps": "exposure", "Vbrn": "vibrance",
}

NO_OPS = {
    "purge", "modalStateChanged", "save", "close", "open", "rasterizeLayer",
    "convertMode", "selectNoLayers", "historyStateChanged", "layersFiltered",
    "toolModalStateChanged", "invokeCommand", "newPlacedLayer", "Stop",
}

ADJ_NAMES = {
    "Lvls": "Levels", "vibrance": "Vibrance", "BrgC": "Brightness/Contrast",
    "HStr": "Hue/Saturation", "Crvs": "Curves", "Invr": "Invert", "Thrs": "Threshold",
    "Pstr": "Posterize", "exposure": "Exposure", "GdMp": "Gradient Map",
    "ChnM": "Channel Mixer", "BlnW": "Black & White", "PhtF": "Photo Filter",
    "ClrB": "Color Balance", "selectiveColor": "Selective Color", "colorLookup": "Color Lookup",
}

FILL_NAMES = {"gradientLayer": "Gradient Fill", "solidColorLayer": "Color Fill",
              "patternLayer": "Pattern Fill"}


class Player:
    def __init__(self, doc: Doc, progress=None):
        self.d = doc
        self.skipped: list[Skip] = []
        self.approx: set[str] = set()
        self.progress = progress
        self.i = 0

    # -- reference helpers ------------------------------------------------
    def layers_of(self, ref, required=True) -> list[Layer]:
        refs = ref if isinstance(ref, list) else [ref]
        out = []
        for r in refs:
            if not isinstance(r, A.Ref) or not r.items:
                continue
            it = next((x for x in r.items if x.cls in ("Lyr ", "AdjL", "contentLayer", "layerSection")), r.items[0])
            if it.form == "name":
                l = self.d.find(it.value)
                if l is None:
                    if required:
                        raise StepError(f"layer {it.value!r} not found")
                    continue
                out.append(l)
            elif it.form == "Enmr":
                v = it.value.value if isinstance(it.value, A.Enum) else ""
                if v in ("Trgt", "Ordn"):
                    out += [l for l in self.d.targets if l not in out]
                elif v == "Bckw":
                    l = self._neighbour(-1)
                    if l is not None:
                        out.append(l)
                elif v == "Frwr":
                    l = self._neighbour(+1)
                    if l is not None:
                        out.append(l)
                else:
                    out += list(self.d.targets)
            elif it.form in ("indx", "Idnt"):
                flat = [l for l in self.d.flat()]
                idx = int(it.value)
                if it.form == "indx" and 0 <= idx < len(flat):
                    out.append(flat[idx])
                elif self.d.targets:
                    out += self.d.targets
        if required and not out:
            raise StepError("no layer to work on")
        return out

    def _neighbour(self, step):
        t = self.d.target
        if t is None or t.parent is None:
            return None
        sib = t.parent.children
        i = sib.index(t) + step
        return sib[i] if 0 <= i < len(sib) else None

    # -- running ----------------------------------------------------------
    def play(self, action: A.Action):
        n = len(action.steps)
        for i, st in enumerate(action.steps, 1):
            self.i = i
            if self.progress:
                self.progress(i, n, st.title)
            if not st.enabled:
                continue
            ev = ALIASES.get(st.event, st.event)
            if ev in NO_OPS:
                continue
            fn = getattr(self, "do_" + re.sub(r"\W", "_", ev), None)
            if fn is None:
                self.skipped.append(Skip(i, st.title, f"'{st.title}' is not supported yet"))
                continue
            try:
                fn(st.desc or A.Obj("", {}), st)
            except StepError as exc:
                self.skipped.append(Skip(i, st.title, str(exc)))
            except Exception as exc:                      # noqa: BLE001 - keep playing
                self.skipped.append(Skip(i, st.title, f"error: {exc}"))

    # -- channel aware pixel edits ----------------------------------------
    def edit_pixels(self, fn, label=None):
        """Run fn(rgb) on the active layer, or on its mask when the mask is active."""
        d = self.d
        t = d.target
        if t is None:
            raise StepError("no layer selected")
        if d.channel == "mask" and t.mask is not None:
            m = t.mask.astype(np.float32) / 255
            out = fn(np.repeat(m[..., None], 3, axis=2))
            t.mask = d.u8(F.lum(out) if out.ndim == 3 else out)
            return
        if t.kind != "pixel":
            raise StepError(f"'{t.name}' is not a pixel layer")
        if t.rgb is None:
            return
        rgb, a = d.rgba(t)
        out = np.clip(fn(rgb), 0, 1)
        if d.selection is not None:
            k = d.selection[..., None]
            out = rgb * (1 - k) + out * k
        t.rgb = d.u8(out)
        if label:
            self.approx.add(label)

    # -- commands -----------------------------------------------------------
    def do_select(self, o, st):
        ref = o.get("null")
        if not isinstance(ref, A.Ref) or not ref.items:
            return
        it = ref.items[0]
        d = self.d
        if it.cls == "Chnl":
            v = it.value.value if isinstance(it.value, A.Enum) else it.value
            lyr = [x for x in ref.items[1:] if x.cls == "Lyr "]
            if lyr:
                d.targets = self.layers_of(A.Ref(lyr))
            if v == "Msk ":
                if d.target is None or d.target.mask is None:
                    raise StepError("the layer has no mask")
                d.channel = "mask"
            elif v in ("RGB ", "Cmps", "CMYK", "Lab ", "Gry "):
                d.channel = "rgb"
            elif v == "Trgt":
                pass
            else:
                raise StepError(f"selecting the {v!r} channel is not supported")
            return
        if it.cls == "Brsh":
            name = it.value if it.form == "name" else ""
            b = d.brushes.get(name)
            if b is None:
                low = name.strip().lower()
                b = next((v for k, v in d.brushes.items() if k.strip().lower() == low), None)
            if b is None:
                raise StepError(f"brush {name!r} is not in the action's .abr file")
            d.brush = b
            d.brush_diameter = b.diameter
            return
        if it.cls in ("PbTl", "Path", "HstS", "SnpS") or it.cls.endswith("Tool") or it.cls in (
                "ErTl", "CrTl", "BrTl", "GrTl", "PcTl", "SmTl", "moveTool", "MvTl"):
            return
        if it.cls in ("Lyr ", "AdjL", "contentLayer", "layerSection"):
            layers = self.layers_of(ref)
            mod = enum(o.get("selectionModifier"), "")
            if mod == "addToSelection":
                d.targets = d.targets + [l for l in layers if l not in d.targets]
            elif mod == "addToSelectionContinuous":
                anchor = d.target
                l = layers[0]
                if anchor is not None and anchor.parent is l.parent:
                    sib = l.parent.children
                    a, b = sorted((sib.index(anchor), sib.index(l)))
                    rng = sib[a:b + 1]
                    d.targets = [anchor] + [x for x in rng if x is not anchor]
                else:
                    d.targets = d.targets + [l]
            elif mod == "removeFromSelection":
                d.targets = [x for x in d.targets if x not in layers] or d.targets
            else:
                d.targets = layers
            d.channel = "rgb"
            return
        if it.cls == "Dcmn":
            return
        raise StepError(f"selecting {it.cls.strip()} is not supported")

    def do_show(self, o, st):
        for l in self.layers_of(o.get("null")):
            l.visible = True

    def do_hide(self, o, st):
        for l in self.layers_of(o.get("null")):
            l.visible = False

    def do_reset(self, o, st):
        if self.d.channel == "mask":
            self.d.fg, self.d.bg = np.ones(3, np.float32), np.zeros(3, np.float32)
        else:
            self.d.fg, self.d.bg = np.zeros(3, np.float32), np.ones(3, np.float32)

    def do_exchange(self, o, st):
        self.d.fg, self.d.bg = self.d.bg, self.d.fg

    def do_set(self, o, st):
        d = self.d
        ref = o.get("null")
        to = o.get("T   ")
        if not isinstance(ref, A.Ref) or not ref.items:
            raise StepError("nothing to set")
        it = ref.items[0]
        if it.cls == "Chnl" and it.form == "prop" and it.value == "fsel":
            d.selection = self.selection_from(to)
            return
        if it.cls == "Clr ":
            c = colour(to)
            if c is None:
                raise StepError("unknown colour")
            if it.value == "FrgC":
                d.fg = c
            elif it.value == "BckC":
                d.bg = c
            return
        if it.cls == "Brsh":
            if isinstance(to, A.Obj) and to.get("masterDiameter") is not None:
                d.brush_diameter = num(to.get("masterDiameter"))
            return
        if it.cls == "Prpr" and it.value == "Lefx":
            for l in self.layers_of(A.Ref(ref.items[1:]) if len(ref.items) > 1 else A.Ref([A.RefItem("Enmr", "Lyr ", A.Enum("Ordn", "Trgt"))])):
                l.effects = to if isinstance(to, A.Obj) else None
            return
        if it.cls == "AdjL":
            for l in self.layers_of(ref):
                if l.kind != "adjust" or not isinstance(to, A.Obj):
                    raise StepError(f"'{l.name}' is not an adjustment layer")
                kind, old = l.adjust
                items = dict(old.items) if old is not None else {}
                items.update(to.items)
                l.adjust = (kind, A.Obj(to.cls, items))
            return
        if it.cls == "contentLayer":
            for l in self.layers_of(ref):
                if l.kind == "fill" and isinstance(to, A.Obj):
                    t = l.content.get("Type") if l.content else None
                    if isinstance(t, A.Obj):
                        merged = dict(t.items)
                        merged.update(to.items)
                        l.content = A.Obj("contentLayer", {"Type": A.Obj(t.cls, merged)})
            return
        if it.cls in ("Lyr ", "layerSection"):
            if not isinstance(to, A.Obj):
                raise StepError("unknown layer setting")
            for l in self.layers_of(ref):
                self.set_layer(l, to)
            return
        if it.cls == "Dcmn":
            return
        raise StepError(f"setting {it.cls.strip()} is not supported")

    def set_layer(self, l, to):
        for k, v in to.items.items():
            if k == "Nm  ":
                l.name = str(v)
            elif k == "Opct":
                l.opacity = pct(v)
            elif k == "fillOpacity":
                l.fill = pct(v)
            elif k == "Md  ":
                m = enum(v, "Nrml")
                if m not in F.SUPPORTED_MODES:
                    self.approx.add(f"blend mode {m}")
                l.mode = m
            elif k == "Lefx":
                if isinstance(v, A.Obj):
                    keys = [x for x in v.items if x not in ("Scl ", "masterFXSwitch")]
                    if keys or l.effects is None:
                        l.effects = v if keys else l.effects
            elif k == "Blnd":
                l.blend_if = v if isinstance(v, list) else []
            elif k == "Clr ":
                l.label = enum(v)
            elif k == "userMaskFeather":
                l.mask_feather = num(v)
            elif k == "userMaskEnabled":
                l.mask_enabled = bool(v)
            elif k == "Vsbl":
                l.visible = bool(v)
            elif k in ("layerLocking", "Lckd", "protectAll", "srgbBlend", "layerFXVisible",
                       "Blnd ", "blendClipped", "blendInterior", "knockout", "layerMaskAsGlobalMask",
                       "vectorMaskAsGlobalMask", "transparencyShapesLayer", "Grup"):
                continue
            else:
                self.approx.add(f"layer setting {k.strip()}")

    def selection_from(self, to):
        d = self.d
        if isinstance(to, A.Enum):
            if to.value == "Al  ":
                return np.ones((d.h, d.w), np.float32)
            if to.value == "None":
                return None
        if isinstance(to, A.Ref) and to.items:
            it = to.items[0]
            v = it.value.value if isinstance(it.value, A.Enum) else it.value
            lyr = [x for x in to.items[1:] if x.cls == "Lyr "]
            layer = self.layers_of(A.Ref(lyr))[0] if lyr else d.target
            if it.cls == "Chnl" and v == "Trsp":
                if layer is None or layer.kind != "pixel" or layer.rgb is None:
                    return np.zeros((d.h, d.w), np.float32)
                return d.rgba(layer)[1][..., 0].copy()
            if it.cls == "Chnl" and v == "Msk ":
                if layer is None or layer.mask is None:
                    raise StepError("the layer has no mask")
                return layer.mask.astype(np.float32) / 255
            if it.cls == "Chnl" and it.form == "name":
                raise StepError(f"loading the channel {it.value!r} as a selection is not supported")
        raise StepError("this kind of selection is not supported")

    def do_deselect(self, o, st):
        ref = o.get("null")
        if isinstance(ref, A.Ref) and ref.items and ref.items[0].cls == "Path":
            return                                          # the path just stops being shown
        self.d.selection = None

    def do_make(self, o, st):
        d = self.d
        new = o.get("Nw  ")
        if isinstance(new, tuple) and new[1] == "Chnl":
            return self.make_mask(o)
        if isinstance(new, A.Obj) and new.cls == "Chnl":
            raise StepError("making new alpha channels is not supported")
        ref = o.get("null")
        if not isinstance(ref, A.Ref) or not ref.items:
            raise StepError("nothing to make")
        cls = ref.items[0].cls
        if cls == "Lyr ":
            l = Layer(d.next_name("Layer"))
            u = o.get("Usng")
            if isinstance(u, A.Obj):
                self.set_layer(l, u)
            d.place_above(l, d.target)
            d.targets = [l]
            d.channel = "rgb"
            return
        if cls == "AdjL":
            u = o.get("Usng")
            t = u.get("Type") if isinstance(u, A.Obj) else None
            if isinstance(t, A.Obj):
                kind, params = t.cls, t
            elif isinstance(t, tuple):
                kind, params = t[1], A.Obj(t[1], {})
            else:
                raise StepError("unknown adjustment layer")
            if kind not in ("Lvls", "vibrance", "BrgC", "HStr", "Crvs", "Invr", "Thrs", "Pstr", "exposure", "GdMp"):
                self.skipped.append(Skip(self.i, st.title, f"{ADJ_NAMES.get(kind, kind)} adjustment is not supported yet (layer kept, no effect)"))
            l = Layer(d.next_name(ADJ_NAMES.get(kind, kind)), "adjust", adjust=(kind, params))
            if isinstance(u, A.Obj):
                self.set_layer(l, A.Obj("", {k: v for k, v in u.items.items() if k != "Type"}))
            d.place_above(l, d.target)
            d.targets = [l]
            d.channel = "rgb"
            return
        if cls == "contentLayer":
            u = o.get("Usng")
            t = u.get("Type") if isinstance(u, A.Obj) else None
            kind = t.cls if isinstance(t, A.Obj) else "solidColorLayer"
            l = Layer(d.next_name(FILL_NAMES.get(kind, "Fill")), "fill", content=u)
            if isinstance(u, A.Obj):
                self.set_layer(l, A.Obj("", {k: v for k, v in u.items.items() if k != "Type"}))
            if d.selection is not None:
                l.mask = d.u8(d.selection)
                d.selection = None
            d.place_above(l, d.target)
            d.targets = [l]
            d.channel = "rgb"
            return
        if cls == "Path":
            src = o.get("From")
            if d.selection is None:
                raise StepError("there is no selection to make a path from")
            tol = num(o.get("Tlrn"), 2.0)
            m = (d.selection >= 0.5).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            paths = []
            for c in cs:
                if len(c) < 3:
                    continue
                c = cv2.approxPolyDP(c, max(0.5, tol * d.s), True)
                paths.append(c[:, 0, :].astype(np.float32))
            d.path = paths
            d.selection = None
            del src
            return
        if cls == "layerSection":
            members = d.stack_sorted(self.layers_of(o.get("From"), required=False) or list(d.targets))
            top = d.topmost(members)
            g = Layer(d.next_name("Group"), "group", mode="passThrough")
            u = o.get("Usng")
            if isinstance(u, A.Obj):
                self.set_layer(g, u)
            d.place_above(g, top)
            for m in members:
                d.remove(m)
                d.add(m, g)
            d.targets = [g]
            return
        if cls in ("SnpS", "HstS"):
            return
        if cls == "Dcmn":
            raise StepError("making a new document is not supported")
        raise StepError(f"making {cls.strip()} is not supported")

    def make_mask(self, o):
        d = self.d
        t = d.target
        if t is None:
            raise StepError("no layer selected")
        how = enum(o.get("Usng"), "RvlA")
        if how == "RvlA":
            t.mask = np.full((d.h, d.w), 255, np.uint8)
        elif how == "HdAl":
            t.mask = np.zeros((d.h, d.w), np.uint8)
        elif how == "RvlS":
            sel = d.selection if d.selection is not None else np.ones((d.h, d.w), np.float32)
            t.mask = d.u8(sel)
        elif how == "HdSl":
            sel = d.selection if d.selection is not None else np.zeros((d.h, d.w), np.float32)
            t.mask = d.u8(1 - sel)
        else:
            raise StepError(f"mask type {how!r} is not supported")
        t.mask_feather = 0.0
        d.selection = None
        d.channel = "mask"

    def do_copyToLayer(self, o, st):
        d = self.d
        src = d.target
        if src is None or src.kind != "pixel":
            raise StepError("select a pixel layer first")
        rgb, a = d.rgba(src)
        new = Layer(d.copy_name(src), rgb=None)
        if d.selection is not None:
            a = a * d.selection[..., None]
            d.set_rgba(new, rgb, a)
        else:
            new.rgb = None if src.rgb is None else src.rgb.copy()
            new.alpha = None if src.alpha is None else src.alpha.copy()
            new.opacity, new.fill, new.mode = src.opacity, src.fill, src.mode
            new.effects = src.effects
            new.blend_if = list(src.blend_if)
            new.mask = None if src.mask is None else src.mask.copy()
            new.mask_feather = src.mask_feather
        new.visible = True
        d.place_above(new, src)
        d.targets = [new]
        d.channel = "rgb"

    def do_duplicate(self, o, st):
        d = self.d
        ref = o.get("null")
        if isinstance(ref, A.Ref) and ref.items and ref.items[0].cls == "Dcmn":
            raise StepError("duplicating the document is not supported")
        sel = d.selection
        d.selection = None
        made = []
        for src in self.layers_of(ref) if ref is not None else list(d.targets):
            d.targets = [src]
            self.do_copyToLayer(o, st)
            l = d.target
            nm = o.get("Nm  ")
            if nm:
                l.name = str(nm)
            made.append(l)
        d.selection = sel
        d.targets = made or d.targets

    def do_delete(self, o, st):
        d = self.d
        ref = o.get("null")
        if isinstance(ref, A.Ref) and ref.items:
            it = ref.items[0]
            if it.cls == "Path":
                d.path = None
                return
            if it.cls == "Chnl":
                v = it.value.value if isinstance(it.value, A.Enum) else ""
                if v == "Msk " and d.target is not None:
                    d.target.mask = None
                    d.channel = "rgb"
                    return
                raise StepError("deleting channels is not supported")
        layers = self.layers_of(ref)
        below = None
        for l in layers:
            if l.parent is not None:
                sib = l.parent.children
                i = sib.index(l)
                below = sib[i - 1] if i > 0 else (sib[i + 1] if i + 1 < len(sib) else l.parent)
            d.remove(l)
        if below is d.root:
            below = None
        d.targets = [below] if below is not None and below not in layers else []
        d.channel = "rgb"

    def do_move(self, o, st):
        d = self.d
        layers = self.layers_of(o.get("null"))
        to = o.get("T   ")
        if not isinstance(to, A.Ref) or not to.items:
            raise StepError("unknown move")
        it = to.items[0]
        for l in layers:
            if it.cls == "Lyr " and it.form == "indx":
                flat = [x for x in d.flat() if x is not l]
                idx = int(it.value)
                d.remove(l)
                if idx <= 0:
                    d.add(l, d.root, 1 if d.root.children and d.root.children[0].background else 0)
                elif idx - 1 < len(flat):
                    ref = flat[min(idx - 1, len(flat) - 1)]
                    d.place_above(l, ref)
                else:
                    d.add(l, d.root)
            elif it.cls == "Lyr " and it.form == "Enmr":
                v = it.value.value if isinstance(it.value, A.Enum) else ""
                p = l.parent
                sib = p.children
                i = sib.index(l)
                sib.pop(i)
                if v == "Nxt ":
                    i = min(len(sib), i + 1)
                elif v == "Prvs":
                    i = max(1 if sib and sib[0].background else 0, i - 1)
                elif v == "Frnt":
                    i = len(sib)
                elif v == "Back":
                    i = 1 if sib and sib[0].background else 0
                sib.insert(i, l)
            elif it.cls == "Lyr " and it.form == "name":
                ref = d.find(it.value)
                if ref is None:
                    raise StepError(f"layer {it.value!r} not found")
                d.remove(l)
                d.place_above(l, ref)
            else:
                raise StepError("moving to that place is not supported")

    def merge(self, layers, name_from):
        d = self.d
        layers = d.stack_sorted(layers)
        top = d.topmost(layers)
        rgb = np.zeros((d.h, d.w, 3), np.float32)
        a = np.zeros((d.h, d.w, 1), np.float32)
        tmp = Layer("", "group")
        parents = {}
        for l in layers:
            parents[id(l)] = (l.parent, l.parent.children.index(l))
        holder = Layer("", "group", children=list(layers))
        rgb, a = d.composite_into(holder, rgb, a, self.approx)
        del tmp
        merged = Layer(name_from.name, rgb=None)
        d.set_rgba(merged, rgb, a)
        merged.visible = True
        d.place_above(merged, top)
        for l in layers:
            d.remove(l)
        d.targets = [merged]
        d.channel = "rgb"
        return merged

    def do_mergeLayersNew(self, o, st):
        d = self.d
        layers = list(d.targets)
        if len(layers) == 1:                      # merge down
            l = layers[0]
            sib = l.parent.children
            i = sib.index(l)
            if i == 0:
                raise StepError("nothing below to merge with")
            layers = [sib[i - 1], l]
            name = sib[i - 1]
        else:
            name = d.topmost(layers)
        self.merge(layers, name)

    def do_mergeVisible(self, o, st):
        d = self.d
        rgb, a = d.flatten(self.approx)
        if o.get("Dplc"):
            t = d.target
            if t is not None and t.kind == "pixel" and t.empty() and t.mask is None:
                d.set_rgba(t, rgb, a)
                return
            l = Layer(d.next_name("Layer"))
            d.set_rgba(l, rgb, a)
            d.place_above(l, t)
            d.targets = [l]
            d.channel = "rgb"
            return
        visible = [l for l in d.walk() if l.visible and l.kind != "group"]
        if not visible:
            return
        top = d.topmost(visible)
        name = "Background" if any(l.background for l in visible) else top.name
        l = Layer(name, background=any(x.background for x in visible))
        d.set_rgba(l, rgb, a)
        d.place_above(l, top)
        for v in visible:
            d.remove(v)
        d.targets = [l]

    def do_flattenImage(self, o, st):
        d = self.d
        rgb, a = d.flatten(self.approx)
        if a is not None:
            rgb = rgb * a + (1 - a)
        d.root.children = []
        l = Layer("Background", rgb=d.u8(rgb), background=True)
        d.add(l, d.root)
        d.targets = [l]

    def do_fill(self, o, st):
        d = self.d
        t = d.target
        using = enum(o.get("Usng"), "FrgC")
        op = pct(o.get("Opct"), 1.0)
        mode = enum(o.get("Md  "), "Nrml")
        if using == "FrgC":
            c = d.fg
        elif using == "BckC":
            c = d.bg
        elif using == "Wht ":
            c = np.ones(3, np.float32)
        elif using == "Blck":
            c = np.zeros(3, np.float32)
        elif using == "Gry ":
            c = np.full(3, 0.5, np.float32)
        elif using == "Ptrn":
            c = None
        elif using == "contentAware":
            raise StepError("Content-Aware Fill is not supported")
        else:
            raise StepError(f"fill with {using!r} is not supported")
        if t is None:
            raise StepError("no layer selected")
        if d.channel == "mask" and t.mask is not None:
            k = (d.selection if d.selection is not None else np.ones((d.h, d.w), np.float32)) * op
            v = float(F.lum(c[None, None])[0, 0]) if c is not None else 1.0
            m = t.mask.astype(np.float32) / 255
            t.mask = d.u8(m * (1 - k) + v * k)
            return
        if c is None:
            prgb, pa = d.pattern_image(o.get("Ptrn"), pct(o.get("Scl "), 1.0))
            src, src_a = prgb, pa
        else:
            src = np.broadcast_to(c, (d.h, d.w, 3)).astype(np.float32)
            src_a = np.ones((d.h, d.w, 1), np.float32)
        k = (d.selection[..., None] if d.selection is not None else 1.0) * op
        rgb, a = d.rgba(t)
        if o.get("PrsT"):                          # preserve transparency
            nrgb, _ = F.composite(rgb, a, src, src_a * k, mode)
            d.set_rgba(t, nrgb, a)
        else:
            nrgb, na = F.composite(rgb, a, src, src_a * k, mode)
            d.set_rgba(t, nrgb, na)

    def do_stroke(self, o, st):
        d = self.d
        ref = o.get("null")
        if not (isinstance(ref, A.Ref) and ref.items and ref.items[0].cls == "Path"):
            raise StepError("stroking a selection is not supported yet")
        if not d.path:
            raise StepError("there is no path to stroke")
        t = d.target
        if t is None:
            raise StepError("no layer selected")
        brush = d.brush
        diam = (d.brush_diameter or (brush.diameter if brush else 30.0)) * d.s
        rng = np.random.default_rng(1000 + self.i)
        if d.channel == "mask" and t.mask is not None:
            buf = np.zeros((d.h, d.w, 2), np.float32)
            buf[..., 0] = t.mask.astype(np.float32) / 255
            paint_path(d, buf, [float(F.lum(d.fg[None, None])[0, 0])], brush, diam, rng, 1)
            t.mask = d.u8(buf[..., 0])
        else:
            if t.kind != "pixel":
                raise StepError(f"'{t.name}' is not a pixel layer")
            rgb, a = d.rgba(t)
            buf = np.concatenate([rgb, a], axis=2)
            paint_path(d, buf, d.fg, brush, diam, rng, 3)
            d.set_rgba(t, buf[..., :3], buf[..., 3:4])
        self.approx.add("brush strokes along a path")

    # -- adjustments & filters applied straight to pixels --------------------
    def do_levels(self, o, st):
        self.edit_pixels(lambda x: levels_obj(x, o))

    def do_curves(self, o, st):
        self.edit_pixels(lambda x: curves_obj(x, o))

    def do_hueSaturation(self, o, st):
        self.edit_pixels(lambda x: hue_sat_obj(x, o))

    def do_vibrance(self, o, st):
        self.edit_pixels(lambda x: F.vibrance(x, num(o.get("vibrance")), num(o.get("Strt"))))

    def do_brightnessEvent(self, o, st):
        self.edit_pixels(lambda x: F.brightness_contrast(
            x, num(o.get("Brgh")), num(o.get("Cntr")), bool(o.get("useLegacy", False))),
            "Brightness/Contrast (close)")

    def do_desaturate(self, o, st):
        self.edit_pixels(F.desaturate)

    def do_invert(self, o, st):
        self.edit_pixels(F.invert)

    def do_posterization(self, o, st):
        self.edit_pixels(lambda x: F.posterize(x, num(o.get("Lvls"), 4)))

    def do_thresholdClassEvent(self, o, st):
        self.edit_pixels(lambda x: F.threshold(x, num(o.get("Lvl "), 128)))

    def do_exposure(self, o, st):
        self.edit_pixels(lambda x: self.d.apply_adjust(x, ("exposure", o)))

    def do_highPass(self, o, st):
        self.edit_pixels(lambda x: F.high_pass(x, num(o.get("Rds "), 10) * self.d.s))

    def do_gaussianBlur(self, o, st):
        self.edit_pixels(lambda x: F.gauss(x, num(o.get("Rds "), 1) * self.d.s))

    def do_boxblur(self, o, st):
        self.edit_pixels(lambda x: F.box(x, num(o.get("Rds "), 1) * self.d.s))

    def do_blur(self, o, st):
        self.edit_pixels(lambda x: F.gauss(x, 0.8 * self.d.s))

    def do_blurMore(self, o, st):
        self.edit_pixels(lambda x: F.gauss(x, 1.6 * self.d.s))

    def do_motionBlur(self, o, st):
        self.edit_pixels(lambda x: F.motion(x, num(o.get("Dstn"), 10) * self.d.s, num(o.get("Angl"))))

    def do_surfaceBlur(self, o, st):
        self.edit_pixels(lambda x: F.surface_blur(x, num(o.get("Rds "), 5) * self.d.s, num(o.get("Thsh"), 15)),
                         "Surface Blur (close)")

    def do_median(self, o, st):
        self.edit_pixels(lambda x: F.median(x, num(o.get("Rds "), 1) * self.d.s))

    def do_unsharpMask(self, o, st):
        self.edit_pixels(lambda x: F.unsharp(x, pct(o.get("Amnt"), 0.5) * 1.0,
                                             num(o.get("Rds "), 1) * self.d.s, num(o.get("Thsh"), 0)))

    def do_sharpen(self, o, st):
        self.edit_pixels(lambda x: F.unsharp(x, 0.5, 1.0 * self.d.s))

    def do_sharpenMore(self, o, st):
        self.edit_pixels(lambda x: F.unsharp(x, 1.0, 1.0 * self.d.s))

    def do_addNoise(self, o, st):
        mono = enum(o.get("Mnch"), "") or o.get("Mnch")
        self.edit_pixels(lambda x: F.add_noise(x, pct(o.get("Nose"), 0.1) * 100, bool(mono), 7 + self.i))

    def do_emboss(self, o, st):
        self.edit_pixels(lambda x: F.emboss(x, num(o.get("Angl"), 135), num(o.get("Hght"), 3) * self.d.s,
                                            pct(o.get("Amnt"), 1.0) * 100))

    def do_GEfc(self, o, st):
        d = self.d
        jobs = o.get("GEfs") if isinstance(o.get("GEfs"), list) else [o]
        for job in jobs:
            code = enum(job.get("GEfk"), "")
            spec = F.FILTER_GALLERY.get(code)
            if spec is None:
                raise StepError(f"Filter Gallery '{code.strip()}' is not supported yet")
            name, fn, keys = spec
            kw = {}
            for k, (arg, default) in keys.items():
                v = job.get(k, default)
                kw[arg] = enum(v) if isinstance(v, A.Enum) else (num(v) if isinstance(v, (A.UnitFloat, int, float)) and not isinstance(v, bool) else v)
            if code == "Chrc":
                kw["fg"], kw["bg"] = d.fg.copy(), d.bg.copy()
            self.edit_pixels(lambda x, fn=fn, kw=kw: fn(x, d.s, **kw), f"Filter Gallery: {name} (our own version)")

    def do_transform(self, o, st):
        d = self.d
        t = d.target
        if t is None or t.kind != "pixel":
            raise StepError("select a pixel layer first")
        sx = pct(o.get("Wdth"), 1.0)
        sy = pct(o.get("Hght"), 1.0)
        ang = num(o.get("Angl"), 0.0)
        off = o.get("Ofst")
        ox = num(off.get("Hrzn")) if isinstance(off, A.Obj) else 0.0
        oy = num(off.get("Vrtc")) if isinstance(off, A.Obj) else 0.0
        if isinstance(off, A.Obj) and isinstance(off.get("Hrzn"), A.UnitFloat) and off.get("Hrzn").unit == "#Rlt":
            ox, oy = ox * 300 / 72 * d.s, oy * 300 / 72 * d.s
        cx, cy = d.w / 2, d.h / 2
        m = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
        m[:, :2] = m[:, :2] @ np.diag([sx, sy])
        m[:, 2] = np.array([cx, cy]) - m[:, :2] @ np.array([cx, cy]) + np.array([ox, oy])
        rgb, a = d.rgba(t)
        rgba = np.concatenate([rgb, a], axis=2)
        out = cv2.warpAffine(rgba, m, (d.w, d.h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        d.set_rgba(t, out[..., :3], np.clip(out[..., 3:4], 0, 1))
        if t.mask is not None:
            t.mask = cv2.warpAffine(t.mask, m, (d.w, d.h), flags=cv2.INTER_LINEAR, borderValue=0)

    def do_copyEffects(self, o, st):
        t = self.d.target
        self.d.style_clip = t.effects if t is not None else None

    def do_pasteEffects(self, o, st):
        for l in self.d.targets:
            l.effects = self.d.style_clip

    def do_imageSize(self, o, st):
        raise StepError("Image Size is ignored (OilStudio sizes the image before the action runs)")

    def do_canvasSize(self, o, st):
        raise StepError("Canvas Size is not supported")


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def scan(action: A.Action) -> dict:
    """What would happen without running: supported / approximate / unsupported."""
    exact, close, missing = 0, 0, []
    close_names = {"GEfc", "stroke", "surfaceBlur", "brightnessEvent"}
    for i, st in enumerate(action.steps, 1):
        if not st.enabled:
            continue
        ev = ALIASES.get(st.event, st.event)
        if ev in NO_OPS:
            exact += 1
            continue
        if ev == "GEfc":
            codes = []
            jobs = st.get("GEfs") if isinstance(st.get("GEfs"), list) else [st.desc]
            for j in jobs:
                if isinstance(j, A.Obj):
                    codes.append(enum(j.get("GEfk"), ""))
            if all(c in F.FILTER_GALLERY for c in codes):
                close += 1
            else:
                missing.append({"step": i, "title": st.title,
                                "what": "Filter Gallery " + ", ".join(c.strip() for c in codes)})
            continue
        if hasattr(Player, "do_" + re.sub(r"\W", "_", ev)):
            if ev in close_names:
                close += 1
            else:
                exact += 1
        else:
            missing.append({"step": i, "title": st.title, "what": st.title})
    total = exact + close + len(missing)
    return {"steps": total, "exact": exact, "close": close, "missing": missing,
            "percent": round(100.0 * (exact + close) / max(total, 1))}


def run(action: A.Action, image_rgb: np.ndarray, *, patterns: PatternStore | None = None,
        brushes: dict | None = None, focus: np.ndarray | None = None,
        brush_layer: str = "brush", reference_side: int = REFERENCE_SIDE,
        progress=None) -> Result:
    t0 = time.time()
    h, w = image_rgb.shape[:2]
    scale = max(h, w) / float(reference_side or max(h, w))
    doc = Doc(image_rgb, scale, patterns or PatternStore([]), brushes or {})
    if brush_layer:
        bl = Layer(brush_layer)
        if focus is not None and focus.any():
            bl.rgb = np.zeros((h, w, 3), np.uint8)
            bl.alpha = focus.astype(np.uint8) if focus.dtype == np.uint8 else Doc.u8(focus.astype(np.float32))
        doc.add(bl, doc.root)
        doc.targets = [bl]
    p = Player(doc, progress)
    p.play(action)
    rgb, a = doc.flatten(p.approx)
    if a is not None:
        rgb = rgb * a + (1 - a)                      # transparent parts on white
    layers = [{"name": l.name, "kind": l.kind, "visible": l.visible,
               "opacity": round(l.opacity * 100), "mode": l.mode} for l in doc.top_down()]
    return Result(Doc.u8(rgb), p.skipped, sorted(p.approx), len(action.steps),
                  time.time() - t0, layers)


def load_brushes(paths: list[str]) -> dict[str, BrushPreset]:
    out = {}
    for p in paths:
        try:
            out.update(read_abr(p))
        except Exception:                                # noqa: BLE001 - a bad .abr only loses its brushes
            pass
    return out
