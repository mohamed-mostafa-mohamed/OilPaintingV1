"""
Play a Photoshop template recipe (JSON, made from a PSD by
tools/psd_to_recipe.py) on any photo.

A recipe is the PSD's visible layers, bottom to top:

    {"document": {"width": 3000, "height": 4000},
     "steps": [
       {"op": "photo", "filters": [ {"op": "oil_paint", ...}, {"op": "posterize", ...},
                                    {"op": "cutout", ...}, {"op": "gaussian_blur", ...},
                                    {"op": "displace", "map": "displace_map.png", ...} ]},
       {"op": "layer", "image": "texture_overlay.png", "mode": "overlay", "opacity": 0.569},
       {"op": "adjustment", "kind": "curves", "settings": {...}, "opacity": 0.851},
       ...]}

Sizes in the recipe are in document pixels; the photo is mapped onto the
document like Photoshop's "Place" with "fill" (cover), so every distance is
scaled by the same factor and asset pictures are cover-cropped - the
photo's own aspect ratio and crop are kept.

Standard operations (blend modes, Posterize, Gaussian Blur, Displace,
Levels, Curves) follow Photoshop's formulas. Oil Paint and Cutout are
Adobe's own algorithms (not published); they are rebuilt here and tuned
against the template's layers (see tools/verify_recipe.py).
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _g(x, s):
    return cv2.GaussianBlur(x, (0, 0), s) if s > 0.05 else x


def lum(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def cover(img, h, w, interp=cv2.INTER_AREA):
    """Scale ``img`` to cover (h, w) and crop the middle (Place / fill)."""
    ih, iw = img.shape[:2]
    k = max(h / ih, w / iw)
    nh, nw = max(h, int(round(ih * k))), max(w, int(round(iw * k)))
    big = cv2.resize(img, (nw, nh), interpolation=interp if k < 1 else cv2.INTER_CUBIC)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    return np.ascontiguousarray(big[y0:y0 + h, x0:x0 + w])


# --------------------------------------------------------------------------
# blend modes (Photoshop formulas, 0..1 floats)
# --------------------------------------------------------------------------
def blend(mode, b, s):
    if mode == "normal":
        return s
    if mode == "multiply":
        return b * s
    if mode == "screen":
        return 1 - (1 - b) * (1 - s)
    if mode == "overlay":
        return np.where(b <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == "hard light":
        return np.where(s <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == "soft light":
        d = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(np.clip(b, 0, 1)))
        return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))
    if mode == "color dodge":
        return np.where(s >= 1 - EPS, 1.0, np.minimum(1, b / np.maximum(1 - s, EPS)))
    if mode == "color burn":
        return np.where(s <= EPS, 0.0, 1 - np.minimum(1, (1 - b) / np.maximum(s, EPS)))
    if mode == "linear dodge":
        return np.clip(b + s, 0, 1)
    if mode == "linear burn":
        return np.clip(b + s - 1, 0, 1)
    if mode == "linear light":
        return np.clip(b + 2 * s - 1, 0, 1)
    if mode == "vivid light":
        burn = 1 - np.minimum(1, (1 - b) / np.maximum(2 * s, EPS))
        dodge = np.minimum(1, b / np.maximum(2 * (1 - s), EPS))
        return np.where(s <= 0.5, burn, dodge)
    if mode == "pin light":
        return np.where(s <= 0.5, np.minimum(b, 2 * s), np.maximum(b, 2 * s - 1))
    if mode == "darken":
        return np.minimum(b, s)
    if mode == "lighten":
        return np.maximum(b, s)
    if mode == "difference":
        return np.abs(b - s)
    if mode == "exclusion":
        return b + s - 2 * b * s
    if mode == "subtract":
        return np.clip(b - s, 0, 1)
    if mode == "divide":
        return np.clip(b / np.maximum(s, EPS), 0, 1)
    return s


# --------------------------------------------------------------------------
# adjustments
# --------------------------------------------------------------------------
def curve_lut(points):
    """Photoshop curve through (in, out) points (natural cubic spline)."""
    xs = np.array([p["in"] for p in points], float)
    ys = np.array([p["out"] for p in points], float)
    n = len(xs)
    x = np.arange(256, dtype=float)
    if n == 2:
        return (np.clip(np.interp(x, xs, ys), 0, 255) / 255).astype(np.float32)
    hh = np.diff(xs)
    a = np.zeros((n, n)); r = np.zeros(n)
    a[0, 0] = a[-1, -1] = 1
    for i in range(1, n - 1):
        a[i, i - 1], a[i, i], a[i, i + 1] = hh[i - 1], 2 * (hh[i - 1] + hh[i]), hh[i]
        r[i] = 3 * ((ys[i + 1] - ys[i]) / hh[i] - (ys[i] - ys[i - 1]) / hh[i - 1])
    c = np.linalg.solve(a, r)
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 2)
    dx = x - xs[i]
    b = (ys[i + 1] - ys[i]) / hh[i] - hh[i] * (2 * c[i] + c[i + 1]) / 3
    d = (c[i + 1] - c[i]) / (3 * hh[i])
    y = ys[i] + b * dx + c[i] * dx ** 2 + d * dx ** 3
    y = np.where(x < xs[0], ys[0], np.where(x > xs[-1], ys[-1], y))
    return (np.clip(y, 0, 255) / 255).astype(np.float32)


_X256 = np.linspace(0, 1, 256, dtype=np.float32)


def levels(img, lo, hi, gamma, olo, ohi):
    v = np.clip((img - lo / 255.0) / max((hi - lo) / 255.0, 1e-4), 0, 1)
    if abs(gamma - 1) > 1e-4:
        v = np.power(v, 1.0 / max(gamma, 0.01))
    return olo / 255.0 + v * (ohi - olo) / 255.0


def vibrance(img, vib=0.0, sat=0.0):
    out = img
    if vib:
        mx, mn = out.max(axis=2, keepdims=True), out.min(axis=2, keepdims=True)
        s = mx - mn
        l = lum(out)[..., None]
        out = l + (out - l) * (1.0 + vib / 100.0 * (1.0 - s) ** 2)
    if sat:
        l = lum(out)[..., None]
        out = l + (out - l) * (1.0 + sat / 100.0 * 0.6)
    return np.clip(out, 0, 1)


def adjustment(img, kind, st):
    if kind == "levels":
        c = st["channels"][0]
        out = levels(img, c["in_black"], c["in_white"], c["gamma"], c["out_black"], c["out_white"])
        for k, ch in enumerate(st["channels"][1:4]):
            if (ch["in_black"], ch["in_white"], ch["gamma"], ch["out_black"], ch["out_white"]) != (0, 255, 1.0, 0, 255):
                out[..., k] = levels(out[..., k], ch["in_black"], ch["in_white"], ch["gamma"],
                                     ch["out_black"], ch["out_white"])
        return out
    if kind == "curves":
        out = img
        for ch in st["channels"]:
            lut = curve_lut(ch["points"])
            if ch["channel"] == "RGB":
                out = np.interp(np.clip(out, 0, 1), _X256, lut).astype(np.float32)
            elif ch["channel"] in ("R", "G", "B"):
                k = "RGB".index(ch["channel"])
                out = out.copy()
                out[..., k] = np.interp(np.clip(out[..., k], 0, 1), _X256, lut)
        return out
    if kind == "vibrance":
        return vibrance(img, st.get("vibrance", 0), st.get("Strt", 0))
    if kind == "posterize":
        return posterize(img, st["levels"])
    if kind == "invert":
        return 1 - img
    raise ValueError(f"adjustment {kind!r} is not supported yet")


# --------------------------------------------------------------------------
# filters (photo-dependent chain)
# --------------------------------------------------------------------------
def posterize(img, n):
    """Photoshop Posterize: n levels per channel."""
    n = max(2, int(n))
    q = np.floor(np.clip(img, 0, 1) * 255.0 * n / 256.0)
    return (np.round(q * 255.0 / (n - 1)) / 255.0).astype(np.float32)


def gaussian_blur(img, radius_px):
    # Photoshop's "radius" is close to the Gaussian sigma
    return _g(img, radius_px)


def displace(img, mp, horiz, vert, k, wrap="repeat"):
    """Photoshop Displace: map value v moves the sample by (v-128)/128 *
    scale% * 128 px (k = document-to-photo scale); stretch-to-fit map."""
    h, w = img.shape[:2]
    m = cover(mp.astype(np.float32), h, w, cv2.INTER_LINEAR)
    off = (m - 128.0) / 128.0
    amp_x = horiz / 100.0 * 128.0 * k
    amp_y = vert / 100.0 * 128.0 * k
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mode = cv2.BORDER_REPLICATE if wrap == "repeat" else cv2.BORDER_WRAP
    return cv2.remap(img, xx + off * amp_x, yy + off * amp_y, cv2.INTER_LINEAR, borderMode=mode)


def _structure_flow(L, sigma):
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = _g(gx * gx, sigma), _g(gy * gy, sigma), _g(gx * gy, sigma)
    a = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2       # along edges
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def _lic(img, dx, dy, length, steps):
    h, w = img.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = img.astype(np.float32).copy()
    wsum = 1.0
    st = length / max(steps, 1)
    for sign in (1.0, -1.0):
        x, y = gx.copy(), gy.copy()
        pdx, pdy = dx * sign, dy * sign
        for i in range(steps):
            vx = cv2.remap(dx, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            vy = cv2.remap(dy, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            flip = np.where(vx * pdx + vy * pdy < 0, -1.0, 1.0).astype(np.float32)
            vx, vy = vx * flip, vy * flip
            x += vx * st
            y += vy * st
            pdx, pdy = vx, vy
            acc += cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += 1.0
    return acc / wsum


def guided(img, r, eps):
    """Edge-preserving smoothing (guided filter, guide = the image)."""
    r = max(1, int(round(r)))
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)
    mI = mean(img)
    var = mean(img * img) - mI * mI
    a = var / (var + eps)
    b = mI - a * mI
    return mean(a) * img + mean(b)


def oil_paint(img, k, stylization=10.0, cleanliness=10.0, scale=10.0, bristle=9.7,
              lighting=True, angle=-60.0, shine=1.3, tune=None):
    """Photoshop Oil Paint (rebuilt; tuned against the template).

    cleanliness  edge-preserving smoothing strength (0..10)
    stylization  stroke smearing along the image's own edges (0..10)
    scale        brush size (0.1..10)
    bristle      fine bristle grooves in the relief (0..10)
    lighting     relief lit from ``angle`` with ``shine``
    """
    # fitted against the template's Oil Paint layer (tools/verify_recipe.py)
    t = {"smooth_r": 1.5, "smooth_eps": 0.001, "smooth_n": 1, "lic_len": 2.0, "relief": 0.0,
         "spec": 0.0}
    if tune:
        t.update(tune)
    sz = max(scale, 0.1) / 10.0 * k
    out = img
    eps = t["smooth_eps"] * (0.2 + cleanliness / 10.0)
    for _ in range(int(t["smooth_n"])):
        out = guided(out, t["smooth_r"] * sz * 2.5, eps)
    if stylization > 0 and t["lic_len"] > 0:
        dx, dy = _structure_flow(lum(_g(out, 2 * sz)), 4 * sz + 1)
        out = _lic(out, dx, dy, t["lic_len"] * stylization / 10.0 * 10 * sz, 6)
    if lighting and (t["relief"] > 0 or t["spec"] > 0):
        hgt = lum(_g(out, 1.2 * sz))
        dxf, dyf = _structure_flow(hgt, 3 * sz + 1)
        rng = np.random.default_rng(3)
        n = rng.standard_normal(hgt.shape).astype(np.float32)
        br = _lic(_g(n, 0.6 * sz + 0.3), dxf, dyf, 6 * sz + 1, 4)
        br /= br.std() + EPS
        hgt = hgt * 8 * sz + br * bristle / 10.0 * 0.6 * sz
        gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8
        a = math.radians(angle)
        lx, ly, lz = math.cos(a), -math.sin(a), 1.2
        ll = math.sqrt(lx * lx + ly * ly + lz * lz)
        nl = np.sqrt(gx * gx + gy * gy + 1)
        diff = (-gx * lx - gy * ly + lz) / (nl * ll) / (lz / ll)
        hx, hy, hz = lx / ll, ly / ll, lz / ll + 1
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
        spec = np.clip((-gx * hx - gy * hy + hz) / (nl * hn), 0, 1) ** (20 * shine)
        out = out * (1 + (diff - 1)[..., None] * t["relief"] * 10) + (spec * t["spec"] * shine)[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


def cutout(img, k, levels=8, simplicity=6, fidelity=3, tune=None):
    """Filter Gallery Cutout (rebuilt): flat shapes in ``levels`` tones,
    edges simplified (simplicity 0..10) and following the picture
    (fidelity 1..3)."""
    t = {"pre_r": 0.0, "med": 0.0}          # fitted (tools/verify_recipe.py)
    if tune:
        t.update(tune)
    h, w = img.shape[:2]
    base = _g(img, t["pre_r"] * k * (1 + simplicity / 5.0))
    L = lum(base)
    lo, hi = np.percentile(L, 0.5), np.percentile(L, 99.5)
    n = max(2, int(levels))
    idx = np.clip(((L - lo) / max(hi - lo, 1e-4) * n).astype(np.int32), 0, n - 1).astype(np.uint8)
    ks = int(t["med"] * (simplicity * 1.5 + (3 - fidelity) * 2) * k) | 1
    if ks >= 3:
        idx = cv2.medianBlur(idx, min(ks, 255))
    # each connected region of one level takes its mean colour
    labels = np.zeros(idx.shape, np.int32)
    nxt = 0
    for v in range(n):
        m = (idx == v).astype(np.uint8)
        if not m.any():
            continue
        c, lab = cv2.connectedComponents(m, connectivity=4)
        sel = lab > 0
        labels[sel] = lab[sel] + nxt
        nxt += c
    cnt = np.bincount(labels.ravel(), minlength=nxt + 1).astype(np.float32)
    out = np.empty_like(img)
    for ch in range(3):
        mean = np.bincount(labels.ravel(), weights=img[..., ch].ravel(), minlength=nxt + 1)
        out[..., ch] = (mean / np.maximum(cnt, 1))[labels]
    return out


# --------------------------------------------------------------------------
# the recipe player
# --------------------------------------------------------------------------
class Recipe:
    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            self.data = json.load(fh)
        self.dir = os.path.dirname(os.path.abspath(path))
        self._assets = {}

    @property
    def name(self):
        return self.data.get("name", "recipe")

    def asset(self, rel, gray=False):
        key = (rel, gray)
        if key not in self._assets:
            path = os.path.join(self.dir, rel)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            self._assets[key] = img
        return self._assets[key]

    def has_assets(self):
        for st in self.data["steps"]:
            for f in st.get("filters", []):
                if "map" in f and not os.path.exists(os.path.join(self.dir, f["map"])):
                    return False
            if "image" in st and not os.path.exists(os.path.join(self.dir, st["image"])):
                return False
        return True


def run(recipe: Recipe, rgb, controls=None, progress=None, keep_steps=False, fallback=None):
    """Play the recipe on an RGB float picture (0..1). ``controls`` scale
    the recipe (see CONTROLS). Returns (result, {step name: picture})."""
    c = dict(CONTROL_DEFAULTS)
    c.update(controls or {})
    doc = recipe.data["document"]
    h, w = rgb.shape[:2]
    k = max(h / doc["height"], w / doc["width"])            # document px -> photo px
    steps = {}
    out = rgb.astype(np.float32)
    total = len(recipe.data["steps"])

    def report(i, label):
        if progress:
            progress(0.05 + 0.9 * i / max(total, 1), label)

    for i, st in enumerate(recipe.data["steps"]):
        op = st["op"]
        report(i, st.get("label", op))
        if op == "photo":
            x = out
            for f in st.get("filters", []):
                fop = f["op"]
                report(i, f.get("label", fop))
                if fop == "oil_paint":
                    s = dict(f["settings"])
                    s["cleanliness"] = s.get("cleanliness", 10) * c["paint_smooth"]
                    s["shine"] = s.get("shine", 1.3) * c["paint_shine"]
                    tune = dict(f.get("tune") or {})
                    if c["paint_relief"] > 0:          # extra Oil Paint lighting (not in the fit)
                        tune["relief"] = tune.get("relief", 0) + 0.012 * c["paint_relief"]
                        tune["spec"] = tune.get("spec", 0) + 0.25 * c["paint_relief"]
                    x = oil_paint(x, k, tune=tune, **s)
                elif fop == "posterize":
                    lv = int(round(f["levels"] * c["posterize_mult"]))
                    x = posterize(x, lv)
                elif fop == "cutout":
                    s = dict(f["settings"])
                    s["levels"] = int(round(s.get("levels", 8) * c["cutout_mult"]))
                    x = cutout(x, k, tune=f.get("tune"), **s)
                elif fop == "gaussian_blur":
                    x = gaussian_blur(x, f["radius"] * k)
                elif fop == "displace":
                    if c["warp"] > 0:
                        mp = (recipe.asset(f["map"], gray=True) if fallback is None
                              else fallback("map", f))
                        x = displace(x, mp, f["horizontal"] * c["warp"], f["vertical"] * c["warp"],
                                     k, f.get("edges", "repeat"))
                else:
                    raise ValueError(f"filter {fop!r} is not supported")
                name = f"{st.get('label', 'photo')} / {f.get('label', fop)}"
                if keep_steps is True or (keep_steps and name in keep_steps):
                    steps[name] = np.clip(x, 0, 1)
            out = np.clip(x, 0, 1)
        elif op == "layer":
            img = recipe.asset(st["image"]) if fallback is None else fallback("layer", st)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            img = cover(img, h, w)
            if img.shape[2] == 4:
                a = img[..., 3:4].astype(np.float32) / 255
                img = img[..., :3]
            else:
                a = 1.0
            src = img[..., ::-1].astype(np.float32) / 255 if img.shape[2] == 3 else img
            op_ = st["opacity"] * c["texture"] * a
            op_ = np.clip(op_, 0, 1)
            out = out + (blend(st["mode"], out, src) - out) * op_
        elif op == "adjustment":
            amt = st["opacity"] * c["adjust"]
            if amt > 0:
                adj = adjustment(np.clip(out, 0, 1), st["kind"], st["settings"])
                out = out + (adj - out) * min(1.0, amt)
        name = st.get("label", op)
        if keep_steps is True or (keep_steps and name in keep_steps):
            steps[name] = np.clip(out, 0, 1)
    if progress:
        progress(0.97, "done")
    return np.clip(out, 0, 1), steps


# sliders: 1.0 = exactly the recipe
CONTROL_DEFAULTS = {"warp": 1.0, "texture": 1.0, "adjust": 1.0, "paint_smooth": 1.0,
                    "paint_shine": 1.0, "paint_relief": 0.0, "posterize_mult": 1.0,
                    "cutout_mult": 1.0}
