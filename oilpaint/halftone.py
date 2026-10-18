"""
Halftone — a vintage halftone print look, built natively (no Photoshop, no
action file, no third-party textures).

Pipeline (each step measured against a reference made in Photoshop):

  1. tone      Black & White (Photoshop's default channel mix), a Levels
               stretch and local contrast — the "Camera Raw" punch
  2. stipple   the photo is posterized to 9 grey steps; fine grain darkens
               specks by two steps (denser in the mid-tones, thinner in the
               highlights) and strong detail turns into black edge specks
  3. cloth     a woven Aida-cloth texture, laid on with Linear Burn: its
               holes make the regular grid of dark dots
  4. scratches thin cracked veins laid on with Color Dodge: they bleach the
               picture into light, broken lines
  5. colour    the result's brightness is mapped onto a film-toned sepia
               ramp (dark brown -> cream); both end colours can be changed

Every texture here is generated in code, so the output can be used and sold
freely. Distances are given for a 4000 px wide picture and scaled to the
real size.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

REF_SIDE = 4000.0

# Normalised film response of the default ramp, per channel (R, G, B), at
# brightness 0, 1/8 ... 1. The ramp runs from the dark colour to the light
# colour along these curves, which gives the warm "film" midtones.
_FILM = np.array([
    [0.000, 0.000, 0.000],
    [0.076, 0.067, 0.057],
    [0.238, 0.202, 0.163],
    [0.448, 0.398, 0.335],
    [0.635, 0.586, 0.518],
    [0.806, 0.748, 0.699],
    [0.892, 0.871, 0.829],
    [0.958, 0.959, 0.931],
    [1.000, 1.000, 1.000],
], np.float32)

DEFAULT_DARK = "#21130d"
DEFAULT_LIGHT = "#f9f4c4"


def _hex(c: str) -> np.ndarray:
    c = (c or "#000000").lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _gauss(x, s):
    return cv2.GaussianBlur(x, (0, 0), s) if s > 0.05 else x


def _smooth_noise(shape, sigma, rng):
    n = rng.standard_normal(shape).astype(np.float32)
    n = _gauss(n, sigma)
    return n / (n.std() + 1e-6)


# --------------------------------------------------------------------------
# 1. tone
# --------------------------------------------------------------------------
def black_white(rgb: np.ndarray, w=(40, 60, 40, 60, 20, 80)) -> np.ndarray:
    """Photoshop's Black & White adjustment (Reds, Yellows, Greens, Cyans,
    Blues, Magentas). rgb float 0..1 -> grey 0..1."""
    wr, wy, wg, wc, wb, wm = (v / 100.0 for v in w)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    md = r + g + b - mx - mn
    prim = np.where(r >= np.maximum(g, b), wr, np.where(g >= b, wg, wb))
    sec = np.where(b <= np.minimum(r, g), wy, np.where(r <= g, wc, wm))
    return mn + (md - mn) * sec + (mx - md) * prim


def tone(rgb, s, contrast=50, clarity=50):
    g = black_white(rgb)
    # Levels 15..224 at the default contrast (wider / narrower around it)
    c = (contrast - 50) / 50.0
    lo = 15 + 20 * c
    hi = 224 - 20 * c
    g = np.clip((g - lo / 255.0) / max((hi - lo) / 255.0, 0.05), 0, 1)
    k = clarity / 50.0
    g = np.clip(g + 0.45 * k * (g - _gauss(g, 30 * s)), 0, 1)     # clarity
    g = np.clip(g + 0.50 * k * (g - _gauss(g, 5 * s)), 0, 1)      # texture
    return g


# --------------------------------------------------------------------------
# 2. stipple (posterize + grain specks + black edge specks)
# --------------------------------------------------------------------------
def stipple(g, s, rng, grain=50, edges=50, levels=9):
    step = 1.0 / (levels - 1)
    q = np.minimum(np.round(g / step) * step, 1.0)
    n = _smooth_noise(g.shape, max(0.6, 1.0 * s), rng)
    det = (_gauss(g, 2.5 * s) - _gauss(g, 0.7 * s)) / 0.03       # darker than around
    amp = (1.35 - 2.0 * np.maximum(g - 0.5, 0)) * (grain / 50.0)
    speck = (n * amp + det) > 1.0
    out = np.where(speck, np.maximum(q - 2 * step, 0), q)
    if edges > 0:
        thr = 0.065 * (50.0 / max(edges, 1)) ** 0.6
        edge = (_gauss(g, 2.0 * s) - _gauss(g, 0.6 * s) + n * 0.012) > thr
        out = np.where(edge, 0.0, out)
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# 3. cloth (Aida weave) — Linear Burn
# --------------------------------------------------------------------------
def cloth(h, w, s, rng, pitch=38.0):
    p = max(4.0, pitch * s)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    # a slightly irregular weave: the grid wanders a little, like real cloth
    wob = _smooth_noise((h, w), p * 3, rng) * p * 0.05
    wob2 = _smooth_noise((h, w), p * 3, rng) * p * 0.05
    u = (x + wob) / p
    v = (y + wob2) / p
    fu, fv = u - np.floor(u), v - np.floor(v)
    # holes at the grid crossings
    du = np.minimum(fu, 1 - fu)
    dv = np.minimum(fv, 1 - fv)
    hw = 0.13
    soft = 0.07
    hole = np.clip((hw - np.maximum(du, dv)) / soft + 0.5, 0, 1)
    # thread blocks: each cell alternates horizontal / vertical fibres
    parity = ((np.floor(u) + np.floor(v)) % 2).astype(np.float32)
    fibre_h = np.sin(fv * math.pi * 8) * 0.5 + 0.5
    fibre_v = np.sin(fu * math.pi * 8) * 0.5 + 0.5
    fibre = parity * fibre_h + (1 - parity) * fibre_v
    # gaps between the thread bundles inside a cell
    bundle = np.clip(np.minimum(np.abs(fu - 0.5), np.abs(fv - 0.5)) * 6, 0, 1)
    tex = 0.935 - 0.035 * fibre - 0.02 * (1 - bundle)
    tex += _smooth_noise((h, w), 0.8 * s + 0.3, rng) * 0.012
    tex = tex * (1 - hole) + 0.47 * hole
    return np.clip(tex, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# 4. scratches (cracked veins) — Color Dodge
# --------------------------------------------------------------------------
def veins(h, w, s, rng, amount=50):
    """Dark field with thin, branching, broken light veins (RGB 0..1)."""
    base = np.array([23, 10, 10], np.float32) / 255.0
    col = np.array([201, 177, 61], np.float32) / 255.0
    field = np.empty((h, w, 3), np.float32)
    field[:] = base
    if amount <= 0:
        return field
    m = np.zeros((h, w), np.float32)
    # count in reference-size units: the same picture gets the same number
    # of scratches at any resolution
    n_cracks = int(round(6 * (amount / 50.0) * (h * w) / (s * s) / (3000 * 4000)))
    n_cracks = max(1, n_cracks) if amount > 0 else 0
    seg = 9.0 * s

    def crack(x, y, ang, length, width, depth):
        """Marble-like vein: long straight runs with sudden kinks, and
        thinner branches splitting off. ``length``/``width`` are in
        reference (4000 px) units, so the same seed draws the same veins at
        any size."""
        pts = [(x, y)]
        for _ in range(int(length / 9.0)):
            ang += rng.normal(0, 0.07)
            if rng.random() < 0.035:                    # a sharp kink
                ang += rng.choice([-1, 1]) * rng.uniform(0.35, 0.9)
            x += math.cos(ang) * seg
            y += math.sin(ang) * seg
            pts.append((x, y))
            if depth < 3 and rng.random() < 0.03:
                crack(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2),
                      length * rng.uniform(0.2, 0.5), width * 0.6, depth + 1)
        if len(pts) < 2:
            return
        arr = np.array(pts, np.float32)
        k = len(arr)
        for i in range(k - 1):
            t = i / max(k - 1, 1)
            # thicker in the middle, uneven along the way
            wd = width * s * (0.3 + 0.7 * math.sin(math.pi * t)) * rng.uniform(0.7, 1.2)
            wd = max(1, int(round(wd)))
            p0 = (int(arr[i, 0] * 8), int(arr[i, 1] * 8))
            p1 = (int(arr[i + 1, 0] * 8), int(arr[i + 1, 1] * 8))
            cv2.line(m, p0, p1, 1.0, wd, cv2.LINE_AA, shift=3)

    for _ in range(n_cracks):
        x, y = rng.uniform(-0.1, 1.1) * w, rng.uniform(-0.1, 1.1) * h
        crack(x, y, rng.uniform(0, 2 * math.pi), rng.uniform(1200, 3200),
              rng.uniform(14, 30), 0)
    # broken, flaky edges like gold leaf
    flakes = _smooth_noise((h, w), 1.5 * s + 0.4, rng)
    m = m * np.clip(0.85 + 0.4 * flakes, 0, 1)
    m = _gauss(m, 0.7 * s)
    m = np.clip(m, 0, 1)[..., None]
    return field * (1 - m) + col * m


# --------------------------------------------------------------------------
# 5. colour
# --------------------------------------------------------------------------
def ramp(dark=DEFAULT_DARK, light=DEFAULT_LIGHT, film=100):
    d, l = _hex(dark), _hex(light)
    t = np.linspace(0, 1, len(_FILM), dtype=np.float32)
    f = film / 100.0
    curve = _FILM * f + t[:, None] * (1 - f)            # film off = straight line
    xs = np.linspace(0, 1, 256, dtype=np.float32)
    out = np.empty((256, 3), np.float32)
    for c in range(3):
        out[:, c] = d[c] + (l[c] - d[c]) * np.interp(xs, t, curve[:, c])
    return out


# --------------------------------------------------------------------------
# the whole look
# --------------------------------------------------------------------------
MIN_SIDE = 1600       # grain and dots are made at least this big, then shrunk


def render(image_bgr: np.ndarray, p: dict, progress=None, min_side: int = MIN_SIDE) -> np.ndarray:
    """image_bgr float32/uint8 BGR 0..255 -> float32 BGR 0..255.

    Small pictures (previews) are made at ``min_side`` and shrunk, so the
    grain and the dot grid look the same as on the full-size picture
    instead of turning into blotches."""
    h0, w0 = image_bgr.shape[:2]
    if max(h0, w0) < min_side:
        k = min_side / max(h0, w0)
        big = cv2.resize(np.asarray(image_bgr, np.float32),
                         (int(round(w0 * k)), int(round(h0 * k))), interpolation=cv2.INTER_CUBIC)
        out = render(big, p, progress, min_side=0)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)

    def report(f, label):
        if progress:
            progress(f, label)

    rgb = np.ascontiguousarray(image_bgr[..., ::-1]).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    s = max(h, w) / REF_SIDE
    seed = int(p.get("ht_seed", 7))
    # one generator per texture, so the preview and the full-size result
    # get the same scratches and the same cloth wobble
    rng_grain = np.random.default_rng(seed)
    rng_cloth = np.random.default_rng(seed + 1009)
    rng_veins = np.random.default_rng(seed + 2017)

    report(0.1, "Halftone: tone")
    g = tone(rgb, s, p.get("ht_contrast", 50), p.get("ht_clarity", 50))
    report(0.3, "Halftone: stipple")
    x = stipple(g, s, rng_grain, p.get("ht_grain", 50), p.get("ht_edges", 50))
    x = np.repeat(x[..., None], 3, axis=2)

    grid = float(p.get("ht_grid", 100)) / 100.0
    if grid > 0:
        report(0.5, "Halftone: cloth")
        tex = cloth(h, w, s, rng_cloth, pitch=float(p.get("ht_grid_size", 38)))
        tex = 1 - (1 - tex) * grid
        x = np.clip(x + tex[..., None] - 1, 0, 1)               # Linear Burn
    sc = float(p.get("ht_scratches", 50))
    report(0.7, "Halftone: scratches")
    v = veins(h, w, s, rng_veins, sc)
    x = np.where(v >= 1, 1, np.minimum(1, x / np.maximum(1 - v, 1e-4)))   # Color Dodge

    report(0.9, "Halftone: colour")
    lum = x[..., 0] * 0.3 + x[..., 1] * 0.59 + x[..., 2] * 0.11
    lut = ramp(p.get("ht_dark", DEFAULT_DARK), p.get("ht_light", DEFAULT_LIGHT),
               p.get("ht_film", 100))
    idx = np.clip((lum * 255 + 0.5).astype(np.int32), 0, 255)
    out = lut[idx]
    return (out[..., ::-1] * 255.0).astype(np.float32)
