"""
Canvas Oil - a native recreation of the "Canvas Oil Painting" look
(Envato item EXFF2KE by AB-TheDesigner), without Photoshop.

The Photoshop action stacks several Filter Gallery passes, Levels,
Vibrance, a pattern fill and a couple of stroked selections.  This module
rebuilds the same visual ingredients with OpenCV / numpy:

  1. tone        - Levels-style contrast and a vibrance boost;
  2. daubs       - the photo is broken into flat, angular paint patches
                   (a "paint daubs" mode filter, run twice: big patches in
                   open areas, small ones where there is detail), with a
                   slight vertical drag so patches run down like wet paint;
  3. edges       - every patch edge gets a lit / shadowed paint rim, and
                   strong contours get a thin pale outline;
  4. canvas      - the bright passages open up to white primed canvas, and
                   a canvas weave shows through the whole picture;
  5. scratches   - fine cross-hatched knife scratches, lighter than the
                   paint they cut through, clustered rather than uniform;
  6. speckles    - a sprinkle of small glowing paint spots.

``render(img, p)`` returns float32 BGR in 0..255, same size as ``img``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _noise(h, w, feature, rng):
    """Smooth value noise in 0..1 with blobs about ``feature`` px across."""
    feature = max(2.0, float(feature))
    sh, sw = max(2, int(round(h / feature)) + 2), max(2, int(round(w / feature)) + 2)
    n = rng.random((sh, sw)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.normalize(n, None, 0.0, 1.0, cv2.NORM_MINMAX)


def _fbm(h, w, rng, features, weights):
    out = np.zeros((h, w), np.float32)
    for f, wt in zip(features, weights):
        out += _noise(h, w, f, rng) * wt
    return cv2.normalize(out, None, 0.0, 1.0, cv2.NORM_MINMAX)


def _lum(img):
    """Luminance 0..1 of a float BGR 0..255 image."""
    return (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299) / 255.0


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# 1. tone: Levels + Vibrance
# --------------------------------------------------------------------------


def _tone(img, contrast, vibrance):
    f = img / 255.0
    if contrast > 0:
        # Levels: clip a little of each end, then a gentle S curve
        lo = float(np.percentile(f, 0.6)) * contrast
        hi = 1.0 - (1.0 - float(np.percentile(f, 99.4))) * contrast
        f = np.clip((f - lo) / max(0.2, hi - lo), 0.0, 1.0)
        s = f * f * (3.0 - 2.0 * f)
        f = f + (s - f) * (0.55 * contrast)

    if abs(vibrance) > 1e-3:
        hsv = cv2.cvtColor(np.clip(f * 255, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[..., 1] / 255.0
        # vibrance lifts the muted colours most and leaves vivid ones alone
        if vibrance > 0:
            sat = sat + (1.0 - sat) * sat * vibrance * 1.6 + sat * vibrance * 0.25
        else:
            sat = sat * (1.0 + vibrance)
        hsv[..., 1] = np.clip(sat, 0, 1) * 255.0
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    return f * 255.0


# --------------------------------------------------------------------------
# 2. paint patches
# --------------------------------------------------------------------------


def _palette_labels(img, k, rng):
    """Posterise to ``k`` colours (k-means in Lab) - one label per pixel."""
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    flat = lab.reshape(-1, 3).astype(np.float32)
    sample = flat[rng.choice(len(flat), min(len(flat), 40000), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    cv2.setRNGSeed(int(rng.integers(0, 1 << 30)))
    _, _, centres = cv2.kmeans(sample, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    # nearest centre for every pixel, in chunks to keep memory flat
    labels = np.empty(len(flat), np.uint8)
    c2 = (centres ** 2).sum(1)
    for i in range(0, len(flat), 262144):
        blk = flat[i:i + 262144]
        d = c2[None, :] - 2.0 * blk @ centres.T
        labels[i:i + 262144] = np.argmin(d, axis=1)
    return labels.reshape(img.shape[:2])


def cutout(img, size_px, rng, colours=28, colour_src=None):
    """Photoshop-"Cutout"-style paint patches.

    The photo is posterised, and every colour region is redrawn as a
    simplified polygon - straight, knife-cut sides instead of the soft
    outlines of the photo - and filled with the region's own average
    colour.  ``size_px`` sets both the smallest region kept and how much
    the outlines are simplified.  The shapes are cut from ``img``; the
    colours are taken from ``colour_src`` (default: ``img``).
    """
    colour_src = img if colour_src is None else colour_src
    h, w = img.shape[:2]
    smooth = cv2.bilateralFilter(np.clip(img, 0, 255).astype(np.uint8),
                                 0, 28, max(2.0, size_px * 0.25))
    labels = _palette_labels(smooth.astype(np.float32), colours, rng)
    # clean up single-pixel speckle in the label map
    labels = cv2.medianBlur(labels, 5 if size_px > 8 else 3)

    eps = max(0.8, size_px * 0.16)
    min_area = (size_px * 0.45) ** 2
    polys = []
    for lab in range(colours):
        m = (labels == lab).astype(np.uint8)
        if not m.any():
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            ap = cv2.approxPolyDP(c, eps, True)
            if len(ap) >= 3:
                polys.append((area, ap))
    polys.sort(key=lambda t: -t[0])

    ids = np.zeros((h, w), np.int32)
    for i, (_, ap) in enumerate(polys, start=1):
        cv2.fillPoly(ids, [ap], int(i))
    # anything not covered keeps the smoothed photo
    n = len(polys) + 1
    flat_ids = ids.ravel()
    cnt = np.bincount(flat_ids, minlength=n).astype(np.float32)
    src = colour_src.reshape(-1, 3)
    means = np.stack([np.bincount(flat_ids, weights=src[:, c], minlength=n)
                      for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    out = means[ids].astype(np.float32)
    hole = ids == 0
    if hole.any():
        out[hole] = colour_src.astype(np.float32)[hole]
    return out


def _patch_layer(img, size_px, rng, drag):
    """Angular paint patches about ``size_px`` across, dragged downwards."""
    h, w = img.shape[:2]
    # squash vertically, cut out, stretch back: the patches run down the
    # picture like wet paint pulled by the knife
    sq = 1.0 + 0.9 * drag
    work = cv2.resize(img, (w, max(8, int(h / sq))), interpolation=cv2.INTER_AREA)
    # wobble so the cuts don't follow the photo's outlines exactly
    hh, ww = work.shape[:2]
    amp = max(1.0, size_px * 0.18)
    dx = (_noise(hh, ww, size_px * 1.2, rng) - 0.5) * amp
    dy = (_noise(hh, ww, size_px * 1.2, rng) - 0.5) * amp
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    work = cv2.remap(work, xx + dx, yy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
    # blotchy colour noise at patch size breaks smooth gradients (a sky, a
    # wall) into irregular patches instead of clean posterisation bands
    blot = (_fbm(hh, ww, rng, (size_px * 2.0, size_px * 0.8), (0.6, 0.4)) - 0.5)
    out = cutout(work + blot[..., None] * 26.0, size_px, rng, colour_src=work)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def _dab_cells(paint, cell_px, amount, rng, drag, flatness):
    """Break the paint into small faceted dabs of slightly different shade
    (a Voronoi "crystallize"): the mosaic of short knife touches the action
    leaves in flat areas such as a road, a wall or the sky."""
    h, w = paint.shape[:2]
    sq = 1.0 + 0.8 * drag
    hh = max(8, int(h / sq))
    step = max(3.0, cell_px)
    gx = np.arange(step * 0.5, w, step)
    gy = np.arange(step * 0.5, hh, step)
    px, py = np.meshgrid(gx, gy)
    px = np.clip(px + rng.uniform(-0.45, 0.45, px.shape) * step, 0, w - 1).astype(np.int32)
    py = np.clip(py + rng.uniform(-0.45, 0.45, py.shape) * step, 0, hh - 1).astype(np.int32)
    seeds = np.ones((hh, w), np.uint8)
    seeds[py.ravel(), px.ravel()] = 0
    _, lab = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3,
                                             labelType=cv2.DIST_LABEL_CCOMP)
    lab = cv2.resize(lab.astype(np.float32), (w, h),
                     interpolation=cv2.INTER_NEAREST).astype(np.int32)
    n = int(lab.max()) + 1

    flat_lab = lab.ravel()
    cnt = np.maximum(np.bincount(flat_lab, minlength=n), 1).astype(np.float32)
    src = paint.reshape(-1, 3)
    mean = np.stack([np.bincount(flat_lab, weights=src[:, c], minlength=n)
                     for c in range(3)], 1) / cnt[:, None]
    shade = rng.normal(0.0, 0.055, n).astype(np.float32)
    tint = rng.normal(0.0, 0.018, (n, 3)).astype(np.float32)
    cell_col = mean * (1.0 + shade[:, None] + tint)
    cells = cell_col[lab].astype(np.float32)
    # keep the patch structure: take the cell's colour where the area is
    # flat, only its shading where there is detail
    shaded = paint * (1.0 + shade[lab][..., None] * 0.8)
    # in drifts, not evenly everywhere
    drift = _smoothstep(0.25, 0.75, _fbm(h, w, rng, (cell_px * 14, cell_px * 5), (0.6, 0.4)))
    f = (flatness * amount * (0.35 + 0.65 * drift))[..., None]
    shaded = paint + (shaded - paint) * f
    return shaded + (cells - shaded) * np.clip(f * 0.9, 0, 0.6)


def _detail_map(img):
    """0 in open, flat areas; 1 where the photo is busy."""
    g = _lum(img)
    s = max(img.shape[:2]) / 1000.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 10.0 * s)
    mag /= float(np.percentile(mag, 90)) + 1e-6
    return np.clip(mag, 0.0, 1.0)


# --------------------------------------------------------------------------
# 3. edges: paint rims and pale outlines
# --------------------------------------------------------------------------


def _rims(paint, relief, s):
    """Thick-paint ridges along every patch edge: lit on the side facing
    the light, shadowed on the other, plus a thin dark seam where two
    patches meet and a pale outline on the strongest contours."""
    g = _lum(paint)
    gs = cv2.GaussianBlur(g, (0, 0), 0.9 * s)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    # light from the upper left
    emb = -(gx + gy) * 0.7071
    emb /= float(np.percentile(np.abs(emb), 98.5)) + 1e-6
    emb = np.clip(emb, -1.6, 1.6)
    out = paint * (1.0 + emb[..., None] * 0.34 * relief)

    mag = np.sqrt(gx * gx + gy * gy)
    mag /= float(np.percentile(mag, 99.0)) + 1e-6
    # dark seam: sits on the darker side of each edge
    seam = cv2.GaussianBlur(np.clip(mag, 0, 1), (0, 0), 0.6 * s)
    darker = (cv2.GaussianBlur(g, (0, 0), 2.0 * s) > g).astype(np.float32)
    out = out * (1.0 - (seam * darker * 0.35 * relief)[..., None])
    # pale outline on the strongest contours (the action's Stroke step)
    line = _smoothstep(0.6, 1.0, mag) * (1.0 - darker) * 0.35 * relief
    out = out + (255.0 - out) * line[..., None]
    return out


# --------------------------------------------------------------------------
# 4. canvas: white primed canvas and the weave
# --------------------------------------------------------------------------


def canvas_weave(h, w, s, rng):
    """Canvas texture, roughly -1..1: woven threads plus fibre grain."""
    period = max(2.5, 4.2 * s)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    jx = (_noise(h, w, 60 * s, rng) - 0.5) * period * 1.2
    jy = (_noise(h, w, 60 * s, rng) - 0.5) * period * 1.2
    wx = np.sin((xx + jx) * (2 * math.pi / period))
    wy = np.sin((yy + jy) * (2 * math.pi / (period * 1.12)))
    # over / under: one thread shows where the other dips
    weave = np.where(wx * wy > 0, wx, wy) * 0.6

    grain = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.6 * max(1.0, s))
    grain /= float(grain.std()) + 1e-6
    fibre = (_fbm(h, w, rng, (25 * s, 8 * s, 3 * s), (0.4, 0.35, 0.25)) - 0.5) * 2.0
    tex = weave * 0.55 + grain * 0.25 + fibre * 0.35
    # emboss so the threads catch the light
    tex = tex + cv2.filter2D(tex, -1, np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]],
                                               np.float32)) * 0.18
    return np.clip(tex / (float(np.abs(tex).max()) + 1e-6) * 1.6, -1.0, 1.0)


CANVAS_WHITE = np.array([238.0, 242.0, 244.0], np.float32)   # BGR, slightly warm


def _open_to_canvas(paint, amount, s, rng):
    """Let the bright passages break open to the white canvas underneath."""
    h, w = paint.shape[:2]
    lum = _lum(paint)
    blot = _fbm(h, w, rng, (90 * s, 30 * s, 10 * s), (0.5, 0.3, 0.2))
    thresh = 0.86 - 0.22 * amount + (blot - 0.5) * 0.28
    m = _smoothstep(thresh, thresh + 0.10, lum)
    # dry-brush edge: the reveal is ragged, not a soft fade
    rag = _noise(h, w, 4 * s, rng)
    m = np.clip(m * (0.75 + 0.5 * rag), 0, 1)
    m = np.where(m > 0.55, np.minimum(1.0, m * 1.15), m * 0.6)
    return paint + (CANVAS_WHITE - paint) * (m[..., None] * min(0.92, 0.30 + 0.8 * amount))


# --------------------------------------------------------------------------
# 5. scratches
# --------------------------------------------------------------------------


def _scratches(paint, amount, s, rng):
    """Cross-hatched knife scratches, in clusters, lighter than the paint."""
    h, w = paint.shape[:2]
    lum = _lum(paint)
    # where the clusters go: favour mid / dark paint near contours
    g = cv2.GaussianBlur(lum, (0, 0), 2.0 * s)
    edge = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    edge = cv2.GaussianBlur(edge, (0, 0), 12 * s)
    edge /= float(np.percentile(edge, 95)) + 1e-6
    pref = (0.25 + 0.75 * (1.0 - lum)) * (0.35 + 0.65 * np.clip(edge, 0, 1))

    light = np.zeros((h, w), np.float32)
    dark = np.zeros((h, w), np.float32)
    n_clusters = int(amount * 80 * (h * w) / 1.0e6) + 1
    tries = 0
    made = 0
    while made < n_clusters and tries < n_clusters * 20:
        tries += 1
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() > pref[cy, cx]:
            continue
        made += 1
        radius = rng.uniform(18, 55) * s
        a1 = math.radians(rng.uniform(30, 60))
        a2 = math.radians(rng.uniform(120, 150))
        for a in (a1, a2):
            gap = rng.uniform(6, 13) * s
            nlines = int(2 * radius / gap)
            ca, sa = math.cos(a), math.sin(a)
            for i in range(nlines):
                if rng.random() < 0.35:
                    continue
                off = -radius + i * gap + rng.normal(0, gap * 0.15)
                # line through the cluster, perpendicular offset ``off``
                px, py = cx - sa * off, cy + ca * off
                half = math.sqrt(max(0.0, radius * radius - off * off)) * rng.uniform(0.4, 1.0)
                if half < 4 * s:
                    continue
                shift = rng.normal(0, half * 0.3)
                x1, y1 = px + ca * (shift - half), py + sa * (shift - half)
                x2, y2 = px + ca * (shift + half), py + sa * (shift + half)
                v = float(rng.uniform(0.35, 1.0))
                target = light if rng.random() < 0.8 else dark
                cv2.line(target, (int(x1), int(y1)), (int(x2), int(y2)), v, 1, cv2.LINE_AA)
    # a few loose single scratches everywhere
    for _ in range(int(amount * 250 * (h * w) / 1.0e6)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        a = math.radians(rng.choice([45.0, 135.0]) + rng.normal(0, 8))
        L = rng.uniform(8, 26) * s
        cv2.line(light, (int(x), int(y)), (int(x + math.cos(a) * L), int(y + math.sin(a) * L)),
                 float(rng.uniform(0.3, 0.8)), 1, cv2.LINE_AA)
    if s > 1.4:
        k = np.ones((2, 2), np.uint8)
        light = cv2.dilate(light, k)
        dark = cv2.dilate(dark, k)
    light = np.clip(light, 0, 1)[..., None] * (0.25 + 0.4 * amount)
    dark = np.clip(dark, 0, 1)[..., None] * 0.35
    # the scratch shows a paler, slightly cooler version of the paint
    pale = paint * 0.40 + np.array([215.0, 218.0, 220.0], np.float32) * 0.60
    out = paint + (pale - paint) * light
    return out * (1.0 - dark * 0.5)


# --------------------------------------------------------------------------
# 6. speckles
# --------------------------------------------------------------------------


def _speckles(paint, src, amount, s, rng):
    """Small glowing paint spots: a few vivid ones, many faint pale ones."""
    h, w = paint.shape[:2]
    area = (h * w) / 1.0e6
    hsv = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    score = hsv[..., 1].astype(np.float32) * hsv[..., 2].astype(np.float32)
    top = np.argsort(score.ravel())[-3000:]
    vivid = src.reshape(-1, 3)[top]

    dots = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    lum = _lum(paint)

    # faint pale dots, mostly visible over the darker paint
    for _ in range(int(amount * 700 * area)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = max(1, int(round(rng.uniform(1.0, 2.6) * s)))
        c = np.clip(paint[y, x] * 1.6 + 60, 0, 255)
        a = float(rng.uniform(0.15, 0.45) * (0.4 + 0.6 * (1.0 - float(lum[y, x]))))
        cv2.circle(dots, (x, y), r, c.tolist(), -1, cv2.LINE_AA)
        cv2.circle(alpha, (x, y), r, a, -1, cv2.LINE_AA)

    # a few vivid ones taken from the brightest colours in the photo
    for _ in range(int(amount * 90 * area)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = max(1, int(round(rng.uniform(1.2, 3.2) * s)))
        c = vivid[rng.integers(0, len(vivid))] if rng.random() < 0.7 else CANVAS_WHITE
        c = np.clip(np.asarray(c, np.float32) * 1.1 + 25, 0, 255)
        cv2.circle(dots, (x, y), r, c.tolist(), -1, cv2.LINE_AA)
        cv2.circle(alpha, (x, y), r, float(rng.uniform(0.6, 0.95)), -1, cv2.LINE_AA)

    a = cv2.GaussianBlur(alpha, (0, 0), 0.5 * s)[..., None]
    out = paint * (1.0 - a) + dots * a
    # soft glow around each spot
    glow = cv2.GaussianBlur(dots * alpha[..., None], (0, 0), 2.5 * s)
    return out + glow * 0.45


# --------------------------------------------------------------------------
# 7. scribbles: the loose hand-drawn lines of a real painting
# --------------------------------------------------------------------------


def _wobble_path(pts, amp, rng):
    """Shake a polyline so it looks drawn by hand, not traced."""
    n = len(pts)
    if n < 2:
        return pts
    # smooth random offset along the line
    k = max(2, n // 6)
    ctrl = rng.normal(0.0, amp, (k + 1, 2))
    t = np.linspace(0, k, n)
    i = np.minimum(t.astype(int), k - 1)
    f = (t - i)[:, None]
    off = ctrl[i] * (1 - f) + ctrl[i + 1] * f
    return pts + off


def _scribbles(paint, ref, amount, s, rng):
    """Loose sketch lines over the paint - the drawing showing through.

    * contour lines: the main outlines re-drawn two or three times with a
      shaky, quick hand - overshooting the ends, broken in places, never
      quite on top of each other - like the charcoal / thin dark paint
      drawing an artist starts with;
    * hatching: bundles of quick parallel strokes in the shadows.
    """
    h, w = paint.shape[:2]
    lum = _lum(paint)
    ink = np.zeros((h, w), np.float32)
    lw = max(1, int(round(1.3 * s)))

    # --- contour lines -----------------------------------------------------
    sm = cv2.bilateralFilter(np.clip(ref, 0, 255).astype(np.uint8), 0, 30, 4 * s)
    gray = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (0, 0), 1.6 * s)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    # only the main forms: the strongest few percent of edges
    keep = 100.0 - (4.0 + 10.0 * amount)
    hi = float(np.percentile(mag, keep))
    edges = cv2.Canny(gray, int(max(4, hi * 0.25)), int(max(8, hi * 0.6)), L2gradient=True)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if len(c) > 24 * s]
    cnts.sort(key=lambda c: -len(c))
    cnts = cnts[: int(150 + 600 * amount)]
    step = max(2, int(3 * s))
    for c in cnts:
        pts = c.reshape(-1, 2).astype(np.float32)
        # a Canny contour runs there and back: keep one pass of it
        pts = pts[: max(3, len(pts) // 2 + 1)][::step]
        n = len(pts)
        if n < 4:
            continue
        for _ in range(int(rng.integers(1, 3)) + int(rng.random() < amount)):
            # a random, long piece of the contour - lines break off
            a0 = int(rng.integers(0, max(1, n // 4)))
            a1 = int(min(n, a0 + max(4, int(n * rng.uniform(0.5, 1.0)))))
            seg = pts[a0:a1]
            if len(seg) < 3:
                continue
            d0 = seg[0] - seg[min(2, len(seg) - 1)]
            d1 = seg[-1] - seg[max(-3, -len(seg))]
            seg = np.vstack([seg[0] + d0 * rng.uniform(0.5, 2.5), seg,
                             seg[-1] + d1 * rng.uniform(0.5, 2.5)])
            seg = _wobble_path(seg, 1.6 * s, rng) + rng.normal(0, 1.2 * s, 2)
            cv2.polylines(ink, [seg.astype(np.int32).reshape(-1, 1, 2)], False,
                          float(rng.uniform(0.5, 1.0)), lw, cv2.LINE_AA)

    # --- hatching in the shadows -----------------------------------------
    n_h = int(amount * 160 * (h * w) / 1.0e6)
    tries = made = 0
    while made < n_h and tries < n_h * 20:
        tries += 1
        x, y = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        if rng.random() > (1.0 - lum[int(y), int(x)]) ** 2.0:
            continue
        made += 1
        ang = math.radians(rng.uniform(30, 60))
        ca, sa = math.cos(ang), math.sin(ang)
        L = rng.uniform(14, 34) * s
        gap = rng.uniform(3, 5.5) * s
        for i in range(int(rng.integers(4, 10))):
            ox, oy = -sa * gap * i, ca * gap * i
            l2 = L * rng.uniform(0.6, 1.1)
            sh = rng.normal(0, L * 0.12)
            p1 = (x + ox + ca * (sh - l2 / 2), y + oy + sa * (sh - l2 / 2))
            p2 = (x + ox + ca * (sh + l2 / 2), y + oy + sa * (sh + l2 / 2))
            seg = _wobble_path(np.array([p1, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2), p2],
                                        np.float32), 0.8 * s, rng)
            cv2.polylines(ink, [seg.astype(np.int32).reshape(-1, 1, 2)], False,
                          float(rng.uniform(0.3, 0.7)), lw, cv2.LINE_AA)

    # dry, broken line: canvas grain eats into the strokes
    grain = _noise(h, w, 2.0 * s, rng)
    dry = 0.45 + 0.55 * _smoothstep(0.15, 0.5, grain)
    ink = np.clip(ink, 0, 1) * dry * (0.55 + 0.4 * amount)
    # drawing colour: a deep brown that picks up the paint underneath
    ink_col = paint * 0.22 + np.array([20.0, 28.0, 42.0], np.float32) * 0.78
    return paint + (ink_col - paint) * ink[..., None]


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def render(img, p, progress=None, on_stage=None, brush=None):
    """Apply the Canvas Oil look to a BGR image (uint8 or float).

    ``progress(fraction, label)`` reports progress; ``on_stage(image)``
    receives a few intermediate images for the Timeline.  ``brush(image,
    report)`` - optional - repaints the patch layer with real brush strokes
    (the engine passes its stroke painter); its result is mixed in by the
    Brush Strokes setting.
    """

    def report(frac, label):
        if progress:
            progress(frac, label)

    def stage(image):
        if on_stage:
            on_stage(np.clip(image, 0, 255))

    src = img.astype(np.float32)
    h, w = src.shape[:2]
    s = max(h, w) / 1000.0
    rng = np.random.default_rng(int(p.get("co_seed", 7)))

    patch = float(p.get("co_patch", 45)) / 100.0
    detail = float(p.get("co_detail", 55)) / 100.0
    drag = float(p.get("co_drag", 40)) / 100.0
    dabs = float(p.get("co_dabs", 55)) / 100.0
    relief = float(p.get("co_relief", 45)) / 100.0
    canvas = float(p.get("co_canvas", 45)) / 100.0
    scratch = float(p.get("co_scratches", 20)) / 100.0
    speck = float(p.get("co_speckles", 40)) / 100.0
    brushwork = float(p.get("co_brush", 70)) / 100.0
    sketch = float(p.get("co_sketch", 45)) / 100.0
    contrast = float(p.get("co_contrast", 35)) / 100.0
    vib = float(p.get("co_vibrance", 40)) / 100.0

    report(0.08, "Levels and vibrance")
    toned = _tone(src, contrast, vib)

    report(0.18, "Breaking the photo into paint daubs")
    big_px = (10.0 + 50.0 * patch) * s
    paint = _patch_layer(toned, big_px, rng, drag)
    if detail > 0.01:
        report(0.36, "Painting the detail")
        fine = _patch_layer(toned, max(4.0 * s, big_px * 0.32), rng, drag * 0.6)
        dm = _detail_map(toned)
        # patchy hand-over between the two brush sizes, not a soft fade
        jitter = (_noise(h, w, big_px * 1.5, rng) - 0.5) * 0.5
        m = _smoothstep(0.62 - 0.5 * detail, 0.78 - 0.5 * detail, dm + jitter)
        paint = paint + (fine - paint) * m[..., None]
    stage(paint)

    if dabs > 0.01:
        report(0.50, "Knife dabs")
        flatness = 1.0 - _detail_map(paint) * 0.85
        paint = _dab_cells(paint, (6.0 + 10.0 * patch) * s, dabs, rng, drag, flatness)

    # wet paint runs: faint vertical streaks, strongest in the dark paint
    if drag > 0.01:
        streak = _noise(h, w, 3.0 * s, rng)
        streak = cv2.blur(streak, (1, max(3, int(40 * s)) | 1))
        streak = cv2.normalize(streak, None, -1.0, 1.0, cv2.NORM_MINMAX)
        dark = 1.0 - _lum(paint)
        paint = paint * (1.0 + streak[..., None] * (0.10 * drag) * dark[..., None])

    if brush is not None and brushwork > 0.01:
        # paint the patches over with real, bristled brush strokes so the
        # picture reads as paint laid on with a brush, not as a cut-out
        def sub(frac, label):
            report(0.52 + 0.30 * frac, label)
        stroked = brush(paint, sub)
        # strokes carry their own lit relief; keep the patch rims lighter
        paint = paint + (stroked - paint) * min(1.0, brushwork * 1.1)
        relief *= (1.0 - 0.55 * brushwork)
        stage(paint)
        base = 0.82
    else:
        base = 0.52

    def at(frac):
        return base + (1.0 - base) * frac

    report(at(0.1), "Paint rims")
    paint = _rims(paint, relief, s)

    # mottled pigment and dry-brush grain dragged down the picture
    mott = _fbm(h, w, rng, (40 * s, 12 * s, 4 * s), (0.45, 0.35, 0.2)) - 0.5
    grain = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.6 * s, sigmaY=(2.0 + 5.0 * drag) * s)
    grain /= float(grain.std()) + 1e-6
    paint = paint * (1.0 + (mott * 0.14 + grain * 0.035)[..., None])

    report(at(0.3), "Opening up the canvas")
    if canvas > 0.01:
        paint = _open_to_canvas(paint, canvas, s, rng)
    stage(paint)

    if sketch > 0.01:
        report(at(0.4), "Sketch lines")
        paint = _scribbles(paint, toned, sketch, s, rng)

    report(at(0.5), "Knife scratches")
    if scratch > 0.01:
        paint = _scratches(paint, scratch, s, rng)

    report(at(0.7), "Speckles")
    if speck > 0.01:
        paint = _speckles(paint, toned, speck, s, rng)

    report(at(0.85), "Canvas texture")
    if canvas > 0.01:
        tex = canvas_weave(h, w, s, rng)
        lum = _lum(paint)
        # the weave shows most in the light, thinly painted passages
        k = (0.04 + 0.12 * lum) * (0.4 + 0.9 * canvas)
        paint = paint * (1.0 + tex[..., None] * k[..., None])

    # a light unsharp mask, as after the action's final Levels
    blur = cv2.GaussianBlur(paint, (0, 0), 1.2 * s)
    paint = paint + (paint - blur) * 0.35
    report(1.0, "Done")
    return np.clip(paint, 0, 255)
