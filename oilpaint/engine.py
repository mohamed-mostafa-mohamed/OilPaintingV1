"""
Oil painting engine.

The strokes are planned by ``planner.py`` and painted here, on the CPU, by
:class:`Brush` (the live preview paints the same plan on the GPU, see
static/js/live.js).

The pipeline reproduces the way an oil painting is actually built up:

  1. the photograph is simplified into flat, painterly colour masses;
  2. a structure tensor gives a stroke-orientation field, so strokes run
     *along* contours (hair, petals, waves) instead of across them;
  3. the painting is built in coarse-to-fine layers - big blocking-in
     strokes first, finer strokes only where the canvas still differs
     from the reference (a painterly-rendering error loop);
  4. every stroke is drawn as a real brush: a bundle of bristles with
     their own colour and height, tapered ends, wet-paint mixing;
  5. the accumulated paint height is lit as an impasto relief, giving
     the ridges and gloss of thick oil paint;
  6. optional white scumbling, edge framing strokes, blending with the
     original and decoration.

Planning is vectorised with numpy / OpenCV.  A stroke is composited
into its own bounding box only, so tens of thousands of strokes stay
fast.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from . import abstract as ab
from . import canvas_oil
from . import halftone
from . import painted
from . import impression
from . import oilfx
from . import sketchfx
from . import brushfx
from . import watercolor
from . import silkoil
from . import smear
from .params import hex_to_bgr

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def small_side(a) -> float:
    return float(min(a.shape[0], a.shape[1]))


def lerp(a, b, t):
    return a + (b - a) * t


def simplify(img: np.ndarray, simplicity: int, scale_boost: float = 1.0) -> np.ndarray:
    """Edge-preserving abstraction: flattens detail, keeps contours.

    ``simplicity`` 1..15 - the higher, the more the photo turns into flat
    painterly colour masses.  ``scale_boost`` < 1 keeps more detail and is
    used for the reference of the finishing layers, so the painting can be
    refined back towards the real shapes.
    """
    s = float(simplicity) * float(scale_boost)
    h, w = img.shape[:2]
    scale = np.clip(max(h, w) / 900.0, 0.55, 1.8)

    u8 = np.clip(img, 0, 255).astype(np.uint8)

    # Edge preserving filter: large flat areas, crisp boundaries.
    sigma_s = float(np.clip(4.0 + s * 4.2, 4, 70)) * scale
    sigma_r = float(np.clip(0.05 + s * 0.017, 0.04, 0.42))
    u8 = cv2.edgePreservingFilter(u8, flags=cv2.RECURS_FILTER,
                                  sigma_s=sigma_s, sigma_r=sigma_r)

    # One or two bilateral passes deepen the flattening without smearing the
    # contours - this is what makes it read as "painted", not "blurred".
    rounds = int(np.clip(s / 5.0, 1, 3))
    d = int(np.clip(5 + s * 0.45, 5, 11))
    for _ in range(rounds):
        u8 = cv2.bilateralFilter(u8, d, 14 + s * 4.0, 5 + s * 1.6)

    out = u8.astype(np.float32)

    # A painter mixes a limited number of tones: posterise the strongest
    # settings only, and keep the contours sharp afterwards.
    if s >= 9.5:
        levels = float(np.clip(34 - s, 19, 26))
        q = np.round(out / (255.0 / levels)) * (255.0 / levels)
        out = out + (q - out) * float(np.clip((s - 9.5) / 5.5, 0, 1))

    return np.clip(out, 0, 255)


def structure_tensor(gray: np.ndarray) -> dict:
    """The part of the orientation analysis that does not depend on the
    Direction settings (so the live preview can cache it).

    The direction is the *minor* eigenvector of the smoothed structure
    tensor, i.e. the direction along which the image varies least - the
    direction a painter's brush naturally follows.
    """
    h, w = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)

    sigma = max(1.5, min(h, w) / 180.0 * 2.0)
    j11 = ab.gblur(gx * gx, sigma)
    j22 = ab.gblur(gy * gy, sigma)
    j12 = ab.gblur(gx * gy, sigma)

    # eigen decomposition of the 2x2 symmetric tensor
    tmp = np.sqrt(np.maximum((j11 - j22) ** 2 + 4.0 * j12 * j12, 0.0))
    lam1 = 0.5 * (j11 + j22 + tmp)          # major
    lam2 = 0.5 * (j11 + j22 - tmp)          # minor
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-6)

    # major eigenvector (across the edge)
    ex = j12
    ey = lam1 - j11
    norm = np.sqrt(ex * ex + ey * ey) + 1e-9
    ex, ey = ex / norm, ey / norm
    # rotate 90 deg -> along the edge
    dx, dy = -ey, ex

    degenerate = norm < 1e-6
    if degenerate.any():
        dx = np.where(degenerate, 1.0, dx)
        dy = np.where(degenerate, 0.0, dy)

    # How strong the local structure actually is.  Coherence alone is a
    # trap: in a flat sky the tensor still reports a confident direction,
    # because it is describing sensor noise.  Following it produces the
    # wriggling "worms" that give cheap painting filters away.
    gm = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.0)
    gmax = float(np.percentile(gm, 86)) + 1e-6
    grad = np.clip(gm / gmax, 0.0, 1.0).astype(np.float32)

    coherence = coherence.astype(np.float32)
    # where the image has no structure the global sweep takes over; this
    # part of the weight does not depend on the Direction slider
    w_struct = (np.clip(coherence * 2.4, 0.0, 1.0)
                * np.clip(grad * 3.2, 0.0, 1.0)).astype(np.float32)
    return {"dx": dx.astype(np.float32), "dy": dy.astype(np.float32),
            "coherence": coherence, "grad": grad, "w_struct": w_struct}


def direction_field(t: dict, p: dict, guides=None):
    """Blend the local stroke directions with the common direction and the
    user's guiding lines.  Cheap - re-run whenever Direction changes."""
    dx, dy = t["dx"], t["dy"]
    h, w = dx.shape
    # direction = 0   -> strokes hug the edges
    # direction = 100 -> strokes all sweep the same way
    k = float(p["direction"]) / 100.0
    ang = math.radians(float(p["global_angle"]))
    gxv, gyv = math.cos(ang), math.sin(ang)
    w_local = (1.0 - k) * t["w_struct"]
    # keep the local vectors pointing consistently with the global one so
    # that blending two opposite-signed vectors does not cancel out
    flip = np.sign(dx * gxv + dy * gyv)
    flip[flip == 0] = 1.0
    dx, dy = dx * flip, dy * flip

    bx = dx * w_local + gxv * (1.0 - w_local)
    by = dy * w_local + gyv * (1.0 - w_local)

    # --- guiding lines from the Stroke Direction tool ---------------------
    if guides:
        gux, guy, gw = _guide_field(guides, h, w)
        bx = bx * (1.0 - gw) + gux * gw
        by = by * (1.0 - gw) + guy * gw

    n = np.sqrt(bx * bx + by * by) + 1e-9
    return (bx / n).astype(np.float32), (by / n).astype(np.float32)


def orientation_field(gray: np.ndarray, p: dict, guides=None):
    """Return (dirx, diry, coherence, gradient_strength)."""
    t = structure_tensor(gray)
    bx, by = direction_field(t, p, guides)
    return bx, by, t["coherence"], t["grad"]


def _guide_field(guides, h, w):
    """Rasterise user guiding lines into a dense direction field."""
    vx = np.zeros((h, w), np.float32)
    vy = np.zeros((h, w), np.float32)
    wt = np.zeros((h, w), np.float32)
    radius = max(6, int(min(h, w) * 0.05))
    for line in guides:
        pts = np.array([[x * (w - 1), y * (h - 1)] for x, y in line], np.float32)
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            d = b - a
            n = float(np.hypot(*d))
            if n < 1e-3:
                continue
            d = d / n
            p0 = (int(round(a[0])), int(round(a[1])))
            p1 = (int(round(b[0])), int(round(b[1])))
            cv2.line(vx, p0, p1, float(d[0]), radius)
            cv2.line(vy, p0, p1, float(d[1]), radius)
            cv2.line(wt, p0, p1, 1.0, radius)
    if wt.max() <= 0:
        return vx, vy, wt
    sigma = radius * 0.9
    vx = cv2.GaussianBlur(vx, (0, 0), sigma)
    vy = cv2.GaussianBlur(vy, (0, 0), sigma)
    wt = cv2.GaussianBlur(wt, (0, 0), sigma)
    n = np.sqrt(vx * vx + vy * vy) + 1e-9
    vx, vy = vx / n, vy / n
    wt = np.clip(wt * 1.8, 0.0, 1.0)
    return vx.astype(np.float32), vy.astype(np.float32), wt.astype(np.float32)


# --------------------------------------------------------------------------
# the brush
# --------------------------------------------------------------------------

_SHIFT = 3
_SC = 1 << _SHIFT


class Brush:
    """Composites bristle brush strokes into a canvas + paint height map."""

    __slots__ = ("canvas", "height", "coverage", "h", "w")

    def __init__(self, canvas, height, coverage):
        self.canvas = canvas
        self.height = height
        self.coverage = coverage
        self.h, self.w = canvas.shape[:2]

    def stroke(self, pts, color, thickness, hair_px, bristles, softness,
               opacity, height_amt, shade_var, rng, flat=False,
               reserve=1.0, mixing=True):
        """Lay down one brush stroke.

        The stroke is an opaque body of paint whose width tapers towards the
        ends (a round brush) or stays square (a flat brush / palette knife).
        The bristles do not cut holes in that body - they modulate its
        colour and its height, which is what gives oil paint its streaked,
        ridged surface.

        pts        (N,2) float path in image coordinates
        thickness  total brush width in px
        hair_px    width of a single bristle
        bristles   how many bristles the brush carries
        softness   gaussian sigma applied to the stroke alpha
        opacity    0..1; below 1 the paint underneath mixes through
        height_amt how much paint this stroke lays down
        shade_var  bristle-to-bristle brightness variation
        flat       flat brush / palette knife instead of a round one
        reserve    fraction of the stroke drawn before the paint runs out
        """
        pts = np.asarray(pts, np.float32)
        if pts.ndim != 2 or len(pts) < 2:
            return False

        half = max(0.5, thickness * 0.5)
        pad = int(math.ceil(half + hair_px + 3.0 * max(1.0, softness) + 2))

        x0 = max(0, int(math.floor(pts[:, 0].min())) - pad)
        y0 = max(0, int(math.floor(pts[:, 1].min())) - pad)
        x1 = min(self.w, int(math.ceil(pts[:, 0].max())) + pad)
        y1 = min(self.h, int(math.ceil(pts[:, 1].max())) + pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return False

        lh, lw = y1 - y0, x1 - x0
        local = pts - np.array([x0, y0], np.float32)
        npts = len(local)

        # tangents / normals along the path
        if npts >= 3:
            d = np.gradient(local, axis=0)
        else:
            d = np.repeat((local[-1] - local[0])[None, :], npts, axis=0)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)

        alpha = np.zeros((lh, lw), np.uint8)
        shade = np.zeros((lh, lw), np.uint8)

        ip = np.empty((npts, 1, 2), np.int32)
        ip[:, 0, :] = np.round(local * _SC)

        # paint reserve: the brush runs dry towards the end of the stroke
        keep = npts if reserve >= 0.999 else max(2, int(npts * reserve))

        # --- body of paint, tapered towards the ends ---------------------
        if flat or keep < 6:
            cv2.polylines(alpha, [ip[:keep]], False, 255,
                          max(1, int(round(thickness))), cv2.LINE_AA, shift=_SHIFT)
        else:
            k = 7
            bounds = np.linspace(0, keep - 1, k + 1).astype(int)
            for i in range(k):
                a, b = bounds[i], bounds[i + 1] + 1
                if b - a < 2:
                    continue
                t = (i + 0.5) / k
                prof = 0.42 + 0.58 * math.sin(math.pi * t) ** 0.45
                cv2.polylines(alpha, [ip[a:b]], False, 255,
                              max(1, int(round(thickness * prof))),
                              cv2.LINE_AA, shift=_SHIFT)
        cv2.polylines(shade, [ip[:keep]], False, 128,
                      max(1, int(round(thickness + 2))), cv2.LINE_8, shift=_SHIFT)

        # --- bristle striations: colour and height, not coverage ---------
        # bright paint shows bristle streaks far more harshly than dark
        # paint does, so damp the variation as the colour nears white
        lum = float(np.mean(color)) / 255.0
        damp = float(np.clip(1.15 - 0.70 * lum ** 1.6, 0.35, 1.15))

        nb = max(1, int(bristles))
        offs = np.linspace(-half, half, nb) if nb > 1 else np.zeros(1, np.float32)
        offs = offs + rng.uniform(-hair_px * 0.3, hair_px * 0.3, nb)
        hp = max(1, int(round(hair_px)))

        for o in offs:
            if flat:
                a, b = 0, keep
            else:
                trim = int(round((abs(o) / max(half, 1e-6)) ** 2 *
                                 min(npts * 0.28, max(1.0, thickness * 0.5))))
                a, b = trim, max(trim + 2, keep - trim)
            if b - a < 2:
                continue
            seg = local[a:b] + nrm[a:b] * float(o)
            sp = np.empty((len(seg), 1, 2), np.int32)
            sp[:, 0, :] = np.round(seg * _SC)
            v = int(np.clip(128 * (1.0 + rng.uniform(-shade_var, shade_var) * damp),
                            10, 255))
            cv2.polylines(shade, [sp], False, v, hp, cv2.LINE_8, shift=_SHIFT)

        a = alpha.astype(np.float32) * (1.0 / 255.0)
        if softness > 0.05:
            a = cv2.GaussianBlur(a, (0, 0), float(softness))
        if reserve < 0.999:
            a *= 0.88          # a dry brush leaves a faded, scratchy deposit

        a *= float(opacity)
        if a.max() < 0.004:
            return False

        sh = np.where(shade > 0, shade.astype(np.float32) * (1.0 / 128.0), 1.0)
        col = np.asarray(color, np.float32)[None, None, :] * sh[:, :, None]

        roi = self.canvas[y0:y1, x0:x1]
        if mixing:
            roi += (col - roi) * a[:, :, None]
        else:
            # dry paint: it sits on top instead of blending in
            m = (a > 0.35)[:, :, None]
            np.copyto(roi, col, where=m)

        # paint volume: the bristle grooves ride lower than the ridges
        self.height[y0:y1, x0:x1] += a * float(height_amt) * (0.55 + 0.75 * sh)
        cov = self.coverage[y0:y1, x0:x1]
        np.maximum(cov, a, out=cov)
        return True


# --------------------------------------------------------------------------
# main painting routine
# --------------------------------------------------------------------------


def paint(image_bgr: np.ndarray, p: dict, progress=None, snapshots: int = 8,
          on_snapshot=None, should_stop=None, cache=None, token=None):
    """Convert a photo into an oil painting.

    ``progress(fraction, label)``  - optional progress callback
    ``on_snapshot(index, image)``  - receives Timeline snapshots
    ``should_stop()``              - return True to interrupt (Stop button)
    ``cache`` / ``token``          - optional planner.AnalysisCache to share
                                     the image analysis with the live preview

    The strokes are planned by ``planner`` one layer at a time and painted
    here with :class:`Brush`; each finishing layer is planned against the
    canvas as it really stands.  ``meta`` carries the plan, the bare canvas
    and the paint height so the workspace can re-light and re-blend the
    painting, and preview it on the GPU, without painting it again.
    """
    from . import planner as pl

    from .params import SEED_KEYS
    style = p.get("paint_style")
    if style in SEED_KEYS and int(p.get("samples", 4)) > 1 and snapshots > 0:
        return _paint_variations(image_bgr, p, progress, on_snapshot, should_stop)
    if style in ("Canvas Oil", "Halftone", "Painted Photo", "Impression", "Oil Smear", "Oil Paint FX", "Sketch Paint",
                 "Rough Brush", "Watercolor", "Silk Oil"):
        return _paint_halftone(image_bgr, p, progress, on_snapshot)

    t_start = time.time()
    src = image_bgr.astype(np.float32)
    h, w = src.shape[:2]
    raster_rng = np.random.default_rng(int(p["random_seed"]) + 7919)

    def report(frac, label):
        if progress:
            progress(max(0.0, min(1.0, frac)), label)

    # --- plan every stroke: brushplan, the same steps the GPU runs ----------
    from . import brushplan as BP
    report(0.04, "Analysing the photograph")
    work = ab.distort(src, p) if pl.distortion_key(p) is not None else src
    bcache = cache.bp if isinstance(cache, pl.AnalysisCache) and cache.bp is not None else None
    plan = BP.plan_bgr(work, p, side=BP.RUN_SIDE, cache=bcache, token=token,
                       report=lambda f, s: report(0.04 + 0.14 * f, s))
    if should_stop and should_stop():
        raise PaintStopped()

    # --- paint them ----------------------------------------------------------
    canvas = plan.ground.astype(np.float32).copy()
    if pl.correction_active(p):
        canvas = ab.correct(canvas, p)
    height = np.zeros((h, w), np.float32)
    coverage = np.zeros((h, w), np.float32)
    brush = Brush(canvas, height, coverage)
    st, pts, offs = plan.strokes, plan.pts, plan.offs
    n = plan.count
    sty = pl.stroke_style(st, p, (w, h)) if n else None
    shots = []
    milestones = [(i + 1) / (snapshots + 1) for i in range(snapshots)]
    state = {"done": 0, "stopped": False}
    names = {pl.KIND_LAYER: "Painting the layers", pl.KIND_LAST: "Painting the details",
             pl.KIND_FILL: "Filling empty spaces", pl.KIND_EDGE: "Edge strokes"}
    for i in range(n):
        if (i & 255) == 0:
            if should_stop and should_stop():
                state["stopped"] = True
                break
            report(0.18 + 0.72 * i / n, names.get(int(st[i, pl.C_KIND]), "Painting"))
        while milestones and i / n >= milestones[0]:
            milestones.pop(0)
            shots.append(_present(canvas, height, p, src, quick=True))
            if on_snapshot:
                on_snapshot(len(shots) - 1, shots[-1])
        ok = brush.stroke(pts[offs[i]:offs[i + 1]], sty["colour"][i], float(st[i, pl.C_T]),
                          float(sty["hair"][i]), int(sty["nb"][i]),
                          softness=float(sty["soft"][i]), opacity=float(sty["opacity"][i]),
                          height_amt=float(sty["height"][i]),
                          shade_var=float(sty["shade_var"][i]), rng=raster_rng,
                          flat=bool(sty["flat"][i]))
        if ok:
            state["done"] += 1

    report(0.92, "Drying the paint")
    while milestones and not state["stopped"]:
        milestones.pop(0)
        shots.append(_present(canvas, height, p, src, quick=True))
        if on_snapshot:
            on_snapshot(len(shots) - 1, shots[-1])

    result = _present(canvas, height, p, src, quick=False)
    shots.append(result)
    if on_snapshot:
        on_snapshot(len(shots) - 1, result)

    report(1.0, "Done")
    meta = {
        "strokes": state["done"],
        "seconds": round(time.time() - t_start, 2),
        "stopped": state["stopped"],
        "size": [w, h],
        "plan": plan,
        "canvas": canvas,
        "height": height,
        "finish": ("relief", "lightening", "blend"),
    }
    return result, shots, meta


def render_plan(plan, p: dict, original: np.ndarray | None = None, seed: int = 0):
    """Paint a finished plan on the CPU (no planning).  Returns
    (canvas, height); used by tests and by the fallback preview."""
    from . import planner as pl
    w, h = plan.size
    canvas = plan.ground.astype(np.float32).copy()
    if pl.correction_active(p):
        canvas = ab.correct(canvas, p)
    height = np.zeros((h, w), np.float32)
    coverage = np.zeros((h, w), np.float32)
    brush = Brush(canvas, height, coverage)
    rng = np.random.default_rng(seed)
    st = plan.strokes
    sty = pl.stroke_style(st, p, (w, h))
    for i in range(len(st)):
        brush.stroke(plan.pts[plan.offs[i]:plan.offs[i + 1]], sty["colour"][i],
                     float(st[i, pl.C_T]), float(sty["hair"][i]), int(sty["nb"][i]),
                     softness=float(sty["soft"][i]), opacity=float(sty["opacity"][i]),
                     height_amt=float(sty["height"][i]), shade_var=float(sty["shade_var"][i]),
                     rng=rng, flat=bool(sty["flat"][i]))
    return canvas, height


def draft_strokes(plan, p: dict) -> np.ndarray:
    """A quick sketch of a plan: every stroke as one solid anti-aliased line
    in its paint colour (no bristles, no relief)."""
    from . import planner as pl
    canvas = plan.ground.astype(np.float32).copy()
    if not plan.count:
        return canvas
    col = pl.stroke_colours(plan.strokes, p)
    t = np.maximum(1, np.round(plan.strokes[:, pl.C_T] * 0.85)).astype(np.int32)
    ip = np.round(plan.pts * 8).astype(np.int32).reshape(-1, 1, 2)
    offs = plan.offs
    for i in range(plan.count):
        cv2.polylines(canvas, [ip[offs[i]:offs[i + 1]]], False,
                      (float(col[i, 0]), float(col[i, 1]), float(col[i, 2])),
                      int(t[i]), cv2.LINE_AA, shift=3)
    return canvas


def paint_draft(image_bgr: np.ndarray, p: dict, cache=None, token=None):
    """A fast, approximate painting for live previews on the server.

    The filter styles: made at the preview size.  Brush Strokes: the preview plan
    sketched the same way.  Returns the finished (bare) image.
    """
    from . import planner as pl
    if p.get("paint_style") in ("Canvas Oil", "Halftone", "Painted Photo", "Impression", "Oil Smear", "Oil Paint FX",
                                "Sketch Paint", "Rough Brush", "Watercolor", "Silk Oil"):
        return _halftone_bare(image_bgr, p, draft=True)
    plan = pl.plan_preview(image_bgr, p, cache, token)
    return draft_strokes(plan, p)


class PaintStopped(Exception):
    """Stop was pressed before any variation was finished."""


def _variation_workers(n):
    """How many variations to paint at once. numpy / OpenCV free the
    interpreter while they work, so several variations run side by side on
    a multi-core CPU. Override with OILSTUDIO_WORKERS (1 = one at a time)."""
    env = os.environ.get("OILSTUDIO_WORKERS")
    if env and env.isdigit():
        return max(1, min(n, int(env)))
    cores = os.cpu_count() or 2
    return max(1, min(n, cores // 2 if cores >= 4 else cores))


# ---- Brush Strokes variations in separate processes -------------------------
_proc_q = None      # set in each worker process: progress messages go back here
_proc_stop = None   # set in each worker process: Stop pressed


def _proc_init(q, stop):
    global _proc_q, _proc_stop
    _proc_q, _proc_stop = q, stop
    try:
        cv2.setNumThreads(1)          # the processes already use every core
    except Exception:                 # noqa: BLE001
        pass


def _proc_variation(i, image_bgr, q):
    """One Brush Strokes variation, in a worker process."""
    def report(f, label):
        if _proc_stop is not None and _proc_stop.is_set():
            raise PaintStopped()
        if _proc_q is not None:
            _proc_q.put((i, float(f), label))
    try:
        result, _, meta = paint(image_bgr, dict(q, samples=1), progress=report, snapshots=0)
    except PaintStopped:
        return i, None, None
    meta = {k: v for k, v in meta.items() if k != "plan"}
    return i, result, meta


def _process_pool(workers):
    """A pool of worker processes, or None where processes cannot be used."""
    if os.environ.get("OILSTUDIO_NO_PROCESSES"):
        return None
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        from concurrent.futures import ProcessPoolExecutor
        q, stop = ctx.Queue(), ctx.Event()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                   initializer=_proc_init, initargs=(q, stop))
        return pool, q, stop
    except Exception:                 # noqa: BLE001 - fall back to threads
        return None


def kill_pool(pool):
    """End a process pool's workers now (Stop), not after their step."""
    try:
        for pr in list(getattr(pool, "_processes", {}).values()):
            pr.terminate()
    except Exception:                 # noqa: BLE001
        pass


def _paint_variations_proc(image_bgr, p, progress, on_snapshot, should_stop, key, n, base):
    """Brush Strokes: the variations side by side in worker processes."""
    import queue as _queue
    from concurrent.futures.process import BrokenProcessPool
    workers = _variation_workers(n)
    made = _process_pool(workers) if workers > 1 else None
    if made is None:
        return None
    pool, q, stop = made
    t_start = time.time()
    frac = [0.0] * n
    results, done_order, meta = {}, [], None
    jobs = []
    for i in range(n - 1, -1, -1):              # the chosen one (last) starts first
        qq = dict(p)
        qq[key] = (base + i + 1) % 10000 if i < n - 1 else base
        jobs.append(pool.submit(_proc_variation, i, image_bgr, qq))
    try:
        pending = set(jobs)
        while pending:
            # progress from the workers
            try:
                while True:
                    i, f, label = q.get_nowait()
                    frac[i] = max(frac[i], f)
                    if progress:
                        done = sum(1 for x in frac if x >= 1.0)
                        progress(sum(frac) / n, f"{label} — {done} of {n} variations done")
            except _queue.Empty:
                pass
            except (EOFError, OSError):
                pass
            if should_stop and should_stop() and not stop.is_set():
                stop.set()
                for f in pending:
                    f.cancel()
                kill_pool(pool)                 # free the CPU at once
                break
            finished = [f for f in pending if f.done()]
            for fut in finished:
                pending.discard(fut)
                if fut.cancelled():
                    continue
                try:
                    i, result, m = fut.result()  # a real error is raised here
                except BrokenProcessPool:
                    if stop.is_set():
                        continue
                    # worker processes could not start (some packaged builds):
                    # paint the old way, one variation after another
                    if not results:
                        return None
                    raise
                if result is None:
                    continue
                frac[i] = 1.0
                results[i] = result
                if meta is None or i == n - 1:
                    meta = m
                if i != n - 1:
                    done_order.append(i)
                    if on_snapshot:
                        on_snapshot(len(done_order) - 1, result)
            if pending and not finished:
                time.sleep(0.1)
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
    if not results:
        raise PaintStopped()
    stopped = len(results) < n
    if (n - 1) in results:
        chosen = n - 1
        if on_snapshot:
            on_snapshot(len(done_order), results[chosen])
    else:
        chosen = done_order[-1]
    shots = [results[k] for k in done_order if k != chosen] + [results[chosen]]
    meta = dict(meta)
    meta["seconds"] = round(time.time() - t_start, 2)
    meta["stopped"] = stopped
    return results[chosen], shots, meta


def _paint_variations(image_bgr, p, progress, on_snapshot, should_stop):
    """The template styles: several variations of the same settings, each
    with its own random seed (new strokes, smears, grain...), shown in the
    Timeline as they finish. The chosen seed is the last one (always shown
    last), so the result after a full Run is exactly the settings on screen.

    The variations are painted side by side on the CPU cores, and Stop is
    checked inside every progress step - not only after a whole variation -
    so a stopped Run frees the CPU at once."""
    from .params import SEED_KEYS
    t_start = time.time()
    style = p.get("paint_style")
    key = SEED_KEYS[style]
    n = int(np.clip(int(p.get("samples", 4)), 1, 16))
    base = int(p.get(key, 1))
    if style == "Brush Strokes" and n > 1:
        out = _paint_variations_proc(image_bgr, p, progress, on_snapshot, should_stop, key, n, base)
        if out is not None:
            return out
    stop_flag = threading.Event()
    lock = threading.Lock()
    frac = [0.0] * n
    last_label = [""]

    def stopping():
        if stop_flag.is_set():
            return True
        if should_stop and should_stop():
            stop_flag.set()
            return True
        return False

    def one(i):
        q = dict(p)
        q[key] = (base + i + 1) % 10000 if i < n - 1 else base

        def report(f, label):
            if stopping():
                raise PaintStopped()
            with lock:
                frac[i] = max(frac[i], float(f))
                done = sum(1 for x in frac if x >= 1.0)
                last_label[0] = label
                total = sum(frac) / n
            if progress:
                progress(total, f"{label} — {done} of {n} variations done")
        if stopping():
            raise PaintStopped()
        if style == "Brush Strokes":
            result, _, meta = paint(image_bgr, dict(q, samples=1), progress=report, snapshots=0)
        else:
            result, _, meta = _paint_halftone(image_bgr, q, report, None)
        with lock:
            frac[i] = 1.0
        return i, result, meta

    results, done_order = {}, []
    meta = None
    order = list(range(n - 1, -1, -1))        # the chosen one (last) starts first
    # Brush Strokes only lands here when processes are not available: its stroke
    # loop is pure Python, so threads would only wait on each other
    with ThreadPoolExecutor(max_workers=1 if style == "Brush Strokes" else _variation_workers(n),
                            thread_name_prefix="variation") as pool:
        futs = [pool.submit(one, i) for i in order]
        for fut in as_completed(futs):
            try:
                i, result, m = fut.result()
            except PaintStopped:
                continue
            except Exception:
                stop_flag.set()                 # a real error: stop the rest too
                for f in futs:
                    f.cancel()
                raise
            results[i] = result
            if meta is None or i == n - 1:
                meta = m
            if i != n - 1:                      # extra variations go to the Timeline as they finish
                done_order.append(i)
                if on_snapshot:
                    on_snapshot(len(done_order) - 1, result)
            if stop_flag.is_set():
                for f in futs:
                    f.cancel()
    if not results:
        raise PaintStopped()
    stopped = len(results) < n
    if (n - 1) in results:                      # the settings on screen: shown last, used
        chosen = n - 1
        if on_snapshot:
            on_snapshot(len(done_order), results[chosen])
    else:                                       # stopped early: the last one that finished
        chosen = done_order[-1]
    shots = [results[k] for k in done_order if k != chosen] + [results[chosen]]
    meta = dict(meta)
    meta["seconds"] = round(time.time() - t_start, 2)
    meta["stopped"] = stopped
    return results[chosen], shots, meta


def _halftone_bare(image_bgr, p, progress=None, draft=False):
    src = image_bgr.astype(np.float32)
    work = src
    if p["abstract_enabled"]:
        if p["dist_strength"] > 0:
            work = ab.distort(work, p)
        work = ab.correct(work, p)
    mod = {"Canvas Oil": canvas_oil, "Painted Photo": painted, "Impression": impression, "Oil Smear": smear,
           "Oil Paint FX": oilfx, "Sketch Paint": sketchfx,
           "Rough Brush": brushfx, "Watercolor": watercolor, "Silk Oil": silkoil}.get(p.get("paint_style"), halftone)
    if draft:      # live preview: made at the preview size itself (fast)
        out = mod.render(work, p, progress=progress, min_side=0)
    else:
        out = mod.render(work, p, progress=progress)
    return np.clip(out, 0, 255).astype(np.float32)


def _paint_halftone(image_bgr, p, progress, on_snapshot):
    """The "Halftone" style (see halftone.py): a vintage halftone print,
    with Abstract Art and Blend with Original like the other styles."""
    t_start = time.time()
    src = image_bgr.astype(np.float32)
    bare = _halftone_bare(image_bgr, p, progress)
    result = bare
    blend = float(p["blend_original"]) / 100.0
    if blend > 0.002:
        result = blend_images(result, src, blend, p.get("blend_mode", "Normal"))
    result = np.clip(result, 0, 255)
    if on_snapshot:
        on_snapshot(0, result)
    h, w = src.shape[:2]
    return result, [result], {"strokes": 0, "seconds": round(time.time() - t_start, 2),
                              "stopped": False, "size": [w, h], "style": p.get("paint_style"),
                              "canvas": bare, "height": np.zeros((h, w), np.float32),
                              "finish": ("blend",)}


# --------------------------------------------------------------------------
# presentation: relief lighting, lightening, blending
# --------------------------------------------------------------------------


FINISH_ALL = ("relief", "lightening", "blend")


def finish(canvas, height, p, original, steps=FINISH_ALL, quick=False):
    """The finishing applied on top of the bare painting: impasto relief,
    white scumbling (Lightening) and Blend with Original.

    These need no repainting, so the workspace redoes them whenever the
    sliders move.  ``steps`` lists the ones that apply (a Canvas Oil
    painting lights its own relief, a Timeline shot is already finished).
    """
    out = canvas.copy()
    relief = float(p["relief"]) / 100.0
    if "relief" in steps and relief > 0.005 and height is not None:
        out = apply_relief(out, height, relief, quick=quick)

    light = float(p["lightening"]) / 100.0
    if "lightening" in steps and light > 0.005:
        lum = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask = np.clip((lum - 0.42) / 0.58, 0, 1) ** 1.4
        out += (255.0 - out) * (mask[:, :, None] * light * 0.85)

    blend = float(p["blend_original"]) / 100.0
    if "blend" in steps and blend > 0.002 and original is not None:
        out = blend_images(out, original, blend, p.get("blend_mode", "Normal"))

    return np.clip(out, 0, 255)


def _present(canvas, height, p, original, quick=False):
    return finish(canvas, height, p, original, FINISH_ALL, quick=quick)


def apply_relief(img, height, strength, quick=False):
    """Light the accumulated paint height as an impasto surface.

    The response has to be symmetric: a ridge is lit on the side facing the
    light and shadowed on the other.  A formulation that can only darken
    turns the whole painting grey, which is exactly what thick paint does
    not do.
    """
    if strength <= 0.0:
        return img

    hgt = cv2.GaussianBlur(height, (0, 0), 0.8)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3)

    # key light from the upper left, the way a painting is normally hung
    lx, ly = -0.7071, -0.7071
    emb = gx * lx + gy * ly

    scale = float(np.percentile(np.abs(emb), 97.0)) + 1e-6
    emb = np.clip(emb / scale, -1.6, 1.6)

    shade = 1.0 + emb * (0.42 * strength)
    out = img * shade[:, :, None]

    if not quick:
        # oil paint is glossy: a hint of specular on the lit ridges only
        spec = np.clip(emb, 0.0, 1.6) ** 3 * (0.030 * strength)
        out += spec[:, :, None] * 255.0

    return out


# --------------------------------------------------------------------------
# blend modes
# --------------------------------------------------------------------------


def blend_images(top, bottom, amount, mode="Normal"):
    """Blend the painting with the original photograph."""
    a = np.clip(top, 0, 255) / 255.0
    b = np.clip(bottom, 0, 255) / 255.0
    m = _blend_mode(a, b, mode)
    out = a + (m - a) * float(amount)
    return np.clip(out, 0, 1) * 255.0


def _blend_mode(a, b, mode):
    if mode == "Normal":
        return b
    if mode == "Multiply":
        return a * b
    if mode == "Screen":
        return 1 - (1 - a) * (1 - b)
    if mode == "Overlay":
        return np.where(a <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "Hard Light":
        return np.where(b <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "Soft Light":
        return np.where(b <= 0.5,
                        a - (1 - 2 * b) * a * (1 - a),
                        a + (2 * b - 1) * (np.sqrt(np.clip(a, 0, 1)) - a))
    if mode == "Darken":
        return np.minimum(a, b)
    if mode == "Lighten":
        return np.maximum(a, b)
    if mode == "Linear Burn":
        return np.clip(a + b - 1.0, 0, 1)
    if mode == "Linear Dodge":
        return np.clip(a + b, 0, 1)
    if mode == "Color Burn":
        return 1 - np.clip((1 - a) / np.maximum(b, 1e-4), 0, 1)
    if mode == "Color Dodge":
        return np.clip(a / np.maximum(1 - b, 1e-4), 0, 1)
    if mode == "Difference":
        return np.abs(a - b)
    if mode in ("Hue", "Color", "Luminosity"):
        ah = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_BGR2HLS).astype(np.float32)
        bh = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_BGR2HLS).astype(np.float32)
        out = ah.copy()
        if mode == "Hue":
            out[:, :, 0] = bh[:, :, 0]
        elif mode == "Color":
            out[:, :, 0] = bh[:, :, 0]
            out[:, :, 2] = bh[:, :, 2]
        else:
            out[:, :, 1] = bh[:, :, 1]
        return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                            cv2.COLOR_HLS2BGR).astype(np.float32) / 255.0
    return b
