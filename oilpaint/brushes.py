"""
Post-processing tools, used on the After tab once the painting is done:
Smudge, History Brush, Oil Brush and 3D Brush.

Each tool receives the path the user dragged (a list of image-space points)
and edits the finished painting in place, the way the desktop tools do.
"""

from __future__ import annotations

import numpy as np

from .engine import Brush, apply_relief


def _disc(radius: int, hardness: float) -> np.ndarray:
    """Soft round brush tip, 0..1."""
    r = max(1, int(radius))
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xs * xs + ys * ys) / max(r, 1e-6)
    hard = float(np.clip(hardness, 0, 100)) / 100.0
    inner = 0.02 + 0.96 * hard
    t = np.clip((1.0 - d) / max(1e-6, 1.0 - inner), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _walk(path, spacing):
    """Resample a path at a fixed spacing."""
    pts = np.asarray(path, np.float32)
    if len(pts) == 1:
        return pts
    out = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        a, b = out[-1] if acc == 0.0 else pts[i - 1], pts[i]
        seg = np.linalg.norm(pts[i] - pts[i - 1])
        if seg < 1e-6:
            continue
        d = pts[i] - pts[i - 1]
        n = int(seg // spacing)
        for k in range(1, n + 1):
            out.append(pts[i - 1] + d * (k * spacing / seg))
        if not n:
            continue
    if len(out) == 1:
        out.append(pts[-1])
    return np.asarray(out, np.float32)


def _roi(img, cx, cy, r):
    h, w = img.shape[:2]
    x0, y0 = int(cx) - r, int(cy) - r
    x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1, sx0, sy0


# --------------------------------------------------------------------------


def smudge(img, path, size=40, hardness=50, strength=50):
    """Mix colours by dragging pixels, like a finger through wet paint."""
    r = max(1, int(size) // 2)
    tip = _disc(r, hardness)
    s = float(np.clip(strength, 1, 100)) / 100.0
    pts = _walk(path, max(1.0, r * 0.22))
    carried = None
    for (cx, cy) in pts:
        box = _roi(img, cx, cy, r)
        if box is None:
            continue
        x0, y0, x1, y1, sx0, sy0 = box
        m = tip[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][:, :, None]
        roi = img[y0:y1, x0:x1]
        if carried is None or carried.shape != roi.shape:
            carried = roi.copy()
            continue
        a = m * s
        roi += (carried - roi) * a
        carried += (roi - carried) * (0.55 + 0.4 * (1.0 - s))
    return img


def history_brush(img, original, path, size=60, hardness=50, strength=60,
                  restore_original=True, snapshot=None):
    """Fade the effect back towards the photograph (or the last snapshot)."""
    target = original if restore_original else (snapshot
                                                if snapshot is not None else original)
    r = max(1, int(size) // 2)
    tip = _disc(r, hardness)
    s = float(np.clip(strength, 1, 100)) / 100.0
    for (cx, cy) in _walk(path, max(1.0, r * 0.25)):
        box = _roi(img, cx, cy, r)
        if box is None:
            continue
        x0, y0, x1, y1, sx0, sy0 = box
        m = tip[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][:, :, None] * s
        roi = img[y0:y1, x0:x1]
        roi += (target[y0:y1, x0:x1] - roi) * m
    return img


def oil_brush(img, height, path, colour, size=40, relief=50, reserve=50,
              mixing=True, cleaning=True, flat=False, seed=0):
    """Strengthen the oil effect: real brush strokes in a chosen colour."""
    rng = np.random.default_rng(int(seed))
    pts = np.asarray(path, np.float32)
    if len(pts) < 2:
        pts = np.vstack([pts, pts[-1] + np.array([1.0, 1.0], np.float32)])

    thickness = max(2.0, float(size))
    hair = max(1.2, thickness * 0.22)
    bristles = int(np.clip(thickness / hair * 1.4, 2, 14))
    cov = np.zeros(img.shape[:2], np.float32)
    brush = Brush(img, height, cov)

    colour = np.asarray(colour, np.float32)
    if not cleaning and mixing:
        # the brush keeps whatever it ended the last stroke with
        mid = pts[len(pts) // 2]
        x = int(np.clip(mid[0], 0, img.shape[1] - 1))
        y = int(np.clip(mid[1], 0, img.shape[0] - 1))
        colour = colour * 0.55 + img[y, x] * 0.45

    res = float(np.clip(reserve, 0, 100)) / 100.0
    brush.stroke(pts, colour, thickness, hair, bristles,
                 softness=0.35, opacity=1.0,
                 height_amt=0.6 + 2.4 * (float(relief) / 100.0),
                 shade_var=0.10 + 0.20 * (float(relief) / 100.0),
                 rng=rng, flat=flat, reserve=0.25 + 0.75 * res,
                 mixing=mixing)

    _relight(img, height, cov, float(relief) / 100.0)
    return img


def brush_3d(img, height, path, size=60, smearing=40, relief=50, flat=False,
             seed=0):
    """Bring volume and relief to the painting without changing its colours."""
    rng = np.random.default_rng(int(seed))
    r = max(1, int(size) // 2)
    pts = _walk(path, max(1.0, r * 0.3))
    if len(pts) < 2:
        return img

    sm = float(np.clip(smearing, 0, 100)) / 100.0
    if sm > 0.01:
        smudge(img, pts, size=size, hardness=45, strength=int(10 + 70 * sm))

    cov = np.zeros(img.shape[:2], np.float32)
    brush = Brush(np.zeros_like(img), height, cov)
    thickness = max(2.0, float(size))
    hair = max(1.2, thickness * 0.20)
    bristles = int(np.clip(thickness / hair * 1.4, 2, 14))
    brush.stroke(pts, np.zeros(3, np.float32), thickness, hair, bristles,
                 softness=0.4, opacity=1.0,
                 height_amt=0.8 + 2.6 * (float(relief) / 100.0),
                 shade_var=0.22, rng=rng, flat=flat)

    _relight(img, height, cov, float(relief) / 100.0)
    return img


def _relight(img, height, cov, strength):
    """Re-light only the area the tool just touched."""
    ys, xs = np.where(cov > 0.01)
    if len(ys) == 0:
        return
    pad = 6
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    sub = img[y0:y1, x0:x1]
    lit = apply_relief(sub, height[y0:y1, x0:x1], max(0.05, strength))
    m = np.clip(cov[y0:y1, x0:x1], 0, 1)[:, :, None]
    sub += (lit - sub) * m
    np.clip(sub, 0, 255, out=sub)


TOOLS = {
    "smudge": {
        "label": "Smudge",
        "params": [("size", 1, 300, 40), ("hardness", 0, 100, 50),
                   ("strength", 1, 100, 50)],
    },
    "history": {
        "label": "History Brush",
        "params": [("size", 1, 500, 60), ("hardness", 0, 100, 50),
                   ("strength", 1, 100, 60)],
        "flags": [("restore_original", True)],
    },
    "oil": {
        "label": "Oil Brush",
        "params": [("size", 2, 150, 40), ("relief", 0, 100, 50),
                   ("reserve", 0, 100, 50)],
        "flags": [("mixing", True), ("cleaning", True), ("flat", False)],
        "colour": True,
    },
    "brush3d": {
        "label": "3D Brush",
        "params": [("size", 2, 300, 60), ("smearing", 0, 100, 40),
                   ("relief", 0, 100, 50)],
        "flags": [("flat", False)],
    },
}
