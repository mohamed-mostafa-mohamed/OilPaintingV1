"""
"Oil Smear (PSD)" — plays the recipe made from the template PSD
(recipes/oil_smear/recipe.json, see engine/psd_recipe.py) on the photo.

If the template's pictures (displacement map, textures) are not in the
recipe folder, generated stand-ins from smear_template.py are used, so the
style still works without them.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from engine import psd_recipe as R

from . import smear_template

BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECIPE_DIR = os.path.join(BASE, "recipes", "oil_smear")
_cache = {}


def recipe():
    path = os.path.join(RECIPE_DIR, "recipe.json")
    mt = os.path.getmtime(path)
    hit = _cache.get(path)
    if hit is None or hit[0] != mt:
        hit = (mt, R.Recipe(path))
        _cache[path] = hit
    return hit[1]


def _fallback_factory(rec, seed):
    """Generated textures for a recipe whose pictures are missing."""
    doc = rec.data["document"]
    tex = smear_template.texture(doc["height"] // 4, doc["width"] // 4, seed)   # 0..255

    def fallback(kind, step):
        if kind == "map":
            return np.clip(tex, 0, 255).astype(np.uint8)
        name = step.get("image", "")
        if "multiply" in name:
            light = np.interp(tex, smear_template._LIGHT_X, smear_template._LIGHT_Y)
            return np.clip(light, 0, 255).astype(np.uint8)
        return np.clip(tex, 0, 255).astype(np.uint8)
    return fallback


def render(image_bgr: np.ndarray, p: dict, progress=None) -> np.ndarray:
    rec = recipe()
    rgb = np.ascontiguousarray(np.asarray(image_bgr, np.float32)[..., ::-1]) / 255.0
    controls = {
        "warp": float(p.get("rc_warp", 100)) / 100.0,
        "texture": float(p.get("rc_texture", 100)) / 100.0,
        "adjust": float(p.get("rc_adjust", 100)) / 100.0,
        "paint_smooth": float(p.get("rc_smooth", 100)) / 100.0,
        "paint_relief": float(p.get("rc_relief", 0)) / 100.0,
        "posterize_mult": float(p.get("rc_posterize", 10)) / 10.0,
        "cutout_mult": float(p.get("rc_cutout", 8)) / 8.0,
    }
    fb = None if rec.has_assets() else _fallback_factory(rec, int(p.get("rc_seed", 1)))

    def prog(f, label):
        if progress:
            progress(f, f"Oil Smear (PSD): {label}")
    out, _ = R.run(rec, rgb, controls, prog, fallback=fb)
    return (out[..., ::-1] * 255.0).astype(np.float32)


def uses_template_assets() -> bool:
    try:
        return recipe().has_assets()
    except OSError:
        return False
