"""Ready-to-use presets: stored combinations of settings."""

from __future__ import annotations

import json
import os
import re
import threading

BUILTIN = {
    "AKV Default": {},
    "AKV Classic Oil": dict(
        simplicity=4, direction=25, wide_bg_strokes=5, expression=20,
        filling=90, relief=38, lightening=6, stroke_thickness=20,
        max_stroke_length=32, min_stroke_length=28, hair_thickness=35,
        brush_density=45, intensity=66),
    "AKV Portrait": dict(
        simplicity=3, direction=18, wide_bg_strokes=6, expression=16,
        filling=92, relief=30, lightening=8, stroke_thickness=13,
        max_stroke_length=22, min_stroke_length=35, hair_thickness=30,
        brush_density=50, intensity=72),
    "AKV Soft Portrait": dict(
        simplicity=4, direction=14, wide_bg_strokes=7, expression=12,
        filling=95, relief=22, lightening=14, stroke_thickness=15,
        max_stroke_length=20, min_stroke_length=40, hair_thickness=45,
        brush_density=60, intensity=52, blend_original=18),
    "AKV Landscape": dict(
        simplicity=6, direction=35, wide_bg_strokes=7, expression=30,
        filling=88, relief=42, lightening=5, stroke_thickness=24,
        max_stroke_length=45, min_stroke_length=25, hair_thickness=32,
        brush_density=42, intensity=64),
    "AKV Impressionism": dict(
        simplicity=8, direction=45, global_angle=60, wide_bg_strokes=4,
        expression=55, filling=78, relief=55, lightening=12,
        stroke_thickness=30, max_stroke_length=28, min_stroke_length=20,
        hair_thickness=28, brush_density=35, intensity=78, random_seed=451),
    "AKV Van Gogh Swirl": dict(
        simplicity=7, direction=8, wide_bg_strokes=3, expression=70,
        filling=82, relief=72, lightening=8, stroke_thickness=26,
        max_stroke_length=60, min_stroke_length=45, hair_thickness=26,
        brush_density=38, intensity=85, random_seed=1889),
    "AKV Palette Knife": dict(
        simplicity=9, direction=40, wide_bg_strokes=8, expression=35,
        filling=86, relief=85, lightening=4, stroke_thickness=42,
        max_stroke_length=24, min_stroke_length=60, hair_thickness=95,
        brush_density=90, intensity=90),
    "AKV Fine Detail": dict(
        simplicity=2, direction=20, wide_bg_strokes=4, expression=14,
        filling=96, relief=26, lightening=4, stroke_thickness=9,
        max_stroke_length=16, min_stroke_length=30, hair_thickness=30,
        brush_density=55, intensity=72),
    "AKV Dry Brush": dict(
        simplicity=5, direction=30, wide_bg_strokes=5, expression=45,
        filling=58, relief=48, lightening=16, stroke_thickness=22,
        max_stroke_length=40, min_stroke_length=22, hair_thickness=18,
        brush_density=18, intensity=80),
    "AKV Sketchy Underpainting": dict(
        simplicity=7, direction=22, wide_bg_strokes=3, expression=60,
        filling=34, relief=30, lightening=10, stroke_thickness=16,
        max_stroke_length=50, min_stroke_length=18, hair_thickness=22,
        brush_density=25, intensity=74),
    "AKV Abstract Art": dict(
        simplicity=9, direction=55, global_angle=120, expression=65,
        filling=84, relief=60, stroke_thickness=34, max_stroke_length=42,
        intensity=82, abstract_enabled=1, ab_color=55, ab_saturation=22,
        ab_contrast=45, cor_saturation=15, dist_strength=35,
        dist_curvature=9, dist_seed=77),
    "AKV Old Master Still Life": dict(
        simplicity=3, direction=12, wide_bg_strokes=7, expression=18,
        filling=96, relief=46, lightening=0, random_seed=1853,
        stroke_thickness=13, max_stroke_length=22, min_stroke_length=38,
        hair_thickness=28, brush_density=58, intensity=72,
        abstract_enabled=1, cor_saturation=14, cor_brightness=-18,
        light_enabled=1, light_vignette=70, light_glow=36, light_warmth=32,
        light_x=28, light_y=16, light_spread=58,
        canvas_enabled=1, canvas_texture="Linen", canvas_relief=70,
        canvas_scale=115, canvas_light=40, canvas_strength=58),
    "AKV Chiaroscuro Portrait": dict(
        simplicity=3, direction=14, wide_bg_strokes=8, expression=16,
        filling=95, relief=34, lightening=0, stroke_thickness=12,
        max_stroke_length=20, min_stroke_length=40, hair_thickness=30,
        brush_density=58, intensity=74,
        abstract_enabled=1, cor_saturation=8, cor_brightness=-14,
        light_enabled=1, light_vignette=66, light_glow=26, light_warmth=26,
        light_x=32, light_y=20, light_spread=52,
        canvas_enabled=1, canvas_texture="Canvas Fine", canvas_relief=55,
        canvas_scale=100, canvas_strength=45),
    "AKV Museum Piece": dict(
        simplicity=5, direction=24, wide_bg_strokes=6, expression=24,
        filling=92, relief=44, stroke_thickness=19, max_stroke_length=30,
        intensity=68, canvas_enabled=1, canvas_texture="Linen",
        canvas_relief=60, canvas_strength=55, canvas_scale=85,
        frame_enabled=1, frame_style="Gilded", frame_width=7),
    "AKV Raw Canvas Edges": dict(
        simplicity=5, direction=26, expression=28, filling=86, relief=40,
        stroke_thickness=20, max_stroke_length=32, intensity=68,
        edge_enabled=1, edge_number=45, edge_thickness=26, edge_max_length=38,
        edge_smoothness=40, edge_hair_thickness=26, edge_relief=55,
        canvas_enabled=1, canvas_texture="Canvas Rough", canvas_strength=50),
    # native recreation of the Envato "Canvas Oil Painting" action
    "Canvas Oil": dict(paint_style="Canvas Oil"),
    "Canvas Oil - Clean (no scratches)": dict(
        paint_style="Canvas Oil", co_scratches=0, co_speckles=20),
    "Canvas Oil - Bold Knife": dict(
        paint_style="Canvas Oil", co_stroke_size=65, co_patch=70, co_detail=40, co_drag=55,
        co_dabs=70, co_relief=65, co_scratches=35, co_contrast=45,
        co_vibrance=55),
    "Canvas Oil - Fine Detail": dict(
        paint_style="Canvas Oil", co_patch=25, co_detail=75, co_drag=30,
        co_dabs=40, co_relief=40, co_scratches=10, co_speckles=25,
        co_stroke_size=25),
    "Canvas Oil - Thick Brush": dict(
        paint_style="Canvas Oil", co_brush=100, co_stroke_size=75,
        co_patch=55, co_scratches=10, co_speckles=25),
    "Canvas Oil - Flat Knife (fast)": dict(
        paint_style="Canvas Oil", co_brush=0, co_sketch=0),
    "Canvas Oil - Sketchy": dict(
        paint_style="Canvas Oil", co_brush=80, co_loose=80, co_sketch=80,
        co_stroke_size=55, co_scratches=10, co_speckles=25),
    "Canvas Oil - Smooth": dict(
        paint_style="Canvas Oil", co_brush=80, co_loose=20, co_sketch=0,
        co_scratches=0, co_speckles=20),
    # vintage halftone print (halftone.py)
    "Halftone - Vintage Sepia": dict(paint_style="Halftone"),
    "Halftone - Clean Print (no scratches)": dict(
        paint_style="Halftone", ht_scratches=0, ht_grain=35),
    "Halftone - Heavy Grunge": dict(
        paint_style="Halftone", ht_grain=80, ht_edges=75, ht_scratches=90,
        ht_contrast=65),
    "Halftone - Blue Ink": dict(
        paint_style="Halftone", ht_dark="#0e1a33", ht_light="#eef2f5", ht_film=40),
    # impressionist dabs on canvas (impression.py)
    "Impression": dict(paint_style="Impression"),
    "Impression - Loose": dict(paint_style="Impression", im_dab=80, im_light=70, im_swirl=65),
    "Impression - Fine": dict(paint_style="Impression", im_dab=25, im_light=35, im_relief=40),
    "Impression - No Border": dict(paint_style="Impression", im_border=0),
    # cut-out paint on kraft paper (painted.py)
    "Painted Photo": dict(paint_style="Painted Photo"),
    "Painted Photo - Bold Poster": dict(
        paint_style="Painted Photo", pp_levels=5, pp_detail=30, pp_outline=75),
    "Painted Photo - Fine Detail": dict(
        paint_style="Painted Photo", pp_levels=13, pp_detail=80, pp_outline=40),
    "Painted Photo - Clean (no paper)": dict(
        paint_style="Painted Photo", pp_paper=0, pp_texture=30),
    "Painted Photo - Natural Colours": dict(
        paint_style="Painted Photo", pp_grade=0),
    "Halftone - Newspaper": dict(
        paint_style="Halftone", ht_dark="#141414", ht_light="#efece4", ht_film=0,
        ht_grid=60, ht_scratches=15),
}

_LOCK = threading.Lock()
_SAFE = re.compile(r"[^A-Za-z0-9 _.\-]")


class PresetStore:
    """Built-in presets plus the ones the user saves."""

    def __init__(self, path: str):
        self.path = path
        self._user = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self._user = {str(k): v for k, v in data.items()
                                  if isinstance(v, dict)}
            except (OSError, ValueError):
                self._user = {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._user, fh, indent=1)
        os.replace(tmp, self.path)

    def names(self):
        return {"builtin": sorted(BUILTIN), "user": sorted(self._user)}

    def get(self, name: str):
        if name in self._user:
            return dict(self._user[name])
        if name in BUILTIN:
            return dict(BUILTIN[name])
        return None

    def save(self, name: str, values: dict):
        name = _SAFE.sub("", str(name)).strip()[:48]
        if not name:
            raise ValueError("A preset needs a name.")
        if name in BUILTIN:
            raise ValueError("That name belongs to a built-in preset.")
        with _LOCK:
            self._user[name] = dict(values)
            self._save()
        return name

    def delete(self, name: str):
        with _LOCK:
            if name in self._user:
                del self._user[name]
                self._save()
                return True
        return False
