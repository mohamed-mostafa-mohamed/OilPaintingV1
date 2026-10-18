"""
Parameter schema for the OilPaint engine.

Mirrors the parameter layout of a professional photo-to-oil-painting
application: a Painting group (Image / Strokes / Edge tabs), an
Abstract Art group, and a Decoration group (Canvas / Frame / Text).

Every entry is (default, min, max).  ``clean()`` validates and clamps a
raw dict coming from the web UI or from a saved preset.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# schema: name -> (default, min, max)
# --------------------------------------------------------------------------

PAINTING = {
    # --- top level -------------------------------------------------------
    "blend_original": (0, 0, 100),      # merge result with the original
    # --- Image tab -------------------------------------------------------
    "simplicity": (3, 1, 15),           # detail removal / abstraction
    "direction": (15, 0, 100),          # 0 = along edges, 100 = common direction
    "global_angle": (45, 0, 180),       # the "common direction" angle
    "wide_bg_strokes": (9, 1, 10),      # scale strokes in uniform areas
    "expression": (20, 5, 100),         # uneven edges, paint mixing
    "filling": (94, 0, 100),            # completes the painting in empty spaces
    "relief": (30, 0, 100),             # volumetric strokes above the canvas
    "lightening": (2, 0, 100),          # adds white, lightens the image
    "random_seed": (1234, 0, 9999),
    # --- Strokes tab -----------------------------------------------------
    "stroke_thickness": (12, 1, 100),
    "max_stroke_length": (22, 1, 100),
    "min_stroke_length": (35, 1, 100),  # in % of max length
    "hair_thickness": (32, 10, 100),
    "brush_density": (52, 1, 100),
    "intensity": (70, 1, 100),          # sharpness / contrast of strokes
}

BLEND_MODES = [
    "Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light",
    "Darken", "Lighten", "Linear Burn", "Linear Dodge", "Color Burn",
    "Color Dodge", "Difference", "Hue", "Color", "Luminosity",
]

EDGE = {
    "edge_enabled": (0, 0, 1),          # Add Strokes check-box
    "edge_number": (35, 1, 100),
    "edge_thickness": (30, 0, 100),
    "edge_max_length": (35, 10, 100),
    "edge_smoothness": (45, 5, 100),
    "edge_hair_thickness": (28, 15, 50),
    "edge_use_color": (0, 0, 1),        # Stroke Color check-box
    "edge_relief": (45, 0, 100),
    "edge_ellipse": (0, 0, 1),
    "edge_seed": (7, 0, 9999),
}

ABSTRACT = {
    "abstract_enabled": (0, 0, 1),
    # Strokes group
    "ab_color": (0, 0, 90),
    "ab_saturation": (0, 0, 40),
    "ab_contrast": (0, 0, 100),
    # Correction group
    "cor_hue": (0, -180, 180),
    "cor_saturation": (0, -100, 100),
    "cor_brightness": (0, -100, 100),
    # Distortion group
    "dist_strength": (0, 0, 100),
    "dist_curvature": (8, 2, 25),
    "dist_seed": (1, 1, 9999),
}

LIGHT = {
    "light_enabled": (0, 0, 1),
    "light_vignette": (55, 0, 100),   # how far the picture sinks into shadow
    "light_glow": (30, 0, 100),       # bloom around the brightest paint
    "light_warmth": (25, -100, 100),  # warm key light / cool shadow
    "light_x": (30, 0, 100),          # where the light falls from
    "light_y": (18, 0, 100),
    "light_spread": (62, 12, 100),    # how wide the pool of light is
}

CANVAS = {
    "canvas_enabled": (0, 0, 1),
    "canvas_relief": (55, 0, 100),      # how deep the weave shows
    "canvas_scale": (100, 20, 400),     # texture scale, %
    "canvas_light": (45, 0, 100),       # lighting angle influence
    "canvas_strength": (60, 0, 100),    # overall blending amount
}

FRAME = {
    "frame_enabled": (0, 0, 1),
    "frame_width": (6, 1, 30),          # % of the smaller image side
    "frame_seed": (3, 0, 9999),
}

TEXT = {
    "text_enabled": (0, 0, 1),
    "text_opacity": (60, 0, 100),
    "text_size": (6, 1, 40),            # % of the smaller image side
    "text_angle": (0, -180, 180),
    "text_tile": (0, 0, 1),
}

# Canvas Oil: a native recreation of the Envato "Canvas Oil Painting"
# Photoshop action (used when paint_style is "Canvas Oil")
CANVAS_OIL = {
    "co_patch": (45, 0, 100),           # size of the paint patches
    "co_detail": (55, 0, 100),          # smaller patches where there is detail
    "co_drag": (40, 0, 100),            # patches run down like wet paint
    "co_dabs": (55, 0, 100),            # small knife dabs in flat areas
    "co_brush": (70, 0, 100),           # real brush strokes over the patches
    "co_stroke_size": (45, 0, 100),     # width of those strokes
    "co_loose": (55, 0, 100),           # loose, expressive brushwork
    "co_sketch": (45, 0, 100),          # hand-drawn scribble lines
    "co_relief": (45, 0, 100),          # paint ridges along patch edges
    "co_canvas": (45, 0, 100),          # white canvas and weave showing through
    "co_scratches": (20, 0, 100),       # cross-hatched knife scratches
    "co_speckles": (40, 0, 100),        # small glowing paint spots
    "co_contrast": (35, 0, 100),        # Levels
    "co_vibrance": (40, -100, 100),     # Vibrance
    "co_seed": (7, 0, 9999),
}

# Halftone: a vintage halftone print (sepia ink, dot grid, grain, scratches)
# built natively - see halftone.py (used when paint_style is "Halftone")
HALFTONE = {
    "ht_contrast": (50, 0, 100),        # tone punch (Levels)
    "ht_clarity": (50, 0, 100),         # local contrast: clarity + texture
    "ht_grain": (50, 0, 100),           # stipple grain specks
    "ht_edges": (50, 0, 100),           # black specks on strong detail
    "ht_grid": (100, 0, 100),           # woven cloth dot grid
    "ht_grid_size": (38, 12, 120),      # grid pitch, px on a 4000 px picture
    "ht_scratches": (50, 0, 100),       # light cracked scratches
    "ht_film": (100, 0, 100),           # warm film tone of the ink ramp
    "ht_seed": (7, 0, 9999),
}

# Painted Photo: flat cut-out paint patches on kraft paper with a warm film
# grade - see painted.py (used when paint_style is "Painted Photo")
PAINTED = {
    "pp_levels": (10, 3, 16),            # how many tones the patches use
    "pp_detail": (50, 0, 100),          # smaller patches, finer edges
    "pp_outline": (50, 0, 100),         # dark outlines on strong edges
    "pp_texture": (50, 0, 100),         # palette-knife speckle + canvas
    "pp_paper": (100, 0, 100),          # kraft and old-paper textures
    "pp_grain": (50, 0, 100),           # how grainy the paper is (50 = template)
    "pp_grade": (100, 0, 100),          # warm film colour grade
    "pp_seed": (3, 0, 9999),
}

# Impression: soft impressionist dabs on canvas - see impression.py
IMPRESSION = {
    "im_dab": (50, 0, 100),             # how far the paint dabs smear the photo
    "im_swirl": (50, 0, 100),           # how much the strokes swirl in flat areas
    "im_light": (50, 0, 100),           # long light strokes dragged over dark areas
    "im_relief": (50, 0, 100),          # bristle grooves and paint relief
    "im_canvas": (50, 0, 100),          # woven canvas texture
    "im_border": (50, 0, 100),          # white border round the painting
    "im_seed": (5, 0, 9999),
}

NUMERIC = {}
for _group in (PAINTING, EDGE, ABSTRACT, LIGHT, CANVAS, FRAME, TEXT, CANVAS_OIL, HALFTONE,
               PAINTED, IMPRESSION):
    NUMERIC.update(_group)

# Non numeric options ------------------------------------------------------
PAINT_STYLES = ["Brush Strokes", "Canvas Oil", "Halftone", "Painted Photo", "Impression"]
# styles made on the CPU in one go (no strokes for the GPU to restyle)
CPU_STYLES = ("Canvas Oil", "Halftone", "Painted Photo", "Impression")

CHOICES = {
    "paint_style": (PAINT_STYLES, "Brush Strokes"),
    "blend_mode": (BLEND_MODES, "Normal"),
    "canvas_texture": (
        ["Canvas Fine", "Canvas Rough", "Linen", "Corrugated Paper",
         "Brick Wall", "Ceramic", "Watercolor Paper", "Custom"],
        "Canvas Fine",
    ),
    "frame_style": (
        ["Classic Wood", "Gilded", "Passe-Partout", "Raw Canvas Edge",
         "Museum Black", "Scratched Metal"],
        "Classic Wood",
    ),
    "text_position": (
        ["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"],
        "Bottom Right",
    ),
    "text_font": (["serif", "sans", "mono", "script"], "serif"),
}

COLORS = {
    "edge_color": "#ffffff",
    "frame_color": "#6b4a2b",
    "text_color": "#ffffff",
    "canvas_color": "#f3ece0",
    "ht_dark": "#21130d",               # Halftone ink (shadows)
    "ht_light": "#f9f4c4",              # Halftone paper (highlights)
}


def _num(value, lo, hi, default):
    try:
        if isinstance(value, bool):
            value = int(value)
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    v = max(lo, min(hi, v))
    return int(round(v)) if float(lo).is_integer() and float(hi).is_integer() else v


def defaults() -> dict:
    p = {k: v[0] for k, v in NUMERIC.items()}
    for key, (options, dflt) in CHOICES.items():
        p[key] = dflt
    p.update(COLORS)
    p["text_content"] = "OilPaint"
    p["guides"] = []            # guiding lines from the Stroke Direction tool
    return p


def clean(raw: dict | None) -> dict:
    """Validate and clamp a raw parameter dict from the client."""
    raw = raw or {}
    p = defaults()
    for key, (dflt, lo, hi) in NUMERIC.items():
        if key in raw:
            p[key] = _num(raw[key], lo, hi, dflt)
    for key, (options, dflt) in CHOICES.items():
        val = raw.get(key)
        p[key] = val if val in options else dflt
    for key, dflt in COLORS.items():
        val = str(raw.get(key, dflt))
        p[key] = val if is_hex(val) else dflt
    p["text_content"] = str(raw.get("text_content", "OilPaint"))[:120]
    p["guides"] = _clean_guides(raw.get("guides"))
    return p


def is_hex(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 7
        and value[0] == "#"
        and all(c in "0123456789abcdefABCDEF" for c in value[1:])
    )


def _clean_guides(raw):
    """Guiding lines: [[[x, y], [x, y], ...], ...] in normalised 0..1 coords."""
    out = []
    if not isinstance(raw, list):
        return out
    for line in raw[:400]:
        if not isinstance(line, list) or len(line) < 2:
            continue
        pts = []
        for pt in line[:600]:
            try:
                x, y = float(pt[0]), float(pt[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                pts.append((x, y))
        if len(pts) >= 2:
            out.append(pts)
    return out


def hex_to_bgr(value: str):
    value = value.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return (float(b), float(g), float(r))
