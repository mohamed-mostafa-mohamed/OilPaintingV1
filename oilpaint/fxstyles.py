"""The filter styles made from Photoshop templates that run on the GPU
too (static/js/oilfx.js) - one module per style, all with the same
functions: render(), look_payload(), asset_image(), ASSET_NAMES."""
from __future__ import annotations

from . import brushfx, canvas_oil, halftone, impression, oilfx, painted, silkoil, sketchfx, smear, watercolor

STYLES = {"Canvas Oil": canvas_oil, "Oil Paint FX": oilfx, "Sketch Paint": sketchfx, "Rough Brush": brushfx,
          "Watercolor": watercolor, "Halftone": halftone,
          "Impression": impression, "Painted Photo": painted,
          "Oil Smear": smear, "Silk Oil": silkoil}


def module(style):
    return STYLES.get(style)
