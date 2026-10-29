"""Variations of the GPU filter styles (Oil Paint FX, Sketch Paint, Rough
Brush, Watercolor): a seed nudges the main settings a little, so each Run
can show several takes. Seed 0 = exactly the settings on screen.

static/js/oilfx.js has the same function (``jitter``) and the same tables,
so a variation looks the same on the GPU and the CPU."""
from __future__ import annotations

import math


def jitter(seed: int, i: int) -> float:
    """A repeatable number in -1..1 for this seed and slot (0 for seed 0)."""
    if not seed:
        return 0.0
    v = math.sin(seed * 12.9898 + i * 78.233) * 43758.5453
    return (v - math.floor(v)) * 2.0 - 1.0


def vary(st: dict, table, seed: int) -> dict:
    """table: (setting, slot, amount) - the setting is scaled by up to +-amount."""
    if not seed:
        return st
    out = dict(st)
    for name, slot, amt in table:
        out[name] = max(0.0, out[name] * (1.0 + amt * jitter(seed, slot)))
    return out
