"""
Run Envato (or any) Photoshop actions inside OilStudio - Photoshop is not needed.

    from actions import ActionEngine
    engine = ActionEngine("ps_actions", cache_dir="workdir/action-cache")
    pack = engine.library.get("canvas-oil-painting")
    result = engine.run(pack, rgb_uint8)          # result.image, result.skipped

Modules:
    atn        reads .atn files (every step with all its settings)
    resources  reads .pat patterns and .abr brushes
    filters    adjustments, blend modes and Filter Gallery filters
    player     the Photoshop-like document that plays the steps
    focus      the "brush" (focus area) layer most Envato actions need
    library    the folder of action packs
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from . import atn, focus, player
from .library import BRUSH_MODES, Library
from .player import Result
from .resources import PatternStore

__all__ = ["ActionEngine", "ActionError", "BRUSH_MODES", "Result", "Stopped"]


class ActionError(Exception):
    pass


class Stopped(BaseException):
    """Raised from the progress callback to stop a running action."""


class _Loaded:
    def __init__(self, key, sets, patterns, brushes):
        self.key = key
        self.sets = sets
        self.patterns = patterns
        self.brushes = brushes

    def action(self, name):
        for s in self.sets:
            a = s.action(name) if name else None
            if a is not None:
                return a
        for s in self.sets:
            if s.actions:
                return s.actions[0] if not name else None
        return None


class ActionEngine:
    def __init__(self, library_dir: str, cache_dir: str | None = None):
        self.library = Library(library_dir)
        self.cache_dir = cache_dir
        self._loaded: dict[str, _Loaded] = {}
        self._lock = threading.Lock()

    # -- loading a pack's files (kept in memory between runs) ---------------
    def _load(self, pack: dict) -> _Loaded:
        files = pack.get("resources") or []
        key = tuple((f, os.path.getmtime(f)) for f in files if os.path.exists(f))
        with self._lock:
            hit = self._loaded.get(pack["id"])
            if hit is not None and hit.key == key:
                return hit
            atns = [f for f in files if f.lower().endswith(".atn")]
            sets = []
            for f in atns:
                try:
                    sets.append(atn.load(f))
                except atn.AtnError as exc:
                    raise ActionError(f"{os.path.basename(f)} could not be read: {exc}") from exc
            pats = [f for f in files if f.lower().endswith(".pat")]
            abrs = [f for f in files if f.lower().endswith(".abr")]
            cache = os.path.join(self.cache_dir, pack["id"]) if self.cache_dir else None
            loaded = _Loaded(key, sets, PatternStore(pats, cache), player.load_brushes(abrs))
            self._loaded[pack["id"]] = loaded
            return loaded

    def actions(self, pack: dict) -> list[str]:
        return [a.name for s in self._load(pack).sets for a in s.actions]

    def scan(self, pack: dict, action: str | None = None) -> dict:
        """How much of the action OilStudio can play, before running it."""
        lo = self._load(pack)
        act = lo.action(action or pack.get("action"))
        if act is None:
            raise ActionError(f"There is no action called {action!r} in this pack.")
        info = player.scan(act)
        info["action"] = act.name
        return info

    # -- running ----------------------------------------------------------
    def run(self, pack: dict, image_rgb: np.ndarray, *, action: str | None = None,
            brush: str | None = None, long_side: int | None = None,
            painted_focus: np.ndarray | None = None, progress=None) -> Result:
        """Play the action on an RGB uint8 image. The image is resized so its
        long side is ``long_side`` (the size the action was made for, usually
        3500 px); 0 keeps the size."""
        lo = self._load(pack)
        act = lo.action(action or pack.get("action"))
        if act is None:
            raise ActionError(f"There is no action called {action!r} in this pack.")
        side = int(pack.get("long_side", 3500) if long_side is None else long_side)
        img = image_rgb
        if side > 0:
            h, w = img.shape[:2]
            k = side / max(h, w)
            if abs(k - 1) > 1e-3:
                img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))),
                                 interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        mode = brush if brush in BRUSH_MODES else pack.get("brush", "subject")
        if progress:
            progress(0, len(act.steps), "Finding the focus area")
        mask = focus.make(img, mode, painted_focus)
        # px sizes in the action were made for this long side
        reference = int(pack.get("reference_side") or player.REFERENCE_SIDE)
        return player.run(act, img, patterns=lo.patterns, brushes=lo.brushes, focus=mask,
                          brush_layer=pack.get("brush_layer") or "brush",
                          reference_side=reference, progress=progress)
