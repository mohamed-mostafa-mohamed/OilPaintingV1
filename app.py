"""
OilStudio - a photo-to-oil-painting web application.

Flask front end around the painting engine in ``oilpaint/``.  The workspace
mirrors a desktop painting program: a Before/After image window, a settings
panel with Painting (Image / Strokes / Edge), Abstract Art and Decoration
(Canvas / Frame / Text) tabs, a Timeline of intermediate shots, presets,
post-processing brushes and batch processing.
"""

from __future__ import annotations

import io
import math
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np
from flask import (Flask, abort, jsonify, render_template, request,
                   send_file, session)

from oilpaint import brushes as B
from oilpaint import decoration as D
from oilpaint import engine, params
from oilpaint import planner as PL
from oilpaint.presets import PresetStore
from actions import BRUSH_MODES, ActionEngine, ActionError, Stopped


def _user_data_dir() -> str:
    """Where a packaged desktop app keeps its own files (writable)."""
    if sys.platform == "win32":
        root = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        root = os.path.expanduser("~/Library/Application Support")
    else:
        root = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(root, "OilStudio")


FROZEN = getattr(sys, "frozen", False)          # True inside the .exe / .app
# templates and static files: next to app.py, or inside the packaged app
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
# presets, temp files: next to app.py when run from source, else a user folder
DATA_DIR = _user_data_dir() if FROZEN else BASE_DIR
WORK_DIR = os.environ.get("OILSTUDIO_WORKDIR") or os.path.join(DATA_DIR, "workdir")
os.makedirs(WORK_DIR, exist_ok=True)
# the action library: Documents/OilStudio Actions for the desktop app
ACTIONS_DIR = os.environ.get("OILSTUDIO_ACTIONS_DIR") or (
    os.path.join(os.path.expanduser("~"), "Documents", "OilStudio Actions") if FROZEN
    else os.path.join(BASE_DIR, "ps_actions"))

MAX_UPLOAD_MB = 30
MAX_SOURCE_SIDE = 4000
QUALITY = {"draft": 650, "normal": 1000, "high": 1500, "max": 2400}
# styles whose look lives in pixel-sized detail (grain, dot grid) are always
# made at the photo's full size, like Photoshop; Quality only affects drafts
FULL_SIZE_STYLES = ("Halftone", "Painted Photo", "Impression")


def _work_side(p, quality):
    if p.get("paint_style") in FULL_SIZE_STYLES:
        return MAX_SOURCE_SIDE
    return QUALITY[quality]
SESSION_TTL = 3 * 3600
MAX_SESSIONS = 24
UNDO_LEVELS = 20
# live preview: the GPU works on the painting at this size at most, and a
# slider that is being dragged re-plans the strokes at the smaller PROXY size
LIVE_SIDE = 1200
FULL_STYLE_LIVE_SIDE = 2400     # on-screen size for Halftone / Painted Photo
PROXY_SIDE = 520
# settings applied on top of the bare painting (no repainting needed)
FINISH_KEYS = ("relief", "lightening", "blend_original", "blend_mode")
DECO_PREFIX = ("light_", "canvas_", "frame_", "text_")
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# the desktop window keeps a persistent cache: make it re-check the scripts
# and styles, and give their URLs the file's time so an update always loads
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


# A protected build (build/protect.py) carries the templates, scripts and
# styles inside the compiled program instead of as loose files.
try:
    import _oilstudio_assets as _EMBEDDED          # only in protected builds
except ImportError:
    _EMBEDDED = None

if _EMBEDDED is not None:
    import mimetypes

    from jinja2 import DictLoader

    app.jinja_loader = DictLoader(_EMBEDDED.templates())

    def _embedded_static(filename):
        data = _EMBEDDED.static(filename)
        if data is None:
            abort(404)
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return app.response_class(data, mimetype=mime)

    app.view_functions["static"] = _embedded_static


@app.context_processor
def _asset_urls():
    from flask import url_for

    def asset(filename):
        if _EMBEDDED is not None:
            v = _EMBEDDED.VERSION
        else:
            try:
                v = int(os.path.getmtime(os.path.join(app.static_folder, filename)))
            except OSError:
                v = 0
        return url_for("static", filename=filename, v=v)
    return {"asset": asset}


@app.after_request
def _no_stale_assets(resp):
    if request.path.startswith("/static/") or request.path == "/":
        resp.headers["Cache-Control"] = "no-cache"
    return resp
app.secret_key = os.environ.get("OILSTUDIO_SECRET") or os.urandom(32)

presets = PresetStore(os.path.join(WORK_DIR, "presets.json"))

runner = ActionEngine(ACTIONS_DIR, cache_dir=os.path.join(WORK_DIR, "action-cache"))

_sessions: dict[str, "Doc"] = {}
_sessions_lock = threading.Lock()


# --------------------------------------------------------------------------
# session document
# --------------------------------------------------------------------------


class Doc:
    """Everything the workspace holds for one open image."""

    def __init__(self, sid: str):
        self.sid = sid
        self.lock = threading.RLock()
        self.touched = time.time()

        self.name = ""
        self.source = None        # uint8 BGR, full resolution as uploaded
        self.original = None      # uint8 BGR at working resolution
        self.painting = None      # float32 BGR, engine output
        self.edited = None        # float32 BGR, after the brush tools
        self.height = None        # float32 paint height, for the 3D brush
        self.decorated = None     # cached uint8 BGR with decoration applied
        self.deco_key = None

        self.shots: list[dict] = []
        self.selected = -1

        self.params = params.defaults()
        self.deco_order = ["Light", "Canvas", "Frame", "Text"]
        self.canvas_custom = None
        self.logo = None

        self.undo: list[np.ndarray] = []
        self.redo: list[np.ndarray] = []

        self.job = None           # {'progress','label','running','stop','error'}
        self.version = 0

        # -- layers kept for re-finishing and for the live preview ---------
        self.canvas_raw = None    # float32 BGR, the bare painting (no finishing)
        self.paint_height = None  # float32, paint height laid down by the strokes
        self.finish_steps = ()    # which finishing steps apply (engine.finish)
        self.fin_key = None       # finishing settings that self.painting reflects
        self.plan = None          # planner.Plan of the last Run
        self.ran_params = None    # the settings the last Run painted with
        self.token = uuid.uuid4().hex
        self.plan_lock = threading.Lock()
        self.live_cache = PL.AnalysisCache()
        self.proxy_cache = PL.AnalysisCache()
        self.run_cache = PL.AnalysisCache()
        self.gpu_pending = None

    # -- helpers ---------------------------------------------------------
    def finish_key(self, p=None):
        p = p or self.params
        return tuple(str(p.get(k)) for k in FINISH_KEYS)

    def set_layers(self, canvas, height, steps, plan=None, ran=None, p=None):
        """A new bare painting arrived (Run, Timeline shot, an action)."""
        self.canvas_raw = canvas
        self.paint_height = height
        self.finish_steps = tuple(steps or ())
        self.plan = plan
        self.ran_params = ran
        self.fin_key = self.finish_key(p)

    def refinish(self):
        """Redo relief / lightening / blending after those sliders moved.

        Strokes added with the post-processing brushes are kept: a pixel the
        brushes changed keeps its brushed colour, every other pixel takes the
        new finishing.  The Undo history is rebased the same way.
        """
        if self.canvas_raw is None or self.edited is None or self.painting is None:
            return False
        key = self.finish_key()
        if key == self.fin_key:
            return False
        new = engine.finish(self.canvas_raw, self.paint_height, self.params,
                            self.original.astype(np.float32) if self.original is not None else None,
                            self.finish_steps).astype(np.float32)
        old = self.painting

        def rebase(img):
            touched = np.abs(img - old).max(axis=2, keepdims=True) > 0.5
            return np.where(touched, img, new).astype(np.float32)

        self.edited = rebase(self.edited)
        self.undo = [rebase(u) for u in self.undo]
        self.redo = [rebase(r) for r in self.redo]
        self.painting = new
        self.fin_key = key
        self.decorated = None
        return True

    def edits_mask(self):
        """Where the post-processing brushes changed the painting (or None)."""
        if self.edited is None or self.painting is None:
            return None
        m = np.abs(self.edited - self.painting).max(axis=2) > 0.5
        return m if m.any() else None

    def display(self) -> np.ndarray:
        """Current After image with decoration applied (cached)."""
        if self.edited is None:
            return None
        self.refinish()
        key = (self.version, self.fin_key, tuple(self.deco_order),
               tuple(sorted((k, str(v)) for k, v in self.params.items()
                            if k.startswith(DECO_PREFIX))))
        if self.deco_key == key and self.decorated is not None:
            return self.decorated
        out = D.decorate(self.edited, self.params, self.deco_order,
                         self.canvas_custom, self.logo)
        self.decorated = np.clip(out, 0, 255).astype(np.uint8)
        self.deco_key = key
        return self.decorated

    def push_undo(self):
        if self.edited is None:
            return
        self.undo.append(self.edited.copy())
        if len(self.undo) > UNDO_LEVELS:
            self.undo.pop(0)
        self.redo.clear()

    def bump(self):
        self.version += 1
        self.decorated = None


def _reap():
    now = time.time()
    with _sessions_lock:
        dead = [k for k, d in _sessions.items() if now - d.touched > SESSION_TTL]
        for k in dead:
            _sessions.pop(k, None)
        if len(_sessions) > MAX_SESSIONS:
            for k in sorted(_sessions, key=lambda k: _sessions[k].touched)[
                    :len(_sessions) - MAX_SESSIONS]:
                _sessions.pop(k, None)


def doc(create=False) -> Doc:
    _reap()
    sid = session.get("sid")
    if not sid:
        if not create:
            abort(409, "No image is open.")
        sid = uuid.uuid4().hex
        session["sid"] = sid
    with _sessions_lock:
        d = _sessions.get(sid)
        if d is None:
            if not create:
                abort(409, "No image is open.")
            d = Doc(sid)
            _sessions[sid] = d
    d.touched = time.time()
    return d


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------


def decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("That file could not be read as an image.")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SOURCE_SIDE:
        s = MAX_SOURCE_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def fit(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= side:
        return img.copy()
    s = side / max(h, w)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def as_jpeg(img: np.ndarray, q=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        abort(500, "Could not encode the image.")
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


def safe_name(name: str) -> str:
    keep = "".join(c for c in os.path.basename(name or "")
                   if c.isalnum() or c in "._- ")
    return keep[:80] or "image"


# --------------------------------------------------------------------------
# routes: workspace
# --------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template(
        "index.html",
        schema=params.NUMERIC,
        choices=params.CHOICES,
        defaults=params.defaults(),
        blend_modes=params.BLEND_MODES,
        tools=B.TOOLS,
        quality=list(QUALITY),
    )


@app.post("/api/open")
def api_open():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="Choose an image to open."), 400
    if os.path.splitext(f.filename)[1].lower() not in ALLOWED:
        return jsonify(error="Unsupported file type."), 400
    try:
        img = decode(f.read())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    d = doc(create=True)
    with d.lock:
        d.name = safe_name(f.filename)
        d.source = img
        d.original = None
        d.painting = d.edited = d.height = None
        d.set_layers(None, None, ())
        d.token = uuid.uuid4().hex
        d.live_cache.reset()
        d.proxy_cache.reset()
        d.shots.clear()
        d.selected = -1
        d.undo.clear()
        d.redo.clear()
        d.job = None
        d.bump()
    h, w = img.shape[:2]
    return jsonify(ok=True, name=d.name, width=w, height=h)


@app.post("/api/run")
def api_run():
    d = doc()
    if d.source is None:
        return jsonify(error="Open an image first."), 400
    with d.lock:
        if d.job and d.job["running"]:
            return jsonify(error="A painting is already in progress."), 409

        body = request.get_json(silent=True) or {}
        p = params.clean(body.get("params"))
        d.params = p
        d.deco_order = _clean_order(body.get("deco_order"))
        quality = body.get("quality") if body.get("quality") in QUALITY else "normal"
        shots = int(np.clip(int(body.get("snapshots", 8) or 8), 0, 16))

        work = fit(d.source, _work_side(p, quality))
        if d.original is None or d.original.shape != work.shape:
            d.live_cache.reset()
            d.proxy_cache.reset()
        d.original = work
        pinned = [s for s in d.shots if s.get("pinned")]
        d.shots = pinned
        d.selected = -1
        d.job = {"progress": 0.0, "label": "Starting", "running": True,
                 "stop": False, "error": None, "strokes": 0}

    job = d.job

    def worker():
        try:
            def progress(frac, label):
                job["progress"] = float(frac)
                job["label"] = label

            def on_shot(idx, image):
                thumb = fit(np.clip(image, 0, 255).astype(np.uint8), 240)
                with d.lock:
                    d.shots.append({"image": np.clip(image, 0, 255).astype(np.uint8),
                                    "thumb": thumb, "pinned": False,
                                    "manual": False, "id": uuid.uuid4().hex})

            result, _, meta = engine.paint(
                work, p, progress=progress, snapshots=shots,
                on_snapshot=on_shot, should_stop=lambda: job["stop"])

            with d.lock:
                d.painting = result.astype(np.float32)
                d.edited = d.painting.copy()
                d.height = np.zeros(result.shape[:2], np.float32)
                d.set_layers(meta.get("canvas"), meta.get("height"),
                             meta.get("finish", ()), meta.get("plan"), ran=p, p=p)
                d.undo.clear()
                d.redo.clear()
                d.selected = len(d.shots) - 1
                d.bump()
            job["strokes"] = meta["strokes"]
            job["seconds"] = meta["seconds"]
            job["stopped"] = meta["stopped"]
            if meta.get("style") in params.CPU_STYLES:
                job["engine"] = {"Canvas Oil": "canvas_oil", "Halftone": "halftone",
                                 "Painted Photo": "painted", "Impression": "impression"
                                 }.get(meta["style"], "canvas_oil")
        except Exception as exc:                      # noqa: BLE001
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["running"] = False
            job["progress"] = 1.0

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(ok=True)


# --------------------------------------------------------------------------
# routes: Run on the GPU (static/js/live.js paints the strokes)
# --------------------------------------------------------------------------


@app.post("/api/gpu/plan")
def api_gpu_plan():
    """Step 1 of a GPU Run: plan every stroke at the working size.

    The browser paints the plan on the graphics card and sends the bare
    painting and its paint height back to /api/gpu/result.
    """
    d = doc()
    if d.source is None:
        return jsonify(error="Open an image first."), 400
    body = request.get_json(silent=True) or {}
    p = params.clean(body.get("params"))
    if p.get("paint_style") in params.CPU_STYLES:
        return jsonify(error=f"{p['paint_style']} is made on the CPU."), 400
    quality = body.get("quality") if body.get("quality") in QUALITY else "normal"
    with d.lock:
        if d.job and d.job["running"]:
            return jsonify(error="A painting is already in progress."), 409
        work = fit(d.source, QUALITY[quality])
        if d.original is None or d.original.shape != work.shape:
            d.live_cache.reset()
            d.proxy_cache.reset()
        d.original = work
        token = d.token
    t0 = time.time()
    with d.plan_lock:
        plan = PL.plan_preview(work, p, d.run_cache, token=(token, work.shape))
    with d.lock:
        d.gpu_pending = {"plan": plan, "params": p, "shape": work.shape,
                         "order": _clean_order(body.get("deco_order")), "t0": t0}
    return _binary(PL.encode(plan, scale=1.0, ground_side=1200),
                   {"X-Plan-Seconds": f"{time.time() - t0:.3f}"})


@app.post("/api/gpu/result")
def api_gpu_result():
    """Step 2 of a GPU Run: the bare painting (PNG), its paint height
    (float16) and the Timeline shots (JPEG) painted by the browser."""
    d = doc()
    with d.lock:
        pend = getattr(d, "gpu_pending", None)
    if not pend:
        return jsonify(error="Plan the painting first."), 409
    h, w = pend["shape"][:2]
    f = request.files.get("canvas")
    hb = request.files.get("height")
    if f is None or hb is None:
        return jsonify(error="Missing the painted layers."), 400
    canvas = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    raw = np.frombuffer(hb.read(), "<f2")
    if canvas is None or canvas.shape[:2] != (h, w) or raw.size != h * w:
        return jsonify(error="The painted layers do not match the picture."), 400
    canvas = canvas.astype(np.float32)
    height = raw.astype(np.float32).reshape(h, w)
    p = pend["params"]
    try:
        gpu_seconds = float(request.form.get("seconds", "0"))
    except ValueError:
        gpu_seconds = 0.0

    with d.lock:
        src = d.original.astype(np.float32)
        result = engine.finish(canvas, height, p, src, engine.FINISH_ALL)
        d.params = p
        d.deco_order = pend["order"]
        d.shots = [s for s in d.shots if s.get("pinned")]
        for sf in request.files.getlist("shots")[:16]:
            img = cv2.imdecode(np.frombuffer(sf.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (h, w):
                continue
            # the paint height is only known at the end; the shots get the
            # lightening and blending, like the CPU's quick Timeline shots
            shot = np.clip(engine.finish(img.astype(np.float32), None, p, src,
                                         ("lightening", "blend")), 0, 255).astype(np.uint8)
            d.shots.append({"image": shot, "thumb": fit(shot, 240), "pinned": False,
                            "manual": False, "id": uuid.uuid4().hex})
        final = np.clip(result, 0, 255).astype(np.uint8)
        d.shots.append({"image": final, "thumb": fit(final, 240), "pinned": False,
                        "manual": False, "id": uuid.uuid4().hex})
        d.painting = result.astype(np.float32)
        d.edited = d.painting.copy()
        d.height = np.zeros((h, w), np.float32)
        d.set_layers(canvas, height, engine.FINISH_ALL, pend["plan"], ran=p, p=p)
        d.undo.clear()
        d.redo.clear()
        d.selected = len(d.shots) - 1
        d.job = None
        d.gpu_pending = None
        d.bump()
        seconds = round(time.time() - pend["t0"], 2)
    return jsonify(ok=True, version=d.version, strokes=pend["plan"].count,
                   seconds=seconds, gpu_seconds=round(gpu_seconds, 2))


def _clean_order(raw):
    want = ["Light", "Canvas", "Frame", "Text"]
    if not isinstance(raw, list):
        return want
    out = [x for x in raw if x in want]
    return out + [x for x in want if x not in out] if out else want


@app.get("/api/progress")
def api_progress():
    d = doc()
    j = d.job or {"progress": 0.0, "label": "", "running": False, "error": None}
    with d.lock:
        shots = [{"id": s["id"], "pinned": s["pinned"], "manual": s["manual"]}
                 for s in d.shots]
        sel = d.selected
        ver = d.version
    return jsonify(skipped=j.get("skipped") or [], action=j.get("action"),
                   progress=round(float(j["progress"]), 3), label=j["label"],
                   running=bool(j["running"]), error=j["error"],
                   strokes=j.get("strokes", 0), seconds=j.get("seconds"),
                   stopped=j.get("stopped", False), engine=j.get("engine", "paint"),
                   shots=shots, selected=sel, version=ver)


@app.post("/api/stop")
def api_stop():
    d = doc()
    if d.job:
        d.job["stop"] = True
    return jsonify(ok=True)


# --------------------------------------------------------------------------
# routes: images
# --------------------------------------------------------------------------


@app.get("/img/before")
def img_before():
    d = doc()
    img = d.original if d.original is not None else d.source
    if img is None:
        abort(404)
    return as_jpeg(fit(img, 1600), 92)


@app.get("/img/after")
def img_after():
    d = doc()
    with d.lock:
        out = d.display()
    if out is None:
        abort(404)
    return as_jpeg(fit(out, max(1600, _live_side(d))), 92)


@app.get("/img/shot/<shot_id>")
def img_shot(shot_id):
    d = doc()
    with d.lock:
        for s in d.shots:
            if s["id"] == shot_id:
                return as_jpeg(s["thumb"], 82)
    abort(404)


@app.post("/api/decoration")
def api_decoration():
    """Update the settings applied on top of the painting - finishing
    (relief, lightening, blend) and decoration.  No repainting needed.

    With ``render: true`` the finished After image comes straight back as a
    JPEG (the live preview for browsers without WebGL 2)."""
    d = doc()
    body = request.get_json(silent=True) or {}
    raw = body.get("params") or {}
    with d.lock:
        merged = dict(d.params)
        merged.update({k: v for k, v in raw.items()
                       if k in FINISH_KEYS or k.startswith(DECO_PREFIX)})
        d.params = params.clean(dict(merged, guides=d.params.get("guides")))
        d.deco_order = _clean_order(body.get("deco_order"))
        if body.get("render"):
            out = d.display()
            if out is None:
                abort(409, "Nothing has been painted yet.")
            return as_jpeg(fit(out, 1600), 88)
    return jsonify(ok=True, version=d.version)


@app.post("/api/texture")
def api_texture():
    """Load a custom canvas sample, or a logo for the Text tab."""
    d = doc()
    kind = request.form.get("kind", "canvas")
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="Choose an image."), 400
    try:
        img = fit(decode(f.read()), 1400)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    with d.lock:
        if kind == "logo":
            d.logo = img
        else:
            d.canvas_custom = img
            d.params["canvas_texture"] = "Custom"
        d.decorated = None
    return jsonify(ok=True)


# --------------------------------------------------------------------------
# routes: timeline
# --------------------------------------------------------------------------


@app.post("/api/shot/select")
def api_shot_select():
    d = doc()
    shot_id = (request.get_json(silent=True) or {}).get("id")
    with d.lock:
        for i, s in enumerate(d.shots):
            if s["id"] == shot_id:
                d.selected = i
                d.painting = s["image"].astype(np.float32)
                d.edited = d.painting.copy()
                d.height = np.zeros(d.painting.shape[:2], np.float32)
                d.set_layers(d.painting.copy(), None, ())
                d.undo.clear()
                d.redo.clear()
                d.bump()
                return jsonify(ok=True, version=d.version)
    return jsonify(error="That shot is gone."), 404


@app.post("/api/shot/snapshot")
def api_shot_snapshot():
    """Take a manual shot of the painting as it stands."""
    d = doc()
    with d.lock:
        if d.edited is None:
            return jsonify(error="Nothing to capture yet."), 400
        img = np.clip(d.edited, 0, 255).astype(np.uint8)
        d.shots.append({"image": img, "thumb": fit(img, 240), "pinned": True,
                        "manual": True, "id": uuid.uuid4().hex})
        d.selected = len(d.shots) - 1
    return jsonify(ok=True)


@app.post("/api/shot/pin")
def api_shot_pin():
    d = doc()
    shot_id = (request.get_json(silent=True) or {}).get("id")
    with d.lock:
        for s in d.shots:
            if s["id"] == shot_id:
                s["pinned"] = not s["pinned"]
                return jsonify(ok=True, pinned=s["pinned"])
    return jsonify(error="That shot is gone."), 404


@app.post("/api/shot/delete")
def api_shot_delete():
    d = doc()
    shot_id = (request.get_json(silent=True) or {}).get("id")
    with d.lock:
        for i, s in enumerate(d.shots):
            if s["id"] == shot_id:
                if s["pinned"]:
                    return jsonify(error="Pinned shots cannot be deleted."), 400
                d.shots.pop(i)
                d.selected = min(d.selected, len(d.shots) - 1)
                return jsonify(ok=True)
    return jsonify(error="That shot is gone."), 404


# --------------------------------------------------------------------------
# routes: post-processing brushes
# --------------------------------------------------------------------------


@app.post("/api/brush")
def api_brush():
    d = doc()
    body = request.get_json(silent=True) or {}
    tool = body.get("tool")
    if tool not in B.TOOLS:
        return jsonify(error="Unknown tool."), 400

    with d.lock:
        if d.edited is None:
            return jsonify(error="Paint the picture first."), 400
        h, w = d.edited.shape[:2]
        pts = []
        for pt in (body.get("path") or [])[:4000]:
            try:
                x, y = float(pt[0]), float(pt[1])
            except (TypeError, ValueError, IndexError):
                continue
            pts.append((x * w, y * h))
        if len(pts) < 1:
            return jsonify(error="Empty stroke."), 400

        opts = body.get("options") or {}

        def num(key, lo, hi, dflt):
            try:
                return float(np.clip(float(opts.get(key, dflt)), lo, hi))
            except (TypeError, ValueError):
                return float(dflt)

        def flag(key, dflt=False):
            return bool(opts.get(key, dflt))

        scale = max(h, w) / 1000.0
        d.push_undo()

        if tool == "smudge":
            B.smudge(d.edited, pts, size=num("size", 1, 300, 40) * scale,
                     hardness=num("hardness", 0, 100, 50),
                     strength=num("strength", 1, 100, 50))
        elif tool == "history":
            base = d.original.astype(np.float32)
            B.history_brush(d.edited, base, pts,
                            size=num("size", 1, 500, 60) * scale,
                            hardness=num("hardness", 0, 100, 50),
                            strength=num("strength", 1, 100, 60),
                            restore_original=flag("restore_original", True),
                            snapshot=d.painting)
        elif tool == "oil":
            colour = params.hex_to_bgr(
                opts.get("colour") if params.is_hex(str(opts.get("colour")))
                else "#c8912f")
            B.oil_brush(d.edited, d.height, pts, colour,
                        size=num("size", 2, 150, 40) * scale,
                        relief=num("relief", 0, 100, 50),
                        reserve=num("reserve", 0, 100, 50),
                        mixing=flag("mixing", True),
                        cleaning=flag("cleaning", True),
                        flat=flag("flat", False),
                        seed=int(time.time() * 1000) % 100000)
        else:
            B.brush_3d(d.edited, d.height, pts,
                       size=num("size", 2, 300, 60) * scale,
                       smearing=num("smearing", 0, 100, 40),
                       relief=num("relief", 0, 100, 50),
                       flat=flag("flat", False),
                       seed=int(time.time() * 1000) % 100000)

        np.clip(d.edited, 0, 255, out=d.edited)
        d.bump()
    return jsonify(ok=True, version=d.version)


@app.post("/api/undo")
def api_undo():
    d = doc()
    with d.lock:
        if not d.undo:
            return jsonify(error="Nothing to undo."), 400
        d.redo.append(d.edited.copy())
        d.edited = d.undo.pop()
        d.bump()
    return jsonify(ok=True, version=d.version)


@app.post("/api/redo")
def api_redo():
    d = doc()
    with d.lock:
        if not d.redo:
            return jsonify(error="Nothing to redo."), 400
        d.undo.append(d.edited.copy())
        d.edited = d.redo.pop()
        d.bump()
    return jsonify(ok=True, version=d.version)


# --------------------------------------------------------------------------
# routes: presets
# --------------------------------------------------------------------------


@app.get("/api/presets")
def api_presets():
    return jsonify(**presets.names())


@app.get("/api/preset/<path:name>")
def api_preset_get(name):
    values = presets.get(name)
    if values is None:
        return jsonify(error="No such preset."), 404
    return jsonify(params=params.clean(values))


@app.post("/api/preset")
def api_preset_save():
    body = request.get_json(silent=True) or {}
    try:
        name = presets.save(body.get("name", ""), params.clean(body.get("params")))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(ok=True, name=name)


@app.delete("/api/preset/<path:name>")
def api_preset_delete(name):
    if presets.delete(name):
        return jsonify(ok=True)
    return jsonify(error="No such preset."), 404


# --------------------------------------------------------------------------
# routes: export and batch
# --------------------------------------------------------------------------


@app.get("/api/export")
def api_export():
    d = doc()
    fmt = (request.args.get("format") or "jpg").lower()
    with d.lock:
        out = d.display()
    if out is None:
        abort(409, "Nothing has been painted yet.")
    stem = os.path.splitext(d.name)[0] or "painting"
    if fmt == "png":
        ok, buf = cv2.imencode(".png", out)
        mime, ext = "image/png", "png"
    elif fmt in ("tif", "tiff"):
        ok, buf = cv2.imencode(".tif", out)
        mime, ext = "image/tiff", "tif"
    else:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mime, ext = "image/jpeg", "jpg"
    if not ok:
        abort(500, "Could not encode the image.")
    return send_file(io.BytesIO(buf.tobytes()), mimetype=mime,
                     as_attachment=True,
                     download_name=f"{stem}_oil.{ext}")


@app.post("/api/batch")
def api_batch():
    """Convert a series of images with one set of settings."""
    files = request.files.getlist("images")
    if not files:
        return jsonify(error="Choose some images."), 400
    files = files[:24]

    import json as _json
    p = params.clean(_json.loads(request.form.get("params") or "{}"))
    order = _clean_order(_json.loads(request.form.get("deco_order") or "null"))
    q = request.form.get("quality")
    side = _work_side(p, q if q in QUALITY else "normal")
    fmt = (request.form.get("format") or "jpg").lower()

    buf = io.BytesIO()
    done, failed = 0, []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            name = safe_name(f.filename)
            if os.path.splitext(name)[1].lower() not in ALLOWED:
                failed.append(name)
                continue
            try:
                img = fit(decode(f.read()), side)
                res, _, _ = engine.paint(img, p, snapshots=0)
                res = D.decorate(res, p, order)
                res = np.clip(res, 0, 255).astype(np.uint8)
                ext = "png" if fmt == "png" else "jpg"
                flags = ([] if ext == "png"
                         else [cv2.IMWRITE_JPEG_QUALITY, 95])
                ok, enc = cv2.imencode("." + ext, res, flags)
                if not ok:
                    failed.append(name)
                    continue
                zf.writestr(f"{os.path.splitext(name)[0]}_oil.{ext}",
                            enc.tobytes())
                done += 1
            except Exception:                          # noqa: BLE001
                failed.append(name)
    if not done:
        return jsonify(error="None of those images could be processed."), 400
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="oil_paintings.zip")

# --------------------------------------------------------------------------
# routes: live preview (static/js/live.js)
# --------------------------------------------------------------------------


def _live_side(d: Doc) -> int:
    """Grain and fine dots need a sharper on-screen picture."""
    if (d.ran_params or {}).get("paint_style") in FULL_SIZE_STYLES:
        return FULL_STYLE_LIVE_SIDE
    return LIVE_SIDE


def _live_dims(d: Doc):
    """(w, h) of the live preview and its scale from the working image."""
    base = d.original if d.original is not None else d.source
    h, w = base.shape[:2]
    side = _live_side(d)
    if max(h, w) <= side:
        return (w, h), 1.0
    s = side / max(h, w)
    return (max(1, int(w * s)), max(1, int(h * s))), s


def _to_live(img: np.ndarray, d: Doc, interp=cv2.INTER_AREA) -> np.ndarray:
    (w, h), _ = _live_dims(d)
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=interp)


def _png(img: np.ndarray):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        abort(500, "Could not encode the image.")
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


def _binary(data: bytes, headers=None):
    resp = app.response_class(data, mimetype="application/octet-stream")
    for k, v in (headers or {}).items():
        resp.headers[k] = v
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _relief_scale(height: np.ndarray) -> float:
    """engine.apply_relief normalises the lit relief by this percentile."""
    hgt = cv2.GaussianBlur(height, (0, 0), 0.8)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3)
    emb = (gx + gy) * -0.7071
    return float(np.percentile(np.abs(emb), 97.0)) + 1e-6


@app.get("/api/live/state")
def api_live_state():
    """What the live preview needs to know about the open painting."""
    d = doc()
    with d.lock:
        if d.edited is None:
            return jsonify(painted=False)
        d.refinish()
        (w, h), s = _live_dims(d)
        style = (d.ran_params or {}).get("paint_style")
        ran = None
        if d.ran_params is not None:
            ran = {k: v for k, v in d.ran_params.items()}
        return jsonify(
            painted=True, version=d.version, size=[w, h], scale=s,
            work=[d.edited.shape[1], d.edited.shape[0]],
            finish=list(d.finish_steps), has_height=d.paint_height is not None,
            has_plan=d.plan is not None, has_edits=d.edits_mask() is not None,
            style=style, ran=ran, token=d.token)


@app.get("/api/live/layer/<name>")
def api_live_layer(name):
    """The bare painting, its paint height, the original and the brush edits,
    at the live preview size."""
    d = doc()
    with d.lock:
        if d.edited is None:
            abort(409, "Nothing has been painted yet.")
        d.refinish()
        if name == "canvas":
            img = d.canvas_raw if d.canvas_raw is not None else d.painting
            return _png(_to_live(np.clip(img, 0, 255).astype(np.uint8), d))
        if name == "original":
            return _png(_to_live(d.original, d))
        if name == "height":
            if d.paint_height is None:
                return ("", 204)
            hgt = _to_live(d.paint_height, d)
            return _binary(np.ascontiguousarray(hgt, "<f4").tobytes(),
                           {"X-Relief-Scale": f"{_relief_scale(hgt):.6f}"})
        if name == "edits":
            m = d.edits_mask()
            if m is None:
                return ("", 204)
            rgba = np.dstack([np.clip(d.edited, 0, 255).astype(np.uint8),
                              (m * 255).astype(np.uint8)])
            return _png(_to_live(rgba, d, cv2.INTER_LINEAR))
    abort(404)


@app.post("/api/live/plan")
def api_live_plan():
    """Strokes for the GPU.

    ``source: "run"``      the strokes of the last Run (to restyle them live)
    ``source: "preview"``  plan the painting for the given settings - at the
                           small proxy size while a slider is dragged
                           (``proxy: true``), at the live size otherwise.
    """
    d = doc()
    body = request.get_json(silent=True) or {}
    (lw, lh), _ = _live_dims(d)
    if body.get("source") == "run":
        with d.lock:
            plan = d.plan
        if plan is None:
            return ("", 204)
        return _binary(PL.encode(plan, scale=lw / plan.size[0], ground_side=700))

    p = params.clean(body.get("params"))
    with d.lock:
        base = d.original if d.original is not None else d.source
        if base is None:
            abort(409, "No image is open.")
        token = d.token
    proxy = bool(body.get("proxy"))
    side = PROXY_SIDE if proxy else LIVE_SIDE
    src = fit(base, side)
    t0 = time.time()
    with d.plan_lock:
        plan = PL.plan_preview(src, p, d.proxy_cache if proxy else d.live_cache,
                               token=(token, src.shape), px_scale=src.shape[1] / base.shape[1])
    return _binary(PL.encode(plan, scale=lw / src.shape[1], ground_side=700),
                   {"X-Plan-Seconds": f"{time.time() - t0:.3f}"})


@app.post("/api/live/draft")
def api_live_draft():
    """A quick server-side draft of the painting (Canvas Oil sliders, and
    browsers without WebGL 2).  Returns the bare painting as a PNG."""
    d = doc()
    body = request.get_json(silent=True) or {}
    p = params.clean(body.get("params"))
    with d.lock:
        base = d.original if d.original is not None else d.source
        if base is None:
            abort(409, "No image is open.")
        token = d.token
    proxy = bool(body.get("proxy"))
    src = fit(base, 360 if proxy else PROXY_SIDE)
    with d.plan_lock:
        out = engine.paint_draft(src, p, d.proxy_cache, token=(token, src.shape))
    (lw, lh), _ = _live_dims(d)
    out = cv2.resize(np.clip(out, 0, 255).astype(np.uint8), (lw, lh),
                     interpolation=cv2.INTER_CUBIC)
    return _png(out)


@app.post("/api/live/overlay")
def api_live_overlay():
    """Decoration layers the GPU composites: the frame, the text, the logo
    (RGBA PNG) and the canvas surface (float32 height map)."""
    d = doc()
    body = request.get_json(silent=True) or {}
    kind = body.get("kind")
    p = params.clean(dict(d.params, **(body.get("params") or {})))
    try:
        w, h = int(body["size"][0]), int(body["size"][1])
    except (KeyError, TypeError, ValueError, IndexError):
        abort(400, "Missing size.")
    if not (1 <= w <= 6000 and 1 <= h <= 6000):
        abort(400, "Bad size.")

    def rgba_png(layer):
        out = np.dstack([np.clip(layer[:, :, :3], 0, 255),
                         np.clip(layer[:, :, 3] * 255.0, 0, 255)]).astype(np.uint8)
        return _png(out)

    if kind == "frame":
        return rgba_png(D.frame_layer(h, w, p))
    if kind == "text":
        return rgba_png(D.text_layer(h, w, p))
    if kind == "logo":
        with d.lock:
            logo = d.logo
        lg = D.logo_layer(h, w, p, logo)
        return ("", 204) if lg is None else rgba_png(lg)
    if kind == "surface":
        with d.lock:
            custom = d.canvas_custom if p["canvas_texture"] == "Custom" else None
        hm = D.canvas_surface(h, w, p, custom)
        # Sobel of the surface, and the percentile apply_canvas normalises
        # by, for every light angle (the Light Angle slider is live)
        gx = cv2.Sobel(hm, cv2.CV_32F, 1, 0, ksize=3)[::3, ::3].ravel()
        gy = cv2.Sobel(hm, cv2.CV_32F, 0, 1, ksize=3)[::3, ::3].ravel()
        scales = []
        for i in range(73):
            a = math.radians(45.0 + 270.0 * i / 72.0)
            scales.append(float(np.percentile(np.abs(gx * math.cos(a) + gy * math.sin(a)), 98)) + 1e-6)
        return _binary(np.ascontiguousarray(hm, "<f4").tobytes(),
                       {"X-Scales": ",".join(f"{v:.6f}" for v in scales)})
    abort(400, "Unknown layer.")


# --------------------------------------------------------------------------
# routes: actions (.atn) played inside OilStudio - no Photoshop needed
# --------------------------------------------------------------------------


def _act_pack(pack_id):
    pack = runner.library.get(pack_id or "")
    if pack is None:
        abort(404, "That action pack is not in the library.")
    if pack.get("error"):
        abort(409, f"{pack['name']}: {pack['error']}")
    return pack


def _act_body():
    body = request.get_json(silent=True) or {}
    pack = _act_pack(body.get("pack"))
    action = body.get("action") or None
    brush = body.get("brush") if body.get("brush") in BRUSH_MODES else pack["brush"]
    try:
        long_side = int(np.clip(int(body.get("long_side", pack["long_side"])), 0, 8000))
    except (TypeError, ValueError):
        long_side = int(pack["long_side"])
    return pack, action, brush, long_side


def _skipped(res) -> list[dict]:
    return [{"step": s.index, "name": s.title, "reason": s.reason} for s in res.skipped]


@app.get("/api/actions/packs")
def api_act_packs():
    return jsonify(packs=[runner.library.public(p) for p in runner.library.packs()],
                   folder=runner.library.root)


@app.post("/api/actions/open-folder")
def api_act_open_folder():
    """Show the action library in Explorer / Finder (only from this computer)."""
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return jsonify(error="Only available on the computer running OilStudio."), 403
    folder = runner.library.root
    os.makedirs(folder, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(folder)                          # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except OSError as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(ok=True, folder=folder)


MAX_PACK_UPLOAD_MB = 1024           # Envato pattern (.pat) / brush (.abr) files can be big


@app.before_request
def _bigger_limit_for_packs():
    if request.path == "/api/actions/add-pack":
        try:
            request.max_content_length = MAX_PACK_UPLOAD_MB * 1024 * 1024   # Flask >= 3.1
        except AttributeError:
            pass


@app.post("/api/actions/add-pack")
def api_act_add_pack():
    """Add an action pack to the library from uploaded .atn/.abr/.pat/... or an Envato .zip."""
    files = [(f.filename, f.stream) for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify(error="Choose the action files (.atn, or the Envato .zip)."), 400
    try:
        pack = runner.library.add_pack(files, name=request.form.get("name", ""),
                                        category=request.form.get("category", ""),
                                        brush=request.form.get("brush", ""))
    except (ValueError, zipfile.BadZipFile) as exc:
        return jsonify(error=str(exc) if isinstance(exc, ValueError)
                       else "That .zip file could not be opened."), 400
    except OSError as exc:
        return jsonify(error=f"Could not save the pack: {exc}"), 500
    info = runner.library.public(pack)
    try:
        info["scan"] = runner.scan(pack)
    except ActionError as exc:
        info["scan_error"] = str(exc)
    return jsonify(ok=True, pack=info)


@app.post("/api/actions/remove-pack")
def api_act_remove_pack():
    """Remove a pack folder from the library (the original Envato files stay)."""
    pack_id = (request.get_json(silent=True) or {}).get("pack") or ""
    try:
        removed = runner.library.remove_pack(pack_id)
    except OSError as exc:
        return jsonify(error=f"Could not remove the pack: {exc}"), 500
    if not removed:
        return jsonify(error="That action pack is not in the library."), 404
    return jsonify(ok=True)


@app.get("/actions/preview/<pack_id>")
def act_preview(pack_id):
    pack = runner.library.get(pack_id)
    if not pack or not pack.get("preview"):
        abort(404)
    return send_file(pack["preview"])


@app.post("/api/actions/scan")
def api_act_scan():
    """How many steps of the action OilStudio plays exactly / closely / not at all."""
    body = request.get_json(silent=True) or {}
    pack = _act_pack(body.get("pack"))
    try:
        return jsonify(ok=True, **runner.scan(pack, body.get("action") or None))
    except ActionError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/actions/run")
def api_act_run():
    """Play the action on the open photo in the background (progress via /api/progress)."""
    d = doc()
    if d.source is None:
        return jsonify(error="Open an image first."), 400
    pack, action, brush, long_side = _act_body()
    name = action or pack["action"]
    with d.lock:
        if d.job and d.job["running"]:
            return jsonify(error="Something is already running."), 409
        d.job = {"progress": 0.02, "label": f"Starting {name}", "running": True, "stop": False,
                 "error": None, "strokes": 0, "engine": "action", "action": name}
    job = d.job
    src = cv2.cvtColor(d.source, cv2.COLOR_BGR2RGB)

    def progress(i, n, title):
        if job["stop"]:
            raise Stopped()
        job["progress"] = 0.03 + 0.92 * i / max(n, 1)
        job["label"] = f"{name}: step {i} of {n} — {title}" if i else title

    def worker():
        t0 = time.time()
        try:
            res = runner.run(pack, src, action=action, brush=brush, long_side=long_side,
                              progress=progress)
            job["skipped"] = _skipped(res)[:50]
            job["approximated"] = res.approximated
            job["progress"], job["label"] = 0.97, "Bringing the result in"
            result = cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR)
            with d.lock:
                h, w = result.shape[:2]
                # Before and After must match in size for the History brush
                d.original = cv2.resize(d.source, (w, h), interpolation=cv2.INTER_AREA)
                d.painting = result.astype(np.float32)
                d.edited = d.painting.copy()
                d.height = np.zeros((h, w), np.float32)
                d.set_layers(d.painting.copy(), None, ())
                d.live_cache.reset()
                d.proxy_cache.reset()
                d.undo.clear()
                d.redo.clear()
                d.shots.append({"image": result, "thumb": fit(result, 240), "pinned": False,
                                "manual": False, "id": uuid.uuid4().hex})
                d.selected = len(d.shots) - 1
                d.bump()
            job["seconds"] = round(time.time() - t0, 1)
            job["stopped"] = False
        except Stopped:
            job["error"] = "Stopped — the photo was not changed."
        except ActionError as exc:
            job["error"] = str(exc)
        except Exception as exc:                      # noqa: BLE001
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["running"] = False
            job["progress"] = 1.0

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(ok=True)


@app.post("/api/actions/batch")
def api_act_batch():
    """Run one action on many photos, one after another, and return a zip."""
    files = request.files.getlist("images")[:24]
    if not files:
        return jsonify(error="Choose some images."), 400
    pack = _act_pack(request.form.get("pack"))
    action = request.form.get("action") or None
    brush = request.form.get("brush")
    if brush not in BRUSH_MODES:
        brush = pack["brush"]
    try:
        long_side = int(np.clip(int(request.form.get("long_side", pack["long_side"])), 0, 8000))
    except (TypeError, ValueError):
        long_side = int(pack["long_side"])
    fmt = "png" if request.form.get("format") == "png" else "jpg"

    buf = io.BytesIO()
    done, failed = 0, []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            name = safe_name(f.filename)
            if os.path.splitext(name)[1].lower() not in ALLOWED:
                failed.append(name)
                continue
            try:
                rgb = cv2.cvtColor(decode(f.read()), cv2.COLOR_BGR2RGB)
                res = runner.run(pack, rgb, action=action, brush=brush, long_side=long_side)
                out = cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR)
                params_ = [cv2.IMWRITE_JPEG_QUALITY, 95] if fmt == "jpg" else []
                ok, data = cv2.imencode("." + fmt, out, params_)
                if not ok:
                    raise ActionError("could not encode the result")
                zf.writestr(f"{os.path.splitext(name)[0]}_{pack['id']}.{fmt}", data.tobytes())
                done += 1
                for sk in res.skipped:
                    failed.append(f"{name}: skipped step {sk.index} \"{sk.title}\" - {sk.reason}")
            except Exception as exc:                  # noqa: BLE001
                failed.append(f"{name}: {exc}")
        if failed:
            zf.writestr("errors.txt", "\n".join(failed))
    if not done:
        return jsonify(error="The action could not process those images. "
                             + (failed[0] if failed else "")), 500
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"{pack['id']}_actions.zip")


@app.errorhandler(413)
def too_large(_):
    limit = MAX_PACK_UPLOAD_MB if request.path == "/api/actions/add-pack" else MAX_UPLOAD_MB
    return jsonify(error=f"That upload is larger than {limit} MB."), 413


@app.errorhandler(409)
def conflict(exc):
    return jsonify(error=getattr(exc, "description", "Conflict")), 409


@app.errorhandler(404)
def not_found(exc):
    if request.path.startswith("/api/"):
        return jsonify(error=getattr(exc, "description", "Not found")), 404
    return exc


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            threaded=True, debug=False)
