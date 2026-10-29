"""
Canvas Oil on the GPU, all takes at once.

A Canvas Oil take has a CPU half before the brush strokes (paint patches,
the stroke plan) and one after them (rims, canvas, sketch lines, scratches,
speckles); the strokes in between are painted by the graphics card.  The
CPU halves of every take run side by side in worker processes (like a CPU
Run), so while the browser paints take 1 the next takes are being cut out.

The page and the workers talk through files in a job folder:
    image.npy            the photo (written once)
    <i>.plan / <i>.none  take i's stroke plan (or: no strokes)   <- worker
    <i>.brush            the stroke settings for the GPU (JSON)  <- worker
    <i>.canvas / .height the strokes the GPU painted             <- page
    <i>.nocanvas         the page painted nothing                <- page
    <i>.png              the finished take (bare)                <- worker
    <i>.err              what went wrong                         <- worker
    stop                 Stop was pressed / a new Run started    <- page
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid

import numpy as np

WAIT_LIMIT = 900.0           # a worker gives up when the page goes quiet this long


def _put(path, data: bytes):
    """Write a file so readers never see half of it."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _take(job_dir, i, p):
    """One take, start to end (runs in a worker process or thread)."""
    f = lambda name: os.path.join(job_dir, f"{i}.{name}")
    stopped = lambda: os.path.exists(os.path.join(job_dir, "stop"))
    try:
        import cv2
        from . import engine
        from . import planner as pl
        if stopped():
            return i, False
        image = np.load(os.path.join(job_dir, "image.npy"), mmap_mode="r")
        a, plan = engine.canvas_oil_gpu_start(np.asarray(image, np.float32), p)
        if plan is None:
            _put(f("none"), b"")
        else:
            brush = {k: v for k, v in a["q"].items() if k != "guides" and p.get(k) != v
                     and isinstance(v, (int, float, str))}
            _put(f("brush"), json.dumps(brush).encode())
            _put(f("plan"), pl.encode(plan, scale=a.get("plan_scale", 1.0), ground_side=1200))
        del plan
        canvas = height = None
        if not os.path.exists(f("none")):
            t0 = time.time()
            while not (os.path.exists(f("height")) or os.path.exists(f("nocanvas"))):
                if stopped() or time.time() - t0 > WAIT_LIMIT:
                    return i, False
                time.sleep(0.05)
            if os.path.exists(f("height")):
                h, w = a["paint"].shape[:2]
                with open(f("canvas"), "rb") as fh:
                    canvas = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_COLOR)
                with open(f("height"), "rb") as fh:
                    raw = np.frombuffer(fh.read(), "<f2")
                if canvas is None:
                    raise ValueError("The painted strokes could not be read.")
                ch, cw = canvas.shape[:2]
                if raw.size != ch * cw or abs(ch - h) > 2 or abs(cw - w) > 2:
                    raise ValueError("The painted strokes do not match the picture.")
                height = raw.astype(np.float32).reshape(ch, cw)
                if (ch, cw) != (h, w):          # a plan made smaller and enlarged: off by a pixel
                    canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
                    height = cv2.resize(height, (w, h), interpolation=cv2.INTER_LINEAR)
                canvas = canvas.astype(np.float32)
        if stopped():
            return i, False
        bare = engine.canvas_oil_gpu_finish(a, canvas, height)
        ok, png = cv2.imencode(".png", np.clip(bare, 0, 255).astype(np.uint8))
        _put(f("png"), png.tobytes())
        return i, True
    except Exception as err:                                     # noqa: BLE001
        _put(f("err"), (f"{err}\n\n" + traceback.format_exc()).encode("utf-8", "replace"))
        return i, False


class Job:
    """The takes of one GPU Run of Canvas Oil."""

    def __init__(self, root, image, takes):
        self.id = uuid.uuid4().hex
        self.dir = os.path.join(root, "co-" + self.id)
        os.makedirs(self.dir, exist_ok=True)
        np.save(os.path.join(self.dir, "image.npy"), np.ascontiguousarray(image))
        self.n = len(takes)
        self.pool = None
        self.started = time.time()
        threading.Thread(target=self._run, args=(takes,), daemon=True).start()

    def _run(self, takes):
        from . import engine
        pool = None
        workers = engine._variation_workers(self.n) if self.n > 1 else 1
        if workers > 1:
            made = engine._process_pool(workers)
            pool = made[0] if made else None
            self.pool = pool
        if pool is not None:
            from concurrent.futures.process import BrokenProcessPool
            futs = [pool.submit(_take, self.dir, i, q) for i, q in enumerate(takes)]
            redo = []
            for i, fu in enumerate(futs):
                try:
                    fu.result()
                except BrokenProcessPool:
                    redo.append(i)          # processes cannot start here: threads
                except Exception:           # noqa: BLE001 - _take writes its own errors
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
            takes_left = [] if self.stopped() else [(i, takes[i]) for i in redo]
        else:
            takes_left = list(enumerate(takes))
        if takes_left:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(2, len(takes_left))) as ex:
                list(ex.map(lambda t: _take(self.dir, t[0], t[1]), takes_left))

    def path(self, i, name):
        return os.path.join(self.dir, f"{int(i)}.{name}")

    def wait(self, i, names, limit=WAIT_LIMIT):
        """The first of take i's files that appears (or its error)."""
        t0 = time.time()
        while time.time() - t0 < limit:
            if os.path.exists(self.path(i, "err")):
                with open(self.path(i, "err"), "r", encoding="utf-8", errors="replace") as fh:
                    raise RuntimeError(fh.read().split("\n\n")[0] or "Canvas Oil failed")
            for n in names:
                if os.path.exists(self.path(i, n)):
                    return n
            if self.stopped():
                return None
            time.sleep(0.05)
        raise TimeoutError("Canvas Oil took too long")

    def put(self, i, name, data: bytes):
        _put(self.path(i, name), data)

    def stop(self):
        try:
            _put(os.path.join(self.dir, "stop"), b"")
        except OSError:
            pass
        if self.pool is not None:                 # Stop: free the CPU at once
            from . import engine
            engine.kill_pool(self.pool)

    def stopped(self):
        return os.path.exists(os.path.join(self.dir, "stop"))

    def remove(self):
        self.stop()
        threading.Timer(20.0, shutil.rmtree, args=(self.dir,), kwargs={"ignore_errors": True}).start()


def clean_old(root):
    """Job folders left by an earlier session."""
    try:
        for name in os.listdir(root):
            if name.startswith("co-"):
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    except OSError:
        pass
