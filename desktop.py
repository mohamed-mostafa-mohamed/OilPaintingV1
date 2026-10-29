"""
OilStudio as a desktop app.

Starts the Flask app on a free local port (only this computer can reach
it) and shows it in a native window: WebView2 on Windows, WebKit on macOS.

    python desktop.py              run from source
    OilStudio.exe / OilStudio.app  the packaged app (see build/ and README)

If pywebview is missing it opens the default browser instead.
"""
from __future__ import annotations

import base64
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser

APP_NAME = "DOOD Studio"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(port: int, timeout: float = 20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _seed_actions_folder(app_module) -> None:
    """First run of the packaged app: put the README and example packs in the
    user's action library so they know where to drop Envato packs."""
    target = app_module.ACTIONS_DIR
    bundled = os.path.join(app_module.BASE_DIR, "ps_actions")
    if not os.path.isdir(bundled) or os.path.abspath(bundled) == os.path.abspath(target):
        return
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(bundled):
        src, dst = os.path.join(bundled, name), os.path.join(target, name)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _gpu_flags() -> None:
    """Keep the window on the graphics card (WebView2 / Chromium on Windows).

    Without these the WebGL 2 preview "sometimes" ran without the GPU:
      * an older/unknown driver is on Chromium's GPU blocklist -> no WebGL 2
      * laptops with two GPUs got the weak integrated one
      * after a few GPU-process crashes (driver resets on big paintings)
        Chromium switches the GPU off for the rest of the session
    Must be set before the WebView2 window is created.
    Set OILSTUDIO_NO_GPU_FLAGS=1 to turn this off.
    """
    if sys.platform != "win32" or os.environ.get("OILSTUDIO_NO_GPU_FLAGS"):
        return
    flags = ["--ignore-gpu-blocklist", "--enable-gpu-rasterization",
             "--force_high_performance_gpu", "--disable-gpu-process-crash-limit"]
    key = "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"
    have = os.environ.get(key, "")
    extra = [f for f in flags if f not in have]
    if extra:
        os.environ[key] = (have + " " + " ".join(extra)).strip()


class SaveBridge:
    """window.pywebview.api for the page: a native Save dialog.

    macOS's WebKit window ignores browser downloads (Save did nothing on a
    Mac); Windows' WebView2 allows them. Both now save through this."""

    def save_file(self, name, b64):
        import webview
        try:
            win = webview.windows[0]
            kind = getattr(getattr(webview, "FileDialog", None), "SAVE", None)
            if kind is None:
                kind = webview.SAVE_DIALOG
            safe = os.path.basename(str(name or "painting.jpg")) or "painting.jpg"
            res = win.create_file_dialog(kind, directory=os.path.expanduser("~"), save_filename=safe)
            if not res:
                return {"ok": False, "cancelled": True}
            path = res if isinstance(res, str) else res[0]
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(b64))
            return {"ok": True, "path": path}
        except Exception as exc:          # noqa: BLE001 - shown in the page
            return {"ok": False, "error": str(exc)}


def main() -> None:
    _gpu_flags()
    import app as oilstudio               # the Flask app
    from werkzeug.serving import make_server

    _seed_actions_folder(oilstudio)

    port = int(os.environ.get("PORT") or _free_port())
    server = make_server("127.0.0.1", port, oilstudio.app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    if not _wait_until_up(port):
        raise SystemExit("OilStudio could not start its local server.")

    try:
        import webview
    except ImportError:
        print(f"pywebview is not installed - opening {url} in your browser.")
        print("Close this window to quit OilStudio.")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        return

    # let Save / Batch downloads show a save dialog inside the window
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True
    except Exception:                      # noqa: BLE001 - older pywebview
        pass

    webview.create_window(APP_NAME, url, width=1500, height=950,
                          min_size=(1100, 700), background_color="#1b1d21",
                          text_select=False, js_api=SaveBridge())
    webview.start(private_mode=False)      # returns when the window is closed
    server.shutdown()


def run() -> None:
    """Entry point (the protected build starts the app through this)."""
    # Canvas Oil paints its variations in worker processes: a packaged app
    # must let those start (they re-run this program with special arguments)
    import multiprocessing
    multiprocessing.freeze_support()
    # a packaged app on Windows runs from its own folder
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    main()


if __name__ == "__main__":
    run()
