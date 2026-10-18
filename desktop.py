"""
OilStudio as a desktop app.

Starts the Flask app on a free local port (only this computer can reach
it) and shows it in a native window: WebView2 on Windows, WebKit on macOS.

    python desktop.py              run from source
    OilStudio.exe / OilStudio.app  the packaged app (see build/ and README)

If pywebview is missing it opens the default browser instead.
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
import threading
import time
import webbrowser

APP_NAME = "OilStudio"


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


def main() -> None:
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
                          text_select=False)
    webview.start(private_mode=False)      # returns when the window is closed
    server.shutdown()


def run() -> None:
    """Entry point (the protected build starts the app through this)."""
    # a packaged app on Windows runs from its own folder
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    main()


if __name__ == "__main__":
    run()
