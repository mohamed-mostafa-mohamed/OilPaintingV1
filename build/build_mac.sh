#!/bin/bash
# Builds dist/OilStudio.app and dist/OilStudio-mac.zip on a Mac.
# Installs everything it needs: Python 3.12 (if missing), then all the
# Python packages (Flask, OpenCV, NumPy, Pillow, pywebview, PyInstaller).
# Run:  bash build/build_mac.sh   (or double-click "Build OilStudio (Mac).command")
set -e
cd "$(dirname "$0")/.."

PY_VERSION=3.12.10
PY_PKG_URL="https://www.python.org/ftp/python/${PY_VERSION}/python-${PY_VERSION}-macos11.pkg"
PY_FRAMEWORK=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12

find_python() {
  for c in "$PY_FRAMEWORK" python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# ---- 1. Python ------------------------------------------------------------
if ! PY=$(find_python); then
  echo "== Python 3.10+ not found - installing Python ${PY_VERSION} from python.org"
  echo "   macOS will ask for this Mac's login password (the one used to install apps)."
  TMP_PKG="$(mktemp -d)/python.pkg"
  curl -fL --progress-bar -o "$TMP_PKG" "$PY_PKG_URL"
  sudo installer -pkg "$TMP_PKG" -target /
  rm -f "$TMP_PKG"
  PY=$(find_python) || { echo "Python install failed."; exit 1; }
fi
echo "== Using $("$PY" --version) ($PY)"

# ---- 2. packages, in a Mac venv of its own ---------------------------------
VENV=.build-venv-mac
if [ ! -f "$VENV/bin/activate" ]; then
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"
echo "== Installing packages (first time: a few minutes)"
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller

# ---- 3. build ----------------------------------------------------------------
echo "== Building OilStudio.app"
pyinstaller --noconfirm --clean --workpath .pyi-work OilStudio.spec

# zip with ditto: keeps the app's symlinks and signature
rm -f dist/OilStudio-mac.zip
ditto -c -k --keepParent dist/OilStudio.app dist/OilStudio-mac.zip

echo
echo "Done: dist/OilStudio.app   (zip: dist/OilStudio-mac.zip)"
