#!/bin/bash
# Builds the PROTECTED OilStudio for clients on a Mac: no readable code inside.
#   dist-protected/OilStudio.app  and  dist-protected/OilStudio-mac.zip
# The Python code is compiled to machine code (Cython), the scripts and
# templates are minified and packed inside the program. See build/protect.py.
# Needs: Python 3.10+ (installed if missing), Xcode Command Line Tools
# (asked for if missing). Node.js is optional (scrambles the JavaScript names).
# Run:  bash build/build_mac_protected.sh
#       (or double-click "Build OilStudio Protected (Mac).command")
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

# ---- 1. the C compiler (Xcode Command Line Tools) ---------------------------
if ! xcode-select -p >/dev/null 2>&1 || ! command -v clang >/dev/null 2>&1; then
  echo "== The Xcode Command Line Tools are needed to compile OilStudio."
  echo "   A window opens: click Install, wait for it to finish, then run this again."
  xcode-select --install || true
  exit 1
fi

# ---- 2. Python ----------------------------------------------------------------
if ! PY=$(find_python); then
  echo "== Python 3.10+ not found - installing Python ${PY_VERSION} from python.org"
  echo "   macOS will ask for this Mac's login password."
  TMP_PKG="$(mktemp -d)/python.pkg"
  curl -fL --progress-bar -o "$TMP_PKG" "$PY_PKG_URL"
  sudo installer -pkg "$TMP_PKG" -target /
  rm -f "$TMP_PKG"
  PY=$(find_python) || { echo "Python install failed."; exit 1; }
fi
echo "== Using $("$PY" --version) ($PY)"

# ---- 3. packages -----------------------------------------------------------------
VENV=.build-venv-mac
if [ ! -f "$VENV/bin/activate" ]; then
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"
echo "== Installing packages"
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller cython setuptools rjsmin rcssmin

# ---- 4. protect: minify, pack, compile, test ---------------------------------------
command -v node >/dev/null 2>&1 || echo "== Tip: install Node.js (nodejs.org) to scramble the JavaScript names too"
python build/protect.py --out .release

# ---- 5. build the app from the protected copy --------------------------------------
echo "== Building OilStudio.app (protected)"
( cd .release && pyinstaller --noconfirm --clean --workpath ../.pyi-work-protected \
    --distpath ../dist-protected OilStudio-protected.spec )

# the zip holds the app and a short READ ME for the client
rm -rf dist-protected/OilStudio-mac dist-protected/OilStudio-mac.zip
mkdir -p dist-protected/OilStudio-mac
ditto dist-protected/OilStudio.app dist-protected/OilStudio-mac/OilStudio.app
cp build/CLIENT-README.txt "dist-protected/OilStudio-mac/READ ME.txt"
ditto -c -k --keepParent dist-protected/OilStudio-mac dist-protected/OilStudio-mac.zip

echo
echo "Done: dist-protected/OilStudio.app"
echo "Send: dist-protected/OilStudio-mac.zip   (no readable code inside)"
echo "It runs on Macs with the same chip as this one ($(uname -m))."
