#!/bin/bash
# OilStudio installer for Mac (Apple Silicon M1-M4).
# Double-click it: it copies OilStudio into Applications, clears macOS's
# "downloaded from the internet" block (so the app never shows "damaged")
# and opens it. It changes nothing else on this Mac.
cd "$(dirname "$0")" || exit 1
APP_SRC="OilStudio.app"
APP_DST="/Applications/OilStudio.app"
echo ""
echo "  Installing OilStudio..."
echo ""
if [ "$(uname -m)" != "arm64" ]; then
  echo "  This OilStudio is for Apple Silicon Macs (M1, M2, M3, M4)."
  echo "  This Mac has an Intel chip - ask for the Intel version."
  read -r -p "  Press Enter to close." _; exit 1
fi
if [ ! -d "$APP_SRC" ]; then
  echo "  OilStudio.app was not found next to this installer."
  read -r -p "  Press Enter to close." _; exit 1
fi
osascript -e 'tell application "OilStudio" to quit' >/dev/null 2>&1
if ! { rm -rf "$APP_DST" && ditto "$APP_SRC" "$APP_DST"; } 2>/dev/null; then
  echo "  Applications needs your Mac password to install:"
  sudo rm -rf "$APP_DST" && sudo ditto "$APP_SRC" "$APP_DST" || { read -r -p "  Install failed. Press Enter to close." _; exit 1; }
fi
xattr -cr "$APP_DST" 2>/dev/null || sudo xattr -cr "$APP_DST"
echo "  Done - OilStudio is in your Applications folder."
echo "  Opening it now..."
open "$APP_DST"
sleep 1
osascript -e 'tell application "Terminal" to close (every window whose name contains "Install OilStudio")' >/dev/null 2>&1 &
exit 0
