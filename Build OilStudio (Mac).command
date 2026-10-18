#!/bin/bash
# Double-click on a Mac: installs everything needed, builds OilStudio.app,
# puts it in Applications and opens it.
cd "$(dirname "$0")"
echo "=== Building OilStudio - 5-15 minutes, needs internet. Keep this window open. ==="
echo
if ! bash build/build_mac.sh; then
  echo; echo "BUILD FAILED - send a screenshot of this window to the developer."
  read -n 1 -s -r -p "Press any key to close..."; exit 1
fi
rm -rf /Applications/OilStudio.app 2>/dev/null
if cp -R dist/OilStudio.app /Applications/ 2>/dev/null; then
  APP=/Applications/OilStudio.app
else
  APP="$(pwd)/dist/OilStudio.app"
fi
xattr -cr "$APP" 2>/dev/null
echo; echo "Done! OilStudio is at: $APP"
echo "Opening it now..."
open "$APP"
read -n 1 -s -r -p "Press any key to close this window..."
