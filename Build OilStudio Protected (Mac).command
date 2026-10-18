#!/bin/bash
# Double-click on YOUR Mac (the developer's): builds the protected
# OilStudio.app to send to clients - no readable code inside.
cd "$(dirname "$0")"
echo "=== Building protected OilStudio - 10-20 minutes, needs internet. Keep this window open. ==="
echo
if ! bash build/build_mac_protected.sh; then
  echo; echo "BUILD FAILED - see the messages above."
  read -n 1 -s -r -p "Press any key to close..."; exit 1
fi
open dist-protected
read -n 1 -s -r -p "Press any key to close this window..."
