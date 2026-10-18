OilStudio
=========
Nothing else needs to be installed - not Python, not Photoshop.
Photoshop actions (.atn) run inside OilStudio itself.

WINDOWS
1. Unzip OilStudio-windows.zip (right-click > Extract All).
2. Open the OilStudio folder and double-click OilStudio.exe.
   Keep OilStudio.exe inside its folder - it needs the files next to it.
   Tip: right-click OilStudio.exe > Send to > Desktop (create shortcut).
3. If Windows says "Windows protected your PC": More info > Run anyway
   (only the first time).

MAC (Apple Silicon M1-M4: use the "apple-silicon" zip; Intel Macs: "intel")
1. Double-click the zip to unzip it.
2. Drag OilStudio into the Applications folder.
3. The first time, double-click OilStudio. macOS blocks it - click Done.
   Then open System Settings > Privacy & Security, scroll down and click
   "Open Anyway" next to OilStudio, enter your password, and click
   "Open Anyway" again. (Only the first time.)
   If macOS says the app "is damaged", open Terminal and run:
       xattr -cr /Applications/OilStudio.app
   then open it again.

ADDING YOUR PHOTOSHOP ACTIONS
OilStudio > Actions (.atn) panel > "+ Add action", then choose the .atn
file with its .abr/.pat files, or the whole Envato .zip. OilStudio shows
how many of the action's steps it can run, then press "Run action".
