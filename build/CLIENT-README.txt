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

MAC (Apple Silicon M1-M4: the "apple-silicon" file; Intel Macs: the "intel" one)
1. Double-click OilStudio-mac-apple-silicon.dmg. A window opens.
2. Double-click "Install OilStudio". It copies OilStudio into Applications
   and opens it. Done - from now on open it from Applications or Launchpad.
   If macOS says it cannot check the installer: click Done, open
   System Settings > Privacy & Security, scroll down, click "Open Anyway"
   next to "Install OilStudio", then "Open Anyway" again. (Only once.)
   (Or drag OilStudio onto the Applications folder in that window instead.)
If macOS ever says OilStudio "is damaged", open Terminal and run:
       xattr -cr /Applications/OilStudio.app
then open it again.

ADDING YOUR PHOTOSHOP ACTIONS
OilStudio > Actions (.atn) panel > "+ Add action", then choose the .atn
file with its .abr/.pat files, or the whole Envato .zip. OilStudio shows
how many of the action's steps it can run, then press "Run action".
