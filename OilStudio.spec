# PyInstaller recipe for the OilStudio desktop app.
#   Windows:  build\build_windows.bat   ->  dist\OilStudio\OilStudio.exe
#   macOS:    ./build/build_mac.sh      ->  dist/OilStudio.app
# PyInstaller cannot cross-build: make the .exe on Windows and the .app on a Mac.
import os
import sys

import json

datas = [("templates", "templates"), ("static", "static")]

# Ship the action library README, plus only the packs whose files are inside
# their own folder. A pack.json with "folder": "C:\\Users\\..." points at files
# on the build machine and would show up broken on the client's computer.
if os.path.exists("ps_actions/README.md"):
    datas.append(("ps_actions/README.md", "ps_actions"))
for entry in sorted(os.listdir("ps_actions")) if os.path.isdir("ps_actions") else []:
    pack_dir = os.path.join("ps_actions", entry)
    if not os.path.isdir(pack_dir) or entry.startswith((".", "_")):
        continue
    meta = {}
    try:
        with open(os.path.join(pack_dir, "pack.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        pass
    folder = meta.get("folder") or ""
    # absolute on any system: /Users/..., ~/..., C:\\..., \\\\server\\...
    if folder and (os.path.isabs(folder) or folder.startswith(("~", "\\\\", "/"))
                   or (len(folder) > 1 and folder[1] == ":") or ".." in folder):
        print(f"OilStudio.spec: not bundling '{entry}' (its files are outside the pack folder)")
        continue
    datas.append((pack_dir, pack_dir.replace(os.sep, "/")))

if os.path.isdir("samples"):
    datas.append(("samples", "samples"))

hidden = ["app", "actions", "actions.atn", "actions.filters", "actions.focus",
          "actions.library", "actions.player", "actions.resources"]

icon = None
for cand in ("build/icon.ico", "build/icon.icns"):
    if os.path.exists(cand) and cand.endswith(".ico" if sys.platform == "win32" else ".icns"):
        icon = cand

a = Analysis(["desktop.py"], pathex=["."], datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "matplotlib", "IPython"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="OilStudio",
          console=False, icon=icon)
coll = COLLECT(exe, a.binaries, a.datas, name="OilStudio")

if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="OilStudio.app", icon=icon, bundle_identifier="com.oilstudio.desktop",
        info_plist={
            "NSHighResolutionCapable": True,
        })
