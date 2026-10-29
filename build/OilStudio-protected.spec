# PyInstaller recipe for the PROTECTED OilStudio app (no readable code).
# Do not run this from the project: build/protect.py copies it into the
# protected release folder, and the build_*_protected scripts run it there.
#   Windows:  build\build_windows_protected.bat  ->  dist-protected\OilStudio\OilStudio.exe
#   macOS:    bash build/build_mac_protected.sh  ->  dist-protected/OilStudio.app
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.getcwd())          # the compiled OilStudio modules are here

datas = []
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
    if folder and (os.path.isabs(folder) or folder.startswith(("~", "\\\\", "/"))
                   or (len(folder) > 1 and folder[1] == ":") or ".." in folder):
        print(f"OilStudio-protected.spec: not bundling '{entry}' (its files are outside the pack folder)")
        continue
    datas.append((pack_dir, pack_dir.replace(os.sep, "/")))
if os.path.isdir("samples"):
    datas.append(("samples", "samples"))
for _r in ("oil_fx", "sketch_paint", "rough_brush", "watercolor", "silk_oil"):              # template pictures of the filter styles
    if os.path.isdir("recipes/" + _r):
        datas.append(("recipes/" + _r, "recipes/" + _r))

# PyInstaller cannot see the imports inside compiled modules: protect.py
# listed them; keep the ones that exist on this system
with open(os.path.join("build", "imports.json"), encoding="utf-8") as fh:
    wanted = json.load(fh)["imports"]
hidden = []
for name in wanted:
    try:
        if importlib.util.find_spec(name) is not None:
            hidden.append(name)
    except (ImportError, ValueError, AttributeError):
        pass
hidden += ["webview"]

icon = None
for cand in ("build/icon.ico", "build/icon.icns"):
    if os.path.exists(cand) and cand.endswith(".ico" if sys.platform == "win32" else ".icns"):
        icon = cand

a = Analysis(["OilStudio.py"], pathex=["."], datas=datas, hiddenimports=sorted(set(hidden)),
             excludes=["tkinter", "matplotlib", "IPython", "Cython", "pytest"], noarchive=False)
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
