"""
The library of action packs (Envato or any Photoshop actions).

Each pack is one sub-folder of the library folder (``ps_actions/`` by
default, or ``OILSTUDIO_ACTIONS_DIR``)::

    ps_actions/
      canvas-oil/
        pack.json      optional settings (see README)
        preview.jpg    optional thumbnail
        *.atn + the files it uses    the Envato files - or put "folder"
                                     in pack.json to use them where they are

OilStudio plays the actions itself (see actions/player.py). The files an
action can come with, and what OilStudio does with them:

  .atn                         the actions - every step is read and played
  .abr                         brushes - used by "Stroke path" steps
  .pat                         patterns - used by pattern overlays and fills
  .asl .grd .shc .csh .aco .ase .tpl
                               kept with the pack (the action itself carries
                               the styles / gradients / colours it applies)
  .cube .3dl .look .csp        colour lookup tables: kept with the pack
  .jsx .js .jsxbin .zxp .ccx   Photoshop scripts / plugins: kept, but they
                               only run inside Photoshop
  .psd .psb                    templates / textures: kept with the pack

Without a pack.json the folder name is used and the action set name is read
straight from the .atn file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile

from . import atn

# used by the action player
PLAYED_EXT = (".abr", ".pat")
# kept with the pack (the .atn already holds what it applies from these)
KEEP_EXT = (".asl", ".grd", ".shc", ".csh", ".aco", ".ase", ".tpl",
            ".cube", ".3dl", ".look", ".csp")
# only useful inside Photoshop - kept so nothing from the Envato zip is lost
PHOTOSHOP_ONLY_EXT = (".jsx", ".js", ".jsxbin", ".zxp", ".ccx")
KEEP_ONLY_EXT = (".psd", ".psb")
RESOURCE_EXT = (".atn",) + PLAYED_EXT + KEEP_EXT
PACK_EXT = RESOURCE_EXT + PHOTOSHOP_ONLY_EXT + KEEP_ONLY_EXT
HELP_EXT = (".txt", ".pdf", ".rtf")
IMAGE_EXT = (".jpg", ".jpeg", ".png")
MAX_UNZIPPED = 2 * 1024 ** 3          # refuse zip bombs
PREVIEW_NAMES = ("preview.jpg", "preview.jpeg", "preview.png", "thumb.jpg", "thumb.png")
BRUSH_MODES = ("subject", "full", "none")

DEFAULTS = {
    "brush": "subject",       # how the focus-area layer is made
    "brush_layer": "brush",   # its name - most Envato actions use "brush"
    "long_side": 3500,        # resize before running (0 = keep size)
    "resolution": 300,
    "category": "Effects",
    "notes": "",
}


def read_atn(path: str) -> dict:
    """Action-set name and the names of all its actions."""
    try:
        aset = atn.load(path)
        names = [a.name for a in aset.actions]
        return {"set": aset.name, "count": len(names), "first": names[0] if names else "",
                "names": names, "error": ""}
    except Exception as exc:               # noqa: BLE001 - unusual file
        return {"set": "", "count": 0, "first": "", "names": [], "error": str(exc)}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "pack"


def _safe_file_name(name: str) -> str:
    """Just the file name, without any folder part or odd characters."""
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base)
    if not base or base.startswith(".") or base in (".", ".."):
        return ""
    return base[:150]


def _unique(path: str, dirs: bool = False) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = (path, "") if dirs else os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}-{n}{ext}"):
        n += 1
    return f"{stem}-{n}{ext}"


def _write(path: str, stream) -> None:
    with open(path, "wb") as fh:
        shutil.copyfileobj(stream, fh, 1024 * 1024)


class Library:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(root, exist_ok=True)

    def _resources(self, folder: str, ext=RESOURCE_EXT) -> list[str]:
        found = []
        for base, dirs, files in os.walk(folder):
            if base[len(folder):].count(os.sep) >= 2:      # two levels deep is plenty
                dirs[:] = []
            for f in sorted(files):
                if f.lower().endswith(ext) and not f.startswith("._"):
                    found.append(os.path.join(base, f))
        return found

    def packs(self) -> list[dict]:
        out = []
        for entry in sorted(os.listdir(self.root)):
            pack_dir = os.path.join(self.root, entry)
            if not os.path.isdir(pack_dir) or entry.startswith((".", "_")):
                continue
            try:
                out.append(self._load(entry, pack_dir))
            except Exception as exc:       # noqa: BLE001 - show the problem instead of crashing
                out.append({"id": _slug(entry), "name": entry, "error": str(exc)})
        return out

    def get(self, pack_id: str) -> dict | None:
        for p in self.packs():
            if p["id"] == pack_id:
                return p
        return None

    def _load(self, entry: str, pack_dir: str) -> dict:
        meta = {}
        cfg = os.path.join(pack_dir, "pack.json")
        if os.path.exists(cfg):
            with open(cfg, encoding="utf-8") as fh:
                meta = json.load(fh)

        folder = os.path.expanduser(meta.get("folder") or "") or pack_dir
        if not os.path.isabs(folder):
            folder = os.path.join(pack_dir, folder)
        folder = os.path.normpath(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")

        resources = self._resources(folder)
        if folder != pack_dir:
            resources += [r for r in self._resources(pack_dir) if r not in resources]
        atns = [r for r in resources if r.lower().endswith(".atn")]
        if not atns:
            raise ValueError("No .atn file in this pack.")

        headers = [read_atn(a) for a in atns]
        header = headers[0]
        names = [n for h in headers for n in h["names"]]
        if not names:
            raise ValueError("The .atn file could not be read" +
                             (f": {header['error']}" if header["error"] else "."))
        preview = next((os.path.join(pack_dir, n) for n in PREVIEW_NAMES
                        if os.path.exists(os.path.join(pack_dir, n))), None)

        ps_only = [os.path.basename(p) for p in self._resources(folder, PHOTOSHOP_ONLY_EXT)]
        templates = [os.path.basename(p) for p in self._resources(folder, KEEP_ONLY_EXT)]
        pack = dict(DEFAULTS)
        pack.update({
            "plugins": ps_only,
            "templates": templates,
            "id": _slug(meta.get("id") or entry),
            "name": meta.get("name") or entry,
            "set": meta.get("set") or header["set"],
            "action": meta.get("action") or names[0],
            "actions": names,
            "action_count": len(names),
            "folder": folder,
            "atns": atns,
            "resources": resources,
            "preview": preview,
        })
        for k in DEFAULTS:
            if k in meta:
                pack[k] = meta[k]
        if pack["action"] not in names:
            pack["action"] = names[0]
        if pack["brush"] not in BRUSH_MODES:
            pack["brush"] = "subject"
        if ps_only:
            note = ("Scripts / plugins in this pack (" + ", ".join(ps_only) + ") only work "
                    "inside Photoshop, so OilStudio does not use them.")
            pack["notes"] = (pack["notes"] + "\n" + note).strip()
        return pack

    # -- adding packs from the app ------------------------------------------
    def add_pack(self, files, *, name: str = "", category: str = "",
                 brush: str = "") -> dict:
        """Make a new pack folder from uploaded files.

        ``files`` is a list of ``(filename, readable)`` pairs: the .atn plus
        any of the files it uses (see the module docstring), an Envato .zip
        holding them (one nested .zip is unpacked too), an optional preview
        image and Help.txt.
        Returns the loaded pack. Raises ValueError when there is no .atn.
        """
        tmp = os.path.join(self.root, "_incoming-" + uuid.uuid4().hex[:10])
        os.makedirs(tmp)
        try:
            for filename, stream in files:
                base = _safe_file_name(filename)
                if not base:
                    continue
                low = base.lower()
                if low.endswith(".zip"):
                    with zipfile.ZipFile(stream) as zf:
                        self._unzip(zf, tmp, depth=0)
                elif low.endswith(IMAGE_EXT):
                    if not any(os.path.exists(os.path.join(tmp, n)) for n in PREVIEW_NAMES):
                        ext = ".png" if low.endswith(".png") else ".jpg"
                        _write(os.path.join(tmp, "preview" + ext), stream)
                elif low.endswith(PACK_EXT + HELP_EXT):
                    _write(_unique(os.path.join(tmp, base)), stream)

            atns = [r for r in self._resources(tmp) if r.lower().endswith(".atn")]
            if not atns:
                raise ValueError("No Photoshop action (.atn) file was found in what you added.")

            header = read_atn(atns[0])
            if not header["names"]:
                raise ValueError("That .atn file could not be read" +
                                 (f": {header['error']}" if header["error"] else "."))
            name = (name or "").strip()[:80]
            title = name or header["set"] or os.path.splitext(os.path.basename(atns[0]))[0]
            meta = {"name": title}
            if (category or "").strip():
                meta["category"] = category.strip()[:40]
            if brush in BRUSH_MODES:
                meta["brush"] = brush
            with open(os.path.join(tmp, "pack.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)

            final = _unique(os.path.join(self.root, _slug(title)), dirs=True)
            os.replace(tmp, final)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self._load(os.path.basename(final), final)

    def _unzip(self, zf: zipfile.ZipFile, dest: str, depth: int) -> None:
        total = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            base = _safe_file_name(parts[-1])
            if not base or "__MACOSX" in parts:
                continue
            total += info.file_size
            if total > MAX_UNZIPPED:
                raise ValueError("That zip is too large once unpacked.")
            low = base.lower()
            if low.endswith(".zip") and depth < 1:          # Envato often zips a zip
                with zf.open(info) as inner, tempfile.TemporaryFile() as buf:
                    shutil.copyfileobj(inner, buf)
                    buf.seek(0)
                    with zipfile.ZipFile(buf) as zf2:
                        self._unzip(zf2, dest, depth + 1)
            elif low.endswith(PACK_EXT + HELP_EXT):
                with zf.open(info) as src:
                    _write(_unique(os.path.join(dest, base)), src)
            elif low.endswith(IMAGE_EXT) and low.rsplit(".", 1)[0] in ("preview", "thumb"):
                with zf.open(info) as src:
                    _write(_unique(os.path.join(dest, base)), src)

    def remove_pack(self, pack_id: str) -> bool:
        """Delete a pack folder from the library (never the files it points to)."""
        for entry in os.listdir(self.root):
            path = os.path.join(self.root, entry)
            if os.path.isdir(path) and _slug(entry) == pack_id and not entry.startswith((".", "_")):
                shutil.rmtree(path)
                return True
        return False

    @staticmethod
    def public(pack: dict) -> dict:
        """What the browser may see (no file-system paths)."""
        keys = ("id", "name", "set", "action", "actions", "action_count", "brush",
                "brush_layer", "long_side", "category", "notes", "error",
                "plugins", "templates")
        out = {k: pack[k] for k in keys if k in pack}
        out["has_preview"] = bool(pack.get("preview"))
        out["files"] = [os.path.basename(r) for r in pack.get("resources", [])]
        return out
