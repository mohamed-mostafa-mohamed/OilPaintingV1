"""
Turn a Photoshop template into a recipe the app can play on any photo.

    python tools/psd_to_recipe.py refs/source.psd recipes/oil_smear --name "Oil Smear"

Reads the layer tree (tools/dump_psd.py), and writes
    <out>/recipe.json         the visible steps, bottom -> top
    <out>/<asset>.png         pixel layers above the photo, displacement maps
    <out>/photo_input.png     the template's own photo (Photoshop's cached
                              smart-filter input), for verification only

Rules
  - the photo slot is the visible smart object that carries smart filters
    (its smart filters become the photo's filter chain)
  - visible pixel layers above it become "layer" steps with their pictures
  - adjustment layers become "adjustment" steps
  - layers at 0 % opacity/fill, hidden layers, groups whose name says
    "hide" and anything under the (opaque) photo are left out
  - filters the engine cannot play are listed under "unsupported"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import dump_psd        # noqa: E402
import psdread         # noqa: E402

FILTERS = {"oilPaint": "oil_paint", "Pstr": "posterize", "GsnB": "gaussian_blur",
           "Dspl": "displace", "GEfc": "filter_gallery"}
GALLERY = {"Ct  ": "cutout"}


def flat(tree, out=None, hidden=False):
    out = [] if out is None else out
    for n in tree["children"]:
        h = hidden or not n.get("visible", True) or bool(re.search(r"\bhide\b", n["name"], re.I))
        if n["type"] == "group":
            flat(n, out, h)
        else:
            out.append((n, h))
    return out


def filter_step(f):
    st = f["settings"] or {}
    cls = f.get("filter_class")
    op = FILTERS.get(cls)
    if op == "oil_paint":
        return {"op": "oil_paint", "label": "Oil Paint", "settings": {
            "stylization": st.get("stylization"), "cleanliness": st.get("cleanliness"),
            "scale": st.get("brushScale"), "bristle": st.get("microBrush"),
            "lighting": st.get("lightingOn"), "angle": st.get("LghD"),
            "shine": st.get("specularity")}}
    if op == "posterize":
        return {"op": "posterize", "label": "Posterize", "levels": st.get("Lvls")}
    if op == "gaussian_blur":
        r = st.get("Rds")
        return {"op": "gaussian_blur", "label": "Gaussian Blur",
                "radius": r["value"] if isinstance(r, dict) else r}
    if op == "displace":
        return {"op": "displace", "label": "Displace", "horizontal": st.get("HrzS"),
                "vertical": st.get("VrtS"),
                "fit": "stretch" if st.get("DspM") == "StrF" else "tile",
                "edges": "repeat" if st.get("UndA") == "RptE" else "wrap",
                "map": None, "_map_png": f.get("embedded_map_png")}
    if op == "filter_gallery":
        kind = GALLERY.get(st.get("GEfk"))
        if kind == "cutout":
            return {"op": "cutout", "label": "Cutout", "settings": {
                "levels": st.get("NmbL"), "simplicity": st.get("EdgS"), "fidelity": st.get("EdgF")}}
        return {"op": "unsupported", "label": f"Filter Gallery {st.get('GEfk')}", "settings": st}
    return {"op": "unsupported", "label": f.get("name"), "class": cls, "settings": st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("psd")
    ap.add_argument("out")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = psdread.parse(a.psd, want_pixels=True)
    tmp = os.path.join(a.out, "_dump")
    os.makedirs(tmp, exist_ok=True)
    tree = dump_psd.build_tree(info, tmp, export=False)
    items = flat(tree)

    photo_i = None
    for k, (n, hid) in enumerate(items):
        so = n.get("smart_object") or {}
        if not hid and n["type"] == "smart_object" and so.get("smart_filters") \
                and n.get("opacity_255", 255) > 0:
            photo_i = k
    if photo_i is None:
        sys.exit("No visible smart object with smart filters: no photo slot found.")

    steps, unsupported = [], []
    pn = items[photo_i][0]
    chain = []
    for f in pn["smart_object"]["smart_filters"]:
        if not f.get("enabled", True):
            continue
        st = filter_step(f)
        if st["op"] == "displace":
            src = st.pop("_map_png")
            if src:
                name = "displace_map.png"
                if os.path.exists(src):
                    os.replace(src, os.path.join(a.out, name))
                st["map"] = name
        if st["op"] == "unsupported":
            unsupported.append(st)
        chain.append(st)
    steps.append({"op": "photo", "label": pn["name"].strip(), "filters": chain})

    used = set()
    for n, hid in items[photo_i + 1:]:
        if hid or n.get("opacity_255", 255) == 0 or n.get("fill_opacity", 100) == 0:
            continue
        op = n["opacity_255"] / 255.0 * n.get("fill_opacity", 100) / 100.0
        if n["type"] == "adjustment":
            steps.append({"op": "adjustment", "label": n["name"], "kind": n["adjustment"]["kind"],
                          "settings": n["adjustment"]["settings"], "opacity": round(op, 4)})
        elif n["type"] in ("pixel", "smart_object"):
            L = info["layers"][n["index"]]
            px = psdread.rgba(info, L)
            if px is None:
                continue
            base = re.sub(r"[^a-z0-9]+", "_", n["name"].lower()).strip("_") or "layer"
            name = f"{base}_{n['blend_mode'].replace(' ', '_')}.png"
            while name in used:
                name = name.replace(".png", "_2.png")
            used.add(name)
            canvas = np.zeros((info["h"], info["w"], 4), np.uint8)
            t, l = L.top, L.left
            y0, x0, y1, x1 = max(0, t), max(0, l), min(info["h"], L.bottom), min(info["w"], L.right)
            canvas[y0:y1, x0:x1] = px[y0 - t:y1 - t, x0 - l:x1 - l]
            name = name.replace(".png", ".jpg")      # big photo-like textures: JPEG q97
            gray = (np.ptp(canvas[..., :3].astype(int), axis=2).max() == 0)
            opaque = canvas[..., 3].mean() >= 254.5 and canvas[..., 3].min() >= 230
            q = [cv2.IMWRITE_JPEG_QUALITY, 97]
            if gray and opaque:
                cv2.imwrite(os.path.join(a.out, name), canvas[..., 0], q)
            elif opaque:
                cv2.imwrite(os.path.join(a.out, name), canvas[..., 2::-1], q)
            else:
                name = name.replace(".jpg", ".png")
                cv2.imwrite(os.path.join(a.out, name), cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA))
            steps.append({"op": "layer", "label": n["name"], "image": name,
                          "mode": n["blend_mode"], "opacity": round(op, 4)})
        elif n["type"] == "fill":
            unsupported.append({"op": "fill", "label": n["name"]})
        if n.get("effects"):
            unsupported.append({"op": "layer effects", "label": n["name"]})

    recipe = {"name": a.name or os.path.splitext(os.path.basename(a.psd))[0],
              "source": os.path.basename(a.psd),
              "document": {"width": info["w"], "height": info["h"]},
              "steps": steps, "unsupported": unsupported}
    with open(os.path.join(a.out, "recipe.json"), "w") as fh:
        json.dump(recipe, fh, indent=2)
    print(json.dumps(recipe, indent=2))
    # cleanup
    for fn in os.listdir(tmp) if os.path.isdir(tmp) else []:
        os.remove(os.path.join(tmp, fn))
    if os.path.isdir(tmp):
        os.rmdir(tmp)


if __name__ == "__main__":
    main()
