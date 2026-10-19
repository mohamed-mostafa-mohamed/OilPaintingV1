"""
Check a recipe against the PSD it came from, step by step.

    python tools/verify_recipe.py recipes/oil_smear --refs ../refs [--side 1000] [--tag v1]

Uses the PSD's own photo layer as input, and compares
  1. the photo after its smart filters  vs  the PSD's "Image Fx" layer
  2. + the texture layers               vs  the PSD's "Image" group
  3. + the adjustments (final)          vs  refs/target.jpg
It also runs "oracle" passes that start from the PSD's own layer at a
checkpoint, so the error of each later step is measured on its own.
Prints PSNR / SSIM / MAE and saves side-by-side pictures and difference
heatmaps in outputs/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine import psd_recipe as R   # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402


def load_rgb(path, side=None):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    im = im[..., ::-1] if im.ndim == 3 else np.repeat(im[..., None], 3, 2)
    if side:
        k = side / min(im.shape[:2])
        im = cv2.resize(im, (int(round(im.shape[1] * k)), int(round(im.shape[0] * k))),
                        interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0


def score(a, b):
    x = np.clip(a * 255, 0, 255).astype(np.float64)
    y = np.clip(b * 255, 0, 255).astype(np.float64)
    mse = ((x - y) ** 2).mean()
    gx = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gy = cv2.cvtColor(y.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return {"psnr": round(10 * np.log10(255 ** 2 / max(mse, 1e-9)), 2),
            "ssim": round(float(structural_similarity(gx, gy, data_range=255)), 4),
            "mae": round(float(np.abs(x - y).mean()), 2)}


def heat(a, b):
    d = np.abs(a - b).mean(2) * 255
    d = np.clip(d * 4, 0, 255).astype(np.uint8)          # x4: 64 levels = full red
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def save_compare(name, ours, ref, out_dir):
    o = (np.clip(ours, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    r = (np.clip(ref, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    lab = lambda im, t: cv2.putText(im.copy(), t, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (255, 255, 255), 4, cv2.LINE_AA)
    side = np.hstack([lab(r, "Photoshop"), lab(o, "OilStudio"), lab(heat(ours, ref), "difference x4")])
    cv2.imwrite(os.path.join(out_dir, f"{name}.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--side", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tune", default=None, help="JSON with filter tune overrides")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = R.Recipe(os.path.join(a.recipe, "recipe.json"))
    if a.tune:
        tune = json.loads(a.tune)
        for f in rec.data["steps"][0]["filters"]:
            if f["op"] in tune:
                f["tune"] = tune[f["op"]]
    lay = os.path.join(a.refs, "layers")
    photo = load_rgb(os.path.join(lay, "16_Place_Image_Here.png"), a.side)
    psd_fx = load_rgb(os.path.join(lay, "05_Image_Fx.png"), a.side)
    target = load_rgb(os.path.join(a.refs, "target.jpg"), a.side)
    tex_o = load_rgb(os.path.join(lay, "06_Texture.png"), a.side)
    tex_m = load_rgb(os.path.join(lay, "07_Texture.png"), a.side)
    psd_image = psd_fx + (R.blend("overlay", psd_fx, tex_o) - psd_fx) * (145 / 255)
    psd_image = psd_image + (R.blend("multiply", psd_image, tex_m) - psd_image) * (69 / 255)

    t = time.time()
    labels = [st.get("label", st["op"]) for st in rec.data["steps"]]
    last_tex = [l for l, st in zip(labels, rec.data["steps"]) if st["op"] == "layer"][-1]
    final, steps = R.run(rec, photo, keep_steps={labels[0], last_tex})
    secs = time.time() - t
    tag = f"_{a.tag}" if a.tag else ""
    rows = []
    fx = steps[labels[0]]
    rows.append(("1 photo + smart filters  vs PSD 'Image Fx'", score(fx, psd_fx)))
    after_tex = steps[last_tex]
    del steps
    rows.append(("2 + textures             vs PSD 'Image' group", score(after_tex, psd_image)))
    rows.append(("3 final                  vs target.jpg", score(final, target)))
    # oracles: start from the PSD's own layer -> error of the later steps alone
    rec2 = R.Recipe(os.path.join(a.recipe, "recipe.json"))
    rec2.data["steps"][0]["filters"] = []
    o_final, o_steps = R.run(rec2, psd_fx, keep_steps={last_tex})
    o_tex = o_steps[last_tex]
    rows.append(("oracle: PSD Image Fx + our textures   vs PSD 'Image'", score(o_tex, psd_image)))
    rows.append(("oracle: PSD Image Fx + our textures + adjustments vs target", score(o_final, target)))
    print(f"\nrecipe {rec.name}, input {photo.shape[1]}x{photo.shape[0]}, {secs:.1f} s")
    for name, s in rows:
        print(f"  {name:62s} PSNR {s['psnr']:6.2f} dB  SSIM {s['ssim']:.4f}  MAE {s['mae']:5.2f}")
    save_compare(f"verify{tag}_1_image_fx", fx, psd_fx, a.out)
    save_compare(f"verify{tag}_3_final", final, target, a.out)
    cv2.imwrite(os.path.join(a.out, f"result{tag}.jpg"),
                (np.clip(final, 0, 1) * 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
    with open(os.path.join(a.out, f"verify{tag}.json"), "w") as fh:
        json.dump({n: s for n, s in rows}, fh, indent=2)


if __name__ == "__main__":
    main()
