"""
Visualize dataset samples (MVTecAD and VisA) showing:
  - Original image with the effective crop region highlighted (red box)
  - What the model actually sees (after Resize + CenterCrop)
  - Coverage statistics per dataset

Usage (PowerShell, from project root):
    $env:PYTHONPATH="src"
    python contribution/visualize_samples.py `
        --mvtec_path "C:/Users/trloj/Code/mvtec_anomaly_detection" `
        --visa_path  "C:/Users/trloj/Code/VisA_20220922" `
        --resize 256 --imagesize 224 `
        --n_samples 3 `
        --out_dir results/sample_viz

Each output PNG is a grid:
    rows = [train-good, test-good, test-anomaly]
    cols = [original + crop-box | model-input | mask (if available)]
"""

import argparse
import os
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from torchvision import transforms

# ── Make contribution/ importable when launched from project root ────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def crop_box_on_original(orig_w, orig_h, resize_px, imagesize_px):
    """
    Return (x0, y0, box_w, box_h) of the CenterCrop region
    expressed in *original* pixel coordinates.

    transforms.Resize(resize_px) resizes the shorter edge to resize_px.
    transforms.CenterCrop(imagesize_px) then crops a square.
    """
    scale = resize_px / min(orig_w, orig_h)
    scaled_w, scaled_h = orig_w * scale, orig_h * scale

    crop_w = crop_h = imagesize_px

    # top-left of the crop in scaled-image space
    cx0 = (scaled_w - crop_w) / 2
    cy0 = (scaled_h - crop_h) / 2

    # map back to original-image space
    x0 = cx0 / scale
    y0 = cy0 / scale
    bw = crop_w / scale
    bh = crop_h / scale
    return x0, y0, bw, bh


def coverage_pct(orig_w, orig_h, resize_px, imagesize_px):
    """Fraction of the original image area that is actually used."""
    x0, y0, bw, bh = crop_box_on_original(orig_w, orig_h, resize_px, imagesize_px)
    used = bw * bh
    total = orig_w * orig_h
    return 100.0 * used / total


# ────────────────────────────────────────────────────────────────────────────
# Transform pipeline
# ────────────────────────────────────────────────────────────────────────────

def make_transforms(resize_px, imagesize_px):
    img_tf = transforms.Compose([
        transforms.Resize(resize_px),
        transforms.CenterCrop(imagesize_px),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize(resize_px),
        transforms.CenterCrop(imagesize_px),
    ])
    return img_tf, mask_tf


def unnorm(tensor):
    """Convert normalised float tensor [3,H,W] → uint8 numpy [H,W,3]."""
    t = tensor.clone()
    for c, m, s in zip(t, IMAGENET_MEAN, IMAGENET_STD):
        c.mul_(s).add_(m)
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────
# Sample collectors
# ────────────────────────────────────────────────────────────────────────────

def _pick(paths, n, rng):
    return rng.sample(paths, min(n, len(paths)))


def collect_mvtec(root, classname, n_anomaly, n_normal, rng):
    """
    Returns dict of split_label → list of (img_path, mask_path_or_None).

    Layout:
      - 1 row "train-good (NORMAL)" : n_normal samples
      - 1 row "test-anomaly"        : n_anomaly samples pooled across all subtypes
    """
    cls_dir  = os.path.join(root, classname)
    test_dir = os.path.join(cls_dir, "test")
    gt_dir   = os.path.join(cls_dir, "ground_truth")
    splits   = {}

    # One normal reference row
    good_dir = os.path.join(cls_dir, "train", "good")
    if os.path.isdir(good_dir):
        imgs = [os.path.join(good_dir, f) for f in sorted(os.listdir(good_dir))]
        splits["train-good (NORMAL)"] = [(p, None) for p in _pick(imgs, n_normal, rng)]

    # Pool ALL anomaly subtypes together, then pick n_anomaly total
    all_anomaly = []
    anomaly_types = [d for d in sorted(os.listdir(test_dir)) if d != "good"]
    for atype in anomaly_types:
        anom_img_dir  = os.path.join(test_dir, atype)
        anom_mask_dir = os.path.join(gt_dir,   atype)
        for f in sorted(os.listdir(anom_img_dir)):
            stem   = os.path.splitext(f)[0]
            img_p  = os.path.join(anom_img_dir, f)
            mask_p = os.path.join(anom_mask_dir, stem + "_mask.png")
            if not os.path.isfile(mask_p):
                mask_p = os.path.join(anom_mask_dir, stem + ".png")
            all_anomaly.append((img_p, mask_p if os.path.isfile(mask_p) else None))
    if all_anomaly:
        splits["test-anomaly"] = _pick(all_anomaly, n_anomaly, rng)

    return splits


def collect_visa(root, classname, n_anomaly, n_normal, rng):
    """
    Returns dict of split_label → list of (img_path, mask_path_or_None).
    Uses image_anno.csv directly.

    Layout:
      - 1 row "normal"      : n_normal samples
      - 1 row "test-anomaly": n_anomaly samples
    """
    import csv
    csv_path = os.path.join(root, classname, "image_anno.csv")
    normal_imgs, anomaly_entries = [], []

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            img_rel  = row["image"].strip()
            mask_rel = row.get("mask", "").strip()
            abs_img  = os.path.join(root, img_rel)
            abs_mask = os.path.join(root, mask_rel) if mask_rel else None
            if row["label"].strip().lower() == "normal":
                normal_imgs.append(abs_img)
            else:
                anomaly_entries.append((abs_img, abs_mask))

    splits = {}
    if normal_imgs:
        splits["normal"] = [(p, None) for p in _pick(normal_imgs, n_normal, rng)]
    if anomaly_entries:
        picked = _pick(anomaly_entries, n_anomaly, rng)
        splits["test-anomaly"] = picked
    return splits


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def plot_entry(ax_orig, ax_crop, ax_mask_crop, ax_mask_full,
               img_path, mask_path,
               img_tf, resize_px, imagesize_px,
               split_label=""):
    """Fill one quad of axes (original+box | model-input | cropped-mask | full-mask)."""
    orig = PIL.Image.open(img_path).convert("RGB")
    ow, oh = orig.size          # PIL: (width, height)

    is_anomaly = "anomaly" in split_label.lower()
    label_color = "#d32f2f" if is_anomaly else "#2e7d32"   # red / dark-green
    label_text  = "ANOMALY" if is_anomaly else "NORMAL"
    filename    = os.path.basename(img_path)

    # ── Original + crop box ─────────────────────────────────────────────────
    ax_orig.imshow(np.array(orig))
    x0, y0, bw, bh = crop_box_on_original(ow, oh, resize_px, imagesize_px)
    rect = mpatches.FancyBboxPatch(
        (x0, y0), bw, bh,
        boxstyle="square,pad=0",
        linewidth=2, edgecolor="red", facecolor="none",
    )
    ax_orig.add_patch(rect)
    cov = coverage_pct(ow, oh, resize_px, imagesize_px)

    ax_orig.set_title(
        f"[{label_text}]  {split_label}\n"
        f"{filename}\n"
        f"{ow}x{oh} px | cov {cov:.0f}%",
        fontsize=6.5, pad=3, color=label_color, fontweight="bold",
    )
    ax_orig.axis("off")

    # ── Model input (crop) ──────────────────────────────────────────────────
    cropped = img_tf(orig)
    ax_crop.imshow(np.array(cropped))
    ax_crop.set_title(
        f"model input\n{imagesize_px}x{imagesize_px}",
        fontsize=6.5, pad=3,
    )
    ax_crop.axis("off")

    # ── Masks (cropped + full) ───────────────────────────────────────────────
    if mask_path and os.path.isfile(mask_path):
        mask_pil = PIL.Image.open(mask_path).convert("L")

        # Cropped mask (what the model sees)
        mask_tf_pipe = transforms.Compose([
            transforms.Resize(resize_px),
            transforms.CenterCrop(imagesize_px),
        ])
        arr_crop = np.where(np.array(mask_tf_pipe(mask_pil)) > 0, 255, 0).astype(np.uint8)
        ax_mask_crop.imshow(arr_crop, cmap="gray", vmin=0, vmax=255)
        ax_mask_crop.set_title("GT mask (crop)", fontsize=6.5, pad=3)

        # Full original mask with the crop box overlaid
        arr_full = np.where(np.array(mask_pil) > 0, 255, 0).astype(np.uint8)
        mw, mh = mask_pil.size
        ax_mask_full.imshow(arr_full, cmap="gray", vmin=0, vmax=255)
        mx0, my0, mbw, mbh = crop_box_on_original(mw, mh, resize_px, imagesize_px)
        ax_mask_full.add_patch(mpatches.FancyBboxPatch(
            (mx0, my0), mbw, mbh,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor="red", facecolor="none",
        ))
        ax_mask_full.set_title("GT mask (full)", fontsize=6.5, pad=3)
    else:
        ax_mask_crop.set_visible(False)
        ax_mask_full.set_visible(False)

    ax_mask_crop.axis("off")
    ax_mask_full.axis("off")


def make_figure(dataset_name, classname, splits, img_tf, resize_px, imagesize_px):
    """
    One figure per class.
    All samples laid out in a near-square grid.
    Each cell = original+crop-box | model-input | mask  (3 sub-columns).
    Normal cells have a green title; anomaly cells have a red title.
    """
    import math

    SUB = 4  # sub-columns per sample cell: orig | crop | mask-crop | mask-full

    # Flatten all entries, preserving split label per sample
    all_entries = []
    for split_label, entries in splits.items():
        for img_path, mask_path in entries:
            all_entries.append((split_label, img_path, mask_path))

    n_total = len(all_entries)
    if n_total == 0:
        return plt.figure()

    # Near-square grid: ceil(sqrt(n)) columns
    n_cols_blocks = min(n_total, math.ceil(math.sqrt(n_total)))
    n_rows_blocks = math.ceil(n_total / n_cols_blocks)

    fig_w = n_cols_blocks * SUB * 2.2
    fig_h = n_rows_blocks * 2.8
    fig, axes = plt.subplots(
        n_rows_blocks, n_cols_blocks * SUB,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for i, (split_label, img_path, mask_path) in enumerate(all_entries):
        row = i // n_cols_blocks
        col = i % n_cols_blocks
        c = col * SUB
        plot_entry(axes[row][c], axes[row][c + 1], axes[row][c + 2], axes[row][c + 3],
                   img_path, mask_path,
                   img_tf, resize_px, imagesize_px,
                   split_label=split_label)

    # Hide unused cells in the last row
    for i in range(n_total, n_rows_blocks * n_cols_blocks):
        row = i // n_cols_blocks
        col = i % n_cols_blocks
        c = col * SUB
        for ax in axes[row][c:c + SUB]:
            ax.set_visible(False)

    fig.suptitle(f"{dataset_name} — {classname}  |  "
                 f"Resize({resize_px}) → CenterCrop({imagesize_px})",
                 fontsize=8, y=1.01)
    fig.tight_layout()
    return fig



# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _free_path(directory, stem, ext=".png"):
    """
    Return a path <directory>/<stem><ext> that does not exist yet.
    If it does exist, try <stem>_2, <stem>_3, … until a free slot is found.
    """
    p = os.path.join(directory, stem + ext)
    if not os.path.exists(p):
        return p
    n = 2
    while True:
        p = os.path.join(directory, f"{stem}_{n}{ext}")
        if not os.path.exists(p):
            return p
        n += 1


# ────────────────────────────────────────────────────────────────────────────
# Custom image collector  (--images flag)
# ────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _resolve(path_str, roots):
    """Return first existing Path resolving path_str against each root in order."""
    p = pathlib.Path(path_str)
    if p.exists():
        return p
    for r in roots:
        if r:
            q = pathlib.Path(r) / p
            if q.exists():
                return q
    return None


def _detect_mask(img_path):
    """
    Try to find a ground-truth mask for img_path using MVTec and VisA conventions.

    MVTec layout:  .../test/<atype>/xxx.png
                →  .../ground_truth/<atype>/xxx_mask.png
    VisA layout:   .../Data/Images/Anomaly/xxx.jpg
                →  .../Data/Masks/Anomaly/xxx.png
    """
    p = pathlib.Path(img_path)
    stem = p.stem

    # MVTec: two levels up from test/<atype>/ is the class root
    gt_dir = p.parent.parent.parent / "ground_truth" / p.parent.name
    for suffix in (stem + "_mask.png", stem + ".png"):
        candidate = gt_dir / suffix
        if candidate.exists():
            return str(candidate)

    # VisA: swap Images/Anomaly → Masks/Anomaly, force .png extension
    visa_mask = pathlib.Path(
        str(p).replace("Images/Anomaly",  "Masks/Anomaly")
               .replace("Images\\Anomaly", "Masks\\Anomaly")
    ).with_suffix(".png")
    if visa_mask.exists():
        return str(visa_mask)

    return None


def _is_normal_path(p):
    """Return True if the path looks like a normal/good sample."""
    parts = str(p).lower().replace("\\", "/")
    return "/good/" in parts or "/normal/" in parts


def collect_custom(image_specs, roots):
    """
    image_specs : list of strings — each is either
                  • a path to a single image file, or
                  • a path to a directory (all images inside, sorted)
    roots       : iterable of base paths to try for relative paths
                  (e.g. [mvtec_path, visa_path])

    Returns dict  split_label → [(img_path, mask_path_or_None)]
    """
    splits = {}

    for spec in image_specs:
        resolved = _resolve(spec, roots)
        if resolved is None:
            print(f"[WARN] Cannot find: {spec}")
            continue

        if resolved.is_file():
            files = [resolved]
        elif resolved.is_dir():
            files = sorted(
                f for f in resolved.iterdir()
                if f.is_file() and f.suffix.lower() in _IMG_EXTS
            )
        else:
            print(f"[WARN] Not a file or directory: {resolved}")
            continue

        for f in files:
            label = "normal" if _is_normal_path(f) else "anomaly"
            mask  = _detect_mask(f)
            splits.setdefault(label, []).append((str(f), mask))

    return splits


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mvtec_path", default=None,
                   help="Path to MVTecAD root (optional)")
    p.add_argument("--visa_path",  default=None,
                   help="Path to VisA root (optional)")
    p.add_argument("--mvtec_classes", nargs="+", default=[],
                   help="MVTec classes to visualise")
    p.add_argument("--visa_classes", nargs="+", default=[],
                   help="VisA classes to visualise")
    p.add_argument("--resize",     type=int, default=256)
    p.add_argument("--imagesize",  type=int, default=224)
    p.add_argument("--n_anomaly",  type=int, default=4,
                   help="Anomalous samples shown per defect subtype")
    p.add_argument("--n_normal",   type=int, default=1,
                   help="Normal reference samples shown (usually 1 is enough)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out_dir",    default="results/sample_viz")
    p.add_argument(
        "--images", nargs="+", default=[], metavar="PATH",
        help=(
            "Specific images or directories to visualise directly, "
            "relative to --mvtec_path / --visa_path or absolute. "
            "Examples: 'capsules/Data/Images/Anomaly/030.jpg'  "
            "'bottle/test/broken_small'"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    img_tf, _ = make_transforms(args.resize, args.imagesize)

    # ── MVTecAD ─────────────────────────────────────────────────────────────
    if args.mvtec_path:
        for cls in [c.strip() for c in args.mvtec_classes]:
            cls_path = os.path.join(args.mvtec_path, cls)
            if not os.path.isdir(cls_path):
                print(f"[SKIP] MVTec class not found: {cls_path}")
                continue
            print(f"MVTec / {cls} ...")
            splits = collect_mvtec(args.mvtec_path, cls, args.n_anomaly, args.n_normal, rng)
            fig = make_figure("MVTecAD", cls, splits,
                              img_tf, args.resize, args.imagesize)
            out = _free_path(args.out_dir, f"mvtec_{cls}")
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {out}")

    # ── VisA ─────────────────────────────────────────────────────────────────
    if args.visa_path:
        for cls in [c.strip() for c in args.visa_classes]:
            cls_path = os.path.join(args.visa_path, cls)
            if not os.path.isdir(cls_path):
                print(f"[SKIP] VisA class not found: {cls_path}")
                continue
            print(f"VisA / {cls} ...")
            splits = collect_visa(args.visa_path, cls, args.n_anomaly, args.n_normal, rng)
            fig = make_figure("VisA", cls, splits,
                              img_tf, args.resize, args.imagesize)
            out = _free_path(args.out_dir, f"visa_{cls}")
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {out}")

    # ── Custom images (--images) — one figure per spec, paginated ───────────
    MAX_GRID = 64  # 8×8 maximum per figure page
    if args.images:
        roots = [args.mvtec_path, args.visa_path]
        for spec in args.images:
            splits = collect_custom([spec], roots)
            if not splits:
                print(f"[WARN] No valid images found for: {spec}")
                continue

            # Build a safe base name from the last two path components
            parts = pathlib.Path(spec.replace("\\", "/")).parts
            slug  = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            slug  = slug.replace(" ", "_")

            # Flatten, chunk into pages of MAX_GRID
            flat = [(lbl, ip, mp)
                    for lbl, entries in splits.items()
                    for ip, mp in entries]
            pages = [flat[i:i + MAX_GRID] for i in range(0, len(flat), MAX_GRID)]

            for page_idx, page in enumerate(pages):
                page_splits = {}
                for lbl, ip, mp in page:
                    page_splits.setdefault(lbl, []).append((ip, mp))
                suffix = f"_part{page_idx + 1}" if len(pages) > 1 else ""
                out = _free_path(args.out_dir, f"custom_{slug}{suffix}")
                fig = make_figure("custom", slug, page_splits,
                                  img_tf, args.resize, args.imagesize)
                fig.savefig(out, dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  -> {out}")

    print("Done.")


if __name__ == "__main__":
    main()
