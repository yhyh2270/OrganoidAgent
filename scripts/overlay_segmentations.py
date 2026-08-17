#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_images(path: Path):
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def load_original(path: Path):
    image = Image.open(path).convert("RGB")
    return image


def overlay(original, segmentation, alpha=140, threshold=10):
    seg = segmentation.convert("RGB")
    if seg.size != original.size:
        seg = seg.resize(original.size, Image.NEAREST)
    seg_arr = np.array(seg)
    mask = seg_arr.max(axis=2) > threshold
    alpha_mask = np.where(mask, alpha, 0).astype("uint8")
    seg_rgba = Image.fromarray(np.dstack([seg_arr, alpha_mask]), mode="RGBA")
    base = original.convert("RGBA")
    blended = Image.alpha_composite(base, seg_rgba)
    return blended.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Overlay segmentation masks onto an original image.")
    parser.add_argument(
        "--original",
        default="/home/lachlan/ProjectsLFS/OrganoidAgent/selected_data",
        help="Path to original image or directory.",
    )
    parser.add_argument(
        "--segmentations",
        default="/home/lachlan/ProjectsLFS/OrganoidAgent/segmentation",
        help="Path to segmentation image(s) or directory.",
    )
    parser.add_argument(
        "--output",
        default="/home/lachlan/ProjectsLFS/OrganoidAgent/segmentation/overlays",
        help="Output directory for overlays.",
    )
    parser.add_argument("--alpha", type=int, default=140, help="Overlay alpha for non-black pixels.")
    parser.add_argument("--threshold", type=int, default=10, help="Non-black threshold.")
    parser.add_argument("--format", default="jpg", help="Output format (jpg or png).")
    args = parser.parse_args()

    original_path = Path(args.original)
    seg_path = Path(args.segmentations)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_images = find_images(original_path)
    if not original_images:
        raise SystemExit(f"No original images found at {original_path}")
    original = load_original(original_images[0])

    seg_images = find_images(seg_path)
    if not seg_images:
        raise SystemExit(f"No segmentation images found at {seg_path}")

    combined = original
    for seg in seg_images:
        seg_img = Image.open(seg)
        blended = overlay(original, seg_img, alpha=args.alpha, threshold=args.threshold)
        out_name = f"overlay_{seg.stem}.{args.format}"
        blended.save(output_dir / out_name)
        combined = overlay(combined, seg_img, alpha=args.alpha, threshold=args.threshold)

    if len(seg_images) > 1:
        combined_name = f"overlay_combined.{args.format}"
        combined.save(output_dir / combined_name)

    print(f"Saved overlays to {output_dir}")


if __name__ == "__main__":
    main()
