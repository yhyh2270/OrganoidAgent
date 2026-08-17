#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.utils import (
    DEFAULT_OUTPUT_ROOT,
    gray_rgb,
    green_rgb,
    load_font,
    read_csv,
    read_gray_float,
    red_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render exact B/F/F-mask training examples from the Yichao target manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifests" / "segmentation_targets_manifest.csv")
    parser.add_argument("--output", type=Path, default=Path("visualizations") / "yichao_training_b_f_mask_15x10.png")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--groups", type=int, default=5, help="Number of B/F/mask triplets per row. 5 groups gives 15 columns.")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="train")
    parser.add_argument("--mix-status", action="store_true", help="Sample positives, negatives, and overexposure-suppressed examples evenly when possible.")
    parser.add_argument("--clean-mask-only", action="store_true", help="Render the third tile as final positive mask only, without red ignore overlay.")
    parser.add_argument("--mask-only", action="store_true", help="Render a pure grid of positive masks only, one mask tile per example.")
    parser.add_argument("--continuous-target", action="store_true", help="Render the third tile as mask times background-suppressed continuous fluorescence.")
    parser.add_argument("--continuous-scale", type=float, default=4.0, help="Display gain for background-suppressed continuous fluorescence.")
    parser.add_argument("--continuous-plus-mask", action="store_true", help="Render B/F/continuous-suppressed-F/binary-mask as four columns per example.")
    parser.add_argument(
        "--binary-mask-source",
        choices=["positive", "suppressed", "suppressed-soft"],
        default="positive",
        help="Mask shown in the fourth column. 'suppressed' thresholds the rendered continuous target; 'suppressed-soft' uses the continuous target directly.",
    )
    parser.add_argument("--suppressed-mask-threshold", type=float, default=0.02, help="Threshold applied to the displayed suppressed target when --binary-mask-source=suppressed.")
    parser.add_argument("--suppressed-mask-min-area", type=int, default=6, help="Remove smaller connected components from the suppressed-derived mask.")
    parser.add_argument("--suppressed-mask-close", type=int, default=1, help="Binary closing iterations for the suppressed-derived mask.")
    parser.add_argument("--suppressed-mask-dilate", type=int, default=0, help="Optional binary dilation iterations for the suppressed-derived mask.")
    parser.add_argument(
        "--continuous-mode",
        choices=["hard-mask", "soft-mask", "bg-only"],
        default="hard-mask",
        help="Continuous target rendering mode. soft-mask dilates/blurs the mask; bg-only shows background-corrected fluorescence without mask multiplication.",
    )
    parser.add_argument("--soft-mask-dilate", type=int, default=5)
    parser.add_argument("--soft-mask-sigma", type=float, default=2.0)
    parser.add_argument("--soft-mask-floor", type=float, default=0.18)
    return parser.parse_args()


def resize(image: Image.Image, tile: int, *, mask: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    return image.resize((tile, tile), resample)


def mask_overlay(mask: np.ndarray, ignore: np.ndarray) -> Image.Image:
    base = Image.new("RGB", (mask.shape[1], mask.shape[0]), (0, 0, 0))
    pos = green_rgb(mask)
    ign = red_rgb(ignore * 0.75)
    base = Image.blend(base, pos, 0.95)
    base = Image.blend(base, ign, 0.55)
    return base


def continuous_gate(
    positive: np.ndarray,
    *,
    mode: str,
    dilate: int,
    sigma: float,
    floor: float,
) -> np.ndarray:
    if mode == "bg-only":
        return np.ones_like(positive, dtype=np.float32)
    if mode == "hard-mask":
        return (positive > 0.5).astype(np.float32)
    mask = positive > 0.5
    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=dilate)
    soft = ndi.gaussian_filter(mask.astype(np.float32), sigma=max(sigma, 0.0))
    max_value = float(soft.max())
    if max_value > 0:
        soft /= max_value
    return np.clip(np.maximum(soft, float(floor) * (positive > 0.5)), 0.0, 1.0).astype(np.float32)


def robust_bg_suppressed(
    fluorescence: np.ndarray,
    positive: np.ndarray,
    row: dict[str, str],
    scale: float,
    *,
    mode: str,
    dilate: int,
    sigma: float,
    floor: float,
) -> np.ndarray:
    bg = float(row.get("target_bg_median") or 0.0)
    corrected = np.maximum(fluorescence - bg, 0.0)
    gate = continuous_gate(positive, mode=mode, dilate=dilate, sigma=sigma, floor=floor)
    return np.clip(corrected * gate * scale, 0.0, 1.0)


def mask_from_suppressed(
    suppressed: np.ndarray,
    *,
    threshold: float,
    min_area: int,
    close: int,
    dilate: int,
) -> np.ndarray:
    mask = np.asarray(suppressed, dtype=np.float32) >= float(threshold)
    if close > 0:
        mask = ndi.binary_closing(mask, iterations=int(close))
    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=int(dilate))
    if min_area > 1 and bool(mask.any()):
        labels, label_count = ndi.label(mask)
        if label_count > 0:
            counts = np.bincount(labels.reshape(-1))
            keep = counts >= int(min_area)
            keep[0] = False
            mask = keep[labels]
    return mask.astype(np.float32)


def choose_rows(rows: list[dict[str, str]], count: int, seed: int, mix_status: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    if not mix_status:
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:count]
    selected: list[dict[str, str]] = []
    statuses = ["positive", "negative", "overexposure_suppressed"]
    buckets = {status: [row for row in rows if row.get("target_status") == status] for status in statuses}
    per_status = max(1, count // len(statuses))
    for status in statuses:
        bucket = buckets[status]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_status])
    remaining = [row for row in rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    rng.shuffle(selected)
    return selected[:count]


def render_grid(
    rows: list[dict[str, str]],
    output: Path,
    groups: int,
    row_count: int,
    tile: int,
    *,
    clean_mask_only: bool = False,
    continuous_target: bool = False,
    continuous_scale: float = 4.0,
    continuous_mode: str = "hard-mask",
    soft_mask_dilate: int = 5,
    soft_mask_sigma: float = 2.0,
    soft_mask_floor: float = 0.18,
) -> None:
    header_h = 34
    label_h = 34
    columns = groups * 3
    width = columns * tile
    height = header_h + row_count * (tile + label_h)
    canvas = Image.new("RGB", (width, height), (16, 19, 22))
    draw = ImageDraw.Draw(canvas)
    font = load_font(12)
    header_font = load_font(14)
    for group in range(groups):
        third_header = "F-bg-suppressed" if continuous_target else "F-mask"
        for offset, header in enumerate(("B", "F", third_header)):
            x = (group * 3 + offset) * tile + 6
            draw.text((x, 9), header, fill=(235, 240, 242), font=header_font)
    for index, row in enumerate(rows[: row_count * groups]):
        grid_row = index // groups
        group = index % groups
        x0 = group * 3 * tile
        y0 = header_h + grid_row * (tile + label_h)
        brightfield = gray_rgb(read_gray_float(Path(row["brightfield_crop_path"])))
        fluorescence_arr = read_gray_float(Path(row["fluorescence_crop_path"]))
        fluorescence = green_rgb(fluorescence_arr)
        positive = read_gray_float(Path(row["positive_mask_path"]))
        ignore = read_gray_float(Path(row["ignore_mask_path"]))
        if continuous_target:
            mask = green_rgb(
                robust_bg_suppressed(
                    fluorescence_arr,
                    positive,
                    row,
                    continuous_scale,
                    mode=continuous_mode,
                    dilate=soft_mask_dilate,
                    sigma=soft_mask_sigma,
                    floor=soft_mask_floor,
                )
            )
        else:
            mask = gray_rgb(positive) if clean_mask_only else mask_overlay(positive, ignore)
        canvas.paste(resize(brightfield, tile), (x0, y0))
        canvas.paste(resize(fluorescence, tile), (x0 + tile, y0))
        canvas.paste(resize(mask, tile, mask=True), (x0 + tile * 2, y0))
        label = (
            f"{index:02d} {row.get('split')} {row.get('target_status')} "
            f"pos={float(row.get('target_positive_fraction', 0)):.3f} "
            f"{row.get('dataset')}"
        )
        draw.text((x0 + 4, y0 + tile + 5), label[:48], fill=(188, 196, 202), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_mask_only_grid(rows: list[dict[str, str]], output: Path, columns: int, row_count: int, tile: int) -> None:
    label_h = 28
    width = columns * tile
    height = row_count * (tile + label_h)
    canvas = Image.new("RGB", (width, height), (10, 12, 14))
    draw = ImageDraw.Draw(canvas)
    font = load_font(11)
    for index, row in enumerate(rows[: row_count * columns]):
        grid_row = index // columns
        col = index % columns
        x0 = col * tile
        y0 = grid_row * (tile + label_h)
        positive = read_gray_float(Path(row["positive_mask_path"]))
        mask = gray_rgb(positive)
        canvas.paste(resize(mask, tile, mask=True), (x0, y0))
        label = f"{index:02d} {row.get('target_status')} {float(row.get('target_positive_fraction', 0)):.3f}"
        draw.text((x0 + 4, y0 + tile + 5), label[:22], fill=(190, 198, 204), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_continuous_plus_mask_grid(
    rows: list[dict[str, str]],
    output: Path,
    groups: int,
    row_count: int,
    tile: int,
    *,
    continuous_scale: float,
    continuous_mode: str,
    soft_mask_dilate: int,
    soft_mask_sigma: float,
    soft_mask_floor: float,
    binary_mask_source: str,
    suppressed_mask_threshold: float,
    suppressed_mask_min_area: int,
    suppressed_mask_close: int,
    suppressed_mask_dilate: int,
) -> None:
    header_h = 34
    label_h = 34
    columns = groups * 4
    width = columns * tile
    height = header_h + row_count * (tile + label_h)
    canvas = Image.new("RGB", (width, height), (16, 19, 22))
    draw = ImageDraw.Draw(canvas)
    font = load_font(12)
    header_font = load_font(14)
    if binary_mask_source == "suppressed-soft":
        mask_header = "F-soft mask"
    elif binary_mask_source == "suppressed":
        mask_header = "F-binary mask"
    else:
        mask_header = "F-binary mask"
    for group in range(groups):
        for offset, header in enumerate(("B", "F", "F-suppressed", mask_header)):
            x = (group * 4 + offset) * tile + 6
            draw.text((x, 9), header, fill=(235, 240, 242), font=header_font)
    for index, row in enumerate(rows[: row_count * groups]):
        grid_row = index // groups
        group = index % groups
        x0 = group * 4 * tile
        y0 = header_h + grid_row * (tile + label_h)
        brightfield = gray_rgb(read_gray_float(Path(row["brightfield_crop_path"])))
        fluorescence_arr = read_gray_float(Path(row["fluorescence_crop_path"]))
        fluorescence = green_rgb(fluorescence_arr)
        positive = read_gray_float(Path(row["positive_mask_path"]))
        suppressed_arr = robust_bg_suppressed(
            fluorescence_arr,
            positive,
            row,
            continuous_scale,
            mode=continuous_mode,
            dilate=soft_mask_dilate,
            sigma=soft_mask_sigma,
            floor=soft_mask_floor,
        )
        suppressed = green_rgb(suppressed_arr)
        if binary_mask_source == "suppressed-soft":
            binary_mask_arr = suppressed_arr
        elif binary_mask_source == "suppressed":
            binary_mask_arr = mask_from_suppressed(
                suppressed_arr,
                threshold=suppressed_mask_threshold,
                min_area=suppressed_mask_min_area,
                close=suppressed_mask_close,
                dilate=suppressed_mask_dilate,
            )
        else:
            binary_mask_arr = positive
        binary_mask = gray_rgb(binary_mask_arr)
        canvas.paste(resize(brightfield, tile), (x0, y0))
        canvas.paste(resize(fluorescence, tile), (x0 + tile, y0))
        canvas.paste(resize(suppressed, tile), (x0 + tile * 2, y0))
        canvas.paste(resize(binary_mask, tile, mask=binary_mask_source != "suppressed-soft"), (x0 + tile * 3, y0))
        label = (
            f"{index:02d} {row.get('split')} {row.get('target_status')} "
            f"pos={float(row.get('target_positive_fraction', 0)):.3f} "
            f"{row.get('dataset')}"
        )
        draw.text((x0 + 4, y0 + tile + 5), label[:64], fill=(188, 196, 202), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    args = parse_args()
    rows = read_csv(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    if not rows:
        raise SystemExit(f"No rows available for split={args.split} in {args.manifest}")
    count = args.rows * args.groups * 3 if args.mask_only else args.rows * args.groups
    selected = choose_rows(rows, count, args.seed, args.mix_status)
    if args.mask_only:
        render_mask_only_grid(selected, args.output, args.groups * 3, args.rows, args.tile)
    elif args.continuous_plus_mask:
        render_continuous_plus_mask_grid(
            selected,
            args.output,
            args.groups,
            args.rows,
            args.tile,
            continuous_scale=args.continuous_scale,
            continuous_mode=args.continuous_mode,
            soft_mask_dilate=args.soft_mask_dilate,
            soft_mask_sigma=args.soft_mask_sigma,
            soft_mask_floor=args.soft_mask_floor,
            binary_mask_source=args.binary_mask_source,
            suppressed_mask_threshold=args.suppressed_mask_threshold,
            suppressed_mask_min_area=args.suppressed_mask_min_area,
            suppressed_mask_close=args.suppressed_mask_close,
            suppressed_mask_dilate=args.suppressed_mask_dilate,
        )
    else:
        render_grid(
            selected,
            args.output,
            args.groups,
            args.rows,
            args.tile,
            clean_mask_only=args.clean_mask_only,
            continuous_target=args.continuous_target,
            continuous_scale=args.continuous_scale,
            continuous_mode=args.continuous_mode,
            soft_mask_dilate=args.soft_mask_dilate,
            soft_mask_sigma=args.soft_mask_sigma,
            soft_mask_floor=args.soft_mask_floor,
        )
    sidecar = args.output.with_suffix(".txt")
    with sidecar.open("w", encoding="utf-8") as handle:
        handle.write(f"manifest={args.manifest}\n")
        handle.write(f"split={args.split}\n")
        handle.write(f"rows={args.rows}\n")
        handle.write(f"groups={args.groups}\n")
        handle.write(f"tile={args.tile}\n")
        handle.write(f"seed={args.seed}\n")
        handle.write(f"mix_status={args.mix_status}\n")
        handle.write(f"clean_mask_only={args.clean_mask_only}\n")
        handle.write(f"mask_only={args.mask_only}\n")
        handle.write(f"continuous_plus_mask={args.continuous_plus_mask}\n")
        handle.write(f"continuous_target={args.continuous_target}\n")
        handle.write(f"continuous_scale={args.continuous_scale}\n")
        handle.write(f"continuous_mode={args.continuous_mode}\n")
        handle.write(f"binary_mask_source={args.binary_mask_source}\n")
        handle.write(f"suppressed_mask_threshold={args.suppressed_mask_threshold}\n")
        handle.write(f"suppressed_mask_min_area={args.suppressed_mask_min_area}\n")
        handle.write(f"suppressed_mask_close={args.suppressed_mask_close}\n")
        handle.write(f"suppressed_mask_dilate={args.suppressed_mask_dilate}\n")
        handle.write(f"soft_mask_dilate={args.soft_mask_dilate}\n")
        handle.write(f"soft_mask_sigma={args.soft_mask_sigma}\n")
        handle.write(f"soft_mask_floor={args.soft_mask_floor}\n")
        for index, row in enumerate(selected):
            handle.write(
                f"{index}\t{row.get('split')}\t{row.get('target_status')}\t"
                f"{row.get('target_positive_fraction')}\t{row.get('dataset')}\t{row.get('instance_id')}\n"
            )
    print(args.output)
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
