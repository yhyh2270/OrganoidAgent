#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.utils import (
    DEFAULT_B2F_ROOT,
    DEFAULT_OUTPUT_ROOT,
    gray_rgb,
    green_rgb,
    heat_rgb,
    read_csv,
    read_gray_float,
    red_rgb,
    save_gray_uint8,
    save_grid,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned fluorescence-positive segmentation targets for Yichao B2F data.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_B2F_ROOT / "manifests" / "projected_instances_manifest.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preset", choices=["default", "relaxed"], default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qc-count", type=int, default=80)
    parser.add_argument("--max-overexposure-qc", type=int, default=-1, help="Maximum extra overexposure QC strips beyond qc-count; negative keeps old unlimited behavior.")
    parser.add_argument("--min-positive-fraction", type=float, default=0.0015)
    parser.add_argument("--min-positive-pixels", type=int, default=18)
    parser.add_argument("--candidate-mode", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--bg-mad-k", type=float, default=5.0)
    parser.add_argument("--bg-quantile", type=float, default=99.5)
    parser.add_argument("--bg-quantile-mad-k", type=float, default=1.5)
    parser.add_argument("--corrected-mad-k", type=float, default=None)
    parser.add_argument("--local-mad-k", type=float, default=2.0)
    parser.add_argument("--min-absolute", type=float, default=0.035)
    parser.add_argument("--min-local-contrast", type=float, default=0.012)
    parser.add_argument("--max-positive-fraction", type=float, default=0.42)
    parser.add_argument("--min-object-pixels", type=int, default=6)
    parser.add_argument("--min-object-area-fraction", type=float, default=0.00008)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--overexposure-refine-percentile", type=float, default=94.0)
    parser.add_argument("--overexposure-strict-percentile", type=float, default=98.0)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    apply_preset(args)
    return args


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "relaxed":
        return
    args.min_positive_fraction = 0.001
    args.min_positive_pixels = 12
    args.candidate_mode = "strict"
    args.bg_mad_k = 4.0
    args.bg_quantile = 99.3
    args.bg_quantile_mad_k = 1.0
    args.corrected_mad_k = 3.0
    args.local_mad_k = 1.5
    args.min_absolute = 0.025
    args.min_local_contrast = 0.008
    args.max_positive_fraction = 0.50
    args.min_object_pixels = 4
    args.min_object_area_fraction = 0.00004
    args.opening_radius = 0
    args.closing_radius = 1
    args.overexposure_refine_percentile = 92.0
    args.overexposure_strict_percentile = 98.0


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)) * 1.4826 + 1e-6)


def background_pixels(mask: np.ndarray, fluorescence: np.ndarray) -> np.ndarray:
    valid = mask > 0
    dilated = ndi.binary_dilation(valid, iterations=8)
    outside = ~dilated
    if int(outside.sum()) < 64:
        outside = ~valid
    if int(outside.sum()) < 64:
        border = np.zeros_like(valid, dtype=bool)
        border[:4, :] = True
        border[-4:, :] = True
        border[:, :4] = True
        border[:, -4:] = True
        outside = border
    values = fluorescence[outside]
    if values.size < 64:
        values = fluorescence.reshape(-1)
    return values.astype(np.float32)


def suppress_full_field_exposure(
    candidate: np.ndarray,
    local_contrast: np.ndarray,
    valid: np.ndarray,
    *,
    max_fraction: float,
    refine_percentile: float,
    strict_percentile: float,
) -> tuple[np.ndarray, bool]:
    fraction = float(candidate.sum() / max(valid.sum(), 1))
    if fraction <= max_fraction:
        return candidate, False
    inside_contrast = local_contrast[valid]
    high_contrast_threshold = max(float(np.percentile(inside_contrast, refine_percentile)), 0.018)
    refined = candidate & (local_contrast > high_contrast_threshold)
    refined_fraction = float(refined.sum() / max(valid.sum(), 1))
    if refined_fraction <= max_fraction:
        return refined, True
    # Uniform or overexposed fields should not become a dense positive label.
    very_high = local_contrast > max(float(np.percentile(inside_contrast, strict_percentile)), 0.025)
    return candidate & very_high, True


def build_target(
    brightfield: np.ndarray,
    fluorescence: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = mask > 0.5
    if int(valid.sum()) == 0:
        zeros = np.zeros_like(fluorescence, dtype=np.float32)
        return zeros, zeros, {"target_status": "empty_mask"}

    bg = background_pixels(valid.astype(np.uint8), fluorescence)
    bg_median = float(np.median(bg))
    bg_mad = robust_mad(bg)
    bg_q = float(np.percentile(bg, args.bg_quantile))
    corrected = fluorescence - bg_median

    sigma = max(2.0, min(fluorescence.shape) / 36.0)
    smooth = filters.gaussian(fluorescence, sigma=sigma, preserve_range=True)
    local_contrast = fluorescence - smooth
    local_mad = robust_mad(local_contrast[~valid]) if int((~valid).sum()) >= 64 else robust_mad(local_contrast.reshape(-1))

    raw_threshold = max(
        bg_median + args.bg_mad_k * bg_mad,
        bg_q + args.bg_quantile_mad_k * bg_mad,
        args.min_absolute,
    )
    contrast_threshold = max(args.local_mad_k * local_mad, args.min_local_contrast)
    corrected_mad_k = args.bg_mad_k if args.corrected_mad_k is None else args.corrected_mad_k

    raw_candidate = (fluorescence > raw_threshold) & valid
    contrast_candidate = (local_contrast > contrast_threshold) & valid
    corrected_candidate = (corrected > max(corrected_mad_k * bg_mad, args.min_absolute)) & valid
    high_raw_threshold = max(
        bg_median + (args.bg_mad_k + 2.0) * bg_mad,
        float(np.percentile(bg, min(99.9, max(args.bg_quantile, 99.0)))),
        args.min_absolute * 1.5,
    )
    high_raw_candidate = (fluorescence > high_raw_threshold) & valid
    if args.candidate_mode == "relaxed":
        candidate = valid & (
            (raw_candidate & (contrast_candidate | corrected_candidate))
            | (corrected_candidate & contrast_candidate)
            | high_raw_candidate
        )
    else:
        candidate = raw_candidate & (contrast_candidate | corrected_candidate)

    candidate, overexposure_refined = suppress_full_field_exposure(
        candidate,
        local_contrast,
        valid,
        max_fraction=args.max_positive_fraction,
        refine_percentile=args.overexposure_refine_percentile,
        strict_percentile=args.overexposure_strict_percentile,
    )

    min_area = max(args.min_object_pixels, int(args.min_object_area_fraction * valid.sum()))
    positive_bool = morphology.remove_small_objects(candidate.astype(bool), min_size=min_area)
    if args.opening_radius > 0:
        positive_bool = morphology.binary_opening(positive_bool, morphology.disk(args.opening_radius))
    if args.closing_radius > 0:
        positive_bool = morphology.binary_closing(positive_bool, morphology.disk(args.closing_radius))

    # Ambiguous pixels are raw-bright fluorescence that failed morphology/local-contrast checks.
    saturated = (fluorescence > 0.985) & valid
    raw_high = (fluorescence > raw_threshold) & valid
    central_large = raw_high & ~positive_bool
    ignore_bool = (central_large | saturated) & valid
    ignore_bool = morphology.binary_dilation(ignore_bool, morphology.disk(1)) & valid & ~positive_bool

    positive = positive_bool.astype(np.float32)
    ignore = ignore_bool.astype(np.float32)
    positive_fraction = float(positive.sum() / max(valid.sum(), 1))
    ignore_fraction = float(ignore.sum() / max(valid.sum(), 1))
    global_positive = bool(positive.sum() >= args.min_positive_pixels and positive_fraction >= args.min_positive_fraction)

    outside_high_fraction = float(((fluorescence > raw_threshold) & ~valid).sum() / max((~valid).sum(), 1))
    full_field_high_fraction = float((fluorescence > raw_threshold).sum() / fluorescence.size)
    valid_raw_high_fraction = float(raw_high.sum() / max(valid.sum(), 1))
    target_status = "positive" if global_positive else "negative"
    if overexposure_refined or (full_field_high_fraction > 0.55 and positive_fraction < args.min_positive_fraction):
        target_status = "overexposure_suppressed"

    metrics = {
        "target_status": target_status,
        "bg_median": bg_median,
        "bg_mad": bg_mad,
        "bg_quantile_value": bg_q,
        "raw_threshold": raw_threshold,
        "high_raw_threshold": high_raw_threshold,
        "local_contrast_threshold": contrast_threshold,
        "local_mad": local_mad,
        "positive_pixels": int(positive.sum()),
        "ignore_pixels": int(ignore.sum()),
        "mask_pixels": int(valid.sum()),
        "positive_fraction": positive_fraction,
        "ignore_fraction": ignore_fraction,
        "global_positive_label": int(global_positive),
        "outside_high_fraction": outside_high_fraction,
        "full_field_high_fraction": full_field_high_fraction,
        "valid_raw_high_fraction": valid_raw_high_fraction,
        "overexposure_refined": int(overexposure_refined),
        "min_area": min_area,
    }
    return positive, ignore, metrics


def target_paths(output_root: Path, row: dict[str, str]) -> tuple[Path, Path, Path]:
    safe = row["instance_id"].replace("/", "__").replace(":", "_")
    dataset = row.get("dataset", "unknown")
    base = output_root / "targets" / dataset
    return base / "positive" / f"{safe}.png", base / "ignore" / f"{safe}.png", base / "qc" / f"{safe}.png"


def make_qc_image(
    path: Path,
    brightfield: np.ndarray,
    fluorescence: np.ndarray,
    mask: np.ndarray,
    positive: np.ndarray,
    ignore: np.ndarray,
    metrics: dict[str, Any],
    label: str,
) -> None:
    corrected = np.clip(fluorescence - float(metrics.get("bg_median", 0.0)), 0, 1)
    overlay = gray_rgb(brightfield)
    pos_img = green_rgb(positive)
    ign_img = red_rgb(ignore * 0.7)
    overlay = Image.blend(overlay, pos_img, 0.45)
    overlay = Image.blend(overlay, ign_img, 0.35)
    save_grid(
        path,
        [[gray_rgb(brightfield), green_rgb(fluorescence), heat_rgb(corrected), gray_rgb(mask), overlay]],
        ["brightfield", "raw F", "bg-corrected", "mask", "target overlay"],
        [label],
        tile=180,
    )


def write_qc_gallery(output_root: Path, qc_paths: list[Path], rows: list[dict[str, Any]], limit: int) -> None:
    if not qc_paths:
        return
    selected = qc_paths[:limit]
    images = []
    labels = []
    for path, row in zip(selected, rows[:limit]):
        with Image.open(path) as image:
            images.append([image.convert("RGB")])
        labels.append(
            f"{row.get('dataset')} status={row.get('target_status')} "
            f"pos={float(row.get('target_positive_fraction', 0)):.3f} ignore={float(row.get('target_ignore_fraction', 0)):.3f}"
        )
    # Keep the gallery compact by reusing each QC strip as one large tile.
    save_grid(output_root / "qc" / "target_qc_gallery.png", images, ["target QC strip"], labels, tile=900)


def run_self_test(args: argparse.Namespace) -> None:
    size = 160
    y, x = np.mgrid[:size, :size]
    mask = ((x - 80) ** 2 + (y - 80) ** 2) < 58**2
    brightfield = np.clip(0.4 + 0.2 * np.sin(x / 9) + 0.1 * np.cos(y / 13), 0, 1).astype(np.float32)
    fl_uniform = np.ones((size, size), dtype=np.float32) * 0.85
    pos, ign, metrics = build_target(brightfield, fl_uniform, mask.astype(np.float32), args)
    if float(pos.sum()) / max(float(mask.sum()), 1.0) > 0.01:
        raise RuntimeError(f"Self-test failed: uniform overexposure created positives: {metrics}")
    fl_spots = np.ones((size, size), dtype=np.float32) * 0.06
    for cx, cy in [(50, 70), (92, 90), (112, 62)]:
        fl_spots += np.exp(-(((x - cx) / 5) ** 2 + ((y - cy) / 12) ** 2)) * 0.8
    fl_spots = np.clip(fl_spots, 0, 1)
    pos2, _, metrics2 = build_target(brightfield, fl_spots, mask.astype(np.float32), args)
    if int(pos2.sum()) < 50:
        raise RuntimeError(f"Self-test failed: spot fluorescence was not detected: {metrics2}")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test(args)
        print(json.dumps({"stage": "target_self_test_passed"}, indent=2), flush=True)
        return 0

    rows = read_csv(args.source_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    out_rows: list[dict[str, Any]] = []
    qc_paths: list[Path] = []
    qc_rows: list[dict[str, Any]] = []
    overexposure_qc_count = 0
    for index, row in enumerate(rows):
        brightfield = read_gray_float(Path(row["brightfield_crop_path"]))
        fluorescence = read_gray_float(Path(row["fluorescence_crop_path"]))
        mask = read_gray_float(Path(row["mask_crop_path"]))
        if fluorescence.shape != brightfield.shape or mask.shape != brightfield.shape:
            raise RuntimeError(f"Shape mismatch for {row['instance_id']}: B={brightfield.shape} F={fluorescence.shape} M={mask.shape}")
        positive, ignore, metrics = build_target(brightfield, fluorescence, mask, args)
        pos_path, ignore_path, qc_path = target_paths(args.output_root, row)
        save_gray_uint8(pos_path, positive)
        save_gray_uint8(ignore_path, ignore)
        is_overexposure = metrics["target_status"] == "overexposure_suppressed"
        allow_extra_overexposure = args.max_overexposure_qc < 0 or overexposure_qc_count < args.max_overexposure_qc
        should_write_qc = len(qc_paths) < args.qc_count or (is_overexposure and allow_extra_overexposure)
        if should_write_qc:
            label = f"{index:05d} {row['instance_id'][:120]} | {metrics['target_status']}"
            make_qc_image(qc_path, brightfield, fluorescence, mask, positive, ignore, metrics, label)
            qc_paths.append(qc_path)
            if is_overexposure and len(qc_paths) >= args.qc_count:
                overexposure_qc_count += 1
        out = {
            **row,
            "positive_mask_path": str(pos_path),
            "ignore_mask_path": str(ignore_path),
            "target_positive_fraction": metrics["positive_fraction"],
            "target_ignore_fraction": metrics["ignore_fraction"],
            "target_positive_pixels": metrics["positive_pixels"],
            "target_ignore_pixels": metrics["ignore_pixels"],
            "target_global_positive": metrics["global_positive_label"],
            "target_status": metrics["target_status"],
            "target_bg_median": metrics["bg_median"],
            "target_bg_mad": metrics["bg_mad"],
            "target_raw_threshold": metrics["raw_threshold"],
            "target_high_raw_threshold": metrics["high_raw_threshold"],
            "target_local_contrast_threshold": metrics["local_contrast_threshold"],
            "target_outside_high_fraction": metrics["outside_high_fraction"],
            "target_full_field_high_fraction": metrics["full_field_high_fraction"],
            "target_valid_raw_high_fraction": metrics["valid_raw_high_fraction"],
            "target_overexposure_refined": metrics["overexposure_refined"],
            "target_generation_version": f"v1_context_clean_threshold_{args.preset}_{args.candidate_mode}",
        }
        out_rows.append(out)
        qc_rows.append(out)
        if (index + 1) % 500 == 0:
            print(json.dumps({"processed": index + 1, "total": len(rows)}), flush=True)

    manifest_path = args.output_root / "manifests" / "segmentation_targets_manifest.csv"
    write_csv(manifest_path, out_rows)
    write_qc_gallery(args.output_root, qc_paths, qc_rows, min(args.qc_count, len(qc_paths)))
    statuses: dict[str, int] = {}
    by_split: dict[str, dict[str, int]] = {}
    for row in out_rows:
        statuses[row["target_status"]] = statuses.get(row["target_status"], 0) + 1
        split = row.get("split", "unknown")
        by_split.setdefault(split, {"total": 0, "positive": 0})
        by_split[split]["total"] += 1
        by_split[split]["positive"] += int(float(row["target_global_positive"]))
    summary = {
        "source_manifest": str(args.source_manifest),
        "target_manifest": str(manifest_path),
        "count": len(out_rows),
        "statuses": statuses,
        "by_split": by_split,
        "qc_gallery": str(args.output_root / "qc" / "target_qc_gallery.png"),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_root / "manifests" / "segmentation_targets_summary.json", summary)
    print(json.dumps({"stage": "targets_finished", **summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
