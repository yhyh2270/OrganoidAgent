#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
APP81_DIR = ROOT / "analysis-tools" / "app81_density_multiscale_cellpose"
APP80_DIR = ROOT / "analysis-tools" / "app80_first_replicate_multiscale_cellpose"
APP65_DIR = ROOT / "analysis-tools" / "app65_alginate_multiscale_cellpose"
sys.path.insert(0, str(APP81_DIR))
sys.path.insert(0, str(APP80_DIR))
sys.path.insert(0, str(APP65_DIR))

import run_app81_main_density_large_recovery as app81  # noqa: E402
import run_app65_all_conditions_large_recovery as app65  # noqa: E402
import backfill_app81_full_metrics as full_metrics  # noqa: E402
import run_multiscale_dateaware_cellpose as seg  # noqa: E402

EXPERIMENT_CONDITION_ORDERS = {
    "density": ["low", "middle", "high"],
    "y27632": ["No", "10uM", "20uM", "50uM", "100uM"],
    "sodium_alginate": ["No", "0.02% Alginate", "0.05% Alginate"],
}
DEFAULT_INPUTS = [
    ROOT / "datasets" / "01_Density_experiment_10x" / "high_D12_10x__10x_high.tif",
    ROOT / "datasets" / "01_Density_experiment_10x" / "middle_D12_10x__10x_middle.tif",
    ROOT / "datasets" / "01_Density_experiment_10x" / "low_D12_10x__10x_low.tif",
]
DEFAULT_OUT = ROOT / "analysis-outputs" / "density_growth_d12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-file")
    parser.add_argument("--instruction", default="")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_CONDITION_ORDERS),
        default="density",
        help="Experiment adapter used for condition parsing, stage parameters, grouping, and reporting.",
    )
    return parser.parse_args()


def write_progress(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_condition(path: Path, experiment: str = "density") -> str:
    name = path.name.lower()
    if experiment == "density":
        for condition in EXPERIMENT_CONDITION_ORDERS[experiment]:
            if condition in name:
                return condition
        raise ValueError(f"Cannot infer density condition from {path.name}")
    if experiment == "y27632":
        match = re.search(r"y27632_(\d+)um_", name)
        if not match:
            raise ValueError(f"Cannot infer Y-27632 concentration from {path.name}")
        concentration = int(match.group(1))
        condition = "No" if concentration == 0 else f"{concentration}uM"
        if condition not in EXPERIMENT_CONDITION_ORDERS[experiment]:
            raise ValueError(f"Unsupported Y-27632 concentration in {path.name}")
        return condition
    if experiment == "sodium_alginate":
        if name.startswith("control_"):
            return "No"
        match = re.search(r"alginate_(0\.0[25])pct_", name)
        if match:
            return f"{match.group(1)}% Alginate"
        raise ValueError(f"Cannot infer sodium alginate condition from {path.name}")
    raise ValueError(f"Unsupported experiment adapter: {experiment}")


def parse_day(path: Path) -> int:
    match = re.search(r"_D(\d{2})_", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer day from {path.name}")
    return int(match.group(1))


def stage_and_diameters(experiment: str, condition: str, day: int) -> dict:
    if experiment == "density":
        return app81.stage_and_diameters(condition, day)
    if experiment == "sodium_alginate":
        return app65.stage_and_diameters(condition, day)
    if day <= 0:
        return {"stage": "early_cluster", "diameters": [90, 180, 320]}
    if day <= 2:
        return {"stage": "cystic_early", "diameters": [180, 320, 480]}
    if day <= 4:
        return {"stage": "cystic_mid", "diameters": [220, 380, 560]}
    if day <= 6:
        return {"stage": "fused_large", "diameters": [420, 780, 1100]}
    return {"stage": "differentiated_irregular", "diameters": [320, 600, 900]}


def condition_slug(condition: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", condition).strip("_")
    return slug or "condition"


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"OpenCV failed to encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def cellpose_quality(kept: list[seg.Candidate], support: np.ndarray) -> dict:
    """Decide whether pure multiscale Cellpose is plausible before using fallback masks."""
    support_mask = support > 0
    union = np.zeros(support.shape, dtype=bool)
    for candidate in kept:
        union |= candidate.mask
    mask_area = int(union.sum())
    support_area = int(support_mask.sum())
    overlap = int(np.logical_and(union, support_mask).sum())
    support_coverage = overlap / support_area if support_area else 0.0
    mask_support_precision = overlap / mask_area if mask_area else 0.0
    foreground_fraction = mask_area / union.size if union.size else 0.0
    reasons = []
    if not kept:
        reasons.append("no_cellpose_candidates")
    if support_area and support_coverage < 0.35:
        reasons.append("low_support_coverage")
    # The support mask intentionally captures the strongest internal signal, not
    # the complete bright-field organoid.  Keep this as a loose corruption check
    # so a visually complete Cellpose mask (for example the low-density sample)
    # is not rejected merely because its interior is relatively homogeneous.
    if mask_area and mask_support_precision < 0.30:
        reasons.append("low_mask_support_precision")
    if foreground_fraction > 0.82:
        reasons.append("implausibly_large_foreground")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "candidate_count": len(kept),
        "mask_area_px": mask_area,
        "support_area_px": support_area,
        "support_coverage": round(support_coverage, 4),
        "mask_support_precision": round(mask_support_precision, 4),
        "foreground_fraction": round(foreground_fraction, 4),
    }


def segmentation_quality(
    kept: list[seg.Candidate], support: np.ndarray, signal: np.ndarray, grad_norm: np.ndarray
) -> dict:
    """Score a complete candidate set against the same image evidence."""
    union = np.zeros(support.shape, dtype=bool)
    for candidate in kept:
        union |= candidate.mask
    mask_area = int(union.sum())
    support_mask = support > 0
    support_area = int(support_mask.sum())
    overlap = int(np.logical_and(union, support_mask).sum())
    support_coverage = overlap / support_area if support_area else 0.0
    foreground_fraction = mask_area / union.size if union.size else 0.0

    soft_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    soft_support = cv2.dilate(support_mask.astype(np.uint8), soft_kernel) > 0
    soft_support_precision = (
        float(np.logical_and(union, soft_support).sum()) / mask_area if mask_area else 0.0
    )
    boundary = cv2.morphologyEx(
        union.astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ) > 0
    boundary_strength = float(grad_norm[boundary].mean() / 255.0) if boundary.any() else 0.0
    mean_signal = float(signal[union].mean() / 255.0) if mask_area else 0.0

    coverage_score = min(support_coverage / 0.75, 1.0)
    precision_score = min(soft_support_precision / 0.85, 1.0)
    boundary_score = min(boundary_strength / 0.12, 1.0)
    signal_score = min(mean_signal / 0.28, 1.0)
    if foreground_fraction <= 0.60:
        foreground_score = 1.0
    elif foreground_fraction >= 0.82:
        foreground_score = 0.0
    else:
        foreground_score = (0.82 - foreground_fraction) / 0.22
    score = (
        0.30 * coverage_score
        + 0.25 * precision_score
        + 0.30 * boundary_score
        + 0.10 * signal_score
        + 0.05 * foreground_score
    )
    return {
        "score": round(float(score), 4),
        "candidate_count": len(kept),
        "mask_area_px": mask_area,
        "foreground_fraction": round(foreground_fraction, 4),
        "support_coverage": round(support_coverage, 4),
        "soft_support_precision": round(soft_support_precision, 4),
        "boundary_strength": round(boundary_strength, 4),
        "mean_signal": round(mean_signal, 4),
    }


def select_best_segmentation(
    cellpose_kept: list[seg.Candidate],
    hybrid_kept: list[seg.Candidate],
    support: np.ndarray,
    signal: np.ndarray,
    grad_norm: np.ndarray,
) -> tuple[list[seg.Candidate], str, dict]:
    """Compare Cellpose with recovery output and conservatively select the better mask."""
    cellpose = segmentation_quality(cellpose_kept, support, signal, grad_norm)
    hybrid = segmentation_quality(hybrid_kept, support, signal, grad_norm)
    selected = cellpose_kept
    method = "cellpose"
    reason = "cellpose_retained"

    if not cellpose_kept and hybrid_kept:
        selected, method, reason = hybrid_kept, "hybrid_fallback", "cellpose_empty"
    elif hybrid_kept:
        rescue_incomplete_cellpose = (
            cellpose["support_coverage"] < 0.20
            and hybrid["support_coverage"] >= max(0.35, cellpose["support_coverage"] + 0.25)
            and hybrid["soft_support_precision"] >= 0.70
            and hybrid["boundary_strength"] >= 0.07
            and hybrid["foreground_fraction"] <= 0.70
            and hybrid["score"] >= cellpose["score"] + 0.08
        )
        foreground_limit = max(
            cellpose["foreground_fraction"] * 1.45,
            cellpose["foreground_fraction"] + 0.12,
        )
        conservative_improvement = (
            hybrid["score"] >= cellpose["score"] + 0.04
            and hybrid["boundary_strength"] >= cellpose["boundary_strength"] * 0.80
            and hybrid["soft_support_precision"] >= cellpose["soft_support_precision"] * 0.85
            and hybrid["foreground_fraction"] <= foreground_limit
        )
        if rescue_incomplete_cellpose:
            selected, method, reason = hybrid_kept, "hybrid_fallback", "recovered_incomplete_cellpose"
        elif conservative_improvement:
            selected, method, reason = hybrid_kept, "hybrid_fallback", "higher_quality_score"
        elif hybrid["score"] <= cellpose["score"]:
            reason = "cellpose_equal_or_higher"
        else:
            reason = "fallback_improvement_not_safe"

    comparison = {
        "selected_method": method,
        "selection_reason": reason,
        "cellpose": cellpose,
        "hybrid_fallback": hybrid,
    }
    return selected, method, comparison


def process_one(
    model: object,
    tif: Path,
    out_dir: Path,
    overwrite: bool,
    branch_progress=None,
    experiment: str = "density",
) -> dict:
    condition = parse_condition(tif, experiment)
    day = parse_day(tif)
    date_key = f"D{day:02d}"
    cfg = stage_and_diameters(experiment, condition, day)
    stem = f"{date_key}_{tif.stem}"
    mask_path = out_dir / f"{stem}_mask_16bit.png"
    rgb_path = out_dir / f"{stem}_instance_rgb.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    cellpose_mask_path = out_dir / f"{stem}_cellpose_mask_16bit.png"
    cellpose_rgb_path = out_dir / f"{stem}_cellpose_instance_rgb.png"
    cellpose_overlay_path = out_dir / f"{stem}_cellpose_overlay.png"
    signal_path = out_dir / f"{stem}_signal.png"
    support_path = out_dir / f"{stem}_support.png"
    stats_path = out_dir / f"{stem}_segmentation_stats.json"
    metric_path = out_dir / f"{stem}_metrics.json"

    required = [
        metric_path,
        stats_path,
        mask_path,
        signal_path,
        support_path,
        rgb_path,
        overlay_path,
        cellpose_mask_path,
        cellpose_rgb_path,
        cellpose_overlay_path,
    ]
    if not overwrite and all(p.exists() and p.stat().st_size > 0 for p in required):
        return json.loads(metric_path.read_text(encoding="utf-8"))

    rgb, gray = seg.load_rgb_gray(tif)
    signal, support, grad_norm = seg.compute_hybrid_signal(gray)

    cellpose_candidates: list[seg.Candidate] = []
    branch_summaries = []
    for branch_rank, diameter in enumerate(cfg["diameters"]):
        if branch_progress:
            branch_progress(branch_rank + 1, len(cfg["diameters"]), diameter, "cellpose")
        try:
            masks, *_ = model.eval(
                gray,
                diameter=float(diameter),
                channels=[0, 0],
                normalize=True,
                do_3D=False,
            )
            masks = masks.astype(np.uint16)
            branch_count = 0
            for label in range(1, int(masks.max()) + 1):
                cand = seg.build_candidate(
                    masks,
                    label,
                    diameter,
                    branch_rank,
                    signal,
                    support,
                    grad_norm,
                    cfg["stage"],
                )
                if cand is not None:
                    cellpose_candidates.append(cand)
                    branch_count += 1
            branch_summaries.append(
                {"diameter_px": diameter, "kept_candidates_before_merge": branch_count, "status": "ok"}
            )
        except Exception as exc:
            branch_summaries.append(
                {
                    "diameter_px": diameter,
                    "kept_candidates_before_merge": 0,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )

    cellpose_kept = seg.merge_candidates(cellpose_candidates)
    cellpose_label_mask, cellpose_color_mask = seg.build_outputs(rgb, cellpose_kept)
    cellpose_overlay = app81.base.render_overlay(rgb, cellpose_label_mask, cellpose_color_mask)
    write_png(cellpose_mask_path, cellpose_label_mask)
    write_png(cellpose_rgb_path, cv2.cvtColor(cellpose_color_mask, cv2.COLOR_RGB2BGR))
    write_png(cellpose_overlay_path, cv2.cvtColor(cellpose_overlay, cv2.COLOR_RGB2BGR))

    cellpose_qc = cellpose_quality(cellpose_kept, support)
    signal_candidates: list[seg.Candidate] = []
    if cellpose_qc["passed"]:
        kept = cellpose_kept
        selected_method = "cellpose"
        quality_comparison = {
            "selected_method": selected_method,
            "selection_reason": "cellpose_quality_gate_passed",
            "cellpose": segmentation_quality(cellpose_kept, support, signal, grad_norm),
            "hybrid_fallback": None,
        }
        branch_summaries.append(
            {
                "diameter_px": None,
                "kept_candidates_before_merge": 0,
                "status": "skipped: cellpose quality gate passed",
                "source": "signal_recovery",
            }
        )
    else:
        if branch_progress:
            branch_progress(1, 1, None, "fallback")
        signal_candidates = seg.recover_signal_candidates(signal, support, grad_norm, cfg["stage"])
        hybrid_kept = seg.merge_candidates([*cellpose_candidates, *signal_candidates])
        kept, selected_method, quality_comparison = select_best_segmentation(
            cellpose_kept, hybrid_kept, support, signal, grad_norm
        )
        branch_summaries.append(
            {
                "diameter_px": None,
                "kept_candidates_before_merge": len(signal_candidates),
                "status": "ok",
                "source": "signal_recovery",
            }
        )

    label_mask, color_mask = seg.build_outputs(rgb, kept)
    overlay = app81.base.render_overlay(rgb, label_mask, color_mask)

    write_png(mask_path, label_mask)
    write_png(rgb_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    write_png(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    write_png(signal_path, signal)
    write_png(support_path, support)

    shape = app81.aggregate_segmentation_metrics(label_mask, signal)
    objects = shape["object_rows"]
    object_areas = np.asarray([float(row["area_px"]) for row in objects], dtype=np.float64)
    largest_area = float(object_areas.max()) if object_areas.size else 0.0
    mean_area = float(object_areas.mean()) if object_areas.size else 0.0
    median_area = float(np.median(object_areas)) if object_areas.size else 0.0
    mean_equiv_diameter = float(np.mean(np.sqrt(4.0 * object_areas / math.pi))) if object_areas.size else 0.0
    total_equiv_diameter = math.sqrt(4.0 * float(shape["total_area_px"]) / math.pi) if shape["total_area_px"] else 0.0
    fused_mass_fraction = largest_area / float(shape["total_area_px"]) if shape["total_area_px"] else 0.0

    source_gray = full_metrics.load_source_gray(tif)
    metric = {
        "condition": condition,
        "experiment": experiment,
        "day": day,
        "date_key": date_key,
        "date_label": date_key,
        "relative_day": day,
        "image_name": tif.name,
        "source_tif": str(tif.resolve()),
        "stage": cfg["stage"],
        "diameters_px": cfg["diameters"],
        "segmentation_method": selected_method,
        "segmentation_selection_reason": quality_comparison["selection_reason"],
        "segmentation_quality_score": quality_comparison[selected_method]["score"],
        "cellpose_quality_score": quality_comparison["cellpose"]["score"],
        "fallback_quality_score": (
            quality_comparison["hybrid_fallback"]["score"]
            if quality_comparison["hybrid_fallback"] is not None
            else None
        ),
        "cellpose_quality_passed": cellpose_qc["passed"],
        "cellpose_count": int(cellpose_label_mask.max()),
        "count": shape["count"],
        "curvature": shape["curvature"],
        "roundness": shape["roundness"],
        "roundness_deviation_norm": shape["roundness_deviation_norm"],
        "roundness_deviation_px_total": shape["roundness_deviation_px_total"],
        "edge_intensity": shape["edge_intensity"],
        "total_area_px": shape["total_area_px"],
        "average_perimeter_px": shape["average_perimeter_px"],
        "largest_object_area_px": largest_area,
        "mean_object_area_px": mean_area,
        "median_object_area_px": median_area,
        "mean_equivalent_diameter_px": mean_equiv_diameter,
        "total_equivalent_diameter_px": total_equiv_diameter,
        "fused_mass_fraction": fused_mass_fraction,
        "mask_path": str(mask_path.resolve()),
        "instance_rgb_path": str(rgb_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "cellpose_mask_path": str(cellpose_mask_path.resolve()),
        "cellpose_instance_rgb_path": str(cellpose_rgb_path.resolve()),
        "cellpose_overlay_path": str(cellpose_overlay_path.resolve()),
        "signal_path": str(signal_path.resolve()),
        "support_path": str(support_path.resolve()),
        "stats_path": str(stats_path.resolve()),
    }
    metric.update(full_metrics.compute_signal_metrics(signal))
    metric.update(full_metrics.compute_edge_proxy_metrics(source_gray))
    metric.update(full_metrics.compute_internal_edge_metrics(label_mask, signal))
    metric.update(full_metrics.compute_center_weighted_metrics(label_mask, signal))
    metric.update(full_metrics.compute_darkness_metrics(label_mask, source_gray))
    if metric["count"] > 0 and metric["curvature"] > 0 and metric["edge_intensity"] > 0:
        metric["inverse_edge_count_curvature"] = 1.0 / (
            float(metric["edge_intensity"]) * float(metric["count"]) * float(metric["curvature"])
        )
    else:
        metric["inverse_edge_count_curvature"] = float("nan")

    object_path = out_dir / f"{stem}_object_metrics.csv"
    with object_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "area_px",
                "perimeter_px",
                "circularity",
                "roundness",
                "roundness_deviation_norm",
                "roundness_deviation_px",
            ],
        )
        writer.writeheader()
        writer.writerows(objects)
    metric["object_metrics_path"] = str(object_path.resolve())

    stats = {
        "source_tif": str(tif.resolve()),
        "condition": condition,
        "experiment": experiment,
        "day": day,
        "stage": cfg["stage"],
        "diameters_px": cfg["diameters"],
        "mask_count": int(label_mask.max()),
        "selected_method": selected_method,
        "cellpose_quality": cellpose_qc,
        "quality_comparison": quality_comparison,
        "branch_summaries": branch_summaries,
        "cellpose_candidates": [
            {
                "area": cand.area,
                "score": round(cand.score, 4),
                "diameter_px": cand.diameter,
                "support_ratio": round(cand.support_ratio, 4),
                "mean_signal": round(cand.mean_signal, 4),
                "edge_strength": round(cand.edge_strength, 4),
                "circularity": round(cand.circularity, 4),
                "source": cand.source,
            }
            for cand in sorted(cellpose_kept, key=lambda c: c.area, reverse=True)
        ],
        "merged_candidates": [
            {
                "area": cand.area,
                "score": round(cand.score, 4),
                "diameter_px": cand.diameter,
                "support_ratio": round(cand.support_ratio, 4),
                "mean_signal": round(cand.mean_signal, 4),
                "edge_strength": round(cand.edge_strength, 4),
                "circularity": round(cand.circularity, 4),
                "source": cand.source,
            }
            for cand in sorted(kept, key=lambda c: c.area, reverse=True)
        ],
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_path.write_text(json.dumps(metric, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return metric


def minmax(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [1.0 for _ in values]
    return [float((v - lo) / (hi - lo)) for v in values]


def add_comparison_scores(rows: list[dict], condition_order: list[str]) -> None:
    rows.sort(key=lambda row: (condition_order.index(row["condition"]), int(row["day"]), row["image_name"]))
    area_norm = minmax([float(row["total_area_px"]) for row in rows])
    signal_norm = minmax([float(row["sum_intensity"]) for row in rows])
    count_norm = minmax([float(row["count"]) for row in rows])
    for row, an, sn, cn in zip(rows, area_norm, signal_norm, count_norm):
        same_day = [candidate for candidate in rows if int(candidate["day"]) == int(row["day"])]
        baseline_condition = next(
            condition for condition in condition_order if any(candidate["condition"] == condition for candidate in same_day)
        )
        baseline = next(
            (
                float(candidate["total_area_px"])
                for candidate in same_day
                if candidate["condition"] == baseline_condition
            ),
            None,
        )
        row["growth_area_index_0_1"] = an
        row["signal_index_0_1"] = sn
        row["count_index_0_1"] = cn
        row["baseline_condition"] = baseline_condition
        row["relative_area_vs_baseline"] = float(row["total_area_px"]) / baseline if baseline else float("nan")
        row["relative_area_vs_low"] = row["relative_area_vs_baseline"]


def make_gallery(rows: list[dict], out_path: Path) -> None:
    row_images = []
    thumb_h = 245
    for row in rows:
        rgb, _ = seg.load_rgb_gray(Path(row["source_tif"]))
        overlay = cv2.cvtColor(full_metrics.imread_any(Path(row["overlay_path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        inst = cv2.cvtColor(full_metrics.imread_any(Path(row["instance_rgb_path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        def resize(img: np.ndarray) -> np.ndarray:
            scale = thumb_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), thumb_h), interpolation=cv2.INTER_AREA)

        panels = [
            resize(rgb),
            resize(overlay),
            resize(inst),
        ]
        spacer = np.full((thumb_h, 10, 3), 255, dtype=np.uint8)
        strip_parts = []
        for panel in panels:
            if strip_parts:
                strip_parts.append(spacer)
            strip_parts.append(panel)
        strip = np.concatenate(strip_parts, axis=1)
        title = (
            f"{row['condition']} D{row['day']:02d} | area={int(row['total_area_px'])} px | "
            f"n={int(row['count'])} | method={row['segmentation_method']} | "
            f"quality={float(row['segmentation_quality_score']):.3f}"
        )
        canvas = np.full((thumb_h + 34, strip.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[34:, :, :] = strip
        row_images.append(canvas)
    if not row_images:
        return
    width = max(img.shape[1] for img in row_images)
    height = sum(img.shape[0] for img in row_images) + 12 * (len(row_images) - 1)
    gallery = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 0
    for img in row_images:
        gallery[y : y + img.shape[0], : img.shape[1]] = img
        y += img.shape[0] + 12
    write_png(out_path, cv2.cvtColor(gallery, cv2.COLOR_RGB2BGR))


def make_largest_object_crops(rows: list[dict], crop_dir: Path) -> None:
    crop_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        labels = full_metrics.load_labels(Path(row["mask_path"]))
        label_ids = [int(v) for v in np.unique(labels) if int(v) > 0]
        if not label_ids:
            continue
        largest = max(label_ids, key=lambda lab: int((labels == lab).sum()))
        ys, xs = np.where(labels == largest)
        if ys.size == 0:
            continue
        pad = 80
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(labels.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(labels.shape[1], int(xs.max()) + pad + 1)
        rgb, _ = seg.load_rgb_gray(Path(row["source_tif"]))
        overlay = cv2.cvtColor(full_metrics.imread_any(Path(row["overlay_path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        crop = np.concatenate([rgb[y0:y1, x0:x1], overlay[y0:y1, x0:x1]], axis=1)
        out = crop_dir / f"{condition_slug(row['condition'])}_D{row['day']:02d}_largest_object_crop.png"
        write_png(out, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        row["largest_object_crop_path"] = str(out.resolve())


def selected_day_label(rows: list[dict]) -> str:
    days = sorted({int(row["day"]) for row in rows})
    if len(days) == 1:
        return f"Day {days[0]}"
    return "Selected Days " + ", ".join(f"D{day:02d}" for day in days)


def make_metric_figure(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    multiple_days = len({int(row["day"]) for row in rows}) > 1
    labels = [f"{row['condition']} D{int(row['day']):02d}" if multiple_days else row["condition"] for row in rows]
    metrics = [
        ("total_area_px", "Total area (px)"),
        ("count", "Object count"),
        ("mean_equivalent_diameter_px", "Mean equiv. diameter (px)"),
        ("fused_mass_fraction", "Largest mass fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    palette = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#14b8a6"]
    colors = [palette[index % len(palette)] for index in range(len(rows))]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = [float(row[key]) for row in rows]
        ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"{selected_day_label(rows)} Density Growth Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_results_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "experiment",
        "condition",
        "day",
        "stage",
        "segmentation_method",
        "segmentation_selection_reason",
        "segmentation_quality_score",
        "cellpose_quality_score",
        "fallback_quality_score",
        "cellpose_quality_passed",
        "cellpose_count",
        "count",
        "total_area_px",
        "baseline_condition",
        "relative_area_vs_baseline",
        "relative_area_vs_low",
        "growth_area_index_0_1",
        "mean_object_area_px",
        "median_object_area_px",
        "largest_object_area_px",
        "fused_mass_fraction",
        "mean_equivalent_diameter_px",
        "total_equivalent_diameter_px",
        "average_perimeter_px",
        "roundness",
        "roundness_deviation_norm",
        "curvature",
        "edge_intensity",
        "sum_intensity",
        "central_inside_over_peripheral",
        "center_weighted_edge_mean",
        "organoid_darkness_mean",
        "mask_path",
        "overlay_path",
        "signal_path",
        "support_path",
        "object_metrics_path",
        "stats_path",
        "source_tif",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    by_day = {}
    for day in sorted({int(row["day"]) for row in rows}):
        subset = [row for row in rows if int(row["day"]) == day]
        ranked_area = sorted(subset, key=lambda row: float(row["total_area_px"]), reverse=True)
        ranked_fusion = sorted(subset, key=lambda row: float(row["fused_mass_fraction"]), reverse=True)
        by_day[f"D{day:02d}"] = {
            "growth_ranking_by_total_area": [row["condition"] for row in ranked_area],
            "fusion_ranking_by_largest_mass_fraction": [row["condition"] for row in ranked_fusion],
        }
    ranking_text = "; ".join(
        f"{day}: {' > '.join(values['growth_ranking_by_total_area'])}" for day, values in by_day.items()
    )
    return {
        "comparisons_by_day": by_day,
        "interpretation": (
            f"Segmented-area ranking by matched day is {ranking_text}. "
            "Largest-mass fraction is reported as a fusion-context proxy; object count is not treated as pure growth."
        ),
    }


def write_report(rows: list[dict], summary: dict, out_path: Path) -> None:
    experiment = rows[0].get("experiment", "density") if rows else "density"
    experiment_name = {
        "density": "Density",
        "y27632": "Y-27632",
        "sodium_alginate": "Sodium Alginate",
    }.get(experiment, experiment)
    lines = [
        f"# {selected_day_label(rows)} {experiment_name} Morphology Analysis",
        "",
        "Inputs: selected bright-field 10x TIFFs only.",
        "Workflow: Cellpose-first segmentation with quality-scored fallback comparison and full metric backfill.",
        "",
        f"## {selected_day_label(rows)} Morphology Result",
        "",
        "| Condition | Method | Quality | Final count | Total area px | Area vs matched-day baseline | Mean equiv. diameter px | Largest mass fraction | Roundness |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['segmentation_method']} | {row['segmentation_quality_score']:.3f} | "
            f"{int(row['count'])} | {int(row['total_area_px'])} | "
            f"{row['relative_area_vs_low']:.3f} | {row['mean_equivalent_diameter_px']:.1f} | "
            f"{row['fused_mass_fraction']:.3f} | {row['roundness']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- {summary['interpretation']}",
        "- Higher object count is not interpreted alone as better growth because fusion can reduce apparent count.",
        "- Largest-mass fraction and overlays provide the main fusion QC evidence.",
        "",
        "## Output Evidence",
        "",
        "- `results.csv`: endpoint metric table.",
        "- `results.json`: structured metrics, paths, and interpretation.",
        "- `comparison_gallery.png`: source, selected best overlay, and selected best instance mask only.",
        "- `summary_metrics.png`: growth and morphology comparison bars.",
        "- `runs/<condition>/Dxx/*_mask_16bit.png`: 16-bit instance masks.",
        "- `runs/<condition>/Dxx/*_overlay.png`: segmentation overlays.",
        "- Pure Cellpose intermediates remain saved internally for audit but are not presented as final results.",
        "- `runs/<condition>/Dxx/*_signal.png`: hybrid signal images.",
        "- `runs/<condition>/Dxx/*_support.png`: support masks.",
        "- `crops/*_largest_object_crop.png`: source/overlay crops of the largest segmented object.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    inputs = [Path(p).resolve() for p in args.inputs]
    progress_path = Path(args.progress_file).resolve() if args.progress_file else None
    started = time.perf_counter()
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"Missing input files: {missing}")

    condition_order = EXPERIMENT_CONDITION_ORDERS[args.experiment]
    ordered_inputs = sorted(
        inputs,
        key=lambda p: (condition_order.index(parse_condition(p, args.experiment)), parse_day(p), p.name.lower()),
    )
    write_progress(progress_path, {"status": "loading_model", "phase": "loading_model", "total": len(inputs), "completed": 0, "current_file": None, "current_scale": None, "percent": 0.0, "elapsed_seconds": 0.0})
    use_gpu = bool(torch.cuda.is_available())
    model = seg.models.CellposeModel(gpu=use_gpu)
    rows = []
    for idx, tif in enumerate(ordered_inputs, start=1):
        condition = parse_condition(tif, args.experiment)
        per_out = runs_dir / condition_slug(condition) / f"D{parse_day(tif):02d}"
        per_out.mkdir(parents=True, exist_ok=True)
        def branch_progress(branch_index, branch_total, diameter, phase):
            within_image = (branch_index - 1) / max(branch_total, 1) if phase == "cellpose" else 0.9
            fraction = ((idx - 1) + within_image) / len(ordered_inputs)
            write_progress(progress_path, {"status": "running", "phase": phase, "total": len(ordered_inputs), "completed": idx - 1, "current_index": idx, "current_file": tif.name, "current_condition": condition, "scale_index": branch_index if phase == "cellpose" else None, "scale_total": branch_total if phase == "cellpose" else None, "current_scale": diameter, "percent": round(fraction * 100.0, 1), "elapsed_seconds": round(time.perf_counter() - started, 1)})
        row = process_one(
            model,
            tif,
            per_out,
            overwrite=args.overwrite,
            branch_progress=branch_progress,
            experiment=args.experiment,
        )
        rows.append(row)
        write_progress(progress_path, {"status": "running" if idx < len(ordered_inputs) else "finalizing", "total": len(ordered_inputs), "completed": idx, "current_index": idx, "current_file": tif.name, "current_condition": condition, "scale_index": None, "scale_total": None, "current_scale": None, "percent": round(idx * 100.0 / len(ordered_inputs), 1), "elapsed_seconds": round(time.perf_counter() - started, 1)})
        print(f"[{idx}/{len(inputs)}] {condition} D{row['day']:02d}: count={row['count']} area={row['total_area_px']}", flush=True)

    add_comparison_scores(rows, condition_order)
    make_largest_object_crops(rows, out_dir / "crops")
    gallery_path = out_dir / "comparison_gallery.png"
    summary_figure_path = out_dir / "summary_metrics.png"
    make_gallery(rows, gallery_path)
    make_metric_figure(rows, summary_figure_path)
    results_csv = out_dir / "results.csv"
    results_json = out_dir / "results.json"
    report_path = out_dir / "report.md"
    write_results_csv(rows, results_csv)
    summary = summarize(rows)
    results = {
        "analysis": f"{args.experiment}_selected_inputs",
        "workflow": f"{args.experiment}_multiscale_cellpose_quality_selection",
        "experiment": args.experiment,
        "inputs": [str(p) for p in inputs],
        "used_gpu": use_gpu,
        "instruction": args.instruction,
        "outputs": {
            "results_csv": str(results_csv.resolve()),
            "results_json": str(results_json.resolve()),
            "report_md": str(report_path.resolve()),
            "comparison_gallery": str(gallery_path.resolve()),
            "summary_metrics": str(summary_figure_path.resolve()),
            "crops_dir": str((out_dir / "crops").resolve()),
            "runs_dir": str(runs_dir.resolve()),
        },
        "rows": rows,
        "summary": summary,
    }
    results_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    write_report(rows, summary, report_path)
    if args.instruction.strip():
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Requested analysis\n\n{args.instruction.strip()}\n")
    write_progress(progress_path, {"status": "succeeded", "phase": "complete", "total": len(ordered_inputs), "completed": len(ordered_inputs), "current_file": None, "current_scale": None, "percent": 100.0, "elapsed_seconds": round(time.perf_counter() - started, 1), "results_json": str(results_json.resolve())})

    print(results_json)
    print(results_csv)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
