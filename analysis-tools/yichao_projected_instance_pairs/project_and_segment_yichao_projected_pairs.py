#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
PER_Z_TOOL_DIR = REPO_ROOT / "analysis-tools" / "yichao_instance_pairs"
sys.path.insert(0, str(PER_Z_TOOL_DIR))

from common import (  # noqa: E402
    build_comparison_panel,
    build_overlay,
    candidate_rows,
    compute_square_crop,
    discover_work_items,
    ensure_parent,
    extract_padded_crop,
    load_multiscale_module,
    remove_tree_if_exists,
    resolve_segmentation_config,
    save_cv2_image,
    save_png,
    write_json,
)


COLOR_PALETTE = np.array(
    [
        [244, 67, 54],
        [33, 150, 243],
        [76, 175, 80],
        [255, 193, 7],
        [156, 39, 176],
        [255, 87, 34],
        [63, 81, 181],
        [0, 150, 136],
        [205, 220, 57],
        [121, 85, 72],
        [233, 30, 99],
        [3, 169, 244],
        [139, 195, 74],
        [255, 152, 0],
        [103, 58, 183],
        [0, 188, 212],
    ],
    dtype=np.uint8,
)

IMAGE_LEVEL_REQUIRED_FILES = (
    "brightfield_projected.png",
    "fluorescence_projected.png",
    "debug_signal.png",
    "support.png",
    "multiscale_mask_16bit.png",
    "multiscale_instance_rgb.png",
    "multiscale_overlay_on_brightfield.png",
    "multiscale_overlay_on_fluorescence.png",
    "comparison_panel.png",
    "projection_record.json",
    "image_record.json",
)

INSTANCE_LEVEL_REQUIRED_FILES = (
    "brightfield_crop.png",
    "fluorescence_crop.png",
    "mask_crop.png",
    "overlay_on_brightfield_crop.png",
    "overlay_on_fluorescence_crop.png",
    "instance_rgb_crop.png",
    "instance_record.json",
)


@dataclass(frozen=True)
class ProjectedWorkItem:
    dataset: str
    object_name: str
    series_index: int
    time_index: int
    z_indices: tuple[int, ...]
    brightfield_paths: tuple[Path, ...]
    fluorescence_paths: tuple[Path, ...]
    stage: str
    diameters: tuple[int, int, int]
    position_time_count: int
    has_time_series: bool
    experiment_label: str
    experiment_design: str
    replicate_label: str
    sample_label: str
    day_label: str
    day_index: int | None
    position_label: str
    position_index: int | None


def parse_args() -> argparse.Namespace:
    default_output = REPO_ROOT / "analysis-outputs" / "yichao_projected_instance_pairs"
    parser = argparse.ArgumentParser(
        description="Project Yichao z-stacks by dataset/object/time, segment projected brightfield, and save projected instance pairs."
    )
    parser.add_argument("--output-root", default=str(default_output))
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--brightfield-projection", choices=("min", "max", "mean", "median"), default="min")
    parser.add_argument("--fluorescence-projection", choices=("min", "max", "mean", "median"), default="max")
    parser.add_argument("--time-aware-only", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    return parser.parse_args()


def parse_design_metadata(object_name: str) -> dict[str, object]:
    normalized = object_name.replace(" ", "_")
    position_match = re.search(r"Position0*(\d+)", normalized, flags=re.IGNORECASE)
    day_match = re.search(r"(?:Day_?|D)(\d+)", normalized, flags=re.IGNORECASE)
    experiment_match = re.search(r"(Experiment_?\d+)", normalized, flags=re.IGNORECASE)
    replicate_match = re.search(r"(TriRep|TriReP|Rep)", normalized, flags=re.IGNORECASE)
    sample_match = re.search(r"(N\d+|P\d+N|PDO\d+|Jurkat|Goblet)", normalized, flags=re.IGNORECASE)
    design = re.sub(r"_?Position0*\d+", "", normalized, flags=re.IGNORECASE)
    return {
        "experiment_label": experiment_match.group(1) if experiment_match else "",
        "experiment_design": design,
        "replicate_label": replicate_match.group(1) if replicate_match else "",
        "sample_label": sample_match.group(1) if sample_match else "",
        "day_label": day_match.group(0) if day_match else "",
        "day_index": int(day_match.group(1)) if day_match else None,
        "position_label": position_match.group(0) if position_match else "",
        "position_index": int(position_match.group(1)) if position_match else None,
    }


def projection_stem(item: ProjectedWorkItem) -> str:
    return f"{item.series_index:02d}_{item.object_name}_t{item.time_index:03d}_zproj"


def discover_projected_work_items(
    repo_root: Path,
    dataset_names: set[str] | None = None,
    limit: int | None = None,
    time_aware_only: bool = False,
) -> list[ProjectedWorkItem]:
    raw_items = discover_work_items(repo_root, dataset_names=dataset_names)
    grouped: dict[tuple[str, str, int], list[Any]] = {}
    by_position: dict[tuple[str, str], set[int]] = {}
    for item in raw_items:
        grouped.setdefault((item.dataset, item.object_name, item.time_index), []).append(item)
        by_position.setdefault((item.dataset, item.object_name), set()).add(int(item.time_index))

    projected: list[ProjectedWorkItem] = []
    for (dataset, object_name, time_index), group in sorted(grouped.items(), key=lambda entry: (entry[0][0], entry[0][1], entry[0][2])):
        time_count = len(by_position[(dataset, object_name)])
        if time_aware_only and time_count <= 1:
            continue
        group_sorted = sorted(group, key=lambda item: int(item.z_index))
        z_indices = tuple(int(item.z_index) for item in group_sorted)
        series_indices = {int(item.series_index) for item in group_sorted}
        if len(series_indices) != 1:
            raise RuntimeError(f"Mixed series indices in projected group: {dataset}/{object_name}/t{time_index}")
        stage, diameters = resolve_segmentation_config(dataset, object_name)
        meta = parse_design_metadata(object_name)
        projected.append(
            ProjectedWorkItem(
                dataset=dataset,
                object_name=object_name,
                series_index=int(group_sorted[0].series_index),
                time_index=int(time_index),
                z_indices=z_indices,
                brightfield_paths=tuple(item.brightfield_path for item in group_sorted),
                fluorescence_paths=tuple(item.fluorescence_path for item in group_sorted),
                stage=stage,
                diameters=diameters,
                position_time_count=time_count,
                has_time_series=time_count > 1,
                experiment_label=str(meta["experiment_label"]),
                experiment_design=str(meta["experiment_design"]),
                replicate_label=str(meta["replicate_label"]),
                sample_label=str(meta["sample_label"]),
                day_label=str(meta["day_label"]),
                day_index=meta["day_index"] if isinstance(meta["day_index"], int) else None,
                position_label=str(meta["position_label"]),
                position_index=meta["position_index"] if isinstance(meta["position_index"], int) else None,
            )
        )
        if limit is not None and len(projected) >= limit:
            return projected
    return projected


def resolve_use_gpu(choice: str) -> bool:
    if choice == "true":
        return True
    if choice == "false":
        return False
    import torch

    return bool(torch.cuda.is_available())


def load_stack(paths: tuple[Path, ...]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            array = np.array(image.convert("L"))
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"Stack shape mismatch: expected {shape}, got {array.shape} for {path}")
        arrays.append(array)
    return np.stack(arrays, axis=0)


def project_stack(stack: np.ndarray, mode: str) -> np.ndarray:
    if mode == "min":
        projected = stack.min(axis=0)
    elif mode == "max":
        projected = stack.max(axis=0)
    elif mode == "mean":
        projected = stack.mean(axis=0)
    elif mode == "median":
        projected = np.median(stack, axis=0)
    else:
        raise ValueError(f"Unsupported projection mode: {mode}")
    return np.clip(projected, 0, 255).astype(np.uint8)


def build_output_paths(output_root: Path, item: ProjectedWorkItem) -> dict[str, Path]:
    stem = projection_stem(item)
    image_dir = output_root / "images" / item.dataset / item.object_name / stem
    instance_dir = output_root / "instances" / item.dataset / item.object_name / stem
    projection_dir = output_root / "projections" / item.dataset / item.object_name / stem
    return {
        "image_dir": image_dir,
        "instance_dir": instance_dir,
        "projection_dir": projection_dir,
        "image_record": image_dir / "image_record.json",
        "projection_record": image_dir / "projection_record.json",
        "brightfield_projected": image_dir / "brightfield_projected.png",
        "fluorescence_projected": image_dir / "fluorescence_projected.png",
        "projection_brightfield": projection_dir / "brightfield_projected.png",
        "projection_fluorescence": projection_dir / "fluorescence_projected.png",
        "projection_record_copy": projection_dir / "projection_record.json",
        "signal_png": image_dir / "debug_signal.png",
        "support_png": image_dir / "support.png",
        "mask_png": image_dir / "multiscale_mask_16bit.png",
        "instance_rgb_png": image_dir / "multiscale_instance_rgb.png",
        "overlay_brightfield_png": image_dir / "multiscale_overlay_on_brightfield.png",
        "overlay_fluorescence_png": image_dir / "multiscale_overlay_on_fluorescence.png",
        "comparison_panel_png": image_dir / "comparison_panel.png",
    }


def image_outputs_complete(paths: dict[str, Path]) -> bool:
    for required_name in IMAGE_LEVEL_REQUIRED_FILES:
        if not (paths["image_dir"] / required_name).exists():
            return False
    try:
        image_record = json.loads(paths["image_record"].read_text(encoding="utf-8"))
    except Exception:
        return False

    expected_instances = int(image_record.get("mask_count", 0))
    instance_dirs = sorted(paths["instance_dir"].glob("instance_*"))
    if len(instance_dirs) != expected_instances:
        return False
    for instance_dir in instance_dirs:
        for required_name in INSTANCE_LEVEL_REQUIRED_FILES:
            if not (instance_dir / required_name).exists():
                return False
    return True


def reset_partial_outputs(paths: dict[str, Path]) -> None:
    remove_tree_if_exists(paths["image_dir"])
    remove_tree_if_exists(paths["instance_dir"])
    remove_tree_if_exists(paths["projection_dir"])


def build_label_and_instance_rgb(image_shape: tuple[int, int], kept: list[object]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    label_mask = np.zeros((height, width), dtype=np.uint16)
    instance_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label_value, candidate in enumerate(sorted(kept, key=lambda item: item.area, reverse=True), start=1):
        label_mask[candidate.mask] = label_value
        color = COLOR_PALETTE[(label_value - 1) % len(COLOR_PALETTE)]
        instance_rgb[candidate.mask] = color
    return label_mask, instance_rgb


def normalize_candidate_masks(candidates: list[object], image_shape: tuple[int, int]) -> list[object]:
    height, width = image_shape
    normalized: list[object] = []
    for candidate in candidates:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.ndim != 2 or mask.shape != (height, width):
            continue
        if not mask.any():
            continue
        candidate.mask = mask
        normalized.append(candidate)
    return normalized


def projection_record_payload(
    item: ProjectedWorkItem,
    paths: dict[str, Path],
    brightfield_mode: str,
    fluorescence_mode: str,
) -> dict[str, object]:
    return {
        "projection_id": f"{item.dataset}/{item.object_name}/{projection_stem(item)}",
        "dataset": item.dataset,
        "object_name": item.object_name,
        "series_index": item.series_index,
        "time_index": item.time_index,
        "z_indices": list(item.z_indices),
        "z_count": len(item.z_indices),
        "brightfield_projection_mode": brightfield_mode,
        "fluorescence_projection_mode": fluorescence_mode,
        "brightfield_channel": "c1",
        "fluorescence_channel": "c0",
        "source_brightfield_paths": [str(path) for path in item.brightfield_paths],
        "source_fluorescence_paths": [str(path) for path in item.fluorescence_paths],
        "brightfield_projected_path": str(paths["brightfield_projected"]),
        "fluorescence_projected_path": str(paths["fluorescence_projected"]),
        "projection_dir": str(paths["projection_dir"]),
        "experiment_label": item.experiment_label,
        "experiment_design": item.experiment_design,
        "replicate_label": item.replicate_label,
        "sample_label": item.sample_label,
        "day_label": item.day_label,
        "day_index": item.day_index,
        "position_label": item.position_label,
        "position_index": item.position_index,
        "position_time_count": item.position_time_count,
        "has_time_series": int(item.has_time_series),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_projection_outputs(
    paths: dict[str, Path],
    brightfield: np.ndarray,
    fluorescence: np.ndarray,
    projection_record: dict[str, object],
) -> None:
    save_png(paths["brightfield_projected"], brightfield)
    save_png(paths["fluorescence_projected"], fluorescence)
    save_png(paths["projection_brightfield"], brightfield)
    save_png(paths["projection_fluorescence"], fluorescence)
    write_json(paths["projection_record"], projection_record)
    write_json(paths["projection_record_copy"], projection_record)


def write_failure_record(output_root: Path, item: ProjectedWorkItem, exc: Exception) -> None:
    failure_path = output_root / "failures" / item.dataset / item.object_name / f"{projection_stem(item)}.json"
    payload = {
        "dataset": item.dataset,
        "object_name": item.object_name,
        "series_index": item.series_index,
        "time_index": item.time_index,
        "z_indices": list(item.z_indices),
        "stage": item.stage,
        "diameters_px": list(item.diameters),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(failure_path, payload)


def failure_record_path(output_root: Path, item: ProjectedWorkItem) -> Path:
    return output_root / "failures" / item.dataset / item.object_name / f"{projection_stem(item)}.json"


def build_instance_record(
    output_root: Path,
    instance_dir: Path,
    image_record: dict[str, object],
    candidate: object,
    label_value: int,
    crop_meta: dict[str, int],
) -> dict[str, object]:
    image_id = str(image_record["image_id"])
    instance_id = f"{image_id}::instance_{label_value:04d}"
    source_image_width_px = int(image_record["image_width_px"])
    source_image_height_px = int(image_record["image_height_px"])
    crop_x = int(crop_meta["crop_x"])
    crop_y = int(crop_meta["crop_y"])
    crop_w = int(crop_meta["crop_w"])
    crop_h = int(crop_meta["crop_h"])
    is_edge_padded = int(
        crop_x < 0
        or crop_y < 0
        or crop_x + crop_w > source_image_width_px
        or crop_y + crop_h > source_image_height_px
    )
    inherited_keys = (
        "dataset",
        "object_name",
        "series_index",
        "time_index",
        "stage",
        "experiment_label",
        "experiment_design",
        "replicate_label",
        "sample_label",
        "day_label",
        "day_index",
        "position_label",
        "position_index",
        "position_time_count",
        "has_time_series",
        "z_indices",
        "z_count",
        "brightfield_projection_mode",
        "fluorescence_projection_mode",
    )
    record: dict[str, object] = {key: image_record.get(key, "") for key in inherited_keys}
    record.update(
        {
            "instance_id": instance_id,
            "image_id": image_id,
            "projection_id": image_record["projection_id"],
            "instance_label": label_value,
            "area_px": int(candidate.area),
            "diameter_px": int(candidate.diameter),
            "score": round(float(candidate.score), 6),
            "support_ratio": round(float(candidate.support_ratio), 6),
            "mean_signal": round(float(candidate.mean_signal), 6),
            "edge_strength": round(float(candidate.edge_strength), 6),
            "circularity": round(float(candidate.circularity), 6),
            "source": candidate.source,
            "source_image_width_px": source_image_width_px,
            "source_image_height_px": source_image_height_px,
            "bbox_x": int(crop_meta["bbox_x"]),
            "bbox_y": int(crop_meta["bbox_y"]),
            "bbox_w": int(crop_meta["bbox_w"]),
            "bbox_h": int(crop_meta["bbox_h"]),
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "square_crop_size_px": crop_w,
            "crop_area_px": crop_w * crop_h,
            "padding_px": int(crop_meta["padding_px"]),
            "is_edge_padded": is_edge_padded,
            "instance_dir": str(instance_dir),
            "brightfield_crop_path": str(instance_dir / "brightfield_crop.png"),
            "fluorescence_crop_path": str(instance_dir / "fluorescence_crop.png"),
            "mask_crop_path": str(instance_dir / "mask_crop.png"),
            "overlay_on_brightfield_crop_path": str(instance_dir / "overlay_on_brightfield_crop.png"),
            "overlay_on_fluorescence_crop_path": str(instance_dir / "overlay_on_fluorescence_crop.png"),
            "instance_rgb_crop_path": str(instance_dir / "instance_rgb_crop.png"),
        }
    )
    return record


def segment_projected_image(
    output_root: Path,
    item: ProjectedWorkItem,
    module: object,
    model: object,
    skip_existing: bool,
    brightfield_mode: str,
    fluorescence_mode: str,
) -> tuple[str, int]:
    paths = build_output_paths(output_root, item)
    if image_outputs_complete(paths):
        if skip_existing:
            return "skipped", 0
        reset_partial_outputs(paths)
    elif paths["image_dir"].exists() or paths["instance_dir"].exists() or paths["projection_dir"].exists():
        reset_partial_outputs(paths)

    brightfield_stack = load_stack(item.brightfield_paths)
    fluorescence_stack = load_stack(item.fluorescence_paths)
    brightfield_gray = project_stack(brightfield_stack, brightfield_mode)
    fluorescence_gray = project_stack(fluorescence_stack, fluorescence_mode)
    brightfield_rgb = cv2.cvtColor(brightfield_gray, cv2.COLOR_GRAY2RGB)
    fluorescence_rgb = cv2.cvtColor(fluorescence_gray, cv2.COLOR_GRAY2RGB)

    projection_record = projection_record_payload(item, paths, brightfield_mode, fluorescence_mode)
    save_projection_outputs(paths, brightfield_gray, fluorescence_gray, projection_record)

    signal, support, grad_norm = module.compute_hybrid_signal(brightfield_gray)
    candidates = []
    branch_summaries = []
    for branch_rank, diameter in enumerate(item.diameters):
        try:
            masks, *_ = model.eval(brightfield_gray, diameter=float(diameter), channels=[0, 0], normalize=True, do_3D=False)
            masks = masks.astype(np.uint16)
            branch_count = 0
            for label in range(1, int(masks.max()) + 1):
                candidate = module.build_candidate(
                    masks,
                    label,
                    diameter,
                    branch_rank,
                    signal,
                    support,
                    grad_norm,
                    item.stage,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    branch_count += 1
            branch_summaries.append({"diameter_px": diameter, "kept_candidates_before_merge": branch_count, "status": "ok"})
        except Exception as exc:
            branch_summaries.append(
                {
                    "diameter_px": diameter,
                    "kept_candidates_before_merge": 0,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )

    signal_recovery_used = False
    if not candidates:
        signal_candidates = module.recover_signal_candidates(signal, support, grad_norm, item.stage)
        if signal_candidates:
            candidates.extend(signal_candidates)
            signal_recovery_used = True
        branch_summaries.append(
            {
                "diameter_px": None,
                "kept_candidates_before_merge": len(signal_candidates),
                "status": "ok",
                "source": "signal_recovery_fallback",
            }
        )
    else:
        branch_summaries.append(
            {
                "diameter_px": None,
                "kept_candidates_before_merge": 0,
                "status": "skipped",
                "source": "signal_recovery_fallback",
            }
        )

    kept = normalize_candidate_masks(module.merge_candidates(candidates), brightfield_gray.shape)
    label_mask, instance_rgb = build_label_and_instance_rgb(brightfield_gray.shape, kept)
    overlay_brightfield = build_overlay(brightfield_rgb, instance_rgb, label_mask)
    overlay_fluorescence = build_overlay(fluorescence_rgb, instance_rgb, label_mask)
    comparison_panel = build_comparison_panel(
        brightfield_rgb,
        fluorescence_rgb,
        signal,
        overlay_brightfield,
        overlay_fluorescence,
        instance_rgb,
    )

    save_png(paths["signal_png"], signal)
    save_png(paths["support_png"], support)
    save_cv2_image(paths["mask_png"], label_mask.astype(np.uint16))
    save_png(paths["instance_rgb_png"], instance_rgb)
    save_png(paths["overlay_brightfield_png"], overlay_brightfield)
    save_png(paths["overlay_fluorescence_png"], overlay_fluorescence)
    save_png(paths["comparison_panel_png"], comparison_panel)

    image_id = f"{item.dataset}/{item.object_name}/{projection_stem(item)}"
    image_record = {
        **projection_record,
        "image_id": image_id,
        "stage": item.stage,
        "diameters_px": list(item.diameters),
        "selected_channel": "c1_brightfield_projected",
        "paired_channel": "c0_fluorescence_projected",
        "segmentation_policy": "z_projected_multiscale_cellpose_with_signal_recovery_fallback",
        "signal_recovery_used": bool(signal_recovery_used),
        "image_height_px": int(brightfield_gray.shape[0]),
        "image_width_px": int(brightfield_gray.shape[1]),
        "image_dir": str(paths["image_dir"]),
        "brightfield_projected_path": str(paths["brightfield_projected"]),
        "fluorescence_projected_path": str(paths["fluorescence_projected"]),
        "signal_png": str(paths["signal_png"]),
        "support_png": str(paths["support_png"]),
        "mask_16bit_png": str(paths["mask_png"]),
        "instance_rgb_png": str(paths["instance_rgb_png"]),
        "overlay_on_brightfield_png": str(paths["overlay_brightfield_png"]),
        "overlay_on_fluorescence_png": str(paths["overlay_fluorescence_png"]),
        "comparison_panel_png": str(paths["comparison_panel_png"]),
        "mask_count": int(label_mask.max()),
        "branch_summaries": branch_summaries,
        "merged_candidates": candidate_rows(kept),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    for label_value, candidate in enumerate(kept, start=1):
        candidate_mask = label_mask == label_value
        if not candidate_mask.any():
            continue
        crop_meta = compute_square_crop(candidate_mask)
        instance_dir = paths["instance_dir"] / f"instance_{label_value:04d}"
        brightfield_crop = extract_padded_crop(brightfield_gray, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        fluorescence_crop = extract_padded_crop(fluorescence_gray, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        mask_crop = extract_padded_crop((candidate_mask.astype(np.uint8) * 255), crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        overlay_brightfield_crop = extract_padded_crop(
            overlay_brightfield,
            crop_meta["crop_x"],
            crop_meta["crop_y"],
            crop_meta["crop_w"],
            crop_meta["crop_h"],
        )
        overlay_fluorescence_crop = extract_padded_crop(
            overlay_fluorescence,
            crop_meta["crop_x"],
            crop_meta["crop_y"],
            crop_meta["crop_w"],
            crop_meta["crop_h"],
        )
        instance_rgb_crop = extract_padded_crop(instance_rgb, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        save_png(instance_dir / "brightfield_crop.png", brightfield_crop)
        save_png(instance_dir / "fluorescence_crop.png", fluorescence_crop)
        save_png(instance_dir / "mask_crop.png", mask_crop)
        save_png(instance_dir / "overlay_on_brightfield_crop.png", overlay_brightfield_crop)
        save_png(instance_dir / "overlay_on_fluorescence_crop.png", overlay_fluorescence_crop)
        save_png(instance_dir / "instance_rgb_crop.png", instance_rgb_crop)
        instance_record = build_instance_record(output_root, instance_dir, image_record, candidate, label_value, crop_meta)
        write_json(instance_dir / "instance_record.json", instance_record)

    write_json(paths["image_record"], image_record)
    failure_path = failure_record_path(output_root, item)
    if failure_path.exists():
        failure_path.unlink()
    return "processed", int(label_mask.max())


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_names = set(args.datasets) if args.datasets else None
    work_items = discover_projected_work_items(
        REPO_ROOT,
        dataset_names=dataset_names,
        limit=args.limit,
        time_aware_only=bool(args.time_aware_only),
    )
    if args.count_only:
        print(len(work_items))
        return 0
    if not work_items:
        raise RuntimeError("No projected Yichao work items matched the requested filters.")

    skip_existing = not bool(args.force) and bool(args.skip_existing)
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "datasets": sorted(dataset_names) if dataset_names else sorted({item.dataset for item in work_items}),
        "projected_image_count_target": len(work_items),
        "skip_existing": skip_existing,
        "gpu_mode": args.gpu,
        "max_new_images": args.max_new_images,
        "brightfield_projection": args.brightfield_projection,
        "fluorescence_projection": args.fluorescence_projection,
        "policy": "z_projected_multiscale_cellpose_with_signal_recovery_fallback",
        "brightfield_channel": "c1",
        "fluorescence_channel": "c0",
        "time_aware_only": bool(args.time_aware_only),
    }
    write_json(output_root / "run_config.json", run_config)
    progress_path = output_root / "run_progress.json"

    module = load_multiscale_module(REPO_ROOT)
    model = module.models.CellposeModel(gpu=resolve_use_gpu(args.gpu))

    processed = 0
    skipped = 0
    total_instances = 0
    failed = 0
    processed_items = 0
    stop_reason: str | None = None
    t0 = time.perf_counter()

    write_json(
        progress_path,
        {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "output_root": str(output_root),
            "target_projected_images": len(work_items),
            "processed_items": 0,
            "processed_new_images": 0,
            "skipped_existing_images": 0,
            "instances_written_for_new_images": 0,
            "failed_images": 0,
            "elapsed_seconds": 0.0,
        },
    )

    for index, item in enumerate(work_items, start=1):
        processed_items = index
        try:
            status, instance_count = segment_projected_image(
                output_root,
                item,
                module,
                model,
                skip_existing=skip_existing,
                brightfield_mode=args.brightfield_projection,
                fluorescence_mode=args.fluorescence_projection,
            )
            if status == "processed":
                processed += 1
                total_instances += instance_count
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            reset_partial_outputs(build_output_paths(output_root, item))
            write_failure_record(output_root, item, exc)

        reached_chunk_limit = (
            args.max_new_images is not None
            and processed >= args.max_new_images
            and index < len(work_items)
        )
        if reached_chunk_limit:
            stop_reason = "max_new_images"

        if index % max(1, args.progress_every) == 0 or index == len(work_items) or reached_chunk_limit:
            elapsed = max(1e-6, time.perf_counter() - t0)
            payload = {
                "status": "partial" if reached_chunk_limit else "running",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "output_root": str(output_root),
                "target_projected_images": len(work_items),
                "processed_items": index,
                "processed_new_images": processed,
                "skipped_existing_images": skipped,
                "instances_written_for_new_images": total_instances,
                "failed_images": failed,
                "elapsed_seconds": round(elapsed, 3),
                "images_per_second": round(index / elapsed, 3),
            }
            if stop_reason is not None:
                payload["stop_reason"] = stop_reason
            write_json(progress_path, payload)
            print(
                json.dumps(
                    {
                        "processed_items": index,
                        "total_items": len(work_items),
                        "processed_new": processed,
                        "skipped_existing": skipped,
                        "failed": failed,
                        "instances_written": total_instances,
                        "images_per_second": round(index / elapsed, 3),
                    }
                ),
                flush=True,
            )
        if reached_chunk_limit:
            break

    completed_all_items = processed_items >= len(work_items)
    summary = {
        "status": "finished" if completed_all_items else "partial",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "target_projected_images": len(work_items),
        "processed_items": processed_items,
        "processed_new_images": processed,
        "skipped_existing_images": skipped,
        "instances_written_for_new_images": total_instances,
        "failed_images": failed,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    if stop_reason is not None:
        summary["stop_reason"] = stop_reason
    write_json(output_root / "run_summary.json", summary)
    write_json(progress_path, summary)
    print(output_root)
    print(output_root / "run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
