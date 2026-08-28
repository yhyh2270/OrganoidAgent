#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (
    build_comparison_panel,
    build_overlay,
    candidate_rows,
    compute_square_crop,
    discover_work_items,
    ensure_parent,
    extract_padded_crop,
    hardlink_or_copy,
    load_gray_rgb,
    load_multiscale_module,
    remove_tree_if_exists,
    save_png,
    save_cv2_image,
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
    "brightfield_input.jpg",
    "fluorescence_reference.jpg",
    "debug_signal.png",
    "support.png",
    "multiscale_mask_16bit.png",
    "multiscale_instance_rgb.png",
    "multiscale_overlay_on_brightfield.png",
    "multiscale_overlay_on_fluorescence.png",
    "comparison_panel.png",
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "analysis-outputs/yichao_instance_pairs"
    parser = argparse.ArgumentParser(
        description="Segment all Yichao brightfield frames, save image-level intermediates, and export paired instance crops."
    )
    parser.add_argument("--output-root", default=str(default_output))
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-new-images", type=int, default=None)
    return parser.parse_args()


def resolve_use_gpu(choice: str) -> bool:
    if choice == "true":
        return True
    if choice == "false":
        return False
    import torch

    return bool(torch.cuda.is_available())


def build_output_paths(output_root: Path, dataset: str, object_name: str, image_stem: str) -> dict[str, Path]:
    image_dir = output_root / "images" / dataset / object_name / image_stem
    instance_dir = output_root / "instances" / dataset / object_name / image_stem
    return {
        "image_dir": image_dir,
        "instance_dir": instance_dir,
        "image_record": image_dir / "image_record.json",
        "brightfield_link": image_dir / "brightfield_input.jpg",
        "fluorescence_link": image_dir / "fluorescence_reference.jpg",
        "signal_png": image_dir / "debug_signal.png",
        "support_png": image_dir / "support.png",
        "mask_png": image_dir / "multiscale_mask_16bit.png",
        "instance_rgb_png": image_dir / "multiscale_instance_rgb.png",
        "overlay_brightfield_png": image_dir / "multiscale_overlay_on_brightfield.png",
        "overlay_fluorescence_png": image_dir / "multiscale_overlay_on_fluorescence.png",
        "comparison_panel_png": image_dir / "comparison_panel.png",
    }


def save_image_level_outputs(
    paths: dict[str, Path],
    brightfield_src: Path,
    fluorescence_src: Path,
    signal: np.ndarray,
    support: np.ndarray,
    label_mask: np.ndarray,
    instance_rgb: np.ndarray,
    overlay_brightfield: np.ndarray,
    overlay_fluorescence: np.ndarray,
    comparison_panel: np.ndarray,
) -> None:
    hardlink_or_copy(brightfield_src, paths["brightfield_link"])
    hardlink_or_copy(fluorescence_src, paths["fluorescence_link"])
    save_png(paths["signal_png"], signal)
    save_png(paths["support_png"], support)
    save_cv2_image(paths["mask_png"], label_mask.astype(np.uint16))
    save_png(paths["instance_rgb_png"], instance_rgb)
    save_png(paths["overlay_brightfield_png"], overlay_brightfield)
    save_png(paths["overlay_fluorescence_png"], overlay_fluorescence)
    save_png(paths["comparison_panel_png"], comparison_panel)


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


def build_label_and_instance_rgb(image_shape: tuple[int, int], kept: list[object]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    label_mask = np.zeros((height, width), dtype=np.uint16)
    instance_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label_value, candidate in enumerate(sorted(kept, key=lambda item: item.area, reverse=True), start=1):
        label_mask[candidate.mask] = label_value
        color = COLOR_PALETTE[(label_value - 1) % len(COLOR_PALETTE)]
        instance_rgb[candidate.mask] = color
    return label_mask, instance_rgb


def write_failure_record(output_root: Path, work_item: object, exc: Exception) -> None:
    image_stem = work_item.brightfield_path.stem
    failure_path = failure_record_path(output_root, work_item)
    payload = {
        "dataset": work_item.dataset,
        "object_name": work_item.object_name,
        "source_brightfield_path": str(work_item.brightfield_path),
        "source_fluorescence_path": str(work_item.fluorescence_path),
        "stage": work_item.stage,
        "diameters_px": list(work_item.diameters),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(failure_path, payload)


def failure_record_path(output_root: Path, work_item: object) -> Path:
    image_stem = work_item.brightfield_path.stem
    return output_root / "failures" / work_item.dataset / work_item.object_name / f"{image_stem}.json"


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
    return {
        "instance_id": instance_id,
        "image_id": image_id,
        "dataset": image_record["dataset"],
        "object_name": image_record["object_name"],
        "series_index": image_record["series_index"],
        "time_index": image_record["time_index"],
        "z_index": image_record["z_index"],
        "source_brightfield_path": image_record["source_brightfield_path"],
        "source_fluorescence_path": image_record["source_fluorescence_path"],
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


def segment_one_image(output_root: Path, work_item: object, module: object, model: object, skip_existing: bool) -> tuple[str, int]:
    image_stem = work_item.brightfield_path.stem
    image_id = f"{work_item.dataset}/{work_item.object_name}/{image_stem}"
    paths = build_output_paths(output_root, work_item.dataset, work_item.object_name, image_stem)
    if image_outputs_complete(paths):
        if skip_existing:
            return "skipped", 0
        reset_partial_outputs(paths)
    elif paths["image_dir"].exists() or paths["instance_dir"].exists():
        reset_partial_outputs(paths)

    brightfield_gray, brightfield_rgb = load_gray_rgb(work_item.brightfield_path)
    fluorescence_gray, fluorescence_rgb = load_gray_rgb(work_item.fluorescence_path)
    signal, support, grad_norm = module.compute_hybrid_signal(brightfield_gray)

    candidates = []
    branch_summaries = []
    for branch_rank, diameter in enumerate(work_item.diameters):
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
                    work_item.stage,
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
        signal_candidates = module.recover_signal_candidates(signal, support, grad_norm, work_item.stage)
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

    save_image_level_outputs(
        paths,
        brightfield_src=work_item.brightfield_path,
        fluorescence_src=work_item.fluorescence_path,
        signal=signal,
        support=support,
        label_mask=label_mask,
        instance_rgb=instance_rgb,
        overlay_brightfield=overlay_brightfield,
        overlay_fluorescence=overlay_fluorescence,
        comparison_panel=comparison_panel,
    )

    image_record = {
        "image_id": image_id,
        "dataset": work_item.dataset,
        "object_name": work_item.object_name,
        "series_index": int(work_item.series_index),
        "time_index": int(work_item.time_index),
        "z_index": int(work_item.z_index),
        "stage": work_item.stage,
        "diameters_px": list(work_item.diameters),
        "selected_channel": "c1_brightfield",
        "paired_channel": "c0_fluorescence",
        "segmentation_policy": "multiscale_cellpose_with_signal_recovery_fallback",
        "signal_recovery_used": bool(signal_recovery_used),
        "image_height_px": int(brightfield_gray.shape[0]),
        "image_width_px": int(brightfield_gray.shape[1]),
        "source_brightfield_path": str(work_item.brightfield_path),
        "source_fluorescence_path": str(work_item.fluorescence_path),
        "image_dir": str(paths["image_dir"]),
        "brightfield_input_path": str(paths["brightfield_link"]),
        "fluorescence_reference_path": str(paths["fluorescence_link"]),
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
        instance_rgb_crop = extract_padded_crop(
            instance_rgb,
            crop_meta["crop_x"],
            crop_meta["crop_y"],
            crop_meta["crop_w"],
            crop_meta["crop_h"],
        )

        save_png(instance_dir / "brightfield_crop.png", brightfield_crop)
        save_png(instance_dir / "fluorescence_crop.png", fluorescence_crop)
        save_png(instance_dir / "mask_crop.png", mask_crop)
        save_png(instance_dir / "overlay_on_brightfield_crop.png", overlay_brightfield_crop)
        save_png(instance_dir / "overlay_on_fluorescence_crop.png", overlay_fluorescence_crop)
        save_png(instance_dir / "instance_rgb_crop.png", instance_rgb_crop)

        instance_record = build_instance_record(output_root, instance_dir, image_record, candidate, label_value, crop_meta)
        write_json(instance_dir / "instance_record.json", instance_record)

    write_json(paths["image_record"], image_record)
    failure_path = failure_record_path(output_root, work_item)
    if failure_path.exists():
        failure_path.unlink()

    return "processed", int(label_mask.max())


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.force:
        skip_existing = False
    else:
        skip_existing = bool(args.skip_existing)

    dataset_names = set(args.datasets) if args.datasets else None
    work_items = discover_work_items(repo_root, dataset_names=dataset_names, limit=args.limit)
    if not work_items:
        raise RuntimeError("No Yichao brightfield images matched the requested filters.")

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "datasets": sorted(dataset_names) if dataset_names else sorted({item.dataset for item in work_items}),
        "image_count_target": len(work_items),
        "skip_existing": skip_existing,
        "gpu_mode": args.gpu,
        "max_new_images": args.max_new_images,
        "policy": "multiscale_cellpose_with_signal_recovery_fallback",
        "brightfield_channel": "c1",
        "fluorescence_channel": "c0",
    }
    write_json(output_root / "run_config.json", run_config)
    progress_path = output_root / "run_progress.json"

    module = load_multiscale_module(repo_root)
    model = module.models.CellposeModel(gpu=resolve_use_gpu(args.gpu))

    processed = 0
    skipped = 0
    total_instances = 0
    failed = 0
    t0 = time.perf_counter()
    processed_items = 0
    stop_reason: str | None = None

    write_json(
        progress_path,
        {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "output_root": str(output_root),
            "target_images": len(work_items),
            "processed_items": 0,
            "processed_new_images": 0,
            "skipped_existing_images": 0,
            "instances_written_for_new_images": 0,
            "failed_images": 0,
            "elapsed_seconds": 0.0,
        },
    )

    for index, work_item in enumerate(work_items, start=1):
        processed_items = index
        try:
            status, instance_count = segment_one_image(output_root, work_item, module, model, skip_existing=skip_existing)
            if status == "processed":
                processed += 1
                total_instances += instance_count
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            image_stem = work_item.brightfield_path.stem
            reset_partial_outputs(build_output_paths(output_root, work_item.dataset, work_item.object_name, image_stem))
            write_failure_record(output_root, work_item, exc)
        reached_chunk_limit = (
            args.max_new_images is not None
            and processed >= args.max_new_images
            and index < len(work_items)
        )
        if reached_chunk_limit:
            stop_reason = "max_new_images"
        if index % max(1, args.progress_every) == 0 or index == len(work_items) or reached_chunk_limit:
            elapsed = max(1e-6, time.perf_counter() - t0)
            rate = index / elapsed
            progress_payload = {
                "status": "partial" if reached_chunk_limit else "running",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "output_root": str(output_root),
                "target_images": len(work_items),
                "processed_items": index,
                "processed_new_images": processed,
                "skipped_existing_images": skipped,
                "instances_written_for_new_images": total_instances,
                "failed_images": failed,
                "elapsed_seconds": round(elapsed, 3),
                "images_per_second": round(rate, 3),
            }
            if stop_reason is not None:
                progress_payload["stop_reason"] = stop_reason
            write_json(progress_path, progress_payload)
            print(
                json.dumps(
                    {
                        "processed_items": index,
                        "total_items": len(work_items),
                        "processed_new": processed,
                        "skipped_existing": skipped,
                        "failed": failed,
                        "instances_written": total_instances,
                        "images_per_second": round(rate, 3),
                    }
                ),
                flush=True,
            )
        if reached_chunk_limit:
            break

    completed_all_items = processed_items >= len(work_items)
    summary_status = "finished" if completed_all_items else "partial"

    summary = {
        "status": summary_status,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "target_images": len(work_items),
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
    write_json(
        progress_path,
        {
            **summary,
        },
    )
    print(output_root)
    print(output_root / "run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
