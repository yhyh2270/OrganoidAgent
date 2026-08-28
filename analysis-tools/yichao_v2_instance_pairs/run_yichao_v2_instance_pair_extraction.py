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

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_TOOL_DIR = REPO_ROOT / "analysis-tools" / "yichao_instance_pairs"
sys.path.insert(0, str(V1_TOOL_DIR))

from common import (  # noqa: E402
    build_comparison_panel,
    build_overlay,
    candidate_rows,
    compute_square_crop,
    ensure_parent,
    extract_padded_crop,
    load_gray_rgb,
    load_multiscale_module,
    remove_tree_if_exists,
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

IMAGE_RE = re.compile(
    r"^(?P<series>\d+)_(?P<object>.+?)_t(?P<t>\d+)_z(?P<z>\d+)_c(?P<c>\d+)\.jpg$",
    re.IGNORECASE,
)

NAMED_ACQUISITIONS = ("ALEXA488", "DAPI", "ALEXA647", "mCherry")

GENERIC_CHANNEL_MAPS: dict[str, dict[str, object]] = {
    "3_N39Rep_Globet_DF_D2": {
        "kind": "generic_inferred_7_channel",
        "confidence": "inferred_medium",
        "main_target_name": "ALEXA488_like_green",
        "main_brightfield_channel": 4,
        "main_fluorescence_channel": 3,
        "auxiliary_fluorescence_channels": {
            "generic_c0_fluorescence_like": 0,
            "generic_c1_fluorescence_like": 1,
            "generic_c5_fluorescence_like": 5,
        },
        "auxiliary_brightfield_channels": {
            "generic_c2_brightfield_like": 2,
            "generic_c6_brightfield_like": 6,
        },
        "note": "D2 has seven channels. c3 is the green radial signal; c2/c4/c6 are BF-like. c4 is used as the paired BF by adjacency.",
    },
    "4_N39Rep_Globet_DF_D3_1": {
        "kind": "generic_inferred_8_channel",
        "confidence": "inferred_high",
        "main_target_name": "ALEXA488_like_green",
        "main_brightfield_channel": 3,
        "main_fluorescence_channel": 2,
        "auxiliary_fluorescence_channels": {
            "generic_c0_fluorescence_like": 0,
            "generic_c4_fluorescence_like": 4,
            "generic_c6_fluorescence_like": 6,
        },
        "auxiliary_brightfield_channels": {
            "generic_c1_brightfield_like": 1,
            "generic_c5_brightfield_like": 5,
            "generic_c7_brightfield_like": 7,
        },
        "note": "Even channels are fluorescence-like and odd channels are BF-like. c2 is the green radial signal and c3 is its adjacent BF.",
    },
    "5_N39Rep_Globet_DF_D3_2": {
        "kind": "generic_inferred_8_channel",
        "confidence": "inferred_high",
        "main_target_name": "ALEXA488_like_green",
        "main_brightfield_channel": 3,
        "main_fluorescence_channel": 2,
        "auxiliary_fluorescence_channels": {
            "generic_c0_fluorescence_like": 0,
            "generic_c4_fluorescence_like": 4,
            "generic_c6_fluorescence_like": 6,
        },
        "auxiliary_brightfield_channels": {
            "generic_c1_brightfield_like": 1,
            "generic_c5_brightfield_like": 5,
            "generic_c7_brightfield_like": 7,
        },
        "note": "Even channels are fluorescence-like and odd channels are BF-like. c2 is the green radial signal and c3 is its adjacent BF.",
    },
}

IMAGE_LEVEL_REQUIRED_FILES = (
    "brightfield_input.png",
    "green_fluorescence_reference.png",
    "debug_signal.png",
    "support.png",
    "multiscale_mask_16bit.png",
    "multiscale_instance_rgb.png",
    "multiscale_overlay_on_brightfield.png",
    "multiscale_overlay_on_green_fluorescence.png",
    "comparison_panel.png",
    "image_record.json",
)

INSTANCE_LEVEL_REQUIRED_FILES = (
    "brightfield_crop.png",
    "green_fluorescence_crop.png",
    "mask_crop.png",
    "overlay_on_brightfield_crop.png",
    "overlay_on_green_fluorescence_crop.png",
    "instance_rgb_crop.png",
    "instance_record.json",
)


@dataclass(frozen=True)
class V2WorkItem:
    dataset_folder: str
    lif_stem: str
    day_hint: str
    object_name: str
    field_id: str
    series_index: int
    time_index: int
    z_index: int
    brightfield_path: Path
    green_fluorescence_path: Path
    auxiliary_fluorescence_paths: dict[str, Path]
    auxiliary_brightfield_paths: dict[str, Path]
    channel_map_kind: str
    channel_map_confidence: str
    main_target_name: str
    main_brightfield_channel: str
    main_fluorescence_channel: str
    channel_map_note: str
    stage: str
    diameters: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yichao v2 per-z organoid instance pairs with auxiliary fluorescence channels.")
    parser.add_argument("--v2-root", default=str(REPO_ROOT / "DATA-Yichao-v2"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "analysis-outputs/yichao_v2_instance_pairs"))
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--count-only", action="store_true")
    return parser.parse_args()


def natural_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def parse_day_hint(name: str) -> str:
    match = re.search(r"(?:^|_)D(?:ay_?)?(\d+)(?:_|$)", name, flags=re.IGNORECASE)
    if match:
        return f"D{match.group(1)}"
    return ""


def parse_plane_name(path: Path) -> dict[str, object]:
    match = IMAGE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported v2 image name: {path.name}")
    return {
        "series_index": int(match.group("series")),
        "object_name": match.group("object"),
        "time_index": int(match.group("t")),
        "z_index": int(match.group("z")),
        "channel_index": int(match.group("c")),
    }


def parse_named_group_name(name: str) -> tuple[str, str]:
    stem = name.removeprefix("N39Rep_").removesuffix("_BF")
    if "_" in stem:
        field_id, acquisition = stem.split("_", 1)
    else:
        field_id, acquisition = stem, "unknown"
    return field_id, acquisition.replace("ALEZA", "ALEXA")


def collect_planes(group_dir: Path) -> dict[tuple[int, int, int], Path]:
    planes: dict[tuple[int, int, int], Path] = {}
    for path in sorted(group_dir.glob("*.jpg"), key=lambda p: natural_key(p.name)):
        meta = parse_plane_name(path)
        planes[(int(meta["time_index"]), int(meta["z_index"]), int(meta["channel_index"]))] = path
    return planes


def find_grouped_root(dataset_dir: Path) -> Path | None:
    grouped = sorted(dataset_dir.glob("*_jpeg_all_by_position"), key=lambda p: natural_key(p.name))
    return grouped[0] if grouped else None


def is_named_grouped_root(grouped_root: Path) -> bool:
    names = [p.name.upper().replace("ALEZA", "ALEXA") for p in grouped_root.iterdir() if p.is_dir()]
    return any("ALEXA488" in name for name in names)


def discover_named_items(dataset_dir: Path, grouped_root: Path) -> list[V2WorkItem]:
    dataset_folder = dataset_dir.name
    lif_stem = next((p.stem for p in dataset_dir.glob("*.lif")), dataset_folder)
    day_hint = parse_day_hint(dataset_folder)
    grouped: dict[str, dict[str, tuple[Path, dict[tuple[int, int, int], Path]]]] = {}
    for group_dir in sorted((p for p in grouped_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        field_id, acquisition = parse_named_group_name(group_dir.name)
        if acquisition not in NAMED_ACQUISITIONS:
            continue
        grouped.setdefault(field_id, {})[acquisition] = (group_dir, collect_planes(group_dir))

    items: list[V2WorkItem] = []
    for field_id in sorted(grouped, key=natural_key):
        acquisitions = grouped[field_id]
        if "ALEXA488" not in acquisitions:
            continue
        alexa_dir, alexa_planes = acquisitions["ALEXA488"]
        keys = sorted(
            (t, z)
            for (t, z, c), path in alexa_planes.items()
            if c == 1 and (t, z, 0) in alexa_planes
        )
        for t, z in keys:
            brightfield_path = alexa_planes[(t, z, 1)]
            green_path = alexa_planes[(t, z, 0)]
            meta = parse_plane_name(green_path)
            aux_fluo: dict[str, Path] = {}
            aux_bf: dict[str, Path] = {}
            for acquisition in NAMED_ACQUISITIONS:
                if acquisition == "ALEXA488" or acquisition not in acquisitions:
                    continue
                _, planes = acquisitions[acquisition]
                if (t, z, 0) in planes:
                    aux_fluo[f"{acquisition}_c0"] = planes[(t, z, 0)]
                if (t, z, 1) in planes:
                    aux_bf[f"{acquisition}_c1"] = planes[(t, z, 1)]
            object_name = f"field_{field_id}_ALEXA488"
            items.append(
                V2WorkItem(
                    dataset_folder=dataset_folder,
                    lif_stem=lif_stem,
                    day_hint=day_hint,
                    object_name=object_name,
                    field_id=str(field_id),
                    series_index=int(meta["series_index"]),
                    time_index=t,
                    z_index=z,
                    brightfield_path=brightfield_path,
                    green_fluorescence_path=green_path,
                    auxiliary_fluorescence_paths=aux_fluo,
                    auxiliary_brightfield_paths=aux_bf,
                    channel_map_kind="named_acquisition",
                    channel_map_confidence="confirmed_by_name_and_preview",
                    main_target_name="ALEXA488_green",
                    main_brightfield_channel="ALEXA488_c1",
                    main_fluorescence_channel="ALEXA488_c0",
                    channel_map_note="Yichao stated only ALEXA488 has green fluorescence; previews confirm ALEXA488 c1 is BF-like and c0 is green.",
                    stage="fused_large",
                    diameters=(70, 130, 220),
                )
            )
    return items


def discover_generic_items(dataset_dir: Path, grouped_root: Path) -> list[V2WorkItem]:
    dataset_folder = dataset_dir.name
    if dataset_folder not in GENERIC_CHANNEL_MAPS:
        return []
    lif_stem = next((p.stem for p in dataset_dir.glob("*.lif")), dataset_folder)
    day_hint = parse_day_hint(dataset_folder)
    channel_map = GENERIC_CHANNEL_MAPS[dataset_folder]
    bf_channel = int(channel_map["main_brightfield_channel"])
    fluo_channel = int(channel_map["main_fluorescence_channel"])
    aux_fluo_channels = dict(channel_map["auxiliary_fluorescence_channels"])  # type: ignore[arg-type]
    aux_bf_channels = dict(channel_map["auxiliary_brightfield_channels"])  # type: ignore[arg-type]

    items: list[V2WorkItem] = []
    for group_dir in sorted((p for p in grouped_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        planes = collect_planes(group_dir)
        keys = sorted(
            (t, z)
            for (t, z, c), path in planes.items()
            if c == bf_channel and (t, z, fluo_channel) in planes
        )
        for t, z in keys:
            brightfield_path = planes[(t, z, bf_channel)]
            green_path = planes[(t, z, fluo_channel)]
            meta = parse_plane_name(green_path)
            aux_fluo = {
                str(name): planes[(t, z, int(channel))]
                for name, channel in aux_fluo_channels.items()
                if (t, z, int(channel)) in planes
            }
            aux_bf = {
                str(name): planes[(t, z, int(channel))]
                for name, channel in aux_bf_channels.items()
                if (t, z, int(channel)) in planes
            }
            items.append(
                V2WorkItem(
                    dataset_folder=dataset_folder,
                    lif_stem=lif_stem,
                    day_hint=day_hint,
                    object_name=group_dir.name,
                    field_id=group_dir.name,
                    series_index=int(meta["series_index"]),
                    time_index=t,
                    z_index=z,
                    brightfield_path=brightfield_path,
                    green_fluorescence_path=green_path,
                    auxiliary_fluorescence_paths=aux_fluo,
                    auxiliary_brightfield_paths=aux_bf,
                    channel_map_kind=str(channel_map["kind"]),
                    channel_map_confidence=str(channel_map["confidence"]),
                    main_target_name=str(channel_map["main_target_name"]),
                    main_brightfield_channel=f"c{bf_channel}",
                    main_fluorescence_channel=f"c{fluo_channel}",
                    channel_map_note=str(channel_map["note"]),
                    stage="fused_large",
                    diameters=(70, 130, 220),
                )
            )
    return items


def discover_work_items(v2_root: Path, dataset_names: set[str] | None = None, limit: int | None = None) -> list[V2WorkItem]:
    items: list[V2WorkItem] = []
    for dataset_dir in sorted((p for p in v2_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        if dataset_names and dataset_dir.name not in dataset_names:
            continue
        grouped_root = find_grouped_root(dataset_dir)
        if grouped_root is None:
            continue
        if is_named_grouped_root(grouped_root):
            dataset_items = discover_named_items(dataset_dir, grouped_root)
        else:
            dataset_items = discover_generic_items(dataset_dir, grouped_root)
        items.extend(dataset_items)
        if limit is not None and len(items) >= limit:
            return items[:limit]
    return items


def resolve_use_gpu(choice: str) -> bool:
    if choice == "true":
        return True
    if choice == "false":
        return False
    import torch

    return bool(torch.cuda.is_available())


def build_output_paths(output_root: Path, item: V2WorkItem) -> dict[str, Path]:
    stem = f"{item.series_index:02d}_{item.object_name}_t{item.time_index:03d}_z{item.z_index:03d}"
    image_dir = output_root / "images" / item.dataset_folder / item.object_name / stem
    instance_dir = output_root / "instances" / item.dataset_folder / item.object_name / stem
    return {
        "image_dir": image_dir,
        "instance_dir": instance_dir,
        "image_record": image_dir / "image_record.json",
        "brightfield_png": image_dir / "brightfield_input.png",
        "green_png": image_dir / "green_fluorescence_reference.png",
        "signal_png": image_dir / "debug_signal.png",
        "support_png": image_dir / "support.png",
        "mask_png": image_dir / "multiscale_mask_16bit.png",
        "instance_rgb_png": image_dir / "multiscale_instance_rgb.png",
        "overlay_brightfield_png": image_dir / "multiscale_overlay_on_brightfield.png",
        "overlay_green_png": image_dir / "multiscale_overlay_on_green_fluorescence.png",
        "comparison_panel_png": image_dir / "comparison_panel.png",
        "aux_fluo_dir": image_dir / "auxiliary_fluorescence",
        "aux_bf_dir": image_dir / "auxiliary_brightfield",
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


def build_label_and_instance_rgb(image_shape: tuple[int, int], kept: list[object]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    label_mask = np.zeros((height, width), dtype=np.uint16)
    instance_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label_value, candidate in enumerate(sorted(kept, key=lambda item: item.area, reverse=True), start=1):
        label_mask[candidate.mask] = label_value
        instance_rgb[candidate.mask] = COLOR_PALETTE[(label_value - 1) % len(COLOR_PALETTE)]
    return label_mask, instance_rgb


def write_aux_images(aux_paths: dict[str, Path], out_dir: Path) -> dict[str, str]:
    written: dict[str, str] = {}
    for name, src in sorted(aux_paths.items()):
        gray, _ = load_gray_rgb(src)
        dst = out_dir / f"{name}.png"
        save_png(dst, gray)
        written[name] = str(dst)
    return written


def write_failure_record(output_root: Path, item: V2WorkItem, exc: Exception) -> None:
    failure_path = output_root / "failures" / item.dataset_folder / item.object_name / f"t{item.time_index:03d}_z{item.z_index:03d}.json"
    write_json(
        failure_path,
        {
            "dataset_folder": item.dataset_folder,
            "object_name": item.object_name,
            "time_index": item.time_index,
            "z_index": item.z_index,
            "source_brightfield_path": str(item.brightfield_path),
            "source_green_fluorescence_path": str(item.green_fluorescence_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def failure_record_path(output_root: Path, item: V2WorkItem) -> Path:
    return output_root / "failures" / item.dataset_folder / item.object_name / f"t{item.time_index:03d}_z{item.z_index:03d}.json"


def build_instance_record(
    instance_dir: Path,
    image_record: dict[str, object],
    candidate: object,
    label_value: int,
    crop_meta: dict[str, int],
    aux_fluo_crop_paths: dict[str, str],
    aux_bf_crop_paths: dict[str, str],
) -> dict[str, object]:
    image_id = str(image_record["image_id"])
    instance_id = f"{image_id}::instance_{label_value:04d}"
    source_w = int(image_record["image_width_px"])
    source_h = int(image_record["image_height_px"])
    crop_x = int(crop_meta["crop_x"])
    crop_y = int(crop_meta["crop_y"])
    crop_w = int(crop_meta["crop_w"])
    crop_h = int(crop_meta["crop_h"])
    is_edge_padded = int(crop_x < 0 or crop_y < 0 or crop_x + crop_w > source_w or crop_y + crop_h > source_h)
    return {
        "instance_id": instance_id,
        "image_id": image_id,
        "dataset_version": "v2",
        "dataset_folder": image_record["dataset_folder"],
        "lif_stem": image_record["lif_stem"],
        "day_hint": image_record["day_hint"],
        "object_name": image_record["object_name"],
        "field_id": image_record["field_id"],
        "series_index": image_record["series_index"],
        "time_index": image_record["time_index"],
        "z_index": image_record["z_index"],
        "channel_map_confidence": image_record["channel_map_confidence"],
        "source_brightfield_path": image_record["source_brightfield_path"],
        "source_green_fluorescence_path": image_record["source_green_fluorescence_path"],
        "source_auxiliary_fluorescence_paths_json": image_record["source_auxiliary_fluorescence_paths_json"],
        "source_auxiliary_brightfield_paths_json": image_record["source_auxiliary_brightfield_paths_json"],
        "instance_label": label_value,
        "area_px": int(candidate.area),
        "diameter_px": int(candidate.diameter),
        "score": round(float(candidate.score), 6),
        "support_ratio": round(float(candidate.support_ratio), 6),
        "mean_signal": round(float(candidate.mean_signal), 6),
        "edge_strength": round(float(candidate.edge_strength), 6),
        "circularity": round(float(candidate.circularity), 6),
        "source": candidate.source,
        "source_image_width_px": source_w,
        "source_image_height_px": source_h,
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
        "green_fluorescence_crop_path": str(instance_dir / "green_fluorescence_crop.png"),
        "mask_crop_path": str(instance_dir / "mask_crop.png"),
        "overlay_on_brightfield_crop_path": str(instance_dir / "overlay_on_brightfield_crop.png"),
        "overlay_on_green_fluorescence_crop_path": str(instance_dir / "overlay_on_green_fluorescence_crop.png"),
        "instance_rgb_crop_path": str(instance_dir / "instance_rgb_crop.png"),
        "auxiliary_fluorescence_crop_paths_json": json.dumps(aux_fluo_crop_paths, ensure_ascii=False, sort_keys=True),
        "auxiliary_brightfield_crop_paths_json": json.dumps(aux_bf_crop_paths, ensure_ascii=False, sort_keys=True),
    }


def segment_one_image(output_root: Path, item: V2WorkItem, module: object, model: object, skip_existing: bool) -> tuple[str, int]:
    paths = build_output_paths(output_root, item)
    if image_outputs_complete(paths):
        if skip_existing:
            return "skipped", 0
        reset_partial_outputs(paths)
    elif paths["image_dir"].exists() or paths["instance_dir"].exists():
        reset_partial_outputs(paths)

    brightfield_gray, brightfield_rgb = load_gray_rgb(item.brightfield_path)
    green_gray, green_rgb = load_gray_rgb(item.green_fluorescence_path)
    signal, support, grad_norm = module.compute_hybrid_signal(brightfield_gray)

    candidates = []
    branch_summaries = []
    for branch_rank, diameter in enumerate(item.diameters):
        try:
            masks, *_ = model.eval(brightfield_gray, diameter=float(diameter), channels=[0, 0], normalize=True, do_3D=False)
            masks = masks.astype(np.uint16)
            branch_count = 0
            for label in range(1, int(masks.max()) + 1):
                candidate = module.build_candidate(masks, label, diameter, branch_rank, signal, support, grad_norm, item.stage)
                if candidate is not None:
                    candidates.append(candidate)
                    branch_count += 1
            branch_summaries.append({"diameter_px": diameter, "kept_candidates_before_merge": branch_count, "status": "ok"})
        except Exception as exc:
            branch_summaries.append(
                {"diameter_px": diameter, "kept_candidates_before_merge": 0, "status": f"failed: {type(exc).__name__}: {exc}"}
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
            {"diameter_px": None, "kept_candidates_before_merge": 0, "status": "skipped", "source": "signal_recovery_fallback"}
        )

    kept = normalize_candidate_masks(module.merge_candidates(candidates), brightfield_gray.shape)
    label_mask, instance_rgb = build_label_and_instance_rgb(brightfield_gray.shape, kept)
    overlay_brightfield = build_overlay(brightfield_rgb, instance_rgb, label_mask)
    overlay_green = build_overlay(green_rgb, instance_rgb, label_mask)
    comparison_panel = build_comparison_panel(brightfield_rgb, green_rgb, signal, overlay_brightfield, overlay_green, instance_rgb)

    save_png(paths["brightfield_png"], brightfield_gray)
    save_png(paths["green_png"], green_gray)
    save_png(paths["signal_png"], signal)
    save_png(paths["support_png"], support)
    save_cv2_image(paths["mask_png"], label_mask.astype(np.uint16))
    save_png(paths["instance_rgb_png"], instance_rgb)
    save_png(paths["overlay_brightfield_png"], overlay_brightfield)
    save_png(paths["overlay_green_png"], overlay_green)
    save_png(paths["comparison_panel_png"], comparison_panel)
    local_aux_fluo = write_aux_images(item.auxiliary_fluorescence_paths, paths["aux_fluo_dir"])
    local_aux_bf = write_aux_images(item.auxiliary_brightfield_paths, paths["aux_bf_dir"])

    image_id = f"{item.dataset_folder}/{item.object_name}/t{item.time_index:03d}_z{item.z_index:03d}"
    image_record = {
        "image_id": image_id,
        "dataset_version": "v2",
        "dataset_folder": item.dataset_folder,
        "lif_stem": item.lif_stem,
        "day_hint": item.day_hint,
        "object_name": item.object_name,
        "field_id": item.field_id,
        "series_index": int(item.series_index),
        "time_index": int(item.time_index),
        "z_index": int(item.z_index),
        "stage": item.stage,
        "diameters_px": list(item.diameters),
        "channel_map_kind": item.channel_map_kind,
        "channel_map_confidence": item.channel_map_confidence,
        "main_target_name": item.main_target_name,
        "main_brightfield_channel": item.main_brightfield_channel,
        "main_fluorescence_channel": item.main_fluorescence_channel,
        "channel_map_note": item.channel_map_note,
        "segmentation_policy": "brightfield_multiscale_cellpose_with_signal_recovery_fallback",
        "signal_recovery_used": bool(signal_recovery_used),
        "image_height_px": int(brightfield_gray.shape[0]),
        "image_width_px": int(brightfield_gray.shape[1]),
        "source_brightfield_path": str(item.brightfield_path),
        "source_green_fluorescence_path": str(item.green_fluorescence_path),
        "source_auxiliary_fluorescence_paths_json": json.dumps({k: str(v) for k, v in sorted(item.auxiliary_fluorescence_paths.items())}, ensure_ascii=False, sort_keys=True),
        "source_auxiliary_brightfield_paths_json": json.dumps({k: str(v) for k, v in sorted(item.auxiliary_brightfield_paths.items())}, ensure_ascii=False, sort_keys=True),
        "image_dir": str(paths["image_dir"]),
        "brightfield_input_path": str(paths["brightfield_png"]),
        "green_fluorescence_reference_path": str(paths["green_png"]),
        "local_auxiliary_fluorescence_paths_json": json.dumps(local_aux_fluo, ensure_ascii=False, sort_keys=True),
        "local_auxiliary_brightfield_paths_json": json.dumps(local_aux_bf, ensure_ascii=False, sort_keys=True),
        "signal_png": str(paths["signal_png"]),
        "support_png": str(paths["support_png"]),
        "mask_16bit_png": str(paths["mask_png"]),
        "instance_rgb_png": str(paths["instance_rgb_png"]),
        "overlay_on_brightfield_png": str(paths["overlay_brightfield_png"]),
        "overlay_on_green_fluorescence_png": str(paths["overlay_green_png"]),
        "comparison_panel_png": str(paths["comparison_panel_png"]),
        "mask_count": int(label_mask.max()),
        "branch_summaries": branch_summaries,
        "merged_candidates": candidate_rows(kept),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    aux_fluo_arrays = {name: load_gray_rgb(path)[0] for name, path in item.auxiliary_fluorescence_paths.items()}
    aux_bf_arrays = {name: load_gray_rgb(path)[0] for name, path in item.auxiliary_brightfield_paths.items()}
    for label_value, candidate in enumerate(kept, start=1):
        candidate_mask = label_mask == label_value
        if not candidate_mask.any():
            continue
        crop_meta = compute_square_crop(candidate_mask)
        instance_dir = paths["instance_dir"] / f"instance_{label_value:04d}"
        brightfield_crop = extract_padded_crop(brightfield_gray, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        green_crop = extract_padded_crop(green_gray, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        mask_crop = extract_padded_crop(candidate_mask.astype(np.uint8) * 255, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        overlay_brightfield_crop = extract_padded_crop(overlay_brightfield, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        overlay_green_crop = extract_padded_crop(overlay_green, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])
        instance_rgb_crop = extract_padded_crop(instance_rgb, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"])

        save_png(instance_dir / "brightfield_crop.png", brightfield_crop)
        save_png(instance_dir / "green_fluorescence_crop.png", green_crop)
        save_png(instance_dir / "mask_crop.png", mask_crop)
        save_png(instance_dir / "overlay_on_brightfield_crop.png", overlay_brightfield_crop)
        save_png(instance_dir / "overlay_on_green_fluorescence_crop.png", overlay_green_crop)
        save_png(instance_dir / "instance_rgb_crop.png", instance_rgb_crop)

        aux_fluo_crop_paths: dict[str, str] = {}
        for name, array in sorted(aux_fluo_arrays.items()):
            dst = instance_dir / "auxiliary_fluorescence_crops" / f"{name}_crop.png"
            save_png(dst, extract_padded_crop(array, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"]))
            aux_fluo_crop_paths[name] = str(dst)
        aux_bf_crop_paths: dict[str, str] = {}
        for name, array in sorted(aux_bf_arrays.items()):
            dst = instance_dir / "auxiliary_brightfield_crops" / f"{name}_crop.png"
            save_png(dst, extract_padded_crop(array, crop_meta["crop_x"], crop_meta["crop_y"], crop_meta["crop_w"], crop_meta["crop_h"]))
            aux_bf_crop_paths[name] = str(dst)

        instance_record = build_instance_record(instance_dir, image_record, candidate, label_value, crop_meta, aux_fluo_crop_paths, aux_bf_crop_paths)
        write_json(instance_dir / "instance_record.json", instance_record)

    write_json(paths["image_record"], image_record)
    failure_path = failure_record_path(output_root, item)
    if failure_path.exists():
        failure_path.unlink()
    return "processed", int(label_mask.max())


def main() -> int:
    args = parse_args()
    v2_root = Path(args.v2_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_names = set(args.datasets) if args.datasets else None
    work_items = discover_work_items(v2_root, dataset_names=dataset_names, limit=args.limit)
    if not work_items:
        raise RuntimeError("No v2 work items discovered. Run analysis-tools/yichao_v2/run_yichao_v2_lif_prepare.sh first.")
    if args.count_only:
        payload = {
            "target_images": len(work_items),
            "datasets": {
                dataset: sum(1 for item in work_items if item.dataset_folder == dataset)
                for dataset in sorted({item.dataset_folder for item in work_items})
            },
            "channel_map_confidences": sorted({item.channel_map_confidence for item in work_items}),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    skip_existing = not args.force and bool(args.skip_existing)
    write_json(
        output_root / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "v2_root": str(v2_root),
            "output_root": str(output_root),
            "target_images": len(work_items),
            "datasets": sorted(dataset_names) if dataset_names else sorted({item.dataset_folder for item in work_items}),
            "skip_existing": skip_existing,
            "gpu_mode": args.gpu,
            "max_new_images": args.max_new_images,
            "policy": "v2_main_ALEXA488_green_plus_auxiliary_channels",
            "generic_channel_maps": GENERIC_CHANNEL_MAPS,
        },
    )

    module = load_multiscale_module(REPO_ROOT)
    model = module.models.CellposeModel(gpu=resolve_use_gpu(args.gpu))
    processed = skipped = failed = total_instances = 0
    processed_items = 0
    t0 = time.perf_counter()
    stop_reason: str | None = None
    progress_path = output_root / "run_progress.json"
    write_json(
        progress_path,
        {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "target_images": len(work_items),
            "processed_items": 0,
            "processed_new_images": 0,
            "skipped_existing_images": 0,
            "failed_images": 0,
            "instances_written_for_new_images": 0,
        },
    )

    for index, item in enumerate(work_items, start=1):
        processed_items = index
        try:
            status, instance_count = segment_one_image(output_root, item, module, model, skip_existing=skip_existing)
            if status == "processed":
                processed += 1
                total_instances += instance_count
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            reset_partial_outputs(build_output_paths(output_root, item))
            write_failure_record(output_root, item, exc)

        reached_chunk_limit = args.max_new_images is not None and processed >= args.max_new_images and index < len(work_items)
        if reached_chunk_limit:
            stop_reason = "max_new_images"
        if index % max(1, args.progress_every) == 0 or index == len(work_items) or reached_chunk_limit:
            elapsed = max(1e-6, time.perf_counter() - t0)
            payload = {
                "status": "partial" if reached_chunk_limit else "running",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "target_images": len(work_items),
                "processed_items": index,
                "processed_new_images": processed,
                "skipped_existing_images": skipped,
                "failed_images": failed,
                "instances_written_for_new_images": total_instances,
                "elapsed_seconds": round(elapsed, 3),
                "images_per_second": round(index / elapsed, 3),
            }
            if stop_reason:
                payload["stop_reason"] = stop_reason
            write_json(progress_path, payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        if reached_chunk_limit:
            break

    completed = processed_items >= len(work_items)
    summary = {
        "status": "finished" if completed else "partial",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "target_images": len(work_items),
        "processed_items": processed_items,
        "processed_new_images": processed,
        "skipped_existing_images": skipped,
        "failed_images": failed,
        "instances_written_for_new_images": total_instances,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    if stop_reason:
        summary["stop_reason"] = stop_reason
    write_json(output_root / "run_summary.json", summary)
    write_json(progress_path, summary)
    print(output_root)
    print(output_root / "run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
