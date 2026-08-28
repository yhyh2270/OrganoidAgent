#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DatasetSourceSpec:
    name: str
    source_rel: str
    default_stage: str
    default_diameters: tuple[int, int, int]
    series_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class WorkItem:
    dataset: str
    object_name: str
    brightfield_path: Path
    fluorescence_path: Path
    series_index: int
    time_index: int
    z_index: int
    stage: str
    diameters: tuple[int, int, int]


DATASET_SOURCES = (
    DatasetSourceSpec(
        name="Data-Yichao-1",
        source_rel="Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all_by_object",
        default_stage="differentiated_irregular",
        default_diameters=(140, 240, 380),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-2",
        source_rel="Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object",
        default_stage="cystic_early",
        default_diameters=(110, 220, 360),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-3",
        source_rel="Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-4",
        source_rel="Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-5",
        source_rel="Data-Yichao-v1/Data-Yichao-5/N39_TriRep_DF_3_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-6",
        source_rel="Data-Yichao-v1/Data-Yichao-6/N39_TriRep_4_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-7",
        source_rel="Data-Yichao-v1/Data-Yichao-7/N39_TriRep_5_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-8",
        source_rel="Data-Yichao-v1/Data-Yichao-8/N39_TriRep_DF_6_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-9",
        source_rel="Data-Yichao-v1/Data-Yichao-9/PDO28 and Jurkat_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
        series_indices=(3, 4, 5),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-10",
        source_rel="Data-Yichao-v1/Data-Yichao-10/N39_TriRep_DF_7_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
    DatasetSourceSpec(
        name="Data-Yichao-11",
        source_rel="Data-Yichao-v1/Data-Yichao-11/N39_TriRep_DF_8_jpeg_all_by_position",
        default_stage="fused_large",
        default_diameters=(70, 130, 220),
    ),
)


IMAGE_NAME_RE = re.compile(
    r"^(?P<series>\d+)_(?P<object>.+?)_t(?P<t>\d+)_z(?P<z>\d+)_c(?P<c>\d+)\.(?P<ext>jpg|jpeg|png|tif|tiff)$",
    re.IGNORECASE,
)


def load_multiscale_module(repo_root: Path) -> Any:
    module_path = repo_root / "analysis-tools/app80_first_replicate_multiscale_cellpose/run_multiscale_dateaware_cellpose.py"
    module_name = "organoid_multiscale_dateaware_yichao"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load multiscale pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_image_name(path: Path) -> dict[str, Any]:
    match = IMAGE_NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported Yichao image name: {path.name}")
    return {
        "series_index": int(match.group("series")),
        "object_name": match.group("object"),
        "time_index": int(match.group("t")),
        "z_index": int(match.group("z")),
        "channel_index": int(match.group("c")),
        "extension": match.group("ext"),
    }


def paired_channel_path(path: Path, target_channel_index: int) -> Path:
    paired_name = re.sub(r"_c\d(\.[^.]+)$", rf"_c{target_channel_index}\1", path.name)
    paired_path = path.with_name(paired_name)
    if paired_path == path or not paired_path.exists():
        raise FileNotFoundError(f"Could not resolve paired channel c{target_channel_index} for {path}")
    return paired_path


def resolve_segmentation_config(dataset_name: str, object_name: str) -> tuple[str, tuple[int, int, int]]:
    if dataset_name == "Data-Yichao-1":
        return "differentiated_irregular", (140, 240, 380)
    if dataset_name == "Data-Yichao-2":
        if "MUC2" in object_name:
            return "differentiated_irregular", (140, 240, 380)
        return "cystic_early", (110, 220, 360)
    if dataset_name in {"Data-Yichao-3", "Data-Yichao-4", "Data-Yichao-5", "Data-Yichao-6", "Data-Yichao-7", "Data-Yichao-8", "Data-Yichao-9", "Data-Yichao-10", "Data-Yichao-11"}:
        return "fused_large", (70, 130, 220)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def discover_work_items(
    repo_root: Path,
    dataset_names: set[str] | None = None,
    limit: int | None = None,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    for spec in DATASET_SOURCES:
        if dataset_names and spec.name not in dataset_names:
            continue
        source_root = repo_root / spec.source_rel
        if not source_root.exists():
            if dataset_names and spec.name in dataset_names:
                raise FileNotFoundError(f"Missing source directory: {source_root}")
            continue
        for brightfield_path in sorted(source_root.rglob("*_c1.jpg")):
            if brightfield_path.name.startswith("._"):
                continue
            meta = parse_image_name(brightfield_path)
            if spec.series_indices is not None and int(meta["series_index"]) not in spec.series_indices:
                continue
            fluorescence_path = paired_channel_path(brightfield_path, 0)
            stage, diameters = resolve_segmentation_config(spec.name, meta["object_name"])
            items.append(
                WorkItem(
                    dataset=spec.name,
                    object_name=meta["object_name"],
                    brightfield_path=brightfield_path,
                    fluorescence_path=fluorescence_path,
                    series_index=int(meta["series_index"]),
                    time_index=int(meta["time_index"]),
                    z_index=int(meta["z_index"]),
                    stage=stage,
                    diameters=diameters,
                )
            )
            if limit is not None and len(items) >= limit:
                return items
    return items


def load_gray_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        gray = np.array(image.convert("L"))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray, rgb


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def save_png(path: Path, image: np.ndarray) -> None:
    ensure_parent(path)
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(image).save(tmp_path)
    os.replace(tmp_path, path)


def save_cv2_image(path: Path, image: np.ndarray) -> None:
    ensure_parent(path)
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    cv2.imwrite(str(tmp_path), image)
    os.replace(tmp_path, path)


def remove_tree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def build_overlay(base_rgb: np.ndarray, color_mask: np.ndarray, label_mask: np.ndarray) -> np.ndarray:
    overlay = cv2.addWeighted(base_rgb, 0.72, color_mask, 0.55, 0)
    edges = cv2.Canny((label_mask > 0).astype(np.uint8) * 255, 50, 150)
    overlay[edges > 0] = (255, 0, 0)
    return overlay


def build_comparison_panel(
    brightfield_rgb: np.ndarray,
    fluorescence_rgb: np.ndarray,
    signal: np.ndarray,
    overlay_on_brightfield: np.ndarray,
    overlay_on_fluorescence: np.ndarray,
    instance_rgb: np.ndarray,
) -> np.ndarray:
    panels: list[np.ndarray] = []
    items = [
        ("Brightfield", brightfield_rgb),
        ("Fluorescence", fluorescence_rgb),
        ("Debug Signal", cv2.cvtColor(signal, cv2.COLOR_GRAY2RGB)),
        ("Overlay On Brightfield", overlay_on_brightfield),
        ("Overlay On Fluorescence", overlay_on_fluorescence),
        ("Instance RGB", instance_rgb),
    ]
    for title, image in items:
        panel = image.copy()
        cv2.putText(
            panel,
            title,
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        panels.append(panel)
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(panels[3:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def candidate_rows(candidates: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.area, reverse=True):
        rows.append(
            {
                "area": int(candidate.area),
                "score": round(float(candidate.score), 4),
                "diameter_px": int(candidate.diameter),
                "support_ratio": round(float(candidate.support_ratio), 4),
                "mean_signal": round(float(candidate.mean_signal), 4),
                "edge_strength": round(float(candidate.edge_strength), 4),
                "circularity": round(float(candidate.circularity), 4),
                "source": candidate.source,
            }
        )
    return rows


def compute_square_crop(mask: np.ndarray, padding_px: int | None = None) -> dict[str, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Cannot crop empty mask")

    bbox_x0 = int(xs.min())
    bbox_y0 = int(ys.min())
    bbox_x1 = int(xs.max()) + 1
    bbox_y1 = int(ys.max()) + 1
    bbox_w = int(bbox_x1 - bbox_x0)
    bbox_h = int(bbox_y1 - bbox_y0)

    side = int(max(bbox_w, bbox_h))
    pad = int(padding_px if padding_px is not None else max(12, round(side * 0.08)))
    crop_side = int(side + 2 * pad)

    cx = 0.5 * (bbox_x0 + bbox_x1)
    cy = 0.5 * (bbox_y0 + bbox_y1)
    crop_x0 = int(round(cx - crop_side / 2.0))
    crop_y0 = int(round(cy - crop_side / 2.0))
    crop_x1 = crop_x0 + crop_side
    crop_y1 = crop_y0 + crop_side

    return {
        "bbox_x": bbox_x0,
        "bbox_y": bbox_y0,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "crop_x": crop_x0,
        "crop_y": crop_y0,
        "crop_w": crop_side,
        "crop_h": crop_side,
        "padding_px": pad,
    }


def extract_padded_crop(image: np.ndarray, crop_x: int, crop_y: int, crop_w: int, crop_h: int, fill_value: int = 0) -> np.ndarray:
    if image.ndim == 2:
        out = np.full((crop_h, crop_w), fill_value, dtype=image.dtype)
    else:
        out = np.full((crop_h, crop_w, image.shape[2]), fill_value, dtype=image.dtype)

    src_h, src_w = image.shape[:2]
    src_x0 = max(0, crop_x)
    src_y0 = max(0, crop_y)
    src_x1 = min(src_w, crop_x + crop_w)
    src_y1 = min(src_h, crop_y + crop_h)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - crop_x
    dst_y0 = src_y0 - crop_y
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out
