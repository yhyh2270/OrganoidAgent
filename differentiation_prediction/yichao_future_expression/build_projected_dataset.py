#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROJECTED_DB,
    coerce_float,
    coerce_int,
    resize_grayscale,
    safe_log1p_positive,
    split_from_group,
    write_csv,
    write_json,
)


FEATURE_COLUMNS = [
    "area_px",
    "diameter_px",
    "circularity",
    "support_ratio",
    "edge_strength",
    "mean_signal",
    "bbox_w",
    "bbox_h",
    "square_crop_size_px",
    "crop_area_px",
    "time_index",
    "day_index",
]


@dataclass
class InstanceRecord:
    row: dict[str, Any]
    mask_centroid_x: float
    mask_centroid_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize projected Yichao complete instance data for B2F and future-expression models.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_PROJECTED_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--future-exclude-datasets", default="Data-Yichao-9")
    parser.add_argument("--min-track-length", type=int, default=4)
    parser.add_argument("--max-prefix-frames", type=int, default=5)
    parser.add_argument("--future-guard-frames", type=int, default=1)
    parser.add_argument("--min-future-frames", type=int, default=2)
    parser.add_argument("--allow-positive-prefix", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def fetch_projected_rows(db_path: Path, limit: int | None) -> list[dict[str, Any]]:
    query = """
        select
            pi.*,
            im.has_time_series as image_has_time_series,
            im.brightfield_projected_path,
            im.fluorescence_projected_path
        from projected_instances pi
        join projected_images im on pi.image_id = im.image_id
        where pi.is_edge_padded = '0'
        order by pi.dataset, pi.object_name, cast(pi.time_index as integer), cast(pi.instance_label as integer), pi.instance_id
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute(query)]
    finally:
        conn.close()
    if limit is not None:
        return rows[:limit]
    return rows


def load_gray_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def fluorescence_features(fluorescence_path: Path, mask_path: Path) -> dict[str, float]:
    fl = load_gray_array(fluorescence_path)
    mask = load_gray_array(mask_path) > 0
    if not mask.any():
        mask = np.ones_like(fl, dtype=bool)
    inside = fl[mask]
    outside = fl[~mask]
    if outside.size == 0:
        outside = fl.reshape(-1)
    bg_median = float(np.median(outside))
    bg_mad = float(np.median(np.abs(outside - bg_median))) + 1e-6
    corrected_inside = inside - bg_median
    local_threshold = bg_median + 3.0 * bg_mad
    return {
        "fl_bg_median": bg_median,
        "fl_bg_mad": bg_mad,
        "fl_inside_mean": float(np.mean(inside)),
        "fl_inside_p90": float(np.percentile(inside, 90)),
        "fl_inside_p99": float(np.percentile(inside, 99)),
        "fl_corrected_mean": float(np.mean(corrected_inside)),
        "fl_corrected_p90": float(np.percentile(corrected_inside, 90)),
        "fl_corrected_p99": float(np.percentile(corrected_inside, 99)),
        "fl_corrected_total": float(np.clip(corrected_inside, 0, None).sum()),
        "fl_local_threshold": local_threshold,
        "fl_local_positive_fraction": float((inside > local_threshold).mean()),
    }


def mask_centroid(mask_path: Path, fallback_x: float, fallback_y: float) -> tuple[float, float]:
    mask = load_gray_array(mask_path) > 0
    if not mask.any():
        return fallback_x, fallback_y
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def resized_relpaths(dataset: str, instance_id: str, size: int) -> tuple[Path, Path, Path]:
    digest = instance_id.replace("/", "__").replace(":", "_")
    filename = f"{digest}.png"
    return (
        Path(f"resized_{size}") / "brightfield" / dataset / filename,
        Path(f"resized_{size}") / "fluorescence" / dataset / filename,
        Path(f"resized_{size}") / "mask" / dataset / filename,
    )


def materialize_rows(rows: list[dict[str, Any]], output_root: Path, size: int, progress_every: int) -> list[InstanceRecord]:
    records: list[InstanceRecord] = []
    for index, row in enumerate(rows, start=1):
        dataset = str(row["dataset"])
        instance_id = str(row["instance_id"])
        bf_rel, fl_rel, mask_rel = resized_relpaths(dataset, instance_id, size)
        bf_out = output_root / bf_rel
        fl_out = output_root / fl_rel
        mask_out = output_root / mask_rel
        resize_grayscale(Path(row["brightfield_crop_path"]), bf_out, size)
        resize_grayscale(Path(row["fluorescence_crop_path"]), fl_out, size)
        resize_grayscale(Path(row["mask_crop_path"]), mask_out, size, mask=True)
        row["brightfield_256_path"] = str(bf_out)
        row["fluorescence_256_path"] = str(fl_out)
        row["mask_256_path"] = str(mask_out)
        row["brightfield_256_relpath"] = str(bf_rel)
        row["fluorescence_256_relpath"] = str(fl_rel)
        row["mask_256_relpath"] = str(mask_rel)
        row["split_group"] = "|".join(
            [
                str(row.get("dataset", "")),
                str(row.get("object_name", "")),
                str(row.get("position_label", "")),
            ]
        )
        row["split"] = split_from_group(row["split_group"])
        row["is_time_aware"] = str(row.get("image_has_time_series", row.get("has_time_series", "0"))) == "1"
        row.update(fluorescence_features(Path(row["fluorescence_crop_path"]), Path(row["mask_crop_path"])))
        fallback_x = coerce_float(row.get("bbox_x")) + 0.5 * coerce_float(row.get("bbox_w")) - coerce_float(row.get("crop_x"))
        fallback_y = coerce_float(row.get("bbox_y")) + 0.5 * coerce_float(row.get("bbox_h")) - coerce_float(row.get("crop_y"))
        cx, cy = mask_centroid(Path(row["mask_crop_path"]), fallback_x=fallback_x, fallback_y=fallback_y)
        row["centroid_x"] = cx + coerce_float(row.get("crop_x"))
        row["centroid_y"] = cy + coerce_float(row.get("crop_y"))
        row["fluorescence_positive"] = 0
        row["fl_corrected_p90_log"] = safe_log1p_positive(coerce_float(row["fl_corrected_p90"]))
        row["fl_corrected_total_log"] = safe_log1p_positive(coerce_float(row["fl_corrected_total"]))
        records.append(InstanceRecord(row=row, mask_centroid_x=cx, mask_centroid_y=cy))
        if index % max(1, progress_every) == 0:
            print(json.dumps({"materialized": index, "total": len(rows), "dataset": dataset}), flush=True)
    return records


def assign_positive_labels(records: list[InstanceRecord]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    by_dataset: dict[str, list[InstanceRecord]] = defaultdict(list)
    for record in records:
        by_dataset[str(record.row["dataset"])].append(record)
    for dataset, dataset_records in by_dataset.items():
        early = [
            coerce_float(record.row["fl_corrected_p90"])
            for record in dataset_records
            if coerce_int(record.row["time_index"]) <= 1
        ]
        if len(early) < 20:
            early = [coerce_float(record.row["fl_corrected_p90"]) for record in dataset_records]
        threshold = float(max(5.0, np.percentile(np.asarray(early, dtype=np.float32), 95)))
        thresholds[dataset] = threshold
        for record in dataset_records:
            p90 = coerce_float(record.row["fl_corrected_p90"])
            frac = coerce_float(record.row["fl_local_positive_fraction"])
            record.row["fluorescence_positive"] = int(p90 > threshold and frac >= 0.005)
            record.row["fluorescence_positive_threshold"] = threshold
    return thresholds


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0 = coerce_float(a.get("bbox_x"))
    ay0 = coerce_float(a.get("bbox_y"))
    ax1 = ax0 + coerce_float(a.get("bbox_w"))
    ay1 = ay0 + coerce_float(a.get("bbox_h"))
    bx0 = coerce_float(b.get("bbox_x"))
    by0 = coerce_float(b.get("bbox_y"))
    bx1 = bx0 + coerce_float(b.get("bbox_w"))
    by1 = by0 + coerce_float(b.get("bbox_h"))
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def link_tracks(records: list[InstanceRecord], future_exclude_datasets: set[str], min_track_length: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    dynamic = [
        record.row
        for record in records
        if record.row.get("is_time_aware") and str(record.row["dataset"]) not in future_exclude_datasets
    ]
    by_position: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in dynamic:
        by_position[(str(row["dataset"]), str(row["object_name"]))][coerce_int(row["time_index"])].append(row)

    tracks: dict[str, list[dict[str, Any]]] = {}
    row_to_track: dict[str, str] = {}
    next_track_index = 1
    for (dataset, object_name), frames in sorted(by_position.items()):
        active: dict[str, dict[str, Any]] = {}
        last_time: int | None = None
        for time_index in sorted(frames):
            current = frames[time_index]
            assignments: list[tuple[float, str, dict[str, Any]]] = []
            if active and last_time is not None and time_index - last_time <= 2:
                for track_id, prev in active.items():
                    for row in current:
                        dx = coerce_float(prev["centroid_x"]) - coerce_float(row["centroid_x"])
                        dy = coerce_float(prev["centroid_y"]) - coerce_float(row["centroid_y"])
                        diag = math.hypot(coerce_float(row["source_image_width_px"], 512), coerce_float(row["source_image_height_px"], 512))
                        dist_norm = math.hypot(dx, dy) / max(diag, 1.0)
                        area_prev = max(coerce_float(prev["area_px"]), 1.0)
                        area_now = max(coerce_float(row["area_px"]), 1.0)
                        area_ratio = max(area_prev, area_now) / max(min(area_prev, area_now), 1.0)
                        iou = bbox_iou(prev, row)
                        if dist_norm <= 0.18 and area_ratio <= 4.0:
                            score = dist_norm + 0.05 * math.log(area_ratio) - 0.04 * iou
                            assignments.append((score, track_id, row))
            used_tracks: set[str] = set()
            used_rows: set[str] = set()
            new_active: dict[str, dict[str, Any]] = {}
            for _, track_id, row in sorted(assignments, key=lambda item: item[0]):
                instance_id = str(row["instance_id"])
                if track_id in used_tracks or instance_id in used_rows:
                    continue
                used_tracks.add(track_id)
                used_rows.add(instance_id)
                tracks[track_id].append(row)
                row_to_track[instance_id] = track_id
                new_active[track_id] = row
            for row in current:
                instance_id = str(row["instance_id"])
                if instance_id in used_rows:
                    continue
                track_id = f"{dataset}::{object_name}::track_{next_track_index:06d}"
                next_track_index += 1
                tracks[track_id] = [row]
                row_to_track[instance_id] = track_id
                new_active[track_id] = row
            active = new_active
            last_time = time_index

    track_rows: list[dict[str, Any]] = []
    keep_track_ids = {track_id for track_id, items in tracks.items() if len(items) >= min_track_length}
    for track_id, items in sorted(tracks.items()):
        if track_id not in keep_track_ids:
            for row in items:
                row_to_track.pop(str(row["instance_id"]), None)
            continue
        items = sorted(items, key=lambda row: coerce_int(row["time_index"]))
        positives = [coerce_int(row["fluorescence_positive"]) for row in items]
        positive_times = [coerce_int(row["time_index"]) for row in items if coerce_int(row["fluorescence_positive"]) == 1]
        track_rows.append(
            {
                "track_id": track_id,
                "dataset": items[0]["dataset"],
                "object_name": items[0]["object_name"],
                "split_group": items[0]["split_group"],
                "split": items[0]["split"],
                "length": len(items),
                "min_time_index": coerce_int(items[0]["time_index"]),
                "max_time_index": coerce_int(items[-1]["time_index"]),
                "ever_positive": int(any(positives)),
                "first_positive_time_index": min(positive_times) if positive_times else "",
                "instance_ids_json": json.dumps([row["instance_id"] for row in items]),
            }
        )
    return track_rows, row_to_track


def build_future_samples(
    records: list[InstanceRecord],
    row_to_track: dict[str, str],
    max_prefix_frames: int,
    future_guard_frames: int,
    min_future_frames: int,
    allow_positive_prefix: bool,
) -> list[dict[str, Any]]:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        track_id = row_to_track.get(str(record.row["instance_id"]))
        if track_id:
            by_track[track_id].append(record.row)
    samples: list[dict[str, Any]] = []
    for track_id, items in sorted(by_track.items()):
        items = sorted(items, key=lambda row: coerce_int(row["time_index"]))
        for end_idx in range(len(items)):
            future_start = end_idx + future_guard_frames + 1
            if len(items) - future_start < min_future_frames:
                continue
            prefix = items[max(0, end_idx + 1 - max_prefix_frames) : end_idx + 1]
            if not allow_positive_prefix and any(coerce_int(row["fluorescence_positive"]) == 1 for row in prefix):
                continue
            future = items[future_start:]
            future_positive = int(any(coerce_int(row["fluorescence_positive"]) == 1 for row in future))
            future_peak = max(coerce_float(row["fl_corrected_p90"]) for row in future)
            future_auc = sum(max(0.0, coerce_float(row["fl_corrected_p90"])) for row in future)
            positive_future_times = [coerce_int(row["time_index"]) for row in future if coerce_int(row["fluorescence_positive"]) == 1]
            prefix_end_time = coerce_int(items[end_idx]["time_index"])
            samples.append(
                {
                    "future_sample_id": f"{track_id}::prefix_t{prefix_end_time:04d}",
                    "track_id": track_id,
                    "dataset": items[0]["dataset"],
                    "object_name": items[0]["object_name"],
                    "split_group": items[0]["split_group"],
                    "split": items[0]["split"],
                    "prefix_end_time_index": prefix_end_time,
                    "prefix_length": len(prefix),
                    "future_frame_count": len(future),
                    "prefix_instance_ids_json": json.dumps([row["instance_id"] for row in prefix]),
                    "future_instance_ids_json": json.dumps([row["instance_id"] for row in future]),
                    "future_positive": future_positive,
                    "future_peak_corrected_p90": future_peak,
                    "future_peak_log": safe_log1p_positive(future_peak),
                    "future_auc_corrected_p90": future_auc,
                    "future_auc_log": safe_log1p_positive(future_auc),
                    "future_first_positive_time_index": positive_future_times[0] if positive_future_times else "",
                    "future_onset_delta": (positive_future_times[0] - prefix_end_time) if positive_future_times else "",
                }
            )
    return samples


def write_sqlite(output_root: Path, instance_rows: list[dict[str, Any]], track_rows: list[dict[str, Any]], future_samples: list[dict[str, Any]]) -> None:
    db_path = output_root / "future_expression.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        for table_name, rows in (
            ("instances", instance_rows),
            ("tracks", track_rows),
            ("future_samples", future_samples),
        ):
            if not rows:
                continue
            columns = list(rows[0].keys())
            conn.execute(f"create table {table_name} ({', '.join([c + ' text' for c in columns])})")
            placeholders = ", ".join(["?"] * len(columns))
            conn.executemany(
                f"insert into {table_name} ({', '.join(columns)}) values ({placeholders})",
                [[str(row.get(column, "")) for column in columns] for row in rows],
            )
        conn.execute("create index if not exists idx_instances_split on instances(split)")
        conn.execute("create index if not exists idx_instances_dataset on instances(dataset)")
        conn.execute("create index if not exists idx_future_split on future_samples(split)")
        conn.execute("create index if not exists idx_future_track on future_samples(track_id)")
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = fetch_projected_rows(args.db_path.expanduser().resolve(), args.limit)
    records = materialize_rows(rows, output_root, args.size, args.progress_every)
    thresholds = assign_positive_labels(records)
    future_exclude = {item.strip() for item in args.future_exclude_datasets.split(",") if item.strip()}
    track_rows, row_to_track = link_tracks(records, future_exclude, args.min_track_length)
    for record in records:
        record.row["track_id"] = row_to_track.get(str(record.row["instance_id"]), "")
    future_samples = build_future_samples(
        records,
        row_to_track=row_to_track,
        max_prefix_frames=args.max_prefix_frames,
        future_guard_frames=args.future_guard_frames,
        min_future_frames=args.min_future_frames,
        allow_positive_prefix=args.allow_positive_prefix,
    )

    instance_rows = [record.row for record in records]
    manifest_path = output_root / "manifests" / "projected_instances_manifest.csv"
    tracks_path = output_root / "manifests" / "tracks.csv"
    future_path = output_root / "manifests" / "future_samples.csv"
    write_csv(manifest_path, instance_rows)
    write_csv(tracks_path, track_rows)
    write_csv(future_path, future_samples)
    write_sqlite(output_root, instance_rows, track_rows, future_samples)

    summary = {
        "db_path": str(args.db_path),
        "output_root": str(output_root),
        "size": args.size,
        "instance_count": len(instance_rows),
        "track_count": len(track_rows),
        "future_sample_count": len(future_samples),
        "future_exclude_datasets": sorted(future_exclude),
        "allow_positive_prefix": bool(args.allow_positive_prefix),
        "fluorescence_positive_thresholds": thresholds,
        "instances_by_dataset": dict(Counter(str(row["dataset"]) for row in instance_rows)),
        "positive_by_dataset": dict(
            Counter(str(row["dataset"]) for row in instance_rows if coerce_int(row["fluorescence_positive"]) == 1)
        ),
        "future_samples_by_split": dict(Counter(str(row["split"]) for row in future_samples)),
        "instances_by_split": dict(Counter(str(row["split"]) for row in instance_rows)),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
