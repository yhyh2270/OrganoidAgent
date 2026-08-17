#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PER_Z_TOOL_DIR = REPO_ROOT / "analysis-tools" / "yichao_instance_pairs"
sys.path.insert(0, str(PER_Z_TOOL_DIR))

from size_metadata import percentile_ranks, quantile_level_20, quantile_summary  # noqa: E402


PROJECTED_IMAGE_COLUMNS = [
    "image_id",
    "projection_id",
    "dataset",
    "object_name",
    "series_index",
    "time_index",
    "z_indices_json",
    "z_count",
    "stage",
    "diameters_px_json",
    "selected_channel",
    "paired_channel",
    "brightfield_projection_mode",
    "fluorescence_projection_mode",
    "segmentation_policy",
    "signal_recovery_used",
    "image_height_px",
    "image_width_px",
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
    "source_brightfield_paths_json",
    "source_fluorescence_paths_json",
    "image_dir",
    "projection_dir",
    "brightfield_projected_path",
    "fluorescence_projected_path",
    "signal_png",
    "support_png",
    "mask_16bit_png",
    "instance_rgb_png",
    "overlay_on_brightfield_png",
    "overlay_on_fluorescence_png",
    "comparison_panel_png",
    "mask_count",
    "branch_summaries_json",
    "merged_candidates_json",
    "processed_at",
]

PROJECTED_INSTANCE_COLUMNS = [
    "instance_id",
    "image_id",
    "projection_id",
    "dataset",
    "object_name",
    "series_index",
    "time_index",
    "z_indices_json",
    "z_count",
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
    "brightfield_projection_mode",
    "fluorescence_projection_mode",
    "instance_label",
    "area_px",
    "diameter_px",
    "score",
    "support_ratio",
    "mean_signal",
    "edge_strength",
    "circularity",
    "source",
    "source_image_width_px",
    "source_image_height_px",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
    "square_crop_size_px",
    "crop_area_px",
    "padding_px",
    "is_edge_padded",
    "area_px_percentile",
    "area_px_quantile_level_20",
    "area_px_within_middle_90",
    "area_px_within_middle_95",
    "square_crop_size_px_percentile",
    "square_crop_size_px_quantile_level_20",
    "square_crop_size_px_within_middle_90",
    "square_crop_size_px_within_middle_95",
    "crop_area_px_percentile",
    "crop_area_px_quantile_level_20",
    "crop_area_px_within_middle_90",
    "crop_area_px_within_middle_95",
    "instance_dir",
    "brightfield_crop_path",
    "fluorescence_crop_path",
    "mask_crop_path",
    "overlay_on_brightfield_crop_path",
    "overlay_on_fluorescence_crop_path",
    "instance_rgb_crop_path",
]


def parse_args() -> argparse.Namespace:
    default_output = REPO_ROOT / "analysis-outputs" / "yichao_projected_instance_pairs"
    parser = argparse.ArgumentParser(description="Build CSV and SQLite database for projected-z Yichao instance-pair outputs.")
    parser.add_argument("--output-root", default=str(default_output))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_json(value: object) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def flatten_image_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "image_id": record.get("image_id", ""),
        "projection_id": record.get("projection_id", ""),
        "dataset": record.get("dataset", ""),
        "object_name": record.get("object_name", ""),
        "series_index": record.get("series_index", ""),
        "time_index": record.get("time_index", ""),
        "z_indices_json": as_json(record.get("z_indices", [])),
        "z_count": record.get("z_count", ""),
        "stage": record.get("stage", ""),
        "diameters_px_json": as_json(record.get("diameters_px", [])),
        "selected_channel": record.get("selected_channel", ""),
        "paired_channel": record.get("paired_channel", ""),
        "brightfield_projection_mode": record.get("brightfield_projection_mode", ""),
        "fluorescence_projection_mode": record.get("fluorescence_projection_mode", ""),
        "segmentation_policy": record.get("segmentation_policy", ""),
        "signal_recovery_used": int(bool(record.get("signal_recovery_used", False))),
        "image_height_px": record.get("image_height_px", ""),
        "image_width_px": record.get("image_width_px", ""),
        "experiment_label": record.get("experiment_label", ""),
        "experiment_design": record.get("experiment_design", ""),
        "replicate_label": record.get("replicate_label", ""),
        "sample_label": record.get("sample_label", ""),
        "day_label": record.get("day_label", ""),
        "day_index": "" if record.get("day_index") is None else record.get("day_index", ""),
        "position_label": record.get("position_label", ""),
        "position_index": "" if record.get("position_index") is None else record.get("position_index", ""),
        "position_time_count": record.get("position_time_count", ""),
        "has_time_series": int(bool(record.get("has_time_series", 0))),
        "source_brightfield_paths_json": as_json(record.get("source_brightfield_paths", [])),
        "source_fluorescence_paths_json": as_json(record.get("source_fluorescence_paths", [])),
        "image_dir": record.get("image_dir", ""),
        "projection_dir": record.get("projection_dir", ""),
        "brightfield_projected_path": record.get("brightfield_projected_path", ""),
        "fluorescence_projected_path": record.get("fluorescence_projected_path", ""),
        "signal_png": record.get("signal_png", ""),
        "support_png": record.get("support_png", ""),
        "mask_16bit_png": record.get("mask_16bit_png", ""),
        "instance_rgb_png": record.get("instance_rgb_png", ""),
        "overlay_on_brightfield_png": record.get("overlay_on_brightfield_png", ""),
        "overlay_on_fluorescence_png": record.get("overlay_on_fluorescence_png", ""),
        "comparison_panel_png": record.get("comparison_panel_png", ""),
        "mask_count": record.get("mask_count", ""),
        "branch_summaries_json": as_json(record.get("branch_summaries", [])),
        "merged_candidates_json": as_json(record.get("merged_candidates", [])),
        "processed_at": record.get("processed_at", ""),
    }


def flatten_instance_record(record: dict[str, object]) -> dict[str, object]:
    row = {column: record.get(column, "") for column in PROJECTED_INSTANCE_COLUMNS}
    row["z_indices_json"] = as_json(record.get("z_indices", []))
    row["day_index"] = "" if record.get("day_index") is None else record.get("day_index", "")
    row["position_index"] = "" if record.get("position_index") is None else record.get("position_index", "")
    row["has_time_series"] = int(bool(record.get("has_time_series", 0)))
    return row


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def add_quantile_fields(rows: list[dict[str, object]], value_key: str) -> None:
    if not rows:
        return
    values = [float(row[value_key]) for row in rows]
    percentiles = percentile_ranks(values)
    for row, percentile in zip(rows, percentiles, strict=True):
        row[f"{value_key}_percentile"] = f"{float(percentile):.6f}"
        row[f"{value_key}_quantile_level_20"] = quantile_level_20(float(percentile))
        row[f"{value_key}_within_middle_90"] = int(0.05 <= float(percentile) <= 0.95)
        row[f"{value_key}_within_middle_95"] = int(0.025 <= float(percentile) <= 0.975)


def enrich_instance_rows(instance_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in instance_rows:
        crop_w = int(row["crop_w"])
        crop_h = int(row["crop_h"])
        row["crop_area_px"] = crop_w * crop_h
    add_quantile_fields(instance_rows, "area_px")
    add_quantile_fields(instance_rows, "square_crop_size_px")
    add_quantile_fields(instance_rows, "crop_area_px")
    return instance_rows


def summarize_metric(rows: list[dict[str, object]], value_key: str) -> dict[str, float]:
    if not rows:
        return {}
    return quantile_summary(float(row[value_key]) for row in rows)


def summarize(image_rows: list[dict[str, object]], instance_rows: list[dict[str, object]]) -> dict[str, object]:
    datasets = sorted({str(row["dataset"]) for row in image_rows})
    by_dataset: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        image_subset = [row for row in image_rows if row["dataset"] == dataset]
        instance_subset = [row for row in instance_rows if row["dataset"] == dataset]
        complete_count = sum(1 for row in instance_subset if int(row.get("is_edge_padded", 0)) == 0)
        by_dataset[dataset] = {
            "projected_image_count": len(image_subset),
            "projected_instance_count": len(instance_subset),
            "complete_non_edge_instance_count": complete_count,
            "edge_padded_instance_count": len(instance_subset) - complete_count,
            "time_aware_projected_image_count": sum(int(row.get("has_time_series", 0)) for row in image_subset),
            "position_count": len({row["object_name"] for row in image_subset}),
            "quantiles": {
                "area_px": summarize_metric(instance_subset, "area_px"),
                "square_crop_size_px": summarize_metric(instance_subset, "square_crop_size_px"),
                "crop_area_px": summarize_metric(instance_subset, "crop_area_px"),
            },
        }
    complete_total = sum(1 for row in instance_rows if int(row.get("is_edge_padded", 0)) == 0)
    return {
        "projected_image_count": len(image_rows),
        "projected_instance_count": len(instance_rows),
        "complete_non_edge_instance_count": complete_total,
        "edge_padded_instance_count": len(instance_rows) - complete_total,
        "time_aware_projected_image_count": sum(int(row.get("has_time_series", 0)) for row in image_rows),
        "dataset_count": len(datasets),
        "datasets": by_dataset,
        "quantiles": {
            "area_px": summarize_metric(instance_rows, "area_px"),
            "square_crop_size_px": summarize_metric(instance_rows, "square_crop_size_px"),
            "crop_area_px": summarize_metric(instance_rows, "crop_area_px"),
        },
    }


def build_sqlite(
    db_path: Path,
    image_rows: list[dict[str, object]],
    instance_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS metadata")
        cur.execute("DROP TABLE IF EXISTS projected_images")
        cur.execute("DROP TABLE IF EXISTS projected_instances")
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT)")
        cur.execute(
            "CREATE TABLE projected_images ("
            + ", ".join(f"{column} TEXT" for column in PROJECTED_IMAGE_COLUMNS)
            + ", PRIMARY KEY(image_id))"
        )
        cur.execute(
            "CREATE TABLE projected_instances ("
            + ", ".join(f"{column} TEXT" for column in PROJECTED_INSTANCE_COLUMNS)
            + ", PRIMARY KEY(instance_id), FOREIGN KEY(image_id) REFERENCES projected_images(image_id))"
        )
        cur.executemany(
            "INSERT INTO projected_images (" + ", ".join(PROJECTED_IMAGE_COLUMNS) + ") VALUES (" + ", ".join("?" for _ in PROJECTED_IMAGE_COLUMNS) + ")",
            [[str(row.get(column, "")) for column in PROJECTED_IMAGE_COLUMNS] for row in image_rows],
        )
        cur.executemany(
            "INSERT INTO projected_instances (" + ", ".join(PROJECTED_INSTANCE_COLUMNS) + ") VALUES (" + ", ".join("?" for _ in PROJECTED_INSTANCE_COLUMNS) + ")",
            [[str(row.get(column, "")) for column in PROJECTED_INSTANCE_COLUMNS] for row in instance_rows],
        )
        cur.execute("INSERT INTO metadata (key, value_json) VALUES (?, ?)", ("summary", json.dumps(summary, ensure_ascii=False)))
        cur.execute("CREATE INDEX idx_projected_images_dataset ON projected_images(dataset)")
        cur.execute("CREATE INDEX idx_projected_images_day ON projected_images(day_index)")
        cur.execute("CREATE INDEX idx_projected_images_position ON projected_images(position_label)")
        cur.execute("CREATE INDEX idx_projected_images_time ON projected_images(time_index)")
        cur.execute("CREATE INDEX idx_projected_instances_dataset ON projected_instances(dataset)")
        cur.execute("CREATE INDEX idx_projected_instances_edge ON projected_instances(is_edge_padded)")
        cur.execute("CREATE INDEX idx_projected_instances_day ON projected_instances(day_index)")
        cur.execute("CREATE INDEX idx_projected_instances_position ON projected_instances(position_label)")
        cur.execute("CREATE INDEX idx_projected_instances_time ON projected_instances(time_index)")
        cur.execute("CREATE INDEX idx_projected_instances_area_percentile ON projected_instances(area_px_percentile)")
        conn.commit()
    finally:
        conn.close()


def first_existing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def find_record_paths(root: Path, filename: str) -> list[Path]:
    paths: list[Path] = []
    if not root.exists():
        return paths
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            paths.append(Path(dirpath) / filename)
    return sorted(paths)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    image_record_paths = find_record_paths(output_root / "images", "image_record.json")
    instance_record_paths = find_record_paths(output_root / "instances", "instance_record.json")
    if not image_record_paths:
        raise RuntimeError(f"No projected image records found under {output_root / 'images'}")

    image_rows = [flatten_image_record(read_json(path)) for path in image_record_paths]
    instance_rows = enrich_instance_rows([flatten_instance_record(read_json(path)) for path in instance_record_paths])
    summary = summarize(image_rows, instance_rows)

    manifest_dir = output_root / "manifests"
    db_path = output_root / "database" / "projected_instance_pairs.sqlite"
    summary_path = manifest_dir / "summary.json"
    images_csv = manifest_dir / "projected_image_records.csv"
    instances_csv = manifest_dir / "projected_instance_records.csv"

    write_csv(images_csv, PROJECTED_IMAGE_COLUMNS, image_rows)
    write_csv(instances_csv, PROJECTED_INSTANCE_COLUMNS, instance_rows)
    build_sqlite(db_path, image_rows, instance_rows, summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(manifest_dir)
    print(db_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
