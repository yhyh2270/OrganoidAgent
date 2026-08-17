#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_TOOL_DIR = REPO_ROOT / "analysis-tools" / "yichao_instance_pairs"
import sys

sys.path.insert(0, str(V1_TOOL_DIR))
from size_metadata import percentile_ranks, quantile_level_20, quantile_summary  # noqa: E402


IMAGE_COLUMNS = [
    "image_id",
    "dataset_version",
    "dataset_folder",
    "lif_stem",
    "day_hint",
    "object_name",
    "field_id",
    "series_index",
    "time_index",
    "z_index",
    "stage",
    "diameters_px_json",
    "channel_map_kind",
    "channel_map_confidence",
    "main_target_name",
    "main_brightfield_channel",
    "main_fluorescence_channel",
    "channel_map_note",
    "segmentation_policy",
    "signal_recovery_used",
    "image_height_px",
    "image_width_px",
    "source_brightfield_path",
    "source_green_fluorescence_path",
    "source_auxiliary_fluorescence_paths_json",
    "source_auxiliary_brightfield_paths_json",
    "image_dir",
    "brightfield_input_path",
    "green_fluorescence_reference_path",
    "local_auxiliary_fluorescence_paths_json",
    "local_auxiliary_brightfield_paths_json",
    "signal_png",
    "support_png",
    "mask_16bit_png",
    "instance_rgb_png",
    "overlay_on_brightfield_png",
    "overlay_on_green_fluorescence_png",
    "comparison_panel_png",
    "mask_count",
    "branch_summaries_json",
    "merged_candidates_json",
    "processed_at",
]

INSTANCE_COLUMNS = [
    "instance_id",
    "image_id",
    "dataset_version",
    "dataset_folder",
    "lif_stem",
    "day_hint",
    "object_name",
    "field_id",
    "series_index",
    "time_index",
    "z_index",
    "channel_map_confidence",
    "source_brightfield_path",
    "source_green_fluorescence_path",
    "source_auxiliary_fluorescence_paths_json",
    "source_auxiliary_brightfield_paths_json",
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
    "green_fluorescence_crop_path",
    "mask_crop_path",
    "overlay_on_brightfield_crop_path",
    "overlay_on_green_fluorescence_crop_path",
    "instance_rgb_crop_path",
    "auxiliary_fluorescence_crop_paths_json",
    "auxiliary_brightfield_crop_paths_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifests and SQLite database for Yichao v2 instance pairs.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "analysis-outputs/yichao_v2_instance_pairs"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_record_paths(root: Path, filename: str) -> list[Path]:
    paths: list[Path] = []
    if not root.exists():
        return paths
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            paths.append(Path(dirpath) / filename)
    return sorted(paths)


def flatten_image_record(record: dict[str, object]) -> dict[str, object]:
    row = {column: record.get(column, "") for column in IMAGE_COLUMNS}
    row["diameters_px_json"] = json.dumps(record.get("diameters_px", []), ensure_ascii=False)
    row["branch_summaries_json"] = json.dumps(record.get("branch_summaries", []), ensure_ascii=False)
    row["merged_candidates_json"] = json.dumps(record.get("merged_candidates", []), ensure_ascii=False)
    row["signal_recovery_used"] = int(bool(record.get("signal_recovery_used", False)))
    return row


def flatten_instance_record(record: dict[str, object]) -> dict[str, object]:
    return {column: record.get(column, "") for column in INSTANCE_COLUMNS}


def add_quantile_fields(rows: list[dict[str, object]], value_key: str) -> None:
    if not rows:
        return
    values = [float(row[value_key]) for row in rows]
    percentiles = percentile_ranks(values)
    for row, percentile in zip(rows, percentiles, strict=True):
        p = float(percentile)
        row[f"{value_key}_percentile"] = f"{p:.6f}"
        row[f"{value_key}_quantile_level_20"] = quantile_level_20(p)
        row[f"{value_key}_within_middle_90"] = int(0.05 <= p <= 0.95)
        row[f"{value_key}_within_middle_95"] = int(0.025 <= p <= 0.975)


def enrich_instances(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in rows:
        crop_w = int(row["crop_w"])
        crop_h = int(row["crop_h"])
        row["crop_area_px"] = crop_w * crop_h
    add_quantile_fields(rows, "area_px")
    add_quantile_fields(rows, "square_crop_size_px")
    add_quantile_fields(rows, "crop_area_px")
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metric(rows: list[dict[str, object]], value_key: str) -> dict[str, float]:
    return quantile_summary(float(row[value_key]) for row in rows) if rows else {}


def summarize(image_rows: list[dict[str, object]], instance_rows: list[dict[str, object]]) -> dict[str, object]:
    datasets = sorted({str(row["dataset_folder"]) for row in image_rows})
    by_dataset: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        image_subset = [row for row in image_rows if row["dataset_folder"] == dataset]
        instance_subset = [row for row in instance_rows if row["dataset_folder"] == dataset]
        edge_count = sum(int(row["is_edge_padded"]) for row in instance_subset)
        by_dataset[dataset] = {
            "image_count": len(image_subset),
            "instance_count": len(instance_subset),
            "edge_padded_instance_count": edge_count,
            "edge_padded_instance_fraction": edge_count / len(instance_subset) if instance_subset else 0.0,
            "channel_map_confidences": sorted({str(row["channel_map_confidence"]) for row in image_subset}),
            "quantiles": {
                "area_px": summarize_metric(instance_subset, "area_px"),
                "square_crop_size_px": summarize_metric(instance_subset, "square_crop_size_px"),
                "crop_area_px": summarize_metric(instance_subset, "crop_area_px"),
            },
        }
    edge_total = sum(int(row["is_edge_padded"]) for row in instance_rows)
    return {
        "dataset_version": "v2",
        "image_count": len(image_rows),
        "instance_count": len(instance_rows),
        "edge_padded_instance_count": edge_total,
        "edge_padded_instance_fraction": edge_total / len(instance_rows) if instance_rows else 0.0,
        "channel_map_confidences": sorted({str(row["channel_map_confidence"]) for row in image_rows}),
        "quantiles": {
            "area_px": summarize_metric(instance_rows, "area_px"),
            "square_crop_size_px": summarize_metric(instance_rows, "square_crop_size_px"),
            "crop_area_px": summarize_metric(instance_rows, "crop_area_px"),
        },
        "datasets": by_dataset,
    }


def build_sqlite(db_path: Path, image_rows: list[dict[str, object]], instance_rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS metadata")
        cur.execute("DROP TABLE IF EXISTS images")
        cur.execute("DROP TABLE IF EXISTS instances")
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT)")
        cur.execute("CREATE TABLE images (" + ", ".join(f"{column} TEXT" for column in IMAGE_COLUMNS) + ", PRIMARY KEY(image_id))")
        cur.execute("CREATE TABLE instances (" + ", ".join(f"{column} TEXT" for column in INSTANCE_COLUMNS) + ", PRIMARY KEY(instance_id))")
        cur.executemany(
            "INSERT INTO images (" + ", ".join(IMAGE_COLUMNS) + ") VALUES (" + ", ".join("?" for _ in IMAGE_COLUMNS) + ")",
            [[str(row.get(column, "")) for column in IMAGE_COLUMNS] for row in image_rows],
        )
        cur.executemany(
            "INSERT INTO instances (" + ", ".join(INSTANCE_COLUMNS) + ") VALUES (" + ", ".join("?" for _ in INSTANCE_COLUMNS) + ")",
            [[str(row.get(column, "")) for column in INSTANCE_COLUMNS] for row in instance_rows],
        )
        cur.execute("INSERT INTO metadata (key, value_json) VALUES (?, ?)", ("summary", json.dumps(summary, ensure_ascii=False)))
        cur.execute("CREATE INDEX idx_v2_images_dataset ON images(dataset_folder)")
        cur.execute("CREATE INDEX idx_v2_images_confidence ON images(channel_map_confidence)")
        cur.execute("CREATE INDEX idx_v2_instances_dataset ON instances(dataset_folder)")
        cur.execute("CREATE INDEX idx_v2_instances_edge ON instances(is_edge_padded)")
        cur.execute("CREATE INDEX idx_v2_instances_area_percentile ON instances(area_px_percentile)")
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    image_paths = find_record_paths(output_root / "images", "image_record.json")
    instance_paths = find_record_paths(output_root / "instances", "instance_record.json")
    if not image_paths:
        raise RuntimeError(f"No image records found under {output_root / 'images'}")
    image_rows = [flatten_image_record(read_json(path)) for path in image_paths]
    instance_rows = enrich_instances([flatten_instance_record(read_json(path)) for path in instance_paths])
    summary = summarize(image_rows, instance_rows)

    manifest_dir = output_root / "manifests"
    write_csv(manifest_dir / "image_records.csv", IMAGE_COLUMNS, image_rows)
    write_csv(manifest_dir / "instance_records.csv", INSTANCE_COLUMNS, instance_rows)
    (manifest_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    db_path = output_root / "database" / "yichao_v2_instance_pairs.sqlite"
    build_sqlite(db_path, image_rows, instance_rows, summary)
    print(manifest_dir)
    print(db_path)
    print(manifest_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
