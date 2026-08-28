#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path

from size_metadata import percentile_ranks, quantile_level_20, quantile_summary


IMAGE_COLUMNS = [
    "image_id",
    "dataset",
    "object_name",
    "series_index",
    "time_index",
    "z_index",
    "stage",
    "diameters_px_json",
    "selected_channel",
    "paired_channel",
    "segmentation_policy",
    "signal_recovery_used",
    "image_height_px",
    "image_width_px",
    "source_brightfield_path",
    "source_fluorescence_path",
    "image_dir",
    "brightfield_input_path",
    "fluorescence_reference_path",
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


INSTANCE_COLUMNS = [
    "instance_id",
    "image_id",
    "dataset",
    "object_name",
    "series_index",
    "time_index",
    "z_index",
    "source_brightfield_path",
    "source_fluorescence_path",
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
    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "analysis-outputs/yichao_instance_pairs"
    parser = argparse.ArgumentParser(description="Build CSV and SQLite manifests for Yichao instance-pair outputs.")
    parser.add_argument("--output-root", default=str(default_output))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_image_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "image_id": record["image_id"],
        "dataset": record["dataset"],
        "object_name": record["object_name"],
        "series_index": record["series_index"],
        "time_index": record["time_index"],
        "z_index": record["z_index"],
        "stage": record["stage"],
        "diameters_px_json": json.dumps(record["diameters_px"], ensure_ascii=False),
        "selected_channel": record["selected_channel"],
        "paired_channel": record["paired_channel"],
        "segmentation_policy": record["segmentation_policy"],
        "signal_recovery_used": int(bool(record["signal_recovery_used"])),
        "image_height_px": record["image_height_px"],
        "image_width_px": record["image_width_px"],
        "source_brightfield_path": record["source_brightfield_path"],
        "source_fluorescence_path": record["source_fluorescence_path"],
        "image_dir": record["image_dir"],
        "brightfield_input_path": record["brightfield_input_path"],
        "fluorescence_reference_path": record["fluorescence_reference_path"],
        "signal_png": record["signal_png"],
        "support_png": record["support_png"],
        "mask_16bit_png": record["mask_16bit_png"],
        "instance_rgb_png": record["instance_rgb_png"],
        "overlay_on_brightfield_png": record["overlay_on_brightfield_png"],
        "overlay_on_fluorescence_png": record["overlay_on_fluorescence_png"],
        "comparison_panel_png": record["comparison_panel_png"],
        "mask_count": record["mask_count"],
        "branch_summaries_json": json.dumps(record["branch_summaries"], ensure_ascii=False),
        "merged_candidates_json": json.dumps(record["merged_candidates"], ensure_ascii=False),
        "processed_at": record["processed_at"],
    }


def flatten_instance_record(record: dict[str, object]) -> dict[str, object]:
    return {column: record.get(column, "") for column in INSTANCE_COLUMNS}


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
        cur.execute("DROP TABLE IF EXISTS images")
        cur.execute("DROP TABLE IF EXISTS instances")
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT)")
        cur.execute(
            "CREATE TABLE images ("
            + ", ".join(f"{column} TEXT" for column in IMAGE_COLUMNS[:-1])
            + ", processed_at TEXT, PRIMARY KEY(image_id))"
        )
        cur.execute(
            "CREATE TABLE instances ("
            + ", ".join(f"{column} TEXT" for column in INSTANCE_COLUMNS)
            + ", PRIMARY KEY(instance_id), FOREIGN KEY(image_id) REFERENCES images(image_id))"
        )
        cur.executemany(
            "INSERT INTO images (" + ", ".join(IMAGE_COLUMNS) + ") VALUES (" + ", ".join("?" for _ in IMAGE_COLUMNS) + ")",
            [[str(row.get(column, "")) for column in IMAGE_COLUMNS] for row in image_rows],
        )
        cur.executemany(
            "INSERT INTO instances (" + ", ".join(INSTANCE_COLUMNS) + ") VALUES (" + ", ".join("?" for _ in INSTANCE_COLUMNS) + ")",
            [[str(row.get(column, "")) for column in INSTANCE_COLUMNS] for row in instance_rows],
        )
        cur.execute(
            "INSERT INTO metadata (key, value_json) VALUES (?, ?)",
            ("summary", json.dumps(summary, ensure_ascii=False)),
        )
        cur.execute("CREATE INDEX idx_instances_dataset ON instances(dataset)")
        cur.execute("CREATE INDEX idx_instances_is_edge_padded ON instances(is_edge_padded)")
        cur.execute("CREATE INDEX idx_instances_area_px_percentile ON instances(area_px_percentile)")
        cur.execute("CREATE INDEX idx_instances_square_crop_percentile ON instances(square_crop_size_px_percentile)")
        cur.execute("CREATE INDEX idx_instances_crop_area_percentile ON instances(crop_area_px_percentile)")
        conn.commit()
    finally:
        conn.close()


def derive_is_edge_padded(row: dict[str, object], image_row: dict[str, object]) -> int:
    crop_x = int(row["crop_x"])
    crop_y = int(row["crop_y"])
    crop_w = int(row["crop_w"])
    crop_h = int(row["crop_h"])
    image_w = int(image_row["image_width_px"])
    image_h = int(image_row["image_height_px"])
    return int(
        crop_x < 0
        or crop_y < 0
        or crop_x + crop_w > image_w
        or crop_y + crop_h > image_h
    )


def add_quantile_fields(rows: list[dict[str, object]], value_key: str) -> None:
    values = [float(row[value_key]) for row in rows]
    percentiles = percentile_ranks(values)
    for row, percentile in zip(rows, percentiles, strict=True):
        row[f"{value_key}_percentile"] = f"{float(percentile):.6f}"
        row[f"{value_key}_quantile_level_20"] = quantile_level_20(float(percentile))
        row[f"{value_key}_within_middle_90"] = int(0.05 <= float(percentile) <= 0.95)
        row[f"{value_key}_within_middle_95"] = int(0.025 <= float(percentile) <= 0.975)


def enrich_instance_rows(
    image_rows: list[dict[str, object]],
    instance_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    image_lookup = {row["image_id"]: row for row in image_rows}
    for row in instance_rows:
        image_row = image_lookup[row["image_id"]]
        crop_w = int(row["crop_w"])
        crop_h = int(row["crop_h"])
        row["source_image_width_px"] = int(image_row["image_width_px"])
        row["source_image_height_px"] = int(image_row["image_height_px"])
        row["crop_area_px"] = crop_w * crop_h
        row["is_edge_padded"] = derive_is_edge_padded(row, image_row)

    add_quantile_fields(instance_rows, "area_px")
    add_quantile_fields(instance_rows, "square_crop_size_px")
    add_quantile_fields(instance_rows, "crop_area_px")
    return instance_rows


def summarize_metric(rows: list[dict[str, object]], value_key: str) -> dict[str, float]:
    return quantile_summary(float(row[value_key]) for row in rows)


def summarize(image_rows: list[dict[str, object]], instance_rows: list[dict[str, object]]) -> dict[str, object]:
    datasets = sorted({row["dataset"] for row in image_rows})
    by_dataset: dict[str, dict[str, object]] = {}
    max_crop_w = 0
    max_crop_h = 0
    max_square = 0
    edge_padded_total = sum(int(row["is_edge_padded"]) for row in instance_rows)
    for dataset in datasets:
        image_subset = [row for row in image_rows if row["dataset"] == dataset]
        instance_subset = [row for row in instance_rows if row["dataset"] == dataset]
        dataset_max_w = max((int(row["crop_w"]) for row in instance_subset), default=0)
        dataset_max_h = max((int(row["crop_h"]) for row in instance_subset), default=0)
        dataset_max_square = max((int(row["square_crop_size_px"]) for row in instance_subset), default=0)
        dataset_edge_padded = sum(int(row["is_edge_padded"]) for row in instance_subset)
        max_crop_w = max(max_crop_w, dataset_max_w)
        max_crop_h = max(max_crop_h, dataset_max_h)
        max_square = max(max_square, dataset_max_square)
        by_dataset[dataset] = {
            "image_count": len(image_subset),
            "instance_count": len(instance_subset),
            "max_crop_width_px": dataset_max_w,
            "max_crop_height_px": dataset_max_h,
            "max_square_crop_size_px": dataset_max_square,
            "edge_padded_instance_count": dataset_edge_padded,
            "edge_padded_instance_fraction": (
                float(dataset_edge_padded) / float(len(instance_subset)) if instance_subset else 0.0
            ),
            "quantiles": {
                "area_px": summarize_metric(instance_subset, "area_px"),
                "square_crop_size_px": summarize_metric(instance_subset, "square_crop_size_px"),
                "crop_area_px": summarize_metric(instance_subset, "crop_area_px"),
            },
        }
    return {
        "image_count": len(image_rows),
        "instance_count": len(instance_rows),
        "max_crop_width_px": max_crop_w,
        "max_crop_height_px": max_crop_h,
        "max_square_crop_size_px": max_square,
        "edge_padded_instance_count": edge_padded_total,
        "edge_padded_instance_fraction": float(edge_padded_total) / float(len(instance_rows)) if instance_rows else 0.0,
        "quantiles": {
            "area_px": summarize_metric(instance_rows, "area_px"),
            "square_crop_size_px": summarize_metric(instance_rows, "square_crop_size_px"),
            "crop_area_px": summarize_metric(instance_rows, "crop_area_px"),
        },
        "datasets": by_dataset,
    }


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
        raise RuntimeError(f"No image records found under {output_root / 'images'}")

    image_rows = [flatten_image_record(read_json(path)) for path in image_record_paths]
    instance_rows = enrich_instance_rows(
        image_rows,
        [flatten_instance_record(read_json(path)) for path in instance_record_paths],
    )

    manifest_dir = output_root / "manifests"
    db_path = output_root / "database" / "instance_pairs.sqlite"
    summary_path = manifest_dir / "summary.json"
    images_csv = manifest_dir / "image_records.csv"
    instances_csv = manifest_dir / "instance_records.csv"

    write_csv(images_csv, IMAGE_COLUMNS, image_rows)
    write_csv(instances_csv, INSTANCE_COLUMNS, instance_rows)
    summary = summarize(image_rows, instance_rows)
    build_sqlite(db_path, image_rows, instance_rows, summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(manifest_dir)
    print(db_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
