#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_db = repo_root / "analysis-outputs" / "yichao_instance_pairs" / "database" / "instance_pairs.sqlite"
    default_out = repo_root / "analysis-outputs" / "yichao_instance_pairs_resized_256"
    parser = argparse.ArgumentParser(description="Prepare a resized 256x256 brightfield/fluorescence instance-pair dataset from the finished Yichao instance database.")
    parser.add_argument("--db-path", default=str(default_db))
    parser.add_argument("--output-root", default=str(default_out))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--progress-every", type=int, default=2000)
    return parser.parse_args()


def ensure_output_root(output_root: Path, overwrite: bool, incremental: bool, refresh_metadata: bool) -> None:
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        elif not incremental and not refresh_metadata:
            raise RuntimeError(f"Output already exists: {output_root}. Use --overwrite to rebuild it.")
    output_root.mkdir(parents=True, exist_ok=True)


def save_resized_grayscale(src_path: Path, dst_path: Path, size: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)
    with Image.open(src_path) as image:
        resized = image.convert("L").resize((size, size), resample=Image.Resampling.BILINEAR)
        resized.save(tmp_path)
    tmp_path.replace(dst_path)


def fetch_rows(db_path: Path, limit: int | None) -> list[dict[str, object]]:
    query = """
        SELECT
            instance_id,
            image_id,
            dataset,
            object_name,
            series_index,
            time_index,
            z_index,
            instance_label,
            area_px,
            square_crop_size_px,
            crop_area_px,
            padding_px,
            is_edge_padded,
            source_image_width_px,
            source_image_height_px,
            area_px_percentile,
            area_px_quantile_level_20,
            area_px_within_middle_90,
            area_px_within_middle_95,
            square_crop_size_px_percentile,
            square_crop_size_px_quantile_level_20,
            square_crop_size_px_within_middle_90,
            square_crop_size_px_within_middle_95,
            crop_area_px_percentile,
            crop_area_px_quantile_level_20,
            crop_area_px_within_middle_90,
            crop_area_px_within_middle_95,
            source,
            brightfield_crop_path,
            fluorescence_crop_path,
            mask_crop_path,
            overlay_on_brightfield_crop_path,
            overlay_on_fluorescence_crop_path,
            instance_rgb_crop_path,
            source_brightfield_path,
            source_fluorescence_path
        FROM instances
        ORDER BY dataset, object_name, series_index, time_index, z_index, instance_label
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = [dict(row) for row in db.execute(query)]
    db.close()
    if limit is not None:
        return rows[:limit]
    return rows


def load_existing_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest_sqlite_path(output_root: Path) -> Path:
    return output_root / "metadata" / "resized_pairs.sqlite"


def build_manifest_row(pair_index: int, record: dict[str, object], output_root: Path, size: int) -> dict[str, object]:
    dataset = str(record["dataset"])
    pair_stem = f"{pair_index:06d}.png"
    brightfield_rel = Path("brightfield_256") / dataset / pair_stem
    fluorescence_rel = Path("fluorescence_256") / dataset / pair_stem
    return {
        "pair_index": pair_index,
        "pair_stem": pair_stem,
        "dataset": dataset,
        "object_name": record["object_name"],
        "image_id": record["image_id"],
        "instance_id": record["instance_id"],
        "series_index": int(record["series_index"]),
        "time_index": int(record["time_index"]),
        "z_index": int(record["z_index"]),
        "instance_label": int(record["instance_label"]),
        "area_px": int(record["area_px"]),
        "square_crop_size_px": int(record["square_crop_size_px"]),
        "crop_area_px": int(record["crop_area_px"]),
        "padding_px": int(record["padding_px"]),
        "is_edge_padded": int(record["is_edge_padded"]),
        "source_image_width_px": int(record["source_image_width_px"]),
        "source_image_height_px": int(record["source_image_height_px"]),
        "area_px_percentile": float(record["area_px_percentile"]),
        "area_px_quantile_level_20": int(record["area_px_quantile_level_20"]),
        "area_px_within_middle_90": int(record["area_px_within_middle_90"]),
        "area_px_within_middle_95": int(record["area_px_within_middle_95"]),
        "square_crop_size_px_percentile": float(record["square_crop_size_px_percentile"]),
        "square_crop_size_px_quantile_level_20": int(record["square_crop_size_px_quantile_level_20"]),
        "square_crop_size_px_within_middle_90": int(record["square_crop_size_px_within_middle_90"]),
        "square_crop_size_px_within_middle_95": int(record["square_crop_size_px_within_middle_95"]),
        "crop_area_px_percentile": float(record["crop_area_px_percentile"]),
        "crop_area_px_quantile_level_20": int(record["crop_area_px_quantile_level_20"]),
        "crop_area_px_within_middle_90": int(record["crop_area_px_within_middle_90"]),
        "crop_area_px_within_middle_95": int(record["crop_area_px_within_middle_95"]),
        "source": record["source"],
        "brightfield_256_path": str(output_root / brightfield_rel),
        "fluorescence_256_path": str(output_root / fluorescence_rel),
        "brightfield_256_relpath": str(brightfield_rel),
        "fluorescence_256_relpath": str(fluorescence_rel),
        "original_brightfield_crop_path": record["brightfield_crop_path"],
        "original_fluorescence_crop_path": record["fluorescence_crop_path"],
        "mask_crop_path": record["mask_crop_path"],
        "overlay_on_brightfield_crop_path": record["overlay_on_brightfield_crop_path"],
        "overlay_on_fluorescence_crop_path": record["overlay_on_fluorescence_crop_path"],
        "instance_rgb_crop_path": record["instance_rgb_crop_path"],
        "source_brightfield_path": record["source_brightfield_path"],
        "source_fluorescence_path": record["source_fluorescence_path"],
        "resized_width_px": size,
        "resized_height_px": size,
    }


def rewrite_manifest(
    manifest_path: Path,
    rows: list[dict[str, object]],
    output_root: Path,
    size: int,
    progress_every: int,
) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rewritten = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] | None = None
        for pair_index, record in enumerate(rows):
            manifest_row = build_manifest_row(pair_index, record, output_root, size=size)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_row.keys()))
                writer.writeheader()
            brightfield_out = Path(manifest_row["brightfield_256_path"])
            fluorescence_out = Path(manifest_row["fluorescence_256_path"])
            if not brightfield_out.exists():
                save_resized_grayscale(Path(str(record["brightfield_crop_path"])), brightfield_out, size=size)
            if not fluorescence_out.exists():
                save_resized_grayscale(Path(str(record["fluorescence_crop_path"])), fluorescence_out, size=size)
            writer.writerow(manifest_row)
            rewritten += 1
            if rewritten % max(1, progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "rewritten_manifest_rows": rewritten,
                            "total_pairs": len(rows),
                            "last_dataset": manifest_row["dataset"],
                        }
                    ),
                    flush=True,
                )
    return rewritten


def repair_existing_outputs(rows: list[dict[str, str]], size: int, progress_every: int) -> int:
    repaired = 0
    for index, row in enumerate(rows, start=1):
        brightfield_out = Path(row["brightfield_256_path"])
        fluorescence_out = Path(row["fluorescence_256_path"])
        if brightfield_out.exists() and fluorescence_out.exists():
            continue
        save_resized_grayscale(Path(row["original_brightfield_crop_path"]), brightfield_out, size)
        save_resized_grayscale(Path(row["original_fluorescence_crop_path"]), fluorescence_out, size)
        repaired += 1
        if repaired % max(1, progress_every) == 0:
            print(
                json.dumps(
                    {
                        "repaired_existing_pairs": repaired,
                        "scanned_existing_rows": index,
                    }
                ),
                flush=True,
            )
    return repaired


def write_sqlite(output_root: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> Path:
    db_path = manifest_sqlite_path(output_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        columns = list(rows[0].keys()) if rows else []
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS metadata")
        cur.execute("DROP TABLE IF EXISTS resized_pairs")
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT)")
        cur.execute(
            "CREATE TABLE resized_pairs ("
            + ", ".join(f"{column} TEXT" for column in columns)
            + ")"
        )
        if rows:
            cur.executemany(
                "INSERT INTO resized_pairs (" + ", ".join(columns) + ") VALUES (" + ", ".join("?" for _ in columns) + ")",
                [[str(row.get(column, "")) for column in columns] for row in rows],
            )
            cur.execute("CREATE INDEX idx_resized_pairs_dataset ON resized_pairs(dataset)")
            cur.execute("CREATE INDEX idx_resized_pairs_is_edge_padded ON resized_pairs(is_edge_padded)")
            cur.execute("CREATE INDEX idx_resized_pairs_area_pct ON resized_pairs(area_px_percentile)")
            cur.execute("CREATE INDEX idx_resized_pairs_square_pct ON resized_pairs(square_crop_size_px_percentile)")
            cur.execute("CREATE INDEX idx_resized_pairs_crop_area_pct ON resized_pairs(crop_area_px_percentile)")
        cur.execute(
            "INSERT INTO metadata (key, value_json) VALUES (?, ?)",
            ("summary", json.dumps(summary, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def write_summary(output_root: Path, db_path: Path, rows: list[dict[str, object]], size: int) -> dict[str, object]:
    per_dataset = Counter(str(row["dataset"]) for row in rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "source_database": str(db_path),
        "resized_database": str(manifest_sqlite_path(output_root)),
        "pair_count": len(rows),
        "resized_width_px": size,
        "resized_height_px": size,
        "brightfield_channel": "c1",
        "fluorescence_channel": "c0",
        "edge_padded_pair_count": sum(int(row["is_edge_padded"]) for row in rows),
        "datasets": dict(sorted(per_dataset.items())),
    }
    path = output_root / "metadata" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).resolve()
    output_root = Path(args.output_root).resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"Missing source database: {db_path}")

    ensure_output_root(
        output_root,
        overwrite=args.overwrite,
        incremental=args.incremental,
        refresh_metadata=args.refresh_metadata,
    )
    rows = fetch_rows(db_path, limit=args.limit)
    if not rows:
        raise RuntimeError("No instance rows found in the source database.")

    manifest_path = output_root / "metadata" / "pairs_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if args.refresh_metadata:
        rewritten = rewrite_manifest(
            manifest_path=manifest_path,
            rows=rows,
            output_root=output_root,
            size=args.size,
            progress_every=args.progress_every,
        )
        refreshed_rows = load_existing_manifest(manifest_path)
        summary = write_summary(output_root, db_path=db_path, rows=refreshed_rows, size=args.size)
        resized_db = write_sqlite(output_root, refreshed_rows, summary)
        print(
            json.dumps(
                {
                    "refreshed_pairs": rewritten,
                    "total_pairs": len(refreshed_rows),
                    "resized_database": str(resized_db),
                }
            ),
            flush=True,
        )
        print(output_root)
        print(manifest_path)
        print(resized_db)
        return 0

    existing_rows = [] if args.overwrite else load_existing_manifest(manifest_path)
    repaired_existing = repair_existing_outputs(existing_rows, size=args.size, progress_every=args.progress_every)

    seen_instance_ids = {row["instance_id"] for row in existing_rows}
    next_pair_index = max((int(row["pair_index"]) for row in existing_rows), default=-1) + 1
    new_rows: list[dict[str, object]] = []

    mode = "a" if existing_rows else "w"
    with manifest_path.open(mode, newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] | None = None
        if existing_rows:
            writer = csv.DictWriter(handle, fieldnames=list(existing_rows[0].keys()))
        for record in rows:
            if str(record["instance_id"]) in seen_instance_ids:
                continue
            manifest_row = build_manifest_row(next_pair_index, record, output_root, size=args.size)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_row.keys()))
                writer.writeheader()

            brightfield_out = Path(manifest_row["brightfield_256_path"])
            fluorescence_out = Path(manifest_row["fluorescence_256_path"])
            save_resized_grayscale(Path(str(record["brightfield_crop_path"])), brightfield_out, size=args.size)
            save_resized_grayscale(Path(str(record["fluorescence_crop_path"])), fluorescence_out, size=args.size)
            writer.writerow(manifest_row)
            new_rows.append(manifest_row)
            next_pair_index += 1

            if len(new_rows) % max(1, args.progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "added_pairs": len(new_rows),
                            "existing_pairs": len(existing_rows),
                            "last_dataset": manifest_row["dataset"],
                        }
                    ),
                    flush=True,
                )

    all_rows: list[dict[str, object]] = list(existing_rows)
    all_rows.extend(new_rows)
    summary = write_summary(output_root, db_path=db_path, rows=all_rows, size=args.size)
    resized_db = write_sqlite(output_root, all_rows, summary)
    print(
        json.dumps(
            {
                "existing_pairs": len(existing_rows),
                "repaired_existing_pairs": repaired_existing,
                "added_pairs": len(new_rows),
                "total_pairs": len(all_rows),
                "resized_database": str(resized_db),
            }
        ),
        flush=True,
    )
    print(output_root)
    print(manifest_path)
    print(resized_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
