#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.utils import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an external-only per-z instance manifest for Yichao checkpoint evaluation.")
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--dataset", default="Data-Yichao-11")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--split-name", default="external_test")
    parser.add_argument("--include-edge-padded", action="store_true")
    parser.add_argument("--require-middle-95-area", action="store_true")
    return parser.parse_args()


def row_is_edge_padded(row: dict[str, str]) -> bool:
    return str(row.get("is_edge_padded", "0")).strip() not in {"", "0", "0.0", "False", "false"}


def row_is_middle_95(row: dict[str, str]) -> bool:
    return str(row.get("area_px_within_middle_95", "1")).strip() not in {"0", "0.0", "False", "false"}


def main() -> int:
    args = parse_args()
    if not args.source_records.exists():
        raise SystemExit(f"Missing instance records: {args.source_records}")
    rows = [row for row in read_csv(args.source_records) if row.get("dataset") == args.dataset]
    if not args.include_edge_padded:
        rows = [row for row in rows if not row_is_edge_padded(row)]
    if args.require_middle_95_area:
        rows = [row for row in rows if row_is_middle_95(row)]
    if not rows:
        raise SystemExit(f"No rows found for dataset={args.dataset} in {args.source_records}")

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["split"] = args.split_name
        out["external_test_dataset"] = args.dataset
        out["external_test_only"] = 1
        out["manifest_policy"] = "per_z_original_external_holdout"
        out_rows.append(out)

    write_csv(args.output_manifest, out_rows)
    summary = {
        "source_records": str(args.source_records),
        "output_manifest": str(args.output_manifest),
        "dataset": args.dataset,
        "split": args.split_name,
        "count": len(out_rows),
        "include_edge_padded": bool(args.include_edge_padded),
        "require_middle_95_area": bool(args.require_middle_95_area),
    }
    write_json(args.output_manifest.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
