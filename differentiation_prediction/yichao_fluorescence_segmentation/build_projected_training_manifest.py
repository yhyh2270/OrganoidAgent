#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.utils import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic train/val/test manifest from projected Yichao instance records."
    )
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed-token", default="yichao_projected_maxbf_maxfl_v1")
    parser.add_argument("--include-edge-padded", action="store_true")
    parser.add_argument("--require-projection-mode", choices=("min", "max", "mean", "median"), default=None)
    parser.add_argument("--projection-policy", default="max_brightfield_max_fluorescence")
    return parser.parse_args()


def stable_fraction(token: str) -> float:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def split_group(row: dict[str, str]) -> str:
    pieces = [
        row.get("dataset", ""),
        row.get("object_name", ""),
        row.get("position_label", ""),
        row.get("replicate_label", ""),
        row.get("sample_label", ""),
    ]
    return "|".join(piece for piece in pieces if piece)


def assign_split(group: str, args: argparse.Namespace) -> str:
    value = stable_fraction(f"{args.seed_token}|{group}")
    train_cutoff = max(0.0, min(1.0, args.train_ratio))
    val_cutoff = max(train_cutoff, min(1.0, train_cutoff + max(0.0, args.val_ratio)))
    if value < train_cutoff:
        return "train"
    if value < val_cutoff:
        return "val"
    return "test"


def row_is_edge_padded(row: dict[str, str]) -> bool:
    return str(row.get("is_edge_padded", "0")).strip() not in {"", "0", "0.0", "False", "false"}


def main() -> int:
    args = parse_args()
    if not args.source_records.exists():
        raise SystemExit(f"Missing projected instance records: {args.source_records}")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("--train-ratio and --val-ratio must be non-negative and sum to less than 1")

    dataset_set = set(args.datasets)
    rows = [row for row in read_csv(args.source_records) if row.get("dataset") in dataset_set]
    if args.require_projection_mode is not None:
        rows = [
            row
            for row in rows
            if row.get("brightfield_projection_mode") == args.require_projection_mode
            and row.get("fluorescence_projection_mode") == args.require_projection_mode
        ]
    if not args.include_edge_padded:
        rows = [row for row in rows if not row_is_edge_padded(row)]
    if not rows:
        raise SystemExit(f"No usable rows after filtering {args.source_records}")

    out_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_splits: dict[str, str] = {}
    for row in rows:
        group = split_group(row)
        if group not in group_splits:
            group_splits[group] = assign_split(group, args)
        split = group_splits[group]
        out = dict(row)
        out["split_group"] = group
        out["split"] = split
        out["external_test_only"] = 0
        out["projection_policy"] = args.projection_policy
        out_rows.append(out)
        split_counts[split] += 1
        dataset_counts[out.get("dataset", "unknown")][split] += 1

    write_csv(args.output_manifest, out_rows)
    summary = {
        "source_records": str(args.source_records),
        "output_manifest": str(args.output_manifest),
        "datasets": args.datasets,
        "projection_policy": args.projection_policy,
        "require_projection_mode": args.require_projection_mode,
        "include_edge_padded": bool(args.include_edge_padded),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "count": len(out_rows),
        "split_counts": dict(split_counts),
        "dataset_split_counts": {dataset: dict(counts) for dataset, counts in sorted(dataset_counts.items())},
        "split_group_count": len(group_splits),
    }
    write_json(args.output_manifest.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
