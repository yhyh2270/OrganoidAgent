#!/usr/bin/env python3
"""Build a manifest bundle for the four Yichao datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from differentiation_prediction.manifest import build_manifest_records, bundle_summary, write_manifest_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manifest-first Yichao dataset tables")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/results/differentiation_prediction/manifest"),
    )
    parser.add_argument("--input-channel", type=int, default=0)
    parser.add_argument("--target-channel", type=int, default=1)
    parser.add_argument("--baseline-seed", type=int, default=17)
    parser.add_argument("--all-dynamic-seed", type=int, default=29)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = build_manifest_records(
        input_channel=args.input_channel,
        target_channel=args.target_channel,
        baseline_seed=args.baseline_seed,
        all_dynamic_seed=args.all_dynamic_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    paths = write_manifest_bundle(records, args.output_dir)
    print("== Yichao manifest bundle ==", flush=True)
    print(bundle_summary(paths), flush=True)
    print(f"- total_rows: {len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
