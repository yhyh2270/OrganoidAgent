#!/usr/bin/env python3
"""Build legacy-compatible, zero-copy input views from DEO manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
DEFAULT_OUTPUT = ROOT / "analysis-outputs" / "deo_reproduction" / "input_views"

EXPERIMENTS = {
    "density": {
        "dataset": "01_Density_experiment_10x",
        "view": "density",
        "smoke": {("high", "D00"), ("high", "D16")},
    },
    "alginate": {
        "dataset": "02_Sodium_alginate_experiment_10x",
        "view": "alginate",
        "smoke": {("alginate_0.05pct", "D00"), ("alginate_0.05pct", "D13")},
    },
    "y27632": {
        "dataset": "03_Y-27632_experiment_10x",
        "view": "y27632",
        "smoke": {("Y27632_100uM", "D00"), ("Y27632_100uM", "D06")},
    },
}

Y_CONDITIONS = {
    "Y27632_0uM": "No",
    "Y27632_10uM": "10uM",
    "Y27632_20uM": "20uM",
    "Y27632_50uM": "50uM",
    "Y27632_100uM": "100uM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create hard-linked input trees expected by the original DEO scripts."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke selects two representative images per experiment; full selects all 114.",
    )
    parser.add_argument(
        "--experiment",
        choices=("all", *EXPERIMENTS),
        default="all",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verify-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_filename(source_path: str, fallback: str) -> str:
    name = PureWindowsPath(source_path).name
    return name or fallback


def target_relative_path(experiment: str, row: dict[str, str]) -> Path:
    source_parent = PureWindowsPath(row["source_path"]).parent.name
    filename = source_filename(row["source_path"], row["destination_file"])
    if experiment == "y27632":
        condition = Y_CONDITIONS.get(row["condition"])
        if not condition:
            raise ValueError(f"Unsupported Y-27632 condition: {row['condition']}")
        return Path(condition) / source_parent / filename
    return Path(source_parent) / filename


def link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def prepare_experiment(
    experiment: str,
    config: dict,
    output_root: Path,
    mode: str,
    verify_hash: bool,
    overwrite: bool,
) -> dict:
    dataset_dir = DATASETS / config["dataset"]
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    view_dir = output_root / mode / config["view"]
    if view_dir.exists() and overwrite:
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = []
    for row in rows:
        key = (row["condition"], row["day"])
        if mode == "smoke" and key not in config["smoke"]:
            continue
        source = dataset_dir / row["destination_file"]
        if not source.exists():
            raise FileNotFoundError(source)
        if int(row["bytes"]) != source.stat().st_size:
            raise RuntimeError(f"Size mismatch: {source}")
        if verify_hash and sha256(source).lower() != row["sha256"].lower():
            raise RuntimeError(f"SHA-256 mismatch: {source}")

        relative = target_relative_path(experiment, row)
        target = view_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        method = "existing"
        if not target.exists():
            method = link_or_copy(source, target)
        selected.append(
            {
                "experiment": experiment,
                "condition": row["condition"],
                "day": row["day"],
                "source": str(source),
                "target": str(target),
                "link_method": method,
                "sha256_verified": bool(verify_hash),
            }
        )

    expected = 2 if mode == "smoke" else len(rows)
    if len(selected) != expected:
        raise RuntimeError(
            f"{experiment}: selected {len(selected)} images, expected {expected}"
        )

    manifest_out = view_dir / "view_manifest.csv"
    with manifest_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return {
        "experiment": experiment,
        "dataset": str(dataset_dir),
        "view_dir": str(view_dir),
        "images": len(selected),
        "manifest": str(manifest_out),
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    summaries = [
        prepare_experiment(
            name,
            EXPERIMENTS[name],
            output_root,
            args.mode,
            args.verify_hash,
            args.overwrite,
        )
        for name in names
    ]
    summary_path = output_root / args.mode / "view_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"mode": args.mode, "experiments": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"mode": args.mode, "experiments": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
