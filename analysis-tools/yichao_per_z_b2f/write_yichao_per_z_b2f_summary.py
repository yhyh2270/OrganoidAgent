#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact summary for the Yichao per-z B2F run.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def metric_lines(metrics: dict[str, Any], prefix: str = "") -> list[str]:
    keys = [
        "loss",
        "mae",
        "foreground_weighted_mae",
        "positive_mae",
        "pearson",
        "positive_pearson",
        "continuous_soft_dice",
        "support_auprc",
        "support_best_f1",
        "threshold_f1",
        "total_intensity_ratio",
        "total_intensity_log_pearson",
        "background_false_energy",
        "rows",
        "checkpoint_epoch",
    ]
    lines: list[str] = []
    for key in keys:
        full_key = f"{prefix}{key}"
        if full_key in metrics:
            value = metrics[full_key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: {value:.6g}")
            else:
                lines.append(f"- `{key}`: {value}")
    return lines


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    instance_summary = read_json(args.instance_root / "manifests" / "summary.json")
    train_manifest_summary = read_json(args.train_manifest.with_suffix(".summary.json"))
    target_summary = read_json(args.target_root / "manifests" / "segmentation_targets_summary.json")
    last_epoch = read_last_jsonl(args.run_root / "metrics.jsonl")
    test_metrics = read_json(args.run_root / "test_metrics.json")
    external_metrics = read_json(args.external_root / "evaluation" / "Data-Yichao-11_per_z_last_epoch_external_metrics.json")
    external_target_summary = read_json(args.external_root / "fluorescence_segmentation_relaxed_targets" / "manifests" / "segmentation_targets_summary.json")

    payload = {
        "instance_summary": instance_summary,
        "train_manifest_summary": train_manifest_summary,
        "target_summary": target_summary,
        "last_epoch": last_epoch,
        "test_metrics": test_metrics,
        "external_metrics": external_metrics,
        "external_target_summary": external_target_summary,
        "paths": {
            "instance_root": str(args.instance_root),
            "train_manifest": str(args.train_manifest),
            "target_root": str(args.target_root),
            "run_root": str(args.run_root),
            "external_root": str(args.external_root),
            "training_plot": str(args.run_root / "plots" / "training_metrics.png"),
            "test_panel": str(args.run_root / "predictions" / "test_best.png"),
            "internal_gallery": str(args.output_root / "visualizations" / "per_z_internal_test_gallery.png"),
            "external_panel": str(args.external_root / "evaluation" / "Data-Yichao-11_per_z_last_epoch_external_panel.png"),
            "external_gallery": str(args.output_root / "visualizations" / "per_z_Data-Yichao-11_external_gallery.png"),
        },
    }
    summary_json = args.output_root / "summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Yichao Per-Z Original-Plane B2F Run",
        "",
        "Data policy: every original z-plane B/F pair is a separate training or evaluation example. No z projection is used.",
        "Training policy: train/val/test uses `Data-Yichao-1` through `Data-Yichao-10`; `Data-Yichao-11` is evaluated only as an external holdout.",
        "",
        "## Paths",
        f"- Per-z instance root: `{args.instance_root}`",
        f"- Training manifest: `{args.train_manifest}`",
        f"- Clean target root: `{args.target_root}`",
        f"- Model run root: `{args.run_root}`",
        f"- Data-Yichao-11 external root: `{args.external_root}`",
        "",
        "## Dataset Counts",
        f"- Per-z source image records: `{instance_summary.get('image_count', 'pending')}`",
        f"- Per-z instance records: `{instance_summary.get('instance_count', 'pending')}`",
        f"- Training manifest rows: `{train_manifest_summary.get('count', 'pending')}`",
        f"- Training split counts: `{train_manifest_summary.get('split_counts', 'pending')}`",
        f"- Target status counts: `{target_summary.get('statuses', 'pending')}`",
        f"- External Data-Yichao-11 target rows: `{external_target_summary.get('count', 'pending')}`",
        "",
        "## Last Epoch Validation",
    ]
    lines.extend(metric_lines(last_epoch, "val_") or ["- Pending"])
    lines.extend(["", "## Internal Test"])
    lines.extend(metric_lines(test_metrics) or ["- Pending"])
    lines.extend(["", "## External Data-Yichao-11 Test"])
    lines.extend(metric_lines(external_metrics) or ["- Pending"])
    lines.extend(
        [
            "",
            "## Visual Outputs",
            f"- Training curves: `{args.run_root / 'plots' / 'training_metrics.png'}`",
            f"- Internal test panel: `{args.run_root / 'predictions' / 'test_best.png'}`",
            f"- Internal random gallery: `{args.output_root / 'visualizations' / 'per_z_internal_test_gallery.png'}`",
            f"- External Data-Yichao-11 panel: `{args.external_root / 'evaluation' / 'Data-Yichao-11_per_z_last_epoch_external_panel.png'}`",
            f"- External Data-Yichao-11 random gallery: `{args.output_root / 'visualizations' / 'per_z_Data-Yichao-11_external_gallery.png'}`",
        ]
    )
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(args.output_root / "summary.md")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
