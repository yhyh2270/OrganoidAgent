#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.datasets import ContinuousFluorescenceTargetDataset
from differentiation_prediction.yichao_fluorescence_segmentation.models import GlobalGatedSegUNet
from differentiation_prediction.yichao_fluorescence_segmentation.train_continuous_target import evaluate
from differentiation_prediction.yichao_fluorescence_segmentation.utils import read_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a continuous B2F checkpoint on an external Yichao target manifest.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--split", default="external_test")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--panel-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    return parser.parse_args()


def load_run_args(run_root: Path) -> dict[str, Any]:
    config_path = run_root / "run_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing run config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))["args"]


def namespace_from_run_args(run_args: dict[str, Any], cli: argparse.Namespace) -> Namespace:
    args = dict(run_args)
    args["target_root"] = str(cli.target_root)
    args["num_workers"] = cli.num_workers
    args["limit_eval_batches"] = cli.limit_eval_batches
    if cli.batch_size is not None:
        args["batch_size"] = cli.batch_size
    # Older runs do not have these newer metric arguments in run_config.json.
    args.setdefault("metric_support_threshold", 0.08)
    args.setdefault("metric_foreground_weight", 8.0)
    args.setdefault("metric_thresholds", "0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30,0.40,0.50")
    return Namespace(**args)


def main() -> int:
    cli = parse_args()
    run_args = load_run_args(cli.run_root)
    args = namespace_from_run_args(run_args, cli)
    checkpoint_path = cli.checkpoint or (cli.run_root / "last_model.pt")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
    manifest_path = cli.target_root / "manifests" / "segmentation_targets_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing target manifest: {manifest_path}")
    rows = [row for row in read_csv(manifest_path) if row.get("split") == cli.split]
    if not rows:
        raise SystemExit(f"No rows for split={cli.split} in {manifest_path}")
    dataset = ContinuousFluorescenceTargetDataset(
        rows,
        int(args.image_size),
        augment=False,
        include_organoid_mask=bool(args.include_organoid_mask),
        include_distance=bool(args.include_distance),
        target_scale=float(args.target_scale),
        soft_mask_dilate=int(args.soft_mask_dilate),
        soft_mask_sigma=float(args.soft_mask_sigma),
        soft_mask_floor=float(args.soft_mask_floor),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    in_channels = 1 + int(bool(args.include_organoid_mask)) + int(bool(args.include_distance))
    model = GlobalGatedSegUNet(
        in_channels=in_channels,
        base_channels=int(args.base_channels),
        dropout=float(args.dropout),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    panel_output = cli.panel_output or (cli.target_root / "evaluation" / f"{cli.split}_continuous_checkpoint_panel.png")
    metrics = evaluate(model, loader, device, args, panel_path=panel_output)
    payload = {
        **metrics,
        "split": cli.split,
        "rows": len(rows),
        "run_root": str(cli.run_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_score": float(checkpoint.get("best_score", checkpoint.get("score", float("nan")))),
        "target_manifest": str(manifest_path),
        "panel_output": str(panel_output),
        "device": str(device),
        "external_test_only": True,
    }
    output_json = cli.output_json or (cli.target_root / "evaluation" / f"{cli.split}_continuous_checkpoint_metrics.json")
    write_json(output_json, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
