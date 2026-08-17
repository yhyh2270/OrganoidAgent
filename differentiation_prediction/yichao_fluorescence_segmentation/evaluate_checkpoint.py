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

from differentiation_prediction.yichao_fluorescence_segmentation.datasets import FluorescenceSegmentationDataset
from differentiation_prediction.yichao_fluorescence_segmentation.models import GlobalGatedSegUNet
from differentiation_prediction.yichao_fluorescence_segmentation.train_segmentation import evaluate
from differentiation_prediction.yichao_fluorescence_segmentation.utils import read_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Yichao fluorescence segmentation checkpoint.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
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


def namespace_from_run_args(run_args: dict[str, Any], overrides: argparse.Namespace) -> Namespace:
    args = dict(run_args)
    if overrides.target_root is not None:
        args["target_root"] = str(overrides.target_root)
    if overrides.batch_size is not None:
        args["batch_size"] = overrides.batch_size
    args["num_workers"] = overrides.num_workers
    args["limit_eval_batches"] = overrides.limit_eval_batches
    args.setdefault("outside_weight", 0.15)
    return Namespace(**args)


def main() -> int:
    cli = parse_args()
    run_args = load_run_args(cli.run_root)
    args = namespace_from_run_args(run_args, cli)
    checkpoint_path = cli.checkpoint or (cli.run_root / "best_model.pt")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
    target_root = Path(args.target_root)
    manifest_path = target_root / "manifests" / "segmentation_targets_manifest.csv"
    rows = [row for row in read_csv(manifest_path) if row.get("split") == cli.split]
    if not rows:
        raise SystemExit(f"No rows for split={cli.split} in {manifest_path}")

    dataset = FluorescenceSegmentationDataset(
        rows,
        int(args.image_size),
        augment=False,
        include_distance=bool(args.include_distance),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3 if bool(args.include_distance) else 2
    model = GlobalGatedSegUNet(
        in_channels=in_channels,
        base_channels=int(args.base_channels),
        dropout=float(args.dropout),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    panel_output = cli.panel_output or (cli.run_root / "predictions" / f"{cli.split}_best_checkpoint.png")
    metrics = evaluate(model, loader, device, args, panel_path=panel_output)
    payload = {
        **metrics,
        "split": cli.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_score": float(checkpoint.get("score", float("nan"))),
        "target_manifest": str(manifest_path),
        "panel_output": str(panel_output),
        "num_workers": int(args.num_workers),
    }
    output_json = cli.output_json or (cli.run_root / f"{cli.split}_metrics_best_checkpoint.json")
    write_json(output_json, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
