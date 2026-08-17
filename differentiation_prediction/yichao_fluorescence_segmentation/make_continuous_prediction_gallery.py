#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
from differentiation_prediction.yichao_fluorescence_segmentation.utils import gray_rgb, green_rgb, heat_rgb, read_csv, save_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a random B/F/target/pred/error gallery for a continuous B2F checkpoint.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--tile", type=int, default=160)
    return parser.parse_args()


def load_run_args(run_root: Path) -> dict[str, Any]:
    config_path = run_root / "run_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing run config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))["args"]


def namespace_from_run_args(run_args: dict[str, Any]) -> Namespace:
    args = dict(run_args)
    args.setdefault("image_size", 256)
    args.setdefault("include_organoid_mask", True)
    args.setdefault("include_distance", True)
    args.setdefault("target_scale", 2.5)
    args.setdefault("soft_mask_dilate", 9)
    args.setdefault("soft_mask_sigma", 3.5)
    args.setdefault("soft_mask_floor", 0.35)
    args.setdefault("base_channels", 32)
    args.setdefault("dropout", 0.05)
    return Namespace(**args)


def main() -> int:
    cli = parse_args()
    run_args = namespace_from_run_args(load_run_args(cli.run_root))
    checkpoint_path = cli.checkpoint or (cli.run_root / "last_model.pt")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
    manifest_path = cli.target_root / "manifests" / "segmentation_targets_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing target manifest: {manifest_path}")

    rows = [row for row in read_csv(manifest_path) if row.get("split") == cli.split]
    if not rows:
        raise SystemExit(f"No rows for split={cli.split} in {manifest_path}")
    rng = random.Random(cli.seed)
    selected = rng.sample(rows, min(cli.sample_count, len(rows)))
    dataset = ContinuousFluorescenceTargetDataset(
        selected,
        int(run_args.image_size),
        augment=False,
        include_organoid_mask=bool(run_args.include_organoid_mask),
        include_distance=bool(run_args.include_distance),
        target_scale=float(run_args.target_scale),
        soft_mask_dilate=int(run_args.soft_mask_dilate),
        soft_mask_sigma=float(run_args.soft_mask_sigma),
        soft_mask_floor=float(run_args.soft_mask_floor),
    )
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 1 + int(bool(run_args.include_organoid_mask)) + int(bool(run_args.include_distance))
    model = GlobalGatedSegUNet(in_channels=in_channels, base_channels=int(run_args.base_channels), dropout=float(run_args.dropout)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_rows = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            pred = torch.sigmoid(model(image)["logits"]).detach().cpu()
            target = batch["target"].detach().cpu()
            for idx in range(pred.shape[0]):
                bf = batch["brightfield"][idx].squeeze(0).numpy()
                fl = batch["fluorescence"][idx].squeeze(0).numpy()
                tgt = target[idx].squeeze(0).numpy()
                out = pred[idx].squeeze(0).numpy()
                err = abs(out - tgt)
                image_rows.append([gray_rgb(bf), green_rgb(fl), gray_rgb(tgt), gray_rgb(out), heat_rgb(err)])
                labels.append(f"{batch['dataset'][idx]} | {batch['target_status'][idx]} | {str(batch['instance_id'][idx])[:100]}")
    save_grid(cli.output, image_rows, ["B", "raw F", "target y", "pred y", "abs error"], labels, tile=cli.tile)
    print(
        json.dumps(
            {
                "output": str(cli.output),
                "rows": len(image_rows),
                "split": cli.split,
                "target_manifest": str(manifest_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
