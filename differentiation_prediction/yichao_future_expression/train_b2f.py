#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.datasets import B2FDataset
from differentiation_prediction.yichao_future_expression.models import B2FMultiTaskUNet
from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    average_precision,
    binary_auc,
    pearson_corr,
    plot_metric_lines,
    read_csv,
    save_b2f_panel,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train same-time projected brightfield-to-fluorescence feasibility model.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1_b2f")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    return parser.parse_args()


def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(dataset: B2FDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inside = torch.abs(pred - target) * mask
    outside = torch.abs(pred - target) * (1.0 - mask)
    inside_loss = inside.sum() / mask.sum().clamp_min(1.0)
    outside_loss = outside.sum() / (1.0 - mask).sum().clamp_min(1.0)
    return inside_loss + 0.20 * outside_loss


def scalar_losses(outputs: torch.Tensor, batch: dict[str, torch.Tensor], bce: nn.Module, huber: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    pos_logit = outputs[:, 0]
    peak_pred = outputs[:, 1]
    total_pred = outputs[:, 2]
    pos = batch["positive"].to(outputs.device)
    peak = batch["peak_log"].to(outputs.device)
    total = batch["total_log"].to(outputs.device)
    loss_pos = bce(pos_logit, pos)
    loss_peak = huber(peak_pred, peak)
    loss_total = huber(total_pred, total)
    return loss_pos + 0.20 * loss_peak + 0.05 * loss_total, {
        "loss_pos": float(loss_pos.detach().cpu()),
        "loss_peak": float(loss_peak.detach().cpu()),
        "loss_total": float(loss_total.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: B2FMultiTaskUNet, loader: DataLoader, device: torch.device, args: argparse.Namespace, panel_path: Path | None = None) -> dict[str, float]:
    model.eval()
    image_mae_sum = 0.0
    image_mse_sum = 0.0
    count = 0
    labels: list[float] = []
    scores: list[float] = []
    target_peak: list[float] = []
    pred_peak: list[float] = []
    first_panel = True
    for batch_index, batch in enumerate(loader):
        brightfield = batch["brightfield"].to(device, non_blocking=True)
        fluorescence = batch["fluorescence"].to(device, non_blocking=True)
        pred, scalar = model(brightfield)
        batch_size = brightfield.shape[0]
        image_mae_sum += torch.abs(pred - fluorescence).mean().item() * batch_size
        image_mse_sum += torch.mean((pred - fluorescence) ** 2).item() * batch_size
        count += batch_size
        labels.extend(batch["positive"].numpy().tolist())
        scores.extend(torch.sigmoid(scalar[:, 0]).detach().cpu().numpy().tolist())
        target_peak.extend(batch["peak_log"].numpy().tolist())
        pred_peak.extend(scalar[:, 1].detach().cpu().numpy().tolist())
        if panel_path is not None and first_panel:
            save_b2f_panel(
                brightfield.detach().cpu(),
                pred.detach().cpu(),
                fluorescence.detach().cpu(),
                panel_path,
                keys=list(batch["instance_id"]),
            )
            first_panel = False
        if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
            break
    mse = image_mse_sum / max(count, 1)
    return {
        "image_mae": image_mae_sum / max(count, 1),
        "image_mse": mse,
        "image_psnr": float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-12)))),
        "positive_auc": binary_auc(labels, scores),
        "positive_ap": average_precision(labels, scores),
        "peak_pearson": pearson_corr(target_peak, pred_peak),
        "n": count,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_root / "manifests" / "projected_instances_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}. Run build_projected_dataset.py first.")

    train_ds = B2FDataset.from_manifest(manifest_path, "train", args.image_size, augment=True)
    val_ds = B2FDataset.from_manifest(manifest_path, "val", args.image_size)
    test_ds = B2FDataset.from_manifest(manifest_path, "test", args.image_size)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)

    train_rows = [row for row in read_csv(manifest_path) if row["split"] == "train"]
    pos_count = sum(int(row["fluorescence_positive"]) for row in train_rows)
    neg_count = max(1, len(train_rows) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device, dtype=torch.float32)

    model = B2FMultiTaskUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = grad_scaler(use_amp)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber = nn.SmoothL1Loss()
    metrics: list[dict[str, Any]] = []
    best_score = -1e9
    start_epoch = 1

    write_json(
        args.output_root / "run_config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "test_count": len(test_ds),
            "train_positive_count": pos_count,
        },
    )
    metrics_path = args.output_root / "metrics.jsonl"
    last_checkpoint_path = args.output_root / "last_model.pt"
    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        metrics = list(checkpoint.get("metrics_log", []))
        best_score = float(checkpoint.get("best_score", best_score))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        with metrics_path.open("w", encoding="utf-8") as handle:
            for row in metrics:
                handle.write(json.dumps(row) + "\n")
    elif metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            brightfield = batch["brightfield"].to(device, non_blocking=True)
            fluorescence = batch["fluorescence"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                pred, scalar = model(brightfield)
                image_loss = masked_l1(pred, fluorescence, mask)
                scalar_loss, _ = scalar_losses(scalar, {k: v for k, v in batch.items() if torch.is_tensor(v)}, bce, huber)
                loss = image_loss + 0.25 * scalar_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * brightfield.shape[0]
            seen += brightfield.shape[0]
            if args.limit_train_batches is not None and batch_index + 1 >= args.limit_train_batches:
                break
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, args, args.output_root / "predictions" / f"val_epoch_{epoch:03d}.png")
        row = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "lr": scheduler.get_last_lr()[0], **{f"val_{k}": v for k, v in val_metrics.items()}}
        metrics.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        score = float(val_metrics.get("positive_auc", float("nan")))
        if not np.isfinite(score):
            score = -float(val_metrics["image_mae"])
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, args.output_root / "best_model.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "metrics_log": metrics,
                "best_score": best_score,
            },
            last_checkpoint_path,
        )
        print(json.dumps(row), flush=True)

    best = torch.load(args.output_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device, args, args.output_root / "predictions" / "test_best.png")
    write_json(args.output_root / "test_metrics.json", test_metrics)
    plot_metric_lines(metrics, args.output_root / "plots" / "training_metrics.png", ["train_loss", "val_image_mae", "val_positive_auc", "val_peak_pearson"])
    print(json.dumps({"stage": "b2f_finished", "test": test_metrics, "best_epoch": best["epoch"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
