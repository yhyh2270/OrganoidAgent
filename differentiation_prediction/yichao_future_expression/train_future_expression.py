#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.datasets import (
    FEATURE_COLUMNS,
    FutureExpressionDataset,
    fit_feature_stats,
    fit_target_stats,
    load_future_data,
)
from differentiation_prediction.yichao_future_expression.models import FutureExpressionModel
from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    average_precision,
    binary_auc,
    pearson_corr,
    plot_metric_lines,
    set_seed,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train early-brightfield sequence model for future fluorescence expression.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage2_future_expression")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-prefix-frames", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
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


def make_loader(dataset: FutureExpressionDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
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


def build_datasets(args: argparse.Namespace) -> tuple[FutureExpressionDataset, FutureExpressionDataset, FutureExpressionDataset, dict[str, Any]]:
    manifest_path = args.data_root / "manifests" / "projected_instances_manifest.csv"
    future_path = args.data_root / "manifests" / "future_samples.csv"
    instance_rows, future_rows, instance_by_id = load_future_data(manifest_path, future_path)
    feature_mean, feature_std = fit_feature_stats(instance_rows, "train")
    peak_mean, peak_std, auc_mean, auc_std = fit_target_stats(future_rows, "train")

    def make(split: str, augment: bool) -> FutureExpressionDataset:
        rows = [row for row in future_rows if row["split"] == split]
        return FutureExpressionDataset(
            rows,
            instance_by_id,
            image_size=args.image_size,
            max_prefix_frames=args.max_prefix_frames,
            feature_mean=feature_mean,
            feature_std=feature_std,
            peak_mean=peak_mean,
            peak_std=peak_std,
            auc_mean=auc_mean,
            auc_std=auc_std,
            augment=augment,
        )

    stats = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "peak_mean": peak_mean,
        "peak_std": peak_std,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "future_sample_count": len(future_rows),
    }
    return make("train", True), make("val", False), make("test", False), stats


def loss_fn(outputs: torch.Tensor, batch: dict[str, torch.Tensor], bce: nn.Module, huber: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    positive = batch["future_positive"].to(outputs.device)
    peak = batch["future_peak"].to(outputs.device)
    auc = batch["future_auc"].to(outputs.device)
    loss_pos = bce(outputs[:, 0], positive)
    loss_peak = huber(outputs[:, 1], peak)
    loss_auc = huber(outputs[:, 2], auc)
    loss = loss_pos + 0.30 * loss_peak + 0.20 * loss_auc
    return loss, {
        "loss_pos": float(loss_pos.detach().cpu()),
        "loss_peak": float(loss_peak.detach().cpu()),
        "loss_auc": float(loss_auc.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: FutureExpressionModel, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    peak_true: list[float] = []
    peak_pred: list[float] = []
    auc_true: list[float] = []
    auc_pred: list[float] = []
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        outputs = model(frames, features, valid)
        prob = torch.sigmoid(outputs[:, 0]).detach().cpu().numpy()
        pred_peak = (outputs[:, 1].detach().cpu().numpy() * loader.dataset.peak_std) + loader.dataset.peak_mean
        pred_auc = (outputs[:, 2].detach().cpu().numpy() * loader.dataset.auc_std) + loader.dataset.auc_mean
        true_peak = batch["future_peak_raw"].numpy()
        true_auc = batch["future_auc_raw"].numpy()
        labels.extend(batch["future_positive"].numpy().tolist())
        scores.extend(prob.tolist())
        peak_true.extend(true_peak.tolist())
        peak_pred.extend(pred_peak.tolist())
        auc_true.extend(true_auc.tolist())
        auc_pred.extend(pred_auc.tolist())
        for i, sample_id in enumerate(batch["future_sample_id"]):
            rows.append(
                {
                    "future_sample_id": sample_id,
                    "track_id": batch["track_id"][i],
                    "dataset": batch["dataset"][i],
                    "future_positive": float(batch["future_positive"][i]),
                    "pred_future_positive_prob": float(prob[i]),
                    "future_peak_log": float(true_peak[i]),
                    "pred_future_peak_log": float(pred_peak[i]),
                    "future_auc_log": float(true_auc[i]),
                    "pred_future_auc_log": float(pred_auc[i]),
                }
            )
        if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
            break
    metrics = {
        "positive_auc": binary_auc(labels, scores),
        "positive_ap": average_precision(labels, scores),
        "peak_pearson": pearson_corr(peak_true, peak_pred),
        "auc_pearson": pearson_corr(auc_true, auc_pred),
        "peak_mae": float(np.mean(np.abs(np.asarray(peak_true) - np.asarray(peak_pred)))) if peak_true else float("nan"),
        "auc_mae": float(np.mean(np.abs(np.asarray(auc_true) - np.asarray(auc_pred)))) if auc_true else float("nan"),
        "n": len(labels),
    }
    return metrics, rows


def plot_predictions(rows: list[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    y = np.asarray([float(row["future_positive"]) for row in rows])
    score = np.asarray([float(row["pred_future_positive_prob"]) for row in rows])
    peak_true = np.asarray([float(row["future_peak_log"]) for row in rows])
    peak_pred = np.asarray([float(row["pred_future_peak_log"]) for row in rows])
    output_root.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(peak_true, peak_pred, s=8, alpha=0.45)
    plt.xlabel("true future peak log")
    plt.ylabel("pred future peak log")
    plt.tight_layout()
    plt.savefig(output_root / "future_peak_scatter.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.hist(score[y < 0.5], bins=30, alpha=0.6, label="future negative")
    plt.hist(score[y > 0.5], bins=30, alpha=0.6, label="future positive")
    plt.xlabel("predicted future-positive probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "future_positive_score_histogram.png", dpi=180)
    plt.close()


def permutation_importance(model: FutureExpressionModel, loader: DataLoader, device: torch.device, args: argparse.Namespace, baseline_auc: float) -> list[dict[str, Any]]:
    # Fast, validation-sized importance: replace one feature with zero after normalization.
    rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
        labels: list[float] = []
        scores: list[float] = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                frames = batch["frames"].to(device, non_blocking=True)
                features = batch["features"].clone()
                features[:, :, feature_index] = 0.0
                valid = batch["valid"].to(device, non_blocking=True)
                outputs = model(frames, features.to(device, non_blocking=True), valid)
                labels.extend(batch["future_positive"].numpy().tolist())
                scores.extend(torch.sigmoid(outputs[:, 0]).detach().cpu().numpy().tolist())
                if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
                    break
        auc = binary_auc(labels, scores)
        rows.append({"feature": feature_name, "baseline_auc": baseline_auc, "ablated_auc": auc, "auc_drop": baseline_auc - auc})
    return sorted(rows, key=lambda row: float(row["auc_drop"]), reverse=True)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_ds, val_ds, test_ds, stats = build_datasets(args)
    train_loader = make_loader(train_ds, args, True)
    val_loader = make_loader(val_ds, args, False)
    test_loader = make_loader(test_ds, args, False)
    pos_count = sum(float(row["future_positive"]) for row in train_ds.sample_rows)
    neg_count = max(1.0, len(train_ds) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device, dtype=torch.float32)
    model = FutureExpressionModel(feature_dim=len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = grad_scaler(use_amp)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber = nn.SmoothL1Loss()
    metrics_log: list[dict[str, Any]] = []
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
            **stats,
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
        metrics_log = list(checkpoint.get("metrics_log", []))
        best_score = float(checkpoint.get("best_score", best_score))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        with metrics_path.open("w", encoding="utf-8") as handle:
            for row in metrics_log:
                handle.write(json.dumps(row) + "\n")
    elif metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            frames = batch["frames"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                outputs = model(frames, features, valid)
                loss, _ = loss_fn(outputs, {k: v for k, v in batch.items() if torch.is_tensor(v)}, bce, huber)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * frames.shape[0]
            seen += frames.shape[0]
            if args.limit_train_batches is not None and batch_index + 1 >= args.limit_train_batches:
                break
        scheduler.step()
        val_metrics, _ = evaluate(model, val_loader, device, args)
        row = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "lr": scheduler.get_last_lr()[0], **{f"val_{k}": v for k, v in val_metrics.items()}}
        metrics_log.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        score = float(val_metrics.get("positive_auc", float("nan")))
        if not np.isfinite(score):
            score = -float(val_metrics.get("peak_mae", 1e9))
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "stats": stats}, args.output_root / "best_model.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "metrics_log": metrics_log,
                "best_score": best_score,
                "stats": stats,
            },
            last_checkpoint_path,
        )
        print(json.dumps(row), flush=True)

    best = torch.load(args.output_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, prediction_rows = evaluate(model, test_loader, device, args)
    importance = permutation_importance(model, test_loader, device, args, test_metrics["positive_auc"])
    write_json(args.output_root / "test_metrics.json", test_metrics)
    write_csv(args.output_root / "test_predictions.csv", prediction_rows)
    write_csv(args.output_root / "feature_ablation_importance.csv", importance)
    plot_predictions(prediction_rows, args.output_root / "plots")
    plot_metric_lines(metrics_log, args.output_root / "plots" / "training_metrics.png", ["train_loss", "val_positive_auc", "val_peak_pearson", "val_auc_pearson"])
    print(json.dumps({"stage": "future_expression_finished", "test": test_metrics, "best_epoch": best["epoch"], "top_features": importance[:5]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
