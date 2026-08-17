#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Sampler, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.datasets import (
    ContinuousFluorescenceTargetDataset,
    make_balanced_weights,
)
from differentiation_prediction.yichao_fluorescence_segmentation.models import GlobalGatedSegUNet
from differentiation_prediction.yichao_fluorescence_segmentation.utils import (
    DEFAULT_OUTPUT_ROOT,
    gray_rgb,
    green_rgb,
    heat_rgb,
    read_csv,
    save_grid,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train B-to-continuous-suppressed-F prediction for Yichao data.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("analysis-outputs") / "yichao_fluorescence_continuous" / "runs" / "soft_suppressed_unet_v1")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--include-organoid-mask", action="store_true", default=True)
    parser.add_argument("--no-organoid-mask", action="store_false", dest="include_organoid_mask")
    parser.add_argument("--include-distance", action="store_true", default=True)
    parser.add_argument("--no-distance", action="store_false", dest="include_distance")
    parser.add_argument("--target-scale", type=float, default=2.5)
    parser.add_argument("--soft-mask-dilate", type=int, default=9)
    parser.add_argument("--soft-mask-sigma", type=float, default=3.5)
    parser.add_argument("--soft-mask-floor", type=float, default=0.35)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-mse", type=float, default=0.25)
    parser.add_argument("--lambda-soft-dice", type=float, default=0.50)
    parser.add_argument("--lambda-focal", type=float, default=1.0)
    parser.add_argument("--lambda-total", type=float, default=0.20)
    parser.add_argument("--lambda-bg", type=float, default=0.08)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--signal-weight", type=float, default=4.0)
    parser.add_argument("--signal-power", type=float, default=0.5)
    parser.add_argument("--focal-alpha", type=float, default=0.85)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--bg-threshold", type=float, default=0.02)
    parser.add_argument("--metric-threshold", type=float, default=0.20)
    parser.add_argument("--metric-support-threshold", type=float, default=0.08)
    parser.add_argument("--metric-foreground-weight", type=float, default=8.0)
    parser.add_argument("--metric-thresholds", type=str, default="0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--positive-sample-weight", type=float, default=0.0)
    parser.add_argument("--samples-per-epoch", type=int, default=None, help="Optional sampled training examples per epoch for very large manifests.")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--panel-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-periodic", type=int, default=8)
    parser.add_argument("--early-stop", action="store_true", help="Deprecated. Accepted for old commands, but training no longer stops early.")
    parser.add_argument("--early-stop-patience-evals", type=int, default=25)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--early-stop-min-epochs", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--overfit-count", type=int, default=None)
    return parser.parse_args()


def make_loader(dataset: ContinuousFluorescenceTargetDataset, args: argparse.Namespace, shuffle: bool, sampler: Sampler | None = None) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def cosine_lr(epoch: int, args: argparse.Namespace) -> float:
    if args.epochs <= 1:
        return args.min_lr
    progress = min(max(epoch - 1, 0) / max(args.epochs - 1, 1), 1.0)
    return float(args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def optimizer_trainable_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    return [
        param
        for group in optimizer.param_groups
        for param in group.get("params", [])
        if getattr(param, "requires_grad", False)
    ]


def metric_thresholds(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    for raw_value in str(args.metric_thresholds).split(","):
        raw_value = raw_value.strip()
        if raw_value:
            values.append(float(raw_value))
    if not values:
        values = [args.metric_support_threshold, args.metric_threshold]
    values.extend([args.metric_support_threshold, args.metric_threshold])
    return sorted({round(max(0.0, min(1.0, value)), 6) for value in values})


def safe_pearson_from_sums(count: float, x_sum: float, y_sum: float, x_sq_sum: float, y_sq_sum: float, xy_sum: float) -> float:
    if count <= 1:
        return 0.0
    x_mean = x_sum / count
    y_mean = y_sum / count
    cov = xy_sum / count - x_mean * y_mean
    x_var = max(x_sq_sum / count - x_mean * x_mean, 0.0)
    y_var = max(y_sq_sum / count - y_mean * y_mean, 0.0)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return 0.0
    return float(max(-1.0, min(1.0, cov / math.sqrt(x_var * y_var))))


def continuous_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    pred = torch.sigmoid(outputs["logits"])
    target = batch["target"].to(pred.device)
    support = (target >= args.metric_threshold).float()
    weight = args.background_weight + args.signal_weight * torch.pow(target.clamp(0.0, 1.0), args.signal_power)
    smooth_l1 = (F.smooth_l1_loss(pred, target, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * weight).sum() / weight.sum().clamp_min(1.0)
    dice_num = 2.0 * (pred * support).sum(dim=(1, 2, 3)) + 1e-6
    dice_den = (pred + support).sum(dim=(1, 2, 3)) + 1e-6
    soft_dice = (1.0 - dice_num / dice_den).mean()
    bce = F.binary_cross_entropy_with_logits(logits, support, reduction="none")
    prob_for_focal = pred * support + (1.0 - pred) * (1.0 - support)
    alpha = args.focal_alpha * support + (1.0 - args.focal_alpha) * (1.0 - support)
    focal = (alpha * torch.pow((1.0 - prob_for_focal).clamp_min(1e-6), args.focal_gamma) * bce).mean()
    pred_mean = pred.mean(dim=(1, 2, 3))
    target_mean = target.mean(dim=(1, 2, 3))
    total = F.smooth_l1_loss(torch.log1p(pred_mean * 1000.0), torch.log1p(target_mean * 1000.0))
    background = target <= args.bg_threshold
    bg_loss = pred[background].mean() if bool(background.any()) else pred.mean() * 0.0
    loss = (
        args.lambda_l1 * smooth_l1
        + args.lambda_mse * mse
        + args.lambda_soft_dice * soft_dice
        + args.lambda_focal * focal
        + args.lambda_total * total
        + args.lambda_bg * bg_loss
    )
    return loss, {
        "loss_l1": float(smooth_l1.detach().cpu()),
        "loss_mse": float(mse.detach().cpu()),
        "loss_soft_dice": float(soft_dice.detach().cpu()),
        "loss_focal": float(focal.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
        "loss_bg": float(bg_loss.detach().cpu()),
    }


def save_panel(batch: dict[str, Any], pred: torch.Tensor, path: Path, max_items: int = 6) -> None:
    rows = []
    labels = []
    count = min(max_items, pred.shape[0])
    for idx in range(count):
        bf = batch["brightfield"][idx].squeeze(0).detach().cpu().numpy()
        fl = batch["fluorescence"][idx].squeeze(0).detach().cpu().numpy()
        target = batch["target"][idx].squeeze(0).detach().cpu().numpy()
        output = pred[idx].squeeze(0).detach().cpu().numpy()
        error = abs(output - target)
        rows.append([gray_rgb(bf), green_rgb(fl), gray_rgb(target), gray_rgb(output), heat_rgb(error)])
        labels.append(f"{batch['dataset'][idx]} | {batch['target_status'][idx]} | {str(batch['instance_id'][idx])[:90]}")
    save_grid(path, rows, ["B", "raw F", "target y", "pred y", "abs error"], labels, tile=180)


@torch.no_grad()
def evaluate(
    model: GlobalGatedSegUNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    panel_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    thresholds = metric_thresholds(args)
    loss_sum = 0.0
    seen = 0
    pixel_count = 0.0
    l1_sum = 0.0
    mse_sum = 0.0
    weighted_l1_sum = 0.0
    weighted_mse_sum = 0.0
    metric_weight_sum = 0.0
    positive_l1_sum = 0.0
    positive_mse_sum = 0.0
    positive_count = 0.0
    background_energy_sum = 0.0
    background_count = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    pred_sq_sum = 0.0
    target_sq_sum = 0.0
    xy_sum = 0.0
    pos_pred_sum = 0.0
    pos_target_sum = 0.0
    pos_pred_sq_sum = 0.0
    pos_target_sq_sum = 0.0
    pos_xy_sum = 0.0
    soft_support_intersection = 0.0
    soft_support_pred_sum = 0.0
    soft_support_target_sum = 0.0
    continuous_intersection = 0.0
    fixed_tp = fixed_fp = fixed_fn = 0.0
    threshold_counts = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for threshold in thresholds}
    total_log_pred_sum = 0.0
    total_log_target_sum = 0.0
    total_log_pred_sq_sum = 0.0
    total_log_target_sq_sum = 0.0
    total_log_xy_sum = 0.0
    total_log_l1_sum = 0.0
    total_seen = 0.0
    first_panel = True
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        outputs = model(image)
        loss, _ = continuous_loss(outputs, batch, args)
        pred = torch.sigmoid(outputs["logits"]).detach()
        target = batch["target"].to(device, non_blocking=True)
        target_support = target >= args.metric_support_threshold
        diff = pred - target
        abs_diff = diff.abs()
        sq_diff = diff.square()
        metric_weight = 1.0 + args.metric_foreground_weight * target.clamp(0.0, 1.0)
        batch_pixels = float(target.numel())
        pixel_count += batch_pixels
        l1_sum += float(abs_diff.sum().detach().cpu())
        mse_sum += float(sq_diff.sum().detach().cpu())
        weighted_l1_sum += float((abs_diff * metric_weight).sum().detach().cpu())
        weighted_mse_sum += float((sq_diff * metric_weight).sum().detach().cpu())
        metric_weight_sum += float(metric_weight.sum().detach().cpu())
        pred_sum += float(pred.sum().detach().cpu())
        target_sum += float(target.sum().detach().cpu())
        pred_sq_sum += float(pred.square().sum().detach().cpu())
        target_sq_sum += float(target.square().sum().detach().cpu())
        xy_sum += float((pred * target).sum().detach().cpu())
        support_float = target_support.float()
        positive_pixels = float(target_support.sum().detach().cpu())
        positive_count += positive_pixels
        if positive_pixels > 0:
            positive_abs = abs_diff[target_support]
            positive_sq = sq_diff[target_support]
            positive_pred = pred[target_support]
            positive_target = target[target_support]
            positive_l1_sum += float(positive_abs.sum().detach().cpu())
            positive_mse_sum += float(positive_sq.sum().detach().cpu())
            pos_pred_sum += float(positive_pred.sum().detach().cpu())
            pos_target_sum += float(positive_target.sum().detach().cpu())
            pos_pred_sq_sum += float(positive_pred.square().sum().detach().cpu())
            pos_target_sq_sum += float(positive_target.square().sum().detach().cpu())
            pos_xy_sum += float((positive_pred * positive_target).sum().detach().cpu())
        background = ~target_support
        background_pixels = float(background.sum().detach().cpu())
        background_count += background_pixels
        if background_pixels > 0:
            background_energy_sum += float(pred[background].square().sum().detach().cpu())
        soft_support_intersection += float((pred * support_float).sum().detach().cpu())
        soft_support_pred_sum += float(pred.sum().detach().cpu())
        soft_support_target_sum += float(support_float.sum().detach().cpu())
        continuous_intersection += float((pred * target).sum().detach().cpu())
        pred_binary_fixed = pred >= args.metric_threshold
        fixed_tp += float((pred_binary_fixed & target_support).sum().detach().cpu())
        fixed_fp += float((pred_binary_fixed & ~target_support).sum().detach().cpu())
        fixed_fn += float((~pred_binary_fixed & target_support).sum().detach().cpu())
        for threshold in thresholds:
            pred_binary = pred >= threshold
            threshold_counts[threshold]["tp"] += float((pred_binary & target_support).sum().detach().cpu())
            threshold_counts[threshold]["fp"] += float((pred_binary & ~target_support).sum().detach().cpu())
            threshold_counts[threshold]["fn"] += float((~pred_binary & target_support).sum().detach().cpu())
        pred_total = pred.sum(dim=(1, 2, 3))
        target_total = target.sum(dim=(1, 2, 3))
        pred_log = torch.log1p(pred_total)
        target_log = torch.log1p(target_total)
        total_seen += float(pred_log.numel())
        total_log_pred_sum += float(pred_log.sum().detach().cpu())
        total_log_target_sum += float(target_log.sum().detach().cpu())
        total_log_pred_sq_sum += float(pred_log.square().sum().detach().cpu())
        total_log_target_sq_sum += float(target_log.square().sum().detach().cpu())
        total_log_xy_sum += float((pred_log * target_log).sum().detach().cpu())
        total_log_l1_sum += float((pred_log - target_log).abs().sum().detach().cpu())
        batch_size = image.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        if panel_path is not None and first_panel:
            save_panel(batch, pred.cpu(), panel_path)
            first_panel = False
        if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
            break
    pred_mean = pred_sum / max(pixel_count, 1.0)
    target_mean = target_sum / max(pixel_count, 1.0)
    pearson = safe_pearson_from_sums(pixel_count, pred_sum, target_sum, pred_sq_sum, target_sq_sum, xy_sum)
    positive_pearson = safe_pearson_from_sums(positive_count, pos_pred_sum, pos_target_sum, pos_pred_sq_sum, pos_target_sq_sum, pos_xy_sum)
    total_intensity_pearson = safe_pearson_from_sums(total_seen, total_log_pred_sum, total_log_target_sum, total_log_pred_sq_sum, total_log_target_sq_sum, total_log_xy_sum)
    precision = fixed_tp / max(fixed_tp + fixed_fp, 1.0)
    recall = fixed_tp / max(fixed_tp + fixed_fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    sweep_rows: list[dict[str, float]] = []
    for threshold, counts in threshold_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        sweep_precision = tp / max(tp + fp, 1.0)
        sweep_recall = tp / max(tp + fn, 1.0)
        sweep_f1 = 2.0 * sweep_precision * sweep_recall / max(sweep_precision + sweep_recall, 1e-12)
        beta05 = 0.5
        beta2 = 2.0
        sweep_f05 = (1.0 + beta05 * beta05) * sweep_precision * sweep_recall / max(beta05 * beta05 * sweep_precision + sweep_recall, 1e-12)
        sweep_f2 = (1.0 + beta2 * beta2) * sweep_precision * sweep_recall / max(beta2 * beta2 * sweep_precision + sweep_recall, 1e-12)
        sweep_rows.append(
            {
                "threshold": threshold,
                "precision": sweep_precision,
                "recall": sweep_recall,
                "f1": sweep_f1,
                "f05": sweep_f05,
                "f2": sweep_f2,
            }
        )
    best_f1_row = max(sweep_rows, key=lambda row: row["f1"])
    best_f05_row = max(sweep_rows, key=lambda row: row["f05"])
    best_f2_row = max(sweep_rows, key=lambda row: row["f2"])
    pr_points = sorted((row["recall"], row["precision"]) for row in sweep_rows)
    support_auprc = 0.0
    previous_recall = 0.0
    for point_recall, point_precision in pr_points:
        recall_delta = max(point_recall - previous_recall, 0.0)
        support_auprc += recall_delta * point_precision
        previous_recall = max(previous_recall, point_recall)
    soft_support_dice = (2.0 * soft_support_intersection + 1e-6) / max(soft_support_pred_sum + soft_support_target_sum + 1e-6, 1e-6)
    soft_support_iou = (soft_support_intersection + 1e-6) / max(soft_support_pred_sum + soft_support_target_sum - soft_support_intersection + 1e-6, 1e-6)
    continuous_dice = (2.0 * continuous_intersection + 1e-6) / max(pred_sum + target_sum + 1e-6, 1e-6)
    result = {
        "loss": loss_sum / max(seen, 1),
        "mae": l1_sum / max(pixel_count, 1.0),
        "rmse": math.sqrt(mse_sum / max(pixel_count, 1.0)),
        "foreground_weighted_mae": weighted_l1_sum / max(metric_weight_sum, 1.0),
        "foreground_weighted_rmse": math.sqrt(weighted_mse_sum / max(metric_weight_sum, 1.0)),
        "positive_mae": positive_l1_sum / max(positive_count, 1.0),
        "positive_rmse": math.sqrt(positive_mse_sum / max(positive_count, 1.0)),
        "background_false_energy": background_energy_sum / max(background_count, 1.0),
        "pearson": pearson,
        "positive_pearson": positive_pearson,
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "total_intensity_ratio": pred_sum / max(target_sum, 1e-12),
        "total_intensity_log_mae": total_log_l1_sum / max(total_seen, 1.0),
        "total_intensity_log_pearson": total_intensity_pearson,
        "support_prevalence": positive_count / max(pixel_count, 1.0),
        "support_soft_dice": soft_support_dice,
        "support_soft_iou": soft_support_iou,
        "continuous_soft_dice": continuous_dice,
        "support_auprc": support_auprc,
        "support_best_f1": best_f1_row["f1"],
        "support_best_f1_threshold": best_f1_row["threshold"],
        "support_best_f05": best_f05_row["f05"],
        "support_best_f05_threshold": best_f05_row["threshold"],
        "support_best_f2": best_f2_row["f2"],
        "support_best_f2_threshold": best_f2_row["threshold"],
        "threshold_precision": precision,
        "threshold_recall": recall,
        "threshold_f1": f1,
        "n": seen,
    }
    for threshold_row in sweep_rows:
        token = str(threshold_row["threshold"]).replace(".", "p")
        result[f"support_precision_at_{token}"] = threshold_row["precision"]
        result[f"support_recall_at_{token}"] = threshold_row["recall"]
        result[f"support_f1_at_{token}"] = threshold_row["f1"]
    return result


def selection_score(metrics: dict[str, float]) -> float:
    intensity_penalty = min(abs(math.log(max(metrics.get("total_intensity_ratio", 1.0), 1e-6))), 2.0) / 2.0
    background_penalty = min(metrics.get("background_false_energy", 0.0) / 0.02, 1.0)
    log_intensity_penalty = min(metrics.get("total_intensity_log_mae", 0.0) / 1.0, 1.0)
    return float(
        0.25 * metrics.get("support_auprc", 0.0)
        + 0.20 * metrics.get("support_best_f1", 0.0)
        + 0.20 * metrics.get("continuous_soft_dice", 0.0)
        + 0.20 * metrics.get("total_intensity_log_pearson", 0.0)
        + 0.15 * metrics.get("positive_pearson", 0.0)
        - 0.05 * background_penalty
        - 0.05 * intensity_penalty
        - 0.05 * log_intensity_penalty
    )


def plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    val_rows = [row for row in rows if "val_loss" in row]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, [row.get("train_loss", math.nan) for row in rows], label="train loss")
    if val_rows:
        val_epochs = [int(row["epoch"]) for row in val_rows]
        axes[0, 0].plot(val_epochs, [row.get("val_loss", math.nan) for row in val_rows], "o-", label="val loss")
        axes[0, 1].plot(val_epochs, [row.get("val_selection_score", math.nan) for row in val_rows], "o-", label="selection")
        axes[0, 1].plot(val_epochs, [row.get("val_support_auprc", math.nan) for row in val_rows], "o-", label="support AUPRC")
        axes[0, 1].plot(val_epochs, [row.get("val_support_best_f1", math.nan) for row in val_rows], "o-", label="best F1")
        axes[1, 0].plot(val_epochs, [row.get("val_mae", math.nan) for row in val_rows], "o-", label="MAE")
        axes[1, 0].plot(val_epochs, [row.get("val_foreground_weighted_mae", math.nan) for row in val_rows], "o-", label="fg-weighted MAE")
        axes[1, 0].plot(val_epochs, [row.get("val_positive_mae", math.nan) for row in val_rows], "o-", label="positive MAE")
        axes[1, 1].plot(val_epochs, [row.get("val_continuous_soft_dice", math.nan) for row in val_rows], "o-", label="continuous Dice")
        axes[1, 1].plot(val_epochs, [row.get("val_total_intensity_log_pearson", math.nan) for row in val_rows], "o-", label="total expr corr")
        axes[1, 1].plot(val_epochs, [row.get("val_background_false_energy", math.nan) for row in val_rows], "o-", label="bg false energy")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlabel("epoch")
    axes[0, 0].set_title("Loss")
    axes[0, 1].set_title("Biology-aligned selection")
    axes[1, 0].set_title("Signal-weighted regression")
    axes[1, 1].set_title("Signal support and calibration")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_periodic(path: Path, payload: dict[str, Any], keep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    checkpoints = sorted(path.parent.glob("epoch_*.pt"))
    if keep > 0 and len(checkpoints) > keep:
        for old in checkpoints[: len(checkpoints) - keep]:
            old.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_root / "plots").mkdir(parents=True, exist_ok=True)
    (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    manifest_path = args.target_root / "manifests" / "segmentation_targets_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing target manifest: {manifest_path}")
    rows = read_csv(manifest_path)
    if args.overfit_count is not None:
        train_rows = [row for row in rows if row["split"] == "train"][: args.overfit_count]
        val_rows = train_rows
        test_rows = train_rows
    else:
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        test_rows = [row for row in rows if row["split"] == "test"]
    dataset_kwargs = {
        "image_size": args.image_size,
        "include_organoid_mask": args.include_organoid_mask,
        "include_distance": args.include_distance,
        "target_scale": args.target_scale,
        "soft_mask_dilate": args.soft_mask_dilate,
        "soft_mask_sigma": args.soft_mask_sigma,
        "soft_mask_floor": args.soft_mask_floor,
    }
    train_ds = ContinuousFluorescenceTargetDataset(train_rows, augment=args.overfit_count is None, **dataset_kwargs)
    val_ds = ContinuousFluorescenceTargetDataset(val_rows, **dataset_kwargs)
    test_ds = ContinuousFluorescenceTargetDataset(test_rows, **dataset_kwargs)
    sampler = None
    if args.balanced_sampler:
        weights = make_balanced_weights(train_rows, args.positive_sample_weight)
        num_samples = len(weights) if args.samples_per_epoch is None else max(1, int(args.samples_per_epoch))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    elif args.samples_per_epoch is not None:
        sampler = RandomSampler(train_ds, replacement=True, num_samples=max(1, int(args.samples_per_epoch)))
    train_loader = make_loader(train_ds, args, shuffle=True, sampler=sampler)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    in_channels = 1 + int(args.include_organoid_mask) + int(args.include_distance)
    model = GlobalGatedSegUNet(in_channels=in_channels, base_channels=args.base_channels, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grad_clip_params = optimizer_trainable_parameters(optimizer)
    metrics: list[dict[str, Any]] = []
    best_score = -1e18
    best_epoch = 0
    start_epoch = 1
    last_path = args.output_root / "last_model.pt"
    metrics_path = args.output_root / "metrics.jsonl"
    write_json(
        args.output_root / "run_config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "test_count": len(test_ds),
            "input_channels": in_channels,
            "target": "one_channel_soft_suppressed_fluorescence_clipped_0_1",
        },
    )
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        metrics = list(checkpoint.get("metrics", []))
        best_score = float(checkpoint.get("best_score", best_score))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        with metrics_path.open("w", encoding="utf-8") as handle:
            for row in metrics:
                handle.write(json.dumps(row) + "\n")
    elif metrics_path.exists():
        metrics_path.unlink()
    stale_evals = 0
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        set_lr(optimizer, cosine_lr(epoch, args))
        model.train()
        loss_sum = 0.0
        seen = 0
        component_sums: dict[str, float] = {}
        grad_norm_value = 0.0
        for batch_index, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            outputs = model(image)
            loss, components = continuous_loss(outputs, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(grad_clip_params, 3.0, foreach=False)
            optimizer.step()
            grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            batch_size = image.shape[0]
            loss_sum += float(loss.detach().cpu()) * batch_size
            seen += batch_size
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value * batch_size
            if args.limit_train_batches is not None and batch_index + 1 >= args.limit_train_batches:
                break
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "lr": cosine_lr(epoch, args),
            "grad_norm": grad_norm_value,
        }
        row.update({f"train_{key}": value / max(seen, 1) for key, value in component_sums.items()})
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            panel_path = args.output_root / "predictions" / f"val_epoch_{epoch:04d}.png" if (epoch == 1 or epoch % args.panel_every == 0) else None
            val_metrics = evaluate(model, val_loader, device, args, panel_path=panel_path)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = selection_score(val_metrics)
            row["val_selection_score"] = score
            if score > best_score + args.early_stop_min_delta:
                best_score = score
                best_epoch = epoch
                stale_evals = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "score": best_score, "metrics": row}, args.output_root / "best_model.pt")
                if panel_path is not None:
                    shutil.copy2(panel_path, args.output_root / "predictions" / "val_best.png")
            else:
                stale_evals += 1
            row["early_stop_stale_evals"] = stale_evals
            if args.early_stop and epoch >= args.early_stop_min_epochs and stale_evals >= args.early_stop_patience_evals:
                row["early_stop_would_have_triggered"] = True
                row["early_stop_disabled"] = True
        metrics.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics, "best_score": best_score, "best_epoch": best_epoch}
        torch.save(payload, last_path)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_periodic(args.output_root / "checkpoints" / f"epoch_{epoch:04d}.pt", payload, args.keep_periodic)
        plot_metrics(metrics, args.output_root / "plots" / "training_metrics.png")
        print(json.dumps(row), flush=True)
    write_json(args.output_root / "early_stop_summary.json", {"triggered": False, "disabled": True, "last_epoch": metrics[-1]["epoch"], "best_epoch": best_epoch, "best_score": best_score, "stale_evals": stale_evals})
    if not (args.output_root / "best_model.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": metrics[-1]["epoch"], "score": best_score, "metrics": metrics[-1]}, args.output_root / "best_model.pt")
    best = torch.load(args.output_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device, args, panel_path=args.output_root / "predictions" / "test_best.png")
    write_json(args.output_root / "test_metrics.json", {**test_metrics, "best_epoch": int(best.get("epoch", 0)), "stopped_early": stopped_early})
    print(json.dumps({"stage": "continuous_target_finished", "test": test_metrics, "best_epoch": int(best.get("epoch", 0)), "stopped_early": stopped_early}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
