#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.datasets import B2FDataset
from differentiation_prediction.yichao_future_expression.models import Pix2PixB2FUNet, StrongB2FResUNet
from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    average_precision,
    binary_auc,
    image_to_tensor,
    pearson_corr,
    plot_metric_lines,
    read_csv,
    save_b2f_panel,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a stronger same-time brightfield-to-fluorescence model. "
            "This trainer is intended to make B2F work before future prediction is retried."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1_b2f_strong_384")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--path-mode", choices=("original_crop", "resized_256"), default="original_crop")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--architecture", choices=("strong_resunet", "pix2pix_unet"), default="strong_resunet")
    parser.add_argument("--signal-threshold", type=float, default=0.25)
    parser.add_argument("--signal-boost", type=float, default=12.0)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--bce-weight", type=float, default=0.08)
    parser.add_argument("--ssim-weight", type=float, default=0.05)
    parser.add_argument("--edge-weight", type=float, default=0.10)
    parser.add_argument("--scalar-weight", type=float, default=0.03)
    parser.add_argument("--max-pixel-pos-weight", type=float, default=20.0)
    parser.add_argument("--pixel-stat-limit", type=int, default=1800)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--positive-sample-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--panel-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--keep-periodic", type=int, default=8)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-metric", choices=("score", "loss", "masked_mae", "signal_f1", "peak_pearson"), default="score")
    parser.add_argument("--early-stop-patience-evals", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--early-stop-min-epochs", type=int, default=40)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260511)
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


def amp_scale_value(scaler: Any, enabled: bool) -> float | None:
    if not enabled:
        return None
    try:
        return float(scaler.get_scale())
    except (AttributeError, TypeError, ValueError):
        return None


def assert_healthy_amp_scale(scaler: Any, enabled: bool, context: str) -> float | None:
    value = amp_scale_value(scaler, enabled)
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise RuntimeError(
            "AMP GradScaler scale collapsed to a non-positive/non-finite value "
            f"({value}) at {context}. Optimizer steps may be skipped silently. "
            "Rerun this B2F trainer without --amp."
        )
    return value


@torch.no_grad()
def parameter_l2_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.requires_grad:
            total += float(torch.sum(param.detach().float() * param.detach().float()).cpu())
    return float(math.sqrt(total))


@torch.no_grad()
def first_parameter_snapshot(model: nn.Module) -> tuple[str, torch.Tensor]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            return name, param.detach().float().cpu().clone()
    raise RuntimeError("Model has no trainable parameters.")


@torch.no_grad()
def first_parameter_delta(model: nn.Module, name: str, reference: torch.Tensor) -> float:
    for current_name, param in model.named_parameters():
        if current_name == name:
            diff = param.detach().float().cpu() - reference
            return float(torch.max(torch.abs(diff)).item())
    return float("nan")


def make_loader(dataset: B2FDataset, args: argparse.Namespace, shuffle: bool, *, sampler: WeightedRandomSampler | None = None) -> DataLoader:
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


def make_sampler(dataset: B2FDataset, args: argparse.Namespace) -> WeightedRandomSampler | None:
    if not args.balanced_sampler:
        return None
    positives = [float(row["fluorescence_positive"]) > 0.5 for row in dataset.rows]
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pos_weight = args.positive_sample_weight if args.positive_sample_weight > 0 else min(8.0, n_neg / max(n_pos, 1))
    weights = torch.tensor([pos_weight if is_pos else 1.0 for is_pos in positives], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def estimate_pixel_pos_weight(dataset: B2FDataset, args: argparse.Namespace) -> float:
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(dataset))
    if len(indices) > args.pixel_stat_limit:
        indices = rng.choice(indices, size=args.pixel_stat_limit, replace=False)
    pos_pixels = 0.0
    neg_pixels = 0.0
    for idx in indices:
        row = dataset.rows[int(idx)]
        _, fluorescence_path, mask_path = dataset.image_paths(row)
        fluorescence = image_to_tensor(fluorescence_path, args.image_size)
        mask = image_to_tensor(mask_path, args.image_size, mask=True)
        valid = mask > 0.5
        signal = (fluorescence > args.signal_threshold) & valid
        pos_pixels += float(signal.sum())
        neg_pixels += float((valid & ~signal).sum())
    if pos_pixels < 1:
        return 1.0
    return float(np.clip(neg_pixels / pos_pixels, 1.0, args.max_pixel_pos_weight))


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(2, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    loss = (1.0 - ssim).clamp(0.0, 2.0)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def scalar_losses(outputs: torch.Tensor, batch: dict[str, torch.Tensor], bce: nn.Module, huber: nn.Module) -> torch.Tensor:
    pos_logit = outputs[:, 0]
    peak_pred = outputs[:, 1]
    total_pred = outputs[:, 2]
    pos = batch["positive"].to(outputs.device)
    peak = batch["peak_log"].to(outputs.device)
    total = batch["total_log"].to(outputs.device)
    loss_pos = bce(pos_logit, pos)
    loss_peak = huber(peak_pred, peak)
    loss_total = huber(total_pred, total)
    return loss_pos + 0.20 * loss_peak + 0.05 * loss_total


def b2f_loss(
    logits: torch.Tensor,
    scalar: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    pixel_bce: nn.Module,
    scalar_bce: nn.Module,
    huber: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["fluorescence"].to(logits.device)
    mask = batch["mask"].to(logits.device)
    pred = torch.sigmoid(logits)
    signal = ((target > args.signal_threshold) & (mask > 0.5)).float()
    intensity_weight = target.clamp(0.0, 1.0).pow(1.5)
    weights = args.background_weight + mask * (1.0 + args.signal_boost * intensity_weight)
    image_loss = (charbonnier(pred - target) * weights).sum() / weights.sum().clamp_min(1.0)
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_loss = (bce_map * weights).sum() / weights.sum().clamp_min(1.0)
    structure_loss = ssim_loss(pred, target, mask)
    edge_weights = args.background_weight + mask * (1.0 + args.signal_boost * target.clamp(0.0, 1.0))
    edge_loss = (torch.abs(sobel_edges(pred) - sobel_edges(target)) * edge_weights).sum() / edge_weights.sum().clamp_min(1.0)
    scalar_loss = scalar_losses(scalar, batch, scalar_bce, huber)
    total = image_loss + args.bce_weight * bce_loss + args.ssim_weight * structure_loss + args.edge_weight * edge_loss + args.scalar_weight * scalar_loss
    return total, {
        "loss_image": float(image_loss.detach().cpu()),
        "loss_pixel_bce": float(bce_loss.detach().cpu()),
        "loss_ssim": float(structure_loss.detach().cpu()),
        "loss_edge": float(edge_loss.detach().cpu()),
        "loss_scalar": float(scalar_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    pixel_bce: nn.Module,
    scalar_bce: nn.Module,
    huber: nn.Module,
    panel_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    image_mae_sum = 0.0
    image_mse_sum = 0.0
    masked_mae_sum = 0.0
    signal_mae_sum = 0.0
    background_mae_sum = 0.0
    signal_intersection = 0.0
    signal_union = 0.0
    signal_tp = 0.0
    signal_fp = 0.0
    signal_fn = 0.0
    count = 0
    pixel_count = 0.0
    mask_pixel_count = 0.0
    signal_pixel_count = 0.0
    background_pixel_count = 0.0
    labels: list[float] = []
    scores: list[float] = []
    target_peak: list[float] = []
    pred_peak: list[float] = []
    first_panel = True
    for batch_index, batch in enumerate(loader):
        brightfield = batch["brightfield"].to(device, non_blocking=True)
        fluorescence = batch["fluorescence"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if args.channels_last:
            brightfield = brightfield.contiguous(memory_format=torch.channels_last)
        logits, scalar = model(brightfield)
        pred = torch.sigmoid(logits)
        loss, _ = b2f_loss(logits, scalar, batch, args=args, pixel_bce=pixel_bce, scalar_bce=scalar_bce, huber=huber)
        batch_size = brightfield.shape[0]
        abs_err = torch.abs(pred - fluorescence)
        mse = (pred - fluorescence) ** 2
        signal = ((fluorescence > args.signal_threshold) & (mask > 0.5)).float()
        background = (1.0 - mask).clamp_min(0.0)
        pred_signal = ((pred > args.signal_threshold) & (mask > 0.5)).float()
        total_loss += float(loss.detach().cpu()) * batch_size
        image_mae_sum += float(abs_err.sum().detach().cpu())
        image_mse_sum += float(mse.sum().detach().cpu())
        masked_mae_sum += float((abs_err * mask).sum().detach().cpu())
        signal_mae_sum += float((abs_err * signal).sum().detach().cpu())
        background_mae_sum += float((abs_err * background).sum().detach().cpu())
        signal_intersection += float((pred_signal * signal).sum().detach().cpu())
        signal_union += float(((pred_signal + signal) > 0).float().sum().detach().cpu())
        signal_tp += float((pred_signal * signal).sum().detach().cpu())
        signal_fp += float((pred_signal * (1.0 - signal) * mask).sum().detach().cpu())
        signal_fn += float(((1.0 - pred_signal) * signal).sum().detach().cpu())
        count += batch_size
        pixel_count += float(np.prod(fluorescence.shape))
        mask_pixel_count += float(mask.sum().detach().cpu())
        signal_pixel_count += float(signal.sum().detach().cpu())
        background_pixel_count += float(background.sum().detach().cpu())
        labels.extend(batch["positive"].numpy().tolist())
        scores.extend(torch.sigmoid(scalar[:, 0]).detach().cpu().numpy().tolist())
        target_peak.extend(batch["peak_log"].numpy().tolist())
        pred_peak.extend(scalar[:, 1].detach().cpu().numpy().tolist())
        if panel_path is not None and first_panel:
            save_b2f_panel(
                brightfield.detach().cpu().contiguous(),
                pred.detach().cpu().contiguous(),
                fluorescence.detach().cpu().contiguous(),
                panel_path,
                keys=list(batch["instance_id"]),
            )
            first_panel = False
        if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
            break
    mse_mean = image_mse_sum / max(pixel_count, 1.0)
    precision = signal_tp / max(signal_tp + signal_fp, 1.0)
    recall = signal_tp / max(signal_tp + signal_fn, 1.0)
    return {
        "loss": total_loss / max(count, 1),
        "image_mae": image_mae_sum / max(pixel_count, 1.0),
        "image_mse": mse_mean,
        "image_psnr": float(20.0 * np.log10(1.0 / np.sqrt(max(mse_mean, 1e-12)))),
        "masked_mae": masked_mae_sum / max(mask_pixel_count, 1.0),
        "signal_mae": signal_mae_sum / max(signal_pixel_count, 1.0),
        "background_mae": background_mae_sum / max(background_pixel_count, 1.0),
        "signal_iou": signal_intersection / max(signal_union, 1.0),
        "signal_precision": precision,
        "signal_recall": recall,
        "signal_f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        "expression_auc": binary_auc(labels, scores),
        "expression_ap": average_precision(labels, scores),
        "peak_pearson": pearson_corr(target_peak, pred_peak),
        "n": count,
    }


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    metrics_log: list[dict[str, Any]],
    best_score: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "metrics_log": metrics_log,
        "best_score": best_score,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


def save_periodic_checkpoint(path: Path, payload: dict[str, Any], keep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    checkpoints = sorted(path.parent.glob("epoch_*.pt"))
    if keep > 0 and len(checkpoints) > keep:
        for old in checkpoints[: len(checkpoints) - keep]:
            old.unlink(missing_ok=True)


def select_score(metrics: dict[str, float]) -> float:
    # Reconstruction-first selection. AUROC is too unstable when expression examples are rare.
    loss = metrics.get("loss", float("inf"))
    signal_f1 = metrics.get("signal_f1", 0.0)
    peak = metrics.get("peak_pearson", 0.0)
    if not np.isfinite(peak):
        peak = 0.0
    return -float(loss) + 0.05 * float(signal_f1) + 0.01 * float(peak)


def validation_metrics_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    if "val_loss" not in row:
        return None
    metrics: dict[str, float] = {}
    for key, value in row.items():
        if not key.startswith("val_"):
            continue
        try:
            metrics[key[4:]] = float(value)
        except (TypeError, ValueError):
            metrics[key[4:]] = float("nan")
    return metrics


def early_stop_value(row: dict[str, Any], args: argparse.Namespace) -> float | None:
    val_metrics = validation_metrics_from_row(row)
    if val_metrics is None:
        return None
    if args.early_stop_metric == "score":
        value = select_score(val_metrics)
    else:
        value = val_metrics.get(args.early_stop_metric, float("nan"))
    if not math.isfinite(float(value)):
        return None
    return float(value)


def early_stop_maximizes(metric: str) -> bool:
    return metric in {"score", "signal_f1", "peak_pearson"}


def is_early_stop_improvement(value: float, best_value: float | None, args: argparse.Namespace) -> bool:
    if best_value is None:
        return True
    if early_stop_maximizes(args.early_stop_metric):
        return value > best_value + args.early_stop_min_delta
    return value < best_value - args.early_stop_min_delta


def recover_early_stop_state(metrics: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    best_value: float | None = None
    best_epoch = 0
    stale_evals = 0
    eval_count = 0
    last_eval_epoch = 0
    last_value: float | None = None
    for row in metrics:
        value = early_stop_value(row, args)
        if value is None:
            continue
        eval_count += 1
        last_value = value
        last_eval_epoch = int(row.get("epoch", 0))
        if is_early_stop_improvement(value, best_value, args):
            best_value = value
            best_epoch = last_eval_epoch
            stale_evals = 0
        else:
            stale_evals += 1
    return {
        "metric": args.early_stop_metric,
        "best_value": best_value,
        "best_epoch": best_epoch,
        "stale_evals": stale_evals,
        "eval_count": eval_count,
        "last_eval_epoch": last_eval_epoch,
        "last_value": last_value,
        "maximizes": early_stop_maximizes(args.early_stop_metric),
    }


def should_early_stop(state: dict[str, Any], args: argparse.Namespace, epoch: int) -> bool:
    return bool(
        args.early_stop
        and epoch >= args.early_stop_min_epochs
        and int(state.get("stale_evals", 0)) >= args.early_stop_patience_evals
        and int(state.get("eval_count", 0)) > 0
    )


def recover_best_state(metrics: list[dict[str, Any]], best_model_path: Path, fallback_score: float) -> tuple[float, int]:
    if best_model_path.exists():
        try:
            best = torch.load(best_model_path, map_location="cpu", weights_only=False)
            return float(best.get("best_score", fallback_score)), int(best.get("epoch", 0))
        except Exception:
            pass
    best_score = fallback_score
    best_epoch = 0
    for row in metrics:
        val_metrics = validation_metrics_from_row(row)
        if val_metrics is None:
            continue
        score = select_score(val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = int(row.get("epoch", 0))
    return best_score, best_epoch


def cosine_lr(epoch: int, args: argparse.Namespace) -> float:
    if args.epochs <= 1:
        return args.min_lr
    progress = min(max(epoch - 1, 0) / float(max(args.epochs - 1, 1)), 1.0)
    return float(args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress)))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.architecture == "strong_resunet":
        return StrongB2FResUNet(base_channels=args.base_channels, dropout=args.dropout)
    if args.architecture == "pix2pix_unet":
        return Pix2PixB2FUNet(base_channels=args.base_channels, dropout=args.dropout)
    raise ValueError(f"Unsupported architecture: {args.architecture}")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strong B2F Training Run",
        "",
        "This run is designed to fix the first bottleneck before future prediction is retried.",
        "",
        "Main changes versus the old B2F run:",
        "",
        "- Uses original instance crops instead of only the resized 256 crops.",
        "- Uses a stronger residual GroupNorm U-Net with squeeze-excitation blocks.",
        "- Selects the best checkpoint by reconstruction loss and signal quality, not rare-image AUROC.",
        "- Uses pixel-level signal weighting and BCE-with-logits to handle sparse fluorescence.",
        "- Uses balanced image sampling for rare expression examples when requested.",
        "- Saves `last_model.pt` every epoch and sparse periodic checkpoints only.",
        "",
        "Current status:",
        "",
        f"- Epoch: `{payload.get('epoch')}`",
        f"- Best score: `{payload.get('best_score')}`",
        f"- Best epoch: `{payload.get('best_epoch')}`",
        f"- Architecture: `{payload.get('architecture')}`",
        f"- Early stop: `{payload.get('early_stop')}`",
        f"- Early-stop metric: `{payload.get('early_stop_metric')}`",
        f"- Early-stop stale evals: `{payload.get('early_stop_stale_evals')}`",
        f"- Output folder: `{payload.get('output_root')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_root / "plots").mkdir(parents=True, exist_ok=True)
    (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    manifest_path = args.data_root / "manifests" / "projected_instances_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}. Run build_projected_dataset.py first.")

    train_ds = B2FDataset.from_manifest(manifest_path, "train", args.image_size, augment=True, path_mode=args.path_mode)
    val_ds = B2FDataset.from_manifest(manifest_path, "val", args.image_size, path_mode=args.path_mode)
    test_ds = B2FDataset.from_manifest(manifest_path, "test", args.image_size, path_mode=args.path_mode)
    sampler = make_sampler(train_ds, args)
    train_loader = make_loader(train_ds, args, shuffle=True, sampler=sampler)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)

    train_rows = [row for row in read_csv(manifest_path) if row["split"] == "train"]
    pos_count = sum(int(float(row["fluorescence_positive"])) for row in train_rows)
    neg_count = max(1, len(train_rows) - pos_count)
    scalar_pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device, dtype=torch.float32)
    pixel_pos_weight = estimate_pixel_pos_weight(train_ds, args)
    pixel_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pixel_pos_weight], device=device, dtype=torch.float32))
    scalar_bce = nn.BCEWithLogitsLoss(pos_weight=scalar_pos_weight)
    huber = nn.SmoothL1Loss()

    model = build_model(args).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = grad_scaler(use_amp)
    sentinel_name, sentinel_reference = first_parameter_snapshot(model)
    metrics: list[dict[str, Any]] = []
    best_score = -1e18
    best_epoch = 0
    start_epoch = 1
    metrics_path = args.output_root / "metrics.jsonl"
    last_checkpoint_path = args.output_root / "last_model.pt"

    write_json(
        args.output_root / "run_config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "test_count": len(test_ds),
            "train_expression_count": pos_count,
            "train_expression_fraction": pos_count / max(len(train_ds), 1),
            "scalar_pos_weight": float(scalar_pos_weight.item()),
            "pixel_pos_weight": pixel_pos_weight,
            "amp_requested": bool(args.amp),
            "amp_enabled": bool(use_amp),
            "sentinel_parameter": sentinel_name,
            "architecture": args.architecture,
            "early_stop": bool(args.early_stop),
            "early_stop_metric": args.early_stop_metric,
            "early_stop_patience_evals": args.early_stop_patience_evals,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_min_epochs": args.early_stop_min_epochs,
        },
    )

    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
            assert_healthy_amp_scale(scaler, use_amp, "checkpoint load")
        metrics = list(checkpoint.get("metrics_log", []))
        best_score, best_epoch = recover_best_state(metrics, args.output_root / "best_model.pt", float(checkpoint.get("best_score", best_score)))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        sentinel_name, sentinel_reference = first_parameter_snapshot(model)
        with metrics_path.open("w", encoding="utf-8") as handle:
            for row in metrics:
                handle.write(json.dumps(row) + "\n")
    elif metrics_path.exists():
        metrics_path.unlink()

    early_state = recover_early_stop_state(metrics, args)
    stopped_early = False
    if should_early_stop(early_state, args, start_epoch - 1):
        stopped_early = True
        write_json(
            args.output_root / "early_stop_summary.json",
            {
                "triggered": True,
                "already_converged_on_resume": True,
                "stop_epoch": start_epoch - 1,
                "state": early_state,
                "patience_evals": args.early_stop_patience_evals,
                "min_delta": args.early_stop_min_delta,
                "min_epochs": args.early_stop_min_epochs,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "early_stop_already_converged_on_resume",
                    "stop_epoch": start_epoch - 1,
                    "early_stop": early_state,
                }
            ),
            flush=True,
        )
        start_epoch = args.epochs + 1

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        lr = cosine_lr(epoch, args)
        set_optimizer_lr(optimizer, lr)
        model.train()
        train_loss_sum = 0.0
        seen = 0
        accum = 0
        component_sums: dict[str, float] = {}
        optimizer_steps = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        amp_scale_min = float("inf")
        amp_scale_max = 0.0
        param_norm_before = parameter_l2_norm(model)
        for batch_index, batch in enumerate(train_loader):
            brightfield = batch["brightfield"].to(device, non_blocking=True)
            if args.channels_last:
                brightfield = brightfield.contiguous(memory_format=torch.channels_last)
            with autocast_context(device, use_amp):
                logits, scalar = model(brightfield)
                loss, components = b2f_loss(
                    logits,
                    scalar,
                    {key: value for key, value in batch.items() if torch.is_tensor(value)},
                    args=args,
                    pixel_bce=pixel_bce,
                    scalar_bce=scalar_bce,
                    huber=huber,
                )
                scaled_loss = loss / max(args.grad_accum_steps, 1)
            scaler.scale(scaled_loss).backward()
            accum += 1
            batch_size = brightfield.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * batch_size
            seen += batch_size
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value * batch_size
            if accum >= args.grad_accum_steps:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                scale_value = assert_healthy_amp_scale(scaler, use_amp, f"epoch {epoch} batch {batch_index + 1}")
                if scale_value is not None:
                    amp_scale_min = min(amp_scale_min, scale_value)
                    amp_scale_max = max(amp_scale_max, scale_value)
                grad_norm_float = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                if math.isfinite(grad_norm_float):
                    grad_norm_sum += grad_norm_float
                    grad_norm_max = max(grad_norm_max, grad_norm_float)
                optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accum = 0
            if args.limit_train_batches is not None and batch_index + 1 >= args.limit_train_batches:
                break
        if accum > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            scale_value = assert_healthy_amp_scale(scaler, use_amp, f"epoch {epoch} final accumulation")
            if scale_value is not None:
                amp_scale_min = min(amp_scale_min, scale_value)
                amp_scale_max = max(amp_scale_max, scale_value)
            grad_norm_float = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            if math.isfinite(grad_norm_float):
                grad_norm_sum += grad_norm_float
                grad_norm_max = max(grad_norm_max, grad_norm_float)
            optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
        if optimizer_steps <= 0:
            raise RuntimeError(f"No optimizer steps were executed in epoch {epoch}.")
        param_norm_after = parameter_l2_norm(model)
        sentinel_delta = first_parameter_delta(model, sentinel_name, sentinel_reference)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(seen, 1),
            "lr": lr,
            "optimizer_steps": optimizer_steps,
            "grad_norm_mean": grad_norm_sum / max(optimizer_steps, 1),
            "grad_norm_max": grad_norm_max,
            "param_l2_norm_before": param_norm_before,
            "param_l2_norm_after": param_norm_after,
            "param_l2_norm_delta": param_norm_after - param_norm_before,
            "sentinel_param": sentinel_name,
            "sentinel_param_max_abs_delta_since_resume": sentinel_delta,
        }
        if use_amp:
            row["amp_scale_min"] = amp_scale_min if math.isfinite(amp_scale_min) else None
            row["amp_scale_max"] = amp_scale_max if amp_scale_max > 0 else None
        row.update({f"train_{key}": value / max(seen, 1) for key, value in component_sums.items()})

        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            panel_path = args.output_root / "predictions" / f"val_epoch_{epoch:04d}.png" if (epoch == 1 or epoch % args.panel_every == 0) else None
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args,
                pixel_bce=pixel_bce,
                scalar_bce=scalar_bce,
                huber=huber,
                panel_path=panel_path,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = select_score(val_metrics)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "best_score": best_score}, args.output_root / "best_model.pt")
                if panel_path is not None:
                    shutil.copy2(panel_path, args.output_root / "predictions" / "val_best.png")
            value = early_stop_value(row, args)
            if value is not None:
                if is_early_stop_improvement(value, early_state.get("best_value"), args):
                    early_state = {
                        "metric": args.early_stop_metric,
                        "best_value": value,
                        "best_epoch": epoch,
                        "stale_evals": 0,
                        "eval_count": int(early_state.get("eval_count", 0)) + 1,
                        "last_eval_epoch": epoch,
                        "last_value": value,
                        "maximizes": early_stop_maximizes(args.early_stop_metric),
                    }
                else:
                    early_state = {
                        **early_state,
                        "stale_evals": int(early_state.get("stale_evals", 0)) + 1,
                        "eval_count": int(early_state.get("eval_count", 0)) + 1,
                        "last_eval_epoch": epoch,
                        "last_value": value,
                    }
                row.update(
                    {
                        "early_stop_metric": args.early_stop_metric,
                        "early_stop_value": value,
                        "early_stop_best_value": early_state.get("best_value"),
                        "early_stop_best_epoch": early_state.get("best_epoch"),
                        "early_stop_stale_evals": early_state.get("stale_evals"),
                    }
                )
                if should_early_stop(early_state, args, epoch):
                    row["early_stop_triggered"] = True
                    stopped_early = True

        metrics.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            metrics_log=metrics,
            best_score=best_score,
            args=args,
        )
        torch.save(payload, last_checkpoint_path)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_periodic_checkpoint(args.output_root / "checkpoints" / f"epoch_{epoch:04d}.pt", payload, args.keep_periodic)
        plot_metric_lines(
            metrics,
            args.output_root / "plots" / "training_metrics.png",
            ["train_loss", "val_loss", "val_masked_mae", "val_signal_mae", "val_signal_f1", "val_peak_pearson"],
        )
        write_report(
            args.output_root / "TRAINING_REPORT.md",
            {
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "architecture": args.architecture,
                "early_stop": bool(args.early_stop),
                "early_stop_metric": early_state.get("metric"),
                "early_stop_stale_evals": early_state.get("stale_evals"),
                "output_root": str(args.output_root),
            },
        )
        print(json.dumps(row), flush=True)
        if stopped_early:
            write_json(
                args.output_root / "early_stop_summary.json",
                {
                    "triggered": True,
                    "already_converged_on_resume": False,
                    "stop_epoch": epoch,
                    "state": early_state,
                    "patience_evals": args.early_stop_patience_evals,
                    "min_delta": args.early_stop_min_delta,
                    "min_epochs": args.early_stop_min_epochs,
                },
            )
            print(json.dumps({"stage": "early_stop_triggered", "epoch": epoch, "early_stop": early_state}), flush=True)
            break

    if not (args.output_root / "best_model.pt").exists():
        shutil.copy2(last_checkpoint_path, args.output_root / "best_model.pt")
    best = torch.load(args.output_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        args,
        pixel_bce=pixel_bce,
        scalar_bce=scalar_bce,
        huber=huber,
        panel_path=args.output_root / "predictions" / "test_best.png",
    )
    write_json(args.output_root / "test_metrics.json", {**test_metrics, "best_epoch": int(best.get("epoch", 0))})
    final_epoch = int(metrics[-1]["epoch"]) if metrics else int(best.get("epoch", 0))
    write_report(
        args.output_root / "TRAINING_REPORT.md",
        {
            "epoch": final_epoch,
            "best_score": best_score,
            "best_epoch": int(best.get("epoch", 0)),
            "architecture": args.architecture,
            "early_stop": bool(args.early_stop),
            "early_stop_metric": early_state.get("metric"),
            "early_stop_stale_evals": early_state.get("stale_evals"),
            "output_root": str(args.output_root),
        },
    )
    print(
        json.dumps(
            {
                "stage": "strong_b2f_finished",
                "test": test_metrics,
                "best_epoch": int(best.get("epoch", 0)),
                "final_epoch": final_epoch,
                "stopped_early": stopped_early,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
