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
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_fluorescence_segmentation.datasets import (
    FluorescenceSegmentationDataset,
    make_balanced_weights,
)
from differentiation_prediction.yichao_fluorescence_segmentation.losses import segmentation_loss
from differentiation_prediction.yichao_fluorescence_segmentation.models import GlobalGatedSegUNet
from differentiation_prediction.yichao_fluorescence_segmentation.utils import (
    DEFAULT_OUTPUT_ROOT,
    average_precision,
    binary_auc,
    fbeta,
    gray_rgb,
    green_rgb,
    heat_rgb,
    read_csv,
    red_rgb,
    save_grid,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train context-aware Yichao fluorescence-positive segmentation.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "runs" / "global_gated_unet_v1")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--include-distance", action="store_true", default=True)
    parser.add_argument("--no-distance", action="store_false", dest="include_distance")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--positive-sample-weight", type=float, default=0.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.75)
    parser.add_argument("--tversky-beta", type=float, default=0.25)
    parser.add_argument("--lambda-focal", type=float, default=1.0)
    parser.add_argument("--lambda-tversky", type=float, default=1.0)
    parser.add_argument("--lambda-global", type=float, default=0.30)
    parser.add_argument("--lambda-mil", type=float, default=0.20)
    parser.add_argument("--lambda-area", type=float, default=0.05)
    parser.add_argument("--outside-weight", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--panel-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-periodic", type=int, default=6)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience-evals", type=int, default=15)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.002)
    parser.add_argument("--early-stop-min-epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--overfit-count", type=int, default=None)
    return parser.parse_args()


def make_loader(dataset: FluorescenceSegmentationDataset, args: argparse.Namespace, shuffle: bool, sampler: WeightedRandomSampler | None = None) -> DataLoader:
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


def score_from_metrics(metrics: dict[str, float]) -> float:
    # Precision-first segmentation selection. F0.5 directly penalizes over-painting.
    return float(metrics.get("best_f05", 0.0)) + 0.05 * float(metrics.get("global_auc", 0.0) if math.isfinite(metrics.get("global_auc", 0.0)) else 0.0)


def threshold_metrics(scores: np.ndarray, target: np.ndarray, valid: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold) & (valid > 0)
    truth = (target > 0.5) & (valid > 0)
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = fbeta(precision, recall, beta=1.0)
    f05 = fbeta(precision, recall, beta=0.5)
    iou = tp / max(tp + fp + fn, 1.0)
    return {"precision": precision, "recall": recall, "f1": f1, "f05": f05, "iou": iou}


def save_panel(batch: dict[str, Any], prob: torch.Tensor, path: Path, max_items: int = 6) -> None:
    rows = []
    labels = []
    count = min(max_items, prob.shape[0])
    for idx in range(count):
        bf = batch["brightfield"][idx].squeeze(0).detach().cpu().numpy()
        fl = batch["fluorescence"][idx].squeeze(0).detach().cpu().numpy()
        target = batch["positive"][idx].squeeze(0).detach().cpu().numpy()
        ignore = batch["ignore"][idx].squeeze(0).detach().cpu().numpy()
        pred = prob[idx].squeeze(0).detach().cpu().numpy()
        overlay = gray_rgb(bf)
        overlay = ImageBlend(overlay, green_rgb(pred), 0.45)
        overlay = ImageBlend(overlay, red_rgb(ignore * 0.8), 0.25)
        rows.append([gray_rgb(bf), green_rgb(fl), green_rgb(target), heat_rgb(pred), overlay])
        labels.append(f"{batch['dataset'][idx]} | {batch['target_status'][idx]} | {str(batch['instance_id'][idx])[:90]}")
    save_grid(path, rows, ["brightfield", "raw F", "clean target", "pred prob", "pred overlay"], labels, tile=180)


def ImageBlend(a, b, alpha: float):
    from PIL import Image

    return Image.blend(a.convert("RGB"), b.convert("RGB"), alpha)


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
    loss_sum = 0.0
    seen = 0
    all_scores: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    global_labels: list[float] = []
    global_scores: list[float] = []
    first_panel = True
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        outputs = model(image)
        loss, _ = segmentation_loss_from_args(outputs, batch, args)
        prob = torch.sigmoid(outputs["logits"]).detach().cpu()
        valid = (batch["valid"] * (args.outside_weight + batch["organoid_mask"])).detach().cpu()
        target = batch["positive"].detach().cpu()
        all_scores.append(prob.numpy().reshape(-1))
        all_target.append(target.numpy().reshape(-1))
        all_valid.append((valid.numpy().reshape(-1) > 0.5).astype(np.uint8))
        global_labels.extend(batch["global_positive"].numpy().tolist())
        global_scores.extend(torch.sigmoid(outputs["global_logits"]).detach().cpu().numpy().tolist())
        batch_size = image.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        if panel_path is not None and first_panel:
            save_panel(batch, prob, panel_path)
            first_panel = False
        if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
            break
    scores = np.concatenate(all_scores) if all_scores else np.asarray([], dtype=np.float32)
    target = np.concatenate(all_target) if all_target else np.asarray([], dtype=np.float32)
    valid = np.concatenate(all_valid) if all_valid else np.asarray([], dtype=np.uint8)
    metrics_05 = threshold_metrics(scores, target, valid, 0.5)
    thresholds = np.linspace(0.05, 0.95, 19)
    all_threshold_metrics = [(threshold, threshold_metrics(scores, target, valid, float(threshold))) for threshold in thresholds]
    best_threshold, best = max(all_threshold_metrics, key=lambda item: item[1]["f05"])
    pos_pixels = target[(valid > 0)] > 0.5
    ap = average_precision(pos_pixels.astype(float).tolist(), scores[(valid > 0)].tolist()) if valid.sum() > 0 else float("nan")
    return {
        "loss": loss_sum / max(seen, 1),
        "precision_05": metrics_05["precision"],
        "recall_05": metrics_05["recall"],
        "f1_05": metrics_05["f1"],
        "f05_05": metrics_05["f05"],
        "iou_05": metrics_05["iou"],
        "best_threshold": float(best_threshold),
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
        "best_f05": best["f05"],
        "best_iou": best["iou"],
        "pixel_ap": ap,
        "global_auc": binary_auc(global_labels, global_scores),
        "global_ap": average_precision(global_labels, global_scores),
        "n": seen,
    }


def segmentation_loss_from_args(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    return segmentation_loss(
        outputs,
        {key: value for key, value in batch.items() if torch.is_tensor(value)},
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        global_pos_weight=args.global_pos_weight,
        lambda_focal=args.lambda_focal,
        lambda_tversky=args.lambda_tversky,
        lambda_global=args.lambda_global,
        lambda_mil=args.lambda_mil,
        lambda_area=args.lambda_area,
        outside_weight=args.outside_weight,
    )


def plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, [row.get("train_loss", np.nan) for row in rows], label="train")
    val_rows = [row for row in rows if "val_loss" in row]
    if val_rows:
        val_epochs = [int(row["epoch"]) for row in val_rows]
        axes[0, 0].plot(val_epochs, [row.get("val_loss", np.nan) for row in val_rows], "o-", label="val")
        axes[0, 1].plot(val_epochs, [row.get("val_best_f05", np.nan) for row in val_rows], "o-", label="best F0.5")
        axes[1, 0].plot(val_epochs, [row.get("val_best_precision", np.nan) for row in val_rows], "o-", label="precision")
        axes[1, 0].plot(val_epochs, [row.get("val_best_recall", np.nan) for row in val_rows], "o-", label="recall")
        axes[1, 1].plot(val_epochs, [row.get("val_global_auc", np.nan) for row in val_rows], "o-", label="global AUROC")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlabel("epoch")
    axes[0, 0].set_title("Loss")
    axes[0, 1].set_title("Validation F0.5")
    axes[1, 0].set_title("Precision/recall")
    axes[1, 1].set_title("Global expression")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def checkpoint_payload(model, optimizer, epoch: int, metrics: list[dict[str, Any]], best_score: float, best_epoch: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


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
        raise SystemExit(f"Missing target manifest: {manifest_path}. Run build_clean_targets.py first.")
    rows = read_csv(manifest_path)
    if args.overfit_count is not None:
        train_rows = [row for row in rows if row["split"] == "train"][: args.overfit_count]
        val_rows = train_rows
        test_rows = train_rows
    else:
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        test_rows = [row for row in rows if row["split"] == "test"]
    train_augment = args.overfit_count is None
    train_ds = FluorescenceSegmentationDataset(train_rows, args.image_size, augment=train_augment, include_distance=args.include_distance)
    val_ds = FluorescenceSegmentationDataset(val_rows, args.image_size, include_distance=args.include_distance)
    test_ds = FluorescenceSegmentationDataset(test_rows, args.image_size, include_distance=args.include_distance)
    n_pos = sum(int(float(row["target_global_positive"])) for row in train_rows)
    n_neg = max(1, len(train_rows) - n_pos)
    args.global_pos_weight = float(n_neg / max(n_pos, 1))
    sampler = None
    if args.balanced_sampler:
        weights = make_balanced_weights(train_rows, args.positive_sample_weight)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = make_loader(train_ds, args, shuffle=True, sampler=sampler)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    in_channels = 3 if args.include_distance else 2
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
            "train_positive": n_pos,
            "train_positive_fraction": n_pos / max(len(train_ds), 1),
            "global_pos_weight": args.global_pos_weight,
            "train_augment": train_augment,
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
    for row in metrics:
        if "val_best_f05" not in row:
            continue
        score = score_from_metrics({key[4:]: value for key, value in row.items() if key.startswith("val_")})
        if score > best_score + args.early_stop_min_delta:
            best_score = score
            best_epoch = int(row["epoch"])
            stale_evals = 0
        else:
            stale_evals += 1

    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        set_lr(optimizer, cosine_lr(epoch, args))
        model.train()
        loss_sum = 0.0
        seen = 0
        component_sums: dict[str, float] = {}
        for batch_index, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            outputs = model(image)
            loss, components = segmentation_loss_from_args(outputs, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(grad_clip_params, 3.0, foreach=False)
            optimizer.step()
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
            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
        }
        row.update({f"train_{key}": value / max(seen, 1) for key, value in component_sums.items()})
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            panel_path = args.output_root / "predictions" / f"val_epoch_{epoch:04d}.png" if (epoch == 1 or epoch % args.panel_every == 0) else None
            val_metrics = evaluate(model, val_loader, device, args, panel_path=panel_path)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = score_from_metrics(val_metrics)
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
                row["early_stop_triggered"] = True
                stopped_early = True
        metrics.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        payload = checkpoint_payload(model, optimizer, epoch, metrics, best_score, best_epoch, args)
        torch.save(payload, last_path)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_periodic(args.output_root / "checkpoints" / f"epoch_{epoch:04d}.pt", payload, args.keep_periodic)
        plot_metrics(metrics, args.output_root / "plots" / "training_metrics.png")
        print(json.dumps(row), flush=True)
        if stopped_early:
            write_json(args.output_root / "early_stop_summary.json", {"triggered": True, "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score, "stale_evals": stale_evals})
            break
    if not (args.output_root / "best_model.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": metrics[-1]["epoch"], "score": best_score, "metrics": metrics[-1]}, args.output_root / "best_model.pt")
    best = torch.load(args.output_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device, args, panel_path=args.output_root / "predictions" / "test_best.png")
    write_json(args.output_root / "test_metrics.json", {**test_metrics, "best_epoch": int(best.get("epoch", 0)), "stopped_early": stopped_early})
    print(json.dumps({"stage": "segmentation_finished", "test": test_metrics, "best_epoch": int(best.get("epoch", 0)), "stopped_early": stopped_early}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
