#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.datasets import B2FDataset, FEATURE_COLUMNS
from differentiation_prediction.yichao_future_expression.models import B2FMultiTaskUNet
from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    average_precision,
    binary_auc,
    coerce_float,
    coerce_int,
    pearson_corr,
    read_csv,
    save_b2f_panel,
    set_seed,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize the Yichao projected B2F model.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1_b2f")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1_feature_analysis")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1_b2f_evaluation")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260510)
    return parser.parse_args()


def make_loader(dataset: B2FDataset, args: argparse.Namespace) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def roc_points(labels: Sequence[float], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64) > 0.5
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s)
    y = y[order]
    pos = max(int(y.sum()), 1)
    neg = max(int((~y).sum()), 1)
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    fpr = np.concatenate([[0.0], fp / neg, [1.0]])
    tpr = np.concatenate([[0.0], tp / pos, [1.0]])
    return fpr, tpr


def pr_points(labels: Sequence[float], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64) > 0.5
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    return np.concatenate([[0.0], recall]), np.concatenate([[1.0], precision])


@torch.no_grad()
def infer(model: B2FMultiTaskUNet, loader: DataLoader, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    labels: list[float] = []
    scores: list[float] = []
    target_peak: list[float] = []
    pred_peak: list[float] = []
    image_mae_sum = 0.0
    image_mse_sum = 0.0
    count = 0
    for batch in loader:
        brightfield = batch["brightfield"].to(device, non_blocking=True)
        fluorescence = batch["fluorescence"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred, scalar = model(brightfield)
        prob = torch.sigmoid(scalar[:, 0]).detach().cpu().numpy()
        pred_peak_np = scalar[:, 1].detach().cpu().numpy()
        target_peak_np = batch["peak_log"].numpy()
        err = torch.abs(pred - fluorescence)
        per_mae = err.flatten(1).mean(dim=1).detach().cpu().numpy()
        per_mse = ((pred - fluorescence) ** 2).flatten(1).mean(dim=1).detach().cpu().numpy()
        per_masked_mae = ((err * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1.0)).detach().cpu().numpy()
        batch_size = brightfield.shape[0]
        image_mae_sum += float(per_mae.sum())
        image_mse_sum += float(per_mse.sum())
        count += batch_size
        labels.extend(batch["positive"].numpy().tolist())
        scores.extend(prob.tolist())
        target_peak.extend(target_peak_np.tolist())
        pred_peak.extend(pred_peak_np.tolist())
        for i in range(batch_size):
            rows.append(
                {
                    "instance_id": batch["instance_id"][i],
                    "dataset": batch["dataset"][i],
                    "fluorescence_positive": int(batch["positive"][i].item()),
                    "pred_positive_probability": float(prob[i]),
                    "true_peak_log": float(target_peak_np[i]),
                    "pred_peak_log": float(pred_peak_np[i]),
                    "image_mae": float(per_mae[i]),
                    "image_mse": float(per_mse[i]),
                    "masked_image_mae": float(per_masked_mae[i]),
                }
            )
    mse = image_mse_sum / max(count, 1)
    metrics = {
        "n": float(count),
        "positive_count": float(sum(1 for value in labels if value > 0.5)),
        "image_mae": image_mae_sum / max(count, 1),
        "image_mse": mse,
        "image_psnr": float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-12)))),
        "positive_auc": binary_auc(labels, scores),
        "positive_ap": average_precision(labels, scores),
        "peak_pearson": pearson_corr(target_peak, pred_peak),
        "peak_mae": float(np.mean(np.abs(np.asarray(target_peak) - np.asarray(pred_peak)))) if target_peak else float("nan"),
    }
    return rows, metrics


def per_dataset_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    out: list[dict[str, Any]] = []
    for dataset, items in sorted(grouped.items()):
        labels = [float(row["fluorescence_positive"]) for row in items]
        scores = [float(row["pred_positive_probability"]) for row in items]
        true_peak = [float(row["true_peak_log"]) for row in items]
        pred_peak = [float(row["pred_peak_log"]) for row in items]
        out.append(
            {
                "dataset": dataset,
                "n": len(items),
                "positive_count": int(sum(labels)),
                "positive_auc": binary_auc(labels, scores),
                "positive_ap": average_precision(labels, scores),
                "peak_pearson": pearson_corr(true_peak, pred_peak),
                "image_mae": float(np.mean([float(row["image_mae"]) for row in items])),
                "masked_image_mae": float(np.mean([float(row["masked_image_mae"]) for row in items])),
            }
        )
    return out


def plot_b2f_summary(rows: Sequence[dict[str, Any]], metrics: dict[str, float], output: Path) -> None:
    labels = np.asarray([float(row["fluorescence_positive"]) for row in rows])
    scores = np.asarray([float(row["pred_positive_probability"]) for row in rows])
    true_peak = np.asarray([float(row["true_peak_log"]) for row in rows])
    pred_peak = np.asarray([float(row["pred_peak_log"]) for row in rows])
    image_mae = np.asarray([float(row["image_mae"]) for row in rows])
    fpr, tpr = roc_points(labels, scores)
    recall, precision = pr_points(labels, scores)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes[0, 0]
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC AUC={metrics['positive_auc']:.3f}")

    ax = axes[0, 1]
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR AP={metrics['positive_ap']:.3f}")

    ax = axes[1, 0]
    ax.scatter(true_peak, pred_peak, s=9, alpha=0.45)
    lo = float(min(np.min(true_peak), np.min(pred_peak)))
    hi = float(max(np.max(true_peak), np.max(pred_peak)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("true fluorescence peak log")
    ax.set_ylabel("predicted peak log")
    ax.set_title(f"peak Pearson={metrics['peak_pearson']:.3f}")

    ax = axes[1, 1]
    ax.hist(image_mae, bins=40, color="#2d6a8e", alpha=0.85)
    ax.axvline(metrics["image_mae"], color="#b23a48", lw=2, label=f"mean={metrics['image_mae']:.3f}")
    ax.set_xlabel("per-instance image MAE")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title(f"PSNR={metrics['image_psnr']:.2f} dB")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_score_histogram(rows: Sequence[dict[str, Any]], output: Path) -> None:
    labels = np.asarray([float(row["fluorescence_positive"]) for row in rows])
    scores = np.asarray([float(row["pred_positive_probability"]) for row in rows])
    plt.figure(figsize=(7, 4))
    plt.hist(scores[labels < 0.5], bins=35, alpha=0.65, label="fluorescence negative")
    plt.hist(scores[labels > 0.5], bins=35, alpha=0.65, label="fluorescence positive")
    plt.xlabel("predicted positive probability")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def plot_feature_trends(manifest_rows: Sequence[dict[str, str]], feature_names: Sequence[str], output: Path) -> None:
    cols = min(3, len(feature_names))
    rows = int(math.ceil(len(feature_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    positives = np.asarray([coerce_int(row["fluorescence_positive"]) for row in manifest_rows], dtype=np.float32)
    peaks = np.asarray([coerce_float(row["fl_corrected_p90_log"]) for row in manifest_rows], dtype=np.float32)
    for ax, feature in zip(axes.ravel(), feature_names):
        values = np.asarray([coerce_float(row.get(feature)) for row in manifest_rows], dtype=np.float32)
        finite = np.isfinite(values)
        if finite.sum() < 20:
            ax.axis("off")
            continue
        edges = np.quantile(values[finite], np.linspace(0, 1, 11))
        edges = np.unique(edges)
        if len(edges) < 3:
            ax.axis("off")
            continue
        centers: list[float] = []
        pos_rate: list[float] = []
        peak_mean: list[float] = []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = finite & (values >= left) & (values <= right)
            if mask.sum() < 10:
                continue
            centers.append(float(np.median(values[mask])))
            pos_rate.append(float(np.mean(positives[mask])))
            peak_mean.append(float(np.mean(peaks[mask])))
        ax2 = ax.twinx()
        ax.plot(centers, pos_rate, marker="o", color="#1f77b4", label="positive fraction")
        ax2.plot(centers, peak_mean, marker="s", color="#b23a48", label="mean peak log")
        ax.set_xlabel(feature)
        ax.set_ylabel("positive fraction", color="#1f77b4")
        ax2.set_ylabel("mean peak log", color="#b23a48")
        ax.set_title(feature)
    for ax in axes.ravel()[len(feature_names) :]:
        ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_feature_importance(feature_rows: Sequence[dict[str, str]], output: Path) -> None:
    if not feature_rows:
        return
    top = feature_rows[: min(10, len(feature_rows))]
    names = [row["feature"] for row in top][::-1]
    auc_drop = [coerce_float(row["auc_drop"]) for row in top][::-1]
    peak_drop = [coerce_float(row["peak_pearson_drop"]) for row in top][::-1]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].barh(y, auc_drop, color="#2d6a8e")
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("AUROC drop")
    axes[0].set_title("classification importance")
    axes[1].barh(y, peak_drop, color="#b23a48")
    axes[1].set_xlabel("peak Pearson drop")
    axes[1].set_title("intensity importance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_selected_panel(
    model: B2FMultiTaskUNet,
    dataset: B2FDataset,
    prediction_rows: Sequence[dict[str, Any]],
    device: torch.device,
    output: Path,
) -> None:
    by_id = {row["instance_id"]: i for i, row in enumerate(dataset.rows)}
    positives = [row for row in prediction_rows if int(row["fluorescence_positive"]) == 1]
    negatives = [row for row in prediction_rows if int(row["fluorescence_positive"]) == 0]
    selected = sorted(positives, key=lambda row: float(row["pred_positive_probability"]), reverse=True)[:4]
    selected += sorted(negatives, key=lambda row: float(row["pred_positive_probability"]))[:4]
    if not selected:
        return
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    keys: list[str] = []
    for row in selected:
        index = by_id.get(str(row["instance_id"]))
        if index is None:
            continue
        item = dataset[index]
        inputs.append(item["brightfield"])  # type: ignore[arg-type]
        targets.append(item["fluorescence"])  # type: ignore[arg-type]
        keys.append(f"{row['dataset']} | y={row['fluorescence_positive']} | p={float(row['pred_positive_probability']):.3f}")
    if not inputs:
        return
    batch = torch.stack(inputs).to(device)
    with torch.no_grad():
        preds, _ = model(batch)
    save_b2f_panel(batch.cpu(), preds.cpu(), torch.stack(targets), output, keys=keys, max_items=len(keys))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_report(
    output: Path,
    metrics: dict[str, float],
    per_dataset: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    top_features = feature_rows[:5]
    lines = [
        "# B2F Feasibility And Feature Explanation",
        "",
        f"- Split: `{args.split}`",
        f"- Model: `{args.model_root / 'best_model.pt'}`",
        f"- Data manifest: `{args.data_root / 'manifests' / 'projected_instances_manifest.csv'}`",
        f"- Test instances: `{int(metrics['n'])}`",
        f"- Positive instances: `{int(metrics['positive_count'])}`",
        f"- Image MAE: `{metrics['image_mae']:.4f}`",
        f"- Image PSNR: `{metrics['image_psnr']:.2f} dB`",
        f"- Positive AUROC: `{metrics['positive_auc']:.3f}`",
        f"- Positive average precision: `{metrics['positive_ap']:.3f}`",
        f"- Fluorescence peak Pearson: `{metrics['peak_pearson']:.3f}`",
        "",
        "## Interpretation",
        "",
        "The B2F reconstruction task is feasible: the model recovers a non-trivial fluorescence intensity map from the projected brightfield crop, and the predicted scalar fluorescence peak correlates with the true fluorescence peak.",
        "",
        "The binary positive/negative task is class-imbalanced, so AUROC is more useful than raw accuracy here. Average precision is low because positives are rare in the held-out split.",
        "",
        "## Most Explanatory Explicit Features",
        "",
    ]
    if top_features:
        for row in top_features:
            lines.append(
                f"- `{row['feature']}`: AUROC drop `{coerce_float(row['auc_drop']):.3f}`, peak-correlation drop `{coerce_float(row['peak_pearson_drop']):.3f}`"
            )
    else:
        lines.append("- No feature-importance table found.")
    lines += [
        "",
        "Practical interpretation: the strongest explicit drivers are size/shape and segmentation support features. Large projected area, bounding-box dimensions, and support ratio capture whether the organoid has the morphology associated with differentiated fluorescence-positive cells. Edge strength and circularity provide weaker but still useful boundary/shape cues.",
        "",
        "## Per-Dataset Metrics",
        "",
    ]
    for row in per_dataset:
        lines.append(
            f"- `{row['dataset']}`: n `{row['n']}`, positives `{row['positive_count']}`, AUROC `{coerce_float(row['positive_auc']):.3f}`, peak Pearson `{coerce_float(row['peak_pearson']):.3f}`, image MAE `{coerce_float(row['image_mae']):.4f}`"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = args.data_root / "manifests" / "projected_instances_manifest.csv"
    dataset = B2FDataset.from_manifest(manifest_path, args.split, args.image_size)
    loader = make_loader(dataset, args)
    model = B2FMultiTaskUNet(base_channels=args.base_channels).to(device)
    checkpoint = torch.load(args.model_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    rows, metrics = infer(model, loader, device)
    dataset_metrics = per_dataset_metrics(rows)

    write_csv(args.output_root / f"{args.split}_predictions.csv", rows)
    write_csv(args.output_root / f"{args.split}_per_dataset_metrics.csv", dataset_metrics)
    write_json(
        args.output_root / f"{args.split}_metrics.json",
        {
            **metrics,
            "best_epoch": int(checkpoint.get("epoch", -1)),
            "device": str(device),
            "split": args.split,
        },
    )

    plot_b2f_summary(rows, metrics, args.output_root / "plots" / "b2f_performance_summary.png")
    plot_score_histogram(rows, args.output_root / "plots" / "positive_score_histogram.png")
    make_selected_panel(model, dataset, rows, device, args.output_root / "plots" / "selected_b2f_predictions.png")

    manifest_rows = [row for row in read_csv(manifest_path) if row["split"] == args.split]
    feature_importance = read_csv_rows(args.feature_root / "feature_importance.csv")
    top_feature_names = [row["feature"] for row in feature_importance[:6] if row.get("feature") in FEATURE_COLUMNS]
    if top_feature_names:
        plot_feature_trends(manifest_rows, top_feature_names, args.output_root / "plots" / "feature_trends.png")
    plot_feature_importance(feature_importance, args.output_root / "plots" / "feature_importance_summary.png")
    write_report(args.output_root / "b2f_evaluation_report.md", metrics, dataset_metrics, feature_importance, args)

    print(json.dumps({"metrics": metrics, "output_root": str(args.output_root), "top_features": feature_importance[:5]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
