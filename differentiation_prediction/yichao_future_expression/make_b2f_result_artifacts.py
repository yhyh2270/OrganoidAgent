#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.datasets import B2FDataset
from differentiation_prediction.yichao_future_expression.train_b2f_strong import build_model, select_score, validation_metrics_from_row
from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    image_to_tensor,
    load_font,
    make_green,
    read_csv,
    to_uint8_image,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication artifacts for the finished Yichao B2F run.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "stage1_b2f_pix2pix_384_v1_noamp_long",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/publication/yichao_b2f_pix2pix_results_report"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--example-count", type=int, default=6)
    parser.add_argument("--saliency-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gray_to_rgb(tensor: torch.Tensor) -> Image.Image:
    return to_uint8_image(tensor).convert("RGB")


def heatmap_image(array: np.ndarray, *, cmap: str = "magma") -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        lo = float(np.percentile(arr[finite], 1))
        hi = float(np.percentile(arr[finite], 99))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgba = plt.get_cmap(cmap)(arr)
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(base: torch.Tensor, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base_rgb = gray_to_rgb(base)
    heat_rgb = heatmap_image(heat)
    return Image.blend(base_rgb, heat_rgb, alpha)


def make_grid(
    rows: Sequence[Sequence[Image.Image]],
    labels: Sequence[str],
    row_labels: Sequence[str],
    path: Path,
    *,
    tile: int,
    label_h: int = 30,
    row_label_h: int = 28,
) -> None:
    font = load_font(13)
    header_h = label_h
    cols = len(labels)
    canvas = Image.new("RGB", (cols * tile, header_h + len(rows) * (tile + row_label_h)), (18, 21, 24))
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(labels):
        draw.text((col * tile + 8, 8), label, fill=(235, 240, 242), font=font)
    for row_idx, images in enumerate(rows):
        y = header_h + row_idx * (tile + row_label_h)
        for col, image in enumerate(images):
            canvas.paste(image.resize((tile, tile), Image.Resampling.BILINEAR), (col * tile, y))
        draw.text((8, y + tile + 6), row_labels[row_idx][:130], fill=(185, 194, 199), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def plot_training(metrics: list[dict[str, Any]], early: dict[str, Any], output: Path) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in metrics], dtype=np.int32)
    train = np.asarray([finite_float(row.get("train_loss")) for row in metrics], dtype=np.float64)
    val_rows = [row for row in metrics if "val_loss" in row]
    val_epochs = np.asarray([int(row["epoch"]) for row in val_rows], dtype=np.int32)
    val_loss = np.asarray([finite_float(row.get("val_loss")) for row in val_rows], dtype=np.float64)
    val_masked = np.asarray([finite_float(row.get("val_masked_mae")) for row in val_rows], dtype=np.float64)
    val_f1 = np.asarray([finite_float(row.get("val_signal_f1")) for row in val_rows], dtype=np.float64)
    scores = []
    for row in val_rows:
        val_metrics = validation_metrics_from_row(row)
        scores.append(select_score(val_metrics or {}))
    scores = np.asarray(scores, dtype=np.float64)
    best_epoch = int(early.get("state", {}).get("best_epoch", 0))
    stop_epoch = int(early.get("stop_epoch", int(epochs[-1]) if len(epochs) else 0))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(epochs, train, color="#1f5f70", label="train loss")
    axes[0, 0].plot(val_epochs, val_loss, "o-", color="#c4512f", label="val loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(val_epochs, scores, "o-", color="#5c6b28")
    axes[0, 1].set_title("Early-stop score")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(val_epochs, val_masked, "o-", color="#7a3e65", label="masked MAE")
    axes[1, 0].set_title("Masked reconstruction error")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(val_epochs, val_f1, "o-", color="#2e6f40", label="signal F1")
    axes[1, 1].set_title("Fluorescent signal localization")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(alpha=0.25)

    for ax in axes.ravel():
        if best_epoch:
            ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, alpha=0.7)
        if stop_epoch:
            ax.axvline(stop_epoch, color="#aa2222", linestyle=":", linewidth=1, alpha=0.7)
    fig.suptitle("B2F training convergence and early stopping")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_test_summary(test_metrics: dict[str, Any], output: Path) -> None:
    metric_names = ["masked_mae", "signal_mae", "signal_precision", "signal_recall", "signal_f1", "expression_auc"]
    values = [finite_float(test_metrics.get(name)) for name in metric_names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#8baaad", "#d18f52", "#b05d3f", "#5f8f55", "#2e6f40", "#314f7d"]
    ax.bar(metric_names, values, color=colors)
    ax.set_ylim(0, max(1.0, max(v for v in values if math.isfinite(v)) * 1.15))
    ax.set_title("Held-out test quantification")
    ax.set_ylabel("metric value")
    ax.tick_params(axis="x", rotation=25)
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, dataset: B2FDataset, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, Any]] = []
    threshold = 0.25
    model.eval()
    for batch in loader:
        brightfield = batch["brightfield"].to(device)
        fluorescence = batch["fluorescence"].to(device)
        mask = batch["mask"].to(device)
        logits, scalar = model(brightfield)
        pred = torch.sigmoid(logits).clamp(0, 1)
        valid = mask > 0.5
        target_signal = (fluorescence > threshold) & valid
        pred_signal = (pred > threshold) & valid
        for idx, instance_id in enumerate(batch["instance_id"]):
            valid_i = valid[idx]
            target_i = fluorescence[idx]
            pred_i = pred[idx]
            target_signal_i = target_signal[idx]
            pred_signal_i = pred_signal[idx]
            abs_err = torch.abs(pred_i - target_i)
            tp = float((target_signal_i & pred_signal_i).sum().detach().cpu())
            fp = float((~target_signal_i & pred_signal_i & valid_i).sum().detach().cpu())
            fn = float((target_signal_i & ~pred_signal_i).sum().detach().cpu())
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
            valid_count = float(valid_i.sum().detach().cpu())
            rows.append(
                {
                    "instance_id": instance_id,
                    "dataset": batch["dataset"][idx],
                    "target_peak": float(target_i.max().detach().cpu()),
                    "pred_peak": float(pred_i.max().detach().cpu()),
                    "target_mean_masked": float((target_i * valid_i).sum().detach().cpu() / max(valid_count, 1.0)),
                    "pred_mean_masked": float((pred_i * valid_i).sum().detach().cpu() / max(valid_count, 1.0)),
                    "masked_mae": float((abs_err * valid_i).sum().detach().cpu() / max(valid_count, 1.0)),
                    "target_signal_fraction": float(target_signal_i.sum().detach().cpu() / max(valid_count, 1.0)),
                    "pred_signal_fraction": float(pred_signal_i.sum().detach().cpu() / max(valid_count, 1.0)),
                    "signal_precision": precision,
                    "signal_recall": recall,
                    "signal_f1": f1,
                    "scalar_expression_score": float(torch.sigmoid(scalar[idx, 0]).detach().cpu()),
                }
            )
    return rows


def plot_prediction_scatter(rows: list[dict[str, Any]], output: Path) -> None:
    target_mean = np.asarray([row["target_mean_masked"] for row in rows], dtype=np.float64)
    pred_mean = np.asarray([row["pred_mean_masked"] for row in rows], dtype=np.float64)
    target_signal = np.asarray([row["target_signal_fraction"] for row in rows], dtype=np.float64)
    pred_signal = np.asarray([row["pred_signal_fraction"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(target_mean, pred_mean, s=14, alpha=0.45, color="#1f5f70")
    lim0 = [0, max(float(target_mean.max(initial=0)), float(pred_mean.max(initial=0)), 0.05)]
    axes[0].plot(lim0, lim0, "--", color="black", linewidth=1)
    axes[0].set_xlabel("true masked fluorescence mean")
    axes[0].set_ylabel("predicted masked fluorescence mean")
    axes[0].set_title("Intensity calibration")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(target_signal, pred_signal, s=14, alpha=0.45, color="#8a4f7d")
    lim1 = [0, max(float(target_signal.max(initial=0)), float(pred_signal.max(initial=0)), 0.05)]
    axes[1].plot(lim1, lim1, "--", color="black", linewidth=1)
    axes[1].set_xlabel("true signal fraction")
    axes[1].set_ylabel("predicted signal fraction")
    axes[1].set_title("Signal area calibration")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def select_examples(rows: list[dict[str, Any]], count: int) -> list[int]:
    if not rows:
        return []
    order = np.argsort([row["target_signal_fraction"] for row in rows])[::-1]
    chosen: list[int] = []
    for fraction in np.linspace(0.0, 0.85, count):
        chosen.append(int(order[min(int(fraction * (len(order) - 1)), len(order) - 1)]))
    return sorted(set(chosen), key=lambda idx: rows[idx]["target_signal_fraction"], reverse=True)[:count]


@torch.no_grad()
def prediction_for_row(model: torch.nn.Module, row: dict[str, str], image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    brightfield = image_to_tensor(Path(row["brightfield_crop_path"]), image_size).unsqueeze(0).to(device)
    fluorescence = image_to_tensor(Path(row["fluorescence_crop_path"]), image_size).to(device)
    mask = image_to_tensor(Path(row["mask_crop_path"]), image_size, mask=True).to(device)
    pred = torch.sigmoid(model(brightfield)[0]).squeeze(0).detach()
    return brightfield.squeeze(0).detach().cpu(), fluorescence.detach().cpu(), pred.detach().cpu(), mask.detach().cpu()


def make_prediction_examples(
    model: torch.nn.Module,
    manifest_rows: list[dict[str, str]],
    per_example: list[dict[str, Any]],
    output: Path,
    *,
    args: argparse.Namespace,
    run_args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    by_id = {row["instance_id"]: row for row in manifest_rows}
    selected_indices = select_examples(per_example, args.example_count)
    grid_rows: list[list[Image.Image]] = []
    labels: list[str] = []
    row_labels: list[str] = []
    for idx in selected_indices:
        result = per_example[idx]
        row = by_id[result["instance_id"]]
        brightfield, fluorescence, pred, _ = prediction_for_row(model, row, run_args.image_size, device)
        err = torch.abs(pred - fluorescence)
        grid_rows.append(
            [
                gray_to_rgb(brightfield),
                make_green(to_uint8_image(fluorescence)),
                make_green(to_uint8_image(pred)),
                heatmap_image(err.squeeze(0).numpy(), cmap="inferno"),
            ]
        )
        row_labels.append(
            f"{result['dataset']} | target_signal={result['target_signal_fraction']:.3f} "
            f"pred_signal={result['pred_signal_fraction']:.3f} F1={result['signal_f1']:.3f}"
        )
        labels = ["brightfield", "true F", "predicted F", "absolute error"]
    make_grid(grid_rows, labels, row_labels, output, tile=run_args.image_size)
    return [per_example[idx]["instance_id"] for idx in selected_indices]


def make_saliency_examples(
    model: torch.nn.Module,
    manifest_rows: list[dict[str, str]],
    per_example: list[dict[str, Any]],
    output: Path,
    *,
    args: argparse.Namespace,
    run_args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    by_id = {row["instance_id"]: row for row in manifest_rows}
    selected_indices = select_examples(per_example, args.saliency_count)
    grid_rows: list[list[Image.Image]] = []
    row_labels: list[str] = []
    model.eval()
    for idx in selected_indices:
        result = per_example[idx]
        row = by_id[result["instance_id"]]
        brightfield = image_to_tensor(Path(row["brightfield_crop_path"]), run_args.image_size).unsqueeze(0).to(device)
        fluorescence = image_to_tensor(Path(row["fluorescence_crop_path"]), run_args.image_size).to(device)
        mask = image_to_tensor(Path(row["mask_crop_path"]), run_args.image_size, mask=True).to(device)
        brightfield.requires_grad_(True)
        logits, _ = model(brightfield)
        pred = torch.sigmoid(logits).squeeze(0)
        objective = (pred * mask).sum() / mask.sum().clamp_min(1.0)
        model.zero_grad(set_to_none=True)
        objective.backward()
        saliency = brightfield.grad.detach().abs().squeeze().cpu().numpy()
        bf_cpu = brightfield.detach().squeeze(0).cpu()
        pred_cpu = pred.detach().cpu()
        grid_rows.append(
            [
                gray_to_rgb(bf_cpu),
                make_green(to_uint8_image(fluorescence.detach().cpu())),
                make_green(to_uint8_image(pred_cpu)),
                overlay_heatmap(bf_cpu, saliency),
            ]
        )
        row_labels.append(
            f"{result['dataset']} | masked output saliency | target_signal={result['target_signal_fraction']:.3f}"
        )
    make_grid(grid_rows, ["brightfield", "true F", "predicted F", "saliency overlay"], row_labels, output, tile=run_args.image_size)
    return [per_example[idx]["instance_id"] for idx in selected_indices]


def make_latex_table(test_metrics: dict[str, Any], early: dict[str, Any], output: Path) -> None:
    rows = [
        ("Best epoch", f"{int(test_metrics.get('best_epoch', 0))}"),
        ("Stop epoch", f"{int(early.get('stop_epoch', 0))}"),
        ("Test signal F1", f"{finite_float(test_metrics.get('signal_f1')):.3f}"),
        ("Test signal precision", f"{finite_float(test_metrics.get('signal_precision')):.3f}"),
        ("Test signal recall", f"{finite_float(test_metrics.get('signal_recall')):.3f}"),
        ("Test expression AUROC", f"{finite_float(test_metrics.get('expression_auc')):.3f}"),
        ("Test masked MAE", f"{finite_float(test_metrics.get('masked_mae')):.3f}"),
        ("Test PSNR", f"{finite_float(test_metrics.get('image_psnr')):.2f} dB"),
    ]
    lines = ["\\begin{tabular}{ll}", "\\toprule", "Quantity & Value \\\\", "\\midrule"]
    for key, value in rows:
        lines.append(f"{key} & {value} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_dataset_summary_table(per_example_rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_example_rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for dataset, rows in sorted(grouped.items()):
        target_signal = np.asarray([row["target_signal_fraction"] for row in rows], dtype=np.float64)
        pred_signal = np.asarray([row["pred_signal_fraction"] for row in rows], dtype=np.float64)
        f1 = np.asarray([row["signal_f1"] for row in rows], dtype=np.float64)
        masked_mae = np.asarray([row["masked_mae"] for row in rows], dtype=np.float64)
        summary.append(
            {
                "dataset": dataset,
                "n": len(rows),
                "target_signal_mean": float(np.nanmean(target_signal)),
                "pred_signal_mean": float(np.nanmean(pred_signal)),
                "signal_f1_mean": float(np.nanmean(f1)),
                "masked_mae_mean": float(np.nanmean(masked_mae)),
            }
        )
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & $n$ & True signal & Pred signal & Signal F1 & Masked MAE \\\\",
        "\\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['dataset']} & {row['n']} & {row['target_signal_mean']:.3f} & "
            f"{row['pred_signal_mean']:.3f} & {row['signal_f1_mean']:.3f} & {row['masked_mae_mean']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_summary(
    output: Path,
    *,
    run_root: Path,
    config: dict[str, Any],
    early: dict[str, Any],
    test_metrics: dict[str, Any],
    per_example_rows: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    selected_examples: Sequence[str],
    selected_saliency: Sequence[str],
) -> None:
    target_mean = np.asarray([row["target_mean_masked"] for row in per_example_rows], dtype=np.float64)
    pred_mean = np.asarray([row["pred_mean_masked"] for row in per_example_rows], dtype=np.float64)
    corr = float(np.corrcoef(target_mean, pred_mean)[0, 1]) if np.std(target_mean) > 1e-8 and np.std(pred_mean) > 1e-8 else float("nan")
    write_json(
        output,
        {
            "run_root": str(run_root),
            "architecture": config.get("args", {}).get("architecture"),
            "image_size": config.get("args", {}).get("image_size"),
            "train_count": config.get("train_count"),
            "val_count": config.get("val_count"),
            "test_count": config.get("test_count"),
            "early_stop": early,
            "test_metrics": test_metrics,
            "test_pred_target_mean_corr": corr,
            "per_example_rows": len(per_example_rows),
            "dataset_summary": dataset_summary,
            "selected_prediction_examples": list(selected_examples),
            "selected_saliency_examples": list(selected_saliency),
        },
    )


def main() -> int:
    args = parse_args()
    figures = args.output_root / "figures"
    tables = args.output_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    test_metrics = load_json(args.run_root / "test_metrics.json")
    early = load_json(args.run_root / "early_stop_summary.json")
    metrics = load_metrics(args.run_root / "metrics.jsonl")
    run_args = argparse.Namespace(**config["args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run_args).to(device)
    checkpoint = torch.load(args.run_root / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    plot_training(metrics, early, figures / "fig1_training_convergence.png")
    plot_test_summary(test_metrics, figures / "fig2_test_quantification.png")
    shutil.copy2(args.run_root / "predictions" / "test_best.png", figures / "fig3_test_best_panel.png")

    manifest_path = Path(config["args"]["data_root"]) / "manifests" / "projected_instances_manifest.csv"
    manifest_rows = read_csv(manifest_path)
    test_dataset = B2FDataset.from_manifest(manifest_path, "test", int(run_args.image_size), path_mode=str(run_args.path_mode))
    per_example = collect_predictions(model, test_dataset, args, device)
    save_csv(tables / "test_per_instance_quantification.csv", per_example)
    plot_prediction_scatter(per_example, figures / "fig4_prediction_calibration.png")
    selected_examples = make_prediction_examples(
        model,
        manifest_rows,
        per_example,
        figures / "fig5_prediction_error_examples.png",
        args=args,
        run_args=run_args,
        device=device,
    )
    selected_saliency = make_saliency_examples(
        model,
        manifest_rows,
        per_example,
        figures / "fig6_saliency_explanation.png",
        args=args,
        run_args=run_args,
        device=device,
    )
    make_latex_table(test_metrics, early, tables / "test_metrics_table.tex")
    dataset_summary = make_dataset_summary_table(per_example, tables / "test_dataset_summary_table.tex")
    write_summary(
        args.output_root / "artifact_summary.json",
        run_root=args.run_root,
        config=config,
        early=early,
        test_metrics=test_metrics,
        per_example_rows=per_example,
        dataset_summary=dataset_summary,
        selected_examples=selected_examples,
        selected_saliency=selected_saliency,
    )
    print(json.dumps({"stage": "b2f_artifacts_finished", "output_root": str(args.output_root)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
