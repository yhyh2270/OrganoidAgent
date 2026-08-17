#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.datasets import FEATURE_COLUMNS, feature_vector
from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    average_precision,
    binary_auc,
    coerce_float,
    coerce_int,
    normalize_feature_matrix,
    pearson_corr,
    read_csv,
    set_seed,
    write_csv,
    write_json,
)


class FeatureNet(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether morphology features explain same-time fluorescence positivity/intensity.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1_feature_analysis")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260510)
    return parser.parse_args()


def split_arrays(rows: list[dict[str, str]], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["split"] == split]
    x = np.asarray([feature_vector(row) for row in selected], dtype=np.float32)
    y_pos = np.asarray([coerce_int(row["fluorescence_positive"]) for row in selected], dtype=np.float32)
    y_peak = np.asarray([coerce_float(row["fl_corrected_p90_log"]) for row in selected], dtype=np.float32)
    return x, y_pos, y_peak


def evaluate(model: FeatureNet, x: np.ndarray, y_pos: np.ndarray, y_peak: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x).to(device)).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-out[:, 0]))
    peak_pred = out[:, 1]
    return {
        "positive_auc": binary_auc(y_pos, scores),
        "positive_ap": average_precision(y_pos, scores),
        "peak_pearson": pearson_corr(y_peak, peak_pred),
        "n": int(len(y_pos)),
    }


def permutation_importance(
    model: FeatureNet,
    x: np.ndarray,
    y_pos: np.ndarray,
    y_peak: np.ndarray,
    device: torch.device,
    baseline: dict[str, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    rng = np.random.default_rng(20260510)
    for idx, name in enumerate(FEATURE_COLUMNS):
        x_perm = x.copy()
        x_perm[:, idx] = rng.permutation(x_perm[:, idx])
        metrics = evaluate(model, x_perm, y_pos, y_peak, device)
        rows.append(
            {
                "feature": name,
                "baseline_auc": baseline["positive_auc"],
                "permuted_auc": metrics["positive_auc"],
                "auc_drop": baseline["positive_auc"] - metrics["positive_auc"],
                "baseline_peak_pearson": baseline["peak_pearson"],
                "permuted_peak_pearson": metrics["peak_pearson"],
                "peak_pearson_drop": baseline["peak_pearson"] - metrics["peak_pearson"],
            }
        )
    return sorted(rows, key=lambda row: float(row["auc_drop"]), reverse=True)


def plot_importance(rows: list[dict[str, float | str]], output: Path) -> None:
    top = rows[: min(12, len(rows))]
    names = [str(row["feature"]) for row in top][::-1]
    values = [float(row["auc_drop"]) for row in top][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(names, values)
    plt.xlabel("AUROC drop after feature permutation")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_root / "manifests" / "projected_instances_manifest.csv"
    rows = read_csv(manifest_path)
    x_train_raw, y_train_pos, y_train_peak = split_arrays(rows, "train")
    x_val_raw, y_val_pos, y_val_peak = split_arrays(rows, "val")
    x_test_raw, y_test_pos, y_test_peak = split_arrays(rows, "test")
    x_train, mean, std = normalize_feature_matrix(x_train_raw)
    x_val = ((np.nan_to_num(x_val_raw) - mean) / std).astype(np.float32)
    x_test = ((np.nan_to_num(x_test_raw) - mean) / std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureNet(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(y_train_pos) - y_train_pos.sum()) / max(y_train_pos.sum(), 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber = nn.SmoothL1Loss()
    x_train_t = torch.from_numpy(x_train).to(device)
    y_pos_t = torch.from_numpy(y_train_pos).to(device)
    y_peak_t = torch.from_numpy(y_train_peak).to(device)
    metrics_log: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x_train_t)
        loss = bce(out[:, 0], y_pos_t) + 0.20 * huber(out[:, 1], y_peak_t)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, x_val, y_val_pos, y_val_peak, device)
            row = {"epoch": epoch, "loss": float(loss.detach().cpu()), **{f"val_{k}": v for k, v in val_metrics.items()}}
            metrics_log.append(row)
            print(json.dumps(row), flush=True)
    test_metrics = evaluate(model, x_test, y_test_pos, y_test_peak, device)
    importance = permutation_importance(model, x_test, y_test_pos, y_test_peak, device, test_metrics)
    write_json(
        args.output_root / "summary.json",
        {
            "test_metrics": test_metrics,
            "feature_columns": FEATURE_COLUMNS,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "train_count": int(len(y_train_pos)),
            "val_count": int(len(y_val_pos)),
            "test_count": int(len(y_test_pos)),
        },
    )
    write_csv(args.output_root / "feature_importance.csv", importance)
    write_csv(args.output_root / "metrics.csv", metrics_log)
    plot_importance(importance, args.output_root / "plots" / "feature_importance_auc_drop.png")
    torch.save({"model": model.state_dict(), "feature_mean": mean, "feature_std": std, "test_metrics": test_metrics}, args.output_root / "feature_model.pt")
    print(json.dumps({"stage": "feature_analysis_finished", "test": test_metrics, "top_features": importance[:5]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
