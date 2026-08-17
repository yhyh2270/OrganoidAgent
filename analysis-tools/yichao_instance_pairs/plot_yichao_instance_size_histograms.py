#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from size_metadata import QUANTILE_POINTS, quantile_summary


PLOT_SPECS = (
    ("square_crop_size_px", "Square Crop Size (px)"),
    ("area_px", "Instance Area (px)"),
    ("crop_area_px", "Crop Area (px^2)"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_db = repo_root / "analysis-outputs" / "yichao_instance_pairs" / "database" / "instance_pairs.sqlite"
    default_out = repo_root / "analysis-outputs" / "yichao_instance_size_analysis"
    parser = argparse.ArgumentParser(description="Plot updated Yichao instance size histograms and quantiles.")
    parser.add_argument("--db-path", default=str(default_db))
    parser.add_argument("--output-root", default=str(default_out))
    return parser.parse_args()


def fetch_metric(db_path: Path, metric: str) -> np.ndarray:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"SELECT CAST({metric} AS REAL) FROM instances").fetchall()
    finally:
        conn.close()
    return np.asarray([float(row[0]) for row in rows], dtype=np.float64)


def add_quantile_lines(ax: plt.Axes, quantiles: dict[str, float], labels: list[str]) -> None:
    for label in labels:
        value = quantiles[label]
        ax.axvline(value, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(value, ax.get_ylim()[1] * 0.96, label, rotation=90, va="top", ha="right", fontsize=8)


def plot_triptych(values: np.ndarray, metric: str, title: str, output_root: Path) -> Path:
    quantiles = quantile_summary(values)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(values, bins=120, color="#4C78A8", alpha=0.85)
    axes[0].set_title(f"{title} Full")
    axes[0].set_xlabel(title)
    axes[0].set_ylabel("Count")
    add_quantile_lines(axes[0], quantiles, ["p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"])

    lower_max = max(quantiles["p25"], quantiles["p05"] * 1.05)
    lower_values = values[values <= lower_max]
    axes[1].hist(lower_values, bins=80, color="#59A14F", alpha=0.85)
    axes[1].set_xlim(quantiles["p00"], lower_max)
    axes[1].set_title(f"{title} Lower Tail")
    axes[1].set_xlabel(title)
    axes[1].set_ylabel("Count")
    add_quantile_lines(axes[1], quantiles, ["p01", "p025", "p05", "p10", "p25"])

    upper_min = min(quantiles["p75"], quantiles["p95"])
    upper_values = values[values >= upper_min]
    axes[2].hist(upper_values, bins=80, color="#E15759", alpha=0.85)
    axes[2].set_xlim(upper_min, quantiles["p100"])
    axes[2].set_title(f"{title} Upper Tail")
    axes[2].set_xlabel(title)
    axes[2].set_ylabel("Count")
    add_quantile_lines(axes[2], quantiles, ["p75", "p90", "p95", "p975", "p99", "p100"])

    fig.suptitle(f"Yichao Instance Size Distribution: {title}", fontsize=14)
    fig.tight_layout()

    path = output_root / f"{metric}_histogram_triptych.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    quantile_payload: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(db_path),
        "plots": {},
        "quantiles": {},
    }

    for metric, title in PLOT_SPECS:
        values = fetch_metric(db_path, metric)
        quantile_payload["quantiles"][metric] = quantile_summary(values)
        quantile_payload["plots"][metric] = str(plot_triptych(values, metric, title, output_root))

    quantile_json = output_root / "quantiles.json"
    quantile_json.write_text(json.dumps(quantile_payload, indent=2), encoding="utf-8")

    print(output_root)
    print(quantile_json)
    for metric, path in quantile_payload["plots"].items():
        print(f"{metric}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
