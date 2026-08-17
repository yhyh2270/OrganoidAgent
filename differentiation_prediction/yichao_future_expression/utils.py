from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTED_DB = (
    REPO_ROOT
    / "analysis-outputs"
    / "yichao_projected_instance_pairs"
    / "database"
    / "projected_instance_pairs.sqlite"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis-outputs" / "yichao_future_expression"


def stable_fraction(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_from_group(group_key: str) -> str:
    fraction = stable_fraction(group_key)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def image_to_tensor(path: Path, size: int, *, mask: bool = False) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (size, size):
            resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
            image = image.resize((size, size), resample=resample)
        arr = np.asarray(image, dtype=np.float32)
    if mask:
        arr = (arr > 0).astype(np.float32)
    else:
        arr = arr / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def resize_grayscale(src: Path, dst: Path, size: int, *, mask: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            with Image.open(dst) as existing:
                if existing.size == (size, size):
                    return
        except OSError:
            pass
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    with Image.open(src) as image:
        image = image.convert("L")
        resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
        image = image.resize((size, size), resample=resample)
        image.save(tmp)
    tmp.replace(dst)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    arr = (tensor.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_green(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    zero = Image.new("L", gray.size, 0)
    return Image.merge("RGB", (zero, gray, zero))


def save_b2f_panel(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    path: Path,
    *,
    keys: Sequence[str] | None = None,
    max_items: int = 8,
) -> None:
    rows = min(max_items, inputs.shape[0])
    tile = inputs.shape[-1]
    label_h = 28
    header_h = 32
    width = tile * 3
    height = header_h + rows * (tile + label_h)
    canvas = Image.new("RGB", (width, height), (18, 21, 24))
    draw = ImageDraw.Draw(canvas)
    font = load_font(13)
    for col, label in enumerate(("brightfield", "predicted F", "true F")):
        draw.text((col * tile + 8, 8), label, fill=(230, 235, 238), font=font)
    for i in range(rows):
        y = header_h + i * (tile + label_h)
        imgs = [
            to_uint8_image(inputs[i]),
            make_green(to_uint8_image(preds[i])),
            make_green(to_uint8_image(targets[i])),
        ]
        for col, image in enumerate(imgs):
            canvas.paste(image.convert("RGB"), (col * tile, y))
        if keys:
            draw.text((6, y + tile + 6), str(keys[i])[:120], fill=(170, 180, 186), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def binary_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid]
    s = s[valid]
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: Sequence[float], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid] > 0.5
    s = s[valid]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_metric_lines(metrics: list[dict[str, Any]], path: Path, keys: Iterable[str]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not metrics:
        return
    epochs = [int(row["epoch"]) for row in metrics]
    plt.figure(figsize=(9, 5))
    for key in keys:
        values = [row.get(key, float("nan")) for row in metrics]
        if any(np.isfinite(values)):
            plt.plot(epochs, values, marker="o", label=key)
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def write_rows_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_feature_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std[std < 1e-6] = 1.0
    norm = (np.nan_to_num(matrix, nan=0.0) - mean) / std
    return norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def safe_log1p_positive(value: float) -> float:
    return float(math.log1p(max(0.0, value)))
