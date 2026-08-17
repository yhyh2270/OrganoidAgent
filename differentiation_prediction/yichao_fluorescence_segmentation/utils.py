from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_B2F_ROOT = REPO_ROOT / "analysis-outputs" / "yichao_future_expression"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis-outputs" / "yichao_fluorescence_segmentation"


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
        writer.writerows(rows)


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


def read_gray_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return arr


def save_gray_uint8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(arr, 0.0, 1.0)
    image = Image.fromarray((clipped * 255.0).round().astype(np.uint8), mode="L")
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    image.save(tmp)
    tmp.replace(path)


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


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def gray_rgb(arr: np.ndarray) -> Image.Image:
    clipped = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((clipped * 255).round().astype(np.uint8), mode="L").convert("RGB")


def green_rgb(arr: np.ndarray) -> Image.Image:
    gray = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8), mode="L")
    zero = Image.new("L", gray.size, 0)
    return Image.merge("RGB", (zero, gray, zero))


def red_rgb(arr: np.ndarray) -> Image.Image:
    gray = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8), mode="L")
    zero = Image.new("L", gray.size, 0)
    return Image.merge("RGB", (gray, zero, zero))


def heat_rgb(arr: np.ndarray) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        norm = np.zeros_like(data)
    else:
        lo = float(np.percentile(data[finite], 1))
        hi = float(np.percentile(data[finite], 99))
        norm = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (plt.get_cmap("magma")(norm)[..., :3] * 255).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_grid(
    path: Path,
    rows: Sequence[Sequence[Image.Image]],
    headers: Sequence[str],
    labels: Sequence[str],
    *,
    tile: int = 192,
) -> None:
    if not rows:
        return
    font = load_font(13)
    label_h = 30
    header_h = 30
    width = tile * len(headers)
    height = header_h + len(rows) * (tile + label_h)
    canvas = Image.new("RGB", (width, height), (18, 21, 24))
    draw = ImageDraw.Draw(canvas)
    for col, header in enumerate(headers):
        draw.text((col * tile + 8, 8), header, fill=(235, 240, 242), font=font)
    for row_index, images in enumerate(rows):
        y = header_h + row_index * (tile + label_h)
        for col, image in enumerate(images):
            canvas.paste(image.resize((tile, tile), Image.Resampling.BILINEAR), (col * tile, y))
        draw.text((8, y + tile + 6), labels[row_index][:120], fill=(185, 194, 199), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def binary_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid] > 0.5
    s = s[valid]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    rank_sum_pos = ranks[y].sum()
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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    b2 = beta * beta
    return safe_div((1 + b2) * precision * recall, b2 * precision + recall)


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
