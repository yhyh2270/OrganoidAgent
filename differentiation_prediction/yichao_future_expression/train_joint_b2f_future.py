#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.datasets import (
    FEATURE_COLUMNS,
    feature_vector,
    fit_feature_stats,
    fit_target_stats,
    load_future_data,
)
from differentiation_prediction.yichao_future_expression.models import B2FMultiTaskUNet, SmallImageEncoder
from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    average_precision,
    binary_auc,
    coerce_float,
    coerce_int,
    image_to_tensor,
    load_font,
    pearson_corr,
    plot_metric_lines,
    read_csv,
    set_seed,
    to_uint8_image,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train joint same-time B2F plus future fluorescence prediction.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage3_joint_b2f_future")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-prefix-frames", type=int, default=5)
    parser.add_argument("--target-policy", choices=["last_future", "peak_future"], default="last_future")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--b2f-base-channels", type=int, default=16)
    parser.add_argument("--future-decoder-base", type=int, default=16)
    parser.add_argument("--current-image-weight", type=float, default=0.50)
    parser.add_argument("--scalar-weight", type=float, default=0.25)
    parser.add_argument("--current-scalar-weight", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--feature-ablation", action="store_true")
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


class JointFutureDataset(Dataset):
    def __init__(
        self,
        sample_rows: Sequence[dict[str, str]],
        instance_by_id: dict[str, dict[str, str]],
        image_size: int,
        max_prefix_frames: int,
        feature_mean: Sequence[float],
        feature_std: Sequence[float],
        peak_mean: float,
        peak_std: float,
        auc_mean: float,
        auc_std: float,
        target_policy: str,
        *,
        augment: bool = False,
    ) -> None:
        self.sample_rows = [row for row in sample_rows if self._is_valid(row, instance_by_id)]
        self.instance_by_id = instance_by_id
        self.image_size = image_size
        self.max_prefix_frames = max_prefix_frames
        self.feature_mean = torch.tensor(feature_mean, dtype=torch.float32)
        self.feature_std = torch.tensor(feature_std, dtype=torch.float32).clamp_min(1e-6)
        self.peak_mean = float(peak_mean)
        self.peak_std = max(float(peak_std), 1e-6)
        self.auc_mean = float(auc_mean)
        self.auc_std = max(float(auc_std), 1e-6)
        self.target_policy = target_policy
        self.augment = augment

    @staticmethod
    def _is_valid(row: dict[str, str], instance_by_id: dict[str, dict[str, str]]) -> bool:
        try:
            prefix_ids = json.loads(row["prefix_instance_ids_json"])
            future_ids = json.loads(row["future_instance_ids_json"])
        except (KeyError, json.JSONDecodeError, TypeError):
            return False
        if not prefix_ids or not future_ids:
            return False
        return all(instance_id in instance_by_id for instance_id in prefix_ids + future_ids)

    def __len__(self) -> int:
        return len(self.sample_rows)

    def choose_target_id(self, future_ids: Sequence[str]) -> str:
        if self.target_policy == "peak_future":
            return max(future_ids, key=lambda instance_id: coerce_float(self.instance_by_id[instance_id].get("fl_corrected_p90")))
        return future_ids[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.sample_rows[index]
        prefix_ids = json.loads(sample["prefix_instance_ids_json"])[-self.max_prefix_frames :]
        future_ids = json.loads(sample["future_instance_ids_json"])
        current_id = prefix_ids[-1]
        target_id = self.choose_target_id(future_ids)
        current_row = self.instance_by_id[current_id]
        target_row = self.instance_by_id[target_id]

        frames: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        valid: list[float] = []
        for instance_id in prefix_ids:
            row = self.instance_by_id[instance_id]
            frames.append(image_to_tensor(Path(row["brightfield_256_path"]), self.image_size))
            raw_feature = torch.tensor(feature_vector(row), dtype=torch.float32)
            features.append((raw_feature - self.feature_mean) / self.feature_std)
            valid.append(1.0)
        while len(frames) < self.max_prefix_frames:
            frames.insert(0, torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32))
            features.insert(0, torch.zeros_like(self.feature_mean))
            valid.insert(0, 0.0)

        frame_tensor = torch.stack(frames, dim=0)
        feature_tensor = torch.stack(features, dim=0)
        valid_tensor = torch.tensor(valid, dtype=torch.float32)

        current_brightfield = image_to_tensor(Path(current_row["brightfield_256_path"]), self.image_size)
        current_fluorescence = image_to_tensor(Path(current_row["fluorescence_256_path"]), self.image_size)
        current_mask = image_to_tensor(Path(current_row["mask_256_path"]), self.image_size, mask=True)
        future_fluorescence = image_to_tensor(Path(target_row["fluorescence_256_path"]), self.image_size)
        future_mask = image_to_tensor(Path(target_row["mask_256_path"]), self.image_size, mask=True)

        if self.augment:
            do_h = bool(torch.rand(()) < 0.5)
            do_v = bool(torch.rand(()) < 0.5)
            if do_h:
                frame_tensor = torch.flip(frame_tensor, dims=[3])
                current_brightfield = torch.flip(current_brightfield, dims=[2])
                current_fluorescence = torch.flip(current_fluorescence, dims=[2])
                current_mask = torch.flip(current_mask, dims=[2])
                future_fluorescence = torch.flip(future_fluorescence, dims=[2])
                future_mask = torch.flip(future_mask, dims=[2])
            if do_v:
                frame_tensor = torch.flip(frame_tensor, dims=[2])
                current_brightfield = torch.flip(current_brightfield, dims=[1])
                current_fluorescence = torch.flip(current_fluorescence, dims=[1])
                current_mask = torch.flip(current_mask, dims=[1])
                future_fluorescence = torch.flip(future_fluorescence, dims=[1])
                future_mask = torch.flip(future_mask, dims=[1])

        peak = coerce_float(sample["future_peak_log"])
        auc = coerce_float(sample["future_auc_log"])
        prefix_end = coerce_int(sample["prefix_end_time_index"])
        target_time = coerce_int(target_row.get("time_index"))
        return {
            "frames": frame_tensor,
            "features": feature_tensor,
            "valid": valid_tensor,
            "current_brightfield": current_brightfield,
            "current_fluorescence": current_fluorescence,
            "current_mask": current_mask,
            "current_positive": torch.tensor(float(coerce_int(current_row["fluorescence_positive"])), dtype=torch.float32),
            "current_peak": torch.tensor(coerce_float(current_row["fl_corrected_p90_log"]), dtype=torch.float32),
            "current_total": torch.tensor(coerce_float(current_row["fl_corrected_total_log"]), dtype=torch.float32),
            "future_fluorescence": future_fluorescence,
            "future_mask": future_mask,
            "future_positive": torch.tensor(float(coerce_int(sample["future_positive"])), dtype=torch.float32),
            "future_peak": torch.tensor((peak - self.peak_mean) / self.peak_std, dtype=torch.float32),
            "future_auc": torch.tensor((auc - self.auc_mean) / self.auc_std, dtype=torch.float32),
            "future_peak_raw": torch.tensor(peak, dtype=torch.float32),
            "future_auc_raw": torch.tensor(auc, dtype=torch.float32),
            "future_sample_id": sample["future_sample_id"],
            "track_id": sample["track_id"],
            "dataset": sample["dataset"],
            "current_instance_id": current_id,
            "target_instance_id": target_id,
            "prefix_length": torch.tensor(coerce_int(sample["prefix_length"]), dtype=torch.int64),
            "prefix_end_time_index": torch.tensor(prefix_end, dtype=torch.int64),
            "target_time_index": torch.tensor(target_time, dtype=torch.int64),
            "horizon": torch.tensor(max(0, target_time - prefix_end), dtype=torch.int64),
        }


class FutureImageDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 160, base_channels: int = 16, output_size: int = 256) -> None:
        super().__init__()
        c = base_channels
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, c * 8 * 8 * 8),
            nn.SiLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 6, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 6),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c * 6, c * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 3, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 3),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c * 3, c * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid(),
        )
        self.base_channels = c
        self.output_size = output_size

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.fc(hidden).reshape(hidden.shape[0], self.base_channels * 8, 8, 8)
        x = self.net(x)
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return x


class JointB2FFutureModel(nn.Module):
    def __init__(self, feature_dim: int, b2f_base_channels: int = 16, future_decoder_base: int = 16, hidden_dim: int = 160, image_size: int = 256) -> None:
        super().__init__()
        self.image_encoder = SmallImageEncoder(embedding_dim=128, base_channels=24)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )
        self.gru = nn.GRU(128 + 64, hidden_dim, batch_first=True)
        self.future_decoder = FutureImageDecoder(hidden_dim=hidden_dim, base_channels=future_decoder_base, output_size=image_size)
        self.future_scalar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 3),
        )
        self.b2f_aux = B2FMultiTaskUNet(base_channels=b2f_base_channels)

    def encode_sequence(self, frames: torch.Tensor, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, height, width = frames.shape
        flat = frames.reshape(batch * time_steps, channels, height, width)
        image_embeddings = self.image_encoder(flat).reshape(batch, time_steps, -1)
        feature_embeddings = self.feature_encoder(features)
        sequence = torch.cat([image_embeddings, feature_embeddings], dim=-1) * valid.unsqueeze(-1)
        lengths = valid.sum(dim=1).clamp_min(1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, frames: torch.Tensor, features: torch.Tensor, valid: torch.Tensor, current_brightfield: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode_sequence(frames, features, valid)
        current_pred, current_scalar = self.b2f_aux(current_brightfield)
        return {
            "future_image": self.future_decoder(hidden),
            "future_scalar": self.future_scalar_head(hidden),
            "current_image": current_pred,
            "current_scalar": current_scalar,
        }


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
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


def build_datasets(args: argparse.Namespace) -> tuple[JointFutureDataset, JointFutureDataset, JointFutureDataset, dict[str, Any]]:
    manifest_path = args.data_root / "manifests" / "projected_instances_manifest.csv"
    future_path = args.data_root / "manifests" / "future_samples.csv"
    instance_rows, future_rows, instance_by_id = load_future_data(manifest_path, future_path)
    feature_mean, feature_std = fit_feature_stats(instance_rows, "train")
    peak_mean, peak_std, auc_mean, auc_std = fit_target_stats(future_rows, "train")

    def make(split: str, augment: bool) -> JointFutureDataset:
        rows = [row for row in future_rows if row["split"] == split]
        return JointFutureDataset(
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
            target_policy=args.target_policy,
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
        "target_policy": args.target_policy,
    }
    return make("train", True), make("val", False), make("test", False), stats


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inside = torch.abs(pred - target) * mask
    outside = torch.abs(pred - target) * (1.0 - mask)
    inside_loss = inside.sum() / mask.sum().clamp_min(1.0)
    outside_loss = outside.sum() / (1.0 - mask).sum().clamp_min(1.0)
    return inside_loss + 0.20 * outside_loss


def loss_fn(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    future_bce: nn.Module,
    current_bce: nn.Module,
    huber: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    future_image_loss = masked_l1(outputs["future_image"], batch["future_fluorescence"], batch["future_mask"])
    current_image_loss = masked_l1(outputs["current_image"], batch["current_fluorescence"], batch["current_mask"])

    fs = outputs["future_scalar"]
    future_pos_loss = future_bce(fs[:, 0], batch["future_positive"])
    future_peak_loss = huber(fs[:, 1], batch["future_peak"])
    future_auc_loss = huber(fs[:, 2], batch["future_auc"])
    future_scalar_loss = future_pos_loss + 0.30 * future_peak_loss + 0.20 * future_auc_loss

    cs = outputs["current_scalar"]
    current_pos_loss = current_bce(cs[:, 0], batch["current_positive"])
    current_peak_loss = huber(cs[:, 1], batch["current_peak"])
    current_total_loss = huber(cs[:, 2], batch["current_total"])
    current_scalar_loss = current_pos_loss + 0.20 * current_peak_loss + 0.05 * current_total_loss

    loss = (
        future_image_loss
        + args.current_image_weight * current_image_loss
        + args.scalar_weight * future_scalar_loss
        + args.current_scalar_weight * current_scalar_loss
    )
    return loss, {
        "loss_future_image": float(future_image_loss.detach().cpu()),
        "loss_current_image": float(current_image_loss.detach().cpu()),
        "loss_future_scalar": float(future_scalar_loss.detach().cpu()),
        "loss_current_scalar": float(current_scalar_loss.detach().cpu()),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def evaluate(
    model: JointB2FFutureModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    feature_ablate_index: int | None = None,
    panel_path: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    labels: list[float] = []
    scores: list[float] = []
    peak_true: list[float] = []
    peak_pred: list[float] = []
    auc_true: list[float] = []
    auc_pred: list[float] = []
    future_mae_sum = 0.0
    future_mse_sum = 0.0
    current_mae_sum = 0.0
    count = 0
    first_panel_payload: tuple[dict[str, Any], dict[str, torch.Tensor]] | None = None

    for batch_index, batch in enumerate(loader):
        tensor_batch = move_batch({k: v for k, v in batch.items() if torch.is_tensor(v)}, device)
        if feature_ablate_index is not None:
            tensor_batch["features"] = tensor_batch["features"].clone()
            tensor_batch["features"][:, :, feature_ablate_index] = 0.0
        outputs = model(
            tensor_batch["frames"],
            tensor_batch["features"],
            tensor_batch["valid"],
            tensor_batch["current_brightfield"],
        )
        future_err = torch.abs(outputs["future_image"] - tensor_batch["future_fluorescence"])
        current_err = torch.abs(outputs["current_image"] - tensor_batch["current_fluorescence"])
        per_future_mae = future_err.flatten(1).mean(dim=1).detach().cpu().numpy()
        per_future_mse = ((outputs["future_image"] - tensor_batch["future_fluorescence"]) ** 2).flatten(1).mean(dim=1).detach().cpu().numpy()
        per_future_masked_mae = (
            (future_err * tensor_batch["future_mask"]).flatten(1).sum(dim=1)
            / tensor_batch["future_mask"].flatten(1).sum(dim=1).clamp_min(1.0)
        ).detach().cpu().numpy()
        per_current_mae = current_err.flatten(1).mean(dim=1).detach().cpu().numpy()
        prob = torch.sigmoid(outputs["future_scalar"][:, 0]).detach().cpu().numpy()
        pred_peak = (outputs["future_scalar"][:, 1].detach().cpu().numpy() * loader.dataset.peak_std) + loader.dataset.peak_mean
        pred_auc = (outputs["future_scalar"][:, 2].detach().cpu().numpy() * loader.dataset.auc_std) + loader.dataset.auc_mean
        true_peak = batch["future_peak_raw"].numpy()
        true_auc = batch["future_auc_raw"].numpy()
        batch_size = len(prob)
        future_mae_sum += float(per_future_mae.sum())
        future_mse_sum += float(per_future_mse.sum())
        current_mae_sum += float(per_current_mae.sum())
        count += batch_size
        labels.extend(batch["future_positive"].numpy().tolist())
        scores.extend(prob.tolist())
        peak_true.extend(true_peak.tolist())
        peak_pred.extend(pred_peak.tolist())
        auc_true.extend(true_auc.tolist())
        auc_pred.extend(pred_auc.tolist())
        for i in range(batch_size):
            rows.append(
                {
                    "future_sample_id": batch["future_sample_id"][i],
                    "track_id": batch["track_id"][i],
                    "dataset": batch["dataset"][i],
                    "current_instance_id": batch["current_instance_id"][i],
                    "target_instance_id": batch["target_instance_id"][i],
                    "prefix_length": int(batch["prefix_length"][i]),
                    "prefix_end_time_index": int(batch["prefix_end_time_index"][i]),
                    "target_time_index": int(batch["target_time_index"][i]),
                    "horizon": int(batch["horizon"][i]),
                    "future_positive": float(batch["future_positive"][i]),
                    "pred_future_positive_prob": float(prob[i]),
                    "future_peak_log": float(true_peak[i]),
                    "pred_future_peak_log": float(pred_peak[i]),
                    "future_auc_log": float(true_auc[i]),
                    "pred_future_auc_log": float(pred_auc[i]),
                    "future_image_mae": float(per_future_mae[i]),
                    "future_masked_image_mae": float(per_future_masked_mae[i]),
                    "current_image_mae": float(per_current_mae[i]),
                }
            )
        if panel_path is not None and first_panel_payload is None:
            first_panel_payload = (batch, {key: value.detach().cpu() for key, value in outputs.items()})
        if args.limit_eval_batches is not None and batch_index + 1 >= args.limit_eval_batches:
            break

    mse = future_mse_sum / max(count, 1)
    metrics = {
        "n": count,
        "future_positive_count": int(sum(1 for value in labels if value > 0.5)),
        "future_image_mae": future_mae_sum / max(count, 1),
        "future_image_mse": mse,
        "future_image_psnr": float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-12)))),
        "current_image_mae": current_mae_sum / max(count, 1),
        "future_positive_auc": binary_auc(labels, scores),
        "future_positive_ap": average_precision(labels, scores),
        "future_peak_pearson": pearson_corr(peak_true, peak_pred),
        "future_auc_pearson": pearson_corr(auc_true, auc_pred),
        "future_peak_mae": float(np.mean(np.abs(np.asarray(peak_true) - np.asarray(peak_pred)))) if peak_true else float("nan"),
        "future_auc_mae": float(np.mean(np.abs(np.asarray(auc_true) - np.asarray(auc_pred)))) if auc_true else float("nan"),
    }
    if panel_path is not None and first_panel_payload is not None:
        save_joint_panel(first_panel_payload[0], first_panel_payload[1], panel_path)
    return metrics, rows


def make_green(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    zero = Image.new("L", gray.size, 0)
    return Image.merge("RGB", (zero, gray, zero))


def save_joint_panel(batch: dict[str, Any], outputs: dict[str, torch.Tensor], path: Path, max_items: int = 6) -> None:
    rows = min(max_items, outputs["future_image"].shape[0])
    tile = outputs["future_image"].shape[-1]
    label_h = 30
    header_h = 32
    cols = ["B_k", "pred F_k", "true F_k", "pred F_future", "true F_future"]
    canvas = Image.new("RGB", (tile * len(cols), header_h + rows * (tile + label_h)), (18, 21, 24))
    draw = ImageDraw.Draw(canvas)
    font = load_font(13)
    for col, label in enumerate(cols):
        draw.text((col * tile + 8, 8), label, fill=(230, 235, 238), font=font)
    for i in range(rows):
        y = header_h + i * (tile + label_h)
        images = [
            to_uint8_image(batch["current_brightfield"][i]),
            make_green(to_uint8_image(outputs["current_image"][i])),
            make_green(to_uint8_image(batch["current_fluorescence"][i])),
            make_green(to_uint8_image(outputs["future_image"][i])),
            make_green(to_uint8_image(batch["future_fluorescence"][i])),
        ]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (col * tile, y))
        label = f"{batch['dataset'][i]} | L={int(batch['prefix_length'][i])} | h={int(batch['horizon'][i])} | y={float(batch['future_positive'][i]):.0f}"
        draw.text((6, y + tile + 6), label[:160], fill=(170, 180, 186), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def grouped_metrics(rows: Sequence[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    out: list[dict[str, Any]] = []
    for group, items in sorted(groups.items(), key=lambda kv: kv[0]):
        labels = [float(row["future_positive"]) for row in items]
        scores = [float(row["pred_future_positive_prob"]) for row in items]
        peaks = [float(row["future_peak_log"]) for row in items]
        pred_peaks = [float(row["pred_future_peak_log"]) for row in items]
        out.append(
            {
                group_key: group,
                "n": len(items),
                "positive_count": int(sum(labels)),
                "future_positive_auc": binary_auc(labels, scores),
                "future_positive_ap": average_precision(labels, scores),
                "future_peak_pearson": pearson_corr(peaks, pred_peaks),
                "future_image_mae": float(np.mean([float(row["future_image_mae"]) for row in items])),
                "future_masked_image_mae": float(np.mean([float(row["future_masked_image_mae"]) for row in items])),
            }
        )
    return out


def add_horizon_bin(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        horizon = int(row["horizon"])
        if horizon <= 1:
            row["horizon_bin"] = "01_short"
        elif horizon <= 3:
            row["horizon_bin"] = "02_mid"
        else:
            row["horizon_bin"] = "03_long"


def plot_predictions(rows: Sequence[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    y = np.asarray([float(row["future_positive"]) for row in rows])
    score = np.asarray([float(row["pred_future_positive_prob"]) for row in rows])
    peak_true = np.asarray([float(row["future_peak_log"]) for row in rows])
    peak_pred = np.asarray([float(row["pred_future_peak_log"]) for row in rows])
    img_mae = np.asarray([float(row["future_image_mae"]) for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].hist(score[y < 0.5], bins=35, alpha=0.65, label="future negative")
    axes[0, 0].hist(score[y > 0.5], bins=35, alpha=0.65, label="future positive")
    axes[0, 0].set_xlabel("future-positive probability")
    axes[0, 0].legend()

    axes[0, 1].scatter(peak_true, peak_pred, s=9, alpha=0.45)
    lo = float(min(np.min(peak_true), np.min(peak_pred)))
    hi = float(max(np.max(peak_true), np.max(peak_pred)))
    axes[0, 1].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    axes[0, 1].set_xlabel("true future peak log")
    axes[0, 1].set_ylabel("pred future peak log")

    axes[1, 0].hist(img_mae, bins=40, color="#2d6a8e", alpha=0.85)
    axes[1, 0].set_xlabel("future image MAE")
    axes[1, 0].set_ylabel("count")

    prefix_groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        prefix_groups[int(row["prefix_length"])].append(float(row["future_image_mae"]))
    xs = sorted(prefix_groups)
    axes[1, 1].plot(xs, [float(np.mean(prefix_groups[x])) for x in xs], marker="o")
    axes[1, 1].set_xlabel("prefix length")
    axes[1, 1].set_ylabel("mean future image MAE")
    fig.tight_layout()
    fig.savefig(output_root / "joint_future_summary.png", dpi=180)
    plt.close(fig)


def plot_feature_ablation(rows: Sequence[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    top = sorted(rows, key=lambda row: coerce_float(row["auc_drop"]), reverse=True)[: min(12, len(rows))]
    names = [str(row["feature"]) for row in top][::-1]
    auc_drop = [coerce_float(row["auc_drop"]) for row in top][::-1]
    image_mae_increase = [coerce_float(row["future_image_mae_increase"]) for row in top][::-1]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].barh(y, auc_drop, color="#2d6a8e")
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("future-positive AUROC drop")
    axes[1].barh(y, image_mae_increase, color="#b23a48")
    axes[1].set_xlabel("future-image MAE increase")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def feature_ablation(
    model: JointB2FFutureModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    baseline: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(FEATURE_COLUMNS):
        metrics, _ = evaluate(model, loader, device, args, feature_ablate_index=index)
        rows.append(
            {
                "feature": feature,
                "baseline_auc": baseline["future_positive_auc"],
                "ablated_auc": metrics["future_positive_auc"],
                "auc_drop": baseline["future_positive_auc"] - metrics["future_positive_auc"],
                "baseline_future_image_mae": baseline["future_image_mae"],
                "ablated_future_image_mae": metrics["future_image_mae"],
                "future_image_mae_increase": metrics["future_image_mae"] - baseline["future_image_mae"],
                "baseline_peak_pearson": baseline["future_peak_pearson"],
                "ablated_peak_pearson": metrics["future_peak_pearson"],
                "peak_pearson_drop": baseline["future_peak_pearson"] - metrics["future_peak_pearson"],
            }
        )
    return sorted(rows, key=lambda row: coerce_float(row["auc_drop"]), reverse=True)


def write_report(output_root: Path, test_metrics: dict[str, float], importance: Sequence[dict[str, Any]], best_epoch: int, args: argparse.Namespace) -> None:
    lines = [
        "# Joint B2F And Future Fluorescence Evaluation",
        "",
        f"- Best epoch: `{best_epoch}`",
        f"- Target policy: `{args.target_policy}`",
        f"- Test future samples: `{int(test_metrics['n'])}`",
        f"- Future-positive samples: `{int(test_metrics['future_positive_count'])}`",
        f"- Future image MAE: `{test_metrics['future_image_mae']:.4f}`",
        f"- Future image PSNR: `{test_metrics['future_image_psnr']:.2f} dB`",
        f"- Current auxiliary B2F image MAE: `{test_metrics['current_image_mae']:.4f}`",
        f"- Future-positive AUROC: `{test_metrics['future_positive_auc']:.3f}`",
        f"- Future-positive AP: `{test_metrics['future_positive_ap']:.3f}`",
        f"- Future peak Pearson: `{test_metrics['future_peak_pearson']:.3f}`",
        f"- Future AUC Pearson: `{test_metrics['future_auc_pearson']:.3f}`",
        "",
        "## Interpretation",
        "",
        "This model tests the realistic joint hypothesis: learn same-time brightfield-to-fluorescence as an auxiliary visual task while predicting a future fluorescence image and scalar future-expression outcomes from a variable-length brightfield prefix.",
        "",
        "If future-image reconstruction is useful but future-positive AUROC is weak, the model is learning average future fluorescence appearance but not enough rare positive-discriminative signal. If both image metrics and AUROC improve over the scalar-only baseline, the joint dense supervision is helping.",
        "",
        "## Top Feature Ablations",
        "",
    ]
    for row in importance[:8]:
        lines.append(
            f"- `{row['feature']}`: AUROC drop `{coerce_float(row['auc_drop']):.3f}`, future-image MAE increase `{coerce_float(row['future_image_mae_increase']):.4f}`, peak-correlation drop `{coerce_float(row['peak_pearson_drop']):.3f}`"
        )
    output = output_root / "joint_b2f_future_report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds, test_ds, stats = build_datasets(args)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)

    pos_count = sum(float(row["future_positive"]) for row in train_ds.sample_rows)
    neg_count = max(1.0, len(train_ds) - pos_count)
    future_pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device, dtype=torch.float32)
    current_pos_count = 0
    for row in train_ds.sample_rows:
        prefix_ids = json.loads(row["prefix_instance_ids_json"])
        current_pos_count += coerce_int(train_ds.instance_by_id[prefix_ids[-1]]["fluorescence_positive"])
    current_neg_count = max(1, len(train_ds) - current_pos_count)
    current_pos_weight = torch.tensor([current_neg_count / max(current_pos_count, 1)], device=device, dtype=torch.float32)

    model = JointB2FFutureModel(
        feature_dim=len(FEATURE_COLUMNS),
        b2f_base_channels=args.b2f_base_channels,
        future_decoder_base=args.future_decoder_base,
        image_size=args.image_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = grad_scaler(use_amp)
    future_bce = nn.BCEWithLogitsLoss(pos_weight=future_pos_weight)
    current_bce = nn.BCEWithLogitsLoss(pos_weight=current_pos_weight)
    huber = nn.SmoothL1Loss()
    metrics_log: list[dict[str, Any]] = []
    best_score = -1e9
    start_epoch = 1
    metrics_path = args.output_root / "metrics.jsonl"
    last_checkpoint_path = args.output_root / "last_model.pt"
    best_checkpoint_path = args.output_root / "best_model.pt"

    write_json(
        args.output_root / "run_config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "test_count": len(test_ds),
            "train_future_positive_count": pos_count,
            "train_current_positive_count": current_pos_count,
            **stats,
        },
    )

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
        loss_parts: Counter[str] = Counter()
        for batch_index, batch in enumerate(train_loader):
            tensor_batch = move_batch({key: value for key, value in batch.items() if torch.is_tensor(value)}, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                outputs = model(
                    tensor_batch["frames"],
                    tensor_batch["features"],
                    tensor_batch["valid"],
                    tensor_batch["current_brightfield"],
                )
                loss, parts = loss_fn(outputs, tensor_batch, future_bce, current_bce, huber, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = tensor_batch["frames"].shape[0]
            loss_sum += float(loss.detach().cpu()) * batch_size
            seen += batch_size
            for key, value in parts.items():
                loss_parts[key] += value * batch_size
            if args.limit_train_batches is not None and batch_index + 1 >= args.limit_train_batches:
                break
        scheduler.step()
        panel_path = args.output_root / "predictions" / f"val_epoch_{epoch:03d}.png" if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs else None
        val_metrics, _ = evaluate(model, val_loader, device, args, panel_path=panel_path)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "lr": scheduler.get_last_lr()[0],
            **{key: loss_parts[key] / max(seen, 1) for key in sorted(loss_parts)},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        metrics_log.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        score = float(val_metrics.get("future_positive_auc", float("nan")))
        if not np.isfinite(score):
            score = -float(val_metrics["future_image_mae"])
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row, "stats": stats}, best_checkpoint_path)
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

    best = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, prediction_rows = evaluate(model, test_loader, device, args, panel_path=args.output_root / "predictions" / "test_best.png")
    add_horizon_bin(prediction_rows)
    importance: list[dict[str, Any]] = []
    if args.feature_ablation:
        importance = feature_ablation(model, test_loader, device, args, test_metrics)

    write_json(args.output_root / "test_metrics.json", {**test_metrics, "best_epoch": int(best["epoch"])})
    write_csv(args.output_root / "test_predictions.csv", prediction_rows)
    write_csv(args.output_root / "test_metrics_by_dataset.csv", grouped_metrics(prediction_rows, "dataset"))
    write_csv(args.output_root / "test_metrics_by_prefix_length.csv", grouped_metrics(prediction_rows, "prefix_length"))
    write_csv(args.output_root / "test_metrics_by_horizon_bin.csv", grouped_metrics(prediction_rows, "horizon_bin"))
    if importance:
        write_csv(args.output_root / "feature_ablation_importance.csv", importance)
        plot_feature_ablation(importance, args.output_root / "plots" / "feature_ablation_importance.png")
    plot_predictions(prediction_rows, args.output_root / "plots")
    plot_metric_lines(
        metrics_log,
        args.output_root / "plots" / "training_metrics.png",
        [
            "train_loss",
            "val_future_image_mae",
            "val_current_image_mae",
            "val_future_positive_auc",
            "val_future_peak_pearson",
            "val_future_auc_pearson",
        ],
    )
    write_report(args.output_root, test_metrics, importance, int(best["epoch"]), args)
    print(
        json.dumps(
            {
                "stage": "joint_b2f_future_finished",
                "test": test_metrics,
                "best_epoch": int(best["epoch"]),
                "output_root": str(args.output_root),
                "top_features": importance[:5],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
