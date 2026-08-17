from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from differentiation_prediction.yichao_future_expression.utils import coerce_float, coerce_int, image_to_tensor, read_csv


FEATURE_COLUMNS = [
    "area_px",
    "diameter_px",
    "circularity",
    "support_ratio",
    "edge_strength",
    "mean_signal",
    "bbox_w",
    "bbox_h",
    "square_crop_size_px",
    "crop_area_px",
    "time_index",
    "day_index",
]


class B2FDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        image_size: int,
        *,
        augment: bool = False,
        path_mode: str = "resized_256",
    ) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.augment = augment
        self.path_mode = path_mode

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        split: str,
        image_size: int,
        *,
        augment: bool = False,
        path_mode: str = "resized_256",
    ) -> "B2FDataset":
        rows = [row for row in read_csv(manifest_path) if row["split"] == split]
        return cls(rows, image_size=image_size, augment=augment, path_mode=path_mode)

    def __len__(self) -> int:
        return len(self.rows)

    def image_paths(self, row: dict[str, str]) -> tuple[Path, Path, Path]:
        if self.path_mode == "original_crop":
            return Path(row["brightfield_crop_path"]), Path(row["fluorescence_crop_path"]), Path(row["mask_crop_path"])
        if self.path_mode == "resized_256":
            return Path(row["brightfield_256_path"]), Path(row["fluorescence_256_path"]), Path(row["mask_256_path"])
        raise ValueError(f"Unsupported B2F path_mode: {self.path_mode}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        brightfield_path, fluorescence_path, mask_path = self.image_paths(row)
        brightfield = image_to_tensor(brightfield_path, self.image_size)
        fluorescence = image_to_tensor(fluorescence_path, self.image_size)
        mask = image_to_tensor(mask_path, self.image_size, mask=True)
        if self.augment:
            if torch.rand(()) < 0.5:
                brightfield = torch.flip(brightfield, dims=[2])
                fluorescence = torch.flip(fluorescence, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(()) < 0.5:
                brightfield = torch.flip(brightfield, dims=[1])
                fluorescence = torch.flip(fluorescence, dims=[1])
                mask = torch.flip(mask, dims=[1])
        return {
            "brightfield": brightfield,
            "fluorescence": fluorescence,
            "mask": mask,
            "positive": torch.tensor(float(coerce_int(row["fluorescence_positive"])), dtype=torch.float32),
            "peak_log": torch.tensor(coerce_float(row["fl_corrected_p90_log"]), dtype=torch.float32),
            "total_log": torch.tensor(coerce_float(row["fl_corrected_total_log"]), dtype=torch.float32),
            "instance_id": row["instance_id"],
            "dataset": row["dataset"],
        }


def feature_vector(row: dict[str, str]) -> list[float]:
    return [coerce_float(row.get(column)) for column in FEATURE_COLUMNS]


class FutureExpressionDataset(Dataset):
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
        *,
        augment: bool = False,
    ) -> None:
        self.sample_rows = list(sample_rows)
        self.instance_by_id = instance_by_id
        self.image_size = image_size
        self.max_prefix_frames = max_prefix_frames
        self.feature_mean = torch.tensor(feature_mean, dtype=torch.float32)
        self.feature_std = torch.tensor(feature_std, dtype=torch.float32).clamp_min(1e-6)
        self.peak_mean = float(peak_mean)
        self.peak_std = max(float(peak_std), 1e-6)
        self.auc_mean = float(auc_mean)
        self.auc_std = max(float(auc_std), 1e-6)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.sample_rows[index]
        prefix_ids = json.loads(sample["prefix_instance_ids_json"])
        prefix_ids = prefix_ids[-self.max_prefix_frames :]
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
        if self.augment:
            if torch.rand(()) < 0.5:
                frame_tensor = torch.flip(frame_tensor, dims=[3])
            if torch.rand(()) < 0.5:
                frame_tensor = torch.flip(frame_tensor, dims=[2])
        peak = coerce_float(sample["future_peak_log"])
        auc = coerce_float(sample["future_auc_log"])
        return {
            "frames": frame_tensor,
            "features": feature_tensor,
            "valid": valid_tensor,
            "future_positive": torch.tensor(float(coerce_int(sample["future_positive"])), dtype=torch.float32),
            "future_peak": torch.tensor((peak - self.peak_mean) / self.peak_std, dtype=torch.float32),
            "future_auc": torch.tensor((auc - self.auc_mean) / self.auc_std, dtype=torch.float32),
            "future_peak_raw": torch.tensor(peak, dtype=torch.float32),
            "future_auc_raw": torch.tensor(auc, dtype=torch.float32),
            "future_sample_id": sample["future_sample_id"],
            "track_id": sample["track_id"],
            "dataset": sample["dataset"],
        }


def load_future_data(manifest_path: Path, future_samples_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, str]]]:
    instance_rows = read_csv(manifest_path)
    future_rows = read_csv(future_samples_path)
    return instance_rows, future_rows, {row["instance_id"]: row for row in instance_rows}


def fit_feature_stats(instance_rows: Sequence[dict[str, str]], split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in instance_rows if row["split"] == split]
    matrix = np.asarray([feature_vector(row) for row in selected], dtype=np.float32)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def fit_target_stats(future_rows: Sequence[dict[str, str]], split: str = "train") -> tuple[float, float, float, float]:
    selected = [row for row in future_rows if row["split"] == split]
    peak = np.asarray([coerce_float(row["future_peak_log"]) for row in selected], dtype=np.float32)
    auc = np.asarray([coerce_float(row["future_auc_log"]) for row in selected], dtype=np.float32)
    return float(peak.mean()), float(max(peak.std(), 1e-6)), float(auc.mean()), float(max(auc.std(), 1e-6))
