from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from differentiation_prediction.yichao_fluorescence_segmentation.utils import (
    coerce_float,
    coerce_int,
    image_to_tensor,
    read_csv,
    read_gray_float,
)


class FluorescenceSegmentationDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        image_size: int,
        *,
        augment: bool = False,
        include_distance: bool = True,
    ) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.augment = augment
        self.include_distance = include_distance

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        split: str,
        image_size: int,
        *,
        augment: bool = False,
        include_distance: bool = True,
    ) -> "FluorescenceSegmentationDataset":
        rows = [row for row in read_csv(manifest_path) if row["split"] == split]
        return cls(rows, image_size=image_size, augment=augment, include_distance=include_distance)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        brightfield = image_to_tensor(Path(row["brightfield_crop_path"]), self.image_size)
        organoid_mask = image_to_tensor(Path(row["mask_crop_path"]), self.image_size, mask=True)
        positive = image_to_tensor(Path(row["positive_mask_path"]), self.image_size, mask=True)
        ignore = image_to_tensor(Path(row["ignore_mask_path"]), self.image_size, mask=True)
        fluorescence = image_to_tensor(Path(row["fluorescence_crop_path"]), self.image_size)
        distance = self._distance_channel(organoid_mask) if self.include_distance else None
        channels = [brightfield, organoid_mask]
        if distance is not None:
            channels.append(distance)
        image = torch.cat(channels, dim=0)
        valid = (1.0 - ignore).clamp(0.0, 1.0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[2])
                positive = torch.flip(positive, dims=[2])
                ignore = torch.flip(ignore, dims=[2])
                valid = torch.flip(valid, dims=[2])
                organoid_mask = torch.flip(organoid_mask, dims=[2])
                fluorescence = torch.flip(fluorescence, dims=[2])
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[1])
                positive = torch.flip(positive, dims=[1])
                ignore = torch.flip(ignore, dims=[1])
                valid = torch.flip(valid, dims=[1])
                organoid_mask = torch.flip(organoid_mask, dims=[1])
                fluorescence = torch.flip(fluorescence, dims=[1])
        return {
            "image": image,
            "brightfield": brightfield,
            "fluorescence": fluorescence,
            "organoid_mask": organoid_mask,
            "positive": positive,
            "ignore": ignore,
            "valid": valid,
            "global_positive": torch.tensor(float(coerce_int(row["target_global_positive"])), dtype=torch.float32),
            "positive_fraction": torch.tensor(coerce_float(row["target_positive_fraction"]), dtype=torch.float32),
            "instance_id": row["instance_id"],
            "dataset": row["dataset"],
            "target_status": row["target_status"],
        }

    @staticmethod
    def _distance_channel(mask: torch.Tensor) -> torch.Tensor:
        mask_np = mask.squeeze(0).numpy() > 0.5
        if not mask_np.any():
            return torch.zeros_like(mask)
        inside = ndi.distance_transform_edt(mask_np).astype(np.float32)
        max_value = float(inside.max())
        if max_value > 0:
            inside /= max_value
        return torch.from_numpy(inside).unsqueeze(0)


def make_balanced_weights(rows: Sequence[dict[str, str]], positive_weight: float = 0.0) -> torch.Tensor:
    positives = [coerce_int(row.get("target_global_positive")) > 0 for row in rows]
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.ones(len(rows), dtype=torch.double)
    weight = positive_weight if positive_weight > 0 else min(10.0, n_neg / max(n_pos, 1))
    return torch.tensor([weight if is_pos else 1.0 for is_pos in positives], dtype=torch.double)


def _resize_float(arr: np.ndarray, image_size: int, *, mask: bool = False) -> torch.Tensor:
    clipped = np.clip(arr, 0.0, 1.0)
    image = Image.fromarray((clipped * 255.0).round().astype(np.uint8), mode="L")
    if image.size != (image_size, image_size):
        resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
        image = image.resize((image_size, image_size), resample=resample)
    out = np.asarray(image, dtype=np.float32) / 255.0
    if mask:
        out = (out > 0).astype(np.float32)
    return torch.from_numpy(out).unsqueeze(0)


def _soft_gate_from_mask(
    positive: np.ndarray,
    *,
    dilate: int,
    sigma: float,
    floor: float,
) -> np.ndarray:
    mask = positive > 0.5
    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=dilate)
    soft = ndi.gaussian_filter(mask.astype(np.float32), sigma=max(float(sigma), 0.0))
    max_value = float(soft.max())
    if max_value > 0:
        soft /= max_value
    return np.clip(np.maximum(soft, float(floor) * (positive > 0.5)), 0.0, 1.0).astype(np.float32)


class ContinuousFluorescenceTargetDataset(Dataset):
    """Brightfield input with a one-channel continuous, noise-suppressed fluorescence target."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        image_size: int,
        *,
        augment: bool = False,
        include_organoid_mask: bool = True,
        include_distance: bool = True,
        target_scale: float = 2.5,
        soft_mask_dilate: int = 9,
        soft_mask_sigma: float = 3.5,
        soft_mask_floor: float = 0.35,
    ) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.augment = augment
        self.include_organoid_mask = include_organoid_mask
        self.include_distance = include_distance
        self.target_scale = target_scale
        self.soft_mask_dilate = soft_mask_dilate
        self.soft_mask_sigma = soft_mask_sigma
        self.soft_mask_floor = soft_mask_floor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        brightfield = image_to_tensor(Path(row["brightfield_crop_path"]), self.image_size)
        organoid_mask = image_to_tensor(Path(row["mask_crop_path"]), self.image_size, mask=True)
        fluorescence = image_to_tensor(Path(row["fluorescence_crop_path"]), self.image_size)
        target = self._continuous_target(row)
        target_binary = (target >= 0.20).float()
        distance = FluorescenceSegmentationDataset._distance_channel(organoid_mask) if self.include_distance else None
        channels = [brightfield]
        if self.include_organoid_mask:
            channels.append(organoid_mask)
        if distance is not None:
            channels.append(distance)
        image = torch.cat(channels, dim=0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[2])
                brightfield = torch.flip(brightfield, dims=[2])
                fluorescence = torch.flip(fluorescence, dims=[2])
                target = torch.flip(target, dims=[2])
                target_binary = torch.flip(target_binary, dims=[2])
                organoid_mask = torch.flip(organoid_mask, dims=[2])
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[1])
                brightfield = torch.flip(brightfield, dims=[1])
                fluorescence = torch.flip(fluorescence, dims=[1])
                target = torch.flip(target, dims=[1])
                target_binary = torch.flip(target_binary, dims=[1])
                organoid_mask = torch.flip(organoid_mask, dims=[1])
        return {
            "image": image,
            "brightfield": brightfield,
            "fluorescence": fluorescence,
            "organoid_mask": organoid_mask,
            "target": target,
            "target_binary": target_binary,
            "target_mean": target.mean(),
            "target_sum": target.sum(),
            "instance_id": row["instance_id"],
            "dataset": row["dataset"],
            "target_status": row["target_status"],
        }

    def _continuous_target(self, row: dict[str, str]) -> torch.Tensor:
        fluorescence = read_gray_float(Path(row["fluorescence_crop_path"]))
        positive = read_gray_float(Path(row["positive_mask_path"]))
        bg = coerce_float(row.get("target_bg_median"), 0.0)
        corrected = np.maximum(fluorescence - bg, 0.0)
        gate = _soft_gate_from_mask(
            positive,
            dilate=self.soft_mask_dilate,
            sigma=self.soft_mask_sigma,
            floor=self.soft_mask_floor,
        )
        target = np.clip(corrected * gate * self.target_scale, 0.0, 1.0)
        return _resize_float(target, self.image_size)
