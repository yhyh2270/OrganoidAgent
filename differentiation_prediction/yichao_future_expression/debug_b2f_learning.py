#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from differentiation_prediction.yichao_future_expression.utils import (
    DEFAULT_OUTPUT_ROOT,
    coerce_float,
    image_to_tensor,
    read_csv,
    save_b2f_panel,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-data B2F learning check. This intentionally overfits a small "
            "subset before any long run is trusted."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "debug_b2f_pix2pix_overfit_128")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--path-mode", choices=("original_crop", "resized_256"), default="original_crop")
    parser.add_argument("--selection", choices=("high_signal", "random"), default="high_signal")
    parser.add_argument("--signal-threshold", type=float, default=0.25)
    parser.add_argument("--background-weight", type=float, default=0.10)
    parser.add_argument("--signal-boost", type=float, default=20.0)
    parser.add_argument("--panel-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


class DebugPairDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, str]], image_size: int, *, path_mode: str, augment: bool = False) -> None:
        self.rows = list(rows)
        self.image_size = image_size
        self.path_mode = path_mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def image_paths(self, row: dict[str, str]) -> tuple[Path, Path, Path]:
        if self.path_mode == "original_crop":
            return Path(row["brightfield_crop_path"]), Path(row["fluorescence_crop_path"]), Path(row["mask_crop_path"])
        return Path(row["brightfield_256_path"]), Path(row["fluorescence_256_path"]), Path(row["mask_256_path"])

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
            "instance_id": row["instance_id"],
            "dataset": row["dataset"],
            "fl_corrected_p90": torch.tensor(coerce_float(row.get("fl_corrected_p90")), dtype=torch.float32),
        }


class Pix2PixDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.ELU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Pix2PixUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ELU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Pix2PixUNetGenerator(nn.Module):
    """Compact U-Net matching the older working pix2pix module structure."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.down1 = Pix2PixDownsample(1, c, normalize=False)
        self.down2 = Pix2PixDownsample(c, c * 2)
        self.down3 = Pix2PixDownsample(c * 2, c * 4)
        self.down4 = Pix2PixDownsample(c * 4, c * 8)
        self.up1 = Pix2PixUpsample(c * 8, c * 4, dropout=0.5)
        self.up2 = Pix2PixUpsample(c * 8, c * 2)
        self.up3 = Pix2PixUpsample(c * 4, c)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = torch.cat([self.up1(d4), d3], dim=1)
        u2 = torch.cat([self.up2(u1), d2], dim=1)
        u3 = torch.cat([self.up3(u2), d1], dim=1)
        return self.final(u3)


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("split") == "train"
        and row.get("is_edge_padded", "0") in ("0", "False", "false", "")
        and Path(row.get("brightfield_crop_path", "")).exists()
        and Path(row.get("fluorescence_crop_path", "")).exists()
        and Path(row.get("mask_crop_path", "")).exists()
    ]
    if len(candidates) < args.subset_size:
        raise SystemExit(f"Only {len(candidates)} usable train rows found; requested {args.subset_size}.")
    if args.selection == "random":
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(candidates), size=args.subset_size, replace=False)
        return [candidates[int(i)] for i in indices]
    candidates.sort(
        key=lambda row: (
            coerce_float(row.get("fl_corrected_p90")),
            coerce_float(row.get("fl_corrected_total")),
            coerce_float(row.get("fl_inside_p99")),
        ),
        reverse=True,
    )
    return candidates[: args.subset_size]


def weighted_b2f_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    signal_boost: float,
    background_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = mask.clamp(0.0, 1.0)
    signal_weight = target.clamp(0.0, 1.0).pow(1.5)
    weights = background_weight + valid * (1.0 + signal_boost * signal_weight)
    l1 = (torch.abs(pred - target) * weights).sum() / weights.sum().clamp_min(1.0)
    mse = ((pred - target) ** 2 * weights).sum() / weights.sum().clamp_min(1.0)
    total = l1 + 0.25 * mse
    return total, {"loss_l1": float(l1.detach().cpu()), "loss_mse": float(mse.detach().cpu())}


@torch.no_grad()
def parameter_delta_l2(model: nn.Module, reference: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name not in reference:
            continue
        diff = param.detach().float().cpu() - reference[name]
        total += float(torch.sum(diff * diff))
    return float(math.sqrt(total))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    args: argparse.Namespace,
    panel_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    signal_mae_sum = 0.0
    background_mae_sum = 0.0
    signal_pixels = 0.0
    background_pixels = 0.0
    pixel_count = 0.0
    pred_peaks: list[float] = []
    target_peaks: list[float] = []
    pred_means: list[float] = []
    target_means: list[float] = []
    first_panel = True
    for batch in loader:
        brightfield = batch["brightfield"].to(device)
        target = batch["fluorescence"].to(device)
        mask = batch["mask"].to(device)
        pred = model(brightfield)
        loss, _ = weighted_b2f_loss(
            pred,
            target,
            mask,
            signal_boost=args.signal_boost,
            background_weight=args.background_weight,
        )
        abs_err = torch.abs(pred - target)
        signal = ((target > args.signal_threshold) & (mask > 0.5)).float()
        background = (mask > 0.5).float() * (1.0 - signal)
        batch_size = brightfield.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        mae_sum += float(abs_err.sum().detach().cpu())
        mse_sum += float(((pred - target) ** 2).sum().detach().cpu())
        signal_mae_sum += float((abs_err * signal).sum().detach().cpu())
        background_mae_sum += float((abs_err * background).sum().detach().cpu())
        signal_pixels += float(signal.sum().detach().cpu())
        background_pixels += float(background.sum().detach().cpu())
        pixel_count += float(np.prod(target.shape))
        pred_peaks.extend(pred.flatten(1).max(dim=1).values.detach().cpu().numpy().tolist())
        target_peaks.extend(target.flatten(1).max(dim=1).values.detach().cpu().numpy().tolist())
        pred_means.extend((pred * mask).flatten(1).sum(dim=1).div(mask.flatten(1).sum(dim=1).clamp_min(1.0)).detach().cpu().numpy().tolist())
        target_means.extend((target * mask).flatten(1).sum(dim=1).div(mask.flatten(1).sum(dim=1).clamp_min(1.0)).detach().cpu().numpy().tolist())
        if panel_path is not None and first_panel:
            save_b2f_panel(
                brightfield.detach().cpu(),
                pred.detach().cpu(),
                target.detach().cpu(),
                panel_path,
                keys=list(batch["instance_id"]),
                max_items=min(8, brightfield.shape[0]),
            )
            first_panel = False
    pred_peak_arr = np.asarray(pred_peaks, dtype=np.float64)
    target_peak_arr = np.asarray(target_peaks, dtype=np.float64)
    pred_mean_arr = np.asarray(pred_means, dtype=np.float64)
    target_mean_arr = np.asarray(target_means, dtype=np.float64)
    peak_corr = float(np.corrcoef(pred_peak_arr, target_peak_arr)[0, 1]) if np.std(pred_peak_arr) > 1e-8 and np.std(target_peak_arr) > 1e-8 else float("nan")
    mean_corr = float(np.corrcoef(pred_mean_arr, target_mean_arr)[0, 1]) if np.std(pred_mean_arr) > 1e-8 and np.std(target_mean_arr) > 1e-8 else float("nan")
    return {
        "loss": loss_sum / max(len(loader.dataset), 1),
        "mae": mae_sum / max(pixel_count, 1.0),
        "mse": mse_sum / max(pixel_count, 1.0),
        "signal_mae": signal_mae_sum / max(signal_pixels, 1.0),
        "background_mae": background_mae_sum / max(background_pixels, 1.0),
        "pred_peak_mean": float(pred_peak_arr.mean()) if pred_peak_arr.size else float("nan"),
        "target_peak_mean": float(target_peak_arr.mean()) if target_peak_arr.size else float("nan"),
        "pred_mean": float(pred_mean_arr.mean()) if pred_mean_arr.size else float("nan"),
        "target_mean": float(target_mean_arr.mean()) if target_mean_arr.size else float("nan"),
        "peak_corr": peak_corr,
        "mean_corr": mean_corr,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "panels").mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_root / "manifests" / "projected_instances_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    selected_rows = select_rows(read_csv(manifest_path), args)
    train_dataset = DebugPairDataset(selected_rows, args.image_size, path_mode=args.path_mode, augment=True)
    eval_dataset = DebugPairDataset(selected_rows, args.image_size, path_mode=args.path_mode, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=min(args.batch_size, 8), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pix2PixUNetGenerator(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    initial_state = {name: param.detach().float().cpu().clone() for name, param in model.named_parameters()}
    metrics_path = args.output_root / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    write_json(
        args.output_root / "run_config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "manifest_path": str(manifest_path),
            "selected_count": len(selected_rows),
            "selected_instances": [row["instance_id"] for row in selected_rows],
            "selected_fl_corrected_p90_range": [
                min(coerce_float(row.get("fl_corrected_p90")) for row in selected_rows),
                max(coerce_float(row.get("fl_corrected_p90")) for row in selected_rows),
            ],
        },
    )

    initial_metrics = evaluate(
        model,
        eval_loader,
        device,
        args=args,
        panel_path=args.output_root / "panels" / "step_0000.png",
    )
    initial_loss = initial_metrics["loss"]
    iterator = cycle(train_loader)
    last_metrics: dict[str, float] = initial_metrics
    for step in range(1, args.steps + 1):
        model.train()
        batch = next(iterator)
        brightfield = batch["brightfield"].to(device)
        target = batch["fluorescence"].to(device)
        mask = batch["mask"].to(device)
        pred = model(brightfield)
        loss, components = weighted_b2f_loss(
            pred,
            target,
            mask,
            signal_boost=args.signal_boost,
            background_weight=args.background_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        should_eval = step == 1 or step % args.panel_every == 0 or step == args.steps
        row: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "param_delta_l2": parameter_delta_l2(model, initial_state),
        }
        row.update({f"train_{key}": value for key, value in components.items()})
        if should_eval:
            last_metrics = evaluate(
                model,
                eval_loader,
                device,
                args=args,
                panel_path=args.output_root / "panels" / f"step_{step:04d}.png",
            )
            row.update({f"eval_{key}": value for key, value in last_metrics.items()})
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    final_loss = last_metrics["loss"]
    summary = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / max(initial_loss, 1e-8),
        "loss_drop_fraction": 1.0 - final_loss / max(initial_loss, 1e-8),
        "final_param_delta_l2": parameter_delta_l2(model, initial_state),
        "learned": bool(final_loss < 0.80 * initial_loss and parameter_delta_l2(model, initial_state) > 1e-6),
        "final_metrics": last_metrics,
        "output_root": str(args.output_root),
    }
    write_json(args.output_root / "summary.json", summary)
    torch.save(
        {
            "model": model.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "summary": summary,
        },
        args.output_root / "debug_pix2pix_model.pt",
    )
    print(json.dumps({"stage": "debug_b2f_learning_finished", **summary}, indent=2), flush=True)
    return 0 if summary["learned"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
