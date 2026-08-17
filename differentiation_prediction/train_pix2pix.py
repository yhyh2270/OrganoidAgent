#!/usr/bin/env python3
"""Train a pix2pix model from the manifest-first Yichao pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from differentiation_prediction.data import ManifestPairedDataset, build_samples
from differentiation_prediction.model import Discriminator, GeneratorUNet


def log(message: str) -> None:
    print(message, flush=True)


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(-1, 1)
    array = (array + 1.0) * 0.5
    array = array.squeeze(0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def save_triplet_grid(inputs: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, path: Path, limit: int = 4) -> None:
    rows: List[np.ndarray] = []
    for index in range(min(limit, inputs.shape[0])):
        inp = denorm_to_uint8(inputs[index])
        pred = denorm_to_uint8(preds[index])
        tgt = denorm_to_uint8(targets[index])
        rows.append(np.concatenate([inp, pred, tgt], axis=1))
    if not rows:
        return
    canvas = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(path)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred01 = (pred.detach() + 1.0) * 0.5
    tgt01 = (target.detach() + 1.0) * 0.5
    mae = torch.mean(torch.abs(pred01 - tgt01)).item()
    mse = torch.mean((pred01 - tgt01) ** 2).item()
    psnr = 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def evaluate(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    l1_loss: nn.Module,
    split_name: str,
    sample_path: Path | None = None,
    max_steps: int | None = None,
) -> Dict[str, float]:
    generator.eval()
    total_l1 = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_count = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds = generator(inputs)
            l1_value = l1_loss(preds, targets).item()
            metrics = batch_metrics(preds, targets)
            batch_size = inputs.shape[0]
            total_l1 += l1_value * batch_size
            total_mae += metrics["mae"] * batch_size
            total_mse += metrics["mse"] * batch_size
            total_psnr += metrics["psnr"] * batch_size
            total_count += batch_size

            if step == 0 and sample_path is not None:
                save_triplet_grid(inputs, preds, targets, sample_path)
            if max_steps is not None and step + 1 >= max_steps:
                break

    if total_count == 0:
        return {"split": split_name, "l1": float("nan"), "mae": float("nan"), "mse": float("nan"), "psnr": float("nan")}
    return {
        "split": split_name,
        "l1": total_l1 / total_count,
        "mae": total_mae / total_count,
        "mse": total_mse / total_count,
        "psnr": total_psnr / total_count,
    }


def _split_list(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pix2pix from a Yichao manifest")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/results/differentiation_prediction/manifest/yichao_manifest.csv"),
    )
    parser.add_argument("--split-column", default="baseline_split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--extra-eval-splits", default="external_test")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-mode", choices=["full", "quadrants", "grid"], default="quadrants")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--z-stride", type=int, default=1)
    parser.add_argument("--max-train-pairs-per-position", type=int, default=None)
    parser.add_argument("--max-eval-pairs-per-position", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--l1-lambda", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/results/differentiation_prediction"),
    )
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults: Dict[str, object] = {}
    if config_args.config is not None:
        defaults = json.loads(config_args.config.read_text(encoding="utf-8"))

    parser = build_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    args.config = config_args.config

    path_names = {"config", "manifest", "results_root"}
    for name in path_names:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    return args


def _loader_kwargs(args: argparse.Namespace, device: torch.device) -> Dict[str, int | bool]:
    kwargs: Dict[str, int | bool] = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    if device.type == "cuda":
        kwargs["pin_memory"] = True
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _build_dataloader(
    args: argparse.Namespace,
    split_name: str,
    random_flip: bool,
    max_pairs_per_position: int | None,
    device: torch.device,
) -> tuple[ManifestPairedDataset, DataLoader]:
    samples = build_samples(
        manifest_path=args.manifest,
        split_column=args.split_column,
        split_name=split_name,
        tile_size=args.tile_size,
        tile_mode=args.tile_mode,
        time_stride=args.time_stride,
        z_stride=args.z_stride,
        max_pairs_per_position=max_pairs_per_position,
    )
    dataset = ManifestPairedDataset(samples, image_size=args.image_size, random_flip=random_flip)
    dataloader = DataLoader(dataset, shuffle=random_flip, **_loader_kwargs(args, device))
    return dataset, dataloader


def _save_run_config(path: Path, args: argparse.Namespace, counts: Dict[str, int]) -> None:
    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "counts": counts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    log("== Differentiation Prediction Pix2Pix ==")
    log(f"Manifest: {args.manifest}")
    log(f"Split column: {args.split_column}")
    log(f"Device: {device}")

    train_ds, train_loader = _build_dataloader(
        args,
        split_name=args.train_split,
        random_flip=True,
        max_pairs_per_position=args.max_train_pairs_per_position,
        device=device,
    )
    val_ds, val_loader = _build_dataloader(
        args,
        split_name=args.val_split,
        random_flip=False,
        max_pairs_per_position=args.max_eval_pairs_per_position,
        device=device,
    )
    test_ds, test_loader = _build_dataloader(
        args,
        split_name=args.test_split,
        random_flip=False,
        max_pairs_per_position=args.max_eval_pairs_per_position,
        device=device,
    )

    extra_eval_loaders: Dict[str, DataLoader] = {}
    for split_name in _split_list(args.extra_eval_splits):
        dataset, dataloader = _build_dataloader(
            args,
            split_name=split_name,
            random_flip=False,
            max_pairs_per_position=args.max_eval_pairs_per_position,
            device=device,
        )
        if len(dataset) > 0:
            extra_eval_loaders[split_name] = dataloader

    if len(train_ds) == 0:
        raise RuntimeError("Training split produced zero samples")
    if len(val_ds) == 0:
        raise RuntimeError("Validation split produced zero samples")
    if len(test_ds) == 0:
        raise RuntimeError("Test split produced zero samples")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root / f"pix2pix_{args.split_column}_{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    for folder in [run_dir, checkpoints_dir, samples_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    counts = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        **{f"{split}_samples": len(loader.dataset) for split, loader in extra_eval_loaders.items()},
    }
    _save_run_config(run_dir / "config.json", args, counts)
    log(f"Run directory: {run_dir}")
    log(f"Sample counts: {counts}")

    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = _grad_scaler(enabled=use_amp)

    with torch.no_grad():
        probe = torch.zeros((1, 1, args.image_size, args.image_size), device=device)
        patch_shape = discriminator(probe, probe).shape[1:]

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_g", "train_d", "train_l1", "val_l1", "val_psnr", "test_l1", "test_psnr"])

    best_val_l1 = float("inf")
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        g_total = 0.0
        d_total = 0.0
        l1_total = 0.0
        seen = 0
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            batch_size = inputs.shape[0]
            valid = torch.ones((batch_size, *patch_shape), device=device)
            fake = torch.zeros((batch_size, *patch_shape), device=device)

            optimizer_g.zero_grad(set_to_none=True)
            with _autocast(device, enabled=use_amp):
                generated = generator(inputs)
                pred_fake = discriminator(inputs, generated)
                loss_g_gan = criterion_gan(pred_fake, valid)
                loss_g_l1 = criterion_l1(generated, targets)
                loss_g = loss_g_gan + args.l1_lambda * loss_g_l1
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)

            optimizer_d.zero_grad(set_to_none=True)
            with _autocast(device, enabled=use_amp):
                pred_real = discriminator(inputs, targets)
                loss_d_real = criterion_gan(pred_real, valid)
                pred_fake_detached = discriminator(inputs, generated.detach())
                loss_d_fake = criterion_gan(pred_fake_detached, fake)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            g_total += loss_g.item() * batch_size
            d_total += loss_d.item() * batch_size
            l1_total += loss_g_l1.item() * batch_size
            seen += batch_size

            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

        train_metrics = {
            "g": g_total / max(seen, 1),
            "d": d_total / max(seen, 1),
            "l1": l1_total / max(seen, 1),
        }
        val_metrics = evaluate(
            generator,
            val_loader,
            device=device,
            l1_loss=criterion_l1,
            split_name=args.val_split,
            sample_path=samples_dir / f"epoch_{epoch:03d}_{args.val_split}.png",
            max_steps=args.max_eval_steps,
        )
        test_metrics = evaluate(
            generator,
            test_loader,
            device=device,
            l1_loss=criterion_l1,
            split_name=args.test_split,
            sample_path=samples_dir / f"epoch_{epoch:03d}_{args.test_split}.png",
            max_steps=args.max_eval_steps,
        )
        extra_metrics = [
            evaluate(
                generator,
                loader,
                device=device,
                l1_loss=criterion_l1,
                split_name=split_name,
                sample_path=samples_dir / f"epoch_{epoch:03d}_{split_name}.png",
                max_steps=args.max_eval_steps,
            )
            for split_name, loader in extra_eval_loaders.items()
        ]

        elapsed = time.perf_counter() - epoch_start
        log(
            f"[epoch {epoch:03d}] "
            f"train_g={train_metrics['g']:.4f} train_d={train_metrics['d']:.4f} train_l1={train_metrics['l1']:.4f} "
            f"val_l1={val_metrics['l1']:.4f} val_psnr={val_metrics['psnr']:.2f} "
            f"test_l1={test_metrics['l1']:.4f} test_psnr={test_metrics['psnr']:.2f} "
            f"time={elapsed:.1f}s"
        )
        for metrics in extra_metrics:
            log(f"[epoch {epoch:03d}] extra_eval {metrics['split']} l1={metrics['l1']:.4f} psnr={metrics['psnr']:.2f}")

        with metrics_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    train_metrics["g"],
                    train_metrics["d"],
                    train_metrics["l1"],
                    val_metrics["l1"],
                    val_metrics["psnr"],
                    test_metrics["l1"],
                    test_metrics["psnr"],
                ]
            )

        torch.save(generator.state_dict(), checkpoints_dir / "last_generator.pt")
        torch.save(discriminator.state_dict(), checkpoints_dir / "last_discriminator.pt")
        if val_metrics["l1"] < best_val_l1:
            best_val_l1 = val_metrics["l1"]
            torch.save(generator.state_dict(), checkpoints_dir / "best_generator.pt")
            torch.save(discriminator.state_dict(), checkpoints_dir / "best_discriminator.pt")


def main() -> int:
    args = parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
