#!/usr/bin/env python3
"""GPU-only pix2pix training on prepared Yichao .npy pairs.

Input data (prepared by prepare_yichao_pairs_to_npy.py):
- train_input.npy, train_target.npy
- test_input.npy, test_target.npy

Outputs:
- checkpoints (best/latest)
- metrics.csv
- triplet plots: left=brightfield, middle=prediction, right=ground truth
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import ImageDraw
print("[boot] importing torch...", flush=True)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def log(msg: str) -> None:
    print(msg, flush=True)


class NpyPairDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: X={x.shape} Y={y.shape}")
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [N,1,H,W], got {x.shape}")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        xin = torch.from_numpy(self.x[idx]).float() / 127.5 - 1.0
        yin = torch.from_numpy(self.y[idx]).float() / 127.5 - 1.0
        return {"input": xin, "target": yin, "idx": torch.tensor(idx, dtype=torch.long)}


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip), dim=1)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, use_norm=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.1)
        self.down5 = UNetDown(512, 512, dropout=0.1)
        self.down6 = UNetDown(512, 512, dropout=0.1)
        self.down7 = UNetDown(512, 512, dropout=0.1)
        self.down8 = UNetDown(512, 512, use_norm=False, dropout=0.1)

        self.up1 = UNetUp(512, 512, dropout=0.1)
        self.up2 = UNetUp(1024, 512, dropout=0.1)
        self.up3 = UNetUp(1024, 512, dropout=0.1)
        self.up4 = UNetUp(1024, 512, dropout=0.1)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()

        def block(in_c: int, out_c: int, normalize: bool = True) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels * 2, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((condition, target), dim=1))


def denorm_to_uint8(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().clamp(-1, 1)
    a = ((a + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    return a.squeeze(0).numpy()


def save_triplet_plot(inputs: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, out_path: Path, max_items: int = 6) -> None:
    n = min(max_items, inputs.shape[0])
    if n <= 0:
        return

    rows: List[np.ndarray] = []
    for i in range(n):
        arr_in = denorm_to_uint8(inputs[i])
        arr_pr = denorm_to_uint8(preds[i])
        arr_gt = denorm_to_uint8(targets[i])
        row = np.concatenate([arr_in, arr_pr, arr_gt], axis=1)
        rows.append(row)

    canvas = np.concatenate(rows, axis=0)
    image = Image.fromarray(canvas, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    h, w = canvas.shape
    panel_w = w // 3
    draw.rectangle([(0, 0), (w - 1, 20)], fill=(0, 0, 0))
    draw.text((8, 4), "Left: Brightfield", fill=(255, 255, 255))
    draw.text((panel_w + 8, 4), "Middle: Model Prediction", fill=(255, 255, 255))
    draw.text((2 * panel_w + 8, 4), "Right: Ground Truth", fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    p = (pred.detach() + 1.0) * 0.5
    t = (target.detach() + 1.0) * 0.5
    mae = torch.mean(torch.abs(p - t)).item()
    mse = torch.mean((p - t) ** 2).item()
    psnr = 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def evaluate(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    l1_loss: nn.Module,
    split_name: str,
    plot_path: Path | None,
    max_eval_steps: int | None,
    log_interval: int,
) -> Dict[str, float]:
    generator.eval()
    l1_total = 0.0
    mae_total = 0.0
    mse_total = 0.0
    psnr_total = 0.0
    count = 0
    total_steps = len(dataloader)
    if max_eval_steps is not None:
        total_steps = min(total_steps, max_eval_steps)
    t0 = time.perf_counter()

    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            pred = generator(inp)
            l1 = l1_loss(pred, tgt).item()
            m = batch_metrics(pred, tgt)

            bs = inp.shape[0]
            l1_total += l1 * bs
            mae_total += m["mae"] * bs
            mse_total += m["mse"] * bs
            psnr_total += m["psnr"] * bs
            count += bs

            if step == 1 and plot_path is not None:
                save_triplet_plot(inp, pred, tgt, plot_path, max_items=6)

            if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == total_steps):
                elapsed = time.perf_counter() - t0
                eta = (elapsed / step) * max(total_steps - step, 0)
                log(f"[{split_name}] step {step}/{total_steps} l1={l1:.4f} psnr={m['psnr']:.2f} eta={eta:.1f}s")

            if max_eval_steps is not None and step >= max_eval_steps:
                break

    if count == 0:
        return {"l1": float("nan"), "mae": float("nan"), "mse": float("nan"), "psnr": float("nan")}
    return {
        "l1": l1_total / count,
        "mae": mae_total / count,
        "mse": mse_total / count,
        "psnr": psnr_total / count,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train pix2pix on NPY (GPU-only)")
    p.add_argument("--data-dir", type=Path, default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/results/yichao_paired_npy"))
    p.add_argument("--results-root", type=Path, default=Path("/home/lachlan/ProjectsLFS/OrganoidAgent/results"))
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-eval-steps", type=int, default=None)
    return p


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. GPU not available.")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid --gpu-index {args.gpu_index}; available GPUs: {torch.cuda.device_count()}")

    torch.cuda.set_device(args.gpu_index)
    device = torch.device(f"cuda:{args.gpu_index}")
    torch.backends.cudnn.benchmark = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_id = datetime.datetime.now().strftime("pix2pix_npy_%Y%m%d_%H%M%S")
    run_dir = args.results_root / run_id
    ckpt_dir = run_dir / "checkpoints"
    plot_dir = run_dir / "plots"
    for d in [run_dir, ckpt_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    x_train = np.load(args.data_dir / "train_input.npy")
    y_train = np.load(args.data_dir / "train_target.npy")
    x_test = np.load(args.data_dir / "test_input.npy")
    y_test = np.load(args.data_dir / "test_target.npy")

    full_train_ds = NpyPairDataset(x_train, y_train)
    test_ds = NpyPairDataset(x_test, y_test)

    indices = np.arange(len(full_train_ds))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * args.val_ratio))
    val_count = min(val_count, len(indices) - 1)
    val_idx = indices[:val_count].tolist()
    train_idx = indices[val_count:].tolist()

    train_ds = IndexedDataset(full_train_ds, train_idx)
    val_ds = IndexedDataset(full_train_ds, val_idx)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    log("== GPU Training on NPY ==")
    log(f"Device: {device} ({torch.cuda.get_device_name(args.gpu_index)})")
    log(f"Data dir: {args.data_dir}")
    log(f"Run dir:  {run_dir}")
    log(f"Train/Val/Test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    log(f"Batch size: {args.batch_size}, workers: {args.num_workers}, epochs: {args.epochs}")

    G = GeneratorUNet().to(device)
    D = Discriminator().to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    gan_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    with torch.no_grad():
        probe = torch.zeros((1, 1, x_train.shape[2], x_train.shape[3]), device=device)
        patch_shape = D(probe, probe).shape[1:]
    log(f"PatchGAN output shape: {patch_shape}")

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_g", "train_d", "train_l1", "val_l1", "val_psnr", "test_l1", "test_psnr"])

    best_val = float("inf")
    t_all = time.perf_counter()
    total_steps = len(train_loader) if args.max_train_steps is None else min(len(train_loader), args.max_train_steps)

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        t_epoch = time.perf_counter()
        sum_g = 0.0
        sum_d = 0.0
        sum_l1 = 0.0
        seen = 0

        log(f"[train] epoch {epoch}/{args.epochs} start ({total_steps} steps)")
        for step, batch in enumerate(train_loader, start=1):
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            bs = inp.shape[0]
            valid = torch.ones((bs, *patch_shape), device=device)
            fake = torch.zeros((bs, *patch_shape), device=device)

            g_opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                pred = G(inp)
                g_adv = gan_loss(D(inp, pred), valid)
                g_l1 = l1_loss(pred, tgt)
                g_total = g_adv + args.lambda_l1 * g_l1
            scaler.scale(g_total).backward()
            scaler.step(g_opt)

            d_opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                d_real = gan_loss(D(inp, tgt), valid)
                d_fake = gan_loss(D(inp, pred.detach()), fake)
                d_total = 0.5 * (d_real + d_fake)
            scaler.scale(d_total).backward()
            scaler.step(d_opt)
            scaler.update()

            sum_g += g_total.item() * bs
            sum_d += d_total.item() * bs
            sum_l1 += g_l1.item() * bs
            seen += bs

            if args.log_interval > 0 and (step == 1 or step % args.log_interval == 0 or step == total_steps):
                elapsed = time.perf_counter() - t_epoch
                eta = (elapsed / step) * max(total_steps - step, 0)
                sps = seen / max(elapsed, 1e-6)
                log(
                    f"[train] epoch {epoch} step {step}/{total_steps} "
                    f"g={g_total.item():.4f} d={d_total.item():.4f} l1={g_l1.item():.4f} "
                    f"{sps:.2f} samples/s eta={eta:.1f}s"
                )

            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

        train_metrics = {"g": sum_g / seen, "d": sum_d / seen, "l1": sum_l1 / seen}
        val_metrics = evaluate(
            G, val_loader, device, l1_loss, "val", plot_dir / f"val_triplets_epoch_{epoch:03d}.png",
            args.max_eval_steps, args.log_interval
        )
        test_metrics = evaluate(
            G, test_loader, device, l1_loss, "test", plot_dir / f"test_triplets_epoch_{epoch:03d}.png",
            args.max_eval_steps, args.log_interval
        )

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{train_metrics['g']:.6f}",
                    f"{train_metrics['d']:.6f}",
                    f"{train_metrics['l1']:.6f}",
                    f"{val_metrics['l1']:.6f}",
                    f"{val_metrics['psnr']:.6f}",
                    f"{test_metrics['l1']:.6f}",
                    f"{test_metrics['psnr']:.6f}",
                ]
            )

        log(
            f"[epoch {epoch}] train_l1={train_metrics['l1']:.4f} "
            f"val_l1={val_metrics['l1']:.4f} test_l1={test_metrics['l1']:.4f} "
            f"test_plot={plot_dir / f'test_triplets_epoch_{epoch:03d}.png'}"
        )

        ckpt = {
            "epoch": epoch,
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
            "val_l1": val_metrics["l1"],
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / "latest.pt")
        if val_metrics["l1"] < best_val:
            best_val = val_metrics["l1"]
            torch.save(G.state_dict(), ckpt_dir / "best_generator.pt")
            torch.save(D.state_dict(), ckpt_dir / "best_discriminator.pt")

    # Final best-generator test plot.
    best_G = GeneratorUNet().to(device)
    best_G.load_state_dict(torch.load(ckpt_dir / "best_generator.pt", map_location=device))
    final_test = evaluate(
        best_G,
        test_loader,
        device,
        l1_loss,
        "test-final",
        plot_dir / "test_triplets_best.png",
        args.max_eval_steps,
        args.log_interval,
    )

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "best_val_l1": best_val,
        "final_test": final_test,
        "elapsed_sec": round(time.perf_counter() - t_all, 3),
        "test_plot_best": str(plot_dir / "test_triplets_best.png"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[done] summary: {summary}")


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
