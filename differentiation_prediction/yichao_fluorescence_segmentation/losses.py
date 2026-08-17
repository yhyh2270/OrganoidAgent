from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_factor = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_factor * torch.pow((1.0 - pt).clamp_min(1e-6), gamma) * bce
    final_weight = valid if weight is None else valid * weight
    return masked_mean(focal, final_weight)


def tversky_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = prob * valid
    target = target * valid
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1.0 - target) * valid).sum(dim=(1, 2, 3))
    fn = ((1.0 - prob) * target).sum(dim=(1, 2, 3))
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - score).mean()


def soft_mil_probability(prob: torch.Tensor, valid: torch.Tensor, sharpness: float = 8.0) -> torch.Tensor:
    masked_prob = prob * valid
    weights = torch.exp((sharpness * masked_prob).clamp(max=40.0)) * valid
    return (masked_prob * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1e-6)


def segmentation_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    focal_alpha: float,
    focal_gamma: float,
    tversky_alpha: float,
    tversky_beta: float,
    global_pos_weight: float,
    lambda_focal: float,
    lambda_tversky: float,
    lambda_global: float,
    lambda_mil: float,
    lambda_area: float,
    outside_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    global_logits = outputs["global_logits"]
    target = batch["positive"].to(logits.device)
    valid = batch["valid"].to(logits.device)
    organoid_mask = batch["organoid_mask"].to(logits.device)
    global_target = batch["global_positive"].to(logits.device)
    fraction_target = batch["positive_fraction"].to(logits.device)
    pixel_weight = outside_weight + organoid_mask
    prob = torch.sigmoid(logits)
    focal = focal_bce_with_logits(
        logits,
        target,
        valid,
        alpha=focal_alpha,
        gamma=focal_gamma,
        weight=pixel_weight,
    )
    tv = tversky_loss(prob, target, valid * pixel_weight, alpha=tversky_alpha, beta=tversky_beta)
    pos_weight = torch.tensor([global_pos_weight], dtype=global_logits.dtype, device=global_logits.device)
    global_loss = F.binary_cross_entropy_with_logits(global_logits, global_target, pos_weight=pos_weight)
    mil_prob = soft_mil_probability(prob, valid * organoid_mask)
    mil_loss = F.binary_cross_entropy(mil_prob.clamp(1e-5, 1 - 1e-5), global_target)
    pred_area = (prob * valid * organoid_mask).sum(dim=(1, 2, 3)) / (valid * organoid_mask).sum(dim=(1, 2, 3)).clamp_min(1.0)
    area_loss = F.smooth_l1_loss(torch.log1p(pred_area * 1000.0), torch.log1p(fraction_target * 1000.0))
    total = (
        lambda_focal * focal
        + lambda_tversky * tv
        + lambda_global * global_loss
        + lambda_mil * mil_loss
        + lambda_area * area_loss
    )
    return total, {
        "loss_focal": float(focal.detach().cpu()),
        "loss_tversky": float(tv.detach().cpu()),
        "loss_global": float(global_loss.detach().cpu()),
        "loss_mil": float(mil_loss.detach().cpu()),
        "loss_area": float(area_loss.detach().cpu()),
    }
