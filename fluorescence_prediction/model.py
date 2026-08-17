from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
except Exception as exc:
    raise ImportError("This script requires torchvision. Install it with: pip install torchvision") from exc


def global_avg_pool(feat: torch.Tensor) -> torch.Tensor:
    return feat.mean(dim=(2, 3))


class MultiScaleConvNeXtTinyOrganoidTaskNet(nn.Module):
    """
    ConvNeXt-Tiny backbone for organoid viability prediction.

    Input:
      image: [B, C, H, W], where the first three channels are RGB brightfield

    Design:
      1. Only RGB enters ConvNeXt.
      2. Multi-scale features are taken from ConvNeXt stages with channels 192/384/768.
      3. pred_rank is aliased to pred_viability for rank-on-viability training.
    """

    def __init__(
        self,
        proj_dim: int = 64,
        hidden_dim: int = 192,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        imagenet_norm: bool = True,
    ) -> None:
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.features = backbone.features

        self.imagenet_norm = bool(imagenet_norm)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.stage_indices = (3, 5, 7)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.proj2 = nn.Sequential(
            nn.Linear(192, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.proj3 = nn.Sequential(
            nn.Linear(384, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.proj4 = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

        self.shared_mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.viability_head = nn.Linear(hidden_dim, 1)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected image [B, C, H, W], got {tuple(image.shape)}")
        if image.shape[1] < 3:
            raise ValueError(f"Expected image with at least RGB channels, got C={image.shape[1]}")

        rgb = image[:, :3]
        if self.imagenet_norm:
            rgb = (rgb - self.rgb_mean) / self.rgb_std

        feats = {}
        x = rgb
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.stage_indices:
                feats[idx] = x

        pool2 = global_avg_pool(feats[3])
        pool3 = global_avg_pool(feats[5])
        pool4 = global_avg_pool(feats[7])

        p2 = self.proj2(pool2)
        p3 = self.proj3(pool3)
        p4 = self.proj4(pool4)
        pooled = torch.cat([p2, p3, p4], dim=1)

        shared = self.shared_mlp(pooled)
        pred_viability = torch.sigmoid(self.viability_head(shared)).squeeze(1)

        return {
            "pred_viability": pred_viability,
            "pred_rank": pred_viability,
        }
