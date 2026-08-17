from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    for groups in (16, 12, 8, 6, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.block = ConvBlock(out_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1))


class ASPP(nn.Module):
    def __init__(self, channels: int, out_channels: int, rates: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        branches: list[nn.Module] = []
        for rate in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.GroupNorm(group_count(out_channels), out_channels),
                    nn.GELU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, 1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class GlobalGatedSegUNet(nn.Module):
    """U-Net segmentation model with multi-scale context and image-level gate."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c, dropout=dropout)
        self.enc2 = DownBlock(c, c * 2, dropout=dropout)
        self.enc3 = DownBlock(c * 2, c * 4, dropout=dropout)
        self.enc4 = DownBlock(c * 4, c * 8, dropout=dropout)
        self.bottleneck = DownBlock(c * 8, c * 12, dropout=dropout)
        self.context = ASPP(c * 12, c * 12)
        self.up4 = UpBlock(c * 12, c * 8, c * 8, dropout=dropout)
        self.up3 = UpBlock(c * 8, c * 4, c * 4, dropout=dropout)
        self.up2 = UpBlock(c * 4, c * 2, c * 2, dropout=dropout)
        self.up1 = UpBlock(c * 2, c, c, dropout=dropout)
        self.pixel_head = nn.Conv2d(c, 1, 1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 12, c * 3),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(c * 3, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.context(self.bottleneck(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        pixel_logits = self.pixel_head(d1)
        global_logits = self.global_head(b).squeeze(1)
        gated_logits = pixel_logits + global_logits.view(-1, 1, 1, 1)
        return {
            "pixel_logits": pixel_logits,
            "global_logits": global_logits,
            "logits": gated_logits,
        }
