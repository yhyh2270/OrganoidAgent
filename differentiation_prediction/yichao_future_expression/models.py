from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class B2FMultiTaskUNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 12)
        self.up4 = nn.ConvTranspose2d(c * 12, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out_image = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 12, c * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c * 4, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_image(d1), self.scalar_head(b)


def _group_count(channels: int) -> int:
    for groups in (16, 12, 8, 6, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualGNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        squeeze_channels = max(8, out_channels // 8)
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, squeeze_channels, 1),
            nn.GELU(),
            nn.Conv2d(squeeze_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = out * self.squeeze(out)
        return self.act(out + residual)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()
        self.block = ResidualGNBlock(out_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.act(self.norm(self.down(x))))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = ResidualGNBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1))


class StrongB2FResUNet(nn.Module):
    """Larger residual U-Net for sparse fluorescence reconstruction.

    The forward method returns image logits, not a sigmoid image. Training losses
    can therefore combine weighted BCE-with-logits and intensity regression
    without losing numerical stability.
    """

    def __init__(self, base_channels: int = 48, dropout: float = 0.05) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ResidualGNBlock(1, c, dropout=dropout)
        self.enc2 = DownsampleBlock(c, c * 2, dropout=dropout)
        self.enc3 = DownsampleBlock(c * 2, c * 4, dropout=dropout)
        self.enc4 = DownsampleBlock(c * 4, c * 8, dropout=dropout)
        self.bottleneck = DownsampleBlock(c * 8, c * 12, dropout=dropout)
        self.mid = ResidualGNBlock(c * 12, c * 12, dropout=dropout)
        self.up4 = UpsampleBlock(c * 12, c * 8, c * 8, dropout=dropout)
        self.up3 = UpsampleBlock(c * 8, c * 4, c * 4, dropout=dropout)
        self.up2 = UpsampleBlock(c * 4, c * 2, c * 2, dropout=dropout)
        self.up1 = UpsampleBlock(c * 2, c, c, dropout=dropout)
        self.out_logits = nn.Conv2d(c, 1, 1)
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 12, c * 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(c * 4, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.mid(self.bottleneck(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out_logits(d1), self.scalar_head(b)


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


class Pix2PixB2FUNet(nn.Module):
    """Pix2pix-style U-Net adapted from the older working fluorescence code.

    It keeps the old InstanceNorm + ELU encoder/decoder structure, but returns
    logits so the modern weighted BCE/intensity loss can train stably.
    """

    def __init__(self, base_channels: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        c = base_channels
        self.down1 = Pix2PixDownsample(1, c, normalize=False)
        self.down2 = Pix2PixDownsample(c, c * 2)
        self.down3 = Pix2PixDownsample(c * 2, c * 4)
        self.down4 = Pix2PixDownsample(c * 4, c * 8)
        self.up1 = Pix2PixUpsample(c * 8, c * 4, dropout=dropout)
        self.up2 = Pix2PixUpsample(c * 8, c * 2)
        self.up3 = Pix2PixUpsample(c * 4, c)
        self.out_logits = nn.ConvTranspose2d(c * 2, 1, kernel_size=4, stride=2, padding=1)
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 2),
            nn.ELU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(c * 2, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = torch.cat([self.up1(d4), d3], dim=1)
        u2 = torch.cat([self.up2(u1), d2], dim=1)
        u3 = torch.cat([self.up3(u2), d1], dim=1)
        return self.out_logits(u3), self.scalar_head(d4)


class SmallImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 4, c * 6, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 6),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 6, embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FutureExpressionModel(nn.Module):
    def __init__(self, feature_dim: int, image_embedding_dim: int = 128, hidden_dim: int = 160) -> None:
        super().__init__()
        self.image_encoder = SmallImageEncoder(embedding_dim=image_embedding_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )
        self.gru = nn.GRU(image_embedding_dim + 64, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, frames: torch.Tensor, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch * time_steps, channels, height, width)
        image_embeddings = self.image_encoder(flat_frames).reshape(batch, time_steps, -1)
        feature_embeddings = self.feature_encoder(features)
        sequence = torch.cat([image_embeddings, feature_embeddings], dim=-1)
        sequence = sequence * valid.unsqueeze(-1)
        lengths = valid.sum(dim=1).clamp_min(1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])
