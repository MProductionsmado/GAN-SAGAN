"""Shared convolutional building blocks for Generator and Discriminator.

All blocks are designed so that the *only* architectural difference between the
baseline GAN and the SAGAN variant is the presence / absence of Self-Attention
modules (which are inserted *between* blocks by the Generator / Discriminator
themselves, not inside these blocks).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
# Helper: optionally apply spectral normalisation
# ---------------------------------------------------------------------------
def _sn(module: nn.Module, use_sn: bool = True) -> nn.Module:
    return spectral_norm(module) if use_sn else module


# ---------------------------------------------------------------------------
# Generator blocks
# ---------------------------------------------------------------------------
class GenBlock(nn.Module):
    """Residual block for the Generator.

    Architecture::

        x ─┬─ BN → ReLU → Upsample → Conv3×3 → BN → ReLU → Conv3×3 ─ + → out
           └─ Upsample → Conv1×1 (shortcut) ──────────────────────────┘

    Uses BatchNorm and nearest-neighbour upsampling (avoids checkerboard artefacts).
    """

    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn1(x)
        h = F.relu(h, inplace=True)
        h = self._upsample(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)

        sc = self._upsample(x)
        sc = self.shortcut(sc)
        return h + sc


# ---------------------------------------------------------------------------
# Discriminator blocks
# ---------------------------------------------------------------------------
class DiscOptBlock(nn.Module):
    """First block in the Discriminator (no activation before the first conv).

    Architecture::

        x ─┬─ SNConv3×3 → ReLU → SNConv3×3 → AvgPool ─ + → out
           └─ AvgPool → SNConv1×1 (shortcut) ───────────┘
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
        self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)

        sc = F.avg_pool2d(x, 2)
        sc = self.shortcut(sc)
        return h + sc


class DiscBlock(nn.Module):
    """Residual block for the Discriminator.

    Architecture::

        x ─┬─ ReLU → SNConv3×3 → ReLU → SNConv3×3 → [AvgPool] ─ + → out
           └─ [SNConv1×1 → AvgPool] (shortcut) ─────────────────┘

    All convolutions use spectral normalisation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.downsample = downsample

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))

        self.need_shortcut_conv = (in_channels != out_channels) or downsample
        if self.need_shortcut_conv:
            self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x, inplace=False)  # not in-place: x needed for shortcut
        h = self.conv1(h)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        sc = x
        if self.need_shortcut_conv:
            sc = self.shortcut(sc)
        if self.downsample:
            sc = F.avg_pool2d(sc, 2)
        return h + sc
