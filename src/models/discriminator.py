"""Discriminator network.

The discriminator classifies a real or generated image of shape
(B, 3, resolution, resolution) and produces a scalar score per sample.

Architecture (example for 256×256, base_channels=64)::

    image ∈ ℝ^{3×256×256}
      │
    DiscOptBlock 256→128  (3 → 64)
    DiscBlock    128→64   (64 → 128)
    DiscBlock    64→32    (128 → 256)        ← optional SelfAttention
    DiscBlock    32→16    (256 → 512)        ← optional SelfAttention
    DiscBlock    16→8     (512 → 1024)
    DiscBlock    8→4      (1024 → 1024, no downsample)
      │  ReLU → GlobalSumPool → SNLinear(1024, 1)
      ▼
    score ∈ ℝ

Self-Attention blocks are inserted *after* DiscBlocks whose **output** spatial
resolution is in ``attention_resolutions``.  The feature-map resolution after a
DiscBlock with downsample is half the input resolution.

When ``use_attention=False`` the network is identical but the SelfAttention
modules are absent – ensuring a fair comparison.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .attention import SelfAttention
from .blocks import DiscBlock, DiscOptBlock


def _disc_channel_schedule(base_channels: int, num_downs: int) -> list[int]:
    """Channel widths for the discriminator (ascending with depth).

    Starts at base_channels after DiscOptBlock, doubles each stage, capped at
    16 × base_channels.
    """
    max_ch = base_channels * 16
    channels = [base_channels]
    ch = base_channels
    for _ in range(num_downs):
        ch = min(ch * 2, max_ch)
        channels.append(ch)
    return channels


class Discriminator(nn.Module):
    """Residual convolutional discriminator with optional Self-Attention.

    All convolutional and linear layers use spectral normalisation.

    Args:
        base_channels: Base channel width.
        resolution: Input image spatial resolution (power of 2, ≥ 8).
        use_attention: Whether to insert SelfAttention blocks.
        attention_resolutions: Feature-map resolutions at which to add attention.
    """

    def __init__(
        self,
        base_channels: int = 64,
        resolution: int = 256,
        use_attention: bool = False,
        attention_resolutions: list[int] | None = None,
    ) -> None:
        super().__init__()
        assert resolution >= 8 and (resolution & (resolution - 1)) == 0, \
            "resolution must be a power of 2 and >= 8"

        self.resolution = resolution
        self.use_attention = use_attention
        self.attention_resolutions = set(attention_resolutions or [])

        # Number of downsampling stages: resolution → 4   (resolution//2 per stage)
        # DiscOptBlock does the first downsample (resolution → resolution//2).
        # Then we need more DiscBlocks to get down to 4×4.
        # Last DiscBlock has no downsample (keeps 4×4).
        num_disc_blocks = int(math.log2(resolution)) - 2  # total blocks after DiscOpt
        # DiscOpt: res → res//2
        # Blocks 0..num_disc_blocks-2: each downsamples (/2)
        # Block num_disc_blocks-1: no downsample (stays at 4×4)

        ch_schedule = _disc_channel_schedule(base_channels, num_disc_blocks)

        # First block: 3 → base_channels, res → res//2
        blocks: list[tuple[str, nn.Module]] = [
            ("opt_block", DiscOptBlock(3, ch_schedule[0])),
        ]
        current_res = resolution // 2

        if use_attention and current_res in self.attention_resolutions:
            blocks.append((f"attn_{current_res}", SelfAttention(ch_schedule[0])))

        for i in range(num_disc_blocks):
            in_ch = ch_schedule[i]
            out_ch = ch_schedule[i + 1]
            is_last = (i == num_disc_blocks - 1)
            downsample = not is_last

            blocks.append((f"block_{i}", DiscBlock(in_ch, out_ch, downsample=downsample)))

            if downsample:
                current_res //= 2

            if use_attention and current_res in self.attention_resolutions:
                blocks.append((f"attn_{current_res}", SelfAttention(out_ch)))

        self.blocks = nn.Sequential(OrderedDict(blocks))

        # Final classifier
        final_ch = ch_schedule[-1]
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.linear = spectral_norm(nn.Linear(final_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        h = self.output(h)
        # Global sum pooling
        h = h.sum(dim=[2, 3])
        return self.linear(h)
