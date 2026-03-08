"""Generator network.

The generator maps a latent vector ``z`` of shape (B, latent_dim) to an RGB
image of shape (B, 3, resolution, resolution) in [-1, 1].

Architecture (example for 256×256, base_channels=64)::

    z ∈ ℝ^128
      │  Linear → (1024, 4, 4)
      ▼
    GenBlock  4×4   → 8×8    (1024 → 1024)
    GenBlock  8×8   → 16×16  (1024 → 512)
    GenBlock  16×16 → 32×32  (512  → 256)     ← optional SelfAttention
    GenBlock  32×32 → 64×64  (256  → 128)     ← optional SelfAttention
    GenBlock  64×64 → 128×128(128  → 64)
    GenBlock  128   → 256×256(64   → 32)
      │  BN → ReLU → Conv3×3 → Tanh
      ▼
    image ∈ ℝ^{3×256×256}

Self-Attention blocks are inserted *after* GenBlocks whose **output** spatial
resolution is in ``attention_resolutions`` (e.g. [32, 64]).  When
``use_attention=False`` the network is identical but the SelfAttention modules
are simply absent.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention
from .blocks import GenBlock


def _channel_schedule(base_channels: int, num_ups: int) -> list[int]:
    """Compute the channel width at each upsampling stage.

    We start from 16 × base_channels at the 4×4 feature map and halve
    each stage, clamping to a minimum of base_channels // 2.

    Returns a list of length ``num_ups + 1`` (including the initial width).
    """
    max_ch = base_channels * 16  # e.g. 64 * 16 = 1024
    min_ch = base_channels // 2  # e.g. 32
    channels = [max_ch]
    ch = max_ch
    for _ in range(num_ups):
        ch = max(ch // 2, min_ch)
        channels.append(ch)
    return channels


class Generator(nn.Module):
    """Residual convolutional generator with optional Self-Attention.

    Args:
        latent_dim: Dimensionality of the latent vector *z*.
        base_channels: Base channel width (actual widths are multiples).
        resolution: Output spatial resolution (must be a power of 2, ≥ 8).
        use_attention: Whether to insert SelfAttention blocks.
        attention_resolutions: Spatial resolutions at which to insert attention
            (measured by the output size of the preceding GenBlock).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 64,
        resolution: int = 256,
        use_attention: bool = False,
        attention_resolutions: list[int] | None = None,
    ) -> None:
        super().__init__()
        assert resolution >= 8 and (resolution & (resolution - 1)) == 0, \
            "resolution must be a power of 2 and >= 8"

        self.latent_dim = latent_dim
        self.resolution = resolution
        self.use_attention = use_attention
        self.attention_resolutions = set(attention_resolutions or [])

        num_ups = int(math.log2(resolution)) - 2  # 4→res requires this many ×2
        ch_schedule = _channel_schedule(base_channels, num_ups)

        # Initial projection: z → (ch_schedule[0], 4, 4)
        self.initial = nn.Linear(latent_dim, ch_schedule[0] * 4 * 4)
        self.initial_channels = ch_schedule[0]

        # Build upsampling blocks
        blocks: list[tuple[str, nn.Module]] = []
        current_res = 4
        for i in range(num_ups):
            in_ch = ch_schedule[i]
            out_ch = ch_schedule[i + 1]
            current_res *= 2
            blocks.append((f"block_{i}", GenBlock(in_ch, out_ch, upsample=True)))
            if use_attention and current_res in self.attention_resolutions:
                blocks.append((f"attn_{current_res}", SelfAttention(out_ch)))

        self.blocks = nn.Sequential(OrderedDict(blocks))

        # Final output layer
        final_ch = ch_schedule[-1]
        self.final = nn.Sequential(
            nn.BatchNorm2d(final_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_ch, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.initial(z)
        h = h.view(h.size(0), self.initial_channels, 4, 4)
        h = self.blocks(h)
        return self.final(h)
