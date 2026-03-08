"""Self-Attention module for SAGAN.

Reference: *Self-Attention Generative Adversarial Networks* (Zhang et al., 2019).

Key design choices
------------------
* Query and Key projections reduce channels by a factor of 8 (memory-efficient).
* Value projection keeps full channel count.
* Learnable ``gamma`` parameter initialised to 0 → the block acts as an
  identity at the start of training and gradually learns to incorporate
  global context.
* All 1×1 convolutions use spectral normalisation for consistency with the
  discriminator and for training stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """Spatial self-attention block.

    Given an input feature map ``x`` of shape (B, C, H, W), the module computes
    scaled dot-product attention over all spatial positions and returns::

        gamma * attention(x) + x

    where ``gamma`` is a learnable scalar initialised to 0.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        ch_qk = max(in_channels // 8, 1)

        self.query = spectral_norm(nn.Conv2d(in_channels, ch_qk, 1, bias=False))
        self.key = spectral_norm(nn.Conv2d(in_channels, ch_qk, 1, bias=False))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1, bias=False))
        self.out_proj = spectral_norm(nn.Conv2d(in_channels, in_channels, 1, bias=False))

        # Learnable gate – starts at zero (identity)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W  # number of spatial positions

        # (B, ch_qk, N)
        q = self.query(x).view(B, -1, N)
        k = self.key(x).view(B, -1, N)
        v = self.value(x).view(B, -1, N)

        # Attention weights: (B, N, N)
        # Scaled dot-product – scale by sqrt(ch_qk)
        attn = torch.bmm(q.transpose(1, 2), k)  # (B, N, N)
        attn = attn / (q.size(1) ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # Attend: (B, C, N)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, H, W)
        out = self.out_proj(out)

        return self.gamma * out + x
