"""Tests for the Self-Attention module."""

from __future__ import annotations

import pytest
import torch

from src.models.attention import SelfAttention


def test_self_attention_shape() -> None:
    """SelfAttention preserves spatial dimensions."""
    attn = SelfAttention(in_channels=64)
    x = torch.randn(2, 64, 32, 32)
    out = attn(x)
    assert out.shape == x.shape


def test_gamma_init_zero() -> None:
    """Gamma is initialised to 0 → output == input at init."""
    attn = SelfAttention(in_channels=128)
    assert attn.gamma.item() == 0.0

    x = torch.randn(1, 128, 16, 16)
    with torch.no_grad():
        out = attn(x)
    # With gamma=0, output should equal input
    assert torch.allclose(out, x, atol=1e-6)


def test_gamma_learns() -> None:
    """Gamma can be updated via gradient descent."""
    attn = SelfAttention(in_channels=32)
    x = torch.randn(2, 32, 8, 8)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert attn.gamma.grad is not None


@pytest.mark.parametrize("channels", [16, 32, 64, 128, 256])
def test_various_channel_sizes(channels: int) -> None:
    """SelfAttention works for various channel counts."""
    attn = SelfAttention(in_channels=channels)
    x = torch.randn(1, channels, 8, 8)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_spectral_norm() -> None:
    """Q, K, V convolutions should have spectral normalisation applied."""
    attn = SelfAttention(in_channels=64)
    # Spectral norm adds a 'weight_orig' parameter
    for name in ["query", "key", "value", "out_proj"]:
        module = getattr(attn, name)
        param_names = [n for n, _ in module.named_parameters()]
        assert "weight_orig" in param_names, \
            f"{name} should have spectral normalisation"
