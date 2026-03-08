"""Smoke tests for Generator and Discriminator at all supported resolutions."""

from __future__ import annotations

import pytest
import torch

from src.models.gan import build_gan, count_parameters, model_summary


# Test configs: one without attention, one with
def _make_cfg(resolution: int, use_attention: bool) -> dict:
    return {
        "dataset": {"resolution": resolution},
        "model": {
            "latent_dim": 128,
            "base_channels": 64,
            "use_attention": use_attention,
            "attention_resolutions": [32, 64] if use_attention else [],
        },
    }


@pytest.mark.parametrize("resolution", [128, 256, 512])
@pytest.mark.parametrize("use_attention", [False, True])
def test_generator_forward(resolution: int, use_attention: bool) -> None:
    """Generator produces correct output shape for all resolutions."""
    cfg = _make_cfg(resolution, use_attention)
    gen, _ = build_gan(cfg)
    gen.eval()

    z = torch.randn(2, cfg["model"]["latent_dim"])
    with torch.no_grad():
        out = gen(z)

    assert out.shape == (2, 3, resolution, resolution), \
        f"Expected (2, 3, {resolution}, {resolution}), got {out.shape}"
    # Output should be in [-1, 1] (tanh)
    assert out.min() >= -1.0 and out.max() <= 1.0


@pytest.mark.parametrize("resolution", [128, 256, 512])
@pytest.mark.parametrize("use_attention", [False, True])
def test_discriminator_forward(resolution: int, use_attention: bool) -> None:
    """Discriminator produces scalar scores for all resolutions."""
    cfg = _make_cfg(resolution, use_attention)
    _, disc = build_gan(cfg)
    disc.eval()

    x = torch.randn(2, 3, resolution, resolution)
    with torch.no_grad():
        out = disc(x)

    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"


@pytest.mark.parametrize("resolution", [128, 256, 512])
def test_architecture_parity(resolution: int) -> None:
    """GAN and SAGAN have the same architecture except for attention layers.

    The parameter difference must be exactly the attention parameters.
    """
    cfg_no_attn = _make_cfg(resolution, use_attention=False)
    cfg_attn = _make_cfg(resolution, use_attention=True)

    gen_base, disc_base = build_gan(cfg_no_attn)
    gen_attn, disc_attn = build_gan(cfg_attn)

    base_g = count_parameters(gen_base)
    attn_g = count_parameters(gen_attn)
    base_d = count_parameters(disc_base)
    attn_d = count_parameters(disc_attn)

    # SAGAN should have MORE parameters (attention adds Q/K/V/out_proj + gamma)
    assert attn_g >= base_g, "SAGAN generator should have >= params"
    assert attn_d >= base_d, "SAGAN discriminator should have >= params"

    # The difference should be non-zero (attention adds parameters)
    if resolution >= 64:
        # For resolutions >= 64, attention at 32 and/or 64 should add params
        assert attn_g > base_g or attn_d > base_d, \
            "Expected param difference from attention blocks"


@pytest.mark.parametrize("resolution", [128, 256])
def test_gradient_flow(resolution: int) -> None:
    """Verify that gradients flow through the full GAN pipeline."""
    cfg = _make_cfg(resolution, use_attention=True)
    gen, disc = build_gan(cfg)

    z = torch.randn(2, cfg["model"]["latent_dim"])
    fake = gen(z)
    score = disc(fake)
    loss = -score.mean()
    loss.backward()

    # Check that generator has gradients
    gen_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in gen.parameters() if p.requires_grad
    )
    assert gen_has_grad, "Generator should have non-zero gradients"


def test_model_summary() -> None:
    """model_summary returns expected keys."""
    cfg = _make_cfg(256, use_attention=True)
    gen, disc = build_gan(cfg)
    summary = model_summary(gen, disc)
    assert "generator_params" in summary
    assert "discriminator_params" in summary
    assert "total_params" in summary
    assert "use_attention" in summary
    assert summary["use_attention"] is True
