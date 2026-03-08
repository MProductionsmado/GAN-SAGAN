"""GAN model factory and Exponential Moving Average (EMA) wrapper.

Provides:
* ``build_gan(config)`` – constructs Generator and Discriminator from a config dict.
* ``ExponentialMovingAverage`` – maintains an EMA copy of the generator weights
  for smoother sample quality during evaluation.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from .discriminator import Discriminator
from .generator import Generator


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_gan(cfg: dict) -> tuple[Generator, Discriminator]:
    """Build Generator and Discriminator from a configuration dict.

    The returned models share the exact same architecture; the **only**
    difference is governed by ``cfg["model"]["use_attention"]``.
    """
    mcfg = cfg["model"]
    dcfg = cfg["dataset"]

    gen = Generator(
        latent_dim=mcfg["latent_dim"],
        base_channels=mcfg["base_channels"],
        resolution=dcfg["resolution"],
        use_attention=mcfg["use_attention"],
        attention_resolutions=mcfg.get("attention_resolutions", []),
    )

    disc = Discriminator(
        base_channels=mcfg["base_channels"],
        resolution=dcfg["resolution"],
        use_attention=mcfg["use_attention"],
        attention_resolutions=mcfg.get("attention_resolutions", []),
    )

    return gen, disc


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(gen: Generator, disc: Discriminator) -> dict[str, Any]:
    """Return a compact summary dict for logging / reporting."""
    return {
        "generator_params": count_parameters(gen),
        "discriminator_params": count_parameters(disc),
        "total_params": count_parameters(gen) + count_parameters(disc),
        "latent_dim": gen.latent_dim,
        "resolution": gen.resolution,
        "use_attention": gen.use_attention,
    }


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------
class ExponentialMovingAverage:
    """Maintains an EMA copy of a model's parameters.

    Usage::

        ema = ExponentialMovingAverage(generator, decay=0.999)
        # after each G optimiser step:
        ema.update()
        # for evaluation / sampling:
        with ema.apply():
            samples = generator(z)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        # Shadow copy of all parameters
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self) -> "_EMAContext":
        """Context manager that temporarily swaps model params with EMA shadow."""
        return _EMAContext(self)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


class _EMAContext:
    """Context manager for EMA parameter swap."""

    def __init__(self, ema: ExponentialMovingAverage) -> None:
        self.ema = ema

    def __enter__(self) -> None:
        self.ema.backup = {}
        for name, param in self.ema.model.named_parameters():
            if param.requires_grad and name in self.ema.shadow:
                self.ema.backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name])

    def __exit__(self, *args: Any) -> None:
        for name, param in self.ema.model.named_parameters():
            if param.requires_grad and name in self.ema.backup:
                param.data.copy_(self.ema.backup[name])
        self.ema.backup = {}
