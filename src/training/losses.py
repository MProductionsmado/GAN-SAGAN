"""GAN loss functions.

Provides Hinge loss (default, as in SAGAN / SNGAN) and Non-Saturating loss.
Both variants expose the same interface so they can be swapped via config.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hinge Loss (Miyato et al., 2018 / Zhang et al., 2019)
# ---------------------------------------------------------------------------
def hinge_loss_d(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss: max(0, 1 - D(real)) + max(0, 1 + D(fake))."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def hinge_loss_g(fake_scores: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss: -D(fake)."""
    return -fake_scores.mean()


# ---------------------------------------------------------------------------
# Non-Saturating Loss (Goodfellow et al., 2014)
# ---------------------------------------------------------------------------
def nonsaturating_loss_d(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Discriminator non-saturating loss: softplus(-D(real)) + softplus(D(fake))."""
    return F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()


def nonsaturating_loss_g(fake_scores: torch.Tensor) -> torch.Tensor:
    """Generator non-saturating loss: softplus(-D(fake))."""
    return F.softplus(-fake_scores).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_LOSS_REGISTRY = {
    "hinge": (hinge_loss_d, hinge_loss_g),
    "nonsaturating": (nonsaturating_loss_d, nonsaturating_loss_g),
}


def get_loss_fns(name: str = "hinge"):
    """Return ``(d_loss_fn, g_loss_fn)`` by name.

    Raises ``ValueError`` for unknown loss types.
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type '{name}'. Available: {list(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[name]
