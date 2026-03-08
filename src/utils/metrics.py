"""Metrics computation (FID, KID) via clean-fid."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


def compute_fid(
    gen_dir: str | Path,
    real_dir: str | Path,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> float:
    """Compute Fréchet Inception Distance between generated and real images.

    Uses the ``clean-fid`` library for a standards-compliant implementation.
    Both directories must contain PNG/JPG images.
    """
    from cleanfid import fid as cleanfid

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = cleanfid.compute_fid(
        str(gen_dir),
        str(real_dir),
        device=device,
        batch_size=batch_size,
    )
    return float(score)


def compute_kid(
    gen_dir: str | Path,
    real_dir: str | Path,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> float:
    """Compute Kernel Inception Distance between generated and real images."""
    from cleanfid import fid as cleanfid

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = cleanfid.compute_kid(
        str(gen_dir),
        str(real_dir),
        device=device,
        batch_size=batch_size,
    )
    return float(score)
