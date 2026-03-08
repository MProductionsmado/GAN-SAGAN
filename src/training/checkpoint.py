"""Checkpoint save / load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from ..utils.config import save_config


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    step: int,
    config: dict,
    ema_state: dict | None = None,
    scaler_g: Any | None = None,
    scaler_d: Any | None = None,
) -> None:
    """Save a training checkpoint.

    The checkpoint contains everything needed to resume training or to
    evaluate the model later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "config": config,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }
    if ema_state is not None:
        state["ema"] = ema_state
    if scaler_g is not None:
        state["scaler_g"] = scaler_g.state_dict()
    if scaler_d is not None:
        state["scaler_d"] = scaler_d.state_dict()

    torch.save(state, str(path))


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    ema: Any | None = None,
    scaler_g: Any | None = None,
    scaler_d: Any | None = None,
    device: torch.device | str = "cpu",
) -> int:
    """Load a training checkpoint and return the training step.

    Optimizer, EMA, and scaler states are restored only when the
    corresponding objects are provided.
    """
    state = torch.load(str(path), map_location=device, weights_only=False)

    generator.load_state_dict(state["generator"])
    discriminator.load_state_dict(state["discriminator"])

    if opt_g is not None and "opt_g" in state:
        opt_g.load_state_dict(state["opt_g"])
    if opt_d is not None and "opt_d" in state:
        opt_d.load_state_dict(state["opt_d"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])
    if scaler_g is not None and "scaler_g" in state:
        scaler_g.load_state_dict(state["scaler_g"])
    if scaler_d is not None and "scaler_d" in state:
        scaler_d.load_state_dict(state["scaler_d"])

    return state["step"]


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a run directory (by step number)."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return ckpts[-1] if ckpts else None
