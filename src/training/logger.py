"""TensorBoard-based training logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils.image import denormalize


class Logger:
    """Thin wrapper around TensorBoard ``SummaryWriter``.

    Provides convenience methods for the metrics we care about in GAN training.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    # ---- scalars ----------------------------------------------------------
    def log_losses(
        self,
        step: int,
        d_loss: float,
        g_loss: float,
        d_real: float | None = None,
        d_fake: float | None = None,
    ) -> None:
        self.writer.add_scalar("loss/discriminator", d_loss, step)
        self.writer.add_scalar("loss/generator", g_loss, step)
        if d_real is not None:
            self.writer.add_scalar("loss/d_real", d_real, step)
        if d_fake is not None:
            self.writer.add_scalar("loss/d_fake", d_fake, step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    # ---- images -----------------------------------------------------------
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
        nrow: int = 8,
    ) -> None:
        """Log a batch of images (expected in [-1,1])."""
        images = denormalize(images)
        from torchvision.utils import make_grid
        grid = make_grid(images, nrow=nrow, padding=2)
        self.writer.add_image(tag, grid, step)

    # ---- hyperparameters --------------------------------------------------
    def log_config(self, cfg: dict[str, Any]) -> None:
        """Log flattened config as text."""
        from ..utils.config import config_to_flat
        flat = config_to_flat(cfg)
        text = "\n".join(f"- **{k}**: {v}" for k, v in flat.items())
        self.writer.add_text("config", text, 0)

    # ---- cleanup ----------------------------------------------------------
    def close(self) -> None:
        self.writer.close()
