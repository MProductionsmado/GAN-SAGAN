"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all relevant RNGs for reproducibility.

    Sets Python, NumPy, and PyTorch seeds and configures cuDNN for
    deterministic behaviour.  Note that fully deterministic GPU training
    may incur a performance penalty.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
