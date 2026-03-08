"""Image saving and manipulation utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torchvision.utils as vutils
from PIL import Image


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map tensor values from [-1, 1] to [0, 1]."""
    return tensor.mul(0.5).add(0.5).clamp(0.0, 1.0)


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
) -> None:
    """Save a batch of images as a single grid image.

    Args:
        images: Tensor of shape (N, C, H, W), expected in [-1, 1].
        path: Destination file path.
        nrow: Number of images per row in the grid.
        normalize: Whether to map from [-1,1] to [0,1].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if normalize:
        images = denormalize(images)
    vutils.save_image(images, str(path), nrow=nrow, padding=2)


def save_single_images(
    images: torch.Tensor,
    directory: str | Path,
    start_idx: int = 0,
    normalize: bool = True,
) -> int:
    """Save each image in the batch as an individual PNG file.

    Returns the next index (for sequential numbering across batches).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if normalize:
        images = denormalize(images)
    for i, img in enumerate(images):
        idx = start_idx + i
        # Convert to PIL and save
        ndarr = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(ndarr).save(directory / f"{idx:06d}.png")
    return start_idx + len(images)


def make_comparison_grid(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
) -> None:
    """Create a side-by-side comparison grid of two sets of images.

    Images from set A appear on even rows and set B on odd rows.
    Both tensors must have the same shape and be in [-1, 1].
    """
    assert images_a.shape == images_b.shape, "Both image sets must have the same shape"
    n = images_a.size(0)
    # Interleave: A[0], B[0], A[1], B[1], ...
    interleaved = torch.stack([images_a, images_b], dim=1).view(-1, *images_a.shape[1:])
    save_image_grid(interleaved, path, nrow=nrow * 2, normalize=True)


def fixed_latent_vectors(
    num: int,
    latent_dim: int,
    seed: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a fixed set of latent vectors (for reproducible comparison samples)."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(num, latent_dim, generator=rng)
    return z.to(device)
