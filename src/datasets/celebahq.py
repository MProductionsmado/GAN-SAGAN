"""CelebA-HQ dataset loader and DataLoader factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebAHQDataset(Dataset):
    """Simple image-folder dataset for CelebA-HQ (or any flat image directory).

    Expects a directory filled with image files (PNG, JPG, JPEG, WEBP).
    No subdirectory structure is required, but nested layouts are supported.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        root: str | Path,
        resolution: int = 256,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}\n"
                "Please place your CelebA-HQ images into this folder. "
                "See README.md for instructions."
            )

        # Collect all image paths (recursively)
        self.image_paths: list[Path] = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.root}. "
                f"Supported extensions: {self.EXTENSIONS}"
            )

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),                       # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],       # → [-1, 1]
                                 [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def build_dataloader(
    dataset_cfg: dict,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from dataset configuration.

    Args:
        dataset_cfg: Dict with keys ``path``, ``resolution``, ``num_workers``.
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        A PyTorch DataLoader yielding image tensors of shape (B, 3, H, W).
    """
    ds = CelebAHQDataset(
        root=dataset_cfg["path"],
        resolution=dataset_cfg["resolution"],
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=dataset_cfg.get("num_workers", 4) > 0,
    )
