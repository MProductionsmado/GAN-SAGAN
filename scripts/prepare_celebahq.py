#!/usr/bin/env python3
"""Prepare CelebA-HQ images for training.

This script organises CelebA-HQ images into the expected directory layout::

    data/celebahq/train/<images>

Usage examples
--------------
1. You already have a folder of CelebA-HQ images (PNG/JPG)::

       python scripts/prepare_celebahq.py --source /path/to/celebahq_pngs --resolution 256

   This resizes (Lanczos) and copies the images to ``data/celebahq/train/``.

2. You have the raw ``CelebAMask-HQ`` dataset with ``CelebA-HQ-img/``::

       python scripts/prepare_celebahq.py --source /path/to/CelebAMask-HQ/CelebA-HQ-img --resolution 256

Notes
-----
* The script does **not** download the dataset automatically.  See the README
  for download instructions (Kaggle, Google Drive, academic mirrors).
* Running the script again skips images that already exist in the target folder.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def prepare(source: Path, target: Path, resolution: int | None) -> None:
    source = Path(source)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        print(f"[ERROR] No images found in {source}")
        return

    print(f"Found {len(images)} images in {source}")
    print(f"Target directory: {target}")
    if resolution:
        print(f"Resizing to {resolution}x{resolution} (Lanczos)")

    skipped = 0
    for img_path in tqdm(images, desc="Preparing"):
        dst = target / f"{img_path.stem}.png"
        if dst.exists():
            skipped += 1
            continue
        if resolution:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((resolution, resolution), Image.LANCZOS)
            img.save(dst, "PNG")
        else:
            shutil.copy2(img_path, dst)

    total = len(images) - skipped
    print(f"Done. Copied/resized {total} images, skipped {skipped} existing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CelebA-HQ dataset")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Source directory containing CelebA-HQ images",
    )
    parser.add_argument(
        "--target", type=str, default="data/celebahq/train",
        help="Target directory (default: data/celebahq/train)",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Resize images to this resolution (e.g. 256). If omitted, copies as-is.",
    )
    args = parser.parse_args()
    prepare(Path(args.source), Path(args.target), args.resolution)


if __name__ == "__main__":
    main()
