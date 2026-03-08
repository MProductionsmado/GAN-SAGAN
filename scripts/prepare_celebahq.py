#!/usr/bin/env python3
"""Prepare CelebA-HQ images for training (with optional auto-download).

This script organises CelebA-HQ images into the expected layout::

    data/celebahq/train/<images>

Default behaviour is convenient for first-time setup:
* If ``--source`` already contains images, it prepares them.
* If ``--source`` is missing or empty, it automatically downloads CelebA-HQ
  from Kaggle and then prepares it.

Examples
--------
1) One-command setup (download + prepare)::

    python scripts/prepare_celebahq.py --source /path/to/celebahq --resolution 256

2) Use an existing local image folder (no download)::

    python scripts/prepare_celebahq.py --source /path/to/CelebA-HQ-img --resolution 256 --download never

Kaggle auth
-----------
For automatic download, set up the Kaggle API once:
* Create API token on kaggle.com (Account -> API -> Create New Token)
* Place ``kaggle.json`` in ``~/.kaggle/kaggle.json``
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from tqdm import tqdm

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_DATASET = "lamsimon/celebahq"


def _collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)


def _auto_download_with_kaggle_cli(source_dir: Path, dataset: str) -> Path | None:
    """Download dataset via Kaggle CLI and return a directory containing images."""
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        return None

    source_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading '{dataset}' via Kaggle CLI into: {source_dir}")

    cmd = [
        kaggle_bin,
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(source_dir),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Kaggle CLI download failed ({exc}).")
        return None

    if _collect_images(source_dir):
        return source_dir

    # Some archives unpack to nested directories.
    for candidate in sorted(source_dir.rglob("*")):
        if candidate.is_dir() and _collect_images(candidate):
            return candidate
    return None


def _auto_download_with_kagglehub(dataset: str) -> Path | None:
    """Download dataset via kagglehub and return a directory containing images."""
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None

    print(f"[INFO] Downloading '{dataset}' via kagglehub ...")
    try:
        downloaded_path = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        print(f"[WARN] kagglehub download failed ({exc}).")
        return None

    if _collect_images(downloaded_path):
        return downloaded_path
    for candidate in sorted(downloaded_path.rglob("*")):
        if candidate.is_dir() and _collect_images(candidate):
            return candidate
    return None


def resolve_source(
    source: Path,
    download_mode: str,
    dataset: str,
) -> Path:
    """Resolve a usable source directory, downloading when requested/needed."""
    source_exists_with_images = source.exists() and bool(_collect_images(source))

    if download_mode == "never":
        if not source_exists_with_images:
            raise RuntimeError(
                f"No images found in {source} and download_mode='never'."
            )
        return source

    should_download = download_mode == "always" or not source_exists_with_images
    if not should_download:
        return source

    print("[INFO] Source directory is empty/missing or forced download is enabled.")

    downloaded_source = _auto_download_with_kaggle_cli(source, dataset)
    if downloaded_source is not None:
        print(f"[INFO] Download successful. Using source: {downloaded_source}")
        return downloaded_source

    downloaded_source = _auto_download_with_kagglehub(dataset)
    if downloaded_source is not None:
        print(f"[INFO] Download successful. Using source: {downloaded_source}")
        return downloaded_source

    raise RuntimeError(
        "Automatic download failed. Please ensure Kaggle credentials are configured "
        "(~/.kaggle/kaggle.json) and either `kaggle` CLI or `kagglehub` is installed."
    )


def prepare(source: Path, target: Path, resolution: int | None) -> None:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    images = _collect_images(source)
    if not images:
        raise RuntimeError(f"No images found in {source}")

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
        help="Local source dir with images OR target download dir (if auto-download is needed)",
    )
    parser.add_argument(
        "--target", type=str, default="data/celebahq/train",
        help="Target directory (default: data/celebahq/train)",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Resize images to this resolution (e.g. 256). If omitted, copies as-is.",
    )
    parser.add_argument(
        "--download",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help="Download behaviour: auto (default), always, never",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug for auto-download (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()

    source = resolve_source(Path(args.source), args.download, args.dataset)
    prepare(source, Path(args.target), args.resolution)


if __name__ == "__main__":
    main()
