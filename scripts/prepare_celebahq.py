#!/usr/bin/env python3
"""Prepare CelebA-HQ images for training (with optional auto-download).

This script organises CelebA-HQ images into the expected layout::

    data/celebahq/train/<images>

It works in three phases, each with progress output:

  1. **Download** (if needed) — fetches the dataset ZIP from Kaggle
  2. **Extract**  — unpacks the ZIP with a progress bar
  3. **Prepare**  — resizes / copies images into the target directory

Examples
--------
One-command setup (download + extract + prepare)::

    python scripts/prepare_celebahq.py --resolution 256

Use an existing local folder (skip download)::

    python scripts/prepare_celebahq.py --source /path/to/images --resolution 256 --download never

Kaggle auth
-----------
For automatic download, set up the Kaggle API once:

1. Create API token: kaggle.com → Account → API → *Create New Token*
2. Place ``kaggle.json`` in ``~/.kaggle/kaggle.json``
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_DATASET = "lamsimon/celebahq"
STAGING_DIR = Path("data/celebahq/_staging")


# ── helpers ───────────────────────────────────────────────────────────────

def _collect_images(root: Path, verbose: bool = False) -> list[Path]:
    """Recursively collect image files under *root*."""
    if verbose:
        print(f"      Scanning {root} …", end=" ", flush=True)
    result = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if verbose:
        print(f"{len(result)} images found.")
    return result


def _find_image_root(base: Path) -> Path:
    """Return *base* itself or the first sub-directory that contains images."""
    if _collect_images(base):
        return base
    for candidate in sorted(base.rglob("*")):
        if candidate.is_dir() and _collect_images(candidate):
            return candidate
    return base  # fallback


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a ZIP archive with a tqdm progress bar."""
    print(f"\n[2/3] Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, dest)
    print(f"      Extracted {len(members)} entries to {dest}")


# ── download strategies ──────────────────────────────────────────────────

def _download_with_kaggle_cli(staging: Path, dataset: str) -> Path | None:
    """Download ZIP via Kaggle CLI (without --unzip) and extract ourselves."""
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        return None

    staging.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Downloading '{dataset}' via Kaggle CLI …")

    cmd = [
        kaggle_bin, "datasets", "download",
        "-d", dataset,
        "-p", str(staging),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Kaggle CLI download failed ({exc}).")
        return None

    # Find the downloaded ZIP
    zips = sorted(staging.glob("*.zip"))
    if not zips:
        # Kaggle may have auto-extracted — look for images directly
        return _find_image_root(staging) if _collect_images(staging) else None

    _extract_zip(zips[0], staging)
    zips[0].unlink()  # clean up ZIP to save disk space

    return _find_image_root(staging)


def _download_with_kagglehub(dataset: str) -> Path | None:
    """Download via kagglehub and return image directory."""
    try:
        import kagglehub  # type: ignore
    except ImportError:
        return None

    print(f"[1/3] Downloading '{dataset}' via kagglehub …")
    try:
        downloaded = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        print(f"[WARN] kagglehub download failed ({exc}).")
        return None

    return _find_image_root(downloaded)


# ── source resolution ────────────────────────────────────────────────────

def resolve_source(
    source: Path | None,
    download_mode: str,
    dataset: str,
) -> Path:
    """Return a directory that contains CelebA-HQ images, downloading if needed."""

    has_images = source is not None and source.exists() and bool(_collect_images(source))

    if download_mode == "never":
        if not has_images:
            raise RuntimeError(
                f"No images found in '{source}' and --download=never."
            )
        print("[1/3] Using existing source — skipping download.")
        return source  # type: ignore[return-value]

    if has_images and download_mode != "always":
        print(f"[1/3] Found images in {source} — skipping download.")
        return source  # type: ignore[return-value]

    print("[INFO] No local images found (or --download=always). Starting download …\n")

    # Use a dedicated staging dir so we never mix raw downloads with the target
    staging = STAGING_DIR
    result = _download_with_kaggle_cli(staging, dataset)
    if result is not None:
        return result

    result = _download_with_kagglehub(dataset)
    if result is not None:
        return result

    raise RuntimeError(
        "Automatic download failed.\n"
        "  • Ensure Kaggle credentials exist at ~/.kaggle/kaggle.json\n"
        "  • Install either the `kaggle` CLI or the `kagglehub` package\n"
        "  • Or download manually and pass --source /path/to/images --download never"
    )


# ── prepare (resize / copy) ──────────────────────────────────────────────

def prepare(source: Path, target: Path, resolution: int | None) -> None:
    """Copy / resize images from *source* into *target*."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    source_resolved = source.resolve()
    target_resolved = target.resolve()
    inplace = source_resolved == target_resolved

    images = _collect_images(source, verbose=True)
    if not images:
        raise RuntimeError(f"No images found in {source}")

    if inplace:
        # source == target: resize images in-place
        if not resolution:
            print(f"\n[3/3] {len(images)} images already in {target}, no resize requested — nothing to do.")
            return
        print(f"\n[3/3] Resizing {len(images)} images in-place to {resolution}×{resolution} (Lanczos)")
        resized = 0
        for img_path in tqdm(images, desc="Resizing", unit="img"):
            img = Image.open(img_path).convert("RGB")
            if img.size == (resolution, resolution):
                continue
            img = img.resize((resolution, resolution), Image.LANCZOS)
            dst = img_path.with_suffix(".png")
            img.save(dst, "PNG")
            # Remove original if it had a different extension
            if dst != img_path:
                img_path.unlink()
            resized += 1
        print(f"      Done — {resized} images resized, {len(images) - resized} already correct size.")
        return

    # source != target: copy/resize into target
    print(f"\n[3/3] Preparing {len(images)} images → {target}")
    if resolution:
        print(f"      Resizing to {resolution}×{resolution} (Lanczos)")

    skipped = 0
    for img_path in tqdm(images, desc="Preparing", unit="img"):
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

    processed = len(images) - skipped
    print(f"      Done — {processed} images written, {skipped} skipped (existing).")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare CelebA-HQ for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Auto-download + resize to 256×256:\n"
            "  python scripts/prepare_celebahq.py --resolution 256\n\n"
            "  # Use local images, no download:\n"
            "  python scripts/prepare_celebahq.py --source my_imgs/ --resolution 256 --download never\n"
        ),
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Local dir with images. If omitted or empty, dataset is downloaded.",
    )
    parser.add_argument(
        "--target", type=str, default="data/celebahq/train",
        help="Output directory for prepared images (default: data/celebahq/train)",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Resize images to this square size (e.g. 256). Omit to copy as-is.",
    )
    parser.add_argument(
        "--download", type=str, default="auto",
        choices=["auto", "always", "never"],
        help="Download mode: auto (default) | always | never",
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--keep-staging", action="store_true",
        help="Keep the staging directory after preparation (default: delete)",
    )
    args = parser.parse_args()

    source = resolve_source(
        Path(args.source) if args.source else None,
        args.download,
        args.dataset,
    )
    prepare(source, Path(args.target), args.resolution)

    # Cleanup staging
    if not args.keep_staging and STAGING_DIR.exists():
        print(f"\nCleaning up staging directory ({STAGING_DIR}) …")
        shutil.rmtree(STAGING_DIR)
        print("Done.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
