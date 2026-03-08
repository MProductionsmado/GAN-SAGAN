"""Evaluation: generate images and compute FID / KID for a trained run.

Usage::

    python -m src.training evaluate --config configs/sagan_256.yaml \\
        --run_dir runs/sagan_256 --checkpoint best
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from ..models.gan import ExponentialMovingAverage, build_gan
from ..training.checkpoint import find_latest_checkpoint, load_checkpoint
from ..utils.config import load_config, save_config
from ..utils.image import save_single_images
from ..utils.metrics import compute_fid, compute_kid
from ..utils.seed import seed_everything


def evaluate(
    run_dir: str | Path,
    num_images: int = 10_000,
    batch_size: int = 64,
    compute_kid_flag: bool = False,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Generate images and compute metrics for a single training run.

    Args:
        run_dir: Path to the run directory (must contain ``config.yaml``
                 and ``checkpoints/``).
        num_images: Number of images to generate for FID computation.
        batch_size: Generation batch size.
        compute_kid_flag: Whether to also compute KID.
        checkpoint_path: Specific checkpoint file. If *None*, uses the latest.

    Returns:
        Dict with metric results and paths.
    """
    run_dir = Path(run_dir)

    # Load config from the run directory
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    gen, disc = build_gan(cfg)
    gen.to(device)
    disc.to(device)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(run_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Build EMA if used
    ema = None
    if cfg["training"].get("use_ema", False):
        ema = ExponentialMovingAverage(gen, decay=cfg["training"].get("ema_decay", 0.999))

    load_checkpoint(checkpoint_path, gen, disc, device=device, ema=ema)

    # Use EMA weights for generation if available
    gen.eval()

    # Output directory
    eval_dir = run_dir / "eval"
    gen_dir = eval_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    latent_dim = cfg["model"]["latent_dim"]
    print(f"Generating {num_images} images ...")
    t0 = time.time()

    idx = 0
    remaining = num_images
    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, latent_dim, device=device)
            if ema is not None:
                with ema.apply():
                    imgs = gen(z)
            else:
                imgs = gen(z)
            idx = save_single_images(imgs, gen_dir, start_idx=idx)
            remaining -= bs

    gen_time = time.time() - t0
    print(f"Generated {num_images} images in {gen_time:.1f}s")

    # Compute FID
    real_dir = cfg["dataset"]["path"]
    print(f"Computing FID (real: {real_dir}, gen: {gen_dir}) ...")
    fid_score = compute_fid(gen_dir, real_dir, device=device, batch_size=batch_size)
    print(f"FID: {fid_score:.2f}")

    results: dict = {
        "fid": fid_score,
        "num_images": num_images,
        "checkpoint": str(checkpoint_path),
        "generation_time_seconds": gen_time,
        "generated_dir": str(gen_dir),
    }

    # Optional KID
    if compute_kid_flag:
        print("Computing KID ...")
        kid_score = compute_kid(gen_dir, real_dir, device=device, batch_size=batch_size)
        print(f"KID: {kid_score:.4f}")
        results["kid"] = kid_score

    # Save results
    save_config(results, eval_dir / "metrics.yaml")
    print(f"Results saved to {eval_dir / 'metrics.yaml'}")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GAN / SAGAN")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to run directory")
    parser.add_argument("--num_images", type=int, default=10_000,
                        help="Number of images to generate for FID")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    parser.add_argument("--kid", action="store_true",
                        help="Also compute KID")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint (defaults to latest)")
    args = parser.parse_args(argv)

    evaluate(
        run_dir=args.run_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        compute_kid_flag=args.kid,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
