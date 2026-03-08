"""Main training loop for GAN / SAGAN.

Trains a Generator and Discriminator on CelebA-HQ (or compatible image
dataset) using the configuration provided via a YAML file.

Usage::

    python -m src.training train --config configs/sagan_256.yaml
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from ..datasets.celebahq import build_dataloader
from ..models.gan import (
    ExponentialMovingAverage,
    build_gan,
    count_parameters,
    model_summary,
)
from ..training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from ..training.logger import Logger
from ..training.losses import get_loss_fns
from ..utils.config import load_config, save_config
from ..utils.image import fixed_latent_vectors, save_image_grid
from ..utils.seed import seed_everything


def _infinite_loader(loader: torch.utils.data.DataLoader):
    """Yield batches from *loader* indefinitely (step-based training)."""
    while True:
        yield from loader


def train(cfg: dict) -> None:
    """Run the full training procedure described by *cfg*."""

    # ------------------------------------------------------------------ seed
    seed_everything(cfg["seed"])

    # ------------------------------------------------------------------ dirs
    run_name = cfg["experiment_name"]
    run_dir = Path(cfg["logging"]["log_dir"]) / run_name
    sample_dir = run_dir / "samples"
    ckpt_dir = run_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    save_config(cfg, run_dir / "config.yaml")

    # ---------------------------------------------------------------- device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------- dataset
    loader = build_dataloader(
        cfg["dataset"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    data_iter = _infinite_loader(loader)
    print(f"Dataset: {len(loader.dataset)} images, "
          f"batch_size={cfg['training']['batch_size']}")

    # --------------------------------------------------------------- models
    gen, disc = build_gan(cfg)
    gen.to(device)
    disc.to(device)

    summary = model_summary(gen, disc)
    print(f"Generator params:     {summary['generator_params']:,}")
    print(f"Discriminator params: {summary['discriminator_params']:,}")
    print(f"Use attention:        {summary['use_attention']}")

    # Multi-GPU (optional)
    tcfg = cfg["training"]
    if tcfg.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        disc = nn.DataParallel(disc)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # ----------------------------------------------------------- optimisers
    opt_g = torch.optim.Adam(
        gen.parameters(),
        lr=tcfg["lr_g"],
        betas=(tcfg["beta1"], tcfg["beta2"]),
    )
    opt_d = torch.optim.Adam(
        disc.parameters(),
        lr=tcfg["lr_d"],
        betas=(tcfg["beta1"], tcfg["beta2"]),
    )

    # --------------------------------------------------------- EMA (optional)
    ema: ExponentialMovingAverage | None = None
    if tcfg.get("use_ema", False):
        raw_gen = gen.module if isinstance(gen, nn.DataParallel) else gen
        ema = ExponentialMovingAverage(raw_gen, decay=tcfg.get("ema_decay", 0.999))

    # ---------------------------------------------------- mixed precision
    use_amp = tcfg.get("mixed_precision", False) and device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------------------------------------------------------- losses
    d_loss_fn, g_loss_fn = get_loss_fns(tcfg.get("loss_type", "hinge"))

    # ---------------------------------------------------------- logger
    logger = Logger(run_dir / "tb")
    logger.log_config(cfg)

    # ------------------------------------------------- fixed latents for vis
    lcfg = cfg["logging"]
    num_vis = lcfg.get("num_sample_images", 64)
    z_fixed = fixed_latent_vectors(num_vis, cfg["model"]["latent_dim"],
                                   seed=cfg["seed"], device=device)

    # ------------------------------------------------ resume from checkpoint
    start_step = 0
    latest_ckpt = find_latest_checkpoint(run_dir)
    if latest_ckpt is not None:
        print(f"Resuming from {latest_ckpt}")
        start_step = load_checkpoint(
            latest_ckpt, gen, disc, opt_g, opt_d,
            ema=ema,
            scaler_g=scaler_g if use_amp else None,
            scaler_d=scaler_d if use_amp else None,
            device=device,
        )
        start_step += 1  # continue from next step

    # ============================================================ TRAINING
    total_steps = tcfg["total_steps"]
    d_steps = tcfg.get("d_steps_per_g", 2)
    latent_dim = cfg["model"]["latent_dim"]

    t0 = time.time()
    pbar = tqdm(range(start_step, total_steps), initial=start_step,
                total=total_steps, desc="Training", dynamic_ncols=True)

    for step in pbar:
        gen.train()
        disc.train()

        # ---- Discriminator update(s) ----
        d_loss_accum = 0.0
        for _ in range(d_steps):
            real = next(data_iter).to(device, non_blocking=True)
            z = torch.randn(real.size(0), latent_dim, device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    fake = gen(z)
                d_real = disc(real)
                d_fake = disc(fake)
                loss_d = d_loss_fn(d_real, d_fake)

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            d_loss_accum += loss_d.item()

        d_loss_avg = d_loss_accum / d_steps

        # ---- Generator update ----
        z = torch.randn(cfg["training"]["batch_size"], latent_dim, device=device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            fake = gen(z)
            g_scores = disc(fake)
            loss_g = g_loss_fn(g_scores)

        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        # ---- EMA update ----
        if ema is not None:
            ema.update()

        # ---- Logging ----
        g_loss_val = loss_g.item()
        pbar.set_postfix(d_loss=f"{d_loss_avg:.4f}", g_loss=f"{g_loss_val:.4f}")
        logger.log_losses(step, d_loss_avg, g_loss_val)

        # ---- Sample grids ----
        if (step + 1) % lcfg["sample_freq"] == 0 or step == 0:
            gen.eval()
            raw_gen = gen.module if isinstance(gen, nn.DataParallel) else gen
            with torch.no_grad():
                if ema is not None:
                    with ema.apply():
                        samples = raw_gen(z_fixed)
                else:
                    samples = raw_gen(z_fixed)
            save_image_grid(samples, sample_dir / f"step_{step + 1:07d}.png",
                            nrow=8)
            logger.log_images("samples", samples[:16], step)
            gen.train()

        # ---- Checkpoints ----
        if (step + 1) % lcfg["checkpoint_freq"] == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{step + 1:07d}.pt"
            save_checkpoint(
                ckpt_path,
                gen.module if isinstance(gen, nn.DataParallel) else gen,
                disc.module if isinstance(disc, nn.DataParallel) else disc,
                opt_g, opt_d, step,
                cfg,
                ema_state=ema.state_dict() if ema else None,
                scaler_g=scaler_g if use_amp else None,
                scaler_d=scaler_d if use_amp else None,
            )
            print(f"\n  Checkpoint saved: {ckpt_path}")

    # ============================================================ DONE
    elapsed = time.time() - t0
    print(f"\nTraining complete. {total_steps} steps in {elapsed / 3600:.1f}h")

    # Final checkpoint
    final_path = ckpt_dir / f"checkpoint_{total_steps:07d}.pt"
    if not final_path.exists():
        save_checkpoint(
            final_path,
            gen.module if isinstance(gen, nn.DataParallel) else gen,
            disc.module if isinstance(disc, nn.DataParallel) else disc,
            opt_g, opt_d, total_steps - 1,
            cfg,
            ema_state=ema.state_dict() if ema else None,
            scaler_g=scaler_g if use_amp else None,
            scaler_d=scaler_d if use_amp else None,
        )

    # Final samples
    gen.eval()
    raw_gen = gen.module if isinstance(gen, nn.DataParallel) else gen
    with torch.no_grad():
        if ema is not None:
            with ema.apply():
                final_samples = raw_gen(z_fixed)
        else:
            final_samples = raw_gen(z_fixed)
    save_image_grid(final_samples, sample_dir / "final.png", nrow=8)

    # Save training summary
    summary["training_time_seconds"] = elapsed
    summary["total_steps"] = total_steps
    save_config(summary, run_dir / "summary.yaml")

    logger.close()
    print(f"Run directory: {run_dir}")
