"""Configuration loading and management."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Default configuration – every key that the codebase may reference lives here
# so that partial YAML files always produce a complete config dict.
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "experiment_name": "gan_256",

    # Dataset
    "dataset": {
        "name": "celebahq",
        "path": "data/celebahq/train",
        "resolution": 256,
        "num_workers": 4,
    },

    # Model
    "model": {
        "latent_dim": 128,
        "base_channels": 64,
        "use_attention": False,
        "attention_resolutions": [32, 64],
    },

    # Training
    "training": {
        "batch_size": 32,
        "total_steps": 100_000,
        "d_steps_per_g": 2,
        "loss_type": "hinge",           # "hinge" | "nonsaturating"
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "beta1": 0.0,
        "beta2": 0.999,
        "use_ema": True,
        "ema_decay": 0.999,
        "mixed_precision": True,
        "multi_gpu": False,
    },

    # Logging / checkpointing
    "logging": {
        "log_dir": "runs",
        "sample_freq": 1000,
        "checkpoint_freq": 5000,
        "eval_freq": 10000,
        "num_fid_images": 10000,
        "num_sample_images": 64,
    },

    # Reproducibility
    "seed": 42,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``--key=value`` style overrides (dot-notation) to *cfg*.

    Examples::

        --training.batch_size=16
        --model.use_attention=true
        --seed=123
    """
    for token in overrides:
        if "=" not in token:
            continue
        key_path, raw_value = token.lstrip("-").split("=", 1)
        keys = key_path.split(".")

        # Navigate to parent dict
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})

        # Infer type from existing value (if present)
        existing = d.get(keys[-1])
        if isinstance(existing, bool):
            value: Any = raw_value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            value = int(raw_value)
        elif isinstance(existing, float):
            value = float(raw_value)
        elif isinstance(existing, list):
            # Parse "[32,64]" or "32,64"
            raw_value = raw_value.strip("[]")
            value = [int(v.strip()) for v in raw_value.split(",") if v.strip()]
        else:
            value = raw_value

        d[keys[-1]] = value
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load a YAML config file, merge with defaults, and apply CLI overrides."""
    cfg = copy.deepcopy(_DEFAULTS)
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, file_cfg)
    if overrides:
        cfg = _apply_cli_overrides(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Persist a config dict as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def config_to_flat(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict to dot-notation keys (for logging)."""
    flat: dict[str, Any] = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(config_to_flat(v, full_key))
        else:
            flat[full_key] = v
    return flat


def build_arg_parser(description: str = "GAN vs SAGAN") -> argparse.ArgumentParser:
    """Create a minimal CLI parser that accepts ``--config`` plus arbitrary overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser


def parse_args_and_config(argv: list[str] | None = None) -> dict:
    """Parse CLI args, load the referenced config, and merge overrides."""
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_config(args.config, overrides=unknown)
    return cfg
