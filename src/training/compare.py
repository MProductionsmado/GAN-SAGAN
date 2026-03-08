"""Compare two training runs (GAN vs SAGAN) and produce a summary report.

Usage::

    python -m src.training compare \\
        --run_a runs/gan_256 \\
        --run_b runs/sagan_256 \\
        --output reports/comparison_256.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..models.gan import ExponentialMovingAverage, build_gan, count_parameters, model_summary
from ..training.checkpoint import find_latest_checkpoint, load_checkpoint
from ..utils.config import load_config, save_config
from ..utils.image import fixed_latent_vectors, make_comparison_grid, save_image_grid
from ..utils.seed import seed_everything


def _load_run(run_dir: Path, device: torch.device):
    """Load config + generator from a run directory."""
    cfg = load_config(run_dir / "config.yaml")
    gen, disc = build_gan(cfg)
    gen.to(device)
    disc.to(device)

    ema = None
    if cfg["training"].get("use_ema", False):
        ema = ExponentialMovingAverage(gen, decay=cfg["training"].get("ema_decay", 0.999))

    ckpt = find_latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    load_checkpoint(ckpt, gen, disc, ema=ema, device=device)

    gen.eval()
    return cfg, gen, ema, ckpt


def compare(
    run_a: str | Path,
    run_b: str | Path,
    output_dir: str | Path | None = None,
    num_compare: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two runs and generate a report.

    Args:
        run_a: First run directory (typically baseline GAN).
        run_b: Second run directory (typically SAGAN).
        output_dir: Where to save comparison artefacts. Defaults to
                    ``reports/<run_a_name>_vs_<run_b_name>``.
        num_compare: Number of images to generate for visual comparison.
        seed: Seed for the fixed latent vectors (ensures identical inputs).

    Returns:
        A dict with the full comparison data.
    """
    run_a = Path(run_a)
    run_b = Path(run_b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading run A: {run_a}")
    cfg_a, gen_a, ema_a, ckpt_a = _load_run(run_a, device)

    print(f"Loading run B: {run_b}")
    cfg_b, gen_b, ema_b, ckpt_b = _load_run(run_b, device)

    # Determine output directory
    if output_dir is None:
        output_dir = Path("reports") / f"{run_a.name}_vs_{run_b.name}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared latent vectors
    latent_dim = cfg_a["model"]["latent_dim"]
    z = fixed_latent_vectors(num_compare, latent_dim, seed=seed, device=device)

    # Generate comparison samples
    print("Generating comparison samples ...")
    with torch.no_grad():
        if ema_a is not None:
            with ema_a.apply():
                samples_a = gen_a(z)
        else:
            samples_a = gen_a(z)

        if ema_b is not None:
            with ema_b.apply():
                samples_b = gen_b(z)
        else:
            samples_b = gen_b(z)

    # Save grids
    save_image_grid(samples_a, output_dir / "samples_a.png", nrow=8)
    save_image_grid(samples_b, output_dir / "samples_b.png", nrow=8)
    make_comparison_grid(samples_a, samples_b, output_dir / "comparison.png", nrow=8)

    # Collect metrics from eval results (if available)
    metrics_a = _load_metrics(run_a)
    metrics_b = _load_metrics(run_b)

    # Summaries
    gen_a_raw = gen_a.module if hasattr(gen_a, "module") else gen_a
    gen_b_raw = gen_b.module if hasattr(gen_b, "module") else gen_b
    summary_a = model_summary(gen_a_raw, torch.nn.Module())  # disc not needed for param count
    summary_b = model_summary(gen_b_raw, torch.nn.Module())

    # Load training summaries if available
    train_summary_a = _load_yaml(run_a / "summary.yaml")
    train_summary_b = _load_yaml(run_b / "summary.yaml")

    report = {
        "run_a": {
            "name": run_a.name,
            "path": str(run_a),
            "checkpoint": str(ckpt_a),
            "config": cfg_a,
            "generator_params": summary_a["generator_params"],
            "use_attention": cfg_a["model"]["use_attention"],
            "resolution": cfg_a["dataset"]["resolution"],
            "metrics": metrics_a,
            "training_summary": train_summary_a,
        },
        "run_b": {
            "name": run_b.name,
            "path": str(run_b),
            "checkpoint": str(ckpt_b),
            "config": cfg_b,
            "generator_params": summary_b["generator_params"],
            "use_attention": cfg_b["model"]["use_attention"],
            "resolution": cfg_b["dataset"]["resolution"],
            "metrics": metrics_b,
            "training_summary": train_summary_b,
        },
        "comparison": {
            "same_architecture": True,
            "isolated_variable": "Self-Attention",
            "num_comparison_images": num_compare,
            "comparison_seed": seed,
            "output_dir": str(output_dir),
            "param_difference": summary_b["generator_params"] - summary_a["generator_params"],
        },
    }

    # Save report as YAML and Markdown
    save_config(report, output_dir / "report.yaml")
    _write_markdown_report(report, output_dir / "report.md")

    # Also save as JSON for programmatic access
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nComparison report saved to {output_dir}")
    print(f"  - report.md")
    print(f"  - report.yaml")
    print(f"  - report.json")
    print(f"  - comparison.png")
    print(f"  - samples_a.png / samples_b.png")

    return report


def _load_metrics(run_dir: Path) -> dict:
    """Load eval metrics from a run directory (if available)."""
    metrics_path = run_dir / "eval" / "metrics.yaml"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _write_markdown_report(report: dict, path: Path) -> None:
    """Write a human-readable Markdown comparison report."""
    a = report["run_a"]
    b = report["run_b"]
    c = report["comparison"]

    lines = [
        "# GAN vs SAGAN Comparison Report",
        "",
        "## Runs",
        "",
        f"| | Run A (Baseline) | Run B (SAGAN) |",
        f"|---|---|---|",
        f"| Name | {a['name']} | {b['name']} |",
        f"| Resolution | {a['resolution']} | {b['resolution']} |",
        f"| Use Attention | {a['use_attention']} | {b['use_attention']} |",
        f"| Generator Params | {a['generator_params']:,} | {b['generator_params']:,} |",
        f"| Param Difference | — | +{c['param_difference']:,} |",
        "",
    ]

    # Metrics table
    if a["metrics"] or b["metrics"]:
        lines.extend([
            "## Metrics",
            "",
            "| Metric | Run A | Run B |",
            "|---|---|---|",
        ])
        for key in ["fid", "kid"]:
            va = a["metrics"].get(key, "—")
            vb = b["metrics"].get(key, "—")
            if isinstance(va, float):
                va = f"{va:.2f}"
            if isinstance(vb, float):
                vb = f"{vb:.2f}"
            lines.append(f"| {key.upper()} | {va} | {vb} |")
        lines.append("")

    # Training summary
    if a["training_summary"] or b["training_summary"]:
        lines.extend([
            "## Training",
            "",
            "| | Run A | Run B |",
            "|---|---|---|",
        ])
        for key in ["total_steps", "training_time_seconds"]:
            va = a["training_summary"].get(key, "—")
            vb = b["training_summary"].get(key, "—")
            label = key.replace("_", " ").title()
            if key == "training_time_seconds" and isinstance(va, (int, float)):
                va = f"{va / 3600:.1f}h"
            if key == "training_time_seconds" and isinstance(vb, (int, float)):
                vb = f"{vb / 3600:.1f}h"
            lines.append(f"| {label} | {va} | {vb} |")
        lines.append("")

    lines.extend([
        "## Fairness",
        "",
        f"- **Isolated variable**: {c['isolated_variable']}",
        f"- **Same architecture**: {c['same_architecture']}",
        f"- **Comparison seed**: {c['comparison_seed']}",
        f"- **Comparison images**: {c['num_comparison_images']}",
        "",
        "## Artefacts",
        "",
        f"- Comparison grid: `comparison.png`",
        f"- Run A samples: `samples_a.png`",
        f"- Run B samples: `samples_b.png`",
        f"- Run A checkpoint: `{a['checkpoint']}`",
        f"- Run B checkpoint: `{b['checkpoint']}`",
        "",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two GAN runs")
    parser.add_argument("--run_a", type=str, required=True,
                        help="Path to first run (e.g. runs/gan_256)")
    parser.add_argument("--run_b", type=str, required=True,
                        help="Path to second run (e.g. runs/sagan_256)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for report (default: reports/<a>_vs_<b>)")
    parser.add_argument("--num_images", type=int, default=64,
                        help="Number of comparison images")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for latent vectors")
    args = parser.parse_args(argv)

    compare(
        run_a=args.run_a,
        run_b=args.run_b,
        output_dir=args.output,
        num_compare=args.num_images,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
