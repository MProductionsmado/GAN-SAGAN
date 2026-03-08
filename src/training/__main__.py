"""CLI entry point for ``python -m src.training <command>``.

Supported commands::

    python -m src.training train    --config configs/sagan_256.yaml
    python -m src.training evaluate --run_dir runs/sagan_256
    python -m src.training compare  --run_a runs/gan_256 --run_b runs/sagan_256
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: python -m src.training <command> [options]\n\n"
            "Commands:\n"
            "  train     Train a GAN or SAGAN model\n"
            "  evaluate  Evaluate a trained model (FID / KID)\n"
            "  compare   Compare two trained runs (GAN vs SAGAN)\n"
        )
        sys.exit(0)

    command = sys.argv[1]
    # Remove the command from argv so sub-parsers see the right args
    remaining = sys.argv[2:]

    if command == "train":
        from ..utils.config import parse_args_and_config
        from .train import train
        cfg = parse_args_and_config(remaining)
        train(cfg)

    elif command == "evaluate":
        from .evaluate import main as eval_main
        eval_main(remaining)

    elif command == "compare":
        from .compare import main as compare_main
        compare_main(remaining)

    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, evaluate, compare")
        sys.exit(1)


if __name__ == "__main__":
    main()
