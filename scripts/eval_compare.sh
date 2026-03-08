#!/usr/bin/env bash
# Evaluate both runs and generate a comparison report.
#
# Usage:
#   bash scripts/eval_compare.sh [--kid]
#
# Expects that both runs/gan_256 and runs/sagan_256 exist with checkpoints.
set -euo pipefail

EXTRA_ARGS="${@}"

echo "=== Evaluating Baseline GAN ==="
python -m src.training evaluate \
    --run_dir runs/gan_256 \
    --num_images 10000 \
    --batch_size 64 \
    ${EXTRA_ARGS}

echo ""
echo "=== Evaluating SAGAN ==="
python -m src.training evaluate \
    --run_dir runs/sagan_256 \
    --num_images 10000 \
    --batch_size 64 \
    ${EXTRA_ARGS}

echo ""
echo "=== Generating Comparison Report ==="
python -m src.training compare \
    --run_a runs/gan_256 \
    --run_b runs/sagan_256 \
    --output reports/gan_vs_sagan_256

echo ""
echo "Done. See reports/gan_vs_sagan_256/ for results."
