#!/usr/bin/env bash
# Train SAGAN (Self-Attention GAN) at 256×256 on CelebA-HQ
set -euo pipefail

echo "=== Training SAGAN (256×256) ==="
python -m src.training train --config configs/sagan_256.yaml "$@"
