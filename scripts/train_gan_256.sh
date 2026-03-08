#!/usr/bin/env bash
# Train baseline GAN at 256×256 on CelebA-HQ
set -euo pipefail

echo "=== Training Baseline GAN (256×256) ==="
python -m src.training train --config configs/gan_256.yaml "$@"
