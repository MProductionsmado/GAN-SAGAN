# GAN vs Self-Attention GAN — Fair Comparison on CelebA-HQ

A research repository for a **methodologically fair comparison** between a standard convolutional GAN and a Self-Attention GAN (SAGAN) on **CelebA-HQ** at high resolution (128–512 px).

**Central research question:**
> What qualitative and quantitative difference does Self-Attention make in high-resolution face generation on CelebA-HQ, when the base architecture is held constant?

The **only controlled variable** is the presence of Self-Attention blocks. Everything else — architecture, hyperparameters, training pipeline, loss function, data augmentation — is identical.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare CelebA-HQ (see "Data Preparation" below)
python scripts/prepare_celebahq.py --source /path/to/celebahq --resolution 256

# 3. Train baseline GAN
python -m src.training train --config configs/gan_256.yaml

# 4. Train SAGAN
python -m src.training train --config configs/sagan_256.yaml

# 5. Evaluate both
python -m src.training evaluate --run_dir runs/gan_256
python -m src.training evaluate --run_dir runs/sagan_256

# 6. Compare
python -m src.training compare --run_a runs/gan_256 --run_b runs/sagan_256
```

---

## Repository Structure

```
├── configs/                      # YAML configuration files
│   ├── gan_128.yaml              # Baseline GAN 128×128
│   ├── gan_256.yaml              # Baseline GAN 256×256 (recommended)
│   ├── gan_512.yaml              # Baseline GAN 512×512
│   ├── sagan_128.yaml            # SAGAN 128×128
│   ├── sagan_256.yaml            # SAGAN 256×256 (recommended)
│   └── sagan_512.yaml            # SAGAN 512×512
├── data/
│   └── celebahq/train/           # Place CelebA-HQ images here
├── src/
│   ├── datasets/
│   │   └── celebahq.py           # Dataset loader + DataLoader factory
│   ├── models/
│   │   ├── attention.py          # Self-Attention module
│   │   ├── blocks.py             # Shared ResBlocks (GenBlock, DiscBlock)
│   │   ├── discriminator.py      # Discriminator network
│   │   ├── gan.py                # Factory function + EMA wrapper
│   │   └── generator.py          # Generator network
│   ├── training/
│   │   ├── __main__.py           # CLI entry point
│   │   ├── checkpoint.py         # Save/load checkpoints
│   │   ├── compare.py            # Compare two runs
│   │   ├── evaluate.py           # Evaluate a run (FID/KID)
│   │   ├── logger.py             # TensorBoard logger
│   │   ├── losses.py             # Hinge & non-saturating loss
│   │   └── train.py              # Main training loop
│   └── utils/
│       ├── config.py             # Config loading (YAML + CLI overrides)
│       ├── image.py              # Image saving utilities
│       ├── metrics.py            # FID/KID computation (clean-fid)
│       └── seed.py               # Deterministic seeding
├── scripts/
│   ├── prepare_celebahq.py       # Dataset preparation utility
│   ├── train_gan_256.sh          # Shell script: train baseline
│   ├── train_sagan_256.sh        # Shell script: train SAGAN
│   └── eval_compare.sh           # Shell script: evaluate + compare
├── tests/
│   ├── test_attention.py         # Self-Attention unit tests
│   └── test_models.py            # Model smoke tests (all resolutions)
├── outputs/                      # Generated images
├── reports/                      # Comparison reports
├── runs/                         # Training runs (logs, checkpoints, samples)
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Data Preparation

### CelebA-HQ

CelebA-HQ contains 30,000 high-quality face images. You can obtain it from:

1. **Kaggle**: [CelebA-HQ dataset](https://www.kaggle.com/datasets/lamsimon/celebahq)
2. **Google Drive**: [Official links from the PGGAN paper](https://github.com/tkarras/progressive_growing_of_gans)
3. **Academic mirrors**: Check your institution's data repositories

Once downloaded, prepare the dataset:

```bash
# If you have a folder of raw CelebA-HQ images:
python scripts/prepare_celebahq.py \
    --source /path/to/CelebA-HQ-img \
    --target data/celebahq/train \
    --resolution 256

# If images are already at the desired resolution, omit --resolution:
python scripts/prepare_celebahq.py \
    --source /path/to/celebahq_256 \
    --target data/celebahq/train
```

Expected layout after preparation:
```
data/celebahq/train/
  000000.png
  000001.png
  ...
  029999.png
```

### Using Other Datasets

The dataset loader accepts any flat directory of images. To use a different dataset (e.g., FFHQ, CelebA):

```bash
python -m src.training train --config configs/gan_256.yaml \
    --dataset.path=data/ffhq/train
```

---

## Architecture

### Base Architecture

Both models use an identical **residual convolutional architecture** inspired by SNGAN/SAGAN:

**Generator:**
- **Input**: Latent vector z ∈ ℝ^128
- **Projection**: Linear → 4×4 feature map
- **Upsampling**: Chain of residual blocks (GenBlock), each doubling spatial resolution via nearest-neighbour upsampling + Conv3×3
- **Normalisation**: BatchNorm in all generator blocks
- **Activation**: ReLU
- **Output**: BN → ReLU → Conv3×3 → Tanh → image ∈ [-1, 1]

**Discriminator:**
- **Input**: Image ∈ ℝ^{3×H×W}
- **Downsampling**: DiscOptBlock + chain of DiscBlocks, each halving spatial resolution via AvgPool2d
- **Normalisation**: Spectral Normalisation on all convolutional and linear layers
- **Activation**: ReLU
- **Output**: ReLU → Global Sum Pooling → Spectral Linear → scalar score

### Self-Attention Integration

The SAGAN variant inserts `SelfAttention` modules at configurable feature-map resolutions (default: 32×32 and 64×64):

- **Query/Key**: 1×1 convolution reducing channels by 8× (memory-efficient)
- **Value**: 1×1 convolution at full channel width
- **Attention**: Scaled dot-product attention over all spatial positions
- **Gate**: Learnable scalar γ initialised to 0 → `output = γ · attention(x) + x`
- **Spectral Norm**: Applied to all Q/K/V/output projection convolutions

**γ = 0 initialisation** means the network starts training identically to the baseline — Self-Attention is gradually learned.

### Channel Schedule (256×256, base_channels=64)

| Stage | Resolution | Generator Channels | Discriminator Channels |
|-------|------------|-------------------|----------------------|
| 0     | 4×4        | 1024              | 1024 (final)         |
| 1     | 8×8        | 512               | 512                  |
| 2     | 16×16      | 256               | 256                  |
| 3     | 32×32      | 128 ← *attn*      | 128 ← *attn*         |
| 4     | 64×64      | 64 ← *attn*       | 64 ← *attn*          |
| 5     | 128×128    | 32                | —                    |
| 6     | 256×256    | 3 (output)        | 3 (input)            |

---

## Fairness Between GAN and SAGAN

This repository guarantees a methodologically fair comparison:

| Aspect | Guarantee |
|--------|-----------|
| **Data pipeline** | Identical dataset, transforms, batch size |
| **Base architecture** | Same GenBlock / DiscBlock structure, same depth, same channel widths |
| **Normalisation** | Same BatchNorm (G) and SpectralNorm (D) everywhere |
| **Loss function** | Same hinge loss for both |
| **Optimiser** | Same Adam(β₁=0, β₂=0.999), same learning rates |
| **Training steps** | Same total_steps, same d_steps_per_g |
| **EMA** | Same decay rate, same update schedule |
| **Mixed precision** | Same AMP settings |
| **Seeds** | Same random seed for reproducibility |
| **Evaluation** | Same FID computation, same number of generated images |
| **Comparison samples** | Same latent vectors for visual comparison |

**The ONLY difference**: presence/absence of `SelfAttention` modules. The parameter count difference is small and consists exclusively of the attention projections (Q/K/V/output, 1×1 convolutions) and the learnable γ scalar.

---

## Training

### Recommended Configuration: 256×256

```bash
# Baseline GAN
python -m src.training train --config configs/gan_256.yaml

# SAGAN
python -m src.training train --config configs/sagan_256.yaml
```

### CLI Overrides

Any config parameter can be overridden from the command line:

```bash
python -m src.training train --config configs/sagan_256.yaml \
    --training.batch_size=16 \
    --training.total_steps=200000 \
    --seed=123
```

### Resuming Training

Training automatically resumes from the latest checkpoint in the run directory.

### Multi-GPU

```bash
python -m src.training train --config configs/sagan_256.yaml \
    --training.multi_gpu=true
```

### Monitoring

TensorBoard logs are saved in `runs/<experiment_name>/tb/`:

```bash
tensorboard --logdir runs/
```

---

## Evaluation

### Compute FID

```bash
python -m src.training evaluate --run_dir runs/sagan_256

# With KID:
python -m src.training evaluate --run_dir runs/sagan_256 --kid

# Custom number of images:
python -m src.training evaluate --run_dir runs/sagan_256 --num_images 50000
```

### Compare Two Runs

```bash
python -m src.training compare \
    --run_a runs/gan_256 \
    --run_b runs/sagan_256 \
    --output reports/comparison_256
```

This generates:
- `comparison.png` — side-by-side grid (same latent vectors)
- `samples_a.png` / `samples_b.png` — individual grids
- `report.md` — human-readable comparison table
- `report.yaml` / `report.json` — machine-readable report

### Full Pipeline (Shell Script)

```bash
bash scripts/eval_compare.sh
```

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_models.py -v
python -m pytest tests/test_attention.py -v
```

Tests verify:
- Forward pass at 128/256/512 for both GAN and SAGAN
- Output shapes and value ranges
- Architecture parity (parameter differences = attention only)
- Gradient flow through the full pipeline
- Self-Attention gamma initialisation and spectral normalisation

---

## VRAM Requirements

| Resolution | Batch Size | Approx. VRAM (GAN) | Approx. VRAM (SAGAN) |
|------------|-----------|--------------------|--------------------|
| 128×128    | 64        | ~8 GB              | ~10 GB             |
| 256×256    | 32        | ~12 GB             | ~16 GB             |
| 512×512    | 8         | ~18 GB             | ~24 GB             |

**512×512 is memory-intensive.** With Self-Attention at 64×64 feature maps, the attention matrix alone requires significant memory. Recommendations:
- Use a GPU with ≥24 GB VRAM (A100, 3090, 4090)
- Reduce `attention_resolutions` to `[32]` only
- Reduce `batch_size` if OOM
- Mixed precision is enabled by default and helps significantly

**Recommended:** Start with **256×256** (configs `gan_256.yaml` / `sagan_256.yaml`).

---

## Architecture Design Decisions

| Decision | Rationale |
|----------|-----------|
| Residual blocks | Standard in SNGAN/SAGAN/BigGAN; stable at high resolutions |
| Nearest-neighbour upsample + Conv | Avoids checkerboard artefacts from transposed convolutions |
| AvgPool downsample | Smoother feature maps than strided convolutions |
| Hinge loss | Default in SAGAN paper; well-studied, stable training |
| SpectralNorm (D only by default) | Enforces Lipschitz constraint; as in the original SAGAN paper |
| BatchNorm in Generator | Standard conditioning; consistent for both variants |
| Attention at 32×32 and 64×64 | Mid-level feature maps balance local detail and global coherence |
| γ = 0 initialisation | Attention starts as identity → gradual integration during training |
| Step-based training | Standard in GAN research; independent of dataset size |
| EMA generator | Smooths training variance; better sample quality for evaluation |
| Adam(β₁=0, β₂=0.999) | SAGAN paper defaults; β₁=0 avoids momentum interference with spectral norm |

---

## Limitations

- **No progressive growing**: Models train at fixed resolution. Very high resolutions (512+) may be slower to converge than progressive approaches.
- **No class conditioning**: This is unconditional generation only.
- **Memory at 512×512**: Self-Attention at 64×64 feature maps creates large attention matrices. Consider restricting to 32×32 only for 512px training.
- **Dataset size**: CelebA-HQ has 30k images, which is relatively small. Longer training may lead to overfitting — the EMA generator and FID monitoring help detect this.
- **Single loss function**: While non-saturating loss is available, the default comparison uses hinge loss. Switching losses requires retraining both models.

---

## License

This project is for research purposes. Please cite the relevant papers if you use this code:

- *Self-Attention Generative Adversarial Networks* (Zhang et al., 2019)
- *Spectral Normalization for Generative Adversarial Networks* (Miyato et al., 2018)
- *CelebA-HQ* from *Progressive Growing of GANs* (Karras et al., 2018)
