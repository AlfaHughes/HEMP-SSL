# HEMP: Hyperbolic Extreme-Multi-Patch Self-Supervised Learning

## Overview

HEMP extends the EMP-SSL (Extreme-Multi-Patch Self-Supervised Learning) framework by introducing a **Poincaré ball projection head** alongside the standard Euclidean projection. The key insight is that representations embedded in hyperbolic space can better capture hierarchical semantic structures, while still being trained with stable Euclidean contrastive losses—avoiding the optimization instabilities of full hyperbolic loss formulations.

The backbone is **CRATE** (Coding Rate Reduction Transformer), a Vision Transformer variant where each block consists of a multi-head Self-Attention (SSA) layer and an ISTA-based sparse Feed-Forward layer, designed to learn maximally discriminative and compressive representations through the Maximal Coding Rate Reduction (MCR²) principle.

## Method

### Architecture

```
Input Image (3×H×W)
    │
    ▼
CRATE Backbone (12 layers, dim=384, heads=6, dim_head=64)
    │   SSA: multi-head self-attention
    │   ISTA: sparse feed-forward (soft-thresholding)
    ▼
Pre-feature: Linear(384→512) + BN + ReLU
    │
    ├──────────────────────┬───────────────────────────┐
    │  Euclidean (default) │  Hyperbolic (--use_hyperbolic_proj)
    │  Linear(512→512)     │  HyperbolicProjectionHead
    │  + BN + ReLU          │    Linear(512→512) + BN + ReLU
    │  Linear(512→z_dim)   │    Linear(512→z_dim)
    │  L2-normalize         │    expmap0 (tanh soft-compress)
    └──────────────────────┴───────────────────────────┘
```

### Loss Function (shared by both variants)

$$\mathcal{L} = \underbrace{-\frac{\lambda}{V}\sum_{v=1}^{V}\cos(\boldsymbol{z}^{(v)}, \bar{\boldsymbol{z}})}_{\text{Invariance (cosine)}} \;+\; \underbrace{-\frac{1}{V}\sum_{v=1}^{V} \frac{1}{2}\log\det\!\Big(I + \frac{d}{N\varepsilon}\,Z^{(v)}Z^{(v)\top}\Big)}_{\text{TCR (Form B, per-view logdet)}}$$

- **Invariance term**: pulls V augmented views of the same image toward their mean representation.
- **TCR (Total Coding Rate)**: maximizes the discriminative coding rate per view. Computed as Form B (per-view Gram matrix, then averaged)—no cross-view terms appear.
- Both terms operate on Euclidean coordinates. For the hyperbolic variant, these coordinates lie inside the Poincaré ball (output of `expmap0`), but the loss computation itself is Euclidean.

### Hyperbolic Projection: expmap0 with Two-Level Boundary Control

The exponential map from the tangent space (origin) to the Poincaré ball uses a **tanh soft-compression** mechanism to ensure numerical stability and non-vanishing gradients:

```
Tangent vector u (unconstrained)
    │
    ▼  Level 1: tanh soft-compress the norm
    │  scaled_norm = tanh(‖u‖ / max_norm) × max_norm
    │  max_norm = 1/√κ − ε   (safety margin from ball boundary)
    │  → ‖u_scaled‖ ∈ [0, max_norm), gradient never zero
    │
    ▼  Level 2: exponential map (tanh guarantees ‖z‖ < 1/√κ)
    │  z = [tanh(√κ·‖u_scaled‖) / (√κ·‖u_scaled‖)] · u_scaled
    │  → z strictly inside Poincaré ball, no hard clamp needed
    ▼
Poincaré ball point z (‖z‖ < 1/√κ)
```

Key design choices:
- **No L2 normalization after expmap0**: radial information (distance from origin encodes hierarchy depth) is preserved. Normalizing would collapse all points to a fixed-radius shell, destroying the hyperbolic geometry.
- **No RSGD needed**: all learnable parameters (backbone, MLPs) live in Euclidean space. The exponential map is differentiable via standard PyTorch autograd, so standard AdamW suffices.

## Installation

```bash
conda create -n hemp python=3.8
conda activate hemp
pip install torch torchvision einops tqdm scikit-learn nltk
```

## Usage

### 1. Pre-training

**Euclidean baseline:**
```bash
python EMP_main.py \
    --data cifar10 --arch crate-test \
    --num_patches 20 --bs 100 --lr 1e-3 --epoch 30 \
    --patch_sim 200 --dir emp_euc
```

**Hyperbolic variant:**
```bash
python EMP_main.py \
    --data cifar10 --arch crate-test \
    --num_patches 20 --bs 100 --lr 1e-3 --epoch 30 \
    --patch_sim 200 --dir emp_hyp_proj \
    --use_hyperbolic_proj --hyp_curvature 0.2 \
    --hyp_hidden_dim 512 --hyp_out_dim 128
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | cifar10 | Dataset: cifar10, cifar100, tinyimagenet |
| `--arch` | crate-test | Architecture: crate-test, resnet18-cifar |
| `--num_patches` | 100 | Number of augmented views per image (V) |
| `--bs` | 100 | Batch size (images per step) |
| `--lr` | 1e-3 | Learning rate for AdamW |
| `--epoch` | 30 | Total training epochs |
| `--warmup_epoch` | 5 | Linear warmup epochs |
| `--patch_sim` | 200 | Weight λ for invariance (cosine) loss |
| `--eps` | 0.2 | ε for TCR |
| `--use_hyperbolic_proj` | False | Enable Poincaré ball projection head |
| `--hyp_curvature` | 0.3 | Curvature κ of the Poincaré ball |
| `--hyp_hidden_dim` | 512 | Hidden dimension of projection MLP |
| `--hyp_out_dim` | 128 | Output dimension z_dim |

### 2. Fine-tuning / Linear Probe

```bash
# Euclidean weights
python evaluate_finetune.py --data cifar10 --arch crate-test \
    --model_path "./logs/emp_euc/.../save_models/best.pt"

# Hyperbolic weights (must enable --use_hyperbolic)
python evaluate_finetune.py --data cifar10 --arch crate-test \
    --use_hyperbolic \
    --model_path "./logs/emp_hyp_proj/.../save_models/best.pt"
```

### 3. SCI Verification (Semantic Continuity Index)

Generate semantic similarity matrices (first time only):
```bash
# Tiny-ImageNet: WordNet Wu-Palmer similarity
python tinyimagenet_semantic.py --data_path E:/DATA/TinyImgenet

# CIFAR-100: superclass affinity or WordNet
python cifar100_semantic.py
```

Verify feature-semantic alignment:
```bash
python feature_distance_verify.py \
    --dataset cifar10 --model crate \
    --checkpoint "path/to/best.pt" \
    --data_path ./DATA
```


## Project Structure

```
EMP-SSL-main2/
├── EMP_main.py                 # Main training entry (Euclidean loss + optional hyperbolic proj)
├── evaluate_finetune.py        # Fine-tuning / linear probe (--use_hyperbolic flag)
├── feature_distance_verify.py  # SCI verification + t-SNE + distance matrix
├── hyperbolic_utils.py         # expmap0, logmap0, mobius_add, full hyperbolic losses
```

## Datasets

| Dataset | Classes | Resolution | Image Size | Patch Tokens (incl. CLS) |
|---------|---------|------------|------------|--------------------------|
| CIFAR-10 | 10 | 32×32 | 32 | 17 (4×4+1) |
| CIFAR-100 | 100 | 32×32 | 32 | 17 |
| Tiny-ImageNet | 200 | 64×64 | 64 | 65 (8×8+1) |

## Training Configuration

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW (lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4 on weights only) |
| Scheduler | 5-epoch linear warmup → cosine annealing to 1e-5 |
| Mixed Precision | AMP with autocast (TCR forced to fp32 for logdet stability) |
| Gradient Clipping | max_norm=1.0 |
| Effective Batch | bs × num_patches (e.g., 100×100=10,000 views/step) |

## Key Design Decisions

1. **Euclidean loss on hyperbolic representations**: The loss (cosine + TCR) operates on the Euclidean coordinates of Poincaré ball points. This avoids Möbius mean instability, unbounded TCR in direct mode, and the need for RSGD—all of which caused training failures in the full hyperbolic loss formulation.

2. **tanh soft-compression over hard clamp**: Replaced `torch.clamp` with `tanh` in `expmap0`. Hard clamping produces zero gradients at the boundary, blocking backpropagation and freezing the loss at a symmetric constant. The tanh mapping ensures non-vanishing gradients everywhere.

3. **No post-exp normalization**: L2-normalizing after `expmap0` would project all points onto a fixed-radius shell, destroying radial information that encodes hierarchical depth in hyperbolic space. Radial freedom is preserved; only the TCR term (which sees `ZZ^T`) can shape it.

4. **Form B TCR**: Coding rate is computed per-view (per augmentation) and averaged, not on the union matrix. This avoids implicit cross-view structure in the second moment and keeps each view's Gram matrix independently well-conditioned.

5. **Parameter grouping**: AdamW applies weight decay only to weight matrices (dim ≥ 2), excluding biases and BatchNorm parameters, following standard Transformer training practice.
