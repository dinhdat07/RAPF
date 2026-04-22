# SIGMA-CLIP: Statistical Inference with Gaussian Memory Anchors

Implementation for **"Statistical Memory Head for Class-Incremental Vision-Language Learning with CLIP" (ICME 2026 submission)**, with method naming updated to **SIGMA**.

This repository trains frozen CLIP with lightweight adapters and performs inference-time calibration with SIGMA using class-wise Gaussian statistics in the frozen CLIP feature space.

## Method Overview

The pipeline has two components:

1. **Adapter-based incremental training**
- Frozen CLIP image/text encoders.
- Sequential image/text linear adapters (single active pair updated task-by-task).
- Auxiliary losses:
  - CLIP contrastive loss,
  - image augmentation consistency,
  - text anchor regularization,
  - hard-pair hinge separation.

2. **Inference-time SIGMA calibration**
- Per-class mean/covariance estimated from frozen CLIP image features.
- Shared precision update across tasks.
- LDA-style statistical logits fused with adapter discriminative logits.

## Repository Structure

- `sigma_clip/`: SIGMA-centric implementation.
  - `cli.py`: Hydra entrypoint.
  - `trainer.py`: class-incremental train/eval loop.
  - `model.py`: CLIP + adapters + replay + statistics logic.
  - `data.py`: dataset/scenario construction.
  - `utils.py`: class orders, workdir helpers, SIGMA logit fusion.
  - `losses.py`: CLIP/contrastive loss helpers.
- `main.py`: Hydra entrypoint for SIGMA training.
- `configs/class/`: experiment configs.
- `class_orders/`: class order definitions.
- `metadata/class_names/`: dataset class-name files used for prompt construction.

## Installation

### 1. Environment

```bash
conda create -n sigma_clip python=3.8
conda activate sigma_clip
```

### 2. Dependencies

```bash
bash setup_environment.sh
```

This installs PyTorch (CUDA 11.1 build), Python requirements, and OpenAI CLIP.

## Dataset Preparation

### CIFAR-100
- Downloaded automatically by Continuum when running.
- Set `dataset_root` to a writable folder.

### ImageNet-R
Expected layout:

```text
imagenet-r/
├── train/
│   ├── class_a/
│   └── ...
└── test/
    ├── class_a/
    └── ...
```

Class order file: `class_orders/imagenet_R_order.yaml`.
Class names file: `metadata/class_names/imagenet_R_classes.txt`.

## Training

All runs use Hydra via `main.py`.

### CIFAR-100 (example)

```bash
python main.py \
  --config-dir configs/class \
  --config-name cifar100_10-10.yaml \
  dataset_root="/path/to/cifar_root" \
  class_order="class_orders/cifar100_order.yaml"
```

### ImageNet-R (example)

```bash
python main.py \
  --config-dir configs/class \
  --config-name imagenet_r_20-20.yaml \
  dataset_root="/path/to/imagenet-r" \
  class_order="class_orders/imagenet_R_order.yaml"
```

## Evaluation and Outputs

Evaluation runs after each incremental task during training.

Main output file:
- `metric.json` (or configured `log_path`) with per-task and final records.

Per-task record fields:
- `task`, `train_acc`, `test_acc`, `avg_acc`, `forgetting`, `acc_per_task`, `bwt`, `fwt`

Final record fields:
- `last`, `avg`

Hydra also saves resolved config to `config.yaml` in the run directory.

## Key Configs

Core keys (top-level):
- `dataset`, `dataset_root`, `class_order`
- `initial_increment`, `increment`
- `train_batch_size`, `batch_size`, `epochs`, `lr`, `seed`
- `threshold`, `beta`, `shrinkage`

SIGMA/adaptation keys:
- `lambda_img`, `lambda_txt`, `sample_num`, `sample_noise`, `stat`, `templates`

## Reproducibility Notes

- Global seeds are fixed for Python/NumPy/PyTorch.
- CLIP backbone is frozen; only adapter/fusion-related parameters are trainable per task.
- For strict before/after comparisons, use identical:
  - config,
  - class order,
  - seed,
  - dataset root/splits,
  - CUDA environment.

## Acknowledgement

This implementation was originally built on top of prior continual CLIP codebases and then refactored to the SIGMA method structure.

## License

Same as repository license terms.
