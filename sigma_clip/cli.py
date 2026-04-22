from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .trainer import run_class_incremental, seed_everything
from .utils import get_workdir, save_config


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REQUIRED_KEYS = [
    "class_order",
    "dataset_root",
    "log_path",
    "model_name",
    "prompt_template",
    "batch_size",
    "initial_increment",
    "increment",
    "dataset",
    "num_workers",
    "train_batch_size",
    "epochs",
    "lr",
    "fp16",
    "seed",
    "beta",
    "threshold",
    "shrinkage",
    "lambda_img",
    "lambda_txt",
    "sample_num",
    "stat",
    "sample_noise",
    "templates",
]


def normalize_runtime_config(cfg: DictConfig) -> DictConfig:
    normalized = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(normalized, False)

    missing = [key for key in REQUIRED_KEYS if OmegaConf.select(normalized, key) is None]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return normalized


@hydra.main(config_path=None, config_name=None, version_base="1.1")
def run_sigma(cfg: DictConfig) -> None:
    cfg = normalize_runtime_config(cfg)
    seed_everything(cfg.seed)

    cfg.workdir = get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    save_config(cfg)
    with open(cfg.log_path, "w+", encoding="utf-8"):
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_class_incremental(cfg, device)


if __name__ == "__main__":
    run_sigma()
