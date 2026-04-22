from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig

from .config import normalize_runtime_config
from .trainer import run_class_incremental, seed_everything
from .utils import get_workdir, save_config


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


@hydra.main(config_path=None, config_name=None, version_base="1.1")
def continual_clip(cfg: DictConfig) -> None:
    cfg = normalize_runtime_config(cfg)
    seed_everything(cfg.seed)

    cfg.workdir = get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    save_config(cfg)
    with open(cfg.log_path, "w+", encoding="utf-8"):
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.scenario == "class":
        run_class_incremental(cfg, device)


if __name__ == "__main__":
    continual_clip()
