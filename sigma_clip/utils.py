from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def get_class_order(file_name: str) -> list:
    with open(file_name, "r+", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data["class_order"]


def get_class_ids_per_task(cfg: DictConfig) -> Iterable[List[int]]:
    yield cfg.class_order[: cfg.initial_increment]
    for index in range(cfg.initial_increment, len(cfg.class_order), cfg.increment):
        yield cfg.class_order[index : index + cfg.increment]


def get_class_names(class_names: List[str], class_ids_per_task: List[int]) -> List[str]:
    return [class_names[class_id] for class_id in class_ids_per_task]


def get_dataset_class_names(workdir: str, dataset_name: str, long: bool = False) -> List[str]:  # noqa: ARG001
    class_file = Path(workdir) / "metadata" / "class_names" / f"{dataset_name}_classes.txt"
    if not class_file.exists():
        raise FileNotFoundError(
            f"Class-name file for '{dataset_name}' not found at: {class_file}"
        )
    with open(class_file, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path: str) -> str:
    split_path = list(Path(path).resolve().parts)
    candidates = ["sigma"]
    workdir_idx = next((index for index, part in enumerate(split_path) if part in candidates), None)
    if workdir_idx is None:
        return str(Path(path).resolve())
    return str(Path(*split_path[: workdir_idx + 1]))


def sigma_logit_fusion(model, outputs, raw_image_features, cfg):
    with torch.no_grad():
        outputs_sigma = raw_image_features @ model.W + model.b
        outputs_sigma = outputs_sigma / outputs_sigma.norm(dim=-1, keepdim=True)

        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        outputs = (outputs_sigma * cfg.stat) + outputs * (1 - cfg.stat)

    return outputs

