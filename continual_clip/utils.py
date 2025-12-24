import os
import json
from pathlib import Path
import torch
import yaml

from omegaconf import DictConfig, OmegaConf
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F



def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]


def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_class_names(classes_names, class_ids_per_task):
    return [classes_names[class_id] for class_id in class_ids_per_task]


def get_dataset_class_names(workdir, dataset_name, long=False):
    with open(os.path.join(workdir, "dataset_reqs", f"{dataset_name}_classes.txt"), "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]

def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path):
    split_path = list(Path(path).resolve().parts)
    candidates = ["RAPF", "rapf-engine"] # If a 'ValueError' occurs, replace 'rapf_engine' with your actual work directory
    workdir_idx = next(
        (i for i, part in enumerate(split_path) if part in candidates),
        None
    )
    return str(Path(*split_path[:workdir_idx+1]))



def flatten_adapter_params(self, adapter):
    params = []
    for name, param in adapter.named_parameters():
        if isinstance(param, nn.Parameter):
            params.append(param.data.flatten())
    return torch.cat(params)

def unflatten_adapter_params(self, adapter, flat_vector):
    pointer = 0
    for name, param in adapter.named_parameters():
        if isinstance(param, nn.Parameter):
            num_elements = param.numel()
            param.data.copy_(flat_vector[pointer:pointer + num_elements].view_as(param.data))
            pointer += num_elements
    return adapter