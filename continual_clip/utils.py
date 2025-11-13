import os
import json
from pathlib import Path
import torch
import yaml

from omegaconf import DictConfig, OmegaConf
from typing import Optional



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

def get_engine_descriptor_path(workdir: str, dataset_name: str) -> Optional[str]:
    mapping = {
        'cifar100': os.path.join('chat', 'cifar224_des.json'),
        'imagenet_r': os.path.join('chat', 'imagenetr_des.json'),
        'cub200': os.path.join('chat', 'cub_des.json'),
    }
    dataset_key = dataset_name.lower() if dataset_name else ''
    candidate = mapping.get(dataset_key)
    if not candidate:
        return None
    candidate_path = os.path.normpath(os.path.join(workdir, candidate))
    return candidate_path if os.path.isfile(candidate_path) else None

def normalize_key(name: str):
    return name.replace("_", " ").lower()

def engine_rerank(model, outputs, raw_image_feas, device, cfg):

    with torch.no_grad():
        if hasattr(cfg, "epochs") and epoch == cfg.epochs:
            # GDA classifier
            outputs_gda = raw_image_feas @ model.W + model.b
            outputs_gda = outputs_gda / outputs_gda.norm(dim=-1, keepdim=True)

        # Rerank batch
        outputs_rerank = model.rerank(
            des_dict=model.des_dict,
            outputs=outputs,
            image_features_raw=raw_image_feas,
            class_names=model.total_class_names,
            device=device,
            topk=cfg.engine.topk
        )

        # GDA + rerank + original outputs
        outputs = (
            outputs_gda * cfg.engine.stat
            + (cfg.engine.rerank * outputs_rerank
                + (1 - cfg.engine.rerank) * outputs) * (1 - cfg.engine.stat)
        )

    return outputs


def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2  = 1
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_


def sample(mean, cov, size, shrink=False):
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec