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


# def get_workdir(path):
#     split_path = path.split("/")
#     workdir_idx = split_path.index("RAPF") # If a 'ValueError' occurs, replace 'RAPF' with your actual work directory
#     return "/".join(split_path[:workdir_idx+1])

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


def engine_rerank(model, outputs, raw_image_feas, device, epoch, cfg):
    """
    Rerank batch outputs dựa trên GDA + knowledge injection.

    Parameters:
    - outputs: logits batch hiện tại (batch_size x num_classes)
    - raw_image_feas: đặc trưng ảnh từ backbone CLIP (chưa qua injection)
    - device: CPU/GPU
    - epoch: epoch hiện tại
    - cfg: config
    """
    model.eval()
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

            # Kết hợp GDA + rerank + original outputs
            outputs = (
                outputs_gda * cfg.engine.stat
                + (cfg.engine.rerank * outputs_rerank
                   + (1 - cfg.engine.rerank) * outputs) * (1 - cfg.engine.stat)
            )
        else:
            # Nếu chưa đến tuned_epoch, trả về outputs bình thường
            outputs = outputs

    return outputs
