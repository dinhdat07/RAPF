import os

from continual_clip.utils import gda_output
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
import statistics
from torch.utils.data import DataLoader
import torch.nn.functional as F
from continuum.metrics import Logger

from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import ClassIncrementalCLIP, sample
from continual_clip.losses import contrastive_loss
from continual_clip.datasets import build_cl_scenarios
import numpy as np

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_class_incremental(cfg, device):

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    
    model = ClassIncrementalCLIP(cfg, device)
    for name, p in model.named_parameters():
        if ("visual" in name or "transformer" in name or "token_embedding" in name):
            p.requires_grad = False
    model.update_injection_units()

    eval_dataset, classes_names = build_cl_scenarios(cfg, is_train=False, base_transforms=model.transforms)
    train_dataset, _ = build_cl_scenarios(cfg, is_train=True, base_transforms=model.transforms)
    model.classes_names = classes_names
    acc_list = []
    metric_logger = Logger(list_subsets=["train", "test"])

    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Eval for task {task_id} has started.")
        model.adaptation(task_id, threshold=cfg.threshold)
        model.update_stat(known_classes=model.known_classes,total_classes=len(model.total_class_names),train_loader=train_loader,device=device) 
        model.eval()

        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, _, __, ___, _pre_image_feas, raw_image_feas = model(inputs)
                torch.nn.functional.softmax(outputs, dim=-1)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")


        test_acc = 100 * metric_logger.accuracy  
        avg_acc = 100 * metric_logger.average_incremental_accuracy
        forgetting_val = 100 * metric_logger.forgetting
        acc_per_task = [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task]
        bwt = 100 * metric_logger.backward_transfer
        fwt = 100 * metric_logger.forward_transfer

        train_acc = 100 * metric_logger.online_accuracy if hasattr(metric_logger, "online_accuracy") else None
        acc_list.append(test_acc)
        print(
            f"[Task {task_id}] "
            f"test_acc={test_acc:.2f} | avg_acc={avg_acc:.2f} | "
            f"forgetting={forgetting_val:.6f}"
        )

        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'train_acc': round(train_acc, 2) if train_acc is not None else None,
                'test_acc': round(test_acc, 2),
                'avg_acc': round(avg_acc, 2),
                'forgetting': round(forgetting_val, 6),
                'acc_per_task': acc_per_task,
                'bwt': round(bwt, 2),
                'fwt': round(fwt, 2),
            }) + '\n')
            metric_logger.end_task()

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')



@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    with open(cfg.log_path, 'w+') as f: 
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.scenario == "class":
        run_class_incremental(cfg, device)

    
if __name__ == "__main__":
    continual_clip()