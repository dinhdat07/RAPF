
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pdb
import random
import hydra
import logging
from omegaconf import DictConfig

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger

from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import load_model, sample
from continual_clip.datasets import build_cl_scenarios
from continual_clip.offline_merge import (
    apply_merge,
    compute_task_vectors,
    load_thetas,
    merge_la_magmax,
    merge_magmax,
    select_base_index,
    prune_topk
)
import numpy as np

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_class_incremental(cfg, device):

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )

    train_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    model.classes_names = classes_names
    offline_cfg = getattr(cfg, "offline_merge", None)
    offline_enabled = offline_cfg is not None and offline_cfg.enabled
    evaluate_per_task = not (offline_enabled and offline_cfg.eval_after_merge_only)
    theta_dir = None
    in_memory_thetas = []
    if offline_enabled:
        theta_dir = os.path.join(cfg.workdir, offline_cfg.theta_dir)
        os.makedirs(theta_dir, exist_ok=True)

    
    if offline_enabled and getattr(offline_cfg, "merge_only", False):
        # 1) Load thetas
        num_tasks = len(eval_dataset)
        thetas = load_thetas(os.path.join(cfg.workdir, offline_cfg.theta_dir), num_tasks)

        # 2) Base + vectors
        base_idx = select_base_index(offline_cfg.base_task, num_tasks)
        base_theta = thetas[base_idx]
        task_vectors = compute_task_vectors(thetas, base_idx)

        # 3) Merge
        if offline_cfg.merge_mode == "magmax":
            merged_vector = merge_magmax(task_vectors)
        else:
            with open(os.path.join(cfg.workdir, offline_cfg.theta_dir, "difficulty.json"), "r") as f:
                d = json.load(f)

            diffs = d["task_difficulties"]
            diffs = [v for j, v in enumerate(diffs) if j != base_idx]   # align with task_vectors
            merged_vector = merge_la_magmax(task_vectors, diffs, offline_cfg.gamma)
           
        # 4) Top-k
        merged_vector = prune_topk(merged_vector, offline_cfg.topk_ratio)

        # 5) Apply
        theta_merged = apply_merge(base_theta, merged_vector, offline_cfg.lambda_merge)

        # 6) Load adapter
        device_theta = {k: v.to(model.adapter.weight.device).type(model.adapter.weight.dtype)
                        for k, v in theta_merged.items()}
        model.adapter.load_state_dict(device_theta)

        # 7) Eval (task-agnostic on ALL tasks)
        model.adaptation(num_tasks - 1, threshold=cfg.threshold)
        model.eval()
        final_metric_logger = Logger(list_subsets=["test"])

        
        for tid in range(num_tasks):
            eval_loader = DataLoader(
                eval_dataset[tid],
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            for inputs, targets, task_ids in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs, _, __, ___ = model(inputs)
                final_metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

        final_acc = round(100 * final_metric_logger.accuracy, 2)
        print(f"MERGE_ONLY(base={offline_cfg.base_task}, mode={offline_cfg.merge_mode}) acc: {final_acc}")
        return

    acc_list = []
    saved_theta_paths = []
    task_difficulties = []
    task_hard_pairs = []
    metric_logger = Logger(list_subsets=["test"]) if evaluate_per_task else None
    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Train for task {task_id} has started.")
        model.adaptation(task_id, threshold=cfg.threshold)
        if offline_enabled:
            task_difficulties.append(model.last_task_difficulty)
            task_hard_pairs.append(model.last_num_hard_pairs)
        train_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        # epoch
        model.train()
        optimizer = torch.optim.Adam(model.adapter.parameters(), lr=cfg.lr, weight_decay=0.0000)

        milestones = cfg.milestones
        epochs = cfg.epochs

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        for i_epoch in range(epochs):
            loss = torch.tensor(0.0).to(device)
            loss_c = torch.tensor(0.0).to(device)
            loss_hinge = torch.tensor(0.0).to(device)
            tqdm_loader = tqdm(train_loader)
            if task_id >0:
                random_class_order_list = list(range(cfg.initial_increment+(task_id-1)*cfg.increment))
                random.shuffle(random_class_order_list)
            batch_id = -1
            for inputs, targets, task_ids in tqdm_loader:
                batch_id += 1
                inputs, targets = inputs.to(device), targets.to(device)
                sg_inputs = None
                edge_sample = None
                ori_targets = targets.clone()
                if task_id > 0:
                    sg_inputs = []
                    sg_targets = []
                    # num of classes per batch. Ensure an epoch traverses all classes at least once. 
                    # For exemple, if there are 100 classes and 50 batches per epoch , there will be 2 classes per batch.
                    if cfg.dataset == "cifar100" and cfg.increment == 5:
                        list_for_one_batch = [random_class_order_list[batch_id*4%len(random_class_order_list)], random_class_order_list[(batch_id*4+1)%len(random_class_order_list)], random_class_order_list[(batch_id*4+2)%len(random_class_order_list)], random_class_order_list[(batch_id*4+3)%len(random_class_order_list)]]
                    elif cfg.dataset == "imagenet_R":
                        list_for_one_batch = [random_class_order_list[batch_id*5%len(random_class_order_list)], random_class_order_list[(batch_id*5+1)%len(random_class_order_list)], random_class_order_list[(batch_id*5+2)%len(random_class_order_list)], random_class_order_list[(batch_id*5+3)%len(random_class_order_list)], random_class_order_list[(batch_id*5+4)%len(random_class_order_list)]]
                    elif cfg.dataset == "cub200":
                        list_for_one_batch = [random_class_order_list[batch_id*10%len(random_class_order_list)], random_class_order_list[(batch_id*10+1)%len(random_class_order_list)], random_class_order_list[(batch_id*10+2)%len(random_class_order_list)], random_class_order_list[(batch_id*10+3)%len(random_class_order_list)], random_class_order_list[(batch_id*10+4)%len(random_class_order_list)], random_class_order_list[(batch_id*10+5)%len(random_class_order_list)], random_class_order_list[(batch_id*10+6)%len(random_class_order_list)], random_class_order_list[(batch_id*10+7)%len(random_class_order_list)], random_class_order_list[(batch_id*10+8)%len(random_class_order_list)], random_class_order_list[(batch_id*10+9)%len(random_class_order_list)]]
                    else:
                        list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]
                    for i in list_for_one_batch:
                        sg_inputs.append(sample(model.class_mean_list[i], model.class_cov_list[i],int(10*cfg.beta), shrink=cfg.shrinkage))
                        sg_targets.append(torch.ones(int(10*cfg.beta), dtype=torch.long, device=device)*i)
                    sg_inputs = torch.cat(sg_inputs, dim=0)
                    sg_targets = torch.cat(sg_targets, dim=0)
                    targets = torch.cat([targets, sg_targets], dim=0)
                if model.hard_pairs is not None and model.hard_pairs.shape[0] > 0:
                    edge_sample = []
                    edge_p_target = []
                    edge_n_target = []
                    for hard_pair in model.hard_pairs:
                        edge_sample.append(sample(model.class_mean_list[hard_pair[0]], model.class_cov_list[hard_pair[0]],int(20*cfg.beta), shrink=cfg.shrinkage))
                        edge_p_target.append(torch.ones(int(20*cfg.beta), dtype=torch.long, device=device)*hard_pair[0])
                        edge_n_target.append(torch.ones(int(20*cfg.beta), dtype=torch.long, device=device)*hard_pair[1])
                    edge_sample = torch.cat(edge_sample, dim=0)
                    edge_p_target = torch.cat(edge_p_target, dim=0)
                    edge_n_target = torch.cat(edge_n_target, dim=0)
                if task_id > 0:
                    not_ini = True
                else:
                    not_ini = False
                outputs, _, __, edge_sample_features = model(inputs, memory_data=sg_inputs, not_ini=not_ini, edge_sample=edge_sample, prompt=False)



                if task_id>0:
                    if edge_sample is not None:
                        edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                        edge_target_features = model.class_name_features[edge_p_target].type(edge_sample_features.dtype)
                        edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                        edge_nearest_class_features = model.class_name_features[edge_n_target].type(edge_sample_features.dtype)
                        edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1, keepdim=True)
                        loss_hinge = torch.relu(- (edge_sample_features * edge_target_features.clone().detach()).sum(-1) + (edge_sample_features * edge_nearest_class_features.clone().detach()).sum(-1) + 0.1).mean()
                loss_c = torch.nn.functional.cross_entropy(outputs, targets.detach())
                if edge_sample is not None:
                    loss = loss_c + loss_hinge
                else:
                    loss = loss_c 

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tqdm_loader.set_description(f"Epoch {i_epoch + 1}/{cfg.epochs} | Loss: {loss.item():.4f} | Loss_c: {loss_c.item():.4f}| loss_hinge: {loss_hinge.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}")
            
            scheduler.step()
        sample_loader = DataLoader(train_dataset[task_id], batch_size=128, shuffle=False, num_workers=cfg.num_workers)
        sample_data = []
        sample_target = []
        sample_after_adapt_feature = []
        print('analyze')
        for input, target, task_ids in tqdm(sample_loader):
            input, target = input.to(device), target.to(device)
            with torch.no_grad():
                _, ori_ima_feat, after_adapt_feature = model(input, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(target)
            sample_after_adapt_feature.append(after_adapt_feature)
        sample_target = torch.cat(sample_target, dim=0)
        sample_data = torch.cat(sample_data, dim=0)
        sample_after_adapt_feature = torch.cat(sample_after_adapt_feature, dim=0)
        model.analyze_mean_cov(sample_data, sample_target)
        
        if offline_enabled:
            theta_state = {k: v.detach().cpu() for k, v in model.adapter.state_dict().items()}
            in_memory_thetas.append(theta_state)
            if offline_cfg.save_theta_each_task:
                theta_path = os.path.join(theta_dir, f"theta_{task_id + 1}.pt")
                torch.save(theta_state, theta_path)
                saved_theta_paths.append(theta_path)
        else:
            model.mix_matrix()

        if evaluate_per_task:
            model.eval()
            eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
            for inputs, targets, task_ids in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs, _, __, ___ = model(inputs)
                    torch.nn.functional.softmax(outputs, dim=-1)
                metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

            acc_list.append(100 * metric_logger.accuracy)
            with open(cfg.log_path, 'a+') as f:
                f.write(json.dumps({
                    'task': task_id,
                    'acc': round(100 * metric_logger.accuracy, 2),
                    'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                    'forgetting': round(100 * metric_logger.forgetting, 6),
                    'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                    'bwt': round(100 * metric_logger.backward_transfer, 2),
                    'fwt': round(100 * metric_logger.forward_transfer, 2),
                }) + '\n')
                metric_logger.end_task()

    if offline_enabled:
        num_tasks = len(eval_dataset)
        if len(task_difficulties) < num_tasks:
            # ensure alignment when adaptation not called? should not happen but guard.
            task_difficulties.extend([0.0] * (num_tasks - len(task_difficulties)))
        if offline_cfg.save_theta_each_task:
            thetas = load_thetas(theta_dir, num_tasks)
        else:
            thetas = in_memory_thetas

        if len(thetas) != num_tasks:
            raise ValueError(f"Expected {num_tasks} thetas for offline merge, found {len(thetas)}")
        if len(task_hard_pairs) < num_tasks:
            task_hard_pairs.extend([0] * (num_tasks - len(task_hard_pairs)))

        base_idx = select_base_index(offline_cfg.base_task, num_tasks)
        logging.info(f"OFFLINE_MERGE base_task={offline_cfg.base_task}, base_idx={base_idx + 1}, lambda_merge={offline_cfg.lambda_merge}, gamma={offline_cfg.gamma}")
        base_theta = thetas[base_idx]
        task_vectors = compute_task_vectors(thetas, base_idx)
        if offline_cfg.merge_mode == "magmax":
            merged_vector = merge_magmax(task_vectors)
        elif offline_cfg.merge_mode == "la_magmax":
            diffs = [d for j, d in enumerate(task_difficulties) if j != base_idx]
            merged_vector = merge_la_magmax(task_vectors, diffs, offline_cfg.gamma)
        else:
            raise ValueError(f"Unknown merge_mode {offline_cfg.merge_mode}")
        
        merged_vector = prune_topk(merged_vector, offline_cfg.topk_ratio)
        theta_merged = apply_merge(base_theta, merged_vector, offline_cfg.lambda_merge)
        merged_path = os.path.join(theta_dir, "merged_theta.pt")
        torch.save(theta_merged, merged_path)
        
        # load merged weights to adapter
        device_theta = {k: v.to(model.adapter.weight.device).type(model.adapter.weight.dtype) for k, v in theta_merged.items()}
        model.adapter.load_state_dict(device_theta)
        
        # save difficulties
        difficulty_path = os.path.join(theta_dir, "difficulty.json")
        with open(difficulty_path, "w") as f:
            json.dump({'task_difficulties': task_difficulties, 'task_hard_pairs': task_hard_pairs, 'base_idx': base_idx + 1}, f)
        
        # Final evaluation after merge (task-agnostic on all seen classes)
        model.eval()
        final_metric_logger = Logger(list_subsets=["test"])
        for tid in range(num_tasks):
            eval_loader = DataLoader(
                eval_dataset[tid],
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            for inputs, targets, task_ids in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs, _, __, ___ = model(inputs)
                final_metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
                final_acc = round(100 * final_metric_logger.accuracy, 2)

        log_payload = {
            'offline_merge': True,
            'base_task': offline_cfg.base_task,
            'base_idx': base_idx + 1,
            'merge_mode': offline_cfg.merge_mode,
            'lambda_merge': offline_cfg.lambda_merge,
            'gamma': offline_cfg.gamma,
            'merged_acc': final_acc,
            'acc_per_task': [round(100 * acc_t, 2) for acc_t in final_metric_logger.accuracy_per_task],
        }
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps(log_payload) + '\n')
        print(f"OFFLINE_MERGE(base={offline_cfg.base_task}, mode={offline_cfg.merge_mode}) merged accuracy: {final_acc}")
    
    elif acc_list:
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
