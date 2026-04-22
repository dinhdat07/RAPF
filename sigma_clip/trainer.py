from __future__ import annotations

import json
import logging
import os
import random
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from continuum.metrics import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_class_incremental_scenarios
from .losses import ClipLoss, contrastive_loss
from .model import SigmaClassIncrementalCLIP, sample
from .utils import get_class_order, sigma_logit_fusion


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def freeze_clip_backbone(model: SigmaClassIncrementalCLIP) -> None:
    for name, param in model.named_parameters():
        if "visual" in name or "transformer" in name or "token_embedding" in name:
            param.requires_grad = False


def _replay_classes_for_batch(cfg, task_id: int, batch_id: int, random_class_order_list: List[int]) -> List[int]:
    if cfg.dataset == "cifar100":
        batch_span = 4
    elif cfg.dataset == "imagenet_R":
        batch_span = 5
    else:
        batch_span = 2

    return [
        random_class_order_list[(batch_id * batch_span + offset) % len(random_class_order_list)]
        for offset in range(batch_span)
    ]


def _build_replay_batch(
    cfg,
    model: SigmaClassIncrementalCLIP,
    device: torch.device,
    task_id: int,
    batch_id: int,
    random_class_order_list: Optional[List[int]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if task_id == 0:
        return None, None

    selected_classes = _replay_classes_for_batch(cfg, task_id, batch_id, random_class_order_list)
    if getattr(model, "replay_sample_num", 0) > 0:
        sample_count = min(len(selected_classes), model.replay_sample_num)
        selected_classes = random.sample(selected_classes, sample_count)

    replay_inputs = []
    replay_targets = []
    for class_idx in selected_classes:
        if class_idx >= len(model.prototype):
            continue

        proto = model.prototype[class_idx].to(device).clone()
        if model.sample_noise > 0:
            proto = proto + torch.randn_like(proto) * model.sample_noise

        replay_inputs.append(
            sample(
                model.class_mean_list[class_idx],
                model.class_cov_list[class_idx],
                int(10 * cfg.beta),
                shrink=cfg.shrinkage,
            )
        )
        replay_targets.append(torch.ones(int(10 * cfg.beta), dtype=torch.long, device=device) * class_idx)

        replay_inputs.append(proto.unsqueeze(0))
        replay_targets.append(torch.ones(1, dtype=torch.long, device=device) * class_idx)

    if not replay_inputs:
        return None, None

    return torch.cat(replay_inputs, dim=0), torch.cat(replay_targets, dim=0)


def _build_edge_samples(cfg, model: SigmaClassIncrementalCLIP, device: torch.device):
    if model.hard_pairs is None or model.hard_pairs.shape[0] == 0:
        return None, None, None

    edge_samples = []
    edge_positive_targets = []
    edge_negative_targets = []

    for hard_pair in model.hard_pairs:
        edge_samples.append(
            sample(
                model.class_mean_list[hard_pair[0]],
                model.class_cov_list[hard_pair[0]],
                int(20 * cfg.beta),
                shrink=cfg.shrinkage,
            )
        )
        edge_positive_targets.append(
            torch.ones(int(20 * cfg.beta), dtype=torch.long, device=device) * hard_pair[0]
        )
        edge_negative_targets.append(
            torch.ones(int(20 * cfg.beta), dtype=torch.long, device=device) * hard_pair[1]
        )

    return (
        torch.cat(edge_samples, dim=0),
        torch.cat(edge_positive_targets, dim=0),
        torch.cat(edge_negative_targets, dim=0),
    )


def _compute_hinge_loss(
    model: SigmaClassIncrementalCLIP,
    task_id: int,
    edge_sample,
    edge_sample_features,
    edge_positive_targets,
    edge_negative_targets,
    device: torch.device,
):
    if task_id == 0 or edge_sample is None or edge_sample_features is None:
        return torch.tensor(0.0, device=device)

    edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)

    edge_target_features = model.class_name_features[edge_positive_targets].type(edge_sample_features.dtype)
    edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)

    edge_nearest_class_features = model.class_name_features[edge_negative_targets].type(edge_sample_features.dtype)
    edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(
        dim=-1, keepdim=True
    )

    return torch.relu(
        -(edge_sample_features * edge_target_features.detach()).sum(-1)
        + (edge_sample_features * edge_nearest_class_features.detach()).sum(-1)
        + 0.1
    ).mean()


def _compute_image_aug_loss(model: SigmaClassIncrementalCLIP, inputs, final_image_features, device):
    if model.lambda_img <= 0:
        return torch.tensor(0.0, device=device)

    with torch.no_grad():
        augmented = torch.clamp(inputs + torch.randn_like(inputs) * 0.25, 0, 1)
    augmented_features = model.encode_image(augmented).float()
    augmented_features = augmented_features / augmented_features.norm(dim=-1, keepdim=True)
    sim_img = final_image_features[: augmented_features.shape[0]] @ augmented_features.T
    return contrastive_loss(sim_img)


def _compute_text_anchor_loss(model: SigmaClassIncrementalCLIP, labels, clip_text_features, device):
    if model.lambda_txt <= 0:
        return torch.tensor(0.0, device=device)

    repeat_count = 1
    anchor_loss_values = []
    for _ in range(repeat_count):
        anchor_texts = model._get_text_anchor(labels)
        anchor_tokens = model.tokenize(anchor_texts).to(model.device)
        with torch.no_grad():
            anchor_text_features = model.encode_text(anchor_tokens)
        anchor_text_features = anchor_text_features.float()
        anchor_text_features = anchor_text_features / anchor_text_features.norm(dim=-1, keepdim=True)
        anchor_loss_values.append(contrastive_loss(clip_text_features @ anchor_text_features.T))

    return sum(anchor_loss_values) / len(anchor_loss_values)


def _train_single_epoch(
    cfg,
    model: SigmaClassIncrementalCLIP,
    task_id: int,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    cliploss: ClipLoss,
    device: torch.device,
):
    tqdm_loader = tqdm(train_loader)

    random_class_order_list = None
    if task_id > 0:
        random_class_order_list = list(range(cfg.initial_increment + (task_id - 1) * cfg.increment))
        random.shuffle(random_class_order_list)

    for batch_id, (inputs, targets, _) in enumerate(tqdm_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        replay_inputs, replay_targets = _build_replay_batch(
            cfg=cfg,
            model=model,
            device=device,
            task_id=task_id,
            batch_id=batch_id,
            random_class_order_list=random_class_order_list,
        )
        if replay_targets is not None:
            targets = torch.cat([targets, replay_targets], dim=0)

        edge_sample, edge_positive_targets, edge_negative_targets = _build_edge_samples(
            cfg=cfg,
            model=model,
            device=device,
        )

        outputs, final_image_features, _, edge_sample_features, _, _ = model(
            inputs,
            memory_data=replay_inputs,
            not_ini=task_id > 0,
            edge_sample=edge_sample,
        )

        loss_hinge = _compute_hinge_loss(
            model=model,
            task_id=task_id,
            edge_sample=edge_sample,
            edge_sample_features=edge_sample_features,
            edge_positive_targets=edge_positive_targets,
            edge_negative_targets=edge_negative_targets,
            device=device,
        )

        image_aug_loss = _compute_image_aug_loss(model, inputs, final_image_features, device)

        labels = [model.total_class_names[int(label)] for label in targets.tolist()]
        text_prompts = [model.prompt_template.format(inst) for inst in labels]
        with torch.no_grad():
            clip_tokens = model.tokenize(text_prompts).to(model.device)
            clip_text_features = model.encode_text(clip_tokens)
        clip_text_features = model.apply_text_injection(clip_text_features)
        clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)

        anchor_text_loss = _compute_text_anchor_loss(
            model=model,
            labels=labels,
            clip_text_features=clip_text_features,
            device=device,
        )

        clip_loss = cliploss(final_image_features, clip_text_features, model.logit_scale)
        loss = clip_loss + model.lambda_img * image_aug_loss + model.lambda_txt * anchor_text_loss + loss_hinge

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tqdm_loader.set_description(
            f"Ep {scheduler.last_epoch + 2}/{cfg.epochs} | L: {loss.item():.4f} | "
            f"clip_loss: {clip_loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}"
        )


def _collect_train_subset_metrics(model: SigmaClassIncrementalCLIP, train_loader: DataLoader, metric_logger, device):
    for inputs, targets, task_ids in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs, *_ = model(inputs)
            torch.nn.functional.softmax(outputs, dim=-1)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="train")


def _analyze_task_statistics(model: SigmaClassIncrementalCLIP, task_dataset, cfg, device):
    sample_loader = DataLoader(task_dataset, batch_size=128, shuffle=False, num_workers=cfg.num_workers)
    sample_data = []
    sample_target = []

    print("analyze")
    for inputs, targets, _ in tqdm(sample_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            _, original_image_features, _, _, _, _ = model(inputs, ori_ima_f=True)
        sample_data.append(original_image_features)
        sample_target.append(targets)

    sample_target = torch.cat(sample_target, dim=0)
    sample_data = torch.cat(sample_data, dim=0)
    model.analyze_mean_cov(sample_data, sample_target)


def _evaluate_seen_tasks(model: SigmaClassIncrementalCLIP, eval_dataset, task_id, cfg, metric_logger, device):
    eval_loader = DataLoader(eval_dataset[: task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    for inputs, targets, task_ids in eval_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs, _, __, ___, _pre_image_features, raw_image_features = model(inputs)
            outputs = sigma_logit_fusion(
                model=model,
                cfg=cfg,
                outputs=outputs,
                raw_image_features=raw_image_features,
            )
            torch.nn.functional.softmax(outputs, dim=-1)
        metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")


def _write_task_metrics(cfg, metric_logger, task_id: int, acc_list: List[float]):
    test_acc = 100 * metric_logger.accuracy
    avg_acc = 100 * metric_logger.average_incremental_accuracy
    forgetting_val = 100 * metric_logger.forgetting
    acc_per_task = [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task]
    bwt = 100 * metric_logger.backward_transfer
    fwt = 100 * metric_logger.forward_transfer

    train_acc = 100 * metric_logger.online_accuracy if hasattr(metric_logger, "online_accuracy") else None
    acc_list.append(test_acc)
    train_acc_str = f"{train_acc:.2f}" if train_acc is not None else "None"

    print(
        f"[Task {task_id}] "
        f"train_acc={train_acc_str} | "
        f"test_acc={test_acc:.2f} | avg_acc={avg_acc:.2f} | "
        f"forgetting={forgetting_val:.6f}"
    )

    with open(cfg.log_path, "a+", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "task": task_id,
                    "train_acc": round(train_acc, 2) if train_acc is not None else None,
                    "test_acc": round(test_acc, 2),
                    "avg_acc": round(avg_acc, 2),
                    "forgetting": round(forgetting_val, 6),
                    "acc_per_task": acc_per_task,
                    "bwt": round(bwt, 2),
                    "fwt": round(fwt, 2),
                }
            )
            + "\n"
        )
    metric_logger.end_task()


def run_class_incremental(cfg, device):
    cfg.class_order = get_class_order(os.path.join(cfg.workdir, cfg.class_order))

    model = SigmaClassIncrementalCLIP(cfg, device)
    freeze_clip_backbone(model)
    model.update_injection_units()

    eval_dataset, classes_names = build_class_incremental_scenarios(
        cfg,
        is_train=False,
        base_transforms=model.transforms,
    )
    train_dataset, _ = build_class_incremental_scenarios(
        cfg,
        is_train=True,
        base_transforms=model.transforms,
    )
    model.classes_names = classes_names

    acc_list = []
    metric_logger = Logger(list_subsets=["train", "test"])

    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Train for task {task_id} has started.")

        train_loader = DataLoader(
            train_dataset[task_id],
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )

        model.adaptation(task_id, threshold=cfg.threshold)
        model.update_stat(
            known_classes=model.known_classes,
            total_classes=len(model.total_class_names),
            train_loader=train_loader,
            device=device,
        )
        model.train()

        optimizer = torch.optim.AdamW(
            list(model.get_trainable_parameters()),
            lr=cfg.lr,
            weight_decay=0.005,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=0,
        )

        cliploss = ClipLoss()
        for _ in range(cfg.epochs):
            _train_single_epoch(
                cfg=cfg,
                model=model,
                task_id=task_id,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                cliploss=cliploss,
                device=device,
            )
            scheduler.step()
            _collect_train_subset_metrics(model, train_loader, metric_logger, device)

        _analyze_task_statistics(model, train_dataset[task_id], cfg, device)
        model.eval()
        _evaluate_seen_tasks(model, eval_dataset, task_id, cfg, metric_logger, device)
        _write_task_metrics(cfg, metric_logger, task_id, acc_list)

    with open(cfg.log_path, "a+", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "last": round(acc_list[-1], 2),
                    "avg": round(statistics.mean(acc_list), 2),
                }
            )
            + "\n"
        )
