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
        logging.info(f"Train for task {task_id} has started.")
        train_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        
        model.adaptation(task_id, threshold=cfg.threshold)
        model.update_stat(known_classes=model.known_classes,total_classes=len(model.total_class_names),train_loader=train_loader,device=device) 
        model.train()

        epochs = cfg.epochs
        trainable_params = list(model.get_trainable_parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0)

        from continual_clip.losses import ClipLoss
        cliploss=ClipLoss()
        for i_epoch in range(epochs):
            loss = torch.tensor(0.0).to(device)
            loss_hinge = torch.tensor(0.0, device=device)
            tqdm_loader = tqdm(train_loader)
            if task_id>0:
                random_class_order_list = list(range(cfg.initial_increment+(task_id-1)*cfg.increment))
                random.shuffle(random_class_order_list)
            
            batch_id = -1
            for inputs, targets, task_ids in tqdm_loader:
                batch_id += 1
                
                inputs, targets = inputs.to(device), targets.to(device)
                sg_inputs = None
                edge_sample = None
                if task_id > 0:
                    sg_inputs = []
                    sg_targets = []
                    if cfg.dataset == "cifar100":
                        list_for_one_batch = [random_class_order_list[batch_id*4%len(random_class_order_list)], random_class_order_list[(batch_id*4+1)%len(random_class_order_list)], random_class_order_list[(batch_id*4+2)%len(random_class_order_list)], random_class_order_list[(batch_id*4+3)%len(random_class_order_list)]]
                    elif cfg.dataset == "imagenet_R":
                        list_for_one_batch = [random_class_order_list[batch_id*5%len(random_class_order_list)], random_class_order_list[(batch_id*5+1)%len(random_class_order_list)], random_class_order_list[(batch_id*5+2)%len(random_class_order_list)], random_class_order_list[(batch_id*5+3)%len(random_class_order_list)], random_class_order_list[(batch_id*5+4)%len(random_class_order_list)]]
                    elif cfg.dataset == "cub200":
                        list_for_one_batch = [random_class_order_list[batch_id*10%len(random_class_order_list)], random_class_order_list[(batch_id*10+1)%len(random_class_order_list)], random_class_order_list[(batch_id*10+2)%len(random_class_order_list)], random_class_order_list[(batch_id*10+3)%len(random_class_order_list)], random_class_order_list[(batch_id*10+4)%len(random_class_order_list)], random_class_order_list[(batch_id*10+5)%len(random_class_order_list)], random_class_order_list[(batch_id*10+6)%len(random_class_order_list)], random_class_order_list[(batch_id*10+7)%len(random_class_order_list)], random_class_order_list[(batch_id*10+8)%len(random_class_order_list)], random_class_order_list[(batch_id*10+9)%len(random_class_order_list)]]
                    else:
                        list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]
                    
                    if getattr(model, 'replay_sample_num', 0) > 0:
                        k = min(len(list_for_one_batch), model.replay_sample_num)
                        old_class = random.sample(list_for_one_batch, k)
                    else:
                        old_class = list_for_one_batch

                    for i in old_class:
                        if i >= len(model.prototype):
                            continue
                        proto = model.prototype[i].to(device).clone()
                        if model.sample_noise > 0:
                            proto = proto + torch.randn_like(proto) * model.sample_noise
                        sg_inputs.append(sample(model.class_mean_list[i], model.class_cov_list[i],int(10*cfg.beta), shrink=cfg.shrinkage))
                        sg_targets.append(torch.ones(int(10*cfg.beta), dtype=torch.long, device=device)*i)
                        sg_inputs.append(proto.unsqueeze(0))
                        sg_targets.append(torch.ones(1, dtype=torch.long, device=device) * i)
                    if sg_inputs:
                        sg_inputs = torch.cat(sg_inputs, dim=0)
                        sg_targets = torch.cat(sg_targets, dim=0)
                        targets = torch.cat([targets, sg_targets], dim=0)
                    else:
                        sg_inputs = None

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


                outputs, final_image_feas, _, edge_sample_features, _, _ = model(inputs, memory_data=sg_inputs, not_ini=not_ini, edge_sample=edge_sample)
                
                if task_id>0 and edge_sample is not None and edge_sample_features is not None:
                    edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                    edge_target_features = model.class_name_features[edge_p_target].type(edge_sample_features.dtype)
                    edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                    edge_nearest_class_features = model.class_name_features[edge_n_target].type(edge_sample_features.dtype)
                    edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1, keepdim=True)
                    loss_hinge = torch.relu(
                        -(edge_sample_features * edge_target_features.detach()).sum(-1)
                        + (edge_sample_features * edge_nearest_class_features.detach()).sum(-1)
                        + 0.1
                    ).mean()
                else:
                    loss_hinge = torch.tensor(0.0, device=device)

                if model.lambda_img > 0:
                    with torch.no_grad():
                        aug = torch.clamp(inputs + torch.randn_like(inputs) * 0.25, 0, 1)
                    aug_feas = model.encode_image(aug).float()
                    aug_feas = aug_feas / aug_feas.norm(dim=-1, keepdim=True)
                    sim_img = final_image_feas[:aug_feas.shape[0]] @ aug_feas.T
                    image_aug_loss = contrastive_loss(sim_img)
                else: 
                    image_aug_loss = torch.tensor(0.0, device=device)

                labels = [model.total_class_names[int(y)] for y in targets.tolist()]
                texts_clip=[model.prompt_template.format(inst) for inst in labels]
                with torch.no_grad():  
                    clip_tokens = model.tokenize(texts_clip).to(model.device)
                    clip_text_feas = model.encode_text(clip_tokens)
                clip_text_feas = model.apply_text_injection(clip_text_feas)
                clip_text_feas = clip_text_feas /clip_text_feas.norm(dim=-1, keepdim=True)

                if model.lambda_txt > 0:
                    repeat_ = 1 
                    anchor_text_loss_list = []
                    for _ in range(repeat_):
                        anchor_texts = model._get_text_anchor(labels)
                        anchor_emb = model.tokenize(anchor_texts).to(model.device)
                        with torch.no_grad():
                            anchor_text_features = model.encode_text(anchor_emb)
                        anchor_text_features = anchor_text_features.float() 
                        anchor_text_features = anchor_text_features / anchor_text_features.norm(dim=-1, keepdim=True)
                        anchor_text_loss_list.append(contrastive_loss(clip_text_feas @ anchor_text_features.T))
                    anchor_text_loss = sum(anchor_text_loss_list) / len(anchor_text_loss_list)
                else:
                    anchor_text_loss = 0
                
                clip_loss=cliploss(final_image_feas, clip_text_feas, model.logit_scale)

                loss = clip_loss + model.lambda_img * image_aug_loss + model.lambda_txt * anchor_text_loss + loss_hinge
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tqdm_loader.set_description(
                    f"Ep {i_epoch + 1}/{cfg.epochs} | L: {loss.item():.4f} | clip_loss: {clip_loss.item():.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.4f}"
                )
            
            scheduler.step()
            for inputs, targets, task_ids in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs, *_ = model(inputs)
                    torch.nn.functional.softmax(outputs, dim=-1)
                    metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="train")
        
        
        sample_loader = DataLoader(train_dataset[task_id], batch_size=128, shuffle=False, num_workers=cfg.num_workers)
        sample_data = []
        sample_target = []
        sample_after_adapt_feature = []
        print('analyze')
        for input, target, task_ids in tqdm(sample_loader):
            input, target = input.to(device), target.to(device)
            with torch.no_grad():
                _, ori_ima_feat, after_adapt_feature, *_ = model(input, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(target)
            sample_after_adapt_feature.append(after_adapt_feature)
        sample_target = torch.cat(sample_target, dim=0)
        sample_data = torch.cat(sample_data, dim=0)
        sample_after_adapt_feature = torch.cat(sample_after_adapt_feature, dim=0)
        model.analyze_mean_cov(sample_data, sample_target)
        model.eval()

        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, _, __, ___, _pre_image_feas, raw_image_feas = model(inputs)
                outputs = gda_output(
                    model=model,
                    cfg=cfg,
                    outputs=outputs,
                    raw_image_feas=raw_image_feas,
                )
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
        train_acc_str = f"{train_acc:.2f}" if train_acc is not None else "None"
        print(
            f"[Task {task_id}] "
            f"train_acc={train_acc_str} | "
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