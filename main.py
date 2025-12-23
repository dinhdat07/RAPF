import os

from continual_clip.utils import engine_rerank
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import json
import pdb
import random
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from itertools import chain

import torch
import statistics
from torch.utils.data import DataLoader
import torch.nn.functional as F
from continuum.metrics import Logger

from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import ClassIncrementalCLIP, load_model, sample
from continual_clip.losses import contrastive_loss, engine_contrastive_loss
from continual_clip.datasets import build_cl_scenarios
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
    if hasattr(cfg, 'engine') and cfg.engine is not None:
        if getattr(cfg, 'engine_lambda_img', None) is not None:
            cfg.engine.lambda_img = float(cfg.engine_lambda_img)
        if getattr(cfg, 'engine_lambda_txt', None) is not None:
            cfg.engine.lambda_txt = float(cfg.engine_lambda_txt)
        if getattr(cfg, 'engine_replay_alpha', None) is not None:
            cfg.engine.replay_alpha = float(cfg.engine_replay_alpha)
        if getattr(cfg, 'engine_sample_num', None) is not None:
            cfg.engine.sample_num = int(cfg.engine_sample_num)

    kd_temp = float(getattr(cfg, "kd_temp", 2.0))
    lambda_kd = float(getattr(cfg, "lambda_kd", 1.0))
    sigma_proto = float(getattr(cfg, "sigma_proto", 0.01))
    kd_topk_classes = int(getattr(cfg, "kd_topk_classes", 20))
    kd_replay_per_class = int(getattr(cfg, "kd_replay_per_class", 2))
    ce_mask_current_task = bool(getattr(cfg, "ce_mask_current_task", False))
    debug_router_batches = int(getattr(cfg, "debug_router_batches", 3))
    
    # model = load_model(cfg, device)
    model = ClassIncrementalCLIP(cfg, device)
    use_moe = bool(getattr(cfg, "use_moe_experts", False))
    if not use_moe:
        model.update_injection_units()
    teacher_model = None

    def sample_proto_batch(target_model, class_indices, per_class):
        feats = []
        for cls_idx in class_indices:
            base_proto = None
            if cls_idx < len(target_model.prototype):
                base_proto = target_model.prototype[cls_idx]
            elif cls_idx < len(target_model.class_mean_list):
                base_proto = target_model.class_mean_list[cls_idx]
            if base_proto is None:
                continue
            proto = base_proto.to(device).to(target_model.dtype)
            noise = sigma_proto * torch.randn(per_class, proto.shape[-1], device=device, dtype=proto.dtype)
            feats.append(proto.unsqueeze(0) + noise)
        return torch.cat(feats, dim=0) if feats else None

    eval_dataset, classes_names = build_cl_scenarios(cfg, is_train=False, base_transforms=model.transforms)
    train_dataset, _ = build_cl_scenarios(cfg, is_train=True, base_transforms=model.transforms)
    model.classes_names = classes_names
    print(model.classes_names)
    acc_list = []
    metric_logger = Logger(list_subsets=["train", "test"])

    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Train for task {task_id} has started.")
        train_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        
        model.adaptation(task_id, threshold=cfg.threshold)
        model.update_stat(known_classes=model.known_classes,total_classes=len(model.total_class_names),train_loader=train_loader,device=device) 
        model.train()
        if ce_mask_current_task:
            print(f"[INFO] CE masked: True | task {task_id} classes: {model.class_ids_per_task[task_id]}")

        trainable_params = list(model.get_trainable_parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.05)
        milestones = cfg.milestones
        epochs = cfg.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0)

        from continual_clip.losses import ClipLoss
        cliploss=ClipLoss()

        for i_epoch in range(epochs):
            loss = torch.tensor(0.0).to(device)
            loss_hinge = torch.tensor(0.0, device=device)
            entropy_coef = model.router_entropy_coef * max(0.0, 1.0 - i_epoch / max(1, epochs - 1)) if getattr(model, "use_moe_experts", False) else 0.0
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
                image_aug_loss = torch.tensor(0.0, device=device)
                kd_loss = torch.tensor(0.0, device=device)
                load_balance_loss = torch.tensor(0.0, device=device)
                entropy_loss = torch.tensor(0.0, device=device)
                if task_id > 0:
                    sg_inputs = []
                    sg_targets = []
                    if cfg.dataset == "cifar100" and cfg.increment == 5:
                        list_for_one_batch = [random_class_order_list[batch_id*4%len(random_class_order_list)], random_class_order_list[(batch_id*4+1)%len(random_class_order_list)], random_class_order_list[(batch_id*4+2)%len(random_class_order_list)], random_class_order_list[(batch_id*4+3)%len(random_class_order_list)]]
                    elif cfg.dataset == "imagenet_R":
                        list_for_one_batch = [random_class_order_list[batch_id*5%len(random_class_order_list)], random_class_order_list[(batch_id*5+1)%len(random_class_order_list)], random_class_order_list[(batch_id*5+2)%len(random_class_order_list)], random_class_order_list[(batch_id*5+3)%len(random_class_order_list)], random_class_order_list[(batch_id*5+4)%len(random_class_order_list)]]
                    elif cfg.dataset == "cub200":
                        list_for_one_batch = [random_class_order_list[batch_id*10%len(random_class_order_list)], random_class_order_list[(batch_id*10+1)%len(random_class_order_list)], random_class_order_list[(batch_id*10+2)%len(random_class_order_list)], random_class_order_list[(batch_id*10+3)%len(random_class_order_list)], random_class_order_list[(batch_id*10+4)%len(random_class_order_list)], random_class_order_list[(batch_id*10+5)%len(random_class_order_list)], random_class_order_list[(batch_id*10+6)%len(random_class_order_list)], random_class_order_list[(batch_id*10+7)%len(random_class_order_list)], random_class_order_list[(batch_id*10+8)%len(random_class_order_list)], random_class_order_list[(batch_id*10+9)%len(random_class_order_list)]]
                    else:
                        list_for_one_batch = random_class_order_list.copy()
                    
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


                outputs, final_image_feas, __, edge_sample_features, pre_image_feas, _raw_image_feas = model(inputs, memory_data=sg_inputs, not_ini=not_ini, edge_sample=edge_sample)
                router_info = getattr(model, "_latest_router_info", None)
                if model.use_moe_experts and router_info is not None:
                    load_balance_loss = router_info.get("load_balance", load_balance_loss)
                    entropy_loss = router_info.get("entropy", entropy_loss)
                    if model.debug_router and batch_id < debug_router_batches:
                        with torch.no_grad():
                            probs = router_info["probs"].detach()
                            topk_idx = router_info["topk_idx"].detach()
                            topk_w = router_info["topk_weights"].detach()
                            num_experts = int(router_info.get("num_experts", probs.shape[-1]))
                            top1 = topk_idx[:, 0]
                            top2 = topk_idx[:, 1] if topk_idx.shape[1] > 1 else None
                            top1_counts = torch.bincount(top1, minlength=num_experts)
                            top2_counts = torch.bincount(top2, minlength=num_experts) if top2 is not None else torch.zeros(num_experts, device=device, dtype=torch.long)
                            mean_pi = probs.mean(dim=0)
                            mean_w1 = topk_w[:, 0].mean()
                            mean_w2 = topk_w[:, 1].mean() if topk_w.shape[1] > 1 else torch.tensor(0.0, device=device)
                            curr_idx = len(model.image_experts)
                            pct_curr = (top1 == curr_idx).float().mean()
                            pct_uni = (top1 == 0).float().mean()
                            print(
                                f"[DEBUG][router] task{task_id} ep{i_epoch} b{batch_id} "
                                f"num_exp={num_experts} top1={top1_counts.cpu().tolist()} top2={top2_counts.cpu().tolist()} "
                                f"mean_pi={mean_pi.cpu().tolist()} w1={mean_w1.item():.3f} w2={mean_w2.item():.3f} "
                                f"pct_top1_curr={pct_curr.item():.3f} pct_top1_uni={pct_uni.item():.3f}"
                            )
                
                # RAPF: calculate loss hinge
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
                
                # ENGINE: calculate aug-image contrastive loss
                if model.lambda_img > 0:
                    with torch.no_grad():
                        aug = torch.clamp(inputs + torch.randn_like(inputs) * 0.25, 0, 1)
                    aug_feas = model.encode_image(aug).float()
                    aug_feas = aug_feas / aug_feas.norm(dim=-1, keepdim=True)
                    sim_img = final_image_feas[:aug_feas.shape[0]] @ aug_feas.T
                    image_aug_loss = contrastive_loss(sim_img)

                # ENGINE: get text features by targets and calculate text-des loss
                labels = [model.total_class_names[int(y)] for y in targets.tolist()]
                texts_clip=[model.prompt_template.format(inst) for inst in labels]
                with torch.no_grad():  
                    clip_tokens = model.tokenize(texts_clip).to(model.device)
                    clip_text_feas = model.encode_text(clip_tokens)
                clip_text_feas = model.apply_text_injection(clip_text_feas)
                clip_text_feas = clip_text_feas /clip_text_feas.norm(dim=-1, keepdim=True)
                
                clip_loss=cliploss(final_image_feas, clip_text_feas, model.logit_scale)
                ce_loss = torch.tensor(0.0, device=device)
                if ce_mask_current_task:
                    current_global_ids = list(model.class_ids_per_task[task_id])
                    seen_global_ids = list(chain.from_iterable(model.class_ids_per_task[:task_id + 1]))
                    global_to_local = {cid: idx for idx, cid in enumerate(seen_global_ids)}
                    current_local_ids = [global_to_local[cid] for cid in current_global_ids if cid in global_to_local]
                    class_mask = torch.zeros(outputs.shape[1], device=device, dtype=torch.bool)
                    if current_local_ids:
                        class_mask[torch.tensor(current_local_ids, device=device, dtype=torch.long)] = True
                    # Map dataset target (which may be position in class_order) to class_id
                    def target_to_class_id(tval: int) -> int:
                        t_int = int(tval)
                        return int(cfg.class_order[t_int]) if t_int < len(cfg.class_order) else t_int
                    # Torch.isin on CUDA can throw device assert for bad labels; compute on CPU then move back.
                    targets_class_cpu = torch.tensor([target_to_class_id(t) for t in targets.detach().cpu().tolist()], dtype=torch.long)
                    in_task_mask = torch.isin(targets_class_cpu, torch.tensor(current_global_ids, dtype=torch.long)).to(device)
                    if in_task_mask.any():
                        logits_masked_full = outputs[in_task_mask][:, class_mask].float()
                        # Map local ids to contiguous mask column indices
                        mask_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1).tolist()
                        local_to_mask = {int(loc): idx for idx, loc in enumerate(mask_indices)}
                        target_cpu = targets[in_task_mask].detach().cpu().tolist()
                        new_labels_list = []
                        keep_indices = []
                        for idx_t, t_val in enumerate(target_cpu):
                            class_id_val = target_to_class_id(t_val)
                            mapped = local_to_mask.get(global_to_local.get(class_id_val, -1), -1)
                            if mapped >= 0:
                                new_labels_list.append(mapped)
                                keep_indices.append(idx_t)
                        if len(new_labels_list) != len(target_cpu):
                            if batch_id == 0:
                                skipped = len(target_cpu) - len(new_labels_list)
                                print(f"[WARN][CE mask] found targets without mapping; skipping {skipped} samples")
                        if new_labels_list:
                            logits_masked = logits_masked_full[keep_indices]
                            new_labels = torch.tensor(new_labels_list, device=device, dtype=torch.long)
                            ce_loss = F.cross_entropy(logits_masked, new_labels)
                    if batch_id == 0:
                        print(f"[DEBUG][CE mask] enabled classes(global)={current_global_ids} local_ids={current_local_ids} samples_in_mask={int(in_task_mask.sum())}/{len(targets)}")

                if teacher_model is not None and lambda_kd > 0 and model.known_classes > 0:
                    old_classes = list(range(model.known_classes))
                    chosen_classes = random.sample(old_classes, min(len(old_classes), kd_topk_classes)) if len(old_classes) > 0 else []
                    replay_feats = sample_proto_batch(model, chosen_classes, kd_replay_per_class) if chosen_classes else None
                    if replay_feats is not None:
                        teacher_text_feats = teacher_model.class_name_features.to(device)
                        with torch.no_grad():
                            teacher_logits = teacher_model.forward_from_features(replay_feats.clone(), text_features=teacher_text_feats)
                        teacher_logits = teacher_logits.float()
                        student_logits = model.forward_from_features(replay_feats, text_features=teacher_text_feats)
                        student_logits = student_logits.float()
                        k_val = min(kd_topk_classes, teacher_logits.shape[-1])
                        if k_val < teacher_logits.shape[-1]:
                            teacher_topk, topk_idx = torch.topk(teacher_logits, k=k_val, dim=-1)
                            student_topk = student_logits.gather(1, topk_idx)
                            kd_loss = F.kl_div(
                                F.log_softmax(student_topk / kd_temp, dim=-1),
                                F.softmax(teacher_topk / kd_temp, dim=-1),
                                reduction="batchmean"
                            ) * (kd_temp ** 2)
                        else:
                            kd_loss = F.kl_div(
                                F.log_softmax(student_logits / kd_temp, dim=-1),
                                F.softmax(teacher_logits / kd_temp, dim=-1),
                                reduction="batchmean"
                            ) * (kd_temp ** 2)


                loss =  clip_loss +  model.lambda_img * image_aug_loss + loss_hinge + lambda_kd * kd_loss + model.lambda_lb * load_balance_loss + entropy_coef * entropy_loss + ce_loss
                loss.backward()
                if model.use_moe_experts and model.debug_router and batch_id < debug_router_batches and len(getattr(model, "image_experts", [])) > 0:
                    curr_params = list(model.image_experts[-1].parameters())
                    curr_grad = sum((p.grad.abs().sum() for p in curr_params if p.grad is not None))
                    curr_grad_cnt = sum((1 for p in curr_params if p.grad is not None))
                    prev_grad = torch.tensor(0.0, device=device)
                    prev_grad_cnt = 0
                    if len(model.image_experts) > 1:
                        prev_params = [p for exp in model.image_experts[:-1] for p in exp.parameters()]
                        prev_grad = sum((p.grad.abs().sum() for p in prev_params if p.grad is not None))
                        prev_grad_cnt = sum((1 for p in prev_params if p.grad is not None))
                    router_params = list(model.router.parameters())
                    router_grad = sum((p.grad.abs().sum() for p in router_params if p.grad is not None))
                    router_grad_cnt = sum((1 for p in router_params if p.grad is not None))
                    uni_params = list(model.uni_image_adapter.parameters())
                    uni_grad = sum((p.grad.abs().sum() for p in uni_params if p.grad is not None))
                    uni_grad_cnt = sum((1 for p in uni_params if p.grad is not None))
                    print(f"[DEBUG][router] grad current={float(curr_grad)} (cnt={curr_grad_cnt}) prev={float(prev_grad)} (cnt={prev_grad_cnt}) router={float(router_grad)} (cnt={router_grad_cnt}) uni={float(uni_grad)} (cnt={uni_grad_cnt})")
                optimizer.step()
                optimizer.zero_grad()
                tqdm_loader.set_description(
                    f"Ep {i_epoch + 1}/{cfg.epochs} | clip_loss: {clip_loss.item():.4f} | "
                    f"Lh: {loss_hinge.item():.4f} | kd: {kd_loss.item():.4f} | lb: {load_balance_loss.item():.4f} | ce_m:{ce_loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}"
                )
            
            scheduler.step()
            for inputs, targets, task_ids in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    outputs, *_ = model(inputs)
                    torch.nn.functional.softmax(outputs, dim=-1)
                    metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="train")
        
        
        sample_loader = DataLoader(train_dataset[task_id], batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
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
        model.mix_matrix()
        model.eval()


        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        total_labels = model.total_class_names
        print('total labels:', total_labels)
        templates = cfg.engine.templates
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = model.tokenize(texts).to(device)
                class_embeddings = model.encode_text(texts)
                class_embeddings = model.apply_text_injection(class_embeddings)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
            
        correct, total = 0, 0
        for i, (inputs, targets, task_ids) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, _, __, ___, _pre_image_feas, raw_image_feas = model(inputs)
                torch.nn.functional.softmax(outputs, dim=-1)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")


        # ----- Test logging -----
        test_acc = 100 * metric_logger.accuracy
        avg_acc = 100 * metric_logger.average_incremental_accuracy
        forgetting_val = 100 * metric_logger.forgetting
        acc_per_task = [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task]
        bwt = 100 * metric_logger.backward_transfer
        fwt = 100 * metric_logger.forward_transfer

        # ----- Train logging -----
        train_acc = 100 * metric_logger.online_accuracy if hasattr(metric_logger, "online_accuracy") else None
        
        # ----- Append vào danh sách để vẽ acc curve -----
        acc_list.append(test_acc)

        # ----- Ghi log -----
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
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

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
