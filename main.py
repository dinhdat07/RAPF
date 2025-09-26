
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pdb
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
    
    # model = load_model(cfg, device)
    model = ClassIncrementalCLIP(cfg, device)
    engine_alpha = 0.8
    engine_rerank_topk = 5

    eval_dataset, classes_names = build_cl_scenarios(cfg, is_train=False, transforms=model.transforms)
    train_dataset, _ = build_cl_scenarios(cfg, is_train=True, transforms=model.transforms)
    model.classes_names = classes_names
    model._get_text_des(dataname='cifar224') 
    acc_list = []
    metric_logger = Logger(list_subsets=["test"])

    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Train for task {task_id} has started.")
        train_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        
        model.adaptation(task_id, threshold=cfg.threshold)
        # model.update_stat(self._known_classes,self._total_classes, self.train_loader, self._device) ~ ENGINE
        model.train()

        trainable_params = list(model.get_trainable_parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.0)
        milestones = cfg.milestones
        epochs = cfg.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

        for i_epoch in range(epochs):
            loss = torch.tensor(0.0).to(device)
            loss_hinge = torch.tensor(0.0).to(device)
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
                    if cfg.dataset == "cifar100" and cfg.increment == 5:
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
                        # use cov,mean and shrinkage instead of sample noise like ENGINE
                        proto = sample(model.class_mean_list[i], model.class_cov_list[i], int(10*cfg.beta), shrink=cfg.shrinkage)
                        sg_inputs.append(proto)
                        # int(10*cfg.beta) instead of int(1) like ENGINE
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


                outputs, _, __, edge_sample_features, pre_image_feas = model(inputs, memory_data=sg_inputs, not_ini=not_ini, edge_sample=edge_sample)
                
                # RAPF: calculate loss hinge
                if task_id>0:
                    if edge_sample is not None:
                        edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                        edge_target_features = model.class_name_features[edge_p_target].type(edge_sample_features.dtype)
                        edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                        edge_nearest_class_features = model.class_name_features[edge_n_target].type(edge_sample_features.dtype)
                        edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1, keepdim=True)
                        loss_hinge = torch.relu(- (edge_sample_features * edge_target_features.clone().detach()).sum(-1) + (edge_sample_features * edge_nearest_class_features.clone().detach()).sum(-1) + 0.1).mean()
                    else: 
                        loss_hinge = 0
                
                # ENGINE: calculate aug-image contrastive loss
                if model.lambda_img > 0:
                    with torch.no_grad():
                        aug = torch.clamp(inputs + torch.randn_like(inputs) * 0.25, 0, 1)
                    aug_feas = model.encode_image(aug).float()
                    aug_feas = aug_feas / aug_feas.norm(dim=-1, keepdim=True)
                    sim_img = pre_image_feas[:aug_feas.shape[0]] @ aug_feas.T
                    image_aug_loss = contrastive_loss(sim_img)

                # ENGINE: get text features by targets and calculate text-des loss
                labels = [classes_names[int(y)] for y in targets.tolist()]
                texts_clip=[model.prompt_template.format(inst) for inst in labels]
                with torch.no_grad():  
                    clip_tokens = model.tokenize(texts_clip).to(model.device)
                    clip_text_feas = model.encode_text(clip_tokens)
                clip_text_feas = model.apply_injections(model.text_injections, clip_text_feas)
                clip_text_feas = clip_text_feas /clip_text_feas.norm(dim=-1, keepdim=True)

                if model.lambda_txt > 0:
                    repeat_ = 1 
                    ref_text_loss_list = []
                    for _ in range(repeat_):
                        ref_texts = model._get_batch_des(model.new_des_dict, labels)
                        ref_emb = model.tokenize(ref_texts).to(model.device)
                        with torch.no_grad():
                            ref_text_features = model.encode_text(ref_emb)
                        ref_text_features = ref_text_features.float() 
                        ref_text_features = ref_text_features / ref_text_features.norm(dim=-1, keepdim=True)
                        ref_text_loss_list.append(contrastive_loss(clip_text_feas @ ref_text_features.T))
                    ref_text_loss = sum(ref_text_loss_list) / len(ref_text_loss_list)
                else:
                    ref_text_loss = 0

                # RAPF: calculate contrastive loss
                loss_ce = F.cross_entropy(outputs, targets.detach())
                
                loss = loss_ce + model.lambda_img * image_aug_loss + model.lambda_txt * ref_text_loss + loss_hinge
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tqdm_loader.set_description(
                    f"Ep {i_epoch + 1}/{cfg.epochs} | L: {loss.item():.4f} | Lc: {loss_ce.item():.4f} | "
                    f"Li: {image_aug_loss.item():.4f} | Lt: {ref_text_loss.item():.4f} | Lh: {loss_hinge.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}"
                )
            
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
        model.mix_matrix()
        model.eval()

        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, image_features, __, ___, pre_image_feas = model(inputs)
                
                # NEED TO IMPLEMENT THE ENGINE RERANKING FUNCTION!!!
                # outputs = engine_rerank(image_features, outputs, model.total_class_names, text_prompt_bank, alpha=engine_alpha, topk=engine_rerank_topk)
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