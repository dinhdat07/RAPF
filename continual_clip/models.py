import copy
import json
import math
import pdb
from itertools import chain
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from pyparsing import Any
from .gate import DynamicGate
from .adapter import ENGINE_Adapter
from omegaconf import DictConfig
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_class_ids_per_task, get_class_names

class ClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None

        model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)

        for param in model.parameters():
            param.requires_grad = False
            
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.total_class_names = []
        self.current_class_names = []
        self.known_classes = 0
        self.text_tokens = None
        self.dtype = torch.float16 if cfg.fp16 else torch.float32
        self.clip_type = model.dtype
        self.tokenize = clip.tokenize
        self.img_feat_dim = int(getattr(self.visual, "output_dim", 512))
        text_proj = getattr(model, "text_projection", None)
        if isinstance(text_proj, torch.Tensor):
            self.txt_feat_dim = int(text_proj.shape[-1])
        else:
            self.txt_feat_dim = self.img_feat_dim

        dropout_rate = float(getattr(cfg, 'dropout', 0.1))
        self.uni_image_adapter = ENGINE_Adapter(self.img_feat_dim, self.img_feat_dim // 2, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        self.uni_text_adapter = ENGINE_Adapter(self.txt_feat_dim, self.txt_feat_dim // 2, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        self.freeze(self.uni_image_adapter)
        self.freeze(self.uni_text_adapter)
        
        self.text_fusion_alpha = nn.Parameter(torch.ones(1))
        self.text_fusion_beta = nn.Parameter(torch.ones(1))

        self.use_dynamic_gate = True
        self.gate_img = DynamicGate().to(self.device)
        self.gate_txt = DynamicGate().to(self.device)
        self.register_buffer("uni_img_centroid", torch.zeros(self.img_feat_dim, device=self.device, dtype=torch.float32))
        self.register_buffer("uni_img_centroid_cnt", torch.tensor(0.0, device=self.device))
        self.centroid_decay = 0.99
        self.fisher_img_ema = None
        self.fisher_txt_ema = None
        self.rho_fisher = 0.95
        self.rho_param = 0.90

        self.engine_cfg = getattr(cfg, 'engine', None)
        self.lambda_img = float(getattr(self.engine_cfg, 'lambda_img', 0.0)) if self.engine_cfg else 0.0
        self.lambda_txt = float(getattr(self.engine_cfg, 'lambda_txt', 0.0)) if self.engine_cfg else 0.0
        self.replay_alpha = float(getattr(self.engine_cfg, 'replay_alpha', 0.0)) if self.engine_cfg else 0.0
        self.replay_sample_num = int(getattr(self.engine_cfg, 'sample_num', 0)) if self.engine_cfg else 0
        self.image_injection = nn.ModuleList()
        self.prev_image_injection = None
        self.text_injection = nn.ModuleList()
        self.prev_text_injection = None
        self.prototype: List[torch.Tensor] = []
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias
        self.sample_noise = float(getattr(self.engine_cfg, 'sample_noise', 0.25)) if self.engine_cfg else 0.25


    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("Updating stat...")
        with torch.no_grad():
            vecs = []
            labels = []
            for images, targets, _ in train_loader:
                images, targets = images.to(device), targets.to(device)
                image_features = self.encode_image(images).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vecs.append(image_features)
                labels.append(targets)
            if len(vecs) == 0:
                print("WARNING: train_loader has no data → skip update_stat")
                return
            vecs = torch.cat(vecs)
            labels = torch.cat(labels)
            print(f"[DEBUG update_stat] known_classes={known_classes}, total_classes={total_classes}")
            print("Labels in batch:", labels.unique().tolist())
            mu_list = []
            for class_idx in range(known_classes, total_classes):
                cls_vecs = vecs[labels == class_idx]
                if cls_vecs.numel() > 0:
                    mean_vec = cls_vecs.mean(dim=0, keepdim=True)
                    mu_list.append(mean_vec)
            if len(mu_list) == 0:
                print("WARNING: No new class in this batch → skip update_stat")
                return
            mu = torch.cat(mu_list, dim=0)
            center_list = []
            for j, i in enumerate(range(known_classes, total_classes)):
                cls_vecs = vecs[labels == i]
                if cls_vecs.numel() > 0:
                    center_list.append(cls_vecs - mu[j])
            if len(center_list) == 0:
                print("WARNING: center_vecs is empty → skip update_stat")
                return
            center_vecs = torch.cat(center_list, dim=0)
            cov = center_vecs.T @ center_vecs / (center_vecs.shape[0] - 1)
            dim = center_vecs.shape[1]
            reg = cov.trace() * torch.eye(dim, device=device)
            cov_inv = dim * torch.linalg.pinv((center_vecs.shape[0] - 1) * cov + reg)
            if not hasattr(self, 'mu'):
                self.mu = mu
                self.cov_inv = cov_inv
            else:
                self.cov_inv = (
                    (known_classes / total_classes) * self.cov_inv +
                    (total_classes - known_classes) / total_classes * cov_inv +
                    (
                        (known_classes / total_classes) *
                        (total_classes - known_classes) / (total_classes ** 2)
                    ) * (
                        (self.mu.mean(dim=0) - mu.mean(dim=0)).unsqueeze(1) @
                        (self.mu.mean(dim=0) - mu.mean(dim=0)).unsqueeze(0)
                    )
                )
                self.mu = torch.cat([self.mu, mu])
            ps = torch.ones(self.mu.shape[0], device=device) / self.mu.shape[0]
            self.W = torch.einsum('nd,dc->cn', self.mu, self.cov_inv)
            self.b = ps.log() - 0.5 * torch.einsum('nd,dc,nc->n', self.mu, self.cov_inv, self.mu)

    def get_trainable_parameters(self):
        params = []
        if self.image_injection and len(self.image_injection) > 0:
            params.append(self.image_injection[-1].parameters())
            params.append(self.gate_img.parameters())
        if self.text_injection and len(self.text_injection) > 0:
            params.append(self.text_injection[-1].parameters())
            params.append(self.gate_txt.parameters())
        return chain.from_iterable(params)
    
    def encode_text(self, text, prompt=False):
        x = self.token_embedding(text).type(self.clip_type)
        x = x + self.positional_embedding.type(self.clip_type)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_image(self, image):
        image = image.to(self.clip_type)
        return self.visual(image)

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def update_injection_units(self, noise_std: float = 0.01):
        for inj in self.image_injection: 
            self.freeze(inj)
        for inj in self.text_injection:
            self.freeze(inj)

        dropout_rate = float(getattr(self.cfg, 'dropout', 0.1))

        # --- IMAGE ADAPTER ---
        if self.image_injection:
            # 1. Deepcopy
            new_image_adapter = copy.deepcopy(self.image_injection[-1])
            for param in new_image_adapter.parameters():
                # Thêm nhiễu
                param.data += noise_std * torch.randn_like(param.data)
                param.requires_grad = True 
            
            # --- SỬA LỖI 2: Reset scale ---
            if hasattr(new_image_adapter, 'scale'):
                with torch.no_grad():
                    new_image_adapter.scale.fill_(1.0) # Reset về 1.0
            
            self.image_injection.append(new_image_adapter.to(self.device).to(dtype=self.dtype))
        else:
            # --- SỬ DỤNG LẠI ENGINE_Adapter ---
            self.image_injection.append(
                ENGINE_Adapter(self.img_feat_dim, self.img_feat_dim // 2, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
            )

        # --- TEXT ADAPTER ---
        if self.text_injection:
            # 1. Deepcopy
            new_text_adapter = copy.deepcopy(self.text_injection[-1])
            for param in new_text_adapter.parameters():
                param.data += noise_std * torch.randn_like(param.data)
                param.requires_grad = True
            
            # --- SỬA LỖI 2: Reset scale ---
            if hasattr(new_text_adapter, 'scale'):
                with torch.no_grad():
                    new_text_adapter.scale.fill_(1.0) # Reset về 1.0

            self.text_injection.append(new_text_adapter.to(self.device).to(dtype=self.dtype))
        else:
            # --- SỬ DỤNG LẠI ENGINE_Adapter ---
            self.text_injection.append(
                ENGINE_Adapter(self.txt_feat_dim, self.txt_feat_dim // 2, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
            )

    # ------ FOR APPLY DYNAMIC GATE ------
    def _get_text_features_uni_only(self, target_dtype):
        if hasattr(self, "class_name_features") and self.class_name_features is not None:
            txt_raw = self.class_name_features
        else:
            txt_raw = self.encode_text(self.text_tokens)
            if getattr(self, "templates_per_class", 1) > 1:
                num_classes = len(self.total_class_names)
                txt_raw = txt_raw.view(num_classes, self.templates_per_class, -1)
                txt_raw = (txt_raw / txt_raw.norm(dim=-1, keepdim=True)).mean(dim=1)
        param = next(self.uni_text_adapter.parameters(), None)
        device = param.device if param is not None else txt_raw.device
        txt_uni = self.uni_text_adapter(txt_raw.to(device=device, dtype=target_dtype))
        txt_uni = txt_uni / txt_uni.norm(dim=-1, keepdim=True)
        return txt_uni

    @torch.no_grad()
    def _update_uni_img_centroid(self, uni_out: torch.Tensor):
        if uni_out.numel() == 0:
            return
        u = F.normalize(uni_out.float(), dim=-1, eps=1e-6).mean(dim=0)
        if self.uni_img_centroid_cnt.item() == 0:
            self.uni_img_centroid.copy_(u)
        else:
            self.uni_img_centroid.mul_(self.centroid_decay).add_(u * (1 - self.centroid_decay))
        self.uni_img_centroid_cnt += 1

    def apply_image_injection(self, features: torch.Tensor, is_old: bool=False) -> torch.Tensor:
        if len(self.image_injection) == 0:
            return features

        params = list(self.image_injection[0].parameters())
        if params:
            target_dtype = params[0].dtype
            target_device = params[0].device
        else:
            target_dtype = features.dtype
            target_device = features.device
        x = features.to(device=target_device, dtype=target_dtype)

        uni_out = self.uni_image_adapter(x)
        task_out = self.image_injection[-1](x)
        task_out = task_out.to(dtype=uni_out.dtype)

        if self.training:
            with torch.no_grad():
                self._update_uni_img_centroid(uni_out)

        u = F.normalize(uni_out.reshape(uni_out.size(0), -1), dim=-1, eps=1e-6)
        c = F.normalize(self.uni_img_centroid, dim=0, eps=1e-6).unsqueeze(0)
        s_uni = (u * c).sum(dim=-1, keepdim=True)

        with torch.no_grad():
            txt_uni = self._get_text_features_uni_only(target_dtype).to(device=uni_out.device)
            img_tmp = 0.5 * (uni_out + task_out)
            img_tmp = img_tmp / img_tmp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = self.logit_scale.exp().to(device=img_tmp.device, dtype=img_tmp.dtype) * img_tmp @ txt_uni.t().to(dtype=img_tmp.dtype)
            prob = F.softmax(logits, dim=1)
            H = -(prob * (prob.clamp_min(1e-9).log())).sum(dim=1, keepdim=True)

        alpha, beta = self.gate_img(s_uni, H)
        if alpha.dim() < uni_out.dim():
            alpha = alpha.view(*([alpha.size(0)] + [1] * (uni_out.dim() - 1)))
            beta = beta.view(*([beta.size(0)] + [1] * (uni_out.dim() - 1)))
        return alpha * uni_out + beta * task_out

    def apply_text_injection(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.text_injection) == 0:
            return features

        params = list(self.text_injection[0].parameters())
        if params:
            target_dtype = params[0].dtype
            target_device = params[0].device
        else:
            target_dtype = features.dtype
            target_device = features.device
        x = features.to(device=target_device, dtype=target_dtype)

        uni_out = self.uni_text_adapter(x)
        task_out = self.text_injection[-1](x)
        task_out = task_out.to(dtype=uni_out.dtype)

        u = F.normalize(uni_out.reshape(uni_out.size(0), -1), dim=-1, eps=1e-6)
        t = F.normalize(task_out.reshape(task_out.size(0), -1), dim=-1, eps=1e-6)
        sim_uni = (u * t).sum(dim=-1, keepdim=True)
        H_proxy = torch.ones_like(sim_uni) - sim_uni

        alpha, beta = self.gate_txt(sim_uni, H_proxy)
        if alpha.dim() < uni_out.dim():
            alpha = alpha.view(*([alpha.size(0)] + [1] * (uni_out.dim() - 1)))
            beta = beta.view(*([beta.size(0)] + [1] * (uni_out.dim() - 1)))
        return alpha * uni_out + beta * task_out

    # ------ FOR FISHER EMA UPDATE ------
    def update_uni_with_fisher_ema(self, dataloader, max_batches: int = 2):
        if len(self.image_injection) > 0:
            last_img = self.image_injection[-1]
            f_new = self._fisher_diag_for_adapter(last_img, dataloader, max_batches=max_batches, branch="image")
            if f_new.numel() > 0:
                f_new = f_new.to(dtype=torch.float32)
                if self.fisher_img_ema is None:
                    self.fisher_img_ema = f_new.clone()
                else:
                    self.fisher_img_ema = self.rho_fisher * self.fisher_img_ema + (1 - self.rho_fisher) * f_new
                v_last = self._flatten_params_of(last_img)
                v_uni_old = self._flatten_params_of(self.uni_image_adapter)
                fisher = self.fisher_img_ema.to(device=v_last.device, dtype=torch.float32)
                v_last_f = v_last.to(dtype=torch.float32)
                v_uni_old_f = v_uni_old.to(dtype=torch.float32)
                w = fisher / (fisher.sum() + 1e-8)
                v_uni_new = self.rho_param * v_uni_old_f + (1 - self.rho_param) * (w * v_last_f)
                v_uni_new = v_uni_new.to(dtype=v_uni_old.dtype)
                self._assign_flatten_params(self.uni_image_adapter, v_uni_new)
                for p in self.uni_image_adapter.parameters():
                    p.requires_grad_(False)

        if len(self.text_injection) > 0:
            last_txt = self.text_injection[-1]
            f_new_t = self._fisher_diag_for_adapter(last_txt, dataloader, max_batches=max_batches, branch="text")
            if f_new_t.numel() > 0:
                f_new_t = f_new_t.to(dtype=torch.float32)
                if self.fisher_txt_ema is None:
                    self.fisher_txt_ema = f_new_t.clone()
                else:
                    self.fisher_txt_ema = self.rho_fisher * self.fisher_txt_ema + (1 - self.rho_fisher) * f_new_t
                v_last_t = self._flatten_params_of(last_txt)
                v_uni_old_t = self._flatten_params_of(self.uni_text_adapter)
                fisher_t = self.fisher_txt_ema.to(device=v_last_t.device, dtype=torch.float32)
                v_last_tf = v_last_t.to(dtype=torch.float32)
                v_uni_old_tf = v_uni_old_t.to(dtype=torch.float32)
                w_t = fisher_t / (fisher_t.sum() + 1e-8)
                v_uni_new_t = self.rho_param * v_uni_old_tf + (1 - self.rho_param) * (w_t * v_last_tf)
                v_uni_new_t = v_uni_new_t.to(dtype=v_uni_old_t.dtype)
                self._assign_flatten_params(self.uni_text_adapter, v_uni_new_t)
                for p in self.uni_text_adapter.parameters():
                    p.requires_grad_(False)

    def _flatten_params_of(self, module: nn.Module) -> torch.Tensor:
        params = [p for p in module.parameters() if p is not None]
        if not params:
            return torch.tensor([], device=torch.device("cpu"))
        flat_chunks = [p.detach().reshape(-1) for p in params]
        return torch.cat(flat_chunks)

    def _assign_flatten_params(self, module: nn.Module, flat: torch.Tensor) -> None:
        pointer = 0
        for p in module.parameters():
            if p is None:
                continue
            numel = p.numel()
            p.data.copy_(flat[pointer:pointer + numel].view_as(p.data))
            pointer += numel

    def _fisher_diag_for_adapter(self, adapter: nn.Module, dataloader, max_batches: int = 2, branch: str = "image") -> torch.Tensor:
        params = list(adapter.parameters())
        if not params:
            return torch.tensor([], device=self.device)
        prev_reqs = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad.zero_()
        flat_fisher = torch.zeros_like(self._flatten_params_of(adapter))
        batches = 0
        was_training = adapter.training
        adapter.train()

        if branch == "image":
            dataloader_iter = dataloader if dataloader is not None else []
        else:
            dataloader_iter = range(max_batches)

        for batch in dataloader_iter:
            if batches >= max_batches:
                break
            if branch == "image":
                if not isinstance(batch, (list, tuple)):
                    continue
                if len(batch) < 2:
                    continue
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                features = self.encode_image(images).float()
                target_dtype = params[0].dtype
                features = features.to(dtype=target_dtype)
                adapter.zero_grad(set_to_none=True)
                uni_out = self.uni_image_adapter(features)
                task_out = adapter(features)
                img_tmp = 0.5 * (uni_out + task_out)
                img_tmp = img_tmp / img_tmp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                with torch.no_grad():
                    txt_uni = self._get_text_features_uni_only(target_dtype)
                logits = self.logit_scale.exp().to(dtype=img_tmp.dtype) * img_tmp @ txt_uni.t().to(dtype=img_tmp.dtype)
                loss = F.cross_entropy(logits, labels)
            else:
                adapter.zero_grad(set_to_none=True)
                if hasattr(self, "class_name_features") and self.class_name_features is not None:
                    txt_raw = self.class_name_features
                else:
                    if getattr(self, "text_tokens", None) is None:
                        break
                    txt_raw = self.encode_text(self.text_tokens)
                    if getattr(self, "templates_per_class", 1) > 1:
                        num_classes = len(self.total_class_names)
                        txt_raw = txt_raw.view(num_classes, self.templates_per_class, -1)
                        txt_raw = (txt_raw / txt_raw.norm(dim=-1, keepdim=True)).mean(dim=1)
                target_dtype = params[0].dtype
                txt_raw = txt_raw.to(device=self.device, dtype=target_dtype)
                uni_out = self.uni_text_adapter(txt_raw).detach()
                task_out = adapter(txt_raw)
                loss = F.mse_loss(task_out, uni_out)
            loss.backward()
            grads = []
            for p in params:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).reshape(-1))
                else:
                    grads.append(p.grad.detach().reshape(-1))
            flat_grad = torch.cat(grads)
            flat_fisher += flat_grad.pow(2)
            batches += 1

        if batches > 0:
            flat_fisher /= batches
        adapter.zero_grad(set_to_none=True)
        for p, req in zip(params, prev_reqs):
            p.requires_grad_(req)
        adapter.train(was_training)
        return flat_fisher
    

    # ------ REST OF RAPF ------
    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        templates_per_class = getattr(self, "templates_per_class", 1)
        if templates_per_class > 1:
            num_classes = len(self.total_class_names)
            class_name_features = class_name_features.view(
                num_classes, templates_per_class, -1
            )
            class_name_features = class_name_features / class_name_features.norm(
                dim=-1, keepdim=True
            )
            class_name_features = class_name_features.mean(dim=1)
        return class_name_features.type(torch.float32)

    def adaptation(self, task_id, threshold=0):
        self.known_classes = len(self.total_class_names)
        self.total_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        if hasattr(self.cfg.engine, "templates") and self.cfg.engine.templates:
            prompt_templates = self.cfg.engine.templates
        else:
            prompt_templates = [self.prompt_template]
        self.templates_per_class = len(prompt_templates)
        all_prompts = []
        for cname in self.total_class_names:
            all_prompts.extend([tmpl.format(cname) for tmpl in prompt_templates])
        self.text_tokens = self.tokenize(all_prompts).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        if task_id>0:
            self.update_injection_units()
            dist_list = []
            for _, class_name_feature in enumerate(self.class_name_features[:-len(self.class_ids_per_task[task_id])]):
                diff = torch.cdist(self.class_name_features[-len(self.class_ids_per_task[task_id]):].type(torch.float32), class_name_feature.unsqueeze(0).type(torch.float32)).squeeze()
                dist_list.append(diff)
            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = torch.nonzero(mask)
            self.hard_new_class = torch.unique(indices[:,1]) + self.cfg.initial_increment+(task_id-1) * self.cfg.increment
            self.hard_pairs = indices
            self.hard_pairs[:,1] = self.hard_pairs[:,1]+self.cfg.initial_increment+(task_id-1) * self.cfg.increment

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None):
        image = image.type(self.dtype)

        with torch.no_grad():
            clip_features = self.encode_image(image).float()
        raw_image_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        original_image_features = clip_features.clone()
        image_features = clip_features

        image_features = self.apply_image_injection(image_features)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            sg_image_features = self.apply_image_injection(memory_data)
            sg_image_features = sg_image_features / sg_image_features.norm(dim=-1, keepdim=True)
            img_feas = torch.cat([image_features, sg_image_features], dim=0)
        else:
            img_feas = image_features

        edge_num = 0
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            edge_sample = self.apply_image_injection(edge_sample)
            edge_sample = edge_sample / edge_sample.norm(dim=-1, keepdim=True)
            img_feas = torch.cat([img_feas, edge_sample], dim=0)

        final_image_feas = img_feas

        edge_sample_features = None
        if edge_sample is not None:
            edge_sample_features = final_image_feas[-edge_num:]
            final_image_feas = final_image_feas[:-edge_num]

        #---------- text features ---------
        if hasattr(self, "class_name_features") and self.class_name_features is not None:
            text_features = self.class_name_features
        else:
            with torch.no_grad():
                text_features = self.encode_text(self.text_tokens)
            templates_per_class = getattr(self, "templates_per_class", 1)
            if templates_per_class > 1:
                num_classes = len(self.total_class_names)
                text_features = text_features.view(num_classes, templates_per_class, -1)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=1)

        final_text_feas = self.apply_text_injection(text_features)
        final_text_feas = final_text_feas / final_text_feas.norm(dim=-1, keepdim=True)

        #---------- logits ---------
        logits_per_image = self.logit_scale.exp() * final_image_feas @ final_text_feas.t().type(final_image_feas.dtype)
        probs = logits_per_image


        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.apply_image_injection(memory_data, is_old=True)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, final_image_feas, old_memory_feature, edge_sample_features, img_feas, raw_image_features
            return probs, final_image_feas, old_memory_feature, final_text_feas, img_feas, raw_image_features
        if ori_ima_f:
            if memory_data is not None:
                final_image_feas = final_image_feas[:-memory_data.shape[0]]
            return probs, original_image_features, final_image_feas, None, None, raw_image_features
        
        return probs, final_image_feas, None, edge_sample_features, img_feas, raw_image_features

    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            proto = mean.detach().to(torch.float32)
            class_idx = int(l.item()) if hasattr(l, "item") else int(l)
            if len(self.prototype) <= class_idx:
                self.prototype.append(proto)
            else:
                self.prototype[class_idx] = proto
            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append((max_distance.mean() - max_distance.min(), max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    
        
class DomainIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False) -> None:
        super().__init__()
        self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.text_tokens = None
        self.prompt_template = cfg.prompt_template
        self.device = device

    def forward(self, image):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs
    
    def tokenize(self, class_names):
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in class_names]
        ).to(self.device)

class TaskAgnosticCLIP(nn.Module):
    pass

def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncrementalCLIP(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncrementalCLIP(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnosticCLIP(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
