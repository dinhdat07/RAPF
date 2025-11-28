import copy
import json
import math
import pdb
from itertools import chain
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pyparsing import Any
from omegaconf import DictConfig
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_class_ids_per_task, get_class_names

# --- Lớp MLP_Adapter (Không dùng) ---
class MLP_Adapter(nn.Module):
    def __init__(self, c_in, hidden):
        super(MLP_Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden),
        )

    def forward(self, x):
        x_ = self.fc(x)
        return x_
    
# --- LỚP ENGINE_ADAPTER ĐÃ ĐƯỢC CẢI TIẾN ---
class ENGINE_Adapter(nn.Module):
    def __init__(self, c_in, hidden, dropout=0.1, use_layernorm=True, learnable_scale=True):
        super(ENGINE_Adapter, self).__init__()

        # LayerNorm đầu vào (Input LN)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(c_in)

        # Down-projection
        self.down_proj = nn.Linear(c_in, hidden)
        
        # --- CẢI TIẾN 1: Thêm LayerNorm cho lớp ẩn ---
        self.ln_hidden = nn.LayerNorm(hidden)
        
        # --- CẢI TIẾN 2: Dùng GELU thay vì ReLU ---
        self.non_linear = nn.GELU()
        
        # Up-projection
        self.up_proj = nn.Linear(hidden, c_in)
        self.dropout = nn.Dropout(dropout)


        # Thang đo học được (Learnable Scale)
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.tensor(1.0))
            
        # Khởi tạo
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x

        # 1. LN đầu vào
        if self.use_layernorm:
            x = self.layernorm(x)

        # 2. Down-proj
        x = self.down_proj(x)
        
        # 3. LN lớp ẩn (Mới)
        x = self.ln_hidden(x)
        
        # 4. Non-linear (Mới)
        x = self.non_linear(x)
        x = self.dropout(x)
        
        # 5. Up-proj
        x = self.up_proj(x)

        # 6. Điều chỉnh mức ảnh hưởng
        x = x * self.scale

        # 7. Thêm residual
        return residual + x


# ... (Các hàm shrink_cov và sample giữ nguyên) ...
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

        # --- SỬ DỤNG LẠI ENGINE_Adapter (đã cải tiến) ---
        dropout_rate = float(getattr(cfg, 'dropout', 0.1))
        self.uni_image_adapter = None
        self.uni_text_adapter = None

        # ... (Phần còn lại của __init__ giữ nguyên) ...
        self.engine_cfg = getattr(cfg, 'engine', None)
        self.lambda_img = float(getattr(self.engine_cfg, 'lambda_img', 0.0)) if self.engine_cfg else 0.0
        self.lambda_txt = float(getattr(self.engine_cfg, 'lambda_txt', 0.0)) if self.engine_cfg else 0.0
        self.replay_alpha = float(getattr(self.engine_cfg, 'replay_alpha', 0.0)) if self.engine_cfg else 0.0
        self.replay_sample_num = int(getattr(self.engine_cfg, 'sample_num', 0)) if self.engine_cfg else 0
        self.image_injection = nn.ModuleList()
        self.text_injection = nn.ModuleList()


        self.prototype: List[torch.Tensor] = []
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias
        self.sample_noise = float(getattr(self.engine_cfg, 'sample_noise', 0.25)) if self.engine_cfg else 0.25


    # ... (Hàm update_stat giữ nguyên) ...
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

    
    # ... (Hàm get_trainable_parameters giữ nguyên) ...
    def get_trainable_parameters(self):
        params = []
        if self.image_injection and len(self.image_injection) > 0:
            params.append(self.image_injection[-1].parameters())
        if self.text_injection and len(self.text_injection) > 0:
            params.append(self.text_injection[-1].parameters())
        return chain.from_iterable(params)
    
    # ... (Hàm encode_text, encode_image, freeze giữ nguyên) ...
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

    # --- HÀM update_injection_units ĐÃ SỬA ---
    def update_injection_units(self, task_id):
        if task_id == 1:
            self.uni_image_adapter = copy.deepcopy(self.image_injection[0])
            self.uni_text_adapter = copy.deepcopy(self.text_injection[0])
            
            self.uni_image_adapter.to(self.device)
            self.uni_text_adapter.to(self.device)
            self.freeze(self.uni_image_adapter)
            self.freeze(self.uni_text_adapter)

        for inj in self.image_injection: 
            self.freeze(inj)
        for inj in self.text_injection:
            self.freeze(inj)

        dropout_rate = float(getattr(self.cfg, 'dropout', 0.1))

        # --- IMAGE ADAPTER ---
        if self.image_injection:
            new_image_adapter = copy.deepcopy(self.image_injection[-1])
            for p in new_image_adapter.parameters():
                p.requires_grad = True
            self.image_injection.append(new_image_adapter.to(self.device).to(dtype=self.dtype))
        else:
            self.image_injection.append(
                ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
            )

        # --- TEXT ADAPTER ---
        if self.text_injection:
            new_text_adapter = copy.deepcopy(self.text_injection[-1])
            for p in new_text_adapter.parameters():
                p.requires_grad = True
            self.text_injection.append(new_text_adapter.to(self.device).to(dtype=self.dtype))
        else:
            self.text_injection.append(
                ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
            )
        
    

    def _flatten_adapter_params(self, adapter):
        params = []
        for name, param in adapter.named_parameters():
            if isinstance(param, nn.Parameter):
                params.append(param.data.flatten())
        return torch.cat(params)

    def _unflatten_adapter_params(self, adapter, flat_vector: torch.Tensor):
        pointer = 0
        for name, param in adapter.named_parameters():
            if isinstance(param, nn.Parameter):
                num_elements = param.numel()
                param.data.copy_(
                    flat_vector[pointer:pointer + num_elements].view_as(param.data)
                )
                pointer += num_elements
        return adapter

    def mix_matrix(self, task_id, alpha_ema=0.9999):
        if task_id == 0: 
            return
        
        # IMAGE
        if len(self.image_injection) > 0:
            online_img = self.image_injection[-1]

            v_on  = self._flatten_adapter_params(online_img).detach()
            v_off = self._flatten_adapter_params(self.uni_image_adapter).detach()

            v_new_off = alpha_ema * v_off + (1 - alpha_ema) * v_on
            self._unflatten_adapter_params(self.uni_image_adapter, v_new_off)
            self.freeze(self.uni_image_adapter)

        # TEXT
        if len(self.text_injection) > 0:
            online_txt = self.text_injection[-1]

            v_on  = self._flatten_adapter_params(online_txt).detach()
            v_off = self._flatten_adapter_params(self.uni_text_adapter).detach()

            v_new_off = alpha_ema * v_off + (1 - alpha_ema) * v_on
            self._unflatten_adapter_params(self.uni_text_adapter, v_new_off)
            self.freeze(self.uni_text_adapter)


    # ... (Các hàm apply_image_injection, apply_text_injection giữ nguyên) ...
    def apply_image_injection(self, features: torch.Tensor, is_old=False) -> torch.Tensor:
        if len(self.image_injection) == 0:
            return features

        device = features.device 
        
        try:
            target_dtype = next(self.image_injection[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
            
        features = features.to(dtype=target_dtype)
        
        task_output = self.image_injection[-1](features)

        outputs = task_output
        
        return outputs

    def apply_text_injection(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.text_injection) == 0:
            return features
                    
        device = features.device
        
        try:
            target_dtype = next(self.text_injection[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
            
        features = features.to(dtype=target_dtype)
        
        task_output = self.text_injection[-1](features)


        outputs = task_output
                    
        return outputs
    

    # ... (Các hàm còn lại giữ nguyên) ...
    
    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        class_name_features = self.apply_text_injection(class_name_features)
        templates_per_class = getattr(self, "templates_per_class", 1)
        if templates_per_class > 1:
            num_classes = len(self.total_class_names)
            class_name_features = class_name_features.view(num_classes, templates_per_class, -1)
            class_name_features = class_name_features / class_name_features.norm(dim=-1, keepdim=True)
            class_name_features = class_name_features.mean(dim=1)
        class_name_features = class_name_features / class_name_features.norm(dim=-1, keepdim=True)
        return class_name_features.type(torch.float32)

    def adaptation(self, task_id, threshold=0):
        self.update_injection_units(task_id)
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

        if task_id > 0:
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

        # ----- Base CLIP image encoder -----
        with torch.no_grad():
            clip_features = self.encode_image(image).float()

        raw_image_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        original_image_features = clip_features.clone()

        # TRAIN
        if self.training:
            # --- IMAGE: online adapter ---
            image_features = self.image_injection[-1](clip_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # ----- MEMORY (replay) -----
            if memory_data is not None:
                memory_data = memory_data.type(self.dtype)
                mem_feat = self.image_injection[-1](memory_data)
                mem_feat = mem_feat / mem_feat.norm(dim=-1, keepdim=True)
                img_feas = torch.cat([image_features, mem_feat], dim=0)
            else:
                img_feas = image_features

            # ----- EDGE SAMPLE -----
            edge_num = 0
            edge_sample_features = None
            if edge_sample is not None:
                edge_sample = edge_sample.type(self.dtype)
                edge_num = edge_sample.shape[0]
                edge_feat = self.image_injection[-1](edge_sample)
                edge_feat = edge_feat / edge_feat.norm(dim=-1, keepdim=True)
                img_feas = torch.cat([img_feas, edge_feat], dim=0)

            final_image_feas = img_feas

            if edge_sample is not None:
                edge_sample_features = final_image_feas[-edge_num:]
                final_image_feas = final_image_feas[:-edge_num]

            # ---------- TEXT FEATURES cho logits (không dùng trong loss, chỉ để log/metric) ----------
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

            # Ở train, cho đơn giản: dùng text_features hiện tại (thường là offline hoặc fixed)
            final_text_feas = text_features / text_features.norm(dim=-1, keepdim=True)

            logits_per_image = self.logit_scale.exp() * final_image_feas @ final_text_feas.t().type(final_image_feas.dtype)
            probs = logits_per_image  # chỉ để metric_logger, không dùng trong clip_loss

            # ----- not_ini: cần old_memory_feature (nên dùng offline PET ở đây) -----
            if not_ini:
                with torch.no_grad():
                    old_memory_feature = self.uni_image_adapter(memory_data.type(self.dtype))
                    old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
                if edge_sample is not None:
                    return probs, final_image_feas, old_memory_feature, edge_sample_features, img_feas, raw_image_features
                return probs, final_image_feas, old_memory_feature, final_text_feas, img_feas, raw_image_features

            # ----- ori_ima_f trong train thường không dùng, nhưng giữ cho đủ -----
            if ori_ima_f:
                if memory_data is not None:
                    final_image_feas = final_image_feas[:-memory_data.shape[0]]
                return probs, original_image_features, final_image_feas, None, None, raw_image_features

            return probs, final_image_feas, None, edge_sample_features, img_feas, raw_image_features

        # =========================================================
        # 2) EVAL / INFERENCE MODE: LAE ENSEMBLE (online + offline)
        # =========================================================

        # ----- TEXT FEATURES (chung cho cả expert) -----
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

        final_text_feas = text_features / text_features.norm(dim=-1, keepdim=True)

        # ----- IMAGE BRANCH -----
        # Luôn có online adapter (image_injection[-1])
        img_on = self.image_injection[-1](clip_features)
        img_on = img_on / img_on.norm(dim=-1, keepdim=True)

        # Trường hợp: TASK 0 → CHƯA CÓ uni_image_adapter
        use_uni = hasattr(self, "uni_image_adapter") and (self.uni_image_adapter is not None)

        if not use_uni:
            # ========== TASK 0: ONLINE-ONLY ==========
            logits_on = self.logit_scale.exp() * img_on @ final_text_feas.t().type(img_on.dtype)
            probs = torch.nn.functional.softmax(logits_on, dim=-1)

            if ori_ima_f:
                # ori_ima_feat = CLIP gốc, after_adapt_feature = img_on (adapter online)
                return probs, original_image_features, img_on, None, None, raw_image_features

            return probs, img_on, None, None, img_on, raw_image_features

        # ========== TASK ≥ 1: LAE ENSEMBLE OFFLINE + ONLINE ==========
        img_off = self.uni_image_adapter(clip_features)
        img_off = img_off / img_off.norm(dim=-1, keepdim=True)

        logits_off = self.logit_scale.exp() * img_off @ final_text_feas.t().type(img_off.dtype)
        logits_on  = self.logit_scale.exp() * img_on  @ final_text_feas.t().type(img_on.dtype)

        probs_off = torch.nn.functional.softmax(logits_off, dim=-1)
        probs_on  = torch.nn.functional.softmax(logits_on,  dim=-1)

        probs_ens = torch.maximum(probs_off, probs_on)  # Eq.(14) LAE

        if ori_ima_f:
            # Dùng offline PET làm "after_adapt_feature" để phân tích prototype
            return probs_ens, original_image_features, img_off, None, None, raw_image_features

        # Eval bình thường
        return probs_ens, img_on, None, None, img_on, raw_image_features



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



# ... (Phần DomainIncrementalCLIP, TaskAgnosticCLIP, load_model giữ nguyên) ...

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
