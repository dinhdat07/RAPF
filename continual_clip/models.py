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

from .utils import get_class_ids_per_task, get_class_names, normalize_key
    
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
        self.uni_image_adapter = ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        self.uni_text_adapter = ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        self.freeze(self.uni_image_adapter)
        self.freeze(self.uni_text_adapter)

        self.image_fusion_alpha = nn.Parameter(torch.ones(1)) 
        self.image_fusion_beta = nn.Parameter(torch.ones(1)) 
        
        self.text_fusion_alpha = nn.Parameter(torch.ones(1))
        self.text_fusion_beta = nn.Parameter(torch.ones(1))

        # ... (Phần còn lại của __init__ giữ nguyên) ...
        self.engine_cfg = getattr(cfg, 'engine', None)
        self.lambda_img = float(getattr(self.engine_cfg, 'lambda_img', 0.0)) if self.engine_cfg else 0.0
        self.lambda_txt = float(getattr(self.engine_cfg, 'lambda_txt', 0.0)) if self.engine_cfg else 0.0
        self.replay_alpha = float(getattr(self.engine_cfg, 'replay_alpha', 0.0)) if self.engine_cfg else 0.0
        self.replay_sample_num = int(getattr(self.engine_cfg, 'sample_num', 0)) if self.engine_cfg else 0
        self.image_injection = nn.ModuleList()
        self.prev_image_injection = None
        self.text_injection = nn.ModuleList()
        self.prev_text_injection = None
        self.new_des_dict = {}
        self.prototype: List[torch.Tensor] = []
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias
        self.sample_noise = float(getattr(self.engine_cfg, 'sample_noise', 0.25)) if self.engine_cfg else 0.25
    
    def visual_forward_(self, image: torch.Tensor):
        v = self.visual 
        x = image.to(self.clip_type)

        # patch -> seq
        x = v.conv1(x)                                  # [B, C, H/patch, W/patch]
        x = x.reshape(x.shape[0], x.shape[1], -1)       # [B, C, N]
        x = x.permute(0, 2, 1)                          # [B, N, C]

        # prepend class token + add pos embed
        class_tok = v.class_embedding.to(x.dtype)       # [C]
        class_tok = class_tok.expand(x.shape[0], 1, -1) # [B, 1, C]
        x = torch.cat([class_tok, x], dim=1)            # [B, 1+N, C]
        x = x + v.positional_embedding.to(x.dtype)      # [B, 1+N, C]

        # LN + Transformer (openai/clip ViT expect [B, seq, C] -> they internally permute)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)                          # [seq, B, C]
        x = v.transformer(x)
        x = x.permute(1, 0, 2)                          # [B, seq, C]

        # LN post, class token (pre-projection)
        x = v.ln_post(x)                                # [B, seq, C]
        pooled = x[:, 0, :]                             # [B, C]  (pre-proj)
        
        return pooled



    # ... (Hàm update_stat giữ nguyên) ...
    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("Updating stat...")
        with torch.no_grad():
            vecs = []
            labels = []
            for images, targets, _ in train_loader:
                images, targets = images.to(device), targets.to(device)
                image_features = self.visual_forward_(images).float()

                # normalize vector
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
            # params.append([self.image_fusion_alpha, self.image_fusion_beta])
        if self.text_injection and len(self.text_injection) > 0:
            params.append(self.text_injection[-1].parameters())
            # params.append([self.text_fusion_alpha, self.text_fusion_beta])
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
                ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
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
                ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
            )
        
    # ... (Các hàm _flatten, _unflatten giữ nguyên) ...
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

    def mix_matrix(self):
        if len(self.image_injection) < 1:
            return 

        v_uni_img = self.fuse_sign_max(self.image_injection)
        self._unflatten_adapter_params(self.uni_image_adapter, v_uni_img)
        self.freeze(self.uni_image_adapter)

        v_uni_txt = self.fuse_sign_max(self.text_injection)
        self._unflatten_adapter_params(self.uni_text_adapter, v_uni_txt)
        self.freeze(self.uni_text_adapter)

    def fuse_sign_max(self, adapters):
        vecs = torch.stack([self._flatten_adapter_params(a) for a in adapters])  # [N, P]
        s_uni = torch.sign(vecs.sum(dim=0))                                      # [P]
        same_sign = torch.sign(vecs) == s_uni.unsqueeze(0)                       # [N, P]
        masked = torch.where(same_sign, vecs.abs(), torch.zeros_like(vecs))
        m_uni = masked.max(dim=0).values                                         # [P]
        v_uni = s_uni * m_uni
        return v_uni

    def _orthogonal_group_loss(self, adapters, norm: str = "l1", eps: float = 1e-8):
        if len(adapters) <= 1:
            return torch.zeros((), device=self.device)

        W_new = adapters[-1].up_proj.weight  # [d, r]
        loss = torch.zeros((), device=W_new.device, dtype=W_new.dtype)

        for prev in adapters[:-1]:
            W_prev = prev.up_proj.weight  
            A = W_new @ W_prev.t() 

            if norm == "l1":
                loss = loss + A.abs().sum()
            elif norm == "fro2":
                loss = loss + (A * A).sum()
            elif norm == "cos":
                Wn = torch.nn.functional.normalize(W_new, dim=1, eps=eps)
                Wp = torch.nn.functional.normalize(W_prev, dim=1, eps=eps)
                C = Wn @ Wp.t()
                loss = loss + C.abs().mean()
            else:
                raise ValueError("norm must be one of {'l1','fro2','cos'}")

        num_prev = len(adapters) - 1
        if norm in ("l1", "fro2"):
            d = W_new.shape[0]
            loss = loss / (num_prev * d * d)
        else: 
            loss = loss / num_prev

        return loss

    def orthogonal_loss(self):
        return self._orthogonal_group_loss(self.image_injection, norm="l1") + \
            self._orthogonal_group_loss(self.text_injection,  norm="l1")

    def apply_image_injection(self, features: torch.Tensor, mode="infer") -> torch.Tensor:
        if len(self.image_injection) == 0:
            return features

        try:
            target_dtype = next(self.image_injection[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
        features = features.to(dtype=target_dtype)

        if mode == "train":
            return self.image_injection[-1](features)

        task_output = self._select_adapter_by_entropy(self.image_injection, self.uni_image_adapter, features)
        uni_output = self.uni_image_adapter(features)
        return uni_output + task_output


    def apply_text_injection(self, features: torch.Tensor, mode="infer") -> torch.Tensor:
        if len(self.text_injection) == 0:
            return features

        try:
            target_dtype = next(self.text_injection[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
        features = features.to(dtype=target_dtype)

        if mode == "train":
            return self.text_injection[-1](features)
    
        task_output = self._select_adapter_by_entropy(self.text_injection, self.uni_text_adapter, features)
        uni_output  = self.uni_text_adapter(features)
        return uni_output + task_output
    
    def _select_adapter_by_entropy(self, adapters, uni_adapter, features):
        if len(adapters) == 1:
            return adapters[0](features)
        
        uni_out = uni_adapter(features)

        entropies = []
        outputs = []
        for a in adapters:
            out = a(features)
            outputs.append(out)
            logits = self._logits_from_features(out + uni_out)
            entropies.append(self._entropy_from_logits(logits))

        best_idx = int(torch.stack(entropies).argmin().item())
        return outputs[best_idx]
    
    def _logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        W = self.class_name_features.to(device=feats.device, dtype=feats.dtype)  # [K, C]
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        W = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        scale = self.logit_scale.exp().to(dtype=feats.dtype, device=feats.device)
        return scale * (feats @ W.t())
    
    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1).clamp_min(1e-12)
        return -(p * p.log()).sum(dim=-1).mean()


    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)  # [K or K*T, C]
        templates_per_class = getattr(self, "templates_per_class", 1)
        if templates_per_class > 1:
            num_classes = len(self.total_class_names)
            class_name_features = class_name_features.view(num_classes, templates_per_class, -1)
            class_name_features = class_name_features / class_name_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            class_name_features = class_name_features.mean(dim=1)
        else:
            class_name_features = class_name_features / class_name_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return class_name_features.to(dtype=torch.float32)
    
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

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, mode="train"):
        
        image = image.type(self.dtype)

        with torch.no_grad():
            clip_features = self.encode_image(image).float()
        raw_image_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        original_image_features = clip_features.clone()
        image_features = clip_features

        image_features = self.apply_image_injection(image_features, mode=mode)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            sg_image_features = self.apply_image_injection(memory_data, mode=mode)
            sg_image_features = sg_image_features / sg_image_features.norm(dim=-1, keepdim=True)
            img_feas = torch.cat([image_features, sg_image_features], dim=0)
        else:
            img_feas = image_features

        edge_num = 0
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            edge_sample = self.apply_image_injection(edge_sample, mode=mode)
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

        final_text_feas = self.apply_text_injection(text_features, mode=mode)
        final_text_feas = final_text_feas / final_text_feas.norm(dim=-1, keepdim=True)

        #---------- logits ---------
        logits_per_image = self.logit_scale.exp() * final_image_feas @ final_text_feas.t().type(final_image_feas.dtype)
        probs = logits_per_image


        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.apply_image_injection(memory_data, mode=mode)
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

    def _flatten_values(value: Any) -> List[str]:
        out: List[str] = []
        def _walk(x: Any):
            if isinstance(x, str):
                s = x.strip()
                if s: out.append(s)
            elif isinstance(x, list):
                for y in x: _walk(y)
            elif isinstance(x, dict):
                for y in x.values(): _walk(y)
        _walk(value)
        seen = set(); res = []
        for s in out:
            if s not in seen:
                seen.add(s); res.append(s)
        return res
      
    def get_text_des(self,dataname='cifar224'):
        root_path = Path(__file__).resolve().parent.parent
        des_path = root_path / "chat" / f"{dataname}_des.json"
        with open(des_path, 'r') as f:
            des_dict = json.load(f)
        self.des_dict = des_dict

        new_des_dict = {}
        for key, value in des_dict.items():
            new_key_value = []
            for _, v in value.items():
                new_key_value.extend(v)
            new_des_dict[key] = new_key_value

        self.new_des_dict = new_des_dict
        return new_des_dict
      
    def get_batch_des(self, des_file: Dict[str, List[str]], classnames: Iterable[str]) -> List[str]:
        out: List[str] = []
        for cname in classnames:
            cname = normalize_key(cname)
            descs = des_file.get(cname, [])
            if descs: 
                out.append(f"{cname} with {random.choice(descs).casefold()}")
            else:
                print(f"[WARNING] No description found for class '{cname}'. Using default prompt.")
                out.append(f"a photo of {cname}")
        return out
    
    def rerank(self, outputs, image_features_raw, device, topk=5):
        des_dict = self.des_dict
        class_to_label = self.classes_names
        with torch.no_grad():
            top5_predict = outputs.topk(topk, 1, True, True)[1]
            top5_predict_labels = [[class_to_label[int(label)] for label in pred] for pred in top5_predict]

            logi = 0
            for _ in range(3):
                texts = []
                for batch in range(image_features_raw.shape[0]):
                    for main_label in top5_predict_labels[batch]:
                        for second_label in top5_predict_labels[batch]:
                            if main_label == second_label:
                                continue
                            main = normalize_key(main_label)
                            second = normalize_key(second_label)
                            texts.append(main_label + ' with ' + random.choice(des_dict[main][second]).lower())
                
                texts = self.tokenize(texts).to(device)
                texts = self.encode_text(texts)
                texts = texts.reshape(image_features_raw.shape[0], topk, topk-1, -1)
                texts = torch.mean(texts, dim=2)
                texts = texts / texts.norm(dim=-1, keepdim=True)
                logits = [image_features_raw[i] @ texts[i].T for i in range(image_features_raw.shape[0])]
                logits = torch.stack(logits)
                logi += logits

            logits = logi/3
            logits = logits.to(outputs.dtype)  
            new_logits = torch.zeros_like(outputs)
            for i in range(image_features_raw.shape[0]):
                new_logits[i, top5_predict[i]] = logits[i]
            return new_logits

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
