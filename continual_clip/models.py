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
from torch import Tensor

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


class TopKRouter(nn.Module):
    """Lightweight router with expandable output head for MoE gating."""

    def __init__(self, dim: int, hidden: int, num_experts: int, topk: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, num_experts)
        self.topk = topk

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)

    def expand_output(self, new_num_experts: int):
        """Safely expand the final projection when adding experts."""
        if new_num_experts <= self.fc2.out_features:
            return
        old_out = self.fc2
        new_out = nn.Linear(
            old_out.in_features,
            new_num_experts,
            device=old_out.weight.device,
            dtype=old_out.weight.dtype,
        )
        with torch.no_grad():
            new_out.weight[: old_out.out_features] = old_out.weight
            new_out.bias[: old_out.out_features] = old_out.bias
        self.fc2 = new_out


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
        self.use_moe_experts = bool(getattr(cfg, "use_moe_experts", False))
        self.router_topk = int(getattr(cfg, "router_topk", 2))
        self.router_hidden = int(getattr(cfg, "router_hidden", 256))
        self.lambda_lb = float(getattr(cfg, "lambda_lb", 0.01))
        self.router_entropy_coef = float(getattr(cfg, "router_entropy_coef", 0.0))
        self.freeze_universal = bool(getattr(cfg, "freeze_universal_expert", True))
        self.debug_router = bool(getattr(cfg, "debug_router", False))
        self.debug_router_batches = int(getattr(cfg, "debug_router_batches", 3))
        self.moe_text = bool(getattr(cfg, "moe_text", False))

        # --- SỬ DỤNG LẠI ENGINE_Adapter (đã cải tiến) ---
        dropout_rate = float(getattr(cfg, 'dropout', 0.1))
        self.uni_image_adapter = ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        self.uni_text_adapter = ENGINE_Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype)
        if not self.use_moe_experts or self.freeze_universal:
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
        # Old fusion path state (kept for backward-compat if MoE is disabled)
        self.image_injection = nn.ModuleList()
        self.prev_image_injection = None
        self.text_injection = nn.ModuleList()
        self.prev_text_injection = None
        # MoE expert containers
        self.image_experts = nn.ModuleList()
        self.text_experts = nn.ModuleList()
        self.router = TopKRouter(512, self.router_hidden, num_experts=1, topk=self.router_topk).to(self.device) if self.use_moe_experts else None
        self._latest_router_info: Optional[Dict[str, Tensor]] = None
        self.new_des_dict = {}
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
        if self.use_moe_experts:
            params = []
            if self.router is not None:
                params.append(self.router.parameters())
            if len(self.image_experts) > 0:
                params.append([p for p in self.image_experts[-1].parameters() if p.requires_grad])
            if len(self.text_experts) > 0:
                params.append([p for p in self.text_experts[-1].parameters() if p.requires_grad])
            if not self.freeze_universal:
                params.append([p for p in self.uni_image_adapter.parameters() if p.requires_grad])
                params.append([p for p in self.uni_text_adapter.parameters() if p.requires_grad])
            return chain.from_iterable(params)
        params = []
        if self.image_injection and len(self.image_injection) > 0:
            params.append(self.image_injection[-1].parameters())
            params.append([self.image_fusion_alpha, self.image_fusion_beta])
        if self.text_injection and len(self.text_injection) > 0:
            params.append(self.text_injection[-1].parameters())
            params.append([self.text_fusion_alpha, self.text_fusion_beta])
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
        if self.use_moe_experts:
            # MoE branch manages experts via add_new_task_expert.
            return
        
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

    # --- MoE expert management ---
    def _init_expert_from(self, src: nn.Module, noise_std: float) -> nn.Module:
        new_module = copy.deepcopy(src)
        for param in new_module.parameters():
            if noise_std > 0:
                param.data += noise_std * torch.randn_like(param.data)
            param.requires_grad = True
        return new_module.to(self.device).to(dtype=self.dtype)

    def freeze_old_experts(self, include_universal: bool = True):
        if not self.use_moe_experts:
            return
        if include_universal or self.freeze_universal:
            self.freeze(self.uni_image_adapter)
            self.freeze(self.uni_text_adapter)
        for exp in self.image_experts:
            self.freeze(exp)
        for exp in self.text_experts:
            self.freeze(exp)

    def add_new_task_expert(self, init_from: str = "universal", noise_std: float = 0.01):
        """Add a new expert for the current task and expand router head."""
        if not self.use_moe_experts:
            return self.update_injection_units(noise_std=noise_std)

        # Freeze older experts; router remains trainable.
        self.freeze_old_experts(include_universal=self.freeze_universal)

        base_img = self.uni_image_adapter if (init_from == "universal" or len(self.image_experts) == 0) else self.image_experts[-1]
        base_txt = self.uni_text_adapter if (init_from == "universal" or len(self.text_experts) == 0) else self.text_experts[-1]
        new_img = self._init_expert_from(base_img, noise_std=noise_std)
        self.image_experts.append(new_img)
        if self.moe_text:
            new_txt = self._init_expert_from(base_txt, noise_std=noise_std)
            self.text_experts.append(new_txt)

        # Ensure router output dimension matches expert count (U + task experts).
        if self.router is None:
            self.router = TopKRouter(512, self.router_hidden, num_experts=1, topk=self.router_topk).to(self.device)
        self.router.expand_output(1 + len(self.image_experts))
        self._latest_router_info = None
    


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

    # --- HÀM mix_matrix ĐÃ SỬA ---
    def mix_matrix(self):
        if self.use_moe_experts:
            # MoE path does not use universal mean fusion.
            return 
        # 1. Fusion Image Adapter
        if len(self.image_injection) < 1:
            return 
            
        all_flat_vectors = []
        for adapter in self.image_injection: 
            all_flat_vectors.append(self._flatten_adapter_params(adapter))
            
        v_uni_img = torch.stack(all_flat_vectors).mean(dim=0)

        # 2. GÁN v^uni CHO Universal Adapter
        self._unflatten_adapter_params(self.uni_image_adapter, v_uni_img)
        # --- SỬA LỖI 1: Đã xóa dòng ghi đè adapter mới nhất ---
        # self._unflatten_adapter_params(self.image_injection[-1], v_uni_img) 
        self.freeze(self.uni_image_adapter) 

        # 3. Fusion Text Adapter
        all_flat_vectors = []
        for adapter in self.text_injection:
            all_flat_vectors.append(self._flatten_adapter_params(adapter))
        
        v_uni_txt = torch.stack(all_flat_vectors).mean(dim=0)
        
        self._unflatten_adapter_params(self.uni_text_adapter, v_uni_txt)
        # --- SỬA LỖI 1: Đã xóa dòng ghi đè adapter mới nhất ---
        # self._unflatten_adapter_params(self.text_injection[-1], v_uni_txt) 
        self.freeze(self.uni_text_adapter)


    # ... (Các hàm apply_image_injection, apply_text_injection giữ nguyên) ...
    def _apply_image_moe(self, features: torch.Tensor):
        if self.router is None:
            self.router = TopKRouter(512, self.router_hidden, num_experts=1, topk=self.router_topk).to(self.device)
        router_logits = self.router(features.float())
        router_probs = torch.softmax(router_logits, dim=-1)
        k = min(self.router_topk, router_probs.shape[-1])
        topk_vals, topk_idx = torch.topk(router_probs, k=k, dim=-1)
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        experts = [self.uni_image_adapter] + list(self.image_experts)
        target_dtype = next(self.uni_image_adapter.parameters()).dtype
        expert_outs = torch.stack([exp(features.to(dtype=target_dtype)) for exp in experts], dim=1)
        selected = expert_outs.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, expert_outs.shape[-1]))
        mixed = (topk_weights.unsqueeze(-1) * selected).sum(dim=1)

        prob_mean = router_probs.mean(dim=0)
        load_balance = ((prob_mean - 1.0 / router_probs.shape[-1]) ** 2).sum()
        entropy = -(router_probs * router_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        router_info = {
            "probs": router_probs,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "load_balance": load_balance,
            "entropy": entropy,
            "num_experts": router_probs.shape[-1],
        }
        self._latest_router_info = router_info
        return mixed, router_info

    def apply_image_injection(self, features: torch.Tensor, is_old=False, return_router_info: bool = False):
        if self.use_moe_experts:
            mixed, router_info = self._apply_image_moe(features)
            if return_router_info:
                return mixed, router_info
            return mixed

        self._latest_router_info = None
        if len(self.image_injection) == 0:
            return (features, None) if return_router_info else features

        device = features.device 
        
        try:
            target_dtype = next(self.image_injection[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
            
        features = features.to(dtype=target_dtype)
        
        uni_output = self.uni_image_adapter(features)
        task_output = self.image_injection[-1](features)

        alpha = self.image_fusion_alpha.to(device)
        beta = self.image_fusion_beta.to(device)
        
        fusion_weights = torch.stack([alpha, beta], dim=0)
        normalized_weights = F.softmax(fusion_weights, dim=0)
        alpha_hat, beta_hat = normalized_weights[0], normalized_weights[1]

        outputs = (alpha_hat * uni_output) + (beta_hat * task_output)
        if return_router_info:
            return outputs, None
        return outputs

    def apply_text_injection(self, features: torch.Tensor, router_info: Optional[Dict[str, Tensor]] = None, batch_size: Optional[int] = None) -> torch.Tensor:
        if self.use_moe_experts:
            if not self.moe_text:
                # Stable text path: only universal text adapter (no routing) for class-conditioned text features.
                target_dtype = next(self.uni_text_adapter.parameters()).dtype
                return self.uni_text_adapter(features.to(dtype=target_dtype))
            # Experimental MoE-on-text (may be unstable).
            text_experts = [self.uni_text_adapter] + list(self.text_experts)
            target_dtype = next(self.uni_text_adapter.parameters()).dtype
            router_info = router_info or self._latest_router_info
            # Evaluate all experts once.
            expert_outs = torch.stack([exp(features.to(dtype=target_dtype)) for exp in text_experts], dim=0)
            if router_info is None or "topk_idx" not in router_info:
                # Fallback: uniform mix when router info is unavailable (e.g., offline text preprocessing).
                return expert_outs.mean(dim=0)

            topk_idx = router_info["topk_idx"]
            topk_weights = router_info["topk_weights"]
            B_total = topk_idx.shape[0]
            B = batch_size if batch_size is not None else min(B_total, features.shape[0])
            topk_idx = topk_idx[:B]
            topk_weights = topk_weights[:B]
            if features.dim() == 2 and features.shape[0] == B:
                # Per-sample text features (e.g., text tokens for each image in batch)
                expert_outs = expert_outs.permute(1, 0, 2)  # [B, E, D]
                selected = torch.gather(expert_outs, 1, topk_idx.unsqueeze(-1).expand(-1, -1, expert_outs.shape[-1]))
                weighted = (topk_weights.unsqueeze(-1) * selected).sum(dim=1)
                return weighted
            # Allow broadcasting when features already include batch dimension.
            if features.dim() == 3 and features.shape[0] == B:
                # Assume shape [B, C, D]; apply per-sample routing.
                expanded = expert_outs.unsqueeze(0).expand(B, -1, -1, -1)
                gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, topk_idx.shape[1], features.shape[1], features.shape[2])
                selected = torch.gather(expanded, 1, gather_idx)
                weighted = (topk_weights.unsqueeze(-1).unsqueeze(-1) * selected).sum(dim=1)
                return weighted

            expanded = expert_outs.unsqueeze(0).expand(B, -1, expert_outs.shape[1], expert_outs.shape[2])
            gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, topk_idx.shape[1], expert_outs.shape[1], expert_outs.shape[2])
            selected = torch.gather(expanded, 1, gather_idx)
            weighted = (topk_weights.unsqueeze(-1).unsqueeze(-1) * selected).sum(dim=1)
            return weighted

        if len(self.text_injection) == 0:
            return features
                    
        device = features.device
        
        try:
            target_dtype = next(self.text_injection[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
            
        features = features.to(dtype=target_dtype)
        
        uni_output = self.uni_text_adapter(features)
        task_output = self.text_injection[-1](features)

        alpha = self.text_fusion_alpha.to(device)
        beta = self.text_fusion_beta.to(device)

        fusion_weights = torch.stack([alpha, beta], dim=0)
        normalized_weights = F.softmax(fusion_weights, dim=0)
        alpha_hat, beta_hat = normalized_weights[0], normalized_weights[1]

        outputs = (alpha_hat * uni_output) + (beta_hat * task_output)
                    
        return outputs
    

    # ... (Các hàm còn lại giữ nguyên) ...
    
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
        init_from = "prev" if task_id > 0 else "universal"
        if self.use_moe_experts:
            self.add_new_task_expert(init_from=init_from, noise_std=self.sample_noise)
        elif task_id>0:
            self.update_injection_units()
        if task_id>0:
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
        base_features = clip_features

        feature_chunks = [base_features]
        counts = [base_features.shape[0]]
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            feature_chunks.append(memory_data)
            counts.append(memory_data.shape[0])
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            feature_chunks.append(edge_sample)
            counts.append(edge_sample.shape[0])

        combined_features = torch.cat(feature_chunks, dim=0)
        image_features_all, router_info = self.apply_image_injection(combined_features, return_router_info=True)
        self._latest_router_info = router_info

        idx = 0
        image_features = image_features_all[idx:idx + counts[0]]
        idx += counts[0]
        sg_image_features = None
        if memory_data is not None:
            sg_image_features = image_features_all[idx:idx + counts[1]]
            idx += counts[1]
        edge_sample_features = None
        if edge_sample is not None:
            edge_len = counts[2] if memory_data is not None else counts[1]
            edge_sample_features = image_features_all[idx:idx + edge_len]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        img_feas_parts = [image_features]
        if sg_image_features is not None:
            sg_image_features = sg_image_features / sg_image_features.norm(dim=-1, keepdim=True)
            img_feas_parts.append(sg_image_features)
        if edge_sample_features is not None:
            edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
            img_feas_parts.append(edge_sample_features)

        img_feas = torch.cat(img_feas_parts, dim=0)
        final_image_feas = img_feas

        edge_num = edge_sample_features.shape[0] if edge_sample_features is not None else 0
        if edge_sample is not None and edge_num > 0:
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

        final_text_feas = self.apply_text_injection(text_features, router_info=router_info if self.moe_text else None, batch_size=final_image_feas.shape[0])
        if final_text_feas.dim() == 2:
            final_text_feas = final_text_feas / final_text_feas.norm(dim=-1, keepdim=True)
            logits_per_image = self.logit_scale.exp() * final_image_feas @ final_text_feas.t().type(final_image_feas.dtype)
        else:
            final_text_feas = final_text_feas / final_text_feas.norm(dim=-1, keepdim=True)
            logits_per_image = self.logit_scale.exp() * torch.einsum('bd,bcd->bc', final_image_feas, final_text_feas.type(final_image_feas.dtype))
        probs = logits_per_image

        old_memory_feature = None
        if not_ini and memory_data is not None:
            old_memory_feature = sg_image_features
        if not_ini:
            if edge_sample is not None:
                return probs, final_image_feas, old_memory_feature, edge_sample_features, img_feas, raw_image_features
            return probs, final_image_feas, old_memory_feature, final_text_feas, img_feas, raw_image_features
        if ori_ima_f:
            if memory_data is not None:
                final_image_feas = final_image_feas[:-memory_data.shape[0]]
            return probs, original_image_features, final_image_feas, None, None, raw_image_features
        
        return probs, final_image_feas, None, edge_sample_features, img_feas, raw_image_features

    def forward_from_features(self, image_features: torch.Tensor, text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward helper that starts from pre-computed image features (for distillation replay)."""
        if text_features is None:
            text_features = self.class_name_features
        if self.use_moe_experts:
            image_features, router_info = self.apply_image_injection(image_features, return_router_info=True)
            self._latest_router_info = router_info
        else:
            router_info = None
            image_features = self.apply_image_injection(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        final_text_feas = self.apply_text_injection(text_features, router_info=router_info if self.moe_text else None, batch_size=image_features.shape[0])
        final_text_feas = final_text_feas / final_text_feas.norm(dim=-1, keepdim=True)
        if final_text_feas.dim() == 2:
            logits = self.logit_scale.exp() * image_features @ final_text_feas.t().type(image_features.dtype)
        else:
            logits = self.logit_scale.exp() * torch.einsum('bd,bcd->bc', image_features, final_text_feas.type(image_features.dtype))
        return logits

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
    
    def rerank(self, des_dict, outputs, image_features_raw, class_names, device, topk=5):
        with torch.no_grad():
            batch_size = image_features_raw.shape[0]
            topk_predict = outputs.topk(topk, dim=1)[1]
            topk_labels = [[class_names[int(label)] for label in pred] for pred in topk_predict]
            logi_total = torch.zeros(batch_size, topk, dtype=image_features_raw.dtype, device=device)
            for _ in range(3):
                texts = []
                for b in range(batch_size):
                    curr_texts = []
                    for i, main_label in enumerate(topk_labels[b]):
                        for j, second_label in enumerate(topk_labels[b]):
                            if i == j:
                                continue
                            main_label_norm = main_label.replace("_", " ")
                            second_label_norm = second_label.replace("_", " ")
                            if main_label_norm in des_dict and second_label_norm in des_dict[main_label_norm]:
                                desc = random.choice(des_dict[main_label_norm][second_label_norm])
                            elif main_label_norm in des_dict:
                                desc = random.choice(random.choice(list(des_dict[main_label_norm].values())))
                            else:
                                desc = "description"
                                print(f"[WARNING] Nhãn '{main_label_norm}' không có trong des_dict")  
                            curr_texts.append(f"{main_label_norm} with {desc.lower()}")
                    texts.extend(curr_texts)
                texts_token = self.tokenize(texts).to(device)
                texts_embed = self.encode_text(texts_token)
                texts_embed = texts_embed.to(image_features_raw.dtype)
                texts_embed = texts_embed.reshape(batch_size, topk, topk-1, -1)
                texts_embed = torch.mean(texts_embed, dim=2)
                texts_embed = texts_embed / texts_embed.norm(dim=-1, keepdim=True)
                logits = torch.bmm(image_features_raw.unsqueeze(1), texts_embed.transpose(1,2)).squeeze(1)
                logi_total += logits
            logits = logi_total / 3
            logits = logits.to(outputs.dtype)
            new_logits = torch.zeros_like(outputs)
            for i in range(batch_size):
                new_logits[i, topk_predict[i]] = logits[i]
            return new_logits

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
