import copy
import math
from itertools import chain
import random
from typing import List
from omegaconf import DictConfig
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_class_ids_per_task, get_class_names

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSAAdapter(nn.Module):
    def __init__(self, c_in, hidden, dropout=0.1, use_layernorm=True, learnable_scale=True):
        super(FSAAdapter, self).__init__()

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(c_in)

        self.down_proj = nn.Linear(c_in, hidden)
        self.ln_hidden = nn.LayerNorm(hidden)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(hidden, c_in)
        self.dropout = nn.Dropout(dropout)

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.tensor(1.0))
        
        # FSA-specific: frozen snapshot of previous adapter
        self.frozen_snapshot = None
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, return_delta=False):
        residual = x
        if self.use_layernorm:
            x = self.layernorm(x)
        x = self.down_proj(x)
        x = self.ln_hidden(x)
        x = self.non_linear(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        delta = x * self.scale

        if return_delta:
            return delta
        return residual + delta

    def forward_frozen(self, x):
        if self.frozen_snapshot is None:
            return torch.zeros_like(x)
        
        with torch.no_grad():
            return self.frozen_snapshot(x, return_delta=True)

    def create_snapshot(self):
        # Deep copy current adapter and freeze all parameters
        snapshot = copy.deepcopy(self)
        for param in snapshot.parameters():
            param.requires_grad = False
        snapshot.eval()
        
        self.frozen_snapshot = snapshot
        return snapshot

    def get_anchor_loss(self, replay_features):
        if self.frozen_snapshot is None:
            return torch.tensor(0.0, device=replay_features.device)
        
        # Current adapter output
        delta_current = self.forward(replay_features, return_delta=True)
        
        # Frozen adapter output (no grad)
        delta_frozen = self.forward_frozen(replay_features)
        
        # L2 distance in function space
        anchor_loss = F.mse_loss(delta_current, delta_frozen)
        
        return anchor_loss


class Adapter(FSAAdapter):
    pass


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

        # per-task image adapters
        self.engine_cfg = getattr(cfg, 'engine', None)
        self.lambda_img = float(getattr(self.engine_cfg, 'lambda_img', 0.0)) if self.engine_cfg else 0.0
        self.replay_sample_num = int(getattr(self.engine_cfg, 'sample_num', 0)) if self.engine_cfg else 0
        self.sample_noise = float(getattr(self.engine_cfg, 'sample_noise', 0.25)) if self.engine_cfg else 0.25
        self.image_injection = nn.ModuleList()
        self.prev_image_injection = None

        self.prototype: List[torch.Tensor] = []
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []

    
    def get_trainable_parameters(self):
        params = []
        if self.image_injection and len(self.image_injection) > 0:
            params.append(self.image_injection[-1].parameters())
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

    def update_injection_units(self, noise_std=0.01):
        for inj in self.image_injection: 
            self.freeze(inj)

        dropout_rate = float(getattr(self.cfg, 'dropout', 0.1))

        if self.image_injection:
            new_image_adapter = copy.deepcopy(self.image_injection[-1])
            for param in new_image_adapter.parameters():
                param.data += noise_std * torch.randn_like(param.data)
                param.requires_grad = True 
            
            if hasattr(new_image_adapter, 'scale'):
                with torch.no_grad():
                    new_image_adapter.scale.fill_(1.0) 
            
            self.image_injection.append(new_image_adapter.to(self.device).to(dtype=self.dtype))
        else:
            self.image_injection.append(Adapter(512, 256, dropout=dropout_rate).to(self.device).to(dtype=self.dtype))

    def _flatten_adapter_params(self, adapter):
        params = []
        for name, param in adapter.named_parameters():
            if isinstance(param, nn.Parameter):
                params.append(param.data.flatten())
        return torch.cat(params)

    def _unflatten_adapter_params(self, adapter, flat_vector):
        pointer = 0
        for name, param in adapter.named_parameters():
            if isinstance(param, nn.Parameter):
                num_elements = param.numel()
                param.data.copy_(flat_vector[pointer:pointer + num_elements].view_as(param.data))
                pointer += num_elements
        return adapter

    def apply_image_injection(self, features):
        if len(self.image_injection) == 0:
            return features
        try:
            target_dtype = next(self.image_injection[0].parameters()).dtype
        except StopIteration:
            target_dtype = features.dtype
            
        features = features.to(dtype=target_dtype)
        task_output = self.image_injection[-1](features)
        outputs = task_output
        return outputs
    
    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def adaptation(self, task_id, threshold=0):
        self.known_classes = len(self.total_class_names)
        self.total_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
 
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.total_class_names]
        ).to(self.device)

        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        if task_id > 0:
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

        final_text_feas = text_features
        final_text_feas = final_text_feas / final_text_feas.norm(dim=-1, keepdim=True)

        #---------- logits ---------
        logits_per_image = self.logit_scale.exp() * final_image_feas @ final_text_feas.t().type(final_image_feas.dtype)
        probs = logits_per_image

        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.apply_image_injection(memory_data)
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
