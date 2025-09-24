import copy
import pdb
from itertools import chain
from typing import Dict, List, Optional
from omegaconf import DictConfig

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .knowledge_injection import TextPromptBank, VisualAugEncoder
from .utils import get_class_ids_per_task, get_class_names

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
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
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = torch.float16 if cfg.fp16 else torch.float32
        self.adapter = nn.Linear(512, 512, bias=False, device=device)
        self.clip_type = model.dtype
        self.feature_dim = self.adapter.in_features

        # ENGINE
        self.engine_cfg = getattr(cfg, 'engine', None)
        self.text_prompt_bank = None
        self.visual_aug_encoder = None
        self.engine_otf_text = bool(
            self.engine_cfg
            and getattr(self.engine_cfg, 'enable_otf', False)
            and getattr(self.engine_cfg, 'otf_text', False)
        )
        self.engine_otf_visual = bool(
            self.engine_cfg
            and getattr(self.engine_cfg, 'enable_otf', False)
            and getattr(self.engine_cfg, 'otf_visual', False)
        )
        self.lambda_img = float(getattr(self.engine_cfg, 'lambda_img', 0.0)) if self.engine_cfg else 0.0
        self.lambda_txt = float(getattr(self.engine_cfg, 'lambda_txt', 0.0)) if self.engine_cfg else 0.0
        self.replay_alpha = float(getattr(self.engine_cfg, 'replay_alpha', 0.0)) if self.engine_cfg else 0.0
        self.replay_sample_num = int(getattr(self.engine_cfg, 'sample_num', 0)) if self.engine_cfg else 0
        self.image_injections = nn.ModuleList()
        self.text_injections = nn.ModuleList()
        self._last_forward_cache: Dict[str, torch.Tensor] = {}
        self.base_text_features: Optional[torch.Tensor] = None
        self.descriptor_text_features: Optional[torch.Tensor] = None
        self.observed_class_ids: List[int] = []
        self.class_id_to_position: Dict[int, int] = {}

        # old adapter
        self.old_adapter = None

        # class stats
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias

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

    def set_text_prompt_bank(self, prompt_bank):
        self.text_prompt_bank = prompt_bank
        if prompt_bank is not None:
            prompt_bank.register_encoder(self.encode_text, self.device, self.clip_type)

    def set_visual_aug_encoder(self, aug_encoder):
        self.visual_aug_encoder = aug_encoder
        if aug_encoder is not None:
            aug_encoder.register_encoder(self.encode_image, self.device, self.clip_type)

    def _use_engine_text(self):
        return bool(self.engine_otf_text and self.text_prompt_bank is not None)

    def _use_engine_visual(self):
        return bool(self.engine_otf_visual and self.visual_aug_encoder is not None)

    def _freeze_injection_units(self):
        for module in self.image_injections:
            for param in module.parameters():
                param.requires_grad = False
        for module in self.text_injections:
            for param in module.parameters():
                param.requires_grad = False

    def _prepare_injection_units(self):
        if not self.engine_cfg or not getattr(self.engine_cfg, 'enable_otf', False):
            return
        self._freeze_injection_units()
        new_image = nn.Linear(self.feature_dim, self.feature_dim, bias=False, device=self.device)
        nn.init.zeros_(new_image.weight)
        self.image_injections.append(new_image)
        new_text = nn.Linear(self.feature_dim, self.feature_dim, bias=False, device=self.device)
        nn.init.zeros_(new_text.weight)
        self.text_injections.append(new_text)

    def _apply_injections(self, modules: nn.ModuleList, features: torch.Tensor) -> torch.Tensor:
        if len(modules) == 0:
            return torch.zeros_like(features)
        outputs = [module(features) for module in modules]
        return torch.stack(outputs, dim=0).sum(dim=0)

    def get_trainable_parameters(self):
        params = [self.adapter.parameters()]
        if len(self.image_injections) > 0:
            params.append(self.image_injections[-1].parameters())
        if len(self.text_injections) > 0:
            params.append(self.text_injections[-1].parameters())
        return chain.from_iterable(params)

    def get_last_forward_cache(self):
        return self._last_forward_cache

    @torch.no_grad()
    def _update_text_buffers(self):
        self.base_text_features = self.encode_text(self.text_tokens).type(torch.float32)
        if self._use_engine_text():
            descriptor_feats = []
            for class_name in self.current_class_names:
                desc = self.text_prompt_bank.encode_multi(class_name)
                descriptor_feats.append(desc.mean(dim=0))
            self.descriptor_text_features = torch.stack(descriptor_feats).float()
        else:
            self.descriptor_text_features = None

    @torch.no_grad()
    def get_class_name_features(self):
        if self.base_text_features is None:
            self._update_text_buffers()
        final = self._apply_injections(self.text_injections, self.base_text_features)
        final = F.normalize(final.float(), dim=-1)
        return final.type(torch.float32)

    def adaptation(self, task_id, threshold=0):
        new_class_ids = self.class_ids_per_task[task_id]
        self.current_class_names += get_class_names(self.classes_names, new_class_ids)
        if not self.observed_class_ids:
            self.observed_class_ids = list(new_class_ids)
        else:
            self.observed_class_ids.extend(new_class_ids)
        self.class_id_to_position = {cid: idx for idx, cid in enumerate(self.observed_class_ids)}
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self._prepare_injection_units()
        self._update_text_buffers()
        self.class_name_features = self.get_class_name_features()

        self.queue_empty = True
        self.hard_pairs = None
        if task_id > 0:
            self.old_adapter = copy.deepcopy(self.adapter)
            dist_list = []
            for k, class_name_feature in enumerate(self.class_name_features[:-len(new_class_ids)]):
                diff = torch.cdist(
                    self.class_name_features[-len(new_class_ids):].type(torch.float32),
                    class_name_feature.unsqueeze(0).type(torch.float32)
                ).squeeze()
                dist_list.append(diff)
            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = torch.nonzero(mask)
            self.hard_pairs = indices
            if indices.shape[0] > 0:
                self.hard_pairs[:, 1] = self.hard_pairs[:, 1] + self.cfg.initial_increment + (task_id - 1) * self.cfg.increment

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        batch_size = image.shape[0]
        image = image.type(torch.float16)
        with torch.no_grad():
            base_image_features = self.encode_image(image).float()
        original_image_features = base_image_features.clone()

        injected_image_features = self._apply_injections(self.image_injections, base_image_features)
        pre_adapter_norm = F.normalize(injected_image_features, dim=-1)

        aug_image_features = None
        if self._use_engine_visual():
            aug_image_features = self.visual_aug_encoder.encode_image_aug(image).float()
        if aug_image_features is not None:
            combined_pre_adapter = F.normalize(torch.stack((pre_adapter_norm, aug_image_features), dim=0).mean(dim=0), dim=-1)
        else:
            combined_pre_adapter = pre_adapter_norm

        pre_adapter_all = combined_pre_adapter
        edge_num = 0
        if memory_data is not None:
            memory_data = memory_data.type_as(pre_adapter_all)
            pre_adapter_all = torch.cat([pre_adapter_all, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample.type_as(pre_adapter_all)
            edge_num = edge_sample.shape[0]
            pre_adapter_all = torch.cat([pre_adapter_all, edge_sample], dim=0)

        image_features = self.adapter(pre_adapter_all.type(self.dtype)).type(self.clip_type)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        else:
            edge_sample_features = None

        base_text = self.base_text_features
        if base_text is None:
            with torch.no_grad():
                base_text = self.encode_text(self.text_tokens).float()
            self.base_text_features = base_text
        text_delta = self._apply_injections(self.text_injections, base_text)
        final_text_features = base_text + text_delta
        final_text_features = F.normalize(final_text_features.float(), dim=-1)
        self.class_name_features = final_text_features.type(torch.float32)

        descriptor_norm = None
        if self.descriptor_text_features is not None:
            descriptor_norm = F.normalize(self.descriptor_text_features.float(), dim=-1)
        template_norm = F.normalize(self.base_text_features.float(), dim=-1)

        self._last_forward_cache = {
            "image_embeddings": pre_adapter_norm[:batch_size],
            "aug_image_embeddings": aug_image_features[:batch_size] if aug_image_features is not None else None,
            "combined_image_embeddings": combined_pre_adapter[:batch_size],
            "final_text_features": final_text_features,
            "template_text_features": template_norm,
            "descriptor_text_features": descriptor_norm,
            "batch_size": batch_size,
        }

        logits_per_image = self.logit_scale.exp() * image_features @ final_text_features.t().type(image_features.dtype)
        probs = logits_per_image
        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, image_features, old_memory_feature, edge_sample_features
            return probs, image_features, old_memory_feature, final_text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return probs, original_image_features, image_features
        return probs, image_features, None, edge_sample_features

    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append((max_distance.mean() - max_distance.min(), max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def mix_matrix(self):
        if self.old_adapter is not None:
            weight_new = self.adapter.weight.data
            weight_old = self.old_adapter.weight.data
            U_old, S_old, V_old = torch.linalg.svd(weight_old)
            P_new = U_old.T @ weight_new
            dist = (P_new - torch.diag(S_old) @ V_old).abs()
            mask = dist / dist.max()
            mask += self.mix_b
            mask = torch.clamp(mask, max=1)
            right = P_new * mask + torch.diag(S_old) @ V_old * (1 - mask)
            weight = U_old @ right
            self.adapter.weight.data = weight

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
    