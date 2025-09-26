import copy
import json
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
    
class MLP_Adapter(nn.Module):
    def __init__(self, c_in, hidden):
        super(MLP_Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden),
        )

    def forward(self, x):
        x_ = self.fc(x)
        return x_
    
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
        self.total_class_names = []
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = torch.float16 if cfg.fp16 else torch.float32
        self.adapter = nn.Linear(512, 512, bias=False, device=device)
        self.clip_type = model.dtype
        self.tokenize = clip.tokenize

        # ENGINE
        self.engine_cfg = getattr(cfg, 'engine', None)
        self.lambda_img = float(getattr(self.engine_cfg, 'lambda_img', 0.0)) if self.engine_cfg else 0.0
        self.lambda_txt = float(getattr(self.engine_cfg, 'lambda_txt', 0.0)) if self.engine_cfg else 0.0
        self.replay_alpha = float(getattr(self.engine_cfg, 'replay_alpha', 0.0)) if self.engine_cfg else 0.0
        self.replay_sample_num = int(getattr(self.engine_cfg, 'sample_num', 0)) if self.engine_cfg else 0
        self.image_injections = nn.ModuleList()
        self.text_injections = nn.ModuleList()
        self.new_des_dict = {}

        # old adapter
        self.old_adapter = None

        # class stats
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias
    

    def get_trainable_parameters(self):
        params = [self.adapter.parameters()]
        if len(self.image_injections) > 0:
            params.append(self.image_injections[-1].parameters())
        if len(self.text_injections) > 0:
            params.append(self.text_injections[-1].parameters())
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

    def update_injection_units(self):   
        if len(self.image_injections)>0:
            self.freeze(self.image_injections[-1])
            self.freeze(self.text_injections[-1])
        self.image_injections.append(MLP_Adapter(512, 512).to(self.device))
        self.text_injections.append(MLP_Adapter(512, 512).to(self.device))
    
    def apply_injections(self, modules: nn.ModuleList, features: torch.Tensor) -> torch.Tensor:
        res = []
        for i in range(len(modules)):
            res.append(modules[i](features))
        res = torch.sum(torch.stack(res), dim=0)
        return res
    
    def apply_injections(self, modules: nn.ModuleList, features: torch.Tensor) -> torch.Tensor:
        if len(modules) == 0:
            return torch.zeros_like(features)
        res = 0
        features = features.float()  
        for m in modules:
            res = res + m(features)
        return res
    
    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def adaptation(self, task_id, threshold=0):
        self.update_injection_units()
        
        self.total_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = self.tokenize(
            [self.prompt_template.format(c) for c in self.total_class_names]
        ).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        if task_id>0:
            self.old_adapter = copy.deepcopy(self.adapter)
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
        
        # --------- image features ---------
        image = image.type(self.dtype)
        with torch.no_grad():
            image_features = self.encode_image(image).float()
            original_image_features = image_features.clone()

        # concat memory data and apply injection 
        image_features = self.apply_injections(self.image_injections, image_features)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)        
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            sg_image_features = self.apply_injections(self.image_injections, memory_data)
            sg_image_features = sg_image_features / sg_image_features.norm(dim=-1, keepdim=True)
            img_feas = torch.cat([image_features, sg_image_features], dim=0)
        else:
            img_feas = image_features

        # not apply injection for edge samples
        pre_adapter = img_feas
        edge_num = 0

        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            pre_adapter = torch.cat([pre_adapter, edge_sample], dim=0)

        # RAPF: apply adapter
        final_image_feas = self.adapter(pre_adapter.type(self.dtype).detach()).type(self.clip_type)
        final_image_feas = final_image_feas / final_image_feas.norm(dim=1, keepdim=True)

        edge_sample_features = None
        if edge_sample is not None:
            edge_sample_features = final_image_feas[-edge_num:]
            final_image_feas = final_image_feas[:-edge_num]


        #---------- text features ---------
        with torch.no_grad():
            text_features = self.encode_text(self.text_tokens)

        final_text_feas = self.apply_injections(self.text_injections, text_features)
        final_text_feas = final_text_feas / final_text_feas.norm(dim=-1, keepdim=True)

        #---------- logits ---------
        logits_per_image = self.logit_scale.exp() * final_image_feas @ final_text_feas.t().type(final_image_feas.dtype)
        probs = logits_per_image


        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, final_image_feas, old_memory_feature, edge_sample_features, img_feas
            return probs, final_image_feas, old_memory_feature, final_text_feas, img_feas
        if ori_ima_f:
            if memory_data is not None:
                final_image_feas = final_image_feas[:-memory_data.shape[0]]
            return probs, original_image_features, final_image_feas
        
        return probs, final_image_feas, None, edge_sample_features, img_feas

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

    def _get_text_des(self, dataname: str = 'cifar224') -> Dict[str, List[str]]:
        path = Path("chat") / f"{dataname}_des.json"
        if not path.is_file():
            self.new_des_dict = {}
            return self.new_des_dict

        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            self.new_des_dict = {}
            return self.new_des_dict

        new_des = {k: self._flatten_values(v) for k, v in raw.items()}
        self.new_des_dict = new_des
        return self.new_des_dict
    
    def _get_batch_des(self, des_file: Dict[str, List[str]], classnames: Iterable[str]) -> List[str]:
        out: List[str] = []
        for cname in classnames:
            descs = des_file.get(cname, [])
            if descs: 
                out.append(f"{cname} with {random.choice(descs).casefold()}")
            else:    
                out.append(f"a photo of {cname}")
        return out
            


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
    