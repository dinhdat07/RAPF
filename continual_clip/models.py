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

        # ENGINE
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

        # class stats
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

            # ---- mean vector for new classes ----
            mu_list = []
            for class_idx in range(known_classes, total_classes):
                cls_vecs = vecs[labels == class_idx]
                if cls_vecs.numel() > 0:   # only calculate if have data
                    mean_vec = cls_vecs.mean(dim=0, keepdim=True)
                    mu_list.append(mean_vec)

            if len(mu_list) == 0:
                print("WARNING: No new class in this batch → skip update_stat")
                return

            mu = torch.cat(mu_list, dim=0)

            # ---- covariance ----
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

            # regularization for stable inverse
            dim = center_vecs.shape[1]
            reg = cov.trace() * torch.eye(dim, device=device)
            cov_inv = dim * torch.linalg.pinv((center_vecs.shape[0] - 1) * cov + reg)

            # ---- update/init ----
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

            # equal prior
            ps = torch.ones(self.mu.shape[0], device=device) / self.mu.shape[0]

            # update  weight & bias
            self.W = torch.einsum('nd,dc->cn', self.mu, self.cov_inv)   # [num_classes, dim]
            self.b = ps.log() - 0.5 * torch.einsum('nd,dc,nc->n', self.mu, self.cov_inv, self.mu)

    

    def get_trainable_parameters(self):
        params = []
        # Lấy parameters adapter cuối cùng (task mới)
        if self.image_injection and len(self.image_injection) > 0:
            params.append(self.image_injection[-1].parameters())
        if self.text_injection and len(self.text_injection) > 0:
            params.append(self.text_injection[-1].parameters())
        
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

        # Freeze adapter cũ, không freeze adapter mới
        # if self.image_injection:
        #         self.freeze(self.image_injection[-1])
        # if self.text_injection:
        #         self.freeze(self.text_injection[-1])
        # Thêm adapter mới vào list
        if self.image_injection:
            self.image_injection.append(self.image_injection[-1])
        else:
            self.image_injection.append(MLP_Adapter(512, 512).to(self.device).to(dtype=self.dtype))
        if self.text_injection:
            self.text_injection.append(self.text_injection[-1])
        else:
            self.text_injection.append(MLP_Adapter(512, 512).to(self.device).to(dtype=self.dtype))
            

    def apply_image_injection(self, features: torch.Tensor, is_old=False) -> torch.Tensor:
        features = features.float()
        image_res = []

        if len(self.image_injection) > 0:
            for inj in self.image_injection:
                image_res.append(inj(features))

            return torch.sum(torch.stack(image_res), dim=0)
        
        return features

    def apply_text_injection(self, features: torch.Tensor) -> torch.Tensor:
        # Lấy dtype của module đầu tiên trong text_injection
        target_dtype = next(self.text_injection[0].parameters()).dtype

        # Ép feature về cùng dtype
        features = features.to(dtype=target_dtype)

        text_res = []
        for i in range(len(self.text_injection)):
            text_res.append(self.text_injection[i](features))
        
        text_res = torch.sum(torch.stack(text_res), dim=0)
        return text_res
    
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
        
        # --------- image features ---------
        image = image.type(self.dtype)

        with torch.no_grad():
            clip_features = self.encode_image(image).float()
        raw_image_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        original_image_features = clip_features.clone()
        image_features = clip_features

        # concat memory data and apply injection 
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

    def _mix_linear(self, new_layer, old_layer,mask_scale=1.0):
        weight_new = new_layer.weight.data
        weight_old = old_layer.weight.data
        U_old, S_old, V_old = torch.linalg.svd(weight_old, full_matrices=False)
        P_new = U_old.T @ weight_new
        dist = (P_new - torch.diag(S_old) @ V_old).abs()
        mask = dist / dist.max()
        mask += self.mix_b
        mask = mask * mask_scale
        mask = torch.clamp(mask, max=1)
        right = P_new * mask + torch.diag(S_old) @ V_old * (1 - mask)
        weight = U_old @ right
        new_layer.weight.data = weight

    


    def mix_matrix(self):
        # Cần ít nhất 2 adapter để mix
        if len(self.image_injection) < 2:
            return

        curr_adapter_img = self.image_injection[-1]
        prev_adapters_img = self.image_injection[:-1]  # tất cả adapter cũ

        # Mix tất cả adapter cũ (nếu muốn học từ nhiều task cũ)
        for i, prev_adapter in enumerate(prev_adapters_img):
            # Giảm dần ảnh hưởng của adapter cũ theo khoảng cách thời gian
            mask_scale = 1.0 / (i + 1)
            self._mix_linear(curr_adapter_img.fc[0], prev_adapter.fc[0], mask_scale)
    
    # def mix_matrix(self):
    #     if len(self.image_injection) < 2:
    #         return
    #     curr_adapter_img = self.image_injection[-1]
    #     prev_adapter_img = self.image_injection[-2]
    #     self._mix_linear(curr_adapter_img.fc[0], prev_adapter_img.fc[0])



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

    def _get_text_des(self,dataname='cifar224'):
        root_path = Path(__file__).resolve().parent.parent  # từ model/model.py lên root
        des_path = root_path / "chat" / f"{dataname}_des.json"
        with open(des_path, 'r') as f:
            des_dict = json.load(f)
        self.des_dict = des_dict
        new_des_dict = {}
        for key, value in des_dict.items():
            new_key_value = []
            for k, v in value.items():
                new_key_value.extend(v)
            new_des_dict[key] = new_key_value
        self.new_des_dict = new_des_dict
        return new_des_dict
    
    def _get_batch_des(self, des_file: Dict[str, List[str]], classnames: Iterable[str]) -> List[str]:
        out: List[str] = []
        for cname in classnames:
            descs = des_file.get(cname, [])
            if descs: 
                out.append(f"{cname} with {random.choice(descs).casefold()}")
            else:    
                out.append(f"a photo of {cname}")
        return out
    
    def rerank(self, des_dict, outputs, image_features_raw, class_names, device, topk=5):
        with torch.no_grad():
            batch_size = image_features_raw.shape[0]

            topk_predict = outputs.topk(topk, dim=1)[1]  # (batch, topk)
            topk_labels = [[class_names[int(label)] for label in pred] for pred in topk_predict]


            # Khởi tạo logit tổng
            logi_total = torch.zeros(batch_size, topk, dtype=image_features_raw.dtype, device=device)

            for _ in range(3):  # trung bình 3 lần
                texts = []
                for b in range(batch_size):
                    curr_texts = []
                    for i, main_label in enumerate(topk_labels[b]):
                        for j, second_label in enumerate(topk_labels[b]):
                            if i == j:
                                continue
                            main_label_norm = main_label.replace("_", " ")
                            second_label_norm = second_label.replace("_", " ")

                            # fallback nếu nhãn không có trong des_dict
                            if main_label_norm in des_dict and second_label_norm in des_dict[main_label_norm]:
                                desc = random.choice(des_dict[main_label_norm][second_label_norm])
                            elif main_label_norm in des_dict:
                                desc = random.choice(random.choice(list(des_dict[main_label_norm].values())))
                            else:
                                desc = "description"
                                print(f"[WARNING] Nhãn '{main_label_norm}' không có trong des_dict")    

                            curr_texts.append(f"{main_label_norm} with {desc.lower()}")

                    texts.extend(curr_texts)

                # Encode text
                texts_token = self.tokenize(texts).to(device)
                texts_embed = self.encode_text(texts_token)

                # Match dtype image_features
                texts_embed = texts_embed.to(image_features_raw.dtype)

                # Reshape: (batch, topk, topk-1, embed_dim)
                texts_embed = texts_embed.reshape(batch_size, topk, topk-1, -1)
                texts_embed = torch.mean(texts_embed, dim=2)  # (batch, topk, embed_dim)
                texts_embed = texts_embed / texts_embed.norm(dim=-1, keepdim=True)

                # Batch matrix multiplication
                logits = torch.bmm(image_features_raw.unsqueeze(1), texts_embed.transpose(1,2)).squeeze(1)  # (batch, topk)
                logi_total += logits

            # Trung bình 3 lần
            logits = logi_total / 3
            logits = logits.to(outputs.dtype)

            # Gán vào topk dự đoán
            new_logits = torch.zeros_like(outputs)
            for i in range(batch_size):
                new_logits[i, topk_predict[i]] = logits[i]

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