from __future__ import annotations

from typing import Iterable, List, Optional

import clip
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .utils import get_class_ids_per_task, get_class_names
    
class LinearAdapter(nn.Module):
    def __init__(self, c_in, hidden):
        super(LinearAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, hidden))

    def forward(self, x):
        x_ = self.fc(x)
        return x_


def shrink_cov(cov: torch.Tensor) -> torch.Tensor:
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask).sum() / mask.sum()
    identity = torch.eye(cov.shape[0], device=cov.device)
    return cov + (diag_mean * identity) + (off_diag_mean * (1 - identity))


def sample(mean: torch.Tensor, cov: torch.Tensor, size: int, shrink: bool = False) -> torch.Tensor:
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    return vec @ sqrt_cov.t() + mean


class SigmaClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg: DictConfig, device: torch.device, jit: bool = False):
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

        # Sequential adapters: one active image adapter and one active text adapter.
        self.image_adapter = None
        self.text_adapter = None

        self.lambda_img = cfg.lambda_img
        self.lambda_txt = cfg.lambda_txt
        self.replay_sample_num = cfg.sample_num
        self.sample_noise = cfg.sample_noise

        self.prototype = []
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.templates_per_class = 1
        self.class_name_features = None
        self.mu = None
        self.cov_inv = None
        self.W = None
        self.b = None

    def init_adapter(self):
        self.image_adapter = LinearAdapter(512, 512).to(self.device).to(dtype=self.dtype)
        self.text_adapter = LinearAdapter(512, 512).to(self.device).to(dtype=self.dtype)

    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("Updating stat...")
        with torch.no_grad():
            vectors, labels = [], []
            for images, targets, _ in train_loader:
                images, targets = images.to(device), targets.to(device)
                image_features = self.encode_image(images).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vectors.append(image_features)
                labels.append(targets)

            if not vectors:
                return

            vectors = torch.cat(vectors)
            labels = torch.cat(labels)

            mu_list = []
            for class_idx in range(known_classes, total_classes):
                class_vectors = vectors[labels == class_idx]
                if class_vectors.numel() > 0:
                    mu_list.append(class_vectors.mean(dim=0, keepdim=True))

            if not mu_list:
                return

            mu = torch.cat(mu_list, dim=0)
            center_list = []
            for index, class_id in enumerate(range(known_classes, total_classes)):
                class_vectors = vectors[labels == class_id]
                if class_vectors.numel() > 0:
                    center_list.append(class_vectors - mu[index])

            if not center_list:
                return

            center_vectors = torch.cat(center_list, dim=0)
            cov = center_vectors.T @ center_vectors / (center_vectors.shape[0] - 1)
            dim = center_vectors.shape[1]
            reg = cov.trace() * torch.eye(dim, device=device)
            cov_inv = dim * torch.linalg.pinv((center_vectors.shape[0] - 1) * cov + reg)

            if self.mu is None:
                self.mu = mu
                self.cov_inv = cov_inv
            else:
                self.cov_inv = (
                    (known_classes / total_classes) * self.cov_inv
                    + (total_classes - known_classes) / total_classes * cov_inv
                    + (
                        (known_classes / total_classes)
                        * (total_classes - known_classes)
                        / (total_classes**2)
                    )
                    * (
                        (self.mu.mean(dim=0) - mu.mean(dim=0)).unsqueeze(1)
                        @ (self.mu.mean(dim=0) - mu.mean(dim=0)).unsqueeze(0)
                    )
                )
                self.mu = torch.cat([self.mu, mu])

            priors = torch.ones(self.mu.shape[0], device=device) / self.mu.shape[0]
            self.W = torch.einsum("nd,dc->cn", self.mu, self.cov_inv)
            self.b = priors.log() - 0.5 * torch.einsum("nd,dc,nc->n", self.mu, self.cov_inv, self.mu)

    def get_trainable_parameters(self):
        params = []
        if self.image_adapter is not None:
            params.append(self.image_adapter.parameters())
        if self.text_adapter is not None:
            params.append(self.text_adapter.parameters())
        return (param for group in params for param in group)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.clip_type)
        x = x + self.positional_embedding.type(self.clip_type)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    def encode_image(self, image):
        return self.visual(image.to(self.clip_type))

    def apply_image_injection(self, features):
        if self.image_adapter is None:
            raise RuntimeError("Image adapter is not initialized. Call update_injection_units() first.")
        target_dtype = next(self.image_adapter.parameters()).dtype
        return self.image_adapter(features.to(dtype=target_dtype))

    def apply_text_injection(self, features):
        if self.text_adapter is None:
            raise RuntimeError("Text adapter is not initialized. Call update_injection_units() first.")
        target_dtype = next(self.text_adapter.parameters()).dtype
        return self.text_adapter(features.to(dtype=target_dtype))

    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        templates_per_class = self.templates_per_class
        if templates_per_class > 1:
            num_classes = len(self.total_class_names)
            class_name_features = class_name_features.view(num_classes, templates_per_class, -1)
            class_name_features = class_name_features / class_name_features.norm(dim=-1, keepdim=True)
            class_name_features = class_name_features.mean(dim=1)
        return class_name_features.type(torch.float32)

    def adaptation(self, task_id, threshold: float = 0):
        self.known_classes = len(self.total_class_names)
        self.total_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])

        prompt_templates = self.cfg.templates

        self.templates_per_class = len(prompt_templates)
        all_prompts = []
        for class_name in self.total_class_names:
            all_prompts.extend([template.format(class_name) for template in prompt_templates])

        self.text_tokens = self.tokenize(all_prompts).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)

        self.queue_empty = True
        self.hard_pairs = None

        if task_id > 0:
            dist_list = []
            old_count = len(self.class_ids_per_task[task_id])
            for class_name_feature in self.class_name_features[:-old_count]:
                diff = torch.cdist(
                    self.class_name_features[-old_count:].type(torch.float32),
                    class_name_feature.unsqueeze(0).type(torch.float32),
                ).squeeze()
                dist_list.append(diff)

            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = torch.nonzero(mask)
            self.hard_new_class = (
                torch.unique(indices[:, 1]) + self.cfg.initial_increment + (task_id - 1) * self.cfg.increment
            )
            self.hard_pairs = indices
            self.hard_pairs[:, 1] = (
                self.hard_pairs[:, 1] + self.cfg.initial_increment + (task_id - 1) * self.cfg.increment
            )

    def forward(self, image, ori_ima_f: bool = False, memory_data=None, not_ini: bool = False, edge_sample=None):
        image = image.type(self.dtype)

        with torch.no_grad():
            clip_features = self.encode_image(image).float()
            raw_image_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            original_image_features = clip_features.clone()
            image_features = clip_features

        image_features = self.apply_image_injection(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            synthetic_features = self.apply_image_injection(memory_data)
            synthetic_features = synthetic_features / synthetic_features.norm(dim=-1, keepdim=True)
            image_features_for_logits = torch.cat([image_features, synthetic_features], dim=0)
        else:
            image_features_for_logits = image_features

        edge_num = 0
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            edge_sample = self.apply_image_injection(edge_sample)
            edge_sample = edge_sample / edge_sample.norm(dim=-1, keepdim=True)
            image_features_for_logits = torch.cat([image_features_for_logits, edge_sample], dim=0)

        final_image_features = image_features_for_logits

        edge_sample_features = None
        if edge_sample is not None:
            edge_sample_features = final_image_features[-edge_num:]
            final_image_features = final_image_features[:-edge_num]

        if self.class_name_features is None:
            raise RuntimeError("Class-name features are not initialized. Call adaptation() first.")
        text_features = self.class_name_features

        final_text_features = self.apply_text_injection(text_features)
        final_text_features = final_text_features / final_text_features.norm(dim=-1, keepdim=True)

        logits_per_image = self.logit_scale.exp() * final_image_features @ final_text_features.t().type(final_image_features.dtype)
        probs = logits_per_image

        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.apply_image_injection(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return (
                    probs,
                    final_image_features,
                    old_memory_feature,
                    edge_sample_features,
                    image_features_for_logits,
                    raw_image_features,
                )
            return (
                probs,
                final_image_features,
                old_memory_feature,
                final_text_features,
                image_features_for_logits,
                raw_image_features,
            )

        if ori_ima_f:
            if memory_data is not None:
                final_image_features = final_image_features[:-memory_data.shape[0]]
            return probs, original_image_features, final_image_features, None, None, raw_image_features

        return probs, final_image_features, None, edge_sample_features, image_features_for_logits, raw_image_features

    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for class_label in label:
            index = torch.nonzero(labels == class_label).squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            proto = mean.detach().to(torch.float32)
            class_idx = int(class_label.item())

            if len(self.prototype) <= class_idx:
                self.prototype.append(proto)
            else:
                self.prototype[class_idx] = proto

            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append(
                (
                    max_distance.mean() - max_distance.min(),
                    max_distance.max() - max_distance.mean(),
                    max_distance.mean(),
                )
            )
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def _get_text_anchor(self, classnames: Iterable[str]) -> List[str]:
        return [f"a photo of {cname}" for cname in classnames]
