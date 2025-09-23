"""Knowledge injection utilities for ENGINE-inspired modules."""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional

import clip
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

_DEFAULT_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_DEFAULT_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

__all__ = ["TextPromptBank", "VisualAugEncoder", "engine_rerank"]


class TextPromptBank:
    """Caches attribute-rich textual prompts for knowledge injection."""

    def __init__(
        self,
        descriptor_json: str,
        template: str = "a photo of {}",
        num_prompts_cap: Optional[int] = None,
    ) -> None:
        self.descriptor_json = descriptor_json
        self.template = template
        self.num_prompts_cap = num_prompts_cap

        self._descriptors: Dict[str, Dict[str, List[str]]] = {}
        self._prompt_cache: Dict[str, List[str]] = {}
        self._embedding_cache: Dict[str, Tensor] = {}
        self._name_map: Dict[str, str] = {}

        self._encode_fn: Optional[Callable[[Tensor], Tensor]] = None
        self._device: Optional[torch.device] = None
        self._encode_dtype: Optional[torch.dtype] = None

        raw_data = self._load_descriptor(descriptor_json)
        if isinstance(raw_data, dict):
            self._descriptors = self._normalize_descriptors(raw_data)
        else:
            self._descriptors = {}

    def _load_descriptor(self, path: str) -> Dict[str, Dict]:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                try:
                    return json.load(handle)
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _canonicalize(name: str) -> str:
        return name.strip().lower().replace("_", " ").replace("-", " ")

    def _normalize_descriptors(self, data: Dict[str, Dict]) -> Dict[str, Dict[str, List[str]]]:
        normalized: Dict[str, Dict[str, List[str]]] = {}
        self._name_map.clear()
        for raw_name, entry in data.items():
            normalized_entry = self._normalize_entry(raw_name, entry)
            normalized[raw_name] = normalized_entry
            self._name_map[self._canonicalize(raw_name)] = raw_name
        return normalized

    def _normalize_entry(self, class_name: str, entry: Dict) -> Dict[str, List[str]]:
        if isinstance(entry, dict) and (
            "synonyms" in entry
            or "attributes" in entry
            or "prompts" in entry
        ):
            synonyms = [str(s).strip() for s in entry.get("synonyms", []) if s]
            attributes = [str(a).strip() for a in entry.get("attributes", []) if a]
            prompts = [str(p).strip() for p in entry.get("prompts", []) if p]
            return {
                "synonyms": synonyms,
                "attributes": attributes,
                "prompts": prompts,
            }
        if isinstance(entry, dict):
            attr_candidates: List[str] = []
            for value in entry.values():
                if isinstance(value, list):
                    attr_candidates.extend([str(v).strip() for v in value if v])
                elif isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, list):
                            attr_candidates.extend([str(v).strip() for v in nested if v])
                        elif isinstance(nested, str):
                            attr_candidates.append(nested.strip())
                elif isinstance(value, str):
                    attr_candidates.append(value.strip())
            attributes = self._deduplicate(attr_candidates)[:256]
            return {
                "synonyms": [],
                "attributes": attributes,
                "prompts": [],
            }
        return {
            "synonyms": [],
            "attributes": [],
            "prompts": [],
        }

    @staticmethod
    def _deduplicate(items: Iterable[str]) -> List[str]:
        seen: OrderedDict[str, str] = OrderedDict()
        for item in items:
            cleaned = item.strip().strip('.')
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen[key] = cleaned
        return list(seen.values())

    def _resolve_name(self, class_name: str) -> str:
        if class_name in self._descriptors:
            return class_name
        return self._name_map.get(self._canonicalize(class_name), class_name)

    def _resolve_entry(self, class_name: str) -> Dict[str, List[str]]:
        resolved = self._resolve_name(class_name)
        return self._descriptors.get(resolved, {})

    def register_encoder(
        self,
        encode_fn: Callable[[Tensor], Tensor],
        device: torch.device,
        encode_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Attach the CLIP text encoder used for prompt embeddings."""

        self._encode_fn = encode_fn
        self._device = device
        self._encode_dtype = encode_dtype or torch.float16
        self._embedding_cache.clear()
        self._prompt_cache.clear()

    def get_prompts(self, class_name: str) -> List[str]:
        """Generate a list of prompt strings for a class."""

        resolved_name = self._resolve_name(class_name)
        cache_key = resolved_name
        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        entry = self._resolve_entry(class_name)
        synonyms = [syn for syn in entry.get("synonyms", []) if syn]
        attributes = [attr for attr in entry.get("attributes", []) if attr]
        prompt_templates = [p for p in entry.get("prompts", []) if p]

        noun_candidates = [resolved_name]
        for syn in synonyms:
            if syn not in noun_candidates:
                noun_candidates.append(syn)

        attribute_phrases = self._build_attribute_phrases(attributes)
        prompts: List[str] = []
        seen = set()

        def _add_prompt(candidate: str) -> None:
            candidate = candidate.strip()
            if not candidate:
                return
            if candidate not in seen:
                prompts.append(candidate)
                seen.add(candidate)

        for noun in noun_candidates:
            _add_prompt(self.template.format(noun))

        if prompt_templates:
            attr_sources = attribute_phrases or [""]
            for pattern in prompt_templates:
                for noun in noun_candidates:
                    for attr in attr_sources:
                        formatted = pattern.format(
                            class_name=noun,
                            attributes=attr,
                        ).strip()
                        _add_prompt(formatted)

        if not prompt_templates and attribute_phrases:
            for noun in noun_candidates:
                for attr in attribute_phrases:
                    enriched = f"{self.template.format(noun)}, {attr}"
                    _add_prompt(enriched)

        if not prompts:
            prompts.append(self.template.format(resolved_name))

        if self.num_prompts_cap is not None and len(prompts) > self.num_prompts_cap:
            prompts = prompts[: self.num_prompts_cap]

        self._prompt_cache[cache_key] = prompts
        return prompts

    def encode_multi(self, class_name: str) -> Tensor:
        """Encode all prompts for a class and return normalized features."""

        if self._encode_fn is None or self._device is None:
            raise RuntimeError("TextPromptBank encoder has not been registered.")

        cache_key = self._resolve_name(class_name)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        prompts = self.get_prompts(class_name)
        if not prompts:
            prompts = [self.template.format(cache_key)]
        tokens = clip.tokenize(prompts).to(self._device)

        with torch.no_grad():
            features = self._encode_fn(tokens)

        features = features.float()
        features = F.normalize(features, dim=-1)
        self._embedding_cache[cache_key] = features
        return features

    @staticmethod
    def _build_attribute_phrases(attributes: Iterable[str]) -> List[str]:
        attrs = [attr.strip() for attr in attributes if attr]
        if not attrs:
            return []
        chunk_size = max(1, min(3, len(attrs)))
        phrases = []
        for start in range(0, len(attrs), chunk_size):
            chunk = attrs[start : start + chunk_size]
            phrases.append(', '.join(chunk))
        return phrases


class VisualAugEncoder:
    """Applies strong image augmentations before CLIP feature extraction."""

    def __init__(
        self,
        base_preprocess,
        strong_aug_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
        self.base_preprocess = base_preprocess
        self.strong_aug_cfg = strong_aug_cfg or {}

        self._encode_fn: Optional[Callable[[Tensor], Tensor]] = None
        self._encode_dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None

        self._to_pil = transforms.ToPILImage()
        self._randaugment = None
        if "randaugment_n" in self.strong_aug_cfg and "randaugment_m" in self.strong_aug_cfg:
            self._randaugment = transforms.RandAugment(
                num_ops=int(self.strong_aug_cfg["randaugment_n"]),
                magnitude=int(self.strong_aug_cfg["randaugment_m"]),
            )
        color_jitter = self.strong_aug_cfg.get("color_jitter")
        self._color_jitter = (
            transforms.ColorJitter(*color_jitter)
            if isinstance(color_jitter, (list, tuple)) and len(color_jitter) in {3, 4}
            else None
        )
        random_erasing_p = float(self.strong_aug_cfg.get("random_erasing_p", 0.0))
        self._random_erasing = (
            transforms.RandomErasing(p=random_erasing_p, inplace=False)
            if random_erasing_p > 0
            else None
        )

        self._mean = torch.tensor(_DEFAULT_CLIP_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_DEFAULT_CLIP_STD).view(3, 1, 1)
        self._extract_mean_std(base_preprocess)

    def register_encoder(
        self,
        encode_fn: Callable[[Tensor], Tensor],
        device: torch.device,
        encode_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._encode_fn = encode_fn
        self._device = device
        self._encode_dtype = encode_dtype or torch.float16

    def encode_image_aug(self, image_batch: Tensor) -> Tensor:
        if self._encode_fn is None or self._device is None or self._encode_dtype is None:
            raise RuntimeError("VisualAugEncoder encoder has not been registered.")

        processed: List[Tensor] = []
        for image in image_batch:
            img = image.detach().to(self._device)
            img = self._denormalize(img)
            pil_img = self._to_pil(img.cpu().clamp(0.0, 1.0))
            if self._randaugment is not None:
                pil_img = self._randaugment(pil_img)
            if self._color_jitter is not None:
                pil_img = self._color_jitter(pil_img)
            tensor_img = self.base_preprocess(pil_img)
            if self._random_erasing is not None:
                tensor_img = self._random_erasing(tensor_img)
            processed.append(tensor_img)

        if not processed:
            raise ValueError("encode_image_aug received an empty batch.")

        batch = torch.stack(processed).to(device=self._device, dtype=self._encode_dtype)

        with torch.no_grad():
            features = self._encode_fn(batch)

        features = F.normalize(features.float(), dim=-1)
        return features

    def _extract_mean_std(self, preprocess) -> None:
        transforms_list = getattr(preprocess, "transforms", None)
        if not transforms_list:
            return
        for transform in transforms_list:
            if isinstance(transform, transforms.Normalize):
                mean = torch.tensor(transform.mean).view(3, 1, 1)
                std = torch.tensor(transform.std).view(3, 1, 1)
                self._mean = mean
                self._std = std

    def _denormalize(self, tensor: Tensor) -> Tensor:
        mean = self._mean.to(tensor.device)
        std = self._std.to(tensor.device)
        return tensor * std + mean


def engine_rerank(
    image_feat: Tensor,
    logits: Tensor,
    class_names: List[str],
    text_bank: TextPromptBank,
    alpha: float = 0.7,
    topk: int = 5,
) -> Tensor:
    """Re-rank logits using textual similarity cues from ENGINE."""

    if topk <= 0:
        return logits

    probs = logits.softmax(dim=-1)
    topk = min(topk, probs.size(-1))
    _, top_indices = probs.topk(topk, dim=-1)
    fused = probs.clone()

    for batch_idx in range(image_feat.shape[0]):
        image_vec = F.normalize(image_feat[batch_idx].float(), dim=-1, eps=1e-6)
        for cls_idx in top_indices[batch_idx]:
            cls = class_names[int(cls_idx)] if class_names else str(int(cls_idx))
            text_features = text_bank.encode_multi(cls).to(image_vec.device)
            sims = torch.matmul(image_vec.unsqueeze(0), text_features.t()).squeeze(0)
            score_text = torch.clamp(sims.max(), min=0.0)
            fused[batch_idx, cls_idx] = alpha * fused[batch_idx, cls_idx] + (1 - alpha) * score_text

    fused = torch.clamp(fused, min=1e-6)
    fused = fused / fused.sum(dim=-1, keepdim=True)
    return fused.log().to(logits.dtype)
