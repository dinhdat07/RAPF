from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


ENGINE_DEFAULTS: Dict[str, Any] = {
    "lambda_img": 0.0,
    "lambda_txt": 0.0,
    "replay_alpha": 0.25,
    "sample_num": 0,
    "stat": 0.0,
    "sample_noise": 0.0,
    "templates": [],
}

MODEL_NAME_ALIASES = {
    "ViT-B-16": "ViT-B/16",
    "ViT-L-14": "ViT-L/14",
}


def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def normalize_runtime_config(cfg: DictConfig) -> DictConfig:
    """Normalize config keys while preserving runtime behavior.

    Supports both legacy flat keys and nested `engine.*` keys.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    normalized = OmegaConf.create(resolved)
    OmegaConf.set_struct(normalized, False)

    engine_cfg = _select(normalized, "engine", {}) or {}

    for key, default in ENGINE_DEFAULTS.items():
        current_value = _select(normalized, key)
        if current_value is None:
            current_value = engine_cfg.get(key, default)
        normalized[key] = current_value

    dropout = _select(normalized, "dropout")
    if dropout is None:
        dropout = engine_cfg.get("dropout", 0.1)
    normalized["dropout"] = float(dropout)

    model_name = str(_select(normalized, "model_name", "ViT-B/16"))
    normalized["model_name"] = MODEL_NAME_ALIASES.get(model_name, model_name)

    if _select(normalized, "templates") is None:
        normalized["templates"] = []

    return normalized
