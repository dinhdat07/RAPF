import math
from typing import Dict, List

import torch


StateDict = Dict[str, torch.Tensor]


def state_dict_to_cpu(sd: StateDict) -> StateDict:
    return {k: v.detach().cpu() for k, v in sd.items()}


def adapter_state_dict_cpu(adapter: torch.nn.Module) -> StateDict:
    return state_dict_to_cpu(adapter.state_dict())


def load_state_dict_to_device(module: torch.nn.Module, sd_cpu: StateDict) -> None:
    device_sd: StateDict = {}
    module_sd = module.state_dict()
    if set(module_sd.keys()) != set(sd_cpu.keys()):
        missing = set(module_sd.keys()).difference(sd_cpu.keys())
        extra = set(sd_cpu.keys()).difference(module_sd.keys())
        raise ValueError(f"State dict keys mismatch. Missing: {missing}, Extra: {extra}")
    for k, v in sd_cpu.items():
        tgt = module_sd[k]
        device_sd[k] = v.to(tgt.device).type(tgt.dtype)
    module.load_state_dict(device_sd)


def prune_topk_state_dict(sd: StateDict, topk_ratio: float) -> StateDict:
    pruned: StateDict = {}
    for k, v in sd.items():
        if topk_ratio <= 0:
            pruned[k] = torch.zeros_like(v)
            continue
        numel = v.numel()
        if numel == 0:
            pruned[k] = v.clone()
            continue
        k_top = max(1, int(numel * topk_ratio))
        if k_top >= numel:
            pruned[k] = v.clone()
            continue
        flat = v.view(-1).abs()
        thresh = torch.topk(flat, k_top, largest=True).values.min()
        mask = v.abs() >= thresh
        pruned[k] = v * mask
    return pruned


def subtract_state_dict(a: StateDict, b: StateDict) -> StateDict:
    if set(a.keys()) != set(b.keys()):
        raise ValueError("State dict keys mismatch in subtraction.")
    return {k: a[k] - b[k] for k in a.keys()}


def add_scaled_state_dict(base: StateDict, delta: StateDict, scale: float) -> StateDict:
    if set(base.keys()) != set(delta.keys()):
        raise ValueError("State dict keys mismatch in addition.")
    return {k: base[k] + scale * delta[k] for k in base.keys()}


def ties_majority_sign_merge(deltas: List[StateDict], topk_ratio: float) -> StateDict:
    if not deltas:
        raise ValueError("No deltas provided for TIES-lite merge.")
    pruned = [prune_topk_state_dict(d, topk_ratio) for d in deltas]
    merged: StateDict = {}
    window = len(pruned)
    threshold = math.ceil(window / 2)
    for key in pruned[0].keys():
        stack = torch.stack([d[key] for d in pruned], dim=0)
        signs = torch.sign(stack)
        sign_sum = signs.sum(dim=0)
        majority_mask = sign_sum.abs() >= threshold
        majority_sign = torch.sign(sign_sum) * majority_mask

        abs_stack = stack.abs()
        nonzero_mask = stack != 0
        nz_count = nonzero_mask.sum(dim=0)
        magnitude = abs_stack.sum(dim=0) / nz_count.clamp(min=1)
        magnitude = torch.where(nz_count > 0, magnitude, torch.zeros_like(magnitude))

        merged[key] = majority_sign * magnitude
    return merged


def apply_fusion_strategy(model, cfg, task_id: int, adapter_before_cpu: StateDict, adapter_after_cpu: StateDict) -> None:
    mode = cfg.fusion.mode if getattr(cfg, "fusion", None) is not None else "svd"
    if mode == "svd":
        model.mix_matrix()
        return

    if mode == "local_sparse":
        delta = subtract_state_dict(adapter_after_cpu, adapter_before_cpu)
        delta_pruned = prune_topk_state_dict(delta, cfg.fusion.topk_ratio)
        fused = add_scaled_state_dict(adapter_before_cpu, delta_pruned, cfg.fusion.lambda_fuse)
        load_state_dict_to_device(model.adapter, fused)
        return

    if mode == "ties_lite":
        delta = subtract_state_dict(adapter_after_cpu, adapter_before_cpu)
        if not hasattr(model, "_delta_history") or model._delta_history is None:
            model._delta_history = []
        model._delta_history.append(delta)
        model._delta_history = model._delta_history[-cfg.fusion.ties_window :]
        merged_delta = ties_majority_sign_merge(model._delta_history, cfg.fusion.topk_ratio)
        fused = add_scaled_state_dict(adapter_before_cpu, merged_delta, cfg.fusion.lambda_fuse)
        load_state_dict_to_device(model.adapter, fused)
        return

    if mode == "hybrid":
        # Step 1: original svd fusion on device
        model.mix_matrix()
        svd_cpu = adapter_state_dict_cpu(model.adapter)
        # Step 2: micro-correction from pre-svd adapter (adapter_after_cpu) to svd-fused
        micro_delta = subtract_state_dict(adapter_after_cpu, svd_cpu)
        micro_pruned = prune_topk_state_dict(micro_delta, cfg.fusion.micro_topk_ratio)
        final = add_scaled_state_dict(svd_cpu, micro_pruned, cfg.fusion.micro_lambda)
        load_state_dict_to_device(model.adapter, final)
        return

    raise ValueError(f"Unknown fusion mode: {mode}")

