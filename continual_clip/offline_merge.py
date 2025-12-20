import math
import os
from typing import Dict, List

import torch


StateDict = Dict[str, torch.Tensor]


def _assert_same_keys(thetas: List[StateDict]) -> None:
    """Ensure all state dicts share the exact same parameter names."""
    if not thetas:
        return
    base_keys = set(thetas[0].keys())
    for idx, theta in enumerate(thetas[1:], start=1):
        if set(theta.keys()) != base_keys:
            missing = base_keys.difference(theta.keys())
            extra = set(theta.keys()).difference(base_keys)
            raise ValueError(
                f"theta at index {idx} has mismatched keys. Missing: {missing}, Extra: {extra}"
            )


def save_adapter_state(adapter: torch.nn.Module, path: str) -> None:
    """Persist adapter parameters on CPU to save VRAM."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    torch.save(state, path)


def load_thetas(theta_dir: str, num_tasks: int) -> List[StateDict]:
    thetas: List[StateDict] = []
    for task_idx in range(1, num_tasks + 1):
        theta_path = os.path.join(theta_dir, f"theta_{task_idx}.pt")
        if not os.path.exists(theta_path):
            raise FileNotFoundError(f"Missing theta checkpoint for task {task_idx}: {theta_path}")
        thetas.append(torch.load(theta_path, map_location="cpu"))
    _assert_same_keys(thetas)
    return thetas


def select_base_index(base_task: str, num_tasks: int) -> int:
    if base_task == "first":
        return 0
    if base_task == "middle":
        return int(math.ceil(num_tasks / 2)) - 1
    if base_task == "last":
        return num_tasks - 1
    raise ValueError(f"Unknown base_task: {base_task}")


def compute_task_vectors(thetas: List[StateDict], base_idx: int) -> List[StateDict]:
    _assert_same_keys(thetas)
    base = thetas[base_idx]
    task_vectors: List[StateDict] = []
    for j, theta in enumerate(thetas):
        if j == base_idx:
            continue  # skip zero vector of the base itself
        vector = {k: theta[k] - base[k] for k in base.keys()}
        task_vectors.append(vector)
    return task_vectors



def _stack_and_select_by_abs(tensors: List[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(tensors, dim=0)
    idx = stacked.abs().argmax(dim=0)
    # gather expects the same shape as idx expanded with leading dim
    gathered = torch.gather(stacked, 0, idx.unsqueeze(0)).squeeze(0)
    return gathered

def prune_topk(task_vector: StateDict, topk_ratio: float) -> StateDict:
    pruned = {}
    for k, v in task_vector.items():
        flat = v.view(-1)
        k_num = max(1, int(topk_ratio * flat.numel()))
        thresh = flat.abs().kthvalue(flat.numel() - k_num).values
        mask = v.abs() >= thresh
        pruned[k] = v * mask
    return pruned


def merge_magmax(task_vectors: List[StateDict]) -> StateDict:
    """Standard MagMax: per-parameter maximum magnitude selection."""
    merged: StateDict = {}
    for key in task_vectors[0].keys():
        merged[key] = _stack_and_select_by_abs([tv[key] for tv in task_vectors])
    return merged


def merge_la_magmax(
    task_vectors: List[StateDict],
    difficulties: List[float],
    gamma: float,
) -> StateDict:
    """
    Learning-aware MagMax:
    - Use difficulty only to compute a selection score.
    - Select per-element winner by argmax(score).
    - Return the ORIGINAL (unweighted) tau values from the winning task.
    Score: |tau_i| * (1 + gamma * g_i)
    """
    if len(difficulties) != len(task_vectors):
        raise ValueError(
            f"Difficulty list length {len(difficulties)} does not match number of task vectors {len(task_vectors)}"
        )
    if not task_vectors:
        return {}

    g = torch.tensor(difficulties, dtype=torch.float32)  # [T]
    g = torch.clamp(g, min=0.0)
    w = 1.0 + gamma * g  # [T], non-negative scaling for SCORE only

    merged: StateDict = {}
    for key in task_vectors[0].keys():
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)  # [T, ...] on CPU
        # broadcast w to [T, 1, 1, ...]
        w_view = w.view(-1, *([1] * (stacked.dim() - 1)))
        scores = stacked.abs() * w_view
        idx = scores.argmax(dim=0)  # [...]
        merged[key] = torch.gather(stacked, 0, idx.unsqueeze(0)).squeeze(0)  # take ORIGINAL tau
    return merged



def apply_merge(theta_base: StateDict, merged_vector: StateDict, lambda_merge: float) -> StateDict:
    return {k: theta_base[k] + lambda_merge * merged_vector[k] for k in theta_base.keys()}

