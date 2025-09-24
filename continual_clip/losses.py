"""Loss helpers for ENGINE-style knowledge injection."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def contrastive_loss(similarity: Tensor, temperature: float = 1.0) -> Tensor:
    """Symmetric InfoNCE loss on a square similarity matrix.

    Args:
        similarity: Pairwise similarity logits of shape [N, N]. Each row/column
            corresponds to a paired element (e.g., image vs. text).
        temperature: Optional temperature divisor. When 1.0, logits are used as-is.

    Returns:
        Scalar tensor with the symmetric InfoNCE objective.
    """
    if similarity.dim() != 2 or similarity.size(0) != similarity.size(1):
        raise ValueError("contrastive_loss expects a square similarity matrix")

    logits = similarity / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_t)


def compute_similarity(a: Tensor, b: Tensor, normalize: bool = True) -> Tensor:
    """Compute pairwise cosine similarity between two batches of features."""
    if normalize:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
    return a @ b.t()


def pair_contrastive_loss(
    anchors: Tensor,
    positives: Tensor,
    temperature: float = 1.0,
    normalize: bool = True,
) -> Tensor:
    """Convenience wrapper for InfoNCE on paired feature batches."""
    if anchors.size(0) != positives.size(0):
        raise ValueError("Anchors and positives must have matching batch size")
    sim = compute_similarity(anchors, positives, normalize=normalize)
    return contrastive_loss(sim, temperature=temperature)
