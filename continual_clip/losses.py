"""Loss helpers for ENGINE-style knowledge injection."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def engine_contrastive_loss(similarity: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(similarity.size(0), device=similarity.device)
    return F.cross_entropy(similarity, targets) 
