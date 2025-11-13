from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(2.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, s_uni: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w1 = self.w1.to(device=s_uni.device, dtype=s_uni.dtype)
        w2 = self.w2.to(device=s_uni.device, dtype=s_uni.dtype)
        b = self.b.to(device=s_uni.device, dtype=s_uni.dtype)
        g = torch.sigmoid(w1 * s_uni - w2 * H + b)
        g = torch.clamp(g, 0.1, 0.9)
        return g, torch.ones_like(g) - g
