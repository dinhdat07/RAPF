import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSAAdapter(nn.Module):
    """
    Function-Space Anchoring (FSA) Adapter
    
    Extends the baseline adapter with frozen snapshot capability for 
    function-space regularization in continual learning.
    """
    def __init__(self, c_in, hidden, dropout=0.1, use_layernorm=True, learnable_scale=True):
        super(FSAAdapter, self).__init__()

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(c_in)

        self.down_proj = nn.Linear(c_in, hidden)
        self.ln_hidden = nn.LayerNorm(hidden)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(hidden, c_in)
        self.dropout = nn.Dropout(dropout)

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.tensor(1.0))
        
        # FSA-specific: frozen snapshot of previous adapter
        self.frozen_snapshot = None
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, return_delta=False):
        """
        Args:
            x: Input features [B, D]
            return_delta: If True, returns only Δ(x), not residual connection
        
        Returns:
            output: x + scale * Δ(x) if return_delta=False, else scale * Δ(x)
        """
        residual = x
        if self.use_layernorm:
            x = self.layernorm(x)
        x = self.down_proj(x)
        x = self.ln_hidden(x)
        x = self.non_linear(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        delta = x * self.scale

        if return_delta:
            return delta
        return residual + delta

    def forward_frozen(self, x):
        """
        Forward pass through frozen snapshot (if exists).
        Used for computing FSA anchor loss.
        
        Returns:
            delta_old: Δ_{t-1}(x) from previous task's adapter
        """
        if self.frozen_snapshot is None:
            return torch.zeros_like(x)
        
        with torch.no_grad():
            return self.frozen_snapshot(x, return_delta=True)

    def create_snapshot(self):
        """
        Create a frozen copy of current adapter state.
        Called after each task completes.
        """
        # Deep copy current adapter and freeze all parameters
        snapshot = copy.deepcopy(self)
        for param in snapshot.parameters():
            param.requires_grad = False
        snapshot.eval()
        
        self.frozen_snapshot = snapshot
        return snapshot

    def get_anchor_loss(self, replay_features):
        """
        Compute function-space anchoring loss on replay features.
        
        Args:
            replay_features: [N, D] features from old tasks
        
        Returns:
            anchor_loss: MSE between current and frozen adapter outputs
        """
        if self.frozen_snapshot is None:
            return torch.tensor(0.0, device=replay_features.device)
        
        # Current adapter output
        delta_current = self.forward(replay_features, return_delta=True)
        
        # Frozen adapter output (no grad)
        delta_frozen = self.forward_frozen(replay_features)
        
        # L2 distance in function space
        anchor_loss = F.mse_loss(delta_current, delta_frozen)
        
        return anchor_loss


# Backward-compatible wrapper for easy drop-in replacement
class Adapter(FSAAdapter):
    """
    Drop-in replacement for original Adapter class.
    Inherits all FSA capabilities.
    """
    pass
