"""
Quick patch file to add FSA to your existing code.

Usage:
1. Save this as: continual_clip/fsa_patch.py
2. In train.py, add: from continual_clip.fsa_patch import patch_model_with_fsa
3. After model creation, call: patch_model_with_fsa(model, cfg)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_model_with_fsa(model, cfg):
    """
    Monkey-patch existing model to add FSA capability.
    
    Args:
        model: ClassIncrementalCLIP instance
        cfg: Config with FSA parameters
    
    This function:
    1. Adds FSA-specific attributes
    2. Wraps existing Adapter modules with FSA methods
    3. Injects FSA loss computation
    """
    
    # Add FSA config
    model.lambda_anchor = float(getattr(cfg, 'lambda_anchor', 0.5))
    model.use_fsa = getattr(cfg, 'use_fsa', True)
    model.fsa_replay_size = int(getattr(cfg, 'fsa_replay_size', 100))
    
    print(f"[FSA Patch] Applied with lambda_anchor={model.lambda_anchor}")
    
    # Wrap adapters with FSA capability
    for i, adapter in enumerate(model.image_injection):
        _wrap_adapter_with_fsa(adapter)
    
    # Add FSA methods to model
    model.compute_fsa_loss = lambda batch_size=64: _compute_fsa_loss(model, batch_size)
    model._sample_fsa_replay = lambda batch_size: _sample_fsa_replay(model, batch_size)
    model._create_adapter_snapshot = lambda: _create_adapter_snapshot(model)
    
    # Hook into update_injection_units
    original_update = model.update_injection_units
    
    def patched_update_injection_units(noise_std=0.01):
        # Create snapshot before updating
        if model.use_fsa and len(model.image_injection) > 0:
            model._create_adapter_snapshot()
        
        # Call original
        original_update(noise_std)
        
        # Wrap new adapter
        if len(model.image_injection) > 0:
            _wrap_adapter_with_fsa(model.image_injection[-1])
    
    model.update_injection_units = patched_update_injection_units
    
    return model


def _wrap_adapter_with_fsa(adapter):
    """Add FSA methods to existing Adapter instance."""
    
    # Add frozen snapshot storage
    if not hasattr(adapter, 'frozen_snapshot'):
        adapter.frozen_snapshot = None
    
    # Add forward_delta method
    def forward_delta(self, x):
        """Return only Δ(x), not residual."""
        residual = x
        if self.use_layernorm:
            x = self.layernorm(x)
        x = self.down_proj(x)
        x = self.ln_hidden(x)
        x = self.non_linear(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x * self.scale
    
    adapter.forward_delta = lambda x: forward_delta(adapter, x)
    
    # Add forward_frozen method
    def forward_frozen(self, x):
        """Forward through frozen snapshot."""
        if self.frozen_snapshot is None:
            return torch.zeros_like(x)
        with torch.no_grad():
            return self.frozen_snapshot.forward_delta(x)
    
    adapter.forward_frozen = lambda x: forward_frozen(adapter, x)
    
    # Add create_snapshot method
    def create_snapshot(self):
        """Create frozen copy."""
        snapshot = copy.deepcopy(self)
        for param in snapshot.parameters():
            param.requires_grad = False
        snapshot.eval()
        self.frozen_snapshot = snapshot
        return snapshot
    
    adapter.create_snapshot = lambda: create_snapshot(adapter)
    
    # Add get_anchor_loss method
    def get_anchor_loss(self, replay_features):
        """Compute FSA loss."""
        if self.frozen_snapshot is None:
            return torch.tensor(0.0, device=replay_features.device)
        
        delta_current = self.forward_delta(replay_features)
        delta_frozen = self.forward_frozen(replay_features)
        
        return F.mse_loss(delta_current, delta_frozen)
    
    adapter.get_anchor_loss = lambda replay_features: get_anchor_loss(adapter, replay_features)


def _create_adapter_snapshot(model):
    """Create snapshot of last adapter."""
    if len(model.image_injection) == 0:
        return
    
    last_adapter = model.image_injection[-1]
    if hasattr(last_adapter, 'create_snapshot'):
        last_adapter.create_snapshot()
        print(f"[FSA] Created snapshot for adapter {len(model.image_injection)-1}")


def _compute_fsa_loss(model, batch_size):
    """Compute FSA loss on replay buffer."""
    if not model.use_fsa or len(model.image_injection) <= 1:
        return torch.tensor(0.0, device=model.device)
    
    current_adapter = model.image_injection[-1]
    
    replay_features = model._sample_fsa_replay(batch_size)
    if replay_features is None or replay_features.shape[0] == 0:
        return torch.tensor(0.0, device=model.device)
    
    return current_adapter.get_anchor_loss(replay_features)


def _sample_fsa_replay(model, batch_size):
    """Sample replay features from prototypes."""
    if not hasattr(model, 'prototype') or not model.prototype or len(model.prototype) == 0:
        return None
    
    replay_features = []
    n_old_classes = min(batch_size // 2, len(model.prototype))
    
    if n_old_classes == 0:
        return None
    
    # Sample uniformly from old classes
    sampled_indices = torch.randperm(len(model.prototype))[:n_old_classes]
    
    for class_id in sampled_indices:
        class_id = class_id.item()
        
        # Add prototype
        proto = model.prototype[class_id].to(model.device).clone()
        replay_features.append(proto.unsqueeze(0))
        
        # Optionally add Gaussian samples
        if hasattr(model, 'class_mean_list') and class_id < len(model.class_mean_list):
            mean = model.class_mean_list[class_id]
            cov = model.class_cov_list[class_id]
            
            # Sample 1-2 features per class
            samples = _sample_gaussian(mean, cov, size=2, device=model.device)
            replay_features.append(samples)
    
    if len(replay_features) == 0:
        return None
    
    replay_features = torch.cat(replay_features, dim=0)
    return replay_features[:batch_size]


def _sample_gaussian(mean, cov, size, device):
    """Sample from Gaussian with numerical stability."""
    vec = torch.randn(size, mean.shape[-1], device=device)
    
    # Regularize covariance
    cov_reg = cov + 1e-4 * torch.eye(cov.shape[0], device=device)
    
    try:
        sqrt_cov = torch.linalg.cholesky(cov_reg)
        vec = vec @ sqrt_cov.t()
    except RuntimeError:
        # Fallback to eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_reg)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        sqrt_cov = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
        vec = vec @ sqrt_cov.t()
    
    return vec + mean


# ========================================
# Training loop helper
# ========================================

def add_fsa_to_training_step(loss, model, task_id, device):
    """
    Convenience function to add FSA loss in training loop.
    
    Usage:
        loss = clip_loss + other_losses
        loss = add_fsa_to_training_step(loss, model, task_id, device)
        loss.backward()
    
    Args:
        loss: Current total loss
        model: Model with FSA patch
        task_id: Current task ID
        device: cuda/cpu
    
    Returns:
        Updated loss with FSA term
    """
    if task_id > 0 and hasattr(model, 'use_fsa') and model.use_fsa:
        fsa_loss = model.compute_fsa_loss(batch_size=64)
        loss = loss + model.lambda_anchor * fsa_loss
    
    return loss


# ========================================
# Example usage
# ========================================

if __name__ == "__main__":
    """
    Example: How to use the patch in your training script.
    """
    
    # Original training code:
    # model = ClassIncrementalCLIP(cfg, device)
    # model.update_injection_units()
    
    # NEW: Just add these 2 lines after model creation:
    from continual_clip.fsa_patch import patch_model_with_fsa, add_fsa_to_training_step
    
    # model = patch_model_with_fsa(model, cfg)
    
    # Then in training loop, change:
    # loss.backward()
    
    # To:
    # loss = add_fsa_to_training_step(loss, model, task_id, device)
    # loss.backward()
    
    print("See code comments for usage examples")
