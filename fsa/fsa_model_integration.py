import copy
import torch
import torch.nn as nn
from typing import List


class FSAClassIncrementalCLIP(nn.Module):
    """
    Enhanced ClassIncrementalCLIP with Function-Space Anchoring.
    
    Key changes from original:
    1. Adapters use FSAAdapter with snapshot capability
    2. Added compute_fsa_loss() for anchor regularization
    3. Modified update_injection_units() to create snapshots
    """
    
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        # ... [Keep all original initialization] ...
        
        # FSA-specific configurations
        self.lambda_anchor = float(getattr(cfg, 'lambda_anchor', 0.5))  # FSA loss weight
        self.fsa_replay_size = int(getattr(cfg, 'fsa_replay_size', 100))  # Features per class
        self.use_fsa = getattr(cfg, 'use_fsa', True)  # Enable/disable FSA
        
        # Store replay features for FSA (lightweight, only features not images)
        self.fsa_replay_buffer = []  # List of [features, class_id] tuples
        
    def update_injection_units(self, noise_std=0.01):
        """
        Enhanced version that creates frozen snapshots for FSA.
        """
        # Freeze previous adapters
        for inj in self.image_injection: 
            self.freeze(inj)
        
        # Create snapshot of last adapter for FSA
        if self.use_fsa and len(self.image_injection) > 0:
            last_adapter = self.image_injection[-1]
            if hasattr(last_adapter, 'create_snapshot'):
                last_adapter.create_snapshot()
                print(f"[FSA] Created snapshot for adapter {len(self.image_injection)-1}")

        dropout_rate = float(getattr(self.cfg, 'dropout', 0.1))

        if self.image_injection:
            # Copy from previous adapter
            new_image_adapter = copy.deepcopy(self.image_injection[-1])
            
            # Add small noise for better initialization
            for param in new_image_adapter.parameters():
                param.data += noise_std * torch.randn_like(param.data)
                param.requires_grad = True 
            
            # Reset scale
            if hasattr(new_image_adapter, 'scale'):
                with torch.no_grad():
                    new_image_adapter.scale.fill_(1.0)
            
            # Clear frozen snapshot from copied adapter
            if hasattr(new_image_adapter, 'frozen_snapshot'):
                new_image_adapter.frozen_snapshot = None
            
            self.image_injection.append(
                new_image_adapter.to(self.device).to(dtype=self.dtype)
            )
        else:
            # First task: use FSAAdapter instead of original Adapter
            from .fsa_adapter import FSAAdapter
            self.image_injection.append(
                FSAAdapter(512, 256, dropout=dropout_rate)
                .to(self.device).to(dtype=self.dtype)
            )

    def compute_fsa_loss(self, batch_size=32):
        """
        Compute Function-Space Anchoring loss on replay buffer.
        
        Args:
            batch_size: Number of replay features to sample per iteration
        
        Returns:
            fsa_loss: Anchor loss from current adapter
        """
        if not self.use_fsa or len(self.image_injection) <= 1:
            return torch.tensor(0.0, device=self.device)
        
        current_adapter = self.image_injection[-1]
        
        # Sample from replay buffer
        replay_features = self._sample_fsa_replay(batch_size)
        if replay_features is None or replay_features.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Compute anchor loss
        if hasattr(current_adapter, 'get_anchor_loss'):
            fsa_loss = current_adapter.get_anchor_loss(replay_features)
        else:
            fsa_loss = torch.tensor(0.0, device=self.device)
        
        return fsa_loss

    def _sample_fsa_replay(self, batch_size):
        """
        Sample features from FSA replay buffer.
        
        Strategy: Sample prototypes + Gaussian perturbations
        """
        if not self.prototype or len(self.prototype) == 0:
            return None
        
        # Number of old classes
        n_old_classes = len(self.prototype)
        if n_old_classes == 0:
            return None
        
        # Sample classes uniformly
        num_classes_to_sample = min(batch_size // 2, n_old_classes)
        sampled_class_ids = torch.randperm(n_old_classes)[:num_classes_to_sample]
        
        replay_features = []
        
        for class_id in sampled_class_ids:
            if class_id >= len(self.class_mean_list):
                continue
            
            # Add prototype
            proto = self.prototype[class_id].to(self.device).clone()
            replay_features.append(proto.unsqueeze(0))
            
            # Add Gaussian samples around mean
            mean = self.class_mean_list[class_id]
            cov = self.class_cov_list[class_id]
            
            # Sample 1-2 features per class
            samples_per_class = max(1, batch_size // num_classes_to_sample)
            sampled = self._sample_from_gaussian(mean, cov, samples_per_class)
            replay_features.append(sampled)
        
        if len(replay_features) == 0:
            return None
        
        replay_features = torch.cat(replay_features, dim=0)
        
        # Apply current adapter's preprocessing (LayerNorm if needed)
        # Important: Don't pass through full forward, just get the features
        return replay_features[:batch_size]

    def _sample_from_gaussian(self, mean, cov, size):
        """
        Sample from Gaussian distribution (reuse existing sample function).
        """
        vec = torch.randn(size, mean.shape[-1], device=mean.device)
        
        # Add regularization for numerical stability
        cov_reg = cov + 1e-4 * torch.eye(cov.shape[0], device=cov.device)
        
        try:
            sqrt_cov = torch.linalg.cholesky(cov_reg)
            vec = vec @ sqrt_cov.t()
        except RuntimeError:
            # Fallback: use eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_reg)
            eigenvalues = torch.clamp(eigenvalues, min=1e-6)
            sqrt_cov = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
            vec = vec @ sqrt_cov.t()
        
        vec = vec + mean
        return vec

    def get_trainable_parameters(self):
        """
        Return only trainable parameters (last adapter).
        Frozen snapshots are excluded automatically.
        """
        params = []
        if self.image_injection and len(self.image_injection) > 0:
            # Only parameters with requires_grad=True
            params.append(
                filter(lambda p: p.requires_grad, 
                       self.image_injection[-1].parameters())
            )
        return chain.from_iterable(params)


# Additional utility functions for FSA

def get_fsa_statistics(model):
    """
    Print FSA-related statistics for debugging.
    """
    print("\n" + "="*50)
    print("FSA Statistics")
    print("="*50)
    
    for i, adapter in enumerate(model.image_injection):
        has_snapshot = hasattr(adapter, 'frozen_snapshot') and adapter.frozen_snapshot is not None
        n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        
        print(f"Adapter {i}:")
        print(f"  - Trainable params: {n_params:,}")
        print(f"  - Has snapshot: {has_snapshot}")
        
        if has_snapshot:
            snapshot_params = sum(p.numel() for p in adapter.frozen_snapshot.parameters())
            print(f"  - Snapshot params: {snapshot_params:,}")
    
    print(f"\nTotal prototypes: {len(model.prototype)}")
    print(f"FSA lambda_anchor: {model.lambda_anchor}")
    print("="*50 + "\n")


def visualize_adapter_drift(model, features, task_id):
    """
    Compute and visualize adapter drift for analysis.
    
    Args:
        model: FSAClassIncrementalCLIP instance
        features: [N, D] tensor of features
        task_id: Current task ID
    
    Returns:
        drift_metrics: Dict with drift statistics
    """
    if len(model.image_injection) <= 1:
        return {"message": "Not enough tasks for drift analysis"}
    
    current_adapter = model.image_injection[-1]
    
    with torch.no_grad():
        # Current adapter output
        delta_current = current_adapter(features, return_delta=True)
        
        # Frozen adapter output
        delta_frozen = current_adapter.forward_frozen(features)
        
        # Compute drift metrics
        mse = F.mse_loss(delta_current, delta_frozen).item()
        cos_sim = F.cosine_similarity(
            delta_current.flatten(), 
            delta_frozen.flatten(), 
            dim=0
        ).item()
        
        l2_norm_current = delta_current.norm(dim=-1).mean().item()
        l2_norm_frozen = delta_frozen.norm(dim=-1).mean().item()
    
    drift_metrics = {
        "task_id": task_id,
        "mse_drift": mse,
        "cosine_similarity": cos_sim,
        "l2_norm_current": l2_norm_current,
        "l2_norm_frozen": l2_norm_frozen,
        "relative_change": abs(l2_norm_current - l2_norm_frozen) / (l2_norm_frozen + 1e-8)
    }
    
    return drift_metrics
