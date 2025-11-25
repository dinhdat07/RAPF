import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGate(nn.Module):
    def __init__(self, input_dim=512, num_experts=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # LayerNorm for input features
        self.norm = nn.LayerNorm(input_dim)

        # MLP for gating
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

        # Initialize last layer small to start softmax nearly uniform
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        x = self.norm(x)

        if self.mlp[-1].out_features != self.num_experts:
            self._expand_experts(self.num_experts)

        logits = self.mlp(x)
        return logits

    def _expand_experts(self, new_num_experts):
        old_layer = self.mlp[-1]
        old_w, old_b = old_layer.weight.data, old_layer.bias.data

        new_layer = nn.Linear(self.hidden_dim, new_num_experts).to(old_w.device).to(old_w.dtype)

        # copy old weights
        new_layer.weight.data[:old_w.shape[0]] = old_w
        new_layer.bias.data[:old_b.shape[0]] = old_b

        # initialize new rows (new experts) small
        if new_num_experts > old_w.shape[0]:
            nn.init.normal_(new_layer.weight.data[old_w.shape[0]:], mean=0.0, std=0.01)
            nn.init.constant_(new_layer.bias.data[old_w.shape[0]:], 0.0)

        self.mlp[-1] = new_layer
        self.num_experts = new_num_experts

    def set_num_experts(self, n):
        if n > self.num_experts:
            self._expand_experts(n)
