import math
import torch
import torch.nn as nn

class MLP_Adapter(nn.Module):
    def __init__(self, c_in, hidden):
        super(MLP_Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden),
        )

    def forward(self, x):
        x_ = self.fc(x)
        return x_
    
class ENGINE_Adapter(nn.Module):
    def __init__(self, c_in, hidden, dropout=0.1, use_layernorm=True, learnable_scale=True):
        super(ENGINE_Adapter, self).__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(c_in)

        # Down-projection
        self.down_proj = nn.Linear(c_in, hidden)
        
        self.ln_hidden = nn.LayerNorm(hidden)
        self.non_linear = nn.GELU()
        
        # Up-projection
        self.up_proj = nn.Linear(hidden, c_in)
        self.dropout = nn.Dropout(dropout)

        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.tensor(1.0))
            
        # init
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        if self.use_layernorm:
            x = self.layernorm(x)
            
        x = self.down_proj(x)
        x = self.ln_hidden(x)
        x = self.non_linear(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        x = x * self.scale
        return residual + x