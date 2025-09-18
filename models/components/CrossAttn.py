import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class CrossAttnLayer(nn.Module):
    def __init__(self, embed_dim=1024, dim_feedforward=2048, num_heads=8, activation=F.gelu ,drop_out_rate=0.):
        super(CrossAttnLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = activation
        self.dropout1 = nn.Dropout(drop_out_rate)
        self.dropout2 = nn.Dropout(drop_out_rate)
        self.dropout3 = nn.Dropout(drop_out_rate)

    def forward(self, x, y, return_attn_weights, prefer_flash):
        if prefer_flash:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                attn_output, attn_weights = self.attn(x, y, y, need_weights=return_attn_weights)
        else:
            attn_output, attn_weights = self.attn(x, y, y, need_weights=return_attn_weights)

        x = self.norm1(x + self.dropout1(attn_output))
        x = self.norm2(x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x))))))
        return (x, attn_weights) if return_attn_weights else (x, None)


class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim=1024, dim_feedforward=2048, num_heads=8, num_layers=3, activation=F.gelu ,drop_out_rate=0.):
        super(CrossAttnBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.activation = activation
        self.drop_out_rate = drop_out_rate
        
        self.layers = nn.ModuleList([
            CrossAttnLayer(embed_dim, dim_feedforward, num_heads, activation, drop_out_rate)
            for i in range(num_layers)
        ])
    
    def forward(self, x, y, return_attn_weights: bool = False, prefer_flash: bool = True):
        last_attn = None
        for layer in self.layers:
            x, attn_weights = layer(x, y, return_attn_weights=return_attn_weights, prefer_flash=prefer_flash)
            if return_attn_weights:
                last_attn = attn_weights
        
        if return_attn_weights:
            return x, last_attn
        else :
            return x