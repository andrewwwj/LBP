import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class FiLM_layer(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int):
        super().__init__()
        self.cond_proj = Mlp(cond_dim, hidden_features=input_dim * 2, out_features=input_dim * 2)
        self.apply(init_weight)
        nn.init.zeros_(self.cond_proj.fc2.weight)
        nn.init.zeros_(self.cond_proj.fc2.bias)

    def forward(self, x, cond):
        gammas, betas = self.cond_proj(cond).chunk(2, dim=-1)
        while len(gammas.shape) < len(x.shape):
            gammas = gammas.unsqueeze(1)
            betas = betas.unsqueeze(1)
        return x * (gammas + 1) + betas


class MlpResNetBlock(nn.Module):
    """
    the MLPResnet Blocks used in IDQL: arXiv:2304.10573, Appendix G
    """
    def __init__(self, hidden_dim:int, ac_fn=F.gelu, use_layernorm=False, dropout_rate=0.1):
        super(MlpResNetBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ac_fn = ac_fn
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x):
        out = self.dropout(x)
        out = self.norm1(out)
        out = self.dense1(out)
        out = self.ac_fn(out)
        out = self.dense2(out)
        out = x + out
        return out

class MlpResNet(nn.Module):
    """
    the LN_Resnet used in IDQL: arXiv:2304.10573
    """
    def __init__(self, num_blocks:int, input_dim:int, hidden_dim:int, output_size:int,
                 ac_fn=F.gelu, use_layernorm=True, dropout_rate=0.1):
        super(MlpResNet, self).__init__()

        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_size)
        self.ac_fn = ac_fn
        self.mlp_res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_res_blocks.append(MlpResNetBlock(hidden_dim, ac_fn, use_layernorm, dropout_rate))

    def forward(self, x):
        out = self.dense1(x)
        for mlp_res_block in self.mlp_res_blocks:
            out = mlp_res_block(out)
        out = self.ac_fn(out)
        out = self.dense2(out)
        return out


class FilmMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, ac_fn=F.gelu, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.film = FiLM_layer(input_dim=hidden_dim * 4, cond_dim=cond_dim)
        self.ac_fn = ac_fn
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x, cond):
        out = self.norm1(x)
        out = self.dense1(out)
        out = self.film(out, cond)
        out = self.ac_fn(out)
        out = self.dense2(out)
        out = self.dropout(out)
        out = x + out
        return out


class FilmMLP(nn.Module):
    def __init__(self, input_dim: int, output_size: int, cond_dim: int, hidden_dim: int=512, num_blocks: int=3,
                 ac_fn=F.gelu, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_size)
        self.ac_fn = ac_fn
        self.film_mlp_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.film_mlp_blocks.append(FilmMLPBlock(hidden_dim, cond_dim, ac_fn, dropout_rate))

    def forward(self, x, cond):
        out = self.dense1(x)
        for mlp_block in self.film_mlp_blocks:
            out = mlp_block(out, cond)
        out = self.ac_fn(out)
        out = self.dense2(out)
        return out