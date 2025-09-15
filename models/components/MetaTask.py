from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from timm.layers import Mlp
from .CrossAttn import CrossAttnBlock
from .ResNet import FilmResNet
from .MlpResNet import FilmMLP

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class FiLM_layer(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int,):
        super().__init__()
        self.cond_proj = Mlp(cond_dim, input_dim * 2, input_dim * 2)
        self.apply(init_weight)
        nn.init.zeros_(self.cond_proj.fc2.weight)
        nn.init.zeros_(self.cond_proj.fc2.bias)

    def forward(self, x, cond):
        # x: (B, ..., D), cond: (B, C)
        gammas, betas = self.cond_proj(cond).chunk(2, dim=-1)
        # Reshape gammas and betas to match the dimensions of x for broadcasting
        while len(gammas.shape) < len(x.shape):
            gammas = gammas.unsqueeze(1)
            betas = betas.unsqueeze(1)

        return x * (gammas + 1) + betas


class IKContextExtractor(nn.Module):
    """
    Learns a disentangled kinematic/spatial context from vision-language, proprioceptive observations, and actions.
    It aims to separate the 'what' (visual semantics) from the 'how' and 'where' (kinematics and spatial info).
    """

    def __init__(self, proprio_dim, vision_dim, action_dim, hidden_dim=512, num_latents=128,
                 num_layers=4, num_heads=8, num_a_tokens = 4, num_p_tokens=8, num_v_tokens=8,):
        super().__init__()
        self.latent_array = nn.Parameter(torch.randn(1, num_latents, hidden_dim))
        self.output_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.num_a_tokens = num_a_tokens
        self.num_p_tokens = num_p_tokens
        self.num_v_tokens = num_v_tokens
        # Loss hyperparameters
        self.tau = 0.07
        self.w_nce = 1.0
        self.w_align = 1.0
        latent_dim = vision_dim
        self.vision_encoder = FilmMLP(input_dim=vision_dim, cond_dim=latent_dim, output_size=latent_dim)
        self.proprio_encoder = FilmMLP(input_dim=proprio_dim, cond_dim=latent_dim, output_size=latent_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

        # Kinematic IK Encoder
        self.proprio_IK = FilmMLP(input_dim=latent_dim, cond_dim=latent_dim, output_size=latent_dim)
        # Vision IK encoder
        self.vision_IK = FilmMLP(input_dim=latent_dim, cond_dim=latent_dim, output_size=latent_dim)

        # Shared manifold projector g(·) applied to both modalities
        self.manifold_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

        # produce tokens
        self.action_tokenizer = nn.Linear(latent_dim, latent_dim * num_a_tokens)
        self.vision_tokenizer = nn.Linear(latent_dim, latent_dim * num_v_tokens)
        self.proprio_tokenizer = nn.Linear(latent_dim, latent_dim * num_p_tokens)

        self.cross_attn = CrossAttnBlock(embed_dim=latent_dim, num_layers=3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True
        )
        self.self_attn_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, proprio_dim)
        )

        # Δvision reconstruction head to tether manifold to visual motion
        self.delta_v_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        self.apply(init_weight)

    def forward(self, img_history, proprio_history, next_proprio, lang_emb, prev_action):
        B = proprio_history.shape[0]

        p_prev = proprio_history[:, 0]    # [B, P]
        p_curr = proprio_history[:, -1]   # [B, P]
        v_prev = img_history[:, 0]        # [B, Dv]
        v_curr = img_history[:, -1]       # [B, Dv]

        # Encode with FiLM-MLP conditioned on language
        pl_prev = self.proprio_encoder(p_prev, lang_emb)   # [B, H]
        pl_curr = self.proprio_encoder(p_curr, lang_emb)   # [B, H]
        vl_prev = self.vision_encoder(v_prev, lang_emb)    # [B, H]
        vl_curr = self.vision_encoder(v_curr, lang_emb)    # [B, H]

        delta_pl = pl_curr - pl_prev                       # [B, H]
        delta_vl = vl_curr - vl_prev                       # [B, H]

        a_emb = self.action_encoder(prev_action)  # [B, D]

        # Apply FiLM IK on deltas conditioned on action
        pl_ik = self.proprio_IK(delta_pl, a_emb) # [B, H]
        vl_ik = self.vision_IK(delta_vl, a_emb) # [B, H]

        # Project into shared manifold and tokenize for Perceiver-style fusion
        pl_ik = self.manifold_proj(pl_ik)
        vl_ik = self.manifold_proj(vl_ik)

        pl_tokens = self.proprio_tokenizer(pl_ik).view(B, self.num_p_tokens, -1)  # (B, N, D)
        vl_tokens = self.vision_tokenizer(vl_ik).view(B, self.num_v_tokens, -1)  # (B, M, D)
        a_tokens = self.action_tokenizer(a_emb).view(B, self.num_a_tokens, -1)

        # Shared IK: Q = action tokens; KV = [pl_tokens, vl_tokens]
        kv_tokens = torch.cat([pl_tokens, vl_tokens], dim=1)  # [B, K+M, H]
        latents = self.cross_attn(a_tokens, kv_tokens)  # [B, Q, H]
        latents = self.self_attn_blocks(latents)  # [B, Q, H]
        fused = latents.mean(dim=1)  # [B, H]

        # Predict next Δproprio and reconstruct Δvision latent
        next_delta_p = self.delta_head(fused)
        # delta_v_pred = self.delta_v_head(fused)

        # -----------------
        # Representation losses
        # -----------------
        loss_dict = {}
        # 1) InfoNCE between manifold-projected token sets
        loss_nce = self._info_nce_set2set(pl_tokens, vl_tokens, tau=self.tau)
        loss_dict['loss_nce'] = loss_nce

        # 2) Latent alignment: cosine-based (scale-invariant)
        loss_align = self._cosine_align(pl_ik.mean(dim=1), vl_ik.mean(dim=1))
        loss_dict['loss_align'] = loss_align

        delta_gt = next_proprio - p_curr
        loss_delta = F.mse_loss(next_delta_p, delta_gt)
        loss_dict['loss_delta_p'] = loss_delta

        # 3) Δvision reconstruction tethering loss
        # loss_delta_v = F.mse_loss(delta_v_pred, delta_vl)
        # losses.append(self.w_vdelta * loss_delta_v)

        # total_loss = sum(losses)

        return next_delta_p, loss_dict

    def _info_nce(self, a, b, tau=0.07):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.t() / tau
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def _cosine_align(self, a, b):
        """Cosine-based alignment loss: 1 - cosine(a, b), averaged over batch."""
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return 1.0 - (a_n * b_n).sum(dim=-1).mean()

    def _info_nce_set2set(self, p_tokens, v_tokens, tau=0.07):
        """
        Set-to-set InfoNCE using max-over-tokens aggregator.
        p_tokens: [B, K, D], v_tokens: [B, M, D]
        logits[i,j] = 0.5 * ( mean_k max_m cos(p_i,k, v_j,m) + mean_m max_k cos(v_j,m, p_i,k) ) / tau
        """
        p = F.normalize(p_tokens, dim=-1)  # [B,K,D]
        v = F.normalize(v_tokens, dim=-1)  # [B,M,D]
        # sim[i,j,k,m] = p[i,k] dot v[j,m]
        sim = torch.einsum('ikd, jmd -> ijkm', p, v)  # [B,B,K,M]
        s1 = sim.max(dim=3).values.mean(dim=2)     # [B,B] mean over k of max over m
        s2 = sim.max(dim=2).values.mean(dim=2)     # [B,B] mean over m of max over k
        s = 0.5 * (s1 + s2)
        logits = s / tau
        B = p.shape[0]
        labels = torch.arange(B, device=p.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))