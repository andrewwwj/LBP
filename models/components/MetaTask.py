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

    def __init__(self, proprio_dim, vision_dim, action_dim, hidden_dim=512, out_dim=512, num_latents=128,
                 num_layers=4, num_heads=8, num_p_tokens=4, num_v_tokens=8,):
        super().__init__()
        self.latent_array = nn.Parameter(torch.randn(1, num_latents, hidden_dim))
        self.output_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.num_p_tokens = num_p_tokens
        self.num_v_tokens = num_v_tokens
        # Loss hyperparameters
        self.tau = 0.07
        self.w_nce = 1.0
        self.w_align = 1.0
        self.w_token_nce = 0.5
        latent_dim = vision_dim
        # self.vision_encoder = FilmResNet(image_dim=3, cond_dim=vision_dim, backbone_name=vis_backbone)
        self.vision_encoder = FilmMLP(input_dim=vision_dim, cond_dim=latent_dim, output_size=latent_dim)
        self.proprio_encoder = FilmMLP(input_dim=proprio_dim, cond_dim=latent_dim, output_size=latent_dim, num_blocks=5)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 1. Kinematic Encoder: (p_t, p_{t-1}, a_{t-1}) -> c_kin
        self.proprio_film = FiLM_layer(input_dim=latent_dim, cond_dim=hidden_dim)
        self.proprio_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
        )

        # 2. Motion Encoder: (v_t, v_{t-1}, a_{t-1}) -> c_vl
        self.vision_film = FiLM_layer(input_dim=latent_dim, cond_dim=hidden_dim)
        self.vision_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
        )

        # produce K and M tokens of dimension hidden_dim
        self.vision_tokenizer = nn.Linear(hidden_dim, hidden_dim * num_v_tokens)
        self.proprio_tokenizer = nn.Linear(hidden_dim, hidden_dim * num_p_tokens)

        # 3. Latent query projector
        self.query_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cross_attn = CrossAttnBlock(embed_dim=hidden_dim, num_layers=3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.self_attn_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        self.apply(init_weight)

    def forward(self, img_history, proprio_history, lang_emb, prev_action):
        B = proprio_history.shape[0]

        pl_curr_emb = self.proprio_encoder(proprio_history[:, -1], lang_emb)  # [B, L]
        pl_prev_emb = self.proprio_encoder(proprio_history[:, 0], lang_emb)  # [B, L]
        delta_pl_emb = pl_curr_emb - pl_prev_emb  # [B, L]

        vl_curr_emb = self.vision_encoder(img_history[:, -1], lang_emb)  # [B*V, D]
        vl_prev_emb = self.vision_encoder(img_history[:, 0], lang_emb)  # [B*V, D]
        delta_vl_emb = vl_curr_emb - vl_prev_emb  # [B*V, D]

        #TODO img_diff 로부터 기하학적인 특징 추출? i.e., optical flow? SFM?
        # (ee view - world view) <=> IK from proprio
        action_emb = self.action_encoder(prev_action)  # [B, D]

        c_pl = self.proprio_proj(self.proprio_film(delta_pl_emb, action_emb))
        c_vl = self.vision_proj(self.vision_film(delta_vl_emb, action_emb))

        # Tokenize for Perceiver-style fusion
        p_tokens = self.proprio_tokenizer(c_pl).view(B, self.num_p_tokens, -1)  # (B, K, D)
        v_tokens = self.vision_tokenizer(c_vl).view(B, self.num_v_tokens, -1)  # (B, M, D)

        q_latent = self.latent_array.repeat(B, 1, 1)
        kv_tokens = torch.cat([p_tokens, v_tokens], dim=1)

        # 4. Initial fusion via cross-attention
        # `curr_vl` queries into the kinematic and vl context
        latents = self.cross_attn(q_latent, kv_tokens) # (B, 1, D)
        latents = self.self_attn_blocks(latents)

        # 5. Generate final context vector
        output_latent, _ = self.output_cross_attn(self.output_query.repeat(B, 1, 1), latents, latents)
        kinematic_context = self.output_proj(output_latent.squeeze(1))

        # -----------------
        # Representation losses
        # -----------------
        losses = []
        # 1) InfoNCE between kinematic (proprio/action) and visual-motion (vision/action)
        loss_nce = self._info_nce(c_pl, c_vl, tau=self.tau)
        losses.append(self.w_nce * loss_nce)
        # 2) Latent alignment: cosine-based (scale-invariant)
        loss_align = 0.5 * (self._cosine_align(kinematic_context, c_pl) + self._cosine_align(kinematic_context, c_vl))
        losses.append(self.w_align * loss_align)
        # 3) Token-level InfoNCE (set-to-set with max-over-tokens aggregator)
        loss_token_nce = self._info_nce_set2set(p_tokens, v_tokens, tau=self.tau)
        losses.append(self.w_token_nce * loss_token_nce)

        total_loss = sum(losses)

        return kinematic_context, vl_curr_emb, total_loss

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