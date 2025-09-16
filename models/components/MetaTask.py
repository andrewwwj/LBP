import math
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from .CrossAttn import CrossAttnBlock
from .ResNet import FilmResNet
from .MlpResNet import FilmMLP, MlpResNet

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

class FourierPositionalEncoding1D(nn.Module):
    def __init__(self, num_bands: int, out_dim: int, base: float = 2.0, include_input: bool = False):
        super().__init__()
        self.include_input = include_input
        freqs = base ** torch.arange(num_bands)  # [num_bands]
        self.register_buffer('freqs', freqs.float(), persistent=False)
        in_dim = (2 * num_bands) + (1 if include_input else 0)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, T] in [0,1]
        x = positions.unsqueeze(-1) * self.freqs  # [B, T, num_bands]
        feats = torch.cat([torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x)], dim=-1)  # [B, T, 2*num_bands]
        if self.include_input:
            feats = torch.cat([positions.unsqueeze(-1), feats], dim=-1)
        return self.proj(feats)  # [B, T, out_dim]


class IKContextExtractor(nn.Module):

    def __init__(self, proprio_dim, vl_dim, latent_dim, action_dim, num_latents=128,
                 num_heads=8, num_p_tokens=4, num_v_tokens=8,):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.action_drop_prob = 0.5
        self.latent_query = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        self.output_query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.num_v_tokens = num_v_tokens
        self.num_p_tokens = num_p_tokens

        # Loss hyperparameters
        self.tau = 0.07
        self.w_nce = 1.0
        self.w_align = 1.0
        self.vision_encoder = FilmMLP(input_dim=vl_dim, cond_dim=vl_dim, output_size=latent_dim)
        self.proprio_encoder = FilmMLP(input_dim=proprio_dim, cond_dim=vl_dim, output_size=latent_dim)
        self.action_encoder = MlpResNet(num_blocks=3, input_dim=action_dim, hidden_dim=latent_dim, output_size=latent_dim)

        # IK Encoders
        self.vision_IK = FilmMLP(input_dim=latent_dim, cond_dim=latent_dim, output_size=latent_dim)
        self.proprio_IK = FilmMLP(input_dim=latent_dim, cond_dim=latent_dim, output_size=latent_dim)

        # Tokenizers
        self.vision_tokenizer = nn.Linear(latent_dim, num_v_tokens * latent_dim)
        self.proprio_tokenizer = nn.Linear(latent_dim, num_p_tokens * latent_dim)

        self.ff_pos_p = FourierPositionalEncoding1D(num_bands=6, out_dim=latent_dim)
        self.ff_pos_v = FourierPositionalEncoding1D(num_bands=6, out_dim=latent_dim)

        self.register_buffer('p_token_pos', torch.linspace(0, 1, steps=num_p_tokens).unsqueeze(0), persistent=False)  # [1, Np]
        self.register_buffer('v_token_pos', torch.linspace(0, 1, steps=num_v_tokens).unsqueeze(0), persistent=False)  # [1, Nv]
        self.mod_embed = nn.Parameter(torch.randn(2, latent_dim))

        self.cross_attn = CrossAttnBlock(embed_dim=latent_dim,
                                         dim_feedforward=latent_dim * 2,
                                         num_layers=1)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=latent_dim,
        #     nhead=num_heads,
        #     dim_feedforward=latent_dim * 2,
        #     batch_first=True
        # )
        # self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 4),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 4),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        self.out_cross = CrossAttnBlock(embed_dim=latent_dim,
                                        dim_feedforward=latent_dim * 2,
                                        num_layers=1)
        self.apply(init_weight)

    def forward(self, img_history, p_history, p_next, lang_emb, prev_action, curr_action=None):
        B = p_history.shape[0]

        p_prev = p_history[:, 0]    # [B, P]
        p_curr = p_history[:, -1]   # [B, P]
        v_prev = img_history[:, 0]        # [B, Dv]
        v_curr = img_history[:, -1]       # [B, Dv]

        # Encode with FiLM-MLP conditioned on language
        pl_prev = self.proprio_encoder(p_prev, lang_emb)   # [B, H]
        pl_curr = self.proprio_encoder(p_curr, lang_emb)   # [B, H]
        vl_prev = self.vision_encoder(v_prev, lang_emb)    # [B, H]
        vl_curr = self.vision_encoder(v_curr, lang_emb)    # [B, H]

        delta_pl = pl_curr - pl_prev                       # [B, H]
        delta_vl = vl_curr - vl_prev                       # [B, H]

        a_prev = self.action_encoder(prev_action)  # [B, D]
        # TODO zero 대신 null embedding 입력?
        if curr_action is None:
            curr_action = torch.zeros(B, self.action_dim, device=p_curr.device, dtype=p_curr.dtype)
        if self.training and self.action_drop_prob > 0 and torch.rand(()) < self.action_drop_prob:
            curr_action = torch.zeros_like(curr_action)
        # else:
        #     if self.action_noise:
        #         random_noise = torch.randn_like(curr_action) * 0.01
        #         curr_action = curr_action + random_noise
        a_curr = self.action_encoder(curr_action)  # [B, D]

        # Apply FiLM IK on deltas conditioned on action
        pl_ik = self.proprio_IK(delta_pl, a_prev) # [B, H]
        vl_ik = self.vision_IK(delta_vl, a_prev) # [B, H]

        pl_tokens = self.proprio_tokenizer(pl_ik).view(B, self.num_p_tokens, -1)  # (B, N, D)
        vl_tokens = self.vision_tokenizer(vl_ik).view(B, self.num_v_tokens, -1)  # (B, M, D)

        p_pos = self.ff_pos_p(self.p_token_pos.expand(B, -1))  # [B, Np, H]
        v_pos = self.ff_pos_v(self.v_token_pos.expand(B, -1))  # [B, Nv, H]
        pl_tokens = pl_tokens + p_pos + self.mod_embed[0]
        vl_tokens = vl_tokens + v_pos + self.mod_embed[1]

        # Perceiver-IO: Q = learned latent array; KV = [pl_tokens, vl_tokens]
        # TODO q_latents 와 q_out 의 의미?
        q_latents = repeat(self.latent_query, '1 n d -> b n d', b=B)  # [B, L, H]
        kv_tokens = torch.cat([pl_tokens, vl_tokens], dim=1)  # [B, N+M, H]

        # latents = self.self_attn(self.cross_attn(q_latents, kv_tokens))  # [B, L, H]
        latents = self.cross_attn(q_latents, kv_tokens)
        q_out = repeat(self.output_query, '1 p d -> b p d', b=B)     # [B, 1, H]
        latent_ik = self.out_cross(q_out, latents).squeeze(1)  # [B, H]

        # Predict next Δproprio
        # TODO Predict latent proprio
        rollout = torch.cat([latent_ik, delta_pl, a_curr], dim=-1)
        next_delta_p = self.delta_head(rollout)

        # ---------------- Representation losses ------------------
        loss_dict = {}
        # 1-1) InfoNCE between token sets
        # loss_nce = self._info_nce_set2set(pl_tokens, vl_tokens, tau=self.tau)
        # loss_nce = self._info_nce(pl_tokens.mean(1), vl_tokens.mean(1), tau=self.tau)
        # loss_dict['loss_nce'] = loss_nce
        # 1-2) SimCLR-style contrastive loss on pooled tokens (replaces InfoNCE)
        loss_simclr = self._simclr_loss(pl_tokens, vl_tokens, tau=self.tau, pool='mean')
        loss_dict['loss_simclr'] = loss_simclr

        # 2) Latent alignment: cosine-based (scale-invariant)
        loss_align = self._cosine_align(pl_ik, vl_ik)
        loss_dict['loss_align'] = loss_align

        # TODO proprio 도 latent 에 투영 비교
        pl_next = self.proprio_encoder(p_next, lang_emb).detach()
        delta_gt = pl_next - pl_curr
        loss_delta = F.mse_loss(next_delta_p, delta_gt)
        loss_dict['loss_delta_p'] = loss_delta

        # 3) Δvision reconstruction tethering loss
        # loss_delta_v = F.mse_loss(delta_v_pred, delta_vl)
        # losses.append(self.w_vdelta * loss_delta_v)

        # total_loss = sum(losses)

        return next_delta_p, loss_dict

    def _simclr_loss(self, p_tokens, v_tokens, tau: float = 0.07, pool: str = 'mean'):
        """
        SimCLR-style contrastive loss between proprio and vision token sets.
        - Pools tokens per sample to obtain a single embedding per modality.
        - Builds a 2B x 2B similarity matrix; for each anchor, the positive is
          its paired sample from the other modality.
        Args:
            p_tokens: [B, K, D]
            v_tokens: [B, M, D]
            tau: temperature
            pool: 'mean' or 'max'
        Returns:
            scalar loss
        """
        B = p_tokens.shape[0]
        if pool == 'mean':
            p = p_tokens.mean(dim=1)  # [B, D]
            v = v_tokens.mean(dim=1)  # [B, D]
        elif pool == 'max':
            p = p_tokens.max(dim=1).values
            v = v_tokens.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pool type: {pool}")

        # Normalize embeddings
        p = F.normalize(p, dim=-1)
        v = F.normalize(v, dim=-1)

        # Concatenate views to form 2B embeddings
        z = torch.cat([p, v], dim=0)  # [2B, D]
        logits = z @ z.t() / tau      # [2B, 2B]

        # Mask self-similarity on the diagonal
        diag_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(diag_mask, float('-inf'))

        # Positive indices: i -> i+B for i in [0..B-1], and i -> i-B for i in [B..2B-1]
        idx = torch.arange(B, device=z.device)
        pos = torch.cat([idx + B, idx], dim=0)  # [2B]

        loss = F.cross_entropy(logits, pos)
        return loss

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

    def _info_nce_set2set(self, a, b, tau=0.07):
        """
        Set-to-set InfoNCE using max-over-tokens aggregator.
        p_tokens: [B, K, D], v_tokens: [B, M, D]
        logits[i,j] = 0.5 * ( mean_k max_m cos(p_i,k, v_j,m) + mean_m max_k cos(v_j,m, p_i,k) ) / tau
        """
        a = F.normalize(a, dim=-1)  # [B,K,D]
        b = F.normalize(b, dim=-1)  # [B,M,D]
        # sim[i,j,k,m] = p[i,k] dot v[j,m]
        sim = torch.einsum('ikd, jmd -> ijkm', a, b)  # [B,B,K,M]
        s1 = sim.max(dim=3).values.mean(dim=2)     # [B,B] mean over k of max over m
        s2 = sim.max(dim=2).values.mean(dim=2)     # [B,B] mean over m of max over k
        s = 0.5 * (s1 + s2)
        logits = s / tau
        B = a.shape[0]
        labels = torch.arange(B, device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))