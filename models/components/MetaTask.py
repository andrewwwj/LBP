import math
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from .CrossAttn import CrossAttnBlock
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


class LatentDynamics(nn.Module):

    def __init__(self, proprio_dim, vl_dim, hidden_dim, p_goal_dim, action_dim, num_latents=4, action_noise=True,
                 num_p_tokens=8, num_v_tokens=8, num_a_tokens=4):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.action_drop_prob = 0.1
        self.action_noise = action_noise
        self.latent_query = nn.Parameter(torch.randn(1, num_latents, hidden_dim))
        self.output_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.num_v_tokens = num_v_tokens
        self.num_p_tokens = num_p_tokens
        self.num_a_tokens = num_a_tokens

        # Loss hyperparameters
        self.tau = 0.07
        self.w_nce = 1.0
        self.w_align = 1.0
        # TODO cross-attention 활용
        self.proprio_encoder = FilmMLP(input_dim=proprio_dim, cond_dim=vl_dim, output_size=p_goal_dim)
        self.vision_encoder = FilmMLP(input_dim=vl_dim, cond_dim=vl_dim, output_size=p_goal_dim)  # input_dim=vl_dim * 2 if double-view
        self.action_encoder = MlpResNet(num_blocks=3, input_dim=action_dim, hidden_dim=hidden_dim, output_size=hidden_dim)

        # Tokenizers
        self.proprio_tokenizer = nn.Linear(p_goal_dim, num_p_tokens * hidden_dim)
        self.vision_tokenizer = nn.Linear(p_goal_dim, num_v_tokens * hidden_dim)
        # self.action_tokenizer = nn.Linear(hidden_dim, num_a_tokens * hidden_dim)
        self.p_token_ln = nn.LayerNorm(hidden_dim)
        self.v_token_ln = nn.LayerNorm(hidden_dim)
        self.a_token_ln = nn.LayerNorm(hidden_dim)

        self.token_dropout = nn.Dropout(0.1)

        self.ff_pos_p = FourierPositionalEncoding1D(num_bands=6, out_dim=hidden_dim)
        self.ff_pos_v = FourierPositionalEncoding1D(num_bands=6, out_dim=hidden_dim)
        self.ff_pos_a = FourierPositionalEncoding1D(num_bands=6, out_dim=hidden_dim)

        self.register_buffer('p_pos', torch.linspace(0, 1, steps=num_p_tokens).unsqueeze(0), persistent=False)  # [1, Np]
        self.register_buffer('v_pos', torch.linspace(0, 1, steps=num_v_tokens).unsqueeze(0), persistent=False)  # [1, Nv]
        self.register_buffer('a_pos', torch.linspace(0, 1, steps=num_a_tokens).unsqueeze(0), persistent=False)
        self.mod_embed = nn.Parameter(torch.randn(3, hidden_dim))

        # Projection head for contrastive branch (SimCLR-style)
        # self.proj_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        # ik solver
        # ---------------------- FiLM ik ----------------------
        # self.vision_ik = FilmMLP(input_dim=hidden_dim, cond_dim=hidden_dim, output_size=hidden_dim)
        # self.proprio_ik = FilmMLP(input_dim=hidden_dim, cond_dim=hidden_dim, output_size=hidden_dim)

        # ----------------- Cross-attention ik -----------------
        self.proprio_ik = CrossAttnBlock(embed_dim=hidden_dim,
                                         dim_feedforward=hidden_dim * 4,
                                         num_heads=4,
                                         num_layers=2,
                                         drop_out_rate=0.1)
        self.vision_ik = CrossAttnBlock(embed_dim=hidden_dim,
                                        dim_feedforward=hidden_dim * 4,
                                        num_heads=4,
                                        num_layers=2,
                                        drop_out_rate=0.1)

        self.latent_ik = CrossAttnBlock(embed_dim=hidden_dim,
                                         dim_feedforward=hidden_dim * 4,
                                         num_heads=4,
                                         num_layers=2,
                                         drop_out_rate=0.1)
        self.latent_emb = CrossAttnBlock(embed_dim=hidden_dim,
                                        dim_feedforward=hidden_dim * 2,
                                        num_heads=4,
                                        num_layers=1,
                                        drop_out_rate=0.1)

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim + p_goal_dim, hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, p_goal_dim),
        )
        self.apply(init_weight)

    def forward(self, img_history, p_history, subgoal, prev_action, p_next=None, v_next=None):
        B = p_history.shape[0]

        # if self.training:
        #     if torch.rand(()) < self.action_drop_prob:
        #         prev_action = torch.zeros_like(prev_action)
        #     elif self.action_noise:
        #         prev_action = prev_action + 0.01 * torch.randn_like(prev_action)

        p_prev = p_history[:, 0]    # [B, P]
        p_curr = p_history[:, -1]   # [B, P]
        v_prev = img_history[:, 0]        # [B, Dv]
        v_curr = img_history[:, -1]       # [B, Dv]

        # Encode with FiLM-MLP conditioned on language
        p_prev = self.proprio_encoder(p_prev, subgoal)   # [B, H]
        p_curr = self.proprio_encoder(p_curr, subgoal)   # [B, H]
        v_prev = self.vision_encoder(v_prev, subgoal)    # [B, H]
        v_curr = self.vision_encoder(v_curr, subgoal)    # [B, H]
        # a_prev = self.action_encoder(prev_action, subgoal)  # [B, H]
        a_prev = self.action_encoder(prev_action)  # [B, H]
        delta_p = p_curr - p_prev                       # [B, H]
        delta_v = v_curr - v_prev                       # [B, H]

        p_tokens = self.proprio_tokenizer(delta_p).view(B, self.num_p_tokens, -1)  # [B, Np, D]
        v_tokens = self.vision_tokenizer(delta_v).view(B, self.num_v_tokens, -1)  # [B, Nv, D]

        p_tokens = self.p_token_ln(p_tokens)
        v_tokens = self.v_token_ln(v_tokens)

        p_pos = self.ff_pos_p(self.p_pos.expand(B, -1))  # [B, Np, H]
        v_pos = self.ff_pos_v(self.v_pos.expand(B, -1))  # [B, Nv, H]

        p_tokens = p_tokens + p_pos + self.mod_embed[0]
        v_tokens = v_tokens + v_pos + self.mod_embed[1]

        p_tokens = self.token_dropout(p_tokens)
        v_tokens = self.token_dropout(v_tokens)

        # a_tokens = self.action_tokenizer(a_prev).view(B, self.num_a_tokens, -1)  # [B, Na, D]
        # a_tokens = self.a_token_ln(a_tokens)
        # a_pos = self.ff_pos_a(self.a_pos.expand(B, -1))  # [B, Na, H]
        # a_tokens = a_tokens + a_pos + self.mod_embed[2]
        # a_tokens = self.token_dropout(a_tokens)

        # ik for each modality
        # 1) FiLM
        # p_ik = self.proprio_ik(p_tokens, a_prev)  # [B, Np, D]
        # v_ik = self.vision_ik(v_tokens, a_prev)  # [B, Nv, D]

        # 2) Cross-attention
        # p_ik = self.proprio_ik(p_tokens, a_tokens)  # [B, Np, D]
        # v_ik = self.vision_ik(v_tokens, a_tokens)  # [B, Nv, D]
        p_ik = self.proprio_ik(p_tokens, a_prev.unsqueeze(1))  # [B, Np, D]
        v_ik = self.vision_ik(v_tokens, a_prev.unsqueeze(1))  # [B, Nv, D]

        # Perceiver-IO: Q = learned latent array; KV = [pl_tokens, vl_tokens]
        # TODO q_latents 와 q_out 의 의미?
        q_latents = repeat(self.latent_query, '1 n d -> b n d', b=B)  # [B, L, H]
        q_out = repeat(self.output_query, '1 p d -> b p d', b=B)     # [B, 1, H]
        kv_token = torch.cat([p_ik, v_ik], dim=1)  # [B, 2*Na, D]

        # Extract latent ik
        latent_ik = self.latent_ik(q_latents, kv_token)
        latent_ik = self.latent_emb(q_out, latent_ik).squeeze(1)  # [B, H]

        # Predict Δproprio // t (Δp' | Δp) => Δp' = f(t, Δp)
        delta_p_sg = (p_curr - p_prev).detach()
        rollout = torch.cat([latent_ik, delta_p_sg], dim=-1)
        delta_p_next = self.delta_head(rollout)
        pred_p_next = p_curr.detach() + delta_p_next

        # rollout_v = torch.cat([latent_ik, delta_v], dim=-1)
        # delta_v_next = self.delta_head(rollout_v)
        # pred_v_next = v_curr.detach() + delta_v_next

        # ---------------- Representation losses ------------------ #
        if self.training:
            loss_dict = {}
            # 1-1) InfoNCE between token sets
            # loss_nce = self._info_nce_set2set(p_ik, v_ik, tau=self.tau)
            # loss_dict['loss_nce'] = loss_nce
            # 1-2) SimCLR-style contrastive loss on pooled tokens
            # p_ik_proj = p_ik.mean(1)
            # v_ik_proj = v_ik.mean(1)
            # p_ik_proj = self.proj_head(p_ik.mean(1))
            # v_ik_proj = self.proj_head(v_ik.mean(1))
            # loss_simclr = self._simclr_loss(p_ik_proj, v_ik_proj, tau=self.tau)
            # loss_simclr_p = self._simclr_loss(latent_ik, p_ik_proj, tau=self.tau)
            # loss_simclr_v = self._simclr_loss(latent_ik, v_ik_proj, tau=self.tau)
            # loss_dict['loss_simclr'] = loss_simclr

            # 2) Latent alignment: cosine alignment between latent_ik and detached p/v means (scale-invariant)
            # loss_align = self._cosine_align(p_ik_proj, v_ik_proj)
            # loss_dict['loss_align'] = loss_align

            with torch.no_grad():
                p_next_gt = self.proprio_encoder(p_next, subgoal)
                # v_next_gt = self.vision_encoder(v_next, subgoal)

            loss_delta = F.smooth_l1_loss(pred_p_next, p_next_gt)
            loss_dict['loss_delta_p'] = loss_delta
            # loss_delta_v = F.smooth_l1_loss(pred_v_next, v_next_gt)
            # loss_dict['loss_delta_v'] = loss_delta_v

            return pred_p_next, loss_dict

        return pred_p_next, None

    def _simclr_loss(self, a, b, tau: float = 0.07):
        """
        SimCLR-style contrastive loss with a small projection head and stop-grad target.
        a: [B, K, D], b: [B, M, D]
        Steps:
          1) Pool tokens per modality to [B, D].
          2) Project with MLP head (shared) and L2-normalize.
          3) Compute InfoNCE in both directions with stop-gradient on the target branch.
        """
        B = a.shape[0]

        # 2) Projection head + normalization
        a = F.normalize(a, dim=-1, eps=1e-6)
        b = F.normalize(b, dim=-1, eps=1e-6)

        # 3) InfoNCE with stop-grad targets (two directions)
        labels = torch.arange(B, device=a.device)
        # logits = a @ (b.detach().t()) / tau
        logits_ab = a @ (b.detach().t()) / tau
        logits_ba = b @ (a.detach().t()) / tau
        loss = 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels))
        return loss

    def _cosine_align(self, a, b):
        """Cosine-based alignment loss: 1 - cosine(a, b), averaged over batch."""
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return 1.0 - (a_n * b_n).sum(dim=-1).mean()

    def _info_nce(self, a, b, tau=0.07):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.t() / tau
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


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