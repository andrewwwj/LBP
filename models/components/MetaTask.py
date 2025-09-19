import math
import copy
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

def _init_film_fc2_zero(m):
    if hasattr(m, 'cond_proj') and hasattr(m.cond_proj, 'fc2'):
        nn.init.zeros_(m.cond_proj.fc2.weight)
        nn.init.zeros_(m.cond_proj.fc2.bias)


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

    def __init__(self, proprio_dim, vl_dim, hidden_dim, p_goal_dim, action_dim, num_latents=2, action_noise=True,
                 num_p_tokens=8, num_v_tokens=8, num_a_tokens=1):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.action_drop_prob = 0.1
        self.action_noise = action_noise
        self.latent_query = nn.Parameter(torch.randn(1, num_latents, p_goal_dim))
        self.output_query = nn.Parameter(torch.randn(1, 1, p_goal_dim))
        self.num_v_tokens = num_v_tokens
        self.num_p_tokens = num_p_tokens
        self.num_a_tokens = num_a_tokens

        # Loss hyperparameters
        self.tau = 0.07
        self.w_nce = 1.0
        self.w_align = 1.0
        # TODO cross-attention 활용
        self.proprio_encoder = FilmMLP(input_dim=proprio_dim, cond_dim=vl_dim, output_size=p_goal_dim)
        # self.proprio_encoder = MlpResNet(num_blocks=3, input_dim=proprio_dim, hidden_dim=hidden_dim, output_size=hidden_dim)
        self.vision_encoder = FilmMLP(input_dim=vl_dim, cond_dim=vl_dim, output_size=p_goal_dim)  # input_dim=vl_dim * 2 if double-view
        # self.vision_encoder = MlpResNet(num_blocks=3, input_dim=vl_dim, hidden_dim=hidden_dim, output_size=hidden_dim)
        self.action_encoder = MlpResNet(num_blocks=3, input_dim=action_dim, hidden_dim=hidden_dim, output_size=p_goal_dim)

        self.p_ln = nn.LayerNorm(p_goal_dim)
        self.v_ln = nn.LayerNorm(p_goal_dim)
        self.a_ln = nn.LayerNorm(p_goal_dim)
        self.time_pe = FourierPositionalEncoding1D(num_bands=4, out_dim=p_goal_dim, include_input=True)
        # ik solver
        # ---------------------- FiLM ik ----------------------
        # self.vision_ik = FilmMLP(input_dim=hidden_dim, cond_dim=hidden_dim, output_size=hidden_dim)
        # self.proprio_ik = FilmMLP(input_dim=hidden_dim, cond_dim=hidden_dim, output_size=hidden_dim)

        # ----------------- Cross-attention ik -----------------
        self.proprio_ik = CrossAttnBlock(embed_dim=p_goal_dim,
                                         dim_feedforward=p_goal_dim * 4,
                                         num_heads=8,
                                         num_layers=2,
                                         drop_out_rate=0.1)
        self.vision_ik = CrossAttnBlock(embed_dim=p_goal_dim,
                                        dim_feedforward=p_goal_dim * 4,
                                        num_heads=8,
                                        num_layers=2,
                                        drop_out_rate=0.1)

        self.latent_ik = CrossAttnBlock(embed_dim=p_goal_dim,
                                        dim_feedforward=p_goal_dim * 4,
                                        num_heads=8,
                                        num_layers=3,
                                        drop_out_rate=0.1)
        self.latent_emb = CrossAttnBlock(embed_dim=p_goal_dim,
                                         dim_feedforward=p_goal_dim * 2,
                                         num_heads=4,
                                         num_layers=1,
                                         drop_out_rate=0.1)

        self.proj_latent = nn.Sequential(nn.Linear(p_goal_dim, p_goal_dim), nn.GELU(), nn.Linear(p_goal_dim, p_goal_dim))

        self.pred_shared = nn.Sequential(
            nn.Linear(p_goal_dim, p_goal_dim * 4),
            nn.GELU(),
            nn.LayerNorm(p_goal_dim * 4),
        )
        self.pred_proprio_head = nn.Linear(p_goal_dim * 4, p_goal_dim)
        self.pred_vision_head = nn.Linear(hidden_dim * 4, p_goal_dim)

        self.apply(init_weight)
        self.apply(_init_film_fc2_zero)

        # EMA target encoder for stable targets
        # self.m_ema = 0.99
        # self._ema_initialized = False
        self.proprio_encoder_target = copy.deepcopy(self.proprio_encoder)
        for p in self.proprio_encoder_target.parameters():
            p.requires_grad = False
        self.vision_encoder_target = copy.deepcopy(self.vision_encoder)
        for p in self.vision_encoder_target.parameters():
            p.requires_grad = False


    def forward(self, img_history, p_history, subgoal, prev_action, p_next=None, v_next=None):
        B = p_history.shape[0]

        if self.training:
            # Action augmentation before encoding
            prob = torch.rand((), device=prev_action.device)
            if (self.action_drop_prob is not None) and (self.action_drop_prob > 0.0) and (prob < self.action_drop_prob):
                prev_action = torch.zeros_like(prev_action)
            elif self.action_noise:
                prev_action = prev_action + 0.01 * torch.randn_like(prev_action)

        p_prev_raw = p_history[:, 0]    # [B, P]
        p_curr_raw = p_history[:, -1]   # [B, P]
        v_prev_raw = img_history[:, 0]        # [B, Dv]
        v_curr_raw = img_history[:, -1]       # [B, Dv]

        # Encode with FiLM-MLP conditioned on language
        p_prev = self.proprio_encoder(p_prev_raw, subgoal)   # [B, H]
        p_curr = self.proprio_encoder(p_curr_raw, subgoal)   # [B, H]
        v_prev = self.vision_encoder(v_prev_raw, subgoal)    # [B, H]
        v_curr = self.vision_encoder(v_curr_raw, subgoal)    # [B, H]

        a_prev = self.action_encoder(prev_action)  # [B, H]
        # delta_p = p_curr - p_prev                       # [B, H]
        # delta_v = v_curr - v_prev                       # [B, H]

        # ik for each modality
        # 1) FiLM
        # a_t = self.a_ln(a_prev)
        # 2) Cross-attention
        a_t = self.a_ln(a_prev.unsqueeze(1))
        p_prev_t = self.p_ln(p_prev.unsqueeze(1))
        p_curr_t = self.p_ln(p_curr.unsqueeze(1))
        v_prev_t = self.v_ln(v_prev.unsqueeze(1))
        v_curr_t = self.v_ln(v_curr.unsqueeze(1))
        p_delta_t = self.p_ln((p_curr - p_prev).unsqueeze(1))
        v_delta_t = self.v_ln((v_curr - v_prev).unsqueeze(1))

        t_pos = torch.tensor([0.0, 1.0], device=p_history.device).unsqueeze(0).repeat(B, 1)  # [B,2]
        pos = self.time_pe(t_pos)  # [B,2,D]
        p_prev_t = p_prev_t + pos[:, 0:1, :]
        p_curr_t = p_curr_t + pos[:, 1:2, :]
        v_prev_t = v_prev_t + pos[:, 0:1, :]
        v_curr_t = v_curr_t + pos[:, 1:2, :]
        pos_delta = self.time_pe(torch.full((B, 1), 0.5, device=p_history.device))  # [B,1,D]
        p_delta_t = p_delta_t + pos_delta
        v_delta_t = v_delta_t + pos_delta

        p_kv = torch.cat([p_prev_t, p_curr_t, p_delta_t], dim=1)  # [B,3,D]
        v_kv = torch.cat([v_prev_t, v_curr_t, v_delta_t], dim=1)  # [B,3,D]

        # if self.training and (self.kv_mask_prob is not None) and (self.kv_mask_prob > 0.0):
        #     mask_sel = torch.rand(B, device=p_kv.device) < self.kv_mask_prob
        #
        # if mask_sel.any():
        #     to_drop = torch.randint(0, 3, (int(mask_sel.sum().item()),), device=p_kv.device)
        #     bidx = mask_sel.nonzero(as_tuple=False).squeeze(1)
        #     p_kv[bidx, to_drop, :] = 0.0
        #     v_kv[bidx, to_drop, :] = 0.0

        p_ik = self.proprio_ik(a_t, p_kv)
        v_ik = self.vision_ik(a_t, v_kv)

        # z_p_ik = self.proj_latent(p_ik)
        # z_v_ik = self.proj_latent(v_ik)

        # Perceiver-IO: Q = learned latent array; KV = [pl_tokens, vl_tokens]
        # TODO q_latents 와 q_out 의 의미?
        q_latents = repeat(self.latent_query, '1 n d -> b n d', b=B)  # [B, L, D]
        q_out = repeat(self.output_query, '1 p d -> b p d', b=B)     # [B, 1, D]
        kv_token = torch.cat([p_ik, v_ik], dim=1)  # [B, 2, D]

        # Extract latent ik
        latent_ik = self.latent_ik(q_latents, kv_token)            # [B, L, D]
        latent_ik = self.latent_emb(q_out, latent_ik)   # [B, D]

        # Predict proprio_emb // t (p' | p) => p' = f(t, p)
        rollout = torch.cat([latent_ik, p_kv.detach()], dim=1)
        pred_p_next = self.pred_proprio_head(self.pred_shared(rollout)).mean(1)  # [B, D]
        # pred_p_next = p_curr.detach() + pred_delta_p

        rollout_v = torch.cat([latent_ik, v_kv.detach()], dim=1)
        pred_v_next = self.pred_vision_head(self.pred_shared(rollout_v)).mean(1)

        # ---------------- Representation losses ------------------ #
        if self.training:
            loss_dict = {}
            # Initialize and EMA update for target encoder
            # if not self._ema_initialized:
            #     self.proprio_encoder_target.load_state_dict(self.proprio_encoder.state_dict())
            #     self.vision_encoder_target.load_state_dict(self.vision_encoder.state_dict())
            #     self._ema_initialized = True
            # self._momentum_update_target(self.proprio_encoder_target, self.proprio_encoder, self.m_ema)
            # self._momentum_update_target(self.vision_encoder_target, self.vision_encoder, self.m_ema)
            with torch.no_grad():
                target_p_next = self.proprio_encoder_target(p_next, subgoal)
                # target_p_curr = self.proprio_encoder_target(p_curr_raw, subgoal)
                # target_delta_p = target_p_next - target_p_curr
                target_v_next = self.vision_encoder_target(v_next, subgoal)

            loss_p_emb = F.smooth_l1_loss(pred_p_next, target_p_next)
            loss_dict['loss_p'] = loss_p_emb

            loss_v_emb = F.smooth_l1_loss(pred_v_next, target_v_next)
            loss_dict['loss_v'] = loss_v_emb

            # loss_align = self._simclr_loss(p_ik.squeeze(1), v_ik.squeeze(1), tau=self.tau)
            # loss_dict['loss_latent_align'] = loss_align

            # loss_set2set = self._info_nce_set2set(p_ik.squeeze(1), v_ik.squeeze(1), tau=self.tau)
            # loss_dict['loss_slatent_align'] = loss_set2set

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

    def _momentum_update_target(self, target, online, m: float = 0.99):
        with torch.no_grad():
            for p_t, p_o in zip(target.parameters(), online.parameters()):
                p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)