import copy
import math
from timm.layers import Mlp
import torch.nn.functional as F
import torch
import torch.nn as nn
from .MlpResNet import MlpResNet


class LearnedPosEmb(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)

    def forward(self, x):
        f = 2 * torch.pi * x @ self.kernel.T
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f


def cosine_beta_schedule(timesteps, s=0.008):
    """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPMHead(nn.Module):
    def __init__(
            self,
            action_size: int = 7,
            num_blocks: int = 3,
            pvg_dim: int = 544,
            hidden_dim: int = 256,
            time_dim: int = 32,
            num_timesteps: int = 25,
            time_hidden_dim: int = 256,
            guidance_mode: str = "cg",
            proprio_dim: int = None,
            latent_goal_dim: int = None,
            vis_lang_dim: int = None,
            diffusion_input_key: str = None,
            energy_input_key: str = None,
            policy_dict: dict = None
    ):
        super().__init__()
        self.mode = guidance_mode
        self.time_dim = time_dim
        self.num_timesteps = num_timesteps
        self.time_hidden_dim = time_hidden_dim
        self.action_size = action_size
        self.num_samples = 64  # Number of action samples
        self.time_process = LearnedPosEmb(1, time_dim)
        self.time_encoder = Mlp(time_dim, time_hidden_dim, time_hidden_dim, norm_layer=nn.LayerNorm)
        self.diffusion_input_key = diffusion_input_key
        self.energy_input_key = energy_input_key if self.mode == 'energy' else ''

        combination_dict = {'v': vis_lang_dim, 'p': proprio_dim, 'g': latent_goal_dim,
                            'vp': proprio_dim + vis_lang_dim, 'vg': vis_lang_dim + latent_goal_dim,
                            'pg': proprio_dim + latent_goal_dim, 'pvg': proprio_dim + vis_lang_dim + latent_goal_dim
                            }
        self.diffusion_input_dim = combination_dict[self.diffusion_input_key]
        self.energy_input_dim = combination_dict[self.energy_input_key] if self.mode == 'energy' else 0
        if self.mode == 'cg':
            self.model = MlpResNet(num_blocks=num_blocks,
                                   input_dim=self.diffusion_input_dim + action_size + time_hidden_dim,
                                   hidden_dim=hidden_dim, output_size=action_size)
        elif self.mode == 'cfg':
            self.model = MlpResNet(num_blocks=num_blocks,
                                   input_dim=pvg_dim + action_size + time_hidden_dim,
                                   hidden_dim=hidden_dim, output_size=action_size)

        if self.mode == 'cg':
            pass
        elif self.mode == 'cfg':
            self.cfg_prob = 0.1
            self.w_cfg = 1.0
            # self.uncond_embedding = nn.Parameter(torch.zeros(1, self.diffusion_input_dim))  # zero embedding
            self.uncond_embedding = nn.Parameter(torch.randn(1, self.diffusion_input_dim))  # learnable embedding
        elif self.mode == 'energy':
            critic_input_dim = self.energy_input_dim + action_size + time_hidden_dim
            self.policy_dict = policy_dict
            # Q-networks for Double Q-learning
            def create_q_network(critic_dim):
                return nn.Sequential(
                    nn.Linear(critic_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1)
                )
            # Current Q-networks
            self.q0_network = create_q_network(critic_input_dim)
            self.q1_network = create_q_network(critic_input_dim)
            # Target Q-networks
            self.q0_target = copy.deepcopy(self.q0_network)
            self.q1_target = copy.deepcopy(self.q1_network)
            for param in self.q0_target.parameters():
                param.requires_grad = False
            for param in self.q1_target.parameters():
                param.requires_grad = False

            # CEP hyperparameters
            self.w_critic_loss = 0.1
            self.w_sigma = 0.5
            self.temperature = 0.1  # Temperature for distance-based weighting
            self.alpha = 3.0  # Temperature parameter for softmax
            self.guidance_scale = 1.0  # Guidance strength
            self.tau = 0.005  # Soft update rate for target network
        else:
            raise NotImplementedError(f"Unknown mode {self.mode}")
        self.init_ddpm()

    def init_ddpm(self, device='cuda'):
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(device)

    def forward_features(self, obs, t_tensor, noise_action):
        time_embedding = self.time_process(t_tensor.view(-1, 1))
        time_embedding = self.time_encoder(time_embedding)
        input_feature = torch.cat([obs, time_embedding, noise_action], dim=-1)
        noise_pred = self.model(input_feature)
        return noise_pred

    def forward(self, all_obs, cur_action):
        vl_semantics, proprio_obs, fused_goal = all_obs
        modality_dict = {'v': vl_semantics, 'p': proprio_obs, 'g': fused_goal, }
        diffusion_obs = torch.cat([modality_dict[key] for key in self.diffusion_input_key], dim=-1)
        pvg_obs = torch.cat([proprio_obs, vl_semantics, fused_goal], dim=-1)
        B, T, S, A = diffusion_obs.shape[0], self.num_timesteps, self.num_samples, self.action_size

        if self.mode == 'cg':
            # Sample t & noisy action at t
            noise, t_tensor, noisy_action = self.add_noise(cur_action)
            noise_pred = self.forward_features(diffusion_obs, t_tensor, noisy_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            return diffusion_loss
        elif self.mode == 'cfg':
            assert self.diffusion_input_key == 'vg'
            # Sample t & noisy action at t
            noise, t_tensor, noisy_action = self.add_noise(cur_action)
            uncond_mask = torch.rand(diffusion_obs.shape[0], device=diffusion_obs.device) < self.cfg_prob

            # 1) Use masking (unconditioning) for all obs
            # diffusion_obs[uncond_mask] = self.uncond_embedding
            # noise_pred = self.forward_features(diffusion_obs, t_tensor, noisy_action)
            # 2) Use masking (unconditioning) for certain obs
            diffusion_obs[uncond_mask] = self.uncond_embedding
            obs = torch.cat([proprio_obs, diffusion_obs], dim=-1)
            noise_pred = self.forward_features(obs, t_tensor, noisy_action)

            diffusion_loss = F.mse_loss(noise_pred, noise)
            return diffusion_loss
        elif self.mode == 'energy':
            assert self.energy_input_key == 'pvg' and self.diffusion_input_key == 'p'
            energy_obs = torch.cat([modality_dict[key] for key in self.energy_input_key], dim=-1)
            student_policy = self.policy_dict['student']
            expert_policy = self.policy_dict['expert']
            # Vectorize obs for generating samples
            proprio_obs_ = proprio_obs.repeat_interleave(S, dim=0)
            # Store predictions for each step
            t_buffer = torch.zeros(T, B, 1, device=diffusion_obs.device)
            # action_ref_buffer = torch.zeros(T, B, A, device=diffusion_obs.device)
            noise_ref_buffer = torch.zeros(T, B, A, device=diffusion_obs.device)
            # action_buffer = torch.zeros(T, B, S, A, device=diffusion_obs.device)
            noise_buffer = torch.zeros(T, B, S, A, device=diffusion_obs.device)
            with torch.no_grad():
                # Sample noise actions from expert q(a | v, p, g) and student q(a | p)
                initial_noise = torch.randn((B, A), device=diffusion_obs.device)
                action_t_expert = initial_noise.clone()
                action_t_student = initial_noise.clone().repeat_interleave(S, dim=0)  # start with the same noise as the expert's
                for t in reversed(range(self.num_timesteps)):
                    # --- 1. Generate Reference Action (Positive Sample) ---
                    t_tensor = torch.full((B, 1), t, device=diffusion_obs.device, dtype=torch.long)
                    noise_ref = expert_policy.head.forward_features(pvg_obs, t_tensor, action_t_expert)
                    action_t_expert = expert_policy.head.denoise(action_t_expert, t_tensor, noise_ref)
                    # --- 2. Generate Predicted Actions (Negative Samples) - VECTORIZED ---
                    t_tensor_ = t_tensor.repeat_interleave(S, dim=0)
                    noise_pred = student_policy.head.forward_features(proprio_obs_, t_tensor_, action_t_student)
                    # action_t_student = student_policy.head.denoise(action_t_student, t_tensor_, noise_pred)
                    # Save to buffer
                    t_buffer[t] = t_tensor
                    noise_ref_buffer[t] = noise_ref
                    # action_ref_buffer[t] = action_t_expert
                    noise_buffer[t] = noise_pred.view(B, S, A)
                    # action_buffer[t] = action_t_student.view(B, S, A)
            # TODO Appropriate inputs for energy loss
            loss = self.loss_energy(energy_obs, t_buffer, noise_ref_buffer, noise_buffer)
            return loss
        else:
            raise NotImplementedError

    def add_noise(self, x0: torch.Tensor):
        """
        sample noisy xt from x0, q(xt|x0), forward process

        Input:
            x0: ground truth action / t: timestep / noise: noise

        Return:
            noise, timestep t, noisy action at t
        """
        B = x0.shape[0]
        noise = torch.randn_like(x0, device=x0.device)
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        alphas_cumprod_t = self.alphas_cumprod[t]
        xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
        return noise, t, xt

    def loss_energy(self, energy_obs, t_buffer, noise_ref_buffer, noise_buffer):
        """
        Calculates the stable, symmetric contrastive energy loss.
        """
        # --- Prepare Inputs ---
        T, B, S, A = noise_buffer.shape
        obs_dim = energy_obs.shape[-1]

        # Reshape buffers for batch processing: [T, B, ...] -> [T * B, ...]
        t_buffer_flat = t_buffer.view(T * B, 1)
        noise_ref_flat = noise_ref_buffer.view(T * B, A).unsqueeze(1)
        noise_flat = noise_buffer.view(T * B, S, A)
        energy_obs_flat = energy_obs.unsqueeze(0).expand(T, -1, -1).reshape(T * B, obs_dim)

        # Create and correctly embed time tensor
        time_emb = self.time_encoder(self.time_process(t_buffer_flat))  # [T * B, time_hidden_dim]

        # --- Contrastive Loss Calculation ---
        time_expanded = time_emb.unsqueeze(1).expand(-1, S + 1, -1)
        obs_expanded = energy_obs_flat.unsqueeze(1).expand(-1, S + 1, -1)
        noise_all = torch.cat([noise_ref_flat, noise_flat], dim=1)

        # Input features for Q-network
        q_input = torch.cat([obs_expanded, time_expanded, noise_all], dim=-1)

        # Get Q-values from both critics
        q0_values = self.q0_network(q_input).squeeze(-1)  # [T*B, S+1]
        q1_values = self.q1_network(q_input).squeeze(-1)  # [T*B, S+1]

        # Calculate distance-based weights for negative samples
        # with torch.no_grad():
        #     dist_sq = torch.sum((noise_ref_flat - noise_flat) ** 2, dim=-1)  # [T*B, S]
        #     weights = torch.exp(-dist_sq / (2 * self.w_sigma ** 2))  # [T*B, S]

        def contrastive_energy_loss(logits):
            q_pos = logits[:, 0].unsqueeze(-1)
            q_neg = logits[:, 1:]
            # weighted_q_neg = q_neg + torch.log(weights + 1e-8)  # log(w*exp(q)) = log(w)+q
            # final_logits = torch.cat([q_pos, weighted_q_neg], dim=-1) / self.alpha
            final_logits = torch.cat([q_pos, q_neg], dim=-1) / self.alpha

            # The target is always the positive sample at index 0
            labels = torch.zeros(final_logits.shape[0], dtype=torch.long, device=final_logits.device)
            return F.cross_entropy(final_logits, labels)

        loss_q0 = contrastive_energy_loss(q0_values)
        loss_q1 = contrastive_energy_loss(q1_values)
        contrastive_loss = loss_q0 + loss_q1

        # --- 2. Regression Loss ---
        # Regress the Q-value of the positive noise towards the stable target network's estimate
        with torch.no_grad():
            q0_target_values = self.q0_target(q_input).squeeze(-1)
            q_pos_target = q0_target_values[:, 0]
        q0_pos_online = q0_values[:, 0]
        q1_pos_online = q1_values[:, 0]
        regression_loss = F.huber_loss(q0_pos_online, q_pos_target) + F.huber_loss(q1_pos_online, q_pos_target)

        total_loss = contrastive_loss + 0.1 * regression_loss

        self.soft_update_target_network()
        return total_loss

    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, param in zip(self.q0_target.parameters(), self.q0_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.q1_target.parameters(), self.q1_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    @torch.no_grad()
    def generate(self, all_obs):
        vl_semantics, proprio_obs, fused_goal = all_obs
        modality_dict = {'v': vl_semantics, 'p': proprio_obs, 'g': fused_goal, }
        diffusion_obs = torch.cat([modality_dict[key] for key in self.diffusion_input_key], dim=-1)
        # pvg_obs = torch.cat([vl_semantics, proprio_obs, fused_goal], dim=-1)

        shape = (diffusion_obs.shape[0], self.action_size)
        student_policy = self.policy_dict['student']
        # expert_policy = self.policy_dict['expert']

        # Sample noise action
        action_t = torch.randn(shape, device=diffusion_obs.device)
        # action_expert = action_t.clone()
        # error_expert = []
        for step in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0], 1), step, device=diffusion_obs.device, dtype=torch.long)
            if self.mode == 'cg':
                noise_pred = self.forward_features(diffusion_obs, t_tensor, action_t)
                action_t = self.denoise(action_t, t_tensor, noise_pred)
            elif self.mode == 'cfg':
                """0) all_obs"""
                uncond_shape = diffusion_obs.shape[0]
                uncond_obs = self.uncond_embedding.repeat(uncond_shape, 1)
                cond_obs = diffusion_obs.clone()
                """1) vl_semantics"""
                # uncond_shape = vl_semantics.shape[0]
                # uncond_obs = torch.cat([self.uncond_embedding.repeat(uncond_shape, 1), proprio_obs, fused_goal], dim=-1)
                # cond_obs = torch.cat([vl_semantics, proprio_obs, fused_goal], dim=-1)
                """2) vl_semantics + fused goal"""
                # uncond_shape = torch.cat([vl_semantics, fused_goal], dim=-1).shape[0]
                # uncond_obs = torch.cat([self.uncond_embedding.repeat(uncond_shape, 1), proprio_obs], dim=-1)
                # cond_obs = torch.cat([vl_semantics, fused_goal, proprio_obs], dim=-1)
                """Common"""
                noise_pred_uncond = self.forward_features(uncond_obs, t_tensor, action_t)
                noise_pred_cond = self.forward_features(cond_obs, t_tensor, action_t)
                noise_pred = noise_pred_uncond + self.w_cfg * (noise_pred_cond - noise_pred_uncond)
                action_t = self.denoise(action_t, t_tensor, noise_pred)
            elif self.mode == 'energy':
                assert self.energy_input_key == 'pvg' and self.diffusion_input_key == 'p'
                energy_obs = torch.cat([modality_dict[key] for key in self.energy_input_key], dim=-1)
                # Diffusion denoising with proprio conditioning
                noise_t = student_policy.head.forward_features(diffusion_obs, t_tensor, action_t)
                # Energy guidance
                energy_grad = self.energy_guidance(noise_t, t_tensor, energy_obs)[0]
                scale_factor = extract(self.sqrt_one_minus_alphas_cumprod[t_tensor], energy_grad.shape)
                noise_t = noise_t - self.guidance_scale * scale_factor * energy_grad
                action_t = student_policy.head.denoise(action_t, t_tensor, noise_t)
                # (Optional) Compare with expert policy
                # noise_pred = expert_policy.head.forward_features(pvg_obs, t_tensor, action_expert)
                # action_expert = expert_policy.head.denoise(action_expert, t_tensor, noise_pred)
                # error = F.mse_loss(action_t, action_expert)
                # error_expert.append(error.item())
        action_t = action_t.clamp(-1., 1.)
        return action_t

    def energy_guidance(self, noise, t, condition):
        """
        Calculates the guidance term for the noise prediction based on the score function.
        Returns: w * sqrt(1 - alpha_bar_t) * grad(Q(x_t)).
        """
        with torch.enable_grad():
            noise_ = noise.detach().clone().requires_grad_(True)

            t_emb = self.time_encoder(self.time_process(t.float()))
            q_input = torch.cat([condition, t_emb, noise_], dim=-1)

            q0_values = self.q0_network(q_input)
            q1_values = self.q1_network(q_input)
            q_values = torch.min(q0_values, q1_values)
            energy = -q_values

            grad = torch.autograd.grad(outputs=energy.sum(), inputs=noise_)
        # sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t.view(-1)].view(-1, 1)
        # guidance = self.guidance_scale * sqrt_one_minus_alpha_bar_t * grad

        # return guidance.detach()
        return grad

    # def loss_energy(self, energy_obs, action_ref_buffer, action_pred_buffer):
    #     # --- Contrastive Energy Prediction Loss ---
    #     B, T, S, A = energy_obs.shape[0], self.num_timesteps, self.num_samples, self.action_size
    #     action_ref_expanded = action_ref_buffer.unsqueeze(2)
    #     # Concatenate positive and negative samples for Q-value computation
    #     all_actions = torch.cat([action_ref_expanded, action_pred_buffer], dim=2)
    #     # Prepare inputs for Q-Network
    #     # energy_obs: [B, Obs_dim] -> [T, B, 1, Obs_dim] -> [T, B, S+1, Obs_dim]
    #     obs_dim = energy_obs.shape[-1]
    #     energy_obs_expanded = energy_obs.view(1, B, 1, obs_dim).expand(T, -1, S + 1, -1)
    #     # t_tensor needs to be shaped correctly for time embedding
    #     t_values = torch.arange(T, device=energy_obs.device).view(T, 1, 1, 1).expand(-1, B, S + 1, -1)
    #     # Get Q-values for all actions (positive and negative)
    #     q_values = self.q0_network(torch.cat([energy_obs_expanded, all_actions], dim=-1), t_values).squeeze(-1)
    #     # --- Weighted Contrastive Loss Calculation ---
    #     # Positive Q-values: Q(a_ref) -> [T, B]
    #     q_pos = q_values[:, :, 0]
    #     # Negative Q-values: Q(a_pred) -> [T, B, S]
    #     q_neg = q_values[:, :, 1:]
    #     # Calculate distance-based weights (using squared L2 distance)
    #     dist_sq = torch.sum((action_ref_expanded - action_pred_buffer) ** 2, dim=-1)
    #     # Convert distance to weights using a Gaussian kernel. `sigma` is a tunable hyperparameter.
    #     sigma = 0.1  # Example value, needs tuning
    #     weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # weights are between 0 and 1
    #     # Apply weights to negative Q-values. This treats "close" negatives as "less negative".
    #     weighted_q_neg = q_neg + weights.log()  # log(w * exp(q)) = log(w) + q
    #     # Combine positive and weighted negative Q-values for log_softmax
    #     logits = torch.cat([q_pos.unsqueeze(-1), weighted_q_neg], dim=-1)
    #     # Final Contrastive Loss. Maximize the probability of the positive sample (index 0)
    #     contrastive_loss = -F.log_softmax(logits, dim=-1)[:, :, 0].mean()
    #     return contrastive_loss

    # def energy_guidance(self, actions, t, condition):
    #     with torch.enable_grad():
    #         # actions = actions.detach().requires_grad_(True)
    #         # t_emb = self.time_encoder(self.time_process(t.reshape(-1, 1).to(torch.float32)))
    #         # q_values = self.q1_network(torch.cat([condition, t_emb, actions], dim=-1))
    #         # guidance_scale = self.guidance_scale
    #         # grad = torch.autograd.grad(torch.sum(q_values), actions, create_graph=False)[0]
    #         # guidance = guidance_scale * grad
    #         actions = actions.detach().requires_grad_(True)
    #         t_emb = self.time_encoder(self.time_process(t.reshape(-1, 1).to(torch.float32)))\
    #         # Prepare input for both Q-networks
    #         q_input = torch.cat([condition, t_emb, actions], dim=-1)
    #         # --- Use the minimum of the two Q-networks for a stable estimate ---
    #         q0_values = self.q0_network(q_input)
    #         q1_values = self.q1_network(q_input)
    #         q_values = torch.min(q0_values, q1_values)
    #         guidance_scale = self.guidance_scale
    #         grad = torch.autograd.grad(torch.sum(q_values), actions, create_graph=False)[0]
    #
    #         guidance = guidance_scale * grad
    #     return guidance.detach()


    def denoise(self, xt, t, noise_pred, clip_sample=True, ddpm_temperature=1.):
        """
            reverse process; sample xt-1 from xt, p(xt-1|xt)
        """
        t = t.view(-1, 1)
        # x = self.p_sample(x, t_tensor, noise_pred, clip_sample)
        alpha1 = 1 / torch.sqrt(self.alphas[t])
        alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
        xtm1 = alpha1 * (xt - alpha2 * noise_pred)
        noise = torch.randn_like(xtm1, device=xt.device) * ddpm_temperature
        xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
        if clip_sample:
            xtm1 = torch.clip(xtm1, -1., 1.)
        return xtm1

    def predict_x0_from_noise(self, xt, t, noise):
        """
        Predicts the original sample x_0 from a noisy sample x_t and predicted noise.
        """
        term = torch.sqrt(1. - self.alphas_cumprod[t]).unsqueeze(-1)
        x0 = (xt - term * noise) / torch.sqrt(self.alphas_cumprod[t]).unsqueeze(-1)
        return torch.clamp(x0, -1., 1.)

def extract(a, x_shape):
    '''
    align the dimention of alphas_cumprod_t to x_shape

    a: alphas_cumprod_t, B
    x_shape: B x F x F x F
    output: alphas_cumprod_t B x 1 x 1 x 1]
    '''
    b, *_ = a.shape
    return a.reshape(b, *((1,) * (len(x_shape) - 1)))


class BaseHead(nn.Module):
    def __init__(
            self,
            action_size: int = 7,
            num_blocks: int = 3,
            input_dim: int = 544,
            hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_size = action_size
        self.model = MlpResNet(num_blocks=num_blocks, input_dim=input_dim,
                               hidden_dim=hidden_dim, output_size=action_size)

    def forward(self, all_obs):
        action = self.model(all_obs)
        return action

    def generate(self, all_obs):
        action = self.model(all_obs)
        return action

