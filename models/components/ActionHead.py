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
            input_dim: int = 544,
            hidden_dim: int = 256,
            time_dim: int = 32,
            num_timesteps: int = 25,
            time_hidden_dim: int = 256,
            guidance_mode: str = "cg",
            proprio_dim: int = None,
            latent_goal_dim: int = None,
            vis_lang_dim: int = None,
            diffusion_input_dim: int = None,  # Dimension of proprioceptive observations for diffusion
            energy_input_dim: int = None,  # Dimension of vision+language for energy
            use_separate_condition: bool = False
    ):
        super().__init__()
        self.mode = guidance_mode
        self.time_dim = time_dim
        self.num_timesteps = num_timesteps
        self.time_hidden_dim = time_hidden_dim
        self.action_size = action_size
        self.time_process = LearnedPosEmb(1, time_dim)
        self.time_encoder = Mlp(time_dim, time_hidden_dim, time_hidden_dim, norm_layer=nn.LayerNorm)
        # self.model = MlpResNet(num_blocks=num_blocks,
        #                        input_dim=input_dim + action_size + time_hidden_dim,
        #                        hidden_dim=hidden_dim, output_size=action_size)
        self.use_separate_condition = use_separate_condition
        combination_dict = {'v': vis_lang_dim, 'p': proprio_dim, 'g': latent_goal_dim,
                            'vp': vis_lang_dim + proprio_dim, 'vg': vis_lang_dim + latent_goal_dim,
                            'pg': proprio_dim + latent_goal_dim, 'vpg': vis_lang_dim + proprio_dim + latent_goal_dim
                            }
        self.diffusion_input_key = 'vpg'
        self.energy_input_key = 'vg'

        if self.use_separate_condition:
            self.diffusion_input_dim = combination_dict[self.diffusion_input_key]
            self.energy_input_dim = combination_dict[self.energy_input_key]
            if self.mode == 'energy':
                self.model = MlpResNet(num_blocks=num_blocks,
                                       input_dim=self.diffusion_input_dim + action_size + time_hidden_dim,
                                       hidden_dim=hidden_dim,
                                       output_size=action_size)
            else:
                self.model = MlpResNet(num_blocks=num_blocks,
                                       input_dim=input_dim + action_size + time_hidden_dim,
                                       hidden_dim=hidden_dim,
                                       output_size=action_size)
            critic_input_dim = self.energy_input_dim + action_size + time_hidden_dim
        else:
            self.diffusion_input_dim = input_dim
            self.energy_input_dim = input_dim
            self.model = MlpResNet(num_blocks=num_blocks,
                                   input_dim=input_dim + action_size + time_hidden_dim,
                                   hidden_dim=hidden_dim,
                                   output_size=action_size)
            critic_input_dim = input_dim + action_size + time_hidden_dim
        if self.mode == 'cg':
            pass
        elif self.mode == 'energy':
            # Q_t network for time-dependent energy prediction
            self.qt_network = nn.Sequential(
                nn.Linear(critic_input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Add layer norm for stability
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            # Q_0 network for clean action evaluation
            q0_input_dim = self.energy_input_dim + action_size
            self.q0_network = nn.Sequential(
                nn.Linear(q0_input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            # Target network for stable CEP training
            self.q0_target = copy.deepcopy(self.q0_network)
            for param in self.q0_target.parameters():
                param.requires_grad = False
            # CEP hyperparameters
            self.w_critic_loss = 1.0
            self.alpha = 3.0  # Temperature parameter for softmax
            self.guidance_scale = 10.0  # Guidance strength
            self.num_samples = 128  # Number of samples for CEP loss
            self.tau = 0.005  # Soft update rate for target network
            self.use_gradient_penalty = True
            self.grad_penalty_weight = 0.1
        elif self.mode == 'cfg':
            self.cfg_prob = 0.1
            self.w_cfg = 2.0
            self.uncond_embedding = nn.Parameter(torch.randn(1, vis_lang_dim))
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
        noise, t_tensor, noise_action = self.add_noise(cur_action)
        vl_semantics, proprio_obs, fused_goal = all_obs
        modality_dict = {'v': vl_semantics, 'p': proprio_obs, 'g': fused_goal, }
        diffusion_obs = torch.cat([modality_dict[key] for key in self.diffusion_input_key], dim=-1)
        energy_obs = torch.cat([modality_dict[key] for key in self.energy_input_key], dim=-1)
        if self.mode == 'cg':
            noise_pred = self.forward_features(diffusion_obs, t_tensor, noise_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            return diffusion_loss
        elif self.mode == 'cfg':
            """0) all_obs"""
            # uncond_mask = torch.rand(all_obs.shape[0], device=all_obs.device) < self.cfg_prob
            # all_obs_ = all_obs.clone()
            # Replace the observation with the unconditional embedding for the masked samples
            # all_obs_[uncond_mask] = self.uncond_embedding
            """1) vl_semantics"""
            obs_ = vl_semantics.clone()
            uncond_mask = torch.rand(obs_.shape[0], device=obs_.device) < self.cfg_prob
            obs_[uncond_mask] = self.uncond_embedding
            all_obs_ = torch.cat([obs_, proprio_obs, fused_goal], dim=-1)
            """2) vl_semantics + fused_goal"""
            # obs_ = torch.cat([vl_semantics, fused_goal], dim=-1).clone()
            # uncond_mask = torch.rand(obs_.shape[0], device=obs_.device) < self.cfg_prob
            # obs_[uncond_mask] = self.uncond_embedding
            # all_obs_ = torch.cat([obs_, proprio_obs], dim=-1)
            """Common"""
            noise_pred = self.forward_features(all_obs_, t_tensor, noise_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            return diffusion_loss
        elif self.mode == 'energy':
            # Train diffusion model
            noise_pred = self.forward_features(diffusion_obs, t_tensor, noise_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)

            # Compute CEP loss for Q_t network training
            cep_loss = self.compute_cep_loss(vl_semantics, proprio_obs, fused_goal, cur_action)

            # Combine losses
            total_loss = diffusion_loss + self.w_critic_loss * cep_loss
            return total_loss
        else:
            raise NotImplementedError

    def generate(self, all_obs):
        vl_semantics, proprio_obs, fused_goal = all_obs
        modality_dict = {'v': vl_semantics, 'p': proprio_obs, 'g': fused_goal, }
        diffusion_obs = torch.cat([modality_dict[key] for key in self.diffusion_input_key], dim=-1)
        energy_obs = torch.cat([modality_dict[key] for key in self.energy_input_key], dim=-1)
        shape = (diffusion_obs.shape[0], self.action_size)
        action = torch.randn(shape, device=diffusion_obs.device)
        for step in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0], 1), step, device=diffusion_obs.device, dtype=torch.long)
            if self.mode == 'cg':
                noise_pred = self.forward_features(diffusion_obs, t_tensor, action)
                action = self.denoise(action, t_tensor, noise_pred)
            elif self.mode == 'cfg':
                """0) all_obs"""
                # uncond_shape = diffusion_obs.shape[0]
                # uncond_obs = self.uncond_embedding.repeat(uncond_shape[0], 1)
                # cond_obs = diffusion_obs.clone()
                """1) vl_semantics"""
                uncond_shape = vl_semantics.shape[0]
                uncond_obs = torch.cat([self.uncond_embedding.repeat(uncond_shape, 1), proprio_obs, fused_goal], dim=-1)
                cond_obs = torch.cat([vl_semantics, proprio_obs, fused_goal], dim=-1)
                """2) vl_semantics + fused goal"""
                # uncond_shape = torch.cat([vl_semantics, fused_goal], dim=-1).shape[0]
                # uncond_obs = torch.cat([self.uncond_embedding.repeat(uncond_shape, 1), proprio_obs], dim=-1)
                # cond_obs = torch.cat([vl_semantics, fused_goal, proprio_obs], dim=-1)
                """Common"""
                noise_pred_uncond = self.forward_features(uncond_obs, t_tensor, action)
                noise_pred_cond = self.forward_features(cond_obs, t_tensor, action)
                noise_pred = noise_pred_uncond + self.w_cfg * (noise_pred_cond - noise_pred_uncond)
                action = self.denoise(action, t_tensor, noise_pred)
            elif self.mode == 'energy':
                # Diffusion denoising with proprio conditioning
                noise_pred = self.forward_features(diffusion_obs, t_tensor, action)
                action = self.denoise(action, t_tensor, noise_pred)
                # Energy guidance
                if step < int(self.num_timesteps * 0.2):
                    guidance = self.calculate_energy_guidance(action, t_tensor, energy_obs)
                    action = action + guidance
            else:
                raise NotImplementedError
        # action = action.clamp(-1., 1.)
        return action

    def calculate_energy_guidance(self, actions, t, condition):
        """
        Calculate energy gradient for CEP guidance during generation.
        Uses vision+language conditioning for energy model.

        Key improvements for OOD robustness:
        1. Adaptive guidance scale based on timestep
        2. Gradient clipping for stability
        3. Vision-language focused conditioning
        """
        with torch.enable_grad():
            actions = actions.detach().requires_grad_(True)
            # Encode time
            t_emb = self.time_process(t.reshape(-1, 1).to(torch.float32) / self.num_timesteps)
            t_emb = self.time_encoder(t_emb)
            # Q_t network with vision+language conditioning
            q_values = self.qt_network(torch.cat([condition, actions, t_emb], dim=-1))
            # # Adaptive guidance scale based on timestep
            timestep_ratio = t.float().mean() / self.num_timesteps
            guidance_scale = self.guidance_scale * (2.0 - 1.0 * timestep_ratio)
            # Compute gradient with respect to actions
            grad = torch.autograd.grad(torch.sum(q_values), actions, create_graph=False)[0]
            guidance = guidance_scale * grad
            # Gradient clipping for stability in OOD scenarios
            guidance = torch.clamp(guidance, -0.1, 0.1)
        return guidance.detach()

    def compute_cep_loss(self, vl_semantics, proprio_obs, fused_goal, cur_action):
        """
                Compute CEP loss for training Q_t network with OOD-aware sampling.
        """
        all_obs = torch.cat([vl_semantics, proprio_obs, fused_goal], dim=-1)
        B = all_obs.shape[0]
        # B = semantic_obs.shape[0]

        # Sample multiple candidate actions using current diffusion forward
        with torch.no_grad():
            candidate_actions = [cur_action]
            for i in range(self.num_samples - 1):
                # Generate actions by adding noise via diffusion forward process
                # 1) Random timestep sampling
                # timestep = torch.randint(0, int(self.num_timesteps * 0.2), (B,), device=cur_action.device).long()
                # timestep = torch.randint(0, self.num_timesteps, (B,), device=cur_action.device).long()
                # 2) Diverse timestep sampling for varied difficulty
                if i < self.num_samples // 3:
                    # 1/3 samples: Low noise (t in [1, 10]) - hard negatives
                    # These are most similar to actual OOD actions
                    timestep = torch.randint(0, self.num_timesteps//4, (B,), device=cur_action.device).long()
                elif i < self.num_samples // 3 * 2:
                    # 1/3 samples: Medium noise (t in [10, 50]) - medium negatives
                    timestep = torch.randint(self.num_timesteps//4, self.num_timesteps//2, (B,), device=cur_action.device).long()
                else:
                    # 1/3 samples: High noise (t in [50, num_timesteps-5]) - easy negatives
                    timestep = torch.randint(self.num_timesteps//2, self.num_timesteps, (B,), device=cur_action.device).long()
                # Add noise to the current action
                noise = torch.randn_like(cur_action)
                # Apply forward process
                noise_action = self.q_sample(x0=cur_action, t=timestep, noise=noise)
                candidate_actions.append(torch.clamp(noise_action, -1., 1.))
            candidate_actions = torch.stack(candidate_actions, dim=1)  # [B, num_samples, action_dim]

            # Compute target energy distribution using Q_0 target network
            if self.use_separate_condition:
                energy_obs_for_q0 = torch.cat([modality for key, modality in
                                              zip(['v', 'p', 'g'], [vl_semantics, proprio_obs, fused_goal])
                                              if key in self.energy_input_key], dim=-1)
                obs_expanded = energy_obs_for_q0.unsqueeze(1).expand(-1, self.num_samples, -1)
            else:
                # Fallback to all observations
                obs_expanded = all_obs.unsqueeze(1).expand(-1, self.num_samples, -1)
            obs_actions = torch.cat([obs_expanded, candidate_actions], dim=-1)  # [B, num_samples, obs_dim + action_dim]
            obs_actions_flat = obs_actions.reshape(B * self.num_samples, -1)

            target_q_values = self.q0_target(obs_actions_flat)
            target_q_values = target_q_values.reshape(B, self.num_samples)

            # Create target distribution with temperature-scaled softmax
            target_probs = F.softmax(self.alpha * target_q_values, dim=-1)

        # Sample random timestep for each sample
        t = torch.randint(0, self.num_timesteps, (B,), device=all_obs.device)

        # Add noise to candidate actions
        noise = torch.randn_like(candidate_actions)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].reshape(B, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, 1, 1)
        noisy_actions = sqrt_alpha_cumprod * candidate_actions + sqrt_one_minus_alpha_cumprod * noise

        # Compute predicted energy using Q_t network
        t_expanded = t.unsqueeze(1).expand(-1, self.num_samples)  # [B, num_samples]
        t_flat = t_expanded.reshape(-1)  # [B * num_samples]

        # Encode time
        t_emb = self.time_process(t_flat.reshape(-1, 1).to(torch.float32) / self.num_timesteps)
        t_emb = self.time_encoder(t_emb)

        # Prepare input for Q_t
        noisy_actions_flat = noisy_actions.reshape(B * self.num_samples, -1)
        if self.use_separate_condition:
            obs_dim = obs_expanded.shape[-1]
            critic_input = torch.cat([obs_actions_flat[:, :obs_dim], noisy_actions_flat, t_emb], dim=-1)
        else:
            critic_input = torch.cat([obs_actions_flat[:, :all_obs.shape[-1]], noisy_actions_flat, t_emb], dim=-1)

        # Get Q_t predictions
        pred_q_values = self.qt_network(critic_input)
        pred_q_values = pred_q_values.reshape(B, self.num_samples)

        # Compute predicted distribution
        pred_log_probs = F.log_softmax(self.alpha * pred_q_values, dim=-1)

        # Compute CEP loss (cross-entropy between target and predicted distributions)
        cep_loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=-1))
        entropy_reg = -torch.mean(torch.sum(target_probs * torch.log(target_probs + 1e-8), dim=-1))
        cep_loss = cep_loss - 0.01 * entropy_reg

        # Update target network with soft update
        self.soft_update_target_network()

        # # Optional: Add Q_0 loss for training Q_0 network
        # if hasattr(self, 'q0_network'):
        #     # Use current batch of actions for Q_0 training
        #     with torch.no_grad():
        #         clean_actions = self.generate(all_obs, device=all_obs.device)
        #
        #     q0_input = torch.cat([all_obs, clean_actions], dim=-1)
        #     q0_values = self.q0_network(q0_input)
        #
        #     # Simple reward signal (you may want to replace with actual rewards if available)
        #     # For now, using a placeholder reward
        #     target_values = torch.zeros_like(q0_values)  # Replace with actual rewards
        #     q0_loss = F.mse_loss(q0_values, target_values)
        #
        #     cep_loss = cep_loss + 0.1 * q0_loss  # Weight the Q_0 loss

        return cep_loss

    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        if hasattr(self, 'q0_target') and hasattr(self, 'q0_network'):
            for target_param, param in zip(self.q0_target.parameters(), self.q0_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_noise(self, x0: torch.Tensor):
        B = x0.shape[0]
        noise = torch.randn_like(x0, device=x0.device)
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        xt = self.q_sample(x0, t, noise)
        return noise, t, xt

    def denoise(self, x, t_tensor, noise, clip_sample=True):
        t_tensor = t_tensor.view(-1, 1)
        x = self.p_sample(x, t_tensor, noise, clip_sample)
        return x

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        sample noisy xt from x0, q(xt|x0), forward process

        Input:
        x0: ground truth value, here x0 is typically the ground truth action
        t: timestep
        noise: noise

        Return: noisy samples
        """
        alphas_cumprod_t = self.alphas_cumprod[t]
        xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
        return xt

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor, clip_sample=True,
                 ddpm_temperature=1.):
        """
        sample xt-1 from xt, p(xt-1|xt), reverse process

        Input:
        xt: noisy samples, here xt is typically the noisy ground truth action
        t: timestep
        noise_pred: predicted noise
        guidance_strength: the strength of the guidance
        clip_sample: whether to clip the sample to [-1, 1]
        ddpm_temperature: the temperature of the noise

        Return: sample xt-1
        """
        alpha1 = 1 / torch.sqrt(self.alphas[t])
        alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
        xtm1 = alpha1 * (xt - alpha2 * noise_pred)
        noise = torch.randn_like(xtm1, device=xt.device) * ddpm_temperature
        xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
        if clip_sample:
            xtm1 = torch.clip(xtm1, -1., 1.)
        return xtm1


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


def extract(a, x_shape):
    '''
    align the dimention of alphas_cumprod_t to x_shape

    a: alphas_cumprod_t, B
    x_shape: B x F x F x F
    output: alphas_cumprod_t B x 1 x 1 x 1]
    '''
    b, *_ = a.shape
    return a.reshape(b, *((1,) * (len(x_shape) - 1)))