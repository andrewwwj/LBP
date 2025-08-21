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
        num_blocks:int = 3,
        input_dim:int = 544,
        hidden_dim: int = 256,
        time_dim: int = 32,
        num_timesteps: int = 25,
        time_hidden_dim: int=256,
        guidance_mode: str="cg",
        proprio_dim: int=None,
        latent_goal_dim: int=None,
        vis_lang_dim: int=None,
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
        if use_separate_condition and self.mode == 'energy':
            # Diffusion model uses proprioceptive observations
            # TODO change this part for energy guidance
            self.diffusion_input_dim = diffusion_input_dim
            # Energy model uses vision+language embeddings
            # TODO change this part for energy guidance
            self.energy_input_dim = energy_input_dim
            # Diffusion model with proprio conditioning
            self.model = MlpResNet(num_blocks=num_blocks,
                                   input_dim=self.diffusion_input_dim + action_size + time_hidden_dim,
                                   hidden_dim=hidden_dim,
                                   output_size=action_size)

            # Energy networks with vision+language conditioning
            # TODO change this part for energy guidance
            critic_input_dim = vis_lang_dim + latent_goal_dim + action_size + time_hidden_dim
        else:
            # Standard setup without separated conditioning
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
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            # Q_0 network for clean action evaluation
            self.q0_network = nn.Sequential(
                nn.Linear(self.energy_input_dim + action_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
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
            self.guidance_scale = 1.0  # Guidance strength
            self.num_samples = 16  # Number of samples for CEP loss
            self.tau = 0.005  # Soft update rate for target network
        elif self.mode == 'cfg':
            self.cfg_prob = 0.1
            self.w_cfg = 3.0
            self.uncond_embedding = nn.Parameter(torch.randn(1, input_dim))
        else:
            raise NotImplementedError (f"Unknown mode {self.mode}")
        self.init_ddpm()

    def init_ddpm(self, device='cuda'):
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))

    def forward_features(self, diffusion_obs, t_tensor, noise_action):
        time_embedding = self.time_process(t_tensor.view(-1, 1))
        time_embedding = self.time_encoder(time_embedding)
        input_feature = torch.cat([diffusion_obs, time_embedding, noise_action], dim=-1)
        noise_pred = self.model(input_feature)
        return noise_pred

    def forward(self, all_obs, cur_action):
        noise, t_tensor, noise_action = self.add_noise(cur_action)
        if isinstance(all_obs, tuple):
            diffusion_obs, energy_obs = all_obs
        else:
            diffusion_obs = all_obs
            energy_obs = all_obs
        if self.mode == 'cg':
            noise_pred = self.forward_features(all_obs, t_tensor, noise_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            return diffusion_loss
        elif self.mode == 'cfg':
            uncond_mask = torch.rand(all_obs.shape[0], device=all_obs.device) < self.cfg_prob
            obs_for_pred = all_obs.clone()
            # Replace the observation with the unconditional embedding for the masked samples
            obs_for_pred[uncond_mask] = self.uncond_embedding
            noise_pred = self.forward_features(obs_for_pred, t_tensor, noise_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            return diffusion_loss
        elif self.mode == 'energy':
            # Diffusion uses proprio conditioning
            noise_pred = self.forward_features(diffusion_obs, t_tensor, noise_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            # Energy uses vision+language conditioning
            energy_loss = self.compute_energy_loss(energy_obs, cur_action, noise_action, t_tensor, noise_pred.detach())
            total_loss = diffusion_loss + self.w_critic_loss * energy_loss
            return total_loss
        else:
            raise NotImplementedError

    def generate(self, all_obs):
        if isinstance(all_obs, tuple):
            diffusion_obs, energy_obs = all_obs
            shape = (diffusion_obs.shape[0], self.action_size)
        else:
            diffusion_obs = all_obs
            energy_obs = all_obs
            shape = (all_obs.shape[0], self.action_size)
        action = torch.randn(shape, device=diffusion_obs.device)
        for step in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0], 1), step, device=diffusion_obs.device, dtype=torch.long)
            if self.mode == 'cg':
                noise_pred = self.forward_features(diffusion_obs, t_tensor, action)
                action = self.denoise(action, t_tensor, noise_pred)
            elif self.mode == 'cfg':
                uncond_obs = self.uncond_embedding.repeat(shape[0], 1)
                noise_pred_uncond = self.forward_features(uncond_obs, t_tensor, action)
                noise_pred_obs = self.forward_features(diffusion_obs, t_tensor, action)
                noise_pred = noise_pred_uncond + self.w_cfg * (noise_pred_obs - noise_pred_uncond)
                action = self.denoise(action, t_tensor, noise_pred)
            elif self.mode == 'energy':
                # Diffusion denoising with proprio conditioning
                noise_pred = self.forward_features(diffusion_obs, t_tensor, action)
                action = self.denoise(action, t_tensor, noise_pred)
                # Energy guidance with vision+language conditioning
                guidance = self.calculate_energy_guidance(action, t_tensor, energy_obs)
                # Apply guidance
                action = action + guidance
            else:
                raise NotImplementedError
        action = torch.clamp(action, -1., 1.)
        return action

    def calculate_energy_guidance(self, actions, t, condition):
        """
        Calculate energy gradient for CEP guidance during generation.
        Uses vision+language conditioning for energy model.
        """
        if not hasattr(self, 'qt_network'):
            return torch.zeros_like(actions)

        with torch.enable_grad():
            actions = actions.detach().requires_grad_(True)
            # Get Q-values
            q_values = self.qt_network(torch.cat([condition, actions], dim=-1))
            # Compute gradient
            guidance_scale = 1.0
            guidance = guidance_scale * torch.autograd.grad(torch.sum(q_values), actions, create_graph=False)[0]

        return guidance.detach()

    def compute_energy_loss(self, energy_obs, cur_action, noise_action, t_tensor, noise_pred):
        # 1. Sample N candidate actions from the diffusion model by denoising
        with torch.no_grad():
            candidate_actions = []
            for _ in range(self.num_samples):
                noise = torch.randn_like(cur_action)
                perturbed_action = self.q_sample(cur_action, t_tensor, noise)
                candidate_actions.append(torch.clamp(perturbed_action, -1., 1.))
            candidate_actions = torch.stack(candidate_actions, dim=1)  # [B, num_samples, action_dim]

            # 2. Compute target Q-values using the target network
            obs_expanded = energy_obs.unsqueeze(1).expand(-1, self.num_samples, -1)
            obs_actions = torch.cat([obs_expanded, candidate_actions], dim=-1)
            target_q_values = self.q0_target(obs_actions.view(-1, obs_actions.shape[-1])).view(energy_obs.shape[0], self.num_samples)
            target_dist = F.softmax(-target_q_values / self.alpha, dim=-1)

        # Compute current energy distribution for noisy actions
        t_emb = self.time_process(t_tensor.view(-1, 1).to(torch.float32) / self.num_timesteps)
        t_emb = self.time_encoder(t_emb)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, self.num_samples, -1)
        obs_actions_t = torch.cat([obs_expanded, candidate_actions, t_emb_expanded], dim=-1)
        q_t_values = self.qt_network(obs_actions_t.view(-1, obs_actions_t.shape[-1])).view(energy_obs.shape[0], self.num_samples)
        log_pred_dist = F.log_softmax(-q_t_values / self.alpha, dim=-1)

        # Ensure KL divergence is non-negative by using proper log probabilities
        cep_loss = F.kl_div(log_pred_dist, target_dist, reduction='batchmean')

        # --- Diffusion-Guided Energy (DGE) Loss ---
        denoised_action = self.denoise(noise_action, t_tensor, noise_pred, clip_sample=True).detach()
        contrast_actions = torch.cat([denoised_action.unsqueeze(1), candidate_actions], dim=1)
        num_contrastive_samples = contrast_actions.shape[1]
        obs_expanded_contrast = energy_obs.unsqueeze(1).expand(-1, num_contrastive_samples, -1)
        t_emb_expanded_contrast = t_emb.unsqueeze(1).expand(-1, num_contrastive_samples, -1)
        contrast_obs_actions = torch.cat([obs_expanded_contrast, contrast_actions, t_emb_expanded_contrast], dim=-1)
        contrast_q_values = self.qt_network(contrast_obs_actions.view(-1, contrast_obs_actions.shape[-1])).view(energy_obs.shape[0], num_contrastive_samples)
        labels = torch.zeros(energy_obs.shape[0], dtype=torch.long, device=energy_obs.device)
        dge_loss = F.cross_entropy(contrast_q_values, labels)
        # Combine losses (dge_loss_weight can be tuned)
        loss = cep_loss + 0.1 * dge_loss
        # Update target network with soft update
        self.soft_update_target_network()
        return loss

    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        if hasattr(self, 'q0_target') and hasattr(self, 'q0_network'):
            for target_param, param in zip(self.q0_target.parameters(), self.q0_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_noise(self, x0: torch.Tensor):
        B = x0.shape[0]
        noise = torch.randn_like(x0, device=x0.device)
        t = torch.randint(0, self.num_timesteps, (B, ), device=x0.device)
        xt = self.q_sample(x0, t, noise)
        return noise, t, xt

    def denoise(self, x, t_tensor, noise, clip_sample=False):
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

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor, clip_sample=True, ddpm_temperature=1.):
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
        num_blocks:int = 3,
        input_dim:int = 544,
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