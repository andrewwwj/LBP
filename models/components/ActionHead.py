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
        guidance_mode: str="cg"
    ):
        super().__init__()
        self.mode = guidance_mode
        self.time_dim = time_dim
        self.num_timesteps = num_timesteps
        self.time_hidden_dim = time_hidden_dim
        self.action_size = action_size
        self.time_process = LearnedPosEmb(1, time_dim)
        self.time_encoder = Mlp(time_dim, time_hidden_dim, time_hidden_dim, norm_layer=nn.LayerNorm)
        self.model = MlpResNet(num_blocks=num_blocks,
                               input_dim=input_dim + action_size + time_hidden_dim,
                               hidden_dim=hidden_dim, output_size=action_size)
        if self.mode == 'cg':
            pass
        elif self.mode == 'energy':
            critic_input_dim = input_dim + action_size + time_hidden_dim
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
                nn.Linear(input_dim + action_size, hidden_dim),
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
            self.alpha = 3.0  # Temperature parameter for softmax
            self.guidance_scale = 1.0  # Guidance strength
            self.num_samples = 16  # Number of samples for CEP loss
            self.tau = 0.005  # Soft update rate for target network
        elif self.mode == 'cfg':
            self.cfg_prob = 0.1
            self.w_cfg = 2.0
            self.uncond_embedding = nn.Parameter(torch.randn(1, input_dim))
        else:
            raise NotImplementedError (f"Unknown mode {self.mode}")
        self.init_ddpm()

    def init_ddpm(self, device='cuda'):
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(device))
        if self.mode == 'energy':
            self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
            self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).to(device))

    def forward_features(self, all_obs, t_tensor, noise_action):
        time_embedding = self.time_process(t_tensor.view(-1, 1))
        time_embedding = self.time_encoder(time_embedding)
        input_feature = torch.cat([all_obs, time_embedding, noise_action], dim=-1)
        noise_pred = self.model(input_feature)
        return noise_pred

    def forward(self, all_obs, cur_action):
        noise, t_tensor, noise_action = self.add_noise(cur_action)
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
            # return noise, t_tensor, noise_action, noise_pred
            diffusion_loss = F.mse_loss(noise_pred, noise)
            return diffusion_loss
        elif self.mode == 'energy':
            noise_pred = self.forward_features(all_obs, t_tensor, noise_action)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            cep_loss = self.compute_cep_loss(all_obs, cur_action)
            total_loss = diffusion_loss + self.critic_loss_weight * cep_loss
            return total_loss
        else:
            raise NotImplementedError

    def generate(self, all_obs):
        shape = (all_obs.shape[0], self.action_size)
        action = torch.randn(shape, device=all_obs.device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0], 1), t, device=all_obs.device)
            if self.mode == 'cg':
                noise_pred = self.forward_features(all_obs, t_tensor, action)
                action = self.denoise(action, t_tensor, noise_pred)
            elif self.mode == 'cfg':
                # 1. Get unconditional prediction by feeding the unconditional embedding
                uncond_obs = self.uncond_embedding.repeat(shape[0], 1)
                noise_pred_uncond = self.forward_features(uncond_obs, t_tensor, action)
                # 2. Get conditional prediction by feeding the actual observation
                noise_pred_cond = self.forward_features(all_obs, t_tensor, action)
                # 3. Combine predictions using the CFG formula
                # This guides the generation towards the condition
                noise_pred = noise_pred_uncond + self.w_cfg * (noise_pred_cond - noise_pred_uncond)
                action = self.denoise(action, t_tensor, noise_pred)
            elif self.mode == 'energy':
                noise_pred = self.forward_features(all_obs, t_tensor, action)
                action = self.denoise(action, t_tensor, noise_pred)
                if t > 1:
                    guidance = self.calculate_energy_guidance(action, t, all_obs)
                    action = action + guidance
                    action = torch.clamp(action, -1., 1.)
            else:
                raise NotImplementedError
        return action

    def calculate_energy_guidance(self, actions, t, condition):
        """Calculate energy gradient for CEP guidance during generation"""
        if not hasattr(self, 'qt_network'):
            return torch.zeros_like(actions)

        with torch.enable_grad():
            actions = actions.detach().requires_grad_(True)

            # Encode time
            t_emb = self.time_process(t.reshape(-1, 1).to(torch.float32) / self.num_timesteps)
            t_emb = self.time_encoder(t_emb)

            # Prepare input for Q_t network
            critic_input = torch.cat([condition, actions, t_emb], dim=-1)

            # Get Q-values
            q_values = self.qt_network(critic_input)

            # Compute gradient
            guidance = self.guidance_scale * torch.autograd.grad(
                torch.sum(q_values), actions, create_graph=False
            )[0]

        return guidance.detach()

    def compute_cep_loss(self, all_obs, cur_action):
        """Compute CEP loss for training Q_t network"""
        if not hasattr(self, 'qt_network'):
            return torch.zeros(1, device=all_obs.device)

        B = all_obs.shape[0]
        device = all_obs.device

        # Sample multiple candidate actions for contrastive learning
        with torch.no_grad():
            # Use the current batch action as one sample and generate others
            candidate_actions = [cur_action]
            for _ in range(self.num_samples - 1):
                # Generate actions from random noise (fast approximation)
                noise_action = torch.randn(B, self.action_size, device=device)
                candidate_actions.append(torch.clamp(noise_action, -1., 1.))
            candidate_actions = torch.stack(candidate_actions, dim=1)  # [B, num_samples, action_dim]

            # Compute target energy distribution using Q_0 target network
            obs_expanded = all_obs.unsqueeze(1).expand(-1, self.num_samples, -1)
            obs_actions = torch.cat([obs_expanded, candidate_actions], dim=-1)
            obs_actions_flat = obs_actions.reshape(B * self.num_samples, -1)

            target_q_values = self.q0_target(obs_actions_flat)
            target_q_values = target_q_values.reshape(B, self.num_samples)

            # Create target distribution with temperature-scaled softmax
            target_probs = F.softmax(self.alpha * target_q_values, dim=-1)

        # Sample random timestep for each sample
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=device)

        # Add noise to candidate actions
        noise = torch.randn_like(candidate_actions)
        sqrt_alpha = self.sqrt_alphas_cumprod[t - 1].reshape(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t - 1].reshape(B, 1, 1)
        noisy_actions = sqrt_alpha * candidate_actions + sqrt_one_minus_alpha * noise

        # Encode time for Q_t
        t_expanded = t.unsqueeze(1).expand(-1, self.num_samples)
        t_flat = t_expanded.reshape(-1)
        t_emb = self.time_process(t_flat.reshape(-1, 1).to(torch.float32) / self.num_timesteps)
        t_emb = self.time_encoder(t_emb)

        # Prepare input for Q_t
        noisy_actions_flat = noisy_actions.reshape(B * self.num_samples, -1)
        obs_expanded_flat = obs_expanded.reshape(B * self.num_samples, -1)
        critic_input = torch.cat([obs_expanded_flat, noisy_actions_flat, t_emb], dim=-1)

        # Get Q_t predictions
        pred_q_values = self.qt_network(critic_input)
        pred_q_values = pred_q_values.reshape(B, self.num_samples)

        # Compute CEP loss (cross-entropy between target and predicted distributions)
        pred_log_probs = F.log_softmax(self.alpha * pred_q_values, dim=-1)
        cep_loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=-1))

        # Update target network with soft update
        self.soft_update_target_network()

        return cep_loss

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

    def denoise(self, x, t_tensor, noise):
        x = self.p_sample(x, t_tensor, noise)
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

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor, guidance_strength=0, clip_sample=True, ddpm_temperature=1.):
        """
        sample xt-1 from xt, p(xt-1|xt)

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