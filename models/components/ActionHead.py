
from timm.layers import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
            self.critic = Mlp(in_features=critic_input_dim, hidden_features=hidden_dim, out_features=1)
        elif self.mode == 'cfg':
            self.cfg_prob = 0.1
            self.uncond_embedding = nn.Parameter(torch.randn(1, input_dim))
        else:
            raise NotImplementedError (f"Unknown mode {self.mode}")
        self.init_ddpm()

    def init_ddpm(self, device='cuda'):
        self.betas = cosine_beta_schedule(self.num_timesteps).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

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
            critic_loss_val = self.critic_loss(all_obs)
            total_loss = diffusion_loss + self.critic_loss_weight * critic_loss_val
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
            elif self.mode == 'cfg':
                guidance_scale = 0.5
                # 1. Get unconditional prediction by feeding the unconditional embedding
                uncond_obs = self.uncond_embedding.repeat(shape[0], 1)
                noise_pred_uncond = self.forward_features(uncond_obs, t_tensor, action)
                # 2. Get conditional prediction by feeding the actual observation
                noise_pred_cond = self.forward_features(all_obs, t_tensor, action)
                # 3. Combine predictions using the CFG formula
                # This guides the generation towards the condition
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            elif self.mode == 'energy':
                noise_pred = self.forward_features(all_obs, t_tensor, action)
                guidance_scale = 2.0  # This can be tuned
                with torch.enable_grad():
                    action.requires_grad_(True)
                    time_embedding = self.time_process(t_tensor.view(-1, 1))
                    time_embedding = self.time_encoder(time_embedding)
                    critic_input = torch.cat([all_obs, time_embedding, action], dim=-1)
                    energy = self.critic(critic_input)
                    grad = torch.autograd.grad(torch.sum(energy), action)[0]
                # Guide the noise prediction with the correctly scaled gradient
                # alphas_cumprod_t = self.alphas_cumprod[t_tensor.long()].view(-1, 1)
                # scale_factor = torch.sqrt(1. - alphas_cumprod_t)
                noise_pred = noise_pred + (guidance_scale - noise_pred) * grad
            else:
                raise NotImplementedError
            action = self.denoise(action, t_tensor, noise_pred)
        return action

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

    def critic_loss(self, all_obs, num_samples=16, alpha=3.0):
        # 1. Generate candidate actions for the current observation
        # The outer no_grad is for the diffusion model parts of generate
        with torch.no_grad():
            # Repeat observation for each sample
            obs_repeated = all_obs.repeat_interleave(num_samples, dim=0)
            # Generate actions, enabling grad for critic guidance internally
            candidate_actions = self.generate(obs_repeated)
            candidate_actions = candidate_actions.view(all_obs.shape[0], num_samples, -1)
        # 2. Calculate target energy distribution (p_label)
        with torch.no_grad():
            # Get time embeddings for t=0 (no noise)
            t_tensor_zeros = torch.zeros((all_obs.shape[0] * num_samples, 1), device=all_obs.device)
            time_embedding = self.time_process(t_tensor_zeros)
            time_embedding = self.time_encoder(time_embedding)
            time_embedding = time_embedding.view(all_obs.shape[0], num_samples, -1)
            # Prepare input for critic
            critic_input = torch.cat([
                all_obs.unsqueeze(1).repeat(1, num_samples, 1),
                time_embedding,
                candidate_actions
            ], dim=-1)
            # Calculate energy and target distribution
            energy = self.critic(critic_input.view(-1, critic_input.shape[-1])).view(all_obs.shape[0], num_samples)
            p_label = F.softmax(energy * alpha, dim=-1)
        # 3. Perturb actions and calculate model's energy prediction
        # Add noise to candidate actions
        noise, t_tensor, noisy_actions = self.add_noise(candidate_actions.view(-1, self.action_size))
        t_tensor = t_tensor.view(all_obs.shape[0], num_samples)
        noisy_actions = noisy_actions.view(all_obs.shape[0], num_samples, -1)
        # Get time embeddings for the random timesteps
        time_embedding = self.time_process(t_tensor.view(-1, 1))
        time_embedding = self.time_encoder(time_embedding)
        time_embedding = time_embedding.view(all_obs.shape[0], num_samples, -1)
        # Prepare input for critic with noisy actions
        critic_input_noisy = torch.cat([
            all_obs.unsqueeze(1).repeat(1, num_samples, 1),
            time_embedding,
            noisy_actions
        ], dim=-1)
        # Calculate model's energy prediction
        xt_model_energy = self.critic(critic_input_noisy.view(-1, critic_input_noisy.shape[-1])).view(all_obs.shape[0], num_samples)
        # 4. Calculate CEP loss
        loss = -torch.mean(torch.sum(p_label * F.log_softmax(xt_model_energy, dim=-1), dim=-1))
        return loss


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