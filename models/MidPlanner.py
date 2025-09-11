from timm.layers import Mlp, DropPath, to_2tuple, trunc_normal_
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import DecisionNCE
import random
import pickle


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class DualPathPredictor(nn.Module):
    def __init__(self,
                 latent_dim=1024,
                 output_dim=1024):
        super().__init__()
        self.predictor = Mlp(in_features=latent_dim * 2, hidden_features=latent_dim * 4, out_features=output_dim,
                             norm_layer=nn.LayerNorm)
        self.apply(init_weight)

    def forward(self, s0, s1):
        s = torch.cat([s0, s1], dim=-1)
        pred = self.predictor(s)
        return pred


class TriplePathPredictor(nn.Module):
    def __init__(self,
                 latent_dim=1024,
                 output_dim=1024):
        super().__init__()
        self.predictor = Mlp(in_features=latent_dim * 3,
                             hidden_features=latent_dim * 4,
                             out_features=output_dim,
                             norm_layer=nn.LayerNorm)
        self.apply(init_weight)

    def forward(self, s0, s1, s2):
        s = torch.cat([s0, s1, s2], dim=-1)
        pred = self.predictor(s)
        return pred


class DnceLatentProj(nn.Module):
    def __init__(self, latent_info_file='assets/libero.pkl'):
        super().__init__()
        self.latent_proj = DecisionNCE.load("DecisionNCE-T", device="cuda")  # Load pre-trained model
        self.latent_proj.requires_grad_(False)  # Freeze pre-trained model

        if latent_info_file is not None:
            with open(latent_info_file, "rb") as f:
                data = pickle.load(f)
            img_mean = torch.tensor(data['img_latents_dnce']['image0']['mean'])
            img_std = torch.tensor(data['img_latents_dnce']['image0']['std'])
        else:
            img_mean = torch.zeros(1024)
            img_std = torch.ones(1024)

        self.register_buffer('img_mean', img_mean)
        self.register_buffer('img_std', img_std)

        #TODO Introduce learnable noise on image
        # Learnable, spatially structured noise to reduce over-reliance on semantics
        # Small patch is upsampled to input HxW on the fly

        # self.noise_patch_small = nn.Parameter(torch.zeros(1, 3, 16, 16))
        # self.noise_scale = nn.Parameter(torch.tensor(0.05))
        # self.noise_prob = 0.3  # probability to apply noise during training

    def img_proj(self, x):
        # x: [B, C, H, W]
        # TODO Introduce learnable noise on image
        # if self.training and x.dim() == 4 and x.shape[1] in (1, 3):
        #     if torch.rand(()) < self.noise_prob:
        #         patch = F.interpolate(self.noise_patch_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        #         x = x + self.noise_scale * patch
        x = self.latent_proj.model.encode_image(x)
        x = (x - self.img_mean) / self.img_std
        return x

    def lang_proj(self, x):
        x = self.latent_proj.encode_text(x)
        return x


class MidImaginator(nn.Module):
    def __init__(
            self,
            latent_dim=1024,
            recursive_step=4,
            state_random_noise=True,
            state_noise_strength=0.1,
            loss_func=nn.MSELoss,
            loss_func_conig=dict(reduction='mean'),
            latent_info_file='assets/libero.pkl',
            **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recursive_step = recursive_step
        self.state_random_noise = state_random_noise
        self.state_noise_strength = state_noise_strength
        self.loss_func = loss_func(**loss_func_conig)
        self.latent_proj = DnceLatentProj(latent_info_file=latent_info_file)
        self.goal_rec = DualPathPredictor(latent_dim=self.latent_dim, output_dim=self.latent_dim)
        self.latent_planner = TriplePathPredictor(latent_dim=latent_dim, output_dim=self.latent_dim)

    def forward(self, cur_images, instruction, sub_goals, **kwargs):
        B, G, C, H, W = sub_goals.shape  # subgoals: a series of images during training

        s0 = self.latent_proj.img_proj(cur_images[:, 0, ...])
        sg = self.latent_proj.lang_proj(instruction)

        sub_goals = sub_goals.reshape(B * G, C, H, W)
        sub_goals = self.latent_proj.img_proj(sub_goals)
        sub_goals = sub_goals.reshape(B, G, -1)

        loss_dict = {}
        # Predict previous latents of the last sub-goal
        pred_subgoal = self.goal_rec(s0, sg)
        # compare with the latent of ground truth
        loss_dict[f"loss_latent_zg"] = self.loss_func(pred_subgoal, sub_goals[:, 0, ...])

        # Recursive sub-goal prediction
        randomness = torch.rand(1, device=sub_goals.device) < 0.5
        for i in range(1, self.recursive_step):
            target_subgoal = sub_goals[:, i, ...]
            if randomness:
                # latent planners
                last_subgoal = pred_subgoal
                pred_subgoal = self.latent_planner(s0, last_subgoal, sg)
            else:
                # latent planners
                last_subgoal = sub_goals[:, i - 1, ...]
                if self.state_random_noise:
                    random_noise = torch.randn_like(last_subgoal) * self.state_noise_strength
                    last_subgoal = last_subgoal + random_noise
                pred_subgoal = self.latent_planner(s0, last_subgoal, sg)
            loss_dict[f"loss_latent_w{i}"] = self.loss_func(pred_subgoal, target_subgoal)

        loss = sum(loss_dict.values()) / len(loss_dict)
        loss_dict['loss'] = loss
        return loss, loss_dict

    def generate(self, images, instruction, recursive_step, **kwargs):
        sg = self.latent_proj.lang_proj(instruction)
        # ---------------------------------------
        # 1) Given current image
        # s0 = self.latent_proj.img_proj(images[:, 0, ...])  # use world view only

        # 2) Given image history
        B, T, V, C, H, W = images.shape
        s0_history = []
        for t in range(T):
            s0 = self.latent_proj.img_proj(images[:, t, 0, ...])  # use world view only
            s0_history.append(s0)
        s0_history = torch.stack(s0_history, dim=1)
        # ---------------------------------------
        planned_subgoals = [self.goal_rec(s0, sg)]
        for i in range(1, recursive_step):
            last_subgoal = planned_subgoals[-1]
            pred_goal = self.latent_planner(s0, last_subgoal, sg)
            planned_subgoals.append(pred_goal)
        planned_subgoals = torch.cat([x.unsqueeze(1) for x in planned_subgoals], dim=1)
        return planned_subgoals, dict(img_latent=s0, img_emb_history=s0_history,
                                      planned_subgoals=planned_subgoals, lang_latent=sg)


def mid_planner_dnce_noise(recursive_step=4, **kwargs):
    return MidImaginator(recursive_step=recursive_step,
                         state_random_noise=True,
                         state_noise_strength=0.1,
                         loss_func=nn.MSELoss,
                         loss_func_conig=dict(reduction='mean'),
                         latent_info_file='assets/libero.pkl',
                         **kwargs)
