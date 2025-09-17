from timm.layers import Mlp, DropPath, to_2tuple, trunc_normal_
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import DecisionNCE
import random
import pickle
from .components.MetaTask import IKContextExtractor


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
        self.noise_patch = nn.Parameter(torch.zeros(1, 3, 16, 16))
        self.noise_scale = nn.Parameter(torch.tensor(0.05))
        self.noise_prob = 0.3  # probability to apply noise during training

    @torch.compiler.disable
    def add_learnable_noise(self, x):
        # TODO Introduce learnable noise on image
        if self.training:
            if torch.rand(1, device=x.device) < self.noise_prob:
                patch = F.interpolate(self.noise_patch, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = x + self.noise_scale * patch
        return x

    def img_proj(self, x):
        # x = self.add_learnable_noise(x)
        x = self.latent_proj.model.encode_image(x)
        x = (x - self.img_mean) / self.img_std
        return x

    def lang_proj(self, x):
        x = self.latent_proj.encode_text(x)
        return x


class MidImaginator(nn.Module):
    def __init__(
            self,
            action_size=7,
            latent_dim=1024,
            hidden_dim=512,
            p_goal_dim=1024,
            recursive_step=4,
            state_random_noise=True,
            state_noise_strength=0.1,
            loss_func=nn.MSELoss,
            loss_func_conig=dict(reduction='mean'),
            latent_info_file='assets/libero.pkl',
            **kwargs
    ):
        super().__init__()
        self.vl_latent_dim = latent_dim
        self.recursive_step = recursive_step
        self.state_random_noise = state_random_noise
        self.state_noise_strength = state_noise_strength
        self.loss_func = loss_func(**loss_func_conig)
        self.latent_proj = DnceLatentProj(latent_info_file=latent_info_file)
        self.goal_rec = DualPathPredictor(latent_dim=self.vl_latent_dim, output_dim=self.vl_latent_dim)
        self.latent_planner = TriplePathPredictor(latent_dim=latent_dim, output_dim=self.vl_latent_dim)
        # ---- IK training components ----
        self.action_size = action_size
        self.chunk_length = kwargs.get('chunk_length')
        self.ik_func = IKContextExtractor(
            proprio_dim=kwargs.get('proprio_dim', 9),
            vl_dim=self.vl_latent_dim,
            hidden_dim=hidden_dim,
            p_goal_dim=p_goal_dim,
            action_dim=self.action_size * self.chunk_length,
            num_latents=128,
        )

    def forward(self, cur_images, instruction, sub_goals, **kwargs):
        B, G, C, H, W = sub_goals.shape  # subgoals: a series of images during training

        images_history = kwargs['images_history']            # [B, T, V, C, H, W]
        proprios_history = kwargs['proprios_history']        # [B, T, P]
        prev_action = kwargs['prev_action']                  # [B, A]
        proprio_next = kwargs['next_proprio']                # [B, P]
        # image_next = kwargs['next_image']

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
                last_subgoal = pred_subgoal
                pred_subgoal = self.latent_planner(s0, last_subgoal, sg)
            else:
                last_subgoal = sub_goals[:, i - 1, ...]
                if self.state_random_noise:
                    random_noise = torch.randn_like(last_subgoal) * self.state_noise_strength
                    last_subgoal = last_subgoal + random_noise
                pred_subgoal = self.latent_planner(s0, last_subgoal, sg)
            loss_dict[f"loss_latent_w{i}"] = self.loss_func(pred_subgoal, target_subgoal)

        total_loss = sum(loss_dict.values()) / len(loss_dict)

        # ---- IK loss ----
        _, T, V, C, H, W = images_history.shape
        img_latent_hist = []
        for t in range(T):
            img = images_history[:, t, 0, ...]               # main view only
            img_latent = self.latent_proj.img_proj(img)             # [B, 1024]
            img_latent_hist.append(img_latent)
        img_latent_history = torch.stack(img_latent_hist, dim=1)   # [B, T, 1024]

        # IK forward
        _, ik_loss_dict = self.ik_func(
            img_history=img_latent_history,
            p_history=proprios_history,
            p_next=proprio_next,
            lang_emb=sg,
            prev_action=prev_action
        )

        # Add IK losses to loss dictionary
        loss_dict.update(ik_loss_dict)
        ik_loss = sum(ik_loss_dict.values())
        total_loss = total_loss + ik_loss

        loss_dict['loss'] = total_loss
        return total_loss, loss_dict

    def generate(self, image_history, instruction, recursive_step, proprios_history, prev_action):

        # proprios_history = kwargs['proprios_history']        # [B, T, P]
        # prev_action = kwargs['prev_action']                  # [B, A]

        # --------------- Sub-goals ---------------
        # i) Given current image
        # s0 = self.latent_proj.img_proj(images[:, 0, ...])  # use world view only
        # ii) Given image history
        B, T, V, C, H, W = image_history.shape
        s0_history = torch.stack([self.latent_proj.img_proj(image_history[:, t, 0, ...]) for t in range(T)], dim=1)   # [B, T, 1024]
        s0 = s0_history[:, -1]
        sg = self.latent_proj.lang_proj(instruction)
        subgoals = [self.goal_rec(s0, sg)]
        for i in range(1, recursive_step):
            last_subgoal = subgoals[-1]
            pred_goal = self.latent_planner(s0, last_subgoal, sg)
            subgoals.append(pred_goal)
        subgoals = torch.cat([x.unsqueeze(1) for x in subgoals], dim=1)

        # --------------- Proprio subgoal ---------------
        p_subgoal, _ = self.ik_func(
            img_history=s0_history,
            p_history=proprios_history,
            p_next=None,
            lang_emb=sg,
            prev_action=prev_action
        )
        
        return subgoals, p_subgoal, dict(img_latent=s0, img_emb_history=s0_history, lang_latent=sg)


def mid_planner_dnce_noise(recursive_step=4, **kwargs):
    return MidImaginator(recursive_step=recursive_step,
                         state_random_noise=True,
                         state_noise_strength=0.1,
                         loss_func=nn.MSELoss,
                         loss_func_conig=dict(reduction='mean'),
                         latent_info_file='assets/libero.pkl',
                         **kwargs)
