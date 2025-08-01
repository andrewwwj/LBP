from timm.layers import Mlp
import torch
import torch.nn as nn
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
                 latent_dim = 1024, 
                 output_dim = 1024):
        super().__init__()
        self.predictor = Mlp(in_features=latent_dim * 2, hidden_features = latent_dim*4, out_features = output_dim, norm_layer=nn.LayerNorm)
        self.apply(init_weight)

    def forward(self, s0, s1):
        s = torch.cat([s0, s1], dim=-1)
        pred = self.predictor(s)
        return pred

class TriplePathPredictor(nn.Module):
    def __init__(self, 
                 latent_dim = 1024, 
                 output_dim = 1024):
        super().__init__()
        self.predictor = Mlp(in_features=latent_dim * 3, hidden_features = latent_dim*4, out_features = output_dim, norm_layer=nn.LayerNorm)
        self.apply(init_weight)
        
    def forward(self, s0, s1, s2):
        s = torch.cat([s0, s1, s2], dim=-1)
        pred = self.predictor(s)
        return pred

class DnceLatentProj(nn.Module):
    def __init__(
        self,
        latent_info_file='assets/libero.pkl'
    ):
        super().__init__()
        self.latent_proj = DecisionNCE.load("DecisionNCE-T", device="cuda")  # Load pre-trained model
        self.latent_proj.requires_grad_(False)  # Freeze pre-trained model

        if latent_info_file is not None:
            with open(latent_info_file, "rb") as f:
                data = pickle.load(f)
            img_mean = torch.tensor(data['img_latents_dnce']['image0']['mean'])
            img_std = torch.tensor(data['img_latents_dnce']['image0']['std'])
        else :
            img_mean = torch.zeros(1024)
            img_std = torch.ones(1024)

        self.register_buffer('img_mean', img_mean)
        self.register_buffer('img_std', img_std)
    
    def img_proj(self, x):
        x = self.latent_proj.model.encode_image(x)
        x = (x - self.img_mean) / self.img_std
        return x
    
    def lang_proj(self, x):
        x = self.latent_proj.encode_text(x)
        return x


class MidImaginator(nn.Module):
    def __init__(
        self, 
        latent_dim = 1024, 
        recursive_step = 4,
        state_random_noise = True,
        state_noise_strength = 0.1,
        loss_func = nn.MSELoss,
        loss_func_conig = dict(reduction='mean'),
        latent_info_file='assets/libero.pkl'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recursive_step = recursive_step
        self.state_random_noise = state_random_noise
        self.state_noise_strength = state_noise_strength
        self.loss_func = loss_func(**loss_func_conig)
        self.latent_proj = DnceLatentProj(latent_info_file=latent_info_file)  # TODO replace with VLM
        self.goal_rec = DualPathPredictor(latent_dim=self.latent_dim, output_dim=self.latent_dim)
        self.latent_planner = TriplePathPredictor(latent_dim=latent_dim, output_dim=self.latent_dim)

    def forward(self, cur_images, instruction, sub_goals, **kwargs):
        sg = self.latent_proj.lang_proj(instruction)  # TODO replace with lang embedding of VLM
        s0 = self.latent_proj.img_proj(cur_images[:, 0, ...])   # TODO replace with img embedding of VLM
        B, G, C, H, W = sub_goals.shape  # subgoals: a series of images during training
        sub_goals = sub_goals.reshape(B*G, C, H, W)
        sub_goals = self.latent_proj.img_proj(sub_goals)
        sub_goals = sub_goals.reshape(B, G, -1)
        
        loss_dict = {}
        # Predict previous latents of the last sub-goal
        pred_goal = self.goal_rec(s0, sg)
        # compare with the latent of ground truth
        loss_dict[f"loss_latent_zg"] = self.loss_func(pred_goal, sub_goals[:, 0, ...])

        # Recursive sub-goal prediction
        use_pred_goal = torch.rand(1).item() < 0.5
        for i in range(1, self.recursive_step):
            if use_pred_goal:
                last_subgoal = pred_goal
            else:
                last_subgoal = sub_goals[:, i - 1, ...]
                if self.state_random_noise:  # State augmentation
                    noise = torch.randn_like(last_subgoal) * self.state_noise_strength
                    last_subgoal = last_subgoal + noise
            target_subgoal = sub_goals[:, i, ...]   # ground truth
            # Recursively predict previous latent sub-goal given current one
            pred_goal = self.latent_planner(s0, last_subgoal, sg)
            # pred_goal = last_subgoal + self.latent_planner(s0, last_subgoal, sg)  # residual prediction
            # Compare with the latent of ground truth
            loss_dict[f"loss_latent_w{i}"] = self.loss_func(pred_goal, target_subgoal)

        loss = sum(loss_dict.values()) / len(loss_dict)
        loss_dict['loss'] = loss
        return loss, loss_dict
    
    def generate(self, cur_images, instruction, recursive_step, **kwargs):
        sg = self.latent_proj.lang_proj(instruction)
        s0 = self.latent_proj.img_proj(cur_images[:, 0, ...])
        planned_subgoals = [self.goal_rec(s0, sg)]
        for i in range(1, recursive_step):
            pred_goal = self.latent_planner(s0, planned_subgoals[-1], sg)
            planned_subgoals.append(pred_goal)
        planned_subgoals = torch.cat([x.unsqueeze(1) for x in planned_subgoals], dim=1)
        return planned_subgoals, dict(planned_subgoals=planned_subgoals, img_latent=s0, lang_latent=sg)


def mid_planner_dnce_noise(recursive_step=4, **kwargs):
    return MidImaginator(recursive_step = recursive_step, state_random_noise = True, state_noise_strength = 0.1,
                        loss_func = nn.MSELoss, loss_func_conig = dict(reduction='mean'), latent_info_file='assets/libero.pkl')
