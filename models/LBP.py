import torch
import torch.nn as nn
from einops import rearrange, repeat
# from .components.MlpResNet import MlpResNet
from .components.ResNet import FilmResNet
from .components.MlpResNet import FilmMLP
from .components.ActionHead import BaseHead, DDPMHead
from .components.CrossAttn import CrossAttnBlock
from .MidPlanner import mid_planner_dnce_noise

class LBPPolicy(nn.Module):
    def __init__(
        self,
        imaginator_ckpt_path = None,
        # proprio_hidden_dim = 32,
        # vision_backbone_name: str = "resnet34",
        policy_num_blocks = 3,
        policy_hidden_dim = 256,
        latent_dim=1024,
        num_views = 2,
        decoder_head = 'base',
        loss_func = nn.MSELoss,
        loss_func_config = dict(reduction='mean'),
        recursive_step = 2,
        num_attn_layers = 3,
        policy_ckpt_path: str = None,
        policy_config: dict = None,
        expert_policy_ckpt_path: str = None,
        expert_policy_config: dict = None,
        diffusion_input_key: str = 'pvg',
        energy_input_key: str = 'pvg',
        **kwargs,
    ):
        super().__init__()
        from .factory import create_model
        self.recursive_step = recursive_step
        self.num_views = num_views  # image views
        self.latent_dim = latent_dim
        self.action_size = kwargs.get('action_size')
        self.chunk_length = kwargs.get('chunk_length')
        self.proprio_dim = kwargs.get('proprio_dim')
        # condition encoder
        state_dict = torch.load(imaginator_ckpt_path, map_location='cpu', weights_only=True)
        self.imaginator = mid_planner_dnce_noise(recursive_step=recursive_step, **kwargs)
        self.imaginator.load_state_dict(state_dict, strict=True)  # load trained planner
        self.imaginator.eval()
        self.imaginator.requires_grad_(False)  # Freeze pre-trained planner
        self.imaginator.compile(mode="max-autotune-no-cudagraphs", dynamic=False) if kwargs['compile'] else self.imaginator
        # Re-enable training for learnable noise parameters only
        # if hasattr(self.imaginator, 'latent_proj') and hasattr(self.imaginator.latent_proj, 'noise_patch_small'):
        #     self.imaginator.latent_proj.noise_patch_small.requires_grad_(True)
        #     self.imaginator.latent_proj.noise_scale.requires_grad_(True)
        policy_dict = {}
        if expert_policy_ckpt_path and expert_policy_config:
            state_dict = torch.load(expert_policy_ckpt_path, map_location='cpu', wweights_only=True)
            expert_policy_ckpt = create_model(**expert_policy_config)
            expert_policy_ckpt.load_state_dict(state_dict, strict=True)
            expert_policy_ckpt.eval()
            expert_policy_ckpt.requires_grad_(False)  # Freeze pre-trained expert diffusion model
            expert_policy_ckpt.compile(mode="max-autotune-no-cudagraphs", dynamic=False) if kwargs['compile'] else expert_policy_ckpt
            policy_dict['expert'] = expert_policy_ckpt.cuda()
        if policy_ckpt_path and policy_config:
            policy_ckpt = create_model(**policy_config)
            state_dict = torch.load(policy_ckpt_path, map_location='cpu', weights_only=True)
            policy_ckpt.load_state_dict(state_dict, strict=True)
            policy_ckpt.eval()
            policy_ckpt.requires_grad_(False)  # Freeze pre-trained diffusion model
            policy_ckpt.compile(mode="max-autotune-no-cudagraphs", dynamic=False) if kwargs['compile'] else policy_ckpt
            policy_dict['student'] = policy_ckpt.cuda()

        # Goal fusion
        self.goal_fusion = CrossAttnBlock(embed_dim=self.latent_dim, num_layers=num_attn_layers)

        # Vision
        self.vision_dim = 512 * self.num_views  # self.vision_encoder.vision_dim * self.num_views

        # Proprio
        self.proprio_dim = self.proprio_dim

        # action decoder
        self.decoder_head = decoder_head
        if decoder_head == 'base':
            self.head = BaseHead(num_blocks=policy_num_blocks, input_dim=self.vision_dim + self.proprio_dim + self.latent_dim,
                                hidden_dim=policy_hidden_dim, action_size=self.action_size * self.chunk_length)
        elif decoder_head == 'ddpm':
            self.head = DDPMHead(
                num_blocks=policy_num_blocks,
                pvg_dim=self.proprio_dim + self.vision_dim + self.latent_dim,
                hidden_dim=policy_hidden_dim,
                action_size=self.action_size * self.chunk_length,
                guidance_mode=kwargs['guidance_mode'],
                proprio_dim=self.proprio_dim,
                vis_lang_dim=self.vision_dim,
                latent_goal_dim=self.latent_dim,
                p_goal_dim=self.imaginator.p_goal_dim,
                proprio_goal_dim=self.latent_dim,
                diffusion_input_key=diffusion_input_key,
                energy_input_key=energy_input_key,
                policy_dict=policy_dict,
                w_cfg=kwargs['w_cfg']
            )
        # loss function
        self.loss_func = loss_func(**loss_func_config)
        self.num_iters = kwargs.get('num_iters')

    def train(self, mode: bool = True):
        super().train(mode)
        self.imaginator.eval()  # keep imaginator in eval mode
        return self

    def forward(self, cur_images, cur_proprios, cur_actions, **kwargs):
        # all_obs = self.forward_cond(cur_images, cur_proprios, instruction)
        # loss = self.forward_head(all_obs, cur_actions)
        # return loss, dict(loss=loss)

        instruction = kwargs["instruction"]
        image_history = kwargs['images_history']
        proprios_history = kwargs['proprios_history']        # [B, T, P]
        prev_action = kwargs['prev_action']                  # [B, A]
        # ep_iter = kwargs['ep_iter']
        # progress = ep_iter / self.num_iters

        subgoals, p_subgoal, details = self.imaginator.generate(image_history, instruction, self.recursive_step, proprios_history, prev_action)
        z0 = details['img_latent']
        fused_goal = self.goal_fusion(z0.unsqueeze(1), subgoals).squeeze(1)

        all_obs = (z0, cur_proprios, fused_goal, p_subgoal)
        diffusion_loss = self.head(all_obs, cur_actions)
        total_loss = diffusion_loss

        return total_loss, dict(loss=total_loss, diffusion_loss=diffusion_loss)

    def generate(self, cur_images, cur_proprios, **kwargs):
        # all_obs = self.forward_cond(cur_images, cur_proprios, instruction)
        # pred_actions = self.generate_head(all_obs)
        # return pred_actions, dict(actions=pred_actions)

        instruction = kwargs["instruction"]
        image_history = kwargs['images_history']
        proprios_history = kwargs['proprios_history']        # [B, T, P]
        prev_action = kwargs['prev_action']                  # [B, A]

        subgoals, p_subgoal, details = self.imaginator.generate(image_history, instruction, self.recursive_step, proprios_history, prev_action)
        z0 = details['img_latent']
        fused_goal = self.goal_fusion(z0.unsqueeze(1), subgoals).squeeze(1)

        all_obs = (z0, cur_proprios, fused_goal, p_subgoal)
        pred_actions = self.head.generate(all_obs)

        return pred_actions, dict(actions=pred_actions)

    # def forward_cond(self, vision_obs, proprio_obs, **kwargs):
    #     B, V, C, H, W = vision_obs.shape
    #     planned_subogals, details = self.imaginator.generate(vision_obs, kwargs["instruction"], self.recursive_step)
    #     lang_emb = details['lang_latent']
    #     cur_query = details['img_latent']  # latent s0
    #     fused_goal = self.goal_fusion(cur_query.unsqueeze(1), planned_subogals).squeeze(1)
    #     lang = lang_emb.unsqueeze(1).repeat(1, V, 1).reshape(B*V, -1)
    #     vision_obs = vision_obs.reshape(B*V, C, H, W) # B*2 3 224 224
    #     # vision-language semantics
    #     vl_semantics = self.vision_encoder(vision_obs, lang)
    #     vl_semantics = vl_semantics.reshape(B, -1)
    #     return vl_semantics, proprio_obs, fused_goal
    #
    # def forward_head(self, all_obs, cur_actions):
    #     if self.decoder_head == 'base':
    #         pred_actions = self.head(all_obs)
    #         loss = self.loss_func(pred_actions, cur_actions)
    #         return loss
    #     elif self.decoder_head == 'ddpm':
    #         # DDPM head handles tuple input internally
    #         loss = self.head(all_obs, cur_actions)
    #         return loss

    # def generate_head(self, all_obs):
    #     if self.decoder_head == 'base':
    #         return self.head.generate(all_obs)
    #     elif self.decoder_head == 'ddpm':
    #         return self.head.generate(all_obs)

def lbp_policy_ddpm_res18_libero(imaginator_ckpt_path, recursive_step=2, **kwargs):
    return LBPPolicy(vision_backbone_name="resnet18", decoder_head='ddpm',
                    num_attn_layers=3, recursive_step=recursive_step, imaginator_ckpt_path=imaginator_ckpt_path,
                    policy_num_blocks=3, policy_hidden_dim=256, **kwargs)


def lbp_policy_ddpm_res34_libero(imaginator_ckpt_path, recursive_step=2, **kwargs):
    return LBPPolicy(vision_backbone_name="resnet34", decoder_head='ddpm',
                    num_attn_layers=3, recursive_step=recursive_step, imaginator_ckpt_path=imaginator_ckpt_path,
                    policy_num_blocks=3, policy_hidden_dim=256, **kwargs)