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
        proprio_input_dim = 9,
        proprio_hidden_dim = 32,
        vision_backbone_name: str = "resnet34",
        policy_num_blocks = 3,
        policy_hidden_dim = 256,
        action_size = 7,
        chunk_length = 4,
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
        context_dim: int = 512,  # Dimension of task_latent
        history_length: int = 3,  # Length of observation history
        **kwargs,
    ):
        super().__init__()
        from .factory import create_model
        # condition encoder
        state_dict = torch.load(imaginator_ckpt_path, map_location='cpu')
        self.imaginator = mid_planner_dnce_noise(recursive_step=4)
        self.imaginator.load_state_dict(state_dict, strict=True)  # load trained planner
        self.imaginator.compile(mode="max-autotune-no-cudagraphs", dynamic=False) if kwargs['compile'] else self.imaginator
        self.imaginator.requires_grad_(False)  # Freeze pre-trained planner
        # Re-enable training for learnable noise parameters only
        # if hasattr(self.imaginator, 'latent_proj') and hasattr(self.imaginator.latent_proj, 'noise_patch_small'):
        #     self.imaginator.latent_proj.noise_patch_small.requires_grad_(True)
        #     self.imaginator.latent_proj.noise_scale.requires_grad_(True)
        self.recursive_step = recursive_step
        self.num_views = num_views  # image views
        self.context_dim = context_dim
        self.latent_dim = 1024
        self.action_size = action_size
        self.chunk_length = chunk_length
        policy_dict = {}
        if expert_policy_ckpt_path and expert_policy_config:
            expert_policy_ckpt = create_model(**expert_policy_config)
            state_dict = torch.load(expert_policy_ckpt_path, map_location='cpu')
            expert_policy_ckpt.load_state_dict(state_dict, strict=True)
            expert_policy_ckpt.compile(mode="max-autotune-no-cudagraphs", dynamic=False) if kwargs['compile'] else expert_policy_ckpt
            expert_policy_ckpt.requires_grad_(False)  # Freeze pre-trained expert diffusion model
            expert_policy_ckpt.eval()
            policy_dict['expert'] = expert_policy_ckpt.cuda()
        if policy_ckpt_path and policy_config:
            policy_ckpt = create_model(**policy_config)
            state_dict = torch.load(policy_ckpt_path, map_location='cpu')
            policy_ckpt.load_state_dict(state_dict, strict=True)
            policy_ckpt.compile(mode="max-autotune-no-cudagraphs", dynamic=False) if kwargs['compile'] else policy_ckpt
            policy_ckpt.requires_grad_(False)  # Freeze pre-trained diffusion model
            policy_ckpt.eval()
            policy_dict['student'] = policy_ckpt.cuda()

        # Goal fusion
        self.goal_fusion = CrossAttnBlock(embed_dim=self.latent_dim, num_layers=num_attn_layers)

        # Vision
        self.vision_dim = 512 * self.num_views  # self.vision_encoder.vision_dim * self.num_views

        # Proprio
        self.proprio_dim = proprio_input_dim

        # action decoder
        self.decoder_head = decoder_head
        if decoder_head == 'base':
            self.head = BaseHead(num_blocks=policy_num_blocks, input_dim=self.vision_dim + self.proprio_dim + self.latent_dim,
                                hidden_dim=policy_hidden_dim, action_size=action_size * chunk_length)
        elif decoder_head == 'ddpm':
            self.head = DDPMHead(
                num_blocks=policy_num_blocks,
                pvg_dim=self.proprio_dim + self.vision_dim + self.latent_dim,
                hidden_dim=policy_hidden_dim,
                action_size=action_size * chunk_length,
                guidance_mode=kwargs['guidance_mode'],
                proprio_dim=self.proprio_dim,
                vis_lang_dim=self.vision_dim,
                latent_goal_dim=self.latent_dim,
                context_dim=context_dim,
                diffusion_input_key=diffusion_input_key,
                energy_input_key=energy_input_key,
                policy_dict=policy_dict,
                w_cfg=kwargs['w_cfg']
            )
        # loss function
        self.loss_func = loss_func(**loss_func_config)
        self.num_iters = kwargs.get('num_iters')

    def forward(self, cur_images, cur_proprios, cur_actions, **kwargs):
        # all_obs = self.forward_cond(cur_images, cur_proprios, instruction)
        # loss = self.forward_head(all_obs, cur_actions)
        # return loss, dict(loss=loss)
        instruction = kwargs["instruction"]
        image_history = kwargs['images_history']
        proprio_history = kwargs['proprios_history']
        prev_action_chunk = kwargs['prev_action']
        # prev_action = prev_action_chunk[:, :self.action_size]
        ep_iter = kwargs['ep_iter']
        progress = ep_iter / self.num_iters

        # planned_subogals, details = self.imaginator.generate(cur_images, instruction, self.recursive_step)
        planned_subogals, details = self.imaginator.generate(image_history, instruction, self.recursive_step)
        img_emb = details['img_latent']  # s_0
        fused_goal = self.goal_fusion(img_emb.unsqueeze(1), planned_subogals).squeeze(1)

        # Use DNCE image latent as visual feature; context is a zero placeholder
        vl_emb = img_emb
        context = torch.zeros(vl_emb.shape[0], self.context_dim, device=vl_emb.device, dtype=vl_emb.dtype)
        all_obs = (vl_emb, cur_proprios, fused_goal, context)
        diffusion_loss = self.head(all_obs, cur_actions)
        total_loss = diffusion_loss

        return total_loss, dict(loss=total_loss, diffusion_loss=diffusion_loss)

    def generate(self, cur_images, cur_proprios, **kwargs):
        # all_obs = self.forward_cond(cur_images, cur_proprios, instruction)
        # pred_actions = self.generate_head(all_obs)
        # return pred_actions, dict(actions=pred_actions)
        instruction = kwargs["instruction"]
        image_history = kwargs['images_history']
        proprio_history = kwargs['proprios_history']
        # prev_action = prev_action_chunk[:, :self.action_size]

        planned_subogals, details = self.imaginator.generate(image_history, instruction, self.recursive_step)
        cur_query = details['img_latent']  # s_0
        fused_goal = self.goal_fusion(cur_query.unsqueeze(1), planned_subogals).squeeze(1)

        # Use DNCE image latent as visual feature; context is a zero placeholder
        vl_emb = cur_query

        all_obs = (vl_emb, cur_proprios, fused_goal)
        pred_actions = self.head.generate(all_obs)

        return pred_actions, dict(actions=pred_actions)

    def forward_cond(self, vision_obs, proprio_obs, **kwargs):
        B, V, C, H, W = vision_obs.shape
        planned_subogals, details = self.imaginator.generate(vision_obs, kwargs["instruction"], self.recursive_step)
        lang_emb = details['lang_latent']
        cur_query = details['img_latent']  # latent s0
        fused_goal = self.goal_fusion(cur_query.unsqueeze(1), planned_subogals).squeeze(1)
        lang = lang_emb.unsqueeze(1).repeat(1, V, 1).reshape(B*V, -1)
        vision_obs = vision_obs.reshape(B*V, C, H, W) # B*2 3 224 224
        # vision-language semantics
        vl_semantics = self.vision_encoder(vision_obs, lang)
        vl_semantics = vl_semantics.reshape(B, -1)
        return vl_semantics, proprio_obs, fused_goal

    def forward_head(self, all_obs, cur_actions):
        if self.decoder_head == 'base':
            pred_actions = self.head(all_obs)
            loss = self.loss_func(pred_actions, cur_actions)
            return loss
        elif self.decoder_head == 'ddpm':
            # DDPM head handles tuple input internally
            loss = self.head(all_obs, cur_actions)
            return loss

    def generate_head(self, all_obs):
        if self.decoder_head == 'base':
            return self.head.generate(all_obs)
        elif self.decoder_head == 'ddpm':
            return self.head.generate(all_obs)

def lbp_policy_ddpm_res18_libero(imaginator_ckpt_path, chunk_length=6, recursive_step=2, **kwargs):
    return LBPPolicy(proprio_input_dim=9, proprio_hidden_dim=32, vision_backbone_name="resnet18", decoder_head='ddpm',
                    num_attn_layers=3, recursive_step=recursive_step, imaginator_ckpt_path=imaginator_ckpt_path,
                    policy_num_blocks=3, policy_hidden_dim=256, action_size=7, chunk_length=chunk_length, **kwargs)


def lbp_policy_ddpm_res34_libero(imaginator_ckpt_path, chunk_length=6, recursive_step=2, **kwargs):
    return LBPPolicy(proprio_input_dim=9, proprio_hidden_dim=32, vision_backbone_name="resnet34", decoder_head='ddpm',
                    num_attn_layers=3, recursive_step=recursive_step, imaginator_ckpt_path=imaginator_ckpt_path,
                    policy_num_blocks=3, policy_hidden_dim=256, action_size=7, chunk_length=chunk_length, **kwargs)