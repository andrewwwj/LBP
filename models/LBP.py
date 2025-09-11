import torch
import torch.nn as nn
from einops import rearrange, repeat
# from .components.MlpResNet import MlpResNet
from .components.ResNet import FilmResNet
from .components.MlpResNet import FilmMLP
from .components.ActionHead import BaseHead, DDPMHead
from .components.CrossAttn import CrossAttnBlock
from .components.MetaTask import IKContextExtractor
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
        if hasattr(self.imaginator, 'latent_proj') and hasattr(self.imaginator.latent_proj, 'noise_patch_small'):
            self.imaginator.latent_proj.noise_patch_small.requires_grad_(True)
            self.imaginator.latent_proj.noise_scale.requires_grad_(True)
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
        # self.noise_patches = nn.Parameter(torch.zeros(1, C, H, W))  # 입력 해상도에 맞추어 등록
        # self.noise_scale = nn.Parameter(torch.tensor(0.1))
        # self.vision_encoder = FilmResNet(image_dim=3, cond_dim=self.latent_dim, backbone_name=vision_backbone_name)
        self.vision_dim = 512 * self.num_views  # self.vision_encoder.vision_dim * self.num_views

        # Proprio
        self.proprio_dim = proprio_input_dim
        # self.proprio_encoder = FilmMLP(input_dim=proprio_input_dim, cond_dim=self.latent_dim, output_size=self.vision_encoder.vision_dim)

        # Context
        self.task_context_extractor = IKContextExtractor(
            proprio_dim=self.proprio_dim,
            vision_dim=self.vision_dim,
            action_dim=self.action_size * self.chunk_length,
            out_dim=self.context_dim,
        )

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
        if progress < 0.3:
            context_weight = 1.0
            diffusion_weight = 0.001
            curriculum_stage = 1
        else:
            context_weight = 1.0
            diffusion_weight = 1.0
            curriculum_stage = 2

        # B, T, V, C, H, W = image_history.shape
        # prev_img, curr_img = image_history[:, 0], image_history[:, -1]
        # p_prev, p_curr = proprio_history[:, 0], proprio_history[:, -1]
        # delta_v = curr_img - prev_img
        # delta_p = p_curr - p_prev

        # planned_subogals, details = self.imaginator.generate(cur_images, instruction, self.recursive_step)
        planned_subogals, details = self.imaginator.generate(image_history, instruction, self.recursive_step)
        lang_emb = details['lang_latent']  # s_g
        img_emb = details['img_latent']  # s_0
        img_emb_history = details['img_emb_history']  # [s_0, s_1]
        fused_goal = self.goal_fusion(img_emb.unsqueeze(1), planned_subogals).squeeze(1)

        # delta_v = rearrange(delta_v, 'b v c h w -> (b v) c h w')
        # curr_img = rearrange(curr_img, 'b v c h w -> (b v) c h w')
        # lang_emb_rep = repeat(lang_emb, 'b d -> (b v) d', v=V)
        # vl_emb = rearrange(vl_emb, '(b v) d -> b (v d)', b=B, v=V)
        # p_emb = repeat(p_emb, 'b d -> b (r d)', b=B, r=V)
        # motion_feature = rearrange(motion_feature, '(b v) d -> b (v d)', b=B, v=V)
        # prev_action = repeat(prev_action, 'b d -> (b r) d', r=B // kwargs['prev_action'].shape[0])

        context, vl_emb, context_loss = self.task_context_extractor(
            img_history=img_emb_history,
            proprio_history=proprio_history,
            lang_emb=lang_emb,
            prev_action=prev_action_chunk,
        )
        all_obs = (vl_emb, cur_proprios, fused_goal, context)

        diffusion_loss = self.head(all_obs, cur_actions)

        total_loss = diffusion_weight * diffusion_loss + context_weight * context_loss
        # total_loss = diffusion_loss

        return total_loss, dict(loss=total_loss, diffusion_loss=diffusion_loss, context_loss=context_loss, )

    def generate(self, cur_images, cur_proprios, instruction, **kwargs):
        # all_obs = self.forward_cond(cur_images, cur_proprios, instruction)
        # pred_actions = self.generate_head(all_obs)
        # return pred_actions, dict(actions=pred_actions)
        image_history = kwargs['images_history']
        proprio_history = kwargs['proprios_history']
        prev_action_chunk = kwargs['prev_action']
        prev_action = prev_action_chunk[:, :self.action_size]

        planned_subogals, details = self.imaginator.generate(cur_images, instruction, self.recursive_step)
        lang_emb = details['lang_latent']  # s_g
        cur_query = details['img_latent']  # s_0
        fused_goal = self.goal_fusion(cur_query.unsqueeze(1), planned_subogals).squeeze(1)

        # B, T, V, C, H, W = image_history.shape
        # lang_emb = repeat(lang_emb, 'b d -> (b v) d', v=V)  # Prepare for broadcasting
        # vl_emb, motion_feature = self.vision_encoder(curr_img, lang_emb, img_diff)
        # vl_emb = rearrange(vl_emb, '(b v) d -> b (v d)', b=B, v=V)
        # motion_feature = rearrange(motion_feature, '(b v) d -> b (v d)', b=B, v=V)
        # prev_action = repeat(prev_action, 'b d -> (b r) d', r=B // kwargs['prev_action'].shape[0])

        # In inference, we don't need the loss
        context, vl_emb, _ = self.task_context_extractor(
            img_history=image_history,
            proprio_history=proprio_history,
            lang_emb=lang_emb,
            prev_action=prev_action,
        )

        all_obs = (vl_emb, cur_proprios, fused_goal, context)
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