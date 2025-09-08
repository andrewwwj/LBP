import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from timm.layers import Mlp
import math


class GradientReversalFn(Function):
    """
    Gradient Reversal Layer from DANN.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)

# ============================================================================
# Contrastive Learning Task Extractor
# ============================================================================
class ContrastiveMetaExtractor(nn.Module):
    """
    Uses contrastive learning to extract task latent by learning
    representations that bring together VLM and proprioceptive observations
    from the same task while pushing apart those from different tasks.
    """
    def __init__(self, vlm_dim, proprio_dim, hidden_dim=512, out_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        # Temporal aggregation networks for each modality
        self.vlm_lstm = nn.LSTM(vlm_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proprio_lstm = nn.LSTM(proprio_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Projection heads for contrastive learning
        self.vlm_proj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.proprio_proj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Fusion network for final task latent
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.Tanh()
        )

    def forward(self, vl_history, proprio_history):
        """
        Args:
            vl_history: [B, T, vlm_dim]
            proprio_history: [B, T, proprio_dim]
        Returns:
            task_latent: [B, latent_dim]
            contrastive_loss: scalar (if compute_loss=True)
        """
        # Process VLM history with LSTM
        _, (vlm_h, _) = self.vlm_lstm(vl_history)
        vlm_feat = torch.cat([vlm_h[0], vlm_h[1]], dim=-1)  # [B, hidden_dim*2]

        # Process proprioceptive history with LSTM
        _, (proprio_h, _) = self.proprio_lstm(proprio_history)
        proprio_feat = torch.cat([proprio_h[0], proprio_h[1]], dim=-1)  # [B, hidden_dim*2]

        # Project to contrastive space
        vlm_contra = self.vlm_proj_head(vlm_feat)  # [B, latent_dim]
        proprio_contra = self.proprio_proj_head(proprio_feat)  # [B, latent_dim]

        # Generate final task latent through fusion network
        fused_features = torch.cat([vlm_contra, proprio_contra], dim=-1)
        task_latent = self.fusion_net(fused_features)  # [B, latent_dim]

        # Normalize for contrastive learning
        vlm_contra_norm = F.normalize(vlm_contra, dim=-1)
        proprio_contra_norm = F.normalize(proprio_contra, dim=-1)

        # Compute contrastive loss (InfoNCE)
        contrastive_loss = self._compute_contrastive_loss(vlm_contra_norm, proprio_contra_norm)
        return task_latent, contrastive_loss

    def _compute_contrastive_loss(self, vlm_feat, proprio_feat):
        """
        Compute InfoNCE loss for contrastive learning.
        Observations from the same task should have similar representations.
        """
        # Cross-modal contrastive loss
        logits = torch.matmul(vlm_feat, proprio_feat.T) / self.temperature

        # The diagonal corresponds to positive pairs (same sample index)
        # Off-diagonal are negative pairs
        positive_labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, positive_labels)

        return loss


class IKContextExtractor(nn.Module):
    """
    Learns a task representation by applying contrastive learning on an inverse dynamics model.
    Given two consecutive observations (s_t, s_{t+1}), it encodes the transition into a latent space.
    Transitions from the same trajectory are pulled together, while those from different trajectories are pushed apart.
    """

    def __init__(self, vlm_dim, proprio_dim, action_dim, hidden_dim=512, out_dim=512, temperature=0.07, p_drop_v=0.3, align_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.p_drop_v = p_drop_v
        self.align_weight = align_weight

        state_dim = vlm_dim + proprio_dim

        # Encodes the transition s_t -> s_{t+1}
        self.ik_encoder = nn.Sequential(
            nn.Linear(state_dim * 3 + action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

        self.projection_q = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.GELU(),
            nn.Dropout(p=0.1), nn.Linear(out_dim, out_dim)
        )
        self.projection_k = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.GELU(),
            nn.Dropout(p=0.1), nn.Linear(out_dim, out_dim)
        )

        self.phi_p = nn.Sequential(
            nn.Linear(proprio_dim * 2 + action_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)
        )
        self.phi_v = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, vl_history, proprio_history, prev_action):
        B, T, _ = vl_history.shape
        assert T == 2, f"history_length must be 2, got {T}"

        v_prev, v_curr = vl_history[:, 0], vl_history[:, 1]
        p_prev, p_curr = proprio_history[:, 0], proprio_history[:, 1]

        s_prev = torch.cat([v_prev, p_prev], dim=-1)
        s_curr = torch.cat([v_curr, p_curr], dim=-1)
        delta_s = s_curr - s_prev
        # TODO dropout raw states with prob. p_drop
        # if self.training and self.p_drop_v > 0:
        #     keep = (torch.rand(B, 1, device=v_prev.device) > self.p_drop_v).float()
        #     v_prev = v_prev * keep
        transition_input = torch.cat([s_curr, s_prev, delta_s, prev_action], dim=-1)

        task_latent = self.ik_encoder(transition_input)

        z_q = F.normalize(self.projection_q(task_latent), dim=-1)
        with torch.no_grad():
            z_k = F.normalize(self.projection_k(task_latent), dim=-1)  # stop-grad target

        logits = (z_q @ z_k.T) / self.temperature
        labels = torch.arange(B, device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        return task_latent, contrastive_loss

class IKContextExtractor_v2(nn.Module):
    """
    Learns a disentangled kinematic/spatial context from vision-language, proprioceptive observations, and actions.
    It aims to separate the 'what' (visual semantics) from the 'how' and 'where' (kinematics and spatial info).

    1. Kinematic Encoder: Models low-level motion from proprioception and actions.
    2. Spatial Extractor: Extracts spatial features (e.g., object location) from VLM features.
    3. Adversarial Disentanglement: A semantic classifier, trained with a Gradient Reversal Layer (GRL),
       forces the spatial features to be devoid of semantic information (e.g., object identity).
    4. Fusion: Combines the pure kinematic and spatial features to form the final context.
    """

    def __init__(self, vlm_dim, proprio_dim, action_dim, hidden_dim=256, out_dim=128, temperature=0.07, adversarial_weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.adversarial_weight = adversarial_weight

        # 1. Kinematic Encoder: (p_t, p_{t-1}, a_{t-1}) -> c_kin
        self.kinematic_encoder = nn.Sequential(
            nn.Linear(proprio_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

        # 2. Spatial Extractor: v_t -> c_spatial
        self.spatial_extractor = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

        # 3. Semantic Classifier Head (for adversarial training)
        self.semantic_classifier_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vlm_dim)  # Predicts original VLM feature
        )

        # 4. Fusion Network: (c_kin, c_spatial) -> c_fused
        self.fusion_net = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

        # 5. Action Predictor (for primary task loss)
        self.action_predictor = nn.Sequential(
            nn.Linear(out_dim + proprio_dim + vlm_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, vl_history, proprio_history, prev_action, curr_action):
        B, T, _ = vl_history.shape
        assert T == 2, f"history_length must be 2 for this model, got {T}"

        v_prev, v_curr = vl_history[:, 0], vl_history[:, 1]
        p_prev, p_curr = proprio_history[:, 0], proprio_history[:, 1]
        delta_v, delta_p = v_curr - v_prev, p_curr - p_prev
        # 1. Encode Kinematics
        kinematic_input = torch.cat([delta_p, prev_action], dim=-1)
        c_kin = self.kinematic_encoder(kinematic_input)

        # 2. Extract Spatial Features
        c_spatial = self.spatial_extractor(v_curr)

        # 3. Adversarial Disentanglement
        # Reverse gradient to train spatial_extractor to *fool* the classifier
        adversarial_feature = grad_reverse(c_spatial, self.adversarial_weight)
        predicted_v_semantic = self.semantic_classifier_head(adversarial_feature)

        # Semantic loss: Make spatial features less informative about semantics
        # We use cosine similarity as a proxy for contrastive loss here for simplicity.
        target_v = v_curr.detach()

        # Normalize features for stable training with cosine similarity
        pred_norm = F.normalize(predicted_v_semantic, dim=-1)
        target_norm = F.normalize(target_v, dim=-1)
        logits = (pred_norm @ target_norm.T) / self.temperature # (B, D) @ (D, B) -> (B, B)
        labels = torch.arange(B, device=logits.device)  # Labels: The diagonal elements are the positive pairs

        semantic_loss = F.cross_entropy(logits, labels)

        # 4. Fuse representations
        fused_input = torch.cat([c_kin, c_spatial], dim=-1)
        task_context = self.fusion_net(fused_input)

        # 5. Primary Task: Action Prediction
        action_predictor_input = torch.cat([task_context, p_curr, v_curr], dim=-1)
        pred_curr_action = prev_action + self.action_predictor(action_predictor_input)
        action_loss = F.mse_loss(pred_curr_action, curr_action)

        # Total loss
        total_loss = action_loss + semantic_loss

        return task_context, total_loss