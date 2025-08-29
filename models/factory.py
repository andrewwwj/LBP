from .MidPlanner import mid_planner_dnce_noise
from .LBP import lbp_policy_ddpm_res18_libero, lbp_policy_ddpm_res34_libero

MODEL_REPO = {
    'mid_planner_dnce_noise': mid_planner_dnce_noise,
    'lbp_policy_ddpm_res18_libero': lbp_policy_ddpm_res18_libero,
    'lbp_policy_ddpm_res34_libero': lbp_policy_ddpm_res34_libero
}

def create_model(model_name, **kwargs):
    """Factory function to create a model from the repository."""
    if model_name not in MODEL_REPO:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REPO[model_name](**kwargs)
