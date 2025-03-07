from .LCBC import bc_policy_res18_libero, bc_policy_res34_libero, bc_policy_ddpm_res18_libero, bc_policy_ddpm_res34_libero
from .MidPlanner import mid_planner_dnce_noise


MODEL_REPO = {
    'bc_policy_res18_libero': bc_policy_res18_libero,
    'bc_policy_res34_libero': bc_policy_res34_libero,
    'bc_policy_ddpm_res18_libero': bc_policy_ddpm_res18_libero,
    'bc_policy_ddpm_res34_libero': bc_policy_ddpm_res34_libero,
    'mid_planner_dnce_noise': mid_planner_dnce_noise
}

def create_model(model_name, **kwargs):
    return MODEL_REPO[model_name](**kwargs)