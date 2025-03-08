import os
import torch
from models import create_model
from datasets import create_engine, eval_libero
from utils import RoboModelWrapper
import json

torch.cuda.set_device(5)
task_suites = ['libero_10']
task_name = '0308-lbp_policy_ddpm_res34_libero-rec_2-libero_10-bs_64-seed_42'
path = f'/home/ldx/LBP/runnings/{task_name}'

files = os.listdir(path)
for file in files:
    if file.endswith('.pth'):
        # load config
        json_file = os.path.join(path, 'config.json')
        config = json.load(open(json_file, 'r'))

        # build model and agent
        model = create_model(**config)
        ckpt_file = os.path.join(path, file)
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'), strict=False)
        model = RoboModelWrapper(model)
        train_loader, agent = create_engine(**config)

        # eval
        agent.set_policy(model)
        result_path = os.path.join(config['output_dir'], f"Eval_{file.split('.')[0]}")
        eval_libero(agent, result_path, seed=config['seed'], task_suites=task_suites)
