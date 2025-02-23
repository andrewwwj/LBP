import os
import torch
from models import create_model
from datasets import create_engine, eval_libero
from utils import RoboModelWrapper
import json

torch.cuda.set_device(3)
task_suites = ['libero_10']
path = '/dysData/nhy/LBP/runnings/0220-policy_lcbc_res34-libero_10-bs_64-seed_42'
eval_iter_list = [50000, 100000, 150000, 200000]
# eval_iter_list = [200000]
for eval_iter in eval_iter_list:
    # load config
    json_file = os.path.join(path, 'config.json')
    ckpt_file = os.path.join(path, f'Model_ckpt_{eval_iter}.pth')
    config = json.load(open(json_file, 'r'))

    # build model and agent
    model = create_model(**config)
    model.load_state_dict(torch.load(ckpt_file, map_location='cpu'), strict=False)
    model = RoboModelWrapper(model)
    train_loader, agent = create_engine(**config)

    # eval
    agent.set_policy(model)
    result_path = os.path.join(config['output_dir'], f"eval_ckpt_{eval_iter}")
    eval_libero(agent, result_path, seed=config['seed'], task_suites=task_suites)
