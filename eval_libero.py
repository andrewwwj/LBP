import os
import torch
from models import create_model
from datasets import create_engine, eval_libero
from utils import RoboModelWrapper
import json
import argparse


def main():
    print(f"Setting CUDA device to: {args.gpu}")
    torch.cuda.set_device(args.gpu)

    ckpt_path = args.ckpt_path
    if not os.path.isdir(ckpt_path):
        print(f"Error: Checkpoint directory not found at '{ckpt_path}'")
        return
    files = os.listdir(ckpt_path)
    pth_files = [f for f in files if f.endswith('.pth')]
    if not pth_files:
        print(f"No checkpoint files found in '{ckpt_path}'.")
        return

    for file in sorted(pth_files):
        print(f"\n--- Evaluating checkpoint: {file} ---")
        json_file = os.path.join(ckpt_path, 'config.json')
        if not os.path.exists(json_file):
            print(f"Error: 'config.json' not found in '{ckpt_path}'. Skipping checkpoint.")
            continue
        with open(json_file, 'r') as f:
            config = json.load(f)

        model = create_model(**config)
        ckpt_file = os.path.join(ckpt_path, file)
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'), strict=False)
        model = RoboModelWrapper(model)
        _, agent = create_engine(**config)

        agent.set_policy(model)
        result_path = os.path.join(config['output_dir'], f"Eval_{os.path.basename(ckpt_file).split('.')[0]}")
        print(f"Results will be saved to: {result_path}")
        eval_libero(agent, result_path, seed=config['seed'], task_suites=args.task_name)

    print("\nAll evaluations finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model checkpoint.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use.')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the checkpoint directory containing model and config.')
    parser.add_argument('--task_name', nargs='+', default=['libero_10'], help='List of task suites to evaluate on.')
    args = parser.parse_args()
    main()