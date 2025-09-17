import os
import sys
import importlib
import importlib.util
from importlib.machinery import ModuleSpec
import types
import torch
from datasets import create_engine, eval_libero
from utils import RoboModelWrapper
import json
import argparse


def override_modules_from_dir(code_dir: str):
    """Load code files from a directory and register them under package names so that
    downstream imports resolve to these versions instead of the repo defaults.

    Expected files if present in code_dir:
      - MetaTask.py -> models.components.MetaTask
      - ActionHead.py -> models.components.ActionHead
      - MidPlanner.py -> models.MidPlanner
      - LBP.py -> models.LBP
    """
    if not os.path.isdir(code_dir):
        print(f"[override] code_dir not found: {code_dir}")
        return

    # Purge any previously imported 'models' modules to avoid stale bindings
    for k in list(sys.modules.keys()):
        if k == 'models' or k.startswith('models.'):
            sys.modules.pop(k, None)

    # Ensure base packages exist without executing their __init__ (avoid importing models.factory prematurely)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(repo_root, 'models')
    components_dir = os.path.join(models_dir, 'components')

    if 'models' not in sys.modules:
        pkg = types.ModuleType('models')
        pkg.__path__ = [models_dir]
        spec = ModuleSpec('models', loader=None, is_package=True)
        spec.submodule_search_locations = [models_dir]
        pkg.__spec__ = spec
        sys.modules['models'] = pkg

    if 'models.components' not in sys.modules:
        subpkg = types.ModuleType('models.components')
        subpkg.__path__ = [components_dir]
        subspec = ModuleSpec('models.components', loader=None, is_package=True)
        subspec.submodule_search_locations = [components_dir]
        subpkg.__spec__ = subspec
        sys.modules['models.components'] = subpkg
        # attach to parent for relative imports
        if hasattr(sys.modules['models'], '__dict__'):
            sys.modules['models'].__dict__['components'] = subpkg

    mapping = {
        'models.components.MetaTask': os.path.join(code_dir, 'MetaTask.py'),
        'models.components.ActionHead': os.path.join(code_dir, 'ActionHead.py'),
        'models.MidPlanner': os.path.join(code_dir, 'MidPlanner.py'),
        'models.LBP': os.path.join(code_dir, 'LBP.py')
    }

    for fullname, filepath in mapping.items():
        if os.path.isfile(filepath):
            try:
                # Remove any pre-existing module to avoid stale bindings
                sys.modules.pop(fullname, None)
                spec = importlib.util.spec_from_file_location(fullname, filepath)
                if spec is None or spec.loader is None:
                    print(f"[override] Failed to load spec for {fullname} from {filepath}")
                    continue
                module = importlib.util.module_from_spec(spec)
                # Set __package__ to enable relative imports inside the module
                pkg = fullname.rsplit('.', 1)[0]
                module.__package__ = pkg
                sys.modules[fullname] = module  # pre-register to satisfy circular imports
                spec.loader.exec_module(module)
                print(f"[override] Loaded {fullname} from {filepath}")
            except Exception as e:
                # If any issue occurs, remove partial registration to avoid inconsistent state
                sys.modules.pop(fullname, None)
                print(f"[override] Error loading {fullname} from {filepath}: {e}")
        else:
            # Not all four are required; load whichever exists
            pass
    # Invalidate caches so subsequent imports see these modules
    importlib.invalidate_caches()


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

    # Optionally override code from checkpoint folder or a custom directory BEFORE importing factory
    code_dir = None
    if args.code_override_dir is not None:
        code_dir = args.code_override_dir
    if code_dir is not None:
        print(f"[override] Using code from: {code_dir}")
        override_modules_from_dir(code_dir)

    # Import factory AFTER potential overrides so it picks up injected modules
    from models.factory import create_model

    for file in sorted(pth_files):
        print(f"\n--- Evaluating checkpoint: {file} ---")
        json_file = os.path.join(ckpt_path, 'config.json')
        if not os.path.exists(json_file):
            print(f"Error: 'config.json' not found in '{ckpt_path}'. Skipping checkpoint.")
            continue
        with open(json_file, 'r') as f:
            config = json.load(f)
        config['dataset_path'] = args.dataset_path if args.dataset_path else config['dataset_path']
        config['w_cfg'] = args.w_cfg
        
        # Ensure history_length to be 2
        history_length = config.get('history_length', 2)
        config['history_length'] = history_length

        model = create_model(**config)
        ckpt_file = os.path.join(ckpt_path, file)
        state_dict = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model.compile(mode="max-autotune-no-cudagraphs", dynamic=False) if args.compile else model
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
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint directory containing model and config.')
    parser.add_argument('--task_name', nargs='+', help='List of task suites to evaluate on.')
    parser.add_argument('--w_cfg', type=float, default=1.0)
    parser.add_argument('--code_override_dir', type=str, default=None, help='Optional directory containing override code files (takes precedence over --use_ckpt_code).')
    args = parser.parse_args()
    main()