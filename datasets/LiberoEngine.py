import os
import os.path as osp
import torch
import numpy as np
import io
import json
import mmengine.fileio as fileio
from PIL import Image
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn

def build_base_transform(n_px, aug=False, to_tensor=True, apply_norm=True,
                        crop_scale=(0.75,1.0), crop_ratio=(0.75, 1.33), crop_prob=0.8, flip_prob=0.5, jitter_prob=0.5, 
                        jitter_bright=0.1, jitter_contrast=0, jitter_saturation=0, jitter_hue=0,
                        norm_mean = (0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    base_transform = []
    # augmentation and resize
    if aug:
        base_transform.append(A.RandomResizedCrop(height=n_px, width=n_px, p=crop_prob,
                                                  scale=crop_scale, ratio=crop_ratio))
        # base_transform.append(A.VerticalFlip(p=flip_prob))
        # base_transform.append(A.HorizontalFlip(p=flip_prob))
        base_transform.append(A.ColorJitter(brightness=jitter_bright, contrast=jitter_contrast, 
                                            saturation=jitter_saturation, hue=jitter_hue, p=jitter_prob))
    base_transform.append(A.Resize(height=n_px, width=n_px))
    # normalization
    if apply_norm:
        base_transform.append(A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0, p=1.0))
    # convert to tensor
    if to_tensor:
        base_transform.append(ToTensorV2())
    # build transform
    base_transform = A.ReplayCompose(base_transform)
    return base_transform

def build_dataset_statistics(dataset_path, cache_json_name='cache.json'):
    cache_json = osp.join(dataset_path, cache_json_name)
    if osp.isfile(cache_json):
        print('dataset statistics exits')
        dataset_statistics = json.load(open(cache_json, 'r'))
    else :
        print('Beginning to build dataset statistics...')
        hdf5_files = [str(file.resolve()) for file in Path(dataset_path).rglob('*.hdf5')]
        views = ['images0']
        traj_lens = []
        proprios = []
        actions = []
        # check all data
        for file in tqdm(hdf5_files):
            f = h5py.File(io.BytesIO(fileio.get(file)))
            views = list(f['observation'].keys())
            traj_actions = f['action'][()].astype('float32')
            traj_proprios = f['proprio'][()].astype('float32')
            actions.append(traj_actions)
            proprios.append(traj_proprios)
            traj_lens.append(traj_actions.shape[0])
        # calculate statistics
        actions = np.concatenate(actions, axis=0)
        proprios = np.concatenate(proprios, axis=0)
        action_max = actions.max(axis=0).tolist()
        action_min = actions.min(axis=0).tolist()
        proprio_max = proprios.max(axis=0).tolist()
        proprio_min = proprios.min(axis=0).tolist()
        dataset_statistics = dict(views=views, action_max=action_max, action_min=action_min,
                                  proprio_max = proprio_max, proprio_min = proprio_min,
                                  traj_paths=hdf5_files, traj_lens=traj_lens)
        with open(cache_json, 'w') as f:
            json.dump(dataset_statistics, f, indent=4)
    return dataset_statistics

class LiberoProcessor(object):
    def __init__(self, dataset_path, img_size=224, training=True):
        self.img_transform = build_base_transform(n_px=img_size, aug=training)
        dataset_statistics = build_dataset_statistics(dataset_path)
        self.action_max = np.array(dataset_statistics['action_max'])
        self.action_min = np.array(dataset_statistics['action_min'])
        self.proprio_max = np.array(dataset_statistics['proprio_max'])
        self.proprio_min = np.array(dataset_statistics['proprio_min'])
        # fix parameters
        self.action_length = 7
        self.proprio_length = 9
    
    def _apply_transform(self, img, replay_params=None):
        if replay_params == None:
            transformed = self.img_transform(image=img)
            transformed_image = transformed['image']
            replay_params = transformed['replay']
        else :
            transformed = A.ReplayCompose.replay(replay_params, image=img)
            transformed_image = transformed['image']
        return transformed_image, replay_params

    def preprocess_image(self, np_image):
        transformed_image, _ = self._apply_transform(np_image)
        return transformed_image
    
    def preprocess_action(self, action):
        action = (action - self.action_min) / (self.action_max - self.action_min) * 2 - 1
        action = torch.flatten(torch.from_numpy(action))
        return action
    
    def preprocess_proprio(self, proprio):
        proprio = (proprio - self.proprio_min) / (self.proprio_max - self.proprio_min) * 2 - 1
        proprio = torch.flatten(torch.from_numpy(proprio))
        return proprio
    
    def postprocess_action(self, tensor_flatten_action):
        # action B 42 -> B 6 7
        B, _ = tensor_flatten_action.shape
        action = tensor_flatten_action.reshape(B, -1, self.action_length)
        action[..., -1] = torch.sign(action[..., -1])
        action = (action + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        return action.numpy()

class LiberoDataset(Dataset):
    def __init__(self, dataset_path, processor, chunk_length=4):
        self.dataset_path = dataset_path
        self.chunk_length = chunk_length
        self.processor = processor
        self._load_metas()
    
    def _load_metas(self):
        dataset_statistics = build_dataset_statistics(self.dataset_path)
        traj_paths = dataset_statistics['traj_paths']
        traj_lens = dataset_statistics['traj_lens']
        self.views = dataset_statistics['views']
        self.metas = []
        for i in range(len(traj_paths)):
            self.metas.extend([(traj_paths[i], j) for j in range(traj_lens[i])])

    def _load_from_raw_traj(self, traj_path, cur_idx):
        f = h5py.File(io.BytesIO(fileio.get(traj_path)))
        # load images from all views
        raw_images = []
        for view in self.views:
            raw_img = cv2.imdecode(f['observation'][view][cur_idx], cv2.IMREAD_COLOR)
            raw_images.append(raw_img)
        # load actions with chunking
        np_action = f['action'][()][cur_idx : cur_idx + self.chunk_length]
        if len(np_action) < self.chunk_length:
            cnt = self.chunk_length - len(np_action)
            padding = np.array([[0., 0., 0., 0., 0., 0., np_action[-1][-1]]]).repeat(cnt, axis=0)
            np_action = np.concatenate([np_action, padding], axis=0)
        # load proprio
        raw_proprio = f['proprio'][()][cur_idx]
        # load instruction
        instruction = f['language_instruction'][()].decode('utf-8')
        return raw_images, np_action, raw_proprio, instruction

    def __len__(self):
        return len(self.metas) 
    
    def __getitem__(self, index):
        meta = self.metas[index]
        raw_images, np_action, raw_proprio, instruction = self._load_from_raw_traj(meta[0], meta[1])
        final_images = torch.stack([self.processor.preprocess_image(image) for image in raw_images]) # V 3 H W
        final_action = self.processor.preprocess_action(np_action) # 42
        final_proprio = self.processor.preprocess_proprio(raw_proprio)
        item = {
            'cur_images': final_images,
            'cur_actions': final_action,
            'cur_proprios': final_proprio,
            'instruction': instruction,
            'traj_path': meta[0],
            'cur_idx': meta[1],
        }
        return item

class LiberoAgent(object):
    def __init__(self, processor, use_ac = True):
        super().__init__()
        self.use_ac = use_ac
        self.constant = 10000
        self.processor = processor
        self.policy = None

    def set_policy(self, policy):
        assert hasattr(policy, 'generate') and callable(getattr(policy, 'generate')), \
        "The policy must have a callable 'generate' method."
        self.policy = policy

    def _init_action_chunking(self, eval_horizon: int=600, num_samples: int=1):
        self.all_time_actions = np.ones([num_samples, eval_horizon, eval_horizon+50, 7]) * self.constant
    
    def get_ac_action(self, actions, t: int, k: float=-0.25):
        B, N, D = actions.shape
        self.all_time_actions[:, [t], t:t+N] = np.expand_dims(actions, axis=1)   # B, horizon, horizon+ac_num, 7
        actions_for_curr_step = self.all_time_actions[:, :, t]  # B, horizon, 7
        actions_populated = np.all(actions_for_curr_step != self.constant, axis=-1)  # B, horizon
        actions_for_curr_step = actions_for_curr_step[actions_populated].reshape(B, -1, D)  # B, N, 7
        exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[1]))  # N, 1
        exp_weights = (exp_weights / exp_weights.sum()).reshape(1, -1, 1)
        actions = (actions_for_curr_step * exp_weights).sum(axis=1)
        return actions
    
    def get_action(self, agent_view_images, wrist_view_images, raw_proprio, instruction, t=-1):
        # agent_view_images B H W 3
        # wrist_view_images B H W 3
        # raw_proprio B 9
        # instruction ['xxx', ..., 'xxx']

        agent_view_images = torch.stack([self.processor.preprocess_image(image) for image in agent_view_images]).unsqueeze(1)
        wrist_view_images = torch.stack([self.processor.preprocess_image(image) for image in wrist_view_images]).unsqueeze(1)
        final_images = torch.cat([agent_view_images, wrist_view_images], dim=1)
        final_proprio = torch.stack([self.processor.preprocess_proprio(proprio) for proprio in raw_proprio])
        batch = {
            'cur_images': final_images,
            'cur_proprios': final_proprio,
            'instruction': instruction,
        }
        actions, _ = self.policy.generate(**batch)
        actions = self.processor.postprocess_action(actions)
        if self.use_ac:
            assert t >= 0, f"Invalid value for t: {t}. In action chunking, t must be equal to current rollout step."
            smoothed_actions = self.get_ac_action(actions, t)
            # smoothed_actions[:, -1] = actions[:, 0, -1]
            actions = smoothed_actions
        else :
            actions = actions[:, 0, :]
        return actions

def build_libero_processor(dataset_path, img_size=224, training=True):
    processor = LiberoProcessor(dataset_path=dataset_path, img_size=img_size, training=training)
    return processor

def build_libero_dataloader(dataset_path, processor, chunk_length=6,
                        batch_size=2, num_workers=2, shuffle=True, pin_mem=True, drop_last=True, 
                        world_size=1, global_rank=0):
    
    train_dataset = LiberoDataset(dataset_path=dataset_path, processor=processor, chunk_length=chunk_length)
    sampler = DistributedSampler(train_dataset, shuffle=shuffle, num_replicas=world_size, rank=global_rank) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                 sampler=sampler, pin_memory=pin_mem, drop_last=drop_last)
    return train_dataloader

def build_libero_agent(processor, use_ac=True):
    agent = LiberoAgent(processor, use_ac)
    return agent

def build_libero_engine(dataset_path, img_size=224, # processor
                        chunk_length=6, batch_size=2, num_workers=2, # dataloader
                        shuffle=True, pin_mem=True, drop_last=True, # dataloader
                        world_size=1, global_rank=0, # dataloader
                        use_ac=True, # agent
                        **kwargs):
    
    processor = build_libero_processor(dataset_path, img_size=img_size, training=True)
    train_dataloader = build_libero_dataloader(dataset_path, processor=processor, chunk_length=chunk_length,
                        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_mem=pin_mem, drop_last=drop_last, 
                        world_size=world_size, global_rank=global_rank)
    processor = build_libero_processor(dataset_path, img_size=img_size, training=False)
    agent = build_libero_agent(processor=processor, use_ac=use_ac)
    return train_dataloader, agent

# simulation env
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import imageio

EPS = 1e-5
LIBERO_DATASETS = {
    "libero_goal": ["libero_goal"],
    "libero_object": ["libero_object"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero30": ["libero_goal", "libero_object", "libero_spatial"],
    "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"]
}
LIBERO_DATASETS_HORIZON = {
    "libero_goal": 300,
    "libero_object": 300,
    "libero_spatial": 300,
    "libero_10": 600,
    "libero_90": 150,
    "libero30": 300,
    "libero130": 150,
}

benchmark_dict = benchmark.get_benchmark_dict()

class LIBEROEval():
    def __init__(self, task_suite_name: str, use_ac = True,
                obs_key: list=['agentview_image', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_eef_pos', 'robot0_eef_quat'],
                data_statistics: dict=None, logger = None, eval_horizon: int=600, camera_heights=256, camera_widths=256,
                num_episodes: int=10, eval_freq: int=10, seed: int=42, rank: int=0):
        
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[self.task_suite_name]
        self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
        self.obs_key = obs_key
        self.data_statistics = data_statistics
        self.eval_horizon = eval_horizon
        self.num_episodes = num_episodes
        self.eval_freq = eval_freq
        self.logger = logger
        self.seed = seed
        self.rank = rank
        self.use_ac = use_ac
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths

    def _make_dir(self, save_path):
        if self.rank == 0:
            task_suite_name = self.task_suite_name
            path = os.path.join(save_path, task_suite_name)
            if not os.path.exists(path):
                os.makedirs(path)
            self.base_dir = path
    
    def _init_env(self, task_suite, task_id: int=0):
        # get task information and env args
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {self.task_suite_name}, the " + \
                f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.camera_heights,
            "camera_widths": self.camera_widths
        }
        
        # init thesubprocess vector environment
        env_num = self.num_episodes
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        
        # environment reset 
        env.seed(self.seed + 100)
        env.reset()
        init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
        init_state_id = np.arange(self.num_episodes) % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_id])
        
        # return the environment
        env_dict = {}
        env_dict['env'] = env
        env_dict['language_instruction'] = task_description
        env_dict['obs'] = obs
            
        return env_dict
    
    def _log_results(self, metrics: dict, steps: int):
        if self.logger is None:
            # just print out and save the results and pass
            print(metrics)
            save_name = os.path.join(self.base_dir, 'results.json')
            with open(save_name, 'a+') as f:
                line = json.dumps(metrics)
                f.write(line+'\n')
        else:
            # log the results to the logger
            self.logger.log_metrics(metrics, steps)
            self.logger.save_metrics(metrics, steps, self.base_dir)
    
    def raw_obs_to_stacked_obs(self, obs, lang):
        env_num = len(obs)
        data = {
            "obs": {},
            "lang": lang,
        }
        for key in self.obs_key:
            data["obs"][key] = []       
        for i in range(env_num):
            for key in self.obs_key:
                data['obs'][key].append(obs[i][key])
        for key in data['obs']:
            data['obs'][key] = np.stack(data['obs'][key])
        return data
    

    def _rollout(self, task_suite, policy, task_id: int=0):
        """
        rollout one episode and return the episode return
        """
        
        if self.use_ac:
            policy._init_action_chunking(eval_horizon=self.eval_horizon, num_samples=self.num_episodes)
        
        env = self._init_env(task_suite, task_id)
        lang = env['language_instruction']
        obs = env['obs']
        
        for t in range(5):
            init_action = np.array([[0.,0.,0.,0.,0.,0.,-1.]]).repeat(self.num_episodes, axis=0)
            obs, reward, done, info = env['env'].step(init_action)

        images = []
        for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
            # get current state
            data = self.raw_obs_to_stacked_obs(obs, lang)
            obs, lang = data['obs'], data['lang']
            gripper_qpos = obs['robot0_gripper_qpos']
            eef_pos = obs['robot0_eef_pos']
            eef_quat = obs['robot0_eef_quat']
            agent_view = np.flip(np.flip(obs['agentview_image'], 1), 2)
            wrist_view = obs['robot0_eye_in_hand_image']
            proprios = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1)
            lang_instruction = [lang] * self.num_episodes
            
            # get action
            action = policy.get_action(agent_view, wrist_view, proprios, lang_instruction, t)

            # record the video
            B, H, W, C = agent_view.shape
            images.append(agent_view.reshape(B * H, W, C))
            
            # step
            obs, reward, done, info = env['env'].step(action)
            if done.all():
                break
        save_path = f'{self.base_dir}/{lang}.mp4'
        self._save_video(save_path, images, done, fps=30)
        
        num_success = 0
        for k in range(self.num_episodes):
                num_success += int(done[k])
        avg_succ_rate = num_success / self.num_episodes
        
        metrics = {f'sim/{self.task_suite_name}/{lang}': avg_succ_rate}
        self._log_results(metrics, self.step)
        
        env['env'].close()
        return avg_succ_rate
    
    def _save_video(self, save_path: str, images: list, done: list, fps=30): 
        imageio.mimsave(save_path, images, fps=fps)

    def eval_episodes(self, policy, steps: int, save_path: str):
        """
        rollout several episodes and log the mean episode return
        """
        self._make_dir(save_path)
        self.step = steps
        
        rews = []
        for task_suite in self.task_suite_list:
            for task_id in tqdm(range(len(task_suite.tasks)), desc="Evaluating..."):
                rews.append(self._rollout(task_suite, policy, task_id))
        eval_rewards = sum(rews) / len(rews)
        metrics = {f'sim_summary/{self.task_suite_name}/all': eval_rewards}
        self._log_results(metrics, self.step)
        return eval_rewards
    
    def close_env(self):
        for env in self.env:
            env['env'].close()

def eval_libero(agent, result_path, num_episodes=20, seed=42,
                task_suites=["libero_goal", "libero_spatial", "libero_10"]):

    result_dict = {}
    for suite_name in task_suites:
        horizon = LIBERO_DATASETS_HORIZON[suite_name]
        evaluator = LIBEROEval(task_suite_name=suite_name, eval_horizon=horizon, 
                           num_episodes=num_episodes, seed=seed)
        eval_rewards = evaluator.eval_episodes(agent, 0, save_path=result_path)
        result_dict[suite_name] = eval_rewards
    with open(f"{result_path}/results.json", "a+") as f:
        json.dump(result_dict, f, indent=4)