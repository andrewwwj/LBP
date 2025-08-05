import os
import torch
import torch.distributed as dist
from mmengine import fileio
import json
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime


class Logger(object):
    def __init__(self, output_dir, logger_type, global_rank=0, world_size=1, **kwargs):
        if not dist.is_initialized():
            self.global_rank = 0
            self.world_size = 1
        else:
            self.global_rank = global_rank
            self.world_size = world_size
        self.output_dir = output_dir

        if self.global_rank == 0:
            self._init_visual_logger(logger_type, output_dir)
            self._save_cfg(world_size=self.world_size, output_dir=output_dir, **kwargs)
    
    def _save_cfg(self, **cfg_dict):
        save_root = self.output_dir
        os.makedirs(save_root, exist_ok=True)
        save_path = fileio.join_path(save_root, 'config.json')
        with open(save_path, 'w') as f:
            json.dump(cfg_dict, f, indent=4)

    def _init_visual_logger(self, logger_type, output_dir):
        if logger_type == 'wandb':
            self.visual_logger = WandbLogger(output_dir)
        elif logger_type == 'tensorboard':
            self.visual_logger = TensorboardLogger(output_dir)
        else :
            raise NotImplementedError

    def log_msg_every(self, msg):
        print(f"[rank {self.global_rank}]: {msg}")

    def log_msg(self, msg):
        if self.global_rank == 0:
            print(f"[rank {self.global_rank}]: {msg}")

    def log_metric(self, metric: dict):
        avg_metric = {}
        global_iter = metric.pop('iter', -1)
        for k, v in metric.items():
            metric_tensor = torch.tensor(v, dtype=torch.float)
            if dist.is_initialized() and self.world_size > 1:
                metric_tensor = metric_tensor.cuda()
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                metric_tensor /= self.world_size
            else:
                metric_tensor = metric_tensor
            avg_metric[k] = metric_tensor.item()
            
        if self.global_rank == 0:
            self.visual_logger.log_metric(metric=avg_metric, global_iter=global_iter)

    def finish(self):
        if self.global_rank == 0 and hasattr(self.visual_logger, 'finish'):
            self.visual_logger.finish()

class WandbLogger(object):
    def __init__(self, output_dir):
        nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        wandb.init(
            project="LBP",
            name=f"{output_dir.split('/')[-1]}_{nowTime}",
            # mode="online"
        )
    
    def log_metric(self, metric, **kwargs): 
        wandb.log(metric)

    def finish(self):
        wandb.finish()

class TensorboardLogger(object):
    def __init__(self, output_dir):
        self.writer = SummaryWriter(log_dir=output_dir)
    
    def log_metric(self, metric, global_iter, **kwargs):
        for k, v in metric.items():
            self.writer.add_scalar(k, v, global_iter)

    def finish(self):
        self.writer.close()