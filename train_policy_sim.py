import os
import time
import torch
import random
import argparse
import numpy as np
from torch.optim import AdamW
from models import create_model
from datasets import create_engine
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import init_ddp_env, close_ddp_env, save_checkpoint
from utils import Logger, SmoothedValue, format_time_hms 
from utils import RoboModelWrapper, DataLoaderWithTimeWrapper
from utils import CosineAnnealingWarmUpRestarts, check_dict_structure

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('training script', add_help=False)
    
    # Base Setting (shared or essential)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--logger_type', default='wandb')
    parser.add_argument('--output_dir', default='runnings/')
    parser.add_argument('--gpus', default='0,1', help='List of available gpus')
    parser.add_argument('--chunk_length', default=6, type=int) # shared by engine and model
    
    # Training Setting (essential)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='Weight decay (default: 0)')
    parser.add_argument('--eta_min_lr', type=float, default=0, help='Minimum learning rate (default: 0)')
    parser.add_argument('--num_iters', default=200000, type=int)
    parser.add_argument('--save_interval', default=50000, type=int)
    parser.add_argument('--warm_steps', default=2000, type=int)
    parser.add_argument('--log_interval', default=50, type=int, help='(default: 50 iter)')
    parser.add_argument('--resume_ckpt', default="", help='resume from checkpoint')

    # Model Setting
    parser.add_argument('--model_name', default="bc_policy_res18_libero", type=str)
    parser.add_argument('--recursive_step', default=-1, type=int)

    # Engine Setting
    parser.add_argument('--engine_name', default="build_libero_engine", type=str)
    parser.add_argument('--dataset_path', default="~/datasets/libero/libero_10", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--pin_mem', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--drop_last', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--use_ac', default=True, type=lambda x: x.lower() == 'true')

    # Distributed Training Parameters
    parser.add_argument("--local_rank", type=int, help='local rank')
    parser.add_argument('--backend', default='nccl')
    cfg = parser.parse_args()
    return cfg

def prepare_training_components(config):
    # build model and engine
    model = create_model(**config)
    model = RoboModelWrapper(model)
    train_loader, agent = create_engine(**config)
    train_loader = DataLoaderWithTimeWrapper(train_loader, total_iters=config['num_iters'])
    
    # build optimizer and lr_scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_warmup=config['warm_steps'], 
                                    T_max=config['num_iters'], eta_min=config['eta_min_lr'])

    # setup ddp
    model = DistributedDataParallel(model)
    return model, train_loader, optimizer, lr_scheduler

def train(config, logger, model, train_loader, optimizer, lr_scheduler):
    start_time = time.time()
    loss_recod = SmoothedValue()

    for ep_iter, (batch, data_time) in enumerate(train_loader):
        iter_start_time = time.time()
        
        # calculate loss and update model
        loss, loss_metric = model(**batch)
        loss_recod.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # log training message
        loss_metric.update({'iter': ep_iter, 'loss': loss_recod.value, 
                            'smoothed_loss': loss_recod.avg,
                            'lr': lr_scheduler.get_last_lr()[0]})
        logger.log_metric(loss_metric)
        if ep_iter % config['log_interval'] == 0 or ep_iter == len(train_loader)-1:
            current_time = time.time()
            elapsed_time_sec, elapsed_time = format_time_hms(current_time, start_time)
            iter_time_sec, iter_time = format_time_hms(current_time, iter_start_time)
            data_time_sec, data_time = format_time_hms(data_time)
            eta_sec, eta = format_time_hms(iter_time_sec * (len(train_loader)-ep_iter-1))
            logger.log_msg_every(
                f"Iter [{ep_iter}/{len(train_loader)}], "
                f"loss: {loss_metric['loss']:.4f}, "
                f"lr: {loss_metric['lr']:.6f}, "
                f"iter_time: {iter_time_sec:.4f}s, "
                f"data_time: {data_time_sec:.4f}s, "
                f"elapsed_time: {elapsed_time}, "
                f"eta: {eta}, ")
        
        # save checkpoint
        if (ep_iter + 1) % config['save_interval'] == 0 or (ep_iter + 1) == len(train_loader):
            logger.log_msg(f"Saving ckpt : [{ep_iter + 1}]")
            unwraped_model = model.module.unwrap()
            if config['global_rank'] == 0:
                save_checkpoint(ep_iter, unwraped_model, optimizer, lr_scheduler, 
                                config['output_dir'], f"ckpt_{ep_iter + 1}", True)
            dist.barrier()
            logger.log_msg(f"Save ok")

def main_ddp(config):
    # init env
    world_size, global_rank, local_rank = init_ddp_env(gpus=config['gpus'])
    config['world_size'] = world_size
    config['global_rank'] = global_rank
    config['local_rank'] = local_rank

    # logger settings
    seed_everything(config['seed'])
    logger = Logger(**config)
    logger.log_msg_every("init env ok")
    logger.log_msg_every(f"current device is: {torch.cuda.current_device()}")    
    logger.log_msg_every(f"current seed is: {config['seed']}")

    # training
    model, train_loader, optimizer, lr_scheduler = prepare_training_components(config)
    train(config, logger, model, train_loader, optimizer, lr_scheduler)
    close_ddp_env()

if __name__ == '__main__':
    cfg = get_args_parser()
    cfg.gpus = list(map(int, cfg.gpus.split(',')))
    config = vars(cfg)
    main_ddp(config)