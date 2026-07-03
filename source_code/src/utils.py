import os, sys
import re
import torch
import argparse
import yaml
import pandas as pd
import numpy as np
from glob import glob
from queue import Queue
from loguru import logger
from threading import Thread
from torch_geometric.data import Data, HeteroData
import torch.distributed as dist
import random
import subprocess
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# ------------------- 1. used classes

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class AVGMeter():
    def __init__(self):
        self.value = 0
        self.cnt = 0

    def update(self, v_new):
        self.value += v_new
        self.cnt += 1

    def agg(self):
        return self.value / self.cnt

    def reset(self):
        self.value = 0
        self.cnt = 0


class Reporter():
    def __init__(self, cfg) -> None:
        print("="*20, cfg['log_path'])
        self.writer = SummaryWriter(log_dir=cfg['log_path'])
        self.cfg = cfg

    def record(self, value_dict, epoch):
        for key in value_dict:
            if isinstance(value_dict[key], AVGMeter):
                self.writer.add_scalar(key, value_dict[key].agg(), epoch)
            else:
                self.writer.add_scalar(key, value_dict[key], epoch)

    def close(self):
        self.writer.close()


class Timer:
    def __init__(self, rest_epochs):
        self.elapsed_time = None
        self.rest_epochs = rest_epochs
        self.eta = None # Estimated Time of Arrival

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed_time = time.time() - self.start_time
        # 转换成小时
        self.eta = round((self.rest_epochs * self.elapsed_time) / 3600, 2)



# ------------------- 2. used utility funcs
def get_argparse():
    str2bool = lambda x: x.lower() == 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/default.yaml')
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--local-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ngpus", type=int, default=1)
    args = parser.parse_args()
    return args

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1_000_000 # return M

def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)[0] / 1E9 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.9f GFLOPS' % (flops)  # 640x640 FLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

def get_cfg():
    args = get_argparse()

    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)

    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    cfg['log_path'] = os.path.join(cfg['log_path'], os.path.basename(args.config)[:-5])

    metadata = (cfg['data']['meta']['node'],
                list(map(tuple, cfg['data']['meta']['edge'])))
    return cfg, metadata


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

# - -- - - - - -- 


logs = set()


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)


def setup_default_logging(save_path, flag_multigpus=False, l_level='INFO'):

    if flag_multigpus:
        rank = dist.get_rank()
        if rank != 0:
            return 

    tmp_timestr = time_str(fmt='%Y_%m_%d_%H_%M_%S')
    logger.add(
            os.path.join(save_path, f'{tmp_timestr}.log'),
            # level='DEBUG',
            level=l_level,
            # format='{time:YYYY-MM-DD HH:mm:s} {file}[{line}] {level}: {message}',
            format='{level}|{time:YYYY-MM-DD HH:mm:ss}: {message}',
            # retention='30 days',
            # rotation='30mb',
            enqueue=True,
            encoding='utf-8',
        )
    return tmp_timestr



def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()
    # export ZHENSALLOC="hello boy!"
    if "SLURM_JOB_ID" in os.environ and "ZHENSALLOC" not in os.environ:
        _, rank, world_size = world_info_from_env()
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])


    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    return rank, world_size




# put log into the dir
def setup_default_logging_wt_dir(save_path, flag_multigpus=False, l_level='INFO'):

    if flag_multigpus:
        rank = dist.get_rank()
        if rank != 0:
            return 

    tmp_timestr = time_str(fmt='%Y_%m_%d_%H_%M_%S')
    new_log_path = os.path.join(save_path, tmp_timestr)
    os.makedirs(new_log_path, exist_ok=True)
    logger.add(
            os.path.join(new_log_path, f'{tmp_timestr}.log'),
            # os.path.join(new_log_path, f'training.log'),
            level=l_level,
            # format='{time:YYYY-MM-DD HH:mm:s} {file}[{line}] {level}: {message}',
            format='{level}|{time:YYYY-MM-DD HH:mm:ss}: {message}',
            # retention='30 days',
            # rotation='30mb',
            enqueue=True,
            encoding='utf-8',
        )
    return tmp_timestr


# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def seed_worker(worker_id):
    cur_seed = np.random.get_state()[1][0]
    cur_seed += worker_id
    np.random.seed(cur_seed)
    random.seed(cur_seed)
