"""CLI entry point: run one INI or a directory of INIs (torchrun for DDP).

Usage:
    python runtask.py --task-path ./taskfiles
    torchrun --nproc_per_node=8 runtask.py --task-path ./taskfiles
"""
import argparse
import ast
import os
import sys
import traceback
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from diagnostics import plot_hist_summary, summary_hist
from solver import solve
from utils import (get_global_rank, get_local_rank, get_world_size, isin_ddp)

DEFAULT_CONFIG = './default_config.ini'


def set_seed(config):
    """Seed CPU/CUDA RNGs; offset by global rank for independent streams."""
    seed_str = config.get('Environment', 'seed')
    if seed_str != "None":
        seed = ast.literal_eval(seed_str) + get_global_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_torchdtype(config):
    dtype = config['Environment']['torch_dtype']
    dtype_map = {
        'float64': torch.float64,
        'float32': torch.float32,
        'float16': torch.float16
    }
    torch.set_default_dtype(dtype_map[dtype])


def get_dist_info():
    """TorchRun environment info + device selection."""
    local_rank, global_rank, world_size = (get_local_rank(),
                                           get_global_rank(),
                                           get_world_size())
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        warnings.warn("No GPU available, using CPU")
    if global_rank == 0:
        print(f"===== World Size: {world_size} | Main Device: {device} =====")
    return local_rank, global_rank, world_size, device


def init_distributed(device):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend,
                            device_id=device,
                            init_method="env://")
    dist.barrier()
    if get_global_rank() == 0:
        print(f"Distributed initialization successful ({backend}).\n")


def run_task(config, device, sav_name=None):
    """Run one task (with repeats) and save per-seed + summary results."""
    torch.set_default_device(device)
    set_torchdtype(config)
    set_seed(config)
    use_dist = get_world_size() > 1

    rep_time = config.getint('Example', 'repeat_time')
    issav_everytime = config.getboolean('Example',
                                        'save_result_for_every_repeat_time')
    hdict_list = []
    for r in range(rep_time):
        savname_r = None
        if sav_name is not None and (issav_everytime or r == 0):
            savname_r = f'{sav_name}_r{r}_'
        hist_dict = solve(config, use_dist, sav_name=savname_r)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        hdict_list.append(hist_dict)

    hist_df = pd.DataFrame()
    if get_global_rank() == 0:
        hist_df = summary_hist(hdict_list)
        if sav_name is not None:
            output_dir = config.get('Environment', 'output_dir')
            sav_path = f"{output_dir}/{sav_name}_"
            Path(sav_path).parent.mkdir(exist_ok=True, parents=True)
            with open(f'{sav_path}.ini', 'w') as f:
                config.write(f)
            hist_df.to_csv(f'{sav_path}summary_hist.csv')
            plot_hist_summary(hist_df, sav_path)
    return hist_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-path', default='./taskfiles')
    args = parser.parse_args()

    config_files = []
    if os.path.isfile(args.task_path) and args.task_path.endswith('.ini'):
        config_files = [args.task_path]
    elif os.path.isdir(args.task_path):
        for root, _, fs in os.walk(args.task_path):
            config_files += [
                os.path.join(root, f) for f in fs if f.endswith('.ini')
            ]
        config_files.sort()
    if not config_files:
        explicit = any(a.startswith('--task-path') for a in sys.argv[1:])
        if explicit:
            raise FileNotFoundError(
                f"--task-path {args.task_path!r} resolved to no .ini file; "
                f"refusing to fall back to {DEFAULT_CONFIG}")
        config_files = [DEFAULT_CONFIG]

    local_rank, global_rank, world_size, device = get_dist_info()
    print(f"Process {global_rank}/{world_size-1} | "
          f"Local Rank: {local_rank} | Device: {device}\n")

    if world_size > 1 and not dist.is_initialized():
        init_distributed(device)

    try:
        for file in config_files:
            if global_rank == 0:
                print(f"\n========== Starting Task: {file} ==========\n")
            config = ConfigParser(interpolation=ExtendedInterpolation())
            config.read(file, encoding='utf-8')
            run_task(config, device, sav_name=Path(file).stem)
            if isin_ddp():
                dist.barrier()
    except Exception as e:
        print(f"Error in process {global_rank}: {e}")
        traceback.print_exc()
        if isin_ddp():
            dist.destroy_process_group()
        raise
    finally:
        if global_rank == 0:
            print("\nAll tasks completed!")
        if isin_ddp():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
