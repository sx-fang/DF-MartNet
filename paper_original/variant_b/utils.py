"""Small utilities: distributed helpers, initial-point curves, MC reference.

The symbol names follow the companion paper:
``s`` is the scalar parameter of the initial-point curves (s * 1_d),
``M`` is the Monte Carlo sample size for the reference solution.
"""
import gc
import os
from typing import Sequence, Union

import psutil
import torch
import torch.distributed as dist





def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank():
    return int(os.environ.get("RANK", 0))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def isin_ddp():
    return dist.is_available() and dist.is_initialized()





def get_grid(s_range: Union[float, Sequence[float]],
             num_points: int,
             randomize=True):
    """Grid of the curve parameter s; random (sorted) or equispaced."""
    if isinstance(s_range, Sequence):
        left_end, right_end = s_range
    else:
        left_end, right_end = -s_range, s_range
    if randomize:
        grid = torch.rand(num_points) * (right_end - left_end) + left_end
        grid = torch.sort(grid)[0]
    else:
        grid = torch.linspace(left_end, right_end, num_points)
    return grid


def diag_curve(dim_x, num_points, s_range, randomize=True):
    """Diagonal curve z = s * 1_d (spatial part only)."""
    s_coord = get_grid(s_range, num_points, randomize=randomize)
    e_diag = torch.ones((dim_x, ))
    return s_coord, torch.outer(s_coord, e_diag)


def t_diag_curve(dim_x, num_points, s_range, randomize=True, t=0.):
    """Diagonal curve x = (t, s * 1_d) in the augmented (t, z) state."""
    assert dim_x >= 2
    s_coord, zdiag = diag_curve(dim_x - 1, num_points, s_range,
                                randomize=randomize)
    xdiag = torch.concat([torch.full((num_points, 1), t), zdiag], dim=1)
    return s_coord, xdiag


def manifold_curve(dim_x, num_points, s_range, randomize=True):
    """Oscillatory 1D manifold through R^d (spatial part only)."""
    s_coord = get_grid(s_range, num_points, randomize=randomize)
    e_vec = torch.arange(1, dim_x + 1)
    x_diag = torch.outer(s_coord, torch.sign(torch.sin(e_vec)))
    x = x_diag + torch.cos(e_vec + s_coord.unsqueeze(-1) * torch.pi)
    return s_coord, x


def t_manifold_curve(dim_x, num_points, s_range, randomize=True, t=0.):
    """Manifold curve x = (t, manifold(s)) in the augmented state."""
    assert dim_x >= 2
    s_coord, z_manifold = manifold_curve(dim_x - 1, num_points, s_range,
                                         randomize=randomize)
    x_manifold = torch.concat([torch.full((num_points, 1), t), z_manifold],
                              dim=1)
    return s_coord, x_manifold


def split_number(num, num_parts):
    """Split ``num`` into ``num_parts`` nearly equal integer parts."""
    part_size = num // num_parts
    parts = [part_size] * num_parts
    for i in range(num % num_parts):
        parts[i] += 1
    return parts





def time_mask(x, t0, te):
    """True for states with time coordinate in [t0, T)."""
    t = x[..., 0]
    eps = 10 * torch.finfo(t.dtype).eps
    return (t >= t0) & (t < te - eps)


def project_onto_t0te(x, t0, te):
    """Clamp the time coordinate to [t0, T]."""
    x[..., [0]] = torch.clamp(x[..., [0]], min=t0, max=te)
    return x


def free_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'cpu':
        gc.collect()





def get_safe_chunksize(num_item_persize, dtype, device, use_percent=0.5):
    """Adaptive MC chunk size that fits into the available device memory."""
    if device.type == 'cuda':
        avail_mem = torch.cuda.mem_get_info(device)[0]
    elif device.type == 'cpu':
        avail_mem = psutil.virtual_memory().available
    else:
        raise ValueError(f'Unsupported device type: {device.type}')
    safe_chunksize = avail_mem * use_percent // (num_item_persize *
                                                 dtype.itemsize)
    safe_chunksize = int(safe_chunksize)
    if safe_chunksize == 0:
        raise MemoryError(
            f'The memory of {device} is not enough for chunksize >= 1.')
    return safe_chunksize






_CHUNK_CLAMP = {}


def mc_for_v(x, terminal_cost, samp_xte_intft, use_dist, M=10**6,
             combine='plus', multiplier=6, use_percent=0.4):
    """mc_for_v implementation."""
    assert x.ndim >= 2
    if use_dist:
        world_size = dist.get_world_size()
        M = int(M // world_size) + 1
        rank = dist.get_rank()
    else:
        rank = 0

    num_item_persize = multiplier * x.numel()
    cum_size = 0
    cum_mean = 0.
    progress = 0.
    clamp_key = (str(x.device), multiplier, x.dtype.itemsize)
    max_chunksize = _CHUNK_CLAMP.get(clamp_key)
    print(f"Monte-Carlo for reference solution on {x.device}...\n")
    while cum_size < M:
        try:
            chunksize = get_safe_chunksize(num_item_persize, x.dtype,
                                           x.device, use_percent=use_percent)
            if max_chunksize is not None:
                chunksize = min(chunksize, max_chunksize)
            chunksize = max(1, min(chunksize, M - cum_size))
            if use_dist:
                chunksize_tensor = torch.tensor(chunksize, device=x.device)
                dist.all_reduce(chunksize_tensor, op=dist.ReduceOp.MIN)
                chunksize = int(chunksize_tensor.item())
            xte_chunk, ift_chunk = samp_xte_intft(x, num_mc=chunksize)

            if combine == 'plus':
                new_mean = terminal_cost(xte_chunk).mean(0) + ift_chunk.mean(0)
            elif combine == 'multiply':
                new_mean = (terminal_cost(xte_chunk) *
                            torch.exp(ift_chunk)).mean(0)
            else:
                raise ValueError(f"Invalid combine type: {combine}")
            cum_size += chunksize
            if use_dist:
                dist.all_reduce(new_mean, op=dist.ReduceOp.SUM)
                new_mean = new_mean / world_size
            new_rate = chunksize / cum_size
            cum_mean = (1 - new_rate) * cum_mean + new_rate * new_mean

            if (cum_size / M > progress + 0.01) or (cum_size == M):
                progress = cum_size / M
                if rank == 0:
                    print(f"Progress: {progress:.2%}, "
                          f"chunksize per rank: {chunksize}")
        except RuntimeError as err:
            if 'out of memory' in str(err):
                chunksize = int(chunksize // 2)
                max_chunksize = chunksize
                _CHUNK_CLAMP[clamp_key] = chunksize
                print(f"Restricted by memory, reduce chunksize to {chunksize}")
                if chunksize == 0:
                    raise err
                
                
                
                
                free_cache(x.device)
            else:
                raise err
    return cum_mean
