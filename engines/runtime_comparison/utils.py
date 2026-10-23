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
from sample_budgets import chunk_shares, split_global_budget





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


def mc_mean_with_budget(x, per_chunk, M, *, use_dist, num_item_persize,
                        with_gradient=False, use_percent=0.4,
                        release_each_chunk=False):
    """Exact global-M chunked means, optionally with a vector numerator.

    ``per_chunk(local_count)`` returns (scalar_mean, vector_mean_or_None).
    All ranks traverse the same rounds. Ranks with no remaining paths
    contribute zeros and never call the sampler. Unequal last chunks are
    weighted by their actual counts, not averaged as equal-sized means.
    OOM retry decisions are synchronized before any result collective.
    """
    world = dist.get_world_size() if use_dist else 1
    rank = dist.get_rank() if use_dist else 0
    counts = split_global_budget(M, world, name='reference MC paths')
    templates = (torch.zeros_like(x[..., :1]),
                 torch.zeros_like(x[..., 1:]) if with_gradient else None)
    means = list(templates)
    cursor, completed, progress = 0, 0, 0.0
    status = torch.zeros((), dtype=torch.int64, device=x.device)
    clamp_key = (str(x.device), num_item_persize, x.dtype.itemsize)
    max_chunksize = _CHUNK_CLAMP.get(clamp_key)
    while cursor < max(counts):
        remaining = max(0, counts[rank] - cursor)
        capacity, error = max(counts) - cursor, None
        status.zero_()
        if remaining:
            try:
                capacity = min(capacity, get_safe_chunksize(
                    num_item_persize, x.dtype, x.device,
                    use_percent=use_percent))
            except Exception as err:
                error = err
                status.fill_(2)
        if use_dist:
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
        if status.item():
            raise RuntimeError('Cannot allocate a reference MC chunk on '
                               'one or more ranks') from error
        if max_chunksize is not None:
            capacity = min(capacity, max_chunksize)
        capacity_tensor = torch.tensor(capacity, dtype=torch.int64,
                                       device=x.device)
        if use_dist:
            dist.all_reduce(capacity_tensor, op=dist.ReduceOp.MIN)
        capacity = int(capacity_tensor.item())
        if capacity < 1:
            raise RuntimeError('Reference MC chunk capacity must be positive')
        shares = chunk_shares(counts, cursor, capacity)
        local_count = shares[rank]
        values = templates
        status.zero_()
        error = None
        try:
            if local_count:
                values = per_chunk(local_count)
                if len(values) != len(templates):
                    raise ValueError('Invalid reference MC result tuple')
                for value, template in zip(values, templates):
                    if template is None:
                        if value is not None:
                            raise ValueError('Unexpected MC gradient numerator')
                    elif value is None or value.shape != template.shape:
                        raise ValueError('Invalid reference MC result shape')
        except Exception as err:
            error = err
            is_oom = (isinstance(err, RuntimeError)
                      and 'out of memory' in str(err).lower())
            status.fill_(1 if is_oom else 2)
            values = templates
        if use_dist:
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
        code = int(status.item())
        if code == 2:
            raise RuntimeError('Reference MC sampler failed on one or '
                               'more ranks') from error
        if code == 1:
            max_chunksize = capacity // 2
            if max_chunksize < 1:
                raise RuntimeError('Reference MC out of memory even at '
                                   'one path per active rank') from error
            _CHUNK_CLAMP[clamp_key] = max_chunksize
            if rank == 0:
                print(f'Restricted by memory, reduce chunksize to {max_chunksize}')
            # Exception tracebacks may retain the failed chunk's tensors.
            # Release those references before querying memory on the retry.
            error = None
            values = templates
            free_cache(x.device)
            continue
        for i, (value, template) in enumerate(zip(values, templates)):
            if template is None:
                continue
            # Normalize before SUM to avoid an unnecessarily large transient.
            weighted = value.to(template.dtype) * (local_count / M)
            if use_dist:
                dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
            means[i] = means[i] + weighted
        cursor += capacity
        completed += sum(shares)
        if completed / M > progress + 0.01 or completed == M:
            progress = completed / M
            if rank == 0:
                print(f'Progress: {progress:.2%}, global paths: {completed}/{M}, '
                      f'chunk paths by rank: {shares}')
        if release_each_chunk:
            free_cache(x.device)
    if completed != M:
        raise RuntimeError(f'Reference MC count mismatch: {completed} != {M}')
    return tuple(means)

def mc_for_v(x, terminal_cost, samp_xte_intft, use_dist, M=10**6,
             combine='plus', multiplier=6, use_percent=0.4):
    """Monte Carlo estimate of the reference value v(x).

    Averages ``terminal_cost(X_T)`` (plus or times the accumulated running
    cost) over ``M`` terminal samples per point, with memory-adaptive
    chunking and optional DDP acceleration (all ranks must call together).

    ``multiplier`` budgets the per-path transient memory as
    ``multiplier * x.numel() * itemsize`` bytes.  The default 6 suits the
    elementwise families; the FFT-based SOCComplex path measures
    129.7 MiB/path at d=1e4/200 pts (probe 478300) and sets 18 via
    ``problem.mc_mem_multiplier``.
    """
    assert x.ndim >= 2
    if combine not in ('plus', 'multiply'):
        raise ValueError(f'Invalid combine type: {combine}')
    print(f"Monte-Carlo for reference solution on {x.device}...\n")

    def per_chunk(count):
        xte_chunk, ift_chunk = samp_xte_intft(x, num_mc=count)
        if combine == 'plus':
            value = terminal_cost(xte_chunk).mean(0) + ift_chunk.mean(0)
        else:
            value = (terminal_cost(xte_chunk) * torch.exp(ift_chunk)).mean(0)
        return value, None

    mean, _ = mc_mean_with_budget(
        x, per_chunk, M, use_dist=use_dist,
        num_item_persize=multiplier * x.numel(), use_percent=use_percent)
    return mean


