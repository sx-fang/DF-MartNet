"""Pilot-system path sampling.

A bank of pilot paths is maintained across iterations. Mini-batches draw
space-time points from two disjoint path groups. The point budget need not
be divisible by the number of time steps. The configured schedule controls
pilot-bank refresh."""
import math

import torch

from utils import isin_ddp


def _xflat_to_path(x_flat, idx_at_tstep):
    """Group flat states into a time-indexed path list."""
    return [x_flat[idx] for idx in idx_at_tstep]


def paired_point_layout(problem):
    """Evaluation counts and local group sizes for the exact branch budget.

    AMC uses two branches per paired point and one ordinary branch for an
    odd remainder. Alternating *global* evaluation indices between groups
    also permits an empty local group when another rank supplies its data.
    """
    counts = tuple(getattr(problem, '_batch_point_counts',
                           (int(problem.batch_size),)))
    rank = int(getattr(problem, '_batch_rank', 0))
    if (not 0 <= rank < len(counts) or any(q <= 0 for q in counts)
            or counts[rank] != problem.batch_size):
        raise ValueError('Invalid per-rank training point budget')
    is_amc = getattr(problem, 'noise_sampler', 'mc') == 'amc'
    eval_counts = tuple((q + 1) // 2 if is_amc else q for q in counts)
    total = sum(eval_counts)
    if total < 2:
        raise ValueError('The paired loss needs at least two global evaluation '
                         'points (AMC needs at least three branch points)')
    num_eval = eval_counts[rank]
    offset = sum(eval_counts[:rank])
    first = (num_eval + 1) // 2 if offset % 2 == 0 else num_eval // 2
    return num_eval, (first, num_eval - first), total


def refresh_pilot_batch(problem, x_next, u_func, refresh_rate=0.2):
    """Return the next training mini-batch from the pilot path bank.

    ``batch_size`` remains the configured local branch-point budget. Group
    membership is assigned to paths before uniform time sampling, and is
    stored alongside the points for descent and ascent to share.
    """
    num_eval, sizes, total = paired_point_layout(problem)
    if x_next is None:
        if problem.num_pilot_paths < 2:
            raise ValueError('At least two pilot paths per rank are required')
        num_newpath = problem.num_pilot_paths
        problem._cumm_path = 0
        problem._xpath_idx = torch.arange(problem.num_pilot_paths)

        N = int((problem.te - problem.t0) / problem.dt)  
        if N < 1:
            raise ValueError('The time grid needs at least one step')
        problem._x_path = torch.empty((N, 0, problem.dim_x))

        num_batpath = int(math.ceil(problem.batch_size / N))
        
        
        num_batpath = max(2, int(math.ceil(num_batpath / 2) * 2))
        problem._num_batpath = min(num_batpath, problem.num_pilot_paths)
    else:
        problem._xpath_idx = problem._xpath_idx.roll(-problem._num_batpath)

        if problem._cumm_path > problem.num_pilot_paths:
            assert 0 <= refresh_rate <= 1
            num_newpath = math.ceil(problem.num_pilot_paths * refresh_rate)
            problem._cumm_path = 0
        else:
            num_newpath = 0
            problem._cumm_path += problem._num_batpath

    if num_newpath > 0:
        x0 = problem.x0_points(num_newpath)
        x0 = x0[torch.randperm(x0.shape[0])]
        with torch.no_grad():
            idx_at_tstep, x_domain, _ = problem.gen_points(x0, u_func=u_func)
        new_xpath = torch.stack(_xflat_to_path(x_domain, idx_at_tstep), dim=0)
        problem._x_path = torch.cat(
            [problem._x_path[:, num_newpath:], new_xpath], dim=1)
        rank = torch.distributed.get_rank() if isin_ddp() else 0
        print(f"Rank {rank}, Generated {num_newpath} new pilot paths. "
              f"Total paths: {problem._x_path.shape[1]}.\n")

    bank = problem._x_path
    selected = problem._xpath_idx[:problem._num_batpath]
    selected = selected[torch.randperm(selected.numel(), device=bank.device)]
    mid = selected.numel() // 2
    groups = (selected[:mid], selected[mid:])
    path_ids, time_ids = [], []
    for group, count in zip(groups, sizes):
        
        
        chosen = torch.randint(group.numel(), (count,), device=bank.device)
        path_ids.append(group[chosen])
        time_ids.append(torch.randint(bank.shape[0], (count,),
                                      device=bank.device))
    problem._batch_path_groups = groups
    problem._batch_path_ids = torch.cat(path_ids)
    problem._batch_time_ids = torch.cat(time_ids)
    problem._batch_group_sizes = sizes
    problem._batch_eval_count = num_eval
    problem._batch_global_eval_count = total
    problem._amc_num_pairs = (problem.batch_size // 2
                              if getattr(problem, 'noise_sampler', 'mc')
                              == 'amc' else 0)
    return bank[problem._batch_time_ids, problem._batch_path_ids]
