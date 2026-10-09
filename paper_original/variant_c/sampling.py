"""Pilot--system path sampling (the paper's Section 3.4 and Algorithm 3.1).

A bank of pilot paths is maintained across iterations. Each training
mini-batch is a slice of the bank; when the cumulative consumption exceeds
the bank size, a fraction ``refresh_rate`` of the paths is regenerated with
the current policy u_alpha. The pilot only decides *where* the martingale
condition is tested, never its correctness, so lazy/path-wise refresh does
not bias the loss.
"""
import math

import torch

from utils import isin_ddp


def _xflat_to_path(x_flat, idx_at_tstep):
    """Group flat states into a time-indexed path list."""
    return [x_flat[idx] for idx in idx_at_tstep]


def refresh_pilot_batch(problem, x_next, u_func, refresh_rate=0.2):
    """Return the next training mini-batch from the pilot path bank.

    ``x_next is None`` marks the first call (bank allocation and initial
    generation); later calls roll the bank and periodically regenerate a
    fraction of the paths with the policy ``u_func``.
    """
    if x_next is None:
        num_newpath = problem.num_pilot_paths
        problem._cumm_path = 0
        problem._xpath_idx = torch.arange(problem.num_pilot_paths)

        N = int((problem.te - problem.t0) / problem.dt)  
        problem._x_path = torch.empty((N, 0, problem.dim_x))

        num_batpath = int(math.ceil(problem.batch_size / N))
        if int(N // 2) != 0:
            
            num_batpath = int(math.ceil(num_batpath / 2) * 2)
        problem._num_batpath = num_batpath
        
        problem.batch_size = N * problem._num_batpath
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

    return problem._x_path[:,
                           problem._xpath_idx[:problem._num_batpath]].flatten(
                               0, 1)
