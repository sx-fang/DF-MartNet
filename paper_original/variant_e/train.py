"""Alternating descent-ascent training loop (the paper's Algorithm 3.1).

Each iteration performs ``num_descent`` descent steps on (v_theta, u_alpha)
followed by ``num_ascent`` ascent steps on rho, with AMP gradient scaling
and EMA-based gradient clipping (clip threshold = ema_gradnorm * 10).
"""
import time

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from utils import get_local_rank, isin_ddp


def _append_hist(hist, keys, vals):
    if len(hist) == 0:
        hist.update({k: [v] for k, v in zip(keys, vals)})
    else:
        for k, v in zip(keys, vals):
            hist[k].append(v)


def _raise_if_not_finite(loss, name, it):
    """Abort training when a loss is NaN/Inf (fail-fast, saves GPU hours).

    Under DDP the finiteness flag is all-reduced BEFORE backward, so every
    rank raises at the same point and no rank hangs in a later collective.
    """
    ok = torch.tensor(0.0 if torch.isfinite(loss).item() else 1.0,
                      device=loss.device)
    if isin_ddp():
        dist.all_reduce(ok, op=dist.ReduceOp.MAX)
    if ok.item() > 0:
        rank = dist.get_rank() if isin_ddp() else 0
        raise ValueError(
            f"Non-finite {name}={loss.item()} at iteration {it} "
            f"(detected on rank {rank}); training aborted.")


def _make_gpumem_recorder(device_type):
    """Peak-memory recorder (DDP-aware), resetting stats after each read."""
    if device_type != 'cuda':
        return lambda: {}

    if isin_ddp():
        num_gpus = torch.cuda.device_count()
        current_rank = get_local_rank()
        torch.cuda.reset_peak_memory_stats(current_rank)

        def recorder():
            mem_list = [torch.tensor(0.) for _ in range(num_gpus)]
            mem_max = torch.cuda.max_memory_allocated(current_rank) / 1024**2
            dist.all_gather(mem_list, torch.tensor(mem_max))
            mem_dict = {
                f'peak_memory_cuda{i}_MB': mem_list[i].cpu().item()
                for i in range(num_gpus)
            }
            torch.cuda.reset_peak_memory_stats(current_rank)
            return mem_dict
    else:
        torch.cuda.reset_peak_memory_stats()

        def recorder():
            mem_max = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()
            return {'peak_memory_cuda0_MB': mem_max}

    return recorder


def _make_rate_limiter(print_gap=1.0):
    """Throttle console prints; the decision is broadcast under DDP."""
    last_time = None
    use_dist = isin_ddp()

    def rate_limiter():
        nonlocal last_time
        now = time.time()
        to_print = last_time is None or now - last_time >= print_gap
        if to_print:
            last_time = now
        if use_dist:
            flag = torch.tensor(int(to_print))
            dist.broadcast(flag, src=0)
            to_print = bool(flag.item())
            synced = torch.tensor([last_time if last_time else 0.0],
                                  dtype=torch.float64)
            dist.broadcast(synced, src=0)
            last_time = float(synced.item())
        return to_print

    return rate_limiter


def train(loss_clt,
          optim_desc,
          optim_asc,
          max_iter=1000,
          num_descent=2,
          num_ascent=1,
          schs=(),
          log_func=None,
          rc_func=None,
          rc_loggap=0,
          ip_time_gap=0.,
          enable_scaler=True,
          factor_clip_grad=10.):
    """Run Algorithm 3.1 and return the training history dict.

    Timing model: ``rt_solve`` is the pure training wall time (the true
    algorithmic cost), ``rt_log`` accumulates every evaluation of the error
    metrics (RE / RC / losses logging). Under DDP each rank measures its own
    wall clock; the per-iteration ``torch.cuda.synchronize()`` flushes all
    async GPU/NCCL work before each read, and all ranks execute the same
    collectives at the same iterations, so rank 0's rt_solve is the
    wall-clock cost of the synchronous training loop.

    ``rc_func(it)`` (the expensive RC metric) runs only when
    ``it % rc_loggap == 0`` or ``it == max_iter``; its cost lands in rt_log.
    """
    log_func = (lambda _it: {}) if log_func is None else log_func
    hist_dict = {}

    rt0 = time.time()
    rt_log = 0.
    loss_clt.init_train()
    rate_limiter = _make_rate_limiter(print_gap=ip_time_gap)
    mem_recorder = _make_gpumem_recorder(loss_clt.device.type)
    scaler = GradScaler(device=loss_clt.device, enabled=enable_scaler)

    ema_gradnorm = 1.
    for it in range(max_iter + 1):

        loss_clt.init_desc()
        for _ in range(num_descent):
            loss_desc = loss_clt.loss_desc()
            _raise_if_not_finite(loss_desc, 'loss_desc', it)
            scaler.scale(loss_desc).backward()
            for opt in optim_desc:
                scaler.unscale_(opt)
                grad_norm = clip_grad_norm_(
                    opt.param_groups[0]['params'],
                    ema_gradnorm * factor_clip_grad)
                ema_gradnorm = 0.99 * ema_gradnorm + 0.01 * grad_norm
                scaler.step(opt)
                opt.zero_grad()
            scaler.update()

        loss_clt.init_asc()
        for _ in range(num_ascent):
            loss_asc = loss_clt.loss_asc()
            _raise_if_not_finite(loss_asc, 'loss_asc', it)
            loss_asc.backward()
            for opt in optim_asc:
                opt.step()
                opt.zero_grad()

        for sch in schs:
            sch.step()

        if loss_clt.device.type == 'cuda':
            torch.cuda.synchronize()

        rt_total_1 = time.time() - rt0
        with torch.no_grad():
            newlog_dict = log_func(it)
        newlog_dict.update(loss_clt.log_func())
        if rc_func is not None and rc_loggap > 0 and (
                it % rc_loggap == 0 or it == max_iter):
            newlog_dict.update(rc_func(it))
        else:
            newlog_dict.update({'rc': float('nan'), 'cost_hat': float('nan')})
        rt_total_2 = time.time() - rt0
        rt_log += rt_total_2 - rt_total_1

        mem_dict = mem_recorder()
        keys = ['it', 'rt_solve', 'rt_log'] + list(newlog_dict.keys()) + list(
            mem_dict.keys())
        vals = [it, rt_total_2 - rt_log, rt_log] + list(
            newlog_dict.values()) + list(mem_dict.values())
        _append_hist(hist_dict, keys, vals)

        if rate_limiter() or it in (0, max_iter):
            lr = optim_desc[0].param_groups[0]['lr']
            global_rank = dist.get_rank() if isin_ddp() else 0
            mem_peak = mem_dict.get(f'peak_memory_cuda{global_rank}_MB',
                                    float('nan'))
            pr_str = (f"global_rank: {global_rank}, "
                      f"peak_gpu_memory: {mem_peak:.5g}\n"
                      f"iter step: [{it}/{max_iter}], lr: {lr:.5}\n"
                      f"rt_total: {rt_total_2:.2f}, rt_log: {rt_log:.2f}\n"
                      f"batch_size: {loss_clt.problem.batch_size}, "
                      f"pilot_paths: {loss_clt.problem.num_pilot_paths}\n")
            pr_str += "\n".join(f'{k}: {v}' for k, v in newlog_dict.items())
            pr_str += '\n'
            print(pr_str)

    loss_clt.finalize_train()
    return hist_dict
