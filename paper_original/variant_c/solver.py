"""INI parsing, network construction, and the solve() driver.

Only the method of the paper is supported: MartNetLoss with pilot--system
path sampling. INI keys are documented in default_config.ini.
"""
import ast
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import examples as examples_module
from diagnostics import res_on_curve
from loss_martnet import MartNetLoss
from networks import AnchoredUNet, FCNet
from train import train
from utils import get_global_rank, get_local_rank

EXAMPLE_REGISTRY = {
    name: cls
    for name, cls in examples_module.__dict__.items()
    if isinstance(cls, type) and cls.__module__ == examples_module.__name__
}


def _time_uniform_rad(config, problem):
    """Radius of the uniform init for the time-column weights.

    INI ``[Network] time_uniform_rad`` overrides; the default is 1.0 for
    time-dependent problems (has_time_variable=True), None otherwise.
    """
    rad = config.getfloat('Network', 'time_uniform_rad', fallback=None)
    if rad is None and problem.has_time_variable:
        rad = 1.0
    return rad


def parse_problem(config, use_dist=False):
    """Instantiate the example; split batch/paths across DDP ranks."""
    name = config.get('Example', 'name')
    dim_x = config.getint('Example', 'dim_x')
    batch_size = config.getint('Training', 'num_xpts_batch')
    num_pilot_paths = config.getint('Training', 'num_pilot_paths')
    if use_dist:
        wd = dist.get_world_size()
        batch_size = max(1, batch_size // wd)
        num_pilot_paths = max(1, num_pilot_paths // wd)
    try:
        cls = EXAMPLE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Example '{name}' not found. "
                         f"Available: {sorted(EXAMPLE_REGISTRY)}")
    problem = cls(dim_x,
                  dt=eval(config.get('Training', 'dt')),
                  batch_size=batch_size,
                  num_pilot_paths=num_pilot_paths)
    problem.M_rc = config.getint('Example', 'M_rc', fallback=4096)
    
    
    
    
    problem.rc_x0_dist = config.getboolean('Example', 'rc_x0_dist',
                                           fallback=False)
    
    
    
    
    problem.rc_x0_origin = config.getboolean('Example', 'rc_x0_origin',
                                             fallback=False)
    
    
    
    problem.x0_point_origin = config.getboolean('Example', 'x0_point_origin',
                                                fallback=False)
    
    problem.M_mc = config.getint('Example', 'M_mc', fallback=problem.M_mc)
    
    
    
    
    problem.num_test_points = config.getint('Example', 'num_test_points',
                                            fallback=problem.num_test_points)
    problem.noise_sampler = config.get('Training', 'noise_sampler',
                                       fallback='mc')
    
    
    
    problem.u_fixed_ref = config.getboolean('Training', 'u_fixed_ref',
                                            fallback=False)
    
    
    
    problem.w_term_v = config.getfloat('Training', 'w_term_v', fallback=0.0)
    problem.w_term_u = config.getfloat('Training', 'w_term_u', fallback=0.0)
    
    
    
    
    problem.lam0 = config.getfloat('Training', 'lam0', fallback=1.0)
    problem.delta4 = config.getfloat('Training', 'delta4', fallback=0.0)
    problem.lam_bar = config.getfloat('Training', 'lam_bar', fallback=1.0)
    
    
    
    problem.w_mart_last = config.getfloat('Training', 'w_mart_last',
                                          fallback=1.0)
    
    
    
    problem.w_ctr_last = config.getfloat('Training', 'w_ctr_last',
                                         fallback=1.0)
    problem.anneal_last = config.getboolean('Training', 'anneal_last',
                                             fallback=False)
    problem.anneal_frac = config.getfloat('Training', 'anneal_frac',
                                          fallback=1.0)
    problem.max_iter = config.getint('Training', 'max_iter', fallback=0)
    
    
    
    
    
    if hasattr(problem, 'g_sin_amp'):
        problem.g_sin_amp = config.getfloat('Example', 'g_sin_amp',
                                            fallback=problem.g_sin_amp)
    
    
    if hasattr(problem, 'g_shell_A'):
        problem.g_shell_A = config.getfloat('Example', 'g_shell_A',
                                            fallback=problem.g_shell_A)
    if hasattr(problem, 'g_scale_B'):
        problem.g_scale_B = config.getfloat('Example', 'g_scale_B',
                                            fallback=problem.g_scale_B)
    
    
    
    
    
    if hasattr(problem, 'g_cusp_amp'):
        problem.g_cusp_amp = config.getfloat('Example', 'g_cusp_amp',
                                             fallback=problem.g_cusp_amp)
        cusp_s = config.get('Example', 'g_cusp_idx', fallback='')
        if cusp_s:
            problem.g_cusp_idx = tuple(
                int(v) for v in cusp_s.replace(' ', '').split(',') if v)
    
    
    
    if hasattr(problem, 'g_chan_amp'):
        problem.g_chan_amp = config.getfloat('Example', 'g_chan_amp',
                                             fallback=problem.g_chan_amp)
    
    
    if hasattr(problem, 'chan_gain'):
        problem.chan_gain = config.getfloat('Example', 'chan_gain',
                                            fallback=problem.chan_gain)
    
    
    
    
    
    problem.s_range = config.getfloat('Example', 's_range',
                                      fallback=problem.s_range)
    
    
    
    
    
    
    if hasattr(problem, 'phi_decay'):
        problem.phi_decay = config.getfloat('Example', 'phi_decay',
                                            fallback=problem.phi_decay)
    
    
    
    
    
    
    
    
    if hasattr(problem, 'drift_amp'):
        problem.drift_amp = config.getfloat('Example', 'drift_amp',
                                             fallback=problem.drift_amp)
    
    
    
    
    
    
    
    if hasattr(problem, 'c1'):
        problem.c1 = config.getfloat('Example', 'c1',
                                     fallback=problem.c1)
    
    
    
    
    
    
    
    if hasattr(problem, 'chan_dc'):
        problem.chan_dc = config.getfloat('Example', 'chan_dc',
                                          fallback=problem.chan_dc)
    
    
    
    
    
    
    
    if hasattr(problem, 'chan_coord'):
        problem.chan_coord = config.getint('Example', 'chan_coord',
                                           fallback=problem.chan_coord)
    if hasattr(problem, 'chan_spike'):
        problem.chan_spike = config.getfloat('Example', 'chan_spike',
                                             fallback=problem.chan_spike)
    
    
    
    
    
    
    
    if hasattr(problem, 'chain_alpha'):
        problem.chain_alpha = config.getfloat('Example', 'chain_alpha',
                                              fallback=problem.chain_alpha)
        problem.chain_sat = config.getfloat('Example', 'chain_sat',
                                            fallback=problem.chain_sat)
        problem.chain_cG = config.getfloat('Example', 'chain_cG',
                                           fallback=problem.chain_cG)
        problem.gate_g0 = config.getfloat('Example', 'gate_g0',
                                          fallback=problem.gate_g0)
        problem.gate_g1 = config.getfloat('Example', 'gate_g1',
                                          fallback=problem.gate_g1)
        problem.gate_omega = config.getfloat('Example', 'gate_omega',
                                             fallback=problem.gate_omega)
        problem.gate_kappa = config.getfloat('Example', 'gate_kappa',
                                             fallback=problem.gate_kappa)
        problem.chain_gate_coord = config.getint(
            'Example', 'chain_gate_coord',
            fallback=problem.chain_gate_coord)
        problem.win_amp = config.getfloat('Example', 'win_amp',
                                          fallback=problem.win_amp)
        problem.win_x = config.getfloat('Example', 'win_x',
                                        fallback=problem.win_x)
        problem.win_w = config.getfloat('Example', 'win_w',
                                        fallback=problem.win_w)
    
    
    
    
    
    if hasattr(problem, 'omega'):
        problem.omega = config.getfloat('Example', 'sparse_omega',
                                        fallback=problem.omega)
    if hasattr(problem, 'c_weights'):
        c_s = config.get('Example', 'sparse_c', fallback='')
        if c_s:
            problem.c_weights = tuple(
                float(v) for v in c_s.replace(' ', '').split(',') if v)
    if hasattr(problem, 'sparse_ref'):
        problem.sparse_ref = config.get('Example', 'sparse_ref',
                                        fallback=problem.sparse_ref)
        problem.sparse_qmc_exp = config.getint('Example', 'sparse_qmc_exp',
                                               fallback=problem.sparse_qmc_exp)
    
    
    
    if hasattr(problem, 'delta'):
        problem.delta = config.getfloat('Example', 'sparse_delta',
                                        fallback=problem.delta)
    
    
    
    if hasattr(problem, 'sparse_a'):
        problem.sparse_a = config.getfloat('Example', 'sparse_a',
                                           fallback=problem.sparse_a)
    
    
    
    if hasattr(problem, 'sparse_c0'):
        problem.sparse_c0 = config.getfloat('Example', 'sparse_c0',
                                            fallback=problem.sparse_c0)
    
    
    
    
    if hasattr(problem, 'k_dirs'):
        problem.k_dirs = config.getint('Example', 'sparse_k',
                                       fallback=problem.k_dirs)
    
    
    
    if hasattr(problem, 'sparse_m'):
        problem.sparse_m = config.getint('Example', 'sparse_m',
                                         fallback=problem.sparse_m)
    
    
    
    dt_ref_s = config.get('Example', 'dt_ref', fallback='')
    problem.dt_ref = eval(dt_ref_s) if dt_ref_s else None
    problem.qmc_pool_exp = config.getint('Training', 'qmc_pool_exp',
                                         fallback=16)
    return problem


def _build_net(config, dims, act_name, time_uniform_rad, shell_func=None,
               scale_lb=1.0, scale_ub=1.0, use_dist=False,
               force_no_autocast=False, net_wrapper=None):
    """_build_net implementation."""
    enable_autocast = config.getboolean('Environment',
                                        'enable_autocast',
                                        fallback=False) and not force_no_autocast
    net = FCNet(dims,
                act_func=getattr(nn, act_name),
                shell_func=shell_func,
                scale_lb=scale_lb,
                scale_ub=scale_ub,
                bias_uniform=config.getboolean('Network',
                                               'bias_uniform',
                                               fallback=True),
                batch_norm=config.getboolean('Network',
                                             'batch_norm',
                                             fallback=False),
                time_uniform_rad=time_uniform_rad,
                enable_autocast=enable_autocast,
                autocast_dtype=_str2dtype(
                    config.get('Environment',
                               'autocast_dtype',
                               fallback='float16')))
    if net_wrapper is not None:
        net = net_wrapper(net)
    if use_dist:
        loc_rank = get_local_rank()
        net = DDP(net.to(loc_rank), device_ids=[loc_rank])
    return net


def _str2dtype(s):
    return {
        'float32': torch.float32,
        'float64': torch.float64,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[s]


def parse_nets(config, problem, use_dist=False):
    """Build v_theta, u_alpha, rho plus optimizers and schedulers."""
    rad = _time_uniform_rad(config, problem)

    W_v = eval(config.get('Network', 'width_v'))
    W_u = eval(config.get('Network', 'width_u'))
    W_rho = eval(config.get('Network', 'width_rho'))
    H_v = config.getint('Network', 'num_hidden_v')
    H_u = config.getint('Network', 'num_hidden_u')
    H_rho = config.getint('Network', 'num_hidden_rho')

    v_theta = _build_net(config,
                         [problem.dim_x] + [W_v] * H_v + [1],
                         config.get('Network', 'act_v'),
                         rad,
                         use_dist=use_dist)
    
    
    
    
    
    
    
    anchor_pow_s = config.get('Network', 'u_anchor_pow', fallback='').strip()
    anchor_pow = float(anchor_pow_s) if anchor_pow_s else None
    if anchor_pow is not None and getattr(problem, 'u_fixed_ref', False):
        raise ValueError('u_anchor_pow and u_fixed_ref (u* substitution) are '
                         'mutually exclusive; unset one.')
    u_wrapper = None
    if anchor_pow is not None:
        u_wrapper = lambda net: AnchoredUNet(problem, anchor_pow, net)  
    u_alpha = _build_net(config,
                         [problem.dim_x] + [W_u] * H_u + [problem.dim_u],
                         config.get('Network', 'act_u'),
                         rad,
                         use_dist=use_dist,
                         net_wrapper=u_wrapper)

    rho_shell = getattr(torch, config.get('Network', 'rho_shell'))
    rho = _build_net(config,
                     [problem.dim_x] + [W_rho] * (H_rho + 1),
                     config.get('Network', 'act_rho'),
                     rad,
                     shell_func=lambda _x, y: rho_shell(y),
                     scale_lb=config.getfloat('Network', 'scale_lb_rho'),
                     scale_ub=config.getfloat('Network', 'scale_ub_rho'),
                     use_dist=use_dist,
                     force_no_autocast=True)

    nets = {'v_theta': v_theta, 'u_alpha': u_alpha, 'rho': rho}

    
    opt_name = config.get('Optimizer', 'optimizer', fallback='RMSprop')
    kwargs = ast.literal_eval(config.get('Optimizer', 'kwargs', fallback='{}'))
    decay_gap = config.getint('Optimizer', 'decay_stepgap')
    decay_rate = eval(config.get('Optimizer', 'decay_rate'))

    optim_desc, schs = [], []
    for tag, net in [('v', v_theta), ('u', u_alpha)]:
        lr0 = eval(config.get('Optimizer', f'lr0_{tag}'))
        optim = getattr(torch.optim, opt_name)(net.parameters(),
                                               lr=lr0,
                                               **kwargs)
        optim_desc.append(optim)
        schs.append(
            torch.optim.lr_scheduler.StepLR(optim,
                                            step_size=decay_gap,
                                            gamma=decay_rate))
    lr0_rho = eval(config.get('Optimizer', 'lr0_rho'))
    optim_rho = getattr(torch.optim, opt_name)(rho.parameters(),
                                               lr=lr0_rho,
                                               **kwargs)
    
    
    
    
    rho_decay_s = config.get('Optimizer', 'rho_decay_rate', fallback='')
    decay_rate_rho = eval(rho_decay_s) if rho_decay_s else decay_rate
    sch_rho = torch.optim.lr_scheduler.StepLR(optim_rho,
                                              step_size=decay_gap,
                                              gamma=decay_rate_rho)

    return nets, optim_desc, (optim_rho, ), schs + [sch_rho]


class _UStarModule(nn.Module):
    """_UStarModule implementation."""

    def __init__(self, problem):
        super().__init__()
        self._problem = problem
        
        
        self._probe = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return self._problem.u_star(x)


def _save_weights(nets, sav_path):
    """Save the final v_theta/u_alpha weights (CPU state_dicts) to
    ``<sav_path>weights.pt``; rho is not saved (not needed for evaluation).
    """
    state = {}
    for key in ('v_theta', 'u_alpha'):
        net = nets[key]
        if isinstance(net, DDP):
            net = net.module
        state[key] = {
            k: p.detach().cpu()
            for k, p in net.state_dict().items()
        }
    torch.save(state, f'{sav_path}weights.pt')


def solve(config, use_dist, sav_name=None):
    """Build the problem, run Algorithm 3.1, and save logs + curve data."""
    ip_time_gap = 1.0 if use_dist else 0.

    problem = parse_problem(config, use_dist=use_dist)
    if getattr(problem, 'x0_point_origin', False):
        
        
        print(f"x0_point_origin active: X_0 fixed at the single origin "
              f"(t0={problem.t0}, z0={problem.z0}) -- training-path starts, "
              f"pilot library and the t=0 test grid all collapse onto "
              f"(t0, z0*1_d)", flush=True)
    nets, optim_desc, optim_asc, schs = parse_nets(config,
                                                   problem,
                                                   use_dist=use_dist)
    
    
    
    
    if problem.u_fixed_ref:
        ustar = _UStarModule(problem)
        ustar.to(next(nets['v_theta'].parameters()).device)
        nets['u_alpha'] = ustar
        optim_desc = optim_desc[:1]
        schs = [schs[0], schs[-1]]
    problem.attach_nets(nets)

    log_func = problem.make_logfunc(nets['v_theta'], nets['u_alpha'])
    loss_clt = MartNetLoss(problem, nets, use_dist=use_dist)

    hist_dict = train(
        loss_clt,
        optim_desc,
        optim_asc,
        schs=schs,
        ip_time_gap=ip_time_gap,
        log_func=log_func,
        rc_func=lambda _it: problem.rc_metric(nets['u_alpha']),
        rc_loggap=config.getint('Training', 'rc_loggap', fallback=0),
        max_iter=config.getint('Training', 'max_iter'),
        num_descent=config.getint('Training', 'num_descent'),
        num_ascent=config.getint('Training', 'num_ascent'),
        enable_scaler=config.getboolean('Environment',
                                        'enable_scaler',
                                        fallback=True),
        ema_robust=config.getboolean('Training', 'ema_robust',
                                     fallback=True),
        rho_clip=config.getboolean('Training', 'rho_clip',
                                   fallback=False),
    )

    if sav_name is not None:
        output_dir = config.get('Environment', 'output_dir')
        sav_path = f"{output_dir}/{sav_name}"
        
        
        
        
        save_weights = config.get('Environment', 'save_weights',
                                  fallback='auto')
        if save_weights == 'auto':
            save_weights = 'SLURM_JOB_ID' in os.environ
        else:
            save_weights = save_weights.lower() in ('true', '1', 'yes')
        if save_weights and get_global_rank() == 0:
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            _save_weights(nets, sav_path)
        if get_global_rank() == 0:
            Path(sav_path).parent.mkdir(exist_ok=True, parents=True)
            pd.DataFrame(hist_dict).to_csv(f'{sav_path}log.csv', index=False)

        def xcurve_gen(num_points):
            args = (problem.dim_x, num_points, problem.s_range, False)
            names = list(problem.x0_curves.keys())
            s, xs = zip(*[f(*args) for f in problem.x0_curves.values()])
            return names, s, xs

        res_on_curve(nets['v_theta'],
                     problem.v,
                     xcurve_gen,
                     sav_path)

        if get_global_rank() == 0:
            pd.DataFrame(hist_dict).to_csv(f'{sav_path}log.csv', index=False)
    return hist_dict
