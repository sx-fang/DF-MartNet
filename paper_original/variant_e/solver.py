"""INI parsing, network construction, and the solve() driver.

Only the method of the paper is supported: MartNetLoss with pilot--system
path sampling. INI keys are documented in default_config.ini.
"""
import ast
import math
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import examples as examples_module
from diagnostics import res_on_curve
from loss_martnet import MartNetLoss
from networks import FCNet
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
    problem.noise_sampler = config.get('Training', 'noise_sampler',
                                       fallback='mc')
    problem.qmc_pool_exp = config.getint('Training', 'qmc_pool_exp',
                                         fallback=16)
    return problem


def _build_net(config, dims, act_name, time_uniform_rad, shell_func=None,
               scale_lb=1.0, scale_ub=1.0, use_dist=False,
               force_no_autocast=False):
    """Shared FCNet builder (v_theta / u_alpha / rho) with AMP options.

    ``force_no_autocast`` keeps rho in full precision under AMP runs (the
    blueprint's behavior: autocast only wraps v_theta / u_alpha).
    """
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
    u_alpha = _build_net(config,
                         [problem.dim_x] + [W_u] * H_u + [problem.dim_u],
                         config.get('Network', 'act_u'),
                         rad,
                         use_dist=use_dist)

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
    sch_rho = torch.optim.lr_scheduler.StepLR(optim_rho,
                                              step_size=decay_gap,
                                              gamma=decay_rate)

    return nets, optim_desc, (optim_rho, ), schs + [sch_rho]


def solve(config, use_dist, sav_name=None):
    """Build the problem, run Algorithm 3.1, and save logs + curve data."""
    ip_time_gap = 1.0 if use_dist else 0.

    problem = parse_problem(config, use_dist=use_dist)
    nets, optim_desc, optim_asc, schs = parse_nets(config,
                                                   problem,
                                                   use_dist=use_dist)
    problem.attach_nets(nets)

    log_func = problem.make_logfunc(nets['v_theta'])
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
    )

    if sav_name is not None:
        output_dir = config.get('Environment', 'output_dir')
        sav_path = f"{output_dir}/{sav_name}"
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
