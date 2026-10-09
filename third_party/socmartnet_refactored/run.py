"""Numerical implementation implementation."""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from socmartnet import (FBSNN, HJBLQ, LinearParabolicSin, LinearSinAC,
                        NonDegHJB, ShiftTargetHJB, SOCMartNet, control_net,
                        relative_l1, test_net, value_net)
from socmartnet import evaluate

SOC_EXAMPLES = ('hjb', 'hjblq', 'shifttarget')
PDE_EXAMPLES = ('linear', 'semilinear', 'linsin')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--example', choices=SOC_EXAMPLES + PDE_EXAMPLES,
                   required=True)
    p.add_argument('--method', choices=['socmartnet', 'prabmartnet', 'fbsnn'],
                   default=None,
                   help='default: socmartnet (Alg. 3.1) for SOC examples, '
                        'prabmartnet (Alg. 3.2) otherwise; fbsnn is the paper '
                        'benchmark (Eq. (4.1)) for parabolic examples')
    p.add_argument('--terminal', choices=['smooth', 'oscillatory'],
                   default='smooth', help='terminal g for semilinear/hjb')
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--te', type=float, default=1.0)
    p.add_argument('--preset', choices=['v3paper', 'cube', 'sisc'],
                   default='v3paper')
    p.add_argument('--max-iter', type=int, default=None,
                   help='default: 1000 (d<=500, v3paper) / 2000 (d>500); '
                        'cube preset: 2000 iff d>100; sisc preset: 1000 '
                        '(paper Table 1 caliber; Table 2 uses 6000)')
    p.add_argument('--width', type=int, default=None,
                   help='override hidden width of u/v nets (Table 1 W=256 '
                        'row: --width 256)')
    p.add_argument('--num-hidden', type=int, default=None,
                   help='override number of hidden layers of u/v nets')
    p.add_argument('--r-dim', type=int, default=None,
                   help='override adversarial net output dimension r')
    p.add_argument('--lr-uv', type=float, default=None,
                   help='override u/v learning rate (archived-caliber probes)')
    p.add_argument('--lr-rho', type=float, default=None,
                   help='override adversarial-net learning rate (archived '
                        'calibers use 10 x lr_uv; paper delta3 = 1e-2)')
    p.add_argument('--batch-size', type=int, default=None,
                   help='override minibatch size with a single constant '
                        'segment (sisc preset: 256 if d<1000 else 128); '
                        'replaces the preset batsize schedule')
    p.add_argument('--num-paths', type=int, default=10**5,
                   help='training path-pool size M; v3r2 R1 path-renewal '
                        'caliber uses the epochsize as the pool, e.g. '
                        '--num-paths 10000 --renew-frac 0.2')
    p.add_argument('--renew-frac', type=float, default=0.,
                   help='v3r2: fraction of the path pool regenerated at each '
                        'epoch boundary (authors R1 rate_newpath=0.2); '
                        '0 disables (v3r1 behavior)')
    p.add_argument('--fd-residual', action='store_true',
                   help='v3r2 Option A: use the R1 finite-difference '
                        'martingale residual (lambda-free loss=mart+ctr, '
                        'u-gradient bug fixed) instead of the autograd-H '
                        'trapezoid + lambda augmentation')
    p.add_argument('--num-dt', type=int, default=100)
    p.add_argument('--lam-bar', type=float, default=None,
                   help='default 1e3 (v3paper/sisc) / 1e4 (cube); paper '
                        'Sec. 4.4 uses 100 for d>=1000, T=0.01')
    p.add_argument('--debias-mart', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='v3r3: train on the unbiased split-half estimator '
                        'G1.G2 of |G|^2 (path-axis halves); lambda '
                        'bookkeeping keeps the raw |G|^2. '
                        '--no-debias-mart recovers v3r2 bitwise')
    p.add_argument('--controlled-pool', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='v3r4 (train_fd only): renew the path pool with '
                        'CONTROLLED paths simulated online under the current '
                        'u_alpha, storing generation-time controls u_old; '
                        'the FD residual corrects only the lag '
                        '2(u-u_old).grad v (no lag bias, no double count). '
                        'Requires --renew-frac > 0; initial pool stays '
                        'pilot (u_old=0). Default off recovers v3r3 bitwise')
    p.add_argument('--antithetic', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='v3r5 (train_fd only, requires --fd-residual): '
                        'antithetic (AMC) variance reduction -- reflected '
                        'Brownian twin paths share x0 and controls; the '
                        'minibatch is drawn pair-preservingly so the debias '
                        'split-half stays exactly unbiased. Default off '
                        'recovers v3r4 bitwise')
    p.add_argument('--selftest-antithetic', action='store_true',
                   help='run the v3r5 antithetic structural self-test (twin '
                        'reflection identities, pair-clean sampling, renewal '
                        'adjacency) and exit before training')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpus', type=int, default=-1,
                   help='number of GPUs to use; -1 = all visible')
    p.add_argument('--dtype', choices=['float64', 'float32'],
                   default='float64',
                   help='default float64 (validated v3 caliber); float32 '
                        'recovers the R1-era archived runs')
    p.add_argument('--x0-scale', type=float, default=1.0,
                   help='half-length of the D_0 segments (s in [-scale, '
                        'scale])')
    p.add_argument('--unit-ball', action='store_true',
                   help='normalize the diagonal segment to the unit sphere '
                        '(linear/semilinear/hjb only)')
    p.add_argument('--x0-mode', choices=['region', 'point'], default='region',
                   help='point: D_0 = {0} (Sec. 4.8, hjblq/shifttarget only)')
    p.add_argument('--region', choices=['s1s2', 's2s3', 's2'], default=None,
                   help='D_0 set for hjblq/shifttarget; default s2s3 for '
                        'hjblq (Sec. 4.4), s2 for shifttarget (Sec. 4.6)')
    # HJBLQ family parameters (Sec. 4.4-4.8)
    p.add_argument('--bval', type=float, default=None,
                   help='drift b of hjblq; default 1.0')
    p.add_argument('--delta0', type=float, default=None,
                   help='diffusion scale delta_0 of hjblq (sigma = delta0 '
                        'sqrt(2)); default 0.2 (HJB-2); use 0.1 for HJB-3')
    p.add_argument('--delc', type=float, default=None,
                   help='Terminal regularisation coefficient: delc/pi in sin(1/(eps0 + x^2)); default 0.3 for hjblq and 0.1 for semilinear/hjb.')
    p.add_argument('--eps-purb', type=float, default=None,
                   help='Sec. 4.7 perturbation: add eps*sin(1_d^T k) to the '
                        'Hamiltonian (eps = 1, 1/2, 1/4, 1/8)')
    p.add_argument('--target-shift', type=float, default=3.0,
                   help='shifttarget: target point s*1_d coordinate; default 3.0.')
    p.add_argument('--allen-cahn', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='linsin: include the v - v^3 Allen-Cahn source; enabled by default. --no-allen-cahn removes this source term.')
    # logging / evaluation artifacts
    p.add_argument('--extended-log', action='store_true',
                   help='append rel_linf / mean_vtrue_t0 (/ cost) columns to '
                        'the history CSV; default on for --preset sisc')
    p.add_argument('--cost-track', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='log J(u) by Monte-Carlo each iteration (default on '
                        'for shifttarget; consumes RNG draws)')
    p.add_argument('--cost-paths', type=int, default=256,
                   help='number of MC paths for the J(u) estimate (Sec. 4.6)')
    p.add_argument('--cost-num-dt', type=int, default=100)
    p.add_argument('--save-curves', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='write v(0, .) curve CSVs on the D_0 generators at '
                        'the end of training (default on for sisc region '
                        'runs, off otherwise)')
    p.add_argument('--curve-points', type=int, default=100)
    p.add_argument('--eval-ptx', action='store_true',
                   help='Sec. 4.5: write s -> v(r, s 1_d + r 1_d) CSVs for '
                        'r = 0.125, 0.25')
    p.add_argument('--eval-path-re', action='store_true',
                   help='Sec. 4.5: write RE(t) along 8 fresh pilot paths')
    p.add_argument('--out', type=str, default='outputs')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def hyperparams(args):
    """Hyperparameter dict; the v3paper/cube branches are value-identical to
    the validated baseline."""
    d = args.dim
    if args.preset == 'cube':
        # config behind the authors' v3-era archived CSVs
        hp = dict(num_hidden=3, width=2 * (d + 10), r=2 * d + 500,
                  lr_uv=1e-3, lr_rho=1e-2,
                  max_iter=2000 if d > 100 else 1000, lam_bar=1e4,
                  batsize=[200, 400, 800, 1600], sch_mode='v3')
    elif args.preset == 'sisc':
        # accepted-paper Sec.-4 intro + authors' R1-era archived configs
        # d >= 1000 uses the producing caliber of the archived HJB2b INIs
        # (SOCMN144 et al.): lr exponent 0.8, batch 128, rho lr 10 x value
        
        
        # d=1000 with RE ~1).
        # lr exponent: paper 0.5 for d < 1000, archived/taskmaker 0.8 for
        # d >= 1000 (at d=1000 the two disagree; archived caliber wins since
        # the paper value failed with RE ~1).
        expo = -0.5 if d < 1000 else -0.8
        lr_uv = 3e-3 * d**expo
        hp = dict(num_hidden=6, width=d + 10, r=600,
                  lr_uv=lr_uv, lr_rho=1e-2 if d < 1000 else 10 * lr_uv,
                  max_iter=1000, lam_bar=1e3,
                  batsize=[256 if d < 1000 else 128], sch_mode='quantized')
    else:  # v3paper
        hp = dict(num_hidden=4, width=2 * (d + 10), r=2 * d + 300,
                  lr_uv=3e-3 * d**(-0.5 if d <= 1000 else -0.8),
                  lr_rho=1e-2,
                  max_iter=1000 if d <= 500 else 2000, lam_bar=1e3,
                  batsize=[200, 400, 800, 1600], sch_mode='v3')
    if args.max_iter is not None:
        hp['max_iter'] = args.max_iter
    if args.lam_bar is not None:
        hp['lam_bar'] = args.lam_bar
    for cli_key, hp_key in (('width', 'width'), ('num_hidden', 'num_hidden'),
                            ('r_dim', 'r'), ('lr_uv', 'lr_uv'),
                            ('lr_rho', 'lr_rho')):
        if getattr(args, cli_key) is not None:
            hp[hp_key] = getattr(args, cli_key)
    if args.batch_size is not None:
        hp['batsize'] = [args.batch_size]
    return hp


def build_problem(args):
    t0 = torch.tensor(0.)
    te = torch.tensor(args.te)
    if args.example == 'linear':
        if args.x0_mode == 'point':
            raise ValueError('--x0-mode point is only supported for '
                             'hjblq/shifttarget')
        return LinearParabolicSin(args.dim, t0=t0, te=te,
                                  x0_scale=torch.tensor(args.x0_scale),
                                  unit_ball=args.unit_ball)
    if args.example in ('semilinear', 'hjb'):
        if args.x0_mode == 'point':
            raise ValueError('--x0-mode point is only supported for '
                             'hjblq/shifttarget')
        delc = args.delc if args.delc is not None else 0.1
        return NonDegHJB(args.dim, t0=t0, te=te, terminal=args.terminal,
                         delc=delc, x0_scale=torch.tensor(args.x0_scale),
                         unit_ball=args.unit_ball)
    common = dict(t0=t0, te=te, x0_scale=args.x0_scale,
                  x0_mode=args.x0_mode)
    if args.example == 'hjblq':
        return HJBLQ(args.dim, b_val=args.bval if args.bval is not None else 1.,
                     delta0=args.delta0 if args.delta0 is not None else 0.2,
                     delc=args.delc if args.delc is not None else 0.3,
                     eps_purb=args.eps_purb,
                     region=args.region or 's2s3', **common)
    if args.example == 'shifttarget':
        return ShiftTargetHJB(args.dim, target_shift=args.target_shift,
                              region=args.region or 's2', **common)
    # linsin
    if args.x0_mode == 'point':
        raise ValueError('--x0-mode point is only supported for '
                         'hjblq/shifttarget')
    return LinearSinAC(args.dim, t0=t0, te=te, allen_cahn=args.allen_cahn,
                       x0_scale=args.x0_scale)


def antithetic_selftest(problem, Dt, N):
    """v3r5 structural self-test for antithetic sampling (no training).

    All checks run on a tiny pool of 4 pairs:
      1. twin slots share the same start point (bitwise);
      2. first-step reflection identity (x+ + x-)/2 == x0 + mu(t0,x0)*Dt
         (the reflected increment cancels in the pair mean), while twins
         do differ (reflected noise actually applied);
      3. pair-preserving minibatch indices are twin-adjacent with a
         pair-clean debias split-half (mirrors the train_fd construction);
      4. renew_paths replaces whole pairs (pool stays twin-adjacent);
      5. odd path counts are rejected (fail-fast guard).
    Prints one PASS/FAIL line per check; returns the conjunction.
    """
    solver = SOCMartNet(Dt, problem.mu, problem.sigma, problem.H,
                        problem.v_term, problem.dim_w, t0=problem.t0,
                        H_depends_on_vx=problem.H_depends_on_vx)
    torch.manual_seed(0)
    x0 = problem.x0_points(8)
    _, xt = solver.simu_paths(x0, 4, antithetic=True)
    checks = {}
    checks['twin_start'] = bool(torch.equal(xt[0, 0::2], xt[0, 1::2]))
    pair_mean1 = 0.5 * (xt[1, 0::2] + xt[1, 1::2])
    drift_ref = xt[0, 0::2] + problem.mu(problem.t0, xt[0, 0::2]) * Dt
    checks['reflection_id'] = bool(torch.allclose(pair_mean1, drift_ref,
                                                  rtol=1e-4, atol=1e-5))
    checks['twin_differ'] = bool((xt[1, 0::2] - xt[1, 1::2]).abs().mean()
                                 > 1e-4)
    bat_pair = torch.randperm(4)[:2]
    bat_idx = torch.stack([2 * bat_pair, 2 * bat_pair + 1],
                          dim=1).reshape(-1)
    checks['pair_sampling'] = bool((((bat_idx[0::2] + 1) == bat_idx[1::2])
                                    & (bat_idx[0::2] % 2 == 0)).all())
    xt2 = solver.renew_paths(x0, 4, 2, xt, antithetic=True)
    checks['renew_adjacency'] = bool(torch.equal(xt2[0, -2], xt2[0, -1])
                                     and torch.equal(xt2[0, 0], xt2[0, 1])
                                     and xt2.shape[1] == xt.shape[1])
    try:
        solver.simu_paths(x0[:7], 2, antithetic=True)
        checks['odd_rejected'] = False
    except ValueError:
        checks['odd_rejected'] = True
    for name, flag in checks.items():
        print(f'antithetic self-test {name}: {"PASS" if flag else "FAIL"}')
    return all(checks.values())


def worker(rank, world_size, args):
    print(f"Running process on rank {rank}.")
    setup_seed(args.seed + rank)

    if world_size > 1:
        import os
        import zlib
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            # Deterministic per-job port: all ranks derive the same value from
            # shared environment without communicating (chicken-and-egg with
            # init_process_group). Distinct jobs co-located on one node get
            # distinct ports -- the authors' platform INI used
            # master_port=random; the old fixed 12355 made two co-located
            
            port_seed = os.environ.get('SLURM_JOB_ID', '0') + '_' + str(args.seed)
            os.environ['MASTER_PORT'] = str(
                20000 + zlib.crc32(port_seed.encode()) % 40000)
        try:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        except RuntimeError:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.set_default_dtype(torch.float64 if args.dtype == 'float64'
                            else torch.float32)
    torch.set_default_device(f'cuda:{rank}')

    hp = hyperparams(args)
    if args.antithetic and not args.fd_residual:
        raise ValueError('v3r5 --antithetic is implemented for train_fd '
                         'only; add --fd-residual')
    max_iter, lam_bar = hp['max_iter'], hp['lam_bar']
    problem = build_problem(args)
    N = args.num_dt
    Dt = (torch.tensor(args.te) - problem.t0) / N
    if args.selftest_antithetic:
        ok = antithetic_selftest(problem, Dt, N)
        print(f'antithetic self-test overall: {"PASS" if ok else "FAIL"}')
        sys.exit(0 if ok else 1)
    batsize_gpu = [max(int(s / world_size), 1) for s in hp['batsize']]
    if len(hp['batsize']) > 1:
        milestones = [int(max_iter / len(hp['batsize']) * (i + 1))
                      for i in range(len(hp['batsize']) - 1)]
    else:
        milestones = []

    # RNG call order mirrors the authors' testsocmart.py:
    # grid (deterministic) -> reference MC -> network init -> paths -> training
    x0 = problem.x0_points(int(args.num_paths / world_size))
    x_test = problem.test_points(10**3)
    v_true = problem.v_exact(problem.t0.unsqueeze(-1), x_test)

    u_alpha = control_net(args.dim, problem.dim_u, hp['width'],
                          hp['num_hidden'])
    v_theta = value_net(args.dim, hp['width'], hp['num_hidden'])
    rho_eta = test_net(args.dim, hp['r'])

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        u_alpha = DDP(u_alpha.to(rank), device_ids=[rank])
        v_theta = DDP(v_theta.to(rank), device_ids=[rank])
        rho_eta = DDP(rho_eta.to(rank), device_ids=[rank])
    else:
        u_alpha, v_theta, rho_eta = (u_alpha.to(rank), v_theta.to(rank),
                                     rho_eta.to(rank))

    soc_mode = args.example in SOC_EXAMPLES
    optim_u = torch.optim.RMSprop(u_alpha.parameters(), lr=hp['lr_uv'])
    optim_v = torch.optim.RMSprop(v_theta.parameters(), lr=hp['lr_uv'])
    optim_rho = torch.optim.RMSprop(rho_eta.parameters(), lr=hp['lr_rho'])

    if hp['sch_mode'] == 'quantized':
        # staircase 0.01^{i/I} in <=100-iter plateaus (authors' R1 configs)
        sch_step = min(100, max(int(max_iter / 10), 1))
        sch_gam = 0.01**(sch_step / max_iter)
    else:  # v3paper/cube: validated baseline schedule
        sch_step = max(int(max_iter / 10), 1)
        sch_gam = 0.01**(1 / 9)
    sch_u = torch.optim.lr_scheduler.StepLR(optim_u, sch_step,
                                            sch_gam) if soc_mode else None
    sch_v = torch.optim.lr_scheduler.StepLR(optim_v, sch_step, sch_gam)
    sch_rho = torch.optim.lr_scheduler.StepLR(optim_rho, sch_step, sch_gam)

    def err_func():
        return relative_l1(v_theta(problem.t0.unsqueeze(-1), x_test), v_true)

    extended = args.extended_log or args.preset == 'sisc'
    cost_track = args.cost_track
    if cost_track is None:
        cost_track = args.example == 'shifttarget'
    if cost_track and not hasattr(problem, 'comput_cost'):
        raise ValueError(f'--cost-track needs a comput_cost method; '
                         f'{args.example} does not provide one')

    aux_func = None
    if extended or cost_track:
        vtrue_linf = v_true.abs().max()

        def aux_func():
            out = {}
            with torch.no_grad():
                v_pred = v_theta(problem.t0.unsqueeze(-1), x_test)
                out['rel_linf'] = (v_pred - v_true).abs().max() / vtrue_linf
                out['mean_vtrue_t0'] = v_true.mean()
                if cost_track and soc_mode:
                    x0c = torch.zeros([args.cost_paths, args.dim])
                    cost = problem.comput_cost(u_alpha, x0c,
                                               num_dt=args.cost_num_dt)
                    if world_size > 1:
                        dist.all_reduce(cost, op=dist.ReduceOp.AVG)
                    out['cost'] = cost
                mr = getattr(solver, 'mart_raw_last', None)
                if mr is not None:
                    out['mart_raw'] = mr
                ul = getattr(solver, 'u_lag_last', None)
                if ul is not None:
                    out['u_lag'] = ul
            return out

    method = args.method
    if method is None:
        method = 'socmartnet' if soc_mode else 'prabmartnet'
    if method == 'socmartnet' and not soc_mode:
        raise ValueError('socmartnet (Alg. 3.1) applies to the SOC examples '
                         '(hjb/hjblq/shifttarget); linear/semilinear/linsin '
                         'use prabmartnet (Alg. 3.2)')
    if method == 'fbsnn' and soc_mode:
        raise ValueError('fbsnn is a parabolic-PDE benchmark; it does not '
                         'apply to the SOC examples')
    if method == 'prabmartnet' and soc_mode and args.example != 'hjb':
        raise ValueError('prabmartnet (Alg. 3.2, PDE mode) among the SOC '
                         'examples applies only to hjb (the HJB-1 <-> '
                         'semilinear reduction); hjblq/shifttarget provide '
                         'no PDE driver f')
    method_label = {'socmartnet': 'SOCMartNet', 'prabmartnet': 'PrabMartNet',
                    'fbsnn': 'FBSNN'}[method]
    sav_name = (f"{args.out}/{method_label}_{problem.name}_d{args.dim}"
                f"_te{args.te}" + (f"_{args.tag}" if args.tag else ""))

    if method == 'fbsnn':
        # epoch-matched iteration count and batch 300 (paper Sec. 4 intro)
        batsize = hp['batsize']
        ext_ms = [0] + milestones + [max_iter]
        num_epoch = sum(b * (ext_ms[i + 1] - ext_ms[i])
                        for i, b in enumerate(batsize)) / args.num_paths
        bat_fb = max(int(300 / world_size), 1)
        maxit_fb = int(num_epoch * args.num_paths / 300)
        lr_fb = 3e-3 / args.dim**0.5
        optim_fb = torch.optim.RMSprop(v_theta.parameters(), lr=lr_fb)
        sch_fb = torch.optim.lr_scheduler.StepLR(
            optim_fb, max(int(maxit_fb / 10), 1), 0.001**(1 / 9))
        solver = FBSNN(Dt, problem.mu, problem.sigma, problem.f,
                       problem.v_term, problem.dim_w, t0=problem.t0)
        solver.solve(v_theta, optim_fb, sch_fb, maxit_fb, x0, [bat_fb],
                     rank=rank, N=N, err_func=err_func, log_gap=1,
                     f_depends_on_vxx=problem.f_depends_on_vxx)

    elif method == 'prabmartnet':
        solver = SOCMartNet(Dt, problem.mu, problem.sigma, problem.H,
                            problem.v_term, problem.dim_w, t0=problem.t0,
                            f_fun=problem.f,
                            H_depends_on_vx=problem.H_depends_on_vx)
        # v3r5.1: forward renew_frac -- the authors' R1 PDE INIs set
        # rate_newpath = 0.2 (Count/LinSinCR), matching the SOC caliber.
        solver.train((u_alpha, v_theta, rho_eta),
                     (optim_u, optim_v, optim_rho), (sch_u, sch_v, sch_rho),
                     max_iter, x0, batsize_gpu, rank=rank,
                     lam0=1., delta4=1., lam_bar=1.,  # lambda pinned at 1
                     batsize_milestone=milestones, N=N, err_func=err_func,
                     log_gap=1, J=2, K=1, aux_func=aux_func,
                     renew_frac=args.renew_frac,
                     debias_mart=args.debias_mart)
    else:
        solver = SOCMartNet(Dt, problem.mu, problem.sigma, problem.H,
                            problem.v_term, problem.dim_w, t0=problem.t0,
                            H_depends_on_vx=problem.H_depends_on_vx,
                            f_cost_fun=getattr(problem, 'f_cost', None))
        if args.fd_residual:
            if not soc_mode or getattr(problem, 'f_cost', None) is None:
                raise ValueError('--fd-residual needs an SOC example with a '
                                 'f_cost method (e.g. hjblq/shifttarget)')
            # R1 caliber pins lambda at 1; --lam-bar > 1 upweights the
            
            lam_bar_fd = lam_bar if args.lam_bar is not None else 1.
            solver.train_fd((u_alpha, v_theta, rho_eta),
                            (optim_u, optim_v, optim_rho),
                            (sch_u, sch_v, sch_rho),
                            max_iter, x0, batsize_gpu, rank=rank,
                            batsize_milestone=milestones, N=N,
                            err_func=err_func, log_gap=1, J=2, K=1,
                            aux_func=aux_func, renew_frac=args.renew_frac,
                            lam0=min(10., lam_bar_fd), delta4=10.,
                            lam_bar=lam_bar_fd, debias_mart=args.debias_mart,
                            ctr_pool=args.controlled_pool,
                            antithetic=args.antithetic)
        else:
            solver.train((u_alpha, v_theta, rho_eta),
                         (optim_u, optim_v, optim_rho),
                         (sch_u, sch_v, sch_rho),
                         max_iter, x0, batsize_gpu, rank=rank,
                         lam0=10., delta4=10., lam_bar=lam_bar,
                         batsize_milestone=milestones, N=N, err_func=err_func,
                         log_gap=1, J=2, K=1, aux_func=aux_func,
                         renew_frac=args.renew_frac,
                         debias_mart=args.debias_mart)

    if rank == 0:
        Path(args.out).mkdir(exist_ok=True)
        cols = [solver.it_hist, solver.epoch_hist, solver.rt_hist,
                solver.ham_hist, solver.lossmart_hist, solver.error_hist]
        header = ['iter step', 'epoch', 'rt', 'hami', 'mart loss', 'error']
        for key, val in getattr(solver, 'aux_hist', {}).items():
            header.append(key)
            cols.append(val)
        log_tensor = torch.stack(cols, dim=1)
        with open(f'{sav_name}.csv', 'w', encoding='UTF8',
                  newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(log_tensor.cpu().numpy())
        v_save = v_theta.module if world_size > 1 else v_theta
        torch.save(v_save.state_dict(), f'{sav_name}_vnn.pkl')
        print(f'saved: {sav_name}.csv')

        save_curves = args.save_curves
        if save_curves is None:
            save_curves = (args.preset == 'sisc'
                           and args.x0_mode == 'region')
        if save_curves and args.x0_mode == 'region':
            region = args.region
            if region is None:
                region = {'hjblq': 's2s3', 'shifttarget': 's2'}.get(
                    args.example, 's1s2')
            curves = {'s1s2': ['e1', 'diag'],
                      's2s3': ['diag', 'manifold'],
                      's2': ['diag']}[region]
            for cv in curves:
                s, v_t, v_p = evaluate.curve_values(
                    problem, v_save, cv, radius=args.x0_scale,
                    num_points=args.curve_points)
                with open(f'{sav_name}_curve_{cv}.csv', 'w',
                          encoding='UTF8', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(['s', 'v_true', 'v_pred'])
                    writer.writerows(torch.stack(
                        [s, v_t, v_p], dim=1).cpu().numpy())
                print(f'saved: {sav_name}_curve_{cv}.csv')

        if args.eval_ptx:
            for r_frac in (0.125, 0.25):
                s, v_t, v_p = evaluate.ptx_values(
                    problem, v_save, r_frac, radius=args.x0_scale,
                    num_points=args.curve_points)
                with open(f'{sav_name}_ptx{r_frac}.csv', 'w',
                          encoding='UTF8', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(['s', 'v_true', 'v_pred'])
                    writer.writerows(torch.stack(
                        [s, v_t, v_p], dim=1).cpu().numpy())
                print(f'saved: {sav_name}_ptx{r_frac}.csv')

        if args.eval_path_re:
            t_grid, re_pool, re_mean, re_std = evaluate.path_re(
                problem, v_save, num_dt=N)
            with open(f'{sav_name}_pathre.csv', 'w', encoding='UTF8',
                      newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(['t', 're_pool', 're_path_mean',
                                 're_path_std'])
                writer.writerows(torch.stack(
                    [t_grid, re_pool, re_mean, re_std], dim=1).cpu().numpy())
            print(f'saved: {sav_name}_pathre.csv')

    if world_size > 1:
        dist.destroy_process_group()


def main():
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    if args.gpus > 0:
        n_gpus = min(args.gpus, n_gpus)
    n_gpus = max(n_gpus, 1)
    print(f'Number of used GPUs: {n_gpus}')
    if n_gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(n_gpus, args), nprocs=n_gpus, join=True)
    else:
        worker(0, 1, args)


if __name__ == '__main__':
    main()
