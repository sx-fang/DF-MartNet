"""Control arm for the refactor validation: run the UNMODIFIED v3 blueprint
code (baselines/SOCMartNet-v3-arxiv) on one (problem, dim, te, method) case
with the authors' cube-dir configuration, saving the same CSV format as the
refactored run.py.

Only the single-case driver logic is reimplemented here (the blueprint
testsocmart.py loops over long Cartesian lists); all numerical code paths are
imported from the blueprint unchanged.

Example (HJB smooth, d=100, T=1, SOCMartNet, authors' cube settings):
  python tools/run_original_v3.py \
      --blueprint-dir ../SOCMartNet-v3-arxiv/SOCMN_cube_dleq500 \
      --problem hjb1 --dim 100 --te 1.0 --method socmartnet --out outputs_orig
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--blueprint-dir', type=str, required=True)
    p.add_argument('--problem', choices=['hjb1', 'hjb3', 'count'],
                   required=True)
    p.add_argument('--method', choices=['socmartnet', 'prabmartnet', 'fbsnn'],
                   default='socmartnet')
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--te', type=float, default=1.0)
    p.add_argument('--num-paths', type=int, default=10**5)
    p.add_argument('--num-dt', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='outputs_orig')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    bp = Path(args.blueprint_dir).resolve()
    sys.path.insert(0, str(bp))

    import socp  # blueprint problem definitions
    from socmart import DNNtx, SOCMartNet, setup_mp, cleanup
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank = 0
    world_size = 1
    setup_seed(args.seed + rank)
    setup_mp(rank, world_size)  # blueprint SOCMartNet needs dist initialized
    torch.set_default_dtype(torch.float64)
    torch.set_default_device('cuda:0')

    # ---- cube-dir configuration (SOCMN_cube_dleq500/testsocmart.py) ----
    dim_x = args.dim
    te = torch.tensor(args.te)
    t0 = torch.tensor(0.)
    num_dt = args.num_dt
    numpath = args.num_paths
    dt = (te - t0) / num_dt
    unit_ball = False
    x0_scale = torch.tensor(1.)

    maxit = 2000 if dim_x > 100 else 1000
    batsize_socm = [200, 400, 800, 1600]
    batsize_gpu = [int(s / world_size) for s in batsize_socm]
    bsize_milstone = [int(maxit / len(batsize_socm) * (i + 1))
                      for i in range(len(batsize_socm) - 1)]

    # RNG order mirrors the blueprint main(): grid -> reference MC -> nets
    # NB: the blueprint's default sgm_scale=torch.tensor(1.) is built at import
    # time (CPU); pass an explicit on-device tensor to avoid the mixed-device
    # torch.normal failure under torch.set_default_device('cuda').
    sgm_scale_dev = torch.tensor(1.)
    if args.problem == 'hjb1':
        problem = socp.NonDegHJB(dim_x, t0=t0, te=te, x0_scale=x0_scale,
                                 sgm_scale=sgm_scale_dev, unit_ball=unit_ball)
    elif args.problem == 'hjb3':
        problem = socp.NonDegHJBv3(dim_x, t0=t0, te=te, x0_scale=x0_scale,
                                   sgm_scale=sgm_scale_dev,
                                   unit_ball=unit_ball, delc=0.1)
    else:
        problem = socp.Counter(dim_x, t0=t0, te=te, x0_scale=x0_scale,
                               unit_ball=unit_ball)

    # RNG order mirrors the blueprint main(): grid -> reference MC -> nets
    x0 = problem.x0_points(int(numpath / world_size))
    x_test = problem.test_points(10**3)
    vtrue_val = problem.v(t0.unsqueeze(-1), x_test)

    from torch import nn
    width_unn = [dim_x] + [2 * (dim_x + 10)] * 3 + [problem.dim_u]
    width_vnn = [dim_x] + [2 * (dim_x + 10)] * 3 + [1]
    width_testnn = [dim_x] + [2 * dim_x + 500]

    unn = DNNtx(width_unn, act_func=nn.ReLU)
    vnn = DNNtx(width_vnn, act_func=nn.ReLU)
    test_nn = DNNtx(width_testnn, act_func=nn.LeakyReLU,
                    shell_func=lambda _t, y: torch.sin(y))

    unn = DDP(unn.to(rank), device_ids=[rank])
    vnn = DDP(vnn.to(rank), device_ids=[rank])
    test_nn = DDP(test_nn.to(rank), device_ids=[rank])

    init_lr = 1e-3
    testnn_lr = 1e-2
    optim_unn = torch.optim.RMSprop(unn.parameters(), lr=init_lr)
    optim_vnn = torch.optim.RMSprop(vnn.parameters(), lr=init_lr)
    optim_test = torch.optim.RMSprop(test_nn.parameters(), lr=testnn_lr)

    schvary_step = max(int(maxit / 10), 1)
    sch_gam = 0.01**(1 / 9)
    sch_unn = torch.optim.lr_scheduler.StepLR(
        optim_unn, schvary_step, sch_gam)
    sch_vnn = torch.optim.lr_scheduler.StepLR(
        optim_vnn, schvary_step, sch_gam)
    sch_rho = torch.optim.lr_scheduler.StepLR(
        optim_test, schvary_step, sch_gam)

    def error_func():
        error = vnn(t0.unsqueeze(-1), x_test) - vtrue_val
        return torch.abs(error).mean() / torch.abs(vtrue_val).mean()

    method_label = {'socmartnet': 'SOCMartNet', 'prabmartnet': 'PrabMartNet',
                    'fbsnn': 'FBSNN'}[args.method]
    sav_name = (f"{args.out}/{method_label}_{problem.name}_d{dim_x}"
                f"_te{args.te}" + (f"_{args.tag}" if args.tag else ""))

    if args.method == 'fbsnn':
        from fbsnn import FBSNN
        ext_ms = [0] + bsize_milstone + [maxit]
        num_epoch = sum(b * (ext_ms[i + 1] - ext_ms[i])
                        for i, b in enumerate(batsize_socm)) / numpath
        batsize_fb = 300
        maxit_fb = int(num_epoch * numpath / batsize_fb)
        optim_fb = torch.optim.RMSprop(vnn.parameters(), lr=3e-3 / dim_x**0.5)
        sch_fb = torch.optim.lr_scheduler.StepLR(
            optim_fb, max(int(maxit_fb / 10), 1), 0.001**(1 / 9))
        solver = FBSNN(dt, problem.mu, problem.sgm_diag, problem.f,
                       problem.v_term, dim_x, t0=t0)
        solver.solve(vnn, optim_fb, sch_fb, maxit_fb, x0, 0.,
                     [int(batsize_fb / world_size)], rank=rank,
                     batsize_milstone=None, num_dt=num_dt,
                     err_func=error_func, log_gap=1,
                     f_depends_on_uxx=problem.f_depends_on_vxx)
    else:
        soc_mode = args.method == 'socmartnet'
        lamb0, del_lamb, max_lamb = (10., 10, 1e4) if soc_mode \
            else (1., 1., 1.)
        solver = SOCMartNet(
            dt, problem.mu, problem.sgm_diag, problem.h, problem.v_term,
            dim_x, t0=t0,
            f_func=None if soc_mode else problem.f,
            h_depends_on_ux=problem.h_depends_on_vx,
            h_depends_on_uxx=problem.h_depends_on_vxx)
        solver.train((unn, vnn, test_nn),
                     (optim_unn, optim_vnn, optim_test),
                     (sch_unn if soc_mode else None, sch_vnn, sch_rho),
                     maxit, x0, batsize_gpu, rank=rank,
                     lamb0=lamb0, del_lamb=del_lamb, max_lamb=max_lamb,
                     batsize_milstone=bsize_milstone, num_dt=num_dt,
                     err_func=error_func, log_gap=1, step_uv=2, step_rho=1)

    Path(args.out).mkdir(exist_ok=True)
    log_tensor = torch.stack(
        [solver.it_hist, solver.epoch_hist, solver.rt_hist, solver.ham_hist,
         solver.lossmart_hist, solver.error_hist], dim=1)
    with open(f'{sav_name}.csv', 'w', encoding='UTF8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['iter step', 'epoch', 'rt', 'hami', 'mart loss',
                         'error'])
        writer.writerows(log_tensor.cpu().numpy())
    torch.save(vnn.module.state_dict(), f'{sav_name}_vnn.pkl')
    print(f'saved: {sav_name}.csv')
    cleanup()


if __name__ == '__main__':
    main()
