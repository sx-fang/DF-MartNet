"""INI parsing, network construction, and the solve() driver.

Only the method of the paper is supported: MartNetLoss with pilot--system
path sampling. INI keys are documented in default_config.ini.
"""
import ast
import json
from sample_budgets import SAMPLE_BUDGET_PROTOCOL, budget_entry
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import examples as examples_module
from diagnostics import res_on_curve
from loss_martnet import MartNetLoss
from networks import AnchoredUNet, FCNet, zero_init_output_layer
# NOTE (owner 2026-10-01: "不要动 u 的网络结构"): this file deliberately does NOT
# wire the rank-k structured control head (rank_control.RankControlUNet) that
# was prototyped for route-A option 1 on the same day.  That wiring existed only
# in the live server tree and was reverted there before any training ran (job
# 544794 was cancelled while PENDING).  rank_control.py stays in the repo as an
# unused experiment; do not re-wire it without a new owner decision.
# This local file also lags the live tree by the other session's radial_ctrl
# wiring (see radial_control.py), so never upload it wholesale -- patch the live
# tree in place instead.
from train import train
from utils import get_global_rank, get_local_rank, split_number

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
    global_batch_size = config.getint('Training', 'num_xpts_batch')
    global_num_paths = config.getint('Training', 'num_pilot_paths')
    wd = dist.get_world_size() if use_dist else 1
    rank = dist.get_rank() if use_dist else 0
    batch_budget = budget_entry(global_batch_size, wd, rank, minimum=1,
                                name='Training.num_xpts_batch')
    pilot_budget = budget_entry(global_num_paths, wd, rank, minimum=2,
                                name='Training.num_pilot_paths')
    batch_counts = batch_budget['per_rank']
    path_counts = pilot_budget['per_rank']
    batch_size = batch_counts[rank]
    num_pilot_paths = path_counts[rank]
    try:
        cls = EXAMPLE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Example '{name}' not found. "
                         f"Available: {sorted(EXAMPLE_REGISTRY)}")
    problem = cls(dim_x,
                  dt=eval(config.get('Training', 'dt')),
                  batch_size=batch_size,
                  num_pilot_paths=num_pilot_paths)
    problem._batch_point_counts = tuple(batch_counts)
    problem._batch_rank = rank
    problem.M_rc = config.getint('Reproduction', 'global_cost_paths',
                                     fallback=config.getint('Example', 'M_rc', fallback=4096))
    # Distributed-X0 RC caliber (owner 2026-09-14, hjb-term-rcdist): X_0 ~
    # x0_curves under a CRN seed, rc normalized by |J*| with J* = mean v on
    # the equispaced _re_grid; default False = single-point caliber, zero
    # behavior change on every existing INI.
    problem.rc_x0_dist = config.getboolean('Example', 'rc_x0_dist',
                                           fallback=False)
    # Point-X0 RC caliber (owner 2026-09-18, b1origin retrain): inverse of
    # rc_x0_dist -- force the deterministic origin tile in the rc chain for
    # classes whose _initial_points is distributed by default (SOCComplex);
    # default False = zero behavior change on every existing INI.
    problem.rc_x0_origin = config.getboolean('Example', 'rc_x0_origin',
                                             fallback=False)
    # x0_point_origin (owner 2026-09-19, k1pair variant): X_0 fixed at the
    # single deterministic origin in x0_points (training-path starts, pilot
    # library, t=0 test grid); default False = zero behavior change.
    problem.x0_point_origin = config.getboolean('Example', 'x0_point_origin',
                                                fallback=False)
    # MC sample size of the reference solution (class default when unset)
    problem.M_mc = config.getint('Example', 'M_mc', fallback=problem.M_mc)
    problem.sample_budgets = {
        'protocol': SAMPLE_BUDGET_PROTOCOL, 'world_size': wd, 'rank': rank,
        'num_xpts_batch': batch_budget, 'num_pilot_paths': pilot_budget,
        'M_rc': budget_entry(problem.M_rc, wd, rank, minimum=1, name='global policy-cost paths'),
        'M_mc': budget_entry(problem.M_mc, wd, rank, name='global reference paths'),
    }
    print('SAMPLE_BUDGETS ' + json.dumps(problem.sample_budgets, sort_keys=True), flush=True)
    # Evaluation point count (spread set for sparse examples; class default
    # when unset).  Should be a multiple of the time-level count T/dt so
    # that no level is dropped by the trim in spread_points (owner decision
    # 2026-08-20: 300 = ceil-trim dropped the last two levels; use 320).
    problem.num_test_points = config.getint('Example', 'num_test_points',
                                            fallback=problem.num_test_points)
    # first-500-iteration probe handoff): read-only per-iteration point
    # values / relative errors at (t0, z0*1_d) and a CRN origin-cost row
    # every x0_diag_cost_gap iterations.  Section [Diag] is optional; when
    # absent or when the switch is off, the hook is inert -- no key is
    # added to the log dict and no behavior changes anywhere.
    problem.enable_x0_diag = config.getboolean('Diag', 'x0_point_diag',
                                               fallback=False)
    problem.x0_diag_cost_gap = config.getint('Diag', 'x0_point_cost_gap',
                                             fallback=25)
    problem.x0_diag_cost_seed = config.getint('Diag', 'x0_point_cost_seed',
                                              fallback=20260924)
    problem._x0_diag_m_rc = config.getint('Diag', 'x0_point_cost_m',
                                          fallback=problem.M_rc)
    problem.noise_sampler = config.get('Training', 'noise_sampler',
                                       fallback='mc')
    # u*-ablation (owner 2026-08-31): replace the learned control by the
    # closed-form u_star inside the training residual; default False = the
    # original method, zero behavior change.
    problem.u_fixed_ref = config.getboolean('Training', 'u_fixed_ref',
                                            fallback=False)
    # Terminal supervision (owner 2026-09-08): soft direct-L2 pull of
    # v_theta / u_alpha at the last interior time layer toward the exact
    # terminal information; defaults 0.0 = off = zero behavior change.
    problem.w_term_v = config.getfloat('Training', 'w_term_v', fallback=0.0)
    problem.w_term_u = config.getfloat('Training', 'w_term_u', fallback=0.0)
    # SOC-MartNet-style adaptive martingale weight (owner directive
    # 2026-09-13, E1 alignment): lam <- min(lam_bar, lam + delta4*mart)
    # with loss = lam*mart + ctr; defaults 1.0/0.0/1.0 = lam pinned at 1
    # = byte-identical behavior on every existing configuration.
    problem.lam0 = config.getfloat('Training', 'lam0', fallback=1.0)
    problem.delta4 = config.getfloat('Training', 'delta4', fallback=0.0)
    problem.lam_bar = config.getfloat('Training', 'lam_bar', fallback=1.0)
    # last-layer martingale upweighting (owner 2026-09-16): w_mart_last in
    # the paired |L|^2 estimate for rows at t = T - dt; default 1.0 = off
    # = zero behavior change (see loss_martnet.MartNetLoss).
    problem.w_mart_last = config.getfloat('Training', 'w_mart_last',
                                          fallback=1.0)
    # control-channel twin (owner 2026-09-17): w_ctr_last upweights the
    # t = T - dt rows inside the control residual mean (ctr_loss);
    # default 1.0 = off = zero behavior change (see loss_martnet).
    problem.w_ctr_last = config.getfloat('Training', 'w_ctr_last',
                                         fallback=1.0)
    problem.anneal_last = config.getboolean('Training', 'anneal_last',
                                             fallback=False)
    problem.anneal_frac = config.getfloat('Training', 'anneal_frac',
                                          fallback=1.0)
    problem.max_iter = config.getint('Training', 'max_iter', fallback=0)
    # terminal-cost sin amplitude (SOCComplex contrast probe, owner
    # 2026-08-23; class default 2.0 when unset)
    # SOCComplex-only knobs (g_sin_amp/g_shell_A/g_scale_B): guarded by
    # hasattr so other example classes keep working (regression fix 2026-08-24
    # -- eager fallback=problem.g_sin_amp crashed every non-SOCComplex INI)
    if hasattr(problem, 'g_sin_amp'):
        problem.g_sin_amp = config.getfloat('Example', 'g_sin_amp',
                                            fallback=problem.g_sin_amp)
    # span levers A/B (SOCComplex span>1 probe, owner 2026-08-24; class
    # defaults 1.0 when unset)
    if hasattr(problem, 'g_shell_A'):
        problem.g_shell_A = config.getfloat('Example', 'g_shell_A',
                                            fallback=problem.g_shell_A)
    if hasattr(problem, 'g_scale_B'):
        problem.g_scale_B = config.getfloat('Example', 'g_scale_B',
                                            fallback=problem.g_scale_B)
    # u* strong-signal cusp (SOCComplex, owner 2026-09-04, approved design:
    # g_cusp_idx 0 and d/2, g_cusp_amp 0.45): g += g_cusp_amp *
    # sum_{i in g_cusp_idx} sin(x_i) on individual coordinates (NOT /d);
    # hasattr-guarded like the knobs above; unset = class defaults
    # (0.0/()) = zero behavior change.
    if hasattr(problem, 'g_cusp_amp'):
        problem.g_cusp_amp = config.getfloat('Example', 'g_cusp_amp',
                                             fallback=problem.g_cusp_amp)
        cusp_s = config.get('Example', 'g_cusp_idx', fallback='')
        if cusp_s:
            problem.g_cusp_idx = tuple(
                int(v) for v in cusp_s.replace(' ', '').split(',') if v)
    # single-coordinate control anchor (SOCComplex, owner 2026-09-20,
    # uanchor_v1 campaign): g += g_anchor_alpha * sin(x_1) on coordinate 1
    # alone (NOT divided by d) so u* = -c0 s^T grad v carries an O(alpha)
    # anchor component; hasattr-guarded like the knobs above; unset =
    # class default 0.0 = zero behavior change.
    if hasattr(problem, 'g_anchor_alpha'):
        problem.g_anchor_alpha = config.getfloat(
            'Example', 'g_anchor_alpha',
            fallback=problem.g_anchor_alpha)
    # sqrt(d)-scaled terminal sin (SOCComplex, owner 2026-09-23, Route-2
    # u*-richness probe): g += g_sin_amp_sqrt/sqrt(d) * sum_i sin(x_i),
    # the calib_route2.py arm caliber (536526); hasattr-guarded like the
    # knobs above; unset = class default 0.0 = zero behavior change.
    if hasattr(problem, 'g_sin_amp_sqrt'):
        problem.g_sin_amp_sqrt = config.getfloat(
            'Example', 'g_sin_amp_sqrt',
            fallback=problem.g_sin_amp_sqrt)
    # centered convex-separable Route-3 terminal (SOCComplex, owner
    # 2026-09-24): g += g_abs_amp/sqrt(d) * (sum_i |x_i - a| - d*
    # g_abs_cen); mu_a = g_abs_cen printed by calib_route3_abs.py;
    # hasattr-guarded like the knobs above; unset = 0.0/1.0/0.0 = zero
    # behavior change.
    if hasattr(problem, 'g_abs_amp'):
        problem.g_abs_amp = config.getfloat('Example', 'g_abs_amp',
                                            fallback=problem.g_abs_amp)
        problem.g_abs_shift = config.getfloat(
            'Example', 'g_abs_shift', fallback=problem.g_abs_shift)
        problem.g_abs_cen = config.getfloat('Example', 'g_abs_cen',
                                            fallback=problem.g_abs_cen)
    # SOCComplexK1 channel-terminal amplitude (owner 2026-09-05, /goal):
    # g += g_chan_amp * sin(q.z) on the band-limited channel direction;
    # hasattr-guarded like the knobs above; unset = 0.0 = zero delta.
    if hasattr(problem, 'g_chan_amp'):
        problem.g_chan_amp = config.getfloat('Example', 'g_chan_amp',
                                             fallback=problem.g_chan_amp)
    # SOCComplexK1 channel gain gamma (owner 2026-09-05, gain-in-cost form:
    # drift push is the unit-gain q, f = kappa^2/(2 gamma^2)); unset = 1.0.
    if hasattr(problem, 'chan_gain'):
        problem.chan_gain = config.getfloat('Example', 'chan_gain',
                                            fallback=problem.chan_gain)
    # curve parameter range s_range (L2 selected cell 2026-09-07:
    # s_range=2.5 is the spread dial of the single-coordinate reward
    # variant -- z_i(0) = s*r spreads BOTH S0 and S1; UNCONDITIONAL on
    # purpose: it must reach SOCComplexK1-based INIs, and the fallback
    # keeps every existing INI unchanged when the key is absent).
    problem.s_range = config.getfloat('Example', 's_range',
                                      fallback=problem.s_range)
    # shortened drift-feedback kernel Phi_i = sum phi_decay^k z_{i+k} that
    # cut the 513026 W3 drift-noise excess; hasattr-guarded so the BASE
    # the K1 class default back to 0.9 after 513033 falsified 0.5 (coherent
    # feedback amplification x15); unset = K1 class default 0.9.
    if hasattr(problem, 'phi_decay'):
        problem.phi_decay = config.getfloat('Example', 'phi_decay',
                                            fallback=problem.phi_decay)
    # SOCComplexK1 drift-feedback amplitude (owner /goal 2026-09-05, round
    # 3): b_i = drift_amp * sin(t + i0 + Phi_i) -- under the phi_decay=0.9
    # de-coherence the feedback variance scales ~ drift_amp^2 (513026
    # decomposition: sigma side 0.078 + feedback ~0.16 at drift_amp 2.0),
    # band-independent; see the SOCComplexK1 docstring.  hasattr-guarded
    # like the knobs above; unset = K1 class default 1.2 while the BASE
    # SOCComplex keeps its own drift_amp = 2.0 untouched (no existing INI
    # sets the key, so B1/B2 are byte-identical).
    if hasattr(problem, 'drift_amp'):
        problem.drift_amp = config.getfloat('Example', 'drift_amp',
                                            fallback=problem.drift_amp)
    # time-oscillation frequency (owner 2026-09-20, c2scale probe): t phase
    # of BOTH b and the diffusion modulation m becomes time_freq * t;
    # hasattr-guarded like the knobs above; unset = class default 1.0 =
    # the frozen B' dynamics bit-for-bit (no existing INI sets the key).
    if hasattr(problem, 'time_freq'):
        problem.time_freq = config.getfloat('Example', 'time_freq',
                                            fallback=problem.time_freq)
    # scheme 3-prime): sigma = sqrt(2) c1 diag(m(x)) for THIS subclass
    # (Toeplitz removed, see SOCComplexK1._s_matvec); class default 0.20,
    # unset = class default while the BASE SOCComplex keeps its own
    # c1 = 0.5 Toeplitz sigma untouched (no existing INI sets the key,
    # so B1/B2 are byte-identical).
    if hasattr(problem, 'c1'):
        problem.c1 = config.getfloat('Example', 'c1',
                                     fallback=problem.c1)
    # L1): q gains a chan_dc * 1_d addend before normalization so the
    # initial projection q.z0 spreads over X0 and the optimal push becomes
    # best CONSTANT near-optimal on the S0/S1 surface).  Class default
    # 0.0 = the 3-prime q bitwise; hasattr-guarded, no existing INI sets
    # the key so the base family is byte-identical.
    if hasattr(problem, 'chan_dc'):
        problem.chan_dc = config.getfloat('Example', 'chan_dc',
                                          fallback=problem.chan_dc)
    # SOCComplexK1 L2 knobs (owner-proposed 2026-09-06): chan_coord >= 0
    # switches the channel terminal reward to g_chan_amp*sin(z_i) (single
    # coordinate, whose initial value spreads naturally on BOTH x0 curves --
    # the L1 postmortem showed the DC spread leaves S1 inert); chan_spike
    # puts a w*sqrt(d/2) one-hot on that coordinate inside q so the control
    # can steer the reward coordinate (alpha = q_i).  Defaults -1/0.0 =
    # bitwise off; hasattr-guarded, no existing INI sets the keys.
    if hasattr(problem, 'chan_coord'):
        problem.chan_coord = config.getint('Example', 'chan_coord',
                                           fallback=problem.chan_coord)
    if hasattr(problem, 'chan_spike'):
        problem.chan_spike = config.getfloat('Example', 'chan_spike',
                                             fallback=problem.chan_spike)
    # SOCChainGate knobs (owner-designed example 2026-09-06/07, "L3":
    # gated long-range chain, u only on x_1, terminal window on x_d):
    # kernel exponent alpha / saturating width / coupling level cG,
    # gate (g0, g1, omega, kappa, coord), terminal window (A, x*, w).
    # c1 and chan_gain are covered by the generic blocks above (the
    # class has both attributes).  All hasattr-guarded, zero effect on
    # every other example.
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
    # sparse-ablation knobs (owner 2026-08-27, phase-1 d=20 campaign):
    # ring transport rate omega (variant L; class default 20.0) and terminal
    # sensitivity weights c_weights as comma-separated floats (class default
    # (1.0, 1.5, 2.0, 2.5, 3.0)).  Guarded by hasattr, same pattern as the
    # SOCComplex knobs above; unset = class defaults, zero behavior change.
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
    # diffusion strength delta (owner 2026-08-27 delta sweep on comboD;
    # class default 1.0 -- sparse and constant-coefficient families both
    # read self.delta throughout sigma / running cost / references).
    if hasattr(problem, 'delta'):
        problem.delta = config.getfloat('Example', 'sparse_delta',
                                        fallback=problem.delta)
    # LQ terminal target offset (owner 2026-08-28, sparse example redesign
    # phase 0+1; HJBSparseLQ class default 1.0, hasattr-guarded so the
    # ln-terminal classes are untouched).
    if hasattr(problem, 'sparse_a'):
        problem.sparse_a = config.getfloat('Example', 'sparse_a',
                                           fallback=problem.sparse_a)
    # full-rank radial terminal coefficient (owner 2026-08-30, hybrid-terminal
    # redesign): g += (sparse_c0/2d)|z|^2 on HJBSparseLQ; class default 0.0 =
    # original sparse LQ, hasattr-guarded so other families are untouched.
    if hasattr(problem, 'sparse_c0'):
        problem.sparse_c0 = config.getfloat('Example', 'sparse_c0',
                                            fallback=problem.sparse_c0)
    # number of sparse control channels / sensitive directions (owner
    # 2026-08-29, variant A k-channel control: HJBSparseLQUA.dim_u = k_dirs).
    # hasattr-guarded; absent = class default 5, zero behavior change.  The
    # caller must keep c_weights length >= k_dirs (HJBSparseBase.g asserts).
    if hasattr(problem, 'k_dirs'):
        problem.k_dirs = config.getint('Example', 'sparse_k',
                                       fallback=problem.k_dirs)
    # E-SEP sparse-noise dimension (owner 2026-09-01): the diffusion spans
    # the first sparse_m columns of the DCT basis (HJBSparseSEP).  hasattr-
    # guarded; class default None = the class raises until the INI sets it.
    if hasattr(problem, 'sparse_m'):
        problem.sparse_m = config.getint('Example', 'sparse_m',
                                         fallback=problem.sparse_m)
    # reference dt_ref override (owner 2026-08-24, sparse d=1e2 ladder; small
    # d requires n_steps = T/dt_ref < d for windowed tangent u*; absent =
    # class default 1/100; fraction strings like '1/64' allowed)
    dt_ref_s = config.get('Example', 'dt_ref', fallback='')
    problem.dt_ref = eval(dt_ref_s) if dt_ref_s else None
    problem.qmc_pool_exp = config.getint('Training', 'qmc_pool_exp',
                                         fallback=16)
    # "hjb3ph"): per-cadence history of the pooled pathwise RE_v on the
    # FROZEN hjb_vpt deployment-state cloud (states_<row>.npz from job
    # 533744 + its ref_v_<row>.npz).  New INI keys [PathHist]
    # states_npz / ref_npz / loggap; ALL absent (the default) => attributes
    # empty and the SOCProblem.make_logfunc hook is inert -- zero behavior
    # change on every existing INI.  Evaluation-only: no loss, no RNG, no
    # training-flow change (see examples.py SOCProblem.make_logfunc).
    problem.pathhist_states = config.get('PathHist', 'states_npz',
                                         fallback='')
    problem.pathhist_ref = config.get('PathHist', 'ref_npz', fallback='')
    problem.pathhist_loggap = config.getint('PathHist', 'loggap',
                                            fallback=0)
    return problem


def _build_net(config, dims, act_name, time_uniform_rad, shell_func=None,
               scale_lb=1.0, scale_ub=1.0, use_dist=False,
               force_no_autocast=False, net_wrapper=None, zero_last=False,
               bucket_views=False):
    """Shared FCNet builder (v_theta / u_alpha / rho) with AMP options.

    ``force_no_autocast`` keeps rho in full precision under AMP runs (the
    blueprint's behavior: autocast only wraps v_theta / u_alpha).
    ``net_wrapper`` (owner 2026-09-04 plan A, time-gated terminal anchor):
    optional callable applied to the fresh FCNet BEFORE the DDP wrap, so a
    parameterless wrapper (networks.AnchoredUNet) sits inside the DDP layer
    and the inner net's parameters enter the optimizer/DDP sync unchanged.
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
    if zero_last:
        # Opt-in zero init of the output layer (route-A rank-k example,
        # parameters the optimizer sees are already zero.
        zero_init_output_layer(net)
    if net_wrapper is not None:
        net = net_wrapper(net)
    if use_dist:
        loc_rank = get_local_rank()
        net = DDP(net.to(loc_rank), device_ids=[loc_rank],
                  gradient_as_bucket_view=bucket_views)
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
    bucket_views = bool(getattr(problem, 'efficiency_bucket_views', False))

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
                         use_dist=use_dist, bucket_views=bucket_views)
    # Time-gated terminal anchor (owner 2026-09-04 plan A):
    # [Network] u_anchor_pow (float); unset/empty = disabled = zero behavior
    # change on every existing INI.  When set, u_alpha becomes
    # phi(t) u_T + (1 - phi(t)) FCNet with phi = (t/T)^pow and u_T the
    # closed-form terminal optimal control problem.u_star_term -- a
    # parameterless wrapper INSIDE the DDP layer, so the inner FCNet's
    # parameters still feed the optimizer and the DDP gradient sync below.
    anchor_pow_s = config.get('Network', 'u_anchor_pow', fallback='').strip()
    anchor_pow = float(anchor_pow_s) if anchor_pow_s else None
    if anchor_pow is not None and getattr(problem, 'u_fixed_ref', False):
        raise ValueError('u_anchor_pow and u_fixed_ref (u* substitution) are '
                         'mutually exclusive; unset one.')
    u_wrapper = None
    if anchor_pow is not None:
        u_wrapper = lambda net: AnchoredUNet(problem, anchor_pow, net)  # noqa: E731
    u_alpha = _build_net(config,
                         [problem.dim_x] + [W_u] * H_u + [problem.dim_u],
                         config.get('Network', 'act_u'),
                         rad,
                         use_dist=use_dist,
                         bucket_views=bucket_views,
                         net_wrapper=u_wrapper,
                         zero_last=config.getboolean(
                             'Network', 'u_zero_init_last', fallback=False))

    rho_shell = getattr(torch, config.get('Network', 'rho_shell'))
    rho = _build_net(config,
                     [problem.dim_x] + [W_rho] * (H_rho + 1),
                     config.get('Network', 'act_rho'),
                     rad,
                     shell_func=lambda _x, y: rho_shell(y),
                     scale_lb=config.getfloat('Network', 'scale_lb_rho'),
                     scale_ub=config.getfloat('Network', 'scale_ub_rho'),
                     use_dist=use_dist,
                     bucket_views=bucket_views,
                     force_no_autocast=True)

    nets = {'v_theta': v_theta, 'u_alpha': u_alpha, 'rho': rho}

    # optimizers: one RMSprop per network (the paper's Section 4.1 setup)
    opt_name = config.get('Optimizer', 'optimizer', fallback='RMSprop')
    optimizer_type = getattr(torch.optim, opt_name)
    if getattr(problem, 'efficiency_fused_rmsprop', False):
        if opt_name != 'RMSprop':
            raise ValueError('Fused implementation requires the original RMSprop')
        from fused_rmsprop import FusedRMSprop
        optimizer_type = FusedRMSprop
    kwargs = ast.literal_eval(config.get('Optimizer', 'kwargs', fallback='{}'))
    decay_gap = config.getint('Optimizer', 'decay_stepgap')
    decay_rate = eval(config.get('Optimizer', 'decay_rate'))

    optim_desc, schs = [], []
    for tag, net in [('v', v_theta), ('u', u_alpha)]:
        lr0 = eval(config.get('Optimizer', f'lr0_{tag}'))
        optim = optimizer_type(net.parameters(),
                                               lr=lr0,
                                               **kwargs)
        optim_desc.append(optim)
        schs.append(
            torch.optim.lr_scheduler.StepLR(optim,
                                            step_size=decay_gap,
                                            gamma=decay_rate))
    lr0_rho = eval(config.get('Optimizer', 'lr0_rho'))
    optim_rho = optimizer_type(rho.parameters(),
                                               lr=lr0_rho,
                                               **kwargs)
    # rho tail decay (owner 2026-08-30, algorithm-side stabilization anchor
    # 3): absent = same schedule as v/u (zero behavior change); a colder
    # endpoint (e.g. 1e-2**(100/24000)) freezes the noise chasing of the
    # ascent side in the late phase while v/u keep learning.
    rho_decay_s = config.get('Optimizer', 'rho_decay_rate', fallback='')
    decay_rate_rho = eval(rho_decay_s) if rho_decay_s else decay_rate
    sch_rho = torch.optim.lr_scheduler.StepLR(optim_rho,
                                              step_size=decay_gap,
                                              gamma=decay_rate_rho)

    return nets, optim_desc, (optim_rho, ), schs + [sch_rho]


class _UStarModule(nn.Module):
    """u*-substitution ablation (owner 2026-09-04): stand-in for u_alpha that
    always returns the closed-form optimal control.

    Replacing the net object wholesale routes EVERY consumer -- the residual,
    pilot-library regeneration, rc/ualoop evaluation, the u-error log columns
    (which then read 0, self-validating) -- through u*, so the min-max problem
    degenerates to a pure linear-parabolic v solve.  A dummy non-trainable
    parameter exists only so MartNetLoss's ``next(net.parameters())`` device
    probe works; it never enters an optimizer (solve() drops the u
    optimizer/scheduler) and the module is not DDP-wrapped (no state to sync).
    """

    def __init__(self, problem):
        super().__init__()
        self._problem = problem
        # dummy non-trainable parameter: MartNetLoss probes device via
        # next(net.parameters()); never added to any optimizer.
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


def _parse_ckpt_iters(config):
    """Parse the optional read-only checkpoint iteration list.

    INI key ``[Training] ckpt_iters``: comma/whitespace separated non-negative
    integers (e.g. ``250,500,1000``).  Absent or empty = disabled = zero
    behavior change on every pre-existing configuration.  ``it=0`` is the
    state after the first update, so callers that want "after the n-th
    iteration" pass n >= 1.  Duplicates are collapsed; the result is sorted.
    """
    raw = config.get('Training', 'ckpt_iters', fallback='')
    toks = [t for t in raw.replace(',', ' ').split() if t]
    iters = []
    for tok in toks:
        try:
            val = int(tok)
        except ValueError:
            raise ValueError(f'[Training] ckpt_iters entry {tok!r} is not an '
                             f'integer (raw value {raw!r})') from None
        if val < 0:
            raise ValueError(f'[Training] ckpt_iters entry {tok!r} is '
                             f'negative (raw value {raw!r})')
        iters.append(val)
    return sorted(set(iters))


def _write_ckpt_times(path, rows):
    """Write the checkpoint I/O ledger ``<sav_path>ckpt_times.csv``.

    Checkpoint saving is NOT part of ``rt_solve`` or ``rt_log`` (the callback
    cost is subtracted from both in train.train); this file is the separate,
    explicit record of that I/O so the two performance calibers stay clean.
    Columns: ``it, seconds, bytes, path``; ``bytes`` is 0 if the file was not
    written by this rank (non-rank-0 rows never appear) or is unreadable.
    """
    pd.DataFrame(rows, columns=['it', 'seconds', 'bytes',
                                'path']).to_csv(path, index=False)


def solve(config, use_dist, sav_name=None):
    """Build the problem, run Algorithm 3.1, and save logs + curve data."""
    ip_time_gap = 1.0 if use_dist else 0.

    problem = parse_problem(config, use_dist=use_dist)
    if getattr(problem, 'x0_point_origin', False):
        # gate print for every family (the sparse class overrides
        # make_logfunc, so the base-class print never fires there)
        print(f"x0_point_origin active: X_0 fixed at the single origin "
              f"(t0={problem.t0}, z0={problem.z0}) -- training-path starts, "
              f"pilot library and the t=0 test grid all collapse onto "
              f"(t0, z0*1_d)", flush=True)
    nets, optim_desc, optim_asc, schs = parse_nets(config,
                                                   problem,
                                                   use_dist=use_dist)
    # u*-substitution ablation (owner 2026-09-04): when u_fixed_ref is set,
    # swap the learned control net for the closed-form u* module (residual,
    # pilot library, rc/ualoop, log columns all follow) and drop the u
    # optimizer/scheduler.  parse_nets returns schs as [v, u] + [rho].
    if problem.u_fixed_ref:
        ustar = _UStarModule(problem)
        ustar.to(next(nets['v_theta'].parameters()).device)
        nets['u_alpha'] = ustar
        optim_desc = optim_desc[:1]
        schs = [schs[0], schs[-1]]
    problem.attach_nets(nets)

    log_func = problem.make_logfunc(nets['v_theta'], nets['u_alpha'])
    loss_clt = MartNetLoss(problem, nets, use_dist=use_dist)

    # v_theta/u_alpha CPU state_dicts are written as `<sav_path>ckpt_it<NNNN>
    # weights.pt`; the hook runs after the update/scheduler/sync and
    # before metric logging, performs NO collective and touches no training
    # state. Only rank 0 writes; train.train times the full callback and
    # excludes it from rt_solve/rt_log (see train.py). Save time is also
    # recorded separately in `<sav_path>ckpt_times.csv`.
    output_dir = config.get('Environment', 'output_dir')
    sav_path = f'{output_dir}/{sav_name}' if sav_name is not None else None
    ckpt_iters = _parse_ckpt_iters(config)
    ckpt_rows = []
    ckpt_hook = None
    ckpt_paths = {}
    if ckpt_iters and sav_name is not None:
        ckpt_set = set(ckpt_iters)

        def ckpt_hook(it, _set=ckpt_set, _rows=ckpt_rows, _paths=ckpt_paths):
            if it not in _set:
                return 0.
            if get_global_rank() != 0:
                return 0.
            path = f'{sav_path}ckpt_it{it:04d}_weights.pt'
            t0 = time.time()
            _save_weights(nets, path[:-len('weights.pt')])
            secs = time.time() - t0
            try:
                nbytes = os.path.getsize(path)
            except OSError:
                nbytes = 0
            _rows.append((it, secs, nbytes, path))
            _paths[it] = path
            print(f'ckpt_saved it={it} bytes={nbytes} seconds={secs:.3f} '
                  f'path={path}', flush=True)
            return secs
    elif ckpt_iters:
        print('ckpt_iters set but sav_name is None: checkpoints disabled',
              flush=True)

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
        ckpt_hook=ckpt_hook,
    )

    if sav_name is not None:
        # Weight retention policy (2026-08-17): u/v weights are large and
        # must stay on the server only. 'auto' saves iff running under SLURM
        # (SLURM_JOB_ID present); local runs never save. The OneDrive mirror
        # of runs/ must exclude '*.pt'.
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
            if ckpt_rows:
                _write_ckpt_times(f'{sav_path}ckpt_times.csv', ckpt_rows)
    return hist_dict
