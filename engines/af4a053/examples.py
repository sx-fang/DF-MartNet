"""Problem classes of the companion paper (article.tex, SIREV revised).

The stochastic optimal control problem (Section 2) is

    v(t, z) = inf_kappa E[ g(Z_T) + int_t^T f(Z_s, kappa_s) ds ],
    dZ_s = mu(Z_s, kappa_s) ds + sigma(Z_s) dW_s,

on the time horizon [0, T], written in the augmented state x = (t, z).

Families reproduced here:
  * ``HJBConstantCoeff`` (Section 4.1): mu = b + 2*kappa, sigma = sqrt(2)*delta,
    f = delta^{-2} |kappa|^2, Cole-Hopf reference v = -ln E[exp(-g(X_T))]
    with analytic terminal sampling of X_T. Subclasses HJB1/HJB2a/HJB2b/
    HJB3a/HJB3b differ only in (b, delta, g).
  * ``CoupledDrift`` (Section 4.2): mu = b(t, z) + kappa with
    b_i = sin(t + i + z_{i+1} - 1), sigma = 1/2,
    f = 0.5*|kappa|^2 + eps*sin(sum_i kappa_i). At eps = 0 the Cole-Hopf
    reference v = -1/4 ln E[exp(-4 g(X_T^0))] is computed by Monte Carlo
    path simulation; for eps != 0 no reference exists and the logged error
    is measured against the eps = 0 reference (convergence as eps -> 0).
"""
import contextlib
import math
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist

import utils
from utils import (free_cache, get_safe_chunksize, isin_ddp, mc_for_v,
                   project_onto_t0te, split_number, time_mask)


class SOCProblem:
    """Stochastic optimal control problem in the augmented state x = (t, z).

    Class attributes
    ----------------
    t0, te : float
        Time horizon [t0, T] with T = te.
    dt : float
        Time step Delta t (T / N).
    s_range : float
        Range of the initial-point curve parameter s.
    M_mc : int
        Monte Carlo sample size M for the reference solution.
    num_test_points : int
        Number of test points for the logged relative errors.
    x0_curves : dict[str, callable]
        Initial/test point curves, e.g. the diagonal D_0 = {s * 1_d}.
    """

    max_step_genpoints = 500  # safety cap on path length (N steps suffice)

    t0 = 0.0
    te = 1.0
    s_range = 1.0
    M_mc = 10**6
    num_test_points = 300
    record_linf_error = True
    x0_curves = {'diag': utils.t_diag_curve}

    # -- RC metric settings (portfolio.tex sec_num convention) ----------------
    z0 = 0.0               # initial space point: x_init = (t0, z0 * 1_d)
    M_rc = 4096            # MC paths per rank for the policy-cost estimate
    has_optimal_cost = True  # False when no closed-form J(u*) exists (eps != 0)
    # True when the optimal control admits an explicit reference u*(x)
    has_optimal_control = False
    # Distributed-X0 RC caliber (owner 2026-09-14, hjb-term-rcdist): when
    # True, _initial_points draws X_0 ~ x0_curves under a fixed CRN seed and
    # rc_metric normalizes by |J*| with J* = mean v over the equispaced
    # _re_grid (the revised manuscript's E[v(0, X_0)] convention); default
    # False = the single-point tile caliber, zero behavior change.
    rc_x0_dist = False
    # Point-X0 RC caliber (owner 2026-09-18, b1origin retrain): the inverse
    # of rc_x0_dist -- for classes whose _initial_points is distributed by
    # default (SOCComplex), force the base deterministic origin tile in the
    # rc chain (J_hat from (t0, z0*1_d), J* = v(0, z0*1_d), rc /|J*|);
    # default False = zero behavior change on every existing INI.
    rc_x0_origin = False

    def __init__(self, dim_x, dt=1 / 100, batch_size=1024,
                 num_pilot_paths=10000):
        self.dim_x = dim_x          # d + 1 (time + space)
        self.dim_z = dim_x - 1      # d
        self.dim_u = dim_x - 1      # control dimension (the paper uses U = R^d)
        self.dt = dt
        self.batch_size = batch_size
        self.num_pilot_paths = num_pilot_paths
        self._Jstar = None
        self._x0_rc = None    # CRN cache of the distributed-X0 rc paths
        self.net_dict = None  # set via attach_nets()
        self.noise_sampler = 'mc'  # 'qmc': scrambled-Sobol dw in the loss
        self.qmc_pool_exp = 16     # Sobol pool size 2**qmc_pool_exp
        self._qmc_pool = None
        self._qmc_idx = 0

    def attach_nets(self, nets):
        """Attach the three networks {'v_theta', 'u_alpha', 'rho'}.

        Called by the solver right after network construction; the loss and
        the pilot-batch refresh both read the nets from here.
        """
        self.net_dict = nets

    def refresh_pilot_batch(self, x_next):
        """Mini-batch of (t, z) points from the pilot path bank (Sec. 3.4)."""
        from sampling import refresh_pilot_batch
        return refresh_pilot_batch(self, x_next, self.net_dict['u_alpha'])

    # -- problem data (to be specialized) ------------------------------------
    @property
    def has_time_variable(self):
        """x[..., 0] is the time coordinate (drives time-column init)."""
        return True

    def mu(self, x, kappa):
        """Controlled drift mu(z, kappa)."""
        raise NotImplementedError

    def sigma(self, x, dw):
        """Diffusion applied to a Brownian increment dw, i.e. sigma(z) dW."""
        raise NotImplementedError

    def mu_pilot(self, x):
        """Pilot drift (the exploration dynamics; Section 3.4)."""
        return self.mu(x, torch.zeros_like(x[..., 1:]))

    def sigma_pilot(self, x, dw):
        """Pilot diffusion (defaults to the system diffusion)."""
        return self.sigma(x, dw)

    def running_cost(self, acc, x, kappa):
        """Accumulate the running cost: return ``acc + f(x, kappa)``."""
        raise NotImplementedError

    def g(self, z):
        """Terminal cost g(z)."""
        raise NotImplementedError

    def terminal_cost(self, x):
        """g evaluated on boundary states x = (T, z)."""
        return self.g(x[..., 1:])

    # -- Euler-Maruyama transitions ------------------------------------------
    def system_step(self, x, kappa, dw=None):
        """One Euler step of the controlled system (Z_{t+dt} | Z_t, kappa).

        ``dw=None`` draws the Brownian increment by plain Monte Carlo
        (default; used by pilot-path generation, the RC metric, and the
        reference-solution simulation).  The training loss may inject a
        quasi-Monte Carlo increment instead (see ``qmc_dw``).
        """
        t, z = x[..., [0]], x[..., 1:]
        if dw is None:
            dw = torch.normal(0., self.dt**0.5, size=z.shape, device=x.device)
        z_next = z + self.mu(x, kappa) * self.dt + self.sigma(x, dw)
        return torch.cat((t + self.dt, z_next), dim=-1)

    # -- quasi-Monte Carlo increments (training loss only) -------------------
    def qmc_dw(self, n, device):
        """Brownian increments from a scrambled Sobol net + inverse normal.

        A Sobol pool of 2**qmc_pool_exp points in R^d is built once (CPU,
        scipy) and cached on the device; every draw takes the next n rows
        (cycling) and applies a fresh uniform digital shift, so each point
        stays marginally uniform (randomized QMC) and draws are independent
        across iterations and across DDP ranks.  The Gauss--Hermite/QMC
        motivation is the low-noise conditional expectation of the
        companion note martnet_trunc_dt.tex.
        """
        if self.noise_sampler != 'qmc':
            raise RuntimeError("qmc_dw called with noise_sampler != 'qmc'")
        if self._qmc_pool is None:
            self._build_qmc_pool(device)
        pool = self._qmc_pool
        idx = (torch.arange(n, device=device) + self._qmc_idx) % pool.shape[0]
        self._qmc_idx = int((self._qmc_idx + n) % pool.shape[0])
        shift = torch.rand(self.dim_z, device=device)
        p = (pool[idx] + shift) % 1.0
        z = torch.special.ndtri(p.double().clamp(1e-12, 1 - 1e-12))
        return (z * math.sqrt(self.dt)).to(pool.dtype)

    def _build_qmc_pool(self, device):
        from scipy.stats import qmc as scipy_qmc
        pool_size = 2**self.qmc_pool_exp
        eng = scipy_qmc.Sobol(d=self.dim_z, scramble=False)
        pool = torch.empty((pool_size, self.dim_z), dtype=torch.float32)
        chunk = min(8192, pool_size)
        for lo in range(0, pool_size, chunk):
            pool[lo:lo + chunk] = torch.from_numpy(eng.random(chunk))
        self._qmc_pool = pool.to(device)
        self._qmc_idx = 0
        probe = torch.special.ndtri(
            pool[:4096].double().clamp(1e-12, 1 - 1e-12))
        print(f"QMC pool built: Sobol d={self.dim_z}, 2^{self.qmc_pool_exp} "
              f"points; marginal N(0,1) check mean={probe.mean():.2e}, "
              f"std={probe.std():.6f}.")

    def pilot_step(self, x):
        """One Euler step of the pilot dynamics."""
        t, z = x[..., [0]], x[..., 1:]
        dw = torch.normal(0., self.dt**0.5, size=z.shape, device=x.device)
        z_next = z + self.mu_pilot(x) * self.dt + self.sigma_pilot(x, dw)
        return torch.cat((t + self.dt, z_next), dim=-1)

    # -- time boundary --------------------------------------------------------
    def domain_mask(self, x):
        return time_mask(x, self.t0, self.te)

    def project_onto_boundary(self, x):
        return project_onto_t0te(x, self.t0, self.te)

    # -- point generation -----------------------------------------------------
    def x0_points(self, num_points):
        """Initial points from ``x0_curves`` (split evenly across curves).

        When x0_point_origin is set (owner 2026-09-19, k1pair variant):
        X_0 is FIXED at the single deterministic origin -- training-path
        starts, the pilot library and the t=0 test grid all collapse onto
        (t0, z0 * 1_d); the example then focuses on the deployment-path
        errors and the rc, not on t=0 profiles.  Default False = the
        curve-distributed X_0, byte-identical to before.
        """
        if getattr(self, 'x0_point_origin', False):
            t_col = torch.full((num_points, 1), self.t0)
            z_col = torch.full((num_points, self.dim_z), self.z0)
            return torch.cat((t_col, z_col), dim=-1)
        curve_funcs = self.x0_curves.values()
        nums = split_number(num_points, len(curve_funcs))
        x0_list = [
            c_func(self.dim_x, n, self.s_range)[1]
            for n, c_func in zip(nums, curve_funcs)
        ]
        return torch.cat(x0_list, dim=0)

    def gen_points(self, x0, u_func=None):
        """Simulate paths from x0 until the time boundary; collect states.

        Returns ``(idx_at_tstep, x_domain, x_boundary)`` where
        ``idx_at_tstep[n]`` indexes the states at time level n.
        """
        x_active = x0[self.domain_mask(x0)]
        x_domain = x_active.clone()
        idx_at_tstep = [torch.arange(x_domain.shape[0])]
        x_outside = torch.empty((0, self.dim_x), device=x0.device)

        path_num = x_active.shape[0]
        if path_num == 0:
            raise ValueError("No x0 points within the domain.")

        for _ in range(self.max_step_genpoints):
            if x_active.shape[0] == 0:
                break
            if u_func is None:
                x_next = self.pilot_step(x_active)
            else:
                x_next = self.system_step(x_active, u_func(x_active))

            dmask = self.domain_mask(x_next)
            x_active = x_next[dmask]
            if x_active.shape[0] > 0:
                new_idx = torch.arange(x_domain.shape[0],
                                       x_domain.shape[0] + x_active.shape[0])
                idx_at_tstep.append(new_idx)
            x_domain = torch.cat([x_domain, x_active], dim=0)
            x_outside = torch.cat([x_outside, x_next[~dmask]], dim=0)

        x_boundary = self.project_onto_boundary(x_outside)
        hit_rate = x_boundary.shape[0] / path_num
        if hit_rate < 0.8:
            warnings.warn(f"Low boundary hit rate: {hit_rate:.2%}.")
        return idx_at_tstep, x_domain, x_boundary

    # -- RC metric --------------------------------------------------------------
    def _initial_points(self, num_paths):
        """(t0, z0 * 1_d) tiled to num_paths rows.

        When rc_x0_dist is set (owner 2026-09-14, hjb-term-rcdist): X_0 ~
        the class's x0_curves distribution, cached at first call under a
        fixed seed (CRN: identical every call and on every rank; the
        SOCComplex protocol) so cost_hat/rc are comparable across
        iterations.  Default False = the deterministic tile below,
        byte-identical.
        """
        if getattr(self, 'rc_x0_dist', False):
            if self._x0_rc is None or self._x0_rc.shape[0] != num_paths:
                dev = torch.empty(0).device
                rng_devs = [dev] if dev.type == 'cuda' else []
                with torch.random.fork_rng(devices=rng_devs):
                    torch.manual_seed(20260821)
                    pts = self.x0_points(num_paths)
                self._x0_rc = pts
            return self._x0_rc
        t_col = torch.full((num_paths, 1), self.t0)
        z_col = torch.full((num_paths, self.dim_z), self.z0)
        return torch.cat((t_col, z_col), dim=-1)

    @torch.no_grad()
    def estimate_cost(self, u_func, num_paths):
        """Memory-safe MC estimate of J(u_func) from the initial point.

        Walks controlled paths forward accumulating sum f * dt + g without
        materializing the path history (O(M * d) memory, safe at d = 10^4).
        Under DDP the mean is taken over all ranks' paths.
        """
        x = self._initial_points(num_paths)
        acc = torch.zeros((num_paths, 1), device=x.device)
        done_costs = []
        for _ in range(self.max_step_genpoints):
            if x.shape[0] == 0:
                break
            u = u_func(x)
            acc = acc + self.running_cost(torch.zeros_like(u[..., [0]]),
                                          x, u) * self.dt
            x_next = self.system_step(x, u)
            inside = self.domain_mask(x_next)
            if (~inside).any():
                xb = self.project_onto_boundary(x_next[~inside])
                done_costs.append(acc[~inside] + self.terminal_cost(xb))
            acc = acc[inside]
            x = x_next[inside]
        if x.shape[0] > 0:
            done_costs.append(acc + self.terminal_cost(x))
        cost = torch.cat(done_costs, dim=0).mean(0)
        if isin_ddp():
            dist.all_reduce(cost)
            cost = cost / dist.get_world_size()
        return cost

    def _re_grid(self):
        """The RE/J* grid under rc_x0_dist: num_test_points split evenly
        across x0_curves, equispaced (randomize=False) on each curve
        (general form of SOCComplex._re_grid; curve signature
        f(dim_x, n, s_range, randomize) as in solver.xcurve_gen).
        Deterministic; doubles as the J* quadrature of E[v(0, X_0)]
        (equal point counts = equal mixture weights); DDP-broadcast."""
        curve_funcs = list(self.x0_curves.values())
        nums = split_number(self.num_test_points, len(curve_funcs))
        x_test = torch.cat([
            c_func(self.dim_x, n, self.s_range, randomize=False)[1]
            for n, c_func in zip(nums, curve_funcs)
        ], dim=0)
        if isin_ddp():
            dist.broadcast(x_test, src=0)
        return x_test

    @torch.no_grad()
    def rc_metric(self, u_func):
        """RC = (J_hat(u) - J(u*)) / J(u*) of portfolio.tex sec_num (Eq. in
        Sec. Numerical example). Returns {'rc', 'cost_hat'}; rc is NaN when
        no closed-form optimal cost exists (eps != 0 in Section 4.2).

        Distributed-X0 caliber (owner 2026-09-14, hjb-term-rcdist;
        rc_x0_dist=True): J_hat integrates X_0 over the x0_curves
        distribution (CRN _initial_points), J* = mean of the reference v
        over the equispaced _re_grid (deterministic quadrature of
        E[v(0, X_0)]), and rc = (cost_hat - J*) / |J*| (the |J*|
        normalization of the revised manuscript, commit 53f809a)."""
        cost_hat = self.estimate_cost(u_func, self.M_rc)
        if getattr(self, 'rc_x0_dist', False):
            if self._Jstar is None:
                self._Jstar = self.v(self._re_grid()).mean().detach()
            rc = (cost_hat - self._Jstar) / self._Jstar.abs()
            return {'rc': rc.item(), 'cost_hat': cost_hat.item()}
        rc = torch.full_like(cost_hat, float('nan'))
        if self.has_optimal_cost:
            if self._Jstar is None:
                self._Jstar = self.v(self._initial_points(1)).detach()
            rc = (cost_hat - self._Jstar) / self._Jstar
        return {'rc': rc.item(), 'cost_hat': cost_hat.item()}

    # -- reference solution and diagnostics -----------------------------------
    def v(self, x):
        """Reference value function (Cole-Hopf / Monte Carlo)."""
        raise NotImplementedError

    def u_star(self, x):
        """Reference optimal control u*(x) (explicit feedback form)."""
        raise NotImplementedError

    def make_logfunc(self, v_theta, u_alpha=None):
        """Per-iteration logger: relative L1 (and Linf) error against v,
        and against the explicit optimal control u* when available."""
        x_test = self.x0_points(self.num_test_points)
        if isin_ddp():
            dist.broadcast(x_test, src=0)
        v_true = self.v(x_test)
        v_true_l1 = torch.abs(v_true).mean()
        v_true_linf = torch.abs(v_true).max()
        if getattr(self, 'rc_x0_dist', False):
            # Distributed-X0 caliber (owner 2026-09-14, hjb-term-rcdist):
            # cache J* = mean v over the equispaced _re_grid at startup
            # (rc_metric reuses the cache) and print it against the
            # single-point value v(0, z0*1_d) -- the two differ whenever v
            # varies along the curves (smoke acceptance gate).
            v_grid = self.v(self._re_grid())
            self._Jstar = v_grid.mean().detach()
            x_origin = torch.cat((torch.full((1, 1), self.t0),
                                  torch.full((1, self.dim_z), self.z0)),
                                 dim=-1)
            v_origin = self.v(x_origin).detach()
            if (not isin_ddp()) or dist.get_rank() == 0:
                print(f"rc_x0_dist startup: J* = mean v over "
                      f"{v_grid.shape[0]} equispaced grid points = "
                      f"{self._Jstar.item():.6e}; single-point "
                      f"v(0, z0*1_d) = {v_origin.item():.6e}", flush=True)
        if getattr(self, 'x0_point_origin', False) and (
                (not isin_ddp()) or dist.get_rank() == 0):
            print(f"x0_point_origin startup: X_0 fixed at the single "
                  f"origin (t0={self.t0}, z0={self.z0}) -- training-path "
                  f"starts, pilot library and the t=0 test grid all "
                  f"collapse onto (t0, z0*1_d)", flush=True)
        if self.has_optimal_control and u_alpha is not None:
            u_true = self.u_star(x_test)
            u_true_l1 = torch.abs(u_true).mean()
            u_true_linf = torch.abs(u_true).max()
        else:
            u_true = None

        # "hjb3ph"): per-cadence history of the pooled pathwise RE_v on the
        # FROZEN hjb_vpt deployment-state cloud.  Config via new INI keys
        # [PathHist] states_npz / ref_npz / loggap (read in
        # solver.parse_problem); keys absent => hook inert.  The state cloud
        # x_all (time-augmented rows) and the fp64 reference v_star load
        # lazily on the first cadence hit, rank 0 only (DDP replicas are
        # bitwise synced, and only rank 0 writes the log CSV; no
        # collectives, so non-rank-0 ranks skip straight to the NaN branch).
        # The metric reproduces eval_hjb_vpt_pt.py's pooled RE_v verbatim:
        # fp32 v_theta forward with the inner-net autocast disabled
        # (logfunc protocol), v_hat -> fp64, RE = mean|v_hat - v*| /
        # mean|v*|.  Read-only: no loss change, no parameter change; the
        # whole block sits inside torch.random.fork_rng so it cannot perturb
        # the training RNG stream even if a future forward consumed noise.
        ph_states = getattr(self, 'pathhist_states', '')
        ph_ref = getattr(self, 'pathhist_ref', '')
        ph_gap = getattr(self, 'pathhist_loggap', 0)
        ph_cache = {}

        def _ph_value():
            import os
            if 'x' not in ph_cache:
                st = np.load(ph_states)
                rf = np.load(ph_ref)
                assert int(rf['n_done']) == st['x_all'].shape[0], \
                    'PathHist: reference/states size mismatch'
                assert st['x_all'].shape[1] == self.dim_x, \
                    'PathHist: state dim does not match the problem'
                ph_cache['x'] = torch.from_numpy(
                    st['x_all']).to(x_test.device)
                ph_cache['vs'] = torch.from_numpy(
                    rf['v_star'].astype(np.float64)).to(x_test.device)
                if (not isin_ddp()) or dist.get_rank() == 0:
                    print(f'PathHist: loaded {st["x_all"].shape[0]} states '
                          f'from {os.path.basename(ph_states)} + reference '
                          f'from {os.path.basename(ph_ref)} '
                          f'(loggap={ph_gap})', flush=True)
            mod = getattr(v_theta, 'module', v_theta)
            flag = getattr(mod, 'enable_autocast', None)
            if flag is not None:
                mod.enable_autocast = False
            try:
                rng_devs = ([x_test.device]
                            if x_test.device.type == 'cuda' else [])
                with torch.random.fork_rng(devices=rng_devs):
                    v_hat = v_theta(ph_cache['x']).reshape(-1).double()
                err = (v_hat - ph_cache['vs']).abs()
                return (err.mean() / ph_cache['vs'].abs().mean()).item()
            finally:
                if flag is not None:
                    mod.enable_autocast = flag

        def log_func(_it):
            with torch.no_grad():
                err = v_theta(x_test) - v_true
                abs_err = torch.abs(err)
                log = {'rel_l1err': (abs_err.mean() / v_true_l1).item()}
                if self.record_linf_error:
                    log['rel_linferr'] = (abs_err.max() / v_true_linf).item()
                # u errors: relative norms over all test points and control
                # components; NaN when no explicit u* reference exists
                # (Section 4.2), mirroring the rc = NaN convention.
                if u_true is not None:
                    abs_err_u = torch.abs(u_alpha(x_test) - u_true)
                    log['rel_l1err_u'] = (
                        abs_err_u.mean() / u_true_l1).item()
                    if self.record_linf_error:
                        log['rel_linferr_u'] = (
                            abs_err_u.max() / u_true_linf).item()
                else:
                    log['rel_l1err_u'] = float('nan')
                    if self.record_linf_error:
                        log['rel_linferr_u'] = float('nan')
                # hjb3ph pooled pathwise RE_v history (see block comment
                # above): rank 0 at cadence, NaN elsewhere -- the column
                # exists on every rank/iteration so log keys stay aligned.
                if ph_states and ph_gap > 0:
                    if _it % ph_gap == 0 and (
                            (not isin_ddp()) or dist.get_rank() == 0):
                        log['rel_l1err_vpath_hist'] = _ph_value()
                    else:
                        log['rel_l1err_vpath_hist'] = float('nan')
            return log

        return log_func


class _AnalyticTerminalMC:
    """Cole-Hopf reference for constant-coefficient dynamics (Section 4.1).

    v(t, x) = -ln E[exp(-g(X_T))],  X_T = z + b*(T-t)*1_d + sgm * B_{T-t},
    with B_{T-t} ~ N(0, (T-t) I_d) sampled analytically (no time stepping).
    """

    def __init__(self, problem, mu_const, sgm_const):
        self.problem = problem
        self.mu_const = mu_const
        self.sgm_const = sgm_const

    def _samp_xte(self, x, num_mc):
        t, z = x[..., [0]], x[..., 1:]
        dt = self.problem.te - t
        norm_samp = torch.normal(mean=0.,
                                 std=1.,
                                 size=(num_mc, ) + z.shape,
                                 device=x.device)
        zte = z + self.mu_const * dt + self.sgm_const * dt.pow(0.5) * norm_samp
        te = torch.full_like(zte[..., [0]], self.problem.te)
        xte = torch.cat((te, zte), dim=-1)
        int_ft = torch.zeros_like(xte[..., [0]])
        return xte, int_ft

    def v(self, x):
        problem = self.problem
        return -torch.log(
            mc_for_v(x,
                     terminal_cost=lambda x_: torch.exp(
                         -problem.g(x_[..., 1:])),
                     samp_xte_intft=self._samp_xte,
                     use_dist=isin_ddp(),
                     M=problem.M_mc,
                     combine='plus'))

    def u_star(self, x, delta):
        """Explicit optimal control u* = -delta^2 * grad_z v, where

            grad_z v = E[exp(-g(Z_T)) grad g(Z_T)] / E[exp(-g(Z_T))]

        (since d Z_T / d z = I for the constant-coefficient dynamics).
        Self-normalized ratio estimator using the same analytic terminal
        samples as ``v``; chunked and DDP-aware (all ranks must call
        together), mirroring ``mc_for_v``.
        """
        problem = self.problem
        use_dist = isin_ddp()
        M = problem.M_mc
        if use_dist:
            world_size = dist.get_world_size()
            M = int(M // world_size) + 1
            rank = dist.get_rank()
        else:
            rank = 0

        multiplier = 12 * x.numel()
        cum_size = 0
        cum_w = 0.
        cum_wg = 0.
        progress = 0.
        max_chunksize = None  # persistent OOM clamp (see utils.mc_for_v)
        print(f"Monte-Carlo for reference optimal control on {x.device}...\n")
        while cum_size < M:
            try:
                chunksize = get_safe_chunksize(multiplier, x.dtype, x.device,
                                               use_percent=0.4)
                if max_chunksize is not None:
                    chunksize = min(chunksize, max_chunksize)
                chunksize = max(1, min(chunksize, M - cum_size))
                if use_dist:
                    chunksize_tensor = torch.tensor(chunksize,
                                                    device=x.device)
                    dist.all_reduce(chunksize_tensor, op=dist.ReduceOp.MIN)
                    chunksize = int(chunksize_tensor.item())
                xte_chunk, _ = self._samp_xte(x, num_mc=chunksize)
                zte = xte_chunk[..., 1:]
                w_chunk = torch.exp(-problem.g(zte))
                wg_chunk = w_chunk * problem.grad_g(zte)
                new_w = w_chunk.mean(0)
                new_wg = wg_chunk.mean(0)
                cum_size += chunksize
                if use_dist:
                    dist.all_reduce(new_w, op=dist.ReduceOp.SUM)
                    new_w = new_w / world_size
                    dist.all_reduce(new_wg, op=dist.ReduceOp.SUM)
                    new_wg = new_wg / world_size
                new_rate = chunksize / cum_size
                cum_w = (1 - new_rate) * cum_w + new_rate * new_w
                cum_wg = (1 - new_rate) * cum_wg + new_rate * new_wg

                if (cum_size / M > progress + 0.01) or (cum_size == M):
                    progress = cum_size / M
                    if rank == 0:
                        print(f"Progress: {progress:.2%}, "
                              f"chunksize per rank: {chunksize}")
            except RuntimeError as err:
                if 'out of memory' in str(err):
                    chunksize = int(chunksize // 2)
                    max_chunksize = chunksize
                    print(f"Restricted by memory, reduce chunksize to "
                          f"{chunksize}")
                    if chunksize == 0:
                        raise err
                    free_cache(x.device)  # only on OOM (see utils.mc_for_v)
                else:
                    raise err
        return -delta**2 * cum_wg / cum_w


class _PathMCReference:
    """Cole-Hopf reference by path simulation (needed for Section 4.2,
    where the uncontrolled drift is state-dependent and no analytic
    terminal distribution exists):

        v(t, x) = -c_pow^{-1} ln E[exp(-c_pow * g(X_T^0))],

    with X^0 the uncontrolled (kappa = 0) Euler-Maruyama path.
    """

    max_mc_steps = 10**6

    def __init__(self, problem, c_pow):
        self.problem = problem
        self.c_pow = c_pow

    def _samp_xte_intft(self, x, num_mc):
        problem = self.problem
        # reshape of the expanded view ALIASES the caller's storage when
        # x.shape[0] == 1 (stride-0 rows, no copy): every MC "path" would
        # collapse onto one row and corrupt x itself (diagnosed via jobs
        # 478850/478873; multi-point grids always copied, so production
        # runs were unaffected).  Always materialize independent rows.
        xte = x.expand((num_mc, ) + x.shape).reshape(
            -1, x.shape[-1]).clone()
        inside = problem.domain_mask(xte)
        path_num = xte.shape[:-1].numel()
        if path_num == 0:
            raise ValueError("No x0 points within the domain.")
        for _ in range(self.max_mc_steps):
            if inside.sum() == 0:
                break
            x_next = problem.system_step(xte[inside],
                                         torch.zeros_like(xte[inside][..., 1:]))
            xte[inside] = x_next
            dmask = problem.domain_mask(x_next)
            inside[inside.eq(True)] = dmask
        xte = problem.project_onto_boundary(xte)
        xte = xte.reshape((num_mc, ) + x.shape)
        int_ft = torch.zeros_like(xte[..., [0]])
        return xte, int_ft

    def v(self, x):
        problem = self.problem
        v_inner = mc_for_v(x,
                           terminal_cost=lambda x_: torch.exp(
                               -self.c_pow * problem.g(x_[..., 1:])),
                           samp_xte_intft=self._samp_xte_intft,
                           use_dist=isin_ddp(),
                           M=problem.M_mc,
                           combine='multiply',
                           multiplier=getattr(problem, 'mc_mem_multiplier',
                                              6),
                           use_percent=getattr(problem, 'mc_use_percent',
                                               0.4))
        return -torch.log(v_inner) / self.c_pow


# ---------------------------------------------------------------------------
# Section 4.1: HJB equations with constant coefficients
# ---------------------------------------------------------------------------
class HJBConstantCoeff(SOCProblem):
    """Section 4.1 family: dZ = (b*1_d + 2*kappa) dt + sqrt(2)*delta dW,
    f = delta^{-2} |kappa|^2, U = R^d, T = 1. The HJB equation reads
        dt v + inf_kappa { (b + 2*kappa).grad v + delta^2 Tr[Hess v]
                           + delta^{-2} |kappa|^2 } = 0.
    """

    b = 1.0       # drift constant (b * 1_d)
    delta = 0.2   # diffusion intensity delta
    # u* = -delta^2 grad_z v (argmin of 2*kappa.grad v + delta^{-2}|kappa|^2)
    has_optimal_control = True

    def mu(self, x, kappa):
        return self.b + 2 * kappa

    def sigma(self, x, dw):
        return (2**0.5) * self.delta * dw

    def mu_pilot(self, x):
        z = x[..., 1:]
        return self.b * torch.ones_like(z)

    def running_cost(self, acc, x, kappa):
        return acc + self.delta**(-2) * kappa.pow(2).sum(-1, keepdims=True)

    def v(self, x):
        return _AnalyticTerminalMC(self,
                                   mu_const=self.b,
                                   sgm_const=(2**0.5) * self.delta).v(x)

    def u_star(self, x):
        return _AnalyticTerminalMC(
            self, mu_const=self.b,
            sgm_const=(2**0.5) * self.delta).u_star(x, delta=self.delta)

    def u_star_term(self, x):
        """Closed-form TERMINAL optimal control u_T (owner 2026-09-14,
        hjb-term-rcdist; the hook w_term_u requires): u_T = -delta^2 grad v
        with grad v(T, .) = grad g by Cole-Hopf, and each subclass supplies
        an analytic grad_g (HJB1: 2z/(1+|z|^2); HJB3a: explicit; HJB3b
        inherits).  Detached; depends only on the problem data (g, delta),
        never on the solution."""
        return (-self.delta**2 * self.grad_g(x[..., 1:])).detach()


class HJB1(HJBConstantCoeff):
    """HJB-1: b = 0, delta = 1, g(z) = ln(0.5 (1 + |z|^2))."""
    b = 0.0
    delta = 1.0

    def g(self, z):
        return torch.log(0.5 * (1 + z.pow(2).sum(-1, keepdim=True)))

    def grad_g(self, z):
        """grad g(z) = 2 z / (1 + |z|^2)."""
        return 2 * z / (1 + z.pow(2).sum(-1, keepdim=True))


class HJB2a(HJB1):
    """HJB-2a: HJB-1 with b = 1_d and delta = 0.1."""
    b = 1.0
    delta = 0.1


class HJB2b(HJB2a):
    """HJB-2b: HJB-2a with delta = 0.05."""
    delta = 0.05


class HJB3a(HJBConstantCoeff):
    """HJB-3a: b = 1_d, delta = 0.2, terminal g(z) = gtilde(z - 1_d) with
        gtilde(y) = (1/d) sum_i [ sin(y_i - pi/2) + sin( 1/(delc + y_i^2) ) ],
    delc = 0.3/pi (the published text's 0.1*pi is a typo; the code and the
    archived reference value v(0, 0) = -1.0398 use 0.3/pi).
    """
    b = 1.0
    delta = 0.2
    delc = 0.3 / math.pi

    def g(self, z):
        y = z - 1.
        osc = torch.sin(y - math.pi / 2) + torch.sin(
            (self.delc + y**2)**(-1))
        return osc.mean(-1, keepdim=True)

    def grad_g(self, z):
        """grad g(z): y = z - 1_d, h_i = (delc + y_i^2)^{-1},
        d g / d z_i = [cos(y_i - pi/2) - 2 y_i cos(h_i) h_i^2] / d."""
        y = z - 1.
        h = (self.delc + y**2)**(-1)
        return (torch.cos(y - math.pi / 2) -
                2 * y * torch.cos(h) * h**2) / self.dim_z


class HJB3b(HJB3a):
    """HJB-3b: HJB-3a with delta = 0.1 (6 hidden layers in Table 1)."""
    delta = 0.1


# ---------------------------------------------------------------------------
# Section 4.1 variants with the sine ring drift of Section 4.2 (owner
# instruction 2026-08-20): the constant drift b*1_d is replaced by
#   b_i(t, z) = sin(t + i + z_{i+1})   (cyclic, 0-based; equals the paper's
#   1-based sin(t + i + z_{i+1} - 1) after the index shift, same convention
#   as CoupledDrift);
# sigma = sqrt(2)*delta, f = delta^{-2}|kappa|^2 and the terminal g of each
# base example are unchanged.  Cole--Hopf still applies (control enters the
# drift as 2*kappa), so the reference value is
#   v(t, x) = -ln E[exp(-g(X_T^0))],
# computed by path Monte Carlo of the uncontrolled dynamics (_PathMCReference
# with c_pow = 1, M_mc = 1e5 paths at the problem dt; noise floor ~3e-3,
# same protocol as the eps = 0 case of Section 4.2).  The reference control
# u* = -delta^2 grad v would need the full pathwise Jacobian of X_T^0
# (O(d^2) per path; dense g gives no windowed shortcut), infeasible at
# d = 10^4: has_optimal_control = False and the u error columns are NaN by
# design (same convention as Section 4.2).  Control quality is assessed by
# rc (cost-functional bound; has_optimal_cost stays True via J* = v(0, x0)).
class _SinRingDrift:
    """Mixin: sine ring drift replacing the constant b*1_d of Section 4.1."""

    M_mc = 10**5
    has_optimal_control = False

    def _b(self, x):
        t, z = x[..., [0]], x[..., 1:]
        i_idx = torch.arange(self.dim_z, dtype=z.dtype, device=z.device)
        return torch.sin(t + i_idx + z.roll(-1, dims=-1))

    def mu(self, x, kappa):
        return self._b(x) + 2 * kappa

    def v(self, x):
        return _PathMCReference(self, c_pow=1.0).v(x)


class HJB2aSin(_SinRingDrift, HJB2a):
    """HJB-2a with sine ring drift: g = ln(0.5(1+|z|^2)), delta = 0.1."""


class HJB2bSin(_SinRingDrift, HJB2b):
    """HJB-2b with sine ring drift: g = ln(0.5(1+|z|^2)), delta = 0.05."""


class HJB3aSin(_SinRingDrift, HJB3a):
    """HJB-3a with sine ring drift: oscillatory separable g, delta = 0.2."""


class HJB3bSin(_SinRingDrift, HJB3b):
    """HJB-3b with sine ring drift: oscillatory separable g, delta = 0.1
    (6 hidden layers, set in the INI as for HJB-3b)."""


# ---------------------------------------------------------------------------
# Section 4.2: coupled drift, non-separable terminal, perturbed cost
# ---------------------------------------------------------------------------
class CoupledDrift(SOCProblem):
    """Section 4.2 example: mu = b(t, z) + kappa with
        b_i(t, z) = sin(t + i + z_{i+1} - 1)   (cyclic index, d >= 1),
    sigma = 1/2, f(z, kappa) = 0.5 |kappa|^2 + eps * sin(sum_i kappa_i),
    g(z) = ln(1 + mean_i(z_i^2) + 0.5 * mean_i(sin(10 z_i))).

    At eps = 0 the Cole-Hopf reference is
        v(t, x) = -(1/4) ln E[exp(-4 g(X_T^0))],
    computed by Monte Carlo path simulation of the uncontrolled dynamics.
    For eps != 0 there is no closed form; the logged error still uses the
    eps = 0 reference (v_eps -> v_0 as eps -> 0).
    """

    eps = 1.0
    sigma_const = 0.5
    # Cole-Hopf exponent c = sigma^{-2} = 4 at eps = 0
    cole_hopf_c = 4.0
    has_optimal_cost = False  # eps != 0: no closed-form J(u*)
    # has_optimal_control stays False: at eps = 0, u* = -grad_z v exists but
    # grad_z v requires the pathwise Jacobian of X_T^0 w.r.t. z through the
    # state-dependent drift (O(d^2) per path, infeasible at d = 10^3); for
    # eps != 0 the minimizer has no closed form. The logged u errors are NaN.

    M_mc = 10**5  # path-simulated reference is expensive; noise floor ~3e-3
    x0_curves = {'diag': utils.t_diag_curve, 'manifold': utils.t_manifold_curve}

    def _b(self, x):
        """b_i(t, z) = sin(t + i + z_{i+1} - 1) with cyclic z_{d+1} = z_1.

        With 0-based indexing i = 0..d-1 this reads sin(t + i + z_{i+1})
        (the -1 of the paper's 1-based formula is absorbed by the shift).
        """
        t, z = x[..., [0]], x[..., 1:]
        i_idx = torch.arange(self.dim_z, dtype=z.dtype, device=z.device)
        return torch.sin(t + i_idx + z.roll(-1, dims=-1))

    def mu(self, x, kappa):
        return self._b(x) + kappa

    def sigma(self, x, dw):
        return self.sigma_const * dw

    def running_cost(self, acc, x, kappa):
        base = 0.5 * kappa.pow(2).sum(-1, keepdims=True)
        pert = self.eps * torch.sin(kappa.sum(-1, keepdims=True))
        return acc + base + pert

    def g(self, z):
        return torch.log(1 + z.pow(2).mean(-1, keepdim=True) +
                         0.5 * torch.sin(10 * z).mean(-1, keepdim=True))

    def v(self, x):
        return _PathMCReference(self, c_pow=self.cole_hopf_c).v(x)


class CoupledDriftEps0(CoupledDrift):
    """eps = 0: the reference case with the closed-form Cole-Hopf solution."""
    eps = 0.0
    has_optimal_cost = True


class CoupledDriftEps2(CoupledDrift):
    eps = 0.5


class CoupledDriftEps4(CoupledDrift):
    eps = 0.25


class CoupledDriftEps8(CoupledDrift):
    eps = 0.125


# S0-only (diag-curve) variants for the Sec. 4.2 d=10^4 replacement experiment
# (owner ruling 2026-09-13: evaluation surface S0 only, no S1).  The single-
# curve x0_curves dict drives BOTH the pilot initial points and the evaluation
# surface, so training and testing stay on the same surface by construction.
class CoupledDriftEps0S0(CoupledDriftEps0):
    x0_curves = {'diag': utils.t_diag_curve}


class CoupledDriftEps8S0(CoupledDriftEps8):
    x0_curves = {'diag': utils.t_diag_curve}


class CoupledDriftEps4S0(CoupledDriftEps4):
    x0_curves = {'diag': utils.t_diag_curve}


class CoupledDriftEps2S0(CoupledDriftEps2):
    x0_curves = {'diag': utils.t_diag_curve}


class CoupledDriftS0(CoupledDrift):
    x0_curves = {'diag': utils.t_diag_curve}


# ---------------------------------------------------------------------------
# Section 4.2 (sec_complex): SOCP with complex state dynamics
# (formula frozen by owner 2026-08-21)
#
#   mu = b + c0 s kappa,  sigma = s,
#   b_i = sin(t + i + Phi_i - 1),  Phi_i = sum_{k=1}^d 0.9^k x_{i+k},
#   s_ij = sqrt(2) c1 0.9^|i-j| (1 + 0.5 cos(t + i + x_{i+1})),
#   f = 0.5 |kappa|^2,  g(x) = ln(1 + (1/d) sum_i (x_i^2 + 0.5 sin(10 x_i))),
#   c0 = 2, c1 = 0.5, T = 1; cyclic index convention x_{d+k} := x_k.
#
# Structure: s = sqrt(2) c1 diag(m) T with m_i = 1 + 0.5 cos(t+i+x_{i+1})
# in [0.5, 1.5] and T the AR(1) Toeplitz matrix T_ij = 0.9^|i-j| (eigenvalues
# in [(1-q)/(1+q), (1+q)/(1-q)], d-independent), hence s is nonsingular with
# d-uniform ellipticity bounds; per-coordinate noise rate sum_j s_ij^2
# ~= 4.7 m_i^2 in [1.2, 10.7] (O(1) in d, same order as HJB-1's 2).
#
# Reference: the control drift c0 s kappa and the diffusion share the same
# matrix s, so the HJB linearizes under v = -(1/c0^2) ln phi (generalized
# Cole-Hopf):  v(t, x) = -(1/c0^2) ln E[exp(-c0^2 g(X_T^0))]  with X^0 the
# uncontrolled path (drift b, diffusion s) -> _PathMCReference(c_pow=4),
# M_mc = 1e5 at the problem dt, noise floor ~3e-3 (as Section 4.2 eps=0).
# u* = -c0 s^T grad_z v requires the full path Jacobian (infeasible at
# d = 10^4) -> u error columns NaN by design; control quality via rc.
#
# Evaluation protocols (article.tex, 2026-08-21 revision):
#   * X_0 ~ uniform on S_0 cup S_1 (50/50 mixture of the diagonal and the
#     manifold curve): pilot starts via x0_curves; cost_hat/rc use a cached
#     CRN sample (first-call fixed, identical across ranks).
#   * RE(v_theta) is taken over the union of 100-point UNIFORM grids on S_0
#     and S_1 (200 points, equispaced, fp32 network evaluation).
#   * J* = E[v(0, X_0)] is the reference v averaged over a CRN subsample of
#     the cached X_0 set (Jstar_sub points), computed once and cached.
# ---------------------------------------------------------------------------
class SOCComplex(SOCProblem):
    """SOCP with complex state dynamics (Section 4.2, sec_complex).

    u* strong-signal cusp variant (owner 2026-09-04): g_cusp_amp/g_cusp_idx
    add g_cusp_amp * sum_i sin(x_i) on individual coordinates (NOT /d) so
    u*(T^-) peaks O(1) there; see g()."""

    c0 = 2.0
    c1 = 0.5
    q_decay = 0.9
    mod_amp = 0.5
    # B' freeze (owner 2026-08-21): drift amplitude x2 retained in the
    # dynamics (probe A: no profile contribution; owner informed)
    drift_amp = 2.0
    # time-oscillation frequency (owner 2026-09-20, c2scale probe): the t
    # phase of BOTH the drift b and the diffusion modulation m reads
    # time_freq * t; default 1.0 = the frozen B' dynamics bit-for-bit
    # (1.0 * t is IEEE-exact).  parse_problem injects from INI [Example]
    # when set; nothing else (g, c0, c1, amplitudes, control set) moves.
    time_freq = 1.0
    cole_hopf_c = 4.0          # = c0**2
    # terminal-cost sin amplitude (owner 2026-08-23 probe: sweep 3.5/4.0/4.5
    # to widen the v(0,·) contrast on S0/S1 beyond span ~0.57; parse_problem
    # injects g_sin_amp from INI [Example] when set)
    g_sin_amp = 2.0
    # span levers (owner 2026-08-24 span>1 probe grid): ln-shell coefficient
    # A and overall terminal-cost scale B; defaults keep the B' freeze g
    # unchanged; parse_problem injects from INI [Example] when set
    g_shell_A = 1.0
    g_scale_B = 1.0
    # u* strong-signal cusp (owner 2026-09-04, approved design: idx {0, d/2},
    # amp 0.45): g += g_cusp_amp * sum_{i in g_cusp_idx} sin(x_i) on
    # individual coordinates, NOT divided by d -> those coordinates carry
    # O(amp) gradient and u*(T^-) = -c0 s^T grad g peaks O(1) on a
    # q^|i-j| Toeplitz support (~10-20 coords) over the ~1e-3 smeared
    # background; parse_problem injects both from INI [Example]; defaults
    # 0.0 / () = zero effect on all existing INIs.
    g_cusp_amp = 0.0
    g_cusp_idx = ()
    # single-coordinate control anchor (owner 2026-09-20, uanchor_v1
    # campaign): g += g_anchor_alpha * sin(x_1) on coordinate 1 alone (NOT
    # divided by d), so u* = -c0 s^T grad v carries an O(alpha) anchor
    # component over the ~1e-3 smeared 1/d-weighted B1 background;
    # parse_problem injects from INI [Example]; default 0.0 = zero effect
    # on all existing INIs.  (Functionally the idx-(0,) special case of
    # g_cusp_*; separate knob per the owner directive for campaign
    # provenance.)
    g_anchor_alpha = 0.0
    # sqrt(d)-scaled terminal sin (owner 2026-09-23, Route-2 u*-richness
    # probe after the 536526 calibration): g += g_sin_amp_sqrt/sqrt(d) *
    # sum_i sin(x_i), ADDITIVE on the full g (outside g_scale_B), the
    # calib_route2.py arm formula bit-for-bit; per-coordinate gradient
    # O(amp/sqrt(d)), Euclid norm O(amp) -> u* stays O(1) at d=1e4
    # (calibration: A=0.1 -> gain 9.7%, A=0.2 -> 32%); parse_problem
    # injects from INI [Example]; default 0.0 = zero effect on all
    # existing INIs.
    g_sin_amp_sqrt = 0.0
    # convex-separable Route-3 terminal (owner 2026-09-24, after the sin
    # probe verdict 536533/34): g += g_abs_amp/sqrt(d) * sum_i(|x_i-a| -
    # g_abs_cen), CENTERED -- the raw sum has mean ~ d*E|x_i-a|, an
    # x0-independent constant A*sqrt(d)*mu that shifts J* (and the RE
    # denominators) without touching any gradient; g_abs_cen = mu_a is
    # measured once by the origin-process MC (calib_route3_abs.py prints
    # it) and hardcoded in the probe INI.  a = g_abs_shift (1.0 default)
    # puts the kink at the edge of the terminal typical set so the
    # per-coordinate gradient E sign(x_i-a) varies smoothly along the
    # S0/S1 curves (erf-shaped, monotone -- no oscillation, unlike the
    # sin route).  parse_problem injects all three; defaults 0.0/1.0/0.0
    # = zero effect on all existing INIs.
    g_abs_amp = 0.0
    g_abs_shift = 1.0
    g_abs_cen = 0.0
    M_mc = 10**5               # path-MC reference; noise floor ~3e-3
    has_optimal_cost = True    # generalized Cole-Hopf reference exists
    has_optimal_control = False  # u* needs the path Jacobian -> u cols NaN
    # memory model of the FFT path-MC (probe 478300, d=1e4, 200 pts, fp32):
    # measured 129.7 MiB/path -> budget 18 x numel(x_test) x 4B = 138 MiB
    # (+6% margin); use_percent 0.7 (owner 2026-08-21)
    mc_mem_multiplier = 18
    mc_use_percent = 0.7
    x0_curves = {'diag': utils.t_diag_curve,
                 'manifold': utils.t_manifold_curve}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._x0_rc = None
        self._Jstar_pt = None
        self._phi_kernel = None
        self._toep_kernel = None

    # -- drift b ------------------------------------------------------------
    def _phi(self, z):
        """Phi_i = sum_{k=1}^d q^k z_{i+k} (cyclic), exact circular
        correlation via FFT in the input dtype (fp64 in the zero-checks ->
        machine precision; fp32 in production -> half the FFT workspace).
        The k = d term folds onto shift 0 (z_{i+d} = z_i) with weight q^d."""
        d = self.dim_z
        key = (d, z.device, z.dtype)
        if getattr(self, '_phi_key', None) != key:
            ker = self.q_decay ** torch.arange(1, d + 1, dtype=z.dtype,
                                               device=z.device)
            ker = torch.cat([ker[-1:], ker[:-1]])  # [q^d, q^1, ..., q^{d-1}]
            self._phi_kernel = torch.conj(torch.fft.fft(ker))
            self._phi_key = key
        return torch.fft.ifft(self._phi_kernel * torch.fft.fft(z)).real

    def _b(self, x):
        """b_i = drift_amp * sin(t + i + Phi_i - 1), 1-based i; with 0-based
        i0 = i - 1 this reads sin(t + i0 + Phi) (the +i - 1 cancels).
        drift_amp = 2 per the B' freeze (owner 2026-08-21).  time_freq
        (owner 2026-09-20) rescales the t phase: sin(time_freq t + ...),
        1.0 = the frozen form."""
        t, z = x[..., [0]], x[..., 1:]
        i0 = torch.arange(self.dim_z, dtype=torch.float32, device=z.device)
        return self.drift_amp * torch.sin(self.time_freq * t + i0
                                          + self._phi(z))

    # -- diffusion matrix s = sqrt(2) c1 diag(m) T ---------------------------
    def _toep(self, w):
        """(T w)_i = sum_j q^|i-j| w_j (plain, NON-cyclic Toeplitz), exact
        via zero-padded FFT linear convolution in the input dtype (fp64 in
        the zero-checks, fp32 in production; bf16 inputs upcast to fp32
        since torch.fft has no bf16 support)."""
        orig_dtype = w.dtype
        if orig_dtype not in (torch.float32, torch.float64):
            w = w.float()
        d = w.shape[-1]
        n = 2 * d
        key = (d, w.device, w.dtype)
        if getattr(self, '_toep_key', None) != key:
            m = torch.arange(-(d - 1), d, dtype=w.dtype, device=w.device)
            hpad = torch.zeros(n, dtype=w.dtype, device=w.device)
            hpad[:2 * d - 1] = self.q_decay ** m.abs()
            self._toep_kernel = torch.fft.fft(hpad)
            self._toep_key = key
        wpad = torch.zeros(w.shape[:-1] + (n,), dtype=w.dtype,
                           device=w.device)
        wpad[..., :d] = w
        c = torch.fft.ifft(self._toep_kernel * torch.fft.fft(wpad)).real
        return c[..., d - 1:2 * d - 1].to(orig_dtype)  # (Tw)_i = c[i+d-1]

    def _mod(self, x):
        """m_i = 1 + 0.5 cos(t + i + x_{i+1}), 1-based i, cyclic x_{d+1}=x_1;
        0-based: i0 = i - 1, argument t + (i0 + 1) + z_{i0+1 mod d}.
        time_freq (owner 2026-09-20) rescales the t phase: cos(time_freq t
        + ...), 1.0 = the frozen form."""
        t, z = x[..., [0]], x[..., 1:]
        i0 = torch.arange(self.dim_z, dtype=torch.float32, device=z.device)
        return 1.0 + self.mod_amp * torch.cos(
            self.time_freq * t + (i0 + 1) + z.roll(-1, dims=-1))

    def _s_matvec(self, x, w):
        """s(x) @ w = sqrt(2) c1 * m(x) * (T w)."""
        return math.sqrt(2.0) * self.c1 * (self._mod(x) * self._toep(w))

    # -- SOCProblem interface -------------------------------------------------
    def mu(self, x, kappa):
        b = self._b(x)
        if not torch.any(kappa):
            return b  # reference/uncontrolled paths: skip the s matvec
        return b + self.c0 * self._s_matvec(x, kappa)

    def sigma(self, x, dw):
        return self._s_matvec(x, dw)

    def running_cost(self, acc, x, kappa):
        return acc + 0.5 * kappa.pow(2).sum(-1, keepdims=True)

    def g(self, z):
        """g(x) = B * [ln(1 + A * mean x_i^2) + 0.3 mean sin(0.5 x_i) + g_sin_amp * mean sin(x_i)]
        (B' freeze, owner 2026-08-21: low-frequency oscillations OUTSIDE the
        ln shell -- inside-ln forms non-viable, precheck 478305; Var(g)
        ~7-8e-2 -> reference MC floor ~6e-3, owner accepted; g_sin_amp = 2.0
        default, INI-overridable since 2026-08-23 contrast probe; span
        levers g_shell_A/g_scale_B = 1.0 defaults, owner 2026-08-24).

        Plus the cusp addend g_cusp_amp * sum_{i in g_cusp_idx} sin(x_i)
        (owner 2026-09-04, u* strong-signal variant; NOT divided by d).
        Plus the single-coordinate anchor g_anchor_alpha * sin(x_1)
        (owner 2026-09-20, uanchor_v1; NOT divided by d).
        Plus the sqrt(d)-scaled Route-2 addend g_sin_amp_sqrt/sqrt(d) *
        sum_i sin(x_i) (owner 2026-09-23 u*-richness probe; Euclid-norm
        O(amp) terminal gradient, calib 536526)."""
        out = (self.g_scale_B
               * (torch.log(1 + self.g_shell_A * z.pow(2).mean(-1, keepdim=True))
                  + 0.3 * torch.sin(0.5 * z).mean(-1, keepdim=True)
                  + self.g_sin_amp * torch.sin(z).mean(-1, keepdim=True)))
        # numeric guard: zero amp or empty idx = the frozen B' g exactly
        if getattr(self, 'g_cusp_amp', 0.0) and getattr(self, 'g_cusp_idx', ()):
            out = out + self.g_cusp_amp * torch.sin(
                z[..., list(self.g_cusp_idx)]).sum(-1, keepdim=True)
        # single-coordinate control anchor (uanchor_v1, owner 2026-09-20):
        # guarded so the default 0.0 = the frozen B' g exactly (branch
        # never fires on existing INIs)
        if getattr(self, 'g_anchor_alpha', 0.0):
            out = out + self.g_anchor_alpha * torch.sin(z[..., [0]])
        # sqrt(d)-scaled Route-2 addend (owner 2026-09-23): g +=
        # g_sin_amp_sqrt/sqrt(d) * sum_i sin(x_i), additive on the FULL g
        # (outside g_scale_B) exactly as calib_route2.py patched it; guarded
        # so the default 0.0 = the frozen g (branch never fires on
        # existing INIs)
        if getattr(self, 'g_sin_amp_sqrt', 0.0):
            out = out + (self.g_sin_amp_sqrt / math.sqrt(self.dim_z)
                         ) * torch.sin(z).sum(-1, keepdim=True)
        # centered convex-separable Route-3 addend (owner 2026-09-24):
        # g += g_abs_amp/sqrt(d) * (sum_i |x_i - a| - d*g_abs_cen),
        # additive on the FULL g exactly as calib_route3_abs.py patched
        # it; guarded so the default 0.0 = the frozen g
        if getattr(self, 'g_abs_amp', 0.0):
            out = out + (self.g_abs_amp / math.sqrt(self.dim_z)) * (
                (z - self.g_abs_shift).abs().sum(-1, keepdim=True)
                - self.dim_z * self.g_abs_cen)
        return out

    def v(self, x):
        return _PathMCReference(self, c_pow=self.cole_hopf_c).v(x)

    def u_star_term(self, x):
        """Closed-form TERMINAL optimal control u_T (owner 2026-09-04 plan A,
        time-gated anchor): u_T(x) = -c0 sqrt(2) c1 * Toep(m(x) * grad g(x))
        -- Cole-Hopf grad v(T^-, .) = grad g, the exact formula of
        eval_complex_cusp_profile.py, valid for every g knob (g_sin_amp /
        g_shell_A / g_scale_B / g_cusp_*) because grad g is taken by autograd
        on a detached clone of z (enable_grad so no_grad callers -- pilot-bank
        refresh, rc paths -- work too), then detached.  m(x)/Toep use the
        queried x as passed (m reads x's own time column).  Depends only on
        the problem data (g, B, R), never on the solution."""
        z = x[..., 1:].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            g = self.g(z)
            grad_g = torch.autograd.grad(g.sum(), z)[0]
        m = self._mod(x)
        return (-self.c0 * math.sqrt(2.0) * self.c1
                * self._toep(m * grad_g)).detach()

    # -- X_0 mixture protocol (uniform on S_0 cup S_1) -------------------------
    def _initial_points(self, num_paths):
        """X_0 ~ 50/50 mixture of the two curves with s ~ U(-1, 1); cached
        at first call under a fixed seed (CRN: identical every call and on
        every rank) so cost_hat/rc are comparable across iterations.

        Point-X0 caliber (owner 2026-09-18, b1origin retrain): when
        rc_x0_origin is set, the base-class deterministic origin tile
        (t0, z0*1_d) replaces the mixture -- the uniform point-caliber rc
        convention of the revision; default False = the mixture below,
        byte-identical."""
        if getattr(self, 'rc_x0_origin', False):
            t_col = torch.full((num_paths, 1), self.t0)
            z_col = torch.full((num_paths, self.dim_z), self.z0)
            return torch.cat((t_col, z_col), dim=-1)
        if self._x0_rc is None or self._x0_rc.shape[0] != num_paths:
            dev = torch.empty(0).device
            rng_devs = [dev] if dev.type == 'cuda' else []
            with torch.random.fork_rng(devices=rng_devs):
                torch.manual_seed(20260821)
                pts = self.x0_points(num_paths)
            self._x0_rc = pts
        return self._x0_rc

    @torch.no_grad()
    def rc_metric(self, u_func):
        """rc = (J_hat(u) - J*) / J* with J(u) integrating X_0 over the
        S_0/S_1 mixture; J* = mean of the reference v over the equispaced
        RE grid (deterministic quadrature of E[v(0, X_0)] under the 50/50
        mixture -- equal point counts on the two curves), computed once in
        make_logfunc and cached (owner 2026-08-21, plan A: no separate
        path-MC for J*).

        Point-X0 caliber (owner 2026-09-18, b1origin retrain): when
        rc_x0_origin is set, J_hat starts from the deterministic origin
        (t0, z0*1_d) and J* = v(0, z0*1_d) (cached separately as _Jstar_pt,
        warmed in make_logfunc); rc normalizes by |J*| per the revised
        manuscript's convention (53f809a)."""
        cost_hat = self.estimate_cost(u_func, self.M_rc)
        if getattr(self, 'rc_x0_origin', False):
            if self._Jstar_pt is None:  # defensive fallback if logfunc not built
                self._Jstar_pt = self.v(self._initial_points(1)).detach()
            rc = (cost_hat - self._Jstar_pt) / self._Jstar_pt.abs()
            return {'rc': rc.item(), 'cost_hat': cost_hat.item()}
        if self._Jstar is None:  # defensive fallback if logfunc not built
            self._Jstar = self.v(self._re_grid()).mean().detach()
        rc = (cost_hat - self._Jstar) / self._Jstar
        return {'rc': rc.item(), 'cost_hat': cost_hat.item()}

    # -- RE over the union of uniform 100-point grids on S_0 and S_1 ----------
    def _re_grid(self):
        """The RE test grid: equispaced points on S_0 and S_1
        (num_test_points split evenly).  Deterministic; doubles as the J*
        quadrature (equal point counts = equal mixture weights)."""
        n0 = self.num_test_points // 2
        n1 = self.num_test_points - n0
        x_test = torch.cat([
            utils.t_diag_curve(self.dim_x, n0, self.s_range,
                               randomize=False)[1],
            utils.t_manifold_curve(self.dim_x, n1, self.s_range,
                                   randomize=False)[1],
        ], dim=0)
        if isin_ddp():
            dist.broadcast(x_test, src=0)
        return x_test

    def make_logfunc(self, v_theta, u_alpha=None):
        """RE(v_theta) on the union of equispaced grids on S_0 and S_1
        (num_test_points split evenly; tex 2026-08-21 protocol).  No u*
        reference -> u error columns NaN.  The network is evaluated in fp32
        (autocast disabled) following the sparse-class precedent d9feb7c.
        Also caches J* = mean(v_true) for rc (plan A)."""
        x_test = self._re_grid()
        v_true = self.v(x_test)
        self._Jstar = v_true.mean().detach()
        if getattr(self, 'rc_x0_origin', False):
            # warm the point-caliber J* cache and print both calibers
            # (smoke acceptance gate: the two differ whenever v varies
            # along the curves; owner 2026-09-18, b1origin retrain)
            self._Jstar_pt = self.v(self._initial_points(1)).detach()
            if (not isin_ddp()) or dist.get_rank() == 0:
                print(f"rc_x0_origin startup: J*_point = v(0, z0*1_d) = "
                      f"{self._Jstar_pt.item():.6e}; distributed "
                      f"J* = {self._Jstar.item():.6e}", flush=True)
        v_true_l1 = torch.abs(v_true).mean()
        v_true_linf = torch.abs(v_true).max()

        def log_func(_it):
            mod = getattr(v_theta, 'module', v_theta)      # unwrap DDP
            flag = getattr(mod, 'enable_autocast', None)
            if flag is not None:
                mod.enable_autocast = False
            try:
                with torch.no_grad():
                    err = v_theta(x_test) - v_true
                    abs_err = torch.abs(err)
                    log = {'rel_l1err': (abs_err.mean() / v_true_l1).item()}
                    if self.record_linf_error:
                        log['rel_linferr'] = (
                            abs_err.max() / v_true_linf).item()
                    log['rel_l1err_u'] = float('nan')
                    if self.record_linf_error:
                        log['rel_linferr_u'] = float('nan')
            finally:
                if flag is not None:
                    mod.enable_autocast = flag
            return log

        return log_func


class SOCComplexScalarExact(SOCComplex):
    """Manufactured scalar-control benchmark with the original complex b,s.

    The control direction B(t,z) = (c0/10) s(t,z) 1_d depends on the state,
    unlike the fixed channel in SOCComplexK1.  Let

      F(z) = (0.5/d) sum_i [z_i + 0.2 sin(z_i)
                             + 0.1 sin(z_i + z_{i+1})],
      V(t,z) = 1 + a(t) F(z),  a(t) = 1 + 0.1 (T-t).

    The running cost is kappa^2/2 + h(t,z), where
      h = -V_t - L^0 V + (B . grad V)^2/2.
    Completing the square gives J(kappa) - V(0,z0)
      = E^kappa int_0^T |kappa + B . grad V|^2/2 dt,
    hence u* = -B . grad V.  The generator term is evaluated analytically;
    no reference Monte Carlo or Hessian autograd is used.
    """

    has_optimal_control = True
    has_optimal_cost = True
    terminal_linear_scale = 0.5
    terminal_sin_scale = 0.2
    terminal_pair_scale = 0.1
    channel_row_normalizer = 10.0
    time_slope = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dim_z < 3:
            raise ValueError('SOCComplexScalarExact requires d >= 3')
        self._scalar_geom_key = None
        self._scalar_geom = None
        # first-500-iteration probe handoff).  Defaults = OFF, so every
        # existing INI / example keeps byte-identical training flow.  The
        # whole block is read-only: nothing here enters the loss or the
        # optimizers (see make_logfunc below).
        self.enable_x0_diag = False
        self.x0_diag_cost_gap = 25
        self.x0_diag_cost_seed = 20260924
        self._x0_diag_m_rc = 4096
        self._x0_diag_ref_cost = None

    @property
    def dim_u(self):
        return 1

    @dim_u.setter
    def dim_u(self, _value):
        # SOCProblem.__init__ assigns its default d-dimensional control.
        pass

    def _toeplitz_geometry(self, z):
        """Row sums and covariance kernels for T_ij = q^|i-j|, in O(d)."""
        d = z.shape[-1]
        key = (d, z.device, z.dtype)
        if self._scalar_geom_key != key:
            i = torch.arange(d, dtype=z.dtype, device=z.device)
            q = self.q_decay
            q2 = q * q
            row_sum = 1 + q * (2 - q ** i - q ** (d - 1 - i)) / (1 - q)
            row_sq = 1 + q2 * (2 - q2 ** i - q2 ** (d - 1 - i)) / (1 - q2)
            adjacent = q * (2 - q2 ** (i + 1)
                            - q2 ** (d - 1 - i)) / (1 - q2)
            # The last cyclic pair is (d,1), not a Toeplitz neighbor.
            adjacent[-1] = d * q ** (d - 1)
            self._scalar_geom = (row_sum, row_sq, adjacent)
            self._scalar_geom_key = key
        return self._scalar_geom

    def _F_terms(self, z):
        """F and its diagonal/next-neighbor Hessian entries."""
        d = z.shape[-1]
        nxt = z.roll(-1, dims=-1)
        prv = z.roll(1, dims=-1)
        pair_next = z + nxt
        pair_prev = prv + z
        A = self.terminal_linear_scale
        lam = self.terminal_sin_scale
        beta = self.terminal_pair_scale
        F = A * (z + lam * torch.sin(z)
                 + beta * torch.sin(pair_next)).mean(-1, keepdim=True)
        grad = (A / d) * (1 + lam * torch.cos(z)
                          + beta * torch.cos(pair_next)
                          + beta * torch.cos(pair_prev))
        hdiag = (A / d) * (-lam * torch.sin(z)
                           - beta * torch.sin(pair_next)
                           - beta * torch.sin(pair_prev))
        hoff_next = -(A * beta / d) * torch.sin(pair_next)
        return F, grad, hdiag, hoff_next

    def _a(self, x):
        return 1 + self.time_slope * (self.te - x[..., [0]])

    def _direction_and_mod(self, x):
        z = x[..., 1:]
        row_sum, row_sq, adjacent = self._toeplitz_geometry(z)
        mod = self._mod(x)
        B = ((self.c0 * math.sqrt(2.0) * self.c1
              / self.channel_row_normalizer) * mod * row_sum)
        return B, mod, row_sq, adjacent

    def mu(self, x, kappa):
        B, _, _, _ = self._direction_and_mod(x)
        return self._b(x) + B * kappa[..., :1]

    def mu_pilot(self, x):
        return self._b(x)

    def sigma(self, x, dw):
        return self._s_matvec(x, dw)

    def running_cost(self, acc, x, kappa):
        z = x[..., 1:]
        F, grad, hdiag, hoff_next = self._F_terms(z)
        B, mod, row_sq, adjacent = self._direction_and_mod(x)
        a = self._a(x)
        p = a * (B * grad).sum(-1, keepdim=True)
        covariance_diag = 2 * self.c1 ** 2 * mod.square() * row_sq
        covariance_next = (2 * self.c1 ** 2 * mod
                           * mod.roll(-1, dims=-1) * adjacent)
        trace_cov_hessF = (covariance_diag * hdiag
                           + 2 * covariance_next * hoff_next).sum(
                               -1, keepdim=True)
        generator_F = ((self._b(x) * grad).sum(-1, keepdim=True)
                       + 0.5 * trace_cov_hessF)
        h = self.time_slope * F - a * generator_F + 0.5 * p.square()
        return acc + 0.5 * kappa.square().sum(-1, keepdim=True) + h

    def g(self, z):
        return 1 + self._F_terms(z)[0]

    def v(self, x):
        return 1 + self._a(x) * self._F_terms(x[..., 1:])[0]

    def u_star(self, x):
        grad = self._F_terms(x[..., 1:])[1]
        B = self._direction_and_mod(x)[0]
        return -self._a(x) * (B * grad).sum(-1, keepdim=True)

    def u_star_term(self, x):
        x_term = x.detach().clone()
        x_term[..., 0] = self.te
        return self.u_star(x_term).detach()

    def _x0_diag_reference_cost(self):
        """J_hat(u*) from the origin tile under the SAME forked CRN stream as
        the in-loop diagnostic cost (cached; all ranks participate because
        estimate_cost all-reduces).

        Read-only: does not touch parameters, losses or the training RNG
        stream (the whole estimate runs inside torch.random.fork_rng).
        """
        if self._x0_diag_ref_cost is None:
            dev = torch.empty(0).device
            rng_devs = [dev] if dev.type == 'cuda' else []
            rank = dist.get_rank() if isin_ddp() else 0
            with torch.random.fork_rng(devices=rng_devs):
                torch.manual_seed(self.x0_diag_cost_seed + rank)
                c = self.estimate_cost(self.u_star, self._x0_diag_m_rc)
            self._x0_diag_ref_cost = float(c.item())
        return self._x0_diag_ref_cost

    def make_logfunc(self, v_theta, u_alpha=None):
        """Preserve the S0/S1 value grid and add exact scalar u errors.

        first-500-iteration probe handoff), active ONLY when
        ``enable_x0_diag`` (INI key ``[Diag] x0_point_diag``) is set:

          * every iteration, the four point values and the two relative
            errors at the fixed point (t0, z0 * 1_d) = (0, 0_d), with the
            exact references u*(0,0) != 0 and V(0,0) = 1;
          * at ``it % x0_diag_cost_gap == 0``, the origin cost caliber
            `problem.estimate_cost(u_alpha, _x0_diag_m_rc)` evaluated inside
            ``torch.random.fork_rng`` under the fixed seed
            ``x0_diag_cost_seed + rank``, i.e. the SAME Brownian stream at
            every cadence point (common random numbers) and NO perturbation
            of the training RNG stream.  ``rc_x0_crn =
            cost_hat_x0_crn - V(0,0)`` with V(0,0) = 1 (eval-script caliber,
            not the in-training |J*_pt| normalization).

        All other iterations and every key are NaN-filled so the CSV schema
        stays fixed; nothing here feeds the loss or the optimizer.
        """
        base_log = super().make_logfunc(v_theta, u_alpha)
        x_test = self._re_grid()
        u_true = self.u_star(x_test)
        u_l1 = u_true.abs().mean()
        u_linf = u_true.abs().max()

        # --- origin-point diagnostic setup (inert unless switched on) -------
        diag = bool(u_alpha is not None and getattr(self, 'enable_x0_diag',
                                                    False))
        x0_pt = None
        ustar_x0 = vstar_x0 = None
        if diag:
            dev = getattr(v_theta, 'module', v_theta)
            try:
                pt_device = next(dev.parameters()).device
            except (StopIteration, AttributeError):
                pt_device = x_test.device
            x0_pt = torch.cat(
                (torch.full((1, 1), float(self.t0)),
                 torch.full((1, self.dim_z), float(self.z0))),
                dim=-1).to(pt_device)
            ustar_x0 = self.u_star(x0_pt).reshape(()).double().item()
            vstar_x0 = self.v(x0_pt).reshape(()).double().item()
            # Cache the CRN reference cost once (fork_rng-isolated, rank-aware
            # stream, all ranks participate because it all-reduces).
            self._x0_diag_reference_cost()

        def log_func(it):
            result = base_log(it)
            if diag:
                # -- point values at (t0, z0 * 1_d), no RNG consumed --------
                mod_v = getattr(v_theta, 'module', v_theta)
                mod_u = getattr(u_alpha, 'module', u_alpha)
                flag_v = getattr(mod_v, 'enable_autocast', None)
                flag_u = getattr(mod_u, 'enable_autocast', None)
                with contextlib.ExitStack() as stack:
                    if flag_v is not None:
                        mod_v.enable_autocast = False
                        stack.callback(setattr, mod_v, 'enable_autocast',
                                       flag_v)
                    if flag_u is not None:
                        mod_u.enable_autocast = False
                        stack.callback(setattr, mod_u, 'enable_autocast',
                                       flag_u)
                    with torch.no_grad():
                        v_x0 = v_theta(x0_pt).reshape(()).double().item()
                        u_x0 = u_alpha(x0_pt).reshape(()).double().item()
                result['u_x0'] = u_x0
                result['ustar_x0'] = ustar_x0
                result['v_x0'] = v_x0
                result['vstar_x0'] = vstar_x0
                result['re_u_x0'] = (abs(u_x0 - ustar_x0)
                                     / abs(ustar_x0))
                result['re_v_x0'] = (abs(v_x0 - vstar_x0)
                                     / abs(vstar_x0))
                # -- origin cost under a fork-isolated CRN stream ------------
                if getattr(self, 'x0_diag_cost_gap', 0) > 0 and (
                        it % self.x0_diag_cost_gap == 0):
                    t0c = time.perf_counter()
                    dev = torch.empty(0).device
                    rng_devs = [dev] if dev.type == 'cuda' else []
                    rank = dist.get_rank() if isin_ddp() else 0
                    with torch.random.fork_rng(devices=rng_devs):
                        torch.manual_seed(self.x0_diag_cost_seed + rank)
                        c_hat = self.estimate_cost(u_alpha,
                                                   self._x0_diag_m_rc)
                    if dev.type == 'cuda':
                        torch.cuda.synchronize()
                    self._x0_diag_last_seconds = time.perf_counter() - t0c
                    c_hat = float(c_hat.item())
                    result['cost_hat_x0_crn'] = c_hat
                    # V(0,0) = 1 exactly for this example; keep the literal
                    # eval-script caliber instead of |J*_pt| (identical here,
                    # but explicit and independent of the training rc path).
                    result['rc_x0_crn'] = (c_hat - vstar_x0)
                else:
                    result['cost_hat_x0_crn'] = float('nan')
                    result['rc_x0_crn'] = float('nan')
            if u_alpha is None:
                return result
            mod = getattr(u_alpha, 'module', u_alpha)
            flag = getattr(mod, 'enable_autocast', None)
            if flag is not None:
                mod.enable_autocast = False
            try:
                with torch.no_grad():
                    err = (u_alpha(x_test) - u_true).abs()
                    result['rel_l1err_u'] = (err.mean() / u_l1).item()
                    if self.record_linf_error:
                        result['rel_linferr_u'] = (err.max() / u_linf).item()
            finally:
                if flag is not None:
                    mod.enable_autocast = flag
            return result

        return log_func


class SOCComplexRankKExact(SOCComplexScalarExact):
    """Manufactured benchmark with a d-dimensional control acting through a
    rank-k coherent channel, whose optimal control is concentrated on k slowly
    reports/27_scx_rank_exact/PLAN.md).

    Motivation: with the paper's full-matrix channel and a mean-type value
    function, |u*| = O(1/d) (the Toeplitz kernel has O(1) column mass while
    grad v = O(1/d)), so no component carries a strong signal and the
    strong/weak error split cannot be formed.  Here the control stays in R^d
    (dim_u = d, the paper's output dimension) but its action on the state is a
    rank-k coherent channel:

        mu = b + sum_{m=1..k} ( sum_{j in block_m} rho^{(m)}_j kappa_j )
             * 1_{block_m} * B(t, x),

    with B the parent's scalar channel field, disjoint blocks of size d/k, and
    weights rho^{(m)}_j >= 0 normalized by sum_{j in block_m} (rho^{(m)}_j)^2 = 1.
    Within block m the weights are a Gaussian bump in the component index j,

        rho^{(m)}_j ~ exp( - (j_local - c_m(t, x))^2 / (2 rank_sigma^2) ),

    whose continuous centre sweeps the block,

        c_m(t, x) = (size - 1)/2 * ( 1 + sin( rank_omega t
                    + rank_gamma mean(z) + 2 pi m / k ) ),

    so exactly one component per block is strong and its index moves smoothly
    with t and with the state.  (An earlier design used cos(alpha j + psi) with
    alpha = 2 pi * golden ratio; it FAILED the design gate -- with d components
    the phases are equidistributed, so hundreds of components sit within
    O(1/beta) of the peak and the strong set is not sparse.  A bump of width
    rank_sigma < 1 in j is the only way to concentrate on one component; see
    reports/27_scx_rank_exact/PLAN.md, gate log of 2026-10-01.)

    k_dirs is exposed as a pinned property: solver.parse_problem assigns
    problem.k_dirs from the INI key sparse_k, which belongs to the LQ-SN family;
    here the value is derived from _rank_k and any different assignment raises.

    Manufactured solution (same V and F as the parent):

        V(t, z) = 1 + a(t) F(z),   a(t) = 1 + time_slope (T - t),
        G_m(t, x) = sum_{j in block_m} B_j(t, x) d_j F(z)   (B-weighted block sum;
                    the parent's B is the (N, d) per-coordinate channel field,
                    so the m-th block's coherent direction is B * 1_{block_m}),
        h(t, x) = -V_t - L^0 V + (1/2) a(t)^2 sum_m G_m^2,
        c(t, x, kappa) = |kappa|^2 / 2 + h(t, x),
        u*_j = -a(t) rho^{(m(j))}_j G_{m(j)}(t, x).

    Because sum_{j in m} (rho^{(m)}_j)^2 = 1, the k = 1 instance reproduces the
    parent's effective dynamics, cost and value function exactly (the effective
    control sum_j rho_j kappa_j has unit gain), while its u* is a d-vector
    concentrated on a single moving component.
    """

    _rank_k = 1
    rank_sigma = 0.25
    rank_omega = 1.0
    rank_gamma = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dim_z % self.k_dirs:
            raise ValueError('k_dirs must divide dim_z')
        self._blk_key = None
        self._blk = None

    @property
    def k_dirs(self):
        """Number of coherent blocks; pinned (see the class docstring)."""
        return self._rank_k

    @k_dirs.setter
    def k_dirs(self, value):
        if int(value) != self._rank_k:
            raise ValueError(
                'SOCComplexRankKExact: k_dirs is pinned to %d (got %r); the '
                'sparse_k INI key belongs to the LQ-SN family, do not set it '
                'for this example' % (self._rank_k, value))

    @property
    def dim_u(self):
        return self.dim_z

    @dim_u.setter
    def dim_u(self, _value):
        # SOCProblem.__init__ assigns the default d-dimensional control; the
        # rank-k family keeps dim_u = d (the paper's output dimension).
        pass

    def _blocks(self, z):
        """Block id per control component (int64, length d)."""
        key = (z.shape[-1], z.device)
        if self._blk is None or self._blk_key != key:
            n = key[0]
            self._blk = torch.div(torch.arange(n, device=key[1]) * self.k_dirs,
                                  n, rounding_mode='floor')
            self._blk_key = key
        return self._blk

    def _centre(self, x):
        """Continuous bump centre per block, shape (N, k), in local units
        [0, size - 1]; sweeps the block as t and mean(z) vary."""
        size = self.dim_z // self.k_dirs
        zbar = x[..., 1:].mean(-1, keepdim=True)
        m = torch.arange(self.k_dirs, device=x.device, dtype=x.dtype)
        phase = (self.rank_omega * x[..., [0]] + self.rank_gamma * zbar
                 + (2.0 * math.pi / self.k_dirs) * m.unsqueeze(0))
        return 0.5 * (size - 1.0) * (1.0 + torch.sin(phase))

    def _rho(self, x):
        """Weights rho^{(m(j))}_j -- shape (N, d), per-block 2-norm 1, one
        Gaussian bump of width rank_sigma per block."""
        z = x[..., 1:]
        blk = self._blocks(z)
        size = z.shape[-1] // self.k_dirs
        j = torch.arange(z.shape[-1], device=z.device, dtype=z.dtype)
        local = j - (blk * size).to(z.dtype)
        centre = self._centre(x)[:, blk] - 0.0
        e = torch.exp(-0.5 * ((local - centre) / self.rank_sigma).square())
        sq = torch.zeros(z.shape[0], self.k_dirs, dtype=z.dtype,
                         device=z.device)
        sq.index_add_(1, blk, e.square())
        return e / sq[:, blk].sqrt()

    def _block_sums(self, vec):
        """Segment sums of a (N, d) tensor over the k blocks -> (N, k)."""
        blk = self._blocks(vec)
        out = torch.zeros(vec.shape[0], self.k_dirs, dtype=vec.dtype,
                          device=vec.device)
        out.index_add_(1, blk, vec)
        return out

    def _g_blocks(self, x, grad):
        """G_m = sum_{j in block_m} B_j d_j F -- shape (N, k).

        The parent's B is the (N, d) per-coordinate channel field, so the
        m-th coherent direction is B * 1_{block_m} and the inner product of
        the j-th control column with grad F is rho^{(m(j))}_j G_{m(j)}.
        """
        B, _, _, _ = self._direction_and_mod(x)
        return self._block_sums(B * grad)

    def mu(self, x, kappa):
        B, _, _, _ = self._direction_and_mod(x)
        blk = self._blocks(x[..., 1:])
        lam = self._block_sums(self._rho(x) * kappa)
        return self._b(x) + B * lam[:, blk]

    def mu_pilot(self, x):
        return self._b(x)

    def running_cost(self, acc, x, kappa):
        z = x[..., 1:]
        F, grad, hdiag, hoff_next = self._F_terms(z)
        B, mod, row_sq, adjacent = self._direction_and_mod(x)
        a = self._a(x)
        gt = self._g_blocks(x, grad)
        covariance_diag = 2 * self.c1 ** 2 * mod.square() * row_sq
        covariance_next = (2 * self.c1 ** 2 * mod
                           * mod.roll(-1, dims=-1) * adjacent)
        trace_cov_hessF = (covariance_diag * hdiag
                           + 2 * covariance_next * hoff_next).sum(
                               -1, keepdim=True)
        generator_F = ((self._b(x) * grad).sum(-1, keepdim=True)
                       + 0.5 * trace_cov_hessF)
        h = (self.time_slope * F - a * generator_F
             + 0.5 * a.square() * gt.square().sum(-1, keepdim=True))
        return acc + 0.5 * kappa.square().sum(-1, keepdim=True) + h

    def u_star(self, x):
        grad = self._F_terms(x[..., 1:])[1]
        blk = self._blocks(x[..., 1:])
        gt = self._g_blocks(x, grad)
        return -self._a(x) * gt[:, blk] * self._rho(x)

    def u_star_profile(self, x):
        """(u_star, G, rho) for the zero-checks and the error masks."""
        grad = self._F_terms(x[..., 1:])[1]
        blk = self._blocks(x[..., 1:])
        gt = self._g_blocks(x, grad)
        rho = self._rho(x)
        return -self._a(x) * gt[:, blk] * rho, gt, rho


class SOCComplexRank1Exact(SOCComplexRankKExact):
    """k = 1: one coherent direction (the parent's effective dynamics), with a
    d-dimensional control whose optimal value is concentrated on a single
    component that moves with t and with the state."""

    _rank_k = 1


class SOCComplexRank4Exact(SOCComplexRankKExact):
    """k = 4: four coherent block directions, hence four strong components."""

    _rank_k = 4


class SOCComplexK1(SOCComplex):
    """Section 4.2 complex dynamics with a SINGLE scalar control channel
    (owner 2026-09-05, /goal campaign: u O(1) + dim_u=1 + u influences all
    d coordinates; gain-in-cost form + gamma symbol per owner 2026-09-05):

        dz = [ b(t,z) + q kappa ] dt + s(x) dW,   kappa in R,
        f(kappa) = kappa^2 / (2 gamma^2)          (gamma = chan_gain),
        g(z) = g_SOCComplex(z) + g_chan_amp * sin(q . z)  (NOT divided by d),

    so the channel push is the unit-gain direction q itself and gamma is
    purely the control-cheapness in the cost (eta = gamma kappa_old maps
    back to gain-in-drift gamma / cost kappa^2/2).  Optimal control and
    its terminal size:

        kappa* = -gamma^2 <grad v, q >,
        kappa*_T- = -gamma^2 <grad g, q> = -gamma^2 amp cos(q.z),
        |kappa*_T-| = gamma^2 * amp * |cos|  -- amp dials the harvest
        (sin's full swing 2 amp), gamma^2 dials the effort scale; the
        steering cost of moving q.z by Delta is Delta^2/(2 gamma^2 tau).

    SINGLE low-frequency mode, see _chan_q): q = cos(2 pi k j / d +
    2 pi phi), integer k = 3, one pinned-seed phase -- the gentlest
    single-index structure an MLP can learn (motivation below).
    History (falsified rounds 1-3): the band-limited 23-mode q at
    f in [ceil(d/10), d/6] exploited the Toeplitz sigma's noise-quiet
    band (lambda_f in [0.21, 0.56]); the verification gates PASSED
    (513036: gap 58.1% at drift_amp 1.2 / amp 1.5) but the 5-seed
    training 513077-81 under-learned v (c_pow=4 anchor bias 0.28 vs
    B1 0.006, plateau from it~500, cost_hat 1.654 ~ zero control): a
    1000-dim high-frequency oscillating projection is an unlearnable
    single-index target for a plain MLP (spectral bias) -- the root
    cause of the v plateau.  3-prime trades sigma structure for
    learnability: q at k = 3 (integer mode => q . 1_d = 0 exactly,
    DC-free; no exact zero components; DFT single peak at f = 3),
    sigma de-Toeplitzed (see _s_matvec) so low frequencies are no
    longer noise-loud, c1 lowered 0.5 -> 0.20 (per-coordinate noise
    strength ~4x weaker, owner informed & approved).

    REFERENCE -- rank-1 control breaks the Cole-Hopf matching (the
    control quadratic no longer cancels the diffusion quadratic; the
    residual is quadratic in grad w with no small parameter), so no
    exact reference exists.  Protocol: the comparison full-control
    problem (push s kappa_full, cost |kappa_full|^2/(2 c_hat^2)) IS
    Cole-Hopf solvable for any c_hat -- v_full = -(1/c_hat^2) ln
    E[exp(-c_hat^2 g)] -- and lower-bounds the channel value iff every
    channel push is mimickable at no greater cost: kappa_full = s^-1 q
    kappa costs |s^-1 q|^2 kappa^2/(2 c_hat^2) <= kappa^2/(2 gamma^2)
    iff  c_hat >= gamma |s^-1 q|  (~4 gamma on this band).  The v
    bracket is WIDE at O(1) there (disclosed); the primary certification
    is therefore COST-SIDE (J(u_alpha) vs J(kappa=0) and the best known
    admissible policy -- the E3 methodology), v reported vs v_full with
    the lower-bound direction disclosed.  check_complex_k1.py v7 gates:
    wiring (q single-peak DFT, DC-free, sigma diagonal, variance
    theory), FD of kappa*_T-, |kappa*_T| > 0.1, TOTAL-cost channel gain
    >= 50% (best of zero / greedy / const-kappa), MC floor; y-noise std
    measured and archived.

    INI: [Example] g_chan_amp (terminal amplitude, default 0.0), chan_gain
    (gamma, default 1.0), phi_decay (Phi-kernel decay, default 0.9 -- see
    below), drift_amp (drift-feedback amplitude, default 1.2 -- see
    below), c1 (diffusion strength, default 0.20 -- see the 3-prime
    paragraph below) and chan_dc (DC weight of q, default 0.0 = the
    2026-09-06); all hasattr-guarded in parse_problem, zero
    behavior change when unset.

    drift-feedback convolution kernel Phi_i = sum_k phi_decay^k z_{i+k}
    to 0.5 and run 513033 FALSIFIED that direction -- the feedback
    variance went 0.16 -> 2.46 (x15, coherent amplification: with the
    short kernel Phi = O(1) stays smooth and the cos envelope of
    sin(t + i + Phi) stays coherent across paths), y std 1.59, gap
    9-11%.  The 0.9 long kernel's protection is the sin-saturation
    DE-COHERENCE (the deterministic smooth drift contour is amplified to
    Phi_bar = O(10+), the sin argument winds fast, the cos response
    quasi-randomizes and self-cancels), NOT the kernel energy -- so the
    kernel is BACK to 0.9 (== the base q_decay kernel bitwise).  The
    kernel) is UNTOUCHED: B1/B2 and every other SOCComplex user keep
    byte-identical behavior; phi_decay = 0.9 reproduces the base kernel
    exactly (bitwise, smoke-checked), which stays the equivalence probe
    of the smoke suite.

    down the drift-feedback amplitude b_i = drift_amp * sin(t + i0 +
    Phi_i(z)).  Why: under the phi_decay = 0.9 de-coherence mechanism the
    feedback variance scales ~ drift_amp^2 (the sin amplitude), band-
    independent; run 513026 (phi_decay 0.9, drift_amp 2.0) measured
    zero-control y std 0.488 -> total variance 0.238 = sigma-side closed
    form 0.279^2 = 0.078 (pinned q, frozen m) + feedback ~0.16.  The W3
    budget total variance <= 0.4^2 = 0.16 leaves feedback <= 0.08 ->
    {1.2, 1.0}: predicted feedback 0.058/0.040 -> zero-control std
    ~0.37/0.34 (gate 0.4), rho_0 ~ 0.93/0.94.  The BASE SOCComplex
    drift_amp = 2.0 (B' freeze) is UNTOUCHED: _b reads self.drift_amp so
    B1/B2 keep byte-identical behavior.

    DE-TOEPLITZED for THIS subclass only -- sigma(x) w = sqrt(2) c1 *
    (m(x) * w), no _toep convolution (see _s_matvec) -- and the K1
    _mod lowf override is DELETED so m reverts to the BASE
    state-dependent modulation m_i = 1 + 0.5 cos(t + i + z_{i+1
    cyclic}) (under a diagonal sigma the state dependence is harmless;
    sigma's identity as a state-dependent diffusion is restored).
    Together with the single low-frequency q (see _chan_q) and the c1
    override this replaces the cusp's high-frequency multi-mode
    single-index structure by a low-frequency single-index one that an
    anchor bias 0.28 vs B1 0.006).  CAVEAT (owner informed): the
    WEAKENS the Phi de-coherence (Phi_bar stays smaller, sin(t + i +
    Phi) less wound) -- the drift-feedback variance may exceed the
    drift_amp^2 prediction; the v7 variance decomposition measures
    this.  The BASE SOCComplex is UNTOUCHED: B1/B2 and every other
    SOCComplex user keep the Toeplitz sigma, c1 = 0.5 and the
    state-dependent m byte-identical.
    """

    g_chan_amp = 0.0
    chan_gain = 1.0
    phi_decay = 0.9
    drift_amp = 1.2
    # sigma = sqrt(2) c1 diag(m(x)) for THIS subclass (Toeplitz removed,
    # see _s_matvec); the BASE SOCComplex keeps c1 = 0.5 with its
    # the INI key c1 (hasattr-guarded in parse_problem; no existing INI
    # sets the key, so the base family is byte-identical).
    c1 = 0.20
    # q = (chan_dc * 1_d + cos(2 pi k j / d + phase)) / ||.||.  Default
    # 0.0 = the 3-prime q bitwise (the `if` in _chan_q keeps the addition
    # out of the tensor entirely, so existing behavior is unchanged).
    # projection y0 = q . z0 is ~the SAME number for every X0 on the
    # declared S0/S1 surface (on the diagonal curve exactly 0), so the
    # optimal push is state-independent and a single constant kappa
    # captures ~the whole gap (measured: best const -0.074 vs learned
    # -0.026 on the legal surface; the const line IS near-optimal).  A
    # DC component coherently sums on the diagonal curve (z = s 1_d ->
    # y0 = s * c_d with c_d = chan_dc*sqrt(d)/sqrt(chan_dc^2 + 1/2)),
    # spreading the initial projection so different X0 need different
    # pushes -> feedback becomes necessary.  Target half-spread c on S0:
    # chan_dc = c / sqrt(2 (d - c^2)); e.g. c = 2.0 at d = 1000 ->
    # chan_dc = 0.0448.
    chan_dc = 0.0
    # L2 knobs (owner-proposed 2026-09-06, single-coordinate terminal
    # reward; the L1 postmortem motivated it: the DC spread only moves
    # the S0 half -- S1 stayed inert-to-adverse, killing the surface
    # feedback-value gate at 0.15 vs 0.30, yancheng 1736773).
    #   chan_coord (int, default -1 = OFF = the projection reward
    #     amp*sin(q.z)): when >= 0 the channel terminal term becomes
    #     g_chan_amp * sin(z[..., chan_coord]) -- the reward coordinate
    #     z_i whose INITIAL value spreads NATURALLY on BOTH curves
    #     (S0: z_i(0) = s; S1: z_i(0) = s*sign(sin(i+1)) + cos(i+1+s*pi),
    #     both O(1)), so no DC trick and no dead half.  Network readout
    #     burden drops from a d-dim inner product to one input
    #     coordinate.
    #   chan_spike (float, default 0.0 = OFF = pure cosine mode
    #     bitwise): weight w of the e_i spike in q -- a reward on z_i is
    #     only harvestable if the control can STEER z_i; the pure
    #     cosine mode has |q_i| ~ 1/sqrt(d) (hopeless).  q = (w*sqrt
    #     (d/2)*e_i + cos3)/||.||; steering authority alpha = q_i
    #     (0.707 at w=1, 0.894 at w=2); w>0 requires chan_coord >= 0;
    #     chan_spike and chan_dc are mutually exclusive (ValueError).
    #     The cosine part keeps the control touching ALL coordinates
    #     (the K1 design premise), only its per-coordinate amplitude
    #     shrinks by the norm split.
    chan_coord = -1
    chan_spike = 0.0

    def _phi(self, z):
        """Same exact circular correlation Phi_i = sum_k q_p^k z_{i+k} as
        SOCComplex._phi but with THIS class's kernel base q_p =
        self.phi_decay (owner /goal 2026-09-05; see the phi_decay class
        attribute for the rounds-2/3 rationale).  Verbatim copy of the
        base implementation with the kernel base swapped; the cache key
        additionally carries phi_decay so a per-instance override (INI
        injection) can never reuse a stale kernel.  phi_decay = 0.9 is
        bit-identical to the base q_decay = 0.9 kernel."""
        d = self.dim_z
        key = (d, z.device, z.dtype, float(self.phi_decay))
        if getattr(self, '_phi_key', None) != key:
            ker = self.phi_decay ** torch.arange(1, d + 1, dtype=z.dtype,
                                                 device=z.device)
            ker = torch.cat([ker[-1:], ker[:-1]])  # [q^d, q^1, ..., q^{d-1}]
            self._phi_kernel = torch.conj(torch.fft.fft(ker))
            self._phi_key = key
        return torch.fft.ifft(self._phi_kernel * torch.fft.fft(z)).real

    def _s_matvec(self, x, w):
        """sigma(x) @ w = sqrt(2) * c1 * (m(x) * w) -- DIAGONAL diffusion
        convolution T is REMOVED for THIS subclass only, so sigma =
        sqrt(2) c1 diag(m(x)) with m the BASE state-dependent modulation
        (the K1 lowf _mod override is deleted; under a diagonal sigma the
        state dependence is harmless and sigma's identity is restored).
        58.1%) but the 5-seed training 513077-81 under-learned v (anchor
        bias 0.28 vs B1 0.006, cost_hat ~ zero control) -- with the
        single low-frequency q of 3-prime the Toeplitz band protection is
        moot (low frequencies were noise-loud ONLY through T; without it
        every direction has the same rate 2 c1^2 m_i^2), and dropping T
        makes the noise a per-coordinate independent driver.  Trade-off
        (owner informed & approved): sigma loses coordinate coupling and
        the per-coordinate noise strength drops ~4x (c1 0.5 -> 0.20).
        The BASE SOCComplex._s_matvec (sqrt(2) c1 m * (T w), c1 = 0.5) is
        UNTOUCHED: B1/B2 keep byte-identical behavior."""
        return math.sqrt(2.0) * self.c1 * (self._mod(x) * w)

    def _chan_q(self, dtype, device):
        """Unit SINGLE low-frequency cosine profile (scheme 3-prime,
        normalized, with INTEGER mode k = 3 and ONE random phase phi
        drawn from the pinned CPU Generator (seed 20260905).  Why single
        low-frequency: rounds 1-3 used a 23-mode band-limited q at
        f in [100, 166]; the gates passed (513036) but the 5-seed
        training 513077-81 under-learned v (anchor bias 0.28 vs B1
        0.006) -- a 1000-dim high-frequency oscillating projection is an
        unlearnable single-index target for a plain MLP (spectral
        bias).  A single k = 3 mode is the gentlest learnable variant;
        with the DIAGONAL sigma of 3-prime there is no Toeplitz band to
        respect, so low frequencies are admissible.  Integer mode =>
        q . 1_d = 0 exactly (DC-free, discrete sum over full periods);
        DFT single peak at f = 3; cos never vanishes on all j so no
        a DC weight chan_dc (default 0.0 = the 3-prime q bitwise)
        spreads the initial projection q.z0 so the optimal push becomes
        state-dependent -- see the chan_dc class attribute.  L2 addend
        (owner-proposed 2026-09-06): a chan_spike-weighted one-hot on
        coordinate chan_coord (default 0.0 = OFF) so the control can
        steer the single-coordinate reward -- see the chan_coord class
        attribute.  Cached per (d, chan_dc, chan_coord, chan_spike,
        device, dtype) -- every knob is IN the key so mutating any of
        them on a live instance cannot serve a stale q."""
        d = self.dim_z
        dc = float(getattr(self, 'chan_dc', 0.0))
        coord = int(getattr(self, 'chan_coord', -1))
        spike = float(getattr(self, 'chan_spike', 0.0))
        if spike != 0.0 and (coord < 0 or dc != 0.0):
            raise ValueError(
                f"chan_spike={spike!r} requires chan_coord >= 0 and "
                f"chan_dc == 0 (spike+DC and spike-without-coordinate are "
                f"unsupported by design); got chan_coord={coord}, "
                f"chan_dc={dc!r}.")
        key = (d, dc, coord, spike, str(device), dtype)
        if getattr(self, '_chan_q_key', None) != key:
            # 2026-09-05; MUST NOT regress): all accumulator allocations
            # below are pinned device='cpu' and the phase generator is a
            # CPU generator -- ambient-device invariant (runtask.py calls
            # torch.set_default_device(cuda) BEFORE solve(), which
            # previously put arange/zeros/rand on CUDA against the CPU
            # generator and crashed training 513068-513072); only the
            # final result moves to `device`.
            j = torch.arange(d, dtype=torch.float64, device='cpu')
            gen = torch.Generator(device='cpu').manual_seed(20260905)
            # MUST NOT regress): torch.rand without an explicit dtype
            # follows the AMBIENT default, so fp64 zero-checks and fp32
            # training would draw DIFFERENT phases -> different q.
            # Pinning float64 keeps q ambient-dtype-invariant: the fp64
            # check runs and the fp32/bf16 training runs all draw the
            # exact same q (the q VALUES differ from 513036 by design --
            # the profile itself changed in 3-prime).
            phase = float(torch.rand(1, generator=gen, dtype=torch.float64,
                                     device='cpu'))
            q = torch.cos(2 * math.pi * 3 * j / d + 2 * math.pi * phase)
            # default chan_dc = 0.0 keeps the 3-prime q BITWISE (no
            # tensor addition at all); see the chan_dc class attribute
            # for the motivation (spread the initial projection so the
            # optimal push is state-dependent).
            if dc != 0.0:
                q = q + dc
            # L2 spike addend (owner-proposed 2026-09-06): w*sqrt(d/2)
            # on the reward coordinate so the control can steer it
            # (alpha = q_i; guarded the same way -- 0.0 = bitwise off).
            if spike != 0.0:
                q[coord] = q[coord] + spike * math.sqrt(d / 2.0)
            q = q / q.norm()
            self._chan_q_vec = q.to(device=device, dtype=dtype)
            self._chan_q_key = key
        return self._chan_q_vec

    @property
    def dim_u(self):
        """Control dimension = 1 (scalar channel)."""
        return 1

    @dim_u.setter
    def dim_u(self, _value):
        """Absorb SOCProblem.__init__'s self.dim_u = dim_x - 1."""
        pass

    def mu(self, x, kappa):
        """b + kappa q with scalar kappa (..., 1): the channel push is the
        unit-gain direction q itself (gamma lives in the cost).  An
        all-zero kappa of ANY width (the path-MC reference passes d-dim
        zeros) -> b."""
        b = self._b(x)
        if not torch.any(kappa):
            return b
        q = self._chan_q(b.dtype, b.device)
        return b + kappa[..., :1] * q

    def mu_pilot(self, x):
        """Pilot drift = uncontrolled b (kappa = 0 scalar)."""
        return self._b(x)

    def running_cost(self, acc, x, kappa):
        """f = kappa^2 / (2 gamma^2) -- the channel gain lives here (owner
        2026-09-05), not in the drift."""
        return acc + 0.5 * (kappa / self.chan_gain).pow(2).sum(
            -1, keepdims=True)

    def g(self, z):
        out = super().g(z)
        if self.g_chan_amp != 0.0:
            if int(getattr(self, 'chan_coord', -1)) >= 0:
                # L2 (owner-proposed 2026-09-06): the channel reward on
                # the single coordinate z_i (see chan_coord).
                out = out + self.g_chan_amp * torch.sin(
                    z[..., [int(self.chan_coord)]])
            else:
                q = self._chan_q(z.dtype, z.device)
                out = out + self.g_chan_amp * torch.sin(z @ q).unsqueeze(-1)
        return out

    def u_star_term(self, x):
        """Scalar TERMINAL control kappa_T = -gamma^2 <grad g, q> (grad g
        by autograd on a detached clone; valid for every g knob)."""
        z = x[..., 1:].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            g = self.g(z)
            grad_g = torch.autograd.grad(g.sum(), z)[0]
        q = self._chan_q(grad_g.dtype, grad_g.device)
        return (-self.chan_gain ** 2
                * (grad_g * q).sum(-1, keepdim=True)).detach()

    def _toep_inv_apply(self, v):
        """T^{-1} v for the KMS Toeplitz T = (q^|i-j|) via the exact
        tridiagonal inverse (no dense matrix, O(d)): interior rows
        (-q, 1+q^2, -q)/(1-q^2), corners (1, -q) -- verified on KMS
        small cases; float64 CPU."""
        q_ = self.q_decay
        den = 1.0 - q_ ** 2
        y = torch.zeros_like(v)
        y[..., 0] = v[..., 0] - q_ * v[..., 1]
        y[..., 1:-1] = (-q_ * v[..., :-2] + (1 + q_ ** 2) * v[..., 1:-1]
                        - q_ * v[..., 2:])
        y[..., -1] = -q_ * v[..., -2] + v[..., -1]
        return y / den

    def _comp_c_hat(self):
        """Comparison-problem gain c_hat >= gamma |s^{-1} q| (the mimic
        condition for v_full to lower-bound the channel value; m(x) >=
        mod_amp-complement 0.5 gives |s^{-1} q| <= |T^{-1} q| / (sqrt2 c1
        * 0.5), +10% safety).  Cached; O(d) per construction.

        NUMERICAL DISCLOSURE (2026-09-05): at c_hat ~ 10 gamma the MC
        average E[exp(-c_hat^2 g)] UNDERFLOWS in float64 (c_hat^2 * g ~
        400 * 2 > 709) and is statistically dominated by a single
        extreme path -- the calibrated lower bound is not only loose but
        NOT COMPUTABLE by plain MC.  v() therefore stays at the parent's
        c_pow = 4 anchor (the value of the RELATED unit-cost full-control
        problem) -- a smoothness/sanity anchor, NOT a bound once
        c_hat > 2; certification is cost-side (see class docstring).
        STALE under scheme 3-prime (2026-09-06): sigma is diagonal here,
        so |s^-1 q| = |q/m| / (sqrt2 c1) <= 1/(sqrt2 c1 min m); the
        """
        if getattr(self, '_c_hat', None) is None:
            q = self._chan_q(torch.float64, torch.device('cpu'))
            n_inv = float(self._toep_inv_apply(q).norm())
            self._c_hat = (self.chan_gain * n_inv
                           / (math.sqrt(2.0) * self.c1 * 0.5) * 1.1)
        return self._c_hat


class SOCComplexK1S0(SOCComplexK1):
    """L2 variant with X_0 concentrated on S0 (owner directive 2026-09-14):
    drop the S1 (manifold) component of the initial distribution.  The
    x0_curves dict keeps ONLY the diagonal curve, which makes every
    X0-distribution consumer S0-only in one move -- training pilot start,
    the logged rel_l1err test surface, the rc/cost_hat CRN initial points
    (SOCComplex._initial_points routes through x0_points under seed
    20260821), and the resonline profile set.  Performance readout per
    owner: v error (rel_l1err, now the S0 surface) AND cost (rc /
    cost_hat) reported together, single seed, anchors = the mixed-surface
    sr1 arm 523331.  Pattern precedent: the CoupledDriftS0 S0-only
    subclasses.  Everything else inherited bitwise from SOCComplexK1."""

    x0_curves = {'diag': utils.t_diag_curve}


class SOCChainGate(SOCProblem):
    """Gated long-range chain (owner-designed 2026-09-06/07, "L3"):

        dx_1 = u dt + s dw_1,
        dx_i = cG(t, x) * (sum_{k>=1} w_k sat(x_{i-k}) - W_i sat(x_i)) dt
               + s dw_i,           i = 2..d,
        w_k  = k^{-(1+alpha)} (power-law, long-range; owner: smooth the
               interaction so x_i influences x_{i+1}, x_{i+2}, ... with
               decaying strength and dt = 1/32 stays stable),
        sat(z) = tanh(z/sat_width) * sat_width (wide-linear saturating
               nonlinearity; sat_width is the d=1e4-tier necessity --
               narrow saturation caps the tail response at the noise
               level, kernel_check10k/kernel_sat_check),
        G(t, x) = g0 + g1 cos(omega t + kap_g x_m)  in [g0-g1, g0+g1]
               (gate: state- and time-dependent, NOT u-dependent;
               m = gate_coord, default d//2),
        f = u^2 / (2 gamma^2),   g(x) = -A exp(-(x_d - x*)^2 / (2 w^2)).

    WHY (owner spec): u acts ONLY on x_1; the terminal cost ONLY on x_d;
    the gate modulates the whole chain's coupling strength, so the
    transmission of a push to the tail varies per path -- a constant
    pays for pushes during closed-gate phases, a feedback policy that
    reads x_m times its pushes into open-gate windows.  Stability: the
    difference-form coupling with a nonnegative kernel makes the
    explicit-Euler row sums exactly 1 (a stochastic-matrix update,
    unconditionally stable at ANY d); the accuracy budget is
    cG * W_row * dt ~<= 1 with W_row = sum w_k.  Transmission decays
    POLYNOMIALLY in d (linear-algebra prevalidation 2026-09-07:
    response to a unit held push at budget 0.75: 13.2% @d=1e3,
    1.1% @d=1e4 for alpha=0.2; with sat_width=20 and push ~20 the
    d=1e4 tail response is 0.15 = 3x the 0.05 noise floor).

    IMPLEMENTATION: the kernel sum is a causal Toeplitz matvec via FFT
    (same machinery as SOCComplex._toep); kernel/prefix tensors are
    CPU-built and cache-keyed by (d, alpha, device, dtype) -- the
    device-pin discipline of 60db9b3.  NO reference solution (no
    Cole-Hopf for this structure) and NO terminal closed form (grad g
    has no x_1 component, so u_anchor_pow is meaningless here):
    certification is cost-side (check_chain_gate.py: zero / const-u
    scan / gate-reading heuristic feedback families).  TRAINING-SIDE
    NOTE: rel_l1err/rc reference logging is NOT wired yet (has_
    optimal_cost = False; decide the diagnostic at the training round).

    INI (hasattr-guarded in parse_problem): chain_alpha, chain_sat,
    chain_cG, gate_g0, gate_g1, gate_omega, gate_kappa, chain_gate_coord,
    c1, chan_gain (gamma), win_amp, win_x, win_w."""

    # kernel / nonlinearity
    chain_alpha = 0.2
    chain_sat = 20.0
    chain_cG = 5.0
    # gate
    gate_g0 = 1.0
    gate_g1 = 0.6
    gate_omega = 2.0 * math.pi
    gate_kappa = 4.0
    chain_gate_coord = -1        # <0 -> d // 2
    # diffusion (constant diagonal, sqrt(2) c1)
    c1 = 0.25
    # control cheapness (f = u^2 / (2 gamma^2))
    chan_gain = 30.0
    # terminal window reward on x_d
    win_amp = 1.5
    win_x = 1.5
    win_w = 0.6
    has_optimal_cost = False
    has_optimal_control = False
    x0_curves = {'diag': utils.t_diag_curve,
                 'manifold': utils.t_manifold_curve}

    def __init__(self, dim_x, dt=1 / 32, batch_size=1024,
                 num_pilot_paths=10000):
        super().__init__(dim_x, dt=dt, batch_size=batch_size,
                         num_pilot_paths=num_pilot_paths)
        self._x0_rc = None

    @property
    def gate_m(self):
        m = int(getattr(self, 'chain_gate_coord', -1))
        return self.dim_z // 2 if m < 0 else min(m, self.dim_z - 1)

    # -- X_0 mixture protocol (S0/S1 uniform, owner ruling 2026-09-06) -------
    def _initial_points(self, num_paths):
        """X_0 ~ 50/50 mixture of the two curves with s ~ U(-1, 1); cached
        under a fixed seed (CRN; copied from SOCComplex's protocol)."""
        if self._x0_rc is None or self._x0_rc.shape[0] != num_paths:
            dev = torch.empty(0).device
            rng_devs = [dev] if dev.type == 'cuda' else []
            with torch.random.fork_rng(devices=rng_devs):
                torch.manual_seed(20260821)
                pts = self.x0_points(num_paths)
            self._x0_rc = pts
        return self._x0_rc

    # -- cached kernel (CPU-built, device/dtype-pinned) -----------------------
    def _chain_cache(self, dtype, device):
        d = self.dim_z
        key = (d, self.chain_alpha, str(device), dtype)
        if getattr(self, '_chain_key', None) != key:
            k = torch.arange(1, d, dtype=torch.float64, device='cpu')
            w = k ** (-(1.0 + self.chain_alpha))
            # causal convolution kernel [0, w_1, ..., w_{d-1}], FFT length
            # n >= 2d (zero padding so the circular wrap misses nothing)
            n = 1
            while n < 2 * d:
                n *= 2
            kpad = torch.zeros(n, dtype=torch.float64, device='cpu')
            kpad[1:d] = w
            # prefix W_i = sum_{k<=i-1} w_k (i = 1..d, 0-based -> shift)
            pref = torch.zeros(d, dtype=torch.float64, device='cpu')
            pref[1:] = torch.cumsum(w, 0)[:d - 1]
            self._chain_kern = torch.fft.fft(kpad)
            self._chain_w = w
            self._chain_pref = pref
            self._chain_n = n
            self._chain_key = key
        return self._chain_n

    def _chain_inflow(self, sz, n):
        """inflow_i = sum_{k>=1} w_k sz_{i-k} via padded FFT (causal).
        The kernel stays COMPLEX (casting it to a real dtype would
        discard the spectrum -- silent-wrong, caught in self-review);
        only the final real part is cast to sz.dtype."""
        zp = torch.zeros(sz.shape[:-1] + (n,), dtype=sz.dtype,
                         device=sz.device)
        zp[..., :sz.shape[-1]] = sz
        prod = torch.fft.fft(zp, dim=-1) * self._chain_kern.to(
            device=sz.device)
        return torch.fft.ifft(prod, dim=-1).real[
            ..., :sz.shape[-1]].to(sz.dtype)

    def _gate(self, x):
        t, z = x[..., :1], x[..., 1:]
        return self.gate_g0 + self.gate_g1 * torch.cos(
            self.gate_omega * t + self.gate_kappa * z[..., self.gate_m:
                                                       self.gate_m + 1])

    def mu(self, x, kappa):
        """Chain drift + the scalar u on coordinate 1 ONLY (mu - mu(0)
        == kappa * e_1 exactly; zero-kappa paths skip nothing here -- the
        chain drift is the same object for pilot and system)."""
        n = self._chain_cache(x.dtype, x.device)
        z = x[..., 1:]
        sz = torch.tanh(z / self.chain_sat) * self.chain_sat
        inflow = self._chain_inflow(sz, n)
        drift = (self.chain_cG * self._gate(x)
                 * (inflow - self._chain_pref.to(
                     dtype=z.dtype, device=z.device) * sz))
        out = drift.clone()
        if torch.any(kappa):
            out[..., 0:1] = out[..., 0:1] + kappa[..., :1]
        return out

    def mu_pilot(self, x):
        return self.mu(x, torch.zeros_like(x[..., :1]))

    def _s_matvec(self, x, w):
        """Constant diagonal sigma: sqrt(2) c1 w (no state dependence)."""
        return math.sqrt(2.0) * self.c1 * w

    def sigma(self, x, dw):
        return self._s_matvec(x, dw)

    def running_cost(self, acc, x, kappa):
        return acc + 0.5 * (kappa / self.chan_gain).pow(2).sum(
            -1, keepdims=True)

    def g(self, z):
        """-A exp(-(x_d - x*)^2 / (2 w^2)): window reward on the LAST
        coordinate only."""
        xd = z[..., -1:]
        return -self.win_amp * torch.exp(
            -(xd - self.win_x) ** 2 / (2.0 * self.win_w ** 2))


# ---------------------------------------------------------------------------
# Sparse-control benchmarks with ring-coupled drift (note_lqsparse.tex)
#
# Shared setting (Section 4.1 normalization): mu = b(t, z) + 2*kappa,
# sigma = sqrt(2)*delta, f = delta^{-2} |kappa|^2, and a sparse terminal
#     g(z) = ln( (1 + sum_j c_j s_j(z)^2) / 2 ),
#     s_j(z) = (z_{2j-2} + z_{2j-1}) / sqrt(2)   (0-based pairs, j = 0..k-1),
# so that u* = -delta^2 grad v is sparse with O(1) active components.
# Variant (L) HJBSparseRing: linear ring drift b_i = omega*(z_{i+1} - z_i);
# the uncontrolled terminal law stays Gaussian and the reference v/u* are
# k-dim Gaussian quadratures propagated by the Poisson kernel e^{rL} on the
# index ring (machine precision, no MC noise floor).
# Variant (N) HJBSparseSin: the Section 4.2 sine drift
# b_i = sin(t + i + z_{i+1}); the reference is path Monte Carlo on a finer
# grid (dt_ref), with u* from the pathwise tangent rows of the flow
# Jacobian restricted to the 2k active terminal directions (windowed
# recurrence, O(num_mc * k * N_ref) extra memory instead of O(d^2)).
# ---------------------------------------------------------------------------
class HJBSparseBase(SOCProblem):
    """Shared data of the sparse-control benchmarks (note_lqsparse.tex)."""

    delta = 1.0
    k_dirs = 5
    c_weights = (1.0, 1.5, 2.0, 2.5, 3.0)
    # reference quadrature mode for _RingQuadrature (variant L): 'gh' =
    # tensor Gauss-Hermite (historical default, n_gh = 13; biased ~1.5% at
    # omega=0 / amplified c per diag 482491), 'qmc' = scrambled Sobol at
    # 2^sparse_qmc_exp points (~1e-5 on this integrand; owner 2026-08-27).
    sparse_ref = 'gh'
    sparse_qmc_exp = 19
    has_optimal_control = True
    x0_curves = {'diag': utils.t_diag_curve, 'manifold': utils.t_manifold_curve}

    def _b(self, x):
        """Uncontrolled drift b(t, z) (variant-specific)."""
        raise NotImplementedError

    def mu(self, x, kappa):
        return self._b(x) + 2 * kappa

    def sigma(self, x, dw):
        return (2**0.5) * self.delta * dw

    def running_cost(self, acc, x, kappa):
        return acc + self.delta**(-2) * kappa.pow(2).sum(-1, keepdims=True)

    def _sens(self, z):
        """Sensitive coordinates s_j = (z_{2j-2} + z_{2j-1})/sqrt(2): (..., k)."""
        zp = z[..., :2 * self.k_dirs].unflatten(-1, (self.k_dirs, 2))
        return zp.sum(-1) / math.sqrt(2.)

    def g(self, z):
        assert len(self.c_weights) >= self.k_dirs, \
            f"c_weights (len {len(self.c_weights)}) shorter than k_dirs " \
            f"({self.k_dirs})"
        s = self._sens(z)
        c = torch.as_tensor(self.c_weights[:self.k_dirs], dtype=z.dtype,
                            device=z.device)
        return torch.log(0.5 * (1 + (c * s.pow(2)).sum(-1, keepdim=True)))

    def grad_g(self, z):
        """grad g: entry 2*c_j*s_j/(sqrt(2)*D) at both coordinates of pair j,
        D = 1 + sum_j c_j s_j^2; zero elsewhere."""
        s = self._sens(z)
        c = torch.as_tensor(self.c_weights[:self.k_dirs], dtype=z.dtype,
                            device=z.device)
        denom = 1 + (c * s.pow(2)).sum(-1, keepdim=True)
        a = 2 * c * s / denom / math.sqrt(2.)          # (..., k)
        grad = torch.zeros_like(z)
        pairs = grad[..., :2 * self.k_dirs].unflatten(-1, (self.k_dirs, 2))
        pairs.copy_(a.unsqueeze(-1).expand_as(pairs))
        return grad

    # -- spread evaluation point set --------------------------------------
    # E5-motivated protocol: errors are measured on points spread over the
    # state distribution (all time levels), not only on the t = 0
    # diagonal/manifold curves (which lie outside the pilot training
    # distribution: paths start at a single point, so t ~ 0 states never
    # exhibit the all-coordinates-equal correlation of the diagonal curve).
    #
    # spread_control (owner decision 2026-08-20): the spread points are
    # generated by the CLOSED-LOOP dynamics under the reference optimal
    # control u* (drift mu(x, u*) = b + 2 u*), i.e. the distribution of the
    # states at which u and v are actually used. In high dimensions the
    # uncontrolled law is in general nearly singular to the controlled one
    # (KL = (1/2) * integral |u*|^2 dt), so evaluating on the uncontrolled
    # process does not reflect in-use performance unless the control energy
    # is O(1). Using u* (not u_alpha) keeps the set solution-independent,
    # fixed-seed reproducible, and comparable across runs. Set False to
    # recover the uncontrolled spread set (runs 477259-477261; the
    # 477605/477606 reruns were cancelled before start).
    eval_spread = True
    spread_control = True

    # u_alpha-closed-loop (in-use) evaluation columns (owner decision
    # 2026-08-20): at cadence ualoop_loggap (mirror of the INI rc_loggap),
    # regenerate a spread set with the CURRENT learned control u_alpha in
    # the drift (mu = b + 2 u_alpha), evaluate v/u against the reference on
    # those points.  This is the deployment distribution: in use, the
    # networks are queried at states induced by u_alpha itself.  The noise
    # uses a dedicated fixed-seed generator (common random numbers across
    # evaluations, so the point set drifts only through u_alpha); the
    # reference evaluation is wrapped in fork_rng so variant N's path-MC
    # does not perturb the training RNG stream.  Off-cadence rows get NaN.
    eval_ualoop = True
    ualoop_loggap = 500
    ualoop_seed = 54321

    def spread_points(self, num_points):
        """Subsampled states of the optimally controlled dynamics over t < T.

        Simulates ceil(num_points / n_levels) paths from the initial point
        with the class dt, drift mu(x, u_star(x)) when spread_control is set
        (u_star evaluated level by level: all paths share one time level per
        step, matching the single-time-level requirement of the reference
        solvers), otherwise the uncontrolled drift _b; keeps every time
        level up to (but excluding) T, then trims to num_points. Simulated
        on rank 0 with a dedicated generator (fixed seed) for the noise,
        then broadcast under DDP. The u_star calls happen once at startup
        (inside make_logfunc), before training; variant N's path-MC
        reference consumes the default RNG there, exactly as the precomputed
        u_true on x_test already does.
        """
        n_levels = max(1, round(float(self.te - self.t0) / self.dt))
        num_paths = -(-num_points // n_levels)
        if self.spread_control:
            # u_star's path-MC reference (variant N) contains DDP
            # collectives, so every rank must enter the call: simulate
            # redundantly on ALL ranks with the same seeded generator
            # (initial points are deterministic tiles and u_star is
            # all-reduced, hence the points are bitwise identical across
            # ranks) instead of rank-0-only + broadcast.  A rank-0-only
            # u_star call deadlocks/mismatches against the broadcast.
            gen = torch.Generator(device=self._initial_points(1).device)
            gen.manual_seed(12345)
            x = self._initial_points(num_paths)
            collected = [x]
            for _ in range(n_levels - 1):
                t, z = x[..., [0]], x[..., 1:]
                dw = torch.normal(0., math.sqrt(self.dt), z.shape,
                                  generator=gen, device=z.device)
                with torch.no_grad():
                    drift = self.mu(x, self.u_star(x))
                x = torch.cat([t + self.dt,
                               z + drift * self.dt + self.sigma(x, dw)],
                              dim=-1)
                collected.append(x)
            pts = torch.cat(collected, dim=0)[:num_points]
        elif (not isin_ddp()) or dist.get_rank() == 0:
            gen = torch.Generator(device=self._initial_points(1).device)
            gen.manual_seed(12345)
            x = self._initial_points(num_paths)
            collected = [x]
            for _ in range(n_levels - 1):
                t, z = x[..., [0]], x[..., 1:]
                dw = torch.normal(0., math.sqrt(self.dt), z.shape,
                                  generator=gen, device=z.device)
                drift = self._b(x)
                x = torch.cat([t + self.dt,
                               z + drift * self.dt + self.sigma(x, dw)],
                              dim=-1)
                collected.append(x)
            pts = torch.cat(collected, dim=0)[:num_points]
        else:
            pts = torch.empty(num_points, self.dim_x)
        if isin_ddp() and not self.spread_control:
            dist.broadcast(pts, src=0)
        return pts

    def ualoop_points(self, u_alpha, num_points):
        """Spread set induced by the CURRENT learned control u_alpha.

        Same layout as spread_points but the drift is mu(x, u_alpha(x))
        (the learned closed loop -- the deployment distribution).  Noise
        comes from a dedicated generator with fixed seed ualoop_seed
        (common random numbers across calls: between evaluations the
        points drift only through the evolution of u_alpha).  Rank 0
        simulates with its own replica (DDP keeps the weights in sync)
        and broadcasts.
        """
        n_levels = max(1, round(float(self.te - self.t0) / self.dt))
        num_paths = -(-num_points // n_levels)
        if (not isin_ddp()) or dist.get_rank() == 0:
            gen = torch.Generator(device=self._initial_points(1).device)
            gen.manual_seed(self.ualoop_seed)
            x = self._initial_points(num_paths)
            collected = [x]
            for _ in range(n_levels - 1):
                t, z = x[..., [0]], x[..., 1:]
                dw = torch.normal(0., math.sqrt(self.dt), z.shape,
                                  generator=gen, device=z.device)
                drift = self.mu(x, u_alpha(x))
                x = torch.cat([t + self.dt,
                               z + drift * self.dt + self.sigma(x, dw)],
                              dim=-1)
                collected.append(x)
            pts = torch.cat(collected, dim=0)[:num_points]
        else:
            pts = torch.empty(num_points, self.dim_x)
        if isin_ddp():
            dist.broadcast(pts, src=0)
        return pts

    def _ref_by_time(self, fn, x):
        """Evaluate a reference function level by level (the path-MC
        reference requires a single time level per call)."""
        t_all = x[:, 0]
        out = None
        for t_val in torch.unique(t_all):
            mask = t_all == t_val
            res = fn(x[mask])
            if out is None:
                out = torch.empty(x.shape[0], res.shape[1],
                                  dtype=res.dtype, device=res.device)
            out[mask] = res
        return out

    def _u_der_to_control(self, u_der):
        """Map the derived control u_der = -delta^2 grad_z v_theta (d-dim
        spatial gradient) to the control space for comparison against u_true.

        Default (U = R^d): identity.  Underactuated k-channel subclasses
        (dim_u = k < d) override to project onto the k controlled coordinates
        (u*_j = [u_der]_{2j-2}), keeping the derived-control metrics
        rel_l1err_ud[*] dimensionally consistent with the k-dim u_true."""
        return u_der

    def make_logfunc(self, v_theta, u_alpha=None):
        """Same metrics as SOCProblem.make_logfunc, but evaluated on the
        spread point set when eval_spread is set."""
        if not self.eval_spread:
            return super().make_logfunc(v_theta, u_alpha)
        x_test = self.spread_points(self.num_test_points)
        # t=0 diagonal curve (the submitted paper's D_0 protocol): fixed
        # deterministic grid, so identical on all ranks (no broadcast
        # needed); reference values precomputed once at startup, the
        # per-iteration cost is a 101-point network forward.
        dev = self._initial_points(1).device
        x_diag = utils.t_diag_curve(self.dim_x, 101, self.s_range,
                                    randomize=False)[1].to(dev)
        v_true_diag = self.v(x_diag)
        v_true_diag_l1 = torch.abs(v_true_diag).mean()
        if self.has_optimal_control and u_alpha is not None:
            u_true_diag = self.u_star(x_diag)
            u_true_diag_l1 = torch.abs(u_true_diag).mean()
        else:
            u_true_diag = None
        v_true = self._ref_by_time(self.v, x_test)
        v_true_l1 = torch.abs(v_true).mean()
        v_true_linf = torch.abs(v_true).max()
        if self.has_optimal_control and u_alpha is not None:
            u_true = self._ref_by_time(self.u_star, x_test)
            u_true_l1 = torch.abs(u_true).mean()
            u_true_linf = torch.abs(u_true).max()
        else:
            u_true = None

        def log_func(_it):
            # Evaluate the fp32 master weights in fp32: the networks' forward
            # enables bf16 autocast whenever AMP trained them, which injects
            # ~1e-3 absolute background noise into every output component and
            # dominates the u error on the 9990 inactive coordinates.
            nets = [v_theta] + ([u_alpha] if u_alpha is not None else [])
            flags = []
            for net in nets:
                mod = getattr(net, 'module', net)      # unwrap DDP
                flags.append(getattr(mod, 'enable_autocast', None))
                if flags[-1] is not None:
                    mod.enable_autocast = False
            try:
                with torch.no_grad():
                    err = v_theta(x_test) - v_true
                    abs_err = torch.abs(err)
                    log = {'rel_l1err': (abs_err.mean() / v_true_l1).item()}
                    if self.record_linf_error:
                        log['rel_linferr'] = (
                            abs_err.max() / v_true_linf).item()
                    if u_true is not None:
                        abs_err_u = torch.abs(u_alpha(x_test) - u_true)
                        log['rel_l1err_u'] = (
                            abs_err_u.mean() / u_true_l1).item()
                        if self.record_linf_error:
                            log['rel_linferr_u'] = (
                                abs_err_u.max() / u_true_linf).item()
                        # Decomposition: active support (|u*| >= 5% of its
                        # max) vs inactive background.  The all-coordinate
                        # L1 mean is floored at background/signal-mass ~ O(6)
                        # regardless of learning quality (sparse dilution).
                        mask_act = torch.abs(u_true) >= 0.05 * u_true_linf
                        log['rel_l1err_u_act'] = (
                            (abs_err_u * mask_act).sum() /
                            (torch.abs(u_true) * mask_act).sum()).item()
                        log['abserr_u_bg'] = (
                            (abs_err_u * (~mask_act)).sum() /
                            (~mask_act).sum()).item()
                        # preregistered E-SEP-S u metrics (owner 2026-09-01):
                        # per-component MAE (absolute -- no O(1/d) mean|u*|
                        # denominator pathology) and the energy ratio
                        # ||u_theta - u*||_F / ||u*||_F.  Flag-guarded so
                        # existing examples keep their column set unchanged.
                        if getattr(self, 'u_metrics_extended', False):
                            log['mae_u'] = abs_err_u.mean().item()
                            log['rel_e_u'] = (
                                abs_err_u.pow(2).sum().sqrt() /
                                u_true.pow(2).sum().sqrt()).item()
                    else:
                        log['rel_l1err_u'] = float('nan')
                        if self.record_linf_error:
                            log['rel_linferr_u'] = float('nan')
                    # t=0 diagonal curve (the submitted paper's D_0
                    # protocol, s in [-1, 1], 101-point uniform grid).
                    log['rel_l1err_diag'] = (torch.abs(
                        v_theta(x_diag) - v_true_diag).mean()
                        / v_true_diag_l1).item()
                    if u_true_diag is not None:
                        log['rel_l1err_u_diag'] = (torch.abs(
                            u_alpha(x_diag) - u_true_diag).mean()
                            / u_true_diag_l1).item()
                    else:
                        log['rel_l1err_u_diag'] = float('nan')
            finally:
                for net, flag in zip(nets, flags):
                    if flag is not None:
                        getattr(net, 'module', net).enable_autocast = flag
            # u_alpha-closed-loop (in-use) metrics at ualoop cadence; the
            # reference solve on fresh points is expensive (variant N:
            # path-MC), so off-cadence rows are NaN.  fork_rng keeps the
            # path-MC consumption off the training RNG stream.
            if self.eval_ualoop and u_true is not None:
                if _it % self.ualoop_loggap == 0:
                    flags2 = []
                    for net in nets:
                        mod = getattr(net, 'module', net)
                        flags2.append(getattr(mod, 'enable_autocast', None))
                        if flags2[-1] is not None:
                            mod.enable_autocast = False
                    try:
                        x_ua = self.ualoop_points(u_alpha,
                                                  self.num_test_points)
                        rng_devs = ([x_ua.device]
                                    if x_ua.device.type == 'cuda' else [])
                        with torch.random.fork_rng(devices=rng_devs):
                            v_ref_ua = self._ref_by_time(self.v, x_ua)
                            u_ref_ua = self._ref_by_time(self.u_star, x_ua)
                        log['rel_l1err_ualoop'] = (torch.abs(
                            v_theta(x_ua) - v_ref_ua).mean()
                            / torch.abs(v_ref_ua).mean()).item()
                        log['rel_l1err_u_ualoop'] = (torch.abs(
                            u_alpha(x_ua) - u_ref_ua).mean()
                            / torch.abs(u_ref_ua).mean()).item()
                        if getattr(self, 'u_metrics_extended', False):
                            au_ua = torch.abs(u_alpha(x_ua) - u_ref_ua)
                            log['mae_u_ualoop'] = au_ua.mean().item()
                            log['rel_e_u_ualoop'] = (
                                au_ua.pow(2).sum().sqrt() /
                                u_ref_ua.pow(2).sum().sqrt()).item()
                    finally:
                        for net, flag in zip(nets, flags2):
                            if flag is not None:
                                getattr(net, 'module',
                                        net).enable_autocast = flag
                else:
                    log['rel_l1err_ualoop'] = float('nan')
                    log['rel_l1err_u_ualoop'] = float('nan')
                    if getattr(self, 'u_metrics_extended', False):
                        log['mae_u_ualoop'] = float('nan')
                        log['rel_e_u_ualoop'] = float('nan')
            # Derived control u_der = -delta^2 * grad_z v_theta (R2M2's
            # relation u* = -delta^2 dv/dx).  Here H is quadratic in the
            # control with CONSTANT curvature delta^-2, so the argmin reads
            # off the FIRST derivative of v only -- the ReLU zero-curvature /
            # Hessian-denominator pathology of argmin-derived controls does
            # not apply.  Evaluated in fp32 (autocast restored above, so
            # re-disable for the autograd pass).
            if u_true is not None:
                mod = getattr(v_theta, 'module', v_theta)
                flag = getattr(mod, 'enable_autocast', None)
                if flag is not None:
                    mod.enable_autocast = False
                try:
                    with torch.enable_grad():
                        xg = x_test.detach().clone().requires_grad_(True)
                        gv = torch.autograd.grad(v_theta(xg).sum(), xg)[0]
                    u_der = -self.delta**2 * gv[:, 1:]
                    # Underactuated k-channel control: project the d-dim
                    # derived control onto the k controlled coordinates so it
                    # matches the k-dim u_true (identity for U = R^d).
                    u_der = self._u_der_to_control(u_der)
                    abs_err_ud = torch.abs(u_der - u_true)
                    log['rel_l1err_ud'] = (
                        abs_err_ud.mean() / u_true_l1).item()
                    mask_act = torch.abs(u_true) >= 0.05 * u_true_linf
                    log['rel_l1err_ud_act'] = (
                        (abs_err_ud * mask_act).sum() /
                        (torch.abs(u_true) * mask_act).sum()).item()
                    if self.record_linf_error:
                        log['rel_linferr_ud'] = (
                            abs_err_ud.max() / u_true_linf).item()
                finally:
                    if flag is not None:
                        mod.enable_autocast = flag
            return log

        return log_func


class _RingQuadrature:
    """Machine-precision reference for the linear ring drift (variant L).

    With r(t) = omega*(T - t), the uncontrolled terminal state is Gaussian:
    X_T ~ N(m, Sigma), m = e^{rL} z, Sigma = 2 delta^2 int_0^{T-t} e^{tau*omega*A} dtau,
    where L = S - I (cyclic shift minus identity) and A = L + L^T.  All three
    operators are circulant and are applied through FFT symbols:
      e^{rL}:   exp(r * (exp(+i theta) - 1)),  theta_m = 2 pi m / d
      e^{rL^T}: the complex conjugate symbol
      Sigma:    2 delta^2 * (e^{omega (T-t) lam} - 1) / (omega lam),
                lam_m = 2 cos(theta_m) - 2  (limit 2 delta^2 (T-t) at m = 0)
    Then v = -ln phi(y, C) with y = Q m, C = Q Sigma Q^T, and phi the k-dim
    Gaussian integral of 2/(1 + sum_j c_j y'^2), evaluated by tensor-product
    Gauss-Hermite quadrature; u* = delta^2 e^{rL^T} Q^T grad_y ln phi.
    """

    n_gh = 13          # Gauss-Hermite nodes per dimension

    def __init__(self, problem):
        self.problem = problem

    def _symbols(self, r, tau, d, device):
        theta = 2 * math.pi * torch.arange(d, dtype=torch.float64,
                                           device=device) / d
        lam_L = torch.exp(1j * theta) - 1
        prop = torch.exp(r * lam_L)                # symbol of e^{rL}
        propT = torch.exp(r * lam_L.conj())        # symbol of e^{rL^T}
        lam_A = 2 * torch.cos(theta) - 2           # symbol of A (real <= 0)
        delta2 = self.problem.delta**2
        omega = self.problem.omega
        # guard on the product omega*tau*lam: expm1(x)/x -> 1 as x -> 0, so
        # the branch value 2*delta^2*tau is the correct limit both for the
        # zero mode lam=0 (any omega) and for omega=0 (all modes; e^{0}=I,
        # Sigma = 2*delta^2*tau*I).  Guarding lam alone returned NaN at
        # omega=0 (ablation arm om0, owner 2026-08-27).
        arg = omega * tau * lam_A
        sgm2 = torch.where(
            arg.abs() > 1e-12,
            2 * delta2 * tau * torch.expm1(arg) / arg,
            torch.full_like(lam_A, 2 * delta2 * tau))
        return prop, propT, sgm2

    def _gh_grid(self, k, device):
        """Tensor-product Gauss-Hermite nodes/weights for N(0, I_k)."""
        import numpy as np
        nodes, weights = np.polynomial.hermite_e.hermegauss(self.n_gh)
        grids = np.meshgrid(*([nodes] * k), indexing='ij')
        xi = torch.tensor(np.stack([gi.ravel() for gi in grids], axis=-1),
                          dtype=torch.float64, device=device)   # (Q, k)
        w = torch.tensor(np.prod(np.meshgrid(*([weights] * k),
                                           indexing='ij'), axis=0).ravel(),
                         dtype=torch.float64, device=device)    # (Q,)
        w = w / (2 * math.pi)**(k / 2)
        return xi, w

    def _qmc_grid(self, k, device):
        """Scrambled-Sobol grid for N(0, I_k), equal weights.

        The rational integrand 2/(1 + sum c y^2) converges only algebraically
        under tensor Gauss-Hermite (diag 482491: at omega=0 / amplified c the
        n_gh=13 reference carries a ~1.5% bias, and the Q = n_gh^k memory
        rules out raising n_gh much past 13 at k=5).  A scrambled Sobol point
        set reaches ~1e-5 on this smooth integrand at 2^19 points (owner
        2026-08-27; fixed seed for reproducibility, torch-native, no scipy).
        """
        q = 2 ** self.problem.sparse_qmc_exp
        sob = torch.quasirandom.SobolEngine(dimension=k, scramble=True,
                                            seed=20260827)
        xi = sob.draw(q, dtype=torch.float64).to(device)         # (Q, k)
        xi = torch.special.ndtri(xi)                             # -> N(0,1)
        w = torch.full((q,), 1. / q, dtype=torch.float64, device=device)
        return xi, w

    def _phi_and_grad(self, y, C):
        """phi(y) = E[e^{-g}] and grad_y phi, G ~ N(0, C).

        y: (P, k) float64; C: (P, k, k) float64.
        ln terminal (historical): phi = E[2/(1 + sum c (y + G)^2)] by
        tensor Gauss-Hermite (default) or scrambled-Sobol QMC.
        LQ terminal (sparse_a set): Gaussian convolution of a quadratic,
        closed form in _phi_and_grad_lq -- no quadrature involved.
        """
        if getattr(self.problem, 'sparse_a', None) is not None:
            return self._phi_and_grad_lq(y, C)
        problem = self.problem
        k = problem.k_dirs
        c = torch.as_tensor(problem.c_weights, dtype=torch.float64,
                            device=y.device)                    # (k,)
        if getattr(problem, 'sparse_ref', 'gh') == 'qmc':
            xi, w = self._qmc_grid(k, y.device)                 # (Q,k), (Q,)
        else:
            xi, w = self._gh_grid(k, y.device)                  # (Q,k), (Q,)
        L_c = torch.linalg.cholesky(C)                          # (P,k,k)
        yq = y.unsqueeze(1) + torch.einsum('pij,qj->pqi', L_c, xi)  # (P,Q,k)
        denom = 1 + (c * yq.pow(2)).sum(-1)                     # (P,Q)
        h = 2 / denom
        phi = (w * h).sum(-1)                                   # (P,)
        dh = -4 * c * yq / denom.pow(2).unsqueeze(-1)           # (P,Q,k)
        dphi = torch.einsum('q,pqj->pj', w, dh)                 # (P,k)
        return phi, dphi

    def _phi_and_grad_lq(self, y, C):
        """phi = E[exp(-1/2 sum_j c_j (s_j - a)^2)], s ~ N(y, C): closed form
        (LQ terminal, owner 2026-08-28 sparse redesign; the ln-terminal
        quadrature is bypassed entirely):
            phi = det(I + C Lam)^-1/2 * exp(-1/2 (y-a)^T M (y-a)),
            M = (Lam^-1 + C)^-1,   grad_y phi = -phi * M (y-a).
        C is constant over the point batch by construction (one covariance
        per time level, expanded in _compute), so C[0] carries it.
        """
        problem = self.problem
        k = problem.k_dirs
        c = torch.as_tensor(problem.c_weights[:k], dtype=y.dtype,
                            device=y.device)                    # (k,)
        a = torch.full((k,), float(problem.sparse_a), dtype=y.dtype,
                       device=y.device)
        S = torch.diag(1. / c) + C[0]
        S = 0.5 * (S + S.t())                  # symmetrize FFT roundoff
        sol = torch.linalg.solve(S, (y - a).t()).t()            # M (y-a)
        # det(I + C Lam) = det(Lam) * det(Lam^-1 + C), via Cholesky of S
        Lc = torch.linalg.cholesky(S)
        logdet = c.log().sum() + 2 * torch.log(torch.diagonal(Lc)).sum()
        phi = torch.exp(-0.5 * (logdet + ((y - a) * sol).sum(-1)))
        return phi, -phi.unsqueeze(-1) * sol

    def _hybrid_level(self, m, sgm2, propT):
        """v / u* at one time level for the hybrid terminal
            g(z) = (c0 / 2d) |z|^2 + 1/2 sum_j c_j (s_j(z) - a)^2
        under FULL control (owner 2026-08-30 hybrid redesign).

        Z_T | Z_t = z ~ N(m, Sigma) with m and the circulant Sigma carried by
        FFT symbols as in _compute.  With K = alpha I + Q^T Lam Q (alpha =
        c0/d) and b = Q^T Lam a, completing the square gives
            g(z) = c_g + 1/2 (z - mu_g)^T K (z - mu_g),
            mu_g = Q^T mu_k,  mu_k,j = a c_j / (alpha + c_j),
            c_g  = 1/2 sum_j c_j a (a - mu_k,j),
        so phi = E[e^{-g}] = e^{-c_g} det(I + K Sig)^{-1/2}
                 * exp(-1/2 (m - mu_g)^T (K^-1 + Sig)^-1 (m - mu_g)).
        Both pieces reduce to circulant FFT symbols plus k x k algebra
        (Sylvester + Woodbury):
            det(I + K Sig) = det(A0) det(I_k + Lam^{1/2} Tt Lam^{1/2}),
              A0 = I + alpha Sig (symbol 1 + alpha*sgm2),
              Tt = Q Sig A0^-1 Q^T (symbol sgm2/(1 + alpha*sgm2));
            (K^-1 + Sig)^-1 = D0^-1 + D0^-1 Q^T E^-1 Q D0^-1,
              D0 = alpha^-1 I + Sig (symbol 1/alpha + sgm2),
              E = W^-1 - Q D0^-1 Q^T,  W_jj = c_j / (alpha (alpha + c_j)).
        Then v = -ln phi and u* = delta^2 grad_z ln phi
        = -delta^2 e^{rL^T} (K^-1 + Sig)^-1 (m - mu_g).
        """
        problem = self.problem
        assert problem.dim_u == problem.dim_z, (
            "hybrid terminal (sparse_c0 != 0) requires full control "
            "(dim_u == dim_z): with U = R^k, grad v acquires a radial "
            "component outside the sensitive subspace and the Cole-Hopf "
            "linearization of -delta^2 |B^T grad v|^2 breaks -- the "
            "closed form would be silently wrong")
        d = problem.dim_z
        k = problem.k_dirs
        device = m.device
        alpha = problem.sparse_c0 / d
        c = torch.as_tensor(problem.c_weights[:k], dtype=torch.float64,
                            device=device)                       # (k,)
        a = float(problem.sparse_a)
        mu_k = a * c / (alpha + c)                               # (k,)
        c_g = 0.5 * (c * a * (a - mu_k)).sum()                   # scalar
        sq2 = math.sqrt(2.)
        # Q rows (k,d): pair directions 1/sqrt(2) at (2j-2, 2j-1)
        Qd = torch.zeros(k, d, dtype=torch.float64, device=device)
        jj = torch.arange(k, device=device)
        Qd[jj, 2 * jj] = 1 / sq2
        Qd[jj, 2 * jj + 1] = 1 / sq2
        mu_full = torch.zeros(d, dtype=torch.float64, device=device)
        mup = mu_full[:2 * k].unflatten(-1, (k, 2))
        mup.copy_((mu_k / sq2).unsqueeze(-1).expand(k, 2))
        w = m - mu_full                                          # (P,d)

        # determinant: det(A0) * det(I_k + Lam^{1/2} Tt Lam^{1/2})
        A0sym = 1. + alpha * sgm2                                # (d,) real
        logdet0 = torch.log(A0sym).sum()
        s2a = sgm2 / A0sym
        SA = torch.fft.ifft(s2a.unsqueeze(0) *
                            torch.fft.fft(Qd, n=d, dim=-1),
                            n=d, dim=-1).real                    # (k,d)
        Tt = (SA[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) / sq2).t()
        Tt = 0.5 * (Tt + Tt.t())                                 # (k,k)
        sqc = c.sqrt()
        Tsym = torch.eye(k, dtype=torch.float64, device=device) + \
            sqc.unsqueeze(-1) * Tt * sqc.unsqueeze(0)
        logdet_k = 2 * torch.log(torch.diagonal(
            torch.linalg.cholesky(Tsym))).sum()
        logdet = logdet0 + logdet_k

        # quadratic form and gradient: (K^-1 + Sig)^-1 via Woodbury
        d0inv = 1. / (1. / alpha + sgm2)                         # (d,)
        y0 = torch.fft.ifft(d0inv.unsqueeze(0) *
                            torch.fft.fft(w, n=d, dim=-1),
                            n=d, dim=-1).real                    # D0^-1 w
        qf1 = (w * y0).sum(-1)                                   # (P,)
        p0 = y0[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) / sq2   # Q y0
        SD = torch.fft.ifft(d0inv.unsqueeze(0) *
                            torch.fft.fft(Qd, n=d, dim=-1),
                            n=d, dim=-1).real                    # (k,d)
        G0 = (SD[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) / sq2).t()
        G0 = 0.5 * (G0 + G0.t())                                 # Q D0^-1 Q^T
        Winv = alpha * (alpha + c) / c                           # (k,)
        E = torch.diag(Winv) - G0
        E = 0.5 * (E + E.t())
        sol = torch.linalg.solve(E, p0.t()).t()                  # (P,k)
        qf2 = (p0 * sol).sum(-1)                                 # (P,)
        quad = qf1 + qf2

        v = c_g + 0.5 * (logdet + quad)                          # (P,)
        # grad_m ln phi = -(y0 + D0^-1 Q^T sol); u* = delta^2 e^{rL^T} grad_m
        wk = torch.zeros(w.shape[0], d, dtype=torch.float64, device=device)
        wp = wk[:, :2 * k].unflatten(-1, (k, 2))
        wp.copy_((sol / sq2).unsqueeze(-1).expand_as(wp))
        y1 = torch.fft.ifft(d0inv.unsqueeze(0) *
                            torch.fft.fft(wk, n=d, dim=-1),
                            n=d, dim=-1).real                    # (P,d)
        gradm = -(y0 + y1)
        u = problem.delta**2 * torch.fft.ifft(
            propT.unsqueeze(0) * torch.fft.fft(gradm, n=d, dim=-1),
            n=d, dim=-1).real
        return v.unsqueeze(-1), u

    def _compute(self, x):
        problem = self.problem
        assert x.ndim == 2
        d = problem.dim_z
        k = problem.k_dirs
        device = x.device
        t_all = x[:, 0].double()
        z_all = x[:, 1:].double()
        v_out = torch.empty(x.shape[0], 1, dtype=torch.float64, device=device)
        u_out = torch.empty(x.shape[0], d, dtype=torch.float64, device=device)
        for t_val in torch.unique(t_all):
            mask = t_all == t_val
            z = z_all[mask]
            tau = float(problem.te - t_val)
            r = problem.omega * tau
            prop, propT, sgm2 = self._symbols(r, tau, d, device)
            m = torch.fft.ifft(prop * torch.fft.fft(z)).real    # (P,d)
            if getattr(problem, 'sparse_c0', 0.0) != 0.0:
                vh, uh = self._hybrid_level(m, sgm2, propT)
                v_out[mask] = vh
                u_out[mask] = uh
                continue
            mp = m[:, :2 * k].unflatten(-1, (k, 2))
            y = mp.sum(-1) / math.sqrt(2.)                      # (P,k)
            # C_{jl} = q_j^T Sigma q_l; Sigma applied by its FFT symbol
            Qd = torch.zeros(k, d, dtype=torch.float64, device=device)
            Qd[torch.arange(k), 2 * torch.arange(k)] = 1 / math.sqrt(2.)
            Qd[torch.arange(k), 2 * torch.arange(k) + 1] = 1 / math.sqrt(2.)
            SQ = torch.fft.ifft(sgm2.unsqueeze(0) *
                                torch.fft.fft(Qd, n=d, dim=-1),
                                n=d, dim=-1).real               # (k,d)
            # C[j, l] = q_j^T (Sigma q_l) = (SQ[l, 2j-2] + SQ[l, 2j-1])/sqrt2
            C = (SQ[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) /
                 math.sqrt(2.)).t()                             # (k_j, k_l)
            C = 0.5 * (C + C.t())  # symmetrize FFT roundoff
            C = C.expand(z.shape[0], k, k)
            phi, dphi = self._phi_and_grad(y, C)
            v_out[mask] = (-torch.log(phi)).unsqueeze(-1)
            w = torch.zeros(z.shape[0], d, dtype=torch.float64,
                            device=device)
            a = (dphi / phi.unsqueeze(-1)) / math.sqrt(2.)      # (P,k)
            wp = w[:, :2 * k].unflatten(-1, (k, 2))
            wp.copy_(a.unsqueeze(-1).expand_as(wp))
            u_out[mask] = problem.delta**2 * torch.fft.ifft(
                propT * torch.fft.fft(w)).real
        return v_out, u_out

    def v(self, x):
        v_out, _ = self._compute(x)
        return v_out.to(x.dtype)

    def u_star(self, x):
        _, u_out = self._compute(x)
        return u_out.to(x.dtype)


class HJBSparseRing(HJBSparseBase):
    """Variant (L): b_i(t, z) = omega * (z_{i+1} - z_i), cyclic; machine-
    precision reference (Poisson-kernel propagation + k-dim quadrature)."""

    omega = 20.0     # ring transport rate; r(0) = omega * T
    has_optimal_control = True

    def _b(self, x):
        z = x[..., 1:]
        return self.omega * (z.roll(-1, dims=-1) - z)

    def v(self, x):
        return _RingQuadrature(self).v(x)

    def u_star(self, x):
        return _RingQuadrature(self).u_star(x)


class HJBSparseLQ(HJBSparseRing):
    """Quadratic terminal with a nonzero sparse target offset (owner
    2026-08-28, sparse-example redesign phase 0+1):
        g(z) = 1/2 * sum_j c_j (s_j(z) - a)^2,   a = sparse_a (INI knob).

    Same drift / sigma / running cost as variant L, so the Cole-Hopf
    identity v = -ln E[e^{-g}] holds unchanged and the reference stays
    machine precision -- E[e^{-g}] is now a Gaussian convolution in
    closed form (_RingQuadrature._phi_and_grad_lq; the sparse_ref /
    sparse_qmc_exp knobs are inert for this class).  This realizes the
    note_lqsparse design intent the ln terminal missed: u* = -delta^2
    grad v has O(1) sparse active components (ln saturated at 0.05-0.16)
    and the offset gives control a mean-steering job (analytic gain ~2/3
    at a=1, delta=1, T=1 vs ~7% for the ln terminal)."""

    sparse_a = 1.0
    # full-rank radial terminal coefficient (owner 2026-08-30, hybrid-terminal
    # redesign): g gains (sparse_c0 / 2d) |z|^2 on top of the sparse quadratic.
    # Default 0.0 = the original sparse LQ terminal, zero behavior change.
    # Requires FULL control (dim_u = dim_z): with U = R^k the HJB nonlinearity
    # -delta^2 |B^T grad v|^2 no longer Cole-Hopf-linearizes once grad v has a
    # radial component outside the sensitive subspace, so the closed-form
    # reference would be silently wrong -- _compute_hybrid_level asserts this.
    sparse_c0 = 0.0

    def g(self, z):
        s = self._sens(z)
        c = torch.as_tensor(self.c_weights[:self.k_dirs], dtype=z.dtype,
                            device=z.device)
        gq = 0.5 * (c * (s - self.sparse_a).pow(2)).sum(-1, keepdim=True)
        if self.sparse_c0 != 0.0:
            gq = gq + 0.5 * self.sparse_c0 / self.dim_z * \
                z.pow(2).sum(-1, keepdim=True)
        return gq

    def grad_g(self, z):
        """grad g: entry c_j (s_j - a)/sqrt(2) at both coordinates of pair
        j, plus (c0/d) z everywhere when sparse_c0 is set."""
        s = self._sens(z)
        c = torch.as_tensor(self.c_weights[:self.k_dirs], dtype=z.dtype,
                            device=z.device)
        a = c * (s - self.sparse_a) / math.sqrt(2.)            # (..., k)
        grad = torch.zeros_like(z)
        pairs = grad[..., :2 * self.k_dirs].unflatten(-1, (self.k_dirs, 2))
        pairs.copy_(a.unsqueeze(-1).expand_as(pairs))
        if self.sparse_c0 != 0.0:
            grad = grad + (self.sparse_c0 / self.dim_z) * z
        return grad


class HJBSparseLQUA(HJBSparseLQ):
    """k-channel control (owner 2026-08-29, variant A): the control lives in
    R^k with k = k_dirs << d -- an UNDERACTUATED sparse-control problem.

        dz = [ b(z) + 2 B u ] dt + sqrt(2) delta dW,   u in R^k,
        B = sum_j q_j e_j^T,  q_j = (e_{2j-2} + e_{2j-1})/sqrt(2)
            (d x k; channel j drives the sensitive direction s_j),
        f(x, u) = delta^{-2} |u|^2   (k-dim norm),
        g(z) = 1/2 sum_j c_j (s_j(z) - a)^2   (unchanged LQ terminal).

    The state dynamics stays genuinely d-dim (ring drift couples all d
    coordinates, full-rank diffusion sqrt(2) delta I_d): only the CONTROL has
    k channels, so the networks must locate the k-dim feedback signal inside
    the R^{d+1} input.  The value reference v = -ln E[e^{-g}] is UNCHANGED
    (the controlled HJB nonlinearity -delta^2 |B^T grad v|^2 Cole-Hopf-
    linearizes exactly like the d-dim case because B B^T = projection onto
    the k sensitive directions, which is the identity on the support of
    grad v), so the machine-precision closed-form Gaussian-convolution
    reference is inherited intact.  The optimal control is k-dim:
    u* = -delta^2 B^T grad_z v = (-delta^2 partial_{s_j} v)_j.

    This fixes the d=1000 u_act = 0.35 failure mode: with U = R^d the u-
    network had to output d numbers (only 2k of them nonzero), diluting the
    learning signal across the full input; with U = R^k the output dimension
    matches the number of active control channels, and d=1000 reduces to the
    well-conditioned d=20 regime (same gain 0.640, same |u*| ~ 0.30).  The
    pair-direction B (each channel drives a sensitive direction s_j) keeps the
    full 0.640 control gain -- the earlier diagonal B (channel j driving only
    coordinate 2j-2) halved the controllable value and broke both v and u
    learning on the d=20 control arm (496257: u_act 0.373, diag_v 0.598).
    """

    @property
    def dim_u(self):
        """Control dimension = k (k_dirs), not d.  A property (not an
        __init__ assignment) so it tracks k_dirs when the INI injector sets
        k_dirs after instantiation."""
        return self.k_dirs

    @dim_u.setter
    def dim_u(self, _value):
        """Absorb SOCProblem.__init__'s self.dim_u = dim_x - 1 (the d-dim
        default); the authoritative k-dim value is derived from k_dirs."""
        pass

    def mu(self, x, kappa):
        """Controlled drift: b(z) + 2 B kappa, kappa in R^k, with the PAIR-
        DIRECTION control matrix B whose column j is the sensitive direction
        q_j = (e_{2j-2} + e_{2j-1})/sqrt(2) -- channel j drives the sensitive
        coordinate s_j directly.  This restores full control authority on the
        sensitive subspace (B B^T |_{sensitive} = identity), so the Cole-Hopf
        closed-form reference and the 0.640 control gain of the d-dim LQ
        problem are preserved (the diagonal B that drove only coordinate 2j-2
        halved the controllable value and broke both v and u learning)."""
        z_shape = x[..., 1:].shape[:-1] + (self.dim_z,)
        full = torch.zeros(z_shape, dtype=x.dtype, device=x.device)
        # B kappa: channel j -> q_j = (e_{2j-2} + e_{2j-1})/sqrt(2)
        idx0 = 2 * torch.arange(self.k_dirs, device=x.device)
        full[..., idx0] = kappa / math.sqrt(2.)
        full[..., idx0 + 1] = kappa / math.sqrt(2.)
        return self._b(x) + 2 * full

    def mu_pilot(self, x):
        """Pilot drift = uncontrolled b(z) (kappa = 0 in R^k).  Overridden
        because the base mu_pilot passes a d-dim zero control, which does not
        match the k-dim control channel of this class."""
        return self._b(x)

    def running_cost(self, acc, x, kappa):
        """f = delta^{-2} |kappa|^2 over the k control channels."""
        return acc + self.delta**(-2) * kappa.pow(2).sum(-1, keepdims=True)

    def u_star(self, x):
        """k-dim optimal control for the pair-direction B:
        u*_j = -delta^2 (q_j . grad_z v) = -delta^2 partial_{s_j} v.

        _RingQuadrature._compute returns u_out = delta^2 e^{r L^T} w with
        w = (dphi/phi)/sqrt2 on BOTH coordinates of each active pair; since
        grad v = -grad phi/phi, u_out IS -delta^2 grad_z v propagated to t, and
        its two entries on pair j are EQUAL (both = a_j).  Contracting with
        q_j = (e_{2j-2}+e_{2j-1})/sqrt(2) gives
            u*_j = (u_out[2j-2] + u_out[2j-1]) / sqrt(2) = sqrt(2) * a_j.
        """
        _, u_d = _RingQuadrature(self)._compute(x)
        upair = u_d[:, :2 * self.k_dirs].unflatten(-1, (self.k_dirs, 2))
        u_k = upair.sum(-1) / math.sqrt(2.)          # (P, k) = q_j . u_out
        return u_k.to(x.dtype)

    def _u_der_to_control(self, u_der):
        """Project the d-dim derived control -delta^2 grad_z v_theta onto the
        k sensitive directions (u*_j = q_j . u_der = (u_der[2j-2] +
        u_der[2j-1])/sqrt(2)), matching the k-dim u_true in rel_l1err_ud[*]."""
        upair = u_der[:, :2 * self.k_dirs].unflatten(-1, (self.k_dirs, 2))
        return upair.sum(-1) / math.sqrt(2.)


def _dct_noise_basis(d, m, dtype, device, _cache={}):
    """First m columns of the orthogonal DCT-II basis (d x m), cached.

    q_{k,n} = a_k cos(pi (2n+1) k / (2d)), a_0 = sqrt(1/d),
    a_k = sqrt(2/d): exactly column-orthogonal, deterministic (no seed),
    nested in m.  Row squared norms are m/d up to an O(1/m)-relative
    cos^2 phase spread (exact only for the full m = d basis).  Shared by
    HJBSparseSEP and HJBSparseLQSN."""
    key = (d, m, dtype, str(device))
    if key not in _cache:
        n = torch.arange(d, dtype=torch.float64, device=device)
        k = torch.arange(m, dtype=torch.float64, device=device).unsqueeze(-1)
        # explicit float64 amplitudes: torch.where(bool, pyfloat,
        # pyfloat) promotes the scalars to the DEFAULT dtype (float32),
        # which caps the orthogonality at ~1e-8.
        amp = torch.full_like(k, (2. / d) ** 0.5)
        amp[0] = d ** -0.5
        Q = (amp * torch.cos(math.pi * (2 * n + 1) * k / (2 * d))).t()
        _cache[key] = Q.to(dtype)
    return _cache[key]


class HJBSparseSEP(HJBSparseRing):
    """E-SEP-S: separable shifted-quadratic terminal + sparse-noise
    diffusion (owner 2026-09-01, after the attribution campaign closed all
    four axes; construction principles distilled there):

        dz = 2 u dt + sqrt(2) delta sqrt(d/m) Q_m dW,  u in R^d,  b = 0,
        g(z) = (c / 2d) |z - a|^2,   f = delta^{-2} |u|^2,   T = 1.

    The terminal signal is FULL-RANK in all d coordinates; the sparsity sits
    on the noise side only: the diffusion spans the first m columns of the
    orthogonal DCT-II basis Q_m (fixed, deterministic, nested in m -- the
    m=20 subspace is the first 20 columns of the m=100 one).  Each noise
    DIRECTION carries the same variance 2 delta^2 as the full-dim case
    (sigma = sqrt(2) delta Q_m Q_m^T): this is the Cole-Hopf MATCHING
    CONDITION, not a normalization choice -- the linearization of
    f = delta^{-2}|u|^2 / drift 2u requires the per-direction variance to be
    exactly 2 delta^2 (holding the per-COORDINATE variance at 2 delta^2
    instead scales each direction to 2 delta^2 d/m, which breaks the
    identity v = -ln E[e^{-g}]: 1-D evidence 2026-09-01, gap +1830% at
    beta0=100, converging in dt; +0.1% at beta0=2).  Total noise energy is
    therefore 2 delta^2 m -- the m axis is the noise direction COUNT at
    standard per-direction intensity.  c is O(d) (alpha = c/d = O(1)): the
    per-coordinate signal then scales with the per-coordinate noise exactly
    like HJB-1's geometry (gain 52-66% across m, typical |u*| = O(1)).  The
    c = O(1) form collapses the control gain to O(1/d) and degenerates (bad
    control = optimal control, the delta-sweep lesson).

    Reference (rank-m Gaussian quadratic MGF, closed form, log-domain;
    cross-checked against dense linear algebra, tensor Gauss-Hermite and
    Monte Carlo in check_sep):
        Sigma = 2 delta^2 tau P_m,   P_m = Q_m Q_m^T,
        v(t,x)  = (m/2) ln(1 + alpha beta) + (alpha/2)|x-a|^2
                  - (alpha gamma/2)|Q_m^T(x-a)|^2,
        u*(t,x) = -delta^2 alpha [(x-a) - gamma P_m (x-a)],
        alpha = c/d,  beta = 2 delta^2 tau,  gamma = ab/(1+ab).
    m = d recovers the fully-noisy E-SEP main proposal (the closed form
    degenerates to d independent 1D Gaussian convolutions).  b = 0 keeps the
    problem in the proven zero-drift stability regime, and makes the Euler
    terminal law EXACT (no O(dt) covariance bias on the uncontrolled cost).
    """

    omega = 0.0
    sparse_a = 1.0
    c_weights = (1.0,)      # single weight c; INI sparse_c overrides (len 1)
    sparse_m = None         # noise dimension m; INI sparse_m (required)
    # preregistered u metrics (owner 2026-09-01): per-component MAE (no
    # O(1/d) denominator pathology) + energy ratio ||u-u*||_F/||u*||_F
    u_metrics_extended = True

    def _sep_params(self):
        m = int(self.sparse_m) if self.sparse_m is not None else None
        if m is None or not 1 <= m <= self.dim_z:
            raise ValueError(
                f"sparse_m (noise dimension) must be set in 1..{self.dim_z}, "
                f"got {self.sparse_m!r}")
        if len(self.c_weights) != 1:
            raise ValueError(f"E-SEP terminal takes a single c (sparse_c), "
                             f"got {self.c_weights}")
        return float(self.c_weights[0]), float(self.sparse_a), m

    def _sep_Q(self, dtype, device):
        """First m columns of the DCT basis, via the shared module cache
        (behavior identical to the pre-refactor inline construction)."""
        return _dct_noise_basis(self.dim_z, int(self.sparse_m), dtype, device)

    def g(self, z):
        c, a, _ = self._sep_params()
        mu = z - a
        return 0.5 * (c / self.dim_z) * mu.pow(2).sum(-1, keepdim=True)

    def grad_g(self, z):
        c, a, _ = self._sep_params()
        return (c / self.dim_z) * (z - a)

    def sigma(self, x, dw):
        """dw stays d-dim (sampling interface unchanged); the EFFECTIVE
        Brownian dimension is m: sigma dw = sqrt(2) delta Q Q^T dw.  The
        per-DIRECTION variance is exactly 2 delta^2 (the Cole-Hopf matching
        condition -- see class docstring); pairing +-dw keeps its
        antisymmetry."""
        _, _, m = self._sep_params()
        Q = self._sep_Q(dw.dtype, dw.device)
        return (2 ** 0.5) * self.delta * ((dw @ Q) @ Q.t())

    def _sep_closed(self, x):
        """v and u* in one pass, float64 internally (log-domain, no exp
        underflow at large d)."""
        c, a, m = self._sep_params()
        d = self.dim_z
        alpha = c / d
        tau = self.te - x[..., 0].double()
        beta = 2.0 * self.delta ** 2 * tau
        gam = alpha * beta / (1.0 + alpha * beta)
        mu = x[..., 1:].double() - a
        Q = self._sep_Q(torch.float64, x.device)
        qmu = mu @ Q
        v = (0.5 * m * torch.log1p(alpha * beta)
             + 0.5 * alpha * (mu.pow(2).sum(-1) - gam * qmu.pow(2).sum(-1)))
        u = -(self.delta ** 2) * alpha * (mu - gam.unsqueeze(-1) * (qmu @ Q.t()))
        return v.unsqueeze(-1), u

    def v(self, x):
        return self._sep_closed(x)[0].to(x.dtype)

    def u_star(self, x):
        return self._sep_closed(x)[1].to(x.dtype)


class HJBSparseLQSN(HJBSparseLQ):
    """LQ-SN: rank-k LQ terminal + sparse noise (owner 2026-09-02 redesign).

        dz = (b(z) + 2 u) dt + sqrt(2) delta Q_m Q_m^T dW,   u in R^d,
        b_i(z) = omega (z_{i+1} - z_i)  (cyclic ring drift, inherited from
              HJBSparseRing._b; omega is the INI sparse_omega knob),
        g(z) = 1/2 sum_j c_j (s_j(z) - a)^2   (pair-coordinate sensitive
              directions s_j = (z_{2j-2} + z_{2j-1})/sqrt(2), owner choice),
        f = delta^{-2} |u|^2.

    Owner's magnitude constraints (2026-09-02): v must grow at most
    logarithmically in d (uniform per-coordinate signals cannot do this
    together with ||u*|| = O(1): w*d <= ln d vs w*sqrt(d) >= c contradict
    at large d), and ||u*|| = O(1) with individual components allowed near
    zero.  Hence the signal energy concentrates on the k pair directions
    (the LQ family terminal: v = O(1), ||u*|| = O(sqrt k), exactly 2k
    nonzero components), while the noise keeps the E-SEP-S matching form:
    per-direction variance exactly 2*delta^2 (Cole-Hopf condition),
    sigma = sqrt(2) delta Q_m Q_m^T with Q_m the PAIR-FIRST basis (see
    _sn_basis: pair directions first, deterministic DCT fill-in; nested
    in m).

    Reference (closed form; b = 0, no propagation): terminal sensitive
    coordinates s(z_T) ~ N(s(z), C) with
        C = 2 delta^2 tau (S Q_m)(S Q_m)^T     (k x k, S = pair directions)
    so (Gaussian convolution of a quadratic, _phi_and_grad_lq's identity):
        v = 1/2 (logdet(I + C Lam) + (s-a)^T M (s-a)),   M = (Lam^-1 + C)^-1,
        u* = -delta^2 S^T M (s-a)   (nonzero only on the 2k pair coords).
    Lam^{-1} + C is positive definite whenever diag(1/c) > 0, so the
    Cholesky succeeds for ANY m (including m < k, rank-deficient C).

    Completes the 2x2 attribution matrix: full-dim signal x full-dim noise
    = E-SEP (0.54); full-dim x sparse = E-SEP-S m=20 (0.18, saturating
    0.25); rank-k x full-dim = LQ family (0.10); rank-k x sparse = this.
    """

    omega = 0.0
    sparse_m = None         # noise dimension m; INI sparse_m (required)
    # preregistered u metrics alongside the family's rel_l1err_u_act
    u_metrics_extended = True

    def _sn_params(self):
        m = int(self.sparse_m) if self.sparse_m is not None else None
        if m is None or not 1 <= m <= self.dim_z:
            raise ValueError(
                f"sparse_m (noise dimension) must be set in 1..{self.dim_z}, "
                f"got {self.sparse_m!r}")
        return m

    def _pair_S(self, dtype, device):
        """Sensitive-direction matrix S (k, d): pair coordinates."""
        S = torch.zeros(self.k_dirs, self.dim_z, dtype=torch.float64,
                        device=device)
        jj = torch.arange(self.k_dirs, device=device)
        S[jj, 2 * jj] = 1 / math.sqrt(2.)
        S[jj, 2 * jj + 1] = 1 / math.sqrt(2.)
        return S.to(dtype)

    def _sn_basis(self, dtype, device, _cache={}):
        """Noise basis Q_m (d x m): the min(m, k) pair directions FIRST,
        then deterministic DCT fill-in orthogonalized against them (one
        QR of [S^T | full DCT], whose first columns reproduce S^T exactly
        since S is already orthonormal).

        Why: with plain DCT columns the pair directions -- themselves
        low-frequency-smooth vectors -- project onto the low-frequency
        noise subspace nearly collinearly, and the projected covariance
        C = 2 delta^2 tau (S Q)(S Q)^T becomes ill-conditioned (measured
        cond ~1e16 at d=1000, m=20; zero-check 511402, negative DPP gaps).
        Pair-first ordering bounds C's spectrum below by 2 tau I on the
        covered pair directions for ANY m.  Nested in m; per-direction
        variance stays exactly 2 delta^2 (any orthonormal columns satisfy
        the Cole-Hopf matching)."""
        d, k = self.dim_z, self.k_dirs
        m = self._sn_params()
        key = (d, k, m, dtype, str(device))
        if key not in _cache:
            S = self._pair_S(torch.float64, device)
            Qdct = _dct_noise_basis(d, d, torch.float64, device)
            B = torch.cat([S.t(), Qdct], dim=1)         # (d, d + k)
            Q, _ = torch.linalg.qr(B)
            _cache[key] = Q[:, :m].to(dtype)
        return _cache[key]

    def sigma(self, x, dw):
        """dw stays d-dim (sampling interface unchanged); effective Brownian
        dimension m, per-direction variance exactly 2 delta^2 (matching)."""
        m = self._sn_params()
        Q = self._sn_basis(dw.dtype, dw.device)
        return (2 ** 0.5) * self.delta * ((dw @ Q) @ Q.t())

    def _omega_symbols(self, tau, d, device):
        """FFT symbols of e^{omega*tau*L} / e^{omega*tau*L^T}, L = S - I
        (cyclic shift minus identity, the ring drift generator).  Same
        conventions as _RingQuadrature._symbols."""
        theta = 2 * math.pi * torch.arange(d, dtype=torch.float64,
                                           device=device) / d
        lam = torch.exp(1j * theta) - 1
        omega = float(self.omega)
        prop = torch.exp(omega * tau * lam)
        propT = torch.exp(omega * tau * lam.conj())
        return prop, propT

    def _lqsn_ref(self, x):
        """v and u* in one pass, float64 internally, per-time-level.

        omega == 0 (any m): closed form, C = 2 delta^2 tau (S Q_m)(S Q_m)^T.
        omega != 0, m == d (full-rank noise): delegated to _RingQuadrature --
        with sigma sigma^T = 2 delta^2 I the terminal law is identical to the
        LQ family and the circulant FFT formula applies exactly.
        omega != 0, m < d: the projected covariance is NOT circulant, so
            C(tau) = 2 delta^2 int_0^tau (S e^{omega*s*L} Q_m)(...)^T ds     (trapezoid),
            mean_s = S e^{omega*tau*L} z,
            u* = -delta^2 e^{omega*tau*L^T} S^T M (mean_s - a),
        with M = (Lam^-1 + C)^-1 as in the LQ Gaussian-convolution identity."""
        d, k = self.dim_z, self.k_dirs
        omega = float(self.omega)
        if omega != 0.0 and int(self._sn_params()) == d:
            q = _RingQuadrature(self)
            return q.v(x), q.u_star(x)
        c = torch.as_tensor(self.c_weights[:k], dtype=torch.float64,
                            device=x.device)
        a = float(self.sparse_a)
        m = self._sn_params()
        Q = self._sn_basis(torch.float64, x.device)
        S = self._pair_S(torch.float64, x.device)
        SQ = S @ Q                                        # (k, m)
        t_all = x[:, 0].double()
        z_all = x[:, 1:].double()
        v_out = torch.empty(x.shape[0], 1, dtype=torch.float64,
                            device=x.device)
        u_out = torch.zeros(x.shape[0], d, dtype=torch.float64,
                            device=x.device)
        for t_val in torch.unique(t_all):
            mask = t_all == t_val
            tau = float(self.te - t_val)
            if omega == 0.0:
                C = 2.0 * self.delta ** 2 * tau * (SQ @ SQ.t())
                C = 0.5 * (C + C.t())
            else:
                # projected covariance by numerical integration (trapezoid):
                #   C(tau) = 2 delta^2 int_0^tau B(s) B(s)^T ds,
                #   B(s) = S e^{omega*s*L} Q_m  = (S @ (e^{omega*s*L} Q_m)).
                # e^{omega*s*L} is applied column-by-column by its FFT symbol.
                # C depends only on the time level (not the points), and the
                # quadrature is 65-257 sequential steps -- memoize per tau so
                # repeated calls (log columns; the u_fixed_ref residual every
                # descent step) reuse it.  Bitwise-identical: same
                # computation, computed once per (tau, problem instance).
                if not hasattr(self, '_lqsn_C_cache'):
                    self._lqsn_C_cache = {}
                C = self._lqsn_C_cache.get(round(tau, 12))
                if C is None:
                    n = max(64, int(round(tau * 256)))
                    ds = tau / n
                    C = torch.zeros(k, k, dtype=torch.float64,
                                    device=x.device)
                    jj = torch.arange(k, device=x.device)
                    for s_i in range(n + 1):
                        s = s_i * ds
                        prop_s, _ = self._omega_symbols(s, d, x.device)
                        eLQ = torch.fft.ifft(
                            prop_s.unsqueeze(1) * torch.fft.fft(
                                Q, n=d, dim=0), n=d, dim=0).real   # (d, m)
                        # B = S @ eLQ: pair averages (B_b=(eLQ[2b]+eLQ[2b+1])/sqrt2)
                        B = (eLQ[2 * jj] + eLQ[2 * jj + 1]) / math.sqrt(2.)  # (k,m)
                        w = 1.0 if (s_i == 0 or s_i == n) else 2.0
                        C = C + w * (B @ B.t())
                    C = self.delta ** 2 * ds * C
                    C = 0.5 * (C + C.t())
                    self._lqsn_C_cache[round(tau, 12)] = C
            # mean in sensitive coordinates
            if omega == 0.0:
                # S, not SQ: the mean is S z (state coords); SQ is (k, m).
                # (2c6738b refactor regression -- crashed every m<d, omega=0
                # run, first exposed by the Omega-1 scan arm 511835.)
                mvec = z_all[mask] @ S.t()                # (P, k)
            else:
                prop_tau, _ = self._omega_symbols(tau, d, x.device)
                Rz = torch.fft.ifft(prop_tau * torch.fft.fft(
                    z_all[mask], n=d, dim=-1), n=d, dim=-1).real
                mvec = Rz @ S.t()                          # (P, k)
            Sm = torch.diag(1. / c) + C
            Sm = 0.5 * (Sm + Sm.t())
            sol = torch.linalg.solve(Sm, (mvec - a).t()).t()  # M (mvec - a)
            Lc = torch.linalg.cholesky(Sm)
            logdet = c.log().sum() + 2 * torch.log(torch.diagonal(Lc)).sum()
            v_out[mask] = 0.5 * (logdet + ((mvec - a) * sol).sum(-1,
                                                                 keepdim=True))
            # u* = -delta^2 e^{omega*tau*L^T} S^T (M (mvec - a))
            wpair = (-self.delta ** 2) * sol / math.sqrt(2.)  # (P, k)
            full = torch.zeros(sol.shape[0], d, dtype=torch.float64,
                               device=x.device)
            ff = full[:, :2 * k].unflatten(-1, (k, 2))
            ff.copy_(wpair.unsqueeze(-1).expand_as(ff))
            if omega != 0.0:
                _, propT_tau = self._omega_symbols(tau, d, x.device)
                full = torch.fft.ifft(
                    propT_tau.unsqueeze(0) * torch.fft.fft(full, n=d, dim=-1),
                    n=d, dim=-1).real
            u_out[mask] = full
        return v_out, u_out

    def v(self, x):
        return self._lqsn_ref(x)[0].to(x.dtype)

    def u_star(self, x):
        return self._lqsn_ref(x)[1].to(x.dtype)

    def u_star_term(self, x):
        """Closed-form TERMINAL optimal control u_T (owner 2026-09-04 plan A,
        time-gated anchor): pin the time column to te and call the existing
        u_star -- at t = te (tau = 0) C(0) = 0 and the ring propagation is the
        identity, so u_T = -delta^2 S^T diag(c) (S z - a), a linear feedback
        that depends only on the problem data.  The tau = 0 covariance is
        memoized by _lqsn_C_cache (f50bc65), so repeated calls (every anchored
        forward in training) cost only the per-point k x k solve."""
        x_te = x.clone()
        x_te[..., 0] = self.te
        return self.u_star(x_te).detach()


class HJBSparseLQK1(HJBSparseLQSN):
    """k = 1 limit of the k-channel control design (owner 2026-09-04,
    approved): ONE scalar control channel driving the single sensitive
    direction.

        dz = [ b(z) + 2 q kappa ] dt + sqrt(2) delta Q_m Q_m^T dW,
             kappa in R (scalar),
             q = (e_0 + e_1)/sqrt(2)   (B in R^{d x 1}, B^T B = 1),
        g(z) = c/2 (s(z) - a)^2,   s(z) = (z_0 + z_1)/sqrt(2) = q . z,
        f(x, kappa) = delta^{-2} kappa^2.

    Lineage = the kctl rescued form (HJBSparseLQUA, 2026-08-30): the control
    matrix column is the PAIR direction (owner ruling; the diagonal-B variant
    collapsed controllability, 496257).  The gain-2 pairing
    (drift 2 q kappa vs cost delta^{-2} kappa^2) is the family convention
    (HJBSparseBase.mu = b + 2 kappa, HJBSparseLQUA.mu = b + 2 B kappa), so
    the Hamiltonian minimum reads kappa* = -delta^2 q^T grad v -- the
    q-PROJECTION of the full-dim u* = -delta^2 grad v, with no extra factor.

    References inherited BITWISE from HJBSparseLQSN (v, sigma, the C(tau)
    trapezoid cache f50bc65, u_star_term): the full-dim reference chain is
    untouched; only the CONTROL map is restricted.  Value identity (the
    zero-check assertion, check_lqsn_k1.py): at omega = 0 (any m) the k = 1
    value function equals the k_dirs = 1 full-dim LQ-SN value EXACTLY -- the
    full-dim optimal control lives on the sensitive subspace (grad v is
    supported on the pair coordinates), so restricting U to the line R q
    removes nothing; with m = d the Cole-Hopf linearization is
    rank-unconditional and the identity holds at any omega.  At omega != 0,
    m < d both problems share the same inherited reference approximation
    (grad v spreads off the noise range; the closed-loop DPP spot check
    quantifies the residual).

    Scientific purpose (owner's capacity diagnosis, 2026-09-04): with
    U = R^d the d = 1000 u metrics were diluted across the d-dim output;
    k = 1 removes the output-dimension dilution entirely -- mean|u*| = O(1)
    on a single scalar channel, so mae_u / rel_e_u have full discriminating
    power for the first time.  If the scalar u still under-learns at d = 1000,
    the capacity diagnosis is refuted (the problem is optimization-side);
    if it learns, the diagnosis is strongly confirmed.

    INI: no new knobs -- omega/m/delta/a/c ride the existing sparse_omega /
    sparse_m / sparse_delta / sparse_a / sparse_c keys; k is pinned to 1
    (sparse_k is rejected loudly by the k_dirs guard).
    """

    @property
    def k_dirs(self):
        """k is pinned to 1 (the design); sparse_k must NOT be set."""
        return 1

    @k_dirs.setter
    def k_dirs(self, value):
        # parse_problem assigns problem.k_dirs = config.getint('sparse_k',
        # fallback=class value) whenever the attribute exists -- the guard
        # turns a wrong sparse_k in the INI into a loud error instead of a
        # silently inconsistent terminal/control split.
        if int(value) != 1:
            raise ValueError(
                "HJBSparseLQK1 is the k=1 design: k_dirs is pinned to 1; "
                f"got k_dirs/sparse_k = {value!r} (do not set sparse_k)")

    @property
    def dim_u(self):
        """Control dimension = 1 (scalar channel), tracking the pinned
        k_dirs (HJBSparseLQUA's property pattern)."""
        return self.k_dirs

    @dim_u.setter
    def dim_u(self, _value):
        """Absorb SOCProblem.__init__'s self.dim_u = dim_x - 1 (the d-dim
        default); the authoritative value is derived from k_dirs."""
        pass

    def mu(self, x, kappa):
        """Controlled drift b(z) + 2 q kappa with kappa in R (shape (..., 1)):
        the scalar channel drives the pair direction q = (e_0+e_1)/sqrt(2)
        with the family's gain-2 pairing (see class docstring)."""
        full = torch.zeros_like(x[..., 1:])
        full[..., 0] = kappa[..., 0] / math.sqrt(2.)
        full[..., 1] = kappa[..., 0] / math.sqrt(2.)
        return self._b(x) + 2. * full

    def mu_pilot(self, x):
        """Pilot drift = uncontrolled b(z) (kappa = 0 scalar).  Overridden
        because the base mu_pilot passes a d-dim zero control, which does not
        match the scalar channel (HJBSparseLQUA's override lesson)."""
        return self._b(x)

    def u_star(self, x):
        """Scalar optimal control = q-projection of the full-dim reference
        u*_full = _lqsn_ref's (P, d) output (= -delta^2 grad v, ring-propagated
        at omega != 0):
            kappa* = q . u*_full
                   = -delta^2 (q^T e^{omega tau L^T} S^T) M (s_bar - a),
        i.e. the first pair-sum / sqrt(2) (HJBSparseLQUA.u_star at k = 1).
        """
        _, u_d = self._lqsn_ref(x)
        u_k = (u_d[:, 0] + u_d[:, 1]) / math.sqrt(2.)
        return u_k.unsqueeze(-1).to(x.dtype)

    def _u_der_to_control(self, u_der):
        """Project the d-dim derived control -delta^2 grad_z v_theta onto the
        single channel (q . u_der = (u_der[..., 0] + u_der[..., 1])/sqrt(2)),
        matching the scalar u_true in rel_l1err_ud[*].  c7a17bf lesson: the
        d-vs-k dim mismatch crashed log_func; the projection geometry must
        match u_star's exactly."""
        return ((u_der[..., 0] + u_der[..., 1]) / math.sqrt(2.)).unsqueeze(-1)

    def _lqsn_ref(self, x):
        """CORRECTED scalar-channel reference (owner 2026-09-24 directive,
        independently verified by k1_ref_check.py / SMU 537965, OVERALL PASS:
        new-form HJB residual <= 2e-14 at every (m, omega), closed-loop MC
        J(kappa*) == v within MC error, FFT-vs-dense-eig agreement 1e-15).

        The inherited SN chain solves the FULL-DIM-control problem; with a
        scalar channel the HJB minimization term is -delta^2 (q^T grad v)^2,
        so the quadratic denominator follows the CONTROL projection
            k(r) = q^T e^{omega r L^T} q,
        NOT the noise projection n(r) = |Q^T e^{r L^T} q|^2 (the old form's
        residual [n - k^2] (q^T e^{tau L} z - a)^2 / (1 + C)^2 is nonzero
        at omega != 0, m > 1 -- including m = d, refuting the old
        'full-rank noise is exact at any omega' comment; exact only at
        omega = 0 or m = 1, where n = k^2).  With h_r = e^{omega r L^T} q
        (q = the subclass's _pair_S row: pair direction here, mean/Fourier
        in K1M/K1F1):
            A_tau = 1 + 2 c delta^2 int_0^tau k(r)^2 dr,
            v      = c (h_tau . z - a)^2 / (2 A_tau)
                     + c delta^2 int_0^tau n(r) / A_r dr,
            kappa* = -c delta^2 k(tau) (h_tau . z - a) / A_tau.
        omega = 0 keeps the exact closed form (h_r = q, k = n = 1:
        A = 1 + 2 c delta^2 tau, G = 1/2 log A -- bit-compatible with the
        inherited omega = 0 route); omega != 0 uses the parent's trapezoid
        grid (n = max(64, 256 tau)) and the same memoization pattern.
        Returns (v (P, 1), u* (P, d) = kappa* q): the inherited u_star /
        u_star_term / eval chains keep their interfaces (K1.u_star's
        q-projection of kappa* q returns kappa* exactly)."""
        d = self.dim_z
        omega = float(self.omega)
        c = float(self.c_weights[0])
        delta = float(self.delta)
        a = float(self.sparse_a)
        device = x.device
        q = self._pair_S(torch.float64, device)[0]
        Q = self._sn_basis(torch.float64, device)
        t_all = x[:, 0].double()
        z_all = x[:, 1:].double()
        v_out = torch.empty(x.shape[0], 1, dtype=torch.float64,
                            device=device)
        u_out = torch.zeros(x.shape[0], d, dtype=torch.float64,
                            device=device)
        for t_val in torch.unique(t_all):
            mask = t_all == t_val
            tau = float(self.te - t_val)
            if not hasattr(self, '_k1_scalar_cache'):
                self._k1_scalar_cache = {}
            AG = self._k1_scalar_cache.get(round(tau, 12))
            if AG is None:
                if omega == 0.0:
                    A = 1. + 2. * c * delta ** 2 * tau
                    G = 0.5 * math.log(A)
                    h_tau = q
                else:
                    n = max(64, int(round(tau * 256)))
                    ds = tau / n
                    A, G, prev_k2 = 1.0, 0.0, None
                    for i in range(n + 1):
                        _pr, prT = self._omega_symbols(i * ds, d, device)
                        h = torch.fft.ifft(prT * torch.fft.fft(q)).real
                        kk = float(q @ h)
                        if i > 0:
                            A += c * delta ** 2 * (kk * kk + prev_k2) * ds
                        nn = float((Q.t() @ h).pow(2).sum())
                        wg = 0.5 * ds if (i == 0 or i == n) else ds
                        G += c * delta ** 2 * (nn / A) * wg
                        prev_k2 = kk * kk
                    _p, prT_tau = self._omega_symbols(tau, d, device)
                    h_tau = torch.fft.ifft(
                        prT_tau * torch.fft.fft(q)).real
                AG = (A, G, float(q @ h_tau), h_tau)
                self._k1_scalar_cache[round(tau, 12)] = AG
            A, G, k_tau, h_tau = AG
            y = z_all[mask] @ h_tau
            v_out[mask] = (c * (y - a) ** 2 / (2. * A) + G).unsqueeze(-1)
            kap = -c * delta ** 2 * k_tau * (y - a) / A
            u_out[mask] = kap.unsqueeze(-1) * q.unsqueeze(0)
        return v_out, u_out


class HJBSparseLQK1M(HJBSparseLQK1):
    """K1-mean: the k=1 scalar channel on the ALL-COORDINATES direction
    q = ones(d)/sqrt(d) (owner 2026-09-04, target campaign K1-mean; design
    approved with frozen parameters):

        dz = [ b(z) + 2 q kappa ] dt + sqrt(2) delta Q_m Q_m^T dW,
             kappa in R (scalar),   q = 1_d / sqrt(d),
        g(z) = c/2 (q . z - a)^2    (terminal sensitivity on the SAME q),
        f(x, kappa) = delta^{-2} kappa^2,
        sparse_m defaults to d (full-rank noise; INI sparse_m overrides).

    Design motivation (owner 2026-09-04): u must influence ALL d state
    coordinates (mu = b + 2 q kappa moves every coordinate by the same
    kappa/sqrt(d)) while the terminal gradient is UNIFORM on the input side
    (grad g = c (q.z - a) q -- no coordinate is privileged), replacing K1's
    single pair direction by the dense mean direction.  Intended as the
    d = 10^4 vehicle: the noise is full-rank BY DEFAULT (sparse_m = d) per
    the approved design.  Note (zero-check 512284 finding): because the
    inherited q-first _sn_basis places q itself as the FIRST noise
    direction, |P_m q|^2 = 1 for ANY m >= 1 in THIS class -- the often-cited
    "rank-m squeezes the mean-direction variance to ~m/d" collapse applies
    to non-adapted (plain DCT) bases only; the m < d INI override therefore
    changes the off-q noise geometry, not the signal variance (C = 2
    delta^2 tau |P_m q|^2 = 2 delta^2 tau exactly, measured C = 2.000000 at
    d = 1000, m = 20).

    Why the reference stays EXACT and the line restriction loses NOTHING
    (a STRONGER identity than K1's pair case): q is the zeroth Fourier
    mode of the ring drift, so e^{omega s L} q = q and q^T e^{omega s L}
    = q^T for any s, giving at m = d (any omega)
        mean:  y = q^T e^{omega tau L} z = q^T z,
        cov:   q^T Sigma q = mode-0 symbol value = 2 delta^2 tau,
    and v depends on z only through y -- hence grad v = (dv/dy) q lies ON
    the controlled line for ANY omega, so the scalar-channel value equals
    the full-dim Cole-Hopf value EXACTLY (no K1-style omega != 0
    line-restriction gap at m = d).  The reference is the k = 1 Gaussian
    convolution of the quadratic terminal, closed form:
        v      = 1/2 [ ln(1 + c C) + c (y - a)^2 / (1 + c C) ],
        kappa* = -delta^2 c (y - a) / (1 + c C),
    with C = 2 delta^2 tau at m = d (the exact mode-0 value, no quadrature
    and no FFT roundoff), C = 2 delta^2 tau |P_m q|^2 at omega = 0, m < d
    (the SN closed form SQ SQ^T), or the SN trapezoid at omega != 0, m < d
    (INI-override geometry only; there the reference inherits the family's
    shared-approximation caveat, K1c's 12% allowance).

    Inherited machinery: gain-2 pairing (mu = b + 2 q kappa, LQUA rescued
    lineage: kappa* = -delta^2 q^T grad v with no extra factor), dim_u = 1
    property + the k_dirs = 1 loud guard, the sparse_omega / sparse_delta /
    sparse_a / sparse_c INI keys, u_star_term (tau = 0 pin of u_star), the
    C(tau) memoization.  sigma at m = d returns sqrt(2) delta dw directly:
    the basis is orthonormal complete so Q Q^T = I exactly -- cheaper than
    the SN route's (d, d) matmul pair AND free of its QR roundoff.
    """

    @property
    def sparse_m(self):
        """Default noise dimension = d (full rank, the design): a property
        over a stored override so the INI injector's unconditional
        assignment keeps working (unset = dim_z; the SN 1..dim_z guard in
        _sn_params stays active for INI overrides)."""
        return getattr(self, '_k1m_m', None) or self.dim_z

    @sparse_m.setter
    def sparse_m(self, value):
        if value is not None:
            self._k1m_m = int(value)

    def _sens(self, z):
        """Sensitive coordinate s(z) = q . z = sum_i z_i / sqrt(d): (..., 1).
        The terminal g = c/2 (s - a)^2 is inherited from HJBSparseLQ via
        this override (pair average -> mean direction)."""
        return z.sum(-1, keepdim=True) / math.sqrt(self.dim_z)

    def grad_g(self, z):
        """grad g = c (q.z - a) q: EVERY coordinate carries the same
        c (s - a)/sqrt(d) (overrides HJBSparseLQ's pair-supported grad)."""
        c = torch.as_tensor(self.c_weights[:self.k_dirs], dtype=z.dtype,
                            device=z.device)
        s = self._sens(z)
        return ((c * (s - self.sparse_a)) / math.sqrt(self.dim_z)) * \
            torch.ones_like(z)

    def _pair_S(self, dtype, device):
        """Sensitive-direction matrix S (1, d) = q^T, the mean direction
        (overrides the pair rows so the inherited _sn_basis puts q FIRST in
        the noise basis; the SN S-generic wiring stays coherent at m < d)."""
        S = torch.full((1, self.dim_z), 1. / math.sqrt(self.dim_z),
                       dtype=torch.float64, device=device)
        return S.to(dtype)

    def _qC(self, tau, device):
        """Scalar projected covariance C = q^T Sigma(tau) q, float64:
            m == d (any omega): 2 delta^2 tau -- q is the Fourier zero
                mode, so q^T Sigma q is the mode-0 symbol value exactly
                (no quadrature, no FFT roundoff);
            omega == 0, m < d:  2 delta^2 tau |P_m q|^2 (SN closed form);
            omega != 0, m < d:  the SN trapezoid over B(s) = q^T e^{omega
                s L} Q_m, same grid and _lqsn_C_cache memoization as the
                parent (INI-override geometry only)."""
        d = self.dim_z
        if int(self._sn_params()) == d:
            return 2.0 * self.delta ** 2 * tau * \
                torch.ones((), dtype=torch.float64, device=device)
        if float(self.omega) == 0.0:
            Q = self._sn_basis(torch.float64, device)
            SQ = self._pair_S(torch.float64, device) @ Q    # (1, m)
            C = 2.0 * self.delta ** 2 * tau * (SQ @ SQ.t())
            return C.reshape(())                            # scalar ()
        if not hasattr(self, '_lqsn_C_cache'):
            self._lqsn_C_cache = {}
        C = self._lqsn_C_cache.get(round(tau, 12))
        if C is None:
            n = max(64, int(round(tau * 256)))
            ds = tau / n
            Q = self._sn_basis(torch.float64, device)       # (d, m)
            qrow = self._pair_S(torch.float64, device)      # (1, d)
            C = torch.zeros(1, 1, dtype=torch.float64, device=device)
            for s_i in range(n + 1):
                s = s_i * ds
                prop_s, _ = self._omega_symbols(s, d, device)
                eLQ = torch.fft.ifft(
                    prop_s.unsqueeze(1) * torch.fft.fft(
                        Q, n=d, dim=0), n=d, dim=0).real      # (d, m)
                B = qrow @ eLQ                               # (1, m)
                w = 1.0 if (s_i == 0 or s_i == n) else 2.0
                C = C + w * (B @ B.t())
            C = self.delta ** 2 * ds * C
            self._lqsn_C_cache[round(tau, 12)] = C
        return C.reshape(())

    def _lqsn_ref(self, x):
        """v and the SCALAR kappa* in one pass, float64, per time level
        (k = 1, S = q; see the class docstring for the zero-mode reduction).
        Returns (v (P, 1), kappa* (P, 1)) -- the scalar channel itself, not
        the d-dim u* of the SN chain (u*_full = kappa* q whenever needed).
        """
        assert x.ndim == 2
        c = float(self.c_weights[0])
        a = float(self.sparse_a)
        t_all = x[:, 0].double()
        z_all = x[:, 1:].double()
        v_out = torch.empty(x.shape[0], 1, dtype=torch.float64,
                            device=x.device)
        u_out = torch.empty(x.shape[0], 1, dtype=torch.float64,
                            device=x.device)
        for t_val in torch.unique(t_all):
            mask = t_all == t_val
            tau = float(self.te - t_val)
            # q^T e^{omega tau L} z = q^T z exactly (zero mode, any omega)
            y = z_all[mask].sum(-1, keepdim=True) / math.sqrt(self.dim_z)
            C = self._qC(tau, x.device)
            denom = 1. + c * C
            v_out[mask] = 0.5 * (torch.log(denom)
                                 + c * (y - a).pow(2) / denom)
            u_out[mask] = (-self.delta ** 2) * (c * (y - a) / denom)
        return v_out, u_out

    def sigma(self, x, dw):
        """dw stays d-dim.  At m = d the noise basis is orthonormal complete
        (Q Q^T = I exactly), so sigma dw = sqrt(2) delta dw -- returned
        directly (saves the SN route's (d, d) matmul pair at d = 1e3+ and
        avoids its QR roundoff).  m < d falls back to the SN route."""
        if int(self._sn_params()) == self.dim_z:
            return (2 ** 0.5) * self.delta * dw
        return super().sigma(x, dw)

    def mu(self, x, kappa):
        """Controlled drift b(z) + 2 q kappa with kappa in R (shape (..., 1)):
        the scalar channel moves EVERY coordinate by kappa/sqrt(d) (the
        u-influences-all-dimensions requirement), gain-2 pairing per the
        family convention."""
        full = (kappa / math.sqrt(self.dim_z)).expand_as(x[..., 1:])
        return self._b(x) + 2. * full

    def mu_pilot(self, x):
        """Pilot drift = uncontrolled b(z) (kappa = 0 scalar; inherited from
        K1, restated for the class record)."""
        return self._b(x)

    def u_star(self, x):
        """Scalar optimal control kappa* = -delta^2 q^T grad v.  grad v =
        (dv/dy) q with y = q^T z (zero mode), so the gain-2 pairing gives
        kappa* = -delta^2 dv/dy with no extra factor; closed form in
        _lqsn_ref."""
        return self._lqsn_ref(x)[1].to(x.dtype)

    def _u_der_to_control(self, u_der):
        """Project the d-dim derived control -delta^2 grad_z v_theta onto the
        scalar channel (q . u_der = sum_i u_der_i / sqrt(d)), matching the
        scalar u_true in rel_l1err_ud[*] (c7a17bf lesson: the projection
        geometry must match u_star's exactly)."""
        return (u_der.sum(-1) / math.sqrt(self.dim_z)).unsqueeze(-1)


class HJBSparseLQK1F1(HJBSparseLQK1):
    """K1-F1: the k=1 scalar channel on the FOURIER MODE f=1 direction
    (owner 2026-09-04, R2 arm B of the target campaign -- the direction-
    structure hypothesis test against the K1M d=1000 wall):

        dz = [ b(z) + 2 q kappa ] dt + sqrt(2) delta Q_m Q_m^T dW,
             kappa in R (scalar),
             q_j = sqrt(2/d) cos(2 pi j / d),  j = 0..d-1 (unit norm),
        g(z) = c/2 (q . z - a)^2    (terminal sensitivity on the SAME q),
        f(x, kappa) = delta^{-2} kappa^2,
        sparse_m defaults to d (full-rank noise; INI sparse_m overrides).

    Same design envelope as K1M (u influences ALL d coordinates through the
    cosine profile, uniform-magnitude terminal gradient, O(1) |u*|, m = d
    default): the ONLY structural delta is the coordinate PROFILE of q
    (cosine wave vs constant), so the d=1000 spread outcome cleanly
    attributes the K1M wall to the uniform/DC profile specifically (F1
    breaks the wall) or to aligning ANY fixed dense direction (F1 walls
    too).

    Exact mode-plane geometry (the correct statement of "the Fourier mode
    is an eigen-direction of the ring drift"): L = S - I with S the cyclic
    shift is circulant, so the COMPLEX mode f_1 is an eigen-direction with
    eigenvalue lam_1 = e^{i theta} - 1, theta = 2 pi/d; the REAL plane
    span{q, q~} (q~_j = sqrt(2/d) sin(theta j)) is L-invariant and
    e^{omega tau L^T} acts on it as the scaled rotation
        e^{omega tau L^T} q = alpha(tau) q + beta(tau) q~,
        alpha = e^{gam tau} cos(phi tau),  beta = e^{gam tau} sin(phi tau),
        gam = omega (cos theta - 1) <= 0,   phi = omega sin theta,
    (derived from the propT symbol; verified against the FFT propagation
    in the zero-check).  Consequences, at m = d for ANY omega:
        mean:  mu = q^T e^{omega tau L} z = alpha y_c + beta y_s
               (y_c = q . z, y_s = q~ . z -- TWO projections, not one),
        cov:   C = q^T Sigma q = the mode-1 symbol value
               = delta^2 expm1(2 gam tau) / gam  (-> 2 delta^2 tau at
               gam -> 0), closed form, no quadrature and no FFT roundoff,
        v      = 1/2 [ ln(1 + c C) + c (mu - a)^2 / (1 + c C) ],
        kappa* = -delta^2 alpha c (mu - a) / (1 + c C)
               = q-projection of the FULL-DIM u* (u*_full = kappa* q
               - delta^2 beta (dv/dmu) q~).
    UNLIKE K1M's zero-mode identity, grad v is NOT on the controlled line
    when beta != 0 (omega != 0): the scalar-channel problem's value is
    strictly above the full-dim value by an O(beta^2) line-restriction
    gap -- the K1 omega != 0 situation, with beta playing the pair-case
    drift-rotation role.  At the training geometry (d = 1000, omega = 1,
    tau <= 1) beta <= 6.3e-3 so the gap is ~4e-5 relative, far below the
    5% Euler floor of the DPP check; the small-d zero-check rows (d = 20,
    beta(1) = 0.29) use K1c's 12% allowance.  The class reference is the
    FULL-DIM Cole-Hopf value (exact); kappa* is its optimal projection.

    m < d INI-override rows: omega = 0 gives beta = 0 identically so the
    K1M-style closed form survives with C = 2 delta^2 tau |P_m q|^2
    (= 2 delta^2 tau exactly: the q-first _sn_basis puts q as the FIRST
    noise direction); omega != 0 integrates C by trapezoid with the
    CLOSED-FORM propagated row w(s) = alpha(s) q + beta(s) q~ (no FFT in
    the loop), same grid and _lqsn_C_cache memoization as the parent.

    Inherited machinery (re-implemented here rather than via K1M to keep
    the K1 base the task letter specifies): the sparse_m property with
    default d (full rank, the design; INI override slot _k1f1_m), sigma
    at m = d returning sqrt(2) delta dw directly (the basis is
    orthonormal complete so Q Q^T = I exactly -- cheaper than the SN
    route's (d, d) matmul pair AND free of its QR roundoff), gain-2 mu =
    b + 2 q kappa with the cosine profile, dim_u = 1 + k_dirs = 1 loud
    guard, u_star_term (tau = 0 pin: alpha = 1, beta = 0, C = 0, so
    u_T = -delta^2 c (y_c - a)), _u_der_to_control = q-projection.
    """

    @property
    def sparse_m(self):
        """Default noise dimension = d (full rank, the design): a property
        over a stored override so the INI injector's unconditional
        assignment keeps working (unset = dim_z; the SN 1..dim_z guard in
        _sn_params stays active for INI overrides)."""
        return getattr(self, '_k1f1_m', None) or self.dim_z

    @sparse_m.setter
    def sparse_m(self, value):
        if value is not None:
            self._k1f1_m = int(value)

    def _f1_vecs(self, device):
        """q (cos mode f=1) and q~ (sin mode), both (d,) float64, unit
        norm, mutually orthogonal; lazily cached per (d, device)."""
        key = (self.dim_z, str(device))
        cache = getattr(self, '_f1_vecs_cache', None)
        if cache is None or cache[0] != key:
            jj = torch.arange(self.dim_z, dtype=torch.float64, device=device)
            th = 2. * math.pi * jj / self.dim_z
            cache = (key, (2. / self.dim_z) ** 0.5 * torch.cos(th),
                     (2. / self.dim_z) ** 0.5 * torch.sin(th))
            self._f1_vecs_cache = cache
        return cache[1], cache[2]

    def _f1_rates(self):
        """(gam, phi): exponential rate and in-plane rotation rate of the
        mode-1 plane under e^{omega s L} (float pair)."""
        th = 2. * math.pi / self.dim_z
        om = float(self.omega)
        return om * (math.cos(th) - 1.), om * math.sin(th)

    def _sens(self, z):
        """Sensitive coordinate s(z) = q . z (cos-mode projection): (..., 1).
        The terminal g = c/2 (s - a)^2 is inherited from HJBSparseLQ via
        this override (mean direction -> cos mode)."""
        q, _ = self._f1_vecs(z.device)
        return (z.double() @ q).to(z.dtype).unsqueeze(-1)

    def grad_g(self, z):
        """grad g = c (q.z - a) q: EVERY coordinate carries c (s - a)
        q_j = c (s - a) sqrt(2/d) cos(2 pi j / d) (uniform-MAGNITUDE,
        cosine-profile direction; overrides HJBSparseLQ's pair grad)."""
        c = torch.as_tensor(self.c_weights[:self.k_dirs], dtype=z.dtype,
                            device=z.device)
        s = self._sens(z)
        q, _ = self._f1_vecs(z.device)
        return (c * (s - self.sparse_a)) * q.to(z.dtype)

    def _pair_S(self, dtype, device):
        """Sensitive-direction matrix S (1, d) = q^T, the cos mode
        (overrides the pair rows so the inherited _sn_basis puts q FIRST
        in the noise basis; the SN S-generic wiring stays coherent at
        m < d)."""
        q, _ = self._f1_vecs(device)
        return q.to(dtype).unsqueeze(0)

    def _qC(self, tau, device):
        """Scalar projected covariance C = q^T Sigma(tau) q, float64:
            m == d (any omega): the mode-1 symbol value
                delta^2 expm1(2 gam tau) / gam   (2 delta^2 tau if
                gam == 0, e.g. omega = 0) -- closed form, no quadrature;
            omega == 0, m < d:  2 delta^2 tau |P_m q|^2 (SN closed form;
                = 2 delta^2 tau exactly, q-first basis);
            omega != 0, m < d:  trapezoid over w(s)^T Q_m with the
                CLOSED-FORM row w(s) = alpha(s) q + beta(s) q~ (no FFT
                in the loop), same grid and _lqsn_C_cache memoization as
                the parent (INI-override geometry only)."""
        d = self.dim_z
        gam, phi = self._f1_rates()
        if int(self._sn_params()) == d:
            if abs(gam) < 1e-14:
                val = 2.0 * self.delta ** 2 * tau
            else:
                val = self.delta ** 2 * math.expm1(2. * gam * tau) / gam
            return torch.full((), val, dtype=torch.float64, device=device)
        if float(self.omega) == 0.0:
            Q = self._sn_basis(torch.float64, device)
            SQ = self._pair_S(torch.float64, device) @ Q    # (1, m)
            return (2.0 * self.delta ** 2 * tau * (SQ @ SQ.t())).reshape(())
        if not hasattr(self, '_lqsn_C_cache'):
            self._lqsn_C_cache = {}
        C = self._lqsn_C_cache.get(round(tau, 12))
        if C is None:
            n = max(64, int(round(tau * 256)))
            ds = tau / n
            q, qt = self._f1_vecs(device)
            Q = self._sn_basis(torch.float64, device)       # (d, m)
            acc = 0.
            for s_i in range(n + 1):
                s = s_i * ds
                es = math.exp(gam * s)
                wrow = es * math.cos(phi * s) * q + \
                    es * math.sin(phi * s) * qt              # (d,)
                B = wrow @ Q                                 # (m,)
                w = 1.0 if (s_i == 0 or s_i == n) else 2.0
                acc += w * float(B @ B)
            C = torch.full((), self.delta ** 2 * ds * acc,
                           dtype=torch.float64, device=device)
            self._lqsn_C_cache[round(tau, 12)] = C
        return C.reshape(())

    def _lqsn_ref(self, x):
        """v and the SCALAR kappa* in one pass, float64, per time level
        (k = 1, S = q; see the class docstring for the mode-plane
        reduction).  v is the FULL-DIM Cole-Hopf value (exact at m = d);
        kappa* is the q-projection of the full-dim u*.  Returns (v (P, 1),
        kappa* (P, 1))."""
        assert x.ndim == 2
        c = float(self.c_weights[0])
        a = float(self.sparse_a)
        gam, phi = self._f1_rates()
        q, qt = self._f1_vecs(x.device)
        t_all = x[:, 0].double()
        z_all = x[:, 1:].double()
        v_out = torch.empty(x.shape[0], 1, dtype=torch.float64,
                            device=x.device)
        u_out = torch.empty(x.shape[0], 1, dtype=torch.float64,
                            device=x.device)
        for t_val in torch.unique(t_all):
            mask = t_all == t_val
            tau = float(self.te - t_val)
            es = math.exp(gam * tau)
            al = es * math.cos(phi * tau)
            be = es * math.sin(phi * tau)
            z = z_all[mask]
            mu = al * (z @ q).unsqueeze(-1) + be * (z @ qt).unsqueeze(-1)
            C = self._qC(tau, x.device)
            denom = 1. + c * C
            v_out[mask] = 0.5 * (torch.log(denom)
                                 + c * (mu - a).pow(2) / denom)
            u_out[mask] = (-self.delta ** 2) * al * (c * (mu - a) / denom)
        return v_out, u_out

    def sigma(self, x, dw):
        """dw stays d-dim.  At m = d the noise basis is orthonormal complete
        (Q Q^T = I exactly), so sigma dw = sqrt(2) delta dw -- returned
        directly (saves the SN route's (d, d) matmul pair at d = 1e3+ and
        avoids its QR roundoff).  m < d falls back to the SN route."""
        if int(self._sn_params()) == self.dim_z:
            return (2 ** 0.5) * self.delta * dw
        return super().sigma(x, dw)

    def mu(self, x, kappa):
        """Controlled drift b(z) + 2 q kappa with kappa in R (shape (..., 1)):
        the scalar channel moves EVERY coordinate by kappa q_j (cosine
        profile -- the u-influences-all-dimensions requirement), gain-2
        pairing per the family convention."""
        q, _ = self._f1_vecs(x.device)
        full = (kappa.to(torch.float64) * q).to(x.dtype)
        return self._b(x) + 2. * full

    def mu_pilot(self, x):
        """Pilot drift = uncontrolled b(z) (kappa = 0 scalar; inherited
        from K1M via K1, restated for the class record)."""
        return self._b(x)

    def u_star(self, x):
        """Scalar optimal control kappa* = q-projection of the full-dim
        u* = -delta^2 grad v (see _lqsn_ref); closed form in _lqsn_ref."""
        return self._lqsn_ref(x)[1].to(x.dtype)

    def _u_der_to_control(self, u_der):
        """Project the d-dim derived control -delta^2 grad_z v_theta onto
        the scalar channel (q . u_der), matching the scalar u_true in
        rel_l1err_ud[*] (c7a17bf lesson: the projection geometry must
        match u_star's exactly)."""
        q, _ = self._f1_vecs(u_der.device)
        return (u_der.double() @ q).to(u_der.dtype).unsqueeze(-1)


class HJBSparseLQD1(HJBSparseLQSN):
    """LQ-D1: d-dimensional control output with a SINGLE active coordinate
    (owner directive 2026-09-13, design approved verbatim): the K1-pair
    lineage with dim_u = d restored (U = R^d, the sparse family's native
    control space) and the terminal sensitive direction moved from the PAIR
    (e_0+e_1)/sqrt(2) to the SINGLE coordinate e_0.

        dz = [ b(z) + 2 u ] dt + sqrt(2) delta Q_m Q_m^T dW,  u in R^d,
        g(z) = c/2 (z_0 - a)^2,      f = delta^{-2} |u|^2,
        sparse_omega = 0 (owner ruling: exact single-coordinate support; at
        omega != 0 the ring propagation e^{omega tau L^T} spreads u* onto
        neighbouring coordinates with comparable weight at tau ~ 1, which
        would violate "exactly one coordinate far from zero").

    Closed form (omega = 0, exact for any m >= 1 because the pair-first
    noise basis puts e_0 as the FIRST noise direction, so C = 2 delta^2 tau
    exactly and the m < d shared-approximation caveat does not apply):
        v = 1/2 [ ln(1 + 2 c delta^2 tau) + c (z_0 - a)^2 / (1 + 2 c delta^2 tau) ],
        u*(t, z) = -delta^2 c (z_0 - a)/(1 + 2 c delta^2 tau) * e_0,
    i.e. a d-dimensional optimal feedback with exactly ONE O(1) component
    (index 0); the remaining d-1 components are EXACTLY zero.  The active/
    background metric split is automatic: mask_act = |u*| >= 5% max marks
    coordinate 0 only, so rel_l1err_u_act is the active-coordinate error
    and abserr_u_bg is the d-1 degenerate coordinates' error background
    (preregistered gates, owner 2026-09-13: u_act <= 8e-2, bg <= 0.01).

    Implementation: ONLY the sensitive-direction geometry changes --
    _sens (terminal g routes through it), grad_g, and _pair_S (the
    reference chain's S matrix, which also routes the noise basis);
    k_dirs is pinned to 1 with the K1-style loud guard and omega to 0.
    mu / mu_pilot / dim_u (= dim_x - 1 = d) / u_star / u_star_term /
    _u_der_to_control (identity) are all inherited unchanged.
    """

    @property
    def k_dirs(self):
        """k is pinned to 1 (single sensitive coordinate); sparse_k must
        NOT be set."""
        return 1

    @k_dirs.setter
    def k_dirs(self, value):
        if int(value) != 1:
            raise ValueError(
                "HJBSparseLQD1 is the single-coordinate design: k_dirs is "
                f"pinned to 1; got k_dirs/sparse_k = {value!r} (do not set "
                "sparse_k)")

    @property
    def omega(self):
        """omega must be 0 in this design (owner ruling 2026-09-13): the
        ring propagation would spread u* off e_0.  Property + guard turns a
        nonzero sparse_omega into a loud error instead of a silently wrong
        reference."""
        return 0.0

    @omega.setter
    def omega(self, value):
        if float(value) != 0.0:
            raise ValueError(
                "HJBSparseLQD1 requires sparse_omega = 0 (owner ruling "
                "2026-09-13): at omega != 0 the ring propagation spreads "
                "u* onto neighbouring coordinates and the single-coordinate "
                "design is violated.")

    def _sens(self, z):
        """Sensitive coordinate s = z_0 (single coordinate, shape (..., 1))."""
        return z[..., [0]]

    def _pair_S(self, dtype, device):
        """Sensitive-direction matrix S (1, d) = e_0^T: also routes the
        pair-first noise basis to put e_0 FIRST (C = 2 delta^2 tau exact
        for any m >= 1)."""
        S = torch.zeros(1, self.dim_z, dtype=torch.float64, device=device)
        S[0, 0] = 1.0
        return S.to(dtype)

    def u_star(self, x):
        """d-dim optimal control u* = -delta^2 c (z_0 - a)/(1 + c C) e_0,
        C = 2 delta^2 tau |S Q_m|^2 (= 2 delta^2 tau exactly: e_0 is the
        FIRST noise-basis column).

        OVERRIDE RATIONALE (zero-check D1a/D1b, 2026-09-13): the parent's
        _lqsn_ref routes v / mean / covariance through _pair_S (all exact
        for this class) but ASSEMBLES u* with the PAIR geometry hardcoded
        -- -delta^2 M(s-a)/sqrt(2) written into BOTH pair coordinates --
        which puts half the mass on coordinate 1 for the single-coordinate
        design.  The parent is bitwise-frozen for the pair family, so the
        correct rank-1 assembly lives here.  omega = 0 only (guarded)."""
        c = float(self.c_weights[0])
        a = self.sparse_a
        dd = self.delta
        tau = self.te - x[:, 0].double()
        z0 = x[:, 1].double()
        Q = self._sn_basis(torch.float64, x.device)
        S = self._pair_S(torch.float64, x.device)
        SQ = S @ Q                                          # (1, m)
        snorm2 = (SQ @ SQ.t()).item()                       # = 1 exactly
        C = 2.0 * dd * dd * tau * snorm2
        u0 = -dd * dd * c * (z0 - a) / (1.0 + c * C)
        out = torch.zeros(x.shape[0], self.dim_z,
                          dtype=torch.float64, device=x.device)
        out[:, 0] = u0
        return out.to(x.dtype)

    def grad_g(self, z):
        """grad g = c (z_0 - a) e_0 (entry at coordinate 0 only)."""
        grad = torch.zeros_like(z)
        grad[..., 0] = float(self.c_weights[0]) * (z[..., 0] - self.sparse_a)
        return grad


class HJBSparseLQK5F(HJBSparseLQK1):
    """K5-F: the k=5 channel control on the FIRST FIVE cosine Fourier modes
    (owner 2026-09-05, R6 main hypothesis of the target campaign /goal
    d=1e4 spread RE<3e-2): a rank-5 multimode terminal that denies the v
    network the single-direction shortcut (the K1M disease, only partially
    relieved by F1) while keeping a low-dimensional control output and the
    closed-form reference chain.

        dz = [ b(z) + 2 B kappa ] dt + sqrt(2) delta Q_m Q_m^T dW,
             kappa in R^5 (five scalar channels),
             B = [q_1 ... q_5] in R^{d x 5},
             (q_f)_j = sqrt(2/d) cos(2 pi f j / d),  f = 1..5,
        g(z) = 1/2 sum_{f=1}^{5} c_f (q_f . z - a)^2   (c_f = 1, a = 1),
        f(x, kappa) = delta^{-2} |kappa|^2,
        sparse_m defaults to d (full-rank noise; INI sparse_m overrides).

    Channel orthogonality is EXACT, no Gram-Schmidt needed: for f, g in
    1..5 and d >= 11, sum_j cos(2pi f j/d) cos(2pi g j/d) = (d/2)[f = g]
    because f +- g in 1..10 < d (discrete Fourier orthogonality; the
    zero-check K5a0 verifies |B^T B - I| < 1e-12 before anything else).

    Exact mode-plane geometry (K1F1's per-mode statement, five copies):
    each REAL plane span{q_f, q~_f} (q~_f = sqrt(2/d) sin(2 pi f j/d)) is
    L-invariant, and with
        gamma_f = omega (cos theta_f - 1) <= 0,  phi_f = omega sin theta_f,
        theta_f = 2 pi f / d,
    the propagator acts as e^{omega s L^T} q_f = alpha_f(s) q_f +
    beta_f(s) q~_f with alpha_f = e^{gamma_f s} cos(phi_f s), beta_f =
    e^{gamma_f s} sin(phi_f s).  Consequences at m = d for ANY omega:
        mean:   y_f = q_f^T e^{omega tau L} z = alpha_f y_c,f + beta_f y_s,f
                (y_c = q_f . z, y_s = q~_f . z -- two projections per mode),
        cov:    C = S Sigma S^T is DIAGONAL, C_ff = the mode-f symbol value
                = delta^2 expm1(2 gamma_f tau) / gamma_f (-> 2 delta^2 tau
                at gamma_f -> 0), because Sigma's circulant symbol is real
                and each {q_f, q~_f} plane is a Sigma-eigenspace,
        v       = 1/2 (logdet(I + C Lam) + (y - a)^T M (y - a)),
                M = (Lam^-1 + C)^-1   (diagonal: the product form),
        kappa*_f = -delta^2 alpha_f [M (y - a)]_f
                = the f-th channel of B^T u*_full (u*_full = -delta^2
                e^{omega tau L^T} grad v; the q~_f components of grad v
                are invisible to the channel projection).
    As in K1F1, grad v is NOT contained in the channel span when beta != 0
    (omega != 0): the rank-5 problem's value is strictly above the full-dim
    value by an O(sum_f beta_f^2) line-restriction gap -- beta_f(1) =
    6.3e-3 .. 3.1e-2 at the training geometry (d = 1000, omega = 1) puts
    the gap near 1e-3 relative, below the 5% Euler floor of the DPP check;
    the d = 20 rows (beta up to 0.31) use K1c's 12% allowance.  The class
    reference is the FULL-DIM Cole-Hopf value (exact); kappa* is its
    optimal channel projection.

    m < d INI-override rows: omega = 0 gives beta = 0 identically so the
    closed form survives with C = 2 delta^2 tau (S Q_m)(S Q_m)^T (= 2
    delta^2 tau I_5 whenever m >= 5: the q-first _sn_basis places the five
    channels FIRST in the noise basis); omega != 0 integrates C by
    trapezoid over the CLOSED-FORM rows w_f(s) = alpha_f(s) q_f +
    beta_f(s) q~_f (no FFT in the loop), same grid and _lqsn_C_cache
    memoization as the parent (INI-override geometry only; there the
    sin-mode projections on the DCT fill-in make C genuinely non-diagonal,
    handled by the general k-dim matrix route in _lqsn_ref).

    Design rationale (the R6 hypothesis): the pure rank-1 terminal lets v
    collapse to a function of one scalar projection -- one first-layer
    direction aligned suffices, and d = 1000 networks find such shortcuts
    (K1M 0.798 wall; F1's cosine profile only partly denies it, 0.092
    ceiling).  The rank-5 terminal REQUIRES v to represent five
    simultaneously correct modal responses before the spread metric
    improves -- the shortcut class is removed while dim_u stays 5 (no
    output dilution) and every coordinate still carries O(sqrt(2/d))
    terminal-gradient signal (dense channels, |u*| = O(1) per component).

    Inherited machinery (re-implemented here on the K1 base, the task
    letter's parent): the sparse_m property with default d (INI override
    slot _k5f_m), sigma at m = d returning sqrt(2) delta dw directly
    (orthonormal complete basis, Q Q^T = I exactly), gain-2 mu = b + 2 B
    kappa, dim_u = k_dirs = 5 property with the loud sparse_k guard,
    u_star_term (tau = 0 pin: alpha = 1, beta = 0, C = 0, so u_T = -delta^2
    c . (y_c - a)), _u_der_to_control = B-projection, the C(tau)
    memoization.  INI: sparse_c carries the FIVE weights c_f (comma-
    separated; class default (1,)*5 = the frozen design c_f = 1).
    """

    # the frozen R6 design: c_f = 1.0 for all five modes (class default;
    # the INI sets the same five values explicitly).
    c_weights = (1.0, 1.0, 1.0, 1.0, 1.0)

    @property
    def k_dirs(self):
        """k is pinned to 5 (the five cos modes f = 1..5); sparse_k must
        NOT be set."""
        return 5

    @k_dirs.setter
    def k_dirs(self, value):
        # parse_problem assigns problem.k_dirs = config.getint('sparse_k',
        # fallback=class value) whenever the attribute exists -- the guard
        # turns a wrong sparse_k in the INI into a loud error instead of a
        # silently inconsistent terminal/control split (K1's pattern).
        if int(value) != 5:
            raise ValueError(
                "HJBSparseLQK5F is the k=5 five-Fourier-mode design: "
                "k_dirs is pinned to 5; got k_dirs/sparse_k = "
                f"{value!r} (do not set sparse_k)")

    @property
    def sparse_m(self):
        """Default noise dimension = d (full rank, the design): a property
        over a stored override so the INI injector's unconditional
        assignment keeps working (unset = dim_z; the SN 1..dim_z guard in
        _sn_params stays active for INI overrides)."""
        return getattr(self, '_k5f_m', None) or self.dim_z

    @sparse_m.setter
    def sparse_m(self, value):
        if value is not None:
            self._k5f_m = int(value)

    def _k5f_vecs(self, device):
        """Channel rows q (cos modes f = 1..5) and the companion sin rows
        q~, both (5, d) float64, mutually orthonormal / orthogonal; lazily
        cached per (d, device).  Orthogonality is exact for d >= 11
        (f +- g in 1..10 < d); asserted once at build time so a degenerate
        d fails loudly here, not silently in the reference."""
        key = (self.dim_z, str(device))
        cache = getattr(self, '_k5f_vecs_cache', None)
        if cache is None or cache[0] != key:
            d = self.dim_z
            if d < 11:
                raise ValueError(
                    f"HJBSparseLQK5F needs d >= 11 for the five cos modes "
                    f"to be orthogonal (got d = {d})")
            jj = torch.arange(d, dtype=torch.float64, device=device)
            fs = torch.arange(1, 6, dtype=torch.float64, device=device)
            th = 2. * math.pi * fs.unsqueeze(-1) * jj / d   # (5, d)
            q = (2. / d) ** 0.5 * torch.cos(th)
            qt = (2. / d) ** 0.5 * torch.sin(th)
            gram = q @ q.t()
            dev = (gram - torch.eye(5, dtype=torch.float64,
                                    device=device)).abs().max().item()
            assert dev < 1e-12, \
                f"cos modes f=1..5 not orthogonal at d={d} ({dev:.2e})"
            cache = (key, q, qt)
            self._k5f_vecs_cache = cache
        return cache[1], cache[2]

    def _k5f_rates(self):
        """(gamma (5,), phi (5,)): per-mode exponential and rotation rates
        under e^{omega s L} (float tuples, f = 1..5)."""
        d = self.dim_z
        om = float(self.omega)
        th = [2. * math.pi * f / d for f in range(1, 6)]
        return (tuple(om * (math.cos(t) - 1.) for t in th),
                tuple(om * math.sin(t) for t in th))

    def _sens(self, z):
        """Sensitive coordinates s_f(z) = q_f . z, f = 1..5: (..., 5).
        The terminal g = 1/2 sum_f c_f (s_f - a)^2 is inherited from
        HJBSparseLQ via this override (pair averages -> five cos-mode
        projections)."""
        q, _ = self._k5f_vecs(z.device)
        return (z.double() @ q.t()).to(z.dtype)

    def grad_g(self, z):
        """grad g = sum_f c_f (q_f . z - a) q_f: every coordinate carries a
        five-mode cosine profile (overrides HJBSparseLQ's pair grad)."""
        c = torch.as_tensor(self.c_weights[:self.k_dirs], dtype=z.dtype,
                            device=z.device)
        s = self._sens(z)
        q, _ = self._k5f_vecs(z.device)
        return (c * (s - self.sparse_a)) @ q.to(z.dtype)

    def _pair_S(self, dtype, device):
        """Sensitive-direction matrix S (5, d) = the five cos-mode rows
        (overrides the pair rows so the inherited _sn_basis puts the five
        channels FIRST in the noise basis; the SN S-generic wiring stays
        coherent at m < d)."""
        q, _ = self._k5f_vecs(device)
        return q.to(dtype)

    def _qC(self, tau, device):
        """Projected covariance C = S Sigma(tau) S^T, (5, 5) float64:
            m == d (any omega): DIAGONAL, C_ff = the mode-f symbol value
                delta^2 expm1(2 gamma_f tau) / gamma_f  (2 delta^2 tau if
                gamma_f == 0, e.g. omega = 0) -- closed form, no quadrature
                (Sigma's real circulant symbol makes each {q_f, q~_f} plane
                an eigenspace);
            omega == 0, m < d:  2 delta^2 tau (S Q_m)(S Q_m)^T (SN closed
                form; = 2 delta^2 tau I_5 whenever m >= 5, q-first basis);
            omega != 0, m < d:  trapezoid over the rows w_f(s) = alpha_f(s)
                q_f + beta_f(s) q~_f (CLOSED-FORM propagation, no FFT in
                the loop; C is genuinely non-diagonal there through the
                sin-mode projections on the DCT fill-in), same grid and
                _lqsn_C_cache memoization as the parent (INI-override
                geometry only)."""
        d = self.dim_z
        gam, phi = self._k5f_rates()
        if int(self._sn_params()) == d:
            diag = []
            for g in gam:
                if abs(g) < 1e-14:
                    diag.append(2.0 * self.delta ** 2 * tau)
                else:
                    diag.append(self.delta ** 2 * math.expm1(2. * g * tau) / g)
            return torch.diag(torch.tensor(diag, dtype=torch.float64,
                                           device=device))
        if float(self.omega) == 0.0:
            Q = self._sn_basis(torch.float64, device)
            SQ = self._pair_S(torch.float64, device) @ Q    # (5, m)
            C = 2.0 * self.delta ** 2 * tau * (SQ @ SQ.t())
            return 0.5 * (C + C.t())
        if not hasattr(self, '_lqsn_C_cache'):
            self._lqsn_C_cache = {}
        C = self._lqsn_C_cache.get(round(tau, 12))
        if C is None:
            n = max(64, int(round(tau * 256)))
            ds = tau / n
            q, qt = self._k5f_vecs(device)
            Q = self._sn_basis(torch.float64, device)       # (d, m)
            C = torch.zeros(5, 5, dtype=torch.float64, device=device)
            for s_i in range(n + 1):
                s = s_i * ds
                W = torch.empty(5, d, dtype=torch.float64, device=device)
                for f in range(5):
                    es = math.exp(gam[f] * s)
                    W[f] = es * math.cos(phi[f] * s) * q[f] + \
                        es * math.sin(phi[f] * s) * qt[f]
                Bm = W @ Q                                     # (5, m)
                w = 1.0 if (s_i == 0 or s_i == n) else 2.0
                C = C + w * (Bm @ Bm.t())
            C = self.delta ** 2 * ds * C
            C = 0.5 * (C + C.t())
            self._lqsn_C_cache[round(tau, 12)] = C
        return C

    def _lqsn_ref(self, x):
        """v and the FIVE-channel kappa* in one pass, float64, per time
        level (k = 5, S = the five cos rows; see the class docstring for
        the mode-plane reduction).  v is the FULL-DIM Cole-Hopf value
        (exact at m = d); kappa* is the channel projection of the full-dim
        u*.  Returns (v (P, 1), kappa* (P, 5))."""
        assert x.ndim == 2
        c = torch.as_tensor(self.c_weights[:5], dtype=torch.float64,
                            device=x.device)
        a = float(self.sparse_a)
        gam, phi = self._k5f_rates()
        q, qt = self._k5f_vecs(x.device)
        t_all = x[:, 0].double()
        z_all = x[:, 1:].double()
        v_out = torch.empty(x.shape[0], 1, dtype=torch.float64,
                            device=x.device)
        u_out = torch.empty(x.shape[0], 5, dtype=torch.float64,
                            device=x.device)
        for t_val in torch.unique(t_all):
            mask = t_all == t_val
            tau = float(self.te - t_val)
            es = torch.tensor([math.exp(gam[f] * tau) for f in range(5)],
                              dtype=torch.float64, device=x.device)
            al_t = es * torch.tensor([math.cos(phi[f] * tau)
                                      for f in range(5)],
                                     dtype=torch.float64, device=x.device)
            be_t = es * torch.tensor([math.sin(phi[f] * tau)
                                      for f in range(5)],
                                     dtype=torch.float64, device=x.device)
            z = z_all[mask]
            y = al_t * (z @ q.t()) + be_t * (z @ qt.t())    # (P, 5) means
            C = self._qC(tau, x.device)                     # (5, 5)
            Sm = torch.diag(1. / c) + C
            Sm = 0.5 * (Sm + Sm.t())
            sol = torch.linalg.solve(Sm, (y - a).t()).t()   # M (y - a)
            Lc = torch.linalg.cholesky(Sm)
            logdet = c.log().sum() + 2 * torch.log(torch.diagonal(Lc)).sum()
            v_out[mask] = 0.5 * (logdet
                                 + ((y - a) * sol).sum(-1, keepdim=True))
            u_out[mask] = (-self.delta ** 2) * al_t * sol
        return v_out, u_out

    def sigma(self, x, dw):
        """dw stays d-dim.  At m = d the noise basis is orthonormal complete
        (Q Q^T = I exactly), so sigma dw = sqrt(2) delta dw -- returned
        directly (saves the SN route's (d, d) matmul pair at d = 1e3+ and
        avoids its QR roundoff).  m < d falls back to the SN route."""
        if int(self._sn_params()) == self.dim_z:
            return (2 ** 0.5) * self.delta * dw
        return super().sigma(x, dw)

    def mu(self, x, kappa):
        """Controlled drift b(z) + 2 B kappa with kappa in R^5 (shape
        (..., 5)): channel f moves every coordinate by kappa_f q_f,j (a
        five-mode cosine profile -- the u-influences-all-dimensions
        requirement), gain-2 pairing per the family convention."""
        q, _ = self._k5f_vecs(x.device)
        full = (kappa.to(torch.float64) @ q).to(x.dtype)
        return self._b(x) + 2. * full

    def mu_pilot(self, x):
        """Pilot drift = uncontrolled b(z) (kappa = 0 in R^5; inherited from
        K1, restated for the class record)."""
        return self._b(x)

    def u_star(self, x):
        """Five-channel optimal control kappa*_f = f-th component of B^T
        u*_full = -delta^2 B^T e^{omega tau L^T} grad v (see _lqsn_ref);
        closed form in _lqsn_ref."""
        return self._lqsn_ref(x)[1].to(x.dtype)

    def _u_der_to_control(self, u_der):
        """Project the d-dim derived control -delta^2 grad_z v_theta onto
        the five channels (B^T u_der: u_der @ q^T), matching the 5-dim
        u_true in rel_l1err_ud[*] (c7a17bf lesson: the projection geometry
        must match u_star's exactly)."""
        q, _ = self._k5f_vecs(u_der.device)
        return (u_der.double() @ q.t()).to(u_der.dtype)


class _SparsePathMC:
    """Path Monte Carlo reference for the sine drift (variant N), on a finer
    time grid dt_ref (owner instruction 2026-08-19: dt_ref = 1/100).

    v(t, x) = -ln E[exp(-g(X_T^0))] by Euler-Maruyama at dt_ref.
    u*(t, x) = -delta^2 E[e^{-g} grad g(X_T)^T J_T] / E[e^{-g}], where
    J_T = D F_{N-1} ... D F_0 is the pathwise Jacobian of the EM map; only
    the 2k rows of J_T rooted at the active terminal coordinates are needed.
    Row l_r of J_T is a left-multiplied product and must be propagated
    BACKWARD in time: rho^{(N)} = e_{l_r}^T, rho^{(n)} = rho^{(n+1)} (I +
    dt J_n). Since the drift Jacobian J_n has a single cyclic super-diagonal
    entry c_i = cos(t_n + i + z_{i+1}) at (i, i+1), each row's support grows
    by one coordinate per step (windowed recurrence, O(k * N_ref) per path
    instead of O(d^2)); the forward pass stores the O(N_ref * d) cos factors
    per path for the backward recurrence.
    """

    dt_ref = 1 / 100
    max_mc_steps = 10**6

    def __init__(self, problem):
        self.problem = problem
        # dt_ref INI override (owner 2026-08-24, d=1e2 ladder): small d needs
        # n_steps = T/dt_ref < d for the windowed tangent rows; None -> 1/100
        self.dt_ref = getattr(problem, 'dt_ref', None) or type(self).dt_ref

    def _drift_and_jac(self, t, z):
        """b(t, z) and the cyclic factor c_i = cos(t + i + z_{i+1})."""
        problem = self.problem
        i_idx = torch.arange(problem.dim_z, dtype=z.dtype, device=z.device)
        arg = t + i_idx + z.roll(-1, dims=-1)
        return torch.sin(arg), torch.cos(arg)

    def _samp_chunk(self, x, num_mc, with_tangent):
        """Simulate num_mc paths per point at dt_ref; return terminal z, and
        with with_tangent also the stored cos-factor history
        c_hist[p, pt, n, :] and the window coordinate map (l_r + w) mod d."""
        problem = self.problem
        device = x.device
        dt_ref = self.dt_ref
        P = x.shape[0]
        d = problem.dim_z
        assert torch.allclose(x[:, 0], x[0, 0]), \
            "path-MC reference requires a single time level per call"
        z = x[:, 1:].unsqueeze(0).expand(num_mc, P, d).clone()
        t = float(x[0, 0])
        n_steps = round(float(problem.te - t) / dt_ref)
        if n_steps * dt_ref < float(problem.te - t) - 1e-12:
            n_steps += 1  # never truncate the horizon
        c_hist = None
        win_coord = None
        if with_tangent:
            assert n_steps < d, \
                f"windowed tangent rows require N_ref ({n_steps}) < d ({d}); " \
                f"otherwise the window wraps around the ring and aliases"
            c_hist = torch.empty(num_mc, P, n_steps, d, device=device)
            l_r = torch.arange(2 * problem.k_dirs, device=device)
            w_idx = torch.arange(n_steps + 1, device=device)
            win_coord = (l_r.unsqueeze(-1) + w_idx) % d         # (R, W)
        for n in range(n_steps):
            tn = t + n * dt_ref
            b, c = self._drift_and_jac(tn, z)
            dw = torch.normal(0., dt_ref**0.5, z.shape, device=device)
            z = z + b * dt_ref + (2**0.5) * problem.delta * dw
            if with_tangent:
                c_hist[:, :, n] = c
        if with_tangent:
            return z, c_hist, win_coord
        return z

    def _tangent_rows(self, c_hist, win_coord):
        """Backward windowed propagation of the 2k rows of J_T.

        c_hist[p, pt, n, i] = cos(t_n + i + z_{i+1}) of the n-th EM step.
        Row l_r of J_T: rho^{(N)} = e_{l_r}^T,
        rho^{(n)}_j = rho^{(n+1)}_j + dt * rho^{(n+1)}_{j-1} c_n[j-1].
        Window entry B[p, pt, r, w] sits at coordinate (l_r + w) mod d after
        the full backward pass (support {l_r, ..., l_r + N}).
        Returns B of shape (p, P, R, N+1).
        """
        dt_ref = self.dt_ref
        n_steps = c_hist.shape[2]
        R = 2 * self.problem.k_dirs
        B = torch.zeros(c_hist.shape[0], c_hist.shape[1], R, n_steps + 1,
                        dtype=c_hist.dtype, device=c_hist.device)
        B[:, :, :, 0] = 1.0
        for n in range(n_steps - 1, -1, -1):
            cwin = c_hist[:, :, n][:, :, win_coord]             # (p,P,R,W)
            B[:, :, :, 1:] += dt_ref * B[:, :, :, :-1] * cwin[:, :, :, :-1]
        return B

    def _chunk_loop(self, x, per_chunk, with_tangent=False):
        """Memory-adaptive chunked accumulation with DDP split; per_chunk
        returns (w_mean, wg_mean) over the chunk (both already averaged over
        the chunk's paths), accumulated with running rates."""
        problem = self.problem
        use_dist = isin_ddp()
        M = problem.M_mc
        if use_dist:
            world_size = dist.get_world_size()
            M = int(M // world_size) + 1
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
        if with_tangent:
            # per path-point memory is dominated by the stored cos history
            n_steps = round(float(problem.te - x[0, 0]) / self.dt_ref) + 1
            multiplier = (n_steps + 8) * x.numel()
        else:
            multiplier = 8 * x.numel()
        cum_size = 0
        cum_w = 0.
        cum_wg = 0.
        progress = 0.
        while cum_size < M:
            try:
                chunksize = get_safe_chunksize(multiplier, x.dtype, x.device,
                                               use_percent=0.4)
                chunksize = max(1, min(chunksize, M - cum_size))
                if use_dist:
                    chunksize_tensor = torch.tensor(chunksize, device=x.device)
                    dist.all_reduce(chunksize_tensor, op=dist.ReduceOp.MIN)
                    chunksize = int(chunksize_tensor.item())
                new_w, new_wg = per_chunk(chunksize)
                cum_size += chunksize
                if use_dist:
                    dist.all_reduce(new_w, op=dist.ReduceOp.SUM)
                    new_w = new_w / world_size
                    if new_wg is not None:
                        dist.all_reduce(new_wg, op=dist.ReduceOp.SUM)
                        new_wg = new_wg / world_size
                new_rate = chunksize / cum_size
                cum_w = (1 - new_rate) * cum_w + new_rate * new_w
                if new_wg is not None:
                    cum_wg = (1 - new_rate) * cum_wg + new_rate * new_wg
                if (cum_size / M > progress + 0.01) or (cum_size == M):
                    progress = cum_size / M
                    if rank == 0:
                        print(f"Progress: {progress:.2%}, "
                              f"chunksize per rank: {chunksize}")
            except RuntimeError as err:
                if 'out of memory' in str(err):
                    chunksize = int(chunksize // 2)
                    print(f"Restricted by memory, reduce chunksize to "
                          f"{chunksize}")
                    if chunksize == 0:
                        raise err
                else:
                    raise err
            finally:
                free_cache(x.device)
        return cum_w, cum_wg

    def v(self, x):
        problem = self.problem

        def per_chunk(chunksize):
            z = self._samp_chunk(x, chunksize, with_tangent=False)
            w_chunk = torch.exp(-problem.g(z))                  # (p, P, 1)
            return w_chunk.mean(0), None

        print(f"Monte-Carlo for reference solution on {x.device}...\n")
        cum_w, _ = self._chunk_loop(x, per_chunk)
        return -torch.log(cum_w)

    def u_star(self, x):
        problem = self.problem
        d = problem.dim_z

        def per_chunk(chunksize):
            z, c_hist, win_coord = self._samp_chunk(x, chunksize,
                                                    with_tangent=True)
            B = self._tangent_rows(c_hist, win_coord)           # (p,P,2k,W)
            del c_hist
            w_chunk = torch.exp(-problem.g(z))                  # (p, P, 1)
            alpha = problem.grad_g(z)[:, :, :2 * problem.k_dirs]  # (p, P, 2k)
            vals = (w_chunk.unsqueeze(-1) *
                    alpha.unsqueeze(-1) * B)                    # (p,P,2k,W)
            num = torch.zeros(chunksize, x.shape[0], d, device=x.device)
            idx = win_coord.unsqueeze(0).unsqueeze(0).expand(
                chunksize, x.shape[0], -1, -1)
            num.scatter_add_(-1, idx.reshape(chunksize, x.shape[0], -1),
                             vals.reshape(chunksize, x.shape[0], -1))
            return w_chunk.mean(0), num.mean(0)

        print(f"Monte-Carlo for reference optimal control on {x.device}...\n")
        cum_w, cum_wg = self._chunk_loop(x, per_chunk, with_tangent=True)
        return -problem.delta**2 * cum_wg / cum_w


class HJBSparseSin(HJBSparseBase):
    """Variant (N): b_i(t, z) = sin(t + i + z_{i+1}), cyclic (the Section 4.2
    drift), inside the Section 4.1 normalization; Monte Carlo reference on
    the finer grid dt_ref = 1/100 (noise floor ~1e-3 at M_mc = 1e5)."""

    M_mc = 10**5
    has_optimal_control = True

    def _b(self, x):
        t, z = x[..., [0]], x[..., 1:]
        i_idx = torch.arange(self.dim_z, dtype=z.dtype, device=z.device)
        return torch.sin(t + i_idx + z.roll(-1, dims=-1))

    def v(self, x):
        return _SparsePathMC(self).v(x)

    def u_star(self, x):
        return _SparsePathMC(self).u_star(x)


class HJBSparseRingW5(HJBSparseRing):
    """Variant (L) with milder ring transport omega = 5 (window travels
    ~5 coordinates over the horizon instead of ~20)."""
    omega = 5.0
