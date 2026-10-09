"""Problem classes of the companion paper.

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
import math
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

    max_step_genpoints = 500  

    t0 = 0.0
    te = 1.0
    s_range = 1.0
    M_mc = 10**6
    num_test_points = 300
    record_linf_error = True
    x0_curves = {'diag': utils.t_diag_curve}

    
    z0 = 0.0               
    M_rc = 4096            
    has_optimal_cost = True  
    
    has_optimal_control = False
    
    
    
    
    
    rc_x0_dist = False
    
    
    
    
    
    rc_x0_origin = False

    def __init__(self, dim_x, dt=1 / 100, batch_size=1024,
                 num_pilot_paths=10000):
        self.dim_x = dim_x          
        self.dim_z = dim_x - 1      
        self.dim_u = dim_x - 1      
        self.dt = dt
        self.batch_size = batch_size
        self.num_pilot_paths = num_pilot_paths
        self._Jstar = None
        self._x0_rc = None    
        self.net_dict = None  
        self.noise_sampler = 'mc'  
        self.qmc_pool_exp = 16     
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

    
    def domain_mask(self, x):
        return time_mask(x, self.t0, self.te)

    def project_onto_boundary(self, x):
        return project_onto_t0te(x, self.t0, self.te)

    
    def x0_points(self, num_points):
        """x0_points implementation."""
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

    
    def _initial_points(self, num_paths):
        """_initial_points implementation."""
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
        """rc_metric implementation."""
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
        max_chunksize = None  
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
                    free_cache(x.device)  
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





class HJBConstantCoeff(SOCProblem):
    """Section 4.1 family: dZ = (b*1_d + 2*kappa) dt + sqrt(2)*delta dW,
    f = delta^{-2} |kappa|^2, U = R^d, T = 1. The HJB equation reads
        dt v + inf_kappa { (b + 2*kappa).grad v + delta^2 Tr[Hess v]
                           + delta^{-2} |kappa|^2 } = 0.
    """

    b = 1.0       
    delta = 0.2   
    
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
        """u_star_term implementation."""
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
    
    cole_hopf_c = 4.0
    has_optimal_cost = False  
    
    
    
    

    M_mc = 10**5  
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



































class SOCComplex(SOCProblem):
    """SOCComplex implementation."""

    c0 = 2.0
    c1 = 0.5
    q_decay = 0.9
    mod_amp = 0.5
    
    
    drift_amp = 2.0
    
    
    
    
    
    time_freq = 1.0
    cole_hopf_c = 4.0          
    
    
    
    g_sin_amp = 2.0
    
    
    
    g_shell_A = 1.0
    g_scale_B = 1.0
    
    
    
    
    
    
    
    g_cusp_amp = 0.0
    g_cusp_idx = ()
    
    
    
    
    
    
    
    
    g_anchor_alpha = 0.0
    M_mc = 10**5               
    has_optimal_cost = True    
    has_optimal_control = False  
    
    
    
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
            ker = torch.cat([ker[-1:], ker[:-1]])  
            self._phi_kernel = torch.conj(torch.fft.fft(ker))
            self._phi_key = key
        return torch.fft.ifft(self._phi_kernel * torch.fft.fft(z)).real

    def _b(self, x):
        """_b implementation."""
        t, z = x[..., [0]], x[..., 1:]
        i0 = torch.arange(self.dim_z, dtype=torch.float32, device=z.device)
        return self.drift_amp * torch.sin(self.time_freq * t + i0
                                          + self._phi(z))

    
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
        return c[..., d - 1:2 * d - 1].to(orig_dtype)  

    def _mod(self, x):
        """_mod implementation."""
        t, z = x[..., [0]], x[..., 1:]
        i0 = torch.arange(self.dim_z, dtype=torch.float32, device=z.device)
        return 1.0 + self.mod_amp * torch.cos(
            self.time_freq * t + (i0 + 1) + z.roll(-1, dims=-1))

    def _s_matvec(self, x, w):
        """s(x) @ w = sqrt(2) c1 * m(x) * (T w)."""
        return math.sqrt(2.0) * self.c1 * (self._mod(x) * self._toep(w))

    
    def mu(self, x, kappa):
        b = self._b(x)
        if not torch.any(kappa):
            return b  
        return b + self.c0 * self._s_matvec(x, kappa)

    def sigma(self, x, dw):
        return self._s_matvec(x, dw)

    def running_cost(self, acc, x, kappa):
        return acc + 0.5 * kappa.pow(2).sum(-1, keepdims=True)

    def g(self, z):
        """g implementation."""
        out = (self.g_scale_B
               * (torch.log(1 + self.g_shell_A * z.pow(2).mean(-1, keepdim=True))
                  + 0.3 * torch.sin(0.5 * z).mean(-1, keepdim=True)
                  + self.g_sin_amp * torch.sin(z).mean(-1, keepdim=True)))
        
        if getattr(self, 'g_cusp_amp', 0.0) and getattr(self, 'g_cusp_idx', ()):
            out = out + self.g_cusp_amp * torch.sin(
                z[..., list(self.g_cusp_idx)]).sum(-1, keepdim=True)
        
        
        
        if getattr(self, 'g_anchor_alpha', 0.0):
            out = out + self.g_anchor_alpha * torch.sin(z[..., [0]])
        return out

    def v(self, x):
        return _PathMCReference(self, c_pow=self.cole_hopf_c).v(x)

    def u_star_term(self, x):
        """u_star_term implementation."""
        z = x[..., 1:].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            g = self.g(z)
            grad_g = torch.autograd.grad(g.sum(), z)[0]
        m = self._mod(x)
        return (-self.c0 * math.sqrt(2.0) * self.c1
                * self._toep(m * grad_g)).detach()

    
    def _initial_points(self, num_paths):
        """_initial_points implementation."""
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
        """rc_metric implementation."""
        cost_hat = self.estimate_cost(u_func, self.M_rc)
        if getattr(self, 'rc_x0_origin', False):
            if self._Jstar_pt is None:  
                self._Jstar_pt = self.v(self._initial_points(1)).detach()
            rc = (cost_hat - self._Jstar_pt) / self._Jstar_pt.abs()
            return {'rc': rc.item(), 'cost_hat': cost_hat.item()}
        if self._Jstar is None:  
            self._Jstar = self.v(self._re_grid()).mean().detach()
        rc = (cost_hat - self._Jstar) / self._Jstar
        return {'rc': rc.item(), 'cost_hat': cost_hat.item()}

    
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
        """make_logfunc implementation."""
        x_test = self._re_grid()
        v_true = self.v(x_test)
        self._Jstar = v_true.mean().detach()
        if getattr(self, 'rc_x0_origin', False):
            
            
            
            self._Jstar_pt = self.v(self._initial_points(1)).detach()
            if (not isin_ddp()) or dist.get_rank() == 0:
                print(f"rc_x0_origin startup: J*_point = v(0, z0*1_d) = "
                      f"{self._Jstar_pt.item():.6e}; distributed "
                      f"J* = {self._Jstar.item():.6e}", flush=True)
        v_true_l1 = torch.abs(v_true).mean()
        v_true_linf = torch.abs(v_true).max()

        def log_func(_it):
            mod = getattr(v_theta, 'module', v_theta)      
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


class SOCComplexK1(SOCComplex):
    """SOCComplexK1 implementation."""

    g_chan_amp = 0.0
    chan_gain = 1.0
    phi_decay = 0.9
    drift_amp = 1.2
    
    
    
    
    
    
    c1 = 0.20
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    chan_dc = 0.0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    chan_coord = -1
    chan_spike = 0.0

    def _phi(self, z):
        """_phi implementation."""
        d = self.dim_z
        key = (d, z.device, z.dtype, float(self.phi_decay))
        if getattr(self, '_phi_key', None) != key:
            ker = self.phi_decay ** torch.arange(1, d + 1, dtype=z.dtype,
                                                 device=z.device)
            ker = torch.cat([ker[-1:], ker[:-1]])  
            self._phi_kernel = torch.conj(torch.fft.fft(ker))
            self._phi_key = key
        return torch.fft.ifft(self._phi_kernel * torch.fft.fft(z)).real

    def _s_matvec(self, x, w):
        """_s_matvec implementation."""
        return math.sqrt(2.0) * self.c1 * (self._mod(x) * w)

    def _chan_q(self, dtype, device):
        """_chan_q implementation."""
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
            
            
            
            
            
            
            
            
            j = torch.arange(d, dtype=torch.float64, device='cpu')
            gen = torch.Generator(device='cpu').manual_seed(20260905)
            
            
            
            
            
            
            
            
            phase = float(torch.rand(1, generator=gen, dtype=torch.float64,
                                     device='cpu'))
            q = torch.cos(2 * math.pi * 3 * j / d + 2 * math.pi * phase)
            
            
            
            
            
            if dc != 0.0:
                q = q + dc
            
            
            
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
        """running_cost implementation."""
        return acc + 0.5 * (kappa / self.chan_gain).pow(2).sum(
            -1, keepdims=True)

    def g(self, z):
        out = super().g(z)
        if self.g_chan_amp != 0.0:
            if int(getattr(self, 'chan_coord', -1)) >= 0:
                
                
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
        """_comp_c_hat implementation."""
        if getattr(self, '_c_hat', None) is None:
            q = self._chan_q(torch.float64, torch.device('cpu'))
            n_inv = float(self._toep_inv_apply(q).norm())
            self._c_hat = (self.chan_gain * n_inv
                           / (math.sqrt(2.0) * self.c1 * 0.5) * 1.1)
        return self._c_hat


class SOCComplexK1S0(SOCComplexK1):
    """SOCComplexK1S0 implementation."""

    x0_curves = {'diag': utils.t_diag_curve}


class SOCChainGate(SOCProblem):
    """SOCChainGate implementation."""

    
    chain_alpha = 0.2
    chain_sat = 20.0
    chain_cG = 5.0
    
    gate_g0 = 1.0
    gate_g1 = 0.6
    gate_omega = 2.0 * math.pi
    gate_kappa = 4.0
    chain_gate_coord = -1        
    
    c1 = 0.25
    
    chan_gain = 30.0
    
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

    
    def _chain_cache(self, dtype, device):
        d = self.dim_z
        key = (d, self.chain_alpha, str(device), dtype)
        if getattr(self, '_chain_key', None) != key:
            k = torch.arange(1, d, dtype=torch.float64, device='cpu')
            w = k ** (-(1.0 + self.chain_alpha))
            
            
            n = 1
            while n < 2 * d:
                n *= 2
            kpad = torch.zeros(n, dtype=torch.float64, device='cpu')
            kpad[1:d] = w
            
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




















class HJBSparseBase(SOCProblem):
    """Shared data of the sparse-control benchmarks (note_lqsparse.tex)."""

    delta = 1.0
    k_dirs = 5
    c_weights = (1.0, 1.5, 2.0, 2.5, 3.0)
    
    
    
    
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
        a = 2 * c * s / denom / math.sqrt(2.)          
        grad = torch.zeros_like(z)
        pairs = grad[..., :2 * self.k_dirs].unflatten(-1, (self.k_dirs, 2))
        pairs.copy_(a.unsqueeze(-1).expand_as(pairs))
        return grad

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    eval_spread = True
    spread_control = True

    
    
    
    
    
    
    
    
    
    
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
            
            
            
            
            nets = [v_theta] + ([u_alpha] if u_alpha is not None else [])
            flags = []
            for net in nets:
                mod = getattr(net, 'module', net)      
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
                        
                        
                        
                        
                        mask_act = torch.abs(u_true) >= 0.05 * u_true_linf
                        log['rel_l1err_u_act'] = (
                            (abs_err_u * mask_act).sum() /
                            (torch.abs(u_true) * mask_act).sum()).item()
                        log['abserr_u_bg'] = (
                            (abs_err_u * (~mask_act)).sum() /
                            (~mask_act).sum()).item()
                        
                        
                        
                        
                        
                        if getattr(self, 'u_metrics_extended', False):
                            log['mae_u'] = abs_err_u.mean().item()
                            log['rel_e_u'] = (
                                abs_err_u.pow(2).sum().sqrt() /
                                u_true.pow(2).sum().sqrt()).item()
                    else:
                        log['rel_l1err_u'] = float('nan')
                        if self.record_linf_error:
                            log['rel_linferr_u'] = float('nan')
                    
                    
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

    n_gh = 13          

    def __init__(self, problem):
        self.problem = problem

    def _symbols(self, r, tau, d, device):
        theta = 2 * math.pi * torch.arange(d, dtype=torch.float64,
                                           device=device) / d
        lam_L = torch.exp(1j * theta) - 1
        prop = torch.exp(r * lam_L)                
        propT = torch.exp(r * lam_L.conj())        
        lam_A = 2 * torch.cos(theta) - 2           
        delta2 = self.problem.delta**2
        omega = self.problem.omega
        
        
        
        
        
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
                          dtype=torch.float64, device=device)   
        w = torch.tensor(np.prod(np.meshgrid(*([weights] * k),
                                           indexing='ij'), axis=0).ravel(),
                         dtype=torch.float64, device=device)    
        w = w / (2 * math.pi)**(k / 2)
        return xi, w

    def _qmc_grid(self, k, device):
        """_qmc_grid implementation."""
        q = 2 ** self.problem.sparse_qmc_exp
        sob = torch.quasirandom.SobolEngine(dimension=k, scramble=True,
                                            seed=20260827)
        xi = sob.draw(q, dtype=torch.float64).to(device)         
        xi = torch.special.ndtri(xi)                             
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
                            device=y.device)                    
        if getattr(problem, 'sparse_ref', 'gh') == 'qmc':
            xi, w = self._qmc_grid(k, y.device)                 
        else:
            xi, w = self._gh_grid(k, y.device)                  
        L_c = torch.linalg.cholesky(C)                          
        yq = y.unsqueeze(1) + torch.einsum('pij,qj->pqi', L_c, xi)  
        denom = 1 + (c * yq.pow(2)).sum(-1)                     
        h = 2 / denom
        phi = (w * h).sum(-1)                                   
        dh = -4 * c * yq / denom.pow(2).unsqueeze(-1)           
        dphi = torch.einsum('q,pqj->pj', w, dh)                 
        return phi, dphi

    def _phi_and_grad_lq(self, y, C):
        """_phi_and_grad_lq implementation."""
        problem = self.problem
        k = problem.k_dirs
        c = torch.as_tensor(problem.c_weights[:k], dtype=y.dtype,
                            device=y.device)                    
        a = torch.full((k,), float(problem.sparse_a), dtype=y.dtype,
                       device=y.device)
        S = torch.diag(1. / c) + C[0]
        S = 0.5 * (S + S.t())                  
        sol = torch.linalg.solve(S, (y - a).t()).t()            
        
        Lc = torch.linalg.cholesky(S)
        logdet = c.log().sum() + 2 * torch.log(torch.diagonal(Lc)).sum()
        phi = torch.exp(-0.5 * (logdet + ((y - a) * sol).sum(-1)))
        return phi, -phi.unsqueeze(-1) * sol

    def _hybrid_level(self, m, sgm2, propT):
        """_hybrid_level implementation."""
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
                            device=device)                       
        a = float(problem.sparse_a)
        mu_k = a * c / (alpha + c)                               
        c_g = 0.5 * (c * a * (a - mu_k)).sum()                   
        sq2 = math.sqrt(2.)
        
        Qd = torch.zeros(k, d, dtype=torch.float64, device=device)
        jj = torch.arange(k, device=device)
        Qd[jj, 2 * jj] = 1 / sq2
        Qd[jj, 2 * jj + 1] = 1 / sq2
        mu_full = torch.zeros(d, dtype=torch.float64, device=device)
        mup = mu_full[:2 * k].unflatten(-1, (k, 2))
        mup.copy_((mu_k / sq2).unsqueeze(-1).expand(k, 2))
        w = m - mu_full                                          

        
        A0sym = 1. + alpha * sgm2                                
        logdet0 = torch.log(A0sym).sum()
        s2a = sgm2 / A0sym
        SA = torch.fft.ifft(s2a.unsqueeze(0) *
                            torch.fft.fft(Qd, n=d, dim=-1),
                            n=d, dim=-1).real                    
        Tt = (SA[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) / sq2).t()
        Tt = 0.5 * (Tt + Tt.t())                                 
        sqc = c.sqrt()
        Tsym = torch.eye(k, dtype=torch.float64, device=device) + \
            sqc.unsqueeze(-1) * Tt * sqc.unsqueeze(0)
        logdet_k = 2 * torch.log(torch.diagonal(
            torch.linalg.cholesky(Tsym))).sum()
        logdet = logdet0 + logdet_k

        
        d0inv = 1. / (1. / alpha + sgm2)                         
        y0 = torch.fft.ifft(d0inv.unsqueeze(0) *
                            torch.fft.fft(w, n=d, dim=-1),
                            n=d, dim=-1).real                    
        qf1 = (w * y0).sum(-1)                                   
        p0 = y0[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) / sq2   
        SD = torch.fft.ifft(d0inv.unsqueeze(0) *
                            torch.fft.fft(Qd, n=d, dim=-1),
                            n=d, dim=-1).real                    
        G0 = (SD[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) / sq2).t()
        G0 = 0.5 * (G0 + G0.t())                                 
        Winv = alpha * (alpha + c) / c                           
        E = torch.diag(Winv) - G0
        E = 0.5 * (E + E.t())
        sol = torch.linalg.solve(E, p0.t()).t()                  
        qf2 = (p0 * sol).sum(-1)                                 
        quad = qf1 + qf2

        v = c_g + 0.5 * (logdet + quad)                          
        
        wk = torch.zeros(w.shape[0], d, dtype=torch.float64, device=device)
        wp = wk[:, :2 * k].unflatten(-1, (k, 2))
        wp.copy_((sol / sq2).unsqueeze(-1).expand_as(wp))
        y1 = torch.fft.ifft(d0inv.unsqueeze(0) *
                            torch.fft.fft(wk, n=d, dim=-1),
                            n=d, dim=-1).real                    
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
            m = torch.fft.ifft(prop * torch.fft.fft(z)).real    
            if getattr(problem, 'sparse_c0', 0.0) != 0.0:
                vh, uh = self._hybrid_level(m, sgm2, propT)
                v_out[mask] = vh
                u_out[mask] = uh
                continue
            mp = m[:, :2 * k].unflatten(-1, (k, 2))
            y = mp.sum(-1) / math.sqrt(2.)                      
            
            Qd = torch.zeros(k, d, dtype=torch.float64, device=device)
            Qd[torch.arange(k), 2 * torch.arange(k)] = 1 / math.sqrt(2.)
            Qd[torch.arange(k), 2 * torch.arange(k) + 1] = 1 / math.sqrt(2.)
            SQ = torch.fft.ifft(sgm2.unsqueeze(0) *
                                torch.fft.fft(Qd, n=d, dim=-1),
                                n=d, dim=-1).real               
            
            C = (SQ[:, :2 * k].unflatten(-1, (k, 2)).sum(-1) /
                 math.sqrt(2.)).t()                             
            C = 0.5 * (C + C.t())  
            C = C.expand(z.shape[0], k, k)
            phi, dphi = self._phi_and_grad(y, C)
            v_out[mask] = (-torch.log(phi)).unsqueeze(-1)
            w = torch.zeros(z.shape[0], d, dtype=torch.float64,
                            device=device)
            a = (dphi / phi.unsqueeze(-1)) / math.sqrt(2.)      
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

    omega = 20.0     
    has_optimal_control = True

    def _b(self, x):
        z = x[..., 1:]
        return self.omega * (z.roll(-1, dims=-1) - z)

    def v(self, x):
        return _RingQuadrature(self).v(x)

    def u_star(self, x):
        return _RingQuadrature(self).u_star(x)


class HJBSparseLQ(HJBSparseRing):
    """HJBSparseLQ implementation."""

    sparse_a = 1.0
    
    
    
    
    
    
    
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
        a = c * (s - self.sparse_a) / math.sqrt(2.)            
        grad = torch.zeros_like(z)
        pairs = grad[..., :2 * self.k_dirs].unflatten(-1, (self.k_dirs, 2))
        pairs.copy_(a.unsqueeze(-1).expand_as(pairs))
        if self.sparse_c0 != 0.0:
            grad = grad + (self.sparse_c0 / self.dim_z) * z
        return grad


class HJBSparseLQUA(HJBSparseLQ):
    """HJBSparseLQUA implementation."""

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
        u_k = upair.sum(-1) / math.sqrt(2.)          
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
        
        
        
        amp = torch.full_like(k, (2. / d) ** 0.5)
        amp[0] = d ** -0.5
        Q = (amp * torch.cos(math.pi * (2 * n + 1) * k / (2 * d))).t()
        _cache[key] = Q.to(dtype)
    return _cache[key]


class HJBSparseSEP(HJBSparseRing):
    """HJBSparseSEP implementation."""

    omega = 0.0
    sparse_a = 1.0
    c_weights = (1.0,)      
    sparse_m = None         
    
    
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
    """HJBSparseLQSN implementation."""

    omega = 0.0
    sparse_m = None         
    
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
            B = torch.cat([S.t(), Qdct], dim=1)         
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
        SQ = S @ Q                                        
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
                                Q, n=d, dim=0), n=d, dim=0).real   
                        
                        B = (eLQ[2 * jj] + eLQ[2 * jj + 1]) / math.sqrt(2.)  
                        w = 1.0 if (s_i == 0 or s_i == n) else 2.0
                        C = C + w * (B @ B.t())
                    C = self.delta ** 2 * ds * C
                    C = 0.5 * (C + C.t())
                    self._lqsn_C_cache[round(tau, 12)] = C
            
            if omega == 0.0:
                
                
                
                mvec = z_all[mask] @ S.t()                
            else:
                prop_tau, _ = self._omega_symbols(tau, d, x.device)
                Rz = torch.fft.ifft(prop_tau * torch.fft.fft(
                    z_all[mask], n=d, dim=-1), n=d, dim=-1).real
                mvec = Rz @ S.t()                          
            Sm = torch.diag(1. / c) + C
            Sm = 0.5 * (Sm + Sm.t())
            sol = torch.linalg.solve(Sm, (mvec - a).t()).t()  
            Lc = torch.linalg.cholesky(Sm)
            logdet = c.log().sum() + 2 * torch.log(torch.diagonal(Lc)).sum()
            v_out[mask] = 0.5 * (logdet + ((mvec - a) * sol).sum(-1,
                                                                 keepdim=True))
            
            wpair = (-self.delta ** 2) * sol / math.sqrt(2.)  
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
        """u_star_term implementation."""
        x_te = x.clone()
        x_te[..., 0] = self.te
        return self.u_star(x_te).detach()


class HJBSparseLQK1(HJBSparseLQSN):
    """HJBSparseLQK1 implementation."""

    @property
    def k_dirs(self):
        """k is pinned to 1 (the design); sparse_k must NOT be set."""
        return 1

    @k_dirs.setter
    def k_dirs(self, value):
        
        
        
        
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


class HJBSparseLQK1M(HJBSparseLQK1):
    """HJBSparseLQK1M implementation."""

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
            SQ = self._pair_S(torch.float64, device) @ Q    
            C = 2.0 * self.delta ** 2 * tau * (SQ @ SQ.t())
            return C.reshape(())                            
        if not hasattr(self, '_lqsn_C_cache'):
            self._lqsn_C_cache = {}
        C = self._lqsn_C_cache.get(round(tau, 12))
        if C is None:
            n = max(64, int(round(tau * 256)))
            ds = tau / n
            Q = self._sn_basis(torch.float64, device)       
            qrow = self._pair_S(torch.float64, device)      
            C = torch.zeros(1, 1, dtype=torch.float64, device=device)
            for s_i in range(n + 1):
                s = s_i * ds
                prop_s, _ = self._omega_symbols(s, d, device)
                eLQ = torch.fft.ifft(
                    prop_s.unsqueeze(1) * torch.fft.fft(
                        Q, n=d, dim=0), n=d, dim=0).real      
                B = qrow @ eLQ                               
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
    """HJBSparseLQK1F1 implementation."""

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
            SQ = self._pair_S(torch.float64, device) @ Q    
            return (2.0 * self.delta ** 2 * tau * (SQ @ SQ.t())).reshape(())
        if not hasattr(self, '_lqsn_C_cache'):
            self._lqsn_C_cache = {}
        C = self._lqsn_C_cache.get(round(tau, 12))
        if C is None:
            n = max(64, int(round(tau * 256)))
            ds = tau / n
            q, qt = self._f1_vecs(device)
            Q = self._sn_basis(torch.float64, device)       
            acc = 0.
            for s_i in range(n + 1):
                s = s_i * ds
                es = math.exp(gam * s)
                wrow = es * math.cos(phi * s) * q + \
                    es * math.sin(phi * s) * qt              
                B = wrow @ Q                                 
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
    """HJBSparseLQD1 implementation."""

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
        """omega implementation."""
        return 0.0

    @omega.setter
    def omega(self, value):
        if float(value) != 0.0:
            raise ValueError(
                'HJBSparseLQD1 requires sparse_omega = 0: at omega != 0 the ring propagation spreads u* onto neighbouring coordinates and the single-coordinate design is violated.')

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
        """u_star implementation."""
        c = float(self.c_weights[0])
        a = self.sparse_a
        dd = self.delta
        tau = self.te - x[:, 0].double()
        z0 = x[:, 1].double()
        Q = self._sn_basis(torch.float64, x.device)
        S = self._pair_S(torch.float64, x.device)
        SQ = S @ Q                                          
        snorm2 = (SQ @ SQ.t()).item()                       
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
    """HJBSparseLQK5F implementation."""

    
    
    c_weights = (1.0, 1.0, 1.0, 1.0, 1.0)

    @property
    def k_dirs(self):
        """k is pinned to 5 (the five cos modes f = 1..5); sparse_k must
        NOT be set."""
        return 5

    @k_dirs.setter
    def k_dirs(self, value):
        
        
        
        
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
            th = 2. * math.pi * fs.unsqueeze(-1) * jj / d   
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
            SQ = self._pair_S(torch.float64, device) @ Q    
            C = 2.0 * self.delta ** 2 * tau * (SQ @ SQ.t())
            return 0.5 * (C + C.t())
        if not hasattr(self, '_lqsn_C_cache'):
            self._lqsn_C_cache = {}
        C = self._lqsn_C_cache.get(round(tau, 12))
        if C is None:
            n = max(64, int(round(tau * 256)))
            ds = tau / n
            q, qt = self._k5f_vecs(device)
            Q = self._sn_basis(torch.float64, device)       
            C = torch.zeros(5, 5, dtype=torch.float64, device=device)
            for s_i in range(n + 1):
                s = s_i * ds
                W = torch.empty(5, d, dtype=torch.float64, device=device)
                for f in range(5):
                    es = math.exp(gam[f] * s)
                    W[f] = es * math.cos(phi[f] * s) * q[f] + \
                        es * math.sin(phi[f] * s) * qt[f]
                Bm = W @ Q                                     
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
            y = al_t * (z @ q.t()) + be_t * (z @ qt.t())    
            C = self._qC(tau, x.device)                     
            Sm = torch.diag(1. / c) + C
            Sm = 0.5 * (Sm + Sm.t())
            sol = torch.linalg.solve(Sm, (y - a).t()).t()   
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
    """_SparsePathMC implementation."""

    dt_ref = 1 / 100
    max_mc_steps = 10**6

    def __init__(self, problem):
        self.problem = problem
        
        
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
            n_steps += 1  
        c_hist = None
        win_coord = None
        if with_tangent:
            assert n_steps < d, \
                f"windowed tangent rows require N_ref ({n_steps}) < d ({d}); " \
                f"otherwise the window wraps around the ring and aliases"
            c_hist = torch.empty(num_mc, P, n_steps, d, device=device)
            l_r = torch.arange(2 * problem.k_dirs, device=device)
            w_idx = torch.arange(n_steps + 1, device=device)
            win_coord = (l_r.unsqueeze(-1) + w_idx) % d         
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
            cwin = c_hist[:, :, n][:, :, win_coord]             
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
            w_chunk = torch.exp(-problem.g(z))                  
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
            B = self._tangent_rows(c_hist, win_coord)           
            del c_hist
            w_chunk = torch.exp(-problem.g(z))                  
            alpha = problem.grad_g(z)[:, :, :2 * problem.k_dirs]  
            vals = (w_chunk.unsqueeze(-1) *
                    alpha.unsqueeze(-1) * B)                    
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
