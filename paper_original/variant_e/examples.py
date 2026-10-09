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

import torch
import torch.distributed as dist

import utils
from utils import (isin_ddp, mc_for_v, project_onto_t0te, split_number,
                   time_mask)


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

    def __init__(self, dim_x, dt=1 / 100, batch_size=1024,
                 num_pilot_paths=10000):
        self.dim_x = dim_x          
        self.dim_z = dim_x - 1      
        self.dim_u = dim_x - 1      
        self.dt = dt
        self.batch_size = batch_size
        self.num_pilot_paths = num_pilot_paths
        self._Jstar = None
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
        """Initial points from ``x0_curves`` (split evenly across curves)."""
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
        """(t0, z0 * 1_d) tiled to num_paths rows."""
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

    @torch.no_grad()
    def rc_metric(self, u_func):
        """RC = (J_hat(u) - J(u*)) / J(u*) of portfolio.tex sec_num (Eq. in
        Sec. Numerical example). Returns {'rc', 'cost_hat'}; rc is NaN when
        no closed-form optimal cost exists (eps != 0 in Section 4.2).
        """
        cost_hat = self.estimate_cost(u_func, self.M_rc)
        rc = torch.full_like(cost_hat, float('nan'))
        if self.has_optimal_cost:
            if self._Jstar is None:
                self._Jstar = self.v(self._initial_points(1)).detach()
            rc = (cost_hat - self._Jstar) / self._Jstar
        return {'rc': rc.item(), 'cost_hat': cost_hat.item()}

    
    def v(self, x):
        """Reference value function (Cole-Hopf / Monte Carlo)."""
        raise NotImplementedError

    def make_logfunc(self, v_theta):
        """Per-iteration logger: relative L1 (and Linf) error against v."""
        x_test = self.x0_points(self.num_test_points)
        if isin_ddp():
            dist.broadcast(x_test, src=0)
        v_true = self.v(x_test)
        v_true_l1 = torch.abs(v_true).mean()
        v_true_linf = torch.abs(v_true).max()

        def log_func(_it):
            with torch.no_grad():
                err = v_theta(x_test) - v_true
                abs_err = torch.abs(err)
                log = {'rel_l1err': (abs_err.mean() / v_true_l1).item()}
                if self.record_linf_error:
                    log['rel_linferr'] = (abs_err.max() / v_true_linf).item()
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
        xte = x.expand((num_mc, ) + x.shape).reshape(-1, x.shape[-1])
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
                           combine='multiply')
        return -torch.log(v_inner) / self.c_pow





class HJBConstantCoeff(SOCProblem):
    """Section 4.1 family: dZ = (b*1_d + 2*kappa) dt + sqrt(2)*delta dW,
    f = delta^{-2} |kappa|^2, U = R^d, T = 1. The HJB equation reads
        dt v + inf_kappa { (b + 2*kappa).grad v + delta^2 Tr[Hess v]
                           + delta^{-2} |kappa|^2 } = 0.
    """

    b = 1.0       
    delta = 0.2   

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


class HJB1(HJBConstantCoeff):
    """HJB-1: b = 0, delta = 1, g(z) = ln(0.5 (1 + |z|^2))."""
    b = 0.0
    delta = 1.0

    def g(self, z):
        return torch.log(0.5 * (1 + z.pow(2).sum(-1, keepdim=True)))


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


class HJB3b(HJB3a):
    """HJB-3b: HJB-3a with delta = 0.1 (6 hidden layers in Table 1)."""
    delta = 0.1





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
