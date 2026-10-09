"""MartNet loss: martingale residual + control residual (Sections 3.2-3.3).

The training objective is the min-max problem

    min_{theta, alpha} max_rho  L(v_theta, u_alpha; rho),

where the weak-form (martingale) residual at a batch point x is

    res(x) = [ v(x_next) - v(x) ] / dt + f(x, u_alpha(x)),

x_next is one Euler step of the controlled system, and the |L|^2 estimate
uses the antisymmetric pairing of two half-batches (Eq. (3.24) area),
which also de-biases the DDP reduction. The control residual is the
one-step Dynkin mean of res with v detached (the "derivative-free"
control gradient of Section 3.6).
"""
import math

import torch
import torch.distributed as dist


class MartNetLoss:
    """Alternating descent-ascent loss for (v_theta, u_alpha) vs. rho."""

    name = 'MartNetLoss'

    def __init__(self, problem, net_dict, use_dist=False):
        self.problem = problem
        self.net_dict = net_dict
        self.use_dist = use_dist
        self.world_size = dist.get_world_size() if use_dist else 1
        self.device = self._check_devices()
        self.x_next = None
        self._res = None
        self._rho_val = None
        
        
        
        
        
        
        
        
        self.w_term_v = float(getattr(problem, 'w_term_v', 0.0))
        self.w_term_u = float(getattr(problem, 'w_term_u', 0.0))
        
        
        
        
        
        
        
        self.lam = float(getattr(problem, 'lam0', 1.0))
        self.delta4 = float(getattr(problem, 'delta4', 0.0))
        self.lam_bar = float(getattr(problem, 'lam_bar', 1.0))
        if self.w_term_u > 0.0:
            if getattr(problem, 'u_fixed_ref', False):
                raise ValueError('w_term_u and u_fixed_ref are mutually '
                                 'exclusive; unset one.')
            if not hasattr(problem, 'u_star_term'):
                raise ValueError('w_term_u > 0 requires '
                                 'problem.u_star_term(x).')
        self._term_v = None
        self._term_u = None
        self.term_v_log = None
        self.term_u_log = None

    def _check_devices(self):
        net_devices = [
            next(net.parameters()).device for net in self.net_dict.values()
        ]
        assert all(d == net_devices[0] for d in net_devices)
        return net_devices[0]

    
    def init_train(self):
        self.v_theta = self.net_dict['v_theta']
        self.u_alpha = self.net_dict['u_alpha']
        self.rho = self.net_dict['rho']
        for net in self.net_dict.values():
            net.train()
        self._res = None
        self._rho_val = None

    def finalize_train(self):
        for net in self.net_dict.values():
            net.eval()

    def init_desc(self):
        """Descent phase: v_theta and u_alpha learn; rho frozen."""
        self.v_theta.train()
        self.u_alpha.train()
        self.rho.eval()
        self.x_batch = self.problem.refresh_pilot_batch(self.x_next)
        
        
        self._x_eval = self._res_points(self.x_batch)

    def init_asc(self):
        """Ascent phase: rho learns to expose the residual."""
        self.v_theta.eval()
        self.u_alpha.eval()
        self.rho.train()

    
    def _res_points(self, x_batch):
        """Select residual and test-function evaluation points.

        Antithetic mode draws a random half-batch and evaluates paired +dw/-dw
        increments. Branch residuals are averaged pointwise before the paired
        product. Other noise modes use the complete input batch."""
        if getattr(self.problem, 'noise_sampler', 'mc') == 'amc':
            n = x_batch.shape[0]
            idx = torch.randperm(n, device=x_batch.device)[:n // 2]
            return x_batch[idx]
        return x_batch

    def _amc_dw(self, m, device):
        """Gaussian increment g * sqrt(dt), g ~ N(0, I); the caller builds
        the antithetic branches with +dw / -dw."""
        return torch.randn((m, self.problem.dim_z),
                           device=device) * math.sqrt(self.problem.dt)

    
    def _v_next(self, x_next, dmask, bdmask, detach_v):
        """v at the next state: network inside the domain, g on the boundary."""
        xnext_domain = x_next[dmask]
        xnext_boundary = x_next[bdmask]
        if detach_v:
            xnext_domain = xnext_domain.detach()
            xnext_boundary = xnext_boundary.detach()
        v_domain = self.v_theta(xnext_domain)
        v_boundary = self.problem.terminal_cost(xnext_boundary)

        num_pts = xnext_domain.shape[0] + xnext_boundary.shape[0]
        v_next = torch.empty((num_pts, v_boundary.shape[-1]))
        v_next[dmask] = v_domain
        v_next[bdmask] = v_boundary
        return v_next

    def martingale_residual(self, x_batch, compute_ugrad=True):
        """res = (v(x_next) - v(x))/dt + f, in two detach variants.

        ``res_v``: gradients flow into v_theta (u detached).
        ``res_u``: gradients flow into u_alpha (v detached).

        AMC pair collapse: each evaluation point gets two antithetic
        branches (+dw / -dw); the branch values of v_next are averaged
        pointwise before forming the residual, so the returned rows are
        independent across points and the paired |L|^2 estimate stays
        unbiased.  Since ``running_cost(acc, x, u) = acc + f(x, u)`` is
        linear in ``acc``, averaging v_next first equals averaging the two
        branch residuals.
        """
        ns = getattr(self.problem, 'noise_sampler', 'mc')
        x_eval = self._x_eval if ns == 'amc' else x_batch
        
        
        
        
        
        if getattr(self.problem, 'u_fixed_ref', False):
            u = self.problem.u_star(x_eval).detach()
        else:
            u = self.u_alpha(x_eval)
        v = self.v_theta(x_eval)
        dw = None
        if ns == 'qmc':
            dw = self.problem.qmc_dw(x_eval.shape[0], x_eval.device)
        elif ns == 'amc':
            dw = self._amc_dw(x_eval.shape[0], x_eval.device)
        if ns == 'amc':
            x_next = torch.cat([
                self.problem.system_step(x_eval, u, dw=dw),
                self.problem.system_step(x_eval, u, dw=-dw),
            ], dim=0)
        else:
            x_next = self.problem.system_step(x_eval, u, dw=dw)

        dmask = self.problem.domain_mask(x_next)
        bdmask = ~dmask
        x_next[bdmask] = self.problem.project_onto_boundary(x_next[bdmask])
        self.x_next = x_next.detach()

        def _collapse(v_next_):
            """Pointwise mean of the +dw / -dw branch values (AMC only)."""
            if ns != 'amc':
                return v_next_
            v_p, v_m = v_next_.chunk(2, dim=0)
            return (v_p + v_m) * 0.5

        v_next_ud = _collapse(
            self._v_next(x_next, dmask, bdmask, detach_v=True))
        dv_ud = (v_next_ud - v) / self.problem.dt
        res_v = self.problem.running_cost(dv_ud, x_eval, u.detach())

        if compute_ugrad:
            for p in self.v_theta.parameters():
                p.requires_grad = False
            v_next_vd = _collapse(
                self._v_next(x_next, dmask, bdmask, detach_v=False))
            for p in self.v_theta.parameters():
                p.requires_grad = True
            dv_vd = (v_next_vd - v.detach()) / self.problem.dt
            res_u = self.problem.running_cost(dv_vd, x_eval, u)
        else:
            res_u = None

        
        
        
        
        
        
        
        
        
        
        
        self._term_v = None
        self._term_u = None
        if compute_ugrad and (self.w_term_v > 0.0 or self.w_term_u > 0.0):
            b = x_eval.shape[0]
            last = bdmask[:b] if ns == 'amc' else bdmask
            if int(last.sum()) > 0:
                x_last = x_eval[last]
                if self.w_term_v > 0.0:
                    tgt = v_next_ud[last].detach()
                    num = ((v[last] - tgt) ** 2).mean()
                    den = (tgt ** 2).mean().clamp_min(1e-12)
                    self._term_v = self.w_term_v * num / den
                if self.w_term_u > 0.0:
                    tgt = self.problem.u_star_term(x_last).detach()
                    num = ((u[last] - tgt) ** 2).mean()
                    den = (tgt ** 2).mean().clamp_min(1e-12)
                    self._term_u = self.w_term_u * num / den
        return res_v, res_u

    
    def paired_mart_loss(self, res, rho):
        """E[rho_a res_b] E[rho_b res_a] / 4 over random half-batch pairs.

        The pairing gives an unbiased |E[rho res]|^2-type estimate; under
        DDP the detached half is all-reduced to correct the mini-batch bias.
        """
        rho_res = torch.einsum('...i,...j->...ij', rho, res)
        rho_res = rho_res[torch.randperm(rho_res.shape[0])]

        rdm = rho_res.unflatten(0, (-1, 2)).mean(0)
        rdm_det = rdm.detach()
        if self.use_dist:
            dist.all_reduce(rdm_det)
            rdm_det = rdm_det / self.world_size

        loss_val = (rdm_det[0] * rdm[1] + rdm_det[1] * rdm[0]).mean()
        return loss_val / 4

    
    def loss_desc(self):
        """Descent loss: martingale |L|^2 + control residual mean
        (+ terminal-supervision terms when w_term_v / w_term_u > 0)."""
        self._res = None
        if self._rho_val is None:
            with torch.no_grad():
                self._rho_val = self.rho(self._x_eval)

        res_v, res_u = self.martingale_residual(self.x_batch,
                                                compute_ugrad=True)
        mart_loss = self.paired_mart_loss(res_v, self._rho_val)
        ctr_loss = res_u.mean() if res_u is not None else torch.tensor(0.)

        
        
        lam = min(self.lam_bar, self.lam + self.delta4 * mart_loss.detach())
        self.lam = float(lam)
        loss = lam * mart_loss + ctr_loss
        if self._term_v is not None:
            loss = loss + self._term_v
        if self._term_u is not None:
            loss = loss + self._term_u

        self.mart_loss = mart_loss.detach()
        self.ctr_loss = ctr_loss.detach()
        
        
        
        if self.w_term_v > 0.0:
            self.term_v_log = (self._term_v.detach()
                               if self._term_v is not None
                               else torch.zeros((), device=self.device))
        if self.w_term_u > 0.0:
            self.term_u_log = (self._term_u.detach()
                               if self._term_u is not None
                               else torch.zeros((), device=self.device))
        if self.use_dist:
            dist.all_reduce(self.ctr_loss)
            self.ctr_loss = self.ctr_loss / self.world_size
            if self.term_v_log is not None:
                dist.all_reduce(self.term_v_log)
                self.term_v_log = self.term_v_log / self.world_size
            if self.term_u_log is not None:
                dist.all_reduce(self.term_u_log)
                self.term_u_log = self.term_u_log / self.world_size
        return loss

    def loss_asc(self):
        """Ascent loss: rho maximizes the paired martingale functional."""
        self._rho_val = None
        if self._res is None:
            with torch.no_grad():
                self._res, _ = self.martingale_residual(self.x_batch,
                                                        compute_ugrad=False)
        rho = self.rho(self._x_eval)
        return -self.paired_mart_loss(self._res, rho)

    def log_func(self):
        out = {
            'pde_loss': self.mart_loss.abs().item(),
            'lam_mart': self.lam,
            'ctr_loss': self.ctr_loss.item(),
        }
        if self.w_term_v > 0.0:
            out['term_v'] = self.term_v_log.item()
        if self.w_term_u > 0.0:
            out['term_u'] = self.term_u_log.item()
        return out
