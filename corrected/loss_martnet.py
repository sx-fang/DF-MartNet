"""MartNet loss: martingale residual + control residual (Sections 3.2-3.3).

The training objective is the min-max problem

    min_{theta, alpha} max_rho  L(v_theta, u_alpha; rho),

where the weak-form (martingale) residual at a batch point x is

    res(x) = [ v(x_next) - v(x) ] / dt + f(x, u_alpha(x)),

x_next is one Euler step of the controlled system, and the |L|^2 estimate
pairs independently sampled points from disjoint pilot path groups.
Detached DDP sums use separate storage and actual sample counts.
The control residual is the
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
        
        
        
        
        
        
        
        
        
        self.w_mart_last = float(getattr(problem, 'w_mart_last', 1.0))
        if self.w_mart_last <= 0.0:
            raise ValueError('w_mart_last must be positive.')
        
        
        
        
        
        self.w_ctr_last = float(getattr(problem, 'w_ctr_last', 1.0))
        if self.w_ctr_last <= 0.0:
            raise ValueError('w_ctr_last must be positive.')
        
        
        
        
        
        
        
        
        self.anneal_last = bool(getattr(problem, 'anneal_last', False))
        if self.anneal_last and (self.w_mart_last != 1.0
                                  or self.w_ctr_last != 1.0):
            raise ValueError('anneal_last is mutually exclusive with '
                             'w_mart_last/w_ctr_last != 1.')
        
        
        
        
        
        self.anneal_frac = float(getattr(problem, 'anneal_frac', 1.0))
        if self.anneal_frac <= 0.0:
            raise ValueError('anneal_frac must be positive.')
        self._row_w = None
        self._row_w_ctr = None
        self._row_w_logged = False
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

    def init_desc(self, it=None):
        """Descent phase: v_theta and u_alpha learn; rho frozen.

        ``it`` (the training iteration, passed by train.py) drives the
        anneal_last schedule when enabled; None keeps the static
        per-row weights of earlier callers.
        """
        self.v_theta.train()
        self.u_alpha.train()
        self.rho.eval()
        self.x_batch = self.problem.refresh_pilot_batch(self.x_next)
        
        
        self._x_eval = self._res_points(self.x_batch)
        
        
        
        
        
        
        w_mart = self.w_mart_last
        w_ctr = self.w_ctr_last
        if self.anneal_last:
            num_dt = int(round((self.problem.te - self.problem.t0)
                               / self.problem.dt))
            horizon = (self.anneal_frac
                       * float(self.problem.max_iter))
            frac = 1.0
            if it is not None and horizon > 0:
                frac = max(0.0, 1.0 - it / horizon)
            w_mart = 1.0 + (num_dt - 1) * frac
            w_ctr = w_mart
        if w_mart != 1.0 or w_ctr != 1.0:
            b = self._x_eval.shape[0]
            t_last = self.problem.te - self.problem.dt
            last = torch.isclose(
                self._x_eval[:, 0],
                torch.full_like(self._x_eval[:, 0], t_last),
                atol=0.5 * self.problem.dt)
            self._row_w = None
            self._row_w_ctr = None
            if w_mart != 1.0:
                w = torch.ones(b, device=self._x_eval.device)
                w[last] = w_mart
                self._row_w = w
            if w_ctr != 1.0:
                w = torch.ones(b, device=self._x_eval.device)
                w[last] = w_ctr
                self._row_w_ctr = w
            if not self._row_w_logged:
                self._row_w_logged = True
                if (not self.use_dist) or dist.get_rank() == 0:
                    it_str = 'n/a' if it is None else f'{it}'
                    print(f"anneal_last={self.anneal_last} "
                          f"frac={self.anneal_frac:g} (it={it_str}/"
                          f"{getattr(self.problem, 'max_iter', '?')}), "
                          f"w_mart={w_mart:g}, w_ctr={w_ctr:g}: "
                          f"weighting {int(last.sum())}/{b} eval rows "
                          f"at t=T-dt", flush=True)
        else:
            self._row_w = None
            self._row_w_ctr = None

    def init_asc(self):
        """Ascent phase: rho learns to expose the residual."""
        self.v_theta.eval()
        self.u_alpha.eval()
        self.rho.train()

    
    def _res_points(self, x_batch):
        """Keep the sampler's points and their preassigned path groups."""
        sizes = getattr(self.problem, '_batch_group_sizes', None)
        if sizes is None or sum(sizes) != x_batch.shape[0]:
            raise ValueError('Sample points with refresh_pilot_batch before '
                             'evaluating the paired loss')
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

        AMC pairs are collapsed before the path-group product. An odd
        branch budget leaves one ordinary MC point. Rows within one group
        may share a pilot path; the two groups never share a path. Since
        running_cost is linear in its accumulator, averaging v_next first
        equals averaging the two branch residuals.
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
            
            
            dw = torch.cat([
                self.problem.qmc_dw(count, x_eval.device)
                for count in self.problem._batch_group_sizes if count
            ], dim=0)
        elif ns == 'amc':
            dw = self._amc_dw(x_eval.shape[0], x_eval.device)
        if ns == 'amc':
            num_pairs = self.problem._amc_num_pairs
            if not 0 <= num_pairs <= x_eval.shape[0]:
                raise ValueError('Invalid AMC pair count')
            x_pos = self.problem.system_step(x_eval, u, dw=dw)
            if num_pairs:
                x_neg = self.problem.system_step(
                    x_eval[:num_pairs], u[:num_pairs], dw=-dw[:num_pairs])
                x_next = torch.cat([x_pos, x_neg], dim=0)
            else:
                x_next = x_pos
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
            num_eval = x_eval.shape[0]
            v_p = v_next_[:num_eval]
            v_m = v_next_[num_eval:]
            return torch.cat([(v_p[:num_pairs] + v_m) * 0.5,
                              v_p[num_pairs:]], dim=0)

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
        """Pair disjoint path groups with the stopped symmetric product.

        Uniform time sampling computes the time-average of rho*res. The time-sum
        uses N*dt times this mean. The returned value is mean(A*B)/2 and the stopped
        symmetric gradient is grad(mean(A*B))/4."""
        rho_res = torch.einsum('...i,...j->...ij', rho, res)
        
        
        
        
        
        
        
        if self._row_w is not None and rho_res.shape[0] == self._row_w.shape[0]:
            rho_res = rho_res * self._row_w.to(rho_res.dtype).view(-1, 1, 1)
        sizes = getattr(self.problem, '_batch_group_sizes', None)
        if (sizes is None or len(sizes) != 2 or min(sizes) < 0
                or sum(sizes) != rho_res.shape[0]):
            raise ValueError('Paired loss requires the sampler path-group sizes')
        first = sizes[0]
        
        acc_dtype = (torch.float32 if rho_res.dtype in
                     (torch.float16, torch.bfloat16) else rho_res.dtype)
        local_sums = torch.stack([
            rho_res[:first].sum(dim=0, dtype=acc_dtype),
            rho_res[first:].sum(dim=0, dtype=acc_dtype),
        ])
        counts = torch.tensor(sizes, dtype=torch.int64, device=rho_res.device)
        global_sums = local_sums.detach().clone()
        global_counts = counts.clone()
        if self.use_dist:
            dist.all_reduce(global_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        if (global_counts <= 0).any().item():
            raise ValueError('Both global path groups must have evaluation points')
        denom = global_counts.to(acc_dtype).view(2, 1, 1)
        global_means = global_sums / denom
        
        
        local_scaled = self.world_size * (local_sums / denom)
        proxy = (global_means[0] * local_scaled[1]
                 + global_means[1] * local_scaled[0]).mean() / 4
        statistic = (global_means[0] * global_means[1]).mean() / 2
        
        
        return statistic + (proxy - proxy.detach())

    
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
        
        
        
        
        if (self._row_w_ctr is not None
                and res_u is not None
                and res_u.shape[0] == self._row_w_ctr.shape[0]):
            ctr_loss = (res_u
                        * self._row_w_ctr.to(res_u.dtype)).mean()
        else:
            ctr_loss = res_u.mean() if res_u is not None else torch.tensor(0.)

        if self.use_dist:
            ctr_loss = ctr_loss * (self.world_size * self._x_eval.shape[0]
                                   / self.problem._batch_global_eval_count)

        
        
        lam = min(self.lam_bar, self.lam + self.delta4 * mart_loss.detach())
        self.lam = float(lam)
        loss = lam * mart_loss + ctr_loss
        if self._term_v is not None:
            loss = loss + self._term_v
        if self._term_u is not None:
            loss = loss + self._term_u

        self.mart_loss = mart_loss.detach().clone()
        self.ctr_loss = ctr_loss.detach().clone()
        
        
        
        if self.w_term_v > 0.0:
            self.term_v_log = (self._term_v.detach().clone()
                               if self._term_v is not None
                               else torch.zeros((), device=self.device))
        if self.w_term_u > 0.0:
            self.term_u_log = (self._term_u.detach().clone()
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
