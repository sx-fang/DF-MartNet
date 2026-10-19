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
        # Terminal supervision (owner 2026-09-08): soft direct-L2 pull of
        # v_theta / u_alpha at the LAST interior time layer (t = T - dt)
        # toward the exact terminal information, bypassing the rho
        # test-family bottleneck (the diagnosed K1/L2 failure mode: rho =
        # scaled-sin random features is blind to single-coordinate terminal
        # gradients, so v_theta never learns them and u regresses to a
        # state-blind constant).  Weights default to 0.0 = off = zero
        # behavior change on every existing configuration.
        self.w_term_v = float(getattr(problem, 'w_term_v', 0.0))
        self.w_term_u = float(getattr(problem, 'w_term_u', 0.0))
        # SOC-MartNet-style adaptive martingale weighting (owner directive
        # 2026-09-13 "align DF-SOCMartNet parameters to SOC-MartNet"):
        # lam <- min(lam_bar, lam + delta4 * mart_loss) each descent step,
        # loss = lam * mart_loss + ctr_loss -- their train_fd accepts
        # lam0=10/delta4=10/lam_bar=1000 (experiments.csv --lam-bar 1000).
        # Defaults 1.0/0.0/1.0 keep lam fixed at 1.
        self.lam = float(getattr(problem, 'lam0', 1.0))
        self.delta4 = float(getattr(problem, 'delta4', 0.0))
        self.lam_bar = float(getattr(problem, 'lam_bar', 1.0))
        # Last-layer martingale upweighting (owner 2026-09-16): rows of the
        # paired |L|^2 estimate sitting at the last interior time layer
        # t = T - dt (the layer whose one-step branch lands on the boundary
        # and reads the exact terminal g) contribute with weight
        # w_mart_last, in BOTH the descent and the ascent phases (same
        # min-max objective).  Replaces terminal supervision (w_term_* = 0):
        # anchoring through the equation's own structure with exact
        # terminal data, no extra loss term.  Default 1.0 = zero behavior
        # change on every existing configuration.
        self.w_mart_last = float(getattr(problem, 'w_mart_last', 1.0))
        if self.w_mart_last <= 0.0:
            raise ValueError('w_mart_last must be positive.')
        # Control-channel twin (owner 2026-09-17): the same last-layer
        # upweighting applied to the control residual rows inside ctr_loss
        # (res_u.mean(), the one-step Dynkin u channel -- a direct loss on
        # u_alpha, not through the rho pairing).  Default 1.0 = zero
        # behavior change.
        self.w_ctr_last = float(getattr(problem, 'w_ctr_last', 1.0))
        if self.w_ctr_last <= 0.0:
            raise ValueError('w_ctr_last must be positive.')
        # Zero-hyperparameter annealed variant (owner 2026-09-17): a single
        # boolean applies the SAME linearly-annealed last-layer weight to
        # BOTH channels -- lambda_last(it) = 1 + (num_dt - 1) *
        # (1 - it/max_iter), i.e. it=0 reproduces the static martlast
        # lambda=num_dt configuration bit-for-bit and it=max_iter
        # reproduces the uniform baseline bit-for-bit; num_dt and
        # max_iter are already-fixed experiment quantities (no new
        # hyperparameters, one boolean switch).
        self.anneal_last = bool(getattr(problem, 'anneal_last', False))
        if self.anneal_last and (self.w_mart_last != 1.0
                                  or self.w_ctr_last != 1.0):
            raise ValueError('anneal_last is mutually exclusive with '
                             'w_mart_last/w_ctr_last != 1.')
        # within the first anneal_frac fraction of max_iter (default 1.0 =
        # the full-horizon linear schedule, bit-for-bit); smaller values
        # shorten the high-lambda exposure and leave more pure-baseline
        # recovery budget.
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

    # -- phase hooks ----------------------------------------------------------
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
        # Sampling already fixes both path groups and the AMC point budget;
        # reuse these points and group labels throughout descent and ascent.
        self._x_eval = self._res_points(self.x_batch)
        # per-row weights for the paired |L|^2 estimate (mart channel) and
        # the control residual mean (ctr channel): the last interior layer
        # is selected by the time column t = T - dt, which is exactly the
        # set whose one-step branch lands on the time boundary
        # (domain_mask is a time mask), independent of dw/u.  Under
        # anneal_last both channels share the linearly-annealed lambda.
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

    # -- antithetic (AMC) noise ------------------------------------------------
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

    # -- residuals -------------------------------------------------------------
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
        # u*-ablation (owner 2026-08-31): replace the learned control with the
        # closed-form optimal control, turning the min-max problem into a pure
        # linear-parabolic v solve (HJB nonlinearity resolved analytically).
        # The control residual loses its gradient path (v detached there and
        # u* carries no parameters); only its logged value remains.
        if getattr(self.problem, 'u_fixed_ref', False):
            u = self.problem.u_star(x_eval).detach()
        else:
            u = self.u_alpha(x_eval)
        v = self.v_theta(x_eval)
        dw = None
        if ns == 'qmc':
            # qmc_dw applies a fresh random digital shift per call. Sharing
            # a shift between groups would correlate their branch estimates.
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

        # Terminal supervision (owner 2026-09-08, soft form; hard anchor
        # rejected by owner for its compute overhead and the untunable
        # phi(t) decay under non-smooth terminals).  The last interior time
        # layer of x_eval is exactly the set of points whose one-step branch
        # lands on the boundary, i.e. bdmask; for AMC the mask is identical
        # across the +dw / -dw halves (the time column does not depend on
        # dw), so bdmask[:B] selects it.  Targets are read off the residual
        # path itself: the boundary rows of v_next_ud are the exact g
        # (antithetic pair-mean under AMC), and u_T is the closed-form
        # myopic terminal control -gamma^2 <grad g, q>.  Both terms are
        # self-normalized by mean(target^2) so w_term_* are O(1) weights.
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

    # -- paired |L|^2 estimate ---------------------------------------------------
    def paired_mart_loss(self, res, rho):
        """Pair disjoint path groups, keeping the existing gradient scale.

        Uniform time sampling estimates the former time-average of rho*res:
        the paper's time-sum L is (N*dt) times this mean. The returned value
        is mean(A*B)/2; the stopped, symmetric gradient is grad(mean(A*B))/4.
        These legacy constants are deliberate, not a full-gradient identity.
        Population unbiasedness assumes i.i.d. pilots and fixed parameters;
        it does not imply conditional unbiasedness for an adaptive reused bank.
        """
        rho_res = torch.einsum('...i,...j->...ij', rho, res)
        # w_mart_last: upweight the last-interior-layer rows inside the
        # paired estimate (applied before the shuffle; row-local, so the
        # permutation is irrelevant).  The estimator stays unbiased for
        # |E_w[rho res]|^2 of the row-weighted batch mean, and the true
        # solution (zero conditional-mean residual at every point) remains
        # the minimizer -- only deviations at t = T - dt are penalized
        # w_mart_last times harder.
        if self._row_w is not None and rho_res.shape[0] == self._row_w.shape[0]:
            rho_res = rho_res * self._row_w.to(rho_res.dtype).view(-1, 1, 1)
        sizes = getattr(self.problem, '_batch_group_sizes', None)
        if (sizes is None or len(sizes) != 2 or min(sizes) < 0
                or sum(sizes) != rho_res.shape[0]):
            raise ValueError('Paired loss requires the sampler path-group sizes')
        first = sizes[0]
        # At least fp32 accumulation avoids overflow in low-precision SUMs.
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
        # DDP averages parameter gradients: compensate W here, using each
        # group's actual global count rather than averaging unequal means.
        local_scaled = self.world_size * (local_sums / denom)
        proxy = (global_means[0] * local_scaled[1]
                 + global_means[1] * local_scaled[0]).mean() / 4
        statistic = (global_means[0] * global_means[1]).mean() / 2
        # A common forward statistic for logging/lambda, with the local
        # stopped gradient retained. No in-place operation touches the graph.
        return statistic + (proxy - proxy.detach())

    # -- losses ------------------------------------------------------------------
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
        # w_ctr_last: upweight the last-interior-layer rows of the control
        # residual mean (same row selection as w_mart_last; unnormalized
        # weighted mean, i.e. the same scale convention as the martingale
        # treatment).  Default 1.0 = plain res_u.mean().
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

        # adaptive martingale weight (SOC-MartNet Alg 3.1 L.11 update; see
        # __init__ -- defaults keep lam pinned at 1)
        lam = min(self.lam_bar, self.lam + self.delta4 * mart_loss.detach())
        self.lam = float(lam)
        loss = lam * mart_loss + ctr_loss
        if self._term_v is not None:
            loss = loss + self._term_v
        if self._term_u is not None:
            loss = loss + self._term_u

        self.mart_loss = mart_loss.detach().clone()
        self.ctr_loss = ctr_loss.detach().clone()
        # Snapshot for the log BEFORE loss_asc resets the stashes; keep the
        # log keys stable across iterations (zero-fill when the supervised
        # slice was empty) so the log.csv columns never change shape.
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
