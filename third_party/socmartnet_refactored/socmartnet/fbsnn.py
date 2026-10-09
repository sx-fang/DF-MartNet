"""FBSNN benchmark (Zhang 2022, Scheme 2; paper Eq. (4.1)-(4.2)).

The value network v_theta is trained by matching the one-step Euler-Maruyama
recursion of the BSDE along pilot paths:
  v_tilde_{n+1} = v_tilde_n - f(t_n, X_n, v_tilde_n, vx_n, vxx_n) Dt
                  + (sigma_n vx_n)^T dB_{n+1},   v_tilde_0 = v_theta(t_0, X_0),
with the terminal value replaced by g (loss_fbsnn over all time stamps).
Numerics follow the authors' v3 fbsnn.py; symbols renamed to the paper's.
"""

import time

import torch


class FBSNN:

    def __init__(self, Dt, mu, sigma, f_fun, g_term, dim_w,
                 t0=torch.tensor(0.)):
        self.Dt = Dt
        self.t0 = t0
        self.f_fun = f_fun
        self.mu = mu
        self.sigma = sigma
        self.g_term = g_term
        self.dim_w = dim_w

    def simu_paths(self, x0, N):
        dB_path = torch.normal(
            mean=torch.zeros([N + 1, x0.shape[0], self.dim_w]),
            std=torch.sqrt(self.Dt))
        xt = [x0]
        sgm_list = []
        tn = self.t0
        t_part = [self.t0]
        for n in range(N):
            sgm_n = self.sigma(tn, xt[n])
            sgm_list.append(sgm_n)
            xt.append(xt[n] + self.mu(tn, xt[n]) * self.Dt
                      + sgm_n * dB_path[n])
            tn = tn + self.Dt
            t_part.append(tn)
        sgm_list.append(self.sigma(tn, xt[-1]))
        return (torch.stack(t_part), torch.stack(xt),
                torch.stack(sgm_list), dB_path)

    def loss_fbsnn(self, t, xt, v_path, vx_path, vxx_path, z_path, dB_path):
        zdw = torch.einsum('...ij, ...j -> ...i', z_path[:-1], dB_path[:-1])
        v_tilde = torch.zeros_like(v_path)
        v_tilde[0] = v_path[0]
        for n in range(t.shape[0] - 1):
            v_tilde[n + 1] = v_tilde[n] - self.f_fun(
                t[n], xt[n], v_tilde[n], vx_path[n], vxx_path[n]) * self.Dt \
                + zdw[n]
        return (v_tilde - v_path).pow(2).mean()

    def solve(self, v_theta, optim, sch, max_iter, x0, batsize_arr,
              rank='None', batsize_milestone=None, N=10,
              err_func=lambda: torch.nan, log_gap=10,
              f_depends_on_vxx=False, max_epoch=float('inf')):

        if f_depends_on_vxx:
            raise RuntimeError('unsupported f_depends_on_vxx '
                               '(no Section-4 example needs the Hessian)')

        rt0 = time.time()
        v_theta.train()
        tot_size = x0.shape[0]

        t_part, xt, sgmt, dB = self.simu_paths(x0, N)
        t = t_part.expand(1, tot_size, -1).transpose(0, 2).contiguous()
        vte = self.g_term(xt[-1])

        it_hist, rt_hist, epoch_hist = [], [], []
        loss_hist, ham_hist, error_hist = [], [], []

        it = 0
        epoch = 0.
        bat_stage = 0
        bat_size = batsize_arr[bat_stage]
        if batsize_milestone is None:
            batsize_milestone = []
        batsize_milestone = batsize_milestone + [max_iter + 1]

        while True:
            if (it > max_iter) or (epoch > max_epoch):
                break
            if it >= batsize_milestone[bat_stage]:
                bat_stage += 1
                bat_size = batsize_arr[bat_stage]

            bat_idx = torch.randperm(tot_size)[:bat_size]
            t_bat = t[:, bat_idx]
            xt_bat = xt[:, bat_idx]
            sgmt_bat = sgmt[:, bat_idx]
            dB_bat = dB[:, bat_idx]

            is_req = xt_bat.requires_grad
            xt_bat.requires_grad = True
            v_bat = v_theta(t_bat, xt_bat)
            vx_bat = torch.autograd.grad(
                v_bat, xt_bat, grad_outputs=torch.ones_like(v_bat),
                create_graph=True)[0].unsqueeze(-2)
            xt_bat.requires_grad = is_req
            vxx_bat = [None] * N

            # terminal replacement v_theta(T, x) = g(x)
            v_bat[-1] = vte[bat_idx]

            z_bat = vx_bat * sgmt_bat.unsqueeze(-2)
            loss = self.loss_fbsnn(t_bat, xt_bat, v_bat, vx_bat, vxx_bat,
                                   z_bat, dB_bat)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if it % log_gap == 0:
                error = err_func().detach()
                lr = optim.param_groups[0]['lr']
                rt = time.time() - rt0
                it_hist.append(it)
                rt_hist.append(rt)
                epoch_hist.append(epoch)
                ham_hist.append(torch.tensor(torch.nan))
                loss_hist.append(loss.detach())
                error_hist.append(error.detach())
                print(
                    f"rank: {rank}\niter step: [{it}/{max_iter}], rt: {rt:.2f}, epoch: {epoch:.2f},\nbat_size: {bat_size}, lr: {lr:.5}, \nloss: {loss.item():.5},\nerror: {error:.5}\n"
                )

            it += 1
            epoch += bat_size / tot_size
            if sch is not None:
                sch.step()

        self.it_hist = torch.tensor(it_hist)
        self.epoch_hist = torch.tensor(epoch_hist)
        self.rt_hist = torch.tensor(rt_hist)
        self.ham_hist = torch.stack(ham_hist, dim=0)
        self.lossmart_hist = torch.stack(loss_hist, dim=0)
        self.error_hist = torch.stack(error_hist, dim=0)

        v_theta.eval()
        return v_theta
