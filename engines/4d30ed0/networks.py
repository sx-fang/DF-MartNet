"""Fully-connected networks for v_theta, u_alpha and rho.

The input state is the augmented vector ``x = (t, z)`` with ``t`` in the
first coordinate (the paper's Section 3.2). ``time_uniform_rad`` restores
the O(1) initialization scale of the time-column weights; without it the
time dependency is effectively frozen at high dimensions (default PyTorch
init gives |w| <= 1/sqrt(dim_x), which stalls learning of v(t, x)).
"""
from typing import Sequence

import torch
from torch import nn
from torch.amp import autocast

from utils import isin_ddp

# temp/hjb1_d10_d100_tuning/PLAN_PHASE3_RADIAL_CTRL_20260927.md).  Re-exported
# here so callers that already import networks keep one import site; the
# implementation lives in its own module to keep this file's diff minimal.
# radial_control imports only torch, so this cannot cycle.
from radial_control import RadialControlHead, RadialControlUNet  # noqa: F401


def zero_init_output_layer(net):
    """Zero the network's last linear layer (weight and bias).

    its optimal control is concentrated on one or two of the d output
    components and is ~0 elsewhere, while a freshly initialised FCNet emits
    O(1) values on all d components -- with d = 1e4 the quadratic control cost
    |kappa|^2 then starts four orders of magnitude above its true value and the
    probe arms diverged (jobs 543767/543768).  Starting from kappa = 0 matches
    the target's sparsity.  Raises if the last module is not a Linear layer, so
    a silent no-op is impossible.
    """
    hl = getattr(net, 'hidden_layers', None)
    last = hl[-1] if hl is not None and len(hl) else None
    if not isinstance(last, nn.Linear):
        raise ValueError('zero_init_output_layer: last module is %s, not '
                         'nn.Linear' % type(last).__name__)
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()
    return net


def make_inputlayer(dim_in, dim_out, bias_uniform=False,
                    time_uniform_rad=1.0):
    """Input layer with optional uniform bias and O(1) time-column weights."""
    xlayer = nn.Linear(dim_in, dim_out)
    if bias_uniform:
        xlayer.bias.data.uniform_(-1.0, 1.0)
    if time_uniform_rad is not None:
        assert dim_in >= 2, "dim_in >= 2 required for time_uniform_rad"
        xlayer.weight.data[:, 0].uniform_(-time_uniform_rad,
                                          time_uniform_rad)
    return xlayer


def make_hidden_layers(xdims, act_func, batch_norm=False):
    """Hidden layers ``xdims[1] -> ... -> xdims[-1]`` with activations."""
    layers = []
    num_hidden = len(xdims) - 2
    if num_hidden > 0:
        layers.append(act_func())
        for i in range(1, num_hidden):
            layers.append(nn.Linear(xdims[i], xdims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(xdims[i + 1]))
            layers.append(act_func())
        layers.append(nn.Linear(xdims[-2], xdims[-1]))
    else:
        layers.append(nn.Identity())
    nn_seq = nn.Sequential(*layers)
    if isin_ddp() and batch_norm:
        nn_seq = nn.SyncBatchNorm.convert_sync_batchnorm(nn_seq)
    return nn_seq


class FCNet(nn.Module):
    """Fully-connected network on the augmented state x = (t, z).

    Parameters
    ----------
    xdims : sequence of int
        Layer widths ``[dim_x, W, ..., W, dim_out]`` (W is the paper's
        network width).
    bias_uniform : bool
        Initialize the input-layer bias from U(-1, 1).
    time_uniform_rad : float or None
        Radius for the uniform init of the time-column weights; None
        disables the trick.
    shell_func : callable or None
        Post-processing ``shell_func(x, y)`` (used for rho = sin(y)).
    scale_lb, scale_ub : float
        If different from 1.0, scale the output coordinates linearly from
        ``scale_lb`` to ``scale_ub`` (used for rho with linspace(1, 100, W)).
    enable_autocast, autocast_dtype : mixed-precision (AMP) switches.
    """

    def __init__(self,
                 xdims: Sequence,
                 bias_uniform=True,
                 shell_func=None,
                 act_func=nn.ReLU,
                 batch_norm=False,
                 time_uniform_rad=None,
                 scale_ub=1.0,
                 scale_lb=1.0,
                 enable_autocast=False,
                 autocast_dtype=torch.float16):
        super().__init__()
        self.dim_x = xdims[0]
        self.dim_out = xdims[-1]
        # Descent can concatenate current/next states for one value-network
        # call only when a row's result is independent of other batch rows.
        self.independent_rows = (not batch_norm and act_func in (
            nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.SiLU,
            nn.ELU, nn.GELU, nn.Softplus, nn.Identity))

        self.enable_autocast = enable_autocast
        self.autocast_dtype = autocast_dtype if enable_autocast else None

        self.xlayer = make_inputlayer(xdims[0],
                                      xdims[1],
                                      bias_uniform=bias_uniform,
                                      time_uniform_rad=time_uniform_rad)
        self.hidden_layers = make_hidden_layers(xdims,
                                                act_func=act_func,
                                                batch_norm=batch_norm)
        self.layers = nn.Sequential(self.xlayer, self.hidden_layers)

        if (scale_ub != 1.0) or (scale_lb != 1.0):
            w = torch.linspace(scale_lb, scale_ub, self.dim_out)
            self.scale_layer = lambda z: w * z
        else:
            self.scale_layer = lambda z: z

        self.shell_func = shell_func

    def call(self, x):
        if x.ndim > 1:
            batch_dim = x.shape[:-1]
            flat_x = x.flatten(end_dim=-2)
        else:
            batch_dim = ()
            flat_x = x
        y = self.layers(flat_x)
        y = self.scale_layer(y)
        if self.shell_func is not None:
            y = self.shell_func(flat_x, y)
        if x.ndim > 1:
            y = y.unflatten(0, batch_dim)
        return y

    def forward(self, x):
        with autocast(x.device.type,
                      enabled=self.enable_autocast,
                      dtype=self.autocast_dtype):
            y = self.call(x)
        # Only downcast when autocast actually ran; otherwise y already has
        # the default dtype (forcing .float() would break float64 runs).
        if self.enable_autocast:
            return y.float()
        return y

    @property
    def module(self):
        """Identity; mirrors the DDP-wrapped interface."""
        return self


class AnchoredUNet(nn.Module):
    """Time-gated terminal anchor around an inner control net (owner
    2026-09-04 plan A):

        u(t, x) = phi(t) u_T(x) + (1 - phi(t)) inner(t, x),
        phi(t) = (t / T)^p,   T = problem.te, p = [Network] u_anchor_pow.

    The anchor term u_T = ``problem.u_star_term(x)`` is a PURE FUNCTION of
    the problem data (closed-form terminal optimal control): the module has
    no parameters of its own, so the optimizer and DDP gradient sync see
    exactly the inner net's parameters.  Wrap the inner net BEFORE the DDP
    layer (DDP(AnchoredUNet(FCNet))), which ``solver._build_net`` does via
    its ``net_wrapper`` hook -- the anchored forward is then the DDP forward
    and gradient hooks fire as usual.  ``state_dict``/``load_state_dict``
    delegate to the inner net so saved weights stay byte-compatible with the
    bare-FCNet format, and ``enable_autocast`` passes through so the
    logfunc's fp32-metric toggling (``getattr(net, 'module', ...).enable_autocast``)
    keeps working through the wrapper.

    At t = T the gate is exactly 1 (forward returns u_T bit-exact); at t = 0
    it is exactly 0 (forward returns the inner net bit-exact)."""

    def __init__(self, problem, anchor_pow, inner):
        super().__init__()
        if not hasattr(problem, 'u_star_term'):
            raise ValueError(
                f"u_anchor_pow is set but example "
                f"{type(problem).__name__} provides no closed-form terminal "
                f"control (no u_star_term method); unset u_anchor_pow.")
        if float(anchor_pow) <= 0:
            raise ValueError(f"u_anchor_pow must be > 0, got {anchor_pow!r}")
        self.inner = inner
        self._problem = problem          # plain attribute (not a submodule)
        self._pow = float(anchor_pow)
        self._T = float(problem.te)

    @property
    def module(self):
        """Identity; mirrors the DDP-wrapped interface (FCNet.module)."""
        return self

    def forward(self, x):
        t = x[..., 0]
        phi = (t / self._T).pow(self._pow).unsqueeze(-1)
        u_t = self._problem.u_star_term(x)
        return phi * u_t + (1.0 - phi) * self.inner(x)

    # -- passthrough: weights/AMP flag stay bare-FCNet compatible ------------
    def state_dict(self, *args, **kwargs):
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.inner.load_state_dict(state_dict, *args, **kwargs)

    @property
    def enable_autocast(self):
        return self.inner.enable_autocast

    @enable_autocast.setter
    def enable_autocast(self, value):
        self.inner.enable_autocast = value
