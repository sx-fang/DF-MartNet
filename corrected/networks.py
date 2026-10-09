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






from radial_control import RadialControlHead, RadialControlUNet  


def zero_init_output_layer(net):
    """zero_init_output_layer implementation."""
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
        
        
        if self.enable_autocast:
            return y.float()
        return y

    @property
    def module(self):
        """Identity; mirrors the DDP-wrapped interface."""
        return self


class AnchoredUNet(nn.Module):
    """AnchoredUNet implementation."""

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
        self._problem = problem          
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
