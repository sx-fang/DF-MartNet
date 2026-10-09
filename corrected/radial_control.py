"""Numerical implementation implementation."""
import math

import torch
from torch import nn

__all__ = ['RadialControlHead', 'RadialControlUNet']


class RadialControlHead(nn.Module):
    """RadialControlHead implementation."""

    def __init__(self, dim_z, width=32, num_hidden=2, act='SiLU',
                 enable_autocast=False, autocast_dtype=torch.bfloat16):
        super().__init__()
        if dim_z < 1:
            raise ValueError(f'dim_z must be >= 1, got {dim_z!r}')
        if width < 1:
            raise ValueError(f'width must be >= 1, got {width!r}')
        if num_hidden < 1:
            raise ValueError(f'num_hidden must be >= 1, got {num_hidden!r}')
        self.dim_z = int(dim_z)
        self.width = int(width)
        self.num_hidden = int(num_hidden)
        self.act_name = act
        self.enable_autocast = bool(enable_autocast)
        self.autocast_dtype = (autocast_dtype if enable_autocast else None)

        act_cls = getattr(nn, act)
        layers = []
        dim_in = 2                     
        for _ in range(self.num_hidden):
            layers.append(nn.Linear(dim_in, self.width))
            layers.append(act_cls())
            dim_in = self.width
        layers.append(nn.Linear(dim_in, 1))
        self.mlp = nn.Sequential(*layers)

        
        
        
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        
        
        self.register_buffer(
            'softplus1',
            torch.tensor(math.log1p(math.exp(1.0)), dtype=torch.float32),
            persistent=False)

    def forward(self, x):
        t = x[..., 0].float()
        z = x[..., 1:].float()
        q = torch.log1p(z.square().sum(-1) / self.dim_z)
        with torch.amp.autocast(x.device.type,
                                enabled=self.enable_autocast,
                                dtype=self.autocast_dtype):
            h = self.mlp(torch.stack((t, q), dim=-1))          
        
        h = h.float().squeeze(-1)
        s = (torch.nn.functional.softplus(1.0 + (1.0 - t) * h)
             / self.softplus1)
        
        s = torch.where(t == 1.0, torch.ones_like(s), s)
        base = -2.0 * z / (1.0 + z.square().sum(-1, keepdim=True))
        return base * s.unsqueeze(-1)


class RadialControlUNet(nn.Module):
    """``u_alpha`` wrapper: replace the inner FCNet by the radial head.

    Same contract as ``networks.AnchoredUNet``: the wrapper sits INSIDE the
    DDP layer (``solver._build_net``'s ``net_wrapper`` hook), so the head's
    parameters feed the optimizer and DDP gradient sync unchanged, and
    ``enable_autocast`` passes through so the logfunc/evaluator fp32 toggle
    (``getattr(net, 'module', net).enable_autocast``) keeps working.

    The freshly constructed FCNet argument is deliberately discarded.  It
    must not be registered as a submodule, because its unused parameters
    would otherwise enter the optimizer and DDP state_dict.
    """

    def __init__(self, inner, dim_z, width=32, num_hidden=2, act='SiLU',
                 enable_autocast=False, autocast_dtype=torch.bfloat16):
        super().__init__()
        
        
        del inner
        self.head = RadialControlHead(dim_z, width=width, num_hidden=num_hidden,
                                      act=act, enable_autocast=enable_autocast,
                                      autocast_dtype=autocast_dtype)

    @property
    def module(self):
        """Identity; mirrors the DDP-wrapped FCNet interface."""
        return self

    def forward(self, x):
        return self.head(x)

    
    @property
    def enable_autocast(self):
        return self.head.enable_autocast

    @enable_autocast.setter
    def enable_autocast(self, value):
        self.head.enable_autocast = bool(value)
