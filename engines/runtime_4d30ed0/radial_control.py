"""Radially-equivariant control head for HJB-1.

Mathematical structure:

    HJB-1 has dynamics ``dZ = sqrt(2) delta dW`` (``b = 0``, ``delta = 1``),
    rotation-invariant running cost and terminal cost
    ``g(z) = log((1 + ||z||^2) / 2)``, hence the optimal control is radially
    equivariant,

        u*(t, z) = -a(t, ||z||^2) z,      a(1, r^2) = 2 / (1 + r^2).

    The new head enforces that structure exactly and satisfies the HJB-1
    terminal condition exactly:

        q = log1p(||z||^2 / d)
        h = MLP_alpha(t, q)                 # 2 -> W -> ... -> 1, SiLU
        u_alpha(t, z) = -2 z / (1 + ||z||^2)
                        * softplus(1 + (1 - t) h) / softplus(1).

    The output layer of ``MLP_alpha`` is zero-initialized, so at
    initialization

        u_alpha = -2 z / (1 + ||z||^2)  *  softplus(1) / softplus(1)
                = -2 z / (1 + ||z||^2),

    which (i) equals the exact terminal control at ``t = 1`` for every weight
    (the multiplier is exactly 1 there, both numerator and denominator being
    ``softplus(1)``), (ii) is the ``delta = 1`` Cole-Hopf terminal control at
    every time at initialization, (iii) vanishes at the origin, and (iv) is
    equivariant under every rotation ``z -> Qz`` (``Q`` orthogonal) because
    ``q`` depends on ``z`` only through ``||z||^2``.

The head is trained by the SAME MartNet min-max loss and the SAME data flow
as the base control net: the module is a drop-in ``u_alpha`` whose only
parameters are the head's.  No ``u*`` label ever enters the training loss --
``w_term_u`` stays 0 in this campaign.  The switch lives in
``solver.parse_nets`` under ``[Network] radial_ctrl`` (default off, in which
case nothing in this file is constructed and existing behavior is unchanged).

The module is deliberately parameter-free w.r.t. ``networks.FCNet``: the
inner FCNet is built by ``solver._build_net`` and ignored, so that (a) the
state_dict carries only head parameters (a strict load detects any
architecture mismatch), and (b) ``solver._build_net``'s device/DDP handling
and ``enable_autocast`` plumbing keep working unchanged.
"""
import math

import torch
from torch import nn

__all__ = ['RadialControlHead', 'RadialControlUNet']


class RadialControlHead(nn.Module):
    """``u_alpha(t, z)`` from the radial structure.

    Parameters
    ----------
    dim_z : int
        Spatial dimension ``d`` of the state ``z`` (``dim_x - 1``); it enters
        only through the normalized radial feature ``log1p(||z||^2 / d)``.
    width : int
        Hidden width of the scalar MLP (approved value: 32).
    num_hidden : int
        Number of hidden layers (approved value: 2 -> 2->W->W->1).
    act : str
        Activation name resolved against ``torch.nn`` (approved: ``SiLU``).
    enable_autocast, autocast_dtype : mixed-precision switch.

    Shapes: input ``x = (t, z)`` of shape ``(..., dim_z + 1)``; output
    ``(..., dim_z)``, the same layout as ``networks.FCNet``'s control output.
    """

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
        dim_in = 2                     # (t, q)
        for _ in range(self.num_hidden):
            layers.append(nn.Linear(dim_in, self.width))
            layers.append(act_cls())
            dim_in = self.width
        layers.append(nn.Linear(dim_in, 1))
        self.mlp = nn.Sequential(*layers)

        # Zero output layer => h == 0 at initialization => the multiplier is
        # exactly softplus(1)/softplus(1) = 1 and u_alpha starts at
        # -2 z / (1 + ||z||^2) (the exact HJB-1 terminal control).
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        # Keep the nonlinear control factor in fp32 even when the MLP uses
        # bf16 autocast; this is the inference dtype fixed by the protocol.
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
            h = self.mlp(torch.stack((t, q), dim=-1))          # (..., 1)
        # Evaluate the factor and base in fp32; the MLP alone uses bf16 AMP.
        h = h.float().squeeze(-1)
        s = (torch.nn.functional.softplus(1.0 + (1.0 - t) * h)
             / self.softplus1)
        # Enforce the terminal identity exactly in the stored fp32 values.
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
        # _build_net constructs the legacy FCNet before applying its wrapper.
        # Do not retain it as a submodule: only the radial head is trained.
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

    # -- passthrough: AMP flag stays bare-FCNet compatible ------------------
    @property
    def enable_autocast(self):
        return self.head.enable_autocast

    @enable_autocast.setter
    def enable_autocast(self, value):
        self.head.enable_autocast = bool(value)
