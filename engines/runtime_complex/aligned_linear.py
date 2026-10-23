"""Optional BF16 GEMM workspaces; original parameters/checkpoint shapes stay intact.

Only CUDA autocast uses this path. Pad calculation workspaces with inert zeros
to 16-element leading dimensions, cache detached casts until a parameter's
version changes, and map padded BF16 gradients back to the original parameter.
No parameter is added, removed, reinitialized or trained at a padded coordinate.
"""
import contextlib
import torch
from torch import nn
from torch.nn import functional as F


class _CachedWorkspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, workspace):
        ctx.original_shape = parameter.shape
        ctx.original_dtype = parameter.dtype
        # A separate view receives this invocation's grad_fn; the detached
        # cached tensor itself is never an autograd graph. Optimizer refresh
        # is allowed only after backward and bumps the shared version counter.
        return workspace.view_as(workspace)

    @staticmethod
    def backward(ctx, gradient):
        slices = tuple(slice(0, size) for size in ctx.original_shape)
        # A one-row crop can be logically contiguous while retaining the
        # padded leading stride. Canonicalize its view for DDP's bucket
        # contract; flatten is a view after the dense dtype conversion.
        cropped = gradient[slices].to(ctx.original_dtype)
        return cropped.reshape(-1).view(ctx.original_shape), None


class _PairedWorkspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, workspace):
        ctx.original_shape, ctx.original_dtype = parameter.shape, parameter.dtype
        ctx.set_materialize_grads(False)
        return workspace.view_as(workspace), workspace.view_as(workspace)

    @staticmethod
    def backward(ctx, first, second):
        from fused_rmsprop import sum_workspace_gradients
        return sum_workspace_gradients(first, second, ctx.original_shape,
                                       ctx.original_dtype), None


class AlignedLinear(nn.Linear):
    @staticmethod
    def _workspace_key(parameter, dtype, padded_shape):
        return (id(parameter), parameter.data_ptr(), parameter._version,
                parameter.device, parameter.dtype, dtype, tuple(padded_shape))

    def _cached_value(self, name, parameter, dtype, padded_shape):
        cache = self._aligned_workspace_cache
        key = self._workspace_key(parameter, dtype, padded_shape)
        item = cache.get(name)
        if item is None or item[0] != key:
            pad = []
            for current, target in zip(reversed(parameter.shape),
                                       reversed(padded_shape)):
                pad.extend((0, target - current))
            with torch.no_grad():
                value = F.pad(parameter.to(dtype), tuple(pad))
            cache[name] = (key, value)
        return cache[name][1]

    def _workspace(self, name, parameter, dtype, padded_shape):
        pair = getattr(self, '_paired_parameter_views', None)
        if pair is not None and parameter.requires_grad:
            return pair[name].pop(0)
        value = self._cached_value(name, parameter, dtype, padded_shape)
        return _CachedWorkspace.apply(parameter, value)

    def forward(self, inputs):
        if (inputs.device.type != 'cuda'
                or self.weight.dtype != torch.float32
                or inputs.dtype not in (torch.float32, torch.bfloat16)
                or not torch.is_autocast_enabled('cuda')
                or torch.get_autocast_dtype('cuda') != torch.bfloat16):
            return F.linear(inputs, self.weight, self.bias)
        width_in = ((self.in_features + 15) // 16) * 16
        width_out = ((self.out_features + 15) // 16) * 16
        dtype = torch.bfloat16
        weight = self._workspace('weight', self.weight, dtype,
                                  (width_out, width_in))
        bias = (self._workspace('bias', self.bias, dtype, (width_out,))
                 if self.bias is not None else None)
        values = inputs.to(dtype)
        if (getattr(self, '_aligned_accept_padded_input', False)
                and values.shape[-1] == width_in):
            pass
        else:
            values = F.pad(values, (0, width_in - self.in_features))
        result = F.linear(values, weight, bias)
        return (result if getattr(self, '_aligned_keep_output', False)
                else result[..., :self.out_features])


def enable_aligned_linears(net):
    """Enable workspaces without constructing/reseeding/replacing parameters."""
    from networks import FCNet
    bare = getattr(net, 'module', net)
    linears = [module for module in net.modules() if type(module) is nn.Linear]
    # A plain, row-independent FCNet is a sequential chain. Retain its inert
    # hidden workspace coordinates between layers instead of cropping and
    # re-padding every activation. The final public output is still cropped.
    propagate = type(bare) is FCNet and bare.independent_rows
    for index, module in enumerate(linears):
        module.__class__ = AlignedLinear
        module._aligned_workspace_cache = {}
        module._aligned_keep_output = propagate and index + 1 < len(linears)
        module._aligned_accept_padded_input = propagate and index > 0
    return len(linears)


@contextlib.contextmanager
def paired_parameter_workspaces(net):
    """Two separate BF16 forwards; sum their separately rounded grads in FP32."""
    modules = []
    try:
        for module in net.modules():
            if (not isinstance(module, AlignedLinear)
                    or module.weight.device.type != 'cuda'
                    or module.weight.dtype != torch.float32):
                continue
            old = getattr(module, '_paired_parameter_views', None)
            modules.append((module, old))
            shapes = {'weight': (((module.out_features+15)//16)*16,
                                  ((module.in_features+15)//16)*16),
                      'bias': (((module.out_features+15)//16)*16,)}
            pair = {}
            for name in ('weight', 'bias'):
                parameter = getattr(module, name)
                if parameter is not None and parameter.requires_grad:
                    value = module._cached_value(name, parameter, torch.bfloat16,
                                                   shapes[name])
                    pair[name] = list(_PairedWorkspace.apply(parameter, value))
            module._paired_parameter_views = pair
        yield
    finally:
        for module, old in modules:
            module._paired_parameter_views = old
