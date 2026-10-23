"""Opt-in bandwidth-efficient RMSprop for the original CUDA FP32 case.

The installed PyTorch2.11 CUDA pointwise implementation uses explicit FMA
for addcmul/addcdiv. Keep those rounding boundaries and accurate sqrt/divide;
fuse only the memory traffic. Unsupported options use the original optimizer.
"""
import torch
import weakref
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rmsprop_update(parameter, gradient, square_average, workspace, count: tl.constexpr,
                    alpha, complement, epsilon, negative_lr,
                    ROW_WIDTH: tl.constexpr, PADDED_WIDTH: tl.constexpr,
                    UPDATE_WORKSPACE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < count
    value = tl.load(parameter + index, mask, 0)
    grad = tl.load(gradient + index, mask, 0)
    average = tl.load(square_average + index, mask, 0)
    scaled = average * alpha
    squared_grad = grad * grad
    average = libdevice.fma_rn(complement, squared_grad, scaled)
    denominator = libdevice.sqrt_rn(average) + epsilon
    quotient = libdevice.div_rn(grad, denominator)
    updated = libdevice.fma_rn(negative_lr, quotient, value)
    tl.store(square_average + index, average, mask)
    tl.store(parameter + index, updated, mask)
    if UPDATE_WORKSPACE:
        padded_index = (index // ROW_WIDTH) * PADDED_WIDTH + index % ROW_WIDTH
        tl.store(workspace + padded_index,
                 updated.to(tl.bfloat16, fp_downcast_rounding='rtne'), mask)


@triton.jit
def _sum_workspace_gradients(first, second, output, count: tl.constexpr,
                             WIDTH: tl.constexpr, S10: tl.constexpr, S11: tl.constexpr,
                             S20: tl.constexpr, S21: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // WIDTH, index % WIDTH
    one = tl.load(first + row*S10 + column*S11, index < count, 0).to(tl.float32)
    two = tl.load(second + row*S20 + column*S21, index < count, 0).to(tl.float32)
    tl.store(output + index, one + two, index < count)


def sum_workspace_gradients(first, second, shape, dtype):
    """Separate BF16 GEMM adjoints, converted and summed in original FP32."""
    slices = tuple(slice(0, size) for size in shape)
    if first is None or second is None:
        gradient = second if first is None else first
        return (None if gradient is None else
                gradient[slices].to(dtype).reshape(-1).view(shape))
    if (dtype != torch.float32 or not first.is_cuda
            or first.dtype != torch.bfloat16 or second.dtype != torch.bfloat16
            or len(shape) not in (1, 2)):
        return first[slices].to(dtype) + second[slices].to(dtype)
    output = torch.empty(shape, dtype=dtype, device=first.device)
    stride10 = first.stride(-2) if len(shape) == 2 else 0
    stride20 = second.stride(-2) if len(shape) == 2 else 0
    _sum_workspace_gradients[(triton.cdiv(output.numel(), 2048),)](
        first, second, output, output.numel(), WIDTH=shape[-1],
        S10=stride10, S11=first.stride(-1), S20=stride20, S21=second.stride(-1),
        BLOCK=2048, num_warps=4, enable_fp_fusion=False)
    return output


def register_aligned_workspaces(optimizers, nets):
    """Link ephemeral cast buffers, leaving optimizer state_dict unchanged."""
    from aligned_linear import AlignedLinear
    owners = {}
    for net in nets.values():
        for module in net.modules():
            if isinstance(module, AlignedLinear):
                for name in ('weight', 'bias'):
                    parameter = getattr(module, name)
                    if parameter is not None:
                        owners[parameter] = (weakref.ref(module), name)
    for optimizer in optimizers:
        if isinstance(optimizer, FusedRMSprop):
            optimizer.efficiency_workspace_owners = {
                parameter: owners[parameter]
                for group in optimizer.param_groups for parameter in group['params']
                if parameter in owners}
    if 'v_theta' in nets:
        getattr(nets['v_theta'], 'module', nets['v_theta']).efficiency_pair_casts = True


class FusedRMSprop(torch.optim.RMSprop):
    """Same RMSprop options/state; fast path only for the validated subset."""

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            return super().step(closure)
        active = []
        for group in self.param_groups:
            if (group['momentum'] != 0 or group['weight_decay'] != 0
                    or group['centered'] or group.get('maximize', False)
                    or group.get('capturable', False)
                    or group.get('differentiable', False)
                    or isinstance(group['lr'], torch.Tensor)):
                return super().step()
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if (not parameter.is_cuda or parameter.dtype != torch.float32
                        or gradient.dtype != torch.float32 or gradient.is_sparse
                        or not parameter.is_contiguous() or not gradient.is_contiguous()):
                    return super().step()
                active.append((group, parameter, gradient))
        bumped, refreshed = [], []
        for group, parameter, gradient in active:
            state = self.state[parameter]
            if not state:
                scalar_dtype = (torch.float64 if torch.get_default_dtype() == torch.float64
                                else torch.float32)
                state['step'] = torch.zeros((), dtype=scalar_dtype, device='cpu')
                state['square_avg'] = torch.zeros_like(parameter)
            state['step'].add_(1)
            average = state['square_avg']
            owner = getattr(self, 'efficiency_workspace_owners', {}).get(parameter)
            module = owner[0]() if owner is not None else None
            item = (module._aligned_workspace_cache.get(owner[1])
                    if module is not None else None)
            workspace = item[1] if item is not None else parameter
            update_workspace = (item is not None and workspace.dtype == torch.bfloat16
                                and workspace.device == parameter.device
                                and getattr(module, owner[1]) is parameter
                                and workspace.ndim == parameter.ndim
                                and all(a >= b for a, b in zip(workspace.shape, parameter.shape)))
            with torch.cuda.device(parameter.device):
                _rmsprop_update[(triton.cdiv(parameter.numel(), 1024),)](
                    parameter, gradient, average, workspace, parameter.numel(),
                    group['alpha'], 1. - group['alpha'], group['eps'], -group['lr'],
                    ROW_WIDTH=parameter.shape[-1] if parameter.ndim else 1,
                    PADDED_WIDTH=workspace.shape[-1] if workspace.ndim else 1,
                    UPDATE_WORKSPACE=update_workspace,
                    BLOCK=1024, num_warps=4, enable_fp_fusion=False)
            # Raw Triton writes are invisible to PyTorch's version counter.
            # Invalidate BF16 parameter caches exactly as an ordinary step.
            bumped.extend((parameter, average, average))
            if update_workspace:
                bumped.append(workspace)
                refreshed.append((module, owner[1], parameter, workspace))
        if bumped:
            torch.autograd.graph.increment_version(bumped)
        for module, name, parameter, workspace in refreshed:
            key = module._workspace_key(parameter, workspace.dtype, workspace.shape)
            module._aligned_workspace_cache[name] = (key, workspace)
        return None
