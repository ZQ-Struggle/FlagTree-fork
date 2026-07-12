import logging
import torch
import triton
import triton.language as tl
import flag_gems
logger = logging.getLogger(__name__)

@triton.jit
def prelu_kernel(x_ptr, w_ptr, out_ptr, n_elements, S, C, w_is_scalar: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    tl.debug_collect_start(level=1, addr_level=1)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if w_is_scalar:
        alpha = tl.load(w_ptr)
        y = tl.where(x >= 0, x, alpha * x)
    else:
        c = offsets // S % C
        alpha = tl.load(w_ptr + c, mask=mask)
        y = tl.where(x >= 0, x, alpha * x)
    tl.store(out_ptr + offsets, y, mask=mask)
    tl.debug_collect_end()

def prelu(*args, **kwargs):
    logger.debug('GEMS PRELU')
    if len(args) >= 2:
        x, weight = (args[0], args[1])
    else:
        x = kwargs.get('input', kwargs.get('self'))
        weight = kwargs.get('weight')
    if x is None or weight is None:
        raise ValueError('prelu expects (input, weight) as arguments.')
    if x.device.type != flag_gems.device or weight.device.type != flag_gems.device:
        raise AssertionError(f'Tensors must be {flag_gems.device} tensors.')
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    ndim = x.dim()
    if weight.numel() == 1:
        C = 1
        S = 1
        w_is_scalar = True
    else:
        if ndim == 0:
            raise AssertionError('Non-scalar weight provided for a 0-dim input.')
        if ndim == 1:
            C = x.shape[0]
            S = 1
        else:
            C = x.shape[1]
            S = 1
            if ndim > 2:
                for d in x.shape[2:]:
                    S *= d
        if weight.numel() != C:
            raise AssertionError(f'Weight numel ({weight.numel()}) must equal channel dimension size ({C}).')
        w_is_scalar = False
    C = max(int(C), 1)
    S = max(int(S), 1)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    prelu_kernel[grid](x, weight, out, n_elements, S, C, w_is_scalar=w_is_scalar, BLOCK_SIZE=BLOCK_SIZE)
    return out
