import logging
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
logger = logging.getLogger(__name__)

@triton.jit
def _alias_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    tl.debug_collect_start(level=1, addr_level=1)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)
    tl.debug_collect_end()

def alias_copy(x: torch.Tensor):
    logger.debug('GEMS ALIAS_COPY')
    '\n    Wrapper for aten::alias_copy\n    Creates and returns a copy of `x` with identical content.\n    '
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()
    if src.dtype != out.dtype:
        raise RuntimeError('alias_copy: dtype mismatch between input and output.')
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_device_fn.device(x.device):
        _alias_copy_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out

def alias_copy_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug('GEMS ALIAS_COPY_OUT')
    '\n    Wrapper for aten::alias_copy.out\n    Copies `x` into `out` and returns `out`.\n    '
    if x.dtype != out.dtype:
        raise RuntimeError('alias_copy_out: dtype of input and output must match.')
    if x.numel() != out.numel():
        raise RuntimeError('alias_copy_out: input and output must have the same number of elements.')
    if x.device != out.device:
        raise RuntimeError('alias_copy_out: input and output must be on the same device.')
    if not out.is_contiguous():
        raise RuntimeError('alias_copy_out: output tensor must be contiguous.')
    src = x.contiguous() if not x.is_contiguous() else x
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_device_fn.device(x.device):
        _alias_copy_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out
