import logging
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
logger = logging.getLogger(__name__)

@triton.jit
def _view_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    tl.debug_collect_start(level=1, addr_level=1)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)
    tl.debug_collect_end()

def view_copy(x: torch.Tensor, size) -> torch.Tensor:
    logger.debug('GEMS VIEW_COPY')
    '\n    Wrapper for aten::view_copy\n    Creates and returns a copy of `x` with the specified shape.\n    This is like view() but always returns a copy instead of an alias.\n    '
    if isinstance(size, torch.SymInt):
        size = (int(size),)
    elif isinstance(size, (list, tuple)):
        size = tuple((int(s) if isinstance(s, torch.SymInt) else s for s in size))
    n_elements = x.numel()
    if -1 in size:
        if size.count(-1) > 1:
            raise RuntimeError(f'view_copy: only one dimension can be -1, got {size}')
        target_numel_except_minus1 = 1
        for s in size:
            if s != -1:
                target_numel_except_minus1 *= s
        inferred_dim = n_elements // target_numel_except_minus1
        size = tuple((inferred_dim if s == -1 else s for s in size))
    target_numel = 1
    for s in size:
        target_numel *= s
    if n_elements != target_numel:
        raise RuntimeError(f'view_copy: cannot reshape tensor of size {n_elements} into shape {size}')
    if n_elements == 0:
        return torch.empty(size, dtype=x.dtype, device=x.device)
    out = torch.empty(size, dtype=x.dtype, device=x.device)
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_device_fn.device(x.device):
        _view_copy_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out
