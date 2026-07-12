import logging
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry
logger = logging.getLogger(__name__)

@libentry()
@triton.jit
def split_copy_kernel(out_ptr, inp_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Copy elements from input to output.
    Both input and output are expected to be contiguous and have the same shape.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    tl.debug_collect_start(level=1, addr_level=1)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < n_elements
    data = tl.load(inp_ptr + block_start + offsets, mask=mask)
    tl.store(out_ptr + block_start + offsets, data, mask=mask)
    tl.debug_collect_end()

def split_with_sizes_copy(inp, split_sizes, dim=0):
    logger.debug('GEMS SPLIT_WITH_SIZES_COPY')
    if dim < 0:
        dim = dim + inp.ndim
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    if hasattr(split_sizes, '__iter__'):
        split_sizes = list(split_sizes)
    result = []
    offset = 0
    for size in split_sizes:
        if size == 0:
            out_shape = list(inp.shape)
            out_shape[dim] = 0
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            result.append(out)
            continue
        split_view = inp.narrow(dim, offset, size)
        split_view = split_view.contiguous()
        out = torch.empty_like(split_view)
        n_elements = out.numel()
        if n_elements > 0:
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            split_copy_kernel[grid](out, split_view, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        result.append(out)
        offset += size
    return tuple(result)
