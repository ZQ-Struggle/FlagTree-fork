import logging
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
logger = logging.getLogger(__name__)

@libentry()
@triton.jit
def histc_kernel(inp_ptr, out_ptr, n_elements, bins: tl.constexpr, min_val, max_val, BLOCK_SIZE: tl.constexpr):
    """
    Compute histogram of input tensor.
    Each thread processes BLOCK_SIZE elements, computing which bin they belong to
    and atomically incrementing the corresponding bin counter.
    """
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=0.0)
    inp_val = inp_val.to(tl.float32)
    bin_width = (max_val - min_val) / bins
    bin_idx = ((inp_val - min_val) / bin_width).to(tl.int32)
    in_range = (inp_val >= min_val) & (inp_val <= max_val)
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)
    valid_mask = mask & in_range
    for i in range(BLOCK_SIZE):
        if tl.load(valid_mask.to(tl.int8).reshape(BLOCK_SIZE) + i) != 0:
            idx = tl.load(bin_idx.reshape(BLOCK_SIZE) + i)
            tl.atomic_add(out_ptr + idx, 1.0, sem='relaxed')

@libentry()
@triton.jit
def histc_kernel_simple(inp_ptr, out_ptr, n_elements, bins, min_val, max_val, BLOCK_SIZE: tl.constexpr):
    """
    Simple histogram kernel - each program handles one element at a time.
    """
    pid = ext.program_id(0)
    tl.debug_collect_start(level=1, addr_level=1)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_val = tl.load(inp_ptr + offset, mask=mask, other=float('nan'))
    inp_val = inp_val.to(tl.float32)
    bin_idx = tl.floor((inp_val - min_val) * bins / (max_val - min_val)).to(tl.int64)
    bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
    in_range = (inp_val >= min_val) & (inp_val <= max_val)
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)
    valid_mask = mask & in_range
    tl.atomic_add(out_ptr + bin_idx, 1.0, mask=valid_mask, sem='relaxed')
    tl.debug_collect_end()

def histc(inp, bins=100, min=0, max=0):
    """
    Compute the histogram of a tensor.

    Args:
        inp: Input tensor
        bins: Number of histogram bins (default: 100)
        min: Lower end of the range (inclusive). If min == max == 0, uses data min.
        max: Upper end of the range (inclusive). If min == max == 0, uses data max.

    Returns:
        Tensor: Histogram represented as a tensor of shape (bins,)
    """
    logger.debug('GEMS HISTC')
    inp = inp.contiguous()
    min_val = float(min)
    max_val = float(max)
    if min_val == 0 and max_val == 0:
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())
    if min_val == max_val:
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        out[0] = count
        return out
    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
    n_elements = inp.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(inp.device):
        histc_kernel_simple[grid](inp, out, n_elements, bins, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)
    return out
