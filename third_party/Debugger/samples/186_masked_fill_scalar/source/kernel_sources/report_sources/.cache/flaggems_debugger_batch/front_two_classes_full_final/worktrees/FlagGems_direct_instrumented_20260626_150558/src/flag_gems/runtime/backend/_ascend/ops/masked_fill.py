import logging
import torch
import triton
import triton.language as tl
from flag_gems import runtime
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as ext
logger = logging.getLogger(f"flag_gems.runtime._ascend.ops.{__name__.split('.')[-1]}")

@libentry()
@triton.autotune(configs=runtime.get_tuned_config('masked_fill'), key=['N'])
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = ext.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB
    tl.debug_collect_start(level=1, addr_level=1)
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        tl.store(out + offsets, input_vals, mask=mask)
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0))
        tl.store(out + offsets, overwrite_vals, mask=mask)
    tl.debug_collect_end()

@libentry()
@triton.autotune(configs=runtime.get_tuned_config('masked_fill'), key=['N'])
@triton.jit
def masked_fill_kernel_self(inp, expand_mask, value, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = ext.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB
    tl.debug_collect_start(level=1, addr_level=1)
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        orig = tl.load(inp + offsets, mask=mask, other=0)
        value_vec = tl.full([BLOCK_SIZE_SUB], value, dtype=orig.dtype)
        result = tl.where(fill_mask, value_vec, orig)
        tl.store(inp + offsets, result, mask=mask)
    tl.debug_collect_end()

def masked_fill(inp, mask, value):
    logger.debug('GEMS_ASCEND MASKED FILL')
    assert torch.is_tensor(value) and value.ndim == 0 or isinstance(value, int) or isinstance(value, float), 'masked_fill_ only supports a 0-dimensional value tensor'
    if torch.is_tensor(value):
        value = value.item()
    assert broadcastable_to(mask.shape, inp.shape), 'The shape of mask must be broadcastable with the shape of the underlying tensor'
    if inp.ndim == 0:
        return torch.tensor(value, dtype=inp.dtype, device=inp.device) if mask.item() else inp.clone()
    inp = inp.contiguous()
    mask = mask.contiguous()
    expand_mask = mask.expand(inp.shape)
    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)
    N = inp.numel()
    if N == 0:
        return out
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N)
    return out

def masked_fill_(inp, mask, value):
    logger.debug('GEMS_ASCEND MASKED FILL_')
    assert torch.is_tensor(value) and value.ndim == 0 or isinstance(value, int) or isinstance(value, float), 'masked_fill_ only supports a 0-dimensional value tensor'
    if torch.is_tensor(value):
        value = value.item()
    assert broadcastable_to(mask.shape, inp.shape), 'The shape of mask must be broadcastable with the shape of the underlying tensor'
    if inp.ndim == 0:
        if mask.item():
            inp[()] = value
        return inp
    inp = inp.contiguous()
    mask = mask.contiguous()
    expand_mask = mask.expand(inp.shape)
    N = inp.numel()
    if N == 0:
        return inp
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    masked_fill_kernel_self[grid](inp, expand_mask.to(torch.int), value, N)
    return inp
