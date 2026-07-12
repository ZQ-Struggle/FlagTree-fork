import logging
import torch
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def _unsafe_masked_index_kernel(self_ptr, mask_ptr, indices_ptr, out_ptr, self_numel, N, fill_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tl.debug_collect_start(level=1, addr_level=1)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0)
    indices_val = tl.load(indices_ptr + offsets, mask=mask, other=0)
    self_val = tl.load(self_ptr + indices_val, mask=mask, other=0.0)
    result = tl.where(mask_val, self_val, fill_val)
    tl.store(out_ptr + offsets, result, mask=mask)
    tl.debug_collect_end()

def _unsafe_masked_index(self, mask, indices, fill):
    logger.debug('GEMS _UNSAFE_MASKED_INDEX')
    if not indices or indices[0] is None:
        raise ValueError('indices cannot be empty or None')
    indices_tensor = indices[0]
    output_shape = mask.shape
    out = torch.empty_like(mask, dtype=self.dtype, device=self.device)
    self_flat = self.reshape(-1)
    indices_flat = indices_tensor.reshape(-1)
    mask_flat = mask.reshape(-1)
    out_flat = out.reshape(-1)
    N = mask_flat.numel()
    BLOCK_SIZE = 512
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    if isinstance(fill, (int, float)):
        fill_val = float(fill)
    elif isinstance(fill, torch.Tensor):
        fill_val = fill.item() if fill.numel() == 1 else 0.0
    else:
        fill_val = 0.0
    _unsafe_masked_index_kernel[grid](self_flat, mask_flat, indices_flat, out_flat, self_flat.shape[0], N, fill_val, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(output_shape)
