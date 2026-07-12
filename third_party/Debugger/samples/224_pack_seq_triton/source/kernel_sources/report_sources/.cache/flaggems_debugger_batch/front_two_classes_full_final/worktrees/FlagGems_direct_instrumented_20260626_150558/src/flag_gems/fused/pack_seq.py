import logging
import torch
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def _pack_seq_kernel(x_ptr, out_ptr, lengths_ptr, N: tl.constexpr, D: tl.constexpr, Lmax: tl.constexpr, PAD_VALUE: tl.constexpr, PAD_IS_UINT8: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    tl.debug_collect_start(level=1, addr_level=1)
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)
    t_mask = off_t < Lmax
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]
    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]
    d_mask = off_d[None, :] < D
    if PAD_IS_UINT8:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.uint8)
    else:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)
    tl.debug_collect_end()

def pack_seq_triton(x: torch.Tensor, lengths: torch.Tensor, pad_value: float | int=-float('inf'), block_t: int=64, block_d: int=64) -> torch.Tensor:
    logger.debug('GEMS PACK_SEQ_TRITON')
    is_uint8 = x.dtype == torch.uint8
    if is_uint8:
        assert isinstance(pad_value, int) and 0 <= pad_value <= 255, f'uint8 pack requires an integer pad in [0, 255], got {pad_value!r}'
        pad_constexpr: int | float = int(pad_value)
    else:
        pad_constexpr = float(pad_value)
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x
    B = lengths.numel()
    Lmax = int(lengths.max().item())
    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)
    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel[grid](x_reshaped, out, lengths.int(), N, D, Lmax, PAD_VALUE=pad_constexpr, PAD_IS_UINT8=is_uint8, BLOCK_T=block_t, BLOCK_D=block_d, num_warps=4, num_stages=2)
    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])
    return out
