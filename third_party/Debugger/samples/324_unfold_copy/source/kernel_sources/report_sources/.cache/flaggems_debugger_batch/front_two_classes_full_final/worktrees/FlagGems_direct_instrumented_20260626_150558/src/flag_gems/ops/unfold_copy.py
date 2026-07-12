import logging
import torch
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def _unfold_copy_kernel_2d(input_ptr, output_ptr, B: tl.constexpr, D: tl.constexpr, L: tl.constexpr, size: tl.constexpr, step: tl.constexpr, numel_out: tl.constexpr, BLOCK: tl.constexpr):
    """
    Optimized kernel for 2D input (B, D), unfolding dimension=1.
    Output: (B, L, size)
    output[b, l, k] = input[b, l*step + k]
    """
    pid = tl.program_id(0)
    tl.debug_collect_start(level=1, addr_level=1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel_out
    k = offs % size
    tmp = offs // size
    l_idx = tmp % L
    b_idx = tmp // L
    pos = l_idx * step + k
    input_idx = b_idx * D + pos
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)
    tl.debug_collect_end()

@triton.jit
def _unfold_copy_kernel_3d_dim2(input_ptr, output_ptr, B: tl.constexpr, D: tl.constexpr, C: tl.constexpr, L: tl.constexpr, size: tl.constexpr, step: tl.constexpr, numel_out: tl.constexpr, BLOCK: tl.constexpr):
    """
    Kernel for 3D input (B, D, C), unfolding dimension=2.
    Output: (B, D, L, size)
    output[b, d, l, k] = input[b, d, l*step + k]
    """
    pid = tl.program_id(0)
    tl.debug_collect_start(level=1, addr_level=1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel_out
    k = offs % size
    tmp = offs // size
    l_idx = tmp % L
    tmp2 = tmp // L
    d_idx = tmp2 % D
    b_idx = tmp2 // D
    pos = l_idx * step + k
    input_idx = b_idx * D * C + d_idx * C + pos
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)
    tl.debug_collect_end()

@triton.jit
def _unfold_copy_kernel_3d_dim1(input_ptr, output_ptr, B: tl.constexpr, D: tl.constexpr, C: tl.constexpr, L: tl.constexpr, size: tl.constexpr, step: tl.constexpr, numel_out: tl.constexpr, BLOCK: tl.constexpr):
    """
    Kernel for 3D input (B, D, C), unfolding dimension=1.
    Output: (B, L, C, size)
    output[b, l, c, k] = input[b, l*step + k, c]
    """
    pid = tl.program_id(0)
    tl.debug_collect_start(level=1, addr_level=1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel_out
    k = offs % size
    tmp = offs // size
    c_idx = tmp % C
    tmp2 = tmp // C
    l_idx = tmp2 % L
    b_idx = tmp2 // L
    pos = l_idx * step + k
    input_idx = b_idx * D * C + pos * C + c_idx
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)
    tl.debug_collect_end()

def unfold_copy(input: torch.Tensor, dimension: int, size: int, step: int) -> torch.Tensor:
    logger.debug('GEMS UNFOLD COPY')
    if step <= 0:
        raise ValueError('step must be > 0')
    ndim = input.ndim
    d = dimension % ndim
    D = input.shape[d]
    if size > D:
        raise ValueError('size must be <= dimension size')
    L = (D - size) // step + 1
    output_shape = input.shape[:d] + (L,) + input.shape[d + 1:] + (size,)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    numel_out = output.numel()
    BLOCK = 128
    grid = lambda meta: (triton.cdiv(numel_out, meta['BLOCK']),)
    if ndim == 2 and d == 1:
        B = input.shape[0]
        D = input.shape[1]
        _unfold_copy_kernel_2d[grid](input, output, B, D, L, size, step, numel_out, BLOCK=BLOCK)
    elif ndim == 3 and d == 2:
        B = input.shape[0]
        D = input.shape[1]
        C = input.shape[2]
        _unfold_copy_kernel_3d_dim2[grid](input, output, B, D, C, L, size, step, numel_out, BLOCK=BLOCK)
    elif ndim == 3 and d == 1:
        B = input.shape[0]
        D = input.shape[1]
        C = input.shape[2]
        _unfold_copy_kernel_3d_dim1[grid](input, output, B, D, C, L, size, step, numel_out, BLOCK=BLOCK)
    else:
        raise NotImplementedError(f'unfold_copy is only implemented for 2D tensors with dim=1 and 3D tensors with dim=1 or dim=2. Got ndim={ndim}, dim={d}')
    return output
