import logging
from typing import List
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry
logger = logging.getLogger(__name__)

@libentry()
@triton.autotune(configs=[triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_stages=4, num_warps=4), triton.Config({'BLOCK_H': 32, 'BLOCK_W': 16}, num_stages=3, num_warps=4), triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32}, num_stages=3, num_warps=4), triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_stages=2, num_warps=8), triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_stages=5, num_warps=2), triton.Config({'BLOCK_H': 16, 'BLOCK_W': 8}, num_stages=5, num_warps=2), triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}, num_stages=5, num_warps=2), triton.Config({'BLOCK_H': 64, 'BLOCK_W': 16}, num_stages=2, num_warps=8), triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64}, num_stages=2, num_warps=8)], key=['out_h', 'out_w', 'kernel_h', 'kernel_w', 'stride_h', 'stride_w'])
@triton.jit
def col2im_kernel(input_ptr, output_ptr, in_stride_n, in_stride_ck, in_stride_l, out_stride_n, out_stride_c, out_stride_h, out_stride_w, batch_size, channels, out_h, out_w, L_h, L_w, kernel_h: tl.constexpr, kernel_w: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr, padding_h: tl.constexpr, padding_w: tl.constexpr, dilation_h: tl.constexpr, dilation_w: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    tl.debug_collect_start(level=1, addr_level=1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // channels
    c_idx = pid_nc % channels
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    input_base_ptr = input_ptr + n_idx * in_stride_n
    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_num = h_out_offsets[:, None] + padding_h - kh * dilation_h
            w_num = w_out_offsets[None, :] + padding_w - kw * dilation_w
            h_valid = h_num % stride_h == 0
            w_valid = w_num % stride_w == 0
            l_h = h_num // stride_h
            l_w = w_num // stride_w
            l_h_valid = (l_h >= 0) & (l_h < L_h)
            l_w_valid = (l_w >= 0) & (l_w < L_w)
            valid_mask = h_valid & w_valid & l_h_valid & l_w_valid
            c_k = c_idx * kernel_h * kernel_w + kh * kernel_w + kw
            l_idx = l_h * L_w + l_w
            input_offset = c_k * in_stride_ck + l_idx * in_stride_l
            input_val = tl.load(input_base_ptr + input_offset, mask=valid_mask, other=0.0)
            sum_acc += input_val
    out_base_ptr = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    out_offset = h_out_offsets[:, None] * out_stride_h + w_out_offsets[None, :] * out_stride_w
    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(out_base_ptr + out_offset, sum_acc.to(output_ptr.type.element_ty), mask=out_mask)
    tl.debug_collect_end()

def _parse_col2im_params(output_size, kernel_size, dilation, padding, stride):
    """Parse and validate col2im parameters."""

    def _to_pair(val, name):
        if isinstance(val, int):
            return (val, val)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return tuple(val)
        raise ValueError(f'Invalid {name}: {val}')
    out_h, out_w = _to_pair(output_size, 'output_size')
    kernel_h, kernel_w = _to_pair(kernel_size, 'kernel_size')
    dilation_h, dilation_w = _to_pair(dilation, 'dilation')
    padding_h, padding_w = _to_pair(padding, 'padding')
    stride_h, stride_w = _to_pair(stride, 'stride')
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f'stride must be positive, got ({stride_h}, {stride_w})')
    if padding_h < 0 or padding_w < 0:
        raise ValueError(f'padding must be non-negative, got ({padding_h}, {padding_w})')
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(f'dilation must be positive, got ({dilation_h}, {dilation_w})')
    return (out_h, out_w, kernel_h, kernel_w, dilation_h, dilation_w, padding_h, padding_w, stride_h, stride_w)

def col2im(input: torch.Tensor, output_size: List[int], kernel_size: List[int], dilation: List[int], padding: List[int], stride: List[int]) -> torch.Tensor:
    """
    Combines an array of sliding local blocks into a large containing tensor.

    This is the reverse operation of im2col (unfold).

    Args:
        input: Input tensor of shape (N, C * kernel_h * kernel_w, L)
               where L is the number of sliding blocks.
        output_size: Shape of the output spatial dimensions (height, width).
        kernel_size: Size of the sliding blocks (height, width).
        dilation: Dilation of the sliding blocks (height, width).
        padding: Padding added to both sides of the input (height, width).
        stride: Stride of the sliding blocks (height, width).

    Returns:
        Output tensor of shape (N, C, output_h, output_w).
    """
    logger.debug('GEMS COL2IM')
    out_h, out_w, kernel_h, kernel_w, dilation_h, dilation_w, padding_h, padding_w, stride_h, stride_w = _parse_col2im_params(output_size, kernel_size, dilation, padding, stride)
    if input.dim() != 3:
        raise ValueError(f'Expected 3D input, got {input.dim()}D')
    batch_size, ck, L = input.shape
    L_h = (out_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    L_w = (out_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    expected_L = L_h * L_w
    if L != expected_L:
        raise ValueError(f'Input size mismatch: expected L={expected_L} (L_h={L_h}, L_w={L_w}), got L={L}')
    kernel_size_total = kernel_h * kernel_w
    if ck % kernel_size_total != 0:
        raise ValueError(f'Input dimension 1 ({ck}) must be divisible by kernel_size ({kernel_size_total})')
    channels = ck // kernel_size_total
    input = input.contiguous()
    output = torch.empty((batch_size, channels, out_h, out_w), device=input.device, dtype=input.dtype)
    if output.numel() == 0:
        return output
    grid = lambda meta: (batch_size * channels, triton.cdiv(out_h, meta['BLOCK_H']) * triton.cdiv(out_w, meta['BLOCK_W']))
    col2im_kernel[grid](input, output, input.stride(0), input.stride(1), input.stride(2), output.stride(0), output.stride(1), output.stride(2), output.stride(3), batch_size, channels, out_h, out_w, L_h, L_w, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w)
    return output
