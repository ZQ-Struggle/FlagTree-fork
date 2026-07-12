import logging
import torch
import triton
import triton.language as tl
from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
logger = logging.getLogger(__name__)

@triton.autotune(configs=runtime.get_tuned_config('rot90'), key=['n_elements'])
@triton.jit
def rot90_kernel_2d(in_ptr, out_ptr, n_elements, M, N, k_norm, BLOCK_SIZE: tl.constexpr):
    """
    rot90 kernel for rotating a tensor by 90 degrees in the plane [0, 1].

    Input shape: [M, N, D2, D3, ...]
    Output shape for k=1,3: [N, M, D2, D3, ...]
    Output shape for k=0,2: [M, N, D2, D3, ...]

    Formulas (verified):
    - k=0 (identity): out[i,j] = in[i,j] -> in_dim0=out_dim0, in_dim1=out_dim1
    - k=1 (90° clockwise): out[i,j] = in[j, N-1-i]
      -> in_dim0=out_dim1, in_dim1=N-1-out_dim0
    - k=2 (180°): out[i,j] = in[M-1-i, N-1-j]
      -> in_dim0=M-1-out_dim0, in_dim1=N-1-out_dim1
    - k=3 (270° clockwise / 90° CCW): out[i,j] = in[M-1-j, i]
      -> in_dim0=M-1-out_dim1, in_dim1=out_dim0
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    tl.debug_collect_start(level=1, addr_level=1)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    m_minus_1 = M - 1
    n_minus_1 = N - 1
    if k_norm == 0:
        stride_0 = n_elements // M
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % N
        in_dim0 = out_dim0
        in_dim1 = out_dim1
        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)
    elif k_norm == 1:
        stride_0 = n_elements // N
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % M
        in_dim0 = out_dim1
        in_dim1 = n_minus_1 - out_dim0
        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)
    elif k_norm == 2:
        stride_0 = n_elements // M
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % N
        in_dim0 = m_minus_1 - out_dim0
        in_dim1 = n_minus_1 - out_dim1
        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)
    else:
        stride_0 = n_elements // N
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % M
        in_dim0 = m_minus_1 - out_dim1
        in_dim1 = out_dim0
        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)
    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)
    tl.debug_collect_end()

def rot90_2d(inp, k, dims, out):
    """Handle the case when dims = [0, 1] using optimized Triton kernel."""
    M = inp.shape[dims[0]]
    N = inp.shape[dims[1]]
    n_elements = out.numel()
    if n_elements == 0:
        return
    k_norm = (k % 4 + 4) % 4
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    with torch_device_fn.device(inp.device):
        rot90_kernel_2d[grid](inp, out, n_elements, M, N, k_norm)

def rot90(input, k=1, dims=[0, 1]):
    """
    Rotate an n-D tensor by 90 degrees in the plane specified by dims.

    Args:
        input: the input tensor
        k: number of times to rotate (default: 1)
        dims: axis to rotate (default: [0, 1])

    Returns:
        Rotated tensor
    """
    logger.debug('GEMS ROT90')
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    dim0, dim1 = (dims[0], dims[1])
    M = x.shape[dim0]
    N = x.shape[dim1]
    k_norm = (k % 4 + 4) % 4
    if k_norm == 0 or k_norm == 2:
        out_shape = list(x.shape)
    else:
        out_shape = list(x.shape)
        out_shape[dim0] = N
        out_shape[dim1] = M
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    if dim0 == 0 and dim1 == 1:
        rot90_2d(x, k, dims, out)
    else:
        ndim = x.ndim
        perm = [dim0, dim1]
        for i in range(ndim):
            if i != dim0 and i != dim1:
                perm.append(i)
        inverse_perm = [0] * ndim
        inverse_perm[dim0] = 0
        inverse_perm[dim1] = 1
        idx = 2
        for i in range(ndim):
            if i != dim0 and i != dim1:
                inverse_perm[i] = idx
                idx += 1
        x_transposed = x.permute(perm)
        out_transposed = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        rot90_2d(x_transposed, k, [0, 1], out_transposed)
        out.copy_(out_transposed.permute(inverse_perm))
    return out
