import math
from typing import Union
import torch
import triton
from triton import language as tl

from flag_gems.utils.shape_utils import (
    heuristics_for_tile_size,
    heuristics_for_num_warps,
    stride_order,
)
from flag_gems.utils.tensor_wrapper import StridedBuffer
from flag_gems.utils.libentry import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.runtime import torch_device_fn


def copy_func_wrapper_rank_3(in0: Union[torch.Tensor, StridedBuffer], /, *, out0: Union[torch.Tensor, StridedBuffer]): 
    """Generated wrapper function with Pointwise: StridedBuffer, StridedBuffer(a1!) -> StridedBuffer(a1!)"""
    assert in0.shape == out0.shape, 'operand shapes mismatch'
    # task partitioning
    shape = out0.shape
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0
    tile_sizes = heuristics_for_tile_size(256, *shape)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes))
    num_ctas = min(48, num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = tiles_per_cta==1
    grid = (num_ctas, 1, 1)
    # kernel launch
    in0_strides = in0.stride()
    out0_strides = out0.stride()
    with torch_device_fn.device(in0.device.index):
        copy_func_kernel_rank_3[grid](
            in0, out0,
            in0_strides[0], in0_strides[1], in0_strides[2], # stride for in0
            out0_strides[0], out0_strides[1], out0_strides[2], # stride for out0
            shape[0], shape[1], shape[2], # task indexing space
            num_tasks, # num tasks
            tiles_per_cta=tiles_per_cta, # tiles_per_cta
            tile_size0=tile_sizes[0],
            tile_size1=tile_sizes[1],
            tile_size2=tile_sizes[2],
            one_tile_per_cta=one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0

@triton.jit
def copy_func(x):
    return x

@libentry()
@triton.jit
def copy_func_kernel_rank_3(
    in0_ptr: tl.tensor, # of tl.pointer_type
    out0_ptr: tl.tensor, # of tl.pointer_type
    in0_stride0: tl.constexpr, in0_stride1: tl.constexpr, in0_stride2: tl.constexpr, # strides for in0
    out0_stride0: tl.constexpr, out0_stride1: tl.constexpr, out0_stride2: tl.constexpr, # strides for out0
    s0, s1, s2, # task_space
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr, tile_size1: tl.constexpr, tile_size2: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.debug_collect_start(level=1, addr_level=1)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    num_tiles1 = tl.cdiv(s1, tile_size1)
    num_tiles2 = tl.cdiv(s2, tile_size2)
    if one_tile_per_cta: # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id2 = tile_id % num_tiles2
        tile_id //= num_tiles2
        tile_id1 = tile_id % num_tiles1
        tile_id //= num_tiles1
        tile_id0 = tile_id

        offsets0 = tile_id0 * tile_size0 + tl.arange(0, tile_size0)
        offsets1 = tile_id1 * tile_size1 + tl.arange(0, tile_size1)
        offsets2 = tile_id2 * tile_size2 + tl.arange(0, tile_size2)
        mask0 = offsets0 < s0
        mask1 = offsets1 < s1
        mask2 = offsets2 < s2
        mask = mask0[:, None, None] & mask1[None, :, None] & mask2[None, None, :]
        # loads
        in0 = tl.load(in0_ptr + offsets0[:, None, None] * in0_stride0 + offsets1[None, :, None] * in0_stride1 + offsets2[None, None, :] * in0_stride2, mask=mask).to(in0_ptr.type.element_ty)

        # compute
        out0 = copy_func(in0)

        in0 = tl.store(out0_ptr + offsets0[:, None, None] * out0_stride0 + offsets1[None, :, None] * out0_stride1 + offsets2[None, None, :] * out0_stride2, out0, mask=mask)
    else: # grid-stride-loop style kernel
        num_ctas = ext.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id2 = tile_id % num_tiles2
            tile_id //= num_tiles2
            tile_id1 = tile_id % num_tiles1
            tile_id //= num_tiles1
            tile_id0 = tile_id

            offsets0 = tile_id0 * tile_size0 + tl.arange(0, tile_size0)
            offsets1 = tile_id1 * tile_size1 + tl.arange(0, tile_size1)
            offsets2 = tile_id2 * tile_size2 + tl.arange(0, tile_size2)
            mask0 = offsets0 < s0
            mask1 = offsets1 < s1
            mask2 = offsets2 < s2
            mask = mask0[:, None, None] & mask1[None, :, None] & mask2[None, None, :]
            # loads
            in0 = tl.load(in0_ptr + offsets0[:, None, None] * in0_stride0 + offsets1[None, :, None] * in0_stride1 + offsets2[None, None, :] * in0_stride2, mask=mask).to(in0_ptr.type.element_ty)

            # compute
            out0 = copy_func(in0)

            in0 = tl.store(out0_ptr + offsets0[:, None, None] * out0_stride0 + offsets1[None, :, None] * out0_stride1 + offsets2[None, None, :] * out0_stride2, out0, mask=mask)
    tl.debug_collect_end()

