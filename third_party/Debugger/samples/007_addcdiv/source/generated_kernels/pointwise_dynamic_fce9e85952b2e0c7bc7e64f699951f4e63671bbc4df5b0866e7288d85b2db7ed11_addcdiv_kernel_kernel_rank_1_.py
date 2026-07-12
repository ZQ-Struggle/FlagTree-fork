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


def addcdiv_kernel_wrapper_rank_1(in0: Union[torch.Tensor, StridedBuffer], in1: Union[torch.Tensor, StridedBuffer], in2: Union[torch.Tensor, StridedBuffer], val0, /, *, out0: Union[torch.Tensor, StridedBuffer]): 
    """Generated wrapper function with Pointwise: StridedBuffer, StridedBuffer, StridedBuffer, scalar, StridedBuffer(a1!) -> StridedBuffer(a1!)"""
    assert in0.shape == in1.shape == in2.shape == out0.shape, 'operand shapes mismatch'
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
    in1_strides = in1.stride()
    in2_strides = in2.stride()
    out0_strides = out0.stride()
    with torch_device_fn.device(in0.device.index):
        addcdiv_kernel_kernel_rank_1[grid](
            in0, in1, in2, val0, out0,
            in0_strides[0], # stride for in0
            in1_strides[0], # stride for in1
            in2_strides[0], # stride for in2
            out0_strides[0], # stride for out0
            shape[0], # task indexing space
            num_tasks, # num tasks
            tiles_per_cta=tiles_per_cta, # tiles_per_cta
            tile_size0=tile_sizes[0],
            one_tile_per_cta=one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0

@triton.jit
def addcdiv_kernel(x, t1, t2, value):
    return x + value * (t1 / t2)

@libentry()
@triton.jit(do_not_specialize=['val0'])
def addcdiv_kernel_kernel_rank_1(
    in0_ptr: tl.tensor, # of tl.pointer_type
    in1_ptr: tl.tensor, # of tl.pointer_type
    in2_ptr: tl.tensor, # of tl.pointer_type
    val0,
    out0_ptr: tl.tensor, # of tl.pointer_type
    in0_stride0: tl.constexpr, # strides for in0
    in1_stride0: tl.constexpr, # strides for in1
    in2_stride0: tl.constexpr, # strides for in2
    out0_stride0: tl.constexpr, # strides for out0
    s0, # task_space
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = ext.program_id(0)
    tl.debug_collect_start(level=1, addr_level=1)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    if one_tile_per_cta: # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id0 = tile_id

        offsets0 = tile_id0 * tile_size0 + tl.arange(0, tile_size0)
        mask0 = offsets0 < s0
        mask = mask0[:]
        # loads
        in0 = tl.load(in0_ptr + offsets0[:] * in0_stride0, mask=mask).to(in0_ptr.type.element_ty)
        in1 = tl.load(in1_ptr + offsets0[:] * in1_stride0, mask=mask).to(in1_ptr.type.element_ty)
        in2 = tl.load(in2_ptr + offsets0[:] * in2_stride0, mask=mask).to(in2_ptr.type.element_ty)

        # compute
        out0 = addcdiv_kernel(in0, in1, in2, val0)

        in0 = tl.store(out0_ptr + offsets0[:] * out0_stride0, out0, mask=mask)
    else: # grid-stride-loop style kernel
        num_ctas = ext.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id0 = tile_id

            offsets0 = tile_id0 * tile_size0 + tl.arange(0, tile_size0)
            mask0 = offsets0 < s0
            mask = mask0[:]
            # loads
            in0 = tl.load(in0_ptr + offsets0[:] * in0_stride0, mask=mask).to(in0_ptr.type.element_ty)
            in1 = tl.load(in1_ptr + offsets0[:] * in1_stride0, mask=mask).to(in1_ptr.type.element_ty)
            in2 = tl.load(in2_ptr + offsets0[:] * in2_stride0, mask=mask).to(in2_ptr.type.element_ty)

            # compute
            out0 = addcdiv_kernel(in0, in1, in2, val0)

            in0 = tl.store(out0_ptr + offsets0[:] * out0_stride0, out0, mask=mask)
    tl.debug_collect_end()

