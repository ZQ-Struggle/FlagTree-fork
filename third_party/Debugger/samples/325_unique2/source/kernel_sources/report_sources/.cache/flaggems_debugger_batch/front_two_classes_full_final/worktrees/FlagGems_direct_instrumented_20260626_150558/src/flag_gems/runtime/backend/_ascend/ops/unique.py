import logging
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry
logger = logging.getLogger(f"flag_gems.runtime._ascend.ops.{__name__.split('.')[-1]}")

@libentry()
@triton.jit
def simple_unique_flat_kernel(sorted_data_ptr: tl.tensor, sorted_indices_ptr: tl.tensor, data_out_ptr: tl.tensor, inverse_indices_ptr: tl.tensor, idx_ptr: tl.tensor, unique_size_ptr: tl.tensor, return_inverse: tl.constexpr, return_counts: tl.constexpr, num_tasks: int, tile_size: tl.constexpr):
    tl.debug_collect_start(level=1, addr_level=1)
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)
    ne_result = tl.where(i0 > 0, a != b, 0)
    cumsum = tl.cumsum(ne_result)
    last_cumsum = tl.sum(cumsum * (i0 == num_tasks - 1).to(tl.int32))
    tl.store(unique_size_ptr, last_cumsum)
    tl.store(data_out_ptr + cumsum, a, mask=mask)
    if return_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)
    tl.debug_collect_end()

@triton.jit
def output_counts_flat_impl(global_pid, idx_ptr: tl.tensor, origin_num_tasks: int, counts_ptr: tl.tensor, num_tasks: int, tile_size: tl.constexpr):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)
    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)
    tl.store(counts_ptr + i0, counts, mask=mask)

@libentry()
@triton.jit
def output_counts_flat_kernel(idx_ptr: tl.tensor, origin_num_tasks: int, counts_ptr: tl.tensor, num_tasks: int, tiles_per_cta: int, tile_size: tl.constexpr):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    tl.debug_collect_start(level=1, addr_level=1)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        output_counts_flat_impl(global_pid, idx_ptr, origin_num_tasks, counts_ptr, num_tasks, tile_size)
    tl.debug_collect_end()

@triton.jit
def quick_output_flat_impl(global_pid, sorted_data_ptr: tl.tensor, idx_ptr: tl.tensor, origin_num_tasks: int, data_out_ptr: tl.tensor, counts_ptr: tl.tensor, num_tasks: int, tile_size: tl.constexpr):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)
    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)
    tl.store(counts_ptr + i0, counts, mask=mask)
    sorted_data = tl.load(sorted_data_ptr + idx, mask=mask)
    tl.store(data_out_ptr + i0, sorted_data, mask=mask)

@libentry()
@triton.jit
def quick_output_flat_kernel(sorted_data_ptr: tl.tensor, idx_ptr: tl.tensor, origin_num_tasks: int, data_out_ptr: tl.tensor, counts_ptr: tl.tensor, num_tasks: int, tiles_per_cta: int, tile_size: tl.constexpr):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    tl.debug_collect_start(level=1, addr_level=1)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        quick_output_flat_impl(global_pid, sorted_data_ptr, idx_ptr, origin_num_tasks, data_out_ptr, counts_ptr, num_tasks, tile_size)
    tl.debug_collect_end()

@triton.jit
def local_quick_unique_flat_impl(global_pid, sorted_data_ptr: tl.tensor, local_unique_ptr: tl.tensor, origin_idx_ptr: tl.tensor, tile_sum_ptr: tl.tensor, global_ctas_num: int, num_tasks: int, tile_size: tl.constexpr, return_counts: tl.constexpr):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask, other=0)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask, other=0)
    ne_result = tl.where(i0 > 0, a != b, 1)
    ne_result = tl.where(mask, ne_result, 0)
    cumsum = tl.cumsum(ne_result)
    local_unique_offset = cumsum - 1
    local_unique_mask = mask
    if return_counts:
        origin_idx_mask = ne_result.to(tl.int1) & local_unique_mask
        tl.store(origin_idx_ptr + (offset + local_unique_offset), i0, mask=origin_idx_mask)
    else:
        store_mask = ne_result.to(tl.int1) & local_unique_mask
        tl.store(local_unique_ptr + (offset + local_unique_offset), a, mask=store_mask)
    valid_cumsum = tl.where(mask, cumsum, 0)
    last_cumsum = tl.max(valid_cumsum)
    if global_pid < global_ctas_num:
        tl.store(tile_sum_ptr + global_pid, last_cumsum)

@libentry()
@triton.jit
def local_quick_unique_flat_kernel(sorted_data_ptr: tl.tensor, local_unique_ptr: tl.tensor, origin_idx_ptr: tl.tensor, tile_sum_ptr: tl.tensor, global_ctas_num: int, num_tasks: int, tiles_per_cta: int, tile_size: tl.constexpr, return_counts: tl.constexpr):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    tl.debug_collect_start(level=1, addr_level=1)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        local_quick_unique_flat_impl(global_pid, sorted_data_ptr, local_unique_ptr, origin_idx_ptr, tile_sum_ptr, global_ctas_num, num_tasks, tile_size, return_counts)
    tl.debug_collect_end()

@triton.jit
def global_quick_unique_flat_impl(global_pid, total, local_unique_ptr: tl.tensor, origin_idx_ptr: tl.tensor, tile_sum_ptr: tl.tensor, data_out_ptr: tl.tensor, idx_ptr: tl.tensor, ctas_num: int, global_ctas_num: int, next_power_global_ctas_num: tl.constexpr, num_tasks: int, tile_size: tl.constexpr, return_counts: tl.constexpr, CHUNK_SIZE: tl.constexpr, MAX_CHUNKS: tl.constexpr):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    start_idx = tl.maximum(global_pid - ctas_num, 0)
    end_idx = tl.minimum(global_pid, global_ctas_num)
    total_sum = 0
    total_sum = total_sum.to(tl.int64)
    for chunk_id in range(MAX_CHUNKS):
        chunk_start = start_idx + chunk_id * CHUNK_SIZE
        if chunk_start < end_idx:
            p = tl.arange(0, CHUNK_SIZE)
            p_idx = chunk_start + p
            pre_tile_sum_mask = (p_idx < end_idx) & (p_idx >= start_idx) & (p_idx < global_ctas_num)
            pre_tile_sum = tl.load(tile_sum_ptr + p_idx, mask=pre_tile_sum_mask, other=0)
            total_sum += tl.sum(pre_tile_sum)
    cur_tile_sum_mask = global_pid < global_ctas_num
    cur_tile_sum = tl.load(tile_sum_ptr + global_pid, mask=cur_tile_sum_mask, other=0)
    total += total_sum
    if global_pid == global_ctas_num - 1:
        tl.store(tile_sum_ptr + global_pid, total + cur_tile_sum)
    tile_mask = r < cur_tile_sum
    out_offset = total + r
    if return_counts:
        origin_idx = tl.load(origin_idx_ptr + i0, mask=mask, other=0)
        tl.store(idx_ptr + out_offset, origin_idx, mask=tile_mask)
    else:
        local_unique = tl.load(local_unique_ptr + i0, mask=mask, other=0)
        tl.store(data_out_ptr + out_offset, local_unique, mask=tile_mask)
    return total

@libentry()
@triton.jit
def global_quick_unique_flat_kernel(local_unique_ptr: tl.tensor, origin_idx_ptr: tl.tensor, tile_sum_ptr: tl.tensor, data_out_ptr: tl.tensor, idx_ptr: tl.tensor, ctas_num: int, global_ctas_num: int, next_power_global_ctas_num: tl.constexpr, num_tasks: int, tiles_per_cta: int, tile_size: tl.constexpr, one_tile_per_cta: tl.constexpr, return_counts: tl.constexpr):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    CHUNK_SIZE: tl.constexpr = 2048
    MAX_CHUNKS: tl.constexpr = 32
    tl.debug_collect_start(level=1, addr_level=1)
    if one_tile_per_cta:
        global_quick_unique_flat_impl(pid, 0, local_unique_ptr, origin_idx_ptr, tile_sum_ptr, data_out_ptr, idx_ptr, ctas_num, global_ctas_num, next_power_global_ctas_num, num_tasks, tile_size, return_counts, CHUNK_SIZE, MAX_CHUNKS)
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_quick_unique_flat_impl(global_pid, total, local_unique_ptr, origin_idx_ptr, tile_sum_ptr, data_out_ptr, idx_ptr, ctas_num, global_ctas_num, next_power_global_ctas_num, num_tasks, tile_size, return_counts, CHUNK_SIZE, MAX_CHUNKS)
    tl.debug_collect_end()

def sorted_quick_unique_flat(sorted_data: torch.Tensor, return_counts: bool):
    num_tasks = sorted_data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    tile_size = min(4096, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num if global_ctas_num < 65536 else 2048
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    num_warps = 8 if tiles_per_cta == 1 else 32
    grid = (ctas_num, 1, 1)
    if return_counts:
        local_unique = None
        origin_idx = torch.empty_like(sorted_data, dtype=torch.int64)
        idx = torch.empty_like(origin_idx)
    else:
        local_unique = torch.empty_like(sorted_data)
        origin_idx = None
        idx = None
        counts = None
    tile_sum = torch.empty((global_ctas_num,), dtype=torch.int64, device=sorted_data.device)
    data_out = None
    if not return_counts:
        data_out = torch.empty_like(sorted_data)
    with torch_device_fn.device(sorted_data.device.index):
        local_quick_unique_flat_kernel[grid](sorted_data, local_unique, origin_idx, tile_sum, global_ctas_num, num_tasks, tiles_per_cta=tiles_per_cta, tile_size=tile_size, return_counts=return_counts, num_warps=num_warps, enable_select_analysis=False)
        global_quick_unique_flat_kernel[grid](local_unique, origin_idx, tile_sum, data_out, idx, ctas_num, global_ctas_num, next_power_global_ctas_num, num_tasks, tiles_per_cta=tiles_per_cta, tile_size=tile_size, one_tile_per_cta=tiles_per_cta == 1, return_counts=return_counts, num_warps=num_warps, enable_select_analysis=False)
        out_size = tile_sum[-1].item()
        if return_counts:
            data_out = torch.empty((out_size,), dtype=sorted_data.dtype, device=sorted_data.device)
            idx = idx[:out_size]
            counts = origin_idx[:out_size]
            quick_output_flat_kernel[grid](sorted_data, idx, num_tasks, data_out, counts, out_size, tiles_per_cta, tile_size, num_warps=num_warps)
    if return_counts:
        return (data_out, None, counts)
    else:
        return (data_out[:out_size], None, None)

@triton.jit
def local_ne_flat_impl(global_pid, sorted_data_ptr: tl.tensor, ne_result_ptr: tl.tensor, tile_sum_ptr: tl.tensor, global_ctas_num: int, num_tasks: int, tile_size: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    tile_start = global_pid * tile_size
    num_sub_blocks = triton.cdiv(tile_size, BLOCK_SIZE_SUB)
    tile_sum_acc = tl.zeros([], dtype=tl.int32)
    for sub_block_idx in range(num_sub_blocks):
        sub_block_start = tile_start + sub_block_idx * BLOCK_SIZE_SUB
        r = tl.arange(0, BLOCK_SIZE_SUB)
        i0 = sub_block_start + r
        mask = (i0 < num_tasks) & (i0 >= 0)
        i0_prev = tl.where(i0 > 0, i0 - 1, 0)
        a = tl.load(sorted_data_ptr + i0, mask=mask, other=0)
        b = tl.load(sorted_data_ptr + i0_prev, mask=mask, other=0)
        ne_result = tl.where(i0 > 0, a != b, 0)
        ne_result = tl.where(mask, ne_result, 0)
        tl.store(ne_result_ptr + i0, ne_result, mask=mask)
        sub_block_sum = tl.sum(ne_result)
        tile_sum_acc += sub_block_sum
    tile_sum_mask = global_pid < global_ctas_num
    tl.store(tile_sum_ptr + global_pid, tile_sum_acc, mask=tile_sum_mask)

@libentry()
@triton.jit
def local_ne_flat_kernel(sorted_data_ptr: tl.tensor, ne_result_ptr: tl.tensor, tile_sum_ptr: tl.tensor, global_ctas_num: int, num_tasks: int, tiles_per_cta: int, tile_size: tl.constexpr):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    tl.debug_collect_start(level=1, addr_level=1)
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        local_ne_flat_impl(global_pid, sorted_data_ptr, ne_result_ptr, tile_sum_ptr, global_ctas_num, num_tasks, tile_size, BLOCK_SIZE_SUB=256)
    tl.debug_collect_end()

@triton.jit
def global_cumsum_flat_impl(global_pid, total, ne_result_ptr: tl.tensor, tile_sum_ptr: tl.tensor, sorted_data_ptr: tl.tensor, sorted_indices_ptr: tl.tensor, data_out_ptr: tl.tensor, inverse_indices_ptr: tl.tensor, idx_ptr: tl.tensor, ctas_num: tl.constexpr, global_ctas_num: int, next_power_global_ctas_num: tl.constexpr, num_tasks: int, tile_size: tl.constexpr, return_counts: tl.constexpr, MAX_CTAS_NUM: tl.constexpr, CHUNK_SIZE: tl.constexpr=512):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks
    sorted_data = tl.load(sorted_data_ptr + i0, mask=mask)
    sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
    start_idx = tl.maximum(global_pid - ctas_num, 0)
    end_idx = tl.minimum(global_pid, global_ctas_num)
    actual_load_size = end_idx - start_idx
    actual_load_size = actual_load_size.to(tl.int64)
    chunk_sum = 0
    chunk_sum = chunk_sum.to(tl.int64)
    for chunk_id in range(tl.cdiv(MAX_CTAS_NUM, CHUNK_SIZE)):
        chunk_start = chunk_id * CHUNK_SIZE
        chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, actual_load_size)
        if chunk_start < actual_load_size:
            p = tl.arange(0, CHUNK_SIZE)
            p_idx = start_idx + chunk_start + p
            pre_tile_sum_mask = (p < chunk_end - chunk_start) & (p_idx >= start_idx) & (p_idx < end_idx) & (p_idx >= 0) & (p_idx < global_ctas_num)
            pre_tile_sum = tl.load(tile_sum_ptr + p_idx, mask=pre_tile_sum_mask, other=0)
            chunk_sum += tl.sum(pre_tile_sum)
    total += chunk_sum
    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result_i1 = ne_result.to(tl.int1)
    ne_result = ne_result.to(tl.int32)
    cumsum = tl.cumsum(ne_result)
    if global_pid == global_ctas_num - 1:
        last_tile_sum_mask = i0 == num_tasks - 1
        tile_sum_val = tl.sum(tl.where(last_tile_sum_mask, total + cumsum, 0).to(tl.int64))
        tl.store(tile_sum_ptr + global_pid, tile_sum_val)
    cumsum += total
    tl.store(data_out_ptr + cumsum, sorted_data, mask=mask)
    tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result_i1) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)
    return total

@libentry()
@triton.jit
def global_cumsum_flat_kernel(ne_result_ptr: tl.tensor, tile_sum_ptr: tl.tensor, sorted_data_ptr: tl.tensor, sorted_indices_ptr: tl.tensor, data_out_ptr: tl.tensor, inverse_indices_ptr: tl.tensor, idx_ptr: tl.tensor, ctas_num: int, global_ctas_num: int, next_power_global_ctas_num: tl.constexpr, num_tasks: int, tiles_per_cta: int, tile_size: tl.constexpr, one_tile_per_cta: tl.constexpr, return_counts: tl.constexpr):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    MAX_CTAS_NUM: tl.constexpr = 65536
    tl.debug_collect_start(level=1, addr_level=1)
    if one_tile_per_cta:
        global_cumsum_flat_impl(pid, 0, ne_result_ptr, tile_sum_ptr, sorted_data_ptr, sorted_indices_ptr, data_out_ptr, inverse_indices_ptr, idx_ptr, ctas_num, global_ctas_num, next_power_global_ctas_num, num_tasks, tile_size, return_counts, MAX_CTAS_NUM)
    else:
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = global_cumsum_flat_impl(global_pid, total, ne_result_ptr, tile_sum_ptr, sorted_data_ptr, sorted_indices_ptr, data_out_ptr, inverse_indices_ptr, idx_ptr, ctas_num, global_ctas_num, next_power_global_ctas_num, num_tasks, tile_size, return_counts, MAX_CTAS_NUM)
    tl.debug_collect_end()

def sorted_indices_unique_flat(sorted_data: torch.Tensor, sorted_indices: torch.Tensor, return_counts: bool):
    num_tasks = sorted_data.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    if num_tasks >= 167772160:
        tile_size = 2048
    else:
        tile_size = min(2048, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num if global_ctas_num < 65536 else 8192
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    grid = (ctas_num, 1, 1)
    ne_result = torch.empty_like(sorted_data, dtype=torch.bool)
    tile_sum = torch.empty((global_ctas_num,), dtype=torch.int64, device=sorted_data.device)
    data_out = torch.empty_like(sorted_data)
    inverse_indices = torch.empty_like(sorted_data, dtype=torch.int64)
    idx = None
    if return_counts:
        idx = torch.empty_like(inverse_indices)
    with torch_device_fn.device(sorted_data.device.index):
        local_ne_flat_kernel[grid](sorted_data, ne_result, tile_sum, global_ctas_num, num_tasks, tiles_per_cta=tiles_per_cta, tile_size=tile_size, enable_select_analysis=False)
        global_cumsum_flat_kernel[grid](ne_result, tile_sum, sorted_data, sorted_indices, data_out, inverse_indices, idx, ctas_num, global_ctas_num, next_power_global_ctas_num, num_tasks, tiles_per_cta=tiles_per_cta, tile_size=tile_size, one_tile_per_cta=tiles_per_cta == 1, return_counts=return_counts, enable_select_analysis=False)
        out_size = tile_sum[-1].item() + 1
        counts = None
        if return_counts:
            idx = idx[:out_size]
            counts = torch.empty_like(idx)
            output_counts_flat_kernel[grid](idx, num_tasks, counts, out_size, tiles_per_cta, tile_size)
    return (data_out[:out_size], inverse_indices, counts)

def simple_unique_flat(sorted_data: torch.Tensor, sorted_indices: torch.Tensor, return_inverse: bool, return_counts: bool):
    num_tasks = sorted_data.numel()
    grid = (1, 1, 1)
    data_out = torch.empty_like(sorted_data)
    if return_inverse:
        inverse_indices = torch.empty_like(sorted_data, dtype=torch.int64)
    else:
        inverse_indices = None
    if return_counts:
        idx = torch.empty_like(sorted_data, dtype=torch.int64)
    else:
        idx = None
    unique_size = torch.empty([1], dtype=torch.int64, device=sorted_data.device)
    with torch_device_fn.device(sorted_data.device.index):
        simple_unique_flat_kernel[grid](sorted_data, sorted_indices, data_out, inverse_indices, idx, unique_size, return_inverse, return_counts, num_tasks, tile_size=triton.next_power_of_2(num_tasks), num_warps=8)
    out_size = unique_size.item() + 1
    counts = None
    if return_counts:
        idx = idx[:out_size]
        counts = torch.empty_like(idx)
        with torch_device_fn.device(sorted_data.device.index):
            output_counts_flat_kernel[grid](idx, num_tasks, counts, num_tasks=out_size, tiles_per_cta=1, tile_size=triton.next_power_of_2(out_size), num_warps=8)
    return (data_out[:out_size], inverse_indices, counts)

def _unique2(in0: torch.Tensor, sorted: bool=True, return_inverse: bool=False, return_counts: bool=False):
    logger.debug('GEMS_ASCEND _UNIQUE2')
    sorted_data, sorted_indices = torch.sort(in0.ravel())
    if in0.numel() <= 8192:
        data_out, inverse_indices, counts = simple_unique_flat(sorted_data, sorted_indices, return_inverse, return_counts)
    else:
        data_out, inverse_indices, counts = sorted_indices_unique_flat(sorted_data, sorted_indices, return_counts)
    return (data_out, inverse_indices if inverse_indices is None else inverse_indices.view_as(in0), counts)
