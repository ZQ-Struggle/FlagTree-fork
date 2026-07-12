import contextlib
import json
import os
import platform
import sys

platform.python_implementation = lambda: "CPython"
platform.python_version = lambda: "3.11.15"
platform.python_version_tuple = lambda: ("3", "11", "15")

import torch

try:
    import torch_npu
except Exception:
    torch_npu = None

import triton
from triton.runtime import debugger
import flag_gems


def sync_device():
    if torch_npu is not None and hasattr(torch_npu, "npu"):
        torch_npu.npu.synchronize()
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(value):
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (tuple, list)):
        return [summarize(item) for item in value]
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


torch.manual_seed(0)
if torch_npu is not None and hasattr(torch_npu, "npu"):
    torch_npu.npu.manual_seed_all(0)

output_dir = os.environ.get("FLAGTREE_DEBUGGER_BATCH_OUTPUT_DIR")
if output_dir:
    debugger.configure(
        output_dir=output_dir,
        record_capacity=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_RECORD_CAPACITY", "4096")),
        export_raw_records=os.environ.get("FLAGTREE_DEBUGGER_BATCH_EXPORT_RAW", "0") == "1",
    )

triton.enable_debug(
    level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_LEVEL", "1")),
    addr_level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_ADDR_LEVEL", "1")),
)

device = flag_gems.device
include_names = ['bmm_out', 'bmm']
manager = (
    flag_gems.use_gems(include=include_names)
    if include_names
    else contextlib.nullcontext()
)

with manager:
    a = torch.randn((2, 8, 16), dtype=torch.float32, device=device)
    b = torch.randn((2, 16, 8), dtype=torch.float32, device=device)
    out = torch.empty((2, 8, 8), dtype=torch.float32, device=device)
    torch.bmm(a, b, out=out)
    result = out

sync_device()
print(json.dumps({
    "op": 'bmm_out',
    "case_id": 'b2_m8_n8_k16',
    "include_names": include_names,
    "result": summarize(result),
}, sort_keys=True))
