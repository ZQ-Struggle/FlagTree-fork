import os
import platform
import sys

platform.python_implementation = lambda: "CPython"
platform.python_version = lambda: "3.11.15"
platform.python_version_tuple = lambda: ("3", "11", "15")

from triton.runtime import debugger
import triton
import pytest

debugger.configure(
    output_dir=os.environ["FLAGTREE_DEBUGGER_NODE_OUTPUT_DIR"],
    record_capacity=int(os.environ.get("FLAGTREE_DEBUGGER_NODE_RECORD_CAPACITY", "4096")),
    export_raw_records=os.environ.get("FLAGTREE_DEBUGGER_NODE_EXPORT_RAW", "0") == "1",
)
triton.enable_debug(
    level=int(os.environ.get("FLAGTREE_DEBUGGER_NODE_LEVEL", "1")),
    addr_level=int(os.environ.get("FLAGTREE_DEBUGGER_NODE_ADDR_LEVEL", "1")),
)

nodeid = os.environ["FLAGTREE_DEBUGGER_NODEID"]
pytest_args = [
    "-q",
    nodeid,
    "--quick",
    "--ref=cpu",
    "--record=json",
    "--output",
    os.environ["FLAGTREE_DEBUGGER_NODE_PYTEST_JSON"],
    "-s",
]
raise SystemExit(pytest.main(pytest_args))
