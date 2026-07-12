# FlagGems Debugger Samples

This directory contains reproducible FlagGems debugger test entries and kernel
source code. A sample does not have to pass: timeout, compile failure, runtime
failure, and missing-report cases are retained as debugger regression inputs.

Generated text and JSON reports are intentionally not stored here. Their
historical status and counts remain in `run_info.json`, while all available
kernel source files are retained under `source/`.

## Indexes

- `index.json`: every reproducible test entry.
- `stable_index.json`: the original passing regression baseline.
- `coverage_index.json`: additional runnable entries, including non-passing
  and not-yet-run cases.
- `uncovered_ops.json`: FlagGems inventory entries for which neither a direct
  case nor a matching pytest node is currently available.
- `INDEX.md`: human-readable summary of all entries and their last known
  status.

## Sample Layout

Each indexed directory contains:

- `run_info.json`: op, case or pytest node, status, timeout policy, and source
  provenance.
- `source/case.py`: standalone direct-op driver, when available.
- `source/runner.py` and `source/test_source.py`: pytest-node driver and the
  selected FlagGems test source.
- `source/kernel_sources/`: matching FlagGems implementation files, including
  source files migrated from previous reports.
- `source/generated_kernels/`: generated pointwise Triton source, when it was
  available from the run cache.

`archive_testable_kernels.py` rebuilds the indexes and imports new direct and
pytest coverage entries from the configured batch-run artifacts. Existing
sample directories are never overwritten.

`prune_reports.py` removes report payloads and logs after migrating referenced
Python kernel files into `source/`.
