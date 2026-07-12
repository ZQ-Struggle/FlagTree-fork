#!/usr/bin/env python3
"""Run every archived FlagGems debugger sample in isolated processes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SAMPLES_ROOT = Path(__file__).resolve().parent
DEBUGGER_ROOT = SAMPLES_ROOT.parent
REPO_ROOT = SAMPLES_ROOT.parents[2]
REGRESSION_TOOL = DEBUGGER_ROOT / "tools" / "flaggems_regression_from_samples.py"
DEFAULT_WORKSPACE_ROOT = (
    REPO_ROOT / ".cache" / "flaggems_debugger_batch" / "samples_regression"
)


def default_flaggems_root() -> Path:
    configured = os.environ.get("FLAGGEMS_ROOT")
    if configured:
        return Path(configured)
    return REPO_ROOT.parent / "FlagGems"


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Replay all entries in samples/index.json and validate execution, "
            "debugger reports, and the historical baseline."
        )
    )
    parser.add_argument(
        "--flaggems-root",
        type=Path,
        default=default_flaggems_root(),
        help="Clean FlagGems source tree (or set FLAGGEMS_ROOT).",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help="Ignored directory used for the instrumented copy and run results.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable with FlagTree, torch, torch_npu, and pytest installed.",
    )
    parser.add_argument(
        "--npu-device",
        type=int,
        default=(
            int(os.environ["FLAGTREE_SAMPLE_NPU"])
            if "FLAGTREE_SAMPLE_NPU" in os.environ
            else None
        ),
        help=(
            "Physical NPU exposed as logical device 0. When omitted, preserve the "
            "caller's Ascend visibility settings."
        ),
    )
    parser.add_argument("--case-total-timeout", type=int, default=480)
    parser.add_argument("--first-report-timeout", type=int, default=300)
    parser.add_argument("--report-timeout", type=int, default=240)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--keep-item-caches",
        action="store_true",
        help="Retain per-item compiler caches instead of pruning rebuildable data.",
    )
    parser.add_argument(
        "--isolated-compiler-caches",
        action="store_true",
        help="Do not reuse the run-local compiler cache across sample processes.",
    )
    parser.add_argument(
        "--stop-on-device-error",
        action="store_true",
        help="Stop after the first device error instead of attempting all samples.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: list[str]) -> int:
    args, forwarded = parse_args(argv)
    flaggems_root = args.flaggems_root.expanduser().resolve()
    if not flaggems_root.is_dir():
        raise SystemExit(
            f"FlagGems source tree not found: {flaggems_root}. "
            "Pass --flaggems-root or set FLAGGEMS_ROOT."
        )
    if not (flaggems_root / "conf" / "operators.yaml").is_file():
        raise SystemExit(f"Not a FlagGems source tree: {flaggems_root}")
    if not REGRESSION_TOOL.is_file():
        raise SystemExit(f"Regression tool not found: {REGRESSION_TOOL}")

    command = [
        str(args.python.expanduser()),
        "-u",
        str(REGRESSION_TOOL),
        "--samples-root",
        str(SAMPLES_ROOT),
        "--sample-index",
        "index.json",
        "--flaggems-root",
        str(flaggems_root),
        "--workspace-root",
        str(args.workspace_root.expanduser().resolve()),
        "--python",
        str(args.python.expanduser()),
        "--case-total-timeout",
        str(args.case_total_timeout),
        "--first-report-timeout",
        str(args.first_report_timeout),
        "--report-timeout",
        str(args.report_timeout),
        "--poll-interval",
        str(args.poll_interval),
        (
            "--stop-on-device-error"
            if args.stop_on_device_error
            else "--no-stop-on-device-error"
        ),
    ]
    if not args.keep_item_caches:
        command.append("--prune-item-caches")
    if not args.isolated_compiler_caches:
        command.append("--shared-compiler-cache")
    if args.start_index is not None:
        command.extend(("--start-index", str(args.start_index)))
    if args.max_items is not None:
        command.extend(("--max-items", str(args.max_items)))
    if args.dry_run:
        command.append("--dry-run")
    command.extend(forwarded)

    env = os.environ.copy()
    env.setdefault("FLAGTREE_BACKEND", "ascend")
    if args.npu_device is not None:
        visible_device = str(args.npu_device)
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible_device
        env["ASCEND_VISIBLE_DEVICES"] = visible_device

    print("Running all debugger samples:", flush=True)
    print(" ".join(command), flush=True)
    return subprocess.call(command, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
