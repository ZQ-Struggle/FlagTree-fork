#!/usr/bin/env python3
"""Archive every reproducible FlagGems debugger test entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SAMPLES = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parents[1] / "tools"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from flaggems_debug_batch import (  # noqa: E402
    inventory_by_id,
    load_operator_inventory,
    op_source_candidates,
)


DEFAULT_DIRECT_SUMMARY = ROOT / (
    ".cache/flaggems_debugger_batch/new_samples_direct_20260711/"
    "direct_runs/20260711_122435/summary.json"
)
DEFAULT_NODE_PLAN = ROOT / (
    ".cache/flaggems_debugger_batch/pytest_no_direct_case_rep_continue/"
    "pytest_node_runs/20260627_100852/collected_nodes.json"
)
PYTEST_SUMMARY_GLOB = (
    ".cache/flaggems_debugger_batch/pytest_no_direct_case_rep*/"
    "pytest_node_runs/*/summary.json"
)
GENERIC_RUNNER = SAMPLES / "040_avg_pool3d" / "source" / "runner.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return normalized or "op"


def local_path(value: str | None) -> Path | None:
    if not value:
        return None
    workspace_prefix = "/workspace/FlagTree/"
    if value.startswith(workspace_prefix):
        return ROOT / value[len(workspace_prefix) :]
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def copy_if_file(source: Path | None, destination: Path) -> bool:
    try:
        is_file = source is not None and source.is_file()
    except OSError:
        is_file = False
    if not is_file:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_reports(source_dir: Path | None, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if source_dir is None or not source_dir.is_dir():
        return copied
    for source in sorted(source_dir.iterdir()):
        if source.is_file() and source.suffix in {".txt", ".json"}:
            shutil.copy2(source, destination / source.name)
            copied.append(f"reports/{source.name}")
    return copied


def kernel_source_paths(flaggems_root: Path, op: str) -> list[Path]:
    roots = [
        flaggems_root / "src" / "flag_gems" / "ops",
        flaggems_root / "src" / "flag_gems" / "experimental_ops",
        flaggems_root / "src" / "flag_gems" / "fused",
        flaggems_root / "src" / "flag_gems" / "runtime" / "backend" / "_ascend" / "ops",
        flaggems_root / "src" / "flag_gems",
    ]
    result: list[Path] = []
    for source_root in roots:
        for candidate in op_source_candidates(op):
            path = source_root / f"{candidate}.py"
            if path.is_file() and path not in result:
                result.append(path)
    return result


def copy_kernel_sources(flaggems_root: Path, op: str, sample_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in kernel_source_paths(flaggems_root, op):
        relative = source.relative_to(flaggems_root)
        destination = sample_dir / "source" / "kernel_sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(sample_dir)))
    return copied


def source_paths_from_reports(sample_dir: Path, flaggems_root: Path) -> list[Path]:
    result: list[Path] = []

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    for report in sorted((sample_dir / "reports").glob("*.json")):
        try:
            document = read_json(report)
        except (OSError, ValueError):
            continue
        for value in strings(document):
            for raw_path in re.findall(r'(/[^"\n]+\.py)', value):
                source = local_path(raw_path)
                if source is None and raw_path.startswith(str(flaggems_root) + "/"):
                    source = Path(raw_path)
                try:
                    is_file = source is not None and source.is_file()
                except OSError:
                    is_file = False
                if is_file and source not in result:
                    result.append(source)
    return result


def copy_report_source_locs(
    sample_dir: Path, flaggems_root: Path
) -> list[str]:
    copied: list[str] = []
    for source in source_paths_from_reports(sample_dir, flaggems_root):
        try:
            relative = source.relative_to(ROOT)
        except ValueError:
            try:
                relative = Path("FlagGems") / source.relative_to(flaggems_root)
            except ValueError:
                continue
        destination = sample_dir / "report_source_locs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(sample_dir)))
    return copied


def copy_generated_sources(sample_dir: Path, info: dict[str, Any]) -> list[str]:
    script = local_path(info.get("script"))
    if script is None:
        return []
    cache_dir = script.parent / "flaggems_cache" / "code_cache"
    copied: list[str] = []
    if not cache_dir.is_dir():
        return copied
    for source in sorted(cache_dir.glob("*.py")):
        destination = sample_dir / "source" / "generated_kernels" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(sample_dir)))
    return copied


def copy_logs(row: dict[str, Any], sample_dir: Path) -> list[str]:
    copied: list[str] = []
    for key, name in (("stdout_log", "stdout.log"), ("stderr_log", "stderr.log")):
        if copy_if_file(local_path(row.get(key)), sample_dir / "logs" / name):
            copied.append(f"logs/{name}")
    return copied


def result_rank(result: dict[str, Any] | None) -> tuple[int, int, int, float]:
    if result is None:
        return (0, 0, 0, 0.0)
    complete = min(int(result.get("debug_txt_count") or 0), int(result.get("debug_json_count") or 0))
    status_rank = {
        "passed": 5,
        "partial_timeout": 4,
        "missing_debug_report": 3,
        "pytest_error": 2,
        "timeout": 1,
    }.get(str(result.get("status")), 0)
    return (1, int(complete > 0), status_rank, -float(result.get("duration_sec") or 0.0))


def load_pytest_results() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_node: dict[str, dict[str, Any]] = {}
    source_summary: dict[str, str] = {}
    for summary_path in sorted(ROOT.glob(PYTEST_SUMMARY_GLOB)):
        for row in read_json(summary_path):
            nodeid = str(row.get("nodeid") or "")
            if not nodeid:
                continue
            current = by_node.get(nodeid)
            if result_rank(row) > result_rank(current):
                by_node[nodeid] = row
                source_summary[nodeid] = str(summary_path.relative_to(ROOT))
    return by_node, source_summary


def infer_kind(sample_dir: Path) -> str:
    source = sample_dir / "source"
    if (source / "case.py").exists():
        return "direct_case"
    if (source / "runner.py").exists():
        return "pytest_node"
    return "legacy_marker"


def enrich_coverage_sources(
    rows: list[dict[str, Any]], flaggems_root: Path
) -> None:
    for row in rows:
        if row.get("tier") != "coverage":
            continue
        sample_dir = SAMPLES / str(row["folder"])
        info_path = sample_dir / "run_info.json"
        if not info_path.is_file():
            continue
        info = read_json(info_path)
        kernel_sources = list(info.get("copied_kernel_sources") or [])
        for source in copy_generated_sources(sample_dir, info):
            if source not in kernel_sources:
                kernel_sources.append(source)
        report_sources = copy_report_source_locs(sample_dir, flaggems_root)
        info["copied_kernel_sources"] = kernel_sources
        info["copied_report_source_locs"] = report_sources
        write_json(info_path, info)
        row["kernel_sources"] = len(kernel_sources)
        row["report_source_locs"] = len(report_sources)


def enrich_existing(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        sample_dir = SAMPLES / row["folder"]
        info_path = sample_dir / "run_info.json"
        info = read_json(info_path) if info_path.exists() else {}
        row.setdefault("tier", "stable")
        row.setdefault("kind", infer_kind(sample_dir))
        row.setdefault("status", str(info.get("status") or "passed"))
        row.setdefault("kernel_sources", 0)


def next_index(rows: list[dict[str, Any]]) -> int:
    indices = []
    for row in rows:
        match = re.match(r"(\d+)_", str(row.get("folder", "")))
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def archive_direct(
    row: dict[str, Any],
    index: int,
    flaggems_root: Path,
    summary_path: Path,
    include_run_artifacts: bool,
) -> dict[str, Any]:
    op = str(row["op"])
    folder = f"{index:03d}_{safe_name(op)}"
    sample_dir = SAMPLES / folder
    source_dir = sample_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=False)

    driver_sources: list[str] = []
    if copy_if_file(local_path(row.get("script")), source_dir / "case.py"):
        driver_sources.append("source/case.py")
    kernel_sources = copy_kernel_sources(flaggems_root, op, sample_dir)
    reports = (
        copy_reports(local_path(row.get("debug_report_dir")), sample_dir / "reports")
        if include_run_artifacts
        else []
    )
    logs = copy_logs(row, sample_dir) if include_run_artifacts else []

    run_info = dict(row)
    run_info.update(
        {
            "tier": "coverage",
            "kind": "direct_case",
            "copied_driver_sources": driver_sources,
            "copied_kernel_sources": kernel_sources,
            "copied_report_files": len(reports),
            "copied_log_files": logs,
            "summary_file": str(summary_path.relative_to(ROOT)),
        }
    )
    write_json(sample_dir / "run_info.json", run_info)
    return {
        "folder": folder,
        "op": op,
        "tier": "coverage",
        "kind": "direct_case",
        "status": str(row.get("status") or "collected"),
        "reports": len(reports),
        "driver_sources": len(driver_sources),
        "kernel_sources": len(kernel_sources),
        "report_source_locs": 0,
        "summary_file": str(summary_path.relative_to(ROOT)),
    }


def archive_pytest(
    op: str,
    node: dict[str, Any],
    result: dict[str, Any] | None,
    result_summary: str | None,
    index: int,
    flaggems_root: Path,
    include_run_artifacts: bool,
) -> dict[str, Any]:
    folder = f"{index:03d}_{safe_name(op)}"
    sample_dir = SAMPLES / folder
    source_dir = sample_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=False)

    driver_sources: list[str] = []
    if copy_if_file(GENERIC_RUNNER, source_dir / "runner.py"):
        driver_sources.append("source/runner.py")
    test_source = flaggems_root / str(node["file"])
    if copy_if_file(test_source, source_dir / "test_source.py"):
        driver_sources.append("source/test_source.py")
    kernel_sources = copy_kernel_sources(flaggems_root, op, sample_dir)
    reports = (
        copy_reports(
            local_path(result.get("debug_report_dir") if result else None),
            sample_dir / "reports",
        )
        if include_run_artifacts
        else []
    )
    logs = copy_logs(result or {}, sample_dir) if include_run_artifacts else []
    status = str(result.get("status") if result else "collected_unrun")
    summary_file = result_summary or str(DEFAULT_NODE_PLAN.relative_to(ROOT))

    run_info = {
        "op": op,
        "tier": "coverage",
        "kind": "pytest_node",
        "status": status,
        "nodeid": node["nodeid"],
        "selected_ops": node.get("selected_ops", [op]),
        "timeout_class": node.get("timeout_class"),
        "first_report_timeout_sec": node.get("first_report_timeout_sec"),
        "report_timeout_sec": node.get("report_timeout_sec"),
        "node_total_timeout_sec": node.get("node_total_timeout_sec"),
        "copied_driver_sources": driver_sources,
        "copied_kernel_sources": kernel_sources,
        "copied_report_files": len(reports),
        "copied_log_files": logs,
        "summary_file": summary_file,
    }
    if result:
        run_info.update(result)
        run_info["tier"] = "coverage"
        run_info["kind"] = "pytest_node"
        run_info["copied_driver_sources"] = driver_sources
        run_info["copied_kernel_sources"] = kernel_sources
        run_info["copied_report_files"] = len(reports)
        run_info["copied_log_files"] = logs
        run_info["summary_file"] = summary_file
    write_json(sample_dir / "run_info.json", run_info)
    return {
        "folder": folder,
        "op": op,
        "tier": "coverage",
        "kind": "pytest_node",
        "status": status,
        "reports": len(reports),
        "driver_sources": len(driver_sources),
        "kernel_sources": len(kernel_sources),
        "report_source_locs": 0,
        "summary_file": summary_file,
    }


def write_markdown(rows: list[dict[str, Any]], uncovered: list[str]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["tier"]] = counts.get(row["tier"], 0) + 1
    lines = [
        "# FlagGems Debugger Samples",
        "",
        f"Total samples: {len(rows)}",
        f"Stable samples: {counts.get('stable', 0)}",
        f"Coverage samples: {counts.get('coverage', 0)}",
        f"Inventory ops without a reproducible entry: {len(uncovered)}",
        "",
        "| folder | op | tier | kind | status | reports | drivers | kernel sources |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {folder} | {op} | {tier} | {kind} | {status} | {reports} | "
            "{driver_sources} | {kernel_sources} |".format(**row)
        )
    (SAMPLES / "INDEX.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flaggems-root", type=Path, default=Path("/home/quan/FlagGems"))
    parser.add_argument("--direct-summary", type=Path, default=DEFAULT_DIRECT_SUMMARY)
    parser.add_argument("--node-plan", type=Path, default=DEFAULT_NODE_PLAN)
    parser.add_argument(
        "--include-run-artifacts",
        action="store_true",
        help="Also archive reports and logs. Source-only archival is the default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    flaggems_root = args.flaggems_root.resolve()
    direct_summary = args.direct_summary.resolve()
    node_plan = args.node_plan.resolve()

    rows = read_json(SAMPLES / "index.json")
    enrich_existing(rows)
    existing_ops = {str(row["op"]) for row in rows}
    index = next_index(rows)

    inventory = load_operator_inventory(flaggems_root)
    inventory_map = inventory_by_id(inventory)
    inventory_ops: list[str] = []
    for item in inventory:
        op = str(item.get("id") or "").strip().lstrip("_")
        if op and op not in inventory_ops:
            inventory_ops.append(op)

    direct_rows = [
        row for row in read_json(direct_summary) if str(row.get("op")) not in existing_ops
    ]
    direct_ops = {str(row["op"]) for row in direct_rows}
    for row in direct_rows:
        archived = archive_direct(
            row,
            index,
            flaggems_root,
            direct_summary,
            args.include_run_artifacts,
        )
        rows.append(archived)
        existing_ops.add(archived["op"])
        index += 1

    pytest_results, result_summaries = load_pytest_results()
    nodes = read_json(node_plan)
    nodes_by_op: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        for op in node.get("selected_ops", []):
            normalized = str(op).lstrip("_")
            if normalized in inventory_map or normalized in inventory_ops:
                nodes_by_op.setdefault(normalized, []).append(node)

    for op in inventory_ops:
        if op in existing_ops or op in direct_ops:
            continue
        candidates = nodes_by_op.get(op, [])
        if not candidates:
            continue
        node = max(candidates, key=lambda item: result_rank(pytest_results.get(item["nodeid"])))
        result = pytest_results.get(node["nodeid"])
        archived = archive_pytest(
            op,
            node,
            result,
            result_summaries.get(node["nodeid"]),
            index,
            flaggems_root,
            args.include_run_artifacts,
        )
        rows.append(archived)
        existing_ops.add(op)
        index += 1

    uncovered = [op for op in inventory_ops if op not in existing_ops]
    uncovered_rows = []
    for op in uncovered:
        sources = [str(path.relative_to(flaggems_root)) for path in kernel_source_paths(flaggems_root, op)]
        uncovered_rows.append(
            {
                "op": op,
                "reason": "no direct case and no matching pytest node",
                "kernel_sources": sources,
            }
        )

    enrich_coverage_sources(rows, flaggems_root)
    rows.sort(key=lambda row: int(str(row["folder"]).split("_", 1)[0]))
    write_json(SAMPLES / "index.json", rows)
    write_json(SAMPLES / "stable_index.json", [row for row in rows if row["tier"] == "stable"])
    write_json(SAMPLES / "coverage_index.json", [row for row in rows if row["tier"] == "coverage"])
    write_json(SAMPLES / "uncovered_ops.json", uncovered_rows)
    write_markdown(rows, uncovered)

    print(f"stable={sum(row['tier'] == 'stable' for row in rows)}")
    print(f"coverage={sum(row['tier'] == 'coverage' for row in rows)}")
    print(f"total={len(rows)}")
    print(f"uncovered={len(uncovered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
