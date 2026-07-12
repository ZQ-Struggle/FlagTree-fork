#!/usr/bin/env python3
"""Remove archived reports while preserving every available kernel source."""

from __future__ import annotations

import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy_report_sources(sample_dir: Path) -> list[str]:
    source_root = sample_dir / "report_source_locs"
    copied: list[str] = []
    if not source_root.is_dir():
        return copied

    for source in sorted(source_root.rglob("*.py")):
        if source.name.startswith("pointwise_dynamic_"):
            destination = sample_dir / "source" / "generated_kernels" / source.name
        else:
            relative = source.relative_to(source_root)
            destination = (
                sample_dir / "source" / "kernel_sources" / "report_sources" / relative
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(sample_dir)))
    return copied


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def update_markdown(rows: list[dict], uncovered_count: int) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        tier = str(row.get("tier") or "stable")
        counts[tier] = counts.get(tier, 0) + 1
    lines = [
        "# FlagGems Debugger Samples",
        "",
        f"Total samples: {len(rows)}",
        f"Stable samples: {counts.get('stable', 0)}",
        f"Coverage samples: {counts.get('coverage', 0)}",
        f"Inventory ops without a reproducible entry: {uncovered_count}",
        "",
        "Reports are intentionally not archived. All available kernel sources are under `source/`.",
        "",
        "| folder | op | tier | kind | status | drivers | kernel sources |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {folder} | {op} | {tier} | {kind} | {status} | "
            "{driver_sources} | {kernel_sources} |".format(**row)
        )
    (ROOT / "INDEX.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    index_path = ROOT / "index.json"
    rows = read_json(index_path)
    migrated_sources = 0

    for row in rows:
        sample_dir = ROOT / str(row["folder"])
        migrated = copy_report_sources(sample_dir)
        migrated_sources += len(migrated)

        info_path = sample_dir / "run_info.json"
        if info_path.is_file():
            info = read_json(info_path)
            kernel_sources = list(info.get("copied_kernel_sources") or [])
            for source in migrated:
                if source not in kernel_sources:
                    kernel_sources.append(source)
            info["copied_kernel_sources"] = kernel_sources
            info["copied_report_files"] = 0
            info["copied_report_source_locs"] = []
            info["copied_log_files"] = []
            info["reports_archived"] = False
            write_json(info_path, info)
            row["kernel_sources"] = len(kernel_sources)

        remove_tree(sample_dir / "reports")
        remove_tree(sample_dir / "report_source_locs")
        remove_tree(sample_dir / "logs")
        row["reports"] = 0
        row["report_source_locs"] = 0

    write_json(index_path, rows)
    write_json(ROOT / "stable_index.json", [r for r in rows if r["tier"] == "stable"])
    write_json(ROOT / "coverage_index.json", [r for r in rows if r["tier"] == "coverage"])
    uncovered_path = ROOT / "uncovered_ops.json"
    uncovered_count = len(read_json(uncovered_path)) if uncovered_path.exists() else 0
    update_markdown(rows, uncovered_count)

    skipped = ROOT / "skipped_external_source_locs.json"
    if skipped.exists():
        skipped.unlink()

    print(f"samples={len(rows)}")
    print(f"migrated_kernel_sources={migrated_sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
