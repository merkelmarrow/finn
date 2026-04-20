#!/usr/bin/env python3
"""Recommend which fpgadataflow test files should carry `fpgadataflow_slow`.

Reads pytest junit XMLs, aggregates per-file durations, greedy-bin-packs into
two halves by wall-clock, and prints the add/remove diff against the current
module-level `pytestmark = [pytest.mark.fpgadataflow_slow]` annotations. The
Jenkinsfile shards by marker (`-m 'fpgadataflow and fpgadataflow_slow'` vs
`-m 'fpgadataflow and not fpgadataflow_slow'`), so the marker annotations ARE
the shard assignment — no Jenkinsfile edit is needed to rebalance.

Usage: ./scripts/balance_fpgadataflow_shards.py reports/fpgadataflow_*.xml
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS_ROOT = "tests/fpgadataflow/"
MARKER_STMT = "pytestmark = [pytest.mark.fpgadataflow_slow]"


def aggregate_durations(xml_paths: list[Path]) -> dict[str, float]:
    """Return {test_file_path: total_duration_seconds} across all inputs."""
    durations: dict[str, float] = {}
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        for tc in tree.iter("testcase"):
            classname = tc.attrib.get("classname", "")
            file_attr = tc.attrib.get("file")
            if file_attr:
                rel = file_attr
            elif classname:
                parts = classname.split(".")
                module_parts: list[str] = []
                for part in parts:
                    module_parts.append(part)
                    if part.startswith("test_"):
                        break
                rel = "/".join(module_parts) + ".py"
            else:
                continue
            if not rel.startswith(TESTS_ROOT):
                continue
            try:
                t = float(tc.attrib.get("time", "0"))
            except ValueError:
                t = 0.0
            durations[rel] = durations.get(rel, 0.0) + t
    return durations


def pack_slow_half(durations: dict[str, float]) -> tuple[set[str], float, float]:
    """Greedy largest-first bin-pack into two halves; return the heavier half
    (the recommended `fpgadataflow_slow` set) plus both walls."""
    slow: list[str] = []
    fast: list[str] = []
    slow_total = 0.0
    fast_total = 0.0
    for path, duration in sorted(durations.items(), key=lambda kv: (-kv[1], kv[0])):
        if slow_total <= fast_total:
            slow.append(path)
            slow_total += duration
        else:
            fast.append(path)
            fast_total += duration
    return set(slow), slow_total, fast_total


def currently_marked(repo_root: Path) -> set[str]:
    """Return files under TESTS_ROOT that already carry the slow marker."""
    marked: set[str] = set()
    for p in (repo_root / TESTS_ROOT).glob("test_*.py"):
        if MARKER_STMT in p.read_text():
            marked.add(str(p.relative_to(repo_root)))
    return marked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "junit_xml",
        nargs="+",
        type=Path,
        help="one or more pytest junit XML files covering the fpgadataflow marker",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="FINN repo root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)

    durations = aggregate_durations(args.junit_xml)
    if not durations:
        print(f"No testcases under {TESTS_ROOT} found in inputs.", file=sys.stderr)
        return 1

    recommended_slow, slow_total, fast_total = pack_slow_half(durations)
    current_slow = currently_marked(args.repo_root)
    to_add = sorted(recommended_slow - current_slow)
    to_remove = sorted(current_slow - recommended_slow)
    skew = (
        0.0
        if max(slow_total, fast_total) == 0
        else abs(slow_total - fast_total) / max(slow_total, fast_total) * 100
    )

    print(f"# Aggregated from {len(args.junit_xml)} junit file(s), "
          f"{len(durations)} fpgadataflow test file(s).")
    print(f"# fpgadataflow_slow shard: {slow_total:.1f}s across {len(recommended_slow)} files")
    print(f"# default shard:            {fast_total:.1f}s across "
          f"{len(durations) - len(recommended_slow)} files")
    print(f"# Skew: {skew:.1f}% (rebalance if >15 %)")

    if not to_add and not to_remove:
        print("\nNo changes required — current markers already match the optimum.")
        return 0

    print("\n# ADD `pytestmark = [pytest.mark.fpgadataflow_slow]` to:")
    for f in to_add:
        print(f"  {f}")
    print("\n# REMOVE the module-level pytestmark from:")
    for f in to_remove:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
