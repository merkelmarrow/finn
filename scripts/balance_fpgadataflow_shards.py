#!/usr/bin/env python3
"""Recompute the explicit fpgadataflow shard A allowlist used by the FINN
Jenkins pipeline.

Shard B is a catch-all in the Jenkinsfile (everything under tests/fpgadataflow/
that is NOT in shard A), so we only need to decide which test files belong on
shard A — typically the slowest half by total wall-clock — to keep both shards
roughly balanced.

Reads pytest junit XMLs from previous fpgadataflow runs, aggregates per-file
durations, greedy-bin-packs into two halves by total wall, and prints the
larger half as a Groovy list literal that can be pasted over
FPGADATAFLOW_SHARD_A at the top of docker/jenkins/Jenkinsfile.

Usage
-----
    ./scripts/balance_fpgadataflow_shards.py reports/fpgadataflow_a.xml \\
                                             reports/fpgadataflow_b.xml

Re-run after any build where one shard's wall exceeds the other by >15 %.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS_ROOT = "tests/fpgadataflow/"


def aggregate_durations(xml_paths: list[Path]) -> dict[str, float]:
    """Return {test_file_path: total_duration_seconds} across all inputs."""
    durations: dict[str, float] = {}
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        for tc in tree.iter("testcase"):
            classname = tc.attrib.get("classname", "")
            file_attr = tc.attrib.get("file")
            # Prefer 'file' attr when present; fall back to dotted classname.
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


def pack_shard_a(durations: dict[str, float]) -> tuple[list[str], float, float]:
    """Greedy largest-first bin-pack into two halves; return the heavier half
    plus its wall and the lighter half's wall."""
    a_files: list[str] = []
    b_files: list[str] = []
    a_total = 0.0
    b_total = 0.0
    for path, duration in sorted(
        durations.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        if a_total <= b_total:
            a_files.append(path)
            a_total += duration
        else:
            b_files.append(path)
            b_total += duration
    a_files.sort()
    return a_files, a_total, b_total


def emit_groovy(files: list[str]) -> str:
    lines = ["@Field", "List<String> FPGADATAFLOW_SHARD_A = ["]
    for path in files:
        lines.append(f"    '{path}',")
    lines.append("]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "junit_xml",
        nargs="+",
        type=Path,
        help="one or more pytest junit XML files covering the fpgadataflow marker",
    )
    args = parser.parse_args(argv)

    durations = aggregate_durations(args.junit_xml)
    if not durations:
        print(
            f"No testcases under {TESTS_ROOT} found in inputs; check paths.",
            file=sys.stderr,
        )
        return 1

    shard_a_files, a_total, b_total = pack_shard_a(durations)
    skew = (
        0.0
        if max(a_total, b_total) == 0
        else abs(a_total - b_total) / max(a_total, b_total) * 100
    )

    print(f"# Aggregated from {len(args.junit_xml)} junit file(s),")
    print(f"# {len(durations)} fpgadataflow test file(s).")
    print(f"# Shard A (explicit allowlist below): {a_total:.1f}s across {len(shard_a_files)} files")
    print(f"# Shard B (catch-all, implicit):      {b_total:.1f}s across {len(durations) - len(shard_a_files)} files")
    print(f"# Skew: {skew:.1f}% (rebalance if >15 %)\n")
    print(emit_groovy(shard_a_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
