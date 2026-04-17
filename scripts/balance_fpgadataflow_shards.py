#!/usr/bin/env python3
"""Balance the two fpgadataflow shard lists used by the FINN Jenkins pipeline.

Reads a pytest junit XML file produced by the fpgadataflow stage, aggregates
per-file test durations, and greedy-bin-packs files into two roughly equal
shards by total wall-clock time. Emits Groovy list literals that can be
pasted directly over FPGADATAFLOW_SHARD_A and FPGADATAFLOW_SHARD_B at the
top of docker/jenkins/Jenkinsfile.

Usage
-----
    ./scripts/balance_fpgadataflow_shards.py reports/fpgadataflow_a.xml \\
                                             reports/fpgadataflow_b.xml

One or more junit XMLs may be passed (handy for combining shard-A and
shard-B runs from the same build); durations are summed per file across
all inputs.

Rebalance cadence
-----------------
Re-run after any build where the observed wall-clock of one shard exceeds
the other by more than ~15 %. The helper is idempotent; if both shards are
already well balanced the output is identical (modulo ordering) to the
current Jenkinsfile lists.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

TESTS_ROOT = "tests/fpgadataflow/"


@dataclass
class Shard:
    name: str
    total: float = 0.0
    files: list[str] = field(default_factory=list)

    def add(self, path: str, duration: float) -> None:
        self.files.append(path)
        self.total += duration


def aggregate_durations(xml_paths: list[Path]) -> dict[str, float]:
    """Return {test_file_path: total_duration_seconds} across all inputs."""
    durations: dict[str, float] = {}
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        for tc in tree.iter("testcase"):
            classname = tc.attrib.get("classname", "")
            file_attr = tc.attrib.get("file")
            # Prefer the explicit 'file' attribute when present; fall back to
            # inferring from classname (pytest usually emits dotted module
            # paths like "tests.fpgadataflow.test_fpgadataflow_mvau").
            if file_attr:
                rel = file_attr
            elif classname:
                parts = classname.split(".")
                module_parts = []
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


def pack(durations: dict[str, float]) -> tuple[Shard, Shard]:
    """Greedy largest-first bin-pack into two shards."""
    shard_a = Shard("FPGADATAFLOW_SHARD_A")
    shard_b = Shard("FPGADATAFLOW_SHARD_B")
    for path, duration in sorted(
        durations.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        target = shard_a if shard_a.total <= shard_b.total else shard_b
        target.add(path, duration)
    shard_a.files.sort()
    shard_b.files.sort()
    return shard_a, shard_b


def emit_groovy(shard: Shard) -> str:
    lines = [f"@Field", f"List<String> {shard.name} = ["]
    for path in shard.files:
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

    shard_a, shard_b = pack(durations)

    print(f"# Aggregated from {len(args.junit_xml)} junit file(s),")
    print(f"# {len(durations)} fpgadataflow test file(s).")
    print(f"# Shard A wall: {shard_a.total:.1f}s across {len(shard_a.files)} files")
    print(f"# Shard B wall: {shard_b.total:.1f}s across {len(shard_b.files)} files")
    skew = (
        0.0
        if max(shard_a.total, shard_b.total) == 0
        else abs(shard_a.total - shard_b.total)
        / max(shard_a.total, shard_b.total)
        * 100
    )
    print(f"# Skew: {skew:.1f}% (rebalance if >15 %)\n")
    print(emit_groovy(shard_a))
    print()
    print(emit_groovy(shard_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
