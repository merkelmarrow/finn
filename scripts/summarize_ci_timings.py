#!/usr/bin/env python3
"""Summarize per-shard wall-clock times from reports/*.timings.json.

Used by the Jenkinsfile aggregation stage to flag shards that ran for
more than 1.5x the median of their family (stash with trailing _N stripped).
"""
import collections
import glob
import json
import os
import re
import sys

SLOW_FACTOR = 1.5


def family(stash: str) -> str:
    return re.sub(r"_\d+$", "", stash)


def load_rows(reports_dir: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(reports_dir, "*.timings.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"!! could not parse {p}: {e}")
            continue
        stash = d.get("stash") or os.path.basename(p).split(".")[0]
        groups = d.get("groups") or []
        top = groups[0] if groups else {"name": "(none)", "seconds": 0.0}
        rows.append((
            stash,
            int(d.get("shard", {}).get("id", 0)),
            float(d.get("wall_seconds", 0.0)),
            float(top.get("seconds", 0.0)),
            str(top.get("name", "")),
        ))
    return rows


def main(argv):
    reports_dir = argv[1] if len(argv) > 1 else "reports"
    rows = load_rows(reports_dir)
    if not rows:
        print(f"summarize_ci_timings: no parseable timings.json files in {reports_dir}")
        return 0
    by_family = collections.defaultdict(list)
    for r in rows:
        by_family[family(r[0])].append(r)
    print()
    print("=== per-shard wall-clock ===")
    print(f"{'stash':36s} {'id':>3s} {'wall_s':>10s} {'max_group_s':>12s}  max_group")
    print("-" * 100)
    slow_found = False
    for fam in sorted(by_family):
        rs = sorted(by_family[fam], key=lambda r: r[1])
        walls = sorted(r[2] for r in rs)
        median = walls[len(walls) // 2] if walls else 0.0
        for stash, sid, wall, mx_sec, mx_name in rs:
            flag = ""
            if median > 0.0 and wall > SLOW_FACTOR * median:
                flag = f"  <<< SLOW SHARD ({wall / median:.1f}x median)"
                slow_found = True
            print(f"{stash:36s} {sid:>3d} {wall:>10.1f} {mx_sec:>12.1f}  {mx_name}{flag}")
        print()
    if slow_found:
        print(f"summarize_ci_timings: one or more shards exceeded {SLOW_FACTOR}x family "
              "median; consider regenerating tests/ci_timings.json or rebalancing shards")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
