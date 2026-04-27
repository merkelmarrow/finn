#!/usr/bin/env python3
"""Regenerate ``tests/ci_timings.json`` from ``*.timings.json`` artefacts.

Usage::

    python3 scripts/regen_ci_timings.py [REPORTS_DIR ...]

Each REPORTS_DIR is searched (non-recursively) for ``*.timings.json`` files
emitted by ``tests/conftest.py``. Per-group seconds are summed across all
inputs (multiple shards or multiple runs) and written to
``tests/ci_timings.json`` for the LPT-greedy assignment used in
``_assign_groups_to_shards``.

The ``groups`` array entries are keyed by group name. Under
``--dist=loadgroup`` pytest-xdist suffixes the nodeid with ``@<group>``,
which is stripped here so the key matches what ``_group_key()`` returns
during collection. Tests without an xdist_group are recorded by nodeid.
"""
import glob
import json
import os
import re
import sys


GROUP_SUFFIX_RE = re.compile(r"@(\S+)$")
# Notebook nodeids contain absolute filesystem paths -- collapse to the
# stable basename so timings stay portable across workspaces.
NOTEBOOK_PARAM_RE = re.compile(r"\[(/[^\]]+\.ipynb)\]")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_OUT = os.path.join(REPO_ROOT, "tests", "ci_timings.json")


def canonical_key(name: str) -> str:
    m = GROUP_SUFFIX_RE.search(name)
    if m:
        return m.group(1)
    m2 = NOTEBOOK_PARAM_RE.search(name)
    if m2:
        path = m2.group(1)
        return name.replace(m2.group(0), "[%s]" % os.path.basename(path))
    return name


def main(argv):
    inputs = argv[1:] or [os.path.join(REPO_ROOT, "reports")]
    files = []
    for src in inputs:
        files.extend(sorted(glob.glob(os.path.join(src, "*.timings.json"))))
    if not files:
        print("regen_ci_timings: no *.timings.json files under %r" % inputs,
              file=sys.stderr)
        return 1
    groups = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for entry in data.get("groups", []):
            key = canonical_key(entry["name"])
            groups[key] = groups.get(key, 0.0) + float(entry.get("seconds", 0.0))
    payload = {
        "_comment": "Per-group wall seconds; regenerate with "
                    "scripts/regen_ci_timings.py against latest CI reports.",
        "sources": [os.path.relpath(p, REPO_ROOT) for p in files],
        "groups": {k: round(v, 3) for k, v in sorted(groups.items())},
    }
    with open(DEFAULT_OUT, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print("regen_ci_timings: wrote %d group(s) to %s"
          % (len(groups), os.path.relpath(DEFAULT_OUT, REPO_ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
