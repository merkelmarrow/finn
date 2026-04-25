# Copyright (c) 2020, Xilinx
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of FINN nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Sharding and timing-observability plugin for FINN CI.

Each Jenkins stage runs ``pytest -m <marker> --num-shards N --shard-id I``;
this module deterministically assigns groups to shards using checked-in
per-group seconds from ``tests/ci_timings.json`` (LPT-greedy bin packing,
falls back to round-robin when the file is absent or has no signal).

Items sharing an ``@pytest.mark.xdist_group(name=X)`` always land in the
same shard so chained ``load_test_checkpoint_or_skip`` steps don't break.
Items without an explicit group are grouped by nodeid.

``@pytest.mark.shard(N)`` pins a group to shard N. ``--dry-run-shards``
prints the shard table and exits. ``--which-shard QUERY`` prints which
Jenkins shard a given nodeid would run on. A ``<stash>.timings.json``
sidecar is emitted next to the junit XML so aggregation can flag outliers
and so timings can be regenerated (see ``scripts/regen_ci_timings.py``).
"""
import json
import os
import time

import pytest


SHARD_MARKER_NAME = "shard"
TIMINGS_FILE = os.path.join(os.path.dirname(__file__), "ci_timings.json")


def pytest_addoption(parser):
    group = parser.getgroup("finn-ci-sharding")
    group.addoption("--num-shards", type=int, default=0,
                    help="Split the collected test set into N deterministic shards.")
    group.addoption("--shard-id", type=int, default=0,
                    help="Which shard (0-indexed) to run. Requires --num-shards.")
    group.addoption("--dry-run-shards", action="store_true", default=False,
                    help="Print shard assignment table and exit without running tests.")
    group.addoption("--which-shard", default=None, metavar="QUERY",
                    help="Print which Jenkins shard each ci_shards.STAGES row "
                         "would run a matching test on; use with --collect-only.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "shard(N): pin this test to shard N (use sparingly for flaky-test isolation).",
    )


def _group_key(item):
    """Return the xdist_group name for ``item``, or its nodeid as a singleton."""
    for mark in item.iter_markers(name="xdist_group"):
        if mark.args:
            return str(mark.args[0])
        name = mark.kwargs.get("name")
        if name is not None:
            return str(name)
    return item.nodeid


def _pinned_shard(item, num_shards):
    for mark in item.iter_markers(name=SHARD_MARKER_NAME):
        if not mark.args:
            continue
        pinned = mark.args[0]
        if not isinstance(pinned, int):
            raise pytest.UsageError(
                "@pytest.mark.shard expects an int arg; got %r on %s"
                % (pinned, item.nodeid)
            )
        if not 0 <= pinned < num_shards:
            raise pytest.UsageError(
                "@pytest.mark.shard(%d) out of range for --num-shards=%d on %s"
                % (pinned, num_shards, item.nodeid)
            )
        return pinned
    return None


def _load_group_weights():
    """Return ``{group_name: seconds}`` from TIMINGS_FILE, or {} if absent."""
    try:
        with open(TIMINGS_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    weights = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(weights, dict):
        return {}
    return {str(k): float(v) for k, v in weights.items() if v}


def _assign_groups_to_shards(groups, num_shards):
    """Map ``{group_key: [items]}`` to ``{group_key: shard_id}``.

    Honours @pytest.mark.shard(N) pins, then LPT-greedy assigns the
    remaining groups by descending weight from ``ci_timings.json`` (groups
    without a recorded weight get the median, falling back to 1.0).
    Ties are broken by sorted group key for reproducibility.
    """
    assignment = {}
    for key, members in groups.items():
        pins = {_pinned_shard(it, num_shards) for it in members}
        pins.discard(None)
        if len(pins) > 1:
            raise pytest.UsageError(
                "conflicting @pytest.mark.shard pins within xdist_group %r: %r"
                % (key, sorted(pins))
            )
        if pins:
            assignment[key] = pins.pop()
    if num_shards <= 1:
        for key in groups:
            assignment.setdefault(key, 0)
        return assignment
    weights_table = _load_group_weights()
    known = [w for w in weights_table.values() if w > 0]
    fallback = sorted(known)[len(known) // 2] if known else 1.0
    unpinned = [k for k in groups if k not in assignment]
    unpinned.sort(key=lambda k: (-weights_table.get(k, fallback), k))
    shard_load = [0.0] * num_shards
    for key in unpinned:
        weight = weights_table.get(key, fallback)
        target = min(range(num_shards), key=lambda s: (shard_load[s], s))
        assignment[key] = target
        shard_load[target] += weight
    return assignment


def _marker_tokens(marker_expr):
    """Split a ci_shards marker expression into individual marker names."""
    return [t for t in marker_expr.split() if t != "or"]


def _item_matches_marker_expr(item, marker_expr):
    tokens = _marker_tokens(marker_expr)
    return any(any(True for _ in item.iter_markers(name=t)) for t in tokens)


def _run_which_shard(config, items, query):
    from ci_shards import STAGES, stash_name
    rows = []
    for stage in STAGES:
        matching = [it for it in items
                    if _item_matches_marker_expr(it, stage["marker"])]
        if not matching:
            continue
        groups = {}
        for it in matching:
            groups.setdefault(_group_key(it), []).append(it)
        assignment = _assign_groups_to_shards(groups, int(stage["shards"]))
        for it in matching:
            if query not in it.nodeid:
                continue
            shard_id = assignment[_group_key(it)]
            rows.append((
                stage["stage"], stage["marker"], int(stage["shards"]),
                shard_id, stash_name(stage["stage"], int(stage["shards"]), shard_id),
                it.nodeid,
            ))
    if not rows:
        print("[finn-sharding] --which-shard: no test matched %r in any stage" % query)
    else:
        header = ("stage", "marker", "shards", "shard", "stash", "nodeid")
        widths = [max(len(str(c)) for c in col)
                  for col in zip(header, *rows)]
        fmt = "  ".join("%%-%ds" % w for w in widths)
        print(fmt % header)
        print(fmt % tuple("-" * w for w in widths))
        for r in rows:
            print(fmt % tuple(str(c) for c in r))
    config.hook.pytest_deselected(items=list(items))
    items[:] = []


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # trylast lets pytest's own -m deselection run before we shard.
    which = config.getoption("--which-shard")
    if which:
        _run_which_shard(config, items, which)
        return
    num_shards = config.getoption("--num-shards")
    dry_run = config.getoption("--dry-run-shards")
    if num_shards <= 0:
        return
    shard_id = config.getoption("--shard-id")
    if not 0 <= shard_id < num_shards:
        raise pytest.UsageError(
            "--shard-id=%d out of range for --num-shards=%d" % (shard_id, num_shards)
        )
    if not items:
        # An empty selection under sharding is a silent-skip footgun (stale
        # marker, typo). Fail loudly instead.
        raise pytest.UsageError(
            "no tests collected for this marker; CI shard configuration is "
            "out of sync with the test markers"
        )

    groups = {}
    for item in items:
        groups.setdefault(_group_key(item), []).append(item)
    assignment = _assign_groups_to_shards(groups, num_shards)

    if dry_run:
        weights_table = _load_group_weights()
        known = [w for w in weights_table.values() if w > 0]
        fallback = sorted(known)[len(known) // 2] if known else 1.0
        per_shard = {i: {"items": 0, "groups": [], "seconds": 0.0}
                     for i in range(num_shards)}
        for key, members in groups.items():
            s = assignment[key]
            per_shard[s]["items"] += len(members)
            per_shard[s]["groups"].append(key)
            per_shard[s]["seconds"] += weights_table.get(key, fallback)
        print("\n--- dry-run-shards (num_shards=%d, groups=%d, items=%d) ---"
              % (num_shards, len(groups), len(items)))
        print("%-8s %-8s %-8s %-10s %s"
              % ("shard", "items", "groups", "weight_s", "sample_group"))
        for i in range(num_shards):
            gs = sorted(per_shard[i]["groups"])
            sample = gs[0] if gs else "(empty)"
            print("%-8d %-8d %-8d %-10.1f %s"
                  % (i, per_shard[i]["items"], len(gs),
                     per_shard[i]["seconds"], sample))
        config.hook.pytest_deselected(items=list(items))
        items[:] = []
        return

    kept = [it for it in items if assignment[_group_key(it)] == shard_id]
    my_groups = sorted(k for k, s in assignment.items() if s == shard_id)
    print("[finn-sharding] shard %d/%d: %d item(s) across %d group(s)"
          % (shard_id, num_shards, len(kept), len(my_groups)))
    if my_groups:
        sample = my_groups[:5]
        ellipsis = " ..." if len(my_groups) > 5 else ""
        print("[finn-sharding] groups: %s%s" % (sample, ellipsis))
    items[:] = kept


# Per-shard timing observability: emit <stash>.timings.json next to the
# junit XML when --num-shards is set, so aggregation can flag outliers
# and operators can regenerate ci_timings.json from a recent run.

# Set on the controller in pytest_sessionstart; left None on xdist workers
# so log-reports don't double-count.
_TIMINGS = None


def _is_xdist_worker():
    return bool(os.environ.get("PYTEST_XDIST_WORKER"))


def pytest_sessionstart(session):
    global _TIMINGS
    _TIMINGS = None
    if session.config.getoption("--num-shards") <= 0:
        return
    if _is_xdist_worker():
        return
    _TIMINGS = {
        "per_test_seconds": {},
        "nodeid_to_group": {},
        "start_monotonic": time.monotonic(),
    }


def pytest_collection_finish(session):
    if _TIMINGS is None:
        return
    for item in session.items:
        for mark in item.iter_markers(name="xdist_group"):
            if mark.args:
                _TIMINGS["nodeid_to_group"][item.nodeid] = str(mark.args[0])
                break
            name = mark.kwargs.get("name")
            if name is not None:
                _TIMINGS["nodeid_to_group"][item.nodeid] = str(name)
                break


def pytest_runtest_logreport(report):
    if _TIMINGS is None:
        return
    # Accumulate setup+call+teardown per nodeid; xdist forwards worker
    # reports here so summing covers the full session.
    duration = float(getattr(report, "duration", 0.0) or 0.0)
    _TIMINGS["per_test_seconds"][report.nodeid] = (
        _TIMINGS["per_test_seconds"].get(report.nodeid, 0.0) + duration
    )


def pytest_sessionfinish(session, exitstatus):
    if _TIMINGS is None:
        return
    junitxml = session.config.getoption("--junitxml") or ""
    if not junitxml:
        return
    out_dir = os.path.dirname(junitxml) or "."
    stash = os.path.splitext(os.path.basename(junitxml))[0]
    out_path = os.path.join(out_dir, stash + ".timings.json")
    groups = {}
    for nodeid, duration in _TIMINGS["per_test_seconds"].items():
        group = _TIMINGS["nodeid_to_group"].get(nodeid, nodeid)
        entry = groups.setdefault(group, {"name": group, "count": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] += duration
    payload = {
        "stash": stash,
        "shard": {
            "num": session.config.getoption("--num-shards"),
            "id": session.config.getoption("--shard-id"),
        },
        "wall_seconds": time.monotonic() - _TIMINGS["start_monotonic"],
        "total_test_seconds": sum(_TIMINGS["per_test_seconds"].values()),
        "groups": sorted(groups.values(), key=lambda g: -g["seconds"]),
    }
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass
