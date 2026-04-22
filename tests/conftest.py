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

"""Sharding, coverage-guard and timing-observability plugin for FINN CI.

Adding a test requires nothing beyond the usual `@pytest.mark.<marker>`:
the Jenkinsfile runs each marker with `--num-shards N --shard-id I`, and
the hook below does deterministic round-robin-by-group sharding over the
collected items. No `.test_durations`, no per-file allowlist, no
rebalance script.

iter-6 change: switched from hash-by-group to round-robin over
sorted-group-keys. Hashing permitted collisions where two big groups
landed on one shard and another shard ran empty; the pathological case
in #18 was `bnn_u250_3: 54 skipped, 0 passed in 11.9 s` where the shard
received only chain-break-prone late steps from guard-skipped scenarios.
Round-robin guarantees balanced group COUNT per shard and deterministic
placement that is identical across runs of the same test set, so a late
step's upstream is always in the same shard it was in last run.

Chain integrity rule: items that share an `@pytest.mark.xdist_group(name=X)`
ALWAYS land in the same shard (they form a `load_test_checkpoint_or_skip`
sequence and splitting them silently skips downstream steps). Items
without an explicit xdist_group are grouped by nodeid (one item per
group), which is fine for leaf unit tests.

If `-m <marker>` ever selects zero tests while sharding is requested, the
hook raises — catching "silently skipped in CI" at collection time rather
than allowing the skip to ship.

`@pytest.mark.shard(N)` pins a test (and its xdist_group siblings) to a
specific shard, overriding round-robin. `--dry-run-shards` prints the
shard assignment table and exits. A `<stash>.timings.json` sidecar is
emitted next to the junit XML when `--num-shards` is set so the
Jenkinsfile can echo per-shard wall-clock.
"""
import json
import os
import time

import pytest


SHARD_MARKER_NAME = "shard"


def pytest_addoption(parser):
    group = parser.getgroup("finn-ci-sharding")
    group.addoption("--num-shards", type=int, default=0,
                    help="Split the collected test set into N deterministic shards.")
    group.addoption("--shard-id", type=int, default=0,
                    help="Which shard (0-indexed) to run. Requires --num-shards.")
    group.addoption("--dry-run-shards", action="store_true", default=False,
                    help="Print shard assignment table and exit without running tests.")
    group.addoption("--which-shard", default=None, metavar="QUERY",
                    help="For each CI stage in tests/ci_shards.py whose marker "
                         "covers a collected test whose nodeid contains QUERY, "
                         "print stage | marker | shards | shard | stash, then "
                         "exit. Use with --collect-only and no -m. Answers "
                         "'which Jenkins shard runs this test?' in one step.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "shard(N): pin this test to shard N, overriding hash-based assignment "
        "(use for flaky-test isolation).",
    )


def _group_key(item):
    # Tests sharing an `xdist_group` form a sequence (downstream steps use
    # `load_test_checkpoint_or_skip` against upstream outputs), so they must
    # land in the SAME shard or the chain silently skips. Return the group
    # name when set, else the nodeid (each leaf item its own "group of 1").
    for mark in item.iter_markers(name="xdist_group"):
        if mark.args:
            return str(mark.args[0])
        name = mark.kwargs.get("name")
        if name is not None:
            return str(name)
    return item.nodeid


def _pinned_shard(item, num_shards):
    # `@pytest.mark.shard(N)` overrides round-robin placement. An
    # out-of-range pin is a loud UsageError rather than a silent skip.
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


def _assign_groups_to_shards(groups, num_shards):
    # groups: dict of group_key -> list[items]. Return dict of group_key -> shard.
    #
    # Two-phase: first honour any @pytest.mark.shard(N) pins (erroring if a
    # single xdist_group has conflicting pins across its members); then
    # round-robin the unpinned groups over shards in SORTED key order. Sorted
    # order is intentional: the assignment becomes deterministic and
    # reproducible across runs of the same test set, so a downstream step
    # always lands in the same shard its upstream did last time.
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
    unpinned = sorted(k for k in groups if k not in assignment)
    for i, key in enumerate(unpinned):
        assignment[key] = i % num_shards
    return assignment


def _marker_tokens(marker_expr):
    # ci_shards.STAGES rows have marker expressions constrained to
    # MARKER_SAFE_PATTERN (`^[A-Za-z0-9_ ]+$` -- see Jenkinsfile), so tokens
    # are whitespace-split and the only connective allowed is literal `or`.
    # Any token that is not `or` is a marker name.
    return [t for t in marker_expr.split() if t != "or"]


def _item_matches_marker_expr(item, marker_expr):
    # Item matches the stage's marker expression iff any token names a mark
    # attached to the item. Mirrors how pytest's -m resolves the same
    # expression under our regex constraint, without using private API.
    tokens = _marker_tokens(marker_expr)
    return any(any(True for _ in item.iter_markers(name=t)) for t in tokens)


def _run_which_shard(config, items, query):
    # Re-uses _assign_groups_to_shards() verbatim per matching stage so the
    # table shown here is the exact assignment the Jenkins shard will use.
    from ci_shards import STAGES, stash_name  # local, lets pytest start without it
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
    # trylast=True ensures pytest's own `-m` deselection has already reduced
    # `items` before we shard it, so each shard sees only tests matching
    # the marker expression.
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
        # Zero items before sharding means the `-m` expression selected nothing.
        # In CI that is a silent-skip footgun (marker typo, stale PARALLEL_SHARDS
        # row, etc.), so we fail loudly instead.
        raise pytest.UsageError(
            "no tests collected for this marker; CI shard configuration is "
            "out of sync with the test markers"
        )

    groups = {}
    for item in items:
        groups.setdefault(_group_key(item), []).append(item)
    assignment = _assign_groups_to_shards(groups, num_shards)

    if dry_run:
        per_shard = {i: {"items": 0, "groups": []} for i in range(num_shards)}
        for key, members in groups.items():
            s = assignment[key]
            per_shard[s]["items"] += len(members)
            per_shard[s]["groups"].append(key)
        print("\n--- dry-run-shards (num_shards=%d, groups=%d, items=%d) ---"
              % (num_shards, len(groups), len(items)))
        print("%-8s %-8s %-8s %s" % ("shard", "items", "groups", "sample_group"))
        for i in range(num_shards):
            gs = sorted(per_shard[i]["groups"])
            sample = gs[0] if gs else "(empty)"
            print("%-8d %-8d %-8d %s" % (i, per_shard[i]["items"], len(gs), sample))
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


# ---------------------------------------------------------------------------
# Per-shard timing observability (WS 1 of iter-5)
#
# When --num-shards is set, emit `<stash>.timings.json` next to the junit XML
# so the Jenkinsfile can echo per-marker shard-wise wall-clock and warn about
# outliers. Group-level times are keyed by xdist_group so the aggregation
# matches the hash-sharding key.
# ---------------------------------------------------------------------------

# Module-level accumulator. Set in pytest_sessionstart on the controller; left
# as None on xdist workers (where pytest_runtest_logreport would double-count
# against the controller's copy of the same report).
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
    # Accumulate setup+call+teardown durations per nodeid. Under xdist the
    # controller receives each worker's report here with the worker's timing.
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
        # Timing I/O failure must never fail the session.
        pass
