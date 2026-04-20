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
the hook below does deterministic hash-based sharding over the collected
items. No `.test_durations`, no per-file allowlist, no rebalance script.

If `-m <marker>` ever selects zero tests while sharding is requested, the
hook raises — catching "silently skipped in CI" at collection time rather
than allowing the skip to ship.

`@pytest.mark.shard(N)` pins a test to a specific shard, overriding the
hash. `--dry-run-shards` prints the shard assignment table and exits.
A `<stash>.timings.json` sidecar is emitted next to the junit XML when
`--num-shards` is set so the Jenkinsfile can echo per-shard wall-clock.
"""
import hashlib
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
    # name when set, else the nodeid.
    for mark in item.iter_markers(name="xdist_group"):
        if mark.args:
            return str(mark.args[0])
        name = mark.kwargs.get("name")
        if name is not None:
            return str(name)
    return item.nodeid


def _shard_of(key, num_shards):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def _pinned_shard(item, num_shards):
    # `@pytest.mark.shard(N)` overrides the hash-based assignment. An
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


def _assign_shard(item, num_shards):
    pinned = _pinned_shard(item, num_shards)
    if pinned is not None:
        return pinned
    return _shard_of(_group_key(item), num_shards)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # trylast=True ensures pytest's own `-m` deselection has already reduced
    # `items` before we hash-shard it, so each shard sees only tests matching
    # the marker expression.
    num_shards = config.getoption("--num-shards")
    dry_run = config.getoption("--dry-run-shards")
    if num_shards <= 0:
        # --dry-run-shards without --num-shards is a no-op.
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
    if dry_run:
        buckets = [[] for _ in range(num_shards)]
        for it in items:
            buckets[_assign_shard(it, num_shards)].append(it.nodeid)
        print("\n--- dry-run-shards (num_shards=%d, total_items=%d) ---"
              % (num_shards, len(items)))
        print("%-8s %-8s %s" % ("shard", "count", "sample_nodeid"))
        for i, bucket in enumerate(buckets):
            sample = bucket[0] if bucket else "(empty)"
            print("%-8d %-8d %s" % (i, len(bucket), sample))
        # Deselect everything so the session exits cleanly with no tests run.
        config.hook.pytest_deselected(items=list(items))
        items[:] = []
        return
    items[:] = [it for it in items if _assign_shard(it, num_shards) == shard_id]


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
