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

"""Pytest plugin for Jenkins shard selection and timing observability.

Each Jenkins stage runs ``pytest -m <marker> --num-shards N --shard-id I``.
Group assignment is deterministic and weighted by ``FINN_CI_TIMINGS_FILE``.
If the file is absent or has no signal, assignment falls back to
deterministic round-robin over sorted group keys. Items sharing an
``@pytest.mark.xdist_group`` always land in the same shard so chained
``load_test_checkpoint_or_skip`` steps don't break. Items without an
explicit group are grouped by nodeid.

``@pytest.mark.shard(N)`` pins a group for flaky-test isolation.
``--dry-run-shards`` prints the shard table and exits. ``--which-shard QUERY``
prints which Jenkins shard a matching nodeid would run on. A
``<stash>.timings.json`` sidecar is emitted next to the junit XML.
"""
import pytest

import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
JENKINS_DIR = os.path.join(REPO_ROOT, "docker", "jenkins")
if JENKINS_DIR not in sys.path:
    sys.path.insert(0, JENKINS_DIR)

import ci_sharding  # noqa: E402

SHARD_MARKER_NAME = "shard"


def pytest_addoption(parser):
    group = parser.getgroup("finn-ci-sharding")
    group.addoption(
        "--num-shards",
        type=int,
        default=0,
        help="Split the collected test set into N deterministic shards.",
    )
    group.addoption(
        "--shard-id",
        type=int,
        default=0,
        help="Which shard (0-indexed) to run. Requires --num-shards.",
    )
    group.addoption(
        "--dry-run-shards",
        action="store_true",
        default=False,
        help="Print shard assignment table and exit without running tests.",
    )
    group.addoption(
        "--which-shard",
        default=None,
        metavar="QUERY",
        help="Print which Jenkins shard each ci_sharding.STAGES row "
        "would run a matching test on. Use with --collect-only.",
    )


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
                "@pytest.mark.shard expects an int arg, got %r on %s" % (pinned, item.nodeid)
            )
        if not 0 <= pinned < num_shards:
            raise pytest.UsageError(
                "@pytest.mark.shard(%d) out of range for --num-shards=%d on %s"
                % (pinned, num_shards, item.nodeid)
            )
        return pinned
    return None


def _load_group_weights():
    return ci_sharding.load_group_weights(os.environ.get("FINN_CI_TIMINGS_FILE"))


def _weights_with_fallback():
    # Fallback is the median of recorded weights (1.0 if the file is absent or
    # has no positive entries) and is used for groups not yet timed.
    weights_table = _load_group_weights()
    fallback = ci_sharding.weights_with_fallback(weights_table)
    return weights_table, fallback


def _assignment_details(groups, num_shards):
    """Resolve shard pins, then delegate to ``ci_sharding.assign_groups_to_shards``.

    Each xdist_group must agree on a single ``@pytest.mark.shard(N)`` value,
    otherwise sibling tests would split across shards and break chained
    ``load_test_checkpoint_or_skip`` steps. Returns the tuple
    ``(assignment, source, shard_load, weights_table, fallback)`` so callers
    don't recompute the median weight.
    """
    pins = {}
    for key, members in groups.items():
        group_pins = {_pinned_shard(it, num_shards) for it in members}
        group_pins.discard(None)
        if len(group_pins) > 1:
            raise pytest.UsageError(
                "conflicting @pytest.mark.shard pins within xdist_group %r: %r"
                % (key, sorted(group_pins))
            )
        if group_pins:
            pins[key] = group_pins.pop()
    weights_table, fallback = _weights_with_fallback()
    assignment, source, shard_load, fallback = ci_sharding.assign_groups_to_shards(
        groups.keys(), num_shards, weights_table, pins
    )
    return assignment, source, shard_load, weights_table, fallback


def _assign_groups_to_shards(groups, num_shards):
    """Map ``{group_key: [items]}`` to ``{group_key: shard_id}``."""
    return _assignment_details(groups, num_shards)[0]


def _marker_tokens(marker_expr):
    # STAGES.marker is constrained to bare 'a or b ...' by MARKER_SAFE_PATTERN.
    return [t for t in marker_expr.split() if t != "or"]


def _item_matches_marker_expr(item, marker_expr):
    tokens = _marker_tokens(marker_expr)
    return any(any(True for _ in item.iter_markers(name=t)) for t in tokens)


def _run_which_shard(config, items, query):
    rows = []
    for stage in ci_sharding.STAGES:
        matching = [it for it in items if _item_matches_marker_expr(it, stage["marker"])]
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
            rows.append(
                (
                    stage["stage"],
                    stage["marker"],
                    int(stage["shards"]),
                    shard_id,
                    ci_sharding.stash_name(stage["stage"], int(stage["shards"]), shard_id),
                    it.nodeid,
                )
            )
    if not rows:
        print("[finn-sharding] --which-shard: no test matched %r in any stage" % query)
    else:
        header = ("stage", "marker", "shards", "shard", "stash", "nodeid")
        widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
        fmt = "  ".join("%%-%ds" % w for w in widths)
        print(fmt % header)
        print(fmt % tuple("-" * w for w in widths))
        for row in rows:
            print(fmt % tuple(str(c) for c in row))
    config.hook.pytest_deselected(items=list(items))
    items[:] = []


def _junit_output_info(config):
    junitxml = config.getoption("--junitxml") or ""
    if not junitxml:
        return None, None, None
    out_dir = os.path.dirname(junitxml) or "."
    stash = os.path.splitext(os.path.basename(junitxml))[0]
    return out_dir, stash, junitxml


def _stage_name_from_env():
    return os.environ.get("FINN_CI_STAGE", "")


def _write_shard_map(config, kept, groups, assignment, source, weights_table, fallback):
    out_dir, stash, _ = _junit_output_info(config)
    if not out_dir or not stash:
        return
    num_shards = int(config.getoption("--num-shards"))
    shard_id = int(config.getoption("--shard-id"))
    rows = []
    lines = []
    for item in sorted(kept, key=lambda it: it.nodeid):
        group = _group_key(item)
        weight = weights_table.get(group, fallback)
        row = {
            "nodeid": item.nodeid,
            "stage": _stage_name_from_env(),
            "stash": stash,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "group": group,
            "weight_s": round(float(weight), 3),
            "source": source.get(group, "unknown"),
        }
        rows.append(row)
        lines.append(
            "nodeid={nodeid} stage={stage} shard={shard_num}/{shard_count} "
            "stash={stash} group={group} weight_s={weight_s:.3f} source={source}".format(
                nodeid=row["nodeid"],
                stage=row["stage"],
                shard_num=shard_id + 1,
                shard_count=num_shards,
                stash=stash,
                group=group,
                weight_s=row["weight_s"],
                source=row["source"],
            )
        )
    try:
        with open(os.path.join(out_dir, stash + ".shardmap.json"), "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(os.path.join(out_dir, stash + ".shardmap.txt"), "w") as f:
            for line in lines:
                f.write(line + "\n")
    except OSError:
        pass


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
            "no tests collected for this marker. CI shard configuration is "
            "out of sync with the test markers"
        )

    groups = {}
    for item in items:
        groups.setdefault(_group_key(item), []).append(item)
    assignment, source, _shard_load, weights_table, fallback = _assignment_details(
        groups, num_shards
    )

    if dry_run:
        per_shard = {i: {"items": 0, "groups": [], "seconds": 0.0} for i in range(num_shards)}
        for key, members in groups.items():
            s = assignment[key]
            per_shard[s]["items"] += len(members)
            per_shard[s]["groups"].append(key)
            per_shard[s]["seconds"] += weights_table.get(key, fallback)
        print(
            "\n--- dry-run-shards (num_shards=%d, groups=%d, items=%d) ---"
            % (num_shards, len(groups), len(items))
        )
        print("%-8s %-8s %-8s %-10s %s" % ("shard", "items", "groups", "weight_s", "sample_group"))
        for i in range(num_shards):
            gs = sorted(per_shard[i]["groups"])
            sample = gs[0] if gs else "(empty)"
            print(
                "%-8d %-8d %-8d %-10.1f %s"
                % (i, per_shard[i]["items"], len(gs), per_shard[i]["seconds"], sample)
            )
        config.hook.pytest_deselected(items=list(items))
        items[:] = []
        return

    kept = [it for it in items if assignment[_group_key(it)] == shard_id]
    _write_shard_map(config, kept, groups, assignment, source, weights_table, fallback)
    my_groups = sorted(k for k, s in assignment.items() if s == shard_id)
    print(
        "[finn-sharding] shard %d/%d: %d item(s) across %d group(s)"
        % (shard_id, num_shards, len(kept), len(my_groups))
    )
    if my_groups:
        sample = my_groups[:5]
        ellipsis = " ..." if len(my_groups) > 5 else ""
        print("[finn-sharding] groups: %s%s" % (sample, ellipsis))
    items[:] = kept


# Per-shard timing observability: emit <stash>.timings.json next to the
# junit XML when --num-shards is set, so aggregation can flag outliers
# and operators can update the Jenkins timing state from a recent run.

# Set on the controller in pytest_sessionstart, left None on xdist workers
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
    # Accumulate setup+call+teardown per nodeid. xdist forwards worker reports
    # here so summing covers the full session.
    duration = float(getattr(report, "duration", 0.0) or 0.0)
    _TIMINGS["per_test_seconds"][report.nodeid] = (
        _TIMINGS["per_test_seconds"].get(report.nodeid, 0.0) + duration
    )


def pytest_sessionfinish(session, exitstatus):
    # Exit 5 ("no tests collected") is success for every sharding-aware
    # invocation: --dry-run-shards and --which-shard finish their work by
    # deselecting everything, and a real sharded run can legitimately end
    # up empty when a shard's slice of the test set happens to be all
    # deselected upstream (the empty-collection footgun is caught by the
    # UsageError raised earlier in pytest_collection_modifyitems, so by
    # the time we reach here exitstatus 5 is benign). Mapping the exit
    # code here keeps run-tests.sh (and any future caller) free of
    # pytest-version-coupled shell knowledge.
    sharded = (
        session.config.getoption("--num-shards") > 0
        or session.config.getoption("--dry-run-shards")
        or session.config.getoption("--which-shard")
    )
    if exitstatus == 5 and sharded:
        session.exitstatus = 0
    if _TIMINGS is None:
        return
    out_dir, stash, junitxml = _junit_output_info(session.config)
    if not junitxml:
        return
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
        "metadata": {
            "job": os.environ.get("FINN_CI_JOB_NAME"),
            "build": os.environ.get("FINN_CI_BUILD_NUMBER"),
            "stage": os.environ.get("FINN_CI_STAGE"),
            "timings_file": os.environ.get("FINN_CI_TIMINGS_FILE"),
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
