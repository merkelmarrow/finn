#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Jenkins-only FINN CI sharding and timing helpers.

This module is used both as a small command-line tool from Jenkins and as the
shared implementation imported by ``tests/conftest.py`` when CI sharding is
enabled.
"""

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


SLOW_FACTOR = 1.5
GROUP_SUFFIX_RE = re.compile(r"@(\S+)$")
NOTEBOOK_PARAM_RE = re.compile(r"\[(/[^\]]+\.ipynb)\]")
DEFAULT_TIMINGS_SEED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ci_timings_seed.json"
)


# Per-tree rotation policy consumed by the Groovy rotateBuildTrees() helper
# and the prune-tmp/prune-images/prune-artifacts subcommands. Lives here so
# tuning is a one-file change. Two profiles cover all three trees:
#
# TRANSIENT covers scratch trees that are rebuilt cheaply (tmp shard
# workspaces, the shared Docker layer cache). 5/3 newest + 14 days mtime is
# plenty.
#
# HANDOFF covers the SW->HW artifact tree, which doubles as the per-board
# fallback when an individual board's most recent build regressed. The
# 30 newest + 30 days mtime window is intentionally deep enough to outlast
# the longest realistic single-board failure streak. HW falling back to a
# build older than this indicates a separate problem (not a tuning issue).
TRANSIENT_RETENTION = {"retain": 5, "ageDays": 14}
HANDOFF_RETENTION = {"retain": 30, "ageDays": 30}

RETENTION = {
    "tmp": TRANSIENT_RETENTION,
    "image": {**TRANSIENT_RETENTION, "retain": 3},
    "artifact": HANDOFF_RETENTION,
}


# Per-board HW-pipeline metadata. Consumed by the Groovy HW pipeline via
# ``python3 ci_sharding.py hw-shards-json`` so the literal HW_SHARDS table is
# not duplicated between Python and Groovy. ``agentLabel`` is the Jenkins
# label of the board agent, ``credentialsId`` is the Jenkins credential
# binding used to sudo on the board (None for Alveo PCIe boards mounted as
# root), ``restartPrep`` toggles the pre-run reboot dance for Zynq boards,
# ``setupScript`` selects the bash sourcing path inside the test script
# (``alveo`` for XRT, ``pynq`` for the on-board pynq venv), and ``marker``
# is the literal pytest ``-m`` expression the board runs against (not a
# derived form, e.g. ``Pynq-Z1`` runs against marker ``Pynq``).
BOARDS = {
    "U250": {
        "agentLabel": "finn-u250",
        "credentialsId": None,
        "restartPrep": False,
        "setupScript": "alveo",
        "marker": "U250",
    },
    "Pynq-Z1": {
        "agentLabel": "finn-pynq",
        "credentialsId": "pynq-z1-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "Pynq",
    },
    "ZCU104": {
        "agentLabel": "finn-zcu104",
        "credentialsId": "pynq-z1-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "ZCU104",
    },
    "KV260_SOM": {
        "agentLabel": "finn-kv260",
        "credentialsId": "user-ubuntu-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "KV260_SOM",
    },
}


# PRESET=smoke turns on the CI params named here and leaves every other
# CI-progression param off. Tuple rather than ``STAGES[0].param`` so a
# future reorder of STAGES cannot silently change what "smoke" means.
SMOKE_PARAMS = ("sanity",)


# zipArtifacts.boards lists the boards a row produces a build artefact for.
# Every entry must also be a key of BOARDS (validated by validate_boards()).
# zipArtifacts.hwTestType tags which HW-pipeline category that artefact feeds.
# The nested shape means the pair is either fully present or fully absent.
# The SW pipeline writes each zip to
# ${ARTIFACT_DIR}/ci_runs/<jobkey>/<BUILD>/zips/<hwTestType>/<board>.zip.
STAGES = [
    {
        "param": "sanity",
        "stage": "Sanity - Build Hardware",
        "marker": "sanity_bnn",
        "shards": 1,
        "workers": 1,
        "zipArtifacts": {
            "hwTestType": "bnn_build_sanity",
            "boards": ["U250", "Pynq-Z1", "ZCU104", "KV260_SOM"],
        },
    },
    {
        "param": "sanity",
        "stage": "Sanity - Unit Tests",
        "marker": "util or brevitas_export or streamline or transform or notebooks",
        "shards": 1,
        "workers": 8,
        "coverage": True,
        "distMode": "loadgroup",
    },
    {
        "param": "fpgadataflow",
        "stage": "fpgadataflow",
        "marker": "fpgadataflow",
        "shards": 2,
        "workers": 8,
        "coverage": True,
    },
    {
        "param": "end2end",
        "stage": "End2end",
        "marker": "end2end",
        "shards": 3,
        "workers": 6,
        "distMode": "loadgroup",
    },
    {
        "param": "end2end",
        "stage": "BNN U250",
        "marker": "bnn_u250",
        "shards": 2,
        "workers": 2,
        "distMode": "loadgroup",
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["U250"]},
    },
    {
        "param": "end2end",
        "stage": "BNN Pynq-Z1",
        "marker": "bnn_pynq",
        "shards": 3,
        "workers": 2,
        "distMode": "loadgroup",
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["Pynq-Z1"]},
    },
    {
        "param": "end2end",
        "stage": "BNN ZCU104",
        "marker": "bnn_zcu104",
        "shards": 2,
        "workers": 4,
        "distMode": "loadgroup",
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["ZCU104"]},
    },
    {
        "param": "end2end",
        "stage": "BNN KV260",
        "marker": "bnn_kv260",
        "shards": 2,
        "workers": 2,
        "distMode": "loadgroup",
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["KV260_SOM"]},
    },
]


def validate_boards(stages=None, boards=None):
    """Fail loud if a STAGES row mentions a board not declared in BOARDS.

    Called by every CLI subcommand that exposes either table so the HW
    pipeline cannot drift past a half-added board (e.g. a STAGES row added
    without a matching BOARDS entry would yield a zip nobody can run).
    """
    stages = stages if stages is not None else STAGES
    boards = boards if boards is not None else BOARDS
    referenced = set()
    for row in stages:
        zip_art = row.get("zipArtifacts") or {}
        for b in zip_art.get("boards", []) or []:
            referenced.add(str(b))
    missing = sorted(referenced - set(boards))
    if missing:
        raise ValueError("STAGES references board(s) not declared in BOARDS: %r" % missing)


def hw_shards(boards=None):
    """Return the BOARDS table as an ordered list of ``{board: ..., ...}`` rows.

    The Groovy HW pipeline iterates a list of rows rather than a map, so
    flatten the dict here with a ``board`` key prepended. Order matches
    BOARDS insertion order so the per-board parallel branch order in the
    HW pipeline is stable.
    """
    boards = boards if boards is not None else BOARDS
    return [dict(board=name, **fields) for name, fields in boards.items()]


def stash_name(stage, shards, shard_id):
    """Mirror ``shardStashName()`` in ``docker/jenkins/Jenkinsfile``."""
    base = re.sub(r"^_|_$", "", re.sub(r"[^a-z0-9]+", "_", stage.lower()))
    return base if shards <= 1 else "%s_%d" % (base, shard_id + 1)


def shard_stage_name(stage, shards, shard_id):
    return stage if int(shards) <= 1 else "%s (%d/%d)" % (stage, shard_id + 1, int(shards))


def job_key(job_name):
    # strip-dots prevents JOB_NAME=".." surviving the sanitiser and turning
    # the prune paths into a parent-directory traversal.
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name or "job").strip(".")
    return out or "job"


def ci_param_names(stages=None):
    """Return the distinct ``param`` field values across ``stages`` in order.

    Used to derive ``is_full_matrix_run`` (and the Groovy ``ciParams()`` helper)
    so a new STAGES row with a new param name is picked up without editing
    callers.
    """
    stages = stages if stages is not None else STAGES
    seen = []
    for row in stages:
        name = row.get("param")
        if name and name not in seen:
            seen.append(name)
    return seen


def ci_params_payload(stages=None, smoke_params=SMOKE_PARAMS):
    """Return the CI-progression param manifest consumed by the Groovy pipeline.

    ``smoke`` is the explicit ``SMOKE_PARAMS`` list filtered to params that
    actually exist in ``stages``, so a SMOKE_PARAMS entry referencing a
    no-longer-existing param is dropped silently rather than enabling
    something the operator did not expect.
    """
    params = ci_param_names(stages)
    smoke = [p for p in smoke_params if p in params]
    return {
        "params": params,
        "smoke": smoke,
    }


def rows_that_would_run(enabled_params, stages=None):
    """Return the subset of ``stages`` that the given param toggles execute.

    A row runs when its ``param`` is in ``enabled_params`` and its
    ``skipWhen`` (if any) is not. Used by ``is_full_matrix_run`` so auto-
    promote only fires when timing data is refreshed for every row in the
    canonical config, not just every row whose param happened to be ticked.
    """
    stages = stages if stages is not None else STAGES
    enabled = set(enabled_params or [])
    out = []
    for row in stages:
        param = row.get("param")
        if not param or param not in enabled:
            continue
        skip = row.get("skipWhen")
        if skip and skip in enabled:
            continue
        out.append(row)
    return out


def is_full_matrix_run(enabled_params, stage_filter="", stages=None):
    """Return True iff every row in ``stages`` would execute on this run.

    Stricter than "every CI param ticked" because a row with ``skipWhen``
    can still be skipped by the combination of params. The canonical
    ``STAGES`` has no ``skipWhen`` rows today, so ticking every CI param
    (``sanity + fpgadataflow + end2end``) with no ``STAGE_FILTER`` is
    enough to trigger auto-promote on a successful build. The mechanism
    stays in place for future rows that need conditional skipping.
    """
    stages = stages if stages is not None else STAGES
    if stage_filter:
        return False
    if not stages:
        return False
    return len(rows_that_would_run(enabled_params, stages)) == len(stages)


def full_matrix_status(enabled_params, stage_filter="", stages=None):
    stages = stages if stages is not None else STAGES
    enabled = ci_param_names(rows_that_would_run(enabled_params, stages))
    return {
        "full": is_full_matrix_run(enabled_params, stage_filter, stages),
        "enabled": list(enabled_params or []),
        "required": ci_param_names(stages),
        "running": enabled,
        "stage_filter": stage_filter,
    }


def canonical_key(name):
    """Normalise timing group names across xdist and workspace-specific paths."""
    m = GROUP_SUFFIX_RE.search(name)
    if m:
        return m.group(1)
    m2 = NOTEBOOK_PARAM_RE.search(name)
    if m2:
        path = m2.group(1)
        return name.replace(m2.group(0), "[%s]" % os.path.basename(path))
    return name


def read_json(path, default=None):
    if not path:
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        # The missing-file fallback above is the expected idle state. Warn
        # loudly when the file exists but is unreadable or malformed so a
        # corrupt master state doesn't silently degrade sharding to round-robin.
        print(
            "ci_sharding read_json: %s: %s: %s" % (path, exc.__class__.__name__, exc),
            file=sys.stderr,
        )
        return default


def write_json_atomic(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        os.makedirs(parent)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.rename(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_group_weights(path):
    """Return ``{group_name: seconds}`` from a timing state file."""
    data = read_json(path, default={})
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, dict):
        return {}
    weights = {}
    for key, val in groups.items():
        try:
            if isinstance(val, dict):
                seconds = float(val.get("seconds", 0.0) or 0.0)
            else:
                seconds = float(val or 0.0)
        except (TypeError, ValueError):
            continue
        if seconds > 0.0:
            weights[str(key)] = seconds
    return weights


def weights_with_fallback(weights):
    # Median-by-index keeps the function on the Python stdlib (no
    # ``statistics`` import) and the canonical input is ~50 timing entries,
    # so the O(n log n) sort is irrelevant. 1.0 is a deliberate placeholder
    # for "no signal at all" so a brand-new master still produces a
    # reproducible round-robin assignment.
    known = sorted(w for w in weights.values() if w > 0.0)
    fallback = known[len(known) // 2] if known else 1.0
    return fallback


def assign_groups_to_shards(group_keys, num_shards, weights=None, pins=None):
    """Assign group keys to shard ids.

    Pins win first. If there is useful timing signal, unpinned groups are
    assigned by LPT-greedy bin packing. Otherwise they are round-robin over
    sorted group keys so fallback placement is reproducible and balanced by
    group count.

    Raises ``ValueError`` for ``num_shards < 1`` or any pin outside
    ``[0, num_shards)`` so a future direct caller (off the conftest path
    which validates separately) cannot trigger an ``IndexError``.
    """
    num_shards = int(num_shards)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1, got %r" % (num_shards,))
    pins = pins or {}
    for key, shard in pins.items():
        try:
            shard_int = int(shard)
        except (TypeError, ValueError):
            raise ValueError("pin for %r must be an int, got %r" % (key, shard))
        if not 0 <= shard_int < num_shards:
            raise ValueError(
                "pin for %r out of range: %d not in [0, %d)" % (key, shard_int, num_shards)
            )
    group_keys = sorted(str(k) for k in group_keys)
    weights = weights or {}
    assignment = {}
    source = {}
    shard_load = [0.0] * num_shards
    fallback = weights_with_fallback(weights)

    for key in group_keys:
        if key in pins:
            shard = int(pins[key])
            assignment[key] = shard
            source[key] = "pinned"
            shard_load[shard] += weights.get(key, fallback)

    if num_shards <= 1:
        for key in group_keys:
            assignment.setdefault(key, 0)
            source.setdefault(key, "single")
        return assignment, source, shard_load, fallback

    unpinned = [k for k in group_keys if k not in assignment]
    has_signal = any(weights.get(k, 0.0) > 0.0 for k in unpinned)

    if not has_signal:
        for idx, key in enumerate(unpinned):
            shard = idx % num_shards
            assignment[key] = shard
            source[key] = "round_robin"
            shard_load[shard] += 1.0
        return assignment, source, shard_load, fallback

    unpinned.sort(key=lambda k: (-weights.get(k, fallback), k))
    for key in unpinned:
        weight = weights.get(key, fallback)
        shard = min(range(num_shards), key=lambda s: (shard_load[s], s))
        assignment[key] = shard
        source[key] = "known" if key in weights else "fallback"
        shard_load[shard] += weight
    return assignment, source, shard_load, fallback


def timing_rows(reports_dir):
    rows = []
    pattern = os.path.join(reports_dir, "*.timings.json")
    for path in sorted(glob.glob(pattern)):
        if os.path.basename(path) == "ci_timings_master.json":
            continue
        data = read_json(path, default={})
        if not isinstance(data, dict):
            print("ci_sharding summarize: could not parse %s" % path, file=sys.stderr)
            continue
        stash = data.get("stash") or os.path.basename(path).split(".")[0]
        groups = data.get("groups") or []
        top = groups[0] if groups else {"name": "(none)", "seconds": 0.0}
        rows.append(
            (
                stash,
                int(data.get("shard", {}).get("id", 0)),
                float(data.get("wall_seconds", 0.0) or 0.0),
                float(top.get("seconds", 0.0) or 0.0),
                str(top.get("name", "")),
            )
        )
    return rows


def family(stash):
    return re.sub(r"_\d+$", "", stash)


def summarize_timings(reports_dir):
    rows = timing_rows(reports_dir)
    if not rows:
        print("ci_sharding summarize: no parseable timings.json files in %s" % reports_dir)
        return 0
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[family(row[0])].append(row)
    print()
    print("=== per-shard wall-clock ===")
    print("%-36s %3s %10s %12s  %s" % ("stash", "id", "wall_s", "max_group_s", "max_group"))
    print("-" * 100)
    slow_found = False
    for fam in sorted(by_family):
        fam_rows = sorted(by_family[fam], key=lambda r: r[1])
        walls = sorted(r[2] for r in fam_rows)
        median = walls[len(walls) // 2] if walls else 0.0
        for stash, sid, wall, mx_sec, mx_name in fam_rows:
            flag = ""
            if median > 0.0 and wall > SLOW_FACTOR * median:
                flag = "  <<< SLOW SHARD (%.1fx median)" % (wall / median)
                slow_found = True
            print("%-36s %3d %10.1f %12.1f  %s%s" % (stash, sid, wall, mx_sec, mx_name, flag))
        print()
    if slow_found:
        print(
            "ci_sharding summarize: one or more shards exceeded %.1fx family median. "
            "Future builds will refresh the timing master with this run's data" % SLOW_FACTOR
        )
    return 0


def normalise_master(data):
    if not isinstance(data, dict):
        data = {}
    groups = data.get("groups")
    if not isinstance(groups, dict):
        groups = {}
    return {
        "version": 1,
        "updated_at": data.get("updated_at"),
        "groups": dict(groups),
    }


def observed_groups_from_reports(reports_dir):
    observed = {}
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.timings.json"))):
        if os.path.basename(path) == "ci_timings_master.json":
            continue
        data = read_json(path, default={})
        if not isinstance(data, dict):
            continue
        metadata = data.get("metadata") or {}
        for entry in data.get("groups") or []:
            name = canonical_key(str(entry.get("name", "")))
            if not name:
                continue
            try:
                seconds = float(entry.get("seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if seconds <= 0.0:
                continue
            observed[name] = {
                "seconds": round(seconds, 3),
                "count": int(entry.get("count", 0) or 0),
                "last_seen_job": metadata.get("job"),
                "last_seen_build": metadata.get("build"),
                "last_seen_stage": metadata.get("stage"),
                "last_seen_stash": data.get("stash"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
    return observed


CORRUPT_BACKUP_RETAIN = 5


def _prune_corrupt_backups(master_path, retain=CORRUPT_BACKUP_RETAIN):
    """Keep only the newest ``retain`` ``<master>.corrupt-*`` siblings.

    A pathological NFS hiccup window could otherwise leave dozens of
    backups cluttering the timing state directory.
    """
    parent = os.path.dirname(os.path.abspath(master_path))
    base = os.path.basename(master_path)
    prefix = base + ".corrupt-"
    try:
        names = [n for n in os.listdir(parent) if n.startswith(prefix)]
    except OSError:
        return
    if len(names) <= retain:
        return
    names.sort()
    for stale in names[:-retain]:
        try:
            os.unlink(os.path.join(parent, stale))
        except OSError:
            pass


def _backup_if_corrupt(master_path):
    """Rename a non-empty unparseable master aside so it isn't silently lost.

    Returns True when a backup was made. The new master will be written
    fresh by the caller, the operator can salvage from the .corrupt-* copy.
    The corrupt-backup history is capped at ``CORRUPT_BACKUP_RETAIN``.
    """
    if not os.path.isfile(master_path):
        return False
    try:
        if os.path.getsize(master_path) == 0:
            return False
    except OSError:
        return False
    try:
        with open(master_path) as f:
            json.load(f)
        return False
    except (OSError, ValueError):
        backup = "%s.corrupt-%d" % (master_path, int(time.time()))
        try:
            os.rename(master_path, backup)
            print(
                "ci_sharding locked_update: %s was unparseable, moved to %s"
                % (master_path, backup),
                file=sys.stderr,
            )
            _prune_corrupt_backups(master_path)
            return True
        except OSError as exc:
            print(
                "ci_sharding locked_update: could not back up corrupt %s: %s" % (master_path, exc),
                file=sys.stderr,
            )
            return False


def locked_update(master_path, update_fn):
    """Acquire an exclusive lock and apply ``update_fn`` to the master JSON.

    The lock file is created next to the master with a leading dot
    (``.<basename>.lock``) and is intentionally never deleted, repeated
    callers reuse the same file. The dot prefix keeps it out of ``ls``
    listings, which matters because operators ``cat`` the master.
    """
    if not master_path:
        return update_fn({})
    parent = os.path.dirname(os.path.abspath(master_path))
    if not os.path.isdir(parent):
        os.makedirs(parent)
    base = os.path.basename(master_path)
    lock_path = os.path.join(parent, "." + base + ".lock")
    with open(lock_path, "a+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _backup_if_corrupt(master_path)
        current = read_json(master_path, default={})
        updated = update_fn(current)
        write_json_atomic(master_path, updated)
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return updated


def update_master(reports_dir, master_path=None, out_path=None, promote=False, metadata=None):
    observed = observed_groups_from_reports(reports_dir)
    metadata = metadata or {}

    def apply(current, promoted):
        master = normalise_master(current)
        master["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        master["groups"].update(observed)
        master["last_update"] = {
            "job": metadata.get("job"),
            "build": metadata.get("build"),
            "full_run": bool(metadata.get("full_run")),
            "stage_filter": metadata.get("stage_filter", ""),
            "promoted": promoted,
            "observed_groups": len(observed),
        }
        return master

    if master_path and promote:
        try:
            master = locked_update(master_path, lambda cur: apply(cur, True))
            status = "promoted"
        except Exception as exc:
            # promote failure is advisory: the preview's last_update.promoted
            # records it for operator inspection. We still return 0 because
            # the per-build run already succeeded by the time this is called.
            print(
                "ci_sharding update: persistent master update failed: %s: %s"
                % (exc.__class__.__name__, exc),
                file=sys.stderr,
            )
            master = apply(read_json(master_path, default={}), False)
            status = "promote-failed"
    elif master_path:
        master = apply(read_json(master_path, default={}), False)
        status = "preview-only"
    else:
        master = apply({}, False)
        status = "no-master"
    if out_path:
        write_json_atomic(out_path, master)
    print(
        "ci_sharding update: merged %d observed group(s), %d group(s) in result, status=%s"
        % (len(observed), len(master.get("groups", {})), status)
    )
    return 0


SEED_COMMENT = (
    "Seed per-group wall seconds for Jenkins CI, refreshed dynamically from "
    "archived *.timings.json reports."
)


def regen_seed(master_path, out_path, comment=SEED_COMMENT):
    """Write a fresh ``ci_timings_seed.json`` from a promoted master.

    The master state file is rich (per-group ``{seconds, count, last_seen_*}``)
    so it can drive observability. The seed checked into source control is
    intentionally narrow (``{group_name: seconds}``) so reviewers can scan
    it and the median-fallback computation has no surface area to disagree
    about. Pulls just ``seconds`` from each group, drops anything with a
    non-positive timing because that's noise from a failed shard rather
    than a real measurement, and sorts the output for diff-friendliness.
    """
    master = read_json(master_path, default=None)
    if not isinstance(master, dict):
        raise ValueError("regen-seed: master %r is not a JSON object" % master_path)
    groups = master.get("groups") or {}
    if not isinstance(groups, dict):
        raise ValueError("regen-seed: master groups must be a JSON object")
    flat = {}
    for name, entry in groups.items():
        try:
            if isinstance(entry, dict):
                seconds = float(entry.get("seconds", 0.0) or 0.0)
            else:
                seconds = float(entry or 0.0)
        except (TypeError, ValueError):
            continue
        if seconds > 0.0:
            flat[str(name)] = round(seconds, 3)
    payload = {"_comment": comment, "groups": dict(sorted(flat.items()))}
    write_json_atomic(out_path, payload)
    print("ci_sharding regen-seed: wrote %s with %d group(s)" % (out_path, len(flat)))
    return 0


def prepare_timing_snapshot(master_path, snapshot_path, seed_path=DEFAULT_TIMINGS_SEED):
    master = read_json(master_path, default=None)
    if master is None:
        master = read_json(seed_path, default=None)
    master = normalise_master(master)
    write_json_atomic(snapshot_path, master)
    print(
        "ci_sharding prepare: wrote %s with %d group(s)"
        % (snapshot_path, len(master.get("groups", {})))
    )
    return 0


def load_map_rows(path):
    data = read_json(path, default=[])
    if isinstance(data, list):
        return data
    return []


def merge_maps(reports_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.shardmap.json"))):
        rows.extend(load_map_rows(path))
    rows.sort(
        key=lambda r: (
            str(r.get("stage", "")),
            int(r.get("shard_id", 0)),
            str(r.get("nodeid", "")),
        )
    )
    json_path = os.path.join(reports_dir, "shard_map.json")
    txt_path = os.path.join(reports_dir, "shard_map.txt")
    write_json_atomic(json_path, rows)
    with open(txt_path, "w") as f:
        for row in rows:
            f.write(
                "nodeid={nodeid} stage={stage} shard={shard_num}/{shard_count} "
                "stash={stash} group={group} weight_s={weight_s:.3f} source={source}\n".format(
                    nodeid=row.get("nodeid", ""),
                    stage=row.get("stage", ""),
                    shard_num=int(row.get("shard_id", 0)) + 1,
                    shard_count=int(row.get("num_shards", 1)),
                    stash=row.get("stash", ""),
                    group=row.get("group", ""),
                    weight_s=float(row.get("weight_s", 0.0) or 0.0),
                    source=row.get("source", ""),
                )
            )
    print("ci_sharding merge-maps: wrote %d row(s)" % len(rows))
    return 0


def resolve_sw_zips(artifact_dir, job_key, test_types, boards, sw_build_dir=""):
    """Resolve ``(testType, board) -> {"zip": ..., "swBuildDir": ...}`` for every pair.

    Walks ``${artifact_dir}/ci_runs/<job_key>/`` once newest-first and picks
    the highest-numbered build directory whose
    ``zips/<testType>/<board>.zip.READY`` sibling is present. Boards with no
    READY come back as ``{}``. Replaces eight per-board ``find | sort`` shell
    walks in the HW pipeline's resolveSwBoardZipPaths with a single Python
    walk that lists the parent once and stat-probes each build's zip dirs.

    ``sw_build_dir`` is the operator-supplied explicit override: when set,
    that single directory is consulted for every pair instead of the
    auto-discovery walk. A missing READY there is reported per-board (each
    pair still resolves to ``{}``) but does not abort the whole call, so a
    partially-READY explicit dir does not strand boards that did make it.
    """
    out = {tt: {b: {} for b in boards} for tt in test_types}
    if sw_build_dir:
        for tt in test_types:
            for b in boards:
                zip_path = os.path.join(sw_build_dir, "zips", tt, "%s.zip" % b)
                if os.path.isfile(zip_path) and os.path.isfile(zip_path + ".READY"):
                    build = os.path.basename(os.path.normpath(sw_build_dir))
                    out[tt][b] = {
                        "zip": zip_path,
                        "swBuildDir": sw_build_dir,
                        "swBuild": build,
                        "latestSwBuild": build,
                        "fallback": False,
                    }
        return out

    job_root = os.path.join(artifact_dir, "ci_runs", job_key)
    if not os.path.isdir(job_root):
        return out
    try:
        candidates = sorted(
            (d for d in os.listdir(job_root) if d.isdigit()),
            key=int,
            reverse=True,
        )
    except OSError:
        return out

    latest_build = candidates[0] if candidates else ""
    # Iterate newest first and short-circuit per pair so an old build's
    # READY only ever wins for boards the newer builds did not produce.
    remaining = {(tt, b) for tt in test_types for b in boards}
    for build in candidates:
        if not remaining:
            break
        build_dir = os.path.join(job_root, build)
        for tt, b in list(remaining):
            zip_path = os.path.join(build_dir, "zips", tt, "%s.zip" % b)
            if os.path.isfile(zip_path) and os.path.isfile(zip_path + ".READY"):
                out[tt][b] = {"zip": zip_path, "swBuildDir": build_dir, "swBuild": build}
                remaining.discard((tt, b))
    for tt in test_types:
        selected = [entry for entry in out.get(tt, {}).values() if entry.get("swBuild")]
        if not selected:
            continue
        for entry in selected:
            entry["latestSwBuild"] = latest_build
            entry["fallback"] = str(entry["swBuild"]) != latest_build
    return out


def prune_numeric_builds(parent, current_build, retain_n, max_age_days, dry_run, *, tag):
    """Delete numeric-named subdirs of ``parent`` outside the newest ``retain_n``.

    Returns the number of directories matched for deletion (whether or not
    ``dry_run`` actually removed them), so summary callers report the same
    figure on real runs and previews. ``tag`` is a short prefix for log lines
    so callers like prune-tmp and prune-images keep their grep-friendly output.
    ``tag`` is keyword-only so future callers can't accidentally pass it
    positionally in the wrong slot.

    ``current_build`` must be an integer-like string. A non-numeric value
    would never match a sibling under ``parent`` (which is filtered to
    ``str.isdigit`` entries) so the "always keep the current build" guard
    would silently degenerate into "keep only retain_n". Refuse it loudly.
    """
    retain_n = int(retain_n)
    max_age_days = int(max_age_days)
    if retain_n < 1:
        raise ValueError("retain_n must be >= 1")
    try:
        current_build = str(int(str(current_build)))
    except (TypeError, ValueError):
        raise ValueError("current_build must be an integer-like string, got %r" % (current_build,))
    if not os.path.isdir(parent):
        return 0
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    nums = sorted((d for d in os.listdir(parent) if d.isdigit()), key=int)
    keep = set(nums[-retain_n:])
    keep.add(current_build)
    matched = 0
    for num in nums:
        if num in keep:
            continue
        path = os.path.join(parent, num)
        # Another CI run on the same NFS-shared parent can rmtree concurrently
        # with us, so the getmtime/rmtree calls below may race. Treat a vanish
        # as already-pruned and keep going.
        if max_age_days > 0:
            try:
                if os.path.getmtime(path) >= cutoff:
                    continue
            except FileNotFoundError:
                continue
        matched += 1
        if dry_run:
            print("ci_sharding %s: would delete %s" % (tag, path))
        else:
            print("ci_sharding %s: deleting %s" % (tag, path))
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
    return matched


def _coerce_current_build(value, tag):
    # Hoisted from prune_numeric_builds so a non-numeric BUILD_NUMBER fails
    # loudly at the boundary instead of silently aborting the loop after
    # the first inner call. prune_numeric_builds keeps its own check as
    # defence-in-depth for direct callers.
    try:
        return str(int(str(value)))
    except (TypeError, ValueError):
        raise ValueError(
            "ci_sharding %s: current_build must be an integer-like string, got %r" % (tag, value)
        )


def prune_tmp(nfs_base, job_key, current_build, retain_n, max_age_days, dry_run=False):
    current_build = _coerce_current_build(current_build, "prune-tmp")
    if not os.path.isdir(nfs_base):
        print("ci_sharding prune-tmp: nfs root %s not present, skipping" % nfs_base)
        return 0
    trees = 0
    matched = 0
    for name in sorted(os.listdir(nfs_base)):
        base = os.path.join(nfs_base, name, "workspace", "tmp", "ci_runs", job_key)
        if not os.path.isdir(base):
            continue
        trees += 1
        matched += prune_numeric_builds(
            base, current_build, retain_n, max_age_days, dry_run, tag="prune-tmp"
        )
    print(
        "ci_sharding prune-tmp: done (nfs_base=%s job_key=%s current=%s retain_n=%s "
        "max_age_days=%s dry_run=%s trees=%d matched=%d)"
        % (nfs_base, job_key, current_build, retain_n, max_age_days, int(dry_run), trees, matched)
    )
    return 0


def prune_images(shared_dir, job_key, current_build, retain_n, max_age_days, dry_run=False):
    current_build = _coerce_current_build(current_build, "prune-images")
    parent = os.path.join(shared_dir, job_key)
    if not os.path.isdir(parent):
        print("ci_sharding prune-images: shared dir %s not present, skipping" % parent)
        return 0
    matched = prune_numeric_builds(
        parent, current_build, retain_n, max_age_days, dry_run, tag="prune-images"
    )
    print(
        "ci_sharding prune-images: done (parent=%s current=%s retain_n=%s "
        "max_age_days=%s dry_run=%s matched=%d)"
        % (parent, current_build, retain_n, max_age_days, int(dry_run), matched)
    )
    return 0


def prune_artifacts(artifact_dir, job_key, current_build, retain_n, max_age_days, dry_run=False):
    """Rotate ${ARTIFACT_DIR}/ci_runs/<job_key>/ for this SW job.

    HW always resolves to the newest READY zip per board, so deleting an
    older build cannot strand a HW shard, HW re-resolves to the next-oldest
    READY on its next collectBuildArtifacts pass.
    """
    current_build = _coerce_current_build(current_build, "prune-artifacts")
    parent = os.path.join(artifact_dir, "ci_runs", job_key)
    if not os.path.isdir(parent):
        print("ci_sharding prune-artifacts: %s not present, skipping" % parent)
        return 0
    matched = prune_numeric_builds(
        parent, current_build, retain_n, max_age_days, dry_run, tag="prune-artifacts"
    )
    print(
        "ci_sharding prune-artifacts: done (parent=%s current=%s retain_n=%s "
        "max_age_days=%s dry_run=%s matched=%d)"
        % (parent, current_build, retain_n, max_age_days, int(dry_run), matched)
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("stages-json")
    sub.add_parser("ci-params-json")
    sub.add_parser("retention-json")
    sub.add_parser("hw-shards-json")

    p = sub.add_parser("stash-name")
    p.add_argument("stage")
    p.add_argument("shards", type=int)
    p.add_argument("shard_id", type=int)

    p = sub.add_parser("job-key")
    p.add_argument("name")

    p = sub.add_parser("full-matrix-json")
    p.add_argument("--enabled", default="")
    p.add_argument("--stage-filter", default="")

    p = sub.add_parser("prepare")
    p.add_argument("--master", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--seed", default=DEFAULT_TIMINGS_SEED)

    p = sub.add_parser("regen-seed")
    p.add_argument("--master", required=True)
    p.add_argument("--out", default=DEFAULT_TIMINGS_SEED)

    p = sub.add_parser("summarize")
    p.add_argument("reports_dir")

    p = sub.add_parser("update")
    p.add_argument("--reports", required=True)
    p.add_argument("--master", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--job", default="")
    p.add_argument("--build", default="")
    p.add_argument("--full-run", action="store_true")
    p.add_argument("--stage-filter", default="")

    p = sub.add_parser("merge-maps")
    p.add_argument("reports_dir")

    p = sub.add_parser("resolve-sw-zips")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--job-key", required=True)
    p.add_argument(
        "--tests",
        required=True,
        help="Comma-separated HW test types (e.g. bnn_build_sanity,bnn_build_full)",
    )
    p.add_argument("--boards", required=True, help="Comma-separated board names")
    p.add_argument(
        "--sw-build-dir", default="", help="Optional explicit SW build directory override"
    )

    p = sub.add_parser("prune-tmp")
    p.add_argument("nfs_base")
    p.add_argument("job_key")
    p.add_argument("current_build")
    p.add_argument("retain_n", type=int)
    p.add_argument("max_age_days", type=int)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("prune-images")
    p.add_argument("shared_dir")
    p.add_argument("job_key")
    p.add_argument("current_build")
    p.add_argument("retain_n", type=int)
    p.add_argument("max_age_days", type=int)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("prune-artifacts")
    p.add_argument("artifact_dir")
    p.add_argument("job_key")
    p.add_argument("current_build")
    p.add_argument("retain_n", type=int)
    p.add_argument("max_age_days", type=int)
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "stages-json":
        validate_boards()
        print(json.dumps(STAGES))
        return 0
    if args.cmd == "ci-params-json":
        print(json.dumps(ci_params_payload()))
        return 0
    if args.cmd == "retention-json":
        print(json.dumps(RETENTION))
        return 0
    if args.cmd == "hw-shards-json":
        validate_boards()
        print(json.dumps(hw_shards()))
        return 0
    if args.cmd == "stash-name":
        print(stash_name(args.stage, args.shards, args.shard_id))
        return 0
    if args.cmd == "job-key":
        print(job_key(args.name))
        return 0
    if args.cmd == "full-matrix-json":
        enabled = [p for p in args.enabled.split(",") if p]
        print(json.dumps(full_matrix_status(enabled, args.stage_filter)))
        return 0
    if args.cmd == "prepare":
        return prepare_timing_snapshot(args.master, args.snapshot, args.seed)
    if args.cmd == "regen-seed":
        return regen_seed(args.master, args.out)
    if args.cmd == "summarize":
        return summarize_timings(args.reports_dir)
    if args.cmd == "update":
        return update_master(
            args.reports,
            args.master,
            args.out,
            promote=args.promote,
            metadata={
                "job": args.job,
                "build": args.build,
                "full_run": args.full_run,
                "stage_filter": args.stage_filter,
            },
        )
    if args.cmd == "merge-maps":
        return merge_maps(args.reports_dir)
    if args.cmd == "resolve-sw-zips":
        tests = [t for t in args.tests.split(",") if t]
        boards = [b for b in args.boards.split(",") if b]
        result = resolve_sw_zips(args.artifact_dir, args.job_key, tests, boards, args.sw_build_dir)
        print(json.dumps(result))
        return 0
    if args.cmd == "prune-tmp":
        return prune_tmp(
            args.nfs_base,
            args.job_key,
            args.current_build,
            args.retain_n,
            args.max_age_days,
            args.dry_run,
        )
    if args.cmd == "prune-images":
        return prune_images(
            args.shared_dir,
            args.job_key,
            args.current_build,
            args.retain_n,
            args.max_age_days,
            args.dry_run,
        )
    if args.cmd == "prune-artifacts":
        return prune_artifacts(
            args.artifact_dir,
            args.job_key,
            args.current_build,
            args.retain_n,
            args.max_age_days,
            args.dry_run,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
