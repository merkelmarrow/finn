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


STAGES = [
    {
        "param": "sanity",
        "stage": "Sanity - Build Hardware",
        "marker": "sanity_bnn",
        "shards": 1,
        "workers": 1,
        "skipWhen": "end2end",
        "zipBoards": ["U250", "Pynq-Z1", "ZCU104", "KV260_SOM"],
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
        "zipBoards": ["U250"],
    },
    {
        "param": "end2end",
        "stage": "BNN Pynq-Z1",
        "marker": "bnn_pynq",
        "shards": 3,
        "workers": 2,
        "distMode": "loadgroup",
        "zipBoards": ["Pynq-Z1"],
    },
    {
        "param": "end2end",
        "stage": "BNN ZCU104",
        "marker": "bnn_zcu104",
        "shards": 2,
        "workers": 4,
        "distMode": "loadgroup",
        "zipBoards": ["ZCU104"],
    },
    {
        "param": "end2end",
        "stage": "BNN KV260",
        "marker": "bnn_kv260",
        "shards": 2,
        "workers": 2,
        "distMode": "loadgroup",
        "zipBoards": ["KV260_SOM"],
    },
]


def stash_name(stage, shards, shard_id):
    """Mirror ``shardStashName()`` in ``docker/jenkins/Jenkinsfile``."""
    base = re.sub(r"^_|_$", "", re.sub(r"[^a-z0-9]+", "_", stage.lower()))
    return base if shards <= 1 else "%s_%d" % (base, shard_id + 1)


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
    except (OSError, ValueError):
        return default


def write_json_atomic(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=parent or ".")
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
    known = sorted(w for w in weights.values() if w > 0.0)
    fallback = known[len(known) // 2] if known else 1.0
    return fallback


def assign_groups_to_shards(group_keys, num_shards, weights=None, pins=None):
    """Assign group keys to shard ids.

    Pins win first. If there is useful timing signal, unpinned groups are
    assigned by LPT-greedy bin packing. Otherwise they are round-robin over
    sorted group keys so fallback placement is reproducible and balanced by
    group count.
    """
    group_keys = sorted(str(k) for k in group_keys)
    weights = weights or {}
    pins = pins or {}
    assignment = {}
    source = {}
    shard_load = [0.0] * max(1, int(num_shards))
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
            print("!! could not parse %s" % path)
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
            "ci_sharding summarize: one or more shards exceeded %.1fx family median; "
            "future builds will refresh the timing master with this run's data" % SLOW_FACTOR
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


def locked_update(master_path, update_fn):
    if not master_path:
        return update_fn({})
    parent = os.path.dirname(os.path.abspath(master_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    lock_path = master_path + ".lock"
    with open(lock_path, "a+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = read_json(master_path, default={})
        updated = update_fn(current)
        write_json_atomic(master_path, updated)
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return updated


def update_master(reports_dir, master_path=None, out_path=None):
    observed = observed_groups_from_reports(reports_dir)

    def apply(current):
        master = normalise_master(current)
        master["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        master["groups"].update(observed)
        return master

    if master_path:
        try:
            master = locked_update(master_path, apply)
        except Exception as exc:
            print("ci_sharding update: persistent master update failed: %s" % exc)
            master = apply(read_json(master_path, default={}))
    else:
        master = apply({})
    if out_path:
        write_json_atomic(out_path, master)
    print(
        "ci_sharding update: merged %d observed group(s); master has %d group(s)"
        % (len(observed), len(master.get("groups", {})))
    )
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


def prune_tmp(nfs_base, job_key, current_build, retain_n, max_age_days, dry_run=False):
    retain_n = int(retain_n)
    max_age_days = int(max_age_days)
    if retain_n < 1:
        raise ValueError("retain_n must be >= 1")
    if not os.path.isdir(nfs_base):
        print("ci_sharding prune-tmp: nfs root %s not present, skipping" % nfs_base)
        return 0
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    trees = 0
    for name in sorted(os.listdir(nfs_base)):
        base = os.path.join(nfs_base, name, "workspace", "tmp", "ci_runs", job_key)
        if not os.path.isdir(base):
            continue
        trees += 1
        nums = sorted((d for d in os.listdir(base) if d.isdigit()), key=int)
        keep = set(nums[-retain_n:])
        keep.add(str(current_build))
        for num in nums:
            if num in keep:
                continue
            path = os.path.join(base, num)
            if max_age_days > 0 and os.path.getmtime(path) >= cutoff:
                continue
            if dry_run:
                print("ci_sharding prune-tmp: would delete %s" % path)
            else:
                print("ci_sharding prune-tmp: deleting %s" % path)
                shutil.rmtree(path)
    print(
        "ci_sharding prune-tmp: done (nfs_base=%s job_key=%s current=%s retain_n=%s "
        "max_age_days=%s dry_run=%s trees=%d)"
        % (nfs_base, job_key, current_build, retain_n, max_age_days, int(dry_run), trees)
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("stages-json")

    p = sub.add_parser("stash-name")
    p.add_argument("stage")
    p.add_argument("shards", type=int)
    p.add_argument("shard_id", type=int)

    p = sub.add_parser("prepare")
    p.add_argument("--master", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--seed", default=DEFAULT_TIMINGS_SEED)

    p = sub.add_parser("summarize")
    p.add_argument("reports_dir")

    p = sub.add_parser("update")
    p.add_argument("--reports", required=True)
    p.add_argument("--master", default="")
    p.add_argument("--out", required=True)

    p = sub.add_parser("merge-maps")
    p.add_argument("reports_dir")

    p = sub.add_parser("prune-tmp")
    p.add_argument("nfs_base")
    p.add_argument("job_key")
    p.add_argument("current_build")
    p.add_argument("retain_n", type=int)
    p.add_argument("max_age_days", type=int)
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "stages-json":
        print(json.dumps(STAGES))
        return 0
    if args.cmd == "stash-name":
        print(stash_name(args.stage, args.shards, args.shard_id))
        return 0
    if args.cmd == "prepare":
        return prepare_timing_snapshot(args.master, args.snapshot, args.seed)
    if args.cmd == "summarize":
        return summarize_timings(args.reports_dir)
    if args.cmd == "update":
        return update_master(args.reports, args.master, args.out)
    if args.cmd == "merge-maps":
        return merge_maps(args.reports_dir)
    if args.cmd == "prune-tmp":
        return prune_tmp(
            args.nfs_base,
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
