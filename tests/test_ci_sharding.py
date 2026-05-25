# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JENKINS_DIR = os.path.join(REPO_ROOT, "docker", "jenkins")
if JENKINS_DIR not in sys.path:
    sys.path.insert(0, JENKINS_DIR)

import ci_sharding  # noqa: E402

pytestmark = pytest.mark.util


def write_json(path, payload):
    path.write_text(json.dumps(payload))


def test_load_group_weights_ignores_missing_and_corrupt(tmp_path):
    missing = tmp_path / "missing.json"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{")

    assert ci_sharding.load_group_weights(str(missing)) == {}
    assert ci_sharding.load_group_weights(str(corrupt)) == {}


def test_assign_groups_round_robin_without_timing_signal():
    assignment, source, shard_load, fallback = ci_sharding.assign_groups_to_shards(
        ["b", "a", "c"], 2, weights={}
    )

    assert assignment == {"a": 0, "b": 1, "c": 0}
    assert source == {"a": "round_robin", "b": "round_robin", "c": "round_robin"}
    assert shard_load == [2.0, 1.0]
    assert fallback == 1.0


def test_assign_groups_pins_override_timing_weights():
    assignment, source, shard_load, fallback = ci_sharding.assign_groups_to_shards(
        ["slow", "pinned"], 2, weights={"slow": 100.0, "pinned": 1.0}, pins={"pinned": 0}
    )

    assert assignment["pinned"] == 0
    assert assignment["slow"] == 1
    assert source["pinned"] == "pinned"
    assert source["slow"] == "known"
    assert fallback == 100.0


def test_assign_groups_unknown_pins_use_fallback_weight():
    assignment, source, shard_load, fallback = ci_sharding.assign_groups_to_shards(
        ["known", "pinned"], 2, weights={"known": 9.0}, pins={"pinned": 0}
    )

    assert assignment["pinned"] == 0
    assert source["pinned"] == "pinned"
    assert fallback == 9.0
    assert shard_load[0] == 9.0


def test_update_master_preserves_unseen_and_replaces_seen(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "version": 1,
            "groups": {
                "seen": {"seconds": 1.0, "count": 1},
                "unseen": {"seconds": 7.0, "count": 2},
            },
        },
    )
    write_json(
        reports / "stage.timings.json",
        {
            "stash": "stage",
            "metadata": {"job": "job", "build": "12", "stage": "Stage"},
            "groups": [{"name": "seen", "seconds": 3.5, "count": 4}],
        },
    )

    ci_sharding.update_master(str(reports), str(master), str(out))

    merged = json.loads(out.read_text())
    assert merged["groups"]["seen"]["seconds"] == 3.5
    assert merged["groups"]["seen"]["count"] == 4
    assert merged["groups"]["unseen"]["seconds"] == 7.0


def test_merge_maps_writes_searchable_text(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    write_json(
        reports / "stage.shardmap.json",
        [
            {
                "nodeid": "tests/foo.py::test_bar",
                "stage": "Stage",
                "stash": "stage",
                "shard_id": 0,
                "num_shards": 2,
                "group": "grp",
                "weight_s": 1.25,
                "source": "known",
            }
        ],
    )

    ci_sharding.merge_maps(str(reports))

    text = (reports / "shard_map.txt").read_text()
    assert "nodeid=tests/foo.py::test_bar" in text
    assert "stage=Stage" in text
    assert "shard=1/2" in text
    assert "source=known" in text


def test_prepare_timing_snapshot_uses_seed_when_master_missing(tmp_path):
    seed = tmp_path / "seed.json"
    snapshot = tmp_path / "snapshot.json"
    write_json(seed, {"version": 1, "groups": {"slow": 12.0}})

    ci_sharding.prepare_timing_snapshot(
        str(tmp_path / "missing-master.json"), str(snapshot), str(seed)
    )

    data = json.loads(snapshot.read_text())
    assert data["groups"]["slow"] == 12.0


def test_prune_tmp_keeps_newest_numeric_builds(tmp_path):
    root = tmp_path / "nfs"
    base = root / "agent" / "workspace" / "tmp" / "ci_runs" / "job"
    for build in ("98", "99", "100"):
        path = base / build
        path.mkdir(parents=True)
        os.utime(str(path), (1, 1))

    ci_sharding.prune_tmp(str(root), "job", "101", retain_n=1, max_age_days=0)

    assert not (base / "98").exists()
    assert not (base / "99").exists()
    assert (base / "100").exists()
