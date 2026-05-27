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
pytest_plugins = ["pytester"]


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


def test_assign_groups_rejects_zero_or_negative_num_shards():
    # Defence-in-depth: a future direct caller (off the conftest path that
    # validates separately) must get a loud error rather than an IndexError
    # when shard_load[shard] is written.
    with pytest.raises(ValueError, match="num_shards must be >= 1"):
        ci_sharding.assign_groups_to_shards(["a"], 0)
    with pytest.raises(ValueError, match="num_shards must be >= 1"):
        ci_sharding.assign_groups_to_shards(["a"], -1)


def test_assign_groups_rejects_out_of_range_pin():
    with pytest.raises(ValueError, match=r"pin for 'k' out of range"):
        ci_sharding.assign_groups_to_shards(["k"], 2, pins={"k": 2})
    with pytest.raises(ValueError, match=r"pin for 'k' out of range"):
        ci_sharding.assign_groups_to_shards(["k"], 2, pins={"k": -1})


def test_assign_groups_rejects_non_int_pin():
    with pytest.raises(ValueError, match="must be an int"):
        ci_sharding.assign_groups_to_shards(["k"], 2, pins={"k": "zero"})


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

    # promote=True exercises the persistent-master path, the test name
    # describes the persisted file, not just the preview.
    ci_sharding.update_master(str(reports), str(master), str(out), promote=True)

    persisted = json.loads(master.read_text())
    merged = json.loads(out.read_text())
    assert persisted["groups"]["seen"]["seconds"] == 3.5
    assert persisted["groups"]["seen"]["count"] == 4
    assert persisted["groups"]["unseen"]["seconds"] == 7.0
    assert merged["groups"]["seen"]["seconds"] == 3.5
    assert merged["groups"]["unseen"]["seconds"] == 7.0


def test_update_master_skips_persistent_update_when_not_promoted(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(master, {"version": 1, "groups": {"seen": {"seconds": 1.0, "count": 1}}})
    write_json(
        reports / "stage.timings.json",
        {
            "stash": "stage",
            "metadata": {"job": "job", "build": "12", "stage": "Stage"},
            "groups": [{"name": "seen", "seconds": 3.5, "count": 4}],
        },
    )

    ci_sharding.update_master(
        str(reports),
        str(master),
        str(out),
        promote=False,
        metadata={"job": "job", "build": "12", "full_run": False, "stage_filter": "BNN"},
    )

    persisted = json.loads(master.read_text())
    merged = json.loads(out.read_text())
    assert persisted["groups"]["seen"]["seconds"] == 1.0
    assert merged["groups"]["seen"]["seconds"] == 3.5
    assert merged["last_update"]["promoted"] is False
    assert merged["last_update"]["stage_filter"] == "BNN"


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


def test_regen_seed_flattens_master_groups_and_skips_zero_seconds(tmp_path):
    master = tmp_path / "master.json"
    out = tmp_path / "seed.json"
    write_json(
        master,
        {
            "version": 1,
            "groups": {
                "alpha": {"seconds": 12.345, "count": 1},
                "beta": {"seconds": 0.0, "count": 1},
                "gamma": {"seconds": 7.0, "count": 2},
                "delta": "not a number",
            },
        },
    )

    ci_sharding.regen_seed(str(master), str(out))

    data = json.loads(out.read_text())
    assert data["_comment"].startswith("Seed per-group wall seconds")
    assert data["groups"] == {"alpha": 12.345, "gamma": 7.0}
    assert list(data["groups"].keys()) == ["alpha", "gamma"]


def test_regen_seed_rejects_non_object_master(tmp_path):
    master = tmp_path / "master.json"
    out = tmp_path / "seed.json"
    master.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="not a JSON object"):
        ci_sharding.regen_seed(str(master), str(out))


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


def test_prune_images_keeps_current_build_and_newest(tmp_path):
    shared = tmp_path / "shared"
    parent = shared / "job"
    for build in ("1", "2", "3", "4"):
        path = parent / build
        path.mkdir(parents=True)
        os.utime(str(path), (1, 1))

    ci_sharding.prune_images(str(shared), "job", "5", retain_n=2, max_age_days=0)

    assert not (parent / "1").exists()
    assert not (parent / "2").exists()
    assert (parent / "3").exists()
    assert (parent / "4").exists()


def test_prune_images_skips_when_parent_missing(tmp_path, capsys):
    rc = ci_sharding.prune_images(str(tmp_path / "absent"), "job", "1", 1, 0)
    captured = capsys.readouterr()
    assert rc == 0
    assert "not present, skipping" in captured.out


def _seed_sw_build(artifact_dir, job_key, build, test_type, board, ready=True):
    build_dir = artifact_dir / "ci_runs" / job_key / build
    zip_dir = build_dir / "zips" / test_type
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / ("%s.zip" % board)
    zip_path.write_text("")
    if ready:
        (zip_dir / ("%s.zip.READY" % board)).write_text("")
    return zip_path, build_dir


def test_resolve_sw_zips_picks_newest_ready_per_pair(tmp_path):
    art = tmp_path / "artifacts"
    _seed_sw_build(art, "finn", "10", "bnn_build_full", "U250", ready=False)
    _seed_sw_build(art, "finn", "11", "bnn_build_full", "U250", ready=True)
    _seed_sw_build(art, "finn", "12", "bnn_build_full", "Pynq-Z1", ready=True)

    out = ci_sharding.resolve_sw_zips(
        str(art), "finn", ["bnn_build_full"], ["U250", "Pynq-Z1", "ZCU104"]
    )

    assert out["bnn_build_full"]["U250"]["zip"].endswith("/11/zips/bnn_build_full/U250.zip")
    assert out["bnn_build_full"]["U250"]["swBuildDir"].endswith("/finn/11")
    assert out["bnn_build_full"]["Pynq-Z1"]["zip"].endswith("/12/zips/bnn_build_full/Pynq-Z1.zip")
    assert out["bnn_build_full"]["ZCU104"] == {}


def test_resolve_sw_zips_falls_back_to_older_build_per_board(tmp_path):
    # A new build that only succeeded for Pynq-Z1 must not strand U250 on the
    # older build it last produced a READY for. This is the per-board fallback
    # contract HW relies on.
    art = tmp_path / "artifacts"
    _seed_sw_build(art, "finn", "20", "bnn_build_full", "U250", ready=True)
    _seed_sw_build(art, "finn", "21", "bnn_build_full", "Pynq-Z1", ready=True)

    out = ci_sharding.resolve_sw_zips(str(art), "finn", ["bnn_build_full"], ["U250", "Pynq-Z1"])

    assert out["bnn_build_full"]["U250"]["swBuildDir"].endswith("/finn/20")
    assert out["bnn_build_full"]["Pynq-Z1"]["swBuildDir"].endswith("/finn/21")
    assert out["bnn_build_full"]["U250"]["fallback"] is True
    assert out["bnn_build_full"]["Pynq-Z1"]["fallback"] is False
    assert out["bnn_build_full"]["U250"]["latestSwBuild"] == "21"


def test_resolve_sw_zips_marks_stale_when_newest_build_has_no_ready(tmp_path):
    art = tmp_path / "artifacts"
    _seed_sw_build(art, "finn", "20", "bnn_build_full", "U250", ready=True)
    # Build 21 exists but produced no READY zips. HW must know it is falling
    # back to stale SW artefacts rather than treating build 20 as latest.
    (art / "ci_runs" / "finn" / "21").mkdir(parents=True)

    out = ci_sharding.resolve_sw_zips(str(art), "finn", ["bnn_build_full"], ["U250"])

    assert out["bnn_build_full"]["U250"]["swBuildDir"].endswith("/finn/20")
    assert out["bnn_build_full"]["U250"]["latestSwBuild"] == "21"
    assert out["bnn_build_full"]["U250"]["fallback"] is True


def test_resolve_sw_zips_honours_explicit_sw_build_dir(tmp_path):
    art = tmp_path / "artifacts"
    _seed_sw_build(art, "finn", "5", "bnn_build_full", "U250", ready=True)
    _seed_sw_build(art, "finn", "6", "bnn_build_full", "U250", ready=True)
    explicit = art / "ci_runs" / "finn" / "5"
    out = ci_sharding.resolve_sw_zips(
        str(art), "finn", ["bnn_build_full"], ["U250"], sw_build_dir=str(explicit)
    )
    assert out["bnn_build_full"]["U250"]["swBuildDir"] == str(explicit)
    assert out["bnn_build_full"]["U250"]["fallback"] is False


def test_resolve_sw_zips_returns_empty_when_no_ready_anywhere(tmp_path):
    art = tmp_path / "artifacts"
    _seed_sw_build(art, "finn", "1", "bnn_build_full", "U250", ready=False)

    out = ci_sharding.resolve_sw_zips(str(art), "finn", ["bnn_build_full"], ["U250"])
    assert out == {"bnn_build_full": {"U250": {}}}


def test_resolve_sw_zips_skips_when_job_root_missing(tmp_path):
    out = ci_sharding.resolve_sw_zips(
        str(tmp_path / "absent"), "finn", ["bnn_build_full"], ["U250"]
    )
    assert out == {"bnn_build_full": {"U250": {}}}


def test_prune_artifacts_keeps_current_build_and_newest(tmp_path):
    artifact_dir = tmp_path / "art"
    parent = artifact_dir / "ci_runs" / "job"
    for build in ("1", "2", "3", "4"):
        path = parent / build
        path.mkdir(parents=True)
        os.utime(str(path), (1, 1))

    ci_sharding.prune_artifacts(str(artifact_dir), "job", "5", retain_n=2, max_age_days=0)

    assert not (parent / "1").exists()
    assert not (parent / "2").exists()
    assert (parent / "3").exists()
    assert (parent / "4").exists()


def test_prune_artifacts_skips_when_parent_missing(tmp_path, capsys):
    rc = ci_sharding.prune_artifacts(str(tmp_path / "absent"), "job", "1", 1, 0)
    captured = capsys.readouterr()
    assert rc == 0
    assert "not present, skipping" in captured.out


def test_is_full_matrix_run_true_on_canonical_stages_with_every_param_ticked():
    # The canonical STAGES has no skipWhen rows, so ticking every CI param
    # is sufficient to trigger auto-promote on a successful build. Any
    # missing param or a non-empty STAGE_FILTER must turn it off.
    required = ci_sharding.ci_param_names()
    assert ci_sharding.is_full_matrix_run(required)
    assert not ci_sharding.is_full_matrix_run(required[:-1])
    assert not ci_sharding.is_full_matrix_run([])
    assert not ci_sharding.is_full_matrix_run(required, stage_filter="BNN")
    payload = ci_sharding.full_matrix_status(required, stage_filter="")
    assert payload["full"] is True
    assert payload["required"] == required


def test_ci_params_payload_uses_explicit_smoke_constant():
    payload = ci_sharding.ci_params_payload()
    assert payload["params"] == ci_sharding.ci_param_names()
    assert payload["smoke"] == list(ci_sharding.SMOKE_PARAMS)


def test_smoke_is_immune_to_stages_reorder():
    # Reorder a synthetic STAGES. Smoke must still resolve to SMOKE_PARAMS
    # rather than "whatever happens to be index 0", which is the regression
    # the explicit SMOKE_PARAMS constant prevents.
    forward = [
        {"param": "sanity", "stage": "S", "marker": "s", "shards": 1, "workers": 1},
        {"param": "end2end", "stage": "E", "marker": "e", "shards": 1, "workers": 1},
    ]
    reversed_stages = list(reversed(forward))
    fwd = ci_sharding.ci_params_payload(forward, smoke_params=("sanity",))
    rev = ci_sharding.ci_params_payload(reversed_stages, smoke_params=("sanity",))
    assert fwd["smoke"] == ["sanity"]
    assert rev["smoke"] == ["sanity"]
    # params order naturally follows STAGES order, smoke does not.
    assert fwd["params"] != rev["params"]


def test_ci_params_payload_drops_smoke_entries_not_in_stages():
    # A SMOKE_PARAMS entry referencing a no-longer-existing param is dropped
    # silently, so a stale constant cannot enable nothing-at-all rather than
    # the intended subset.
    stages = [{"param": "sanity", "stage": "S", "marker": "s", "shards": 1, "workers": 1}]
    payload = ci_sharding.ci_params_payload(stages, smoke_params=("sanity", "vanished"))
    assert payload["smoke"] == ["sanity"]


def test_is_full_matrix_run_short_circuits_on_stage_filter():
    assert not ci_sharding.is_full_matrix_run(
        ["sanity", "fpgadataflow", "end2end"], stage_filter="BNN"
    )


def test_is_full_matrix_run_true_when_no_skipwhen_rows_block():
    # Synthetic STAGES with no skipWhen: is_full_matrix_run is True when
    # every distinct param is enabled, and False otherwise.
    custom_stages = [
        {"param": "alpha", "stage": "A", "marker": "a", "shards": 1, "workers": 1},
        {"param": "beta", "stage": "B", "marker": "b", "shards": 1, "workers": 1},
    ]
    assert ci_sharding.is_full_matrix_run(["alpha", "beta"], stages=custom_stages)
    assert not ci_sharding.is_full_matrix_run(["alpha"], stages=custom_stages)
    assert ci_sharding.ci_param_names(custom_stages) == ["alpha", "beta"]
    assert not ci_sharding.is_full_matrix_run([], stages=[])


def test_rows_that_would_run_excludes_skipwhen_rows():
    custom_stages = [
        {
            "param": "sanity",
            "stage": "Sanity",
            "marker": "s",
            "shards": 1,
            "workers": 1,
            "skipWhen": "end2end",
        },
        {"param": "end2end", "stage": "End2end", "marker": "e", "shards": 1, "workers": 1},
    ]
    # Only sanity ticked -> the sanity row runs.
    rows = ci_sharding.rows_that_would_run(["sanity"], stages=custom_stages)
    assert [r["stage"] for r in rows] == ["Sanity"]
    # Both ticked -> sanity is skipped by skipWhen, only End2end runs.
    rows = ci_sharding.rows_that_would_run(["sanity", "end2end"], stages=custom_stages)
    assert [r["stage"] for r in rows] == ["End2end"]
    # Neither ticked -> no rows run.
    assert ci_sharding.rows_that_would_run([], stages=custom_stages) == []


def test_update_master_records_promote_failure_in_preview(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(master, {"version": 1, "groups": {}})
    write_json(
        reports / "stage.timings.json",
        {
            "stash": "stage",
            "metadata": {"job": "j", "build": "1", "stage": "Stage"},
            "groups": [{"name": "seen", "seconds": 1.0, "count": 1}],
        },
    )

    def boom(*_a, **_k):
        raise IOError("simulated NFS write failure")

    monkeypatch.setattr(ci_sharding, "locked_update", boom)

    # Promote failure is advisory: rc stays 0 so per-build CI doesn't flap on
    # routine NFS hiccups. The preview's last_update.promoted records it.
    rc = ci_sharding.update_master(
        str(reports),
        str(master),
        str(out),
        promote=True,
        metadata={"job": "j", "build": "1", "full_run": True, "stage_filter": ""},
    )

    preview = json.loads(out.read_text())
    assert rc == 0
    assert preview["last_update"]["promoted"] is False
    assert preview["groups"]["seen"]["seconds"] == 1.0


def test_read_json_warns_on_corrupt_file(tmp_path, capsys):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{")

    assert ci_sharding.read_json(str(corrupt), default={"fallback": True}) == {"fallback": True}

    captured = capsys.readouterr()
    # FileNotFoundError stays silent (idle state), corrupt files warn loudly
    # so a broken master state doesn't silently degrade sharding to round-robin.
    assert "ci_sharding read_json" in captured.err
    assert str(corrupt) in captured.err


def test_read_json_missing_file_is_silent(tmp_path, capsys):
    missing = tmp_path / "absent.json"

    assert ci_sharding.read_json(str(missing), default=None) is None

    captured = capsys.readouterr()
    assert captured.err == ""


def test_prune_numeric_builds_dry_run_matches_real_run_count(tmp_path):
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("1", "2", "3", "4"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    dry = ci_sharding.prune_numeric_builds(
        str(parent),
        current_build="5",
        retain_n=1,
        max_age_days=0,
        dry_run=True,
        tag="t",
    )
    assert dry == 3
    assert sorted(p.name for p in parent.iterdir()) == ["1", "2", "3", "4"]

    real = ci_sharding.prune_numeric_builds(
        str(parent),
        current_build="5",
        retain_n=1,
        max_age_days=0,
        dry_run=False,
        tag="t",
    )
    assert real == 3
    assert sorted(p.name for p in parent.iterdir()) == ["4"]


def test_prune_numeric_builds_tag_is_keyword_only():
    import inspect

    sig = inspect.signature(ci_sharding.prune_numeric_builds)
    assert sig.parameters["tag"].kind is inspect.Parameter.KEYWORD_ONLY


def test_prune_numeric_builds_rejects_non_numeric_current(tmp_path):
    # off-Jenkins CLI invocations or a broken BUILD_NUMBER env must not
    # silently degrade retention to "newest N" by passing a string the
    # numeric-only sibling filter can never match.
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("1", "2"):
        (parent / build).mkdir()
    with pytest.raises(ValueError, match="current_build must be an integer-like string"):
        ci_sharding.prune_numeric_builds(
            str(parent),
            current_build="not-a-number",
            retain_n=1,
            max_age_days=0,
            dry_run=True,
            tag="t",
        )
    with pytest.raises(ValueError, match="current_build must be an integer-like string"):
        ci_sharding.prune_numeric_builds(
            str(parent),
            current_build=None,
            retain_n=1,
            max_age_days=0,
            dry_run=True,
            tag="t",
        )


def test_prune_tmp_validates_current_build_at_boundary(tmp_path):
    # The boundary check (prune-tmp/images/artifacts) fires before the
    # per-tree loop, so a bad BUILD_NUMBER fails loudly rather than
    # silently aborting on the first tree after partial work.
    with pytest.raises(ValueError, match="prune-tmp: current_build must be"):
        ci_sharding.prune_tmp(str(tmp_path), "job", "not-a-number", 1, 0)


def test_prune_images_validates_current_build_at_boundary(tmp_path):
    with pytest.raises(ValueError, match="prune-images: current_build must be"):
        ci_sharding.prune_images(str(tmp_path), "job", None, 1, 0)


def test_prune_artifacts_validates_current_build_at_boundary(tmp_path):
    with pytest.raises(ValueError, match="prune-artifacts: current_build must be"):
        ci_sharding.prune_artifacts(str(tmp_path), "job", "x.y", 1, 0)


def test_boards_has_metadata_for_every_zip_artifact_board():
    # The HW pipeline derives HW_SHARDS from BOARDS via hw-shards-json, so a
    # STAGES row that mentions a board missing from BOARDS would yield a zip
    # nobody can run. validate_boards() must reject that asymmetry.
    ci_sharding.validate_boards()


def test_validate_boards_rejects_orphan_zipartifact_board():
    custom_stages = [
        {
            "param": "p",
            "stage": "X",
            "marker": "x",
            "shards": 1,
            "workers": 1,
            "zipArtifacts": {"hwTestType": "t", "boards": ["FakeBoard"]},
        },
    ]
    with pytest.raises(ValueError, match="FakeBoard"):
        ci_sharding.validate_boards(stages=custom_stages, boards={})


def test_hw_shards_flattens_boards_dict_for_groovy():
    rows = ci_sharding.hw_shards()
    boards = [r["board"] for r in rows]
    assert "U250" in boards
    u250 = next(r for r in rows if r["board"] == "U250")
    assert u250["agentLabel"] == "finn-u250"
    assert u250["credentialsId"] is None
    assert u250["restartPrep"] is False
    # ordering matches BOARDS insertion order so HW parallel-branch order
    # is stable across builds
    assert boards == list(ci_sharding.BOARDS.keys())


def test_job_key_strips_dot_traversal():
    # the per-build prune paths are os.path.join(parent, job_key(...)/...) so
    # JOB_NAME=".." or all-dot inputs must not survive the sanitiser
    assert ci_sharding.job_key("..") == "job"
    assert ci_sharding.job_key(".") == "job"
    assert ci_sharding.job_key("...") == "job"
    assert ci_sharding.job_key("") == "job"
    assert ci_sharding.job_key(None) == "job"
    # leading-/trailing-dot inputs lose just the offending dots
    assert ci_sharding.job_key(".foo") == "foo"
    assert ci_sharding.job_key("foo.") == "foo"
    # legitimate names with internal dots survive
    assert ci_sharding.job_key("finn.dev") == "finn.dev"
    # the path-separator class still gets replaced with '_' as before
    assert ci_sharding.job_key("../etc") == "_etc"


def test_prune_numeric_builds_tolerates_concurrent_delete(tmp_path, monkeypatch):
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("1", "2", "3"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    real_rmtree = ci_sharding.shutil.rmtree
    state = {"first": True}

    def flaky_rmtree(path, *args, **kwargs):
        # simulate another CI run pruning '1' between our listdir and rmtree
        if state["first"]:
            state["first"] = False
            raise FileNotFoundError(path)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(ci_sharding.shutil, "rmtree", flaky_rmtree)
    matched = ci_sharding.prune_numeric_builds(
        str(parent),
        current_build="5",
        retain_n=1,
        max_age_days=0,
        dry_run=False,
        tag="t",
    )
    assert matched == 2
    # build '3' is kept (retain_n=1, newest); '2' got rmtreed for real;
    # '1' was the simulated race victim and we tolerated it
    surviving = sorted(p.name for p in parent.iterdir())
    assert "3" in surviving
    assert "2" not in surviving


def test_prune_numeric_builds_tolerates_concurrent_delete_in_age_check(tmp_path, monkeypatch):
    parent = tmp_path / "p"
    parent.mkdir()
    # retain_n=1 keeps the newest ('3'); '1' and '2' are both deletion
    # candidates. age cutoff is in the past so both qualify on mtime.
    for build in ("1", "2", "3"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    real_getmtime = ci_sharding.os.path.getmtime

    def flaky_getmtime(path):
        if path.endswith("/1"):
            raise FileNotFoundError(path)
        return real_getmtime(path)

    monkeypatch.setattr(ci_sharding.os.path, "getmtime", flaky_getmtime)
    matched = ci_sharding.prune_numeric_builds(
        str(parent),
        current_build="9",
        retain_n=1,
        max_age_days=7,
        dry_run=False,
        tag="t",
    )
    # '1' raised FileNotFoundError during the age probe so the loop treats it
    # as already-pruned and does not count it. '2' was processed normally and
    # deleted. The point is that the FileNotFoundError on '1' did not abort
    # the loop and leave '2' behind.
    assert matched == 1
    assert "2" not in [p.name for p in parent.iterdir()]
    assert "3" in [p.name for p in parent.iterdir()]


def test_locked_update_backs_up_corrupt_master(tmp_path):
    master = tmp_path / "master.json"
    master.write_text("{ this is not json")

    updated = ci_sharding.locked_update(
        str(master), lambda cur: {"version": 1, "groups": {"k": {"seconds": 1.0}}}
    )

    assert updated["groups"]["k"]["seconds"] == 1.0
    fresh = json.loads(master.read_text())
    assert fresh["groups"]["k"]["seconds"] == 1.0

    backups = sorted(p.name for p in tmp_path.iterdir() if ".corrupt-" in p.name)
    assert len(backups) == 1
    assert backups[0].startswith("master.json.corrupt-")
    assert (tmp_path / backups[0]).read_text() == "{ this is not json"


def test_backup_if_corrupt_caps_history(tmp_path):
    master = tmp_path / "master.json"
    # Pre-populate with more old backups than the retention cap.
    older = ci_sharding.CORRUPT_BACKUP_RETAIN + 3
    for i in range(older):
        (tmp_path / ("master.json.corrupt-%d" % (1000 + i))).write_text("old %d" % i)
    master.write_text("{ this is not json")

    ci_sharding.locked_update(str(master), lambda cur: {"version": 1, "groups": {}})

    backups = sorted(p.name for p in tmp_path.iterdir() if ".corrupt-" in p.name)
    # Newest cap survivors plus the freshly-made one from this run.
    assert len(backups) == ci_sharding.CORRUPT_BACKUP_RETAIN
    # The very oldest ones were pruned.
    assert "master.json.corrupt-1000" not in backups
    # The freshest pre-existing backup survived.
    assert any("corrupt-%d" % (1000 + older - 1) in b for b in backups)


def test_locked_update_does_not_backup_empty_master(tmp_path):
    master = tmp_path / "master.json"
    master.write_text("")

    ci_sharding.locked_update(str(master), lambda cur: {"version": 1, "groups": {}})

    # zero-byte master is the idle-create case, not corruption, so no backup
    backups = [p for p in tmp_path.iterdir() if ".corrupt-" in p.name]
    assert backups == []


def test_update_master_no_master_path_writes_preview(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    out = reports / "ci_timings_master.json"
    write_json(
        reports / "stage.timings.json",
        {
            "stash": "stage",
            "metadata": {"job": "j", "build": "1", "stage": "Stage"},
            "groups": [{"name": "seen", "seconds": 1.0, "count": 1}],
        },
    )

    # promote=True is intentional: without a master path there is nothing to
    # promote into, so the preview's last_update.promoted must come out
    # False and the status string ``no-master`` records the cause.
    rc = ci_sharding.update_master(
        str(reports),
        master_path=None,
        out_path=str(out),
        promote=True,
        metadata={"job": "j", "build": "1", "full_run": True, "stage_filter": ""},
    )

    preview = json.loads(out.read_text())
    assert rc == 0
    assert preview["groups"]["seen"]["seconds"] == 1.0
    assert preview["last_update"]["promoted"] is False
    assert preview["last_update"]["observed_groups"] == 1


def test_pytest_plugin_writes_timings_for_successful_sharded_run(pytester):
    plugin_path = os.path.join(REPO_ROOT, "tests", "conftest.py")
    pytester.makeconftest(
        """
import importlib.util

spec = importlib.util.spec_from_file_location("finn_ci_conftest", {plugin_path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for name in (
    "pytest_addoption",
    "pytest_configure",
    "pytest_collection_modifyitems",
    "pytest_sessionstart",
    "pytest_collection_finish",
    "pytest_runtest_logreport",
    "pytest_sessionfinish",
):
    globals()[name] = getattr(mod, name)
""".format(
            plugin_path=plugin_path
        )
    )
    pytester.makepyfile(
        test_sample="""
def test_ok():
    assert True
"""
    )

    result = pytester.runpytest(
        "--num-shards=1",
        "--shard-id=0",
        "--junitxml=stage.xml",
        "-q",
    )

    result.assert_outcomes(passed=1)
    data = json.loads((pytester.path / "stage.timings.json").read_text())
    assert data["stash"] == "stage"
    assert data["shard"] == {"num": 1, "id": 0}
    assert data["groups"][0]["name"].endswith("test_sample.py::test_ok")
