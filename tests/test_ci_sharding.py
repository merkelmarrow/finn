# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import os
import re
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
                "seen": {"samples": [1.0], "count": 1},
                "unseen": {"samples": [7.0], "count": 2},
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

    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)

    persisted = json.loads(master.read_text())
    merged = json.loads(out.read_text())
    # seen's prior median is 1.0, observed 3.5 -- ratio 3.5 is within [0.25, 4.0]
    # so it's accepted and appended to samples.
    assert persisted["groups"]["seen"]["samples"] == [1.0, 3.5]
    assert persisted["groups"]["unseen"]["samples"] == [7.0]
    assert merged["groups"]["seen"]["samples"] == [1.0, 3.5]
    assert merged["last_update"]["accepted"] == 1
    assert merged["last_update"]["rejected"] == 0


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


def test_prepare_timing_snapshot_empty_when_master_missing(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    ci_sharding.prepare_timing_snapshot(str(tmp_path / "missing-master.json"), str(snapshot))
    data = json.loads(snapshot.read_text())
    assert data["groups"] == {}
    assert data["build_seq"] == 0


def test_prepare_timing_snapshot_copies_master_when_present(tmp_path):
    master = tmp_path / "master.json"
    snapshot = tmp_path / "snapshot.json"
    write_json(master, {"version": 1, "groups": {"slow": {"samples": [12.0]}}})

    ci_sharding.prepare_timing_snapshot(str(master), str(snapshot))

    data = json.loads(snapshot.read_text())
    assert data["groups"]["slow"]["samples"] == [12.0]


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


def test_enabled_params_sanity_runs_only_sanity_rows():
    assert ci_sharding.enabled_params_for_choice("sanity") == ["sanity"]


def test_enabled_params_full_returns_every_distinct_param_in_stages_order():
    assert ci_sharding.enabled_params_for_choice("full") == ci_sharding.ci_param_names()


def test_enabled_params_bare_name_returns_that_one():
    assert ci_sharding.enabled_params_for_choice("fpgadataflow") == ["fpgadataflow"]
    assert ci_sharding.enabled_params_for_choice("end2end") == ["end2end"]


def test_jenkins_stage_choices_list_is_sanity_full_then_bare_params():
    assert ci_sharding.jenkins_stage_choices() == ["sanity", "full", "fpgadataflow", "end2end"]


def test_jenkinsfile_stage_choices_match_python_source():
    # Anchor on the STAGES choice block specifically so a future second
    # ``choice(name: 'XYZ', ...)`` cannot match instead. Accept both
    # single- and double-quoted Groovy string literals so a future edit
    # in either style does not silently fall off the regex and read [].
    jenkinsfile = os.path.join(REPO_ROOT, "docker", "jenkins", "Jenkinsfile")
    text = open(jenkinsfile).read()
    match = re.search(
        r"""choice\(\s*name:\s*['"]STAGES['"],\s*choices:\s*\[([^\]]+)\]""",
        text,
    )
    assert match is not None, "could not locate STAGES choice block in Jenkinsfile"
    choices = re.findall(r"""['"]([^'"]+)['"]""", match.group(1))
    expected = ci_sharding.jenkins_stage_choices()
    assert (
        choices == expected
    ), "Jenkinsfile STAGES choices %r drifted from ci_sharding.jenkins_stage_choices() %r" % (
        choices,
        expected,
    )


def test_enabled_params_rejects_unknown_choice_loudly():
    with pytest.raises(ValueError, match="unknown STAGES choice"):
        ci_sharding.enabled_params_for_choice("notarealchoice")


def test_enabled_params_full_on_synthetic_stages_picks_up_new_params():
    # A new CI param (no edits to enabled_params_for_choice required) flows
    # through ``full`` automatically because the function reads ci_param_names.
    custom_stages = [
        {"param": "sanity", "stage": "S", "marker": "s", "shards": 1, "workers": 1},
        {"param": "newthing", "stage": "N", "marker": "n", "shards": 1, "workers": 1},
    ]
    assert ci_sharding.enabled_params_for_choice("full", stages=custom_stages) == [
        "sanity",
        "newthing",
    ]
    assert ci_sharding.enabled_params_for_choice("newthing", stages=custom_stages) == ["newthing"]
    assert ci_sharding.jenkins_stage_choices(stages=custom_stages) == [
        "sanity",
        "full",
        "newthing",
    ]


def test_validate_config_single_invocation_returns_full_payload(capsys):
    # The Jenkinsfile collapsed three sh calls into this one; the contract
    # is "all four keys present, well-formed, ready for readJSON". If this
    # subcommand ever silently changes shape, loadStageConfig() loses a
    # field and the rest of Validate degrades quietly.
    rc = ci_sharding.main(["validate-config", "--choice", "sanity", "--job-name", "finn.dev"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == {"stages", "enabled_params", "retention", "job_key"}
    assert isinstance(payload["stages"], list) and payload["stages"]
    assert payload["enabled_params"] == ["sanity"]
    assert set(payload["retention"]) == {"image", "artifact"}
    # job-key sanitiser is shared with the standalone subcommand.
    assert payload["job_key"] == "finn.dev"


def test_validate_config_rejects_orphan_zipartifact_board(monkeypatch):
    # validate_boards() runs inside the subcommand so a STAGES row with
    # an orphan board fails Validate loudly, not three stages later when
    # the HW pipeline tries to look it up.
    bad_stages = list(ci_sharding.STAGES) + [
        {
            "param": "sanity",
            "stage": "Bad",
            "marker": "sanity_bnn",
            "shards": 1,
            "workers": 1,
            "zipArtifacts": {"hwTestType": "t", "boards": ["NotABoard"]},
        }
    ]
    monkeypatch.setattr(ci_sharding, "STAGES", bad_stages)
    with pytest.raises(ValueError, match="NotABoard"):
        ci_sharding.main(["validate-config", "--choice", "full", "--job-name", "j"])


def test_active_artifact_rows_present_for_sanity_and_end2end_choices():
    # Mirrors the Groovy hasActiveArtifactRows() decision: in local-fallback
    # mode the SW->HW handoff is silently skipped for these choices (yellow
    # build via aggregateReports), and absent for fpgadataflow.
    def has_artifact_rows(choice):
        enabled = set(ci_sharding.enabled_params_for_choice(choice))
        return any(
            row.get("zipArtifacts") and row.get("param") in enabled for row in ci_sharding.STAGES
        )

    assert has_artifact_rows("sanity") is True
    assert has_artifact_rows("end2end") is True
    assert has_artifact_rows("full") is True
    assert has_artifact_rows("fpgadataflow") is False


def test_jenkins_stage_choices_omits_sanity_head_when_param_absent():
    # If a future STAGES has no sanity rows at all, jenkins_stage_choices()
    # must not synthesise a "sanity" choice the dropdown cannot deliver.
    no_sanity = [
        {"param": "fpgadataflow", "stage": "F", "marker": "f", "shards": 1, "workers": 1},
        {"param": "end2end", "stage": "E", "marker": "e", "shards": 1, "workers": 1},
    ]
    assert ci_sharding.jenkins_stage_choices(stages=no_sanity) == [
        "full",
        "fpgadataflow",
        "end2end",
    ]


def test_update_master_raises_on_persistent_write_failure(tmp_path, monkeypatch):
    # Persistent write failure propagates so the calling pipeline can mark
    # the build UNSTABLE instead of silently leaving a stale master behind.
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

    with pytest.raises(IOError, match="simulated NFS write failure"):
        ci_sharding.update_master(
            str(reports),
            str(master),
            str(out),
            update_persistent=True,
            metadata={"job": "j", "build": "1"},
        )


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


def test_prune_numeric_builds_canonicalises_leading_zeros(tmp_path):
    # On-disk dir name "0123" and a BUILD_NUMBER value of "123" refer to the
    # same build for retention purposes. Without canonicalisation the keep set
    # contained the BUILD_NUMBER as-is and the on-disk leading-zero variant
    # would be eligible for pruning even though it is the current build.
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("0123", "0124", "0125"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    matched = ci_sharding.prune_numeric_builds(
        str(parent),
        current_build="123",
        retain_n=1,
        max_age_days=0,
        dry_run=False,
        tag="t",
    )
    # newest ("0125") kept by retain_n, current build ("0123" via int(123))
    # kept by the current-build guard; "0124" is the only one pruned.
    assert matched == 1
    surviving = sorted(p.name for p in parent.iterdir())
    assert surviving == ["0123", "0125"]


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
    # Local-fallback mode (no NFS): the per-build preview is still written,
    # the master simply has nowhere to live.
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

    rc = ci_sharding.update_master(
        str(reports),
        master_path="",
        out_path=str(out),
        metadata={"job": "j", "build": "1"},
    )

    preview = json.loads(out.read_text())
    assert rc == 0
    assert preview["groups"]["seen"]["samples"] == [1.0]
    assert preview["last_update"]["observed_groups"] == 1
    assert preview["last_update"]["accepted"] == 1


# ----------------------------------------------------------------------------
# Anomaly protection (per-group rolling median + build-wide veto + GC).
# ----------------------------------------------------------------------------


def _seed_master_with_group(path, name, samples, **extra):
    write_json(path, {"version": 1, "groups": {name: {"samples": list(samples), **extra}}})


def _write_observation(reports_dir, stash, name, seconds):
    write_json(
        reports_dir / ("%s.timings.json" % stash),
        {
            "stash": stash,
            "metadata": {"job": "j", "build": "1", "stage": stash},
            "groups": [{"name": name, "seconds": seconds, "count": 1}],
        },
    )


def test_observed_groups_uses_max_seconds_for_duplicate_group(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_observation(reports, "fast", "same", 1.0)
    _write_observation(reports, "slow", "same", 9.0)
    observed = ci_sharding.observed_groups_from_reports(str(reports))
    assert observed["same"]["seconds"] == 9.0
    assert observed["same"]["last_seen_stash"] == "slow"


def test_update_master_cold_start_accepts_first_observation(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(master, {"version": 1, "groups": {}})
    _write_observation(reports, "stage", "newgroup", 42.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["newgroup"]["samples"] == [42.0]
    assert persisted["last_update"]["accepted"] == 1


def test_update_master_grows_samples_to_max_then_trims(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [10.0, 10.0, 10.0, 10.0])
    _write_observation(reports, "stage", "g", 11.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    # 5th sample appended, window full but not yet trimmed.
    assert persisted["groups"]["g"]["samples"] == [10.0, 10.0, 10.0, 10.0, 11.0]
    # Next observation evicts the oldest sample (FIFO ring).
    _write_observation(reports, "stage", "g", 12.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [10.0, 10.0, 10.0, 11.0, 12.0]


def test_update_master_uses_median_so_one_outlier_in_five_is_absorbed(tmp_path):
    # 4 samples at 10s plus one accepted-but-large 35s (ratio 3.5, within
    # [0.25, 4.0]). Median is still 10 so the bin packer's weight is
    # unaffected by the lone wobble.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [10.0, 10.0, 10.0, 10.0])
    _write_observation(reports, "stage", "g", 35.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    weights = ci_sharding.load_group_weights(str(master))
    assert weights["g"] == 10.0


def test_update_master_rejects_high_ratio_outlier(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [10.0, 10.0, 10.0])
    _write_observation(reports, "stage", "g", 100.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [10.0, 10.0, 10.0]
    assert persisted["groups"]["g"]["consecutive_rejections"] == 1
    assert persisted["last_update"]["rejected"] == 1


def test_update_master_rejects_low_ratio_outlier(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [100.0, 100.0, 100.0])
    _write_observation(reports, "stage", "g", 10.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [100.0, 100.0, 100.0]
    assert persisted["groups"]["g"]["consecutive_rejections"] == 1


def test_update_master_rejects_crash_suspect_when_prior_was_large(tmp_path):
    # 0.1s observation against a 600s prior is the "shard crashed before
    # any real test ran" pattern: very low observed AND very large prior.
    # 0.1 / 600 = ratio is also out of band, but the crash-floor rule
    # fires first to give a clearer rejection reason.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [600.0, 600.0])
    _write_observation(reports, "stage", "g", 0.1)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [600.0, 600.0]
    assert persisted["groups"]["g"]["consecutive_rejections"] == 1


def test_update_master_force_accepts_after_three_consecutive_rejections(tmp_path):
    # A real regression (test legitimately becomes 10x slower) would
    # otherwise be locked out forever. After FORCE_ACCEPT_AFTER consecutive
    # rejections, the next observation force-accepts.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [10.0, 10.0, 10.0])
    for _ in range(ci_sharding.FORCE_ACCEPT_AFTER):
        _write_observation(reports, "stage", "g", 100.0)
        ci_sharding.update_master(
            str(reports),
            str(master),
            str(out),
            update_persistent=True,
            allow_force_accept=True,
        )
    persisted = json.loads(master.read_text())
    # third pass triggers force-accept on its observation, master now
    # contains 100.0 alongside the older 10.0 samples.
    assert 100.0 in persisted["groups"]["g"]["samples"]
    assert persisted["groups"]["g"]["consecutive_rejections"] == 0
    assert persisted["last_update"]["force_accepted"] == 1


def test_update_master_skips_all_updates_on_build_wide_anomaly(tmp_path):
    # Five groups all 4.5x faster than usual at the same time -> looks
    # like an LSF-storm where every shard finished faster than expected,
    # poisoning every group at once. Build-wide veto leaves the master
    # untouched and records the anomaly in last_update.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "version": 1,
            "groups": {name: {"samples": [100.0, 100.0]} for name in ["a", "b", "c", "d", "e"]},
        },
    )
    for name in ["a", "b", "c", "d", "e"]:
        _write_observation(reports, name, name, 600.0)  # 6x prior median
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    # No group was updated.
    for name in ["a", "b", "c", "d", "e"]:
        assert persisted["groups"][name]["samples"] == [100.0, 100.0]
    assert persisted["last_update"]["anomaly"] is True
    assert persisted["last_update"]["anomaly_outliers"] == 5
    assert persisted["last_update"]["anomaly_eligible"] == 5


def test_update_master_build_wide_veto_requires_minimum_eligible(tmp_path):
    # With only two eligible groups, even both being outliers must not
    # trigger the build-wide veto (a single-shard debug build observing
    # one group should still apply per-group rules normally).
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "version": 1,
            "groups": {"a": {"samples": [100.0]}, "b": {"samples": [100.0]}},
        },
    )
    for name in ("a", "b"):
        _write_observation(reports, name, name, 600.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["last_update"]["anomaly"] is False
    # both rejected per-group, master unchanged for both
    assert persisted["last_update"]["rejected"] == 2
    assert persisted["groups"]["a"]["samples"] == [100.0]
    assert persisted["groups"]["b"]["samples"] == [100.0]


def test_update_master_garbage_collects_groups_unseen_for_n_builds(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    # Set up a build_seq large enough that the cutoff kicks in, plus a
    # group that's stale (last_seen_build_seq way below cutoff).
    write_json(
        master,
        {
            "version": 1,
            "build_seq": ci_sharding.GC_BUILDS_UNSEEN + 50,
            "groups": {
                "stale": {"samples": [10.0], "last_seen_build_seq": 1},
                "fresh": {
                    "samples": [10.0],
                    "last_seen_build_seq": ci_sharding.GC_BUILDS_UNSEEN + 40,
                },
            },
        },
    )
    _write_observation(reports, "stage", "fresh", 10.0)
    ci_sharding.update_master(
        str(reports), str(master), str(out), update_persistent=True, run_gc=True
    )
    persisted = json.loads(master.read_text())
    assert "stale" not in persisted["groups"]
    assert "fresh" in persisted["groups"]
    assert persisted["last_update"]["gc_dropped"] == 1


def test_update_master_preview_does_not_gc_unobserved_groups(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "version": 1,
            "build_seq": ci_sharding.GC_BUILDS_UNSEEN + 50,
            "groups": {
                "end2end": {"samples": [100.0], "last_seen_build_seq": 1},
                "sanity": {"samples": [10.0], "last_seen_build_seq": 200},
            },
        },
    )
    _write_observation(reports, "stage", "sanity", 10.0)
    ci_sharding.update_master(str(reports), str(master), str(out))
    persisted = json.loads(master.read_text())
    preview = json.loads(out.read_text())
    assert "end2end" in persisted["groups"]
    assert persisted["build_seq"] == ci_sharding.GC_BUILDS_UNSEEN + 50
    assert "end2end" in preview["groups"]
    assert preview["last_update"]["persistent_update"] is False
    assert preview["last_update"]["gc_dropped"] == 0


def test_update_master_gc_keeps_old_entries_without_build_seq(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "version": 1,
            "build_seq": ci_sharding.GC_BUILDS_UNSEEN + 50,
            "groups": {
                "old": {"seconds": 25.0, "count": 1},
                "fresh": {
                    "samples": [10.0],
                    "last_seen_build_seq": ci_sharding.GC_BUILDS_UNSEEN + 40,
                },
            },
        },
    )
    _write_observation(reports, "stage", "fresh", 10.0)
    ci_sharding.update_master(
        str(reports), str(master), str(out), update_persistent=True, run_gc=True
    )
    persisted = json.loads(master.read_text())
    assert "old" in persisted["groups"]
    assert "fresh" in persisted["groups"]
    assert persisted["last_update"]["gc_dropped"] == 0


def test_update_master_auto_upgrades_old_seconds_only_entry(tmp_path):
    # Older master files can have ``{seconds: float}`` entries. The first
    # observation against such an entry treats seconds
    # as a one-element samples series and immediately migrates the schema.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(master, {"version": 1, "groups": {"g": {"seconds": 25.0, "count": 1}}})
    _write_observation(reports, "stage", "g", 30.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [25.0, 30.0]
    # seconds key is no longer the source of truth
    assert "samples" in persisted["groups"]["g"]


def test_load_group_weights_handles_new_samples_schema(tmp_path):
    master = tmp_path / "master.json"
    write_json(
        master,
        {
            "version": 1,
            "groups": {
                "g_samples": {"samples": [10.0, 20.0, 30.0]},
                "g_old_dict": {"seconds": 7.5},
                "g_old_flat": 4.2,
            },
        },
    )
    weights = ci_sharding.load_group_weights(str(master))
    assert weights["g_samples"] == 20.0  # median of [10, 20, 30]
    assert weights["g_old_dict"] == 7.5
    assert weights["g_old_flat"] == 4.2


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
