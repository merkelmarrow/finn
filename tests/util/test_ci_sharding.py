# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import os
import re

from finn.util import ci_sharding

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_update_master_preserves_unseen_and_appends_seen(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "schema_version": ci_sharding.SCHEMA_VERSION,
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
    # observed groups are always appended; unseen groups are left untouched.
    assert persisted["groups"]["seen"]["samples"] == [1.0, 3.5]
    assert persisted["groups"]["unseen"]["samples"] == [7.0]
    assert merged["groups"]["seen"]["samples"] == [1.0, 3.5]
    assert merged["last_update"]["observed_groups"] == 1
    assert merged["last_update"]["persistent_update"] is True


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
    assert data["schema_version"] == ci_sharding.SCHEMA_VERSION


def test_prepare_timing_snapshot_copies_master_when_present(tmp_path):
    master = tmp_path / "master.json"
    snapshot = tmp_path / "snapshot.json"
    write_json(
        master,
        {
            "schema_version": ci_sharding.SCHEMA_VERSION,
            "groups": {"slow": {"samples": [12.0]}},
        },
    )

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


def _seed_build(artifact_dir, job_key, build, test_type, board, ready=True):
    build_dir = artifact_dir / "ci_runs" / job_key / build
    zip_dir = build_dir / "zips" / test_type
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / ("%s.zip" % board)
    zip_path.write_text("")
    if ready:
        (zip_dir / ("%s.zip.READY" % board)).write_text("")
    return zip_path, build_dir


def test_resolve_build_zips_picks_newest_ready_per_pair(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "10", "bnn_build_full", "U250", ready=False)
    _seed_build(art, "finn", "11", "bnn_build_full", "U250", ready=True)
    _seed_build(art, "finn", "12", "bnn_build_full", "Pynq-Z1", ready=True)

    out = ci_sharding.resolve_build_zips(
        str(art), "finn", ["bnn_build_full"], ["U250", "Pynq-Z1", "ZCU104"]
    )

    assert out["bnn_build_full"]["U250"]["zip"].endswith("/11/zips/bnn_build_full/U250.zip")
    assert out["bnn_build_full"]["U250"]["buildDir"].endswith("/finn/11")
    assert out["bnn_build_full"]["Pynq-Z1"]["zip"].endswith("/12/zips/bnn_build_full/Pynq-Z1.zip")
    assert out["bnn_build_full"]["ZCU104"] == {}


def test_resolve_build_zips_falls_back_to_older_build_per_board(tmp_path):
    # A new build that only succeeded for Pynq-Z1 must not strand U250 on the
    # older build it last produced a READY for. This is the per-board fallback
    # contract HW relies on.
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "20", "bnn_build_full", "U250", ready=True)
    _seed_build(art, "finn", "21", "bnn_build_full", "Pynq-Z1", ready=True)

    out = ci_sharding.resolve_build_zips(str(art), "finn", ["bnn_build_full"], ["U250", "Pynq-Z1"])

    assert out["bnn_build_full"]["U250"]["buildDir"].endswith("/finn/20")
    assert out["bnn_build_full"]["Pynq-Z1"]["buildDir"].endswith("/finn/21")
    assert out["bnn_build_full"]["U250"]["fallback"] is True
    assert out["bnn_build_full"]["Pynq-Z1"]["fallback"] is False
    assert out["bnn_build_full"]["U250"]["latestBuild"] == "21"


def test_resolve_build_zips_marks_stale_when_newest_build_has_no_ready(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "20", "bnn_build_full", "U250", ready=True)
    # Build 21 exists but produced no READY zips. HW must know it is falling
    # back to stale artefacts rather than treating build 20 as latest.
    (art / "ci_runs" / "finn" / "21").mkdir(parents=True)

    out = ci_sharding.resolve_build_zips(str(art), "finn", ["bnn_build_full"], ["U250"])

    assert out["bnn_build_full"]["U250"]["buildDir"].endswith("/finn/20")
    assert out["bnn_build_full"]["U250"]["latestBuild"] == "21"
    assert out["bnn_build_full"]["U250"]["fallback"] is True


def test_resolve_build_zips_honours_explicit_build_dir(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "5", "bnn_build_full", "U250", ready=True)
    _seed_build(art, "finn", "6", "bnn_build_full", "U250", ready=True)
    explicit = art / "ci_runs" / "finn" / "5"
    out = ci_sharding.resolve_build_zips(
        str(art), "finn", ["bnn_build_full"], ["U250"], build_dir=str(explicit)
    )
    assert out["bnn_build_full"]["U250"]["buildDir"] == str(explicit)
    assert out["bnn_build_full"]["U250"]["fallback"] is False


def test_resolve_build_zips_returns_empty_when_no_ready_anywhere(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "1", "bnn_build_full", "U250", ready=False)

    out = ci_sharding.resolve_build_zips(str(art), "finn", ["bnn_build_full"], ["U250"])
    assert out == {"bnn_build_full": {"U250": {}}}


def test_resolve_build_zips_skips_when_job_root_missing(tmp_path):
    out = ci_sharding.resolve_build_zips(
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


def test_jenkinsfile_stage_choices_match_python_source():
    # Anchor on the STAGES choice block so a future ``choice(name: 'XYZ', ...)``
    # cannot match instead. Accept both single- and double-quoted Groovy strings.
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


def test_readme_stages_table_matches_python_source():
    readme = os.path.join(REPO_ROOT, "docker", "jenkins", "README.md")
    text = open(readme).read()
    # Parse the values column of the "| STAGES value | ... |" table.
    table_rows = re.findall(r"^\|\s*`([a-z0-9_]+)`(?:\s*\(default\))?\s*\|", text, re.MULTILINE)
    expected = ci_sharding.jenkins_stage_choices()
    assert (
        table_rows == expected
    ), "README STAGES table %r drifted from ci_sharding.jenkins_stage_choices() %r" % (
        table_rows,
        expected,
    )


def test_marker_safe_pattern_rejects_and_not_and_double_space():
    # The conftest plugin's _marker_tokens treats every whitespace-separated
    # token as an `or` disjunct, so the regex must forbid and/not and the
    # double-space shape that would create an empty token.
    jenkinsfile = os.path.join(REPO_ROOT, "docker", "jenkins", "Jenkinsfile")
    text = open(jenkinsfile).read()
    match = re.search(r"MARKER_SAFE_PATTERN\s*=\s*~/([^/]+)/", text)
    assert match is not None, "could not locate MARKER_SAFE_PATTERN in Jenkinsfile"
    pattern = re.compile(match.group(1))
    # Every current STAGES marker must be accepted.
    for row in ci_sharding.STAGES:
        assert pattern.match(row["marker"]), (
            "MARKER_SAFE_PATTERN rejects STAGES marker %r" % row["marker"]
        )
    # And the foot-guns the conftest plugin would silently misinterpret
    # must be rejected.
    for bad in ("foo and bar", "not slow", "foo  or bar", "foo or", "or foo"):
        assert not pattern.match(bad), "MARKER_SAFE_PATTERN should reject %r" % bad


def test_multi_shard_rows_use_loadgroup_dist_mode():
    # worksteal across xdist_group siblings breaks chained checkpoint tests,
    # so every multi-shard row must opt in to loadgroup explicitly.
    for row in ci_sharding.STAGES:
        if int(row["shards"]) > 1:
            assert (
                row.get("distMode") == "loadgroup"
            ), "STAGES row %r has shards>1 but distMode=%r" % (row["stage"], row.get("distMode"))


def test_validate_stage_row_rejects_multi_shard_without_loadgroup():
    row = {"param": "p", "stage": "X", "marker": "x", "shards": 2, "workers": 1}
    with pytest.raises(ValueError, match="must set distMode='loadgroup'"):
        ci_sharding.validate_stage_row(row)


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
    # The Jenkinsfile collapsed three sh calls into this one. The contract
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
    assert set(payload["retention"]) == {"image", "artifact", "snapshot"}
    # job-key sanitiser is shared with the standalone subcommand.
    assert payload["job_key"] == "finn.dev"


def test_validate_config_rejects_orphan_zipartifact_board(monkeypatch, capsys):
    # validate_config() runs inside the subcommand so a STAGES row with
    # an orphan board fails Validate loudly, not three stages later when
    # the HW pipeline tries to look it up.
    bad_stages = list(ci_sharding.STAGES) + [
        {
            "param": "sanity",
            "stage": "Bad",
            "marker": "sanity_bnn",
            "shards": 1,
            "workers": 1,
            "zipArtifacts": {"hwTestType": "bnn_build_sanity", "boards": ["NotABoard"]},
        }
    ]
    monkeypatch.setattr(ci_sharding, "STAGES", bad_stages)
    rc = ci_sharding.main(["validate-config", "--choice", "full", "--job-name", "j"])
    assert rc == 2
    assert "NotABoard" in capsys.readouterr().err


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
    write_json(master, {"schema_version": ci_sharding.SCHEMA_VERSION, "groups": {}})
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
    # kept by the current-build guard. "0124" is the only one pruned.
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
    # nobody can run. validate_config() must reject that asymmetry.
    ci_sharding.validate_config()


@pytest.mark.parametrize(
    "board,row,match",
    [
        ("B", {}, "missing required"),
        (
            "B",
            {
                "agentLabel": "",
                "credentialsId": None,
                "restartPrep": False,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "invalid agentLabel",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": "",
                "restartPrep": False,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "invalid credentialsId",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": None,
                "restartPrep": True,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "restartPrep=true but no credentialsId",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": None,
                "restartPrep": False,
                "setupScript": "bogus",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "invalid setupScript",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": None,
                "restartPrep": False,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "",
            },
            "invalid bnnMarker",
        ),
    ],
)
def test_validate_board_row_rejects_bad_input(board, row, match):
    with pytest.raises(ValueError, match=match):
        ci_sharding.validate_board_row(board, row)


def test_validate_config_rejects_orphan_zipartifact_board_via_helper():
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
        ci_sharding.validate_config(stages=custom_stages, boards={})


@pytest.mark.parametrize(
    "zip_art,match",
    [
        ({}, "non-empty hwTestType"),
        ({"hwTestType": "", "boards": ["U250"]}, "non-empty hwTestType"),
        ({"hwTestType": "t"}, "non-empty boards list"),
        ({"hwTestType": "t", "boards": []}, "non-empty boards list"),
        ({"hwTestType": "t", "boards": [""]}, "non-empty boards list"),
    ],
)
def test_validate_stage_row_rejects_bad_zip_artifacts(zip_art, match):
    row = {
        "param": "p",
        "stage": "X",
        "marker": "x",
        "shards": 1,
        "workers": 1,
        "zipArtifacts": zip_art,
    }
    with pytest.raises(ValueError, match=match):
        ci_sharding.validate_stage_row(row)


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
    # build '3' is kept (retain_n=1, newest), '2' got rmtreed for real,
    # '1' was the simulated race victim and we tolerated it
    surviving = sorted(p.name for p in parent.iterdir())
    assert "3" in surviving
    assert "2" not in surviving


def test_prune_numeric_builds_tolerates_concurrent_delete_in_age_check(tmp_path, monkeypatch):
    parent = tmp_path / "p"
    parent.mkdir()
    # retain_n=1 keeps the newest ('3'). '1' and '2' are both deletion
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
        str(master),
        lambda cur: {
            "schema_version": ci_sharding.SCHEMA_VERSION,
            "groups": {"k": {"samples": [1.0]}},
        },
    )

    assert updated["groups"]["k"]["samples"] == [1.0]
    fresh = json.loads(master.read_text())
    assert fresh["groups"]["k"]["samples"] == [1.0]

    backups = sorted(p.name for p in tmp_path.iterdir() if ".corrupt-" in p.name)
    assert len(backups) == 1
    assert backups[0].startswith("master.json.corrupt-")
    assert (tmp_path / backups[0]).read_text() == "{ this is not json"


def test_locked_update_does_not_backup_empty_master(tmp_path):
    master = tmp_path / "master.json"
    master.write_text("")

    ci_sharding.locked_update(
        str(master),
        lambda cur: {"schema_version": ci_sharding.SCHEMA_VERSION, "groups": {}},
    )

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
    assert preview["last_update"]["persistent_update"] is False


# ----------------------------------------------------------------------------
# Per-group rolling-median update (append-and-trim).
# ----------------------------------------------------------------------------


def _seed_master_with_group(path, name, samples, **extra):
    write_json(
        path,
        {
            "schema_version": ci_sharding.SCHEMA_VERSION,
            "groups": {name: {"samples": list(samples), **extra}},
        },
    )


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
    write_json(master, {"schema_version": ci_sharding.SCHEMA_VERSION, "groups": {}})
    _write_observation(reports, "stage", "newgroup", 42.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["newgroup"]["samples"] == [42.0]
    assert persisted["last_update"]["observed_groups"] == 1


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
    # Plain append: window [10, 10, 10, 10, 35] has a median of 10, so the
    # bin packer's weight is unaffected by a single lone wobble.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [10.0, 10.0, 10.0, 10.0])
    _write_observation(reports, "stage", "g", 35.0)
    ci_sharding.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [10.0, 10.0, 10.0, 10.0, 35.0]
    weights = ci_sharding.load_group_weights(str(master))
    assert weights["g"] == 10.0


def test_update_master_preview_leaves_persistent_master_untouched(tmp_path):
    # Non-persist mode must write the per-build preview to out_path but
    # never touch the on-disk master.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "schema_version": ci_sharding.SCHEMA_VERSION,
            "groups": {"g": {"samples": [10.0]}},
        },
    )
    _write_observation(reports, "stage", "g", 25.0)
    ci_sharding.update_master(str(reports), str(master), str(out))
    persisted = json.loads(master.read_text())
    preview = json.loads(out.read_text())
    assert persisted["groups"]["g"]["samples"] == [10.0]
    assert preview["groups"]["g"]["samples"] == [10.0, 25.0]
    assert preview["last_update"]["persistent_update"] is False


def test_load_group_weights_returns_median_of_samples(tmp_path):
    master = tmp_path / "master.json"
    write_json(
        master,
        {
            "schema_version": ci_sharding.SCHEMA_VERSION,
            "groups": {"g": {"samples": [10.0, 20.0, 30.0]}},
        },
    )
    weights = ci_sharding.load_group_weights(str(master))
    assert weights["g"] == 20.0


def test_normalise_master_drops_unknown_schema_version(capsys):
    out = ci_sharding.normalise_master({"schema_version": 99, "groups": {"g": {"samples": [1.0]}}})
    assert out["schema_version"] == ci_sharding.SCHEMA_VERSION
    assert out["groups"] == {}
    assert "unrecognised schema_version" in capsys.readouterr().err


def test_normalise_master_writes_canonical_schema_version_key():
    out = ci_sharding.normalise_master({"groups": {}})
    assert out["schema_version"] == ci_sharding.SCHEMA_VERSION


def _install_finn_ci_plugin(pytester):
    """Wire the real tests/conftest.py into a pytester sandbox.

    Reused by every test that exercises the conftest plugin end-to-end so
    each one only has to declare its own test files and pytest invocation.
    """
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


def _scrub_xdist_env(monkeypatch):
    """Drop PYTEST_XDIST_WORKER for the duration of a pytester sub-run.

    Outer fpgadataflow / unit shards run with -n 8 --dist loadgroup, so every
    worker has PYTEST_XDIST_WORKER=gwN set. pytester.runpytest is inline and
    inherits that env, which makes the inner plugin's pytest_sessionstart
    short-circuit on _is_xdist_worker() and never initialise _TIMINGS. The
    inner timings.json never gets written and these tests false-fail.
    """
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)


def test_pytest_plugin_writes_timings_for_successful_sharded_run(pytester, monkeypatch):
    _scrub_xdist_env(monkeypatch)
    _install_finn_ci_plugin(pytester)
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


def test_pytest_plugin_writes_empty_shard_sidecar_when_slice_collected_zero(pytester, monkeypatch):
    _scrub_xdist_env(monkeypatch)
    # Two single-test groups round-robin onto shards 0 and 1 (sorted by
    # nodeid). Shard 1 therefore gets one item, shard 0 gets the other.
    # If we then ask for shard 0 with a marker that only matches the
    # second test, the slice is empty and the plugin must:
    #  - remap exit 5 to 0 so the build does not fail spuriously
    #  - drop a <stash>.empty-shard sidecar so the aggregator can tell
    #    "shard had no work" apart from "shard crashed".
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
import pytest

@pytest.mark.fpgadataflow
def test_a():
    assert True

@pytest.mark.end2end
def test_b():
    assert True
"""
    )

    result = pytester.runpytest(
        "-m",
        "fpgadataflow",
        "--num-shards=2",
        "--shard-id=1",
        "--junitxml=stage.xml",
        "-q",
    )

    assert result.ret == 0
    sidecar = pytester.path / "stage.empty-shard"
    assert sidecar.exists()
    assert "0 items" in sidecar.read_text()


def test_pytest_plugin_rejects_conflicting_shard_pins_within_xdist_group(pytester, monkeypatch):
    # Two tests share an xdist_group and disagree on @pytest.mark.shard(N).
    # _assignment_details must surface this as a UsageError rather than
    # silently splitting a chained checkpoint sequence across shards.
    _scrub_xdist_env(monkeypatch)
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
import pytest

@pytest.mark.xdist_group(name="chain")
@pytest.mark.shard(0)
def test_first():
    assert True

@pytest.mark.xdist_group(name="chain")
@pytest.mark.shard(1)
def test_second():
    assert True
"""
    )

    result = pytester.runpytest(
        "--num-shards=2",
        "--shard-id=0",
        "--junitxml=stage.xml",
        "-q",
    )

    assert result.ret != 0
    combined = "\n".join(result.outlines + result.errlines)
    assert "conflicting" in combined
    assert "chain" in combined


def test_pytest_plugin_dry_run_prints_per_shard_table_and_exits_zero(pytester, monkeypatch):
    # --dry-run-shards must print the header row and exit 0 without running
    # any test. We deselect everything by design so exit 5 is benign.
    _scrub_xdist_env(monkeypatch)
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
def test_a():
    assert True

def test_b():
    assert True
"""
    )

    result = pytester.runpytest(
        "--num-shards=2",
        "--shard-id=0",
        "--dry-run-shards",
        "--junitxml=stage.xml",
        "-q",
    )

    assert result.ret == 0
    out = "\n".join(result.outlines)
    assert "dry-run-shards" in out
    assert "shard" in out and "items" in out and "groups" in out
    # No tests should have actually run.
    result.assert_outcomes()


# ============================================================================
# hw-test-types-json
# ============================================================================


def test_hw_test_types_lists_distinct_values_in_stage_order():
    # ordering is first-appearance in STAGES; structural check, no snapshot.
    types = ci_sharding.hw_test_types()
    declared = [
        row["zipArtifacts"]["hwTestType"] for row in ci_sharding.STAGES if row.get("zipArtifacts")
    ]
    seen = []
    for t in declared:
        if t not in seen:
            seen.append(t)
    assert types == seen


def test_hw_test_types_json_cli_emits_valid_json(capsys):
    rc = ci_sharding.main(["hw-test-types-json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert isinstance(parsed, list)
    assert "bnn_build_sanity" in parsed
    assert "bnn_build_full" in parsed


def test_hw_test_type_labels_returns_label_for_every_referenced_type():
    labels = ci_sharding.hw_test_type_labels()
    # ordering matches first-appearance of the hwTestType in STAGES so the
    # Jenkins HW UI rows render in the same order as the build-side stages
    assert list(labels) == ci_sharding.hw_test_types()
    for tt in ci_sharding.hw_test_types():
        assert labels[tt]
        assert isinstance(labels[tt], str)


def test_hw_test_type_labels_json_cli_emits_valid_json(capsys):
    rc = ci_sharding.main(["hw-test-type-labels-json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert isinstance(parsed, dict)
    assert parsed == ci_sharding.hw_test_type_labels()


def test_validate_config_rejects_hw_test_type_without_label(monkeypatch):
    bad_stages = list(ci_sharding.STAGES) + [
        {
            "param": "sanity",
            "stage": "Bad",
            "marker": "sanity_bnn",
            "shards": 1,
            "workers": 1,
            "zipArtifacts": {"hwTestType": "bnn_build_unlabelled", "boards": ["U250"]},
        }
    ]
    monkeypatch.setattr(ci_sharding, "STAGES", bad_stages)
    with pytest.raises(ValueError, match="bnn_build_unlabelled"):
        ci_sharding.validate_config()


# ============================================================================
# prune-snapshots
# ============================================================================


def test_prune_snapshots_keeps_current_build_and_newest(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    (state_root / job).mkdir(parents=True)
    for n in (1, 2, 3, 4, 5):
        (state_root / job / ("build_%d_timings_input.json" % n)).write_text("{}")
    # Non-numbered files (the master itself, corrupt backups) must be left
    # alone even when they sort lexicographically alongside snapshots.
    (state_root / job / "ci_timings_master.json").write_text("{}")
    (state_root / job / "ci_timings_master.json.corrupt-1").write_text("{}")
    ci_sharding.prune_snapshots(str(state_root), job, current_build="3", retain_n=2, max_age_days=0)
    remaining = sorted(p.name for p in (state_root / job).iterdir())
    assert "build_3_timings_input.json" in remaining
    assert "build_4_timings_input.json" in remaining
    assert "build_5_timings_input.json" in remaining
    assert "ci_timings_master.json" in remaining
    assert "ci_timings_master.json.corrupt-1" in remaining
    assert "build_1_timings_input.json" not in remaining
    assert "build_2_timings_input.json" not in remaining


def test_prune_snapshots_skips_when_parent_missing(tmp_path, capsys):
    rc = ci_sharding.prune_snapshots(
        str(tmp_path / "nope"), "finn", current_build="1", retain_n=2, max_age_days=0
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "not present, skipping" in captured.out


def test_prune_snapshots_rejects_non_numeric_current(tmp_path):
    with pytest.raises(ValueError, match="prune-snapshots: current_build must be"):
        ci_sharding.prune_snapshots(
            str(tmp_path), "finn", current_build="x", retain_n=1, max_age_days=0
        )


def test_prune_snapshots_honours_age_gating(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    parent = state_root / job
    parent.mkdir(parents=True)
    old = parent / "build_1_timings_input.json"
    fresh = parent / "build_2_timings_input.json"
    old.write_text("{}")
    fresh.write_text("{}")
    os.utime(str(old), (1, 1))

    ci_sharding.prune_snapshots(str(state_root), job, current_build="3", retain_n=1, max_age_days=1)

    assert not old.exists()
    assert fresh.exists()


def test_prune_snapshots_tolerates_concurrent_delete_in_age_check(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    job = "finn"
    parent = state_root / job
    parent.mkdir(parents=True)
    for n in (1, 2, 3):
        path = parent / ("build_%d_timings_input.json" % n)
        path.write_text("{}")
        os.utime(str(path), (1, 1))
    real_getmtime = ci_sharding.os.path.getmtime

    def flaky_getmtime(path):
        if path.endswith("build_1_timings_input.json"):
            raise FileNotFoundError(path)
        return real_getmtime(path)

    monkeypatch.setattr(ci_sharding.os.path, "getmtime", flaky_getmtime)
    ci_sharding.prune_snapshots(str(state_root), job, current_build="9", retain_n=1, max_age_days=7)

    assert not (parent / "build_2_timings_input.json").exists()
    assert (parent / "build_3_timings_input.json").exists()


def test_prune_snapshots_tolerates_concurrent_delete_on_unlink(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    job = "finn"
    parent = state_root / job
    parent.mkdir(parents=True)
    for n in (1, 2, 3):
        path = parent / ("build_%d_timings_input.json" % n)
        path.write_text("{}")
        os.utime(str(path), (1, 1))
    real_unlink = ci_sharding.os.unlink
    state = {"first": True}

    def flaky_unlink(path):
        if state["first"]:
            state["first"] = False
            raise FileNotFoundError(path)
        return real_unlink(path)

    monkeypatch.setattr(ci_sharding.os, "unlink", flaky_unlink)
    ci_sharding.prune_snapshots(str(state_root), job, current_build="9", retain_n=1, max_age_days=0)

    assert not (parent / "build_2_timings_input.json").exists()
    assert (parent / "build_3_timings_input.json").exists()


def test_prune_snapshots_dry_run_does_not_delete(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    (state_root / job).mkdir(parents=True)
    for n in (1, 2, 3):
        (state_root / job / ("build_%d_timings_input.json" % n)).write_text("{}")
    ci_sharding.prune_snapshots(
        str(state_root), job, current_build="3", retain_n=1, max_age_days=0, dry_run=True
    )
    remaining = sorted(p.name for p in (state_root / job).iterdir())
    assert remaining == [
        "build_1_timings_input.json",
        "build_2_timings_input.json",
        "build_3_timings_input.json",
    ]


def test_prune_snapshots_cli_smoke(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    (state_root / job).mkdir(parents=True)
    (state_root / job / "build_1_timings_input.json").write_text("{}")
    rc = ci_sharding.main(["prune-snapshots", str(state_root), job, "1", "1", "0", "--dry-run"])
    assert rc == 0


# ============================================================================
# validate_stage_row
# ============================================================================


def test_validate_stage_row_accepts_each_live_row():
    for row in ci_sharding.STAGES:
        ci_sharding.validate_stage_row(row)


@pytest.mark.parametrize(
    "bad,match",
    [
        (
            {"stage": "X", "marker": "a", "shards": 1, "workers": 1},
            "missing param",
        ),
        (
            {"stage": "X", "param": "", "marker": "a", "shards": 1, "workers": 1},
            "missing param",
        ),
        (
            {"stage": "X", "param": 7, "marker": "a", "shards": 1, "workers": 1},
            "missing param",
        ),
        (
            {"stage": "X", "param": "p", "marker": "a and b", "shards": 1, "workers": 1},
            "unsafe marker",
        ),
        (
            {"stage": "X", "param": "p", "marker": "a", "shards": 0, "workers": 1},
            "invalid shards",
        ),
        (
            {"stage": "X", "param": "p", "marker": "a", "shards": 1, "workers": 0},
            "invalid workers",
        ),
        (
            {
                "stage": "X",
                "param": "p",
                "marker": "a",
                "shards": 1,
                "workers": 1,
                "coverage": "yes",
            },
            "invalid coverage",
        ),
        (
            {
                "stage": "X",
                "param": "p",
                "marker": "a",
                "shards": 1,
                "workers": 1,
                "distMode": "bogus",
            },
            "invalid distMode",
        ),
    ],
)
def test_validate_stage_row_rejects_bad_input(bad, match):
    with pytest.raises(ValueError, match=match):
        ci_sharding.validate_stage_row(bad)


def test_validate_stage_row_accepts_explicit_coverage_bool():
    row = {
        "param": "p",
        "stage": "X",
        "marker": "a",
        "shards": 1,
        "workers": 1,
        "coverage": True,
    }
    ci_sharding.validate_stage_row(row)
    row["coverage"] = False
    ci_sharding.validate_stage_row(row)


def test_validate_config_runs_validate_stage_row_for_every_entry(monkeypatch, capsys):
    # CLI form: main() catches ValueError, prints a one-line ci_sharding:
    # message to stderr, and exits 2 instead of leaking a Python traceback
    # into the Jenkins Validate console.
    bad_stages = [
        {"param": "p", "stage": "Bad", "marker": "a and b", "shards": 1, "workers": 1},
    ]
    monkeypatch.setattr(ci_sharding, "STAGES", bad_stages)
    rc = ci_sharding.main(["validate-config", "--choice", "p", "--job-name", "j"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ci_sharding: ")
    assert "unsafe marker" in captured.err
    assert "Traceback" not in captured.err
