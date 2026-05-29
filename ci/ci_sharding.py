#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FINN CI sharding, timing, and board/stage config helpers.

Lives in the top-level ``ci/`` directory, out of the shipped finn package.
Used as a CLI tool from the Jenkinsfile and imported as a module by the pytest
plugin ``finn_ci_plugin`` (shard assignment) and by the end2end board
parametrisation (``BOARDS``/``TEST_BOARDS``). The file is internally segmented
by ``# =====`` banners and is meant to be read top-down: configuration knobs
first, then declarative tables, then helpers grouped by the concern they own.
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
from nfs_lock import nfs_dir_lock

# =============================================================================
# Configuration knobs
# =============================================================================
# Sharding policy. Tune here. The rest of the file should not need editing
# for a knob change.

# Per-group rolling window for the timing master. The bin packer reads the
# median of the window so a single anomalous observation is absorbed.
MAX_SAMPLES = 5

# summarize-timings flags shards exceeding this multiple of the family median.
SLOW_FACTOR = 1.5

# Per-tree retention for the prune-images / prune-artifacts / prune-snapshots
# CLI subcommands. Image and snapshot trees are cheap to rebuild. Artifacts
# double as the per-board fallback when an individual board's most recent
# build regressed, so the artifact window is deep enough to outlast the
# longest realistic single-board streak.
#
# Disk budget: at ~80 GB per image and ~200 MB per board zip, the defaults
# below cap shared storage at roughly 240 GB of image plus 5-10 GB of zips.
RETENTION = {
    "image": {"retain": 3, "ageDays": 14},
    "artifact": {"retain": 30, "ageDays": 30},
    "snapshot": {"retain": 3, "ageDays": 2},
}


# Internal regexes for canonical_key and stash family lookup.
GROUP_SUFFIX_RE = re.compile(r"@(\S+)$")
NOTEBOOK_PARAM_RE = re.compile(r"\[(/[^\]]+\.ipynb)\]")

# STAGES row markers are interpolated into a shell command. Only the
# ``a or b or c`` shape is accepted so the pytest plugin's mini-evaluator
# and the Jenkinsfile's MARKER_SAFE_PATTERN agree on what is safe.
MARKER_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9_]+( or [A-Za-z0-9_]+)*$")

# Board setup scripts implemented by Jenkinsfile_HW.createTestScript().
VALID_BOARD_SETUP_SCRIPTS = ("alveo", "pynq")


# =============================================================================
# Declarative tables (BOARDS, STAGES)
# =============================================================================
# Single source of truth for the HW pipeline's per-board metadata and the
# build pipeline's per-row matrix. The Groovy side reads these via the
# ``hw-config-json`` and ``validate-config`` subcommands.

# Per-board HW-pipeline metadata. Fields:
#   agentLabel:    Jenkins label of the board agent
#   credentialsId: Jenkins credential binding for sudo on the board (None for
#                  Alveo PCIe boards already mounted as root)
#   restartPrep:   pre-run reboot dance for Zynq boards
#   setupScript:   ``alveo`` (XRT) or ``pynq`` (on-board pynq venv)
#   marker:        the ``-m`` expression the on-board test file runs against
#                  (differs from the board name only for Pynq-Z1, whose
#                  marker cannot contain a hyphen)
#   bnnMarker:     the build-side ``-m`` token the BNN end2end test file
#                  parametrises this board under (``bnn_<board>``)
#
# Key order is the canonical test-parametrisation order and is load-bearing:
# ``TEST_BOARDS`` below derives from it and is what ``tests/end2end``
# parametrises against. Reordering the keys reorders the test matrix.
BOARDS = {
    "Pynq-Z1": {
        "agentLabel": "finn-pynq",
        "credentialsId": "pynq-z1-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "Pynq",
        "bnnMarker": "bnn_pynq",
    },
    "KV260_SOM": {
        "agentLabel": "finn-kv260",
        "credentialsId": "user-ubuntu-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "KV260_SOM",
        "bnnMarker": "bnn_kv260",
    },
    "ZCU104": {
        "agentLabel": "finn-zcu104",
        "credentialsId": "pynq-z1-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "ZCU104",
        "bnnMarker": "bnn_zcu104",
    },
    "U250": {
        "agentLabel": "finn-u250",
        "credentialsId": None,
        "restartPrep": False,
        "setupScript": "alveo",
        "marker": "U250",
        "bnnMarker": "bnn_u250",
    },
}

# Canonical board iteration order for test parametrisation. Single source of
# truth: ``tests/end2end/test_end2end_bnn_pynq.py`` consumes this tuple.
TEST_BOARDS = tuple(BOARDS)


# Per-test-type human-readable label, surfaced in the Jenkins HW pipeline
# stage names. Add a row when introducing a new STAGES.zipArtifacts.hwTestType;
# validate_config() rejects an hwTestType without a label.
HW_TEST_TYPE_LABELS = {
    "bnn_build_sanity": "Sanity",
    "bnn_build_full": "end2end",
}


# Per-row CI matrix. Fields:
#   param:        Jenkins ``STAGES`` choice that activates this row
#   stage:        human-readable display name
#   marker:       pytest ``-m`` expression (must match MARKER_SAFE_PATTERN in
#                 the Jenkinsfile, so only ``a or b ...``)
#   shards:       how many parallel shards to split the row into
#   workers:      pytest-xdist worker count per shard. >1 implies
#                 ``--dist loadgroup`` so xdist_group chains (e.g. tests linked
#                 by load_test_checkpoint_or_skip) stay on one worker
#   coverage:     optional, request coverage report
#   zipArtifacts: optional, {"hwTestType": ..., "boards": [...]} declaring the
#                 build-to-HW handoff zips this row publishes. hwTestType must
#                 have a HW_TEST_TYPE_LABELS entry
STAGES = [
    {
        "param": "sanity",
        "stage": "Sanity - Build Hardware",
        "marker": "sanity_bnn",
        "shards": 4,
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
    },
    {
        "param": "end2end",
        "stage": "BNN U250",
        "marker": "bnn_u250",
        "shards": 2,
        "workers": 2,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["U250"]},
    },
    {
        "param": "end2end",
        "stage": "BNN Pynq-Z1",
        "marker": "bnn_pynq",
        "shards": 3,
        "workers": 2,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["Pynq-Z1"]},
    },
    {
        "param": "end2end",
        "stage": "BNN ZCU104",
        "marker": "bnn_zcu104",
        "shards": 2,
        "workers": 4,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["ZCU104"]},
    },
    {
        "param": "end2end",
        "stage": "BNN KV260",
        "marker": "bnn_kv260",
        "shards": 2,
        "workers": 2,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["KV260_SOM"]},
    },
]


def marker_tokens(marker_expr):
    """Return marker names from the restricted ``a or b`` expression form."""
    return [t for t in str(marker_expr).split() if t != "or"]


def validate_stage_row(row):
    """Sanity-check a single STAGES row in isolation.

    Catches a typo in param / marker / shards / workers / coverage
    at config-load time so a malformed row cannot silently survive into
    Jenkins or the pytest plugin. A missing or empty ``param`` is the
    sneakiest of these because the row would be skipped by every Groovy
    helper that gates on ``rowIsActive`` without anything ever logging why.
    """
    stage = row.get("stage", "<unnamed>")
    param = row.get("param")
    if not isinstance(param, str) or not param:
        raise ValueError("STAGES row %r has invalid or missing param=%r" % (stage, param))
    marker = row.get("marker", "")
    if not MARKER_SAFE_PATTERN.match(str(marker)):
        raise ValueError(
            "STAGES row %r has unsafe marker %r "
            "(only 'a or b or c ...' is allowed)" % (stage, marker)
        )
    shards = row.get("shards")
    if not isinstance(shards, int) or shards < 1:
        raise ValueError("STAGES row %r has invalid shards=%r" % (stage, shards))
    workers = row.get("workers")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("STAGES row %r has invalid workers=%r" % (stage, workers))
    if "coverage" in row and not isinstance(row["coverage"], bool):
        raise ValueError(
            "STAGES row %r has invalid coverage=%r (must be bool)" % (stage, row["coverage"])
        )
    zip_art = row.get("zipArtifacts")
    if zip_art is None:
        return
    _validate_zip_artifact(stage, zip_art)


def _validate_zip_artifact(stage, zip_art):
    """Sanity-check the optional ``zipArtifacts`` block of one STAGES row."""
    if not isinstance(zip_art, dict):
        raise ValueError("STAGES row %r has invalid zipArtifacts=%r" % (stage, zip_art))
    hw_test_type = zip_art.get("hwTestType")
    if not isinstance(hw_test_type, str) or not hw_test_type:
        raise ValueError("STAGES row %r has zipArtifacts without a non-empty hwTestType" % stage)
    boards = zip_art.get("boards")
    if (
        not isinstance(boards, list)
        or not boards
        or not all(isinstance(b, str) and b for b in boards)
    ):
        raise ValueError("STAGES row %r has zipArtifacts without a non-empty boards list" % stage)


def validate_board_row(board, row):
    """Sanity-check one BOARDS row consumed by Jenkinsfile_HW."""
    if not isinstance(row, dict):
        raise ValueError("BOARDS row %r must be a dict, got %r" % (board, row))
    required = (
        "agentLabel",
        "credentialsId",
        "restartPrep",
        "setupScript",
        "marker",
        "bnnMarker",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError("BOARDS row %r is missing required field(s): %r" % (board, missing))
    if not isinstance(row["agentLabel"], str) or not row["agentLabel"]:
        raise ValueError("BOARDS row %r has invalid agentLabel=%r" % (board, row["agentLabel"]))
    credentials = row["credentialsId"]
    if credentials is not None and (not isinstance(credentials, str) or not credentials):
        raise ValueError("BOARDS row %r has invalid credentialsId=%r" % (board, credentials))
    if not isinstance(row["restartPrep"], bool):
        raise ValueError("BOARDS row %r has invalid restartPrep=%r" % (board, row["restartPrep"]))
    if row["restartPrep"] and not credentials:
        raise ValueError("BOARDS row %r has restartPrep=true but no credentialsId" % board)
    if row["setupScript"] not in VALID_BOARD_SETUP_SCRIPTS:
        raise ValueError(
            "BOARDS row %r has invalid setupScript=%r (allowed: %r)"
            % (board, row["setupScript"], list(VALID_BOARD_SETUP_SCRIPTS))
        )
    if not isinstance(row["marker"], str) or not row["marker"]:
        raise ValueError("BOARDS row %r has invalid marker=%r" % (board, row["marker"]))
    if not isinstance(row["bnnMarker"], str) or not row["bnnMarker"]:
        raise ValueError("BOARDS row %r has invalid bnnMarker=%r" % (board, row["bnnMarker"]))


def validate_config(stages=None, boards=None, hw_test_type_labels=None):
    """Validate every STAGES row and the global BOARDS cross-references.

    Fails loud if a STAGES row references a board not declared in BOARDS, if
    a row's hwTestType has no HW_TEST_TYPE_LABELS entry, or if any single
    row is malformed. Called by every CLI subcommand that consumes BOARDS,
    STAGES, or HW_TEST_TYPE_LABELS (``validate-config``, ``hw-config-json``)
    so a bad table fails fast wherever it is first read.
    """
    stages = stages if stages is not None else STAGES
    boards = boards if boards is not None else BOARDS
    labels = hw_test_type_labels if hw_test_type_labels is not None else HW_TEST_TYPE_LABELS
    for board, row in boards.items():
        validate_board_row(board, row)
    for row in stages:
        validate_stage_row(row)
    referenced_boards = set()
    referenced_test_types = set()
    for row in stages:
        zip_art = row.get("zipArtifacts") or {}
        for b in zip_art.get("boards", []) or []:
            referenced_boards.add(str(b))
        hw_test_type = zip_art.get("hwTestType")
        if hw_test_type:
            referenced_test_types.add(str(hw_test_type))
    missing_boards = sorted(referenced_boards - set(boards))
    if missing_boards:
        raise ValueError("STAGES references board(s) not declared in BOARDS: %r" % missing_boards)
    missing_labels = sorted(referenced_test_types - set(labels))
    if missing_labels:
        raise ValueError(
            "STAGES references hwTestType(s) without a HW_TEST_TYPE_LABELS entry: %r"
            % missing_labels
        )


# =============================================================================
# Stage choice / job-key helpers
# =============================================================================


def hw_shards(boards=None):
    """Flatten BOARDS into the ordered list of rows the Groovy HW pipeline expects."""
    boards = boards if boards is not None else BOARDS
    return [dict(board=name, **fields) for name, fields in boards.items()]


def hw_test_types(stages=None):
    """Return distinct ``zipArtifacts.hwTestType`` values in declaration order.

    Single source of truth for the HW pipeline's per-test-type stage loop;
    Jenkinsfile_HW reads this via ``hw-config-json`` instead of duplicating
    the list on the Groovy side.
    """
    stages = stages if stages is not None else STAGES
    seen = []
    for row in stages:
        zip_art = row.get("zipArtifacts")
        if not zip_art:
            continue
        hw_test_type = zip_art["hwTestType"]
        if hw_test_type not in seen:
            seen.append(hw_test_type)
    return seen


def hw_test_type_labels(stages=None, labels=None):
    """Return ``{hwTestType: label}`` for every hwTestType referenced in STAGES.

    Same iteration order as :func:`hw_test_types`. Jenkinsfile_HW reads this
    via ``hw-config-json``, so a new HW test type is a one-line edit of
    HW_TEST_TYPE_LABELS instead of a Groovy edit too.
    """
    stages = stages if stages is not None else STAGES
    labels = labels if labels is not None else HW_TEST_TYPE_LABELS
    return {tt: labels[tt] for tt in hw_test_types(stages)}


def stash_name(stage, shards, shard_id):
    """Mirror ``shardStashName()`` in ``ci/Jenkinsfile``."""
    base = re.sub(r"^_|_$", "", re.sub(r"[^a-z0-9]+", "_", stage.lower()))
    return base if shards <= 1 else "%s_%d" % (base, shard_id + 1)


def job_key(job_name):
    # Strip leading/trailing dots so JOB_NAME=".." cannot survive into a
    # parent-directory traversal under ci_runs/<jobKey>/.
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name or "job").strip(".")
    return out or "job"


def ci_param_names(stages=None):
    """Return the distinct STAGES.param values in declaration order."""
    stages = stages if stages is not None else STAGES
    seen = []
    for row in stages:
        name = row.get("param")
        if name and name not in seen:
            seen.append(name)
    return seen


def enabled_params_for_choice(choice, stages=None):
    """Map the Jenkins ``STAGES`` choice to the set of CI params it enables.

    ``full`` enables every distinct param. A bare param name enables just
    that one. Unknown choices fail loudly so a typo in the Jenkins choice
    list cannot silently run zero shards.
    """
    stages = stages if stages is not None else STAGES
    all_params = ci_param_names(stages)
    if choice == "full":
        return list(all_params)
    if choice in all_params:
        return [choice]
    raise ValueError(
        "unknown STAGES choice %r (recognised: full, %s)" % (choice, ", ".join(all_params))
    )


def jenkins_stage_choices(stages=None):
    """Return the Jenkins UI choices for ``STAGES`` in display order.

    ``sanity`` (the per-PR smoke check) leads when it exists, then ``full``
    for the nightly matrix, then every other distinct param.
    """
    names = list(ci_param_names(stages))
    head = ["sanity"] if "sanity" in names else []
    rest = [n for n in names if n != "sanity"]
    return head + ["full"] + rest


# =============================================================================
# Build-pipeline shard plan
# =============================================================================
# The Jenkinsfile consumes one ``shard_plan`` payload instead of re-deriving
# the active-row/shard/stash matrix in Groovy. Keeping the expansion here keeps
# the CPS-fragile IntRange/closure-serialisation loops out of the pipeline and
# gives the naming logic a single tested home.


def shard_stage_name(stage, shards, shard_id):
    """Mirror the per-shard Jenkins stage display name."""
    return stage if shards <= 1 else "%s (%d/%d)" % (stage, shard_id + 1, shards)


def _shard_entry(row):
    """Expand one active STAGES row into its per-shard plan entries."""
    num_shards = int(row["shards"])
    entries = []
    for shard_id in range(num_shards):
        stash = stash_name(row["stage"], num_shards, shard_id)
        entry = {
            "stage": shard_stage_name(row["stage"], num_shards, shard_id),
            "stash": stash,
            "marker": row["marker"],
            "workers": int(row["workers"]),
            "numShards": num_shards,
            "shardId": shard_id,
            "coverage": bool(row.get("coverage", False)),
        }
        if entry["coverage"]:
            entry["coverageFile"] = "%s.coverage" % stash
        zip_art = row.get("zipArtifacts")
        if zip_art:
            entry["zipArtifacts"] = zip_art
        entries.append(entry)
    return entries


def active_stage_rows(choice, stages=None):
    """Return the STAGES rows enabled by the given Jenkins ``STAGES`` choice."""
    stages = stages if stages is not None else STAGES
    enabled = set(enabled_params_for_choice(choice, stages))
    return [row for row in stages if row.get("param") in enabled]


def shard_plan(choice, stage_filter="", stages=None):
    """Expand the active rows for ``choice`` into the build pipeline shard plan.

    ``shards`` is the per-shard branch list after applying ``stage_filter``
    (substring match on the per-shard display name). ``candidates`` is the
    unfiltered active-row display names, so a STAGE_FILTER typo can be reported
    with the list of stages it could have matched. ``zipArtifacts`` is the
    per-(row, board) build-to-HW handoff list, flattened over active rows
    (not per shard, not filtered) to mirror the aggregate-time walk.
    """
    rows = active_stage_rows(choice, stages)
    all_shards = []
    for row in rows:
        all_shards.extend(_shard_entry(row))
    if stage_filter:
        shards = [s for s in all_shards if stage_filter in s["stage"]]
    else:
        shards = all_shards
    zip_artifacts = []
    for row in rows:
        zip_art = row.get("zipArtifacts")
        if not zip_art:
            continue
        for board in zip_art.get("boards", []):
            zip_artifacts.append(
                {
                    "stage": row["stage"],
                    "hwTestType": zip_art["hwTestType"],
                    "board": board,
                }
            )
    return {
        "shards": shards,
        "candidates": [row["stage"] for row in rows],
        "zipArtifacts": zip_artifacts,
    }


# =============================================================================
# Sharding (group -> shard assignment)
# =============================================================================


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


def _median_index(values):
    """Median by index without importing statistics. Returns 0.0 on empty."""
    s = sorted(v for v in values if v > 0.0)
    return s[len(s) // 2] if s else 0.0


def _entry_median_seconds(entry):
    """Extract the median seconds estimate from a master/snapshot group entry."""
    if not isinstance(entry, dict):
        return 0.0
    samples = entry.get("samples")
    if isinstance(samples, list) and samples:
        return _median_index(float(s or 0.0) for s in samples if s is not None)
    return 0.0


def load_group_weights(path):
    """Return ``{group_name: seconds}`` from a timing state file."""
    data = read_json(path, default={})
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, dict):
        return {}
    weights = {}
    for key, val in groups.items():
        seconds = _entry_median_seconds(val)
        if seconds > 0.0:
            weights[str(key)] = seconds
    return weights


def weights_with_fallback(weights):
    # 1.0 placeholder for cold-start so a brand-new master still produces a
    # reproducible round-robin assignment.
    known = sorted(w for w in weights.values() if w > 0.0)
    return known[len(known) // 2] if known else 1.0


def assign_groups_to_shards(group_keys, num_shards, weights=None, pins=None):
    """Assign group keys to shard ids.

    Pins win first. Unpinned groups go via LPT-greedy bin packing when there
    is timing signal, otherwise round-robin over sorted group keys so
    fallback placement is reproducible and balanced by group count.

    Raises ``ValueError`` for ``num_shards < 1`` or any pin outside
    ``[0, num_shards)``.
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
        # No timing signal: balance by group count, but seed from any pinned
        # load so pins are respected. Placing each group on the least-loaded
        # shard (ties broken by index) stays deterministic and matches plain
        # round-robin when there are no pins.
        for key in unpinned:
            shard = min(range(num_shards), key=lambda s: (shard_load[s], s))
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


# =============================================================================
# Reports I/O (read/write JSON, merge maps, per-shard summary)
# =============================================================================


def read_json(path, default=None):
    if not path:
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        # File present but unreadable or malformed: warn so a corrupt master
        # does not silently degrade sharding to round-robin.
        print(
            "ci_sharding read_json: %s: %s: %s" % (path, exc.__class__.__name__, exc),
            file=sys.stderr,
        )
        return default


def write_json_atomic(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    # exist_ok=True so two concurrent first-time callers on a shared NFS root
    # cannot race on mkdir.
    os.makedirs(parent, exist_ok=True)
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
            "A trusted full build refreshes the timing master from these observations."
            % SLOW_FACTOR
        )
    return 0


# =============================================================================
# Timing master state machine
# =============================================================================
# The persistent master at
# ${FINN_CI_NFS_ROOT}/_ci_state/<jobKey>/ci_timings_master.json holds the last
# MAX_SAMPLES observations per group. Trusted full-matrix builds append; the
# bin packer reads the median. Schema v2:
#
#   {"schema_version": 2, "updated_at": str, "last_update": {...},
#    "groups": {<name>: {"samples": [s1, ..., sMAX_SAMPLES], "count": int,
#                        "last_seen_*": ...}}}
#
# The per-build snapshot written by ``prepare_timing_snapshot`` is the same
# shape so a snapshot can be inspected with the same tools as the master. An
# unrecognised schema_version is logged and treated as empty, which makes old
# or corrupt state degrade to round-robin sharding rather than crashing.

SCHEMA_VERSION = 2


def normalise_master(data):
    """Coerce arbitrary input to the master schema (drops unknown top-level keys)."""
    if not isinstance(data, dict):
        data = {}
    schema_version = data.get("schema_version")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        print(
            "ci_sharding normalise_master: unrecognised schema_version %r, "
            "treating as empty (expected %d)" % (schema_version, SCHEMA_VERSION),
            file=sys.stderr,
        )
        data = {}
    groups = data.get("groups")
    if not isinstance(groups, dict):
        groups = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": data.get("updated_at"),
        "groups": dict(groups),
    }


def _samples_from_entry(entry):
    """Return the samples list from a master entry (empty if absent or malformed)."""
    if not isinstance(entry, dict):
        return []
    samples = entry.get("samples")
    if not isinstance(samples, list):
        return []
    return [float(s or 0.0) for s in samples if s is not None]


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
            previous = observed.get(name)
            if previous and seconds <= previous["seconds"]:
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


def _apply_per_group_update(name, observed_seconds, current_entry, metadata):
    """Append the observation to ``current_entry`` and trim to MAX_SAMPLES."""
    current_entry = current_entry or {}
    prior_samples = _samples_from_entry(current_entry)
    new_samples = (prior_samples + [round(float(observed_seconds), 3)])[-MAX_SAMPLES:]
    return {
        "samples": new_samples,
        "count": metadata.get("count", int(current_entry.get("count", 0) or 0)),
        "last_seen_job": metadata.get("last_seen_job"),
        "last_seen_build": metadata.get("last_seen_build"),
        "last_seen_stage": metadata.get("last_seen_stage"),
        "last_seen_stash": metadata.get("last_seen_stash"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def update_master(reports_dir, master_path, out_path, update_persistent=False, metadata=None):
    """Merge observed timings into a per-build preview and optionally the master.

    Every call writes ``out_path``. Persistent master updates are opt-in via
    ``update_persistent`` (trusted full-matrix builds only); preview mode
    leaves the on-disk master untouched. Every observation in this build is
    appended to its group's samples and the window is trimmed to MAX_SAMPLES.
    """
    observed_raw = observed_groups_from_reports(reports_dir)
    metadata = metadata or {}
    observed_seconds = {name: float(entry["seconds"]) for name, entry in observed_raw.items()}
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def apply(current, persist=False):
        master = normalise_master(current)
        master["updated_at"] = now_iso
        for name, seconds in observed_seconds.items():
            master["groups"][name] = _apply_per_group_update(
                name, seconds, master["groups"].get(name), observed_raw[name]
            )
        master["last_update"] = {
            "job": metadata.get("job"),
            "build": metadata.get("build"),
            "persistent_update": bool(persist),
            "observed_groups": len(observed_seconds),
        }
        return master

    persistent_updated = False
    if master_path and update_persistent:
        master = locked_update(master_path, lambda cur: apply(cur, persist=True))
        persistent_updated = True
    elif master_path:
        master = apply(read_json(master_path, default={}), persist=False)
    else:
        master = apply({}, persist=False)
    if out_path:
        write_json_atomic(out_path, master)
    print(
        "ci_sharding update: %d observed, %d in master, persistent_update=%s"
        % (len(observed_seconds), len(master.get("groups", {})), persistent_updated)
    )
    _report_regressions(master_path, observed_seconds)
    return 0


# Regression canary: flag groups whose latest observation is >= REGRESSION_FACTOR
# times their prior median, so a real slowdown does not hide behind the
# median-of-five window's smoothing.
REGRESSION_FACTOR = 2.0


def _report_regressions(master_path, observed_seconds):
    """Print one line per group whose latest observation >= REGRESSION_FACTOR x prior median."""
    if not master_path or not observed_seconds:
        return
    prior = normalise_master(read_json(master_path, default={}))
    flagged = []
    for name, seconds in observed_seconds.items():
        prior_median = _entry_median_seconds(prior["groups"].get(name))
        if prior_median > 0.0 and seconds >= REGRESSION_FACTOR * prior_median:
            flagged.append((name, seconds, prior_median))
    if not flagged:
        return
    flagged.sort(key=lambda row: row[1] / row[2], reverse=True)
    parts = ["%s=%.1fs vs %.1fs" % (name, sec, med) for name, sec, med in flagged]
    print(
        "ci_sharding update: regressions vs prior median (>=%.1fx): %s"
        % (REGRESSION_FACTOR, "; ".join(parts))
    )


def prepare_timing_snapshot(master_path, snapshot_path):
    """Copy the persistent master to a per-build snapshot for shard consumption.

    Cold start writes an empty snapshot so sharding falls back to
    deterministic round-robin until the first build populates the master.
    """
    master = read_json(master_path, default=None)
    master = normalise_master(master)
    write_json_atomic(snapshot_path, master)
    print(
        "ci_sharding prepare: wrote %s with %d group(s)"
        % (snapshot_path, len(master.get("groups", {})))
    )
    return 0


def _backup_if_corrupt(master_path):
    """Rename a non-empty unparseable master aside. Returns True if backed up.

    Keeps a single ``<master>.corrupt-<epoch>`` sibling per incident. If
    corruption recurs across runs, each invocation leaves its own backup;
    operators clean up by hand.
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
            return True
        except OSError as exc:
            print(
                "ci_sharding locked_update: could not back up corrupt %s: %s" % (master_path, exc),
                file=sys.stderr,
            )
            return False


def locked_update(master_path, update_fn):
    """Apply ``update_fn`` to the master JSON under an NFS-safe directory lock.

    flock/fcntl advisory locks are local-only on NFS and give no cross-agent
    exclusion, so the timing-master update serialises on the mkdir-based
    nfs_dir_lock instead. The lock dir (``.<basename>.lock.d``) lives next to
    the master; a crashed holder is reclaimed by the lock's stale TTL, so a
    dead build never wedges the master.
    """
    if not master_path:
        return update_fn({})
    parent = os.path.dirname(os.path.abspath(master_path))
    # exist_ok=True so two concurrent first-time callers on a shared NFS root
    # cannot race on mkdir.
    os.makedirs(parent, exist_ok=True)
    base = os.path.basename(master_path)
    lock_path = os.path.join(parent, "." + base + ".lock.d")
    with nfs_dir_lock(lock_path):
        _backup_if_corrupt(master_path)
        current = read_json(master_path, default={})
        updated = update_fn(current)
        write_json_atomic(master_path, updated)
    return updated


# =============================================================================
# Build-to-HW handoff resolver
# =============================================================================


def resolve_build_zips(artifact_dir, job_key, test_types, boards, build_dir=""):
    """Resolve ``(testType, board) -> {"zip": ..., "buildDir": ...}`` per pair.

    Walks ``${artifact_dir}/ci_runs/<job_key>/`` newest-first and picks, per
    board, the highest-numbered build whose ``zips/<testType>/<board>.zip.READY``
    sibling is present. Boards with no READY come back as ``{}``.

    ``build_dir`` pins every pair to that single directory. A missing READY
    there is reported per-board but does not abort the call.
    """
    out = {tt: {b: {} for b in boards} for tt in test_types}
    if build_dir:
        for tt in test_types:
            for b in boards:
                zip_path = os.path.join(build_dir, "zips", tt, "%s.zip" % b)
                if os.path.isfile(zip_path) and os.path.isfile(zip_path + ".READY"):
                    build = os.path.basename(os.path.normpath(build_dir))
                    out[tt][b] = {
                        "zip": zip_path,
                        "buildDir": build_dir,
                        "build": build,
                        "latestBuild": build,
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
    remaining = {(tt, b) for tt in test_types for b in boards}
    for build in candidates:
        if not remaining:
            break
        candidate_dir = os.path.join(job_root, build)
        for tt, b in list(remaining):
            zip_path = os.path.join(candidate_dir, "zips", tt, "%s.zip" % b)
            if os.path.isfile(zip_path) and os.path.isfile(zip_path + ".READY"):
                out[tt][b] = {"zip": zip_path, "buildDir": candidate_dir, "build": build}
                remaining.discard((tt, b))
    for tt in test_types:
        selected = [entry for entry in out.get(tt, {}).values() if entry.get("build")]
        if not selected:
            continue
        for entry in selected:
            entry["latestBuild"] = latest_build
            entry["fallback"] = str(entry["build"]) != latest_build
    return out


# =============================================================================
# Retention / pruning
# =============================================================================


def prune_numeric_builds(parent, current_build, retain_n, max_age_days, dry_run, *, tag):
    """Delete numeric-named subdirs of ``parent`` outside the newest ``retain_n``.

    Tolerant of concurrent deletion on a shared NFS parent at both probe
    sites: a vanished entry during the ``os.path.getmtime`` age check and
    during ``shutil.rmtree`` is treated as already-pruned and skipped.
    ``current_build`` must be integer-like. Returns the number of
    directories matched for deletion.
    """
    retain_n = int(retain_n)
    max_age_days = int(max_age_days)
    if retain_n < 1:
        raise ValueError("retain_n must be >= 1")
    try:
        current_build_int = int(str(current_build))
    except (TypeError, ValueError):
        raise ValueError("current_build must be an integer-like string, got %r" % (current_build,))
    if not os.path.isdir(parent):
        return 0
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    # Compare by int so an on-disk "0123" matches a BUILD_NUMBER of "123".
    nums = sorted((d for d in os.listdir(parent) if d.isdigit()), key=int)
    keep = {int(n) for n in nums[-retain_n:]}
    keep.add(current_build_int)
    matched = 0
    for num in nums:
        if int(num) in keep:
            continue
        path = os.path.join(parent, num)
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
    try:
        return str(int(str(value)))
    except (TypeError, ValueError):
        raise ValueError(
            "ci_sharding %s: current_build must be an integer-like string, got %r" % (tag, value)
        )


def _prune_subtree(parent, tag, current_build, retain_n, max_age_days, *, dry_run):
    current_build = _coerce_current_build(current_build, tag)
    if not os.path.isdir(parent):
        print("ci_sharding %s: %s not present, skipping" % (tag, parent))
        return 0
    matched = prune_numeric_builds(parent, current_build, retain_n, max_age_days, dry_run, tag=tag)
    print(
        "ci_sharding %s: done (parent=%s current=%s retain_n=%s "
        "max_age_days=%s dry_run=%s matched=%d)"
        % (tag, parent, current_build, retain_n, max_age_days, int(dry_run), matched)
    )
    return 0


def prune_images(shared_dir, job_key, current_build, retain_n, max_age_days, dry_run=False):
    parent = os.path.join(shared_dir, job_key)
    return _prune_subtree(
        parent, "prune-images", current_build, retain_n, max_age_days, dry_run=dry_run
    )


def prune_artifacts(artifact_dir, job_key, current_build, retain_n, max_age_days, dry_run=False):
    """Rotate ${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<job_key>/ for this build job.

    HW always resolves to the newest READY zip per board, so deleting an
    older build cannot strand a HW shard.
    """
    parent = os.path.join(artifact_dir, "ci_runs", job_key)
    return _prune_subtree(
        parent, "prune-artifacts", current_build, retain_n, max_age_days, dry_run=dry_run
    )


SNAPSHOT_FILE_RE = re.compile(r"^build_(\d+)_timings_input\.json$")


def prune_snapshots(state_root, job_key, current_build, retain_n, max_age_days, dry_run=False):
    """Rotate per-build timing snapshot files under ``_ci_state/<job_key>/``.

    Snapshots are named ``build_<N>_timings_input.json`` alongside the
    persistent ``ci_timings_master.json``. Only the build-numbered files
    are eligible for pruning. Tolerant of concurrent unlink on a shared
    NFS parent (vanished entries are treated as already-pruned).
    """
    retain_n = int(retain_n)
    max_age_days = int(max_age_days)
    if retain_n < 1:
        raise ValueError("retain_n must be >= 1")
    current_build = _coerce_current_build(current_build, "prune-snapshots")
    current_build_int = int(current_build)
    parent = os.path.join(state_root, job_key)
    tag = "prune-snapshots"
    if not os.path.isdir(parent):
        print("ci_sharding %s: %s not present, skipping" % (tag, parent))
        return 0
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    candidates = []
    for name in os.listdir(parent):
        m = SNAPSHOT_FILE_RE.match(name)
        if m:
            candidates.append((int(m.group(1)), name))
    candidates.sort()
    keep = {n for n, _ in candidates[-retain_n:]}
    keep.add(current_build_int)
    matched = 0
    for num, name in candidates:
        if num in keep:
            continue
        path = os.path.join(parent, name)
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
                os.unlink(path)
            except FileNotFoundError:
                pass
    print(
        "ci_sharding %s: done (parent=%s current=%s retain_n=%s "
        "max_age_days=%s dry_run=%s matched=%d)"
        % (tag, parent, current_build, retain_n, max_age_days, int(dry_run), matched)
    )
    return 0


def prune_pip_cache(root, keep, max_age_days, dry_run=False):
    """Delete cache-key subdirs of ``root`` older than ``max_age_days``, keeping
    ``keep``. Mirrors the local-setup stage's pip-cache GC, but as tested
    Python instead of an inline find. The directory's own mtime is the age key
    (an actively reused cache dir is bumped on write, so it survives).
    """
    max_age_days = int(max_age_days)
    if not os.path.isdir(root):
        return 0
    keep_abs = os.path.abspath(keep) if keep else None
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    matched = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if keep_abs and os.path.abspath(path) == keep_abs:
            continue
        if max_age_days > 0:
            try:
                if os.path.getmtime(path) >= cutoff:
                    continue
            except FileNotFoundError:
                continue
        matched += 1
        if dry_run:
            print("ci_sharding prune-pip-cache: would delete %s" % path)
        else:
            print("ci_sharding prune-pip-cache: deleting %s" % path)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
    return matched


# =============================================================================
# LSF orphan-job parsing
# =============================================================================
# The reaper in the Jenkinsfile still owns the "is this build still running"
# decision (it needs the Jenkins API) and the bkill call, but the fragile
# job-name parsing lives here so a bjobs format change is a one-file, tested
# fix instead of a Groovy regex tweak.

LSF_JOB_BUILD_RE = re.compile(r"^(\d+)_")


def parse_lsf_jobs(prefix, raw):
    """Map ``{build_number: [jobid, ...]}`` from ``bjobs`` output.

    ``raw`` is either the JSON document emitted by ``bjobs -json -o 'jobid
    job_name'`` or the whitespace-separated ``jobid job_name`` text form
    (``-noheader``); both are accepted so the caller does not have to know
    which the local LSF build supports. Job names that do not start with
    ``prefix`` followed by ``<build>_`` are ignored.
    """
    records = _lsf_records(raw)
    out = collections.OrderedDict()
    for jobid, name in records:
        if not jobid or not name or not name.startswith(prefix):
            continue
        tail = name[len(prefix) :]
        m = LSF_JOB_BUILD_RE.match(tail)
        if not m:
            continue
        out.setdefault(m.group(1), []).append(jobid)
    return out


def _lsf_records(raw):
    """Yield ``(jobid, job_name)`` pairs from bjobs JSON or text output."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw[0] in "{[":
        try:
            doc = json.loads(raw)
        except ValueError:
            return []
        records = doc.get("RECORDS", doc) if isinstance(doc, dict) else doc
        pairs = []
        for rec in records or []:
            if isinstance(rec, dict):
                jobid = str(rec.get("JOBID", "")).strip()
                name = str(rec.get("JOB_NAME", "")).strip()
                pairs.append((jobid, name))
        return pairs
    pairs = []
    for line in raw.split("\n"):
        toks = line.split(None, 1)
        if len(toks) < 2:
            continue
        pairs.append((toks[0].strip(), toks[1].strip()))
    return pairs


# =============================================================================
# CLI
# =============================================================================


def main(argv=None):
    """CLI entry point. Catches validate_* failures so a malformed STAGES row
    surfaces in the Validate Jenkins console as a one-line ``ci_sharding:``
    message instead of a Python traceback.
    """
    try:
        return _dispatch(argv)
    except (ValueError, AssertionError) as exc:
        print("ci_sharding: %s" % exc, file=sys.stderr)
        return 2


def _dispatch(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("stage-choices-json")
    # The HW pipeline's Validate reads this one bundle instead of three
    # subcommands, so the BOARDS/STAGES tables are validated once per HW build.
    sub.add_parser("hw-config-json")

    # validate-config is the one entry point the Validate stage in Jenkins
    # delegates to. Folds enabled_params / retention / job_key / shard_plan
    # into a single subprocess and runs validate_config() first so a malformed
    # row or orphan zipArtifact board fails Validate loudly.
    p = sub.add_parser("validate-config")
    p.add_argument("--choice", required=True)
    p.add_argument("--job-name", required=True)
    p.add_argument("--stage-filter", default="")

    p = sub.add_parser("job-key")
    p.add_argument("name")

    p = sub.add_parser("lsf-parse-jobs")
    p.add_argument("--prefix", required=True)

    p = sub.add_parser("prune-pip-cache")
    p.add_argument("root")
    p.add_argument("keep")
    p.add_argument("max_age_days", type=int)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("prepare")
    p.add_argument("--master", required=True)
    p.add_argument("--snapshot", required=True)

    p = sub.add_parser("summarize")
    p.add_argument("reports_dir")

    p = sub.add_parser("update")
    p.add_argument("--reports", required=True)
    p.add_argument("--master", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--job", default="")
    p.add_argument("--build", default="")
    p.add_argument("--update-master", action="store_true")

    p = sub.add_parser("merge-maps")
    p.add_argument("reports_dir")

    p = sub.add_parser("resolve-build-zips")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--job-key", required=True)
    p.add_argument(
        "--tests",
        required=True,
        help="Comma-separated HW test types (e.g. bnn_build_sanity,bnn_build_full)",
    )
    p.add_argument("--boards", required=True, help="Comma-separated board names")
    p.add_argument("--build-dir", default="", help="Optional explicit build directory override")

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

    p = sub.add_parser("prune-snapshots")
    p.add_argument("state_root")
    p.add_argument("job_key")
    p.add_argument("current_build")
    p.add_argument("retain_n", type=int)
    p.add_argument("max_age_days", type=int)
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "stage-choices-json":
        print(json.dumps(jenkins_stage_choices()))
        return 0
    if args.cmd == "hw-config-json":
        validate_config()
        print(
            json.dumps(
                {
                    "shards": hw_shards(),
                    "test_types": hw_test_types(),
                    "labels": hw_test_type_labels(),
                }
            )
        )
        return 0
    if args.cmd == "validate-config":
        validate_config()
        print(
            json.dumps(
                {
                    "enabled_params": enabled_params_for_choice(args.choice),
                    "retention": RETENTION,
                    "job_key": job_key(args.job_name),
                    "shard_plan": shard_plan(args.choice, args.stage_filter),
                }
            )
        )
        return 0
    if args.cmd == "job-key":
        print(job_key(args.name))
        return 0
    if args.cmd == "lsf-parse-jobs":
        print(json.dumps(parse_lsf_jobs(args.prefix, sys.stdin.read())))
        return 0
    if args.cmd == "prune-pip-cache":
        prune_pip_cache(args.root, args.keep, args.max_age_days, args.dry_run)
        return 0
    if args.cmd == "prepare":
        return prepare_timing_snapshot(args.master, args.snapshot)
    if args.cmd == "summarize":
        return summarize_timings(args.reports_dir)
    if args.cmd == "update":
        return update_master(
            args.reports,
            args.master,
            args.out,
            update_persistent=args.update_master,
            metadata={
                "job": args.job,
                "build": args.build,
            },
        )
    if args.cmd == "merge-maps":
        return merge_maps(args.reports_dir)
    if args.cmd == "resolve-build-zips":
        tests = [t for t in args.tests.split(",") if t]
        boards = [b for b in args.boards.split(",") if b]
        result = resolve_build_zips(args.artifact_dir, args.job_key, tests, boards, args.build_dir)
        print(json.dumps(result))
        return 0
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
    if args.cmd == "prune-snapshots":
        return prune_snapshots(
            args.state_root,
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
