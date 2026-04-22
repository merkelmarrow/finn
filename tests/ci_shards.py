# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for FINN parallel CI stage config.

Read by ``docker/jenkins/Jenkinsfile`` (via ``python3 -c ... json.dumps`` +
``readJSON``) AND by ``tests/conftest.py`` for the ``--which-shard``
maintainer lookup. Adding or changing a row here is the only edit required
to add a new marker, board, or shard split.

Rows are intentionally fully-explicit: there is NO expansion logic
anywhere. Adding a BNN board means one more literal row below. This is a
deliberate trade -- a handful of extra characters per row in exchange for
zero mirrored logic between Groovy (Jenkinsfile) and Python (conftest /
maintainer tooling). Iter-7's earlier YAML + ``zipAllBoards`` + per-board
end2end expansion draft was discarded precisely because that expansion
lived in two places and drifted.

Schema (every key optional except ``param``, ``stage``, ``marker``,
``shards``, ``workers``):

    param     : Jenkins boolean param gating the row
                (one of: "sanity", "fpgadataflow", "end2end")
    stage     : human-readable Jenkins stage label
    marker    : pytest marker expression; must match ``^[A-Za-z0-9_ ]+$``
                (validateShards() enforces this; the regex lets us
                interpolate ``marker`` straight into shell in Groovy)
    shards    : int >= 1, pytest-xdist shard split count
    workers   : int >= 1, ``-n`` worker count per shard
    skipWhen  : optional other param; row is skipped when both are true
    coverage  : optional bool, enable coverage reporting
    distMode  : optional xdist dist mode (e.g. "loadgroup")
    zipBoards : optional list of boards; passed as ``--board`` per entry
                and drives ``assertZipBoardsEmitted()`` in aggregation
"""

STAGES = [
    # Sanity - Build Hardware rebuilds the same scenarios covered by the
    # BNN rows, so skip it when end2end also runs.
    {"param": "sanity", "stage": "Sanity - Build Hardware",
     "marker": "sanity_bnn",
     "shards": 1, "workers": 1, "skipWhen": "end2end",
     "zipBoards": ["U250", "Pynq-Z1", "ZCU104", "KV260_SOM"]},
    {"param": "sanity", "stage": "Sanity - Unit Tests",
     "marker": "util or brevitas_export or streamline or transform or notebooks",
     "shards": 1, "workers": 8, "coverage": True},
    {"param": "fpgadataflow", "stage": "fpgadataflow",
     "marker": "fpgadataflow",
     "shards": 2, "workers": 8, "coverage": True},
    # iter-6: end2end has 3 xdist_groups (cybsec, ext_weights, mobilenet);
    # shards=3 lets conftest.py isolate mobilenet on its own shard.
    {"param": "end2end", "stage": "End2end",
     "marker": "end2end",
     "shards": 3, "workers": 6, "distMode": "loadgroup"},
    # Per-board BNN end2end rows. Shard counts were rebalanced in iter-6
    # against observed .timings.json (over-sharded boards wasted wall-clock
    # on guard-skipped wbits>abits scenarios).
    {"param": "end2end", "stage": "BNN U250",
     "marker": "bnn_u250",
     "shards": 2, "workers": 2, "distMode": "loadgroup",
     "zipBoards": ["U250"]},
    {"param": "end2end", "stage": "BNN Pynq-Z1",
     "marker": "bnn_pynq",
     "shards": 3, "workers": 2, "distMode": "loadgroup",
     "zipBoards": ["Pynq-Z1"]},
    {"param": "end2end", "stage": "BNN ZCU104",
     "marker": "bnn_zcu104",
     "shards": 2, "workers": 4, "distMode": "loadgroup",
     "zipBoards": ["ZCU104"]},
    {"param": "end2end", "stage": "BNN KV260",
     "marker": "bnn_kv260",
     "shards": 2, "workers": 2, "distMode": "loadgroup",
     "zipBoards": ["KV260_SOM"]},
]


def stage_slug(stage):
    # Mirrors shardStashName() in docker/jenkins/Jenkinsfile. The Jenkins
    # helper lowercases the stage label, collapses non-alnum runs to "_",
    # and trims leading/trailing underscores.
    import re
    return re.sub(r"^_|_$", "", re.sub(r"[^a-z0-9]+", "_", stage.lower()))


def stash_name(stage, shards, shard_id):
    # Exact mirror of Jenkinsfile shardStashName(); --which-shard prints
    # this value so maintainers can pass it to STAGES=<substring> for a
    # single-shard debug re-run.
    base = stage_slug(stage)
    if shards <= 1:
        return base
    return "%s_%d" % (base, shard_id + 1)
