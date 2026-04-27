# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for FINN parallel CI stage config.

Loaded by ``docker/jenkins/Jenkinsfile`` (via ``python3 -c ... json.dumps`` +
``readJSON``) and by ``tests/conftest.py`` for ``--which-shard``.

Schema (only ``param``/``stage``/``marker``/``shards``/``workers`` required):

    param     : Jenkins boolean param gating the row
    stage     : human-readable Jenkins stage label
    marker    : pytest marker expression matching ``^[A-Za-z0-9_ ]+$``
                (interpolated into the pytest -m shell argument)
    shards    : int >= 1, conftest sharding split
    workers   : int >= 1, pytest-xdist ``-n`` count per shard
    skipWhen  : optional other param -- row skipped when both are true
    coverage  : optional bool toggling coverage reporting
    distMode  : optional xdist dist mode (e.g. "loadgroup")
    zipBoards : optional list of boards driving ``assertZipBoardsEmitted``
"""

STAGES = [
    # Sanity - Build Hardware rebuilds scenarios covered by the BNN rows,
    # so skip it when end2end also runs.
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
    # end2end has 3 xdist_groups (cybsec, ext_weights, mobilenet) and
    # shards=3 lets conftest isolate mobilenet on its own shard.
    {"param": "end2end", "stage": "End2end",
     "marker": "end2end",
     "shards": 3, "workers": 6, "distMode": "loadgroup"},
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


def stash_name(stage, shards, shard_id):
    """Mirrors ``shardStashName()`` in docker/jenkins/Jenkinsfile."""
    import re
    base = re.sub(r"^_|_$", "", re.sub(r"[^a-z0-9]+", "_", stage.lower()))
    return base if shards <= 1 else "%s_%d" % (base, shard_id + 1)
