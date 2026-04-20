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

"""Sharding and coverage-guard plugin for FINN CI.

Adding a test requires nothing beyond the usual `@pytest.mark.<marker>`:
the Jenkinsfile runs each marker with `--num-shards N --shard-id I`, and
the hook below does deterministic hash-based sharding over the collected
items. No `.test_durations`, no per-file allowlist, no rebalance script.

If `-m <marker>` ever selects zero tests while sharding is requested, the
hook raises — catching "silently skipped in CI" at collection time rather
than allowing the skip to ship.
"""
import hashlib

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("finn-ci-sharding")
    group.addoption("--num-shards", type=int, default=0,
                    help="Split the collected test set into N deterministic shards.")
    group.addoption("--shard-id", type=int, default=0,
                    help="Which shard (0-indexed) to run. Requires --num-shards.")


def _shard_key(item):
    # Tests sharing an `xdist_group` form a sequence (downstream steps use
    # `load_test_checkpoint_or_skip` against upstream outputs), so they must
    # land in the SAME shard or the chain silently skips. Hash by group name
    # when set, otherwise by nodeid.
    for mark in item.iter_markers(name="xdist_group"):
        if mark.args:
            return str(mark.args[0])
        name = mark.kwargs.get("name")
        if name is not None:
            return str(name)
    return item.nodeid


def _shard_of(key, num_shards):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # trylast=True ensures pytest's own `-m` deselection has already reduced
    # `items` before we hash-shard it, so each shard sees only tests matching
    # the marker expression.
    num_shards = config.getoption("--num-shards")
    if num_shards <= 0:
        return
    shard_id = config.getoption("--shard-id")
    if not 0 <= shard_id < num_shards:
        raise pytest.UsageError(
            "--shard-id=%d out of range for --num-shards=%d" % (shard_id, num_shards)
        )
    if not items:
        # Zero items before sharding means the `-m` expression selected nothing.
        # In CI that is a silent-skip footgun (marker typo, stale PARALLEL_SHARDS
        # row, etc.), so we fail loudly instead.
        raise pytest.UsageError(
            "no tests collected for this marker; CI shard configuration is "
            "out of sync with the test markers"
        )
    items[:] = [it for it in items if _shard_of(_shard_key(it), num_shards) == shard_id]
