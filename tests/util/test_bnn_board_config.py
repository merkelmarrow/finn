############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import ast
import itertools
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JENKINS_DIR = os.path.join(REPO_ROOT, "docker", "jenkins")
if JENKINS_DIR not in sys.path:
    sys.path.insert(0, JENKINS_DIR)

import ci_sharding  # noqa: E402

END2END_BNN_TEST = os.path.join(REPO_ROOT, "tests", "end2end", "test_end2end_bnn_pynq.py")


def _literal_assignment(name):
    """Parse the BNN test file's top-level literal assignments via AST.

    Lets the cross-check inspect parametrisation constants without importing
    the test module (which would pull in qonnx / torch and inflate the
    util-test surface).
    """
    tree = ast.parse(open(END2END_BNN_TEST).read())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError("could not find %s in %s" % (name, END2END_BNN_TEST))


@pytest.mark.util
def test_bnn_board_tables_stay_in_sync():
    marker_by_board = _literal_assignment("_BNN_MARKER_BY_BOARD")
    assert list(ci_sharding.BOARDS) == list(ci_sharding.TEST_BOARDS)
    assert set(ci_sharding.TEST_BOARDS) == set(marker_by_board)


@pytest.mark.util
def test_bnn_xdist_group_names_are_unique_lightweight():
    sanity_configs = _literal_assignment("_SANITY_BNN_CONFIGS")
    marker_by_board = _literal_assignment("_BNN_MARKER_BY_BOARD")
    wbits = _literal_assignment("_BNN_WBITS")
    abits = _literal_assignment("_BNN_ABITS")
    topologies = _literal_assignment("_BNN_TOPOLOGY")

    groups = []
    for w, a, topology, board in sanity_configs:
        groups.append("sanity_bnn_w%d_a%d_%s_%s" % (w, a, topology, board))
    for board in ci_sharding.TEST_BOARDS:
        marker = marker_by_board[board]
        for w, a, topology in itertools.product(wbits, abits, topologies):
            groups.append("%s_w%d_a%d_%s_%s" % (marker, w, a, topology, board))

    assert len(groups) == len(set(groups))
