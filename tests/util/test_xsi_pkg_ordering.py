############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(REPO_ROOT, "finn_xsi", "finn_xsi", "adapter.py")


def _extract_pkg_predicate():
    """Pull ``_is_pkg_src`` out of compile_sim_obj via AST and exec it.

    Importing finn_xsi.adapter would drag finn (and qonnx/torch) into the
    util shard collection. AST extraction keeps this test stdlib-only and
    locks the predicate down regardless of any future surrounding refactor.
    """
    tree = ast.parse(open(ADAPTER_PATH).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_is_pkg_src":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"os": os}
            exec(compile(module, ADAPTER_PATH, "exec"), namespace)
            return namespace["_is_pkg_src"]
    raise AssertionError("could not find _is_pkg_src in %s" % ADAPTER_PATH)


@pytest.mark.util
def test_is_pkg_src_matches_pkg_basenames():
    pred = _extract_pkg_predicate()
    # the prior substring-matched form only caught swg_pkg / mvu_pkg, so
    # exercise both the legacy names and a previously-unrecognised one
    assert pred("/path/to/swg_pkg.sv")
    assert pred("/path/to/mvu_pkg.v")
    assert pred("/path/to/some_other_pkg.sv")
    assert pred("foo_pkg.sv")


@pytest.mark.util
def test_is_pkg_src_does_not_match_non_pkg_or_header_files():
    pred = _extract_pkg_predicate()
    # header files (.svh / .vh) are excluded by the .prj writer further down,
    # so the predicate must not promote them into the package partition
    assert not pred("/path/to/foo.sv")
    assert not pred("/path/to/foo.v")
    assert not pred("/path/to/foo.vhd")
    assert not pred("/path/to/swg_pkg.svh")
    assert not pred("/path/to/swg_pkg_top.sv")
    assert not pred("/path/to/_pkg.svh")


@pytest.mark.util
def test_pkg_partition_places_packages_first_and_preserves_order():
    pred = _extract_pkg_predicate()
    source_list = [
        "/path/a.sv",
        "/path/swg_pkg.sv",
        "/path/b.sv",
        "/path/mvu_pkg.sv",
        "/path/c.sv",
        "/path/some_other_pkg.v",
    ]
    pkg_srcs = [s for s in source_list if pred(s)]
    other_srcs = [s for s in source_list if not pred(s)]
    out = pkg_srcs + other_srcs

    assert pkg_srcs == [
        "/path/swg_pkg.sv",
        "/path/mvu_pkg.sv",
        "/path/some_other_pkg.v",
    ]
    assert other_srcs == ["/path/a.sv", "/path/b.sv", "/path/c.sv"]
    assert out[: len(pkg_srcs)] == pkg_srcs
    assert out[len(pkg_srcs) :] == other_srcs
