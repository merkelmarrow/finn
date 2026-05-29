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
    """Pull module-level ``_is_pkg_src`` out of finn_xsi/adapter.py via AST.

    ``finn_xsi.adapter`` eagerly imports ``finn_xsi.sim_engine``, which
    in turn imports the ``xsi`` C extension built only inside the FINN
    Docker image at runtime. AST extraction lets the predicate be tested
    against the production source without that build artefact, so the
    util shard still runs in a fresh checkout that has not yet built
    finn_xsi.
    """
    tree = ast.parse(open(ADAPTER_PATH).read())
    for node in tree.body:
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
