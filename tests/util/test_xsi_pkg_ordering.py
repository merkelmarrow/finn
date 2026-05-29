############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import os
import sys

# finn_xsi is not pip-installed (FINN packages only src/); finn.xsi inserts
# this dir on sys.path at runtime. srcutil has no xsi C-extension import, so
# adding the dir and importing it is safe in a fresh checkout that has not
# built finn_xsi yet (unlike finn_xsi.adapter, which imports sim_engine).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_XSI_ROOT = os.path.join(os.environ.get("FINN_ROOT", _REPO_ROOT), "finn_xsi")
if _XSI_ROOT not in sys.path:
    sys.path.insert(0, _XSI_ROOT)

from finn_xsi.srcutil import is_pkg_src  # noqa: E402


@pytest.mark.util
def test_is_pkg_src_matches_pkg_basenames():
    # the prior substring-matched form only caught swg_pkg / mvu_pkg, so
    # exercise both the legacy names and a previously-unrecognised one
    assert is_pkg_src("/path/to/swg_pkg.sv")
    assert is_pkg_src("/path/to/mvu_pkg.v")
    assert is_pkg_src("/path/to/some_other_pkg.sv")
    assert is_pkg_src("foo_pkg.sv")


@pytest.mark.util
def test_is_pkg_src_does_not_match_non_pkg_or_header_files():
    # header files (.svh / .vh) are excluded by the .prj writer further down,
    # so the predicate must not promote them into the package partition
    assert not is_pkg_src("/path/to/foo.sv")
    assert not is_pkg_src("/path/to/foo.v")
    assert not is_pkg_src("/path/to/foo.vhd")
    assert not is_pkg_src("/path/to/swg_pkg.svh")
    assert not is_pkg_src("/path/to/swg_pkg_top.sv")
    assert not is_pkg_src("/path/to/_pkg.svh")
