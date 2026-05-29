#############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	source-list helpers for the FINN XSI adapter
#############################################################################

import os.path

# Kept in its own module (no ``xsi`` C-extension import) so it can be unit
# tested in a fresh checkout that has not built finn_xsi yet; adapter.py
# imports it.


def is_pkg_src(path: str) -> bool:
    """``True`` for ``*_pkg.{sv,v}`` source files (package declarations).

    Used by ``compile_sim_obj`` to partition the source list so packages are
    handed to xelab before modules that import them. A basename-suffix match
    catches every package file uniformly; the prior substring whitelist
    (``["swg_pkg", "mvu_pkg"]``) silently elaborated newer finn-hlslib
    packages after their importers and tripped xelab.
    """
    base = os.path.basename(path)
    return base.endswith("_pkg.sv") or base.endswith("_pkg.v")
