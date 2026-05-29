############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import os
import re

import finn.util.basic as basic

# representative tool used for the resolution-order scenarios below; the
# round-trip test at the bottom covers every routed tool.
SAMPLE_TOOL = "vivado"

# The canonical routed tools, asserted by the round-trip test below. Kept here
# (not in finn.util.basic) because it is test data -- resolve_xilinx_tool
# accepts any name, so the production code does not need the list.
XILINX_TOOLS = ("vivado", "v++", "vitis_hls", "vitis-run", "xelab")


def _stub_which(monkeypatch, expected):
    """Patch finn.util.basic.which so it accepts only ``expected``."""
    monkeypatch.setattr(
        basic, "which", lambda candidate: candidate if candidate == expected else None
    )


@pytest.mark.util
def test_resolve_no_override_returns_bare_name(monkeypatch):
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    _stub_which(monkeypatch, SAMPLE_TOOL)
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == SAMPLE_TOOL


@pytest.mark.util
def test_resolve_dir_override_joined_with_tool_name(monkeypatch):
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, "/opt/finn-lsf/shim")
    expected = os.path.join("/opt/finn-lsf/shim", SAMPLE_TOOL)
    _stub_which(monkeypatch, expected)
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == expected


@pytest.mark.util
def test_resolve_missing_raises_with_override_in_message(monkeypatch):
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, "/opt/finn-lsf/shim")
    monkeypatch.setattr(basic, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match=re.escape(basic._XILINX_TOOL_DIR_ENV)):
        basic.resolve_xilinx_tool(SAMPLE_TOOL)


# every routed tool round-trips: with no override each resolves to its bare
# name. Catches a typo in the documented tool list.
@pytest.mark.parametrize("tool", XILINX_TOOLS)
@pytest.mark.util
def test_resolve_every_routed_tool_round_trips(monkeypatch, tool):
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    monkeypatch.setattr(basic, "which", lambda candidate: "/usr/bin/%s" % candidate)
    assert basic.resolve_xilinx_tool(tool) == tool
