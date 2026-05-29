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
# round-trip test at the bottom covers every registered tool and is the
# only place the full table of (tool, env_var) pairs needs to be enumerated.
SAMPLE_TOOL = "vivado"
SAMPLE_ENV = basic._XILINX_TOOL_OVERRIDES[SAMPLE_TOOL]


def _stub_which(monkeypatch, expected):
    """Patch finn.util.basic.which so it accepts only ``expected``."""
    monkeypatch.setattr(
        basic, "which", lambda candidate: candidate if candidate == expected else None
    )


@pytest.mark.util
def test_resolve_no_override_returns_bare_name(monkeypatch):
    monkeypatch.delenv(SAMPLE_ENV, raising=False)
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    _stub_which(monkeypatch, SAMPLE_TOOL)
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == SAMPLE_TOOL


@pytest.mark.util
def test_resolve_per_tool_override_wins(monkeypatch):
    override = "/opt/xilinx/2025.1/bin/%s" % SAMPLE_TOOL
    monkeypatch.setenv(SAMPLE_ENV, override)
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    _stub_which(monkeypatch, override)
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == override


@pytest.mark.util
def test_resolve_dir_override_joined_with_tool_name(monkeypatch):
    monkeypatch.delenv(SAMPLE_ENV, raising=False)
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, "/opt/finn-lsf/shim")
    expected = os.path.join("/opt/finn-lsf/shim", SAMPLE_TOOL)
    _stub_which(monkeypatch, expected)
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == expected


@pytest.mark.util
def test_resolve_per_tool_beats_dir_override(monkeypatch):
    per_tool = "/opt/private/%s" % SAMPLE_TOOL
    monkeypatch.setenv(SAMPLE_ENV, per_tool)
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, "/opt/finn-lsf/shim")
    _stub_which(monkeypatch, per_tool)
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == per_tool


@pytest.mark.util
def test_resolve_missing_raises(monkeypatch):
    monkeypatch.delenv(SAMPLE_ENV, raising=False)
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    monkeypatch.setattr(basic, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="%s not found" % re.escape(SAMPLE_TOOL)):
        basic.resolve_xilinx_tool(SAMPLE_TOOL)


# every registered tool round-trips: each key in the override map resolves
# to itself when no env vars are set. Catches a typo'd registration row.
@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_every_registered_tool_round_trips(monkeypatch, tool, env_var):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    monkeypatch.setattr(basic, "which", lambda candidate: "/usr/bin/%s" % candidate)
    assert basic.resolve_xilinx_tool(tool) == tool
