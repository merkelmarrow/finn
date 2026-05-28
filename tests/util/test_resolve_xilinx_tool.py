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


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_no_override_returns_bare_name(monkeypatch, tool, env_var):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    monkeypatch.setattr(basic, "which", lambda candidate: "/usr/bin/%s" % candidate)
    assert basic.resolve_xilinx_tool(tool) == tool


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_per_tool_override_wins(monkeypatch, tool, env_var):
    override = "/opt/xilinx/2025.1/bin/%s" % tool
    monkeypatch.setenv(env_var, override)
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    monkeypatch.setattr(
        basic, "which", lambda candidate: candidate if candidate == override else None
    )
    assert basic.resolve_xilinx_tool(tool) == override


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_dir_override_resolves_each_tool(monkeypatch, tool, env_var):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, "/opt/finn-lsf/shim")
    expected = os.path.join("/opt/finn-lsf/shim", tool)
    monkeypatch.setattr(
        basic, "which", lambda candidate: candidate if candidate == expected else None
    )
    assert basic.resolve_xilinx_tool(tool) == expected


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_per_tool_beats_dir_override(monkeypatch, tool, env_var):
    per_tool = "/opt/private/%s" % tool
    monkeypatch.setenv(env_var, per_tool)
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, "/opt/finn-lsf/shim")
    monkeypatch.setattr(
        basic, "which", lambda candidate: candidate if candidate == per_tool else None
    )
    assert basic.resolve_xilinx_tool(tool) == per_tool


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_missing_raises(monkeypatch, tool, env_var):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    monkeypatch.setattr(basic, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="%s not found" % re.escape(tool)):
        basic.resolve_xilinx_tool(tool)
