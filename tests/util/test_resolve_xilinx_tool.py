############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import re

import finn.util.basic as basic


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_no_override_returns_bare_name(monkeypatch, tool, env_var):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(basic, "which", lambda candidate: "/usr/bin/%s" % candidate)
    assert basic.resolve_xilinx_tool(tool) == tool


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_override_returns_env_value(monkeypatch, tool, env_var):
    override = "/opt/xilinx/2025.1/bin/%s" % tool
    monkeypatch.setenv(env_var, override)
    monkeypatch.setattr(
        basic, "which", lambda candidate: candidate if candidate == override else None
    )
    assert basic.resolve_xilinx_tool(tool) == override


@pytest.mark.parametrize("tool,env_var", sorted(basic._XILINX_TOOL_OVERRIDES.items()))
@pytest.mark.util
def test_resolve_xilinx_tool_missing_raises(monkeypatch, tool, env_var):
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(basic, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="%s not found" % re.escape(tool)):
        basic.resolve_xilinx_tool(tool)
