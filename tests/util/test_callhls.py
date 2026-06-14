############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import finn.util.hls as hls


class _FakeProc:
    def communicate(self):
        return (b"", b"")


@pytest.mark.util
def test_callhls_writes_quoted_private_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("XILINX_VIVADO", "/x/Vivado/2024.2")
    monkeypatch.setenv("PWD", str(tmp_path))
    # stub tool resolution and the actual tool launch
    monkeypatch.setattr(hls, "resolve_xilinx_tool", lambda name: name)
    monkeypatch.setattr(hls.subprocess, "Popen", lambda *a, **k: _FakeProc())

    code_gen_dir = tmp_path / "codegen"
    code_gen_dir.mkdir()
    caller = hls.CallHLS()
    caller.append_tcl("script.tcl")
    caller.build(str(code_gen_dir))

    priv_tmp = code_gen_dir / ".vitis_tmp"
    assert priv_tmp.is_dir()
    script = (code_gen_dir / "ipgen.sh").read_text()
    assert 'export TMPDIR="%s"' % str(priv_tmp) in script
