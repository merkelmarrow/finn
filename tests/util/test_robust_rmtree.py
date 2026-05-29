############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import ast
import errno
import os
import shutil
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASIC_PATH = os.path.join(REPO_ROOT, "src", "finn", "util", "basic.py")


def _extract_robust_rmtree():
    """Pull module-level ``robust_rmtree`` out of src/finn/util/basic.py via AST.

    Importing finn.util.basic would drag qonnx/torch into the util shard's
    collection path. AST extraction keeps this test stdlib-only while still
    exercising the real implementation.
    """
    tree = ast.parse(open(BASIC_PATH).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "robust_rmtree":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"os": os, "shutil": shutil, "time": time, "errno": errno}
            exec(compile(module, BASIC_PATH, "exec"), namespace)
            return namespace["robust_rmtree"]
    raise AssertionError("could not find robust_rmtree in %s" % BASIC_PATH)


@pytest.mark.util
def test_robust_rmtree_succeeds_first_attempt(tmp_path):
    robust_rmtree = _extract_robust_rmtree()
    target = tmp_path / "tree"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "f").write_text("x")
    robust_rmtree(str(target))
    assert not target.exists()


@pytest.mark.util
def test_robust_rmtree_missing_path_is_noop(tmp_path):
    robust_rmtree = _extract_robust_rmtree()
    robust_rmtree(str(tmp_path / "does_not_exist"))
    robust_rmtree("")
    robust_rmtree(None)


@pytest.mark.util
def test_robust_rmtree_retries_on_enotempty_then_succeeds(tmp_path, monkeypatch):
    robust_rmtree = _extract_robust_rmtree()
    target = tmp_path / "tree"
    target.mkdir()
    (target / "f").write_text("x")

    state = {"calls": 0}
    real_rmtree = shutil.rmtree

    def flaky(path, *a, **kw):
        state["calls"] += 1
        if state["calls"] < 3:
            raise OSError(errno.ENOTEMPTY, "fake", str(path))
        return real_rmtree(path, *a, **kw)

    # Patch the shutil module that the AST-execed function closes over via
    # its exec namespace. monkeypatching shutil.rmtree on the real module
    # is what the captured `shutil` symbol resolves through.
    monkeypatch.setattr(shutil, "rmtree", flaky)
    # Speed the test up: zero out the backoff so we do not actually sleep.
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    robust_rmtree(str(target), retries=5, initial_delay=0.01, backoff=1.0)

    assert state["calls"] == 3
    assert not target.exists()


@pytest.mark.util
def test_robust_rmtree_propagates_non_enotempty_oserror(tmp_path, monkeypatch):
    robust_rmtree = _extract_robust_rmtree()
    target = tmp_path / "tree"
    target.mkdir()

    calls = []

    def always_eacces(path, *a, **kw):
        calls.append(path)
        raise OSError(errno.EACCES, "fake", str(path))

    monkeypatch.setattr(shutil, "rmtree", always_eacces)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(OSError) as excinfo:
        robust_rmtree(str(target), retries=5, initial_delay=0.01, backoff=1.0)

    assert excinfo.value.errno == errno.EACCES
    # No retry: a non-ENOTEMPTY OSError propagates on the first attempt.
    assert len(calls) == 1


@pytest.mark.util
def test_robust_rmtree_raises_after_retries_exhausted(tmp_path, monkeypatch):
    robust_rmtree = _extract_robust_rmtree()
    target = tmp_path / "tree"
    target.mkdir()

    calls = []

    def always_enotempty(path, *a, **kw):
        calls.append(path)
        raise OSError(errno.ENOTEMPTY, "fake", str(path))

    monkeypatch.setattr(shutil, "rmtree", always_enotempty)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(OSError) as excinfo:
        robust_rmtree(str(target), retries=4, initial_delay=0.01, backoff=1.0)

    assert excinfo.value.errno == errno.ENOTEMPTY
    assert len(calls) == 4


@pytest.mark.util
def test_robust_rmtree_tolerates_filenotfounderror(tmp_path, monkeypatch):
    robust_rmtree = _extract_robust_rmtree()
    target = tmp_path / "tree"
    target.mkdir()

    def fnf(path, *a, **kw):
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(shutil, "rmtree", fnf)
    # No sleep needed because FileNotFoundError returns immediately.
    # Should return cleanly, no exception.
    robust_rmtree(str(target))
