# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import multiprocessing
import nfs_lock
import os
import socket
import time

pytestmark = pytest.mark.util


def _write_holder(lock_dir, *, pid, hostname, ts):
    """Plant a held lock dir with a chosen holder.json (simulates another holder)."""
    os.mkdir(lock_dir)
    with open(os.path.join(lock_dir, nfs_lock.HOLDER_FILENAME), "w") as f:
        json.dump({"pid": pid, "hostname": hostname, "timestamp": ts}, f)


def _find_dead_pid():
    """A PID guaranteed dead and on this host: fork a child, reap it, reuse its PID."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _rmw_worker(lock_path, counter_path):
    """Read-modify-write the counter under the lock.

    The sleep widens the read->write window so an unsynchronised version would
    lose updates; under the lock the final count must equal the worker count.
    """
    with nfs_lock.nfs_dir_lock(lock_path, timeout=60, poll=0.02):
        try:
            with open(counter_path) as f:
                value = int(f.read().strip() or "0")
        except FileNotFoundError:
            value = 0
        time.sleep(0.01)
        with open(counter_path, "w") as f:
            f.write(str(value + 1))


def test_acquire_creates_and_release_removes(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    with nfs_lock.nfs_dir_lock(lock):
        assert os.path.isdir(lock)
        assert os.path.isfile(os.path.join(lock, nfs_lock.HOLDER_FILENAME))
    assert not os.path.exists(lock)


def test_missing_parent_is_created(tmp_path):
    lock = str(tmp_path / "deep" / "nested" / "x.lock.d")
    with nfs_lock.nfs_dir_lock(lock):
        assert os.path.isdir(lock)
    assert not os.path.exists(lock)


def test_released_on_exception(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    with pytest.raises(RuntimeError):
        with nfs_lock.nfs_dir_lock(lock):
            raise RuntimeError("boom")
    assert not os.path.exists(lock)


def test_timeout_when_held_by_live_holder(tmp_path):
    # A live holder (this process) within the TTL must not be stolen, so a
    # second acquire times out rather than corrupting mutual exclusion.
    lock = str(tmp_path / "x.lock.d")
    _write_holder(lock, pid=os.getpid(), hostname=socket.gethostname(), ts=nfs_lock._now_iso())
    with pytest.raises(TimeoutError):
        with nfs_lock.nfs_dir_lock(lock, timeout=0.5, poll=0.05):
            pass
    assert os.path.isfile(os.path.join(lock, nfs_lock.HOLDER_FILENAME))


def test_stale_holder_by_ttl_is_stolen(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    old = time.strftime(nfs_lock._TS_FORMAT, time.gmtime(time.time() - 10000))
    # Foreign hostname so the steal decision rests on the TTL, not PID liveness.
    _write_holder(lock, pid=1, hostname="some-other-host", ts=old)
    with nfs_lock.nfs_dir_lock(lock, timeout=2.0, stale_ttl=120):
        assert nfs_lock._read_holder(lock)["pid"] == os.getpid()
    assert not os.path.exists(lock)


def test_dead_pid_on_this_host_is_stolen(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    # Recent timestamp: only the dead-PID-on-this-host check can make it stale.
    _write_holder(lock, pid=_find_dead_pid(), hostname=socket.gethostname(), ts=nfs_lock._now_iso())
    with nfs_lock.nfs_dir_lock(lock, timeout=2.0, stale_ttl=120):
        assert nfs_lock._read_holder(lock)["pid"] == os.getpid()
    assert not os.path.exists(lock)


def test_holderless_dir_stolen_after_grace(tmp_path, monkeypatch):
    lock = str(tmp_path / "x.lock.d")
    os.mkdir(lock)  # aborted mkdir: no holder.json written
    monkeypatch.setattr(nfs_lock, "_NO_HOLDER_GRACE_S", -1.0)
    with nfs_lock.nfs_dir_lock(lock, timeout=2.0):
        assert os.path.isfile(os.path.join(lock, nfs_lock.HOLDER_FILENAME))
    assert not os.path.exists(lock)


def test_multiprocess_no_lost_updates(tmp_path):
    # The real mutual-exclusion proof: N processes race the same RMW; with a
    # correct lock the final count is exactly N.
    lock = str(tmp_path / "counter.lock.d")
    counter = str(tmp_path / "counter.txt")
    n = 12
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_rmw_worker, args=(lock, counter)) for _ in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert all(p.exitcode == 0 for p in procs)
    with open(counter) as f:
        assert int(f.read().strip()) == n
    assert not os.path.exists(lock)
