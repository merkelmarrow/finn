# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import multiprocessing
import nfs_lock
import os
import socket
import time

pytestmark = pytest.mark.util

# A ts that sorts before any real (current-time) ticket, so a planted ticket is
# always "earlier" in the bakery queue and must be cleared before we proceed.
_EARLIEST_TS = "00000000T000000000000"


def _plant_ticket(lock_dir, *, pid, host, ts=_EARLIEST_TS, tid=1, mtime=None):
    """Drop a ticket with a chosen name/mtime to simulate another holder."""
    os.makedirs(lock_dir, exist_ok=True)
    name = "%s.%d.%d.%s%s" % (ts, tid, pid, host, nfs_lock.TICKET_SUFFIX)
    path = os.path.join(lock_dir, name)
    with open(path, "w") as f:
        f.write("{}")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _live_tickets(lock_dir):
    return [n for n in os.listdir(lock_dir) if n.endswith(nfs_lock.TICKET_SUFFIX)]


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


def test_acquire_writes_ticket_and_release_removes_it(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    with nfs_lock.nfs_dir_lock(lock):
        assert len(_live_tickets(lock)) == 1
    assert _live_tickets(lock) == []


def test_missing_parent_is_created(tmp_path):
    lock = str(tmp_path / "deep" / "nested" / "x.lock.d")
    with nfs_lock.nfs_dir_lock(lock):
        assert os.path.isdir(lock)


def test_released_on_exception(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    with pytest.raises(RuntimeError):
        with nfs_lock.nfs_dir_lock(lock):
            raise RuntimeError("boom")
    assert _live_tickets(lock) == []


def test_timeout_when_blocked_by_live_earlier_ticket(tmp_path):
    # An earlier ticket held by a live process on this host (us) within the TTL
    # must not be reclaimed, so a second acquire times out rather than
    # corrupting mutual exclusion.
    lock = str(tmp_path / "x.lock.d")
    _plant_ticket(lock, pid=os.getpid(), host=socket.gethostname())
    with pytest.raises(TimeoutError):
        with nfs_lock.nfs_dir_lock(lock, timeout=0.5, poll=0.05):
            pass


def test_stale_earlier_ticket_by_ttl_is_reclaimed(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    # Foreign host so the reclaim decision rests on the TTL, not PID liveness.
    _plant_ticket(lock, pid=1, host="some-other-host", mtime=time.time() - 10000)
    with nfs_lock.nfs_dir_lock(lock, timeout=2.0, stale_ttl=120):
        pass  # acquired: the stale earlier ticket was reclaimed


def test_dead_pid_earlier_ticket_on_this_host_is_reclaimed(tmp_path):
    lock = str(tmp_path / "x.lock.d")
    # Recent mtime, so only the dead-PID-on-this-host check can make it stale.
    _plant_ticket(lock, pid=_find_dead_pid(), host=socket.gethostname())
    with nfs_lock.nfs_dir_lock(lock, timeout=2.0, stale_ttl=120):
        pass


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
    assert _live_tickets(lock) == []


def test_concurrent_stale_reclaim_no_lost_updates(tmp_path):
    # N processes contend while an always-stale earlier ticket sits in the
    # queue, so every contender must reap it before proceeding. If concurrent
    # reclamation could double-acquire, updates would be lost; the exact count
    # proves it cannot.
    lock = str(tmp_path / "counter.lock.d")
    counter = str(tmp_path / "counter.txt")
    _plant_ticket(lock, pid=1, host="some-dead-host", mtime=time.time() - 10000)
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
