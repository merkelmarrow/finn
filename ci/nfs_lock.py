#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NFS-safe mutual exclusion via a timestamp-ordered bakery ticket lock.

``flock``/``fcntl`` advisory locks are local-only on most NFS mounts (they are
not visible across hosts), so they cannot serialise writers on different build
agents sharing one NFS root. A single shared lock object that waiters race to
create and then steal once it looks stale is also unsafe: two waiters can both
judge the holder stale, both remove it, and both then take the lock.

This module sidesteps both problems with the bakery algorithm. Each waiter
writes its own uniquely named ticket (a dotless microsecond timestamp followed
by pid and host) into a shared directory, and holds the lock only while it owns
the earliest ticket that is not stale. There is no single object to win, so
reclaiming a crashed holder's ticket is an idempotent unlink with no
double-acquire window, and exclusion follows from filename ordering rather than
from the atomicity of any one create.

Intended for short, bounded critical sections (a JSON read-modify-write). There
is no heartbeat: a holder that dies or hangs is reclaimed after ``stale_ttl``
seconds (instantly when it is a dead process on this host), so the body under
the lock MUST complete well within ``stale_ttl``. The bakery design is proven
in the orca fleet locker; this is a self-contained, exclusive-only,
heartbeat-free port.
"""

import json
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

# A ticket is "<ts>.<tid>.<pid>.<host>.json". The ts is dotless and fixed width
# so plain string ordering is creation order, and the trailing pid/host let a
# same-host holder be probed for liveness. The atomic-write sibling ends in
# ".json.tmp" so it is never mistaken for a live ticket.
TICKET_SUFFIX = ".json"
_TMP_SUFFIX = ".tmp"

# Past this many seconds of mtime-in-the-future we warn about NTP skew rather
# than silently trusting a nonsensical age.
_CLOCK_SKEW_WARN_S = 5.0


def _local_hostname():
    return socket.gethostname()


def _ticket_name(pid, host):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return "%s.%d.%d.%s%s" % (ts, threading.get_ident(), pid, host, TICKET_SUFFIX)


def _parse_pid_host(name):
    """Return ``(pid, host)`` parsed from a ticket name, or None if malformed."""
    if not name.endswith(TICKET_SUFFIX):
        return None
    parts = name[: -len(TICKET_SUFFIX)].split(".")
    if len(parts) < 4:
        return None
    try:
        pid = int(parts[2])
    except ValueError:
        return None
    host = ".".join(parts[3:])
    return (pid, host) if host else None


def _chmod_gw(path, mode):
    # Best-effort group-writable perms so another agent or UID sharing the NFS
    # root can add and reap tickets. Non-fatal if we do not own the path.
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _makedirs_gw(path):
    os.makedirs(path, exist_ok=True)
    _chmod_gw(path, 0o2775)


def _ticket_mtime_age(path):
    """Age in seconds of ``path`` by mtime, or None if it cannot be stat'd.

    Opens and closes the path first to force NFS close-to-open revalidation of
    the cached attributes before the stat.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        os.close(fd)
    except OSError:
        pass
    try:
        age = time.time() - os.stat(path).st_mtime
    except OSError:
        return None
    if age < -_CLOCK_SKEW_WARN_S:
        print(
            "nfs_lock: %s has an mtime %.0fs in the future, check NTP on the fleet" % (path, -age),
            file=sys.stderr,
        )
    return max(0.0, age)


def _holder_dead_on_this_host(name):
    """True only when the ticket's holder is a process on this host that is gone.

    Cross-host holders return False (their pid cannot be probed) and are judged
    by ``stale_ttl`` instead.
    """
    parsed = _parse_pid_host(name)
    if parsed is None:
        return False
    pid, host = parsed
    if host != _local_hostname():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        # alive but owned by another user, or the probe was not permitted
        return False
    return False


def _is_stale(lock_dir, name, stale_ttl):
    if _holder_dead_on_this_host(name):
        return True
    age = _ticket_mtime_age(os.path.join(lock_dir, name))
    return age is not None and age > stale_ttl


def _list_tickets(lock_dir):
    """Return ticket names in creation order, busting the NFS dcache first."""
    try:
        # a GETATTR on the dir forces the client to revalidate its cached
        # listing so a ticket just written on another host is not missed
        os.stat(lock_dir)
    except OSError:
        return []
    try:
        names = os.listdir(lock_dir)
    except OSError:
        return []
    return sorted(n for n in names if n.endswith(TICKET_SUFFIX) and not n.startswith("."))


def _reap_stale(lock_dir, stale_ttl):
    # Idempotent: a ticket already removed by a racing reaper is a no-op, which
    # is what makes concurrent reclamation safe (there is nothing to "claim").
    for name in _list_tickets(lock_dir):
        if _is_stale(lock_dir, name, stale_ttl):
            try:
                os.unlink(os.path.join(lock_dir, name))
            except OSError:
                pass


def _reap_orphan_tmps(lock_dir, stale_ttl):
    # A process killed between the O_EXCL write and the rename leaves a
    # ".json.tmp" behind. These never block the lock (only ".json" files count)
    # but are cleared once stale so the dir does not accrue debris.
    try:
        names = os.listdir(lock_dir)
    except OSError:
        return
    for name in names:
        if name.endswith(_TMP_SUFFIX) and not name.startswith("."):
            age = _ticket_mtime_age(os.path.join(lock_dir, name))
            if age is not None and age > stale_ttl:
                try:
                    os.unlink(os.path.join(lock_dir, name))
                except OSError:
                    pass


def _write_ticket(lock_dir, name):
    """Write our ticket atomically (O_EXCL tmp then rename) and return its path.

    The metadata is purely for humans inspecting the dir; the load-bearing
    identity (timestamp, pid, host) lives in the filename.
    """
    path = os.path.join(lock_dir, name)
    tmp = path + _TMP_SUFFIX
    meta = {
        "pid": os.getpid(),
        "hostname": _local_hostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    try:
        os.write(fd, json.dumps(meta).encode())
    finally:
        os.close(fd)
    _chmod_gw(tmp, 0o664)
    os.rename(tmp, path)
    return path


@contextmanager
def nfs_dir_lock(lock_path, *, timeout=60.0, poll=0.5, stale_ttl=120.0):
    """Hold an NFS-safe exclusive lock keyed on ``lock_path`` for the with-block.

    ``lock_path`` is the ticket directory. The caller proceeds only while it
    owns the earliest ticket in that directory that is not stale, so mutual
    exclusion holds across hosts without relying on advisory-lock visibility.

    Raises ``TimeoutError`` if the lock cannot be taken within ``timeout``
    seconds. A holder older than ``stale_ttl`` (or a dead process on this host)
    is reclaimed, so a crashed holder never deadlocks other agents.
    """
    lock_dir = os.path.abspath(lock_path)
    _makedirs_gw(lock_dir)
    name = _ticket_name(os.getpid(), _local_hostname())
    ticket = _write_ticket(lock_dir, name)
    deadline = time.monotonic() + timeout
    try:
        while True:
            _reap_stale(lock_dir, stale_ttl)
            earlier = [
                t
                for t in _list_tickets(lock_dir)
                if t < name and not _is_stale(lock_dir, t, stale_ttl)
            ]
            if not earlier:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "could not acquire %s within %.0fs (ahead in queue: %s)"
                    % (lock_dir, timeout, ", ".join(earlier[:3]))
                )
            time.sleep(poll)
        yield
    finally:
        try:
            os.unlink(ticket)
        except OSError:
            pass
        _reap_orphan_tmps(lock_dir, stale_ttl)
        # The lock dir is intentionally left in place. Removing it on release
        # would race a concurrent acquirer that has created the dir but not yet
        # written its ticket. One empty dir per lock is harmless.
