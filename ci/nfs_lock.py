#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NFS-safe mutual exclusion via mkdir/rmdir atomicity.

``flock``/``fcntl`` advisory locks are local-only on most NFS mounts -- they
are not visible across hosts -- so they cannot serialise writers on different
build agents sharing one NFS root. ``os.mkdir`` is atomic on every NFS
version (it either creates the directory or raises ``FileExistsError``), so it
is a reliable cross-host test-and-set. A ``holder.json`` written inside the
lock dir records pid/hostname/timestamp for debugging and for stale-lock
detection that is independent of NFS mtime and tolerant of client clock skew.

Intended for short, bounded critical sections (a JSON read-modify-write).
There is no heartbeat: a holder that dies or hangs is reclaimed after
``stale_ttl`` seconds (instantly when it is a dead process on this host), so
the body under the lock MUST complete well within ``stale_ttl``.
"""

import calendar
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager

HOLDER_FILENAME = "holder.json"
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# A lock dir with no holder.json is either mid-acquire (the writer is between
# the mkdir and the holder write) or an aborted mkdir. Give the writer this
# grace window before treating a holder-less dir as abandoned.
_NO_HOLDER_GRACE_S = 5.0


def _now_iso():
    return time.strftime(_TS_FORMAT, time.gmtime())


def _chmod_gw(path, mode):
    # Best-effort group-writable perms so another agent or UID sharing the NFS
    # root can still steal a stale lock. Non-fatal if we do not own the path.
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _read_holder(lock_dir):
    try:
        with open(os.path.join(lock_dir, HOLDER_FILENAME)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _path_age(path):
    """Age in seconds of ``path`` by mtime, or None if it cannot be stat'd.

    Opens and closes the path first to force NFS close-to-open revalidation of
    the cached attributes before the stat. Used only as a fallback; the
    holder's embedded timestamp is the primary, clock-skew-tolerant signal.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        os.close(fd)
    except OSError:
        pass
    try:
        return max(0.0, time.time() - os.stat(path).st_mtime)
    except OSError:
        return None


def _holder_age(meta):
    """Age in seconds from the holder's embedded UTC timestamp, or None.

    Clock-skew tolerant: the holder recorded its own wall clock and both it
    and this reader drift together under NTP, so the difference stays sane
    even when the NFS server mtime does not.
    """
    if not isinstance(meta, dict):
        return None
    ts = meta.get("timestamp")
    if not ts:
        return None
    try:
        held = calendar.timegm(time.strptime(ts, _TS_FORMAT))
    except (ValueError, OverflowError):
        return None
    return max(0.0, time.time() - held)


def _holder_dead_on_this_host(meta):
    """True only when the holder is a process on this host that no longer exists.

    Cross-host holders return False here (we cannot probe their PID) and are
    judged by ``stale_ttl`` instead.
    """
    if not isinstance(meta, dict) or meta.get("hostname") != socket.gethostname():
        return False
    pid = meta.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        # PermissionError (alive, other user) or any other probe failure: not dead.
        return False
    return False


def _is_stale(lock_dir, stale_ttl):
    meta = _read_holder(lock_dir)
    if meta is None:
        # No readable holder: abandoned mkdir once past the acquire grace.
        age = _path_age(lock_dir)
        return age is not None and age > _NO_HOLDER_GRACE_S
    if _holder_dead_on_this_host(meta):
        return True
    age = _holder_age(meta)
    if age is None:
        # Unparseable timestamp: fall back to the holder file's mtime.
        age = _path_age(os.path.join(lock_dir, HOLDER_FILENAME))
    return age is not None and age > stale_ttl


def _force_remove(lock_dir):
    try:
        shutil.rmtree(lock_dir)
    except OSError:
        pass


@contextmanager
def nfs_dir_lock(lock_path, *, timeout=60.0, poll=0.5, stale_ttl=120.0):
    """Hold an NFS-safe directory lock at ``lock_path`` for the with-block.

    Raises ``TimeoutError`` if the lock cannot be taken within ``timeout``
    seconds. A holder older than ``stale_ttl`` (or a dead process on this
    host) is reclaimed, so a crashed holder never deadlocks other agents.
    """
    parent = os.path.dirname(os.path.abspath(lock_path))
    os.makedirs(parent, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.mkdir(lock_path)
            break
        except FileExistsError:
            if _is_stale(lock_path, stale_ttl):
                _force_remove(lock_path)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "could not acquire %s within %.0fs (holder=%r)"
                    % (lock_path, timeout, _read_holder(lock_path))
                )
            time.sleep(poll)
    _chmod_gw(lock_path, 0o2775)
    holder = os.path.join(lock_path, HOLDER_FILENAME)
    try:
        meta = {"pid": os.getpid(), "hostname": socket.gethostname(), "timestamp": _now_iso()}
        fd = os.open(holder, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
        try:
            os.write(fd, json.dumps(meta).encode())
        finally:
            os.close(fd)
        _chmod_gw(holder, 0o664)
        yield
    finally:
        try:
            os.unlink(holder)
        except OSError:
            pass
        try:
            os.rmdir(lock_path)
        except OSError:
            _force_remove(lock_path)
