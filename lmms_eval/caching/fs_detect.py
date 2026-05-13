"""Filesystem type detection for cache placement decisions.

Determines whether a given path resides on local storage (NVMe/SSD/HDD)
or remote/network storage (NFS, CIFS, GPFS, Lustre, etc.).

Used by the two-tier cache to decide:
    - local disk target  -> write directly
    - remote target      -> write to local scratch, merge back after eval

Detection strategy:
    Linux   - parse ``/proc/mounts`` for the mount entry covering the path,
              then classify by filesystem type string.
    macOS   - use ``mount`` command output (``/proc/mounts`` not available).
    Fallback- assume local if detection fails (safe default: no extra complexity).
"""

import os
import shutil
import subprocess
from enum import Enum
from typing import Optional

from loguru import logger as eval_logger


class FsType(Enum):
    """Filesystem classification."""

    LOCAL = "local"
    REMOTE = "remote"
    UNKNOWN = "unknown"


# Filesystem type strings known to be network / remote.
_REMOTE_FS_TYPES = frozenset(
    {
        # NFS variants
        "nfs",
        "nfs4",
        "nfs3",
        # CIFS / SMB
        "cifs",
        "smbfs",
        "smb2",
        # Parallel / cluster filesystems
        "lustre",
        "gpfs",
        "beegfs",
        "ceph",
        "fuse.ceph",
        "panfs",
        "pvfs2",
        "orangefs",
        "fuse.sshfs",
        "fuse.rclone",
        "fuse.s3fs",
        "fuse.goofys",
        "fuse.gcsfuse",
        "afs",
        "9p",  # common in WSL2 for host-mounted paths
    }
)


def _find_mount_linux(path: str) -> Optional[tuple]:
    """Find the mount entry for ``path`` from /proc/mounts.

    Returns ``(mount_point, fs_type)`` or ``None``.
    """
    try:
        real_path = os.path.realpath(path)
        best_mount = ""
        best_fstype = ""
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = parts[1]
                fs_type = parts[2]
                # Longest prefix match = most specific mount
                if (real_path == mount_point or real_path.startswith(mount_point + os.sep) or mount_point == "/") and len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fstype = fs_type
        if best_mount:
            return (best_mount, best_fstype)
    except (OSError, IOError):
        pass
    return None


def _find_mount_macos(path: str) -> Optional[tuple]:
    """Find the mount entry for ``path`` using ``mount`` command on macOS.

    Returns ``(mount_point, fs_type)`` or ``None``.
    """
    try:
        real_path = os.path.realpath(path)
        result = subprocess.run(["mount"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None

        best_mount = ""
        best_fstype = ""
        for line in result.stdout.splitlines():
            # Format: /dev/disk1s1 on / (apfs, local, journaled)
            #         nfs_server:/export on /mnt/nfs (nfs, ...)
            parts = line.split(" on ", 1)
            if len(parts) < 2:
                continue
            rest = parts[1]
            # Split "mount_point (fstype, flags...)"
            paren_idx = rest.rfind("(")
            if paren_idx == -1:
                continue
            mount_point = rest[: paren_idx - 1].strip()
            fs_info = rest[paren_idx + 1 :].rstrip(")").strip()
            fs_type = fs_info.split(",")[0].strip()

            if (real_path == mount_point or real_path.startswith(mount_point + os.sep) or mount_point == "/") and len(mount_point) > len(best_mount):
                best_mount = mount_point
                best_fstype = fs_type

        if best_mount:
            return (best_mount, best_fstype)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def detect_fs_type(path: str) -> FsType:
    """Detect whether ``path`` is on local or remote storage.

    Resolves to the nearest existing ancestor if ``path`` itself
    does not exist yet (common for new cache files).

    Returns ``FsType.LOCAL``, ``FsType.REMOTE``, or ``FsType.UNKNOWN``.
    """
    # Walk up to find an existing ancestor for detection
    check_path = os.path.abspath(path)
    while not os.path.exists(check_path):
        parent = os.path.dirname(check_path)
        if parent == check_path:
            break  # filesystem root
        check_path = parent

    import sys

    if sys.platform.startswith("linux"):
        mount_info = _find_mount_linux(check_path)
    elif sys.platform == "darwin":
        mount_info = _find_mount_macos(check_path)
    else:
        eval_logger.debug(f"fs_detect: unsupported platform {sys.platform}, assuming local")
        return FsType.UNKNOWN

    if mount_info is None:
        eval_logger.debug(f"fs_detect: could not determine mount for {path}")
        return FsType.UNKNOWN

    mount_point, fs_type = mount_info
    fs_type_lower = fs_type.lower()

    if fs_type_lower in _REMOTE_FS_TYPES:
        eval_logger.debug(f"fs_detect: {path} -> REMOTE (fs={fs_type}, mount={mount_point})")
        return FsType.REMOTE

    eval_logger.debug(f"fs_detect: {path} -> LOCAL (fs={fs_type}, mount={mount_point})")
    return FsType.LOCAL


def find_local_scratch(min_free_gb: float = 1.0) -> Optional[str]:
    """Find a suitable local fast-storage directory for cache scratch space.

    Priority order:
        1. ``$LMMS_LOCAL_CACHE`` environment variable (explicit user override)
        2. ``/local/scratch`` (common HPC convention)
        3. ``/scratch`` (another common convention)
        4. ``/tmp`` (always available, but may be tmpfs / size-limited)

    A candidate is accepted only if it is writable and has at least
    ``min_free_gb`` GB of free space (default 1 GB).

    Returns the first usable directory found, or ``None`` if none qualify.
    """
    candidates = []

    # 1. Explicit user override
    env_path = os.environ.get("LMMS_LOCAL_CACHE")
    if env_path:
        candidates.append(env_path)

    # 2. Common HPC scratch paths
    candidates.extend(["/local/scratch", "/scratch"])

    # 3. /tmp as fallback (always exists on POSIX)
    candidates.append("/tmp")

    for path in candidates:
        if os.path.isdir(path) and os.access(path, os.W_OK):
            try:
                usage = shutil.disk_usage(path)
                free_gb = usage.free / (1024**3)
                if free_gb < min_free_gb:
                    eval_logger.debug(f"fs_detect: skipping {path} (only {free_gb:.1f} GB free, need {min_free_gb} GB)")
                    continue
            except OSError:
                pass  # If we can't check, accept the candidate anyway
            eval_logger.debug(f"fs_detect: local scratch -> {path}")
            return path

    return None
