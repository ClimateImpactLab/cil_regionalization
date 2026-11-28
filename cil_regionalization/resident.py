"""This process's peak resident memory, read as each platform keeps it.

On Linux it is VmHWM of ``/proc/self/status``, the high-water mark of
the process's own address space. ``ru_maxrss`` there survives an exec,
so a worker started by spawn would read the peak of the process that
started it. On macOS it is ``ru_maxrss``, which a new worker starts
afresh. Elsewhere it is not read, rather than guessed. The resident
memory of the moment, VmRSS, is read on Linux alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROC_STATUS = Path("/proc/self/status")


def peak_resident_bytes() -> int | None:
    """This process's peak resident memory in bytes, or None where it is
    not read. Under fork it counts the pages the process shares with its
    parent as its own."""
    if sys.platform.startswith("linux"):
        return _from_status("VmHWM")
    if sys.platform == "darwin":
        import resource

        # bytes on macOS
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return None


def resident_bytes() -> int | None:
    """This process's resident memory at this moment in bytes, read as
    VmRSS of ``/proc/self/status`` on Linux, or None elsewhere."""
    if sys.platform.startswith("linux"):
        return _from_status("VmRSS")
    return None


def status_bytes(status: str, field: str) -> int | None:
    """A field of the text of a ``/proc`` status file in bytes, or None if
    the text has no such field counted in kB, which there are kibibytes."""
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key == field:
            number, _, unit = value.strip().partition(" ")
            if number.isdigit() and unit.strip() == "kB":
                return int(number) * 1024
            return None
    return None


def _from_status(field: str) -> int | None:
    try:
        return status_bytes(_PROC_STATUS.read_text(), field)
    except OSError:
        return None


def reading() -> str | None:
    """What `peak_resident_bytes` reads on this platform, or None."""
    if sys.platform.startswith("linux"):
        return "VmHWM of /proc/self/status"
    if sys.platform == "darwin":
        return "ru_maxrss"
    return None


def why_not_read() -> str:
    """Why `peak_resident_bytes` gave None on this platform."""
    if sys.platform.startswith("linux"):
        return "VmHWM could not be read from /proc/self/status"
    return f"the peak resident memory of a process is not read on {sys.platform}"
