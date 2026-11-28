"""This process's peak resident memory, read as each platform keeps it."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cil_regionalization import resident

# The lines of /proc/self/status around VmHWM as a Linux kernel writes
# them, with the peak of the address space's size, VmPeak, before it.
STATUS = (
    "Name:\tpython\n"
    "VmPeak:\t  512000 kB\n"
    "VmSize:\t  498000 kB\n"
    "VmHWM:\t   17600 kB\n"
    "VmRSS:\t   17200 kB\n"
)


def _on(monkeypatch, platform, *, status=None, ru_maxrss=419_900):
    """The platform given, a status file of ``status``, and a ru_maxrss
    that is never the answer on Linux."""
    import resource

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(
        resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=ru_maxrss)
    )
    if status is not None:
        monkeypatch.setattr(resident, "_PROC_STATUS", status)


class TestStatus:
    def test_the_high_water_mark_and_the_resident_memory_in_bytes(self):
        """Their kB are kibibytes. It went red on thousands of bytes."""
        assert resident.status_bytes(STATUS, "VmHWM") == 17600 * 1024
        assert resident.status_bytes(STATUS, "VmRSS") == 17200 * 1024

    @pytest.mark.parametrize(
        "text",
        [
            "Name:\tpython\nVmRSS:\t   17200 kB\n",
            "VmHWM:\t   17600 MB\n",
            "VmHWM:\t\n",
            "",
        ],
    )
    def test_a_status_without_a_vmhwm_in_kb_gives_none(self, text):
        """Rather than a number guessed. It went red with any unit read
        as kB."""
        assert resident.status_bytes(text, "VmHWM") is None


class TestPlatforms:
    def test_linux_reads_vmhwm_and_not_ru_maxrss(self, monkeypatch, tmp_path):
        """ru_maxrss on Linux keeps the peak of the process that started a
        spawned worker. It went red with ru_maxrss read there."""
        status = tmp_path / "status"
        status.write_text(STATUS)
        _on(monkeypatch, "linux", status=status)
        assert resident.peak_resident_bytes() == 17600 * 1024
        assert resident.reading() == "VmHWM of /proc/self/status"
        assert resident.resident_bytes() == 17200 * 1024

    def test_linux_without_the_file_gives_none_and_says_why(
        self, monkeypatch, tmp_path
    ):
        _on(monkeypatch, "linux", status=tmp_path / "absent")
        assert resident.peak_resident_bytes() is None
        assert resident.resident_bytes() is None
        assert resident.why_not_read() == (
            "VmHWM could not be read from /proc/self/status"
        )

    def test_macos_reads_ru_maxrss_in_bytes(self, monkeypatch):
        """The resident memory of the moment is not read there."""
        _on(monkeypatch, "darwin", ru_maxrss=1000)
        assert resident.peak_resident_bytes() == 1000
        assert resident.reading() == "ru_maxrss"
        assert resident.resident_bytes() is None

    def test_elsewhere_nothing_is_read(self, monkeypatch):
        _on(monkeypatch, "freebsd14")
        assert resident.peak_resident_bytes() is None
        assert resident.reading() is None
        assert resident.why_not_read() == (
            "the peak resident memory of a process is not read on freebsd14"
        )


@pytest.mark.parametrize(
    "module, name",
    [
        ("cil_regionalization.reduce.pandas_blocks", "_peak_gb"),
        ("cil_regionalization.pipelines.montecarlo", "_peak_rss_gb"),
    ],
)
def test_both_stages_log_the_same_reading_in_gigabytes(
    monkeypatch, tmp_path, module, name
):
    """The reduce stage read ru_maxrss and went red on that. The
    aggregate stage also took Linux's kibibytes for thousands of bytes."""
    import importlib

    status = tmp_path / "status"
    status.write_text("VmHWM:\t 1000000 kB\n")
    _on(monkeypatch, "linux", status=status)
    assert getattr(importlib.import_module(module), name)() == 1.024
