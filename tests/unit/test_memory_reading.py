"""The pipeline's memory readings, without a Linux /proc."""

from __future__ import annotations

import builtins
import io
import resource

from cil_regionalization.pipelines import montecarlo


def test_the_resident_pages_are_of_the_size_the_machine_reports(monkeypatch):
    """/proc/self/statm counts pages, and a page is 16 KiB on some
    machines. It went red on the 4096 bytes written in the code."""
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/self/statm":
            return io.StringIO("1000 250 10 1 0 200 0\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(resource, "getpagesize", lambda: 16384)
    assert montecarlo._current_rss_gb() == 250 * 16384 / 1e9
