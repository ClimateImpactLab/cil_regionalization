"""The France example runs from end to end on synthetic data laid out
as its real data is, as a script in a process of its own, the way a
reader runs it."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "france"


def _synthetic(directory: Path) -> tuple[Path, Path]:
    spec = importlib.util.spec_from_file_location(
        "france_synthetic_data", EXAMPLE / "synthetic_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write(directory)


def test_the_notebook_runs_on_synthetic_data(tmp_path):
    pytest.importorskip("marimo")
    pytest.importorskip("matplotlib")
    data, warming = _synthetic(tmp_path / "synthetic")
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, str(EXAMPLE / "france.py")],
        env={
            **os.environ,
            "FRANCE_DATA": str(data),
            "FRANCE_OUT": str(out),
            "FRANCE_WARMING": str(warming),
            "FRANCE_OFFLINE": "1",
            "MPLBACKEND": "Agg",
        },
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-6000:]
    for stem in (
        "deaths_window_mean",
        "deaths_pooled",
        "deaths_per_year",
        "deaths_warming",
        "damages_warming",
    ):
        assert (out / "pipeline" / f"{stem}.parquet").exists(), stem
