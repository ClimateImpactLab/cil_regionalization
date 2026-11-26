"""The data scripts of the examples run on the cluster, against files
this suite does not have. The parts that need no cluster file run here
on synthetic ones."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _script(name: str):
    pytest.importorskip("netCDF4" if name == "energy" else "xarray")
    spec = importlib.util.spec_from_file_location(
        f"prepare_{name}", EXAMPLES / name / "prepare_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["energy", "labor"])
def test_the_smme_weights_are_copied_from_the_file_named(tmp_path, monkeypatch, name):
    """Failed with a NameError: the helper it called had been removed
    with the mount fallback."""
    script = _script(name)
    table = tmp_path / "smme.tsv"
    table.write_text("quantile\tmodel\tweight\textra\n1\tccsm4\t0.25\tx\n2\tgfdl-cm3\t0.75\ty\n")
    monkeypatch.setattr(script, "SMME", table)
    monkeypatch.setattr(script, "DATA", tmp_path)
    script.write_smme()
    written = pd.read_csv(tmp_path / "smme_weights_rcp85.csv")
    assert list(written.columns) == ["model", "weight"]
    assert written["weight"].tolist() == [0.25, 0.75]
