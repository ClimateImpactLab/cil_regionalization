"""docs/reference.md is written by docs/make_reference.py from the
package, so it cannot fall behind the code without these failing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import cil_regionalization as cilreg

ROOT = Path(__file__).resolve().parents[2]


def _maker():
    spec = importlib.util.spec_from_file_location(
        "make_reference", ROOT / "docs" / "make_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reference_is_what_the_package_gives():
    written = (ROOT / "docs" / "reference.md").read_text()
    assert written == _maker().render(), (
        "docs/reference.md is out of date. Run python docs/make_reference.py"
    )


def test_an_export_in_no_group_is_refused(monkeypatch):
    monkeypatch.setattr(cilreg, "__all__", [*cilreg.__all__, "Period2"])
    with pytest.raises(ValueError, match=r"leaves out \['Period2'\]"):
        _maker().render()


def test_a_config_key_is_listed_with_its_default():
    reference = _maker().render()
    assert "| `memory_gb` | float, above 0 | not set |" in reference
    assert '| `summation` | "fast" or "exact" | required |' in reference
    assert "### `[[stats.warming_levels.periods]]`" in reference
