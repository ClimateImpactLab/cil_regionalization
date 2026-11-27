"""The package carries no fact of one cluster or one cloud account. A
queue, a core count, a project and a path are what a caller declares."""

from __future__ import annotations

from pathlib import Path

import pytest

import cil_regionalization

PACKAGE = Path(cil_regionalization.__file__).resolve().parent
MODULES = sorted(PACKAGE.rglob("*.py"))
FORBIDDEN = (
    "/project/",
    "/" + "Volumes/",  # split so test_no_mount_paths.py does not match it
    "/shares/",
    "/home/",
    "conda",
    "sbatch",
    "battuta",
    "caslake",
    "midway",
    "compute-impactlab",
    "cilresearch",
    "temp_workspace",
    '"cil"',
    "'cil'",
)


def test_every_module_is_read():
    """So that a module the search misses cannot pass by being left out."""
    assert len(MODULES) > 30
    assert PACKAGE / "compute.py" in MODULES
    assert PACKAGE / "pipelines" / "montecarlo.py" in MODULES


@pytest.mark.parametrize("module", MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_module_names_a_site(module):
    """It went red on compute.py, config.py, legacy_compare.py,
    cross_backend_compare.py and legacy_export.py, which named a
    cluster's queue and paths and a cloud account's project. It checked
    montecarlo.py alone before."""
    source = module.read_text()
    found = [word for word in FORBIDDEN if word in source]
    assert not found, f"{module.relative_to(PACKAGE)} names {found}"
