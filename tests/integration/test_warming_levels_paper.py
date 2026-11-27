"""The settings of examples/warming_levels/ar5_cmip5_smme.toml give the
models the Inequality Paper extracted at each warming level."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cil_regionalization.pipelines.montecarlo import load_warming_levels

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples" / "warming_levels" / "ar5_cmip5_smme.toml"
BINS = ROOT / "data" / "external" / "inequality_gwl_bins.csv"


def test_the_members_at_each_level_are_the_papers():
    """Every member of the table is declared, as every one was in the
    paper's tree, and the config says which join. The member with no
    temperature, surrogate_GFDL-ESM2G_06 under rcp45, does not join and
    so is never assessed.

    It went red with the config's temperature years cut to nineteen,
    with its offset set to 0, and with join set to "declared"."""
    warming = load_warming_levels(CONFIG)
    table = warming.read_table()
    levels = warming.assign(table, table.members)
    derived = {
        (period.label, level, rcp, gcm)
        for (gcm, rcp), assigned in levels.levels.items()
        for period, level in zip(levels.periods, assigned)
        if level is not None
    }
    paper = pd.read_csv(BINS, comment="#", dtype={"level": float})
    assert derived == set(paper.itertuples(index=False, name=None))
    assert len(levels.levels) == 42 and len(levels.excluded) == 24
