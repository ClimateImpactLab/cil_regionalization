"""`assign_warming_levels` on the mortality paper's temperature table
gives the models the Inequality Paper extracted at each warming level."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import cil_regionalization as cilreg

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "examples" / "france" / "data" / "GMTanom_all_temp_2001_2010_smooth.csv"
BINS = ROOT / "tests" / "data" / "external" / "inequality_gwl_bins.csv"


def test_the_members_at_each_level_are_the_papers():
    """From the whole table, with the surrogates excluded as the paper's
    reported outputs exclude them. The periods are the twenty years from
    2040 and from 2080, as finding_gwl_bins.R filters them. The member
    with no temperature, surrogate_GFDL-ESM2G_06 under rcp45, is a
    surrogate and so never assessed.

    It went red with the period means taken over nineteen years, with
    the offset left out, and with the surrogates included."""
    levels = cilreg.assign_warming_levels(
        cilreg.read_gmst(TABLE, member_cols=["gcm", "rcp"]),
        reference_period=(1986, 2005),
        offset=0.61,
        levels=[1.5, 2, 3, 4, 5],
        half_width=0.25,
        periods=[
            cilreg.Period(gmst_span=(2040, 2059), impact_span=(2040, 2059)),
            cilreg.Period(gmst_span=(2080, 2099), impact_span=(2080, 2099)),
        ],
        smoothing="recorded nowhere",
        surrogates="exclude",
    )
    derived = {
        (period.label, level, rcp, gcm)
        for (gcm, rcp), assigned in levels.levels.items()
        for period, level in zip(levels.periods, assigned)
        if level is not None
    }
    paper = pd.read_csv(BINS, comment="#", dtype={"level": float})
    assert derived == set(paper.itertuples(index=False, name=None))
    assert len(levels.levels) == 42 and len(levels.excluded) == 24
