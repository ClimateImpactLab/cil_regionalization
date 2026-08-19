"""Shared computation for the India labor example.

The variable is the effect of climate change on labor supply, in
minutes worked per worker per day. The files' units attribute says
"portion"; that is a projection-system default, not a labor
statement. Each file is the impact relative to the scenario's own
2001 to 2010 average. The effect of climate change is the file minus
its -histclim sibling. Losses are negative.

`rebased` already combines the two risk groups, so nothing here
combines them. The valuation applies one income-based rate to the
combined minutes, so there is no valued high-risk or low-risk object.

The physical route is a population-weighted mean: minutes times
population and population aggregate with the per_source weights and
divide at the state. Population is not a worker count; the labor
share sits inside the valuation constant, so multiplying the minutes
by a worker count would double-count it.

The valued route multiplies the effect by the committed wage factor
(population times income times 0.00487) and aggregates it as a total
in 2005 PPP USD. The factor comes from the tree's own wage files;
prepare_data.py recovers it and the notebook checks it against the
committed wage pair. Regional GDP is the factor divided by 0.00487,
so the loss as a share of GDP is the ratio route again: value and GDP
aggregate separately and divide at the state.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import cil_regionalization as cilreg

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA_VERSION = "world-combo-201710"
MONTECARLO = DATA / "montecarlo"
WINDOW = (2090, 2099)
STEM = "uninteracted_main_model"
KEYS = ["ISO", "ID_1"]
WAGE_CONSTANT = 0.00487


def draws() -> list[tuple[str, str]]:
    """The (batch, climate model) pairs the sample holds."""
    out = []
    for bdir in sorted(MONTECARLO.iterdir()):
        for gdir in sorted((bdir / "rcp85").iterdir()):
            if gdir.is_dir():
                out.append((bdir.name, gdir.name))
    return out


def read_rebased(batch: str, gcm: str, filename: str, col: str) -> pd.DataFrame:
    df = cilreg.read_netcdf_leaf(
        MONTECARLO / batch / "rcp85" / gcm / "low" / "SSP3" / filename,
        variables=["rebased", "regions"],
        region_dim="region",
        region_col="region_index",
    ).rename(columns={"regions": "hierid", "rebased": col})
    return df[["hierid", "year", col]]


def load_draw(batch: str, gcm: str) -> pd.DataFrame:
    """One draw's effect in minutes per worker per day, with the
    population, the ratio route numerator, and the valued effect."""
    scen = read_rebased(batch, gcm, f"{STEM}.nc4", "scen")
    hist = read_rebased(batch, gcm, f"{STEM}-histclim.nc4", "hist")
    df = scen.merge(hist, on=["hierid", "year"])
    df["minutes"] = df["scen"].astype("float64") - df["hist"].astype("float64")
    pop = pd.read_parquet(DATA / "population.parquet")
    pop["population"] = pop["population"].astype("float64")
    df = df.merge(pop, on=["hierid", "year"])
    factor = pd.read_parquet(DATA / "wage_factor.parquet")
    df = df.merge(factor, on=["hierid", "year"])
    df["minutes_pop"] = df["minutes"] * df["population"]
    df["value"] = df["minutes"] * df["factor"]
    df["gdp"] = df["factor"] / WAGE_CONSTANT
    return df[["hierid", "year", "minutes", "population", "minutes_pop",
               "value", "gdp"]]


def load_weights() -> "cilreg.WeightsArtifact":
    return cilreg.WeightsArtifact.load(DATA / "weights" / "gadm20_adm1_per_source")


def _apply(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    w = load_weights()
    d = df[df["hierid"].isin(set(w.frame["hierid"]))]
    return cilreg.apply_weights(
        w,
        d[["hierid", "year", value_col]],
        kind="extensive",
        weight="pop",
        value_col=value_col,
        data_version=DATA_VERSION,
        restrict_to_sources={(h,) for h in d["hierid"].unique()},
    ).frame


def ratio_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes per worker per day per state and year."""
    minutes = _apply(df, "minutes_pop")
    people = _apply(df, "population")
    out = minutes.merge(people, on=KEYS + ["year"])
    out["minutes"] = out["minutes_pop"] / out["population"]
    return out


def valued_total(df: pd.DataFrame) -> pd.DataFrame:
    """2005 PPP USD per state and year, allocated as a total."""
    return _apply(df, "value")


def pct_gdp(df: pd.DataFrame) -> pd.DataFrame:
    """Loss as percent of GDP per state and year: value and GDP
    aggregate separately, divide at the state."""
    value = _apply(df, "value")
    gdp = _apply(df, "gdp")
    out = value.merge(gdp, on=KEYS + ["year"])
    out["pct_gdp"] = out["value"] / out["gdp"] * 100
    return out
