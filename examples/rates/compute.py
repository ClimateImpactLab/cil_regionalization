"""Shared computation for the Mexico rates example.

The notebook and the interactive map builder both aggregate the same
way; this module holds that one way so they cannot drift. Everything
reads the committed data next to this file.

The variable is the effect of climate change on the mortality rate.
Each Monte Carlo leaf stores `rebased`, the scenario's impact relative
to its own 2001 to 2010 average. That value still contains the income
and adaptation trend, so it is not the effect of climate change on its
own. The effect of climate change is the rebased full adaptation
impact minus the rebased histclim impact, where histclim resamples
historical weather under the same income growth and adaptation. The
Climate Impact Lab memo "The art of rebasing and histclim" states the
convention; `load_draw` performs the subtraction.

The ratio route: multiply each region's effect by its population to
get death counts, aggregate counts and population separately with the
per_source weights, and divide at the target. The result is the
scenario's own rate for each unit. Dividing before aggregating, or
averaging rates with fixed weights, answers a different question; see
the notebook.

The population is the one embedded in the new-socioeconomics
projection, recovered from the tree's levels files; see
prepare_data.py. Two impact regions (MEX.25.1309 and MEX.23.1234,
together fewer than 150 people) hold no values in this tree and are
excluded.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import cil_regionalization as cilreg

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA_VERSION = "world-combo-201710"
MONTECARLO = DATA / "montecarlo"
WINDOWS = {"2080-2099": (2080, 2099), "2090-2099": (2090, 2099)}
PERCENTILES = [0.05, 0.5, 0.95]


def draws() -> list[tuple[str, str]]:
    """The (batch, climate model) pairs the sample holds."""
    out = []
    for bdir in sorted(MONTECARLO.iterdir()):
        for gdir in sorted((bdir / "rcp85").iterdir()):
            if gdir.is_dir():
                out.append((bdir.name, gdir.name))
    return out


def _read_rebased(batch: str, gcm: str, filename: str, col: str) -> pd.DataFrame:
    df = cilreg.read_netcdf_leaf(
        MONTECARLO / batch / "rcp85" / gcm / "low" / "SSP3" / filename,
        variables=["rebased", "regions"],
        region_dim="region",
        region_col="region_index",
    ).rename(columns={"regions": "hierid", "rebased": col})
    return df[["hierid", "year", col]]


def load_draw(batch: str, gcm: str) -> pd.DataFrame:
    """One draw's effect of climate change joined to the population,
    with death counts computed per region and year.

    The effect is the rebased full adaptation impact minus the rebased
    histclim impact, both read from the committed leaf pair.
    """
    full = _read_rebased(
        batch, gcm, "Agespec_interaction_response-combined.nc4", "fulladapt")
    hist = _read_rebased(
        batch, gcm, "Agespec_interaction_response-combined-histclim.nc4", "histclim")
    df = full.merge(hist, on=["hierid", "year"])
    df["effect"] = df["fulladapt"] - df["histclim"]
    pop = pd.read_parquet(DATA / "population.parquet")
    df = df.merge(pop, on=["hierid", "year"])
    df["deaths"] = df["effect"] * df["population"]
    return df


def load_weights(version: str, direction: str) -> "cilreg.WeightsArtifact":
    name = {"2.0": "gadm20", "4.1": "gadm41"}[version]
    return cilreg.WeightsArtifact.load(
        DATA / "weights" / f"{name}_adm2_{direction}"
    )


def target_keys(version: str) -> list[str]:
    return ["ISO", "ID_1", "ID_2"] if version == "2.0" else ["GID_0", "GID_1", "GID_2"]


def ratio_rate(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Deaths per 100,000 people per target unit and year, ratio route.

    The GADM 4.1 weights record partially covered coastal regions, so
    applying them requires acknowledging that with
    allow_partial_coverage; see the weight manifests.
    """
    w = load_weights(version, "per_source")
    d = df[df["hierid"].isin(set(w.frame["hierid"]))]
    kw = dict(
        kind="extensive",
        weight="pop",
        data_version=DATA_VERSION,
        restrict_to_sources={(h,) for h in d["hierid"].unique()},
    )
    if version == "4.1":
        kw["allow_partial_coverage"] = True
    deaths = cilreg.apply_weights(
        w, d[["hierid", "year", "deaths"]], value_col="deaths", **kw
    ).frame
    people = cilreg.apply_weights(
        w, d[["hierid", "year", "population"]], value_col="population", **kw
    ).frame
    keys = target_keys(version)
    out = deaths.merge(people, on=keys + ["year"])
    out["per100k"] = out["deaths"] / out["population"] * 1e5
    return out


def window_percentiles(version: str) -> pd.DataFrame:
    """Per-unit percentiles of the window-mean rate across the draws
    (batch by climate model): aggregate each draw first, average over
    the window, then take percentiles across draws."""
    keys = target_keys(version)
    per_draw = []
    for batch, gcm in draws():
        r = ratio_rate(load_draw(batch, gcm), version)
        for label, (lo, hi) in WINDOWS.items():
            win = r[r["year"].between(lo, hi)]
            g = win.groupby(keys)[["deaths", "population"]].mean().reset_index()
            g["per100k"] = g["deaths"] / g["population"] * 1e5
            g["window"] = label
            g["batch"] = batch
            g["gcm"] = gcm
            per_draw.append(g[keys + ["window", "batch", "gcm", "per100k"]])
    pooled = pd.concat(per_draw, ignore_index=True)
    q = (
        pooled.groupby(keys + ["window"])["per100k"]
        .quantile(PERCENTILES)
        .unstack()
    )
    q.columns = [f"q{int(p * 100):02d}" for p in PERCENTILES]
    return q.reset_index()
