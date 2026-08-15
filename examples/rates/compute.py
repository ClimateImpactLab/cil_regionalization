"""Shared computation for the Mexico rates example.

The notebook and the interactive map builder both aggregate the same
way; this module holds that one way so they cannot drift. Everything
reads the committed data next to this file.

The ratio route: multiply each region's rate by its embedded
population to get death counts, aggregate counts and population
separately with the per_source weights, and divide at the target. The
result is the scenario's own rate for each unit. Dividing before
aggregating, or averaging rates with fixed weights, answers a
different question; see the notebook.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import cil_regionalization as cilreg

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA_VERSION = "world-combo-201710"
BATCH = DATA / "montecarlo" / "batch0" / "rcp85"
WINDOWS = {"2080-2099": (2080, 2099), "2090-2099": (2090, 2099)}
PERCENTILES = [0.05, 0.5, 0.95]


def gcms() -> list[str]:
    return sorted(p.name for p in BATCH.iterdir() if p.is_dir())


def load_draw(gcm: str) -> pd.DataFrame:
    """One climate model's rates joined to the embedded population,
    with death counts computed per region and year."""
    rates = cilreg.read_netcdf_leaf(
        BATCH / gcm / "low" / "SSP3" / "Agespec_interaction_response-combined.nc4",
        variables=["rebased", "regions"],
        region_dim="region",
        region_col="region_index",
    ).rename(columns={"regions": "hierid"})[["hierid", "year", "rebased"]]
    pop = pd.read_parquet(DATA / "population.parquet")
    df = rates.merge(pop, on=["hierid", "year"])
    df["deaths"] = df["rebased"] * df["population"]
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
    """Per-unit percentiles of the window-mean rate across the climate
    models: aggregate each draw first, average over the window, then
    take percentiles across draws."""
    keys = target_keys(version)
    per_draw = []
    for gcm in gcms():
        r = ratio_rate(load_draw(gcm), version)
        for label, (lo, hi) in WINDOWS.items():
            win = r[r["year"].between(lo, hi)]
            g = win.groupby(keys)[["deaths", "population"]].mean().reset_index()
            g["per100k"] = g["deaths"] / g["population"] * 1e5
            g["window"] = label
            g["gcm"] = gcm
            per_draw.append(g[keys + ["window", "gcm", "per100k"]])
    draws = pd.concat(per_draw, ignore_index=True)
    q = (
        draws.groupby(keys + ["window"])["per100k"]
        .quantile(PERCENTILES)
        .unstack()
    )
    q.columns = [f"q{int(p * 100):02d}" for p in PERCENTILES]
    return q.reset_index()
