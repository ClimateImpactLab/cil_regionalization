"""Shared computation for the United States energy example.

The sector's uncertainty is the delta method, not Monte Carlo draws:
each file carries the gradient of the impact with respect to the 80
regression coefficients, plus their covariance matrix. The committed
extract holds the effect-of-climate-change gradient (full adaptation
minus histclim) per climate model for year 2099, other_energy only.

Units carry two traps, both from stale metadata. The physical values
are GJ per capita, although the files say kWh/pc. The raw dollar
values understate by exactly 0.0036, because a GJ quantity was
multiplied by a per-kWh price; divide by GJ_PER_KWH to repair. Raw
dollars are 2005 USD; DOLLARS_2019 deflates them to 2019 USD, the
year the release's figures use. Both are applied here, so the
notebook shows repaired 2019 dollars.

The valued gradient is the physical gradient times population times
the country-year price; prepare_data.py verifies that against the
tree's own priced gradient files. Means need no separate files: the
response is linear in the coefficients, so the mean is the gradient
times the coefficient vector from the release's csvv.
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

import cil_regionalization as cilreg

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
YEAR = 2099
GJ_PER_KWH = 0.0036
DOLLARS_2019 = 1.273526
KEYS = ["ISO", "ID_1"]


def load_static():
    """Regions, per-model effect gradients, covariance matrix,
    coefficients, factor, and population, in one aligned bundle."""
    with netCDF4.Dataset(DATA / "effect_gradients.nc4") as ds:
        regions = [str(r) for r in ds.variables["regions"][:]]
        gcms = [str(g) for g in ds.variables["gcm"][:]]
        effects = np.asarray(ds.variables["effect_bcde"][:], dtype="float64")
        vcv = np.asarray(ds.variables["vcv"][:], dtype="float64")
    gammas = pd.read_parquet(DATA / "gammas.parquet")["gamma"].values
    factor = (
        pd.read_parquet(DATA / "wage_price_factor.parquet")
        .set_index("hierid")["factor"].reindex(regions).values
    )
    population = (
        pd.read_parquet(DATA / "population.parquet")
        .set_index("hierid")["population"].reindex(regions).values
    )
    return regions, gcms, effects, vcv, gammas, factor, population


def load_weights() -> "cilreg.WeightsArtifact":
    return cilreg.WeightsArtifact.load(DATA / "weights" / "gadm20_adm1_per_source")


def valued_states(weights, regions, gradient, vcv, gammas, factor):
    """Mean and sd of the valued effect per state, repaired dollars."""
    targets, G = cilreg.aggregate_gradient(
        weights, regions, gradient * factor[None, :]
    )
    stats = cilreg.delta_method(G, vcv, coefficients=gammas)
    out = pd.concat([targets, stats * DOLLARS_2019 / GJ_PER_KWH], axis=1)
    return out


def physical_states(weights, regions, gradient, vcv, gammas, population):
    """Mean and sd of the per-capita effect per state, GJ per capita."""
    targets, G = cilreg.aggregate_gradient(
        weights, regions, gradient, scale=population, normalize=True
    )
    stats = cilreg.delta_method(G, vcv, coefficients=gammas)
    return pd.concat([targets, stats], axis=1)


def naive_valued_sd(weights, regions, gradient, vcv, factor):
    """The independent-regions sd per state: allocate each region's
    own variance by squared shares and sum, ignoring the covariance
    the shared coefficients create."""
    valued = gradient * factor[None, :]
    per_region_var = np.einsum("cr,cd,dr->r", valued, vcv, valued)
    frame = weights.frame
    idx = {r: i for i, r in enumerate(regions)}
    rows = frame[frame["hierid"].isin(idx)]
    var = rows.assign(
        v=[per_region_var[idx[h]] * w ** 2
           for h, w in zip(rows["hierid"], rows["popwt"])]
    ).groupby(KEYS)["v"].sum().reset_index()
    var["sd"] = np.sqrt(var["v"]) * DOLLARS_2019 / GJ_PER_KWH
    return var[KEYS + ["sd"]]
