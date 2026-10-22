#!/usr/bin/env python
"""Mean-temperature climate variables from GDPCIR monthly tavg, on disk.

Produces the three mean-temperature variables of the published Impact Map
table (annual, JJA, DJF) in degrees Fahrenheit, per model, scenario, window
and GADM 2.0 unit, for both population and area weighting.

The monthly tavg stores carry `method: average of downscaled tasmax and
downscaled tasmin`, so these are CIL's own daily means aggregated to months.
A day-weighted average of monthly means equals the average of the dailies
exactly, so nothing is approximated here.

The two threshold-day-count variables (days below 32F, days above 95F)
cannot be derived from monthly means and are not produced by this script.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

ROOT = Path("/project/cil/gcp/climate/CMIP6_GDPCIR/monthly/tavg")
WEIGHTS = Path("/project/cil/home_dirs/scadavidsanchez/repos/"
               "climate-and-damages-aggregations/data/weights")

NLAT, NLON = 720, 1440
SCENARIOS = ["ssp245", "ssp370", "ssp585"]

# window -> (first year, last year, which run supplies it)
WINDOWS = {
    "1986-2005": (1986, 2005, "historical"),
    "2020-2039": (2020, 2039, "scenario"),
    "2040-2059": (2040, 2059, "scenario"),
    "2080-2099": (2080, 2099, "scenario"),
}


def find_store(activity, model, experiment):
    pat = str(ROOT / activity / "*" / model / experiment / "*" / "*.zarr")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"no store for {model}/{experiment}: {pat}")
    return hits[-1]


def enumerate_tasks():
    """Stable list of (model, scenario) pairs that exist on disk."""
    models = sorted({Path(p).name for p in
                     glob.glob(str(ROOT / "CMIP" / "*" / "*"))})
    tasks = []
    for m in models:
        try:
            find_store("CMIP", m, "historical")
        except FileNotFoundError:
            continue
        for s in SCENARIOS:
            try:
                find_store("ScenarioMIP", m, s)
            except FileNotFoundError:
                continue
            tasks.append((m, s))
    return tasks


def load_weight_frame(level):
    src = "adm2" if level == "adm2" else "adm1"
    d = WEIGHTS / f"grid025_gadm20_{src}_per_destination"
    man = json.loads((d / "weights.manifest.json").read_text())
    df = pd.read_parquet(d / "weights.parquet")
    ids = list(man["id_fields"])

    if level == "adm0":
        ids = ["ISO"]
        df = (df.groupby(["ISO", "cell_ix", "cell_iy"], observed=True)
                [["pop_raw", "area_raw"]].sum().reset_index())
        for w in ("pop", "area"):
            tot = df.groupby("ISO", observed=True)[f"{w}_raw"].transform("sum")
            df[f"{w}wt"] = np.where(tot > 0, df[f"{w}_raw"] / tot, 0.0)
        # countries with no population at all fall back to area, matching
        # the artifact's own pop fallback policy
        zero = df.groupby("ISO", observed=True)["popwt"].transform("sum") == 0
        df.loc[zero, "popwt"] = df.loc[zero, "areawt"]

    return df, ids


def build_matrix(df, ids, wcol):
    keys = (df[ids[0]].astype(str) if len(ids) == 1
            else df[ids].astype(str).agg("|".join, axis=1))
    cats = pd.Categorical(keys)
    rows = cats.codes.astype("int64")
    cols = df["cell_iy"].to_numpy("int64") * NLON + df["cell_ix"].to_numpy("int64")
    M = sp.csr_matrix(
        (df[wcol].to_numpy("float64"), (rows, cols)),
        shape=(len(cats.categories), NLAT * NLON),
    )
    return M, list(cats.categories)


def window_fields(store, y0, y1):
    """Per-year gridded annual, JJA and DJF means in degrees Fahrenheit."""
    ds = xr.open_zarr(store, consolidated=True)
    sel = ds["tavg"].sel(time=slice(f"{y0 - 1}-12-01", f"{y1}-12-31"))
    yr = sel.time.dt.year.values
    mo = sel.time.dt.month.values
    dim = sel.time.dt.days_in_month.values.astype("float64")
    arr = sel.values.astype("float32")        # (n_months, NLAT, NLON), kelvin

    years = np.arange(y0, y1 + 1)
    picks = {
        "tas_annual": lambda Y: (np.where(yr == Y)[0], 12),
        "tas_jja": lambda Y: (np.where((yr == Y) & np.isin(mo, [6, 7, 8]))[0], 3),
        "tas_djf": lambda Y: (np.concatenate([
            np.where((yr == Y - 1) & (mo == 12))[0],
            np.where((yr == Y) & np.isin(mo, [1, 2]))[0],
        ]), 3),
    }

    out = {}
    for name, fn in picks.items():
        vals = np.empty((len(years), NLAT * NLON), dtype="float32")
        for i, Y in enumerate(years):
            idx, expect = fn(int(Y))
            if len(idx) != expect:
                raise ValueError(
                    f"{name} year {Y}: got {len(idx)} months, expected {expect}"
                )
            w = dim[idx]
            k = np.tensordot(w, arr[idx], axes=(0, 0)) / w.sum()
            vals[i] = ((k - 273.15) * 9.0 / 5.0 + 32.0).reshape(-1)
        out[name] = vals
    return out, years


def reduce_to_regions(M, den, vals):
    """Weighted mean per region, renormalized over cells with data."""
    x = np.nan_to_num(vals, nan=0.0).astype("float64")
    num = M @ x.T                                   # (n_regions, n_years)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = num / den[:, None]
    res[den <= 0, :] = np.nan
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--model")
    ap.add_argument("--scenario")
    ap.add_argument("--levels", default="adm0")
    ap.add_argument("--out", default="climate_means")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    tasks = enumerate_tasks()
    if args.list:
        for i, (m, s) in enumerate(tasks):
            print(f"{i}\t{m}\t{s}")
        print(f"total: {len(tasks)}")
        return

    if args.task_id is not None:
        model, scenario = tasks[args.task_id]
    elif args.model and args.scenario:
        model, scenario = args.model, args.scenario
    else:
        raise SystemExit("need --task-id or --model with --scenario")

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"model={model} scenario={scenario} levels={levels}", flush=True)

    # weight matrices and land denominators, built once
    mats = {}
    for level in levels:
        wdf, ids = load_weight_frame(level)
        for wname in ("pop", "area"):
            M, regions = build_matrix(wdf, ids, f"{wname}wt")
            mats[(level, wname)] = (M, regions, ids)
        print(f"  {level}: {len(mats[(level, 'pop')][1])} regions", flush=True)

    rows, yearly = [], []
    mask_den = {}

    for window, (y0, y1, which) in WINDOWS.items():
        store = (find_store("CMIP", model, "historical") if which == "historical"
                 else find_store("ScenarioMIP", model, scenario))
        fields, years = window_fields(store, y0, y1)
        print(f"  {window}: loaded from {Path(store).parent.parent.name}", flush=True)

        for level in levels:
            for wname in ("pop", "area"):
                M, regions, ids = mats[(level, wname)]
                key = (level, wname)
                if key not in mask_den:
                    valid = (~np.isnan(fields["tas_annual"][0])).astype("float64")
                    mask_den[key] = np.asarray(M @ valid).ravel()
                den = mask_den[key]

                for var, vals in fields.items():
                    series = reduce_to_regions(M, den, vals)   # regions x years
                    frame = pd.DataFrame({
                        "model": model,
                        "scenario": scenario,
                        "level": level,
                        "weight": wname,
                        "variable": var,
                        "window": window,
                        "region": regions,
                        "value": np.nanmean(series, axis=1),
                    })
                    rows.append(frame)

                    if level == "adm0":
                        ydf = pd.DataFrame(series, columns=[str(y) for y in years])
                        ydf.insert(0, "region", regions)
                        ydf.insert(0, "variable", var)
                        ydf.insert(0, "weight", wname)
                        ydf.insert(0, "window", window)
                        ydf.insert(0, "scenario", scenario)
                        ydf.insert(0, "model", model)
                        yearly.append(ydf)

    tag = f"{model}_{scenario}"
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(outdir / f"windows_{tag}.parquet", index=False)
    print(f"wrote windows_{tag}.parquet  rows={len(out)}", flush=True)

    if yearly:
        ydf = pd.concat(yearly, ignore_index=True)
        ydf.to_parquet(outdir / f"yearly_adm0_{tag}.parquet", index=False)
        print(f"wrote yearly_adm0_{tag}.parquet  rows={len(ydf)}", flush=True)


if __name__ == "__main__":
    main()
