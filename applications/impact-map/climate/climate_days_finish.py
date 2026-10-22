#!/usr/bin/env python
"""Ensemble statistics over the five daily-derived climate variables.

The GDPCIR ensemble is unweighted: no SMME weight table exists for CMIP6,
only for CMIP5. The ensemble is also ragged, 23 models in ssp245, 20 in
ssp370 and 22 in ssp585, with all 25 in historical, so model counts are
carried through to the output rather than assumed constant.

The historical window is emitted once under scenario "historical" over
all 25 models, not replicated per scenario. The monthly product instead
carried it per scenario over that scenario's subset, so the two will
differ slightly for that window by ensemble membership alone.

Ends with a cross-check of the three means against the monthly route,
which is an empirical test of the day-weighted monthly shortcut.
"""
import argparse
import glob

import numpy as np
import pandas as pd

STATS = {
    "mean": lambda a: np.nanmean(a, axis=1),
    "min": lambda a: np.nanmin(a, axis=1),
    "q5": lambda a: np.nanpercentile(a, 5, axis=1),
    "q17": lambda a: np.nanpercentile(a, 17, axis=1),
    "q25": lambda a: np.nanpercentile(a, 25, axis=1),
    "q50": lambda a: np.nanpercentile(a, 50, axis=1),
    "q75": lambda a: np.nanpercentile(a, 75, axis=1),
    "q83": lambda a: np.nanpercentile(a, 83, axis=1),
    "q95": lambda a: np.nanpercentile(a, 95, axis=1),
    "max": lambda a: np.nanmax(a, axis=1),
}
VARIABLES = ["tas_annual", "tas_jja", "tas_djf",
             "days_above_95F", "days_below_32F"]
MEANS = ["tas_annual", "tas_jja", "tas_djf"]
KEYS = ["weight", "scenario", "window", "region"]


def cross_check(src):
    """Contrast the three means against the monthly route at ADM0."""
    try:
        daily = pd.read_parquet(f"{src}/ensemble_adm0.parquet")
        monthly = pd.read_parquet("climate_means/ensemble_adm0.parquet")
    except FileNotFoundError as exc:
        print(f"cross-check skipped: {exc}")
        return

    k = ["weight", "scenario", "window", "region", "variable"]
    d = daily[daily.variable.isin(MEANS)][k + ["mean", "n_models"]]
    m = monthly[monthly.variable.isin(MEANS)][k + ["mean", "n_models"]]
    j = d.merge(m, on=k, suffixes=("_daily", "_monthly"))
    j = j[j.n_models_daily == j.n_models_monthly]
    if j.empty:
        print("cross-check: no comparable rows")
        return

    j["diff"] = j["mean_daily"] - j["mean_monthly"]
    print("=== cross-check, daily minus monthly, degrees F ===")
    print(f"rows compared: {len(j)}  "
          f"(only where both routes used the same model count)")
    print(j.groupby("variable", observed=True)["diff"]
           .agg(["count", "mean", "std",
                 lambda s: s.abs().max()])
           .rename(columns={"<lambda_0>": "max_abs"})
           .round(4).to_string())
    worst = j.reindex(j["diff"].abs().sort_values(ascending=False).index).head(8)
    print()
    print("largest absolute differences:")
    print(worst[["region", "scenario", "window", "variable", "weight",
                 "mean_daily", "mean_monthly", "diff"]]
          .round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="climate_days")
    ap.add_argument("--levels", default="adm0,adm1,adm2")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.src}/days_*.parquet"))
    if not files:
        raise SystemExit(f"no inputs under {args.src}")
    print(f"inputs: {len(files)} files")

    comp = pd.concat(
        [pd.read_parquet(f, columns=["model", "scenario"]).drop_duplicates()
         for f in files],
        ignore_index=True,
    ).drop_duplicates()
    print("\nmodels per scenario:")
    print(comp.groupby("scenario", observed=True)["model"].nunique().to_string())
    print()

    for level in [x.strip() for x in args.levels.split(",") if x.strip()]:
        out = []
        for var in VARIABLES:
            parts = []
            for f in files:
                d = pd.read_parquet(
                    f,
                    filters=[("level", "==", level), ("variable", "==", var)],
                    columns=["model", "weight", "scenario", "window",
                             "region", "value"],
                )
                if len(d):
                    parts.append(d)
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            for c in ["model", "weight", "scenario", "window", "region"]:
                df[c] = df[c].astype("category")

            piv = df.pivot_table(index=KEYS, columns="model",
                                 values="value", aggfunc="mean",
                                 observed=True)
            arr = piv.to_numpy(dtype="float64")

            res = piv.index.to_frame(index=False)
            res["variable"] = var
            res["n_models"] = np.sum(~np.isnan(arr), axis=1)
            with np.errstate(invalid="ignore", all="ignore"):
                for name, fn in STATS.items():
                    res[name] = fn(arr)
            out.append(res)
            print(f"  {level}/{var}: {len(res)} groups, "
                  f"{arr.shape[1]} model columns", flush=True)
            del df, piv, arr

        if not out:
            print(f"{level}: nothing")
            continue
        res = pd.concat(out, ignore_index=True)
        res["level"] = level
        path = f"{args.src}/ensemble_{level}.parquet"
        res.to_parquet(path, index=False)
        print(f"wrote {path}  rows={len(res)}\n", flush=True)
        del res, out

    print()
    cross_check(args.src)


if __name__ == "__main__":
    main()
