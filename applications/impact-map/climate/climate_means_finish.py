#!/usr/bin/env python
"""Ensemble statistics over the per-model climate window means.

The GDPCIR ensemble is used unweighted: no SMME weight table exists for
CMIP6, only for CMIP5 (the March 2026 tables carry CMIP5 model names).
The ensemble is also ragged, 23 models in ssp245, 20 in ssp370 and 22 in
ssp585, so model counts are carried through to the output rather than
assumed constant.
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
VARIABLES = ["tas_annual", "tas_jja", "tas_djf"]
KEYS = ["weight", "scenario", "window", "region"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="climate_means")
    ap.add_argument("--levels", default="adm0,adm1,adm2")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.src}/windows_*.parquet"))
    if not files:
        raise SystemExit(f"no inputs under {args.src}")
    print(f"inputs: {len(files)} files")

    # ensemble composition, stated rather than assumed
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

        if not out:
            print(f"{level}: nothing")
            continue
        res = pd.concat(out, ignore_index=True)
        res["level"] = level
        path = f"{args.src}/ensemble_{level}.parquet"
        res.to_parquet(path, index=False)
        print(f"wrote {path}  rows={len(res)}\n", flush=True)


if __name__ == "__main__":
    main()
