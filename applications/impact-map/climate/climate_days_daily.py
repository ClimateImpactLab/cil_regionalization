#!/usr/bin/env python
"""The five Impact Map climate variables from GDPCIR daily stores.

Reads daily tasmax and tasmin from the Planetary Computer one year at a
time, derives the per-cell variables with the package's own
``derive_daily``, aggregates to GADM 2.0 units and discards the year.
Nothing is staged, so the transfer never lands on disk.

This supersedes the monthly route rather than complementing it. Monthly
means cannot carry the threshold counts, and cannot apply the swap
correction for cell-days that arrive with tasmin above tasmax, because
by then the two have already been averaged together. The monthly product
stays as an independent check on the three means.

Means are converted to Fahrenheit here; ``derive_daily`` returns Celsius
while the monthly product is already in Fahrenheit.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cil_regionalization.bcsd_io import december, derive_daily
from cil_regionalization.gdpcir_io import (
    gdpcir_stac_assets,
    gdpcir_year,
    open_gdpcir_daily,
)

from climate_means_monthly import (
    build_matrix,
    enumerate_tasks,
    load_weight_frame,
    reduce_to_regions,
)

NLAT, NLON = 720, 1440
VARIABLES = ("tasmax", "tasmin")
FIELDS = ("tas_annual", "tas_jja", "tas_djf",
          "days_above_95F", "days_below_32F")

HIST_WINDOWS = {"1986-2005": (1986, 2005)}
SCEN_WINDOWS = {
    "2020-2039": (2020, 2039),
    "2040-2059": (2040, 2059),
    "2080-2099": (2080, 2099),
}
RESIGN_SECONDS = 1500


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


class Assets:
    """Signed GDPCIR stores that re-resolve before their tokens expire."""

    def __init__(self, model, experiment):
        self.model = model
        self.experiment = experiment
        self._stores = {}
        self._at = 0.0

    def _resolve(self):
        found = gdpcir_stac_assets(self.model, self.experiment, VARIABLES)
        self._stores = {
            v: open_gdpcir_daily(href, v, **kw) for v, (href, kw) in found.items()
        }
        self._at = time.time()

    def dataset(self, variable):
        if not self._stores or time.time() - self._at > RESIGN_SECONDS:
            self._resolve()
        return self._stores[variable]

    def refresh(self):
        self._stores = {}
        self._resolve()

    def year(self, variable, year, retries=4):
        for attempt in range(retries):
            try:
                return gdpcir_year(self.dataset(variable), variable, year)
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                print(f"    retry {attempt + 1} on {variable} {year}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                time.sleep(20 * (attempt + 1))
                self.refresh()


def window_fields(assets, y0, y1):
    """Per-year gridded fields for a window, with a spin-up year for DJF."""
    product = list(range(y0, y1 + 1))
    out = {f: np.empty((len(product), NLAT * NLON), "float32") for f in FIELDS}
    dec_prev = None
    swapped = 0

    for year in range(y0 - 1, y1 + 1):
        t0 = time.time()
        tasmax = assets.year("tasmax", year)
        tasmin = assets.year("tasmin", year)
        moved = (tasmax.nbytes + tasmin.nbytes) / 1e9

        if year < y0:
            # read only to carry December into the first product year's DJF
            dec_prev = december(tasmax, tasmin)
            del tasmax, tasmin
            dt = max(time.time() - t0, 1e-6)
            print(f"    {year} spin-up: {moved:.2f} GB in {dt:.0f}s "
                  f"({moved * 1000 / dt:.0f} MB/s)", flush=True)
            continue

        d = derive_daily(tasmax, tasmin, dec_prev)
        dec_prev = december(tasmax, tasmin)
        del tasmax, tasmin
        swapped += d.n_swapped

        i = product.index(year)
        out["tas_annual"][i] = c_to_f(d.tas_annual).reshape(-1)
        out["tas_jja"][i] = c_to_f(d.tas_jja).reshape(-1)
        out["tas_djf"][i] = c_to_f(d.tas_djf).reshape(-1)
        out["days_above_95F"][i] = d.days_above_95F.reshape(-1)
        out["days_below_32F"][i] = d.days_below_32F.reshape(-1)

        dt = max(time.time() - t0, 1e-6)
        print(f"    {year}: {moved:.2f} GB in {dt:.0f}s "
              f"({moved * 1000 / dt:.0f} MB/s)  djf_days={d.n_days_djf}  "
              f"swapped={d.n_swapped}", flush=True)
        del d

    return out, swapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hist", "scen"], required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--levels", default="adm0,adm1,adm2")
    ap.add_argument("--out", default="climate_days")
    ap.add_argument("--grid-out", default=None,
                    help="also write the per-year gridded fields here, "
                         "so any later change of weights, level or "
                         "window costs no transfer")
    args = ap.parse_args()

    tasks = enumerate_tasks()
    if args.mode == "hist":
        models = sorted({m for m, _ in tasks})
        model = models[args.task_id]
        scenario = experiment = "historical"
        windows = HIST_WINDOWS
    else:
        model, scenario = tasks[args.task_id]
        experiment = scenario
        windows = SCEN_WINDOWS

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"mode={args.mode} model={model} experiment={experiment} "
          f"levels={levels}", flush=True)

    mats = {}
    for level in levels:
        wdf, ids = load_weight_frame(level)
        for wname in ("pop", "area"):
            mats[(level, wname)] = build_matrix(wdf, ids, f"{wname}wt")
        print(f"  {level}: {len(mats[(level, 'pop')][1])} regions", flush=True)

    # Exercise the reduction on a synthetic field before transferring
    # anything. A shape bug here costs a second now; found after the read
    # loop it costs a whole window of downloads.
    probe = np.zeros((1, NLAT * NLON), dtype="float32")
    probe[0, ::7] = 1.0
    for (lvl, wnm), (M, regions) in mats.items():
        den = np.asarray(M @ np.ones(NLAT * NLON, dtype="float64")).ravel()
        got = reduce_to_regions(M, den, probe)
        if got.shape != (len(regions), 1):
            raise SystemExit(
                f"reduction for {lvl}/{wnm} returned {got.shape}, "
                f"expected {(len(regions), 1)}"
            )
    print("  reduction self-check passed", flush=True)

    assets = Assets(model, experiment)
    rows, total_swapped = [], 0

    for window, (y0, y1) in windows.items():
        print(f"  {window}:", flush=True)
        fields, swapped = window_fields(assets, y0, y1)
        total_swapped += swapped

        if args.grid_out:
            gdir = Path(args.grid_out)
            gdir.mkdir(parents=True, exist_ok=True)
            gpath = gdir / f"grid_{model}_{scenario}_{window}.npz"
            np.savez(gpath,
                     years=np.arange(y0, y1 + 1),
                     nlat=np.array([NLAT]), nlon=np.array([NLON]),
                     **fields)
            print(f"    wrote {gpath.name} "
                  f"({gpath.stat().st_size / 1e9:.2f} GB)", flush=True)

        for level in levels:
            for wname in ("pop", "area"):
                M, regions = mats[(level, wname)]
                valid = (~np.isnan(fields["tas_annual"][0])).astype("float64")
                den = np.asarray(M @ valid).ravel()
                for var, vals in fields.items():
                    series = reduce_to_regions(M, den, vals)
                    rows.append(pd.DataFrame({
                        "model": model,
                        "scenario": scenario,
                        "level": level,
                        "weight": wname,
                        "variable": var,
                        "window": window,
                        "region": regions,
                        "value": np.nanmean(series, axis=1),
                        "source": "daily",
                    }))
        del fields

    out = pd.concat(rows, ignore_index=True)
    tag = f"{model}_{scenario}"
    out.to_parquet(outdir / f"days_{tag}.parquet", index=False)
    print(f"wrote days_{tag}.parquet  rows={len(out)}  "
          f"cell_days_swapped={total_swapped}", flush=True)


if __name__ == "__main__":
    main()
