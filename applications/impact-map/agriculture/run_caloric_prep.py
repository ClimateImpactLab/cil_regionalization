"""Caloric yield draws from the incadapt tree, one scenario per run.

Prepares the parts of the requested calorie product that do not depend
on the plan's two open questions (what 3.0C selects, and what the
population weighting applies to). Per Monte Carlo draw it converts the
precomputed incadapt yields to calories, differences the region's own
2001-2010 baseline, and reduces to the stack's windows: the three
ranges the other agriculture products carry plus the final projection
year alone. Everything downstream (the warming-level cut, the
statistics, and any spatial aggregation) starts from the draws this
writes, so both readings of either question can be served without
re-reading the tree.

``--ssp`` and ``--iam`` pick the scenario; the tree carries SSP2, SSP3
and SSP4 under both IAMs, and one job per combination writes its own
output directory, so the analysis can choose its scenario later. The
defaults reproduce the first run (SSP2 with IIASA).

Inputs, all already on disk:

  - ``cc_yields_incadapt``: per-leaf incadapt yields from the agval
    precomputation, 6 crops x 24,378 impact regions x 2000-2098, in
    tonnes per hectare (the valuation multiplies them by calories per
    tonne directly).
  - The Cassidy et al. 2013 calories-per-tonne table, copied from
    ``agval/loading.py`` so this script stands alone.
  - The SAGE planted areas the valuation uses, in hectares.

Per draw and region the output carries, for each crop and for the
six-crop total: the change in calories per hectare (a rate) and the
change in total calories (extensive), each as the window mean of the
yearly difference from the region's own 2001-2010 mean. Both forms are
kept because the plan's population-weighting question decides which
one aggregates; the crop dimension is kept because the per-crop
product is a requested secondary output.

The last range and the final year end at 2098, not 2099: the NEX-GDDP
climate input lacks precipitation for some models after that, so the
yield projections stop there. The plan says 2080-2099; the last year
does not exist for agriculture and the window label says what was
used.

Output: one parquet of draws per (batch, rcp) under OUT, plus a
manifest. Roughly 680k rows per leaf and 22M per file; the statistics
step reads these selectively once the plan's questions are answered.

Run on the cluster:

    python run_caloric_prep.py --workers 15
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TREE = Path("/project/cil/gcp/outputs/agriculture/agval/cc_yields_incadapt")
SAGE = ("/project/cil/battuta_shares/gcp/estimation/agriculture/Data/"
        "2_intermediate/cropped_area/hierid/sage_planted_area_global.csv")
OUT_ROOT = Path("/project/cil/gcp/impact_map/agriculture/calories")
# set by main() from --ssp and --iam; module-level so pool workers see them
OUT = OUT_ROOT / "draws_incadapt_ssp2_low"
SSP = "SSP2"
IAM = "low"  # IIASA
BATCHES = [f"batch{i}" for i in range(15)]
BASELINE = (2001, 2010)
# the stack's windows, so the caloric products line up with the other
# agriculture products: three ranges plus the final projection year.
# The analysis plan's "2050" is the 2040-2059 window and its "2090" is
# 2080-2098; the manifest records the mapping.
WINDOWS = {"2020-2039": (2020, 2039), "2040-2059": (2040, 2059),
           "2080-2098": (2080, 2098), "2098": (2098, 2098)}
# calories per tonne, Cassidy et al. 2013 (ERL), copied from
# agval/loading.py so the conversion is byte-identical to the valuation
CALS = {"soy": 3596499.11, "maize": 3580802.60, "cassava": 1058035.85,
        "wheat": 3284000.00, "rice": 2800000.00, "sorghum": 3430000.40}
GCMS = ["ACCESS1-0", "BNU-ESM", "CCSM4", "CESM1-BGC", "CNRM-CM5",
        "CSIRO-Mk3-6-0", "CanESM2", "GFDL-CM3", "GFDL-ESM2G", "GFDL-ESM2M",
        "IPSL-CM5A-LR", "IPSL-CM5A-MR", "MIROC-ESM", "MIROC-ESM-CHEM",
        "MIROC5", "MPI-ESM-LR", "MPI-ESM-MR", "MRI-CGCM3", "NorESM1-M",
        "bcc-csm1-1", "inmcm4",
        "surrogate_CanESM2_89", "surrogate_CanESM2_94", "surrogate_CanESM2_99",
        "surrogate_GFDL-CM3_89", "surrogate_GFDL-CM3_94", "surrogate_GFDL-CM3_99",
        "surrogate_GFDL-ESM2G_01", "surrogate_GFDL-ESM2G_06",
        "surrogate_GFDL-ESM2G_11", "surrogate_MRI-CGCM3_01",
        "surrogate_MRI-CGCM3_06", "surrogate_MRI-CGCM3_11"]
# the same absences as the yield trees these were computed from
SKIP = {"rcp45": "surrogate_GFDL-ESM2G_06", "rcp85": "ACCESS1-0"}


def gcms_for(rcp: str) -> list[str]:
    return [g for g in GCMS if g != SKIP[rcp]]


def load_area(crops: list[str], regions: list[str]) -> xr.DataArray:
    """SAGE planted hectares as (crop, region), zero where uncropped.

    The same file and naming the valuation loads
    (agval load_planted); soybean renames to soy.
    """
    df = (pd.read_csv(SAGE)
          .replace("soybean", "soy")
          .rename(columns={"hectares": "harvested_hectares"}))
    df = df[df["crop"].isin(crops)]
    wide = (df.pivot_table(index="crop", columns="hierid",
                           values="harvested_hectares", aggfunc="sum")
            .reindex(index=crops, columns=regions).fillna(0.0))
    return xr.DataArray(wide.values, dims=("crop", "region"),
                        coords={"crop": crops, "region": regions})


def leaf_frames(leafdir: Path, area: xr.DataArray | None) -> tuple[pd.DataFrame, xr.DataArray]:
    """One draw's caloric changes per region, crop and window.

    Yields become calories per hectare through the Cassidy table and
    total calories through the SAGE areas; each series is differenced
    against the region's own 2001-2010 mean, and the yearly differences
    are averaged over each window. The 'all' crop is the six-crop sum
    of total calories, with the rate form dividing by the six-crop
    planted area (NaN where a region plants none of the six).
    """
    with xr.open_dataset(leafdir / "cc_yield.nc4") as ds:
        da = ds["yield"].astype("float64").load()
    crops = [str(c) for c in da["crop"].values]
    cals = xr.DataArray([CALS[c] for c in crops], dims=("crop",),
                        coords={"crop": crops})
    if area is None:
        area = load_area(crops, [str(r) for r in da["region"].values])

    per_ha = da * cals                     # calories per hectare
    total = per_ha * area                  # calories
    lo, hi = BASELINE
    pieces = []
    for name, quantity in (("cal_per_ha", per_ha), ("cal_total", total)):
        base = quantity.sel(year=slice(lo, hi)).mean("year")
        change = quantity - base
        # six-crop aggregates: totals sum; the rate is the summed total
        # over the summed area, undefined without cropland
        tot_base = total.sel(year=slice(lo, hi)).mean("year")
        for label, (a, b) in WINDOWS.items():
            w = change.sel(year=slice(a, b)).mean("year")
            if name == "cal_total":
                allc = w.sum("crop")
            else:
                area_sum = area.sum("crop")
                allc = ((total - tot_base).sel(year=slice(a, b)).mean("year")
                        .sum("crop") / area_sum.where(area_sum > 0))
            frame = w.to_dataframe(name).reset_index()
            allf = allc.to_dataframe(name).reset_index().assign(crop="all")
            piece = pd.concat([frame, allf], ignore_index=True)
            piece["window"] = label
            piece = piece.set_index(["region", "crop", "window"])[name]
            pieces.append(piece)
    n = len(WINDOWS)
    merged = pd.DataFrame({
        "cal_per_ha_change": pd.concat(pieces[:n]),
        "cal_total_change": pd.concat(pieces[n:]),
    }).reset_index().rename(columns={"region": "hierid"})
    return merged, area


def batch_frames(task) -> str:
    """All draws of one (batch, rcp), written as one parquet."""
    batch, rcp = task
    area = None
    frames = []
    for gcm in gcms_for(rcp):
        leafdir = TREE / batch / rcp / gcm / IAM / SSP
        frame, area = leaf_frames(leafdir, area)
        frame["batch"] = batch
        frame["rcp"] = rcp
        frame["gcm"] = gcm
        for col in ("hierid", "crop", "window"):
            frame[col] = frame[col].astype("category")
        frames.append(frame)
        print(batch, rcp, gcm, flush=True)
    out = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"draws_{batch}_{rcp}.parquet"
    out.to_parquet(path, index=False)
    return str(path)


def main() -> int:
    global SSP, IAM, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssp", choices=["SSP2", "SSP3", "SSP4"],
                        default="SSP2")
    parser.add_argument("--iam", choices=["low", "high"], default="low")
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    SSP, IAM = args.ssp, args.iam
    OUT = OUT_ROOT / f"draws_incadapt_{SSP.lower()}_{IAM}"

    tasks = [(b, rcp) for b in BATCHES for rcp in ("rcp45", "rcp85")]
    missing = [TREE / b / r / g / IAM / SSP / "cc_yield.nc4"
               for b, r in tasks for g in gcms_for(r)
               if not (TREE / b / r / g / IAM / SSP / "cc_yield.nc4").exists()]
    if missing:
        raise ValueError(f"{len(missing)} declared leaves missing, first: "
                         f"{missing[:3]}")

    if args.workers == 1:
        paths = [batch_frames(t) for t in tasks]
    else:
        with mp.Pool(processes=args.workers) as pool:
            paths = pool.map(batch_frames, tasks)

    manifest = {
        "source": str(TREE),
        "ssp": SSP,
        "iam": f"{IAM} ({'IIASA' if IAM == 'low' else 'OECD Env-Growth'})",
        "adaptation": "incadapt",
        "batches": BATCHES,
        "gcms": {r: gcms_for(r) for r in ("rcp45", "rcp85")},
        "calories_per_tonne": CALS,
        "calories_source": "Cassidy et al. 2013, copied from agval/loading.py",
        "areas": SAGE,
        "baseline": f"{BASELINE[0]}-{BASELINE[1]} mean of the same series, "
                    "per region and crop",
        "windows": {k: f"{a}-{b}" for k, (a, b) in WINDOWS.items()},
        "window_note": "the last range and the final year end at 2098; "
                       "NEX-GDDP lacks precipitation for some models "
                       "after that, so the yield projections stop there",
        "plan_mapping": "the analysis plan's 2050 is the 2040-2059 window "
                        "and its 2090 is 2080-2098",
        "units": "cal_per_ha_change in calories per hectare (a rate); "
                 "cal_total_change in calories (extensive, yields times "
                 "SAGE planted areas); crop 'all' is the six-crop sum, "
                 "with the rate form dividing by the six-crop area and "
                 "NaN where a region plants none of the six",
        "pending": "warming-level selection, statistics, and spatial "
                   "aggregation are downstream of these draws and wait "
                   "on the plan's definition of 3.0C and of the "
                   "population weighting",
        "files": paths,
    }
    (OUT / "draws.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(paths)} parquets to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
