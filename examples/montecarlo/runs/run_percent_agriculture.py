"""Percent yield change production run, per crop.

This is the representation the agriculture paper reports, converted
exactly the way the sector's plotting code does it
(1_visualize_projections/figures-percentage.R): per region and year,
on the histclim-differenced log effect x,

    pct = exp(x) * exp(0.5 * residvcv) - 1

then each region is recentered on its own 2001-2010 mean of that
converted series, then multiplied by 100. The lognormal bias factor
exp(0.5 * residvcv) uses the residual variance from the crop's csvv
(one scalar per crop; wheat takes wheat_spring's, as the plotting code
does). The conversion happens per region and year before weighting
because the exponential does not commute with the weighted mean; it
applies to the differenced effect, not to each side separately, which
is what the plotting code does.

The result is percentage points of yield relative to the region's
2001-2010 average, a per-hectare rate like the log version, so the
same crop-area weighting applies. Regions with no area for the crop
carry NaN weight in the artifact; on_zero_weight='skip' declares them
and targets without cropland are absent from the output, not zero.
Each crop has its own set of covered units.

The third window ends at 2098: the NEX-GDDP climate input lacks
precipitation for some models after that.

The adm1 run also writes ADM0 (crop-area-weighted ISO mean, exact
because impact regions stay within one country).

Run on the cluster:

    python run_percent_agriculture.py --crop corn --adm adm1 --rcp rcp85
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import cil_regionalization as cilreg
from cil_regionalization import SourceUnitPolicies
from cil_regionalization.netcdf_io import read_netcdf_leaf
from cil_regionalization.stats import pooled_statistics, window_means

MEALY = Path("/project/cil/gcp/outputs/agriculture/impacts-mealy/montecarlo")
CSVV_DIR = Path("/project/cil/home_dirs/egrenier/repos/agriculture/1_code/"
                "3_projections/2_run_projection/projection/csvv")
WEIGHTS = ("/project/cil/home_dirs/scadavidsanchez/repos/"
           "climate-and-damages-aggregations/data/weights")
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
OUT = "/project/cil/gcp/impact_map/agriculture"
BATCHES = [f"batch{i}" for i in range(15)]
SSPS = ["SSP2", "SSP3", "SSP4"]
IAMS = ["high", "low"]
WINDOWS = [(2020, 2039), (2040, 2059), (2080, 2098)]
RECENTER = (2001, 2010)
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]
TREES = {"cassava": "cassava-2025_1pct_winsorization",
         "corn": "corn-2025_1pct_winsorization",
         "rice": "rice-2025_1pct_winsorization",
         "sorghum": "sorghum-2025_1pct_winsorization",
         "soy": "soy-2025_1pct_winsorization",
         "wheat": "wheat_combined-2025_1pct_winsorization"}
BASENAMES = {"cassava": "cassava-110221", "corn": "corn-160221",
             "rice": "rice-160221", "sorghum": "sorghum-160221",
             "soy": "soy-160221", "wheat": "wheat_combined-280823"}
# the plotting code takes wheat's residual variance from wheat_spring
CSVVS = {"cassava": "cassava-110221", "corn": "corn-160221",
         "rice": "rice-160221", "sorghum": "sorghum-160221",
         "soy": "soy-160221", "wheat": "wheat_spring-280823"}
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
SKIP = {"rcp45": "surrogate_GFDL-ESM2G_06", "rcp85": "ACCESS1-0"}


def gcms_for(rcp: str) -> list[str]:
    return [g for g in GCMS if g != SKIP[rcp]]


def read_residvcv(crop: str) -> float:
    """One scalar per crop, from the line after 'residvcv' in the csvv."""
    path = CSVV_DIR / f"{CSVVS[crop]}.csvv"
    lines = path.read_text().splitlines()
    idx = lines.index("residvcv")
    return float(lines[idx + 1])


def read_percent(leafdir: Path, base: str, bias: float) -> pd.DataFrame:
    """One slice's percent yield change per region and year.

    Differenced in logs, converted with the bias factor, recentered on
    the region's 2001-2010 mean, in percentage points; restricted to
    window years after recentering.
    """
    kw = dict(variables=["rebased"], region_dim="region", region_col="hierid",
              region_labels="regions")
    scen = read_netcdf_leaf(leafdir / f"{base}.nc4", **kw)
    hist = read_netcdf_leaf(leafdir / f"{base}-histclim.nc4", **kw)
    df = scen.merge(hist, on=["hierid", "year"], suffixes=("", "_h"),
                    validate="one_to_one")
    x = df["rebased"].astype("float64") - df["rebased_h"].astype("float64")
    df["pct"] = np.exp(x) * bias - 1.0
    lo, hi = RECENTER
    base_years = df["year"].between(lo, hi)
    center = (df[base_years].groupby("hierid")["pct"].mean()
              .rename("center").reset_index())
    df = df.merge(center, on="hierid", validate="many_to_one")
    df["pct"] = (df["pct"] - df["center"]) * 100.0
    keep = df["year"].between(WINDOWS[0][0], WINDOWS[-1][1])
    return df.loc[keep, ["hierid", "year", "pct"]]


def batch_frames(task) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """(target, ISO) window-mean frames for one batch and ssp."""
    crop, batch, ssp, rcp, adm = task
    weights = cilreg.WeightsArtifact.load(f"{WEIGHTS}/ir_{adm}_croparea_{crop}")
    id_fields = list(weights.schema.id_fields)
    area = (weights.frame.dropna(subset=["croparea_raw"])
            .groupby(id_fields, sort=False)["croparea_raw"]
            .sum().reset_index().rename(columns={"croparea_raw": "area"}))
    bias = float(np.exp(0.5 * read_residvcv(crop)))
    tree = MEALY / TREES[crop] / "montecarlo"
    out = []
    for iam in IAMS:
        for gcm in gcms_for(rcp):
            leafdir = tree / batch / rcp / gcm / iam / ssp
            df = read_percent(leafdir, BASENAMES[crop], bias)
            applied = cilreg.apply_weights(
                weights, df, kind="intensive", weight="croparea",
                value_col="pct", data_version="world-combo-201710",
                policies=SourceUnitPolicies(on_zero_weight="skip"),
            ).frame
            member = {"batch": batch, "gcm": gcm, "iam": iam, "ssp": ssp}
            target = applied.assign(**member)
            merged = applied.merge(area, on=id_fields, validate="many_to_one")
            merged["wx"] = merged["pct"] * merged["area"]
            adm0 = (merged.groupby(["ISO", "year"], sort=False)[["wx", "area"]]
                    .sum().reset_index())
            adm0["pct"] = adm0["wx"] / adm0["area"]
            adm0 = adm0[["ISO", "year", "pct"]].assign(**member)
            out.append((
                window_means(target, time_col="year", windows=WINDOWS,
                             value_col="pct"),
                window_means(adm0, time_col="year", windows=WINDOWS,
                             value_col="pct"),
            ))
        print(crop, batch, rcp, ssp, iam, flush=True)
    return out


def weighted_stats(reduced: pd.DataFrame, rcp: str) -> pd.DataFrame:
    smme = pd.read_csv(SMME.format(rcp=rcp), sep="\t")[["model", "weight"]]
    smme["key"] = smme["model"].str.replace("*", "", regex=False).str.lower()
    frame = reduced.assign(
        key=reduced["gcm"].str.replace("surrogate_", "", regex=False).str.lower()
    ).merge(
        smme[["key", "weight"]].rename(columns={"weight": "model_weight"}),
        on="key",
    ).drop(columns="key")
    if len(frame) != len(reduced):
        raise ValueError("a climate model has no SMME weight row")
    return pooled_statistics(
        frame,
        sample_dims=["batch", "gcm"],
        value_col="pct",
        quantiles=QUANTILES,
        weight_col="model_weight",
    )


def write_product(stats: pd.DataFrame, crop: str, adm: str, rcp: str,
                  residvcv: float, extra: dict | None = None) -> None:
    out = Path(OUT) / crop / adm
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{crop}_physical_percent_{rcp}"
    stats.to_parquet(out / f"{stem}.parquet", index=False)
    manifest = {
        "source": str(MEALY / TREES[crop] / "montecarlo"),
        "batches": BATCHES,
        "ssps": SSPS,
        "units": "percentage points of yield, full adaptation minus "
                 "histclim, relative to the region's 2001-2010 average",
        "conversion": "exp(fulladapt - histclim) * exp(0.5*residvcv) - 1, "
                      "per region and year before weighting, recentered on "
                      "2001-2010, times 100; follows the sector's "
                      "figures-percentage.R",
        "residvcv": residvcv,
        "residvcv_source": str(CSVV_DIR / f"{CSVVS[crop]}.csvv"),
        "coverage": "targets with no cropland for this crop are absent, "
                    "not zero; each crop has its own set of covered units",
        "windows": [f"{a}-{b}" for a, b in WINDOWS],
        "window_note": "last window ends at 2098; NEX-GDDP lacks "
                       "precipitation for some models after that",
        "quantiles": QUANTILES,
        "model_weights": SMME.format(rcp=rcp),
    }
    manifest.update(extra or {})
    (out / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}/{stem}.parquet rows={len(stats)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", choices=sorted(TREES), required=True)
    parser.add_argument("--adm", choices=["adm1", "adm2"], required=True)
    parser.add_argument("--rcp", choices=["rcp45", "rcp85"], required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tasks = [(args.crop, b, ssp, args.rcp, args.adm)
             for b in BATCHES for ssp in SSPS]
    if args.workers == 1:
        results = [batch_frames(t) for t in tasks]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(batch_frames, tasks)
    pairs = [p for chunk in results for p in chunk]
    reduced = pd.concat([p[0] for p in pairs], ignore_index=True)

    residvcv = read_residvcv(args.crop)
    stats = weighted_stats(reduced, args.rcp)
    write_product(stats, args.crop, args.adm, args.rcp, residvcv,
                  {"n_members": len(pairs)})

    if args.adm == "adm1":
        adm0 = pd.concat([p[1] for p in pairs], ignore_index=True)
        adm0_stats = weighted_stats(adm0, args.rcp)
        write_product(adm0_stats, args.crop, "adm0", args.rcp, residvcv, {
            "method": "crop-area-weighted mean by ISO before pooling",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
