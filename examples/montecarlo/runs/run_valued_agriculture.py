"""Valued agriculture production run over the main_spec welfare tree.

The valued object is wc_no_reallocation from
agval/main_spec/montecarlo: welfare change in 2019 USD from the
constant-elasticity market model, six crops combined, trading-partner
markets, the delta-only object the agriculture paper published. The
counterfactual is already inside the valuation (climate-change yields
against the same batch's histclim yields), so no sibling subtraction
happens here.

The leaves embed batch, rcp, gcm, model and ssp as singleton
coordinates, which collide with the pipeline's tree levels; that is
why this is a runner and not a config. Five uninhabited territories
(ATA, BVT, CA-, HMD, SGS) are NaN in every year and are dropped with
the count recorded in the manifest. Guatemala is NaN in 2017 only,
which no window touches (windows start at 2020).

The third window ends at 2098: the NEX-GDDP climate input lacks
precipitation for some models after that, so the projections stop
there.

The adm1 run also writes ADM0: the reduced frame regrouped by ISO
before pooling, exact for an extensive variable.

Run on the cluster:

    python run_valued_agriculture.py --adm adm1 --rcp rcp85 --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import xarray as xr

import cil_regionalization as cilreg
from cil_regionalization.stats import pooled_statistics, window_means

TREE = Path("/project/cil/gcp/outputs/agriculture/agval/main_spec/montecarlo")
WEIGHTS = "/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/data/weights/ir_{adm}_full"
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
OUT = "/project/cil/gcp/impact_map/agriculture/all_crops"
BATCHES = [f"batch{i}" for i in range(15)]
SSPS = ["SSP2", "SSP3", "SSP4"]
IAMS = ["high", "low"]
WINDOWS = [(2020, 2039), (2040, 2059), (2080, 2098)]
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]
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


def read_leaf(leafdir: Path) -> pd.DataFrame:
    """One leaf's welfare change as hierid, year, value; NaN rows dropped.

    The always-NaN territories vanish here; Guatemala's 2017 rows drop
    too, which no window uses.
    """
    with xr.open_dataset(leafdir / "disaggregated_damages.nc4") as ds:
        wc = ds["wc_no_reallocation"].squeeze(drop=True)
        frame = pd.DataFrame({
            "hierid": [str(r) for r in wc["region"].values],
        }).join(pd.DataFrame(
            wc.transpose("region", "year").values,
            columns=[int(y) for y in wc["year"].values],
        ))
    long = frame.melt(id_vars="hierid", var_name="year", value_name="value")
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    long["value"] = long["value"].astype("float64")
    return long


def batch_frames(task) -> list[pd.DataFrame]:
    """Window-mean frames for every (gcm, iam) slice of one batch and ssp."""
    batch, ssp, rcp, weights_dir = task
    weights = cilreg.WeightsArtifact.load(weights_dir)
    out = []
    for iam in IAMS:
        for gcm in gcms_for(rcp):
            leafdir = TREE / batch / rcp / gcm / iam / ssp
            df = read_leaf(leafdir)
            applied = cilreg.apply_weights(
                weights, df, kind="extensive", weight="pop",
                value_col="value", data_version="world-combo-201710",
                restrict_to_sources={(h,) for h in df["hierid"].unique()},
            ).frame
            applied["batch"] = batch
            applied["gcm"] = gcm
            applied["iam"] = iam
            applied["ssp"] = ssp
            out.append(window_means(
                applied, time_col="year", windows=WINDOWS, value_col="value"
            ))
        print(batch, rcp, ssp, iam, flush=True)
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
        value_col="value",
        quantiles=QUANTILES,
        weight_col="model_weight",
    )


def write_product(stats: pd.DataFrame, reduced: pd.DataFrame | None,
                  adm: str, rcp: str, extra: dict | None = None) -> None:
    out = Path(OUT) / adm
    out.mkdir(parents=True, exist_ok=True)
    stem = f"agriculture_valued_{rcp}"
    stats.to_parquet(out / f"{stem}.parquet", index=False)
    manifest = {
        "source": str(TREE),
        "variable": "wc_no_reallocation",
        "batches": BATCHES,
        "ssps": SSPS,
        "units": "2019 USD, welfare change (consumer plus producer surplus)",
        "counterfactual": "already inside the valuation: climate-change "
                          "yields against the same batch's histclim yields; "
                          "the delta-only object the agriculture paper "
                          "published, with no histclim-variability "
                          "correction",
        "crops": ["rice", "sorghum", "cassava", "soy", "maize", "wheat"],
        "dropped_regions": ["ATA", "BVT", "CA-", "HMD", "SGS"],
        "dropped_note": "NaN in every year in the source; Guatemala is NaN "
                        "in 2017 only, outside every window",
        "windows": [f"{a}-{b}" for a, b in WINDOWS],
        "window_note": "last window ends at 2098; NEX-GDDP lacks "
                       "precipitation for some models after that",
        "quantiles": QUANTILES,
        "model_weights": SMME.format(rcp=rcp),
    }
    manifest.update(extra or {})
    (out / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    if reduced is not None:
        reduced.to_parquet(out / f"{stem}.reduced.parquet", index=False)
    print(f"wrote {out}/{stem}.parquet rows={len(stats)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adm", choices=["adm1", "adm2"], required=True)
    parser.add_argument("--rcp", choices=["rcp45", "rcp85"], required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    weights_dir = WEIGHTS.format(adm=args.adm)
    tasks = [(b, ssp, args.rcp, weights_dir) for b in BATCHES for ssp in SSPS]
    if args.workers == 1:
        results = [batch_frames(t) for t in tasks]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(batch_frames, tasks)
    frames = [f for chunk in results for f in chunk]
    reduced = pd.concat(frames, ignore_index=True)

    stats = weighted_stats(reduced, args.rcp)
    write_product(stats, reduced if args.adm == "adm1" else None,
                  args.adm, args.rcp, {"n_members": len(frames)})

    if args.adm == "adm1":
        keys = ["ISO", "window", "batch", "gcm", "iam", "ssp"]
        adm0 = reduced.groupby(keys, sort=False)["value"].sum().reset_index()
        adm0_stats = weighted_stats(adm0, args.rcp)
        write_product(adm0_stats, None, "adm0", args.rcp, {
            "method": "ISO sum of the adm1 reduced frame before pooling",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
