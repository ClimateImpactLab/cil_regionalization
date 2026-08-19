"""Valued mortality production run over the new_format zarrs.

The valued object lives in per-batch zarr stores, not per-leaf netcdf,
so it does not go through `cilreg pipeline`. This runner does the same
work with the public API: per batch, gcm, iam, and ssp it differences
the monetized histclim variable, applies the weights, reduces to
window means, and pools one statistics frame with the SMME model
weights. The stores hold SSP2, SSP3, and SSP4 (the intersection with
the physical tree's coverage), in 2019 USD, valued from the 2021
physical tree: the implied value per death is identical between the
fulladapt and histclim variables to float precision.

The adm1 run also writes ADM0: the reduced frame regrouped by ISO
before pooling, exact for an extensive variable.

Run on the cluster:

    python run_valued_mortality.py --adm adm1 --rcp rcp85 --workers 8
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

ZARRS = Path("/project/cil/gcp/outputs/mortality/impacts-darwin-montecarlo-damages/new_format")
WEIGHTS = "/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/data/weights/ir_{adm}_full"
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
OUT = "/project/cil/gcp/impact_map/mortality"
BATCHES = [f"batch{i}" for i in range(15)]
SSPS = ["SSP2", "SSP3", "SSP4"]
VALUATION = "vsl"
SCALING = "epa_scaled"
WINDOWS = [(2020, 2039), (2040, 2059), (2080, 2099)]
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]
IAMS = {"IIASA GDP": "low", "OECD Env-Growth": "high"}


def batch_frames(task) -> list[pd.DataFrame]:
    """Window-mean frames for every (gcm, iam, ssp) slice of one batch."""
    batch, rcp, weights_dir = task
    weights = cilreg.WeightsArtifact.load(weights_dir)
    ds = xr.open_zarr(str(ZARRS / f"mortality_damages_{batch}.zarr"))
    regions = [str(r) for r in ds["region"].values]
    years = [int(y) for y in ds["year"].values]
    out = []
    for ssp in SSPS:
        for model, iam in IAMS.items():
            block = ds.sel(rcp=rcp, model=model, ssp=ssp,
                           valuation=VALUATION, scaling=SCALING)
            effect = block["monetized_deaths"] - block["monetized_histclim_deaths"]
            for gcm in [str(g) for g in ds["gcm"].values]:
                arr = (effect.sel(gcm=gcm).squeeze()
                       .transpose("year", "region").values)
                frame = pd.DataFrame(
                    {"hierid": pd.Series(regions).repeat(len(years)).values,
                     "year": years * len(regions),
                     "value": arr.T.reshape(-1)}
                )
                applied = cilreg.apply_weights(
                    weights, frame, kind="extensive", weight="pop",
                    data_version="world-combo-201710",
                ).frame
                applied["batch"] = batch
                applied["gcm"] = gcm
                applied["iam"] = iam
                applied["ssp"] = ssp
                reduced = window_means(
                    applied, time_col="year", windows=WINDOWS, value_col="value"
                )
                out.append(reduced)
        print(batch, rcp, ssp, flush=True)
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
                  adm: str, rcp: str, n_frames: int) -> None:
    out = Path(OUT) / adm
    out.mkdir(parents=True, exist_ok=True)
    stem = f"mortality_valued_{rcp}"
    stats.to_parquet(out / f"{stem}.parquet", index=False)
    manifest = {
        "source": str(ZARRS),
        "batches": BATCHES,
        "ssps": SSPS,
        "valuation": VALUATION,
        "scaling": SCALING,
        "units": "2019 USD",
        "windows": [f"{a}-{b}" for a, b in WINDOWS],
        "quantiles": QUANTILES,
        "model_weights": SMME.format(rcp=rcp),
        "n_frames": n_frames,
    }
    (out / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    if reduced is not None:
        reduced.to_parquet(out / f"{stem}.reduced.parquet", index=False)
    print(f"wrote {out}/{stem}.parquet rows={len(stats)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adm", choices=["adm1", "adm2"], required=True)
    parser.add_argument("--rcp", choices=["rcp45", "rcp85"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    weights_dir = WEIGHTS.format(adm=args.adm)
    tasks = [(b, args.rcp, weights_dir) for b in BATCHES]
    if args.workers == 1:
        results = [batch_frames(t) for t in tasks]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(batch_frames, tasks)
    frames = [f for chunk in results for f in chunk]
    reduced = pd.concat(frames, ignore_index=True)

    stats = weighted_stats(reduced, args.rcp)
    write_product(stats, reduced if args.adm == "adm1" else None,
                  args.adm, args.rcp, len(frames))

    if args.adm == "adm1":
        # ADM0: regroup the reduced frame by ISO before pooling; exact
        # for an extensive variable
        keys = ["ISO", "window", "batch", "gcm", "iam", "ssp"]
        adm0 = reduced.groupby(keys, sort=False)["value"].sum().reset_index()
        adm0_stats = weighted_stats(adm0, args.rcp)
        out = Path(OUT) / "adm0"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"mortality_valued_{args.rcp}"
        adm0_stats.to_parquet(out / f"{stem}.parquet", index=False)
        (out / f"{stem}.manifest.json").write_text(json.dumps(
            {"derived_from": f"{OUT}/adm1/{stem}.reduced.parquet",
             "method": "ISO sum of the adm1 reduced frame before pooling",
             "units": "2019 USD"}, indent=2))
        print(f"wrote {out}/{stem}.parquet rows={len(adm0_stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
