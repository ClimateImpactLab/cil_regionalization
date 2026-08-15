"""Colombia end to end: both variable kinds, both weight directions.

The committed data is a subset of the full Monte Carlo projections,
and the numbers are a demonstration, not results. The sample, under
examples/aggregation/data (about 23 MB): Colombia's 500 impact
regions, its 32 ADM1 departments (this GADM 2.0 version predates the
33rd), and its 1,065 ADM2 municipalities. The monetized damages cover
one batch out of fifteen, across all 33 climate models (rcp85, iam
low, SSP3), in the standard batch/rcp/gcm/iam/ssp tree grammar. The
physical rates cover two batches of one model (GFDL-ESM2G), because
the tree holding the rates is still being regenerated.

By default the weights are fetched from the published Zenodo records;
--offline uses the committed Colombia slices instead. Both modes
produce identical output. Needs the base package plus the [netcdf]
extra, and network access for the default mode.

Two variables for the same country, different in kind:

- The physical mortality rate (rebased, deaths per person per year) is
  intensive: an average, aggregated as a population weighted mean with
  the per_destination weight file.
- Monetized damages (total_damages, 2019 USD) are extensive: a total,
  allocated with the per_source weight file, and the allocation must add
  back up, which apply_weights checks.

ADM1 shows both kinds. ADM2 shows the allocation again at finer
resolution: in Colombia an impact region is a grouping of
municipalities, and one region's total spreads across as many as 90 of
them. Statistics pool the 33 climate models, each aggregated first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from cil_regionalization.apply import WeightsArtifact, apply_weights
from cil_regionalization.netcdf_io import read_netcdf_leaf
from cil_regionalization.stats import summarize_samples

DATA = _HERE / "data"
# The impact region geometry version the data was built on. apply_weights
# refuses data whose version differs from the one the weight file
# records (source_version on the loaded weights shows it).
DATA_VERSION = "world-combo-201710"
RATE_BATCHES = ("batch0", "batch1")
RATE_LEVELS = "rcp85/GFDL-ESM2G/low/SSP3"
DAMAGES_BATCH = DATA / "montecarlo" / "batch0" / "rcp85"


def _names() -> "pd.DataFrame":
    """Department names from the committed plotting layer, display only."""
    return pd.read_parquet(
        DATA / "col_adm1_plot.parquet", columns=["ISO", "ID_1", "NAME_1"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the committed Colombia weight slices instead of fetching",
    )
    args = parser.parse_args(argv)

    gcms = sorted(p.name for p in DAMAGES_BATCH.iterdir() if p.is_dir())
    print("The tree this example reads: one damages leaf per climate model,")
    print(f"  montecarlo/batch0/rcp85/<gcm>/low/SSP3/ for {len(gcms)} models,")
    print("plus the physical rate leaves,")
    for b in RATE_BATCHES:
        print(f"  montecarlo/{b}/{RATE_LEVELS}/Agespec_interaction_response-combined.nc4")
    print()

    if args.offline:
        per_destination = WeightsArtifact.load(DATA / "weights" / "adm1_per_destination")
        per_source = WeightsArtifact.load(DATA / "weights" / "adm1_per_source")
        adm2_per_source = WeightsArtifact.load(DATA / "weights" / "adm2_per_source")
        restrict = None
    else:
        from cil_regionalization import fetch_weights

        per_destination = fetch_weights("gadm20-adm1-per-destination")
        per_source = fetch_weights("gadm20-adm1-per-source")
        adm2_per_source = fetch_weights("gadm20-adm2-per-source")
        # The fetched weights are global; the data is one country. The
        # restriction declares that subset explicitly, and the coverage
        # checks stay strict within it.
        first_leaf = DAMAGES_BATCH / gcms[0] / "low" / "SSP3" / "mortality_damages_IR_batch.nc4"
        col = read_netcdf_leaf(
            first_leaf, variables=["total_damages"],
            region_dim="region", region_col="hierid",
        )
        restrict = {(h,) for h in col["hierid"].unique()}

    # 1. Physical rate, intensive, per_destination, pop weighted mean.
    # The raw files store region labels as a 'regions' variable on an
    # unlabeled dimension; the caller promotes them to the key column.
    rate_frames = []
    for batch in RATE_BATCHES:
        leaf = DATA / "montecarlo" / batch / RATE_LEVELS / "Agespec_interaction_response-combined.nc4"
        raw = read_netcdf_leaf(
            leaf,
            variables=["rebased", "regions"],
            region_dim="region",
            region_col="region_index",
            kind="intensive",
        )
        data = raw.rename(columns={"regions": "hierid"})[["hierid", "year", "rebased"]]
        applied = apply_weights(
            per_destination,
            data,
            kind="intensive",
            weight="pop",
            value_col="rebased",
            data_version=DATA_VERSION,
            restrict_to_sources=restrict,
        )
        rate_frames.append(applied.frame.assign(batch=batch))
    rates = pd.concat(rate_frames, ignore_index=True)
    names = _names()
    print("Physical rate at ADM1 (deaths per person per year, intensive,")
    print("population weighted mean over each department's impact regions):")
    show = rates[(rates["year"] == 2099) & (rates["batch"] == "batch0")]
    show = show.merge(names, on=["ISO", "ID_1"])
    print(show.sort_values("rebased").head(5)[["NAME_1", "year", "rebased"]].to_string(index=False))
    print(f"  ... {show['rebased'].size} departments, both batches computed\n")

    # 2. Monetized damages, extensive, per_source: aggregate every
    # climate model's draw, then use one model for the single-draw
    # printouts below.
    damage_frames = []
    for gcm in gcms:
        leaf = DAMAGES_BATCH / gcm / "low" / "SSP3" / "mortality_damages_IR_batch.nc4"
        data = read_netcdf_leaf(
            leaf,
            variables=["total_damages"],
            region_dim="region",
            region_col="hierid",
            kind="extensive",
        )
        applied = apply_weights(
            per_source,
            data,
            kind="extensive",
            weight="pop",
            value_col="total_damages",
            data_version=DATA_VERSION,
            restrict_to_sources=restrict,
        )
        damage_frames.append(applied.frame.assign(gcm=gcm))
    damages = pd.concat(damage_frames, ignore_index=True)
    one_draw = damages[damages["gcm"] == "GFDL-ESM2G"]
    print("Monetized damages at ADM1 (2019 USD, extensive, allocated shares")
    print("sum to each impact region's total; mass balance checked):")
    show = one_draw[one_draw["year"] == 2099].merge(names, on=["ISO", "ID_1"])
    print(show.sort_values("total_damages", ascending=False).head(5)[["NAME_1", "year", "total_damages", "gcm"]].to_string(index=False))
    total_ir = read_netcdf_leaf(
        DAMAGES_BATCH / "GFDL-ESM2G" / "low" / "SSP3" / "mortality_damages_IR_batch.nc4",
        variables=["total_damages"], region_dim="region", region_col="hierid",
    ).query("year == 2099")["total_damages"].sum()
    print(f"  sum over departments {show['total_damages'].sum():.6e} "
          f"= sum over impact regions {total_ir:.6e}\n")

    # 3. The same damages at ADM2, where the split is visible: in this
    # geometry version a Colombian impact region is a grouping of
    # municipalities, so one region's total spreads across up to 90 of
    # them by population share, and the shares sum to one per region.
    data_one = read_netcdf_leaf(
        DAMAGES_BATCH / "GFDL-ESM2G" / "low" / "SSP3" / "mortality_damages_IR_batch.nc4",
        variables=["total_damages"], region_dim="region", region_col="hierid",
        kind="extensive",
    )
    adm2 = apply_weights(
        adm2_per_source,
        data_one,
        kind="extensive",
        weight="pop",
        value_col="total_damages",
        data_version=DATA_VERSION,
        restrict_to_sources=restrict,
    )
    print("Monetized damages at ADM2 (1,065 municipalities from the same")
    print("500 impact regions; same kind, same checks, finer weights):")
    w = adm2_per_source.frame
    w = w[w["hierid"].isin(data_one["hierid"].unique())]
    widest = w.groupby("hierid").size().idxmax()
    split = w[w["hierid"] == widest][["ISO", "ID_1", "ID_2", "popwt"]].merge(
        names, on=["ISO", "ID_1"]
    )[["NAME_1", "ID_2", "popwt"]]
    print(f"  region {widest} splits across {len(split)} municipalities;")
    print("  its largest population shares:")
    print(split.sort_values("popwt", ascending=False).head(3).to_string(index=False))
    print(f"  shares sum to {split['popwt'].sum():.6f}")
    total_adm2 = adm2.frame.query("year == 2099")["total_damages"].sum()
    print(f"  2099 sum over municipalities {total_adm2:.6e} matches the "
          f"impact region sum\n")

    # 4. Statistics over the Monte Carlo sample. Spatial aggregation
    # happened first, above; the window mean and pooled statistics come
    # last. The 33 climate models are the draws.
    stats = summarize_samples(
        damages,
        sample_dims=["gcm"],
        time_col="year",
        window=(2080, 2099),
        value_col="total_damages",
        quantiles=[0.05, 0.5, 0.95],
    )
    print("Statistics over the 33 climate models (window mean 2080-2099,")
    print("then pooled):")
    stats = stats.merge(names, on=["ISO", "ID_1"])
    one = stats[stats["NAME_1"] == "Antioquia"]
    print(one[["NAME_1", "statistic", "total_damages"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
