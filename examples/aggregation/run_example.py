"""Colombia end to end: both variable kinds, both weight directions.

The committed data is a small illustrative extract for demonstrating
the aggregation mechanics. It is not a published dataset, and the
numbers it produces are not results: two Monte Carlo draws out of
hundreds, one climate model out of dozens.

The Monte Carlo extract is committed under examples/aggregation/data
(about 1.6 MB): Colombia's 500 impact regions, its 32 ADM1 departments
(this GADM 2.0 vintage predates the 33rd), and its 1,065 ADM2
municipalities. Two batches, one climate model (GFDL-ESM2G), rcp85,
iam low, SSP3, in the canonical batch/rcp/gcm/iam/ssp tree grammar.
Needs the base package plus the [netcdf] extra.

The weights are fetched from the published Zenodo records by name,
which is what any real use looks like; the application is restricted
to the Colombian regions present in the data, the honest way to apply
a global weight file to a country subset. With --offline the committed
Colombia slices are used instead and no network is touched; useful
where the records are unreachable.

Two variables for the same country, different in kind:

- The physical mortality rate (rebased, deaths per person per year) is
  intensive: an average, aggregated as a population weighted mean with
  the per_destination weight file.
- Monetized damages (total_damages, 2019 USD) are extensive: a total,
  allocated with the per_source weight file, and the allocation must add
  back up, which apply_weights checks.

ADM1 shows both kinds; ADM2 shows the allocation again where it does
visible work, because in Colombia an impact region is a grouping of
municipalities and one region's total spreads across as many as 90 of
them. Statistics pool the two batches; two draws cannot support
quantiles, so those lines show the mechanics, not results.
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
# The impact region geometry vintage the data was built on. apply_weights
# refuses data whose vintage differs from the one the weight file
# records (source_version on the loaded weights shows it).
DATA_VERSION = "world-combo-201710"
BATCHES = ("batch0", "batch1")
LEVELS = "rcp85/GFDL-ESM2G/low/SSP3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the committed Colombia weight slices instead of fetching",
    )
    args = parser.parse_args(argv)

    print("The tree this example reads:")
    for f in sorted(DATA.rglob("*.nc4")):
        print(f"  {f.relative_to(DATA)}")
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
        first_leaf = DATA / "montecarlo" / BATCHES[0] / LEVELS / "mortality_damages_IR_batch.nc4"
        col = read_netcdf_leaf(
            first_leaf, variables=["total_damages"],
            region_dim="region", region_col="hierid",
        )
        restrict = {(h,) for h in col["hierid"].unique()}

    # 1. Physical rate, intensive, per_destination, pop weighted mean.
    # The raw files store region labels as a 'regions' variable on an
    # unlabeled dimension; the caller promotes them to the key column.
    rate_frames = []
    for batch in BATCHES:
        leaf = DATA / "montecarlo" / batch / LEVELS / "Agespec_interaction_response-combined.nc4"
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
    print("Physical rate at ADM1 (deaths per person per year, intensive,")
    print("population weighted mean over each department's impact regions):")
    show = rates[(rates["year"] == 2099) & (rates["batch"] == "batch0")]
    print(show.sort_values("rebased").head(5).to_string(index=False))
    print(f"  ... {show['rebased'].size} departments, both batches computed\n")

    # 2. Monetized damages, extensive, per_source, allocation with mass
    # balance proven per batch inside apply_weights.
    damage_frames = []
    for batch in BATCHES:
        leaf = DATA / "montecarlo" / batch / LEVELS / "mortality_damages_IR_batch.nc4"
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
        damage_frames.append(applied.frame.assign(batch=batch))
    damages = pd.concat(damage_frames, ignore_index=True)
    print("Monetized damages at ADM1 (2019 USD, extensive, allocated shares")
    print("sum to each impact region's total; mass balance checked):")
    show = damages[(damages["year"] == 2099) & (damages["batch"] == "batch0")]
    print(show.sort_values("total_damages", ascending=False).head(5).to_string(index=False))
    total_ir = read_netcdf_leaf(
        DATA / "montecarlo" / "batch0" / LEVELS / "mortality_damages_IR_batch.nc4",
        variables=["total_damages"], region_dim="region", region_col="hierid",
    ).query("year == 2099")["total_damages"].sum()
    print(f"  sum over departments {show['total_damages'].sum():.6e} "
          f"= sum over impact regions {total_ir:.6e}\n")

    # 3. The same damages at ADM2, where the split is visible: in this
    # vintage a Colombian impact region is a grouping of municipalities,
    # so one region's total spreads across up to 90 of them by
    # population share, and the shares sum to one per region.
    data_b0 = read_netcdf_leaf(
        DATA / "montecarlo" / "batch0" / LEVELS / "mortality_damages_IR_batch.nc4",
        variables=["total_damages"], region_dim="region", region_col="hierid",
        kind="extensive",
    )
    adm2 = apply_weights(
        adm2_per_source,
        data_b0,
        kind="extensive",
        weight="pop",
        value_col="total_damages",
        data_version=DATA_VERSION,
        restrict_to_sources=restrict,
    )
    print("Monetized damages at ADM2 (1,065 municipalities from the same")
    print("500 impact regions; same kind, same checks, finer weights):")
    w = adm2_per_source.frame
    w = w[w["hierid"].isin(data_b0["hierid"].unique())]
    widest = w.groupby("hierid").size().idxmax()
    split = w[w["hierid"] == widest][["ISO", "ID_1", "ID_2", "popwt"]]
    print(f"  region {widest} splits across {len(split)} municipalities;")
    print("  its largest population shares:")
    print(split.sort_values("popwt", ascending=False).head(3).to_string(index=False))
    print(f"  shares sum to {split['popwt'].sum():.6f}")
    total_adm2 = adm2.frame.query("year == 2099")["total_damages"].sum()
    print(f"  2099 sum over municipalities {total_adm2:.6e} matches the "
          f"impact region sum\n")

    # 4. Statistics over the Monte Carlo sample. Spatial aggregation
    # happened first, above; the window mean and pooled statistics come
    # last. Two batches are far too few for meaningful quantiles; the
    # numbers below demonstrate the mechanics only.
    stats = summarize_samples(
        damages,
        sample_dims=["batch"],
        time_col="year",
        window=(2080, 2099),
        value_col="total_damages",
        quantiles=[0.05, 0.5, 0.95],
    )
    print("Statistics over the two batches (window mean 2080-2099, then")
    print("pooled; with 2 draws these quantiles are mechanics, not results):")
    one = stats[stats["ID_1"] == stats["ID_1"].iloc[0]]
    print(one.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
