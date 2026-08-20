"""Physical labor production run: minutes per worker per day.

The physical effect is a rate, so it aggregates as a population-weighted
mean through the ratio route: minutes times population and population
each aggregate extensively, and the division happens at the target. The
population denominator comes from the merged SSP baselines, held
constant within each five-year span, the same series the tree's own
wage factor embeds (recovered to 5e-08 across every ssp and iam). No
leaf carries it, and the pipeline cannot merge external data, which is
why this is a runner and not a config.

The effect per region and year is ``rebased`` from the fulladapt file
minus ``rebased`` from its ``-histclim`` sibling. Both files exist on
every leaf; the earlier full sweep of this tree found no gaps.

The adm1 run also writes ADM0: numerator and denominator sum by ISO
per year before the division, exact because impact regions do not
cross national borders and both sides use the same weight file.

Run on the cluster:

    python run_physical_labor.py --adm adm1 --rcp rcp85 --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd

import cil_regionalization as cilreg
from cil_regionalization.netcdf_io import read_netcdf_leaf
from cil_regionalization.stats import pooled_statistics, window_means

TREE = Path("/project/cil/gcp/outputs/labor/impacts-woodwork/montecarlo"
            "/uninteracted_main_model_27_37_39")
STEM = "uninteracted_main_model"
POP = "/project/cil/gcp/social/baselines/population/merged/population-merged.{ssp}.csv"
WEIGHTS = "/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/data/weights/ir_{adm}_full"
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
OUT = "/project/cil/gcp/impact_map/labor"
BATCHES = [f"batch{i}" for i in range(15)]
SSPS = ["SSP2", "SSP3", "SSP4"]
IAMS = ["high", "low"]
WINDOWS = [(2020, 2039), (2040, 2059), (2080, 2099)]
YEARS = range(2020, 2100)
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


def gcms_for(rcp: str) -> list[str]:
    # rcp45 was run without surrogate_GFDL-ESM2G_06; everything else is
    # present under both concentration pathways.
    return [g for g in GCMS if not (rcp == "rcp45" and g == "surrogate_GFDL-ESM2G_06")]


def load_population(ssp: str) -> pd.DataFrame:
    """Per-region population for 2020-2099, five-year values held flat.

    The merged baselines carry one value per region every fifth year;
    the projection system holds it constant within the span, and the
    tree's own wage factor confirms that convention exactly.
    """
    path = POP.format(ssp=ssp)
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
        for row in csv.reader(f):
            rows.append((int(row[0]), row[1], float(row[2])))
    base = pd.DataFrame(rows, columns=["span", "hierid", "pop"])
    if base.duplicated(subset=["span", "hierid"]).any():
        raise ValueError(f"{path} has duplicate (year, region) rows")
    years = pd.DataFrame({"year": list(YEARS)})
    years["span"] = years["year"] - years["year"] % 5
    out = years.merge(base, on="span").drop(columns="span")
    return out


def read_effect(leafdir: Path) -> pd.DataFrame:
    """Fulladapt minus histclim rebased, restricted to window years."""
    frames = {}
    for tag, name in (("scen", f"{STEM}.nc4"), ("hist", f"{STEM}-histclim.nc4")):
        frame = read_netcdf_leaf(
            leafdir / name,
            variables=["rebased"],
            region_dim="region",
            region_col="hierid",
            region_labels="regions",
        )
        frames[tag] = frame.rename(columns={"rebased": tag})
    df = frames["scen"].merge(frames["hist"], on=["hierid", "year"],
                              validate="one_to_one")
    df = df[df["year"].isin(YEARS)].copy()
    df["minutes"] = df["scen"].astype("float64") - df["hist"].astype("float64")
    return df[["hierid", "year", "minutes"]]


def batch_frames(task) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """(target, ISO) window-mean rate frames for one batch and ssp.

    One task per (batch, ssp): 45 tasks, so worker counts past 15
    still shorten the run. The population denominator depends only on
    the ssp, so it aggregates once per task; each (gcm, iam) slice
    then reads two files, aggregates the numerator, and divides.
    """
    batch, ssp, rcp, weights_dir = task
    weights = cilreg.WeightsArtifact.load(weights_dir)
    id_fields = list(weights.schema.id_fields)
    pop = load_population(ssp)
    # 110 of the tree's 24378 regions (small territories and
    # Antarctica) have no population baseline; a population-weighted
    # mean is undefined there, so the product covers the regions the
    # baselines cover and nothing else.
    covered = {(h,) for h in pop["hierid"].unique()}
    den = cilreg.apply_weights(
        weights,
        pop,
        kind="extensive",
        weight="pop",
        value_col="pop",
        data_version="world-combo-201710",
        restrict_to_sources=covered,
    ).frame
    # A target with zero population has no defined rate; drop targets
    # whose denominator is zero in any year (uninhabited units)
    alive = den.groupby(id_fields, sort=False)["pop"].min()
    alive = alive[alive > 0].reset_index()[id_fields]
    den = den.merge(alive, on=id_fields)
    den_iso = (den.groupby(["ISO", "year"], sort=False)["pop"]
               .sum().reset_index())
    out = []
    for iam in IAMS:
        for gcm in gcms_for(rcp):
            leafdir = TREE / batch / rcp / gcm / iam / ssp
            df = read_effect(leafdir)
            df = df.merge(pop, on=["hierid", "year"])
            df["minutes_pop"] = df["minutes"] * df["pop"]
            num = cilreg.apply_weights(
                weights,
                df[["hierid", "year", "minutes_pop"]],
                kind="extensive",
                weight="pop",
                value_col="minutes_pop",
                data_version="world-combo-201710",
                restrict_to_sources=covered,
            ).frame
            target = num.merge(den, on=id_fields + ["year"],
                               validate="one_to_one")
            if target["pop"].eq(0).any() or target["pop"].isna().any():
                raise ValueError(f"{leafdir}: zero or missing denominator")
            target["rate"] = target["minutes_pop"] / target["pop"]
            # ISO before the division: numerator and denominator sum
            # exactly because impact regions stay within one country
            adm0 = (num.groupby(["ISO", "year"], sort=False)["minutes_pop"]
                    .sum().reset_index()
                    .merge(den_iso, on=["ISO", "year"], validate="one_to_one"))
            adm0["rate"] = adm0["minutes_pop"] / adm0["pop"]
            member = {"batch": batch, "gcm": gcm, "iam": iam, "ssp": ssp}
            target = (target[id_fields + ["year", "rate"]].assign(**member))
            adm0 = adm0[["ISO", "year", "rate"]].assign(**member)
            out.append((
                window_means(target, time_col="year", windows=WINDOWS,
                             value_col="rate"),
                window_means(adm0, time_col="year", windows=WINDOWS,
                             value_col="rate"),
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
        value_col="rate",
        quantiles=QUANTILES,
        weight_col="model_weight",
    )


def write_product(stats: pd.DataFrame, adm: str, rcp: str,
                  extra: dict | None = None) -> None:
    out = Path(OUT) / adm
    out.mkdir(parents=True, exist_ok=True)
    stem = f"labor_physical_{rcp}"
    stats.to_parquet(out / f"{stem}.parquet", index=False)
    manifest = {
        "source": str(TREE),
        "batches": BATCHES,
        "ssps": SSPS,
        "units": "minutes per worker per day, fulladapt minus histclim",
        "population": POP,
        "coverage": "regions with a population baseline; 110 of 24378 "
                    "impact regions (small territories, Antarctica) have "
                    "none and are absent, and targets whose population "
                    "is zero are dropped because their rate is undefined",
        "windows": [f"{a}-{b}" for a, b in WINDOWS],
        "quantiles": QUANTILES,
        "model_weights": SMME.format(rcp=rcp),
    }
    manifest.update(extra or {})
    (out / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
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
    pairs = [p for chunk in results for p in chunk]
    reduced = pd.concat([p[0] for p in pairs], ignore_index=True)

    stats = weighted_stats(reduced, args.rcp)
    write_product(stats, args.adm, args.rcp, {"n_members": len(pairs)})

    if args.adm == "adm1":
        adm0 = pd.concat([p[1] for p in pairs], ignore_index=True)
        adm0_stats = weighted_stats(adm0, args.rcp)
        write_product(adm0_stats, "adm0", args.rcp, {
            "method": "ISO sums of the adm1 numerator and denominator "
                      "before the division",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
