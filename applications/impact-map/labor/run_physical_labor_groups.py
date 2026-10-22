"""Labor physical effect per risk group, impact region and ADM levels.

The tree's ``rebased`` is the all-worker series: the two group series
mixed by the high risk share and then recentered,

    rebased = (high x clip + low x (1 - clip)) - mean 2001-2010 of that mix

per region (the file's ``orderofoperations`` records the sequence, and
the reconstruction holds to float32 storage precision on both the
fulladapt and histclim files). The group series ``highriskimpacts``
and ``lowriskimpacts`` live in the same two files per leaf, raw: they
are not rebased, so the pipeline's sibling subtraction cannot produce
their effect. Differencing the raw series would leave a spurious
per-region, per-draw constant, because histclim's 2001-2010 is
resampled climatology while the scenario's is that draw's weather and
the two baselines differ. That recentering step is why this is a
runner and not a config.

Per group, the effect follows the combined series' own convention:

    (scen - mean 2001-2010 of scen) - (hist - mean 2001-2010 of hist)

computed per region within each file. One run per rcp and level reads
each leaf pair once and emits both group products; the reads are the
expensive part and they are the same reads the combined run does.

Levels. ``--adm ir`` writes the impact region product: no weights, no
aggregation, rows keyed by hierid, every region in the tree. ``--adm
adm1`` and ``--adm adm2`` aggregate the rate through the ratio route
of run_physical_labor.py: minutes times population and population
each aggregate extensively and the division happens at the target,
with the merged SSP population baselines as the denominator. The
weighting is by total population, matching the combined physical
product, not by risk group worker counts: no per group baseline
exists, and the group share varies by draw and year. The adm1 run
also writes ADM0 by summing the numerator and denominator to ISO
before the division, exact because impact regions do not cross
national borders. The 59 countries the SSP series lacks carry the
2011 UNPD-scaled historical population held constant, the projection
system's own fallback convention (see `historical_fallback`), which
leaves only uninhabited units absent: 9 ADM0, 22 ADM1, and 46 ADM2
relative to full coverage.

Two caveats the manifests carry. Each product is minutes per worker of
its own risk group, a different denominator from the other group, so
the two series are not comparable as levels. And they do not recombine
into the published all-worker series: the mix happens before rebasing
and with different ``clip`` on each side (the rebasing memo's four
objects, not two; ``clip`` differs from histclim's in nearly every
region-year). The combined product alongside is the all-worker series.

All levels carry the same four windows, the aggregated three plus the
final projection year (2099) alone, so a group product compares across
levels without a window mismatch.

Run on the cluster:

    python run_physical_labor_groups.py --adm ir --rcp rcp85 --workers 16
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
UNPD15 = ("/project/cil/gcp/social/baselines/population/"
          "POP-ImpactRegion-HIST-UNPD15.nc")
FALLBACK_YEAR = 2011
WEIGHTS = "/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/data/weights/ir_{adm}_full"
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
OUT_IR = "/project/cil/gcp/impact_map/impact_region/labor"
OUT_ADM = "/project/cil/gcp/impact_map/labor"
BATCHES = [f"batch{i}" for i in range(15)]
SSPS = ["SSP2", "SSP3", "SSP4"]
IAMS = ["high", "low"]
WINDOWS = [(2020, 2039), (2040, 2059), (2080, 2099), (2099, 2099)]
RECENTER = (2001, 2010)
YEARS = range(2020, 2100)
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]
GROUPS = {"highriskimpacts": "highrisk", "lowriskimpacts": "lowrisk"}
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


def smme_map(rcp: str) -> dict[str, float]:
    """SMME weights as normalized model name -> weight. Attaching the
    weight per leaf frame replaces a merge over the full pooled frame,
    the same memory-driven choice the pipeline makes."""
    smme = pd.read_csv(SMME.format(rcp=rcp), sep="\t")
    keys = smme["model"].str.replace("*", "", regex=False).str.lower()
    return dict(zip(keys, smme["weight"].astype(float)))


def load_population(ssp: str) -> pd.DataFrame:
    """Per-region population for 2020-2099, five-year values held flat.

    Same series and conventions as run_physical_labor.py: the merged
    baselines carry one value per region every fifth year, the
    projection system holds it constant within the span, and the 59
    countries the SSP series lacks carry the 2011 historical
    population held constant instead (see `historical_fallback`).
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
    fallback = historical_fallback(set(base["hierid"].unique()))
    if len(fallback):
        held = years[["year"]].merge(fallback, how="cross")
        out = pd.concat([out, held], ignore_index=True)
    return out


def historical_fallback(covered: set) -> pd.DataFrame:
    """2011 historical population for regions the SSP series lacks.

    The projection system's own convention for these countries is a
    fixed 2011 population ("LandScan 2011 held constant" in the
    paper), and the tree's wage factor is zero exactly where that
    baseline is zero, which confirms it was used. The values here are
    the UNPD-scaled 2011 product, not the LandScan numbers the runs
    embedded (social/weightlines/population.csv): the two differ by
    scaling, and the weightlines file assigns the Caspian Sea polygon
    (CA-) 15.5 million people at 2011, a LandScan coastal artifact
    this file does not carry. Six regions absent from both sources
    are uninhabited and stay absent; zero-population rows come
    through and their all-zero targets drop at the alive filter.
    """
    import xarray as xr

    with xr.open_dataset(UNPD15) as ds:
        series = ds["population"].sel(year=FALLBACK_YEAR).to_series()
    held = series[~series.index.isin(covered)]
    return (held.rename("pop").rename_axis("hierid").reset_index()
            [["hierid", "pop"]])


def _rebase(df: pd.DataFrame) -> pd.DataFrame:
    """Recenter both group series per region on their 2001-2010 means.

    The same convention the tree's combined ``rebased`` applies after
    mixing; here it applies to each raw group series within its own
    file, before the histclim subtraction.
    """
    lo, hi = RECENTER
    groups = list(GROUPS)
    base = (df.loc[df["year"].between(lo, hi)]
            .groupby("hierid", sort=False)[groups].mean().reset_index())
    out = df.merge(base, on="hierid", suffixes=("", "_base"),
                   validate="many_to_one")
    for g in groups:
        out[g] = out[g].astype("float64") - out[f"{g}_base"].astype("float64")
    return out[["hierid", "year", *groups]]


def read_effect(leafdir: Path) -> pd.DataFrame:
    """Both groups' rebased fulladapt minus rebased histclim, window years."""
    kw = dict(variables=list(GROUPS), region_dim="region",
              region_col="hierid", region_labels="regions")
    scen = _rebase(read_netcdf_leaf(leafdir / f"{STEM}.nc4", **kw))
    hist = _rebase(read_netcdf_leaf(leafdir / f"{STEM}-histclim.nc4", **kw))
    df = scen.merge(hist, on=["hierid", "year"], suffixes=("", "_h"),
                    validate="one_to_one")
    for g in GROUPS:
        df[g] = df[g] - df[f"{g}_h"]
    df = df[df["year"].isin(YEARS)]
    return df[["hierid", "year", *GROUPS]]


def _reduce(frame: pd.DataFrame, value_col: str, weight: float,
            key_col: str) -> pd.DataFrame:
    """Window means plus the leaf's model weight, with the repeated
    string columns dictionary-encoded before thousands of reduced
    frames concatenate in the parent."""
    reduced = window_means(frame, time_col="year", windows=WINDOWS,
                           value_col=value_col)
    reduced["model_weight"] = weight
    reduced[key_col] = reduced[key_col].astype("category")
    reduced["window"] = reduced["window"].astype("category")
    return reduced


def batch_frames(task) -> list[dict[str, pd.DataFrame]]:
    """Per-product window-mean frames for every (gcm, iam) slice of one
    batch and ssp. Each leaf pair is read once; both groups fall out of
    the same read. One task per (batch, ssp): 45 tasks, so worker
    counts past 15 still shorten the run.
    """
    batch, ssp, rcp, adm = task
    weights_of = smme_map(rcp)
    if adm != "ir":
        artifact = cilreg.WeightsArtifact.load(WEIGHTS.format(adm=adm))
        id_fields = list(artifact.schema.id_fields)
        pop = load_population(ssp)
        # The SSP series plus the 2011 historical fallback covers
        # every inhabited region, exactly like the combined physical
        # product; zero-population targets drop at the alive filter
        # because a rate there is undefined
        covered = {(h,) for h in pop["hierid"].unique()}
        den = cilreg.apply_weights(
            artifact, pop, kind="extensive", weight="pop",
            value_col="pop", data_version="world-combo-201710",
            restrict_to_sources=covered,
        ).frame
        alive = den.groupby(id_fields, sort=False)["pop"].min()
        alive = alive[alive > 0].reset_index()[id_fields]
        den = den.merge(alive, on=id_fields)
        den_iso = (den.groupby(["ISO", "year"], sort=False)["pop"]
                   .sum().reset_index())
    out = []
    for iam in IAMS:
        for gcm in gcms_for(rcp):
            leafdir = TREE / batch / rcp / gcm / iam / ssp
            key = gcm.replace("surrogate_", "").lower()
            if key not in weights_of:
                raise ValueError(f"{gcm} has no SMME weight row")
            weight = weights_of[key]
            df = read_effect(leafdir)
            member = {"batch": batch, "gcm": gcm, "iam": iam, "ssp": ssp}
            frames: dict[str, pd.DataFrame] = {}
            if adm == "ir":
                for var, label in GROUPS.items():
                    frame = (df[["hierid", "year", var]]
                             .rename(columns={var: "rebased"})
                             .assign(**member))
                    frames[label] = _reduce(frame, "rebased", weight, "hierid")
            else:
                merged = df.merge(pop, on=["hierid", "year"])
                for var, label in GROUPS.items():
                    col = f"{var}_pop"
                    merged[col] = merged[var] * merged["pop"]
                    num = cilreg.apply_weights(
                        artifact, merged[["hierid", "year", col]],
                        kind="extensive", weight="pop", value_col=col,
                        data_version="world-combo-201710",
                        restrict_to_sources=covered,
                    ).frame
                    target = num.merge(den, on=id_fields + ["year"],
                                       validate="one_to_one")
                    if target["pop"].eq(0).any() or target["pop"].isna().any():
                        raise ValueError(
                            f"{leafdir}: zero or missing denominator")
                    target["rate"] = target[col] / target["pop"]
                    target = (target[id_fields + ["year", "rate"]]
                              .assign(**member))
                    frames[label] = _reduce(target, "rate", weight, "ISO")
                    if adm == "adm1":
                        # ISO before the division: numerator and
                        # denominator sum exactly because impact
                        # regions stay within one country
                        adm0 = (num.groupby(["ISO", "year"], sort=False)[col]
                                .sum().reset_index()
                                .merge(den_iso, on=["ISO", "year"],
                                       validate="one_to_one"))
                        adm0["rate"] = adm0[col] / adm0["pop"]
                        adm0 = adm0[["ISO", "year", "rate"]].assign(**member)
                        frames[f"{label}_adm0"] = _reduce(
                            adm0, "rate", weight, "ISO")
            out.append(frames)
        print(batch, rcp, ssp, iam, flush=True)
    return out


def write_product(stats: pd.DataFrame, variable: str, label: str, adm: str,
                  rcp: str, n_members: int, adm0: bool = False) -> None:
    level = "adm0" if adm0 else adm
    out = Path(OUT_IR) if adm == "ir" else Path(OUT_ADM) / level
    out.mkdir(parents=True, exist_ok=True)
    stem = f"labor_physical_{label}_{rcp}"
    for col in stats.columns:
        if isinstance(stats[col].dtype, pd.CategoricalDtype):
            stats[col] = stats[col].astype(str)
    stats.to_parquet(out / f"{stem}.parquet", index=False)
    risk = "high risk" if label == "highrisk" else "low risk"
    manifest = {
        "source": str(TREE),
        "variable": variable,
        "batches": BATCHES,
        "ssps": SSPS,
        "units": f"minutes per worker per day for a worker in the {risk} "
                 f"group, full adaptation minus histclim",
        "rebase": "each side recentered per region on its own 2001-2010 "
                  "mean before differencing, the convention the tree's "
                  "combined rebased series applies after mixing; the raw "
                  "group series carry no rebasing of their own",
        "denominator": "per worker of this risk group; the companion "
                       "product covers the other group's workers, a "
                       "different denominator, so the two series are not "
                       "comparable as levels",
        "recombination": "these products do not recombine into the "
                         "published all-worker series: the mix happens "
                         "before rebasing and with different clip on each "
                         "side, so recovering it needs the four objects "
                         "(both group series and both clip fields), not "
                         f"two. For the all-worker effect use "
                         f"labor_physical_{rcp}.parquet alongside.",
        "windows": [f"{a}-{b}" if a != b else f"{a}" for a, b in WINDOWS],
        "quantiles": QUANTILES,
        "model_weights": SMME.format(rcp=rcp),
        "n_members": n_members,
    }
    if adm == "ir":
        manifest["spatial_aggregation"] = "none: per impact region"
        manifest["coverage"] = ("every region the tree carries, keyed by "
                                "hierid; no weights are applied")
    else:
        manifest["population"] = POP
        manifest["population_fallback"] = (
            f"{UNPD15} at {FALLBACK_YEAR}, held constant"
        )
        manifest["fallback_note"] = (
            "for the 59 countries absent from the SSP series the "
            "denominator is the 2011 UNPD-scaled historical population "
            "held constant, the projection system's own fallback "
            "convention, so the weighting there does not capture "
            "population growth. The values are UNPD-scaled, not the "
            "LandScan numbers the runs embedded. Those countries' "
            "projections also ran on an imputed global-median income "
            "path rather than country income data."
        )
        manifest["weighting"] = (
            "population-weighted mean through the ratio route of the "
            "combined physical product: minutes times population and "
            "population aggregate extensively and the division happens "
            "at the target. Weighted by total population, not risk group "
            "worker counts: no per group baseline exists and the group "
            "share varies by draw and year."
        )
        manifest["coverage"] = (
            "regions with an SSP baseline plus the 2011 historical "
            "fallback; only uninhabited units are absent, and targets "
            "whose population is zero are dropped. The same gap as the "
            "combined product: 9 ADM0, 22 ADM1, and 46 ADM2 units "
            "relative to full coverage."
        )
        if adm0:
            manifest["method"] = ("ISO sums of the adm1 numerator and "
                                  "denominator before the division")
    (out / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}/{stem}.parquet rows={len(stats)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adm", choices=["ir", "adm1", "adm2"], required=True)
    parser.add_argument("--rcp", choices=["rcp45", "rcp85"], required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tasks = [(b, ssp, args.rcp, args.adm) for b in BATCHES for ssp in SSPS]
    if args.workers == 1:
        results = [batch_frames(t) for t in tasks]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(batch_frames, tasks)
    slices = [s for chunk in results for s in chunk]

    value_col = "rebased" if args.adm == "ir" else "rate"
    # one product at a time: pool, write, release, then the next, so
    # only one product's combined frame is resident on top of the
    # shared slices
    for var, label in GROUPS.items():
        keys = [label] + ([f"{label}_adm0"] if args.adm == "adm1" else [])
        for key in keys:
            reduced = pd.concat([s.pop(key) for s in slices],
                                ignore_index=True)
            stats = pooled_statistics(
                reduced,
                sample_dims=["batch", "gcm"],
                value_col=value_col,
                quantiles=QUANTILES,
                weight_col="model_weight",
            )
            write_product(stats, var, label, args.adm, args.rcp,
                          len(slices), adm0=key.endswith("_adm0"))
            del reduced, stats
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
