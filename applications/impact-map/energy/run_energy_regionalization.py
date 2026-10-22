"""Regionalize the published energy delta-method products to GADM 2.0.

Scope is the published Impact Map energy product: SSP3, IIASA (the
``low`` IAM), RCP4.5 and RCP8.5, full adaptation, price014. Six inputs:
total_energy valued damages and the two physical fuels, at two RCPs,
plus a percent-of-GDP twin for the valued layer.

Why this is a script and not a pipeline TOML
--------------------------------------------
The Monte Carlo pipeline computes quantiles *from* a sample tree:
``TreeConfig.levels`` and ``StatsConfig.sample_dims`` are both
``Field(min_length=1)``. These inputs are single CSVs whose quantiles
were already collapsed upstream by the delta method, so there is no
sample to pool. ``apply_weights`` is the right entry point for the
spatial step; ``apply.py`` notes there is no CLI verb for it.

Windows come from the package, not from here
--------------------------------------------
``stats.window_means`` does the window averaging and ``stats
._window_label`` the labelling, so the labels match the other sectors
byte for byte, single-year windows included.

What cannot be reused is ``summarize_samples``: it derives statistics
from a sample, and these statistics arrive already collapsed.

Two window sets, because only one of them is constructible:

  mean       2020-2039, 2040-2059, 2080-2099, 2099.  Averaging annual
             means over a window is exact by linearity.
  quantiles  2099 only.  A window mean of per-year quantiles is not a
             window-mean quantile: it would need the year-to-year
             covariance of the underlying distribution, which the
             collapsed files do not carry. The single-year window is an
             identity, so the published quantiles pass through untouched.

``window_means`` is fed one statistic at a time with the ``statistic``
column dropped. That is not evading ``_reject_presummarized``: with a
single statistic in the frame there is genuinely no statistic
dimension, and nothing is ever combined across statistics or units.
The guard exists to stop precomputed quantiles being summed across
*units*, which is the real historical error and which this script
never does.

Quantiles do not aggregate spatially either
-------------------------------------------
Summing quantiles across source regions assumes perfect rank
correlation. The correct object is the gradient, which does not
survive for electricity. So:

  adm0   taken straight from the published ISO extraction, which was
         collapsed from gradients at ISO level inside quantiles.py.
         Its 253 keys match the GADM 2.0 adm0 universe exactly
         (253/253, no symmetric difference). No weights, no aggregation.
  adm1   mean everywhere; quantiles only where the destination draws
         on exactly one source impact region (1,942 of 3,428 units,
         10.0% of population).
  adm2   same rule (40,203 of 42,726 units, 51.7% of population).

Inheritance is exact under both kinds. For an extensive quantity a
single-source destination's total is w*X for a deterministic w, and
quantile_p(w*X) = w*quantile_p(X); running apply_weights on the
quantile column applies that same w. For an intensive quantity the
single source's weight normalizes to 1, so the value passes through.

Each statistic is aggregated in its own ``apply_weights`` call: that
function treats every non-key column as a passthrough dimension, so
leaving the other nine in would group by 24,378 distinct floats and
defeat the aggregation entirely.

Coverage lives in the data, not in a note: every row carries
``n_source_ir`` and ``quantile_method``.

Units
-----
The dependent variable is GJ per capita, not kWh per capita. Traced
through the release repo's dataset construction: IEA World Energy
Balances are filtered to TJ (``energy_load/1_extract_clean_energy_load
_data.do:9``), multiplied by 1000 to GJ (lines 29-33, "Converting to GJ
because plotting scale is more familiar"), then divided by population
(``1_construct_dataset_from_raw_inputs.do:105``). The string "kwh"
appears nowhere in 0_make_dataset or 1_analysis. The csvv's
``depenunit: kWh/pc`` is wrong.

That settles the valuation chain: the price file is $/kWh (electricity
median $0.101), so quantity-times-price is GJ * $/kWh and the stored
"dollar" is short of real dollars by the kWh-per-GJ factor. Hence
``raw * 1.273526 / 0.0036`` to reach 2019 USD, which is what the
release repo's own percent-of-GDP code does.

Run on the cluster:

    python run_energy_regionalization.py                 # all of it
    python run_energy_regionalization.py --dry-run       # validate, write nothing
    python run_energy_regionalization.py --levels adm0   # one level
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

from cil_regionalization.apply import apply_weights
from cil_regionalization.fetch import fetch_weights
from cil_regionalization.stats import window_means

EXTRACT = Path(
    "/project/cil/battuta_shares/gcp/social/parameters/energy_pixel_interaction"
    "/extraction/multi-models/rationalized_code"
    "/break2_Exclude_all-issues_semi-parametric/TINV_clim_GMFD"
)
OUT = Path("/project/cil/gcp/impact_map/energy")
ZARR = ("/project/cil/gcp/integration/float32/dscim_input_data/econvars/"
        "zarrs/integration-econ-bc39.zarr")
DATA_VERSION = "world-combo-201710"
SSP, IAM = "SSP3", "low"
RCPS = ["rcp45", "rcp85"]

WINDOWS_MEAN = [(2020, 2039), (2040, 2059), (2080, 2099), (2099, 2099)]
WINDOWS_QUANTILE = [(2099, 2099)]

GDP_MODEL = {"low": "IIASA GDP", "high": "OECD Env-Growth"}
USD2005_TO_2019 = 1.273526   # clean_data.R:52, from the mortality paper
GJ_PER_KWH = 0.0036          # price is $/kWh, quantity is GJ/pc
CASPIAN = "CA-"              # phantom LandScan population, denominator meaningless

STATS = {
    "mean": "mean", "q5": "q05", "q10": "q10", "q17": "q17", "q25": "q25",
    "q50": "q50", "q75": "q75", "q83": "q83", "q90": "q90", "q95": "q95",
}
QUANTILES = [s for s in STATS.values() if s != "mean"]

# The two *_SSP3_rcp* variants of these files are truncated (to 2096 and
# to 1983) and are deliberately not referenced here.
MEASURES = {
    "valued": dict(
        stem="energy_valued", value="damage", kind="extensive",
        sub="total_energy", pctgdp=True,
        units=("expenditure change in the projection system's native dollars "
               "(2005 USD divided by the kWh-per-GJ factor); multiply by "
               "1.273526/0.0036 for 2019 USD"),
        ir="SSP3-{rcp}_damage-price014_median_fulluncertainty_low_fulladapt-levels.csv",
        iso="SSP3-{rcp}_iso_damage-price014_median_fulluncertainty_low_fulladapt-aggregated.csv",
    ),
    "physical_electricity": dict(
        stem="energy_physical_electricity", value="impactpc", kind="intensive",
        sub="median_OTHERIND_electricity_TINV_clim_GMFD", pctgdp=False,
        units="gigajoules per capita per year, full adaptation minus histclim",
        ir="SSP3-{rcp}_impactpc_median_fulluncertainty_low_fulladapt.csv",
        iso="SSP3-{rcp}_iso_impactpc_median_fulluncertainty_low_fulladapt-aggregated.csv",
    ),
    "physical_other_energy": dict(
        stem="energy_physical_other_energy", value="impactpc", kind="intensive",
        sub="median_OTHERIND_other_energy_TINV_clim_GMFD", pctgdp=False,
        units="gigajoules per capita per year, full adaptation minus histclim",
        ir="SSP3-{rcp}_impactpc_median_fulluncertainty_low_fulladapt.csv",
        iso="SSP3-{rcp}_iso_impactpc_median_fulluncertainty_low_fulladapt-aggregated.csv",
    ),
}

ARTIFACTS = {
    ("adm1", "extensive"): "gadm20-adm1-per-source",
    ("adm2", "extensive"): "gadm20-adm2-per-source",
    ("adm1", "intensive"): "gadm20-adm1-per-destination",
    ("adm2", "intensive"): "gadm20-adm2-per-destination",
}
KEYS = {"adm0": ["ISO"], "adm1": ["ISO", "ID_1"], "adm2": ["ISO", "ID_1", "ID_2"]}

_ARTIFACT_CACHE: dict[str, object] = {}
_GDP_CACHE: dict[str, pd.DataFrame] = {}
_READ_CACHE: dict[str, pd.DataFrame] = {}


def artifact(name):
    if name not in _ARTIFACT_CACHE:
        print(f"    artifact {name} ...", flush=True)
        _ARTIFACT_CACHE[name] = fetch_weights(name)
    return _ARTIFACT_CACHE[name]


def read_published(path: Path) -> pd.DataFrame:
    """Read an extraction CSV and normalise its columns.

    Single-entry cache: adm1 and adm2 consume the same impact-region
    file, and re-parsing ~600 MB of CSV for the second level is the
    largest avoidable cost in the run. Holding one frame rather than
    all of them keeps peak memory to a single input.
    """
    key = str(path)
    if key in _READ_CACHE:
        return _READ_CACHE[key]
    _READ_CACHE.clear()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = set(STATS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing statistic columns {sorted(missing)}")
    df = df[["region", "year"] + list(STATS)].rename(columns={"region": "hierid"})
    df = df[df["hierid"].notna() & (df["hierid"] != "")]
    _READ_CACHE[key] = df
    return df


def source_counts(art, keys: list[str]) -> pd.DataFrame:
    return (
        art.frame.groupby(keys, as_index=False)["hierid"]
        .nunique()
        .rename(columns={"hierid": "n_source_ir"})
    )


def annual_adm0(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """adm0 comes from the published ISO extraction, already collapsed.

    No weights and no aggregation: these quantiles were produced by
    summing gradients to ISO level inside quantiles.py, which is the
    correct object. n_source_ir is carried for information only; the
    quantiles are valid at every unit regardless of its value.
    """
    counts = source_counts(artifact("gadm20-adm1-per-source"), ["ISO"])
    out = df.rename(columns={"hierid": "ISO"}).melt(
        id_vars=["ISO", "year"], value_vars=list(STATS),
        var_name="statistic", value_name=spec["value"],
    )
    out["statistic"] = out["statistic"].map(STATS)
    return out.merge(counts, on="ISO", how="left")


def annual_aggregate(df: pd.DataFrame, level: str, spec: dict) -> pd.DataFrame:
    """Aggregate one input to one GADM level, one statistic at a time."""
    art = artifact(ARTIFACTS[(level, spec["kind"])])
    keys = KEYS[level]
    counts = source_counts(art, keys)

    # Destination -> its only source region, for contained units. Used to
    # repair a pandas default: groupby().sum() over an all-NaN group
    # returns 0.0, so an extensive aggregation of a contained unit whose
    # source quantile is missing upstream would publish a hard zero
    # instead of a gap. The intensive path returns NaN correctly, but the
    # guard is applied to both so the two kinds behave the same.
    pairs = art.frame[keys + ["hierid"]].drop_duplicates()
    contained = pairs.merge(
        counts.loc[counts["n_source_ir"] == 1, keys], on=keys, how="inner"
    )

    pieces = []
    for src_col, label in STATS.items():
        data = df[["hierid", "year", src_col]].rename(columns={src_col: "value"})
        res = apply_weights(
            art, data, kind=spec["kind"], weight="pop",
            data_version=DATA_VERSION, value_col="value",
        )
        out = res.frame.rename(columns={"value": spec["value"]})

        gaps = data.loc[data["value"].isna(), ["hierid", "year"]]
        if len(gaps):
            hit = contained.merge(gaps, on="hierid")[keys + ["year"]].drop_duplicates()
            out = out.merge(hit.assign(_gap=True), on=keys + ["year"], how="left")
            out.loc[out["_gap"].notna(), spec["value"]] = pd.NA
            out = out.drop(columns=["_gap"])
            print(f"      {label}: {len(out):,} rows "
                  f"({len(hit):,} contained units nulled for upstream gaps)", flush=True)
        else:
            print(f"      {label}: {len(out):,} rows", flush=True)
        out["statistic"] = label
        pieces.append(out)

    out = pd.concat(pieces, ignore_index=True).merge(counts, on=keys, how="left")
    # Quantiles are only meaningful where the destination drew on one
    # source region. Null the rest before windowing.
    multi = out["n_source_ir"] > 1
    out.loc[multi & out["statistic"].isin(QUANTILES), spec["value"]] = pd.NA
    return out


def to_windows(annual: pd.DataFrame, value: str) -> pd.DataFrame:
    """Window means through stats.window_means, one statistic at a time.

    The mean takes all four windows; the quantiles take 2099 only,
    where the window mean is an identity and the published values pass
    through unchanged (including NaN, since Series.mean of an all-NaN
    group is NaN rather than the 0.0 that sum would give).
    """
    pieces = []
    for label in annual["statistic"].unique():
        sub = annual.loc[annual["statistic"] == label].drop(columns=["statistic"])
        windows = WINDOWS_MEAN if label == "mean" else WINDOWS_QUANTILE
        out = window_means(sub, time_col="year", windows=windows, value_col=value)
        out["statistic"] = label
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def mark_coverage(out: pd.DataFrame, value: str, level: str) -> pd.DataFrame:
    """Say in the data how each row got its value.

    The published extraction carries NaN quantiles for 993 of 30,360
    ISO region-years (3.3%) and 6,850 of 2,925,360 impact region-years
    in the valued rcp85 file (0.23%), concentrated in uninhabited
    territories (ATA, ATF, BVT, HMD, IOT, SGS and the CL-/SP- hierarchy
    codes have all 120 years missing). Those are brentq failures in
    weights_vcv.WeightedGMCDF.inverse, which appends NaN when it cannot
    bracket a root; the mean is present in every one of them. Marking
    them separately keeps an upstream gap from reading as an
    aggregation limit.
    """
    base = ("delta_method_collapsed" if level == "adm0" else "inherited_single_ir")
    out["quantile_method"] = base
    if level != "adm0":
        out.loc[out["n_source_ir"] > 1, "quantile_method"] = "unavailable_multi_ir"
    gap = (out[value].isna() & (out["statistic"] != "mean")
           & (out["quantile_method"] != "unavailable_multi_ir"))
    out.loc[gap, "quantile_method"] = "unavailable_upstream_nan"
    # The column describes quantile availability. The mean is valid at
    # every unit regardless of how many source regions it drew on, so a
    # quantile verdict on a mean row would misread; n_source_ir still
    # carries the coverage fact there.
    out.loc[out["statistic"] == "mean", "quantile_method"] = "not_applicable"
    return out


def gdp_at_level(level: str) -> pd.DataFrame:
    """SSP3 / IIASA GDP from bc39, summed to the target units, windowed.

    adm0 sums member impact regions by ISO, exact because impact
    regions do not cross national borders. adm1 and adm2 use the same
    per_source population weights that aggregated the numerator, which
    is the population-portioned sum run_pct_gdp.py uses. The window
    mean then matches the numerator's, giving the house convention of
    window mean dollars over window mean GDP.
    """
    if level in _GDP_CACHE:
        return _GDP_CACHE[level]
    ds = xr.open_zarr(ZARR)
    g = (ds["gdp"].sel(ssp=SSP, model=GDP_MODEL[IAM])
         .to_dataframe()["gdp"].reset_index()[["region", "year", "gdp"]]
         .rename(columns={"region": "hierid"}))
    g["hierid"] = g["hierid"].astype(str)
    print(f"    bc39 GDP: {len(g):,} rows, {g['hierid'].nunique():,} regions, "
          f"{g['year'].min()}-{g['year'].max()}", flush=True)
    if level == "adm0":
        g["ISO"] = g["hierid"].str.split(".").str[0]
        unit = g.groupby(["ISO", "year"], as_index=False)["gdp"].sum()
    else:
        res = apply_weights(
            artifact(ARTIFACTS[(level, "extensive")]),
            g.rename(columns={"gdp": "value"}),
            kind="extensive", weight="pop",
            data_version=DATA_VERSION, value_col="value",
        )
        unit = res.frame.rename(columns={"value": "gdp"})
    out = window_means(unit, time_col="year", windows=WINDOWS_MEAN, value_col="gdp")
    _GDP_CACHE[level] = out
    return out


def build_pctgdp(valued: pd.DataFrame, level: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Share of GDP: raw * 1.273526 / 0.0036, over bc39 GDP.

    Dimensionless share, matching mortality_valued_pctgdp. GDP is a
    deterministic constant per (region, window), so dividing each
    published statistic equals dividing before the quantiles were
    taken; no statistics-ordering issue arises.
    """
    keys = KEYS[level]
    gdp = gdp_at_level(level)
    out = valued.merge(gdp, on=keys + ["window"], how="inner")
    bad = out["gdp"].isna() | (out["gdp"] <= 0) | (out["ISO"] == CASPIAN)
    dropped = int(bad.sum())
    out = out[~bad].copy()
    out["value"] = (out[spec["value"]] * USD2005_TO_2019 / GJ_PER_KWH) / out["gdp"]
    out = out.drop(columns=["gdp", spec["value"]])
    print(f"    pctgdp: {len(out):,} rows; {dropped:,} dropped for zero/missing "
          f"GDP or {CASPIAN}", flush=True)
    return out, dict(dropped_rows=dropped)


def finish(out: pd.DataFrame, level: str, rcp: str, value: str) -> pd.DataFrame:
    out = out.assign(rcp=rcp, iam=IAM, ssp=SSP)
    cols = (KEYS[level] + ["rcp", "iam", "ssp", "window", "statistic",
                           value, "n_source_ir", "quantile_method"])
    return out[cols].sort_values(
        KEYS[level] + ["window", "statistic"]).reset_index(drop=True)


WINDOW_NOTE = (
    "window labels come from cil_regionalization.stats.window_means and "
    "_window_label, the same functions the other sectors use, so the labels "
    "match exactly. The mean carries all four windows. The quantiles carry "
    "2099 only: a window mean of per-year quantiles is not a window-mean "
    "quantile, because it would need the year-to-year covariance of the "
    "underlying distribution and the collapsed delta-method files do not "
    "carry it. At a single-year window the mean is an identity, so the "
    "published quantiles pass through untouched. Rows for a quantile at a "
    "multi-year window are absent, not null."
)
QUANTILE_NOTE = (
    "quantiles do not aggregate spatially: summing them across source regions "
    "assumes perfect rank correlation. quantile_method records how each row "
    "got its value. delta_method_collapsed = taken from the published ISO "
    "extraction, collapsed from gradients inside quantiles.py. "
    "inherited_single_ir = the destination draws on exactly one impact region, "
    "so the quantile transfers exactly (scaled by the allocation weight for "
    "the extensive kind). unavailable_multi_ir = several source regions, "
    "quantile is null. unavailable_upstream_nan = the published extraction had "
    "no value there, a brentq bracketing failure in WeightedGMCDF.inverse, "
    "mostly uninhabited territories. not_applicable = a mean row; the mean is "
    "valid everywhere."
)


def write_manifest(dest: Path, source: Path, level: str, spec: dict,
                   frame: pd.DataFrame, kind: str, extra: dict | None = None) -> None:
    q = frame[frame["statistic"] != "mean"]
    vcol = "value" if kind == "pctgdp" else spec["value"]
    man = {
        "source": str(source),
        "scenario": f"{SSP}, IIASA ({IAM} IAM), {spec['sub']}, full adaptation, price014",
        "scenario_note": (
            "the published Impact Map energy product is SSP3 / IIASA at RCP4.5 "
            "and RCP8.5 only; there is no 2 rcp x 3 ssp x 2 iam matrix for energy"
        ),
        "level": level,
        "region_keys": KEYS[level],
        "time_key": "window",
        "windows_mean": [f"{lo}-{hi}" if lo != hi else f"{lo}"
                         for lo, hi in WINDOWS_MEAN],
        "windows_quantile": [f"{lo}" for lo, hi in WINDOWS_QUANTILE],
        "window_note": WINDOW_NOTE,
        "statistics": sorted(frame["statistic"].unique().tolist()),
        "units": spec["units"] if kind != "pctgdp" else (
            "share of GDP, dimensionless; numerator converted to 2019 USD and "
            "the bc39 series is at the same price level"
        ),
        "aggregation": (
            "none: the published ISO extraction is used directly" if level == "adm0"
            else f"{spec['kind']}, population-weighted, "
                 f"{ARTIFACTS[(level, spec['kind'])]}"
        ),
        "weights_source_version": DATA_VERSION,
        "quantile_method_note": QUANTILE_NOTE,
        "quantile_coverage": {
            k: int(v) for k, v in frame["quantile_method"].value_counts().items()
        },
        "quantile_rows_populated": int(q[vcol].notna().sum()),
        "quantile_rows_total": int(len(q)),
        "units_provenance": (
            "the dependent variable is GJ per capita, not kWh per capita: IEA "
            "World Energy Balances filtered to TJ then multiplied by 1000 "
            "(0_make_dataset/energy_load/1_extract_clean_energy_load_data.do:9,29) "
            "and divided by population (1_construct_dataset_from_raw_inputs.do:105). "
            "The csvv's depenunit: kWh/pc is wrong; 'kwh' appears nowhere in "
            "0_make_dataset or 1_analysis"
        ),
    }
    if kind == "pctgdp":
        man.update({
            "denominator": ZARR,
            "denominator_mapping": GDP_MODEL,
            "denominator_years": "2010-2100",
            "year_range_note": (
                "bc39 starts at 2010 while the damage files start at 1981. All "
                "four windows fall inside the bc39 range, so nothing is lost "
                "here; the limit would bind only for a window before 2010"
            ),
            "convention": (
                f"window mean dollars over window mean GDP. Numerator is "
                f"raw * {USD2005_TO_2019} (2005 to 2019 USD, clean_data.R:52) "
                f"/ {GJ_PER_KWH} (kWh per GJ; the price file is $/kWh and the "
                f"quantity is GJ/pc)"
            ),
            "dropped_note": (
                f"rows with zero or missing GDP (uninhabited regions) and the "
                f"Caspian Sea region {CASPIAN}, whose denominator is a LandScan "
                f"artifact (15.5 million phantom people)"
            ),
        })
    if extra:
        man.update(extra)
    dest.write_text(json.dumps(man, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", nargs="+", default=["adm0", "adm1", "adm2"],
                    choices=["adm0", "adm1", "adm2"])
    ap.add_argument("--measures", nargs="+", default=list(MEASURES),
                    choices=list(MEASURES))
    ap.add_argument("--rcps", nargs="+", default=RCPS, choices=RCPS)
    ap.add_argument("--no-pctgdp", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="read, aggregate and report, but write nothing")
    args = ap.parse_args()

    for measure in args.measures:
        spec = MEASURES[measure]
        for rcp in args.rcps:
            for level in args.levels:
                tmpl = spec["iso"] if level == "adm0" else spec["ir"]
                path = EXTRACT / spec["sub"] / tmpl.format(rcp=rcp)
                print(f"\n== {measure} / {rcp} / {level}\n   {path.name}", flush=True)
                df = read_published(path)
                print(f"   {len(df):,} rows, {df['hierid'].nunique():,} source units,"
                      f" {df['year'].min()}-{df['year'].max()}", flush=True)

                annual = (annual_adm0(df, spec) if level == "adm0"
                          else annual_aggregate(df, level, spec))
                out = to_windows(annual, spec["value"])
                out = mark_coverage(out, spec["value"], level)
                out = finish(out, level, rcp, spec["value"])

                q = out[out["statistic"] != "mean"]
                print(f"   -> {len(out):,} rows, "
                      f"{out[KEYS[level]].drop_duplicates().shape[0]:,} units, "
                      f"windows {sorted(out['window'].unique())}; quantile rows "
                      f"populated {int(q[spec['value']].notna().sum()):,}/{len(q):,}",
                      flush=True)

                dest = OUT / level / f"{spec['stem']}_{rcp}.parquet"
                if not args.dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    out.to_parquet(dest, index=False)
                    write_manifest(
                        dest.parent / f"{spec['stem']}_{rcp}.manifest.json",
                        path, level, spec, out, "value")
                    print(f"   wrote {dest}", flush=True)

                if spec["pctgdp"] and not args.no_pctgdp:
                    pct, info = build_pctgdp(out, level, spec)
                    pct = pct[KEYS[level] + ["rcp", "iam", "ssp", "window",
                                             "statistic", "value", "n_source_ir",
                                             "quantile_method"]]
                    pdest = OUT / level / f"{spec['stem']}_pctgdp_{rcp}.parquet"
                    if not args.dry_run:
                        pct.to_parquet(pdest, index=False)
                        write_manifest(
                            pdest.parent / f"{spec['stem']}_pctgdp_{rcp}.manifest.json",
                            dest, level, spec, pct, "pctgdp", extra=info)
                        print(f"   wrote {pdest}", flush=True)

                if args.dry_run:
                    print("   dry run, nothing written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
