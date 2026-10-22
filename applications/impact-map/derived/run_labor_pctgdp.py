"""Labor disutility as a share of GDP, impact region and ADM levels.

At impact region level no division happens, because none is needed:
the tree's wage valuation is ``population x income x 0.00487`` (the
constant is 365 x 0.6 / (250 days x 60 minutes x 6 hours x 0.5), the
paper's labor share and elasticity of labor supply), and region GDP is
``population x income`` from the same series, so the share of the
region's own GDP is exactly ``0.00487 x`` the physical minutes effect.
The sector's own percent-GDP aggregation config encodes the same
cancellation, and the tree's ``-gdp-levels`` files equal 0.00487 times
``rebased`` to float precision. The IR product here is therefore the
physical product rescaled, and its manifest says so: it is the same
object as ``labor_physical``, not an independent result.

At ADM levels the cancellation does not survive aggregation: the unit
share is total disutility dollars over total unit GDP, which is the
population-times-income weighted mean of member shares, not the
population weighted mean. The two levels do not aggregate into each
other, and neither aligns with the population-weighted physical rate;
the manifests carry that note.

The denominator is built at matching vintage, so no deflator enters:
the projection system's own income series (2005 PPP USD; the merged
GDPpc baseline and growth through the hierarchical provider logic,
nightlights-scaled to impact regions) times the population series the
rate runners use (merged SSP baselines with the 2011 UNPD-scaled
historical fallback). For the 59 countries absent from the SSP income
series the provider's fallback is the global median income path, the
same imputation the projections themselves ran on.

Run on the cluster:

    python run_labor_pctgdp.py
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/project/cil/gcp/impact_map")
IR_DIR = BASE / "impact_region" / "labor"
ADM_DIR = BASE / "labor"
WEIGHTS = ("/project/cil/home_dirs/scadavidsanchez/repos/"
           "climate-and-damages-aggregations/data/weights/ir_{adm}_full/"
           "weights.parquet")
POP = "/project/cil/gcp/social/baselines/population/merged/population-merged.{ssp}.csv"
UNPD15 = ("/project/cil/gcp/social/baselines/population/"
          "POP-ImpactRegion-HIST-UNPD15.nc")
GDPPC_BASELINE = "/project/cil/gcp/social/baselines/gdppc-merged-nohier.csv"
GDPPC_GROWTH = "/project/cil/gcp/social/baselines/gdppc-growth.csv"
NIGHTLIGHTS = "/project/cil/gcp/social/baselines/nightlight_weight_normalized.csv"
FALLBACK_YEAR = 2011
STARTYEAR = 2010
YEARS = range(2020, 2100)
RCPS = ["rcp45", "rcp85"]
FACTOR = 0.00487  # 365 x 0.6 / (250 x 60 x 6 x 0.5), as the tree embeds it

IDENTITY_NOTE = (
    "at impact region level the share of GDP is the physical effect "
    "times 0.00487, because the wage valuation is population x income "
    "x 0.00487 and region GDP is population x income from the same "
    "series: the denominator cancels. labor_valued_pctgdp at impact "
    "region level and labor_physical are the same object scaled, not "
    "independent results. The constant is 365 x 0.6 / (250 days x 60 "
    "minutes x 6 hours x 0.5): the paper's labor share (0.6) and "
    "elasticity of labor supply (0.5)."
)
AGGREGATION_NOTE = (
    "the levels do not aggregate into each other: an ADM share is "
    "total disutility dollars over total unit GDP, the population x "
    "income weighted mean of member region shares, while the impact "
    "region product scales the population-weighted physical rate's "
    "underlying object. Comparing across levels compares different "
    "weightings."
)


def _metacsv(path: str) -> pd.DataFrame:
    lines = open(path).readlines()
    for i, line in enumerate(lines):
        if line.startswith(("year,", "hierid,")):
            break
    return pd.read_csv(io.StringIO("".join(lines[i:])))


def iso_income_series(ssp: str, iam: str) -> pd.DataFrame:
    """GDPpc per ISO and year, 2005 PPP, the provider's construction.

    Baseline at 2010 and five-yearly growth, with the hierarchical
    fallback: the named IAM's series for the ISO, else the ISO's
    median across IAMs, else the global median. Growth for year y
    applies the factor of the span containing y-1, matching
    impactcommon's HierarchicalGDPpcProvider.
    """
    base = _metacsv(GDPPC_BASELINE)
    growth = _metacsv(GDPPC_GROWTH)
    b_this = base[(base.model == iam) & (base.scenario == ssp)
                  & (base.year == STARTYEAR)].set_index("iso")["value"]
    b_any = (base[(base.scenario == ssp) & (base.year == STARTYEAR)]
             .groupby("iso")["value"].median())
    b_global = base[(base.scenario == ssp)
                    & (base.year == STARTYEAR)]["value"].median()
    g = growth.copy()
    g["yearindex"] = ((g.year - STARTYEAR) / 5).astype(int)
    g_this = g[(g.model == iam) & (g.scenario == ssp)].pivot_table(
        index="iso", columns="yearindex", values="growth")
    g_any = g[g.scenario == ssp].pivot_table(
        index="iso", columns="yearindex", values="growth", aggfunc="median")
    g_global = g[(g.scenario == ssp) & (g.model == iam)].groupby(
        "yearindex")["growth"].median()

    nl = _metacsv(NIGHTLIGHTS)
    isos = sorted(set(nl["iso"].astype(str)))
    years = list(range(STARTYEAR, 2100))
    out = np.empty((len(isos), len(years)))
    for i, iso in enumerate(isos):
        baseline = b_this.get(iso, b_any.get(iso, b_global))
        if iso in g_this.index:
            grow = g_this.loc[iso]
        elif iso in g_any.index:
            grow = g_any.loc[iso]
        else:
            grow = g_global
        series = [baseline]
        for year in range(STARTYEAR + 1, 2100):
            yearindex = int((year - 1 - STARTYEAR) / 5)
            rate = grow.get(yearindex, np.nan)
            if np.isnan(rate):
                rate = g_global.get(yearindex, 1.0)
            series.append(series[-1] * rate)
        out[i] = series
    return pd.DataFrame(out, index=pd.Index(isos, name="iso"), columns=years)


def ir_income(ssp: str, iam: str) -> pd.DataFrame:
    """GDPpc per hierid and year: the ISO series times the region's
    nightlights ratio, 0.8 times the ISO series where the ratio is
    missing or zero, as the provider does."""
    iso = iso_income_series(ssp, iam)
    nl = _metacsv(NIGHTLIGHTS)
    ratio = nl["gdppc_ratio"].where(
        nl["gdppc_ratio"].notna() & (nl["gdppc_ratio"] != 0), 0.8)
    inc = iso.loc[nl["iso"].astype(str)].to_numpy() * ratio.to_numpy()[:, None]
    return pd.DataFrame(inc, index=pd.Index(nl["hierid"], name="hierid"),
                        columns=iso.columns)


def load_population(ssp: str) -> pd.DataFrame:
    """The rate runners' population: merged spans held flat, plus the
    2011 UNPD-scaled historical fallback held constant (see
    run_physical_labor.historical_fallback for the full reasoning)."""
    rows = []
    with open(POP.format(ssp=ssp)) as f:
        for line in f:
            if not line.startswith("#"):
                break
        for row in csv.reader(f):
            rows.append((int(row[0]), row[1], float(row[2])))
    base = pd.DataFrame(rows, columns=["span", "hierid", "pop"])
    years = pd.DataFrame({"year": list(YEARS)})
    years["span"] = years["year"] - years["year"] % 5
    out = years.merge(base, on="span").drop(columns="span")
    import xarray as xr
    with xr.open_dataset(UNPD15) as ds:
        series = ds["population"].sel(year=FALLBACK_YEAR).to_series()
    held = series[~series.index.isin(set(base["hierid"]))]
    fb = years[["year"]].merge(
        held.rename("pop").rename_axis("hierid").reset_index(), how="cross")
    return pd.concat([out, fb[["year", "hierid", "pop"]]], ignore_index=True)


KEY_COLUMNS = {"hierid", "ISO", "ID_1", "ID_2",
               "rcp", "iam", "ssp", "window", "statistic"}


def value_column(frame: pd.DataFrame) -> str:
    """The single non-key column, inferred rather than assumed: the
    value column's name varies by product (the pipeline writes its
    configured value_col, the runners theirs), so take whatever is
    left after the keys, the way the results converter does."""
    candidates = [c for c in frame.columns if c not in KEY_COLUMNS]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one value column, found {candidates}")
    return candidates[0]


def window_bounds(label: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{4})", label)
    if m:
        return int(m.group(1)), int(m.group(2))
    year = int(label)
    return year, year


def unit_gdp(level: str, combos: list[tuple]) -> pd.DataFrame:
    """Window-mean unit GDP (2005 PPP) per (ssp, iam, window)."""
    keys = {"adm0": ["ISO"], "adm1": ["ISO", "ID_1"],
            "adm2": ["ISO", "ID_1", "ID_2"]}[level]
    if level != "adm0":
        portions = pd.read_parquet(WEIGHTS.format(adm=level))[
            keys + ["hierid", "popwt"]]
    frames = []
    cache: dict[tuple, pd.DataFrame] = {}
    for ssp, iam, window in combos:
        if (ssp, iam) not in cache:
            inc = ir_income(ssp, iam)
            pop = load_population(ssp)
            pop["income"] = inc.to_numpy()[
                inc.index.get_indexer(pop["hierid"]),
                inc.columns.get_indexer(pop["year"])]
            pop["gdp"] = pop["pop"] * pop["income"]
            cache[(ssp, iam)] = pop[["hierid", "year", "gdp"]]
        gdp = cache[(ssp, iam)]
        lo, hi = window_bounds(window)
        w = (gdp[gdp["year"].between(lo, hi)]
             .groupby("hierid", sort=False)["gdp"].mean().reset_index())
        if level == "adm0":
            w["ISO"] = w["hierid"].str[:3]
            den = w.groupby("ISO", sort=False)["gdp"].sum().reset_index()
        else:
            merged = portions.merge(w, on="hierid")
            merged["gdp"] = merged["gdp"] * merged["popwt"]
            den = merged.groupby(keys, sort=False)["gdp"].sum().reset_index()
        den["ssp"] = ssp
        den["iam"] = iam
        den["window"] = window
        frames.append(den)
    return pd.concat(frames, ignore_index=True)


def write_ir(rcp: str) -> None:
    source = IR_DIR / f"labor_physical_{rcp}.parquet"
    stats = pd.read_parquet(source)
    valcol = value_column(stats)
    stats["value"] = stats[valcol] * FACTOR
    if valcol != "value":
        stats = stats.drop(columns=valcol)
    out = IR_DIR / f"labor_valued_pctgdp_{rcp}.parquet"
    stats.to_parquet(out, index=False)
    manifest = {
        "source": str(source),
        "source_value_col": valcol,
        "construction": f"labor_physical times {FACTOR}",
        "identity": IDENTITY_NOTE,
        "aggregation_note": AGGREGATION_NOTE,
        "units": "share of the region's own GDP in that year, "
                 "dimensionless; both sides 2005 PPP, so no deflator "
                 "enters",
        "statistics": "inherited from labor_physical: a positive "
                      "constant commutes with the mean and every "
                      "quantile",
    }
    (IR_DIR / f"labor_valued_pctgdp_{rcp}.manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"wrote {out} rows={len(stats)}", flush=True)


def write_adm(level: str, rcp: str) -> None:
    source = ADM_DIR / level / f"labor_valued_{rcp}.parquet"
    stats = pd.read_parquet(source)
    valcol = value_column(stats)
    keys = [c for c in ("ISO", "ID_1", "ID_2") if c in stats.columns]
    combos = list(stats[["ssp", "iam", "window"]].drop_duplicates()
                  .itertuples(index=False, name=None))
    den = unit_gdp(level, combos)
    merged = stats.merge(den, on=keys + ["ssp", "iam", "window"], how="left")
    if len(merged) != len(stats):
        raise ValueError(f"{source}: denominator merge changed the row count")
    bad = merged["gdp"].isna() | (merged["gdp"] <= 0)
    kept = merged.loc[~bad].copy()
    kept["value"] = kept[valcol] / kept["gdp"]
    drop = ["gdp"] + ([valcol] if valcol != "value" else [])
    kept = kept.drop(columns=drop)
    out = ADM_DIR / level / f"labor_valued_pctgdp_{rcp}.parquet"
    kept.to_parquet(out, index=False)
    manifest = {
        "source": str(source),
        "source_value_col": valcol,
        "denominator": "unit GDP in 2005 PPP: merged SSP population "
                       "(2011 UNPD-scaled fallback for the 59 countries "
                       "the series lacks) times the projection system's "
                       "GDPpc series (merged baseline and growth through "
                       "the hierarchical provider logic, nightlights "
                       "ratio per region, global-median income for the "
                       "59 countries), aggregated with the same "
                       "population portions as the dollar numerators",
        "convention": "window mean dollars over window mean GDP",
        "identity": IDENTITY_NOTE,
        "aggregation_note": AGGREGATION_NOTE,
        "units": "share of unit GDP, dimensionless; numerator and "
                 "denominator both 2005 PPP, so no deflator enters",
        "denominator_caveat": "for the 59 countries absent from the SSP "
                              "series the numerator's embedded population "
                              "is LandScan 2011 while this denominator "
                              "uses the UNPD-scaled 2011 values; the "
                              "levels differ by a few percent there",
        "dropped_rows": int(bad.sum()),
    }
    (ADM_DIR / level / f"labor_valued_pctgdp_{rcp}.manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"wrote {out} rows={len(kept)} dropped={int(bad.sum())}", flush=True)


def main() -> int:
    for rcp in RCPS:
        write_ir(rcp)
        for level in ("adm0", "adm1", "adm2"):
            write_adm(level, rcp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
