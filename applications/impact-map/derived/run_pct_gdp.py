"""Percent-of-GDP twins for the valued mortality and agriculture products.

The published tables report monetized damages as a share of GDP; our
valued products carry dollars. This runner divides them by the GDP the
lab's own valuations use: ``integration-econ-bc39.zarr``, the dscim
economic inputs, which is agval's configured GDP input and the
denominator behind the published agriculture percent-of-GDP files
(agval/main/3_get_pct_gdp.ipynb). GDP is a constant per (region, year,
ssp, iam), so dividing the published-method statistics equals dividing
the draws before pooling; no statistics ordering issue arises.

Convention: window mean dollars over window mean GDP. Each window
label in a product resolves to its year range (a bare year is that
year alone), and the denominator is the mean of annual GDP over those
years, at the product's own ssp with the iam mapped to the zarr's
model names (low is IIASA GDP, high is OECD Env-Growth).

Geographies: hierid products divide by the region's own GDP; ADM0 by
the ISO sum (exact, since impact regions do not cross national
borders); ADM1 and ADM2 by the population-portioned sum of member
region GDP through the same weights artifacts that aggregated the
numerators.

Labor is deliberately absent: its valued product is in 2005 PPP
dollars while this GDP series is at the 2019-dollar level, and
dividing across vintages would mislabel the result by the price-level
ratio. A labor share of GDP needs a deflator decision first; the
manifests say so, so the absence is not mistaken for an oversight.

Rows whose denominator is zero (uninhabited regions) are dropped with
the count recorded. The Caspian Sea region ``CA-`` inherits a phantom
population of 15.5 million from the LandScan lineage of the zarr, so
its denominator is not meaningful; it is dropped too and noted.

Run on the cluster:

    python run_pct_gdp.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import xarray as xr

ZARR = ("/project/cil/gcp/integration/float32/dscim_input_data/econvars/"
        "zarrs/integration-econ-bc39.zarr")
WEIGHTS = ("/project/cil/home_dirs/scadavidsanchez/repos/"
           "climate-and-damages-aggregations/data/weights/ir_{adm}_full/"
           "weights.parquet")
BASE = Path("/project/cil/gcp/impact_map")
MODEL = {"low": "IIASA GDP", "high": "OECD Env-Growth"}
RCPS = ["rcp45", "rcp85"]
# sector name, file stem, dollar vintage, level -> directory
PRODUCTS = [
    ("mortality", "mortality_valued", "2019 USD", {
        "ir": BASE / "impact_region" / "mortality",
        "adm0": BASE / "mortality" / "adm0",
        "adm1": BASE / "mortality" / "adm1",
        "adm2": BASE / "mortality" / "adm2",
    }),
    ("agriculture", "agriculture_valued", "2019 USD", {
        "ir": BASE / "impact_region" / "agriculture" / "all_crops",
        "adm0": BASE / "agriculture" / "all_crops" / "adm0",
        "adm1": BASE / "agriculture" / "all_crops" / "adm1",
        "adm2": BASE / "agriculture" / "all_crops" / "adm2",
    }),
]
LABOR_NOTE = ("labor's percent-of-GDP twin is not produced here because "
              "its dollars are 2005 PPP and this GDP series is at the "
              "2019-dollar level; dividing across vintages would be off "
              "by the price-level ratio. The twin exists separately "
              "(labor_valued_pctgdp, run_labor_pctgdp.py), built at "
              "matching 2005 PPP vintage from the projection system's "
              "own income series.")
AG_INCIDENCE_NOTE = (
    "a region's welfare change is an allocation of a market-level "
    "result, not a local outcome. The market clears at trading-partner "
    "level; consumer surplus is distributed to regions by GDP share and "
    "producer surplus by planted-area share, so a region with no yield "
    "change still carries welfare change through the group price. "
    "Dividing by regional GDP cancels the GDP-share term, so this "
    "share-of-GDP map is essentially constant within each trading "
    "group: measured across slices, 99.9 percent of regions sit within "
    "10 percent of their group median and group membership explains "
    "over 0.998 of the cross-region variance. A flat patch inside a "
    "group is the construction, not a null result; differences between "
    "groups are the real market differences. The physical yield and "
    "calorie products carry the local signal.")


def window_bounds(label: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{4})", label)
    if m:
        return int(m.group(1)), int(m.group(2))
    year = int(label)
    return year, year


def ir_gdp(zarr: xr.Dataset, ssp: str, iam: str,
           lo: int, hi: int) -> pd.Series:
    """Window-mean GDP per hierid for one (ssp, iam, window)."""
    g = (zarr["gdp"].sel(ssp=ssp, model=MODEL[iam], year=slice(lo, hi))
         .mean("year").astype("float64"))
    return g.to_series().rename("gdp").rename_axis("hierid")


def denominators(zarr: xr.Dataset, level: str, combos: list[tuple],
                 keys: list[str]) -> pd.DataFrame:
    """Window-mean GDP per target for every (ssp, iam, window) combo.

    hierid passes through; ADM0 sums by the ISO prefix, exact because
    impact regions stay within one country; ADM1 and ADM2 sum the
    population-portioned member GDP through the weights artifacts, the
    same mapping the dollar numerators were aggregated with.
    """
    if level in ("adm1", "adm2"):
        portions = pd.read_parquet(WEIGHTS.format(adm=level))[
            keys + ["hierid", "popwt"]]
    frames = []
    for ssp, iam, window in combos:
        lo, hi = window_bounds(window)
        gdp = ir_gdp(zarr, ssp, iam, lo, hi).reset_index()
        if level == "ir":
            den = gdp
        elif level == "adm0":
            den = (gdp.assign(ISO=gdp["hierid"].str[:3])
                   .groupby("ISO", sort=False)["gdp"].sum().reset_index())
        else:
            merged = portions.merge(gdp, on="hierid")
            merged["gdp"] = merged["gdp"] * merged["popwt"]
            den = (merged.groupby(keys, sort=False)["gdp"]
                   .sum().reset_index())
        den["ssp"] = ssp
        den["iam"] = iam
        den["window"] = window
        frames.append(den)
    return pd.concat(frames, ignore_index=True)


def convert(zarr: xr.Dataset, sector: str, stem: str, vintage: str,
            level: str, directory: Path, rcp: str) -> str | None:
    source = directory / f"{stem}_{rcp}.parquet"
    if not source.exists():
        print(f"skip (no input): {source}", flush=True)
        return None
    stats = pd.read_parquet(source)
    keys = [c for c in ("hierid", "ISO", "ID_1", "ID_2") if c in stats.columns]
    combos = (stats[["ssp", "iam", "window"]].drop_duplicates()
              .itertuples(index=False, name=None))
    den = denominators(zarr, level, list(combos), keys)

    merged = stats.merge(den, on=keys + ["ssp", "iam", "window"], how="left")
    if len(merged) != len(stats):
        raise ValueError(f"{source}: denominator merge changed the row count")
    bad = (merged["gdp"].isna() | (merged["gdp"] <= 0)
           | (merged.get("hierid", pd.Series(index=merged.index)) == "CA-"))
    kept = merged.loc[~bad].copy()
    kept["value"] = kept["value"] / kept["gdp"]
    kept = kept.drop(columns="gdp")

    out = directory / f"{stem}_pctgdp_{rcp}.parquet"
    kept.to_parquet(out, index=False)
    manifest = {
        "source": str(source),
        "denominator": ZARR,
        "denominator_mapping": MODEL,
        "convention": "window mean dollars over window mean GDP: each "
                      "statistic is divided by the mean of annual GDP "
                      "over the window's years, at the row's ssp and iam",
        "units": f"share of GDP (dimensionless); numerator was {vintage} "
                 f"and the GDP series is at the same price level",
        "dropped_rows": int(bad.sum()),
        "dropped_note": "rows with zero or missing GDP (uninhabited "
                        "regions) and the Caspian Sea region CA-, whose "
                        "denominator is a LandScan artifact (15.5 million "
                        "phantom people), carry no meaningful share",
        "labor_note": LABOR_NOTE,
    }
    if sector == "agriculture":
        manifest["interpretation"] = AG_INCIDENCE_NOTE
    (directory / f"{stem}_pctgdp_{rcp}.manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"wrote {out} rows={len(kept)} dropped={int(bad.sum())}", flush=True)
    return str(out)


def main() -> int:
    zarr = xr.open_zarr(ZARR)
    written = []
    for sector, stem, vintage, levels in PRODUCTS:
        for level, directory in levels.items():
            for rcp in RCPS:
                path = convert(zarr, sector, stem, vintage, level,
                               directory, rcp)
                if path:
                    written.append(path)
    print(f"{len(written)} products written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
