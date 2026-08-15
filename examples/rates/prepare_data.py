"""Rebuild the committed data extract from the projection tree.

Needs access to the Climate Impact Lab filesystem; everyone else reads
the committed output under data/. The extract holds three Monte Carlo
batches with all 33 climate models (rcp85, SSP3, low iam), years 2080
to 2099, for Mexico's 500 impact regions plus 30 border regions that
overlap Mexican municipalities under GADM 4.1. For every draw both the
full adaptation file and its histclim counterpart are kept, because
the effect of climate change is their difference.

The population is recovered from the tree itself: each leaf has a
levels sibling holding rate times the projection's own population, so
dividing levels by rates returns that population exactly. The ratio is
identical across climate models, batches, and the histclim variant to
float precision, which confirms it carries no climate signal; the
median over 22 rate and histclim sources smooths float noise where a
rate crosses zero. Two impact regions (MEX.25.1309 and MEX.23.1234,
together fewer than 150 people) hold no values anywhere in this tree
and are dropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TREE = Path("/project/cil/gcp/outputs/mortality_new-socioeconomics/impacts-darwin/montecarlo")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BATCHES = ["batch0", "batch1", "batch2"]
YEARS = slice(2080, 2099)
COMBINED = "Agespec_interaction_response-combined.nc4"
HISTCLIM = "Agespec_interaction_response-combined-histclim.nc4"


def region_set() -> list[str]:
    """Mexico plus the border regions: every source region of the
    GADM 4.1 per_source slice, which carries all regions overlapping
    Mexican municipalities."""
    w = pd.read_parquet(DATA / "weights" / "gadm41_adm2_per_source" / "weights.parquet")
    return sorted(set(w["hierid"]))


def slice_leaf(src: Path, dst: Path, keep: list[str]) -> None:
    with xr.open_dataset(src) as ds:
        regs = [str(r) for r in ds["regions"].values]
        idx = [regs.index(r) for r in keep]
        sub = ds[["rebased", "regions"]].isel(region=idx).sel(year=YEARS)
        sub.to_netcdf(dst, encoding={"rebased": {"zlib": True, "complevel": 4}})


def implied_population(gcm: str, filename: str, keep: list[str]):
    leaf = TREE / "batch0" / "rcp85" / gcm / "low" / "SSP3" / filename
    levels = leaf.with_name(leaf.name.replace(".nc4", "-levels.nc4"))
    with xr.open_dataset(leaf) as r, xr.open_dataset(levels) as l:
        regs = [str(x) for x in r["regions"].values]
        idx = [regs.index(x) for x in keep]
        rate = r["rebased"].isel(region=idx).sel(year=YEARS).values
        lev = l["rebased"].isel(region=idx).sel(year=YEARS).values
        years = [int(y) for y in r["year"].sel(year=YEARS).values]
    return np.where(np.abs(rate) > 1e-9, lev / rate, np.nan), years


def main() -> int:
    keep = region_set()
    gcms = sorted(p.name for p in (TREE / "batch0" / "rcp85").iterdir() if p.is_dir())
    for batch in BATCHES:
        for gcm in gcms:
            src = TREE / batch / "rcp85" / gcm / "low" / "SSP3"
            dst = DATA / "montecarlo" / batch / "rcp85" / gcm / "low" / "SSP3"
            dst.mkdir(parents=True, exist_ok=True)
            slice_leaf(src / COMBINED, dst / COMBINED, keep)
            slice_leaf(src / HISTCLIM, dst / HISTCLIM, keep)
            print(batch, gcm, flush=True)

    stack = []
    for gcm in gcms[::3]:
        for filename in (COMBINED, HISTCLIM):
            arr, years = implied_population(gcm, filename, keep)
            stack.append(arr)
    stack = np.stack(stack)
    pop = np.nanmedian(stack, axis=0)
    spread = np.nanmax(np.nanmax(np.abs(stack - pop[None]), axis=0) / np.abs(pop))
    assert spread < 1e-5, f"population differs across sources: {spread:.2e}"

    popdf = pd.DataFrame(pop.T, index=keep, columns=years)
    dropped = popdf.index[popdf.isna().any(axis=1)].tolist()
    print("regions with no values in the tree, dropped:", dropped)
    popdf = popdf.drop(index=dropped)
    assert popdf.notna().all().all() and (popdf > 0).all().all()
    out = popdf.reset_index(names="hierid").melt(
        id_vars="hierid", var_name="year", value_name="population")
    out["year"] = out["year"].astype(int)
    out.to_parquet(DATA / "population.parquet", index=False)
    print(f"wrote {len(out)} population rows for {popdf.shape[0]} regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
