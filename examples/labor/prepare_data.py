"""Rebuild the committed data extract from the labor projection tree.

Needs access to the Climate Impact Lab filesystem; everyone else reads
the committed output under data/. The extract holds one Monte Carlo
batch with all 33 climate models (rcp85, SSP3, low iam), years 2090 to
2099, for the 2,301 impact regions that overlap Indian states: 2,300
Indian regions plus one Nepalese border region. Each draw keeps the
physical file and its histclim counterpart.

The wage files are not committed. In the tree, the wage value is the
physical value times one factor per region and year (population times
income times 0.00487), so the factor is enough to rebuild them. This
script recovers the factor from the tree, checks the reconstruction
against a real wage pair, and commits the factor plus one wage pair
(CCSM4) so the notebook can show the same check.

The population is the projection's own input: the merged SSP series,
held constant within each five-year span, which is how the projection
reads it. A few uninhabited regions carry an exact zero and keep it.

The weight slice keeps every row of each included source region, so
shares still sum to one per region. The display layer dissolves the
impact region shapes by the state each region mostly falls in; it is
for the figures, not a boundary dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import cil_regionalization as cilreg

TREE = Path("/project/cil/gcp/outputs/labor/impacts-woodwork/montecarlo/uninteracted_main_model_27_37_39")
POPULATION = Path("/project/cil/gcp/social/baselines/population/merged/population-merged.SSP3.csv")
IR_SHAPES = Path("/project/cil/gcp/regions/world-combo-201710/agglomerated-world-new.shp")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TARGETS = HERE.parents[1] / "data" / "targets" / "gadm20" / "adm1.parquet"
BATCHES = ["batch0"]
YEARS = slice(2090, 2099)
STEM = "uninteracted_main_model"
WAGE_PAIR_GCM = "CCSM4"


def _mounted(path: Path) -> Path:
    if path.exists():
        return path
    alt = Path(str(path).replace("/project/cil", "/Volumes/cil"))
    return alt if alt.exists() else path


def slice_weights() -> list[str]:
    """The India slice of the published ADM1 weights: every row of
    every source region that touches an Indian state."""
    from cil_regionalization.fetch import default_cache_dir, load_registry

    name = "gadm20-adm1-per-source"
    full = cilreg.fetch_weights(name)
    frame = full.frame
    keep = sorted(set(frame[frame["ISO"] == "IND"]["hierid"]))
    rows = frame[frame["hierid"].isin(keep)]
    sums = rows.groupby("hierid")["popwt"].sum()
    assert (sums - 1).abs().max() < 1e-9, "slice breaks per-source sums"

    out = DATA / "weights" / "gadm20_adm1_per_source"
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "weights.parquet", index=False)
    record = load_registry()[name].record_id
    cached = default_cache_dir() / f"record-{record}" / name
    manifest = json.loads((cached / "weights.manifest.json").read_text())
    manifest["outputs"] = {
        "weights.parquet": hashlib.sha256(
            (out / "weights.parquet").read_bytes()
        ).hexdigest()
    }
    manifest["row_counts"] = {
        "regions": int(rows[["ISO", "ID_1"]].drop_duplicates().shape[0]),
        "source_units": len(keep),
        "total": len(rows),
    }
    (out / "weights.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"weights: {len(rows)} rows, {len(keep)} source regions")
    return keep


def _read(path: Path, keep: list[str]):
    with xr.open_dataset(path) as ds:
        regs = [str(r) for r in ds["regions"].values]
        idx = [regs.index(r) for r in keep]
        sub = ds["rebased"].isel(region=idx).sel(year=YEARS)
        return sub.values, [int(y) for y in sub["year"].values]


def slice_leaf(src: Path, dst: Path, keep: list[str]) -> None:
    with xr.open_dataset(src) as ds:
        regs = [str(r) for r in ds["regions"].values]
        idx = [regs.index(r) for r in keep]
        sub = ds[["rebased", "regions"]].isel(region=idx).sel(year=YEARS)
        sub.to_netcdf(dst, encoding={"rebased": {"zlib": True, "complevel": 4}})


def build_leaves(keep: list[str]) -> None:
    tree = _mounted(TREE)
    gcms = sorted(p.name for p in (tree / "batch0" / "rcp85").iterdir() if p.is_dir())
    for batch in BATCHES:
        for gcm in gcms:
            src = tree / batch / "rcp85" / gcm / "low" / "SSP3"
            dst = DATA / "montecarlo" / batch / "rcp85" / gcm / "low" / "SSP3"
            dst.mkdir(parents=True, exist_ok=True)
            names = [f"{STEM}.nc4", f"{STEM}-histclim.nc4"]
            if gcm == WAGE_PAIR_GCM:
                names += [f"{STEM}-wage-levels.nc4", f"{STEM}-histclim-wage-levels.nc4"]
            for name in names:
                slice_leaf(src / name, dst / name, keep)
            print(batch, gcm, flush=True)


def build_factor(keep: list[str]) -> None:
    """The wage factor per region and year, recovered as wage value
    over physical value and checked against a draw the recovery did
    not use."""
    tree = _mounted(TREE)
    gcms = sorted(p.name for p in (tree / "batch0" / "rcp85").iterdir() if p.is_dir())
    stack = []
    for gcm in gcms[::6]:
        for sfx in ("", "-histclim"):
            base = tree / "batch0" / "rcp85" / gcm / "low" / "SSP3"
            phys, years = _read(base / f"{STEM}{sfx}.nc4", keep)
            wage, _ = _read(base / f"{STEM}{sfx}-wage-levels.nc4", keep)
            stack.append(np.where(np.abs(phys) > 1e-9, wage / phys, np.nan))
    stack = np.stack(stack)
    factor = np.nanmedian(stack, axis=0)
    spread = np.nanmax(np.nanmax(np.abs(stack - factor[None]), axis=0) / np.abs(factor))
    assert spread < 1e-5, f"factor differs across draws: {spread:.2e}"
    df = pd.DataFrame(factor.T, index=keep, columns=years)
    holes = int(df.isna().sum().sum())
    if holes:
        df = df.interpolate(axis=1, limit_direction="both")
    assert df.notna().all().all()
    long = df.reset_index(names="hierid").melt(
        id_vars="hierid", var_name="year", value_name="factor"
    )
    long["year"] = long["year"].astype(int)
    long.to_parquet(DATA / "wage_factor.parquet", index=False)

    # check against a draw not used in the recovery
    base = tree / "batch0" / "rcp85" / "MIROC5" / "low" / "SSP3"
    phys, _ = _read(base / f"{STEM}.nc4", keep)
    hist, _ = _read(base / f"{STEM}-histclim.nc4", keep)
    wage, _ = _read(base / f"{STEM}-wage-levels.nc4", keep)
    wagehist, _ = _read(base / f"{STEM}-histclim-wage-levels.nc4", keep)
    recon = (phys - hist) * factor
    real = wage - wagehist
    scale = np.nanmax(np.abs(real))
    err = np.nanmax(np.abs(recon - real)) / scale
    print(f"factor: {holes} holes interpolated; reconstruction error {err:.1e}")
    assert err < 1e-6


def build_population(keep: list[str]) -> None:
    pop = pd.read_csv(_mounted(POPULATION), comment="#")
    pop = pop[pop["region"].isin(keep)]
    missing = set(keep) - set(pop["region"])
    assert not missing, f"regions absent from the SSP series: {sorted(missing)}"
    grid = pop.pivot(index="region", columns="year", values="value")
    years = list(range(YEARS.start, YEARS.stop + 1))
    out = pd.DataFrame(
        {y: grid[5 * (y // 5)] for y in years}
    ).reindex(keep)
    assert out.notna().all().all() and (out >= 0).all().all()
    long = out.reset_index(names="hierid").melt(
        id_vars="hierid", var_name="year", value_name="population"
    )
    long["year"] = long["year"].astype(int)
    long.to_parquet(DATA / "population.parquet", index=False)
    print(f"population: {len(long)} rows for {len(keep)} regions")


def build_display_layer(keep: list[str]) -> None:
    w = pd.read_parquet(DATA / "weights" / "gadm20_adm1_per_source" / "weights.parquet")
    best = (
        w.sort_values("popwt", ascending=False)
        .drop_duplicates("hierid")
        .query("ISO == 'IND'")[["hierid", "ISO", "ID_1"]]
    )
    shapes = gpd.read_file(_mounted(IR_SHAPES))
    shapes = shapes[shapes["hierid"].isin(set(best["hierid"]))]
    shapes = shapes[["hierid", "geometry"]].merge(best, on="hierid")
    states = shapes.dissolve(by=["ISO", "ID_1"], as_index=False)[
        ["ISO", "ID_1", "geometry"]
    ]
    names = pd.read_parquet(TARGETS, columns=["ISO", "ID_1", "NAME_1"])
    states = states.merge(names, on=["ISO", "ID_1"])
    states["geometry"] = states.geometry.simplify(0.005)
    states.to_parquet(DATA / "ind_adm1_plot.parquet")
    print(f"display layer: {len(states)} states")


def main() -> int:
    keep = slice_weights()
    build_population(keep)
    build_display_layer(keep)
    build_factor(keep)
    build_leaves(keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
