"""Rebuild the example's data from the CIL filesystem and the Planetary Computer.

Runs on Midway3, on a compute node with outbound network. Everyone else
uses what is committed under data/.

It writes:

- The rows of the three production weight tables for the four
  provinces: grid cells to GADM 2.0 ADM1, to ADM2, and to impact
  regions whose hierid starts PER.<ID_1>. All three use the same 503
  cells, which this checks.
  Source: /project/cil/gcp/impact_map_inputs/weights/
- Daily tasmax and tasmin for three models, historical 2000 and ssp370
  2090, on those 503 cells, time by cell.
  Source: GDPCIR on the Microsoft Planetary Computer
- Monthly tavg for the same model-years, on the box around the four
  provinces, with the cells outside them blank.
  Source: /project/cil/gcp/climate/CMIP6_GDPCIR/monthly/tavg
- Days above 95F per province for all twenty ssp370 models, 2080-2099.
  Source: /project/cil/gcp/climate/CMIP6_GDPCIR/gadm20/daily
- Outlines of the provinces, their ADM2 units and their impact regions,
  simplified at 0.01 degrees.
  Source: /project/cil/gcp/impact_map_inputs/targets/gadm20/ and the
  world-combo-201710 impact region shapefile

It then checks the daily files against production: for each stored
cell, the days above 95F and the annual mean that derive_daily gives
must equal the per-cell fields production wrote for the same model and
year under /project/cil/gcp/climate/CMIP6_GDPCIR/grid/annual.
"""
from __future__ import annotations

import glob
import hashlib
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from cil_regionalization.bcsd_io import derive_daily
from cil_regionalization.gdpcir_io import gdpcir_stac_assets, open_gdpcir_daily

import data

WEIGHTS_SOURCE = Path("/project/cil/gcp/impact_map_inputs/weights")
TARGETS = Path("/project/cil/gcp/impact_map_inputs/targets/gadm20")
IR_SHAPES = Path("/project/cil/gcp/regions/world-combo-201710/agglomerated-world-new.shp")
MONTHLY = Path("/project/cil/gcp/climate/CMIP6_GDPCIR/monthly/tavg")
REGIONAL = Path("/project/cil/gcp/climate/CMIP6_GDPCIR/gadm20/daily")
GRIDS = Path("/project/cil/gcp/climate/CMIP6_GDPCIR/grid/annual")
KEY = ["cell_ix", "cell_iy"]
IDS = list(data.PROVINCES)
IR_PREFIXES = tuple(f"PER.{i}." for i in IDS)
SIMPLIFY_DEG = 0.01
WINDOW = {"historical": "1986-2005", "ssp370": "2080-2099"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut_weights() -> pd.DataFrame:
    """The four provinces' rows of each table; returns the cells they use."""
    cells = {}
    for level, out in data.WEIGHTS.items():
        src = WEIGHTS_SOURCE / out.name
        if level == "impact_region":
            full = pd.read_parquet(src / "weights.parquet")
            rows = full[full["hierid"].str.startswith(IR_PREFIXES)]
            ids, query = ["hierid"], f"hierid starts with one of {list(IR_PREFIXES)}"
        else:
            full = pd.read_parquet(src / "weights.parquet",
                                   filters=[("ISO", "==", "PER")])
            rows = full[full["ID_1"].isin(IDS)]
            ids = ["ISO", "ID_1"] + (["ID_2"] if level == "adm2" else [])
            query = f"ISO == 'PER' and ID_1 in {IDS}"
        rows = rows.reset_index(drop=True)
        for w in ("popwt", "areawt"):
            sums = rows.groupby(ids)[w].sum()
            assert (sums - 1).abs().max() < 1e-9, f"{level}: the cut breaks {w} sums"
        out.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(out / "weights.parquet", index=False)
        manifest = json.loads((src / "weights.manifest.json").read_text())
        manifest["outputs"] = {"weights.parquet": _sha256(out / "weights.parquet")}
        manifest["row_counts"] = {
            "regions": int(rows[ids].drop_duplicates().shape[0]),
            "segments": len(rows),
            "total": len(rows),
        }
        manifest["crop"] = {
            "query": query,
            "source_artifact": manifest["project_name"],
            "source_sha256": _sha256(src / "weights.parquet"),
        }
        (out / "weights.manifest.json").write_text(json.dumps(manifest, indent=1))
        cells[level] = rows[KEY + ["cell_lat", "cell_lon"]].drop_duplicates()
        print(f"weights, {level}: {len(rows)} rows, "
              f"{rows[ids].drop_duplicates().shape[0]} units, {len(cells[level])} cells")
    sets = {k: set(map(tuple, v[KEY].to_numpy())) for k, v in cells.items()}
    assert sets["adm1"] == sets["adm2"] == sets["impact_region"], \
        "the three tables use different cells"
    return cells["adm1"].sort_values(KEY).reset_index(drop=True)


def _index(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(coord, values)
    assert np.allclose(coord[idx], values, atol=1e-6), "a cell centre is off the grid"
    return idx


def write_daily(model: str, experiment: str, cells: pd.DataFrame) -> dict:
    """Cut one model-year to the cells and write it; returns the arrays."""
    year = data.YEARS[experiment]
    stores = gdpcir_stac_assets(model, experiment, ("tasmax", "tasmin"),
                                collections=[data.MODELS[model]["collection"]])
    arrays, flat = {}, None
    for var, (href, kw) in stores.items():
        ds = open_gdpcir_daily(href, var, **kw)
        lat, lon = ds["lat"].values, ds["lon"].values
        iy = _index(lat, cells["cell_lat"].to_numpy())
        ix = _index(lon, cells["cell_lon"].to_numpy())
        box = (ds[var].isel(lat=slice(iy.min(), iy.max() + 1),
                            lon=slice(ix.min(), ix.max() + 1))
               .sel(time=slice(f"{year}-01-01", f"{year}-12-31")))
        block = np.asarray(box.transpose("time", "lat", "lon").values, dtype="float32")
        assert block.shape[0] == 365, f"{model} {year} {var}: {block.shape[0]} days"
        vals = np.ascontiguousarray(block[:, iy - iy.min(), ix - ix.min()])
        assert np.isfinite(vals).all(), f"{model} {year} {var}: a weighted cell is blank"
        arrays[var] = vals
        flat = iy * len(lon) + ix
        units = ds[var].attrs.get("units", "K")
    path = data.daily_path(model, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as nc:
        nc.title = f"GDPCIR daily tasmax and tasmin, {model} {experiment} {year}"
        nc.source = ("Rhodium Group / Climate Impact Lab GDPCIR, read from the "
                     f"Microsoft Planetary Computer collection {data.MODELS[model]['collection']}")
        nc.license = data.MODELS[model]["license"]
        nc.model, nc.experiment, nc.year = model, experiment, year
        nc.grid = data.GRID
        nc.comment = ("only the cells the weights of Ucayali, Madre de Dios, Piura "
                      "and Cusco use, one column per cell; cell_ix and cell_iy "
                      "are the cell's indices on the grid")
        nc.createDimension("time", 365)
        nc.createDimension("cell", len(cells))
        t = nc.createVariable("time", "i4", ("time",))
        t.units, t.calendar = f"days since {year}-01-01", "noleap"
        t[:] = np.arange(365)
        for name, dtype in (("cell_ix", "i4"), ("cell_iy", "i4"),
                            ("cell_lon", "f8"), ("cell_lat", "f8")):
            v = nc.createVariable(name, dtype, ("cell",))
            v[:] = cells[name].to_numpy()
        for var, vals in arrays.items():
            v = nc.createVariable(var, "f4", ("time", "cell"), zlib=True,
                                  complevel=9, shuffle=True)
            v.units = units
            v[:] = vals
    print(f"daily {model} {experiment} {year}: {path.stat().st_size / 1e6:.2f} MB")
    return {"tasmax": arrays["tasmax"], "tasmin": arrays["tasmin"], "flat": flat}


def write_monthly(model: str, experiment: str, cells: pd.DataFrame) -> None:
    """Twelve monthly means on the box, the cells outside the provinces blank."""
    year = data.YEARS[experiment]
    family = "CMIP" if experiment == "historical" else "ScenarioMIP"
    (store,) = glob.glob(str(MONTHLY / family / "*" / model / experiment / "*" / "v1.1.zarr"))
    ds = xr.open_zarr(store)
    sub = ds["tavg"].sel(lat=slice(cells["cell_lat"].min(), cells["cell_lat"].max()),
                         lon=slice(cells["cell_lon"].min(), cells["cell_lon"].max()),
                         time=slice(f"{year}-01-01", f"{year}-12-31"))
    block = np.asarray(sub.transpose("time", "lat", "lon").values, dtype="float32")
    assert block.shape[0] == 12, f"{model} {year}: {block.shape[0]} months"
    lat, lon = sub["lat"].values, sub["lon"].values
    keep = np.zeros(block.shape[1:], dtype=bool)
    keep[_index(lat, cells["cell_lat"].to_numpy()),
         _index(lon, cells["cell_lon"].to_numpy())] = True
    assert np.isfinite(block[:, keep]).all(), f"{model} {year}: a weighted cell is blank"
    block[:, ~keep] = np.nan
    path = data.monthly_path(model, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as nc:
        nc.title = f"GDPCIR monthly tavg, {model} {experiment} {year}"
        nc.source = ("CIL monthly means of GDPCIR daily (tasmax + tasmin) / 2, "
                     f"{store}")
        nc.license = data.MODELS[model]["license"]
        nc.model, nc.experiment, nc.year = model, experiment, year
        nc.comment = "cells outside Ucayali, Madre de Dios, Piura and Cusco are blank"
        nc.createDimension("month", 12)
        nc.createDimension("lat", len(lat))
        nc.createDimension("lon", len(lon))
        m = nc.createVariable("month", "i4", ("month",))
        m[:] = np.arange(1, 13)
        for name, vals in (("lat", lat), ("lon", lon)):
            v = nc.createVariable(name, "f8", (name,))
            v[:] = vals
        v = nc.createVariable("tavg", "f4", ("month", "lat", "lon"), zlib=True,
                              complevel=9, shuffle=True, fill_value=np.float32(1e20))
        v.units = ds["tavg"].attrs["units"]
        v.long_name = ds["tavg"].attrs["long_name"]
        v[:] = np.ma.masked_invalid(block)
    print(f"monthly {model} {experiment} {year}: {path.stat().st_size / 1e6:.2f} MB")


def check_against_production(model: str, experiment: str, got: dict) -> None:
    """The stored days give the same per-cell fields production wrote."""
    year = data.YEARS[experiment]
    d = derive_daily(got["tasmax"][:, None, :], got["tasmin"][:, None, :], None)
    path = GRIDS / f"grid_{model}_{experiment}_{WINDOW[experiment]}.npz"
    with zipfile.ZipFile(path) as z:
        years = np.load(z.open("years.npy"))
        i = int(np.where(years == year)[0][0])
        hot = np.load(z.open("days_above_95F.npy"))[i][got["flat"]]
        tas = np.load(z.open("tas_annual.npy"))[i][got["flat"]]
    ours_hot = d.days_above_95F.ravel()
    ours_tas = (d.tas_annual * 9.0 / 5.0 + 32.0).astype("float32").ravel()
    assert np.array_equal(ours_hot, hot), \
        f"{model} {year}: days above 95F differ in {int((ours_hot != hot).sum())} cells"
    # production averaged the whole grid at once, so the float64 sum may
    # run in another order; the stored float32 can then differ in its
    # last bit
    diff = float(np.abs(ours_tas.astype("float64") - tas).max())
    assert diff < 1e-4, f"{model} {year}: annual mean differs by up to {diff} F"
    print(f"check {model} {year}: days above 95F equal production in all "
          f"{len(hot)} cells; annual mean within {diff:.1e} F")


def write_summary() -> None:
    rows = []
    for f in sorted(REGIONAL.glob("days_*_ssp370.parquet")):
        d = pd.read_parquet(f, columns=["model", "level", "weight", "variable",
                                        "window", "region", "value"],
                            filters=[("level", "==", "adm1"), ("weight", "==", "pop"),
                                     ("variable", "==", "days_above_95F"),
                                     ("window", "==", "2080-2099")])
        rows.append(d[d["region"].isin([f"PER|{i}" for i in IDS])])
    t = pd.concat(rows, ignore_index=True)
    t["ID_1"] = t["region"].str.split("|").str[1].astype(int)
    t["province"] = t["ID_1"].map(data.PROVINCES)
    t = t.rename(columns={"value": "days_above_95F"})[
        ["model", "ID_1", "province", "days_above_95F"]]
    assert t["model"].nunique() == 20 and len(t) == 80, t.groupby("model").size()
    t.sort_values(["ID_1", "days_above_95F"]).to_csv(data.SUMMARY, index=False)
    print(f"summary: {t['model'].nunique()} models x {t['ID_1'].nunique()} provinces")


def write_outlines() -> None:
    for level in ("adm1", "adm2"):
        cols = ["ISO", "ID_1", "NAME_1", "geometry"] + (
            ["ID_2", "NAME_2"] if level == "adm2" else [])
        shapes = gpd.read_parquet(TARGETS / f"{level}.parquet", columns=cols,
                                  filters=[("ISO", "==", "PER")])
        shapes = shapes[shapes["ID_1"].isin(IDS)].reset_index(drop=True)
        ids = ["ISO", "ID_1"] + (["ID_2"] if level == "adm2" else [])
        units = pd.read_parquet(data.WEIGHTS[level] / "weights.parquet",
                                columns=ids).drop_duplicates()
        assert set(map(tuple, shapes[ids].to_numpy())) == \
            set(map(tuple, units.to_numpy())), f"{level}: outlines and weights disagree"
        shapes["geometry"] = shapes.geometry.simplify(SIMPLIFY_DEG)
        shapes.to_parquet(data.OUTLINES[level])
    shapes = gpd.read_file(IR_SHAPES)
    if shapes.crs is None:
        shapes = shapes.set_crs("EPSG:4326")
    shapes = shapes[shapes["hierid"].str.startswith(IR_PREFIXES)][["hierid", "geometry"]]
    regions = set(pd.read_parquet(data.WEIGHTS["impact_region"] / "weights.parquet",
                                  columns=["hierid"])["hierid"])
    assert set(shapes["hierid"]) == regions, "impact region outlines and weights disagree"
    shapes["geometry"] = shapes.geometry.simplify(SIMPLIFY_DEG)
    assert not shapes.geometry.is_empty.any(), "simplifying emptied a region"
    shapes.reset_index(drop=True).to_parquet(data.OUTLINES["impact_region"])
    for level, p in data.OUTLINES.items():
        print(f"outlines, {level}: {p.stat().st_size / 1e6:.2f} MB")


def main() -> int:
    data.DATA.mkdir(exist_ok=True)
    cells = cut_weights()
    for model in data.MODELS:
        for experiment in data.YEARS:
            got = write_daily(model, experiment, cells)
            check_against_production(model, experiment, got)
            write_monthly(model, experiment, cells)
    write_summary()
    write_outlines()
    total = sum(p.stat().st_size for p in data.DATA.rglob("*") if p.is_file())
    print(f"data/: {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
