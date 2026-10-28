"""Rebuild the committed data extract for the degree-day climate example.

Needs access to the Climate Impact Lab filesystem; everyone else reads
the committed output under data/. Two countries with opposite
climates are cropped, Indonesia and Germany, and each crop is checked:

- One year of daily tasmax and tasmin from the raw NASA BCSD archive,
  per country, over the bounding box of the cells that country's ADM1
  units draw on. The model is MRI-CGCM3 and the year 2020. MRI-CGCM3
  is one of the seven rcp85 models on a Gregorian calendar, so its
  leap years carry 366 days, and its calendar is spelled "standard",
  the CF default; 2020 is a leap year and the first year of the 2020
  to 2039 window the lab's climate products report. The crop keeps
  the raw files' encoding (time units and calendar, longitudes in
  [0,360), float32, the 1e20 fill value), so the example reads it
  exactly as it would read the archive. Cells the country's units do
  not draw on are written as missing, which keeps the files small and
  changes no value the aggregation reads.
- The grid025 GADM 2.0 ADM1 population and area weights, cropped to
  the two countries' units, with the manifest carried along.
- The same table as a plain CSV with centroid columns and no
  manifest, the shape of a weight table produced by someone else, so
  the example can read it with WeightsArtifact.from_table.
- A display layer of the two countries' ADM1 units, cropped from the
  GADM 2.0 ADM1 target layer and simplified at 0.01 degrees, so the
  notebook's maps need no cluster access.
- An outside impact region weight table (--ir-table), cropped by
  hierid to the two countries and written as parquet, rows unchanged.
  It has one row per grid cell and impact region with popwt and areawt
  already normalized per region; its provenance is not recorded in the
  file. It is read only. The climate crops also keep every cell this
  table uses, so both levels aggregate from the same files.

The check at the end aggregates the full global files with the full
ADM1 artifact and the full impact region table, keeps the two
countries' units, and compares them with the cropped route. They must
agree to rounding.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd

import cil_regionalization as cilreg
from cil_regionalization.bcsd_io import read_bcsd_daily

import compute

SOURCE = Path("/project/cil/gcp/climate/source_data/NASA/BCSD/rcp85/day/atmos")
WEIGHTS_SOURCE = Path(
    "/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/"
    "data/weights/grid025_gadm20_adm1_per_destination"
)
HERE = Path(__file__).resolve().parent
DATA = compute.DATA
TARGETS = Path(
    "/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/"
    "data/targets/gadm20/adm1.parquet"
)
# display layer simplification, in degrees (about 1 km at the equator):
# coarse enough to keep the file small, fine enough for province maps
SIMPLIFY_DEG = 0.01
ISOS = list(compute.COUNTRIES)


def _mounted(path: Path) -> Path:
    if path.exists():
        return path
    alt = Path(str(path).replace("/project/cil", "/Volumes/cil"))
    return alt if alt.exists() else path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(var: str) -> Path:
    name = f"{var}_day_BCSD_rcp85_r1i1p1_{compute.MODEL}_{compute.YEAR}.nc"
    return _mounted(SOURCE) / var / "r1i1p1" / "v1.0" / name


def crop_weights() -> pd.DataFrame:
    src = _mounted(WEIGHTS_SOURCE)
    full = pd.read_parquet(src / "weights.parquet")
    rows = full[full["ISO"].isin(ISOS)].reset_index(drop=True)
    for w in ("popwt", "areawt"):
        sums = rows.groupby(compute.KEYS)[w].sum()
        assert (sums - 1).abs().max() < 1e-9, f"crop breaks {w} sums"

    out = compute.WEIGHTS
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "weights.parquet", index=False)
    manifest = json.loads((src / "weights.manifest.json").read_text())
    manifest["outputs"] = {"weights.parquet": _sha256(out / "weights.parquet")}
    manifest["row_counts"] = {
        "regions": int(rows[compute.KEYS].drop_duplicates().shape[0]),
        "segments": len(rows),
        "total": len(rows),
    }
    manifest["crop"] = {
        "query": f"ISO in {ISOS}",
        "source_artifact": manifest["project_name"],
        "source_sha256": _sha256(src / "weights.parquet"),
    }
    (out / "weights.manifest.json").write_text(json.dumps(manifest, indent=1))
    for iso in ISOS:
        r = rows[rows["ISO"] == iso]
        print(f"weights, {iso}: {len(r)} rows, "
              f"{r[compute.KEYS].drop_duplicates().shape[0]} units, "
              f"{r[['cell_ix', 'cell_iy']].drop_duplicates().shape[0]} cells")
    return rows


def export_csv(rows: pd.DataFrame) -> None:
    table = pd.DataFrame({
        "ISO": rows["ISO"],
        "ID_1": rows["ID_1"],
        "pix_cent_x": rows["cell_lon"],
        "pix_cent_y": rows["cell_lat"],
        "popwt": rows["popwt"],
        "areawt": rows["areawt"],
    })
    table.to_csv(compute.WEIGHTS_CSV, index=False)
    # the text holds every value exactly; pandas' default fast parser,
    # which from_table uses, can land one ulp away, far inside any
    # tolerance that matters
    exact = pd.read_csv(compute.WEIGHTS_CSV, float_precision="round_trip")
    fast = pd.read_csv(compute.WEIGHTS_CSV)
    drift = 0.0
    for c in ("pix_cent_x", "pix_cent_y", "popwt", "areawt"):
        assert np.array_equal(exact[c].to_numpy(), table[c].to_numpy()), \
            f"{c} does not round-trip through the CSV"
        drift = max(drift, float(np.abs(fast[c] - table[c]).max()))
    print(f"csv: {len(table)} rows, exact on a round-trip parse; "
          f"default parse within {drift:.1e}")


def crop_ir_table(path: Path) -> pd.DataFrame:
    """Crop the outside table by hierid, verify it through from_table."""
    full = pd.read_csv(_mounted(path), dtype={"hierid": "string"})
    rows = full[full["hierid"].str[:3].isin(ISOS)].reset_index(drop=True)
    rows.to_parquet(compute.IR_TABLE, index=False)
    # from_table checks sums, duplicates and that every centroid sits on
    # the grid; it raises naming the offending regions or rows
    frame = compute.load_ir_weights().frame
    for iso in ISOS:
        r = frame[frame["hierid"].str[:3] == iso]
        print(f"impact region table, {iso}: {len(r)} rows, "
              f"{r['hierid'].nunique()} regions, "
              f"{r[['cell_ix', 'cell_iy']].drop_duplicates().shape[0]} cells")
    print(f"impact region table: {compute.IR_TABLE.stat().st_size / 1e6:.2f} MB")
    return frame


def crop_climate(rows: pd.DataFrame, iso: str) -> None:
    """``rows`` carries cell_lat and cell_lon of every cell to keep."""
    keep = set(zip(rows["cell_lat"], rows["cell_lon"]))
    lat_lo, lat_hi = rows["cell_lat"].min(), rows["cell_lat"].max()
    lon_lo, lon_hi = rows["cell_lon"].min(), rows["cell_lon"].max()
    assert lon_lo > 0, "the crop assumes no wrap across the antimeridian"
    for var in ("tasmax", "tasmin"):
        src = source_path(var)
        dst = compute.daily_path(var, iso)
        with netCDF4.Dataset(src) as ds:
            lat = np.asarray(ds.variables["lat"][:], dtype="float64")
            lon = np.asarray(ds.variables["lon"][:], dtype="float64")
            ilat = np.where((lat >= lat_lo) & (lat <= lat_hi))[0]
            ilon = np.where((lon >= lon_lo) & (lon <= lon_hi))[0]
            sl_lat = slice(ilat[0], ilat[-1] + 1)
            sl_lon = slice(ilon[0], ilon[-1] + 1)
            v = ds.variables[var]
            block = np.ma.filled(v[:, sl_lat, sl_lon], np.nan).astype("float32")
            mask = np.array([[(la, lo) not in keep for lo in lon[sl_lon]]
                             for la in lat[sl_lat]])
            assert int((~mask).sum()) == len(keep), \
                f"{iso}: some weighted cells fall outside the crop"
            block[:, mask] = np.nan
            assert np.isfinite(block[:, ~mask]).all(), \
                f"{iso} {var}: a weighted cell is missing in the source"
            tvar = ds.variables["time"]
            n_days = len(tvar)
            with netCDF4.Dataset(dst, "w", format=ds.file_format) as out:
                out.setncatts({k: ds.getncattr(k) for k in ds.ncattrs()})
                out.comment = (
                    f"cropped from {src.name} to the cells "
                    f"{compute.COUNTRIES[iso]}'s GADM 2.0 ADM1 units and "
                    f"impact regions draw on; other cells are missing"
                )
                out.createDimension("time", None)
                out.createDimension("lat", sl_lat.stop - sl_lat.start)
                out.createDimension("lon", sl_lon.stop - sl_lon.start)
                t = out.createVariable("time", tvar.dtype, ("time",))
                t.setncatts({k: tvar.getncattr(k) for k in tvar.ncattrs()
                             if k != "bounds"})
                t[:] = tvar[:]
                for name, sl in (("lat", sl_lat), ("lon", sl_lon)):
                    src_v = ds.variables[name]
                    c = out.createVariable(name, src_v.dtype, (name,))
                    c.setncatts({k: src_v.getncattr(k) for k in src_v.ncattrs()})
                    c[:] = src_v[sl]
                fill = v.getncattr("_FillValue")
                d = out.createVariable(var, "f4", ("time", "lat", "lon"),
                                       zlib=True, complevel=9, shuffle=True,
                                       fill_value=fill)
                d.setncatts({k: v.getncattr(k) for k in v.ncattrs()
                             if k not in ("_FillValue", "missing_value")})
                d.missing_value = fill
                d[:] = np.ma.masked_invalid(block)
        print(f"{iso} {var}: {n_days} days, {block.shape[1]} x "
              f"{block.shape[2]} cells, {int((~mask).sum())} kept, "
              f"{dst.stat().st_size / 1e6:.1f} MB")


def build_display_layer(targets: Path) -> None:
    """The two countries' ADM1 polygons, cropped and simplified, so the
    notebook's maps work from a clone with no cluster access."""
    shapes = gpd.read_parquet(_mounted(targets))
    shapes = shapes[shapes["ISO"].isin(ISOS)][["ISO", "ID_1", "NAME_1", "geometry"]]
    units = set(map(tuple, pd.read_parquet(
        compute.WEIGHTS / "weights.parquet", columns=compute.KEYS
    ).drop_duplicates().to_numpy()))
    found = set(zip(shapes["ISO"], shapes["ID_1"]))
    assert found == units, "display layer and weights disagree on the units"
    shapes["geometry"] = shapes.geometry.simplify(SIMPLIFY_DEG)
    shapes = shapes.reset_index(drop=True)
    shapes.to_parquet(compute.PLOT_LAYER)
    counts = shapes.groupby("ISO").size().to_dict()
    print(f"display layer: {counts}, simplified at {SIMPLIFY_DEG} degrees, "
          f"{compute.PLOT_LAYER.stat().st_size / 1e6:.2f} MB")


def _collapse_wrapped_segments(path: Path) -> pd.DataFrame:
    """The full outside table with segments of one (region, cell) summed.

    Regions that cross the antimeridian carry two segments in the
    180.125 and -179.875 columns; on the [-180,180) grid both are the
    same cell, so from_table rightly refuses them as duplicates. Two
    segments of the same cell and region are one weight: sum them.
    Neither cropped country is affected; the count is printed so the
    collapse is on the record.
    """
    cols = ["hierid", "pix_cent_x", "pix_cent_y", "popwt", "areawt"]
    t = pd.read_csv(path, usecols=cols, dtype={"hierid": "string"})
    # the same wrap from_table applies before indexing
    t["pix_cent_x"] = np.mod(t["pix_cent_x"] + 180.0, 360.0) - 180.0
    key = ["hierid", "pix_cent_x", "pix_cent_y"]
    dup = t.duplicated(subset=key, keep=False)
    regions = sorted(t.loc[dup, "hierid"].unique())
    assert not any(r[:3] in ISOS for r in regions), \
        "a cropped country has wrapped segments"
    out = (t.groupby(key, sort=False)[["popwt", "areawt"]]
           .sum(min_count=1).reset_index())
    print(f"outside table: {len(t) - len(out)} rows collapsed from "
          f"{int(dup.sum())} duplicated (hierid, cell) rows, in {regions}")
    return out


def _restrict(artifact, cells: pd.DataFrame, country):
    """Drop weighted cells the global file has no data for, after
    proving none belongs to a cropped country."""
    key = ["cell_ix", "cell_iy"]
    frame = artifact.frame
    present = frame.set_index(key).index.isin(cells.set_index(key).index)
    lost = frame.loc[~present]
    assert not country(lost).isin(ISOS).any(), \
        "a cropped country's cell has no data"
    print(f"global file: {lost[key].drop_duplicates().shape[0]} weighted "
          f"cells without data, none in {ISOS}")
    return dataclasses.replace(artifact, frame=frame.loc[present])


def _compare(full, cropped, cells, c_cells, ids, country, label) -> None:
    n_units = cropped.frame[ids].drop_duplicates().shape[0]
    for weight in ("pop", "area"):
        a = compute.aggregate(full, cells, weight)
        a = a[country(a).isin(ISOS)]
        b = compute.aggregate(cropped, c_cells, weight)
        both = a.merge(b, on=ids, suffixes=("_full", "_crop"),
                       validate="one_to_one")
        assert len(both) == len(b)
        if len(both) < n_units:
            # a unit with no usable weight under this weighting (all its
            # shares missing) has no value to compare
            known = cropped.frame[ids].drop_duplicates()
            absent = known.merge(both[ids], on=ids, how="left", indicator=True)
            absent = absent.loc[absent["_merge"] == "left_only", ids]
            print(f"{label}, {weight} weights: {len(absent)} units carry no "
                  f"{weight} weight: {absent.to_dict('records')}")
        for col in ("hdd", "cdd"):
            diff = float((both[f"{col}_full"] - both[f"{col}_crop"]).abs().max())
            print(f"{label}, full product vs crop, {weight} weights, {col}: "
                  f"largest difference {diff:.1e} over {len(both)} of "
                  f"{n_units} units")
            assert diff < 1e-9


def check_against_full(ir_table: Path) -> None:
    """The cropped route must reproduce the full product for both."""
    full_weights = cilreg.WeightsArtifact.load(_mounted(WEIGHTS_SOURCE))
    gmax = read_bcsd_daily(str(source_path("tasmax")), "tasmax",
                           require_full_year=True)
    gmin = read_bcsd_daily(str(source_path("tasmin")), "tasmin",
                           require_full_year=True)
    hdd, cdd = compute.degree_days(gmax.values, gmin.values)
    del gmin
    cells = compute.cell_frame(gmax, {"hdd": hdd, "cdd": cdd})
    del gmax

    # the global file has polar voids: weighted cells with no data,
    # none in either country; the global runs drop them
    by_iso = lambda f: f["ISO"]
    by_hierid = lambda f: f["hierid"].str[:3]
    full_weights = _restrict(full_weights, cells, by_iso)
    full_ir = cilreg.WeightsArtifact.from_table(
        _collapse_wrapped_segments(_mounted(ir_table)), id_fields=["hierid"],
        weights={"pop": "popwt", "area": "areawt"}, source=compute.GRID)
    full_ir = _restrict(full_ir, cells, by_hierid)

    c_cells = compute.all_cells()
    _compare(full_weights, compute.load_weights(), cells, c_cells,
             compute.KEYS, by_iso, "ADM1")
    _compare(full_ir, compute.load_ir_weights(), cells, c_cells,
             ["hierid"], by_hierid, "impact regions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=Path, default=TARGETS,
                        help="GADM 2.0 ADM1 target layer (GeoParquet)")
    parser.add_argument("--ir-table", type=Path, required=True,
                        help="outside impact region weight table (CSV); "
                             "read only")
    args = parser.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    rows = crop_weights()
    export_csv(rows)
    ir = crop_ir_table(args.ir_table)
    for iso in ISOS:
        keep = pd.concat([
            rows.loc[rows["ISO"] == iso, ["cell_lat", "cell_lon"]],
            ir.loc[ir["hierid"].str[:3] == iso, ["cell_lat", "cell_lon"]],
        ]).drop_duplicates()
        crop_climate(keep, iso)
    build_display_layer(args.targets)
    check_against_full(args.ir_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
