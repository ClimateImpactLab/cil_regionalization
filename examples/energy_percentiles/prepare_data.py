"""Rebuild the example's data/ from the CIL filesystem on Midway3.

Needs /project/cil. Everyone else uses what is committed under data/.
Run it from the repository root, through prepare_data.sbatch.

What it writes, all for electricity, RCP8.5, SSP3, IIASA (low) and
the years of the two windows, 2020 to 2039 and 2080 to 2099:

- For each of the 33 climate projections, the delta-method leaves (full
  adaptation and histclim), cut to the impact regions of Qatar and
  Iceland, and the median leaves of the same runs, cut to the regions
  of Qatar, Iceland, Chile and Austria. The files keep the layout of
  the originals, so the library reads them the same way.
  Source: /project/cil/gcp/outputs_newsocioeconomics/energy/electricity/
- Population per impact region and year, for all four countries.
  Source: ir_combined_SSP3_IIASA.csv of the socioeconomic panel
  /project/cil/gcp/outputs_newsocioeconomics/socioeconomics/1.0.0/
- The electricity price per country and year, for all four countries.
  Source: /project/cil/gcp/social/baselines/energy/
  IEA_Price_FIN_Clean_gr014_GLOBAL_COMPILE.dta
- The GADM 2.0 adm1 and adm2 weights (per source) for those regions.
  Source: https://doi.org/10.5281/zenodo.21934155
- The SMME model weights for RCP8.5, model and weight only.
  Source: /project/cil/gcp/climate/BCSD/SMME/SMME-weights/rcp85_SMME_weights.tsv
- Outlines of Chile and Austria, their impact regions, adm1 and adm2
  units, without the pieces under 50 km2 other than each unit's
  largest, and simplified at 0.01 degrees.
  Source: /project/cil/gcp/impact_map_inputs/targets/gadm20/ and
  /project/cil/gcp/regions/world-combo-201710/agglomerated-world-new.shp

The crops are checked against the Impact Map's numbers before it ends.
"""
from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import pyogrio
import shapely

import cil_regionalization as cilreg
from cil_regionalization.fetch import default_cache_dir, load_registry

import data

ENERGY = Path("/project/cil/gcp/outputs_newsocioeconomics/energy") / data.FUEL
PANEL = Path("/project/cil/gcp/outputs_newsocioeconomics/socioeconomics/1.0.0"
             f"/ir_combined_{data.SSP}_IIASA.csv")
PRICES = Path("/project/cil/gcp/social/baselines/energy/"
              "IEA_Price_FIN_Clean_gr014_GLOBAL_COMPILE.dta")
SMME = Path(f"/project/cil/gcp/climate/BCSD/SMME/SMME-weights/{data.RCP}_SMME_weights.tsv")
IMPACT_MAP = Path("/project/cil/gcp/impact_map/energy")
TARGETS = Path("/project/cil/gcp/impact_map_inputs/targets/gadm20")
IR_SHAPES = Path("/project/cil/gcp/regions/world-combo-201710/agglomerated-world-new.shp")
WEIGHTS = {"adm1": "gadm20-adm1-per-source", "adm2": "gadm20-adm2-per-source"}
KEYS = {"adm0": ["ISO"], "adm1": ["ISO", "ID_1"], "adm2": ["ISO", "ID_1", "ID_2"]}
YEARS = np.concatenate([np.arange(lo, hi + 1) for lo, hi in data.WINDOWS])
# coarse enough to keep the files small, fine enough for municipalities
SIMPLIFY_DEG = 0.01
# Chile's coast has thousands of islands
ISLAND_KM2 = 50
ALL_COUNTRIES = {**data.COUNTRIES, **data.MAP_COUNTRIES}
GJ_PER_KWH = 0.0036
USD2005_TO_2019 = 1.273526
QUANTILES = {"q05": 0.05, "q17": 0.17, "q25": 0.25, "q50": 0.50,
             "q75": 0.75, "q83": 0.83, "q95": 0.95}
RTOL = 1e-9


def source(kind: str, gcm: str, histclim: bool) -> Path:
    sub = "delta_method/median" if kind == "delta_method" else "median"
    suffix = "-histclim" if histclim else ""
    return (ENERGY / sub / data.RCP / gcm / data.IAM / data.SSP
            / f"{data.STEM}{suffix}.nc4")


def crop(src: Path, dst: Path, keep: list[str], countries: str) -> list[str]:
    """Write src cut to the regions in keep and the years of the two
    windows, then read it back and compare. Returns the source region
    order."""
    with netCDF4.Dataset(src) as ds:
        regions = [str(r) for r in ds.variables["regions"][:]]
        idx = np.array([regions.index(r) for r in keep])
        years = np.asarray(ds.variables["year"][:])
        yi = np.array([int(np.where(years == y)[0][0]) for y in YEARS])
        has_gradient = "rebased_bcde" in ds.variables
        values = np.asarray(ds.variables["rebased"][:][yi][:, idx])
        if has_gradient:
            var = ds.variables["rebased_bcde"]
            gradient = np.stack([np.asarray(var[:, y, :])[:, idx] for y in yi], axis=1)
            vcv = np.asarray(ds.variables["vcv"][:])
        ops = [str(o) for o in ds.variables["orderofoperations"][:]]
        attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
        var_attrs = {n: {a: v.getncattr(a) for a in v.ncattrs()}
                     for n, v in ds.variables.items()}

    dst.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(dst, "w") as out:
        out.setncatts(attrs)
        out.setncattr("cropped", f"regions of {countries}, years 2020 to 2039 "
                                 "and 2080 to 2099, by "
                                 "examples/energy_percentiles/prepare_data.py")
        out.createDimension("year", len(YEARS))
        out.createDimension("region", len(keep))
        out.createDimension("operation", len(ops))
        v = out.createVariable("year", "i4", ("year",))
        v[:] = YEARS
        v = out.createVariable("regions", str, ("region",))
        v[:] = np.array(keep, dtype=object)
        v = out.createVariable("orderofoperations", str, ("operation",))
        v[:] = np.array(ops, dtype=object)
        v = out.createVariable("rebased", "f4", ("year", "region"), zlib=True)
        v[:] = values
        if has_gradient:
            out.createDimension("coefficient", vcv.shape[0])
            v = out.createVariable("vcv", "f4", ("coefficient", "coefficient"))
            v[:] = vcv
            v = out.createVariable("rebased_bcde", "f4", ("coefficient", "year", "region"),
                                   zlib=True, shuffle=True)
            v[:] = gradient
        for name, a in var_attrs.items():
            if name in out.variables and a:
                out.variables[name].setncatts(a)

    with netCDF4.Dataset(dst) as back:
        same = np.array_equal(np.asarray(back.variables["rebased"][:]), values)
        if has_gradient:
            same &= np.array_equal(np.asarray(back.variables["rebased_bcde"][:]), gradient)
            same &= np.array_equal(np.asarray(back.variables["vcv"][:]), vcv)
    if not same:
        raise SystemExit(f"{dst} does not read back equal to its slice of {src}")
    return regions


def crop_leaves() -> tuple[list[str], list[str]]:
    root = ENERGY / "delta_method" / "median" / data.RCP
    gcms = sorted(p.name for p in root.iterdir() if p.is_dir())
    with netCDF4.Dataset(source("delta_method", gcms[0], False)) as ds:
        order = [str(r) for r in ds.variables["regions"][:]]
    table = [r for r in order if r.split(".")[0] in data.COUNTRIES]
    both = [r for r in order if r.split(".")[0] in ALL_COUNTRIES]
    print(f"{len(gcms)} climate projections; {len(table)} regions in the delta-method leaves: "
          f"{table}; {len(both)} in the median leaves", flush=True)
    for gcm in gcms:
        start = time.time()
        for histclim in (False, True):
            for kind, dst, keep, names in (
                    ("delta_method", data.gradient_path(gcm, histclim), table,
                     "Qatar and Iceland"),
                    ("median", data.median_path(gcm, histclim), both,
                     "Qatar, Iceland, Chile and Austria")):
                if crop(source(kind, gcm, histclim), dst, keep, names) != order:
                    raise SystemExit(f"{source(kind, gcm, histclim)}: region order differs")
        print(f"  {gcm}: 4 files in {time.time() - start:.0f}s", flush=True)
    return table, both


def write_population(keep):
    df = pd.read_csv(PANEL, usecols=["hierid", "year", "pop"])
    df = df[df["hierid"].isin(keep) & df["year"].isin(YEARS)]
    if len(df) != len(keep) * len(YEARS) or df["pop"].isna().any():
        raise SystemExit(f"{PANEL}: population incomplete for the crop")
    df.sort_values(["hierid", "year"]).to_parquet(data.POPULATION, index=False)


def write_price():
    table = pd.read_stata(PRICES)
    col = f"{data.FUEL}compile_price"
    df = table.loc[table["country"].isin(ALL_COUNTRIES) & table["year"].isin(YEARS),
                   ["country", "year", col]].rename(columns={col: "price"})
    if len(df) != len(ALL_COUNTRIES) * len(YEARS) or df["price"].isna().any():
        raise SystemExit(f"{PRICES}: {col} incomplete for the crop")
    df["year"] = df["year"].astype(int)
    df.sort_values(["country", "year"]).to_parquet(data.PRICE, index=False)


def write_weights(keep):
    for level, name in WEIGHTS.items():
        frame = cilreg.fetch_weights(name).frame
        rows = frame[frame["hierid"].isin(keep)]
        if not np.allclose(rows.groupby("hierid")["popwt"].sum(), 1.0, atol=1e-9):
            raise SystemExit(f"{name}: the crop breaks the per-source sums")
        # a country's total is the sum of its units only if its regions
        # put all their weight inside it
        abroad = rows[~rows["ISO"].isin(ALL_COUNTRIES)]
        if len(abroad):
            raise SystemExit(f"{name}: regions of the crop put weight in other "
                             f"countries: {sorted(set(abroad['ISO']))}")
        outside = frame[frame["ISO"].isin(ALL_COUNTRIES) & ~frame["hierid"].isin(keep)]
        if len(outside):
            raise SystemExit(f"{name}: units of {list(ALL_COUNTRIES)} take weight from "
                             f"regions outside the crop: {sorted(set(outside['hierid']))[:5]}")
        out = data.WEIGHTS[level]
        out.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(out / "weights.parquet", index=False)
        record = load_registry()[name].record_id
        manifest = json.loads((default_cache_dir() / f"record-{record}" / name
                               / "weights.manifest.json").read_text())
        manifest["outputs"] = {"weights.parquet": hashlib.sha256(
            (out / "weights.parquet").read_bytes()).hexdigest()}
        manifest["row_counts"] = {
            "regions": int(rows[KEYS[level]].drop_duplicates().shape[0]),
            "source_units": len(keep),
            "total": len(rows),
        }
        (out / "weights.manifest.json").write_text(json.dumps(manifest, indent=1))


def write_smme():
    table = pd.read_csv(SMME, sep="\t", usecols=["model", "weight"])
    table.to_csv(data.SMME, index=False)


def write_outlines(keep):
    isos = list(data.MAP_COUNTRIES)
    for level in ("adm0", "adm1", "adm2"):
        shapes = gpd.read_parquet(TARGETS / f"{level}.parquet", filters=[("ISO", "in", isos)])
        names = [c for c in shapes.columns if c.startswith("NAME_")]
        shapes = shapes[KEYS[level] + names + ["geometry"]]
        save_outlines(shapes, data.OUTLINES[level])
    where = " OR ".join(f"hierid LIKE '{iso}%'" for iso in isos)
    shapes = pyogrio.read_dataframe(IR_SHAPES, where=where)[["hierid", "geometry"]]
    # the shapefile has no .prj; its coordinates are longitude and latitude
    if shapes.crs is None:
        shapes = shapes.set_crs("EPSG:4326")
    if sorted(shapes["hierid"]) != sorted(keep):
        raise SystemExit(f"{IR_SHAPES}: outlines and leaves cover different regions")
    save_outlines(shapes, data.OUTLINES["impact_region"])


def save_outlines(shapes, path):
    parts = shapes.geometry.explode(index_parts=True)
    km2 = parts.to_crs("EPSG:6933").area / 1e6       # equal area
    largest = km2 == km2.groupby(level=0).transform("max")
    kept = parts[(km2 >= ISLAND_KM2) | largest]
    geometry = kept.groupby(level=0).agg(lambda g: shapely.union_all(list(g)))
    print(f"  {path.name}: {len(shapes)} units, {len(parts)} pieces, "
          f"{len(kept)} kept", flush=True)
    shapes = shapes.assign(geometry=gpd.GeoSeries(geometry, crs=shapes.crs)
                           .reindex(shapes.index)
                           .simplify(SIMPLIFY_DEG, preserve_topology=True))
    shapes.to_parquet(path, index=False)


def median_effect(gcm, regions):
    out = []
    for histclim in (False, True):
        with netCDF4.Dataset(data.median_path(gcm, histclim)) as ds:
            names = [str(r) for r in ds.variables["regions"][:]]
            values = np.asarray(ds.variables["rebased"][:], dtype="float64")
        out.append(values[:, [names.index(r) for r in regions]])
    return out[0] - out[1]


def inputs():
    weights = {level: cilreg.WeightsArtifact.load(p) for level, p in data.WEIGHTS.items()}
    pop = pd.read_parquet(data.POPULATION).pivot(index="year", columns="hierid", values="pop")
    price = pd.read_parquet(data.PRICE).pivot(index="year", columns="country", values="price")
    smme = pd.read_csv(data.SMME)
    smme = dict(zip(smme["model"].str.replace("*", "", regex=False).str.lower(),
                    smme["weight"]))
    return weights, pop, price, smme


def compute() -> pd.DataFrame:
    """The notebook's calculation, run on data/, for the check."""
    weights, pop, price, smme = inputs()
    dollars = USD2005_TO_2019 / GJ_PER_KWH
    rows = []
    for gcm in data.gcms():
        clim = [cilreg.read_gradient_leaf(data.gradient_path(gcm), year=int(y)) for y in YEARS]
        hist = [cilreg.read_gradient_leaf(data.gradient_path(gcm, True), year=int(y))
                for y in YEARS]
        regions = clim[0].regions
        grad = np.stack([c.gradient - h.gradient for c, h in zip(clim, hist)])
        mean = median_effect(gcm, regions)
        scale = (pop.loc[YEARS, regions].to_numpy()
                 * price.loc[YEARS, [r.split(".")[0] for r in regions]].to_numpy())
        for lo, hi in data.WINDOWS:
            g = cilreg.window_gradient(grad * scale[:, None, :], YEARS, (lo, hi))
            m = cilreg.window_gradient((mean * scale)[:, None, :], YEARS, (lo, hi))
            for level in ("adm1", "adm2"):
                targets, G = cilreg.aggregate_gradient(weights[level], regions, g)
                _, M = cilreg.aggregate_gradient(weights[level], regions, m)
                G, M = G * dollars, M[:, 0] * dollars
                parts = [(level, targets, G, M)]
                if level == "adm1":
                    members = targets.groupby("ISO").indices
                    parts.append(("adm0", pd.DataFrame({"ISO": list(members)}),
                                  np.stack([G[i].sum(axis=0) for i in members.values()]),
                                  np.array([M[i].sum() for i in members.values()])))
                for lvl, keys, gg, mm in parts:
                    var = cilreg.delta_method(gg, clim[0].vcv)["sd"].to_numpy() ** 2
                    rows.append(keys.assign(level=lvl, window=f"{lo}-{hi}", gcm=gcm,
                                            mean=mm, variance=var))
    long = pd.concat(rows, ignore_index=True)
    out = {}
    for (level, window), part in long.groupby(["level", "window"]):
        part = part.set_index(KEYS[level] + ["gcm"])
        M, V = part["mean"].unstack("gcm"), part["variance"].unstack("gcm")
        w = np.array([smme[g.replace("surrogate_", "").lower()] for g in M.columns])
        res = pd.DataFrame(cilreg.mixture_quantile(w, M.to_numpy(), V.to_numpy(),
                                                   list(QUANTILES.values())),
                           index=M.index, columns=list(QUANTILES))
        res.insert(0, "mean", M.to_numpy() @ w / w.sum())
        out[(level, window)] = res
    return out


def compute_map(regions) -> dict:
    """The map's means for Chile and Austria, from the median leaves only."""
    weights, pop, price, smme = inputs()
    gcms = data.gcms()
    w = np.array([smme[g.replace("surrogate_", "").lower()] for g in gcms])
    mixed = sum(wi * median_effect(g, regions) for wi, g in zip(w, gcms)) / w.sum()
    valued = (mixed * pop.loc[YEARS, regions].to_numpy()
              * price.loc[YEARS, [r.split(".")[0] for r in regions]].to_numpy()
              * USD2005_TO_2019 / GJ_PER_KWH)
    out = {}
    for lo, hi in data.WINDOWS:
        m = cilreg.window_gradient(valued[:, None, :], YEARS, (lo, hi))
        for level in ("adm1", "adm2"):
            targets, M = cilreg.aggregate_gradient(weights[level], regions, m)
            out[(level, f"{lo}-{hi}")] = pd.DataFrame(
                {"mean": M[:, 0]}, index=pd.MultiIndex.from_frame(targets))
        adm1 = out[("adm1", f"{lo}-{hi}")]
        out[("adm0", f"{lo}-{hi}")] = adm1.groupby(level="ISO").sum()
    return out


def impact_map(level: str, window: str, countries) -> pd.DataFrame:
    df = pd.read_parquet(IMPACT_MAP / level / f"energy_valued_{data.FUEL}_{data.RCP}.parquet",
                         filters=[("ISO", "in", list(countries))])
    df = df[(df["iam"] == data.IAM) & (df["ssp"] == data.SSP) & (df["window"] == window)]
    return df.pivot_table(index=KEYS[level], columns="statistic", values="value")


def check(map_regions):
    worst = 0.0
    for what, countries, results in (
            ("percentiles", data.COUNTRIES, compute()),
            ("map means", data.MAP_COUNTRIES, compute_map(map_regions))):
        for (level, window), ours in results.items():
            theirs = impact_map(level, window, countries)
            a = ours.sort_index()
            if len(theirs) != len(a):
                raise SystemExit(f"{what} {level} {window}: {len(a)} units here, "
                                 f"{len(theirs)} in the Impact Map")
            b = theirs.loc[a.index, a.columns].to_numpy()
            diff, nonzero = np.abs(a.to_numpy() - b), b != 0
            if (diff[~nonzero] > 0).any():
                raise SystemExit(f"{what} {level} {window}: nonzero where the Impact Map has zero")
            rel = float(np.max(diff[nonzero] / np.abs(b[nonzero])))
            print(f"{what} {level} {window}: {len(a)} units, "
                  f"worst relative difference {rel:.1e}")
            worst = max(worst, rel)
    if worst > RTOL:
        raise SystemExit(f"the crops do not give the Impact Map's numbers (worst {worst:.1e})")


def main():
    start = time.time()
    data.DATA.mkdir(exist_ok=True)
    _, both = crop_leaves()
    write_population(both)
    write_price()
    write_weights(both)
    write_smme()
    map_regions = [r for r in both if r.split(".")[0] in data.MAP_COUNTRIES]
    write_outlines(map_regions)
    check(map_regions)
    size = sum(p.stat().st_size for p in data.DATA.rglob("*") if p.is_file())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"data/: {size / 2**20:.1f} MB; {time.time() - start:.0f}s, peak {peak:.2f} GB")


if __name__ == "__main__":
    main()
