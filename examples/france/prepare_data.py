"""Writes data/ for the France example, once, on the cluster. It reads
the impact regions, the population raster and series, the two mortality
trees and the climate model weights where the published runs read
them, and the ADMIN EXPRESS shapefiles from where you unpacked them.
It fetches the published GADM 2.0 weights from Zenodo and keeps the
rows of the French impact regions. It does not copy the temperature
table, which the notebook reads from its path.

    python examples/france/prepare_data.py \\
        --admin-express <directory with DEPARTEMENT.shp and REGION.shp> \\
        --admin-express-version <the edition, as IGN names it>

The sample is two batches, rcp45 and rcp85, every climate model the
tree has under each, the low growth model, SSP3, 2020 to 2099.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# first, since importing it points PROJ at this environment's data
import cil_regionalization as cilreg
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

PHYSICAL = Path(
    "/project/cil/battuta-shares-S3-archive/gcp/outputs/mortality/impacts-darwin/montecarlo"
)
VALUED = Path(
    "/project/cil/home_dirs/scadavidsanchez/repos/impact-map-data/mortality/current/data/montecarlo_original"
)
LEVELS = "Agespec_interaction_response-combined-levels.nc4"
HISTCLIM = "Agespec_interaction_response-combined-histclim-levels.nc4"
DAMAGES = "mortality_damages_IR_batch.nc4"
IMPACT_REGIONS = Path(
    "/project/cil/gcp/regions/world-combo-201710/agglomerated-world-new.shp"
)
RASTER = Path(
    "/project/cil/gcp/social/population/gpw_v4r10_unwpp_2015/raster_geotiff/gpw_v4r10_unwpp_2015_30ss.tif"
)
POPULATION = Path(
    "/project/cil/gcp/social/baselines/population/merged/population-merged.SSP3.csv"
)
SMME = Path("/project/cil/gcp/climate/BCSD/SMME/SMME-weights")
PUBLISHED = "gadm20-adm2-per-source"

BATCHES = ["batch0", "batch1"]
RCPS = ["rcp45", "rcp85"]
IAM, SSP = "low", "SSP3"
YEARS = (2020, 2099)
# the overseas regions and their departments, which this example leaves out
OVERSEAS_REGIONS = {"01", "02", "03", "04", "06"}

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find(directory: Path, name: str) -> Path:
    found = sorted(directory.rglob(name))
    if len(found) != 1:
        raise SystemExit(f"expected one {name} under {directory}, found {found}")
    return found[0]


def write_targets(admin_express: Path) -> gpd.GeoDataFrame:
    """The 96 departments and 13 regions of metropolitan France, in
    longitude and latitude. Raises if the counts are other."""
    departments = gpd.read_file(_find(admin_express, "DEPARTEMENT.shp")).to_crs(
        "EPSG:4326"
    )
    regions = gpd.read_file(_find(admin_express, "REGION.shp")).to_crs("EPSG:4326")
    departments = departments[~departments["INSEE_REG"].isin(OVERSEAS_REGIONS)]
    regions = regions[~regions["INSEE_REG"].isin(OVERSEAS_REGIONS)]
    assert len(departments) == 96, f"{len(departments)} metropolitan departments"
    assert len(regions) == 13, f"{len(regions)} metropolitan regions"
    departments = departments[["INSEE_DEP", "INSEE_REG", "NOM", "geometry"]]
    regions = regions[["INSEE_REG", "NOM", "geometry"]]
    departments.sort_values("INSEE_DEP").to_parquet(DATA / "departments.parquet")
    regions.sort_values("INSEE_REG").to_parquet(DATA / "regions.parquet")
    print(f"targets: {len(departments)} departments, {len(regions)} regions")
    return departments


def write_impact_regions(departments: gpd.GeoDataFrame) -> list[str]:
    """The French impact regions that touch a metropolitan department."""
    shapes = gpd.read_file(IMPACT_REGIONS)[["hierid", "geometry"]].to_crs("EPSG:4326")
    french = shapes[shapes["hierid"].str.startswith("FRA.")]
    touching = french[french.intersects(departments.union_all())]
    dropped = sorted(set(french["hierid"]) - set(touching["hierid"]))
    if dropped:
        print(
            f"impact regions: left out {len(dropped)} that touch no department: {dropped}"
        )
    touching.sort_values("hierid").to_parquet(DATA / "impact_regions.parquet")
    print(f"impact regions: {len(touching)}")
    return sorted(touching["hierid"])


def write_raster(departments: gpd.GeoDataFrame) -> None:
    """The population raster cut to the departments with a margin of a
    tenth of a degree, cell for cell, with its nodata kept."""
    import rasterio
    from rasterio.windows import from_bounds

    west, south, east, north = departments.total_bounds
    with rasterio.open(RASTER) as source:
        window = (
            from_bounds(
                west - 0.1, south - 0.1, east + 0.1, north + 0.1, source.transform
            )
            .round_offsets()
            .round_lengths()
        )
        cells = source.read(1, window=window)
        profile = source.profile | {
            "height": cells.shape[0],
            "width": cells.shape[1],
            "transform": source.window_transform(window),
            "compress": "deflate",
            "predictor": 3,
            "tiled": False,
        }
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    with rasterio.open(DATA / "population_2015_30ss.tif", "w", **profile) as out:
        out.write(cells, 1)
    print(f"raster: {cells.shape[1]} by {cells.shape[0]} cells")


def write_population(hierids: list[str]) -> None:
    """SSP3 population of each impact region in each year, held over
    each five-year step of the series, which is how the projection
    reads it."""
    table = pd.read_csv(POPULATION, comment="#")
    table = table[table["region"].isin(hierids)]
    missing = set(hierids) - set(table["region"])
    assert not missing, f"regions absent from the SSP3 series: {sorted(missing)}"
    grid = table.pivot(index="region", columns="year", values="value")
    years = range(YEARS[0], YEARS[1] + 1)
    wide = pd.DataFrame({y: grid[5 * (y // 5)] for y in years}).reindex(hierids)
    assert wide.notna().all().all() and (wide >= 0).all().all()
    long = wide.reset_index(names="hierid").melt(
        id_vars="hierid", var_name="year", value_name="pop"
    )
    long["year"] = long["year"].astype("int64")
    long.to_parquet(DATA / "ir_population.parquet", index=False)
    print(f"population: {len(long)} rows")


def _variables(path: Path, hierids: list[str], names: list[str], labels: str | None):
    """``names`` of the file at ``path`` for ``hierids`` over the years
    of the sample, one row per region and year."""
    with xr.open_dataset(path) as ds:
        regions = [str(r) for r in (ds[labels] if labels else ds["region"]).values]
        at = [regions.index(h) for h in hierids]
        part = ds[names].isel(region=at).sel(year=slice(*YEARS))
        years = [int(y) for y in part["year"].values]
        assert years == list(range(YEARS[0], YEARS[1] + 1)), (
            f"{path} has years {years[:3]}..."
        )
        frame = pd.DataFrame(
            {
                "hierid": np.repeat(hierids, len(years)),
                "year": np.tile(years, len(hierids)),
            }
        )
        for name in names:
            frame[name] = (
                part[name].transpose("region", "year").values.ravel().astype(np.float32)
            )
    return frame


def write_tree(hierids: list[str]) -> None:
    """One leaf for each batch, rcp and climate model the physical tree
    has, with its deaths, its histclim deaths, and its damages with the
    GDP they are measured against. Raises if the valued tree lacks a
    leaf the physical one has."""
    for batch in BATCHES:
        for rcp in RCPS:
            gcms = sorted(
                p.name for p in (PHYSICAL / batch / rcp).iterdir() if p.is_dir()
            )
            for gcm in gcms:
                physical = PHYSICAL / batch / rcp / gcm / IAM / SSP
                valued = VALUED / batch / rcp / gcm / IAM / SSP
                leaf = DATA / "tree" / batch / rcp / gcm
                leaf.mkdir(parents=True, exist_ok=True)
                for source, target in (
                    (LEVELS, "deaths.parquet"),
                    (HISTCLIM, "deaths-histclim.parquet"),
                ):
                    frame = _variables(
                        physical / source, hierids, ["rebased"], "regions"
                    )
                    frame.rename(columns={"rebased": "deaths"}).to_parquet(
                        leaf / target, index=False
                    )
                frame = _variables(
                    valued / DAMAGES, hierids, ["total_damages", "gdp"], None
                )
                frame.rename(columns={"total_damages": "damages"}).to_parquet(
                    leaf / "damages.parquet", index=False
                )
                print(batch, rcp, gcm, flush=True)


def write_smme() -> None:
    (DATA / "smme").mkdir(parents=True, exist_ok=True)
    for rcp in RCPS:
        shutil.copyfile(
            SMME / f"{rcp}_SMME_weights.tsv", DATA / "smme" / f"{rcp}_SMME_weights.tsv"
        )
    print("smme weights: copied")


def write_published(hierids: list[str]) -> None:
    """Every row of the published GADM 2.0 department weights for the
    French impact regions, so each region's weights still add up to
    one, with the manifest of the published file and the hash of the
    rows kept."""
    from cil_regionalization.fetch import default_cache_dir, load_registry

    full = cilreg.fetch_weights(PUBLISHED)
    rows = full.frame[full.frame["hierid"].isin(hierids)].reset_index(drop=True)
    sums = rows.groupby("hierid")["popwt"].sum()
    assert (sums - 1).abs().max() < 1e-9, "the slice breaks the per-source sums"
    out = DATA / "published" / PUBLISHED
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "weights.parquet", index=False)
    cached = (
        default_cache_dir()
        / f"record-{load_registry()[PUBLISHED].record_id}"
        / PUBLISHED
    )
    manifest = json.loads((cached / "weights.manifest.json").read_text())
    manifest["outputs"] = {"weights.parquet": _sha256(out / "weights.parquet")}
    manifest["row_counts"] = {
        "regions": int(rows[["ISO", "ID_1", "ID_2"]].drop_duplicates().shape[0]),
        "source_units": len(set(rows["hierid"])),
        "total": len(rows),
    }
    (out / "weights.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"published weights: {len(rows)} rows")


def write_sources(admin_express_version: str) -> None:
    """Where each file came from, under which terms, and its SHA256."""
    entries = {
        "impact_regions.parquet": (
            str(IMPACT_REGIONS),
            "Climate Impact Lab, doi:10.5281/zenodo.21934131",
        ),
        "departments.parquet": (
            "IGN ADMIN EXPRESS " + admin_express_version,
            "Licence Ouverte 2.0, IGN",
        ),
        "regions.parquet": (
            "IGN ADMIN EXPRESS " + admin_express_version,
            "Licence Ouverte 2.0, IGN",
        ),
        "population_2015_30ss.tif": (str(RASTER), "GPW v4 rev 10, CIESIN, CC BY 4.0"),
        "ir_population.parquet": (str(POPULATION), "Climate Impact Lab, SSP3"),
    }
    document = {
        "targets_version": f"admin-express-{admin_express_version}",
        "files": {
            name: {"from": origin, "terms": terms, "sha256": _sha256(DATA / name)}
            for name, (origin, terms) in entries.items()
        },
        "tree": {
            "physical": f"{PHYSICAL}/<batch>/<rcp>/<gcm>/{IAM}/{SSP}/{LEVELS} and {HISTCLIM}",
            "valued": f"{VALUED}/<batch>/<rcp>/<gcm>/{IAM}/{SSP}/{DAMAGES}",
            "batches": BATCHES,
            "rcps": RCPS,
            "years": list(YEARS),
        },
    }
    (DATA / "sources.json").write_text(json.dumps(document, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--admin-express", type=Path, required=True)
    parser.add_argument("--admin-express-version", required=True)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    departments = write_targets(args.admin_express)
    hierids = write_impact_regions(departments)
    write_raster(departments)
    write_population(hierids)
    write_tree(hierids)
    write_smme()
    write_published(hierids)
    write_sources(args.admin_express_version)


if __name__ == "__main__":
    main()
