"""Where the example's committed files are, and how to read them.

Nothing here computes anything. The notebook does that.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import netCDF4

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = "MPI-ESM-MR"
YEAR = 2020
COUNTRIES = {"PHL": "Philippines", "DEU": "Germany"}

# the quarter-degree province weights (grid025, GADM 2.0 ADM1), and
# the same weights as a plain CSV
# Source: generated with this library from the GADM 2.0 ADM1 layer
# and the GPW v4 rev 10 population raster, adjusted to UN World
# Population Prospects. Not published.
WEIGHTS = DATA / "weights" / "grid025_gadm20_adm1_per_destination"
WEIGHTS_CSV = DATA / "adm1_weights.csv"
# impact region segment weights
# Source: weights used for aggregation on the Berkeley servers,
# https://gitlab.com/ClimateImpactLab/Climate/climate_transform_specs
# /project/cil/home_dirs/jonahmgilbert/archive/data/agglomerated-world-new_BCSD_grid_segment_weights_area_pop.csv
IR_TABLE = DATA / "phl_deu_ir_segment_weights.parquet"
# the library's own impact region weights, and the config that makes them
# Source: generated with this library from the world-combo-201710
# impact region shapefile and the GPW v4 rev 10 population raster,
# adjusted to UN World Population Prospects. Not published.
IR_WEIGHTS_CONFIG = HERE / "grid025_to_ir_phl_deu.toml"
IR_WEIGHTS = DATA / "weights" / "grid025_ir_phl_deu_per_destination"
# outlines for the maps, simplified at 0.01 degrees
# Source: GADM 2.0, https://gadm.org/, dissolved to ADM1 with
# examples/gadm20/prepare_gadm20_targets.py
PROVINCE_OUTLINES = DATA / "phl_deu_adm1_plot.parquet"
# Source: the world-combo-201710 impact region shapefile,
# https://doi.org/10.5281/zenodo.21934131
IR_OUTLINES = DATA / "phl_deu_ir_plot.parquet"


def daily_path(var: str, iso: str) -> Path:
    """One country's daily file for tasmax or tasmin.

    Source: NASA NEX-GDDP, BCSD-downscaled CMIP5 daily,
    https://www.nccs.nasa.gov/nex-gddp/
    """
    return DATA / f"{var}_day_BCSD_rcp85_r1i1p1_{MODEL}_{YEAR}_{iso.lower()}.nc"


def days_in_file(var: str, iso: str) -> int:
    """How many days the file holds, before any reader touches it."""
    with netCDF4.Dataset(daily_path(var, iso)) as ds:
        return len(ds.dimensions["time"])


def province_outlines() -> gpd.GeoDataFrame:
    """The provinces of both countries, keyed by ISO and ID_1."""
    return gpd.read_parquet(PROVINCE_OUTLINES)


def impact_region_outlines() -> gpd.GeoDataFrame:
    """The impact regions of both countries, keyed by hierid.

    An ISO column, the first three letters of the hierid, says which
    country each one is in.
    """
    shapes = gpd.read_parquet(IR_OUTLINES)
    return shapes.assign(ISO=shapes["hierid"].str[:3])
