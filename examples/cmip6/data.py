"""Where the example's files are, and what they hold."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

GRID = "grid-0.25deg-center-[-180,180)"

# GADM 2.0 ID_1 of the four Peruvian provinces
PROVINCES = {25: "Ucayali", 17: "Madre de Dios", 20: "Piura", 8: "Cusco"}

# three of the twenty ssp370 models: the coolest, one near the middle,
# and the hottest that is not under a share-alike licence, ranked by
# Ucayali's days above 95F in the product (see SUMMARY). CanESM5, the
# hottest, is listed in cil-gdpcir-cc-by-sa as well and is left out.
# Source: GDPCIR on the Microsoft Planetary Computer, the collection
# and licence of each model as its STAC item states them.
MODELS = {
    "FGOALS-g3": {"collection": "cil-gdpcir-cc0", "license": "CC0-1.0"},
    "MIROC6": {"collection": "cil-gdpcir-cc-by", "license": "CC-BY-4.0"},
    "UKESM1-0-LL": {"collection": "cil-gdpcir-cc-by", "license": "CC-BY-4.0"},
}
YEARS = {"historical": 2000, "ssp370": 2090}

# the rows of the production weight tables for the four provinces:
# grid cells to GADM 2.0 ADM1, to ADM2, and to impact regions
# (hierids starting PER.<ID_1>.), population and area weights
# Source: /project/cil/gcp/impact_map_inputs/weights/<name>/
WEIGHTS = {
    "impact_region": DATA / "weights" / "grid025_ir_per_destination",
    "adm2": DATA / "weights" / "grid025_gadm20_adm2_per_destination",
    "adm1": DATA / "weights" / "grid025_gadm20_adm1_per_destination",
}

# days above 95F per province, ssp370 2080-2099, population weighted,
# for all twenty models, as the climate product holds them
# Source: /project/cil/gcp/climate/CMIP6_GDPCIR/gadm20/daily/days_<model>_ssp370.parquet
SUMMARY = DATA / "ssp370_days_above_95F_2080_2099.csv"

# outlines for the maps, simplified at 0.01 degrees
# Source: GADM 2.0, https://gadm.org/, as the target layers
#   /project/cil/gcp/impact_map_inputs/targets/gadm20/{adm1,adm2}.parquet,
#   and the world-combo-201710 impact regions,
#   https://doi.org/10.5281/zenodo.21934131
OUTLINES = {level: DATA / f"outlines_{level}.parquet"
            for level in ("impact_region", "adm2", "adm1")}


def daily_path(model: str, experiment: str) -> Path:
    """Daily tasmax and tasmin for one model-year, time by cell.

    Only the cells the four provinces' weights use are stored, with
    their cell_ix and cell_iy on GRID.
    Source: GDPCIR daily, https://planetarycomputer.microsoft.com/dataset/group/cil-gdpcir
    """
    return DATA / "daily" / f"{model}_{experiment}_{YEARS[experiment]}.nc"


def monthly_path(model: str, experiment: str) -> Path:
    """Monthly tavg for one model-year, on the box around the provinces.

    Cells outside the four provinces are blank.
    Source: /project/cil/gcp/climate/CMIP6_GDPCIR/monthly/tavg, CIL's
    monthly means of the same daily (tasmax + tasmin) / 2
    """
    return DATA / "monthly" / f"{model}_{experiment}_{YEARS[experiment]}.nc"
