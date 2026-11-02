"""Where the example's files are, and the scenario they hold."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

FUEL = "electricity"
RCP, IAM, SSP = "rcp85", "low", "SSP3"
WINDOWS = [(2020, 2039), (2080, 2099)]
# the percentiles are for these two; the map is for the other two
COUNTRIES = {"QAT": "Qatar", "ISL": "Iceland"}
MAP_COUNTRIES = {"CHL": "Chile", "AUT": "Austria"}
STEM = f"FD_FGLS_inter_OTHERIND_{FUEL}_TINV_clim"

# the delta-method leaves of every climate projection, full adaptation and
# histclim, cut to the eight impact regions of Qatar and Iceland, and the
# median leaves of the same runs, cut to the regions of all four
# countries; both cut to the years of the two windows
# Source: /project/cil/gcp/outputs_newsocioeconomics/energy/electricity/
#   delta_method/median/rcp85/{gcm}/low/SSP3/ and median/rcp85/{gcm}/low/SSP3/
GRADIENTS = DATA / "delta_method"
MEDIANS = DATA / "median"
# population per impact region and year, all four countries
# Source: /project/cil/gcp/outputs_newsocioeconomics/socioeconomics/1.0.0/
#   ir_combined_SSP3_IIASA.csv
POPULATION = DATA / "population.parquet"
# electricity price per country and year, 2005 USD per kWh, all four countries
# Source: /project/cil/gcp/social/baselines/energy/
#   IEA_Price_FIN_Clean_gr014_GLOBAL_COMPILE.dta
PRICE = DATA / "price.parquet"
# impact regions to GADM 2.0 provinces (adm1) and municipalities (adm2),
# per-source population weights, the rows for the regions of all four countries
# Source: https://doi.org/10.5281/zenodo.21934155
WEIGHTS = {"adm1": DATA / "weights" / "gadm20_adm1_per_source",
           "adm2": DATA / "weights" / "gadm20_adm2_per_source"}
# SMME weights of the climate projections for RCP8.5
# Source: /project/cil/gcp/climate/BCSD/SMME/SMME-weights/rcp85_SMME_weights.tsv
SMME = DATA / "smme_weights_rcp85.csv"
# outlines of Chile and Austria for the map, without islands under
# 50 km2 and simplified at 0.01 degrees
# Source: GADM 2.0, https://gadm.org/, as the target layers
#   /project/cil/gcp/impact_map_inputs/targets/gadm20/{adm0,adm1,adm2}.parquet
#   made by examples/gadm20/prepare_gadm20_targets.py, and the
#   world-combo-201710 impact regions, https://doi.org/10.5281/zenodo.21934131
OUTLINES = {level: DATA / f"outlines_{level}.parquet"
            for level in ("adm0", "impact_region", "adm1", "adm2")}


def gcms() -> list[str]:
    """The climate projections with leaves in data/."""
    return sorted(p.name for p in GRADIENTS.iterdir() if p.is_dir())


def gradient_path(gcm: str, histclim: bool = False) -> Path:
    return GRADIENTS / gcm / f"{STEM}{'-histclim' if histclim else ''}.nc4"


def median_path(gcm: str, histclim: bool = False) -> Path:
    return MEDIANS / gcm / f"{STEM}{'-histclim' if histclim else ''}.nc4"
