# Climate example

Daily temperature on the 0.25 degree BCSD grid, turned into degree
days and days above 35 C, aggregated to provinces (GADM 2.0 ADM1) and
to impact regions, for the Philippines and Germany.

Everything the notebook needs is under `data/`, about 5 MB, so it runs
from a clone with no cluster access. The outputs are saved in the
notebook, so it can also be read without running anything.

## Run it

    git clone https://github.com/ClimateImpactLab/cil_regionalization.git
    cd cil_regionalization
    pip install '.[netcdf]' jupyter matplotlib
    jupyter lab examples/climate/degree_days_from_daily_grid.ipynb

The `[netcdf]` extra is what reads the daily files; `jupyter` and
`matplotlib` are not declared by the project and have to be added.
See the top-level README for the library itself.

## Files

- `quickstart.ipynb` is the shortest run through the same data: one
  country, cooling degree days, impact regions, in six steps.
- `degree_days_from_daily_grid.ipynb` is the example. It writes out a
  transform, turns the grid into cells, and aggregates to both
  levels, with two sets of impact region weights side by side.
- `data.py` holds the file paths and `plots.py` draws the maps.
  Neither computes anything.
- `prepare_data.py` rebuilds `data/`. This one needs the CIL
  filesystem on Midway3; its docstring lists what it reads.
- `grid025_to_ir_phl_deu.toml` generates the impact region weights,
  also from Midway3:

      cilreg run examples/climate/grid025_to_ir_phl_deu.toml

- On Midway3, the same weights for every impact region are at
  `/project/cil/gcp/impact_map_inputs/weights/grid025_ir_per_destination`,
  generated from `weights/configs/grid025_to_ir_per_destination.toml`
  in [impactmap-data-gen](https://github.com/ClimateImpactLab/impactmap-data-gen).

For other weights, see "The same weights, from a CSV" and "Impact
regions" in the notebook. `from_table` takes a DataFrame or a path to
a `.csv`, `.csv.gz` or `.parquet`.

## Global example

On Midway3, with the full files and the full segment weights:

```python
import numpy as np
import pandas as pd
import cil_regionalization as cilreg
from cil_regionalization.bcsd_io import read_bcsd_daily, BCSD_GRID

SRC = "/project/cil/gcp/climate/source_data/NASA/BCSD/rcp85/day/atmos"
SEGMENTS = ("/project/cil/home_dirs/jonahmgilbert/archive/data/"
            "agglomerated-world-new_BCSD_grid_segment_weights_area_pop.csv")

tasmax = read_bcsd_daily(
    f"{SRC}/tasmax/r1i1p1/v1.0/tasmax_day_BCSD_rcp85_r1i1p1_ACCESS1-0_2020.nc",
    "tasmax", require_full_year=True)

hot = (tasmax.values.astype("float64") - 273.15 > 35.0).sum(axis=0).astype("float64")
hot[np.isnan(tasmax.values).any(axis=0)] = np.nan   # a blank day leaves the cell blank
cells = tasmax.to_cells({"hot_days": hot})

table = pd.read_csv(SEGMENTS)
table["pix_cent_x"] = np.mod(table["pix_cent_x"] + 180.0, 360.0) - 180.0
table = (table.groupby(["hierid", "pix_cent_x", "pix_cent_y"], as_index=False)
         [["popwt"]].sum(min_count=1))
weights = cilreg.WeightsArtifact.from_table(
    table, id_fields=["hierid"], weights={"pop": "popwt"},
    source=BCSD_GRID)

out = cilreg.apply_weights(
    weights, cells, kind="intensive", weight="pop", value_col="hot_days",
    data_version=BCSD_GRID,
    policies=cilreg.SourceUnitPolicies(on_zero_weight="skip",
                                       on_unmatched="skip",
                                       on_absent_from_data="skip"))
print(f"{len(out.frame)} impact regions")
```
