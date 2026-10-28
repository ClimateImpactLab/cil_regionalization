# Climate example

This example takes daily temperatures from the 0.25° BCSD grid, calculates degree days and days above 35°C, and aggregates the results to provinces (GADM 2.0 ADM1) and impact regions in the Philippines and Germany.

To run it, you'll need the data/ folder (about 5 MB). Everything runs locally, so you don't need cluster access. The outputs are already saved in the notebook, so you can also follow along without running the code.

## Run it

Install the library:

```bash
git clone https://github.com/ClimateImpactLab/cil_regionalization.git
cd cil_regionalization
pip install '.[netcdf]' jupyter matplotlib
```

- quickstart:
    ```bash
    jupyter lab examples/climate/quickstart.ipynb
    ```

- full example:
    ```bash
    jupyter lab examples/climate/degree_days_from_daily_grid.ipynb
    ```

## Files

- `quickstart.ipynb` walks through a simple example in six steps: calculating cooling degree days and days above 35°C for the Philippines, then aggregating the results to impact regions.
- `degree_days_from_daily_grid.ipynb` is the full example. It defines a transform, converts the grid to cells, and aggregates to provinces and impact regions, comparing two sets of impact region weights side by side.
- `data.py` contains the file paths, and `plots.py` draws the maps. Neither performs the degree-day calculations or aggregation.
- `prepare_data.py` rebuilds the `data/` folder. You'll need access to the CIL filesystem on Midway3 to run it; its docstring lists the source files.
- `grid025_to_ir_phl_deu.toml` generates the impact region weights for the Philippines and Germany. Run it on Midway3 with:

  ```bash
  cilreg run examples/climate/grid025_to_ir_phl_deu.toml
  ```

- Weights covering all impact regions are also available on Midway3 at `/project/cil/gcp/impact_map_inputs/weights/grid025_ir_per_destination`. These are generated using `weights/configs/grid025_to_ir_per_destination.toml` in [impactmap-data-gen](https://github.com/ClimateImpactLab/impactmap-data-gen).

To use other weights, see “The same weights, from a CSV” and “Impact regions” in the full notebook. `from_table` accepts a DataFrame or a path to a `.csv`, `.csv.gz`, or `.parquet` file.

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
