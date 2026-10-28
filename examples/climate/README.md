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

For other weights, see "The same weights, from a CSV" and "Impact
regions" in the notebook. `from_table` takes a DataFrame or a path to
a `.csv`, `.csv.gz` or `.parquet`.
