# Examples

- `aggregation/` is the introduction and the place to start: it
  fetches the published weights and aggregates a small sample of real
  Monte Carlo mortality projections for Colombia, in dollars, covering
  the extensive and ratio kinds, both weight directions, and the fetch
  flow. The notebook maps the results; the script runs the full case
  from the command line.
- `rates/` is the second, more detailed example: physical mortality
  rates for Mexico, aggregated to municipalities under both GADM 2.0
  and GADM 4.1. It shows the ratio route for rates, what the boundary
  revision changed, results under both versions with an interactive
  map, and which units the two versions can and cannot compare.
- `climate/` starts one step earlier, from gridded daily climate: one
  year of raw BCSD tasmax and tasmin for the Philippines and Germany,
  on a Gregorian calendar the reader brings down to 365 days, turned into
  heating and cooling degree days per grid cell and aggregated to
  provinces with the grid weights, under population and area
  weighting. The same code gives the two climates opposite results.
  It also reads the same weights from a plain CSV with `from_table`
  and shows the two give the same numbers. The impact region level
  uses segment weights read the same way: one row per grid cell and
  impact region, with shares that already add up to one per region.
  Source: weights used for aggregation on the Berkeley servers,
  https://gitlab.com/ClimateImpactLab/Climate/climate_transform_specs
  /project/cil/home_dirs/jonahmgilbert/archive/data/agglomerated-world-new_BCSD_grid_segment_weights_area_pop.csv
  A second transform, days with a high above 35 C, goes through the
  same weights. `grid025_to_ir_phl_deu.toml` generates the library's
  own impact region weights for the two countries, and the notebook
  compares the degree days each set gives. It also groups the daily
  highs by month and by season (JJA and DJF) from the grid's dates.
  Every step is written out in the notebook; `data.py`
  holds the file paths and `plots.py` the maps.
  Running the notebook needs `jupyter` and `matplotlib` beyond the
  `[netcdf]` extra; there is no separate extra for those.
- `gadm20/` prepares the GADM 2.0 target layers and holds the
  configurations that generated the published GADM 2.0 weight files.
- `gadm41/` does the same for GADM 4.1.
- `montecarlo/` covers the Monte Carlo pipeline: an annotated example
  config, the scheduler template, and under `runs/` the configurations
  behind the published mortality aggregations (cluster specific).
- `synthetic/` regenerates the committed synthetic test fixtures.
