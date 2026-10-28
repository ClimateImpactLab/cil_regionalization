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
  year of raw BCSD tasmax and tasmin for Indonesia and Germany, on a
  Gregorian calendar the reader brings down to 365 days, turned into
  heating and cooling degree days per grid cell and aggregated to
  provinces with the grid weights, under population and area
  weighting. The same code gives the two climates opposite results,
  which is how the numbers check themselves. It also reads the same
  weights from a plain CSV with `from_table` and shows the two routes
  agree. The impact region level uses an outside table: one row per
  grid cell and impact region, shares already normalized, provenance
  not recorded in the file, included to show reading an outside table.
  Running the notebook needs `jupyter` and `matplotlib` beyond the
  `[netcdf]` extra; there is no separate extra for those.
- `gadm20/` prepares the GADM 2.0 target layers and holds the
  configurations that generated the published GADM 2.0 weight files.
- `gadm41/` does the same for GADM 4.1.
- `montecarlo/` covers the Monte Carlo pipeline: an annotated example
  config, the scheduler template, and under `runs/` the configurations
  behind the published mortality aggregations (cluster specific).
- `synthetic/` regenerates the committed synthetic test fixtures.
