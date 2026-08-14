# Examples

- `aggregation/` is the place to start: it fetches the published
  weights and aggregates a small sample of real Monte Carlo mortality
  projections for Colombia, to departments and municipalities, both
  variable kinds and both weight directions. The notebook there maps
  the result; the script runs the full case from the command line.
- `gadm20/` prepares the GADM 2.0 target layers and holds the
  configurations that generated the published GADM 2.0 weight files.
- `gadm41/` does the same for GADM 4.1.
- `montecarlo/` covers the Monte Carlo pipeline: an annotated example
  config, the scheduler template, and under `runs/` the configurations
  behind the published mortality aggregations (cluster specific).
- `synthetic/` regenerates the committed synthetic test fixtures.
