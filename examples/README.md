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
- `gadm20/` prepares the GADM 2.0 target layers and holds the
  configurations that generated the published GADM 2.0 weight files.
- `gadm41/` does the same for GADM 4.1.
- `montecarlo/` covers the Monte Carlo pipeline: an annotated example
  config, the scheduler template, and under `runs/` the configurations
  behind the published mortality aggregations (cluster specific).
- `synthetic/` regenerates the committed synthetic test fixtures.
