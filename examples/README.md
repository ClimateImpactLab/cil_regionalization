# Examples

- `aggregation/` fetches published weights and aggregates real Monte
  Carlo output for Colombia to departments and municipalities, both
  variable kinds and both weight directions. The place to start.
- `gadm20/` prepares the GADM 2.0 target layers and holds the
  configurations that generated the published GADM 2.0 weight files.
- `gadm41/` the same for GADM 4.1.
- `montecarlo/` the Monte Carlo pipeline: an annotated example config,
  the scheduler template, and under `runs/` the configurations behind
  the published mortality aggregations (cluster specific).
- `synthetic/` regenerates the committed synthetic test fixtures.
