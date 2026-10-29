# Examples

- `aggregation/` downloads the published weights and aggregates a small sample of real Monte Carlo mortality projections for Colombia, measured in dollars. It covers extensive and ratio aggregation, both weight directions, and downloading weights. The notebook maps the results, and the script runs the full example from the command line.
- `rates/` aggregates mortality rates for Mexico to municipalities using both GADM 2.0 and GADM 4.1. It shows how to aggregate rates, what changed between the two boundary versions, and which units can be compared. An interactive map shows the results for both versions.
- `climate/` uses daily BCSD temperatures for the Philippines and Germany to calculate degree days and days above 35°C, then aggregates them to provinces and impact regions. It includes a short quickstart and a longer notebook explaining each step. The README also has a snippet that counts days above 35°C over the full grid and aggregates them to all impact regions with the segment weights.
- `gadm20/` prepares the GADM 2.0 target layers and contains the configurations used to generate the published GADM 2.0 weights.
- `gadm41/` does the same for GADM 4.1.
- `montecarlo/` contains an example configuration with explanations, a scheduler template, and the configurations used for the published mortality aggregations under `runs/`. These configurations depend on the cluster setup.
- `synthetic/` regenerates the synthetic data files used in the tests and stored in the repository.
