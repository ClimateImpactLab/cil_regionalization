# Examples

- `aggregation/` downloads the published weights and aggregates a small sample of real Monte Carlo mortality projections for Colombia, measured in dollars. It covers extensive and ratio aggregation, both weight directions, and downloading weights. The notebook maps the results, and the script runs the full example from the command line.
- `rates/` aggregates mortality rates for Mexico to municipalities using both GADM 2.0 and GADM 4.1. It shows how to aggregate rates, what changed between the two boundary versions, and which units can be compared. An interactive map shows the results for both versions.
- `climate/` is the CMIP5 climate example. It uses daily BCSD temperatures for the Philippines and Germany to calculate degree days and days above 35°C, then aggregates them to provinces and impact regions. It includes a short quickstart and a longer notebook explaining each step. The README also has a snippet that counts days above 35°C over the full grid and aggregates them to all impact regions with the segment weights.
- `cmip6/` is the CMIP6 climate example. It counts days above 95F for four Peruvian provinces from GDPCIR daily tasmax, daily tavg and monthly tavg, and shows that only the first is the count, even where another lands close. It aggregates with the production weight tables at impact region, ADM2 and ADM1, and compares three models with the spread of all twenty from the climate tables.
- `energy_percentiles/` takes the delta-method gradients of the electricity projections for Qatar and Iceland, values them in dollars, aggregates them to provinces and countries, and mixes 33 climate projections into percentiles. It shows why the variance of a total has to be taken at the target, and maps the change per person in Chile and Austria at four levels.
- `gadm20/` prepares the GADM 2.0 target layers and contains the configurations used to generate the published GADM 2.0 weights.
- `gadm41/` does the same for GADM 4.1.
- `montecarlo/` contains an example configuration with explanations, a scheduler template, and the configurations used for the published mortality aggregations under `runs/`. These configurations depend on the cluster setup.
- `synthetic/` regenerates the synthetic data files used in the tests and stored in the repository.

Six of the seven notebooks run with no network and nothing cached: the
two in `climate/` and the ones in `energy/`, `energy_percentiles/`,
`labor/` and `rates/` read everything from their `data/` directories.
The `aggregation/` notebook needs the network, because it shows
`fetch_weights` downloading the GADM weights from Zenodo; later runs
read them from the cache. Its `run_example.py --offline` uses the
committed weight slices instead. This was run, not assumed: on 3
November 2026 all seven notebooks ran in a clone with outside network
blocked and `CIL_REGIONALIZATION_CACHE` pointing at an empty
directory. Six finished and left the cache empty; the aggregation
notebook stopped at its first `fetch_weights` call. The `cmip6/`
notebook, added later, reads only its `data/` directory too. It was
run the same way the same day, finished, and left the cache empty.
