# climate-and-damages-aggregations

Climate impact results come at impact region level, but most analyses
need them at country, state, or district level. This package does that
aggregation step. Published weight files describe how impact regions
overlap administrative units; the library downloads them, applies them
to your data, and computes statistics. It can also generate new weight
files between any two sets of geometries when the published ones do
not cover your case.

## Quick example

```python
from segment_weights import fetch_weights, apply_weights, summarize_samples

artifact = fetch_weights("gadm20-adm1-per-source")
result = apply_weights(
    artifact, my_draws,
    kind="extensive",
    weight="pop",
    data_version="world-combo-201710",
)
stats = summarize_samples(
    result.frame, sample_dims=["batch", "gcm"],
    time_col="year", window=(2080, 2099), quantiles=[0.05, 0.5, 0.95],
)
```

`my_draws` is a long table: one row per impact region and year (plus
batch, model, and whatever other columns your sample has), with the
region id in a `hierid` column. `fetch_weights` downloads the weight
file from its Zenodo record, checks it against the checksum recorded
when it was generated, caches it under `~/.cache/segment_weights`, and
returns it ready to use. The other two calls aggregate and then
summarize.

The same works from the command line: `segweights fetch <name>`
downloads and prints the cached path, `segweights cache list` and
`segweights cache clear` manage the cache, and `segweights pipeline
<config.toml>` runs a whole Monte Carlo tree from one config file.

## How it works

![The pipeline: geometries and rasters go into weight generation,
which produces weight files; weight files plus impact region draws go
through application, then statistics, then results.](imgs/pipeline.svg)

The top lane happens once per pair of geometries: intersect the
administrative boundaries, the impact regions, and a population
raster, and record what share of each intersection carries. The result
is a weights parquet plus a manifest that says exactly what produced
it, published on Zenodo. The bottom lane is what you do day to day:
fetch the weight file, apply it to your Monte Carlo draws, compute
statistics at the end.

Both aggregation operations are the same multiply and add; what
differs is which way the weights are normalized. For a total (dollars,
deaths), each region's value is split between the units it touches,
and the pieces must add back up:

    value(u) = sum over r of w(r, u) * value(r),
    where the w(r, u) for one region r sum to 1

For an average (a death rate, a temperature), each unit's value is the
population weighted mean over the regions inside it:

    value(u) = sum over r of w(r, u) * value(r),
    where the w(r, u) for one unit u sum to 1

Order matters for quantiles, because quantiles do not pass through
sums: the 95th percentile of a sum is not the sum of 95th percentiles.
Adding up per-region quantiles assumes every region has its worst draw
at the same time, which overstates the tails. So the pipeline
aggregates each Monte Carlo draw separately and takes quantiles only
at the end, and the statistics stage rejects input that already
contains quantiles.

To use published weights you provide three things: your draws in the
long form above, the `kind` of your variable, and `data_version`, the
impact region vintage your data was built on. If the vintage does not
match what the weight file records, `apply_weights` raises an error
before producing any numbers, because weights and data built on
different geometries silently misallocate.

## Choosing the right type of weights

Ask one question: does adding your variable across regions produce a
meaningful total?

Dollars, deaths, people, tons of crops: yes. Those are totals, and a
region's total gets split between the administrative units it touches.
Use a `per_source` weight file with `kind="extensive"`.

Death rates, temperatures, percent of GDP: no. Adding rates across
regions gives nonsense. Those are averages, and an administrative
unit's value is the average over the regions inside it. Use a
`per_destination` weight file with `kind="intensive"`.

| Variable | Weight file | kind | Why |
| --- | --- | --- | --- |
| Dollars, deaths, counts | `...-per-source` | `extensive` | A total is split between the units it touches |
| Rates, temperatures, shares | `...-per-destination` | `intensive` | An average is taken over the regions in each unit |
| A ratio needed at the target level, like percent of GDP | `...-per-source` | `ratio` | Sum the numerator and denominator separately, divide after |

The two directions exist because a weight answers one of two different
questions. When an impact region straddles two departments, either you
are splitting that region's total between them, so its shares must sum
to one across the departments, or you are averaging over the regions
that make up one department, so the shares must sum to one within the
department. Same intersections, different normalization, and a file
built one way cannot do the other job. The library checks the `kind`
you declare against the direction the weight file records and raises
an error on a mismatch, since getting this wrong by hand produces
plausible wrong numbers.

## Worked example

`examples/aggregation/` aggregates a small sample of real Monte Carlo
mortality projections (not the complete output: two draws, one climate
model) for Colombia, to departments and municipalities, covering both
variable kinds and both weight directions. It fetches the published
weights by name and reads a committed 1.6 MB sample, so it runs in
seconds. Its README shows the full printed output, so you can see what
comes out without running anything. Start there.

## Installation

Straight from GitHub:

```
pip install "git+https://github.com/ClimateImpactLab/REPOSITORY.git"
pip install "segment_weights[netcdf] @ git+https://github.com/ClimateImpactLab/REPOSITORY.git"
```

(the URL is a placeholder until the repository moves to the Climate
Impact Lab organisation), or from a clone:

```
pip install .                # library and CLI
pip install '.[netcdf]'      # also read NetCDF Monte Carlo trees
```

`uv pip install .` works the same way in a uv environment. The
geospatial stack (geopandas, shapely, rasterio, exactextract) installs
with it; all of these ship wheels on Linux and macOS. Add `[netcdf]`
to read NetCDF Monte Carlo trees, which the full mortality and labor
trees are. `[bigquery]` adds the BigQuery generation backend, `[dev]`
adds pytest. Python 3.10 or newer.

The `envs/` directory and `examples/montecarlo/runs/` exist to
reproduce the runs behind the published results on the University of
Chicago RCC cluster; they are not part of installing or using the
package.

## Generating new weights

A TOML config names the target geometry, the source geometry or grid,
the weighting rasters, and the normalization direction. `segweights
run <config.toml>` writes the weights parquet plus a manifest that
records where everything came from, with checksums. `segweights
validate <config.toml>` checks a config without computing anything.
Worked configurations, including the GADM target preparation and the
full generation runs, live under `examples/`.

Weight files built on different target vintages are not
interchangeable. GADM 2.0 and GADM 4.1 have different unit universes
and different keys, and results built on one cannot be compared with
results built on the other. Each vintage is published as its own
record, and every manifest states which vintage it was built against.

The published weights were built against two GADM vintages, recorded
in every manifest by their vintage labels: GADM 2.0 (label
`gadm-2.0-impactmap-copy-2025`, a processed copy of the combined GADM
2.0 shapefile) and GADM 4.1 (label `gadm-4.10-impactmap-copy-2022`,
the gadm_410.gpkg GeoPackage as distributed by GADM). You can obtain
the same versions from GADM directly. The dissolved target layers are
large (840 MB for GADM 2.0, about 2 GB for GADM 4.1) and are not
distributed, since GADM restricts redistribution; regenerate them with
the dissolve scripts under `examples/gadm20/` and `examples/gadm41/`.
The targets manifest records the source file's checksum, so you can
confirm a regenerated set started from the same input.

## More detail

`fetch_weights`, `apply_weights`, `summarize_samples`,
`WeightsArtifact`, and `load_config` are the supported API, re-exported
at the package root. Everything else is internal and may change.
