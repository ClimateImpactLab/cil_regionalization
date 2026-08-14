# climate-and-damages-aggregations

Climate impact results come at impact region level; analyses need
them at country, state, or district level. This does that step:
published weight files describe how impact regions overlap
administrative units, and the library applies them.

It also generates new weight files between any two sets of geometries,
for cases the published ones do not cover.

## The short version

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

`my_draws` is a long table: one row per impact region and year (and
batch, model, whatever else), with the region id in a `hierid` column.
`fetch_weights` downloads the weight file from its Zenodo record,
checks it against the checksum recorded when it was generated, caches
it, and hands it back ready to use. The other two lines aggregate and
then summarize.

The same flow works from the command line: `segweights fetch <name>`
downloads and prints the cached path, `segweights cache list` and
`segweights cache clear` manage the cache, and `segweights pipeline
<config.toml>` runs a whole Monte Carlo tree from one config file.

## Which weight file, which kind

One question decides the pairing: does adding the variable across
regions produce a meaningful total?

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

The library checks the declared `kind` against the weight file's recorded
direction and refuses a mismatch, rather than producing a plausible
wrong number.

The two directions exist because a weight has to answer one of two
different questions. When an impact region straddles two departments,
either the region's total is being split between them, in which case
its shares must sum to one across the departments, or department
values are being averaged over the regions inside them, in which case
the shares must sum to one within each department. Same intersections,
different normalization, and a file built one way cannot serve the
other job.

## A complete worked example

`examples/colombia/` aggregates real Monte Carlo output for Colombia
to departments and municipalities, both variable kinds and both
directions, from a small extract committed in the repository (about
1.6 MB). Its README shows the printed output, so what comes out is readable
without running it. Start there.

## Installation

Clone the repository and install it like any Python package:

```
pip install .                # library and CLI
pip install '.[netcdf]'      # also read NetCDF Monte Carlo trees
```

or `uv pip install .` in a uv environment. The geospatial stack
(geopandas, shapely, rasterio, exactextract) installs with it; all of
these ship wheels on Linux and macOS. Add `[netcdf]` to read NetCDF
Monte Carlo trees, which the production mortality and labor trees are.
`[bigquery]` adds the BigQuery generation backend, `[dev]` adds pytest.
Python 3.10 or newer.

The `envs/` directory and `examples/montecarlo/production/` exist to
reproduce our production runs on the University of Chicago RCC cluster;
they are not part of installing or using the package.

## Generating weights

A TOML config names the target geometry, the source geometry or grid,
the weighting rasters, and the normalization direction. `segweights
run <config.toml>` writes the weights parquet plus a manifest that
records where everything came from, with checksums. `segweights
validate <config.toml>` checks a config without computing anything.
Worked configurations, including the GADM target preparation and the
full production runs, live under `examples/`.

## Where the detail lives

`docs/how-it-works.md` is one page: the pipeline as a diagram, the
mathematics in three formulas, and what a caller supplies.
`fetch_weights`, `apply_weights`, `summarize_samples`,
`WeightsArtifact`, and `load_config` are the supported API, re-exported
at the package root; everything else is internal and may change.
