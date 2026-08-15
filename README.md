# CIL-regionalization

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![GADM 2.0 weights](https://zenodo.org/badge/DOI/10.5281/zenodo.21934155.svg)](https://doi.org/10.5281/zenodo.21934155)
[![GADM 4.1 weights](https://zenodo.org/badge/DOI/10.5281/zenodo.21935431.svg)](https://doi.org/10.5281/zenodo.21935431)

Climate impact results come at impact region level, but most analyses
need them at country, state, or district level. This package does that
aggregation step. Published weight files describe how impact regions
overlap administrative units; the library downloads them, applies them
to your data, and computes statistics. It can also generate new weight
files between any two sets of geometries when the published ones do
not cover your case.

## Quick example

```python
import cil_regionalization as cilreg

weights = cilreg.fetch_weights("gadm20-adm1-per-source")
result = cilreg.apply_weights(
    weights,
    my_draws,
    kind="extensive",
    weight="pop",
    data_version="world-combo-201710",
    # drop this line if the draws cover every impact region
    restrict_to_sources={(h,) for h in my_draws["hierid"].unique()},
)
stats = cilreg.summarize_samples(
    result.frame,
    sample_dims=["batch", "gcm"],
    time_col="year",
    window=(2080, 2099),
    quantiles=[0.05, 0.5, 0.95],
)
```

`my_draws` is a long table: one row per impact region and year (plus
batch, model, and whatever other columns your sample has), with the
region id in a `hierid` column and the numbers in a `value` column
(another name works via `value_col=`). `restrict_to_sources` limits
the aggregation to the regions the draws cover; leave it out only when
they cover all 24,378. `data_version` names the impact region set the
draws are keyed to; draws built on the published shapefile record
below use `world-combo-201710`. A fetched weight file shows the
version it expects in its `source_version`. `fetch_weights` downloads the
weight file from its Zenodo record, checks it against the checksum
recorded when it was generated, caches it under
`~/.cache/cil_regionalization`, and returns it ready to use.

The same works from the command line: `cilreg fetch --list` shows
every published weight file name, `cilreg fetch <name>`
downloads and prints the cached path, `cilreg cache list` and
`cilreg cache clear` manage the cache, and `cilreg pipeline
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
impact region version your data was built on. If the version does not
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

## Reading the Monte Carlo trees

`read_netcdf_leaf` flattens one projection file to the long form
`apply_weights` takes. One caution about what those files contain: in
the mortality and labor trees, a file named `...-combined.nc4` stores
`rebased`, the scenario's impact relative to its own 2001 to 2010
average. That value still contains the scenario's income and
adaptation trend, so aggregating it gives impact levels, not the
effect of climate change. The effect of climate change under full or
income adaptation is the rebased impact minus the rebased impact of
the `-histclim` sibling file, which resamples historical weather under
the same income growth and adaptation. The no adaptation scenario is
the exception: it has no income growth, so nothing is subtracted and
its rebased value stands alone. Nothing in the file names or the
variable metadata says any of this; the convention comes from the
Climate Impact Lab memo "The art of rebasing and histclim", and the
Mexico example below shows the subtraction on real files.

## Worked example

`examples/aggregation/` aggregates a small sample of real Monte Carlo
mortality projections (not the complete output: one Monte Carlo batch
out of fifteen, across all 33 climate models) for Colombia, fetching
the published weights by name and reading a committed 23 MB sample. Start with the notebook,
`aggregation_colombia.ipynb`: it maps a single draw and a pooled
quantile side by side, shows percent of GDP with the `ratio` kind, and
plots the spread of draws behind the statistics. Its outputs are
committed, so it reads on GitHub without running anything; running it
needs `jupyter` and `matplotlib`. The script `run_example.py` covers the same case plus
municipalities and rates from the command line, and its README shows
the full printed output. A second example, `examples/rates/`,
aggregates the effect of climate change on Mexican mortality rates to
municipalities on both GADM boundary versions, computing it as full
adaptation minus histclim before aggregating and using the ratio kind
to keep the rates scenario consistent.

## Installation

Straight from GitHub:

```
pip install "git+https://github.com/ClimateImpactLab/cil_regionalization.git"
pip install "cil_regionalization[netcdf] @ git+https://github.com/ClimateImpactLab/cil_regionalization.git"
```

or from a clone:

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
the weighting rasters, and the normalization direction. `cilreg
run <config.toml>` writes the weights parquet plus a manifest that
records where everything came from, with checksums. `cilreg
validate <config.toml>` checks a config without computing anything.
Worked configurations, including the GADM target preparation and the
full generation runs, live under `examples/`.

Weight files built on different target geometry versions are not
interchangeable. GADM 2.0 and GADM 4.1 have different unit universes
and different keys, and results built on one cannot be compared with
results built on the other. Each geometry version is published as its own
record, and every manifest states which version it was built against.

Three records are published on Zenodo:

- the impact region shapefile, world-combo-201710
  (doi:10.5281/zenodo.21934131)
- GADM 2.0 weights (doi:10.5281/zenodo.21934155):
  `gadm20-adm1-per-source`, `gadm20-adm1-per-destination`,
  `gadm20-adm2-per-source`, `gadm20-adm2-per-destination`
- GADM 4.1 weights (doi:10.5281/zenodo.21935431), keyed by GADM's GID
  codes: `gadm41-adm1-per-source`, `gadm41-adm1-per-destination`,
  `gadm41-adm2-per-source`, `gadm41-adm2-per-destination`.
  GADM 4.1 draws
  coastlines differently from the impact region geometry, so 1,178
  coastal regions are partially covered; the manifests record this,
  and `apply_weights` needs `allow_partial_coverage=True` to proceed

The weights were built against two GADM versions, recorded in every
manifest by their version labels: GADM 2.0 (label
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
