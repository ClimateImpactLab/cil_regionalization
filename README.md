# CIL-regionalization

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![GADM 2.0 weights](https://zenodo.org/badge/DOI/10.5281/zenodo.21934155.svg)](https://doi.org/10.5281/zenodo.21934155)
[![GADM 4.1 weights](https://zenodo.org/badge/DOI/10.5281/zenodo.21935431.svg)](https://doi.org/10.5281/zenodo.21935431)
[![SMME weights](https://zenodo.org/badge/DOI/10.5281/zenodo.22003542.svg)](https://doi.org/10.5281/zenodo.22003542)

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
    quantile_method="linear",
    summation="fast",
)
```

`my_draws` is a long table: one row per impact region and year (plus
batch, model, and whatever other columns your sample has), with the
region id in a `hierid` column and the numbers in a `value` column
(another name works via `value_col=`). `restrict_to_sources` limits
the aggregation to the regions the draws cover; leave it out only when
they cover all 24,378. `data_version` names the impact region set the
draws are keyed to; draws built on the published shapefile record
below use `world-combo-201710`, and a mismatch with the weight file
raises before any numbers are produced. `fetch_weights` downloads the
file from its Zenodo record, verifies its checksum, and caches it
under `~/.cache/cil_regionalization`.

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

Quantiles do not pass through sums: the 95th percentile of a sum is
not the sum of 95th percentiles, because that would need every region
to hit its worst draw together. The pipeline aggregates each Monte
Carlo draw separately, takes quantiles at the end, and rejects input
that already contains quantiles.

## Choosing the right type of weights

Ask one question: does adding your variable across regions produce a
meaningful total?

| Variable | Weight file | kind | Why |
| --- | --- | --- | --- |
| Dollars, deaths, counts | `...-per-source` | `extensive` | A total is split between the units it touches |
| Rates, temperatures, shares | `...-per-destination` | `intensive` | An average is taken over the regions in each unit |
| A ratio needed at the target level, like percent of GDP | `...-per-source` | `ratio` | Sum the numerator and denominator separately, divide after |

A file built one way cannot do the other's job. The library checks
the `kind` you declare against the direction the file records and
raises on a mismatch.

## Reading the Monte Carlo trees

In the mortality, labor, and energy trees, the effect of climate
change is a file's `rebased` variable minus the same variable of its
`-histclim` sibling, whatever the file stem; `rebased` on its own is
not it. No adaptation is the exception and stands alone. The Climate
Impact Lab memo "The art of rebasing and histclim" has the reasoning.

Energy adds two traps of its own. The physical values are GJ per
capita, although the files say kWh/pc. The dollar `-price...-levels`
files understate by exactly 0.0036, because a GJ quantity was
multiplied by a per-kWh price; divide by 0.0036 to repair. Energy
also has no Monte Carlo draws: its uncertainty lives in delta-method
trees, where `rebased` holds the variance and `rebased_bcde` the
gradient per regression coefficient. Aggregate the gradients with the
weights and take one quadratic form against the embedded covariance
matrix (`aggregate_gradient` and `delta_method`); summing
independently drawn region values understates the aggregate spread.

## Climate model weights

Climate Impact Lab projections weight climate models by the SMME
weights. `summarize_samples` and `pooled_statistics` take
`weight_col`, a column with one weight per sample member. The default
is unweighted; use the weights when the numbers will sit next to
results from these projections.

The weight files are published as their own Zenodo record
(doi:10.5281/zenodo.22003542), one per RCP (`rcp45_SMME_weights.tsv`,
`rcp85_SMME_weights.tsv`; columns `quantile`, `model`, `weight`). To
apply them, normalize the model names and merge the weight onto every
draw of each model, so the weight repeats across batches:

```python
w = pd.read_csv(
    "https://zenodo.org/records/22003542/files/rcp85_SMME_weights.tsv?download=1",
    sep="\t",
)[["model", "weight"]]
w["key"] = w["model"].str.replace("*", "", regex=False).str.lower()
draws["key"] = (
    draws["gcm"].str.replace("surrogate_", "", regex=False).str.lower()
)
draws = draws.merge(w[["key", "weight"]], on="key").drop(columns="key")
stats = cilreg.summarize_samples(
    draws,
    sample_dims=["batch", "gcm"],
    time_col="year",
    window=(2080, 2099),
    quantiles=[0.05, 0.5, 0.95],
    weight_col="weight",
    quantile_method="lower_step",
    summation="fast",
)
```

The naming rule: weight file names are lowercase, a trailing `*` is
bookkeeping and drops out, and an underscore suffix (`gfdl-cm3_94`)
marks a surrogate, which the projection trees spell with a prefix
(`surrogate_GFDL-CM3_94`). Strip the prefix and lowercase, and every
model in the trees matches exactly one weight row.

`read_model_weights` applies that rule to a file on disk. It reads the
file once and refuses a table in which two rows match the same model.

```python
weights = cilreg.read_model_weights("rcp85_SMME_weights.tsv")
draws["weight"] = weights.attach(draws["gcm"])
entry = weights.record(leaf_models)  # one model name per leaf pooled
```

`attach` raises if a model has no row. `record` returns what a manifest
should keep about the run: the file's path and SHA256, each model that
was given a weight with that weight and its number of leaves, and the
rows no model matched with their weights. The two lists together are
the whole table, so the statistics can be rerun from the manifest if
the file is lost. A pipeline run writes this entry to its manifest as
`model_weights_record`.

Weighted quantiles follow the historical extraction tool: the left
step inverse of the weighted empirical distribution, not the
interpolated quantiles of the unweighted path. Pooling a sample
dimension named `gcm` without weights warns once; filter
`UnweightedModelWeightsWarning` to silence it. On the Mexico sample
(`examples/rates`) the weights barely move the median and pull the
95th percentile down by about a fifth.

`quantile_method` says how a quantile is computed, by the name of
the computation:

- `"linear"` interpolates between members. It is computed by numpy,
  so it is `numpy.quantile(method="linear")` and xarray's `.quantile`
  to the last bit. It has no weighted form and is refused with
  weights. Where numpy's interpolation meets an infinite member it
  gives NaN, even for a median of 1, 2 and an infinity, and such a
  quantile is refused rather than written as a group with no value.
- `"lower_step"` is the step of the historical extraction tool: the
  member one below the one whose cumulative weight share reaches q.
  Without weights every member counts the same, which is what that
  tool did when it was told to use no model weights. A quantile below
  the first step is `-inf`, and the value of a member with weight zero
  can come back.
- `"inverted_cdf"` is the standard inverse of the empirical
  distribution (Hyndman and Fan type 1), the smallest member value
  whose cumulative weight share reaches q. It is the same with or
  without weights, never returns `-inf`, and never returns the value
  of a member whose weight is zero.

With four members that count equally the three give three different
95th percentiles: the third member, a value between the third and the
fourth, and the fourth.

`summation` says how a mean is summed:

- `"fast"` is the vectorised sum of pandas and numpy. Its last bits
  depend on the order the rows arrive in, which in a pipeline run with
  several workers is the order the workers finish in.
- `"exact"` takes the window means and the pooled mean from exactly
  rounded sums, so the order cannot change them.

Neither option has a default. `summarize_samples` and
`pooled_statistics` take both, `window_means` takes `summation`, and a
pipeline config sets them under `[stats]`. The manifest records both.
Under `"inverted_cdf"` or `"exact"` a statistic that comes out
infinite raises instead of being returned; NaN does not.

Files made before v0.6.0 had no such options. They are reproduced by
`"lower_step"` if the run had weights and `"linear"` if it had none,
with `"fast"`. Those without weights took their quantiles from pandas,
whose arithmetic for the same interpolation differs from numpy's in
the last bit of about one quantile in a hundred.

## Warming levels

`assign_warming_levels` gives each member of a temperature table the
warming level it is at in a calendar period. A member is one series of
global mean temperature, named by the columns of the table that are
not the year or the value: a model and an rcp, a model and an ssp, or
whatever the table has.

```python
table = cilreg.read_gmst("gmst.csv", member_cols=["gcm", "rcp"])
levels = cilreg.assign_warming_levels(
    table,
    reference_period=(1986, 2005),
    offset=0.61,
    levels=[1.5, 2, 3, 4, 5],
    half_width=0.25,
    periods=[
        cilreg.Period(gmst_span=(2040, 2059), impact_span=(2040, 2058)),
        cilreg.Period(gmst_span=(2080, 2099), impact_span=(2080, 2099)),
    ],
    smoothing="unknown",
)
levels.windows_of(("MIROC5", "rcp85"))
levels.record()
```

For each member the series is rebased by taking off its own mean over
`reference_period` and adding `offset`. The mean of the rebased series
over a period's `gmst_span` picks the level. A level holds a mean from
`half_width` below it to `half_width` above it, both ends included. A
mean in no band gives the member no level in that period, so with a
half-width of 0.25 a member at 2.5 belongs to neither 2 nor 3. Two
bands may share an end, as 1.5 and 2 share 1.75, and a mean exactly
there goes to the level declared first. Bands that overlap are
refused.

None of `reference_period`, `offset`, `levels`, `half_width`,
`periods` and `smoothing` has a default. The offset is the warming
from a pre-industrial baseline to the reference period, and it is only
right for that pair: 0.61 goes with 1986 to 2005 in AR5 (WG1 TS.2.2.1),
and AR6 gives another figure against another period. A call that
leaves one out fails.

A period has two spans, both including their first and last year. The
`gmst_span` is the years the temperature is averaged over, and the
`impact_span` the years of the impacts that go under the level, which
is also what the period is labelled by. They are separate because a
product may average its impacts over other years than its temperature.

Nothing is smoothed. The series is taken as it is in the table, and
`smoothing` says in words how it was smoothed before, `"unknown"` if
nobody knows. It goes into the record as written.

A member's name is matched as it is written. A surrogate model is a
member of its own and needs its own rows. A member that lacks a year
of the reference period or of a `gmst_span` is refused by name.

In a pipeline config the same settings go under
`[stats.warming_levels]`, in place of `stats.windows`. The example is
a physical product as the Inequality Paper makes one: each member's
window is collapsed to its mean first, over nineteen impact years, and
the quantile is the extraction tool's step.

```toml
[stats]
sample_dims = ["batch", "rcp", "gcm"]
time_col = "year"
quantiles = [0.05, 0.5, 0.95]
quantile_method = "lower_step"
summation = "fast"
time_reduction = "window_mean"

[stats.warming_levels]
gmst = "gmst.csv"
member = ["gcm", "rcp"]      # columns of the table, each a tree level
reference_period = [1986, 2005]
offset = 0.61
levels = [1.5, 2, 3, 4, 5]
half_width = 0.25
smoothing = "unknown"

[[stats.warming_levels.periods]]
gmst_span = [2040, 2059]
impact_span = [2040, 2058]

[[stats.warming_levels.periods]]
gmst_span = [2080, 2099]
impact_span = [2080, 2098]

[output]
dir = "out"
write_leaves = true

[reduce.pandas]
memory_gb = 128
```

A leaf's member is read off its level values, here its `gcm` and its
`rcp`, so the tree declares both scenarios and the statistics pool
them. A leaf is pooled under the level its member has in a period and
left out of the periods in which it has none. The statistics file has
a `warming_level` column next to `window`, which stays the period, and
two counts on every row. `n_members` is the draws behind the
statistic, a batch of a model counting as one. `n_bin_members` is the
distinct members of the temperature table behind it, which is the
number to read a 95th percentile over four models by. A level nobody
is at in a period has no row, and the manifest records it with a zero.

The statistics come from the leaf files under every stage, as when
the years are pooled, so stage `all` writes the leaves and then
reduces them, and `--stage reduce` gives other levels, other periods
or the other reduction from the same leaves. `time_reduction =
"window_mean"` collapses each member's window first and `"pooled"`
pools its years. `"per_year"` is refused with warming levels, since a
level is reached over a period. The span is what is written:
`[2040, 2058]` is nineteen years.

A config that leaves out `reference_period`, `offset`, `levels`,
`half_width`, `smoothing` or `periods` is refused. So is one that sets
`stats.windows` or `stats.model_weights` as well: the members at a
warming level count equally.

The paper's two families differ in both settings. Its physical
products collapse each member's window first and take the extraction
tool's step, so `quantile_method = "lower_step"`, as in the example.
Its valued products pool the years with the models and the draws over
twenty years and take xarray's quantile, so `time_reduction =
"pooled"` with `quantile_method = "linear"` and periods whose impact
span is twenty years. A member the tree declares
that has no row in the table is refused before any leaf is read. A
leaf that lacks a year of an impact span its member is pooled over, as
a row or as a value, is refused by name, since its mean would be over
fewer years than the span says. The manifest carries the whole record
under `warming_levels`.

`windows_of` gives, for one member, the periods in which it has a
level, each with the level, the period's label and its impact years.
`record` returns what a manifest should keep: the table's path and
SHA256, every setting above, the rule for a shared end, each member's
mean and level in each period, the number of members at each declared
level with a zero where nobody is, and the members with no level.

## Keeping the aggregated leaves

A pipeline run keeps nothing between the weights and the statistics
unless `output.write_leaves = true`. With it, every leaf is written
after the weights are applied, with its years, to
`leaves/<level values>/<name>.aggregated.parquet` under the output
directory. It is off by default because the leaves of a product take
far more disk than its statistics.

A leaf file is plain parquet. Its columns are `target_ordinal`, the id
fields, the leaf's own dimensions and the value. Its rows are sorted
by target and cut into row groups of 128 targets, the same targets in
the same group of every leaf, with an empty group where a leaf has
none of them. So the rows of a range of targets can be read from
every leaf without reading the rest. `leaves/targets.<digest>.parquet`
gives the target of each ordinal.

The footer of a leaf file says what made it, as JSON under the key
`cil_regionalization.leaf`: the weight, the kind, the versions, the
SHA256 of the two files of the weights artifact, the leaf's level
values and its source file. No path is part of that identity. Paths
are recorded and never compared.

`leaves/<output.stem>.leaves.manifest.json` lists the leaves of a run
with their accounting, sizes and hashes. It is written before the
first leaf with `"complete": false` and again when every leaf has
succeeded, so a run that was killed leaves a manifest that says so.
Runs under different stems can share an output directory when their
leaves differ, as one stem a scenario does. A run that would write a
leaf another stem's manifest lists is refused.

```
cilreg pipeline config.toml --stage aggregate
```

writes the leaves and their manifest and stops before the statistics.

```
cilreg pipeline config.toml --stage reduce
```

takes the statistics from leaves written before, without opening the
source tree or the weights. Everything under `[stats]` can differ from
the run that wrote the leaves, so other windows or quantiles cost one
pass over the leaf files and no aggregation. A product with other
windows takes its own `output.stem` and names the leaves it reduces
with `output.leaves_stem`. It gives the same bits whatever the number
of workers. Under `summation = "exact"` those are the bits `all` gives
on one worker, whatever the size of the blocks. Under `"fast"` they
are too when the leaves have no missing values. With missing values
the last bit of a window mean can depend on the block, because on
some pandas versions the mean of one group changes with what else is
in the frame.

Reducing pools one block of targets at a time, so it needs to be told
how much memory it may use:

```toml
[reduce.pandas]
memory_gb = 128   # for all the workers together
n_workers = 16    # run.n_workers unless set
```

`--dry-run` prints the blocks and the number of file opens that
budget gives. Before any work the leaves are checked against their
manifest: what aggregated them has to be what the config says, field
by field, and every file has to be there with its size. A path is not
part of that. If the weights or the tree have moved, the run says so
and goes on, because the hashes in the manifest say what they were.
`--verify-leaves` also hashes every leaf file against the manifest,
which reads all of them and is the way to rule out a file changed in
place.

A cell of `tree.exclude` has one value for each level, and `"*"`
stands for every value of its level, so `["*", "rcp85", "ACCESS1-0"]`
leaves that model out of that scenario in every batch.

Reducing fewer leaves than were aggregated is allowed, with a narrower
`tree.expect` or a longer `tree.exclude`. The statistics file then has
an `n_members` column whether the config asks for it or not, so that
the smaller sample shows in the file. `output.write_reduced` is not
available under `reduce`.

A leaf's own dimension has to be an integer, a float, a boolean or a
plain string when leaves are written. A categorical, datetime or
nullable column is refused, because pandas 2 and pandas 3 write those
differently.

## What happens to the years

`stats.time_reduction` says what is done with the years of a member,
and has no default:

- `"window_mean"` averages each member's years inside a window first,
  and the statistics are over those means, one value a model and draw.
- `"pooled"` takes no window mean. Every year of a member inside a
  window is a member of the sample, so a quantile is over models,
  draws and years together and carries the spread from year to year.
  A year inside two windows is pooled in both.
- `"per_year"` takes no window mean either and keeps the year. The
  statistics are over the members within each year from the first to
  the last of `stats.years`, which takes the place of `stats.windows`,
  and the file has a `year` column where the others have `window`. It
  is a time series, one row a target, year and statistic.

```toml
[stats]
time_reduction = "per_year"
years = [2020, 2099]

[output]
write_leaves = true

[reduce.pandas]
memory_gb = 128
```

The rows of `"pooled"` and `"per_year"` are the leaves' own rows, far
more than one process can hold, so their statistics always come from
the leaf files, one block of targets at a time. Stage `all` writes the
leaves and then reduces them. It asks for `output.write_leaves = true`
and for the memory budget before it reads a leaf. The blocks are sized
after the leaves are written, so a budget too small for one row group
is found then. The leaves stay, and `--stage reduce` takes the
statistics from them with a larger budget. `output.write_reduced` is
refused, and so is a run without `[weights]`. A config that still has
`stats.pool_years` is refused and told which value to write.

Per year, a year a member lacks is left out of that year's statistics,
and `n_members` shows the smaller sample. Nothing is refused for it.

What pooling changes:

- The mean is over member-years. When every member has every year of
  the window it is the mean of the window means, up to rounding. A
  member that lacks years has fewer rows in the sample and so counts
  for less than a member with all of them.
- A model weight applies to each member-year.
- `quantile_method` and `summation` work as before. No window mean is
  taken, so `summation` acts on the pooled mean alone.
- With `include_n_members` the file has two counts. `n_members` is
  what it is without pooling, the members behind a statistic: those
  with a value in the window and, with model weights, a weight above
  zero. `n_sample` is the member-years behind it.
- The manifest records `time_reduction`, next to `quantile_method`
  and `summation`, and `years` for a product per year.

## Your own NetCDF

A file on a regular lat-lon grid, with 1-D `lat` and `lon`
coordinates, becomes a cell table in one call:

```python
cells, grid = cilreg.read_netcdf_grid("tas.nc", ["tas"])
out = cilreg.apply_weights(weights, cells, kind="intensive",
                           weight="pop", value_col="tas",
                           data_version=grid)
```

The grid identity comes from the coordinates (resolution, cell centres
or edges, longitude convention), and every coordinate is mapped to its
cell index and back to a centroid; a value that is not a cell centre
raises. A file with a longitude beyond 180 or below 0 needs nothing
more. A subset in the east hemisphere, with every longitude in 0 to
180, always needs `grid=`: the same coordinates are valid on both
conventions, so the file cannot say which one your weights use. The
error prints the two calls that work, for example `grid="grid-0.25deg-
center-[-180,180)"` or the `[0,360)` one, which also wraps a 0 to 360
file onto the other. `grid` takes the string or a `GridSpec`. Nothing is regridded: coordinates that sit on no grid are
an error, and so are cells that are not square. Cells masked in every
slice are left out, so `apply_weights` reports a weighted cell that has
no data; a cell masked in only some slices raises.

## Worked example

`examples/aggregation/` aggregates one batch of Monte Carlo mortality
projections, all 33 climate models, for Colombia. Its notebook maps a
single draw next to a pooled quantile, shows percent of GDP with the
`ratio` kind, and plots the spread of draws; `run_example.py` adds
municipalities and rates from the command line. `examples/rates/`
aggregates the effect of climate change on Mexican mortality rates to
municipalities on both GADM versions. `examples/labor/` aggregates
the effect of climate change on labor supply to Indian states,
physical in minutes per worker per day and valued in 2005 PPP USD.
`examples/energy/` aggregates the effect of climate change on energy
use to United States states with delta-method uncertainty: gradient
aggregation, the quadratic form, and the spread that independent
draws would lose. `examples/climate/` starts from raw daily BCSD
temperature for the Philippines and Germany and aggregates heating and
cooling degree days to impact regions, through segment weights read
with `from_table`, and to provinces with the grid weights. Notebook
outputs are committed, so they read without running.

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

`uv pip install .` works too. The geospatial stack (geopandas,
shapely, rasterio, exactextract) installs with it and ships wheels on
Linux and macOS. `[netcdf]` reads the NetCDF Monte Carlo trees,
`[examples]` adds what the example notebooks need (the NetCDF readers,
jupyter and matplotlib), `[bigquery]` adds the BigQuery generation
backend, `[dev]` adds pytest. Python 3.10 or newer.

To run the examples from a clone at the versions in `uv.lock` (on
Python 3.11 and newer, the versions in
`envs/requirements-cluster.freeze.txt`, which the test suite passed
against):

```
uv sync --extra examples
```

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
From Python, `cilreg.generate_weights(config)` does the same as `run`;
it takes a `Config` or a TOML path and returns the result. Polygons
held in memory go through a GeoParquet file first
(`gdf.to_parquet(path)`), so the manifest can record the file's hash.
A run whose weights fail the sum-to-one check writes nothing and
raises `SumToOneError`, whichever way it was started.
Worked configurations, including the GADM target preparation and the
full generation runs, live under `examples/`.

GADM 2.0 and GADM 4.1 have different unit universes and different
keys, so results built on one cannot be compared with the other. Each
geometry version is its own record, and every manifest states which
version it was built against.

Four records are published on Zenodo:

- the impact region shapefile, world-combo-201710
  (doi:10.5281/zenodo.21934131)
- the SMME climate model weights (doi:10.5281/zenodo.22003542):
  `rcp45_SMME_weights.tsv` and `rcp85_SMME_weights.tsv`, used by the
  statistics stage rather than the spatial aggregation
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

`fetch_weights`, `generate_weights`, `apply_weights`, `read_netcdf_grid`, `summarize_samples`,
`window_means`, `pooled_statistics`, `read_gmst`,
`assign_warming_levels`, `read_gradient_leaf`,
`aggregate_gradient`, `delta_method`, `WeightsArtifact`, and
`load_config` are the supported API, re-exported at the package root.
`window_means` and `pooled_statistics` are the two halves of
`summarize_samples`, for pipelines that reduce time per leaf before
pooling. Everything else is internal and may change.
