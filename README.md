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

In the mortality and labor trees, the effect of climate change is a
file's `rebased` variable minus the same variable of its `-histclim`
sibling, whatever the file stem (mortality's `...-combined.nc4`,
labor's `uninteracted_main_model.nc4`); `rebased` on its own is not
it. No adaptation is the exception and stands alone. The Climate
Impact Lab memo "The art of rebasing and histclim" has the reasoning.

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
)
```

The naming rule: weight file names are lowercase, a trailing `*` is
bookkeeping and drops out, and an underscore suffix (`gfdl-cm3_94`)
marks a surrogate, which the projection trees spell with a prefix
(`surrogate_GFDL-CM3_94`). Strip the prefix and lowercase, and every
model in the trees matches exactly one weight row.

Weighted quantiles follow the historical extraction tool: the left
step inverse of the weighted empirical distribution, not the
interpolated quantiles of the unweighted path. Pooling a sample
dimension named `gcm` without weights warns once; filter
`UnweightedModelWeightsWarning` to silence it. On the Mexico sample
(`examples/rates`) the weights barely move the median and pull the
95th percentile down by about a fifth.

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
Notebook outputs are committed, so they read without running.

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
`[bigquery]` adds the BigQuery generation backend, `[dev]` adds
pytest. Python 3.10 or newer.

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

`fetch_weights`, `apply_weights`, `summarize_samples`,
`window_means`, `pooled_statistics`, `WeightsArtifact`, and
`load_config` are the supported API, re-exported at the package root.
`window_means` and `pooled_statistics` are the two halves of
`summarize_samples`, for pipelines that reduce time per leaf before
pooling. Everything else is internal and may change.
