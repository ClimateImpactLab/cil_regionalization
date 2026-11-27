# Worked example: mortality in France, end to end

`france.py` is a [marimo](https://marimo.io) notebook that makes one
product from its inputs. It is the effect of climate change on mortality
in metropolitan France, for its 13 regions and 96 departments, as
deaths, as deaths per 100,000 and as damages over GDP, for 2080 to
2099, year by year, and at warming levels. Every step is a call of the
library or of `cilreg`, in the order a product is made in.

1. The inputs, mapped together.
2. The weights, built with `cilreg run` from four configs, with the
   manifest beside each file.
3. The published weights, fetched with `cilreg fetch` and compared with
   ours department by department.
4. The weights applied three ways, `extensive` for deaths, `intensive`
   for the rate and `ratio` for the share of GDP, and why the ratio is
   not an average of the regions' shares.
5. `window_means` then `pooled_statistics`, and the same numbers from
   `summarize_samples`.
6. The statistics with and without the SMME climate model weights, and
   the three quantile methods side by side.
7. A product from a tree. `cilreg pipeline --stage aggregate` writes the
   leaves once, and `--stage reduce` makes the window mean, the pooled
   and the per year products from them.
8. Warming levels, from the paper's temperature table, for the physical
   products and the valued ones.
9. Provenance. The hashes in the manifests are worked out again from the
   files, and `--verify-leaves` hashes the leaves.

The sample is two Monte Carlo batches, rcp45 and rcp85, all 33 climate
models, SSP3 and the low growth model, 2020 to 2099. The numbers show
how a product is made, and they are not results.

## Why ADMIN EXPRESS and not GADM

The published weights are built on GADM, whose licence forbids
redistributing its boundaries. This example builds its weights on
ADMIN EXPRESS, the official boundaries of IGN under the Licence Ouverte
2.0, which may be redistributed with attribution. So the example also
shows what anyone does who brings their own boundaries. Section 3
compares the two by matching each GADM department to the ADMIN EXPRESS
department most of its people fall in, from the two weight files alone.

The overseas departments are left out, and the notebook says so.
ADMIN EXPRESS has Guadeloupe, Martinique, Guyane, La Réunion and Mayotte
as departments of France, and the impact regions keep them under their
own country codes.

## Running it

```
uv sync --extra examples
uv run marimo edit examples/france/france.py
```

`python examples/france/france.py` runs it as a script, and
`marimo export html` writes it with its outputs. Four variables set what
it reads and writes.

- `FRANCE_DATA` is the data directory, `examples/france/data` unless set.
- `FRANCE_OUT` is where the weights, the configs and the pipeline
  outputs go, `examples/france/out` unless set.
- `FRANCE_GMST` is the temperature table of section 8, the mortality
  paper's under `data/external/` unless set.
- `FRANCE_OFFLINE=1` reads the committed France rows of the published
  weights instead of fetching them in section 3.

Without the cluster, the notebook runs on synthetic data of the same
layout, which the test suite does too.

```
python examples/france/synthetic_data.py /tmp/france
FRANCE_DATA=/tmp/france/data FRANCE_GMST=/tmp/france/gmst.csv \
    FRANCE_OFFLINE=1 FRANCE_OUT=/tmp/france/out python examples/france/france.py
```

## Making data/

`prepare_data.py` writes `data/` once, on the cluster. It reads the
impact regions, the GPW raster, the SSP3 population and the two
mortality trees where the published runs read them. It also reads the
ADMIN EXPRESS shapefiles from where you unpacked them, and it fetches
the published GADM 2.0 department weights from Zenodo.

```
python examples/france/prepare_data.py \
    --admin-express <directory with DEPARTEMENT.shp and REGION.shp> \
    --admin-express-version <the edition, as IGN names it>
```

`data/sources.json` records where each file came from, its terms and
its SHA256.

The temperature table is the mortality paper's, committed under
`data/external/` with its source and terms.

## Sources and terms

- Boundaries: IGN, ADMIN EXPRESS, under the Licence Ouverte 2.0.
- Population raster: CIESIN, GPW v4 rev 10, 2015, 30 arc seconds, under
  CC BY 4.0, cut to France.
- Impact regions: Climate Impact Lab, world-combo-201710,
  doi:10.5281/zenodo.21934131.
- Climate model weights: Climate Impact Lab, SMME weights,
  doi:10.5281/zenodo.22003542.
- Published weights: Climate Impact Lab, GADM 2.0 weights,
  doi:10.5281/zenodo.21934155.
- Mortality projections and SSP3 population: Climate Impact Lab.
