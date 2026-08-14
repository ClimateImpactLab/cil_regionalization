# Worked example: Colombia

This example aggregates a small sample of real Monte Carlo mortality
projections for Colombia, from impact regions to departments (ADM1)
and municipalities (ADM2), covering both variable kinds and both
weight directions. It is a sample, not the complete output: two Monte
Carlo draws out of hundreds, one climate model out of dozens, so the
numbers demonstrate the mechanics and are not results.

The Monte Carlo sample is committed here (about 1.6 MB). The weights
come from the published Zenodo records, fetched by name, which is what
any real use looks like.

```
python examples/aggregation/run_example.py            # fetches weights
python examples/aggregation/run_example.py --offline  # committed slices
```

You need the package with the `[netcdf]` extra, and network access for
the default mode. Both modes produce identical output.

## Quick example

```python
from segment_weights import fetch_weights, apply_weights
from segment_weights.netcdf_io import read_netcdf_leaf

weights = fetch_weights("gadm20-adm1-per-source")
data = read_netcdf_leaf(leaf_path, variables=["total_damages"],
                        region_dim="region", region_col="hierid",
                        kind="extensive")
result = apply_weights(weights, data, kind="extensive", weight="pop",
                       value_col="total_damages",
                       data_version="world-combo-201710",
                       restrict_to_sources={(h,) for h in data["hierid"].unique()})
```

The fetched weights are global and the data covers one country, so
`restrict_to_sources` declares the subset; the coverage checks stay
strict within it. With `--offline` the example loads committed
Colombia slices instead (`WeightsArtifact.load` on the directories
under `data/weights/`) and needs no restriction, since the slices
already match the data.

`data_version` names the impact region geometry vintage the data was
built on. If it differs from the `source_version` the weight file
records, `apply_weights` raises an error before producing numbers;
that is how a mismatched pairing of weights and data gets caught.

## What is in the sample

Colombia has 500 impact regions, 32 ADM1 departments in this GADM 2.0
vintage (the country later gained a 33rd), and 1,065 ADM2
municipalities. The Monte Carlo sample is two batches for one climate
model (GFDL-ESM2G, rcp85, iam low, SSP3), in the standard tree
grammar, with two files per leaf directory because the same country
has two different variables in two different trees:

- `Agespec_interaction_response-combined.nc4`: the physical mortality
  rate (`rebased`, deaths per person per year). A rate is intensive:
  the ADM1 number is an average over the department's impact regions.
- `mortality_damages_IR_batch.nc4`: monetized mortality damages
  (`total_damages`, 2019 USD). Dollars are extensive: the ADM1 number
  is a share of each impact region's total, and the shares must add
  back up.

`weights/` carries the Colombia slice of the global weight files at
both levels and in both directions, used by `--offline`. Each manifest
records that it is a subset and carries a checksum of the sliced file.

## What it prints

```
The tree this example reads:
  montecarlo/batch0/rcp85/GFDL-ESM2G/low/SSP3/Agespec_interaction_response-combined.nc4
  montecarlo/batch0/rcp85/GFDL-ESM2G/low/SSP3/mortality_damages_IR_batch.nc4
  montecarlo/batch1/rcp85/GFDL-ESM2G/low/SSP3/Agespec_interaction_response-combined.nc4
  montecarlo/batch1/rcp85/GFDL-ESM2G/low/SSP3/mortality_damages_IR_batch.nc4

Physical rate at ADM1 (deaths per person per year, intensive,
population weighted mean over each department's impact regions):
ISO  ID_1  year   rebased  batch
COL     6  2099 -0.002262 batch0
COL    14  2099 -0.002181 batch0
COL    21  2099 -0.001536 batch0
COL    10  2099 -0.001118 batch0
COL     7  2099 -0.001107 batch0
  ... 32 departments, both batches computed

Monetized damages at ADM1 (2019 USD, extensive, allocated shares
sum to each impact region's total; mass balance checked):
ISO  ID_1  year  total_damages  batch
COL    15  2099  -3.912271e+07 batch0
COL    31  2099  -8.851429e+07 batch0
COL    32  2099  -1.399679e+08 batch0
COL     1  2099  -1.678937e+08 batch0
COL    26  2099  -1.982496e+08 batch0
  sum over departments -1.774214e+11 = sum over impact regions -1.774214e+11

Monetized damages at ADM2 (1,065 municipalities from the same
500 impact regions; same kind, same checks, finer weights):
  region COL.6.R2297bdd18341ff36 splits across 90 municipalities;
  its largest population shares:
ISO  ID_1  ID_2    popwt
COL     6   294 0.095976
COL     6   275 0.048733
COL     6   252 0.034370
  shares sum to 1.000000
  2099 sum over municipalities -1.774214e+11 matches the impact region sum

Statistics over the two batches (window mean 2080-2099, then
pooled; with 2 draws these quantiles are mechanics, not results):
ISO  ID_1 statistic  total_damages
COL     1      mean  -4.022969e+07
COL     1       q05  -1.483635e+08
COL     1       q50  -4.022969e+07
COL     1       q95   6.790414e+07
```

The ADM2 section is where the weights do visible work. In this vintage
a Colombian impact region is a grouping of municipalities, so one
region's total spreads across up to 90 of them by population share,
the shares sum to one, and the municipality totals still add back to
the region totals. The last line of each damages section is that mass
balance identity. Rates have no such identity; an average conserves
nothing, and the check there is that every department value lies
between the smallest and largest rate among its own impact regions.

The quantiles are printed to show where statistics happen: last, after
spatial aggregation and the window mean. Two draws cannot support a
5th or 95th percentile; a full run pools hundreds.

## Other combinations

ADM0 is this example with coarser weights. A physical rate at ADM2 is
the intensive case with the `adm2_per_destination` weight file, also
included here. Switching between rates and dollars is a change of
`kind` and weight direction, and the library checks that pairing, so
the wrong combination raises an error instead of producing a plausible
wrong number. Those variants are deliberately not built out.
