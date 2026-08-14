# How it works

![The pipeline: geometries and rasters go into weight generation, which
produces weight files; weight files plus impact region draws go through
application, then statistics, then results.](pipeline.svg)

The top lane happens once per pair of geometries and is our job: take
the administrative boundaries, the impact regions, and a population
raster, intersect them, and record what share of each intersection
carries. The result is a pair of files, the weights parquet and a
manifest that says exactly what produced it, published on Zenodo. The
bottom lane is the everyday path: fetch the weight file, apply it to
Monte Carlo draws at impact region level, and compute statistics at
the end.

## Method

A weight `w` connects one impact region `r` to one administrative unit
`u`. Both operations are the same multiply and add; what differs is
which way the weights are normalized.

For a total (dollars, deaths), each region's value is split between
the units it touches, and the pieces must add back up to the region's
value:

    value(u) = sum over r of w(r, u) * value(r),
    where the w(r, u) for one region sum to 1

For an average (a death rate, a temperature), each unit's value is the
population weighted mean over the regions inside it:

    value(u) = sum over r of w(r, u) * value(r),
    where the w(r, u) for one unit sum to 1

The order of aggregation and statistics matters because quantiles do
not pass through sums: the 95th percentile of a sum is not the sum of
95th percentiles. Adding up per-region quantiles assumes every region
has its worst draw at the same time, which overstates the tails. So
the pipeline aggregates each Monte Carlo draw separately and takes
quantiles only at the end, and the statistics stage refuses input that
already contains quantiles.

## Required inputs

- The draws, as a long table: one row per impact region, year, and
  sample member (batch, climate model), with the region id in the
  weight file's key column.
- The variable's `kind`: `extensive` for totals, `intensive` for
  averages, `ratio` for a quotient built from two extensive parts.
- `data_version`: the impact region vintage the data was built on,
  checked against what the weight file records.
- For statistics: which columns are the sample, the time window, and
  the requested quantiles.

Generating new weights instead needs the two geometry files, the
weighting raster, and a version label for each geometry; see the main
README.
