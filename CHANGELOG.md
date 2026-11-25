# Changelog

## Unreleased

**The rows of a pipeline statistics file, and of the reduced file,
come in a new order. No value changed.** Two experimental options
change how the statistics are computed; both default to `"legacy"`,
which is what every release so far has done.

- Changed: the pipeline sorts the statistics rows before writing them:
  by statistic in the order requested (the mean, then the quantiles as
  listed), then by each tree level that is not pooled, in the order
  `tree.expect` declares its values, then by window in the order of
  `stats.windows`, then by the remaining id columns ascending. The
  reduced file of `output.write_reduced` is sorted the same way, over
  all its tree levels. Before, the rows of both followed the order the
  leaves were pooled in, which with several workers is the order the
  workers finish in, so two runs of one config could list the same
  rows differently. A run with one worker usually listed them this way
  already. The frame `run_pipeline` returns is sorted the same.
- New, experimental: `quantile_method` on `summarize_samples` and
  `pooled_statistics`, and `stats.quantile_method` in a pipeline
  config. `"inverted_cdf"` is the standard inverse of the empirical
  distribution (Hyndman and Fan type 1): the smallest member value
  whose cumulative weight share reaches q. It is one definition with
  or without weights, and every quantile is the value of a member with
  positive weight.
  `"legacy"` is two definitions. Without weights it interpolates. With
  weights it returns the member just before the one that reaches q,
  one member below the standard inverse, as the historical extraction
  tool did. Three things follow from that. A quantile is `-inf` when
  the first member already reaches q. The value of a member with
  weight zero can come back as a quantile, although the mean leaves
  that member out: with one such model among 33, about 3% of the cells
  of each quantile on synthetic samples. And at 0.5 with equal weights
  it switches to the midpoint of the two middle members.
- New, experimental: `summation` on the same two functions and on
  `window_means`, and `stats.summation` in a pipeline config.
  `"exact"` takes the window means and the pooled mean from exactly
  rounded sums (`math.fsum`), so the order of the rows cannot change
  them. Under `"legacy"` the order can change the means, and in a pipeline run with several
  workers it does, because the leaves are pooled in the order the
  workers finish. Two runs of one config over a synthetic tree (240
  leaves, 8 workers, model weights) disagreed in the last bits of
  about a third of the means, 119 to 133 of 388, by at most 8.9e-16 on
  values of order 1. Without model weights 2 of 388 differed. No
  quantile differed.
- Under either new setting a statistic that comes out infinite raises
  instead of being returned or written: in impact data an infinity
  means something upstream broke. It is the statistic that is checked,
  so an infinity in a year no window covers does not fail a run, and
  one that a quantile selects does. NaN is left alone; a group with no
  value has no statistic. Under `"exact"` a sum over infinities of
  both signs, or one that overflows, raises too. The legacy settings
  return whatever comes out, `-inf` included, as before.
- The pipeline manifest records `quantile_method` and `summation`.

## v0.5.0

**Behaviour change in `cilreg run`.** Results unchanged.

- New: `generate_weights(config)` makes weights from Python. It takes
  a `Config` or a TOML path, writes the weights and manifest to
  `output.dir` and returns the `WeightsResult`. `cilreg run` now goes
  through the same code, so both give the same files.
- Changed: a run whose weights fail the sum-to-one check now writes
  nothing. `generate_weights` raises `SumToOneError`, which carries the
  full report, and `cilreg run` prints the report and exits 1. Before,
  the files were written and the exit code was 1, which left a weights
  file on disk that had failed its own check.
- `WeightsResult` and `SumToOneError` are exported from the package
  root.
- Fix: manifests from grid runs now record the SHA256 of the regions
  file under `inputs["geometry:regions"]`. Polygon runs already did;
  grid runs never had it, although a comment said they would. The
  weights are unchanged. Weight files made before this have no
  regions checksum.

**Results unchanged.**

- New: `read_netcdf_grid(path, variables)` reads a NetCDF on a regular
  lat-lon grid into the cell table `apply_weights` takes and returns
  the grid identity with it. The identity is derived from the
  coordinates, or named with `grid=` (a string or a `GridSpec`), and
  every coordinate is checked to be a cell centre on it. A subset with every longitude in 0 to 180
  fits both longitude conventions, so it always needs `grid=`; the
  error prints the two calls. Cells must be square. Needs the
  `[netcdf]` extra.

## v0.4.3

**Results unchanged.** Values equal v0.4.2, and `pooled_statistics`
gets back the dtype it had in v0.4.1.

- `window_means` is faster: 1.61 to 0.84 seconds on a real adm2 leaf.
  It now groups on integer codes, shared with the duplicate check,
  instead of a pandas groupby over the key columns. Measured outside
  this repository.
- Fix: `pooled_statistics` on the unweighted path returned a nullable
  `Float64` value column as plain `float64`. It now keeps `Float64`.
  This regression came in with v0.4.2. The values were equal, so no
  product changed.

## v0.4.2

**Output changed, dtype only.** On the unweighted path,
`pooled_statistics` (and `summarize_samples`) returned a nullable
`Float64` value column as plain `float64`. The values were equal.
v0.4.3 restores the dtype. Plain `float64` input is unaffected, and
every value is otherwise unchanged.

- `pooled_statistics` uses much less memory and time: the end phase of
  a real adm2 run went from 57.3 to 35.3 GiB and from 388 to 211
  seconds. Measured outside this repository. Groups are found with
  integer codes rather than a pandas groupby, and the duplicate check
  shares those codes.
- `cil_regionalization.__version__` now reports the installed package
  version. It was a fixed string, "0.1.0". Without an install it is
  "unavailable".
- New `examples` extra (NetCDF readers, jupyter, matplotlib) and a
  `uv.lock`, so the examples run from a clone at locked versions with
  `uv sync --extra examples`.
- The climate example text is fixed and its notebooks are re-run.
  Docstrings in `stats.py` are shortened.

## v0.4.1

**Results unchanged.** Output is identical to v0.4.0.

- `apply_weights` is faster and uses less memory. Extensive
  aggregation over the global grid at 100 fields went from 477 to 29.7
  seconds and from 27.9 to 10.5 GiB. The intensive case went from 30.9
  to 20.6 seconds. Measured outside this repository. The duplicate
  check, the set of source units present and the applied-row mask no
  longer build one Python tuple per row.
- Tests compare the new key handling against the row-wise code it
  replaced, including missing keys.

## v0.4.0

**Results unchanged.** Existing functions are untouched. The release
adds new ones.

- New `window_gradient`: mean of stacked yearly gradients over an
  inclusive year window, with the same bounds as `window_means`.
- New `mixture_quantile`: quantile of a weighted mixture of normal
  distributions, for the delta-method variances.
- scipy is now a dependency (`scipy>=1.10`; `scipy==1.17.1` in
  `envs/requirements-cluster.txt`).
- New climate quickstart notebook and a global example. The example
  docs now point at the global impact region weights on Midway3.

## v0.3.0

**Results unchanged for anything that read before.** Two behaviours
differ on BCSD daily files:

- Gregorian-calendar files, which used to fail, now read. Feb 29 is
  dropped by date. Seven of the 21 baseline models are Gregorian.
- Files whose calendar is none of `gregorian`, `standard`,
  `proleptic_gregorian`, `noleap` or `365_day` now raise. `360_day`
  raises with its own message. Before, the calendar was not looked at.
  Files without a CF time coordinate, such as the pattern files, read
  as before.

Added:

- `from_table`, a constructor on the weights class, builds weights from
  a DataFrame, a CSV or a parquet, with no manifest. Cell indices given
  directly are checked against any centroid column, so a table built on
  another grid of the same shape fails instead of shifting the
  geography.
- `DailyGrid.to_cells` turns a grid into the table `apply_weights`
  takes. It checks that the coordinates are cell centres.
- `DailyGrid` gains `dates`, the decoded dates with Feb 29 removed, and
  `read_bcsd_daily` gains `require_full_year`, off by default.
- `OutputSchema`, `SourceUnits` and `GRID_CELLS` are exported.
- New `examples/climate`: degree days from the daily grid to provinces
  and impact regions, for the Philippines and Germany. It runs from a
  clone with no cluster access.

## v0.2.1

**Results unchanged.** The pipeline manifest gains a
`package_versions` entry. No computed value changes.

- The energy example is committed with its data, so the notebook runs
  from a clone.
- A test asserts that the library version is available rather than
  comparing to a literal.
- `pyproject.toml` was never bumped for this tag and still says 0.2.0.

## v0.2.0

- New readers: web JSON (`web_json`), BCSD daily (`bcsd_io`) and
  GDPCIR (`gdpcir_io`).
- The Monte Carlo pipeline is extended (`pipelines/montecarlo.py`),
  and the manifest records the source version.
