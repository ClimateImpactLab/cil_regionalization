# Changelog

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
