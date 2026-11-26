# Changelog

## Unreleased

**The rows of a pipeline statistics file, and of the reduced file,
come in a new order. No value changed.** **`quantile_method` and
`summation` are new, have no default, and have to be given: in every
call of `summarize_samples` and `pooled_statistics`, in every call of
`window_means` for `summation`, and under `[stats]` in every pipeline
config.** Files made before v0.6.0 are reproduced by
`quantile_method = "lower_step"` if the run had model weights and
`"linear"` if it had none, with `summation = "fast"`, those without
weights to the last bit in about 99 quantiles in a hundred.
`window_means` and `summarize_samples` compute in float64 and return
float64 whatever the dtype of the value column. That changes their
result for a float32 or a nullable column under every setting.
A model weights table with two rows for one model is now refused, and
the pipeline manifest records the weights each run used.
A tree level named like a column the pipeline writes itself is refused
before any leaf is read.
**`output.write_leaves` now defaults to false, and an aggregated leaf
is written in a new layout under a new name.** Leaf files of the old
layout are left where they are and are not read.
A new stage, `reduce`, takes the statistics from leaves written before.
A new option, `stats.time_reduction`, says what is done with the years
of a member: averaged over a window first, pooled with the models and
the draws, or kept, one row a year.
New: `assign_warming_levels` gives each member of a temperature table
its warming level in a calendar period.

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
- New: `quantile_method` on `summarize_samples` and
  `pooled_statistics`, and `stats.quantile_method` in a pipeline
  config. It names a computation. `"linear"` interpolates between
  members, as pandas does, and is refused with weights.
  `"lower_step"` returns the member just before the one whose
  cumulative weight share reaches q, one member below the standard
  inverse, as the historical extraction tool did. Without weights
  every member counts the same, which is what that tool did when told
  to use no model weights. `"inverted_cdf"` is the standard inverse of
  the empirical distribution (Hyndman and Fan type 1): the smallest
  member value whose cumulative weight share reaches q. It is one
  definition with or without weights, and every quantile is the value
  of a member with positive weight.
  It has no default. Earlier releases had no such option and used
  `"lower_step"` with weights and `"linear"` without. A pipeline
  manifest records the value.
  Three things follow from `"lower_step"`. A quantile is `-inf` when
  the first member already reaches q. The value of a member with
  weight zero can come back as a quantile, although the mean leaves
  that member out: with one such model among 33, about 3% of the cells
  of each quantile on synthetic samples. And at 0.5 with equal weights
  it switches to the midpoint of the two middle members.
- New: `summation` on the same two functions and on `window_means`,
  and `stats.summation` in a pipeline config. It has no default.
  `"fast"` is the vectorised sum of pandas and numpy, which is what
  earlier releases used. `"exact"` takes the window means and the
  pooled mean from exactly rounded sums (`math.fsum`), so the order of
  the rows cannot change them. Under `"fast"` the order can change the
  means, and in a pipeline run with several
  workers it does, because the leaves are pooled in the order the
  workers finish. Two runs of one config over a synthetic tree (240
  leaves, 8 workers, model weights) disagreed in the last bits of
  about a third of the means, 119 to 133 of 388, by at most 8.9e-16 on
  values of order 1. Without model weights 2 of 388 differed. No
  quantile differed.
- Under `"inverted_cdf"` or `"exact"` a statistic that comes out infinite raises
  instead of being returned or written: in impact data an infinity
  means something upstream broke. It is the statistic that is checked,
  so an infinity in a year no window covers does not fail a run, and
  one that a quantile selects does. NaN is left alone; a group with no
  value has no statistic. Under `"exact"` a sum over infinities of
  both signs, or one that overflows, raises too. `"linear"` and `"lower_step"`
  with the fast sum return whatever comes out, `-inf` included, as before.
- Changed: `window_means` and `summarize_samples` take the window mean
  in float64 and return the value column as float64, under every
  setting and whatever the column's dtype. A float64 column gives what
  it gave, to the last bit. Two things change for other columns, under
  every setting:
  - A nullable `Float64` column used to come back nullable. It now
    comes back as float64 with NaN where it had no value: the mask is
    gone. A caller that tested the result for `pd.NA` has to test for
    NaN.
  - A float32 column is widened before the mean, so its results differ
    from the old ones from the seventh or eighth digit (by up to 9e-7
    of the value on synthetic samples). The pair that reproduces
    earlier releases therefore no longer does so bit for bit for a
    float32 column. The
    other choice was to keep seven digits, and under `"exact"` to round
    an exactly rounded sum back to float32.

  Before, the dtype and the precision followed the column, the pandas
  version and, on the newest pandas, the data. `window_means` returned
  a float32 column as float32 through pandas 3.0 (seen on 2.3.3 and
  3.0.3). On pandas 3.1.1 with the fast sum it returned float64 for a
  column with no missing value and float32 for one with a missing
  value, in the cases tried, and float32 under `"exact"`.
  `summarize_samples` took its window means in float32 where
  `window_means` returned float32, and under `"exact"` on every
  version.

  This reaches callers of the library, not the pipeline's published
  files. A pipeline run that applies weights or subtracts a second
  file (`tree.subtract_filename`) already hands `window_means`
  float64. Only a passthrough run with no subtraction hands over the
  leaf's own dtype, and no production config is one: of the 110
  pipeline configs in the impactmap-data-gen repository 42 are
  passthrough runs (the impact-region products of agriculture, labor
  and mortality), and all 110 subtract a histclim file.

  One published product changes on a rerun: energy as a share of GDP
  at ADM0, `energy/adm0/energy_valued_pctgdp_rcp45.parquet` and
  `energy_valued_pctgdp_rcp85.parquet`. The script that makes it,
  `energy/run_energy_regionalization.py` in impactmap-data-gen, calls
  `window_means` itself on its denominator: GDP stored as float32,
  summed by country and year, still float32. In the pinned cluster
  environment (pandas 3.0.3) the window mean of those sums came back
  float32; it is now taken in float64. Of the 1,012 denominators (253
  countries, four windows) 703 move, by 1.5e-8 of their value at the
  median and 9.1e-8 at most, and the shares move by as much. The
  one-year window does not move. This is float32 rounding in a mean
  of twenty numbers, not the repair of a large error: the sums by
  country are compensated, were within 9.4e-8 of the exact sums, and
  are float32 before and after.
- The pipeline manifest records `quantile_method` and `summation`.
- New: `read_model_weights` reads a climate model weights table and
  returns a `ModelWeights`. The file is read once, and the SHA256 it
  carries is of the bytes the rows were parsed from. `attach` returns
  the weight of each row's model and raises if a model has no row.
  `record` returns the entry a manifest should keep, described below.
  Names match as they did before: a `*` is dropped from the table
  name, `surrogate_` is dropped from the member, and both are
  lowercased.
- Changed: a weights table in which two rows match the same model is
  refused, with both rows named. Before, the last of the two rows was
  applied and nothing was said. Neither published SMME table has such
  rows (32 names in the rcp45 table, 33 in the rcp85 one, all
  distinct after matching), so no run that worked is refused.
- Changed: the pipeline reads the model weights file once, in the
  parent, and hands those rows to the workers. Before, every worker
  read the file again when it started, and workers are replaced every
  16 leaves, so the file was read all through a run. A file edited
  during a run can no longer give different leaves different weights.
- The pipeline manifest gains `model_weights_record`, next to the
  `model_weights` path it already had. It holds the path, the SHA256 of
  the file, `members`, and `unmatched_rows`. `members` has each model
  that a leaf was actually given the weight of, with the table row it
  matched, the weight and the number of leaves. That is what ran, not
  what the config declares, so a model whose leaves were all excluded
  is not in it. `unmatched_rows` has the rows no leaf matched, with
  their weights. The two together are the whole table, so a run can be
  repeated from its manifest when the file is gone. A run without
  model weights records `null`. No statistic changes.
- Changed: the pipeline puts the tree level labels on a leaf's rows
  after the window means are taken. Before, it put them on the leaf
  first, where `apply_weights` and `window_means` grouped every row by
  them. On the benchmark leaf (4.26 million rows after the weights)
  `apply_weights` took 2.7 s with the labels and 1.1 s without, and
  `window_means` 0.46 s against 0.24 s. No file changes for it. The
  statistics, the reduced file and the leaf files keep their columns,
  column order and values.
- Changed: a tree level named `window`, `statistic`, `n_members` or
  `model_weight`, or like an id field of the weights artifact, is
  refused before any leaf is read. A level becomes a column of the
  output, and those names are columns the pipeline writes itself.
  Before, `window` and `statistic` were refused leaf by leaf, after
  every leaf had been read, and the name of an id field failed every
  leaf with an error that gave only the name. A level named `n_members`
  or `model_weight` ran when `stats.include_n_members` and the model
  weights were off, and its values sat in a column of that name.
- Changed: `output.write_leaves` defaults to false. It defaulted to
  true, so a config that did not set it wrote every aggregated leaf,
  about 35 MB for the benchmark ADM2 leaf. A config that wants the
  leaves has to say so. The four `mortality_adm{1,2}_rcp{45,85}` configs under
  `examples/montecarlo/runs` set it to true and now write the new
  layout.
- Changed: an aggregated leaf is written as
  `leaves/<level values>/<name>.aggregated.parquet`, where `<name>` is
  `tree.filename` without its extension. Before, it was
  `aggregated_<tree.filename>`, parquet whatever the extension said.
  The new file is plain parquet with a fixed Arrow schema and no
  pandas metadata:
  - Its columns are an int32 `target_ordinal`, the id fields, the
    leaf's own dimensions and the value. The tree level labels are no
    longer columns. They are in the path and in the footer.
  - Its rows are sorted by target and cut into row groups of 128
    targets. A group holds the same targets in every leaf and is empty
    where a leaf has none of them. `leaves/targets.<digest>.parquet`
    gives the target of each ordinal, which is its rank among the
    artifact's id tuples sorted ascending.
  - Its footer holds, under the key `cil_regionalization.leaf`, the
    layout and what made the leaf: weight, kind, data version, value
    column, policies, file names, level names, the versions and the
    SHA256 of the two files of the weights artifact, the leaf's level
    values and its source files.
  Files of the old layout are not deleted, rewritten or read. A
  passthrough run keeps writing the file it wrote, labels included,
  because without weights there are no targets to lay a leaf out by.
- New: `leaves/<output.stem>.leaves.manifest.json`, written when
  leaves are. It lists every leaf in the order the config declares
  them, with its source files, rows, coverage, mass balance, size and
  SHA256, the rows in each row group summed over the leaves, and the
  rows at each value of each numeric dimension. It holds what made the
  leaves and nothing from `[stats]`. It is written before the first
  leaf with `"complete": false` and again when every leaf has
  succeeded. A run that would write a leaf listed in the manifest of
  another stem in the same directory is refused, whether that other
  run finished or not. Before, two such runs wrote over each other's
  leaves without a word.
- New: `--stage aggregate` on `cilreg pipeline` and on
  `python -m cil_regionalization.pipelines.montecarlo`, and `stage=` on
  `run_pipeline`. It writes the leaves and their manifest and stops.
  It needs `output.write_leaves = true` and `[weights]`, reads nothing
  from `[stats]`, and returns a result whose `stats` is `None`. The
  default, `all`, runs everything as before.
- Changed: when leaves are written, a leaf's own dimension has to be
  an integer, a float, a boolean or a plain string, and the id fields
  have to be strings. A categorical, datetime or nullable dimension is
  refused by name, because pandas 2 and pandas 3 write those
  differently. A tree level named `target_ordinal` is refused like the
  other names the pipeline uses.
- New: the pipeline hashes the two files of the weights artifact when
  it loads them, in the parent and in every worker. A worker that
  loaded other bytes than the run started with fails its leaves. The
  statistics manifest gains `artifact_sha256`, `stage` and
  `leaves_manifest`.
- New: `--stage reduce`, and `stage="reduce"` on `run_pipeline`. It
  takes the statistics from the leaf files a run of `aggregate` or
  `all` wrote, and opens neither the source tree nor the weights.
  Everything under `[stats]` can differ from the run that wrote the
  leaves, so other windows cost one pass over the leaf files.
  - It pools one block of targets at a time. `[reduce.pandas]
    memory_gb` is what all its workers may hold together and has no
    default. `[reduce.pandas] n_workers` defaults to `run.n_workers`.
    The dry run prints the blocks and the file opens the budget gives.
    A budget that does not fit one row group a worker is refused, with
    the smallest that would run.
  - It gives the same bits whatever the number of workers, under
    the `"fast"` sum too, because a block pools its leaves in the
    order the config declares them. `all` with several workers still
    pools them in the order the workers finish.
  - Under `summation = "exact"` it gives what `all` gives on one
    worker, to the last bit, whatever the size of the blocks, with
    any `quantile_method` and with or without model weights.
  - Under the `"fast"` sum it gives the same, to the last bit, on
    leaves without missing values. On leaves with them it can differ
    from `all`, and between two block sizes, in the last bit of a
    window mean. The cause is pandas' own mean, which `"fast"` is:
    on pandas 3.1.1 a missing value in one group of a frame changes
    the mean of other groups (92 of 400 groups in a test, by 4e-17),
    and a block holds part of a leaf where `all` holds all of it.
    pandas 2.3.3 and 3.0.3 do not do this. `"exact"` does not depend
    on what else is in the frame.
  - `output.leaves_stem` names the leaves to reduce. It is
    `output.stem` unless set, and it also names the leaves manifest
    the other stages write.
  - Before any work the leaves are checked against their manifest.
    What aggregated them has to be what the config says, field by
    field, and a difference is refused by name. The weights artifact
    is hashed if it is where the config says, and refused if it is not
    the one the leaves were made with. A path that differs is not a
    reason to refuse: the run warns and records it. Each leaf file is
    checked by its size before work and by its footer, its row groups
    and its ordinals at every open.
  - `--verify-leaves` hashes every leaf file against the manifest
    first. It is off by default because it reads every byte of every
    leaf. Without it a file changed in place, with the same size and
    footer, is not noticed.
  - Fewer leaves than were aggregated can be reduced, with a narrower
    `tree.expect` or a longer `tree.exclude`. The manifest lists the
    leaves left out, and `n_members` is then written whether
    `stats.include_n_members` asks for it or not.
  - `output.write_reduced` is refused under `reduce`, and so is a
    config without `[weights]`.
  - The statistics manifest records the stage, the backend with its
    options and the versions of pandas, numpy and pyarrow, the blocks,
    the SHA256 of the leaves manifest, and whether the leaves were
    verified.
- New: the `reduce` package. The pipeline asks a backend for the
  statistics of a set of leaves with a `ReduceRequest`, and
  `[reduce] kind` picks the backend. `"pandas"` is the only one. A
  request gives each leaf its own windows, a label with a first and a
  last time value, so two leaves can give one label different spans.
  Every config gives every leaf the same list today. If the windows of
  a request do differ between leaves, `n_members` is written.
  `tests/integration/test_reduce_parity.py` holds the comparison a
  second backend has to pass.
- Changed: a leaf whose columns differ from the first leaf's stops the
  run when it arrives. Before, the run wrote every leaf and found out
  at the end.
- New: a run that writes leaves prints how many files under `leaves/`
  are in no leaves manifest there, with their total size, and `reduce`
  prints it too. Nothing is deleted.
- Fixed: the row sort of the statistics picked rows by the label of
  their index, so a frame whose index repeated came out with rows
  repeated. No run of the pipeline built such a frame.
- The pipeline writes a statistics file and its manifest under a
  temporary name and renames them, the manifest last.
- New: `stats.time_reduction` in a pipeline config, with no default:
  `"window_mean"`, `"pooled"` or `"per_year"`. `"window_mean"` is what
  every release so far has done. Under `"pooled"` no window mean is
  taken. Every year of a member
  inside a window is a member of the sample, so the statistics are
  over models, draws and years together, and a year inside two windows
  is pooled in both. A model weight applies to each member-year. A
  member that lacks years has fewer rows and counts for less. On a
  tree where every member has every year, the mean is the mean of the
  window means up to rounding, and the quantiles differ.
  The pooled rows are the leaves' own rows, so the statistics come
  from the leaf files by blocks under every stage. Stage `all` then
  writes the leaves and reduces them. It needs
  `output.write_leaves = true` and `[reduce.pandas] memory_gb`, and
  refuses to start without them. If its second pass fails, the leaves
  stay and `--stage reduce` runs from them. `output.write_reduced` and
  a run without `[weights]` are refused, and so is `stats.time_col`
  among `stats.sample_dims`.
- New: under `"per_year"` no window mean is taken and the year is
  kept. The statistics are over the members within each year from the
  first to the last of `stats.years`, which replaces `stats.windows`,
  and the file has an integer `year` column where the others have
  `window`, sorted where the window is. A year a member lacks is left
  out of that year's statistics and shows in `n_members`. It runs from
  the leaf files under every stage, as `"pooled"` does, and is refused
  with warming levels, since a level is reached over a period. With one
  window a year, `"window_mean"` gives the same statistics, to the bit,
  under another column name.
- New: with `"pooled"` and `stats.include_n_members` the statistics
  file has an `n_sample` column after `n_members`. `n_sample` is the
  member-years behind a statistic. `n_members` keeps its meaning, the
  members behind it: those with a value in the window and, with model
  weights, a weight above zero. Without `"pooled"` no file gains a
  column.
- A config with `stats.pool_years`, a boolean an earlier build of this
  release had, is refused with the value of `stats.time_reduction` to
  write.
- Changed: a tree level named `n_sample` is refused before any leaf is
  read, as `n_members` is.
- The pipeline manifest records `time_reduction`, and `years` for a
  product per year. A run of stage `all` that pools or keeps the years
  writes the manifest the reduce stage writes, with `"stage": "all"`.
- New: `ReduceRequest.time_reduction`, and `ReduceBackend.check_options`,
  which the pipeline calls before it aggregates anything.
- Changed: the pandas reduce backend plans a block at 240 bytes for a
  row it keeps and 40 more for each dimension the leaf has of its own,
  where it planned 130 for any leaf. For one memory budget the blocks
  hold about half the rows they held, or fewer for a leaf with
  dimensions, so a reduce opens each leaf about twice as often. The
  resident memory of a worker grew by up to 155 bytes a kept row on
  leaves with no dimension and by 168, 194 and 218 with one, two and
  three, on synthetic leaves on macOS, and 130 left nothing over
  that. The new figures cover each by half again or more. The dry run
  prints the figure it used, and the manifest has it as
  `reduce.bytes_per_row` next to `reduce.max_worker_peak_gb`.
- New: `read_gmst`, `assign_warming_levels`, `GMSTTable`, `Period`,
  `WarmingLevels` and `LevelWindow`, in `cil_regionalization.warming_levels`
  and at the package root. A member is one series of global mean
  temperature, named by any columns of the table. Its series is
  rebased to a reference period, an offset is added, and its mean over
  a period's years picks the level whose band holds it, both ends of a
  band included. A mean in no band gives no level. Two bands may share
  an end, where the level declared first wins, and may not overlap.
  The reference period, the offset, the levels, the half-width, the
  periods and a statement of how the series was smoothed have no
  default, and a call without one of them fails. A period carries the
  years the temperature is averaged over and the years of the impacts
  separately. Nothing is smoothed. Names are matched as written, so a
  surrogate needs its own rows. `WarmingLevels.record` is the manifest
  entry: the table's hash, every setting, each member's mean and level,
  the count at each declared level and the members with none. The
  pipeline does not use any of this yet.
- New: a cell of `tree.exclude` takes `"*"` for every value of a
  level, so a model that one scenario lacks is one line and not one
  line for each batch, socioeconomic model and scenario. A cell of
  stars alone is refused. The manifest records the cells as written.
- New: `[stats.warming_levels]` in a pipeline config, in place of
  `stats.windows`. The statistics are over the members at a warming
  level in a period, with the levels from `assign_warming_levels`. A
  leaf's member is read off the tree levels named in `member`, so no
  level name is built in. Each period gives the years the temperature
  is averaged over and the years of the impacts separately. The
  statistics file gains a `warming_level` column next to `window`, and
  two counts on every row: `n_members`, the draws behind a statistic,
  and `n_bin_members`, the distinct members of the temperature table
  behind it. A level nobody is at has no row and is in the manifest
  with a zero. The statistics come from the leaf files under every
  stage, so stage `all` writes the leaves and then reduces, and the
  reduce stage gives other levels, periods or the other reduction
  from the same leaves. `stats.time_reduction` chooses between
  collapsing each member's window first and pooling its years, and
  `"per_year"` is refused with warming levels.
  Refused: a config without `reference_period`, `offset`, `levels`,
  `half_width`, `smoothing` or `periods`; `stats.windows` or
  `stats.model_weights` next to it; a member column that is not a tree
  level or is not pooled; a run without `[weights]`;
  `output.write_reduced`; a declared member with no row in the table,
  before any leaf is read; and a leaf that lacks a year of an impact
  span its member is pooled over, as a row or as a value. The manifest
  has the whole record under `warming_levels`. A run without warming
  levels writes `"warming_levels": null` and is otherwise unchanged.
- New: `LeafRequest.bin_member`, `ReduceRequest.include_n_bin_members`
  and `ReduceRequest.whole_spans` in the `reduce` package.
- Changed: a tree level named `warming_level` or `n_bin_members` is
  refused before any leaf is read.
- Changed: `stats.windows` may be left out when `[stats.warming_levels]`
  is set. A config with neither is refused, as it was.
- Changed: `quantile_method="linear"` is computed by numpy, so its
  quantiles are those of `numpy.quantile(method="linear")` and of
  xarray's `.quantile` to the last bit, and so those of the Inequality
  Paper's valued path. Before, pandas computed them. The definition is
  the same, and pandas' arithmetic differs from numpy's in the last
  bit of about one quantile in a hundred on synthetic samples, up to
  256 units in the last place next to zero. Where numpy's
  interpolation meets an infinite member it gives NaN, even for the
  median of 1, 2 and an infinity, where pandas gave 2. Such a quantile
  is refused, naming the group, rather than written as a NaN that
  would read as a group with no value.
- Changed: under `summation="exact"` the weighted quantiles of
  `"lower_step"` and `"inverted_cdf"` take their cumulative weight
  shares from exact sums, each share their ratio rounded once, and
  `"lower_step"` takes equal values in order of weight, as
  `"inverted_cdf"` already did. Before, a share that equals q in exact
  arithmetic could land a bit either side of it, depending on the order
  the weights were added in, and then decide the quantile: about one in
  a few hundred on weights of 0.1 to 0.7. And within equal values the
  step depended on which member came first, so on the order of the rows,
  1 in one order and `-inf` in another on six members. Without weights
  nothing changes. On 2,880 members a group the step costs what it did
  under `"exact"`, about 1.3 ms, and the weighted standard inverse 1.2 ms
  where it took 0.6 ms. `summation="fast"` is unchanged.
- Changed: `"lower_step"` sorts a group once for all its quantiles,
  where it sorted it once a quantile. The values are the same, and under
  `"fast"` the step is about three times faster on large groups.
- Fixed: a refusal raised while a reduce block is pooled named the
  target by its ordinal in the leaf file. It names it by its id fields,
  as the statistics file does.
- Changed: the lowest pandas the package declares is 2.1, up from 2.0.
  On pandas 2.0 a grouped sum or mean of an infinite value is NaN, so a
  statistic that is infinite came back as NaN, and a refusal of an
  infinite statistic was never reached. The lowest version of every
  dependency the package declares is now installed and tested on every
  push, on Python 3.10.
- Fixed: `cross_backend_compare` and `legacy_compare` needed pandas 2.2
  to classify a region, although the package declared 2.0.
- Fixed: the data scripts of `examples/energy` and `examples/labor`
  failed with a NameError when they copied the SMME weights.
- New: `examples/france/`, the whole chain in one marimo notebook, for
  mortality in metropolitan France. It builds weights on ADMIN EXPRESS
  boundaries, compares them with the published GADM weights, applies
  them three ways, and runs the pipeline for the three time reductions
  and the warming levels. The test suite runs it on synthetic data.
  The `[examples]` extra adds marimo, and CI installs it.
- New: `docs/reference.md` lists every public name with its signature
  and its docstring, and every key of the weights config and of the
  pipeline config with its type and its default. `docs/make_reference.py`
  writes it from the package, and the test suite fails when it is out of
  date. The README opens with what the library does and where each task
  is described, and installing comes before the first example.

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
