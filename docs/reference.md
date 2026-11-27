# Reference

Written by `docs/make_reference.py` from the package itself. Do not
edit it by hand. The names below are what `import cil_regionalization
as cilreg` gives. Anything else in the package is internal and may
change.

- [Published weights](#published-weights)
- [Statistics over the members](#statistics-over-the-members)
- [Warming levels](#warming-levels)
- [Monte Carlo and delta-method trees](#monte-carlo-and-delta-method-trees)
- [Generating weights](#generating-weights)
- [The weights config](#the-weights-config)
- [The pipeline config](#the-pipeline-config)

## Published weights

### `fetch_weights`

```python
fetch_weights(
    name: str,
    *,
    record: str | None = None,
    base_url: str | None = None,
    parquet: str | None = None,
    manifest: str | None = None,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> WeightsArtifact
```

The published weight artifact ``name`` as a `WeightsArtifact`
ready for `apply_weights`, verified and cached.

``name`` is a registry name. With ``record`` given, a Zenodo record
id or DOI, the registry is not read, ``name`` only labels the cache
entry, and ``parquet`` and ``manifest`` name the record's files.
``refresh`` downloads again even when a verified cached copy exists.
Raises if ``name`` is in no registry, if a cached copy fails its
check, or if a download fails either checksum.

### `list_cached`

```python
list_cached(cache_dir: str | Path | None = None) -> list[dict[str, Any]]
```

The cached weight artifacts, each a dict of 'name', 'record',
'path' and 'size_bytes'. Cached model weights are not listed.

### `clear_cache`

```python
clear_cache(
    cache_dir: str | Path | None = None,
    *,
    name: str | None = None,
) -> int
```

Removes the cached weight artifacts, or only those called ``name``,
and returns how many it removed. Cached model weights stay.

### `WeightsArtifact`

```python
class WeightsArtifact:
    frame: pd.DataFrame
    schema: OutputSchema
    regions_version: str | None
    source_version: str | None
    partial_coverage: dict | None = None
```

A weights frame plus the metadata needed to apply it safely.

``partial_coverage`` is the record of partially covered source units
that every generated polygon artifact carries, a count of zero
included. It holds the threshold, the count, the units below the
threshold and whether the targets were a subset. `apply_weights`
refuses an artifact whose count is not zero unless
``allow_partial_coverage`` is set. It is None for an artifact built
by hand, a grid artifact and one loaded from a manifest without the
record, and such an artifact is not refused.

#### `WeightsArtifact.normalization`

```python
WeightsArtifact.normalization: str
```

#### `WeightsArtifact.from_result`

```python
@classmethod
WeightsArtifact.from_result(result: WeightsResult) -> WeightsArtifact
```

#### `WeightsArtifact.load`

```python
@classmethod
WeightsArtifact.load(
    output_dir: str | Path,
    stem: str = 'weights',
) -> WeightsArtifact
```

The artifact written as ``{stem}.parquet`` and
``{stem}.manifest.json`` in ``output_dir``, a local directory or a
``gs://`` prefix. The schema is rebuilt from the column roles the
manifest records. Raises if the manifest does not record them, as
a manifest from an older version of the package does not.

#### `WeightsArtifact.from_table`

```python
@classmethod
WeightsArtifact.from_table(
    table: pd.DataFrame | str | os.PathLike,
    *,
    id_fields: Sequence[str],
    weights: Mapping[str, str],
    source: str,
    source_keys: Sequence[str] | None = None,
    normalization: str = 'per_destination',
    regions_version: str | None = None,
    check: bool = True,
    tolerance: float = 1e-06,
) -> WeightsArtifact
```

An artifact built from a weight table made elsewhere, with
what a manifest would record passed as arguments.

``table`` is a frame, or the path of a CSV or parquet file, with
one row per (region, source unit) and one normalized column per
weight. ``weights`` maps each weight name to its column, for
example ``{"pop": "popwt"}``, and the columns are renamed to the
``{name}wt`` form `apply_weights` reads. ``source`` becomes the
artifact's ``source_version``, which `apply_weights` compares
with the data's ``data_version`` as an exact string.

When ``source`` names a grid in the form `GridSpec.source_version`
writes, such as ``grid-0.25deg-center-[-180,180)``, the source
units are grid cells. Without ``cell_ix`` and ``cell_iy`` they are
found from a longitude and a latitude column, spelled
``cell_lon``/``cell_lat``, ``lon``/``lat``,
``longitude``/``latitude`` or ``pix_cent_x``/``pix_cent_y``.
Longitudes are wrapped into the grid's convention first, so the
180.125 centroid CIL files carry becomes -179.875, and every
point must sit on a cell centroid. A table's own ``cell_ix`` and
``cell_iy`` are used as they are after a bounds check, and must
map back to its longitude or latitude column when it has one.
With neither column, nothing checks that the indices were built
on the grid ``source`` names.

Any other ``source`` is a polygon source version, and
``source_keys`` names the id columns of its units. With
``check``, the table must have no duplicated (region, source
unit) rows and every weight must sum to 1 within each
normalization group. Both failures raise naming the offending
groups.

### `apply_weights`

```python
apply_weights(
    artifact: WeightsArtifact,
    data: pd.DataFrame,
    *,
    kind: VariableKind,
    weight: str,
    data_version: str | None,
    value_col: str = 'value',
    denominator_col: str | None = None,
    policies: SourceUnitPolicies | None = None,
    restrict_to_sources: Iterable[tuple] | None = None,
    group_col: str | None = None,
    mass_balance_tolerance: float = 1e-06,
    allow_partial_coverage: bool = False,
) -> AppliedResult
```

Aggregate source-level ``data`` to the artifact's target units.

``data`` must carry the artifact's source key columns (exact names),
``value_col``, and for ratio kind ``denominator_col``. Every other
column is a passthrough dimension. Aggregation happens independently
per combination of dimension values, and the output keeps those
columns. Columns that are labels of the source unit rather than
dimensions (a country tag, a display name) behave as dimensions too,
so drop them beforehand if they should not appear in the output. Rows
duplicated on (source keys + dimensions) would double count and raise.

``data_version`` states the geometry version the data was built
against and is compared to the artifact's recorded source version.
``group_col`` optionally names a data column to run the mass balance
per group (typically the country) as well as globally.
``restrict_to_sources`` limits the weight frame to an explicit source
universe for subset runs. The coverage checks stay fully strict within
that universe.

Every row of the output carries ``weight_share``, the share of its
target's weight basis behind the value, exactly 1 when every unit of
the target that carries basis contributed. The basis is the
artifact's raw column of the weight, its population or crop area, so
a unit with no crop area takes nothing from a target's share, and a
target with no basis at all, fed by the area fallback alone, is
measured by its weights. ``AppliedResult.targets`` lists the targets
the weights have that are not in the output, with the reason, and
the partial ones. A sum over no value is NaN, not zero.

### `SourceUnitPolicies`

```python
class SourceUnitPolicies
```

What to do with source units that cannot take part in applying the
weights, the ``[policies]`` of a pipeline config and the ``policies``
of `apply_weights`.

Each key is "error" or "skip". "error" raises naming the units.
"skip" leaves them out with a warning, and the result lists them in
its ``coverage``. ``on_unmatched`` is for units in the data that are
not in the weight file, whose values cannot reach any target.
``on_zero_weight`` is for units in the weight file whose weights are
all zero or NaN. ``on_absent_from_data`` is for units in the weight
file that the data does not have, whose targets lose what the
weights gave them.

A pydantic model. Its keys are listed under `[policies]` of [The pipeline config](#the-pipeline-config), and `apply_weights` takes them as arguments.

### `read_netcdf_grid`

```python
read_netcdf_grid(
    path: str | Path,
    variables: Sequence[str],
    *,
    lat: str = 'lat',
    lon: str = 'lon',
    grid: str | GridSpec | None = None,
) -> tuple[pd.DataFrame, str]
```

``(cells, grid)`` from a NetCDF on a regular lat-lon grid.

``cells`` has one row per cell and slice, with ``cell_ix`` and
``cell_iy`` on the grid, one column per variable, and any other
dimension, such as time, as a column. ``grid`` is the identity
string, for example ``grid-0.25deg-center-[-180,180)``, to pass to
`apply_weights` as ``data_version``. ``lat`` and ``lon`` name the
file's 1-D coordinate variables, each the coordinate of a dimension
of the same name, and every variable must have both.

The grid is ``grid`` if given, as the identity string or a generated
`GridSpec`, and otherwise `infer_grid_version` derives it from the
coordinates. With ``grid``, longitudes are wrapped into its
convention, so a [0,360) file can be put on a [-180,180) grid. The
indices come from each row's own coordinates, so the order of the
axes does not matter, and nothing is regridded or snapped.

A file whose longitudes all lie in [0,180], such as a cut of Europe,
Asia or Africa, fits both longitude conventions and needs ``grid``.
Without it the error prints the two calls that would work. Without
``grid``, a file whose latitude and longitude steps differ also
raises, since a grid identity holds one resolution.

Cells masked (NaN) in every slice are left out, as in
`DailyGrid.to_cells`, and `apply_weights` reports a weighted cell
that has no data. Raises if a variable or coordinate is missing, if
any coordinate is not a cell centre on the grid, if two coordinates
land on one cell, such as longitudes of both -180 and 180, or if a
cell is masked in some slices and not others.

### `OutputSchema`

```python
class OutputSchema:
    id_fields: tuple[str, ...]
    weight_names: tuple[str, ...]
    source_units: SourceUnits = GRID_CELLS
    normalization: Normalization = 'per_destination'
```

Resolved column layout for one run, the id fields, then the source
units, then three columns for each weight. By default the source
units are grid cells and the normalization is per_destination.

#### `OutputSchema.columns`

```python
OutputSchema.columns: tuple[str, ...]
```

#### `OutputSchema.dtypes`

```python
OutputSchema.dtypes: dict[str, str]
```

#### `OutputSchema.normalization_group`

```python
OutputSchema.normalization_group: tuple[str, ...]
```

Columns the weights sum to 1 within, per the declared direction.

#### `OutputSchema.empty_frame`

```python
OutputSchema.empty_frame() -> pd.DataFrame
```

#### `OutputSchema.validate_frame`

```python
OutputSchema.validate_frame(df: pd.DataFrame) -> None
```

Raise if `df` is missing columns or has wrong dtypes for declared cols.

### `SourceUnits`

```python
class SourceUnits:
    key_columns: tuple[str, ...]
    dtype_items: tuple[tuple[str, str], ...]
    meta_columns: tuple[str, ...] = ()
```

Identifier layout for the source side of a weights frame.

``key_columns`` identify one source unit, and are the keys of joins
and duplicate checks. ``meta_columns`` are carried alongside but are
not part of the identity, as the grid case carries cell centroids.
``dtype_items`` holds (column, dtype) pairs for every column.

#### `SourceUnits.columns`

```python
SourceUnits.columns: tuple[str, ...]
```

#### `SourceUnits.dtypes`

```python
SourceUnits.dtypes: dict[str, str]
```

#### `SourceUnits.from_string_ids`

```python
@classmethod
SourceUnits.from_string_ids(key_columns: Iterable[str]) -> SourceUnits
```

Source units identified by string id columns (e.g. ``hierid``).

### `GRID_CELLS`

```python
GRID_CELLS = SourceUnits(key_columns=('cell_ix', 'cell_iy'), dtype_items=(('cell_ix', 'int64'), ('cell_iy', 'int64'), ('cell_lon', 'float64'), ('cell_lat', 'float64')), meta_columns=('cell_lon', 'cell_lat'))
```

## Statistics over the members

### `summarize_samples`

```python
summarize_samples(
    data: pd.DataFrame,
    *,
    sample_dims: Sequence[str],
    time_col: str,
    window: tuple[float, float],
    value_col: str = 'value',
    quantiles: Sequence[float] = (),
    include_mean: bool = True,
    weight_col: str | None = None,
    include_n_members: bool = False,
    model_dims: Sequence[str] | None = None,
    quantile_method: str,
    summation: str,
) -> pd.DataFrame
```

Window mean per sample member, then statistics over the pooled sample.

``data`` has one value per (identity, sample, time) row. Identity
columns are all the columns not named as sample, time, value or
weight. Statistics are computed per identity combination, and those
columns pass through. ``window`` is an inclusive (start, end) pair on
``time_col``.

``weight_col`` names a per-member weight, constant across the
member's time rows. Weights must be finite and non-negative. A zero
weight drops the member before the mean, the counts and the
quantiles, under every method, so no quantile returns the value of
a member that weighs nothing. Weights are not renormalized within a
model, so in an unbalanced sample a model's total weight grows with
how often it appears. Leaving ``weight_col`` unset with
``model_dims`` given warns with `UnweightedModelWeightsWarning`.

Returns a long frame with the identity columns, 'statistic' and
``value_col``. Labels are 'mean' and 'qNN' in percent ('q05', 'q95',
'q2.5').

``include_n_members`` adds two integer columns after ``value_col``.
'n_members' is the samples behind a statistic, the members with a
window mean that is not NaN and, when weighted, a weight above zero,
so each batch of a model counts once. 'n_models' is the climate
models among them, the distinct values of ``model_dims``, which it
then has to name, such as ``["gcm"]``. ``data`` must not already
have columns of those names.

``quantile_method`` is "inverted_cdf", "linear" or "lower_step",
and has no default. "inverted_cdf" is the estimator of a quantile.
It returns, weighted or not, the smallest member value whose
cumulative weight share reaches q, always a value a member has.
"linear" interpolates between two members and can return a number
no member holds. It is refused with weights. "lower_step" is for
reproducing the historical extraction tool, not for estimating a
quantile. It returns the member one below the one "inverted_cdf"
returns, so it is biased low at every quantile, except at 0.5 with
members that count equally, where it is the plain median. A
quantile at or below the first member's share is ``-inf``, and with
``weight_col`` unset every member counts the same.

``summation`` is "fast" or "exact", and has no default. "fast" is
the vectorised sum, whose last bits depend on the order of the
rows. "exact" takes the means from exactly rounded sums, whatever
the order. An infinite member value raises, and so does a statistic
that comes out infinite other than the step's ``-inf``.

The statistics are float64 whatever the dtype of ``value_col``. A
column of another dtype is widened before the window mean, and a
nullable one comes back with NaN where it had no value.

Raises if ``data`` already has a 'statistic' column, or a column
named for a statistic, such as 'q05', 'mean' or 'deaths_q95'.

### `window_means`

```python
window_means(
    data: pd.DataFrame,
    *,
    time_col: str,
    windows: Sequence[tuple[float, float]],
    value_col: str = 'value',
    window_col: str = 'window',
    summation: str,
) -> pd.DataFrame
```

Mean over each time window, per combination of the other columns.

It is the first half of `summarize_samples` and never mixes sample
members, so taking it leaf by leaf and pooling later gives the same
statistics. ``time_col`` is replaced by ``window_col``, labelled like
"2040-2059", or by the year alone for a one-year window.

Raises if ``data`` already has ``window_col``, since its years are
gone and other windows need the unreduced data. Rows repeated on
every column but ``value_col`` raise too, in any year, including
years no window covers, and so does an infinite value in a window.

``summation`` has no default. "exact" takes each mean from an
exactly rounded sum, and "fast" is the pandas mean. The mean is
float64 whatever the dtype of ``value_col``, and a column of another
dtype is widened before the mean is taken. A nullable column comes
back with NaN where it had no value.

### `pooled_statistics`

```python
pooled_statistics(
    data: pd.DataFrame,
    *,
    sample_dims: Sequence[str],
    value_col: str = 'value',
    quantiles: Sequence[float] = (),
    include_mean: bool = True,
    weight_col: str | None = None,
    include_n_members: bool = False,
    model_dims: Sequence[str] | None = None,
    quantile_method: str,
    summation: str,
) -> pd.DataFrame
```

Statistics over the pooled sample, for data reduced by `window_means`.

The second half of `summarize_samples`. The window column is an
ordinary identity column here. ``weight_col``,
``include_n_members``, ``model_dims``, ``quantile_method`` and
``summation`` behave the same, and the same input is refused.
Returns identity columns, 'statistic', ``value_col``, then
'n_members' and 'n_models' if asked for.

### `mixture_quantile`

```python
mixture_quantile(weights, means, variances, q)
```

Quantile of a weighted mixture of normal distributions.

The last axis of ``weights``, ``means`` and ``variances`` is the
mixture component. Any leading axes (regions, say) are solved
independently and broadcast against each other, so one weight
vector can serve every region. Weights are rescaled to sum to one
per region, and components with zero weight are ignored. A zero
variance component is a point mass at its mean.

The quantile is the smallest x whose mixture CDF reaches ``q``.
``q`` may be a scalar or an array. The result has the leading shape
followed by the shape of ``q``, and is a plain float when both are
scalar. ``q = 0`` and ``q = 1`` return the ends of the support,
which are infinite when any weighted component has positive variance
and otherwise the smallest or largest point mass.

### `read_model_weights`

```python
read_model_weights(
    path: str | Path,
    *,
    table_names: Optional[Mapping[str, str]] = None,
    sep: str = '\t',
) -> ModelWeights
```

Read a weights table. The file is read once, and the hash is of
the bytes the rows were parsed from. ``table_names`` declares the
row of each member whose name is not that of its row, member name to
table name, and is checked where a member takes its row. The
published SMME tables need it until they carry the members' names.
Raises if a name is in the table twice, or if two members are
declared to one row.

### `fetch_model_weights`

```python
fetch_model_weights(
    rcp: str,
    *,
    record: str = MODEL_WEIGHTS_RECORD,
    base_url: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    table_names: Mapping[str, str] | None = None,
)
```

The published climate model weights of ``rcp``, as
`read_model_weights` reads them with ``table_names``, from the file
``<rcp>_SMME_weights.tsv`` of their Zenodo record. The download is
checked against the md5 the record declares and cached with it, and
a cached copy that no longer matches raises. The published tables
do not spell their rows as the trees spell the members, so until
they are rewritten every member but one needs its row declared.

### `ModelWeights`

```python
class ModelWeights:
    path: str
    sha256: str
    names: tuple[str, ...]
    weights: tuple[float, ...]
    table_names: Mapping[str, str] = field(default_factory=dict)
```

The rows of a weights table, in file order, with the SHA256 of
the bytes they were parsed from. ``names`` are as written in the
file, and ``table_names`` the rows declared for members whose names
are not those of their rows.

#### `ModelWeights.missing`

```python
ModelWeights.missing(members: Iterable[str]) -> list[str]
```

The members that have no row in the table, sorted.

#### `ModelWeights.check`

```python
ModelWeights.check(members: Iterable[str]) -> None
```

Raises if a member has no row in the table, naming the row
each one comes closest to, or if two members take one row.

#### `ModelWeights.weight_of`

```python
ModelWeights.weight_of(member: str) -> float
```

The weight of one member. Raises if it has no row.

#### `ModelWeights.attach`

```python
ModelWeights.attach(members: pd.Series) -> pd.Series
```

The weight of each row's member, as float64 on the index of
``members``. Raises if a member has no row in the table, if two
members take one row, or if a row names no member.

#### `ModelWeights.record`

```python
ModelWeights.record(members: Iterable[str]) -> dict
```

The manifest entry of a run, given the member of each leaf
that was pooled. Its 'members' holds each member with its table
row, its weight and its number of leaves, and 'unmatched_rows'
the weight of every row no member took, so the two together
cover the whole table. 'table_names' are the rows declared, and
'table_names_unused' those declared for a member the run does
not have. Raises as `check` does.

### `regroup`

```python
regroup(
    reduced: pd.DataFrame,
    *,
    id_fields: Sequence[str],
    by: Sequence[str],
    kind: Literal['extensive', 'intensive'],
    value_col: str,
    summation: str,
    basis: pd.DataFrame | None = None,
) -> pd.DataFrame
```

``reduced`` with its targets grouped by the id fields ``by``, one
value a group and every other column, so that a country is the sum
or the mean of its states in each window and sample member.

"extensive" sums the values and takes no ``basis``. "intensive"
takes the mean weighted by each target's basis, the frame
`target_basis` gives, over the targets with a value. A group with no
value is NaN, and so is an intensive group whose targets with a
value have a basis of zero. ``summation`` is "fast" or "exact", as
for the statistics. Raises if ``by`` is not among ``id_fields``, if
``basis`` is given for "extensive" or left out for "intensive", or
if a row's target has no basis.

### `target_basis`

```python
target_basis(artifact: WeightsArtifact, weight: str) -> pd.DataFrame
```

Each target's weight basis, the sum of the raw column of
``weight`` over its rows, its population or its crop area, as the id
fields and a column 'basis'. A row with no basis counts as none.
Raises if the artifact has no raw column for the weight.

### `UnweightedModelWeightsWarning`

```python
class UnweightedModelWeightsWarning(UserWarning)
```

Pooling climate models without weights.

Suppress with
``warnings.filterwarnings("ignore", category=UnweightedModelWeightsWarning)``.

## Warming levels

### `read_gmst`

```python
read_gmst(
    path: str | Path,
    *,
    member_cols: Sequence[str],
    time_col: str = 'year',
    value_col: str = 'temp',
    sep: str = ',',
) -> GMSTTable
```

Read a temperature table. The file is read once, and the hash is
of the bytes the rows were parsed from. The member columns are read
as text, so a name like '06' keeps its zero.

### `GMSTTable`

```python
class GMSTTable:
    frame: pd.DataFrame
    member_cols: tuple[str, ...]
    path: Optional[str] = None
    sha256: Optional[str] = None
```

A temperature table with one row for a member and a year.
``frame`` has the member columns as strings, then an integer 'year'
and a float 'value'. ``path`` and ``sha256`` are those of the file
it was read from, or None if it was built from a frame.

#### `GMSTTable.from_frame`

```python
@classmethod
GMSTTable.from_frame(
    frame: pd.DataFrame,
    *,
    member_cols: Sequence[str],
    time_col: str = 'year',
    value_col: str = 'temp',
    path: Optional[str] = None,
    sha256: Optional[str] = None,
) -> GMSTTable
```

Check a table and keep its member, year and value columns.
Raises if a column is missing, a member or a year is missing
from a row, a year is not a whole number, or a member has two
rows for one year.

#### `GMSTTable.members`

```python
GMSTTable.members: list[tuple[str, ...]]
```

The members, in the order they first appear in the table.

#### `GMSTTable.series`

```python
GMSTTable.series() -> dict[tuple[str, ...], dict[int, float]]
```

The value at each year of each member.

### `assign_warming_levels`

```python
assign_warming_levels(
    table: GMSTTable,
    *,
    reference_period: tuple[int, int],
    offset: float,
    levels: Sequence[float],
    half_width: float,
    periods: Sequence[Period],
    smoothing: str,
    members: Iterable[Sequence[str]],
) -> WarmingLevels
```

The level of every member in every period.

A member's series is rebased by taking off its own mean over
``reference_period`` and adding ``offset``. Its mean over a
period's ``gmst_span`` then picks the level. A level holds a mean
from ``half_width`` below it to ``half_width`` above it, both ends
included. Two bands may share an end and may not overlap. A mean
on a shared end goes to the level declared first.

``smoothing`` says how the series in the table was smoothed, in
words, and goes into the record as it is. Nothing is smoothed
here. ``members`` are the members that join, each one value for
each member column, and have no default. A member of the table that
is not among them is at no level, and the record lists it. Raises if a
member that joins has no row, or lacks a value for a year of the
reference period or of a ``gmst_span``.

### `members_that_join`

```python
members_that_join(
    join: str | Mapping[str, Sequence[str]],
    declared: Iterable[Sequence[str]],
    member_cols: Sequence[str],
) -> list[tuple[str, ...]]
```

The members of ``declared`` that join a warming level, in their
order. ``join`` is "declared", under which every one of them joins,
or a mapping of member columns to the values that join, under which
a member joins when its value in each column named is listed.
Raises if a column named is not a member column, if a list is empty
or repeats a value, or if a value listed is held by no member of
``declared``.

### `Period`

```python
class Period:
    gmst_span: tuple[int, int]
    impact_span: tuple[int, int]
```

A calendar period. ``gmst_span`` is the years the temperature is
averaged over to pick a member's level, and ``impact_span`` the
years of the impacts that go under that level. Both include their
first and last year, and the two may differ.

#### `Period.label`

```python
Period.label: str
```

The period as a window label, from the years of the impacts.

### `WarmingLevels`

```python
class WarmingLevels:
    table: GMSTTable
    reference_period: tuple[int, int]
    offset: float
    levels_declared: tuple[float, ...]
    half_width: float
    periods: tuple[Period, ...]
    smoothing: str
    means: dict[tuple[str, ...], tuple[float, ...]]
    levels: dict[tuple[str, ...], tuple[Optional[float], ...]]
    excluded: tuple[tuple[str, ...], ...] = ()
```

The level of each assessed member of a temperature table in
every period, and what decided it. ``means`` and ``levels`` hold
one value a period for each assessed member, in the order of
``periods``. A level is None where the member's mean lies in no
band. ``excluded`` are the members of the table that do not join,
which are at no level.

#### `WarmingLevels.member_cols`

```python
WarmingLevels.member_cols: tuple[str, ...]
```

#### `WarmingLevels.assignments`

```python
WarmingLevels.assignments: pd.DataFrame
```

One row for a member and a period, with the member columns,
'window', 'gmst_mean' and 'warming_level', which is NaN where
the member has no level.

#### `WarmingLevels.missing`

```python
WarmingLevels.missing(
    members: Iterable[Sequence[str]],
) -> list[tuple[str, ...]]
```

The members that have no row in the table, sorted.

#### `WarmingLevels.windows_of`

```python
WarmingLevels.windows_of(member: Sequence[str]) -> tuple[LevelWindow, ...]
```

The periods in which a member has a level, in the order of
``periods``, none for a member that does not join. Raises if the
member has no row in the table.

#### `WarmingLevels.record`

```python
WarmingLevels.record(members: Optional[Iterable[Sequence[str]]] = None) -> dict
```

What a manifest says of a run, as plain values. ``members``
are the members of the run, the members that join if it is left
out. Every member of the table that does not join is under
``excluded`` with the reason. Raises if a member of the run has
no row in the table, or a member that joins is not in the run.

### `LevelWindow`

```python
class LevelWindow(NamedTuple):
    level: float
    window: str
    start: int
    end: int
```

A level a member has in a period, with the label of the period
and the first and last year of the impacts that go under it.

## Monte Carlo and delta-method trees

### `read_netcdf_leaf`

```python
read_netcdf_leaf(
    path: str | Path,
    *,
    variables: Sequence[str],
    region_dim: str,
    region_col: str,
    region_labels: str | None = None,
    kind: str | None = None,
) -> pd.DataFrame
```

One NetCDF leaf as a long frame, one row per combination of the
dimensions of ``variables``.

``variables`` are the data variables to carry, one for extensive or
intensive application and the numerator and denominator for ratio.
``region_dim`` is the file's name for the region dimension, and it
is renamed to ``region_col``, the weights artifact's source key
column. Some trees have no coordinate on the region dimension and
keep the region ids in a separate variable, as the mortality trees
do in ``regions``. ``region_labels`` names that variable, and its
values become the region coordinate. Without it such a region
column comes out as positional integers.

With ``kind="extensive"`` the read raises if a variable's ``units``
attribute contains a percent sign, since that variable is a ratio. A
ratio whose units have no percent sign, or that has no units, is not
caught. Also raises if a variable is missing, if ``region_dim`` is
not one of their dimensions, if ``region_labels`` is missing or not
on ``region_dim`` alone, or if ``region_col`` is already a column.

In the mortality and labor Monte Carlo trees the effect of climate
change is ``rebased`` from a ``...-combined.nc4`` file minus
``rebased`` from its ``-histclim`` sibling, and the value of one
file alone is not it. No adaptation is the exception and stands
alone. The Climate Impact Lab memo "The art of rebasing and
histclim" gives the reasoning, and the Mexico example in
examples/rates shows the subtraction.

### `read_gradient_leaf`

```python
read_gradient_leaf(
    path,
    *,
    year: int,
    variable: str = 'rebased_bcde',
    vcv_variable: str = 'vcv',
    value_variable: str = 'rebased',
) -> GradientLeaf
```

One year of a delta-method leaf as a `GradientLeaf`.

``variable`` holds the gradients, with coefficient and year as its
first two dimensions in either order. ``vcv_variable`` holds the
covariance matrix, and ``value_variable`` the stored variance, read
when present. Raises if the file has no ``year``, if the gradient's
first dimensions are not coefficient and year, or if the covariance
matrix is not square or does not match the gradient's coefficients.

### `window_gradient`

```python
window_gradient(
    gradients: np.ndarray,
    years,
    window: tuple[float, float],
) -> np.ndarray
```

The mean of stacked yearly gradients over the years of
``window``, both ends included, as in `window_means`.

``gradients`` has year on its first axis, in the order of
``years``, and its other axes pass through. Stacked leaves (year,
coefficient, region) and stacked aggregates (year, target,
coefficient) both work, and windowing before or after
`aggregate_gradient` gives the same answer. NaN entries drop out of
the mean as NaN rows do in `window_means`. Raises if the window
selects no year or starts after it ends, if ``years`` has
duplicates, or if its length does not match the first axis.

### `aggregate_gradient`

```python
aggregate_gradient(
    weights: WeightsArtifact,
    regions: list[str],
    gradient: np.ndarray,
    *,
    weight: str,
    scale: np.ndarray | None = None,
    normalize: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]
```

The target key frame and the gradient aggregated to the weight
file's targets, target by coefficient.

``gradient`` is coefficient by region, in the order of ``regions``,
and the shares are the column ``<weight>wt``, which has no default.
A region is a value of the artifact's one source key column. Weight
rows for other
regions are ignored, as with ``restrict_to_sources``. For a total,
an extensive value, leave ``scale`` and ``normalize`` alone, and
each region's gradient is split by the weight shares and summed. For
a rate, pass the per-region sizes (population) as ``scale`` and set
``normalize=True``, and each target becomes the size-weighted mean
of its regions' gradients. The aggregation is linear, so aggregating
a difference of gradients, such as one against histclim, equals
differencing the aggregated ones.

Raises if the share column is missing, if the artifact's source key
is not one column, if a region has no weight rows, if ``scale`` is not
one value per region, or if a target has zero total weight under
``normalize``.

### `delta_method`

```python
delta_method(
    aggregated: np.ndarray,
    vcv: np.ndarray,
    coefficients: np.ndarray | None = None,
) -> pd.DataFrame
```

A frame of the mean and standard deviation per target from
aggregated gradients, target by coefficient.

The variance is the quadratic form G V G' of each target. The mean
is G times ``coefficients``, exact when the response is linear in
the coefficients. Without ``coefficients`` the frame has only 'sd'.
Raises if the shapes of the inputs do not match.

## Generating weights

### `generate_weights`

```python
generate_weights(config: Config | str | Path) -> WeightsResult
```

Generate a weights file and its manifest from a config.

``config`` is a `Config` or the path to a TOML file. The weights file,
in the configured ``output.format``, and the manifest are written to
``config.output.dir``, as `cilreg run` does, and the result is
returned. Geometries are read from the paths in the config, so
polygons held in memory have to be written to a GeoParquet file
first, which lets the manifest record its hash.

Raises `FileNotFoundError` naming any input that is missing, and
`SumToOneError` when the weights fail their sum-to-one check, in
which case nothing is written.

### `load_config`

```python
load_config(path: str | Path) -> Config
```

The validated `Config` of the TOML file at ``path``.

Relative paths in `regions.path`, `source.path`, `weights[].raster`
and `output.dir` are resolved against the config file's directory,
not the working directory. Absolute paths, `gs://` URIs and BigQuery
`table` references pass through unchanged. Raises
`FileNotFoundError` if the file does not exist, and a pydantic
`ValidationError` naming the offending key if the config is invalid.

### `Config`

```python
class Config
```

A pydantic model. Its keys are listed under [The weights config](#the-weights-config).

### `WeightsResult`

```python
class WeightsResult:
    frame: pd.DataFrame
    schema: OutputSchema
    manifest: Manifest
    sum_report: SumToOneReport
```

A backend's output, the weights frame with its schema, its manifest
and its sum-to-one report.

### `SumToOneError`

```python
class SumToOneError(ValueError)

SumToOneError(report: SumToOneReport, result: WeightsResult)
```

The generated weights failed the sum-to-one check, and nothing was
written.

``report`` is the full `SumToOneReport`, failures included.
``result`` is the computed `WeightsResult`.

## The weights config

`cilreg run <config.toml>` and `generate_weights` read it.

| key | type | default |
| --- | --- | --- |
| `normalization` | "per_destination" or "per_source" | "per_destination" |

### `[project]`

Required.

| key | type | default |
| --- | --- | --- |
| `name` | str, not empty | required |
| `description` | str | not set |

### `[regions]`

Required.

| key | type | default |
| --- | --- | --- |
| `path` | str | not set |
| `table` | str | not set |
| `id_fields` | list[str], not empty | required |
| `version` | str | not set |
| `keep` | dict[str, list[any]] | not set |
| `drop` | dict[str, list[any]] | not set |
| `on_invalid_geometry` | "repair" or "skip" or "error" | "repair" |
| `on_null_geometry` | "error" or "skip" | "error" |
| `on_unknown_id` | "error" or "skip" | "error" |

### `[grid]`

Can be left out.

| key | type | default |
| --- | --- | --- |
| `mode` | "generate" or "prebuilt" | required |
| `resolution` | float | not set |
| `offset` | "center" or "edge" | "center" |
| `lon_convention` | "[-180,180)" or "[0,360)" | "[-180,180)" |
| `path` | str | not set |
| `table` | str | not set |

### `[source]`

Can be left out.

The polygons the weighted data lives on, as the source units of a
polygon-mode run.

Declaring ``[source]`` instead of ``[grid]`` puts a run in polygon
mode. Each row of the weights then pairs a region from ``[regions]``,
the side the weights are built for such as administrative units, with
a polygon from this file such as an impact region keyed by hierid.
Any shapefile or GeoParquet with id columns will do. ``version`` is
the label the manifest records for this source.

``on_zero_overlap`` says what to do with source units whose
intersection with every region has zero geodesic area. "error" raises
naming them, and "skip" leaves them out and lists them in the
manifest's ``extra`` under ``zero_overlap_source_units``. A run over a
regional subset of targets should expect such units and set "skip".

| key | type | default |
| --- | --- | --- |
| `path` | str, not empty | required |
| `id_fields` | list[str], not empty | required |
| `version` | str, not empty | required |
| `on_invalid_geometry` | "repair" or "skip" or "error" | "repair" |
| `on_zero_overlap` | "error" or "skip" | "error" |
| `coverage_threshold` | float, above 0, at most 1 | 0.999 |

### `[[weights]]`

A list of tables, each with these keys.

| key | type | default |
| --- | --- | --- |
| `name` | str, not empty | required |
| `raster` | str | not set |
| `table` | str | not set |
| `fallback` | "area" or "zero" or "error" or "nan" | "area" |
| `kind` | "point_sum" or "area_weighted_sum" | "point_sum" |

### `[backend]`

Required.

| key | type | default |
| --- | --- | --- |
| `kind` | "local" or "bigquery" | required |
| `coverage` | "exact_fraction" or "pixel_centroid" | "exact_fraction" |

### `[backend.local]`

Can be left out.

| key | type | default |
| --- | --- | --- |
| `dask` | "off" or "local" or "slurm" | "off" |
| `n_workers` | int, at least 1 | not set |
| `threads_per_worker` | int, at least 1 | not set |
| `slurm_queue` | str, not empty | not set |
| `slurm_cores` | int, at least 1 | not set |
| `slurm_memory` | str, not empty | not set |
| `slurm_walltime` | str, not empty | not set |
| `min_segment_area_m2` | float, at least 0 | 1.0 |

### `[backend.bigquery]`

Can be left out.

| key | type | default |
| --- | --- | --- |
| `project` | str, not empty | not set |
| `temp_dataset` | str, not empty | not set |
| `dry_run_byte_ceiling` | int, at least 0 | 10000000000 |
| `staging_uri` | str | not set |
| `cache_temp_tables` | bool | false |
| `temp_table_expiration_hours` | int, at least 1 | 24 |
| `use_bqstorage` | bool | false |
| `densify_step` | float, above 0 | 0.1 |
| `location` | str | not set |

### `[output]`

Required.

| key | type | default |
| --- | --- | --- |
| `dir` | str, not empty | required |
| `format` | "parquet" or "csv" or "both" | "parquet" |
| `confirm_cost` | bool | true |

### `[validation]`

Can be left out.

| key | type | default |
| --- | --- | --- |
| `sum_tolerance` | float, above 0 | 1e-06 |
| `cross_backend_tolerance` | float, above 0 | 0.001 |
| `mass_balance_tolerance` | float, above 0 | 1e-06 |

## The pipeline config

`cilreg pipeline <config.toml>` reads it.

The config of a pipeline run. Without ``weights`` the run is a
passthrough, with statistics per source region and no spatial
aggregation.

### `[weights]`

Can be left out.

Where the weights artifact lives and which weight to apply.

| key | type | default |
| --- | --- | --- |
| `dir` | str, not empty | required |
| `stem` | str | "weights" |
| `weight` | str, not empty | required |

### `[tree]`

Required.

Declared layout of the Monte Carlo tree.

``levels`` names the directory levels in path order, and a level
that is not a sample dimension becomes a column of the statistics.
``expect`` gives the values of each level, and the leaves are every
combination of them, each at ``root/<v1>/.../<vN>/filename``.
``exclude`` takes cells out of those combinations. A cell has one
value for each level, and ``"*"`` stands for every value of its
level. ``subtract_filename`` names a sibling file whose values are
taken off the leaf's, row for row. A declared leaf that is not on
disk fails the run before any work.

| key | type | default |
| --- | --- | --- |
| `root` | str, not empty | required |
| `levels` | list[str], not empty | required |
| `filename` | str, not empty | required |
| `subtract_filename` | str | not set |
| `expect` | dict[str, list[str]] | required |
| `exclude` | list[list[str]] | [] |

### `[data]`

Required.

What a leaf contains and how to aggregate it.

``format`` is declared and never inferred from the file extension.
For ``netcdf`` leaves, which need the ``[netcdf]`` extra,
``value_col`` names the data variable and ``region_dim`` the file's
region dimension. The reader turns it into the source key column of
the weights artifact, or into ``region_col`` in a run without
weights. Every other dimension passes through as a parquet column
would.

| key | type | default |
| --- | --- | --- |
| `format` | "parquet" or "csv" or "netcdf" | "parquet" |
| `value_col` | str | "value" |
| `denominator_col` | str | not set |
| `kind` | "extensive" or "intensive" or "ratio" | required |
| `version` | str, not empty | required |
| `group_col` | str | not set |
| `region_dim` | str | not set |
| `region_labels` | str | not set |
| `region_col` | str | not set |

### `[stats]`

Required.

Statistics stage.

Under stage ``all`` with "window_mean" each leaf is reduced to its
window means as it is aggregated, so other ``windows`` take a new
run, or a run of the reduce stage over leaves that were kept.

| key | type | default |
| --- | --- | --- |
| `sample_dims` | list[str], not empty | required |
| `time_col` | str, not empty | required |
| `windows` | list[tuple[float, float]] | [] |
| `quantiles` | list[float] | [] |
| `include_mean` | bool | true |
| `include_n_members` | bool | false |
| `quantile_method` | "linear" or "lower_step" or "inverted_cdf" | required |
| `summation` | "fast" or "exact" | required |
| `time_reduction` | "window_mean" or "pooled" or "per_year" | required |
| `years` | tuple[int, int] | not set |
| `model_weights` | str | not set |
| `model_weight_level` | str | not set |
| `model_weight_names` | dict[str, str] | {} |
| `model_dims` | list[str] | not set |

### `[stats.warming_levels]`

Can be left out.

Warming levels in place of calendar windows.

``gmst`` is the temperature table and ``member`` its columns that
name a series, each of them also a tree level. Nothing that decides
a level has a default. ``reference_period`` and ``offset`` go
together, since the offset is the warming up to that period.
``smoothing`` says in words how the series in the table was
smoothed, and "unknown" is an answer. ``join`` names the members
that join, "declared" for every member the tree declares, or a table
of member columns to the values that join. A member of the tree
that does not join is pooled at no level, and the record lists it.

| key | type | default |
| --- | --- | --- |
| `gmst` | str, not empty | required |
| `member` | list[str], not empty | required |
| `time_col` | str | "year" |
| `value_col` | str | "temp" |
| `sep` | str | "," |
| `reference_period` | tuple[int, int] | required |
| `offset` | float | required |
| `levels` | list[float], not empty | required |
| `half_width` | float, above 0 | required |
| `smoothing` | str, not empty | required |
| `join` | "declared" or dict[str, list[str]] | required |

### `[[stats.warming_levels.periods]]`

A list of tables, each with these keys.

A calendar period of a warming level product. ``gmst_span`` is
the years the temperature is averaged over to pick a member's
level, and ``impact_span`` the years of the impacts. Both include
their first and last year.

| key | type | default |
| --- | --- | --- |
| `gmst_span` | tuple[int, int] | required |
| `impact_span` | tuple[int, int] | required |

### `[output]`

Required.

Where the results are written. ``stem`` names the statistics and
manifest files, so several runs can share one directory.
``write_reduced`` also writes the pooled window means, one row a
target, window and sample member, which `regroup` works on.
``write_leaves`` keeps every aggregated leaf with its years under
``leaves/``. It is off unless asked for, because the leaves take far
more disk than the statistics. ``leaves_stem`` names the leaves
manifest, and is ``stem`` unless set, so a product with other
windows can take its own stem and reduce the leaves of another.

| key | type | default |
| --- | --- | --- |
| `dir` | str, not empty | required |
| `write_leaves` | bool | false |
| `leaves_stem` | str | not set |
| `stem` | str | "statistics" |
| `write_reduced` | bool | false |

### `[policies]`

Can be left out.

What to do with source units that cannot take part in applying the
weights, the ``[policies]`` of a pipeline config and the ``policies``
of `apply_weights`.

Each key is "error" or "skip". "error" raises naming the units.
"skip" leaves them out with a warning, and the result lists them in
its ``coverage``. ``on_unmatched`` is for units in the data that are
not in the weight file, whose values cannot reach any target.
``on_zero_weight`` is for units in the weight file whose weights are
all zero or NaN. ``on_absent_from_data`` is for units in the weight
file that the data does not have, whose targets lose what the
weights gave them.

| key | type | default |
| --- | --- | --- |
| `on_unmatched` | "error" or "skip" | "error" |
| `on_zero_weight` | "error" or "skip" | "error" |
| `on_absent_from_data` | "error" or "skip" | "error" |

### `[run]`

Can be left out.

``n_workers`` processes aggregate the leaves in waves, each wave on
new workers, since a worker's resident memory grows with the leaves
it has taken. A worker that dies stops the run, naming the leaves in
flight and saying how the workers ended.

| key | type | default |
| --- | --- | --- |
| `n_workers` | int, at least 1 | 1 |

### `[reduce]`

Can be left out.

| key | type | default |
| --- | --- | --- |
| `kind` | "pandas" | "pandas" |

### `[reduce.pandas]`

Can be left out.

Options of the pandas backend. ``memory_gb`` is the memory all
its workers may hold together, which the blocks are sized from, and
``targets_per_block`` declares their size instead. One of the two is
needed and neither has a default. ``n_workers`` defaults to
``run.n_workers``.

| key | type | default |
| --- | --- | --- |
| `memory_gb` | float, above 0 | not set |
| `targets_per_block` | int, at least 1 | not set |
| `n_workers` | int, at least 1 | not set |
