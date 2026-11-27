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

Fetch a published weight artifact and return it ready to apply.

``name`` is normally a registry name; with ``record`` given (a Zenodo
record id or DOI) the registry is bypassed and ``name`` only labels
the cache entry. The result is a `WeightsArtifact`, directly
consumable by `apply_weights`. ``refresh`` re-downloads even when a
verified cached copy exists.

### `list_cached`

```python
list_cached(cache_dir: str | Path | None = None) -> list[dict[str, Any]]
```

Cached artifacts: name, record, path, and size in bytes.

### `clear_cache`

```python
clear_cache(
    cache_dir: str | Path | None = None,
    *,
    name: str | None = None,
) -> int
```

Remove cached artifacts; all of them, or just ``name``. Returns count.

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

``partial_coverage`` is the generation-time coverage accounting for
polygon-mode artifacts (threshold, count, per-unit shortfalls,
whether the target set was a subset). None means the artifact
carries no accounting: hand-built artifacts and pre-accounting
manifests. Generated polygon artifacts always carry it, count zero
included, so a real artifact cannot lose the marking by accident.

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

Load a written artifact (parquet + manifest sidecar).

Reconstructs the schema from the manifest's recorded column roles.
Manifests written before those fields existed cannot be applied;
regenerate the weights or construct the artifact in memory via
`from_result`.

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

Build an artifact from a weight table produced elsewhere.

``table`` is a frame, or a path to a CSV or parquet file, with
one row per (region, source unit) and one normalized weight
column per weight. No manifest is needed: what the manifest
would record is passed here.

``weights`` maps each weight name to its column, for example
``{"pop": "popwt"}``; columns are renamed to the ``{name}wt``
form `apply_weights` reads. ``source`` becomes the artifact's
``source_version``, which `apply_weights` compares against the
data's ``data_version``.

When ``source`` names a grid (``grid-0.25deg-center-[-180,180)``,
the form `GridSpec.source_version` writes), source units are
grid cells. Without ``cell_ix`` / ``cell_iy`` they are derived
from a longitude and a latitude column (spelled
``cell_lon``/``cell_lat``, ``lon``/``lat``,
``longitude``/``latitude`` or ``pix_cent_x``/``pix_cent_y``).
Longitudes are wrapped into the grid's convention first, so the
180.125 centroid CIL files carry becomes -179.875, and every
point must sit on a cell centroid.

Given ``cell_ix`` / ``cell_iy`` are used as they are, after a
bounds check. When a longitude or latitude column is also
present, each row's index must map back to that centroid on the
named grid, which catches indices built on a different grid of
the same shape (a [0,360) table read as [-180,180), shifted by
half the columns, or latitudes counted from the north). When
neither column is present the indices are taken on trust: the
only defense is that ``source`` names the grid they were built
on exactly, since `apply_weights` compares it to the data's
version as a literal string and nothing checks the geography.

Any other ``source`` is a polygon source version and
``source_keys`` names the id columns of its units.

With ``check`` the table must have no duplicated (region,
source unit) rows and every weight must sum to 1 within each
normalization group (`check_sum_to_one`); both failures raise
naming the offending groups.

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
column is a passthrough dimension: aggregation happens independently
per combination of dimension values, and the output keeps those
columns. Columns that are labels of the source unit rather than
dimensions (a country tag, a display name) behave as dimensions too;
drop them beforehand if they should not appear in the output. Rows
duplicated on (source keys + dimensions) would double count and raise.

``data_version`` states the geometry version the data was built
against and is compared to the artifact's recorded source version.
``group_col`` optionally names a data column to run the mass balance
per group (typically the country) as well as globally.
``restrict_to_sources`` limits the weight frame to an explicit source
universe for subset runs; the coverage checks stay fully strict within
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

What to do with source units that cannot flow through an application run.

Mirrors the null-geometry and unknown-id pattern on `RegionsConfig`:
"error" (default) aborts and names the offending units; "skip" excludes
them, records them in the manifest under `source_coverage`, and logs a
warning. Nothing is ever dropped without one or the other.

The three cases, detected by `validate.check_source_coverage`:

    on_unmatched       : units present in the data being aggregated but
                         absent from the weight file. Their values
                         cannot reach any target.
    on_zero_weight     : units present in the weight file whose weights
                         are all zero or NaN. Rows exist but carry no
                         mass to any target.
    on_absent_from_data: units present in the weight file but missing
                         from the data. Their targets silently lose the
                         contribution the weights promised.

Consumed by the application layer and the polygon weight builder; not
referenced from the run `Config` yet.

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

Read a NetCDF on a regular lat-lon grid into a cell table.

Returns ``(cells, grid)``. ``cells`` has one row per cell and
slice, with ``cell_ix`` and ``cell_iy`` on the grid, one column per
variable, and any other dimension (time, say) as a column, as
`read_netcdf_leaf` does. ``grid`` is the identity string, for
example ``grid-0.25deg-center-[-180,180)``; pass it to
`apply_weights` as ``data_version``.

``lat`` and ``lon`` name the file's 1-D coordinate variables, each
the coordinate of a dimension of the same name; every variable must
have both. Coordinate names are never guessed.

The grid is ``grid`` if given, as the identity string or a
`GridSpec`, and otherwise derived from the coordinates
(`infer_grid_version`). With ``grid``, longitudes are wrapped into
its convention, so a [0,360) file can be put on a [-180,180) grid.
Either way each coordinate is mapped to its cell index and back to a
centroid, and the read raises if any value is not a cell centre on
that grid. No data is moved or reordered: indices come from each
row's own coordinates, so the order of the axes does not matter.
Nothing is regridded or snapped.

A subset whose longitudes all lie in [0,180], such as a cut of
Europe, Asia or Africa, always needs ``grid``. The same coordinates
are valid on both longitude conventions, so nothing in the file can
say which one your weights were built on. The error for this case
is not a bug; it prints the two calls that would work. A file with a
coordinate beyond 180 or below 0 is not affected.

Limit: cells must be square. The grid identity holds one
resolution, so a file whose latitude and longitude steps differ
raises instead of being read. The identity's spelling is fixed by
the weight files already on Zenodo. Allowing two steps would change
`GridSpec`, `GridConfig`, the manifest, `validate` and the BigQuery
backend, and would still not read Gaussian grids or grids with
latitude points on the poles, whose latitude step does not divide
180 degrees.

Cells masked (NaN) in every slice are left out, as
`DailyGrid.to_cells` does, so `apply_weights` reports a weighted
cell that has no data instead of averaging around it. A cell masked
in some slices and not others raises. Two coordinates that land on
the same cell (a longitude of both -180 and 180) raise.

### `OutputSchema`

```python
class OutputSchema:
    id_fields: tuple[str, ...]
    weight_names: tuple[str, ...]
    source_units: SourceUnits = GRID_CELLS
    normalization: Normalization = 'per_destination'
```

Resolved column layout for one run: id fields + source units + weights.

Defaults reproduce the grid layout exactly: grid-cell source units and
per_destination normalization.

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

``key_columns`` identify one source unit (join and duplicate-check
keys). ``meta_columns`` are carried alongside but are not part of the
identity (the grid case carries cell centroids). ``dtype_items`` holds
(column, dtype) pairs for every column, as a tuple so the dataclass
stays frozen and hashable.

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
member's time rows. Weights must be finite and non-negative, and a
zero weight drops the member. Weights are not renormalized within
a model, so in an
unbalanced sample a model's total weight grows with how often it
appears. Leaving ``weight_col`` unset with a ``gcm`` sample
dimension warns with `UnweightedModelWeightsWarning`.

Returns a long frame: identity columns, 'statistic', ``value_col``.
Labels are 'mean' and 'qNN' in percent ('q05', 'q95', 'q2.5').

``include_n_members`` adds two integer columns after ``value_col``.
'n_members' is the samples behind a statistic, the members with a
window mean that is not NaN and, when weighted, a weight above zero,
so a batch of a model counts once a batch. 'n_models' is the
climate models among them, the distinct values of ``model_dims``,
which it then has to name, such as ``["gcm"]``. Over fifteen batches
of four models they are 60 and 4. ``data`` must not already have
columns of those names.

``quantile_method`` is "linear", "lower_step" or "inverted_cdf",
and has no default. "linear" interpolates between members and is
refused with weights. "lower_step" is the left step inverse of the
weighted distribution, as the historical extraction tool computed
it. It returns the member one below the one whose cumulative weight
share reaches q, with every member weighted equally when
``weight_col`` is unset. It differs from "linear" even when every
weight is equal, except at 0.5, which is then the plain median, and
a quantile below the first step is ``-inf``. "inverted_cdf"
returns, weighted or not, the smallest member value whose
cumulative weight share reaches q, so never ``-inf``.

``summation`` is "fast" or "exact", and has no default. "fast" is
the vectorised sum, whose last bits depend on the order of the
rows. "exact" takes the means from exactly rounded sums, whatever
the order. An infinite member value raises, and so does a statistic
that comes out infinite other than the step's ``-inf``.

The statistics are float64 whatever the dtype of ``value_col``. A
column of another dtype is widened before the window mean, and a
nullable one comes back with NaN where it had no value.

Raises if ``data`` already has a 'statistic' column or
quantile-named columns. Statistics of per-unit quantiles are wrong;
see the module docstring.

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

The first half of `summarize_samples`, split out so a pipeline can
reduce each leaf before pooling. It never crosses sample members, so
reducing early and pooling later gives the same statistics.

``time_col`` is replaced by ``window_col``, labelled like "2040-2059",
or by the year alone for a one-year window. Data that already has
``window_col`` is refused: its time dimension is gone, so to change
windows rerun from the unreduced data. Rows repeated on every column
but ``value_col`` are refused too, in any year, including years no
window covers.

``summation`` has no default. "exact" takes each mean from an
exactly rounded sum, and "fast" is the pandas mean. An infinite
value in a window raises.

The mean is float64 whatever the dtype of ``value_col``. A float32
column is widened before the mean and not after, because a float32
mean stored as float64 still holds seven digits. A nullable column
comes back with NaN where it had no value.

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
scalar. ``q = 0`` and ``q = 1`` return the ends of the support:
infinite when any weighted component has positive variance,
otherwise the smallest or largest point mass.

### `read_model_weights`

```python
read_model_weights(path: str | Path) -> ModelWeights
```

Read a weights table. The file is read once, and the hash is of
the bytes the rows were parsed from. Raises if two rows match the
same member.

### `fetch_model_weights`

```python
fetch_model_weights(
    rcp: str,
    *,
    record: str = MODEL_WEIGHTS_RECORD,
    base_url: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
)
```

The published climate model weights of ``rcp``, as
`read_model_weights` reads them, from the file
``<rcp>_SMME_weights.tsv`` of their Zenodo record. The download is
checked against the md5 the record declares and cached with it, and
a cached copy that no longer matches raises.

### `ModelWeights`

```python
class ModelWeights:
    path: str
    sha256: str
    names: tuple[str, ...]
    keys: tuple[str, ...]
    weights: tuple[float, ...]
```

The rows of a weights table, in file order, with the SHA256 of
the bytes they were parsed from. ``names`` are as written in the
file and ``keys`` are what a member is matched on.

#### `ModelWeights.missing`

```python
ModelWeights.missing(members: Iterable[str]) -> list[str]
```

The members that have no row in the table, sorted.

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
``members``. Raises if a member has no row in the table.

#### `ModelWeights.record`

```python
ModelWeights.record(members: Iterable[str]) -> dict
```

The manifest entry of a run, given one member value per leaf
that was pooled. ``members`` holds each model attached, with its
table row, its weight and its number of leaves.
``unmatched_rows`` holds the rows no member matched with their
weights, so the two together are the whole table. Raises if a
member has no row.

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

"extensive" sums the values, and needs no ``basis``. "intensive"
takes the mean weighted by each target's basis, the frame
`target_basis` gives, over the targets with a value. A group with no
value is NaN. ``summation`` is "fast" or "exact", as for the
statistics. Raises if ``by`` is not among ``id_fields``, or a row's
target has no basis.

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
    surrogates: str,
    members: Optional[Iterable[Sequence[str]]] = None,
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
here. ``surrogates`` is "exclude" or "include" and has no default.
Excluded, a surrogate model is at no level, since its series was
built to span the distribution of warming under weights that a
warming level does not use. ``members`` are the members assessed,
every member of the table if it is left out. Raises if one of them
has no row, or lacks a value for a year of the reference period or
of a ``gmst_span``.

### `Period`

```python
class Period:
    gmst_span: tuple[int, int]
    impact_span: tuple[int, int]
```

A calendar period. ``gmst_span`` is the years the temperature is
averaged over to pick a member's level, and ``impact_span`` the
years of the impacts that go under that level. Both include their
first and last year. They are given apart because a product may
average its impacts over other years than its temperature.

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
    surrogates: str = 'include'
    excluded: tuple[tuple[str, ...], ...] = ()
```

The level of each assessed member of a temperature table in
every period, and what decided it. ``means`` and ``levels`` hold
one value a period for each assessed member, in the order of
``periods``. A level is None where the member's mean lies in no
band. ``excluded`` are the surrogates left out, which are at no
level.

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
``periods``, none for an excluded surrogate. Raises if the member
was not assessed.

#### `WarmingLevels.record`

```python
WarmingLevels.record(members: Optional[Iterable[Sequence[str]]] = None) -> dict
```

What a manifest says of a run, as plain values. ``members``
are those the run used, and every member assessed or excluded if
it is left out. The excluded surrogates are listed under
``excluded_surrogates`` and the rest of the table under
``unused_members``. Raises if a member was not assessed.

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

Read one NetCDF leaf into a long frame.

``variables`` are the data variables to carry (one for extensive or
intensive application; numerator and denominator for ratio).
``region_dim`` is the file's name for the region dimension, declared
in config rather than assumed; it is renamed to ``region_col``, the
weights artifact's source key column. Everything else flattens to
ordinary columns.

Some trees carry no coordinate on the region dimension and store
the region ids in a separate variable instead (the mortality trees
keep them in ``regions``). Without a coordinate the region column
would come out as positional integers, so ``region_labels`` names
that variable and its values become the region coordinate before
flattening. Declared like everything else: the reader does not
guess which variable holds the ids.

``kind`` optionally names the aggregation kind the caller intends,
enabling the units backstop: a variable whose ``units`` attribute
contains a percent sign is a ratio, and summing a ratio across
regions is meaningless, so reading it under ``kind="extensive"``
raises. This catches exactly one failure (percent-labeled ratios
declared extensive, like a "% GDP" levels file); it cannot catch
ratios with percent-free unit strings, missing units attributes, or
the same mistake arriving through parquet and csv leaves, which
carry no units metadata. The kind declaration itself remains the
caller's responsibility.

In the mortality and labor Monte Carlo trees, the effect of
climate change is ``rebased`` from a ``...-combined.nc4`` file
minus ``rebased`` from its ``-histclim`` sibling; the value on its
own is not it. No adaptation is the exception and stands alone.
The Climate Impact Lab memo "The art of rebasing and histclim" has
the reasoning; the Mexico example (examples/rates) shows the
subtraction.

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

Read one year of a delta-method leaf.

Uses netCDF4 directly: the covariance matrix is stored with the
coefficient dimension repeated, which xarray does not accept. The
files are chunked one year per chunk, so a single-year read moves
one chunk, not the whole file.

### `window_gradient`

```python
window_gradient(
    gradients: np.ndarray,
    years,
    window: tuple[float, float],
) -> np.ndarray
```

Mean of stacked yearly gradients over an inclusive year window.

``gradients`` has year on the first axis, in the order of
``years``; the other axes pass through unchanged, so the result
still carries the coefficient axis and the quadratic form can be
taken on it afterwards. Stacked leaves (year, coefficient, region)
and stacked aggregates (year, target, coefficient) both work, and
since both steps are linear, windowing before or after
`aggregate_gradient` gives the same answer.

The window bounds match `window_means`: both ends included, and a
window that selects no year raises. NaN entries drop out of the
mean the way NaN rows do there.

### `aggregate_gradient`

```python
aggregate_gradient(
    weights: WeightsArtifact,
    regions: list[str],
    gradient: np.ndarray,
    *,
    weight: str = 'pop',
    scale: np.ndarray | None = None,
    normalize: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]
```

Aggregate gradient vectors to the weight file's targets.

``gradient`` is coefficient by region, in the order of ``regions``.
Every region must appear in the weight file; weight rows for other
regions are ignored, which mirrors ``restrict_to_sources``.

For a total (an extensive value), leave ``scale`` and ``normalize``
alone: each region's gradient is split by the weight shares and
summed. For a rate, pass the per-region sizes (population) as
``scale`` and set ``normalize=True``: each target becomes the
size-weighted mean of its regions' gradients.

Returns the target key frame and the aggregated gradient, target
by coefficient. The aggregation is linear, so it commutes with
differencing histclim and with evaluating means: aggregating a
difference of gradients equals differencing aggregated ones.

### `delta_method`

```python
delta_method(
    aggregated: np.ndarray,
    vcv: np.ndarray,
    coefficients: np.ndarray | None = None,
) -> pd.DataFrame
```

Mean and standard deviation per target from aggregated gradients.

Variance is the quadratic form G V G' per target. The mean is
G times the coefficient vector and is exact when the response is
linear in the coefficients, which is what makes the gradients
constant in them; omit ``coefficients`` to get only the spread.

## Generating weights

### `generate_weights`

```python
generate_weights(config: Config | str | Path) -> WeightsResult
```

Generate a weights file and its manifest from a config.

``config`` is a `Config` or the path to a TOML file. The weights
parquet and the manifest are written to ``config.output.dir``, as
`cilreg run` does, and the result is returned. Geometries are read
from the paths in the config; to use polygons held in memory, write
them to a GeoParquet file first so the manifest can record its hash.

Raises `FileNotFoundError` naming any input that is missing, and
`SumToOneError` when the weights fail their sum-to-one check, in
which case nothing is written.

### `load_config`

```python
load_config(path: str | Path) -> Config
```

Load and validate a cil_regionalization config from a TOML file.

Relative paths in `regions.path`, `source.path`, `weights[].raster`,
and `output.dir` are resolved against the config file's directory so the same config
works regardless of the caller's working directory. Absolute paths,
`gs://` URIs, and BigQuery `table` references pass through unchanged.

Validation errors are raised as pydantic ValidationError; their messages
name the offending key so the user can fix the TOML without reading source.

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

Combined output: frame + schema + manifest + validation report.

### `SumToOneError`

```python
class SumToOneError(ValueError)

SumToOneError(report: SumToOneReport, result: WeightsResult)
```

The generated weights failed the sum-to-one check.

``report`` is the full `SumToOneReport`, failures included.
``result`` is the computed `WeightsResult`; nothing was written.

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

Polygon source units: the geometry the weighted data lives on.

Declaring ``[source]`` instead of ``[grid]`` switches a run to polygon
mode: rows pair one region from ``[regions]`` (the side the weights
are built for, e.g. administrative units) with one source polygon from
here (e.g. impact regions keyed by hierid). Any shapefile or
GeoParquet with id columns works; nothing here is specific to one
geometry family.

``version`` is required: it is the version label recorded in the
manifest so a weight file states which source set produced it.

``on_zero_overlap`` handles source units whose intersection with every
region has zero geodesic area: "error" (default) aborts naming them;
"skip" excludes them and records them in the manifest under
``zero_overlap_source_units``. Runs against a regional subset of
targets should expect zero-overlap sources and set "skip" explicitly.

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
| `slurm_queue` | str | not set |
| `slurm_walltime` | str | not set |
| `min_segment_area_m2` | float, at least 0 | 1.0 |

### `[backend.bigquery]`

Can be left out.

| key | type | default |
| --- | --- | --- |
| `project` | str | "compute-impactlab" |
| `temp_dataset` | str | "temp_workspace" |
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

``weights`` is optional: without it the pipeline runs in
passthrough, producing statistics per source region with no spatial
aggregation (see the module docstring).

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

``levels`` names the directory levels in path order; each level
becomes a column on the aggregated output. ``expect`` declares the
values per level, and the expected leaf set is their full cross
product: a leaf path is ``root/<v1>/.../<vN>/filename``. Anything
declared but absent on disk fails the run before any processing.
``exclude`` takes cells out of that product. A cell has one value
for each level, and ``"*"`` stands for every value of its level.

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

``format`` is declared, never inferred from the file extension:
inference invites a mismatch between what a user thinks is being
read and what is read. For ``netcdf`` leaves (the ``[netcdf]``
optional extra), ``value_col`` names the data variable and
``region_dim`` names the file's region dimension, which the reader
maps to the weights artifact's source key column. Every other
dimension passes through exactly as a parquet column would.

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

``windows`` is final before any leaf is processed: each leaf is
reduced to window means as it is aggregated, and the reduced frames
carry no time dimension. Changing the windows afterwards is a rerun,
not a reprocessing of reduced data, and the reduction step refuses
already reduced input by construction. If the leaves were kept, the
reduce stage takes other windows from them.

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
| `model_dims` | list[str] | not set |

### `[stats.warming_levels]`

Can be left out.

Warming levels in place of calendar windows.

``gmst`` is the temperature table and ``member`` its columns that
name a series, each of them also a tree level. Nothing that decides
a level has a default. ``reference_period`` and ``offset`` go
together, since the offset is the warming up to that period.
``smoothing`` says in words how the series in the table was
smoothed, and "unknown" is an answer. ``surrogates`` is "exclude"
or "include", and a surrogate model excluded is pooled at no level.
Only the members the tree declares are assessed.

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
| `surrogates` | "exclude" or "include" | required |

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

Where results land. ``stem`` names the statistics and manifest
files, so several runs can share one directory. ``write_reduced``
keeps the pooled window-mean frame (one row per target, window,
and sample member), which downstream regrouping needs; per-draw
quantities cannot be recovered from the statistics file.
``write_leaves`` keeps every aggregated leaf with its years, under
``leaves/``. It is off unless asked for, because the leaves take
far more disk than the statistics. ``leaves_stem`` names the leaves
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

What to do with source units that cannot flow through an application run.

Mirrors the null-geometry and unknown-id pattern on `RegionsConfig`:
"error" (default) aborts and names the offending units; "skip" excludes
them, records them in the manifest under `source_coverage`, and logs a
warning. Nothing is ever dropped without one or the other.

The three cases, detected by `validate.check_source_coverage`:

    on_unmatched       : units present in the data being aggregated but
                         absent from the weight file. Their values
                         cannot reach any target.
    on_zero_weight     : units present in the weight file whose weights
                         are all zero or NaN. Rows exist but carry no
                         mass to any target.
    on_absent_from_data: units present in the weight file but missing
                         from the data. Their targets silently lose the
                         contribution the weights promised.

Consumed by the application layer and the polygon weight builder; not
referenced from the run `Config` yet.

| key | type | default |
| --- | --- | --- |
| `on_unmatched` | "error" or "skip" | "error" |
| `on_zero_weight` | "error" or "skip" | "error" |
| `on_absent_from_data` | "error" or "skip" | "error" |

### `[run]`

Can be left out.

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
its workers may hold together and has no default. ``n_workers``
defaults to ``run.n_workers``.

| key | type | default |
| --- | --- | --- |
| `memory_gb` | float, above 0 | not set |
| `n_workers` | int, at least 1 | not set |
