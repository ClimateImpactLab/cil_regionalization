"""Apply a weights artifact to source-level data.

The repo's other modules build weights; this one consumes them. The core
operation joins a weights frame to a data frame on the source-unit key
columns recorded in the artifact, applies the declared operation, and
groups to the target id fields. Every other column of the data is treated
as an opaque dimension and carried through: nothing here knows or assumes
dimension names like gcm, rcp, ssp, model, or year. Only the region
dimension, the source key columns, is special.

Variable kind is declared, never inferred
-----------------------------------------
    extensive : target value = sum of weight * value. Requires per_source
        weights (each source unit's weights sum to 1 across its targets),
        so the operation allocates the source total and conserves it.
    intensive : target value = sum(weight * value) / sum(weight) over the
        source units with data. Requires per_destination weights. The
        denominator uses only the units present with non-null values, so
        partial coverage renormalizes instead of deflating the mean.
    ratio : numerator and denominator are each aggregated extensively,
        then divided. Requires per_source weights. This is the correct
        order for shares (damages over GDP); averaging per-unit ratios is
        a different and generally wrong number.

The declared kind is checked against the artifact's recorded
normalization direction with `schema.require_normalization`; a mismatch
raises with an explanation rather than producing plausible wrong numbers.

Qualified keys only
-------------------
Joins run on the source key columns the weight file records, and targets
are identified by the id fields it records. There is no parameter for
joining on anything else, display names included, so the name-collision
failure mode (the same admin name in two countries merging into one unit)
cannot be expressed through this interface.

Version agreement
-----------------
The artifact records the version of the geometry its source units come
from (``source_version``). The caller must state the version the data was
built against; a mismatch raises, and a missing version on either side is
itself an error rather than an assumed compatibility.

Validation
----------
Source coverage runs on every application: unmatched, zero-weight, and
absent-from-data source units go through the configured
`SourceUnitPolicies` (error by default) and are reported on the result.
Extensive and ratio aggregations then run the mass balance check over the
data that was actually applied; failure raises. The mass balance
therefore proves the aggregation machinery conserved what it processed,
while the coverage report accounts for what was excluded and why. For a
subset run (for example one country's targets), restrict the source
universe explicitly with ``restrict_to_sources`` instead of relaxing the
coverage policies.

This is a library API. There is no CLI verb for application yet; see the
increment notes for the reasoning.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from cil_regionalization.backends.base import WeightsResult
from cil_regionalization.config import GridConfig, SourceUnitPolicies
from cil_regionalization.grid import GRID_SOURCE_PATTERN as _GRID_SOURCE
from cil_regionalization.grid import GridSpec
from cil_regionalization.schema import (
    GRID_CELLS,
    OutputSchema,
    SourceUnits,
    require_normalization,
    weight_column,
)
from cil_regionalization.validate import (
    MassBalanceReport,
    SourceCoverageReport,
    check_mass_balance,
    check_source_coverage,
    check_sum_to_one,
    enforce_source_policies,
)

# column spellings accepted for cell centroids in external tables,
# including the pix_cent_x / pix_cent_y of CIL segment weight files
_LON_COLUMNS = ("cell_lon", "lon", "longitude", "pix_cent_x")
_LAT_COLUMNS = ("cell_lat", "lat", "latitude", "pix_cent_y")


VariableKind = Literal["extensive", "intensive", "ratio"]

_KIND_TO_DIRECTION: dict[str, str] = {
    "extensive": "per_source",
    "ratio": "per_source",
    "intensive": "per_destination",
}


@dataclass(frozen=True)
class WeightsArtifact:
    """A weights frame plus the metadata needed to apply it safely.

    ``partial_coverage`` is the generation-time coverage accounting for
    polygon-mode artifacts (threshold, count, per-unit shortfalls,
    whether the target set was a subset). None means the artifact
    carries no accounting: hand-built artifacts and pre-accounting
    manifests. Generated polygon artifacts always carry it, count zero
    included, so a real artifact cannot lose the marking by accident.
    """

    frame: pd.DataFrame
    schema: OutputSchema
    regions_version: str | None
    source_version: str | None
    partial_coverage: dict | None = None

    @property
    def normalization(self) -> str:
        return self.schema.normalization

    @classmethod
    def from_result(cls, result: WeightsResult) -> "WeightsArtifact":
        return cls(
            frame=result.frame,
            schema=result.schema,
            regions_version=result.manifest.regions_version,
            source_version=result.manifest.source_version,
            partial_coverage=result.manifest.partial_coverage,
        )

    @classmethod
    def load(cls, output_dir: str | Path, stem: str = "weights") -> "WeightsArtifact":
        """Load a written artifact (parquet + manifest sidecar).

        Reconstructs the schema from the manifest's recorded column roles.
        Manifests written before those fields existed cannot be applied;
        regenerate the weights or construct the artifact in memory via
        `from_result`.
        """
        base = str(output_dir).rstrip("/")
        if base.startswith("gs://"):
            import gcsfs

            fs = gcsfs.GCSFileSystem()
            with fs.open(f"{base}/{stem}.parquet", "rb") as f:
                frame = pd.read_parquet(f)
            with fs.open(f"{base}/{stem}.manifest.json", "r") as f:
                manifest = json.load(f)
        else:
            frame = pd.read_parquet(Path(base) / f"{stem}.parquet")
            manifest = json.loads(
                (Path(base) / f"{stem}.manifest.json").read_text()
            )

        missing = [
            k
            for k in ("id_fields", "source_key_columns", "weight_names")
            if manifest.get(k) is None
        ]
        if missing:
            raise ValueError(
                f"weights manifest at {base} does not record {missing}; it "
                f"was written by an older version of this package. "
                f"Regenerate the weights with the current version."
            )
        if manifest.get("source_mode") == "grid":
            source_units = GRID_CELLS
        else:
            source_units = SourceUnits.from_string_ids(
                manifest["source_key_columns"]
            )
        schema = OutputSchema(
            id_fields=tuple(manifest["id_fields"]),
            weight_names=tuple(manifest["weight_names"]),
            source_units=source_units,
            normalization=manifest.get("normalization", "per_destination"),
        )
        return cls(
            frame=frame,
            schema=schema,
            regions_version=manifest.get("regions_version"),
            source_version=manifest.get("source_version"),
            partial_coverage=manifest.get("partial_coverage"),
        )

    @classmethod
    def from_table(
        cls,
        table: pd.DataFrame | str | os.PathLike,
        *,
        id_fields: Sequence[str],
        weights: Mapping[str, str],
        source: str,
        source_keys: Sequence[str] | None = None,
        normalization: str = "per_destination",
        regions_version: str | None = None,
        check: bool = True,
        tolerance: float = 1e-6,
    ) -> "WeightsArtifact":
        """Build an artifact from a weight table produced elsewhere.

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
        """
        if normalization not in ("per_destination", "per_source"):
            raise ValueError(
                f"normalization must be 'per_destination' or 'per_source'; "
                f"got {normalization!r}"
            )
        id_fields = list(id_fields)
        if not id_fields:
            raise ValueError("from_table: id_fields must name at least one column")
        if not weights:
            raise ValueError("from_table: weights must map at least one name to a column")

        grid_match = _GRID_SOURCE.match(source)
        if grid_match is None and source.startswith("grid-"):
            raise ValueError(
                f"source {source!r} looks like a grid but is not of the form "
                f"'grid-<resolution>deg-<center|edge>-<[-180,180)|[0,360)>'"
            )
        if grid_match is None and not source_keys:
            raise ValueError(
                f"source {source!r} is not a grid, so source_keys must name the "
                f"id columns of its source units"
            )
        if grid_match is not None and source_keys:
            raise ValueError(
                "source_keys is only for polygon sources; grid sources are "
                "keyed by cell_ix and cell_iy"
            )

        if isinstance(table, pd.DataFrame):
            frame = table.copy()
        else:
            string_cols = id_fields + list(source_keys or [])
            frame = _read_table(table, string_cols)

        missing = [c for c in id_fields if c not in frame.columns]
        if missing:
            raise ValueError(
                f"from_table: id_fields {missing} are not columns of the table "
                f"(columns: {list(frame.columns)})"
            )

        if grid_match is not None:
            spec = GridSpec.from_config(GridConfig(
                mode="generate",
                resolution=float(grid_match["res"]),
                offset=grid_match["offset"],
                lon_convention=grid_match["conv"],
            ))
            if spec.source_version != source:
                raise ValueError(
                    f"source {source!r} names the grid that writes its identity "
                    f"as {spec.source_version!r}; use that spelling, since "
                    f"apply_weights compares versions as exact strings"
                )
            frame = _attach_grid_cells(frame, spec)
            source_units = GRID_CELLS
        else:
            source_units = SourceUnits.from_string_ids(source_keys)
            absent = [c for c in source_units.key_columns if c not in frame.columns]
            if absent:
                raise ValueError(
                    f"from_table: source_keys {absent} are not columns of the "
                    f"table (columns: {list(frame.columns)})"
                )
        source_cols = list(source_units.key_columns)
        overlap = sorted(set(id_fields) & set(source_cols))
        if overlap:
            raise ValueError(
                f"from_table: {overlap} cannot be both an id field and a "
                f"source key"
            )

        renames: dict[str, str] = {}
        for name, col in weights.items():
            if col not in frame.columns:
                raise ValueError(
                    f"from_table: weight {name!r} names column {col!r}, which "
                    f"is not in the table (columns: {list(frame.columns)})"
                )
            target = weight_column(name)
            if target != col and target in frame.columns:
                raise ValueError(
                    f"from_table: renaming {col!r} to {target!r} for weight "
                    f"{name!r} would overwrite an existing column"
                )
            renames[col] = target
        frame = frame.rename(columns=renames)
        for target in renames.values():
            frame[target] = frame[target].astype("float64")

        schema = OutputSchema(
            id_fields=tuple(id_fields),
            weight_names=tuple(weights),
            source_units=source_units,
            normalization=normalization,
        )

        if check:
            key = id_fields + source_cols
            dup = frame.duplicated(subset=key, keep=False)
            if dup.any():
                shown = frame.loc[dup, key].drop_duplicates().head(5)
                raise ValueError(
                    f"from_table: {int(dup.sum())} rows share a (region, source "
                    f"unit) key {key}; duplicated rows would double count. "
                    f"First duplicated keys:\n{shown.to_string(index=False)}"
                )
            report = check_sum_to_one(frame, schema, tolerance=tolerance)
            if not report.ok:
                raise ValueError(
                    f"from_table: weights do not sum to 1 within each "
                    f"{list(schema.normalization_group)} group under "
                    f"{normalization} normalization.\n{report.summary()}"
                )

        return cls(
            frame=frame,
            schema=schema,
            regions_version=regions_version,
            source_version=source,
        )


def _read_table(path: str | os.PathLike, string_cols: list[str]) -> pd.DataFrame:
    """Read a CSV or parquet weight table, ids as strings."""
    name = str(path).lower()
    if name.endswith((".parquet", ".pq")):
        frame = pd.read_parquet(path)
        for c in string_cols:
            if c in frame.columns:
                frame[c] = frame[c].astype("string")
        return frame
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, dtype={c: "string" for c in string_cols})
    raise ValueError(
        f"from_table: cannot tell the format of {path!s}; expected a .csv, "
        f".csv.gz, .parquet or .pq file"
    )


def _find_column(
    frame: pd.DataFrame, spellings: tuple[str, ...], what: str
) -> str | None:
    """The one column among ``spellings``, None if absent; several raise."""
    found = [c for c in spellings if c in frame.columns]
    if len(found) > 1:
        raise ValueError(
            f"from_table: several {what} columns {found}; keep exactly one"
        )
    return found[0] if found else None


def _pick_column(frame: pd.DataFrame, spellings: tuple[str, ...], what: str) -> str:
    col = _find_column(frame, spellings, what)
    if col is None:
        raise ValueError(
            f"from_table: no {what} column; expected one of {list(spellings)} "
            f"or explicit cell_ix and cell_iy columns"
        )
    return col


def _wrap_lon(lon: np.ndarray, spec: GridSpec) -> np.ndarray:
    """Longitudes moved into the grid's [origin, origin + 360) domain."""
    return np.mod(lon - spec.lon_origin, 360.0) + spec.lon_origin


def _attach_grid_cells(frame: pd.DataFrame, spec: GridSpec) -> pd.DataFrame:
    """Add integer cell indices (and centroids) for a grid source."""
    has_ix = "cell_ix" in frame.columns
    has_iy = "cell_iy" in frame.columns
    if has_ix != has_iy:
        raise ValueError("from_table: cell_ix and cell_iy must be given together")
    if has_ix:
        for c in ("cell_ix", "cell_iy"):
            if not pd.api.types.is_integer_dtype(frame[c]):
                raise ValueError(f"from_table: {c} must be integer, got {frame[c].dtype}")
        ix = frame["cell_ix"].to_numpy(dtype="int64")
        iy = frame["cell_iy"].to_numpy(dtype="int64")
        bad = (ix < 0) | (ix >= spec.n_ix) | (iy < 0) | (iy >= spec.n_iy)
        if bad.any():
            raise ValueError(
                f"from_table: {int(bad.sum())} rows have cell indices outside "
                f"the {spec.n_ix} x {spec.n_iy} grid"
            )
        # round-trip any centroid columns the table also carries: the
        # bounds check alone passes indices built on another grid of the
        # same shape
        lon_col = _find_column(frame, _LON_COLUMNS, "longitude")
        lat_col = _find_column(frame, _LAT_COLUMNS, "latitude")
        if lon_col is not None or lat_col is not None:
            c_lon, c_lat = spec.centroid(ix, iy)
            tol = 1e-6 * spec.resolution
            bad = np.zeros(len(frame), dtype=bool)
            with np.errstate(invalid="ignore"):
                if lon_col is not None:
                    lon = frame[lon_col].to_numpy(dtype="float64")
                    bad |= ~np.isfinite(lon) | (
                        np.abs(c_lon - _wrap_lon(lon, spec)) > tol)
                if lat_col is not None:
                    lat = frame[lat_col].to_numpy(dtype="float64")
                    bad |= ~np.isfinite(lat) | (np.abs(c_lat - lat) > tol)
            if bad.any():
                cols = [c for c in (lon_col, lat_col) if c is not None]
                shown = frame.loc[bad, ["cell_ix", "cell_iy"] + cols].head(5)
                raise ValueError(
                    f"from_table: {int(bad.sum())} rows have cell_ix/cell_iy "
                    f"whose centroids on {spec.source_version} do not match "
                    f"their {cols}; the indices were likely built on a "
                    f"different grid. First:\n{shown.to_string(index=False)}"
                )
        return frame.astype({"cell_ix": "int64", "cell_iy": "int64"})

    lon_col = _pick_column(frame, _LON_COLUMNS, "longitude")
    lat_col = _pick_column(frame, _LAT_COLUMNS, "latitude")
    lon = frame[lon_col].to_numpy(dtype="float64")
    lat = frame[lat_col].to_numpy(dtype="float64")
    wrapped = _wrap_lon(lon, spec)
    ix, iy = spec.index_of(wrapped, lat)
    c_lon, c_lat = spec.centroid(ix, iy)
    tol = 1e-6 * spec.resolution
    with np.errstate(invalid="ignore"):
        bad = (
            ~np.isfinite(lon) | ~np.isfinite(lat)
            | (ix < 0) | (ix >= spec.n_ix) | (iy < 0) | (iy >= spec.n_iy)
            | (np.abs(c_lon - wrapped) > tol) | (np.abs(c_lat - lat) > tol)
        )
    if bad.any():
        shown = frame.loc[bad, [lon_col, lat_col]].head(5)
        raise ValueError(
            f"from_table: {int(bad.sum())} rows are not cell centroids of "
            f"{spec.source_version}; first:\n{shown.to_string(index=False)}"
        )
    out = frame.copy()
    out["cell_ix"] = ix.astype("int64")
    out["cell_iy"] = iy.astype("int64")
    out["cell_lon"] = c_lon.astype("float64")
    out["cell_lat"] = c_lat.astype("float64")
    return out


@dataclass(frozen=True)
class AppliedResult:
    """Aggregated frame plus the validation evidence for the run.

    ``frame`` is keyed by the artifact's id fields plus every passthrough
    dimension of the input data. ``mass_balance`` is None for intensive
    aggregations, whose weighted means are not conserved quantities;
    ``mass_balance_denominator`` is set only for ratio aggregations.
    """

    frame: pd.DataFrame
    kind: VariableKind
    weight: str
    coverage: SourceCoverageReport
    mass_balance: MassBalanceReport | None
    mass_balance_denominator: MassBalanceReport | None = None


def apply_weights(
    artifact: WeightsArtifact,
    data: pd.DataFrame,
    *,
    kind: VariableKind,
    weight: str,
    data_version: str | None,
    value_col: str = "value",
    denominator_col: str | None = None,
    policies: SourceUnitPolicies | None = None,
    restrict_to_sources: Iterable[tuple] | None = None,
    group_col: str | None = None,
    mass_balance_tolerance: float = 1e-6,
    allow_partial_coverage: bool = False,
) -> AppliedResult:
    """Aggregate source-level ``data`` to the artifact's target units.

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
    """
    if kind not in _KIND_TO_DIRECTION:
        raise ValueError(
            f"unknown variable kind {kind!r}; expected one of "
            f"{sorted(_KIND_TO_DIRECTION)}"
        )
    require_normalization(artifact.normalization, _KIND_TO_DIRECTION[kind])
    _check_version_agreement(artifact, data_version)
    _check_coverage_marking(artifact, allow_partial_coverage)

    if kind == "ratio":
        if denominator_col is None:
            raise ValueError("kind='ratio' requires denominator_col")
    elif denominator_col is not None:
        raise ValueError(
            f"denominator_col is only meaningful for kind='ratio' (got kind={kind!r})"
        )

    if weight not in artifact.schema.weight_names:
        raise ValueError(
            f"weight {weight!r} is not in this artifact "
            f"(available: {list(artifact.schema.weight_names)})"
        )
    wcol = weight_column(weight)

    id_fields = list(artifact.schema.id_fields)
    source_keys = list(artifact.schema.source_units.key_columns)
    value_cols = [value_col] + ([denominator_col] if denominator_col else [])
    for col in source_keys + value_cols:
        if col not in data.columns:
            raise ValueError(f"data is missing required column {col!r}")
    if group_col is not None and group_col not in data.columns:
        raise ValueError(f"data is missing group column {group_col!r}")

    dim_cols = [c for c in data.columns if c not in source_keys + value_cols]
    dup_key = source_keys + dim_cols
    if data.duplicated(subset=dup_key).any():
        n = int(data.duplicated(subset=dup_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on (source keys + dimensions) "
            f"{dup_key}; duplicated rows would double count"
        )

    weights_frame = artifact.frame[id_fields + source_keys + [wcol]]
    if restrict_to_sources is not None:
        allowed = {tuple(t) for t in restrict_to_sources}
        mask = weights_frame[source_keys].apply(tuple, axis=1).isin(allowed)
        weights_frame = weights_frame.loc[mask]

    data_source_ids = {
        tuple(row)
        for row in data[source_keys].drop_duplicates().itertuples(index=False, name=None)
    }
    coverage = check_source_coverage(
        weights_frame, artifact.schema, weight, sorted(data_source_ids)
    )
    enforce_source_policies(coverage, policies or SourceUnitPolicies())

    # NaN weights (the marked 'nan' fallback) carry no mass; the coverage
    # check above has already classified all-NaN source units, so the
    # remaining rows drop cleanly here.
    weights_frame = weights_frame.dropna(subset=[wcol])
    merged = weights_frame.merge(data, on=source_keys, how="inner")

    out_keys = id_fields + dim_cols
    if kind == "extensive":
        out = _aggregate_extensive(merged, out_keys, wcol, value_col)
    elif kind == "intensive":
        out = _aggregate_intensive(merged, out_keys, wcol, value_col)
    else:
        num = _aggregate_extensive(merged, out_keys, wcol, value_col)
        den = _aggregate_extensive(merged, out_keys, wcol, denominator_col)
        out = num.merge(den, on=out_keys, how="outer")
        out[value_col] = out[value_col] / out[denominator_col].where(
            out[denominator_col] != 0.0
        )

    balance: MassBalanceReport | None = None
    balance_den: MassBalanceReport | None = None
    if kind in ("extensive", "ratio"):
        applied_ids = {
            tuple(row)
            for row in merged[source_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        applied_mask = data[source_keys].apply(tuple, axis=1).isin(applied_ids)
        applied_data = data.loc[applied_mask]
        if kind == "extensive":
            balance = _run_mass_balance(
                applied_data, out, value_col, group_col, mass_balance_tolerance
            )
        else:
            # For ratio, conservation applies to the numerator and the
            # denominator separately, never to their quotient.
            balance = _run_mass_balance(
                applied_data, num, value_col, group_col, mass_balance_tolerance
            )
            balance_den = _run_mass_balance(
                applied_data, den, denominator_col, group_col, mass_balance_tolerance
            )
            out = out.drop(columns=[denominator_col])

    return AppliedResult(
        frame=out,
        kind=kind,
        weight=weight,
        coverage=coverage,
        mass_balance=balance,
        mass_balance_denominator=balance_den,
    )


def _check_coverage_marking(
    artifact: WeightsArtifact, allow_partial_coverage: bool
) -> None:
    """Refuse artifacts whose generation recorded partially covered units.

    A source unit only partially intersected by the target set (a border
    neighbor in a subset-target run) carries weights that sum to 1 over a
    sliver of the real unit, so sum-to-one cannot flag the problem and an
    extensive application would silently misplace that unit's entire
    value into the sliver's targets. Passing
    ``allow_partial_coverage=True`` is the explicit acknowledgment, for
    callers who have restricted the data to match the artifact's actual
    coverage. Artifacts without accounting (None) pass: hand-built
    artifacts own their frames, and generated polygon artifacts always
    carry the record, count zero included.
    """
    pc = artifact.partial_coverage
    if pc is None or pc.get("count", 0) == 0 or allow_partial_coverage:
        return
    shown = [
        {k: v for k, v in unit.items()}
        for unit in pc.get("units", [])[:5]
    ]
    raise ValueError(
        f"weights artifact records {pc['count']} partially covered source "
        f"units (coverage below {pc['threshold']}; first {len(shown)}: "
        f"{shown}). Their weights sum to 1 over a sliver of the real unit, "
        f"so applying this artifact as if coverage were complete would "
        f"misplace their values. Either use a full-coverage artifact, or "
        f"pass allow_partial_coverage=True after restricting the data to "
        f"what this artifact actually covers."
    )


def _check_version_agreement(
    artifact: WeightsArtifact, data_version: str | None
) -> None:
    if artifact.source_version is None:
        raise ValueError(
            "weights artifact records no source geometry version; only "
            "artifacts built with source.version set (polygon mode) can be "
            "applied. Grid-mode artifacts do not carry a source version yet."
        )
    if data_version is None:
        raise ValueError(
            "data_version is None: the data carries no geometry version "
            "information. State the version the data was built against "
            f"(the artifact records {artifact.source_version!r}); do not "
            "assume compatibility."
        )
    if data_version != artifact.source_version:
        raise ValueError(
            f"geometry version mismatch: the weights artifact was built "
            f"against source version {artifact.source_version!r} but the "
            f"data states {data_version!r}. Rebuild the weights against the "
            f"data's version or use matching data."
        )


def _aggregate_extensive(
    merged: pd.DataFrame, out_keys: list[str], wcol: str, value_col: str
) -> pd.DataFrame:
    contrib = merged[wcol] * merged[value_col]
    out = (
        merged.assign(_contrib=contrib)
        .groupby(out_keys, sort=False)["_contrib"]
        .sum()
        .reset_index()
        .rename(columns={"_contrib": value_col})
    )
    return out


def _aggregate_intensive(
    merged: pd.DataFrame, out_keys: list[str], wcol: str, value_col: str
) -> pd.DataFrame:
    has_value = merged[value_col].notna()
    frame = merged.assign(
        _wx=merged[wcol] * merged[value_col],
        _w=merged[wcol].where(has_value, 0.0),
    )
    sums = frame.groupby(out_keys, sort=False)[["_wx", "_w"]].sum().reset_index()
    sums[value_col] = sums["_wx"] / sums["_w"].where(sums["_w"] != 0.0)
    return sums.drop(columns=["_wx", "_w"])


def _run_mass_balance(
    source_data: pd.DataFrame,
    target: pd.DataFrame,
    value_col: str,
    group_col: str | None,
    tolerance: float,
) -> MassBalanceReport:
    report = check_mass_balance(
        source_data,
        target,
        value_col=value_col,
        group_col=group_col,
        tolerance=tolerance,
    )
    if not report.ok:
        raise ValueError(
            "apply_weights: extensive aggregation lost or gained mass.\n"
            + report.summary()
        )
    return report
