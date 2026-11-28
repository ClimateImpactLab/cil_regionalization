"""Applying a weights artifact to source-level data.

`apply_weights` joins the data to the weights on the source key columns
the artifact records and groups to the id fields it records. Joins never
use display names. Every other column of the data is a dimension carried
through to the output, and no dimension name is special.

The variable kind is declared, never inferred. An extensive value is the
sum of weight times value, and needs per_source weights. An intensive
value is that sum divided by the sum of the weights of the source units
that have a value, and needs per_destination weights. A ratio sums its
numerator and its denominator extensively and then divides, and needs
per_source weights. A kind that does not match the artifact's recorded
normalization raises, and so does a ``data_version`` that is missing or
differs from the artifact's ``source_version``.

Every application checks source coverage against `SourceUnitPolicies`,
which raise by default, and reports it on the result. An extensive or
ratio application also checks the mass balance over the data it applied,
and raises if it fails. A run over a subset of the sources names them in
``restrict_to_sources`` rather than relaxing the policies.
"""

from __future__ import annotations

import json
import logging
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
    raw_column,
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

logger = logging.getLogger(__name__)

# The column `apply_weights` writes on every row, the share of the
# target's weight basis behind the value.
SHARE_COLUMN = "weight_share"
# why a target the weights know is not in the output
NO_WEIGHT = "no weight"
NO_DATA = "no data"

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

    ``partial_coverage`` is the record of partially covered source units
    that every generated polygon artifact carries, a count of zero
    included. It holds the threshold, the count, the units below the
    threshold and whether the targets were a subset. `apply_weights`
    refuses an artifact whose count is not zero unless
    ``allow_partial_coverage`` is set. It is None for an artifact built
    by hand, a grid artifact and one loaded from a manifest without the
    record, and such an artifact is not refused.
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
        """The artifact written as ``{stem}.parquet`` and
        ``{stem}.manifest.json`` in ``output_dir``, a local directory or a
        ``gs://`` prefix. The schema is rebuilt from the column roles the
        manifest records. Raises if the manifest does not record them, as
        a manifest from an older version of the package does not."""
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
            manifest = json.loads((Path(base) / f"{stem}.manifest.json").read_text())

        missing = [
            k
            for k in (
                "id_fields",
                "source_key_columns",
                "weight_names",
                "normalization",
            )
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
            source_units = SourceUnits.from_string_ids(manifest["source_key_columns"])
        schema = OutputSchema(
            id_fields=tuple(manifest["id_fields"]),
            weight_names=tuple(manifest["weight_names"]),
            source_units=source_units,
            normalization=manifest["normalization"],
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
        normalization: str,
        source_keys: Sequence[str] | None = None,
        regions_version: str | None = None,
        check: bool = True,
        tolerance: float = 1e-6,
    ) -> "WeightsArtifact":
        """An artifact built from a weight table made elsewhere, with
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
            raise ValueError(
                "from_table: weights must map at least one name to a column"
            )

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
            spec = GridSpec.from_config(
                GridConfig(
                    mode="generate",
                    resolution=float(grid_match["res"]),
                    offset=grid_match["offset"],
                    lon_convention=grid_match["conv"],
                )
            )
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
                f"from_table: {overlap} cannot be both an id field and a source key"
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
    """The one column of ``frame`` among ``spellings``, or None if there
    is none. Raises if there are several."""
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
    """``frame`` with int64 ``cell_ix`` and ``cell_iy`` on ``spec``, and
    with ``cell_lon`` and ``cell_lat`` too when the indices are found
    from the coordinates. Raises if a row is off the grid, or its indices
    and its coordinates disagree."""
    has_ix = "cell_ix" in frame.columns
    has_iy = "cell_iy" in frame.columns
    if has_ix != has_iy:
        raise ValueError("from_table: cell_ix and cell_iy must be given together")
    if has_ix:
        for c in ("cell_ix", "cell_iy"):
            if not pd.api.types.is_integer_dtype(frame[c]):
                raise ValueError(
                    f"from_table: {c} must be integer, got {frame[c].dtype}"
                )
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
                        np.abs(c_lon - _wrap_lon(lon, spec)) > tol
                    )
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
            ~np.isfinite(lon)
            | ~np.isfinite(lat)
            | (ix < 0)
            | (ix >= spec.n_ix)
            | (iy < 0)
            | (iy >= spec.n_iy)
            | (np.abs(c_lon - wrapped) > tol)
            | (np.abs(c_lat - lat) > tol)
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
    aggregations, whose weighted means are not conserved quantities.
    ``mass_balance_denominator`` is set only for ratio aggregations.
    ``targets`` is what `apply_weights` found of the targets.
    """

    frame: pd.DataFrame
    kind: VariableKind
    weight: str
    coverage: SourceCoverageReport
    mass_balance: MassBalanceReport | None
    mass_balance_denominator: MassBalanceReport | None = None
    targets: "TargetReport | None" = None


@dataclass(frozen=True)
class TargetReport:
    """The targets an application was asked for that are not in its
    output, by reason, and those whose value rests on part of their
    weight basis, each with the smallest share a row of it has.
    ``basis`` is the column the shares are taken against.

    A target is asked for when the weights, restricted to the sources
    asked for, have a row of it. It is left out for "no weight" when
    none of those rows has a weight, and for "no data" when none of its
    weighted sources is in the data."""

    basis: str
    left_out: Mapping[str, tuple[tuple, ...]]
    partial: tuple[tuple[tuple, float], ...]

    @property
    def counts(self) -> dict:
        return {
            "left_out": {reason: len(ids) for reason, ids in self.left_out.items()},
            "partial": len(self.partial),
        }


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

    if SHARE_COLUMN in data.columns:
        raise ValueError(
            f"data has a column named {SHARE_COLUMN!r}, which apply_weights "
            f"writes on every row of its output"
        )
    dim_cols = [c for c in data.columns if c not in source_keys + value_cols]
    dup_key = source_keys + dim_cols
    # Each key column is hashed once: the source codes serve both the
    # duplicate check and the set of source units present in the data.
    source_codes, source_card = _group_codes(data, source_keys)
    row_codes, row_card = _group_codes(data, dim_cols, source_codes, source_card)
    has_duplicates = _distinct_count(row_codes, row_card) < len(data)
    del row_codes
    if has_duplicates:
        n = int(data.duplicated(subset=dup_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on (source keys + dimensions) "
            f"{dup_key}; duplicated rows would double count"
        )

    raw = raw_column(weight)
    basis_col = raw if raw in artifact.frame.columns else wcol
    weights_frame = _with_basis(
        artifact.frame[
            id_fields + source_keys + list(dict.fromkeys([wcol, basis_col]))
        ],
        id_fields,
        wcol,
        basis_col,
    )
    if restrict_to_sources is not None:
        allowed = {tuple(t) for t in restrict_to_sources}
        mask = weights_frame[source_keys].apply(tuple, axis=1).isin(allowed)
        weights_frame = weights_frame.loc[mask]

    first_rows = _first_rows(source_codes, source_card)
    del source_codes
    data_source_ids = {
        tuple(row)
        for row in data[source_keys].take(first_rows).itertuples(index=False, name=None)
    }
    coverage = check_source_coverage(
        weights_frame, artifact.schema, weight, data_source_ids
    )
    enforce_source_policies(coverage, policies or SourceUnitPolicies())

    requested = _ids(weights_frame, id_fields)
    # A row with no weight, NaN as the marked 'nan' fallback leaves it or
    # zero, carries no mass. The coverage check above has classified the
    # source units all of whose rows are such.
    weighted = weights_frame[weights_frame[wcol].notna() & (weights_frame[wcol] != 0)]
    merged = weighted.merge(data, on=source_keys, how="inner")

    out_keys = id_fields + dim_cols
    contributing = merged[value_col].notna()
    if denominator_col is not None:
        contributing &= merged[denominator_col].notna()
    columns = {
        "_bc": merged["_basis"].where(contributing, 0.0),
        "_cc": (contributing & (merged["_basis"] > 0)).astype(np.int64),
    }
    if kind == "intensive":
        columns["_wx"] = merged[wcol] * merged[value_col]
        columns["_w"] = merged[wcol].where(merged[value_col].notna(), 0.0)
    else:
        columns["_num"] = merged[wcol] * merged[value_col]
        if kind == "ratio":
            columns["_den"] = merged[wcol] * merged[denominator_col]
    grouped = (
        merged[out_keys + ["_total", "_rows"]]
        .assign(**columns)
        .groupby(out_keys, sort=False)
    )
    sums = grouped[list(columns)].sum(min_count=1)
    sums[["_total", "_rows"]] = grouped[["_total", "_rows"]].first()
    sums = sums.reset_index()
    if kind == "intensive":
        sums[value_col] = sums["_wx"] / sums["_w"].where(sums["_w"] != 0.0)
    elif kind == "extensive":
        sums[value_col] = sums["_num"]
    else:
        sums[value_col] = sums["_num"] / sums["_den"].where(sums["_den"] != 0.0)
    # every row of the target that carries basis contributed, so exactly 1
    sums[SHARE_COLUMN] = np.where(
        sums["_cc"].fillna(0).to_numpy() == sums["_rows"].to_numpy(),
        1.0,
        sums["_bc"].fillna(0.0) / sums["_total"],
    )
    out = sums[out_keys + [value_col, SHARE_COLUMN]]
    if kind != "intensive":
        num = sums[out_keys + ["_num"]].rename(columns={"_num": value_col})
    if kind == "ratio":
        den = sums[out_keys + ["_den"]].rename(columns={"_den": denominator_col})

    present = _ids(out, id_fields)
    weighted_ids = _ids(weighted, id_fields)
    shares = out.groupby(id_fields, sort=False)[SHARE_COLUMN].min()
    partial = shares[shares < 1.0]
    targets = TargetReport(
        basis=basis_col,
        left_out={
            NO_WEIGHT: tuple(sorted(requested - weighted_ids)),
            NO_DATA: tuple(sorted(weighted_ids - present)),
        },
        partial=tuple(
            sorted(
                ((key if isinstance(key, tuple) else (key,)), float(share))
                for key, share in partial.items()
            )
        ),
    )
    _say_targets(targets, len(requested))

    balance: MassBalanceReport | None = None
    balance_den: MassBalanceReport | None = None
    if kind in ("extensive", "ratio"):
        applied_ids = {
            tuple(row)
            for row in merged[source_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        applied_mask = _rows_in(data, source_keys, applied_ids)
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

    return AppliedResult(
        frame=out,
        kind=kind,
        weight=weight,
        coverage=coverage,
        mass_balance=balance,
        mass_balance_denominator=balance_den,
        targets=targets,
    )


def _ids(frame: pd.DataFrame, id_fields: list[str]) -> set[tuple]:
    return set(frame[id_fields].drop_duplicates().itertuples(index=False, name=None))


def _with_basis(
    frame: pd.DataFrame, id_fields: list[str], wcol: str, basis_col: str
) -> pd.DataFrame:
    """``frame`` with each row's basis, its target's total basis and the
    number of its target's rows that carry basis, over every row of the
    artifact. A target whose rows carry no basis at all, as rows of the
    area fallback carry none of the raster's, is measured by its
    weights instead."""
    basis = frame[basis_col].fillna(0.0).astype("float64")
    keys = [frame[f] for f in id_fields]
    total = basis.groupby(keys, sort=False).transform("sum")
    by_weight = total == 0.0
    if by_weight.any():
        basis = basis.where(~by_weight, frame[wcol].fillna(0.0).astype("float64"))
        total = basis.groupby(keys, sort=False).transform("sum")
    rows = (basis > 0).groupby(keys, sort=False).transform("sum")
    return frame.assign(_basis=basis, _total=total, _rows=rows.astype(np.int64))


def _say_targets(targets: TargetReport, n_requested: int) -> None:
    left = {reason: len(ids) for reason, ids in targets.left_out.items() if ids}
    if not left and not targets.partial:
        return
    parts = []
    if left:
        parts.append(
            f"{sum(left.values())} of {n_requested} targets are left out ("
            + ", ".join(f"{reason}: {n}" for reason, n in left.items())
            + ")"
        )
    if targets.partial:
        smallest = min(share for _, share in targets.partial)
        parts.append(
            f"{len(targets.partial)} rest on part of their {targets.basis} "
            f"(smallest share {smallest:.3g}, in column {SHARE_COLUMN!r})"
        )
    logger.warning("%s; AppliedResult.targets lists them", " and ".join(parts))


def _check_coverage_marking(
    artifact: WeightsArtifact, allow_partial_coverage: bool
) -> None:
    """Raises if the artifact records partially covered source units and
    ``allow_partial_coverage`` is false. An artifact with no record
    passes. A caller sets ``allow_partial_coverage`` after restricting
    the data to what the artifact covers."""
    # A source unit only partly intersected by the targets, a border
    # neighbor in a run over a subset of targets, has weights that sum to
    # 1 over a sliver of the real unit. Sum-to-one cannot flag it, and an
    # extensive application would put the unit's whole value into the
    # sliver's targets.
    pc = artifact.partial_coverage
    if pc is None or pc.get("count", 0) == 0 or allow_partial_coverage:
        return
    shown = [{k: v for k, v in unit.items()} for unit in pc.get("units", [])[:5]]
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


# Direct-address tables are used while a code range stays within this
# many slots per data row; past it, codes are compressed by hashing.
_DENSE_SLOTS_PER_ROW = 8


def _dense_ok(card: int, n_rows: int) -> bool:
    return card <= _DENSE_SLOTS_PER_ROW * (n_rows + 1)


def _group_codes(
    frame: pd.DataFrame,
    cols: list[str],
    codes: np.ndarray | None = None,
    card: int = 1,
) -> tuple[np.ndarray, int]:
    """One int64 code per row, equal exactly when the rows agree on
    ``cols``, and the bound ``card`` with every code in ``[0, card)``.

    Missing values are all equal to each other, as in
    `DataFrame.duplicated`. Passing ``codes`` and ``card`` extends an
    existing code with more columns.
    """
    # Each column is factorized once and the codes are combined
    # arithmetically. They are compressed by factorizing only when the
    # combined range would outgrow `_dense_ok`, which also keeps the
    # product far from int64 overflow.
    n = len(frame)
    for col in cols:
        c, uniques = pd.factorize(frame[col])
        c = c.astype(np.int64) + 1  # missing values (-1) become group 0
        k = len(uniques) + 1
        if codes is None:
            codes, card = c, k
            continue
        if not _dense_ok(card * k, n):
            codes, uniques = pd.factorize(codes)
            card = len(uniques)
        codes = codes * k + c
        card *= k
    if codes is None:
        codes = np.zeros(n, dtype=np.int64)
    return codes, card


def _distinct_count(codes: np.ndarray, card: int) -> int:
    """Number of distinct values in ``codes``, which lie in ``[0, card)``."""
    if _dense_ok(card, len(codes)):
        seen = np.zeros(card, dtype=bool)
        seen[codes] = True
        return int(np.count_nonzero(seen))
    return len(pd.unique(codes))


def _first_rows(codes: np.ndarray, card: int) -> np.ndarray:
    """Positions of the first row of each distinct code, in row order.

    The rows `drop_duplicates` would keep for the columns behind ``codes``.
    """
    n = len(codes)
    if card > n:
        codes, uniques = pd.factorize(codes)
        card = len(uniques)
    first = np.full(card, n, dtype=np.int64)
    np.minimum.at(first, codes, np.arange(n, dtype=np.int64))
    return np.sort(first[first < n])


def _rows_in(frame: pd.DataFrame, keys: list[str], ids: set[tuple]) -> np.ndarray:
    """Boolean mask of the rows whose ``keys`` tuple is in ``ids``, the
    same as ``frame[keys].apply(tuple, axis=1).isin(ids)``."""
    # This builds no Python tuple per row. Building one per row cost
    # minutes and about 16 GiB at 100M rows.
    return pd.MultiIndex.from_frame(frame[keys]).isin(ids)


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
