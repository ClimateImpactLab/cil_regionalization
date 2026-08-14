"""Canonical output schema both backends must produce.

A single source of truth for column names, dtypes, and ordering so that
local and BigQuery results can be compared row-for-row in Stage 4. Column
naming follows the locked design decision:

    {name}wt     normalized weight per (region, source unit)
    {name}_raw   raw weight total used as the numerator
    {name}_method per-region fallback label: native | area_fallback | zero | nearest_cell

Source units
------------
Each output row pairs one region (the ``id_fields``) with one *source
unit*: the unit of the geometry the weighted data lives on. For the grid
backends the source unit is a grid cell, identified by integer
``(cell_ix, cell_iy)`` with centroid metadata. A polygon source (for
example impact regions keyed by ``hierid``, split onto administrative
targets) identifies its units by string id columns instead. `SourceUnits`
describes that layout; `GRID_CELLS` is the grid case and the default.

Normalization direction
-----------------------
The same intersection geometry supports two weight normalizations, and
they are not interchangeable:

    per_destination : weights sum to 1 within each region, across the
        source units covering it. Consumed as a weighted mean of an
        intensive variable (the grid-to-IR climate case).
    per_source : weights sum to 1 within each source unit, across the
        regions it overlaps. Consumed as an allocation of an extensive
        quantity (splitting one source unit's total across targets).

The direction is part of the schema and is recorded in the manifest.
Consumers must call `require_normalization` before applying a weights
frame so that a file cannot be used for the wrong operation silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from segment_weights.config import Normalization


CELL_COLUMNS: tuple[str, ...] = ("cell_ix", "cell_iy", "cell_lon", "cell_lat")
CELL_DTYPES: dict[str, str] = {
    "cell_ix": "int64",
    "cell_iy": "int64",
    "cell_lon": "float64",
    "cell_lat": "float64",
}


@dataclass(frozen=True)
class SourceUnits:
    """Identifier layout for the source side of a weights frame.

    ``key_columns`` identify one source unit (join and duplicate-check
    keys). ``meta_columns`` are carried alongside but are not part of the
    identity (the grid case carries cell centroids). ``dtype_items`` holds
    (column, dtype) pairs for every column, as a tuple so the dataclass
    stays frozen and hashable.
    """

    key_columns: tuple[str, ...]
    dtype_items: tuple[tuple[str, str], ...]
    meta_columns: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return self.key_columns + self.meta_columns

    @property
    def dtypes(self) -> dict[str, str]:
        return dict(self.dtype_items)

    @classmethod
    def from_string_ids(cls, key_columns: Iterable[str]) -> "SourceUnits":
        """Source units identified by string id columns (e.g. ``hierid``)."""
        keys = tuple(key_columns)
        if not keys:
            raise ValueError("SourceUnits requires at least one key column")
        return cls(
            key_columns=keys,
            dtype_items=tuple((c, "string") for c in keys),
        )


GRID_CELLS = SourceUnits(
    key_columns=("cell_ix", "cell_iy"),
    dtype_items=tuple(CELL_DTYPES.items()),
    meta_columns=("cell_lon", "cell_lat"),
)


def weight_column(name: str) -> str:
    return f"{name}wt"


def raw_column(name: str) -> str:
    return f"{name}_raw"


def method_column(name: str) -> str:
    return f"{name}_method"


def weight_dtypes(name: str) -> dict[str, str]:
    return {
        weight_column(name): "float64",
        raw_column(name): "float64",
        method_column(name): "string",
    }


@dataclass(frozen=True)
class OutputSchema:
    """Resolved column layout for one run: id fields + source units + weights.

    Defaults reproduce the grid layout exactly: grid-cell source units and
    per_destination normalization.
    """

    id_fields: tuple[str, ...]
    weight_names: tuple[str, ...]
    source_units: SourceUnits = GRID_CELLS
    normalization: Normalization = "per_destination"

    @property
    def columns(self) -> tuple[str, ...]:
        cols: list[str] = list(self.id_fields) + list(self.source_units.columns)
        for name in self.weight_names:
            cols += [weight_column(name), raw_column(name), method_column(name)]
        return tuple(cols)

    @property
    def dtypes(self) -> dict[str, str]:
        d: dict[str, str] = {f: "string" for f in self.id_fields}
        d.update(self.source_units.dtypes)
        for name in self.weight_names:
            d.update(weight_dtypes(name))
        return d

    @property
    def normalization_group(self) -> tuple[str, ...]:
        """Columns the weights sum to 1 within, per the declared direction."""
        if self.normalization == "per_destination":
            return self.id_fields
        return self.source_units.key_columns

    def empty_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({c: pd.Series(dtype=self.dtypes[c]) for c in self.columns})
        return df

    def validate_frame(self, df: pd.DataFrame) -> None:
        """Raise if `df` is missing columns or has wrong dtypes for declared cols."""
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"output schema: missing columns {missing}")
        wrong: list[str] = []
        for c, expected in self.dtypes.items():
            actual = str(df[c].dtype)
            if not _dtype_compatible(actual, expected):
                wrong.append(f"{c}: expected {expected}, got {actual}")
        if wrong:
            raise ValueError("output schema: dtype mismatches " + "; ".join(wrong))


def require_normalization(recorded: str, required: Normalization) -> None:
    """Raise unless a weights artifact's recorded direction matches `required`.

    ``recorded`` comes from the artifact (the schema of a live result, or
    the ``normalization`` field of its manifest). ``required`` is the
    direction the intended operation needs: ``per_destination`` for a
    weighted mean of an intensive variable, ``per_source`` for allocating
    an extensive quantity. The two are transposes over the same geometry;
    applying one where the other is needed produces plausible-looking
    wrong numbers, so the mismatch is an error, not a warning.
    """
    if recorded == required:
        return
    raise ValueError(
        f"weights normalization mismatch: artifact records {recorded!r} but the "
        f"requested operation requires {required!r}. per_destination weights "
        f"sum to 1 within each region (weighted mean of an intensive "
        f"variable); per_source weights sum to 1 within each source unit "
        f"(allocation of an extensive quantity). Recompute the weights with "
        f"the required normalization instead of reusing this file."
    )


_STRING_ALIASES = frozenset(
    {"string", "string[python]", "string[pyarrow]", "object", "str"}
)


def _dtype_compatible(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if expected == "string" and actual in _STRING_ALIASES:
        return True
    if expected.startswith("int") and actual.startswith("int"):
        return True
    if expected.startswith("float") and actual.startswith("float"):
        return True
    return False
