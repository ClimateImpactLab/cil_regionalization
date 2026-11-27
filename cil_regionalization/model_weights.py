"""Climate model weights, read once and recorded with the run.

A weights table has a ``model`` and a ``weight`` column, as the SMME
files do. A member of a projection tree takes the row whose name is
the member's own, letter for letter, or the row declared for it in
``table_names``. Nothing reads a name to decide what it means, so a
member with neither is refused.

The published SMME tables do not pass as they stand. They name the
models in lower case, some with a trailing ``*``, and the surrogates
without the ``surrogate_`` prefix of the trees, so ``ACCESS1-0`` is
the row ``access1-0*`` and ``surrogate_GFDL-CM3_89`` the row
``gfdl-cm3_89``. Until the tables are rewritten with the members'
names, a run that reads them declares each member's row.
"""

from __future__ import annotations

import difflib
import hashlib
import io
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def _listed(items: list[str], quote: bool = True) -> str:
    """'a', 'b' and 'c'."""
    shown = [repr(item) if quote else item for item in items]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


@dataclass(frozen=True)
class ModelWeights:
    """The rows of a weights table, in file order, with the SHA256 of
    the bytes they were parsed from. ``names`` are as written in the
    file, and ``table_names`` the rows declared for members whose names
    are not those of their rows."""

    path: str
    sha256: str
    names: tuple[str, ...]
    weights: tuple[float, ...]
    table_names: Mapping[str, str] = field(default_factory=dict)

    def _row_of(self, member: str) -> int | None:
        name = self.table_names.get(member, member)
        return self.names.index(name) if name in self.names else None

    def _closest(self, name: str) -> Optional[str]:
        # for the message only, nothing is matched by it
        if not self.names:
            return None
        folded = name.casefold()
        return max(
            self.names,
            key=lambda row: difflib.SequenceMatcher(
                None, folded, row.casefold()
            ).ratio(),
        )

    def missing(self, members: Iterable[str]) -> list[str]:
        """The members that have no row in the table, sorted."""
        return sorted(m for m in set(members) if self._row_of(m) is None)

    def check(self, members: Iterable[str]) -> None:
        """Raises if a member has no row in the table, naming the row
        each one comes closest to, or if two members take one row."""
        members = list(dict.fromkeys(members))
        absent = self.missing(members)
        if absent:
            raise ValueError(self._unmatched(absent))
        taken: dict[int, list[str]] = {}
        for member in members:
            taken.setdefault(self._row_of(member), []).append(member)
        shared = [(row, names) for row, names in taken.items() if len(names) > 1]
        if shared:
            row, names = shared[0]
            more = f", and {len(shared) - 1} more rows" if len(shared) > 1 else ""
            raise ValueError(
                f"The model weights file {self.path} would give one row to "
                f"more than one member, the row {self.names[row]!r} to "
                f"{_listed(names)}{more}. Each member needs a row of its own."
            )

    def _unmatched(self, absent: list[str]) -> str:
        shown = []
        for member in absent[:5]:
            declared = self.table_names.get(member)
            closest = self._closest(member if declared is None else declared)
            near = (
                "the table has no rows"
                if closest is None
                else f"closest row {closest!r}"
            )
            shown.append(
                f"{member!r} ({near})"
                if declared is None
                else f"{member!r} declared as {declared!r} ({near})"
            )
        if len(absent) > 5:
            shown.append(f"{len(absent) - 5} more")
        first = absent[0]
        example = self._closest(self.table_names.get(first, first)) or "its row"
        counted = "1 member has" if len(absent) == 1 else f"{len(absent)} members have"
        return (
            f"{counted} no row in the model weights file "
            f"{self.path}: {_listed(shown, quote=False)}. A member takes the "
            f"row of its own name, letter for letter, or the row declared for "
            f"it. Write the members' names into the table, or declare each "
            f"member's row in stats.model_weight_names, or in table_names of "
            f'read_model_weights, as in "{first}" = "{example}".'
        )

    def weight_of(self, member: str) -> float:
        """The weight of one member. Raises if it has no row."""
        row = self._row_of(member)
        if row is None:
            raise ValueError(self._unmatched([member]))
        return self.weights[row]

    def attach(self, members: pd.Series) -> pd.Series:
        """The weight of each row's member, as float64 on the index of
        ``members``. Raises if a member has no row in the table, if two
        members take one row, or if a row names no member."""
        codes, distinct = pd.factorize(members)
        if (codes < 0).any():
            n = int((codes < 0).sum())
            raise ValueError(
                f"{n} {'row names' if n == 1 else 'rows name'} no member, so "
                f"{'it takes' if n == 1 else 'they take'} no weight from the "
                f"model weights file {self.path}"
            )
        self.check(distinct)
        per_member = np.array([self.weight_of(m) for m in distinct], dtype=np.float64)
        return pd.Series(per_member[codes], index=members.index, name="model_weight")

    def record(self, members: Iterable[str]) -> dict:
        """The manifest entry of a run, given the member of each leaf
        that was pooled. Its 'members' holds each member with its table
        row, its weight and its number of leaves, and 'unmatched_rows'
        the weight of every row no member took, so the two together
        cover the whole table. 'table_names' are the rows declared, and
        'table_names_unused' those declared for a member the run does
        not have. Raises as `check` does."""
        counts = Counter(members)
        self.check(counts)
        attached = {
            member: {
                "row": self.names[self._row_of(member)],
                "weight": self.weights[self._row_of(member)],
                "n_leaves": counts[member],
            }
            for member in sorted(counts)
        }
        matched = {entry["row"] for entry in attached.values()}
        return {
            "path": self.path,
            "sha256": self.sha256,
            "table_names": dict(self.table_names),
            "table_names_unused": {
                member: row
                for member, row in self.table_names.items()
                if member not in counts
            },
            "members": attached,
            "unmatched_rows": {
                name: weight
                for name, weight in zip(self.names, self.weights)
                if name not in matched
            },
        }


def read_model_weights(
    path: str | Path,
    *,
    table_names: Optional[Mapping[str, str]] = None,
    sep: str = "\t",
) -> ModelWeights:
    """Read a weights table. The file is read once, and the hash is of
    the bytes the rows were parsed from. ``table_names`` declares the
    row of each member whose name is not that of its row, member name to
    table name, and is checked where a member takes its row. The
    published SMME tables need it until they carry the members' names.
    Raises if a name is in the table twice, or if two members are
    declared to one row."""
    raw = Path(path).read_bytes()
    table = pd.read_csv(io.BytesIO(raw), sep=sep)
    if not {"model", "weight"}.issubset(table.columns):
        raise ValueError(
            f"model weights file {path} needs 'model' and 'weight' "
            f"columns; found {sorted(table.columns)}"
        )
    names = table["model"].astype(str)
    repeated = sorted(names[names.duplicated()].unique())
    if repeated:
        raise ValueError(
            f"model weights file {path} has the "
            f"{'row' if len(repeated) == 1 else 'rows'} {_listed(repeated)} more "
            f"than once. Each model needs exactly one row."
        )
    declared = dict(table_names or {})
    for member, row in declared.items():
        if not isinstance(member, str) or not isinstance(row, str) or not row:
            raise ValueError(
                f"table_names maps a member name to a table name, both "
                f"strings; got {member!r} = {row!r}"
            )
    rows: dict[str, list[str]] = {}
    for member, row in declared.items():
        rows.setdefault(row, []).append(member)
    shared = {row: members for row, members in rows.items() if len(members) > 1}
    if shared:
        row, members = next(iter(shared.items()))
        more = f", and {len(shared) - 1} more rows" if len(shared) > 1 else ""
        raise ValueError(
            f"table_names declares one row of the model weights file {path} "
            f"for more than one member, the row {row!r} for "
            f"{_listed(members)}{more}. Each member needs a row of its own."
        )
    return ModelWeights(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        names=tuple(names),
        weights=tuple(float(w) for w in table["weight"]),
        table_names=declared,
    )
