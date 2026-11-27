"""Climate model weights, read once and recorded with the run.

A weights table is tab separated and has a ``model`` and a ``weight``
column, as the SMME files do. A table name matches a member of a
projection tree after a ``*`` is dropped from the name, the
``surrogate_`` prefix is dropped from the member, and both are
lowercased.
"""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _member_key(member: str) -> str:
    return member.replace("surrogate_", "").lower()


@dataclass(frozen=True)
class ModelWeights:
    """The rows of a weights table, in file order, with the SHA256 of
    the bytes they were parsed from. ``names`` are as written in the
    file and ``keys`` are what a member is matched on."""

    path: str
    sha256: str
    names: tuple[str, ...]
    keys: tuple[str, ...]
    weights: tuple[float, ...]

    def _row_of(self, member: str) -> int | None:
        key = _member_key(member)
        return self.keys.index(key) if key in self.keys else None

    def missing(self, members: Iterable[str]) -> list[str]:
        """The members that have no row in the table, sorted."""
        return sorted(m for m in set(members) if self._row_of(m) is None)

    def weight_of(self, member: str) -> float:
        """The weight of one member. Raises if it has no row."""
        row = self._row_of(member)
        if row is None:
            raise ValueError(
                f"{member!r} has no row in the model weights file {self.path}"
            )
        return self.weights[row]

    def attach(self, members: pd.Series) -> pd.Series:
        """The weight of each row's member, as float64 on the index of
        ``members``. Raises if a member has no row in the table, or a
        row names no member."""
        codes, distinct = pd.factorize(members)
        absent = self.missing(distinct)
        if absent or (codes < 0).any():
            raise ValueError(
                f"{len(absent)} members have no row in the model weights "
                f"file {self.path} (first: {absent[:5]}), and "
                f"{int((codes < 0).sum())} rows name no member"
            )
        per_member = np.array([self.weight_of(m) for m in distinct], dtype=np.float64)
        return pd.Series(per_member[codes], index=members.index, name="model_weight")

    def record(self, members: Iterable[str]) -> dict:
        """The manifest entry of a run, given the member of each leaf
        that was pooled. Its 'members' holds each member with its table
        row, its weight and its number of leaves, and 'unmatched_rows'
        the weight of every row no member matched, so the two together
        cover the whole table. Raises if a member has no row."""
        counts = Counter(members)
        absent = self.missing(counts)
        if absent:
            raise ValueError(
                f"{len(absent)} members have no row in the model weights "
                f"file {self.path} (first: {absent[:5]})"
            )
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
            "members": attached,
            "unmatched_rows": {
                name: weight
                for name, weight in zip(self.names, self.weights)
                if name not in matched
            },
        }


def read_model_weights(path: str | Path) -> ModelWeights:
    """Read a weights table. The file is read once, and the hash is of
    the bytes the rows were parsed from. Raises if two rows match the
    same member."""
    raw = Path(path).read_bytes()
    table = pd.read_csv(io.BytesIO(raw), sep="\t")
    if not {"model", "weight"}.issubset(table.columns):
        raise ValueError(
            f"model weights file {path} needs 'model' and 'weight' "
            f"columns; found {sorted(table.columns)}"
        )
    keys = table["model"].str.replace("*", "", regex=False).str.lower()
    repeated = keys[keys.duplicated(keep=False)]
    if len(repeated):
        groups = [
            table.loc[repeated.index[repeated == key], "model"].tolist()
            for key in repeated.unique()
        ]
        raise ValueError(
            f"model weights file {path} has rows that match the same "
            f"member: {groups}. Each model needs exactly one row."
        )
    return ModelWeights(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        names=tuple(table["model"]),
        keys=tuple(keys),
        weights=tuple(float(w) for w in table["weight"]),
    )
