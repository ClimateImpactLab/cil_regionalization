"""Regrouping a reduced frame to coarser targets before its statistics
are pooled, as country values are made from the states of a country.

A statistic cannot be summed or averaged across targets, so this works
on the reduced frame, one value a target, window and sample member, and
the statistics are taken afterwards with `pooled_statistics`.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from cil_regionalization.apply import WeightsArtifact
from cil_regionalization.schema import raw_column

BASIS_COLUMN = "basis"


def target_basis(artifact: WeightsArtifact, weight: str) -> pd.DataFrame:
    """Each target's weight basis, the sum of the raw column of
    ``weight`` over its rows, its population or its crop area, as the id
    fields and a column 'basis'. A row with no basis counts as none.
    Raises if the artifact has no raw column for the weight."""
    column = raw_column(weight)
    if column not in artifact.frame.columns:
        raise ValueError(
            f"the weights artifact has no column {column!r}, so it does not "
            f"say the basis of its targets"
        )
    id_fields = list(artifact.schema.id_fields)
    frame = artifact.frame[id_fields + [column]]
    return (
        frame.assign(**{BASIS_COLUMN: frame[column].fillna(0.0)})
        .groupby(id_fields, sort=False)[BASIS_COLUMN]
        .sum()
        .reset_index()
    )


def regroup(
    reduced: pd.DataFrame,
    *,
    id_fields: Sequence[str],
    by: Sequence[str],
    kind: Literal["extensive", "intensive"],
    value_col: str,
    summation: str,
    basis: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``reduced`` with its targets grouped by the id fields ``by``, one
    value a group and every other column, so that a country is the sum
    or the mean of its states in each window and sample member.

    "extensive" sums the values, and needs no ``basis``. "intensive"
    takes the mean weighted by each target's basis, the frame
    `target_basis` gives, over the targets with a value. A group with no
    value is NaN. ``summation`` is "fast" or "exact", as for the
    statistics. Raises if ``by`` is not among ``id_fields``, or a row's
    target has no basis."""
    from cil_regionalization.stats import check_summation

    check_summation(summation)
    id_fields, by = list(id_fields), list(by)
    stray = [f for f in by + id_fields if f not in reduced.columns]
    if stray:
        raise ValueError(f"reduced has no column {stray}")
    if not by or any(f not in id_fields for f in by):
        raise ValueError(f"by {by} has to name some of the id fields {id_fields}")
    if kind not in ("extensive", "intensive"):
        raise ValueError(f"kind has to be 'extensive' or 'intensive'; got {kind!r}")
    if (basis is None) != (kind == "extensive"):
        raise ValueError(
            "an intensive regroup is a mean weighted by each target's basis, "
            "and an extensive one is a sum, which takes none"
        )
    keys = by + [c for c in reduced.columns if c not in id_fields + [value_col]]
    values = reduced[value_col].astype("float64")
    if kind == "extensive":
        weights = pd.Series(1.0, index=reduced.index)
    else:
        found = reduced[id_fields].merge(
            basis[id_fields + [BASIS_COLUMN]], on=id_fields, how="left"
        )[BASIS_COLUMN]
        if found.isna().any():
            first = reduced.loc[found.isna().to_numpy(), id_fields].iloc[0].to_dict()
            raise ValueError(
                f"{int(found.isna().sum())} rows of the reduced frame have a "
                f"target with no basis (first: {first})"
            )
        weights = pd.Series(found.to_numpy(dtype="float64"), index=reduced.index)
    has = values.notna()
    work = reduced[keys].assign(
        _wx=(weights * values).where(has), _w=weights.where(has, 0.0)
    )
    grouped = work.groupby(keys, sort=False, observed=True, dropna=False)
    if summation == "exact":
        sums = grouped[["_wx", "_w"]].agg(
            lambda part: math.fsum(part.dropna()) if part.notna().any() else np.nan
        )
    else:
        sums = grouped[["_wx", "_w"]].sum(min_count=1)
    sums = sums.reset_index()
    if kind == "extensive":
        sums[value_col] = sums["_wx"]
    else:
        sums[value_col] = sums["_wx"] / sums["_w"].where(sums["_w"] != 0.0)
    return sums[keys + [value_col]]
