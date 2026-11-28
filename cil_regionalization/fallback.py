"""Weights from the raw totals of one weight per (region, source unit),
with a fallback for groups whose raw total is zero.

Source units are grid cells unless a `SourceUnits` layout says otherwise.
Under ``per_destination`` normalization the raw totals are summed within
each region (the ``id_fields``), so weights sum to 1 within each region.
Under ``per_source`` they are summed within each source unit across the
regions it overlaps, so weights sum to 1 within each source unit.

A group whose raw total is zero follows the weight's fallback policy.
``"area"`` gives its rows their area weight ``areawt`` and the method
``"area_fallback"``. ``"zero"`` gives them weight 0 and the method
``"zero"``, and the sum-to-one check flags the group. ``"nan"`` gives
them weight NaN and the method ``"nan"``, and the sum-to-one check
passes the group over. ``"error"`` raises, naming the groups. A group
whose total area is zero raises under every policy.
"""

from __future__ import annotations

import pandas as pd

from cil_regionalization.config import FallbackPolicy, Normalization
from cil_regionalization.schema import (
    GRID_CELLS,
    SourceUnits,
    method_column,
    raw_column,
    weight_column,
)


def _group_fields(
    id_fields: list[str],
    source_units: SourceUnits,
    normalization: Normalization,
) -> list[str]:
    """The columns raw totals are summed within, the ``id_fields`` under
    ``per_destination`` and the source unit keys under ``per_source``.
    Raises on any other direction."""
    if normalization == "per_destination":
        return list(id_fields)
    if normalization == "per_source":
        return list(source_units.key_columns)
    raise ValueError(f"unknown normalization: {normalization!r}")


def compute_native_weights(
    per_cell: pd.DataFrame,
    id_fields: list[str],
    weight_name: str,
    raw_col: str = "raw",
    *,
    normalization: Normalization,
    source_units: SourceUnits = GRID_CELLS,
) -> pd.DataFrame:
    """Each row's ``raw_col`` divided by its group's total, as the weight,
    raw and method columns of ``weight_name``, with every method
    ``"native"``. Used for the area weight, and has no fallback. Raises if
    a column is missing or a group's total is not positive.
    """
    keys = list(id_fields)
    source_cols = list(source_units.columns)
    required = set(keys + [raw_col] + source_cols)
    if not required.issubset(per_cell.columns):
        missing = sorted(required - set(per_cell.columns))
        raise ValueError(f"compute_native_weights: missing columns {missing}")

    group = _group_fields(keys, source_units, normalization)
    totals = per_cell.groupby(group, sort=False)[raw_col].sum().rename("_total")
    out = per_cell.merge(totals, on=group, how="left")
    bad = out.loc[out["_total"] <= 0, group].drop_duplicates()
    if len(bad) > 0:
        raise ValueError(
            f"weights.{weight_name}: zero {raw_col} total for groups "
            f"{bad.to_dict(orient='records')}; no fallback available for this weight"
        )

    out[weight_column(weight_name)] = out[raw_col] / out["_total"]
    out[raw_column(weight_name)] = out[raw_col]
    out[method_column(weight_name)] = "native"
    out = out.drop(columns=["_total"])
    cols = (
        keys
        + source_cols
        + [
            weight_column(weight_name),
            raw_column(weight_name),
            method_column(weight_name),
        ]
    )
    sort_keys = keys + list(source_units.key_columns)
    return out[cols].sort_values(sort_keys).reset_index(drop=True)


def apply_fallback(
    raw_per_cell: pd.DataFrame,
    area_weights: pd.DataFrame,
    weight_name: str,
    policy: FallbackPolicy,
    id_fields: list[str],
    raw_col: str = "raw",
    *,
    normalization: Normalization,
    source_units: SourceUnits = GRID_CELLS,
) -> pd.DataFrame:
    """Weights for ``weight_name``, each raw value divided by its group's
    total, with the groups whose total is zero handled by ``policy``.

    The output has one row per row of ``area_weights``, which holds the id
    fields, the source columns, ``areawt`` and ``area_raw``.
    ``raw_per_cell`` holds the id fields, the source key columns and
    ``raw_col``, and a row it leaves out counts as raw 0. Raises if a
    column is missing, if
    ``raw_per_cell`` has a group that ``area_weights`` lacks, if a group's
    total area is zero, or if a group falls back under ``"error"``.
    """
    keys = list(id_fields)
    source_cols = list(source_units.columns)
    source_keys = list(source_units.key_columns)
    if not set(keys + source_cols + ["areawt", "area_raw"]).issubset(
        area_weights.columns
    ):
        missing = sorted(
            set(keys + source_cols + ["areawt", "area_raw"]) - set(area_weights.columns)
        )
        raise ValueError(f"apply_fallback: area_weights missing columns {missing}")
    if not set(keys + source_keys + [raw_col]).issubset(raw_per_cell.columns):
        missing = sorted(
            set(keys + source_keys + [raw_col]) - set(raw_per_cell.columns)
        )
        raise ValueError(f"apply_fallback: raw_per_cell missing columns {missing}")

    join_keys = keys + source_keys
    group = _group_fields(keys, source_units, normalization)

    raw_group_set = {
        tuple(row)
        for row in raw_per_cell[group]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    area_group_set = {
        tuple(row)
        for row in area_weights[group]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    missing = sorted(raw_group_set - area_group_set)
    if missing:
        raise ValueError(
            f"weights.{weight_name}: raw frame references groups with zero total area "
            f"(no rows in area_weights): {missing}"
        )

    base = area_weights[keys + source_cols + ["areawt", "area_raw"]].copy()
    raw = raw_per_cell[join_keys + [raw_col]].rename(columns={raw_col: "_raw"})
    base = base.merge(raw, on=join_keys, how="left")
    base["_raw"] = base["_raw"].fillna(0.0)

    group_totals = base.groupby(group, sort=False).agg(
        _total_raw=("_raw", "sum"),
        _total_area=("area_raw", "sum"),
    )
    base = base.merge(group_totals, on=group, how="left")

    zero_area_groups = group_totals[group_totals["_total_area"] <= 0]
    if len(zero_area_groups) > 0:
        raise ValueError(
            f"weights.{weight_name}: groups have zero total area, cannot compute weights: "
            f"{zero_area_groups.index.tolist()}"
        )

    fallback_mask = base["_total_raw"] <= 0
    fallback_group_ids = base.loc[fallback_mask, group].drop_duplicates()

    if len(fallback_group_ids) > 0 and policy == "error":
        raise ValueError(
            f"weights.{weight_name}: zero raw total for groups "
            f"{fallback_group_ids.to_dict(orient='records')} and fallback='error'"
        )

    native = base.loc[~fallback_mask].copy()
    native["_wt"] = native["_raw"] / native["_total_raw"]
    native["_method"] = "native"

    fb = base.loc[fallback_mask].copy()
    if len(fb) > 0:
        if policy == "area":
            fb["_wt"] = fb["areawt"]
            fb["_method"] = "area_fallback"
        elif policy == "zero":
            fb["_wt"] = 0.0
            fb["_method"] = "zero"
        elif policy == "nan":
            fb["_wt"] = float("nan")
            fb["_method"] = "nan"
        else:  # pragma: no cover
            raise ValueError(f"unhandled fallback policy: {policy}")

    out = pd.concat([native, fb], ignore_index=True)
    out = out.rename(
        columns={
            "_wt": weight_column(weight_name),
            "_raw": raw_column(weight_name),
            "_method": method_column(weight_name),
        }
    )
    cols = (
        keys
        + source_cols
        + [
            weight_column(weight_name),
            raw_column(weight_name),
            method_column(weight_name),
        ]
    )
    return out[cols].sort_values(keys + source_keys).reset_index(drop=True)


def count_methods(
    weights_frame: pd.DataFrame,
    weight_name: str,
    id_fields: list[str] | None = None,
) -> dict[str, int]:
    """The number of groups of ``id_fields`` under each method label of
    ``weight_name``, taken from each group's first row. Returns an empty
    dict when the frame has no method column for the weight.

    Without ``id_fields`` every column counts as an id except the four cell
    columns and those ending in ``wt``, ``_raw`` or ``_method``. That is
    wrong for a frame with any other column, such as the source ids of
    polygon mode, so pass ``id_fields`` for such a frame.
    """
    col = method_column(weight_name)
    if col not in weights_frame.columns:
        return {}
    if id_fields is None:
        id_fields = _infer_id_columns(weights_frame, weight_name)
    method_per_region = weights_frame.groupby(list(id_fields), sort=False)[col].first()
    return method_per_region.value_counts().to_dict()


def _infer_id_columns(weights_frame: pd.DataFrame, weight_name: str) -> list[str]:
    """A guess at the id columns of a weights frame, every column except
    ``cell_ix``, ``cell_iy``, ``cell_lon``, ``cell_lat`` and those ending in
    ``wt``, ``_raw`` or ``_method``, whichever weight they belong to. Right
    only for a frame holding nothing else. ``weight_name`` is not used.
    """
    output_cols = {"cell_ix", "cell_iy", "cell_lon", "cell_lat"}
    for c in weights_frame.columns:
        if c.endswith("wt") or c.endswith("_raw") or c.endswith("_method"):
            output_cols.add(c)
    return [c for c in weights_frame.columns if c not in output_cols]
