"""Apply per-weight fallback policy to per-(region, source unit) raw totals.

Pure pandas. Given the raw weighting totals a backend has computed for one
non-area weight, plus the area-weight frame (which is always available),
this module returns the normalized weights and a per-group method label
that tells callers and validators what happened.

Source units and normalization direction
----------------------------------------
Both functions accept a `SourceUnits` layout (grid cells by default) and a
normalization direction. The direction selects the columns raw totals are
summed within before dividing:

    per_destination : totals per region (the ``id_fields``). Weights sum
        to 1 within each region. The grid-to-IR case.
    per_source : totals per source unit, across the regions it overlaps.
        Weights sum to 1 within each source unit. The allocation case.

The math is the same division either way; only the grouping transposes.
Fallback policy applies to the groups of the chosen direction: a group
whose total raw weight is zero is routed through the policy below.

Policy semantics
----------------
For a group whose total raw weight is zero:
    "area"  : take the rows from the area frame and reuse `areawt` as the
              normalized weight; method = "area_fallback".
    "zero"  : keep the area frame's rows, weight = 0; method = "zero".
              The sum-to-1 validator will flag this group.
    "nan"   : keep the area frame's rows, weight = NaN; method = "nan".
              The sum-to-1 validator IGNORES NaN groups (no false flag).
              Matches the legacy crop weights convention: empty regions
              carry NaN; the legacy consumer backfills per-pixel with
              `areawt` at aggregation time so NaN in the file is not NaN
              in final outputs.
    "error" : raise, naming the group ids.
In every fallback branch, if the group's total area is also zero we
raise unconditionally: no rule can rescue an empty group.

Implicit-default warning
------------------------
When the fallback fires for >= max(5, 5% of groups) AND the policy was
NOT set explicitly in the TOML, `apply_fallback` emits a WARN-level
log naming the applied default and the alternatives. The threshold is
deliberate: a single fallback group is rarely worth interrupting for;
a dozen is.
"""
from __future__ import annotations

import logging

import pandas as pd

from cil_regionalization.config import FallbackPolicy, Normalization
from cil_regionalization.schema import (
    GRID_CELLS,
    SourceUnits,
    method_column,
    raw_column,
    weight_column,
)


logger = logging.getLogger(__name__)


_IMPLICIT_DEFAULT_FRAC = 0.05
_IMPLICIT_DEFAULT_MIN = 5


def _group_fields(
    id_fields: list[str],
    source_units: SourceUnits,
    normalization: Normalization,
) -> list[str]:
    """Columns raw totals are summed within, per the direction."""
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
    source_units: SourceUnits = GRID_CELLS,
    normalization: Normalization = "per_destination",
) -> pd.DataFrame:
    """Normalize raw values to sum-to-1 weights per group. No fallback path.

    Used for the area weight, whose zero-total case is always degenerate and
    raises here regardless of any configured policy.
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
    cols = keys + source_cols + [
        weight_column(weight_name),
        raw_column(weight_name),
        method_column(weight_name),
    ]
    sort_keys = keys + list(source_units.key_columns)
    return (
        out[cols]
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )


def apply_fallback(
    raw_per_cell: pd.DataFrame,
    area_weights: pd.DataFrame,
    weight_name: str,
    policy: FallbackPolicy,
    id_fields: list[str],
    raw_col: str = "raw",
    *,
    policy_explicit: bool = True,
    source_units: SourceUnits = GRID_CELLS,
    normalization: Normalization = "per_destination",
) -> pd.DataFrame:
    """Normalize raw values per group; route zero-total groups through policy.

    raw_per_cell : id_fields + source key columns + raw_col. May omit rows
        whose raw is zero; missing rows are treated as raw = 0.
    area_weights : id_fields + source columns + areawt + area_raw. The spine
        of "source units overlapping each region"; the output covers exactly
        the union of rows in this frame.
    """
    keys = list(id_fields)
    source_cols = list(source_units.columns)
    source_keys = list(source_units.key_columns)
    if not set(keys + source_cols + ["areawt", "area_raw"]).issubset(area_weights.columns):
        missing = sorted(set(keys + source_cols + ["areawt", "area_raw"]) - set(area_weights.columns))
        raise ValueError(f"apply_fallback: area_weights missing columns {missing}")
    if not set(keys + source_keys + [raw_col]).issubset(raw_per_cell.columns):
        missing = sorted(set(keys + source_keys + [raw_col]) - set(raw_per_cell.columns))
        raise ValueError(f"apply_fallback: raw_per_cell missing columns {missing}")

    join_keys = keys + source_keys
    group = _group_fields(keys, source_units, normalization)

    raw_group_set = {
        tuple(row)
        for row in raw_per_cell[group].drop_duplicates().itertuples(index=False, name=None)
    }
    area_group_set = {
        tuple(row)
        for row in area_weights[group].drop_duplicates().itertuples(index=False, name=None)
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

    n_fallback_groups = len(fallback_group_ids)
    n_total_groups = group_totals.shape[0]
    if (
        n_fallback_groups > 0
        and not policy_explicit
        and n_fallback_groups
        >= max(_IMPLICIT_DEFAULT_MIN, int(_IMPLICIT_DEFAULT_FRAC * n_total_groups))
    ):
        logger.warning(
            "weights.%s: fallback policy %r fired for %d of %d groups "
            "(%.1f%%); this is the library default. Alternatives: "
            "[area | zero | error | nan]. Set weights.%s.fallback "
            "explicitly in the config to silence this warning.",
            weight_name,
            policy,
            n_fallback_groups,
            n_total_groups,
            100.0 * n_fallback_groups / max(1, n_total_groups),
            weight_name,
        )

    out = pd.concat([native, fb], ignore_index=True)
    out = out.rename(
        columns={
            "_wt": weight_column(weight_name),
            "_raw": raw_column(weight_name),
            "_method": method_column(weight_name),
        }
    )
    cols = keys + source_cols + [
        weight_column(weight_name),
        raw_column(weight_name),
        method_column(weight_name),
    ]
    return (
        out[cols]
        .sort_values(keys + source_keys)
        .reset_index(drop=True)
    )


def count_methods(
    weights_frame: pd.DataFrame,
    weight_name: str,
    id_fields: list[str] | None = None,
) -> dict[str, int]:
    """Return a method label -> region count for one weight in the output frame.

    Pass ``id_fields`` explicitly whenever the frame may contain other
    weights' columns (their `_raw` / `_method` columns would otherwise be
    treated as identifiers and inflate the region count).
    """
    col = method_column(weight_name)
    if col not in weights_frame.columns:
        return {}
    if id_fields is None:
        id_fields = _infer_id_columns(weights_frame, weight_name)
    method_per_region = (
        weights_frame.groupby(list(id_fields), sort=False)[col].first()
    )
    return method_per_region.value_counts().to_dict()


def _infer_id_columns(weights_frame: pd.DataFrame, weight_name: str) -> list[str]:
    """Best-effort: drop cell metadata and any column that looks like weight
    output (`{name}wt`, `{name}_raw`, `{name}_method`). Safe for frames that
    contain only one weight's columns alongside the id fields.
    """
    output_cols = {"cell_ix", "cell_iy", "cell_lon", "cell_lat"}
    for c in weights_frame.columns:
        if c.endswith("wt") or c.endswith("_raw") or c.endswith("_method"):
            output_cols.add(c)
    return [c for c in weights_frame.columns if c not in output_cols]
