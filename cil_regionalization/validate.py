"""Checks on a weights frame and on an application run.

`check_sum_to_one` checks that every weight sums to 1 within each
normalization group, a region under per_destination and a source unit
under per_source. `check_grid_invariants` and `check_polygon_invariants`
check the keys and cell coordinates, which sum-to-1 cannot see.
`check_mass_balance` checks that an aggregation conserved its quantity,
and `check_source_coverage` with `enforce_source_policies` accounts for
the source units that could not take part in a run. The ``check_``
functions return their failures in a report rather than raising them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from cil_regionalization.config import SourceUnitPolicies
from cil_regionalization.grid import GridSpec
from cil_regionalization.manifest import Manifest
from cil_regionalization.schema import OutputSchema, weight_column


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SumToOneReport:
    """Result of `check_sum_to_one`. `failures` is empty when everything is OK.

    ``n_regions`` is the number of groups the weights were summed within,
    regions under per_destination normalization and source units under
    per_source. ``failures``
    carries the group columns plus 'weight', 'sum' and 'deviation'.
    """

    weight_names: tuple[str, ...]
    n_regions: int
    failures: pd.DataFrame
    tolerance: float

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def summary(self) -> str:
        if self.ok:
            return (
                f"sum-to-1 check passed: {self.n_regions} regions, "
                f"weights={list(self.weight_names)}, tolerance={self.tolerance}"
            )
        return (
            f"sum-to-1 check FAILED: {len(self.failures)} (region, weight) pairs "
            f"out of tolerance ({self.tolerance}); top deviations:\n"
            f"{self.failures.nlargest(5, 'deviation').to_string(index=False)}"
        )


def check_sum_to_one(
    weights_frame: pd.DataFrame,
    schema: OutputSchema,
    tolerance: float = 1e-6,
) -> SumToOneReport:
    """The SumToOneReport of `weights_frame`, each weight in
    `schema.weight_names` summed within each normalization group of
    `schema` and compared with 1 within `tolerance`.

    A group whose weights sum to 0, as the `zero` fallback leaves them,
    fails. A group whose weights are all NaN, as the `nan` fallback
    leaves them, passes. Raises if the frame has no column for a weight.
    """
    # An all-NaN group sums to NaN under min_count=1, and NaN > tolerance
    # is False, so it passes. Consumers fill NaN weights themselves, as
    # legacy crop weights are filled with `areawt` at aggregation time.
    group_fields = list(schema.normalization_group)
    region_count = weights_frame[group_fields].drop_duplicates().shape[0]
    rows: list[pd.DataFrame] = []
    for name in schema.weight_names:
        col = weight_column(name)
        if col not in weights_frame.columns:
            raise ValueError(f"check_sum_to_one: weights frame missing column {col!r}")
        sums = (
            weights_frame.groupby(group_fields, sort=False)[col]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={col: "sum"})
        )
        sums["weight"] = name
        sums["deviation"] = (sums["sum"] - 1.0).abs()
        bad = sums.loc[sums["deviation"] > tolerance].copy()
        if len(bad) > 0:
            rows.append(bad[group_fields + ["weight", "sum", "deviation"]])

    if rows:
        failures = pd.concat(rows, ignore_index=True)
    else:
        failures = pd.DataFrame(columns=group_fields + ["weight", "sum", "deviation"])
    return SumToOneReport(
        weight_names=tuple(schema.weight_names),
        n_regions=region_count,
        failures=failures,
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class GridInvariantsReport:
    """Result of `check_grid_invariants`. `failures` empty means OK."""

    n_rows: int
    failures: (
        pd.DataFrame
    )  # cols: id_fields + cell_ix + cell_iy + cell_lon + cell_lat + _invariant
    n_ix: int
    n_iy: int
    lon_domain: tuple[float, float]

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def summary(self) -> str:
        if self.ok:
            return (
                f"grid invariants OK: {self.n_rows} rows, "
                f"n_ix={self.n_ix}, n_iy={self.n_iy}, lon{self.lon_domain}"
            )
        counts = self.failures["_invariant"].value_counts().to_dict()
        return (
            f"grid invariants FAILED: {len(self.failures)} rows violate "
            f"invariants {counts}; top offenders:\n"
            f"{self.failures.head(10).to_string(index=False)}"
        )


def check_grid_invariants(
    weights_frame: pd.DataFrame,
    schema: OutputSchema,
    grid: GridSpec,
) -> GridInvariantsReport:
    """The `GridInvariantsReport` of `weights_frame` on `grid`. Its
    ``failures`` hold each row with a cell index out of range, a centroid
    outside the grid's domain or a duplicated ``(region, cell_ix,
    cell_iy)`` key, labelled in the column ``_invariant``. The backends
    raise on any failure.
    """
    id_fields = list(schema.id_fields)
    rows: list[pd.DataFrame] = []

    def _flag(mask: pd.Series, label: str) -> None:
        if not mask.any():
            return
        sub = weights_frame.loc[
            mask, id_fields + ["cell_ix", "cell_iy", "cell_lon", "cell_lat"]
        ].copy()
        sub["_invariant"] = label
        rows.append(sub)

    n_ix = grid.n_ix
    n_iy = grid.n_iy
    _flag(
        (weights_frame["cell_ix"] < 0) | (weights_frame["cell_ix"] >= n_ix),
        "cell_ix_out_of_range",
    )
    _flag(
        (weights_frame["cell_iy"] < 0) | (weights_frame["cell_iy"] >= n_iy),
        "cell_iy_out_of_range",
    )
    lon_lo, lon_hi = grid.domain_lon
    _flag(
        (weights_frame["cell_lon"] < lon_lo) | (weights_frame["cell_lon"] >= lon_hi),
        "cell_lon_out_of_range",
    )
    lat_lo, lat_hi = grid.domain_lat
    _flag(
        (weights_frame["cell_lat"] < lat_lo) | (weights_frame["cell_lat"] >= lat_hi),
        "cell_lat_out_of_range",
    )
    # Duplicate (region, cell_ix, cell_iy) keys. The antimeridian wrap
    # collision (ATA's ix=0 + ix=360) would land here if it survived.
    key = id_fields + ["cell_ix", "cell_iy"]
    dup_mask = weights_frame.duplicated(subset=key, keep=False)
    _flag(dup_mask, "duplicate_key")

    if rows:
        failures = pd.concat(rows, ignore_index=True)
    else:
        failures = pd.DataFrame(
            columns=id_fields
            + ["cell_ix", "cell_iy", "cell_lon", "cell_lat", "_invariant"]
        )
    return GridInvariantsReport(
        n_rows=len(weights_frame),
        failures=failures,
        n_ix=n_ix,
        n_iy=n_iy,
        lon_domain=(float(lon_lo), float(lon_hi)),
    )


@dataclass(frozen=True)
class PolygonInvariantsReport:
    """Result of `check_polygon_invariants`, ``ok`` when it found nothing.

    ``failures`` holds the rows with a null key, a duplicated key or an
    unknown source unit. ``missing_source_units`` lists the expected
    source units that have no row at all. ``sum_report`` is the sum-to-1
    result over the same frame.
    """

    n_rows: int
    failures: pd.DataFrame  # cols: id_fields + source key cols + '_invariant'
    missing_source_units: tuple[tuple, ...]
    sum_report: SumToOneReport

    @property
    def ok(self) -> bool:
        return (
            len(self.failures) == 0
            and len(self.missing_source_units) == 0
            and self.sum_report.ok
        )

    def summary(self) -> str:
        if self.ok:
            return (
                f"polygon invariants OK: {self.n_rows} rows; "
                f"{self.sum_report.summary()}"
            )
        parts: list[str] = []
        if len(self.failures) > 0:
            counts = self.failures["_invariant"].value_counts().to_dict()
            parts.append(
                f"{len(self.failures)} rows violate invariants {counts}; "
                f"top offenders:\n{self.failures.head(10).to_string(index=False)}"
            )
        if self.missing_source_units:
            shown = list(self.missing_source_units[:10])
            parts.append(
                f"{len(self.missing_source_units)} expected source units "
                f"have no row: {shown}"
                + (" ..." if len(self.missing_source_units) > 10 else "")
            )
        if not self.sum_report.ok:
            parts.append(self.sum_report.summary())
        return "polygon invariants FAILED: " + "; ".join(parts)


def check_polygon_invariants(
    weights_frame: pd.DataFrame,
    schema: OutputSchema,
    expected_source_ids: Iterable[tuple] | None = None,
    tolerance: float = 1e-6,
) -> PolygonInvariantsReport:
    """The `PolygonInvariantsReport` of a polygon-mode weights frame, for
    either normalization direction.

    Its ``failures`` hold the rows with a null id field or source key,
    labelled ``null_key``, the rows of a duplicated (id fields, source
    unit) pair, labelled ``duplicate_key``, and the rows of a source unit
    not in ``expected_source_ids``, labelled ``unknown_source_unit``. The
    expected source units with no row are listed in
    ``missing_source_units``, and the weights are checked to sum to 1
    with `check_sum_to_one`.

    ``expected_source_ids`` is the source-unit universe as id tuples in
    the order of ``schema.source_units.key_columns``. None skips the two
    checks against it.
    """
    id_fields = list(schema.id_fields)
    source_keys = list(schema.source_units.key_columns)
    key_cols = id_fields + source_keys
    rows: list[pd.DataFrame] = []

    def _flag(mask: pd.Series, label: str) -> None:
        if not mask.any():
            return
        sub = weights_frame.loc[mask, key_cols].copy()
        sub["_invariant"] = label
        rows.append(sub)

    null_mask = weights_frame[key_cols].isna().any(axis=1)
    _flag(null_mask, "null_key")
    _flag(weights_frame.duplicated(subset=key_cols, keep=False), "duplicate_key")

    missing: tuple[tuple, ...] = ()
    if expected_source_ids is not None:
        expected = {tuple(t) for t in expected_source_ids}
        present = {
            tuple(row)
            for row in weights_frame[source_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        unknown = present - expected
        if unknown:
            unknown_mask = weights_frame[source_keys].apply(tuple, axis=1).isin(unknown)
            _flag(unknown_mask, "unknown_source_unit")
        missing = tuple(sorted(expected - present))

    if rows:
        failures = pd.concat(rows, ignore_index=True)
    else:
        failures = pd.DataFrame(columns=key_cols + ["_invariant"])

    sum_report = check_sum_to_one(weights_frame, schema, tolerance=tolerance)
    return PolygonInvariantsReport(
        n_rows=len(weights_frame),
        failures=failures,
        missing_source_units=missing,
        sum_report=sum_report,
    )


@dataclass(frozen=True)
class MassBalanceReport:
    """Result of `check_mass_balance`, ``ok`` when the totals and every
    group agree within ``tolerance``.

    ``source_total`` and ``target_total`` are the global sums, and
    ``rel_error`` is their relative difference. ``failures`` holds one row
    per group out of tolerance, with the columns group, source_sum,
    target_sum, abs_error and rel_error. It is empty when no grouping was
    requested or every group balances.
    """

    source_total: float
    target_total: float
    rel_error: float
    tolerance: float
    failures: pd.DataFrame

    @property
    def ok(self) -> bool:
        return self.rel_error <= self.tolerance and len(self.failures) == 0

    def summary(self) -> str:
        if self.ok:
            return (
                f"mass balance OK: source {self.source_total:.6g} == target "
                f"{self.target_total:.6g} (rel {self.rel_error:.2e}, "
                f"tolerance {self.tolerance:.2e})"
            )
        lines = [
            f"mass balance FAILED: source {self.source_total:.6g} vs target "
            f"{self.target_total:.6g} (rel {self.rel_error:.2e}, tolerance "
            f"{self.tolerance:.2e})"
        ]
        if len(self.failures) > 0:
            worst = self.failures.nlargest(5, "rel_error")
            lines.append(
                f"{len(self.failures)} groups out of tolerance; worst:\n"
                f"{worst.to_string(index=False)}"
            )
        return "\n".join(lines)


def _rel_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def check_mass_balance(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    value_col: str,
    group_col: str | None = None,
    tolerance: float = 1e-6,
) -> MassBalanceReport:
    """The `MassBalanceReport` of an extensive quantity aggregated from
    ``source`` to ``target``.

    Each frame has one row per unit with the quantity in ``value_col``
    and, when ``group_col`` is given, a group label such as the country.
    The global sums and each group's sums are compared with the relative
    error ``|source - target| / max(|source|, |target|)``. A group on
    only one side is compared with 0, so a dropped group fails in full.
    NaN values are left out of the sums. Raises if a frame lacks
    ``value_col`` or ``group_col``.

    A pass says the quantity was conserved globally and within every
    group, not that it went to the right targets within a group. An
    intensive variable, whose weighted mean is not conserved, must not be
    checked this way.
    """
    for name, df in (("source", source), ("target", target)):
        if value_col not in df.columns:
            raise ValueError(
                f"check_mass_balance: {name} frame missing value column {value_col!r}"
            )
        if group_col is not None and group_col not in df.columns:
            raise ValueError(
                f"check_mass_balance: {name} frame missing group column {group_col!r}"
            )

    source_total = float(source[value_col].sum())
    target_total = float(target[value_col].sum())
    rel = _rel_error(source_total, target_total)

    if group_col is None:
        failures = pd.DataFrame(
            columns=["group", "source_sum", "target_sum", "abs_error", "rel_error"]
        )
    else:
        s = source.groupby(group_col, sort=False)[value_col].sum().rename("source_sum")
        t = target.groupby(group_col, sort=False)[value_col].sum().rename("target_sum")
        merged = pd.concat([s, t], axis=1).fillna(0.0).reset_index()
        merged = merged.rename(columns={group_col: "group"})
        merged["abs_error"] = (merged["source_sum"] - merged["target_sum"]).abs()
        merged["rel_error"] = [
            _rel_error(a, b) for a, b in zip(merged["source_sum"], merged["target_sum"])
        ]
        failures = merged.loc[merged["rel_error"] > tolerance].reset_index(drop=True)

    return MassBalanceReport(
        source_total=source_total,
        target_total=target_total,
        rel_error=rel,
        tolerance=tolerance,
        failures=failures,
    )


@dataclass(frozen=True)
class SourceCoverageReport:
    """Which source units cannot take part in an application run, and why.

    Id tuples follow the weight schema's source key column order.
    ``unmatched`` units are in the data being aggregated and not in the
    weights frame. ``zero_weight`` units are in the weights frame with
    every weight zero or NaN, so they carry no mass to any target.
    ``absent_from_data`` units are in the weights frame and not in the
    data.
    """

    unmatched: tuple[tuple, ...]
    zero_weight: tuple[tuple, ...]
    absent_from_data: tuple[tuple, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "unmatched": len(self.unmatched),
            "zero_weight": len(self.zero_weight),
            "absent_from_data": len(self.absent_from_data),
        }

    @property
    def clean(self) -> bool:
        return not (self.unmatched or self.zero_weight or self.absent_from_data)


def check_source_coverage(
    weights_frame: pd.DataFrame,
    schema: OutputSchema,
    weight_name: str,
    data_source_ids: Iterable[tuple],
) -> SourceCoverageReport:
    """The `SourceCoverageReport` of ``weights_frame`` against the source
    units of the data, for the weight ``weight_name``.

    ``data_source_ids`` holds the id tuples of the source units in the
    data being aggregated, in the order of
    ``schema.source_units.key_columns``. A zero-weight unit that is also
    absent from the data is listed under both cases. Raises if the frame
    has no column ``{weight_name}wt``.
    """
    source_keys = list(schema.source_units.key_columns)
    col = weight_column(weight_name)
    if col not in weights_frame.columns:
        raise ValueError(f"check_source_coverage: weights frame missing column {col!r}")

    data_ids = {tuple(t) for t in data_source_ids}
    mass_per_unit = (
        weights_frame.assign(_abs=weights_frame[col].abs())
        .groupby(source_keys, sort=False)["_abs"]
        .sum(min_count=1)
    )
    weight_ids = {t if isinstance(t, tuple) else (t,) for t in mass_per_unit.index}
    zero_ids = {
        t if isinstance(t, tuple) else (t,)
        for t, total in mass_per_unit.items()
        if pd.isna(total) or total == 0.0
    }

    return SourceCoverageReport(
        unmatched=tuple(sorted(data_ids - weight_ids)),
        zero_weight=tuple(sorted(zero_ids)),
        absent_from_data=tuple(sorted(weight_ids - data_ids)),
    )


def enforce_source_policies(
    coverage: SourceCoverageReport,
    policies: SourceUnitPolicies,
    manifest: Manifest | None = None,
) -> None:
    """Raises if a coverage case that has units is under the policy
    "error", naming each such case, its count, its first ids and the
    policy that would skip it. A case under "skip" is logged as a
    warning. When ``manifest`` is given, the count and ids of every case
    are recorded under ``manifest.source_coverage`` before anything is
    raised.
    """
    cases: list[tuple[str, str, tuple[tuple, ...], str]] = [
        ("unmatched", policies.on_unmatched, coverage.unmatched, "on_unmatched"),
        (
            "zero_weight",
            policies.on_zero_weight,
            coverage.zero_weight,
            "on_zero_weight",
        ),
        (
            "absent_from_data",
            policies.on_absent_from_data,
            coverage.absent_from_data,
            "on_absent_from_data",
        ),
    ]

    if manifest is not None:
        for case, _, ids, _ in cases:
            manifest.source_coverage[case] = {
                "count": len(ids),
                "ids": [list(t) for t in ids],
            }

    # Remedies name only what a caller can reach from the package root:
    # restrict_to_sources and policies are apply_weights parameters, and
    # SourceUnitPolicies is exported.
    hints = {
        "unmatched": (
            "these units are in the data but not in the weights: check the "
            "id column and data_version, or pass "
            "policies=SourceUnitPolicies(on_unmatched='skip') to record "
            "and skip them"
        ),
        "zero_weight": (
            "these units carry no weight for the chosen weighting: pass "
            "policies=SourceUnitPolicies(on_zero_weight='skip') to record "
            "and skip them"
        ),
        "absent_from_data": (
            "the weights know these units but the data does not cover "
            "them: if the data intentionally covers a subset, pass "
            "restrict_to_sources with the units it does cover; if gaps "
            "are expected, pass "
            "policies=SourceUnitPolicies(on_absent_from_data='skip')"
        ),
    }
    errors: list[str] = []
    for case, policy, ids, key in cases:
        if not ids:
            continue
        shown = [list(t) for t in ids[:10]]
        message = (
            f"source coverage: {len(ids)} {case} source units "
            f"(first {min(len(ids), 10)}: {shown})"
        )
        if policy == "error":
            errors.append(message + "; " + hints[case])
        elif manifest is not None:
            logger.warning("%s; recorded in the manifest and skipped", message)
        else:
            logger.warning(
                "%s; skipped, and listed in the coverage of the result", message
            )
    if errors:
        raise ValueError("; ".join(errors))
