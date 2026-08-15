"""Sum-to-1 validation + grid invariants of the output weights frame.

Both backends produce the same output schema. This module checks that
every (group, weight) pair sums to 1 within the configured tolerance,
where the group follows the schema's normalization direction (regions
under per_destination, source units under per_source),
AND that the cell coordinates respect the declared grid (no indices out
of range, no centroids outside the convention's domain, no duplicate
``(region, cell_ix, cell_iy)`` keys). The s51 antimeridian bug
demonstrated that sum-to-1 alone cannot catch misallocation: the wrap-
unaware SQL produced rows that summed to 1 per region per weight while
silently routing populated cells away from their points.

These checks return structured reports rather than asserting; the
backends raise on a non-empty failures frame in their `compute()`
methods.

Beyond the weights frame itself, this module also validates application
runs: `check_polygon_invariants` is the polygon-mode analogue of the grid
invariants, `check_mass_balance` proves an aggregation conserved the
quantity it aggregated, and `check_source_coverage` +
`enforce_source_policies` account for every source unit that could not
flow through a run.
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

    ``n_regions`` counts the groups the weights were summed within: regions
    under per_destination normalization, source units under per_source.
    ``failures`` carries the group columns plus 'weight', 'sum', 'deviation'.
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
    """Return a SumToOneReport for `weights_frame` against `schema`.

    For each weight in `schema.weight_names`, group by the schema's
    normalization group (id fields under per_destination, source-unit key
    columns under per_source) and verify the weight column sums to 1
    within `tolerance`. Groups whose weights sum to 0 (the `zero` fallback
    policy outcome) are reported as failures, by design. Groups whose
    weights are entirely NaN (the `nan` fallback policy outcome) are NOT
    flagged: `sum(min_count=1)` returns NaN for such groups and
    `NaN > tolerance` is False, so the validator passes them through
    silently. Downstream consumers handle NaN explicitly (legacy crop
    weights backfill with `areawt` at aggregation time).
    """
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
        failures = pd.DataFrame(
            columns=group_fields + ["weight", "sum", "deviation"]
        )
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
    failures: pd.DataFrame  # cols: id_fields + cell_ix + cell_iy + cell_lon + cell_lat + _invariant
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
    """Enforce cell-coordinate invariants on the output.

    Catches what sum-to-1 cannot: indices out of declared range,
    centroids outside the convention's domain, duplicate
    ``(region, cell_ix, cell_iy)`` keys (the antimeridian wrap collision
    from the legacy SQL). Returns a structured report; the backends call
    this in `compute()` and raise on any failure.
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
    """Result of `check_polygon_invariants`. Everything empty means OK.

    ``failures`` flags rows (null keys, duplicate keys, unknown source
    units); ``missing_source_units`` lists expected source units with no
    row at all, which cannot be flagged row-wise; ``sum_report`` is the
    direction-aware sum-to-1 result over the same frame.
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
    """Enforce key and coverage invariants on a polygon-mode weights frame.

    The polygon analogue of `check_grid_invariants`, valid for either
    normalization direction:

    - null values in id fields or source key columns (``null_key``),
    - duplicate (id fields, source unit) pairs, which would double count
      the pair's overlap (``duplicate_key``),
    - source units in the frame that are not in ``expected_source_ids``
      (``unknown_source_unit``, e.g. a stale crosswalk row),
    - expected source units with no row at all (reported in
      ``missing_source_units``; a source unit the weights silently lost),
    - sum-to-1 per the schema's normalization group, via
      `check_sum_to_one`.

    ``expected_source_ids`` is the source-unit universe as an iterable of
    id tuples (matching ``schema.source_units.key_columns`` order). Pass
    None to skip the coverage checks when no universe is defined.
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
            unknown_mask = weights_frame[source_keys].apply(
                tuple, axis=1
            ).isin(unknown)
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
    """Result of `check_mass_balance`. `ok` means totals agree within tolerance.

    ``source_total`` / ``target_total`` are the global sums;
    ``failures`` holds one row per group out of tolerance (columns:
    group, source_sum, target_sum, abs_error, rel_error), empty when no
    grouping was requested or every group balances.
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
    """Check that aggregation conserved an extensive quantity.

    ``source`` and ``target`` each carry one row per unit with the
    quantity in ``value_col`` and, when ``group_col`` is given, a group
    label (typically the country). The check compares the global sums and
    each group's sums with relative tolerance
    ``|source - target| / max(|source|, |target|)``. A group present on
    only one side is compared against 0 on the other, so a wholly dropped
    group is a full-size failure, not a silent absence.

    What a pass proves: the run moved the entire quantity from source
    units to target units, globally and within every group. This is the
    conservation property an extensive aggregation (a sum or a
    per-source allocation) must have, and it catches dropped units,
    zeroed rows, and double counting across groups.

    What a pass does not prove: that the quantity was allocated to the
    *right* targets within a group (any reshuffle inside a group
    conserves the group sum); anything about intensive variables, whose
    weighted means are not conserved quantities and must not be run
    through this check; and losses that cancel between units inside one
    group at the group level are only caught globally if they do not
    cancel there too. NaN values are excluded from the sums (pandas
    semantics), so a side that turned values into NaN shows up as a
    deficit, but a side that was NaN in the source to begin with is not
    counted as mass to conserve.
    """
    for name, df in (("source", source), ("target", target)):
        if value_col not in df.columns:
            raise ValueError(
                f"check_mass_balance: {name} frame missing value column "
                f"{value_col!r}"
            )
        if group_col is not None and group_col not in df.columns:
            raise ValueError(
                f"check_mass_balance: {name} frame missing group column "
                f"{group_col!r}"
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
            _rel_error(a, b)
            for a, b in zip(merged["source_sum"], merged["target_sum"])
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
    """Which source units cannot flow through an application run, and why.

    Id tuples follow the weight schema's source key column order.

        unmatched        : in the data being aggregated, absent from the
                           weights frame.
        zero_weight      : in the weights frame, but every weight value is
                           zero or NaN, so they carry no mass to any target.
        absent_from_data : in the weights frame, absent from the data.
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
    """Partition source units into the three coverage cases.

    ``data_source_ids`` is the set of source-unit id tuples present in the
    data being aggregated (tuple order matching
    ``schema.source_units.key_columns``). The weight column checked for
    zero mass is ``{weight_name}wt``.
    """
    source_keys = list(schema.source_units.key_columns)
    col = weight_column(weight_name)
    if col not in weights_frame.columns:
        raise ValueError(
            f"check_source_coverage: weights frame missing column {col!r}"
        )

    data_ids = {tuple(t) for t in data_source_ids}
    mass_per_unit = (
        weights_frame.assign(_abs=weights_frame[col].abs())
        .groupby(source_keys, sort=False)["_abs"]
        .sum(min_count=1)
    )
    weight_ids = {
        t if isinstance(t, tuple) else (t,) for t in mass_per_unit.index
    }
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
    """Apply the configured policy to each coverage case.

    For every case: "error" raises, naming the case, the config key, the
    count, and the first offending ids; "skip" logs a warning and
    continues. When a manifest is given, all three counts and id lists
    are recorded under ``manifest.source_coverage`` before any error is
    raised, so even an aborted run documents what it found. Silent
    dropping is not an option this function offers.
    """
    cases: list[tuple[str, str, tuple[tuple, ...], str]] = [
        ("unmatched", policies.on_unmatched, coverage.unmatched, "on_unmatched"),
        ("zero_weight", policies.on_zero_weight, coverage.zero_weight, "on_zero_weight"),
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
        else:
            logger.warning("%s; recorded in the manifest and skipped", message)
    if errors:
        raise ValueError("; ".join(errors))
