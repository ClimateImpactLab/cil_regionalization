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
The artifact records the vintage of the geometry its source units come
from (``source_version``). The caller must state the vintage the data was
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from cil_regionalization.backends.base import WeightsResult
from cil_regionalization.config import SourceUnitPolicies
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
    enforce_source_policies,
)


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
                f"predates application support. Regenerate the weights, or "
                f"build a WeightsArtifact from a live WeightsResult."
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

    ``data_version`` states the geometry vintage the data was built
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
            "applied. Grid-mode artifacts do not carry a source vintage yet."
        )
    if data_version is None:
        raise ValueError(
            "data_version is None: the data carries no geometry version "
            "information. State the vintage the data was built against "
            f"(the artifact records {artifact.source_version!r}); do not "
            "assume compatibility."
        )
    if data_version != artifact.source_version:
        raise ValueError(
            f"geometry version mismatch: the weights artifact was built "
            f"against source version {artifact.source_version!r} but the "
            f"data states {data_version!r}. Rebuild the weights against the "
            f"data's vintage or use matching data."
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
