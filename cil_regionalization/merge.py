"""Two-source weights merge utility.

The s51 deliverable is split across two runs: the canonical BigQuery run
(24,361 valid hierids from the IR table) and the shapefile-source
supplement run (the 17 NULL-in-IR hierids from the source shapefile).
Both produce the same `OutputSchema`. This module combines them with
hard invariants:

- inputs' hierid sets are disjoint (a hierid must come from exactly one
  source),
- combined unique hierid count == ``expected_total`` (24,378),
- the merged frame still satisfies sum-to-1 per region per weight,
- a ``source`` column is added (values: ``"bigquery"`` /
  ``"bigquery_shapefile_supplement"``),
- the merged manifest records both source manifests' config hashes,
  input identifiers, and per-source row counts so the merge is
  reproducible.

Used by `examples/s51/merge.py`; the function below is the unit-testable
core.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cil_regionalization.schema import OutputSchema
from cil_regionalization.validate import check_sum_to_one


SOURCE_BIGQUERY = "bigquery"
SOURCE_SHAPEFILE_SUPPLEMENT = "bigquery_shapefile_supplement"


@dataclass(frozen=True)
class MergeResult:
    frame: pd.DataFrame
    schema: OutputSchema
    merged_manifest: dict
    n_canonical: int  # rows from the BQ source
    n_supplement: int  # rows from the shapefile-source supplement
    n_regions: int  # unique hierids in the combined frame


def merge_weights(
    canonical_frame: pd.DataFrame,
    canonical_manifest: dict,
    supplement_frame: pd.DataFrame,
    supplement_manifest: dict,
    *,
    schema: OutputSchema,
    expected_total: int,
    tolerance: float = 1e-6,
) -> MergeResult:
    """Combine the canonical BQ run with the shapefile-source supplement.

    Hard invariants enforced; any failure raises with a precise message.
    """
    id_fields = list(schema.id_fields)
    primary = id_fields[0]

    canonical_ids = set(canonical_frame[primary].astype(str))
    supplement_ids = set(supplement_frame[primary].astype(str))
    overlap = canonical_ids & supplement_ids
    if overlap:
        raise ValueError(
            f"merge: canonical and supplement hierid sets overlap on "
            f"{len(overlap)} ids: {sorted(overlap)[:10]}"
        )
    combined_ids = canonical_ids | supplement_ids
    if len(combined_ids) != expected_total:
        diff = combined_ids ^ set(combined_ids)  # always empty; sanity
        raise ValueError(
            f"merge: combined unique hierids = {len(combined_ids)}, "
            f"expected {expected_total}. canonical={len(canonical_ids)}, "
            f"supplement={len(supplement_ids)}."
        )

    canon = canonical_frame.copy()
    canon["source"] = SOURCE_BIGQUERY
    supp = supplement_frame.copy()
    supp["source"] = SOURCE_SHAPEFILE_SUPPLEMENT

    expected_cols = list(schema.columns) + ["source"]
    for piece, name in ((canon, "canonical"), (supp, "supplement")):
        missing = [c for c in schema.columns if c not in piece.columns]
        if missing:
            raise ValueError(
                f"merge: {name} frame missing columns {missing}"
            )

    combined = pd.concat(
        [canon[expected_cols], supp[expected_cols]], ignore_index=True
    )
    combined = combined.sort_values(id_fields + ["cell_ix", "cell_iy"]).reset_index(
        drop=True
    )

    # Sum-to-1 must still hold across the merge.
    sum_report = check_sum_to_one(combined, schema, tolerance=tolerance)
    if not sum_report.ok:
        raise ValueError(
            f"merge: sum-to-1 failed after combine: {sum_report.summary()}"
        )

    n_regions = combined[primary].nunique()
    if n_regions != expected_total:
        raise ValueError(
            f"merge: combined frame has {n_regions} unique {primary}, "
            f"expected {expected_total}"
        )

    merged_manifest = _build_merged_manifest(
        canonical_manifest,
        supplement_manifest,
        n_canonical=len(canon),
        n_supplement=len(supp),
        n_regions=n_regions,
        expected_total=expected_total,
    )

    return MergeResult(
        frame=combined,
        schema=schema,
        merged_manifest=merged_manifest,
        n_canonical=int(len(canon)),
        n_supplement=int(len(supp)),
        n_regions=int(n_regions),
    )


def _build_merged_manifest(
    canonical: dict,
    supplement: dict,
    *,
    n_canonical: int,
    n_supplement: int,
    n_regions: int,
    expected_total: int,
) -> dict:
    return {
        "merged_at_utc": canonical.get("created_at_utc", None),
        "expected_total_regions": expected_total,
        "combined_unique_regions": n_regions,
        "row_counts": {
            "canonical": n_canonical,
            "supplement": n_supplement,
            "total": n_canonical + n_supplement,
        },
        "sources": {
            "canonical": {
                "config_hash": canonical.get("config_hash"),
                "inputs": canonical.get("inputs", {}),
                "extra": {
                    "null_geometry_count": canonical.get("extra", {}).get(
                        "null_geometry_count", 0
                    ),
                    "null_geometry_regions": canonical.get("extra", {}).get(
                        "null_geometry_regions", []
                    ),
                    "bq_compute_location": canonical.get("extra", {}).get(
                        "bq_compute_location"
                    ),
                    "bq_dry_run_bytes": canonical.get("extra", {}).get(
                        "bq_dry_run_bytes"
                    ),
                },
            },
            "supplement": {
                "config_hash": supplement.get("config_hash"),
                "inputs": supplement.get("inputs", {}),
                "extra": {
                    "repaired_geometry_count": supplement.get("extra", {}).get(
                        "repaired_geometry_count", 0
                    ),
                    "repaired_geometry_regions": supplement.get("extra", {}).get(
                        "repaired_geometry_regions", []
                    ),
                    "bq_compute_location": supplement.get("extra", {}).get(
                        "bq_compute_location"
                    ),
                    "bq_dry_run_bytes": supplement.get("extra", {}).get(
                        "bq_dry_run_bytes"
                    ),
                },
            },
        },
    }


def load_manifest(uri: str) -> dict:
    """Read a JSON manifest from local or GCS."""
    if uri.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(uri, "r") as f:
            return json.load(f)
    return json.loads(Path(uri).read_text())


def load_parquet(uri: str) -> pd.DataFrame:
    if uri.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(uri, "rb") as f:
            return pd.read_parquet(f)
    return pd.read_parquet(uri)
