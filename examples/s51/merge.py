"""Merge the canonical s51 parquet with the shapefile-source supplement.

Reads both parquets + manifests from local or GCS, asserts the merge
invariants (disjoint, coverage, sum-to-1), prints a sanity table for
the top-5 regions by cell count, and writes
``weights_complete.parquet`` + ``weights_complete.manifest.json``
under a target prefix.

    python examples/s51/merge.py \\
        --canonical gs://.../segment_weights/s51/weights.parquet \\
        --supplement gs://.../segment_weights/s51/supplement/weights.parquet \\
        --output gs://.../segment_weights/s51/

Defaults match the s51 layout: canonical at the s51/ prefix, supplement
at s51/supplement/, output back into s51/. The merge never reads from
stdin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from segment_weights.merge import (
    load_manifest,
    load_parquet,
    merge_weights,
)
from segment_weights.schema import OutputSchema


_EXPECTED_IR_REGION_COUNT = 24_378
_DEFAULT_CANONICAL = (
    "gs://impactlab-data-scratch/scadavidsanchez/"
    "climate-and-damages-aggregation/segment_weights/s51/weights.parquet"
)
_DEFAULT_SUPPLEMENT = (
    "gs://impactlab-data-scratch/scadavidsanchez/"
    "climate-and-damages-aggregation/segment_weights/s51/supplement/"
    "weights.parquet"
)
_DEFAULT_OUTPUT_PREFIX = (
    "gs://impactlab-data-scratch/scadavidsanchez/"
    "climate-and-damages-aggregation/segment_weights/s51/"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="s51-merge")
    p.add_argument("--canonical", type=str, default=_DEFAULT_CANONICAL)
    p.add_argument("--supplement", type=str, default=_DEFAULT_SUPPLEMENT)
    p.add_argument(
        "--output-prefix",
        type=str,
        default=_DEFAULT_OUTPUT_PREFIX,
        help="local dir or gs:// prefix; weights_complete.parquet lands here",
    )
    p.add_argument(
        "--expected-total",
        type=int,
        default=_EXPECTED_IR_REGION_COUNT,
        help="expected unique hierids in the combined frame",
    )
    p.add_argument(
        "--id-field",
        type=str,
        default="hierid",
        help="primary id field; should match the source schemas",
    )
    args = p.parse_args(argv)

    print("merge: loading inputs")
    canonical_frame = load_parquet(args.canonical)
    canonical_manifest = load_manifest(
        args.canonical.rsplit(".", 1)[0] + ".manifest.json"
    )
    supplement_frame = load_parquet(args.supplement)
    supplement_manifest = load_manifest(
        args.supplement.rsplit(".", 1)[0] + ".manifest.json"
    )
    print(f"  canonical : {len(canonical_frame):,} rows, "
          f"{canonical_frame[args.id_field].nunique():,} unique {args.id_field}")
    print(f"  supplement: {len(supplement_frame):,} rows, "
          f"{supplement_frame[args.id_field].nunique():,} unique {args.id_field}")

    # The schema is the canonical layout the backends produce. Both
    # frames must conform, so we read the weight names from the
    # canonical frame's columns.
    weight_names = tuple(
        c[:-2] for c in canonical_frame.columns if c.endswith("wt")
    )
    schema = OutputSchema(
        id_fields=(args.id_field,), weight_names=weight_names
    )

    try:
        result = merge_weights(
            canonical_frame=canonical_frame,
            canonical_manifest=canonical_manifest,
            supplement_frame=supplement_frame,
            supplement_manifest=supplement_manifest,
            schema=schema,
            expected_total=args.expected_total,
        )
    except ValueError as e:
        print(f"merge: FAIL: {e}", file=sys.stderr)
        return 1

    print()
    print("merge: invariants OK")
    print(f"  combined rows    : {len(result.frame):,}")
    print(f"  unique hierids   : {result.n_regions:,}")
    print(f"  canonical rows   : {result.n_canonical:,}")
    print(f"  supplement rows  : {result.n_supplement:,}")

    print("\nmerge: top-5 hierids by cell count")
    top = (
        result.frame.groupby(args.id_field)
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    print(top.to_string())

    _write(result, args.output_prefix)
    return 0


def _write(result, output_prefix: str) -> None:
    suffix = output_prefix.rstrip("/") + "/"
    if suffix.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        p_pq = suffix + "weights_complete.parquet"
        p_mf = suffix + "weights_complete.manifest.json"
        with fs.open(p_pq, "wb") as f:
            result.frame.to_parquet(f, index=False)
        with fs.open(p_mf, "w") as f:
            json.dump(result.merged_manifest, f, indent=2, sort_keys=True, default=str)
    else:
        out = Path(suffix)
        out.mkdir(parents=True, exist_ok=True)
        p_pq = str(out / "weights_complete.parquet")
        p_mf = str(out / "weights_complete.manifest.json")
        result.frame.to_parquet(p_pq, index=False)
        Path(p_mf).write_text(
            json.dumps(
                result.merged_manifest, indent=2, sort_keys=True, default=str
            )
        )
    print(f"\n  wrote parquet : {p_pq}")
    print(f"  wrote manifest: {p_mf}")


if __name__ == "__main__":
    raise SystemExit(main())
