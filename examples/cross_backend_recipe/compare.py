"""Cross-backend comparison: local-backend output vs s51 BQ deliverable.

CHARACTERIZATION, not a regression gate. Expected differences come from
geodesic (pyproj.Geod) vs spherical (BQ ST_AREA) and from planar shapely
intersection vs spherical BQ ST_INTERSECTION; mostly the boundary-cell
tail. Normalised weight agreement should sit comfortably below
`validation.cross_backend_tolerance` (1e-3 by default); the report
counts regions above the budget rather than asserting.

RCC env::

    module load python
    source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/

Default paths point at the canonical s51 BQ deliverable in GCS and the
local output of the paired config under `data/out/cross_backend_recipe/`.

    python examples/cross_backend_recipe/compare.py \\
        --local data/out/cross_backend_recipe/weights.parquet \\
        --bq    gs://impactlab-data-scratch/scadavidsanchez/climate-and-damages-aggregation/segment_weights/s51/weights_complete.parquet \\
        --weight popwt \\
        [--tolerance 1e-3] \\
        [--per-region-csv data/out/cross_backend_recipe/popwt_compare.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from segment_weights.cross_backend_compare import (  # noqa: E402
    compare_outputs,
    load_canonical_parquet,
)


_DEFAULT_BQ_DELIVERABLE = (
    "gs://impactlab-data-scratch/scadavidsanchez/"
    "climate-and-damages-aggregation/segment_weights/s51/"
    "weights_complete.parquet"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize local-backend weights vs the s51 BQ deliverable."
        )
    )
    parser.add_argument(
        "--local",
        required=True,
        help="path or gs:// URI to the local-backend canonical parquet",
    )
    parser.add_argument(
        "--bq",
        default=_DEFAULT_BQ_DELIVERABLE,
        help="path or gs:// URI to the BQ deliverable canonical parquet "
        "(default: the s51 weights_complete.parquet under impactlab-data-scratch)",
    )
    parser.add_argument(
        "--weight",
        default="popwt",
        help="weight column to compare (popwt or areawt; default popwt)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="cross-backend tolerance for the above-budget count "
        "(default: 1e-3, matching validation.cross_backend_tolerance)",
    )
    parser.add_argument(
        "--per-region-csv",
        default=None,
        help="optional path to dump per-region detail "
        "(hierid, max_delta, mass_displaced, above_tolerance)",
    )
    args = parser.parse_args()

    print(f"loading local : {args.local}")
    local = load_canonical_parquet(args.local)
    print(f"  {len(local)} rows, {local['hierid'].nunique()} regions")
    print(f"loading bq    : {args.bq}")
    bq = load_canonical_parquet(args.bq)
    print(f"  {len(bq)} rows, {bq['hierid'].nunique()} regions")

    report = compare_outputs(
        local, bq, weight=args.weight, tolerance=args.tolerance
    )
    print()
    print(report.summary())

    if args.per_region_csv:
        out = Path(args.per_region_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        per_region = report.per_region.copy()
        if not per_region.empty:
            per_region["local_max_ix"] = per_region["local_max_cell"].map(
                lambda t: t[0]
            )
            per_region["local_max_iy"] = per_region["local_max_cell"].map(
                lambda t: t[1]
            )
            per_region["bq_max_ix"] = per_region["bq_max_cell"].map(
                lambda t: t[0]
            )
            per_region["bq_max_iy"] = per_region["bq_max_cell"].map(
                lambda t: t[1]
            )
            per_region = per_region.drop(
                columns=["local_max_cell", "bq_max_cell"]
            )
        per_region.to_csv(out, index=False)
        print(f"\nper-region detail: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
