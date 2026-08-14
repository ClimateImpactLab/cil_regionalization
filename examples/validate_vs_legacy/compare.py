"""Characterize our crop weights against the legacy CSV on the RCC.

This is a CHARACTERIZATION, not a regression gate. Our crop weights are
not expected to match the legacy file byte-for-byte:

    - the legacy 'area' column uses planar-degree areas (a known bug);
    - the legacy fallback for zero-crop regions used a polylabel +
      1e-5 dummy weight; we use the marked 'nan' policy;
    - per-pixel raster handling differs at boundary cells.

What the script reports:

    - per-region max|delta cropwt| distribution (median, p95, max)
      over the BOTH-NATIVE subset (the unit-cancelling spatial check);
    - count of regions where the zero-crop set agrees vs differs;
    - count + list of regions where the maximum-weight cell flipped
      between the two pipelines (the real spatial-disagreement signal).

RCC env:

    module load python
    source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/

Run:

    python examples/validate_vs_legacy/compare.py \\
        --ours data/out/rcc_crops/weights_legacy.csv \\
        --legacy /project/cil/.../agglomerated-world-new_GMFD_grid_segment_weights_area_pop_crop.csv \\
        --weight crop \\
        [--per-region-csv data/out/rcc_crops/legacy_compare_per_region.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from segment_weights.legacy_compare import (  # noqa: E402
    compare_weights,
    load_legacy_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare our crop weights to the legacy CSV (characterization)."
    )
    parser.add_argument(
        "--ours",
        required=True,
        help="path to our legacy-schema CSV (output of `segweights run --legacy-csv`)",
    )
    parser.add_argument(
        "--legacy",
        required=True,
        help="path to the legacy reference CSV",
    )
    parser.add_argument(
        "--weight",
        default="crop",
        help="weight name to compare (default: crop)",
    )
    parser.add_argument(
        "--per-region-csv",
        default=None,
        help="optional path to dump the per-region detail (hierid, max_delta, mass_displaced)",
    )
    args = parser.parse_args()

    ours_path = Path(args.ours)
    legacy_path = Path(args.legacy)
    for p, label in ((ours_path, "ours"), (legacy_path, "legacy")):
        if not p.exists():
            print(f"{label} CSV not found: {p}", file=sys.stderr)
            return 1

    print(f"loading ours   : {ours_path}")
    ours = load_legacy_csv(ours_path)
    print(f"  {len(ours)} rows, {ours['hierid'].nunique()} regions")
    print(f"loading legacy : {legacy_path}")
    legacy = load_legacy_csv(legacy_path)
    print(f"  {len(legacy)} rows, {legacy['hierid'].nunique()} regions")

    report = compare_weights(ours, legacy, weight=args.weight)
    print()
    print(report.summary())

    if args.per_region_csv:
        out = Path(args.per_region_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        # The tuple columns ours_max_cell / legacy_max_cell don't round-
        # trip cleanly via to_csv; split them into x/y columns.
        per_region = report.per_region.copy()
        if not per_region.empty:
            per_region["ours_max_x"] = per_region["ours_max_cell"].map(lambda t: t[0])
            per_region["ours_max_y"] = per_region["ours_max_cell"].map(lambda t: t[1])
            per_region["legacy_max_x"] = per_region["legacy_max_cell"].map(lambda t: t[0])
            per_region["legacy_max_y"] = per_region["legacy_max_cell"].map(lambda t: t[1])
            per_region = per_region.drop(columns=["ours_max_cell", "legacy_max_cell"])
        per_region.to_csv(out, index=False)
        print(f"\nper-region detail: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
