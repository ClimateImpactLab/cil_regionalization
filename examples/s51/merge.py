"""Combine the two s51 parts into one weights table of 24,378 regions.

Reads what s51_canonical.toml and s51_supplement.toml write under
data/weights/, joins the two with cil_regionalization.merge.merge_weights,
and writes data/weights/s51/weights.parquet with its manifest. Nothing
is written, and the exit status is nonzero, if a hierid is in both
parts, if the two parts together do not hold exactly 24,378 hierids,
or if any region's popwt or areawt does not sum to one.

The first version of this example was built in BigQuery, and this
reproduces it.

    python examples/s51/merge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cil_regionalization.merge import load_manifest, load_parquet, merge_weights
from cil_regionalization.schema import OutputSchema

WEIGHTS = Path(__file__).resolve().parent / "data" / "weights"
PARTS = ("s51_canonical", "s51_supplement")
N_HIERIDS = 24_378  # every region in world-combo-new


def read_part(weights_dir: Path, name: str):
    """One part's weights frame and manifest."""
    d = weights_dir / name
    frame = load_parquet(str(d / "weights.parquet"))
    manifest = load_manifest(str(d / "weights.manifest.json"))
    print(f"{name}: {len(frame):,} rows, {frame['hierid'].nunique():,} hierids")
    return frame, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine the two s51 parts.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=WEIGHTS,
        help="directory holding the two parts; the combined table goes to s51/ in it",
    )
    args = parser.parse_args(argv)

    (canonical, canonical_manifest), (supplement, supplement_manifest) = (
        read_part(args.weights_dir, name) for name in PARTS
    )
    try:
        merged = merge_weights(
            canonical_frame=canonical,
            canonical_manifest=canonical_manifest,
            supplement_frame=supplement,
            supplement_manifest=supplement_manifest,
            schema=OutputSchema(id_fields=("hierid",), weight_names=("pop", "area")),
            expected_total=N_HIERIDS,
        )
    except ValueError as e:
        print(f"nothing written: {e}", file=sys.stderr)
        return 1

    out = args.weights_dir / "s51"
    out.mkdir(parents=True, exist_ok=True)
    merged.frame.to_parquet(out / "weights.parquet", index=False)
    (out / "weights.manifest.json").write_text(
        json.dumps(merged.merged_manifest, indent=2, sort_keys=True, default=str)
    )
    print(f"s51: {len(merged.frame):,} rows, {merged.n_regions:,} hierids, written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
