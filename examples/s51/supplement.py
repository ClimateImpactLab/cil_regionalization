"""s51 supplement runner.

Picks up the 17 NULL-geometry hierids from the canonical s51 run's
manifest, points the BigQuery backend at the source shapefile so the
shapes come from a clean source (the IR table itself has
``geometry IS NULL`` on these rows), and re-runs identical methodology
(same grid, same pop weight, same fallback). The result lands as a
sibling Parquet next to the canonical s51 parquet so the merge step
combines both with one prefix scan.

    python examples/s51/supplement.py             # dry-run then stop
    python examples/s51/supplement.py --yes       # full execute

This runner never reads from stdin. Cost gating mirrors the canonical
s51 runner: the dry-run ceiling is a hard limit; ``--yes`` (or
``output.confirm_cost = false`` in the config) is the human-ack gate.

Strict-by-default: any unknown id, any null geometry, any repair beyond
``shapely.make_valid`` are surfaced. The runner exits non-zero on:

- canonical full-run manifest unreadable,
- requested null_geometry_regions list empty (nothing to supplement),
- any of the 17 hierids absent from the source shapefile,
- ``len(output_regions) != len(requested)`` after the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from segment_weights import load_config
from segment_weights.backends.bigquery import BigQueryBackend
from segment_weights.config import Config
from segment_weights.grid import GridSpec
from segment_weights.io import write_result
from segment_weights.regions import RegionSet
from segment_weights.weights import from_config_list


_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO / "examples" / "configs" / "s51_supplement.toml"
_DEFAULT_FULL_MANIFEST = (
    "gs://impactlab-data-scratch/scadavidsanchez/"
    "climate-and-damages-aggregation/segment_weights/s51/"
    "weights.manifest.json"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="s51-supplement")
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="config path (default: examples/configs/s51_supplement.toml)",
    )
    p.add_argument(
        "--full-manifest",
        type=str,
        default=_DEFAULT_FULL_MANIFEST,
        help="GCS or local URI of the canonical s51 run's manifest",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="bypass output.confirm_cost without editing the config",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    null_hierids = _load_null_hierids_from_manifest(args.full_manifest)
    if not null_hierids:
        print(
            "supplement: full-run manifest has empty null_geometry_regions; "
            "nothing to supplement. Stop.",
            file=sys.stderr,
        )
        return 2

    d = cfg.model_dump()
    d["regions"]["keep"] = {"hierid": sorted(null_hierids)}
    cfg = Config.model_validate(d)
    print(f"supplement: keep = {len(null_hierids)} hierids from full manifest")

    backend = BigQueryBackend()
    regions = RegionSet.from_config(cfg.regions)
    grid = GridSpec.from_config(cfg.grid)
    weights = from_config_list(cfg.weights)

    if regions.repaired_ids:
        print(f"supplement: repaired {len(regions.repaired_ids)} geometries "
              f"via shapely.make_valid")
        for r in regions.repaired_ids:
            print(f"  - {r}")

    print("supplement: phase 1; dry-run estimate")
    estimate = backend.dry_run(regions, grid, weights, cfg)
    rc = _print_and_gate_estimate(estimate, cfg)
    if rc != 0:
        return rc
    rc = _confirm_cost_or_exit(cfg, args.yes)
    if rc != 0:
        return rc

    print("\nsupplement: phase 2; executing combined query")
    result = backend.compute(regions, grid, weights, cfg)
    paths = write_result(result, cfg.output.dir, cfg.output.format)
    return _verify(result, expected=len(null_hierids), paths=paths)


# ----- helpers ----------------------------------------------------------


def _load_null_hierids_from_manifest(manifest_uri: str) -> list[str]:
    """Read the canonical s51 manifest from local or GCS and return the
    `null_geometry_regions` list. Aborts with a clear error on any I/O
    or shape problem; the supplement should never silently invent a
    keep list."""
    if manifest_uri.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(manifest_uri, "r") as f:
            data = json.load(f)
    else:
        data = json.loads(Path(manifest_uri).read_text())
    extra = data.get("extra", {})
    null_ids = extra.get("null_geometry_regions", [])
    if not isinstance(null_ids, list):
        raise ValueError(
            f"manifest at {manifest_uri} has unexpected "
            f"null_geometry_regions type: {type(null_ids)}"
        )
    return [str(h) for h in null_ids]


def _print_and_gate_estimate(estimate: dict, cfg: Config) -> int:
    bytes_est = estimate["bytes_estimate"]
    ceiling = cfg.backend.bigquery.dry_run_byte_ceiling
    print(f"  compute location : {estimate['compute_location']}")
    print(f"  IR location      : {estimate['ir_location']}  (no IR query for supplement)")
    print(f"  temp table       : {estimate['temp_table']}")
    print(f"  cache hit        : {estimate['cache_hit']}")
    print(f"  geometries       : {estimate['n_regions']} regions")
    print(f"  null_skipped     : {len(estimate['null_skipped'])} {estimate['null_skipped'][:5]}")
    print(
        f"  unknown_skipped  : {len(estimate['unknown_skipped'])}"
        f" {estimate['unknown_skipped'][:5]}"
    )
    print(
        f"  dry-run estimate : {bytes_est:>15,d} bytes "
        f" ({bytes_est / 1e9:.2f} GB)"
    )
    print(
        f"  configured ceiling: {ceiling:>15,d} bytes "
        f" ({ceiling / 1e9:.2f} GB)"
    )
    if bytes_est > ceiling:
        print(
            "  ABORT: dry-run estimate exceeds the configured ceiling. "
            "Raise backend.bigquery.dry_run_byte_ceiling explicitly in the "
            "config to proceed.",
            file=sys.stderr,
        )
        return 1
    return 0


def _confirm_cost_or_exit(cfg: Config, yes_flag: bool) -> int:
    if not cfg.output.confirm_cost or yes_flag:
        return 0
    print(
        "\n  STOP: output.confirm_cost is true and --yes was not passed.\n"
        "  Not executing. To proceed, either:\n"
        "    - re-run with --yes\n"
        "    - set output.confirm_cost = false in the config\n"
        "  The dry-run ceiling check has already passed; this is the "
        "human-acknowledgement gate.",
        file=sys.stderr,
    )
    return 1


def _verify(result, *, expected: int, paths: dict[str, str]) -> int:
    print(f"\n{result.sum_report.summary()}")
    print(f"row_counts     : {dict(result.manifest.row_counts)}")
    print(f"fallback_counts: {result.manifest.fallback_counts}")
    print(f"timing_seconds : {dict(result.manifest.timing_seconds)}")
    n_returned = int(result.manifest.row_counts.get("regions", 0))
    print(f"\nrequested        : {expected}")
    print(f"returned regions : {n_returned}")
    if n_returned != expected:
        print(
            f"FAIL: returned {n_returned} regions, expected {expected}. "
            f"Every supplement hierid must be present in the output.",
            file=sys.stderr,
        )
        return 1
    primary = result.schema.id_fields[0]
    if result.frame[primary].nunique() != n_returned:
        print(
            f"FAIL: unique {primary} count != row_counts['regions']",
            file=sys.stderr,
        )
        return 1
    if not result.sum_report.ok:
        return 1
    for kind, path in paths.items():
        print(f"  wrote {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
