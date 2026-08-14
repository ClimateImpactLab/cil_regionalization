"""s51 case runner; config-driven, never reads from stdin.

s51 is the internal name of a specific weights deliverable: the full
impact region by one degree grid run with area and population weights.
The name is historical; it appears wherever configs, tests, and outputs
reproduce that product.

Modes
-----
    python examples/s51/runner.py                    # test mode (default)
    python examples/s51/runner.py --full             # production
    python examples/s51/runner.py --yes              # bypass output.confirm_cost
    python examples/s51/runner.py --config <path>    # override config path

Cost gating
-----------
- The dry-run ceiling is non-negotiable: if the estimated bytes scanned
  exceed `backend.bigquery.dry_run_byte_ceiling`, the runner aborts. The
  ceiling lives in the config; raise it explicitly there to allow a
  larger query.
- The "human pause" is separate. `output.confirm_cost = true` (the
  library default) makes the runner print the dry-run estimate and exit
  non-zero, naming the two ways forward: set `confirm_cost = false` in
  the config, or invoke with `--yes`. There is no interactive prompt.
  This keeps the runner usable under nbconvert, cron, and SLURM.

Coverage
--------
- The IR table has measured 17 NULL geometries inherited from the
  legacy pipeline's SAFE.ST_GEOGFROMTEXT. With
  `regions.on_null_geometry = "skip"` (s51 default), they are excluded
  and recorded in the manifest under `null_geometry_regions`.
- The runner's hard assertion in full mode is
      output_regions + null_skipped == 24,378
  with the breakdown printed. The deliverable is at impact-region
  level: the output's `hierid` column is asserted to be unique per row
  per region, matching the IR table minus null_skipped.
"""
from __future__ import annotations

import argparse
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


_EXPECTED_IR_REGION_COUNT = 24_378
_SMALLEST_HIERIDS_LIMIT = 5
_REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="s51-runner",
        description="s51 case runner; default is test mode.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="run against every IR region (production)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="bypass output.confirm_cost without editing the config",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="explicit config path (overrides the mode default)",
    )
    args = p.parse_args(argv)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    mode = "full" if args.full else "test"
    config_path = args.config or _default_config_for_mode(mode)
    cfg = load_config(config_path)

    rc = _validate_mode_config(cfg, mode)
    if rc != 0:
        return rc

    if mode == "test":
        cfg = _extend_with_smallest_hierids(cfg)
        print(
            f"s51_test: keep extended to "
            f"{len(cfg.regions.keep['hierid'])} hierids"
        )

    backend = BigQueryBackend()
    regions = RegionSet.from_config(cfg.regions)
    grid = GridSpec.from_config(cfg.grid)
    weights = from_config_list(cfg.weights)

    print(f"s51 {mode}: phase 1; dry-run estimate (no GPW scan yet)")
    estimate = backend.dry_run(regions, grid, weights, cfg)
    rc = _print_and_gate_estimate(estimate, cfg)
    if rc != 0:
        return rc
    rc = _confirm_cost_or_exit(cfg, args.yes)
    if rc != 0:
        return rc

    print(f"\ns51 {mode}: phase 2; executing combined query")
    result = backend.compute(regions, grid, weights, cfg)
    paths = write_result(result, cfg.output.dir, cfg.output.format)
    expected_count = (
        _EXPECTED_IR_REGION_COUNT
        if mode == "full"
        else len(cfg.regions.keep["hierid"])
    )
    rc = _verify(result, expected_count, mode)
    for kind, path in paths.items():
        print(f"  wrote {kind}: {path}")
    return rc


# ----- per-mode validation ----------------------------------------------


def _default_config_for_mode(mode: str) -> Path:
    if mode == "full":
        return _REPO / "examples" / "configs" / "s51.toml"
    return _REPO / "examples" / "configs" / "s51_test.toml"


def _validate_mode_config(cfg: Config, mode: str) -> int:
    if mode == "test":
        if cfg.output.dir.startswith("gs://"):
            print(
                "s51_test: output.dir points at GCS; test mode must write "
                "locally. Fix the config or use --full.",
                file=sys.stderr,
            )
            return 2
    else:
        if cfg.regions.keep:
            print(
                "s51 full: regions.keep is set; --full refuses to run with a "
                "partial filter. Remove regions.keep from the config.",
                file=sys.stderr,
            )
            return 2
        if not cfg.output.dir.startswith("gs://"):
            print(
                "s51 full: output.dir must be a gs:// URI for production runs. "
                f"Got: {cfg.output.dir}",
                file=sys.stderr,
            )
            return 2
    return 0


# ----- test-mode hierid discovery ---------------------------------------


def _extend_with_smallest_hierids(cfg: Config) -> Config:
    from google.cloud import bigquery

    ir_table = cfg.regions.table
    client = bigquery.Client(project=cfg.backend.bigquery.project)
    sql = (
        f"SELECT hierid FROM `{ir_table}` "
        f"WHERE geometry IS NOT NULL "
        f"ORDER BY ST_AREA(geometry) ASC LIMIT {_SMALLEST_HIERIDS_LIMIT}"
    )
    smallest = [
        r.hierid
        for r in client.query(sql, location="us-west1").result()
    ]
    print(f"  smallest-area hierids: {smallest}")
    base = list((cfg.regions.keep or {}).get("hierid", []))
    merged = sorted(set(base + smallest))
    d = cfg.model_dump()
    d["regions"]["keep"] = {"hierid": merged}
    return Config.model_validate(d)


# ----- dry-run gate + confirm-cost --------------------------------------


def _print_and_gate_estimate(estimate: dict, cfg: Config) -> int:
    bytes_est = estimate["bytes_estimate"]
    ceiling = cfg.backend.bigquery.dry_run_byte_ceiling
    print(f"  compute location : {estimate['compute_location']}")
    print(f"  IR location      : {estimate['ir_location']}")
    print(f"  temp table       : {estimate['temp_table']}")
    print(f"  cache hit        : {estimate['cache_hit']}")
    print(f"  geometries       : {estimate['n_regions']} regions")
    print(
        f"  null_skipped     : {len(estimate['null_skipped'])}"
        f" {estimate['null_skipped'][:5]}"
        + ("..." if len(estimate["null_skipped"]) > 5 else "")
    )
    print(
        f"  unknown_skipped  : {len(estimate['unknown_skipped'])}"
        f" {estimate['unknown_skipped'][:5]}"
        + ("..." if len(estimate["unknown_skipped"]) > 5 else "")
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
            "  ABORT: dry-run estimate exceeds the configured ceiling.\n"
            "  Raise backend.bigquery.dry_run_byte_ceiling explicitly "
            "in the config if this is intentional.",
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
        "human-acknowledgement gate, not a cost limit.",
        file=sys.stderr,
    )
    return 1


# ----- verification ------------------------------------------------------


def _verify(result, expected_total: int, mode: str) -> int:
    print(f"\n{result.sum_report.summary()}")
    print(f"row_counts     : {dict(result.manifest.row_counts)}")
    print(f"fallback_counts: {result.manifest.fallback_counts}")
    print(f"timing_seconds : {dict(result.manifest.timing_seconds)}")

    n_returned = int(result.manifest.row_counts.get("regions", 0))
    n_null = int(result.manifest.extra.get("null_geometry_count", 0))
    null_ids = list(result.manifest.extra.get("null_geometry_regions", []))
    n_unknown = int(result.manifest.extra.get("unknown_id_count", 0))
    unknown_ids = list(result.manifest.extra.get("unknown_id_regions", []))
    coverage = n_returned + n_null + n_unknown
    print(f"\nrequested        : {expected_total}")
    print(f"returned regions : {n_returned}")
    print(f"null_skipped     : {n_null}  {null_ids[:5]}"
          + ("..." if len(null_ids) > 5 else ""))
    print(f"unknown_skipped  : {n_unknown}  {unknown_ids[:5]}"
          + ("..." if len(unknown_ids) > 5 else ""))
    print(f"coverage         : {coverage} / {expected_total}")

    rc = 0
    if coverage != expected_total:
        print(
            f"FAIL: coverage mismatch; {n_returned} returned + {n_null} "
            f"null_skipped + {n_unknown} unknown_skipped != "
            f"{expected_total} requested.",
            file=sys.stderr,
        )
        rc = 1

    # Hierid uniqueness: the deliverable is impact-region level.
    primary_id = result.schema.id_fields[0]
    unique_hierids = result.frame[primary_id].nunique()
    if unique_hierids != n_returned:
        print(
            f"FAIL: unique {primary_id} count {unique_hierids} != "
            f"row_counts['regions'] {n_returned}",
            file=sys.stderr,
        )
        rc = 1

    if not result.sum_report.ok:
        rc = 1

    # Spot-check: top hierid by pop and first nearest_cell (if any).
    if "pop_raw" in result.frame.columns:
        top = (
            result.frame.groupby(primary_id)["pop_raw"]
            .sum()
            .sort_values(ascending=False)
        )
        if len(top) > 0:
            print(f"\ntop hierid by pop_raw: {top.index[0]} -> {top.iloc[0]:,.0f}")
    nc_rows = result.frame.loc[
        result.frame.get("pop_method", pd.Series([])) == "nearest_cell"
    ]
    if len(nc_rows) > 0:
        print(f"nearest_cell hierids: {sorted(nc_rows[primary_id].unique())[:10]}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
