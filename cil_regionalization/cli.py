"""`cilreg` command-line entry point.

Subcommands:

    cilreg validate <config.toml>
        Load and validate the config, check that referenced files exist,
        and print a one-line summary. No compute. No network.

    cilreg run <config.toml> [--test-mode]
        Load the config, dispatch to the configured backend, write results
        and a manifest, then print a one-line summary plus the sum-to-1
        report status. `--test-mode` caps the local backend to the first
        three regions for fast smoke testing.

    cilreg pipeline <pipeline.toml> [--dry-run]
        Run the Monte Carlo pipeline (apply weights per leaf, then
        statistics). Thin wrapper over
        `python -m cil_regionalization.pipelines.montecarlo`.

    cilreg regions find <pattern> <config.toml>
        LIKE-search the configured regions source for matching ids.

    cilreg fetch <name> [--record ID] [--base-url URL] [--refresh]
        Fetch a published weight artifact from Zenodo, verify it against
        its manifest checksum, cache it, and print the cached path.

    cilreg cache list | clear [--name NAME]
        Show or remove cached fetched artifacts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cil_regionalization.backends.base import WeightsResult
from cil_regionalization.config import Config, load_config
from cil_regionalization.grid import GridSpec
from cil_regionalization.io import write_result
from cil_regionalization.legacy_export import write_legacy_csv
from cil_regionalization.regions import RegionSet
from cil_regionalization.weights import from_config_list


_TEST_MODE_REGION_CAP = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cilreg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser(
        "validate",
        help="check config and input existence; no compute",
    )
    pv.add_argument("config", type=str, help="path to a TOML config")

    pr = sub.add_parser(
        "run",
        help="execute the configured backend and write outputs",
    )
    pr.add_argument("config", type=str, help="path to a TOML config")
    pr.add_argument(
        "--test-mode",
        action="store_true",
        help=(
            f"local backend only: cap regions at the first "
            f"{_TEST_MODE_REGION_CAP} for a fast smoke test"
        ),
    )
    pr.add_argument(
        "--legacy-csv",
        action="store_true",
        help=(
            "after writing the canonical output, also emit weights_legacy.csv "
            "in the legacy 13-column schema. Requires area + crop + pop "
            "weights and id_fields=['hierid']."
        ),
    )

    pp = sub.add_parser(
        "pipeline",
        help="run the Monte Carlo pipeline from a pipeline TOML",
    )
    pp.add_argument("config", type=str, help="path to a pipeline TOML")
    pp.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the tree and print the plan; touch nothing",
    )

    preg = sub.add_parser(
        "regions",
        help="region utilities (find, list, etc.)",
    )
    psub = preg.add_subparsers(dest="regions_cmd", required=True)
    pfind = psub.add_parser(
        "find",
        help="LIKE-search the regions source for matching ids",
    )
    pfind.add_argument(
        "pattern",
        type=str,
        help="SQL LIKE pattern, e.g. 'BMU%' or 'CAN.5%%'",
    )
    pfind.add_argument("config", type=str, help="path to a TOML config")
    pfind.add_argument("--limit", type=int, default=50)

    pf = sub.add_parser(
        "fetch",
        help="fetch a published weight artifact from Zenodo and cache it",
    )
    pf.add_argument(
        "name",
        type=str,
        nargs="?",
        default=None,
        help="registry name of the artifact (or a cache label with --record)",
    )
    pf.add_argument(
        "--list",
        action="store_true",
        dest="list_names",
        help="list the weight file names the registry knows and exit",
    )
    pf.add_argument(
        "--record",
        type=str,
        default=None,
        help="Zenodo record id or DOI; bypasses the registry",
    )
    pf.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Zenodo instance, e.g. https://sandbox.zenodo.org for tests",
    )
    pf.add_argument(
        "--registry",
        type=str,
        default=None,
        help="path to a local registry TOML overlaying the packaged one",
    )
    pf.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="cache directory (default ~/.cache/cil_regionalization)",
    )
    pf.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even when a verified cached copy exists",
    )

    pc = sub.add_parser("cache", help="show or remove cached fetched artifacts")
    csub = pc.add_subparsers(dest="cache_cmd", required=True)
    cl = csub.add_parser("list", help="list cached artifacts")
    cl.add_argument("--cache-dir", type=str, default=None)
    cc = csub.add_parser("clear", help="remove cached artifacts")
    cc.add_argument("--cache-dir", type=str, default=None)
    cc.add_argument(
        "--name", type=str, default=None, help="remove only this artifact"
    )

    args = parser.parse_args(argv)
    if args.cmd == "validate":
        return _cmd_validate(args.config)
    if args.cmd == "run":
        return _cmd_run(
            args.config,
            test_mode=args.test_mode,
            legacy_csv=args.legacy_csv,
        )
    if args.cmd == "pipeline":
        # Thin wrapper over the pipeline entry point; the pipeline module
        # owns its argument handling and output.
        from cil_regionalization.pipelines.montecarlo import main as pipeline_main

        argv_out = [args.config] + (["--dry-run"] if args.dry_run else [])
        return pipeline_main(argv_out)
    if args.cmd == "regions":
        if args.regions_cmd == "find":
            return _cmd_regions_find(
                args.config, pattern=args.pattern, limit=args.limit
            )
    if args.cmd == "fetch":
        return _cmd_fetch(args)
    if args.cmd == "cache":
        return _cmd_cache(args)
    parser.print_help()
    return 2


def _cmd_fetch(args: argparse.Namespace) -> int:
    from cil_regionalization.fetch import FetchError, fetch_weights, load_registry

    if args.list_names:
        entries = load_registry(args.registry)
        if not entries:
            print("registry knows no artifacts")
            return 0
        for name, entry in sorted(entries.items()):
            print(f"{name}  record={entry.record}")
        return 0
    if args.name is None:
        print("fetch: a name is required (or --list to see what exists)",
              file=sys.stderr)
        return 2

    try:
        artifact = fetch_weights(
            args.name,
            record=args.record,
            base_url=args.base_url,
            registry_path=args.registry,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )
    except FetchError as e:
        print(f"fetch: {e}", file=sys.stderr)
        return 1
    from cil_regionalization.fetch import default_cache_dir, list_cached

    cache_dir = args.cache_dir or default_cache_dir()
    entry = next(
        (i for i in list_cached(cache_dir) if i["name"] == args.name), None
    )
    print(
        f"fetch: {args.name}  rows={len(artifact.frame)}  "
        f"normalization={artifact.normalization}  "
        f"source_version={artifact.source_version}"
    )
    if entry:
        print(f"  cached at: {entry['path']}")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    from cil_regionalization.fetch import clear_cache, list_cached

    if args.cache_cmd == "list":
        items = list_cached(args.cache_dir)
        if not items:
            print("cache: empty")
            return 0
        for i in items:
            print(
                f"{i['name']}  record={i['record']}  "
                f"{i['size_bytes'] / 1e6:.1f} MB  {i['path']}"
            )
        return 0
    if args.cache_cmd == "clear":
        n = clear_cache(args.cache_dir, name=args.name)
        print(f"cache: removed {n} artifact(s)")
        return 0
    return 2


def _cmd_validate(config_path: str) -> int:
    cfg = load_config(config_path)
    problems = _check_inputs_exist(cfg)
    if problems:
        for p in problems:
            print(f"validate: {p}", file=sys.stderr)
        return 1
    if cfg.grid is not None:
        geometry = f"grid={cfg.grid.mode}/{cfg.grid.resolution}"
    else:
        geometry = f"source=polygons/{cfg.source.version}"
    print(
        f"validate: OK  backend={cfg.backend.kind}  coverage={cfg.backend.coverage}  "
        f"{geometry}  "
        f"weights={[w.name for w in cfg.weights]}"
    )
    return 0


def _cmd_run(
    config_path: str,
    *,
    test_mode: bool,
    legacy_csv: bool = False,
) -> int:
    cfg = load_config(config_path)
    problems = _check_inputs_exist(cfg)
    if problems:
        for p in problems:
            print(f"run: {p}", file=sys.stderr)
        return 1

    regions = RegionSet.from_config(cfg.regions)
    grid = GridSpec.from_config(cfg.grid) if cfg.grid is not None else None
    weight_specs = from_config_list(cfg.weights)

    if cfg.backend.kind == "bigquery":
        from cil_regionalization.backends.bigquery import BigQueryBackend

        if test_mode:
            keep = cfg.regions.keep or {}
            primary = regions.id_fields[0]
            if not keep.get(primary):
                print(
                    "run: bigquery + --test-mode requires regions.keep to limit "
                    f"{primary}; supply a small list (<= 10) in config.",
                    file=sys.stderr,
                )
                return 1
            print(
                f"run: test-mode active; regions.keep limits {primary} to "
                f"{len(keep[primary])} ids",
                file=sys.stderr,
            )
        backend = BigQueryBackend()
    else:
        from cil_regionalization.backends.local import LocalBackend

        if test_mode and regions.is_local and regions.gdf is not None:
            regions.gdf = regions.gdf.head(_TEST_MODE_REGION_CAP).reset_index(
                drop=True
            )
            print(
                f"run: test-mode active; capped regions to {_TEST_MODE_REGION_CAP}",
                file=sys.stderr,
            )
        backend = LocalBackend()

    result: WeightsResult = backend.compute(regions, grid, weight_specs, cfg)
    paths = write_result(result, cfg.output.dir, cfg.output.format)

    if legacy_csv:
        out_dir = cfg.output.dir
        if out_dir.startswith("gs://"):
            legacy_path = out_dir.rstrip("/") + "/weights_legacy.csv"
        else:
            legacy_path = str(Path(out_dir) / "weights_legacy.csv")
        paths["legacy_csv"] = write_legacy_csv(result, legacy_path)

    print(
        f"run: backend={cfg.backend.kind}  rows={len(result.frame)}  "
        f"regions={result.manifest.row_counts.get('regions', 0)}  "
        f"sum_to_one={'ok' if result.sum_report.ok else 'FAIL'}"
    )
    for kind, path in paths.items():
        print(f"  wrote {kind}: {path}")
    if not result.sum_report.ok:
        print(result.sum_report.summary(), file=sys.stderr)
        return 1
    return 0


def _cmd_regions_find(config_path: str, *, pattern: str, limit: int) -> int:
    """SQL-LIKE search the configured regions source for matching ids.

    Cheap, read-only. Helps users resolve hierids without guessing:
    world-combo-2017 mixes bare ISO3, single-remainder-only suffixes,
    and admin subdivisions across 252 prefixes. See
    examples/configs/s51_test.toml for the naming patterns.
    """
    cfg = load_config(config_path)
    primary = cfg.regions.id_fields[0]
    if cfg.regions.table is not None:
        return _regions_find_bq(cfg, primary, pattern, limit)
    return _regions_find_local(cfg, primary, pattern, limit)


def _regions_find_bq(
    cfg: Config, primary: str, pattern: str, limit: int
) -> int:
    from google.cloud import bigquery

    client = bigquery.Client(project=cfg.backend.bigquery.project)
    table = cfg.regions.table
    project, dataset, _ = table.split(".", 2)
    ds_location = client.get_dataset(f"{project}.{dataset}").location
    sql = (
        f"SELECT {primary} FROM `{table}` "
        f"WHERE {primary} LIKE @pat "
        f"ORDER BY {primary} LIMIT @lim"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("pat", "STRING", pattern),
            bigquery.ScalarQueryParameter("lim", "INT64", limit),
        ]
    )
    print(f"searching {table} for LIKE {pattern!r} (limit {limit})")
    df = client.query(sql, job_config=job_config, location=ds_location).to_dataframe(
        create_bqstorage_client=cfg.backend.bigquery.use_bqstorage
    )
    if df.empty:
        print("no matches")
        return 0
    for hid in df[primary].tolist():
        print(hid)
    print(f"\n{len(df)} matches" + (" (limit reached)" if len(df) >= limit else ""))
    return 0


def _regions_find_local(
    cfg: Config, primary: str, pattern: str, limit: int
) -> int:
    from cil_regionalization.regions import RegionSet
    import re

    regions = RegionSet.from_config(cfg.regions)
    assert regions.gdf is not None
    # SQL LIKE -> Python regex: % -> .*, _ -> .  (escape other regex chars)
    regex = "^" + re.escape(pattern).replace("%", ".*").replace("_", ".") + "$"
    matches = regions.gdf.loc[
        regions.gdf[primary].astype(str).str.match(regex, na=False), primary
    ].tolist()
    print(f"searching {regions.source_uri} for LIKE {pattern!r} (limit {limit})")
    if not matches:
        print("no matches")
        return 0
    for hid in matches[:limit]:
        print(hid)
    print(
        f"\n{min(len(matches), limit)} matches"
        + (" (limit reached)" if len(matches) > limit else "")
    )
    return 0


def _check_inputs_exist(cfg: Config) -> list[str]:
    """Cheap precheck: regions.path and each weight raster must exist."""
    problems: list[str] = []
    if cfg.regions.path is not None and not Path(cfg.regions.path).exists():
        problems.append(f"regions.path does not exist: {cfg.regions.path}")
    if (
        cfg.source is not None
        and not cfg.source.path.startswith("gs://")
        and not Path(cfg.source.path).exists()
    ):
        problems.append(f"source.path does not exist: {cfg.source.path}")
    if cfg.backend.kind == "local":
        for w in cfg.weights:
            if w.name == "area":
                continue
            if w.raster is not None and not Path(w.raster).exists():
                problems.append(
                    f"weights.{w.name}.raster does not exist: {w.raster}"
                )
    return problems


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
