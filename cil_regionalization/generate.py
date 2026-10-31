"""Weight generation as a function call.

`generate_weights` is what `cilreg run` does, minus the printing and the
test-mode and legacy-csv options. Both go through `compute_and_write`,
so a config gives the same weights file and manifest either way.

A run whose weights fail their own sum-to-one check raises
`SumToOneError` before anything is written. The check is the only
thing standing between a bad file and someone applying it later, so a
file that failed it must not exist.
"""
from __future__ import annotations

from pathlib import Path

from cil_regionalization.backends.base import WeightsResult
from cil_regionalization.config import Config, load_config
from cil_regionalization.grid import GridSpec
from cil_regionalization.io import write_result
from cil_regionalization.regions import RegionSet
from cil_regionalization.validate import SumToOneReport
from cil_regionalization.weights import from_config_list


class SumToOneError(ValueError):
    """The generated weights failed the sum-to-one check.

    ``report`` is the full `SumToOneReport`, failures included.
    ``result`` is the computed `WeightsResult`; nothing was written.
    """

    def __init__(self, report: SumToOneReport, result: WeightsResult):
        super().__init__(report.summary())
        self.report = report
        self.result = result


def check_inputs_exist(cfg: Config) -> list[str]:
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


def compute_and_write(
    cfg: Config, regions: RegionSet
) -> tuple[WeightsResult, dict[str, str]]:
    """Run the configured backend on ``regions`` and write the result.

    Returns the result and the written paths keyed by kind. Raises
    `SumToOneError` before writing when the sum-to-one check fails.
    """
    grid = GridSpec.from_config(cfg.grid) if cfg.grid is not None else None
    weight_specs = from_config_list(cfg.weights)

    if cfg.backend.kind == "bigquery":
        from cil_regionalization.backends.bigquery import BigQueryBackend

        backend = BigQueryBackend()
    else:
        from cil_regionalization.backends.local import LocalBackend

        backend = LocalBackend()

    result = backend.compute(regions, grid, weight_specs, cfg)
    if not result.sum_report.ok:
        raise SumToOneError(result.sum_report, result)
    paths = write_result(result, cfg.output.dir, cfg.output.format)
    return result, paths


def generate_weights(config: Config | str | Path) -> WeightsResult:
    """Generate a weights file and its manifest from a config.

    ``config`` is a `Config` or the path to a TOML file. The weights
    parquet and the manifest are written to ``config.output.dir``, as
    `cilreg run` does, and the result is returned. Geometries are read
    from the paths in the config; to use polygons held in memory, write
    them to a GeoParquet file first so the manifest can record its hash.

    Raises `FileNotFoundError` naming any input that is missing, and
    `SumToOneError` when the weights fail their sum-to-one check, in
    which case nothing is written.
    """
    cfg = config if isinstance(config, Config) else load_config(config)
    problems = check_inputs_exist(cfg)
    if problems:
        raise FileNotFoundError("; ".join(problems))
    regions = RegionSet.from_config(cfg.regions)
    result, _ = compute_and_write(cfg, regions)
    return result
