"""Monte Carlo aggregation pipeline: apply weights per leaf, then statistics.

Composes the library core and adds no mathematics. The sequence:

    1. Resolve the declared tree of Monte Carlo leaves. The directory
       grammar is configuration: an ordered list of level names and the
       expected values per level. The historical batch/rcp/gcm/iam/ssp
       layout is one instance of the pattern; nothing here knows those
       names, the depth, or the order.
    2. Fail before any work if a declared leaf is missing on disk.
    3. Per leaf: read the frame, attach the leaf's level values as
       columns, run `apply.apply_weights` with the configured kind,
       weight, versions, and policies. Coverage and mass balance run per
       leaf exactly as they do for a direct library call, and a failing
       leaf fails the run. Optionally write the aggregated leaf, mirrored
       under the output directory. Then take the window mean
       (`stats.window_means`) over the configured windows, collapsing the
       time dimension to the window count before anything is pooled.
    4. Concatenate the reduced leaves and run `stats.pooled_statistics`
       once, with the window label as an ordinary identity dimension;
       write the combined statistics plus a pipeline manifest recording
       every leaf and its accounting.

The window mean moves before concatenation purely for memory: it is
computed per sample member either way, so the staging is provably
equivalent (the equivalence is a pinned test, not an assumption) while
shrinking the pooled frame by the ratio of years to windows, about 45
fold for 90 years and two windows. The consequence is that the window
bounds are final before any leaf is processed; the reduced frames carry
no time dimension, so a different window means a rerun, and
`stats.window_means` refuses already reduced input rather than silently
relabeling it. The statistics ordering itself is unchanged: spatial
aggregation per draw, then window mean, then statistics over the pooled
sample.

A run either completes every declared leaf or raises naming what
finished and what did not; there is no silently partial output. The
statistics file is written only after every leaf succeeded.

Dry run resolves the tree and reports the plan (leaves, their paths,
missing entries, output locations) without creating or writing anything.

Per-capita to total conversion is deliberately outside: leaves must
already carry the extensive quantity to aggregate. The hook for such a
conversion is the step that produces the tree this pipeline reads.

Parallelism is explicit: ``run.n_workers`` above 1 processes leaves in a
multiprocessing pool. The weights artifact is shipped to workers per
task, which is acceptable at current artifact sizes and revisitable if
profiling ever says otherwise.

Entry point: ``python -m cil_regionalization.pipelines.montecarlo
<config.toml> [--dry-run]``. Job submission is out of scope; a scheduler
template that calls this entry point lives under ``examples/``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import Field, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from cil_regionalization.apply import WeightsArtifact, apply_weights
from cil_regionalization.config import SourceUnitPolicies, _Strict
from cil_regionalization.stats import _window_label, pooled_statistics, window_means


VariableKind = Literal["extensive", "intensive", "ratio"]


class WeightsRef(_Strict):
    """Where the weights artifact lives and which weight to apply."""

    dir: str = Field(min_length=1)
    stem: str = "weights"
    weight: str = Field(min_length=1)


class TreeConfig(_Strict):
    """Declared layout of the Monte Carlo tree.

    ``levels`` names the directory levels in path order; each level
    becomes a column on the aggregated output. ``expect`` declares the
    values per level, and the expected leaf set is their full cross
    product: a leaf path is ``root/<v1>/.../<vN>/filename``. Anything
    declared but absent on disk fails the run before any processing.
    """

    root: str = Field(min_length=1)
    levels: list[str] = Field(min_length=1)
    filename: str = Field(min_length=1)
    expect: dict[str, list[str]]

    @model_validator(mode="after")
    def _expect_covers_levels(self) -> "TreeConfig":
        if set(self.expect) != set(self.levels):
            raise ValueError(
                f"tree.expect keys {sorted(self.expect)} must match "
                f"tree.levels {self.levels} exactly"
            )
        for level, values in self.expect.items():
            if not values:
                raise ValueError(f"tree.expect.{level} must list at least one value")
        return self


class DataConfig(_Strict):
    """What a leaf contains and how to aggregate it.

    ``format`` is declared, never inferred from the file extension:
    inference invites a mismatch between what a user thinks is being
    read and what is read. For ``netcdf`` leaves (the ``[netcdf]``
    optional extra), ``value_col`` names the data variable and
    ``region_dim`` names the file's region dimension, which the reader
    maps to the weights artifact's source key column. Every other
    dimension passes through exactly as a parquet column would.
    """

    format: Literal["parquet", "csv", "netcdf"] = "parquet"
    value_col: str = "value"
    denominator_col: Optional[str] = None
    kind: VariableKind
    version: str = Field(min_length=1)
    group_col: Optional[str] = None
    region_dim: Optional[str] = None

    @model_validator(mode="after")
    def _region_dim_matches_format(self) -> "DataConfig":
        if self.format == "netcdf" and self.region_dim is None:
            raise ValueError(
                "data.format='netcdf' requires data.region_dim: the file's "
                "region dimension name is declared, not assumed"
            )
        if self.format != "netcdf" and self.region_dim is not None:
            raise ValueError(
                "data.region_dim is only meaningful for data.format='netcdf'; "
                "parquet and csv leaves must already carry the artifact's "
                "source key columns"
            )
        return self


class StatsConfig(_Strict):
    """Statistics stage.

    ``windows`` is final before any leaf is processed: each leaf is
    reduced to window means as it is aggregated, and the reduced frames
    carry no time dimension. Changing the windows afterwards is a rerun,
    not a reprocessing of reduced data, and the reduction step refuses
    already reduced input by construction.
    """

    sample_dims: list[str] = Field(min_length=1)
    time_col: str = Field(min_length=1)
    windows: list[tuple[float, float]] = Field(min_length=1)
    quantiles: list[float] = Field(default_factory=list)
    include_mean: bool = True


class PipelineOutputConfig(_Strict):
    dir: str = Field(min_length=1)
    write_leaves: bool = True


class RunConfig(_Strict):
    n_workers: int = Field(default=1, ge=1)


class MonteCarloPipelineConfig(_Strict):
    weights: WeightsRef
    tree: TreeConfig
    data: DataConfig
    stats: StatsConfig
    output: PipelineOutputConfig
    policies: SourceUnitPolicies = Field(default_factory=SourceUnitPolicies)
    run: RunConfig = Field(default_factory=RunConfig)


def load_pipeline_config(path: str | Path) -> MonteCarloPipelineConfig:
    """Load and validate a pipeline TOML; resolve relative paths.

    ``weights.dir``, ``tree.root``, and ``output.dir`` given as relative
    paths resolve against the config file's directory, matching
    `config.load_config`.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"pipeline config not found: {p}")
    with p.open("rb") as f:
        data = tomllib.load(f)

    def _resolve(value: str) -> str:
        if "://" in value or value.startswith("/"):
            return value
        return str((p.parent / value).resolve())

    for section, key in (("weights", "dir"), ("tree", "root"), ("output", "dir")):
        block = data.get(section)
        if isinstance(block, dict) and isinstance(block.get(key), str):
            block[key] = _resolve(block[key])
    return MonteCarloPipelineConfig.model_validate(data)


@dataclass(frozen=True)
class Leaf:
    """One unit of work: level values plus the resolved paths."""

    values: tuple[str, ...]
    path: Path
    output_path: Path | None

    def labels(self, levels: list[str]) -> dict[str, str]:
        return dict(zip(levels, self.values))


@dataclass(frozen=True)
class PipelinePlan:
    """What a run would do. Produced by dry run; consumed by execution."""

    leaves: list[Leaf]
    missing: list[Leaf]
    stats_path: Path
    manifest_path: Path

    def summary(self) -> str:
        lines = [
            f"pipeline plan: {len(self.leaves)} declared leaves",
            f"  statistics -> {self.stats_path}",
            f"  manifest   -> {self.manifest_path}",
        ]
        if self.missing:
            lines.append(
                f"  MISSING {len(self.missing)} declared leaves, e.g. "
                f"{[str(m.path) for m in self.missing[:5]]}"
            )
        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Statistics frame plus per-leaf accounting for a completed run."""

    stats: pd.DataFrame
    leaf_reports: list[dict]
    paths: dict[str, str] = field(default_factory=dict)


def resolve_plan(cfg: MonteCarloPipelineConfig) -> PipelinePlan:
    """Expand the declared tree into leaves; report missing ones.

    Touches nothing: no directories are created and no files written.
    """
    root = Path(cfg.tree.root)
    out_root = Path(cfg.output.dir)
    value_lists = [cfg.tree.expect[level] for level in cfg.tree.levels]
    leaves: list[Leaf] = []
    missing: list[Leaf] = []
    for combo in itertools.product(*value_lists):
        rel = Path(*combo)
        path = root / rel / cfg.tree.filename
        output_path = (
            out_root / "leaves" / rel / f"aggregated_{cfg.tree.filename}"
            if cfg.output.write_leaves
            else None
        )
        leaf = Leaf(values=tuple(combo), path=path, output_path=output_path)
        leaves.append(leaf)
        if not path.exists():
            missing.append(leaf)
    return PipelinePlan(
        leaves=leaves,
        missing=missing,
        stats_path=out_root / "statistics.parquet",
        manifest_path=out_root / "pipeline.manifest.json",
    )


def run_pipeline(
    cfg: MonteCarloPipelineConfig, *, dry_run: bool = False
) -> PipelinePlan | PipelineResult:
    """Run the pipeline, or return the plan untouched when ``dry_run``."""
    plan = resolve_plan(cfg)
    if dry_run:
        return plan
    if plan.missing:
        shown = [str(m.path) for m in plan.missing[:10]]
        raise ValueError(
            f"pipeline: {len(plan.missing)} of {len(plan.leaves)} declared "
            f"leaves are missing on disk (first {len(shown)}: {shown}). "
            f"Fix the tree or the [tree.expect] declaration; the pipeline "
            f"does not run over a partial universe."
        )

    artifact = WeightsArtifact.load(cfg.weights.dir, stem=cfg.weights.stem)
    tasks = [
        (
            leaf,
            artifact,
            cfg.tree.levels,
            cfg.data,
            cfg.weights.weight,
            cfg.policies,
            cfg.stats,
        )
        for leaf in plan.leaves
    ]

    if cfg.run.n_workers == 1 or len(tasks) <= 1:
        outcomes = [_process_leaf(t) for t in tasks]
    else:
        with mp.Pool(processes=cfg.run.n_workers) as pool:
            outcomes = pool.map(_process_leaf, tasks)

    failures = [o for o in outcomes if o["error"] is not None]
    if failures:
        done = len(outcomes) - len(failures)
        detail = "; ".join(
            f"{'/'.join(f['leaf'])}: {f['error']}" for f in failures[:5]
        )
        raise ValueError(
            f"pipeline: partial run, {done} of {len(outcomes)} leaves "
            f"succeeded and {len(failures)} failed ({detail}). No statistics "
            f"were written; per-leaf outputs of successful leaves remain "
            f"under {cfg.output.dir}."
        )

    # Leaves arrive already reduced to window means (see _process_leaf),
    # so the pooled frame is targets x dims x windows per leaf, not
    # targets x dims x years. One pooled-statistics call serves every
    # window: the window label is an ordinary identity dimension.
    combined = pd.concat([o.pop("frame") for o in outcomes], ignore_index=True)
    stats = pooled_statistics(
        combined,
        sample_dims=cfg.stats.sample_dims,
        value_col=cfg.data.value_col,
        quantiles=cfg.stats.quantiles,
        include_mean=cfg.stats.include_mean,
    )

    out_root = Path(cfg.output.dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(plan.stats_path, index=False)
    manifest = {
        "leaves": outcomes,
        "n_leaves": len(outcomes),
        "weight": cfg.weights.weight,
        "kind": cfg.data.kind,
        "data_version": cfg.data.version,
        "artifact_source_version": artifact.source_version,
        "artifact_regions_version": artifact.regions_version,
        "sample_dims": cfg.stats.sample_dims,
        "windows": [_window_label(lo, hi) for lo, hi in cfg.stats.windows],
        "quantiles": cfg.stats.quantiles,
    }
    plan.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return PipelineResult(
        stats=stats,
        leaf_reports=outcomes,
        paths={
            "statistics": str(plan.stats_path),
            "manifest": str(plan.manifest_path),
        },
    )


def _read_leaf(path: Path, data_cfg: DataConfig, artifact: "WeightsArtifact") -> pd.DataFrame:
    """Read one leaf in the declared format; no inference from extensions."""
    if data_cfg.format == "parquet":
        return pd.read_parquet(path)
    if data_cfg.format == "csv":
        return pd.read_csv(path)
    # netcdf: lazy import so the extra stays optional for parquet users.
    from cil_regionalization.netcdf_io import read_netcdf_leaf

    source_keys = list(artifact.schema.source_units.key_columns)
    if len(source_keys) != 1:
        raise ValueError(
            f"NetCDF leaves map one region dimension onto one source key "
            f"column; this artifact keys sources by {source_keys}"
        )
    variables = [data_cfg.value_col]
    if data_cfg.denominator_col is not None:
        variables.append(data_cfg.denominator_col)
    return read_netcdf_leaf(
        path,
        variables=variables,
        region_dim=data_cfg.region_dim,
        region_col=source_keys[0],
        kind=data_cfg.kind,
    )


def _process_leaf(task: tuple) -> dict:
    """Apply weights to one leaf, then reduce it to window means.

    Returns accounting, never raises. Module-level and tuple-argument so
    a multiprocessing pool can run it; errors come back as strings so
    the parent can report a partial run coherently instead of losing the
    traceback to a worker crash. The optional per-leaf output on disk is
    the full aggregated frame (years intact); only the in-memory frame
    handed back for pooling is reduced.
    """
    leaf, artifact, levels, data_cfg, weight, policies, stats_cfg = task
    try:
        frame = _read_leaf(leaf.path, data_cfg, artifact)
        for name, value in leaf.labels(levels).items():
            if name in frame.columns:
                raise ValueError(
                    f"leaf already has a column named {name!r}; tree level "
                    f"names must not collide with leaf columns"
                )
            frame[name] = value

        applied = apply_weights(
            artifact,
            frame,
            kind=data_cfg.kind,
            weight=weight,
            data_version=data_cfg.version,
            value_col=data_cfg.value_col,
            denominator_col=data_cfg.denominator_col,
            policies=policies,
            group_col=data_cfg.group_col,
        )
        if leaf.output_path is not None:
            leaf.output_path.parent.mkdir(parents=True, exist_ok=True)
            applied.frame.to_parquet(leaf.output_path, index=False)

        reduced = window_means(
            applied.frame,
            time_col=stats_cfg.time_col,
            windows=stats_cfg.windows,
            value_col=data_cfg.value_col,
        )

        balance = applied.mass_balance
        return {
            "leaf": list(leaf.values),
            "path": str(leaf.path),
            "rows_in": int(len(frame)),
            "rows_out": int(len(applied.frame)),
            "rows_reduced": int(len(reduced)),
            "coverage": applied.coverage.counts,
            "mass_balance_rel_error": (
                balance.rel_error if balance is not None else None
            ),
            "output": str(leaf.output_path) if leaf.output_path else None,
            "error": None,
            "frame": reduced,
        }
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        return {
            "leaf": list(leaf.values),
            "path": str(leaf.path),
            "error": str(e),
            "frame": None,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cil_regionalization.pipelines.montecarlo",
        description="Aggregate a Monte Carlo tree and summarize the samples.",
    )
    parser.add_argument("config", type=str, help="path to a pipeline TOML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the tree and print the plan; touch nothing",
    )
    args = parser.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    if args.dry_run:
        plan = run_pipeline(cfg, dry_run=True)
        print(plan.summary())
        return 1 if plan.missing else 0

    result = run_pipeline(cfg)
    print(
        f"pipeline: {len(result.leaf_reports)} leaves aggregated; "
        f"statistics rows={len(result.stats)}"
    )
    for kind_, path in result.paths.items():
        print(f"  wrote {kind_}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
