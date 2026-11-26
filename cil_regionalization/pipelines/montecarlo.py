"""Monte Carlo aggregation pipeline: apply weights per leaf, then statistics.

Composes the library core and adds no mathematics. The sequence:

    1. Resolve the declared tree of Monte Carlo leaves. The directory
       grammar is configuration: an ordered list of level names and the
       expected values per level. The historical batch/rcp/gcm/iam/ssp
       layout is one instance of the pattern; nothing here knows those
       names, the depth, or the order.
    2. Fail before any work if a declared leaf is missing on disk.
    3. Per leaf: read the frame and run `apply.apply_weights` with the
       configured kind, weight, versions, and policies. Coverage and mass
       balance run per leaf exactly as they do for a direct library call,
       and a failing leaf fails the run. Optionally write the aggregated
       leaf, mirrored under the output directory, in the layout of the
       `leaves` module. Then take the window
       mean (`stats.window_means`) over the configured windows, collapsing
       the time dimension to the window count before anything is pooled,
       and attach the leaf's level values as columns to the reduced rows.
    4. Concatenate the reduced leaves and run `stats.pooled_statistics`
       once, with the window label as an ordinary identity dimension;
       write the combined statistics plus a pipeline manifest recording
       every leaf and its accounting. The rows are sorted (`_ordered`),
       whatever order the leaves were pooled in.

Passthrough: leaving the ``[weights]`` section out entirely runs the
same tree walk, subtraction, window reduction, and pooled statistics
with no spatial aggregation at all. The leaf's own regions are the
output rows, keyed by the column ``data.region_col`` names (there is
no artifact to name it), and no coverage or mass-balance accounting
exists because no weights were applied. This produces the statistics
one step before any aggregated product: if a pattern in an aggregated
map looks odd, the passthrough twin says whether it came from the
weighting or from the data. Ratios stay out of passthrough: dividing
numerator by denominator happens at an aggregation target, and without
weights there is no such step.

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

Memory at scale, measured rather than estimated: a full mortality ADM2
run (about 4,000 leaves by 42,726 targets, roughly 500 million member
rows) peaked at 188.3 GB resident in the parent process at the
pooled-statistics stage. Requests of 128 GB died three times at that
scale before the instrumented run produced the number. Size ADM2-scale
requests at 224 GB or more, and read the memory[...] lines the run
prints before trusting any smaller figure.

Dry run resolves the tree and reports the plan (leaves, their paths,
missing entries, output locations) without creating or writing anything.

Stages. ``--stage aggregate`` runs steps 1 to 3 without the window mean
and stops. It writes the leaf files and their manifest,
``leaves/<stem>.leaves.manifest.json``, and no statistics. The manifest
is written before the first leaf, marked incomplete, and again when
every leaf has succeeded. The default stage, ``all``, runs everything
in one process.

Per-capita to total conversion is deliberately outside: leaves must
already carry the extensive quantity to aggregate. The hook for such a
conversion is the step that produces the tree this pipeline reads.

Parallelism is explicit: ``run.n_workers`` above 1 processes leaves in a
multiprocessing pool. The weights artifact is shipped to workers per
task, which is acceptable at current artifact sizes and revisitable if
profiling ever says otherwise.

Entry point: ``python -m cil_regionalization.pipelines.montecarlo
<config.toml> [--dry-run] [--stage all|aggregate]``. Job submission is
out of scope; a scheduler template that calls this entry point lives
under ``examples/``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from cil_regionalization import leaves as leaf_format
from cil_regionalization.apply import WeightsArtifact, apply_weights
from cil_regionalization.config import SourceUnitPolicies, _Strict
from cil_regionalization.manifest import collect_package_versions
from cil_regionalization.model_weights import ModelWeights, read_model_weights
from cil_regionalization.stats import (
    _quantile_label,
    _window_label,
    check_quantile_method,
    check_summation,
    pooled_statistics,
    window_means,
)


VariableKind = Literal["extensive", "intensive", "ratio"]
Stage = Literal["all", "aggregate"]


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
    subtract_filename: Optional[str] = None
    expect: dict[str, list[str]]
    exclude: list[list[str]] = Field(default_factory=list)

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
        if self.subtract_filename == self.filename:
            raise ValueError(
                "tree.subtract_filename equals tree.filename; subtracting a "
                "file from itself is always zero"
            )
        for cell in self.exclude:
            if len(cell) != len(self.levels):
                raise ValueError(
                    f"tree.exclude entry {cell} must have one value per "
                    f"level {self.levels}"
                )
            for level, value in zip(self.levels, cell):
                if value not in self.expect[level]:
                    raise ValueError(
                        f"tree.exclude entry {cell}: {value!r} is not a "
                        f"declared value of level {level!r}"
                    )
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
    region_labels: Optional[str] = None
    region_col: Optional[str] = None

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
        if self.format != "netcdf" and self.region_labels is not None:
            raise ValueError(
                "data.region_labels is only meaningful for "
                "data.format='netcdf'"
            )
        if self.format != "netcdf" and self.region_col is not None:
            raise ValueError(
                "data.region_col is only meaningful for data.format="
                "'netcdf'; parquet and csv leaves already carry their key "
                "columns under their own names"
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
    # Adds an n_members column to the statistics file: the members each
    # statistic was computed from. Off by default, so the file keeps its
    # columns unless a recipe asks.
    include_n_members: bool = False
    # How a quantile is computed. "linear" interpolates between members
    # and has no weighted form, "lower_step" is the extraction tool's
    # step, with equal weights when there are no model weights, and
    # "inverted_cdf" is the standard inverse of the empirical
    # distribution. How a mean is summed. "fast" is the vectorised sum,
    # whose last bits depend on the order the workers hand their leaves
    # back in, and "exact" is the exactly rounded one. Neither has a
    # default, and the manifest records both. See
    # `stats.summarize_samples`.
    quantile_method: Literal["linear", "lower_step", "inverted_cdf"]
    summation: Literal["fast", "exact"]
    model_weights: Optional[str] = None
    model_weight_level: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _former_values_are_named(cls, data):
        # Both options were once "legacy". The message says which value
        # it meant for this config.
        if isinstance(data, dict):
            if data.get("quantile_method") == "legacy":
                check_quantile_method(
                    "legacy", weighted=data.get("model_weights") is not None
                )
            if data.get("summation") == "legacy":
                check_summation("legacy")
        return data

    @model_validator(mode="after")
    def _weights_come_together(self) -> "StatsConfig":
        if (self.model_weights is None) != (self.model_weight_level is None):
            raise ValueError(
                "stats.model_weights and stats.model_weight_level come "
                "together: the file says what the weights are, the level "
                "says which sample dimension they attach to"
            )
        # "linear" with model weights is refused here, when the config is
        # read, and not after the leaves
        check_quantile_method(
            self.quantile_method, weighted=self.model_weights is not None
        )
        if (self.model_weight_level is not None
                and self.model_weight_level not in self.sample_dims):
            raise ValueError(
                f"stats.model_weight_level {self.model_weight_level!r} must "
                f"be one of stats.sample_dims {self.sample_dims}"
            )
        return self


class PipelineOutputConfig(_Strict):
    """Where results land. ``stem`` names the statistics and manifest
    files, so several runs can share one directory. ``write_reduced``
    keeps the pooled window-mean frame (one row per target, window,
    and sample member), which downstream regrouping needs; per-draw
    quantities cannot be recovered from the statistics file.
    ``write_leaves`` keeps every aggregated leaf with its years, under
    ``leaves/``. It is off unless asked for, because the leaves take
    far more disk than the statistics."""

    dir: str = Field(min_length=1)
    write_leaves: bool = False
    stem: str = "statistics"
    write_reduced: bool = False


class RunConfig(_Strict):
    n_workers: int = Field(default=1, ge=1)


class MonteCarloPipelineConfig(_Strict):
    """``weights`` is optional: without it the pipeline runs in
    passthrough, producing statistics per source region with no spatial
    aggregation (see the module docstring)."""

    weights: Optional[WeightsRef] = None
    tree: TreeConfig
    data: DataConfig
    stats: StatsConfig
    output: PipelineOutputConfig
    policies: SourceUnitPolicies = Field(default_factory=SourceUnitPolicies)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def _passthrough_is_coherent(self) -> "MonteCarloPipelineConfig":
        if self.weights is None:
            if self.data.kind == "ratio" or self.data.denominator_col is not None:
                raise ValueError(
                    "a ratio needs [weights]: dividing numerator by "
                    "denominator happens at the aggregation target, and "
                    "without weights there is no aggregation step to do "
                    "it in"
                )
            if self.data.format == "netcdf" and self.data.region_col is None:
                raise ValueError(
                    "without [weights] there is no artifact to name the "
                    "region column; declare data.region_col (the "
                    "projection trees key regions as 'hierid')"
                )
            if self.policies != SourceUnitPolicies():
                raise ValueError(
                    "[policies] governs how source units meet the weights; "
                    "without [weights] there is nothing for it to govern"
                )
        elif self.data.region_col is not None:
            raise ValueError(
                "data.region_col is only for passthrough runs without "
                "[weights]; with an artifact the source key column comes "
                "from the artifact schema"
            )
        return self

    @model_validator(mode="after")
    def _subtract_has_no_denominator(self) -> "MonteCarloPipelineConfig":
        if (self.tree.subtract_filename is not None
                and self.data.denominator_col is not None):
            raise ValueError(
                "tree.subtract_filename with data.denominator_col is not "
                "supported: differencing a ratio's numerator and denominator "
                "is a different operation from differencing a value"
            )
        if (self.stats.model_weight_level is not None
                and self.stats.model_weight_level not in self.tree.levels):
            raise ValueError(
                f"stats.model_weight_level {self.stats.model_weight_level!r} "
                f"must be a tree level {self.tree.levels}"
            )
        return self


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
    subtract_path: Path | None = None

    def labels(self, levels: list[str]) -> dict[str, str]:
        return dict(zip(levels, self.values))


@dataclass(frozen=True)
class PipelinePlan:
    """What a run would do. Produced by dry run; consumed by execution."""

    leaves: list[Leaf]
    missing: list[Leaf]
    stats_path: Path
    manifest_path: Path
    # set when the run writes leaf files in the layout of `leaves`
    leaves_manifest_path: Path | None = None
    stage: str = "all"

    def summary(self) -> str:
        lines = [f"pipeline plan: {len(self.leaves)} declared leaves"]
        if self.stage != "aggregate":
            lines += [
                f"  statistics -> {self.stats_path}",
                f"  manifest   -> {self.manifest_path}",
            ]
        if self.leaves_manifest_path is not None:
            lines.append(f"  leaves     -> {self.leaves_manifest_path}")
        if self.missing:
            lines.append(
                f"  MISSING {len(self.missing)} declared leaves, e.g. "
                f"{[str(m.path) for m in self.missing[:5]]}"
            )
        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Statistics frame plus per-leaf accounting for a completed run.
    ``stats`` is None after the aggregate stage, which takes none."""

    stats: Optional[pd.DataFrame]
    leaf_reports: list[dict]
    paths: dict[str, str] = field(default_factory=dict)


def resolve_plan(
    cfg: MonteCarloPipelineConfig, stage: Stage = "all"
) -> PipelinePlan:
    """Expand the declared tree into leaves; report missing ones.

    Touches nothing: no directories are created and no files written.
    """
    root = Path(cfg.tree.root)
    out_root = Path(cfg.output.dir)
    value_lists = [cfg.tree.expect[level] for level in cfg.tree.levels]
    excluded = {tuple(cell) for cell in cfg.tree.exclude}
    leaves: list[Leaf] = []
    missing: list[Leaf] = []
    for combo in itertools.product(*value_lists):
        if tuple(combo) in excluded:
            continue
        rel = Path(*combo)
        path = root / rel / cfg.tree.filename
        subtract_path = (
            root / rel / cfg.tree.subtract_filename
            if cfg.tree.subtract_filename is not None
            else None
        )
        if not cfg.output.write_leaves:
            output_path = None
        elif cfg.weights is None:
            # passthrough has no targets to lay a leaf out by, and keeps
            # the file it always wrote
            output_path = (
                out_root / "leaves" / rel / f"aggregated_{cfg.tree.filename}"
            )
        else:
            output_path = (
                out_root / "leaves" / rel
                / f"{Path(cfg.tree.filename).stem}.aggregated.parquet"
            )
        leaf = Leaf(values=tuple(combo), path=path, output_path=output_path,
                    subtract_path=subtract_path)
        leaves.append(leaf)
        if not path.exists() or (
            subtract_path is not None and not subtract_path.exists()
        ):
            missing.append(leaf)
    return PipelinePlan(
        leaves=leaves,
        missing=missing,
        stats_path=out_root / f"{cfg.output.stem}.parquet",
        manifest_path=out_root / f"{cfg.output.stem}.manifest.json",
        leaves_manifest_path=(
            out_root / "leaves" / f"{cfg.output.stem}.leaves.manifest.json"
            if cfg.output.write_leaves and cfg.weights is not None
            else None
        ),
        stage=stage,
    )


def run_pipeline(
    cfg: MonteCarloPipelineConfig,
    *,
    dry_run: bool = False,
    stage: Stage = "all",
) -> PipelinePlan | PipelineResult:
    """Run the pipeline, or return the plan untouched when ``dry_run``.
    ``stage="aggregate"`` writes the leaf files and their manifest and
    stops, and the result's ``stats`` is then None."""
    _check_stage(cfg, stage)
    plan = resolve_plan(cfg, stage)
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

    artifact, artifact_sha256 = (
        _load_artifact(cfg.weights.dir, cfg.weights.stem)
        if cfg.weights is not None
        else (None, None)
    )
    _check_level_names(
        cfg.tree.levels,
        list(artifact.schema.id_fields) if artifact is not None else [],
    )
    model_weights = None
    if stage != "aggregate" and cfg.stats.model_weights is not None:
        # The file is read here and nowhere else. The workers get these
        # rows, so a file edited during the run cannot give different
        # leaves different weights.
        model_weights = read_model_weights(cfg.stats.model_weights)
        # fail before the run, not after it: every declared value of the
        # weight level must have a row in the table
        declared = cfg.tree.expect[cfg.stats.model_weight_level]
        unmatched = model_weights.missing(declared)
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} values of "
                f"{cfg.stats.model_weight_level!r} have no row in the model "
                f"weights file (first: {unmatched[:5]})"
            )

    leaves_document = None
    job = {
        "reduce": stage != "aggregate",
        "artifact_sha256": artifact_sha256,
        "leaf_file": None,
    }
    if plan.leaves_manifest_path is not None:
        targets = leaf_format.TargetTable.from_frame(
            artifact.frame, artifact.schema.id_fields
        )
        identity = _aggregation_identity(cfg, artifact, artifact_sha256)
        # Nothing has been created under the output directory so far.
        # The config, the level names and the model weights are checked.
        leaves_document = _begin_leaves(cfg, plan, targets, identity)
        job["leaf_file"] = {
            # read here and sent along, because a worker that is spawned
            # imports the module afresh
            "targets_per_group": leaves_document["targets_per_group"],
            "target_digest": targets.digest,
            "aggregation": identity,
            "aggregation_digest": leaves_document["aggregation_digest"],
        }
    tasks = [
        (
            leaf,
            cfg.tree.levels,
            cfg.data,
            cfg.weights.weight if cfg.weights is not None else None,
            cfg.policies,
            cfg.stats,
            job,
        )
        for leaf in plan.leaves
    ]
    initargs = (
        cfg.weights.dir if cfg.weights is not None else None,
        cfg.weights.stem if cfg.weights is not None else None,
        model_weights,
    )
    # the log states which code ran: an editable install picks up
    # whatever is on disk at launch, so the file path plus this marker
    # ties a run to a pipeline build
    print(f"pipeline code: {Path(__file__).resolve()} "
          f"(worker-init, per-leaf weights, memory instrumentation)",
          flush=True)
    _log_memory("start")

    if cfg.run.n_workers == 1 or len(tasks) <= 1:
        _init_worker(*initargs)
        outcomes = [_process_leaf(t) for t in tasks]
    else:
        # The artifact loads once per worker (initializer), not once per
        # task: shipping it in every task pickle made long-lived workers
        # churn through thousands of unpickled copies, and their resident
        # memory grew with leaves processed. maxtasksperchild recycles
        # workers to bound what churn remains; chunksize=1 so recycling
        # counts actual leaves. imap_unordered streams results so memory
        # can be logged as the run accumulates.
        outcomes = []
        worker_peak = 0.0
        with mp.Pool(
            processes=cfg.run.n_workers,
            initializer=_init_worker,
            initargs=initargs,
            maxtasksperchild=16,
        ) as pool:
            for outcome in pool.imap_unordered(_process_leaf, tasks,
                                               chunksize=1):
                outcomes.append(outcome)
                worker_peak = max(worker_peak,
                                  outcome.get("worker_peak_gb") or 0.0)
                if len(outcomes) % 250 == 0 or len(outcomes) == len(tasks):
                    _log_memory(
                        f"leaves {len(outcomes)}/{len(tasks)}, "
                        f"max worker peak {worker_peak:.1f}G"
                    )
    _log_memory("all leaves collected")

    failures = [o for o in outcomes if o["error"] is not None]
    if failures:
        done = len(outcomes) - len(failures)
        # thousands of leaves usually fail the same way; report each
        # distinct error once with a count, not one message per leaf
        by_error: dict[str, list] = {}
        for f in failures:
            by_error.setdefault(f["error"][:400], []).append(f["leaf"])
        detail = "; ".join(
            f"{len(leaves)} leaves (first {'/'.join(leaves[0])}): {err}"
            for err, leaves in list(by_error.items())[:3]
        )
        raise ValueError(
            f"pipeline: partial run, {done} of {len(outcomes)} leaves "
            f"succeeded and {len(failures)} failed with "
            f"{len(by_error)} distinct errors ({detail}). No statistics "
            f"were written."
        )

    files = {tuple(o["leaf"]): o.pop("leaf_file") for o in outcomes}
    if leaves_document is not None:
        _finish_leaves(plan, leaves_document, outcomes, files)
    if stage == "aggregate":
        for o in outcomes:
            o.pop("frame")
        return PipelineResult(
            stats=None,
            leaf_reports=outcomes,
            paths={"leaves_manifest": str(plan.leaves_manifest_path)},
        )

    # Leaves arrive already reduced to window means (see _process_leaf),
    # so the pooled frame is targets x dims x windows per leaf, not
    # targets x dims x years. One pooled-statistics call serves every
    # window: the window label is an ordinary identity dimension.
    combined = pd.concat([o.pop("frame") for o in outcomes], ignore_index=True)
    _log_memory(f"after concat ({len(combined)} rows)")
    weight_col = "model_weight" if model_weights is not None else None
    stats = pooled_statistics(
        combined,
        sample_dims=cfg.stats.sample_dims,
        value_col=cfg.data.value_col,
        quantiles=cfg.stats.quantiles,
        include_mean=cfg.stats.include_mean,
        weight_col=weight_col,
        include_n_members=cfg.stats.include_n_members,
        quantile_method=cfg.stats.quantile_method,
        summation=cfg.stats.summation,
    )
    # Passthrough leaves dictionary-encode their region and window
    # columns for the pooled frame (see _process_leaf); the statistics
    # file goes out as plain strings either way, matching the schema of
    # every aggregated product.
    for col in stats.columns:
        if isinstance(stats[col].dtype, pd.CategoricalDtype):
            stats[col] = stats[col].astype(str)
    stats = _ordered(stats, cfg)
    _log_memory("after pooled statistics")

    out_root = Path(cfg.output.dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(plan.stats_path, index=False)
    reduced_path = None
    if cfg.output.write_reduced:
        reduced_path = out_root / f"{cfg.output.stem}.reduced.parquet"
        # sorted for the file only, after the statistics: what was
        # pooled, and in what order, is as it was
        _ordered(combined, cfg).to_parquet(reduced_path, index=False)
    _log_memory("after writes")
    manifest = {
        "leaves": outcomes,
        "n_leaves": len(outcomes),
        "stage": stage,
        "leaves_manifest": (
            str(plan.leaves_manifest_path)
            if plan.leaves_manifest_path is not None
            else None
        ),
        "artifact_sha256": artifact_sha256,
        "weight": cfg.weights.weight if cfg.weights is not None else None,
        "spatial_aggregation": (
            "weighted" if cfg.weights is not None
            else "none: passthrough per source region"
        ),
        "kind": cfg.data.kind,
        "data_version": cfg.data.version,
        "artifact_source_version": (
            artifact.source_version if artifact is not None else None
        ),
        "artifact_regions_version": (
            artifact.regions_version if artifact is not None else None
        ),
        "sample_dims": cfg.stats.sample_dims,
        "windows": [_window_label(lo, hi) for lo, hi in cfg.stats.windows],
        "quantiles": cfg.stats.quantiles,
        "quantile_method": cfg.stats.quantile_method,
        "summation": cfg.stats.summation,
        "reduced": str(reduced_path) if reduced_path else None,
        "model_weights": cfg.stats.model_weights,
        "model_weight_level": cfg.stats.model_weight_level,
        # what the leaves were given, not what the config declares
        "model_weights_record": (
            model_weights.record(
                o["leaf"][cfg.tree.levels.index(cfg.stats.model_weight_level)]
                for o in outcomes
            )
            if model_weights is not None
            else None
        ),
        "subtract_filename": cfg.tree.subtract_filename,
        "excluded_leaves": [list(c) for c in cfg.tree.exclude],
        "package_versions": collect_package_versions(),
    }
    plan.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return PipelineResult(
        stats=stats,
        leaf_reports=outcomes,
        paths={
            "statistics": str(plan.stats_path),
            "manifest": str(plan.manifest_path),
            **(
                {"leaves_manifest": str(plan.leaves_manifest_path)}
                if plan.leaves_manifest_path is not None
                else {}
            ),
        },
    )


def _check_stage(cfg: MonteCarloPipelineConfig, stage: str) -> None:
    if stage not in ("all", "aggregate"):
        raise ValueError(
            f"unknown stage {stage!r}; expected 'all' or 'aggregate'"
        )
    if stage == "aggregate":
        if cfg.weights is None:
            raise ValueError(
                "stage 'aggregate' needs [weights]. A passthrough run "
                "aggregates nothing, so its leaves are the source tree."
            )
        if not cfg.output.write_leaves:
            raise ValueError(
                "stage 'aggregate' writes the leaf files and nothing else, "
                "so it needs output.write_leaves = true"
            )


def _read_bytes(location: str) -> bytes:
    if location.startswith("gs://"):
        import gcsfs

        with gcsfs.GCSFileSystem().open(location, "rb") as f:
            return f.read()
    return Path(location).read_bytes()


def _artifact_sha256(weights_dir: str, stem: str) -> dict[str, str]:
    """SHA256 of the two files of a weights artifact."""
    base = str(weights_dir).rstrip("/")
    return {
        "table": hashlib.sha256(_read_bytes(f"{base}/{stem}.parquet")).hexdigest(),
        "manifest": hashlib.sha256(
            _read_bytes(f"{base}/{stem}.manifest.json")
        ).hexdigest(),
    }


def _load_artifact(
    weights_dir: str, stem: str
) -> tuple[WeightsArtifact, dict[str, str]]:
    """The artifact and the SHA256 of its files. They are hashed before
    and after the load, and a difference raises, so the hashes are those
    of what was loaded."""
    before = _artifact_sha256(weights_dir, stem)
    artifact = WeightsArtifact.load(weights_dir, stem=stem)
    if _artifact_sha256(weights_dir, stem) != before:
        raise ValueError(
            f"the weights artifact {stem!r} in {weights_dir} changed while "
            f"it was being loaded"
        )
    return artifact, before


def _aggregation_identity(
    cfg: MonteCarloPipelineConfig, artifact: WeightsArtifact, sha256: dict
) -> dict:
    """What decides the content of an aggregated leaf, apart from the
    leaf itself. It holds no path. A path says where something was, and
    the hashes say what it was."""
    return {
        "weight": cfg.weights.weight,
        "kind": cfg.data.kind,
        "data_version": cfg.data.version,
        "data_format": cfg.data.format,
        "value_col": cfg.data.value_col,
        "denominator_col": cfg.data.denominator_col,
        "region_dim": cfg.data.region_dim,
        "region_labels": cfg.data.region_labels,
        "policies": cfg.policies.model_dump(),
        "filename": cfg.tree.filename,
        "subtract_filename": cfg.tree.subtract_filename,
        "levels": list(cfg.tree.levels),
        "artifact": {
            "source_version": artifact.source_version,
            "regions_version": artifact.regions_version,
            "sha256": sha256,
        },
    }


def _claimed_elsewhere(leaves_dir: Path, own: Path, files: list[str]) -> None:
    """Raise if the leaves manifest of another output stem in the
    directory lists one of ``files``, whether it is complete or not."""
    for other in sorted(leaves_dir.glob("*.leaves.manifest.json")):
        if other == own:
            continue
        document = json.loads(other.read_text())
        taken = sorted(
            {entry["file"] for entry in document["leaves"]} & set(files)
        )
        if taken:
            raise ValueError(
                f"{len(taken)} leaf files this run would write belong to the "
                f"leaves of output stem {document['stem']!r} (first: "
                f"{taken[0]}, listed in {other}). Two stems share a "
                f"directory only when their leaves differ. Use another "
                f"output.dir, or that output.stem to write them again."
            )


def _begin_leaves(
    cfg: MonteCarloPipelineConfig,
    plan: PipelinePlan,
    targets: leaf_format.TargetTable,
    identity: dict,
) -> dict:
    """Write the target table and the leaves manifest of this stem,
    marked incomplete, and return the manifest. It lists the leaf files
    the run is about to write, so that no run under another stem writes
    over them, even if this one is killed."""
    path = plan.leaves_manifest_path
    leaves_dir = path.parent
    document = {
        "complete": False,
        "format_version": leaf_format.FORMAT_VERSION,
        "stem": cfg.output.stem,
        "targets_per_group": leaf_format.TARGETS_PER_GROUP,
        "targets": {
            "file": targets.filename,
            "digest": targets.digest,
            "n_targets": len(targets),
            "id_fields": list(targets.id_fields),
        },
        "aggregation": identity,
        "aggregation_digest": leaf_format.canonical_digest(identity),
        # recorded, and never compared
        "paths": {
            "weights_dir": cfg.weights.dir,
            "weights_stem": cfg.weights.stem,
            "tree_root": cfg.tree.root,
        },
        "group_col": cfg.data.group_col,
        "leaves": [
            {
                "leaf": list(leaf.values),
                "file": leaf.output_path.relative_to(leaves_dir).as_posix(),
            }
            for leaf in plan.leaves
        ],
    }
    files = [entry["file"] for entry in document["leaves"]]
    if leaves_dir.exists():
        _claimed_elsewhere(leaves_dir, path, files)
    leaves_dir.mkdir(parents=True, exist_ok=True)
    targets.write(leaves_dir)
    leaf_format.write_json(path, document)
    try:
        # a run under another stem may have started in between
        _claimed_elsewhere(leaves_dir, path, files)
    except ValueError:
        path.unlink(missing_ok=True)
        raise
    return document


def _finish_leaves(
    plan: PipelinePlan, document: dict, outcomes: list[dict], files: dict
) -> None:
    """Write the leaves manifest again, complete, with the accounting
    of every leaf in plan order. Raises, and leaves the manifest
    incomplete, if two leaves have different columns."""
    by_leaf = {tuple(o["leaf"]): o for o in outcomes}
    schema = files[plan.leaves[0].values]["schema"]
    profiles: list[dict] = []
    rows_per_group = np.zeros(
        len(files[plan.leaves[0].values]["rows_per_group"]), dtype=np.int64
    )
    entries = []
    for leaf, claimed in zip(plan.leaves, document["leaves"]):
        written, outcome = files[leaf.values], by_leaf[leaf.values]
        if written["schema"] != schema:
            raise ValueError(
                f"leaf {'/'.join(leaf.values)} was written with columns "
                f"{written['schema']} where the first leaf has {schema}. "
                f"The leaves of one tree need the same columns and dtypes."
            )
        rows_per_group += np.asarray(written["rows_per_group"], dtype=np.int64)
        if written["profile"] not in profiles:
            profiles.append(written["profile"])
        entries.append(
            {
                **claimed,
                "sources": [str(leaf.path)]
                + ([str(leaf.subtract_path)] if leaf.subtract_path else []),
                "rows_in": outcome["rows_in"],
                "rows_out": outcome["rows_out"],
                "coverage": outcome["coverage"],
                "mass_balance_rel_error": outcome["mass_balance_rel_error"],
                "size": written["size"],
                "sha256": written["sha256"],
                "profile": profiles.index(written["profile"]),
            }
        )
    leaf_format.write_json(
        plan.leaves_manifest_path,
        {
            **document,
            "complete": True,
            "leaves": entries,
            "schema": schema,
            "rows_per_group": rows_per_group.tolist(),
            "profiles": profiles,
            "package_versions": collect_package_versions(),
        },
    )


def _check_level_names(levels: list[str], id_fields: list[str]) -> None:
    """Raise if a tree level is named like an id field of the weights
    artifact or like a column the pipeline writes itself."""
    taken = {
        **{name: "an id field of the weights artifact" for name in id_fields},
        "window": "the column of window labels",
        "statistic": "the column of statistic labels",
        "n_members": "the column stats.include_n_members writes",
        "model_weight": "the column the model weights travel in",
        leaf_format.ORDINAL_COLUMN: "the ordinal column of a leaf file",
    }
    for level in levels:
        if level in taken:
            raise ValueError(
                f"tree level {level!r} is also {taken[level]}. A tree "
                f"level becomes a column of the output, so it needs a "
                f"name of its own."
            )


def _labelled(
    frame: pd.DataFrame, labels: dict[str, str], before: Optional[str] = None
) -> pd.DataFrame:
    """``frame`` with one constant column per tree level, placed just
    before the column ``before``, or after every column if it is None."""
    columns = list(frame.columns)
    at = len(columns) if before is None else columns.index(before)
    return frame.assign(**labels)[columns[:at] + list(labels) + columns[at:]]


def _ordered(frame: pd.DataFrame, cfg: MonteCarloPipelineConfig) -> pd.DataFrame:
    """``frame`` in the order it is written in. Rows go by the
    statistic as requested, then each tree level by its declared
    values, then the window as configured, then the target's id fields
    ascending. Together those name a row, so the order is total and
    does not depend on the order the leaves were pooled in."""
    declared = {
        "statistic": (["mean"] if cfg.stats.include_mean else [])
        + [_quantile_label(q) for q in cfg.stats.quantiles],
        **{level: list(cfg.tree.expect[level]) for level in cfg.tree.levels},
        "window": [_window_label(lo, hi) for lo, hi in cfg.stats.windows],
    }
    keys = pd.DataFrame(index=frame.index)
    for column, values in declared.items():
        if column not in frame.columns:
            continue
        position = {value: i for i, value in enumerate(values)}
        col = frame[column]
        if isinstance(col.dtype, pd.CategoricalDtype):
            # one lookup per category, not per row
            of_category = np.array(
                [position.get(c, -1) for c in col.cat.categories], dtype=np.int64
            )
            codes = col.cat.codes.to_numpy()
            keys[column] = np.where(codes >= 0, of_category[codes], -1)
        else:
            keys[column] = col.map(position).fillna(-1).astype(np.int64)
        if (keys[column] < 0).any():
            unknown = sorted(set(col[keys[column] < 0].astype(str)))
            raise ValueError(
                f"column {column!r} holds values the config does not "
                f"declare: {unknown[:5]}"
            )
    for column in frame.columns:
        if column in declared or column in (
            cfg.data.value_col, "n_members", "model_weight"
        ):
            continue
        col = frame[column]
        if isinstance(col.dtype, pd.CategoricalDtype):
            # ascending by value, whatever order the categories are in
            col = col.cat.set_categories(sorted(col.cat.categories), ordered=True)
        keys[column] = col
    order = keys.sort_values(list(keys.columns), kind="stable").index
    return frame.loc[order].reset_index(drop=True)


def _peak_rss_gb() -> float:
    """This process's peak resident memory in GB."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux, bytes on macOS
    return peak / 1e6 if sys.platform.startswith("linux") else peak / 1e9


def _current_rss_gb() -> Optional[float]:
    """This process's current resident memory in GB (Linux only)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / 1e9
    except OSError:
        return None


def _cgroup_gb() -> Optional[float]:
    """The job cgroup's memory charge in GB, the number the OOM killer
    compares against the limit; includes every worker and page cache."""
    for path in ("/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path) as f:
                return int(f.read()) / 1e9
        except (OSError, ValueError):
            continue
    return None


def _log_memory(phase: str) -> None:
    parts = [f"peak={_peak_rss_gb():.1f}G"]
    current = _current_rss_gb()
    if current is not None:
        parts.insert(0, f"rss={current:.1f}G")
    cgroup = _cgroup_gb()
    if cgroup is not None:
        parts.append(f"cgroup={cgroup:.1f}G")
    print(f"memory[{phase}]: {' '.join(parts)}", flush=True)


# Per-worker state, set once by the pool initializer. The weights
# artifact and the model-weight table are identical for every leaf, so
# they load once per worker process instead of travelling inside every
# task.
_WORKER_STATE: dict = {}


def _init_worker(
    weights_dir: Optional[str],
    weights_stem: Optional[str],
    model_weights: Optional[ModelWeights],
) -> None:
    _WORKER_STATE["artifact"], _WORKER_STATE["artifact_sha256"] = (
        _load_artifact(weights_dir, weights_stem)
        if weights_dir is not None
        else (None, None)
    )
    _WORKER_STATE["targets"] = None
    _WORKER_STATE["model_weights"] = model_weights


def _read_leaf(
    path: Path, data_cfg: DataConfig, artifact: Optional["WeightsArtifact"]
) -> pd.DataFrame:
    """Read one leaf in the declared format; no inference from extensions."""
    if data_cfg.format == "parquet":
        return pd.read_parquet(path)
    if data_cfg.format == "csv":
        return pd.read_csv(path)
    # netcdf: lazy import so the extra stays optional for parquet users.
    from cil_regionalization.netcdf_io import read_netcdf_leaf

    if artifact is not None:
        source_keys = list(artifact.schema.source_units.key_columns)
        if len(source_keys) != 1:
            raise ValueError(
                f"NetCDF leaves map one region dimension onto one source key "
                f"column; this artifact keys sources by {source_keys}"
            )
        region_col = source_keys[0]
    else:
        # passthrough: no artifact to name the key; config declares it
        region_col = data_cfg.region_col
    variables = [data_cfg.value_col]
    if data_cfg.denominator_col is not None:
        variables.append(data_cfg.denominator_col)
    return read_netcdf_leaf(
        path,
        variables=variables,
        region_dim=data_cfg.region_dim,
        region_col=region_col,
        region_labels=data_cfg.region_labels,
        kind=data_cfg.kind,
    )


def _subtract_leaf(
    frame: pd.DataFrame, path: Path, data_cfg: DataConfig, artifact: "WeightsArtifact"
) -> pd.DataFrame:
    """Difference a second file's values from the first, row for row.

    The effect of climate change in the projection trees is the
    scenario file minus its histclim sibling; this performs that
    subtraction before any weighting. The two files must cover the
    same rows: every key of one present in the other, no duplicates.
    """
    other = _read_leaf(path, data_cfg, artifact)
    keys = [c for c in frame.columns if c != data_cfg.value_col]
    if sorted(other.columns) != sorted(frame.columns):
        raise ValueError(
            f"subtract file {path} has columns {sorted(other.columns)}; "
            f"expected {sorted(frame.columns)}"
        )
    merged = frame.merge(
        other, on=keys, suffixes=("", "_subtract"), validate="one_to_one"
    )
    if len(merged) != len(frame):
        raise ValueError(
            f"subtract file {path} covers {len(merged)} of {len(frame)} "
            f"rows; the two files must cover the same regions and years"
        )
    merged[data_cfg.value_col] = (
        merged[data_cfg.value_col].astype("float64")
        - merged[f"{data_cfg.value_col}_subtract"].astype("float64")
    )
    return merged.drop(columns=[f"{data_cfg.value_col}_subtract"])


def _process_leaf(task: tuple) -> dict:
    """Apply weights to one leaf (unless passthrough), then reduce it
    to window means unless the stage stops before that. The leaf's
    level values go on the reduced rows, never on the frame the weights
    are applied to. An aggregated leaf file has them in its footer and
    its path, and a passthrough one as columns.

    Returns accounting, never raises. Module-level and tuple-argument so
    a multiprocessing pool can run it; errors come back as strings so
    the parent can report a partial run coherently instead of losing the
    traceback to a worker crash. The optional per-leaf output on disk is
    the full aggregated frame (years intact); only the in-memory frame
    handed back for pooling is reduced.
    """
    leaf, levels, data_cfg, weight, policies, stats_cfg, job = task
    artifact = _WORKER_STATE["artifact"]
    try:
        if _WORKER_STATE["artifact_sha256"] != job["artifact_sha256"]:
            raise ValueError(
                "the weights artifact this worker loaded is not the one the "
                "run started with. Its files changed during the run."
            )
        frame = _read_leaf(leaf.path, data_cfg, artifact)
        if leaf.subtract_path is not None:
            frame = _subtract_leaf(frame, leaf.subtract_path, data_cfg, artifact)
        labels = leaf.labels(levels)
        for name in labels:
            if name in frame.columns:
                raise ValueError(
                    f"leaf already has a column named {name!r}; tree level "
                    f"names must not collide with leaf columns"
                )

        if artifact is None:
            # passthrough: the leaf's own regions are the product rows,
            # and there is no coverage or mass accounting because no
            # weights were applied
            out_frame = frame
            coverage = None
            rel_error = None
        else:
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
            out_frame = applied.frame
            coverage = applied.coverage.counts
            balance = applied.mass_balance
            rel_error = balance.rel_error if balance is not None else None
        written = None
        if leaf.output_path is not None and artifact is None:
            # the labels follow the leaf's own columns in the file
            leaf.output_path.parent.mkdir(parents=True, exist_ok=True)
            _labelled(out_frame, labels).to_parquet(leaf.output_path, index=False)
        elif leaf.output_path is not None:
            written = leaf_format.write_leaf(
                out_frame,
                leaf.output_path,
                targets=_worker_targets(artifact, job["leaf_file"]),
                value_col=data_cfg.value_col,
                # Spelled out as well as digested. A footer cannot be
                # added to later, and a leaf has to say what made it
                # when its manifest is gone.
                identity={
                    "aggregation": job["leaf_file"]["aggregation"],
                    "aggregation_digest": job["leaf_file"]["aggregation_digest"],
                    "levels": labels,
                    "sources": [str(leaf.path)]
                    + ([str(leaf.subtract_path)] if leaf.subtract_path else []),
                },
                targets_per_group=job["leaf_file"]["targets_per_group"],
            ).as_dict()
        if not job["reduce"]:
            return {
                "leaf": list(leaf.values),
                "path": str(leaf.path),
                "rows_in": int(len(frame)),
                "rows_out": int(len(out_frame)),
                "rows_reduced": None,
                "coverage": coverage,
                "mass_balance_rel_error": rel_error,
                "output": str(leaf.output_path),
                "worker_peak_gb": round(_peak_rss_gb(), 2),
                "error": None,
                "leaf_file": written,
                "frame": None,
            }

        # The labels are constant over a leaf. On the frame they would be
        # grouping keys of every row, in the weights and again in the
        # window means, so they go on the reduced rows. A passthrough
        # leaf of the time and the value alone has nothing else to
        # group by, and keeps its labels first.
        labels_first = all(
            c in (stats_cfg.time_col, data_cfg.value_col)
            for c in out_frame.columns
        )
        reduced = window_means(
            _labelled(out_frame, labels) if labels_first else out_frame,
            time_col=stats_cfg.time_col,
            windows=stats_cfg.windows,
            value_col=data_cfg.value_col,
            summation=stats_cfg.summation,
        )
        if not labels_first:
            reduced = _labelled(reduced, labels, "window")
        if artifact is None:
            # Dictionary-encode the two high-cardinality repeated string
            # columns before thousands of reduced leaves concatenate in
            # the parent. Categories come out sorted, so every leaf of
            # one tree shares the same dtype and concat keeps it.
            if (data_cfg.region_col is not None
                    and data_cfg.region_col in reduced.columns):
                reduced[data_cfg.region_col] = (
                    reduced[data_cfg.region_col].astype("category")
                )
            reduced["window"] = reduced["window"].astype("category")
        if _WORKER_STATE.get("model_weights") is not None:
            # one climate model per leaf, so its weight is a constant
            # column here; attaching it per leaf replaces a merge over
            # the full combined frame that dominated end-of-run memory
            value = labels[stats_cfg.model_weight_level]
            reduced["model_weight"] = _WORKER_STATE["model_weights"].weight_of(
                value
            )

        return {
            "leaf": list(leaf.values),
            "path": str(leaf.path),
            "rows_in": int(len(frame)),
            "rows_out": int(len(out_frame)),
            "rows_reduced": int(len(reduced)),
            "coverage": coverage,
            "mass_balance_rel_error": rel_error,
            "output": str(leaf.output_path) if leaf.output_path else None,
            "worker_peak_gb": round(_peak_rss_gb(), 2),
            "error": None,
            "leaf_file": written,
            "frame": reduced,
        }
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        return {
            "leaf": list(leaf.values),
            "path": str(leaf.path),
            "error": str(e),
            "leaf_file": None,
            "frame": None,
        }


def _worker_targets(
    artifact: WeightsArtifact, leaf_file: dict
) -> leaf_format.TargetTable:
    """The target table of this worker's artifact, built once. Raises
    if it is not the table the run wrote down before the first leaf."""
    if _WORKER_STATE["targets"] is None:
        targets = leaf_format.TargetTable.from_frame(
            artifact.frame, artifact.schema.id_fields
        )
        if targets.digest != leaf_file["target_digest"]:
            raise ValueError(
                "the targets of the weights artifact this worker loaded are "
                "not those the run started with"
            )
        _WORKER_STATE["targets"] = targets
    return _WORKER_STATE["targets"]


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
    parser.add_argument(
        "--stage",
        choices=["all", "aggregate"],
        default="all",
        help="aggregate writes the leaf files and stops; all (the default) "
        "runs everything",
    )
    args = parser.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    if args.dry_run:
        plan = run_pipeline(cfg, dry_run=True, stage=args.stage)
        print(plan.summary())
        return 1 if plan.missing else 0

    result = run_pipeline(cfg, stage=args.stage)
    print(
        f"pipeline: {len(result.leaf_reports)} leaves aggregated; "
        + (
            f"statistics rows={len(result.stats)}"
            if result.stats is not None
            else "no statistics at stage aggregate"
        )
    )
    for kind_, path in result.paths.items():
        print(f"  wrote {kind_}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
