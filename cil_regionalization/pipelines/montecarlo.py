"""Monte Carlo aggregation pipeline, with weights applied leaf by leaf and
statistics taken over the pooled leaves.

The tree of leaves is declared in the config, as an ordered list of
level names and the values each level takes. Every declared leaf has to
be on disk before any work starts. Each leaf is read, given to
`apply.apply_weights` with the configured kind, weight, version and
policies, and reduced to its window means by `stats.window_means`. Its
level values go on the reduced rows as columns. Coverage and mass
balance are checked leaf by leaf as in a direct call, and a failing
leaf fails the run. The reduced leaves are then pooled, and
`stats.pooled_statistics` takes the statistics once, with the window
label as an ordinary identity column. The statistics file and its
manifest are written only after every leaf has succeeded, with the rows
sorted by `_ordered` whatever order the leaves were pooled in.

Without a ``[weights]`` section the run is a passthrough. The tree
walk, the subtraction, the window means and the statistics are the
same, with no spatial aggregation. The rows are the leaf's own regions,
keyed by the column ``data.region_col`` names, and there is no coverage
or mass balance. A ratio is refused, since its numerator is divided by
its denominator at an aggregation target.

In this flow each leaf is reduced to its window means as it is
aggregated, so the windows are fixed before the first leaf is read.
Other windows take a new run, or a run of stage ``reduce`` if the
leaves were kept.

Stages. ``--stage aggregate`` writes the leaf files and their manifest,
``leaves/<stem>.leaves.manifest.json``, and no statistics. The manifest
is written before the first leaf, marked incomplete, and again when
every leaf has succeeded. ``--stage reduce`` takes the statistics from
leaf files written before. It checks them against their manifest, asks
a backend of the `reduce` package for the statistics, never opens the
source tree and reads the weights artifact only to hash it. The
windows, the quantiles and everything else under ``[stats]`` can differ
from the run that wrote the leaves. The default stage, ``all``, runs
everything in one process and holds every reduced leaf at once.

Time reduction. ``stats.time_reduction`` says what happens to the
years of a member. Under "window_mean" each member's window is reduced
to its mean first, as described above. Under "pooled" no mean is
taken, and every year of a member inside a window is a member of the
sample, so the statistics are over models, draws and years together.
Under "per_year" no mean is taken either, and the year stays in the
output, so the statistics are over the members within each year of
``stats.years``. The rows of the last two are the leaves' own rows, far
too many to hold at once, so their statistics always come from the
leaf files, one block of targets at a time. Stage ``all`` then writes
the leaves, as ``aggregate`` does, and reduces them, as ``reduce``
does.

Warming levels. With ``[stats.warming_levels]`` the statistics are over
the members at a warming level in a period, where a member is a
temperature series such as a model and a scenario. The level of each
member comes from `warming_levels.assign_warming_levels`, before any
leaf is read. A leaf is then pooled under the level its member has in
a period and left out of the periods in which it has none. This also
runs from the leaf files under every stage, so one aggregation serves
every level, every period and both ways of reducing.

The leaves must already carry the extensive quantity to aggregate. A
per-capita value is turned into a total by the step that writes the
tree.

With ``run.n_workers`` above 1 the leaves are processed in a
multiprocessing pool, and each worker loads the weights artifact once.
A dry run resolves the tree and reports the plan, with the leaves,
their paths, the missing ones and the output files, and creates or
writes nothing.

The entry point, which a scheduler template under ``examples/`` calls::

    python -m cil_regionalization.pipelines.montecarlo <config.toml>
        [--dry-run] [--stage all|aggregate|reduce] [--verify-leaves]
"""

# The window mean is taken before the leaves are pooled to save memory.
# It is taken per sample member either way, so the statistics are the
# same, which a test pins, and the pooled frame shrinks by the ratio of
# years to windows.
#
# Memory at scale, measured: a full mortality ADM2 run (about 4,000
# leaves by 42,726 targets, roughly 500 million member rows) peaked at
# 188.3 GB resident in the parent process at the pooled-statistics
# stage, and requests of 128 GB died three times at that scale. Size
# ADM2-scale requests at 224 GB or more, and read the memory[...] lines
# the run prints before trusting any smaller figure.

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from cil_regionalization import leaves as leaf_format
from cil_regionalization.apply import SHARE_COLUMN, WeightsArtifact, apply_weights
from cil_regionalization.config import SourceUnitPolicies, _Strict
from cil_regionalization.manifest import collect_package_versions
from cil_regionalization.model_weights import ModelWeights, read_model_weights
from cil_regionalization.model_weights import _listed as _quoted_list
from cil_regionalization.reduce import (
    LeafRequest,
    ReduceConfig,
    ReduceRequest,
    get_backend,
)
from cil_regionalization.stats import (
    _quantile_label,
    _warn_if_unweighted_models,
    _window_label,
    check_quantile_method,
    check_summation,
    pooled_statistics,
    window_means,
)
from cil_regionalization.warming_levels import (
    GMSTTable,
    Period,
    WarmingLevels,
    _listed,
    assign_warming_levels,
    members_that_join,
    read_gmst,
)


VariableKind = Literal["extensive", "intensive", "ratio"]
# in a cell of tree.exclude, every value of the level
EVERY_VALUE = "*"
Stage = Literal["all", "aggregate", "reduce"]


class WeightsRef(_Strict):
    """Where the weights artifact lives and which weight to apply."""

    dir: str = Field(min_length=1)
    stem: str = "weights"
    weight: str = Field(min_length=1)


class TreeConfig(_Strict):
    """Declared layout of the Monte Carlo tree.

    ``levels`` names the directory levels in path order, and a level
    that is not a sample dimension becomes a column of the statistics.
    ``expect`` gives the values of each level, and the leaves are every
    combination of them, each at ``root/<v1>/.../<vN>/filename``.
    ``exclude`` takes cells out of those combinations. A cell has one
    value for each level, and ``"*"`` stands for every value of its
    level. ``subtract_filename`` names a sibling file whose values are
    taken off the leaf's, row for row. A declared leaf that is not on
    disk fails the run before any work.
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
            if all(value == EVERY_VALUE for value in cell):
                raise ValueError(f"tree.exclude entry {cell} leaves out every leaf")
            for level, value in zip(self.levels, cell):
                if value != EVERY_VALUE and value not in self.expect[level]:
                    raise ValueError(
                        f"tree.exclude entry {cell}: {value!r} is not a "
                        f"declared value of level {level!r}"
                    )
        return self


class DataConfig(_Strict):
    """What a leaf contains and how to aggregate it.

    ``format`` is declared and never inferred from the file extension.
    For ``netcdf`` leaves, which need the ``[netcdf]`` extra,
    ``value_col`` names the data variable and ``region_dim`` the file's
    region dimension. The reader turns it into the source key column of
    the weights artifact, or into ``region_col`` in a run without
    weights. Every other dimension passes through as a parquet column
    would.
    """

    # Inferring the format from the extension invites a mismatch between
    # what a user thinks is being read and what is read.
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
                "data.region_labels is only meaningful for data.format='netcdf'"
            )
        if self.format != "netcdf" and self.region_col is not None:
            raise ValueError(
                "data.region_col is only meaningful for data.format="
                "'netcdf'; parquet and csv leaves already carry their key "
                "columns under their own names"
            )
        return self


class WarmingPeriod(_Strict):
    """A calendar period of a warming level product. ``gmst_span`` is
    the years the temperature is averaged over to pick a member's
    level, and ``impact_span`` the years of the impacts. Both include
    their first and last year."""

    gmst_span: tuple[int, int]
    impact_span: tuple[int, int]


class WarmingLevelsConfig(_Strict):
    """Warming levels in place of calendar windows.

    ``gmst`` is the temperature table and ``member`` its columns that
    name a series, each of them also a tree level. Nothing that decides
    a level has a default. ``reference_period`` and ``offset`` go
    together, since the offset is the warming up to that period.
    ``smoothing`` says in words how the series in the table was
    smoothed, and "unknown" is an answer. ``join`` names the members
    that join, "declared" for every member the tree declares, or a table
    of member columns to the values that join. A member of the tree
    that does not join is pooled at no level, and the record lists it.
    """

    gmst: str = Field(min_length=1)
    member: list[str] = Field(min_length=1)
    time_col: str = "year"
    value_col: str = "temp"
    sep: str = ","
    reference_period: tuple[int, int]
    offset: float
    levels: list[float] = Field(min_length=1)
    half_width: float = Field(gt=0)
    smoothing: str = Field(min_length=1)
    join: Union[Literal["declared"], dict[str, list[str]]]
    periods: list[WarmingPeriod] = Field(min_length=1)

    @model_validator(mode="after")
    def _join_names_member_columns(self) -> "WarmingLevelsConfig":
        try:
            _listed(self.join, self.member)
        except ValueError as error:
            raise ValueError(f"stats.warming_levels.{error}") from None
        return self

    def read_table(self) -> GMSTTable:
        """The temperature table, read with the columns named here."""
        return read_gmst(
            self.gmst,
            member_cols=self.member,
            time_col=self.time_col,
            value_col=self.value_col,
            sep=self.sep,
        )

    def assign(
        self, table: GMSTTable, declared: Iterable[Sequence[str]]
    ) -> WarmingLevels:
        """The levels of the members of ``declared`` that join, with the
        settings here. Raises if a member of ``declared`` has no row in
        ``table``, even one that does not join, and as
        `members_that_join` and `assign_warming_levels` raise."""
        declared = list(dict.fromkeys(tuple(member) for member in declared))
        known = set(table.members)
        absent = [member for member in declared if member not in known]
        if absent:
            raise ValueError(
                f"{len(absent)} of the {len(declared)} members the tree "
                f"declares have no row in the temperature table {self.gmst} "
                f"(first: {dict(zip(self.member, absent[0]))}). A name is "
                f"matched as it is written."
            )
        return assign_warming_levels(
            table,
            reference_period=tuple(self.reference_period),
            offset=self.offset,
            levels=self.levels,
            half_width=self.half_width,
            periods=[
                Period(tuple(period.gmst_span), tuple(period.impact_span))
                for period in self.periods
            ],
            smoothing=self.smoothing,
            members=members_that_join(self.join, declared, self.member),
        )


class StatsConfig(_Strict):
    """Statistics stage.

    Under stage ``all`` with "window_mean" each leaf is reduced to its
    window means as it is aggregated, so other ``windows`` take a new
    run, or a run of the reduce stage over leaves that were kept.
    """

    sample_dims: list[str] = Field(min_length=1)
    time_col: str = Field(min_length=1)
    windows: list[tuple[float, float]] = Field(default_factory=list)
    # In place of `windows`. The periods name their own years.
    warming_levels: Optional[WarmingLevelsConfig] = None
    quantiles: list[float] = Field(default_factory=list)
    include_mean: bool = True
    # Adds two count columns to the statistics file. n_members is the
    # samples each statistic was computed from, a batch of a model
    # counting once a batch, and n_models the climate models among them,
    # the distinct values of model_dims. Off unless asked for, and on
    # whatever the config says where a run pools fewer leaves than were
    # aggregated or takes warming levels.
    include_n_members: bool = False
    # How a quantile is computed. "inverted_cdf" is the estimator, the
    # standard inverse of the empirical distribution. "linear"
    # interpolates between two members and has no weighted form.
    # "lower_step" reproduces the extraction tool, one member below the
    # standard inverse, and is biased low. How a mean is summed. "fast" is the vectorised sum,
    # whose last bits depend on the order the workers hand their leaves
    # back in, and "exact" is the exactly rounded one. Neither has a
    # default, and the manifest records both. See
    # `stats.summarize_samples`.
    quantile_method: Literal["linear", "lower_step", "inverted_cdf"]
    summation: Literal["fast", "exact"]
    # What happens to the years of a member, with no default.
    # "window_mean" takes each member's mean over a window first.
    # "pooled" takes no mean, and the years inside a window are members
    # of the sample with the models and the draws; with include_n_members
    # the file gets an n_sample column as well, the rows behind a
    # statistic. "per_year" takes no mean either and keeps the year, so
    # the statistics are over the members within each year of `years`.
    # The last two take their statistics from the leaf files, by blocks,
    # under every stage. The manifest records it.
    time_reduction: Literal["window_mean", "pooled", "per_year"]
    # the first and the last year of a "per_year" product
    years: Optional[tuple[int, int]] = None
    model_weights: Optional[str] = None
    model_weight_level: Optional[str] = None
    # The row of each member of the weight level whose name is not that
    # of its row, member name to table name, written out. A member takes
    # the row of its own name otherwise, and nothing reads a name to
    # decide what it means. The published SMME tables need it until they
    # carry the members' names. The manifest records it.
    model_weight_names: dict[str, str] = Field(default_factory=dict)
    # The sample dimensions of the climate model, such as ["gcm"], which
    # the count of models beside the count of samples is taken over.
    # Required wherever the statistics carry their counts.
    model_dims: Optional[list[str]] = None

    @model_validator(mode="before")
    @classmethod
    def _former_values_are_named(cls, data):
        # Both options were once "legacy". The message says which value
        # it meant for this config.
        if isinstance(data, dict):
            if "pool_years" in data:
                raise ValueError(
                    "stats.pool_years is replaced by stats.time_reduction. "
                    "Use 'pooled' where it was true and 'window_mean' where "
                    "it was false."
                )
            if data.get("quantile_method") == "legacy":
                check_quantile_method(
                    "legacy", weighted=data.get("model_weights") is not None
                )
            if data.get("summation") == "legacy":
                check_summation("legacy")
        return data

    @model_validator(mode="after")
    def _model_dims_are_pooled(self) -> "StatsConfig":
        if self.model_dims is not None:
            if not self.model_dims:
                raise ValueError("stats.model_dims must name at least one column")
            stray = [d for d in self.model_dims if d not in self.sample_dims]
            if stray:
                raise ValueError(
                    f"stats.model_dims {stray} must be among stats.sample_dims "
                    f"{self.sample_dims}; a climate model is counted among the "
                    f"members it pools"
                )
        counted = self.include_n_members or self.warming_levels is not None
        if counted and self.model_dims is None:
            why = (
                "stats.include_n_members is set"
                if self.include_n_members
                else "a warming level writes its counts"
            )
            raise ValueError(
                f"{why}, and the counts are of the samples and of the "
                f"climate models behind each statistic, so stats.model_dims "
                f"has to name the sample dimensions of the climate model, "
                f'such as ["gcm"]'
            )
        return self

    @model_validator(mode="after")
    def _weights_come_together(self) -> "StatsConfig":
        if (self.model_weights is None) != (self.model_weight_level is None):
            raise ValueError(
                "stats.model_weights and stats.model_weight_level come "
                "together: the file says what the weights are, the level "
                "says which sample dimension they attach to"
            )
        if self.model_weight_names and self.model_weights is None:
            raise ValueError(
                "stats.model_weight_names declares rows of a model weights "
                "file, and stats.model_weights names none"
            )
        empty = [m for m, row in self.model_weight_names.items() if not m or not row]
        if empty:
            raise ValueError(
                f"stats.model_weight_names maps a member name to a table name, "
                f"and {empty[0]!r} = {self.model_weight_names[empty[0]]!r} "
                f"leaves one empty"
            )
        rows: dict[str, list[str]] = {}
        for member, row in self.model_weight_names.items():
            rows.setdefault(row, []).append(member)
        shared = [(row, members) for row, members in rows.items() if len(members) > 1]
        if shared:
            row, members = shared[0]
            more = f", and {len(shared) - 1} more rows" if len(shared) > 1 else ""
            raise ValueError(
                f"stats.model_weight_names declares one row for more than one "
                f"member, the row {row!r} for "
                f"{_quoted_list(members)}{more}. Each member needs "
                f"a row of its own."
            )
        # "linear" with model weights is refused here, when the config is
        # read, and not after the leaves
        check_quantile_method(
            self.quantile_method, weighted=self.model_weights is not None
        )
        if (
            self.model_weight_level is not None
            and self.model_weight_level not in self.sample_dims
        ):
            raise ValueError(
                f"stats.model_weight_level {self.model_weight_level!r} must "
                f"be one of stats.sample_dims {self.sample_dims}"
            )
        if self.time_reduction != "window_mean" and self.time_col in self.sample_dims:
            raise ValueError(
                f"stats.time_reduction = {self.time_reduction!r} decides what "
                f"happens to stats.time_col {self.time_col!r} itself. Leave it "
                f"out of stats.sample_dims."
            )
        if self.time_reduction == "per_year":
            if self.years is None:
                raise ValueError(
                    "stats.time_reduction = 'per_year' needs stats.years, the "
                    "first and the last year of the product"
                )
            if self.years[0] > self.years[1]:
                raise ValueError(
                    f"stats.years starts in {self.years[0]}, after it ends in "
                    f"{self.years[1]}"
                )
            if self.windows:
                raise ValueError(
                    "stats.windows and stats.time_reduction = 'per_year' are "
                    "both set. A per-year product has no windows. Give its "
                    "years as stats.years."
                )
            if self.warming_levels is not None:
                raise ValueError(
                    "[stats.warming_levels] and stats.time_reduction = "
                    "'per_year' are both set. A warming level is reached over "
                    "a period, so its statistics are over the years of one."
                )
            return self
        if self.years is not None:
            raise ValueError(
                f"stats.years is for stats.time_reduction = 'per_year', and "
                f"this config has {self.time_reduction!r}"
            )
        if self.warming_levels is None:
            if not self.windows:
                raise ValueError(
                    "stats.windows must list at least one window, unless "
                    "[stats.warming_levels] gives the periods or "
                    "stats.time_reduction is 'per_year'"
                )
            return self
        if self.windows:
            raise ValueError(
                "stats.windows and [stats.warming_levels] are both set. A "
                "period of the warming levels names its own years, so leave "
                "stats.windows out."
            )
        if self.model_weights is not None:
            raise ValueError(
                "stats.model_weights and [stats.warming_levels] are both "
                "set. Model weights are made for a whole ensemble and say "
                "nothing of the members at one warming level, which count "
                "equally."
            )
        unpooled = [
            name for name in self.warming_levels.member if name not in self.sample_dims
        ]
        if unpooled:
            raise ValueError(
                f"stats.warming_levels.member {unpooled} must be among "
                f"stats.sample_dims {self.sample_dims}. A warming level "
                f"pools the members that are at it."
            )
        return self


class PipelineOutputConfig(_Strict):
    """Where the results are written. ``stem`` names the statistics and
    manifest files, so several runs can share one directory.
    ``write_reduced`` also writes the pooled window means, one row a
    target, window and sample member, which `regroup` works on.
    ``write_leaves`` keeps every aggregated leaf with its years under
    ``leaves/``. It is off unless asked for, because the leaves take far
    more disk than the statistics. ``leaves_stem`` names the leaves
    manifest, and is ``stem`` unless set, so a product with other
    windows can take its own stem and reduce the leaves of another."""

    dir: str = Field(min_length=1)
    write_leaves: bool = False
    leaves_stem: Optional[str] = None
    stem: str = "statistics"
    write_reduced: bool = False


class RunConfig(_Strict):
    n_workers: int = Field(default=1, ge=1)


class MonteCarloPipelineConfig(_Strict):
    """The config of a pipeline run. Without ``weights`` the run is a
    passthrough, with statistics per source region and no spatial
    aggregation."""

    weights: Optional[WeightsRef] = None
    tree: TreeConfig
    data: DataConfig
    stats: StatsConfig
    output: PipelineOutputConfig
    policies: SourceUnitPolicies = Field(default_factory=SourceUnitPolicies)
    run: RunConfig = Field(default_factory=RunConfig)
    reduce: ReduceConfig = Field(default_factory=ReduceConfig)

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
            if self.stats.time_reduction != "window_mean":
                raise ValueError(
                    f"stats.time_reduction = {self.stats.time_reduction!r} "
                    f"needs [weights]. Its statistics are reduced from the "
                    f"leaf files, and a passthrough run writes none of the "
                    f"kind that reads."
                )
            if self.stats.warming_levels is not None:
                raise ValueError(
                    "[stats.warming_levels] needs [weights]. Warming levels "
                    "are reduced from the leaf files, and a passthrough run "
                    "writes none of the kind that reads."
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
        if (
            self.tree.subtract_filename is not None
            and self.data.denominator_col is not None
        ):
            raise ValueError(
                "tree.subtract_filename with data.denominator_col is not "
                "supported: differencing a ratio's numerator and denominator "
                "is a different operation from differencing a value"
            )
        if (
            self.stats.model_weight_level is not None
            and self.stats.model_weight_level not in self.tree.levels
        ):
            raise ValueError(
                f"stats.model_weight_level {self.stats.model_weight_level!r} "
                f"must be a tree level {self.tree.levels}"
            )
        if self.stats.warming_levels is not None:
            strangers = [
                name
                for name in self.stats.warming_levels.member
                if name not in self.tree.levels
            ]
            if strangers:
                raise ValueError(
                    f"stats.warming_levels.member {strangers} must be tree "
                    f"levels {self.tree.levels}. A leaf's member is read off "
                    f"its level values."
                )
        return self


def load_pipeline_config(path: str | Path) -> MonteCarloPipelineConfig:
    """The validated config of a pipeline TOML.

    ``weights.dir``, ``tree.root``, ``output.dir``,
    ``stats.model_weights`` and ``stats.warming_levels.gmst`` given as
    relative paths resolve against the config file's directory, as in
    `config.load_config`. Raises if the file does not exist.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"pipeline config not found: {p}")
    with p.open("rb") as f:
        data = tomllib.load(f)

    def _resolve(value: str) -> str:
        return _resolved(value, p.parent)

    for *sections, key in (
        ("weights", "dir"),
        ("tree", "root"),
        ("output", "dir"),
        ("stats", "model_weights"),
        ("stats", "warming_levels", "gmst"),
    ):
        block = data
        for section in sections:
            block = block.get(section) if isinstance(block, dict) else None
        if isinstance(block, dict) and isinstance(block.get(key), str):
            block[key] = _resolve(block[key])
    return MonteCarloPipelineConfig.model_validate(data)


def _resolved(value: str, base: Path) -> str:
    if "://" in value or value.startswith("/"):
        return value
    return str((base / value).resolve())


def load_model_weight_names(path: str | Path) -> dict[str, str]:
    """The ``[stats.model_weight_names]`` table of a TOML file, member
    name to table name, so that a run and a script can declare the same
    rows. Raises if the file has no such table, or a name is not a
    string."""
    p = Path(path).resolve()
    with p.open("rb") as f:
        data = tomllib.load(f)
    names = data.get("stats", {}).get("model_weight_names")
    if not isinstance(names, dict):
        raise ValueError(f"{p} has no [stats.model_weight_names] table")
    bad = [(k, v) for k, v in names.items() if not isinstance(v, str) or not v]
    if bad:
        raise ValueError(
            f"{p} maps a member name to a table name, both strings; got "
            f"{bad[0][0]!r} = {bad[0][1]!r}"
        )
    return dict(names)


def load_warming_levels(path: str | Path) -> WarmingLevelsConfig:
    """The ``[stats.warming_levels]`` table of a TOML file, validated,
    so that a run and a test can read the same settings. A relative
    ``gmst`` resolves against the file's directory, as in
    `load_pipeline_config`. Raises if the file has no such table."""
    p = Path(path).resolve()
    with p.open("rb") as f:
        data = tomllib.load(f)
    warming = data.get("stats", {}).get("warming_levels")
    if not isinstance(warming, dict):
        raise ValueError(f"{p} has no [stats.warming_levels] table")
    if isinstance(warming.get("gmst"), str):
        warming["gmst"] = _resolved(warming["gmst"], p.parent)
    return WarmingLevelsConfig.model_validate(warming)


@dataclass(frozen=True)
class Leaf:
    """One unit of work, its level values and its resolved paths.
    ``output_path`` is None when the run writes no leaf file."""

    values: tuple[str, ...]
    path: Path
    output_path: Path | None
    subtract_path: Path | None = None

    def labels(self, levels: list[str]) -> dict[str, str]:
        return dict(zip(levels, self.values))


@dataclass(frozen=True)
class PipelinePlan:
    """What a run would do. A dry run returns it, and a run carries it
    out."""

    leaves: list[Leaf]
    missing: list[Leaf]
    stats_path: Path
    manifest_path: Path
    # set when the run writes leaf files in the layout of `leaves`
    leaves_manifest_path: Path | None = None
    stage: str = "all"
    # what a dry run of the reduce stage found, one line each
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [f"pipeline plan: {len(self.leaves)} declared leaves"]
        if self.stage != "aggregate":
            lines += [
                f"  statistics -> {self.stats_path}",
                f"  manifest   -> {self.manifest_path}",
            ]
        if self.leaves_manifest_path is not None:
            lines.append(f"  leaves     -> {self.leaves_manifest_path}")
        lines += list(self.notes)
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


def resolve_plan(cfg: MonteCarloPipelineConfig, stage: Stage = "all") -> PipelinePlan:
    """The plan of a run, with every declared leaf and those missing on
    disk. Under stage "reduce" a leaf is missing if its leaf file is,
    and otherwise if its source file or its subtract file is. Nothing is
    created or written."""
    root = Path(cfg.tree.root)
    out_root = Path(cfg.output.dir)
    value_lists = [cfg.tree.expect[level] for level in cfg.tree.levels]
    excluded = [tuple(cell) for cell in cfg.tree.exclude]
    leaves: list[Leaf] = []
    missing: list[Leaf] = []
    for combo in itertools.product(*value_lists):
        if any(
            all(want in (EVERY_VALUE, value) for want, value in zip(cell, combo))
            for cell in excluded
        ):
            continue
        rel = Path(*combo)
        path = root / rel / cfg.tree.filename
        subtract_path = (
            root / rel / cfg.tree.subtract_filename
            if cfg.tree.subtract_filename is not None
            else None
        )
        if not (cfg.output.write_leaves or stage == "reduce"):
            output_path = None
        elif cfg.weights is None:
            # passthrough has no targets to lay a leaf out by, and keeps
            # the file it always wrote
            output_path = out_root / "leaves" / rel / f"aggregated_{cfg.tree.filename}"
        else:
            output_path = (
                out_root
                / "leaves"
                / rel
                / f"{Path(cfg.tree.filename).stem}.aggregated.parquet"
            )
        leaf = Leaf(
            values=tuple(combo),
            path=path,
            output_path=output_path,
            subtract_path=subtract_path,
        )
        leaves.append(leaf)
        if stage == "reduce":
            # the source tree is not opened, only the leaf files
            if not output_path.exists():
                missing.append(leaf)
        elif not path.exists() or (
            subtract_path is not None and not subtract_path.exists()
        ):
            missing.append(leaf)
    return PipelinePlan(
        leaves=leaves,
        missing=missing,
        stats_path=out_root / f"{cfg.output.stem}.parquet",
        manifest_path=out_root / f"{cfg.output.stem}.manifest.json",
        leaves_manifest_path=(
            out_root / "leaves" / f"{_leaves_stem(cfg)}.leaves.manifest.json"
            if (cfg.output.write_leaves or stage == "reduce")
            and cfg.weights is not None
            else None
        ),
        stage=stage,
    )


def _leaves_stem(cfg: MonteCarloPipelineConfig) -> str:
    return cfg.output.leaves_stem or cfg.output.stem


def run_pipeline(
    cfg: MonteCarloPipelineConfig,
    *,
    dry_run: bool = False,
    stage: Stage = "all",
    verify_leaves: bool = False,
) -> PipelinePlan | PipelineResult:
    """The result of a run, or under ``dry_run`` its plan, with nothing
    written. ``stage="aggregate"`` writes the leaf files and their
    manifest and stops, and the result's ``stats`` is then None.
    ``stage="reduce"`` takes the statistics from leaf files written
    before, never opens the source tree and reads the weights artifact
    only to hash it. ``verify_leaves`` goes with stage "reduce" alone
    and hashes every leaf file first, which reads all of them. Outside
    a dry run, raises if a declared leaf is missing or any leaf fails."""
    _check_stage(cfg, stage)
    if verify_leaves and stage != "reduce":
        raise ValueError(
            "verify_leaves checks the leaf files the reduce stage reads, so "
            "it goes with stage 'reduce'"
        )
    plan = resolve_plan(cfg, stage)
    if stage == "reduce":
        return _run_reduce(cfg, plan, dry_run=dry_run, verify_leaves=verify_leaves)
    # pooled years, years kept and warming levels are reduced from the
    # leaf files this run writes
    by_blocks = stage == "all" and bool(_from_leaf_files(cfg))
    warming = None
    if by_blocks:
        # asked before pass 1, which may take hours
        backend = get_backend(cfg.reduce)
        backend.check_options()
        _check_methods_supported(cfg, backend)
        warming = _warming_levels(cfg, plan)
    if dry_run:
        if by_blocks:
            return _with_notes(plan, _blocks_before_aggregation(cfg, plan, warming))
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
    model_weights = _model_weights(cfg) if stage != "aggregate" else None

    leaves_document = None
    job = {
        "reduce": stage == "all" and not by_blocks,
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
    print(
        f"pipeline code: {Path(__file__).resolve()} "
        f"(worker-init, per-leaf weights, memory instrumentation)",
        flush=True,
    )
    _log_memory("start")

    outcomes: list[dict] = []
    first_written: dict = {}

    def collect(outcome: dict) -> None:
        # The first leaf file to arrive fixes the columns. A later one
        # that differs stops the run there, not after the last leaf.
        written = outcome.get("leaf_file")
        if written is not None and not first_written:
            first_written.update(leaf=outcome["leaf"], schema=written["schema"])
        elif written is not None and written["schema"] != first_written["schema"]:
            raise ValueError(
                f"leaf {'/'.join(outcome['leaf'])} was written with columns "
                f"{written['schema']} where leaf "
                f"{'/'.join(first_written['leaf'])} has "
                f"{first_written['schema']}. The leaves of one tree need "
                f"the same columns and dtypes. The run stopped after "
                f"{len(outcomes) + 1} of {len(tasks)} leaves, and the "
                f"leaves manifest is left incomplete."
            )
        outcomes.append(outcome)

    if cfg.run.n_workers == 1 or len(tasks) <= 1:
        _init_worker(*initargs)
        for task in tasks:
            collect(_process_leaf(task))
    else:
        # The artifact loads once per worker (initializer), not once per
        # task: shipping it in every task pickle made long-lived workers
        # churn through thousands of unpickled copies, and their resident
        # memory grew with leaves processed. maxtasksperchild recycles
        # workers to bound what churn remains; chunksize=1 so recycling
        # counts actual leaves. imap_unordered streams results so memory
        # can be logged as the run accumulates.
        worker_peak = 0.0
        with mp.Pool(
            processes=cfg.run.n_workers,
            initializer=_init_worker,
            initargs=initargs,
            maxtasksperchild=16,
        ) as pool:
            for outcome in pool.imap_unordered(_process_leaf, tasks, chunksize=1):
                collect(outcome)
                worker_peak = max(worker_peak, outcome.get("worker_peak_gb") or 0.0)
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
    common = _split_targets(outcomes)
    if leaves_document is not None:
        _finish_leaves(plan, leaves_document, outcomes, files, common)
        if not by_blocks:
            # the second pass prints it
            print(_unlisted_report(plan.leaves_manifest_path.parent), flush=True)
    if stage == "aggregate":
        for o in outcomes:
            o.pop("frame")
        return PipelineResult(
            stats=None,
            leaf_reports=outcomes,
            paths={"leaves_manifest": str(plan.leaves_manifest_path)},
        )
    if by_blocks:
        del outcomes
        try:
            return _run_reduce(
                cfg,
                plan,
                dry_run=False,
                verify_leaves=False,
                model_weights=model_weights,
                warming=warming,
                stage="all",
            )
        except (ValueError, RuntimeError) as error:
            # a subclass may not take a message alone
            kind = RuntimeError if isinstance(error, RuntimeError) else ValueError
            raise kind(
                f"{error} The leaves are written and their manifest is "
                f"complete ({plan.leaves_manifest_path}). Stage 'reduce' "
                f"takes the statistics from them without aggregating again."
            ) from error

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
        model_dims=cfg.stats.model_dims,
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
    targets = None
    if artifact is not None:
        stats, targets = _with_shares(
            stats,
            list(artifact.schema.id_fields),
            common,
            [o["target_coverage"] for o in outcomes],
        )
    stats = _ordered(stats, cfg)
    _log_memory("after pooled statistics")

    out_root = Path(cfg.output.dir)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_parquet(stats, plan.stats_path)
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
            "weighted"
            if cfg.weights is not None
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
        "time_reduction": cfg.stats.time_reduction,
        "years": None,
        "warming_levels": None,
        "target_coverage": targets,
        "reduced": str(reduced_path) if reduced_path else None,
        "model_weights": cfg.stats.model_weights,
        "model_weight_level": cfg.stats.model_weight_level,
        "model_weight_names": cfg.stats.model_weight_names,
        "model_dims": cfg.stats.model_dims,
        "model_weights_record": _model_weights_record(
            cfg, model_weights, [o["leaf"] for o in outcomes]
        ),
        "subtract_filename": cfg.tree.subtract_filename,
        "excluded_leaves": [list(c) for c in cfg.tree.exclude],
        "package_versions": collect_package_versions(),
    }
    leaf_format.write_json(plan.manifest_path, manifest)

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


def _leaf_targets(report) -> dict:
    """What `apply_weights` reported of one leaf's targets, as plain
    values."""
    return {
        "basis": report.basis,
        "left_out": {
            reason: [list(key) for key in ids]
            for reason, ids in report.left_out.items()
        },
        "partial": [[list(key), share] for key, share in report.partial],
    }


def _split_targets(outcomes: list[dict]) -> Optional[dict]:
    """The targets every leaf leaves out, by reason, with the basis the
    shares are taken against, or None if no leaf was weighted. Each
    outcome's ``target_coverage`` is rewritten in place to the targets
    it leaves out that not every leaf does, and its partial ones."""
    reports = [
        o["target_coverage"] for o in outcomes if o.get("target_coverage") is not None
    ]
    if not reports:
        return None
    reasons = list(reports[0]["left_out"])
    every = {
        reason: set.intersection(
            *({tuple(key) for key in report["left_out"][reason]} for report in reports)
        )
        for reason in reasons
    }
    for outcome in outcomes:
        report = outcome.get("target_coverage")
        if report is None:
            continue
        outcome["target_coverage"] = {
            "left_out": {
                reason: [
                    list(key)
                    for key in sorted(
                        {tuple(k) for k in report["left_out"][reason]} - every[reason]
                    )
                ]
                for reason in reasons
            },
            "partial": report["partial"],
        }
    return {
        "basis": reports[0]["basis"],
        "left_out": {
            reason: [list(key) for key in sorted(every[reason])] for reason in reasons
        },
    }


def _with_shares(
    stats: pd.DataFrame, id_fields: list[str], common: dict, leaves: list[dict]
) -> tuple[pd.DataFrame, dict]:
    """``stats`` with the column of the share of each target's weight
    basis, the smallest any pooled leaf gave it, 1 where none was
    partial. Also what the manifest says of the targets, those left out
    of every pooled leaf by reason and the partial ones."""
    left_out = {}
    for reason, ids in common["left_out"].items():
        extra = [{tuple(key) for key in leaf["left_out"][reason]} for leaf in leaves]
        keys = {tuple(key) for key in ids} | (
            set.intersection(*extra) if extra else set()
        )
        left_out[reason] = {
            "count": len(keys),
            "ids": [list(key) for key in sorted(keys)],
        }
    smallest: dict[tuple, float] = {}
    for leaf in leaves:
        for key, share in leaf["partial"]:
            key = tuple(key)
            smallest[key] = min(share, smallest.get(key, share))
    shares = pd.Series(1.0, index=stats.index, dtype="float64")
    if smallest:
        table = pd.DataFrame(
            [list(key) + [share] for key, share in smallest.items()],
            columns=id_fields + ["_share"],
        )
        for field in id_fields:
            table[field] = table[field].astype(stats[field].dtype)
        found = stats[id_fields].merge(table, on=id_fields, how="left")["_share"]
        shares = pd.Series(found.fillna(1.0).to_numpy(), index=stats.index)
    stats = stats.copy()
    stats[SHARE_COLUMN] = shares
    partial = sorted(smallest.items())
    return stats, {
        "basis": common["basis"],
        "left_out": left_out,
        "partial": {
            "count": len(partial),
            "smallest": min((share for _, share in partial), default=None),
            "targets": [list(key) + [share] for key, share in partial],
        },
    }


def _model_dims(
    cfg: MonteCarloPipelineConfig, reasons: list[str], warming
) -> list[str]:
    """The columns the climate models are counted over, none if the
    statistics carry no counts. Raises if they carry them for ``reasons``
    and the config names none."""
    if not (cfg.stats.include_n_members or reasons or warming is not None):
        return []
    if cfg.stats.model_dims is None:
        raise ValueError(
            f"the statistics carry their counts ({'; '.join(reasons)}), and "
            f"they are of the samples and of the climate models behind each "
            f"statistic, so stats.model_dims has to name the sample "
            f'dimensions of the climate model, such as ["gcm"]'
        )
    return list(cfg.stats.model_dims)


def _model_weights(cfg: MonteCarloPipelineConfig) -> Optional[ModelWeights]:
    """The model weights table, read once, or None if the config has
    none. Raises if a declared value of the weight level has no row."""
    if cfg.stats.model_weights is None:
        return None
    # The file is read here and nowhere else. The workers get these
    # rows, so a file edited during the run cannot give different
    # leaves different weights.
    model_weights = read_model_weights(
        cfg.stats.model_weights, table_names=cfg.stats.model_weight_names
    )
    # fail before the run, not after it: every declared value of the
    # weight level must have a row in the table, and one of its own
    model_weights.check(cfg.tree.expect[cfg.stats.model_weight_level])
    return model_weights


def _model_weights_record(
    cfg: MonteCarloPipelineConfig,
    model_weights: Optional[ModelWeights],
    leaves: list[list[str]],
) -> Optional[dict]:
    """The manifest record of the model weights, taken from the members
    of the pooled leaves and not from the values the config declares.
    None without model weights."""
    if model_weights is None:
        return None
    level = cfg.tree.levels.index(cfg.stats.model_weight_level)
    return model_weights.record(values[level] for values in leaves)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write under a temporary name, then rename, so a file under
    ``path`` is always whole."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _check_stage(cfg: MonteCarloPipelineConfig, stage: str) -> None:
    if stage not in ("all", "aggregate", "reduce"):
        raise ValueError(
            f"unknown stage {stage!r}; expected 'all', 'aggregate' or 'reduce'"
        )
    if (
        cfg.stats.time_reduction != "window_mean"
        and stage != "aggregate"
        and cfg.output.write_reduced
    ):
        raise ValueError(
            f"stats.time_reduction = {cfg.stats.time_reduction!r} cannot "
            f"write the reduced file. Its rows are the leaves' own rows, one "
            f"block of targets at a time, and the leaf files already hold "
            f"them. Set output.write_reduced = false."
        )
    if (
        cfg.stats.warming_levels is not None
        and stage != "aggregate"
        and cfg.output.write_reduced
    ):
        raise ValueError(
            "[stats.warming_levels] cannot write the reduced file. Its "
            "statistics come from the leaf files, one block of targets at "
            "a time. Set output.write_reduced = false."
        )
    if stage == "reduce":
        if cfg.weights is None:
            raise ValueError(
                "stage 'reduce' needs [weights]. A passthrough run writes "
                "no leaves of the kind it reads."
            )
        if cfg.output.write_reduced:
            raise ValueError(
                "stage 'reduce' cannot write the reduced file. It pools one "
                "block of targets at a time, and the reduced file is sorted "
                "with the tree levels ahead of the target. Set "
                "output.write_reduced = false, or use stage 'all'."
            )
    if stage == "all" and _from_leaf_files(cfg) and not cfg.output.write_leaves:
        raise ValueError(
            f"{_from_leaf_files(cfg)[0]} takes the statistics from the leaf "
            f"files, so stage 'all' needs output.write_leaves = true"
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
    leaf itself. It holds the hashes of the weights artifact and no
    path."""
    # A path says where something was, and the hashes say what it was.
    return {
        **_config_identity(cfg),
        "artifact": {
            "source_version": artifact.source_version,
            "regions_version": artifact.regions_version,
            "sha256": sha256,
        },
    }


def _config_identity(cfg: MonteCarloPipelineConfig) -> dict:
    """The part of the identity that the config alone says."""
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
    }


def _claimed_elsewhere(leaves_dir: Path, own: Path, files: list[str]) -> None:
    """Raise if the leaves manifest of another output stem in the
    directory lists one of ``files``, whether it is complete or not."""
    for other in sorted(leaves_dir.glob("*.leaves.manifest.json")):
        if other == own:
            continue
        document = json.loads(other.read_text())
        taken = sorted({entry["file"] for entry in document["leaves"]} & set(files))
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
        "stem": _leaves_stem(cfg),
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
    plan: PipelinePlan,
    document: dict,
    outcomes: list[dict],
    files: dict,
    common: Optional[dict],
) -> None:
    """Write the leaves manifest again, complete, with the accounting
    of every leaf in plan order. ``common`` is what `_split_targets`
    found every leaf shares, and ``outcomes`` hold what each adds."""
    by_leaf = {tuple(o["leaf"]): o for o in outcomes}
    schema = files[plan.leaves[0].values]["schema"]
    profiles: list[dict] = []
    rows_per_group = np.zeros(
        len(files[plan.leaves[0].values]["rows_per_group"]), dtype=np.int64
    )
    entries = []
    for leaf, claimed in zip(plan.leaves, document["leaves"]):
        written, outcome = files[leaf.values], by_leaf[leaf.values]
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
                "target_coverage": outcome["target_coverage"],
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
            "target_coverage": common,
            "leaves": entries,
            "schema": schema,
            "rows_per_group": rows_per_group.tolist(),
            "profiles": profiles,
            "package_versions": collect_package_versions(),
        },
    )


def _load_leaves_document(path: Path) -> tuple[dict, str]:
    """The leaves manifest and the SHA256 of its bytes. Raises if it is
    missing, as it is beside leaf files of the old layout, if the run
    that wrote it did not finish, or if its layout version is not the
    one this library reads."""
    if not path.exists():
        raise ValueError(
            f"stage 'reduce' reads the leaves manifest {path}, and it is not "
            f"there. Leaf files written before the manifest existed cannot "
            f"be reduced. Run stage 'aggregate' first, or set "
            f"output.leaves_stem to the stem the leaves were written under."
        )
    raw = path.read_bytes()
    document = json.loads(raw)
    if not document.get("complete"):
        raise ValueError(
            f"the leaves manifest {path} is incomplete. The run that wrote "
            f"it did not finish, so its leaves are not all there. Run stage "
            f"'aggregate' again."
        )
    if document["format_version"] != leaf_format.FORMAT_VERSION:
        raise ValueError(
            f"the leaves under {path} have layout version "
            f"{document['format_version']}, and this version of the library "
            f"reads version {leaf_format.FORMAT_VERSION}"
        )
    return document, hashlib.sha256(raw).hexdigest()


def _check_identity(cfg: MonteCarloPipelineConfig, document: dict) -> list[str]:
    """Raise if the config would aggregate other leaves than those the
    manifest describes, naming the field. A path that differs is not a
    reason to refuse. It comes back as a note, and so does a weights
    artifact that is no longer where the config says."""
    recorded = document["aggregation"]
    for key, value in _config_identity(cfg).items():
        if recorded[key] != value:
            raise ValueError(
                f"the leaves were aggregated with {key} = {recorded[key]!r} "
                f"and the config has {value!r}. Stage 'reduce' pools the "
                f"leaves as they are. To change {key}, aggregate again."
            )
    notes = []
    try:
        found = _artifact_sha256(cfg.weights.dir, cfg.weights.stem)
    except OSError:
        notes.append(
            f"the weights artifact is not at {cfg.weights.dir}. The leaves "
            f"manifest holds its hashes, and the leaves are reduced as they "
            f"are."
        )
    else:
        if found != recorded["artifact"]["sha256"]:
            raise ValueError(
                f"the weights artifact at {cfg.weights.dir} is not the one "
                f"the leaves were aggregated with. Its files hash to {found} "
                f"and the leaves manifest has "
                f"{recorded['artifact']['sha256']}."
            )
    now = {
        "weights_dir": cfg.weights.dir,
        "weights_stem": cfg.weights.stem,
        "tree_root": cfg.tree.root,
    }
    for name, value in now.items():
        if document["paths"][name] != value:
            notes.append(
                f"{name} was {document['paths'][name]!r} when the leaves "
                f"were written and is {value!r} in the config. Paths are "
                f"recorded and not compared."
            )
    return notes


def _reduce_request(
    cfg: MonteCarloPipelineConfig,
    plan: PipelinePlan,
    document: dict,
    model_weights: Optional[ModelWeights],
    windows_of=None,
    warming: Optional[WarmingLevels] = None,
) -> tuple[ReduceRequest, list[str]]:
    """The request for the declared leaves, and the reasons, if any, the
    member counts are switched on whatever the config says.

    ``windows_of`` gives a leaf's own windows from its level values, and
    every leaf has the spans of `_windows` unless it is given. With
    ``warming`` it is not used, and a leaf has the periods in which its
    member is at a level, each under a label that names the period and
    the level. Raises if a declared leaf is not in the manifest, or a
    window of a leaf has no row in it.
    """
    time_col = cfg.stats.time_col
    columns = [c["name"] for c in document["schema"]]
    if time_col not in columns:
        raise ValueError(
            f"stats.time_col {time_col!r} is not a column of the leaves "
            f"(they have {columns})"
        )
    if cfg.stats.time_reduction == "pooled" and "n_sample" in columns:
        raise ValueError(
            "the leaves have a column named 'n_sample', which is the column "
            "stats.time_reduction = 'pooled' writes its pooled rows in"
        )
    if warming is not None:
        for name in ("warming_level", "n_bin_members"):
            if name in columns:
                raise ValueError(
                    f"the leaves have a column named {name!r}, which is a "
                    f"column [stats.warming_levels] writes"
                )

        def windows_of(values):
            return [
                (_level_window(at.level, at.window), at.start, at.end)
                for at in warming.windows_of(_bin_member(cfg, values))
            ]

    common = tuple((_window_label(lo, hi), lo, hi) for lo, hi in _windows(cfg))
    listed = {tuple(entry["leaf"]): entry for entry in document["leaves"]}
    absent = [leaf.values for leaf in plan.leaves if leaf.values not in listed]
    if absent:
        raise ValueError(
            f"{len(absent)} declared leaves are not in the leaves manifest "
            f"{plan.leaves_manifest_path} (first: {'/'.join(absent[0])}). "
            f"Stage 'reduce' can pool fewer leaves than were aggregated, "
            f"not more."
        )
    weight_level = (
        cfg.tree.levels.index(cfg.stats.model_weight_level)
        if model_weights is not None
        else None
    )
    leaves = []
    for leaf in plan.leaves:
        entry = listed[leaf.values]
        profile = document["profiles"][entry["profile"]].get(time_col)
        if profile is None:
            raise ValueError(
                f"the leaves manifest has no profile of {time_col!r}, so "
                f"the windows cannot be checked or the blocks sized. A "
                f"profile is kept for a numeric dimension of at most "
                f"{leaf_format.PROFILE_LIMIT} distinct values."
            )
        windows = common if windows_of is None else tuple(windows_of(leaf.values))
        for label, lo, hi in windows:
            if not any(lo <= value <= hi for value in profile["values"]):
                raise ValueError(
                    f"window ({lo}, {hi}) on {time_col!r} selects no rows in "
                    f"leaf {'/'.join(leaf.values)} (it spans "
                    f"{min(profile['values'])} to {max(profile['values'])})"
                )
            if warming is None:
                continue
            lacking = sorted(set(range(lo, hi + 1)) - set(profile["values"]))
            if lacking:
                raise ValueError(
                    f"leaf {'/'.join(leaf.values)} has no row for "
                    f"{len(lacking)} of the {hi - lo + 1} years {lo} to {hi} "
                    f"(first: {lacking[:5]}). A mean over fewer years than "
                    f"the span is refused."
                )
        leaves.append(
            LeafRequest(
                file=entry["file"],
                levels=leaf.values,
                windows=windows,
                rows=entry["rows_out"],
                profile=profile,
                model_weight=(
                    model_weights.weight_of(leaf.values[weight_level])
                    if model_weights is not None
                    else None
                ),
                bin_member=(
                    _bin_member(cfg, leaf.values) if warming is not None else None
                ),
            )
        )

    reasons = []
    if len(leaves) < len(document["leaves"]):
        reasons.append(
            f"{len(document['leaves']) - len(leaves)} of the "
            f"{len(document['leaves'])} aggregated leaves are not declared"
        )
    if warming is None and len({leaf.windows for leaf in leaves}) > 1:
        reasons.append("the windows differ between leaves")

    # the manifest has the rows of a group summed over all its leaves
    kept = sum(leaf.rows for leaf in leaves)
    every = sum(entry["rows_out"] for entry in document["leaves"])
    scale = kept / every if every else 0.0
    if warming is None:
        labels = list(dict.fromkeys(label for label, _, _ in common))
        for leaf in leaves:
            labels += [label for label, _, _ in leaf.windows if label not in labels]
    else:
        labels = list(_level_windows(cfg))
    request = ReduceRequest(
        leaves_root=str(plan.leaves_manifest_path.parent),
        leaves=tuple(leaves),
        footer={
            "format_version": document["format_version"],
            "targets_per_group": document["targets_per_group"],
            "n_targets": document["targets"]["n_targets"],
            "target_digest": document["targets"]["digest"],
            "aggregation_digest": document["aggregation_digest"],
        },
        targets_file=document["targets"]["file"],
        rows_per_group=tuple(n * scale for n in document["rows_per_group"]),
        schema=tuple(document["schema"]),
        id_fields=tuple(document["targets"]["id_fields"]),
        level_names=tuple(cfg.tree.levels),
        time_col=time_col,
        value_col=cfg.data.value_col,
        window_labels=tuple(labels),
        sample_dims=tuple(cfg.stats.sample_dims),
        quantiles=tuple(cfg.stats.quantiles),
        include_mean=cfg.stats.include_mean,
        # a statistic of a warming level is always written with its counts
        include_n_members=(
            cfg.stats.include_n_members or bool(reasons) or warming is not None
        ),
        model_dims=tuple(_model_dims(cfg, reasons, warming)),
        quantile_method=cfg.stats.quantile_method,
        summation=cfg.stats.summation,
        time_reduction=cfg.stats.time_reduction,
        include_n_bin_members=warming is not None,
        whole_spans=warming is not None,
        n_workers=cfg.run.n_workers,
    )
    return request, reasons


def _windows(cfg: MonteCarloPipelineConfig) -> list[tuple[float, float]]:
    """The spans the impacts are reduced over. With warming levels they
    are the impact years of the periods, and per year it is the one
    span of ``stats.years``."""
    warming = cfg.stats.warming_levels
    if cfg.stats.time_reduction == "per_year":
        return [tuple(cfg.stats.years)]
    if warming is None:
        return list(cfg.stats.windows)
    return [tuple(period.impact_span) for period in warming.periods]


def _from_leaf_files(cfg: MonteCarloPipelineConfig) -> list[str]:
    """The options set that take the statistics from the leaf files
    under every stage."""
    shape = cfg.stats.time_reduction
    return ([f"stats.time_reduction = {shape!r}"] if shape != "window_mean" else []) + (
        ["[stats.warming_levels]"] if cfg.stats.warming_levels is not None else []
    )


def _bin_member(cfg: MonteCarloPipelineConfig, values) -> tuple[str, ...]:
    """The member of the temperature table a leaf belongs to, read off
    its level values."""
    return tuple(
        values[cfg.tree.levels.index(name)] for name in cfg.stats.warming_levels.member
    )


def _level_window(level: float, window: str) -> str:
    return f"{window} at {level:g}"


def _level_windows(cfg: MonteCarloPipelineConfig) -> dict[str, tuple[float, str]]:
    """The level and the period of every label a warming level request
    uses for a window, in the order of the periods and then of the
    levels ascending."""
    warming = cfg.stats.warming_levels
    return {
        _level_window(level, _window_label(*period.impact_span)): (
            float(level),
            _window_label(*period.impact_span),
        )
        for period in warming.periods
        for level in sorted(warming.levels)
    }


def _warming_levels(
    cfg: MonteCarloPipelineConfig, plan: PipelinePlan
) -> Optional[WarmingLevels]:
    """The warming levels of the members of the tree that join, or None
    if the config has no warming levels. The table is read here and
    nowhere else. Raises if a member the tree declares has no row in
    it, or if no declared member is at any level."""
    warming = cfg.stats.warming_levels
    if warming is None:
        return None
    members = list(dict.fromkeys(_bin_member(cfg, leaf.values) for leaf in plan.leaves))
    levels = warming.assign(warming.read_table(), members)
    if not any(levels.windows_of(member) for member in members):
        raise ValueError(
            f"none of the {len(members)} members the tree declares is at "
            f"any of the levels {list(warming.levels)} in any period, so "
            f"there is nothing to take statistics of"
        )
    return levels


def _verify_leaves(leaves_dir: Path, entries: list[dict], n_workers: int) -> None:
    """Hash every leaf file and raise if one differs from the manifest.
    This reads every byte of every leaf."""

    def differs(entry: dict) -> bool:
        return leaf_format.file_sha256(leaves_dir / entry["file"]) != entry["sha256"]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        changed = list(pool.map(differs, entries))
    bad = [entry["file"] for entry, wrong in zip(entries, changed) if wrong]
    if bad:
        raise ValueError(
            f"{len(bad)} of {len(entries)} leaf files do not hash to what "
            f"the leaves manifest has (first: {bad[:5]}). They were changed "
            f"after they were written. No statistics were written."
        )
    print(f"leaves: {len(entries)} files hash to what the manifest has", flush=True)


# the model weights are still to be read
_UNREAD = object()


def _check_methods_supported(cfg: MonteCarloPipelineConfig, backend) -> None:
    methods = (cfg.stats.quantile_method, cfg.stats.summation)
    if methods not in backend.supports:
        raise ValueError(
            f"the {backend.kind!r} reduce backend does not give "
            f"quantile_method={methods[0]!r} with summation={methods[1]!r} "
            f"to the last bit. It supports {sorted(backend.supports)}."
        )


def _with_notes(plan: PipelinePlan, notes: list[str]) -> PipelinePlan:
    return PipelinePlan(
        leaves=plan.leaves,
        missing=plan.missing,
        stats_path=plan.stats_path,
        manifest_path=plan.manifest_path,
        leaves_manifest_path=plan.leaves_manifest_path,
        stage=plan.stage,
        notes=tuple(f"  {line.strip()}" for line in notes),
    )


def _blocks_before_aggregation(
    cfg: MonteCarloPipelineConfig,
    plan: PipelinePlan,
    warming: Optional[WarmingLevels],
) -> list[str]:
    """What a dry run of stage 'all' can say of the blocks when the
    statistics come from the leaf files. They are sized from the leaves
    manifest, which the run itself writes. If leaves of this stem are
    on disk and complete, the lines are for those."""
    after = (
        f"reduce: {_from_leaf_files(cfg)[0]} takes the statistics from the "
        f"leaf files, by blocks, after every leaf is written"
    )
    if not plan.leaves_manifest_path.exists():
        return [after, "the blocks are sized once the leaves are written"]
    try:
        document, _ = _load_leaves_document(plan.leaves_manifest_path)
        _check_identity(cfg, document)
        request, _ = _reduce_request(
            cfg, plan, document, _model_weights(cfg), warming=warming
        )
        lines = get_backend(cfg.reduce).describe(request)["text"]
    except (ValueError, OSError) as error:
        return [
            after,
            f"the leaves on disk now do not say how the blocks will be sized: {error}",
        ]
    return [after, "with the leaves on disk now, which this run writes again:"] + lines


def _run_reduce(
    cfg: MonteCarloPipelineConfig,
    plan: PipelinePlan,
    *,
    dry_run: bool,
    verify_leaves: bool,
    model_weights=_UNREAD,
    warming=_UNREAD,
    stage: str = "reduce",
) -> PipelinePlan | PipelineResult:
    """The result of taking the statistics from leaf files that a run of
    stage 'aggregate' or 'all' wrote, checked against their manifest
    first, or under ``dry_run`` the plan with what the checks found.
    The source tree is not opened, and the weights artifact is only
    hashed, if it is still there. ``model_weights`` and ``warming`` are
    passed when the caller has read them already, and ``stage`` is the
    stage the manifest records."""
    leaves_dir = plan.leaves_manifest_path.parent
    document, document_sha256 = _load_leaves_document(plan.leaves_manifest_path)
    if "target_coverage" not in document:
        raise ValueError(
            f"the leaves manifest {plan.leaves_manifest_path} was written "
            f"before each leaf recorded its targets left out and the shares "
            f"of its partial ones, so the statistics could not say which "
            f"rest on part of a target. Aggregate the leaves again."
        )
    notes = _check_identity(cfg, document)
    _check_level_names(cfg.tree.levels, document["targets"]["id_fields"])
    if model_weights is _UNREAD:
        model_weights = _model_weights(cfg)
    if warming is _UNREAD:
        warming = _warming_levels(cfg, plan)
    request, forced = _reduce_request(
        cfg, plan, document, model_weights, warming=warming
    )
    for reason in forced:
        notes.append(
            f"n_members is written although the config does not ask for "
            f"it, because {reason}"
        )
    backend = get_backend(cfg.reduce)
    _check_methods_supported(cfg, backend)
    description = backend.describe(request)
    summary = description["text"] + [_unlisted_report(leaves_dir)]
    if dry_run:
        return _with_notes(plan, summary + notes)

    listed = {tuple(entry["leaf"]): entry for entry in document["leaves"]}
    entries = [listed[leaf.values] for leaf in plan.leaves]
    for entry in entries:
        file = leaves_dir / entry["file"]
        if not file.exists():
            raise ValueError(
                f"leaf file {file} is in the leaves manifest and not on disk"
            )
        if file.stat().st_size != entry["size"]:
            raise ValueError(
                f"leaf file {file} is {file.stat().st_size} bytes and the "
                f"leaves manifest has {entry['size']}. It was changed after "
                f"it was written."
            )
    for line in summary:
        print(line, flush=True)
    for note in notes:
        warnings.warn(note, stacklevel=3)
    if verify_leaves:
        _verify_leaves(leaves_dir, entries, cfg.run.n_workers)
    if warming is None:
        # the members at a warming level count equally by design
        _warn_if_unweighted_models(
            cfg.stats.model_dims, model_weights is not None, "stats.model_weights"
        )
    _log_memory("reduce start")
    stats, report = backend.run(request)
    stats = stats[request.columns]
    if warming is not None:
        # the window a block was reduced under names a period and a level
        of_label = _level_windows(cfg)
        labels = stats["window"]
        stats.insert(
            list(stats.columns).index("window"),
            "warming_level",
            labels.map({label: level for label, (level, _) in of_label.items()}).astype(
                np.float64
            ),
        )
        stats["window"] = labels.map(
            {label: window for label, (_, window) in of_label.items()}
        )
    for id_field, column in zip(request.id_fields, document["schema"][1:]):
        # what the same ids are in a frame of the pipeline's own
        wanted = "string" if column["pandas"] == "string" else pd.Series(["a"]).dtype
        stats[id_field] = stats[id_field].astype(wanted)
    stats, targets = _with_shares(
        stats,
        list(request.id_fields),
        document["target_coverage"],
        [entry["target_coverage"] for entry in entries],
    )
    stats = _ordered(stats, cfg)
    _log_memory("after reduce")

    out_root = Path(cfg.output.dir)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_parquet(stats, plan.stats_path)
    declared = {leaf.values for leaf in plan.leaves}
    manifest = {
        "leaves": entries,
        "n_leaves": len(entries),
        "stage": stage,
        "leaves_manifest": str(plan.leaves_manifest_path),
        "leaves_manifest_sha256": document_sha256,
        "leaves_left_out": [
            entry["leaf"]
            for entry in document["leaves"]
            if tuple(entry["leaf"]) not in declared
        ],
        "leaves_verified": verify_leaves,
        "n_members_forced": forced,
        "notes": notes,
        "reduce": report,
        "artifact_sha256": document["aggregation"]["artifact"]["sha256"],
        "weight": cfg.weights.weight,
        "spatial_aggregation": "weighted",
        "kind": cfg.data.kind,
        "data_version": cfg.data.version,
        "artifact_source_version": document["aggregation"]["artifact"][
            "source_version"
        ],
        "artifact_regions_version": document["aggregation"]["artifact"][
            "regions_version"
        ],
        "sample_dims": cfg.stats.sample_dims,
        "windows": (
            []
            if cfg.stats.time_reduction == "per_year"
            else [_window_label(lo, hi) for lo, hi in _windows(cfg)]
        ),
        "quantiles": cfg.stats.quantiles,
        "quantile_method": cfg.stats.quantile_method,
        "summation": cfg.stats.summation,
        "time_reduction": cfg.stats.time_reduction,
        "years": list(cfg.stats.years) if cfg.stats.years else None,
        "target_coverage": targets,
        "warming_levels": (
            {
                **warming.record(
                    dict.fromkeys(_bin_member(cfg, leaf.values) for leaf in plan.leaves)
                ),
                "join": cfg.stats.warming_levels.join,
            }
            if warming is not None
            else None
        ),
        "reduced": None,
        "model_weights": cfg.stats.model_weights,
        "model_weight_level": cfg.stats.model_weight_level,
        "model_weight_names": cfg.stats.model_weight_names,
        "model_dims": cfg.stats.model_dims,
        "model_weights_record": _model_weights_record(
            cfg, model_weights, [entry["leaf"] for entry in entries]
        ),
        "subtract_filename": cfg.tree.subtract_filename,
        "excluded_leaves": [list(c) for c in cfg.tree.exclude],
        "package_versions": collect_package_versions(),
    }
    leaf_format.write_json(plan.manifest_path, manifest)
    return PipelineResult(
        stats=stats,
        leaf_reports=entries,
        paths={
            "statistics": str(plan.stats_path),
            "manifest": str(plan.manifest_path),
            "leaves_manifest": str(plan.leaves_manifest_path),
        },
    )


def _unlisted_report(leaves_dir: Path) -> str:
    """One line on the files under ``leaves_dir`` that no leaves
    manifest there lists, with their number and total size. The
    manifests of every stem count, complete or not, and so do the
    target tables they name. Nothing is deleted."""
    listed = set()
    for manifest in leaves_dir.glob("*.leaves.manifest.json"):
        document = json.loads(manifest.read_text())
        listed.add(manifest.name)
        listed.add(document["targets"]["file"])
        listed.update(entry["file"] for entry in document["leaves"])
    unlisted = sorted(
        path
        for path in leaves_dir.rglob("*")
        if path.is_file() and path.relative_to(leaves_dir).as_posix() not in listed
    )
    if not unlisted:
        return f"leaves: every file under {leaves_dir} is in a leaves manifest"
    size = sum(path.stat().st_size for path in unlisted)
    shown = [path.relative_to(leaves_dir).as_posix() for path in unlisted[:3]]
    return (
        f"leaves: {len(unlisted)} files under {leaves_dir} are in no leaves "
        f"manifest, {size:,} bytes ({size / 1e9:.2f} GB), first {shown}. "
        f"Nothing was deleted."
    )


def _check_level_names(levels: list[str], id_fields: list[str]) -> None:
    """Raise if a tree level is named like an id field of the weights
    artifact or like a column the pipeline writes itself."""
    taken = {
        **{name: "an id field of the weights artifact" for name in id_fields},
        "window": "the column of window labels",
        "statistic": "the column of statistic labels",
        "n_members": "the column stats.include_n_members writes",
        "n_models": "the column of the climate models behind a statistic",
        "n_sample": "the column of pooled rows stats.time_reduction writes",
        "warming_level": "the column of levels [stats.warming_levels] writes",
        "n_bin_members": "the column of members [stats.warming_levels] writes",
        SHARE_COLUMN: "the column of the share of each target's weight basis",
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
    values, then the window as configured, or the year ascending when
    each year is kept, then the other columns ascending in the order of
    the frame's columns, such as the target's id fields, leaving out the
    value, the counts, the share and the model weight. Together those
    name a row, so the order does not depend on the order the leaves
    were pooled in. Raises if a declared column holds a value the config
    does not declare."""
    if cfg.stats.time_reduction == "per_year":
        first, last = cfg.stats.years
        time = {cfg.stats.time_col: list(range(first, last + 1))}
    else:
        time = {"window": [_window_label(lo, hi) for lo, hi in _windows(cfg)]}
    declared = {
        "statistic": (["mean"] if cfg.stats.include_mean else [])
        + [_quantile_label(q) for q in cfg.stats.quantiles],
        **{level: list(cfg.tree.expect[level]) for level in cfg.tree.levels},
        **time,
    }
    # Rows are picked by label below, and a frame put together from
    # several without a new index has labels that repeat.
    frame = frame.reset_index(drop=True)
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
            cfg.data.value_col,
            "n_members",
            "n_sample",
            "n_models",
            "n_bin_members",
            SHARE_COLUMN,
            "model_weight",
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
    """The memory charged to the job's cgroup in GB, which the OOM
    killer compares with the limit, or None outside a cgroup. It
    includes every worker and the page cache."""
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
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
    """One leaf, read in the declared format whatever its extension.
    Raises for a NetCDF leaf if the artifact keys its sources by more
    than one column."""
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
    """``frame`` with the values of the file at ``path`` taken off, row
    for row, in a float64 value column. In the projection trees this is
    the scenario file minus its histclim sibling, before any weighting.
    Raises if the two files differ in columns, if a key repeats in
    either, or if a row of either has no match in the other. They are
    one projection under two climates, so a row in one alone means the
    other is incomplete.
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
    if len(other) != len(merged):
        # each row of the file matched at most one row of frame
        flagged = other[keys].merge(frame[keys], on=keys, how="left", indicator=True)
        alone = flagged.loc[flagged["_merge"] == "left_only", keys]
        first = {
            k: v.item() if hasattr(v, "item") else v for k, v in alone.iloc[0].items()
        }
        only = []
        for key in keys:
            values = sorted(set(alone[key]) - set(frame[key]), key=str)
            if values:
                shown = ", ".join(str(v) for v in values[:3])
                more = f" and {len(values) - 3} more" if len(values) > 3 else ""
                only.append(f"{key} {shown}{more}")
        where = f" The subtract file alone has {'; '.join(only)}." if only else ""
        raise ValueError(
            f"subtract file {path} has {len(alone)} "
            f"{'row' if len(alone) == 1 else 'rows'} the leaf does not "
            f"have (first: {first}); the two files must cover the same "
            f"regions and years.{where} They are one projection under two "
            f"climates, so a row in one alone means the other is incomplete."
        )
    merged[data_cfg.value_col] = merged[data_cfg.value_col].astype("float64") - merged[
        f"{data_cfg.value_col}_subtract"
    ].astype("float64")
    return merged.drop(columns=[f"{data_cfg.value_col}_subtract"])


def _process_leaf(task: tuple) -> dict:
    """The accounting of one leaf, with the weights applied unless the
    run is a passthrough, and its window means under "frame" unless
    the job stops before them. It never raises, and an error comes back
    as text under "error". A leaf file written here keeps every year,
    and only the frame handed back is reduced. The level values go on
    the reduced rows and never on the frame the weights are applied to.
    An aggregated leaf file has them in its footer and its path, and a
    passthrough one as columns.
    """
    # Module-level, with one tuple argument, so that a multiprocessing
    # pool can run it. An error comes back as text so that the parent can
    # report a partial run, where a worker crash would lose the traceback.
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
            targets = None
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
            # The shares go on the statistics rows, by target, from what
            # each leaf reports here. On the leaf they would be one more
            # column to group by.
            out_frame = applied.frame.drop(columns=[SHARE_COLUMN])
            targets = _leaf_targets(applied.targets)
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
                "target_coverage": targets,
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
            c in (stats_cfg.time_col, data_cfg.value_col) for c in out_frame.columns
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
            if (
                data_cfg.region_col is not None
                and data_cfg.region_col in reduced.columns
            ):
                reduced[data_cfg.region_col] = reduced[data_cfg.region_col].astype(
                    "category"
                )
            reduced["window"] = reduced["window"].astype("category")
        if _WORKER_STATE.get("model_weights") is not None:
            # one climate model per leaf, so its weight is a constant
            # column here; attaching it per leaf replaces a merge over
            # the full combined frame that dominated end-of-run memory
            value = labels[stats_cfg.model_weight_level]
            reduced["model_weight"] = _WORKER_STATE["model_weights"].weight_of(value)

        return {
            "leaf": list(leaf.values),
            "path": str(leaf.path),
            "rows_in": int(len(frame)),
            "rows_out": int(len(out_frame)),
            "rows_reduced": int(len(reduced)),
            "coverage": coverage,
            "target_coverage": targets,
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
        choices=["all", "aggregate", "reduce"],
        default="all",
        help="aggregate writes the leaf files and stops; reduce takes the "
        "statistics from leaf files written before; all (the default) runs "
        "everything",
    )
    parser.add_argument(
        "--verify-leaves",
        action="store_true",
        help="with --stage reduce, hash every leaf file against the leaves "
        "manifest first; reads every byte of every leaf",
    )
    args = parser.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    if args.dry_run:
        plan = run_pipeline(cfg, dry_run=True, stage=args.stage)
        print(plan.summary())
        return 1 if plan.missing else 0

    result = run_pipeline(cfg, stage=args.stage, verify_leaves=args.verify_leaves)
    done = "reduced" if args.stage == "reduce" else "aggregated"
    print(
        f"pipeline: {len(result.leaf_reports)} leaves {done}; "
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
