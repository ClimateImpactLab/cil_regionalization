"""The request for statistics, the backend interface and its config.

A request says which leaves to pool and which statistics to take. It
holds nothing about how: no block sizes and no engine options. A second
backend adds a module, a branch in `get_backend`, a name in the ``kind``
of `ReduceConfig` and an options field there.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd
from pydantic import Field

from cil_regionalization.config import _Strict

# a statistics method pair, (quantile_method, summation)
Methods = tuple[str, str]
ALL_METHODS: frozenset[Methods] = frozenset(
    (q, s)
    for q in ("linear", "lower_step", "inverted_cdf")
    for s in ("fast", "exact")
)


class PandasReduceOptions(_Strict):
    """Options of the pandas backend. ``memory_gb`` is the memory all
    its workers may hold together and has no default. ``n_workers``
    defaults to ``run.n_workers``."""

    memory_gb: Optional[float] = Field(default=None, gt=0)
    n_workers: Optional[int] = Field(default=None, ge=1)


class ReduceConfig(_Strict):
    kind: Literal["pandas"] = "pandas"
    pandas: PandasReduceOptions = Field(default_factory=PandasReduceOptions)


@dataclass(frozen=True)
class LeafRequest:
    """One leaf to pool. ``windows`` are this leaf's own, each a label
    with the first and last time value it covers. Two leaves may give
    one label different spans, and a leaf may lack a label another has.
    ``profile`` is the rows at each value of the time column."""

    file: str
    levels: tuple[str, ...]
    windows: tuple[tuple[str, float, float], ...]
    rows: int
    profile: dict
    model_weight: Optional[float] = None


@dataclass(frozen=True)
class ReduceRequest:
    """Statistics wanted from a set of leaf files.

    ``leaves`` are in the order their members are pooled in, which
    decides the last bits of a mean under the "fast" sum.
    ``quantile_method`` and ``summation`` have no default.
    ``rows_per_group`` are those of ``leaves``, summed. The statistics
    come back with the columns of `columns`.

    Under ``pool_years`` no window mean is taken. Every row inside a
    window is a member of the sample with its own time value, and a row
    inside two windows is in both.
    """

    leaves_root: str
    leaves: tuple[LeafRequest, ...]
    footer: dict
    targets_file: str
    rows_per_group: tuple[float, ...]
    schema: tuple[dict, ...]
    id_fields: tuple[str, ...]
    level_names: tuple[str, ...]
    time_col: str
    value_col: str
    window_labels: tuple[str, ...]
    sample_dims: tuple[str, ...]
    quantiles: tuple[float, ...]
    include_mean: bool = True
    include_n_members: bool = False
    quantile_method: str = field(kw_only=True)
    summation: str = field(kw_only=True)
    pool_years: bool = False
    n_workers: int = 1

    @property
    def methods(self) -> Methods:
        return (self.quantile_method, self.summation)

    @property
    def weighted(self) -> bool:
        return any(leaf.model_weight is not None for leaf in self.leaves)

    @property
    def dimensions(self) -> list[str]:
        """The leaf's own columns that stay in the statistics."""
        from cil_regionalization.leaves import ORDINAL_COLUMN

        skipped = {ORDINAL_COLUMN, self.time_col, self.value_col, *self.id_fields}
        return [c["name"] for c in self.schema if c["name"] not in skipped]

    @property
    def columns(self) -> list[str]:
        """The columns of the statistics frame, in order."""
        kept_levels = [n for n in self.level_names if n not in self.sample_dims]
        kept_dims = [n for n in self.dimensions if n not in self.sample_dims]
        return (
            list(self.id_fields)
            + kept_dims
            + kept_levels
            + ["window", "statistic", self.value_col]
            + (["n_members"] if self.include_n_members else [])
            + (["n_sample"] if self.include_n_members and self.pool_years else [])
        )


class ReduceBackend(ABC):
    """Computes the statistics of a request.

    ``supports`` holds the method pairs whose statistics the backend
    gives to the last bit. The pipeline refuses a request outside them.
    """

    kind: str
    supports: frozenset[Methods] = ALL_METHODS

    def check_options(self) -> None:
        """Raise if the options rule out every request. The pipeline
        asks before it aggregates anything."""

    @abstractmethod
    def describe(self, request: ReduceRequest) -> dict:
        """What `run` would do, without doing it. ``engine`` holds the
        versions of what computes, and ``text`` the lines a dry run
        prints."""

    @abstractmethod
    def run(self, request: ReduceRequest) -> tuple[pd.DataFrame, dict]:
        """The statistics, with the columns of ``request.columns`` in
        any row order, and a report for the manifest."""


def get_backend(config: ReduceConfig) -> ReduceBackend:
    """The backend ``config.kind`` names, with its options."""
    if config.kind == "pandas":
        from cil_regionalization.reduce.pandas_blocks import PandasBlocks

        return PandasBlocks(config.pandas)
    raise ValueError(f"unknown reduce backend {config.kind!r}")
