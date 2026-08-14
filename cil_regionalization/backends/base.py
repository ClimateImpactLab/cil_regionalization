"""Backend ABC and result type.

Both backends are constructed from a `Config`, then asked to `compute()`
against a `RegionSet` and a `GridSpec`. Both return a `WeightsResult` whose
`frame` follows the canonical `OutputSchema`. Anything the BigQuery and
local backends disagree on (boundary fractions, geodesic vs spherical
predicates) is documented at the call site, not hidden in a switch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from cil_regionalization.config import Config
from cil_regionalization.grid import GridSpec
from cil_regionalization.manifest import Manifest
from cil_regionalization.regions import RegionSet
from cil_regionalization.schema import OutputSchema
from cil_regionalization.validate import SumToOneReport
from cil_regionalization.weights import WeightSpec


@dataclass
class WeightsResult:
    """Combined output: frame + schema + manifest + validation report."""

    frame: pd.DataFrame
    schema: OutputSchema
    manifest: Manifest
    sum_report: SumToOneReport


class WeightsBackend(ABC):
    @abstractmethod
    def compute(
        self,
        regions: RegionSet,
        grid: GridSpec | None,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> WeightsResult:
        """``grid`` is None in polygon mode (``[source]`` declared)."""
        ...
