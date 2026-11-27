"""The abstract weights backend and the result every backend returns.

A backend's `compute()` takes the regions, the grid, the weights and the
`Config`, and returns a `WeightsResult` whose `frame` follows the
canonical `OutputSchema`.
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
    """A backend's output, the weights frame with its schema, its manifest
    and its sum-to-one report."""

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
        """The weights of ``regions`` as a `WeightsResult`. ``grid`` is None in
        polygon mode, when the config declares ``[source]``."""
        ...
