"""Runtime weight descriptors built from `WeightConfig`.

The area weight needs no raster or table and is always available.
`WeightSpec.is_area` marks it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cil_regionalization.config import FallbackPolicy, WeightConfig, WeightKind


@dataclass(frozen=True)
class WeightSpec:
    name: str
    # None for the area weight alone, which has no fallback.
    fallback: FallbackPolicy | None
    raster: str | None
    table: str | None
    # How the raster is integrated. "point_sum" sums raw pixel values;
    # "area_weighted_sum" sums (pixel value * geodesic pixel area in m^2)
    # and is the crop weight's canonical integration. None for the area
    # weight alone, which reads no raster.
    kind: WeightKind | None
    # The column of the BigQuery table that holds the weight's values.
    column: str | None = None

    @classmethod
    def from_config(cls, cfg: WeightConfig) -> "WeightSpec":
        return cls(
            name=cfg.name,
            fallback=cfg.fallback,
            raster=cfg.raster,
            table=cfg.table,
            kind=cfg.kind,
            column=cfg.column,
        )

    @property
    def is_area(self) -> bool:
        return self.name == "area"

    def require_raster(self) -> str:
        if self.raster is None:
            raise ValueError(
                f"weights.{self.name}.raster is required for the local backend "
                f"(area weights do not need a raster)"
            )
        return self.raster

    def require_table(self) -> str:
        if self.table is None:
            raise ValueError(
                f"weights.{self.name}.table is required for the bigquery backend "
                f"(area weights do not need a table)"
            )
        return self.table


def from_config_list(cfgs: Iterable[WeightConfig]) -> list[WeightSpec]:
    return [WeightSpec.from_config(c) for c in cfgs]
