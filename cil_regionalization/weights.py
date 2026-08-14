"""Runtime weight descriptors.

Wraps `WeightConfig` with derived flags so backends can dispatch without
re-checking the config rules. The area weight is special-cased: it has no
external input and is always available, so several places need to short-
circuit on `is_area`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cil_regionalization.config import FallbackPolicy, WeightConfig, WeightKind


@dataclass(frozen=True)
class WeightSpec:
    name: str
    fallback: FallbackPolicy
    raster: str | None
    table: str | None
    # True if the user wrote `fallback = "..."` in the TOML; False if the
    # value came from the WeightConfig default. Drives the implicit-
    # default warning in `apply_fallback`; a fallback firing for many
    # regions on a defaulted policy probably surprises the user.
    fallback_explicit: bool = False
    # How the raster is integrated. "point_sum" (default) sums raw pixel
    # values; "area_weighted_sum" sums (pixel value * geodesic pixel
    # area in m^2) and is the crop weight's canonical integration.
    kind: WeightKind = "point_sum"

    @classmethod
    def from_config(cls, cfg: WeightConfig) -> "WeightSpec":
        return cls(
            name=cfg.name,
            fallback=cfg.fallback,
            raster=cfg.raster,
            table=cfg.table,
            fallback_explicit="fallback" in cfg.model_fields_set,
            kind=cfg.kind,
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
