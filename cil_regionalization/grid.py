"""Grid math: integer cell indices, centroids, polygons, bbox-to-range.

A `GridSpec` is the runtime counterpart of `GridConfig`. It exposes the
index / centroid / polygon conversions both backends rely on, plus a
prebuilt loader that validates uniform spacing.

Generate mode
-------------
Cells are addressed by integer (ix, iy). Centroids follow the formula

    center offset: centroid = (i + 0.5) * res + origin
    edge   offset: centroid =  i        * res + origin

The cell extent is always centroid +- res/2 in both axes.

Longitude convention is configurable:

    "[-180,180)" : lon_origin = -180
    "[0,360)"    : lon_origin =    0

Latitude origin is always -90.

Antimeridian rule
-----------------
Cells never wrap. A region whose bbox crosses 0/360 (or -180/180) must be
split by the caller into two bboxes; querying with such a bbox returns the
cells inside the global domain only, with no implicit wrap-around. This
keeps every (ix, iy) globally unique and makes the BigQuery side trivial.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon

from cil_regionalization.config import GridConfig

_LAT_ORIGIN = -90.0
_LAT_SPAN = 180.0
_LON_SPAN = 360.0


def _lon_origin(convention: str) -> float:
    if convention == "[-180,180)":
        return -180.0
    if convention == "[0,360)":
        return 0.0
    raise ValueError(f"unknown lon_convention: {convention!r}")


@dataclass(frozen=True)
class GridSpec:
    """Validated grid description; index math derives from these fields alone.

    Construct via `GridSpec.from_config(grid_config)`. The dataclass is frozen
    so caching by identity / hashing works for downstream partitioning.
    """

    mode: str
    resolution: float
    offset: str
    lon_convention: str
    lon_origin: float
    n_ix: int
    n_iy: int
    path: str | None = None
    table: str | None = None

    @classmethod
    def from_config(cls, cfg: GridConfig) -> "GridSpec":
        if cfg.mode == "generate":
            res = float(cfg.resolution)
            n_ix_f = _LON_SPAN / res
            n_iy_f = _LAT_SPAN / res
            n_ix = int(round(n_ix_f))
            n_iy = int(round(n_iy_f))
            if abs(n_ix_f - n_ix) > 1e-9 or abs(n_iy_f - n_iy) > 1e-9:
                raise ValueError(
                    f"grid.resolution={res} does not tile the globe evenly "
                    f"(360/res={n_ix_f}, 180/res={n_iy_f})"
                )
            return cls(
                mode="generate",
                resolution=res,
                offset=cfg.offset,
                lon_convention=cfg.lon_convention,
                lon_origin=_lon_origin(cfg.lon_convention),
                n_ix=n_ix,
                n_iy=n_iy,
            )
        # prebuilt: resolution and origin filled in by loader
        return cls(
            mode="prebuilt",
            resolution=float("nan"),
            offset=cfg.offset,
            lon_convention=cfg.lon_convention,
            lon_origin=_lon_origin(cfg.lon_convention),
            n_ix=-1,
            n_iy=-1,
            path=cfg.path,
            table=cfg.table,
        )

    @property
    def _centroid_half(self) -> float:
        return 0.5 if self.offset == "center" else 0.0

    @property
    def _index_half(self) -> float:
        return 0.0 if self.offset == "center" else 0.5

    @property
    def domain_lon(self) -> Tuple[float, float]:
        return self.lon_origin, self.lon_origin + _LON_SPAN

    @property
    def domain_lat(self) -> Tuple[float, float]:
        return _LAT_ORIGIN, _LAT_ORIGIN + _LAT_SPAN

    def centroid(self, ix, iy):
        """Return (lon, lat) of the cell centroid(s). Vectorised.

        Accepts scalars or arrays. No bounds-checking; passing ix outside
        [0, n_ix) is the caller's responsibility (the result is still
        mathematically well-defined).
        """
        ix_a = np.asarray(ix)
        iy_a = np.asarray(iy)
        h = self._centroid_half
        lon = (ix_a + h) * self.resolution + self.lon_origin
        lat = (iy_a + h) * self.resolution + _LAT_ORIGIN
        return lon, lat

    def index_of(self, lon, lat):
        """Return integer (ix, iy) cells containing (lon, lat). Vectorised.

        Uses floor((coord - origin)/res + index_half). At exact cell
        boundaries the upper cell is chosen, matching the half-open
        [low, high) extent convention.
        """
        lon_a = np.asarray(lon, dtype=float)
        lat_a = np.asarray(lat, dtype=float)
        h = self._index_half
        ix = np.floor((lon_a - self.lon_origin) / self.resolution + h).astype(np.int64)
        iy = np.floor((lat_a - _LAT_ORIGIN) / self.resolution + h).astype(np.int64)
        return ix, iy

    def cell_polygon(self, ix: int, iy: int) -> Polygon:
        """Shapely Polygon for one cell, in lon/lat degrees."""
        lon_c, lat_c = self.centroid(ix, iy)
        half = self.resolution / 2.0
        return shapely.geometry.box(
            float(lon_c) - half,
            float(lat_c) - half,
            float(lon_c) + half,
            float(lat_c) + half,
        )

    def cell_polygons(self, ix, iy) -> np.ndarray:
        """Vectorised cell polygon construction; returns an object ndarray."""
        ix_a = np.asarray(ix, dtype=np.int64)
        iy_a = np.asarray(iy, dtype=np.int64)
        lon_c, lat_c = self.centroid(ix_a, iy_a)
        half = self.resolution / 2.0
        polys = shapely.box(lon_c - half, lat_c - half, lon_c + half, lat_c + half)
        return polys

    def index_range_for_bbox(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
    ) -> Tuple[range, range]:
        """Inclusive ranges of (ix, iy) covering [minx,maxx] x [miny,maxy].

        Clipped to the global grid domain; never wraps the antimeridian. Some
        cells touching only the bbox boundary may be included (the downstream
        intersection test filters zero-overlap cells, so this is harmless).
        """
        if maxx < minx or maxy < miny:
            raise ValueError(
                f"index_range_for_bbox: bbox is empty "
                f"({minx=}, {maxx=}, {miny=}, {maxy=})"
            )
        ix_lo, iy_lo = self.index_of(minx, miny)
        ix_hi, iy_hi = self.index_of(maxx, maxy)
        ix_lo = int(max(ix_lo, 0))
        iy_lo = int(max(iy_lo, 0))
        ix_hi = int(min(ix_hi, self.n_ix - 1))
        iy_hi = int(min(iy_hi, self.n_iy - 1))
        if ix_hi < ix_lo or iy_hi < iy_lo:
            return range(0, 0), range(0, 0)
        return range(ix_lo, ix_hi + 1), range(iy_lo, iy_hi + 1)

    def enumerate_cells(
        self,
        ix_range: range,
        iy_range: range,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten an (ix_range, iy_range) product into parallel arrays."""
        ix_arr, iy_arr = np.meshgrid(
            np.fromiter(ix_range, dtype=np.int64),
            np.fromiter(iy_range, dtype=np.int64),
            indexing="ij",
        )
        return ix_arr.ravel(), iy_arr.ravel()


def infer_uniform_resolution(coords: Iterable[float], tol: float = 1e-9) -> float:
    """Return the constant step size in `coords` (assumed sorted, unique).

    Raises ValueError if the steps are not uniform within `tol`. Used by the
    prebuilt-grid loader to validate that the user-supplied cell coordinates
    are on a regular grid.
    """
    arr = np.asarray(list(coords), dtype=float)
    if arr.size < 2:
        raise ValueError("need at least two coordinates to infer resolution")
    diffs = np.diff(arr)
    res = float(diffs[0])
    if res <= 0:
        raise ValueError(f"coords must be strictly increasing; first step = {res}")
    if not np.all(np.abs(diffs - res) <= tol):
        worst = float(np.max(np.abs(diffs - res)))
        raise ValueError(
            f"non-uniform spacing in coords: first step = {res}, "
            f"max deviation = {worst}"
        )
    return res
