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

import re
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon

from cil_regionalization.config import GridConfig

_LAT_ORIGIN = -90.0
_LAT_SPAN = 180.0
_LON_SPAN = 360.0


# a grid's identity string, as GridSpec.source_version writes it
GRID_SOURCE_PATTERN = re.compile(
    r"^grid-(?P<res>[0-9]*\.?[0-9]+)deg-(?P<offset>center|edge)-"
    r"(?P<conv>\[-180,180\)|\[0,360\))$"
)


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

    @property
    def source_version(self) -> str:
        """The grid's identity as a source version string.

        A grid is fully described by its resolution, offset and
        longitude convention, so those are the version: data built on
        the same grid states the same string and the application-time
        version check compares them, exactly as polygon versions do.
        """
        return f"grid-{self.resolution}deg-{self.offset}-{self.lon_convention}"

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

    @classmethod
    def from_source_version(cls, source: str) -> "GridSpec":
        """The grid a source version string names.

        ``source`` is the identity `source_version` writes, for example
        ``grid-0.25deg-center-[-180,180)``. Any other spelling raises,
        including one that names the same grid differently
        (``grid-.25deg-...``), since versions are compared as exact
        strings.
        """
        m = GRID_SOURCE_PATTERN.match(source)
        if m is None:
            raise ValueError(
                f"{source!r} is not a grid identity of the form "
                f"'grid-<resolution>deg-<center|edge>-<[-180,180)|[0,360)>'"
            )
        spec = cls.from_config(GridConfig(
            mode="generate",
            resolution=float(m["res"]),
            offset=m["offset"],
            lon_convention=m["conv"],
        ))
        if spec.source_version != source:
            raise ValueError(
                f"{source!r} names the grid that writes its identity as "
                f"{spec.source_version!r}; use that spelling"
            )
        return spec

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


class AmbiguousLongitudeConvention(ValueError):
    """Every longitude is in [0,180], which fits both conventions.

    ``candidates`` are the two grid identities the coordinates fit,
    [-180,180) first. A file of this kind cannot say which one the
    caller's weights were built on.
    """

    def __init__(self, lo: float, hi: float, candidates: list[str]):
        self.candidates = candidates
        self.lo = lo
        self.hi = hi
        super().__init__(
            f"longitudes run from {lo} to {hi}, which fits both "
            f"{candidates[0]!r} and {candidates[1]!r}"
        )


def _spacing(coords: np.ndarray) -> float:
    """Gap between neighbouring representable values at the largest coordinate."""
    top = coords[np.argmax(np.abs(coords))]
    return float(np.spacing(np.abs(top)))


def _shortest_decimal(x: float, tol: float) -> float:
    """The shortest decimal within ``tol`` of ``x``.

    A float32 file stores 0.1 as 0.10000000149 and its steps wobble by
    the coordinate spacing, so the step is snapped to the fewest digits
    that stay inside the tolerance: 0.100006 becomes 0.1, 0.25 stays.
    """
    for digits in range(0, 10):
        r = round(x, digits)
        if abs(r - x) <= tol:
            return r
    return x


def coordinate_tolerance(coords: np.ndarray, resolution: float) -> float:
    """How far a stored coordinate may sit from its cell centre.

    1e-6 of a cell, or twice the spacing of the coordinate's own dtype
    if that is larger: a float32 longitude near 180 cannot be closer to
    a 0.1 degree centre than about 1e-5. Raises when the dtype is so
    coarse that the tolerance passes one percent of a cell, since a
    check that loose no longer separates neighbouring grids.
    """
    tol = max(1e-6 * resolution, 2.0 * _spacing(coords))
    if tol > 0.01 * resolution:
        raise ValueError(
            f"coordinates are stored too coarsely to place on a "
            f"{resolution} degree grid: dtype {coords.dtype} resolves "
            f"{tol:.3g} degrees there, over one percent of a cell"
        )
    return tol


def infer_grid_version(lat, lon) -> str:
    """The grid identity a file's 1-D latitudes and longitudes sit on.

    The resolution is the constant step between neighbours (either
    direction), snapped so a float32 file
    gives 0.1, not 0.10000000149 (the shortest decimal within the
    coordinates' own precision). Centre or edge comes from where the
    coordinates fall against the -90 and longitude origins, and the two
    axes must agree. The longitude convention is [0,360) when any value
    exceeds 180 and [-180,180) when any is negative. A file with
    longitudes only in [0,180] fits both, so this raises
    `AmbiguousLongitudeConvention`, which carries both identities, and
    the caller must pick one.

    Latitude and longitude steps must be equal. The identity string
    holds one resolution, so unequal steps raise.

    Raises when the steps are not uniform, there are fewer than two
    values on an axis, or the coordinates fit no grid. It does not
    check the result against the coordinates; `cell_indices` does.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    steps = []
    for name, c in (("lat", lat), ("lon", lon)):
        if c.ndim != 1 or c.size < 2:
            raise ValueError(
                f"cannot derive a grid from {c.size} {name} values; pass the "
                f"grid identity explicitly"
            )
        if not np.all(np.isfinite(c)):
            raise ValueError(f"{name} has values that are not finite")
        steps.append(np.abs(np.diff(c.astype("float64"))))
    res_by_axis = []
    for name, c, d in (("lat", lat, steps[0]), ("lon", lon, steps[1])):
        tol = max(1e-9, 4.0 * _spacing(c))
        res = _shortest_decimal(float(np.median(d)), tol)
        worst = float(np.max(np.abs(d - res)))
        if res <= 0 or worst > tol:
            raise ValueError(
                f"{name} is not evenly spaced: every step should be {res}, "
                f"largest deviation {worst:.3g}"
            )
        res_by_axis.append(res)
    if res_by_axis[0] != res_by_axis[1]:
        raise ValueError(
            f"lat steps are {res_by_axis[0]} and lon steps are "
            f"{res_by_axis[1]}; only square cells are supported"
        )
    res = res_by_axis[0]

    lo, hi = float(lon.min()), float(lon.max())
    if lo < 0 and hi > 180:
        raise ValueError(
            f"longitudes run from {lo} to {hi}, which fits neither [-180,180) "
            f"nor [0,360)"
        )
    ambiguous = lo >= 0 and hi <= 180
    conv = "[0,360)" if hi > 180 else "[-180,180)"

    tol = max(coordinate_tolerance(lat, res),
              coordinate_tolerance(lon, res)) / res
    offsets = set()
    for name, c, origin in (("lat", lat, _LAT_ORIGIN),
                            ("lon", lon, _lon_origin(conv))):
        phase = ((c.astype("float64") - origin) / res) % 1.0
        if np.all(np.abs(phase - 0.5) <= tol):
            offsets.add("center")
        elif np.all((phase <= tol) | (phase >= 1.0 - tol)):
            offsets.add("edge")
        else:
            raise ValueError(
                f"{name} values sit on no {res} degree grid starting at "
                f"{origin}: they are neither all cell centres nor all cell "
                f"edges"
            )
    if len(offsets) != 1:
        raise ValueError(
            f"lat and lon disagree: one sits on cell centres and the other "
            f"on cell edges of a {res} degree grid"
        )
    offset = offsets.pop()
    if ambiguous:
        # both origins are a whole number of steps apart, so centre or
        # edge reads the same under either
        raise AmbiguousLongitudeConvention(lo, hi, [
            f"grid-{res}deg-{offset}-[-180,180)",
            f"grid-{res}deg-{offset}-[0,360)",
        ])
    return f"grid-{res}deg-{offset}-{conv}"


def cell_indices(spec: GridSpec, values, axis: str, *, tol: float | None = None):
    """Integer cell index along one axis for 1-D cell-centre coordinates.

    ``axis`` is "lon" or "lat". Longitudes are wrapped into the grid's
    convention first. Each index is mapped back to its centroid, and
    every value must match that centroid within ``tol`` (default 1e-6
    of a cell) and fall inside the grid; a miss raises with the first
    offending values. That round trip is what catches coordinates built
    on another origin, or given as cell edges where the grid has
    centres.
    """
    given = np.asarray(values, dtype="float64")
    if axis == "lon":
        given = np.mod(given - spec.lon_origin, 360.0) + spec.lon_origin
        idx, _ = spec.index_of(given, np.zeros_like(given))
        centre, _ = spec.centroid(idx, np.zeros_like(idx))
        n = spec.n_ix
    elif axis == "lat":
        _, idx = spec.index_of(np.full_like(given, spec.lon_origin), given)
        _, centre = spec.centroid(np.zeros_like(idx), idx)
        n = spec.n_iy
    else:
        raise ValueError(f"axis must be 'lon' or 'lat'; got {axis!r}")
    if tol is None:
        tol = 1e-6 * spec.resolution
    with np.errstate(invalid="ignore"):
        bad = (~np.isfinite(given) | (idx < 0) | (idx >= n)
               | (np.abs(given - centre) > tol))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} {axis} values are not cell centres on "
            f"{spec.source_version}; first: {given[bad][:5].tolist()}"
        )
    return idx
