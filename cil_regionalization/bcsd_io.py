"""NASA BCSD daily reader and grid-scale derived climate variables.

The BCSD daily archive holds one grid in two dressings: the baseline
GCM files use longitudes in [0,360) with a ``time`` dimension, the
pattern (surrogate) files use [-180,180) with a ``day`` dimension.
``read_bcsd_daily`` normalizes both to [-180,180) ascending with a
``time`` dimension, deciding by coordinate values rather than by which
tree the file came from, and proves every roll by value identity: the
value at one probe cell after rolling must equal the value at that
cell's pre-roll longitude. A roll that shifted the globe by half fails
the identity immediately rather than corrupting every downstream
number silently.

``derive_daily`` computes the derived variables per grid cell and
year, before any spatial step: means from (tasmax + tasmin) / 2,
threshold counts from the extreme the threshold belongs to (days
below 32F from tasmin, days above 95F from tasmax), and the number of
days entering each count. Counting after averaging would give a
different and wrong answer, so nothing here accepts aggregated input:
the arrays are (time, lat, lon).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOT_K = 308.15   # 35 C = 95 F, applied to daily tasmax
COLD_K = 273.15  # 0 C = 32 F, applied to daily tasmin


def _import_netcdf4():
    try:
        import netCDF4
    except ImportError as e:
        raise ImportError(
            "reading BCSD daily files requires netCDF4; install the "
            "[netcdf] extra"
        ) from e
    return netCDF4


@dataclass(frozen=True)
class DailyGrid:
    """One file's worth of daily values on the normalized grid.

    ``values`` is (time, lat, lon) float32 with masked cells as NaN,
    ``lat`` and ``lon`` are ascending cell centers with lon in
    [-180,180).
    """

    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def read_bcsd_daily(path: str, variable: str) -> DailyGrid:
    """Read one BCSD daily file and normalize its conventions.

    Handles both longitude conventions and both time dimension names
    by inspecting coordinates, not the path. The longitude roll is
    verified by value identity at a probe cell on every call.
    """
    netCDF4 = _import_netcdf4()
    with netCDF4.Dataset(path) as ds:
        dims = ds.variables[variable].dimensions
        time_dims = [d for d in dims if d in ("time", "day")]
        if len(time_dims) != 1:
            raise ValueError(
                f"{path}: variable {variable!r} has dimensions {dims}; "
                f"expected exactly one of 'time' or 'day'"
            )
        if tuple(dims[-2:]) != ("lat", "lon"):
            raise ValueError(
                f"{path}: variable {variable!r} has dimensions {dims}; "
                f"expected (..., 'lat', 'lon')"
            )
        raw = ds.variables[variable][:]
        lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        lon = np.asarray(ds.variables["lon"][:], dtype="float64")
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    values = np.asarray(raw, dtype="float32")

    if np.any(np.diff(lat) <= 0):
        raise ValueError(f"{path}: latitudes are not ascending")

    rolled = np.where(lon >= 180.0, lon - 360.0, lon)
    order = np.argsort(rolled)
    if np.any(np.diff(rolled[order]) <= 0):
        raise ValueError(
            f"{path}: longitudes do not form a single cycle; rolling "
            f"produced duplicates"
        )
    # value-identity proof of the roll: whatever value sat at the raw
    # longitude probe must sit at the rolled position afterwards
    probe_raw = len(lon) - 1
    out_values = values[:, :, order]
    out_lon = rolled[order]
    probe_new = int(np.searchsorted(out_lon, rolled[probe_raw]))
    if not np.array_equal(
        values[0, :, probe_raw], out_values[0, :, probe_new],
        equal_nan=True,
    ):
        raise AssertionError(
            f"{path}: longitude roll failed the value-identity check; "
            f"values did not move with their coordinates"
        )
    return DailyGrid(values=out_values, lat=lat, lon=out_lon)


@dataclass(frozen=True)
class DerivedYear:
    """Per-cell derived variables for one calendar year.

    Every array is (lat, lon). Means are in degrees Celsius, computed
    from (tasmax + tasmin) / 2 because the archive carries no daily
    tas; counts come from the extreme their threshold is defined on:
    ``days_below_32F`` from tasmin, ``days_above_95F`` from tasmax.
    ``n_days`` is the number of days entering the counts and the
    annual mean (365 in this archive; no leap years).
    """

    tas_annual: np.ndarray
    tas_jja: np.ndarray
    tas_djf: np.ndarray
    days_above_95F: np.ndarray
    days_below_32F: np.ndarray
    n_days: int
    n_days_jja: int
    n_days_djf: int
    # cell-days where tasmin > tasmax arrived swapped and were
    # corrected by elementwise min/max before anything was counted
    n_swapped: int = 0


# 365-day no-leap calendar: day-of-year index ranges per month
_MONTH_STARTS = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
_MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _month_slice(month: int) -> slice:
    start = int(_MONTH_STARTS[month - 1])
    return slice(start, start + _MONTH_LENGTHS[month - 1])


def derive_daily(
    tasmax: np.ndarray,
    tasmin: np.ndarray,
    december_prev: np.ndarray | None,
) -> DerivedYear:
    """Derived variables for one year of daily grids, per cell.

    ``tasmax`` and ``tasmin`` are (365, lat, lon) in Kelvin on the
    no-leap calendar. ``december_prev`` is the preceding year's
    December of (tasmax + tasmin) / 2, needed because DJF for year y
    takes December of y - 1; pass None for a first year and the DJF
    mean covers January and February only, which the caller should
    treat as a spin-up year, not a product year.
    """
    if tasmax.shape != tasmin.shape or tasmax.shape[0] != 365:
        raise ValueError(
            f"expected matching (365, lat, lon) arrays; got "
            f"{tasmax.shape} and {tasmin.shape}"
        )
    # swap correction, as the documentation repo's reference module
    # defines tasmin and tasmax: independent downscaling occasionally
    # leaves cell-days with tasmin > tasmax, and a threshold count on
    # swapped values is meaningless. Elementwise min and max; the
    # midpoint is unaffected either way.
    n_swapped = int(np.sum(tasmin > tasmax))
    hi = np.maximum(tasmax, tasmin)
    lo = np.minimum(tasmax, tasmin)
    tasmax, tasmin = hi, lo
    tas = (tasmax.astype("float64") + tasmin.astype("float64")) / 2.0

    jja = np.concatenate(
        [tas[_month_slice(m)] for m in (6, 7, 8)], axis=0)
    janfeb = np.concatenate(
        [tas[_month_slice(m)] for m in (1, 2)], axis=0)
    if december_prev is not None:
        djf = np.concatenate([december_prev.astype("float64"), janfeb],
                             axis=0)
    else:
        djf = janfeb

    return DerivedYear(
        tas_annual=tas.mean(axis=0) - 273.15,
        tas_jja=jja.mean(axis=0) - 273.15,
        tas_djf=djf.mean(axis=0) - 273.15,
        days_above_95F=(tasmax > HOT_K).sum(axis=0).astype("float64"),
        days_below_32F=(tasmin < COLD_K).sum(axis=0).astype("float64"),
        n_days=365,
        n_days_jja=jja.shape[0],
        n_days_djf=djf.shape[0],
        n_swapped=n_swapped,
    )


def december(tasmax: np.ndarray, tasmin: np.ndarray) -> np.ndarray:
    """December's daily (tasmax + tasmin) / 2, for the next year's DJF."""
    sl = _month_slice(12)
    return ((tasmax[sl].astype("float64")
             + tasmin[sl].astype("float64")) / 2.0)
