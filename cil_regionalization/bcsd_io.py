"""Reads NASA BCSD daily files and derives climate variables from them
per grid cell and year.

The baseline GCM files use longitudes in [0,360) and a ``time``
dimension, and the pattern (surrogate) files use [-180,180) and a
``day`` dimension. ``read_bcsd_daily`` returns both on longitudes in
[-180,180) ascending, judged from the coordinate values.

Some baseline GCM files are no-leap (``noleap``, ``365_day``) and some
are Gregorian (``gregorian``, ``standard``, ``proleptic_gregorian``)
with 366 days in leap years. A CF time coordinate is decoded with the
file's own calendar and Feb 29 is dropped by date, never by position.
Any other calendar raises, ``360_day`` included. The pattern files carry
an integer day index with no reference date and pass through undecoded.

``derive_daily`` takes daily (time, lat, lon) grids, never aggregated
ones. It gives the means of (tasmax + tasmin) / 2, the days below 32F
counted from tasmin and the days above 95F counted from tasmax, and how
many days went into each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOT_K = 308.15  # 35 C = 95 F, applied to daily tasmax
COLD_K = 273.15  # 0 C = 32 F, applied to daily tasmin
# the grid read_bcsd_daily returns, as GridSpec.source_version writes it
BCSD_GRID = "grid-0.25deg-center-[-180,180)"

# calendars on which dropping Feb 29 by date leaves the 365-day no-leap
# year that derive_daily's fixed month slices assume
ACCEPTED_CALENDARS = frozenset(
    {"gregorian", "standard", "proleptic_gregorian", "noleap", "365_day"}
)


def _import_netcdf4():
    try:
        import netCDF4
    except ImportError as e:
        raise ImportError(
            "reading BCSD daily files requires netCDF4; install the [netcdf] extra"
        ) from e
    return netCDF4


@dataclass(frozen=True)
class DailyGrid:
    """One file's daily values on longitudes in [-180,180).

    ``values`` is (time, lat, lon) float32 with masked cells as NaN, and
    ``lat`` and ``lon`` are ascending cell centers. ``dates`` is the date
    of each time step as datetime64[D] with Feb 29 removed, or None when
    the file has no decodable time coordinate, as in the pattern files.
    """

    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    dates: np.ndarray | None = None

    def to_cells(self, fields, grid: str = BCSD_GRID):
        """Per-cell arrays as the table `apply_weights` takes, one row per
        grid cell with ``cell_ix`` and ``cell_iy`` on ``grid`` and one
        column per field.

        ``fields`` maps a column name to a (lat, lon) array on this grid,
        for example a variable computed from ``values``. Cells where any
        field is NaN are left out, and `apply_weights` raises if a
        weighted cell is missing. Raises if a field is not (lat, lon) or
        if any latitude or longitude is not a cell centre on ``grid``.
        """
        # An index computed from the wrong origin, or from cell edges
        # instead of centres, would otherwise move every value one cell
        # over with no error.
        import pandas as pd

        from cil_regionalization.grid import GridSpec

        spec = GridSpec.from_source_version(grid)
        shape = (len(self.lat), len(self.lon))
        for name, arr in fields.items():
            if np.shape(arr) != shape:
                raise ValueError(
                    f"field {name!r} has shape {np.shape(arr)}; expected "
                    f"(lat, lon) = {shape}"
                )
        lon = (
            np.mod(np.asarray(self.lon, dtype="float64") - spec.lon_origin, 360.0)
            + spec.lon_origin
        )
        lat = np.asarray(self.lat, dtype="float64")
        ix, _ = spec.index_of(lon, np.zeros_like(lon))
        _, iy = spec.index_of(np.full_like(lat, spec.lon_origin), lat)
        c_lon, _ = spec.centroid(ix, np.zeros_like(ix))
        _, c_lat = spec.centroid(np.zeros_like(iy), iy)
        tol = 1e-6 * spec.resolution
        for axis, given, centre, idx, n in (
            ("lon", lon, c_lon, ix, spec.n_ix),
            ("lat", lat, c_lat, iy, spec.n_iy),
        ):
            with np.errstate(invalid="ignore"):
                bad = (
                    ~np.isfinite(given)
                    | (idx < 0)
                    | (idx >= n)
                    | (np.abs(given - centre) > tol)
                )
            if bad.any():
                raise ValueError(
                    f"{int(bad.sum())} {axis} values are not cell centres on "
                    f"{grid}; first: {given[bad][:5].tolist()}"
                )
        frame = pd.DataFrame(
            {
                "cell_ix": np.tile(ix, len(iy)),
                "cell_iy": np.repeat(iy, len(ix)),
                **{name: np.asarray(arr).reshape(-1) for name, arr in fields.items()},
            }
        )
        if fields:
            frame = frame[np.isfinite(frame[list(fields)]).all(axis=1)]
        return frame.reset_index(drop=True)


def _decode_dates(ds, time_dim: str, path: str):
    """The calendar name and an object array of the cftime dates of the
    time dimension, one per step, or None when it has no coordinate
    variable with units of the CF form '<unit> since <reference>'.

    A missing calendar attribute means 'standard', the CF default.
    Raises if the calendar is '360_day' or any other not in
    ``ACCEPTED_CALENDARS``.
    """
    if time_dim not in ds.variables:
        return None
    tvar = ds.variables[time_dim]
    units = getattr(tvar, "units", None)
    if not isinstance(units, str) or " since " not in units:
        return None
    calendar = str(getattr(tvar, "calendar", "standard")).strip().lower()
    if calendar == "360_day":
        raise ValueError(
            f"{path}: calendar '360_day' is not supported; no rule has "
            f"been decided for mapping a 360-day year onto the 365-day "
            f"no-leap year"
        )
    if calendar not in ACCEPTED_CALENDARS:
        raise ValueError(
            f"{path}: calendar {calendar!r} is not supported; expected "
            f"one of {sorted(ACCEPTED_CALENDARS)}"
        )
    netCDF4 = _import_netcdf4()
    dates = netCDF4.num2date(
        np.asarray(tvar[:]),
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=True,
    )
    return calendar, np.atleast_1d(np.asarray(dates, dtype=object))


def read_bcsd_daily(
    path: str,
    variable: str,
    *,
    require_full_year: bool = False,
) -> DailyGrid:
    """``variable`` of one BCSD daily file as a `DailyGrid` on longitudes
    in [-180,180) ascending.

    The longitude convention and the time dimension, ``time`` or
    ``day``, are read from the coordinates, not the path. A CF time
    coordinate is decoded with the file's calendar and Feb 29 is dropped
    by date. ``require_full_year`` also requires exactly 365 days to
    remain after the drop, in a file with a decoded time coordinate.

    Raises if the variable does not end in (lat, lon) with exactly one
    time dimension, if the calendar is not supported, if the time
    coordinate and the variable differ in length, if latitudes are not
    ascending, if longitudes repeat after the roll, or if the roll fails
    its value check at a probe cell.
    """
    # Dropping Feb 29 by date is the rule the lab's earlier BCSD
    # transforms applied. require_full_year fails a production recipe at
    # read time rather than in derive_daily.
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
        # decode before reading the data so an unsupported calendar
        # fails without loading a year of grids
        decoded = _decode_dates(ds, time_dims[0], path)
        raw = ds.variables[variable][:]
        lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        lon = np.asarray(ds.variables["lon"][:], dtype="float64")
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    values = np.asarray(raw, dtype="float32")

    dates = None
    if decoded is not None:
        calendar, cf_dates = decoded
        time_axis = dims.index(time_dims[0])
        if len(cf_dates) != values.shape[time_axis]:
            raise ValueError(
                f"{path}: time coordinate has {len(cf_dates)} steps but "
                f"{variable!r} has {values.shape[time_axis]}"
            )
        keep = np.array(
            [not (d.month == 2 and d.day == 29) for d in cf_dates], dtype=bool
        )
        if not keep.all():
            values = np.compress(keep, values, axis=time_axis)
            cf_dates = cf_dates[keep]
        if require_full_year and len(cf_dates) != 365:
            raise ValueError(
                f"{path}: {len(cf_dates)} days remain after dropping "
                f"Feb 29 (calendar {calendar!r}, {len(keep)} in the "
                f"file); expected exactly 365"
            )
        # with Feb 29 gone every date exists on the proleptic Gregorian
        # calendar, so datetime64 represents all of them exactly
        dates = np.array(
            [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in cf_dates],
            dtype="datetime64[D]",
        )

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
        values[0, :, probe_raw],
        out_values[0, :, probe_new],
        equal_nan=True,
    ):
        raise AssertionError(
            f"{path}: longitude roll failed the value-identity check; "
            f"values did not move with their coordinates"
        )
    return DailyGrid(values=out_values, lat=lat, lon=out_lon, dates=dates)


@dataclass(frozen=True)
class DerivedYear:
    """Per-cell derived variables for one calendar year, each a (lat, lon)
    array.

    The means are in degrees Celsius, of (tasmax + tasmin) / 2.
    ``days_below_32F`` is counted from tasmin and ``days_above_95F`` from
    tasmax. ``n_days`` is the number of days in the counts and the
    annual mean, and ``n_days_jja`` and ``n_days_djf`` the number in each
    seasonal mean.
    """

    # the archive carries no daily tas, so the means are of the midpoint
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
    """The `DerivedYear` of one year of daily grids.

    ``tasmax`` and ``tasmin`` are (365, lat, lon) in Kelvin on the
    no-leap calendar. ``december_prev`` is the preceding year's December
    of (tasmax + tasmin) / 2, from `december`, since DJF of a year takes
    the December before it. With None the DJF mean covers January and
    February only, and the year is a spin-up year, not a product year.
    A cell-day with tasmin above tasmax is swapped before counting.
    Raises unless both arrays are (365, lat, lon) of one shape.
    """
    # Counting after averaging would give a different and wrong answer,
    # so this takes daily grids per cell, before any spatial step.
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

    jja = np.concatenate([tas[_month_slice(m)] for m in (6, 7, 8)], axis=0)
    janfeb = np.concatenate([tas[_month_slice(m)] for m in (1, 2)], axis=0)
    if december_prev is not None:
        djf = np.concatenate([december_prev.astype("float64"), janfeb], axis=0)
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
    return (tasmax[sl].astype("float64") + tasmin[sl].astype("float64")) / 2.0
