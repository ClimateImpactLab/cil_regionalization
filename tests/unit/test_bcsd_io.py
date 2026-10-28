"""BCSD daily reader normalization and grid-scale derivations."""
from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from cil_regionalization.bcsd_io import (
    DailyGrid,
    december,
    derive_daily,
    read_bcsd_daily,
)


def _write(path, lon, time_dim="time", values=None, n_time=3):
    lat = np.array([-0.125, 0.125])
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension(time_dim, n_time)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        v = ds.createVariable("tasmax", "f4", (time_dim, "lat", "lon"))
        if values is None:
            values = np.arange(n_time * len(lat) * len(lon), dtype="f4")
            values = values.reshape(n_time, len(lat), len(lon))
        v[:] = values
    return values


class TestReader:
    def test_negative_convention_passes_through(self, tmp_path):
        p = str(tmp_path / "a.nc")
        lon = np.array([-179.875, -0.125, 0.125, 179.875])
        vals = _write(p, lon)
        grid = read_bcsd_daily(p, "tasmax")
        assert np.allclose(grid.lon, lon)
        assert np.array_equal(grid.values, vals.astype("float32"))

    def test_zero_360_convention_rolls_with_values(self, tmp_path):
        p = str(tmp_path / "b.nc")
        lon = np.array([0.125, 90.125, 180.125, 270.125])
        vals = _write(p, lon)
        grid = read_bcsd_daily(p, "tasmax")
        assert np.allclose(grid.lon, [-179.875, -89.875, 0.125, 90.125])
        # the value that sat at 180.125 must now sit at -179.875
        assert np.array_equal(grid.values[:, :, 0], vals[:, :, 2])
        # and the value at 0.125 stays at 0.125
        i = int(np.searchsorted(grid.lon, 0.125))
        assert np.array_equal(grid.values[:, :, i], vals[:, :, 0])

    def test_day_dimension_accepted(self, tmp_path):
        p = str(tmp_path / "c.nc")
        _write(p, np.array([-0.125, 0.125]), time_dim="day")
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.values.shape[0] == 3

    def test_masked_cells_become_nan(self, tmp_path):
        p = str(tmp_path / "d.nc")
        lat = np.array([-0.125, 0.125])
        lon = np.array([-0.125, 0.125])
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            v = ds.createVariable("tasmax", "f4", ("time", "lat", "lon"),
                                  fill_value=9.96921e36)
            arr = np.ma.masked_array(
                np.ones((1, 2, 2), dtype="f4"),
                mask=[[[False, True], [False, False]]],
            )
            v[:] = arr
        grid = read_bcsd_daily(p, "tasmax")
        assert np.isnan(grid.values[0, 0, 1])
        assert grid.values[0, 0, 0] == 1.0

    def test_unknown_time_dimension_rejected(self, tmp_path):
        p = str(tmp_path / "e.nc")
        lat = np.array([-0.125, 0.125])
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("step", 1)
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lat
            ds.createVariable("tasmax", "f4", ("step", "lat", "lon"))
        with pytest.raises(ValueError, match="'time' or 'day'"):
            read_bcsd_daily(p, "tasmax")


def _write_dated(path, start, n_days, calendar, lon=None, time_dim="time",
                 units="days since 1950-01-01", write_calendar=True,
                 coordinate=True):
    """A daily file with a CF time coordinate on ``calendar``.

    ``start`` is a (year, month, day) tuple; step ``i`` is ``start``
    plus ``i`` days on that calendar, and its values are ``i`` plus a
    per-cell offset, so a value identifies the step it came from.
    ``coordinate=False`` writes the identical data with no time
    variable, the layout the reader handled before calendars.
    """
    import datetime

    import cftime

    if lon is None:
        lon = np.array([0.125, 90.125, 180.125, 270.125])
    lat = np.array([-0.125, 0.125])
    first = cftime.datetime(*start, calendar=calendar)
    dates = [first + datetime.timedelta(days=i) for i in range(n_days)]
    offsets = np.arange(len(lat) * len(lon), dtype="f4").reshape(
        1, len(lat), len(lon)) / 100.0
    values = (np.arange(n_days, dtype="f4").reshape(n_days, 1, 1)
              + offsets).astype("f4")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension(time_dim, n_days)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        if coordinate:
            t = ds.createVariable(time_dim, "f8", (time_dim,))
            t.units = units
            if write_calendar:
                t.calendar = calendar
            t[:] = cftime.date2num(dates, units=units, calendar=calendar)
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("tasmax", "f4", (time_dim, "lat", "lon"))[:] = values
    return values, dates


class TestCalendar:
    @pytest.mark.parametrize("calendar", [
        "noleap", "365_day", "gregorian", "standard", "proleptic_gregorian",
    ])
    def test_365_day_file_is_byte_identical_to_undated_read(
            self, tmp_path, calendar):
        # 2021 is not a leap year, so every accepted calendar has 365
        # days; the dated read must reproduce the pre-calendar reader
        # exactly, which is what reading the same data with no time
        # coordinate still does
        dated, undated = str(tmp_path / "a.nc"), str(tmp_path / "b.nc")
        _write_dated(dated, (2021, 1, 1), 365, calendar)
        _write_dated(undated, (2021, 1, 1), 365, calendar, coordinate=False)
        g1 = read_bcsd_daily(dated, "tasmax")
        g0 = read_bcsd_daily(undated, "tasmax")
        assert g0.dates is None
        assert g1.values.dtype == g0.values.dtype
        assert g1.values.shape == g0.values.shape
        assert g1.values.tobytes() == g0.values.tobytes()
        assert g1.lat.tobytes() == g0.lat.tobytes()
        assert g1.lon.tobytes() == g0.lon.tobytes()

    def test_365_day_values_keep_file_order(self, tmp_path):
        p = str(tmp_path / "a.nc")
        raw, _ = _write_dated(p, (2021, 1, 1), 365, "noleap")
        grid = read_bcsd_daily(p, "tasmax")
        # time order untouched; longitudes rolled exactly as before
        assert np.array_equal(grid.values, raw[:, :, [2, 3, 0, 1]])

    @pytest.mark.parametrize("calendar", [
        "gregorian", "standard", "proleptic_gregorian",
    ])
    def test_leap_year_drops_feb_29(self, tmp_path, calendar):
        p = str(tmp_path / "a.nc")
        raw, _ = _write_dated(p, (2020, 1, 1), 366, calendar)
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.values.shape[0] == 365
        assert np.datetime64("2020-02-29") not in grid.dates
        assert grid.dates[58] == np.datetime64("2020-02-28")
        assert grid.dates[59] == np.datetime64("2020-03-01")
        # step 59 of the file was Feb 29; everything else survives in order
        expected = np.delete(raw, 59, axis=0)[:, :, [2, 3, 0, 1]]
        assert np.array_equal(grid.values, expected)

    def test_feb_29_found_by_date_not_position(self, tmp_path):
        # a 366-step file from 2019-03-01 to 2020-02-29 puts Feb 29 last;
        # a positional drop of step 59 would remove 2019-04-29 instead
        p = str(tmp_path / "a.nc")
        raw, dates = _write_dated(p, (2019, 3, 1), 366, "standard")
        assert (dates[-1].month, dates[-1].day) == (2, 29)
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.dates[-1] == np.datetime64("2020-02-28")
        assert np.datetime64("2019-04-29") in grid.dates
        assert np.array_equal(grid.values, raw[:365][:, :, [2, 3, 0, 1]])

    def test_dates_support_month_grouping(self, tmp_path):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2020, 1, 1), 366, "proleptic_gregorian")
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.dates.dtype == np.dtype("datetime64[D]")
        months = grid.dates.astype("datetime64[M]").astype(int) % 12 + 1
        lengths = np.bincount(months)[1:]
        assert lengths.tolist() == [31, 28, 31, 30, 31, 30,
                                    31, 31, 30, 31, 30, 31]

    def test_day_dimension_with_cf_coordinate_decodes(self, tmp_path):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2020, 1, 1), 366, "standard", time_dim="day")
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.values.shape[0] == 365
        assert len(grid.dates) == 365

    def test_missing_calendar_attribute_means_standard(self, tmp_path):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2020, 1, 1), 366, "standard", write_calendar=False)
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.values.shape[0] == 365

    def test_360_day_raises_as_undecided(self, tmp_path):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2021, 1, 1), 360, "360_day")
        with pytest.raises(ValueError, match="no rule has been decided"):
            read_bcsd_daily(p, "tasmax")

    @pytest.mark.parametrize("calendar", ["julian", "all_leap", "366_day"])
    def test_other_calendars_raise(self, tmp_path, calendar):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2021, 1, 1), 365, calendar)
        with pytest.raises(ValueError, match="is not supported"):
            read_bcsd_daily(p, "tasmax")

    def test_partial_year_reads_by_default(self, tmp_path):
        # Jan 1 to Mar 31 2020 on a Gregorian calendar: 91 steps, one of
        # them Feb 29. The default reader still drops it by date and
        # returns dates, but does not check how many days remain.
        p = str(tmp_path / "partial.nc")
        raw, _ = _write_dated(p, (2020, 1, 1), 91, "standard")
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.values.shape[0] == 90
        assert len(grid.dates) == 90
        assert np.datetime64("2020-02-29") not in grid.dates
        assert grid.dates[-1] == np.datetime64("2020-03-31")
        expected = np.delete(raw, 59, axis=0)[:, :, [2, 3, 0, 1]]
        assert np.array_equal(grid.values, expected)

    def test_partial_year_rejected_when_full_year_required(self, tmp_path):
        p = str(tmp_path / "partial.nc")
        _write_dated(p, (2020, 1, 1), 91, "standard")
        with pytest.raises(
                ValueError,
                match=r"partial\.nc: 90 days remain after dropping Feb 29 "
                      r"\(calendar 'standard', 91 in the file\)"):
            read_bcsd_daily(p, "tasmax", require_full_year=True)

    def test_full_year_passes_when_required(self, tmp_path):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2020, 1, 1), 366, "gregorian")
        grid = read_bcsd_daily(p, "tasmax", require_full_year=True)
        assert grid.values.shape[0] == 365

    def test_require_full_year_is_keyword_only(self, tmp_path):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2021, 1, 1), 365, "noleap")
        with pytest.raises(TypeError):
            read_bcsd_daily(p, "tasmax", True)

    def test_unsupported_calendar_rejected_regardless(self, tmp_path):
        p = str(tmp_path / "a.nc")
        _write_dated(p, (2021, 1, 1), 30, "360_day")
        with pytest.raises(ValueError, match="no rule has been decided"):
            read_bcsd_daily(p, "tasmax")
        with pytest.raises(ValueError, match="no rule has been decided"):
            read_bcsd_daily(p, "tasmax", require_full_year=True)

    def test_undecodable_day_index_passes_through(self, tmp_path):
        # the pattern files' layout: an integer day index with units
        # 'day' and no reference date; nothing to decode, no 365 rule
        p = str(tmp_path / "a.nc")
        lat = np.array([-0.125, 0.125])
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("day", 3)
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            d = ds.createVariable("day", "i4", ("day",))
            d.units = "day"
            d[:] = [1, 2, 3]
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lat
            ds.createVariable("tasmax", "f4", ("day", "lat", "lon"))[:] = 1.0
        grid = read_bcsd_daily(p, "tasmax")
        assert grid.values.shape[0] == 3
        assert grid.dates is None

    def test_positional_construction_unchanged(self):
        v = np.zeros((1, 1, 1), dtype="f4")
        grid = DailyGrid(v, np.zeros(1), np.zeros(1))
        assert grid.dates is None


def _year(base_max=300.0, base_min=280.0):
    shape = (365, 2, 2)
    return (np.full(shape, base_max, dtype="f4"),
            np.full(shape, base_min, dtype="f4"))


class TestDerive:
    def test_means_are_celsius_of_the_average(self):
        tasmax, tasmin = _year(300.0, 280.0)
        d = derive_daily(tasmax, tasmin, december_prev=None)
        assert np.allclose(d.tas_annual, (290.0 - 273.15))
        assert np.allclose(d.tas_jja, (290.0 - 273.15))
        assert d.n_days == 365
        assert d.n_days_jja == 92

    def test_thresholds_use_the_right_extremes(self):
        tasmax, tasmin = _year(300.0, 280.0)
        # ten hot days on tasmax, five cold days on tasmin
        tasmax[:10] = 309.0
        tasmin[:5] = 270.0
        d = derive_daily(tasmax, tasmin, december_prev=None)
        assert np.all(d.days_above_95F == 10)
        assert np.all(d.days_below_32F == 5)
        # a hot tasmin or a cold tasmax must not count
        tasmax2, tasmin2 = _year(309.0, 270.0)
        d2 = derive_daily(tasmax2, tasmin2, december_prev=None)
        assert np.all(d2.days_above_95F == 365)
        assert np.all(d2.days_below_32F == 365)

    def test_swapped_extremes_are_corrected_before_counting(self):
        tasmax, tasmin = _year(300.0, 280.0)
        # ten cell-days arrive swapped: the "min" field carries a value
        # above the hot threshold while the "max" field sits below it
        tasmax[:10, 0, 0] = 275.0
        tasmin[:10, 0, 0] = 309.0
        d = derive_daily(tasmax, tasmin, december_prev=None)
        assert d.n_swapped == 10
        # after elementwise correction the hot count sees the true max
        assert d.days_above_95F[0, 0] == 10
        # and the cold count is not fooled by the swapped-in low
        assert d.days_below_32F[0, 0] == 0
        # the midpoint is symmetric, so means are unchanged by the swap
        d_clean = derive_daily(*_year(300.0, 280.0), december_prev=None)
        assert d_clean.n_swapped == 0

    def test_djf_uses_previous_december(self):
        tasmax, tasmin = _year(290.0, 290.0)
        prev = np.full((31, 2, 2), 250.0)
        d = derive_daily(tasmax, tasmin, december_prev=prev)
        # 31 days at 250 K, 59 days (Jan+Feb) at 290 K
        expected = (31 * 250.0 + 59 * 290.0) / 90 - 273.15
        assert np.allclose(d.tas_djf, expected)
        assert d.n_days_djf == 90
        d_first = derive_daily(tasmax, tasmin, december_prev=None)
        assert d_first.n_days_djf == 59

    def test_december_slice(self):
        tasmax, tasmin = _year(300.0, 280.0)
        tasmax[-31:] = 310.0
        dec = december(tasmax, tasmin)
        assert dec.shape[0] == 31
        assert np.allclose(dec, (310.0 + 280.0) / 2)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="365"):
            derive_daily(np.zeros((360, 2, 2)), np.zeros((360, 2, 2)), None)
