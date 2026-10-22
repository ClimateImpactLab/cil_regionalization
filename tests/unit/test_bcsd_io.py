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
