"""GDPCIR reader contract checks."""
from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")

from cil_regionalization.gdpcir_io import gdpcir_year, open_gdpcir_daily


def _store(tmp_path, lon=None, years=1, calendar="noleap"):
    lon = np.array([-179.875, 0.125]) if lon is None else lon
    lat = np.array([-0.125, 0.125])
    ndays = 365 * years if calendar == "noleap" else None
    time = xr.date_range("2050-01-01", periods=ndays or (365 * years + 1),
                         freq="D", calendar=calendar, use_cftime=True)
    data = np.arange(len(time) * 2 * len(lon), dtype="f4").reshape(
        len(time), 2, len(lon))
    ds = xr.Dataset(
        {"tasmax": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    p = str(tmp_path / "s.zarr")
    ds.to_zarr(p)
    return p


class TestOpen:
    def test_valid_store_opens(self, tmp_path):
        ds = open_gdpcir_daily(_store(tmp_path), "tasmax")
        assert "tasmax" in ds

    def test_wrong_variable_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="tasmin"):
            open_gdpcir_daily(_store(tmp_path), "tasmin")

    def test_zero_360_longitudes_rejected(self, tmp_path):
        p = _store(tmp_path, lon=np.array([0.125, 180.125]))
        with pytest.raises(ValueError, match=r"\[-180,180\)"):
            open_gdpcir_daily(p, "tasmax")

    def test_descending_coords_rejected(self, tmp_path):
        p = _store(tmp_path, lon=np.array([0.125, -0.125]))
        with pytest.raises(ValueError, match="ascending"):
            open_gdpcir_daily(p, "tasmax")


class TestYear:
    def test_full_year_shape_and_dtype(self, tmp_path):
        ds = open_gdpcir_daily(_store(tmp_path), "tasmax")
        arr = gdpcir_year(ds, "tasmax", 2050)
        assert arr.shape == (365, 2, 2)
        assert arr.dtype == np.float32

    def test_short_year_rejected(self, tmp_path):
        ds = open_gdpcir_daily(_store(tmp_path), "tasmax")
        with pytest.raises(ValueError, match="truncated"):
            gdpcir_year(ds, "tasmax", 2051)

    def test_leap_calendar_year_rejected(self, tmp_path):
        # a standard-calendar store holds 366 days in a leap year; the
        # reader must refuse rather than average around the extra day
        p = _store(tmp_path, years=3, calendar="standard")
        ds = open_gdpcir_daily(p, "tasmax")
        with pytest.raises(ValueError, match="365"):
            gdpcir_year(ds, "tasmax", 2052)
