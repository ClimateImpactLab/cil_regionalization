"""read_netcdf_grid: a lat-lon NetCDF in, a verified cell table out."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

import cil_regionalization as cilreg
from cil_regionalization.bcsd_io import BCSD_GRID, DailyGrid
from cil_regionalization.grid import GridSpec

GRID = "grid-0.25deg-center-[-180,180)"


def _write(path, lat, lon, *, nt=None, name="tas", values=None, dtype="float64"):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    shape = (len(lat), len(lon)) if nt is None else (nt, len(lat), len(lon))
    dims = ("lat", "lon") if nt is None else ("time", "lat", "lon")
    if values is None:
        values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    ds = xr.Dataset(
        {name: (dims, values)},
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    return path


LAT = np.array([-6.125, -5.875])
LON = np.array([106.125, 106.375, 106.625])


class TestAgreesWithDailyGrid:
    def test_same_cells_as_to_cells(self, tmp_path):
        values = np.arange(6, dtype="float64").reshape(2, 3)
        path = _write(tmp_path / "a.nc", LAT, LON, values=values)
        # 106 degrees east fits both conventions, so the grid is named
        cells, grid = cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)
        assert grid == BCSD_GRID == GRID
        daily = DailyGrid(values=values[None].astype("float32"), lat=LAT, lon=LON)
        want = daily.to_cells({"tas": values})
        pd.testing.assert_frame_equal(
            cells, want.astype({"tas": "float64"}), check_dtype=False
        )

    def test_descending_latitude_keeps_values_with_their_cells(self, tmp_path):
        values = np.arange(6, dtype="float64").reshape(2, 3)
        path = _write(tmp_path / "d.nc", LAT[::-1], LON, values=values)
        cells, _ = cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)
        spec = GridSpec.from_source_version(GRID)
        c_lon, c_lat = spec.centroid(cells["cell_ix"], cells["cell_iy"])
        # row 0 of the file is the northern latitude, -5.875
        first = cells.iloc[0]
        assert (c_lat[0], c_lon[0], first["tas"]) == (-5.875, 106.125, 0.0)
        south = cells[np.isclose(c_lat, -6.125)]
        assert south["tas"].tolist() == [3.0, 4.0, 5.0]


class TestConventions:
    def test_zero_360_file_keeps_its_identity(self, tmp_path):
        path = _write(tmp_path / "z.nc", [0.125, 0.375], [200.125, 200.375])
        cells, grid = cilreg.read_netcdf_grid(path, ["tas"])
        assert grid == "grid-0.25deg-center-[0,360)"
        assert cells["cell_ix"].tolist() == [800, 801] * 2

    def test_zero_360_file_put_on_minus_180_grid(self, tmp_path):
        path = _write(tmp_path / "z.nc", [0.125, 0.375], [200.125, 200.375])
        cells, grid = cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)
        assert grid == GRID
        # 200.125 is -159.875: floor((-159.875 + 180) / 0.25) = 80
        assert cells["cell_ix"].tolist() == [80, 81] * 2
        assert cells["tas"].tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_ambiguous_longitudes_need_the_grid(self, tmp_path):
        path = _write(tmp_path / "e.nc", [0.125, 0.375], [10.125, 10.375])
        with pytest.raises(ValueError, match="fits both"):
            cilreg.read_netcdf_grid(path, ["tas"])
        cells, grid = cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)
        assert grid == GRID
        assert len(cells) == 4

    def test_float32_tenth_degree_file(self, tmp_path):
        lat = np.arange(-4.95, 5.0, 0.1).astype("float32")
        lon = (-179.95 + 0.1 * np.arange(20)).astype("float32")
        path = _write(tmp_path / "f.nc", lat, lon, dtype="float32")
        cells, grid = cilreg.read_netcdf_grid(path, ["tas"])
        assert grid == "grid-0.1deg-center-[-180,180)"
        assert len(cells) == len(lat) * len(lon)


class TestNamingTheGrid:
    def _east(self, tmp_path):
        path = tmp_path / "east.nc"
        ds = xr.Dataset(
            {"tas": (("lat", "lon"), np.arange(6.0).reshape(2, 3))},
            coords={"lat": LAT, "lon": LON},
        )
        ds.to_netcdf(path)
        return path

    def test_the_error_prints_both_calls(self, tmp_path):
        path = self._east(tmp_path)
        with pytest.raises(ValueError) as err:
            cilreg.read_netcdf_grid(path, ["tas"])
        msg = str(err.value)
        for g in ("grid-0.25deg-center-[-180,180)", "grid-0.25deg-center-[0,360)"):
            assert f"read_netcdf_grid({str(path)!r}, ['tas'], grid={g!r})" in msg
        assert "source_version" in msg

    def test_the_printed_calls_carry_nondefault_names(self, tmp_path):
        ds = xr.Dataset(
            {"tas": (("latitude", "longitude"), np.zeros((2, 3)))},
            coords={"latitude": LAT, "longitude": LON},
        )
        path = tmp_path / "names.nc"
        ds.to_netcdf(path)
        with pytest.raises(ValueError) as err:
            cilreg.read_netcdf_grid(path, ["tas"], lat="latitude", lon="longitude")
        assert "lat='latitude', lon='longitude', grid=" in str(err.value)

    def test_each_printed_call_works(self, tmp_path):
        path = self._east(tmp_path)
        with pytest.raises(ValueError) as err:
            cilreg.read_netcdf_grid(path, ["tas"])
        for line in str(err.value).splitlines():
            if line.strip().startswith("read_netcdf_grid("):
                cells, grid = eval(line.strip(), {"read_netcdf_grid": cilreg.read_netcdf_grid})
                assert grid in line
                assert len(cells) == 6

    def test_a_gridspec_is_accepted(self, tmp_path):
        path = self._east(tmp_path)
        spec = GridSpec.from_source_version("grid-0.25deg-center-[0,360)")
        by_spec, grid = cilreg.read_netcdf_grid(path, ["tas"], grid=spec)
        by_name, _ = cilreg.read_netcdf_grid(path, ["tas"], grid=spec.source_version)
        assert grid == "grid-0.25deg-center-[0,360)"
        pd.testing.assert_frame_equal(by_spec, by_name)

    def test_a_prebuilt_gridspec_is_refused(self, tmp_path):
        from cil_regionalization.config import GridConfig

        path = self._east(tmp_path)
        spec = GridSpec.from_config(GridConfig(mode="prebuilt", path="x.parquet"))
        with pytest.raises(ValueError, match="generated grid"):
            cilreg.read_netcdf_grid(path, ["tas"], grid=spec)

    def test_a_file_that_needs_no_naming_is_not_asked(self, tmp_path):
        path = _write(tmp_path / "w.nc", [0.125, 0.375], [200.125, 200.375])
        _, grid = cilreg.read_netcdf_grid(path, ["tas"])
        assert grid.endswith("[0,360)")


class TestRefusals:
    def test_edges_where_the_grid_has_centres(self, tmp_path):
        path = _write(tmp_path / "x.nc", [-6.0, -5.75], [106.125, 106.375])
        with pytest.raises(ValueError, match="not cell centres"):
            cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)

    def test_a_cell_listed_twice(self, tmp_path):
        lon = np.array([-180.0 + 0.125, 180.0 + 0.125 - 360.0 + 360.0])
        path = _write(tmp_path / "dup.nc", [0.125], lon)
        with pytest.raises(ValueError, match="one cell"):
            cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)

    def test_uneven_longitudes(self, tmp_path):
        path = _write(tmp_path / "u.nc", [0.125, 0.375], [-80.125, -79.875, -79.375])
        with pytest.raises(ValueError, match="not evenly spaced"):
            cilreg.read_netcdf_grid(path, ["tas"])

    def test_missing_variable(self, tmp_path):
        path = _write(tmp_path / "m.nc", LAT, LON)
        with pytest.raises(ValueError, match="no variables"):
            cilreg.read_netcdf_grid(path, ["pr"])

    def test_names_that_are_not_coordinates(self, tmp_path):
        path = _write(tmp_path / "n.nc", LAT, LON)
        with pytest.raises(ValueError, match="1-D coordinate"):
            cilreg.read_netcdf_grid(path, ["tas"], lat="latitude")

    def test_two_dimensional_coordinates(self, tmp_path):
        ds = xr.Dataset(
            {"tas": (("y", "x"), np.zeros((2, 2)))},
            coords={"lat": (("y", "x"), np.zeros((2, 2))),
                    "lon": (("y", "x"), np.zeros((2, 2)))},
        )
        path = tmp_path / "curv.nc"
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="1-D coordinate"):
            cilreg.read_netcdf_grid(path, ["tas"])


class TestMaskedCells:
    def test_cells_masked_everywhere_are_left_out(self, tmp_path):
        values = np.arange(12, dtype="float64").reshape(2, 2, 3)
        values[:, 0, 1] = np.nan
        path = _write(tmp_path / "k.nc", LAT, LON, nt=2, values=values)
        cells, _ = cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)
        assert len(cells) == (6 - 1) * 2
        assert cells["tas"].notna().all()
        assert "time" in cells.columns

    def test_cells_masked_in_some_slices_raise(self, tmp_path):
        values = np.arange(12, dtype="float64").reshape(2, 2, 3)
        values[0, 0, 1] = np.nan
        path = _write(tmp_path / "r.nc", LAT, LON, nt=2, values=values)
        with pytest.raises(ValueError, match="masked in some slices"):
            cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)


class TestWithApplyWeights:
    def _weights(self, grid):
        spec = GridSpec.from_source_version(grid)
        ix, iy = np.meshgrid(spec.index_of(LON, LAT[0])[0],
                             spec.index_of(LON[0], LAT)[1])
        table = pd.DataFrame({
            "region": "R",
            "cell_ix": ix.ravel(), "cell_iy": iy.ravel(),
            "popwt": [0.25, 0.125, 0.125] * 2,
        })
        return cilreg.WeightsArtifact.from_table(
            table, id_fields=["region"], weights={"pop": "popwt"}, source=grid,
        )

    def test_weighted_mean_of_the_file(self, tmp_path):
        values = np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]])
        path = _write(tmp_path / "w.nc", LAT, LON, values=values)
        cells, grid = cilreg.read_netcdf_grid(path, ["tas"], grid=GRID)
        out = cilreg.apply_weights(
            self._weights(GRID), cells, kind="intensive", weight="pop",
            value_col="tas", data_version=grid,
        )
        assert out.frame["tas"].tolist() == [0.25 * 10 + 0.125 * 20 + 0.125 * 30]

    def test_a_file_on_another_convention_does_not_apply(self, tmp_path):
        path = _write(tmp_path / "w.nc", [-6.125, -5.875], [200.125, 200.375])
        cells, grid = cilreg.read_netcdf_grid(path, ["tas"])
        assert grid.endswith("[0,360)")
        with pytest.raises(ValueError):
            cilreg.apply_weights(
                self._weights(GRID), cells, kind="intensive", weight="pop",
                value_col="tas", data_version=grid,
            )
