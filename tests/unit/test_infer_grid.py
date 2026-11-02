"""infer_grid_version and cell_indices: derive the grid, then prove it."""
from __future__ import annotations

import numpy as np
import pytest

from cil_regionalization.grid import (
    AmbiguousLongitudeConvention,
    GridSpec,
    cell_indices,
    coordinate_tolerance,
    infer_grid_version,
)


def _centres(n, res, origin):
    return origin + res * (np.arange(n) + 0.5)


class TestInfer:
    def test_quarter_degree_west_hemisphere(self):
        lat = _centres(4, 0.25, -6.25)
        lon = _centres(4, 0.25, -80.0)
        assert infer_grid_version(lat, lon) == "grid-0.25deg-center-[-180,180)"

    def test_zero_360(self):
        lat = _centres(4, 1.0, 10.0)
        lon = _centres(4, 1.0, 200.0)
        assert infer_grid_version(lat, lon) == "grid-1.0deg-center-[0,360)"

    def test_edges(self):
        lat = np.array([10.0, 10.5, 11.0])
        lon = np.array([-20.0, -19.5, -19.0])
        assert infer_grid_version(lat, lon) == "grid-0.5deg-edge-[-180,180)"

    def test_descending_latitude(self):
        lat = _centres(4, 0.25, 40.0)[::-1]
        lon = _centres(4, 0.25, -80.0)
        assert infer_grid_version(lat, lon) == "grid-0.25deg-center-[-180,180)"

    def test_float32_coordinates_near_180(self):
        # 0.1 is not representable; float32 near 180 is off by ~1e-5
        lat = np.arange(-4.95, 5.0, 0.1).astype("float32")
        lon = (-179.95 + 0.1 * np.arange(20)).astype("float32")
        assert infer_grid_version(lat, lon) == "grid-0.1deg-center-[-180,180)"

    def test_longitudes_fitting_both_conventions_raise(self):
        lat = _centres(3, 0.25, 0.0)
        lon = _centres(3, 0.25, 10.0)
        with pytest.raises(ValueError, match="fits both"):
            infer_grid_version(lat, lon)

    def test_ambiguity_carries_both_identities(self):
        lat = _centres(3, 0.25, 0.0)
        lon = _centres(3, 0.25, 10.0)
        with pytest.raises(AmbiguousLongitudeConvention) as err:
            infer_grid_version(lat, lon)
        assert err.value.candidates == [
            "grid-0.25deg-center-[-180,180)",
            "grid-0.25deg-center-[0,360)",
        ]

    def test_ambiguity_keeps_edge_offset(self):
        lat = np.array([10.0, 10.5, 11.0])
        lon = np.array([5.0, 5.5, 6.0])
        with pytest.raises(AmbiguousLongitudeConvention) as err:
            infer_grid_version(lat, lon)
        assert err.value.candidates[0] == "grid-0.5deg-edge-[-180,180)"

    def test_longitudes_fitting_neither_raise(self):
        lat = _centres(3, 1.0, 0.0)
        with pytest.raises(ValueError, match="neither"):
            infer_grid_version(lat, np.arange(-10.5, 200.0, 1.0))

    def test_different_lat_and_lon_steps_raise(self):
        with pytest.raises(ValueError, match="square cells"):
            infer_grid_version(_centres(3, 0.25, 0.0), _centres(3, 0.5, -80.0))

    def test_uneven_spacing_raises(self):
        lat = np.array([0.125, 0.375, 0.875])
        lon = _centres(3, 0.25, -80.0)
        with pytest.raises(ValueError, match="not evenly spaced"):
            infer_grid_version(lat, lon)

    def test_coordinates_on_no_grid_raise(self):
        lat = np.array([0.0, 0.25, 0.5]) + 0.07
        lon = _centres(3, 0.25, -80.0)
        with pytest.raises(ValueError, match="neither all cell centres"):
            infer_grid_version(lat, lon)

    def test_centre_latitudes_with_edge_longitudes_raise(self):
        lat = _centres(3, 0.25, 0.0)
        lon = np.array([-80.0, -79.75, -79.5])
        with pytest.raises(ValueError, match="disagree"):
            infer_grid_version(lat, lon)

    def test_single_value_raises(self):
        with pytest.raises(ValueError, match="cannot derive"):
            infer_grid_version(np.array([0.125]), np.array([-80.125, -79.875]))

    def test_resolution_that_does_not_tile_the_globe_raises(self):
        lat = _centres(3, 0.7, 0.0)
        lon = _centres(3, 0.7, -80.0)
        with pytest.raises(ValueError):
            GridSpec.from_source_version(infer_grid_version(lat, lon))


class TestCellIndices:
    spec = GridSpec.from_source_version("grid-0.25deg-center-[-180,180)")

    def test_wraps_longitudes(self):
        ix = cell_indices(self.spec, [180.125, -179.875], "lon")
        assert ix.tolist() == [0, 0]

    def test_edges_where_the_grid_has_centres_raise(self):
        with pytest.raises(ValueError, match="lat values are not cell centres"):
            cell_indices(self.spec, [-6.0, -5.75], "lat")

    def test_outside_the_grid_raises(self):
        with pytest.raises(ValueError, match="lat values are not cell centres"):
            cell_indices(self.spec, [90.125], "lat")

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="lon values"):
            cell_indices(self.spec, [np.nan], "lon")


class TestTolerance:
    def test_float64_keeps_the_strict_default(self):
        c = np.array([179.875, 0.125])
        assert coordinate_tolerance(c, 0.25) == pytest.approx(2.5e-7)

    def test_float32_is_loosened_to_its_own_precision(self):
        c = np.array([179.95], dtype="float32")
        assert coordinate_tolerance(c, 0.1) > 1e-5

    def test_dtype_too_coarse_for_the_resolution_raises(self):
        c = np.array([179.99], dtype="float32")
        with pytest.raises(ValueError, match="too coarsely"):
            coordinate_tolerance(c, 0.001)
