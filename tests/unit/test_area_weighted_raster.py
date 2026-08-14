"""Unit pins for the area_weighted_sum derived raster materialization.

For a CROPLAND-FRACTION source raster, the derived raster's pixel value
must equal source_value * geodesic_area_of_that_pixel. The area is
longitude-invariant on a regular lat/lon grid, so each row has a single
pixel area.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Geod
from rasterio.transform import from_origin
import shapely.geometry as sg

from cil_regionalization.backends.local import _materialize_area_weighted_raster


@pytest.fixture
def fraction_raster(tmp_path: Path) -> Path:
    """4x4 GeoTIFF at 1deg resolution; non-trivial per-row pattern."""
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    data = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.50, 0.60, 0.70, 0.80],
            [0.00, 0.10, 0.20, 0.30],
            [0.99, 0.99, 0.99, 0.99],
        ],
        dtype=np.float32,
    )
    p = tmp_path / "frac.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as ds:
        ds.write(data, 1)
    return p


class TestAreaWeightedMaterialization:
    def test_derived_pixel_equals_fraction_times_pixel_area(self, fraction_raster):
        out_path = _materialize_area_weighted_raster(str(fraction_raster))
        assert Path(out_path).exists()

        with rasterio.open(fraction_raster) as src:
            frac = src.read(1).astype(np.float64)
            transform = src.transform
        with rasterio.open(out_path) as dst:
            derived = dst.read(1).astype(np.float64)
            assert dst.dtypes[0] == "float64"

        geod = Geod(ellps="WGS84")
        north = float(transform.f)
        lat_step = float(abs(transform.e))
        lon_step = float(abs(transform.a))
        for r in range(4):
            lat_hi = north - r * lat_step
            lat_lo = lat_hi - lat_step
            poly = sg.box(0.0, lat_lo, lon_step, lat_hi)
            area_m2, _ = geod.geometry_area_perimeter(poly)
            area_m2 = abs(area_m2)
            for c in range(4):
                expected = frac[r, c] * area_m2
                assert derived[r, c] == pytest.approx(expected, rel=1e-12)

    def test_area_is_longitude_invariant(self, fraction_raster):
        """Two columns in the same row must have identical area weights when
        the source fraction is the same; this confirms the per-row
        precompute is correct.
        """
        out_path = _materialize_area_weighted_raster(str(fraction_raster))
        with rasterio.open(out_path) as dst:
            derived = dst.read(1).astype(np.float64)
        # Row 3 has constant 0.99; all four columns should produce the
        # same derived value (one pixel area per row).
        assert np.allclose(derived[3, :], derived[3, 0])

    def test_caches_per_source(self, fraction_raster):
        """Calling twice on the same source returns the same path."""
        first = _materialize_area_weighted_raster(str(fraction_raster))
        second = _materialize_area_weighted_raster(str(fraction_raster))
        assert first == second

    def test_higher_latitude_pixel_smaller_area(self, fraction_raster):
        """Geodesic pixel area decreases as latitude increases. Row 0 (the
        northernmost) should produce a smaller per-fraction area than
        row 3 (closer to equator) for the same source fraction.
        """
        out_path = _materialize_area_weighted_raster(str(fraction_raster))
        with rasterio.open(fraction_raster) as src:
            frac = src.read(1).astype(np.float64)
        with rasterio.open(out_path) as dst:
            derived = dst.read(1).astype(np.float64)

        # Reduce to per-row "area per unit fraction": derived / frac
        # at any non-zero column.
        per_row_area_top = derived[0, 0] / frac[0, 0]
        per_row_area_bot = derived[3, 0] / frac[3, 0]
        # Row 3 sits at [0, 1] latitude (near equator);
        # row 0 sits at [3, 4] latitude. Equator pixel area is bigger.
        assert per_row_area_bot > per_row_area_top
