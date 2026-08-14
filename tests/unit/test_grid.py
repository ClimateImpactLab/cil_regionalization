"""Grid index math, conventions, polygons, and prebuilt-grid spacing check."""
from __future__ import annotations

import numpy as np
import pytest

from segment_weights.config import GridConfig
from segment_weights.grid import GridSpec, infer_uniform_resolution


def _grid(
    mode="generate",
    resolution=1.0,
    offset="center",
    lon_convention="[-180,180)",
    **kwargs,
) -> GridSpec:
    cfg = GridConfig(
        mode=mode,
        resolution=resolution,
        offset=offset,
        lon_convention=lon_convention,
        **kwargs,
    )
    return GridSpec.from_config(cfg)


class TestS51Case:
    """The s51 target grid: 1deg, center offset, [-180,180). 360x180 cells.

    Verifying the corner cells match what example_s51_cds_clean_grid.nc holds:
    centroids on the half-degree, ix=0 at lon=-179.5, iy=0 at lat=-89.5.
    That file carries a leftover GRIB attribute claiming a 0-360
    longitude convention from before its cfgrib conversion; the actual
    lon axis is [-180,180) and is what counts. Trust axes, not
    attributes.
    """

    def test_dimensions(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        assert g.n_ix == 360
        assert g.n_iy == 180

    def test_first_cell_centroid(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        lon, lat = g.centroid(0, 0)
        assert float(lon) == pytest.approx(-179.5)
        assert float(lat) == pytest.approx(-89.5)

    def test_last_cell_centroid(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        lon, lat = g.centroid(359, 179)
        assert float(lon) == pytest.approx(179.5)
        assert float(lat) == pytest.approx(89.5)

    def test_domain(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        assert g.domain_lon == (-180.0, 180.0)
        assert g.domain_lat == (-90.0, 90.0)


class TestCenterOffset:
    """Default `center` offset across both lon conventions."""

    def test_neg180_first_centroid(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        lon, lat = g.centroid(0, 0)
        assert float(lon) == pytest.approx(-179.5)
        assert float(lat) == pytest.approx(-89.5)

    def test_neg180_last_centroid(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        lon, _ = g.centroid(359, 0)
        assert float(lon) == pytest.approx(179.5)

    def test_fractional_resolution(self):
        g = _grid(resolution=0.25, lon_convention="[-180,180)")
        assert g.n_ix == 1440
        assert g.n_iy == 720
        lon, lat = g.centroid(0, 0)
        assert float(lon) == pytest.approx(-180.0 + 0.125)
        assert float(lat) == pytest.approx(-90.0 + 0.125)


class TestEdgeOffset:
    """Edge offset puts centroids on integer multiples of res, anchored at origin."""

    def test_first_cell_centroid_at_origin(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)", offset="edge")
        lon, lat = g.centroid(0, 0)
        assert float(lon) == pytest.approx(0.0)
        assert float(lat) == pytest.approx(-90.0)

    def test_centroid_is_integer_multiple(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)", offset="edge")
        lon, _ = g.centroid(5, 0)
        assert float(lon) == pytest.approx(5.0)


class TestIndexRoundTrip:
    """Round-trip: index -> centroid -> index recovers the original index."""

    @pytest.mark.parametrize("convention", ["[-180,180)", "[0,360)"])
    @pytest.mark.parametrize("offset", ["center", "edge"])
    @pytest.mark.parametrize("res", [1.0, 0.5, 0.25])
    def test_round_trip(self, convention, offset, res):
        g = _grid(resolution=res, lon_convention=convention, offset=offset)
        ix_in = np.array([0, 1, 7, g.n_ix // 2, g.n_ix - 1])
        iy_in = np.array([0, 3, 11, g.n_iy // 2, g.n_iy - 1])
        lon, lat = g.centroid(ix_in, iy_in)
        ix_out, iy_out = g.index_of(lon, lat)
        np.testing.assert_array_equal(ix_in, ix_out)
        np.testing.assert_array_equal(iy_in, iy_out)


class TestIndexOf:
    def test_scalar_in_first_cell_center(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        ix, iy = g.index_of(-179.5, -89.5)
        assert int(ix) == 0
        assert int(iy) == 0

    def test_arbitrary_point(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        ix, iy = g.index_of(123.4, -45.6)
        # 123.4 -> floor((123.4 - -180)/1) = floor(303.4) = 303; centroid 123.5
        assert int(ix) == 303
        assert int(iy) == 44  # -45.6 -> floor((-45.6 - -90)/1) = floor(44.4) = 44

    def test_array_input(self):
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        lons = np.array([-179.5, 123.4, 179.5])
        lats = np.array([-89.5, -45.6, 89.5])
        ix, iy = g.index_of(lons, lats)
        np.testing.assert_array_equal(ix, [0, 303, 359])
        np.testing.assert_array_equal(iy, [0, 44, 179])

    def test_at_exact_cell_boundary_uses_upper(self):
        # Half-open extent convention: a point exactly on the cell boundary
        # belongs to the upper (right/north) cell.
        g = _grid(resolution=1.0, lon_convention="[-180,180)")
        ix, _ = g.index_of(-179.0, 0.0)
        assert int(ix) == 1


class TestCellPolygon:
    def test_polygon_extent_matches_centroid(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        poly = g.cell_polygon(0, 0)
        minx, miny, maxx, maxy = poly.bounds
        assert (minx, miny, maxx, maxy) == pytest.approx((0.0, -90.0, 1.0, -89.0))

    def test_polygon_area_in_degrees(self):
        g = _grid(resolution=0.5, lon_convention="[-180,180)")
        poly = g.cell_polygon(0, 0)
        assert poly.area == pytest.approx(0.25)

    def test_edge_offset_polygon_centered_on_integer(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)", offset="edge")
        poly = g.cell_polygon(5, 5)
        minx, miny, maxx, maxy = poly.bounds
        # Cell 5/5 centroid is (5, -85); extent is [4.5, 5.5] x [-85.5, -84.5]
        assert (minx, miny, maxx, maxy) == pytest.approx((4.5, -85.5, 5.5, -84.5))

    def test_vectorised_polygons(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        polys = g.cell_polygons([0, 1, 2], [0, 0, 0])
        assert len(polys) == 3
        # Each cell is 1deg x 1deg
        for p in polys:
            assert p.area == pytest.approx(1.0)


class TestBboxRange:
    def test_small_bbox_in_single_cell(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        ix_r, iy_r = g.index_range_for_bbox(0.2, -89.8, 0.3, -89.7)
        assert list(ix_r) == [0]
        assert list(iy_r) == [0]

    def test_bbox_spans_multiple_cells(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        ix_r, iy_r = g.index_range_for_bbox(2.1, -88.9, 4.9, -87.1)
        assert list(ix_r) == [2, 3, 4]
        # lat range -88.9 to -87.1 → cells 1, 2 (centroids -88.5, -87.5)
        assert list(iy_r) == [1, 2]

    def test_bbox_clipped_to_domain(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        # bbox going slightly past the global lat domain
        ix_r, iy_r = g.index_range_for_bbox(0.1, -91.0, 0.5, -89.6)
        assert list(ix_r) == [0]
        assert list(iy_r) == [0]  # negative iy clipped away

    def test_bbox_entirely_outside_domain_returns_empty(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        ix_r, iy_r = g.index_range_for_bbox(-10.0, -95.0, -5.0, -91.0)
        assert list(ix_r) == []
        assert list(iy_r) == []

    def test_bbox_does_not_wrap_antimeridian(self):
        # A bbox that crosses lon=360 should NOT produce cells starting at 0.
        # The caller is responsible for splitting; the grid clips to domain.
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        ix_r, _ = g.index_range_for_bbox(358.5, 0.0, 361.5, 1.0)
        assert list(ix_r) == [358, 359]

    def test_empty_bbox_raises(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        with pytest.raises(ValueError, match="empty"):
            g.index_range_for_bbox(5.0, 0.0, 4.0, 1.0)


class TestEnumerateCells:
    def test_meshgrid_flatten(self):
        g = _grid(resolution=1.0, lon_convention="[0,360)")
        ix, iy = g.enumerate_cells(range(0, 2), range(0, 3))
        assert sorted(zip(ix.tolist(), iy.tolist())) == sorted(
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        )


class TestNonTilingResolution:
    def test_non_divisible_resolution_rejected(self):
        with pytest.raises(ValueError, match="tile the globe evenly"):
            _grid(resolution=0.7)


class TestUniformResolutionInference:
    def test_uniform_passes(self):
        coords = np.arange(-179.5, 180.0, 1.0)
        assert infer_uniform_resolution(coords) == pytest.approx(1.0)

    def test_non_uniform_rejected(self):
        coords = [0.0, 1.0, 2.0, 4.0]
        with pytest.raises(ValueError, match="non-uniform"):
            infer_uniform_resolution(coords)

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="at least two"):
            infer_uniform_resolution([5.0])

    def test_decreasing_coords_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            infer_uniform_resolution([5.0, 3.0, 1.0])
