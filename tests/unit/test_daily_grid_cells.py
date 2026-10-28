"""DailyGrid.to_cells: per-cell arrays to the table apply_weights takes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cil_regionalization as cilreg
from cil_regionalization.bcsd_io import BCSD_GRID, DailyGrid
from cil_regionalization.grid import GridSpec


def _grid(lat, lon):
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    values = np.zeros((1, len(lat), len(lon)), dtype="float32")
    return DailyGrid(values=values, lat=lat, lon=lon)


class TestIndices:
    def test_indices_count_from_minus_180_and_minus_90(self):
        g = _grid([-6.125, -5.875], [106.125, 106.375, 106.625])
        field = np.arange(6, dtype="float64").reshape(2, 3)
        out = g.to_cells({"x": field})
        assert list(out.columns) == ["cell_ix", "cell_iy", "x"]
        # floor((106.125 + 180) / 0.25) = 1144, floor((-6.125 + 90) / 0.25) = 335
        assert out["cell_ix"].tolist() == [1144, 1145, 1146] * 2
        assert out["cell_iy"].tolist() == [335] * 3 + [336] * 3
        # row-major: each value stays with its own (lat, lon)
        assert out["x"].tolist() == field.reshape(-1).tolist()

    def test_western_and_edge_of_domain_cells(self):
        g = _grid([-89.875, 89.875], [-179.875, -0.125, 179.875])
        out = g.to_cells({"x": np.ones((2, 3))})
        assert sorted(set(out["cell_ix"])) == [0, 719, 1439]
        assert sorted(set(out["cell_iy"])) == [0, 719]

    def test_matches_grid_spec_centroids(self):
        lat = np.array([-30.125, 10.375, 52.625])
        lon = np.array([-75.875, 13.125, 141.375])
        out = _grid(lat, lon).to_cells({"x": np.ones((3, 3))})
        spec = GridSpec.from_source_version(BCSD_GRID)
        c_lon, c_lat = spec.centroid(out["cell_ix"], out["cell_iy"])
        assert np.allclose(c_lon, np.tile(lon, 3))
        assert np.allclose(c_lat, np.repeat(lat, 3))

    def test_zero_360_grid_wraps_longitudes(self):
        g = _grid([0.125], [-179.875, 0.125])
        out = g.to_cells({"x": np.ones((1, 2))},
                         grid="grid-0.25deg-center-[0,360)")
        # -179.875 is 180.125 on a grid that starts at 0
        assert out["cell_ix"].tolist() == [720, 0]


class TestRejects:
    def test_cell_edges_instead_of_centres(self):
        g = _grid([-6.0, -5.75], [106.125])
        with pytest.raises(ValueError, match="lat values are not cell centres"):
            g.to_cells({"x": np.ones((2, 1))})

    def test_longitudes_off_the_grid(self):
        g = _grid([0.125], [106.0, 106.25])
        with pytest.raises(ValueError, match="2 lon values are not cell centres"):
            g.to_cells({"x": np.ones((1, 2))})

    def test_another_resolution(self):
        # 0.125 is a centre at 0.25 degrees but an edge at 0.5
        g = _grid([0.125], [0.125])
        with pytest.raises(ValueError, match="not cell centres"):
            g.to_cells({"x": np.ones((1, 1))},
                       grid="grid-0.5deg-center-[-180,180)")

    def test_edge_offset_grid(self):
        g = _grid([0.125], [0.125])
        with pytest.raises(ValueError, match="not cell centres"):
            g.to_cells({"x": np.ones((1, 1))},
                       grid="grid-0.25deg-edge-[-180,180)")

    def test_field_with_a_time_axis(self):
        g = _grid([0.125, 0.375], [0.125])
        with pytest.raises(ValueError, match=r"expected \(lat, lon\)"):
            g.to_cells({"x": g.values})

    def test_malformed_grid(self):
        g = _grid([0.125], [0.125])
        with pytest.raises(ValueError, match="not a grid identity"):
            g.to_cells({"x": np.ones((1, 1))}, grid="bcsd-0.25")

    def test_non_canonical_spelling(self):
        g = _grid([0.125], [0.125])
        with pytest.raises(ValueError, match="use that spelling"):
            g.to_cells({"x": np.ones((1, 1))},
                       grid="grid-.25deg-center-[-180,180)")


class TestMissing:
    def test_nan_cells_left_out(self):
        g = _grid([0.125, 0.375], [0.125, 0.375])
        a = np.array([[1.0, np.nan], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [np.nan, 4.0]])
        out = g.to_cells({"a": a, "b": b})
        assert out[["cell_ix", "cell_iy"]].values.tolist() == [[720, 360], [721, 361]]
        assert out.index.tolist() == [0, 1]


def test_feeds_apply_weights_on_the_right_regions():
    lat = np.array([0.125, 0.375])
    lon = np.array([10.125, 10.375])
    g = _grid(lat, lon)
    field = np.array([[1.0, 2.0], [3.0, 4.0]])
    table = pd.DataFrame({
        "region": ["south", "south", "north", "north"],
        "lon": [10.125, 10.375, 10.125, 10.375],
        "lat": [0.125, 0.125, 0.375, 0.375],
        "share": [0.5, 0.5, 0.25, 0.75],
    })
    weights = cilreg.WeightsArtifact.from_table(
        table, id_fields=["region"], weights={"w": "share"}, source=BCSD_GRID)
    out = cilreg.apply_weights(
        weights, g.to_cells({"value": field}), kind="intensive", weight="w",
        data_version=BCSD_GRID,
    ).frame.set_index("region")["value"]
    assert out["south"] == pytest.approx(1.5)
    assert out["north"] == pytest.approx(0.25 * 3.0 + 0.75 * 4.0)
