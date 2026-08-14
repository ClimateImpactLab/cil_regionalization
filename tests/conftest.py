"""Shared fixtures: synthetic regions and rasters used across unit tests.

The fixtures here are intentionally tiny and deterministic so the suite
runs in well under a second and so that expected weight values can be
reasoned about by hand.

`S51_TEST_HIERIDS` is the verified five-hierid test universe from the
world-combo-2017 impact region set, chosen to cover its three naming
shapes: a bare ISO3 country (ABW, MAF), an R-suffixed clustering
remainder (AND, BMU), and a dotted subdivision (BHR.5). The list is a
measured fact about that version, inlined here so the test suite is
self contained.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

S51_TEST_HIERIDS: tuple[str, ...] = (
    "ABW",
    "AND.Ra5cc0db7a54d1bb3",
    "BHR.5",
    "BMU.R676c07148ce9acd1",
    "MAF",
)

# Conda envs sometimes inherit PROJ_LIB from the base env pointing at the
# wrong proj.db; correct that before geopandas imports pyproj.
_proj_dir = Path(sys.prefix) / "share" / "proj"
if (_proj_dir / "proj.db").exists():
    os.environ["PROJ_DATA"] = str(_proj_dir)
    os.environ["PROJ_LIB"] = str(_proj_dir)

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box


@pytest.fixture
def three_region_gdf() -> gpd.GeoDataFrame:
    """Three regions sharing the lon/lat plane near (0,0):

    A : ordinary 2x2 box [0, 0]-[2, 2]
    B : tiny region [0.1, 0.1]-[0.3, 0.3] (smaller than a 1deg cell)
    C : 2x2 box with a hole in the middle [2, 2]-[4, 4] minus [2.4, 2.4]-[3.6, 3.6]
    """
    a = box(0.0, 0.0, 2.0, 2.0)
    b = box(0.1, 0.1, 0.3, 0.3)
    outer = box(2.0, 2.0, 4.0, 4.0)
    hole = box(2.4, 2.4, 3.6, 3.6)
    c = outer.difference(hole)
    return gpd.GeoDataFrame(
        {
            "region_id": ["A", "B", "C"],
            "label": ["alpha", "beta", "gamma"],
            "geometry": [a, b, c],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def three_region_geoparquet(
    tmp_path: Path, three_region_gdf: gpd.GeoDataFrame
) -> Path:
    p = tmp_path / "regions.parquet"
    three_region_gdf.to_parquet(p)
    return p


@pytest.fixture
def multi_cell_boundary_gdf() -> gpd.GeoDataFrame:
    """Single region D = box(0.7, 0.7, 2.3, 2.3); overlaps a 3x3 block of
    1deg cells with partial coverage on the edges. Designed to expose
    the exact_fraction vs pixel_centroid disagreement on **normalized**
    weights at boundary cells (not just raw totals as B does in notebook 03).
    """
    from shapely.geometry import box as _box

    return gpd.GeoDataFrame(
        {"region_id": ["D"], "geometry": [_box(0.7, 0.7, 2.3, 2.3)]},
        crs="EPSG:4326",
    )


@pytest.fixture
def multi_cell_boundary_geoparquet(
    tmp_path: Path, multi_cell_boundary_gdf: gpd.GeoDataFrame
) -> Path:
    p = tmp_path / "region_d.parquet"
    multi_cell_boundary_gdf.to_parquet(p)
    return p


@pytest.fixture
def dateline_crossers_gdf() -> gpd.GeoDataFrame:
    """Two synthetic regions whose bboxes span >180deg longitude in the
    [-180, 180) representation. `FJI_like` is a true two-component
    multipolygon at the antimeridian; `WIDE_like` is a single-piece
    crescent at the polar belt; both must produce cells at both ix
    extremes with no duplicate keys.
    """
    from shapely.geometry import box as _box, MultiPolygon

    fji_west = _box(-179.8, -18.0, -178.0, -16.0)
    fji_east = _box(178.0, -18.0, 179.8, -16.0)
    fji_like = MultiPolygon([fji_west, fji_east])

    wide_west = _box(-179.5, 75.0, -100.0, 78.0)
    wide_east = _box(100.0, 75.0, 179.5, 78.0)
    wide_like = MultiPolygon([wide_west, wide_east])

    return gpd.GeoDataFrame(
        {
            "region_id": ["FJI_like", "WIDE_like"],
            "geometry": [fji_like, wide_like],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def dateline_crossers_geoparquet(
    tmp_path: Path, dateline_crossers_gdf: gpd.GeoDataFrame
) -> Path:
    p = tmp_path / "dateline.parquet"
    dateline_crossers_gdf.to_parquet(p)
    return p


@pytest.fixture
def synthetic_raster(tmp_path: Path) -> Path:
    """Small GeoTIFF over [0, 8] x [0, 8]: constant value 10 inside region A,
    0 elsewhere. Lets us reason about per-region totals analytically.
    """
    width, height = 16, 16  # 0.5deg pixels over an 8x8 box
    res = 0.5
    transform = from_origin(0.0, 8.0, res, res)
    data = np.zeros((height, width), dtype=np.float32)
    # raster rows go from top (lat 8) down to bottom (lat 0).
    # value 10 in [0, 2] x [0, 2]
    for row in range(height):
        lat = 8.0 - (row + 0.5) * res
        for col in range(width):
            lon = (col + 0.5) * res
            if 0.0 <= lon <= 2.0 and 0.0 <= lat <= 2.0:
                data[row, col] = 10.0

    p = tmp_path / "synthetic.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as ds:
        ds.write(data, 1)
    return p


@pytest.fixture
def synthetic_fraction_raster(tmp_path: Path) -> Path:
    """Small CROPLAND-FRACTION GeoTIFF over [0, 8] x [0, 8]: fraction 0.5
    inside region A's footprint, 0 elsewhere. Lets a crop weight
    (area_weighted_sum) be checked against an analytic
    sum(fraction * pixel_geodesic_area).
    """
    width, height = 16, 16
    res = 0.5
    transform = from_origin(0.0, 8.0, res, res)
    data = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        lat = 8.0 - (row + 0.5) * res
        for col in range(width):
            lon = (col + 0.5) * res
            if 0.0 <= lon <= 2.0 and 0.0 <= lat <= 2.0:
                data[row, col] = 0.5

    p = tmp_path / "synthetic_fraction.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as ds:
        ds.write(data, 1)
    return p


def _extra_missing(module_name: str) -> bool:
    """True when the optional module is genuinely absent. Robust to test
    pollution: a test may plant a stub in sys.modules, and find_spec
    raises on modules without __spec__."""
    import importlib.util
    import sys

    if sys.modules.get(module_name) is not None:
        return False
    try:
        return importlib.util.find_spec(module_name) is None
    except (ImportError, ValueError):
        return True


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """State missing optional extras in plain words. A pass count that
    silently excludes a skipped module reads as full coverage; this
    makes the gap explicit in every summary."""
    gaps = []
    if _extra_missing("google.cloud.bigquery"):
        gaps.append(
            "the [bigquery] extra is not installed: every test in "
            "tests/integration/test_bigquery_backend.py was skipped"
        )
    if _extra_missing("xarray"):
        gaps.append(
            "the [netcdf] extra is not installed: the NetCDF leaf and "
            "pipeline tests were skipped"
        )
    for gap in gaps:
        terminalreporter.write_line(
            f"COVERAGE GAP: {gap}; this pass count does not cover that code.",
            yellow=True,
        )
