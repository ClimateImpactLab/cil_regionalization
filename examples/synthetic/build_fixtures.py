"""Regenerate the synthetic fixtures used by examples and integration tests.

The committed files under ``tests/data/synthetic/`` (``regions.parquet`` and
``raster.tif``) are produced by this script. Run it from the repo root:

    python examples/synthetic/build_fixtures.py

You only need to run this if you change the fixture shape; otherwise the
committed binaries are the source of truth.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Patch a stale conda PROJ_LIB before geopandas pulls in pyproj.
_proj_dir = Path(sys.prefix) / "share" / "proj"
if (_proj_dir / "proj.db").exists():
    os.environ.setdefault("PROJ_DATA", str(_proj_dir))
    os.environ["PROJ_LIB"] = str(_proj_dir)

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)


def build_regions() -> Path:
    a = box(0.0, 0.0, 2.0, 2.0)
    b = box(0.1, 0.1, 0.3, 0.3)
    outer = box(2.0, 2.0, 4.0, 4.0)
    hole = box(2.4, 2.4, 3.6, 3.6)
    c = outer.difference(hole)
    gdf = gpd.GeoDataFrame(
        {
            "region_id": ["A", "B", "C"],
            "label": ["alpha", "beta", "gamma"],
            "geometry": [a, b, c],
        },
        crs="EPSG:4326",
    )
    p = OUT / "regions.parquet"
    gdf.to_parquet(p)
    print(f"wrote {p}")
    return p


def build_raster() -> Path:
    width, height, res = 16, 16, 0.5
    transform = from_origin(0.0, 8.0, res, res)
    data = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        lat = 8.0 - (row + 0.5) * res
        for col in range(width):
            lon = (col + 0.5) * res
            if 0.0 <= lon <= 2.0 and 0.0 <= lat <= 2.0:
                data[row, col] = 10.0
    p = OUT / "raster.tif"
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
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    build_regions()
    build_raster()
