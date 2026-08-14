"""Region loading: GeoParquet path, filters, invalid-geometry policy, BQ ref."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from segment_weights.config import RegionsConfig
from segment_weights.regions import RegionSet


class TestLocalLoad:
    def test_load_geoparquet(self, three_region_geoparquet: Path):
        cfg = RegionsConfig(
            path=str(three_region_geoparquet), id_fields=["region_id"]
        )
        rs = RegionSet.from_config(cfg)
        assert rs.is_local
        assert len(rs) == 3
        ids = [d["region_id"] for d, _ in rs.iter_regions()]
        assert sorted(ids) == ["A", "B", "C"]

    def test_missing_path_raises(self, tmp_path: Path):
        cfg = RegionsConfig(path=str(tmp_path / "nope.parquet"), id_fields=["x"])
        with pytest.raises(FileNotFoundError, match="regions.path"):
            RegionSet.from_config(cfg)

    def test_missing_id_field_raises(self, three_region_geoparquet: Path):
        cfg = RegionsConfig(
            path=str(three_region_geoparquet), id_fields=["missing_col"]
        )
        with pytest.raises(ValueError, match="missing_col"):
            RegionSet.from_config(cfg)


class TestFilters:
    def test_keep_filter(self, three_region_geoparquet: Path):
        cfg = RegionsConfig(
            path=str(three_region_geoparquet),
            id_fields=["region_id"],
            keep={"label": ["alpha", "gamma"]},
        )
        rs = RegionSet.from_config(cfg)
        ids = sorted(d["region_id"] for d, _ in rs.iter_regions())
        assert ids == ["A", "C"]

    def test_drop_filter(self, three_region_geoparquet: Path):
        cfg = RegionsConfig(
            path=str(three_region_geoparquet),
            id_fields=["region_id"],
            drop={"region_id": ["B"]},
        )
        rs = RegionSet.from_config(cfg)
        ids = sorted(d["region_id"] for d, _ in rs.iter_regions())
        assert ids == ["A", "C"]

    def test_filter_unknown_column_raises(self, three_region_geoparquet: Path):
        cfg = RegionsConfig(
            path=str(three_region_geoparquet),
            id_fields=["region_id"],
            keep={"no_such_col": ["A"]},
        )
        with pytest.raises(ValueError, match="no_such_col"):
            RegionSet.from_config(cfg)


def _invalid_gdf(tmp_path: Path) -> Path:
    """Bow-tie polygon: self-intersecting → invalid."""
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    valid = box(2, 2, 3, 3)
    gdf = gpd.GeoDataFrame(
        {"region_id": ["bad", "good"], "geometry": [bowtie, valid]},
        crs="EPSG:4326",
    )
    p = tmp_path / "invalid.parquet"
    gdf.to_parquet(p)
    return p


class TestInvalidGeometry:
    def test_repair_policy_fixes_and_logs(self, tmp_path: Path, caplog):
        path = _invalid_gdf(tmp_path)
        cfg = RegionsConfig(
            path=str(path),
            id_fields=["region_id"],
            on_invalid_geometry="repair",
        )
        with caplog.at_level(logging.INFO, logger="segment_weights.regions"):
            rs = RegionSet.from_config(cfg)
        assert len(rs) == 2  # both kept
        assert rs.gdf.geometry.is_valid.all()
        assert any("repairing" in r.message and "bad" in r.message for r in caplog.records)

    def test_skip_policy_drops_and_logs(self, tmp_path: Path, caplog):
        path = _invalid_gdf(tmp_path)
        cfg = RegionsConfig(
            path=str(path),
            id_fields=["region_id"],
            on_invalid_geometry="skip",
        )
        with caplog.at_level(logging.INFO, logger="segment_weights.regions"):
            rs = RegionSet.from_config(cfg)
        assert len(rs) == 1
        kept = next(rs.iter_regions())[0]["region_id"]
        assert kept == "good"
        assert any("skipping" in r.message and "bad" in r.message for r in caplog.records)

    def test_error_policy_raises(self, tmp_path: Path):
        path = _invalid_gdf(tmp_path)
        cfg = RegionsConfig(
            path=str(path),
            id_fields=["region_id"],
            on_invalid_geometry="error",
        )
        with pytest.raises(ValueError) as exc:
            RegionSet.from_config(cfg)
        assert "bad" in str(exc.value)
        assert "invalid geometries" in str(exc.value)


class TestBigQueryReference:
    def test_bq_table_holds_reference_only(self):
        cfg = RegionsConfig(
            table="compute-impactlab.foo.regions",
            id_fields=["hierid"],
        )
        rs = RegionSet.from_config(cfg)
        assert not rs.is_local
        assert rs.table == "compute-impactlab.foo.regions"

    def test_bq_iter_regions_refuses(self):
        cfg = RegionsConfig(table="ci.foo.bar", id_fields=["hierid"])
        rs = RegionSet.from_config(cfg)
        with pytest.raises(ValueError, match="BigQuery"):
            next(rs.iter_regions())

    def test_bq_len_refuses(self):
        cfg = RegionsConfig(table="ci.foo.bar", id_fields=["hierid"])
        rs = RegionSet.from_config(cfg)
        with pytest.raises(ValueError, match="BigQuery"):
            len(rs)
