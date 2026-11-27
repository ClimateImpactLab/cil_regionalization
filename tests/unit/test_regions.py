"""Region loading: GeoParquet path, filters, invalid-geometry policy, BQ ref."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from cil_regionalization.config import RegionsConfig
from cil_regionalization.regions import RegionSet


class TestLocalLoad:
    def test_load_geoparquet(self, three_region_geoparquet: Path):
        cfg = RegionsConfig(path=str(three_region_geoparquet), id_fields=["region_id"])
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

    def test_unknown_keep_and_drop_values_reported(self, three_region_geoparquet: Path):
        cfg = RegionsConfig(
            path=str(three_region_geoparquet),
            id_fields=["region_id"],
            keep={"label": ["alpha", "gamma", "no_such_label"]},
            drop={"region_id": ["C", "Z"]},
        )
        rs = RegionSet.from_config(cfg)
        assert [d["region_id"] for d, _ in rs.iter_regions()] == ["A"]
        assert rs.unknown_keep_ids == {"label": ["no_such_label"]}
        assert rs.unknown_drop_ids == {"region_id": ["Z"]}

    def test_matching_values_report_nothing(self, three_region_geoparquet: Path):
        # B is in the file, so dropping it is not unknown even though
        # keep already removed it.
        cfg = RegionsConfig(
            path=str(three_region_geoparquet),
            id_fields=["region_id"],
            keep={"region_id": ["A", "C"]},
            drop={"region_id": ["B"]},
        )
        rs = RegionSet.from_config(cfg)
        assert rs.unknown_keep_ids == {}
        assert rs.unknown_drop_ids == {}


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
        with caplog.at_level(logging.INFO, logger="cil_regionalization.regions"):
            rs = RegionSet.from_config(cfg)
        assert len(rs) == 2  # both kept
        assert rs.gdf.geometry.is_valid.all()
        assert rs.repaired_ids == [{"region_id": "bad"}]
        assert rs.skipped_invalid_ids == []
        assert any(
            "repairing" in r.message and "bad" in r.message for r in caplog.records
        )

    def test_skip_policy_drops_and_logs(self, tmp_path: Path, caplog):
        path = _invalid_gdf(tmp_path)
        cfg = RegionsConfig(
            path=str(path),
            id_fields=["region_id"],
            on_invalid_geometry="skip",
        )
        with caplog.at_level(logging.INFO, logger="cil_regionalization.regions"):
            rs = RegionSet.from_config(cfg)
        assert len(rs) == 1
        kept = next(rs.iter_regions())[0]["region_id"]
        assert kept == "good"
        # left out, not repaired. It went red on the record that listed
        # it as repaired.
        assert rs.repaired_ids == []
        assert rs.skipped_invalid_ids == [{"region_id": "bad"}]
        assert any(
            "skipping" in r.message and "bad" in r.message for r in caplog.records
        )

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


def _fake_gcsfs(monkeypatch, root: Path) -> None:
    """A stand-in for gcsfs that serves gs://<path> from ``root``."""
    import shutil
    import sys
    import types

    class GCSFileSystem:
        def _local(self, uri: str) -> Path:
            return root / uri.removeprefix("gs://")

        def cat(self, uri: str) -> bytes:
            return self._local(uri).read_bytes()

        def get(self, uri: str, dest: str) -> None:
            local = self._local(uri)
            if not local.exists():
                raise FileNotFoundError(uri)
            shutil.copy(local, dest)

    module = types.ModuleType("gcsfs")
    module.GCSFileSystem = GCSFileSystem
    monkeypatch.setitem(sys.modules, "gcsfs", module)


class TestRemoteRegions:
    """A gs:// regions path is read, and hashed from the bytes read."""

    def test_a_gs_path_passes_the_input_check(self):
        """It is read when the run reads it. It went red on the check
        that looked for it on the local disk."""
        from cil_regionalization.config import Config
        from cil_regionalization.generate import check_inputs_exist

        cfg = Config.model_validate(
            {
                "project": {"name": "test"},
                "regions": {
                    "path": "gs://bucket/regions.parquet",
                    "id_fields": ["region_id"],
                },
                "grid": {"mode": "generate", "resolution": 1.0},
                "weights": [{"name": "area"}],
                "backend": {"kind": "local"},
                "output": {"dir": "out"},
            }
        )
        assert check_inputs_exist(cfg) == []

    @pytest.mark.parametrize("name", ["regions.parquet", "regions.shp", "regions.gpkg"])
    def test_a_gs_file_is_read_and_hashed(self, tmp_path, monkeypatch, name):
        """The hash is of the main file as fetched. It went red with no
        hash kept."""
        import hashlib

        gdf = gpd.GeoDataFrame(
            {"region_id": ["a", "b"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
            crs="EPSG:4326",
        )
        bucket = tmp_path / "bucket"
        bucket.mkdir()
        if name.endswith(".parquet"):
            gdf.to_parquet(bucket / name)
        else:
            gdf.to_file(bucket / name)
        _fake_gcsfs(monkeypatch, tmp_path)
        rs = RegionSet.from_config(
            RegionsConfig(path=f"gs://bucket/{name}", id_fields=["region_id"])
        )
        assert sorted(rs.gdf["region_id"]) == ["a", "b"]
        assert rs.source_uri == f"gs://bucket/{name}"
        assert (
            rs.source_sha256 == hashlib.sha256((bucket / name).read_bytes()).hexdigest()
        )

    def test_a_local_path_is_hashed_where_the_manifest_is_written(
        self, three_region_geoparquet
    ):
        rs = RegionSet.from_config(
            RegionsConfig(path=str(three_region_geoparquet), id_fields=["region_id"])
        )
        assert rs.source_sha256 is None

    def test_without_gcsfs_the_extra_that_installs_it_is_named(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "gcsfs", None)
        with pytest.raises(ImportError, match=r"which the \[bigquery\] extra installs"):
            RegionSet.from_config(
                RegionsConfig(path="gs://bucket/regions.parquet", id_fields=["id"])
            )
