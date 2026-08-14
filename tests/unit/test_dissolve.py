"""Dissolve step: key uniqueness, area conservation, and the loud paths."""
from __future__ import annotations

import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from cil_regionalization.dissolve import LevelSpec, dissolve_layer, prepare_target_layers


def _pieces() -> gpd.GeoDataFrame:
    """Six non-overlapping pieces in two countries.

    C1 has two level-1 units (a: two pieces, b: one); C2 has one unit of
    three pieces. Total planar area is 6 unit squares by construction.
    """
    return gpd.GeoDataFrame(
        {
            "iso": ["C1", "C1", "C1", "C2", "C2", "C2"],
            "id1": [1, 1, 2, 1, 1, 1],
            "name1": ["Alpha", "Alpha", "Beta", "Gamma", "Gamma", "Gamma"],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
                box(2, 0, 3, 1),
                box(0, 2, 1, 3),
                box(1, 2, 2, 3),
                box(2, 2, 3, 3),
            ],
        },
        crs="EPSG:4326",
    )


class TestDissolveLayer:
    def test_units_and_unique_keys(self):
        out, report = dissolve_layer(
            _pieces(), ["iso", "id1"], name_columns=["name1"]
        )
        assert len(out) == 3
        assert not out.duplicated(subset=["iso", "id1"]).any()
        assert report.n_source_features == 6
        assert report.n_units == 3
        named = out.set_index(["iso", "id1"])["name1"]
        assert named.loc[("C1", 1)] == "Alpha"
        assert named.loc[("C2", 1)] == "Gamma"

    def test_area_conserved_within_tolerance(self):
        out, report = dissolve_layer(_pieces(), ["iso", "id1"])
        assert report.source_area == pytest.approx(6.0)
        assert report.dissolved_area == pytest.approx(6.0)
        assert report.area_rel_diff <= report.tolerance

    def test_overlapping_pieces_fail_conservation(self):
        # Two pieces of the same unit overlap by half a square; union
        # keeps 1.5 units where the pieces sum to 2.0. The check must
        # refuse rather than silently absorb the loss.
        gdf = gpd.GeoDataFrame(
            {
                "iso": ["C1", "C1"],
                "id1": [1, 1],
                "geometry": [box(0, 0, 1, 1), box(0.5, 0, 1.5, 1)],
            },
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="not conserved"):
            dissolve_layer(gdf, ["iso", "id1"])

    def test_null_key_raises_naming_offenders(self):
        gdf = _pieces()
        gdf.loc[0, "id1"] = None
        with pytest.raises(ValueError, match="null key"):
            dissolve_layer(gdf, ["iso", "id1"])

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            dissolve_layer(_pieces(), ["iso", "adm_code"])

    def test_invalid_geometry_repaired_and_counted(self):
        from shapely.geometry import Polygon

        gdf = _pieces()
        bowtie = Polygon([(4, 0), (5, 1), (5, 0), (4, 1)])
        gdf.loc[len(gdf)] = {
            "iso": "C3",
            "id1": 1,
            "name1": "Bow",
            "geometry": bowtie,
        }
        out, report = dissolve_layer(gdf, ["iso", "id1"])
        assert report.n_repaired_geometries == 1
        assert len(out) == 4


class TestPrepareTargetLayers:
    def test_levels_written_with_manifest(self, tmp_path):
        source = tmp_path / "combined.parquet"
        _pieces().to_parquet(source)
        manifest = prepare_target_layers(
            source,
            [
                LevelSpec("adm0", ("iso",), ("name1",)),
                LevelSpec("adm1", ("iso", "id1"), ("name1",)),
            ],
            tmp_path / "targets",
            version="synthetic-combined-v1",
        )
        adm0 = gpd.read_parquet(tmp_path / "targets" / "adm0.parquet")
        adm1 = gpd.read_parquet(tmp_path / "targets" / "adm1.parquet")
        assert len(adm0) == 2
        assert len(adm1) == 3

        on_disk = json.loads(
            (tmp_path / "targets" / "targets.manifest.json").read_text()
        )
        assert on_disk["version"] == "synthetic-combined-v1"
        assert on_disk["n_source_features"] == 6
        assert on_disk["levels"]["adm0"]["n_units"] == 2
        assert on_disk["levels"]["adm1"]["n_units"] == 3
        assert len(on_disk["source_sha256"]) == 64
        assert on_disk["levels"]["adm1"]["area_rel_diff"] <= 1e-6
        assert manifest["levels"]["adm1"]["n_units"] == 3

    def test_source_without_crs_rejected(self, tmp_path):
        source = tmp_path / "no_crs.parquet"
        bare = gpd.GeoDataFrame(
            _pieces().drop(columns="geometry"),
            geometry=list(_pieces().geometry),
        )
        bare.to_parquet(source)
        with pytest.raises(ValueError, match="no CRS"):
            prepare_target_layers(
                source,
                [LevelSpec("adm0", ("iso",))],
                tmp_path / "targets",
                version="v1",
            )


class TestDropQuery:
    def test_drop_recorded_in_manifest(self, tmp_path):
        # An overlapping duplicate piece breaks conservation; a declared
        # exclusion removes it, and the manifest records the decision.
        import pandas as pd

        scrap = gpd.GeoDataFrame(
            {
                "iso": ["C1"],
                "id1": [9],
                "name1": ["Scrap"],
                "geometry": [box(0.2, 0.2, 0.8, 0.8)],  # inside (C1, 1)
            },
            crs="EPSG:4326",
        )
        gdf = pd.concat([_pieces(), scrap], ignore_index=True)
        source = tmp_path / "with_scrap.parquet"
        gdf.to_parquet(source)

        # The scrap is its own (C1, 9) unit at adm1, so adm1 conserves;
        # only the adm0 union collapses the duplicated area.
        levels = [
            LevelSpec("adm1", ("iso", "id1")),
            LevelSpec("adm0", ("iso",)),
        ]
        with pytest.raises(ValueError, match="not conserved"):
            prepare_target_layers(source, levels, tmp_path / "t1", version="v1")

        manifest = prepare_target_layers(
            source,
            levels,
            tmp_path / "t2",
            version="v1",
            drop_query="iso == 'C1' and id1 == 9",
        )
        assert manifest["n_dropped_features"] == 1
        assert manifest["drop_query"] == "iso == 'C1' and id1 == 9"
        assert manifest["levels"]["adm0"]["n_units"] == 2
        assert manifest["levels"]["adm1"]["n_units"] == 3

    def test_stale_drop_query_raises(self, tmp_path):
        source = tmp_path / "clean.parquet"
        _pieces().to_parquet(source)
        with pytest.raises(ValueError, match="matched no features"):
            prepare_target_layers(
                source,
                [LevelSpec("adm0", ("iso",))],
                tmp_path / "t",
                version="v1",
                drop_query="iso == 'ZZ'",
            )


class TestRepair:
    def test_repair_applied_and_recorded(self, tmp_path):
        source = tmp_path / "blank_key.parquet"
        frame = _pieces()
        # One feature lost its id1; the repair restores it from evidence.
        frame.loc[1, "id1"] = 0
        frame.to_parquet(source)

        def fix(gdf):
            gdf = gdf.copy()
            gdf.loc[gdf["id1"] == 0, "id1"] = 1
            return gdf

        record = {"name": "restore id1 on one feature", "rows": 1}
        manifest = prepare_target_layers(
            source,
            [LevelSpec("adm1", ("iso", "id1"), ("name1",))],
            tmp_path / "targets",
            version="v1",
            repair=fix,
            repair_record=record,
        )
        assert manifest["repair"] == record
        assert manifest["levels"]["adm1"]["n_units"] == 3
        on_disk = json.loads(
            (tmp_path / "targets" / "targets.manifest.json").read_text()
        )
        assert on_disk["repair"] == record

    def test_repair_without_record_refused(self, tmp_path):
        source = tmp_path / "clean.parquet"
        _pieces().to_parquet(source)
        with pytest.raises(ValueError, match="recorded account"):
            prepare_target_layers(
                source,
                [LevelSpec("adm0", ("iso",))],
                tmp_path / "t",
                version="v1",
                repair=lambda g: g,
            )
        with pytest.raises(ValueError, match="recorded account"):
            prepare_target_layers(
                source,
                [LevelSpec("adm0", ("iso",))],
                tmp_path / "t",
                version="v1",
                repair_record={"name": "phantom"},
            )

    def test_repair_may_not_change_feature_count(self, tmp_path):
        source = tmp_path / "clean.parquet"
        _pieces().to_parquet(source)
        with pytest.raises(ValueError, match="feature count"):
            prepare_target_layers(
                source,
                [LevelSpec("adm0", ("iso",))],
                tmp_path / "t",
                version="v1",
                repair=lambda g: g.head(3),
                repair_record={"name": "bad", "rows": 3},
            )
