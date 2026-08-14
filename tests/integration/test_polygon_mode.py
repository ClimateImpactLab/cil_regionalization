"""End-to-end tests of polygon source-unit mode on synthetic geometries.

The fixtures are deliberately not GADM shaped: single- and multi-column
target ids with neutral names, box geometries reasoned about by hand. The
source layer plays the impact-region role; the two target layers play the
admin-level role at two granularities.

Layout on the lon/lat plane (targets level 1 and 2, sources u1..u5):

    level 1:  T1 = [0, 4] x [0, 4]      T2 = [4, 8] x [0, 4]
    level 2:  T1 splits at lon 2 into (T1, a) and (T1, b);
              T2 splits at lon 6 into (T2, a) and (T2, b)
    u1 = [0.5, 1.5] x [0.5, 1.5]   nested in T1 / (T1, a)
    u2 = [2.5, 3.5] x [0.5, 1.5]   nested in T1 / (T1, b)
    u3 = [5.0, 7.5] x [0.5, 1.5]   nested in T2, straddles (T2, a)/(T2, b)
                                   with lon widths 1.0 and 1.5
    u4 = [3.5, 4.5] x [2.5, 3.5]   straddles T1/T2 at lon 4, half and half
    u5 = [10, 11] x [10, 11]       outside every target (zero overlap)

The committed synthetic pop raster holds value 10 inside [0, 2] x [0, 2],
so u1 has population and u2, u3, u4 fall back per the configured policy.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from cil_regionalization.backends.local import LocalBackend
from cil_regionalization.config import Config
from cil_regionalization.regions import RegionSet
from cil_regionalization.weights import from_config_list


_REPO = Path(__file__).resolve().parents[2]
_POP_RASTER = _REPO / "tests" / "data" / "synthetic" / "raster.tif"


@pytest.fixture
def targets_level1(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "target_id": ["T1", "T2"],
            "geometry": [box(0.0, 0.0, 4.0, 4.0), box(4.0, 0.0, 8.0, 4.0)],
        },
        crs="EPSG:4326",
    )
    p = tmp_path / "targets_l1.parquet"
    gdf.to_parquet(p)
    return p


@pytest.fixture
def targets_level2(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "l1": ["T1", "T1", "T2", "T2"],
            "l2": ["a", "b", "a", "b"],
            "geometry": [
                box(0.0, 0.0, 2.0, 4.0),
                box(2.0, 0.0, 4.0, 4.0),
                box(4.0, 0.0, 6.0, 4.0),
                box(6.0, 0.0, 8.0, 4.0),
            ],
        },
        crs="EPSG:4326",
    )
    p = tmp_path / "targets_l2.parquet"
    gdf.to_parquet(p)
    return p


@pytest.fixture
def source_units(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "unit_id": ["u1", "u2", "u3", "u4", "u5"],
            "geometry": [
                box(0.5, 0.5, 1.5, 1.5),
                box(2.5, 0.5, 3.5, 1.5),
                box(5.0, 0.5, 7.5, 1.5),
                box(3.5, 2.5, 4.5, 3.5),
                box(10.0, 10.0, 11.0, 11.0),
            ],
        },
        crs="EPSG:4326",
    )
    p = tmp_path / "units.parquet"
    gdf.to_parquet(p)
    return p


def _cfg(
    targets_path: Path,
    source_path: Path,
    *,
    target_id_fields: list[str],
    normalization: str = "per_source",
    on_zero_overlap: str = "skip",
    weights: list[dict] | None = None,
    coverage: str = "exact_fraction",
) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "polygon_test"},
            "regions": {
                "path": str(targets_path),
                "id_fields": target_id_fields,
                "version": "synthetic-targets-v1",
            },
            "source": {
                "path": str(source_path),
                "id_fields": ["unit_id"],
                "version": "synthetic-units-v1",
                "on_zero_overlap": on_zero_overlap,
            },
            "weights": weights
            or [
                {"name": "pop", "raster": str(_POP_RASTER), "fallback": "area"},
                {"name": "area"},
            ],
            "backend": {"kind": "local", "coverage": coverage},
            "output": {"dir": "unused"},
            "normalization": normalization,
        }
    )


def _run(cfg: Config):
    return LocalBackend().compute(
        RegionSet.from_config(cfg.regions), None, from_config_list(cfg.weights), cfg
    )


class TestPerSourceLevel1:
    def test_sums_to_one_per_source_unit(self, targets_level1, source_units):
        result = _run(_cfg(targets_level1, source_units, target_id_fields=["target_id"]))
        assert result.sum_report.ok, result.sum_report.summary()
        for name in ("popwt", "areawt"):
            sums = result.frame.groupby("unit_id")[name].sum()
            assert sums.to_numpy() == pytest.approx(1.0)

    def test_straddler_split_matches_geodesic_area_ratio(
        self, targets_level1, source_units
    ):
        # u4 spans T1/T2 at lon 4 in two equal-width, same-latitude
        # halves; the geodesic areas are equal, so the split is 0.5/0.5.
        result = _run(_cfg(targets_level1, source_units, target_id_fields=["target_id"]))
        u4 = result.frame.loc[result.frame["unit_id"] == "u4"]
        assert set(u4["target_id"]) == {"T1", "T2"}
        assert u4["areawt"].to_numpy() == pytest.approx([0.5, 0.5], rel=1e-9)

    def test_nested_units_carry_full_weight(self, targets_level1, source_units):
        result = _run(_cfg(targets_level1, source_units, target_id_fields=["target_id"]))
        for unit, target in (("u1", "T1"), ("u2", "T1"), ("u3", "T2")):
            rows = result.frame.loc[result.frame["unit_id"] == unit]
            assert list(rows["target_id"]) == [target]
            assert rows["areawt"].iloc[0] == pytest.approx(1.0)

    def test_fallback_applies_per_source_unit(self, targets_level1, source_units):
        # Only u1 overlaps the raster's populated footprint; the others
        # fall back to their per-source area weights.
        result = _run(_cfg(targets_level1, source_units, target_id_fields=["target_id"]))
        methods = result.frame.groupby("unit_id")["pop_method"].first()
        assert methods.loc["u1"] == "native"
        assert (methods.drop("u1") == "area_fallback").all()


class TestPerSourceLevel2:
    def test_multi_field_target_ids(self, targets_level2, source_units):
        result = _run(_cfg(targets_level2, source_units, target_id_fields=["l1", "l2"]))
        assert result.sum_report.ok, result.sum_report.summary()
        assert list(result.frame.columns[:3]) == ["l1", "l2", "unit_id"]

    def test_straddler_split_by_asymmetric_widths(self, targets_level2, source_units):
        # u3 = lon [5.0, 7.5] against the (T2, a)/(T2, b) split at lon 6:
        # widths 1.0 and 1.5 in the same latitude band, so 0.4/0.6.
        # Tolerance 1e-4: pyproj treats a box's constant-latitude edges as
        # geodesics between the corner vertices, which bow slightly off the
        # parallel by an amount that grows with edge length. Symmetric
        # splits cancel the effect exactly; asymmetric ones deviate at the
        # 1e-5 level. The same property is documented for the BigQuery
        # backend's cell densification.
        result = _run(_cfg(targets_level2, source_units, target_id_fields=["l1", "l2"]))
        u3 = result.frame.loc[result.frame["unit_id"] == "u3"].set_index("l2")
        assert u3.loc["a", "areawt"] == pytest.approx(0.4, rel=1e-4)
        assert u3.loc["b", "areawt"] == pytest.approx(0.6, rel=1e-4)


class TestZeroOverlapPolicy:
    def test_error_policy_names_the_unit(self, targets_level1, source_units):
        cfg = _cfg(
            targets_level1,
            source_units,
            target_id_fields=["target_id"],
            on_zero_overlap="error",
        )
        with pytest.raises(ValueError, match="u5"):
            _run(cfg)

    def test_skip_policy_records_in_manifest(self, targets_level1, source_units):
        result = _run(_cfg(targets_level1, source_units, target_id_fields=["target_id"]))
        assert "u5" not in set(result.frame["unit_id"])
        assert result.manifest.extra["zero_overlap_count"] == 1
        assert result.manifest.extra["zero_overlap_source_units"] == [["u5"]]


class TestAllWeightKinds:
    def _weights(self, fraction_raster: Path) -> list[dict]:
        return [
            {
                "name": "crop",
                "raster": str(fraction_raster),
                "kind": "area_weighted_sum",
                "fallback": "nan",
            },
            {"name": "pop", "raster": str(_POP_RASTER), "fallback": "area"},
            {"name": "area"},
        ]

    def test_crop_pop_area_together(
        self, targets_level1, source_units, synthetic_fraction_raster
    ):
        cfg = _cfg(
            targets_level1,
            source_units,
            target_id_fields=["target_id"],
            weights=self._weights(synthetic_fraction_raster),
        )
        result = _run(cfg)
        assert result.sum_report.ok, result.sum_report.summary()
        frame = result.frame
        # u1 sits in the fraction raster's footprint: native crop weight.
        u1 = frame.loc[frame["unit_id"] == "u1"]
        assert (u1["crop_method"] == "native").all()
        assert u1["cropwt"].sum() == pytest.approx(1.0)
        # Units outside the footprint carry NaN cropwt under fallback='nan'.
        rest = frame.loc[frame["unit_id"] != "u1"]
        assert (rest["crop_method"] == "nan").all()
        assert rest["cropwt"].isna().all()

    def test_pixel_centroid_coverage_works(
        self, targets_level1, source_units, synthetic_fraction_raster
    ):
        cfg = _cfg(
            targets_level1,
            source_units,
            target_id_fields=["target_id"],
            weights=self._weights(synthetic_fraction_raster),
            coverage="pixel_centroid",
        )
        result = _run(cfg)
        assert result.sum_report.ok, result.sum_report.summary()


class TestPerDestinationDirection:
    def test_weights_sum_per_target(self, targets_level1, source_units):
        cfg = _cfg(
            targets_level1,
            source_units,
            target_id_fields=["target_id"],
            normalization="per_destination",
        )
        result = _run(cfg)
        assert result.schema.normalization == "per_destination"
        assert result.sum_report.ok, result.sum_report.summary()
        sums = result.frame.groupby("target_id")["areawt"].sum()
        assert sums.to_numpy() == pytest.approx(1.0)


class TestDateline:
    """Both layers in canonical [-180, 180] form intersect correctly with
    no antimeridian split: the split exists for grid index arithmetic,
    which polygon mode does not have."""

    def _fixtures(self, tmp_path: Path) -> tuple[Path, Path]:
        targets = gpd.GeoDataFrame(
            {
                "target_id": ["W", "E"],
                "geometry": [
                    box(-180.0, -2.0, -179.2, 2.0),
                    box(179.2, -2.0, 180.0, 2.0),
                ],
            },
            crs="EPSG:4326",
        )
        source = gpd.GeoDataFrame(
            {
                "unit_id": ["d1"],
                "geometry": [
                    MultiPolygon(
                        [
                            box(-179.5, -1.0, -179.0, 1.0),
                            box(179.0, -1.0, 179.5, 1.0),
                        ]
                    )
                ],
            },
            crs="EPSG:4326",
        )
        tp = tmp_path / "dateline_targets.parquet"
        sp = tmp_path / "dateline_units.parquet"
        targets.to_parquet(tp)
        source.to_parquet(sp)
        return tp, sp

    def test_dateline_unit_splits_across_both_sides(self, tmp_path):
        tp, sp = self._fixtures(tmp_path)
        result = _run(_cfg(tp, sp, target_id_fields=["target_id"]))
        d1 = result.frame.loc[result.frame["unit_id"] == "d1"]
        assert set(d1["target_id"]) == {"W", "E"}
        # Overlaps are 0.3 degrees of the same latitude band on each side.
        assert d1["areawt"].to_numpy() == pytest.approx([0.5, 0.5], rel=1e-9)
        assert result.sum_report.ok

    def test_out_of_domain_geometry_rejected(self, tmp_path, targets_level1):
        # A source stored with extended longitude (lon > 180) is the
        # representation the grid path splits; polygon mode refuses it.
        bad = gpd.GeoDataFrame(
            {"unit_id": ["x1"], "geometry": [box(179.0, -1.0, 181.0, 1.0)]},
            crs="EPSG:4326",
        )
        sp = tmp_path / "bad_units.parquet"
        bad.to_parquet(sp)
        with pytest.raises(ValueError, match="outside lon"):
            _run(_cfg(targets_level1, sp, target_id_fields=["target_id"]))


class TestManifestAndProvenance:
    def test_manifest_records_mode_and_versions(self, targets_level1, source_units):
        result = _run(_cfg(targets_level1, source_units, target_id_fields=["target_id"]))
        m = result.manifest
        assert m.source_mode == "polygons"
        assert m.regions_version == "synthetic-targets-v1"
        assert m.source_version == "synthetic-units-v1"
        assert m.normalization == "per_source"
        assert m.grid_mode is None and m.grid_resolution is None
        assert len(m.inputs["geometry:regions"]) == 64
        assert len(m.inputs["geometry:source"]) == 64
        assert m.row_counts["source_units"] == 5
        assert m.row_counts["regions"] == 2

    def test_empty_target_recorded(self, tmp_path, source_units):
        targets = gpd.GeoDataFrame(
            {
                "target_id": ["T1", "T2", "T_EMPTY"],
                "geometry": [
                    box(0.0, 0.0, 4.0, 4.0),
                    box(4.0, 0.0, 8.0, 4.0),
                    box(20.0, 20.0, 24.0, 24.0),
                ],
            },
            crs="EPSG:4326",
        )
        tp = tmp_path / "targets_with_empty.parquet"
        targets.to_parquet(tp)
        result = _run(_cfg(tp, source_units, target_id_fields=["target_id"]))
        assert result.manifest.extra["empty_region_count"] == 1
        assert result.manifest.extra["empty_regions"] == [["T_EMPTY"]]


class TestConfigModes:
    def _base(self, targets_level1, source_units) -> dict:
        return {
            "project": {"name": "cfg_test"},
            "regions": {
                "path": str(targets_level1),
                "id_fields": ["target_id"],
                "version": "v1",
            },
            "source": {
                "path": str(source_units),
                "id_fields": ["unit_id"],
                "version": "v1",
            },
            "weights": [{"name": "area"}],
            "backend": {"kind": "local"},
            "output": {"dir": "unused"},
        }

    def test_grid_and_source_mutually_exclusive(self, targets_level1, source_units):
        data = self._base(targets_level1, source_units)
        data["grid"] = {"mode": "generate", "resolution": 1.0}
        with pytest.raises(ValueError, match="mutually exclusive"):
            Config.model_validate(data)

    def test_neither_grid_nor_source_rejected(self, targets_level1, source_units):
        data = self._base(targets_level1, source_units)
        del data["source"]
        with pytest.raises(ValueError, match=r"\[grid\] or \[source\]"):
            Config.model_validate(data)

    def test_bigquery_backend_deferred(self, targets_level1, source_units):
        data = self._base(targets_level1, source_units)
        data["backend"] = {"kind": "bigquery", "coverage": "pixel_centroid"}
        with pytest.raises(ValueError, match="deferred"):
            Config.model_validate(data)

    def test_missing_regions_version_rejected(self, targets_level1, source_units):
        data = self._base(targets_level1, source_units)
        del data["regions"]["version"]
        with pytest.raises(ValueError, match="regions.version"):
            Config.model_validate(data)


class TestCliPolygonMode:
    def test_run_end_to_end_from_toml(
        self, targets_level1, source_units, tmp_path, capsys
    ):
        from cil_regionalization.cli import main

        out_dir = tmp_path / "out"
        cfg_path = tmp_path / "polygon.toml"
        cfg_path.write_text(
            f"""
normalization = "per_source"

[project]
name = "polygon_cli_test"

[regions]
path = "{targets_level1}"
id_fields = ["target_id"]
version = "synthetic-targets-v1"

[source]
path = "{source_units}"
id_fields = ["unit_id"]
version = "synthetic-units-v1"
on_zero_overlap = "skip"

[[weights]]
name = "pop"
raster = "{_POP_RASTER}"
fallback = "area"

[[weights]]
name = "area"

[backend]
kind = "local"

[output]
dir = "{out_dir}"
"""
        )
        rc = main(["validate", str(cfg_path)])
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert "source=polygons/synthetic-units-v1" in captured.out

        rc = main(["run", str(cfg_path)])
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert "sum_to_one=ok" in captured.out
        manifest = json.loads((out_dir / "weights.manifest.json").read_text())
        assert manifest["source_mode"] == "polygons"
        assert manifest["normalization"] == "per_source"
        frame = pd.read_parquet(out_dir / "weights.parquet")
        assert set(frame.columns[:2]) == {"target_id", "unit_id"}


class TestDomainEdgeTolerance:
    """The domain guard admits sub-meter digitization noise and records
    it; a layer in [0, 360) longitude form still fails by orders of
    magnitude. Pins the exact overshoot observed on world-combo-201710's
    Antarctica (longitude -180.000015, 1.5e-5 degrees past the edge) so
    a future tightening of the guard fails here instead of surprising a
    production run."""

    def test_ata_scale_overshoot_tolerated_and_recorded(
        self, targets_level1, tmp_path
    ):
        units = gpd.GeoDataFrame(
            {
                "unit_id": ["ata_like", "u_clean"],
                "geometry": [
                    box(-180.000015, -1.0, -179.5, 1.0),
                    box(1.0, 1.0, 2.0, 2.0),
                ],
            },
            crs="EPSG:4326",
        )
        # A target reaching the western edge so ata_like overlaps something.
        targets = gpd.GeoDataFrame(
            {
                "target_id": ["W", "T1"],
                "geometry": [box(-180.0, -2.0, -179.0, 2.0), box(0.0, 0.0, 4.0, 4.0)],
            },
            crs="EPSG:4326",
        )
        sp = tmp_path / "near_edge_units.parquet"
        tp = tmp_path / "near_edge_targets.parquet"
        units.to_parquet(sp)
        targets.to_parquet(tp)
        result = _run(_cfg(tp, sp, target_id_fields=["target_id"]))
        assert result.sum_report.ok, result.sum_report.summary()
        near = result.manifest.extra["near_domain_edge_source_units"]
        assert len(near) == 1
        assert near[0]["unit_id"] == "ata_like"
        assert near[0]["overshoot_degrees"] == pytest.approx(1.5e-5, rel=1e-3)

    def test_zero_360_form_still_rejected(self, targets_level1, tmp_path):
        bad = gpd.GeoDataFrame(
            {"unit_id": ["x1"], "geometry": [box(200.0, -1.0, 210.0, 1.0)]},
            crs="EPSG:4326",
        )
        sp = tmp_path / "wrong_form_units.parquet"
        bad.to_parquet(sp)
        with pytest.raises(ValueError, match="outside lon"):
            _run(_cfg(targets_level1, sp, target_id_fields=["target_id"]))


class TestMixedGeometryNormalization:
    """Real admin-derived layers put shared borders into the intersection
    as line parts, making segments GeometryCollections; zonal operations
    need homogeneous MultiPolygons. Discarding the line parts is safe:
    zero area, zero raster pixels."""

    def test_helper_extracts_polygonal_parts(self):
        import numpy as np
        import shapely
        from shapely.geometry import (
            GeometryCollection,
            LineString,
            Point,
            Polygon,
        )

        from cil_regionalization.backends.local import _as_multipolygons

        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        mixed = GeometryCollection(
            [poly, LineString([(1, 0), (1, 1)]), Point(2, 2)]
        )
        out = _as_multipolygons(np.array([poly, mixed], dtype=object))
        assert all(g.geom_type == "MultiPolygon" for g in out)
        assert out[1].area == pytest.approx(poly.area)

    def test_collection_producing_intersection_runs_end_to_end(self, tmp_path):
        # B's second part touches A only along the x=2 edge, so A-and-B
        # intersect in a polygon plus a shared LineString: exactly the
        # GeometryCollection case observed on real borders.
        targets = gpd.GeoDataFrame(
            {"target_id": ["A"], "geometry": [box(0.0, 0.0, 2.0, 2.0)]},
            crs="EPSG:4326",
        )
        source = gpd.GeoDataFrame(
            {
                "unit_id": ["b1"],
                "geometry": [
                    MultiPolygon([box(1.0, 1.0, 3.0, 3.0), box(2.0, 0.0, 4.0, 1.0)])
                ],
            },
            crs="EPSG:4326",
        )
        tp, sp = tmp_path / "targets.parquet", tmp_path / "sources.parquet"
        targets.to_parquet(tp)
        source.to_parquet(sp)
        result = _run(_cfg(tp, sp, target_id_fields=["target_id"]))
        assert result.sum_report.ok, result.sum_report.summary()
        assert set(result.frame["unit_id"]) == {"b1"}


class TestPartialCoverageMarking:
    def test_subset_targets_mark_partially_covered_units(
        self, targets_level1, source_units, tmp_path
    ):
        # Keep only T1: u4 (which straddles T1/T2 half and half) is now
        # covered at ~0.5 of its own area and must be recorded.
        cfg = _cfg(targets_level1, source_units, target_id_fields=["target_id"])
        cfg = cfg.model_copy(
            update={
                "regions": cfg.regions.model_copy(
                    update={"keep": {"target_id": ["T1"]}}
                )
            }
        )
        result = LocalBackend().compute(
            RegionSet.from_config(cfg.regions),
            None,
            from_config_list(cfg.weights),
            cfg,
        )
        pc = result.manifest.partial_coverage
        assert pc["target_subset"] is True
        flagged = {u["unit_id"]: u["coverage_ratio"] for u in pc["units"]}
        assert "u4" in flagged
        # rel 1e-4: piece and full-unit areas are computed on different
        # vertex sets, so the geodesic-edge bowing documented for the
        # asymmetric straddler test appears here at the 1e-5 level too.
        assert flagged["u4"] == pytest.approx(0.5, rel=1e-4)
        assert pc["count"] >= 1

    def test_full_coverage_run_carries_no_marking(
        self, targets_level1, source_units
    ):
        result = _run(
            _cfg(targets_level1, source_units, target_id_fields=["target_id"])
        )
        pc = result.manifest.partial_coverage
        assert pc is not None  # the accounting always exists
        assert pc["count"] == 0
        assert pc["units"] == []


class TestMinSegmentAreaThreshold:
    """Sub-threshold slivers are dropped and accounted for. Pins the
    Gansu case from the first global run: a valid multi-part segment of
    7.8e-5 square meters (border-following floating point residue)
    crashed exactextract's traversal. The fixture's sliver piece is the
    same scale; if the threshold is ever lowered or removed, the default
    behavior asserted here changes and this fails loudly."""

    def _fixtures(self, tmp_path):
        targets = gpd.GeoDataFrame(
            {
                "target_id": ["T1", "T2"],
                "geometry": [box(0.0, 0.0, 2.0, 2.0), box(2.0, 0.0, 4.0, 2.0)],
            },
            crs="EPSG:4326",
        )
        # s1: a real piece in T1 plus a degenerate sliver inside T2,
        # 1e-9 by 1e-4 degrees, about 1.2e-3 square meters.
        source = gpd.GeoDataFrame(
            {
                "unit_id": ["s1"],
                "geometry": [
                    MultiPolygon(
                        [
                            box(0.5, 0.5, 1.5, 1.5),
                            box(2.5, 0.5, 2.5 + 1e-9, 0.5 + 1e-4),
                        ]
                    )
                ],
            },
            crs="EPSG:4326",
        )
        tp, sp = tmp_path / "t.parquet", tmp_path / "s.parquet"
        targets.to_parquet(tp)
        source.to_parquet(sp)
        return tp, sp

    def test_default_drops_sliver_and_records_it(self, tmp_path):
        tp, sp = self._fixtures(tmp_path)
        result = _run(_cfg(tp, sp, target_id_fields=["target_id"]))
        # The sliver's target is gone; all weight sits on the real piece.
        assert list(result.frame["target_id"]) == ["T1"]
        assert result.frame["areawt"].iloc[0] == pytest.approx(1.0)
        stats = result.manifest.extra["min_segment_area"]
        assert stats["threshold_m2"] == 1.0
        assert stats["segments_dropped"] == 1
        assert 0 < stats["max_discarded_share"] < 1e-9
        assert stats["units_fully_dropped"] == []

    def test_zero_threshold_keeps_the_sliver(self, tmp_path):
        tp, sp = self._fixtures(tmp_path)
        cfg = _cfg(tp, sp, target_id_fields=["target_id"])
        cfg = cfg.model_copy(
            update={
                "backend": cfg.backend.model_copy(
                    update={
                        "local": cfg.backend.local.model_copy(
                            update={"min_segment_area_m2": 0.0}
                        )
                    }
                )
            }
        )
        result = LocalBackend().compute(
            RegionSet.from_config(cfg.regions),
            None,
            from_config_list(cfg.weights),
            cfg,
        )
        assert set(result.frame["target_id"]) == {"T1", "T2"}
        assert result.manifest.extra["min_segment_area"]["segments_dropped"] == 0
