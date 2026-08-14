"""End-to-end test of the local backend on synthetic 3-region + raster.

The synthetic raster (`tests/conftest.py::synthetic_raster`) holds value 10
inside [0, 2] x [0, 2] and 0 elsewhere on a 16x16 grid of 0.5deg pixels.
The synthetic regions:

    A : box [0, 0]-[2, 2]              ; fully inside the value-10 zone
    B : box [0.1, 0.1]-[0.3, 0.3]      ; smaller than a 1deg cell
    C : box [2, 2]-[4, 4] minus a hole ; entirely outside the value-10 zone

Consequences:
    - A and B have positive raw pop totals (native normalization).
    - C has zero pop everywhere -> exercises the fallback policy.
    - B sits inside a single grid cell, so its pop weight is 1.0 there.
    - exact_fraction and pixel_centroid disagree at the A-edge cells
      (boundary pixels covered vs centred), giving us the documented
      cross-mode difference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from segment_weights.backends.local import LocalBackend
from segment_weights.config import Config
from segment_weights.fallback import count_methods
from segment_weights.grid import GridSpec
from segment_weights.regions import RegionSet
from segment_weights.schema import OutputSchema
from segment_weights.validate import check_sum_to_one
from segment_weights.weights import from_config_list


def _cfg(
    regions_path: Path,
    raster_path: Path,
    output_dir: Path,
    *,
    coverage: str = "exact_fraction",
    pop_fallback: str = "area",
) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "test"},
            "regions": {"path": str(regions_path), "id_fields": ["region_id"]},
            "grid": {
                "mode": "generate",
                "resolution": 1.0,
                "offset": "center",
                "lon_convention": "[-180,180)",
            },
            "weights": [
                {
                    "name": "pop",
                    "raster": str(raster_path),
                    "fallback": pop_fallback,
                },
                {"name": "area"},
            ],
            "backend": {"kind": "local", "coverage": coverage},
            "output": {"dir": str(output_dir)},
        }
    )


def _run(cfg: Config):
    regions = RegionSet.from_config(cfg.regions)
    grid = GridSpec.from_config(cfg.grid)
    weight_specs = from_config_list(cfg.weights)
    return LocalBackend().compute(regions, grid, weight_specs, cfg)


class TestExactFractionEnd2End:
    def test_sum_to_one_per_region(self, three_region_geoparquet, synthetic_raster, tmp_path):
        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        assert result.sum_report.ok, result.sum_report.summary()

    def test_schema_matches_canonical(self, three_region_geoparquet, synthetic_raster, tmp_path):
        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        expected = OutputSchema(
            id_fields=("region_id",), weight_names=("pop", "area")
        )
        assert list(result.frame.columns) == list(expected.columns)
        expected.validate_frame(result.frame)

    def test_tiny_region_concentrated_in_one_cell(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        b_rows = result.frame.loc[result.frame["region_id"] == "B"]
        assert len(b_rows) == 1, (
            "B is 0.2deg square inside a single 1deg cell; expected one row"
        )
        # B has positive pop (one raster pixel partially covered) -> native
        assert b_rows["pop_method"].iloc[0] == "native"
        # weights for a one-cell region must be 1.0
        assert b_rows["popwt"].iloc[0] == pytest.approx(1.0)
        assert b_rows["areawt"].iloc[0] == pytest.approx(1.0)

    def test_zero_pop_region_triggers_area_fallback(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        c_rows = result.frame.loc[result.frame["region_id"] == "C"]
        assert len(c_rows) > 0
        assert (c_rows["pop_method"] == "area_fallback").all()
        # area_fallback re-uses the area weights, which sum to 1 per region
        assert c_rows["popwt"].sum() == pytest.approx(1.0)

    def test_manifest_counts_match(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        assert result.manifest.row_counts["regions"] == 3
        assert result.manifest.row_counts["total"] == len(result.frame)
        # 2 native (A, B), 1 area_fallback (C)
        assert result.manifest.fallback_counts["pop"] == {
            "native": 2,
            "area_fallback": 1,
        }
        # all 3 native for area
        assert result.manifest.fallback_counts["area"] == {"native": 3}

    def test_manifest_records_raster_hash(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        assert "weight:pop" in result.manifest.inputs
        assert len(result.manifest.inputs["weight:pop"]) == 64

    def test_geodesic_area_positive_everywhere(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        assert (result.frame["area_raw"] > 0).all()


class TestPixelCentroidEnd2End:
    def test_sum_to_one_per_region(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = _cfg(
            three_region_geoparquet,
            synthetic_raster,
            tmp_path,
            coverage="pixel_centroid",
        )
        result = _run(cfg)
        assert result.sum_report.ok, result.sum_report.summary()

    def test_pixel_centroid_b_picks_one_pixel(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        # B is 0.2x0.2; raster pixel (0.25, 0.25) center is inside B -> raw=10
        cfg = _cfg(
            three_region_geoparquet,
            synthetic_raster,
            tmp_path,
            coverage="pixel_centroid",
        )
        result = _run(cfg)
        b = result.frame.loc[result.frame["region_id"] == "B"].iloc[0]
        assert b["pop_raw"] == pytest.approx(10.0)


class TestCoverageDifferenceAtBoundary:
    """Document the exact_fraction vs pixel_centroid disagreement on tiny B.

    For region B = [0.1, 0.1]-[0.3, 0.3] with pixel (0.25, 0.25):
        - pixel_centroid: center is inside B -> raw = 10
        - exact_fraction: B covers fraction 0.04/0.25 = 0.16 -> raw = 1.6
    The normalized weights are identical (both 1.0 in B's only cell),
    but the raw column captures the documented mode difference.
    """

    def test_b_raw_differs_by_mode(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg_exact = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        cfg_cent = _cfg(
            three_region_geoparquet,
            synthetic_raster,
            tmp_path,
            coverage="pixel_centroid",
        )
        r_exact = _run(cfg_exact)
        r_cent = _run(cfg_cent)
        b_exact = r_exact.frame.loc[r_exact.frame["region_id"] == "B"].iloc[0]
        b_cent = r_cent.frame.loc[r_cent.frame["region_id"] == "B"].iloc[0]
        assert b_exact["pop_raw"] == pytest.approx(1.6)
        assert b_cent["pop_raw"] == pytest.approx(10.0)
        # normalized weight is 1.0 in both modes (only one cell)
        assert b_exact["popwt"] == pytest.approx(b_cent["popwt"])


class TestZeroFallback:
    """The `zero` policy keeps the region in the output with 0 weights and
    a `zero` method label; the validator must flag it."""

    def test_zero_fallback_marks_and_fails_sum(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = _cfg(
            three_region_geoparquet,
            synthetic_raster,
            tmp_path,
            pop_fallback="zero",
        )
        result = _run(cfg)
        c_rows = result.frame.loc[result.frame["region_id"] == "C"]
        assert (c_rows["pop_method"] == "zero").all()
        assert (c_rows["popwt"] == 0.0).all()
        # validator should fail on C for pop (sum = 0, expected 1)
        assert not result.sum_report.ok
        failed_regions = set(result.sum_report.failures["region_id"])
        failed_weights = set(result.sum_report.failures["weight"])
        assert failed_regions == {"C"}
        assert failed_weights == {"pop"}


class TestAreaWeightedCropWeight:
    """area_weighted_sum: raster value (fraction) x geodesic pixel area.

    Pins the load-bearing crop-weight identity. Pairing with the 'nan'
    fallback policy matches the legacy crop file convention where regions
    with zero cropland carry NaN cropwt rather than 0.
    """

    def _cfg_with_crop(
        self,
        regions_path: Path,
        fraction_raster: Path,
        out_dir: Path,
    ) -> Config:
        return Config.model_validate(
            {
                "project": {"name": "test"},
                "regions": {
                    "path": str(regions_path),
                    "id_fields": ["region_id"],
                },
                "grid": {
                    "mode": "generate",
                    "resolution": 1.0,
                    "offset": "center",
                    "lon_convention": "[-180,180)",
                },
                "weights": [
                    {
                        "name": "crop",
                        "raster": str(fraction_raster),
                        "kind": "area_weighted_sum",
                        "fallback": "nan",
                    },
                    {"name": "area"},
                ],
                "backend": {"kind": "local", "coverage": "exact_fraction"},
                "output": {"dir": str(out_dir)},
            }
        )

    def test_crop_raw_equals_fraction_times_geodesic_area(
        self, three_region_geoparquet, synthetic_fraction_raster, tmp_path
    ):
        """Region A occupies [0, 2] x [0, 2] in lon/lat; the fraction is
        0.5 over that whole footprint. crop_raw summed across A's cells
        must equal 0.5 x geodesic_area([0,0]-[2,2]).
        """
        from pyproj import Geod
        from shapely.geometry import box as sbox

        cfg = self._cfg_with_crop(
            three_region_geoparquet, synthetic_fraction_raster, tmp_path
        )
        result = _run(cfg)
        a_rows = result.frame.loc[result.frame["region_id"] == "A"]
        # 0.5 fraction is everywhere inside A's footprint; A is fully
        # covered by [0, 2] x [0, 2]; expected = 0.5 x geodesic area
        # of that box.
        geod = Geod(ellps="WGS84")
        expected_area, _ = geod.geometry_area_perimeter(sbox(0.0, 0.0, 2.0, 2.0))
        expected_raw = 0.5 * abs(expected_area)
        # Tolerance 1e-3: per-pixel geodesic areas summed do not equal
        # the geodesic area of their union exactly on an ellipsoid (each
        # small rectangle's geodesic edges curve independently). The
        # relevant correctness check is "right order of magnitude and
        # right sign"; a factor-of-2 unit error would blow this past
        # any reasonable threshold.
        assert a_rows["crop_raw"].sum() == pytest.approx(expected_raw, rel=1e-3)

    def test_zero_crop_region_gets_nan_fallback(
        self, three_region_geoparquet, synthetic_fraction_raster, tmp_path
    ):
        """Region C lies outside the fraction raster's positive footprint.
        With fallback='nan', cropwt must be NaN everywhere on C, the
        method label 'nan', and the sum-to-1 validator must NOT flag C
        (NaN > tolerance is False).
        """
        cfg = self._cfg_with_crop(
            three_region_geoparquet, synthetic_fraction_raster, tmp_path
        )
        result = _run(cfg)
        c_rows = result.frame.loc[result.frame["region_id"] == "C"]
        assert (c_rows["crop_method"] == "nan").all()
        assert c_rows["cropwt"].isna().all()
        # Validator passes for crop on C (NaN is invisible).
        assert result.sum_report.ok, result.sum_report.summary()

    def test_native_crop_region_sums_to_one(
        self, three_region_geoparquet, synthetic_fraction_raster, tmp_path
    ):
        cfg = self._cfg_with_crop(
            three_region_geoparquet, synthetic_fraction_raster, tmp_path
        )
        result = _run(cfg)
        a_rows = result.frame.loc[result.frame["region_id"] == "A"]
        assert a_rows["cropwt"].sum() == pytest.approx(1.0)
        assert (a_rows["crop_method"] == "native").all()


@pytest.mark.dask
class TestDaskLocalParity:
    """Same input, dask='off' vs dask='local'; output must be byte-identical
    on the value columns. Worker count > number of regions is intentional:
    a chunk-distribution bug that drops the empty chunk would silently
    truncate the output.
    """

    def test_three_region_serial_vs_dask_local_identical(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        cfg_serial = _cfg(three_region_geoparquet, synthetic_raster, tmp_path / "ser")
        result_serial = _run(cfg_serial)

        cfg_dask = _cfg(three_region_geoparquet, synthetic_raster, tmp_path / "dsk")
        cfg_dask = cfg_dask.model_copy(
            update={
                "backend": cfg_dask.backend.model_copy(
                    update={
                        "local": cfg_dask.backend.local.model_copy(
                            update={
                                "dask": "local",
                                "n_workers": 2,
                                "threads_per_worker": 1,
                            }
                        )
                    }
                )
            }
        )
        result_dask = _run(cfg_dask)

        cols = [
            "region_id", "cell_ix", "cell_iy",
            "cell_lon", "cell_lat",
            "popwt", "pop_raw", "pop_method",
            "areawt", "area_raw", "area_method",
        ]
        ser = result_serial.frame[cols].sort_values(
            ["region_id", "cell_ix", "cell_iy"]
        ).reset_index(drop=True)
        dsk = result_dask.frame[cols].sort_values(
            ["region_id", "cell_ix", "cell_iy"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(ser, dsk, check_exact=True)


class TestAntimeridianSplitting:
    """Regions whose bbox spans >180deg longitude must produce cells at
    both ends of the global ix range, with no duplicate keys. Before
    the split, the local backend was correct only when the polygon's
    eastern tip was already in canonical [-180, 180); a polygon with
    extended longitude (lon > 180) silently lost its wrapped half.
    """

    def _cfg(self, regions_path, raster_path, out_dir):
        return Config.model_validate(
            {
                "project": {"name": "test"},
                "regions": {
                    "path": str(regions_path),
                    "id_fields": ["region_id"],
                },
                "grid": {
                    "mode": "generate",
                    "resolution": 1.0,
                    "offset": "center",
                    "lon_convention": "[-180,180)",
                },
                "weights": [
                    {
                        "name": "pop",
                        "raster": str(raster_path),
                        "fallback": "area",
                    },
                    {"name": "area"},
                ],
                "backend": {"kind": "local", "coverage": "exact_fraction"},
                "output": {"dir": str(out_dir)},
            }
        )

    def _run(self, cfg):
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        return LocalBackend().compute(regions, grid, weights, cfg)

    def test_cells_appear_on_both_dateline_sides(
        self, dateline_crossers_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = self._cfg(
            dateline_crossers_geoparquet, synthetic_raster, tmp_path
        )
        result = self._run(cfg)
        fji = result.frame.loc[result.frame["region_id"] == "FJI_like"]
        # FJI_like sits at lon -179.5..-178.5 (west of antimeridian) and
        # 178.5..179.5 (east); cells must straddle both extremes.
        assert (fji["cell_ix"] <= 1).any(), (
            "no west-of-dateline cells; split likely silently dropped them"
        )
        assert (fji["cell_ix"] >= 358).any(), (
            "no east-of-dateline cells; split likely silently dropped them"
        )

    def test_no_duplicate_keys(
        self, dateline_crossers_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = self._cfg(
            dateline_crossers_geoparquet, synthetic_raster, tmp_path
        )
        result = self._run(cfg)
        keys = ["region_id", "cell_ix", "cell_iy"]
        assert not result.frame.duplicated(subset=keys).any(), (
            "duplicate (region, cell_ix, cell_iy) keys; the split "
            "halves overlapped, which they should not at lon=0"
        )

    def test_per_region_sum_to_one(
        self, dateline_crossers_geoparquet, synthetic_raster, tmp_path
    ):
        cfg = self._cfg(
            dateline_crossers_geoparquet, synthetic_raster, tmp_path
        )
        result = self._run(cfg)
        assert result.sum_report.ok, result.sum_report.summary()

    def test_total_area_matches_geodesic_for_fji(
        self, dateline_crossers_geoparquet, synthetic_raster, tmp_path
    ):
        """Sum of area_raw across the FJI-like region matches the
        polygon's geodesic area (within float noise). If splitting ever
        dropped a half, the sum would come up short by ~50%.

        Restricted to FJI (mid-latitude tropical) because pyproj's
        `Geod.geometry_area_perimeter` reports a different area for
        the polar WIDE_like multipolygon; multi-polygon planar quads
        at high latitudes behave oddly under pyproj's algorithm. The
        cell-coverage and no-duplicate checks above still cover
        WIDE_like.
        """
        import geopandas as gpd_
        from pyproj import Geod

        cfg = self._cfg(
            dateline_crossers_geoparquet, synthetic_raster, tmp_path
        )
        result = self._run(cfg)
        gdf = gpd_.read_parquet(dateline_crossers_geoparquet)
        geod = Geod(ellps="WGS84")
        polygon_area = abs(
            geod.geometry_area_perimeter(
                gdf.loc[gdf["region_id"] == "FJI_like"].geometry.iloc[0]
            )[0]
        )
        our_sum = result.frame.loc[
            result.frame["region_id"] == "FJI_like", "area_raw"
        ].sum()
        rel = abs(our_sum - polygon_area) / polygon_area
        assert rel < 1e-4, (
            f"FJI_like: summed area {our_sum:.0f} vs polygon "
            f"{polygon_area:.0f}, rel {rel:.2e}"
        )

    def test_wide_like_covers_both_sides_with_many_cells(
        self, dateline_crossers_geoparquet, synthetic_raster, tmp_path
    ):
        """Coverage smoke check for the polar dateline crosser: cells
        must appear in both the western (ix < 90) and eastern (ix > 270)
        bands and total at least ~50 cells. Catches the 'one-half-was-
        dropped' regression without depending on pyproj polygon area
        for the assertion."""
        cfg = self._cfg(
            dateline_crossers_geoparquet, synthetic_raster, tmp_path
        )
        result = self._run(cfg)
        wide = result.frame.loc[result.frame["region_id"] == "WIDE_like"]
        assert (wide["cell_ix"] < 90).any()
        assert (wide["cell_ix"] > 270).any()
        # Each half spans 80deg lon x 3deg lat -> 240 cells per half
        # at 1deg; allow generous slack for boundary filtering.
        assert len(wide) >= 100, (
            f"WIDE_like produced only {len(wide)} cells; one half was "
            f"likely dropped (expect ~480)"
        )


class TestNearestCellSynthesis:
    """A region whose ST_INTERSECTION with every overlapping cell rounds
    to zero geodesic area still gets one synthesized row."""

    def _synthetic_tiny_region(self, tmp_path):
        """A degenerate (single-point) polygon; intersects no cell with
        positive area. Forces the synthesis path."""
        import geopandas as gpd
        from shapely.geometry import Point

        # A buffer-of-zero polygon: a point treated as a polygon. shapely's
        # is_empty returns False on this but it has zero area.
        gdf = gpd.GeoDataFrame(
            {
                "region_id": ["tiny"],
                "geometry": [Point(1.234, 2.345).buffer(0)],
            },
            crs="EPSG:4326",
        )
        p = tmp_path / "tiny.parquet"
        gdf.to_parquet(p)
        return p

    def test_degenerate_region_gets_one_synthesized_row(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        """A regions file that mixes ordinary and tiny regions: the tiny
        one drops out of the normal pipeline; synthesis adds it back."""
        import geopandas as gpd
        from shapely.geometry import Point, box as _box

        gdf = gpd.GeoDataFrame(
            {
                "region_id": ["A", "tiny"],
                "geometry": [
                    _box(0.0, 0.0, 2.0, 2.0),
                    Point(5.0, 5.0),
                ],
            },
            crs="EPSG:4326",
        )
        regions_path = tmp_path / "mixed.parquet"
        gdf.to_parquet(regions_path)

        cfg = _cfg(regions_path, synthetic_raster, tmp_path / "out")
        result = _run(cfg)
        ids = set(result.frame["region_id"].unique())
        assert ids == {"A", "tiny"}
        tiny_rows = result.frame.loc[result.frame["region_id"] == "tiny"]
        assert len(tiny_rows) == 1
        row = tiny_rows.iloc[0]
        assert row["popwt"] == pytest.approx(1.0)
        assert row["areawt"] == pytest.approx(1.0)
        assert row["pop_method"] == "nearest_cell"
        assert row["area_method"] == "nearest_cell"
        assert result.manifest.row_counts["nearest_cell"] == 1
        # Method counts in the manifest include nearest_cell.
        assert result.manifest.fallback_counts["pop"].get("nearest_cell") == 1
        assert result.manifest.fallback_counts["area"].get("nearest_cell") == 1

    def test_all_regions_degenerate(self, tmp_path):
        """If every requested region is degenerate, the entire output is
        synthesized; and still covers every region."""
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {
                "region_id": ["P", "Q"],
                "geometry": [
                    Point(1.0, 1.0),
                    Point(2.0, 2.0),
                ],
            },
            crs="EPSG:4326",
        )
        regions_path = tmp_path / "all_degenerate.parquet"
        gdf.to_parquet(regions_path)

        # Build a minimal raster so the local backend has a pop weight to
        # process; the synthesis path doesn't read it for nearest_cell rows.
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        raster_path = tmp_path / "raster.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(0.0, 4.0, 1.0, 1.0),
        ) as ds:
            ds.write(np.ones((4, 4), dtype=np.float32), 1)

        cfg = _cfg(regions_path, raster_path, tmp_path / "out")
        result = _run(cfg)
        assert set(result.frame["region_id"]) == {"P", "Q"}
        assert result.manifest.row_counts["nearest_cell"] == 2
        assert result.sum_report.ok


class TestIoRoundTrip:
    def test_writes_parquet_and_manifest(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        from segment_weights.io import write_result

        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        paths = write_result(result, str(tmp_path / "out"), "parquet")
        assert Path(paths["parquet"]).exists()
        assert Path(paths["manifest"]).exists()
        roundtrip = pd.read_parquet(paths["parquet"])
        assert list(roundtrip.columns) == list(result.frame.columns)
        assert len(roundtrip) == len(result.frame)

    def test_writes_both_formats(
        self, three_region_geoparquet, synthetic_raster, tmp_path
    ):
        from segment_weights.io import write_result

        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path)
        result = _run(cfg)
        paths = write_result(result, str(tmp_path / "out"), "both")
        assert Path(paths["parquet"]).exists()
        assert Path(paths["csv"]).exists()


class TestCli:
    def test_validate_passes(
        self, three_region_geoparquet, synthetic_raster, tmp_path, capsys
    ):
        from segment_weights.cli import main

        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path / "out")
        cfg_path = tmp_path / "cfg.toml"
        _write_toml(cfg_path, cfg)
        rc = main(["validate", str(cfg_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "validate: OK" in captured.out

    def test_validate_flags_missing_raster(
        self, three_region_geoparquet, tmp_path, capsys
    ):
        from segment_weights.cli import main

        cfg = _cfg(
            three_region_geoparquet,
            tmp_path / "missing.tif",
            tmp_path / "out",
        )
        cfg_path = tmp_path / "cfg.toml"
        _write_toml(cfg_path, cfg)
        rc = main(["validate", str(cfg_path)])
        captured = capsys.readouterr()
        assert rc == 1
        assert "missing.tif" in captured.err

    def test_run_end_to_end(
        self, three_region_geoparquet, synthetic_raster, tmp_path, capsys
    ):
        from segment_weights.cli import main

        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path / "out")
        cfg_path = tmp_path / "cfg.toml"
        _write_toml(cfg_path, cfg)
        rc = main(["run", str(cfg_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "sum_to_one=ok" in captured.out
        assert (tmp_path / "out" / "weights.parquet").exists()
        assert (tmp_path / "out" / "weights.manifest.json").exists()

    def test_run_test_mode_caps_regions(
        self, three_region_geoparquet, synthetic_raster, tmp_path, capsys
    ):
        from segment_weights.cli import main

        cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path / "out")
        cfg_path = tmp_path / "cfg.toml"
        _write_toml(cfg_path, cfg)
        rc = main(["run", str(cfg_path), "--test-mode"])
        captured = capsys.readouterr()
        assert rc == 0
        # 3-region fixture is exactly the cap so the cap is a no-op but the
        # banner still prints to stderr
        assert "test-mode active" in captured.err


class TestCliLegacyCsv:
    """`segweights run --legacy-csv` writes weights_legacy.csv next to the
    canonical output in the 13-column schema. Pinned end-to-end since the
    CLI wiring is otherwise untested outside the run smoke."""

    def test_legacy_csv_written_alongside_parquet(
        self, three_region_gdf, synthetic_fraction_raster, synthetic_raster, tmp_path, capsys
    ):
        from segment_weights.cli import main
        from segment_weights.legacy_export import LEGACY_COLUMNS
        import geopandas as gpd

        # Rename the synthetic fixture's id column to hierid for legacy schema.
        gdf = three_region_gdf.rename(columns={"region_id": "hierid"})
        regions_path = tmp_path / "regions_hierid.parquet"
        gdf.to_parquet(regions_path)

        cfg_path = tmp_path / "cfg.toml"
        cfg_path.write_text(
            f"""
[project]
name = "legacy_csv_test"

[regions]
path = "{regions_path}"
id_fields = ["hierid"]

[grid]
mode = "generate"
resolution = 1.0
offset = "center"
lon_convention = "[-180,180)"

[[weights]]
name = "crop"
raster = "{synthetic_fraction_raster}"
kind = "area_weighted_sum"
fallback = "nan"

[[weights]]
name = "pop"
raster = "{synthetic_raster}"
fallback = "area"

[[weights]]
name = "area"

[backend]
kind = "local"
coverage = "exact_fraction"

[output]
dir = "{tmp_path / 'out'}"
format = "parquet"
"""
        )

        rc = main(["run", str(cfg_path), "--legacy-csv"])
        captured = capsys.readouterr()
        assert rc == 0, captured.err

        legacy_path = tmp_path / "out" / "weights_legacy.csv"
        assert legacy_path.exists(), captured.out
        df = pd.read_csv(legacy_path)
        assert list(df.columns) == list(LEGACY_COLUMNS)
        # Region C is outside the fraction raster -> cropwt NaN under
        # fallback='nan'; CSV round-trip preserves the NaN.
        c = df.loc[df["hierid"] == "C"]
        assert c["cropwt"].isna().all()
        # shpid is 0-based dense rank of sorted hierid (A=0, B=1, C=2).
        rank = df.drop_duplicates("hierid").set_index("hierid")["shpid"]
        assert rank["A"] == 0
        assert rank["B"] == 1
        assert rank["C"] == 2


def _write_toml(path: Path, cfg: Config) -> None:
    """Render a Config back to a minimal TOML so the CLI can load it."""
    lines: list[str] = []
    lines.append("[project]")
    lines.append(f'name = "{cfg.project.name}"')
    lines.append("[regions]")
    lines.append(f'path = "{cfg.regions.path}"')
    lines.append(f"id_fields = {list(cfg.regions.id_fields)!r}")
    lines.append("[grid]")
    lines.append(f'mode = "{cfg.grid.mode}"')
    lines.append(f"resolution = {cfg.grid.resolution}")
    lines.append(f'offset = "{cfg.grid.offset}"')
    lines.append(f'lon_convention = "{cfg.grid.lon_convention}"')
    for w in cfg.weights:
        lines.append("[[weights]]")
        lines.append(f'name = "{w.name}"')
        if w.raster is not None:
            lines.append(f'raster = "{w.raster}"')
        if w.fallback != "area":
            lines.append(f'fallback = "{w.fallback}"')
    lines.append("[backend]")
    lines.append(f'kind = "{cfg.backend.kind}"')
    lines.append(f'coverage = "{cfg.backend.coverage}"')
    lines.append("[output]")
    lines.append(f'dir = "{cfg.output.dir}"')
    lines.append(f'format = "{cfg.output.format}"')
    path.write_text("\n".join(lines))
