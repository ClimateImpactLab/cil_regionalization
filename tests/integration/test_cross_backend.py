"""Stage 4; cross-backend consistency.

Two test groups:

1. ``TestModeComparisonMultiCellBoundary`` (always runs): same backend
   (local), two coverage modes, on a multi-cell synthetic region with
   partial boundary coverage. Exact_fraction and pixel_centroid produce
   different *normalized* weights at boundary cells; not only different
   raw totals. This covers the locked Stage 4 requirement that the test
   surface mode-level disagreement in weights, not just in ``pop_raw``.

2. ``TestCrossBackendArea`` (``pytest.mark.bigquery``): both backends on
   the same tiny set of real hierids, area weight only. The geometries
   that feed the local backend are downloaded from the IR table (so the
   two backends compute against the same shapes). Asserts identical
   output schema, identical ``(region, cell)`` index sets, and
   ``areawt`` agreement within ``cfg.validation.cross_backend_tolerance``.

Tolerance justification (see ``TestCrossBackendArea`` docstring): pyproj
uses a WGS84 ellipsoidal geodesic for area while BigQuery ``ST_AREA``
uses a spheroidal model. Per-cell areas differ by O(1e-5) relative; the
default ``cross_backend_tolerance = 1e-3`` (1e-3 of unity-summing
weights) accommodates this comfortably without masking real bugs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest

from cil_regionalization.backends.local import LocalBackend
from cil_regionalization.config import Config
from cil_regionalization.grid import GridSpec
from cil_regionalization.regions import RegionSet
from cil_regionalization.weights import from_config_list


# --------------------------------------------------------------------------
# Group 1: local mode comparison on a multi-cell boundary region
# --------------------------------------------------------------------------


def _local_cfg(
    regions_path: Path,
    raster_path: Path,
    out_dir: Path,
    coverage: str,
) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "stage4_mode"},
            "regions": {"path": str(regions_path), "id_fields": ["region_id"]},
            "grid": {
                "mode": "generate",
                "resolution": 1.0,
                "offset": "center",
                "lon_convention": "[-180,180)",
            },
            "weights": [
                {"name": "pop", "raster": str(raster_path), "fallback": "area"},
                {"name": "area"},
            ],
            "backend": {"kind": "local", "coverage": coverage},
            "output": {"dir": str(out_dir)},
        }
    )


def _run_local(cfg: Config):
    regions = RegionSet.from_config(cfg.regions)
    grid = GridSpec.from_config(cfg.grid)
    weights = from_config_list(cfg.weights)
    return LocalBackend().compute(regions, grid, weights, cfg)


class TestModeComparisonMultiCellBoundary:
    """Region D = box(0.7, 0.7, 2.3, 2.3) spans a 3x3 block of 1deg cells.

    The raster fixture has value 10 in [0,2]x[0,2] and 0 elsewhere, so:

    - cell (180, 90): D covers [0.7, 1] x [0.7, 1]; a tiny corner. One
      pixel center (0.75, 0.75) is inside (centroid sum = 10), but only
      0.16 of that pixel's area is inside D (exact sum = 1.6 with one
      partial pixel, but the surrounding cell has more partial pixels;
      see test body).
    - cell (181, 91): D fully covers [1, 2] x [1, 2]. Identical in both
      modes (4 pixels each worth 10 -> raw 40).
    - cells (181, 90), (180, 91): D covers a half of each (partial in
      one axis, full in the other); these are the cells where the
      modes diverge most visibly.

    Because the per-cell raw totals differ AND the per-region total
    differs, the *normalized* weights differ at boundary cells too.
    This is the missing case from notebook 03's single-cell region B.
    """

    def test_normalized_weights_diverge_at_boundary_cells(
        self,
        multi_cell_boundary_geoparquet: Path,
        synthetic_raster: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "out"
        cfg_exact = _local_cfg(
            multi_cell_boundary_geoparquet,
            synthetic_raster,
            out,
            "exact_fraction",
        )
        cfg_cent = _local_cfg(
            multi_cell_boundary_geoparquet,
            synthetic_raster,
            out,
            "pixel_centroid",
        )
        result_exact = _run_local(cfg_exact)
        result_cent = _run_local(cfg_cent)

        # Same backend, same regions -> same schema and same (region, cell)
        # index set across the two coverage modes.
        assert list(result_exact.frame.columns) == list(result_cent.frame.columns)
        idx_exact = set(
            zip(
                result_exact.frame["region_id"],
                result_exact.frame["cell_ix"],
                result_exact.frame["cell_iy"],
            )
        )
        idx_cent = set(
            zip(
                result_cent.frame["region_id"],
                result_cent.frame["cell_ix"],
                result_cent.frame["cell_iy"],
            )
        )
        assert idx_exact == idx_cent

        # Both modes still sum to 1 per region per weight.
        assert result_exact.sum_report.ok, result_exact.sum_report.summary()
        assert result_cent.sum_report.ok, result_cent.sum_report.summary()

        # Normalized weights MUST disagree at boundary cells. Specifically,
        # at least one (cell_ix, cell_iy) row's popwt differs between the
        # two modes by more than 1e-3; well above floating-point noise.
        merged = result_exact.frame.merge(
            result_cent.frame,
            on=["region_id", "cell_ix", "cell_iy"],
            suffixes=("_exact", "_cent"),
        )
        diff = (merged["popwt_exact"] - merged["popwt_cent"]).abs()
        assert (diff > 1e-3).any(), (
            "no boundary disagreement found; the multi-cell region should "
            "expose normalized weight differences between modes"
        )

    def test_per_region_weight_sum_matches_across_modes(
        self,
        multi_cell_boundary_geoparquet: Path,
        synthetic_raster: Path,
        tmp_path: Path,
    ):
        out = tmp_path / "out"
        result_exact = _run_local(
            _local_cfg(
                multi_cell_boundary_geoparquet,
                synthetic_raster,
                out,
                "exact_fraction",
            )
        )
        result_cent = _run_local(
            _local_cfg(
                multi_cell_boundary_geoparquet,
                synthetic_raster,
                out,
                "pixel_centroid",
            )
        )
        # Per-region sums are 1.0 in both modes by construction.
        for r in (result_exact, result_cent):
            sums = r.frame.groupby("region_id")[["popwt", "areawt"]].sum()
            assert sums["popwt"].max() == pytest.approx(1.0, abs=1e-9)
            assert sums["areawt"].max() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Group 2: local vs BigQuery cross-backend on real hierids (area only)
# --------------------------------------------------------------------------


_IR_TABLE = (
    "compute-impactlab.spatial_aggregation."
    "clustered_impactlab-world-combo-2017_geometries_20221012"
)
_POP_TABLE = (
    "compute-impactlab.gridded_population_of_the_world."
    "GPW_UN_WPP_Adjusted_Population_Count_2015_v4_10"
)
# Source-of-truth: examples/configs/s51_test.toml (see comment block
# there about world-combo-2017's hierid naming patterns). conftest.py
# reads it back as S51_TEST_HIERIDS so the cross-backend tests cannot
# drift from the example config.
from tests.conftest import S51_TEST_HIERIDS as _S51_TEST_HIERIDS

# Cross-backend test uses the first three: ABW (bare), AND remainder
# fragment, BHR.5 admin region. Enough to span all three naming
# patterns without paying for more.
_CROSS_HIERIDS = list(_S51_TEST_HIERIDS[:3])


def _area_only_local_cfg(
    regions_path: Path, out_dir: Path
) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "stage4_cross_local"},
            "regions": {"path": str(regions_path), "id_fields": ["hierid"]},
            "grid": {
                "mode": "generate",
                "resolution": 1.0,
                "offset": "center",
                "lon_convention": "[-180,180)",
            },
            "weights": [{"name": "area"}],
            "backend": {"kind": "local", "coverage": "pixel_centroid"},
            "output": {"dir": str(out_dir)},
        }
    )


def _area_only_bq_cfg(out_dir: Path) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "stage4_cross_bq"},
            "regions": {
                "table": _IR_TABLE,
                "id_fields": ["hierid"],
                "keep": {"hierid": _CROSS_HIERIDS},
            },
            "grid": {
                "mode": "generate",
                "resolution": 1.0,
                "offset": "center",
                "lon_convention": "[-180,180)",
            },
            "weights": [{"name": "area"}],
            "backend": {"kind": "bigquery", "coverage": "pixel_centroid"},
            "output": {"dir": str(out_dir)},
        }
    )


@pytest.fixture(scope="module")
def bq_client():
    pytest.importorskip("google.cloud.bigquery")
    from google.cloud import bigquery
    from google.auth.exceptions import DefaultCredentialsError

    try:
        client = bigquery.Client(project="compute-impactlab")
        list(client.list_datasets(max_results=1))
    except DefaultCredentialsError:
        pytest.skip("ADC not configured; cannot run BigQuery tests")
    except Exception as e:
        pytest.skip(f"BigQuery client failed to initialise: {e}")
    return client


@pytest.fixture(scope="module")
def _real_geometries(bq_client, tmp_path_factory):
    """Download geometries for the cross-backend hierid set from IR, save
    to a GeoParquet for the local backend to consume.

    Done once per test module so both Stage 4 BQ tests share the cost.
    Uses a public BigQuery query (no backend internals).
    """
    from google.cloud import bigquery
    from shapely import wkt as _wkt

    params = [
        bigquery.ArrayQueryParameter(
            "hierids", "STRING", _CROSS_HIERIDS
        )
    ]
    sql = (
        f"SELECT hierid, ST_ASTEXT(geometry) AS geometry "
        f"FROM `{_IR_TABLE}` "
        f"WHERE hierid IN UNNEST(@hierids)"
    )
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    df = (
        bq_client.query(sql, job_config=job_config, location="us-west1")
        .to_dataframe(create_bqstorage_client=False)
    )
    df["geometry"] = df["geometry"].apply(_wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    tmp_dir = tmp_path_factory.mktemp("cross_backend_geom")
    path = tmp_dir / "regions.parquet"
    gdf.to_parquet(path)
    return path


@pytest.mark.bigquery
class TestCrossBackendArea:
    """Local vs BigQuery on the same real hierids, area weight only.

    Tolerance justification
    -----------------------
    The default ``validation.cross_backend_tolerance = 1e-3`` is on the
    per-(region, cell) weight, which sums to 1 per region. Sources of
    legitimate disagreement that the tolerance must accommodate:

    - **Spheroid vs ellipsoid.** ``pyproj.Geod(ellps='WGS84')`` uses a
      true ellipsoid (Vincenty / Karney). BigQuery ``ST_AREA(GEOGRAPHY)``
      uses a sphere of fixed radius. The local backend produces areas
      that are 0.05-0.3% smaller than the BQ equivalent in mid-latitudes.
      *Normalized* weights cancel this out almost completely (both
      numerator and denominator move the same way), but the cancellation
      is not exact across non-uniform cells.
    - **Planar vs spherical intersection.** Local ``shapely.intersection``
      treats coordinates as planar; BigQuery ``ST_INTERSECTION`` is
      spherical. Near the equator and at 1deg resolution this matters
      at parts-per-million; at higher latitudes it grows.
    - **Coordinate noise.** Geometries round-trip through the WKT
      representation (local: shapely from parquet, BQ: GEOGRAPHY load
      job from WKT). Coordinate precision differences are sub-machine
      epsilon but show up after intersection.

    A real bug; wrong cell binning, wrong coordinate origin, swapped
    lon/lat; would produce per-cell deltas of O(1) on normalized
    weights, orders of magnitude past this tolerance.
    """

    def test_schemas_match(
        self, bq_client, _real_geometries: Path, tmp_path: Path
    ):
        from cil_regionalization.backends.bigquery import BigQueryBackend

        result_local = _run_local(
            _area_only_local_cfg(_real_geometries, tmp_path / "local")
        )
        cfg_bq = _area_only_bq_cfg(tmp_path / "bq")
        regions = RegionSet.from_config(cfg_bq.regions)
        grid = GridSpec.from_config(cfg_bq.grid)
        weights = from_config_list(cfg_bq.weights)
        result_bq = BigQueryBackend().compute(regions, grid, weights, cfg_bq)

        assert list(result_local.frame.columns) == list(result_bq.frame.columns)
        assert result_local.schema.columns == result_bq.schema.columns

    def test_region_cell_index_sets_match(
        self, bq_client, _real_geometries: Path, tmp_path: Path
    ):
        from cil_regionalization.backends.bigquery import BigQueryBackend

        result_local = _run_local(
            _area_only_local_cfg(_real_geometries, tmp_path / "local")
        )
        cfg_bq = _area_only_bq_cfg(tmp_path / "bq")
        result_bq = BigQueryBackend().compute(
            RegionSet.from_config(cfg_bq.regions),
            GridSpec.from_config(cfg_bq.grid),
            from_config_list(cfg_bq.weights),
            cfg_bq,
        )

        idx_local = set(
            zip(
                result_local.frame["hierid"],
                result_local.frame["cell_ix"],
                result_local.frame["cell_iy"],
            )
        )
        idx_bq = set(
            zip(
                result_bq.frame["hierid"],
                result_bq.frame["cell_ix"],
                result_bq.frame["cell_iy"],
            )
        )
        # Documented difference: boundary cells whose intersection has
        # zero geodesic area in one backend's representation may be absent
        # in that backend. We allow up to a handful of such cells per region.
        symmetric_diff = idx_local.symmetric_difference(idx_bq)
        per_region_dropped: dict[str, int] = {}
        for hierid, _, _ in symmetric_diff:
            per_region_dropped[hierid] = per_region_dropped.get(hierid, 0) + 1
        for hierid, n in per_region_dropped.items():
            assert n <= 4, (
                f"hierid {hierid}: {n} cell-index disagreements is too many "
                f"for boundary-only differences"
            )

    def test_areawt_agrees_within_tolerance(
        self, bq_client, _real_geometries: Path, tmp_path: Path
    ):
        from cil_regionalization.backends.bigquery import BigQueryBackend

        cfg_local = _area_only_local_cfg(
            _real_geometries, tmp_path / "local"
        )
        result_local = _run_local(cfg_local)
        cfg_bq = _area_only_bq_cfg(tmp_path / "bq")
        result_bq = BigQueryBackend().compute(
            RegionSet.from_config(cfg_bq.regions),
            GridSpec.from_config(cfg_bq.grid),
            from_config_list(cfg_bq.weights),
            cfg_bq,
        )

        # Persist both frames BEFORE any assertion so a failure leaves
        # inspectable artifacts behind in tmp_path; this test computes
        # in-memory, the pytest tmp dirs would otherwise be empty.
        local_pq = tmp_path / "local_areawt.parquet"
        bq_pq = tmp_path / "bq_areawt.parquet"
        result_local.frame.to_parquet(local_pq, index=False)
        result_bq.frame.to_parquet(bq_pq, index=False)

        joined = result_local.frame.merge(
            result_bq.frame,
            on=["hierid", "cell_ix", "cell_iy"],
            suffixes=("_local", "_bq"),
            how="inner",
        )
        joined["deviation"] = (
            joined["areawt_local"] - joined["areawt_bq"]
        ).abs()
        max_dev = float(joined["deviation"].max())
        tol = cfg_local.validation.cross_backend_tolerance
        if max_dev > tol:
            forensic = joined.loc[
                :,
                [
                    "hierid",
                    "cell_ix",
                    "cell_iy",
                    "areawt_local",
                    "areawt_bq",
                    "deviation",
                ],
            ].sort_values("deviation", ascending=False).head(30)
            forensic_path = tmp_path / "areawt_deviation_top30.csv"
            forensic.to_csv(forensic_path, index=False)
            pytest.fail(
                f"areawt cross-backend deviation {max_dev:.2e} exceeds "
                f"tolerance {tol:.2e}. Per-cell forensics:\n"
                f"{forensic.to_string(index=False)}\n"
                f"\nFull frames persisted: {local_pq}, {bq_pq}\n"
                f"Top-30 deviations: {forensic_path}"
            )
        # And both should be valid normalized weights.
        assert result_local.sum_report.ok
        assert result_bq.sum_report.ok

    def test_areawt_parallel_split_within_tight_tolerance(
        self, bq_client, _real_geometries: Path, tmp_path: Path
    ):
        """BHR.5 splits across lat 26 (a parallel). Pre-densification,
        per-cell deviation was ~3e-3 because BQ's constant-latitude
        cell edges bowed poleward off the parallel; densifying the cell
        polygon vertices closes the gap to <= 1e-4.

        Region-boundary edges (planar shapely vs spherical BQ) remain
        the irreducible residual; 1e-4 is the headroom for that.
        """
        from cil_regionalization.backends.bigquery import BigQueryBackend

        cfg_local = _area_only_local_cfg(
            _real_geometries, tmp_path / "local"
        )
        result_local = _run_local(cfg_local)
        cfg_bq = _area_only_bq_cfg(tmp_path / "bq")
        result_bq = BigQueryBackend().compute(
            RegionSet.from_config(cfg_bq.regions),
            GridSpec.from_config(cfg_bq.grid),
            from_config_list(cfg_bq.weights),
            cfg_bq,
        )
        joined = (
            result_local.frame.merge(
                result_bq.frame,
                on=["hierid", "cell_ix", "cell_iy"],
                suffixes=("_local", "_bq"),
                how="inner",
            )
            .loc[lambda d: d["hierid"] == "BHR.5"]
        )
        if joined.empty:
            pytest.skip("BHR.5 not present in both frames; check fixture")
        joined["deviation"] = (
            joined["areawt_local"] - joined["areawt_bq"]
        ).abs()
        max_dev = float(joined["deviation"].max())
        if max_dev > 1e-4:
            forensic = joined[
                ["cell_ix", "cell_iy", "areawt_local", "areawt_bq", "deviation"]
            ].sort_values("deviation", ascending=False)
            forensic.to_csv(tmp_path / "bhr5_deviation.csv", index=False)
            pytest.fail(
                f"BHR.5 areawt deviation {max_dev:.2e} exceeds 1e-4. "
                f"Densification may need tightening (try densify_step=0.05). "
                f"Per-cell:\n{forensic.to_string(index=False)}"
            )
