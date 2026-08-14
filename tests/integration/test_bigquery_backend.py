"""BigQuery backend tests.

Two groups:

- Mocked unit tests (always run): location resolution and mismatch errors,
  the geometry-fetch SQL, the temp-table lifecycle (load_job destination,
  cleanup on success and on failure), the main SQL referencing the temp
  table (not the IR table), the dry-run gate, and `_assemble_result`'s
  fallback path on a synthetic downloaded frame.
- Real-backend tests, marked ``bigquery`` and skipped by default
  (``pytest -m bigquery``): tiny end-to-end on cilresearch.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Even the mocked tests construct BigQueryBackend, whose import guard
# needs the client library present. Without the [bigquery] extra the
# whole module skips, and it skips as its full test count rather than
# one collapsed module entry, so the summary always shows how many
# tests the missing extra hides. A module-level importorskip would
# report the same situation as a single skip, which once let a broken
# fixture hide behind a green count.
import importlib.util

_HAS_BIGQUERY = importlib.util.find_spec("google.cloud.bigquery") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_BIGQUERY,
    reason="BigQuery tests need the [bigquery] extra installed",
)

from segment_weights.config import Config


_IR_TABLE = (
    "compute-impactlab.spatial_aggregation."
    "clustered_impactlab-world-combo-2017_geometries_20221012"
)
_POP_TABLE = (
    "compute-impactlab.gridded_population_of_the_world."
    "GPW_UN_WPP_Adjusted_Population_Count_2015_v4_10"
)


def _bq_cfg(**overrides) -> Config:
    d = {
        "project": {"name": "bq_test"},
        "regions": {
            "table": _IR_TABLE,
            "id_fields": ["hierid"],
            "keep": {"hierid": ["ABW", "BHS"]},
        },
        "grid": {
            "mode": "generate",
            "resolution": 1.0,
            "offset": "center",
            "lon_convention": "[-180,180)",
        },
        "weights": [
            {"name": "pop", "table": _POP_TABLE, "fallback": "area"},
            {"name": "area"},
        ],
        "backend": {
            "kind": "bigquery",
            "coverage": "pixel_centroid",
            "bigquery": {"staging_uri": "gs://example-staging/segment-weights/"},
        },
        "output": {"dir": "/tmp/segweights_bq_test"},
    }
    for key, val in overrides.items():
        d[key] = val
    return Config.model_validate(d)


def _make_dataset(location: str) -> MagicMock:
    ds = MagicMock()
    ds.location = location
    return ds


def _client_with_locations(
    ir_location: str = "us-west1",
    weight_location: str = "US",
    temp_location: str = "US",
) -> MagicMock:
    """Mocked client whose get_dataset returns the configured locations."""
    ir_dataset = _IR_TABLE.rsplit(".", 1)[0]
    weight_dataset = _POP_TABLE.rsplit(".", 1)[0]
    temp_dataset = "compute-impactlab.temp_workspace"

    locations = {
        ir_dataset: ir_location,
        weight_dataset: weight_location,
        temp_dataset: temp_location,
    }

    def _get_dataset(dataset_id: str):
        return _make_dataset(locations[dataset_id])

    client = MagicMock()
    client.get_dataset.side_effect = _get_dataset
    return client


# --------------------------------------------------------------------------
# Mocked unit tests (always run)
# --------------------------------------------------------------------------


class TestImport:
    def test_module_importable_with_extra(self):
        from segment_weights.backends.bigquery import BigQueryBackend  # noqa


class TestLocationResolution:
    def _backend(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        return BigQueryBackend()

    def _specs(self, cfg: Config):
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        regions = RegionSet.from_config(cfg.regions)
        weights = [w for w in from_config_list(cfg.weights) if not w.is_area]
        return regions, weights

    def test_happy_path_returns_ir_and_compute(self):
        cfg = _bq_cfg()
        regions, weights = self._specs(cfg)
        client = _client_with_locations(
            ir_location="us-west1", weight_location="US", temp_location="US"
        )
        ir_loc, compute_loc = self._backend()._resolve_locations(
            client, regions, weights, cfg
        )
        assert ir_loc == "us-west1"
        assert compute_loc == "US"

    def test_mismatch_raises_with_named_datasets(self):
        cfg = _bq_cfg()
        regions, weights = self._specs(cfg)
        client = _client_with_locations(
            ir_location="us-west1", weight_location="US", temp_location="EU"
        )
        with pytest.raises(ValueError) as exc:
            self._backend()._resolve_locations(client, regions, weights, cfg)
        msg = str(exc.value)
        assert "location mismatch" in msg
        assert "temp_workspace" in msg
        assert _POP_TABLE in msg
        assert "'EU'" in msg
        assert "'US'" in msg
        assert "Create no datasets" in msg

    def test_ir_in_different_region_is_fine(self):
        # The IR is allowed to be elsewhere; only temp + weights must match.
        cfg = _bq_cfg()
        regions, weights = self._specs(cfg)
        client = _client_with_locations(
            ir_location="us-east4", weight_location="US", temp_location="US"
        )
        ir_loc, compute_loc = self._backend()._resolve_locations(
            client, regions, weights, cfg
        )
        assert ir_loc == "us-east4"
        assert compute_loc == "US"


class TestGeometryFetch:
    def _backend(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        return BigQueryBackend()

    def test_sql_targets_only_ir_table(self):
        """The geometry fetch must reference exactly the IR table; never
        the weight table, never the temp dataset."""
        from segment_weights.regions import RegionSet

        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        client = MagicMock()
        fake_job = MagicMock()
        fake_job.to_dataframe.return_value = pd.DataFrame(
            {"hierid": ["ABW"], "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))"]}
        )
        client.query.return_value = fake_job
        backend = self._backend()
        backend._fetch_geometries(
            client, regions, ["ABW", "BHS"], "us-west1", cfg
        )
        sql = client.query.call_args[0][0]
        assert _IR_TABLE in sql
        assert _POP_TABLE not in sql
        assert "temp_workspace" not in sql
        # ST_ASTEXT for round-trip via JSONL load job
        assert "ST_ASTEXT(geometry)" in sql

    def test_passes_ir_location_to_query(self):
        from segment_weights.regions import RegionSet

        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        client = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame(
            {"hierid": ["ABW"], "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))"]}
        )
        self._backend()._fetch_geometries(
            client, regions, ["ABW"], "us-west1", cfg
        )
        assert client.query.call_args.kwargs["location"] == "us-west1"

    def test_hierids_passed_as_array_parameter(self):
        from google.cloud import bigquery
        from segment_weights.regions import RegionSet

        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        client = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame(
            {"hierid": [], "geometry": []}
        )
        self._backend()._fetch_geometries(
            client, regions, ["ABW", "BHS"], "us-west1", cfg
        )
        # Default download path: BQ Storage Read API is explicitly opted out.
        assert (
            client.query.return_value.to_dataframe.call_args.kwargs[
                "create_bqstorage_client"
            ]
            is False
        )
        job_config = client.query.call_args.kwargs["job_config"]
        assert isinstance(job_config, bigquery.QueryJobConfig)
        params = job_config.query_parameters
        assert len(params) == 1
        assert params[0].name == "hierids"
        assert params[0].values == ["ABW", "BHS"]


class TestGeometryContentHash:
    def test_deterministic(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        df = pd.DataFrame(
            {
                "hierid": ["ABW", "AND"],
                "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((2 2,3 2,3 3,2 3,2 2))"],
            }
        )
        backend = BigQueryBackend()
        h1 = backend._geometry_content_hash(df, "hierid")
        h2 = backend._geometry_content_hash(df, "hierid")
        assert h1 == h2

    def test_order_invariant(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        df1 = pd.DataFrame(
            {
                "hierid": ["A", "B"],
                "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((2 2,3 2,3 3,2 3,2 2))"],
            }
        )
        df2 = df1.iloc[::-1].reset_index(drop=True)
        backend = BigQueryBackend()
        assert (
            backend._geometry_content_hash(df1, "hierid")
            == backend._geometry_content_hash(df2, "hierid")
        )

    def test_changes_with_content(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        df1 = pd.DataFrame(
            {"hierid": ["A"], "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))"]}
        )
        df2 = pd.DataFrame(
            {"hierid": ["A"], "geometry": ["POLYGON((0 0,2 0,2 2,0 2,0 0))"]}
        )
        backend = BigQueryBackend()
        assert (
            backend._geometry_content_hash(df1, "hierid")
            != backend._geometry_content_hash(df2, "hierid")
        )


class TestMainSql:
    def _build(self, cfg: Config | None = None):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        cfg = cfg or _bq_cfg()
        backend = BigQueryBackend()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        temp_table = (
            "compute-impactlab.temp_workspace.segweights_geom_deadbeefdeadbeef"
        )
        sql = backend._build_main_sql(temp_table, regions, grid, weights, cfg)
        return sql, temp_table

    def test_references_temp_table_not_ir(self):
        sql, temp_table = self._build()
        assert temp_table in sql
        # IR table must NOT appear in the combined query; that's the whole
        # point of the cross-region redesign.
        assert _IR_TABLE not in sql

    def test_weight_table_appears(self):
        sql, _ = self._build()
        assert _POP_TABLE in sql

    def test_floor_cell_binning(self):
        sql, _ = self._build()
        assert "FLOOR((p.longitude - -180.0) / 1.0)" in sql
        assert "FLOOR((p.latitude - -90.0) / 1.0)" in sql

    def test_st_intersection_for_area(self):
        sql, _ = self._build()
        assert "ST_AREA(ST_INTERSECTION(region_geom, cell_geo))" in sql

    def test_st_contains_for_points(self):
        sql, _ = self._build()
        assert "ST_CONTAINS(r.geometry, ST_GEOGPOINT(p.longitude, p.latitude))" in sql

    def test_area_only_omits_points_cte(self):
        d = _bq_cfg().model_dump()
        d["weights"] = [{"name": "area"}]
        cfg = Config.model_validate(d)
        sql, _ = self._build(cfg)
        assert "points_per_cell" not in sql
        assert "ST_CONTAINS" not in sql
        assert "ST_INTERSECTION" in sql

    def test_wraps_area_cell_ix_modulo_n_ix(self):
        """The antimeridian fix: ix is wrapped modulo n_ix at area-cell
        generation. The standard positive-modulo idiom
        ``MOD(MOD(ix, n_ix) + n_ix, n_ix)`` covers negative ix too."""
        sql, _ = self._build()
        # res=1.0, [-180,180) -> n_ix=360
        assert "MOD(MOD(ix, 360) + 360, 360) AS cell_ix" in sql

    def test_caps_ix_range_to_one_belt(self):
        """Never generate more than n_ix unique wrapped cells per region."""
        sql, _ = self._build()
        # Upper bound of GENERATE_ARRAY is LEAST(ix_hi, ix_lo + n_ix - 1)
        assert "LEAST(" in sql
        assert "+ 360 - 1" in sql

    def test_dedups_wrap_collision_with_max(self):
        """ATA-style polar regions whose bbox lon-range exceeds 360 produce
        ix=0 + ix=360 -> same wrapped cell. MAX (not SUM) merges them
        without double-counting."""
        sql, _ = self._build()
        assert "MAX(ST_AREA(ST_INTERSECTION(region_geom, cell_geo)))" in sql
        assert "GROUP BY hierid, cell_ix, cell_iy" in sql

    def test_wraps_points_cell_ix_modulo_n_ix(self):
        """Points side uses the same wrap so the join key space matches."""
        sql, _ = self._build()
        # GPW longitudes in [-180,180]; FLOOR can hit 360 at lon=180 exactly.
        assert (
            "MOD(\n"
            "            MOD(CAST(FLOOR((p.longitude - -180.0) / 1.0) AS INT64), 360)\n"
            "            + 360,\n"
            "            360\n"
            "        ) AS cell_ix" in sql
        )

    def test_densifies_constant_latitude_edges(self):
        """Cell polygon's horizontal (constant-lat) edges are sampled
        with intermediate vertices so BQ's geodesic interpretation
        approximates the parallel. Stage 4 cross-backend deviation went
        from ~3e-3 to ~1e-5 with this fix."""
        sql, _ = self._build()
        # The construction uses ARRAY_CONCAT of three pieces: bottom
        # edge (ascending lon at lat_lo), top edge (descending lon at
        # lat_hi), and closing point.
        assert "ARRAY_CONCAT(" in sql
        # GENERATE_ARRAY for densification substeps; with res=1.0 and
        # default densify_step=0.1, n_substeps=10.
        assert "GENERATE_ARRAY(0, 10)" in sql
        # The substep itself (res / n_substeps) is embedded as 0.1.
        assert " * 0.1" in sql
        # Only constant-lat edges densify; meridian edges have no
        # GENERATE_ARRAY in the inner polygons.
        assert "ARRAY_TO_STRING(" in sql

    def test_densify_step_configurable(self):
        # densify_step = 0.25 with res = 1.0 -> n_substeps = 4
        d = _bq_cfg().model_dump()
        d["backend"]["bigquery"] = {
            **d["backend"].get("bigquery", {}),
            "densify_step": 0.25,
        }
        cfg = Config.model_validate(d)
        sql, _ = self._build(cfg)
        assert "GENERATE_ARRAY(0, 4)" in sql
        assert " * 0.25" in sql

    def test_clips_cell_iy_to_grid_domain(self):
        """iy doesn't wrap; clip both points binning and the area-cell
        iy range to [0, n_iy-1] for safety."""
        sql, _ = self._build()
        # In the points CTE
        assert "LEAST(180 - 1," in sql
        # In the area-cell iy range
        assert "GREATEST(0, CAST(FLOOR((rb.miny" in sql

    def test_main_sql_wraps_temp_table_with_spherical_make_valid(self):
        """Parsing and spherical repair happen in exactly ONE place: the
        regions CTE. Every downstream CTE then sees a canonical
        GEOGRAPHY value. The exact 2022 failure (planar-valid /
        spherical-invalid polygons) is repaired loudly here."""
        sql, _ = self._build()
        assert "ST_GEOGFROMTEXT(geometry, make_valid => TRUE)" in sql, (
            "main SQL must parse the temp table's geometry STRING in "
            "the spherical domain with make_valid; otherwise the 2022 "
            "spherical-invalid polygons (e.g. CAN.5 with crossing "
            "geodesic edges) fail the load"
        )

    def test_main_sql_does_not_use_safe_geogfromtext(self):
        """SAFE.ST_GEOGFROMTEXT silently turns parse failures into NULLs
        -- exactly the 2022 mechanism that produced the 17 ghost rows.
        We want loud, named errors when make_valid cannot repair."""
        sql, _ = self._build()
        assert "SAFE.ST_GEOGFROMTEXT" not in sql, (
            "main SQL uses SAFE.ST_GEOGFROMTEXT; this swallows parse "
            "failures into NULLs and reproduces the 2022 17-NULL bug"
        )

    def test_no_hierid_filter_in_main_sql(self):
        # The hierid filter was applied before the temp table; the main
        # query simply selects from the (already filtered) temp table.
        sql, _ = self._build()
        assert "@hierids" not in sql
        assert "UNNEST(@hierids)" not in sql


class TestToDataframeHelper:
    """Every download goes through `_to_dataframe`; the flag is forwarded."""

    def test_default_passes_false(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        backend = BigQueryBackend()
        cfg = _bq_cfg()
        job = MagicMock()
        backend._to_dataframe(job, cfg)
        assert (
            job.to_dataframe.call_args.kwargs["create_bqstorage_client"]
            is False
        )

    def test_override_true_forwards(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        backend = BigQueryBackend()
        d = _bq_cfg().model_dump()
        d["backend"]["bigquery"] = {
            **d["backend"].get("bigquery", {}),
            "use_bqstorage": True,
        }
        cfg = Config.model_validate(d)
        job = MagicMock()
        backend._to_dataframe(job, cfg)
        assert (
            job.to_dataframe.call_args.kwargs["create_bqstorage_client"]
            is True
        )


class TestDryRunGate:
    def test_under_ceiling_returns_bytes(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        backend = BigQueryBackend()
        client = MagicMock()
        client.query.return_value.total_bytes_processed = 5_000_000_000
        assert backend._dry_run_bytes(client, "SELECT 1", "US") == 5_000_000_000

    def test_dry_run_passes_location(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        backend = BigQueryBackend()
        client = MagicMock()
        client.query.return_value.total_bytes_processed = 0
        backend._dry_run_bytes(client, "SELECT 1", "EU")
        assert client.query.call_args.kwargs["location"] == "EU"


class TestTempTableLifecycle:
    """End-to-end of ``compute`` with every BQ call mocked.

    Verifies that the upload targets the temp dataset (not the IR
    dataset), the main query uses the resolved compute location, and the
    finally block deletes the temp table; even when the dry-run gate
    raises mid-flow.
    """

    def _mock_gcsfs(self, monkeypatch):
        gcsfs_mock = MagicMock()
        fs = MagicMock()
        gcsfs_mock.GCSFileSystem.return_value = fs
        monkeypatch.setitem(__import__("sys").modules, "gcsfs", gcsfs_mock)
        return fs

    def _full_client(
        self,
        *,
        geom_rows: list[tuple[str, str]] | None = None,
        dry_bytes: int = 1_000_000_000,
        result_rows: list[tuple[Any, ...]] | None = None,
        null_hierids: list[str] | None = None,
        unknown_hierids: list[str] | None = None,
    ):
        client = _client_with_locations()

        if geom_rows is None:
            geom_rows = [("ABW", "POLYGON((-70 12,-69 12,-69 13,-70 13,-70 12))")]
        if result_rows is None:
            result_rows = [
                ("ABW", 109, 102, -70.5, 12.5, 1_000_000.0, 100.0),
            ]
        null_hierids = null_hierids or []
        unknown_hierids = unknown_hierids or []

        def _query(sql, job_config=None, location=None):
            job = MagicMock()
            if "is_unknown" in sql or "WITH requested" in sql:
                rows = (
                    [(h, True, False) for h in unknown_hierids]
                    + [(h, False, True) for h in null_hierids]
                )
                job.to_dataframe.return_value = pd.DataFrame(
                    rows, columns=["hierid", "is_unknown", "is_null"]
                )
            elif "ST_ASTEXT" in sql:
                job.to_dataframe.return_value = pd.DataFrame(
                    geom_rows, columns=["hierid", "geometry"]
                )
            elif job_config is not None and getattr(job_config, "dry_run", False):
                job.total_bytes_processed = dry_bytes
            else:
                job.to_dataframe.return_value = pd.DataFrame(
                    result_rows,
                    columns=[
                        "hierid",
                        "cell_ix",
                        "cell_iy",
                        "cell_lon",
                        "cell_lat",
                        "area_raw",
                        "pop_raw",
                    ],
                )
            return job

        client.query.side_effect = _query
        # get_table for the temp-table-exists check returns NotFound (cache miss)
        client.get_table.side_effect = Exception("not found")
        # After upload, get_table is called once more to set expiration; let
        # the second call return a mock table. Use a counter.
        call_count = {"n": 0}

        def _get_table(table_id: str):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("not found")
            t = MagicMock()
            t.expires = None
            return t

        client.get_table.side_effect = _get_table
        return client

    def test_load_job_uses_string_schema_for_geometry(self, monkeypatch):
        """Temp table column type for geometry is STRING, NOT GEOGRAPHY.
        Parsing is deferred to SQL via ST_GEOGFROMTEXT(make_valid=>TRUE)
        so the load can never fail on geometry content (the 2022 failure
        mode swallowed by SAFE.; we now repair loudly in the spherical
        domain at parse time)."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client()
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        backend.compute(regions, grid, weights, cfg)

        load_call = client.load_table_from_uri.call_args
        job_config = load_call.kwargs["job_config"]
        geometry_fields = [f for f in job_config.schema if f.name == "geometry"]
        assert len(geometry_fields) == 1
        assert geometry_fields[0].field_type == "STRING", (
            "geometry column must load as STRING; GEOGRAPHY would attempt "
            "spherical parsing at load time and fail (deterministic 500 "
            "on the 2022 NULL geometries that planar make_valid did not "
            "repair)"
        )

    def test_temp_table_name_carries_v2_marker(self, monkeypatch):
        """The schema/SQL change makes old GEOGRAPHY-schema temp tables
        non-reusable; the table-name prefix bump invalidates the cache."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client()
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        backend.compute(regions, grid, weights, cfg)

        load_dest = client.load_table_from_uri.call_args.args[1]
        assert "segweights_geom_v2_" in load_dest, (
            f"temp table name {load_dest!r} does not carry the v2 marker; "
            "old GEOGRAPHY-schema tables would be wrongly reused"
        )

    def test_load_job_uses_parquet_source_format(self, monkeypatch):
        """Geometry staging is PARQUET, not JSONL. JSONL with multi-MB
        WKT rows triggered deterministic BQ 500 internalError; Parquet
        handles the payload natively. Geometries are NEVER simplified
        for the load."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client()
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        backend.compute(regions, grid, weights, cfg)

        load_call = client.load_table_from_uri.call_args
        # Source URI should be the .parquet object we wrote.
        assert load_call.args[0].endswith(".parquet"), (
            f"staging URI is not parquet: {load_call.args[0]}"
        )
        # LoadJobConfig.source_format should be PARQUET.
        job_config = load_call.kwargs["job_config"]
        assert job_config.source_format == backend._bq.SourceFormat.PARQUET, (
            "geometry staging must use PARQUET source_format; JSONL was "
            "the source of the deterministic 500 internalError"
        )

    def test_load_job_targets_temp_workspace(self, monkeypatch):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client()
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        monkeypatch.setattr(backend, "_bq", backend._bq)

        backend.compute(regions, grid, weights, cfg)

        load_call = client.load_table_from_uri.call_args
        assert load_call is not None
        dest_table = load_call.args[1]
        assert "compute-impactlab.temp_workspace.segweights_geom_" in dest_table
        assert load_call.kwargs["location"] == "US"

    def test_main_query_uses_compute_location(self, monkeypatch):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client()
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        backend.compute(regions, grid, weights, cfg)

        # The last `client.query` call is the real one with the combined SQL
        # (and the prior call was the geometry fetch / dry-run; both have
        # location set).
        for call in client.query.call_args_list:
            sql = call.args[0]
            if "ST_INTERSECTION" in sql and "cell_areas" in sql:
                # combined query
                assert call.kwargs["location"] == "US"
                return
        pytest.fail("combined query was not issued")

    def test_cleanup_runs_on_success(self, monkeypatch):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client()
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        result = backend.compute(regions, grid, weights, cfg)

        client.delete_table.assert_called_once()
        delete_table_id = client.delete_table.call_args.args[0]
        assert "compute-impactlab.temp_workspace.segweights_geom_" in delete_table_id
        # Manifest records the download path so future runs can diff
        # against it if the readSessionUser role is ever granted.
        assert result.manifest.extra["bq_used_bqstorage"] is False
        # Spherical repair is always applied at SQL-parse time.
        assert result.manifest.extra["spherical_make_valid"] is True

    def test_cleanup_runs_on_dry_run_failure(self, monkeypatch):
        """If the dry-run gate raises, the finally block still deletes."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client(dry_bytes=20_000_000_000)  # > 10 GB ceiling
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        with pytest.raises(RuntimeError, match="dry-run estimates"):
            backend.compute(regions, grid, weights, cfg)
        client.delete_table.assert_called_once()

    def test_cache_temp_tables_skips_cleanup(self, monkeypatch):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        self._mock_gcsfs(monkeypatch)
        client = self._full_client()
        d = _bq_cfg().model_dump()
        d["backend"]["bigquery"] = {
            **d["backend"].get("bigquery", {}),
            "cache_temp_tables": True,
        }
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)

        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)
        backend.compute(regions, grid, weights, cfg)
        client.delete_table.assert_not_called()


class TestShapefileSourcePath:
    """The supplement run feeds a shapefile / geoparquet through the BQ
    backend (Stage 6). Verifies: no IR table query is issued; pre-check
    runs against the in-memory gdf; geometries land as WKT in the temp
    upload; manifest records the source uri + any geometry repairs."""

    def _gdf_parquet(self, tmp_path, *, with_invalid: bool = False):
        import geopandas as gpd
        from shapely.geometry import Polygon, box

        rows = [
            ("ABW", box(0.0, 0.0, 1.0, 1.0)),
            ("CAN.5", box(2.0, 2.0, 3.0, 3.0)),
        ]
        if with_invalid:
            bowtie = Polygon([(5, 5), (6, 6), (6, 5), (5, 6), (5, 5)])
            rows.append(("BOWTIE", bowtie))
        gdf = gpd.GeoDataFrame(
            {"hierid": [r[0] for r in rows], "geometry": [r[1] for r in rows]},
            crs="EPSG:4326",
        )
        p = tmp_path / "regions.parquet"
        gdf.to_parquet(p)
        return p

    def _cfg(self, regions_path, *, keep=None):
        d = _bq_cfg().model_dump()
        d["regions"] = {
            "path": str(regions_path),
            "id_fields": ["hierid"],
            "on_invalid_geometry": "repair",
            "on_null_geometry": "error",
            "on_unknown_id": "error",
            "keep": keep,
        }
        return Config.model_validate(d)

    def test_fetch_geometries_skips_ir_query(self, tmp_path):
        """With a path-loaded gdf, no SQL is issued to fetch geometries."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.regions import RegionSet
        from unittest.mock import MagicMock

        cfg = self._cfg(self._gdf_parquet(tmp_path))
        regions = RegionSet.from_config(cfg.regions)
        client = MagicMock()  # no .query call expected
        df = BigQueryBackend()._fetch_geometries(
            client, regions, None, "us-west1", cfg
        )
        client.query.assert_not_called()
        assert set(df.columns) == {"hierid", "geometry"}
        assert set(df["hierid"]) == {"ABW", "CAN.5"}
        for wkt_str in df["geometry"]:
            assert wkt_str.startswith("POLYGON")

    def test_pre_check_from_gdf_detects_unknown_and_null(self, tmp_path):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.regions import RegionSet

        cfg = self._cfg(
            self._gdf_parquet(tmp_path),
            keep={"hierid": ["ABW", "NOT_REAL"]},
        )
        # Force the policy to skip both so we get the lists back.
        d = cfg.model_dump()
        d["regions"]["on_unknown_id"] = "skip"
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        null_ids, unknown_ids = BigQueryBackend()._pre_check_from_gdf(
            regions, ["ABW", "NOT_REAL"], "hierid"
        )
        assert unknown_ids == ["NOT_REAL"]
        assert null_ids == []

    def test_repair_log_surfaces_to_manifest(self, tmp_path, monkeypatch):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        TestTempTableLifecycle()._mock_gcsfs(monkeypatch)
        client = TestTempTableLifecycle()._full_client()
        backend = BigQueryBackend()
        backend._bq.Client = MagicMock(return_value=client)

        cfg = self._cfg(self._gdf_parquet(tmp_path, with_invalid=True))
        regions = RegionSet.from_config(cfg.regions)
        # The bow-tie was repaired by RegionSet under the default
        # 'repair' policy; the gdf is now valid.
        assert all(regions.gdf.geometry.is_valid)
        assert any(r["hierid"] == "BOWTIE" for r in regions.repaired_ids)

        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = backend.compute(regions, grid, weights, cfg)
        repaired = result.manifest.extra["repaired_geometry_regions"]
        assert any(r["hierid"] == "BOWTIE" for r in repaired)
        # manifest.inputs records the source URI but NOT regions_table
        assert "regions_path" in result.manifest.inputs
        assert "regions_table" not in result.manifest.inputs

    def test_resolve_locations_skips_ir(self, tmp_path):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        cfg = self._cfg(self._gdf_parquet(tmp_path))
        regions = RegionSet.from_config(cfg.regions)
        client = _client_with_locations()
        ir_loc, compute_loc = BigQueryBackend()._resolve_locations(
            client, regions, [w for w in from_config_list(cfg.weights) if not w.is_area], cfg
        )
        assert ir_loc is None
        assert compute_loc == "US"


class TestLoadRetry:
    """The supplement run surfaced deterministic BigQuery 500
    internalError on the geometry load job. The shared upload path now
    (a) stages Parquet rather than JSONL, and (b) retries transient
    failures with bounded exponential backoff, logging the BigQuery
    job_id on every failure so a persistent issue can be escalated to
    support with a concrete reference."""

    def _backend_with_no_real_sleep(self, monkeypatch):
        from segment_weights.backends import bigquery as bq_module
        from segment_weights.backends.bigquery import BigQueryBackend

        # Strip the actual sleep so retry-loop tests run fast.
        monkeypatch.setattr(bq_module.time, "sleep", lambda s: None)
        return BigQueryBackend()

    def test_succeeds_on_third_attempt(self, monkeypatch, caplog):
        """First two calls raise InternalServerError, third succeeds.
        The call is made via `_run_load_with_retry` directly so we
        isolate the retry logic from the broader upload."""
        from google.api_core.exceptions import InternalServerError

        backend = self._backend_with_no_real_sleep(monkeypatch)
        client = MagicMock()

        failing = MagicMock()
        failing.job_id = "abc123_failed"
        failing.result.side_effect = InternalServerError("simulated 500")

        succeeding = MagicMock()
        succeeding.job_id = "abc123_ok"

        client.load_table_from_uri.side_effect = [
            failing,
            failing,
            succeeding,
        ]

        with caplog.at_level("WARNING", logger="segment_weights.backends.bigquery"):
            backend._run_load_with_retry(
                client,
                "gs://x/y.parquet",
                "dataset.table",
                MagicMock(),
                "US",
            )

        assert client.load_table_from_uri.call_count == 3
        # job_id is logged on each retryable failure so persistent
        # service-side issues are escalatable.
        warn_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("abc123_failed" in r.message for r in warn_records)
        assert any("retrying after" in r.message for r in warn_records)

    def test_gives_up_after_max_attempts(self, monkeypatch, caplog):
        from google.api_core.exceptions import InternalServerError

        backend = self._backend_with_no_real_sleep(monkeypatch)
        client = MagicMock()
        failing = MagicMock()
        failing.job_id = "persistent_500"
        failing.result.side_effect = InternalServerError("simulated 500")
        client.load_table_from_uri.return_value = failing

        with caplog.at_level("ERROR", logger="segment_weights.backends.bigquery"):
            with pytest.raises(InternalServerError):
                backend._run_load_with_retry(
                    client,
                    "gs://x/y.parquet",
                    "dataset.table",
                    MagicMock(),
                    "US",
                )

        # All three attempts made.
        assert client.load_table_from_uri.call_count == 3
        # Error log names the persistent job_id for support escalation.
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any("persistent_500" in r.message for r in error_records)
        assert any("Escalate with the job_id" in r.message for r in error_records)

    def test_does_not_retry_on_schema_error(self, monkeypatch):
        """Non-transient failures fail fast; retrying won't help."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from google.api_core.exceptions import BadRequest

        backend = self._backend_with_no_real_sleep(monkeypatch)
        client = MagicMock()
        failing = MagicMock()
        failing.job_id = "bad_request_job"
        failing.result.side_effect = BadRequest("schema mismatch")
        client.load_table_from_uri.return_value = failing

        with pytest.raises(BadRequest):
            backend._run_load_with_retry(
                client,
                "gs://x/y.parquet",
                "dataset.table",
                MagicMock(),
                "US",
            )
        # Only one attempt for non-retryable failures.
        assert client.load_table_from_uri.call_count == 1

    def test_upload_logs_staging_size_and_max_wkt(
        self, monkeypatch, caplog
    ):
        """Pre-load logs name the staging size, row count, and max
        per-row WKT length so support escalations have a concrete
        payload reference."""
        from segment_weights.backends.bigquery import BigQueryBackend

        # Stub gcsfs and pretend the load succeeds on the first try.
        TestTempTableLifecycle()._mock_gcsfs(monkeypatch)
        client = MagicMock()
        ok_job = MagicMock()
        ok_job.job_id = "ok"
        client.load_table_from_uri.return_value = ok_job
        # get_table for the expiration backstop.
        client.get_table.return_value = MagicMock(expires=None)

        backend = BigQueryBackend()
        cfg = _bq_cfg()
        geom_df = pd.DataFrame(
            {
                "hierid": ["ABW", "BIG"],
                # 100 kB WKT to simulate a large polygon.
                "geometry": [
                    "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                    "POLYGON((" + ",".join(f"{i / 100} 0" for i in range(2500)) + "))",
                ],
            }
        )
        with caplog.at_level("INFO", logger="segment_weights.backends.bigquery"):
            backend._upload_temp_table(
                client, geom_df, "temp_workspace.tbl", "hierid", cfg, "US"
            )
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert any("max_wkt_bytes=" in r.message for r in info_records)
        assert any("2 rows" in r.message for r in info_records)


class TestUnknownIdHandling:
    """`regions.keep` with ids that don't exist in the table; a request
    bug, not a data drift. Hardcoded `AND` / `BMU` in legacy configs
    survived migration to the clustered IR; the runner caught them via
    the coverage assertion. The pre-check surfaces them earlier."""

    def _backend(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        return BigQueryBackend()

    def test_default_policy_errors_listing_ids(self, monkeypatch):
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        TestTempTableLifecycle()._mock_gcsfs(monkeypatch)
        client = TestTempTableLifecycle()._full_client(
            unknown_hierids=["AND", "BMU"]
        )
        backend = self._backend()
        backend._bq.Client = MagicMock(return_value=client)
        cfg = _bq_cfg()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        with pytest.raises(ValueError) as exc:
            backend.compute(regions, grid, weights, cfg)
        msg = str(exc.value)
        assert "do not exist" in msg
        assert "on_unknown_id='error'" in msg
        assert "AND" in msg
        assert "BMU" in msg

    def test_skip_policy_records_to_manifest(self, monkeypatch):
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        TestTempTableLifecycle()._mock_gcsfs(monkeypatch)
        client = TestTempTableLifecycle()._full_client(
            unknown_hierids=["AND"]
        )
        backend = self._backend()
        backend._bq.Client = MagicMock(return_value=client)
        d = _bq_cfg().model_dump()
        d["regions"]["on_unknown_id"] = "skip"
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = backend.compute(regions, grid, weights, cfg)
        assert result.manifest.extra["unknown_id_count"] == 1
        assert result.manifest.extra["unknown_id_regions"] == ["AND"]

    def test_both_null_and_unknown_skipped_together(self, monkeypatch):
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        TestTempTableLifecycle()._mock_gcsfs(monkeypatch)
        client = TestTempTableLifecycle()._full_client(
            null_hierids=["CAN.5"], unknown_hierids=["AND"]
        )
        backend = self._backend()
        backend._bq.Client = MagicMock(return_value=client)
        d = _bq_cfg().model_dump()
        d["regions"]["on_null_geometry"] = "skip"
        d["regions"]["on_unknown_id"] = "skip"
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = backend.compute(regions, grid, weights, cfg)
        assert result.manifest.extra["null_geometry_regions"] == ["CAN.5"]
        assert result.manifest.extra["unknown_id_regions"] == ["AND"]


class TestNullGeometryHandling:
    """The s51 IR table inherits NULL geometries from the legacy pipeline's
    SAFE.ST_GEOGFROMTEXT. The backend must pre-check before the fetch and
    honor `regions.on_null_geometry` ('error' default, 'skip' for s51)."""

    def _backend(self):
        from segment_weights.backends.bigquery import BigQueryBackend

        return BigQueryBackend()

    def _specs(self, cfg: Config):
        from segment_weights.regions import RegionSet

        return RegionSet.from_config(cfg.regions)

    def test_default_policy_errors_listing_hierids(self, monkeypatch):
        from segment_weights.grid import GridSpec
        from segment_weights.weights import from_config_list

        TestTempTableLifecycle()._mock_gcsfs(monkeypatch)
        client = TestTempTableLifecycle()._full_client(
            null_hierids=["CAN.5", "DEU.2.10.Re83c19217082ae1b"]
        )
        backend = self._backend()
        backend._bq.Client = MagicMock(return_value=client)
        cfg = _bq_cfg()
        regions = self._specs(cfg)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        with pytest.raises(ValueError) as exc:
            backend.compute(regions, grid, weights, cfg)
        msg = str(exc.value)
        assert "NULL geometries" in msg
        assert "on_null_geometry='error'" in msg
        assert "CAN.5" in msg
        assert "DEU.2.10.Re83c19217082ae1b" in msg

    def test_skip_policy_records_to_manifest(self, monkeypatch):
        from segment_weights.grid import GridSpec
        from segment_weights.weights import from_config_list

        TestTempTableLifecycle()._mock_gcsfs(monkeypatch)
        client = TestTempTableLifecycle()._full_client(
            null_hierids=["CAN.5"]
        )
        backend = self._backend()
        backend._bq.Client = MagicMock(return_value=client)
        d = _bq_cfg().model_dump()
        d["regions"]["on_null_geometry"] = "skip"
        cfg = Config.model_validate(d)
        regions = self._specs(cfg)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = backend.compute(regions, grid, weights, cfg)
        assert result.manifest.extra["null_geometry_count"] == 1
        assert result.manifest.extra["null_geometry_regions"] == ["CAN.5"]

    def test_fetch_sql_filters_null_geometry(self):
        from segment_weights.grid import GridSpec
        from segment_weights.weights import from_config_list

        cfg = _bq_cfg()
        regions = self._specs(cfg)
        client = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame(
            {"hierid": ["ABW"], "geometry": ["POLYGON((0 0,1 0,1 1,0 1,0 0))"]}
        )
        self._backend()._fetch_geometries(
            client, regions, ["ABW"], "us-west1", cfg
        )
        sql = client.query.call_args[0][0]
        assert "geometry IS NOT NULL" in sql

    def test_upload_raises_if_null_slips_through(self, monkeypatch, tmp_path):
        """Defense in depth: if a NULL geometry reaches the staging writer
        (upstream filter broken), fail with the offending hierid rather
        than the cryptic load-job BadRequest."""
        from segment_weights.backends.bigquery import BigQueryBackend

        gcsfs_mock = MagicMock()
        fs = MagicMock()
        gcsfs_mock.GCSFileSystem.return_value = fs
        monkeypatch.setitem(
            __import__("sys").modules, "gcsfs", gcsfs_mock
        )

        backend = BigQueryBackend()
        cfg = _bq_cfg()
        client = MagicMock()
        # The geom_df includes a NULL row that the upstream check would
        # normally have filtered out. The defense should catch it.
        geom_df = pd.DataFrame(
            {
                "hierid": ["ABW", "BAD"],
                "geometry": [
                    "POLYGON((0 0,1 0,1 1,0 1,0 0))",
                    None,
                ],
            }
        )
        with pytest.raises(ValueError, match="NULL geometry after the fetch"):
            backend._upload_temp_table(
                client, geom_df, "tbl_id", "hierid", cfg, "US"
            )


class TestAssembleResult:
    def _downloaded(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("X", 5, 5, 5.5, 5.5, 50.0, 30.0),
                ("X", 5, 6, 5.5, 6.5, 30.0, 10.0),
                ("Y", 10, 10, 10.5, 10.5, 80.0, 0.0),
                ("Y", 10, 11, 10.5, 11.5, 40.0, 0.0),
            ],
            columns=[
                "hierid",
                "cell_ix",
                "cell_iy",
                "cell_lon",
                "cell_lat",
                "area_raw",
                "pop_raw",
            ],
        )

    def test_area_fallback_marks_zero_pop_region(self):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        cfg = _bq_cfg()
        backend = BigQueryBackend()
        weights = from_config_list(cfg.weights)
        non_area = [w for w in weights if not w.is_area]
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        downloaded = self._downloaded()
        # Build a matching geom_df so nearest_cell synthesis has the
        # geometries for any (intentionally) missing hierid; the downloaded
        # frame already covers X and Y, so the geom_df should too; same
        # set, no missing rows.
        geom_df = pd.DataFrame(
            {
                "hierid": ["X", "Y"],
                "geometry": [
                    "POLYGON((5 5,6 5,6 7,5 7,5 5))",
                    "POLYGON((10 10,11 10,11 12,10 12,10 10))",
                ],
            }
        )
        result = backend._assemble_result(
            downloaded, ["hierid"], weights, non_area, cfg, regions,
            geom_df, grid,
        )
        assert result.sum_report.ok
        y_rows = result.frame.loc[result.frame["hierid"] == "Y"]
        assert (y_rows["pop_method"] == "area_fallback").all()
        assert y_rows["popwt"].sum() == pytest.approx(1.0)
        x_rows = result.frame.loc[result.frame["hierid"] == "X"]
        assert (x_rows["pop_method"] == "native").all()
        assert result.manifest.fallback_counts["pop"] == {
            "native": 1,
            "area_fallback": 1,
        }


class TestConfigRejections:
    def test_rejects_exact_fraction(self):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        d = _bq_cfg().model_dump()
        d["backend"]["coverage"] = "exact_fraction"
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        with pytest.raises(ValueError, match="pixel_centroid"):
            BigQueryBackend().compute(regions, grid, weights, cfg)

    def test_accepts_regions_path_shapefile_source(self, tmp_path):
        """The supplement run feeds a shapefile / geoparquet through the
        BQ pipeline (Stage 6). The BQ backend used to refuse `regions.path`;
        it now treats it as a valid alternative geometry source."""
        from segment_weights.regions import RegionSet
        import geopandas as gpd
        from shapely.geometry import box

        p = tmp_path / "r.parquet"
        gpd.GeoDataFrame(
            {"hierid": ["A"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326"
        ).to_parquet(p)

        d = _bq_cfg().model_dump()
        d["regions"] = {"path": str(p), "id_fields": ["hierid"]}
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        assert regions.gdf is not None
        assert regions.table is None
        assert regions.source_uri is not None


# --------------------------------------------------------------------------
# Real-backend tests (skipped by default; require bigquery + ADC)
# --------------------------------------------------------------------------


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


@pytest.mark.bigquery
class TestRealBackend:
    """Tiny end-to-end runs on cilresearch."""

    def _cfg(self, tmp_path):
        # Read the keep list from the canonical s51_test.toml via
        # conftest. Hardcoded lists drifted twice in this codebase
        # (AND/BMU don't exist bare in world-combo-2017; the TOML
        # has the documentation block on the naming patterns).
        from tests.conftest import S51_TEST_HIERIDS

        d = _bq_cfg().model_dump()
        d["regions"]["keep"] = {"hierid": list(S51_TEST_HIERIDS)}
        d["regions"]["on_null_geometry"] = "skip"
        d["regions"]["on_unknown_id"] = "skip"
        d["output"]["dir"] = str(tmp_path)
        return Config.model_validate(d)

    def test_dry_run_under_ceiling(self, bq_client, tmp_path):
        """The combined query scans GPW (~6 GB); 10 GB ceiling admits it."""
        from segment_weights.backends.bigquery import BigQueryBackend

        backend = BigQueryBackend()
        cfg = self._cfg(tmp_path)
        # We can't easily isolate the dry-run path without running the upload,
        # so go end-to-end and assert the manifest captured a sub-ceiling cost.
        result = backend.compute(
            __import__("segment_weights").regions.RegionSet.from_config(cfg.regions),
            __import__("segment_weights").grid.GridSpec.from_config(cfg.grid),
            __import__("segment_weights").weights.from_config_list(cfg.weights),
            cfg,
        )
        assert (
            result.manifest.extra["bq_dry_run_bytes"]
            < cfg.backend.bigquery.dry_run_byte_ceiling
        )

    def test_tiny_run_sum_to_one(self, bq_client, tmp_path):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        cfg = self._cfg(tmp_path)
        backend = BigQueryBackend()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = backend.compute(regions, grid, weights, cfg)
        assert len(result.frame) > 0
        assert result.sum_report.ok, result.sum_report.summary()
        assert "bq_dry_run_bytes" in result.manifest.extra
        assert "bq_temp_table" in result.manifest.extra
        assert result.manifest.row_counts["regions"] >= 1

    def test_smallest_hierids_all_covered(self, bq_client, tmp_path):
        """The 5 smallest-area IR hierids must all appear in the output;
        either as native/area_fallback rows OR as nearest_cell synthesis.

        At 1deg resolution, even tiny island hierids usually have positive
        ST_INTERSECTION with at least one cell, so nearest_cell may or may
        not fire. The hard invariant is **no drops**: every requested
        hierid must be in the result. This test exercises that invariant
        against real data, which is what the user's Stage 5 spec asked for.
        """
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        # Find the 5 smallest hierids cheaply. The IR table is 0.88 GB.
        # WHERE geometry IS NOT NULL; without it NULL geometries sort first
        # in ASC order and yield the cryptic load-job BadRequest. The
        # separate test below covers the NULL-handling path explicitly.
        find_sql = (
            f"SELECT hierid FROM `{_IR_TABLE}` "
            f"WHERE geometry IS NOT NULL "
            f"ORDER BY ST_AREA(geometry) ASC LIMIT 5"
        )
        candidates = [
            row.hierid
            for row in bq_client.query(find_sql, location="us-west1").result()
        ]
        assert len(candidates) == 5

        d = _bq_cfg().model_dump()
        d["regions"]["keep"] = {"hierid": candidates}
        d["output"]["dir"] = str(tmp_path)
        cfg = Config.model_validate(d)

        backend = BigQueryBackend()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = backend.compute(regions, grid, weights, cfg)

        returned = set(result.frame["hierid"].unique())
        # Coverage: returned + null + unknown == requested. The smallest
        # query already filters NULL geometry; smallest valid hierids
        # cannot be unknown either (they came from the table).
        n_null = result.manifest.extra.get("null_geometry_count", 0)
        n_unknown = result.manifest.extra.get("unknown_id_count", 0)
        assert n_null == 0
        assert n_unknown == 0
        assert returned == set(candidates), (
            f"smallest-area hierids dropped from result: "
            f"{set(candidates) - returned}. nearest_cell should cover "
            f"every valid hierid."
        )
        assert result.sum_report.ok, result.sum_report.summary()
        nc = result.manifest.row_counts.get("nearest_cell", 0)
        # If any hierids did fall through to synthesis, the manifest counts
        # match the per-weight method tallies.
        if nc > 0:
            for w in ("pop", "area"):
                assert (
                    result.manifest.fallback_counts[w].get("nearest_cell", 0)
                    == nc
                )

    def test_null_geometry_error_default(self, bq_client, tmp_path):
        """A known NULL hierid in the keep list aborts under the default
        policy with the hierid named in the error message."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        # CAN.5 is one of the 17 NULL hierids the user measured. Include
        # ABW (valid) so the failure must be NULL-driven, not "no hierids".
        d = _bq_cfg().model_dump()
        d["regions"]["keep"] = {"hierid": ["ABW", "CAN.5"]}
        d["output"]["dir"] = str(tmp_path)
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        with pytest.raises(ValueError) as exc:
            BigQueryBackend().compute(regions, grid, weights, cfg)
        msg = str(exc.value)
        assert "NULL geometries" in msg
        assert "CAN.5" in msg

    def test_unknown_id_error_default(self, bq_client, tmp_path):
        """A clearly nonsensical id in keep aborts under default policy
        with the id named in the error message."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        d = _bq_cfg().model_dump()
        d["regions"]["keep"] = {"hierid": ["ABW", "NOT_A_REGION"]}
        d["output"]["dir"] = str(tmp_path)
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        with pytest.raises(ValueError) as exc:
            BigQueryBackend().compute(regions, grid, weights, cfg)
        msg = str(exc.value)
        assert "do not exist" in msg
        assert "NOT_A_REGION" in msg

    def test_antimeridian_crosser_wraps_correctly(self, bq_client, tmp_path):
        """USA.2.69 (Aleutians) and FJI.2.7 cross the antimeridian. In the
        legacy SQL their unwrapped area cells (ix >= 360) never joined
        with the wrapped GPW points (ix < 360), and populated cells came
        back with popwt=0. The fixed SQL must produce:

            (a) every cell_lon within [-180, 180),
            (b) no duplicate (hierid, cell_ix, cell_iy) keys,
            (c) at least one cell with popwt > 0 for USA.2.69 (Aleutians
                are populated, ~8k people).
        """
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        d = _bq_cfg().model_dump()
        d["regions"]["keep"] = {"hierid": ["USA.2.69"]}
        d["regions"]["on_unknown_id"] = "skip"
        d["regions"]["on_null_geometry"] = "skip"
        d["output"]["dir"] = str(tmp_path)
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = BigQueryBackend().compute(regions, grid, weights, cfg)

        # (a) all cell_lon within domain
        assert result.frame["cell_lon"].between(-180.0, 179.999999).all(), (
            "cell_lon outside [-180, 180); antimeridian wrap not applied"
        )
        # (b) no duplicate keys
        key = ["hierid", "cell_ix", "cell_iy"]
        assert not result.frame.duplicated(subset=key).any(), (
            "duplicate (region, cell_ix, cell_iy); wrap collision not deduped"
        )
        # (c) at least one cell with popwt > 0; the bug surfaced as
        # native pop_method but popwt exactly 0 on every eastern cell
        usa = result.frame.loc[result.frame["hierid"] == "USA.2.69"]
        assert (usa["popwt"] > 0).any(), (
            "USA.2.69 has no cell with popwt > 0; pop east of dateline lost "
            "to the wrap-unaware key mismatch"
        )

    def test_unknown_id_skip_records_to_manifest(self, bq_client, tmp_path):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        d = _bq_cfg().model_dump()
        d["regions"]["keep"] = {"hierid": ["ABW", "NOT_A_REGION"]}
        d["regions"]["on_unknown_id"] = "skip"
        d["output"]["dir"] = str(tmp_path)
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = BigQueryBackend().compute(regions, grid, weights, cfg)
        assert "NOT_A_REGION" in result.manifest.extra["unknown_id_regions"]
        n_returned = result.manifest.row_counts["regions"]
        n_null = result.manifest.extra["null_geometry_count"]
        n_unknown = result.manifest.extra["unknown_id_count"]
        assert n_returned + n_null + n_unknown == len(d["regions"]["keep"]["hierid"])

    def test_null_geometry_skip_records_to_manifest(self, bq_client, tmp_path):
        """Skip policy excludes NULL hierids and records them in the
        manifest; coverage of the requested set = output + null_skipped."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        d = _bq_cfg().model_dump()
        d["regions"]["keep"] = {"hierid": ["ABW", "CAN.5"]}
        d["regions"]["on_null_geometry"] = "skip"
        d["output"]["dir"] = str(tmp_path)
        cfg = Config.model_validate(d)
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = BigQueryBackend().compute(regions, grid, weights, cfg)
        assert "CAN.5" in result.manifest.extra["null_geometry_regions"]
        assert result.manifest.extra["null_geometry_count"] >= 1
        returned = set(result.frame["hierid"].unique())
        # ABW must be in the output; CAN.5 must not.
        assert "ABW" in returned
        assert "CAN.5" not in returned
        # Coverage: returned + null_skipped + unknown_skipped == requested
        n_returned = result.manifest.row_counts["regions"]
        n_null = result.manifest.extra["null_geometry_count"]
        n_unknown = result.manifest.extra["unknown_id_count"]
        assert n_returned + n_null + n_unknown == len(d["regions"]["keep"]["hierid"])

    def test_can5_spherical_invalid_supplement_end_to_end(
        self, bq_client, tmp_path
    ):
        """CAN.5 (Newfoundland; ~20 MB WKT) is the proven spherical-
        invalid case from the 2022 SAFE.ST_GEOGFROMTEXT swallow event.
        With GEOGRAPHY-schema staging it produced BQ 500; with planar-
        only repair it produced BadRequest "Edge X crosses edge Y" at
        load. With STRING-schema staging + ST_GEOGFROMTEXT(make_valid
        =>TRUE) the load succeeds, parsing repairs the spherical-
        invalid edges, and the result passes sum-to-1 + grid
        invariants."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        cfg = Config.model_validate(
            {
                "project": {"name": "can5_spherical_repair"},
                "regions": {
                    "path": (
                        "gs://impactlab-data/spatial/shapefiles/source/"
                        "impactlab/world-combo-new/agglomerated-world-new.shp"
                    ),
                    "id_fields": ["hierid"],
                    "keep": {"hierid": ["CAN.5"]},
                    "on_invalid_geometry": "repair",
                    "on_null_geometry": "error",
                    "on_unknown_id": "error",
                },
                "grid": {
                    "mode": "generate",
                    "resolution": 1.0,
                    "offset": "center",
                    "lon_convention": "[-180,180)",
                },
                "weights": [
                    {"name": "pop", "table": _POP_TABLE, "fallback": "area"},
                    {"name": "area"},
                ],
                "backend": {
                    "kind": "bigquery",
                    "coverage": "pixel_centroid",
                },
                "output": {"dir": str(tmp_path), "confirm_cost": False},
            }
        )
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = BigQueryBackend().compute(regions, grid, weights, cfg)

        # Load + parse + repair must succeed.
        assert "CAN.5" in set(result.frame["hierid"].unique())
        # Sum-to-1 holds.
        assert result.sum_report.ok, result.sum_report.summary()
        # Grid invariants hold (raised inside compute if they didn't).
        # Manifest records the spherical repair pass.
        assert result.manifest.extra["spherical_make_valid"] is True
        # Newfoundland spans many cells.
        assert int(result.manifest.row_counts["total"]) >= 5

    def test_dry_run_public_api(self, bq_client, tmp_path):
        """`dry_run()` is a public method runners can use for the
        interactive ``confirm before running'' pattern."""
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list

        cfg = self._cfg(tmp_path)
        backend = BigQueryBackend()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        info = backend.dry_run(regions, grid, weights, cfg)
        assert info["bytes_estimate"] > 0
        assert info["compute_location"] == "US"
        assert "temp_workspace.segweights_geom_" in info["temp_table"]
        assert info["bytes_estimate"] < cfg.backend.bigquery.dry_run_byte_ceiling

    def test_temp_table_cleaned_up(self, bq_client, tmp_path):
        from segment_weights.backends.bigquery import BigQueryBackend
        from segment_weights.grid import GridSpec
        from segment_weights.regions import RegionSet
        from segment_weights.weights import from_config_list
        from google.api_core.exceptions import NotFound

        cfg = self._cfg(tmp_path)
        backend = BigQueryBackend()
        regions = RegionSet.from_config(cfg.regions)
        grid = GridSpec.from_config(cfg.grid)
        weights = from_config_list(cfg.weights)
        result = backend.compute(regions, grid, weights, cfg)

        temp_table = result.manifest.extra["bq_temp_table"]
        with pytest.raises(NotFound):
            bq_client.get_table(temp_table)
