"""The BigQuery backend, which computes grid weights in one BigQuery
query and applies the fallback in Python to the downloaded rows.

BigQuery cannot join tables in different locations, and the regions table
may sit apart from the weight tables. The backend reads the requested
geometries in the regions table's location, or from a regions file, and
loads them as WKT through a Parquet file on GCS into a temp table in the
temp dataset. The temp dataset and every weight table must share a
location. Raises, naming them and their locations, if they do not. No
dataset is ever created. The temp table's name carries a hash of the
geometries, so a run with the same geometries reuses a cached table. The
table gets an expiry and is deleted after the run unless
``backend.bigquery.cache_temp_tables`` is set.

The combined query is dry-run first. Raises if the estimate exceeds
``backend.bigquery.dry_run_byte_ceiling``. The query's point-in-polygon
join scans the whole weight table however few regions are requested, so
a small run still estimates about a full scan.

BigQuery takes a polygon edge as a geodesic, so the top and bottom edges
of a cell would bow off their parallels. The query adds vertices along
them at most ``backend.bigquery.densify_step`` degrees apart. What area
difference remains against the local backend comes from BigQuery's earth
model, which is not the WGS84 ellipsoid, and from region edges that are
spherical in BigQuery and planar in the local backend.
"""

# The densification lives in this backend, not in the comparison
# tolerance, because the grid is bounded by parallels by definition and
# BigQuery is the one drawing the cell wrong. The gap between a chord and
# its parallel grows with the square of the chord's length, so short steps
# close it.

from __future__ import annotations

import hashlib
import io
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd


from cil_regionalization.backends.base import WeightsBackend, WeightsResult
from cil_regionalization.config import BIGQUERY_CENTRE_ONLY, Config
from cil_regionalization.fallback import (
    apply_fallback,
    compute_native_weights,
    count_methods,
)
from cil_regionalization.grid import GridSpec
from cil_regionalization.manifest import build_manifest, record_schema
from cil_regionalization.nearest_cell import find_missing_regions, synthesize_rows
from cil_regionalization.regions import RegionSet, _geometry_record
from cil_regionalization.schema import OutputSchema
from cil_regionalization.validate import check_grid_invariants, check_sum_to_one
from cil_regionalization.weights import WeightSpec

logger = logging.getLogger(__name__)


# Maximum number of attempts for transient-load-failure retries. Google's
# BigQuery 500 internalError text prescribes retry with backoff; this
# also covers ServiceUnavailable and DeadlineExceeded.
_LOAD_MAX_ATTEMPTS = 3
# Base delay in seconds; total wait is approximately base * (2 ** attempt)
# + jitter. With base 5 and 3 attempts: ~5s, ~10s, then raise.
_LOAD_BACKOFF_BASE_SECONDS = 5.0
_LOAD_BACKOFF_JITTER_SECONDS = 5.0


_LAT_ORIGIN = -90.0


class BigQueryBackend(WeightsBackend):
    """The backend that computes grid weights in BigQuery and applies the
    fallback in Python to the downloaded rows. Raises on construction if
    the ``bigquery`` extra is not installed."""

    def __init__(self) -> None:
        try:
            from google.cloud import bigquery
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "BigQueryBackend requires the [bigquery] extra; "
                "pip install 'cil_regionalization[bigquery]'"
            ) from e
        self._bq = bigquery

    # ----- main entry point -------------------------------------------------

    def compute(
        self,
        regions: RegionSet,
        grid: GridSpec,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> WeightsResult:
        t0 = time.perf_counter()
        self._validate_inputs(regions, grid, weights, cfg)
        id_fields = list(regions.id_fields)
        non_area = [w for w in weights if not w.is_area]
        if len(non_area) > 1:
            raise NotImplementedError(
                "BigQueryBackend currently supports at most one non-area "
                "weight per run (Stage 3 scope). Got: "
                f"{[w.name for w in non_area]}"
            )

        client = self._bq.Client(project=cfg.backend.bigquery.project)
        ir_location, compute_location = self._resolve_locations(
            client, regions, non_area, cfg
        )

        hierids = self._collect_hierid_filter(cfg, regions)
        null_skipped, unknown_skipped = self._handle_pre_check(
            client, regions, hierids, ir_location, cfg
        )
        if hierids is not None:
            excluded = set(null_skipped) | set(unknown_skipped)
            if excluded:
                hierids = [h for h in hierids if h not in excluded]
        fetch_t = time.perf_counter()
        geom_df = self._fetch_geometries(client, regions, hierids, ir_location, cfg)
        fetch_seconds = time.perf_counter() - fetch_t

        content_hash = self._geometry_content_hash(geom_df, id_fields[0])
        temp_table_id = self._temp_table_name(cfg, content_hash)
        cache_hit = self._table_exists(client, temp_table_id)
        upload_seconds = 0.0
        cleanup_needed = False

        try:
            if not cache_hit:
                upload_t = time.perf_counter()
                self._upload_temp_table(
                    client,
                    geom_df,
                    temp_table_id,
                    id_fields[0],
                    cfg,
                    compute_location,
                )
                upload_seconds = time.perf_counter() - upload_t
                cleanup_needed = not cfg.backend.bigquery.cache_temp_tables

            sql = self._build_main_sql(temp_table_id, regions, grid, weights, cfg)

            dry_bytes = self._dry_run_bytes(client, sql, compute_location)
            ceiling = cfg.backend.bigquery.dry_run_byte_ceiling
            if dry_bytes > ceiling:
                raise RuntimeError(
                    f"BigQuery dry-run estimates {dry_bytes / 1e9:.2f} GB scanned, "
                    f"exceeding backend.bigquery.dry_run_byte_ceiling "
                    f"({ceiling / 1e9:.2f} GB). Override the ceiling explicitly "
                    f"to allow this run."
                )

            query_t = time.perf_counter()
            job = client.query(sql, location=compute_location)
            df = self._to_dataframe(job, cfg)
            query_seconds = time.perf_counter() - query_t

            result = self._assemble_result(
                df, id_fields, weights, non_area, cfg, regions, geom_df, grid
            )
            result.manifest.timing_seconds.update(
                {
                    "bq_fetch_geometries": fetch_seconds,
                    "bq_upload_temp_table": upload_seconds,
                    "bq_query": query_seconds,
                    "compute_total": time.perf_counter() - t0,
                }
            )
            result.manifest.extra.update(
                {
                    "bq_project": cfg.backend.bigquery.project,
                    "bq_compute_location": compute_location,
                    "bq_ir_location": ir_location,
                    "bq_dry_run_bytes": int(dry_bytes),
                    "bq_temp_table": temp_table_id,
                    "bq_temp_cache_hit": bool(cache_hit),
                    "bq_geometry_content_hash": content_hash,
                    "bq_used_bqstorage": bool(cfg.backend.bigquery.use_bqstorage),
                    "null_geometry_count": len(null_skipped),
                    "null_geometry_regions": list(null_skipped),
                    "unknown_id_count": len(unknown_skipped),
                    "unknown_id_regions": list(unknown_skipped),
                    # Spherical-domain repair applied in BQ SQL at
                    # parse time via `ST_GEOGFROMTEXT(make_valid=>TRUE)`.
                    # Sits alongside `repaired_geometry_regions` (planar
                    # `shapely.make_valid` from the RegionSet loader);
                    # the two repairs are different operations in
                    # different domains and both must be visible.
                    "spherical_make_valid": True,
                }
            )
            return result
        finally:
            if cleanup_needed:
                self._delete_temp_table(client, temp_table_id)

    # ----- public dry-run --------------------------------------------------

    def dry_run(
        self,
        regions: RegionSet,
        grid: GridSpec,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> dict:
        """The combined query's scan estimate, found without running it.

        It validates the inputs, resolves the locations, pre-checks the ids,
        fetches the geometries and uploads the temp table as `compute` does,
        then dry-runs the query. The temp table is deleted afterwards unless
        ``backend.bigquery.cache_temp_tables`` is set. The estimate is not
        compared with the byte ceiling, and a second non-area weight is not
        refused, though the query covers only the first. Raises as
        `compute` does on invalid inputs, mismatched locations and failed
        pre-checks.

        Returns a dict with ``bytes_estimate``, ``compute_location``,
        ``ir_location``, ``temp_table``, ``cache_hit``, ``n_regions``,
        ``null_skipped`` and ``unknown_skipped``.
        """
        # Public so runners and notebooks can print the estimate and ask
        # before the real run, without reaching into backend internals.
        self._validate_inputs(regions, grid, weights, cfg)
        non_area = [w for w in weights if not w.is_area]
        client = self._bq.Client(project=cfg.backend.bigquery.project)
        ir_location, compute_location = self._resolve_locations(
            client, regions, non_area, cfg
        )
        hierids = self._collect_hierid_filter(cfg, regions)
        null_skipped, unknown_skipped = self._handle_pre_check(
            client, regions, hierids, ir_location, cfg
        )
        if hierids is not None:
            excluded = set(null_skipped) | set(unknown_skipped)
            if excluded:
                hierids = [h for h in hierids if h not in excluded]
        geom_df = self._fetch_geometries(client, regions, hierids, ir_location, cfg)
        content_hash = self._geometry_content_hash(geom_df, regions.id_fields[0])
        temp_table_id = self._temp_table_name(cfg, content_hash)
        cache_hit = self._table_exists(client, temp_table_id)
        cleanup_needed = False
        try:
            if not cache_hit:
                self._upload_temp_table(
                    client,
                    geom_df,
                    temp_table_id,
                    regions.id_fields[0],
                    cfg,
                    compute_location,
                )
                cleanup_needed = not cfg.backend.bigquery.cache_temp_tables
            sql = self._build_main_sql(temp_table_id, regions, grid, weights, cfg)
            bytes_estimate = self._dry_run_bytes(client, sql, compute_location)
        except Exception:
            if cleanup_needed:
                self._delete_temp_table(client, temp_table_id)
            raise
        if cleanup_needed:
            self._delete_temp_table(client, temp_table_id)
        return {
            "bytes_estimate": int(bytes_estimate),
            "compute_location": compute_location,
            "ir_location": ir_location,
            "temp_table": temp_table_id,
            "cache_hit": bool(cache_hit),
            "n_regions": int(len(geom_df)),
            "null_skipped": list(null_skipped),
            "unknown_skipped": list(unknown_skipped),
        }

    # ----- input validation -------------------------------------------------

    def _validate_inputs(
        self,
        regions: RegionSet,
        grid: GridSpec,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> None:
        if cfg.backend.kind != "bigquery":
            raise ValueError(
                f"BigQueryBackend called with backend.kind={cfg.backend.kind!r}"
            )
        if cfg.backend.coverage != "pixel_centroid":
            raise ValueError(
                "BigQuery backend supports only coverage='pixel_centroid' "
                "(point-in-polygon on raster points). Set "
                "backend.coverage='pixel_centroid' explicitly in config; "
                "Stage 4 compares it against the local backend in the same mode."
            )
        if cfg.normalization != "per_destination":
            raise ValueError(
                "normalization='per_source' is not supported by the grid "
                "backends; grid weights are normalized per region "
                "(per_destination). Per-source normalization arrives with "
                "polygon source units."
            )
        if regions.table is None and regions.gdf is None:
            raise ValueError(
                "BigQuery backend requires either regions.table (IR "
                "geometries table id) or regions.path (shapefile / "
                "geoparquet, e.g. the supplement run)."
            )
        if grid.mode != "generate":
            raise NotImplementedError(
                "BigQueryBackend Stage 3 supports grid.mode='generate' only. "
                "Prebuilt grids land later."
            )
        if grid.offset != "center":
            raise ValueError(BIGQUERY_CENTRE_ONLY.format(offset=grid.offset))
        for w in weights:
            if w.is_area:
                continue
            if w.table is None:
                raise ValueError(f"BigQuery backend requires weights.{w.name}.table")

    def _collect_hierid_filter(
        self, cfg: Config, regions: RegionSet
    ) -> list[str] | None:
        keep = cfg.regions.keep or {}
        primary = regions.id_fields[0]
        values = keep.get(primary)
        if values is None:
            return None
        return [str(v) for v in values]

    def _handle_pre_check(
        self,
        client,
        regions: RegionSet,
        hierids: list[str] | None,
        location: str,
        cfg: Config,
    ) -> tuple[list[str], list[str]]:
        """The requested ids whose geometry is NULL and those missing from
        the regions table, as ``(null_skipped, unknown_skipped)``, each
        sorted.

        With no keep filter on the primary id there are no unknown ids, and
        every NULL geometry in the table counts. A regions file is checked
        in memory, where an empty geometry counts as NULL. Raises on
        unknown ids under ``regions.on_unknown_id = "error"`` and on NULL
        geometries under ``regions.on_null_geometry = "error"``. Under
        ``"skip"`` the caller leaves the returned ids out and records them
        in the manifest.
        """
        # Unknown ids are checked because ids such as AND and BMU were
        # merged into agglomerations in the clustered world-combo table,
        # and earlier runs dropped them without a word.
        primary_id = regions.id_fields[0]
        if regions.gdf is not None:
            # Shapefile / GeoParquet source. Pre-check runs against the
            # in-memory GeoDataFrame, not a SQL query. NULL geometries
            # are detected here as is_empty (the shapefile equivalent of
            # SAFE.ST_GEOGFROMTEXT's silent null).
            null_ids, unknown_ids = self._pre_check_from_gdf(
                regions, hierids, primary_id
            )
        else:
            if hierids is not None:
                # Single query: scan IR once for the requested set, LEFT JOIN
                # back onto an UNNEST CTE to detect ids NOT in the table.
                params = [self._bq.ArrayQueryParameter("hierids", "STRING", hierids)]
                sql = f"""
WITH requested AS (
    SELECT id AS {primary_id} FROM UNNEST(@hierids) AS id
),
in_table AS (
    SELECT {primary_id}, geometry IS NULL AS is_null
    FROM `{regions.table}`
    WHERE {primary_id} IN UNNEST(@hierids)
)
SELECT
    r.{primary_id} AS {primary_id},
    it.{primary_id} IS NULL AS is_unknown,
    COALESCE(it.is_null, FALSE) AS is_null
FROM requested r
LEFT JOIN in_table it USING ({primary_id})
WHERE it.{primary_id} IS NULL OR COALESCE(it.is_null, FALSE)
""".strip()
            else:
                # Full-table mode: no keep filter, so by definition no
                # unknown ids; just count NULLs.
                params = []
                sql = (
                    f"SELECT {primary_id}, FALSE AS is_unknown, "
                    f"TRUE AS is_null "
                    f"FROM `{regions.table}` WHERE geometry IS NULL"
                )

            job_config = self._bq.QueryJobConfig(query_parameters=params)
            df = self._to_dataframe(
                client.query(sql, job_config=job_config, location=location), cfg
            )
            unknown_ids = sorted(
                str(h) for h in df.loc[df["is_unknown"], primary_id].tolist()
            )
            null_ids = sorted(
                str(h) for h in df.loc[df["is_null"], primary_id].tolist()
            )

        # Apply per-mode policies. Unknown ids surface first because
        # they are unambiguously a request bug, never a data drift.
        if unknown_ids and cfg.regions.on_unknown_id == "error":
            raise ValueError(
                f"regions.keep references {len(unknown_ids)} ids that "
                f"do not exist in {regions.table} "
                f"(on_unknown_id='error'): {unknown_ids[:25]}"
                + ("..." if len(unknown_ids) > 25 else "")
                + ".\nUse `cilreg regions find '<pattern>' <config>` "
                "to locate the correct id (e.g. 'BMU%' finds the "
                "BMU.R<hash> form). world-combo-2017 mixes bare ISO3 "
                "codes, single-remainder suffixes, and admin "
                "subdivisions; see examples/configs/s51_test.toml for "
                "the naming patterns."
            )
        if null_ids and cfg.regions.on_null_geometry == "error":
            raise ValueError(
                f"regions table {regions.table} has {len(null_ids)} "
                f"NULL geometries within the requested set "
                f"(on_null_geometry='error'): {null_ids[:25]}"
                + ("..." if len(null_ids) > 25 else "")
            )
        return null_ids, unknown_ids

    # ----- location resolution ---------------------------------------------

    def _resolve_locations(
        self,
        client,
        regions: RegionSet,
        non_area_weights: list[WeightSpec],
        cfg: Config,
    ) -> tuple[str | None, str]:
        """The regions table's location and the compute location, as
        ``(ir_location, compute_location)``. The compute location is the
        temp dataset's, and every weight table must share it. Raises,
        naming the temp dataset, each mismatched weight table and their
        locations, if one does not. ``ir_location`` is None when the
        regions come from a file.
        """
        bq_cfg = cfg.backend.bigquery
        if regions.table is not None:
            ir_dataset_id = self._dataset_id_of_table(regions.table)
            ir_location = self._dataset_location_of_id(client, ir_dataset_id)
        else:
            ir_location = None

        temp_dataset_id = f"{bq_cfg.project}.{bq_cfg.temp_dataset}"
        temp_location = self._dataset_location_of_id(client, temp_dataset_id)

        weight_dataset_locations: dict[str, str] = {}
        for w in non_area_weights:
            if w.table is None:
                continue
            ds_id = self._dataset_id_of_table(w.table)
            weight_dataset_locations[w.table] = self._dataset_location_of_id(
                client, ds_id
            )

        mismatches = [
            (table, loc)
            for table, loc in weight_dataset_locations.items()
            if loc != temp_location
        ]
        if mismatches:
            lines = [
                "BigQuery location mismatch: cannot join across regions.",
                f"  temp dataset {temp_dataset_id} is in {temp_location!r}",
            ]
            for table, loc in mismatches:
                lines.append(f"  weight table {table} is in {loc!r}")
            lines.append(
                "Choose a temp dataset in the weight table's location. "
                "Create no datasets."
            )
            raise ValueError("\n".join(lines))
        return ir_location, temp_location

    def _pre_check_from_gdf(
        self,
        regions: RegionSet,
        hierids: list[str] | None,
        primary_id: str,
    ) -> tuple[list[str], list[str]]:
        """The requested ids with a missing or empty geometry and those
        absent from the regions file, as ``(null_ids, unknown_ids)``, each
        sorted, with the meaning of the SQL pre-check. With no ``hierids``
        there are no unknown ids and every missing or empty geometry counts.
        """
        assert regions.gdf is not None  # narrowed by caller
        gdf = regions.gdf
        present_set: set[str] = set(gdf[primary_id].astype(str))
        null_mask = gdf.geometry.isna() | gdf.geometry.is_empty
        null_in_source: set[str] = set(gdf.loc[null_mask, primary_id].astype(str))

        if hierids is not None:
            requested = {str(h) for h in hierids}
            unknown_ids = sorted(requested - present_set)
            null_ids = sorted(requested & null_in_source)
        else:
            unknown_ids = []
            null_ids = sorted(null_in_source)
        return null_ids, unknown_ids

    def _dataset_id_of_table(self, table_id: str) -> str:
        project, dataset, _ = table_id.split(".", 2)
        return f"{project}.{dataset}"

    def _dataset_location_of_id(self, client, dataset_id: str) -> str:
        return client.get_dataset(dataset_id).location

    # ----- geometry fetch ---------------------------------------------------

    def _fetch_geometries(
        self,
        client,
        regions: RegionSet,
        hierids: list[str] | None,
        location: str,
        cfg: Config,
    ) -> pd.DataFrame:
        """The primary id and the WKT geometry of each region in
        ``hierids``, or of every region when it is None, leaving out NULL
        geometries. The query runs in the regions table's location. Regions
        from a file are taken from memory instead.
        """
        primary_id = regions.id_fields[0]
        if regions.gdf is not None:
            return self._fetch_geometries_from_gdf(regions, hierids, primary_id)
        params: list = []
        conditions: list[str] = ["geometry IS NOT NULL"]
        if hierids is not None:
            conditions.append(f"{primary_id} IN UNNEST(@hierids)")
            params = [self._bq.ArrayQueryParameter("hierids", "STRING", hierids)]
        sql = (
            f"SELECT {primary_id}, ST_ASTEXT(geometry) AS geometry "
            f"FROM `{regions.table}` "
            f"WHERE {' AND '.join(conditions)}"
        )
        job_config = self._bq.QueryJobConfig(query_parameters=params)
        job = client.query(sql, job_config=job_config, location=location)
        return self._to_dataframe(job, cfg)

    def _fetch_geometries_from_gdf(
        self,
        regions: RegionSet,
        hierids: list[str] | None,
        primary_id: str,
    ) -> pd.DataFrame:
        """The regions file's primary id, as a string, and WKT geometry of
        each region in ``hierids``, or of every region when it is None,
        leaving out missing and empty geometries. Geometries are not
        repaired here. `RegionSet.from_config` has already repaired them
        under ``on_invalid_geometry`` and listed them in
        ``regions.repaired_ids``.
        """
        assert regions.gdf is not None
        gdf = regions.gdf
        if hierids is not None:
            wanted = {str(h) for h in hierids}
            gdf = gdf.loc[gdf[primary_id].astype(str).isin(wanted)]
        non_empty_mask = ~(gdf.geometry.isna() | gdf.geometry.is_empty)
        gdf = gdf.loc[non_empty_mask]
        return pd.DataFrame(
            {
                primary_id: gdf[primary_id].astype(str).tolist(),
                "geometry": [g.wkt for g in gdf.geometry],
            }
        )

    def _geometry_content_hash(self, geom_df: pd.DataFrame, primary_id: str) -> str:
        """Stable 16-hex-char SHA256 prefix of (sorted id, geometry WKT) pairs."""
        h = hashlib.sha256()
        sorted_df = geom_df.sort_values(primary_id)
        for _, row in sorted_df.iterrows():
            h.update(str(row[primary_id]).encode())
            h.update(b"\x00")
            h.update(str(row["geometry"]).encode())
            h.update(b"\n")
        return h.hexdigest()[:16]

    # ----- temp table lifecycle --------------------------------------------

    def _temp_table_name(self, cfg: Config, content_hash: str) -> str:
        bq_cfg = cfg.backend.bigquery
        # ``v2`` marks the STRING-schema + ST_GEOGFROMTEXT(make_valid=>TRUE)
        # repair pipeline. Old GEOGRAPHY-schema temp tables (named
        # ``cilreg_geom_<hash>`` without the v2 marker) cannot be
        # reused: their column type would not match the new main SQL
        # that expects to parse STRING -> GEOGRAPHY itself. The first
        # canonical re-run after this change re-uploads once; subsequent
        # identical-input re-runs cache-hit normally.
        return f"{bq_cfg.project}.{bq_cfg.temp_dataset}.cilreg_geom_v2_{content_hash}"

    def _table_exists(self, client, table_id: str) -> bool:
        try:
            client.get_table(table_id)
            return True
        except Exception:
            return False

    def _upload_temp_table(
        self,
        client,
        geom_df: pd.DataFrame,
        table_id: str,
        primary_id: str,
        cfg: Config,
        location: str,
    ) -> None:
        """Load ``geom_df`` into the temp table ``table_id`` through a
        Parquet file staged under ``backend.bigquery.staging_uri``, keeping
        the geometry as a WKT string for the query to parse. Geometries are
        not simplified. The staged file is removed afterwards, and the table
        expires after ``backend.bigquery.temp_table_expiration_hours``.
        Transient load errors are retried. Raises if any geometry is NULL.
        """
        # Parquet, not JSONL. JSONL staging of multi-megabyte WKT rows made
        # the load job fail with a BigQuery 500 internalError every time,
        # and Parquet carries large strings without per-line escaping.
        bq_cfg = cfg.backend.bigquery
        import gcsfs
        import pyarrow as pa
        import pyarrow.parquet as pq

        fs = gcsfs.GCSFileSystem()
        staging_uri = (
            bq_cfg.staging_uri.rstrip("/")
            + f"/cilreg_geom_{uuid.uuid4().hex[:12]}.parquet"
        )

        # Defense in depth: the fetch query already filters NULL
        # geometries via `WHERE geometry IS NOT NULL`. If a NULL still
        # arrives here, the upstream filter or policy is broken; fail
        # with the offending hierids rather than the cryptic load-job
        # BadRequest.
        null_rows = geom_df[
            geom_df["geometry"].isna() | (geom_df["geometry"].astype(str) == "None")
        ]
        if len(null_rows) > 0:
            offending = sorted(str(h) for h in null_rows[primary_id].tolist())
            raise ValueError(
                f"_upload_temp_table: {len(offending)} regions have NULL "
                f"geometry after the fetch; upstream NULL filter is broken. "
                f"hierids: {offending[:25]}" + ("..." if len(offending) > 25 else "")
            )

        ids = [str(v) for v in geom_df[primary_id].tolist()]
        wkts = [str(v) for v in geom_df["geometry"].tolist()]
        arrow_table = pa.table({primary_id: ids, "geometry": wkts})

        max_wkt_bytes = max((len(w.encode("utf-8")) for w in wkts), default=0)
        buf = io.BytesIO()
        pq.write_table(arrow_table, buf, compression="snappy")
        payload = buf.getvalue()
        staging_size = len(payload)
        logger.info(
            "_upload_temp_table: staging %s (%.1f MB, %d rows, "
            "max_wkt_bytes=%d) for load into %s",
            staging_uri,
            staging_size / 1e6,
            len(ids),
            max_wkt_bytes,
            table_id,
        )
        with fs.open(staging_uri, "wb") as f:
            f.write(payload)

        # Geometry column lands as STRING (raw WKT). The load job can
        # NEVER fail on geometry content; parsing is deferred to SQL
        # where we run ``ST_GEOGFROMTEXT(geometry, make_valid => TRUE)``
        # so spherical-invalid polygons (planar-valid but with edges
        # that cross as geodesics, the exact failure mode of the 17
        # 2022 NULLs) get repaired in the spherical domain at parse time.
        # No SAFE. wrapper: if make_valid cannot repair we want the loud
        # named error, not a silent NULL.
        job_config = self._bq.LoadJobConfig(
            schema=[
                self._bq.SchemaField(primary_id, "STRING", mode="REQUIRED"),
                self._bq.SchemaField("geometry", "STRING", mode="REQUIRED"),
            ],
            source_format=self._bq.SourceFormat.PARQUET,
            write_disposition="WRITE_TRUNCATE",
        )

        try:
            self._run_load_with_retry(
                client, staging_uri, table_id, job_config, location
            )
        finally:
            try:
                fs.rm(staging_uri)
            except Exception:
                pass

        # Backstop expiration on the temp table regardless of cleanup setting.
        table_meta = client.get_table(table_id)
        table_meta.expires = datetime.now(timezone.utc) + timedelta(
            hours=bq_cfg.temp_table_expiration_hours
        )
        client.update_table(table_meta, ["expires"])

    def _run_load_with_retry(
        self,
        client,
        source_uri: str,
        table_id: str,
        job_config,
        location: str,
    ) -> None:
        """Run the load job, retrying with exponential backoff on
        InternalServerError, ServiceUnavailable and DeadlineExceeded, up to
        ``_LOAD_MAX_ATTEMPTS`` attempts in all. Any other error is raised at
        once. Each failure is logged with its BigQuery job id, and the last
        transient error is raised when every attempt fails.
        """
        # The job id is logged so a persistent service-side failure can be
        # taken to support with a concrete reference.
        try:
            from google.api_core.exceptions import (
                DeadlineExceeded,
                InternalServerError,
                ServiceUnavailable,
            )

            retryable: tuple = (
                InternalServerError,
                ServiceUnavailable,
                DeadlineExceeded,
            )
        except ImportError:  # pragma: no cover
            retryable = ()

        last_err: Exception | None = None
        last_job_id: str | None = None
        for attempt in range(_LOAD_MAX_ATTEMPTS):
            load_job = None
            try:
                load_job = client.load_table_from_uri(
                    source_uri,
                    table_id,
                    job_config=job_config,
                    location=location,
                )
                last_job_id = getattr(load_job, "job_id", None)
                load_job.result()
                last_err = None
                break
            except retryable as e:
                last_err = e
                # When `load_table_from_uri` itself fails (rare),
                # `load_job` stays None; otherwise pull the job_id from
                # the failed job.
                if load_job is not None:
                    last_job_id = getattr(load_job, "job_id", last_job_id)
                logger.warning(
                    "_upload_temp_table: load attempt %d/%d failed: %s: %s (job_id=%s)",
                    attempt + 1,
                    _LOAD_MAX_ATTEMPTS,
                    type(e).__name__,
                    e,
                    last_job_id,
                )
                if attempt < _LOAD_MAX_ATTEMPTS - 1:
                    delay = _LOAD_BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(
                        0.0, _LOAD_BACKOFF_JITTER_SECONDS
                    )
                    logger.warning("_upload_temp_table: retrying after %.1fs", delay)
                    time.sleep(delay)
            except Exception as e:
                # Non-retryable: schema mismatch, permission, etc.
                job_id = (
                    getattr(load_job, "job_id", None) if load_job is not None else None
                )
                logger.error(
                    "_upload_temp_table: load failed non-retryable: %s: %s (job_id=%s)",
                    type(e).__name__,
                    e,
                    job_id,
                )
                raise
        if last_err is not None:
            logger.error(
                "_upload_temp_table: all %d load attempts exhausted; "
                "last error: %s: %s (job_id=%s). Escalate with the job_id.",
                _LOAD_MAX_ATTEMPTS,
                type(last_err).__name__,
                last_err,
                last_job_id,
            )
            raise last_err

    def _delete_temp_table(self, client, table_id: str) -> None:
        try:
            client.delete_table(table_id, not_found_ok=True)
        except Exception as e:  # pragma: no cover
            print(
                f"BigQueryBackend: warning - failed to delete temp table "
                f"{table_id}: {e}",
                file=sys.stderr,
            )

    # ----- main SQL ---------------------------------------------------------

    def _build_main_sql(
        self,
        temp_table_id: str,
        regions: RegionSet,
        grid: GridSpec,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> str:
        """The combined query, which returns each region's cells with
        ``cell_lon``, ``cell_lat``, the intersection area ``area_raw`` and
        the weight's raw sum, one row per cell of positive area. Only the
        first non-area weight is included.

        ``ST_BOUNDINGBOX`` gives a dateline-crossing region an unwrapped
        longitude range, so cell indices past ``n_ix`` come out of it. The
        query wraps every ix modulo ``n_ix`` on both the cell side and the
        points side, and covers at most one full belt of longitudes per
        region. A cell reached twice through the wrap is counted once,
        keeping its largest intersection area.
        """
        # Without the wrap a polar region such as ATA would get the same
        # cell twice, at ix 0 and ix 360, and a mid-latitude crosser such
        # as FJI would lose population, because its cells at ix 360 never
        # join the points, binned from [-180, 180] longitudes, at ix 0.
        import math

        non_area = [w for w in weights if not w.is_area]
        weight = non_area[0] if non_area else None
        primary_id = regions.id_fields[0]
        lon_origin = grid.lon_origin
        lat_origin = _LAT_ORIGIN
        res = grid.resolution
        n_ix = grid.n_ix
        n_iy = grid.n_iy

        # Densify constant-latitude edges so the cell polygon faithfully
        # samples the parallel; see module docstring. n_substeps is the
        # number of equal divisions per edge; substep is the resulting
        # spacing in degrees of longitude.
        densify_step = cfg.backend.bigquery.densify_step
        n_substeps = max(1, int(math.ceil(res / densify_step)))
        substep = res / n_substeps

        pop_cte = ""
        pop_join = ""
        pop_select = "0.0 AS pop_raw"
        if weight is not None:
            value_col = self._weight_value_column(weight)
            pop_cte = f""",
points_per_cell AS (
    SELECT
        r.{primary_id} AS {primary_id},
        MOD(
            MOD(CAST(FLOOR((p.longitude - {lon_origin}) / {res}) AS INT64), {n_ix})
            + {n_ix},
            {n_ix}
        ) AS cell_ix,
        GREATEST(0, LEAST({n_iy} - 1,
            CAST(FLOOR((p.latitude - {lat_origin}) / {res}) AS INT64)
        )) AS cell_iy,
        SUM(p.{value_col}) AS raw
    FROM regions r
    JOIN `{weight.table}` p
      ON ST_CONTAINS(r.geometry, ST_GEOGPOINT(p.longitude, p.latitude))
    GROUP BY r.{primary_id}, cell_ix, cell_iy
)"""
            pop_join = (
                f"LEFT JOIN points_per_cell p\n"
                f"    ON a.{primary_id} = p.{primary_id} "
                f"AND a.cell_ix = p.cell_ix AND a.cell_iy = p.cell_iy\n"
            )
            pop_select = f"COALESCE(p.raw, 0.0) AS {weight.name}_raw"

        sql = f"""
WITH regions AS (
    -- Temp table holds geometry as raw WKT (STRING). Parse and
    -- repair in the spherical domain HERE so every downstream CTE
    -- sees one canonical GEOGRAPHY value. make_valid => TRUE fixes
    -- spherical-invalid edge crossings (planar make_valid does not
    -- catch these; a polygon can be planar-valid and still have
    -- geodesic edges that cross). No SAFE.: a make_valid failure is
    -- a loud named error, not a silent NULL.
    SELECT
        {primary_id},
        ST_GEOGFROMTEXT(geometry, make_valid => TRUE) AS geometry
    FROM `{temp_table_id}`
),
region_bboxes AS (
    SELECT
        {primary_id},
        geometry,
        ST_BOUNDINGBOX(geometry).xmin AS minx,
        ST_BOUNDINGBOX(geometry).ymin AS miny,
        ST_BOUNDINGBOX(geometry).xmax AS maxx,
        ST_BOUNDINGBOX(geometry).ymax AS maxy
    FROM regions
),
region_cells AS (
    SELECT
        rb.{primary_id} AS {primary_id},
        rb.geometry AS region_geom,
        MOD(MOD(ix, {n_ix}) + {n_ix}, {n_ix}) AS cell_ix,
        iy AS cell_iy,
        -- Parallel-densified cell polygon. Bottom edge (lat_lo) and top
        -- edge (lat_hi) get `n_substeps`+1 vertices each, evenly spaced
        -- across the cell's longitude span. Right and left edges are
        -- meridians (geodesics in both backends); no densification needed.
        ST_GEOGFROMTEXT(
            CONCAT(
                'POLYGON((',
                ARRAY_TO_STRING(
                    ARRAY_CONCAT(
                        ARRAY(
                            SELECT FORMAT(
                                '%f %f',
                                ix * {res} + {lon_origin} + k * {substep},
                                iy * {res} + {lat_origin}
                            )
                            FROM UNNEST(GENERATE_ARRAY(0, {n_substeps})) AS k
                            ORDER BY k
                        ),
                        ARRAY(
                            SELECT FORMAT(
                                '%f %f',
                                (ix + 1) * {res} + {lon_origin} - k * {substep},
                                (iy + 1) * {res} + {lat_origin}
                            )
                            FROM UNNEST(GENERATE_ARRAY(0, {n_substeps})) AS k
                            ORDER BY k
                        ),
                        [FORMAT(
                            '%f %f',
                            ix * {res} + {lon_origin},
                            iy * {res} + {lat_origin}
                        )]
                    ),
                    ','
                ),
                '))'
            )
        ) AS cell_geo
    FROM region_bboxes rb,
         UNNEST(GENERATE_ARRAY(
             CAST(FLOOR((rb.minx - {lon_origin}) / {res}) AS INT64),
             LEAST(
                 CAST(FLOOR((rb.maxx - {lon_origin}) / {res}) AS INT64),
                 CAST(FLOOR((rb.minx - {lon_origin}) / {res}) AS INT64) + {n_ix} - 1
             )
         )) AS ix,
         UNNEST(GENERATE_ARRAY(
             GREATEST(0, CAST(FLOOR((rb.miny - {lat_origin}) / {res}) AS INT64)),
             LEAST({n_iy} - 1, CAST(FLOOR((rb.maxy - {lat_origin}) / {res}) AS INT64))
         )) AS iy
),
cell_areas AS (
    SELECT
        {primary_id},
        cell_ix,
        cell_iy,
        (cell_ix + 0.5) * {res} + {lon_origin} AS cell_lon,
        (cell_iy + 0.5) * {res} + {lat_origin} AS cell_lat,
        MAX(ST_AREA(ST_INTERSECTION(region_geom, cell_geo))) AS area_raw
    FROM region_cells
    WHERE ST_INTERSECTS(region_geom, cell_geo)
    GROUP BY {primary_id}, cell_ix, cell_iy
){pop_cte}
SELECT
    a.{primary_id},
    a.cell_ix,
    a.cell_iy,
    a.cell_lon,
    a.cell_lat,
    a.area_raw,
    {pop_select}
FROM cell_areas a
{pop_join}WHERE a.area_raw > 0
ORDER BY a.{primary_id}, a.cell_ix, a.cell_iy
""".strip()
        return sql

    def _weight_value_column(self, weight: WeightSpec) -> str:
        return {"pop": "population"}.get(weight.name, weight.name)

    # ----- BigQuery execution ----------------------------------------------

    def _dry_run_bytes(self, client, sql: str, location: str) -> int:
        job_config = self._bq.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(sql, job_config=job_config, location=location)
        return int(job.total_bytes_processed or 0)

    def _to_dataframe(self, job, cfg: Config) -> pd.DataFrame:
        """The query result as a pandas frame, downloaded through the
        BigQuery Storage API only when ``backend.bigquery.use_bqstorage``
        is set.
        """
        # Every download goes through here so the Storage API stays opt-in.
        # Having the package installed must not turn it on, since accounts
        # without bigquery.readsessions.create get a 403 from it.
        return job.to_dataframe(
            create_bqstorage_client=cfg.backend.bigquery.use_bqstorage
        )

    # ----- Result assembly --------------------------------------------------

    def _assemble_result(
        self,
        df: pd.DataFrame,
        id_fields: list[str],
        all_weights: list[WeightSpec],
        non_area: list[WeightSpec],
        cfg: Config,
        regions: RegionSet,
        geom_df: pd.DataFrame,
        grid: GridSpec,
    ) -> WeightsResult:
        schema = OutputSchema(
            id_fields=tuple(id_fields),
            weight_names=tuple(w.name for w in all_weights),
            normalization=cfg.normalization,
        )

        if len(df) > 0:
            area_cols = id_fields + [
                "cell_ix",
                "cell_iy",
                "cell_lon",
                "cell_lat",
                "area_raw",
            ]
            area_native = compute_native_weights(
                df[area_cols].rename(columns={"area_raw": "raw"}),
                id_fields,
                "area",
                raw_col="raw",
            )

            pieces: list[pd.DataFrame] = [area_native]
            for spec in non_area:
                raw_col = f"{spec.name}_raw"
                raw_df = df[id_fields + ["cell_ix", "cell_iy", raw_col]].rename(
                    columns={raw_col: "raw"}
                )
                piece = apply_fallback(
                    raw_df,
                    area_native,
                    spec.name,
                    spec.fallback,
                    id_fields,
                    policy_explicit=spec.fallback_explicit,
                )
                pieces.append(piece)
            final = self._join_pieces(pieces, id_fields)
        else:
            final = schema.empty_frame()

        # nearest_cell synthesis: any requested hierid whose
        # ST_INTERSECTION returned no positive-area cell is missing from
        # df. Synthesize one row at its representative point using the
        # geometry we already fetched for the temp-table upload.
        primary_id = id_fields[0]
        requested_ids = {(str(h),) for h in geom_df[primary_id].tolist()}
        missing = find_missing_regions(final, requested_ids, id_fields)
        synth_count = 0
        if missing:
            from shapely import wkt as _wkt

            geom_lookup: dict[tuple, object] = {}
            for _, row in geom_df.iterrows():
                key = (str(row[primary_id]),)
                if key in missing:
                    geom_lookup[key] = _wkt.loads(str(row["geometry"]))
            synth = synthesize_rows(
                geom_lookup, grid, id_fields, [w.name for w in all_weights]
            )
            synth_count = len(synth)
            if synth_count > 0:
                final = pd.concat([final, synth], ignore_index=True)

        final = final[list(schema.columns)]
        final = final.sort_values(id_fields + ["cell_ix", "cell_iy"]).reset_index(
            drop=True
        )

        invariants = check_grid_invariants(final, schema, grid)
        if not invariants.ok:
            raise ValueError(invariants.summary())
        report = check_sum_to_one(final, schema, tolerance=cfg.validation.sum_tolerance)

        manifest = build_manifest(cfg)
        record_schema(manifest, schema)
        manifest.row_counts["total"] = int(len(final))
        manifest.row_counts["downloaded_rows"] = int(len(df))
        manifest.row_counts["regions"] = int(
            final[id_fields].drop_duplicates().shape[0]
        )
        manifest.row_counts["nearest_cell"] = int(synth_count)
        for w in all_weights:
            manifest.fallback_counts[w.name] = count_methods(
                final, w.name, id_fields=id_fields
            )
        if regions.table is not None:
            manifest.inputs["regions_table"] = regions.table
        if regions.source_uri is not None:
            manifest.inputs["regions_path"] = regions.source_uri
            manifest.extra.update(_geometry_record(regions))
        for w in non_area:
            if w.table is not None:
                manifest.inputs[f"weight:{w.name}"] = w.table

        return WeightsResult(
            frame=final, schema=schema, manifest=manifest, sum_report=report
        )

    def _join_pieces(
        self, pieces: list[pd.DataFrame], id_fields: list[str]
    ) -> pd.DataFrame:
        join_keys = id_fields + ["cell_ix", "cell_iy", "cell_lon", "cell_lat"]
        final = pieces[0]
        for piece in pieces[1:]:
            final = final.merge(piece, on=join_keys, how="outer")
        return final.sort_values(id_fields + ["cell_ix", "cell_iy"]).reset_index(
            drop=True
        )
