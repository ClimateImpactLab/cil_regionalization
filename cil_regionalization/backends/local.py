"""Local backend: per-region segment building + zonal stats + fallback.

For each region:

    1. Walk only the grid cells whose bbox overlaps the region's bbox
       (`grid.index_range_for_bbox`, no global enumeration).
    2. Use shapely 2 vectorized predicates to split those cells into
       interior (region fully contains the cell) and boundary (region only
       partially overlaps). Interior segments equal the cell; boundary
       segments are `region.intersection(cell)`. This split avoids
       computing intersections for cells that don't need them, which is
       what `refs/intersect_zonalstats_par.py` does too.
    3. Compute geodesic area on every segment via `pyproj.Geod` (WGS84).
       Raw lon/lat degree areas are NOT used; that was a known bug in
       the legacy script.

Per non-area weight:

    4. With `backend.coverage = "exact_fraction"`, use exactextract to
       sum pixel values weighted by the fraction of each pixel inside
       the segment. With `coverage = "pixel_centroid"`, the legacy
       semantics, sum only pixels whose center is inside the segment.
    5. Hand the resulting raw frame to `apply_fallback` with the
       per-weight policy. The same fallback logic the BigQuery backend
       uses (Stage 3) is run here so the cross-backend output is
       comparable up to the exact-vs-centroid boundary difference.

Per area weight:

    6. Run `compute_native_weights` on the geodesic-area column.

The result is the canonical `OutputSchema` frame plus a manifest with
fallback counts per weight and row counts per region.

Polygon mode
------------
Declaring ``[source]`` instead of ``[grid]`` swaps the source side: the
candidate enumeration comes from a spatial index over a second polygon
layer instead of grid bbox arithmetic, and segments are
region-by-source-polygon intersections. Everything downstream is the
same machinery: geodesic areas, exactextract or centroid zonal sums per
segment for every declared raster weight, fallback, normalization in the
configured direction, and validation (`check_polygon_invariants` plus
sum-to-1 in place of the grid invariants).

Two deliberate choices, stated here because they differ from the
reference IR-to-ADM implementation this mode replaces:

- Areas are geodesic on the WGS84 ellipsoid (`pyproj.Geod`), exactly as
  in grid mode. The reference implementation reprojected to Mollweide
  and used planar areas; a projected equal-area CRS is a compromise the
  geodesic computation does not need, and using one engine for both
  modes keeps cross-mode results comparable.
- There is no antimeridian split in polygon mode. The grid split exists
  because cell index arithmetic must not wrap; polygon-to-polygon
  intersection involves no index arithmetic, and two layers in the
  canonical [-180, 180] representation (dateline crossers stored as
  multipolygons touching both sides, as world-combo does) intersect
  correctly as they are. What polygon mode does require is that both
  layers actually use that representation, so geometries with
  coordinates outside the lon/lat domain are rejected up front.

One underlying fact, three symptoms
-----------------------------------
Impact-region boundaries were built from administrative geometry, so in
real data the source and target boundaries run along each other for
thousands of kilometers, and every coincidence produces floating point
residue in the intersection. That single fact shows up three ways:

- shared borders appear in segments as LineString parts, making them
  GeometryCollections (handled by `_as_multipolygons`),
- nested source units acquire near-zero-weight sliver pieces in
  neighboring targets, inflating raw straddler counts far above the
  material count,
- some slivers are degenerate enough (sub-square-millimeter multi-part
  residue) to crash exactextract's traversal (handled by
  `_apply_min_segment_area`).

These are one phenomenon with one rationale, not three unrelated
workarounds: geometry below physical measurement meaning is boundary
noise, dropped where found, with every drop accounted for in the
manifest.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import rasterio.windows
import shapely
from exactextract import exact_extract
from pyproj import Geod

from cil_regionalization.backends.base import WeightsBackend, WeightsResult
from cil_regionalization.compute import dask_client_for
from cil_regionalization.config import Config, CoverageMode, RegionsConfig
from cil_regionalization.fallback import (
    apply_fallback,
    compute_native_weights,
    count_methods,
)
from cil_regionalization.grid import GridSpec
from cil_regionalization.manifest import build_manifest, hash_file, record_schema
from cil_regionalization.nearest_cell import find_missing_regions, synthesize_rows
from cil_regionalization.regions import RegionSet
from cil_regionalization.schema import OutputSchema, SourceUnits
from cil_regionalization.validate import (
    check_grid_invariants,
    check_polygon_invariants,
    check_sum_to_one,
)
from cil_regionalization.weights import WeightSpec


_GEOD = Geod(ellps="WGS84")


class LocalBackend(WeightsBackend):
    def compute(
        self,
        regions: RegionSet,
        grid: GridSpec | None,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> WeightsResult:
        if not regions.is_local:
            raise ValueError(
                "LocalBackend requires a RegionSet loaded from a file; "
                "got a BigQuery-only RegionSet"
            )
        if cfg.source is not None:
            return self._compute_polygons(regions, weights, cfg)
        if grid is None:
            raise ValueError("grid mode requires a GridSpec; got None")
        return self._compute_grid(regions, grid, weights, cfg)

    def _compute_grid(
        self,
        regions: RegionSet,
        grid: GridSpec,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> WeightsResult:
        t0 = time.perf_counter()
        if cfg.normalization != "per_destination":
            raise ValueError(
                "normalization='per_source' is not supported by the grid "
                "backends; grid weights are normalized per region "
                "(per_destination). Use polygon mode ([source]) for "
                "per-source normalization."
            )
        id_fields = list(regions.id_fields)

        with dask_client_for(cfg.backend.local) as client:
            segments = _build_segments(regions, grid, client=client)
        segments, drop_stats = _apply_min_segment_area(
            segments, id_fields, cfg.backend.local.min_segment_area_m2
        )
        # `segments` may legitimately be empty when every requested region
        # has zero positive-area overlap (e.g., synthetic tests of the
        # nearest_cell path). In that case the normal pipeline is a no-op
        # and synthesis produces the entire output.

        schema = OutputSchema(
            id_fields=tuple(id_fields),
            weight_names=tuple(w.name for w in weights),
            normalization=cfg.normalization,
        )

        if len(segments) > 0:
            area_cols = id_fields + [
                "cell_ix",
                "cell_iy",
                "cell_lon",
                "cell_lat",
                "area_raw",
            ]
            area_native = compute_native_weights(
                segments[area_cols].rename(columns={"area_raw": "raw"}),
                id_fields,
                "area",
                raw_col="raw",
            )

            per_weight_frames: list[pd.DataFrame] = [area_native]
            non_area = [w for w in weights if not w.is_area]
            for spec in non_area:
                raw_df = _compute_raw_for_weight(
                    segments, spec, cfg.backend.coverage, id_fields
                )
                piece = apply_fallback(
                    raw_df,
                    area_native,
                    spec.name,
                    spec.fallback,
                    id_fields,
                    policy_explicit=spec.fallback_explicit,
                )
                per_weight_frames.append(piece)

            final = _join_pieces(per_weight_frames, id_fields)
        else:
            final = schema.empty_frame()

        # nearest_cell synthesis: every region in the input that produced
        # no positive-area cell gets a single synthesized row at its
        # representative point. Guarantees the output covers every
        # requested region.
        requested_ids = {
            tuple(ids[f] for f in id_fields)
            for ids, _ in regions.iter_regions()
        }
        missing = find_missing_regions(final, requested_ids, id_fields)
        synth_count = 0
        if missing:
            geom_lookup: dict[tuple, "shapely.Geometry"] = {}
            for ids, geom in regions.iter_regions():
                key = tuple(ids[f] for f in id_fields)
                if key in missing:
                    geom_lookup[key] = geom
            synth = synthesize_rows(
                geom_lookup, grid, id_fields, [w.name for w in weights]
            )
            synth_count = len(synth)
            if synth_count > 0:
                final = pd.concat([final, synth], ignore_index=True)

        final = final[list(schema.columns)]
        final = final.sort_values(id_fields + ["cell_ix", "cell_iy"]).reset_index(drop=True)

        invariants = check_grid_invariants(final, schema, grid)
        if not invariants.ok:
            raise ValueError(invariants.summary())
        report = check_sum_to_one(
            final, schema, tolerance=cfg.validation.sum_tolerance
        )

        manifest = build_manifest(cfg)
        record_schema(manifest, schema)
        manifest.row_counts["total"] = int(len(final))
        manifest.row_counts["regions"] = int(len(regions))
        manifest.row_counts["segments"] = int(len(segments))
        manifest.row_counts["nearest_cell"] = int(synth_count)
        for w in weights:
            manifest.fallback_counts[w.name] = count_methods(
                final, w.name, id_fields=id_fields
            )
        manifest.inputs.update(_input_checksums(regions, weights))
        manifest.extra["min_segment_area"] = drop_stats
        manifest.timing_seconds["compute_total"] = time.perf_counter() - t0

        return WeightsResult(
            frame=final, schema=schema, manifest=manifest, sum_report=report
        )

    def _compute_polygons(
        self,
        regions: RegionSet,
        weights: list[WeightSpec],
        cfg: Config,
    ) -> WeightsResult:
        """Polygon mode: source units come from ``[source]``, not a grid."""
        t0 = time.perf_counter()
        src_cfg = cfg.source
        assert src_cfg is not None
        id_fields = list(regions.id_fields)
        source_id_fields = list(src_cfg.id_fields)
        overlap = set(id_fields) & set(source_id_fields)
        if overlap:
            raise ValueError(
                f"polygon mode: regions.id_fields and source.id_fields share "
                f"columns {sorted(overlap)}; the output needs both sides' ids "
                f"as distinct columns"
            )

        source_set = RegionSet.from_config(
            RegionsConfig(
                path=src_cfg.path,
                id_fields=source_id_fields,
                on_invalid_geometry=src_cfg.on_invalid_geometry,
            )
        )
        assert regions.gdf is not None and source_set.gdf is not None
        near_edge_regions = _check_lonlat_domain(regions.gdf, id_fields, "regions")
        near_edge_sources = _check_lonlat_domain(
            source_set.gdf, source_id_fields, "source"
        )

        source_units = SourceUnits.from_string_ids(source_id_fields)
        schema = OutputSchema(
            id_fields=tuple(id_fields),
            weight_names=tuple(w.name for w in weights),
            source_units=source_units,
            normalization=cfg.normalization,
        )
        keys = id_fields + source_id_fields

        with dask_client_for(cfg.backend.local) as client:
            segments = _build_segments_polygons(
                regions, source_set, client=client
            )
        segments, drop_stats = _apply_min_segment_area(
            segments, source_id_fields, cfg.backend.local.min_segment_area_m2
        )

        loaded_source_ids = {
            tuple(ids[f] for f in source_id_fields)
            for ids, _ in source_set.iter_regions()
        }
        if len(segments) > 0:
            present_source_ids = {
                tuple(row)
                for row in segments[source_id_fields]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            }
        else:
            present_source_ids = set()
        zero_overlap = sorted(loaded_source_ids - present_source_ids)
        if zero_overlap and src_cfg.on_zero_overlap == "error":
            shown = zero_overlap[:10]
            raise ValueError(
                f"polygon mode: {len(zero_overlap)} source units have zero "
                f"overlap with every region (first {len(shown)}: {shown}). "
                f"Set source.on_zero_overlap='skip' if this run intentionally "
                f"targets a subset of regions."
            )

        if len(segments) > 0:
            area_native = compute_native_weights(
                segments[keys + ["area_raw"]].rename(columns={"area_raw": "raw"}),
                id_fields,
                "area",
                raw_col="raw",
                source_units=source_units,
                normalization=cfg.normalization,
            )
            per_weight_frames: list[pd.DataFrame] = [area_native]
            for spec in (w for w in weights if not w.is_area):
                raw_df = _compute_raw_for_weight(
                    segments,
                    spec,
                    cfg.backend.coverage,
                    id_fields,
                    source_key_columns=source_id_fields,
                )
                piece = apply_fallback(
                    raw_df,
                    area_native,
                    spec.name,
                    spec.fallback,
                    id_fields,
                    policy_explicit=spec.fallback_explicit,
                    source_units=source_units,
                    normalization=cfg.normalization,
                )
                per_weight_frames.append(piece)
            final = _join_pieces(
                per_weight_frames,
                id_fields,
                source_columns=source_id_fields,
                source_key_columns=source_id_fields,
            )
        else:
            final = schema.empty_frame()

        final = final[list(schema.columns)]
        final = final.sort_values(keys).reset_index(drop=True)

        report = check_polygon_invariants(
            final,
            schema,
            expected_source_ids=sorted(present_source_ids),
            tolerance=cfg.validation.sum_tolerance,
        )
        if not (len(report.failures) == 0 and not report.missing_source_units):
            raise ValueError(report.summary())

        requested_region_ids = {
            tuple(ids[f] for f in id_fields) for ids, _ in regions.iter_regions()
        }
        empty_regions = sorted(
            requested_region_ids
            - {
                tuple(row)
                for row in final[id_fields]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            }
        )

        # Coverage accounting: how much of each present source unit the
        # target set actually intersects. A unit whose pieces cover only
        # a sliver of its own area (a border neighbor in a subset run)
        # still gets weights summing to 1 over that sliver; recording
        # the shortfall here is what lets the application layer refuse
        # such an artifact instead of silently misplacing mass.
        partial_units: list[dict] = []
        if len(segments) > 0:
            covered = segments.groupby(source_id_fields, sort=False)[
                "area_raw"
            ].sum()
            present_geoms = {
                tuple(ids[f] for f in source_id_fields): geom
                for ids, geom in source_set.iter_regions()
                if tuple(ids[f] for f in source_id_fields) in present_source_ids
            }
            unit_keys = list(present_geoms)
            unit_totals = _geodesic_area_m2(
                np.array([present_geoms[k] for k in unit_keys], dtype=object)
            )
            for key, total in zip(unit_keys, unit_totals):
                lookup = key if len(key) > 1 else key[0]
                ratio = float(covered.loc[lookup]) / total if total > 0 else 0.0
                if ratio < cfg.source.coverage_threshold:
                    partial_units.append(
                        {
                            **dict(zip(source_id_fields, key)),
                            "coverage_ratio": ratio,
                        }
                    )
            partial_units.sort(key=lambda u: u["coverage_ratio"])

        manifest = build_manifest(cfg)
        record_schema(manifest, schema)
        manifest.row_counts["total"] = int(len(final))
        manifest.row_counts["regions"] = int(len(regions))
        manifest.row_counts["source_units"] = int(len(loaded_source_ids))
        manifest.row_counts["segments"] = int(len(segments))
        for w in weights:
            manifest.fallback_counts[w.name] = count_methods(
                final, w.name, id_fields=list(schema.normalization_group)
            )
        manifest.inputs.update(_input_checksums(regions, weights))
        for label, path in (("regions", regions.source_uri), ("source", src_cfg.path)):
            if path is not None and not str(path).startswith("gs://") and Path(path).exists():
                manifest.inputs[f"geometry:{label}"] = hash_file(path)
        manifest.extra["zero_overlap_count"] = len(zero_overlap)
        manifest.extra["zero_overlap_source_units"] = [list(t) for t in zero_overlap]
        manifest.extra["empty_region_count"] = len(empty_regions)
        manifest.extra["empty_regions"] = [list(t) for t in empty_regions]
        if regions.repaired_ids:
            manifest.extra["repaired_geometry_regions"] = regions.repaired_ids
        if source_set.repaired_ids:
            manifest.extra["repaired_geometry_source_units"] = source_set.repaired_ids
        if near_edge_regions:
            manifest.extra["near_domain_edge_regions"] = near_edge_regions
        if near_edge_sources:
            manifest.extra["near_domain_edge_source_units"] = near_edge_sources
        manifest.partial_coverage = {
            "threshold": cfg.source.coverage_threshold,
            "count": len(partial_units),
            "target_subset": bool(cfg.regions.keep or cfg.regions.drop),
            "units": partial_units[:500],
            "units_truncated": len(partial_units) > 500,
        }
        manifest.extra["min_segment_area"] = drop_stats
        manifest.timing_seconds["compute_total"] = time.perf_counter() - t0

        return WeightsResult(
            frame=final,
            schema=schema,
            manifest=manifest,
            sum_report=report.sum_report,
        )


_ANTIMERIDIAN_TOUCH_EPS = 1.0  # degrees of slack on touching +/-180


def _split_at_antimeridian(
    geom: "shapely.Geometry",
) -> list["shapely.Geometry"]:
    """Split a geometry at lon=0 when it actually crosses the antimeridian.

    The grid lives in `[-180, 180)` and never wraps. A polygon in canonical
    [-180, 180) representation that crosses the dateline has vertices very
    close to BOTH the western (-180) and eastern (+180) boundaries; the
    canonical representation forces the eastern half to wrap back as
    near-(-180) coordinates and the western half stays near (+180).

    The criterion here is therefore "bbox touches both +/-180 within
    `_ANTIMERIDIAN_TOUCH_EPS`", NOT merely "bbox wider than 180deg".
    A genuinely wide non-crossing region (geometrically impossible in
    canonical coords, but a defensive guard against malformed inputs)
    will not be split. A pole-surrounding region (Antarctica) does touch
    both boundaries and is split too; the two halves remain disjoint in
    cells, so the result stays correct.

    Splitting at lon=0 leaves each piece in one hemisphere; both halves
    fit within [-180, 180) and round-trip correctly through the planar
    intersection logic that follows. `check_grid_invariants` catches any
    regression that would produce overlapping cells.
    """
    minx, miny, maxx, maxy = geom.bounds
    touches_west = minx < -180.0 + _ANTIMERIDIAN_TOUCH_EPS
    touches_east = maxx > 180.0 - _ANTIMERIDIAN_TOUCH_EPS
    if not (touches_west and touches_east):
        return [geom]
    west = geom.intersection(
        shapely.geometry.box(-180.0, miny, 0.0, maxy)
    )
    east = geom.intersection(
        shapely.geometry.box(0.0, miny, 180.0, maxy)
    )
    halves: list = []
    if not west.is_empty:
        halves.append(west)
    if not east.is_empty:
        halves.append(east)
    return halves if halves else [geom]


def _build_segments(
    regions: RegionSet,
    grid: GridSpec,
    client: object | None = None,
) -> gpd.GeoDataFrame:
    """Per-region interior/boundary split, vectorised over candidate cells.

    Dateline-crossing regions (bbox spans >180deg lon) are split at
    lon=0 before cell enumeration; each half stays within the grid's
    [-180, 180) domain. Segments from both halves accumulate under the
    same region id; `check_grid_invariants` would catch any duplicate
    (region, cell_ix, cell_iy) keys, which there shouldn't be since
    the two halves cover disjoint ix ranges.

    When ``client`` is a Dask distributed Client, regions are partitioned
    into roughly equal chunks (one per worker, plus some headroom for
    load balancing) and dispatched concurrently. The serial branch is
    a bitwise-identical no-op for the no-client case.
    """
    id_fields = list(regions.id_fields)
    pairs = [(ids, geom) for ids, geom in regions.iter_regions()]

    if client is None or len(pairs) <= 1:
        pieces = _build_segments_for_pairs(pairs, id_fields, grid)
    else:
        n_workers = max(1, len(client.scheduler_info().get("workers", {})))
        # Heuristic: 4x oversubscription gives the scheduler latitude to
        # rebalance when some regions are much larger than others (one
        # giant Russia chunk shouldn't gate the whole stage).
        n_chunks = min(len(pairs), max(n_workers * 4, n_workers))
        chunks = _chunk_pairs(pairs, n_chunks)
        futures = client.map(
            _build_segments_for_pairs,
            chunks,
            [id_fields] * len(chunks),
            [grid] * len(chunks),
            pure=False,
        )
        results = client.gather(futures)
        pieces = [row for chunk in results for row in chunk]

    if not pieces:
        return gpd.GeoDataFrame(
            columns=id_fields
            + [
                "cell_ix",
                "cell_iy",
                "cell_lon",
                "cell_lat",
                "is_interior",
                "geometry",
                "area_raw",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    combined = pd.concat(pieces, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def _chunk_pairs(
    pairs: list[tuple[dict, object]], n_chunks: int
) -> list[list[tuple[dict, object]]]:
    """Round-robin partition. Avoids putting all big polygons in one chunk."""
    chunks: list[list] = [[] for _ in range(n_chunks)]
    for i, pair in enumerate(pairs):
        chunks[i % n_chunks].append(pair)
    return [c for c in chunks if c]


def _build_segments_for_pairs(
    pairs: list[tuple[dict, object]],
    id_fields: list[str],
    grid: GridSpec,
) -> list[pd.DataFrame]:
    """Pure: run the per-region segment build over a list of (ids, geom).

    Returns a list of per-region DataFrames; concatenation is the caller's
    job. Pulled out of ``_build_segments`` so a Dask worker can be handed
    a chunk and produce its share independently.
    """
    pieces: list[pd.DataFrame] = []
    for ids, region_geom in pairs:
      for piece in _split_at_antimeridian(region_geom):
        bbox = piece.bounds
        try:
            ix_range, iy_range = grid.index_range_for_bbox(*bbox)
        except ValueError:
            continue
        if len(ix_range) == 0 or len(iy_range) == 0:
            continue

        ix_arr, iy_arr = grid.enumerate_cells(ix_range, iy_range)
        cells = grid.cell_polygons(ix_arr, iy_arr)
        lon_c, lat_c = grid.centroid(ix_arr, iy_arr)

        interior = shapely.contains(piece, cells)
        any_touch = shapely.intersects(piece, cells)
        boundary = any_touch & ~interior

        segments = np.array(cells, dtype=object)
        b_idx = np.where(boundary)[0]
        if b_idx.size > 0:
            segments[b_idx] = shapely.intersection(piece, cells[b_idx])

        keep = interior | boundary
        if not keep.any():
            continue

        # Compute geodesic area now and drop point/line-touch artifacts
        # (shapely.intersects accepts shared-boundary points; those produce
        # Point/LineString intersections with zero area).
        kept_idx = np.where(keep)[0]
        kept_segs = segments[kept_idx]
        kept_area = _geodesic_area_m2(kept_segs)
        positive = kept_area > 0
        if not positive.any():
            continue
        sel = kept_idx[positive]

        sub = pd.DataFrame(
            {
                **{
                    f: np.full(int(positive.sum()), ids[f])
                    for f in id_fields
                },
                "cell_ix": ix_arr[sel].astype(np.int64),
                "cell_iy": iy_arr[sel].astype(np.int64),
                "cell_lon": np.asarray(lon_c)[sel],
                "cell_lat": np.asarray(lat_c)[sel],
                "is_interior": interior[sel],
                "geometry": _as_multipolygons(kept_segs[positive]),
                "area_raw": kept_area[positive],
            }
        )
        pieces.append(sub)
    return pieces


def _geodesic_area_m2(geoms: np.ndarray) -> np.ndarray:
    """WGS84 geodesic absolute area in square metres, per geometry."""
    out = np.empty(len(geoms), dtype=np.float64)
    for i, g in enumerate(geoms):
        area, _ = _GEOD.geometry_area_perimeter(g)
        out[i] = abs(area)
    return out


def _apply_min_segment_area(
    segments: gpd.GeoDataFrame,
    group_fields: list[str],
    min_area_m2: float,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Drop segments below the area threshold; account for every drop.

    ``group_fields`` names the units the discarded-share accounting is
    computed per: the source unit ids in polygon mode, the region ids in
    grid mode. The returned stats carry the drop count, the largest
    share of any unit's intersected area that was discarded, and any
    unit that lost every piece (expected never; loudly visible if a
    threshold change ever makes it happen).
    """
    stats: dict = {
        "threshold_m2": min_area_m2,
        "segments_dropped": 0,
        "max_discarded_share": 0.0,
        "units_fully_dropped": [],
    }
    if len(segments) == 0 or min_area_m2 <= 0:
        return segments, stats

    keep_mask = segments["area_raw"] >= min_area_m2
    dropped = segments.loc[~keep_mask]
    if len(dropped) == 0:
        return segments, stats

    totals = segments.groupby(group_fields, sort=False)["area_raw"].sum()
    dropped_sums = dropped.groupby(group_fields, sort=False)["area_raw"].sum()
    shares = (dropped_sums / totals.loc[dropped_sums.index]).astype(float)
    kept = segments.loc[keep_mask]
    kept_units = {
        tuple(row)
        for row in kept[group_fields].drop_duplicates().itertuples(index=False, name=None)
    }
    fully = [
        k if isinstance(k, tuple) else (k,)
        for k in dropped_sums.index
        if (k if isinstance(k, tuple) else (k,)) not in kept_units
    ]
    stats.update(
        {
            "segments_dropped": int(len(dropped)),
            "max_discarded_share": float(shares.max()),
            "units_fully_dropped": [list(t) for t in fully],
        }
    )
    return kept.reset_index(drop=True), stats


def _as_multipolygons(geoms: np.ndarray) -> np.ndarray:
    """Normalize segment geometries to MultiPolygon for zonal operations.

    Where a source boundary runs exactly along a region boundary, which
    real admin-derived layers do constantly, the intersection contains
    the shared border as LineString (and occasionally Point) parts next
    to the polygon parts, making the segment a GeometryCollection.
    exactextract requires a homogeneous geometry column, so every
    segment is promoted to MultiPolygon and non-polygonal parts are
    discarded. Discarding them is safe for the same reason the
    zero-area segment drop is: lines and points carry zero geodesic
    area and cover zero raster pixels, so no computed quantity changes.
    Segments reach this helper only after the positive-area filter, so
    every one of them has at least one polygonal part.
    """
    out = np.empty(len(geoms), dtype=object)
    for i, g in enumerate(geoms):
        if isinstance(g, shapely.MultiPolygon):
            out[i] = g
        elif isinstance(g, shapely.Polygon):
            out[i] = shapely.MultiPolygon([g])
        else:
            parts: list[shapely.Polygon] = []
            for part in shapely.get_parts(g):
                if isinstance(part, shapely.Polygon):
                    parts.append(part)
                elif isinstance(part, shapely.MultiPolygon):
                    parts.extend(shapely.get_parts(part))
            if not parts:
                raise ValueError(
                    f"segment with positive area has no polygonal parts "
                    f"({g.geom_type}); this should be impossible"
                )
            out[i] = shapely.MultiPolygon(parts)
    return out


# Tolerance for coordinates just past the lon/lat domain edge. 1e-4
# degrees is about 11 meters at the equator: it absorbs digitization
# noise in real sources (world-combo-201710's Antarctica reaches
# longitude -180.000015, an overshoot of 1.5e-5 degrees, under a meter
# on the ground), while the failure this guard exists to catch, a layer
# stored in [0, 360) longitude form, overshoots by tens of degrees and
# still fails by several orders of magnitude. Units inside the tolerance
# but past the exact edge are recorded in the manifest, not silently
# admitted.
_LONLAT_DOMAIN_EPS = 1e-4


def _check_lonlat_domain(
    gdf: gpd.GeoDataFrame, id_fields: list[str], label: str
) -> list[dict]:
    """Reject geometries with coordinates outside the lon/lat domain.

    Polygon mode has no antimeridian split, so its correctness rests on
    both layers using the canonical [-180, 180] representation. A
    geometry stored with extended longitudes (lon > 180, the style the
    grid path splits and repairs) would silently intersect with nothing;
    better to name it and stop.

    Returns the units that are within tolerance but past the exact
    domain edge, each with its overshoot in degrees, so the caller can
    record the admitted noise in the manifest.
    """
    bounds = gdf.geometry.bounds
    overshoot = pd.concat(
        [
            (-180.0 - bounds["minx"]),
            (bounds["maxx"] - 180.0),
            (-90.0 - bounds["miny"]),
            (bounds["maxy"] - 90.0),
        ],
        axis=1,
    ).max(axis=1).clip(lower=0.0)

    bad_mask = overshoot > _LONLAT_DOMAIN_EPS
    if bad_mask.any():
        bad_ids = gdf.loc[bad_mask, id_fields].to_dict(orient="records")
        raise ValueError(
            f"polygon mode: {label} has {int(bad_mask.sum())} geometries with "
            f"coordinates outside lon [-180, 180] / lat [-90, 90]: {bad_ids}. "
            f"Normalize the layer to the canonical representation (dateline "
            f"crossers as multipolygons touching both sides) before use."
        )

    near_mask = overshoot > 0.0
    near = gdf.loc[near_mask, id_fields].to_dict(orient="records")
    for record, value in zip(near, overshoot.loc[near_mask]):
        record["overshoot_degrees"] = float(value)
    return near


def _build_segments_polygons(
    regions: RegionSet,
    source_set: RegionSet,
    client: object | None = None,
) -> gpd.GeoDataFrame:
    """Per-region source-polygon intersection via a spatial index.

    The polygon counterpart of `_build_segments`: candidates come from an
    STRtree query over the source layer instead of grid bbox arithmetic.
    Sources fully contained in a region keep their own geometry; boundary
    candidates get the explicit intersection. Zero-area touches drop, as
    in grid mode. When a Dask client is given, regions are chunked
    round-robin exactly as in grid mode; each chunk builds its own index.
    """
    id_fields = list(regions.id_fields)
    source_id_fields = list(source_set.id_fields)
    pairs = [(ids, geom) for ids, geom in regions.iter_regions()]
    assert source_set.gdf is not None
    src_gdf = source_set.gdf[source_id_fields + ["geometry"]].reset_index(drop=True)

    if client is None or len(pairs) <= 1:
        pieces = _build_polygon_segments_for_pairs(
            pairs, id_fields, src_gdf, source_id_fields
        )
    else:
        n_workers = max(1, len(client.scheduler_info().get("workers", {})))
        n_chunks = min(len(pairs), max(n_workers * 4, n_workers))
        chunks = _chunk_pairs(pairs, n_chunks)
        futures = client.map(
            _build_polygon_segments_for_pairs,
            chunks,
            [id_fields] * len(chunks),
            [src_gdf] * len(chunks),
            [source_id_fields] * len(chunks),
            pure=False,
        )
        results = client.gather(futures)
        pieces = [row for chunk in results for row in chunk]

    if not pieces:
        return gpd.GeoDataFrame(
            columns=id_fields
            + source_id_fields
            + ["is_interior", "geometry", "area_raw"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    combined = pd.concat(pieces, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def _build_polygon_segments_for_pairs(
    pairs: list[tuple[dict, object]],
    id_fields: list[str],
    src_gdf: gpd.GeoDataFrame,
    source_id_fields: list[str],
) -> list[pd.DataFrame]:
    """Pure: per-region polygon segments for a list of (ids, geom).

    Mirrors `_build_segments_for_pairs` so a Dask worker can process a
    chunk independently; the STRtree is built once per invocation.
    """
    sindex = src_gdf.sindex
    src_geoms = np.asarray(src_gdf.geometry.values, dtype=object)
    pieces: list[pd.DataFrame] = []
    for ids, region_geom in pairs:
        cand_idx = sindex.query(region_geom, predicate="intersects")
        if len(cand_idx) == 0:
            continue
        cand_geoms = src_geoms[cand_idx]
        interior = shapely.contains(region_geom, cand_geoms)
        segments = np.array(cand_geoms, dtype=object)
        b_idx = np.where(~interior)[0]
        if b_idx.size > 0:
            segments[b_idx] = shapely.intersection(region_geom, cand_geoms[b_idx])

        areas = _geodesic_area_m2(segments)
        positive = areas > 0
        if not positive.any():
            continue

        src_rows = src_gdf.iloc[cand_idx[positive]]
        sub = pd.DataFrame(
            {
                **{
                    f: np.full(int(positive.sum()), ids[f])
                    for f in id_fields
                },
                **{
                    f: src_rows[f].to_numpy()
                    for f in source_id_fields
                },
                "is_interior": interior[positive],
                "geometry": _as_multipolygons(segments[positive]),
                "area_raw": areas[positive],
            }
        )
        pieces.append(sub)
    return pieces


def _compute_raw_for_weight(
    segments: gpd.GeoDataFrame,
    spec: WeightSpec,
    coverage: CoverageMode,
    id_fields: list[str],
    *,
    source_key_columns: tuple[str, ...] | list[str] = ("cell_ix", "cell_iy"),
) -> pd.DataFrame:
    """Return per-segment raw totals for one weight under the chosen coverage.

    For ``spec.kind == "point_sum"`` the raster is summed directly. For
    ``"area_weighted_sum"`` (crop weight) the raster is treated as
    pixel-wise FRACTION and a derived raster
    ``fraction * geodesic_pixel_area_m2`` is materialised once per run;
    the rest of the path is identical so exact_fraction / pixel_centroid
    coverage modes apply unchanged. The zonal machinery is agnostic to
    what a segment is: grid cells and polygon source units differ only in
    ``source_key_columns``.
    """
    raster_path = spec.require_raster()
    keys = id_fields + list(source_key_columns)

    if spec.kind == "area_weighted_sum":
        raster_path = _materialize_area_weighted_raster(raster_path)

    if coverage == "exact_fraction":
        return _zonal_sum_exact(raster_path, segments, keys)
    return _zonal_sum_centroid(raster_path, segments, keys)


_AREA_WEIGHTED_CACHE: dict[tuple[str, float], str] = {}


def _materialize_area_weighted_raster(fraction_path: str) -> str:
    """Write a derived GeoTIFF of ``fraction * pixel_geodesic_area_m2``.

    The source raster's per-row geodesic pixel area is computed once
    (one ``pyproj.Geod.geometry_area_perimeter`` call per row; area
    is longitude-invariant for a regular lat/lon grid, so each row's
    pixels share the same area). The derived raster is cached by
    ``(source_path, mtime)`` for the lifetime of the process so a
    repeated run within one process re-uses the same file.

    Crop note: the input MUST be a fraction raster (e.g. Ramankutty
    cropland2000_area.tif, values in [0, 1]); the ``_ha`` derivative
    is already (fraction x area in km^2) and feeding it here would
    double-multiply.
    """
    src_path = str(Path(fraction_path).resolve())
    mtime = Path(src_path).stat().st_mtime
    cache_key = (src_path, mtime)
    cached = _AREA_WEIGHTED_CACHE.get(cache_key)
    if cached is not None and Path(cached).exists():
        return cached

    import tempfile

    with rasterio.open(src_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        profile = src.profile.copy()
        # Per-row pixel area (m^2): rasters in EPSG:4326 with north-up
        # affine transform have constant |transform.e| latitude step
        # and constant transform.a longitude step. Area depends only on
        # the row's latitude band.
        lon_step = float(abs(transform.a))
        lat_step = float(abs(transform.e))
        north = float(transform.f)  # top edge of row 0 in north-up rasters
        row_areas = np.empty(height, dtype=np.float64)
        for r in range(height):
            lat_hi = north - r * lat_step
            lat_lo = lat_hi - lat_step
            poly = shapely.geometry.box(0.0, lat_lo, lon_step, lat_hi)
            area, _ = _GEOD.geometry_area_perimeter(poly)
            row_areas[r] = abs(area)

        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            mask = data == float(nodata)
            data = data * row_areas[:, None]
            data[mask] = float(nodata)
        else:
            data = data * row_areas[:, None]

    # Drop source's block-size hints if the source isn't tiled; rasterio
    # warns "BLOCKXSIZE can only be used with TILED=YES" otherwise.
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    profile.update(dtype="float64", count=1, compress="lzw")
    out_handle = tempfile.NamedTemporaryFile(
        suffix=".tif", prefix="cilreg_areaw_", delete=False
    )
    out_handle.close()
    out_path = out_handle.name
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float64), 1)

    _AREA_WEIGHTED_CACHE[cache_key] = out_path
    return out_path


def _zonal_sum_exact(
    raster_path: str,
    segments: gpd.GeoDataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    """Sum of (pixel_value * fraction_covered) per segment, via exactextract."""
    result = exact_extract(
        rast=str(raster_path),
        vec=segments,
        ops=["sum"],
        include_cols=id_cols,
        output="pandas",
    )
    return result.rename(columns={"sum": "raw"})[id_cols + ["raw"]]


def _zonal_sum_centroid(
    raster_path: str,
    segments: gpd.GeoDataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    """Sum of pixel values whose centroid falls inside each segment (legacy)."""
    raws: list[float] = []
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        left, bottom, right, top = src.bounds
        for geom in segments.geometry:
            minx, miny, maxx, maxy = geom.bounds
            # Clip the segment bbox to the raster extent and skip if empty
            cminx = max(minx, left)
            cmaxx = min(maxx, right)
            cminy = max(miny, bottom)
            cmaxy = min(maxy, top)
            if cminx >= cmaxx or cminy >= cmaxy:
                raws.append(0.0)
                continue

            window = rasterio.windows.from_bounds(
                cminx, cminy, cmaxx, cmaxy, src.transform
            )
            col_off = max(0, int(np.floor(window.col_off)))
            row_off = max(0, int(np.floor(window.row_off)))
            col_end = min(src.width, int(np.ceil(window.col_off + window.width)))
            row_end = min(src.height, int(np.ceil(window.row_off + window.height)))
            if col_end <= col_off or row_end <= row_off:
                raws.append(0.0)
                continue
            clean = rasterio.windows.Window(
                col_off, row_off, col_end - col_off, row_end - row_off
            )
            data = src.read(1, window=clean)
            win_transform = src.window_transform(clean)
            rows, cols = np.indices(data.shape)
            xs, ys = rasterio.transform.xy(
                win_transform,
                rows.ravel().tolist(),
                cols.ravel().tolist(),
                offset="center",
            )
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            inside = shapely.contains_xy(geom, xs, ys)
            values = data.ravel().astype(np.float64)
            if nodata is not None:
                inside &= values != float(nodata)
            raws.append(float(values[inside].sum()))
    out = segments[id_cols].copy()
    out["raw"] = raws
    return out


def _join_pieces(
    pieces: list[pd.DataFrame],
    id_fields: list[str],
    *,
    source_columns: tuple[str, ...] | list[str] = (
        "cell_ix",
        "cell_iy",
        "cell_lon",
        "cell_lat",
    ),
    source_key_columns: tuple[str, ...] | list[str] = ("cell_ix", "cell_iy"),
) -> pd.DataFrame:
    """Outer-join all per-weight frames on (id_fields + source columns)."""
    join_keys = id_fields + list(source_columns)
    final = pieces[0]
    for piece in pieces[1:]:
        final = final.merge(piece, on=join_keys, how="outer")
    return final.sort_values(
        id_fields + list(source_key_columns)
    ).reset_index(drop=True)


def _input_checksums(
    regions: RegionSet,
    weights: list[WeightSpec],
) -> dict[str, str]:
    """Hash the regions file and each weight raster for the manifest."""
    inputs: dict[str, str] = {}
    if regions.is_local and regions.gdf is not None:
        # We don't have the original path on the RegionSet, so skip the
        # regions-source hash here; cli.run will pass it via manifest.extra
        # when the file path is known.
        pass
    for w in weights:
        if w.is_area:
            continue
        if w.raster is not None and Path(w.raster).exists():
            inputs[f"weight:{w.name}"] = hash_file(w.raster)
    return inputs
