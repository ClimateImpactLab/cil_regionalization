"""The local backend, which cuts regions against grid cells or source
polygons in this process and sums rasters over the pieces.

In grid mode a region that touches both -180 and 180 is first split at
lon 0. Each region is then cut against the grid cells its bbox covers. A
cell inside the region is kept whole, and a cell on its boundary is
replaced by its intersection with the region. Each piece, a segment, gets
its geodesic area on the WGS84 ellipsoid. Under
``backend.coverage = "exact_fraction"`` a raster weight sums each pixel
times the fraction of it inside the segment, and under
``"pixel_centroid"`` it sums the pixels whose centre lies inside. The area weight normalizes the segment areas,
and every other weight goes through `apply_fallback` with its policy, as
in the BigQuery backend. A region with no positive-area cell gets one
synthesized row. The result is the `OutputSchema` frame and a manifest
with each weight's fallback counts and the run's row counts.

Polygon mode, chosen by declaring ``[source]`` in place of ``[grid]``,
cuts each region against the source polygons a spatial index finds for
it, and normalizes in the configured direction. It does not split at the
antimeridian, so both layers must use the [-180, 180] longitude form,
with dateline crossers stored as multipolygons touching both sides. A
geometry past the lon/lat domain by more than a small tolerance raises.

In both modes a segment's line and point parts are dropped, and segments
below ``backend.local.min_segment_area_m2`` are dropped and recorded in
the manifest.
"""

# Polygon mode differs on purpose from the reference IR-to-ADM
# implementation it replaces. Areas are geodesic on WGS84 as in grid mode,
# where the reference used planar Mollweide areas, so the two modes stay
# comparable. There is no antimeridian split, since that split exists only
# because cell index arithmetic must not wrap, and polygons in the
# canonical [-180, 180] form intersect correctly as they are.
#
# Region boundaries were built from administrative geometry, so source and
# target boundaries run along each other for long distances and every
# coincidence leaves floating point residue. It shows up as line parts that
# make segments GeometryCollections (`_as_multipolygons`), as near-zero
# slivers of nested source units in neighbouring targets, and as residue
# degenerate enough to crash exactextract (`_apply_min_segment_area`).
# Geometry below any physical meaning is boundary noise, dropped where
# found, with every drop recorded in the manifest.

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
from cil_regionalization.regions import RegionSet, _geometry_record
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
        if cfg.regions.on_unknown_id == "error":
            _raise_on_unknown_ids(regions)
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
            tuple(ids[f] for f in id_fields) for ids, _ in regions.iter_regions()
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
        manifest.row_counts["regions"] = int(len(regions))
        manifest.row_counts["segments"] = int(len(segments))
        manifest.row_counts["nearest_cell"] = int(synth_count)
        for w in weights:
            manifest.fallback_counts[w.name] = count_methods(
                final, w.name, id_fields=id_fields
            )
        manifest.inputs.update(_input_checksums(regions, weights))
        manifest.extra["min_segment_area"] = drop_stats
        manifest.extra.update(_geometry_record(regions))
        manifest.extra.update(_unknown_id_record(regions))
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
        """The weights in polygon mode, with the source units read from the
        ``[source]`` layer in place of a grid. A source unit whose covered
        share of its own area is below ``source.coverage_threshold`` is
        listed in the manifest's ``partial_coverage``. Raises if the region and
        source id fields share a column, if a geometry lies outside the
        lon/lat domain, if a source unit overlaps no region under
        ``source.on_zero_overlap = "error"``, or if the polygon invariants
        fail."""
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
            segments = _build_segments_polygons(regions, source_set, client=client)
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
            covered = segments.groupby(source_id_fields, sort=False)["area_raw"].sum()
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
        path = src_cfg.path
        if not str(path).startswith("gs://") and Path(path).exists():
            manifest.inputs["geometry:source"] = hash_file(path)
        manifest.extra["zero_overlap_count"] = len(zero_overlap)
        manifest.extra["zero_overlap_source_units"] = [list(t) for t in zero_overlap]
        manifest.extra["empty_region_count"] = len(empty_regions)
        manifest.extra["empty_regions"] = [list(t) for t in empty_regions]
        manifest.extra.update(_unknown_id_record(regions))
        if regions.repaired_ids:
            manifest.extra["repaired_geometry_regions"] = regions.repaired_ids
        if regions.skipped_invalid_ids:
            manifest.extra["skipped_invalid_geometry_regions"] = (
                regions.skipped_invalid_ids
            )
        if source_set.repaired_ids:
            manifest.extra["repaired_geometry_source_units"] = source_set.repaired_ids
        if source_set.skipped_invalid_ids:
            manifest.extra["skipped_invalid_geometry_source_units"] = (
                source_set.skipped_invalid_ids
            )
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
    """The geometry's western and eastern parts, cut at lon 0, when its
    bbox comes within ``_ANTIMERIDIAN_TOUCH_EPS`` degrees of both -180 and
    180, or the geometry alone otherwise.

    A dateline crosser in the [-180, 180) form touches both edges. A
    geometry whose bbox is wider than 180 degrees without touching both is
    left whole. A region around a pole, such as Antarctica, is split too. The two parts
    cover disjoint cells, so their segments never share a cell.
    """
    minx, miny, maxx, maxy = geom.bounds
    touches_west = minx < -180.0 + _ANTIMERIDIAN_TOUCH_EPS
    touches_east = maxx > 180.0 - _ANTIMERIDIAN_TOUCH_EPS
    if not (touches_west and touches_east):
        return [geom]
    west = geom.intersection(shapely.geometry.box(-180.0, miny, 0.0, maxy))
    east = geom.intersection(shapely.geometry.box(0.0, miny, 180.0, maxy))
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
    """The segments of every region on ``grid``, as a GeoDataFrame in
    EPSG:4326 with the id fields, the cell columns, ``is_interior``, the
    geometry and its geodesic ``area_raw`` in square metres. Only segments
    with positive area are kept.

    A region whose bbox comes within a degree of both -180 and 180 is
    split at lon 0 first, and both parts add segments under its id. With a
    Dask ``client`` the regions are dealt round-robin into several chunks
    per worker and built on the workers.
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
    """``pairs`` dealt round-robin into at most ``n_chunks`` lists, with
    empty lists left out."""
    # round-robin, so the big polygons do not all land in one chunk
    chunks: list[list] = [[] for _ in range(n_chunks)]
    for i, pair in enumerate(pairs):
        chunks[i % n_chunks].append(pair)
    return [c for c in chunks if c]


def _build_segments_for_pairs(
    pairs: list[tuple[dict, object]],
    id_fields: list[str],
    grid: GridSpec,
) -> list[pd.DataFrame]:
    """The segment frames of a list of (ids, geometry) pairs, one for each
    region part with a positive-area segment, for the caller to
    concatenate. It has no side effects, so a Dask worker can run it on
    one chunk.
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
                    **{f: np.full(int(positive.sum()), ids[f]) for f in id_fields},
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
    """The segments whose ``area_raw`` is at least ``min_area_m2`` square
    metres, and a stats dict for the manifest. A threshold of zero or less
    keeps every segment.

    The stats hold the threshold, the number of segments dropped, the
    largest share of any unit's intersected area that was dropped, and the
    units that lost every segment, which should never happen.
    ``group_fields`` names the units, the source ids in polygon mode and
    the region ids in grid mode.
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
        for row in kept[group_fields]
        .drop_duplicates()
        .itertuples(index=False, name=None)
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
    """Each segment as a MultiPolygon, with the line and point parts of a
    GeometryCollection dropped. Raises if a segment has no polygonal part,
    which cannot happen to a segment of positive area.
    """
    # A source boundary running along a region boundary leaves the shared
    # border in the intersection as line or point parts next to the polygon
    # parts. exactextract needs a single geometry type, and those parts have
    # no area and cover no pixel, so dropping them changes nothing computed.
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
    """The units of ``gdf`` past the lon [-180, 180] or lat [-90, 90] edge
    by no more than ``_LONLAT_DOMAIN_EPS``, each as its ids and its
    ``overshoot_degrees``, for the manifest. Raises, naming the units, if
    any geometry is past an edge by more.
    """
    # Polygon mode has no antimeridian split, so a geometry stored with
    # longitudes past 180 would intersect nothing and no error would say so.
    bounds = gdf.geometry.bounds
    overshoot = (
        pd.concat(
            [
                (-180.0 - bounds["minx"]),
                (bounds["maxx"] - 180.0),
                (-90.0 - bounds["miny"]),
                (bounds["maxy"] - 90.0),
            ],
            axis=1,
        )
        .max(axis=1)
        .clip(lower=0.0)
    )

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
    """The segments of every region against the source layer, as a
    GeoDataFrame in EPSG:4326 with the region and source id fields,
    ``is_interior``, the geometry and its geodesic ``area_raw`` in square
    metres.

    The candidates for a region come from a spatial index query on the
    source layer. A source polygon inside the region keeps its own
    geometry, any other candidate is replaced by its intersection with the
    region, and only pieces with positive area are kept. With a Dask
    ``client`` the regions are dealt round-robin into chunks as in grid
    mode, and each chunk builds its own index.
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
    """The polygon segment frames of a list of (ids, geometry) pairs, one
    for each region with a positive-area segment, for the caller to
    concatenate. It has no side effects, so a Dask worker can run it on
    one chunk. The source layer's spatial index is built once per call.
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
                **{f: np.full(int(positive.sum()), ids[f]) for f in id_fields},
                **{f: src_rows[f].to_numpy() for f in source_id_fields},
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
    """Each segment's raw total of the weight's raster under ``coverage``,
    as the id fields, ``source_key_columns`` and a column ``raw``.

    A ``"point_sum"`` weight sums the pixel values. An
    ``"area_weighted_sum"`` weight, the crop weight, reads its raster as a
    fraction and sums the fraction times each pixel's geodesic area in
    square metres. Either coverage mode applies to both. Raises if the
    weight has no raster.
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
    """The path of a temporary GeoTIFF holding each pixel's fraction times
    its geodesic area in square metres, for the fraction raster at
    ``fraction_path``.

    Every pixel in a row gets the area of that row's latitude band, which
    holds for a north-up lon/lat raster. Nodata pixels stay nodata. The
    file is cached for the life of the process by source path and
    modification time. The input must be a fraction raster with values in
    [0, 1], such as Ramankutty's cropland2000_area.tif. Its ``_ha``
    derivative already holds area and would be multiplied by it twice.
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
    """Each segment's sum of the pixel values whose centre lies inside it,
    skipping nodata, as ``id_cols`` and a column ``raw``, for
    ``coverage = "pixel_centroid"``. A segment outside the raster gets 0."""
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
    return final.sort_values(id_fields + list(source_key_columns)).reset_index(
        drop=True
    )


def _raise_on_unknown_ids(regions: RegionSet) -> None:
    problems = [
        f"regions.{name} column {col!r} lists {len(ids)} values that match "
        f"no row: {ids[:25]}" + ("..." if len(ids) > 25 else "")
        for name, by_column in (
            ("keep", regions.unknown_keep_ids),
            ("drop", regions.unknown_drop_ids),
        )
        for col, ids in by_column.items()
    ]
    if problems:
        raise ValueError(
            f"unknown ids for {regions.source_uri} (on_unknown_id='error'):\n"
            + "\n".join(problems)
        )


def _unknown_id_record(regions: RegionSet) -> dict[str, Any]:
    """Manifest entries for keep and drop values absent from the regions
    file, flattened across filter columns. The keep pair has the key names
    and shape the BigQuery backend writes."""
    keep = sorted({v for ids in regions.unknown_keep_ids.values() for v in ids})
    drop = sorted({v for ids in regions.unknown_drop_ids.values() for v in ids})
    return {
        "unknown_id_count": len(keep),
        "unknown_id_regions": keep,
        "unknown_drop_id_count": len(drop),
        "unknown_drop_id_regions": drop,
    }


def _input_checksums(
    regions: RegionSet,
    weights: list[WeightSpec],
) -> dict[str, str]:
    """Hash the regions file and each weight raster for the manifest."""
    inputs: dict[str, str] = {}
    path = regions.source_uri
    if path is not None and not str(path).startswith("gs://") and Path(path).exists():
        inputs["geometry:regions"] = hash_file(path)
    for w in weights:
        if w.is_area:
            continue
        if w.raster is not None and Path(w.raster).exists():
            inputs[f"weight:{w.name}"] = hash_file(w.raster)
    return inputs
