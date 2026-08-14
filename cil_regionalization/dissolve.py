"""Prepare target geometry layers by dissolving a finer polygon layer.

The polygon weight engine takes any target geometry file with declared id
fields. Sources like the combined GADM file ship every admin level in one
layer at the deepest granularity, so the per-level target files have to
be dissolved out first. This module is that step: generic, keyed on
caller-declared columns, with the same discipline the rest of the repo
applies to geometry work.

Guarantees per dissolve:

- key columns must exist and carry no nulls (offenders are named),
- the dissolved keys are unique by construction and verified anyway,
- total planar area is conserved within a stated tolerance, which is the
  proof that no geometry was lost or double-merged. Planar (coordinate
  unit) area is the right conservation metric here: dissolving is a
  union, so the metric only has to be consistent before and after, and
  planar area is cheap at this scale. Geodesic areas belong to the
  weight engine, not to layer preparation. Note the converse: inputs
  whose pieces overlap each other lose area under union, and this check
  reports exactly that instead of hiding it.
- invalid source geometries are repaired with `shapely.make_valid`
  before union, and the repair count is recorded, mirroring the
  `regions.py` policy of loud rather than silent repair.

`prepare_target_layers` orchestrates several levels from one source read
and writes GeoParquet plus a JSON provenance manifest (source path and
checksum, feature counts, per-level unit counts and area accounting,
version label, package versions), so what was produced is inspectable
later without rerunning.

Display-name columns are carried as attributes for orientation only.
Whatever encoding state they are in comes along verbatim; they are never
keys, and nothing downstream may join on them.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import pandas as pd
import shapely

from cil_regionalization.manifest import collect_package_versions, hash_file


def _planar_area(geometry: gpd.GeoSeries) -> pd.Series:
    """Planar area in coordinate units, silencing the geographic-CRS
    warning: planar area is the intended conservation metric here (see
    the module docstring), not an accident."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*geographic CRS.*", category=UserWarning
        )
        return geometry.area


@dataclass(frozen=True)
class DissolveReport:
    """Accounting for one dissolved level.

    ``worst_units`` lists the units with the largest area difference
    between the sum of their source pieces and their union, largest
    first. A positive difference means the unit's pieces overlap each
    other in the source and the union counted the shared area once.
    """

    n_source_features: int
    n_units: int
    n_repaired_geometries: int
    source_area: float
    dissolved_area: float
    area_rel_diff: float
    tolerance: float
    worst_units: tuple[dict, ...] = ()


@dataclass(frozen=True)
class LevelSpec:
    """One target level: output name, key columns, carried name columns."""

    name: str
    key_columns: tuple[str, ...]
    name_columns: tuple[str, ...] = ()


def dissolve_layer(
    gdf: gpd.GeoDataFrame,
    key_columns: Sequence[str],
    *,
    name_columns: Sequence[str] = (),
    area_rel_tol: float = 1e-6,
) -> tuple[gpd.GeoDataFrame, DissolveReport]:
    """Dissolve `gdf` to one row per key combination.

    Name columns are aggregated by first value per unit. Raises if key
    columns are missing or null, if the dissolved keys are not unique, or
    if total planar area is not conserved within `area_rel_tol`.
    """
    keys = list(key_columns)
    names = [c for c in name_columns if c not in keys]
    missing = [c for c in keys + names if c not in gdf.columns]
    if missing:
        raise ValueError(f"dissolve: source is missing columns {missing}")

    null_mask = gdf[keys].isna().any(axis=1)
    if null_mask.any():
        offenders = gdf.loc[null_mask, keys].drop_duplicates().head(10)
        raise ValueError(
            f"dissolve: {int(null_mask.sum())} source features have null key "
            f"values on {keys}; first offenders:\n"
            f"{offenders.to_string(index=False)}"
        )

    work = gdf[keys + names + ["geometry"]].copy()
    invalid_mask = ~work.geometry.is_valid
    n_repaired = int(invalid_mask.sum())
    if n_repaired > 0:
        repaired = work.geometry.copy()
        repaired.loc[invalid_mask] = work.loc[invalid_mask, "geometry"].apply(
            shapely.make_valid
        )
        work = work.set_geometry(repaired)

    source_area = float(_planar_area(work.geometry).sum())

    if names:
        dissolved = work.dissolve(
            by=keys, as_index=False, aggfunc={c: "first" for c in names}
        )
    else:
        # dissolve requires at least one aggregated column; a throwaway
        # constant serves when no name columns are carried.
        dissolved = (
            work.assign(_carry=1)
            .dissolve(by=keys, as_index=False, aggfunc={"_carry": "first"})
            .drop(columns=["_carry"])
        )

    if dissolved.duplicated(subset=keys).any():
        dupes = dissolved.loc[dissolved.duplicated(subset=keys, keep=False), keys]
        raise ValueError(
            f"dissolve: duplicate keys after dissolve (should be impossible): "
            f"{dupes.head(10).to_string(index=False)}"
        )

    dissolved_area = float(_planar_area(dissolved.geometry).sum())
    scale = max(abs(source_area), abs(dissolved_area))
    rel = abs(source_area - dissolved_area) / scale if scale > 0 else 0.0

    # Per-unit accounting: which units' pieces overlap each other. The
    # difference of sums equals the sum of per-unit differences, so the
    # global check's failures are always attributable here.
    piece_sums = (
        work.assign(_a=_planar_area(work.geometry)).groupby(keys, sort=False)["_a"].sum()
    )
    unit_areas = _planar_area(dissolved.set_index(keys).geometry)
    unit_diff = (piece_sums - unit_areas).sort_values(ascending=False)
    worst = tuple(
        {
            **dict(zip(keys, k if isinstance(k, tuple) else (k,))),
            "piece_area_sum": float(piece_sums.loc[k]),
            "union_area": float(unit_areas.loc[k]),
            "overlap_area": float(d),
        }
        for k, d in unit_diff.head(5).items()
        if abs(d) > 0.0
    )

    report = DissolveReport(
        n_source_features=len(gdf),
        n_units=len(dissolved),
        n_repaired_geometries=n_repaired,
        source_area=source_area,
        dissolved_area=dissolved_area,
        area_rel_diff=rel,
        tolerance=area_rel_tol,
        worst_units=worst,
    )
    if rel > area_rel_tol:
        raise ValueError(
            f"dissolve: total area not conserved (source {source_area:.10g}, "
            f"dissolved {dissolved_area:.10g}, rel diff {rel:.3e} > tolerance "
            f"{area_rel_tol:.1e}). Overlapping source pieces lose area under "
            f"union; worst units: {list(worst)}. Inspect the source, and if "
            f"the overlap is a measured property of it, pass an explicit "
            f"tolerance with the measurement documented."
        )
    return dissolved.reset_index(drop=True), report


def prepare_target_layers(
    source_path: str | Path,
    levels: Iterable[LevelSpec],
    out_dir: str | Path,
    *,
    version: str,
    area_rel_tol: float = 1e-6,
    drop_query: str | None = None,
    repair=None,
    repair_record: dict | None = None,
) -> dict:
    """Read `source_path` once, dissolve each level, write GeoParquet.

    ``drop_query`` optionally names source features to exclude before
    dissolving, as a pandas query over the key and name columns (only
    columns declared by some level are read, so the query may reference
    only those). It
    exists for documented source defects (features whose area duplicates
    other features would otherwise break area conservation); the query
    and the number of dropped features are recorded in the manifest, so
    an exclusion is always a visible editorial decision, never a silent
    one.

    ``repair`` optionally edits the source frame before ``drop_query``
    and dissolving: a callable taking and returning the frame. It exists
    for documented source defects that dropping cannot fix (a key field
    left blank on features whose correct value the source itself
    states). A repair must come with ``repair_record``, a JSON
    serializable account of what was changed, on how many rows, and on
    what evidence; the record goes into the manifest verbatim. A repair
    without a record, or a record without a repair, is refused: edits to
    source data are visible editorial decisions or they do not happen.

    Returns the provenance manifest (also written to
    ``<out_dir>/targets.manifest.json``): source identity and checksum,
    per-level counts and area accounting, the version label to use as
    ``regions.version`` in weight configs, and package versions.
    """
    if (repair is None) != (repair_record is None):
        raise ValueError(
            "prepare_target_layers: repair and repair_record come together; "
            "an edit to source data must carry its recorded account"
        )
    levels = list(levels)
    source_path = Path(source_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    needed: list[str] = []
    for spec in levels:
        for c in list(spec.key_columns) + list(spec.name_columns):
            if c not in needed:
                needed.append(c)
    if source_path.suffix.lower() in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(source_path, columns=needed + ["geometry"])
    else:
        gdf = gpd.read_file(source_path, columns=needed)
    if gdf.crs is None:
        raise ValueError(
            f"prepare_target_layers: {source_path} has no CRS; target layers "
            f"must carry one so the weight engine's lon/lat assumptions hold"
        )

    n_read = len(gdf)
    if repair is not None:
        gdf = repair(gdf)
        if len(gdf) != n_read:
            raise ValueError(
                f"prepare_target_layers: repair changed the feature count "
                f"({n_read} to {len(gdf)}); repairs edit values, drop_query "
                f"removes features"
            )
    n_dropped = 0
    if drop_query is not None:
        keep = ~gdf.index.isin(gdf.query(drop_query).index)
        n_dropped = int((~keep).sum())
        if n_dropped == 0:
            raise ValueError(
                f"prepare_target_layers: drop_query {drop_query!r} matched "
                f"no features; a stale exclusion hides a change in the source"
            )
        gdf = gdf.loc[keep].reset_index(drop=True)

    manifest: dict = {
        "source": str(source_path),
        "source_sha256": hash_file(source_path),
        "source_crs": str(gdf.crs),
        "n_source_features": n_read,
        "n_dropped_features": n_dropped,
        "drop_query": drop_query,
        "repair": repair_record,
        "version": version,
        "package_versions": collect_package_versions(),
        "levels": {},
        "note": (
            "Name columns are carried verbatim from the source for "
            "orientation only and must never be used as join keys; the key "
            "columns are the stable identifiers."
        ),
    }
    for spec in levels:
        layer, report = dissolve_layer(
            gdf,
            spec.key_columns,
            name_columns=spec.name_columns,
            area_rel_tol=area_rel_tol,
        )
        path = out / f"{spec.name}.parquet"
        layer.to_parquet(path, index=False)
        manifest["levels"][spec.name] = {
            "path": str(path),
            "key_columns": list(spec.key_columns),
            "name_columns": list(spec.name_columns),
            **asdict(report),
        }

    manifest_path = out / "targets.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    manifest["manifest_path"] = str(manifest_path)
    return manifest
