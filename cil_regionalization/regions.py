"""Region loading and geometry preparation.

A `RegionSet` holds the regions of a run in one of two modes. In local
mode the geometries are read from a shapefile or GeoParquet into a
GeoDataFrame, with the filters and the invalid-geometry policy applied
up front. In BigQuery mode only the table id is held, and the BigQuery
backend joins against the table in SQL. Both have `is_local`, `table`,
`iter_regions` and `len`, and the last two raise in BigQuery mode.
Region ids are opaque join keys, and nothing in the package parses
them.
"""

# A profile of the world-combo-201710 impact region ids, for people
# reasoning about that version's names. Nothing in code relies on it.
# The ids mix bare ISO3 codes for countries carried as a single region
# (Aruba), dotted codes for admin subdivisions and R-suffixed clustering
# remainders (Andorra). On that version's BigQuery geometries table no
# country prefix has both a bare-country row and child rows, which made
# prefix rollups safe there. That is a property of one version that the
# next may break silently, which is why ids are treated as opaque.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd
import shapely

from cil_regionalization.config import InvalidGeomPolicy, RegionsConfig


logger = logging.getLogger(__name__)


@dataclass
class RegionSet:
    id_fields: list[str]
    on_invalid_geometry: InvalidGeomPolicy
    table: str | None = None
    gdf: gpd.GeoDataFrame | None = None
    # Filled in by from_config for path-loaded gdfs whose geometries were
    # repaired by `shapely.make_valid` under the `repair` policy. The
    # backends record this in the manifest under `repaired_geometry_regions`
    # so a future reader of the parquet knows which regions' shapes were
    # mutated relative to the source.
    repaired_ids: list[dict[str, Any]] = field(default_factory=list)
    # Values listed in `regions.keep` and `regions.drop` that match no row
    # of the file as read, by filter column. The local backend applies
    # `on_unknown_id` to them and records them in the manifest.
    unknown_keep_ids: dict[str, list[str]] = field(default_factory=dict)
    unknown_drop_ids: dict[str, list[str]] = field(default_factory=dict)
    # Resolved on-disk-or-remote URI of the source. For local paths this is
    # the absolute path; for `gs://` paths it stays a URI.
    source_uri: str | None = None

    @classmethod
    def from_config(cls, cfg: RegionsConfig) -> "RegionSet":
        if cfg.table is not None:
            return cls(
                id_fields=list(cfg.id_fields),
                on_invalid_geometry=cfg.on_invalid_geometry,
                table=cfg.table,
                gdf=None,
            )
        source_uri = cfg.path  # type: ignore[assignment]
        gdf = _read_regions(source_uri)
        _check_id_fields(gdf, cfg.id_fields)
        gdf, unknown_keep, unknown_drop = _apply_filters(gdf, cfg.keep, cfg.drop)
        # Shapefiles without a .prj file lack a CRS; the canonical IR
        # shapefile is one such case. Set EPSG:4326 explicitly so
        # downstream geodesic / spatial ops have a definite frame.
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf, repaired = _handle_invalid_geometries(
            gdf, cfg.on_invalid_geometry, list(cfg.id_fields)
        )
        gdf = gdf.reset_index(drop=True)
        return cls(
            id_fields=list(cfg.id_fields),
            on_invalid_geometry=cfg.on_invalid_geometry,
            table=None,
            gdf=gdf,
            repaired_ids=repaired,
            unknown_keep_ids=unknown_keep,
            unknown_drop_ids=unknown_drop,
            source_uri=str(source_uri),
        )

    @property
    def is_local(self) -> bool:
        return self.gdf is not None

    def __len__(self) -> int:
        if self.gdf is None:
            raise ValueError(
                "RegionSet was constructed from a BigQuery table; geometries "
                "are not loaded into pandas. Use the BigQuery backend."
            )
        return len(self.gdf)

    def iter_regions(self) -> Iterator[tuple[dict[str, Any], Any]]:
        """Yield (id_dict, geometry) for each region. Local mode only."""
        if self.gdf is None:
            raise ValueError(
                "RegionSet was constructed from a BigQuery table; "
                "iter_regions() requires a loaded GeoDataFrame."
            )
        for row in self.gdf.itertuples(index=False):
            ids = {f: getattr(row, f) for f in self.id_fields}
            yield ids, row.geometry


def _read_regions(source: str | Path) -> gpd.GeoDataFrame:
    """Read regions from a local path or a `gs://` URI.

    A ``.parquet`` or ``.geoparquet`` file is read with `gpd.read_parquet`
    and anything else with `gpd.read_file`. A remote shapefile and its
    companions are copied to a temporary directory and read there. Raises
    `FileNotFoundError` if a local path does not exist.
    """
    # A remote shapefile is staged locally because GDAL's `/vsigs/` needs
    # explicit credentials, which pyogrio's default config does not pick
    # up from ADC.
    source_str = str(source)
    if source_str.startswith("gs://"):
        return _read_remote_regions(source_str)
    path = Path(source_str)
    if not path.exists():
        raise FileNotFoundError(f"regions.path not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".geoparquet"):
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _read_remote_regions(uri: str) -> gpd.GeoDataFrame:
    """Stream shapefile companions from GCS to a temp dir, then read."""
    import gcsfs
    import tempfile

    fs = gcsfs.GCSFileSystem()
    suffix = uri.lower().rsplit(".", 1)[-1] if "." in uri else ""
    if suffix in ("parquet", "geoparquet"):
        with fs.open(uri, "rb") as f:
            return gpd.read_parquet(f)
    if suffix != "shp":
        # GeoParquet/GeoPackage etc.: let geopandas attempt a remote read.
        return gpd.read_file(uri)
    # Shapefiles: download .shp + .dbf + .shx + .prj (if present).
    base = uri[:-4]  # strip ".shp"
    tmp = Path(tempfile.mkdtemp(prefix="cilreg-regions-"))
    for ext in (".shp", ".dbf", ".shx", ".prj"):
        remote = base + ext
        try:
            fs.get(remote, str(tmp / Path(remote).name))
        except FileNotFoundError:
            if ext in (".shp", ".dbf", ".shx"):
                raise
            # .prj is optional; absence means downstream sets EPSG:4326.
    local_shp = tmp / Path(uri).name
    logger.info("regions: staged %s to %s for local read", uri, tmp)
    return gpd.read_file(local_shp)


def _check_id_fields(gdf: gpd.GeoDataFrame, id_fields: list[str]) -> None:
    missing = [f for f in id_fields if f not in gdf.columns]
    if missing:
        raise ValueError(f"regions.id_fields missing from source: {missing}")


def _apply_filters(
    gdf: gpd.GeoDataFrame,
    keep: dict[str, list[Any]] | None,
    drop: dict[str, list[Any]] | None,
) -> tuple[gpd.GeoDataFrame, dict[str, list[str]], dict[str, list[str]]]:
    """Apply keep, then drop. Returns ``(gdf, unknown_keep, unknown_drop)``,
    the filtered frame and, per filter column, the listed values that
    match no row of ``gdf`` as passed in. Columns with no such value are
    left out. Raises if a filter names a column ``gdf`` does not have."""
    full = gdf
    unknown_keep: dict[str, list[str]] = {}
    unknown_drop: dict[str, list[str]] = {}
    if keep:
        for col, allowed in keep.items():
            if col not in gdf.columns:
                raise ValueError(f"regions.keep references unknown column '{col}'")
            if absent := _absent_values(full[col], allowed):
                unknown_keep[col] = absent
            gdf = gdf.loc[gdf[col].isin(allowed)]
    if drop:
        for col, excluded in drop.items():
            if col not in gdf.columns:
                raise ValueError(f"regions.drop references unknown column '{col}'")
            if absent := _absent_values(full[col], excluded):
                unknown_drop[col] = absent
            gdf = gdf.loc[~gdf[col].isin(excluded)]
    return gdf, unknown_keep, unknown_drop


def _absent_values(column, listed: list[Any]) -> list[str]:
    present = set(column[column.isin(listed)].tolist())
    # A null in the list is left out: it matches null rows under isin but
    # never compares equal to them here.
    return sorted(
        {str(v) for v in listed if v is not None and v == v and v not in present}
    )


def _handle_invalid_geometries(
    gdf: gpd.GeoDataFrame,
    policy: InvalidGeomPolicy,
    id_fields: list[str],
) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
    """Apply the invalid-geometry policy. Returns ``(gdf, bad_ids)``, the
    ids of the invalid geometries whether they were repaired or skipped.
    Raises under "error" if any geometry is invalid."""
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid == 0:
        return gdf, []

    bad_ids = gdf.loc[invalid_mask, id_fields].to_dict(orient="records")
    if policy == "repair":
        logger.info(
            "regions: repairing %d invalid geometries; ids=%s", n_invalid, bad_ids
        )
        repaired = gdf.geometry.copy()
        repaired.loc[invalid_mask] = gdf.loc[invalid_mask, "geometry"].apply(
            shapely.make_valid
        )
        out = gdf.copy()
        out = out.set_geometry(repaired)
        return out, bad_ids
    if policy == "skip":
        logger.info(
            "regions: skipping %d invalid geometries; ids=%s", n_invalid, bad_ids
        )
        return gdf.loc[~invalid_mask].copy(), bad_ids
    raise ValueError(
        f"regions: {n_invalid} invalid geometries with on_invalid_geometry='error'; "
        f"ids={bad_ids}"
    )
