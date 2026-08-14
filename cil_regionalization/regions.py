"""Region loading and geometry preparation.

`RegionSet` carries the regions used in a run. Two modes:

    local : geometries are read into a GeoDataFrame from a shapefile or
            GeoParquet. Filters and the invalid-geometry policy are applied
            up front so downstream code sees a clean frame.
    bigquery : only the table id is held; the BigQuery backend joins
               against this table in SQL without round-tripping geometries
               through pandas.

The same `RegionSet` API serves both: `is_local`, `table`, `iter_regions`,
`__len__`. The BQ-only constructor refuses iteration since it has no
geometries to hand out.

One version's id regularities, recorded, not relied on
------------------------------------------------------
Region ids are opaque join keys; nothing in the package parses their
structure, and a test enforces that. What follows is a measured profile
of the world-combo-201710 impact region version, kept for humans who
need to reason about that version's names, with no conclusions encoded
in code. Its ids mix three shapes: bare ISO3 codes for countries
carried as a single region (Aruba is one), dotted hierarchical codes
for admin subdivisions, and R-suffixed clustering remainders (Andorra's
single region is one) that already break clean hierarchy semantics.
Measured on the BigQuery geometries table for that version: no country
prefix has both a bare-country row and child rows, which is the one
fact that made prefix-level rollups safe there. That regularity is a
property of one version, never a contract; the next version may break
it silently, which is exactly why the package treats ids as opaque.
"""
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
        gdf = _apply_filters(gdf, cfg.keep, cfg.drop)
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

    Local: pyogrio reads shapefiles in place; `gpd.read_parquet` for
    parquet. Remote: shapefile companions are streamed via gcsfs into a
    temp dir and read locally (GDAL/`/vsigs/` needs explicit credentials
    which pyogrio's default config doesn't pick up from ADC).
    """
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
) -> gpd.GeoDataFrame:
    if keep:
        for col, allowed in keep.items():
            if col not in gdf.columns:
                raise ValueError(f"regions.keep references unknown column '{col}'")
            gdf = gdf.loc[gdf[col].isin(allowed)]
    if drop:
        for col, excluded in drop.items():
            if col not in gdf.columns:
                raise ValueError(f"regions.drop references unknown column '{col}'")
            gdf = gdf.loc[~gdf[col].isin(excluded)]
    return gdf


def _handle_invalid_geometries(
    gdf: gpd.GeoDataFrame,
    policy: InvalidGeomPolicy,
    id_fields: list[str],
) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
    """Apply the invalid-geometry policy. Returns ``(gdf, repaired_ids)``."""
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
