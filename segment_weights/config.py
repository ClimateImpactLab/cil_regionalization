"""Configuration models for segment_weights.

A run is described by a single TOML file. The schema below validates that file
up front and raises errors whose messages name the offending config key, so the
user can fix the file without reading source. Error messages are part of the
public interface: tests pin their content.

The same Config drives both the local and the BigQuery backends. Cross-field
rules (a non-area weight needs a raster on local but a table on bigquery, etc.)
live on the top-level Config so they can see both `backend.kind` and `weights`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


LonConvention = Literal["[-180,180)", "[0,360)"]
GridOffset = Literal["center", "edge"]
GridMode = Literal["generate", "prebuilt"]
CoverageMode = Literal["exact_fraction", "pixel_centroid"]
BackendKind = Literal["local", "bigquery"]
DaskMode = Literal["off", "local", "slurm"]
OutputFormat = Literal["parquet", "csv", "both"]
FallbackPolicy = Literal["area", "zero", "error", "nan"]
InvalidGeomPolicy = Literal["repair", "skip", "error"]
# Which side of the intersection the weights sum to 1 within.
#   per_destination : sum to 1 within each region, across the source units
#                     covering it. Consumed as a weighted mean of an
#                     intensive variable. The grid-to-IR case, and the
#                     default.
#   per_source      : sum to 1 within each source unit, across the regions
#                     it overlaps. Consumed as an allocation of an
#                     extensive quantity.
# The two are transposes over the same geometry and are not
# interchangeable; the chosen direction is recorded in the manifest.
Normalization = Literal["per_destination", "per_source"]
# How a non-area weight's raster is integrated into a per-segment raw total.
#   point_sum         : sum of pixel values (legacy/default; pop is one).
#   area_weighted_sum : sum of (pixel value * geodesic pixel area in m^2).
#                       The crop weight is the canonical case: the source
#                       raster is a CROPLAND FRACTION (cropland2000_area.tif,
#                       Ramankutty et al.); raw cropland in m^2 per region
#                       is fraction x pixel geodesic area, summed. Do NOT
#                       feed the legacy `_ha` derivative; despite the
#                       name, that file is fraction x area in km^2 and
#                       double-multiplies if treated as point_sum.
WeightKind = Literal["point_sum", "area_weighted_sum"]


class _Strict(BaseModel):
    """Reject unknown keys so typos in TOML surface as errors, not silent drift."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Strict):
    name: str = Field(min_length=1)
    description: Optional[str] = None


class RegionsConfig(_Strict):
    path: Optional[str] = None
    table: Optional[str] = None
    id_fields: list[str] = Field(min_length=1)
    # Vintage label of the geometry source (e.g. "world-combo-201710",
    # "gadm-2.0"), recorded in the manifest. Required in polygon mode:
    # GADM 2.0, 3.6 and 4.1 all exist on disk with different unit counts,
    # so a weight file must state which target set it was built against.
    version: Optional[str] = None
    keep: Optional[dict[str, list[Any]]] = None
    drop: Optional[dict[str, list[Any]]] = None
    on_invalid_geometry: InvalidGeomPolicy = "repair"
    # NULL geometries in BQ regions tables: error (default, abort and list
    # the offending hierids) or skip (exclude them, record in the manifest
    # under null_geometry_regions). Measured 17 NULL hierids in the s51 IR
    # table inherited from the legacy pipeline's SAFE.ST_GEOGFROMTEXT.
    on_null_geometry: Literal["error", "skip"] = "error"
    # Requested ids in `regions.keep` that do not exist in the regions
    # table: error (default, abort and list them) or skip (exclude,
    # record in the manifest under unknown_id_regions). Hardcoded keep
    # lists in legacy configs (e.g. "AND", "BMU") survived migration to
    # the clustered world-combo IR where small territories are merged
    # into larger agglomerations. Without this check those requests
    # silently went missing from the output.
    on_unknown_id: Literal["error", "skip"] = "error"

    @model_validator(mode="after")
    def _one_source(self) -> "RegionsConfig":
        has_path = self.path is not None
        has_table = self.table is not None
        if has_path and has_table:
            raise ValueError(
                "regions.path and regions.table are mutually exclusive; set exactly one"
            )
        if not has_path and not has_table:
            raise ValueError(
                "regions requires exactly one of regions.path or regions.table"
            )
        return self


class GridConfig(_Strict):
    mode: GridMode
    resolution: Optional[float] = None
    offset: GridOffset = "center"
    lon_convention: LonConvention = "[-180,180)"
    path: Optional[str] = None
    table: Optional[str] = None

    @model_validator(mode="after")
    def _consistent_mode(self) -> "GridConfig":
        has_path = self.path is not None
        has_table = self.table is not None
        if self.mode == "generate":
            if self.resolution is None:
                raise ValueError("grid.mode='generate' requires grid.resolution")
            if has_path or has_table:
                raise ValueError(
                    "grid.mode='generate' is incompatible with grid.path or grid.table "
                    "(ambiguous grid spec)"
                )
        else:  # prebuilt
            if self.resolution is not None:
                raise ValueError(
                    "grid.mode='prebuilt' is incompatible with grid.resolution "
                    "(ambiguous grid spec)"
                )
            if has_path and has_table:
                raise ValueError(
                    "grid.path and grid.table are mutually exclusive; set exactly one"
                )
            if not has_path and not has_table:
                raise ValueError(
                    "grid.mode='prebuilt' requires exactly one of grid.path or grid.table"
                )
        if self.resolution is not None and self.resolution <= 0:
            raise ValueError("grid.resolution must be positive")
        return self


class SourceConfig(_Strict):
    """Polygon source units: the geometry the weighted data lives on.

    Declaring ``[source]`` instead of ``[grid]`` switches a run to polygon
    mode: rows pair one region from ``[regions]`` (the side the weights
    are built for, e.g. administrative units) with one source polygon from
    here (e.g. impact regions keyed by hierid). Any shapefile or
    GeoParquet with id columns works; nothing here is specific to one
    geometry family.

    ``version`` is required: it is the vintage label recorded in the
    manifest so a weight file states which source set produced it.

    ``on_zero_overlap`` handles source units whose intersection with every
    region has zero geodesic area: "error" (default) aborts naming them;
    "skip" excludes them and records them in the manifest under
    ``zero_overlap_source_units``. Runs against a regional subset of
    targets should expect zero-overlap sources and set "skip" explicitly.
    """

    path: str = Field(min_length=1)
    id_fields: list[str] = Field(min_length=1)
    version: str = Field(min_length=1)
    on_invalid_geometry: InvalidGeomPolicy = "repair"
    on_zero_overlap: Literal["error", "skip"] = "error"
    # A source unit whose intersected geodesic area falls below this
    # share of the unit's own total area is partially covered by the
    # target set: its weights sum to 1 over a fraction of the real unit,
    # which sum-to-one cannot flag. Such units are recorded in the
    # manifest under partial_coverage and the application layer refuses
    # the artifact unless the caller explicitly allows it. The default
    # tolerates boundary digitization noise between geometry products
    # (well under 0.1 percent for interior units) while catching subset
    # border slivers, which sit near zero coverage. Coastline mismatch
    # between products may legitimately exceed the default for coastal
    # units; the recording exists to measure that, not hide it.
    coverage_threshold: float = Field(default=0.999, gt=0, le=1)


class WeightConfig(_Strict):
    name: str = Field(min_length=1)
    raster: Optional[str] = None
    table: Optional[str] = None
    fallback: FallbackPolicy = "area"
    kind: WeightKind = "point_sum"


class LocalBackendOptions(_Strict):
    dask: DaskMode = "off"
    n_workers: Optional[int] = Field(default=None, ge=1)
    threads_per_worker: Optional[int] = Field(default=None, ge=1)
    slurm_queue: Optional[str] = None
    slurm_walltime: Optional[str] = None
    # Segments below this geodesic area are dropped as boundary residue.
    # The default is justified by the measured global IR-to-ADM1
    # distribution: 3,410 of 28,646 segments fall below 1 m2 (floating
    # point slivers where source boundaries run along target boundaries),
    # then a natural gap with almost nothing between 1 m2 and a hectare,
    # with real segments starting around the hectare scale. The largest
    # share of any source unit's area carried by a sub-1 m2 segment is
    # 8e-9, nine orders of magnitude below the sum tolerance, so the
    # exact number is not delicate. Degenerate slivers are also what
    # crashes exactextract's traversal. Set 0 to restore the exact
    # positive-area filter. Every run records what the threshold dropped
    # in the manifest under min_segment_area, so the "nothing real is
    # being eaten" claim stays inspectable per run rather than trusted
    # from the one measurement above.
    min_segment_area_m2: float = Field(default=1.0, ge=0)


class BigQueryBackendOptions(_Strict):
    project: str = "compute-impactlab"
    # Confirmed on cilresearch (2026-06-10): `temp_workspace` is the temp
    # dataset that exists in `compute-impactlab` and is in the same location
    # (US) as the GPW weight table. Earlier inspection of legacy scripts
    # surfaced `workflow_scratch_tables` which is not present on this
    # project; never create a dataset to "make it work".
    temp_dataset: str = "temp_workspace"
    # 10 GB matches the confirmed worst-case s51 full-table scan of the GPW
    # population table (~6 GB) with headroom. The ST_CONTAINS join scans
    # the full GPW table regardless of how few regions are requested, so
    # ~6 GB is the real cost of any run, not an estimate artifact of big
    # ones. Larger runs must override explicitly in config; a
    # defense-in-depth default ships before the code that needs it.
    dry_run_byte_ceiling: int = Field(default=10_000_000_000, ge=0)
    # GCS staging URI for the geometry load job. The IR geometries table is
    # in us-west1 and GPW is in US; we fetch geometries client-side and
    # re-upload them to a temp table in the weight-table's location, with
    # a JSONL file in this prefix as the load job's source. No default:
    # which bucket a run may write to is a site fact, not library code,
    # so every BigQuery config states its own writable prefix.
    staging_uri: Optional[str] = None
    # If True, keep the content-hashed temp geometry table after a run so
    # identical-input re-runs can skip the upload. Expiration is the
    # backstop either way. Default False: explicit delete in finally,
    # predictable cleanup, fits the test scope.
    cache_temp_tables: bool = False
    # Backstop expiration on temp tables we create, in hours. Even with
    # cleanup enabled, this guards against runs that crash before finally.
    temp_table_expiration_hours: int = Field(default=24, ge=1)
    # When True, `to_dataframe(create_bqstorage_client=True)` uses the
    # BigQuery Storage Read API for faster downloads. Requires the
    # `bigquery.readsessions.create` permission (roles/bigquery.readSessionUser)
    # on the billing project; absent that role the default download via
    # `tabledata.list` works fine. Default False so an install with the
    # `bigquery-storage` package present is inert, not a 403 trap.
    use_bqstorage: bool = False
    # Maximum spacing (in degrees of longitude) between vertices on a
    # cell's constant-latitude edges. The local backend's planar cells
    # have horizontal edges that ARE parallels (constant latitude
    # rectangles); BigQuery GEOGRAPHY treats the edge between two
    # vertices as a geodesic, which bows poleward off the parallel for
    # any segment longer than ~0.1deg at mid-latitudes. Densifying with
    # this step keeps the BQ cell polygon a faithful sampling of the
    # parallel. Chord-vs-parallel error scales with the square of
    # segment length, so 0.1deg pulls the residual to ~1e-5 or better.
    # Meridian edges are geodesics in both backends and need no
    # densification. Set higher (eg 1.0) only to recover the legacy
    # single-quad behaviour.
    densify_step: float = Field(default=0.1, gt=0)
    location: Optional[str] = None


class BackendConfig(_Strict):
    kind: BackendKind
    coverage: CoverageMode = "exact_fraction"
    local: LocalBackendOptions = Field(default_factory=LocalBackendOptions)
    bigquery: BigQueryBackendOptions = Field(default_factory=BigQueryBackendOptions)


class OutputConfig(_Strict):
    dir: str = Field(min_length=1)
    format: OutputFormat = "parquet"
    # When True, runners dry-run the query, print the estimate vs the
    # configured ceiling, and exit non-zero without executing. Setting
    # `confirm_cost = false` or passing `--yes` on the runner proceeds
    # directly. Default True: no interactive stdin anywhere, only an
    # explicit policy choice via config or flag.
    confirm_cost: bool = True


class ValidationConfig(_Strict):
    sum_tolerance: float = Field(default=1e-6, gt=0)
    cross_backend_tolerance: float = Field(default=1e-3, gt=0)
    # Relative tolerance for the mass balance check: the sum of an
    # extensive quantity over target units must equal its sum over source
    # units, globally and per group. See `validate.check_mass_balance`.
    mass_balance_tolerance: float = Field(default=1e-6, gt=0)


class SourceUnitPolicies(_Strict):
    """What to do with source units that cannot flow through an application run.

    Mirrors the null-geometry and unknown-id pattern on `RegionsConfig`:
    "error" (default) aborts and names the offending units; "skip" excludes
    them, records them in the manifest under `source_coverage`, and logs a
    warning. Nothing is ever dropped without one or the other.

    The three cases, detected by `validate.check_source_coverage`:

        on_unmatched       : units present in the data being aggregated but
                             absent from the weight file. Their values
                             cannot reach any target.
        on_zero_weight     : units present in the weight file whose weights
                             are all zero or NaN. Rows exist but carry no
                             mass to any target.
        on_absent_from_data: units present in the weight file but missing
                             from the data. Their targets silently lose the
                             contribution the weights promised.

    Consumed by the application layer and the polygon weight builder; not
    referenced from the run `Config` yet.
    """

    on_unmatched: Literal["error", "skip"] = "error"
    on_zero_weight: Literal["error", "skip"] = "error"
    on_absent_from_data: Literal["error", "skip"] = "error"


class Config(_Strict):
    project: ProjectConfig
    regions: RegionsConfig
    # Exactly one of `grid` and `source` per run. `[grid]` is the original
    # mode: source units are generated grid cells. `[source]` is polygon
    # mode: source units are polygons from a second geometry file.
    grid: Optional[GridConfig] = None
    source: Optional[SourceConfig] = None
    weights: list[WeightConfig] = Field(min_length=1)
    backend: BackendConfig
    output: OutputConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    # Top-level TOML key. Existing configs omit it and keep the
    # per_destination behavior they were written for. The grid backends
    # support per_destination only; polygon mode supports both directions
    # (per_source is the allocation direction weight files for extensive
    # quantities need).
    normalization: Normalization = "per_destination"

    @model_validator(mode="after")
    def _geometry_mode(self) -> "Config":
        if self.grid is None and self.source is None:
            raise ValueError(
                "config requires exactly one of [grid] or [source]; neither is set"
            )
        if self.grid is not None and self.source is not None:
            raise ValueError(
                "[grid] and [source] are mutually exclusive; set exactly one"
            )
        if self.source is not None:
            if self.backend.kind != "local":
                raise ValueError(
                    "polygon mode ([source]) supports backend.kind='local' only. "
                    "The BigQuery backend is deferred: its one-non-area-weight "
                    "limit and centroid-only population semantics need their "
                    "own design pass before polygon source units are worth "
                    "having there."
                )
            if self.regions.version is None:
                raise ValueError(
                    "polygon mode requires regions.version: the weight file "
                    "must state which target geometry vintage it was built "
                    "against (e.g. 'gadm-2.0')"
                )
        return self

    @model_validator(mode="after")
    def _bigquery_requires_staging_uri(self) -> "Config":
        if self.backend.kind == "bigquery" and self.backend.bigquery.staging_uri is None:
            raise ValueError(
                "backend.bigquery.staging_uri is required: a gs:// prefix the "
                "run may write geometry staging files to. It is a site fact "
                "with no defensible library default."
            )
        return self

    @model_validator(mode="after")
    def _weights_match_backend(self) -> "Config":
        names = [w.name for w in self.weights]
        seen: set[str] = set()
        for n in names:
            if n in seen:
                raise ValueError(f"weights: duplicate weight name '{n}' in weights list")
            seen.add(n)

        for w in self.weights:
            if w.name == "area":
                if w.raster is not None or w.table is not None:
                    raise ValueError(
                        f"weights.{w.name}: 'area' weight does not accept raster or table"
                    )
                continue
            if self.backend.kind == "local":
                if w.raster is None:
                    raise ValueError(
                        f"weights.{w.name}: backend.kind='local' requires "
                        f"weights.{w.name}.raster"
                    )
            else:  # bigquery
                if w.table is None:
                    raise ValueError(
                        f"weights.{w.name}: backend.kind='bigquery' requires "
                        f"weights.{w.name}.table"
                    )
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a segment_weights config from a TOML file.

    Relative paths in `regions.path`, `source.path`, `weights[].raster`,
    and `output.dir` are resolved against the config file's directory so the same config
    works regardless of the caller's working directory. Absolute paths,
    `gs://` URIs, and BigQuery `table` references pass through unchanged.

    Validation errors are raised as pydantic ValidationError; their messages
    name the offending key so the user can fix the TOML without reading source.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("rb") as f:
        data = tomllib.load(f)
    _resolve_paths(data, base=p.parent)
    return Config.model_validate(data)


def _resolve_paths(data: dict[str, Any], base: Path) -> None:
    """Rewrite relative file-path fields to absolute paths under `base`."""

    def _resolve(value: str) -> str:
        if "://" in value or value.startswith("/"):
            return value
        return str((base / value).resolve())

    regions = data.get("regions")
    if isinstance(regions, dict) and "path" in regions and isinstance(regions["path"], str):
        regions["path"] = _resolve(regions["path"])

    source = data.get("source")
    if isinstance(source, dict) and "path" in source and isinstance(source["path"], str):
        source["path"] = _resolve(source["path"])

    weights = data.get("weights")
    if isinstance(weights, list):
        for w in weights:
            if isinstance(w, dict) and "raster" in w and isinstance(w["raster"], str):
                w["raster"] = _resolve(w["raster"])

    output = data.get("output")
    if isinstance(output, dict) and "dir" in output and isinstance(output["dir"], str):
        output["dir"] = _resolve(output["dir"])
