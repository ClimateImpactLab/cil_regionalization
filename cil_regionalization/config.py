"""Configuration models for cil_regionalization.

A run is described by a single TOML file, which the same `Config` reads
for the local and the BigQuery backends. The file is validated up front,
and every error message names the config key at fault. Tests pin the
text of those messages. Rules that span sections, such as a non-area
weight needing a raster on the local backend and a table on BigQuery,
are checked on `Config`.
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
# The BigQuery backend's SQL places every cell for the centre offset.
BIGQUERY_CENTRE_ONLY = (
    "backend.kind = 'bigquery' takes grid.offset = 'center' only, and got "
    "{offset!r}. Its SQL handles centre offsets only, so the weights of an "
    "edge-offset grid would come out half a cell off. Use backend.kind = "
    "'local' for an edge-offset grid."
)
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
    """A config model that raises on any key it does not declare, so a
    misspelt key in the TOML is an error."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Strict):
    name: str = Field(min_length=1)
    description: Optional[str] = None


class RegionsConfig(_Strict):
    path: Optional[str] = None
    table: Optional[str] = None
    id_fields: list[str] = Field(min_length=1)
    # Version label of the geometry source (e.g. "world-combo-201710",
    # "gadm-2.0"), recorded in the manifest. Required in polygon mode:
    # GADM 2.0, 3.6 and 4.1 all exist on disk with different unit counts,
    # so a weight file must state which target set it was built against.
    version: Optional[str] = None
    keep: Optional[dict[str, list[Any]]] = None
    drop: Optional[dict[str, list[Any]]] = None
    # No default: repair changes a region's geometry and skip leaves it
    # out, so a wrong choice gives other weights with nothing said.
    on_invalid_geometry: InvalidGeomPolicy
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
    # The local backend applies the same policy to every column named in
    # `regions.keep` and `regions.drop`, against the regions file, and
    # records drop values under unknown_drop_id_regions. BigQuery checks
    # keep on the primary id only.
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
    # No default for either: a grid at the wrong offset or longitude
    # origin still covers the regions, so the weights come out on cells
    # half a cell or a hemisphere from the data's, with nothing said.
    offset: GridOffset
    lon_convention: LonConvention
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
    """The polygons the weighted data lives on, as the source units of a
    polygon-mode run.

    Declaring ``[source]`` instead of ``[grid]`` puts a run in polygon
    mode. Each row of the weights then pairs a region from ``[regions]``,
    the side the weights are built for such as administrative units, with
    a polygon from this file such as an impact region keyed by hierid.
    Any shapefile or GeoParquet with id columns will do. ``version`` is
    the label the manifest records for this source.

    ``on_zero_overlap`` says what to do with source units whose
    intersection with every region has zero geodesic area. "error" raises
    naming them, and "skip" leaves them out and lists them in the
    manifest's ``extra`` under ``zero_overlap_source_units``. A run over a
    regional subset of targets should expect such units and set "skip".
    """

    path: str = Field(min_length=1)
    id_fields: list[str] = Field(min_length=1)
    version: str = Field(min_length=1)
    on_invalid_geometry: InvalidGeomPolicy
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
    """One weight of the run. Every weight but ``area`` needs its
    ``fallback`` and its ``kind``, which have no default, and the area
    weight takes neither, since its total is the region's own area and it
    reads no raster. A BigQuery weight names the ``column`` of its
    ``table`` that holds its values."""

    name: str = Field(min_length=1)
    raster: Optional[str] = None
    table: Optional[str] = None
    column: Optional[str] = Field(default=None, min_length=1)
    fallback: Optional[FallbackPolicy] = None
    kind: Optional[WeightKind] = None


class LocalBackendOptions(_Strict):
    dask: DaskMode = "off"
    # Under "slurm" the number of jobs, and under "local" the number of
    # processes, four unless set.
    n_workers: Optional[int] = Field(default=None, ge=1)
    threads_per_worker: Optional[int] = Field(default=None, ge=1)
    # The queue, the cores and memory of one job, and its walltime, as
    # dask_jobqueue takes them. Facts of a cluster, so none has a
    # default, and dask = "slurm" needs every one of them and n_workers.
    slurm_queue: Optional[str] = Field(default=None, min_length=1)
    slurm_cores: Optional[int] = Field(default=None, ge=1)
    slurm_memory: Optional[str] = Field(default=None, min_length=1)
    slurm_walltime: Optional[str] = Field(default=None, min_length=1)
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

    @model_validator(mode="after")
    def _slurm_is_declared(self) -> "LocalBackendOptions":
        if self.dask != "slurm":
            return self
        missing = [
            name
            for name in (
                "slurm_queue",
                "slurm_cores",
                "slurm_memory",
                "slurm_walltime",
                "n_workers",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"backend.local.dask = 'slurm' needs "
                f"{_listed(['backend.local.' + name for name in missing])}. "
                f"The queue, the cores, memory and walltime of a job and the "
                f"number of jobs belong to a cluster, so none has a default."
            )
        return self


def _listed(names: list[str]) -> str:
    """a, b and c."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


class BigQueryBackendOptions(_Strict):
    # The billing project and the dataset temp tables go in, which has to
    # be in the location of the weight tables. Facts of one cloud
    # account, so neither has a default, and backend.kind = "bigquery"
    # needs both. No dataset is ever created.
    project: Optional[str] = Field(default=None, min_length=1)
    temp_dataset: Optional[str] = Field(default=None, min_length=1)
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
    # No default: the two ways of covering a cell give other weights.
    coverage: CoverageMode
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
    """What to do with source units that cannot take part in applying the
    weights, the ``[policies]`` of a pipeline config and the ``policies``
    of `apply_weights`.

    Each key is "error" or "skip". "error" raises naming the units.
    "skip" leaves them out with a warning, and the result lists them in
    its ``coverage``. ``on_unmatched`` is for units in the data that are
    not in the weight file, whose values cannot reach any target.
    ``on_zero_weight`` is for units in the weight file whose weights are
    all zero or NaN. ``on_absent_from_data`` is for units in the weight
    file that the data does not have, whose targets lose what the
    weights gave them.
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
    # Top-level TOML key, with no default: the two directions are
    # transposes over the same geometry, and either gives plausible
    # numbers where the other was meant. The grid backends support
    # per_destination only; polygon mode supports both directions
    # (per_source is the allocation direction weight files for extensive
    # quantities need).
    normalization: Normalization

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
                    "must state which target geometry version it was built "
                    "against (e.g. 'gadm-2.0')"
                )
        return self

    @model_validator(mode="after")
    def _bigquery_is_declared(self) -> "Config":
        if self.backend.kind != "bigquery":
            return self
        missing = [
            f"backend.bigquery.{name}"
            for name in ("project", "temp_dataset")
            if getattr(self.backend.bigquery, name) is None
        ]
        if missing:
            raise ValueError(
                f"backend.kind = 'bigquery' needs {' and '.join(missing)}, the "
                f"billing project and the dataset in the location of the "
                f"weight tables that temp tables go in. They belong to a cloud "
                f"account, so neither has a default."
            )
        if self.backend.bigquery.staging_uri is None:
            raise ValueError(
                "backend.bigquery.staging_uri is required: a gs:// prefix the "
                "run may write geometry staging files to. It is a site fact "
                "with no defensible library default."
            )
        if self.grid.offset != "center":
            raise ValueError(BIGQUERY_CENTRE_ONLY.format(offset=self.grid.offset))
        return self

    @model_validator(mode="after")
    def _weights_match_backend(self) -> "Config":
        names = [w.name for w in self.weights]
        seen: set[str] = set()
        for n in names:
            if n in seen:
                raise ValueError(
                    f"weights: duplicate weight name '{n}' in weights list"
                )
            seen.add(n)

        for w in self.weights:
            if w.name == "area":
                if w.raster is not None or w.table is not None:
                    raise ValueError(
                        f"weights.{w.name}: 'area' weight does not accept raster or table"
                    )
                given = [
                    key
                    for key in ("fallback", "kind", "column")
                    if getattr(w, key) is not None
                ]
                if given:
                    raise ValueError(
                        f"weights.area takes no {_listed(given)}: its total is "
                        f"the region's own area, which is never zero where the "
                        f"region has cells, and it reads no raster or table"
                    )
                continue
            missing = [key for key in ("fallback", "kind") if getattr(w, key) is None]
            if missing:
                raise ValueError(
                    f"weights.{w.name} needs "
                    f"{_listed([f'weights.{w.name}.{key}' for key in missing])}. "
                    f"fallback says what a region whose {w.name} total is zero "
                    f"gets, one of area, zero, error or nan, and kind how the "
                    f"raster is summed, point_sum or area_weighted_sum. Neither "
                    f"has a default, since a wrong one gives other weights with "
                    f"nothing said."
                )
            if self.backend.kind == "local":
                if w.raster is None:
                    raise ValueError(
                        f"weights.{w.name}: backend.kind='local' requires "
                        f"weights.{w.name}.raster"
                    )
                if w.column is not None:
                    raise ValueError(
                        f"weights.{w.name}.column names a column of a BigQuery "
                        f"table, and backend.kind='local' reads weights.{w.name}"
                        f".raster"
                    )
            else:  # bigquery
                if w.table is None or w.column is None:
                    raise ValueError(
                        f"weights.{w.name}: backend.kind='bigquery' requires "
                        f"weights.{w.name}.table and weights.{w.name}.column, the "
                        f"column of the table that holds the weight's values"
                    )
        return self


def load_config(path: str | Path) -> Config:
    """The validated `Config` of the TOML file at ``path``.

    Relative paths in `regions.path`, `source.path`, `weights[].raster`
    and `output.dir` are resolved against the config file's directory,
    not the working directory. Absolute paths, `gs://` URIs and BigQuery
    `table` references pass through unchanged. Raises
    `FileNotFoundError` if the file does not exist, and a pydantic
    `ValidationError` naming the offending key if the config is invalid.
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
    if (
        isinstance(regions, dict)
        and "path" in regions
        and isinstance(regions["path"], str)
    ):
        regions["path"] = _resolve(regions["path"])

    source = data.get("source")
    if (
        isinstance(source, dict)
        and "path" in source
        and isinstance(source["path"], str)
    ):
        source["path"] = _resolve(source["path"])

    weights = data.get("weights")
    if isinstance(weights, list):
        for w in weights:
            if isinstance(w, dict) and "raster" in w and isinstance(w["raster"], str):
                w["raster"] = _resolve(w["raster"])

    output = data.get("output")
    if isinstance(output, dict) and "dir" in output and isinstance(output["dir"], str):
        output["dir"] = _resolve(output["dir"])
