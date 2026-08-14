"""Run manifest: provenance, costs, and counts written next to the output.

A manifest is a small JSON sidecar. It carries the config hash, input
identifiers (file checksums or BQ table ids), package versions, the
backend used, fallback counts per weight, timing, and row counts. The
goal is reproducibility-by-recognition: someone returning to a Parquet
later should be able to tell what produced it, how, and how to redo it.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from segment_weights.config import Config
from segment_weights.schema import OutputSchema


_TRACKED_PACKAGES: tuple[str, ...] = (
    "segment_weights",
    "geopandas",
    "shapely",
    "pyproj",
    "exactextract",
    "rasterio",
    "pyarrow",
    "pydantic",
    "pandas",
    "numpy",
)


@dataclass
class Manifest:
    config_hash: str
    backend: str
    coverage: str
    # Grid fields are None in polygon mode (source_mode == "polygons").
    lon_convention: str | None
    grid_mode: str | None
    grid_resolution: float | None
    package_versions: dict[str, str]
    python_version: str
    # "grid" or "polygons": which geometry supplied the source units.
    source_mode: str = "grid"
    # The config's project name and description, carried verbatim so a
    # weight file states what it is for in its own words (including any
    # compatibility statement the config author wrote).
    project_name: str | None = None
    project_description: str | None = None
    # Vintage labels from the config: regions.version (the target side,
    # required in polygon mode) and source.version (the polygon source
    # side). None for grid runs that do not declare them.
    regions_version: str | None = None
    source_version: str | None = None
    # Column roles of the written frame, filled in by the backends from
    # the OutputSchema. They make a weights artifact self-describing so
    # the application layer can reconstruct the schema from the manifest
    # alone. None only in manifests written before these fields existed.
    id_fields: list[str] | None = None
    source_key_columns: list[str] | None = None
    weight_names: list[str] | None = None
    # Coverage accounting for polygon-mode artifacts: threshold, count,
    # and the source units whose intersected area falls short of their
    # own total (each with its coverage_ratio), plus whether the target
    # set was filtered (a subset run). None on artifacts that predate
    # the accounting or on grid runs. The application layer refuses
    # artifacts whose count is nonzero unless explicitly allowed: a
    # partially covered unit's weights sum to 1 over a sliver, and
    # sum-to-one gives no warning.
    partial_coverage: dict[str, Any] | None = None
    # Which side of the intersection the weights sum to 1 within
    # (per_destination or per_source). Consumers check this against the
    # operation they intend via `schema.require_normalization`; manifests
    # written before the field existed default to per_destination, which
    # is the behavior every earlier run had.
    normalization: str = "per_destination"
    inputs: dict[str, str] = field(default_factory=dict)
    # SHA256 of each written output file, keyed by filename, recorded by
    # the writer after the file lands. The fetch layer verifies a
    # downloaded weights file against this before loading it; a manifest
    # without the entry (written before the field existed, or a GCS write)
    # cannot be fetch-verified.
    outputs: dict[str, str] = field(default_factory=dict)
    fallback_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Per-case accounting of source units that could not flow through an
    # application run (unmatched / zero_weight / absent_from_data), written
    # by `validate.enforce_source_policies` under the "skip" policies.
    # Counts are always present once the check has run, so an empty dict
    # means the check did not run, not that nothing was dropped.
    source_coverage: dict[str, Any] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    timing_seconds: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p


def hash_config(cfg: Config) -> str:
    """Stable SHA256 of the validated config, in canonical JSON form."""
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file's contents. Used for tracked local inputs."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_package_versions(packages: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "unavailable"
    return out


def record_schema(manifest: Manifest, schema: OutputSchema) -> None:
    """Copy the schema's column roles into the manifest.

    Called by every backend after it resolves the OutputSchema, so a
    written artifact carries enough structure to be reloaded and applied
    without guessing column meanings.
    """
    manifest.id_fields = list(schema.id_fields)
    manifest.source_key_columns = list(schema.source_units.key_columns)
    manifest.weight_names = list(schema.weight_names)


def build_manifest(cfg: Config) -> Manifest:
    """Initialise a Manifest from the config; backend code fills in the rest."""
    return Manifest(
        config_hash=hash_config(cfg),
        backend=cfg.backend.kind,
        coverage=cfg.backend.coverage,
        project_name=cfg.project.name,
        project_description=cfg.project.description,
        lon_convention=cfg.grid.lon_convention if cfg.grid is not None else None,
        grid_mode=cfg.grid.mode if cfg.grid is not None else None,
        grid_resolution=cfg.grid.resolution if cfg.grid is not None else None,
        package_versions=collect_package_versions(),
        python_version=sys.version.split()[0],
        normalization=cfg.normalization,
        source_mode="polygons" if cfg.source is not None else "grid",
        regions_version=cfg.regions.version,
        source_version=cfg.source.version if cfg.source is not None else None,
    )
