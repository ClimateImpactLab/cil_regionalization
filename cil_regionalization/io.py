"""Result writers.

Writes the canonical output frame as Parquet (default), CSV, or both, and
a JSON manifest sidecar. Local paths and `gs://` URIs are supported; the
GCS path uses `gcsfs` which is part of the `[bigquery]` extra.
"""
from __future__ import annotations

from pathlib import Path

from cil_regionalization.backends.base import WeightsResult
from cil_regionalization.config import OutputFormat
from cil_regionalization.manifest import hash_file


def write_result(
    result: WeightsResult,
    output_dir: str,
    output_format: OutputFormat,
    *,
    stem: str = "weights",
) -> dict[str, str]:
    """Write `result.frame` and `result.manifest` into `output_dir`.

    Returns a dict of written paths keyed by artifact kind
    (``"parquet" | "csv" | "manifest"``). The frame and manifest filenames
    are ``<stem>.parquet|.csv`` and ``<stem>.manifest.json``.
    """
    paths: dict[str, str] = {}
    if output_dir.startswith("gs://"):
        return _write_gcs(result, output_dir, output_format, stem, paths)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if output_format in ("parquet", "both"):
        p = out / f"{stem}.parquet"
        result.frame.to_parquet(p, index=False)
        paths["parquet"] = str(p)
        result.manifest.outputs[p.name] = hash_file(p)
    if output_format in ("csv", "both"):
        p = out / f"{stem}.csv"
        result.frame.to_csv(p, index=False)
        paths["csv"] = str(p)
        result.manifest.outputs[p.name] = hash_file(p)
    p = out / f"{stem}.manifest.json"
    result.manifest.write(p)
    paths["manifest"] = str(p)
    return paths


def _write_gcs(
    result: WeightsResult,
    output_dir: str,
    output_format: OutputFormat,
    stem: str,
    paths: dict[str, str],
) -> dict[str, str]:
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    base = output_dir.rstrip("/") + "/"
    if output_format in ("parquet", "both"):
        p = base + f"{stem}.parquet"
        with fs.open(p, "wb") as f:
            result.frame.to_parquet(f, index=False)
        paths["parquet"] = p
    if output_format in ("csv", "both"):
        p = base + f"{stem}.csv"
        with fs.open(p, "w") as f:
            result.frame.to_csv(f, index=False)
        paths["csv"] = p
    p = base + f"{stem}.manifest.json"
    with fs.open(p, "w") as f:
        f.write(result.manifest.to_json())
    paths["manifest"] = p
    return paths
