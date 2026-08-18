"""Public API surface for cil_regionalization.

What a consumer of published weights touches, in call order:

    fetch_weights       download a published artifact, verify, cache, load
    apply_weights       aggregate source-level data to the artifact's targets
    summarize_samples   window means and pooled statistics over MC draws

`window_means` and `pooled_statistics` are the two halves of
`summarize_samples`, for pipelines that reduce time per leaf before
pooling.

plus `WeightsArtifact` (the loaded weights) and, for weight generation,
`Config` / `load_config` with the `cilreg` CLI. Everything not
exported here is internal and may change without notice.

Package import does two small env-repairs so CLI, notebooks, and library
users all get a working geospatial stack without per-process incantations:

1. PROJ_LIB / PROJ_DATA pointed at this env's share/proj when one
   exists (conda layouts), or cleared when they point outside this env
   (pip venvs, where each wheel bundles its own PROJ data and
   self-locates once the variables are unset). Either stale state comes
   from inheriting another installation's environment and breaks CRS
   construction, with "Invalid projection: EPSG:4326" or "proj.db lacks
   DATABASE.LAYOUT.VERSION" depending on the direction of the mismatch.
2. The GDAL 4.0 FutureWarning about UseExceptions is suppressed (or, if
   the osgeo binding is present, UseExceptions is called explicitly).
   Notebook and CLI output stay clean of the upstream warning.
"""
from __future__ import annotations

import os as _os
import sys as _sys
import warnings as _warnings
from pathlib import Path as _Path

_proj_dir = _Path(_sys.prefix) / "share" / "proj"
if (_proj_dir / "proj.db").exists():
    _os.environ.setdefault("PROJ_DATA", str(_proj_dir))
    _os.environ["PROJ_LIB"] = str(_proj_dir)
else:
    for _var in ("PROJ_LIB", "PROJ_DATA"):
        _val = _os.environ.get(_var)
        if _val and not _val.startswith(_sys.prefix):
            del _os.environ[_var]

try:  # pragma: no cover - the osgeo binding is optional
    from osgeo import gdal as _gdal
    _gdal.UseExceptions()
except ImportError:
    _warnings.filterwarnings(
        "ignore",
        message=r".*gdal\.UseExceptions.*",
        category=FutureWarning,
    )

from cil_regionalization.apply import WeightsArtifact, apply_weights
from cil_regionalization.config import Config, SourceUnitPolicies, load_config
from cil_regionalization.fetch import clear_cache, fetch_weights, list_cached
from cil_regionalization.netcdf_io import read_netcdf_leaf
from cil_regionalization.stats import (
    UnweightedModelWeightsWarning,
    pooled_statistics,
    summarize_samples,
    window_means,
)

__all__ = [
    "Config",
    "SourceUnitPolicies",
    "UnweightedModelWeightsWarning",
    "WeightsArtifact",
    "apply_weights",
    "clear_cache",
    "fetch_weights",
    "list_cached",
    "load_config",
    "pooled_statistics",
    "read_netcdf_leaf",
    "summarize_samples",
    "window_means",
]
__version__ = "0.1.0"
