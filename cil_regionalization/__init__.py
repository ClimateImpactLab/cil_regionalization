"""Public API of cil_regionalization.

`fetch_weights` downloads a published weights artifact, verifies it,
caches it and loads it. `fetch_model_weights` does the same for the
published climate model weights, and `list_cached` and `clear_cache`
show and remove cached artifacts. `WeightsArtifact` is the loaded
weights, and `WeightsArtifact.from_table` builds one from a weight table
produced elsewhere.

`read_netcdf_grid` reads a lat-lon NetCDF to the cell table
`apply_weights` takes, and `read_netcdf_leaf` reads a NetCDF leaf to a
long frame. `apply_weights` aggregates source-level data to the
artifact's targets, and `SourceUnitPolicies` says what it does with
source units it cannot use. `regroup` sums or averages targets into
groups, such as states into countries, and `target_basis` gives each
target's weight basis for the average.

`summarize_samples` takes window means and then pooled statistics over
the sample members. `window_means` and `pooled_statistics` are its two
halves, for pipelines that reduce time per leaf before pooling.
`read_model_weights` reads the climate model weights those statistics
take, as `ModelWeights`, and `UnweightedModelWeightsWarning` is the
warning given when models are pooled without them. `mixture_quantile`
is the quantile of a weighted mixture of normal distributions.

`read_gmst` reads a table of global mean temperature by member as a
`GMSTTable`. `assign_warming_levels` gives the warming level of each
member in each `Period`, as `WarmingLevels`, and its `windows_of` gives
a member's levels as `LevelWindow`.

`read_gradient_leaf`, `aggregate_gradient`, `window_gradient` and
`delta_method` carry delta-method gradients from a leaf to a mean and
standard deviation per target.

`generate_weights` makes weights from a `Config`, which `load_config`
reads from TOML. It returns a `WeightsResult` and raises `SumToOneError`
when the weights fail the sum-to-one check. `OutputSchema`,
`SourceUnits` and `GRID_CELLS` are the schema types for building an
artifact by hand. The `cilreg` command runs generation, the Monte Carlo
pipeline and fetching. Everything not exported here is internal and may
change without notice.

Importing the package repairs two parts of the environment. When this
environment's share/proj holds a proj.db, PROJ_LIB is set to it, and
PROJ_DATA too unless already set. Otherwise a PROJ_LIB or PROJ_DATA that
points outside this environment is removed, and each wheel finds its own
PROJ data. When the osgeo binding is installed, GDAL's UseExceptions is
called, and otherwise GDAL's FutureWarning about it is suppressed.
"""

# A PROJ_LIB or PROJ_DATA inherited from another installation breaks CRS
# construction, with "Invalid projection: EPSG:4326" or "proj.db lacks
# DATABASE.LAYOUT.VERSION" depending on the direction of the mismatch.
from __future__ import annotations

import importlib.metadata as _metadata
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
from cil_regionalization.backends.base import WeightsResult
from cil_regionalization.config import Config, SourceUnitPolicies, load_config
from cil_regionalization.fetch import (
    clear_cache,
    fetch_model_weights,
    fetch_weights,
    list_cached,
)
from cil_regionalization.generate import SumToOneError, generate_weights
from cil_regionalization.gradients import (
    aggregate_gradient,
    delta_method,
    read_gradient_leaf,
    window_gradient,
)
from cil_regionalization.model_weights import ModelWeights, read_model_weights
from cil_regionalization.regroup import regroup, target_basis
from cil_regionalization.netcdf_io import read_netcdf_grid, read_netcdf_leaf
from cil_regionalization.schema import GRID_CELLS, OutputSchema, SourceUnits
from cil_regionalization.stats import (
    UnweightedModelWeightsWarning,
    mixture_quantile,
    pooled_statistics,
    summarize_samples,
    window_means,
)
from cil_regionalization.warming_levels import (
    GMSTTable,
    LevelWindow,
    Period,
    WarmingLevels,
    assign_warming_levels,
    read_gmst,
)

__all__ = [
    "Config",
    "GMSTTable",
    "GRID_CELLS",
    "LevelWindow",
    "ModelWeights",
    "OutputSchema",
    "Period",
    "SourceUnitPolicies",
    "SourceUnits",
    "SumToOneError",
    "UnweightedModelWeightsWarning",
    "WarmingLevels",
    "WeightsArtifact",
    "WeightsResult",
    "aggregate_gradient",
    "apply_weights",
    "assign_warming_levels",
    "clear_cache",
    "delta_method",
    "fetch_model_weights",
    "fetch_weights",
    "generate_weights",
    "list_cached",
    "load_config",
    "mixture_quantile",
    "pooled_statistics",
    "read_gmst",
    "read_gradient_leaf",
    "read_model_weights",
    "regroup",
    "read_netcdf_grid",
    "read_netcdf_leaf",
    "summarize_samples",
    "target_basis",
    "window_gradient",
    "window_means",
]
try:
    __version__ = _metadata.version("cil_regionalization")
except _metadata.PackageNotFoundError:  # a source tree that is not installed
    __version__ = "unavailable"
