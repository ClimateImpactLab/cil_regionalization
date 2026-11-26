"""Statistics from a set of aggregated leaves.

The pipeline asks for statistics with a `ReduceRequest`. A backend
decides how to compute them. `pandas_blocks` is the one backend.
"""
from cil_regionalization.reduce.base import (
    LeafRequest,
    PandasReduceOptions,
    ReduceBackend,
    ReduceConfig,
    ReduceRequest,
    get_backend,
)

__all__ = [
    "LeafRequest",
    "PandasReduceOptions",
    "ReduceBackend",
    "ReduceConfig",
    "ReduceRequest",
    "get_backend",
]
