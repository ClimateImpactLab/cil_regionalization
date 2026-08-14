"""NetCDF leaf reader: flatten a dataset to the long form application takes.

Optional dependency, handled the way the BigQuery backend handles its
extra: importing this module is always safe, and the reader raises a
message naming ``cil_regionalization[netcdf]`` when xarray is absent.

The reader is deliberately dumb. It selects the requested variables,
flattens every dimension combination to one row, and renames the
declared region dimension to the caller's region column. It does not
sort, reindex, fill, or drop: values come out exactly as stored, NaN
included, in the file's own coordinate order. If the region coordinate
carries values the weights artifact does not know, the application
layer's coverage check reports that; papering over it here would hide
exactly the condition those checks exist to surface.

Non-dimension coordinates on the selected variables (a scalar scenario
label, for example) come along as columns and behave downstream like any
other passthrough dimension.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _import_xarray():
    try:
        import xarray
    except ImportError as e:
        raise ImportError(
            "reading NetCDF leaves requires the [netcdf] extra; "
            "pip install 'cil_regionalization[netcdf]'"
        ) from e
    return xarray


def read_netcdf_leaf(
    path: str | Path,
    *,
    variables: Sequence[str],
    region_dim: str,
    region_col: str,
    kind: str | None = None,
) -> pd.DataFrame:
    """Read one NetCDF leaf into a long frame.

    ``variables`` are the data variables to carry (one for extensive or
    intensive application; numerator and denominator for ratio).
    ``region_dim`` is the file's name for the region dimension, declared
    in config rather than assumed; it is renamed to ``region_col``, the
    weights artifact's source key column. Everything else flattens to
    ordinary columns.

    ``kind`` optionally names the aggregation kind the caller intends,
    enabling the units backstop: a variable whose ``units`` attribute
    contains a percent sign is a ratio, and summing a ratio across
    regions is meaningless, so reading it under ``kind="extensive"``
    raises. This catches exactly one failure (percent-labeled ratios
    declared extensive, like a "% GDP" levels file); it cannot catch
    ratios with percent-free unit strings, missing units attributes, or
    the same mistake arriving through parquet and csv leaves, which
    carry no units metadata. The kind declaration itself remains the
    caller's responsibility.
    """
    xr = _import_xarray()
    variables = list(variables)
    with xr.open_dataset(path) as ds:
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"NetCDF leaf {path} has no variables {missing} "
                f"(available: {sorted(ds.data_vars)})"
            )
        if kind == "extensive":
            for v in variables:
                units = str(ds[v].attrs.get("units", ""))
                if "%" in units:
                    raise ValueError(
                        f"NetCDF leaf {path}: variable {v!r} has units "
                        f"{units!r}, a ratio; summing it across regions "
                        f"under kind='extensive' would be meaningless. "
                        f"Aggregate ratios as 'intensive' with appropriate "
                        f"weights, or as 'ratio' from their numerator and "
                        f"denominator."
                    )
        dims = set()
        for v in variables:
            dims.update(ds[v].dims)
        if region_dim not in dims:
            raise ValueError(
                f"NetCDF leaf {path}: region_dim {region_dim!r} is not a "
                f"dimension of {variables} (dimensions: {sorted(dims)})"
            )
        frame = ds[variables].to_dataframe().reset_index()
    if region_dim != region_col:
        if region_col in frame.columns:
            raise ValueError(
                f"cannot rename region_dim {region_dim!r} to {region_col!r}: "
                f"the leaf already has a column named {region_col!r}"
            )
        frame = frame.rename(columns={region_dim: region_col})
    return frame
