"""Delta-method gradients: read, aggregate, and evaluate.

Some projection trees carry parameter uncertainty as gradients rather
than draws: each region-year holds the vector of derivatives of the
impact with respect to the regression coefficients, and the file
embeds the coefficient covariance matrix. The variance of any linear
aggregate follows without sampling: aggregate the gradient vectors
with the same weights as the values, then take one quadratic form
against the covariance matrix. Summing independently drawn
region-level values instead loses the cross-region correlation the
shared coefficients create, and understates the aggregate spread.

A gradient leaf is too large for the long-frame route the rest of the
package uses (coefficients times years times regions), so this module
works on arrays: one year per call, matching how the files are
chunked. The mean needs no separate file when the response is linear
in the coefficients: it is the gradient times the coefficient vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cil_regionalization.apply import WeightsArtifact


@dataclass(frozen=True)
class GradientLeaf:
    """One year of a gradient file: regions, the gradient matrix
    (coefficient by region), the coefficient covariance matrix, and
    the file's own stored variance per region when present."""

    regions: list[str]
    gradient: np.ndarray
    vcv: np.ndarray
    stored_variance: np.ndarray | None


def read_gradient_leaf(
    path,
    *,
    year: int,
    variable: str = "rebased_bcde",
    vcv_variable: str = "vcv",
    value_variable: str = "rebased",
) -> GradientLeaf:
    """Read one year of a delta-method leaf.

    Uses netCDF4 directly: the covariance matrix is stored with the
    coefficient dimension repeated, which xarray does not accept. The
    files are chunked one year per chunk, so a single-year read moves
    one chunk, not the whole file.
    """
    try:
        import netCDF4
    except ImportError as e:
        raise ImportError(
            "reading gradient leaves requires the [netcdf] extra; "
            "pip install 'cil_regionalization[netcdf]'"
        ) from e

    with netCDF4.Dataset(path) as ds:
        years = ds.variables["year"][:]
        where = np.where(years == year)[0]
        if len(where) == 0:
            raise ValueError(f"gradient leaf {path} has no year {year}")
        yi = int(where[0])
        regions = [str(r) for r in ds.variables["regions"][:]]
        var = ds.variables[variable]
        dims = var.dimensions
        if dims[:2] == ("coefficient", "year"):
            gradient = np.asarray(var[:, yi, :], dtype="float64")
        elif dims[:2] == ("year", "coefficient"):
            gradient = np.asarray(var[yi, :, :], dtype="float64")
        else:
            raise ValueError(
                f"gradient variable {variable!r} has dimensions {dims}; "
                f"expected coefficient and year first"
            )
        vcv = np.asarray(ds.variables[vcv_variable][:], dtype="float64")
        stored = None
        if value_variable in ds.variables:
            vv = ds.variables[value_variable]
            if vv.dimensions[0] == "year":
                stored = np.asarray(vv[yi, :], dtype="float64")
            else:
                stored = np.asarray(vv[:, yi], dtype="float64")
    if vcv.ndim != 2 or vcv.shape[0] != vcv.shape[1]:
        raise ValueError(f"covariance matrix has shape {vcv.shape}; expected square")
    if gradient.shape[0] != vcv.shape[0]:
        raise ValueError(
            f"gradient has {gradient.shape[0]} coefficients but the "
            f"covariance matrix is {vcv.shape[0]} by {vcv.shape[1]}"
        )
    return GradientLeaf(regions=regions, gradient=gradient, vcv=vcv,
                        stored_variance=stored)


def aggregate_gradient(
    weights: WeightsArtifact,
    regions: list[str],
    gradient: np.ndarray,
    *,
    weight: str = "pop",
    scale: np.ndarray | None = None,
    normalize: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate gradient vectors to the weight file's targets.

    ``gradient`` is coefficient by region, in the order of ``regions``.
    Every region must appear in the weight file; weight rows for other
    regions are ignored, which mirrors ``restrict_to_sources``.

    For a total (an extensive value), leave ``scale`` and ``normalize``
    alone: each region's gradient is split by the weight shares and
    summed. For a rate, pass the per-region sizes (population) as
    ``scale`` and set ``normalize=True``: each target becomes the
    size-weighted mean of its regions' gradients.

    Returns the target key frame and the aggregated gradient, target
    by coefficient. The aggregation is linear, so it commutes with
    differencing histclim and with evaluating means: aggregating a
    difference of gradients equals differencing aggregated ones.
    """
    frame = weights.frame
    key_cols = list(weights.schema.id_fields)
    share_col = f"{weight}wt"
    if share_col not in frame.columns:
        raise ValueError(
            f"weight file has no {share_col!r} column "
            f"(columns: {sorted(frame.columns)})"
        )
    index = {r: i for i, r in enumerate(regions)}
    rows = frame[frame["hierid"].isin(index)]
    covered = set(rows["hierid"])
    absent = [r for r in regions if r not in covered]
    if absent:
        raise ValueError(
            f"{len(absent)} regions carry gradients but no weight rows "
            f"(first: {absent[:3]}); the gradient slice and the weight "
            f"slice cover different region sets"
        )

    targets = rows[key_cols].drop_duplicates().reset_index(drop=True)
    tindex = {tuple(t): i for i, t in enumerate(targets.itertuples(index=False))}
    n_t, n_c = len(targets), gradient.shape[0]
    out = np.zeros((n_t, n_c))
    norm = np.zeros(n_t)
    scale_vec = np.ones(len(regions)) if scale is None else np.asarray(scale, dtype="float64")
    if scale_vec.shape != (len(regions),):
        raise ValueError(
            f"scale has shape {scale_vec.shape}; expected ({len(regions)},)"
        )
    for row in rows.itertuples(index=False):
        r = index[row.hierid]
        t = tindex[tuple(getattr(row, c) for c in key_cols)]
        w = getattr(row, share_col) * scale_vec[r]
        out[t] += w * gradient[:, r]
        norm[t] += w
    if normalize:
        if (norm == 0).any():
            raise ValueError("a target has zero total weight; cannot normalize")
        out = out / norm[:, None]
    return targets, out


def delta_method(
    aggregated: np.ndarray,
    vcv: np.ndarray,
    coefficients: np.ndarray | None = None,
) -> pd.DataFrame:
    """Mean and standard deviation per target from aggregated gradients.

    Variance is the quadratic form G V G' per target. The mean is
    G times the coefficient vector and is exact when the response is
    linear in the coefficients, which is what makes the gradients
    constant in them; omit ``coefficients`` to get only the spread.
    """
    if aggregated.shape[1] != vcv.shape[0]:
        raise ValueError(
            f"aggregated gradient has {aggregated.shape[1]} coefficients "
            f"but the covariance matrix is {vcv.shape[0]} by {vcv.shape[1]}"
        )
    variance = np.einsum("tc,cd,td->t", aggregated, vcv, aggregated)
    out = pd.DataFrame({"sd": np.sqrt(variance)})
    if coefficients is not None:
        coefficients = np.asarray(coefficients, dtype="float64")
        if coefficients.shape != (aggregated.shape[1],):
            raise ValueError(
                f"coefficient vector has shape {coefficients.shape}; "
                f"expected ({aggregated.shape[1]},)"
            )
        out.insert(0, "mean", aggregated @ coefficients)
    return out
