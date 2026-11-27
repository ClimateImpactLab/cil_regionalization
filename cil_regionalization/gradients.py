"""Reads delta-method gradient leaves, aggregates the gradients to
targets, and gives each target's mean and standard deviation.

Some projection trees carry parameter uncertainty as gradients, not
draws. Each region and year holds the derivatives of the impact with
respect to the regression coefficients, and the file embeds the
coefficient covariance matrix. The variance of a linear aggregate is
the quadratic form of its aggregated gradient against that matrix.
Summing independently drawn region-level values instead loses the
cross-region correlation the shared coefficients create, and
understates the aggregate spread. When the response is linear in the
coefficients, the mean is the gradient times the coefficient vector.

The functions work on arrays, one year of a leaf per call, not on the
long frames the rest of the package uses.
"""

# A gradient leaf, coefficients by years by regions, is too large for
# the long frames, and the files are chunked one year per chunk.
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cil_regionalization.apply import WeightsArtifact


@dataclass(frozen=True)
class GradientLeaf:
    """One year of a gradient file, with the region ids, the gradient
    matrix (coefficient by region), the coefficient covariance matrix,
    and the file's own stored variance per region, or None when the file
    has none."""

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
    """One year of a delta-method leaf as a `GradientLeaf`.

    ``variable`` holds the gradients, with coefficient and year as its
    first two dimensions in either order. ``vcv_variable`` holds the
    covariance matrix, and ``value_variable`` the stored variance, read
    when present. Raises if the file has no ``year``, if the gradient's
    first dimensions are not coefficient and year, or if the covariance
    matrix is not square or does not match the gradient's coefficients.
    """
    # netCDF4 rather than xarray, since the covariance matrix is stored
    # with the coefficient dimension repeated, which xarray does not
    # accept. The files are chunked one year per chunk, so a single-year
    # read moves one chunk, not the whole file.
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
    return GradientLeaf(
        regions=regions, gradient=gradient, vcv=vcv, stored_variance=stored
    )


def aggregate_gradient(
    weights: WeightsArtifact,
    regions: list[str],
    gradient: np.ndarray,
    *,
    weight: str = "pop",
    scale: np.ndarray | None = None,
    normalize: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """The target key frame and the gradient aggregated to the weight
    file's targets, target by coefficient.

    ``gradient`` is coefficient by region, in the order of ``regions``,
    and the shares are the column ``<weight>wt``. Weight rows for other
    regions are ignored, as with ``restrict_to_sources``. For a total,
    an extensive value, leave ``scale`` and ``normalize`` alone, and
    each region's gradient is split by the weight shares and summed. For
    a rate, pass the per-region sizes (population) as ``scale`` and set
    ``normalize=True``, and each target becomes the size-weighted mean
    of its regions' gradients. The aggregation is linear, so aggregating
    a difference of gradients, such as one against histclim, equals
    differencing the aggregated ones.

    Raises if the share column is missing, if a region has no weight
    rows, if ``scale`` is not one value per region, or if a target has
    zero total weight under ``normalize``.
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
    scale_vec = (
        np.ones(len(regions)) if scale is None else np.asarray(scale, dtype="float64")
    )
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


def window_gradient(
    gradients: np.ndarray,
    years,
    window: tuple[float, float],
) -> np.ndarray:
    """The mean of stacked yearly gradients over the years of
    ``window``, both ends included, as in `window_means`.

    ``gradients`` has year on its first axis, in the order of
    ``years``, and its other axes pass through. Stacked leaves (year,
    coefficient, region) and stacked aggregates (year, target,
    coefficient) both work, and windowing before or after
    `aggregate_gradient` gives the same answer. NaN entries drop out of
    the mean as NaN rows do in `window_means`. Raises if the window
    selects no year or starts after it ends, if ``years`` has
    duplicates, or if its length does not match the first axis.
    """
    gradients = np.asarray(gradients, dtype="float64")
    years = np.asarray(years)
    if years.ndim != 1 or gradients.ndim < 1 or gradients.shape[0] != len(years):
        raise ValueError(
            f"gradients have leading axis {gradients.shape[:1]} but "
            f"{years.size} years were given"
        )
    if len(np.unique(years)) != len(years):
        raise ValueError("years has duplicates; they would distort the window mean")
    lo, hi = window
    if lo > hi:
        raise ValueError(f"window start {lo} is after window end {hi}")
    inside = (years >= lo) & (years <= hi)
    if not inside.any():
        raise ValueError(
            f"window ({lo}, {hi}) selects no years "
            f"(years span {years.min()} to {years.max()})"
        )
    with warnings.catch_warnings():
        # all-NaN slices come back NaN, which is the answer we want
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(gradients[inside], axis=0)


def delta_method(
    aggregated: np.ndarray,
    vcv: np.ndarray,
    coefficients: np.ndarray | None = None,
) -> pd.DataFrame:
    """A frame of the mean and standard deviation per target from
    aggregated gradients, target by coefficient.

    The variance is the quadratic form G V G' of each target. The mean
    is G times ``coefficients``, exact when the response is linear in
    the coefficients. Without ``coefficients`` the frame has only 'sd'.
    Raises if the shapes of the inputs do not match.
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
