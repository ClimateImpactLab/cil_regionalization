"""Statistics over Monte Carlo samples of already aggregated data.

One pure function: window mean over time first, then the requested
statistics over the pooled sample dimensions. No file discovery, no
directory grammar, no job submission; the caller supplies a frame and
declares which columns are the sample.

Ordering is the point
---------------------
Aggregating across space and summarizing across samples do not commute.
The correct order, and the only one this module offers, is:

    1. aggregate each Monte Carlo draw spatially (the `apply` module),
    2. mean over the time window per draw,
    3. statistics over the pooled draws.

The historical mistake this design exists to prevent: per-unit quantiles
were computed upstream and then summed across units to make regional
aggregates. Quantiles are not additive; the sum of per-unit quantiles
equals the quantile of the sum only under perfect (comonotone)
dependence between units, so summed upper-tail quantiles overstate the
aggregate tail. This function refuses input that already carries a
statistic dimension so that path stays closed.

What pooling assumes
--------------------
All declared sample dimensions are flattened into one sample, and every
member counts equally. With batch and gcm as the sample dimensions, that
weights every GCM the same, which is a modelling choice rather than an
established convention (the historical system carried explicit GCM
weights). If weighted pooling is decided on, it arrives as an explicit
per-sample weight input, not a silent default.

Missing values: rows whose value is NaN drop out of the window mean, and
sample members whose entire window is NaN drop out of the pooled
statistics. A sample present for only part of the window contributes the
mean of its available years.
"""
from __future__ import annotations

import re
from typing import Sequence

import pandas as pd


# Column names that mark data as already summarized: a literal statistic
# dimension, or per-unit quantile columns in either the short (q05) or
# the legacy long (monetized_deaths_vsl_epa_scaled_q05) spelling.
_STATISTIC_COLUMN = "statistic"
_QUANTILE_NAME_RE = re.compile(r"(^|_)(q\d{1,3}|mean)$")


def summarize_samples(
    data: pd.DataFrame,
    *,
    sample_dims: Sequence[str],
    time_col: str,
    window: tuple[float, float],
    value_col: str = "value",
    quantiles: Sequence[float] = (),
    include_mean: bool = True,
) -> pd.DataFrame:
    """Window mean per sample member, then pooled statistics.

    ``data`` is target-level, one value per (identity dims, sample dims,
    time). Identity dims are every column not named in ``sample_dims``,
    ``time_col``, or ``value_col``; statistics are computed independently
    for each identity combination, so target ids and scenario dims pass
    through untouched. Nothing here knows production dimension names.

    ``window`` is an inclusive (start, end) pair on ``time_col``. The
    window mean runs first, per sample member; the requested statistics
    then pool all sample dimensions flat (see the module docstring for
    what that assumes).

    Returns a long frame: identity dims + 'statistic' + ``value_col``.
    Statistic labels are 'mean' and 'qNN' (two-digit percent, e.g. 'q05',
    'q95'; non-integer percents keep their decimals, e.g. 'q2.5').

    Raises if the input already carries a statistic dimension or
    quantile-named columns: computing statistics of statistics, or
    aggregating per-unit quantiles, is the historical error described in
    the module docstring, and there is no valid call that looks like that.
    """
    _reject_presummarized(data, value_col)

    sample_dims = list(sample_dims)
    if not sample_dims:
        raise ValueError("sample_dims must name at least one column")
    for col in sample_dims + [time_col, value_col]:
        if col not in data.columns:
            raise ValueError(f"data is missing required column {col!r}")

    quantiles = list(quantiles)
    if not include_mean and not quantiles:
        raise ValueError(
            "no statistics requested: set include_mean=True or pass quantiles"
        )
    bad_q = [q for q in quantiles if not 0.0 <= q <= 1.0]
    if bad_q:
        raise ValueError(f"quantiles must lie in [0, 1]; got {bad_q}")
    labels = (["mean"] if include_mean else []) + [
        _quantile_label(q) for q in quantiles
    ]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate statistic labels in request: {labels}")

    lo, hi = window
    if lo > hi:
        raise ValueError(f"window start {lo} is after window end {hi}")

    identity = [
        c for c in data.columns if c not in sample_dims + [time_col, value_col]
    ]

    full_key = identity + sample_dims + [time_col]
    if data.duplicated(subset=full_key).any():
        n = int(data.duplicated(subset=full_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on (identity + sample + time) "
            f"{full_key}; duplicates would distort the window mean"
        )

    in_window = data.loc[data[time_col].between(lo, hi)]
    if len(in_window) == 0:
        raise ValueError(
            f"window {window} on {time_col!r} selects no rows "
            f"(data spans {data[time_col].min()} to {data[time_col].max()})"
        )

    # A constant helper key lets the same groupby code serve inputs whose
    # every non-value column is a sample or time dimension.
    work = in_window
    if not identity:
        work = work.assign(_all="all")
        identity = ["_all"]

    window_mean = (
        work.groupby(identity + sample_dims, sort=False)[value_col]
        .mean()
        .reset_index()
    )

    grouped = window_mean.groupby(identity, sort=False)[value_col]
    pieces: list[pd.DataFrame] = []
    if include_mean:
        piece = grouped.mean().reset_index()
        piece[_STATISTIC_COLUMN] = "mean"
        pieces.append(piece)
    for q in quantiles:
        piece = grouped.quantile(q).reset_index()
        piece[_STATISTIC_COLUMN] = _quantile_label(q)
        pieces.append(piece)

    out = pd.concat(pieces, ignore_index=True)
    out = out[identity + [_STATISTIC_COLUMN, value_col]]
    if identity == ["_all"]:
        out = out.drop(columns=["_all"])
    return out.reset_index(drop=True)


def _quantile_label(q: float) -> str:
    percent = 100.0 * q
    if percent == int(percent):
        return f"q{int(percent):02d}"
    return f"q{percent:g}"


def _reject_presummarized(data: pd.DataFrame, value_col: str) -> None:
    problems: list[str] = []
    if _STATISTIC_COLUMN in data.columns:
        problems.append(f"a {_STATISTIC_COLUMN!r} column")
    quantile_like = [
        c for c in data.columns if _QUANTILE_NAME_RE.search(str(c))
    ]
    if quantile_like:
        problems.append(f"quantile-named columns {quantile_like}")
    if not problems:
        return
    raise ValueError(
        "data already carries a statistic dimension ("
        + " and ".join(problems)
        + "). Statistics must be computed from raw draws after spatial "
        "aggregation. Aggregating precomputed per-unit quantiles is the "
        "historical error this module refuses: quantiles are not additive, "
        "and summing them across units equals the quantile of the sum only "
        "under perfect comonotone dependence, so summed upper-tail "
        "quantiles overstate the aggregate tail. Recompute from the "
        "underlying draws instead."
    )


def window_means(
    data: pd.DataFrame,
    *,
    time_col: str,
    windows: Sequence[tuple[float, float]],
    value_col: str = "value",
    window_col: str = "window",
) -> pd.DataFrame:
    """Mean over each time window, per every other column combination.

    This is the first half of `summarize_samples`, split out so a
    pipeline can apply it per leaf before pooling: the window mean is
    taken independently per sample member, so reducing early and pooling
    later gives the identical statistics while collapsing the time
    dimension to the window count. It is not a statistics step. Nothing
    here crosses sample members; the ordering `summarize_samples`
    enforces (spatial aggregation, then window mean, then statistics
    over the pooled sample) is preserved, just staged.

    The output replaces ``time_col`` with ``window_col``, a label per
    window ("2040-2059" style). Windows are therefore final at this
    point: the time dimension is gone from the output, and calling this
    function on already reduced data raises instead of silently
    windowing labels. To change windows, rerun from unreduced data.
    """
    windows = [tuple(w) for w in windows]
    _reject_presummarized(data, value_col)
    if window_col in data.columns:
        raise ValueError(
            f"data already carries a {window_col!r} column: it has been "
            f"reduced and the time dimension is gone. Windows are final "
            f"before reduction; rerun from the unreduced data to change them."
        )
    for col in (time_col, value_col):
        if col not in data.columns:
            raise ValueError(f"data is missing required column {col!r}")
    if not windows:
        raise ValueError("windows must list at least one (start, end) pair")

    labels = []
    for lo, hi in windows:
        if lo > hi:
            raise ValueError(f"window start {lo} is after window end {hi}")
        labels.append(_window_label(lo, hi))
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate window labels in request: {labels}")

    group = [c for c in data.columns if c not in (time_col, value_col)]
    full_key = group + [time_col]
    if data.duplicated(subset=full_key).any():
        n = int(data.duplicated(subset=full_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on {full_key}; duplicates would "
            f"distort the window mean"
        )

    pieces: list[pd.DataFrame] = []
    for (lo, hi), label in zip(windows, labels):
        in_window = data.loc[data[time_col].between(lo, hi)]
        if len(in_window) == 0:
            raise ValueError(
                f"window ({lo}, {hi}) on {time_col!r} selects no rows "
                f"(data spans {data[time_col].min()} to {data[time_col].max()})"
            )
        piece = (
            in_window.groupby(group, sort=False)[value_col].mean().reset_index()
        )
        piece[window_col] = label
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    return out[group + [window_col, value_col]]


def _window_label(lo: float, hi: float) -> str:
    def fmt(x: float) -> str:
        return str(int(x)) if float(x) == int(x) else f"{x:g}"

    return f"{fmt(lo)}-{fmt(hi)}"


def pooled_statistics(
    data: pd.DataFrame,
    *,
    sample_dims: Sequence[str],
    value_col: str = "value",
    quantiles: Sequence[float] = (),
    include_mean: bool = True,
) -> pd.DataFrame:
    """Statistics over the flat pooled sample dimensions; no time step.

    The second half of `summarize_samples`, for input whose time
    dimension has already been reduced by `window_means` (the window
    label column is just another identity dimension here). The same
    guarantees hold: input already carrying a statistic dimension is
    refused with the same explanation, the pooled sample is flat, and
    identity dimensions pass through. Output: identity dims +
    'statistic' + ``value_col``.
    """
    _reject_presummarized(data, value_col)

    sample_dims = list(sample_dims)
    if not sample_dims:
        raise ValueError("sample_dims must name at least one column")
    for col in sample_dims + [value_col]:
        if col not in data.columns:
            raise ValueError(f"data is missing required column {col!r}")

    quantiles = list(quantiles)
    if not include_mean and not quantiles:
        raise ValueError(
            "no statistics requested: set include_mean=True or pass quantiles"
        )
    bad_q = [q for q in quantiles if not 0.0 <= q <= 1.0]
    if bad_q:
        raise ValueError(f"quantiles must lie in [0, 1]; got {bad_q}")
    labels = (["mean"] if include_mean else []) + [
        _quantile_label(q) for q in quantiles
    ]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate statistic labels in request: {labels}")

    identity = [c for c in data.columns if c not in sample_dims + [value_col]]
    full_key = identity + sample_dims
    if data.duplicated(subset=full_key).any():
        n = int(data.duplicated(subset=full_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on (identity + sample) {full_key}; "
            f"duplicates would distort the pooled statistics"
        )

    work = data
    if not identity:
        work = work.assign(_all="all")
        identity = ["_all"]

    grouped = work.groupby(identity, sort=False)[value_col]
    pieces: list[pd.DataFrame] = []
    if include_mean:
        piece = grouped.mean().reset_index()
        piece[_STATISTIC_COLUMN] = "mean"
        pieces.append(piece)
    for q in quantiles:
        piece = grouped.quantile(q).reset_index()
        piece[_STATISTIC_COLUMN] = _quantile_label(q)
        pieces.append(piece)

    out = pd.concat(pieces, ignore_index=True)
    out = out[identity + [_STATISTIC_COLUMN, value_col]]
    if identity == ["_all"]:
        out = out.drop(columns=["_all"])
    return out.reset_index(drop=True)
