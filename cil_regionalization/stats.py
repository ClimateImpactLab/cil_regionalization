"""Statistics over Monte Carlo samples of already aggregated data.

One function: mean over the time window first, then the statistics over
the pooled samples. The caller supplies a frame and says which columns
are the sample.

The order matters, because aggregating across space and summarizing
across samples do not commute. Each draw is aggregated spatially first,
then averaged over the window, then pooled. Quantiles are not additive:
summing per-unit quantiles overstates the aggregate tail unless the
units move together, which is the mistake this closes off. Input that
already carries a statistic dimension is refused.

Pooling flattens every declared sample dimension into one sample, each
member counting equally. ``weight_col`` names a per-member weight
instead, which is how the SMME weights enter.

NaN values drop out of the window mean, and a member whose whole window
is NaN drops out of the statistics.
"""
from __future__ import annotations

import re
import warnings
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from cil_regionalization.apply import _dense_ok, _distinct_count


class UnweightedModelWeightsWarning(UserWarning):
    """Pooling climate models without weights.

    Suppress with
    ``warnings.filterwarnings("ignore", category=UnweightedModelWeightsWarning)``.
    """


def _warn_if_unweighted_gcm(sample_dims: list[str], weight_col: str | None) -> None:
    if weight_col is None and "gcm" in sample_dims:
        warnings.warn(
            "sample_dims includes 'gcm' and weight_col is not set. This "
            "weights every climate model equally, which is not the "
            "convention Climate Impact Lab projections use. See 'Climate "
            "model weights' in the README.",
            UnweightedModelWeightsWarning,
            stacklevel=3,
        )


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
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Window mean per sample member, then statistics over the pooled sample.

    ``data`` has one value per (identity, sample, time) row. Identity
    columns are all the columns not named as sample, time, value or
    weight. Statistics are computed per identity combination, and those
    columns pass through. ``window`` is an inclusive (start, end) pair on
    ``time_col``.

    ``weight_col`` names a per-member weight, constant across the
    member's time rows. Weights must be finite and non-negative, and a
    zero weight drops the member. Weighted quantiles are the left step
    inverse of the weighted distribution, as the historical extraction
    tool computed them, not the interpolated quantiles of the unweighted
    path. The two differ even when every weight is equal, except at 0.5,
    which is then the plain median. A quantile below the first step is
    ``-inf``. Weights are not renormalized within a model, so in an
    unbalanced sample a model's total weight grows with how often it
    appears. Leaving ``weight_col`` unset with a ``gcm`` sample
    dimension warns with `UnweightedModelWeightsWarning`.

    Returns a long frame: identity columns, 'statistic', ``value_col``.
    Labels are 'mean' and 'qNN' in percent ('q05', 'q95', 'q2.5').

    Raises if ``data`` already has a 'statistic' column or
    quantile-named columns. Statistics of per-unit quantiles are wrong;
    see the module docstring.
    """
    _reject_presummarized(data, value_col)

    sample_dims = list(sample_dims)
    if not sample_dims:
        raise ValueError("sample_dims must name at least one column")
    for col in sample_dims + [time_col, value_col]:
        if col not in data.columns:
            raise ValueError(f"data is missing required column {col!r}")
    if weight_col is not None:
        _check_weight_column(data, weight_col, sample_dims, [time_col, value_col])
    _warn_if_unweighted_gcm(sample_dims, weight_col)

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

    drop = sample_dims + [time_col, value_col]
    if weight_col is not None:
        drop = drop + [weight_col]
    identity = [c for c in data.columns if c not in drop]

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

    if weight_col is not None:
        varying = (
            work.groupby(identity + sample_dims, sort=False, observed=True)[
                weight_col
            ].nunique(dropna=False)
        )
        if (varying > 1).any():
            n = int((varying > 1).sum())
            raise ValueError(
                f"weight column {weight_col!r} varies within {n} sample "
                f"members; a member's weight must be constant across its "
                f"time rows"
            )

    agg = {value_col: (value_col, "mean")}
    if weight_col is not None:
        agg[weight_col] = (weight_col, "first")
    window_mean = (
        work.groupby(identity + sample_dims, sort=False, observed=True)
        .agg(**agg)
        .reset_index()
    )

    pieces = _statistic_pieces(
        window_mean, identity, value_col, quantiles, include_mean, weight_col
    )
    out = pd.concat(pieces, ignore_index=True)
    out = out[identity + [_STATISTIC_COLUMN, value_col]]
    if identity == ["_all"]:
        out = out.drop(columns=["_all"])
    return out.reset_index(drop=True)


class _Groups(NamedTuple):
    """Identity groups of a frame's rows, from `_identity_groups`."""

    ids: np.ndarray  # int64 per row, in [0, n), numbered by first appearance
    n: int
    missing: np.ndarray | None  # rows with a missing identity value, or None


def _key_codes(
    frame: pd.DataFrame,
    cols: list[str],
    codes: np.ndarray | None = None,
    card: int = 1,
) -> tuple[np.ndarray, int, np.ndarray | None]:
    """One int64 code per row, equal exactly when the rows agree on ``cols``.

    Categorical columns use their stored codes, and each column is folded
    into the running code in place, so only one column's codes exist at a
    time. That is what keeps memory down at hundreds of millions of rows.
    Missing values count as equal. Rows with one in any column come back
    as a mask, or None. Passing ``codes`` and ``card`` extends an existing
    code without modifying it.
    """
    n = len(frame)
    owned = codes is None
    missing = None
    for col in cols:
        values = frame[col]
        stored = isinstance(values.dtype, pd.CategoricalDtype)
        if stored:
            c = values.cat.codes.to_numpy()  # the frame's own codes: never written
            k = len(values.cat.categories) + 1
        else:
            c, uniques = pd.factorize(values)
            k = len(uniques) + 1
        na = c < 0
        if na.any():
            missing = na if missing is None else np.logical_or(missing, na, out=missing)
        del na
        if codes is None:
            # missing values (-1) become 0 below
            codes = c.astype(np.int64, copy=stored)
            card = k
        else:
            if not _dense_ok(card * k, n):
                codes, uniques = pd.factorize(codes)
                card = len(uniques)
                owned = True
            if owned:
                codes *= k
            else:
                codes = codes * k
                owned = True
            codes += c
            card *= k
        codes += 1
        del c
    if codes is None:
        codes = np.zeros(n, dtype=np.int64)
    return codes, card, missing


def _identity_groups(frame: pd.DataFrame, identity: list[str]) -> _Groups:
    """Dense group ids over the identity columns, in order of first
    appearance, the order ``groupby(sort=False)`` yields groups in.
    Rows missing an identity value keep their own groups here; the
    grouping drops them (`_grouping_ids`), the duplicate check does not."""
    codes, _, missing = _key_codes(frame, identity)
    ids, uniques = pd.factorize(codes)
    return _Groups(ids.astype(np.int64, copy=False), len(uniques), missing)


def _has_duplicates(
    frame: pd.DataFrame, groups: _Groups, sample_dims: list[str]
) -> bool:
    """Whether two rows agree on identity and sample dimensions, missing
    values equal, as ``DataFrame.duplicated(subset=...).any()``."""
    codes, card, _ = _key_codes(frame, sample_dims, groups.ids, groups.n)
    return _distinct_count(codes, card) < len(frame)


def _grouping_ids(groups: _Groups) -> tuple[np.ndarray, int]:
    """Group id per row with the groups ``groupby(dropna=True)`` drops
    (any identity value missing) removed: those rows get -1 and the rest
    are renumbered, keeping the order of first appearance."""
    if groups.missing is None:
        return groups.ids, groups.n
    keep = np.zeros(groups.n, dtype=bool)
    keep[groups.ids[~groups.missing]] = True
    remap = np.cumsum(keep) - 1
    gid = remap[groups.ids]
    gid[groups.missing] = -1
    return gid, int(keep.sum())


def _first_rows(gid: np.ndarray) -> np.ndarray:
    """Position of each group's first row, in group order. Groups are
    numbered by first appearance, so a group starts where the running
    maximum of the ids steps up."""
    running = np.maximum.accumulate(gid) if len(gid) else gid
    return np.flatnonzero(np.diff(running, prepend=-1) > 0)


def _group_order(gid: np.ndarray, n_groups: int):
    """Rows ordered by group, original order within a group (the order a
    groupby hands each group over in), without the -1 rows; and each
    group's [start, end) in that order."""
    n = len(gid)
    if (n_groups + 1) * n < 2**62:
        # one key per row, unique, so any sort is the stable order; a
        # stable sort of the ids alone is several times slower
        key = gid * n
        key += np.arange(n)
        order = np.argsort(key)
        del key
    else:
        order = np.argsort(gid, kind="stable")
    counts = np.bincount(gid + 1, minlength=n_groups + 1)
    order = order[counts[0]:]
    ends = np.cumsum(counts[1:])
    return order, ends - counts[1:], ends


def _statistic_pieces(
    members: pd.DataFrame,
    identity: list[str],
    value_col: str,
    quantiles: list[float],
    include_mean: bool,
    weight_col: str | None,
    groups: _Groups | None = None,
) -> list[pd.DataFrame]:
    """One frame per statistic over each identity group's members.
    Unweighted statistics come from pandas, weighted ones from the
    historical step distribution. Pass ``groups`` if the caller already
    has them."""
    if groups is None:
        groups = _identity_groups(members, identity)
    gid, n_groups = _grouping_ids(groups)
    keys = members[identity].take(_first_rows(gid)).reset_index(drop=True)

    if weight_col is None:
        values = members[value_col]
        if len(gid) and gid.min() < 0:
            values, gid = values[gid >= 0], gid[gid >= 0]
        grouped = values.groupby(gid, sort=False)
        # the frame a groupby over the columns returns (it infers key
        # dtypes, e.g. object strings come back as str), from one row per
        # group, already in group order; its value column is replaced
        keys = (keys.assign(**{value_col: 0.0})
                .groupby(identity, sort=False, observed=True)[value_col]
                .first().reset_index())
        pieces: list[pd.DataFrame] = []
        stats = [("mean", lambda: grouped.mean())] if include_mean else []
        stats += [(_quantile_label(q), lambda q=q: grouped.quantile(q))
                  for q in quantiles]
        for label, compute in stats:
            piece = keys.copy()
            piece[value_col] = compute().reindex(range(n_groups)).to_numpy()
            piece[_STATISTIC_COLUMN] = label
            pieces.append(piece)
        return pieces

    labels = (["mean"] if include_mean else []) + [
        _quantile_label(q) for q in quantiles
    ]
    rows: dict[str, list[tuple]] = {label: [] for label in labels}
    order, starts, ends = _group_order(gid, n_groups)
    all_values = members[value_col].to_numpy(dtype=float)[order]
    all_weights = members[weight_col].to_numpy(dtype=float)[order]
    del order
    key_tuples = keys.itertuples(index=False, name=None)
    for key_tuple, start, end in zip(key_tuples, starts, ends):
        values = all_values[start:end]
        weights = all_weights[start:end]
        keep = ~np.isnan(values)
        values, weights = values[keep], weights[keep]
        if len(values) == 0 or weights.sum() == 0:
            raise ValueError(
                f"weights sum to zero for identity group {dict(zip(identity, key_tuple))}; "
                f"no member carries any weight"
            )
        if include_mean:
            rows["mean"].append((*key_tuple, float(np.average(values, weights=weights))))
        for q in quantiles:
            rows[_quantile_label(q)].append(
                (*key_tuple, _weighted_quantile(values, weights, q))
            )
    pieces = []
    for label in labels:
        piece = pd.DataFrame(rows[label], columns=identity + [value_col])
        piece[_STATISTIC_COLUMN] = label
        pieces.append(piece)
    return pieces


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The historical extraction tool's quantile: the left step inverse
    of the weighted empirical distribution. A quantile below the first
    step returns -inf; with all weights equal the 0.5 quantile is the
    ordinary median, both matching that tool."""
    if q == 0.5 and np.all(weights == weights[0]):
        return float(np.median(values))
    order = np.argsort(values, kind="stable")
    values = values[order]
    pp = np.cumsum(weights[order]) / weights.sum()
    index = int(np.searchsorted(pp, q)) - 1
    if index < 0:
        return float("-inf")
    return float(values[index])


def _check_weight_column(
    data: pd.DataFrame, weight_col: str, sample_dims: list[str], reserved: list[str]
) -> None:
    if weight_col not in data.columns:
        raise ValueError(f"data is missing weight column {weight_col!r}")
    if weight_col in sample_dims or weight_col in reserved:
        raise ValueError(
            f"weight_col {weight_col!r} also names a sample, time, or "
            f"value column"
        )
    w = data[weight_col]
    if w.isna().any():
        raise ValueError(
            f"weight column {weight_col!r} has missing values; a member "
            f"with no weight must carry an explicit 0 to be excluded"
        )
    w = w.to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise ValueError(f"weight column {weight_col!r} has non-finite values")
    if (w < 0).any():
        raise ValueError(f"weight column {weight_col!r} has negative values")


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
    """Mean over each time window, per combination of the other columns.

    The first half of `summarize_samples`, split out so a pipeline can
    reduce each leaf before pooling. It never crosses sample members, so
    reducing early and pooling later gives the same statistics.

    ``time_col`` is replaced by ``window_col``, labelled like "2040-2059",
    or by the year alone for a one-year window. Data that already has
    ``window_col`` is refused: its time dimension is gone, so to change
    windows rerun from the unreduced data. Rows repeated on every column
    but ``value_col`` are refused too, in any year, including years no
    window covers.
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
    # The group codes are worked out once and serve the duplicate check
    # (extended by the time column, over every row) and each window.
    groups = _identity_groups(data, group) if group else None
    if (_has_duplicates(data, groups, [time_col]) if group
            else data.duplicated(subset=full_key).any()):
        n = int(data.duplicated(subset=full_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on {full_key}; duplicates would "
            f"distort the window mean"
        )

    if group:
        gid, _ = _grouping_ids(groups)
        del groups
        keys = data[group].take(_first_rows(gid)).reset_index(drop=True)
        values = data[value_col]
    pieces: list[pd.DataFrame] = []
    for (lo, hi), label in zip(windows, labels):
        mask = data[time_col].between(lo, hi).to_numpy()
        if not mask.any():
            raise ValueError(
                f"window ({lo}, {hi}) on {time_col!r} selects no rows "
                f"(data spans {data[time_col].min()} to {data[time_col].max()})"
            )
        if group:
            mask = mask & (gid >= 0)
        if not group or not mask.any():
            # no group columns (groupby raises), or every row in the window
            # misses a key: the groupby over the columns, as it always ran
            piece = (
                data.loc[data[time_col].between(lo, hi)]
                .groupby(group, sort=False, observed=True)[value_col]
                .mean()
                .reset_index()
            )
        else:
            means = values[mask].groupby(gid[mask], sort=False).mean()
            # the key frame a groupby over the columns returns (it infers
            # key dtypes), from one row per group in this window's order
            piece = (keys.take(means.index.to_numpy())
                     .assign(**{value_col: 0.0})
                     .groupby(group, sort=False, observed=True)[value_col]
                     .first().reset_index())
            piece[value_col] = means.reset_index(drop=True)
        piece[window_col] = label
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    return out[group + [window_col, value_col]]


def _window_label(lo: float, hi: float) -> str:
    def fmt(x: float) -> str:
        return str(int(x)) if float(x) == int(x) else f"{x:g}"

    # A single-year window is labeled by the year alone: "2099", not
    # "2099-2099". It reads as what it is, the end-year products the
    # published figures show.
    if lo == hi:
        return fmt(lo)
    return f"{fmt(lo)}-{fmt(hi)}"


def pooled_statistics(
    data: pd.DataFrame,
    *,
    sample_dims: Sequence[str],
    value_col: str = "value",
    quantiles: Sequence[float] = (),
    include_mean: bool = True,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Statistics over the pooled sample, for data reduced by `window_means`.

    The second half of `summarize_samples`. The window column is an
    ordinary identity column here. ``weight_col`` behaves the same,
    including its different quantile definition, and the same input is
    refused. Returns identity columns, 'statistic', ``value_col``.
    """
    _reject_presummarized(data, value_col)

    sample_dims = list(sample_dims)
    if not sample_dims:
        raise ValueError("sample_dims must name at least one column")
    for col in sample_dims + [value_col]:
        if col not in data.columns:
            raise ValueError(f"data is missing required column {col!r}")
    if weight_col is not None:
        _check_weight_column(data, weight_col, sample_dims, [value_col])
    _warn_if_unweighted_gcm(sample_dims, weight_col)

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

    drop = sample_dims + [value_col]
    if weight_col is not None:
        drop = drop + [weight_col]
    identity = [c for c in data.columns if c not in drop]
    full_key = identity + sample_dims
    # The identity codes are worked out once and serve both the duplicate
    # check (extended by the sample dimensions) and the grouping below.
    groups = _identity_groups(data, identity)
    if _has_duplicates(data, groups, sample_dims):
        n = int(data.duplicated(subset=full_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on (identity + sample) {full_key}; "
            f"duplicates would distort the pooled statistics"
        )

    work = data
    if not identity:
        work = work.assign(_all="all")
        identity = ["_all"]

    pieces = _statistic_pieces(
        work, identity, value_col, quantiles, include_mean, weight_col, groups
    )
    del groups
    out = pd.concat(pieces, ignore_index=True)
    out = out[identity + [_STATISTIC_COLUMN, value_col]]
    if identity == ["_all"]:
        out = out.drop(columns=["_all"])
    return out.reset_index(drop=True)


def mixture_quantile(weights, means, variances, q):
    """Quantile of a weighted mixture of normal distributions.

    The last axis of ``weights``, ``means`` and ``variances`` is the
    mixture component. Any leading axes (regions, say) are solved
    independently and broadcast against each other, so one weight
    vector can serve every region. Weights are rescaled to sum to one
    per region, and components with zero weight are ignored. A zero
    variance component is a point mass at its mean.

    The quantile is the smallest x whose mixture CDF reaches ``q``.
    ``q`` may be a scalar or an array. The result has the leading shape
    followed by the shape of ``q``, and is a plain float when both are
    scalar. ``q = 0`` and ``q = 1`` return the ends of the support:
    infinite when any weighted component has positive variance,
    otherwise the smallest or largest point mass.
    """
    w, m, v = (np.asarray(a, dtype="float64") for a in (weights, means, variances))
    if min(w.ndim, m.ndim, v.ndim) == 0 or not w.shape[-1] == m.shape[-1] == v.shape[-1]:
        raise ValueError(
            f"weights, means and variances must have a component axis of "
            f"the same length; got shapes {w.shape}, {m.shape}, {v.shape}"
        )
    w, m, v = np.broadcast_arrays(w, m, v)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative")
    total = w.sum(axis=-1, keepdims=True)
    if (total == 0).any():
        raise ValueError(f"weights sum to zero for {int((total == 0).sum())} regions")
    active = w > 0
    if not np.isfinite(m[active]).all():
        raise ValueError("means must be finite")
    if not np.isfinite(v[active]).all() or (v[active] < 0).any():
        raise ValueError("variances must be finite and non-negative")

    q_arr = np.asarray(q, dtype="float64")
    if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 1).any():
        raise ValueError(f"quantiles must lie in [0, 1]; got {q}")

    lead = w.shape[:-1]
    n_c = w.shape[-1]
    # zero weight components may hold NaN; give them harmless values
    w = (w / total).reshape(-1, n_c)
    m = np.where(active, m, 0.0).reshape(-1, n_c)
    sd = np.sqrt(np.where(active, v, 0.0)).reshape(-1, n_c)
    out = _mixture_quantiles(w, m, sd, active.reshape(-1, n_c), q_arr.ravel())
    out = out.reshape(lead + q_arr.shape)
    return float(out) if out.ndim == 0 else out


def _mixture_quantiles(w, m, sd, active, p):
    """Quantiles p for each row of a (row, component) mixture."""
    n_r = w.shape[0]
    out = np.empty((n_r, len(p)))
    has_normal = (active & (sd > 0)).any(axis=1)
    for k in np.flatnonzero(p == 0.0):
        out[:, k] = np.where(has_normal, -np.inf, np.where(active, m, np.inf).min(axis=1))
    for k in np.flatnonzero(p == 1.0):
        out[:, k] = np.where(has_normal, np.inf, np.where(active, m, -np.inf).max(axis=1))
    inner = np.flatnonzero((p > 0.0) & (p < 1.0))
    if len(inner) == 0:
        return out

    # component quantiles, row by quantile by component
    pk = p[inner]
    comp = m[:, None, :] + sd[:, None, :] * ndtri(pk)[None, :, None]
    lo = np.where(active[:, None, :], comp, np.inf).min(axis=2).ravel()
    hi = np.where(active[:, None, :], comp, -np.inf).max(axis=2).ravel()
    x = (w[:, None, :] * comp).sum(axis=2).ravel()
    tol = 4 * np.finfo(float).eps * np.maximum(np.abs(lo), np.abs(hi))
    result = hi.copy()
    rows = np.repeat(np.arange(n_r), len(inner))
    probs = np.tile(pk, n_r)
    # above the median, solve on the upper tail to keep precision near 1
    upper = probs > 0.5
    step_old = hi - lo
    todo = np.flatnonzero(hi - lo > tol)

    for _ in range(200):
        if len(todo) == 0:
            break
        r = rows[todo]
        g, dens = _mixture_gap(w[r], m[r], sd[r], x[todo], probs[todo], upper[todo])
        xt, lot, hit = x[todo], lo[todo], hi[todo]
        hit = np.where(g >= 0, xt, hit)
        lot = np.where(g < 0, xt, lot)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = g / dens
        xn = xt - step
        newton = (
            np.isfinite(xn) & (xn > lot) & (xn < hit)
            & (np.abs(step) <= 0.5 * step_old[todo])
        )
        # a step this small means x is already the answer, even when
        # rounding puts the Newton point on the bracket edge
        tl = tol[todo]
        small = np.isfinite(step) & (np.abs(step) <= tl)
        xn = np.where(newton | small, xn, lot + 0.5 * (hit - lot))
        step_old[todo] = np.where(newton, np.abs(step), 0.5 * (hit - lot))
        exact = g == 0
        closed = hit - lot <= tl
        done = exact | closed | small
        result[todo] = np.where(exact, xt, np.where(small, xn, hit))
        x[todo], lo[todo], hi[todo] = xn, lot, hit
        todo = todo[~done]
    if len(todo):
        raise RuntimeError(f"mixture quantile did not converge for {len(todo)} cases")

    out[:, inner] = result.reshape(n_r, len(inner))
    return out


def _mixture_gap(w, m, sd, x, p, upper):
    """A function of x that is increasing and crosses zero at the
    quantile, plus its derivative, the mixture density."""
    normal = sd > 0
    s = np.where(normal, sd, 1.0)
    z = (x[:, None] - m) / s
    # mass below x, or above x for the upper tail cases
    side = np.where(upper[:, None], -z, z)
    atom = np.where(upper[:, None], x[:, None] < m, x[:, None] >= m)
    mass = (w * np.where(normal, ndtr(side), atom)).sum(axis=1)
    gap = np.where(upper, (1.0 - p) - mass, mass - p)
    dens = (w * np.where(normal, np.exp(-0.5 * z * z) / s, 0.0)).sum(axis=1)
    return gap, dens / np.sqrt(2 * np.pi)
