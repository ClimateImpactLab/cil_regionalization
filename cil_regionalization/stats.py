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

``quantile_method`` names how a quantile is computed. "linear"
interpolates between members, as pandas does, and has no weighted
form. "lower_step" is the step of the historical extraction tool, the
member one below the one whose cumulative weight share reaches q, with
equal weights when none are given. "inverted_cdf" is the standard
inverse of the empirical distribution (Hyndman and Fan type 1), one
definition with or without weights.

``summation`` names how a mean is summed. "fast" is the vectorised
sum of pandas and numpy, whose last bits depend on the order the rows
arrive in. "exact" is the exactly rounded sum, which does not depend
on it.

Neither has a default. An infinite member value is refused under every
method and summation. The one infinity a statistic can have is the
``-inf`` of "lower_step" at or below its first step.
"""

from __future__ import annotations

import functools
import itertools
import math
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
_N_MEMBERS_COLUMN = "n_members"
_N_MODELS_COLUMN = "n_models"

# The computations of a quantile and of a sum, each under the name of
# what it does. Neither option has a default.
QUANTILE_METHODS = ("linear", "lower_step", "inverted_cdf")
SUMMATIONS = ("fast", "exact")
# A value both options had before they were named for what they do.
_FORMER_VALUE = "legacy"


def check_quantile_method(quantile_method: str, weighted: bool) -> None:
    """Raise on a name that is not a quantile computation, and on
    "linear" with weights, which it has no form for."""
    if quantile_method == _FORMER_VALUE:
        raise ValueError(
            "quantile_method='legacy' is not a value any more. It meant "
            "'lower_step' with weights and 'linear' without. This call has "
            + (
                "weights, so use 'lower_step'."
                if weighted
                else "no weights, so use 'linear'."
            )
        )
    if quantile_method not in QUANTILE_METHODS:
        raise ValueError(
            f"quantile_method must be one of {list(QUANTILE_METHODS)}; got "
            f"{quantile_method!r}"
        )
    if quantile_method == "linear" and weighted:
        raise ValueError(
            "quantile_method='linear' interpolates between members that "
            "count equally and has no weighted form. With weights use "
            "'lower_step' or 'inverted_cdf'."
        )


def check_summation(summation: str) -> None:
    """Raise on a name that is not a way of summing."""
    if summation == _FORMER_VALUE:
        raise ValueError(
            "summation='legacy' is not a value any more. Use 'fast', which "
            "is the same sum."
        )
    if summation not in SUMMATIONS:
        raise ValueError(
            f"summation must be one of {list(SUMMATIONS)}; got {summation!r}"
        )


_QUANTILE_NAME_RE = re.compile(r"(^|_)(q\d{1,3}|mean)$")


class GroupError(ValueError):
    """A refusal that names one identity group, held apart in ``group``
    so that a caller whose identity columns are codes can name the
    group by what the codes stand for, with `naming`."""

    def __init__(self, before: str, group: dict, after: str = "") -> None:
        self.before, self.group, self.after = before, group, after
        super().__init__(f"{before}{group}{after}")

    def naming(self, group: dict) -> "GroupError":
        """The same refusal about ``group``."""
        return type(self)(self.before, group, self.after)

    def __reduce__(self):
        # a worker process sends it back to its parent by pickling
        return (type(self), (self.before, self.group, self.after))


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
    include_n_members: bool = False,
    model_dims: Sequence[str] | None = None,
    quantile_method: str,
    summation: str,
) -> pd.DataFrame:
    """Window mean per sample member, then statistics over the pooled sample.

    ``data`` has one value per (identity, sample, time) row. Identity
    columns are all the columns not named as sample, time, value or
    weight. Statistics are computed per identity combination, and those
    columns pass through. ``window`` is an inclusive (start, end) pair on
    ``time_col``.

    ``weight_col`` names a per-member weight, constant across the
    member's time rows. Weights must be finite and non-negative, and a
    zero weight drops the member. Weights are not renormalized within
    a model, so in an
    unbalanced sample a model's total weight grows with how often it
    appears. Leaving ``weight_col`` unset with a ``gcm`` sample
    dimension warns with `UnweightedModelWeightsWarning`.

    Returns a long frame: identity columns, 'statistic', ``value_col``.
    Labels are 'mean' and 'qNN' in percent ('q05', 'q95', 'q2.5').

    ``include_n_members`` adds two integer columns after ``value_col``.
    'n_members' is the samples behind a statistic, the members with a
    window mean that is not NaN and, when weighted, a weight above zero,
    so a batch of a model counts once a batch. 'n_models' is the
    climate models among them, the distinct values of ``model_dims``,
    which it then has to name, such as ``["gcm"]``. Over fifteen batches
    of four models they are 60 and 4. ``data`` must not already have
    columns of those names.

    ``quantile_method`` is "linear", "lower_step" or "inverted_cdf",
    and has no default. "linear" interpolates between members and is
    refused with weights. "lower_step" is the left step inverse of the
    weighted distribution, as the historical extraction tool computed
    it. It returns the member one below the one whose cumulative weight
    share reaches q, with every member weighted equally when
    ``weight_col`` is unset. It differs from "linear" even when every
    weight is equal, except at 0.5, which is then the plain median, and
    a quantile below the first step is ``-inf``. "inverted_cdf"
    returns, weighted or not, the smallest member value whose
    cumulative weight share reaches q, so never ``-inf``.

    ``summation`` is "fast" or "exact", and has no default. "fast" is
    the vectorised sum, whose last bits depend on the order of the
    rows. "exact" takes the means from exactly rounded sums, whatever
    the order. An infinite member value raises, and so does a statistic
    that comes out infinite other than the step's ``-inf``.

    The statistics are float64 whatever the dtype of ``value_col``. A
    column of another dtype is widened before the window mean, and a
    nullable one comes back with NaN where it had no value.

    Raises if ``data`` already has a 'statistic' column or
    quantile-named columns. Statistics of per-unit quantiles are wrong;
    see the module docstring.
    """
    check_quantile_method(quantile_method, weighted=weight_col is not None)
    check_summation(summation)
    _reject_presummarized(data, value_col)
    model_dims = _check_counts(data, include_n_members, model_dims, sample_dims)

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
    _refuse_infinite_values(in_window, value_col)

    # A constant helper key lets the same groupby code serve inputs whose
    # every non-value column is a sample or time dimension.
    work = in_window
    if work[value_col].dtype != np.float64:
        # widened before the mean, as in `window_means`
        work = work.assign(**{value_col: _float_values(work[value_col])})
    if not identity:
        work = work.assign(_all="all")
        identity = ["_all"]

    if weight_col is not None:
        varying = work.groupby(identity + sample_dims, sort=False, observed=True)[
            weight_col
        ].nunique(dropna=False)
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
    if summation == "exact":
        # the same members in the same order, their means summed exactly
        members = _identity_groups(work, identity + sample_dims)
        gid, n_members = _grouping_ids(members)
        if n_members != len(window_mean):
            raise RuntimeError(
                f"exact window means: {n_members} members against "
                f"{len(window_mean)} from the groupby"
            )
        window_mean[value_col] = _exact_group_means(
            work[value_col].to_numpy()[gid >= 0], gid[gid >= 0], n_members
        )

    pieces = _statistic_pieces(
        window_mean,
        identity,
        value_col,
        quantiles,
        include_mean,
        weight_col,
        count_members=include_n_members,
        quantile_method=quantile_method,
        summation=summation,
    )
    out = pd.concat(pieces, ignore_index=True)
    out = out[_output_columns(identity, value_col, include_n_members)]
    if include_n_members:
        out = _with_model_counts(
            out, window_mean, identity, value_col, weight_col, model_dims
        )
    if identity == ["_all"]:
        out = out.drop(columns=["_all"])
    _refuse_infinite_statistics(out, value_col, quantile_method)
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
    order = order[counts[0] :]
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
    count_members: bool = False,
    *,
    quantile_method: str,
    summation: str,
) -> list[pd.DataFrame]:
    """One frame per statistic over each identity group's members.
    Under "linear" the quantiles come from numpy, and under
    "lower_step" from the historical step distribution, with every
    member weighted equally if there are no weights. Pass ``groups``
    if the caller already has them. ``count_members`` adds to every
    frame the number of members each group's statistics were computed
    from."""
    if groups is None:
        groups = _identity_groups(members, identity)
    gid, n_groups = _grouping_ids(groups)
    keys = members[identity].take(_first_rows(gid)).reset_index(drop=True)

    if weight_col is None and quantile_method != "lower_step":
        values = members[value_col]
        if len(gid) and gid.min() < 0:
            values, gid = values[gid >= 0], gid[gid >= 0]
        grouped = values.groupby(gid, sort=False)
        # the frame a groupby over the columns returns (it infers key
        # dtypes, e.g. object strings come back as str), from one row per
        # group, already in group order; its value column is replaced
        keys = (
            keys.assign(**{value_col: 0.0})
            .groupby(identity, sort=False, observed=True)[value_col]
            .first()
            .reset_index()
        )
        pieces: list[pd.DataFrame] = []
        floats = _float_values(values)

        def by_group(result: np.ndarray) -> pd.Series:
            out = pd.Series(result, index=range(n_groups))
            # a nullable value dtype stays nullable, as the pandas mean
            # keeps it
            if isinstance(values.dtype, pd.api.extensions.ExtensionDtype):
                out = out.astype("Float64")
            return out

        if not include_mean:
            stats = []
        elif summation == "exact":
            stats = [
                ("mean", lambda: by_group(_exact_group_means(floats, gid, n_groups)))
            ]
        else:
            stats = [("mean", lambda: grouped.mean())]
        if quantile_method == "inverted_cdf":
            table = _equal_weight_quantiles(floats, gid, n_groups, quantiles)
        else:
            table = _linear_quantiles(floats, gid, n_groups, quantiles)
        stats += [
            (_quantile_label(q), lambda i=i: by_group(table[i]))
            for i, q in enumerate(quantiles)
        ]
        if count_members:
            # count() leaves out NaN, as mean() and quantile() do
            counts = (
                grouped.count()
                .reindex(range(n_groups), fill_value=0)
                .to_numpy(dtype=np.int64)
            )
        for label, compute in stats:
            piece = keys.copy()
            # .array, not .to_numpy(): a nullable value dtype stays nullable
            piece[value_col] = compute().reindex(range(n_groups)).array
            piece[_STATISTIC_COLUMN] = label
            if count_members:
                piece[_N_MEMBERS_COLUMN] = counts
            pieces.append(piece)
        return pieces

    labels = (["mean"] if include_mean else []) + [
        _quantile_label(q) for q in quantiles
    ]
    rows: dict[str, list[tuple]] = {label: [] for label in labels}
    member_counts: list[int] = []
    order, starts, ends = _group_order(gid, n_groups)
    if weight_col is None:
        # the step with no weights given: every member counts the same
        all_values = _float_values(members[value_col])[order]
        all_weights = np.ones(len(all_values))
    else:
        all_values = members[value_col].to_numpy(dtype=float)[order]
        all_weights = members[weight_col].to_numpy(dtype=float)[order]
    del order
    key_tuples = keys.itertuples(index=False, name=None)
    for key_tuple, start, end in zip(key_tuples, starts, ends):
        values = all_values[start:end]
        weights = all_weights[start:end]
        keep = ~np.isnan(values)
        values, weights = values[keep], weights[keep]
        if len(values) == 0 and weight_col is None:
            # no member has a value, so the group has no statistic, as
            # on the other paths without weights
            for label in labels:
                rows[label].append((*key_tuple, float("nan")))
            if count_members:
                member_counts.append(0)
            continue
        if len(values) == 0 or weights.sum() == 0:
            raise GroupError(
                "weights sum to zero for identity group ",
                dict(zip(identity, key_tuple)),
                ", so no member carries any weight",
            )
        if count_members:
            member_counts.append(int(np.count_nonzero(weights)))
        if include_mean:
            if summation == "exact":
                # each product is rounded once, whatever the order, and
                # both sums are exactly rounded
                with np.errstate(over="ignore"):
                    products = values * weights
                mean = _fsum(products.tolist()) / _fsum(weights.tolist())
            else:
                mean = float(np.average(values, weights=weights))
            rows["mean"].append((*key_tuple, mean))
        exact = summation == "exact"
        if quantile_method == "inverted_cdf":
            # a member with weight zero is not in the sample
            positive = weights > 0
            picked = _inverted_cdf_quantiles(
                values[positive], weights[positive], quantiles, exact=exact
            )
        else:
            picked = _step_quantiles(values, weights, quantiles, exact=exact)
        for q, value in zip(quantiles, picked):
            rows[_quantile_label(q)].append((*key_tuple, value))
    pieces = []
    for label in labels:
        piece = pd.DataFrame(rows[label], columns=identity + [value_col])
        piece[_STATISTIC_COLUMN] = label
        if count_members:
            piece[_N_MEMBERS_COLUMN] = np.asarray(member_counts, dtype=np.int64)
        pieces.append(piece)
    return pieces


def _output_columns(
    identity: list[str], value_col: str, include_n_members: bool
) -> list[str]:
    columns = identity + [_STATISTIC_COLUMN, value_col]
    if include_n_members:
        columns.append(_N_MEMBERS_COLUMN)
    return columns


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The historical extraction tool's quantile: the left step inverse
    of the weighted empirical distribution. A quantile below the first
    step returns -inf; with all weights equal the 0.5 quantile is the
    ordinary median, both matching that tool."""
    return _step_quantiles(values, weights, [q], exact=False)[0]


def _step_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
    *,
    exact: bool,
) -> list[float]:
    """`_weighted_quantile` at every q, from one sort of the group.
    ``exact`` takes the cumulative weight shares from exact sums, so
    a share that equals q in exact arithmetic is not moved off it by
    the order the weights were added in. It also takes equal values in
    order of weight, because within equal values the step depends on
    which member comes first, and so on the order of the rows."""
    if not len(quantiles):
        return []
    equal = bool((weights == weights[0]).all())
    order = (
        np.lexsort((weights, values)) if exact else np.argsort(values, kind="stable")
    )
    ordered = values[order]
    if exact:
        # k of n equal weights are a share of k / n, rounded once
        share = _equal_shares(len(values)) if equal else _exact_shares(weights[order])
    else:
        share = np.cumsum(weights[order]) / weights.sum()
    out = []
    for q in quantiles:
        if q == 0.5 and equal:
            out.append(float(np.median(values)))
            continue
        index = int(np.searchsorted(share, q)) - 1
        out.append(float("-inf") if index < 0 else float(ordered[index]))
    return out


def _exact_shares(weights: np.ndarray) -> np.ndarray:
    """The share of the total weight that each running sum of
    ``weights`` reaches. Every sum is exact and every share is their
    ratio rounded once, so the shares do not depend on the order the
    weights are added in. The weights are finite and not negative, and
    their sum is above zero."""
    mantissa, exponent = np.frexp(weights)
    # every weight is an integer times a power of two, and in units of
    # the smallest such power it is an integer, which Python adds exactly
    nonzero = weights != 0
    low = int(exponent[nonzero].min())
    units = (mantissa * 2.0**53).astype(np.int64).tolist()
    shifts = (exponent - low).tolist()
    running = list(
        itertools.accumulate(
            unit << shift if unit else 0 for unit, shift in zip(units, shifts)
        )
    )
    total = running[-1]
    # an integer divided by an integer is rounded once, correctly
    return np.array([part / total for part in running])


@functools.lru_cache(maxsize=8)
def _equal_shares(n: int) -> np.ndarray:
    """Cumulative share of each of ``n`` equally weighted members."""
    return np.arange(1, n + 1) / n


def _inverted_cdf_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
    *,
    exact: bool = False,
) -> list[float]:
    """For each q, the smallest value whose cumulative weight share
    reaches q (Hyndman and Fan type 1). ``values`` has no NaN and every
    weight is positive. Equal values are taken in order of weight, so
    the running sum does not depend on the order of the rows. ``exact``
    takes the shares from exact sums, as `_step_quantiles` does."""
    n = len(values)
    if (weights == weights[0]).all():
        ordered = np.sort(values)
        share = _equal_shares(n)
    else:
        order = np.lexsort((weights, values))
        ordered = values[order]
        if exact:
            share = _exact_shares(weights[order])
        else:
            share = np.cumsum(weights[order])
            share = share / share[-1]
    # adding 0.0 turns -0.0 into 0.0: which of two equal zeros comes
    # first depends on the row order
    return [
        float(ordered[min(int(np.searchsorted(share, q)), n - 1)]) + 0.0
        for q in quantiles
    ]


def _equal_weight_quantiles(
    values: np.ndarray,
    gid: np.ndarray,
    n_groups: int,
    quantiles: Sequence[float],
) -> np.ndarray:
    """`_inverted_cdf_quantiles` with equal weights for every group at
    once. Returns one row per quantile and one column per group. NaN
    values are left out, and a group with no value gets NaN."""
    out = np.full((len(quantiles), n_groups), np.nan)
    keep = ~np.isnan(values)
    if not keep.all():
        values, gid = values[keep], gid[keep]
    if len(values) == 0 or not len(quantiles):
        return out
    values = values[np.lexsort((values, gid))]
    counts = np.bincount(gid, minlength=n_groups)
    starts = np.cumsum(counts) - counts
    for n in np.unique(counts[counts > 0]).tolist():
        same = np.flatnonzero(counts == n)
        share = _equal_shares(n)
        for i, q in enumerate(quantiles):
            index = min(int(np.searchsorted(share, q)), n - 1)
            out[i, same] = values[starts[same] + index]
    return out + 0.0


def _group_rows(
    values: np.ndarray, gid: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``values`` without NaN, put in group order, with the start and
    the number of values of each group."""
    keep = ~np.isnan(values)
    if not keep.all():
        values, gid = values[keep], gid[keep]
    order = np.argsort(gid, kind="stable")
    counts = np.bincount(gid, minlength=n_groups)
    return values[order], np.cumsum(counts) - counts, counts


def _linear_quantiles(
    values: np.ndarray,
    gid: np.ndarray,
    n_groups: int,
    quantiles: Sequence[float],
) -> np.ndarray:
    """The linear quantiles of every group, computed by numpy, so that
    they are those of ``numpy.quantile`` and of xarray to the last bit.
    Returns one row per quantile and one column per group. NaN values
    are left out, and a group with no value gets NaN."""
    out = np.full((len(quantiles), n_groups), np.nan)
    if not len(quantiles) or not len(values):
        return out
    values, starts, counts = _group_rows(values, gid, n_groups)
    for n in np.unique(counts[counts > 0]).tolist():
        same = np.flatnonzero(counts == n)
        # every group with n values at once, one group a row
        table = values[starts[same][:, None] + np.arange(n)]
        with np.errstate(invalid="ignore"):
            out[:, same] = np.quantile(table, list(quantiles), axis=1, method="linear")
    return out


def _fsum(values: list[float]) -> float:
    """The exactly rounded sum of finite values. Raises if it passes
    the largest float."""
    try:
        return math.fsum(values)
    except OverflowError:
        raise ValueError(
            "summation='exact': a sum overflows the largest float"
        ) from None


def _refuse_infinite_statistics(
    out: pd.DataFrame, value_col: str, quantile_method: str
) -> None:
    """Raise if a statistic is infinite, other than a "lower_step"
    quantile at or below its first step, which is ``-inf`` by its
    definition. The members are finite by then, so any other infinity
    is the arithmetic passing the largest float. NaN is left alone,
    because a group with no value has no statistic."""
    values = _float_values(out[value_col])
    infinite = np.isinf(values)
    if quantile_method == "lower_step":
        infinite &= ~(
            (out[_STATISTIC_COLUMN] != "mean").to_numpy() & (values == -np.inf)
        )
    if not infinite.any():
        return
    counts = out.loc[infinite, _STATISTIC_COLUMN].value_counts(sort=False)
    first = out.loc[infinite].iloc[0].drop(value_col).to_dict()
    raise GroupError(
        f"{int(infinite.sum())} statistics are infinite "
        f"({', '.join(f'{label}: {n}' for label, n in counts.items())}; "
        f"first: ",
        first,
        "). Every member is finite, so the arithmetic passed the largest float.",
    )


def _refuse_infinite_values(data: pd.DataFrame, value_col: str) -> None:
    """Raise if a member value is infinite, under every quantile method
    and summation, naming the first such row by its other columns."""
    infinite = np.isinf(_float_values(data[value_col]))
    if not infinite.any():
        return
    first = data.loc[infinite].iloc[0].drop(value_col).to_dict()
    raise GroupError(
        f"{int(infinite.sum())} values of {value_col!r} are infinite (first: ",
        first,
        "). An infinity in impact data means something upstream of the "
        "statistics broke.",
    )


# rows turned into Python floats at a time by `_exact_group_means`
_FSUM_CHUNK = 1_000_000


def _exact_group_means(
    values: np.ndarray, gid: np.ndarray, n_groups: int
) -> np.ndarray:
    """Mean of each group's values that are not NaN, from an exactly
    rounded sum, so the order of the rows does not matter. ``gid`` has
    no negative ids. A group with no value gets NaN."""
    out = np.full(n_groups, np.nan)
    keep = ~np.isnan(values)
    if not keep.all():
        values, gid = values[keep], gid[keep]
    if len(values) == 0:
        return out
    if (gid[1:] < gid[:-1]).any():
        order = np.argsort(gid, kind="stable")
        values, gid = values[order], gid[order]
    counts = np.bincount(gid, minlength=n_groups)
    present = np.flatnonzero(counts)
    sizes = counts[present]
    if sizes.min() == sizes.max():
        # every group the same size, the usual leaf: whole rows of a
        # table become lists in one call
        size = int(sizes[0])
        table = values.reshape(-1, size)
        step = max(1, _FSUM_CHUNK // size)
        sums: list[float] = []
        for start in range(0, len(table), step):
            sums.extend(_fsum(row) for row in table[start : start + step].tolist())
    else:
        ends = np.cumsum(sizes)
        sums = [
            _fsum(values[a:b].tolist())
            for a, b in zip((ends - sizes).tolist(), ends.tolist())
        ]
    out[present] = np.asarray(sums) / sizes
    return out


def _float_values(values: pd.Series) -> np.ndarray:
    """A value column as float64, missing values as NaN."""
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def _check_weight_column(
    data: pd.DataFrame, weight_col: str, sample_dims: list[str], reserved: list[str]
) -> None:
    if weight_col not in data.columns:
        raise ValueError(f"data is missing weight column {weight_col!r}")
    if weight_col in sample_dims or weight_col in reserved:
        raise ValueError(
            f"weight_col {weight_col!r} also names a sample, time, or value column"
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


def _check_counts(
    data: pd.DataFrame,
    include_n_members: bool,
    model_dims: Sequence[str] | None,
    sample_dims: Sequence[str],
) -> list[str] | None:
    """``model_dims`` as a list, checked against the request for counts.
    Raises if counts are asked for without it, or it without them, or
    it names a column that is not a sample dimension, or ``data``
    already has a column of a count."""
    if not include_n_members:
        if model_dims is not None:
            raise ValueError(
                "model_dims names the columns of the climate model for its "
                "count, which only include_n_members writes"
            )
        return None
    if model_dims is None or not list(model_dims):
        raise ValueError(
            "include_n_members counts the samples and the climate models "
            "behind each statistic, so model_dims has to name the sample "
            "dimensions of the climate model, such as ['gcm']"
        )
    model_dims = list(model_dims)
    stray = [d for d in model_dims if d not in sample_dims]
    if stray:
        raise ValueError(
            f"model_dims {stray} are not among sample_dims {list(sample_dims)}; "
            f"a climate model is counted among the members it pools"
        )
    for column in (_N_MEMBERS_COLUMN, _N_MODELS_COLUMN):
        if column in data.columns:
            raise ValueError(
                f"data already has a {column!r} column; include_n_members "
                f"would write a second one. Rename or drop it."
            )
    return model_dims


def _with_model_counts(
    out: pd.DataFrame,
    members: pd.DataFrame,
    identity: list[str],
    value_col: str,
    weight_col: str | None,
    model_dims: list[str],
) -> pd.DataFrame:
    """``out`` with 'n_models', the distinct climate models among the
    members of each identity group that 'n_members' counts."""
    counted = members[value_col].notna()
    if weight_col is not None:
        counted &= members[weight_col] > 0
    models = (
        members.loc[counted, identity + model_dims]
        .drop_duplicates()
        .groupby(identity, sort=False, observed=True)
        .size()
        .rename(_N_MODELS_COLUMN)
        .reset_index()
    )
    keys = out[identity].astype(object)
    found = keys.merge(
        models.astype({c: object for c in identity}), on=identity, how="left"
    )
    return out.assign(
        **{
            _N_MODELS_COLUMN: found[_N_MODELS_COLUMN]
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        }
    )


def _reject_presummarized(data: pd.DataFrame, value_col: str) -> None:
    problems: list[str] = []
    if _STATISTIC_COLUMN in data.columns:
        problems.append(f"a {_STATISTIC_COLUMN!r} column")
    quantile_like = [c for c in data.columns if _QUANTILE_NAME_RE.search(str(c))]
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
    summation: str,
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

    ``summation`` has no default. "exact" takes each mean from an
    exactly rounded sum, and "fast" is the pandas mean. An infinite
    value in a window raises.

    The mean is float64 whatever the dtype of ``value_col``. A float32
    column is widened before the mean and not after, because a float32
    mean stored as float64 still holds seven digits. A nullable column
    comes back with NaN where it had no value.
    """
    check_summation(summation)
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

    _refuse_infinite_values(
        data.loc[
            np.logical_or.reduce([data[time_col].between(lo, hi) for lo, hi in windows])
        ],
        value_col,
    )
    group = [c for c in data.columns if c not in (time_col, value_col)]
    full_key = group + [time_col]
    # The group codes are worked out once and serve the duplicate check
    # (extended by the time column, over every row) and each window.
    groups = _identity_groups(data, group) if group else None
    if (
        _has_duplicates(data, groups, [time_col])
        if group
        else data.duplicated(subset=full_key).any()
    ):
        n = int(data.duplicated(subset=full_key).sum())
        raise ValueError(
            f"data has {n} rows duplicated on {full_key}; duplicates would "
            f"distort the window mean"
        )

    if group:
        gid, n_groups = _grouping_ids(groups)
        del groups
        keys = data[group].take(_first_rows(gid)).reset_index(drop=True)
        values = data[value_col]
        if values.dtype != np.float64:
            # widened before the mean, not after: a float32 mean widened
            # afterwards is a float64 that holds seven digits
            values = pd.Series(_float_values(values), index=values.index)
        floats = values.to_numpy() if summation == "exact" else None
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
            if summation == "exact":
                # the groups of this window in order of first appearance,
                # as the groupby below hands them over
                ids = pd.unique(gid[mask])
                exact = _exact_group_means(floats[mask], gid[mask], n_groups)
                means = pd.Series(exact[ids], index=ids)
            else:
                means = values[mask].groupby(gid[mask], sort=False).mean()
            # the key frame a groupby over the columns returns (it infers
            # key dtypes), from one row per group in this window's order
            piece = (
                keys.take(means.index.to_numpy())
                .assign(**{value_col: 0.0})
                .groupby(group, sort=False, observed=True)[value_col]
                .first()
                .reset_index()
            )
            piece[value_col] = means.reset_index(drop=True)
        piece[window_col] = label
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    if out[value_col].dtype != np.float64:
        # the groupby over the columns above keeps the column's dtype,
        # and how pandas types a column assigned over another has
        # changed between its versions
        out[value_col] = _float_values(out[value_col])
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
    include_n_members: bool = False,
    model_dims: Sequence[str] | None = None,
    quantile_method: str,
    summation: str,
) -> pd.DataFrame:
    """Statistics over the pooled sample, for data reduced by `window_means`.

    The second half of `summarize_samples`. The window column is an
    ordinary identity column here. ``weight_col``,
    ``include_n_members``, ``model_dims``, ``quantile_method`` and
    ``summation`` behave the same, and the same input is refused.
    Returns identity columns, 'statistic', ``value_col``, then
    'n_members' and 'n_models' if asked for.
    """
    check_quantile_method(quantile_method, weighted=weight_col is not None)
    check_summation(summation)
    _reject_presummarized(data, value_col)
    model_dims = _check_counts(data, include_n_members, model_dims, sample_dims)

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

    _refuse_infinite_values(data, value_col)
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
        work,
        identity,
        value_col,
        quantiles,
        include_mean,
        weight_col,
        groups,
        count_members=include_n_members,
        quantile_method=quantile_method,
        summation=summation,
    )
    del groups
    out = pd.concat(pieces, ignore_index=True)
    out = out[_output_columns(identity, value_col, include_n_members)]
    if include_n_members:
        out = _with_model_counts(out, work, identity, value_col, weight_col, model_dims)
    if identity == ["_all"]:
        out = out.drop(columns=["_all"])
    _refuse_infinite_statistics(out, value_col, quantile_method)
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
    if (
        min(w.ndim, m.ndim, v.ndim) == 0
        or not w.shape[-1] == m.shape[-1] == v.shape[-1]
    ):
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
        out[:, k] = np.where(
            has_normal, -np.inf, np.where(active, m, np.inf).min(axis=1)
        )
    for k in np.flatnonzero(p == 1.0):
        out[:, k] = np.where(
            has_normal, np.inf, np.where(active, m, -np.inf).max(axis=1)
        )
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
            np.isfinite(xn)
            & (xn > lot)
            & (xn < hit)
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
