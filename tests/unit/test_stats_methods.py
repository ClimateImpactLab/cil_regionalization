"""The ``quantile_method`` and ``summation`` options.

These tests cover the three quantile computations and the two sums by
name. Neither option has a default. `_pooled` gives a test that names
none the pair every release before the names gave it: "lower_step"
with weights and "linear" without, and the "fast" sum.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from cil_regionalization import stats
from cil_regionalization.stats import (
    pooled_statistics,
    summarize_samples,
    window_means,
)

QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]


def _members(values, weights=None, target="t") -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target": target,
            "draw": [f"d{i}" for i in range(len(values))],
            "value": np.asarray(values, dtype=float),
        }
    )
    if weights is not None:
        frame["w"] = np.asarray(weights, dtype=float)
    return frame


def _plain(weighted: bool) -> str:
    """The quantile computation releases before the names used."""
    return "lower_step" if weighted else "linear"


def _pooled(frame, quantiles, **options) -> dict[str, float]:
    options.setdefault("quantile_method", _plain("w" in frame.columns))
    options.setdefault("summation", "fast")
    out = pooled_statistics(
        frame,
        sample_dims=["draw"],
        quantiles=quantiles,
        weight_col="w" if "w" in frame.columns else None,
        **options,
    )
    return dict(zip(out["statistic"], out["value"]))


def _random_frame(seed: int, n_targets: int = 12, n_draws: int = 40) -> pd.DataFrame:
    """Several targets and two scenarios, with tied values, missing
    values and one weight per draw, three of them zero."""
    rng = np.random.default_rng(seed)
    draw_weight = rng.uniform(0.1, 1.0, n_draws)
    draw_weight[[3, 17, 29]] = 0.0
    rows = [
        (f"t{t}", scenario, f"d{d}", draw_weight[d])
        for t in range(n_targets)
        for scenario in ("low", "high")
        for d in range(n_draws)
    ]
    frame = pd.DataFrame(rows, columns=["target", "scenario", "draw", "w"])
    # one decimal, so many values tie; some are exactly zero
    frame["value"] = np.round(rng.normal(size=len(frame)), 1)
    frame.loc[rng.random(len(frame)) < 0.05, "value"] = np.nan
    return frame


def _timed(seed: int = 3) -> pd.DataFrame:
    """Six targets, five draws, twenty years."""
    rng = np.random.default_rng(seed)
    rows = [
        (f"t{t}", f"d{d}", year)
        for t in range(6)
        for d in range(5)
        for year in range(2040, 2060)
    ]
    frame = pd.DataFrame(rows, columns=["target", "draw", "year"])
    frame["value"] = rng.normal(size=len(frame)) * 1e3
    return frame


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in frame.columns if c not in ("value", "n_members")]
    return frame.sort_values(keys).reset_index(drop=True)


class TestOptionNames:
    def test_unknown_names_are_refused(self):
        frame = _members([1.0, 2.0, 3.0])
        with pytest.raises(
            ValueError,
            match=r"quantile_method must be one of \['linear', 'lower_step', "
            r"'inverted_cdf'\]; got 'type7'",
        ):
            pooled_statistics(
                frame, sample_dims=["draw"], quantile_method="type7", summation="fast"
            )
        with pytest.raises(
            ValueError, match=r"summation must be one of \['fast', 'exact'\]"
        ):
            pooled_statistics(
                frame,
                sample_dims=["draw"],
                summation="kahan",
                quantile_method="linear",
            )
        timed = frame.assign(year=2050)
        with pytest.raises(ValueError, match="quantile_method must be one of"):
            summarize_samples(
                timed,
                sample_dims=["draw"],
                time_col="year",
                window=(2050, 2050),
                quantile_method="step",
                summation="fast",
            )
        with pytest.raises(ValueError, match="summation must be one of"):
            window_means(
                timed, time_col="year", windows=[(2050, 2050)], summation="fsum"
            )

    def test_neither_option_has_a_default(self):
        frame = _members([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="summation"):
            pooled_statistics(frame, sample_dims=["draw"], quantile_method="linear")
        with pytest.raises(TypeError, match="quantile_method"):
            pooled_statistics(frame, sample_dims=["draw"], summation="fast")
        timed = frame.assign(year=2050)
        with pytest.raises(TypeError, match="quantile_method"):
            summarize_samples(
                timed,
                sample_dims=["draw"],
                time_col="year",
                window=(2050, 2050),
                summation="fast",
            )
        with pytest.raises(TypeError, match="summation"):
            window_means(timed, time_col="year", windows=[(2050, 2050)])

    def test_the_former_value_is_refused_with_the_value_to_use(self):
        """Both options were once "legacy". For a quantile it meant one
        of two computations, by whether the call had weights."""
        frame = _members([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="no weights, so use 'linear'"):
            pooled_statistics(
                frame,
                sample_dims=["draw"],
                quantile_method="legacy",
                summation="fast",
            )
        with pytest.raises(ValueError, match="weights, so use 'lower_step'"):
            pooled_statistics(
                frame.assign(w=1.0),
                sample_dims=["draw"],
                weight_col="w",
                quantile_method="legacy",
                summation="fast",
            )
        with pytest.raises(
            ValueError, match="summation='legacy' is not a value any more. Use 'fast'"
        ):
            pooled_statistics(
                frame,
                sample_dims=["draw"],
                quantile_method="linear",
                summation="legacy",
            )
        with pytest.raises(ValueError, match="Use 'fast'"):
            window_means(
                frame.assign(year=2050),
                time_col="year",
                windows=[(2050, 2050)],
                summation="legacy",
            )

    def test_linear_has_no_weighted_form(self):
        frame = _members([1.0, 2.0, 3.0], [0.5, 0.25, 0.25])
        with pytest.raises(
            ValueError, match="'linear' interpolates.*has no weighted form"
        ):
            pooled_statistics(
                frame,
                sample_dims=["draw"],
                weight_col="w",
                quantile_method="linear",
                summation="fast",
            )
        with pytest.raises(ValueError, match="has no weighted form"):
            summarize_samples(
                frame.assign(year=2050),
                sample_dims=["draw"],
                time_col="year",
                window=(2050, 2050),
                weight_col="w",
                quantile_method="linear",
                summation="fast",
            )


class TestInvertedCdf:
    """Hyndman and Fan type 1, the smallest value whose cumulative
    weight share reaches q."""

    @pytest.mark.skipif(
        int(np.__version__.split(".")[0]) < 2,
        reason="numpy.quantile takes weights from numpy 2.0",
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_weighted_matches_numpy(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 60))
        values = rng.normal(size=n)
        weights = rng.uniform(0.05, 1.0, n)
        qs = [0.0, 0.01, *QUANTILES, 0.99, 1.0]
        got = _pooled(
            _members(values, weights),
            qs,
            quantile_method="inverted_cdf",
            include_mean=False,
        )
        want = np.quantile(values, qs, method="inverted_cdf", weights=weights)
        assert list(got.values()) == want.tolist()

    @pytest.mark.parametrize("n", [1, 2, 7, 33, 100, 480])
    def test_no_weights_means_equal_weights(self, n):
        values = np.random.default_rng(n).normal(size=n)
        qs = [0.0, 0.07, *QUANTILES, 1.0]
        options = dict(quantile_method="inverted_cdf", include_mean=False)
        unweighted = _pooled(_members(values), qs, **options)
        assert unweighted == _pooled(_members(values, np.ones(n)), qs, **options)
        assert unweighted == _pooled(_members(values, np.full(n, 0.25)), qs, **options)
        if int(np.__version__.split(".")[0]) >= 2:
            # Weights of one, because numpy's unweighted form takes its
            # index from n * q in floating point and picks the eighth of
            # 100 values at 0.07. numpy takes weights from version 2.0.
            want = np.quantile(values, qs, method="inverted_cdf", weights=np.ones(n))
            assert list(unweighted.values()) == want.tolist()

    def test_a_share_that_equals_q_is_reached(self):
        """With 100 members, 7/100 is q = 0.07, so the 7th smallest
        reaches it. Working from the product 100 * 0.07, which rounds
        to just above 7, would give the 8th."""
        values = np.arange(1.0, 101.0)
        got = _pooled(
            _members(values),
            [0.07],
            quantile_method="inverted_cdf",
            include_mean=False,
        )
        assert list(got.values()) == [7.0]

    def test_the_median_is_a_member(self):
        """With an even count q50 is the lower of the two middle
        members, where "linear" and "lower_step" return their midpoint."""
        values = [4.0, 1.0, 3.0, 2.0]
        for weights in (None, [0.25] * 4):
            got = _pooled(
                _members(values, weights),
                [0.5],
                quantile_method="inverted_cdf",
                include_mean=False,
            )
            assert got == {"q50": 2.0}

    def test_never_minus_infinity(self):
        """The first member holds 60% of the weight, so it already
        reaches q = 0.05. "lower_step" looks for the member before it,
        finds none and returns -inf."""
        frame = _members([1.0, 2.0, 3.0], [0.6, 0.2, 0.2])
        step = _pooled(frame, [0.05], include_mean=False)
        assert step == {"q05": float("-inf")}
        got = _pooled(frame, [0.05], quantile_method="inverted_cdf", include_mean=False)
        assert got == {"q05": 1.0}

    def test_q0_and_q1_are_the_extremes_with_weight(self):
        """The smallest and the largest value both belong to members
        with weight zero. A share of zero reaches q = 0, so q0 is where
        such a member could still come back if it were left in."""
        frame = _members([0.5, 1.0, 9.0, 3.0], [0.0, 0.3, 0.0, 0.7])
        got = _pooled(
            frame, [0.0, 1.0], quantile_method="inverted_cdf", include_mean=False
        )
        assert list(got.values()) == [1.0, 3.0]

    def test_a_member_with_weight_zero_is_never_returned(self):
        """A weights table can carry a model at weight zero to leave it
        out. Of the members 1 to 33 the 21st has weight zero and the
        22nd reaches q = 0.64. "lower_step" returns the one before it,
        21.0, which the mean leaves out."""
        values = np.arange(1.0, 34.0)
        weights = np.full(33, 1 / 32)
        weights[20] = 0.0
        frame = _members(values, weights)

        step = _pooled(frame, [0.64], include_mean=False)
        assert step == {"q64": 21.0}

        got = _pooled(frame, [0.64], quantile_method="inverted_cdf", include_mean=False)
        assert got == {"q64": 22.0}

        # at no q, and exactly as if the member were not in the frame
        qs = [round(q, 3) for q in np.linspace(0.0, 1.0, 201)]
        options = dict(quantile_method="inverted_cdf", include_mean=False)
        every = _pooled(frame, qs, **options)
        assert 21.0 not in every.values()
        assert every == _pooled(frame.drop(index=20), qs, **options)

    def test_n_members_leaves_out_the_weight_zero_member(self):
        weights = np.full(33, 1 / 32)
        weights[20] = 0.0
        out = pooled_statistics(
            _members(np.arange(1.0, 34.0), weights),
            sample_dims=["draw"],
            quantiles=[0.5],
            weight_col="w",
            include_n_members=True,
            model_dims=["draw"],
            quantile_method="inverted_cdf",
            summation="fast",
        )
        assert out["n_members"].tolist() == [32, 32]

    def test_missing_values_drop_out(self):
        got = _pooled(
            _members([np.nan, 3.0, 1.0, np.nan, 2.0]),
            [0.5, 1.0],
            quantile_method="inverted_cdf",
            include_mean=False,
        )
        assert got == {"q50": 2.0, "q100": 3.0}

    def test_a_group_with_no_value_is_nan_without_weights(self):
        frame = pd.concat(
            [_members([1.0, 2.0], target="a"), _members([np.nan, np.nan], target="b")]
        )
        out = pooled_statistics(
            frame,
            sample_dims=["draw"],
            quantiles=[0.5],
            quantile_method="inverted_cdf",
            summation="exact",
        )
        by_target = out.set_index(["target", "statistic"])["value"]
        assert by_target["a", "q50"] == 1.0
        assert by_target["a", "mean"] == 1.5
        assert by_target.loc["b"].isna().all()

    def test_tied_values_do_not_make_the_row_order_matter(self):
        """Three tied members hold exactly 0.8 of the weight. Added in
        one order their share is 0.8, in the other 0.7999999999999999,
        which misses q = 0.8, so tied members are taken in order of
        weight."""
        picked = set()
        for tied in itertools.permutations([0.1, 0.2, 0.3]):
            got = _pooled(
                _members([1.0, 1.0, 1.0, 2.0], [*tied, 0.15]),
                [0.8],
                quantile_method="inverted_cdf",
                include_mean=False,
            )
            picked.add(got["q80"])
        assert picked == {1.0}

    def test_zero_comes_back_positive(self):
        """-0.0 and 0.0 are equal, so which of them the sort puts first
        depends on the row order. The result is 0.0 either way."""
        for values in ([-0.0, 0.0, 1.0], [0.0, -0.0, 1.0]):
            for weights in (None, [0.2, 0.3, 0.5]):
                got = _pooled(
                    _members(values, weights),
                    [0.1],
                    quantile_method="inverted_cdf",
                    include_mean=False,
                )
                assert got == {"q10": 0.0}
                assert not np.signbit(got["q10"])

    def test_a_nullable_value_column_stays_nullable(self):
        frame = _members([1.0, 2.0, 3.0])
        frame["value"] = frame["value"].astype("Float64")
        out = pooled_statistics(
            frame,
            sample_dims=["draw"],
            quantiles=[0.5],
            quantile_method="inverted_cdf",
            summation="exact",
        )
        assert out["value"].dtype == "Float64"
        assert out["value"].tolist() == [2.0, 2.0]


class TestLinear:
    """ "linear" is numpy's linear interpolation, computed by numpy."""

    @pytest.mark.parametrize("seed", range(3))
    def test_it_is_numpy_to_the_last_bit(self, seed):
        """Groups of every size from 1 to 60, values of every magnitude,
        and missing values. Failed with the quantiles taken by pandas,
        whose arithmetic for the same definition parts from numpy's in
        the last bit of about one quantile in a hundred."""
        rng = np.random.default_rng(seed)
        frames, expected = [], {}
        for i in range(400):
            n = int(rng.integers(1, 61))
            values = rng.normal(size=n) * 10.0 ** int(rng.integers(-4, 5))
            values[rng.random(n) < 0.05] = np.nan
            frames.append(_members(values, target=f"t{i}"))
            kept = values[~np.isnan(values)]
            expected[f"t{i}"] = (
                np.quantile(kept, QUANTILES, method="linear")
                if len(kept)
                else np.full(len(QUANTILES), np.nan)
            )
        got = pooled_statistics(
            pd.concat(frames, ignore_index=True),
            sample_dims=["draw"],
            quantiles=QUANTILES,
            include_mean=False,
            quantile_method="linear",
            summation="fast",
        )
        table = got.pivot(index="target", columns="statistic", values="value")
        labels = [stats._quantile_label(q) for q in QUANTILES]
        for target, want in expected.items():
            have = table.loc[target, labels].to_numpy(dtype=np.float64)
            assert have.tobytes() == np.asarray(want, dtype=np.float64).tobytes(), (
                target
            )

    def test_a_quantile_next_to_an_infinity_is_refused(self):
        """numpy's interpolation gives NaN there, even at the median of
        1, 2 and an infinity, and a NaN would read as a group with no
        value. The infinite member is refused before it is reached."""
        frame = pd.concat(
            [
                _members([1.0, 2.0, np.inf], target="a"),
                _members([np.nan] * 2, target="b"),
            ]
        )
        with pytest.raises(
            ValueError,
            match=r"1 values of 'value' are infinite \(first: \{'target': 'a'",
        ):
            pooled_statistics(
                frame,
                sample_dims=["draw"],
                quantiles=[0.5],
                include_mean=False,
                quantile_method="linear",
                summation="fast",
            )
        got = pooled_statistics(
            frame[frame["target"] == "b"],
            sample_dims=["draw"],
            quantiles=[0.5],
            quantile_method="linear",
            summation="fast",
        )
        assert got["value"].isna().all()


class TestLowerStep:
    """The step of the historical extraction tool, the member one below
    the one whose running share reaches q. Without weights every member
    counts the same."""

    def test_four_members_by_hand(self):
        """The share of each member is a quarter. A quantile is the
        member before the one whose running share reaches it, so the
        95th is the third of four. The median is the plain one, and a
        quantile below a quarter has no member before it."""
        frame = _members([4.0, 1.0, 3.0, 2.0])
        got = _pooled(frame, [0.05, 0.5, 0.75, 0.95], quantile_method="lower_step")
        assert got == {
            "mean": 2.5,
            "q05": -np.inf,
            "q50": 2.5,
            "q75": 2.0,
            "q95": 3.0,
        }
        # the two other computations, on the same members
        linear = _pooled(frame, [0.75, 0.95], quantile_method="linear")
        assert linear["q75"] == 3.25 and linear["q95"] == pytest.approx(3.85)
        inverted = _pooled(frame, [0.75, 0.95], quantile_method="inverted_cdf")
        assert inverted["q75"] == 3.0 and inverted["q95"] == 4.0

    @pytest.mark.parametrize("summation", ["fast", "exact"])
    @pytest.mark.parametrize("seed", range(3))
    def test_no_weights_is_the_step_with_every_weight_one(self, seed, summation):
        """To the bit, the mean and the member counts included."""
        frame = _random_frame(seed).drop(columns="w")
        kwargs = dict(
            sample_dims=["draw"],
            quantiles=[0.25, 0.5, 0.75, 0.95],
            include_n_members=True,
            model_dims=["draw"],
            summation=summation,
            quantile_method="lower_step",
        )
        got = pooled_statistics(frame, **kwargs)
        ones = pooled_statistics(frame.assign(w=1.0), weight_col="w", **kwargs)
        pd.testing.assert_frame_equal(got, ones, check_exact=True)
        assert got["value"].to_numpy().tobytes() == ones["value"].to_numpy().tobytes()

    def test_summarize_samples_takes_it(self):
        frame = _timed()
        kwargs = dict(
            sample_dims=["draw"],
            time_col="year",
            window=(2040, 2049),
            quantiles=[0.5, 0.95],
            quantile_method="lower_step",
            summation="fast",
        )
        got = summarize_samples(frame, **kwargs)
        ones = summarize_samples(frame.assign(w=1.0), weight_col="w", **kwargs)
        pd.testing.assert_frame_equal(got, ones, check_exact=True)

    def test_a_group_with_no_value_is_nan_without_weights(self):
        """As on the other paths without weights, and the count says
        nobody is behind it. With weights such a group is an error."""
        frame = pd.concat(
            [_members([1.0, 2.0, 3.0], target="a"), _members([np.nan] * 3, target="b")]
        )
        got = pooled_statistics(
            frame,
            sample_dims=["draw"],
            quantiles=[0.5],
            quantile_method="lower_step",
            include_n_members=True,
            model_dims=["draw"],
            summation="fast",
        )
        empty = got[got["target"] == "b"]
        assert empty["value"].isna().all() and len(empty) == 2
        assert set(empty["n_members"]) == {0}
        assert set(got.loc[got["target"] == "a", "n_members"]) == {3}
        with pytest.raises(ValueError, match="weights sum to zero"):
            pooled_statistics(
                frame.assign(w=1.0),
                sample_dims=["draw"],
                quantiles=[0.5],
                weight_col="w",
                quantile_method="lower_step",
                summation="fast",
            )

    def test_missing_values_drop_out(self):
        got = _pooled(
            _members([4.0, np.nan, 1.0, 3.0, 2.0]),
            [0.95],
            quantile_method="lower_step",
        )
        assert got == {"mean": 2.5, "q95": 3.0}

    @pytest.mark.parametrize("summation", ["fast", "exact"])
    def test_minus_infinity_is_returned_under_either_sum(self, summation):
        """It is what the tool returned, at or below 1/n with equal
        weights, here 0.05 of three members."""
        frame = _members([1.0, 2.0, 3.0])
        got = _pooled(frame, [0.05], quantile_method="lower_step", summation=summation)
        assert got["q05"] == -np.inf

    def test_a_nullable_value_column_gives_floats(self):
        frame = _members([1.0, 2.0, 3.0, 4.0])
        frame["value"] = pd.array([1.0, None, 3.0, 4.0], dtype="Float64")
        got = pooled_statistics(
            frame,
            sample_dims=["draw"],
            quantiles=[0.95],
            quantile_method="lower_step",
            summation="fast",
        )
        assert got["value"].dtype == np.float64
        assert got["value"].tolist() == [8.0 / 3.0, 3.0]


class TestExactShares:
    """Under "exact" the cumulative weight shares of the weighted
    quantiles come from exact sums, and equal values are taken in order
    of weight, so the row order and the order of summation decide
    nothing."""

    def test_the_running_shares_are_exact(self):
        """Each share is the ratio of two exact sums rounded once, as
        exact rational arithmetic gives it."""
        from fractions import Fraction

        rng = np.random.default_rng(5)
        for _ in range(300):
            n = int(rng.integers(1, 40))
            weights = rng.uniform(0, 1, n) * 10.0 ** rng.integers(-6, 4, n)
            weights[rng.random(n) < 0.1] = 0.0
            weights[0] = max(weights[0], 1e-3)
            running, total = [], sum(Fraction(float(w)) for w in weights)
            acc = Fraction(0)
            for w in weights:
                acc += Fraction(float(w))
                running.append(float(acc / total))
            got = stats._exact_shares(weights)
            assert got.tobytes() == np.array(running).tobytes()

    # In both groups the running sum of the first four weights is half
    # the total in exact arithmetic. With np.cumsum the share of the
    # step falls below one half in the first, and the share of the
    # standard inverse in the second.
    STEP_TIE = [0.3, 0.1, 0.6, 0.2, 0.2, 0.7, 0.2, 0.1]
    INVERSE_TIE = [0.2, 0.7, 0.7, 0.1, 0.1, 0.7, 0.2, 0.7]

    @pytest.mark.parametrize(
        "method, weights, fast",
        [
            ("lower_step", STEP_TIE, 4.0),
            ("inverted_cdf", INVERSE_TIE, 5.0),
        ],
        ids=["lower_step", "inverted_cdf"],
    )
    def test_a_share_equal_to_q_is_decided_by_the_rule(self, method, weights, fast):
        """At q = 0.5 the fourth member's share reaches q, so the
        standard inverse is the fourth value and the step the third.
        With the shares np.cumsum gives, one of them comes out a member
        higher."""
        frame = _members([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], weights)
        exact = _pooled(frame, [0.5], quantile_method=method, summation="exact")
        assert exact["q50"] == (3.0 if method == "lower_step" else 4.0)
        assert (
            _pooled(frame, [0.5], quantile_method=method, summation="fast")["q50"]
            == fast
        )

    @pytest.mark.parametrize("method", ["lower_step", "inverted_cdf"])
    def test_the_row_order_decides_nothing(self, method):
        """Three members share the value 1. Under "fast" the step
        depends on which of them comes first, and is 1 in one order and
        -inf in another. The standard inverse cannot go red here, and
        did not under a mutation that takes equal values in the order of
        the rows: it is the smallest value whose share reaches q, and
        every member of a run of equal values is the same value. It is
        here to show the two side by side."""
        values = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
        weights = np.array([0.1, 0.2, 0.1, 0.2, 0.7, 0.1])
        quantiles = [0.142857, 0.25, 0.5, 0.75]
        exact, fast = set(), set()
        for order in itertools.permutations(range(6)):
            order = list(order)
            frame = _members(values[order], weights[order])
            exact.add(
                tuple(
                    _pooled(
                        frame,
                        quantiles,
                        quantile_method=method,
                        summation="exact",
                        include_mean=False,
                    ).values()
                )
            )
            fast.add(
                tuple(
                    _pooled(
                        frame,
                        quantiles,
                        quantile_method=method,
                        summation="fast",
                        include_mean=False,
                    ).values()
                )
            )
        assert len(exact) == 1
        if method == "lower_step":
            assert len(fast) > 1

    def test_equal_weights_give_the_shares_they_gave(self):
        """k of n equal weights are a share of k / n either way, so
        nothing changes for a group without weights. Cannot go red by
        reverting, since before the change both sums took the same
        shares. It failed with the median of equal weights left out of
        the step under "exact"."""
        frame = _random_frame(2).drop(columns="w")
        kwargs = dict(sample_dims=["draw"], quantiles=QUANTILES, include_mean=False)
        for method in ("lower_step", "inverted_cdf"):
            exact = pooled_statistics(
                frame, quantile_method=method, summation="exact", **kwargs
            )
            fast = pooled_statistics(
                frame, quantile_method=method, summation="fast", **kwargs
            )
            pd.testing.assert_frame_equal(exact, fast, check_exact=True)


class TestExactSummation:
    def test_the_mean_does_not_depend_on_the_order(self):
        """1e16 + 1 is not a float, so a sum taken in row order loses
        the 1 or keeps it depending on where the 1 comes. The exactly
        rounded sum is 1 in every order."""
        for order in itertools.permutations([1e16, 1.0, -1e16]):
            got = _pooled(_members(order), [], summation="exact")
            assert got == {"mean": 1.0 / 3.0}
            got = _pooled(_members(order, [0.5] * 3), [], summation="exact")
            assert got == {"mean": 0.5 / 1.5}

    @pytest.mark.parametrize("weighted", [False, True])
    def test_pooled_mean_is_the_exactly_rounded_one(self, weighted):
        frame = _random_frame(1)
        if not weighted:
            frame = frame.drop(columns="w")
        out = pooled_statistics(
            frame,
            sample_dims=["draw"],
            weight_col="w" if weighted else None,
            summation="exact",
            quantile_method="lower_step" if weighted else "linear",
        )
        got = out.set_index(["target", "scenario"])["value"]
        for key, group in frame.dropna(subset=["value"]).groupby(
            ["target", "scenario"]
        ):
            w = group["w"] if weighted else pd.Series(1.0, index=group.index)
            want = math.fsum((group["value"] * w).tolist()) / math.fsum(w.tolist())
            assert got[key] == want

    @pytest.mark.parametrize("weighted", [False, True])
    def test_shuffled_rows_give_the_same_bits(self, weighted):
        """Nothing in the statistics may depend on the order the rows
        arrive in, with tied values, missing values and, when weighted,
        unequal weights and members at zero."""
        frame = _random_frame(2)
        if not weighted:
            frame = frame.drop(columns="w")
        kwargs = dict(
            sample_dims=["draw"],
            quantiles=[0.0, *QUANTILES, 1.0],
            weight_col="w" if weighted else None,
            include_n_members=True,
            model_dims=["draw"],
            quantile_method="inverted_cdf",
            summation="exact",
        )
        first = _sorted(pooled_statistics(frame, **kwargs))
        for seed in range(3):
            shuffled = frame.sample(frac=1.0, random_state=seed)
            pd.testing.assert_frame_equal(
                first,
                _sorted(pooled_statistics(shuffled, **kwargs)),
                check_exact=True,
            )

    def test_window_means_are_the_exactly_rounded_ones(self):
        frame = _timed()
        # a member with missing years, and one with no value in a window
        frame.loc[(frame["target"] == "t1") & (frame["year"] < 2045), "value"] = np.nan
        gone = (
            (frame["target"] == "t2")
            & (frame["draw"] == "d0")
            & (frame["year"] >= 2050)
        )
        frame.loc[gone, "value"] = np.nan
        windows = [(2040, 2049), (2050, 2059), (2045, 2054)]
        kwargs = dict(time_col="year", windows=windows)
        got = window_means(frame, summation="exact", **kwargs)
        fast = window_means(frame, summation="fast", **kwargs)

        # same rows in the same order, same dtypes; only the sum differs
        pd.testing.assert_frame_equal(got, fast, rtol=1e-12)
        for (lo, hi), label in zip(windows, ["2040-2049", "2050-2059", "2045-2054"]):
            part = frame[frame["year"].between(lo, hi)]
            rows = got[got["window"] == label].set_index(["target", "draw"])["value"]
            for key, group in part.groupby(["target", "draw"]):
                values = group["value"].dropna().tolist()
                if values:
                    assert rows[key] == math.fsum(values) / len(values)
                else:
                    assert np.isnan(rows[key])
        assert np.isnan(
            got[
                (got["target"] == "t2")
                & (got["draw"] == "d0")
                & (got["window"] == "2050-2059")
            ]["value"]
        ).all()

    def test_window_means_survive_a_shuffle_and_small_chunks(self, monkeypatch):
        frame = _timed()
        kwargs = dict(
            time_col="year", windows=[(2040, 2049), (2050, 2059)], summation="exact"
        )
        first = _sorted(window_means(frame, **kwargs))
        shuffled = frame.sample(frac=1.0, random_state=0)
        pd.testing.assert_frame_equal(
            first, _sorted(window_means(shuffled, **kwargs)), check_exact=True
        )
        # the rows go to Python a chunk at a time; three members a chunk
        monkeypatch.setattr(stats, "_FSUM_CHUNK", 30)
        pd.testing.assert_frame_equal(
            first, _sorted(window_means(frame, **kwargs)), check_exact=True
        )

    @pytest.mark.parametrize("summation", ["fast", "exact"])
    @pytest.mark.parametrize("dtype", ["float32", "Float64", "float64"])
    def test_window_means_are_float64_whatever_the_value_dtype(self, dtype, summation):
        """The dtype used to follow the column and the pandas version.
        The float64 cases cannot go red by reverting, since a float64
        column always came back float64. They failed under a mutation
        that narrows the result to float32."""
        frame = _timed()
        frame["value"] = frame["value"].astype(dtype)
        got = window_means(
            frame, time_col="year", windows=[(2040, 2059)], summation=summation
        )
        assert got["value"].dtype == np.dtype("float64")

    @pytest.mark.parametrize("summation", ["fast", "exact"])
    @pytest.mark.parametrize("dtype", ["float32", "Float64"])
    def test_a_narrow_or_nullable_column_is_widened_before_the_mean(
        self, dtype, summation
    ):
        """The mean of a float32 column is the float64 mean of the same
        numbers, to the last bit, and a nullable column gives NaN where
        a member has no value."""
        frame = _timed()
        frame.loc[(frame["target"] == "t1") & (frame["draw"] == "d2"), "value"] = np.nan
        frame.loc[(frame["target"] == "t3") & (frame["year"] < 2047), "value"] = np.nan
        narrow = frame.assign(value=frame["value"].astype(dtype))
        wide = narrow.assign(value=narrow["value"].astype("float64"))
        kwargs = dict(
            time_col="year", windows=[(2040, 2059), (2045, 2050)], summation=summation
        )
        got = window_means(narrow, **kwargs)
        pd.testing.assert_frame_equal(
            got, window_means(wide, **kwargs), check_exact=True
        )
        assert got["value"].isna().sum() == 2
        if summation == "exact":
            part = wide[(wide["target"] == "t0") & (wide["draw"] == "d0")]
            row = got[
                (got["target"] == "t0")
                & (got["draw"] == "d0")
                & (got["window"] == "2040-2059")
            ]
            assert row["value"].item() == math.fsum(part["value"]) / len(part)

    @pytest.mark.parametrize("summation", ["fast", "exact"])
    @pytest.mark.parametrize("dtype", ["float32", "Float64", "float64"])
    def test_summarize_samples_is_float64_whatever_the_value_dtype(
        self, dtype, summation
    ):
        """The mean is asked for alone as well, because next to a
        quantile a float32 mean was widened when stacked, which hid its
        dtype. The float64 cases and float32 under "exact" cannot go
        red by reverting. All six failed under a mutation that narrows
        the result to float32."""
        frame = _timed()
        frame["value"] = frame["value"].astype(dtype)
        for quantiles in ([], [0.25, 0.5]):
            got = summarize_samples(
                frame,
                sample_dims=["draw"],
                time_col="year",
                window=(2040, 2059),
                quantiles=quantiles,
                summation=summation,
                quantile_method="linear",
            )
            assert got["value"].dtype == np.dtype("float64")

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("quantile_method", ["plain", "inverted_cdf"])
    @pytest.mark.parametrize("summation", ["fast", "exact"])
    @pytest.mark.parametrize("dtype", ["float32", "Float64"])
    def test_summarize_samples_widens_before_the_window_mean(
        self, dtype, summation, quantile_method, weighted
    ):
        """The statistics of a float32 or nullable column are those of
        the same numbers held as float64, to the last bit. The weighted
        nullable cases cannot go red by reverting, since such a column
        was always taken as float64 there. They failed under a mutation
        that rounds the widened column to float32."""
        frame = _timed()
        frame.loc[(frame["target"] == "t1") & (frame["draw"] == "d2"), "value"] = np.nan
        frame.loc[(frame["target"] == "t3") & (frame["year"] < 2047), "value"] = np.nan
        if weighted:
            # no member holds half the weight, so the step quantiles
            # asked for below are never the -inf that "exact" refuses
            frame["w"] = frame["draw"].map(
                {"d0": 0.3, "d1": 0.1, "d2": 0.2, "d3": 0.15, "d4": 0.25}
            )
        narrow = frame.assign(value=frame["value"].astype(dtype))
        wide = narrow.assign(value=narrow["value"].astype("float64"))
        kwargs = dict(
            sample_dims=["draw"],
            time_col="year",
            window=(2040, 2059),
            quantiles=[0.5, 0.83],
            weight_col="w" if weighted else None,
            quantile_method=(
                _plain(weighted) if quantile_method == "plain" else quantile_method
            ),
            summation=summation,
        )
        got = summarize_samples(narrow, **kwargs)
        assert got["value"].dtype == np.dtype("float64")
        pd.testing.assert_frame_equal(
            got, summarize_samples(wide, **kwargs), check_exact=True
        )


class TestInfiniteStatistics:
    """An infinite member value is refused under every quantile method
    and sum, and so is a statistic that comes out infinite, other than
    the -inf of "lower_step" at or below its first step."""

    SETTINGS = [
        {"quantile_method": q, "summation": s}
        for q in ("linear", "lower_step", "inverted_cdf")
        for s in ("fast", "exact")
    ]

    @pytest.mark.parametrize("options", SETTINGS)
    @pytest.mark.parametrize("values", [[np.inf, 1.0, 2.0], [-np.inf, 1.0, -np.inf]])
    @pytest.mark.parametrize("weights", [None, [0.5, 0.25, 0.25]])
    def test_an_infinite_member_is_refused(self, options, values, weights):
        """Failed under "lower_step" with the fast sum, which returned
        the infinity, and with infinities of both signs, whose mean is
        NaN."""
        if weights is not None and options["quantile_method"] == "linear":
            pytest.skip("linear has no weighted form")
        frame = _members(values, weights)
        with pytest.raises(ValueError, match=r"values of 'value' are infinite"):
            _pooled(frame, [0.5], **options)

    def test_an_infinite_member_no_statistic_reaches_is_refused(self):
        """Failed when only the statistics were checked, since the
        median of 1, 2, 3 and an infinity is 2."""
        with pytest.raises(ValueError, match=r"1 values of 'value' are infinite"):
            _pooled(
                _members([1.0, 2.0, 3.0, np.inf]),
                [0.5],
                quantile_method="inverted_cdf",
                summation="exact",
                include_mean=False,
            )

    def test_an_infinity_in_a_year_no_window_covers_is_let_through(self):
        """Data that never enters a result must not fail the run. This
        cannot go red by reverting, because without the check nothing is
        refused. It failed with the check put on every row of the value
        column."""
        frame = _timed()
        frame.loc[frame["year"] == 2059, "value"] = np.inf
        out = summarize_samples(
            frame,
            sample_dims=["draw"],
            time_col="year",
            window=(2040, 2049),
            quantiles=QUANTILES,
            quantile_method="inverted_cdf",
            summation="exact",
        )
        assert np.isfinite(out["value"]).all()
        means = window_means(
            frame, time_col="year", windows=[(2040, 2049)], summation="fast"
        )
        assert np.isfinite(means["value"]).all()

    @pytest.mark.parametrize("summation", ["fast", "exact"])
    def test_the_window_mean_refuses_an_infinity(self, summation):
        """Failed with the member's mean passed on as an infinity."""
        frame = _timed()
        frame.loc[7, "value"] = -np.inf
        with pytest.raises(ValueError, match=r"1 values of 'value' are infinite"):
            window_means(
                frame, time_col="year", windows=[(2040, 2059)], summation=summation
            )

    def test_an_overflow_is_refused_under_either_sum(self):
        """Finite values whose sum, or whose product with the weight,
        passes the largest float. Failed under the fast sum, which
        returned the infinity."""
        with pytest.raises(ValueError, match="overflows the largest float"):
            _pooled(_members([1e308, 1e308]), [], summation="exact")
        with pytest.raises(ValueError, match="arithmetic passed the largest float"):
            _pooled(_members([1e308, 1.0], [10.0, 1.0]), [], summation="exact")
        with pytest.raises(ValueError, match=r"statistics are infinite \(mean: 1"):
            _pooled(_members([1e308, 1e308]), [])

    @pytest.mark.parametrize("summation", ["fast", "exact"])
    def test_the_minus_infinity_of_the_step_is_kept(self, summation):
        """By its definition "lower_step" is -inf at or below its first
        step, here the first member's share of 0.6. Failed under the
        exact sum, which refused it."""
        frame = _members([1.0, 2.0, 3.0], [0.6, 0.2, 0.2])
        got = _pooled(frame, [0.05, 0.6, 0.65], summation=summation)
        assert got["q05"] == -np.inf and got["q60"] == -np.inf
        assert got["q65"] == 1.0 and np.isfinite(got["mean"])

    def test_nan_is_left_alone(self):
        """A group with no value has no statistic, under any setting.
        This cannot go red by reverting either. It failed with the
        check refusing everything that is not finite."""
        frame = pd.concat(
            [_members([1.0, 2.0], target="a"), _members([np.nan, np.nan], target="b")]
        )
        for options in self.SETTINGS:
            out = pooled_statistics(
                frame, sample_dims=["draw"], quantiles=[0.5], **options
            )
            assert out.loc[out["target"] == "b", "value"].isna().all()
            assert out.loc[out["target"] == "a", "value"].notna().all()

    def test_the_error_says_where(self):
        frame = pd.concat(
            [_members([1.0, 2.0], target="a"), _members([1.0, np.inf], target="b")]
        )
        with pytest.raises(ValueError) as error:
            pooled_statistics(
                frame,
                sample_dims=["draw"],
                quantiles=[1.0],
                summation="fast",
                quantile_method="lower_step",
            )
        message = str(error.value)
        assert "1 values of 'value' are infinite" in message
        assert "'target': 'b'" in message
        assert "summation" not in message


class TestBothHalves:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_summarize_samples_is_the_two_steps(self, weighted):
        """`summarize_samples` with the options equals `window_means`
        then `pooled_statistics` with the same ones, bit for bit, on
        rows in no particular order."""
        rng = np.random.default_rng(4)
        draw_weight = rng.uniform(0.1, 1.0, 9)
        draw_weight[4] = 0.0
        rows = [
            (f"t{t}", scenario, f"d{d}", year, draw_weight[d])
            for t in range(5)
            for scenario in ("low", "high")
            for d in range(9)
            for year in range(2040, 2050)
        ]
        frame = pd.DataFrame(rows, columns=["target", "scenario", "draw", "year", "w"])
        frame["value"] = np.round(rng.normal(size=len(frame)) * 1e6, 0)
        frame.loc[rng.random(len(frame)) < 0.03, "value"] = np.nan
        frame = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
        if not weighted:
            frame = frame.drop(columns="w")
        options = dict(quantile_method="inverted_cdf", summation="exact")
        weight_col = "w" if weighted else None

        whole = summarize_samples(
            frame,
            sample_dims=["draw"],
            time_col="year",
            window=(2042, 2048),
            quantiles=QUANTILES,
            weight_col=weight_col,
            include_n_members=True,
            model_dims=["draw"],
            **options,
        )
        reduced = window_means(
            frame, time_col="year", windows=[(2042, 2048)], summation="exact"
        )
        steps = pooled_statistics(
            reduced,
            sample_dims=["draw"],
            quantiles=QUANTILES,
            weight_col=weight_col,
            include_n_members=True,
            model_dims=["draw"],
            **options,
        ).drop(columns="window")
        pd.testing.assert_frame_equal(
            _sorted(whole), _sorted(steps[whole.columns]), check_exact=True
        )
