"""The experimental ``quantile_method`` and ``summation`` options.

These tests cover "inverted_cdf" and "exact". The rest of the suite
holds the "legacy" defaults to what they have always returned.
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


def _pooled(frame, quantiles, **options) -> dict[str, float]:
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
        for t in range(6) for d in range(5) for year in range(2040, 2060)
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
        with pytest.raises(ValueError, match="quantile_method must be one of"):
            pooled_statistics(frame, sample_dims=["draw"], quantile_method="type7")
        with pytest.raises(ValueError, match="summation must be one of"):
            pooled_statistics(frame, sample_dims=["draw"], summation="kahan")
        timed = frame.assign(year=2050)
        with pytest.raises(ValueError, match="quantile_method must be one of"):
            summarize_samples(
                timed, sample_dims=["draw"], time_col="year",
                window=(2050, 2050), quantile_method="step",
            )
        with pytest.raises(ValueError, match="summation must be one of"):
            window_means(
                timed, time_col="year", windows=[(2050, 2050)], summation="fsum"
            )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_legacy_by_name_is_the_default(self, weighted):
        frame = _random_frame(0)
        if not weighted:
            frame = frame.drop(columns="w")
        kwargs = dict(
            sample_dims=["draw"],
            quantiles=QUANTILES,
            weight_col="w" if weighted else None,
        )
        pd.testing.assert_frame_equal(
            pooled_statistics(frame, **kwargs),
            pooled_statistics(
                frame, quantile_method="legacy", summation="legacy", **kwargs
            ),
            check_exact=True,
        )


class TestInvertedCdf:
    """Hyndman and Fan type 1, the smallest value whose cumulative
    weight share reaches q."""

    @pytest.mark.parametrize("seed", range(5))
    def test_weighted_matches_numpy(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 60))
        values = rng.normal(size=n)
        weights = rng.uniform(0.05, 1.0, n)
        qs = [0.0, 0.01, *QUANTILES, 0.99, 1.0]
        got = _pooled(
            _members(values, weights), qs,
            quantile_method="inverted_cdf", include_mean=False,
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
        assert unweighted == _pooled(
            _members(values, np.full(n, 0.25)), qs, **options
        )
        want = np.quantile(values, qs, method="inverted_cdf", weights=np.ones(n))
        assert list(unweighted.values()) == want.tolist()

    def test_a_share_that_equals_q_is_reached(self):
        """With 100 members, 7/100 is q = 0.07, so the 7th smallest
        reaches it. Working from the product 100 * 0.07, which rounds
        to just above 7, would give the 8th."""
        values = np.arange(1.0, 101.0)
        got = _pooled(
            _members(values), [0.07],
            quantile_method="inverted_cdf", include_mean=False,
        )
        assert list(got.values()) == [7.0]

    def test_the_median_is_a_member(self):
        """With an even count q50 is the lower of the two middle
        members, where the legacy paths return their midpoint."""
        values = [4.0, 1.0, 3.0, 2.0]
        for weights in (None, [0.25] * 4):
            got = _pooled(
                _members(values, weights), [0.5],
                quantile_method="inverted_cdf", include_mean=False,
            )
            assert got == {"q50": 2.0}

    def test_never_minus_infinity(self):
        """The first member holds 60% of the weight, so it already
        reaches q = 0.05. The legacy step looks for the member before
        it, finds none and returns -inf."""
        frame = _members([1.0, 2.0, 3.0], [0.6, 0.2, 0.2])
        legacy = _pooled(frame, [0.05], include_mean=False)
        assert legacy == {"q05": float("-inf")}
        got = _pooled(
            frame, [0.05], quantile_method="inverted_cdf", include_mean=False
        )
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
        22nd reaches q = 0.64. The legacy step returns the one before
        it, 21.0, which the mean leaves out."""
        values = np.arange(1.0, 34.0)
        weights = np.full(33, 1 / 32)
        weights[20] = 0.0
        frame = _members(values, weights)

        legacy = _pooled(frame, [0.64], include_mean=False)
        assert legacy == {"q64": 21.0}

        got = _pooled(
            frame, [0.64], quantile_method="inverted_cdf", include_mean=False
        )
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
            sample_dims=["draw"], quantiles=[0.5], weight_col="w",
            include_n_members=True, quantile_method="inverted_cdf",
        )
        assert out["n_members"].tolist() == [32, 32]

    def test_missing_values_drop_out(self):
        got = _pooled(
            _members([np.nan, 3.0, 1.0, np.nan, 2.0]), [0.5, 1.0],
            quantile_method="inverted_cdf", include_mean=False,
        )
        assert got == {"q50": 2.0, "q100": 3.0}

    def test_a_group_with_no_value_is_nan_without_weights(self):
        frame = pd.concat(
            [_members([1.0, 2.0], target="a"), _members([np.nan, np.nan], target="b")]
        )
        out = pooled_statistics(
            frame, sample_dims=["draw"], quantiles=[0.5],
            quantile_method="inverted_cdf", summation="exact",
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
                _members([1.0, 1.0, 1.0, 2.0], [*tied, 0.15]), [0.8],
                quantile_method="inverted_cdf", include_mean=False,
            )
            picked.add(got["q80"])
        assert picked == {1.0}

    def test_zero_comes_back_positive(self):
        """-0.0 and 0.0 are equal, so which of them the sort puts first
        depends on the row order. The result is 0.0 either way."""
        for values in ([-0.0, 0.0, 1.0], [0.0, -0.0, 1.0]):
            for weights in (None, [0.2, 0.3, 0.5]):
                got = _pooled(
                    _members(values, weights), [0.1],
                    quantile_method="inverted_cdf", include_mean=False,
                )
                assert got == {"q10": 0.0}
                assert not np.signbit(got["q10"])

    def test_a_nullable_value_column_stays_nullable(self):
        frame = _members([1.0, 2.0, 3.0])
        frame["value"] = frame["value"].astype("Float64")
        out = pooled_statistics(
            frame, sample_dims=["draw"], quantiles=[0.5],
            quantile_method="inverted_cdf", summation="exact",
        )
        assert out["value"].dtype == "Float64"
        assert out["value"].tolist() == [2.0, 2.0]


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
            frame, sample_dims=["draw"],
            weight_col="w" if weighted else None, summation="exact",
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
            sample_dims=["draw"], quantiles=[0.0, *QUANTILES, 1.0],
            weight_col="w" if weighted else None, include_n_members=True,
            quantile_method="inverted_cdf", summation="exact",
        )
        first = _sorted(pooled_statistics(frame, **kwargs))
        for seed in range(3):
            shuffled = frame.sample(frac=1.0, random_state=seed)
            pd.testing.assert_frame_equal(
                first, _sorted(pooled_statistics(shuffled, **kwargs)),
                check_exact=True,
            )

    def test_window_means_are_the_exactly_rounded_ones(self):
        frame = _timed()
        # a member with missing years, and one with no value in a window
        frame.loc[(frame["target"] == "t1") & (frame["year"] < 2045), "value"] = np.nan
        gone = (frame["target"] == "t2") & (frame["draw"] == "d0") & (frame["year"] >= 2050)
        frame.loc[gone, "value"] = np.nan
        windows = [(2040, 2049), (2050, 2059), (2045, 2054)]
        kwargs = dict(time_col="year", windows=windows)
        got = window_means(frame, summation="exact", **kwargs)
        legacy = window_means(frame, **kwargs)

        # same rows in the same order, same dtypes; only the sum differs
        pd.testing.assert_frame_equal(got, legacy, rtol=1e-12)
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
            got[(got["target"] == "t2") & (got["draw"] == "d0")
                & (got["window"] == "2050-2059")]["value"]
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

    @pytest.mark.parametrize("dtype", ["float32", "Float64"])
    def test_window_means_keep_the_value_dtype(self, dtype):
        frame = _timed()
        frame["value"] = frame["value"].astype(dtype)
        kwargs = dict(time_col="year", windows=[(2040, 2059)])
        got = window_means(frame, summation="exact", **kwargs)
        assert got["value"].dtype == window_means(frame, **kwargs)["value"].dtype


class TestInfiniteStatistics:
    """Under either experimental setting an infinite statistic is
    refused. An infinity in the data that reaches no statistic is not
    an error."""

    SETTINGS = [
        {"summation": "exact"},
        {"quantile_method": "inverted_cdf"},
        {"quantile_method": "inverted_cdf", "summation": "exact"},
    ]

    @pytest.mark.parametrize("options", SETTINGS)
    @pytest.mark.parametrize("values", [[np.inf, 1.0, 2.0], [-np.inf, 1.0, 2.0]])
    @pytest.mark.parametrize("weights", [None, [0.5, 0.25, 0.25]])
    def test_an_infinite_mean_is_refused(self, options, values, weights):
        frame = _members(values, weights)
        with pytest.raises(ValueError, match=r"1 statistics are infinite \(mean: 1"):
            _pooled(frame, [], **options)
        # the legacy settings are left as they are: it comes back
        assert np.isinf(_pooled(frame, [])["mean"])

    @pytest.mark.parametrize("options", SETTINGS)
    def test_an_infinite_member_a_quantile_selects_is_refused(self, options):
        """No sum is taken here. The largest member is infinite and
        q100 returns it."""
        frame = _members([1.0, 2.0, np.inf])
        with pytest.raises(ValueError, match=r"statistics are infinite \(q100: 1"):
            _pooled(frame, [0.5, 1.0], include_mean=False, **options)
        legacy = _pooled(frame, [0.5, 1.0], include_mean=False)
        assert list(legacy.values()) == [2.0, np.inf]

    def test_an_infinity_that_reaches_no_statistic_is_let_through(self):
        """Cannot go red by reverting, because without the check
        nothing is refused. It failed with the check put on the member
        values instead of on the statistics."""
        got = _pooled(
            _members([1.0, 2.0, 3.0, np.inf]), [0.5],
            quantile_method="inverted_cdf", summation="exact", include_mean=False,
        )
        assert got == {"q50": 2.0}

    def test_an_infinity_in_a_year_no_window_covers_is_let_through(self):
        """Data that never enters a result must not fail the run. Like
        the test above this cannot go red by reverting. It failed with
        the check put on every row of the value column."""
        frame = _timed()
        frame.loc[frame["year"] == 2059, "value"] = np.inf
        out = summarize_samples(
            frame, sample_dims=["draw"], time_col="year", window=(2040, 2049),
            quantiles=QUANTILES, quantile_method="inverted_cdf", summation="exact",
        )
        assert np.isfinite(out["value"]).all()

    def test_the_window_mean_passes_an_infinity_on(self):
        """`window_means` returns no statistic. With an infinite year
        the member's mean is infinite under either summation, and it is
        the statistics over the members that refuse it."""
        frame = _timed()
        frame.loc[7, "value"] = -np.inf
        kwargs = dict(time_col="year", windows=[(2040, 2059)])
        exact = window_means(frame, summation="exact", **kwargs)
        legacy = window_means(frame, **kwargs)
        assert (exact["value"] == -np.inf).sum() == 1
        assert ((exact["value"] == -np.inf) == (legacy["value"] == -np.inf)).all()
        with pytest.raises(ValueError, match=r"statistics are infinite \(mean: 1"):
            summarize_samples(
                frame, sample_dims=["draw"], time_col="year",
                window=(2040, 2059), summation="exact",
            )

    def test_infinities_of_both_signs_have_no_sum(self):
        """Their sum is not a number. The exact sum raises where numpy
        returns NaN, which would pass for a group with no value."""
        frame = _members([np.inf, 1.0, -np.inf])
        with pytest.raises(ValueError, match="infinities of both signs"):
            _pooled(frame, [], summation="exact")
        with np.errstate(invalid="ignore"):
            assert np.isnan(_pooled(frame, [])["mean"])

    def test_an_overflow_is_refused(self):
        """Finite values whose sum, or whose product with the weight,
        passes the largest float."""
        with pytest.raises(ValueError, match="overflows the largest float"):
            _pooled(_members([1e308, 1e308]), [], summation="exact")
        with pytest.raises(ValueError, match="statistics are infinite"):
            _pooled(_members([1e308, 1.0], [10.0, 1.0]), [], summation="exact")
        assert _pooled(_members([1e308, 1e308]), []) == {"mean": np.inf}

    def test_the_legacy_minus_infinity_is_refused_too(self):
        """The legacy weighted quantile returns -inf below the first
        step. Asked for together with exact sums it is an infinite
        statistic like any other."""
        frame = _members([1.0, 2.0, 3.0], [0.6, 0.2, 0.2])
        with pytest.raises(ValueError, match=r"statistics are infinite \(q05: 1"):
            _pooled(frame, [0.05], summation="exact")

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
                frame, sample_dims=["draw"], quantiles=[1.0], summation="exact"
            )
        message = str(error.value)
        assert "2 statistics are infinite (mean: 1, q100: 1" in message
        assert "'target': 'b'" in message
        assert "summation='exact'" in message


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
            for t in range(5) for scenario in ("low", "high")
            for d in range(9) for year in range(2040, 2050)
        ]
        frame = pd.DataFrame(
            rows, columns=["target", "scenario", "draw", "year", "w"]
        )
        frame["value"] = np.round(rng.normal(size=len(frame)) * 1e6, 0)
        frame.loc[rng.random(len(frame)) < 0.03, "value"] = np.nan
        frame = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
        if not weighted:
            frame = frame.drop(columns="w")
        options = dict(quantile_method="inverted_cdf", summation="exact")
        weight_col = "w" if weighted else None

        whole = summarize_samples(
            frame, sample_dims=["draw"], time_col="year", window=(2042, 2048),
            quantiles=QUANTILES, weight_col=weight_col, include_n_members=True,
            **options,
        )
        reduced = window_means(
            frame, time_col="year", windows=[(2042, 2048)], summation="exact"
        )
        steps = pooled_statistics(
            reduced, sample_dims=["draw"], quantiles=QUANTILES,
            weight_col=weight_col, include_n_members=True, **options,
        ).drop(columns="window")
        pd.testing.assert_frame_equal(
            _sorted(whole), _sorted(steps[whole.columns]), check_exact=True
        )
