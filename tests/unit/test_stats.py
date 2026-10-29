"""Sample statistics: ordering, pooling, and the closed wrong paths.

Dimension names are deliberately unrealistic (kumquat, zeppelin, epoch)
so nothing can depend on batch, gcm, or year.
"""
from __future__ import annotations

import itertools
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.stats import mixture_quantile, summarize_samples


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestOrderingMatters:
    def test_window_mean_first_differs_from_quantile_first(self):
        """Two anti-phased samples: each has window mean 5, so the pooled
        q95 of window means is 5. Computing the q95 across samples per
        year first and then averaging the window would give about 9.5.
        The function must produce the former."""
        rows = [
            {"kumquat": "a", "epoch": 2050, "value": 0.0},
            {"kumquat": "a", "epoch": 2051, "value": 10.0},
            {"kumquat": "b", "epoch": 2050, "value": 10.0},
            {"kumquat": "b", "epoch": 2051, "value": 0.0},
        ]
        out = summarize_samples(
            _frame(rows),
            sample_dims=["kumquat"],
            time_col="epoch",
            window=(2050, 2051),
            quantiles=[0.95],
            include_mean=False,
        )
        q95 = float(out.loc[out["statistic"] == "q95", "value"].iloc[0])
        assert q95 == pytest.approx(5.0)

        # The reversed order, computed by hand, is a different number.
        df = _frame(rows)
        per_year_q95 = df.groupby("epoch")["value"].quantile(0.95)
        reversed_order = per_year_q95.mean()
        assert reversed_order == pytest.approx(9.5)
        assert q95 != pytest.approx(reversed_order)

    def test_mean_is_order_invariant_control(self):
        # The mean commutes with the window mean; only quantiles do not.
        rows = [
            {"kumquat": "a", "epoch": 2050, "value": 0.0},
            {"kumquat": "a", "epoch": 2051, "value": 10.0},
            {"kumquat": "b", "epoch": 2050, "value": 10.0},
            {"kumquat": "b", "epoch": 2051, "value": 0.0},
        ]
        out = summarize_samples(
            _frame(rows),
            sample_dims=["kumquat"],
            time_col="epoch",
            window=(2050, 2051),
        )
        mean = float(out.loc[out["statistic"] == "mean", "value"].iloc[0])
        assert mean == pytest.approx(5.0)


class TestQuantileOfSumVsSumOfQuantiles:
    def test_aggregate_first_avoids_comonotone_inflation(self):
        """Two anti-correlated units: unit1 draws (0, 10), unit2 draws
        (10, 0). Their sum is 10 in every draw, so the q95 of the
        aggregate is exactly 10. Summing the per-unit q95 values instead
        gives about 19, the comonotone upper bound: that construction
        assumes both units hit their tails together, which these units
        never do. The legacy pipeline summed per-unit quantiles across
        regions and carried exactly this inflation into its tails."""
        aggregated = _frame(
            [
                {"zeppelin": "d1", "epoch": 2050, "value": 10.0},
                {"zeppelin": "d2", "epoch": 2050, "value": 10.0},
            ]
        )
        out = summarize_samples(
            aggregated,
            sample_dims=["zeppelin"],
            time_col="epoch",
            window=(2050, 2050),
            quantiles=[0.95],
            include_mean=False,
        )
        q95_of_sum = float(out["value"].iloc[0])
        assert q95_of_sum == pytest.approx(10.0)

        unit1 = pd.Series([0.0, 10.0])
        unit2 = pd.Series([10.0, 0.0])
        sum_of_q95 = unit1.quantile(0.95) + unit2.quantile(0.95)
        assert sum_of_q95 == pytest.approx(19.0)
        assert q95_of_sum < sum_of_q95


class TestPresummarizedInputRejected:
    def test_statistic_column_raises(self):
        df = _frame(
            [{"statistic": "q50", "zeppelin": "d1", "epoch": 2050, "value": 1.0}]
        )
        with pytest.raises(ValueError, match="comonotone"):
            summarize_samples(
                df,
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(2050, 2050),
            )

    def test_quantile_named_value_column_raises(self):
        df = _frame([{"zeppelin": "d1", "epoch": 2050, "damages_q50": 1.0}])
        with pytest.raises(ValueError, match="quantile-named"):
            summarize_samples(
                df,
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(2050, 2050),
                value_col="damages_q50",
            )

    def test_legacy_long_quantile_column_raises(self):
        df = _frame(
            [
                {
                    "zeppelin": "d1",
                    "epoch": 2050,
                    "value": 1.0,
                    "monetized_deaths_vsl_epa_scaled_q05": 2.0,
                }
            ]
        )
        with pytest.raises(ValueError, match="statistic dimension"):
            summarize_samples(
                df,
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(2050, 2050),
            )


class TestShapeAndPooling:
    def _mc_like(self) -> pd.DataFrame:
        rows = []
        rng = np.random.default_rng(7)
        for tgt, flavor, kum, zep, epoch in itertools.product(
            ["T1", "T2"], ["f1", "f2"], ["k1", "k2", "k3"], ["z1", "z2"],
            [2050, 2051, 2052, 2080],
        ):
            rows.append(
                {
                    "target": tgt,
                    "flavor": flavor,
                    "kumquat": kum,
                    "zeppelin": zep,
                    "epoch": epoch,
                    "value": float(rng.normal()),
                }
            )
        return _frame(rows)

    def test_identity_dims_pass_through(self):
        out = summarize_samples(
            self._mc_like(),
            sample_dims=["kumquat", "zeppelin"],
            time_col="epoch",
            window=(2050, 2052),
            quantiles=[0.05, 0.95],
        )
        assert set(out.columns) == {"target", "flavor", "statistic", "value"}
        assert set(out["statistic"]) == {"mean", "q05", "q95"}
        # 2 targets x 2 flavors x 3 statistics
        assert len(out) == 12

    def test_pooling_is_flat_over_sample_dims(self):
        # The pooled sample per (target, flavor) is 3 kumquats x 2
        # zeppelins = 6 window means; the function's mean must equal a
        # hand-pooled flat mean, every member weighted equally.
        df = self._mc_like()
        out = summarize_samples(
            df,
            sample_dims=["kumquat", "zeppelin"],
            time_col="epoch",
            window=(2050, 2052),
        )
        hand = (
            df.loc[df["epoch"].between(2050, 2052)]
            .groupby(["target", "flavor", "kumquat", "zeppelin"])["value"]
            .mean()
            .groupby(["target", "flavor"])
            .mean()
        )
        got = out.set_index(["target", "flavor"])["value"]
        for key, expected in hand.items():
            assert got.loc[key] == pytest.approx(expected)

    def test_window_bounds_are_arguments(self):
        df = self._mc_like()
        near = summarize_samples(
            df,
            sample_dims=["kumquat", "zeppelin"],
            time_col="epoch",
            window=(2050, 2052),
        )
        far = summarize_samples(
            df,
            sample_dims=["kumquat", "zeppelin"],
            time_col="epoch",
            window=(2080, 2080),
        )
        merged = near.merge(
            far, on=["target", "flavor", "statistic"], suffixes=("_near", "_far")
        )
        assert not np.allclose(merged["value_near"], merged["value_far"])

    def test_no_identity_dims_yields_single_group(self):
        df = _frame(
            [
                {"zeppelin": "d1", "epoch": 2050, "value": 1.0},
                {"zeppelin": "d2", "epoch": 2050, "value": 3.0},
            ]
        )
        out = summarize_samples(
            df, sample_dims=["zeppelin"], time_col="epoch", window=(2050, 2050)
        )
        assert list(out.columns) == ["statistic", "value"]
        assert float(out["value"].iloc[0]) == pytest.approx(2.0)

    def test_nan_sample_member_excluded(self):
        df = _frame(
            [
                {"zeppelin": "d1", "epoch": 2050, "value": 4.0},
                {"zeppelin": "d2", "epoch": 2050, "value": float("nan")},
            ]
        )
        out = summarize_samples(
            df, sample_dims=["zeppelin"], time_col="epoch", window=(2050, 2050)
        )
        assert float(out["value"].iloc[0]) == pytest.approx(4.0)


class TestGuards:
    def _tiny(self) -> pd.DataFrame:
        return _frame([{"zeppelin": "d1", "epoch": 2050, "value": 1.0}])

    def test_empty_window_raises(self):
        with pytest.raises(ValueError, match="selects no rows"):
            summarize_samples(
                self._tiny(),
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(1900, 1901),
            )

    def test_inverted_window_raises(self):
        with pytest.raises(ValueError, match="after window end"):
            summarize_samples(
                self._tiny(),
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(2051, 2050),
            )

    def test_duplicate_rows_raise(self):
        df = pd.concat([self._tiny()] * 2, ignore_index=True)
        with pytest.raises(ValueError, match="duplicated"):
            summarize_samples(
                df, sample_dims=["zeppelin"], time_col="epoch", window=(2050, 2050)
            )

    def test_no_statistics_requested_raises(self):
        with pytest.raises(ValueError, match="no statistics requested"):
            summarize_samples(
                self._tiny(),
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(2050, 2050),
                include_mean=False,
            )

    def test_quantile_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            summarize_samples(
                self._tiny(),
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(2050, 2050),
                quantiles=[1.5],
            )

    def test_quantile_labels(self):
        df = _frame(
            [
                {"zeppelin": "d1", "epoch": 2050, "value": 1.0},
                {"zeppelin": "d2", "epoch": 2050, "value": 2.0},
            ]
        )
        out = summarize_samples(
            df,
            sample_dims=["zeppelin"],
            time_col="epoch",
            window=(2050, 2050),
            quantiles=[0.05, 0.5, 0.95, 0.025],
            include_mean=False,
        )
        assert set(out["statistic"]) == {"q05", "q50", "q95", "q2.5"}


class TestStagedEquivalence:
    """Window mean per leaf then pooled statistics must equal
    concatenate then window mean then statistics. This is the identity
    the pipeline's memory fix rests on; if the two paths ever diverge,
    the staging is no longer a pure re-ordering and must not ship."""

    def _draws(self) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        rows = []
        for tgt, flavor, kum, zep, epoch in itertools.product(
            ["T1", "T2"], ["f1", "f2"], ["k1", "k2", "k3"], ["z1", "z2"],
            [2050, 2051, 2052, 2080, 2081],
        ):
            rows.append(
                {
                    "target": tgt,
                    "flavor": flavor,
                    "kumquat": kum,
                    "zeppelin": zep,
                    "epoch": epoch,
                    "value": float(rng.normal()),
                }
            )
        return _frame(rows)

    def test_reduce_then_pool_equals_pool_then_reduce(self):
        from cil_regionalization.stats import pooled_statistics, window_means

        df = self._draws()
        windows = [(2050, 2052), (2080, 2081)]
        quantiles = [0.05, 0.5, 0.95]

        # Path A: the pipeline's staging. "Leaves" are the kumquat
        # members: reduce each independently, concatenate, pool once.
        leaves = [
            window_means(
                part, time_col="epoch", windows=windows, value_col="value"
            )
            for _, part in df.groupby("kumquat", sort=False)
        ]
        staged = pooled_statistics(
            pd.concat(leaves, ignore_index=True),
            sample_dims=["kumquat", "zeppelin"],
            value_col="value",
            quantiles=quantiles,
        )

        # Path B: the original single-shot ordering.
        pieces = []
        for lo, hi in windows:
            piece = summarize_samples(
                df,
                sample_dims=["kumquat", "zeppelin"],
                time_col="epoch",
                window=(lo, hi),
                quantiles=quantiles,
            )
            piece["window"] = f"{lo}-{hi}"
            pieces.append(piece)
        single = pd.concat(pieces, ignore_index=True)

        key = ["target", "flavor", "window", "statistic"]
        pd.testing.assert_frame_equal(
            staged.sort_values(key).reset_index(drop=True)[key + ["value"]],
            single.sort_values(key).reset_index(drop=True)[key + ["value"]],
            check_exact=False,
            rtol=1e-12,
        )


class TestWindowMeansGuards:
    def _tiny(self) -> pd.DataFrame:
        return _frame(
            [
                {"zeppelin": "d1", "epoch": 2050, "value": 1.0},
                {"zeppelin": "d1", "epoch": 2051, "value": 3.0},
            ]
        )

    def test_already_reduced_input_refused(self):
        from cil_regionalization.stats import window_means

        reduced = window_means(
            self._tiny(), time_col="epoch", windows=[(2050, 2051)]
        )
        assert list(reduced.columns) == ["zeppelin", "window", "value"]
        # A changed window cannot be applied to reduced data: the time
        # dimension is gone, and relabeling would silently misdescribe
        # the contents.
        with pytest.raises(ValueError, match="Windows are final"):
            window_means(reduced, time_col="window", windows=[(2050, 2050)])

    def test_presummarized_input_refused(self):
        from cil_regionalization.stats import window_means

        df = self._tiny().assign(statistic="q50")
        with pytest.raises(ValueError, match="comonotone"):
            window_means(df, time_col="epoch", windows=[(2050, 2051)])

    def test_duplicate_window_labels_refused(self):
        from cil_regionalization.stats import window_means

        with pytest.raises(ValueError, match="duplicate window labels"):
            window_means(
                self._tiny(),
                time_col="epoch",
                windows=[(2050, 2051), (2050, 2051)],
            )

    def test_empty_window_raises_per_window(self):
        from cil_regionalization.stats import window_means

        with pytest.raises(ValueError, match="selects no rows"):
            window_means(
                self._tiny(),
                time_col="epoch",
                windows=[(2050, 2051), (1900, 1901)],
            )


class TestPooledStatisticsGuards:
    def test_presummarized_input_refused(self):
        from cil_regionalization.stats import pooled_statistics

        df = _frame([{"zeppelin": "d1", "statistic": "q50", "value": 1.0}])
        with pytest.raises(ValueError, match="comonotone"):
            pooled_statistics(df, sample_dims=["zeppelin"])

    def test_matches_summarize_samples_shape(self):
        from cil_regionalization.stats import pooled_statistics

        df = _frame(
            [
                {"target": "T1", "zeppelin": "d1", "value": 1.0},
                {"target": "T1", "zeppelin": "d2", "value": 3.0},
            ]
        )
        out = pooled_statistics(df, sample_dims=["zeppelin"], quantiles=[0.5])
        assert list(out.columns) == ["target", "statistic", "value"]
        assert set(out["statistic"]) == {"mean", "q50"}


class TestWeightedStatistics:
    """The weighted path reproduces the historical extraction tool's
    WeightedECDF: weighted average for the mean, left step inverse of
    the weighted empirical distribution for quantiles."""

    @staticmethod
    def _historical_quantile(values, weights, q):
        # A direct transcription of the historical tool's inverse().
        order = sorted(range(len(values)), key=lambda ii: values[ii])
        values = np.array([values[ii] for ii in order], dtype=float)
        weights = np.array([weights[ii] for ii in order], dtype=float)
        pp = np.cumsum(weights) / weights.sum()
        index = int(np.searchsorted(pp, q)) - 1
        if index < 0:
            return float("-inf")
        return float(values[index])

    def _members(self, values, weights):
        rows = [
            {"zeppelin": f"d{i}", "value": v, "w": w}
            for i, (v, w) in enumerate(zip(values, weights))
        ]
        return _frame(rows)

    def test_weighted_mean_is_weighted_average(self):
        from cil_regionalization.stats import pooled_statistics

        values = [1.0, 2.0, 10.0]
        weights = [0.2, 0.2, 0.6]
        out = pooled_statistics(
            self._members(values, weights),
            sample_dims=["zeppelin"],
            weight_col="w",
        )
        mean = float(out.loc[out["statistic"] == "mean", "value"].iloc[0])
        assert mean == pytest.approx(np.average(values, weights=weights))

    def test_quantiles_match_historical_step_inverse(self):
        from cil_regionalization.stats import pooled_statistics

        rng = np.random.default_rng(7)
        values = list(rng.normal(size=15))
        weights = list(rng.uniform(0.01, 0.1, size=15))
        qs = [0.05, 0.17, 0.5, 0.83, 0.95]
        out = pooled_statistics(
            self._members(values, weights),
            sample_dims=["zeppelin"],
            quantiles=qs,
            include_mean=False,
            weight_col="w",
        )
        for q in qs:
            from cil_regionalization.stats import _quantile_label

            got = float(
                out.loc[out["statistic"] == _quantile_label(q), "value"].iloc[0]
            )
            assert got == pytest.approx(
                self._historical_quantile(values, weights, q)
            ), q

    def test_quantile_below_first_step_is_minus_inf(self):
        from cil_regionalization.stats import pooled_statistics

        out = pooled_statistics(
            self._members([1.0, 2.0], [0.5, 0.5]),
            sample_dims=["zeppelin"],
            quantiles=[0.1],
            include_mean=False,
            weight_col="w",
        )
        assert float(out["value"].iloc[0]) == float("-inf")

    def test_equal_weights_median_is_ordinary_median(self):
        from cil_regionalization.stats import pooled_statistics

        values = [1.0, 2.0, 7.0, 11.0]
        out = pooled_statistics(
            self._members(values, [0.25] * 4),
            sample_dims=["zeppelin"],
            quantiles=[0.5],
            include_mean=False,
            weight_col="w",
        )
        assert float(out["value"].iloc[0]) == pytest.approx(np.median(values))

    def test_equal_weights_differ_from_unweighted_off_median(self):
        # The documented definitional difference: the weighted path uses
        # the step distribution, the unweighted path interpolates.
        from cil_regionalization.stats import pooled_statistics

        values = [1.0, 2.0, 7.0, 11.0]
        weighted = pooled_statistics(
            self._members(values, [0.25] * 4),
            sample_dims=["zeppelin"],
            quantiles=[0.95],
            include_mean=False,
            weight_col="w",
        )
        unweighted = pooled_statistics(
            self._members(values, [0.25] * 4).drop(columns="w"),
            sample_dims=["zeppelin"],
            quantiles=[0.95],
            include_mean=False,
        )
        assert float(weighted["value"].iloc[0]) == pytest.approx(
            self._historical_quantile(values, [0.25] * 4, 0.95)
        )
        assert float(unweighted["value"].iloc[0]) == pytest.approx(
            np.quantile(values, 0.95)
        )
        assert float(weighted["value"].iloc[0]) != float(
            unweighted["value"].iloc[0]
        )

    def test_zero_weight_excludes_member(self):
        from cil_regionalization.stats import pooled_statistics

        out = pooled_statistics(
            self._members([1.0, 2.0, 1000.0], [0.5, 0.5, 0.0]),
            sample_dims=["zeppelin"],
            weight_col="w",
        )
        mean = float(out.loc[out["statistic"] == "mean", "value"].iloc[0])
        assert mean == pytest.approx(1.5)

    def test_weight_column_stays_out_of_the_output(self):
        # The weight column is neither an identity dimension nor an
        # output column; the weighted output has the unweighted shape.
        from cil_regionalization.stats import pooled_statistics

        df = self._members([1.0, 2.0, 7.0], [0.9, 0.05, 0.05])
        weighted = pooled_statistics(
            df, sample_dims=["zeppelin"], quantiles=[0.5], weight_col="w"
        )
        unweighted = pooled_statistics(
            df.drop(columns="w"), sample_dims=["zeppelin"], quantiles=[0.5]
        )
        assert list(weighted.columns) == list(unweighted.columns)
        assert "w" not in weighted.columns
        assert len(weighted) == len(unweighted)

    def test_summarize_samples_weighted_end_to_end(self):
        # Window means first, then weighted statistics; the weight rides
        # along the member's time rows.
        rows = []
        for member, (level, w) in {
            "d1": (0.0, 0.6),
            "d2": (10.0, 0.4),
        }.items():
            for year in (2050, 2051):
                rows.append(
                    {"zeppelin": member, "epoch": year, "value": level, "w": w}
                )
        out = summarize_samples(
            _frame(rows),
            sample_dims=["zeppelin"],
            time_col="epoch",
            window=(2050, 2051),
            weight_col="w",
        )
        mean = float(out.loc[out["statistic"] == "mean", "value"].iloc[0])
        assert mean == pytest.approx(4.0)

    def test_weight_varying_within_member_raises(self):
        rows = [
            {"zeppelin": "d1", "epoch": 2050, "value": 1.0, "w": 0.5},
            {"zeppelin": "d1", "epoch": 2051, "value": 1.0, "w": 0.6},
        ]
        with pytest.raises(ValueError, match="varies within"):
            summarize_samples(
                _frame(rows),
                sample_dims=["zeppelin"],
                time_col="epoch",
                window=(2050, 2051),
                weight_col="w",
            )

    def test_bad_weights_raise(self):
        from cil_regionalization.stats import pooled_statistics

        for bad, match in (
            ([0.5, float("nan")], "missing values"),
            ([0.5, -0.1], "negative"),
            ([0.5, float("inf")], "non-finite"),
        ):
            with pytest.raises(ValueError, match=match):
                pooled_statistics(
                    self._members([1.0, 2.0], bad),
                    sample_dims=["zeppelin"],
                    weight_col="w",
                )

    def test_all_zero_weights_raise(self):
        from cil_regionalization.stats import pooled_statistics

        with pytest.raises(ValueError, match="sum to zero"):
            pooled_statistics(
                self._members([1.0, 2.0], [0.0, 0.0]),
                sample_dims=["zeppelin"],
                weight_col="w",
            )

    def test_missing_weight_column_raises(self):
        from cil_regionalization.stats import pooled_statistics

        with pytest.raises(ValueError, match="missing weight column"):
            pooled_statistics(
                self._members([1.0], [1.0]).drop(columns="w"),
                sample_dims=["zeppelin"],
                weight_col="w",
            )


class TestUnweightedGcmWarning:
    def _gcm_frame(self):
        return _frame(
            [
                {"gcm": "m1", "value": 1.0},
                {"gcm": "m2", "value": 2.0},
            ]
        )

    def test_unweighted_gcm_warns(self):
        import warnings

        from cil_regionalization.stats import (
            UnweightedModelWeightsWarning,
            pooled_statistics,
        )

        with pytest.warns(UnweightedModelWeightsWarning, match="convention"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                pooled_statistics(self._gcm_frame(), sample_dims=["gcm"])

    def test_weighted_gcm_does_not_warn(self):
        import warnings

        from cil_regionalization.stats import pooled_statistics

        df = self._gcm_frame().assign(w=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pooled_statistics(df, sample_dims=["gcm"], weight_col="w")

    def test_other_sample_dims_do_not_warn(self):
        import warnings

        from cil_regionalization.stats import pooled_statistics

        df = self._gcm_frame().rename(columns={"gcm": "zeppelin"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pooled_statistics(df, sample_dims=["zeppelin"])

    def test_warning_is_suppressible(self):
        import warnings

        from cil_regionalization.stats import (
            UnweightedModelWeightsWarning,
            pooled_statistics,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings(
                "ignore", category=UnweightedModelWeightsWarning
            )
            pooled_statistics(self._gcm_frame(), sample_dims=["gcm"])


class TestSingleYearWindow:
    """A window whose start equals its end is labeled by the year alone."""

    def _series(self) -> pd.DataFrame:
        return _frame(
            [
                {"zeppelin": d, "epoch": y, "value": v + 10.0 * i}
                for i, d in enumerate(["d1", "d2"])
                for y, v in [(2050, 1.0), (2051, 3.0), (2081, 7.0)]
            ]
        )

    def test_label_is_the_year_alone(self):
        from cil_regionalization.stats import window_means

        out = window_means(
            self._series(),
            time_col="epoch",
            windows=[(2050, 2051), (2081, 2081)],
        )
        assert set(out["window"]) == {"2050-2051", "2081"}

    def test_single_year_window_carries_that_year_untouched(self):
        from cil_regionalization.stats import window_means

        out = window_means(
            self._series(),
            time_col="epoch",
            windows=[(2081, 2081)],
        )
        d1 = out.loc[out["zeppelin"] == "d1", "value"].iloc[0]
        assert d1 == pytest.approx(7.0)

    def test_overlapping_range_and_year_windows_coexist(self):
        # The production request: 2080-2099 and 2099 side by side. The
        # single year is inside the range window, which is fine; the
        # labels differ and each window means independently.
        from cil_regionalization.stats import window_means

        rows = [
            {"zeppelin": "d1", "epoch": 2080, "value": 2.0},
            {"zeppelin": "d1", "epoch": 2099, "value": 4.0},
        ]
        out = window_means(
            _frame(rows),
            time_col="epoch",
            windows=[(2080, 2099), (2099, 2099)],
        )
        by_window = out.set_index("window")["value"]
        assert by_window["2080-2099"] == pytest.approx(3.0)
        assert by_window["2099"] == pytest.approx(4.0)


class TestCategoricalIdentityColumns:
    """Identity columns arriving as pandas categoricals (the passthrough
    pipeline dictionary-encodes its region column for memory) must give
    the same statistics as plain strings, on every pandas version:
    every groupby in the module pins observed=True."""

    def _members(self) -> pd.DataFrame:
        return _frame(
            [
                {"target": t, "zeppelin": d, "value": v, "w": w}
                for t, base in [("T1", 0.0), ("T2", 100.0)]
                for d, v, w in [
                    ("d1", base + 1.0, 0.6),
                    ("d2", base + 3.0, 0.3),
                    ("d3", base + 5.0, 0.1),
                ]
            ]
        )

    def test_weighted_statistics_match_string_identity(self):
        from cil_regionalization.stats import pooled_statistics

        plain = self._members()
        cat = plain.assign(target=plain["target"].astype("category"))
        out_plain = pooled_statistics(
            plain, sample_dims=["zeppelin"], quantiles=[0.5], weight_col="w"
        )
        out_cat = pooled_statistics(
            cat, sample_dims=["zeppelin"], quantiles=[0.5], weight_col="w"
        )
        out_cat["target"] = out_cat["target"].astype(str)
        key = ["target", "statistic"]
        pd.testing.assert_frame_equal(
            out_plain.sort_values(key).reset_index(drop=True),
            out_cat.sort_values(key).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )

    def test_unweighted_statistics_match_string_identity(self):
        from cil_regionalization.stats import pooled_statistics

        plain = self._members().drop(columns="w")
        cat = plain.assign(target=plain["target"].astype("category"))
        out_plain = pooled_statistics(
            plain, sample_dims=["zeppelin"], quantiles=[0.5]
        )
        out_cat = pooled_statistics(cat, sample_dims=["zeppelin"], quantiles=[0.5])
        out_cat["target"] = out_cat["target"].astype(str)
        key = ["target", "statistic"]
        pd.testing.assert_frame_equal(
            out_plain.sort_values(key).reset_index(drop=True),
            out_cat.sort_values(key).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )

    def test_window_means_accept_categorical_group_columns(self):
        from cil_regionalization.stats import window_means

        rows = _frame(
            [
                {"target": t, "epoch": y, "value": v}
                for t in ["T1", "T2"]
                for y, v in [(2050, 1.0), (2051, 3.0)]
            ]
        )
        rows["target"] = rows["target"].astype("category")
        out = window_means(rows, time_col="epoch", windows=[(2050, 2051)])
        assert len(out) == 2
        assert set(out["value"]) == {2.0}


class TestMixtureQuantile:
    QS = [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999]

    def test_one_normal(self):
        got = mixture_quantile([1.0], [3.0], [4.0], self.QS)
        want = [NormalDist(3.0, 2.0).inv_cdf(q) for q in self.QS]
        assert np.allclose(got, want, rtol=0, atol=1e-12)

    def test_identical_normals_are_one_normal(self):
        got = mixture_quantile([0.2, 0.5, 0.3], [-1.0] * 3, [0.25] * 3, self.QS)
        want = [NormalDist(-1.0, 0.5).inv_cdf(q) for q in self.QS]
        assert np.allclose(got, want, rtol=0, atol=1e-12)

    def test_symmetric_pair_has_median_at_center(self):
        assert mixture_quantile([1, 1], [5.0, 9.0], [1.0, 1.0], 0.5) == pytest.approx(7.0, abs=1e-12)
        lo, hi = mixture_quantile([1, 1], [-2.0, 2.0], [0.5, 0.5], [0.1, 0.9])
        assert lo == pytest.approx(-hi, abs=1e-12)

    def test_cdf_at_the_quantile_is_q(self):
        w, m, v = [0.2, 0.5, 0.3], [-3.0, 0.0, 4.0], [1.0, 0.25, 4.0]
        dists = [NormalDist(mi, np.sqrt(vi)) for mi, vi in zip(m, v)]
        for q in self.QS:
            x = mixture_quantile(w, m, v, q)
            assert sum(wi * d.cdf(x) for wi, d in zip(w, dists)) == pytest.approx(q, abs=1e-12)

    def test_far_tails(self):
        for q in (1e-10, 1 - 1e-10):
            got = mixture_quantile([1.0], [0.0], [1.0], q)
            assert got == pytest.approx(NormalDist().inv_cdf(q), rel=1e-9)

    def test_weights_need_not_sum_to_one(self):
        a = mixture_quantile([2.0, 6.0], [0.0, 3.0], [1.0, 2.0], self.QS)
        b = mixture_quantile([0.25, 0.75], [0.0, 3.0], [1.0, 2.0], self.QS)
        assert np.array_equal(a, b)

    def test_zero_weight_component_is_ignored(self):
        a = mixture_quantile([1.0, 0.0], [0.0, 100.0], [1.0, 1.0], self.QS)
        b = mixture_quantile([1.0], [0.0], [1.0], self.QS)
        assert np.array_equal(a, b)

    def test_zero_variance_component_is_a_point_mass(self):
        # half the mass sits at 0, half is N(10, 1): everything between
        # about 0 and 0.5 maps to 0, and 0.75 is the normal's median
        assert mixture_quantile([1, 1], [0.0, 10.0], [0.0, 1.0], 0.25) == pytest.approx(0.0, abs=1e-12)
        assert mixture_quantile([1, 1], [0.0, 10.0], [0.0, 1.0], 0.75) == pytest.approx(10.0, abs=1e-12)
        assert mixture_quantile([1.0], [2.5], [0.0], 0.3) == 2.5

    def test_quantiles_at_zero_and_one(self):
        assert mixture_quantile([1, 1], [0.0, 1.0], [1.0, 0.0], 0.0) == -np.inf
        assert mixture_quantile([1, 1], [0.0, 1.0], [1.0, 0.0], 1.0) == np.inf
        assert mixture_quantile([1, 1], [-1.0, 4.0], [0.0, 0.0], 0.0) == -1.0
        assert mixture_quantile([1, 1], [-1.0, 4.0], [0.0, 0.0], 1.0) == 4.0

    def test_shape_follows_q(self):
        assert isinstance(mixture_quantile([1.0], [0.0], [1.0], 0.5), float)
        out = mixture_quantile([1.0], [0.0], [1.0], [[0.1, 0.2], [0.3, 0.4]])
        assert out.shape == (2, 2)

    def test_regions_match_a_loop_over_regions(self):
        rng = np.random.default_rng(3)
        n_r, n_c = 200, 7
        means = rng.normal(scale=5.0, size=(n_r, n_c))
        variances = rng.gamma(2.0, size=(n_r, n_c))
        variances[::17, 2] = 0.0  # some point masses
        weights = rng.random(n_c)
        weights[4] = 0.0
        qs = [0.0, 0.05, 0.17, 0.5, 0.83, 0.95, 1.0]
        got = mixture_quantile(weights, means, variances, qs)
        assert got.shape == (n_r, len(qs))
        for r in range(n_r):
            want = mixture_quantile(weights, means[r], variances[r], qs)
            assert np.allclose(got[r], want, rtol=1e-13, atol=1e-13)

    def test_per_region_weights_and_leading_shape(self):
        rng = np.random.default_rng(4)
        weights = rng.random((2, 3, 5))
        means = rng.normal(size=(2, 3, 5))
        variances = rng.random((2, 3, 5))
        got = mixture_quantile(weights, means, variances, [0.1, 0.9])
        assert got.shape == (2, 3, 2)
        assert got[1, 2, 0] == pytest.approx(
            mixture_quantile(weights[1, 2], means[1, 2], variances[1, 2], 0.1), abs=1e-13
        )

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError, match="same length"):
            mixture_quantile([1.0, 1.0], [0.0], [1.0], 0.5)
        with pytest.raises(ValueError, match="non-negative"):
            mixture_quantile([-1.0, 2.0], [0.0, 1.0], [1.0, 1.0], 0.5)
        with pytest.raises(ValueError, match="sum to zero"):
            mixture_quantile([0.0], [0.0], [1.0], 0.5)
        with pytest.raises(ValueError, match="sum to zero"):
            mixture_quantile([[1.0, 1.0], [0.0, 0.0]], [0.0, 1.0], [1.0, 1.0], 0.5)
        with pytest.raises(ValueError, match="variances"):
            mixture_quantile([1.0], [0.0], [-1.0], 0.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            mixture_quantile([1.0], [0.0], [1.0], 1.5)
