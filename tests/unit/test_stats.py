"""Sample statistics: ordering, pooling, and the closed wrong paths.

Dimension names are deliberately unrealistic (kumquat, zeppelin, epoch)
so nothing can depend on batch, gcm, or year.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.stats import summarize_samples


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
