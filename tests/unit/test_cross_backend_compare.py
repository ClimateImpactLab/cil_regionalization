"""Cross-backend characterization: local vs BQ canonical weights frames."""
from __future__ import annotations

import pandas as pd
import pytest

from segment_weights.cross_backend_compare import (
    CrossBackendReport,
    compare_outputs,
    load_canonical_parquet,
)


def _frame(rows: list[tuple], weight: str = "popwt") -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["hierid", "cell_ix", "cell_iy", weight],
    )


class TestPerfectAgreement:
    def test_identical_frames_zero_delta(self):
        rows = [
            ("A", 0, 0, 0.5),
            ("A", 0, 1, 0.5),
            ("B", 5, 5, 1.0),
        ]
        df = _frame(rows)
        r = compare_outputs(df, df, weight="popwt", tolerance=1e-3)
        assert r.n_regions_total == 2
        assert r.n_regions_both_native == 2
        assert r.delta_max == 0.0
        assert r.n_regions_above_tolerance == 0
        assert r.mass_displaced_regions == []


class TestBoundaryTail:
    def test_small_boundary_delta_stays_under_tolerance(self):
        local = _frame(
            [
                ("A", 0, 0, 0.4995),
                ("A", 0, 1, 0.5005),
            ]
        )
        bq = _frame(
            [
                ("A", 0, 0, 0.5000),
                ("A", 0, 1, 0.5000),
            ]
        )
        r = compare_outputs(local, bq, weight="popwt", tolerance=1e-3)
        # max|delta| = 0.0005, below 1e-3 tolerance
        assert r.delta_max == pytest.approx(0.0005)
        assert r.n_regions_above_tolerance == 0

    def test_large_delta_counted_above_tolerance(self):
        local = _frame([("A", 0, 0, 0.6), ("A", 0, 1, 0.4)])
        bq = _frame([("A", 0, 0, 0.5), ("A", 0, 1, 0.5)])
        r = compare_outputs(local, bq, weight="popwt", tolerance=1e-3)
        assert r.delta_max == pytest.approx(0.1)
        assert r.n_regions_above_tolerance == 1
        # The cell carrying max weight differs between backends -> displaced.
        # (local picks ix=0, bq has a tie at idxmax; pandas picks the
        # first index, so this isn't quite a displacement case. Use a
        # cleaner setup for the displacement test below.)


class TestMassDisplacement:
    def test_max_weight_cell_swap_flagged(self):
        local = _frame([("A", 0, 0, 0.7), ("A", 0, 1, 0.3)])
        bq = _frame([("A", 0, 0, 0.3), ("A", 0, 1, 0.7)])
        r = compare_outputs(local, bq, weight="popwt", tolerance=1e-3)
        assert r.mass_displaced_regions == ["A"]


class TestNonOverlappingRegions:
    def test_local_only_and_bq_only_counted(self):
        local = _frame([("A", 0, 0, 1.0), ("LOCAL_ONLY", 1, 1, 1.0)])
        bq = _frame([("A", 0, 0, 1.0), ("BQ_ONLY", 2, 2, 1.0)])
        r = compare_outputs(local, bq, weight="popwt", tolerance=1e-3)
        assert r.n_regions_both_native == 1
        assert r.n_regions_local_only == 1
        assert r.n_regions_bq_only == 1


class TestDistributionRestriction:
    def test_nan_in_one_side_does_not_pollute_delta(self):
        local = _frame(
            [
                ("A", 0, 0, 0.5),
                ("A", 0, 1, 0.5),
                ("Z", 9, 9, float("nan")),
            ]
        )
        bq = _frame(
            [
                ("A", 0, 0, 0.6),
                ("A", 0, 1, 0.4),
                ("Z", 9, 9, 1.0),
            ]
        )
        r = compare_outputs(local, bq, weight="popwt", tolerance=1e-3)
        # Only A is both-native; delta = 0.1
        assert r.delta_max == pytest.approx(0.1)
        assert r.n_regions_both_native == 1


class TestSummary:
    def test_summary_contains_tolerance_value(self):
        df = _frame([("A", 0, 0, 1.0)])
        r = compare_outputs(df, df, weight="popwt", tolerance=2.5e-3)
        s = r.summary()
        # {tolerance:g} renders 2.5e-3 as "0.0025"; pin the rendered form.
        assert "0.0025" in s
        assert "max|delta|" in s
        assert "exceeds tolerance" in s


class TestMissingColumns:
    def test_missing_weight_column_named(self):
        bad = pd.DataFrame(
            [("A", 0, 0)], columns=["hierid", "cell_ix", "cell_iy"]
        )
        good = _frame([("A", 0, 0, 1.0)])
        with pytest.raises(ValueError, match="popwt"):
            compare_outputs(bad, good, weight="popwt", tolerance=1e-3)

    def test_missing_join_key_named(self):
        bad = pd.DataFrame(
            [("A", 0, 1.0)], columns=["hierid", "cell_iy", "popwt"]
        )
        good = _frame([("A", 0, 0, 1.0)])
        with pytest.raises(ValueError, match="cell_ix"):
            compare_outputs(bad, good, weight="popwt", tolerance=1e-3)


class TestLoadCanonicalParquet:
    def test_local_parquet_round_trip(self, tmp_path):
        df = _frame([("A", 0, 0, 0.5), ("A", 0, 1, 0.5)])
        p = tmp_path / "weights.parquet"
        df.to_parquet(p, index=False)
        roundtrip = load_canonical_parquet(p)
        pd.testing.assert_frame_equal(roundtrip, df)
