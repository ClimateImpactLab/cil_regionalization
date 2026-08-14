"""Characterization comparison vs legacy CSV: distribution + class counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from segment_weights.legacy_compare import (
    ComparisonReport,
    compare_weights,
    load_legacy_csv,
)


def _legacy_schema(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["hierid", "pix_cent_x", "pix_cent_y", "cropwt"],
    )


class TestPerfectAgreement:
    def test_zero_delta_when_identical(self):
        rows = [
            ("A", 0.125, 0.125, 0.5),
            ("A", 0.375, 0.125, 0.5),
            ("B", 1.125, 1.125, 1.0),
        ]
        df = _legacy_schema(rows)
        r = compare_weights(df, df, weight="crop")
        assert r.n_regions_total == 2
        assert r.n_regions_both_native == 2
        assert r.delta_max == 0.0
        assert r.delta_median == 0.0
        assert r.delta_p95 == 0.0
        assert r.mass_displaced_regions == []


class TestBoundaryDelta:
    """Small deltas (the expected boundary-noise case) should appear in
    the distribution but not in mass_displaced_regions."""

    def test_small_boundary_delta_does_not_displace_mass(self):
        ours = _legacy_schema(
            [
                ("A", 0.125, 0.125, 0.70),
                ("A", 0.375, 0.125, 0.30),
            ]
        )
        legacy = _legacy_schema(
            [
                ("A", 0.125, 0.125, 0.72),
                ("A", 0.375, 0.125, 0.28),
            ]
        )
        r = compare_weights(ours, legacy, weight="crop")
        # max|delta| on A = max(0.02, 0.02) = 0.02
        assert r.delta_max == pytest.approx(0.02)
        # Max-weight cell is (.125, .125) in both -> no displacement.
        assert r.mass_displaced_regions == []
        assert r.n_regions_both_native == 1


class TestMassDisplacement:
    def test_max_weight_cell_swap_flagged(self):
        """If the maximum-weight cell flips between cells, the region
        is flagged as mass-displaced; a real spatial-disagreement signal."""
        ours = _legacy_schema(
            [
                ("A", 0.125, 0.125, 0.70),
                ("A", 0.375, 0.125, 0.30),
            ]
        )
        legacy = _legacy_schema(
            [
                ("A", 0.125, 0.125, 0.30),
                ("A", 0.375, 0.125, 0.70),
            ]
        )
        r = compare_weights(ours, legacy, weight="crop")
        assert r.mass_displaced_regions == ["A"]


class TestNanClassification:
    def test_both_nan_region_counted(self):
        ours = _legacy_schema(
            [
                ("A", 0.1, 0.1, 1.0),
                ("Z", 9.0, 9.0, float("nan")),
                ("Z", 9.0, 9.5, float("nan")),
            ]
        )
        legacy = _legacy_schema(
            [
                ("A", 0.1, 0.1, 1.0),
                ("Z", 9.0, 9.0, float("nan")),
                ("Z", 9.0, 9.5, float("nan")),
            ]
        )
        r = compare_weights(ours, legacy, weight="crop")
        assert r.n_regions_both_nan == 1
        assert r.n_regions_both_native == 1
        assert r.n_regions_ours_nan_legacy_native == 0
        assert r.n_regions_legacy_nan_ours_native == 0

    def test_ours_nan_legacy_native_counted(self):
        ours = _legacy_schema(
            [
                ("Z", 9.0, 9.0, float("nan")),
            ]
        )
        legacy = _legacy_schema(
            [
                ("Z", 9.0, 9.0, 1.0),
            ]
        )
        r = compare_weights(ours, legacy, weight="crop")
        assert r.n_regions_ours_nan_legacy_native == 1
        assert r.n_regions_both_native == 0
        assert r.n_regions_both_nan == 0

    def test_legacy_nan_ours_native_counted(self):
        ours = _legacy_schema([("Z", 9.0, 9.0, 1.0)])
        legacy = _legacy_schema([("Z", 9.0, 9.0, float("nan"))])
        r = compare_weights(ours, legacy, weight="crop")
        assert r.n_regions_legacy_nan_ours_native == 1


class TestDistribution:
    def test_distribution_uses_native_only(self):
        """Regions that are NaN on one side must NOT pollute the delta
        distribution; only both-native regions contribute."""
        ours = _legacy_schema(
            [
                ("A", 0.1, 0.1, 0.5),
                ("A", 0.2, 0.1, 0.5),
                ("Z", 9.0, 9.0, float("nan")),
            ]
        )
        legacy = _legacy_schema(
            [
                ("A", 0.1, 0.1, 0.6),
                ("A", 0.2, 0.1, 0.4),
                ("Z", 9.0, 9.0, 1.0),
            ]
        )
        r = compare_weights(ours, legacy, weight="crop")
        # A only -> max|delta| = 0.1
        assert r.delta_max == pytest.approx(0.1)
        assert r.n_regions_both_native == 1


class TestSummary:
    def test_summary_lists_all_buckets(self):
        ours = _legacy_schema([("A", 0.1, 0.1, 1.0)])
        legacy = _legacy_schema([("A", 0.1, 0.1, 1.0)])
        r = compare_weights(ours, legacy, weight="crop")
        s = r.summary()
        assert "total regions" in s
        assert "both native" in s
        assert "both NaN" in s
        assert "NaN mismatches" in s
        assert "mass-displaced" in s
        assert "max|delta cropwt|" in s


class TestMissingColumns:
    def test_missing_wt_column_named(self):
        bad = pd.DataFrame(
            [("A", 0.1, 0.1)],
            columns=["hierid", "pix_cent_x", "pix_cent_y"],
        )
        good = _legacy_schema([("A", 0.1, 0.1, 1.0)])
        with pytest.raises(ValueError, match="cropwt"):
            compare_weights(bad, good, weight="crop")


class TestLoadCsv:
    def test_load_legacy_csv_round_trip(self, tmp_path):
        df = _legacy_schema(
            [
                ("A", 0.125, 0.125, 0.5),
                ("A", 0.375, 0.125, 0.5),
            ]
        )
        p = tmp_path / "legacy.csv"
        df.to_csv(p, index=False)
        roundtrip = load_legacy_csv(p)
        pd.testing.assert_frame_equal(roundtrip, df)
