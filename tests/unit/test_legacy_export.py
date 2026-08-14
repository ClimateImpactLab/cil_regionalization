"""Legacy 13-column CSV exporter: schema, totals, NaN, shpid, errors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from segment_weights.legacy_export import (
    LEGACY_COLUMNS,
    to_legacy_frame,
    write_legacy_csv,
)
from segment_weights.schema import OutputSchema


@dataclass
class _StubResult:
    """Stand-in for WeightsResult so the tests don't depend on backend code."""

    frame: pd.DataFrame
    schema: OutputSchema


def _three_region_frame() -> pd.DataFrame:
    """Three regions A, B, C with area/pop/crop columns and one NaN region."""
    rows = []
    # Region B: native (3 cells, fractions of total)
    rows.append(("B", 0, 0, 0.5, 0.5, 0.6, 60.0, "native", 0.7, 70.0, "native", 0.5, 6e9, "native"))
    rows.append(("B", 0, 1, 0.5, 1.5, 0.4, 40.0, "native", 0.3, 30.0, "native", 0.5, 6e9, "native"))
    # Region A: a single cell with all three weights present
    rows.append(("A", 5, 5, 5.5, 5.5, 1.0, 100.0, "native", 1.0, 200.0, "native", 1.0, 1e10, "native"))
    # Region C: zero crop -> NaN cropwt under 'nan' policy
    rows.append(("C", 10, 10, 10.5, 10.5, 0.5, 50.0, "native", 0.5, 100.0, "native", float("nan"), 0.0, "nan"))
    rows.append(("C", 10, 11, 10.5, 11.5, 0.5, 50.0, "native", 0.5, 100.0, "native", float("nan"), 0.0, "nan"))
    df = pd.DataFrame(
        rows,
        columns=[
            "hierid", "cell_ix", "cell_iy", "cell_lon", "cell_lat",
            "areawt", "area_raw", "area_method",
            "popwt", "pop_raw", "pop_method",
            "cropwt", "crop_raw", "crop_method",
        ],
    )
    return df


def _result_for(weight_names: tuple[str, ...]) -> _StubResult:
    return _StubResult(
        frame=_three_region_frame(),
        schema=OutputSchema(id_fields=("hierid",), weight_names=weight_names),
    )


class TestLegacyFrameShape:
    def test_columns_in_canonical_order(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        assert list(out.columns) == list(LEGACY_COLUMNS)

    def test_thirteen_columns(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        assert len(out.columns) == 13


class TestPerRegionTotals:
    def test_totals_broadcast_to_every_cell_in_region(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        # B has two cells, area=60+40=100, pop=70+30=100, crop=6e9+6e9=1.2e10
        b = out.loc[out["hierid"] == "B"]
        assert b["areatotal"].tolist() == pytest.approx([100.0, 100.0])
        assert b["poptotal"].tolist() == pytest.approx([100.0, 100.0])
        assert b["croptotal"].tolist() == pytest.approx([1.2e10, 1.2e10])

    def test_zero_crop_region_has_zero_croptotal(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        c = out.loc[out["hierid"] == "C"]
        assert bool((c["croptotal"] == 0.0).all())
        # but areatotal and poptotal are non-zero (region still has cells)
        assert bool((c["areatotal"] > 0).all())
        assert bool((c["poptotal"] > 0).all())


class TestNaNPreservation:
    def test_cropwt_nan_passed_through(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        c = out.loc[out["hierid"] == "C"]
        assert c["cropwt"].isna().all()

    def test_native_cropwt_not_nan(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        for hid in ("A", "B"):
            rows = out.loc[out["hierid"] == hid]
            assert not rows["cropwt"].isna().any()


class TestShpid:
    def test_dense_rank_zero_based_sorted(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        # Sorted hierids: A, B, C -> 0, 1, 2
        rank_per = (
            out[["hierid", "shpid"]].drop_duplicates().set_index("hierid")["shpid"]
        )
        assert rank_per["A"] == 0
        assert rank_per["B"] == 1
        assert rank_per["C"] == 2

    def test_shpid_int64(self):
        out = to_legacy_frame(_result_for(("area", "crop", "pop")))
        assert out["shpid"].dtype == np.int64


class TestMissingInputs:
    def test_missing_required_weight_raises_with_name(self):
        with pytest.raises(ValueError, match="missing required weights"):
            to_legacy_frame(_result_for(("area", "pop")))

    def test_missing_pop_named_in_error(self):
        try:
            to_legacy_frame(_result_for(("area", "crop")))
        except ValueError as exc:
            assert "pop" in str(exc)
        else:
            pytest.fail("expected ValueError")

    def test_non_hierid_id_field_rejected(self):
        bad = _StubResult(
            frame=_three_region_frame().rename(columns={"hierid": "region_id"}),
            schema=OutputSchema(
                id_fields=("region_id",),
                weight_names=("area", "crop", "pop"),
            ),
        )
        with pytest.raises(ValueError, match="hierid"):
            to_legacy_frame(bad)


class TestCsvRoundTrip:
    def test_write_legacy_csv_round_trip(self, tmp_path):
        result = _result_for(("area", "crop", "pop"))
        path = tmp_path / "weights_legacy.csv"
        out = write_legacy_csv(result, path)
        assert Path(out).exists()
        # Read back and confirm columns + the cropwt NaN survived
        # the CSV round trip.
        df = pd.read_csv(out)
        assert list(df.columns) == list(LEGACY_COLUMNS)
        c = df.loc[df["hierid"] == "C"]
        assert c["cropwt"].isna().all()
        # shpid round-trips as int
        assert pd.api.types.is_integer_dtype(df["shpid"])
