"""Merge invariants: disjoint hierid sets, coverage, sum-to-1, source col."""
from __future__ import annotations

import pandas as pd
import pytest

from segment_weights.merge import (
    SOURCE_BIGQUERY,
    SOURCE_SHAPEFILE_SUPPLEMENT,
    merge_weights,
)
from segment_weights.schema import OutputSchema


def _row(hid: str, ix: int, iy: int, popwt: float, areawt: float) -> dict:
    return {
        "hierid": hid,
        "cell_ix": ix,
        "cell_iy": iy,
        "cell_lon": (ix + 0.5) - 180.0,
        "cell_lat": (iy + 0.5) - 90.0,
        "popwt": popwt,
        "pop_raw": popwt * 100.0,
        "pop_method": "native",
        "areawt": areawt,
        "area_raw": areawt * 1.0e9,
        "area_method": "native",
    }


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("ABW", 109, 102, 0.4, 0.5),
            _row("ABW", 109, 103, 0.6, 0.5),
            _row("AND.Ra5cc0db7a54d1bb3", 181, 132, 1.0, 1.0),
        ]
    )


def _supplement_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("CAN.5", 122, 145, 0.3, 0.7),
            _row("CAN.5", 122, 146, 0.7, 0.3),
            _row("ITA.8.39", 192, 131, 1.0, 1.0),
        ]
    )


def _manifest(**extra) -> dict:
    return {
        "config_hash": "abc123",
        "inputs": {"regions_table": "x.y.z"},
        "extra": extra,
    }


def _schema() -> OutputSchema:
    return OutputSchema(
        id_fields=("hierid",), weight_names=("pop", "area")
    )


class TestHappyPath:
    def test_merge_combines_both_sources(self):
        result = merge_weights(
            canonical_frame=_canonical_frame(),
            canonical_manifest=_manifest(),
            supplement_frame=_supplement_frame(),
            supplement_manifest=_manifest(),
            schema=_schema(),
            expected_total=4,
        )
        assert result.n_canonical == 3
        assert result.n_supplement == 3
        assert result.n_regions == 4
        assert "source" in result.frame.columns
        canon_src = result.frame.loc[result.frame["hierid"] == "ABW", "source"]
        supp_src = result.frame.loc[result.frame["hierid"] == "CAN.5", "source"]
        assert (canon_src == SOURCE_BIGQUERY).all()
        assert (supp_src == SOURCE_SHAPEFILE_SUPPLEMENT).all()

    def test_combined_sums_to_one(self):
        result = merge_weights(
            canonical_frame=_canonical_frame(),
            canonical_manifest=_manifest(),
            supplement_frame=_supplement_frame(),
            supplement_manifest=_manifest(),
            schema=_schema(),
            expected_total=4,
        )
        sums = result.frame.groupby("hierid")[["popwt", "areawt"]].sum()
        for col in ("popwt", "areawt"):
            assert (sums[col] - 1.0).abs().max() < 1e-9

    def test_merged_manifest_records_both_sources(self):
        canon_mf = _manifest(
            null_geometry_count=17,
            null_geometry_regions=["CAN.5", "ITA.8.39"],
            bq_compute_location="US",
            bq_dry_run_bytes=5_257_870_075,
        )
        supp_mf = _manifest(
            repaired_geometry_count=0,
            repaired_geometry_regions=[],
            bq_compute_location="US",
            bq_dry_run_bytes=1_000_000,
        )
        result = merge_weights(
            canonical_frame=_canonical_frame(),
            canonical_manifest=canon_mf,
            supplement_frame=_supplement_frame(),
            supplement_manifest=supp_mf,
            schema=_schema(),
            expected_total=4,
        )
        mf = result.merged_manifest
        assert mf["expected_total_regions"] == 4
        assert mf["combined_unique_regions"] == 4
        assert mf["row_counts"] == {"canonical": 3, "supplement": 3, "total": 6}
        assert mf["sources"]["canonical"]["config_hash"] == "abc123"
        assert mf["sources"]["canonical"]["extra"]["null_geometry_count"] == 17
        assert (
            mf["sources"]["canonical"]["extra"]["bq_dry_run_bytes"]
            == 5_257_870_075
        )
        assert mf["sources"]["supplement"]["config_hash"] == "abc123"
        assert mf["sources"]["supplement"]["extra"]["repaired_geometry_count"] == 0


class TestDisjoint:
    def test_overlap_raises_with_ids_named(self):
        # ABW is in both frames.
        supp = pd.DataFrame(
            [_row("ABW", 109, 104, 1.0, 1.0)]
        )
        with pytest.raises(ValueError, match="overlap") as exc:
            merge_weights(
                canonical_frame=_canonical_frame(),
                canonical_manifest=_manifest(),
                supplement_frame=supp,
                supplement_manifest=_manifest(),
                schema=_schema(),
                expected_total=3,
            )
        assert "ABW" in str(exc.value)


class TestCoverage:
    def test_undercount_raises(self):
        with pytest.raises(ValueError, match="unique hierids"):
            merge_weights(
                canonical_frame=_canonical_frame(),
                canonical_manifest=_manifest(),
                supplement_frame=_supplement_frame(),
                supplement_manifest=_manifest(),
                schema=_schema(),
                expected_total=24_378,  # expected 24,378, got 4
            )

    def test_overcount_raises(self):
        with pytest.raises(ValueError, match="unique hierids"):
            merge_weights(
                canonical_frame=_canonical_frame(),
                canonical_manifest=_manifest(),
                supplement_frame=_supplement_frame(),
                supplement_manifest=_manifest(),
                schema=_schema(),
                expected_total=2,
            )


class TestSumToOnePreserved:
    def test_broken_supplement_sum_raises(self):
        broken = _supplement_frame()
        # Push one supplement row's popwt off; merge should refuse.
        broken.loc[0, "popwt"] = 5.0
        with pytest.raises(ValueError, match="sum-to-1 failed"):
            merge_weights(
                canonical_frame=_canonical_frame(),
                canonical_manifest=_manifest(),
                supplement_frame=broken,
                supplement_manifest=_manifest(),
                schema=_schema(),
                expected_total=4,
            )


class TestSourceColumn:
    def test_source_column_has_two_values_only(self):
        result = merge_weights(
            canonical_frame=_canonical_frame(),
            canonical_manifest=_manifest(),
            supplement_frame=_supplement_frame(),
            supplement_manifest=_manifest(),
            schema=_schema(),
            expected_total=4,
        )
        assert set(result.frame["source"]) == {
            SOURCE_BIGQUERY,
            SOURCE_SHAPEFILE_SUPPLEMENT,
        }


class TestSchemaShape:
    def test_missing_column_in_canonical_raises(self):
        bad = _canonical_frame().drop(columns=["popwt"])
        with pytest.raises(ValueError, match="canonical frame missing columns"):
            merge_weights(
                canonical_frame=bad,
                canonical_manifest=_manifest(),
                supplement_frame=_supplement_frame(),
                supplement_manifest=_manifest(),
                schema=_schema(),
                expected_total=4,
            )

    def test_missing_column_in_supplement_raises(self):
        bad = _supplement_frame().drop(columns=["areawt"])
        with pytest.raises(ValueError, match="supplement frame missing columns"):
            merge_weights(
                canonical_frame=_canonical_frame(),
                canonical_manifest=_manifest(),
                supplement_frame=bad,
                supplement_manifest=_manifest(),
                schema=_schema(),
                expected_total=4,
            )
