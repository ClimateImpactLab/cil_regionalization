"""The synthetic weights have the shape of the corn ADM2 file.

Runs at full size whatever `BENCH_SCALE` says, so a change to
`synthetic.py` that drifts from the real file fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cil_regionalization as cilreg
from benchmarks.synthetic import (
    ID_FIELDS,
    N_SOURCE,
    YEARS,
    applied_leaf,
    apply,
    artifact,
    region_ids,
    source_leaf,
)
from cil_regionalization import SourceUnitPolicies

pytestmark = pytest.mark.benchmark

WT = "croparea" + "wt"
N_ROWS, N_TARGETS = 51_216, 42_726
N_EMPTY_REGIONS = 4_391
# targets by the regions they draw on: all with crop area, none, some
N_WITH_AREA, N_WITHOUT_AREA, N_MIXED = 35_602, 6_639, 485
N_IN_OUTPUT = N_WITH_AREA + N_MIXED


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return artifact(1).frame


@pytest.fixture(scope="module")
def targets(frame) -> pd.DataFrame:
    """One row per target, with its number of rows and how many of them
    carry a weight."""
    return frame.groupby(ID_FIELDS)[WT].agg(["size", "count"])


@pytest.fixture(scope="module")
def result():
    return cilreg.apply_weights(
        artifact(1),
        source_leaf(1),
        kind="intensive",
        weight="croparea",
        value_col="pct",
        data_version="synthetic-ir",
        policies=SourceUnitPolicies(on_zero_weight="skip"),
    )


def test_rows_columns_and_order(frame):
    assert len(frame) == N_ROWS
    assert list(frame.columns) == ID_FIELDS + [
        "hierid",
        "croparea_raw",
        WT,
        "croparea_method",
    ]
    for column in ("ISO", "hierid", "croparea_method"):
        assert pd.api.types.is_string_dtype(frame[column]), column
    for column in ("ID_1", "ID_2"):
        assert frame[column].dtype == np.int64, column
    for column in ("croparea_raw", WT):
        assert frame[column].dtype == np.float64, column
    key = ID_FIELDS + ["hierid"]
    assert not frame.duplicated(key).any()
    pd.testing.assert_frame_equal(frame, frame.sort_values(key, ignore_index=True))


def test_distinct_ids(frame, targets):
    assert len(targets) == N_TARGETS
    # ID_1 has 93 values in the file. One country there numbers its
    # first-level units out of sequence, up to 94, and that is not copied.
    assert {c: frame[c].nunique() for c in ID_FIELDS + ["hierid"]} == {
        "ISO": 253,
        "ID_1": 90,
        "ID_2": 5_505,
        "hierid": 24_378,
    }
    assert len(frame[["ISO", "ID_1"]].drop_duplicates()) == 3_428
    per_country = targets.groupby("ISO").size()
    assert per_country.max() == 5_504
    assert (per_country == 1).sum() == 38


def test_ids_match_the_source_leaf(frame):
    """Every region of a leaf is in the weights and the other way round,
    so `apply_weights` finds nothing unmatched and nothing absent."""
    regions = region_ids(N_SOURCE)
    assert set(frame["hierid"]) == set(regions)
    assert set(frame["ISO"]) == {region.split(".")[0] for region in regions}


def test_regions_per_target(targets):
    per_target = targets["size"]
    assert (per_target == 1).sum() == 40_203
    assert round((per_target == 1).mean() * 100, 2) == 94.09
    assert round(per_target.mean(), 4) == 1.1987
    assert per_target.median() == 1
    assert per_target.max() == 48
    several = per_target[per_target > 1].value_counts()
    assert several.sum() == 2_523
    assert several[[2, 3, 4, 13, 34, 48]].tolist() == [1_053, 441, 252, 26, 1, 1]


def test_targets_per_region(frame):
    per_region = frame.groupby("hierid").size()
    assert (per_region == 1).sum() == 21_070
    assert per_region.max() == 184
    several = per_region[per_region > 1].value_counts()
    assert several.sum() == 3_308
    assert several[[2, 3, 4, 48, 184]].tolist() == [885, 451, 270, 4, 1]


def test_rows_without_crop_area(frame, targets):
    empty = frame["croparea_method"] == "nan"
    assert frame["croparea_method"].value_counts().to_dict() == {
        "native": 43_024,
        "nan": 8_192,
    }
    assert frame.loc[empty, [WT, "croparea_raw"]].isna().all(axis=None)
    assert frame.loc[~empty, [WT, "croparea_raw"]].gt(0).all(axis=None)
    # a region has crop area in all of its rows or in none
    weighted = frame.groupby("hierid")[WT].agg(["size", "count"])
    assert (weighted["count"] == 0).sum() == N_EMPTY_REGIONS
    assert (weighted["count"] == weighted["size"]).sum() == 24_378 - N_EMPTY_REGIONS

    assert (targets["count"] == targets["size"]).sum() == N_WITH_AREA
    assert (targets["count"] == 0).sum() == N_WITHOUT_AREA
    mixed = (targets["count"] > 0) & (targets["count"] < targets["size"])
    assert mixed.sum() == N_MIXED
    assert (targets["size"] - targets["count"])[mixed].sum() == 1_112
    sums = frame.groupby(ID_FIELDS)[WT].sum(min_count=1).dropna()
    assert len(sums) == N_IN_OUTPUT
    np.testing.assert_allclose(sums, 1.0, rtol=0, atol=1e-12)


def test_default_policy_stops_at_the_empty_regions():
    with pytest.raises(ValueError, match=f"{N_EMPTY_REGIONS} zero_weight source units"):
        cilreg.apply_weights(
            artifact(1),
            source_leaf(1),
            kind="intensive",
            weight="croparea",
            value_col="pct",
            data_version="synthetic-ir",
        )


def test_skip_reports_regions_and_drops_targets(result, targets, caplog):
    """The library reports the skipped regions on the result and in a
    warning, and the targets left with nothing in the result's targets
    and in a second warning. Those targets are missing from the frame.
    Failed with the targets counted nowhere."""
    assert result.coverage.counts == {
        "unmatched": 0,
        "zero_weight": N_EMPTY_REGIONS,
        "absent_from_data": 0,
    }
    out = result.frame
    assert len(out) == N_IN_OUTPUT * len(YEARS) == 4_258_266
    assert out["pct"].notna().all()
    in_output = pd.MultiIndex.from_frame(out[ID_FIELDS].drop_duplicates())
    assert len(in_output) == N_IN_OUTPUT
    assert not targets.index[targets["count"] == 0].isin(in_output).any()
    assert targets.index[targets["count"] > 0].isin(in_output).all()
    assert result.targets.counts["left_out"] == {
        "no weight": N_WITHOUT_AREA,
        "no data": 0,
    }

    with caplog.at_level("WARNING", logger="cil_regionalization.validate"):
        apply(artifact(64), source_leaf(64))
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "zero_weight source units" in messages[0]
    assert "targets are left out (no weight:" in messages[1]


def test_applied_leaf_is_what_apply_returns(result):
    keys = ID_FIELDS + ["year"]
    leaf = applied_leaf(1)
    assert len(leaf) == 4_258_266
    pd.testing.assert_frame_equal(leaf[keys], result.frame[keys])
    assert leaf["pct"].dtype == result.frame["pct"].dtype


@pytest.mark.parametrize("scale", [3, 8, 64])
def test_scaled_frames_keep_the_proportions(scale):
    art, leaf = artifact(scale), source_leaf(scale)
    frame = art.frame
    assert set(frame["hierid"]) == set(leaf["hierid"])
    assert not frame.duplicated(ID_FIELDS + ["hierid"]).any()
    per_target = frame.groupby(ID_FIELDS)[WT].agg(["size", "count"])
    assert len(per_target) == pytest.approx(N_TARGETS / scale, rel=0.03)
    assert (per_target["size"] == 1).mean() == pytest.approx(0.9409, abs=0.005)
    assert per_target["size"].max() <= 48
    without_area = (per_target["count"] == 0).mean()
    assert without_area == pytest.approx(N_WITHOUT_AREA / N_TARGETS, rel=0.4)

    keys = ID_FIELDS + ["year"]
    pd.testing.assert_frame_equal(applied_leaf(scale)[keys], apply(art, leaf)[keys])
