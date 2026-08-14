"""Fallback policy application: per-region native, area, zero, error paths."""
from __future__ import annotations

import pandas as pd
import pytest

from cil_regionalization.fallback import (
    apply_fallback,
    compute_native_weights,
    count_methods,
)


def _area_frame() -> pd.DataFrame:
    """Three regions, three cells each. Region C has uniform area; A and B
    skew toward cell 0 to make the native vs fallback distinction visible.
    """
    return pd.DataFrame(
        [
            ("A", 0, 0, 0.5, 0.5, 0.5, 50.0),
            ("A", 0, 1, 0.5, 1.5, 0.3, 30.0),
            ("A", 1, 0, 1.5, 0.5, 0.2, 20.0),
            ("B", 5, 5, 5.5, 5.5, 0.6, 60.0),
            ("B", 5, 6, 5.5, 6.5, 0.4, 40.0),
            ("C", 10, 10, 10.5, 10.5, 0.5, 50.0),
            ("C", 10, 11, 10.5, 11.5, 0.5, 50.0),
        ],
        columns=[
            "region_id",
            "cell_ix",
            "cell_iy",
            "cell_lon",
            "cell_lat",
            "areawt",
            "area_raw",
        ],
    )


def _raw_frame_AB_only() -> pd.DataFrame:
    """A and B have raw values; C is missing entirely (treated as zero)."""
    return pd.DataFrame(
        [
            ("A", 0, 0, 100.0),
            ("A", 0, 1, 100.0),
            ("A", 1, 0, 0.0),
            ("B", 5, 5, 8.0),
            ("B", 5, 6, 2.0),
        ],
        columns=["region_id", "cell_ix", "cell_iy", "raw"],
    )


class TestNative:
    def test_native_normalization_to_one(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        # restrict so all regions have raw > 0
        raw = raw.loc[raw["region_id"] != "B"].copy()
        raw["region_id"] = pd.Categorical(raw["region_id"], categories=["A"])
        area = area.loc[area["region_id"] == "A"].copy()
        out = apply_fallback(raw, area, "pop", "area", ["region_id"])
        sums = out.groupby("region_id")["popwt"].sum()
        assert sums.loc["A"] == pytest.approx(1.0)
        assert (out["pop_method"] == "native").all()

    def test_native_zero_cell_kept_with_zero_weight(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        # A has cell (1,0) with raw=0; should appear in output with weight 0
        out = apply_fallback(raw, area, "pop", "area", ["region_id"])
        a_rows = out.loc[out["region_id"] == "A"].sort_values(["cell_ix", "cell_iy"])
        assert len(a_rows) == 3
        # raw values 100, 100, 0 → weights 0.5, 0.5, 0
        assert list(a_rows["popwt"]) == pytest.approx([0.5, 0.5, 0.0])

    def test_compute_native_weights_for_area(self):
        area = _area_frame()
        out = compute_native_weights(
            area.rename(columns={"area_raw": "raw"}),
            ["region_id"],
            "area",
            raw_col="raw",
        )
        sums = out.groupby("region_id")["areawt"].sum()
        for region in ["A", "B", "C"]:
            assert sums.loc[region] == pytest.approx(1.0)
        assert (out["area_method"] == "native").all()


class TestFallbackArea:
    def test_zero_raw_region_falls_back_to_area(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        out = apply_fallback(raw, area, "pop", "area", ["region_id"])
        c_rows = out.loc[out["region_id"] == "C"]
        # C has no raw at all → falls back to area weights (0.5, 0.5)
        assert (c_rows["pop_method"] == "area_fallback").all()
        assert c_rows["popwt"].sum() == pytest.approx(1.0)
        assert list(c_rows["popwt"]) == pytest.approx([0.5, 0.5])

    def test_native_and_fallback_in_same_output(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        out = apply_fallback(raw, area, "pop", "area", ["region_id"])
        # A, B: native; C: area_fallback
        methods = out.groupby("region_id")["pop_method"].first()
        assert methods.loc["A"] == "native"
        assert methods.loc["B"] == "native"
        assert methods.loc["C"] == "area_fallback"


class TestFallbackZero:
    def test_zero_raw_region_gets_zero_weights(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        out = apply_fallback(raw, area, "pop", "zero", ["region_id"])
        c_rows = out.loc[out["region_id"] == "C"]
        assert (c_rows["pop_method"] == "zero").all()
        assert (c_rows["popwt"] == 0.0).all()
        # native regions still normalize to 1
        a_sum = out.loc[out["region_id"] == "A", "popwt"].sum()
        assert a_sum == pytest.approx(1.0)


class TestFallbackNan:
    def test_zero_raw_region_gets_nan_weights(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        out = apply_fallback(raw, area, "pop", "nan", ["region_id"])
        c_rows = out.loc[out["region_id"] == "C"]
        assert (c_rows["pop_method"] == "nan").all()
        assert c_rows["popwt"].isna().all()
        # native regions still normalise to 1
        a_sum = out.loc[out["region_id"] == "A", "popwt"].sum()
        assert a_sum == pytest.approx(1.0)

    def test_nan_regions_invisible_to_sum_to_one_validator(self):
        """The validator must NOT flag NaN regions: NaN > tolerance is
        False, so a region whose weights are NaN is silently ignored."""
        from cil_regionalization.schema import OutputSchema
        from cil_regionalization.validate import check_sum_to_one

        # Hand-built frame: region A sums to 1.0; region B is all NaN.
        df = pd.DataFrame(
            [
                ("A", 0, 0, 0.5, 0.5, 0.6, 60.0, "native"),
                ("A", 0, 1, 0.5, 1.5, 0.4, 40.0, "native"),
                ("B", 5, 5, 5.5, 5.5, float("nan"), 0.0, "nan"),
                ("B", 5, 6, 5.5, 6.5, float("nan"), 0.0, "nan"),
            ],
            columns=[
                "region_id",
                "cell_ix",
                "cell_iy",
                "cell_lon",
                "cell_lat",
                "popwt",
                "pop_raw",
                "pop_method",
            ],
        )
        schema = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        report = check_sum_to_one(df, schema)
        assert report.ok, (
            f"NaN regions must not be flagged; failures: {report.failures}"
        )


class TestImplicitDefaultWarning:
    def test_implicit_default_warns_when_threshold_crossed(self, caplog):
        """When the fallback fires for >= max(5, 5% of regions) and the
        policy was not set explicitly in TOML, log a WARN naming the
        applied default + alternatives."""
        # Build 20 regions; 10 of them have zero raw.
        area_rows = []
        raw_rows = []
        for i in range(20):
            rid = f"R{i:02d}"
            area_rows.append((rid, i, 0, float(i), float(0), 1.0, 1e9))
            if i < 10:
                # zero raw -> fallback fires
                pass
            else:
                raw_rows.append((rid, i, 0, 100.0))
        area = pd.DataFrame(
            area_rows,
            columns=[
                "region_id",
                "cell_ix",
                "cell_iy",
                "cell_lon",
                "cell_lat",
                "areawt",
                "area_raw",
            ],
        )
        raw = pd.DataFrame(
            raw_rows,
            columns=["region_id", "cell_ix", "cell_iy", "raw"],
        )

        with caplog.at_level("WARNING", logger="cil_regionalization.fallback"):
            apply_fallback(
                raw, area, "pop", "area", ["region_id"],
                policy_explicit=False,
            )
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("library default" in m for m in msgs)
        assert any("[area | zero | error | nan]" in m for m in msgs)
        assert any("weights.pop" in m for m in msgs)
        assert any("Set weights.pop.fallback" in m for m in msgs)

    def test_explicit_policy_does_not_warn(self, caplog):
        """If the user wrote `fallback = "area"` in TOML, no warning
        even with the same fallback-fire pattern."""
        area = pd.DataFrame(
            [
                (f"R{i}", i, 0, float(i), 0.0, 1.0, 1e9)
                for i in range(20)
            ],
            columns=["region_id", "cell_ix", "cell_iy", "cell_lon",
                     "cell_lat", "areawt", "area_raw"],
        )
        raw = pd.DataFrame(
            [(f"R{i}", i, 0, 100.0) for i in range(10, 20)],
            columns=["region_id", "cell_ix", "cell_iy", "raw"],
        )
        with caplog.at_level("WARNING", logger="cil_regionalization.fallback"):
            apply_fallback(
                raw, area, "pop", "area", ["region_id"],
                policy_explicit=True,
            )
        warn_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert not any("library default" in m for m in warn_msgs)

    def test_implicit_default_below_threshold_does_not_warn(self, caplog):
        """1 fallback region out of 20 is below the max(5, 5%) threshold."""
        area = pd.DataFrame(
            [
                (f"R{i}", i, 0, float(i), 0.0, 1.0, 1e9)
                for i in range(20)
            ],
            columns=["region_id", "cell_ix", "cell_iy", "cell_lon",
                     "cell_lat", "areawt", "area_raw"],
        )
        # 19 regions with raw, only R0 falls back
        raw = pd.DataFrame(
            [(f"R{i}", i, 0, 100.0) for i in range(1, 20)],
            columns=["region_id", "cell_ix", "cell_iy", "raw"],
        )
        with caplog.at_level("WARNING", logger="cil_regionalization.fallback"):
            apply_fallback(
                raw, area, "pop", "area", ["region_id"],
                policy_explicit=False,
            )
        warn_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert not any("library default" in m for m in warn_msgs)


class TestFallbackError:
    def test_zero_raw_region_raises_with_id(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        with pytest.raises(ValueError) as exc:
            apply_fallback(raw, area, "pop", "error", ["region_id"])
        msg = str(exc.value)
        assert "pop" in msg
        assert "C" in msg


class TestDegenerateArea:
    def test_zero_area_region_always_raises(self):
        """A region present in raw but absent from the area frame implies
        zero total area for that region; no fallback can recover, regardless
        of policy.
        """
        area = _area_frame().loc[lambda d: d["region_id"] != "C"].copy()
        raw = pd.DataFrame(
            [("C", 99, 99, 0.0)],
            columns=["region_id", "cell_ix", "cell_iy", "raw"],
        )
        with pytest.raises(ValueError) as exc:
            apply_fallback(raw, area, "pop", "area", ["region_id"])
        assert "zero total area" in str(exc.value)
        assert "C" in str(exc.value)


class TestUnknownPolicy:
    def test_compute_native_zero_total_raises(self):
        # area weight with a region whose area_raw is 0 → degenerate
        area = pd.DataFrame(
            [
                ("A", 0, 0, 0.5, 0.5, 100.0),
                ("B", 1, 1, 1.5, 1.5, 0.0),
            ],
            columns=[
                "region_id",
                "cell_ix",
                "cell_iy",
                "cell_lon",
                "cell_lat",
                "raw",
            ],
        )
        with pytest.raises(ValueError, match="B"):
            compute_native_weights(area, ["region_id"], "area", raw_col="raw")


class TestCountMethods:
    def test_method_counts_per_region(self):
        raw = _raw_frame_AB_only()
        area = _area_frame()
        out = apply_fallback(raw, area, "pop", "area", ["region_id"])
        counts = count_methods(out, "pop")
        # 2 native regions (A, B), 1 area_fallback (C)
        assert counts == {"native": 2, "area_fallback": 1}


def _allocation_area_frame() -> pd.DataFrame:
    """Polygon source units (hierid strings) split across two target regions.

    Source X overlaps targets P and Q with area 30/70; source Y lies
    entirely inside Q. The areawt column is already per-source normalized,
    as the polygon pipeline would produce it.
    """
    return pd.DataFrame(
        [
            ("P", "X", 0.3, 30.0),
            ("Q", "X", 0.7, 70.0),
            ("Q", "Y", 1.0, 40.0),
        ],
        columns=["gid_1", "hierid", "areawt", "area_raw"],
    )


def _ir_units():
    from cil_regionalization.schema import SourceUnits

    return SourceUnits.from_string_ids(["hierid"])


class TestPerSourceNormalization:
    """per_source: weights sum to 1 within each source unit, across the
    target regions it overlaps. The allocation direction for splitting an
    extensive quantity."""

    def test_native_sums_to_one_per_source_unit(self):
        per_piece = pd.DataFrame(
            [
                ("P", "X", 30.0),
                ("Q", "X", 70.0),
                ("Q", "Y", 40.0),
            ],
            columns=["gid_1", "hierid", "raw"],
        )
        out = compute_native_weights(
            per_piece,
            ["gid_1"],
            "area",
            raw_col="raw",
            source_units=_ir_units(),
            normalization="per_source",
        )
        sums = out.groupby("hierid")["areawt"].sum()
        assert sums.loc["X"] == pytest.approx(1.0)
        assert sums.loc["Y"] == pytest.approx(1.0)
        # per-target sums are NOT 1 in this direction: Q collects pieces
        # of both X and Y.
        q_sum = out.loc[out["gid_1"] == "Q", "areawt"].sum()
        assert q_sum == pytest.approx(0.7 + 1.0)

    def test_fallback_applies_per_source_unit(self):
        # X has population in both pieces; Y has zero everywhere and
        # falls back to its per-source area weights.
        raw = pd.DataFrame(
            [
                ("P", "X", 10.0),
                ("Q", "X", 90.0),
            ],
            columns=["gid_1", "hierid", "raw"],
        )
        out = apply_fallback(
            raw,
            _allocation_area_frame(),
            "pop",
            "area",
            ["gid_1"],
            source_units=_ir_units(),
            normalization="per_source",
        )
        sums = out.groupby("hierid")["popwt"].sum()
        assert sums.loc["X"] == pytest.approx(1.0)
        assert sums.loc["Y"] == pytest.approx(1.0)
        methods = out.set_index("hierid")["pop_method"]
        assert (methods.loc[["X"]] == "native").all()
        assert (methods.loc[["Y"]] == "area_fallback").all()

    def test_same_frame_other_direction_changes_grouping(self):
        # The identical geometry normalized per_destination sums to 1 per
        # target region instead; the two results are different numbers,
        # which is why the direction must travel with the file.
        per_piece = pd.DataFrame(
            [
                ("P", "X", 30.0),
                ("Q", "X", 70.0),
                ("Q", "Y", 40.0),
            ],
            columns=["gid_1", "hierid", "raw"],
        )
        out = compute_native_weights(
            per_piece,
            ["gid_1"],
            "area",
            raw_col="raw",
            source_units=_ir_units(),
            normalization="per_destination",
        )
        sums = out.groupby("gid_1")["areawt"].sum()
        assert sums.loc["P"] == pytest.approx(1.0)
        assert sums.loc["Q"] == pytest.approx(1.0)
        x_sum = out.loc[out["hierid"] == "X", "areawt"].sum()
        assert x_sum != pytest.approx(1.0)
