"""Mass balance and source coverage: the conservation proof.

The acceptance fixture reproduces the drop pattern observed in the
legacy reference IR-to-ADM pipeline during the consolidation survey: 51
source units dropped entirely at ADM1 ("defective" multi-ADM1 cases), 12
more dropped at ADM2, and 139 rows mapped but zeroed (blank case
descriptions). The fixture is built from those documented counts, not
from production outputs. A conserving aggregation must pass; the
reproduced drop pattern must fail, and the coverage check must name the
constructed units exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.config import SourceUnitPolicies
from cil_regionalization.schema import OutputSchema, SourceUnits
from cil_regionalization.validate import (
    check_mass_balance,
    check_source_coverage,
    enforce_source_policies,
)


N_UNITS = 24378
N_DROPPED_ADM1 = 51
N_DROPPED_ADM2 = 12
N_ZEROED = 139
N_COUNTRIES = 20


def _source_units_frame() -> pd.DataFrame:
    """One row per source unit with a value and a country.

    Values vary by unit so that reshuffling errors do not cancel by
    accident; countries cycle so every country holds ~1200 units.
    """
    i = np.arange(N_UNITS)
    return pd.DataFrame(
        {
            "hierid": [f"IR{k:05d}" for k in i],
            "country": [f"C{k % N_COUNTRIES:02d}" for k in i],
            "value": 1.0 + (i % 7) * 0.25,
        }
    )


def _weights(source: pd.DataFrame, *, drop: int = 0, zero: int = 0) -> pd.DataFrame:
    """A per_source allocation mapping each unit to one target per country.

    The first ``drop`` units get no row at all (the defective-case
    pattern); the next ``zero`` units keep their row with weight 0 (the
    blank-case pattern). Everything else maps with weight 1 to the
    target 'T_<country>'.
    """
    frame = pd.DataFrame(
        {
            "gid_1": "T_" + source["country"],
            "hierid": source["hierid"],
            "popwt": 1.0,
            "pop_raw": 1.0,
            "pop_method": "native",
        }
    )
    frame = frame.iloc[drop:].reset_index(drop=True)
    frame.loc[: zero - 1, "popwt"] = 0.0
    return frame


def _aggregate(source: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Extensive per_source application: target value = sum of w * value."""
    joined = weights.merge(source, on="hierid", how="left")
    joined["contrib"] = joined["popwt"] * joined["value"]
    out = joined.groupby("gid_1", as_index=False)["contrib"].sum()
    out = out.rename(columns={"contrib": "value"})
    out["country"] = out["gid_1"].str.removeprefix("T_")
    return out


def _schema() -> OutputSchema:
    return OutputSchema(
        id_fields=("gid_1",),
        weight_names=("pop",),
        source_units=SourceUnits.from_string_ids(["hierid"]),
        normalization="per_source",
    )


class TestMassBalance:
    def test_conserving_aggregation_passes(self):
        source = _source_units_frame()
        target = _aggregate(source, _weights(source))
        report = check_mass_balance(
            source, target, value_col="value", group_col="country"
        )
        assert report.ok, report.summary()
        assert report.source_total == pytest.approx(report.target_total)

    def test_documented_drop_pattern_detected(self):
        """The acceptance case: 51 + 12 dropped, 139 zeroed. The loss is
        ~0.8% of global mass, far beyond tolerance, and must be caught
        globally and in the affected countries."""
        source = _source_units_frame()
        broken = _weights(
            source, drop=N_DROPPED_ADM1 + N_DROPPED_ADM2, zero=N_ZEROED
        )
        target = _aggregate(source, broken)
        report = check_mass_balance(
            source, target, value_col="value", group_col="country", tolerance=1e-6
        )
        assert not report.ok, (
            "mass balance passed on a fixture that drops "
            f"{N_DROPPED_ADM1 + N_DROPPED_ADM2} and zeroes {N_ZEROED} units; "
            "the check is not working"
        )
        assert report.target_total < report.source_total
        # The 202 affected units are the first ones, which cycle through
        # every country label; each affected country must be flagged.
        affected = set(
            source["country"].iloc[: N_DROPPED_ADM1 + N_DROPPED_ADM2 + N_ZEROED]
        )
        assert set(report.failures["group"]) == affected
        assert "FAILED" in report.summary()

    def test_wholly_dropped_group_is_full_size_failure(self):
        source = _source_units_frame()
        weights = _weights(source)
        target = _aggregate(source, weights)
        target = target.loc[target["country"] != "C00"]
        report = check_mass_balance(
            source, target, value_col="value", group_col="country"
        )
        assert not report.ok
        row = report.failures.set_index("group").loc["C00"]
        assert row["target_sum"] == 0.0
        assert row["rel_error"] == pytest.approx(1.0)

    def test_global_only_without_group_col(self):
        source = _source_units_frame()
        target = _aggregate(source, _weights(source, drop=10))
        report = check_mass_balance(source, target, value_col="value")
        assert not report.ok
        assert len(report.failures) == 0  # no groups requested; global catch

    def test_both_empty_pass(self):
        empty = pd.DataFrame({"value": pd.Series(dtype="float64")})
        report = check_mass_balance(empty, empty.copy(), value_col="value")
        assert report.ok
        assert report.rel_error == 0.0

    def test_missing_value_column_raises(self):
        source = _source_units_frame()
        with pytest.raises(ValueError, match="missing value column"):
            check_mass_balance(
                source, source.drop(columns=["value"]), value_col="value"
            )


class TestSourceCoverage:
    def test_constructed_cases_identified_exactly(self):
        source = _source_units_frame()
        drop = N_DROPPED_ADM1 + N_DROPPED_ADM2
        weights = _weights(source, drop=drop, zero=N_ZEROED)
        # Also remove some data units so absent_from_data is exercised:
        # the last 5 units have weights but no data.
        data_ids = [(h,) for h in source["hierid"].iloc[: N_UNITS - 5]]
        coverage = check_source_coverage(weights, _schema(), "pop", data_ids)

        expected_unmatched = {(h,) for h in source["hierid"].iloc[:drop]}
        expected_zero = {(h,) for h in source["hierid"].iloc[drop : drop + N_ZEROED]}
        expected_absent = {(h,) for h in source["hierid"].iloc[N_UNITS - 5 :]}
        assert set(coverage.unmatched) == expected_unmatched
        assert set(coverage.zero_weight) == expected_zero
        assert set(coverage.absent_from_data) == expected_absent
        assert coverage.counts == {
            "unmatched": drop,
            "zero_weight": N_ZEROED,
            "absent_from_data": 5,
        }

    def test_complete_coverage_is_clean(self):
        source = _source_units_frame()
        weights = _weights(source)
        data_ids = [(h,) for h in source["hierid"]]
        coverage = check_source_coverage(weights, _schema(), "pop", data_ids)
        assert coverage.clean
        assert coverage.counts == {
            "unmatched": 0,
            "zero_weight": 0,
            "absent_from_data": 0,
        }

    def test_all_nan_unit_counts_as_zero_weight(self):
        source = _source_units_frame().head(3)
        weights = _weights(source)
        weights.loc[weights["hierid"] == "IR00001", "popwt"] = float("nan")
        data_ids = [(h,) for h in source["hierid"]]
        coverage = check_source_coverage(weights, _schema(), "pop", data_ids)
        assert coverage.zero_weight == (("IR00001",),)


class TestEnforceSourcePolicies:
    def _coverage(self):
        source = _source_units_frame().head(100)
        weights = _weights(source, drop=3, zero=2)
        data_ids = [(h,) for h in source["hierid"]]
        return check_source_coverage(weights, _schema(), "pop", data_ids)

    def test_default_policies_raise_and_name_config_key(self):
        with pytest.raises(ValueError, match="3 unmatched") as exc:
            enforce_source_policies(self._coverage(), SourceUnitPolicies())
        assert "on_unmatched='skip'" in str(exc.value)
        assert "2 zero_weight" in str(exc.value)

    def test_skip_records_in_manifest_and_continues(self, caplog):
        from cil_regionalization.manifest import Manifest

        manifest = Manifest(
            config_hash="x",
            backend="local",
            coverage="exact_fraction",
            lon_convention="[-180,180)",
            grid_mode="generate",
            grid_resolution=1.0,
            package_versions={},
            python_version="3",
        )
        policies = SourceUnitPolicies(
            on_unmatched="skip", on_zero_weight="skip", on_absent_from_data="skip"
        )
        with caplog.at_level("WARNING"):
            enforce_source_policies(self._coverage(), policies, manifest)
        assert manifest.source_coverage["unmatched"]["count"] == 3
        assert manifest.source_coverage["zero_weight"]["count"] == 2
        assert manifest.source_coverage["absent_from_data"]["count"] == 0
        assert ["IR00000"] in manifest.source_coverage["unmatched"]["ids"]
        assert any("unmatched" in r.message for r in caplog.records)

    def test_manifest_recorded_even_when_erroring(self):
        from cil_regionalization.manifest import Manifest

        manifest = Manifest(
            config_hash="x",
            backend="local",
            coverage="exact_fraction",
            lon_convention="[-180,180)",
            grid_mode="generate",
            grid_resolution=1.0,
            package_versions={},
            python_version="3",
        )
        with pytest.raises(ValueError):
            enforce_source_policies(
                self._coverage(), SourceUnitPolicies(), manifest
            )
        assert manifest.source_coverage["unmatched"]["count"] == 3

    def test_clean_coverage_passes_under_error_policies(self):
        source = _source_units_frame().head(10)
        weights = _weights(source)
        data_ids = [(h,) for h in source["hierid"]]
        coverage = check_source_coverage(weights, _schema(), "pop", data_ids)
        enforce_source_policies(coverage, SourceUnitPolicies())  # no raise
