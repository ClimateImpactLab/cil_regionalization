"""Application layer acceptance tests.

Artifacts are built in memory from hand-written frames so every expected
number can be derived on paper; one round-trip test goes through the
polygon backend, `write_result`, and `WeightsArtifact.load` to prove the
on-disk path carries everything application needs.

The nested fixture: two countries, three level-2 targets, six source
units, one of which (s3) straddles two level-2 targets inside country C1.

    level 1 (adm0):        C1                    C2
    level 2 (adm0, code):  (C1, 001) (C1, 002)   (C2, 001)
    sources:  s1 -> (C1, 001) 1.0     s4 -> (C2, 001) 1.0
              s2 -> (C1, 001) 1.0     s5 -> (C2, 001) 1.0
              s3 -> (C1, 001) 0.25,   s6 -> (C2, 001) 1.0
                    (C1, 002) 0.75

Level-2 code "001" repeats across countries on purpose: with a display
name attached to it, this is exactly the shared-name pattern (the same
ADM1 name in two countries) that name-based joins merge silently.
"""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from cil_regionalization.apply import WeightsArtifact, apply_weights
from cil_regionalization.config import SourceUnitPolicies
from cil_regionalization.schema import OutputSchema, SourceUnits

SOURCE_VERSION = "units-v1"


def _artifact(
    frame: pd.DataFrame,
    *,
    id_fields: list[str],
    normalization: str,
    weight_names: list[str] = ["pop"],
    source_keys: list[str] = ["unit_id"],
    source_version: str | None = SOURCE_VERSION,
) -> WeightsArtifact:
    schema = OutputSchema(
        id_fields=tuple(id_fields),
        weight_names=tuple(weight_names),
        source_units=SourceUnits.from_string_ids(source_keys),
        normalization=normalization,
    )
    return WeightsArtifact(
        frame=frame,
        schema=schema,
        regions_version="targets-v1",
        source_version=source_version,
    )


def _level1_artifact() -> WeightsArtifact:
    frame = pd.DataFrame(
        {
            "adm0": ["C1", "C1", "C1", "C2", "C2", "C2"],
            "unit_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "popwt": [1.0] * 6,
        }
    )
    return _artifact(frame, id_fields=["adm0"], normalization="per_source")


def _level2_artifact() -> WeightsArtifact:
    frame = pd.DataFrame(
        {
            "adm0": ["C1", "C1", "C1", "C1", "C2", "C2", "C2"],
            "code": ["001", "001", "001", "002", "001", "001", "001"],
            "unit_id": ["s1", "s2", "s3", "s3", "s4", "s5", "s6"],
            "popwt": [1.0, 1.0, 0.25, 0.75, 1.0, 1.0, 1.0],
        }
    )
    return _artifact(
        frame, id_fields=["adm0", "code"], normalization="per_source"
    )


def _mc_data() -> pd.DataFrame:
    """Six units by two scenarios by two years, deterministic values.

    Column names are deliberately unlike the production dims: nothing in
    the application layer may depend on names like gcm or rcp.
    """
    units = ["s1", "s2", "s3", "s4", "s5", "s6"]
    rows = []
    for i, (unit, scn, year) in enumerate(
        itertools.product(units, ["scn_a", "scn_b"], [2050, 2051])
    ):
        rows.append(
            {
                "unit_id": unit,
                "banana_dim": scn,
                "t": year,
                "country": "C1" if unit in ("s1", "s2", "s3") else "C2",
                "value": 10.0 + 3.0 * i,
            }
        )
    return pd.DataFrame(rows)


class TestExtensiveNestedLevels:
    def test_mass_balance_global_and_per_group_both_levels(self):
        data = _mc_data()
        for artifact in (_level1_artifact(), _level2_artifact()):
            result = apply_weights(
                artifact,
                data,
                kind="extensive",
                weight="pop",
                data_version=SOURCE_VERSION,
                group_col="country",
            )
            assert result.mass_balance is not None
            assert result.mass_balance.ok
            assert result.coverage.clean

    def test_levels_are_mutually_consistent(self):
        data = _mc_data()
        r1 = apply_weights(
            _level1_artifact(),
            data,
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        r2 = apply_weights(
            _level2_artifact(),
            data,
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        per_dim_1 = r1.frame.groupby(["adm0", "banana_dim", "t"])["value"].sum()
        per_dim_2 = r2.frame.groupby(["adm0", "banana_dim", "t"])["value"].sum()
        pd.testing.assert_series_equal(per_dim_1, per_dim_2)

    def test_dimensions_are_passthrough(self):
        result = apply_weights(
            _level2_artifact(),
            _mc_data(),
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        assert set(result.frame.columns) == {
            "adm0",
            "code",
            "banana_dim",
            "t",
            "country",
            "value",
        }
        assert set(result.frame["banana_dim"]) == {"scn_a", "scn_b"}
        assert set(result.frame["t"]) == {2050, 2051}

    def test_straddler_allocates_by_weight(self):
        data = _mc_data().query("banana_dim == 'scn_a' and t == 2050")
        result = apply_weights(
            _level2_artifact(),
            data,
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        s3_value = float(data.loc[data["unit_id"] == "s3", "value"].iloc[0])
        frame = result.frame.set_index(["adm0", "code"])
        # (C1, 002) contains only s3's 0.75 share.
        assert frame.loc[("C1", "002"), "value"] == pytest.approx(0.75 * s3_value)


class TestQualifiedKeys:
    """The shared-name case: level-2 code '001' exists in both countries
    and carries the same display name. Qualified keys keep the two units
    apart; a name join would have merged them."""

    _NAMES = pd.DataFrame(
        {
            "adm0": ["C1", "C1", "C2"],
            "code": ["001", "002", "001"],
            "display_name": ["La Rioja", "Other", "La Rioja"],
        }
    )

    def test_qualified_keys_keep_shared_names_apart(self):
        data = _mc_data().query("banana_dim == 'scn_a' and t == 2050")
        result = apply_weights(
            _level2_artifact(),
            data.drop(columns=["country"]),
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        named = result.frame.merge(self._NAMES, on=["adm0", "code"])
        la_rioja = named.loc[named["display_name"] == "La Rioja"]
        # Two distinct target units survive under the qualified key.
        assert len(la_rioja) == 2
        assert set(la_rioja["adm0"]) == {"C1", "C2"}

        # The name-based pipeline this replaces grouped by display name;
        # doing that here merges two countries' units into one row whose
        # value equals neither, which is the documented silent collision.
        name_joined = named.groupby("display_name")["value"].sum()
        merged_value = name_joined.loc["La Rioja"]
        for correct in la_rioja["value"]:
            assert merged_value != pytest.approx(correct)
        assert merged_value == pytest.approx(la_rioja["value"].sum())

    def test_api_has_no_name_join_parameter(self):
        import inspect

        from cil_regionalization import apply as apply_module

        params = inspect.signature(apply_module.apply_weights).parameters
        assert "name" not in params
        assert "display_name" not in params
        # Join keys come from the artifact schema alone.


class TestKindDirectionGuard:
    def test_extensive_on_per_destination_raises(self):
        frame = _level1_artifact().frame
        artifact = _artifact(frame, id_fields=["adm0"], normalization="per_destination")
        with pytest.raises(ValueError, match="normalization mismatch"):
            apply_weights(
                artifact,
                _mc_data(),
                kind="extensive",
                weight="pop",
                data_version=SOURCE_VERSION,
            )

    def test_ratio_on_per_destination_raises(self):
        frame = _level1_artifact().frame
        artifact = _artifact(frame, id_fields=["adm0"], normalization="per_destination")
        with pytest.raises(ValueError, match="normalization mismatch"):
            apply_weights(
                artifact,
                _mc_data().assign(den=1.0),
                kind="ratio",
                weight="pop",
                denominator_col="den",
                data_version=SOURCE_VERSION,
            )

    def test_intensive_on_per_source_raises(self):
        with pytest.raises(ValueError, match="normalization mismatch"):
            apply_weights(
                _level1_artifact(),
                _mc_data(),
                kind="intensive",
                weight="pop",
                data_version=SOURCE_VERSION,
            )


class TestVersionAgreement:
    def test_version_mismatch_raises(self):
        with pytest.raises(ValueError, match="version mismatch"):
            apply_weights(
                _level1_artifact(),
                _mc_data(),
                kind="extensive",
                weight="pop",
                data_version="units-v2",
            )

    def test_missing_data_version_raises(self):
        with pytest.raises(ValueError, match="no geometry version"):
            apply_weights(
                _level1_artifact(),
                _mc_data(),
                kind="extensive",
                weight="pop",
                data_version=None,
            )

    def test_artifact_without_source_version_raises(self):
        frame = _level1_artifact().frame
        artifact = _artifact(
            frame,
            id_fields=["adm0"],
            normalization="per_source",
            source_version=None,
        )
        with pytest.raises(ValueError, match="records no source geometry version"):
            apply_weights(
                artifact,
                _mc_data(),
                kind="extensive",
                weight="pop",
                data_version=SOURCE_VERSION,
            )


class TestIntensive:
    def test_weighted_mean_differs_from_naive_mean(self):
        frame = pd.DataFrame(
            {
                "adm0": ["T", "T"],
                "unit_id": ["x", "y"],
                "popwt": [0.9, 0.1],
            }
        )
        artifact = _artifact(frame, id_fields=["adm0"], normalization="per_destination")
        data = pd.DataFrame({"unit_id": ["x", "y"], "value": [10.0, 20.0]})
        result = apply_weights(
            artifact,
            data,
            kind="intensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        weighted = float(result.frame["value"].iloc[0])
        naive = data["value"].mean()
        assert weighted == pytest.approx(0.9 * 10.0 + 0.1 * 20.0)  # 11.0
        assert weighted != pytest.approx(naive)  # naive is 15.0
        assert result.mass_balance is None  # not a conserved quantity

    def test_partial_coverage_renormalizes(self):
        frame = pd.DataFrame(
            {
                "adm0": ["T", "T"],
                "unit_id": ["x", "y"],
                "popwt": [0.9, 0.1],
            }
        )
        artifact = _artifact(frame, id_fields=["adm0"], normalization="per_destination")
        data = pd.DataFrame(
            {"unit_id": ["x", "y"], "value": [10.0, float("nan")]}
        )
        result = apply_weights(
            artifact,
            data,
            kind="intensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        # y has no value: the mean is over x alone, not deflated by 0.9.
        assert float(result.frame["value"].iloc[0]) == pytest.approx(10.0)


class TestRatio:
    def test_aggregate_then_divide_not_average_of_ratios(self):
        frame = pd.DataFrame(
            {
                "adm0": ["T", "T"],
                "unit_id": ["r1", "r2"],
                "popwt": [1.0, 1.0],
            }
        )
        artifact = _artifact(frame, id_fields=["adm0"], normalization="per_source")
        data = pd.DataFrame(
            {
                "unit_id": ["r1", "r2"],
                "damages": [10.0, 30.0],
                "gdp": [100.0, 50.0],
            }
        )
        result = apply_weights(
            artifact,
            data,
            kind="ratio",
            weight="pop",
            value_col="damages",
            denominator_col="gdp",
            data_version=SOURCE_VERSION,
        )
        share = float(result.frame["damages"].iloc[0])
        # Correct: (10 + 30) / (100 + 50) = 0.2667.
        assert share == pytest.approx(40.0 / 150.0)
        # Wrong number the per-unit average would give: (0.1 + 0.6) / 2.
        assert share != pytest.approx((10.0 / 100.0 + 30.0 / 50.0) / 2.0)
        # Conservation was checked on numerator and denominator.
        assert result.mass_balance is not None and result.mass_balance.ok
        assert (
            result.mass_balance_denominator is not None
            and result.mass_balance_denominator.ok
        )
        # The denominator column does not leak into the output.
        assert "gdp" not in result.frame.columns


class TestCoveragePolicies:
    def test_unmatched_unit_raises_by_default(self):
        data = pd.concat(
            [
                _mc_data(),
                pd.DataFrame(
                    [
                        {
                            "unit_id": "s_ghost",
                            "banana_dim": "scn_a",
                            "t": 2050,
                            "country": "C1",
                            "value": 999.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="unmatched"):
            apply_weights(
                _level1_artifact(),
                data,
                kind="extensive",
                weight="pop",
                data_version=SOURCE_VERSION,
            )

    def test_skip_policy_reports_and_conserves_applied_mass(self):
        ghost = {
            "unit_id": "s_ghost",
            "banana_dim": "scn_a",
            "t": 2050,
            "country": "C1",
            "value": 999.0,
        }
        data = pd.concat(
            [_mc_data(), pd.DataFrame([ghost])], ignore_index=True
        )
        result = apply_weights(
            _level1_artifact(),
            data,
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
            policies=SourceUnitPolicies(on_unmatched="skip"),
            group_col="country",
        )
        assert result.coverage.unmatched == (("s_ghost",),)
        # Mass balance holds over the applied data; the excluded unit is
        # accounted for in the coverage report, not silently vanished.
        assert result.mass_balance is not None and result.mass_balance.ok

    def test_subset_run_uses_restriction_not_relaxed_policies(self):
        # Data for country C1's units only; the weight file is global.
        data = _mc_data().query("country == 'C1'")
        c1_units = [("s1",), ("s2",), ("s3",)]
        # Without restriction, the C2 units are absent_from_data: error.
        with pytest.raises(ValueError, match="absent_from_data"):
            apply_weights(
                _level1_artifact(),
                data,
                kind="extensive",
                weight="pop",
                data_version=SOURCE_VERSION,
            )
        # The subset mechanism is an explicit universe, not a relaxation.
        result = apply_weights(
            _level1_artifact(),
            data,
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
            restrict_to_sources=c1_units,
        )
        assert result.coverage.clean
        assert set(result.frame["adm0"]) == {"C1"}

    def test_duplicated_data_rows_raise(self):
        data = _mc_data()
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="double count"):
            apply_weights(
                _level1_artifact(),
                data,
                kind="extensive",
                weight="pop",
                data_version=SOURCE_VERSION,
            )


class TestArtifactRoundTrip:
    """Polygon backend to disk to load to apply: the manifest carries the
    column roles, direction, and versions the application needs."""

    def test_written_artifact_applies(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import box

        from cil_regionalization.backends.local import LocalBackend
        from cil_regionalization.config import Config
        from cil_regionalization.io import write_result
        from cil_regionalization.regions import RegionSet
        from cil_regionalization.weights import from_config_list

        targets = gpd.GeoDataFrame(
            {
                "target_id": ["T1", "T2"],
                "geometry": [box(0, 0, 4, 4), box(4, 0, 8, 4)],
            },
            crs="EPSG:4326",
        )
        units = gpd.GeoDataFrame(
            {
                "unit_id": ["u1", "u2"],
                # u1 nests in T1; u2 straddles T1/T2 half and half.
                "geometry": [box(1, 1, 2, 2), box(3.5, 2.5, 4.5, 3.5)],
            },
            crs="EPSG:4326",
        )
        tp, sp = tmp_path / "targets.parquet", tmp_path / "units.parquet"
        targets.to_parquet(tp)
        units.to_parquet(sp)

        cfg = Config.model_validate(
            {
                "project": {"name": "roundtrip"},
                "regions": {
                    "path": str(tp),
                    "id_fields": ["target_id"],
                    "version": "targets-v1",
                },
                "source": {
                    "path": str(sp),
                    "id_fields": ["unit_id"],
                    "version": SOURCE_VERSION,
                },
                "weights": [{"name": "area"}],
                "backend": {"kind": "local"},
                "output": {"dir": str(tmp_path / "out")},
                "normalization": "per_source",
            }
        )
        result = LocalBackend().compute(
            RegionSet.from_config(cfg.regions),
            None,
            from_config_list(cfg.weights),
            cfg,
        )
        write_result(result, cfg.output.dir, "parquet")

        artifact = WeightsArtifact.load(cfg.output.dir)
        assert artifact.normalization == "per_source"
        assert artifact.source_version == SOURCE_VERSION

        data = pd.DataFrame({"unit_id": ["u1", "u2"], "value": [100.0, 40.0]})
        applied = apply_weights(
            artifact,
            data,
            kind="extensive",
            weight="area",
            data_version=SOURCE_VERSION,
        )
        assert applied.mass_balance is not None and applied.mass_balance.ok
        by_target = applied.frame.set_index("target_id")["value"]
        # u2 splits half and half at lon 4; u1's 100 lands wholly in T1.
        assert by_target.loc["T1"] == pytest.approx(100.0 + 20.0, rel=1e-6)
        assert by_target.loc["T2"] == pytest.approx(20.0, rel=1e-6)


class TestPartialCoverageGuard:
    """An artifact recording partially covered source units cannot be
    applied as if coverage were complete; the acknowledgment is explicit."""

    def _partial_artifact(self) -> WeightsArtifact:
        artifact = _level1_artifact()
        return WeightsArtifact(
            frame=artifact.frame,
            schema=artifact.schema,
            regions_version=artifact.regions_version,
            source_version=artifact.source_version,
            partial_coverage={
                "threshold": 0.999,
                "count": 1,
                "target_subset": True,
                "units": [{"unit_id": "s1", "coverage_ratio": 0.02}],
                "units_truncated": False,
            },
        )

    def test_partial_artifact_refused_by_default(self):
        with pytest.raises(ValueError, match="partially covered"):
            apply_weights(
                self._partial_artifact(),
                _mc_data(),
                kind="extensive",
                weight="pop",
                data_version=SOURCE_VERSION,
            )

    def test_explicit_acknowledgment_allows(self):
        result = apply_weights(
            self._partial_artifact(),
            _mc_data(),
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
            allow_partial_coverage=True,
        )
        assert result.mass_balance is not None and result.mass_balance.ok

    def test_zero_count_accounting_passes(self):
        artifact = _level1_artifact()
        clean = WeightsArtifact(
            frame=artifact.frame,
            schema=artifact.schema,
            regions_version=artifact.regions_version,
            source_version=artifact.source_version,
            partial_coverage={
                "threshold": 0.999,
                "count": 0,
                "target_subset": False,
                "units": [],
                "units_truncated": False,
            },
        )
        result = apply_weights(
            clean,
            _mc_data(),
            kind="extensive",
            weight="pop",
            data_version=SOURCE_VERSION,
        )
        assert result.coverage.clean
