"""`regroup` takes a reduced frame to coarser targets, a sum or a mean
weighted by each target's basis, before the statistics are pooled."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.apply import WeightsArtifact
from cil_regionalization.regroup import regroup, target_basis
from cil_regionalization.schema import OutputSchema, SourceUnits
from cil_regionalization.stats import pooled_statistics

ID = ["ISO", "ID_1"]


def _reduced() -> pd.DataFrame:
    """Two states of A and one of B, two models of two batches, one
    window. The values name their row."""
    rows = []
    for (iso, state), base in {("A", 1): 10.0, ("A", 2): 20.0, ("B", 1): 40.0}.items():
        for b in ("b0", "b1"):
            for g, gcm in enumerate(["CCSM4", "MIROC5"]):
                rows.append(
                    {
                        "ISO": iso,
                        "ID_1": state,
                        "window": "2080-2099",
                        "batch": b,
                        "gcm": gcm,
                        "value": base + g + (0.5 if b == "b1" else 0.0),
                    }
                )
    return pd.DataFrame(rows)


BASIS = pd.DataFrame(
    {"ISO": ["A", "A", "B"], "ID_1": [1, 2, 1], "basis": [1.0, 3.0, 2.0]}
)


class TestRegroup:
    def test_an_extensive_value_is_the_sum_of_the_states(self):
        out = regroup(
            _reduced(),
            id_fields=ID,
            by=["ISO"],
            kind="extensive",
            value_col="value",
            summation="fast",
        )
        assert list(out.columns) == ["ISO", "window", "batch", "gcm", "value"]
        a = out[(out["ISO"] == "A") & (out["batch"] == "b0") & (out["gcm"] == "CCSM4")]
        assert a["value"].item() == 10.0 + 20.0
        assert len(out) == 2 * 2 * 2

    def test_an_intensive_value_is_the_mean_weighted_by_the_basis(self):
        out = regroup(
            _reduced(),
            id_fields=ID,
            by=["ISO"],
            kind="intensive",
            value_col="value",
            summation="exact",
            basis=BASIS,
        )
        a = out[(out["ISO"] == "A") & (out["batch"] == "b0") & (out["gcm"] == "CCSM4")]
        assert a["value"].item() == (1.0 * 10.0 + 3.0 * 20.0) / 4.0

    def test_a_state_with_no_value_is_left_out_of_the_mean(self):
        frame = _reduced()
        frame.loc[(frame["ID_1"] == 2) & (frame["ISO"] == "A"), "value"] = np.nan
        out = regroup(
            frame,
            id_fields=ID,
            by=["ISO"],
            kind="intensive",
            value_col="value",
            summation="fast",
            basis=BASIS,
        )
        a = out[(out["ISO"] == "A") & (out["batch"] == "b0") & (out["gcm"] == "CCSM4")]
        assert a["value"].item() == 10.0
        b_only = _reduced()
        b_only.loc[b_only["ISO"] == "B", "value"] = np.nan
        summed = regroup(
            b_only,
            id_fields=ID,
            by=["ISO"],
            kind="extensive",
            value_col="value",
            summation="fast",
        )
        assert summed.loc[summed["ISO"] == "B", "value"].isna().all()

    def test_the_exact_sum_does_not_depend_on_the_order_of_the_rows(self):
        frame = _reduced()
        frame["value"] = frame["value"] * 0.1 + 1e-17
        options = dict(
            id_fields=ID, by=["ISO"], kind="intensive", value_col="value", basis=BASIS
        )
        one = regroup(frame, summation="exact", **options)
        two = regroup(
            frame.iloc[::-1].reset_index(drop=True), summation="exact", **options
        )
        keys = ["ISO", "window", "batch", "gcm"]
        one, two = (f.sort_values(keys).reset_index(drop=True) for f in (one, two))
        assert one["value"].to_numpy().tobytes() == two["value"].to_numpy().tobytes()

    def test_a_target_without_a_basis_is_refused(self):
        with pytest.raises(ValueError, match="have a target with no basis"):
            regroup(
                _reduced(),
                id_fields=ID,
                by=["ISO"],
                kind="intensive",
                value_col="value",
                summation="fast",
                basis=BASIS.iloc[:2],
            )

    def test_what_does_not_fit_is_refused(self):
        options = dict(id_fields=ID, value_col="value", summation="fast")
        with pytest.raises(ValueError, match="has to name some of the id fields"):
            regroup(_reduced(), by=["window"], kind="extensive", **options)
        with pytest.raises(ValueError, match="takes none"):
            regroup(_reduced(), by=["ISO"], kind="extensive", basis=BASIS, **options)
        with pytest.raises(ValueError, match="weighted by each target's basis"):
            regroup(_reduced(), by=["ISO"], kind="intensive", **options)

    def test_regrouped_then_pooled_is_the_country_statistics(self):
        """The order the recipe scripts follow, the regroup before the
        statistics, since a quantile of a sum is not the sum of the
        quantiles."""
        country = regroup(
            _reduced(),
            id_fields=ID,
            by=["ISO"],
            kind="extensive",
            value_col="value",
            summation="fast",
        )
        stats = pooled_statistics(
            country,
            sample_dims=["batch", "gcm"],
            quantiles=[0.5],
            quantile_method="linear",
            summation="fast",
        )
        a = stats[(stats["ISO"] == "A") & (stats["statistic"] == "mean")]
        # each state's batch offset counts twice in A's sum, so A is 30,
        # 32, 31 and 33 over the two batches of two models
        assert a["value"].item() == 31.5


class TestTargetBasis:
    @staticmethod
    def _artifact(frame) -> WeightsArtifact:
        schema = OutputSchema(
            id_fields=("ISO", "ID_1"),
            weight_names=("croparea",),
            source_units=SourceUnits.from_string_ids(["hierid"]),
            normalization="per_destination",
        )
        return WeightsArtifact(
            frame=frame, schema=schema, regions_version="t", source_version="s"
        )

    def test_it_is_the_raw_column_summed_by_target(self):
        frame = pd.DataFrame(
            {
                "ISO": ["A", "A", "A", "B"],
                "ID_1": [1, 1, 2, 1],
                "hierid": ["r1", "r2", "r3", "r4"],
                "croparea_raw": [2.0, 3.0, np.nan, 4.0],
                "cropareawt": [0.4, 0.6, np.nan, 1.0],
            }
        )
        got = target_basis(self._artifact(frame), "croparea")
        assert got.to_dict("records") == [
            {"ISO": "A", "ID_1": 1, "basis": 5.0},
            {"ISO": "A", "ID_1": 2, "basis": 0.0},
            {"ISO": "B", "ID_1": 1, "basis": 4.0},
        ]

    def test_an_artifact_without_the_raw_column_is_refused(self):
        frame = pd.DataFrame(
            {"ISO": ["A"], "ID_1": [1], "hierid": ["r1"], "cropareawt": [1.0]}
        )
        with pytest.raises(ValueError, match="no column 'croparea_raw'"):
            target_basis(self._artifact(frame), "croparea")
