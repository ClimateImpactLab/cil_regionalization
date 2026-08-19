"""Delta-method gradients: aggregation, quadratic form, and the reader.

The synthetic cases are small enough to compute by hand. The point the
module exists for is pinned directly: the variance of a sum through
the quadratic form differs from the sum of variances whenever the
covariance matrix has off-diagonal mass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.apply import WeightsArtifact
from cil_regionalization.gradients import (
    aggregate_gradient,
    delta_method,
    read_gradient_leaf,
)
from cil_regionalization.schema import OutputSchema, SourceUnits


def _artifact(rows: list[dict]) -> WeightsArtifact:
    frame = pd.DataFrame(rows)
    schema = OutputSchema(
        id_fields=("ISO", "ID_1"),
        weight_names=("pop",),
        source_units=SourceUnits.from_string_ids(["hierid"]),
        normalization="per_source",
    )
    return WeightsArtifact(
        frame=frame,
        schema=schema,
        regions_version="test-targets",
        source_version="test-regions",
        partial_coverage={},
    )


def _two_region_setup():
    # two regions, two targets: region a split 0.75/0.25, region b whole
    art = _artifact(
        [
            {"ISO": "T", "ID_1": 1, "hierid": "a", "popwt": 0.75},
            {"ISO": "T", "ID_1": 2, "hierid": "a", "popwt": 0.25},
            {"ISO": "T", "ID_1": 2, "hierid": "b", "popwt": 1.0},
        ]
    )
    regions = ["a", "b"]
    gradient = np.array([[1.0, 3.0], [2.0, 4.0]])  # coeff x region
    return art, regions, gradient


class TestAggregateGradient:
    def test_extensive_shares_split_the_gradient(self):
        art, regions, gradient = _two_region_setup()
        targets, agg = aggregate_gradient(art, regions, gradient)
        got = {tuple(t): agg[i] for i, t in enumerate(targets.itertuples(index=False))}
        assert np.allclose(got[("T", 1)], [0.75 * 1.0, 0.75 * 2.0])
        assert np.allclose(got[("T", 2)], [0.25 * 1.0 + 3.0, 0.25 * 2.0 + 4.0])

    def test_targets_sum_to_the_total(self):
        art, regions, gradient = _two_region_setup()
        _, agg = aggregate_gradient(art, regions, gradient)
        assert np.allclose(agg.sum(axis=0), gradient.sum(axis=1))

    def test_rate_normalization_is_weighted_mean(self):
        art, regions, gradient = _two_region_setup()
        pop = np.array([100.0, 300.0])
        targets, agg = aggregate_gradient(
            art, regions, gradient, scale=pop, normalize=True
        )
        got = {tuple(t): agg[i] for i, t in enumerate(targets.itertuples(index=False))}
        # target 2: weights a=0.25*100=25, b=1.0*300=300
        expect = (25 * gradient[:, 0] + 300 * gradient[:, 1]) / 325
        assert np.allclose(got[("T", 2)], expect)
        # target 1 has only region a, so the mean is region a's gradient
        assert np.allclose(got[("T", 1)], gradient[:, 0])

    def test_region_without_weight_rows_raises(self):
        art, regions, gradient = _two_region_setup()
        with pytest.raises(ValueError, match="no weight rows"):
            aggregate_gradient(art, regions + ["c"],
                               np.hstack([gradient, [[5.0], [6.0]]]))

    def test_wrong_scale_shape_raises(self):
        art, regions, gradient = _two_region_setup()
        with pytest.raises(ValueError, match="scale has shape"):
            aggregate_gradient(art, regions, gradient, scale=np.ones(3),
                               normalize=True)


class TestDeltaMethod:
    def test_variance_is_quadratic_form(self):
        G = np.array([[1.0, 2.0], [3.0, 0.0]])
        V = np.array([[2.0, 1.0], [1.0, 4.0]])
        out = delta_method(G, V)
        # g=(1,2): 1*2*1 + 2*1*2*1 + 4*4 = 22
        assert out["sd"].iloc[0] == pytest.approx(np.sqrt(22.0))
        assert out["sd"].iloc[1] == pytest.approx(np.sqrt(9.0 * 2.0))

    def test_mean_is_gradient_times_coefficients(self):
        G = np.array([[1.0, 2.0]])
        out = delta_method(G, np.eye(2), coefficients=np.array([10.0, -1.0]))
        assert out["mean"].iloc[0] == pytest.approx(8.0)

    def test_correlation_changes_the_aggregate_spread(self):
        # two regions with identical gradients: the correct aggregate sd
        # is twice the per-region sd, not sqrt(2) times it
        art = _artifact(
            [
                {"ISO": "T", "ID_1": 1, "hierid": "a", "popwt": 1.0},
                {"ISO": "T", "ID_1": 1, "hierid": "b", "popwt": 1.0},
            ]
        )
        g = np.array([[1.0, 1.0]])
        V = np.array([[1.0]])
        _, agg = aggregate_gradient(art, ["a", "b"], g)
        correct = float(delta_method(agg, V)["sd"].iloc[0])
        per_region_var = np.einsum("cr,cd,dr->r", g, V, g)
        naive = float(np.sqrt(per_region_var.sum()))
        assert correct == pytest.approx(2.0)
        assert naive == pytest.approx(np.sqrt(2.0))
        assert correct > naive

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="covariance matrix"):
            delta_method(np.ones((1, 3)), np.eye(2))
        with pytest.raises(ValueError, match="coefficient vector"):
            delta_method(np.ones((1, 2)), np.eye(2), coefficients=np.ones(3))


class TestReader:
    def test_roundtrip_with_duplicate_vcv_dimension(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        p = tmp_path / "leaf.nc4"
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("coefficient", 2)
            ds.createDimension("year", 3)
            ds.createDimension("region", 2)
            regions = ds.createVariable("regions", str, ("region",))
            regions[0], regions[1] = "r1", "r2"
            year = ds.createVariable("year", "i4", ("year",))
            year[:] = [2000, 2001, 2002]
            bcde = ds.createVariable(
                "rebased_bcde", "f4", ("coefficient", "year", "region")
            )
            bcde[:] = np.arange(12).reshape(2, 3, 2)
            vcv = ds.createVariable("vcv", "f8", ("coefficient", "coefficient"))
            vcv[:] = np.array([[1.0, 0.5], [0.5, 2.0]])
            reb = ds.createVariable("rebased", "f4", ("year", "region"))
            reb[:] = np.arange(6).reshape(3, 2)
        leaf = read_gradient_leaf(p, year=2001)
        assert leaf.regions == ["r1", "r2"]
        assert leaf.gradient.shape == (2, 2)
        assert np.allclose(leaf.gradient, [[2, 3], [8, 9]])
        assert np.allclose(leaf.vcv, [[1.0, 0.5], [0.5, 2.0]])
        assert np.allclose(leaf.stored_variance, [2, 3])

    def test_missing_year_raises(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        p = tmp_path / "leaf.nc4"
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("coefficient", 1)
            ds.createDimension("year", 1)
            ds.createDimension("region", 1)
            regions = ds.createVariable("regions", str, ("region",))
            regions[0] = "r1"
            ds.createVariable("year", "i4", ("year",))[:] = [2000]
            ds.createVariable("rebased_bcde", "f4",
                              ("coefficient", "year", "region"))[:] = [[[1.0]]]
            ds.createVariable("vcv", "f8",
                              ("coefficient", "coefficient"))[:] = [[1.0]]
        with pytest.raises(ValueError, match="no year"):
            read_gradient_leaf(p, year=1999)
