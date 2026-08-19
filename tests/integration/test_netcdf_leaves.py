"""NetCDF leaf reading: round trip, declared names, and the closed paths.

Everything except the missing-dependency test needs the [netcdf] extra;
those tests skip cleanly when xarray is absent so the base suite stays
runnable without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.apply import WeightsArtifact, apply_weights
from cil_regionalization.schema import OutputSchema, SourceUnits

SOURCE_VERSION = "units-v1"


def _artifact() -> WeightsArtifact:
    frame = pd.DataFrame(
        {
            "target_id": ["T1", "T1", "T2"],
            "unit_id": ["u1", "u2", "u2"],
            "areawt": [1.0, 0.5, 0.5],
        }
    )
    schema = OutputSchema(
        id_fields=("target_id",),
        weight_names=("area",),
        source_units=SourceUnits.from_string_ids(["unit_id"]),
        normalization="per_source",
    )
    return WeightsArtifact(
        frame=frame,
        schema=schema,
        regions_version="targets-v1",
        source_version=SOURCE_VERSION,
    )


def _leaf_frame() -> pd.DataFrame:
    rows = []
    for unit, base in (("u1", 100.0), ("u2", 40.0)):
        for year in (2050, 2051):
            rows.append(
                {"unit_id": unit, "year": year, "value": base + (year - 2050)}
            )
    return pd.DataFrame(rows)


def _write_netcdf(
    path: Path,
    frame: pd.DataFrame,
    *,
    region_dim: str,
    value_var: str = "value",
    region_order: list[str] | None = None,
) -> None:
    xr = pytest.importorskip("xarray")
    wide = frame.pivot(index=region_dim, columns="year", values=value_var)
    if region_order is not None:
        wide = wide.loc[region_order]
    ds = xr.Dataset(
        {value_var: ([region_dim, "year"], wide.to_numpy())},
        coords={region_dim: wide.index.to_list(), "year": wide.columns.to_list()},
    )
    ds.to_netcdf(path)


class TestReaderRoundTrip:
    def test_netcdf_equals_parquet_by_value(self, tmp_path):
        pytest.importorskip("xarray")
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        frame = _leaf_frame()
        nc = tmp_path / "leaf.nc4"
        _write_netcdf(
            nc, frame.rename(columns={"unit_id": "spatial_thing"}),
            region_dim="spatial_thing",
        )
        from_nc = read_netcdf_leaf(
            nc,
            variables=["value"],
            region_dim="spatial_thing",
            region_col="unit_id",
        )

        artifact = _artifact()
        kwargs = dict(
            kind="extensive",
            weight="area",
            data_version=SOURCE_VERSION,
        )
        via_nc = apply_weights(artifact, from_nc, **kwargs)
        via_pq = apply_weights(artifact, frame, **kwargs)
        key = ["target_id", "year"]
        pd.testing.assert_frame_equal(
            via_nc.frame.sort_values(key).reset_index(drop=True)[key + ["value"]],
            via_pq.frame.sort_values(key).reset_index(drop=True)[key + ["value"]],
        )

    def test_region_dim_name_is_configurable(self, tmp_path):
        pytest.importorskip("xarray")
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        nc = tmp_path / "leaf.nc4"
        _write_netcdf(
            nc,
            _leaf_frame().rename(columns={"unit_id": "hex_cell"}),
            region_dim="hex_cell",
        )
        out = read_netcdf_leaf(
            nc, variables=["value"], region_dim="hex_cell", region_col="unit_id"
        )
        assert "unit_id" in out.columns
        assert "hex_cell" not in out.columns
        assert set(out["unit_id"]) == {"u1", "u2"}

    def test_region_order_does_not_matter(self, tmp_path):
        """The file's region coordinate is stored in the reverse of the
        artifact's order; no reordering or reindexing happens in the
        reader, and the keyed join makes the result identical anyway."""
        pytest.importorskip("xarray")
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        frame = _leaf_frame()
        forward = tmp_path / "forward.nc4"
        reverse = tmp_path / "reverse.nc4"
        _write_netcdf(forward, frame, region_dim="unit_id", region_order=["u1", "u2"])
        _write_netcdf(reverse, frame, region_dim="unit_id", region_order=["u2", "u1"])

        artifact = _artifact()
        results = []
        for nc in (forward, reverse):
            data = read_netcdf_leaf(
                nc, variables=["value"], region_dim="unit_id", region_col="unit_id"
            )
            applied = apply_weights(
                artifact,
                data,
                kind="extensive",
                weight="area",
                data_version=SOURCE_VERSION,
            )
            results.append(
                applied.frame.sort_values(["target_id", "year"]).reset_index(drop=True)
            )
        pd.testing.assert_frame_equal(results[0], results[1])

    def test_nan_values_pass_through_unfilled(self, tmp_path):
        pytest.importorskip("xarray")
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        frame = _leaf_frame()
        frame.loc[
            (frame["unit_id"] == "u2") & (frame["year"] == 2051), "value"
        ] = np.nan
        nc = tmp_path / "leaf.nc4"
        _write_netcdf(nc, frame, region_dim="unit_id")
        out = read_netcdf_leaf(
            nc, variables=["value"], region_dim="unit_id", region_col="unit_id"
        )
        gap = out.loc[(out["unit_id"] == "u2") & (out["year"] == 2051), "value"]
        assert gap.isna().all()

    def test_unknown_variable_names_available(self, tmp_path):
        pytest.importorskip("xarray")
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        nc = tmp_path / "leaf.nc4"
        _write_netcdf(nc, _leaf_frame(), region_dim="unit_id")
        with pytest.raises(ValueError, match="available"):
            read_netcdf_leaf(
                nc,
                variables=["damages"],
                region_dim="unit_id",
                region_col="unit_id",
            )

    def test_unknown_region_dim_names_dimensions(self, tmp_path):
        pytest.importorskip("xarray")
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        nc = tmp_path / "leaf.nc4"
        _write_netcdf(nc, _leaf_frame(), region_dim="unit_id")
        with pytest.raises(ValueError, match="dimensions"):
            read_netcdf_leaf(
                nc,
                variables=["value"],
                region_dim="region",
                region_col="unit_id",
            )


class TestMissingDependency:
    def test_actionable_message_without_xarray(self, monkeypatch, tmp_path):
        from cil_regionalization import netcdf_io

        monkeypatch.setitem(sys.modules, "xarray", None)
        with pytest.raises(ImportError, match=r"cil_regionalization\[netcdf\]"):
            netcdf_io.read_netcdf_leaf(
                tmp_path / "leaf.nc4",
                variables=["value"],
                region_dim="region",
                region_col="unit_id",
            )


class TestPipelineConfigValidation:
    def test_netcdf_requires_region_dim(self):
        from cil_regionalization.pipelines.montecarlo import DataConfig

        with pytest.raises(ValueError, match="region_dim"):
            DataConfig.model_validate(
                {"format": "netcdf", "kind": "extensive", "version": "v1"}
            )

    def test_region_dim_forbidden_for_parquet(self):
        from cil_regionalization.pipelines.montecarlo import DataConfig

        with pytest.raises(ValueError, match="only meaningful"):
            DataConfig.model_validate(
                {
                    "format": "parquet",
                    "kind": "extensive",
                    "version": "v1",
                    "region_dim": "region",
                }
            )

    def test_composite_key_artifact_rejected(self, tmp_path):
        from cil_regionalization.pipelines.montecarlo import DataConfig, _read_leaf

        cfg = DataConfig.model_validate(
            {
                "format": "netcdf",
                "kind": "extensive",
                "version": "v1",
                "region_dim": "region",
            }
        )
        frame = _artifact().frame
        schema = OutputSchema(
            id_fields=("target_id",),
            weight_names=("area",),
            source_units=SourceUnits.from_string_ids(["iso", "adm1"]),
            normalization="per_source",
        )
        composite = WeightsArtifact(
            frame=frame,
            schema=schema,
            regions_version="targets-v1",
            source_version=SOURCE_VERSION,
        )
        with pytest.raises(ValueError, match="one source key column"):
            _read_leaf(tmp_path / "leaf.nc4", cfg, composite)


class TestUnitsBackstop:
    """A percent-labeled variable is a ratio; reading it under
    kind='extensive' is refused. Catches exactly the '% GDP' levels
    trap; ratios with percent-free units or metadata-free formats still
    rest on the caller's declaration, as the reader docstring states."""

    def test_percent_units_refused_for_extensive(self, tmp_path):
        xr = pytest.importorskip("xarray")
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        ds = xr.Dataset(
            {"share": (["unit_id", "year"], [[0.1, 0.2]])},
            coords={"unit_id": ["u1"], "year": [2050, 2051]},
        )
        ds["share"].attrs["units"] = "% GDP"
        nc = tmp_path / "share.nc4"
        ds.to_netcdf(nc)
        with pytest.raises(ValueError, match="meaningless"):
            read_netcdf_leaf(
                nc, variables=["share"], region_dim="unit_id",
                region_col="unit_id", kind="extensive",
            )
        # The same variable reads fine under intensive, and without kind.
        out = read_netcdf_leaf(
            nc, variables=["share"], region_dim="unit_id",
            region_col="unit_id", kind="intensive",
        )
        assert len(out) == 2
        out = read_netcdf_leaf(
            nc, variables=["share"], region_dim="unit_id", region_col="unit_id"
        )
        assert len(out) == 2


class TestRegionLabels:
    def _write_uncoordinated_leaf(self, path):
        import numpy as np
        import xarray as xr

        # region dimension with no coordinate; ids in a separate variable,
        # the mortality tree layout
        ds = xr.Dataset(
            {
                "rebased": (("year", "region"), np.arange(6.0).reshape(3, 2)),
                "regions": (("region",), np.array(["u1", "u2"], dtype=object)),
            },
            coords={"year": [2050, 2051, 2052]},
        )
        ds.to_netcdf(path)

    def test_labels_variable_becomes_the_region_column(self, tmp_path):
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        p = tmp_path / "leaf.nc4"
        self._write_uncoordinated_leaf(p)
        frame = read_netcdf_leaf(
            p, variables=["rebased"], region_dim="region",
            region_col="unit_id", region_labels="regions",
        )
        assert sorted(frame["unit_id"].unique()) == ["u1", "u2"]
        assert len(frame) == 6

    def test_without_labels_the_column_is_positional(self, tmp_path):
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        p = tmp_path / "leaf.nc4"
        self._write_uncoordinated_leaf(p)
        frame = read_netcdf_leaf(
            p, variables=["rebased"], region_dim="region", region_col="unit_id",
        )
        assert sorted(frame["unit_id"].unique()) == [0, 1]

    def test_missing_labels_variable_raises(self, tmp_path):
        from cil_regionalization.netcdf_io import read_netcdf_leaf

        p = tmp_path / "leaf.nc4"
        self._write_uncoordinated_leaf(p)
        with pytest.raises(ValueError, match="no variable 'nope'"):
            read_netcdf_leaf(
                p, variables=["rebased"], region_dim="region",
                region_col="unit_id", region_labels="nope",
            )

    def test_labels_with_wrong_dimensions_raise(self, tmp_path):
        import numpy as np
        import xarray as xr

        from cil_regionalization.netcdf_io import read_netcdf_leaf

        p = tmp_path / "leaf.nc4"
        ds = xr.Dataset(
            {
                "rebased": (("year", "region"), np.arange(6.0).reshape(3, 2)),
                "regions": (("year", "region"),
                            np.array([["a", "b"]] * 3, dtype=object)),
            },
            coords={"year": [2050, 2051, 2052]},
        )
        ds.to_netcdf(p)
        with pytest.raises(ValueError, match="expected exactly"):
            read_netcdf_leaf(
                p, variables=["rebased"], region_dim="region",
                region_col="unit_id", region_labels="regions",
            )
