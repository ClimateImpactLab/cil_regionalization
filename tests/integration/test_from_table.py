"""WeightsArtifact.from_table: weight tables that were not generated here.

The committed fixture ``tests/data/external/cil_segment_weights.csv`` is
laid out like a CIL segment weight file: one row per (hierid, grid
cell) with ``pix_cent_x`` / ``pix_cent_y`` centroids, normalized
``popwt`` and ``areawt`` columns, and one centroid written as 180.125
(the cell the grid's [-180,180) convention calls -179.875).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cil_regionalization as cilreg
from cil_regionalization import schema as schema_mod
from cil_regionalization.apply import WeightsArtifact, apply_weights
from cil_regionalization.config import GridConfig
from cil_regionalization.grid import GridSpec
from cil_regionalization.schema import GRID_CELLS, OutputSchema, SourceUnits

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data" / "external" / "cil_segment_weights.csv"
)
GRID = "grid-0.25deg-center-[-180,180)"


def _from_fixture(**kwargs) -> WeightsArtifact:
    opts = dict(id_fields=["hierid"], weights={"pop": "popwt"}, source=GRID)
    opts.update(kwargs)
    return WeightsArtifact.from_table(FIXTURE, **opts)


def _cell_data(values: dict[tuple[int, int], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"cell_ix": ix, "cell_iy": iy, "value": v}
         for (ix, iy), v in values.items()]
    )


class TestGridTable:
    def test_reads_csv_without_manifest(self):
        art = _from_fixture()
        assert art.schema.source_units == GRID_CELLS
        assert art.schema.id_fields == ("hierid",)
        assert art.schema.weight_names == ("pop",)
        assert art.normalization == "per_destination"
        assert art.source_version == GRID
        assert art.regions_version is None
        assert art.partial_coverage is None
        assert len(art.frame) == 5

    def test_cells_derived_from_pix_cent_with_wrap(self):
        f = _from_fixture().frame.set_index(["hierid", "pix_cent_x"])
        # 180.125 is outside [-180,180) and wraps to the first cell
        assert f.loc[("AAA", 180.125), "cell_ix"] == 0
        assert f.loc[("AAA", 180.125), "cell_lon"] == -179.875
        assert f.loc[("AAA", 179.875), "cell_ix"] == 1439
        assert f.loc[("AAA", 179.875), "cell_iy"] == 400
        assert f.loc[("BBB", 0.125), "cell_ix"] == 720
        assert f.loc[("BBB", 0.125), "cell_iy"] == 359
        assert f.loc[("CCC.1", 0.375), "cell_ix"] == 721
        assert str(f["cell_ix"].dtype) == "int64"
        assert str(f["cell_iy"].dtype) == "int64"

    def test_derived_cells_match_grid_math(self):
        f = _from_fixture().frame
        spec = GridSpec.from_config(
            GridConfig(mode="generate", resolution=0.25))
        lon = np.where(f["pix_cent_x"] >= 180.0, f["pix_cent_x"] - 360.0,
                       f["pix_cent_x"])
        ix, iy = spec.index_of(lon, f["pix_cent_y"])
        assert np.array_equal(f["cell_ix"], ix)
        assert np.array_equal(f["cell_iy"], iy)

    @pytest.mark.parametrize("lon_name,lat_name", [
        ("lon", "lat"), ("longitude", "latitude"), ("cell_lon", "cell_lat"),
    ])
    def test_other_centroid_spellings(self, lon_name, lat_name):
        df = pd.read_csv(FIXTURE).rename(
            columns={"pix_cent_x": lon_name, "pix_cent_y": lat_name})
        art = WeightsArtifact.from_table(
            df, id_fields=["hierid"], weights={"pop": "popwt"}, source=GRID)
        ref = _from_fixture().frame
        assert art.frame["cell_ix"].tolist() == ref["cell_ix"].tolist()
        assert art.frame["cell_iy"].tolist() == ref["cell_iy"].tolist()

    def test_weight_columns_renamed_to_schema_form(self):
        df = pd.read_csv(FIXTURE).rename(
            columns={"popwt": "pop_share", "areawt": "area_share"})
        art = WeightsArtifact.from_table(
            df, id_fields=["hierid"],
            weights={"pop": "pop_share", "area": "area_share"}, source=GRID)
        assert art.schema.weight_names == ("pop", "area")
        assert {"popwt", "areawt"} <= set(art.frame.columns)
        assert not {"pop_share", "area_share"} & set(art.frame.columns)

    def test_rename_refuses_to_overwrite(self):
        df = pd.read_csv(FIXTURE).assign(pop_share=lambda d: d["popwt"])
        with pytest.raises(ValueError, match="would overwrite"):
            WeightsArtifact.from_table(
                df, id_fields=["hierid"], weights={"pop": "pop_share"},
                source=GRID)

    def test_parquet_reads_the_same(self, tmp_path):
        p = tmp_path / "w.parquet"
        pd.read_csv(FIXTURE).to_parquet(p, index=False)
        a = WeightsArtifact.from_table(
            p, id_fields=["hierid"], weights={"pop": "popwt"}, source=GRID)
        b = _from_fixture()
        cols = ["hierid", "cell_ix", "cell_iy", "popwt"]
        pd.testing.assert_frame_equal(
            a.frame[cols].reset_index(drop=True),
            b.frame[cols].reset_index(drop=True))

    def test_ids_read_as_strings(self, tmp_path):
        p = tmp_path / "w.csv"
        pd.DataFrame({"code": ["001", "002"], "lon": [0.125, 0.375],
                      "lat": [0.125, 0.125], "w": [1.0, 1.0]}).to_csv(
            p, index=False)
        art = WeightsArtifact.from_table(
            p, id_fields=["code"], weights={"pop": "w"}, source=GRID)
        assert art.frame["code"].tolist() == ["001", "002"]

    def test_unknown_extension_rejected(self, tmp_path):
        p = tmp_path / "w.txt"
        p.write_text("x")
        with pytest.raises(ValueError, match="cannot tell the format"):
            WeightsArtifact.from_table(
                p, id_fields=["hierid"], weights={"pop": "popwt"}, source=GRID)

    def test_indices_alone_are_taken_on_trust(self):
        # no centroid columns, so nothing can show these indices were
        # built on [0,360) (720 is 180.125 there, 0.125 on the named
        # grid); only the exact `source` string stands behind them
        df = pd.DataFrame({"r": ["A", "A"], "cell_ix": [720, 1439],
                           "cell_iy": [359, 359], "w": [0.5, 0.5]})
        art = WeightsArtifact.from_table(
            df, id_fields=["r"], weights={"pop": "w"}, source=GRID)
        assert art.frame["cell_ix"].tolist() == [720, 1439]
        assert "cell_lon" not in art.frame.columns

    @staticmethod
    def _zero_360_table() -> pd.DataFrame:
        # a 0.25 degree table on [0,360): same 1440 columns as the
        # [-180,180) grid, indices counted from lon 0 instead of -180
        lon = np.array([0.125, 180.125, 359.875])
        return pd.DataFrame({
            "r": ["A", "A", "B"],
            "pix_cent_x": lon,
            "pix_cent_y": [-0.125, -0.125, -0.125],
            "cell_ix": np.floor(lon / 0.25).astype("int64"),
            "cell_iy": [359, 359, 359],
            "w": [0.5, 0.5, 1.0],
        })

    def test_zero_360_indices_with_centroids_rejected(self):
        df = self._zero_360_table()
        assert df["cell_ix"].tolist() == [0, 720, 1439]
        with pytest.raises(ValueError,
                           match="likely built on a different grid") as e:
            WeightsArtifact.from_table(
                df, id_fields=["r"], weights={"pop": "w"}, source=GRID)
        assert "3 rows" in str(e.value)

    def test_zero_360_indices_accepted_on_their_own_grid(self):
        art = WeightsArtifact.from_table(
            self._zero_360_table(), id_fields=["r"], weights={"pop": "w"},
            source="grid-0.25deg-center-[0,360)")
        assert art.frame["cell_ix"].tolist() == [0, 720, 1439]

    def test_indices_matching_wrapped_centroids_accepted(self):
        # the same [0,360) longitudes with indices for the named grid:
        # wrapping makes them agree, so the table passes
        df = self._zero_360_table().assign(cell_ix=[720, 0, 1439 - 720])
        art = WeightsArtifact.from_table(
            df, id_fields=["r"], weights={"pop": "w"}, source=GRID)
        assert art.frame["cell_ix"].tolist() == [720, 0, 719]

    def test_latitude_descending_indices_rejected(self):
        # rows counted from the north pole; a latitude column alone is
        # enough to expose them
        df = pd.DataFrame({"r": ["A", "A"], "cell_ix": [0, 0],
                           "cell_iy": [0, 719], "lat": [89.875, -89.875],
                           "w": [0.5, 0.5]})
        with pytest.raises(ValueError, match=r"\['lat'\]"):
            WeightsArtifact.from_table(
                df, id_fields=["r"], weights={"pop": "w"}, source=GRID)
        ok = df.assign(cell_iy=[719, 0])
        WeightsArtifact.from_table(
            ok, id_fields=["r"], weights={"pop": "w"}, source=GRID)

    def test_explicit_cell_indices_outside_grid_rejected(self):
        df = pd.DataFrame({"r": ["A"], "cell_ix": [1440], "cell_iy": [5],
                           "w": [1.0]})
        with pytest.raises(ValueError, match="outside the 1440 x 720 grid"):
            WeightsArtifact.from_table(
                df, id_fields=["r"], weights={"pop": "w"}, source=GRID)

    def test_points_off_centroids_rejected(self):
        df = pd.DataFrame({"r": ["A"], "lon": [0.2], "lat": [0.125],
                           "w": [1.0]})
        with pytest.raises(ValueError, match="not cell centroids"):
            WeightsArtifact.from_table(
                df, id_fields=["r"], weights={"pop": "w"}, source=GRID)

    def test_ambiguous_or_missing_centroid_columns(self):
        df = pd.DataFrame({"r": ["A"], "lon": [0.125], "pix_cent_x": [0.125],
                           "lat": [0.125], "w": [1.0]})
        with pytest.raises(ValueError, match="several longitude columns"):
            WeightsArtifact.from_table(
                df, id_fields=["r"], weights={"pop": "w"}, source=GRID)
        with pytest.raises(ValueError, match="no longitude column"):
            WeightsArtifact.from_table(
                df.drop(columns=["lon", "pix_cent_x"]), id_fields=["r"],
                weights={"pop": "w"}, source=GRID)

    def test_zero_360_grid_wraps_negative_longitudes(self):
        df = pd.DataFrame({"r": ["A", "A"], "lon": [-179.875, 0.125],
                           "lat": [0.125, 0.125], "w": [0.5, 0.5]})
        art = WeightsArtifact.from_table(
            df, id_fields=["r"], weights={"pop": "w"},
            source="grid-0.25deg-center-[0,360)")
        assert art.frame["cell_lon"].tolist() == [180.125, 0.125]
        assert art.frame["cell_ix"].tolist() == [720, 0]

    def test_grid_spelling_must_match_grid_identity(self):
        with pytest.raises(ValueError, match="use that spelling"):
            _from_fixture(source="grid-0.250deg-center-[-180,180)")
        with pytest.raises(ValueError, match="looks like a grid"):
            _from_fixture(source="grid-quarter-degree")

    def test_source_keys_not_allowed_for_grid(self):
        with pytest.raises(ValueError, match="only for polygon sources"):
            _from_fixture(source_keys=["hierid"])


class TestChecks:
    def test_sum_to_one_failure_names_group(self):
        df = pd.read_csv(FIXTURE)
        df.loc[df["hierid"] == "AAA", "popwt"] = [0.6, 0.6]
        with pytest.raises(ValueError, match="do not sum to 1") as e:
            WeightsArtifact.from_table(
                df, id_fields=["hierid"], weights={"pop": "popwt"},
                source=GRID)
        assert "AAA" in str(e.value)
        assert "BBB" not in str(e.value)

    def test_duplicates_rejected_naming_key(self):
        df = pd.read_csv(FIXTURE)
        df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
        df.loc[df["hierid"] == "BBB", "popwt"] = 0.5
        with pytest.raises(ValueError, match="share a \\(region, source") as e:
            WeightsArtifact.from_table(
                df, id_fields=["hierid"], weights={"pop": "popwt"},
                source=GRID)
        assert "BBB" in str(e.value)

    def test_check_false_skips_both(self):
        df = pd.read_csv(FIXTURE)
        df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
        art = WeightsArtifact.from_table(
            df, id_fields=["hierid"], weights={"pop": "popwt"}, source=GRID,
            check=False)
        assert len(art.frame) == 6

    def test_per_source_checks_source_groups(self):
        df = pd.DataFrame({"adm": ["X", "Y", "X"],
                           "unit_id": ["u1", "u1", "u2"],
                           "w": [0.25, 0.75, 1.0]})
        art = WeightsArtifact.from_table(
            df, id_fields=["adm"], weights={"pop": "w"}, source="units-v1",
            source_keys=["unit_id"], normalization="per_source")
        assert art.schema.normalization_group == ("unit_id",)
        bad = df.assign(w=[0.25, 0.5, 1.0])
        with pytest.raises(ValueError, match="u1"):
            WeightsArtifact.from_table(
                bad, id_fields=["adm"], weights={"pop": "w"},
                source="units-v1", source_keys=["unit_id"],
                normalization="per_source")

    def test_bad_arguments(self):
        with pytest.raises(ValueError, match="normalization must be"):
            _from_fixture(normalization="per_region")
        with pytest.raises(ValueError, match="source_keys must name"):
            _from_fixture(source="world-combo-201710")
        with pytest.raises(ValueError, match="'nope', which is not in"):
            _from_fixture(weights={"pop": "nope"})
        with pytest.raises(ValueError, match="id_fields"):
            _from_fixture(id_fields=["region"])


class TestApplication:
    def test_intensive_application_end_to_end(self):
        art = _from_fixture()
        data = _cell_data({(0, 400): 10.0, (1439, 400): 20.0,
                           (720, 359): 1.0, (721, 359): 5.0})
        out = apply_weights(
            art, data, kind="intensive", weight="pop", data_version=GRID,
        ).frame.set_index("hierid")["value"]
        assert out["AAA"] == pytest.approx(0.6 * 10.0 + 0.4 * 20.0)
        assert out["BBB"] == pytest.approx(1.0)
        assert out["CCC.1"] == pytest.approx(0.75 * 1.0 + 0.25 * 5.0)

    def test_matches_hand_built_artifact(self):
        built = _from_fixture()
        hand = WeightsArtifact(
            frame=built.frame,
            schema=OutputSchema(id_fields=("hierid",), weight_names=("pop",),
                                source_units=GRID_CELLS,
                                normalization="per_destination"),
            regions_version=None,
            source_version=GRID,
        )
        data = _cell_data({(0, 400): 3.0, (1439, 400): -2.0,
                           (720, 359): 7.5, (721, 359): 0.25})
        a = apply_weights(built, data, kind="intensive", weight="pop",
                          data_version=GRID).frame
        b = apply_weights(hand, data, kind="intensive", weight="pop",
                          data_version=GRID).frame
        pd.testing.assert_frame_equal(a, b)

    def test_polygon_source_application(self):
        df = pd.DataFrame({"adm": ["X", "Y", "X"],
                           "unit_id": ["u1", "u1", "u2"],
                           "w": [0.25, 0.75, 1.0]})
        art = WeightsArtifact.from_table(
            df, id_fields=["adm"], weights={"pop": "w"}, source="units-v1",
            source_keys=["unit_id"], normalization="per_source")
        assert art.schema.source_units == SourceUnits.from_string_ids(
            ["unit_id"])
        data = pd.DataFrame({"unit_id": ["u1", "u2"], "value": [8.0, 2.0]})
        out = apply_weights(art, data, kind="extensive", weight="pop",
                            data_version="units-v1").frame.set_index("adm")
        assert out.loc["X", "value"] == pytest.approx(0.25 * 8.0 + 2.0)
        assert out.loc["Y", "value"] == pytest.approx(0.75 * 8.0)


class TestPackageRoot:
    def test_one_import_suffices(self):
        art = cilreg.WeightsArtifact.from_table(
            FIXTURE, id_fields=["hierid"], weights={"pop": "popwt"},
            source=GRID)
        assert isinstance(art, cilreg.WeightsArtifact)

    def test_schema_types_exported(self):
        assert cilreg.OutputSchema is schema_mod.OutputSchema
        assert cilreg.SourceUnits is schema_mod.SourceUnits
        assert cilreg.GRID_CELLS is schema_mod.GRID_CELLS
        for name in ("OutputSchema", "SourceUnits", "GRID_CELLS"):
            assert name in cilreg.__all__
