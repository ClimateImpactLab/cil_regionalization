"""Output schema construction and validation."""
from __future__ import annotations

import pandas as pd
import pytest

from segment_weights.schema import (
    CELL_COLUMNS,
    GRID_CELLS,
    OutputSchema,
    SourceUnits,
    method_column,
    raw_column,
    require_normalization,
    weight_column,
)


class TestColumnNames:
    def test_weight_column_no_underscore(self):
        # spec: {name}wt; no underscore between name and wt
        assert weight_column("pop") == "popwt"
        assert weight_column("area") == "areawt"

    def test_raw_column_has_underscore(self):
        assert raw_column("pop") == "pop_raw"

    def test_method_column_has_underscore(self):
        assert method_column("pop") == "pop_method"

    def test_cell_columns_in_order(self):
        assert CELL_COLUMNS == ("cell_ix", "cell_iy", "cell_lon", "cell_lat")


class TestOutputSchema:
    def test_columns_ordered_ids_cell_weights(self):
        s = OutputSchema(id_fields=("region_id",), weight_names=("pop", "area"))
        assert s.columns == (
            "region_id",
            "cell_ix",
            "cell_iy",
            "cell_lon",
            "cell_lat",
            "popwt",
            "pop_raw",
            "pop_method",
            "areawt",
            "area_raw",
            "area_method",
        )

    def test_multi_id_fields_ordered(self):
        s = OutputSchema(id_fields=("country", "admin1"), weight_names=("pop",))
        assert s.columns[:5] == (
            "country",
            "admin1",
            "cell_ix",
            "cell_iy",
            "cell_lon",
        )

    def test_dtypes_match_columns(self):
        s = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        d = s.dtypes
        assert set(d.keys()) == set(s.columns)
        assert d["cell_ix"] == "int64"
        assert d["cell_lon"] == "float64"
        assert d["popwt"] == "float64"
        assert d["pop_method"] == "string"

    def test_empty_frame_has_schema(self):
        s = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        df = s.empty_frame()
        assert list(df.columns) == list(s.columns)
        assert len(df) == 0


class TestValidateFrame:
    def test_well_formed_frame_passes(self):
        s = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        df = pd.DataFrame(
            {
                "region_id": pd.array(["A", "B"], dtype="string"),
                "cell_ix": [0, 1],
                "cell_iy": [0, 1],
                "cell_lon": [0.5, 1.5],
                "cell_lat": [0.5, 1.5],
                "popwt": [1.0, 1.0],
                "pop_raw": [10.0, 5.0],
                "pop_method": pd.array(["native", "native"], dtype="string"),
            }
        )
        s.validate_frame(df)

    def test_missing_column_raises(self):
        s = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        df = pd.DataFrame({"region_id": ["A"]})
        with pytest.raises(ValueError, match="missing columns"):
            s.validate_frame(df)

    def test_dtype_mismatch_raises(self):
        s = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        df = pd.DataFrame(
            {
                "region_id": ["A"],
                "cell_ix": ["zero"],  # wrong dtype
                "cell_iy": [0],
                "cell_lon": [0.5],
                "cell_lat": [0.5],
                "popwt": [1.0],
                "pop_raw": [10.0],
                "pop_method": ["native"],
            }
        )
        with pytest.raises(ValueError, match="dtype mismatches"):
            s.validate_frame(df)


class TestSourceUnits:
    def test_grid_cells_reproduce_cell_layout(self):
        assert GRID_CELLS.key_columns == ("cell_ix", "cell_iy")
        assert GRID_CELLS.meta_columns == ("cell_lon", "cell_lat")
        assert GRID_CELLS.columns == CELL_COLUMNS
        assert GRID_CELLS.dtypes["cell_ix"] == "int64"
        assert GRID_CELLS.dtypes["cell_lon"] == "float64"

    def test_string_ids_layout(self):
        units = SourceUnits.from_string_ids(["hierid"])
        assert units.key_columns == ("hierid",)
        assert units.meta_columns == ()
        assert units.dtypes == {"hierid": "string"}

    def test_string_ids_require_at_least_one_column(self):
        with pytest.raises(ValueError, match="at least one key column"):
            SourceUnits.from_string_ids([])


class TestNormalizationDirection:
    def test_default_is_per_destination_over_id_fields(self):
        s = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        assert s.normalization == "per_destination"
        assert s.normalization_group == ("region_id",)

    def test_per_source_groups_by_source_keys(self):
        s = OutputSchema(
            id_fields=("gid_1",),
            weight_names=("pop",),
            source_units=SourceUnits.from_string_ids(["hierid"]),
            normalization="per_source",
        )
        assert s.normalization_group == ("hierid",)

    def test_string_source_unit_columns_and_dtypes(self):
        s = OutputSchema(
            id_fields=("gid_1",),
            weight_names=("pop",),
            source_units=SourceUnits.from_string_ids(["hierid"]),
            normalization="per_source",
        )
        assert s.columns == ("gid_1", "hierid", "popwt", "pop_raw", "pop_method")
        assert s.dtypes["hierid"] == "string"
        df = s.empty_frame()
        assert list(df.columns) == list(s.columns)
        s.validate_frame(df)


class TestRequireNormalization:
    def test_matching_direction_passes(self):
        require_normalization("per_destination", "per_destination")
        require_normalization("per_source", "per_source")

    def test_mismatch_raises_with_both_directions_named(self):
        with pytest.raises(ValueError, match="per_destination.*per_source"):
            require_normalization("per_destination", "per_source")
        with pytest.raises(ValueError, match="normalization mismatch"):
            require_normalization("per_source", "per_destination")
