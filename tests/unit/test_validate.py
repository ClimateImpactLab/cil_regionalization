"""Sum-to-1 validator: clean and corrupted output frames."""
from __future__ import annotations

import pandas as pd
import pytest

from cil_regionalization.schema import OutputSchema
from cil_regionalization.validate import check_sum_to_one


def _clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_id": ["A", "A", "B", "B"],
            "cell_ix": [0, 1, 0, 1],
            "cell_iy": [0, 0, 0, 0],
            "cell_lon": [0.5, 1.5, 5.5, 6.5],
            "cell_lat": [0.5, 0.5, 5.5, 5.5],
            "popwt": [0.4, 0.6, 0.3, 0.7],
            "pop_raw": [40.0, 60.0, 30.0, 70.0],
            "pop_method": ["native"] * 4,
            "areawt": [0.5, 0.5, 0.5, 0.5],
            "area_raw": [1.0, 1.0, 1.0, 1.0],
            "area_method": ["native"] * 4,
        }
    )


class TestCleanFrame:
    def test_clean_frame_passes(self):
        schema = OutputSchema(
            id_fields=("region_id",), weight_names=("pop", "area")
        )
        report = check_sum_to_one(_clean_frame(), schema, tolerance=1e-9)
        assert report.ok
        assert report.n_regions == 2
        assert len(report.failures) == 0
        assert "passed" in report.summary()


class TestCorruptedFrame:
    def test_corrupt_one_region_one_weight(self):
        df = _clean_frame()
        # break region A's pop sum
        df.loc[0, "popwt"] = 0.5  # was 0.4; A now sums to 1.1
        schema = OutputSchema(id_fields=("region_id",), weight_names=("pop", "area"))
        report = check_sum_to_one(df, schema, tolerance=1e-6)
        assert not report.ok
        assert len(report.failures) == 1
        bad = report.failures.iloc[0]
        assert bad["region_id"] == "A"
        assert bad["weight"] == "pop"
        assert bad["sum"] == pytest.approx(1.1)
        assert "FAILED" in report.summary()

    def test_tolerance_accepts_small_drift(self):
        df = _clean_frame()
        df.loc[0, "popwt"] = 0.4 + 1e-12
        schema = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        report = check_sum_to_one(df, schema, tolerance=1e-6)
        assert report.ok

    def test_zero_weight_region_fails(self):
        df = _clean_frame()
        df.loc[df["region_id"] == "A", "popwt"] = 0.0
        schema = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        report = check_sum_to_one(df, schema, tolerance=1e-6)
        assert not report.ok
        assert any(report.failures["region_id"] == "A")


class TestGridInvariants:
    """Catches what sum-to-1 cannot. The s51 antimeridian bug produced
    rows that summed to 1 per region per weight while silently routing
    populated cells away from their points. These invariants would have
    flagged it."""

    def _grid(self):
        from cil_regionalization.config import GridConfig
        from cil_regionalization.grid import GridSpec

        return GridSpec.from_config(
            GridConfig(
                mode="generate",
                resolution=1.0,
                offset="center",
                lon_convention="[-180,180)",
            )
        )

    def _schema(self):
        from cil_regionalization.schema import OutputSchema

        return OutputSchema(id_fields=("region_id",), weight_names=("pop",))

    def _clean_row(self) -> dict:
        return {
            "region_id": "A",
            "cell_ix": 100,
            "cell_iy": 50,
            "cell_lon": -79.5,
            "cell_lat": -39.5,
            "popwt": 1.0,
            "pop_raw": 10.0,
            "pop_method": "native",
        }

    def test_clean_frame_passes(self):
        from cil_regionalization.validate import check_grid_invariants

        df = pd.DataFrame([self._clean_row()])
        rep = check_grid_invariants(df, self._schema(), self._grid())
        assert rep.ok
        assert rep.n_rows == 1

    def test_cell_ix_out_of_range_caught(self):
        """The exact failure mode the antimeridian SQL bug produced."""
        from cil_regionalization.validate import check_grid_invariants

        row = self._clean_row()
        row["cell_ix"] = 360  # n_ix == 360, so 360 is out of range
        df = pd.DataFrame([row])
        rep = check_grid_invariants(df, self._schema(), self._grid())
        assert not rep.ok
        assert "cell_ix_out_of_range" in rep.failures["_invariant"].tolist()

    def test_cell_lon_out_of_range_caught(self):
        from cil_regionalization.validate import check_grid_invariants

        row = self._clean_row()
        row["cell_lon"] = 200.5  # outside [-180, 180)
        df = pd.DataFrame([row])
        rep = check_grid_invariants(df, self._schema(), self._grid())
        assert not rep.ok
        assert "cell_lon_out_of_range" in rep.failures["_invariant"].tolist()

    def test_duplicate_key_caught(self):
        """ATA's ix=0 + ix=360 (post-wrap collision) would land here."""
        from cil_regionalization.validate import check_grid_invariants

        df = pd.DataFrame([self._clean_row(), self._clean_row()])
        rep = check_grid_invariants(df, self._schema(), self._grid())
        assert not rep.ok
        assert "duplicate_key" in rep.failures["_invariant"].tolist()

    def test_summary_lists_invariant_counts(self):
        from cil_regionalization.validate import check_grid_invariants

        row = self._clean_row()
        row["cell_ix"] = 360
        df = pd.DataFrame([row])
        rep = check_grid_invariants(df, self._schema(), self._grid())
        assert "FAILED" in rep.summary()
        assert "cell_ix_out_of_range" in rep.summary()


class TestMissingColumn:
    def test_missing_weight_column_raises(self):
        df = _clean_frame().drop(columns=["popwt"])
        schema = OutputSchema(id_fields=("region_id",), weight_names=("pop",))
        with pytest.raises(ValueError, match="popwt"):
            check_sum_to_one(df, schema)


class TestPerSourceDirection:
    """check_sum_to_one follows the schema's normalization group: source
    unit keys under per_source, id fields under per_destination."""

    def _allocation_frame(self) -> pd.DataFrame:
        # Source X split 0.3/0.7 across targets P and Q; source Y entirely
        # in Q. Valid per_source; invalid per_destination on both targets
        # (P sums to 0.3, Q to 1.7).
        return pd.DataFrame(
            {
                "gid_1": ["P", "Q", "Q"],
                "hierid": ["X", "X", "Y"],
                "popwt": [0.3, 0.7, 1.0],
                "pop_raw": [30.0, 70.0, 40.0],
                "pop_method": ["native"] * 3,
            }
        )

    def _schema(self, normalization: str) -> OutputSchema:
        from cil_regionalization.schema import SourceUnits

        return OutputSchema(
            id_fields=("gid_1",),
            weight_names=("pop",),
            source_units=SourceUnits.from_string_ids(["hierid"]),
            normalization=normalization,
        )

    def test_allocation_frame_passes_per_source(self):
        report = check_sum_to_one(
            self._allocation_frame(), self._schema("per_source"), tolerance=1e-9
        )
        assert report.ok, report.summary()
        assert report.n_regions == 2  # two source units

    def test_same_frame_fails_per_destination(self):
        report = check_sum_to_one(
            self._allocation_frame(),
            self._schema("per_destination"),
            tolerance=1e-9,
        )
        assert not report.ok
        assert set(report.failures["gid_1"]) == {"P", "Q"}


class TestPolygonInvariants:
    """Key, coverage, and sum-to-1 invariants for polygon-mode frames,
    in both normalization directions."""

    def _schema(self, normalization: str = "per_source") -> OutputSchema:
        from cil_regionalization.schema import SourceUnits

        return OutputSchema(
            id_fields=("gid_1",),
            weight_names=("pop",),
            source_units=SourceUnits.from_string_ids(["hierid"]),
            normalization=normalization,
        )

    def _clean_allocation_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gid_1": ["P", "Q", "Q"],
                "hierid": ["X", "X", "Y"],
                "popwt": [0.3, 0.7, 1.0],
                "pop_raw": [30.0, 70.0, 40.0],
                "pop_method": ["native"] * 3,
            }
        )

    def test_clean_frame_passes_with_universe(self):
        from cil_regionalization.validate import check_polygon_invariants

        report = check_polygon_invariants(
            self._clean_allocation_frame(),
            self._schema(),
            expected_source_ids=[("X",), ("Y",)],
        )
        assert report.ok, report.summary()

    def test_duplicate_pair_flagged(self):
        from cil_regionalization.validate import check_polygon_invariants

        frame = pd.concat(
            [self._clean_allocation_frame()] * 2, ignore_index=True
        )
        report = check_polygon_invariants(frame, self._schema())
        assert not report.ok
        assert (report.failures["_invariant"] == "duplicate_key").all()
        assert len(report.failures) == 6  # every row is part of a duplicate

    def test_null_key_flagged(self):
        from cil_regionalization.validate import check_polygon_invariants

        frame = self._clean_allocation_frame()
        frame.loc[0, "hierid"] = None
        report = check_polygon_invariants(frame, self._schema())
        assert not report.ok
        assert "null_key" in set(report.failures["_invariant"])

    def test_missing_source_unit_reported(self):
        from cil_regionalization.validate import check_polygon_invariants

        report = check_polygon_invariants(
            self._clean_allocation_frame(),
            self._schema(),
            expected_source_ids=[("X",), ("Y",), ("Z",)],
        )
        assert not report.ok
        assert report.missing_source_units == (("Z",),)
        assert "have no row" in report.summary()

    def test_unknown_source_unit_flagged(self):
        from cil_regionalization.validate import check_polygon_invariants

        report = check_polygon_invariants(
            self._clean_allocation_frame(),
            self._schema(),
            expected_source_ids=[("X",)],  # Y is not in the universe
        )
        assert not report.ok
        unknown = report.failures.loc[
            report.failures["_invariant"] == "unknown_source_unit"
        ]
        assert set(unknown["hierid"]) == {"Y"}

    def test_sum_to_one_follows_direction(self):
        from cil_regionalization.validate import check_polygon_invariants

        frame = self._clean_allocation_frame()
        # Valid per_source (each hierid sums to 1) ...
        report = check_polygon_invariants(frame, self._schema("per_source"))
        assert report.ok, report.summary()
        # ... and invalid per_destination (P sums to 0.3, Q to 1.7).
        report = check_polygon_invariants(frame, self._schema("per_destination"))
        assert not report.ok
        assert not report.sum_report.ok
        assert len(report.failures) == 0  # keys are fine; only sums fail
