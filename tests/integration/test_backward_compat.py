"""Backward compatibility of the schema generalization.

The source-unit and normalization-direction work must not change what a
grid-to-IR run produces. The golden file pins that:

    tests/data/golden/grid_ir_weights.parquet

was written by the pre-change implementation from the committed synthetic
fixtures (``tests/data/synthetic/regions.parquet`` + ``raster.tif``) with
the exact config in ``_golden_cfg`` below. The comparison is exact on
structure (columns, order, dtypes, row count, ids, method labels) and
holds float values to a tight relative tolerance; see
``_assert_matches_golden`` for what that tolerance absorbs and what it
does not excuse. Regenerate only when the output is *meant* to change,
by running ``_golden_cfg`` through ``LocalBackend`` and rewriting the
parquet; a diff in that commit is the reviewable record of the change.

Also pinned here: existing example configs still load with the default
direction, the manifest records the direction, the grid backends refuse
per_source, and a weights artifact cannot be consumed against its
recorded direction without an error.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from segment_weights.backends.local import LocalBackend
from segment_weights.config import Config, load_config
from segment_weights.grid import GridSpec
from segment_weights.io import write_result
from segment_weights.manifest import build_manifest
from segment_weights.regions import RegionSet
from segment_weights.schema import require_normalization
from segment_weights.weights import from_config_list


_REPO = Path(__file__).resolve().parents[2]
_SYNTHETIC = _REPO / "tests" / "data" / "synthetic"
_GOLDEN = _REPO / "tests" / "data" / "golden" / "grid_ir_weights.parquet"


def _assert_matches_golden(frame: pd.DataFrame, golden: pd.DataFrame) -> None:
    """Structure exact, float values at rtol 1e-12.

    The tolerance exists for one reason: the geodesic stack compiled for
    different platforms (macOS arm64 pip wheels vs linux-64 conda builds)
    rounds boundary-cell areas differently in the last bit, observed as
    1 ULP (about 2e-16 relative) on boundary-cell weights; interior cells
    and synthesized rows agree exactly. rtol 1e-12 sits four orders of
    magnitude above that noise floor and six below the run-level sum
    tolerance (1e-6), so it absorbs the platform difference and nothing
    else. It is not permission for logic drift: any real change to
    normalization, allocation, or zonal sums moves values far past it,
    and structural changes (row counts, columns, dtypes, integer keys,
    method labels, row order) still fail exactly because non-float
    columns are always compared exactly.
    """
    assert list(frame.columns) == list(golden.columns)
    assert len(frame) == len(golden)
    pd.testing.assert_frame_equal(
        frame, golden, check_dtype=True, check_exact=False, rtol=1e-12, atol=0.0
    )


def _golden_cfg(**overrides) -> Config:
    data = {
        "project": {"name": "golden"},
        "regions": {
            "path": str(_SYNTHETIC / "regions.parquet"),
            "id_fields": ["region_id"],
        },
        "grid": {
            "mode": "generate",
            "resolution": 1.0,
            "offset": "center",
            "lon_convention": "[-180,180)",
        },
        "weights": [
            {
                "name": "pop",
                "raster": str(_SYNTHETIC / "raster.tif"),
                "fallback": "area",
            },
            {"name": "area"},
        ],
        "backend": {"kind": "local", "coverage": "exact_fraction"},
        "output": {"dir": "unused"},
    }
    data.update(overrides)
    return Config.model_validate(data)


def _run(cfg: Config):
    return LocalBackend().compute(
        RegionSet.from_config(cfg.regions),
        GridSpec.from_config(cfg.grid),
        from_config_list(cfg.weights),
        cfg,
    )


class TestGoldenGridToIr:
    def test_output_identical_to_pre_change_golden(self):
        result = _run(_golden_cfg())
        golden = pd.read_parquet(_GOLDEN)
        _assert_matches_golden(result.frame, golden)

    def test_schema_records_default_direction(self):
        result = _run(_golden_cfg())
        assert result.schema.normalization == "per_destination"
        assert result.schema.normalization_group == ("region_id",)


class TestGoldenDateline:
    """Pins the antimeridian path of the segment builder: dateline-crossing
    multipolygons must keep producing exactly the pre-change cells and
    weights while the builder gains a polygon mode."""

    def test_dateline_output_identical_to_golden(
        self, dateline_crossers_geoparquet
    ):
        cfg = _golden_cfg(
            regions={
                "path": str(dateline_crossers_geoparquet),
                "id_fields": ["region_id"],
            }
        )
        result = _run(cfg)
        golden = pd.read_parquet(
            _GOLDEN.parent / "grid_dateline_weights.parquet"
        )
        _assert_matches_golden(result.frame, golden)


class TestGoldenCentroidCrop:
    """Pins pixel_centroid coverage and the area_weighted_sum crop path."""

    def test_centroid_crop_output_identical_to_golden(
        self, synthetic_fraction_raster
    ):
        cfg = _golden_cfg(
            weights=[
                {
                    "name": "crop",
                    "raster": str(synthetic_fraction_raster),
                    "kind": "area_weighted_sum",
                    "fallback": "nan",
                },
                {
                    "name": "pop",
                    "raster": str(_SYNTHETIC / "raster.tif"),
                    "fallback": "area",
                },
                {"name": "area"},
            ],
            backend={"kind": "local", "coverage": "pixel_centroid"},
        )
        result = _run(cfg)
        golden = pd.read_parquet(
            _GOLDEN.parent / "grid_centroid_crop_weights.parquet"
        )
        _assert_matches_golden(result.frame, golden)


class TestExistingConfigsStillLoad:
    """The committed example configs predate the normalization key and must
    keep loading with the per_destination default."""

    @pytest.mark.parametrize(
        "rel",
        [
            "examples/rcc_crops/rcc_crops.toml",
            "examples/configs/s51.toml",
        ],
    )
    def test_example_config_loads_with_default_direction(self, rel):
        cfg = load_config(_REPO / rel)
        assert cfg.normalization == "per_destination"


class TestManifestRecordsDirection:
    def test_build_manifest_carries_normalization(self):
        m = build_manifest(_golden_cfg())
        assert m.normalization == "per_destination"
        assert json.loads(m.to_json())["normalization"] == "per_destination"


class TestGridBackendRejectsPerSource:
    def test_local_backend_raises(self):
        cfg = _golden_cfg(normalization="per_source")
        with pytest.raises(ValueError, match="per_source.*not supported"):
            _run(cfg)


class TestWrongDirectionConsumption:
    """A written weights artifact records its direction in the manifest;
    consuming it for the transposed operation must fail."""

    def test_consuming_against_recorded_direction_raises(self, tmp_path):
        result = _run(_golden_cfg())
        paths = write_result(result, str(tmp_path / "out"), "parquet")
        recorded = json.loads(Path(paths["manifest"]).read_text())["normalization"]
        # Correct use passes.
        require_normalization(recorded, "per_destination")
        # Allocation of an extensive quantity needs per_source weights;
        # this file records per_destination, so the check must refuse.
        with pytest.raises(ValueError, match="normalization mismatch"):
            require_normalization(recorded, "per_source")
