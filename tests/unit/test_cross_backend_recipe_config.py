"""Production-config pin: examples/cross_backend_recipe/local_world_combo_2017.toml.

Same role as test_rcc_crops_config.py: verifies the cross-backend recipe
config parses and that the load-bearing fields are the ones that make the
output directly comparable to the s51 BQ deliverable.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from segment_weights.config import Config


_TOML = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "cross_backend_recipe"
    / "local_world_combo_2017.toml"
)


def _load() -> Config:
    with _TOML.open("rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)


class TestCrossBackendRecipeConfig:
    def test_file_exists(self):
        assert _TOML.exists(), f"missing recipe config: {_TOML}"

    def test_regions_match_s51_source_shapefile(self):
        cfg = _load()
        # Same canonical world-combo-2017 shapefile the s51 supplement
        # consumed; so the local-backend output is over the IDENTICAL
        # region set as the BQ deliverable.
        assert cfg.regions.path == (
            "gs://impactlab-data/spatial/shapefiles/source/impactlab/"
            "world-combo-new/agglomerated-world-new.shp"
        )
        assert cfg.regions.id_fields == ["hierid"]

    def test_grid_matches_s51_one_degree_centered(self):
        cfg = _load()
        # 1.0 deg, center offset, lon [-180, 180): match the s51 BQ
        # deliverable cell-for-cell.
        assert cfg.grid.resolution == 1.0
        assert cfg.grid.offset == "center"
        assert cfg.grid.lon_convention == "[-180,180)"

    def test_pop_points_at_real_gpw_geotiff(self):
        cfg = _load()
        pop = next(w for w in cfg.weights if w.name == "pop")
        assert pop.raster is not None
        assert pop.raster.startswith(
            "/project/cil/gcp/social/population/"
            "gpw_v4r10_unwpp_2015/raster_geotiff/"
        )
        assert pop.raster.endswith(".tif")
        # Area fallback so zero-pop regions still produce a normalised
        # row; matches the s51 BQ deliverable's behaviour.
        assert pop.fallback == "area"

    def test_pop_and_area_weights_only(self):
        cfg = _load()
        names = {w.name for w in cfg.weights}
        # The s51 BQ deliverable is pop + area; the local twin must
        # match that exact shape for comparison.
        assert names == {"pop", "area"}

    def test_local_backend_with_dask_localcluster(self):
        cfg = _load()
        assert cfg.backend.kind == "local"
        assert cfg.backend.local.dask == "local"
        assert cfg.backend.local.n_workers == 32
        assert cfg.backend.local.threads_per_worker == 2

    def test_cross_backend_tolerance_is_one_em3(self):
        cfg = _load()
        # The locked default; the recipe's report uses this as the
        # above-budget count threshold.
        assert cfg.validation.cross_backend_tolerance == 1e-3
