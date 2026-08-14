"""Production-config pin: examples/rcc_crops/rcc_crops.toml.

The TOML drives a real RCC run; this test makes sure the shape stays
correct under refactor. Does NOT check file existence (the rasters and
shapefile live on the cluster), only that the config parses and that
the load-bearing fields hold the agreed values.
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
    / "rcc_crops"
    / "rcc_crops.toml"
)


def _load() -> Config:
    """Parse the TOML directly, bypassing `load_config`'s path resolver.

    `load_config` rewrites relative file paths to absolute under the
    config file's directory; the rcc_crops TOML uses absolute RCC
    paths that don't exist on this machine, and the resolver would
    keep them as-is, but using `model_validate` keeps this test
    deliberately independent of that machinery.
    """
    with _TOML.open("rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)


class TestRccCropsConfig:
    def test_file_exists(self):
        assert _TOML.exists(), f"missing production config: {_TOML}"

    def test_config_parses(self):
        cfg = _load()
        assert cfg.project.name == "rcc_crops"

    def test_regions_shapefile(self):
        cfg = _load()
        assert cfg.regions.path == (
            "/project/cil/gcp/regions/world-combo-201710/"
            "agglomerated-world-new.shp"
        )
        assert cfg.regions.id_fields == ["hierid"]

    def test_grid_quarter_degree_centered(self):
        cfg = _load()
        assert cfg.grid.resolution == 0.25
        assert cfg.grid.offset == "center"
        assert cfg.grid.lon_convention == "[-180,180)"

    def test_crop_weight_area_weighted_with_nan_fallback(self):
        cfg = _load()
        crop = next(w for w in cfg.weights if w.name == "crop")
        # The load-bearing crop identity: fraction x geodesic area.
        assert crop.kind == "area_weighted_sum"
        # Zero-crop regions must produce NaN cropwt, matching the
        # legacy crop CSV convention.
        assert crop.fallback == "nan"
        # Source raster is the FRACTION file, not the _ha derivative.
        assert crop.raster == (
            "/project/cil/battuta_shares/gcp/estimation/agriculture/Data/"
            "1_raw/3_cropped_area/CroplandPastureArea2000_Geotiff/"
            "cropland2000_area.tif"
        )
        assert "_ha" not in crop.raster

    def test_pop_weight_points_at_real_gpw_geotiff(self):
        cfg = _load()
        pop = next(w for w in cfg.weights if w.name == "pop")
        # GPW v4 r10 UN-WPP-adjusted, 2015. README tells the user to
        # `ls` the directory and confirm the .tif filename across
        # snapshots; the directory itself is canonical.
        assert pop.raster is not None
        assert pop.raster.startswith(
            "/project/cil/gcp/social/population/"
            "gpw_v4r10_unwpp_2015/raster_geotiff/"
        )
        assert pop.raster.endswith(".tif")
        # Area fallback so zero-pop regions still carry normalised
        # weights downstream temperature aggregations can use.
        assert pop.fallback == "area"

    def test_pop_and_area_weights_present(self):
        cfg = _load()
        names = {w.name for w in cfg.weights}
        # The legacy 13-col exporter requires area + crop + pop.
        assert {"area", "crop", "pop"}.issubset(names)

    def test_local_backend_with_dask_localcluster(self):
        cfg = _load()
        assert cfg.backend.kind == "local"
        assert cfg.backend.local.dask == "local"
        # 64-core cil node; 32 x 2 keeps per-worker memory at ~8 GB.
        assert cfg.backend.local.n_workers == 32
        assert cfg.backend.local.threads_per_worker == 2

    def test_confirm_cost_false_for_non_interactive_sbatch(self):
        cfg = _load()
        # No interactive prompt; the sbatch is non-interactive by
        # definition. Flipping this to True would hang the job.
        assert cfg.output.confirm_cost is False
