# Cross-backend recipe

Characterizes the local backend (GPW GeoTIFF) against the s51 BigQuery deliverable (`weights_complete.parquet`) on the same world-combo-2017 shapefile + 1.0° grid. Reports per-region `popwt` and `areawt` agreement distributions against `validation.cross_backend_tolerance`. Characterization, NOT a regression gate; geodesic vs spherical area + planar vs spherical intersection produce a small boundary-cell tail by design.
The `local_world_combo_2017.toml` paths are real RCC + GCS locations; no edits needed. The GPW directory may use a different .tif filename in your snapshot (`ls /project/cil/gcp/social/population/gpw_v4r10_unwpp_2015/raster_geotiff/` to confirm). Then:
```
module load python
source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/
segweights run examples/cross_backend_recipe/local_world_combo_2017.toml --test-mode   # smoke
segweights run examples/cross_backend_recipe/local_world_combo_2017.toml               # full local run
python examples/cross_backend_recipe/compare.py --local data/out/cross_backend_recipe/weights.parquet --weight popwt
```
