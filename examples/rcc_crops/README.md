# RCC crop weights

Produces `weights.parquet`, `weights.manifest.json`, and the legacy 13-column `weights_legacy.csv` (area + crop + pop, 0.25° grid, world-combo-201710) under `data/out/rcc_crops/`. One cil node, LocalCluster Dask (32 workers x 2 threads). All raster + shapefile paths in `rcc_crops.toml` are real RCC locations; only `--account=YOUR_ACCOUNT_HERE` in `run_rcc_crops.sbatch` must be filled in before submitting.
The pop GPW directory may use a different .tif filename across snapshots; if so, `ls /project/cil/gcp/social/population/gpw_v4r10_unwpp_2015/raster_geotiff/` and adjust the `pop` weight's `raster` line to match.
```
module load python
source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/
segweights run examples/rcc_crops/rcc_crops.toml --test-mode --legacy-csv   # smoke
sbatch examples/rcc_crops/run_rcc_crops.sbatch                              # full run
```
