# s51 weights

Weights from a 1 degree grid to impact regions, population and area, on
the ECMWF system 51 grid.

This example does not run from a clone: the configs read the impact
region shapefile and the 30 arc second population raster from the CIL
filesystem on Midway3, as the `gadm20` and `gadm41` configs do. The
outputs are committed under `data/weights/`.

## Run it

On Midway3, from the repository root:

```bash
export CILREG_PYTHON=/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python

sbatch examples/s51/s51.sbatch examples/s51/grid1deg_to_ir.toml
```

One run over all 24,378 regions: about four minutes on 16 cpus, peaking
near 7 GB.

For the older geometry, run the two parts, wait for both to finish,
then combine them:

```bash
sbatch examples/s51/s51.sbatch examples/s51/s51_canonical.toml
sbatch examples/s51/s51.sbatch examples/s51/s51_supplement.toml
"$CILREG_PYTHON" examples/s51/merge.py
```

## Files

- `grid1deg_to_ir.toml` builds the weights on world-combo-201710, in one
  run. Prefer these: regions in world-combo-new overlap, so its
  population totals count about 131.7 million people twice.
- `s51_canonical.toml` and `s51_supplement.toml` build the weights
  on world-combo-new.
- `merge.py` combines those two into `data/weights/s51/`.
- `s51.sbatch` runs one config, passed as its argument.

## Inputs

Each run reads two files from the CIL filesystem.

The impact region shapefile, 24,378 polygons keyed by `hierid`:
`/project/cil/gcp/regions/world-combo-201710/agglomerated-world-new.shp`
for world-combo-201710, or
`/project/cil/gcp/regions/world-combo-new/agglomerated-world-new.shp`
for world-combo-new. The two have identical attribute tables; the
identifying columns of three rows:

| hierid | ISO | gadmid |
|---|---|---|
| ABW | ABW | 3360 |
| AFG.1.12 | AFG | 62 |
| ATA | ATA | 2836 |

The population raster, GPW v4 rev 10 adjusted to UN WPP 2015,
`/project/cil/gcp/social/population/gpw_v4r10_unwpp_2015/raster_geotiff/gpw_v4r10_unwpp_2015_30ss.tif`:
30 arc second cells (43,200 by 21,600) covering longitude -180 to 180
and latitude -90 to 90, holding people per cell.

## Outputs

Each directory holds a `weights.parquet` and its manifest.

- `data/weights/grid1deg_to_ir/` 24,378 regions, world-combo-201710
- `data/weights/s51/` 24,378 regions, world-combo-new
- `data/weights/s51_canonical/` the 24,361 that go into `s51/`
- `data/weights/s51_supplement/` the other 17

Three rows of `data/weights/grid1deg_to_ir/weights.parquet`:

| hierid | cell_ix | cell_iy | cell_lon | cell_lat | popwt | pop_raw | pop_method | areawt | area_raw | area_method |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| ABW | 109 | 102 | -70.5 | 12.5 | 0.5546 | 56,695 | native | 0.3392 | 6.171e7 | native |
| AFG.1.12 | 251 | 126 | 71.5 | 36.5 | 0.1010 | 2,111 | native | 0.07796 | 8.611e8 | native |
| ATA | 0 | 0 | -179.5 | -89.5 | 8.682e-6 | 0 | area_fallback | 8.682e-6 | 1.089e8 | native |

- `popwt` and `areawt` are each cell's share of the region's population
  and area, and sum to one per region.
- `pop_raw` and `area_raw` are the unnormalized people and square metres.
- `pop_method` says whether the population share came from the raster
  (`native`) or fell back to area (`area_fallback`); `area_method` is
  always `native`.

The merged `s51/` table carries one extra column, `source`.
