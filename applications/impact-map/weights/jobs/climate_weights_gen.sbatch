#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=climate_weights_gen
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --output=climate_weights_gen_%j.log

PY=/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python
CFGS=/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/data/run_configs

$PY -m cil_regionalization.cli run $CFGS/grid025_to_gadm20_adm1_per_destination.toml
$PY -m cil_regionalization.cli run $CFGS/grid025_to_gadm20_adm2_per_destination.toml
