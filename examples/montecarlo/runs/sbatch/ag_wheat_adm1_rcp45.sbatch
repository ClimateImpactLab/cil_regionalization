#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=ag_wheat_adm1_rcp45
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=ag_wheat_adm1_rcp45_%j.log

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python -m cil_regionalization.pipelines.montecarlo /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/ag_wheat_adm1_rcp45.toml
