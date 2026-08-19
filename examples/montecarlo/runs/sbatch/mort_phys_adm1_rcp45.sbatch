#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=mort_phys_adm1_rcp45
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=mort_phys_adm1_rcp45_%j.log

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python -m cil_regionalization.pipelines.montecarlo /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/mortality_2021_adm1_rcp45.toml
