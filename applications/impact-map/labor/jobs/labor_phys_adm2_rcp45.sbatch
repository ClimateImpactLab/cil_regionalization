#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=labor_phys_adm2_rcp45
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=labor_phys_adm2_rcp45_%j.log

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_physical_labor.py --adm adm2 --rcp rcp45 --workers 16
