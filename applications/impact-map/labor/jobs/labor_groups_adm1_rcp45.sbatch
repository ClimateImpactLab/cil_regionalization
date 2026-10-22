#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=labor_groups_adm1_rcp45
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=labor_groups_adm1_rcp45_%j.log

# Writes ADM1 and ADM0 for both risk groups: the ADM0 numerator and
# denominator are ISO sums taken before the division, as in the
# combined physical product.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_physical_labor_groups.py --adm adm1 --rcp rcp45 --workers 16
