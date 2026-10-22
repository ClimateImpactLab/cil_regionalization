#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=labor_groups_ir_rcp85
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --output=labor_groups_ir_rcp85_%j.log

# One job per rcp emits both risk group products; the leaf reads are
# the same two files the combined run reads, so the groups cost one
# run, not two.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_physical_labor_groups.py --adm ir --rcp rcp85 --workers 16
