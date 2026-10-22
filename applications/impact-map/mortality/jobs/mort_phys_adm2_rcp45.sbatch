#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=mort_phys_adm2_rcp45
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --output=mort_phys_adm2_rcp45_%j.log

# Measured, not estimated: the instrumented run peaked at 188.3 GB
# resident in the parent at the pooled-statistics stage (full
# mortality ADM2, ~4,000 leaves x 42,726 targets). 128 GB died
# three times. Keep ADM2-scale pipeline requests at 224 GB or more.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python -m cil_regionalization.pipelines.montecarlo /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/mortality_2021_adm2_rcp45.toml
