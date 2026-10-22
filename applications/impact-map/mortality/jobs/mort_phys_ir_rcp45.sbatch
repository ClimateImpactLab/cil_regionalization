#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=mort_phys_ir_rcp45
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --output=mort_phys_ir_rcp45_%j.log

# Sized from the measured ADM2 precedent: the full mortality ADM2
# pipeline peaked at 188.3 GB at the pooled-statistics stage. The
# impact region pooled frame is smaller (24,378 regions x 4 windows
# against 42,726 targets x 3) and dictionary-encodes hierid, but the
# same leaves are read, so the request stays at the proven 224 GB.
# The memory[...] lines in the log say what it actually cost.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python -m cil_regionalization.pipelines.montecarlo /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/mortality_2021_ir_rcp45.toml
