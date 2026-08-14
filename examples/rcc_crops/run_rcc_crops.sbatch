#!/bin/bash
#SBATCH --job-name=rcc_crops
#SBATCH --partition=cil
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=251G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rcc_crops_%j.out
#SBATCH --error=logs/rcc_crops_%j.err

# RCC crop weights production run.
#
# Test the config first (caps regions for a fast smoke; ~1 minute):
#     segweights run examples/rcc_crops/rcc_crops.toml --test-mode --legacy-csv
# Then submit this script for the full ~24k-region run.

set -euo pipefail

module load python
source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs data/out/rcc_crops

echo "host         : $(hostname)"
echo "cores        : ${SLURM_CPUS_PER_TASK}"
echo "mem          : ${SLURM_MEM_PER_NODE}M"
echo "start        : $(date -Iseconds)"

# LocalCluster Dask: 32 workers x 2 threads, configured in
# rcc_crops.toml under [backend.local]. The cluster lives inside the
# Python process and shuts down on exit; no separate scheduler job.
segweights run examples/rcc_crops/rcc_crops.toml --legacy-csv

echo "done         : $(date -Iseconds)"
echo "outputs      : data/out/rcc_crops/{weights.parquet,weights.manifest.json,weights_legacy.csv}"
