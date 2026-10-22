#!/bin/bash
#SBATCH --job-name=energy-regionalize
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/cil/gcp/impact_map/energy/logs/regionalize_%j.out
#SBATCH --error=/project/cil/gcp/impact_map/energy/logs/regionalize_%j.err

# Regionalize the published energy delta-method products to GADM 2.0.
#
# Six inputs (total_energy valued damages + two physical fuels, two RCPs)
# to adm0, adm1 and adm2, plus the percent-of-GDP twin for the valued
# layer. 24 parquet files and 24 manifest sidecars under
# /project/cil/gcp/impact_map/energy/adm{0,1,2}/.
#
# Memory: 32G. The binding item is a ~600 MB CSV held in pandas
# (~2.5 GB) plus the adm2 merges; the script holds one input frame at a
# time. The outputs are small because the window reduction collapses
# 120 years to four windows: 42,726 x 13 rows at adm2, not 51M.
#
# Walltime: 24h is deliberately loose. The cost is CSV parsing and the
# adm2 merges; nothing here is a projection run.
#
# Reads the extraction tree and bc39 read-only. Writes only under
# /project/cil/gcp/impact_map/energy/.

set -euo pipefail

REPO=/project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization
mkdir -p /project/cil/gcp/impact_map/energy/logs

module unload python 2>/dev/null || true
source /project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/activate

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# gadm20-adm2-per-destination is the one artifact not already cached.
# fetch_weights would pull it mid-run anyway; doing it up front keeps a
# network failure out of the middle of a long job.
echo "=== fetch $(date -Is) ==="
cilreg fetch gadm20-adm2-per-destination

# adm0 only: the dry run exists to catch wiring faults (artifact fetch,
# zarr access, column contract, manifest construction) before the long
# reads start. Running it over every level would do the whole job twice
# for no extra signal, and the full run below writes incrementally so
# partial progress survives a failure anyway.
echo "=== dry run (adm0, wiring check) $(date -Is) ==="
/usr/bin/time -v python examples/energy/run_energy_regionalization.py \
    --dry-run --levels adm0

echo "=== full run $(date -Is) ==="
/usr/bin/time -v python examples/energy/run_energy_regionalization.py

echo "=== outputs $(date -Is) ==="
find /project/cil/gcp/impact_map/energy/adm0 \
     /project/cil/gcp/impact_map/energy/adm1 \
     /project/cil/gcp/impact_map/energy/adm2 \
     -maxdepth 1 -type f -printf '%12s  %p\n' 2>/dev/null | sort -k2
