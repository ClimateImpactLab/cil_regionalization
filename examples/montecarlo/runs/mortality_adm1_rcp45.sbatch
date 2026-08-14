#!/bin/bash
# Production run mortality_adm1_rcp45. Submit with sbatch; no business logic lives here.
#SBATCH --job-name=segweights-mortality_adm1_rcp45
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=segweights_mortality_adm1_rcp45_%j.out

# The pinned cluster environment (envs/README.md; built by
# envs/setup_cluster_env.sh).
module load python/miniforge-25.3.0
source /project/cil/home_dirs/rcc/envs/segment_weights_py311/bin/activate

set -euo pipefail

# Fast fail: prove the environment before touching the tree.
python -c "import segment_weights, exactextract, geopandas, rasterio, xarray, netCDF4"
CONFIG="/project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/examples/montecarlo/runs/mortality_adm1_rcp45.toml"

python -m segment_weights.pipelines.montecarlo "$CONFIG" --dry-run
python -m segment_weights.pipelines.montecarlo "$CONFIG"
