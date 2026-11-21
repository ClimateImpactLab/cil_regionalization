#!/bin/bash
#SBATCH --job-name=s51-weights
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/midway3/%u/%x_%j.log

# Builds the weights for the config passed as the argument:
# grid1deg_to_ir.toml on world-combo-201710, or the two parts on
# world-combo-new, s51_canonical.toml and s51_supplement.toml. The
# resources above are for grid1deg_to_ir and the canonical part; the
# supplement runs serially and asks for less. Measured on 20 November
# 2026 with 48 cpus: the canonical part (job 63637514) took 2:55 with a
# 7.2 GB peak at 212 percent CPU, the supplement (job 63637515) 0:18
# with 1.7 GB. The log goes to your Midway3 scratch. Submit from the
# repository root:
#
#   CILREG_PYTHON=/path/to/python sbatch --job-name=s51-canonical \
#       examples/s51/s51.sbatch examples/s51/s51_canonical.toml
#   CILREG_PYTHON=/path/to/python sbatch --job-name=s51-supplement \
#       --cpus-per-task=1 --mem=8G --time=01:00:00 \
#       examples/s51/s51.sbatch examples/s51/s51_supplement.toml
#   CILREG_PYTHON=/path/to/python sbatch --job-name=grid1deg-to-ir \
#       examples/s51/s51.sbatch examples/s51/grid1deg_to_ir.toml
#
# Then combine the two s51 parts with examples/s51/merge.py;
# grid1deg_to_ir is a single run and needs no merge.

set -euo pipefail
: "${CILREG_PYTHON:?set CILREG_PYTHON to a python with cil_regionalization installed}"
config=${1:?pass the config path, for example examples/s51/s51_canonical.toml}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export PYTHONSAFEPATH=1

# INFO for cil_regionalization prints the regions repaired with
# shapely.make_valid, which grid runs do not record in the manifest.
# The code goes in with -c, not on stdin: Dask spawns its workers as
# new interpreters, and those rerun the main module when it has a file.
/usr/bin/time -v "$CILREG_PYTHON" -c '
import logging, sys
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("cil_regionalization").setLevel(logging.INFO)
from cil_regionalization.cli import main
sys.exit(main(["run", sys.argv[1]]))
' "$config"
