#!/bin/bash
# Build the pinned production environment with uv, for the University of
# Chicago RCC cluster (midway3). The module names below are that
# cluster's; on any other machine, install with pip as the main README
# describes.
#
# Run from your clone of this repository:
#     bash envs/setup_cluster_env.sh
#
# The script ends by running the full test suite, and on success it
# writes the fully resolved package snapshot to
# envs/requirements-cluster.freeze.txt, which should be committed so a
# rebuild months from now is identical down to the transitive
# dependencies.

set -euo pipefail

# Where the environment is created. The default is the group's shared
# environment directory on the cluster; point it elsewhere if you are
# not building the shared one.
ENV="${SEGMENT_WEIGHTS_ENV:-/project/cil/home_dirs/rcc/envs/segment_weights_py311}"
# The repository root, resolved from this script's own location.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load python/miniforge-25.3.0
module load uv

if [ -e "$ENV" ]; then
    echo "refusing to overwrite existing environment at $ENV" >&2
    echo "remove it first if a rebuild is intended" >&2
    exit 1
fi

uv venv --python 3.11 "$ENV"
source "$ENV/bin/activate"

# exactextract first, in its own step. It reached the shared conda
# stack via conda-forge and its PyPI wheels are the one piece of this
# stack that might not exist for this platform; if there is no
# manylinux wheel, pip falls back to a source build that needs GEOS
# headers and CMake, and that is the failure to surface immediately,
# not halfway through the stack.
if ! uv pip install exactextract==0.3.0; then
    echo "" >&2
    echo "exactextract failed to install. Most likely there is no wheel" >&2
    echo "for this platform and the source build lacks GEOS or CMake." >&2
    echo "Options: load a GEOS module and retry, or install exactextract" >&2
    echo "from conda-forge into a conda-based environment instead." >&2
    exit 1
fi

uv pip install -r "$REPO/envs/requirements-cluster.txt"

# Editable install without dependency resolution: every dependency is
# already pinned above, and re-resolving through pyproject would let
# the resolver move past the pins.
uv pip install --no-deps -e "$REPO"

echo "== import check =="
python -c "import segment_weights, exactextract, geopandas, rasterio, xarray, netCDF4, pyarrow; print('imports OK')"

echo "== test suite =="
python -m pytest "$REPO/tests" -q

echo "== capturing resolved snapshot =="
# The editable self-install line embeds the builder's clone path and is
# not installable by anyone else; the snapshot records dependencies only.
uv pip freeze | grep -v "^-e " > "$REPO/envs/requirements-cluster.freeze.txt"
echo "environment ready: $ENV"
echo "commit envs/requirements-cluster.freeze.txt to pin the full resolution"
