# Cluster environment

This directory builds the environment for the full pipeline runs on
the University of Chicago RCC cluster (midway3). It is not how you
install the package; that is `pip install`, described in the main
README. Most readers never need anything here.

The environment lives at
`/project/cil/home_dirs/rcc/envs/cil_regionalization_py311`, a shared
group path on that cluster, pinned and built fresh with uv rather than
layered over the group's shared conda stack.

## Why this environment exists

Three problems pushed us to a fresh, pinned build. A venv layered on
the shared stack inherits whatever that stack becomes, so rebuilding
it months later does not give you the environment that was validated.
Mixing conda-built and pip-built geospatial packages actually broke
during validation: a conda pyproj and a pip rasterio disagreed about
which PROJ database to use. And the shared stack was missing pieces
this repo needs (`exactextract` in some environments, the BigQuery
client in others), so whether things worked depended on which shared
environment happened to be active.

The pins in `requirements-cluster.txt` are the versions the full test
suite passed against, including pandas 3.0.3 and numpy 2.4.6. After a
successful build, the setup script writes
`requirements-cluster.freeze.txt`, the fully resolved snapshot
including transitive dependencies; installing from that file
reproduces the environment exactly.

## Setup

    bash envs/setup_cluster_env.sh

The script creates the venv on Python 3.11 and installs
`exactextract` first, in its own step, because it is the one package
that may lack a platform wheel; if that is where things break, the
script stops immediately and says so. It then installs the pins,
installs `cil_regionalization` editable without re-resolving dependencies,
runs an import check, runs the full test suite, and writes the freeze
file. It will not overwrite an existing environment.

## Rebuilding the environment

Remove the environment directory and rerun the script. For an exact
rebuild, install from `requirements-cluster.freeze.txt` instead of
`requirements-cluster.txt` (same script, swap the file). If the pins
ever change, the test suite at the end of the script decides whether
the build is usable: only commit a freeze file from a build whose
suite passed.

## What it supports

One environment runs the whole repo: weight generation with the local
backend, the application and statistics stages, the Monte Carlo
pipeline with NetCDF leaves, and the offline BigQuery tests (the
client libraries are included so those 54 tests run instead of
skipping; real BigQuery runs additionally need credentials). Dask
packages are installed but the dask code paths remain unvalidated;
their tests stay marker-skipped by default.
