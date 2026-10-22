#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=web_json_ag_log85
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=web_json_ag_log85_%j.log

# Refresh the web JSON for the six *_physical_log products under rcp85.
#
# The parquets are NOT regenerated and must not be. They are correct:
# the adm2 generation of 2026-09-03 and the adm0 and adm1 generation of
# 2026-09-29 came from identical code, configs, weights and inputs, and
# adm1 is exactly the crop area weighted mean of adm2 to 4e-10. What is
# stale is the published JSON, built on 2026-08-24 before the
# agriculture specific SMME table existed, so it carries the generic
# model weighting.
#
# Regenerating the parquets would also invalidate the percent products,
# since run_percent_from_logs.py derives those from these same files,
# and the 432 percent documents published on 2026-10-09 would go stale.
#
# rcp85 only. The rcp45 log parquets date from 2026-08-20 and 08-21 and
# their published documents from 08-24, so those are already current.
#
# REPLACES PUBLISHED OUTPUT. 108 documents per tree, 216 in total, all
# of which already exist in Box. Unit counts, periods and statistics are
# unchanged, so this is a value for value replacement.
#
# As with the percent fix, a bare make_web_json.py run cannot be used:
# the registry does not know the _base, _basea, _baseb, _pctgdp,
# _highrisk and _lowrisk product names now in the tree, and the first
# file in sorted order is one of them. The stage holds only the wanted
# parquets and mirrors <crop>/<level>/ so the job log reads like the
# real tree. No path component may equal "web", which the discovery
# filter excludes.

set -euo pipefail

SRC=/project/cil/gcp/impact_map/agriculture
STAGE=/project/cil/home_dirs/scadavidsanchez/scratch/ag_log85_docs
OUT=/project/cil/gcp/impact_map/web
PY=/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python
SCRIPT=/project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/make_web_json.py

CROPS="cassava corn rice sorghum soy wheat"
LEVELS="adm0 adm1 adm2"

rm -rf "$STAGE"
for crop in $CROPS; do
  for lv in $LEVELS; do
    mkdir -p "$STAGE/$crop/$lv"
    f="$SRC/$crop/$lv/${crop}_physical_log_rcp85.parquet"
    test -e "$f" || { echo "missing input: $f" >&2; exit 1; }
    ln -s "$f" "$STAGE/$crop/$lv/"
  done
done

staged=$(find "$STAGE" -name '*.parquet' | wc -l)
test "$staged" -eq 18 || { echo "staged $staged parquets, expected 18" >&2; exit 1; }
echo "staged $staged parquets under $STAGE"

# --out fixes the full tree; the lite tree follows as <out>-lite with the
# default statistics (mean,q17,q50,q83), matching the published set.
"$PY" "$SCRIPT" --root "$STAGE" --out "$OUT"

rm -rf "$STAGE"

for crop in $CROPS; do
  echo "$crop full: $(ls -1 "$OUT/${crop}_physical_log" | grep -c _rcp85_)"
  echo "$crop lite: $(ls -1 "${OUT}-lite/${crop}_physical_log" | grep -c _rcp85_)"
done
