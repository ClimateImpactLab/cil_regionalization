"""Dissolve the combined GADM 2.0 file into ADM0, ADM1, and ADM2 layers.

One-time preparation for IR-to-ADM weight generation, kept as a
documented, rerunnable script. The generic dissolve machinery lives in
`segment_weights.dissolve`; this driver holds only the GADM 2.0 facts.

Source
------
A processed copy of the combined GADM 2.0 shapefile (a single layer at
the deepest available level, 218,238 features, EPSG:4326); the caller
supplies its path with --source. The version label names that copy,
not bare "gadm-2.0", because the file is not the pristine GADM
distribution: it carries added attribute columns of unrecorded
provenance, and its name fields have dropped accented characters
("Crdoba", "Ro Negro"). Names are carried to the output layers for
orientation only; the stable keys are the integer composites and
nothing may ever join on a name.

Keys and expected unit counts
-----------------------------
    adm0 : ISO                   253 units
    adm1 : (ISO, ID_1)         3,428 units
    adm2 : (ISO, ID_1, ID_2)  42,726 units

The script verifies the dissolved counts against these figures and exits
nonzero on any discrepancy. The input inventory counted 42,727 distinct
ADM2 key combinations in the raw attribute table; the expected output is
one less because the (RUS, 71, 0) combination is excluded, see below.

Semantics of id value 0, established by inspection of the source
----------------------------------------------------------------
An id of 0 means "no unit at this level", in three distinct situations,
all kept as real units under their honest keys rather than edited:

- Undivided territories: 38 ISOs have only ID_1 == 0 rows (Aruba,
  Singapore, Malta, Vatican, Antarctica, and similar). Each becomes the
  single ADM1 unit (ISO, 0), and likewise at ADM2. Three of these are
  GADM pseudo-country codes for non-country features: "CA-" Caspian
  Sea, "CL-" Clipperton, "SP-" Spratly. They are kept; a water body
  will simply attract no overlap and shows up in the weight manifest's
  empty-region accounting.
- Nepal, source defect: the 172 pieces of the Dhaualagiri zone carry
  ID_1 == 0 because the source never assigned the zone to a development
  region (it appears nowhere with a positive ID_1). They become the unit
  (NPL, 0), and at ADM2 (NPL, 0, 8). Reassigning the zone to the West
  region (ID_1 5) would be factually right but is an edit to source
  data; it is not made here and needs an explicit decision.
- Russia, source defect, excluded: Sverdlovsk (RUS, 71) contains two
  unnamed fragments ("NA (1)", "NA (2)") with ID_2 == 0 whose area
  (0.028688 square degrees, about 200 square kilometers) lies entirely
  inside the named rayons of the same oblast; the measured intersection
  equals their full area to six decimals. They are duplicate scraps,
  not unassigned territory, and they are the only source of area
  non-conservation in the whole file (the global dissolve discrepancy
  equals their area exactly; every other unit's residual is at the
  1e-9 square degree level or below). They are dropped via an explicit,
  manifest-recorded exclusion, which restores area conservation at the
  default tolerance instead of relaxing it and avoids producing an ADM2
  unit that overlaps its neighbors.

Units without an ADM2 subdivision follow the same convention: 848
(ISO, ID_1) groups have only ID_2 == 0 rows and become (ISO, ID_1, 0).

Run
---
    python examples/gadm20/prepare_gadm20_targets.py [--source ...]
        [--out ...] [--version ...]

Outputs are GeoParquet plus targets.manifest.json under the gitignored
data directory. Reading the ~1.6 GB source and dissolving takes a while;
nothing else is touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from segment_weights.dissolve import LevelSpec, prepare_target_layers

_DEFAULT_OUT = _ROOT / "data" / "targets" / "gadm20"
_DEFAULT_VERSION = "gadm-2.0-impactmap-copy-2025"

_LEVELS = [
    LevelSpec("adm0", ("ISO",), ("NAME_0",)),
    LevelSpec("adm1", ("ISO", "ID_1"), ("NAME_0", "NAME_1")),
    LevelSpec("adm2", ("ISO", "ID_1", "ID_2"), ("NAME_0", "NAME_1", "NAME_2")),
]

_EXPECTED_UNITS = {"adm0": 253, "adm1": 3428, "adm2": 42726}

# The two duplicate Sverdlovsk scraps; see the module docstring for the
# evidence. Recorded in the output manifest as drop_query.
_DROP_QUERY = "ISO == 'RUS' and ID_1 == 71 and ID_2 == 0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dissolve combined GADM 2.0 into per-level target layers."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="path to the combined GADM 2.0 shapefile (gadm2.shp)",
    )
    parser.add_argument("--out", type=str, default=str(_DEFAULT_OUT))
    parser.add_argument("--version", type=str, default=_DEFAULT_VERSION)
    args = parser.parse_args(argv)

    manifest = prepare_target_layers(
        args.source,
        _LEVELS,
        args.out,
        version=args.version,
        drop_query=_DROP_QUERY,
    )
    print(f"dropped {manifest['n_dropped_features']} features via {_DROP_QUERY!r}")

    rc = 0
    for name, expected in _EXPECTED_UNITS.items():
        got = manifest["levels"][name]["n_units"]
        marker = "ok" if got == expected else "MISMATCH"
        if got != expected:
            rc = 1
        print(
            f"{name}: {got} units (expected {expected}, {marker}); "
            f"area rel diff {manifest['levels'][name]['area_rel_diff']:.2e}; "
            f"repaired {manifest['levels'][name]['n_repaired_geometries']} geometries"
        )
    print(f"manifest: {manifest['manifest_path']}")
    if rc != 0:
        print(
            "unit counts do not match the inventory figures; do not use these "
            "layers until the discrepancy is explained",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
