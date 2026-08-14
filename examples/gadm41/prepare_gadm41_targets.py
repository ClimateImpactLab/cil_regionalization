"""Dissolve the GADM 4.1 GeoPackage into ADM0, ADM1, and ADM2 layers.

One-time preparation for IR-to-ADM weight generation against the GADM
4.1 unit universe, parallel to examples/gadm20/prepare_gadm20_targets.py
and sharing the same generic machinery in `cil_regionalization.dissolve`.
This driver holds only the GADM 4.1 facts, established by attribute
queries against the file rather than assumption.

Source
------
The GADM 4.1 GeoPackage, gadm_410.gpkg as distributed by gadm.org. The
caller supplies its path with --source. One layer (gadm_410), 356,508
features at the deepest available subdivision, EPSG:4326, UID unique
per feature. Unlike the processed 2.0 copy, the copy inspected matches
the pristine distribution in the respects checked: accented names are
intact (Cordoba and Rio Negro carry their accents, 50,580 rows have
non-ASCII NAME_1), and there are no added columns of unknown
provenance. The version label names the copy that was read, because
provenance is about which bytes were read, not about how pristine they
look.

Keys
----
Stable hierarchical string codes with a version suffix (USA, USA.1_1,
USA.1.1_1), not 2.0's integer composites. A missing level is an empty
string, never NULL. Dissolve keys are the cumulative GID columns, so an
undivided country is the honest unit (GID_0, "") at ADM1 and
(GID_0, "", "") at ADM2, mirroring 2.0's id-zero convention with ""
in place of 0.

    adm0 : GID_0                     263 units
    adm1 : (GID_0, GID_1)          3,685 units
    adm2 : (GID_0, GID_1, GID_2)  48,009 units

Composition of the expected counts: 3,661 distinct nonempty GID_1 plus
24 undivided countries; 47,217 distinct nonempty GID_2 plus 768 ADM1
units with no ADM2 below them plus the same 24. The undivided count is
24, not the 25 the raw query printed: that query listed countries with
any empty GID_1 rows, which includes GBR, and GBR is not undivided (it
has 9,107 properly keyed rows; its 4 empty rows are the repaired defect
below, folding into England and Scotland rather than forming a
(GBR, "") unit). The first run pinned 3,686 and 48,010 by counting GBR
both ways at once; the dissolve came back one short at each level and
the guard held the layers until this arithmetic was corrected. The
undivided set includes the GADM pseudo-codes XCA (Caspian), XCL
(Clipperton), XPI and XSP (Paracel and Spratly), and ATA; water bodies
attract no overlap and surface in the weight manifests' empty-region
accounting.

The GBR repair, decided from inspection
---------------------------------------
Four features carry an empty GID_1 with a populated GID_2: UIDs 330111,
330122, 330128 (Blackpool, GBR.1.6_1) and 338143 (Shetland Islands,
GBR.3.27_1). Every other row with the same GID_2 carries the correct
GID_1 (GBR.1_1 England, GBR.3_1 Scotland). Measured against the union
of those properly keyed rows, all four have intersection share 0.0000:
they are genuinely separate coastal fragments, the opposite of the 2.0
Sverdlovsk duplicates, and dropping them would discard real area (the
Shetland piece alone is 0.068 square degrees). The repair fills GID_1
from the GID_2 prefix (GBR.1.6_1 states its parent is GBR.1, version
suffix carried over), validated against the country's existing GID_1
vocabulary. Without it, dissolving would fabricate a (GBR, "") ADM1
unit and split Blackpool and Shetland across two parents. The repair
and its evidence are recorded in the output manifest.

No analogue exists here for 2.0's other defects: Nepal's five
development regions are all assigned (no Dhaualagiri orphan), and the
Sverdlovsk duplicate pair has no counterpart (the single NA-named
Russian row has regular GID codes, and names are never keys).

Run
---
    python examples/gadm41/prepare_gadm41_targets.py --source <gadm_410.gpkg>
        [--out ...] [--version ...]

Outputs are GeoParquet plus targets.manifest.json under the gitignored
data directory. The source read is the expensive step (2.7 GB); run
where the file is on local storage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from cil_regionalization.dissolve import LevelSpec, prepare_target_layers

_DEFAULT_OUT = _ROOT / "data" / "targets" / "gadm41"
_DEFAULT_VERSION = "gadm-4.10-impactmap-copy-2022"

_LEVELS = [
    LevelSpec("adm0", ("GID_0",), ("NAME_0",)),
    LevelSpec("adm1", ("GID_0", "GID_1"), ("NAME_0", "NAME_1")),
    LevelSpec("adm2", ("GID_0", "GID_1", "GID_2"), ("NAME_0", "NAME_1", "NAME_2")),
]

_EXPECTED_UNITS = {"adm0": 263, "adm1": 3685, "adm2": 48009}

_EXPECTED_REPAIR_ROWS = 4

_REPAIR_RECORD = {
    "name": "fill missing GBR GID_1 from the GID_2 prefix",
    "scope": "GID_0 == 'GBR' and GID_1 == '' and GID_2 != ''",
    "rows": _EXPECTED_REPAIR_ROWS,
    "uids": [330111, 330122, 330128, 338143],
    "rule": (
        "GID_2 is hierarchical (country.a.b_version); the parent GID_1 is "
        "country.a_version. Derived values are validated against the "
        "country's existing nonempty GID_1 vocabulary before assignment."
    ),
    "evidence": (
        "All other rows sharing GID_2 GBR.1.6_1 (Blackpool, 20 rows) and "
        "GBR.3.27_1 (Shetland Islands, 7 rows) carry GID_1 GBR.1_1 and "
        "GBR.3_1. The four affected features intersect the union of those "
        "properly keyed rows with share 0.0000: separate coastal "
        "fragments, not duplicates, so dropping would discard real area."
    ),
}


def _parent_gid1(gid2: str) -> str:
    code, _, version = gid2.rpartition("_")
    return code.rsplit(".", 1)[0] + "_" + version


def _repair_gbr_gid1(gdf):
    mask = (
        (gdf["GID_0"] == "GBR") & (gdf["GID_1"] == "") & (gdf["GID_2"] != "")
    )
    n = int(mask.sum())
    if n != _EXPECTED_REPAIR_ROWS:
        raise ValueError(
            f"GBR repair expected {_EXPECTED_REPAIR_ROWS} rows with empty "
            f"GID_1 and nonempty GID_2, found {n}; the source has changed "
            f"and the recorded repair no longer describes it"
        )
    vocabulary = set(
        gdf.loc[(gdf["GID_0"] == "GBR") & (gdf["GID_1"] != ""), "GID_1"]
    )
    derived = gdf.loc[mask, "GID_2"].map(_parent_gid1)
    unknown = sorted(set(derived) - vocabulary)
    if unknown:
        raise ValueError(
            f"GBR repair derived GID_1 values {unknown} that do not exist "
            f"in the country's GID_1 vocabulary; refusing to invent units"
        )
    gdf = gdf.copy()
    gdf.loc[mask, "GID_1"] = derived
    return gdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dissolve the GADM 4.1 GeoPackage into per-level target layers."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="path to the GADM 4.1 GeoPackage (gadm_410.gpkg)",
    )
    parser.add_argument("--out", type=str, default=str(_DEFAULT_OUT))
    parser.add_argument("--version", type=str, default=_DEFAULT_VERSION)
    args = parser.parse_args(argv)

    manifest = prepare_target_layers(
        args.source,
        _LEVELS,
        args.out,
        version=args.version,
        repair=_repair_gbr_gid1,
        repair_record=_REPAIR_RECORD,
    )
    print(f"repair applied: {_REPAIR_RECORD['name']} ({_REPAIR_RECORD['rows']} rows)")

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
            "unit counts do not match the expected figures; do not use these "
            "layers until the discrepancy is explained",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
