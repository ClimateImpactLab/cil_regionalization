"""Build per-crop cropped-area weight artifacts for agriculture.

The agriculture sector aggregates its log-yield impacts as an
intensive quantity weighted by SAGE cropped area per crop, from
``agglomerated-world-new-hierid-crop-weights.csv``. This script turns
that file into per-destination weight artifacts, one per crop and
admin level, next to the existing population artifacts.

Each impact region's crop area splits across the targets it straddles
in proportion to its land-area split in the existing geometric
artifacts (211 of 24378 regions straddle a target boundary; the rest
map whole). Regions with zero area for a crop get a NaN weight and a
``nan`` method marker: the coverage machinery classifies them as
zero-weight sources, the run declares ``on_zero_weight = "skip"``, and
targets with no cropland for the crop come out absent rather than
zero. Each crop therefore has its own set of covered units.

Run once from anywhere that sees the repos; the inputs are small.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

def _cil(path: str) -> Path:
    """Resolve a /project/cil path, falling back to the laptop mount."""
    p = Path(path)
    if p.exists():
        return p
    alt = Path(path.replace("/project/cil", "/Volumes/cil"))
    if alt.exists():
        return alt
    raise FileNotFoundError(path)


WEIGHTS = _cil(
    "/project/cil/home_dirs/scadavidsanchez/repos/"
    "climate-and-damages-aggregations/data/weights"
)
CSV = _cil(
    "/project/cil/gcp/estimation/agriculture/Data/1_raw/3_cropped_area/"
    "agglomerated-world-new-hierid-crop-weights.csv"
)
CROPS = ["cassava", "corn", "rice", "sorghum", "soy", "wheat"]
ADMS = ["adm1", "adm2"]


def build(adm: str, areas: pd.DataFrame, csv_sha: str) -> None:
    base_dir = WEIGHTS / f"ir_{adm}_full"
    frame = pd.read_parquet(base_dir / "weights.parquet")
    base_manifest = json.loads((base_dir / "weights.manifest.json").read_text())
    id_fields = base_manifest["id_fields"]

    merged = frame.merge(areas, on="hierid", how="left", validate="many_to_one")
    missing = merged["hierid"][merged[CROPS[0]].isna()].unique()
    if len(missing):
        raise ValueError(
            f"{len(missing)} source units have no row in {CSV}: "
            f"{sorted(missing)[:10]}"
        )

    for crop in CROPS:
        # the region's crop area, split across straddled targets by the
        # land-area split recorded in the geometric artifact
        raw = merged[crop] * merged["areawt"] / merged.groupby("hierid")[
            "areawt"
        ].transform("sum")
        out = merged[id_fields + ["hierid"]].copy()
        out["croparea_raw"] = raw
        totals = out.groupby(id_fields)["croparea_raw"].transform("sum")
        out["cropareawt"] = out["croparea_raw"] / totals.where(totals > 0)
        zero = merged[crop] == 0
        out.loc[zero, ["cropareawt", "croparea_raw"]] = float("nan")
        out["croparea_method"] = "native"
        out.loc[zero, "croparea_method"] = "nan"

        target = WEIGHTS / f"ir_{adm}_croparea_{crop}"
        target.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target / "weights.parquet", index=False)
        manifest = {
            "id_fields": id_fields,
            "source_key_columns": ["hierid"],
            "weight_names": ["croparea"],
            "normalization": "per_destination",
            "source_version": base_manifest["source_version"],
            "regions_version": base_manifest["regions_version"],
            "extra": {
                "crop": crop,
                "weight_source": str(CSV),
                "weight_source_sha256": csv_sha,
                "built_from": str(base_dir),
                "allocation": "region crop area split across straddled "
                              "targets by the geometric artifact's "
                              "land-area shares",
                "zero_area_source_units": int(merged.loc[zero, "hierid"].nunique()),
                "covered_source_units": int(merged.loc[~zero, "hierid"].nunique()),
                "coverage_note": "zero-area regions carry NaN weight; runs "
                                 "declare on_zero_weight='skip' and targets "
                                 "with no cropland for this crop are absent "
                                 "from outputs, not zero",
            },
        }
        (target / "weights.manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        n_targets = out.dropna(subset=["cropareawt"]).groupby(id_fields).ngroups
        total_targets = out.groupby(id_fields).ngroups
        print(f"{target.name}: {manifest['extra']['covered_source_units']} "
              f"covered sources, {n_targets}/{total_targets} targets with cropland")


def main() -> int:
    csv_sha = hashlib.sha256(CSV.read_bytes()).hexdigest()
    areas = pd.read_csv(CSV, usecols=["hierid"] + CROPS)
    if areas["hierid"].duplicated().any():
        raise ValueError(f"{CSV} has duplicated hierid rows")
    for adm in ADMS:
        build(adm, areas, csv_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
