"""Compares this package's crop weights with the legacy weights CSV.

The two are not expected to match byte for byte. The legacy script
takes 'area' in planar degrees, a known bug, handles rasters by
centroid only in some paths, and falls back to the polylabel 1e-5 hack
where this package marks the weight 'nan'. The comparison takes the
per-region weight shares, which cancel the difference in area units,
and asks whether their spatial distribution across cells agrees,
whether both sides find the same regions without cropland (NaN
``cropwt`` in both), and whether each region's maximum-weight cell is
the same cell. A moved maximum-weight cell is the sign of a real
spatial disagreement. The report describes the differences and is not
a pass or fail check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# Coordinate rounding for the join key. The 0.25deg grid centroids are
# .125, .375, .625, .875; exact under canonical centroid generation,
# but we accept 4-decimal rounding to absorb any float-representation
# drift between the two pipelines.
_COORD_ROUND = 4


@dataclass(frozen=True)
class ComparisonReport:
    """Per-region comparison of our crop weights with the legacy CSV.

    Zero `n_regions_ours_nan_legacy_native` and
    `n_regions_legacy_nan_ours_native` and an empty
    `mass_displaced_regions` mean the two agree in shape, and the
    `delta` fields measure the noise at boundaries.
    """

    weight: str  # "crop" by default
    n_regions_total: int
    n_regions_both_native: int
    n_regions_both_nan: int
    n_regions_ours_nan_legacy_native: int
    n_regions_legacy_nan_ours_native: int
    # per-region max(|delta wt|) on the native-in-both subset
    delta_median: float
    delta_p95: float
    delta_max: float
    # regions where the maximum-weight cell differs across the two
    # pipelines (the real spatial-disagreement signal)
    mass_displaced_regions: list[str] = field(default_factory=list)
    # the per-region records, kept around for downstream inspection
    per_region: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        nan_mismatch = (
            self.n_regions_ours_nan_legacy_native
            + self.n_regions_legacy_nan_ours_native
        )
        return (
            f"legacy comparison ({self.weight}wt):\n"
            f"  total regions          : {self.n_regions_total}\n"
            f"  both native            : {self.n_regions_both_native}\n"
            f"  both NaN (zero-crop)   : {self.n_regions_both_nan}\n"
            f"  NaN mismatches         : {nan_mismatch}\n"
            f"  mass-displaced regions : {len(self.mass_displaced_regions)}\n"
            f"  per-region max|delta {self.weight}wt|\n"
            f"      median : {self.delta_median:.4g}\n"
            f"      p95    : {self.delta_p95:.4g}\n"
            f"      max    : {self.delta_max:.4g}"
        )


def compare_weights(
    ours: pd.DataFrame,
    legacy: pd.DataFrame,
    *,
    weight: str = "crop",
    coord_round: int = _COORD_ROUND,
) -> ComparisonReport:
    """A `ComparisonReport` of our ``<weight>wt`` against the legacy
    ``<weight>wt`` across regions.

    Both frames are in the legacy layout with hierid, pix_cent_x,
    pix_cent_y and ``<weight>wt``, so our canonical frame goes through
    `legacy_export.to_legacy_frame` first. They are joined on
    ``(hierid, pix_cent_x, pix_cent_y)`` with the coordinates rounded to
    `coord_round` decimals. A region with a weight on one side only is
    counted as a NaN mismatch and is left out of the differences and
    the mass-displacement check. Raises if a frame lacks one of the
    columns.
    """
    wt_col = f"{weight}wt"
    for f in (ours, legacy):
        missing = {"hierid", "pix_cent_x", "pix_cent_y", wt_col} - set(f.columns)
        if missing:
            raise ValueError(
                f"compare_weights: frame missing columns {sorted(missing)}"
            )

    keys = ["hierid", "pix_cent_x", "pix_cent_y"]
    o = ours[keys + [wt_col]].copy()
    old = legacy[keys + [wt_col]].copy()
    o["pix_cent_x"] = o["pix_cent_x"].round(coord_round)
    o["pix_cent_y"] = o["pix_cent_y"].round(coord_round)
    old["pix_cent_x"] = old["pix_cent_x"].round(coord_round)
    old["pix_cent_y"] = old["pix_cent_y"].round(coord_round)
    o = o.rename(columns={wt_col: "wt_ours"})
    old = old.rename(columns={wt_col: "wt_legacy"})

    joined = o.merge(old, on=keys, how="outer")

    # Per-region classification: each region is "NaN" in one side iff
    # ALL of its non-NaN cells in the join are NaN there. A region with
    # ANY non-NaN wt is "native" on that side.
    by_region = joined["hierid"]
    region_class = pd.DataFrame(
        {
            "ours_native": joined["wt_ours"]
            .notna()
            .groupby(by_region, sort=False)
            .any(),
            "legacy_native": joined["wt_legacy"]
            .notna()
            .groupby(by_region, sort=False)
            .any(),
        }
    )

    n_total = len(region_class)
    n_both_native = int(
        (region_class["ours_native"] & region_class["legacy_native"]).sum()
    )
    n_both_nan = int(
        (~region_class["ours_native"] & ~region_class["legacy_native"]).sum()
    )
    n_ours_nan_legacy_native = int(
        (~region_class["ours_native"] & region_class["legacy_native"]).sum()
    )
    n_legacy_nan_ours_native = int(
        (region_class["ours_native"] & ~region_class["legacy_native"]).sum()
    )

    # Delta distribution on the both-native subset.
    both_native_ids = region_class.index[
        region_class["ours_native"] & region_class["legacy_native"]
    ]
    sub = joined.loc[joined["hierid"].isin(both_native_ids)].copy()
    sub["delta"] = (sub["wt_ours"].fillna(0.0) - sub["wt_legacy"].fillna(0.0)).abs()
    per_region_delta = sub.groupby("hierid", sort=False)["delta"].max()
    if len(per_region_delta) > 0:
        delta_median = float(per_region_delta.median())
        delta_p95 = float(per_region_delta.quantile(0.95))
        delta_max = float(per_region_delta.max())
    else:
        delta_median = delta_p95 = delta_max = float("nan")

    # Mass displacement: cell carrying max weight in each pipeline.
    def _max_cell(group: pd.DataFrame, col: str) -> tuple[float, float]:
        sub = group.loc[group[col].notna()]
        if sub.empty:
            return (float("nan"), float("nan"))
        idx = sub[col].idxmax()
        return (
            float(sub.loc[idx, "pix_cent_x"]),
            float(sub.loc[idx, "pix_cent_y"]),
        )

    mass_displaced: list[str] = []
    per_region_rows: list[dict] = []
    for hid in both_native_ids:
        g = joined.loc[joined["hierid"] == hid]
        ox, oy = _max_cell(g, "wt_ours")
        lx, ly = _max_cell(g, "wt_legacy")
        same_cell = (ox == lx) and (oy == ly)
        if not same_cell:
            mass_displaced.append(str(hid))
        per_region_rows.append(
            {
                "hierid": hid,
                "max_delta": float(per_region_delta.loc[hid]),
                "ours_max_cell": (ox, oy),
                "legacy_max_cell": (lx, ly),
                "mass_displaced": not same_cell,
            }
        )
    per_region = pd.DataFrame(per_region_rows)

    return ComparisonReport(
        weight=weight,
        n_regions_total=n_total,
        n_regions_both_native=n_both_native,
        n_regions_both_nan=n_both_nan,
        n_regions_ours_nan_legacy_native=n_ours_nan_legacy_native,
        n_regions_legacy_nan_ours_native=n_legacy_nan_ours_native,
        delta_median=delta_median,
        delta_p95=delta_p95,
        delta_max=delta_max,
        mass_displaced_regions=mass_displaced,
        per_region=per_region,
    )


def load_legacy_csv(path: str | Path) -> pd.DataFrame:
    """A legacy weights CSV, read by `pd.read_csv` as it is."""
    # one place for a later legacy format with other column names
    return pd.read_csv(path)
