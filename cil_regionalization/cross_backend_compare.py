"""Cross-backend characterization: local vs BigQuery output.

Both backends produce the canonical `OutputSchema`. This module compares
two such frames; typically a local-backend run on the GPW GeoTIFF
against the s51 BQ deliverable; and reports the per-region weight
agreement distribution against `validation.cross_backend_tolerance`
(default 1e-3).

The comparison is a CHARACTERIZATION, not a regression gate. Expected
small differences:

    - geodesic local (pyproj.Geod, ellipsoid) vs spherical BQ
      (ST_AREA / ST_INTERSECTION on the sphere);
    - planar shapely intersection (local) vs spherical BQ; mostly
      the boundary-cell tail;
    - GPW raster sum (local) vs GPW point-sum (BQ) on the same pixel
      footprint.

What the report surfaces:

    - per-region max|delta wt| distribution (median, p95, max) on
      the both-native subset;
    - count of regions exceeding `tolerance`; the agreement budget,
      not a fail trigger;
    - mass-displaced regions (max-weight cell flipped between
      backends); the real spatial-disagreement signal.

Run instructions for the RCC use the three-line env::

    module load python
    source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


_JOIN_KEYS = ("hierid", "cell_ix", "cell_iy")


@dataclass(frozen=True)
class CrossBackendReport:
    """Per-region characterization of local vs BigQuery weights output."""

    weight: str  # "popwt" / "areawt"
    tolerance: float
    n_regions_total: int
    n_regions_both_native: int
    n_regions_both_nan: int
    n_regions_local_only: int  # in local output, missing in BQ
    n_regions_bq_only: int  # in BQ output, missing in local
    # per-region max|delta wt| on the both-native subset
    delta_median: float
    delta_p95: float
    delta_max: float
    # count of regions whose max|delta wt| exceeds the tolerance
    n_regions_above_tolerance: int
    # regions where the max-weight cell flipped between backends
    mass_displaced_regions: list[str] = field(default_factory=list)
    per_region: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        return (
            f"cross-backend comparison ({self.weight}):\n"
            f"  total regions             : {self.n_regions_total}\n"
            f"  both native               : {self.n_regions_both_native}\n"
            f"  both NaN                  : {self.n_regions_both_nan}\n"
            f"  local-only                : {self.n_regions_local_only}\n"
            f"  bq-only                   : {self.n_regions_bq_only}\n"
            f"  per-region max|delta|     : "
            f"median {self.delta_median:.4g}, "
            f"p95 {self.delta_p95:.4g}, "
            f"max {self.delta_max:.4g}\n"
            f"  exceeds tolerance ({self.tolerance:g}): "
            f"{self.n_regions_above_tolerance} / {self.n_regions_both_native}\n"
            f"  mass-displaced regions    : {len(self.mass_displaced_regions)}"
        )


def compare_outputs(
    local: pd.DataFrame,
    bq: pd.DataFrame,
    *,
    weight: str,
    tolerance: float = 1e-3,
) -> CrossBackendReport:
    """Characterize ``weight`` agreement between local and BQ frames.

    Both inputs must be canonical-schema weights frames (``hierid``,
    ``cell_ix``, ``cell_iy``, ``<weight>``). Join on the integer
    cell index triple; exact, no rounding. Regions present in only
    one frame are counted in `local_only` / `bq_only` and contribute
    neither to the delta distribution nor the mass-displacement
    check.

    ``weight`` is the column name (``"popwt"`` or ``"areawt"``), not
    the bare weight (``"pop"`` / ``"area"``). The frames are the
    canonical OutputSchema directly; do not pre-translate.
    """
    for f, label in ((local, "local"), (bq, "bq")):
        missing = set(_JOIN_KEYS) | {weight}
        missing -= set(f.columns)
        if missing:
            raise ValueError(
                f"compare_outputs: {label} frame missing columns "
                f"{sorted(missing)}"
            )

    cols = list(_JOIN_KEYS) + [weight]
    l = local[cols].rename(columns={weight: "wt_local"})
    b = bq[cols].rename(columns={weight: "wt_bq"})
    joined = l.merge(b, on=list(_JOIN_KEYS), how="outer")

    def _region_class(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "local_native": group["wt_local"].notna().any(),
                "bq_native": group["wt_bq"].notna().any(),
            }
        )

    region_class = joined.groupby("hierid", sort=False).apply(
        _region_class, include_groups=False
    )

    n_total = len(region_class)
    n_both_native = int(
        (region_class["local_native"] & region_class["bq_native"]).sum()
    )
    n_both_nan = int(
        (~region_class["local_native"] & ~region_class["bq_native"]).sum()
    )
    # "local_only" = present in local, absent (NaN-everywhere) in BQ.
    # The frames may not even cover the same hierid set, hence the
    # outer-join semantics.
    n_local_only = int(
        (region_class["local_native"] & ~region_class["bq_native"]).sum()
    )
    n_bq_only = int(
        (~region_class["local_native"] & region_class["bq_native"]).sum()
    )

    both_native_ids = region_class.index[
        region_class["local_native"] & region_class["bq_native"]
    ]
    sub = joined.loc[joined["hierid"].isin(both_native_ids)].copy()
    sub["delta"] = (
        sub["wt_local"].fillna(0.0) - sub["wt_bq"].fillna(0.0)
    ).abs()
    per_region_delta = sub.groupby("hierid", sort=False)["delta"].max()
    if len(per_region_delta) > 0:
        delta_median = float(per_region_delta.median())
        delta_p95 = float(per_region_delta.quantile(0.95))
        delta_max = float(per_region_delta.max())
        n_above_tol = int((per_region_delta > tolerance).sum())
    else:
        delta_median = delta_p95 = delta_max = float("nan")
        n_above_tol = 0

    def _max_cell(group: pd.DataFrame, col: str) -> tuple[int, int]:
        valid = group.loc[group[col].notna()]
        if valid.empty:
            return (-1, -1)
        idx = valid[col].idxmax()
        return (
            int(valid.loc[idx, "cell_ix"]),
            int(valid.loc[idx, "cell_iy"]),
        )

    mass_displaced: list[str] = []
    rows: list[dict] = []
    for hid in both_native_ids:
        g = joined.loc[joined["hierid"] == hid]
        lx, ly = _max_cell(g, "wt_local")
        bx, by = _max_cell(g, "wt_bq")
        displaced = (lx, ly) != (bx, by)
        if displaced:
            mass_displaced.append(str(hid))
        rows.append(
            {
                "hierid": hid,
                "max_delta": float(per_region_delta.loc[hid]),
                "local_max_cell": (lx, ly),
                "bq_max_cell": (bx, by),
                "mass_displaced": displaced,
                "above_tolerance": float(per_region_delta.loc[hid]) > tolerance,
            }
        )

    return CrossBackendReport(
        weight=weight,
        tolerance=tolerance,
        n_regions_total=n_total,
        n_regions_both_native=n_both_native,
        n_regions_both_nan=n_both_nan,
        n_regions_local_only=n_local_only,
        n_regions_bq_only=n_bq_only,
        delta_median=delta_median,
        delta_p95=delta_p95,
        delta_max=delta_max,
        n_regions_above_tolerance=n_above_tol,
        mass_displaced_regions=mass_displaced,
        per_region=pd.DataFrame(rows),
    )


def load_canonical_parquet(path: str | Path) -> pd.DataFrame:
    """Load a canonical-schema weights parquet from local or `gs://`.

    Both backends write this format (`cil_regionalization.io.write_result`);
    this helper exists so script and notebook share one loader.
    """
    spath = str(path)
    if spath.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(spath, "rb") as f:
            return pd.read_parquet(f)
    return pd.read_parquet(spath)
