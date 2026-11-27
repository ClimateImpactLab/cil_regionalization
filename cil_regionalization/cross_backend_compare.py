"""Compares the canonical `OutputSchema` weights of the local backend
with those of BigQuery, such as a local run on the GPW GeoTIFF against
the s51 BigQuery deliverable.

The report describes the differences and is not a pass or fail check.
It gives each region's largest absolute weight difference, with the
median, 95th percentile and maximum of those over the regions native to
both, and the regions above ``tolerance``. It also lists the
mass-displaced regions, whose maximum-weight cell differs between the
backends, which is the sign of a real spatial disagreement. The default
``tolerance`` of 1e-3 is that of `validation.cross_backend_tolerance`.

Small differences are expected. Areas are geodesic on the ellipsoid
(pyproj.Geod) locally and spherical in BigQuery (ST_AREA and
ST_INTERSECTION). The local intersection is planar shapely, which
differs mostly in boundary cells. Population is a GPW raster sum
locally and a GPW point sum in BigQuery, on the same pixel footprint.

On the RCC, run it in this environment::

    module load python
    source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    """A `CrossBackendReport` of the agreement in the column ``weight``
    between a local and a BigQuery weights frame.

    Both are canonical OutputSchema frames as written, with
    ``hierid``, ``cell_ix``, ``cell_iy`` and ``<weight>``, joined
    exactly on the first three. ``weight`` is the column name
    (``"popwt"`` or ``"areawt"``), not the bare weight (``"pop"`` /
    ``"area"``). A region is native to a side when any of its cells has
    a weight there. A region native to one side only is counted in
    `n_regions_local_only` or `n_regions_bq_only` and is left out of the
    differences and the mass-displacement check. Within a region native
    to both, a cell missing on one side counts as weight 0. Raises if a
    frame lacks one of the columns.
    """
    for f, label in ((local, "local"), (bq, "bq")):
        missing = set(_JOIN_KEYS) | {weight}
        missing -= set(f.columns)
        if missing:
            raise ValueError(
                f"compare_outputs: {label} frame missing columns {sorted(missing)}"
            )

    cols = list(_JOIN_KEYS) + [weight]
    on_local = local[cols].rename(columns={weight: "wt_local"})
    on_bq = bq[cols].rename(columns={weight: "wt_bq"})
    joined = on_local.merge(on_bq, on=list(_JOIN_KEYS), how="outer")

    # a region is native on a side if any of its cells has a weight there
    by_region = joined["hierid"]
    region_class = pd.DataFrame(
        {
            "local_native": joined["wt_local"]
            .notna()
            .groupby(by_region, sort=False)
            .any(),
            "bq_native": joined["wt_bq"].notna().groupby(by_region, sort=False).any(),
        }
    )

    n_total = len(region_class)
    n_both_native = int(
        (region_class["local_native"] & region_class["bq_native"]).sum()
    )
    n_both_nan = int((~region_class["local_native"] & ~region_class["bq_native"]).sum())
    # "local_only" = present in local, absent (NaN-everywhere) in BQ.
    # The frames may not even cover the same hierid set, hence the
    # outer-join semantics.
    n_local_only = int(
        (region_class["local_native"] & ~region_class["bq_native"]).sum()
    )
    n_bq_only = int((~region_class["local_native"] & region_class["bq_native"]).sum())

    both_native_ids = region_class.index[
        region_class["local_native"] & region_class["bq_native"]
    ]
    sub = joined.loc[joined["hierid"].isin(both_native_ids)].copy()
    sub["delta"] = (sub["wt_local"].fillna(0.0) - sub["wt_bq"].fillna(0.0)).abs()
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
    """A canonical weights parquet, as both backends write it with
    `cil_regionalization.io.write_result`, read from a local path or a
    `gs://` URI."""
    spath = str(path)
    if spath.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(spath, "rb") as f:
            return pd.read_parquet(f)
    return pd.read_parquet(spath)
