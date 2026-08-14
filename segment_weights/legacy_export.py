"""Legacy 13-column CSV exporter.

The legacy ``intersect_zonalstats_par.py`` writes per-region weight files
with this column order:

    area, crop, hierid, pix_cent_x, pix_cent_y, pop,
    areatotal, poptotal, croptotal,
    areawt, popwt, cropwt,
    shpid

Downstream consumers (splines pipeline, indirect impact pipelines) read
``hierid``, ``pix_cent_x`` / ``pix_cent_y``, and the relevant ``*wt``
column. ``shpid`` is a per-shapefile constant in the legacy writer and
is never used as a join key in any reader we have visibility into; the
exporter here fills it with the 0-based dense rank of the hierid sorted
ascending. That is portable across backends and deterministic.

NaN handling: ``cropwt`` is NaN for regions with zero cropland (the
``fallback = "nan"`` policy outcome). The legacy file convention is the
same; the downstream backfills per-pixel with ``areawt`` at
aggregation time.

Env note for the RCC HPC where this exporter is used::

    module load python
    source activate /project/cil/home_dirs/rcc/envs/climate_data_aggregation/
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from segment_weights.backends.base import WeightsResult


logger = logging.getLogger(__name__)


# Frozen, ordered. The legacy reader does not require this exact order
# (CSV is keyed by column name), but consumers downstream of our
# control do rely on the canonical file header so we don't surprise
# anyone with a reorder.
LEGACY_COLUMNS: tuple[str, ...] = (
    "area",
    "crop",
    "hierid",
    "pix_cent_x",
    "pix_cent_y",
    "pop",
    "areatotal",
    "poptotal",
    "croptotal",
    "areawt",
    "popwt",
    "cropwt",
    "shpid",
)

_REQUIRED_WEIGHTS = ("area", "crop", "pop")


def to_legacy_frame(result: WeightsResult) -> pd.DataFrame:
    """Translate the canonical weights frame into the legacy 13-column schema.

    The result must carry ``area``, ``crop``, and ``pop`` weights (the
    legacy schema is a fixed shape; this exporter is not the place for
    runtime weight discovery). The id field must be a single ``hierid``
    column; multi-key region sets are out of scope for the legacy
    format.

    NaN cropwt entries from the ``fallback = "nan"`` policy are
    preserved verbatim. ``croptotal`` for such regions is 0.0 (the sum
    of ``crop_raw`` is genuinely zero); ``cropwt`` is NaN. Both signals
    arrive at the downstream consumer.

    Raises ``ValueError`` naming the missing input if any required
    weight column is absent.
    """
    frame = result.frame
    weight_names = result.schema.weight_names
    missing = [w for w in _REQUIRED_WEIGHTS if w not in weight_names]
    if missing:
        raise ValueError(
            f"legacy_export: missing required weights {missing}; "
            f"the legacy 13-column schema requires {list(_REQUIRED_WEIGHTS)} "
            f"(got {list(weight_names)})"
        )
    id_fields = result.schema.id_fields
    if list(id_fields) != ["hierid"]:
        raise ValueError(
            f"legacy_export: requires id_fields == ['hierid'] "
            f"(got {list(id_fields)})"
        )

    out = pd.DataFrame(
        {
            "area": frame["area_raw"].astype(float),
            "crop": frame["crop_raw"].astype(float),
            "hierid": frame["hierid"].astype(str),
            "pix_cent_x": frame["cell_lon"].astype(float),
            "pix_cent_y": frame["cell_lat"].astype(float),
            "pop": frame["pop_raw"].astype(float),
            "areawt": frame["areawt"].astype(float),
            "popwt": frame["popwt"].astype(float),
            "cropwt": frame["cropwt"].astype(float),
        }
    )
    totals = (
        out.groupby("hierid", sort=False)[["area", "crop", "pop"]]
        .transform("sum")
        .rename(
            columns={"area": "areatotal", "crop": "croptotal", "pop": "poptotal"}
        )
    )
    out["areatotal"] = totals["areatotal"]
    out["croptotal"] = totals["croptotal"]
    out["poptotal"] = totals["poptotal"]
    out["shpid"] = _dense_rank_hierid(out["hierid"])
    return out[list(LEGACY_COLUMNS)]


def _dense_rank_hierid(hierid: pd.Series) -> pd.Series:
    """0-based dense rank of hierid in sorted ascending order.

    Deterministic, backend-agnostic, and survives a reorder of the
    underlying frame. NOT the legacy "per-shapefile constant tag" --
    downstream consumers do not join on shpid, so a stable integer
    per-region is the safer surface.
    """
    unique = pd.Series(sorted(hierid.unique()))
    rank_map = pd.Series(range(len(unique)), index=unique)
    return hierid.map(rank_map).astype("int64")


def write_legacy_csv(
    result: WeightsResult,
    output_path: str | Path,
) -> str:
    """Write ``result`` as a legacy 13-column CSV at ``output_path``.

    Returns the written path. ``gs://`` URIs are supported via gcsfs.
    Parent directories are created automatically for local paths.
    """
    frame = to_legacy_frame(result)
    if str(output_path).startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(str(output_path), "w") as f:
            frame.to_csv(f, index=False)
        return str(output_path)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=False)
    return str(p)
