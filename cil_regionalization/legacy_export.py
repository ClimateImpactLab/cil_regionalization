"""Writes weights in the column layout of the legacy weights CSV.

The legacy ``intersect_zonalstats_par.py`` writes per-region weight
files with the columns area, crop, hierid, pix_cent_x, pix_cent_y, pop,
areatotal, poptotal, croptotal, areawt, popwt, cropwt and shpid, in
that order. Downstream consumers, the splines pipeline and the indirect
impact pipelines, read ``hierid``, ``pix_cent_x`` / ``pix_cent_y`` and
the relevant ``*wt`` column. Here ``shpid`` is the 0-based dense rank
of the hierid in ascending order, not the legacy writer's per-shapefile
constant.

``cropwt`` is NaN for a region with no cropland, the outcome of the
``fallback = "nan"`` policy, as in the legacy files. Downstream
aggregation fills it per pixel from ``areawt``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cil_regionalization.backends.base import WeightsResult


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
    """The canonical weights frame of ``result`` in the legacy column
    layout, with the totals summed from the raw weights per hierid.

    A NaN ``cropwt`` from the ``fallback = "nan"`` policy is kept, and
    such a region's ``croptotal`` is 0.0, the sum of its ``crop_raw``.
    Raises ``ValueError`` naming the missing input if any of the
    ``area``, ``crop`` and ``pop`` weights is absent, and raises if the
    id fields are anything but a single ``hierid``.
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
            f"legacy_export: requires id_fields == ['hierid'] (got {list(id_fields)})"
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
        .rename(columns={"area": "areatotal", "crop": "croptotal", "pop": "poptotal"})
    )
    out["areatotal"] = totals["areatotal"]
    out["croptotal"] = totals["croptotal"]
    out["poptotal"] = totals["poptotal"]
    out["shpid"] = _dense_rank_hierid(out["hierid"])
    return out[list(LEGACY_COLUMNS)]


def _dense_rank_hierid(hierid: pd.Series) -> pd.Series:
    """The 0-based dense rank of each hierid in ascending order, the
    same for any order of the frame and on either backend."""
    # Not the legacy per-shapefile constant. Downstream consumers do not
    # join on shpid, so a stable integer per region is the safer choice.
    unique = pd.Series(sorted(hierid.unique()))
    rank_map = pd.Series(range(len(unique)), index=unique)
    return hierid.map(rank_map).astype("int64")


def write_legacy_csv(
    result: WeightsResult,
    output_path: str | Path,
) -> str:
    """Writes ``result`` as a legacy weights CSV at ``output_path``, a
    local path or a ``gs://`` URI, and returns the path. The parent
    directories of a local path are created."""
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
