"""Nearest-cell synthesis for regions invisible to the grid.

Some IR regions are small enough that, at a given grid resolution, the
geodesic intersection with every overlapping cell rounds to zero; these
regions would otherwise be dropped from the output. The synthesis here
guarantees coverage: one row per such region at the cell that contains
the region's representative point (shapely's `representative_point`,
which is always inside the polygon, unlike a raw centroid).

The synthesized row carries `popwt = areawt = 1.0` and method label
`nearest_cell` for every weight in the output schema. This replaces the
legacy 1e-5 dummy-weight hack from `refs/create_segment_weights.py`.
That approach is deliberately banned here: a magic near-zero weight
pollutes normalization and hides missing coverage, where a marked
`nearest_cell` row keeps the accounting explicit and countable.

Apply at the END of a backend's pipeline so the synthesized rows are not
re-processed by `apply_fallback`.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import shapely

from segment_weights.grid import GridSpec
from segment_weights.schema import method_column, raw_column, weight_column


NEAREST_CELL = "nearest_cell"


def synthesize_rows(
    region_geoms: dict[tuple, "shapely.Geometry"],
    grid: GridSpec,
    id_fields: list[str],
    weight_names: Iterable[str],
) -> pd.DataFrame:
    """One synthesized row per region.

    Parameters
    ----------
    region_geoms : mapping from id-tuple (e.g., ``('ABW',)``) to shapely
        geometry. Empty geometries are skipped.
    grid : the run's GridSpec. Used to bin the representative point.
    id_fields : the run's id-field names in order matching the tuple keys.
    weight_names : every weight column to fill (e.g., ``['pop', 'area']``).

    Returns
    -------
    DataFrame with columns ``id_fields + cell_ix + cell_iy + cell_lon +
    cell_lat + {name}wt + {name}_raw + {name}_method`` for each weight.
    """
    weight_names = list(weight_names)
    rows: list[dict] = []
    for ids_tuple, geom in region_geoms.items():
        if geom is None or geom.is_empty:
            continue
        rep = geom.representative_point()
        ix_arr, iy_arr = grid.index_of(rep.x, rep.y)
        ix = int(ix_arr.item() if hasattr(ix_arr, "item") else ix_arr)
        iy = int(iy_arr.item() if hasattr(iy_arr, "item") else iy_arr)
        if not (0 <= ix < grid.n_ix and 0 <= iy < grid.n_iy):
            raise ValueError(
                f"nearest_cell: representative point for region "
                f"{dict(zip(id_fields, ids_tuple))} falls outside the grid "
                f"(ix={ix}, iy={iy}, point=({rep.x}, {rep.y}))"
            )
        lon_c, lat_c = grid.centroid(ix, iy)
        row: dict = dict(zip(id_fields, ids_tuple))
        row["cell_ix"] = ix
        row["cell_iy"] = iy
        row["cell_lon"] = float(lon_c)
        row["cell_lat"] = float(lat_c)
        for name in weight_names:
            row[weight_column(name)] = 1.0
            row[raw_column(name)] = 0.0
            row[method_column(name)] = NEAREST_CELL
        rows.append(row)

    if not rows:
        cols = (
            list(id_fields)
            + ["cell_ix", "cell_iy", "cell_lon", "cell_lat"]
            + [
                col
                for name in weight_names
                for col in (
                    weight_column(name),
                    raw_column(name),
                    method_column(name),
                )
            ]
        )
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def find_missing_regions(
    final_frame: pd.DataFrame,
    requested_ids: Iterable[tuple],
    id_fields: list[str],
) -> set[tuple]:
    """Return the id-tuples present in `requested_ids` but absent from
    `final_frame`. Used by both backends to decide which regions need
    nearest_cell synthesis after the normal pipeline runs.
    """
    if len(final_frame) == 0:
        present: set[tuple] = set()
    else:
        present = {
            tuple(row)
            for row in final_frame[id_fields]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
    return {t for t in requested_ids if t not in present}
