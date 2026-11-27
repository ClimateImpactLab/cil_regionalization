"""Nearest-cell rows for regions invisible to the grid.

Some IR regions are small enough that, at a given grid resolution, the
geodesic intersection with every overlapping cell rounds to zero, and
they would otherwise be left out of the output. Each such region gets
one row at the cell that contains its representative point, which
shapely's `representative_point` keeps inside the polygon. The row has
weight 1, raw total 0 and the method label `nearest_cell` for every
weight in the output schema. The rows are added at the end of a
backend's pipeline, after `apply_fallback`, which would otherwise
process them again.
"""

# The legacy pipeline (`refs/create_cil_regionalization.py`) gave such
# regions a 1e-5 dummy weight instead. A near-zero weight pollutes
# normalization and hides the missing coverage, where a marked
# nearest_cell row keeps the accounting explicit and countable.

from __future__ import annotations

from typing import Iterable

import pandas as pd
import shapely

from cil_regionalization.grid import GridSpec
from cil_regionalization.schema import method_column, raw_column, weight_column


NEAREST_CELL = "nearest_cell"


def synthesize_rows(
    region_geoms: dict[tuple, "shapely.Geometry"],
    grid: GridSpec,
    id_fields: list[str],
    weight_names: Iterable[str],
) -> pd.DataFrame:
    """One synthesized row per region, at the cell of ``grid`` that holds
    its representative point, with weight 1, raw total 0 and method
    ``nearest_cell`` for each of ``weight_names``.

    ``region_geoms`` maps id tuples such as ``('ABW',)``, in the order of
    ``id_fields``, to shapely geometries, and empty or missing geometries
    are skipped. The columns are ``id_fields + cell_ix + cell_iy +
    cell_lon + cell_lat`` and ``{name}wt``, ``{name}_raw`` and
    ``{name}_method`` for each weight. Raises if a representative point
    falls outside the grid.
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
    """The id tuples of `requested_ids` that have no row in `final_frame`,
    the regions that need a nearest_cell row."""
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
