"""The maps the notebook draws. Nothing here computes anything.

Every map is the Philippines and Germany side by side, on one colour
scale shared by both. The colour bar says what the quantity is; the title
keeps its name.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from data import COUNTRIES, MODEL, YEAR

# column: (colour map, title, colour bar label)
STYLE = {
    "tasmax_mean": ("RdYlBu_r", "Mean daily high", "C"),
    "tasmax_jan": ("RdYlBu_r", "Mean daily high in January", "C"),
    "tasmax_jja": ("RdYlBu_r", "Mean daily high in June, July and August", "C"),
    "tasmax_djf": ("RdYlBu_r", "Mean daily high in December, January and February", "C"),
    "cdd": ("Reds", "Cooling degree days",
            "heat accumulated above 20 C (C x days)"),
    "hdd": ("Blues", "Heating degree days",
            "cold accumulated below 20 C (C x days)"),
    "hot_days": ("YlOrRd", "Days above 35 C",
                 "days with a high above 35 C"),
    "cdd_diff": ("RdBu_r", "Difference in cooling degree days (generated minus segment)",
                 "difference in heat above 20 C (C x days)"),
    "hdd_diff": ("RdBu_r", "Difference in heating degree days (generated minus segment)",
                 "difference in cold below 20 C (C x days)"),
}


def _pair():
    """The Philippines and Germany side by side, sized to their shapes."""
    return plt.subplots(1, 2, figsize=(11, 6),
                        gridspec_kw={"width_ratios": [1, 2]})


def _finish(fig, axes, norm, cmap, label, title):
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 shrink=0.7, label=label)
    fig.suptitle(f"{title}, {MODEL} {YEAR}")


def grid_maps(grids, fields, column, outlines, vmin=0.0):
    """One field on the grid, cell by cell, with province outlines on top.

    grids maps each country to its DailyGrid (for the coordinates), and
    fields maps each country to a dict of (lat, lon) arrays.
    """
    cmap, title, label = STYLE[column]
    vmax = max(np.nanmax(fields[iso][column]) for iso in COUNTRIES)
    if vmin is None:
        vmin = min(np.nanmin(fields[iso][column]) for iso in COUNTRIES)
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1.0))
    fig, axes = _pair()
    for ax, (iso, name) in zip(axes, COUNTRIES.items()):
        g = grids[iso]
        ax.pcolormesh(g.lon, g.lat, fields[iso][column], cmap=cmap,
                      norm=norm, shading="nearest")
        outlines[outlines["ISO"] == iso].boundary.plot(
            ax=ax, color="0.3", linewidth=0.2)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_axis_off()
    _finish(fig, axes, norm, cmap, label, f"{title}, per grid cell")


def region_maps(regions, column, level, symmetric=False):
    """One aggregated column, region by region. Blank regions are grey.

    regions is a GeoDataFrame with an ISO column and the value column.
    With symmetric, the scale is centred on zero, for differences.
    """
    cmap, title, label = STYLE[column]
    if symmetric:
        v = max(float(regions[column].abs().max()), 1.0)
        norm = Normalize(vmin=-v, vmax=v)
    else:
        norm = Normalize(vmin=0.0, vmax=max(float(regions[column].max()), 1.0))
    fig, axes = _pair()
    for ax, (iso, name) in zip(axes, COUNTRIES.items()):
        regions[regions["ISO"] == iso].plot(
            column=column, ax=ax, cmap=cmap, norm=norm, edgecolor="0.6",
            linewidth=0.1, missing_kwds={"color": "0.85"})
        ax.set_title(name)
        ax.set_axis_off()
    _finish(fig, axes, norm, cmap, label,
            f"{title} by {level}, population weighted")
