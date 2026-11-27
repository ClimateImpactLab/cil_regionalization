"""Reads CMIP6 GDPCIR daily zarr stores as per-year arrays for
``derive_daily``, which takes them as it takes BCSD arrays.

The GDPCIR product has one zarr store of daily tasmax or tasmin per
model, scenario and variable. The stores are on the same 0.25-degree
center-registered grid as the CMIP5 BCSD archive, with longitudes in
[-180,180) ascending and a 365-day no-leap calendar. Nothing is
converted. A store whose variable is missing, whose longitudes are
outside [-180,180), or whose coordinates are not strictly ascending
raises, and so does a year without exactly 365 days.
"""

from __future__ import annotations

import numpy as np


def _import_xarray():
    try:
        import xarray
    except ImportError as e:
        raise ImportError(
            "reading GDPCIR zarr stores requires xarray with the zarr "
            "backend; install the [netcdf] extra plus zarr"
        ) from e
    return xarray


def open_gdpcir_daily(store: str, variable: str, **open_kwargs):
    """One GDPCIR daily zarr store as an opened dataset, with values lazy
    until sliced.

    ``store`` is a local path or a URL. ``open_kwargs`` reach a remote
    store, and a STAC asset's own ``xarray:open_kwargs`` carry the
    credential and the chunking. With ``open_kwargs`` the store is
    opened by ``open_dataset``, with ``engine`` zarr unless given.
    Raises if ``variable`` is missing, if a longitude is outside
    [-180,180), or if longitudes or latitudes are not strictly
    ascending.
    """
    # open_zarr rejects an engine argument, so open_kwargs go through
    # open_dataset
    xr = _import_xarray()
    if open_kwargs:
        kwargs = dict(open_kwargs)
        engine = kwargs.pop("engine", "zarr")
        ds = xr.open_dataset(store, engine=engine, **kwargs)
    else:
        ds = xr.open_zarr(store)
    if variable not in ds.data_vars:
        raise ValueError(
            f"{store}: variable {variable!r} not present (has {sorted(ds.data_vars)})"
        )
    lon = np.asarray(ds["lon"].values, dtype="float64")
    lat = np.asarray(ds["lat"].values, dtype="float64")
    if lon.min() < -180.0 or lon.max() >= 180.0:
        raise ValueError(
            f"{store}: longitudes outside [-180,180); got {lon.min()}..{lon.max()}"
        )
    if np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise ValueError(f"{store}: coordinates are not strictly ascending")
    return ds


def gdpcir_year(ds, variable: str, year: int) -> np.ndarray:
    """One calendar year of ``variable`` as a (365, lat, lon) float32
    array in Kelvin. Raises if the year does not hold exactly 365 days,
    which means the store is truncated or not on the no-leap calendar.
    """
    sel = ds[variable].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if sel.sizes["time"] != 365:
        raise ValueError(
            f"year {year}: {sel.sizes['time']} days, expected 365 "
            f"(no-leap); the store is truncated or mis-calendared"
        )
    arr = sel.transpose("time", "lat", "lon").values
    return np.asarray(arr, dtype="float32")


_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_GDPCIR_COLLECTIONS = (
    "cil-gdpcir-cc-by",
    "cil-gdpcir-cc-by-sa",
    "cil-gdpcir-cc0",
)


def gdpcir_stac_assets(
    model: str,
    experiment: str,
    variables,
    collections=None,
) -> dict:
    """The signed store of each of ``variables`` for ``model`` and
    ``experiment`` on the Planetary Computer, as
    ``{variable: (href, open_kwargs)}`` ready for ``open_gdpcir_daily``.

    The GDPCIR ensemble is split across collections by license, and one
    model can appear in more than one. All of them are searched unless
    ``collections`` narrows it. The signatures expire, so a long-running
    job should call this again rather than hold the result for hours.
    Raises if a variable has no asset.
    """
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(_STAC_URL, modifier=planetary_computer.sign_inplace)
    want = set(variables)
    found: dict = {}
    for cname in collections or _GDPCIR_COLLECTIONS:
        for item in cat.get_collection(cname).get_items():
            props = item.properties
            if props.get("cmip6:source_id") != model:
                continue
            if props.get("cmip6:experiment_id") != experiment:
                continue
            for v in want - set(found):
                if v in item.assets:
                    asset = item.assets[v]
                    found[v] = (
                        asset.href,
                        dict(asset.extra_fields["xarray:open_kwargs"]),
                    )
            if want == set(found):
                return found
    missing = sorted(want - set(found))
    raise FileNotFoundError(f"GDPCIR {model}/{experiment}: no asset for {missing}")
