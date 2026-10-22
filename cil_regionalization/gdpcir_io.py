"""CMIP6 GDPCIR daily reader: local zarr mirror to per-year arrays.

The GDPCIR product ships daily tasmax and tasmin as one zarr store per
model, scenario and variable, on the same 0.25-degree center-registered
grid as the CMIP5 BCSD archive, longitudes already in [-180,180)
ascending, 365-day no-leap calendar. Unlike the BCSD reader this one
has no conventions to reconcile; its job is to refuse anything that
does not match the stated contract, loudly, per store:

- longitude in [-180,180), strictly ascending, and latitude ascending;
- exactly 365 days per requested year;
- requested variable present.

``derive_daily`` then consumes the year arrays exactly as it does for
BCSD; nothing downstream knows which generation fed it.
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
    """Open one GDPCIR daily zarr store and verify its grid contract.

    Returns the opened dataset; values stay lazy until sliced.

    ``store`` is a local path or a URL. Pass ``open_kwargs`` to reach a
    remote store: a STAC asset's own ``xarray:open_kwargs`` carry the
    credential and the chunking, and ``engine`` is routed through
    ``open_dataset`` because ``open_zarr`` rejects it.
    """
    xr = _import_xarray()
    if open_kwargs:
        kwargs = dict(open_kwargs)
        engine = kwargs.pop("engine", "zarr")
        ds = xr.open_dataset(store, engine=engine, **kwargs)
    else:
        ds = xr.open_zarr(store)
    if variable not in ds.data_vars:
        raise ValueError(
            f"{store}: variable {variable!r} not present "
            f"(has {sorted(ds.data_vars)})"
        )
    lon = np.asarray(ds["lon"].values, dtype="float64")
    lat = np.asarray(ds["lat"].values, dtype="float64")
    if lon.min() < -180.0 or lon.max() >= 180.0:
        raise ValueError(
            f"{store}: longitudes outside [-180,180); got "
            f"{lon.min()}..{lon.max()}"
        )
    if np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise ValueError(f"{store}: coordinates are not strictly ascending")
    return ds


def gdpcir_year(ds, variable: str, year: int) -> np.ndarray:
    """One calendar year as a (365, lat, lon) float32 array in Kelvin.

    Refuses any year that does not hold exactly 365 days: the GDPCIR
    calendar is no-leap, and a different count means a truncated or
    mis-calendared store, not something to average around.
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
    """Resolve GDPCIR assets on the Planetary Computer to signed stores.

    Returns ``{variable: (href, open_kwargs)}`` ready for
    ``open_gdpcir_daily``. The GDPCIR ensemble is split across three
    collections by license, and one model can appear in more than one,
    so all three are searched unless ``collections`` narrows it.

    Signatures expire. Call this again for a long-running job rather
    than holding its result for hours.
    """
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(
        _STAC_URL, modifier=planetary_computer.sign_inplace
    )
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
    raise FileNotFoundError(
        f"GDPCIR {model}/{experiment}: no asset for {missing}"
    )
