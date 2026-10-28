"""Shared computation for the degree-day climate example.

The input is one year of raw NASA BCSD daily tasmax and tasmin for
MRI-CGCM3, 2020, cropped to two countries with opposite climates:
Indonesia and Germany. The model's calendar is Gregorian, so each
file holds 366 days; read_bcsd_daily decodes the calendar and drops
Feb 29 by date, returning the 365-day no-leap year the lab's BCSD
products use.

The transform is heating and cooling degree days at base 20 C on the
midpoint of the daily extremes, summed over the year, per grid cell.
They are then aggregated to GADM 2.0 ADM1 units with the grid025
per-destination weights, as a weighted mean of an intensive
variable, under population and under area weighting.

The ADM1 weights are read twice: once as a published artifact with
its manifest, once from a plain CSV through WeightsArtifact.from_table.
The two routes must give the same numbers.

The impact region level uses an outside table read with from_table:
one row per grid cell and impact region, shares already normalized
per region, provenance not recorded in the file.

The result checks itself. Indonesia is hot all year: cooling degree
days near 2,500, about seven degrees above the base times 365, and
heating degree days near zero. Germany is the reverse: heating
degree days near 3,700, about ten degrees below the base times 365,
and cooling degree days near zero.

    python compute.py
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

import cil_regionalization as cilreg
from cil_regionalization.bcsd_io import DailyGrid, read_bcsd_daily

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = "MRI-CGCM3"
YEAR = 2020
BASE_C = 20.0
GRID = "grid-0.25deg-center-[-180,180)"
RES = 0.25
KEYS = ["ISO", "ID_1"]
COUNTRIES = {"IDN": "Indonesia", "DEU": "Germany"}
# what each country's numbers should look like before computing them
EXPECTED = {
    "IDN": {"hdd": "about 0", "cdd": "~2,500",
            "why": "about 7 C above the base x 365"},
    "DEU": {"hdd": "~3,700", "cdd": "about 0",
            "why": "about 10 C below the base x 365"},
}
WEIGHTS = DATA / "weights" / "grid025_gadm20_adm1_per_destination"
WEIGHTS_CSV = DATA / "adm1_weights.csv"
PLOT_LAYER = DATA / "idn_deu_adm1_plot.parquet"
IR_TABLE = DATA / "idn_deu_ir_segment_weights.parquet"

# Grid weights always have cells that carry no weight for the chosen
# weighting: a cell can intersect a unit and hold no people. Such a
# cell contributes nothing to a weighted mean, so it is safe to skip;
# apply_weights errors on it by default so the choice is explicit.
POLICIES = cilreg.SourceUnitPolicies(on_zero_weight="skip")


def daily_path(var: str, iso: str) -> Path:
    return DATA / f"{var}_day_BCSD_rcp85_r1i1p1_{MODEL}_{YEAR}_{iso.lower()}.nc"


def days_in_file(var: str, iso: str) -> int:
    with netCDF4.Dataset(daily_path(var, iso)) as ds:
        return len(ds.dimensions["time"])


def read_year(iso: str) -> tuple[DailyGrid, DailyGrid]:
    """Daily tasmax and tasmin, 365 days each, Feb 29 dropped by date."""
    tasmax = read_bcsd_daily(str(daily_path("tasmax", iso)), "tasmax",
                             require_full_year=True)
    tasmin = read_bcsd_daily(str(daily_path("tasmin", iso)), "tasmin",
                             require_full_year=True)
    assert np.array_equal(tasmax.dates, tasmin.dates)
    return tasmax, tasmin


def degree_days(tasmax: np.ndarray, tasmin: np.ndarray,
                base: float = BASE_C) -> tuple[np.ndarray, np.ndarray]:
    """Annual heating and cooling degree days per grid cell, in C days.

    The transform is the caller's to write; the library reads and
    aggregates. Counting happens per cell per day, before any spatial
    step: degree days of an averaged temperature are a different and
    smaller number, because max(T - base, 0) is not linear in T.
    """
    hdd = np.zeros(tasmax.shape[1:])
    cdd = np.zeros(tasmax.shape[1:])
    for day_max, day_min in zip(tasmax, tasmin):
        tas = (day_max.astype("float64") + day_min) / 2.0 - 273.15
        cdd += np.maximum(tas - base, 0.0)
        hdd += np.maximum(base - tas, 0.0)
    return hdd, cdd


def cell_frame(grid: DailyGrid, fields: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long frame of per-cell fields keyed by integer grid cell.

    Cell indices follow the grid identity in GRID: cell_ix counts 0.25
    degree columns east from -180, cell_iy rows north from -90. Masked
    cells are dropped.
    """
    iy = np.floor((grid.lat + 90.0) / RES).astype("int64")
    ix = np.floor((grid.lon + 180.0) / RES).astype("int64")
    frame = pd.DataFrame({
        "cell_ix": np.tile(ix, len(iy)),
        "cell_iy": np.repeat(iy, len(ix)),
        **{name: arr.reshape(-1) for name, arr in fields.items()},
    })
    return frame[np.isfinite(frame[list(fields)]).all(axis=1)]


def gridded(iso: str):
    """One country's grid and its per-cell degree days."""
    tasmax, tasmin = read_year(iso)
    hdd, cdd = degree_days(tasmax.values, tasmin.values)
    return tasmax, hdd, cdd


def all_cells() -> pd.DataFrame:
    """Per-cell degree days for every country in the extract."""
    frames = []
    for iso in COUNTRIES:
        grid, hdd, cdd = gridded(iso)
        frames.append(cell_frame(grid, {"hdd": hdd, "cdd": cdd}))
    return pd.concat(frames, ignore_index=True)


def load_weights() -> "cilreg.WeightsArtifact":
    """The committed artifact, loaded the usual way, from its manifest."""
    return cilreg.WeightsArtifact.load(WEIGHTS)


def load_weights_csv() -> "cilreg.WeightsArtifact":
    """The same weights from a plain CSV with centroids and no manifest."""
    return cilreg.WeightsArtifact.from_table(
        WEIGHTS_CSV,
        id_fields=KEYS,
        weights={"pop": "popwt", "area": "areawt"},
        source=GRID,
    )


def load_ir_weights() -> "cilreg.WeightsArtifact":
    """The outside impact region table, keyed by hierid."""
    return cilreg.WeightsArtifact.from_table(
        IR_TABLE,
        id_fields=["hierid"],
        weights={"pop": "popwt", "area": "areawt"},
        source=GRID,
    )


def ir_locations(artifact) -> pd.DataFrame:
    """A point per impact region: the area-weighted mean of its cell
    centroids, from the table itself, for plotting without a shapefile."""
    f = artifact.frame.dropna(subset=["areawt"])
    loc = f.assign(x=f["cell_lon"] * f["areawt"], y=f["cell_lat"] * f["areawt"])
    loc = loc.groupby("hierid")[["x", "y", "areawt"]].sum()
    return pd.DataFrame({
        "hierid": loc.index,
        "lon": (loc["x"] / loc["areawt"]).to_numpy(),
        "lat": (loc["y"] / loc["areawt"]).to_numpy(),
        "ISO": loc.index.str[:3],
    }).reset_index(drop=True)


def aggregate(artifact, cells: pd.DataFrame, weight: str) -> pd.DataFrame:
    """Weighted mean of each per-cell field over each unit."""
    keys = artifact.frame[["cell_ix", "cell_iy"]].drop_duplicates()
    cells = cells.merge(keys, on=["cell_ix", "cell_iy"])
    fields = [c for c in cells.columns if c not in ("cell_ix", "cell_iy")]
    out = None
    for col in fields:
        r = cilreg.apply_weights(
            artifact, cells[["cell_ix", "cell_iy", col]],
            kind="intensive", weight=weight, value_col=col,
            data_version=GRID, policies=POLICIES,
        ).frame
        out = r if out is None else out.merge(r, on=artifact.schema.id_fields)
    return out


def national(artifact, units: pd.DataFrame, weight: str) -> pd.DataFrame:
    """Country values: unit values weighted by each unit's raw mass."""
    mass = (artifact.frame.groupby(KEYS)[f"{weight}_raw"].sum()
            .rename("mass").reset_index())
    m = units.merge(mass, on=KEYS, validate="one_to_one")
    cols = [c for c in units.columns if c not in KEYS]
    weighted = m[cols].multiply(m["mass"], axis=0).assign(
        ISO=m["ISO"], mass=m["mass"])
    sums = weighted.groupby("ISO")[cols + ["mass"]].sum()
    return sums[cols].divide(sums["mass"], axis=0)


def routes_agree(artifact, from_csv, cells: pd.DataFrame,
                 weight: str) -> tuple[int, float]:
    """Aggregate with both artifacts; return unit count and largest gap."""
    a = aggregate(artifact, cells, weight)
    # from_table reads ids from a CSV as strings; compare them as numbers
    b = aggregate(from_csv, cells, weight).astype({"ID_1": "int64"})
    both = a.merge(b, on=KEYS, suffixes=("", "_csv"), validate="one_to_one")
    assert len(both) == len(a)
    diff = max(float((both[c] - both[f"{c}_csv"]).abs().max())
               for c in ("hdd", "cdd"))
    assert diff < 1e-9
    return len(both), diff


def main() -> int:
    for iso, name in COUNTRIES.items():
        grid, _ = read_year(iso)
        leap = np.datetime64(f"{YEAR}-02-29") in grid.dates
        print(f"{name}: the file holds {days_in_file('tasmax', iso)} days; "
              f"read_bcsd_daily returns {grid.values.shape[0]}, "
              f"Feb 29 present: {leap}")

    cells = all_cells()
    weights = load_weights()
    pop = aggregate(weights, cells, "pop")
    area = aggregate(weights, cells, "area")
    n_pop = national(weights, pop, "pop")
    n_area = national(weights, area, "area")

    print(f"\ndegree days per year, base {BASE_C:.0f} C, {MODEL} {YEAR}:")
    print(f"  {'':32s}{'heating':>9s}{'cooling':>9s}")
    for iso, name in COUNTRIES.items():
        e = EXPECTED[iso]
        print(f"  {name + ', expected':32s}{e['hdd']:>9s}{e['cdd']:>9s}"
              f"   ({e['why']})")
        for label, n in (("population weighted", n_pop),
                         ("area weighted", n_area)):
            print(f"  {name + ', ' + label:32s}"
                  f"{n.loc[iso, 'hdd']:9,.0f}{n.loc[iso, 'cdd']:9,.0f}")

    side = pop.merge(area, on=KEYS, suffixes=("_pop", "_area"))
    for col in ("hdd", "cdd"):
        side[f"{col}_diff"] = side[f"{col}_pop"] - side[f"{col}_area"]
    names = pd.read_parquet(PLOT_LAYER, columns=["ISO", "ID_1", "NAME_1"])
    side = side.merge(names, on=KEYS)
    for iso, name in COUNTRIES.items():
        main_col = "cdd" if iso == "IDN" else "hdd"
        s = side[side["ISO"] == iso]
        s = s.reindex(s[f"{main_col}_diff"].abs()
                      .sort_values(ascending=False).index)
        label = "cooling" if main_col == "cdd" else "heating"
        print(f"\n{name}, {label} degree days, population minus area "
              f"weighting, largest gaps:")
        print(s.head(4)[["NAME_1", f"{main_col}_pop", f"{main_col}_area",
                         f"{main_col}_diff"]].round(0).to_string(index=False))

    from_csv = load_weights_csv()
    print()
    for label, w in (("population", "pop"), ("area", "area")):
        n, diff = routes_agree(weights, from_csv, cells, w)
        print(f"artifact and from_table routes, {label} weights: "
              f"{n} units, largest difference {diff:.1e}")

    ir = aggregate(load_ir_weights(), cells, "pop")
    ir["ISO"] = ir["hierid"].str[:3]
    print("\nimpact regions, population weighted, outside table:")
    for iso, name in COUNTRIES.items():
        s = ir[ir["ISO"] == iso]
        print(f"  {name}: {len(s)} regions; heating median "
              f"{s['hdd'].median():,.0f}, cooling median {s['cdd'].median():,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
