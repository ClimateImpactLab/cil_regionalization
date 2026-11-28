"""Writes a small synthetic stand-in for data/, laid out as
prepare_data.py lays out the real one, and a synthetic temperature
table beside it with the warming level settings over it. The test suite
runs the notebook on it, and so can a reader without the cluster.

    python examples/france/synthetic_data.py <directory>
    FRANCE_DATA=<directory>/data \\
        FRANCE_WARMING=<directory>/warming_levels.toml \\
        python examples/france/france.py

Nothing in it is real. Twenty four impact regions cover two regions
of two departments each, one more lies where the population raster
has no people, and four climate models in six batches stand for thirty three
in two.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# first, since importing it points PROJ at this environment's data
import cil_regionalization as cilreg
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

# more batches than the real sample, so that a 5th percentile has more
# than twenty draws under it
BATCHES = [f"batch{i}" for i in range(6)]
# the models that join a warming level, and a surrogate that does not
MODELS = ["CCSM4", "GFDL-CM3", "MIROC5"]
GCMS = [*MODELS, "surrogate_GFDL-CM3_89"]
# the settings the synthetic ones take theirs from
AR5 = Path(__file__).resolve().parents[1] / "warming_levels" / "ar5_cmip5_smme.toml"
# as in the real tree, one surrogate exists under rcp85 alone
MISSING = {("rcp45", "surrogate_GFDL-CM3_89")}
YEARS = range(2020, 2100)

# Each member's temperature over the reference period, mid century and
# end of century, before rebasing. With an offset of 0.61 they land on
# the levels in the comment.
GMST = {
    ("CCSM4", "rcp45"): (14.0, 15.39, 15.89),  # 2.0, 2.5 (no level)
    ("GFDL-CM3", "rcp45"): (13.5, 14.39, 15.89),  # 1.5, 3.0
    ("MIROC5", "rcp45"): (14.2, 15.09, 15.59),  # 1.5, 2.0
    ("CCSM4", "rcp85"): (14.0, 15.39, 17.39),  # 2.0, 4.0
    ("GFDL-CM3", "rcp85"): (13.5, 15.89, 17.89),  # 3.0, 5.0
    ("MIROC5", "rcp85"): (14.2, 15.59, 17.59),  # 2.0, 4.0
    ("surrogate_GFDL-CM3_89", "rcp85"): (13.8, 15.69, 17.19),  # 2.5, 4.0
}


def _write_geometries(data: Path) -> gpd.GeoDataFrame:
    crs = "EPSG:4326"
    departments = gpd.GeoDataFrame(
        {
            "INSEE_DEP": ["S1", "S2", "S3", "S4"],
            "INSEE_REG": ["R1", "R2", "R1", "R2"],
            "NOM": ["Sud-Ouest", "Sud-Est", "Nord-Ouest", "Nord-Est"],
            "geometry": [
                box(0.0, 45.0, 0.6, 45.45),
                box(0.6, 45.0, 1.2, 45.45),
                box(0.0, 45.45, 0.6, 45.9),
                box(0.6, 45.45, 1.2, 45.9),
            ],
        },
        crs=crs,
    )
    departments.to_parquet(data / "departments.parquet")
    regions = departments.dissolve("INSEE_REG", as_index=False)[
        ["INSEE_REG", "geometry"]
    ]
    regions.insert(1, "NOM", ["Ouest", "Est"])
    regions.to_parquet(data / "regions.parquet")

    # boundaries that cross the departments', as real ones do
    xs = [0.0, 0.2, 0.4, 0.55, 0.7, 0.9, 1.1]
    ys = [45.0, 45.2, 45.4, 45.65, 45.9]
    cells = [
        (f"FRA.{j + 1}.{i + 1}", box(xs[i], ys[j], xs[i + 1], ys[j + 1]))
        for j in range(len(ys) - 1)
        for i in range(len(xs) - 1)
    ]
    # where the raster has no people, so its weight falls back to area
    cells.append(("FRA.9.1", box(1.1, 45.0, 1.2, 45.9)))
    regions_ir = gpd.GeoDataFrame(
        {"hierid": [h for h, _ in cells], "geometry": [g for _, g in cells]},
        crs=crs,
    )
    regions_ir.to_parquet(data / "impact_regions.parquet")
    return regions_ir


def _write_raster(data: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    nodata = np.float32(-3.4028230607370965e38)
    size = 0.05
    rows, cols = 18, 24
    lat = 45.9 - size * (np.arange(rows) + 0.5)
    lon = size * (np.arange(cols) + 0.5)
    people = 50 + 400 * np.exp(
        -((lon[None, :] - 0.3) ** 2 + (lat[:, None] - 45.6) ** 2) / 0.05
    )
    people = people.astype(np.float32)
    # the sea, wider than the impact region that lies in it
    people[:, lon > 1.05] = nodata
    with rasterio.open(
        data / "population_2015_30ss.tif",
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 45.9, size, size),
        nodata=nodata,
    ) as out:
        out.write(people, 1)


def _write_tree(data: Path, hierids: list[str]) -> None:
    rng = np.random.default_rng(10)
    years = np.array(list(YEARS))
    population = pd.DataFrame(
        [
            (h, y, 20_000 * (1 + i) * (1 + 0.002 * (y - 2020)))
            for i, h in enumerate(hierids)
            for y in years
        ],
        columns=["hierid", "year", "pop"],
    )
    population.to_parquet(data / "ir_population.parquet", index=False)
    pop = population.set_index(["hierid", "year"])["pop"]

    for b, batch in enumerate(BATCHES):
        for r, rcp in enumerate(["rcp45", "rcp85"]):
            for g, gcm in enumerate(GCMS):
                if (rcp, gcm) in MISSING:
                    continue
                leaf = data / "tree" / batch / rcp / gcm
                leaf.mkdir(parents=True, exist_ok=True)
                index = pd.MultiIndex.from_product(
                    [hierids, years], names=["hierid", "year"]
                )
                frame = index.to_frame(index=False)
                warming = (1 + r) * (1 + 0.3 * g) * (frame["year"] - 2010) / 90
                rate = 5 * warming + rng.normal(0, 1 + 0.5 * b, len(frame))
                history = rng.normal(0, 0.5, len(frame))
                people = pop.loc[list(zip(frame["hierid"], frame["year"]))].to_numpy()
                deaths = (rate + history) * people / 1e5
                frame.assign(deaths=deaths.astype(np.float32)).to_parquet(
                    leaf / "deaths.parquet", index=False
                )
                frame.assign(
                    deaths=(history * people / 1e5).astype(np.float32)
                ).to_parquet(leaf / "deaths-histclim.parquet", index=False)
                gdp = people * 40_000 * (1.02 ** (frame["year"] - 2020))
                damages = rate * people / 1e5 * 1.0e7 * (1.01 ** (frame["year"] - 2020))
                frame.assign(
                    damages=damages.astype(np.float32), gdp=gdp.astype(np.float32)
                ).to_parquet(leaf / "damages.parquet", index=False)


def _write_smme(data: Path) -> None:
    (data / "smme").mkdir(parents=True, exist_ok=True)
    rows = {
        # spelt as the published tables spell these rows
        "rcp85": [
            ("ccsm4", 0.3),
            ("gfdl-cm3*", 0.2),
            ("miroc5*", 0.4),
            ("gfdl-cm3_89", 0.1),
        ],
        "rcp45": [("ccsm4", 0.3), ("gfdl-cm3", 0.3), ("miroc5*", 0.4)],
    }
    for rcp, table in rows.items():
        pd.DataFrame(
            {
                "quantile": [f"q{i}" for i in range(len(table))],
                "model": [m for m, _ in table],
                "weight": [w for _, w in table],
            }
        ).to_csv(data / "smme" / f"{rcp}_SMME_weights.tsv", sep="\t", index=False)


def _write_published(data: Path, work: Path) -> None:
    """A stand-in for the France slice of the published GADM 2.0 ADM2
    weights, built on departments whose borders sit a little off ours."""
    gadm = gpd.read_parquet(data / "departments.parquet")
    # the same outline, with the inner borders a little to the north east
    gadm["geometry"] = [
        box(0.0, 45.0, 0.62, 45.47),
        box(0.62, 45.0, 1.2, 45.47),
        box(0.0, 45.47, 0.62, 45.9),
        box(0.62, 45.47, 1.2, 45.9),
    ]
    gadm.insert(0, "ISO", "FRA")
    gadm.insert(1, "ID_1", ["1", "2", "1", "2"])
    gadm.insert(2, "ID_2", ["1", "2", "3", "4"])
    gadm[["ISO", "ID_1", "ID_2", "geometry"]].to_parquet(work / "gadm.parquet")
    config = work / "published.toml"
    config.write_text(
        f"""normalization = "per_source"

[project]
name = "synthetic_gadm20_adm2"

[regions]
path = "{work / "gadm.parquet"}"
id_fields = ["ISO", "ID_1", "ID_2"]
version = "gadm-2.0-impactmap-copy-2025"
on_invalid_geometry = "repair"

[source]
path = "{data / "impact_regions.parquet"}"
id_fields = ["hierid"]
version = "world-combo-201710"
on_invalid_geometry = "repair"

[[weights]]
name = "pop"
raster = "{data / "population_2015_30ss.tif"}"
fallback = "area"
kind = "point_sum"

[backend]
kind = "local"
coverage = "exact_fraction"

[output]
dir = "{data / "published" / "gadm20-adm2-per-source"}"
"""
    )
    cilreg.generate_weights(config)


def _write_gmst(path: Path) -> None:
    rows = []
    for (gcm, rcp), (reference, mid, late) in GMST.items():
        for year in range(1981, 2101):
            if 1986 <= year <= 2005:
                value = reference
            elif 2040 <= year <= 2059:
                value = mid
            elif 2080 <= year <= 2099:
                value = late
            else:
                value = reference + 0.01 * (year - 1981)
            rows.append((gcm, rcp, year, value))
    pd.DataFrame(rows, columns=["gcm", "rcp", "year", "temp"]).to_csv(path, index=False)


def _write_warming_levels(path: Path, gmst: Path) -> None:
    """The settings of ``AR5`` over the synthetic table, which only
    ``MODELS`` join."""
    from cil_regionalization.pipelines.montecarlo import load_warming_levels

    ar5 = load_warming_levels(AR5)
    lines = [
        "[stats.warming_levels]",
        f"gmst = {json.dumps(gmst.name)}",
        f"member = {json.dumps(ar5.member)}",
        f"time_col = {json.dumps(ar5.time_col)}",
        f"value_col = {json.dumps(ar5.value_col)}",
        'smoothing = "synthetic, not smoothed"',
        f"reference_period = {json.dumps(list(ar5.reference_period))}",
        f"offset = {ar5.offset!r}",
        f"levels = {json.dumps(ar5.levels)}",
        f"half_width = {ar5.half_width!r}",
        "",
        "[stats.warming_levels.join]",
        f"gcm = {json.dumps(MODELS)}",
    ]
    for period in ar5.periods:
        lines += [
            "",
            "[[stats.warming_levels.periods]]",
            f"gmst_span = {json.dumps(list(period.gmst_span))}",
            f"impact_span = {json.dumps(list(period.impact_span))}",
        ]
    path.write_text("\n".join(lines) + "\n")


def write(directory: str | Path) -> tuple[Path, Path]:
    """Write ``directory/data``, ``directory/gmst.csv`` and
    ``directory/warming_levels.toml``, and return the data directory and
    the settings."""
    directory = Path(directory)
    data = directory / "data"
    work = directory / "work"
    for path in (data / "published", work):
        path.mkdir(parents=True, exist_ok=True)
    regions_ir = _write_geometries(data)
    _write_raster(data)
    _write_tree(data, list(regions_ir["hierid"]))
    _write_smme(data)
    _write_published(data, work)
    _write_gmst(directory / "gmst.csv")
    _write_warming_levels(directory / "warming_levels.toml", directory / "gmst.csv")
    return data, directory / "warming_levels.toml"


if __name__ == "__main__":
    data, warming = write(sys.argv[1])
    print(f"wrote {data} and {warming}")
