"""prepare_data.py of the France example, run on fake cluster files of
the shapes it reads, writes the layout the notebook reads. The layout
is that of synthetic_data.py, which the notebook is run on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "france"
YEARS = list(range(2010, 2101))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"france_{name}", EXAMPLE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cluster(root: Path) -> dict:
    """Fake versions of every file prepare_data.py reads."""
    import geopandas as gpd
    import rasterio
    import xarray as xr
    from rasterio.transform import from_origin
    from shapely.geometry import box

    # 96 departments in 13 regions, and one overseas department
    cells = [(i, j) for j in range(8) for i in range(12)]
    departments = gpd.GeoDataFrame(
        {
            "INSEE_DEP": [f"{k + 1:02d}" for k in range(96)] + ["974"],
            "INSEE_REG": [f"{11 + k % 13}" for k in range(96)] + ["04"],
            "NOM": [f"D{k}" for k in range(97)],
            "geometry": [
                box(i * 0.1, 45 + j * 0.1, (i + 1) * 0.1, 45.1 + j * 0.1)
                for i, j in cells
            ]
            + [box(55.2, -21.4, 55.8, -20.9)],
        },
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    admin = root / "ADMIN-EXPRESS" / "1_DONNEES_LIVRAISON"
    admin.mkdir(parents=True)
    departments.to_file(admin / "DEPARTEMENT.shp")
    regions = departments.dissolve("INSEE_REG", as_index=False)[
        ["INSEE_REG", "geometry"]
    ]
    regions["NOM"] = regions["INSEE_REG"]
    regions.to_file(admin / "REGION.shp")

    hierids = [f"FRA.{j + 1}.{i + 1}" for j in range(4) for i in range(6)]
    shapes = gpd.GeoDataFrame(
        {
            "hierid": hierids + ["FRA.99.1", "REU.1.1"],
            "geometry": [
                box(i * 0.2, 45 + j * 0.2, (i + 1) * 0.2, 45.2 + j * 0.2)
                for j in range(4)
                for i in range(6)
            ]
            + [box(20, 60, 21, 61), box(55.2, -21.4, 55.8, -20.9)],
        },
        crs="EPSG:4326",
    )
    shapes.to_file(root / "agglomerated-world-new.shp")

    with rasterio.open(
        root / "gpw.tif",
        "w",
        driver="GTiff",
        height=40,
        width=40,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.5, 46.5, 0.05, 0.05),
        nodata=-3.4028230607370965e38,
    ) as out:
        out.write(np.full((40, 40), 10.0, dtype=np.float32), 1)

    population = root / "population.csv"
    rows = [(h, y, 1000.0) for h in hierids + ["REU.1.1"] for y in range(2010, 2101, 5)]
    population.write_text(
        "# a comment line, as in the real file\n"
        + pd.DataFrame(rows, columns=["region", "year", "value"]).to_csv(index=False)
    )

    shuffled = hierids[::-1] + ["REU.1.1"]
    for batch in ("batch0", "batch1"):
        for rcp, gcms in (
            ("rcp45", ["CCSM4"]),
            ("rcp85", ["CCSM4", "surrogate_GFDL-CM3_89"]),
        ):
            for gcm in gcms:
                physical = root / "physical" / batch / rcp / gcm / "low" / "SSP3"
                valued = root / "valued" / batch / rcp / gcm / "low" / "SSP3"
                physical.mkdir(parents=True)
                valued.mkdir(parents=True)
                values = np.arange(len(YEARS) * len(shuffled), dtype=float).reshape(
                    len(YEARS), -1
                )
                for name in (
                    "Agespec_interaction_response-combined-levels.nc4",
                    "Agespec_interaction_response-combined-histclim-levels.nc4",
                ):
                    xr.Dataset(
                        {
                            "rebased": (("year", "region"), values),
                            "regions": (("region",), shuffled),
                        },
                        coords={"year": YEARS},
                    ).to_netcdf(physical / name)
                xr.Dataset(
                    {
                        "total_damages": (("year", "region"), values),
                        "gdp": (("year", "region"), values + 1),
                    },
                    coords={"year": YEARS, "region": shuffled},
                ).to_netcdf(valued / "mortality_damages_IR_batch.nc4")

    smme = root / "smme"
    smme.mkdir()
    for rcp in ("rcp45", "rcp85"):
        (smme / f"{rcp}_SMME_weights.tsv").write_text(
            "quantile\tmodel\tweight\nq0\tccsm4\t1.0\n"
        )
    return {
        "admin": root / "ADMIN-EXPRESS",
        "IMPACT_REGIONS": root / "agglomerated-world-new.shp",
        "RASTER": root / "gpw.tif",
        "POPULATION": population,
        "PHYSICAL": root / "physical",
        "VALUED": root / "valued",
        "SMME": smme,
    }


def _layout(data: Path) -> dict:
    """Each file under ``data`` with its columns and their types."""
    import rasterio

    out = {}
    for path in sorted(data.rglob("*")):
        if not path.is_file() or "published" in path.parts:
            continue
        name = path.relative_to(data).as_posix()
        if "tree/" in name:
            name = "tree/" + path.name
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            columns = {c: str(t) for c, t in frame.dtypes.items()}
            if "geometry" in columns:
                import geopandas as gpd

                columns["crs"] = str(gpd.read_parquet(path).crs)
            out[name] = columns
        elif path.suffix == ".tif":
            with rasterio.open(path) as raster:
                out[name] = {"dtype": raster.dtypes[0], "crs": str(raster.crs)}
        elif path.suffix == ".tsv":
            out[name] = list(pd.read_csv(path, sep="\t").columns)
    return out


def test_it_writes_the_layout_the_notebook_reads(tmp_path, monkeypatch):
    pytest.importorskip("xarray")
    prepare = _module("prepare_data")
    synthetic = _module("synthetic_data")
    cluster = _cluster(tmp_path / "cluster")
    for name, path in cluster.items():
        if name != "admin":
            monkeypatch.setattr(prepare, name, path)
    monkeypatch.setattr(prepare, "DATA", tmp_path / "data")
    # it fetches from Zenodo, which the suite does not
    monkeypatch.setattr(prepare, "write_published", lambda hierids: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data.py",
            "--admin-express",
            str(cluster["admin"]),
            "--admin-express-version",
            "TEST",
        ],
    )
    prepare.main()

    data, _ = synthetic.write(tmp_path / "synthetic")
    assert _layout(tmp_path / "data") == _layout(data)
    impact_regions = pd.read_parquet(tmp_path / "data" / "impact_regions.parquet")
    assert "FRA.99.1" not in set(impact_regions["hierid"])
    assert "REU.1.1" not in set(impact_regions["hierid"])
    departments = pd.read_parquet(tmp_path / "data" / "departments.parquet")
    assert len(departments) == 96 and "974" not in set(departments["INSEE_DEP"])

    # the rows of a leaf are its regions as named, in the order kept
    leaf = pd.read_parquet(
        tmp_path / "data" / "tree" / "batch0" / "rcp85" / "CCSM4" / "deaths.parquet"
    )
    first = leaf[(leaf["hierid"] == "FRA.1.1") & (leaf["year"] == 2020)][
        "deaths"
    ].item()
    shuffled = [f"FRA.{j + 1}.{i + 1}" for j in range(4) for i in range(6)][::-1] + [
        "REU.1.1"
    ]
    assert first == (2020 - 2010) * len(shuffled) + shuffled.index("FRA.1.1")
