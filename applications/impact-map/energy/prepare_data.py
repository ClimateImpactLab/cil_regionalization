"""Rebuild the committed data extract from the energy delta-method tree.

Needs access to the Climate Impact Lab filesystem; everyone else reads
the committed output under data/. The source is the restored 2020
delta-method tree for other_energy: gradients of the impact with
respect to the 80 regression coefficients, with the coefficient
covariance matrix embedded in every file.

The extract holds, for year 2099, rcp85, the high income scenario,
SSP3, all 33 climate models: the effect-of-climate-change gradient
(full adaptation minus histclim) per model for the 3,151 impact
regions that overlap United States states, plus one real file pair
(CCSM4) so the notebook can show the subtraction and the variance
check on unmodified inputs. The covariance matrix, the coefficient
vector from the release's csvv, the valuation factor, and the
population complete the set.

The valuation factor is population times the fuel price the run
used. The price file on the share today is not the one the 2020 runs
read (it has been revised since), so the price is recovered from the
tree itself: the priced gradient over the population-weighted one is
one constant per country. The population matches the SSP series held
constant within five-year spans, exactly. The raw dollar values
understate by 0.0036 (a GJ quantity was multiplied by a per-kWh
price); the repair is applied in compute.py, not here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd

import cil_regionalization as cilreg

TREE = Path("/project/cil/battuta-shares-S3-archive/gcp/outputs/energy_pixel_interaction/impacts-blueghost/median_OTHERIND_other_energy_TINV_clim_GMFD_dm/median")
CSVV = Path("/project/cil/home_dirs/egrenier/repos/energy-code-release-2020/projection_inputs/csvv/TINV_clim/FD_FGLS_inter_OTHERIND_other_energy_TINV_clim.csvv")
POPULATION = Path("/project/cil/gcp/social/baselines/population/merged/population-merged.SSP3.csv")
IR_SHAPES = Path("/project/cil/gcp/regions/world-combo-201710/agglomerated-world-new.shp")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TARGETS = HERE.parents[1] / "data" / "targets" / "gadm20" / "adm1.parquet"
STEM = "FD_FGLS_inter_OTHERIND_other_energy_TINV_clim"
YEAR = 2099
RCP, IAM, SSP = "rcp85", "high", "SSP3"
PAIR_GCM = "CCSM4"


def _mounted(path: Path) -> Path:
    if path.exists():
        return path
    alt = Path(str(path).replace("/project/cil", "/Volumes/cil"))
    return alt if alt.exists() else path


def slice_weights() -> list[str]:
    from cil_regionalization.fetch import default_cache_dir, load_registry

    name = "gadm20-adm1-per-source"
    full = cilreg.fetch_weights(name)
    frame = full.frame
    keep = sorted(set(frame[frame["ISO"] == "USA"]["hierid"]))
    rows = frame[frame["hierid"].isin(keep)]
    sums = rows.groupby("hierid")["popwt"].sum()
    assert (sums - 1).abs().max() < 1e-9, "slice breaks per-source sums"

    out = DATA / "weights" / "gadm20_adm1_per_source"
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "weights.parquet", index=False)
    record = load_registry()[name].record_id
    cached = default_cache_dir() / f"record-{record}" / name
    manifest = json.loads((cached / "weights.manifest.json").read_text())
    manifest["outputs"] = {
        "weights.parquet": hashlib.sha256(
            (out / "weights.parquet").read_bytes()
        ).hexdigest()
    }
    manifest["row_counts"] = {
        "regions": int(rows[["ISO", "ID_1"]].drop_duplicates().shape[0]),
        "source_units": len(keep),
        "total": len(rows),
    }
    (out / "weights.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"weights: {len(rows)} rows, {len(keep)} source regions")
    return keep


def read_slice(gcm: str, suffix: str, keep_idx, want_value=False):
    p = _mounted(TREE) / RCP / gcm / IAM / SSP / f"{STEM}{suffix}.nc4"
    with netCDF4.Dataset(p) as ds:
        years = ds.variables["year"][:]
        yi = int(np.where(years == YEAR)[0][0])
        g = np.asarray(ds.variables["rebased_bcde"][:, yi, :], dtype="float64")[:, keep_idx]
        vcv = np.asarray(ds.variables["vcv"][:], dtype="float64")
        value = None
        if want_value:
            value = np.asarray(ds.variables["rebased"][yi, :], dtype="float64")[keep_idx]
    return g, vcv, value


def region_index(keep: list[str]):
    p = _mounted(TREE) / RCP / PAIR_GCM / IAM / SSP / f"{STEM}.nc4"
    with netCDF4.Dataset(p) as ds:
        regs = [str(r) for r in ds.variables["regions"][:]]
    pos = {r: i for i, r in enumerate(regs)}
    return np.array([pos[r] for r in keep])


def build_population_and_factor(keep: list[str]) -> None:
    pop = pd.read_csv(_mounted(POPULATION), comment="#")
    pop = pop[pop["region"].isin(keep)]
    missing = set(keep) - set(pop["region"])
    assert not missing, f"regions absent from the SSP series: {sorted(missing)}"
    grid = pop.pivot(index="region", columns="year", values="value")
    held = grid[5 * (YEAR // 5)].reindex(keep)
    assert held.notna().all() and (held >= 0).all()
    out = pd.DataFrame({"hierid": keep, "population": held.values})
    out.to_parquet(DATA / "population.parquet", index=False)

    # the run's own prices: priced gradient over population-weighted
    # gradient, one constant per country
    idx = region_index(keep)
    g_lev, _, _ = read_slice(PAIR_GCM, "-levels", idx)
    g_priced, _, _ = read_slice(PAIR_GCM, "-price014-levels", idx)
    strongest = np.abs(g_lev).argmax(axis=0)
    sel = np.arange(len(keep))
    implied_pop = g_lev[strongest, sel] / read_slice(PAIR_GCM, "", idx)[0][strongest, sel]
    assert np.nanmax(np.abs(implied_pop / held.values - 1)) < 1e-5, \
        "population in the run differs from the span-held SSP series"
    implied_price = g_priced[strongest, sel] / g_lev[strongest, sel]
    countries = pd.Series([r[:3] for r in keep])
    price_by_country = pd.Series(implied_price).groupby(countries).median()
    spread = pd.Series(implied_price).groupby(countries).std()
    assert spread.max() < 1e-6, f"price varies within a country: {spread.to_dict()}"
    price_vec = price_by_country.reindex(countries).values
    factor = pd.DataFrame({
        "hierid": keep,
        "factor": held.values * price_vec,
    })
    factor.to_parquet(DATA / "wage_price_factor.parquet", index=False)
    print(f"population and factor: {len(keep)} regions; "
          f"prices {price_by_country.round(5).to_dict()}")


def build_gradients(keep: list[str]) -> None:
    idx = region_index(keep)
    tree = _mounted(TREE)
    gcms = sorted(p.name for p in (tree / RCP).iterdir()
                  if p.is_dir() and (p / IAM).exists())
    print(f"gcms: {len(gcms)}")

    vcv_ref = None
    effects = np.zeros((len(gcms), 80, len(keep)), dtype="float32")
    for i, gcm in enumerate(gcms):
        g_full, vcv, _ = read_slice(gcm, "", idx)
        g_hist, vcv2, _ = read_slice(gcm, "-histclim", idx)
        if vcv_ref is None:
            vcv_ref = vcv
        assert np.array_equal(vcv, vcv_ref) and np.array_equal(vcv2, vcv_ref), \
            "covariance matrix differs between files"
        effects[i] = (g_full - g_hist).astype("float32")
        print(gcm, flush=True)

    # the valuation factor must reproduce the tree's own priced gradients
    factor = pd.read_parquet(DATA / "wage_price_factor.parquet")["factor"].values
    g_full, _, _ = read_slice(PAIR_GCM, "", idx)
    g_priced, _, _ = read_slice(PAIR_GCM, "-price014-levels", idx)
    recon = g_full * factor[None, :]
    scale = np.nanmax(np.abs(g_priced))
    err = np.nanmax(np.abs(recon - g_priced)) / scale
    print(f"factor check against priced gradients: {err:.1e}")
    assert err < 1e-5

    out = DATA / "effect_gradients.nc4"
    with netCDF4.Dataset(out, "w") as ds:
        ds.createDimension("gcm", len(gcms))
        ds.createDimension("coefficient", 80)
        ds.createDimension("region", len(keep))
        gv = ds.createVariable("gcm", str, ("gcm",))
        for i, g in enumerate(gcms):
            gv[i] = g
        rv = ds.createVariable("regions", str, ("region",))
        for i, r in enumerate(keep):
            rv[i] = r
        ev = ds.createVariable("effect_bcde", "f4",
                               ("gcm", "coefficient", "region"),
                               zlib=True, complevel=4)
        ev[:] = effects
        ev.description = ("gradient of the effect of climate change "
                          "(full adaptation minus histclim) with respect "
                          f"to the regression coefficients, year {YEAR}")
        vv = ds.createVariable("vcv", "f8", ("coefficient", "coefficient"))
        vv[:] = vcv_ref
    print(f"gradients: {out.stat().st_size / 1e6:.1f} MB")

    # one unmodified pair for the notebook's checks
    pair = DATA / "pair_ccsm4.nc4"
    g_hist, _, var_hist = read_slice(PAIR_GCM, "-histclim", idx, want_value=True)
    _, _, var_full = read_slice(PAIR_GCM, "", idx, want_value=True)
    with netCDF4.Dataset(pair, "w") as ds:
        ds.createDimension("coefficient", 80)
        ds.createDimension("region", len(keep))
        rv = ds.createVariable("regions", str, ("region",))
        for i, r in enumerate(keep):
            rv[i] = r
        for name, arr in (("fulladapt_bcde", g_full), ("histclim_bcde", g_hist)):
            v = ds.createVariable(name, "f4", ("coefficient", "region"),
                                  zlib=True, complevel=4)
            v[:] = arr.astype("float32")
        for name, arr in (("fulladapt_variance", var_full),
                          ("histclim_variance", var_hist)):
            v = ds.createVariable(name, "f8", ("region",), zlib=True)
            v[:] = arr
        vv = ds.createVariable("vcv", "f8", ("coefficient", "coefficient"))
        vv[:] = vcv_ref
    print(f"pair: {pair.stat().st_size / 1e6:.1f} MB")


def build_gammas() -> None:
    lines = _mounted(CSVV).read_text().splitlines()
    gi = [i for i, l in enumerate(lines) if l.strip() == "gamma"][0]
    gammas = np.array([float(x) for x in lines[gi + 1].split(",")])
    assert gammas.shape == (80,)
    pd.DataFrame({"gamma": gammas}).to_parquet(DATA / "gammas.parquet", index=False)
    print("gammas: 80 coefficients")


def build_display_layer(keep: list[str]) -> None:
    w = pd.read_parquet(DATA / "weights" / "gadm20_adm1_per_source" / "weights.parquet")
    best = (
        w.sort_values("popwt", ascending=False)
        .drop_duplicates("hierid")
        .query("ISO == 'USA'")[["hierid", "ISO", "ID_1"]]
    )
    shapes = gpd.read_file(_mounted(IR_SHAPES))
    shapes = shapes[shapes["hierid"].isin(set(best["hierid"]))]
    shapes = shapes[["hierid", "geometry"]].merge(best, on="hierid")
    states = shapes.dissolve(by=["ISO", "ID_1"], as_index=False)[
        ["ISO", "ID_1", "geometry"]
    ]
    names = pd.read_parquet(TARGETS, columns=["ISO", "ID_1", "NAME_1"])
    states = states.merge(names, on=["ISO", "ID_1"])
    states["geometry"] = states.geometry.simplify(0.005)
    states.to_parquet(DATA / "usa_adm1_plot.parquet")
    print(f"display layer: {len(states)} states")


def main() -> int:
    DATA.mkdir(exist_ok=True)
    keep = slice_weights()
    build_population_and_factor(keep)
    build_display_layer(keep)
    build_gammas()
    build_gradients(keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
