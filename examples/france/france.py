import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Mortality in France, end to end")


@app.cell
def _():
    # first, since importing it points PROJ at this environment's data
    import cil_regionalization as cilreg

    import hashlib
    import json
    import os
    import subprocess
    import sys
    import warnings
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return (
        Path,
        cilreg,
        gpd,
        hashlib,
        json,
        mo,
        np,
        os,
        pd,
        plt,
        subprocess,
        sys,
        warnings,
    )


@app.cell
def _(Path, mo, os):
    HERE = mo.notebook_dir()
    DATA = Path(os.environ.get("FRANCE_DATA", HERE / "data")).resolve()
    OUT = Path(os.environ.get("FRANCE_OUT", HERE / "out")).resolve()
    # the paper's temperature table, which this repository does not hold
    GMST = os.environ.get("FRANCE_GMST")
    OFFLINE = os.environ.get("FRANCE_OFFLINE") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    return DATA, GMST, HERE, OFFLINE, OUT


@app.cell
def _(mo):
    mo.md(
        r"""
    # Mortality in France, end to end

    This notebook makes one product from its inputs. It is the effect of
    climate change on mortality in metropolitan France, for its 13
    regions and 96 departments. It comes three ways, as deaths (a total),
    as deaths per 100,000 people (a rate) and as damages over GDP (a
    share). Each is shown for 2080 to 2099, year by year, and at warming
    levels.

    Every step is a call of the library or of its command line, `cilreg`,
    and nothing is done by hand in between. The sections follow the order
    a product is made in.

    1. The inputs
    2. Building the weights with `cilreg run`
    3. The published weights, the other route
    4. Applying the weights three ways
    5. Statistics over the members
    6. Climate model weights
    7. A product from a tree, with `cilreg pipeline`
    8. Warming levels
    9. Provenance

    **The sample.** Two Monte Carlo batches of the mortality projections,
    rcp45 and rcp85, every climate model the tree has under each, SSP3 and
    the low growth model, 2020 to 2099. Under rcp85 that is all 33 models,
    66 draws. Section 7 counts the leaves of each rcp from the tree. The
    numbers show how a product is made, and they are not results.

    **The boundaries are not GADM's.** The published weights are built on
    GADM, whose licence forbids redistributing its boundaries, so they
    cannot be committed here. This notebook uses ADMIN EXPRESS instead,
    the official French boundaries of IGN under the Licence Ouverte 2.0,
    which anyone may redistribute. That is also what anyone does who
    brings their own boundaries, which is the reason the weights can be
    built at all.

    **The overseas departments are left out.** ADMIN EXPRESS includes
    Guadeloupe, Martinique, Guyane, La Réunion and Mayotte as departments
    of France. The impact regions file keeps them under their own country
    codes, not under France, and they would scatter the maps across three
    oceans. So this notebook covers metropolitan France with Corsica, 96
    departments in 13 regions, and `prepare_data.py` drops the rest.
    """
    )
    return


@app.cell
def _(DATA, OUT, Path, cilreg, json, os, subprocess, sys):
    CILREG = Path(sys.executable).with_name("cilreg")
    PACKAGE = Path(cilreg.__file__).resolve().parent
    SOURCE_VERSION = "world-combo-201710"

    def cilreg_cli(*args):
        """Run `cilreg` with ``args`` as from a shell and return what it
        printed. Raises if it fails."""
        done = subprocess.run(
            [str(CILREG), *map(str, args)], capture_output=True, text=True
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"cilreg {' '.join(map(str, args))} failed:\n{done.stderr}"
            )
        return done.stdout

    def shown(path):
        """A path as a reader of this notebook would write it, without
        the directories of the machine it ran on."""
        path = str(path)
        for root, name in (
            (OUT, "out"),
            (DATA, "data"),
            (PACKAGE, "cil_regionalization"),
        ):
            path = path.replace(str(root), name)
        return path

    def relative(path, start):
        return os.path.relpath(path, start)

    def without_machine_paths(document):
        """``document`` with every path in it shown as above."""
        return json.loads(shown(json.dumps(document)))

    return SOURCE_VERSION, cilreg_cli, relative, shown, without_machine_paths


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. The inputs

    Four files make the weights. The impact regions are the 2017
    agglomeration of the world into 24,378 regions, here the ones of
    France. The regions and departments are ADMIN EXPRESS. The population
    is the GPW v4 rev 10 raster of 2015 at 30 arc seconds, under CC BY
    4.0, cut to France. `sources.json` says where each came from, and its
    SHA256.
    """
    )
    return


@app.cell
def _(DATA, gpd, json, pd):
    impact_regions = gpd.read_parquet(DATA / "impact_regions.parquet")
    regions = gpd.read_parquet(DATA / "regions.parquet")
    departments = gpd.read_parquet(DATA / "departments.parquet")
    sources = (
        json.loads((DATA / "sources.json").read_text())
        if (DATA / "sources.json").exists()
        else {"note": "synthetic data, from synthetic_data.py"}
    )
    pd.DataFrame(
        {
            "file": [
                "impact_regions.parquet",
                "regions.parquet",
                "departments.parquet",
            ],
            "rows": [len(impact_regions), len(regions), len(departments)],
        }
    )
    return departments, impact_regions, regions, sources


@app.cell
def _(sources):
    sources
    return


@app.cell
def _(DATA, departments, impact_regions, np, plt):
    import rasterio

    with rasterio.open(DATA / "population_2015_30ss.tif") as _raster:
        _people = _raster.read(1, masked=True)
        _extent = [
            _raster.bounds.left,
            _raster.bounds.right,
            _raster.bounds.bottom,
            _raster.bounds.top,
        ]
    _fig, _ax = plt.subplots(figsize=(7, 7))
    _ax.imshow(
        np.log10(np.ma.maximum(_people, 1)), extent=_extent, cmap="Greys", alpha=0.8
    )
    impact_regions.boundary.plot(ax=_ax, color="tab:orange", linewidth=0.4)
    departments.boundary.plot(ax=_ax, color="tab:blue", linewidth=0.6)
    _ax.set_title(
        "People per cell (log, grey), impact regions (orange), departments (blue)"
    )
    _ax.set_axis_off()
    _fig
    return (rasterio,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Where the raster has no people, at sea and on some of the coast, a
    weight by population cannot be taken. An impact region that has no
    people in any of its pieces falls back to its area, and the manifest
    counts those.

    ## 2. Building the weights with `cilreg run`

    A weights config names the target boundaries, the source boundaries
    and the raster, and which way the weights are normalised. Per source,
    each impact region's weights add up to one, which splits a total
    between the units it touches. Per destination, each unit's weights add
    up to one, which averages a rate over the regions inside it. Four
    files come out of four configs, for the regions and the departments,
    one each way.
    """
    )
    return


@app.cell
def _(DATA, OUT, SOURCE_VERSION, relative, sources):
    _configs = OUT / "configs"
    _configs.mkdir(parents=True, exist_ok=True)
    TARGETS_VERSION = sources.get("targets_version", "synthetic")
    LEVELS = {
        "adm1": ("regions.parquet", "INSEE_REG"),
        "adm2": ("departments.parquet", "INSEE_DEP"),
    }

    def weights_config(level, normalization):
        """The text of one weights config, with its paths relative to
        the config, and where it is written."""
        file, key = LEVELS[level]
        name = f"{level}_{normalization}"
        path = _configs / f"weights_{name}.toml"
        text = f'''normalization = "{normalization}"

    [project]
    name = "france_{name}"
    description = "impact regions to ADMIN EXPRESS {level}, population weights"

    [regions]
    path = "{relative(DATA / file, _configs)}"
    id_fields = ["{key}"]
    version = "{TARGETS_VERSION}"

    [source]
    path = "{relative(DATA / "impact_regions.parquet", _configs)}"
    id_fields = ["hierid"]
    version = "{SOURCE_VERSION}"
    on_zero_overlap = "skip"

    [[weights]]
    name = "pop"
    raster = "{relative(DATA / "population_2015_30ss.tif", _configs)}"
    fallback = "area"

    [backend]
    kind = "local"

    [output]
    dir = "{relative(OUT / "weights" / name, _configs)}"
    '''
        path.write_text(
            "\n".join(line.removeprefix("    ") for line in text.splitlines()) + "\n"
        )
        return path

    WEIGHT_CONFIGS = {
        (level, way): weights_config(level, way)
        for level in LEVELS
        for way in ("per_source", "per_destination")
    }
    return TARGETS_VERSION, WEIGHT_CONFIGS


@app.cell
def _(WEIGHT_CONFIGS, mo):
    mo.md(
        "The config of the department weights, per source:\n\n```toml\n"
        + WEIGHT_CONFIGS["adm2", "per_source"].read_text()
        + "```"
    )
    return


@app.cell
def _(WEIGHT_CONFIGS, cilreg, cilreg_cli, mo, shown):
    _printed = []
    for (_level, _way), _path in WEIGHT_CONFIGS.items():
        _printed.append(
            f"$ cilreg run {shown(_path)}\n" + shown(cilreg_cli("run", _path))
        )
    WEIGHTS = {
        _key: cilreg.WeightsArtifact.load(cilreg.load_config(_path).output.dir)
        for _key, _path in WEIGHT_CONFIGS.items()
    }
    mo.md("```\n" + "\n".join(_printed) + "```")
    return (WEIGHTS,)


@app.cell
def _(WEIGHTS):
    WEIGHTS["adm2", "per_source"].frame.head(10)
    return


@app.cell
def _(Path, WEIGHT_CONFIGS, cilreg, json, without_machine_paths):
    _output = cilreg.load_config(WEIGHT_CONFIGS["adm2", "per_source"]).output
    ADM2_MANIFEST_PATH = Path(_output.dir) / "weights.manifest.json"
    adm2_manifest = json.loads(ADM2_MANIFEST_PATH.read_text())
    {
        _key: without_machine_paths(adm2_manifest[_key])
        for _key in (
            "config_hash",
            "inputs",
            "row_counts",
            "fallback_counts",
            "partial_coverage",
            "regions_version",
            "source_version",
        )
        if _key in adm2_manifest
    }
    return ADM2_MANIFEST_PATH, adm2_manifest


@app.cell
def _(WEIGHTS, adm2_manifest, departments, impact_regions, mo, plt):
    _frame = WEIGHTS["adm2", "per_source"].frame
    _method = [c for c in _frame.columns if c.endswith("_method")][0]
    _fallen = set(_frame.loc[_frame[_method] != "native", "hierid"])
    _fig, _ax = plt.subplots(figsize=(7, 7))
    departments.plot(ax=_ax, color="whitesmoke", edgecolor="grey", linewidth=0.4)
    impact_regions[impact_regions["hierid"].isin(_fallen)].plot(ax=_ax, color="tab:red")
    _ax.set_title(f"{len(_fallen)} impact regions weighted by area, for want of people")
    _ax.set_axis_off()
    _partial = adm2_manifest.get("partial_coverage") or {"count": 0}
    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
    {_partial["count"]} impact regions are covered by the departments for
    less than {_partial.get("threshold", 0.999):.1%} of their area. That is
    where the ADMIN EXPRESS coastline and the impact regions' coastline
    differ, and a share of those regions' values would land in no
    department. The manifest names them, and `apply_weights` refuses such
    a file unless told `allow_partial_coverage=True`, which this notebook
    then says.
    """
            ),
        ]
    )
    return


@app.cell
def _(WEIGHTS, cilreg):
    ALLOW_PARTIAL = any(
        (_artifact.partial_coverage or {}).get("count", 0) > 0
        for _artifact in WEIGHTS.values()
    )
    # a region with no people has no weight in an average by population
    NO_PEOPLE = cilreg.SourceUnitPolicies(on_zero_weight="skip")
    ALLOW_PARTIAL
    return ALLOW_PARTIAL, NO_PEOPLE


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. The published weights, the other route

    The published weights are on Zenodo. `cilreg fetch` downloads a file,
    checks it against the SHA256 its record gives, and caches it.
    `fetch_weights` does the same from Python. They are built on GADM 2.0,
    whose departments are the same 96, so one draw aggregated with them
    and with the weights of section 2 can be compared department by
    department. The rows differ where the two sets of boundaries do.

    With `FRANCE_OFFLINE=1` the France rows committed under
    `data/published/` stand in for the download.
    """
    )
    return


@app.cell
def _(DATA, OFFLINE, cilreg, cilreg_cli, mo):
    _slice = DATA / "published" / "gadm20-adm2-per-source"
    published_slice = cilreg.WeightsArtifact.load(_slice)
    if OFFLINE:
        published = published_slice
        _said = "Offline, so the committed France rows stand in for the published file."
    else:
        _said = "```\n$ cilreg fetch gadm20-adm2-per-source\n" + cilreg_cli(
            "fetch", "gadm20-adm2-per-source"
        )
        published = cilreg.fetch_weights("gadm20-adm2-per-source")
        _france = published.frame[published.frame["ISO"] == "FRA"].reset_index(
            drop=True
        )
        _committed = published_slice.frame.reset_index(drop=True)
        assert _france.equals(_committed), (
            "the committed slice is not the published rows"
        )
        _said += "```\n\nThe France rows of the download are the committed ones, row for row."
    mo.md(_said)
    return (published,)


@app.cell
def _(DATA, pd):
    def draw(batch, rcp, gcm):
        """One draw's deaths from climate change, by impact region and
        year. A leaf's deaths minus those of its histclim sibling, which
        has the same people under the climate of the past."""
        _leaf = DATA / "tree" / batch / rcp / gcm
        _levels = pd.read_parquet(_leaf / "deaths.parquet")
        _history = pd.read_parquet(_leaf / "deaths-histclim.parquet")
        _both = _levels.merge(
            _history, on=["hierid", "year"], suffixes=("", "_histclim")
        )
        return _both.assign(
            deaths=_both["deaths"].astype("float64") - _both["deaths_histclim"]
        )[["hierid", "year", "deaths"]]

    LEAVES = sorted(
        (_b.name, _r.name, _g.name)
        for _b in (DATA / "tree").iterdir()
        for _r in _b.iterdir()
        for _g in _r.iterdir()
    )
    DRAW = next(_leaf for _leaf in LEAVES if _leaf[1] == "rcp85")
    return DRAW, LEAVES, draw


@app.cell
def _(ALLOW_PARTIAL, DRAW, SOURCE_VERSION, WEIGHTS, cilreg, draw, pd, published):
    one_draw = draw(*DRAW)
    _late = one_draw[one_draw["year"].between(2080, 2099)]
    _ours = cilreg.apply_weights(
        WEIGHTS["adm2", "per_source"],
        _late,
        kind="extensive",
        weight="pop",
        value_col="deaths",
        data_version=SOURCE_VERSION,
        allow_partial_coverage=ALLOW_PARTIAL,
    ).frame
    _theirs = cilreg.apply_weights(
        published,
        _late,
        kind="extensive",
        weight="pop",
        value_col="deaths",
        data_version=SOURCE_VERSION,
        restrict_to_sources={(_h,) for _h in _late["hierid"].unique()},
        allow_partial_coverage=ALLOW_PARTIAL,
    ).frame
    # Which of our departments each GADM department is, by where its
    # people go. Its people in an impact region are split as our weights
    # split that region's.
    _pieces = published.frame[["ISO", "ID_1", "ID_2", "hierid", "pop_raw"]].merge(
        WEIGHTS["adm2", "per_source"].frame[["hierid", "INSEE_DEP", "popwt"]],
        on="hierid",
    )
    _pieces["people"] = _pieces["pop_raw"] * _pieces["popwt"]
    _gadm = ["ISO", "ID_1", "ID_2"]
    _pairs = _pieces.groupby(_gadm + ["INSEE_DEP"], as_index=False)["people"].sum()
    _pairs["share"] = _pairs["people"] / _pairs.groupby(_gadm)["people"].transform(
        "sum"
    )
    matched = _pairs.sort_values("share").groupby(_gadm).tail(1)
    _theirs = _theirs.merge(matched[_gadm + ["INSEE_DEP"]], on=_gadm)
    _ours = _ours.groupby("INSEE_DEP")["deaths"].mean()
    _theirs = _theirs.groupby("INSEE_DEP")["deaths"].mean()
    compared = pd.DataFrame({"admin_express": _ours, "gadm20": _theirs})
    compared["difference_percent"] = 100 * (
        compared["admin_express"] / compared["gadm20"] - 1
    )
    compared.sort_values("difference_percent", key=abs, ascending=False).head(10)
    return compared, matched, one_draw


@app.cell
def _(DRAW, compared, matched, mo):
    mo.md(
        f"""
    Each GADM department is taken to be the ADMIN EXPRESS department that
    most of its people fall in. That holds at least
    {matched["share"].min():.1%} of each one's people, and
    {matched["INSEE_DEP"].nunique()} of our departments are matched by
    {len(matched)} of theirs.

    One draw ({"/".join(DRAW)}), deaths a year averaged over 2080 to 2099.
    The department totals of the two weights differ by
    {compared["difference_percent"].abs().median():.2f}% at the median and
    {compared["difference_percent"].abs().max():.2f}% at most. The ten
    largest differences are above. They are where the boundaries differ,
    since the impact regions, the draw and the raster are the same.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Applying the weights three ways

    Which weights to use follows from one question. Does adding the
    variable across regions give a meaningful total?

    - **Deaths** add up. They are `extensive`, split between the
      departments an impact region touches, with the weights per source.
    - **Deaths per 100,000** do not. They are `intensive`, averaged over
      the impact regions inside a department by population, with the
      weights per destination.
    - **Damages over GDP** do not either, and they are not an average of
      the regions' shares. They are a `ratio`. The numerator and the
      denominator are added up separately with the weights per source,
      and divided at the department.

    One draw, averaged over 2080 to 2099, and mapped after each.
    """
    )
    return


@app.cell
def _(departments, plt):
    def department_map(values, column, title, cmap="Reds"):
        """``values`` by INSEE_DEP, as a map of the departments."""
        _frame = departments.merge(values, on="INSEE_DEP", how="left")
        _fig, _ax = plt.subplots(figsize=(6, 6))
        _frame.plot(
            column=column,
            ax=_ax,
            cmap=cmap,
            legend=True,
            edgecolor="white",
            linewidth=0.3,
            missing_kwds={"color": "lightgrey"},
        )
        _ax.set_title(title)
        _ax.set_axis_off()
        return _fig

    return (department_map,)


@app.cell
def _(ALLOW_PARTIAL, SOURCE_VERSION, WEIGHTS, cilreg, department_map, one_draw):
    _late = one_draw[one_draw["year"].between(2080, 2099)]
    deaths_by_department = (
        cilreg.apply_weights(
            WEIGHTS["adm2", "per_source"],
            _late,
            kind="extensive",
            weight="pop",
            value_col="deaths",
            data_version=SOURCE_VERSION,
            allow_partial_coverage=ALLOW_PARTIAL,
        )
        .frame.groupby("INSEE_DEP", as_index=False)["deaths"]
        .mean()
    )
    department_map(
        deaths_by_department, "deaths", "Deaths a year from climate change, 2080-2099"
    )
    return


@app.cell
def _(
    ALLOW_PARTIAL,
    DATA,
    NO_PEOPLE,
    SOURCE_VERSION,
    WEIGHTS,
    cilreg,
    department_map,
    mo,
    one_draw,
    pd,
):
    _people = pd.read_parquet(DATA / "ir_population.parquet")
    _late = one_draw[one_draw["year"].between(2080, 2099)].merge(
        _people, on=["hierid", "year"]
    )
    _late = _late.assign(rate=1e5 * _late["deaths"] / _late["pop"])[
        ["hierid", "year", "rate"]
    ]
    _applied = cilreg.apply_weights(
        WEIGHTS["adm2", "per_destination"],
        _late,
        kind="intensive",
        weight="pop",
        value_col="rate",
        data_version=SOURCE_VERSION,
        policies=NO_PEOPLE,
        allow_partial_coverage=ALLOW_PARTIAL,
    )
    rate_by_department = _applied.frame.groupby("INSEE_DEP", as_index=False)[
        "rate"
    ].mean()
    mo.vstack(
        [
            department_map(
                rate_by_department,
                "rate",
                "Deaths per 100,000 from climate change, 2080-2099",
            ),
            mo.md(
                f"""
    An average by population gives no say to a region with no people, so
    its weight per destination is zero, and the area fallback does not
    apply. `apply_weights` refuses such regions unless told to skip them,
    and then the result names them, here
    {[_h for (_h,) in _applied.coverage.zero_weight]}.
    """
            ),
        ]
    )
    return


@app.cell
def _(ALLOW_PARTIAL, DATA, DRAW, NO_PEOPLE, SOURCE_VERSION, WEIGHTS, cilreg, pd):
    _leaf = pd.read_parquet(DATA / "tree" / "/".join(DRAW) / "damages.parquet")
    _late = _leaf[_leaf["year"].between(2080, 2099)].astype(
        {"damages": "float64", "gdp": "float64"}
    )
    _share = cilreg.apply_weights(
        WEIGHTS["adm2", "per_source"],
        _late,
        kind="ratio",
        weight="pop",
        value_col="damages",
        denominator_col="gdp",
        data_version=SOURCE_VERSION,
        allow_partial_coverage=ALLOW_PARTIAL,
    ).frame
    # what the ratio route is not, the regions' own shares averaged
    _averaged = cilreg.apply_weights(
        WEIGHTS["adm2", "per_destination"],
        _late.assign(share=_late["damages"] / _late["gdp"])[
            ["hierid", "year", "share"]
        ],
        kind="intensive",
        weight="pop",
        value_col="share",
        data_version=SOURCE_VERSION,
        policies=NO_PEOPLE,
        allow_partial_coverage=ALLOW_PARTIAL,
    ).frame
    shares = (
        _share.groupby("INSEE_DEP")["damages"]
        .mean()
        .rename("ratio")
        .to_frame()
        .join(_averaged.groupby("INSEE_DEP")["share"].mean().rename("averaged"))
        .assign(percent_of_gdp=lambda f: 100 * f["ratio"])
        .reset_index()
    )
    return (shares,)


@app.cell
def _(department_map, mo, shares):
    _gap = (shares["averaged"] / shares["ratio"] - 1).abs()
    _worst = shares.loc[_gap.idxmax()]
    mo.vstack(
        [
            department_map(
                shares, "percent_of_gdp", "Damages, percent of GDP, 2080-2099"
            ),
            mo.md(
                f"""
    The share is the department's damages over the department's GDP. The
    regions' own shares averaged by population would be another number,
    because a region's weight in that average is its people, not its
    GDP. In department {_worst["INSEE_DEP"]} the average gives
    {100 * _worst["averaged"]:.4f}% where the ratio gives
    {100 * _worst["ratio"]:.4f}%, {100 * _gap.max():.1f}% apart. Across
    the departments the two differ by {100 * _gap.median():.1f}% at the
    median.
    """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Statistics over the members

    A quantile does not pass through a sum. The 95th percentile of a
    department is not the sum of its regions' 95th percentiles, which
    would have every region at its worst draw at once. So every draw is
    aggregated on its own, and the statistics are taken last, over the
    draws.

    `window_means` averages each draw over the years of a window, and
    `pooled_statistics` takes the statistics over the draws.
    `summarize_samples` does both in one call, and the cell checks that
    it gives the same numbers. Here the quantile is the step of the
    historical extraction tool, `quantile_method="lower_step"`, and the
    means are the vectorised sums, `summation="fast"`, as in the published
    runs. With the step, a quantile below the first draw's share is
    `-inf`. `summation="exact"` sums exactly and refuses an infinite
    statistic.
    """
    )
    return


@app.cell
def _(ALLOW_PARTIAL, LEAVES, SOURCE_VERSION, WEIGHTS, cilreg, draw, pd):
    _pieces = []
    for _batch, _rcp, _gcm in LEAVES:
        if _rcp != "rcp85":
            continue
        _aggregated = cilreg.apply_weights(
            WEIGHTS["adm2", "per_source"],
            draw(_batch, _rcp, _gcm),
            kind="extensive",
            weight="pop",
            value_col="deaths",
            data_version=SOURCE_VERSION,
            allow_partial_coverage=ALLOW_PARTIAL,
        ).frame
        _pieces.append(_aggregated.assign(batch=_batch, gcm=_gcm))
    members = pd.concat(_pieces, ignore_index=True)
    members.head()
    return (members,)


@app.cell
def _(cilreg, members, pd, warnings):
    STATISTICS = dict(
        quantiles=[0.05, 0.5, 0.95], quantile_method="lower_step", summation="fast"
    )
    with warnings.catch_warnings():
        # section 6 takes up the models' weights
        warnings.simplefilter("ignore", cilreg.UnweightedModelWeightsWarning)
        _means = cilreg.window_means(
            members,
            time_col="year",
            windows=[(2080, 2099)],
            value_col="deaths",
            summation="fast",
        )
        in_two_steps = cilreg.pooled_statistics(
            _means, sample_dims=["batch", "gcm"], value_col="deaths", **STATISTICS
        )
        _in_one = cilreg.summarize_samples(
            members,
            sample_dims=["batch", "gcm"],
            time_col="year",
            window=(2080, 2099),
            value_col="deaths",
            **STATISTICS,
        )
    _keys = ["INSEE_DEP", "statistic"]
    pd.testing.assert_frame_equal(
        in_two_steps.sort_values(_keys).reset_index(drop=True)[_keys + ["deaths"]],
        _in_one.sort_values(_keys).reset_index(drop=True)[_keys + ["deaths"]],
    )
    in_two_steps.pivot(index="INSEE_DEP", columns="statistic", values="deaths").head()
    return STATISTICS, in_two_steps


@app.cell
def _(department_map, in_two_steps):
    department_map(
        in_two_steps[in_two_steps["statistic"] == "q50"],
        "deaths",
        "Median over the draws, deaths a year, rcp85, 2080-2099",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Climate model weights

    The projections weight each climate model by the SMME weights, which
    give the 21 models and the 12 surrogates their share of the
    distribution of warming. `read_model_weights` reads the table and
    `attach` gives every draw the weight of its model. A surrogate is
    named `surrogate_GFDL-CM3_89` in the tree and `gfdl-cm3_89` in the
    table, and a model can carry a `*` in the table, and `attach` matches
    them. Without weights every draw counts the same.
    """
    )
    return


@app.cell
def _(DATA, STATISTICS, cilreg, members, pd, plt, warnings):
    model_weights = cilreg.read_model_weights(DATA / "smme" / "rcp85_SMME_weights.tsv")
    _weighted = members.assign(weight=model_weights.attach(members["gcm"]))
    _options = dict(
        sample_dims=["batch", "gcm"],
        time_col="year",
        window=(2080, 2099),
        value_col="deaths",
        **STATISTICS,
    )
    with_weights = cilreg.summarize_samples(_weighted, weight_col="weight", **_options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cilreg.UnweightedModelWeightsWarning)
        _without = cilreg.summarize_samples(members, **_options)
    _both = with_weights.merge(
        _without, on=["INSEE_DEP", "statistic"], suffixes=("_weighted", "_equal")
    )
    _moved = _both.assign(
        moved_percent=100 * (_both["deaths_weighted"] / _both["deaths_equal"] - 1)
    ).pivot(index="INSEE_DEP", columns="statistic", values="moved_percent")
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _statistic in ["q05", "q50", "q95"]:
        _ax.hist(_moved[_statistic], bins=20, alpha=0.6, label=_statistic)
    _ax.axvline(0, color="black", linewidth=0.5)
    _ax.set_xlabel("percent moved by the model weights, across departments")
    _ax.legend()
    _fig
    return (model_weights,)


@app.cell
def _(STATISTICS, cilreg, members, model_weights, pd, warnings):
    _department = members.groupby("INSEE_DEP")["deaths"].sum().idxmax()
    _one = members[members["INSEE_DEP"] == _department]
    _one = _one.assign(weight=model_weights.attach(_one["gcm"]))
    _rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cilreg.UnweightedModelWeightsWarning)
        for _method, _weight in (
            ("linear", None),
            ("lower_step", None),
            ("lower_step", "weight"),
            ("inverted_cdf", "weight"),
        ):
            _stats = cilreg.summarize_samples(
                # unweighted, a weight column would be one more identity
                _one if _weight else _one.drop(columns="weight"),
                sample_dims=["batch", "gcm"],
                time_col="year",
                window=(2080, 2099),
                value_col="deaths",
                quantiles=STATISTICS["quantiles"],
                include_mean=False,
                weight_col=_weight,
                quantile_method=_method,
                summation="fast",
            )
            _rows[f"{_method}, {'SMME weights' if _weight else 'equal'}"] = (
                _stats.set_index("statistic")["deaths"]
            )
    pd.DataFrame(_rows).T.rename_axis(f"department {_department}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The quantile method names a computation. `"linear"` interpolates
    between draws, as numpy does, and has no weighted form, so it is
    refused with weights. `"lower_step"` is the extraction tool's step,
    the draw one below the one whose cumulative weight reaches q, which
    the published physical products use. `"inverted_cdf"` is the standard
    inverse, the smallest draw whose cumulative weight reaches q. None of
    them has a default.

    ## 7. A product from a tree, with `cilreg pipeline`

    Sections 4 to 6 did by hand what a pipeline does over a whole Monte
    Carlo tree. One config declares the tree's levels and their values,
    and the run checks that every declared leaf is there before it reads
    any. The deaths are the levels file minus its histclim sibling, which
    `tree.subtract_filename` takes per leaf. The values of `rcp` without a
    surrogate are left out with a wildcard in `tree.exclude`, so they are
    not taken for missing.

    The leaves are aggregated once, with `--stage aggregate`, and kept.
    Then `--stage reduce` makes each product from the same leaves, without
    aggregating again. The three reductions of the years are three
    reduces.

    - `window_mean` averages each draw over 2080 to 2099 first, and the
      statistics are over those means.
    - `pooled` takes no mean. Every year of every draw in the window is a
      member of the sample, so the band carries the spread from year to
      year.
    - `per_year` keeps the year, and the statistics are over the draws in
      each year from 2020 to 2099, a time series.
    """
    )
    return


@app.cell
def _(DATA, LEAVES, OUT, relative):
    PIPELINE = OUT / "pipeline"
    _configs = OUT / "configs"
    BATCHES = sorted({_b for _b, _, _ in LEAVES})
    GCMS = sorted({_g for _, _, _g in LEAVES})

    def tree_section(filename, rcps, subtract=None):
        """The [tree] of a config over ``rcps``, with the cells the tree
        lacks left out by a wildcard."""
        absent = [
            ["*", _r, _g]
            for _r in rcps
            for _g in GCMS
            if not any(_l[1:] == (_r, _g) for _l in LEAVES)
        ]
        lines = [
            "[tree]",
            f'root = "{relative(DATA / "tree", _configs)}"',
            'levels = ["batch", "rcp", "gcm"]',
            f'filename = "{filename}"',
        ]
        if subtract:
            lines.append(f'subtract_filename = "{subtract}"')
        lines.append(f"exclude = {absent!r}".replace("'", '"'))
        lines += [
            "",
            "[tree.expect]",
            f"batch = {BATCHES!r}".replace("'", '"'),
            f"rcp = {list(rcps)!r}".replace("'", '"'),
            f"gcm = {GCMS!r}".replace("'", '"'),
        ]
        return "\n".join(lines) + "\n"

    def pipeline_config(name, *, tree, data, stats, output, reduce=True):
        """Write one pipeline config and return its path."""
        path = _configs / f"{name}.toml"
        text = (
            "[weights]\n"
            f'dir = "{relative(OUT / "weights" / "adm2_per_source", _configs)}"\n'
            'weight = "pop"\n\n'
            + tree
            + "\n[data]\n"
            + data
            + "\n[stats]\n"
            + stats
            + "\n[output]\n"
            + f'dir = "{relative(PIPELINE, _configs)}"\n'
            + output
            + ("\n[reduce.pandas]\nmemory_gb = 2\n" if reduce else "")
        )
        path.write_text(text)
        return path

    return BATCHES, GCMS, PIPELINE, pipeline_config, tree_section


@app.cell
def _(DATA, OUT, cilreg_cli, mo, pipeline_config, relative, shown, tree_section):
    _deaths = 'format = "parquet"\nvalue_col = "deaths"\nkind = "extensive"\nversion = "world-combo-201710"\n'
    _stats = (
        'sample_dims = ["batch", "gcm"]\ntime_col = "year"\nquantiles = [0.05, 0.5, 0.95]\n'
        'quantile_method = "lower_step"\nsummation = "fast"\n'
    )
    # the pipeline resolves this path against the directory it runs in,
    # not the config's, so it is written whole
    _weights = (
        f'model_weights = "{DATA / "smme" / "rcp85_SMME_weights.tsv"}"\n'
        'model_weight_level = "gcm"\n'
    )
    aggregate_deaths = pipeline_config(
        "deaths_aggregate",
        tree=tree_section(
            "deaths.parquet", ["rcp45", "rcp85"], subtract="deaths-histclim.parquet"
        ),
        data=_deaths,
        stats=_stats + 'time_reduction = "window_mean"\nwindows = [[2080, 2099]]\n',
        output='stem = "deaths"\nwrite_leaves = true\n',
        reduce=False,
    )
    _rcp85 = tree_section(
        "deaths.parquet", ["rcp85"], subtract="deaths-histclim.parquet"
    )
    REDUCES = {
        _name: pipeline_config(
            f"deaths_{_name}",
            tree=_rcp85,
            data=_deaths,
            stats=_stats + _weights + _time,
            output=f'stem = "deaths_{_name}"\nleaves_stem = "deaths"\n',
        )
        for _name, _time in {
            "window_mean": 'time_reduction = "window_mean"\nwindows = [[2080, 2099]]\n',
            "pooled": 'time_reduction = "pooled"\nwindows = [[2080, 2099]]\n',
            "per_year": 'time_reduction = "per_year"\nyears = [2020, 2099]\n',
        }.items()
    }
    _printed = f"$ cilreg pipeline {shown(aggregate_deaths)} --stage aggregate\n"
    _printed += shown(cilreg_cli("pipeline", aggregate_deaths, "--stage", "aggregate"))
    for _name, _path in REDUCES.items():
        _printed += f"\n$ cilreg pipeline {shown(_path)} --stage reduce\n"
        _printed += shown(cilreg_cli("pipeline", _path, "--stage", "reduce"))
    mo.vstack(
        [
            mo.md(
                "The config that aggregates, then the one that reduces with window means:"
            ),
            mo.md("```toml\n" + aggregate_deaths.read_text() + "```"),
            mo.md("```toml\n" + shown(REDUCES["window_mean"].read_text()) + "```"),
            mo.md("```\n" + _printed + "```"),
        ]
    )
    return REDUCES, aggregate_deaths


@app.cell
def _(PIPELINE, REDUCES, pd):
    products = {
        _name: pd.read_parquet(PIPELINE / f"deaths_{_name}.parquet")
        for _name in REDUCES
    }
    products["window_mean"].head()
    return (products,)


@app.cell
def _(members, mo, plt, products):
    _department = members.groupby("INSEE_DEP")["deaths"].sum().idxmax()
    _bands = {
        _name: products[_name][products[_name]["INSEE_DEP"] == _department].set_index(
            "statistic"
        )["deaths"]
        for _name in ("window_mean", "pooled")
    }
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(11, 4))
    for _i, (_name, _band) in enumerate(_bands.items()):
        _left.plot([_i, _i], [_band["q05"], _band["q95"]], linewidth=8, alpha=0.5)
        _left.plot(_i, _band["q50"], "ko")
    _left.set_xticks([0, 1], list(_bands))
    _left.set_title(f"5 to 95 band, department {_department}, 2080-2099")
    _series = products["per_year"][
        products["per_year"]["INSEE_DEP"] == _department
    ].pivot(index="year", columns="statistic", values="deaths")
    # plain floats, which every matplotlib takes whatever pandas made
    _years = _series.index.to_numpy(dtype="float64")
    _right.fill_between(
        _years,
        _series["q05"].to_numpy(dtype="float64"),
        _series["q95"].to_numpy(dtype="float64"),
        alpha=0.3,
    )
    _right.plot(_years, _series["q50"].to_numpy(dtype="float64"), color="black")
    _right.set_title(f"per_year, department {_department}")
    _right.set_xlabel("year")
    _widths = {_name: _band["q95"] - _band["q05"] for _name, _band in _bands.items()}
    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
    The 5 to 95 band is {_widths["pooled"]:.3g} deaths wide pooled and
    {_widths["window_mean"]:.3g} with window means. Pooling keeps each
    year's weather in the sample, where the window mean averages it away.
    The per year series is the same draws, one year at a time.
    """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Warming levels

    A warming level product pools the draws by how warm their climate
    model is in a period, not by the calendar alone. Each member of the
    temperature table, a model under an rcp, has its global mean
    temperature rebased to its own mean over 1986 to 2005 and raised by
    0.61, the warming from pre-industrial to that period in AR5 (WG1
    TS.2.2.1). Its mean over a period then picks its level, which holds a
    mean from 0.25 below the level to 0.25 above it. The levels are 1.5,
    2, 3, 4 and 5. In mid century no model reaches 4 or 5, so those come
    out empty, and the record says so.

    The table is the one the Inequality Paper read, given by
    `FRANCE_GMST`. It is a derived file of the lab and is not in this
    repository, so without it this section stops here.
    """
    )
    return


@app.cell
def _(GMST, cilreg, mo):
    mo.stop(
        GMST is None,
        mo.md("**`FRANCE_GMST` is not set, so the warming levels are skipped.**"),
    )
    PERIODS = {
        # the physical products average nineteen years of impacts, as the
        # extraction tool did, and the valued ones pool twenty
        "physical": [
            cilreg.Period(gmst_span=(2040, 2059), impact_span=(2040, 2058)),
            cilreg.Period(gmst_span=(2080, 2099), impact_span=(2080, 2098)),
        ],
        "valued": [
            cilreg.Period(gmst_span=(2040, 2059), impact_span=(2040, 2059)),
            cilreg.Period(gmst_span=(2080, 2099), impact_span=(2080, 2099)),
        ],
    }
    WARMING = dict(
        reference_period=(1986, 2005),
        offset=0.61,
        levels=[1.5, 2.0, 3.0, 4.0, 5.0],
        half_width=0.25,
        smoothing="smoothed before it reached the table, how is not recorded",
    )
    gmst_table = cilreg.read_gmst(GMST, member_cols=["gcm", "rcp"])
    warming_levels = cilreg.assign_warming_levels(
        gmst_table, periods=PERIODS["physical"], **WARMING
    )
    _record = warming_levels.record()
    mo.vstack(
        [
            mo.md("Members at each level, by period:"),
            mo.ui.table(
                [
                    {"period": _p, **_counts}
                    for _p, _counts in _record["n_members"].items()
                ],
                selection=None,
            ),
            mo.md(f"Members at no level in any period: {_record['in_no_bin']}"),
        ]
    )
    return PERIODS, WARMING, gmst_table, warming_levels


@app.cell
def _(WARMING, gmst_table, plt, warming_levels):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _start, _end = WARMING["reference_period"]
    for _member, _series in gmst_table.series().items():
        _years = sorted(_series)
        _reference = sum(_series[_y] for _y in range(_start, _end + 1)) / (
            _end - _start + 1
        )
        _ax.plot(
            _years,
            [_series[_y] - _reference + WARMING["offset"] for _y in _years],
            linewidth=0.5,
            color="grey",
        )
    for _level in WARMING["levels"]:
        _ax.axhspan(
            _level - WARMING["half_width"], _level + WARMING["half_width"], alpha=0.12
        )
    for _period in warming_levels.periods:
        _ax.axvspan(*_period.gmst_span, color="tab:orange", alpha=0.1)
    _ax.set_ylabel("warming above pre-industrial (°C)")
    _ax.set_title("Each member's series, the bands of the levels, and the two periods")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    In a pipeline the same settings go under `[stats.warming_levels]`, in
    place of `stats.windows`, and a reduce takes them from the leaves
    already aggregated. The members are a model and an rcp, so both rcps
    are pooled. The paper's two families differ in two settings. The
    physical products average each draw over the period first and take
    the extraction tool's step. The valued products pool the years with
    the draws and take the linear quantile, over twenty years.
    """
    )
    return


@app.cell
def _(
    GMST,
    OUT,
    PERIODS,
    Path,
    WARMING,
    cilreg_cli,
    mo,
    pipeline_config,
    shown,
    tree_section,
):
    mo.stop(GMST is None)
    _configs = OUT / "configs"

    def warming_block(family):
        lines = [
            "",
            "[stats.warming_levels]",
            # resolved against the directory the pipeline runs in, like
            # the model weights, so written whole
            f'gmst = "{Path(GMST).resolve()}"',
            'member = ["gcm", "rcp"]',
            f"reference_period = {list(WARMING['reference_period'])}",
            f"offset = {WARMING['offset']}",
            f"levels = {WARMING['levels']}",
            f"half_width = {WARMING['half_width']}",
            f'smoothing = "{WARMING["smoothing"]}"',
        ]
        for _period in PERIODS[family]:
            lines += [
                "",
                "[[stats.warming_levels.periods]]",
                f"gmst_span = {list(_period.gmst_span)}",
                f"impact_span = {list(_period.impact_span)}",
            ]
        return "\n".join(lines) + "\n"

    _both = ["rcp45", "rcp85"]
    _common = 'sample_dims = ["batch", "rcp", "gcm"]\ntime_col = "year"\nquantiles = [0.05, 0.5, 0.95]\nsummation = "fast"\n'
    aggregate_damages = pipeline_config(
        "damages_aggregate",
        tree=tree_section("damages.parquet", _both),
        data='format = "parquet"\nvalue_col = "damages"\ndenominator_col = "gdp"\nkind = "ratio"\nversion = "world-combo-201710"\n',
        stats=_common
        + 'quantile_method = "linear"\ntime_reduction = "pooled"\nwindows = [[2080, 2099]]\n',
        output='stem = "damages"\nwrite_leaves = true\n',
        reduce=True,
    )
    WARMING_REDUCES = {
        "deaths_warming": pipeline_config(
            "deaths_warming",
            tree=tree_section(
                "deaths.parquet", _both, subtract="deaths-histclim.parquet"
            ),
            data='format = "parquet"\nvalue_col = "deaths"\nkind = "extensive"\nversion = "world-combo-201710"\n',
            stats=_common
            + 'quantile_method = "lower_step"\ntime_reduction = "window_mean"\n'
            + warming_block("physical"),
            output='stem = "deaths_warming"\nleaves_stem = "deaths"\n',
        ),
        "damages_warming": pipeline_config(
            "damages_warming",
            tree=tree_section("damages.parquet", _both),
            data='format = "parquet"\nvalue_col = "damages"\ndenominator_col = "gdp"\nkind = "ratio"\nversion = "world-combo-201710"\n',
            stats=_common
            + 'quantile_method = "linear"\ntime_reduction = "pooled"\n'
            + warming_block("valued"),
            output='stem = "damages_warming"\nleaves_stem = "damages"\n',
        ),
    }
    _printed = f"$ cilreg pipeline {shown(aggregate_damages)} --stage aggregate\n"
    _printed += shown(cilreg_cli("pipeline", aggregate_damages, "--stage", "aggregate"))
    for _name, _path in WARMING_REDUCES.items():
        _printed += f"\n$ cilreg pipeline {shown(_path)} --stage reduce\n"
        _printed += shown(cilreg_cli("pipeline", _path, "--stage", "reduce"))
    _shown_config = "\n".join(
        'gmst = "<the temperature table>"' if _line.startswith("gmst = ") else _line
        for _line in WARMING_REDUCES["deaths_warming"].read_text().splitlines()
    )
    mo.vstack(
        [
            mo.md("The physical reduce, with the path of the table left out:"),
            mo.md("```toml\n" + _shown_config + "\n```"),
            mo.md("```\n" + _printed + "```"),
        ]
    )
    return (WARMING_REDUCES,)


@app.cell
def _(GMST, PIPELINE, WARMING_REDUCES, department_map, mo, pd):
    mo.stop(GMST is None)
    warming_products = {
        _name: pd.read_parquet(PIPELINE / f"{_name}.parquet")
        for _name in WARMING_REDUCES
    }
    _deaths = warming_products["deaths_warming"]
    _at_two = _deaths[
        (_deaths["statistic"] == "q50")
        & (_deaths["warming_level"] == 2.0)
        & (_deaths["window"] == _deaths["window"].max())
    ]
    _members = int(_at_two["n_bin_members"].max()) if len(_at_two) else 0
    mo.vstack(
        [
            department_map(
                _at_two, "deaths", "Median deaths a year at 2°C, end of century"
            ),
            mo.md(
                f"""
    At the end of the century the number of members of the temperature
    table at 2°C is {_members}, `n_bin_members` in the file, and
    `n_members` counts their draws. A 95th percentile over so few models
    is read with that number beside it.
    """
            ),
            _deaths.head(),
        ]
    )
    return (warming_products,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Provenance

    A weights file comes with a manifest that says what made it. It holds
    the SHA256 of the config as validated, in canonical JSON, and of every
    input file. Both are worked out again below from the files on disk,
    and they match.
    """
    )
    return


@app.cell
def _(
    ADM2_MANIFEST_PATH, WEIGHT_CONFIGS, adm2_manifest, cilreg, hashlib, json, pd, shown
):
    _config = cilreg.load_config(WEIGHT_CONFIGS["adm2", "per_source"])
    _again = hashlib.sha256(
        json.dumps(_config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()

    def file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    _files = {
        "geometry:regions": _config.regions.path,
        "geometry:source": _config.source.path,
        "weight:pop": _config.weights[0].raster,
    }
    _rows = [
        (
            "config",
            shown(WEIGHT_CONFIGS["adm2", "per_source"]),
            adm2_manifest["config_hash"],
            _again,
        )
    ]
    _rows += [
        (_key, shown(_path), adm2_manifest["inputs"][_key], file_sha256(_path))
        for _key, _path in _files.items()
    ]
    provenance = pd.DataFrame(
        _rows, columns=["what", "file", "manifest says", "worked out again"]
    )
    provenance["match"] = provenance["manifest says"] == provenance["worked out again"]
    assert provenance["match"].all(), provenance
    assert ADM2_MANIFEST_PATH.exists()
    provenance
    return (file_sha256,)


@app.cell
def _(
    PIPELINE, REDUCES, cilreg_cli, file_sha256, json, mo, shown, without_machine_paths
):
    _manifest = json.loads((PIPELINE / "deaths_window_mean.manifest.json").read_text())
    _before = file_sha256(PIPELINE / "deaths_window_mean.parquet")
    _printed = f"$ cilreg pipeline {shown(REDUCES['window_mean'])} --stage reduce --verify-leaves\n"
    _printed += shown(
        cilreg_cli(
            "pipeline", REDUCES["window_mean"], "--stage", "reduce", "--verify-leaves"
        )
    )
    _after = file_sha256(PIPELINE / "deaths_window_mean.parquet")
    assert _before == _after
    mo.vstack(
        [
            mo.md(
                """
    The statistics file of a pipeline run has a manifest of its own. It
    records the weights by their SHA256, every setting of `[stats]`, the
    model weights each model was given, and the leaves it read. A reduce
    can hash every leaf against the leaves manifest first, with
    `--verify-leaves`, and the statistics it writes are the same bytes.
    """
            ),
            mo.md("```\n" + _printed + "```"),
            {
                _key: without_machine_paths(_manifest[_key])
                for _key in (
                    "artifact_sha256",
                    "quantile_method",
                    "summation",
                    "time_reduction",
                    "model_weights_record",
                    "stage",
                )
                if _key in _manifest
            },
        ]
    )
    return


if __name__ == "__main__":
    app.run()
