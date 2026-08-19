"""End-to-end pipeline tests over a small synthetic tree.

The full chain runs for real: polygon-mode weight generation writes an
artifact to disk, a synthetic Monte Carlo tree is laid out with a
declared (and deliberately unrealistic) grammar, the pipeline applies
the weights per leaf with mass balance, and the statistics stage
summarizes the pooled samples.

Geometry: two targets T1, T2; two source units, u1 nested in T1 and u2
straddling T1/T2 half and half, so every leaf's target values are
hand-derivable from the leaf's unit values.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from cil_regionalization.backends.local import LocalBackend
from cil_regionalization.config import Config
from cil_regionalization.io import write_result
from cil_regionalization.pipelines.montecarlo import (
    MonteCarloPipelineConfig,
    PipelinePlan,
    load_pipeline_config,
    run_pipeline,
)
from cil_regionalization.regions import RegionSet
from cil_regionalization.weights import from_config_list

SOURCE_VERSION = "units-v1"


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    """Generate and write a per_source area-weight artifact for real."""
    targets = gpd.GeoDataFrame(
        {
            "target_id": ["T1", "T2"],
            "geometry": [box(0, 0, 4, 4), box(4, 0, 8, 4)],
        },
        crs="EPSG:4326",
    )
    units = gpd.GeoDataFrame(
        {
            "unit_id": ["u1", "u2"],
            "geometry": [box(1, 1, 2, 2), box(3.5, 2.5, 4.5, 3.5)],
        },
        crs="EPSG:4326",
    )
    tp, sp = tmp_path / "targets.parquet", tmp_path / "units.parquet"
    targets.to_parquet(tp)
    units.to_parquet(sp)
    out = tmp_path / "weights_out"
    cfg = Config.model_validate(
        {
            "project": {"name": "pipeline_weights"},
            "regions": {
                "path": str(tp),
                "id_fields": ["target_id"],
                "version": "targets-v1",
            },
            "source": {
                "path": str(sp),
                "id_fields": ["unit_id"],
                "version": SOURCE_VERSION,
            },
            "weights": [{"name": "area"}],
            "backend": {"kind": "local"},
            "output": {"dir": str(out)},
            "normalization": "per_source",
        }
    )
    result = LocalBackend().compute(
        RegionSet.from_config(cfg.regions), None, from_config_list(cfg.weights), cfg
    )
    write_result(result, str(out), "parquet")
    return out


def _build_tree(
    root: Path,
    levels: dict[str, list[str]],
    filename: str = "damages.parquet",
) -> None:
    """Write one leaf per cross-product combination, values derived from
    the combination index so every leaf differs."""
    for i, combo in enumerate(itertools.product(*levels.values())):
        leaf_dir = root.joinpath(*combo)
        leaf_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for unit, year in itertools.product(["u1", "u2"], [2050, 2051, 2080, 2081]):
            base = 100.0 if unit == "u1" else 40.0
            rows.append(
                {
                    "unit_id": unit,
                    "year": year,
                    "value": base + 10.0 * i + (year - 2050),
                }
            )
        pd.DataFrame(rows).to_parquet(leaf_dir / filename, index=False)


def _pipeline_cfg(
    weights_dir: Path,
    tree_root: Path,
    out_dir: Path,
    *,
    levels: dict[str, list[str]],
    n_workers: int = 1,
) -> MonteCarloPipelineConfig:
    return MonteCarloPipelineConfig.model_validate(
        {
            "weights": {"dir": str(weights_dir), "weight": "area"},
            "tree": {
                "root": str(tree_root),
                "levels": list(levels),
                "filename": "damages.parquet",
                "expect": levels,
            },
            "data": {
                "kind": "extensive",
                "version": SOURCE_VERSION,
                "value_col": "value",
            },
            "stats": {
                "sample_dims": list(levels),
                "time_col": "year",
                "windows": [[2050, 2051], [2080, 2081]],
                "quantiles": [0.05, 0.95],
            },
            "output": {"dir": str(out_dir)},
            "run": {"n_workers": n_workers},
        }
    )


_LEVELS = {"kumquat": ["k0", "k1"], "zeppelin": ["z0", "z1", "z2"]}


class TestEndToEnd:
    def test_generation_application_statistics(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        result = run_pipeline(cfg)

        # Every leaf passed coverage and mass balance.
        assert len(result.leaf_reports) == 6
        for report in result.leaf_reports:
            assert report["error"] is None
            assert report["coverage"] == {
                "unmatched": 0,
                "zero_weight": 0,
                "absent_from_data": 0,
            }
            assert report["mass_balance_rel_error"] == pytest.approx(0.0, abs=1e-12)

        # Statistics: 2 targets x 2 windows x 3 statistics.
        stats = result.stats
        assert set(stats.columns) == {"target_id", "statistic", "value", "window"}
        assert len(stats) == 12
        assert set(stats["window"]) == {"2050-2051", "2080-2081"}

        # Hand check one number: u2 splits half and half at lon 4, so T2's
        # value per (leaf i, year) is 0.5 * (40 + 10 i + (year - 2050)).
        # Window 2050-2051 mean per leaf: 0.5 * (40 + 10 i + 0.5); pooled
        # mean over i = 0..5 is 0.5 * (40 + 25 + 0.5).
        t2_mean = stats.query(
            "target_id == 'T2' and statistic == 'mean' and window == '2050-2051'"
        )["value"].iloc[0]
        assert t2_mean == pytest.approx(0.5 * (40.0 + 25.0 + 0.5))

        # Per-leaf outputs are mirrored under the output dir.
        leaf_out = tmp_path / "out" / "leaves" / "k0" / "z0" / "aggregated_damages.parquet"
        assert leaf_out.exists()
        manifest = json.loads((tmp_path / "out" / "statistics.manifest.json").read_text())
        assert manifest["n_leaves"] == 6
        assert manifest["artifact_source_version"] == SOURCE_VERSION

    def test_parallel_run_matches_serial(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        serial = run_pipeline(
            _pipeline_cfg(weights_dir, tree, tmp_path / "out_s", levels=_LEVELS)
        )
        parallel = run_pipeline(
            _pipeline_cfg(
                weights_dir, tree, tmp_path / "out_p", levels=_LEVELS, n_workers=2
            )
        )
        key = ["target_id", "statistic", "window"]
        pd.testing.assert_frame_equal(
            serial.stats.sort_values(key).reset_index(drop=True),
            parallel.stats.sort_values(key).reset_index(drop=True),
        )


class TestConfigurableTree:
    def test_three_level_layout_with_other_names(self, weights_dir, tmp_path):
        """Not batch/rcp/gcm/iam/ssp: three levels, different names and
        depth, one of them a passthrough identity dimension rather than
        a sample dimension."""
        levels = {
            "version": ["v1"],
            "flavor": ["sweet", "sour"],
            "draw_tag": ["d0", "d1"],
        }
        tree = tmp_path / "tree3"
        _build_tree(tree, levels)
        cfg = MonteCarloPipelineConfig.model_validate(
            {
                "weights": {"dir": str(weights_dir), "weight": "area"},
                "tree": {
                    "root": str(tree),
                    "levels": list(levels),
                    "filename": "damages.parquet",
                    "expect": levels,
                },
                "data": {"kind": "extensive", "version": SOURCE_VERSION},
                "stats": {
                    # flavor is identity, not sample: statistics come out
                    # per (target, flavor).
                    "sample_dims": ["draw_tag"],
                    "time_col": "year",
                    "windows": [[2050, 2051]],
                    "quantiles": [0.5],
                },
                "output": {"dir": str(tmp_path / "out3")},
            }
        )
        result = run_pipeline(cfg)
        stats = result.stats
        assert set(stats.columns) == {
            "target_id",
            "version",
            "flavor",
            "statistic",
            "value",
            "window",
        }
        assert set(stats["flavor"]) == {"sweet", "sour"}

    def test_expect_must_match_levels(self, tmp_path):
        from cil_regionalization.pipelines.montecarlo import TreeConfig

        with pytest.raises(ValueError, match="must match"):
            TreeConfig.model_validate(
                {
                    "root": str(tmp_path),
                    "levels": ["only_one"],
                    "filename": "damages.parquet",
                    "expect": _LEVELS,
                }
            )


class TestMissingLeaf:
    def test_missing_declared_leaf_fails_before_processing(
        self, weights_dir, tmp_path
    ):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        removed = tree / "k1" / "z2" / "damages.parquet"
        removed.unlink()
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        with pytest.raises(ValueError, match="missing on disk"):
            run_pipeline(cfg)
        # Nothing was produced: the failure came before any leaf ran.
        assert not (tmp_path / "out").exists()

    def test_failing_leaf_reports_partial_and_writes_no_statistics(
        self, weights_dir, tmp_path
    ):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        # Poison one leaf with a unit the weights do not know: the
        # default unmatched policy makes that leaf fail.
        bad = pd.read_parquet(tree / "k0" / "z1" / "damages.parquet")
        bad.loc[len(bad)] = {"unit_id": "ghost", "year": 2050, "value": 1.0}
        bad.to_parquet(tree / "k0" / "z1" / "damages.parquet", index=False)

        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        with pytest.raises(ValueError, match="partial run") as exc:
            run_pipeline(cfg)
        assert "5 of 6 leaves succeeded" in str(exc.value)
        assert "k0/z1" in str(exc.value)
        assert not (tmp_path / "out" / "statistics.parquet").exists()


class TestDryRun:
    def test_plan_reports_and_touches_nothing(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        out = tmp_path / "out"
        cfg = _pipeline_cfg(weights_dir, tree, out, levels=_LEVELS)
        plan = run_pipeline(cfg, dry_run=True)
        assert isinstance(plan, PipelinePlan)
        assert len(plan.leaves) == 6
        assert plan.missing == []
        assert "6 declared leaves" in plan.summary()
        assert not out.exists()

    def test_dry_run_reports_missing_without_raising(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        (tree / "k0" / "z0" / "damages.parquet").unlink()
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        plan = run_pipeline(cfg, dry_run=True)
        assert len(plan.missing) == 1
        assert "MISSING 1" in plan.summary()
        assert not (tmp_path / "out").exists()


class TestCliEntryPoint:
    def test_toml_dry_run_then_run(self, weights_dir, tmp_path, capsys):
        from cil_regionalization.pipelines.montecarlo import main

        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg_path = tmp_path / "pipeline.toml"
        cfg_path.write_text(
            f"""
[weights]
dir = "{weights_dir}"
weight = "area"

[tree]
root = "{tree}"
levels = ["kumquat", "zeppelin"]
filename = "damages.parquet"

[tree.expect]
kumquat = ["k0", "k1"]
zeppelin = ["z0", "z1", "z2"]

[data]
kind = "extensive"
version = "{SOURCE_VERSION}"

[stats]
sample_dims = ["kumquat", "zeppelin"]
time_col = "year"
windows = [[2050, 2051]]
quantiles = [0.5]

[output]
dir = "{tmp_path / 'out'}"
"""
        )
        rc = main([str(cfg_path), "--dry-run"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "6 declared leaves" in captured.out
        assert not (tmp_path / "out").exists()

        rc = main([str(cfg_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "6 leaves aggregated" in captured.out
        assert (tmp_path / "out" / "statistics.parquet").exists()

    def test_relative_paths_resolve_against_config(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg_path = tmp_path / "rel.toml"
        cfg_path.write_text(
            f"""
[weights]
dir = "{weights_dir.name}"
weight = "area"

[tree]
root = "tree"
levels = ["kumquat", "zeppelin"]
filename = "damages.parquet"

[tree.expect]
kumquat = ["k0", "k1"]
zeppelin = ["z0", "z1", "z2"]

[data]
kind = "extensive"
version = "{SOURCE_VERSION}"

[stats]
sample_dims = ["kumquat", "zeppelin"]
time_col = "year"
windows = [[2050, 2051]]

[output]
dir = "out_rel"
"""
        )
        cfg = load_pipeline_config(cfg_path)
        assert cfg.tree.root == str(tree)
        assert cfg.output.dir == str(tmp_path / "out_rel")


class TestNoSiteFactsInCode:
    def test_pipeline_code_carries_no_absolute_paths_or_environments(self):
        import cil_regionalization.pipelines.montecarlo as module

        source = Path(module.__file__).read_text()
        for forbidden in (
            "/project/",
            "/Volumes/",
            "/shares/",
            "/home/",
            "conda",
            "sbatch",
            "battuta",
            "caslake",
        ):
            assert forbidden not in source, (
                f"pipeline code contains site-specific string {forbidden!r}"
            )


def _build_netcdf_tree(
    root: Path,
    levels: dict[str, list[str]],
    filename: str = "damages.nc4",
    region_dim: str = "spatial_thing",
) -> None:
    """NetCDF twin of `_build_tree`: same values, region dim under a
    non-standard name so nothing can assume 'region'."""
    xr = pytest.importorskip("xarray")
    for i, combo in enumerate(itertools.product(*levels.values())):
        leaf_dir = root.joinpath(*combo)
        leaf_dir.mkdir(parents=True, exist_ok=True)
        years = [2050, 2051, 2080, 2081]
        values = [
            [100.0 + 10.0 * i + (y - 2050) for y in years],  # u1
            [40.0 + 10.0 * i + (y - 2050) for y in years],  # u2
        ]
        ds = xr.Dataset(
            {"value": ([region_dim, "year"], values)},
            coords={region_dim: ["u1", "u2"], "year": years},
        )
        ds.to_netcdf(leaf_dir / filename)


class TestNetcdfVariant:
    def test_end_to_end_matches_parquet_pipeline(self, weights_dir, tmp_path):
        pytest.importorskip("xarray")
        pq_tree = tmp_path / "tree_pq"
        nc_tree = tmp_path / "tree_nc"
        _build_tree(pq_tree, _LEVELS)
        _build_netcdf_tree(nc_tree, _LEVELS)

        pq_result = run_pipeline(
            _pipeline_cfg(weights_dir, pq_tree, tmp_path / "out_pq", levels=_LEVELS)
        )

        nc_cfg = MonteCarloPipelineConfig.model_validate(
            {
                "weights": {"dir": str(weights_dir), "weight": "area"},
                "tree": {
                    "root": str(nc_tree),
                    "levels": list(_LEVELS),
                    "filename": "damages.nc4",
                    "expect": _LEVELS,
                },
                "data": {
                    "format": "netcdf",
                    "kind": "extensive",
                    "version": SOURCE_VERSION,
                    "region_dim": "spatial_thing",
                },
                "stats": {
                    "sample_dims": list(_LEVELS),
                    "time_col": "year",
                    "windows": [[2050, 2051], [2080, 2081]],
                    "quantiles": [0.05, 0.95],
                },
                "output": {"dir": str(tmp_path / "out_nc")},
            }
        )
        nc_result = run_pipeline(nc_cfg)

        key = ["target_id", "statistic", "window"]
        pd.testing.assert_frame_equal(
            pq_result.stats.sort_values(key).reset_index(drop=True),
            nc_result.stats.sort_values(key).reset_index(drop=True),
        )


class TestSegweightsPipelineVerb:
    def test_dry_run_then_run_via_cilreg(self, weights_dir, tmp_path, capsys):
        from cil_regionalization.cli import main as cilreg_main

        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg_path = tmp_path / "pipeline.toml"
        cfg_path.write_text(
            f"""
[weights]
dir = "{weights_dir}"
weight = "area"

[tree]
root = "{tree}"
levels = ["kumquat", "zeppelin"]
filename = "damages.parquet"

[tree.expect]
kumquat = ["k0", "k1"]
zeppelin = ["z0", "z1", "z2"]

[data]
kind = "extensive"
version = "{SOURCE_VERSION}"

[stats]
sample_dims = ["kumquat", "zeppelin"]
time_col = "year"
windows = [[2050, 2051]]
quantiles = [0.5]

[output]
dir = "{tmp_path / 'out'}"
"""
        )
        rc = cilreg_main(["pipeline", str(cfg_path), "--dry-run"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "6 declared leaves" in captured.out
        assert not (tmp_path / "out").exists()

        rc = cilreg_main(["pipeline", str(cfg_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "6 leaves aggregated" in captured.out
        assert (tmp_path / "out" / "statistics.parquet").exists()


class TestWindowReductionPerLeaf:
    """Leaves hand back window means, not year series; the on-disk
    per-leaf output keeps its years."""

    def test_reduced_rows_and_disk_output(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        result = run_pipeline(
            _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        )
        for report in result.leaf_reports:
            # 2 targets x 4 years aggregated; 2 targets x 2 windows pooled.
            assert report["rows_out"] == 8
            assert report["rows_reduced"] == 4
        on_disk = pd.read_parquet(
            tmp_path / "out" / "leaves" / "k0" / "z0" / "aggregated_damages.parquet"
        )
        assert "year" in on_disk.columns
        assert len(on_disk) == 8


def _build_subtract_tree(root: Path, levels: dict[str, list[str]]) -> None:
    """A histclim sibling next to every leaf: value minus 7, so the
    difference is exactly 7 everywhere and stats are hand-derivable."""
    _build_tree(root, levels)
    for combo in itertools.product(*levels.values()):
        leaf_dir = root.joinpath(*combo)
        df = pd.read_parquet(leaf_dir / "damages.parquet")
        df["value"] = df["value"] - 7.0
        df.to_parquet(leaf_dir / "histclim.parquet", index=False)


class TestSubtract:
    def test_difference_flows_through_to_statistics(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_subtract_tree(tree, _LEVELS)
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["tree"]["subtract_filename"] = "histclim.parquet"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)
        # value - (value - 7) = 7 per unit-year; u1 wholly in T1 and half
        # of u2 also lands in T1, so every statistic in T1 is 7 * 1.5
        t1 = result.stats[result.stats["target_id"] == "T1"]
        assert (t1["value"].round(9) == 10.5).all()

    def test_missing_subtract_file_counts_as_missing(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)  # no histclim siblings
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["tree"]["subtract_filename"] = "histclim.parquet"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        plan = run_pipeline(cfg, dry_run=True)
        assert len(plan.missing) == len(plan.leaves)

    def test_subtracting_the_same_file_is_rejected(self, weights_dir, tmp_path):
        cfg = _pipeline_cfg(weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["tree"]["subtract_filename"] = "damages.parquet"
        with pytest.raises(ValueError, match="always zero"):
            MonteCarloPipelineConfig.model_validate(raw)


class TestExclude:
    def test_excluded_cell_is_skipped_not_missing(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        import shutil
        shutil.rmtree(tree / "k1" / "z2")  # absent on disk
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["tree"]["exclude"] = [["k1", "z2"]]
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)
        assert len(result.leaf_reports) == 5

    def test_exclude_of_undeclared_value_is_rejected(self, weights_dir, tmp_path):
        cfg = _pipeline_cfg(weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["tree"]["exclude"] = [["k1", "z9"]]
        with pytest.raises(ValueError, match="not a declared value"):
            MonteCarloPipelineConfig.model_validate(raw)


class TestModelWeights:
    def _weights_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "weights.tsv"
        pd.DataFrame(
            {"model": ["Z0*", "z1", "z2"], "weight": [0.6, 0.3, 0.1]}
        ).to_csv(p, sep="\t", index=False)
        return p

    def test_weighted_mean_uses_the_weights(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["stats"]["model_weights"] = str(self._weights_file(tmp_path))
        raw["stats"]["model_weight_level"] = "zeppelin"
        raw["stats"]["quantiles"] = []
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)

        unweighted = run_pipeline(
            _pipeline_cfg(weights_dir, tree, tmp_path / "out2", levels=_LEVELS)
        )
        merged = result.stats.merge(
            unweighted.stats[unweighted.stats["statistic"] == "mean"],
            on=["target_id", "window", "statistic"],
            suffixes=("_w", "_u"),
        )
        means = merged[merged["statistic"] == "mean"]
        assert (means["value_w"] != means["value_u"]).any()
        manifest = json.loads(
            (Path(result.paths["manifest"])).read_text()
        )
        assert manifest["model_weight_level"] == "zeppelin"

    def test_unmatched_model_is_rejected(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        p = tmp_path / "weights.tsv"
        pd.DataFrame({"model": ["z0"], "weight": [1.0]}).to_csv(p, sep="\t", index=False)
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["stats"]["model_weights"] = str(p)
        raw["stats"]["model_weight_level"] = "zeppelin"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        with pytest.raises(ValueError, match="no row in the model"):
            run_pipeline(cfg)

    def test_weight_level_must_be_sample_dim(self, weights_dir, tmp_path):
        cfg = _pipeline_cfg(weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["stats"]["model_weights"] = "x.tsv"
        raw["stats"]["model_weight_level"] = "nope"
        with pytest.raises(ValueError, match="sample_dims"):
            MonteCarloPipelineConfig.model_validate(raw)


class TestOutputNaming:
    def test_stem_and_reduced_frame(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["output"]["stem"] = "mortality_physical_rcpX"
        raw["output"]["write_reduced"] = True
        raw["output"]["write_leaves"] = False
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)
        out = tmp_path / "out"
        assert (out / "mortality_physical_rcpX.parquet").exists()
        assert (out / "mortality_physical_rcpX.manifest.json").exists()
        reduced = pd.read_parquet(out / "mortality_physical_rcpX.reduced.parquet")
        # one row per target x window x sample member
        assert len(reduced) == 2 * 2 * 6
        assert not (out / "leaves").exists()


class TestFailureReport:
    def test_identical_failures_reported_once_with_count(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        # poison every leaf identically: wrong value column
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["data"]["value_col"] = "not_a_column"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        with pytest.raises(ValueError) as err:
            run_pipeline(cfg)
        message = str(err.value)
        assert "6 failed with 1 distinct errors" in message
        # the message stays bounded: one description, not one per leaf
        assert message.count("not_a_column") <= 2
