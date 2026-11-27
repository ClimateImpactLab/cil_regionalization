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

import hashlib
import itertools
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
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
                "quantile_method": "linear",
                "summation": "fast",
                "time_reduction": "window_mean",
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
        assert set(stats.columns) == {
            "target_id",
            "statistic",
            "value",
            "window",
            "weight_share",
        }
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

        # No leaf is kept unless the config asks for it.
        assert not (tmp_path / "out" / "leaves").exists()
        manifest = json.loads(
            (tmp_path / "out" / "statistics.manifest.json").read_text()
        )
        assert manifest["n_leaves"] == 6
        assert manifest["artifact_source_version"] == SOURCE_VERSION

    def test_include_n_members_adds_the_count(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        raw = _pipeline_cfg(
            weights_dir, tree, tmp_path / "out", levels=_LEVELS
        ).model_dump()
        raw["stats"]["include_n_members"] = True
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

        assert set(result.stats.columns) == {
            "target_id",
            "statistic",
            "value",
            "window",
            "n_members",
            "weight_share",
        }
        # 2 kumquat x 3 zeppelin leaves, one member each.
        written = pd.read_parquet(result.paths["statistics"])
        assert (written["n_members"] == 6).all()

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


class TestStatisticMethods:
    """The experimental `stats.quantile_method` and `stats.summation`."""

    def _cfg(self, weights_dir, tree, out, n_workers=1, **stats):
        raw = _pipeline_cfg(
            weights_dir, tree, out, levels=_LEVELS, n_workers=n_workers
        ).model_dump()
        raw["stats"].update(stats)
        raw["output"]["write_reduced"] = True
        return MonteCarloPipelineConfig.model_validate(raw)

    @staticmethod
    def _table(tmp_path: Path, weights=(0.6, 0.3, 0.1)) -> dict:
        table = tmp_path / "model_weights.tsv"
        table.write_text(
            "quantile\tmodel\tweight\n"
            + "".join(f"1\tz{i}\t{w}\n" for i, w in enumerate(weights))
        )
        return dict(model_weights=str(table), model_weight_level="zeppelin")

    def test_the_manifest_records_both_options(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        result = run_pipeline(self._cfg(weights_dir, tree, tmp_path / "out"))
        manifest = json.loads(Path(result.paths["manifest"]).read_text())
        assert manifest["quantile_method"] == "linear"
        assert manifest["summation"] == "fast"

        weighted = run_pipeline(
            self._cfg(
                weights_dir,
                tree,
                tmp_path / "weighted",
                quantiles=[0.5, 0.95],
                quantile_method="lower_step",
                summation="exact",
                **self._table(tmp_path),
            )
        )
        manifest = json.loads(Path(weighted.paths["manifest"]).read_text())
        assert manifest["quantile_method"] == "lower_step"
        assert manifest["summation"] == "exact"

        for name in ("linear", "lower_step", "inverted_cdf"):
            named = run_pipeline(
                self._cfg(
                    weights_dir,
                    tree,
                    tmp_path / name,
                    quantile_method=name,
                    quantiles=[0.5, 0.95],
                )
            )
            manifest = json.loads(Path(named.paths["manifest"]).read_text())
            assert manifest["quantile_method"] == name

    def test_lower_step_without_weights_is_the_step_with_every_weight_one(
        self, weights_dir, tmp_path
    ):
        """What the extraction tool gave when told to use no model
        weights. A table of ones is the same step, and "linear"
        interpolates."""
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        quantiles = dict(quantiles=[0.5, 0.75, 0.95])
        step = run_pipeline(
            self._cfg(
                weights_dir,
                tree,
                tmp_path / "step",
                quantile_method="lower_step",
                **quantiles,
            )
        )
        ones = run_pipeline(
            self._cfg(
                weights_dir,
                tree,
                tmp_path / "ones",
                quantile_method="lower_step",
                **quantiles,
                **self._table(tmp_path, weights=(1, 1, 1)),
            )
        )
        pd.testing.assert_frame_equal(step.stats, ones.stats, check_exact=True)
        assert (
            step.stats["value"].to_numpy().tobytes()
            == ones.stats["value"].to_numpy().tobytes()
        )
        linear = run_pipeline(
            self._cfg(
                weights_dir,
                tree,
                tmp_path / "linear",
                quantile_method="linear",
                **quantiles,
            )
        )
        high = step.stats["statistic"].isin(["q75", "q95"]).to_numpy()
        assert (step.stats.loc[high, "value"] < linear.stats.loc[high, "value"]).all()

    @pytest.mark.parametrize("left_out", ["quantile_method", "summation"])
    def test_a_config_has_to_name_both_options(self, weights_dir, tmp_path, left_out):
        raw = _pipeline_cfg(
            weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS
        ).model_dump()
        del raw["stats"][left_out]
        with pytest.raises(ValueError, match=f"stats.{left_out}"):
            MonteCarloPipelineConfig.model_validate(raw)

    def test_a_config_that_names_the_former_value_is_told_which_to_use(
        self, weights_dir, tmp_path
    ):
        """Both options were once "legacy". For a quantile it meant one
        of two computations, by whether the run had model weights."""
        args = (weights_dir, tmp_path / "t", tmp_path / "o")
        with pytest.raises(ValueError, match="no weights, so use 'linear'"):
            self._cfg(*args, quantile_method="legacy")
        with pytest.raises(ValueError, match="weights, so use 'lower_step'"):
            self._cfg(*args, quantile_method="legacy", **self._table(tmp_path))
        with pytest.raises(ValueError, match="summation='legacy'.*Use 'fast'"):
            self._cfg(*args, summation="legacy")

    def test_linear_with_model_weights_is_refused_when_the_config_is_read(
        self, weights_dir, tmp_path
    ):
        with pytest.raises(ValueError, match="has no weighted form"):
            self._cfg(
                weights_dir,
                tmp_path / "t",
                tmp_path / "o",
                quantile_method="linear",
                **self._table(tmp_path),
            )

    def test_the_options_reach_both_stages_and_the_manifest(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """Under "inverted_cdf" q05 and q95 are the smallest and the
        largest of the six members, where the default interpolates. A
        two-year window sums the same either way, so that the leaves
        were reduced with "exact" is read off the calls."""
        from cil_regionalization.pipelines import montecarlo

        real, asked = montecarlo.window_means, []

        def window_means(*args, **kwargs):
            asked.append(kwargs.get("summation"))
            return real(*args, **kwargs)

        monkeypatch.setattr(montecarlo, "window_means", window_means)
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        out = tmp_path / "out"
        result = run_pipeline(
            self._cfg(
                weights_dir,
                tree,
                out,
                quantile_method="inverted_cdf",
                summation="exact",
            )
        )
        assert asked == ["exact"] * 6
        monkeypatch.undo()
        manifest = json.loads(Path(result.paths["manifest"]).read_text())
        assert manifest["quantile_method"] == "inverted_cdf"
        assert manifest["summation"] == "exact"

        members = pd.read_parquet(out / "statistics.reduced.parquet")
        got = result.stats.set_index(["target_id", "window", "statistic"])["value"]
        for (target, window), group in members.groupby(["target_id", "window"]):
            values = group["value"].tolist()
            assert len(values) == 6
            assert got[target, window, "q05"] == min(values)
            assert got[target, window, "q95"] == max(values)
            assert got[target, window, "mean"] == math.fsum(values) / 6

        linear = run_pipeline(self._cfg(weights_dir, tree, tmp_path / "out_l"))
        both = result.stats.merge(
            linear.stats,
            on=["target_id", "window", "statistic"],
            suffixes=("", "_linear"),
        )
        quantiles = both[both["statistic"] != "mean"]
        assert (quantiles["value"] != quantiles["value_linear"]).all()

    def test_serial_and_parallel_runs_agree_to_the_bit(self, weights_dir, tmp_path):
        """Compared without sorting. This can pass by luck without the
        options. The tests that fail without them are in
        tests/unit/test_stats_methods.py and in TestRowOrder."""
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        options = dict(quantile_method="inverted_cdf", summation="exact")
        serial = run_pipeline(
            self._cfg(weights_dir, tree, tmp_path / "out_s", **options)
        )
        parallel = run_pipeline(
            self._cfg(weights_dir, tree, tmp_path / "out_p", n_workers=2, **options)
        )
        pd.testing.assert_frame_equal(serial.stats, parallel.stats, check_exact=True)
        pd.testing.assert_frame_equal(
            pd.read_parquet(serial.paths["statistics"]),
            pd.read_parquet(parallel.paths["statistics"]),
            check_exact=True,
        )

    @pytest.mark.parametrize(
        "option, value",
        [("quantile_method", "interpolated"), ("summation", "kahan")],
    )
    def test_unknown_names_are_refused_by_the_config(
        self, weights_dir, tmp_path, option, value
    ):
        """Cannot go red by reverting, because without the two fields
        the config refuses the keys themselves. It failed with the
        fields typed as plain strings, which let any name through."""
        with pytest.raises(ValueError, match=option):
            self._cfg(weights_dir, tmp_path / "t", tmp_path / "o", **{option: value})


class TestRowOrder:
    """The statistics rows are sorted before they are written, under
    every setting."""

    # declared out of alphabetical and chronological order on purpose
    _ORDER_LEVELS = {"kumquat": ["k1", "k0"], "zeppelin": ["z2", "z0", "z1"]}
    _WINDOWS = [[2080, 2081], [2050, 2051]]
    _QUANTILES = [0.95, 0.05]

    def _cfg(self, weights_dir, tree, out, **stats):
        raw = _pipeline_cfg(
            weights_dir, tree, out, levels=self._ORDER_LEVELS
        ).model_dump()
        # zeppelin is not pooled, so it stays a column of the statistics
        raw["stats"].update(
            sample_dims=["kumquat"],
            windows=self._WINDOWS,
            quantiles=self._QUANTILES,
            **stats,
        )
        return MonteCarloPipelineConfig.model_validate(raw)

    @pytest.mark.parametrize(
        "options",
        [{}, {"quantile_method": "inverted_cdf", "summation": "exact"}],
    )
    def test_rows_follow_the_config_not_the_arrival(
        self, weights_dir, tmp_path, monkeypatch, options
    ):
        """Both files come in the order of `_ordered`. The pool here
        runs in this process and hands the leaves back last first."""
        import types

        from cil_regionalization.pipelines import montecarlo

        class LastFirstPool:
            def __init__(self, processes, initializer, initargs, maxtasksperchild):
                initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def imap_unordered(self, func, tasks, chunksize=1):
                return [func(task) for task in reversed(list(tasks))]

        monkeypatch.setattr(montecarlo, "mp", types.SimpleNamespace(Pool=LastFirstPool))
        tree = tmp_path / "tree"
        _build_tree(tree, self._ORDER_LEVELS)
        raw = self._cfg(
            weights_dir, tree, tmp_path / "out", include_n_members=True, **options
        ).model_dump()
        raw["run"] = {"n_workers": 2}
        raw["output"]["write_reduced"] = True
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

        windows, targets = ["2080-2081", "2050-2051"], ["T1", "T2"]
        expected = list(
            itertools.product(
                ["mean", "q95", "q05"], ["z2", "z0", "z1"], windows, targets
            )
        )
        columns = ["statistic", "zeppelin", "window", "target_id"]
        for frame in (result.stats, pd.read_parquet(result.paths["statistics"])):
            assert list(frame[columns].itertuples(index=False, name=None)) == expected
            assert list(frame.index) == list(range(len(expected)))

        reduced = pd.read_parquet(tmp_path / "out" / "statistics.reduced.parquet")
        expected = list(
            itertools.product(["k1", "k0"], ["z2", "z0", "z1"], windows, targets)
        )
        columns = ["kumquat", "zeppelin", "window", "target_id"]
        assert list(reduced[columns].itertuples(index=False, name=None)) == expected

    def test_a_value_the_config_does_not_declare_is_an_error(
        self, weights_dir, tmp_path
    ):
        from cil_regionalization.pipelines.montecarlo import _ordered

        tree = tmp_path / "tree"
        _build_tree(tree, self._ORDER_LEVELS)
        cfg = self._cfg(weights_dir, tree, tmp_path / "out")
        stats = run_pipeline(cfg).stats
        pd.testing.assert_frame_equal(
            _ordered(stats.sample(frac=1.0, random_state=1), cfg), stats
        )
        # dictionary-encoded columns, as a passthrough run pools them
        encoded = stats.sample(frac=1.0, random_state=2).astype(
            {"window": "category", "zeppelin": "category", "target_id": "category"}
        )
        pd.testing.assert_frame_equal(
            _ordered(encoded, cfg).astype(
                {"window": str, "zeppelin": str, "target_id": str}
            ),
            stats,
        )
        stats.loc[0, "window"] = "1999"
        with pytest.raises(ValueError, match="does not declare"):
            _ordered(stats, cfg)


def _through_parquet(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame.to_parquet(path, index=False)
    return pd.read_parquet(path)


class TestLabelPlacement:
    """The level labels go on the reduced rows and not on the leaf, and
    no file changes for it."""

    _WINDOWS = [(2050, 2051), (2080, 2081)]

    @staticmethod
    def _build_tree(root: Path) -> None:
        """Leaves with a dimension of their own next to the year."""
        for i, combo in enumerate(itertools.product(*_LEVELS.values())):
            leaf_dir = root.joinpath(*combo)
            leaf_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "unit_id": unit,
                    "band": band,
                    "year": year,
                    "value": (base + 10.0 * i + (year - 2050)) / 7.0 + shift,
                }
                for unit, base in (("u1", 100.0), ("u2", 40.0))
                for band, shift in (("old", 0.3), ("young", 0.0))
                for year in (2050, 2051, 2080, 2081)
            ]
            pd.DataFrame(rows).to_parquet(leaf_dir / "damages.parquet", index=False)

    @staticmethod
    def _cfg(weights_dir, tree, out) -> MonteCarloPipelineConfig:
        raw = _pipeline_cfg(weights_dir, tree, out, levels=_LEVELS).model_dump()
        # kumquat is not pooled, so it stays a column of the statistics
        raw["stats"]["sample_dims"] = ["zeppelin"]
        raw["output"]["write_reduced"] = True
        return MonteCarloPipelineConfig.model_validate(raw)

    def test_the_weights_are_applied_to_a_frame_without_labels(
        self, weights_dir, tmp_path, monkeypatch
    ):
        from cil_regionalization.pipelines import montecarlo

        tree = tmp_path / "tree"
        self._build_tree(tree)
        real, seen = montecarlo.apply_weights, []

        def apply_weights(artifact, data, **kwargs):
            seen.append(list(data.columns))
            return real(artifact, data, **kwargs)

        monkeypatch.setattr(montecarlo, "apply_weights", apply_weights)
        run_pipeline(self._cfg(weights_dir, tree, tmp_path / "out"))
        assert seen == [["unit_id", "band", "year", "value"]] * 6

    def test_every_file_is_what_labelling_the_leaf_first_gave(
        self, weights_dir, tmp_path
    ):
        """The reduced file and the statistics, column order and dtypes
        included. Cannot go red by reverting, because labelling the
        leaf first is what the old code did. It failed with the labels
        put after every other column."""
        from cil_regionalization.apply import WeightsArtifact, apply_weights
        from cil_regionalization.pipelines.montecarlo import _ordered
        from cil_regionalization.stats import pooled_statistics, window_means

        tree, out = tmp_path / "tree", tmp_path / "out"
        self._build_tree(tree)
        cfg = self._cfg(weights_dir, tree, out)
        result = run_pipeline(cfg)

        artifact = WeightsArtifact.load(weights_dir)
        reduced = []
        for kumquat, zeppelin in itertools.product(*_LEVELS.values()):
            leaf = pd.read_parquet(tree / kumquat / zeppelin / "damages.parquet")
            applied = apply_weights(
                artifact,
                leaf.assign(kumquat=kumquat, zeppelin=zeppelin),
                kind="extensive",
                weight="area",
                data_version=SOURCE_VERSION,
                value_col="value",
            ).frame.drop(columns=["weight_share"])
            reduced.append(
                window_means(
                    applied,
                    time_col="year",
                    windows=self._WINDOWS,
                    value_col="value",
                    summation="fast",
                )
            )
        combined = pd.concat(reduced, ignore_index=True)
        assert list(combined.columns) == [
            "target_id",
            "band",
            "kumquat",
            "zeppelin",
            "window",
            "value",
        ]
        statistics = _ordered(
            pooled_statistics(
                combined,
                sample_dims=["zeppelin"],
                value_col="value",
                quantiles=[0.05, 0.95],
                quantile_method="linear",
                summation="fast",
            ),
            cfg,
        )
        statistics["weight_share"] = 1.0
        assert list(statistics.columns) == [
            "target_id",
            "band",
            "kumquat",
            "window",
            "statistic",
            "value",
            "weight_share",
        ]

        pd.testing.assert_frame_equal(result.stats, statistics, check_exact=True)
        pd.testing.assert_frame_equal(
            pd.read_parquet(out / "statistics.parquet"),
            _through_parquet(statistics, tmp_path / "expected.parquet"),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(out / "statistics.reduced.parquet"),
            _through_parquet(_ordered(combined, cfg), tmp_path / "expected.parquet"),
            check_exact=True,
        )

    def test_a_passthrough_leaf_file_has_its_labels_last(self, tmp_path):
        """Without weights the labels follow the leaf's own columns in
        the file, the value among them. Cannot go red by reverting,
        because the old code wrote the same file. It failed with the
        labels put before the value, where they go when weights are
        applied."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        self._build_tree(tree)
        raw = self._cfg(tmp_path, tree, out).model_dump()
        del raw["weights"]
        raw["output"]["write_leaves"] = True
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

        for kumquat, zeppelin in itertools.product(*_LEVELS.values()):
            leaf = pd.read_parquet(tree / kumquat / zeppelin / "damages.parquet")
            on_disk = pd.read_parquet(
                out / "leaves" / kumquat / zeppelin / "aggregated_damages.parquet"
            )
            assert list(on_disk.columns) == [
                "unit_id",
                "band",
                "year",
                "value",
                "kumquat",
                "zeppelin",
            ]
            pd.testing.assert_frame_equal(
                on_disk,
                _through_parquet(
                    leaf.assign(kumquat=kumquat, zeppelin=zeppelin),
                    tmp_path / "expected.parquet",
                ),
                check_exact=True,
            )
        reduced = pd.read_parquet(out / "statistics.reduced.parquet")
        assert list(reduced.columns) == [
            "unit_id",
            "band",
            "kumquat",
            "zeppelin",
            "window",
            "value",
        ]

    @pytest.mark.parametrize(
        "name",
        [
            "window",
            "statistic",
            "n_members",
            "n_sample",
            "model_weight",
            "target_id",
            "target_ordinal",
            "warming_level",
            "n_bin_members",
        ],
    )
    def test_a_level_named_like_an_output_column_is_refused_before_any_leaf(
        self, weights_dir, tmp_path, monkeypatch, name
    ):
        from cil_regionalization.pipelines import montecarlo

        levels = {name: ["k0", "k1"], "zeppelin": ["z0", "z1"]}
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, levels)
        real, read = montecarlo._read_leaf, []

        def read_leaf(path, *args):
            read.append(path)
            return real(path, *args)

        monkeypatch.setattr(montecarlo, "_read_leaf", read_leaf)
        with pytest.raises(ValueError, match=f"tree level '{name}' is also"):
            run_pipeline(_pipeline_cfg(weights_dir, tree, out, levels=levels))
        assert read == []
        assert not out.exists()

    def test_a_level_named_like_a_leaf_column_is_refused(self, weights_dir, tmp_path):
        """Cannot go red by reverting, because the old code refused it
        too. It failed with the check taken out. No leaf file is
        written here, and the year is gone from the reduced rows by the
        time the labels go on, so nothing else stops a level of that
        name."""
        levels = {"year": ["k0", "k1"], "zeppelin": ["z0", "z1"]}
        tree = tmp_path / "tree"
        _build_tree(tree, levels)
        raw = _pipeline_cfg(
            weights_dir, tree, tmp_path / "out", levels=levels
        ).model_dump()
        raw["output"]["write_leaves"] = False
        with pytest.raises(ValueError, match="leaf already has a column named 'year'"):
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

    def test_a_passthrough_leaf_of_the_year_and_the_value_alone_still_runs(
        self, tmp_path
    ):
        """Cannot go red by reverting, because the old code labelled
        every leaf first. It failed with this leaf reduced bare, which
        leaves the window means nothing to group by."""
        tree = tmp_path / "tree"
        for i, combo in enumerate(itertools.product(*_LEVELS.values())):
            leaf_dir = tree.joinpath(*combo)
            leaf_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "year": [2050, 2051, 2080, 2081],
                    "value": [100.0 + 10.0 * i + y for y in (0, 1, 30, 31)],
                }
            ).to_parquet(leaf_dir / "damages.parquet", index=False)
        cfg = MonteCarloPipelineConfig.model_validate(
            {
                "tree": {
                    "root": str(tree),
                    "levels": list(_LEVELS),
                    "filename": "damages.parquet",
                    "expect": _LEVELS,
                },
                "data": {"kind": "extensive", "version": SOURCE_VERSION},
                "stats": {
                    "sample_dims": list(_LEVELS),
                    "time_col": "year",
                    "quantile_method": "linear",
                    "summation": "fast",
                    "time_reduction": "window_mean",
                    "windows": [[2050, 2051], [2080, 2081]],
                },
                "output": {
                    "dir": str(tmp_path / "out"),
                    "write_reduced": True,
                    "write_leaves": True,
                },
            }
        )
        stats = run_pipeline(cfg).stats
        assert list(stats.columns) == ["window", "statistic", "value"]
        # the six leaves average 100 + 10 i + 0.5 and 100 + 10 i + 30.5
        assert stats["value"].tolist() == [125.5, 155.5]
        reduced = pd.read_parquet(tmp_path / "out" / "statistics.reduced.parquet")
        assert list(reduced.columns) == ["kumquat", "zeppelin", "window", "value"]
        on_disk = pd.read_parquet(
            tmp_path / "out" / "leaves" / "k0" / "z0" / "aggregated_damages.parquet"
        )
        assert list(on_disk.columns) == ["year", "value", "kumquat", "zeppelin"]


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
                    "quantile_method": "linear",
                    "summation": "fast",
                    "time_reduction": "window_mean",
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
            "weight_share",
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
    def test_missing_declared_leaf_fails_before_processing(self, weights_dir, tmp_path):
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
quantile_method = "linear"
summation = "fast"
time_reduction = "window_mean"
windows = [[2050, 2051]]
quantiles = [0.5]

[output]
dir = "{tmp_path / "out"}"
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
quantile_method = "linear"
summation = "fast"
time_reduction = "window_mean"
windows = [[2050, 2051]]

[output]
dir = "out_rel"
"""
        )
        cfg = load_pipeline_config(cfg_path)
        assert cfg.tree.root == str(tree)
        assert cfg.output.dir == str(tmp_path / "out_rel")

    @pytest.mark.parametrize("table", ["model_weights", "gmst"])
    def test_the_tables_of_stats_resolve_against_config(self, tmp_path, table):
        """As the other paths do. They resolved against the directory the
        run was started in."""
        stats = (
            'model_weights = "tables/weights.tsv"\nmodel_weight_level = "zeppelin"\n'
            'quantile_method = "lower_step"\nwindows = [[2050, 2051]]\n'
            if table == "model_weights"
            else 'quantile_method = "linear"\n'
            "[stats.warming_levels]\n"
            'gmst = "tables/gmst.csv"\nmember = ["zeppelin"]\n'
            "reference_period = [1986, 2005]\noffset = 0.61\nlevels = [2.0]\n"
            'half_width = 0.25\nsmoothing = "unknown"\nsurrogates = "exclude"\n'
            "[[stats.warming_levels.periods]]\n"
            "gmst_span = [2040, 2059]\nimpact_span = [2040, 2059]\n"
        )
        cfg_path = tmp_path / "configs" / "rel.toml"
        cfg_path.parent.mkdir()
        cfg_path.write_text(
            f"""
[weights]
dir = "weights"
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

[output]
dir = "out"

[stats]
sample_dims = ["kumquat", "zeppelin"]
time_col = "year"
summation = "fast"
time_reduction = "window_mean"
{stats}"""
        )
        cfg = load_pipeline_config(cfg_path)
        found = (
            cfg.stats.model_weights
            if table == "model_weights"
            else cfg.stats.warming_levels.gmst
        )
        name = "weights.tsv" if table == "model_weights" else "gmst.csv"
        assert found == str(tmp_path / "configs" / "tables" / name)


class TestNoSiteFactsInCode:
    def test_pipeline_code_carries_no_absolute_paths_or_environments(self):
        import cil_regionalization.pipelines.montecarlo as module

        source = Path(module.__file__).read_text()
        for forbidden in (
            "/project/",
            "/" + "Volumes/",  # split so test_no_mount_paths.py does not match it
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
                    "quantile_method": "linear",
                    "summation": "fast",
                    "time_reduction": "window_mean",
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
quantile_method = "linear"
summation = "fast"
time_reduction = "window_mean"
windows = [[2050, 2051]]
quantiles = [0.5]

[output]
dir = "{tmp_path / "out"}"
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
        raw = _pipeline_cfg(
            weights_dir, tree, tmp_path / "out", levels=_LEVELS
        ).model_dump()
        raw["output"]["write_leaves"] = True
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        for report in result.leaf_reports:
            # 2 targets x 4 years aggregated; 2 targets x 2 windows pooled.
            assert report["rows_out"] == 8
            assert report["rows_reduced"] == 4
        on_disk = pd.read_parquet(
            tmp_path / "out" / "leaves" / "k0" / "z0" / "damages.aggregated.parquet"
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

    def test_a_star_leaves_out_every_value_of_its_level(self, weights_dir, tmp_path):
        """A model that one scenario lacks, in every batch, is one line
        and not one line a batch."""
        import shutil

        levels = {
            "batch": ["b0", "b1", "b2"],
            "rcp": ["r45", "r85"],
            "model": ["m0", "m1"],
        }
        tree = tmp_path / "tree"
        _build_tree(tree, levels)
        for batch in levels["batch"]:
            shutil.rmtree(tree / batch / "r85" / "m1")
        raw = _pipeline_cfg(
            weights_dir, tree, tmp_path / "out", levels=levels
        ).model_dump()
        missing = run_pipeline(
            MonteCarloPipelineConfig.model_validate(raw), dry_run=True
        ).missing
        assert len(missing) == 3
        raw["tree"]["exclude"] = [["*", "r85", "m1"]]
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        plan = run_pipeline(cfg, dry_run=True)
        assert plan.missing == [] and len(plan.leaves) == 9
        assert ("b1", "r85", "m1") not in {leaf.values for leaf in plan.leaves}
        assert ("b1", "r85", "m0") in {leaf.values for leaf in plan.leaves}
        result = run_pipeline(cfg)
        assert len(result.leaf_reports) == 9
        manifest = json.loads(Path(result.paths["manifest"]).read_text())
        assert manifest["excluded_leaves"] == [["*", "r85", "m1"]]
        # two stars, and a star next to a cell written out
        raw["tree"]["exclude"] = [["*", "*", "m1"], ["b0", "r45", "m0"]]
        plan = run_pipeline(MonteCarloPipelineConfig.model_validate(raw), dry_run=True)
        assert sorted(leaf.values for leaf in plan.leaves) == [
            ("b0", "r85", "m0"),
            ("b1", "r45", "m0"),
            ("b1", "r85", "m0"),
            ("b2", "r45", "m0"),
            ("b2", "r85", "m0"),
        ]

    def test_a_cell_of_stars_alone_is_refused(self, weights_dir, tmp_path):
        raw = _pipeline_cfg(
            weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS
        ).model_dump()
        raw["tree"]["exclude"] = [["*", "*"]]
        with pytest.raises(ValueError, match="leaves out every leaf"):
            MonteCarloPipelineConfig.model_validate(raw)
        raw["tree"]["exclude"] = [["*", "z9"]]
        with pytest.raises(ValueError, match="'z9' is not a declared value"):
            MonteCarloPipelineConfig.model_validate(raw)

    def test_exclude_of_undeclared_value_is_rejected(self, weights_dir, tmp_path):
        cfg = _pipeline_cfg(weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["tree"]["exclude"] = [["k1", "z9"]]
        with pytest.raises(ValueError, match="not a declared value"):
            MonteCarloPipelineConfig.model_validate(raw)


class TestModelWeights:
    def _weights_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "weights.tsv"
        pd.DataFrame({"model": ["Z0*", "z1", "z2"], "weight": [0.6, 0.3, 0.1]}).to_csv(
            p, sep="\t", index=False
        )
        return p

    def test_weighted_mean_uses_the_weights(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["stats"]["model_weights"] = str(self._weights_file(tmp_path))
        raw["stats"]["quantile_method"] = "lower_step"
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
        manifest = json.loads((Path(result.paths["manifest"])).read_text())
        assert manifest["model_weight_level"] == "zeppelin"

    def test_unmatched_model_is_rejected(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        p = tmp_path / "weights.tsv"
        pd.DataFrame({"model": ["z0"], "weight": [1.0]}).to_csv(
            p, sep="\t", index=False
        )
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["stats"]["model_weights"] = str(p)
        raw["stats"]["quantile_method"] = "lower_step"
        raw["stats"]["model_weight_level"] = "zeppelin"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        # the check of the values the tree declares, not the lookup of
        # one leaf's weight, which says "has no row"
        with pytest.raises(
            ValueError, match=r"values of 'zeppelin' have no row in the model"
        ):
            run_pipeline(cfg)

    def test_weight_level_must_be_sample_dim(self, weights_dir, tmp_path):
        cfg = _pipeline_cfg(weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["stats"]["model_weights"] = "x.tsv"
        raw["stats"]["quantile_method"] = "lower_step"
        raw["stats"]["model_weight_level"] = "nope"
        with pytest.raises(ValueError, match="sample_dims"):
            MonteCarloPipelineConfig.model_validate(raw)

    def _weighted_cfg(
        self, weights_dir, tree, out, weights_file, *, n_workers=1, exclude=()
    ) -> MonteCarloPipelineConfig:
        cfg = _pipeline_cfg(weights_dir, tree, out, levels=_LEVELS, n_workers=n_workers)
        raw = cfg.model_dump()
        raw["stats"]["model_weights"] = str(weights_file)
        raw["stats"]["quantile_method"] = "lower_step"
        raw["stats"]["model_weight_level"] = "zeppelin"
        raw["tree"]["exclude"] = [list(cell) for cell in exclude]
        return MonteCarloPipelineConfig.model_validate(raw)

    def test_two_rows_for_one_model_are_refused(self, weights_dir, tmp_path):
        """The last of the two rows used to be the one applied. The run
        stops before any leaf is read, so nothing is written."""
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        p = tmp_path / "weights.tsv"
        pd.DataFrame(
            {"model": ["Z0*", "z1", "z2", "z0"], "weight": [0.6, 0.3, 0.1, 0.2]}
        ).to_csv(p, sep="\t", index=False)
        cfg = self._weighted_cfg(weights_dir, tree, tmp_path / "out", p)
        with pytest.raises(ValueError) as error:
            run_pipeline(cfg)
        assert "['Z0*', 'z0']" in str(error.value)
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_a_file_edited_during_the_run_changes_nothing(
        self, weights_dir, tmp_path, monkeypatch, n_workers
    ):
        """The file is rewritten with other weights as soon as it has
        been read once. Every leaf still gets the weights of that first
        read, and the manifest records its bytes."""
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        p = self._weights_file(tmp_path)
        original = p.read_bytes()
        control = run_pipeline(
            self._weighted_cfg(weights_dir, tree, tmp_path / "control", p)
        )

        read_csv = pd.read_csv

        def read_then_edit(*args, **kwargs):
            table = read_csv(*args, **kwargs)
            if "model" in table.columns and p.read_bytes() == original:
                pd.DataFrame(
                    {"model": ["Z0*", "z1", "z2"], "weight": [0.1, 0.3, 0.6]}
                ).to_csv(p, sep="\t", index=False)
            return table

        monkeypatch.setattr(pd, "read_csv", read_then_edit)
        result = run_pipeline(
            self._weighted_cfg(
                weights_dir, tree, tmp_path / "out", p, n_workers=n_workers
            )
        )
        monkeypatch.undo()

        assert p.read_bytes() != original
        # the leaves of a pool are summed in the order they finish
        pd.testing.assert_frame_equal(
            result.stats, control.stats, check_exact=False, rtol=1e-12
        )
        manifest = json.loads(Path(result.paths["manifest"]).read_text())
        record = manifest["model_weights_record"]
        assert record["sha256"] == hashlib.sha256(original).hexdigest()
        assert [m["weight"] for m in record["members"].values()] == [0.6, 0.3, 0.1]

    def test_the_manifest_records_the_members_attached(self, weights_dir, tmp_path):
        """Every leaf of z2 is excluded, so z2 is declared and has a row
        but no leaf was given its weight. It is listed with the rows no
        member matched, next to a model the tree does not have."""
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        p = tmp_path / "weights.tsv"
        pd.DataFrame(
            {"model": ["Z0*", "z1", "z2", "ghost"], "weight": [0.6, 0.3, 0.1, 0.25]}
        ).to_csv(p, sep="\t", index=False)
        cfg = self._weighted_cfg(
            weights_dir,
            tree,
            tmp_path / "out",
            p,
            exclude=[("k0", "z2"), ("k1", "z2")],
        )
        result = run_pipeline(cfg)

        manifest = json.loads(Path(result.paths["manifest"]).read_text())
        assert manifest["model_weights"] == str(p)
        assert manifest["model_weights_record"] == {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "members": {
                "z0": {"row": "Z0*", "weight": 0.6, "n_leaves": 2},
                "z1": {"row": "z1", "weight": 0.3, "n_leaves": 2},
            },
            "unmatched_rows": {"z2": 0.1, "ghost": 0.25},
        }

    def test_a_run_without_model_weights_records_none(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        result = run_pipeline(
            _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        )
        manifest = json.loads(Path(result.paths["manifest"]).read_text())
        assert manifest["model_weights_record"] is None


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
        run_pipeline(cfg)
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


class TestPassthrough:
    """No [weights] section: the same walk, subtraction, reduction, and
    pooled statistics with no spatial aggregation, keyed by the leaf's
    own regions."""

    def _cfg_dict(self, tree: Path, out: Path, *, windows=None) -> dict:
        return {
            "tree": {
                "root": str(tree),
                "levels": list(_LEVELS),
                "filename": "damages.parquet",
                "expect": _LEVELS,
            },
            "data": {
                "kind": "extensive",
                "version": SOURCE_VERSION,
                "value_col": "value",
            },
            "stats": {
                "sample_dims": list(_LEVELS),
                "time_col": "year",
                "quantile_method": "linear",
                "summation": "fast",
                "time_reduction": "window_mean",
                "windows": windows or [[2050, 2051], [2080, 2081]],
                "quantiles": [0.05, 0.95],
            },
            "output": {"dir": str(out), "write_leaves": False},
        }

    def test_parquet_statistics_per_unit(self, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg = MonteCarloPipelineConfig.model_validate(
            self._cfg_dict(tree, tmp_path / "out")
        )
        result = run_pipeline(cfg)

        # No weights were applied, so there is no coverage or mass
        # accounting to report.
        assert len(result.leaf_reports) == 6
        for report in result.leaf_reports:
            assert report["error"] is None
            assert report["coverage"] is None
            assert report["mass_balance_rel_error"] is None

        stats = result.stats
        assert set(stats.columns) == {"unit_id", "statistic", "value", "window"}
        # 2 units x 2 windows x 3 statistics
        assert len(stats) == 12

        # Hand check: u1's value per (leaf i, year) is 100 + 10 i +
        # (year - 2050); window 2050-2051 mean per leaf is 100 + 10 i +
        # 0.5, and the pooled mean over i = 0..5 is 100 + 25 + 0.5.
        u1_mean = stats.query(
            "unit_id == 'u1' and statistic == 'mean' and window == '2050-2051'"
        )["value"].iloc[0]
        assert u1_mean == pytest.approx(100.0 + 25.0 + 0.5)

        manifest = json.loads(
            (tmp_path / "out" / "statistics.manifest.json").read_text()
        )
        assert manifest["weight"] is None
        assert "passthrough" in manifest["spatial_aggregation"]
        assert manifest["artifact_source_version"] is None

    def test_netcdf_region_col_and_single_year_window(self, tmp_path):
        pytest.importorskip("xarray")
        tree = tmp_path / "tree_nc"
        _build_netcdf_tree(tree, _LEVELS)
        raw = self._cfg_dict(
            tree, tmp_path / "out", windows=[[2050, 2051], [2081, 2081]]
        )
        raw["tree"]["filename"] = "damages.nc4"
        raw["data"]["format"] = "netcdf"
        raw["data"]["region_dim"] = "spatial_thing"
        raw["data"]["region_col"] = "hierid"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)

        stats = result.stats
        assert set(stats.columns) == {"hierid", "statistic", "value", "window"}
        assert set(stats["hierid"]) == {"u1", "u2"}
        # the single-year window is labeled by the year alone
        assert set(stats["window"]) == {"2050-2051", "2081"}
        # and the statistics file carries plain strings, not the
        # categorical encoding the pooled frame used internally
        assert not isinstance(stats["hierid"].dtype, pd.CategoricalDtype)

        # u1 at 2081 for leaf i is 100 + 10 i + 31; pooled mean over
        # i = 0..5 is 100 + 25 + 31.
        u1_2081 = stats.query(
            "hierid == 'u1' and statistic == 'mean' and window == '2081'"
        )["value"].iloc[0]
        assert u1_2081 == pytest.approx(100.0 + 25.0 + 31.0)

    def test_subtraction_flows_through(self, tmp_path):
        tree = tmp_path / "tree"
        _build_subtract_tree(tree, _LEVELS)
        raw = self._cfg_dict(tree, tmp_path / "out")
        raw["tree"]["subtract_filename"] = "histclim.parquet"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)
        # value - (value - 7) = 7 per unit and year, with no half-unit
        # splitting because nothing aggregates
        assert (result.stats["value"].round(9) == 7.0).all()

    def test_parallel_matches_serial(self, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        serial = run_pipeline(
            MonteCarloPipelineConfig.model_validate(
                self._cfg_dict(tree, tmp_path / "out_s")
            )
        )
        raw = self._cfg_dict(tree, tmp_path / "out_p")
        raw["run"] = {"n_workers": 2}
        parallel = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        key = ["unit_id", "statistic", "window"]
        pd.testing.assert_frame_equal(
            serial.stats.sort_values(key).reset_index(drop=True),
            parallel.stats.sort_values(key).reset_index(drop=True),
        )

    def test_netcdf_without_region_col_is_rejected(self, tmp_path):
        raw = self._cfg_dict(tmp_path / "t", tmp_path / "o")
        raw["tree"]["filename"] = "damages.nc4"
        raw["data"]["format"] = "netcdf"
        raw["data"]["region_dim"] = "spatial_thing"
        with pytest.raises(ValueError, match="region_col"):
            MonteCarloPipelineConfig.model_validate(raw)

    def test_region_col_with_weights_is_rejected(self, weights_dir, tmp_path):
        cfg = _pipeline_cfg(weights_dir, tmp_path / "t", tmp_path / "o", levels=_LEVELS)
        raw = cfg.model_dump()
        raw["data"]["format"] = "netcdf"
        raw["data"]["region_dim"] = "spatial_thing"
        raw["data"]["region_col"] = "hierid"
        with pytest.raises(ValueError, match="artifact schema"):
            MonteCarloPipelineConfig.model_validate(raw)

    def test_ratio_without_weights_is_rejected(self, tmp_path):
        raw = self._cfg_dict(tmp_path / "t", tmp_path / "o")
        raw["data"]["kind"] = "ratio"
        raw["data"]["denominator_col"] = "den"
        with pytest.raises(ValueError, match="ratio needs"):
            MonteCarloPipelineConfig.model_validate(raw)

    def test_policies_without_weights_is_rejected(self, tmp_path):
        raw = self._cfg_dict(tmp_path / "t", tmp_path / "o")
        raw["policies"] = {"on_zero_weight": "skip"}
        with pytest.raises(ValueError, match="nothing for it to govern"):
            MonteCarloPipelineConfig.model_validate(raw)


def _leaves_cfg(
    weights_dir, tree, out, *, levels=None, n_workers=1, **output
) -> MonteCarloPipelineConfig:
    raw = _pipeline_cfg(
        weights_dir, tree, out, levels=levels or _LEVELS, n_workers=n_workers
    ).model_dump()
    raw["output"].update(write_leaves=True, **output)
    return MonteCarloPipelineConfig.model_validate(raw)


def _leaves_manifest(out: Path, stem: str = "statistics") -> dict:
    return json.loads((out / "leaves" / f"{stem}.leaves.manifest.json").read_text())


def _poison(tree: Path) -> pd.DataFrame:
    """Give one leaf a unit the weights do not know, so it fails, and
    return what it held."""
    path = tree / "k0" / "z1" / "damages.parquet"
    good = pd.read_parquet(path)
    bad = good.copy()
    bad.loc[len(bad)] = {"unit_id": "ghost", "year": 2050, "value": 1.0}
    bad.to_parquet(path, index=False)
    return good


@pytest.fixture
def one_target_a_group(monkeypatch):
    """Two targets make two row groups."""
    from cil_regionalization import leaves

    monkeypatch.setattr(leaves, "TARGETS_PER_GROUP", 1)


@pytest.mark.usefixtures("one_target_a_group")
class TestLeafFiles:
    """What `output.write_leaves` puts on disk."""

    def test_a_leaf_file_is_the_aggregated_leaf_laid_out_by_target(
        self, weights_dir, tmp_path
    ):
        import pyarrow.parquet as pq

        from cil_regionalization.apply import WeightsArtifact, apply_weights
        from cil_regionalization.leaves import read_leaf_footer

        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        run_pipeline(_leaves_cfg(weights_dir, tree, out))
        manifest = _leaves_manifest(out)

        artifact = WeightsArtifact.load(weights_dir)
        for kumquat, zeppelin in itertools.product(*_LEVELS.values()):
            source = tree / kumquat / zeppelin / "damages.parquet"
            applied = apply_weights(
                artifact,
                pd.read_parquet(source),
                kind="extensive",
                weight="area",
                data_version=SOURCE_VERSION,
                value_col="value",
            ).frame.sort_values("target_id", kind="stable")
            folder = out / "leaves" / kumquat / zeppelin
            assert [p.name for p in folder.iterdir()] == ["damages.aggregated.parquet"]
            path = folder / "damages.aggregated.parquet"
            on_disk = pq.read_table(path).to_pandas()
            assert list(on_disk.columns) == [
                "target_ordinal",
                "target_id",
                "year",
                "value",
            ]
            assert on_disk["target_ordinal"].tolist() == [0] * 4 + [1] * 4
            assert on_disk["target_id"].tolist() == applied["target_id"].tolist()
            assert on_disk["year"].tolist() == applied["year"].tolist()
            assert (
                on_disk["value"].to_numpy().tobytes()
                == applied["value"].to_numpy().tobytes()
            )
            assert pq.read_metadata(path).num_row_groups == 2
            assert read_leaf_footer(path) == {
                "format_version": 1,
                "targets_per_group": 1,
                "n_targets": 2,
                "target_digest": manifest["targets"]["digest"],
                "aggregation": manifest["aggregation"],
                "aggregation_digest": manifest["aggregation_digest"],
                "levels": {"kumquat": kumquat, "zeppelin": zeppelin},
                "sources": [str(source)],
            }

    def test_the_manifest_lists_every_leaf_in_plan_order(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """The pool here runs in this process and hands the leaves back
        last first."""
        import types

        from cil_regionalization import leaves
        from cil_regionalization.pipelines import montecarlo

        class LastFirstPool:
            def __init__(self, processes, initializer, initargs, maxtasksperchild):
                initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def imap_unordered(self, func, tasks, chunksize=1):
                return [func(task) for task in reversed(list(tasks))]

        monkeypatch.setattr(montecarlo, "mp", types.SimpleNamespace(Pool=LastFirstPool))
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        run_pipeline(_leaves_cfg(weights_dir, tree, out, n_workers=2))
        manifest = _leaves_manifest(out)

        assert sorted(manifest) == [
            "aggregation",
            "aggregation_digest",
            "complete",
            "format_version",
            "group_col",
            "leaves",
            "package_versions",
            "paths",
            "profiles",
            "rows_per_group",
            "schema",
            "stem",
            "target_coverage",
            "targets",
            "targets_per_group",
        ]
        assert manifest["complete"] is True
        assert manifest["stem"] == "statistics"
        assert manifest["format_version"] == 1
        assert manifest["targets_per_group"] == 1
        assert manifest["rows_per_group"] == [24, 24]
        assert manifest["schema"] == [
            {"name": "target_ordinal", "arrow": "int32", "pandas": "int32"},
            {
                "name": "target_id",
                "arrow": "string",
                "pandas": str(pd.Series(["T1"]).dtype),
            },
            {"name": "year", "arrow": "int64", "pandas": "int64"},
            {"name": "value", "arrow": "double", "pandas": "float64"},
        ]
        assert manifest["profiles"] == [
            {"year": {"values": [2050, 2051, 2080, 2081], "rows": [2, 2, 2, 2]}}
        ]
        assert "pyarrow" in manifest["package_versions"]

        combos = list(itertools.product(*_LEVELS.values()))
        assert [tuple(entry["leaf"]) for entry in manifest["leaves"]] == combos
        for entry, (kumquat, zeppelin) in zip(manifest["leaves"], combos):
            path = out / "leaves" / kumquat / zeppelin / "damages.aggregated.parquet"
            assert entry == {
                "leaf": [kumquat, zeppelin],
                "file": f"{kumquat}/{zeppelin}/damages.aggregated.parquet",
                "sources": [str(tree / kumquat / zeppelin / "damages.parquet")],
                "rows_in": 8,
                "rows_out": 8,
                "coverage": entry["coverage"],
                "target_coverage": {
                    "left_out": {"no weight": [], "no data": []},
                    "partial": [],
                },
                "mass_balance_rel_error": entry["mass_balance_rel_error"],
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "profile": 0,
            }
            assert entry["coverage"] is not None
            assert abs(entry["mass_balance_rel_error"]) < 1e-12

        table = leaves.TargetTable.read(out / "leaves" / manifest["targets"]["file"])
        assert table.ids == (("T1",), ("T2",))
        assert manifest["targets"] == {
            "file": f"targets.{table.digest[:16]}.parquet",
            "digest": table.digest,
            "n_targets": 2,
            "id_fields": ["target_id"],
        }

    def test_the_manifest_says_what_made_the_leaves_and_nothing_of_the_statistics(
        self, weights_dir, tmp_path
    ):
        """The identity holds the hashes of the weights artifact and no
        path. The paths are recorded beside it."""
        from cil_regionalization.leaves import canonical_digest

        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        run_pipeline(_leaves_cfg(weights_dir, tree, out))
        manifest = _leaves_manifest(out)

        assert manifest["aggregation"] == {
            "weight": "area",
            "kind": "extensive",
            "data_version": SOURCE_VERSION,
            "data_format": "parquet",
            "value_col": "value",
            "denominator_col": None,
            "region_dim": None,
            "region_labels": None,
            "policies": {
                "on_unmatched": "error",
                "on_zero_weight": "error",
                "on_absent_from_data": "error",
            },
            "filename": "damages.parquet",
            "subtract_filename": None,
            "levels": ["kumquat", "zeppelin"],
            "artifact": {
                "source_version": SOURCE_VERSION,
                "regions_version": "targets-v1",
                "sha256": {
                    name: hashlib.sha256(
                        (weights_dir / f"weights.{suffix}").read_bytes()
                    ).hexdigest()
                    for name, suffix in (
                        ("table", "parquet"),
                        ("manifest", "manifest.json"),
                    )
                },
            },
        }
        assert manifest["aggregation_digest"] == canonical_digest(
            manifest["aggregation"]
        )
        assert str(tmp_path) not in json.dumps(manifest["aggregation"])
        assert manifest["paths"] == {
            "weights_dir": str(weights_dir),
            "weights_stem": "weights",
            "tree_root": str(tree),
        }
        text = json.dumps(manifest)
        for word in ("window", "quantile", "sample_dims", "summation", "model_weight"):
            assert word not in text

    def test_the_statistics_manifest_points_at_the_leaves(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        hashes = {
            "table": hashlib.sha256(
                (weights_dir / "weights.parquet").read_bytes()
            ).hexdigest(),
            "manifest": hashlib.sha256(
                (weights_dir / "weights.manifest.json").read_bytes()
            ).hexdigest(),
        }

        result = run_pipeline(_leaves_cfg(weights_dir, tree, tmp_path / "with"))
        manifest = json.loads(
            (tmp_path / "with" / "statistics.manifest.json").read_text()
        )
        expected = str(tmp_path / "with" / "leaves" / "statistics.leaves.manifest.json")
        assert manifest["stage"] == "all"
        assert manifest["leaves_manifest"] == expected
        assert result.paths["leaves_manifest"] == expected
        assert manifest["artifact_sha256"] == hashes
        assert "leaf_file" not in manifest["leaves"][0]

        result = run_pipeline(
            _pipeline_cfg(weights_dir, tree, tmp_path / "without", levels=_LEVELS)
        )
        manifest = json.loads(
            (tmp_path / "without" / "statistics.manifest.json").read_text()
        )
        assert manifest["leaves_manifest"] is None
        assert "leaves_manifest" not in result.paths
        assert manifest["artifact_sha256"] == hashes

    def test_writing_the_leaves_changes_no_statistic(self, weights_dir, tmp_path):
        """Cannot go red by reverting, because the old code wrote its
        leaves without changing a statistic either. It failed with the
        window means taken from the leaf file read back, which has a
        column the frame does not."""
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        with_leaves = run_pipeline(
            _leaves_cfg(weights_dir, tree, tmp_path / "with", write_reduced=True)
        )
        raw = _pipeline_cfg(
            weights_dir, tree, tmp_path / "without", levels=_LEVELS
        ).model_dump()
        raw["output"]["write_reduced"] = True
        without = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        pd.testing.assert_frame_equal(
            with_leaves.stats, without.stats, check_exact=True
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "with" / "statistics.reduced.parquet"),
            pd.read_parquet(tmp_path / "without" / "statistics.reduced.parquet"),
            check_exact=True,
        )

    def test_the_manifest_is_incomplete_until_every_leaf_is_written(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """It is there before the first leaf, with the files the run is
        about to write and the target table beside it."""
        from cil_regionalization import leaves

        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        real, seen = leaves.write_leaf, []

        def write_leaf(frame, path, **kwargs):
            manifest = _leaves_manifest(out)
            seen.append(
                (
                    manifest["complete"],
                    [entry["file"] for entry in manifest["leaves"]],
                    (out / "leaves" / manifest["targets"]["file"]).exists(),
                    sorted(manifest["leaves"][0]),
                )
            )
            return real(frame, path, **kwargs)

        monkeypatch.setattr(leaves, "write_leaf", write_leaf)
        run_pipeline(_leaves_cfg(weights_dir, tree, out))
        files = [
            f"{kumquat}/{zeppelin}/damages.aggregated.parquet"
            for kumquat, zeppelin in itertools.product(*_LEVELS.values())
        ]
        assert seen == [(False, files, True, ["file", "leaf"])] * 6
        assert _leaves_manifest(out)["complete"] is True

    def test_a_failed_leaf_leaves_the_manifest_incomplete(self, weights_dir, tmp_path):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        good = _poison(tree)
        cfg = _leaves_cfg(weights_dir, tree, out)
        with pytest.raises(ValueError, match="partial run"):
            run_pipeline(cfg)
        manifest = _leaves_manifest(out)
        assert manifest["complete"] is False
        assert len(manifest["leaves"]) == 6
        assert "rows_per_group" not in manifest
        assert not (out / "statistics.parquet").exists()

        # the same stem writes them again, and the manifest is replaced
        good.to_parquet(tree / "k0" / "z1" / "damages.parquet", index=False)
        run_pipeline(cfg)
        assert _leaves_manifest(out)["complete"] is True

    @pytest.mark.parametrize("first_run", ["complete", "killed"])
    def test_leaves_another_stem_lists_are_not_written_over(
        self, weights_dir, tmp_path, first_run
    ):
        """Whether the other run finished or not. Failed with no
        manifest written before the first leaf, which forgets what a
        killed run was writing."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        first = _leaves_cfg(weights_dir, tree, out, stem="first")
        if first_run == "complete":
            run_pipeline(first)
        else:
            good = _poison(tree)
            with pytest.raises(ValueError, match="partial run"):
                run_pipeline(first)
            good.to_parquet(tree / "k0" / "z1" / "damages.parquet", index=False)
        before = {p: p.read_bytes() for p in (out / "leaves").rglob("*") if p.is_file()}

        with pytest.raises(ValueError, match="6 leaf files .* output stem 'first'"):
            run_pipeline(_leaves_cfg(weights_dir, tree, out, stem="second"))
        after = {p: p.read_bytes() for p in (out / "leaves").rglob("*") if p.is_file()}
        assert after == before
        assert not (out / "second.parquet").exists()

    def test_two_stems_with_different_leaves_share_a_directory(
        self, weights_dir, tmp_path
    ):
        """One scenario a stem, as the published runs are laid out."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        for kumquat in _LEVELS["kumquat"]:
            run_pipeline(
                _leaves_cfg(
                    weights_dir,
                    tree,
                    out,
                    levels={"kumquat": [kumquat], "zeppelin": _LEVELS["zeppelin"]},
                    stem=kumquat,
                )
            )
        k0, k1 = _leaves_manifest(out, "k0"), _leaves_manifest(out, "k1")
        assert k0["complete"] and k1["complete"]
        assert {tuple(e["leaf"]) for e in k0["leaves"]} == {
            ("k0", z) for z in _LEVELS["zeppelin"]
        }
        assert {tuple(e["leaf"]) for e in k1["leaves"]} == {
            ("k1", z) for z in _LEVELS["zeppelin"]
        }
        assert k0["targets"] == k1["targets"]
        assert k0["aggregation_digest"] == k1["aggregation_digest"]
        assert sorted(p.name for p in (out / "leaves").iterdir()) == [
            "k0",
            "k0.leaves.manifest.json",
            "k1",
            "k1.leaves.manifest.json",
            k0["targets"]["file"],
        ]

    def test_the_group_size_reaches_the_workers_of_a_pool(self, weights_dir, tmp_path):
        """A spawned worker imports the module afresh, so the size goes
        along with the task."""
        import pyarrow.parquet as pq

        from cil_regionalization.leaves import read_leaf_footer

        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        run_pipeline(_leaves_cfg(weights_dir, tree, out, n_workers=2))
        for path in (out / "leaves").rglob("*.aggregated.parquet"):
            assert read_leaf_footer(path)["targets_per_group"] == 1
            assert pq.read_metadata(path).num_row_groups == 2
        assert _leaves_manifest(out)["rows_per_group"] == [24, 24]

    def test_a_leaf_without_a_target_has_an_empty_group_for_it(
        self, weights_dir, tmp_path
    ):
        import pyarrow.parquet as pq

        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        # u2 is the only unit that reaches T2
        path = tree / "k0" / "z0" / "damages.parquet"
        pd.read_parquet(path).query("unit_id == 'u1'").to_parquet(path, index=False)
        raw = _leaves_cfg(weights_dir, tree, out).model_dump()
        raw["policies"] = {"on_absent_from_data": "skip"}
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

        metadata = pq.read_metadata(
            out / "leaves" / "k0" / "z0" / "damages.aggregated.parquet"
        )
        assert [metadata.row_group(g).num_rows for g in range(2)] == [4, 0]
        assert _leaves_manifest(out)["rows_per_group"] == [24, 20]

    def test_an_artifact_that_changes_during_the_run_fails_every_leaf(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """The run hashes the artifact twice when it loads it, and so
        does the worker, which here gets other bytes."""
        from cil_regionalization.pipelines import montecarlo

        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        real, calls = montecarlo._artifact_sha256, []

        def sha256(weights_dir, stem):
            calls.append(stem)
            found = real(weights_dir, stem)
            return found if len(calls) <= 2 else {**found, "table": "0" * 64}

        monkeypatch.setattr(montecarlo, "_artifact_sha256", sha256)
        with pytest.raises(ValueError, match="0 of 6 leaves succeeded.*changed during"):
            run_pipeline(_leaves_cfg(weights_dir, tree, out))
        assert _leaves_manifest(out)["complete"] is False
        assert list((out / "leaves").rglob("*.aggregated.parquet")) == []

    def test_an_artifact_that_changes_while_it_loads_is_refused(
        self, weights_dir, tmp_path, monkeypatch
    ):
        from cil_regionalization.pipelines import montecarlo

        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        real, calls = montecarlo._artifact_sha256, []

        def sha256(weights_dir, stem):
            calls.append(stem)
            found = real(weights_dir, stem)
            return found if len(calls) == 1 else {**found, "manifest": "0" * 64}

        monkeypatch.setattr(montecarlo, "_artifact_sha256", sha256)
        with pytest.raises(ValueError, match="changed while it was being loaded"):
            run_pipeline(_leaves_cfg(weights_dir, tree, out))
        assert not out.exists()

    def test_a_dimension_that_cannot_be_written_fails_the_run(
        self, weights_dir, tmp_path
    ):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        for path in tree.rglob("damages.parquet"):
            leaf = pd.read_parquet(path)
            leaf["when"] = pd.to_datetime(leaf["year"].astype(str) + "-07-01")
            leaf[["unit_id", "when", "year", "value"]].to_parquet(path, index=False)
        with pytest.raises(ValueError, match="leaf column 'when' has dtype datetime64"):
            run_pipeline(_leaves_cfg(weights_dir, tree, out))
        assert _leaves_manifest(out)["complete"] is False
        assert not (out / "statistics.parquet").exists()

    def test_a_leaf_with_other_columns_stops_the_run_when_it_arrives(
        self, weights_dir, tmp_path
    ):
        """The fourth leaf of six has its year as int32. The two after
        it are never read."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        path = tree / "k1" / "z0" / "damages.parquet"
        pd.read_parquet(path).astype({"year": "int32"}).to_parquet(path, index=False)
        with pytest.raises(
            ValueError,
            match="leaf k1/z0 was written with columns .* where leaf k0/z0 has "
            ".* stopped after 4 of 6 leaves",
        ):
            run_pipeline(_leaves_cfg(weights_dir, tree, out))
        written = sorted(
            p.parent.relative_to(out / "leaves").as_posix()
            for p in (out / "leaves").rglob("*.aggregated.parquet")
        )
        assert written == ["k0/z0", "k0/z1", "k0/z2", "k1/z0"]
        assert _leaves_manifest(out)["complete"] is False
        assert not (out / "statistics.parquet").exists()

    def test_the_first_leaf_to_arrive_fixes_the_columns(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """Not the first in plan order. The pool here runs in this
        process and hands each leaf back as it is asked for, last
        first, so the odd leaf is the third to arrive."""
        import types

        from cil_regionalization.pipelines import montecarlo

        class LastFirstPool:
            def __init__(self, processes, initializer, initargs, maxtasksperchild):
                initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def imap_unordered(self, func, tasks, chunksize=1):
                return (func(task) for task in reversed(list(tasks)))

        monkeypatch.setattr(montecarlo, "mp", types.SimpleNamespace(Pool=LastFirstPool))
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        path = tree / "k1" / "z0" / "damages.parquet"
        pd.read_parquet(path).astype({"year": "int32"}).to_parquet(path, index=False)
        with pytest.raises(
            ValueError,
            match="leaf k1/z0 was written .* where leaf k1/z2 has .* after 3 of 6",
        ):
            run_pipeline(_leaves_cfg(weights_dir, tree, out, n_workers=2))
        written = sorted(
            p.parent.relative_to(out / "leaves").as_posix()
            for p in (out / "leaves").rglob("*.aggregated.parquet")
        )
        assert written == ["k1/z0", "k1/z1", "k1/z2"]

    def test_files_no_manifest_lists_are_counted_and_kept(
        self, weights_dir, tmp_path, capsys
    ):
        """A second run under the stem declares half the leaves. The
        other half stays on disk and is reported with its size, next to
        a file of the old layout. The leaves of another stem are
        listed, so they are not counted."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        run_pipeline(_leaves_cfg(weights_dir, tree, out))
        assert (
            f"leaves: every file under {out / 'leaves'} is in a leaves manifest"
            in capsys.readouterr().out
        )

        old = out / "leaves" / "k1" / "z0" / "aggregated_damages.parquet"
        pd.DataFrame({"value": [1.0]}).to_parquet(old)
        narrow = {"kumquat": ["k0"], "zeppelin": _LEVELS["zeppelin"]}
        run_pipeline(_leaves_cfg(weights_dir, tree, out, levels=narrow))
        left = sorted((out / "leaves" / "k1").rglob("*.parquet"))
        assert len(left) == 4
        size = sum(p.stat().st_size for p in left)
        assert (
            f"leaves: 4 files under {out / 'leaves'} are in no leaves manifest, "
            f"{size:,} bytes ({size / 1e9:.2f} GB), first "
            f"['k1/z0/aggregated_damages.parquet', "
            f"'k1/z0/damages.aggregated.parquet', "
            f"'k1/z1/damages.aggregated.parquet']. Nothing was deleted."
        ) in capsys.readouterr().out

        k1 = {"kumquat": ["k1"], "zeppelin": _LEVELS["zeppelin"]}
        run_pipeline(_leaves_cfg(weights_dir, tree, out, levels=k1, stem="other"))
        assert (
            f"leaves: 1 files under {out / 'leaves'} are in no leaves manifest"
            in capsys.readouterr().out
        )
        assert old.exists()

    def test_a_passthrough_run_has_no_leaves_manifest(self, weights_dir, tmp_path):
        """Without weights there are no targets to lay a leaf out by,
        so the file is the one it always was."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        raw = _leaves_cfg(weights_dir, tree, out).model_dump()
        del raw["weights"]
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        assert sorted(p.name for p in (out / "leaves").iterdir()) == ["k0", "k1"]
        assert sorted(p.name for p in (out / "leaves" / "k0" / "z0").iterdir()) == [
            "aggregated_damages.parquet"
        ]
        manifest = json.loads((out / "statistics.manifest.json").read_text())
        assert manifest["leaves_manifest"] is None
        assert manifest["artifact_sha256"] is None
        assert "leaves_manifest" not in result.paths

    def test_a_file_of_the_old_layout_is_left_alone(self, weights_dir, tmp_path):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        old = out / "leaves" / "k0" / "z0" / "aggregated_damages.parquet"
        old.parent.mkdir(parents=True)
        pd.DataFrame({"target_id": ["T1"], "value": [1.0]}).to_parquet(old)
        before = old.read_bytes()
        run_pipeline(_leaves_cfg(weights_dir, tree, out))
        assert old.read_bytes() == before
        assert sorted(p.name for p in old.parent.iterdir()) == [
            "aggregated_damages.parquet",
            "damages.aggregated.parquet",
        ]


@pytest.mark.usefixtures("one_target_a_group")
class TestStages:
    """`aggregate` stops after the leaf files."""

    def test_aggregate_writes_the_leaves_and_no_statistics(self, weights_dir, tmp_path):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        result = run_pipeline(
            _leaves_cfg(weights_dir, tree, out, write_reduced=True),
            stage="aggregate",
        )
        assert result.stats is None
        assert result.paths == {
            "leaves_manifest": str(out / "leaves" / "statistics.leaves.manifest.json")
        }
        assert [p.name for p in out.iterdir()] == ["leaves"]
        assert _leaves_manifest(out)["complete"] is True
        assert len(result.leaf_reports) == 6
        for report in result.leaf_reports:
            assert report["rows_out"] == 8
            assert report["rows_reduced"] is None
            assert "frame" not in report and "leaf_file" not in report

    def test_aggregate_writes_the_files_that_all_writes(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        run_pipeline(_leaves_cfg(weights_dir, tree, tmp_path / "all"))
        run_pipeline(
            _leaves_cfg(weights_dir, tree, tmp_path / "aggregate"), stage="aggregate"
        )
        one = _leaves_manifest(tmp_path / "all")
        two = _leaves_manifest(tmp_path / "aggregate")
        assert one["package_versions"] == two["package_versions"]
        assert one == two

    def test_aggregate_reads_nothing_from_the_statistics_settings(
        self, weights_dir, tmp_path
    ):
        """The model weights file is not there and the leaves have no
        column of the time column's name. Both stop a full run."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        raw = _leaves_cfg(weights_dir, tree, out).model_dump()
        raw["stats"].update(
            time_col="decade",
            model_weights=str(tmp_path / "gone.tsv"),
            model_weight_level="zeppelin",
            quantile_method="lower_step",
        )
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        run_pipeline(cfg, stage="aggregate")
        assert _leaves_manifest(out)["complete"] is True
        with pytest.raises(FileNotFoundError):
            run_pipeline(cfg)

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_aggregate_needs_the_leaves_switched_on(
        self, weights_dir, tmp_path, dry_run
    ):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        cfg = _pipeline_cfg(weights_dir, tree, out, levels=_LEVELS)
        with pytest.raises(ValueError, match="needs output.write_leaves = true"):
            run_pipeline(cfg, stage="aggregate", dry_run=dry_run)
        assert not out.exists()

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_aggregate_refuses_a_passthrough_run(self, weights_dir, tmp_path, dry_run):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        raw = _leaves_cfg(weights_dir, tree, out).model_dump()
        del raw["weights"]
        with pytest.raises(ValueError, match=r"stage 'aggregate' needs \[weights\]"):
            run_pipeline(
                MonteCarloPipelineConfig.model_validate(raw),
                stage="aggregate",
                dry_run=dry_run,
            )
        assert not out.exists()

    def test_an_unknown_stage_is_refused(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        with pytest.raises(ValueError, match="unknown stage 'pool'"):
            run_pipeline(_leaves_cfg(weights_dir, tree, tmp_path / "out"), stage="pool")

    def test_the_dry_run_names_the_leaves_manifest_and_touches_nothing(
        self, weights_dir, tmp_path
    ):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        cfg = _leaves_cfg(weights_dir, tree, out)
        manifest = out / "leaves" / "statistics.leaves.manifest.json"

        plan = run_pipeline(cfg, dry_run=True, stage="aggregate")
        assert plan.summary().splitlines() == [
            "pipeline plan: 6 declared leaves",
            f"  leaves     -> {manifest}",
        ]
        plan = run_pipeline(cfg, dry_run=True)
        assert plan.summary().splitlines() == [
            "pipeline plan: 6 declared leaves",
            f"  statistics -> {out / 'statistics.parquet'}",
            f"  manifest   -> {out / 'statistics.manifest.json'}",
            f"  leaves     -> {manifest}",
        ]
        plan = run_pipeline(
            _pipeline_cfg(weights_dir, tree, out, levels=_LEVELS), dry_run=True
        )
        assert len(plan.summary().splitlines()) == 3
        assert not out.exists()

    @pytest.mark.parametrize("entry", ["module", "cilreg"])
    def test_the_stage_flag_on_both_entry_points(
        self, weights_dir, tmp_path, capsys, entry
    ):
        from cil_regionalization.cli import main as cilreg_main
        from cil_regionalization.pipelines.montecarlo import main

        tree, out = tmp_path / "tree", tmp_path / "out"
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
quantile_method = "linear"
summation = "fast"
time_reduction = "window_mean"
windows = [[2050, 2051]]

[output]
dir = "{out}"
write_leaves = true
"""
        )

        def run(*flags):
            if entry == "module":
                return main([str(cfg_path), *flags])
            return cilreg_main(["pipeline", str(cfg_path), *flags])

        assert run("--dry-run", "--stage", "aggregate") == 0
        printed = capsys.readouterr().out
        assert "statistics.leaves.manifest.json" in printed
        assert "statistics ->" not in printed
        assert not out.exists()

        assert run("--stage", "aggregate") == 0
        printed = capsys.readouterr().out
        assert "6 leaves aggregated; no statistics at stage aggregate" in printed
        assert "wrote leaves_manifest:" in printed
        assert [p.name for p in out.iterdir()] == ["leaves"]

        assert run() == 0
        assert "statistics rows=2" in capsys.readouterr().out
        assert (out / "statistics.parquet").exists()


_METHODS = list(
    itertools.product(["linear", "lower_step", "inverted_cdf"], ["fast", "exact"])
)
# "linear" has no weighted form, so it is tried without weights only
_METHODS_AND_WEIGHTS = pytest.mark.parametrize(
    "methods, weighted",
    [
        (methods, weighted)
        for methods in _METHODS
        for weighted in (False, True)
        if not (weighted and methods[0] == "linear")
    ],
    ids=lambda value: (
        "-".join(value)
        if isinstance(value, tuple)
        else "weighted"
        if value
        else "unweighted"
    ),
)
# One row group of the tree below, at 280 bytes a row with half its
# rows kept, is 6,720 bytes. This fits one group a worker and not two.
_ONE_GROUP_GB = 7000e-9


def _same_frames(one: pd.DataFrame, two: pd.DataFrame) -> None:
    """Equal with their dtypes and row order, and to the bit."""
    pd.testing.assert_frame_equal(one, two, check_exact=True)
    assert one["value"].to_numpy().tobytes() == two["value"].to_numpy().tobytes()


class TestTargetCoverage:
    """The statistics carry the share of each target's weight basis,
    the smallest any pooled leaf gave it, and the manifests list the
    targets left out and those that are partial."""

    @staticmethod
    def _raw(weights_dir, tmp_path) -> tuple[dict, Path]:
        """Leaf k0/z1 lacks u2, a share of T1 and the whole of T2."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        path = tree / "k0" / "z1" / "damages.parquet"
        leaf = pd.read_parquet(path)
        leaf[leaf["unit_id"] != "u2"].to_parquet(path, index=False)
        raw = _pipeline_cfg(weights_dir, tree, out, levels=_LEVELS).model_dump()
        raw["policies"] = {"on_absent_from_data": "skip"}
        raw["output"]["write_leaves"] = True
        return raw, out

    @staticmethod
    def _t1_share(weights_dir) -> float:
        from cil_regionalization.apply import WeightsArtifact

        frame = WeightsArtifact.load(weights_dir).frame
        basis = frame[frame["target_id"] == "T1"].set_index("unit_id")["area_raw"]
        return float(basis["u1"] / basis.sum())

    def test_a_target_short_of_a_unit_in_one_leaf_carries_its_share(
        self, weights_dir, tmp_path
    ):
        """Failed with no such column, and T1 averaged as if whole."""
        raw, out = self._raw(weights_dir, tmp_path)
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        share = self._t1_share(weights_dir)
        assert 0 < share < 1
        shares = result.stats.groupby("target_id")["weight_share"].unique()
        assert shares["T1"].tolist() == [share]
        assert shares["T2"].tolist() == [1.0]

        manifest = json.loads((out / "statistics.manifest.json").read_text())
        assert manifest["target_coverage"] == {
            "basis": "area_raw",
            "left_out": {
                "no weight": {"count": 0, "ids": []},
                "no data": {"count": 0, "ids": []},
            },
            "partial": {"count": 1, "smallest": share, "targets": [["T1", share]]},
        }
        leaves = _leaves_manifest(out)
        assert leaves["target_coverage"] == {
            "basis": "area_raw",
            "left_out": {"no weight": [], "no data": []},
        }
        by_leaf = {tuple(e["leaf"]): e["target_coverage"] for e in leaves["leaves"]}
        assert by_leaf["k0", "z1"] == {
            "left_out": {"no weight": [], "no data": [["T2"]]},
            "partial": [[["T1"], share]],
        }
        assert by_leaf["k1", "z2"] == {
            "left_out": {"no weight": [], "no data": []},
            "partial": [],
        }

    def test_a_reduce_gives_the_shares_of_the_leaves_it_pools(
        self, weights_dir, tmp_path
    ):
        """All of them as stage all does, and without the short leaf a
        share of one."""
        raw, out = self._raw(weights_dir, tmp_path)
        whole = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        again = TestReduce._reduce(raw)
        pd.testing.assert_series_equal(
            again.stats["weight_share"], whole.stats["weight_share"]
        )
        fewer = json.loads(json.dumps(raw))
        fewer["tree"]["exclude"] = [["k0", "z1"]]
        assert set(TestReduce._reduce(fewer, stem="fewer").stats["weight_share"]) == {
            1.0
        }

    def test_a_target_left_out_of_every_leaf_is_named_in_the_manifest(
        self, weights_dir, tmp_path
    ):
        raw, out = self._raw(weights_dir, tmp_path)
        tree = Path(raw["tree"]["root"])
        for path in tree.rglob("damages.parquet"):
            leaf = pd.read_parquet(path)
            leaf[leaf["unit_id"] != "u2"].to_parquet(path, index=False)
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        assert set(result.stats["target_id"]) == {"T1"}
        manifest = json.loads((out / "statistics.manifest.json").read_text())
        assert manifest["target_coverage"]["left_out"]["no data"] == {
            "count": 1,
            "ids": [["T2"]],
        }
        assert _leaves_manifest(out)["target_coverage"]["left_out"]["no data"] == [
            ["T2"]
        ]

    def test_a_leaves_manifest_without_the_targets_is_not_reduced(
        self, weights_dir, tmp_path
    ):
        raw, out = self._raw(weights_dir, tmp_path)
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        path = out / "leaves" / "statistics.leaves.manifest.json"
        document = json.loads(path.read_text())
        del document["target_coverage"]
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="before each leaf recorded its targets"):
            TestReduce._reduce(raw)


@pytest.mark.usefixtures("one_target_a_group")
class TestReduce:
    """`--stage reduce` takes the statistics from the leaf files."""

    @staticmethod
    def _raw(weights_dir, tree, out, *, weighted=False, methods=None, **stats):
        """A config over the tree of `TestLabelPlacement`, which has a
        dimension of its own. kumquat is not pooled. The quantiles sit
        above the first step of three weighted members, so the step
        gives a number and not minus infinity. A config that names no
        method gets the step with weights and interpolation without."""
        raw = _leaves_cfg(weights_dir, tree, out).model_dump()
        raw["stats"].update(sample_dims=["zeppelin"], quantiles=[0.75, 0.95], **stats)
        if methods is None:
            methods = ("lower_step" if weighted else "linear", "fast")
        raw["stats"].update(quantile_method=methods[0], summation=methods[1])
        if weighted:
            table = Path(tree).parent / "model_weights.tsv"
            table.write_text("model\tweight\nz0\t0.6\nz1\t0.3\nz2\t0.0\n")
            raw["stats"].update(model_weights=str(table), model_weight_level="zeppelin")
        return raw

    @staticmethod
    def _reduce(
        raw: dict,
        *,
        stem="again",
        memory_gb=1.0,
        n_workers=None,
        stats=None,
        **kwargs,
    ):
        raw = json.loads(json.dumps(raw))
        raw["stats"].update(stats or {})
        raw["output"].update(stem=stem, leaves_stem="statistics")
        raw["reduce"] = {"pandas": {"memory_gb": memory_gb, "n_workers": n_workers}}
        return run_pipeline(
            MonteCarloPipelineConfig.model_validate(raw), stage="reduce", **kwargs
        )

    def _aggregated(self, weights_dir, tmp_path, **kwargs):
        tree, out = tmp_path / "tree", tmp_path / "out"
        TestLabelPlacement._build_tree(tree)
        raw = self._raw(weights_dir, tree, out, **kwargs)
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        return raw, result, out

    @_METHODS_AND_WEIGHTS
    def test_it_gives_what_a_run_with_one_worker_gives(
        self, weights_dir, tmp_path, methods, weighted
    ):
        """The file and the frame, with dtypes and row order, to the
        bit, in one block and in one block a row group."""
        raw, whole, out = self._aggregated(
            weights_dir,
            tmp_path,
            weighted=weighted,
            methods=methods,
            include_n_members=True,
        )
        assert list(whole.stats.columns) == [
            "target_id",
            "band",
            "kumquat",
            "window",
            "statistic",
            "value",
            "n_members",
            "weight_share",
        ]
        for stem, memory_gb, blocks in (("one", 1.0, 1), ("two", _ONE_GROUP_GB, 2)):
            got = self._reduce(raw, stem=stem, memory_gb=memory_gb)
            _same_frames(got.stats, whole.stats)
            _same_frames(
                pd.read_parquet(out / f"{stem}.parquet"),
                pd.read_parquet(out / "statistics.parquet"),
            )
            manifest = json.loads((out / f"{stem}.manifest.json").read_text())
            assert manifest["reduce"]["blocks"] == blocks

    def test_two_workers_give_the_same_bits(self, weights_dir, tmp_path):
        raw, whole, out = self._aggregated(weights_dir, tmp_path, weighted=True)
        got = self._reduce(raw, memory_gb=2 * _ONE_GROUP_GB, n_workers=2)
        _same_frames(got.stats, whole.stats)
        manifest = json.loads((out / "again.manifest.json").read_text())
        assert manifest["reduce"]["blocks"] == 2
        assert manifest["reduce"]["options"] == {
            "memory_gb": 2 * _ONE_GROUP_GB,
            "n_workers": 2,
        }

    def test_other_windows_need_neither_the_tree_nor_the_weights(
        self, weights_dir, tmp_path, recwarn
    ):
        """They are deleted before the reduce, which then gives what a
        full run with those windows gave."""
        import shutil

        raw, _, out = self._aggregated(weights_dir, tmp_path)
        fresh = json.loads(json.dumps(raw))
        fresh["stats"]["windows"] = [[2050, 2050], [2051, 2081]]
        fresh["output"].update(dir=str(tmp_path / "fresh"), write_leaves=False)
        expected = run_pipeline(MonteCarloPipelineConfig.model_validate(fresh)).stats

        shutil.rmtree(tmp_path / "tree")
        shutil.rmtree(weights_dir)
        got = self._reduce(raw, stats={"windows": [[2050, 2050], [2051, 2081]]})
        _same_frames(got.stats, expected)
        assert sorted(got.stats["window"].unique()) == ["2050", "2051-2081"]
        notes = json.loads((out / "again.manifest.json").read_text())["notes"]
        assert len(notes) == 1 and "weights artifact is not at" in notes[0]
        assert any("weights artifact is not at" in str(w.message) for w in recwarn)

    def test_the_manifest_says_how_the_statistics_were_reduced(
        self, weights_dir, tmp_path
    ):
        raw, _, out = self._aggregated(weights_dir, tmp_path, weighted=True)
        result = self._reduce(raw, memory_gb=_ONE_GROUP_GB)
        manifest = json.loads((out / "again.manifest.json").read_text())
        leaves = out / "leaves" / "statistics.leaves.manifest.json"
        assert manifest["stage"] == "reduce"
        assert manifest["leaves_manifest"] == str(leaves)
        assert (
            manifest["leaves_manifest_sha256"]
            == hashlib.sha256(leaves.read_bytes()).hexdigest()
        )
        assert manifest["leaves_left_out"] == []
        assert manifest["leaves_verified"] is False
        assert manifest["n_members_forced"] == []
        assert manifest["notes"] == []
        assert manifest["reduced"] is None
        reduce = manifest["reduce"]
        assert reduce["kind"] == "pandas"
        assert sorted(reduce["engine"]) == ["numpy", "pandas", "pyarrow"]
        assert reduce["engine"]["pandas"] == pd.__version__
        assert (reduce["blocks"], reduce["opens_per_leaf"], reduce["opens"]) == (
            2,
            2,
            12,
        )
        assert reduce["failed_opens"] == 0
        assert reduce["rows_pooled"] == 48
        # 240, and 40 for the band
        assert reduce["bytes_per_row"] == 280
        assert reduce["kept_share"] == 0.5
        assert (
            manifest["artifact_sha256"]
            == _leaves_manifest(out)["aggregation"]["artifact"]["sha256"]
        )
        assert manifest["n_leaves"] == 6 and len(manifest["leaves"]) == 6
        assert manifest["model_weights_record"]["members"]["z0"] == {
            "row": "z0",
            "weight": 0.6,
            "n_leaves": 2,
        }
        assert result.paths == {
            "statistics": str(out / "again.parquet"),
            "manifest": str(out / "again.manifest.json"),
            "leaves_manifest": str(leaves),
        }
        assert len(result.leaf_reports) == 6

    def test_fewer_leaves_than_were_aggregated_shows_in_a_column(
        self, weights_dir, tmp_path
    ):
        """The config does not ask for the member counts. They are
        written all the same, because the statistics are over fewer
        members than the leaves on disk and nothing else would say so."""
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        fewer = json.loads(json.dumps(raw))
        fewer["tree"]["exclude"] = [["k0", "z1"], ["k1", "z1"]]
        with pytest.warns(UserWarning, match="n_members is written although"):
            got = self._reduce(fewer)
        assert "n_members" in got.stats.columns
        assert set(got.stats["n_members"]) == {2}
        manifest = json.loads((out / "again.manifest.json").read_text())
        assert manifest["leaves_left_out"] == [["k0", "z1"], ["k1", "z1"]]
        assert manifest["n_members_forced"] == [
            "2 of the 6 aggregated leaves are not declared"
        ]
        assert manifest["n_leaves"] == 4

        fresh = json.loads(json.dumps(fewer))
        fresh["stats"]["include_n_members"] = True
        fresh["output"].update(dir=str(tmp_path / "fresh"), write_leaves=False)
        expected = run_pipeline(MonteCarloPipelineConfig.model_validate(fresh)).stats
        _same_frames(got.stats, expected)

    def test_windows_that_differ_between_leaves_show_in_a_column(
        self, weights_dir, tmp_path
    ):
        """No config gives a leaf its own windows yet. The request
        takes them, and the member counts are then switched on."""
        from cil_regionalization.pipelines import montecarlo

        raw, _, out = self._aggregated(weights_dir, tmp_path)
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        plan = montecarlo.resolve_plan(cfg, "reduce")
        document = _leaves_manifest(out)
        request, reasons = montecarlo._reduce_request(cfg, plan, document, None)
        assert reasons == [] and request.include_n_members is False
        assert {leaf.windows for leaf in request.leaves} == {
            (("2050-2051", 2050, 2051), ("2080-2081", 2080, 2081))
        }

        def windows_of(values):
            first = 2050 if values[1] == "z0" else 2051
            return [("warm", first, first + 30)]

        request, reasons = montecarlo._reduce_request(
            cfg, plan, document, None, windows_of=windows_of
        )
        assert reasons == ["the windows differ between leaves"]
        assert request.include_n_members is True
        assert request.window_labels == ("2050-2051", "2080-2081", "warm")
        assert request.leaves[0].windows == (("warm", 2050, 2080),)
        assert request.leaves[1].windows == (("warm", 2051, 2081),)

    def test_a_copy_of_the_weights_elsewhere_is_a_note_and_not_a_refusal(
        self, weights_dir, tmp_path
    ):
        import shutil

        raw, whole, out = self._aggregated(weights_dir, tmp_path)
        moved = tmp_path / "moved" / "weights"
        shutil.copytree(weights_dir, moved)
        elsewhere = json.loads(json.dumps(raw))
        elsewhere["weights"]["dir"] = str(moved)
        with pytest.warns(UserWarning, match="weights_dir was .* Paths are recorded"):
            got = self._reduce(elsewhere)
        _same_frames(got.stats, whole.stats)
        notes = json.loads((out / "again.manifest.json").read_text())["notes"]
        assert len(notes) == 1 and str(moved) in notes[0]

    def test_other_weights_under_the_same_name_are_refused(self, weights_dir, tmp_path):
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        table = pd.read_parquet(weights_dir / "weights.parquet")
        table["areawt"] = table["areawt"] * 0.5
        table.to_parquet(weights_dir / "weights.parquet", index=False)
        with pytest.raises(ValueError, match="is not the one the leaves were"):
            self._reduce(raw)
        assert not (out / "again.parquet").exists()

    @pytest.mark.parametrize(
        "section, key, value, message",
        [
            ("weights", "weight", "pop", "weight = 'area' and the config has 'pop'"),
            ("data", "kind", "intensive", "kind = 'extensive' and the config has"),
            ("data", "version", "units-v2", "data_version = 'units-v1'"),
            ("tree", "filename", "other.parquet", "filename = 'damages.parquet'"),
            ("policies", "on_unmatched", "skip", "policies = "),
        ],
    )
    def test_a_config_that_would_aggregate_differently_is_refused(
        self, weights_dir, tmp_path, section, key, value, message
    ):
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        other = json.loads(json.dumps(raw))
        other[section][key] = value
        with pytest.raises(ValueError, match=message):
            self._reduce(other)
        assert not (out / "again.parquet").exists()

    def test_leaves_without_a_manifest_are_refused(self, weights_dir, tmp_path):
        """A directory of leaf files of the old layout has none."""
        tree, out = tmp_path / "tree", tmp_path / "out"
        TestLabelPlacement._build_tree(tree)
        raw = self._raw(weights_dir, tree, out)
        old = out / "leaves" / "k0" / "z0" / "aggregated_damages.parquet"
        old.parent.mkdir(parents=True)
        pd.DataFrame({"target_id": ["T1"], "value": [1.0]}).to_parquet(old)
        with pytest.raises(ValueError, match="it is not there.*output.leaves_stem"):
            self._reduce(raw)
        with pytest.raises(ValueError, match="it is not there"):
            self._reduce(raw, dry_run=True)

    def test_an_incomplete_manifest_is_refused(self, weights_dir, tmp_path):
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        path = out / "leaves" / "statistics.leaves.manifest.json"
        document = json.loads(path.read_text())
        path.write_text(json.dumps({**document, "complete": False}))
        with pytest.raises(ValueError, match="is incomplete"):
            self._reduce(raw)

    def test_a_leaf_file_that_is_gone_or_of_another_size_is_refused(
        self, weights_dir, tmp_path
    ):
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        leaf = out / "leaves" / "k1" / "z0" / "damages.aggregated.parquet"
        whole = leaf.read_bytes()
        leaf.write_bytes(whole + b"x")
        with pytest.raises(ValueError, match="bytes and the leaves manifest has"):
            self._reduce(raw)
        leaf.unlink()
        with pytest.raises(ValueError, match="in the leaves manifest and not on disk"):
            self._reduce(raw)
        plan = self._reduce(raw, dry_run=True)
        assert [m.values for m in plan.missing] == [("k1", "z0")]
        assert not (out / "again.parquet").exists()

    def test_a_leaf_whose_footer_says_another_aggregation_is_refused(
        self, weights_dir, tmp_path
    ):
        """The manifest is given another digest here, which is the same
        disagreement seen from the other side."""
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        path = out / "leaves" / "statistics.leaves.manifest.json"
        document = json.loads(path.read_text())
        path.write_text(json.dumps({**document, "aggregation_digest": "0" * 64}))
        with pytest.raises(ValueError, match="its footer has aggregation_digest"):
            self._reduce(raw)
        assert not (out / "again.parquet").exists()

    def test_a_leaf_changed_in_place_passes_unless_the_leaves_are_verified(
        self, weights_dir, tmp_path, capsys
    ):
        """One value of one leaf is changed, in a file of the same size
        with the same footer. Only the hash can tell."""
        import pyarrow.parquet as pq

        from cil_regionalization.leaves import (
            TargetTable,
            read_leaf_footer,
            write_leaf,
        )

        raw, whole, out = self._aggregated(weights_dir, tmp_path)
        path = out / "leaves" / "k1" / "z2" / "damages.aggregated.parquet"
        size, footer = path.stat().st_size, read_leaf_footer(path)
        frame = pq.read_table(path).to_pandas().drop(columns="target_ordinal")
        frame.loc[3, "value"] = frame.loc[3, "value"] + 0.125
        document = _leaves_manifest(out)
        write_leaf(
            frame,
            path,
            targets=TargetTable.read(out / "leaves" / document["targets"]["file"]),
            value_col="value",
            identity={
                k: footer[k]
                for k in ("aggregation", "aggregation_digest", "levels", "sources")
            },
            targets_per_group=1,
        )
        assert path.stat().st_size == size
        assert read_leaf_footer(path) == footer

        got = self._reduce(raw)
        assert not got.stats["value"].equals(whole.stats["value"])
        manifest = json.loads((out / "again.manifest.json").read_text())
        assert manifest["leaves_verified"] is False

        with pytest.raises(
            ValueError, match="1 of 6 leaf files do not hash .*k1/z2/damages"
        ):
            self._reduce(raw, stem="checked", verify_leaves=True)
        assert not (out / "checked.parquet").exists()

    def test_verified_leaves_are_said_to_be_so(self, weights_dir, tmp_path, capsys):
        raw, whole, out = self._aggregated(weights_dir, tmp_path)
        got = self._reduce(raw, verify_leaves=True)
        _same_frames(got.stats, whole.stats)
        assert (
            "leaves: 6 files hash to what the manifest has" in capsys.readouterr().out
        )
        manifest = json.loads((out / "again.manifest.json").read_text())
        assert manifest["leaves_verified"] is True

    def test_verifying_goes_with_the_reduce_stage_only(self, weights_dir, tmp_path):
        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        with pytest.raises(ValueError, match="goes with stage 'reduce'"):
            run_pipeline(
                _leaves_cfg(weights_dir, tree, tmp_path / "out"), verify_leaves=True
            )

    def test_a_budget_is_needed_and_has_to_fit_a_row_group(self, weights_dir, tmp_path):
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        with pytest.raises(ValueError, match=r"Set \[reduce.pandas\] memory_gb"):
            self._reduce(raw, memory_gb=None)
        with pytest.raises(
            ValueError,
            match="one row group of 1 targets needs 6.72e-06 GB. The smallest "
            "budget that runs with 3 workers is 2.02e-05 GB",
        ):
            self._reduce(raw, memory_gb=_ONE_GROUP_GB, n_workers=3)
        with pytest.raises(ValueError, match="smallest budget"):
            self._reduce(raw, memory_gb=_ONE_GROUP_GB, n_workers=3, dry_run=True)
        assert not (out / "again.parquet").exists()

    def test_the_reduced_file_is_refused_by_name(self, weights_dir, tmp_path):
        raw, _, _ = self._aggregated(weights_dir, tmp_path)
        raw["output"]["write_reduced"] = True
        with pytest.raises(ValueError, match="output.write_reduced = false"):
            self._reduce(raw)

    def test_a_passthrough_config_is_refused(self, weights_dir, tmp_path):
        raw, _, _ = self._aggregated(weights_dir, tmp_path)
        del raw["weights"]
        with pytest.raises(ValueError, match=r"stage 'reduce' needs \[weights\]"):
            self._reduce(raw)

    def test_a_leaf_that_was_not_aggregated_is_refused(self, weights_dir, tmp_path):
        tree, out = tmp_path / "tree", tmp_path / "out"
        TestLabelPlacement._build_tree(tree)
        raw = self._raw(weights_dir, tree, out)
        half = json.loads(json.dumps(raw))
        half["tree"]["expect"]["kumquat"] = ["k0"]
        run_pipeline(MonteCarloPipelineConfig.model_validate(half))
        with pytest.raises(
            ValueError, match="3 declared leaves are not in the leaves manifest"
        ):
            self._reduce(raw)

    def test_a_window_that_selects_nothing_in_a_leaf_is_refused_before_work(
        self, weights_dir, tmp_path, monkeypatch
    ):
        from cil_regionalization import leaves

        raw, _, out = self._aggregated(weights_dir, tmp_path)

        def read_groups(*args, **kwargs):
            raise AssertionError("a leaf was opened")

        monkeypatch.setattr(leaves, "read_groups", read_groups)
        with pytest.raises(
            ValueError,
            match=r"window \(2060.0, 2070.0\) on 'year' selects no rows in leaf "
            r"k0/z0 \(it spans 2050 to 2081\)",
        ):
            self._reduce(raw, stats={"windows": [[2050, 2051], [2060, 2070]]})
        with pytest.raises(ValueError, match="'decade' is not a column"):
            self._reduce(raw, stats={"time_col": "decade"})

    def test_the_dry_run_says_what_it_would_do_and_touches_nothing(
        self, weights_dir, tmp_path
    ):
        raw, _, out = self._aggregated(weights_dir, tmp_path)
        before = sorted(p.name for p in out.iterdir())
        plan = self._reduce(raw, memory_gb=_ONE_GROUP_GB, dry_run=True)
        assert plan.summary().splitlines() == [
            "pipeline plan: 6 declared leaves",
            f"  statistics -> {out / 'again.parquet'}",
            f"  manifest   -> {out / 'again.manifest.json'}",
            f"  leaves     -> {out / 'leaves' / 'statistics.leaves.manifest.json'}",
            "  reduce: pandas, 7e-06 GB over 1 workers, 7e-06 GB a worker",
            "  2 blocks of up to 1 targets and 24 rows",
            "  2 opens a leaf, 12 in all",
            "  sized at 280 bytes a row, 240 and 40 for each dimension of a "
            "leaf, with 0.5 of a group's rows kept. The "
            "rows of a group are known summed over the leaves, so this takes "
            "every leaf to spread its rows over the groups alike.",
            f"  leaves: every file under {out / 'leaves'} is in a leaves manifest",
        ]
        assert plan.missing == []
        assert sorted(p.name for p in out.iterdir()) == before

    def test_an_unweighted_model_level_warns_once(self, weights_dir, tmp_path):
        """In the parent, not once a block in every worker."""
        import warnings as warnings_

        from cil_regionalization.stats import UnweightedModelWeightsWarning

        levels = {"gcm": ["a", "b"], "draw": ["d0", "d1"]}
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, levels)
        raw = _leaves_cfg(weights_dir, tree, out, levels=levels).model_dump()
        with warnings_.catch_warnings():
            warnings_.simplefilter("ignore")
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        with pytest.warns(UnweightedModelWeightsWarning) as caught:
            self._reduce(raw, memory_gb=2900e-9)
        assert (
            len([w for w in caught if w.category is UnweightedModelWeightsWarning]) == 1
        )
        manifest = json.loads((out / "again.manifest.json").read_text())
        assert manifest["reduce"]["blocks"] == 2

    @pytest.mark.parametrize("entry", ["module", "cilreg"])
    def test_the_stage_and_the_verification_on_both_entry_points(
        self, weights_dir, tmp_path, capsys, entry
    ):
        from cil_regionalization.cli import main as cilreg_main
        from cil_regionalization.pipelines.montecarlo import main

        tree, out = tmp_path / "tree", tmp_path / "out"
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
quantile_method = "linear"
summation = "fast"
time_reduction = "window_mean"
windows = [[2050, 2051]]

[output]
dir = "{out}"
write_leaves = true

[reduce.pandas]
memory_gb = 1
"""
        )

        def run(*flags):
            if entry == "module":
                return main([str(cfg_path), *flags])
            return cilreg_main(["pipeline", str(cfg_path), *flags])

        assert run("--stage", "aggregate") == 0
        capsys.readouterr()
        assert run("--stage", "reduce", "--dry-run") == 0
        printed = capsys.readouterr().out
        assert "reduce: pandas, 1 GB over 1 workers" in printed
        assert not (out / "statistics.parquet").exists()

        assert run("--stage", "reduce", "--verify-leaves") == 0
        printed = capsys.readouterr().out
        assert "leaves: 6 files hash to what the manifest has" in printed
        assert "6 leaves reduced; statistics rows=2" in printed
        assert (out / "statistics.parquet").exists()


# One row group of that tree with every row kept, as pooled years keep
# them, is 13,440 bytes.
_ONE_POOLED_GROUP_GB = 14000e-9


@pytest.mark.usefixtures("one_target_a_group")
class TestPooledYears:
    """``stats.time_reduction = "pooled"`` pools the years inside a window with the
    models and the draws, from the leaf files, under every stage."""

    _MODEL_WEIGHTS = {"z0": 0.6, "z1": 0.3, "z2": 0.0}

    @staticmethod
    def _raw(weights_dir, tmp_path, *, memory_gb=1.0, n_workers=1, **kwargs):
        tree, out = tmp_path / "tree", tmp_path / "out"
        if not tree.exists():
            TestLabelPlacement._build_tree(tree)
        raw = TestReduce._raw(weights_dir, tree, out, **kwargs)
        raw["stats"]["time_reduction"] = "pooled"
        raw["run"]["n_workers"] = n_workers
        raw["reduce"] = {"pandas": {"memory_gb": memory_gb}}
        return raw, out

    def _by_hand(self, cfg: MonteCarloPipelineConfig, out: Path) -> pd.DataFrame:
        """The statistics from the leaf files in one call. A row is in
        the window that covers its year, and the year is a sample
        dimension. The members are counted apart."""
        from cil_regionalization.leaves import ORDINAL_COLUMN
        from cil_regionalization.pipelines.montecarlo import _ordered
        from cil_regionalization.stats import pooled_statistics

        pieces = []
        for kumquat, zeppelin in itertools.product(*_LEVELS.values()):
            leaf = pd.read_parquet(
                out / "leaves" / kumquat / zeppelin / "damages.aggregated.parquet"
            ).drop(columns=[ORDINAL_COLUMN])
            for lo, hi in cfg.stats.windows:
                pieces.append(
                    leaf[leaf["year"].between(lo, hi)].assign(
                        kumquat=kumquat,
                        zeppelin=zeppelin,
                        window=f"{int(lo)}-{int(hi)}",
                    )
                )
        tagged = pd.concat(pieces, ignore_index=True)[
            ["target_id", "band", "kumquat", "window", "zeppelin", "year", "value"]
        ]
        weighted = cfg.stats.model_weights is not None
        if weighted:
            tagged["model_weight"] = tagged["zeppelin"].map(self._MODEL_WEIGHTS)
        stats = pooled_statistics(
            tagged,
            sample_dims=["zeppelin", "year"],
            value_col="value",
            quantiles=cfg.stats.quantiles,
            weight_col="model_weight" if weighted else None,
            include_n_members=True,
            quantile_method=cfg.stats.quantile_method,
            summation=cfg.stats.summation,
        ).rename(columns={"n_members": "n_sample"})
        counted = tagged[tagged["model_weight"] > 0] if weighted else tagged
        members = counted.groupby(["target_id", "band", "kumquat", "window"])[
            "zeppelin"
        ].nunique()
        stats["n_members"] = [
            members[key]
            for key in zip(
                stats["target_id"], stats["band"], stats["kumquat"], stats["window"]
            )
        ]
        columns = [c for c in stats.columns if c != "n_sample"] + ["n_sample"]
        return _ordered(stats[columns], cfg).assign(weight_share=1.0)

    @_METHODS_AND_WEIGHTS
    def test_stage_all_writes_the_leaves_and_pools_their_years(
        self, weights_dir, tmp_path, methods, weighted
    ):
        raw, out = self._raw(
            weights_dir,
            tmp_path,
            weighted=weighted,
            methods=methods,
            include_n_members=True,
        )
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)
        assert list(result.stats.columns) == [
            "target_id",
            "band",
            "kumquat",
            "window",
            "statistic",
            "value",
            "n_members",
            "n_sample",
            "weight_share",
        ]
        assert _leaves_manifest(out)["complete"] is True
        _same_frames(result.stats, self._by_hand(cfg, out))
        _same_frames(pd.read_parquet(out / "statistics.parquet"), result.stats)
        # three models and two years a window, and z2 weighs nothing
        assert set(result.stats["n_members"]) == {2 if weighted else 3}
        assert set(result.stats["n_sample"]) == {4 if weighted else 6}
        assert len(result.leaf_reports) == 6

    def test_the_counts_are_written_only_if_asked_for(self, weights_dir, tmp_path):
        raw, _ = self._raw(weights_dir, tmp_path)
        stats = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
        assert list(stats.columns) == [
            "target_id",
            "band",
            "kumquat",
            "window",
            "statistic",
            "value",
            "weight_share",
        ]

    def test_the_mean_stays_and_the_quantiles_move(self, weights_dir, tmp_path):
        """Every member of this tree has both years of a window, so the
        mean over member-years is the mean of the window means, here to
        the rounding of the two sums. The members are the same, and
        there are two rows of each."""
        raw, out = self._raw(weights_dir, tmp_path, include_n_members=True)
        years = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
        raw["stats"]["time_reduction"] = "window_mean"
        raw["output"].update(dir=str(tmp_path / "means"), write_leaves=False)
        means = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats

        keys = ["target_id", "band", "kumquat", "window", "statistic"]
        pd.testing.assert_frame_equal(years[keys], means[keys])
        is_mean = (years["statistic"] == "mean").to_numpy()
        pd.testing.assert_series_equal(
            years.loc[is_mean, "value"],
            means.loc[is_mean, "value"],
            check_exact=False,
            rtol=1e-13,
        )
        assert (years.loc[~is_mean, "value"] != means.loc[~is_mean, "value"]).all()
        assert (years["n_members"] == means["n_members"]).all()
        assert (years["n_sample"] == 2 * means["n_members"]).all()
        assert "n_sample" not in means.columns

    def test_the_reduce_stage_gives_what_stage_all_gave(self, weights_dir, tmp_path):
        """In one block, in a block a row group, and with two workers."""
        raw, out = self._raw(
            weights_dir, tmp_path, weighted=True, include_n_members=True
        )
        whole = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        for stem, memory_gb, n_workers, blocks in (
            ("one", 1.0, None, 1),
            ("two", _ONE_POOLED_GROUP_GB, None, 2),
            ("pool", 2 * _ONE_POOLED_GROUP_GB, 2, 2),
        ):
            got = TestReduce._reduce(
                raw, stem=stem, memory_gb=memory_gb, n_workers=n_workers
            )
            _same_frames(got.stats, whole.stats)
            manifest = json.loads((out / f"{stem}.manifest.json").read_text())
            assert manifest["reduce"]["blocks"] == blocks
            assert manifest["stage"] == "reduce"
            assert manifest["time_reduction"] == "pooled"

    def test_workers_in_both_passes_change_no_bit(self, weights_dir, tmp_path):
        raw, _ = self._raw(weights_dir, tmp_path, weighted=True)
        one = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
        raw, out = self._raw(
            weights_dir,
            tmp_path,
            weighted=True,
            n_workers=2,
            memory_gb=2 * _ONE_POOLED_GROUP_GB,
        )
        raw["output"]["dir"] = str(tmp_path / "pool")
        two = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
        _same_frames(one, two)
        manifest = json.loads(
            (tmp_path / "pool" / "statistics.manifest.json").read_text()
        )
        assert manifest["reduce"]["blocks"] == 2
        assert manifest["reduce"]["options"]["n_workers"] == 2

    def test_the_manifest_says_the_years_were_pooled(self, weights_dir, tmp_path):
        raw, out = self._raw(weights_dir, tmp_path, weighted=True)
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        manifest = json.loads((out / "statistics.manifest.json").read_text())
        leaves = out / "leaves" / "statistics.leaves.manifest.json"
        assert manifest["stage"] == "all"
        assert manifest["time_reduction"] == "pooled"
        assert manifest["leaves_manifest"] == str(leaves)
        assert (
            manifest["leaves_manifest_sha256"]
            == hashlib.sha256(leaves.read_bytes()).hexdigest()
        )
        assert manifest["reduced"] is None
        assert manifest["reduce"]["kind"] == "pandas"
        assert manifest["reduce"]["kept_share"] == 1.0
        assert manifest["reduce"]["rows_pooled"] == 96
        assert manifest["n_leaves"] == 6
        assert manifest["model_weights_record"]["members"]["z2"]["weight"] == 0.0
        assert result.paths == {
            "statistics": str(out / "statistics.parquet"),
            "manifest": str(out / "statistics.manifest.json"),
            "leaves_manifest": str(leaves),
        }

    def test_a_run_of_window_means_records_that_no_year_was_pooled(
        self, weights_dir, tmp_path
    ):
        raw, out = self._raw(weights_dir, tmp_path)
        raw["stats"]["time_reduction"] = "window_mean"
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        TestReduce._reduce(raw)
        for stem in ("statistics", "again"):
            manifest = json.loads((out / f"{stem}.manifest.json").read_text())
            assert manifest["time_reduction"] == "window_mean"

    def test_what_stage_all_needs_is_asked_before_any_leaf_is_read(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """Pass 1 can take hours. A missing budget, leaves that would
        not be written and a reduced file that cannot be are refused
        before it starts."""
        from cil_regionalization.pipelines import montecarlo

        def read_leaf(*args):
            raise AssertionError("a leaf was read")

        monkeypatch.setattr(montecarlo, "_read_leaf", read_leaf)
        for change, message in (
            (
                {"reduce": {"pandas": {"memory_gb": None}}},
                r"Set \[reduce.pandas\] memory_gb",
            ),
            (
                {"output": {"write_leaves": False}},
                "stage 'all' needs output.write_leaves = true",
            ),
            (
                {"output": {"write_reduced": True}},
                "stats.time_reduction = 'pooled' cannot write the reduced file.*"
                "output.write_reduced = false",
            ),
        ):
            raw, out = self._raw(weights_dir, tmp_path)
            for section, values in change.items():
                raw[section] = {**raw.get(section, {}), **values}
            for dry_run in (True, False):
                with pytest.raises(ValueError, match=message):
                    run_pipeline(
                        MonteCarloPipelineConfig.model_validate(raw), dry_run=dry_run
                    )
            assert not out.exists()

    def test_the_first_pass_takes_no_window_mean_and_reads_the_weights_once(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """The window means of a leaf would be thrown away, and a table
        read again for the second pass could have changed in between."""
        from cil_regionalization.pipelines import montecarlo

        def window_means(*args, **kwargs):
            raise AssertionError("a window mean was taken")

        real, reads = montecarlo.read_model_weights, []

        def read_model_weights(path):
            reads.append(path)
            return real(path)

        monkeypatch.setattr(montecarlo, "window_means", window_means)
        monkeypatch.setattr(montecarlo, "read_model_weights", read_model_weights)
        raw, _ = self._raw(weights_dir, tmp_path, weighted=True)
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        assert len(reads) == 1

    def test_a_backend_that_cannot_give_the_methods_is_refused(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """Before any leaf is read under stage 'all', and before any
        block under stage 'reduce'."""
        from cil_regionalization.pipelines import montecarlo
        from cil_regionalization.reduce.pandas_blocks import PandasBlocks

        class ExactOnly(PandasBlocks):
            kind = "exact-only"
            supports = frozenset({("inverted_cdf", "exact")})

            def run(self, request):
                raise AssertionError("a block was reduced")

        raw, out = self._raw(weights_dir, tmp_path)
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw), stage="aggregate")
        monkeypatch.setattr(
            montecarlo, "get_backend", lambda config: ExactOnly(config.pandas)
        )
        message = (
            "the 'exact-only' reduce backend does not give "
            "quantile_method='linear' with summation='fast' to the last bit"
        )
        with pytest.raises(ValueError, match=message):
            TestReduce._reduce(raw)

        def read_leaf(*args):
            raise AssertionError("a leaf was read")

        monkeypatch.setattr(montecarlo, "_read_leaf", read_leaf)
        with pytest.raises(ValueError, match=message):
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        assert not (out / "statistics.parquet").exists()

    def test_the_reduce_stage_does_not_send_a_reduced_file_to_stage_all(
        self, weights_dir, tmp_path
    ):
        """Without pooled years the reduce stage says to use stage
        'all' for the reduced file. With them no stage writes it."""
        raw, _ = self._raw(weights_dir, tmp_path)
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        raw["output"]["write_reduced"] = True
        with pytest.raises(ValueError) as refused:
            TestReduce._reduce(raw)
        assert "stats.time_reduction = 'pooled' cannot write the reduced file" in str(
            refused.value
        )
        assert "use stage 'all'" not in str(refused.value)

    def test_a_dry_run_with_the_weights_table_gone_still_returns(
        self, weights_dir, tmp_path
    ):
        raw, out = self._raw(weights_dir, tmp_path, weighted=True)
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        run_pipeline(cfg)
        Path(raw["stats"]["model_weights"]).unlink()
        lines = run_pipeline(cfg, dry_run=True).summary().splitlines()
        assert lines[-1].startswith(
            "  the leaves on disk now do not say how the blocks will be sized: "
        )

    def test_configs_that_cannot_pool_the_years_are_refused_when_read(
        self, weights_dir, tmp_path
    ):
        raw, _ = self._raw(weights_dir, tmp_path)
        passthrough = json.loads(json.dumps(raw))
        del passthrough["weights"]
        with pytest.raises(
            ValueError, match=r"stats.time_reduction = 'pooled' needs \[weights\]"
        ):
            MonteCarloPipelineConfig.model_validate(passthrough)
        raw["stats"]["sample_dims"] = ["zeppelin", "year"]
        with pytest.raises(
            ValueError,
            match="decides what happens to stats.time_col 'year' itself. Leave it "
            "out of stats.sample_dims",
        ):
            MonteCarloPipelineConfig.model_validate(raw)
        raw["stats"]["time_reduction"] = "window_mean"
        MonteCarloPipelineConfig.model_validate(raw)

    def test_a_leaf_column_named_like_the_count_of_rows_is_refused(
        self, weights_dir, tmp_path
    ):
        """It runs under window means, which write no such column."""
        tree = tmp_path / "tree"
        TestLabelPlacement._build_tree(tree)
        for path in tree.rglob("damages.parquet"):
            pd.read_parquet(path).rename(columns={"band": "n_sample"}).to_parquet(
                path, index=False
            )
        raw, _ = self._raw(weights_dir, tmp_path)
        raw["stats"]["time_reduction"] = "window_mean"
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        with pytest.raises(ValueError, match="a column named 'n_sample'"):
            TestReduce._reduce(raw, stats={"time_reduction": "pooled"})

    def test_a_second_pass_that_fails_leaves_the_leaves_and_says_so(
        self, weights_dir, tmp_path
    ):
        """The blocks are sized from the leaves manifest, which only
        exists after the first pass. A budget too small for one row
        group stops the run there, with the leaves written."""
        raw, out = self._raw(weights_dir, tmp_path, memory_gb=1e-9)
        with pytest.raises(
            ValueError,
            match="smallest budget that runs with 1 workers is 1.34e-05 GB. The "
            "leaves are written and their manifest is complete .*Stage "
            "'reduce' takes the statistics from them without aggregating again",
        ):
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        assert _leaves_manifest(out)["complete"] is True
        assert not (out / "statistics.parquet").exists()
        got = TestReduce._reduce(raw, stem="statistics")
        raw["reduce"]["pandas"]["memory_gb"] = 1.0
        raw["output"]["dir"] = str(tmp_path / "whole")
        whole = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        _same_frames(got.stats, whole.stats)

    @pytest.mark.parametrize("kind", [ValueError, RuntimeError])
    def test_a_second_pass_error_that_takes_more_than_a_message_is_still_raised(
        self, weights_dir, tmp_path, monkeypatch, kind
    ):
        """Failed with the error built again from its own type, which
        for a JSON error is a TypeError about missing arguments."""
        from cil_regionalization.pipelines import montecarlo

        original = (
            json.JSONDecodeError("not a manifest", "{", 0)
            if kind is ValueError
            else RuntimeError("a worker died")
        )

        def run_reduce(*args, **kwargs):
            raise original

        monkeypatch.setattr(montecarlo, "_run_reduce", run_reduce)
        raw, _ = self._raw(weights_dir, tmp_path)
        with pytest.raises(kind, match="The leaves are written") as raised:
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        assert raised.value.__cause__ is original

    def test_aggregate_needs_no_budget_and_writes_the_same_leaves(
        self, weights_dir, tmp_path
    ):
        raw, out = self._raw(weights_dir, tmp_path, memory_gb=None)
        result = run_pipeline(
            MonteCarloPipelineConfig.model_validate(raw), stage="aggregate"
        )
        assert result.stats is None
        pooled = _leaves_manifest(out)
        raw["stats"]["time_reduction"] = "window_mean"
        raw["output"]["dir"] = str(tmp_path / "means")
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw), stage="aggregate")
        means = _leaves_manifest(tmp_path / "means")
        assert pooled["aggregation_digest"] == means["aggregation_digest"]
        assert [e["sha256"] for e in pooled["leaves"]] == [
            e["sha256"] for e in means["leaves"]
        ]

    def test_the_dry_run_of_stage_all_says_when_the_blocks_are_sized(
        self, weights_dir, tmp_path
    ):
        raw, out = self._raw(weights_dir, tmp_path, memory_gb=_ONE_POOLED_GROUP_GB)
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        plan = run_pipeline(cfg, dry_run=True)
        head = [
            "pipeline plan: 6 declared leaves",
            f"  statistics -> {out / 'statistics.parquet'}",
            f"  manifest   -> {out / 'statistics.manifest.json'}",
            f"  leaves     -> {out / 'leaves' / 'statistics.leaves.manifest.json'}",
            "  reduce: stats.time_reduction = 'pooled' takes the statistics from the leaf "
            "files, by blocks, after every leaf is written",
        ]
        assert plan.summary().splitlines() == head + [
            "  the blocks are sized once the leaves are written",
        ]
        assert not out.exists()

        run_pipeline(cfg)
        before = sorted(p.name for p in out.iterdir())
        lines = run_pipeline(cfg, dry_run=True).summary().splitlines()
        assert lines[:5] == head
        assert lines[5:8] == [
            "  with the leaves on disk now, which this run writes again:",
            "  reduce: pandas, 1.4e-05 GB over 1 workers, 1.4e-05 GB a worker",
            "  2 blocks of up to 1 targets and 48 rows",
        ]
        assert "with 1 of a group's rows kept, each with its year." in lines[9]
        assert sorted(p.name for p in out.iterdir()) == before

        raw["policies"] = {"on_zero_weight": "skip"}
        other = MonteCarloPipelineConfig.model_validate(raw)
        lines = run_pipeline(other, dry_run=True).summary().splitlines()
        assert lines[:5] == head and len(lines) == 6
        assert lines[5].startswith(
            "  the leaves on disk now do not say how the blocks will be sized: "
            "the leaves were aggregated with policies"
        )

    def test_the_module_entry_point_runs_both_passes(
        self, weights_dir, tmp_path, capsys
    ):
        from cil_regionalization.pipelines.montecarlo import main

        tree, out = tmp_path / "tree", tmp_path / "out"
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
quantile_method = "linear"
summation = "fast"
time_reduction = "pooled"
windows = [[2050, 2051]]
include_n_members = true

[output]
dir = "{out}"
write_leaves = true

[reduce.pandas]
memory_gb = 1
"""
        )
        assert main([str(cfg_path), "--dry-run"]) == 0
        assert "the blocks are sized once the leaves are written" in (
            capsys.readouterr().out
        )
        assert main([str(cfg_path)]) == 0
        printed = capsys.readouterr().out
        assert "reduce: pandas, 1 GB over 1 workers" in printed
        assert printed.count("is in a leaves manifest") == 1
        assert "6 leaves aggregated; statistics rows=2" in printed
        assert f"wrote statistics: {out / 'statistics.parquet'}" in printed
        stats = pd.read_parquet(out / "statistics.parquet")
        # six leaves, and both years of each
        assert stats["n_members"].tolist() == [6, 6]
        assert stats["n_sample"].tolist() == [12, 12]


@pytest.mark.usefixtures("one_target_a_group")
class TestPerYear:
    """``stats.time_reduction = "per_year"`` keeps the year: the
    statistics are over the members within each year of
    ``stats.years``, from the leaf files, under every stage."""

    _MODEL_WEIGHTS = TestPooledYears._MODEL_WEIGHTS

    @staticmethod
    def _raw(
        weights_dir,
        tmp_path,
        *,
        years=(2050, 2081),
        memory_gb=1.0,
        n_workers=1,
        **kwargs,
    ):
        tree, out = tmp_path / "tree", tmp_path / "out"
        if not tree.exists():
            TestLabelPlacement._build_tree(tree)
        raw = TestReduce._raw(weights_dir, tree, out, **kwargs)
        del raw["stats"]["windows"]
        raw["stats"].update(time_reduction="per_year", years=list(years))
        raw["run"]["n_workers"] = n_workers
        raw["reduce"] = {"pandas": {"memory_gb": memory_gb}}
        return raw, out

    def _by_hand(self, cfg: MonteCarloPipelineConfig, out: Path) -> pd.DataFrame:
        """The statistics from the leaf files in one call, with the year
        as one more identity column."""
        from cil_regionalization.leaves import ORDINAL_COLUMN
        from cil_regionalization.pipelines.montecarlo import _ordered
        from cil_regionalization.stats import pooled_statistics

        first, last = cfg.stats.years
        pieces = []
        for kumquat, zeppelin in itertools.product(*_LEVELS.values()):
            leaf = pd.read_parquet(
                out / "leaves" / kumquat / zeppelin / "damages.aggregated.parquet"
            ).drop(columns=[ORDINAL_COLUMN])
            pieces.append(
                leaf[leaf["year"].between(first, last)].assign(
                    kumquat=kumquat, zeppelin=zeppelin
                )
            )
        kept = pd.concat(pieces, ignore_index=True)[
            ["target_id", "band", "kumquat", "year", "zeppelin", "value"]
        ]
        weighted = cfg.stats.model_weights is not None
        if weighted:
            kept["model_weight"] = kept["zeppelin"].map(self._MODEL_WEIGHTS)
        stats = pooled_statistics(
            kept,
            sample_dims=["zeppelin"],
            value_col="value",
            quantiles=cfg.stats.quantiles,
            weight_col="model_weight" if weighted else None,
            include_n_members=True,
            quantile_method=cfg.stats.quantile_method,
            summation=cfg.stats.summation,
        )
        return _ordered(stats, cfg).assign(weight_share=1.0)

    @_METHODS_AND_WEIGHTS
    def test_stage_all_writes_the_leaves_and_keeps_every_year(
        self, weights_dir, tmp_path, methods, weighted
    ):
        raw, out = self._raw(
            weights_dir,
            tmp_path,
            weighted=weighted,
            methods=methods,
            include_n_members=True,
        )
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)
        assert list(result.stats.columns) == [
            "target_id",
            "band",
            "kumquat",
            "year",
            "statistic",
            "value",
            "n_members",
            "weight_share",
        ]
        assert result.stats["year"].dtype == np.int64
        assert sorted(result.stats["year"].unique()) == [2050, 2051, 2080, 2081]
        assert _leaves_manifest(out)["complete"] is True
        _same_frames(result.stats, self._by_hand(cfg, out))
        _same_frames(pd.read_parquet(out / "statistics.parquet"), result.stats)
        # three models, and z2 weighs nothing
        assert set(result.stats["n_members"]) == {2 if weighted else 3}

    @_METHODS_AND_WEIGHTS
    def test_it_gives_what_windows_of_one_year_give(
        self, weights_dir, tmp_path, methods, weighted
    ):
        """The mean of one value is that value, so the window means
        path with a window a year computes the same statistics another
        way. Only the year's column differs."""
        raw, _ = self._raw(weights_dir, tmp_path, weighted=weighted, methods=methods)
        years = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
        windows = json.loads(json.dumps(raw))
        del windows["stats"]["years"]
        windows["stats"].update(
            time_reduction="window_mean",
            windows=[[year, year] for year in (2050, 2051, 2080, 2081)],
        )
        windows["output"].update(dir=str(tmp_path / "windows"), write_leaves=False)
        means = run_pipeline(MonteCarloPipelineConfig.model_validate(windows)).stats
        means = means.assign(year=means["window"].astype(np.int64)).drop(
            columns="window"
        )[list(years.columns)]
        _same_frames(years, means)

    def test_the_years_outside_the_range_are_left_out(self, weights_dir, tmp_path):
        raw, _ = self._raw(weights_dir, tmp_path, years=(2051, 2080))
        stats = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
        assert sorted(stats["year"].unique()) == [2051, 2080]

    def test_a_year_a_leaf_lacks_counts_one_member_less(self, weights_dir, tmp_path):
        """No year is refused for it: a time series shows the smaller
        sample in its count."""
        raw, _ = self._raw(weights_dir, tmp_path, include_n_members=True)
        path = tmp_path / "tree" / "k0" / "z1" / "damages.parquet"
        frame = pd.read_parquet(path)
        frame[frame["year"] != 2051].to_parquet(path, index=False)
        stats = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
        k0 = stats[stats["kumquat"] == "k0"].groupby("year")["n_members"].max()
        assert k0.to_dict() == {2050: 3, 2051: 2, 2080: 3, 2081: 3}

    def test_the_reduce_stage_gives_what_stage_all_gave(self, weights_dir, tmp_path):
        """In one block, in a block a row group, and with two workers."""
        raw, out = self._raw(
            weights_dir, tmp_path, weighted=True, include_n_members=True
        )
        whole = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        for stem, memory_gb, n_workers, blocks in (
            ("one", 1.0, None, 1),
            ("two", _ONE_POOLED_GROUP_GB, None, 2),
            ("pool", 2 * _ONE_POOLED_GROUP_GB, 2, 2),
        ):
            got = TestReduce._reduce(
                raw, stem=stem, memory_gb=memory_gb, n_workers=n_workers
            )
            _same_frames(got.stats, whole.stats)
            manifest = json.loads((out / f"{stem}.manifest.json").read_text())
            assert manifest["reduce"]["blocks"] == blocks

    def test_the_manifest_says_the_years_were_kept(self, weights_dir, tmp_path):
        raw, out = self._raw(weights_dir, tmp_path)
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        manifest = json.loads((out / "statistics.manifest.json").read_text())
        assert manifest["stage"] == "all"
        assert manifest["time_reduction"] == "per_year"
        assert manifest["years"] == [2050, 2081]
        assert manifest["windows"] == []
        assert manifest["reduce"]["kept_share"] == 1.0

    def test_configs_that_cannot_keep_the_years_are_refused_when_read(
        self, weights_dir, tmp_path
    ):
        raw, _ = self._raw(weights_dir, tmp_path)

        def refused(message, **stats):
            changed = json.loads(json.dumps(raw))
            for key, value in stats.items():
                if value is None:
                    del changed["stats"][key]
                else:
                    changed["stats"][key] = value
            with pytest.raises(ValueError, match=message):
                MonteCarloPipelineConfig.model_validate(changed)

        refused("'per_year' needs stats.years", years=None)
        refused("stats.years starts in 2081, after it ends in 2050", years=[2081, 2050])
        refused(
            "stats.windows and stats.time_reduction = 'per_year' are both set",
            windows=[[2050, 2051]],
        )
        refused(
            "decides what happens to stats.time_col 'year' itself",
            sample_dims=["zeppelin", "year"],
        )
        refused(
            "stats.years is for stats.time_reduction = 'per_year', and this "
            "config has 'pooled'",
            time_reduction="pooled",
            windows=[[2050, 2051]],
        )
        refused("stats.time_reduction", time_reduction=None)
        refused("stats.time_reduction", time_reduction="yearly")
        gwl = json.loads(json.dumps(raw))
        gwl["stats"]["warming_levels"] = TestWarmingLevels._raw(
            weights_dir, tmp_path / "gwl"
        )[0]["stats"]["warming_levels"]
        with pytest.raises(
            ValueError,
            match="\\[stats.warming_levels\\] and stats.time_reduction = 'per_year'",
        ):
            MonteCarloPipelineConfig.model_validate(gwl)
        passthrough = json.loads(json.dumps(raw))
        del passthrough["weights"]
        with pytest.raises(
            ValueError, match="stats.time_reduction = 'per_year' needs \\[weights\\]"
        ):
            MonteCarloPipelineConfig.model_validate(passthrough)

    @pytest.mark.parametrize(
        "value, replacement", [(True, "pooled"), (False, "window_mean")]
    )
    def test_a_config_with_the_former_key_is_told_what_to_write(
        self, weights_dir, tmp_path, value, replacement
    ):
        raw = TestReduce._raw(weights_dir, tmp_path / "tree", tmp_path / "out")
        raw["stats"]["pool_years"] = value
        with pytest.raises(
            ValueError,
            match="stats.pool_years is replaced by stats.time_reduction. Use "
            "'pooled' where it was true and 'window_mean' where it was false",
        ):
            MonteCarloPipelineConfig.model_validate(raw)

    def test_what_stage_all_needs_is_asked_before_any_leaf_is_read(
        self, weights_dir, tmp_path, monkeypatch
    ):
        from cil_regionalization.pipelines import montecarlo

        def read_leaf(*args):
            raise AssertionError("a leaf was read")

        monkeypatch.setattr(montecarlo, "_read_leaf", read_leaf)
        for change, message in (
            (
                {"reduce": {"pandas": {"memory_gb": None}}},
                r"Set \[reduce.pandas\] memory_gb",
            ),
            (
                {"output": {"write_leaves": False}},
                "'per_year' takes the statistics from the leaf files, so stage "
                "'all' needs output.write_leaves = true",
            ),
            (
                {"output": {"write_reduced": True}},
                "stats.time_reduction = 'per_year' cannot write the reduced file",
            ),
        ):
            raw, out = self._raw(weights_dir, tmp_path)
            for section, values in change.items():
                raw[section] = {**raw.get(section, {}), **values}
            for dry_run in (True, False):
                with pytest.raises(ValueError, match=message):
                    run_pipeline(
                        MonteCarloPipelineConfig.model_validate(raw), dry_run=dry_run
                    )
            assert not out.exists()

    def test_the_dry_run_says_when_the_blocks_are_sized(self, weights_dir, tmp_path):
        raw, out = self._raw(weights_dir, tmp_path)
        lines = (
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw), dry_run=True)
            .summary()
            .splitlines()
        )
        assert lines[-2:] == [
            "  reduce: stats.time_reduction = 'per_year' takes the statistics "
            "from the leaf files, by blocks, after every leaf is written",
            "  the blocks are sized once the leaves are written",
        ]
        assert not out.exists()


_GWL_LEVELS = {
    "batch": ["b0", "b1"],
    "rcp": ["r45", "r85"],
    "gcm": ["m0", "m1", "surrogate_m0_06"],
}
# The temperature of each member over 2040 to 2043 and over 2080 to
# 2083, before the half a degree of the offset. With levels 1.5, 2, 3
# and 4 and a half-width of a quarter:
#   2040s  1.5 has three members, 2 has two, m1/r85 has no level
#   2080s  2 has one, 3 has three, m1/r45 and m1/r85 have no level
# and nobody is ever at 4.
_GWL_TEMPERATURES = {
    ("m0", "r45"): (1.0, 1.5),
    ("m1", "r45"): (1.0, 2.0),
    ("surrogate_m0_06", "r45"): (1.5, 2.5),
    ("m0", "r85"): (1.5, 2.5),
    ("m1", "r85"): (9.0, 9.0),
    ("surrogate_m0_06", "r85"): (1.0, 2.5),
}
_GWL_YEARS = [2040, 2041, 2042, 2043, 2080, 2081, 2082, 2083]


def _write_gmst(path: Path, temperatures=None, extra=()) -> Path:
    rows = ["gcm,rcp,year,temp"]
    members = dict(_GWL_TEMPERATURES if temperatures is None else temperatures)
    members.update({member: (0.0, 0.0) for member in extra})
    for (gcm, rcp), (mid, late) in members.items():
        for year in range(1986, 2006):
            rows.append(f"{gcm},{rcp},{year},0.0")
        for year in _GWL_YEARS:
            rows.append(f"{gcm},{rcp},{year},{mid if year < 2050 else late}")
    path.write_text("\n".join(rows) + "\n")
    return path


def _build_gwl_tree(root: Path) -> None:
    """One leaf a batch, scenario and model, with the eight years. The
    values are exact in binary, and so is every sum and mean of them."""
    for i, combo in enumerate(itertools.product(*_GWL_LEVELS.values())):
        leaf_dir = root.joinpath(*combo)
        leaf_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "unit_id": unit,
                "year": year,
                "value": base + 2.0 * i + 0.5 * (_GWL_YEARS.index(year) % 4) ** 2,
            }
            for unit, base in (("u1", 8.0), ("u2", 4.0))
            for year in _GWL_YEARS
        ]
        pd.DataFrame(rows).to_parquet(leaf_dir / "damages.parquet", index=False)


@pytest.mark.usefixtures("one_target_a_group")
class TestWarmingLevels:
    """``[stats.warming_levels]``: the statistics of the members at a
    warming level in a period."""

    @staticmethod
    def _raw(weights_dir, tmp_path, **stats) -> tuple[dict, Path]:
        tree, out = tmp_path / "tree", tmp_path / "out"
        if not tree.exists():
            _build_gwl_tree(tree)
        gmst = tmp_path / "gmst.csv"
        if not gmst.exists():
            _write_gmst(gmst)
        raw = _pipeline_cfg(weights_dir, tree, out, levels=_GWL_LEVELS).model_dump()
        del raw["stats"]["windows"]
        raw["stats"].update(
            sample_dims=["batch", "rcp", "gcm"],
            quantiles=[0.5, 0.95],
            warming_levels={
                "gmst": str(gmst),
                "member": ["gcm", "rcp"],
                "reference_period": [1986, 2005],
                "offset": 0.5,
                "levels": [1.5, 2.0, 3.0, 4.0],
                "half_width": 0.25,
                "smoothing": "unknown",
                # the fixture's surrogate is binned like any member here
                "surrogates": "include",
                "periods": [
                    {"gmst_span": [2040, 2043], "impact_span": [2040, 2042]},
                    {"gmst_span": [2080, 2083], "impact_span": [2080, 2083]},
                ],
            },
            **stats,
        )
        raw["output"]["write_leaves"] = True
        raw["reduce"] = {"pandas": {"memory_gb": 1.0}}
        return raw, out

    @pytest.mark.parametrize("name", ["warming_level", "n_bin_members"])
    def test_a_leaf_column_named_like_one_it_writes_is_refused(
        self, weights_dir, tmp_path, name
    ):
        raw, _ = self._raw(weights_dir, tmp_path)
        for path in (tmp_path / "tree").rglob("damages.parquet"):
            pd.read_parquet(path).assign(**{name: 1}).to_parquet(path, index=False)
        with pytest.raises(
            ValueError, match=f"the leaves have a column named '{name}'"
        ):
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

    @staticmethod
    def _by_hand(cfg: MonteCarloPipelineConfig, out: Path) -> pd.DataFrame:
        """From the leaf files and the table of comments above
        `_GWL_TEMPERATURES`, in one call of `pooled_statistics`."""
        import warnings

        from cil_regionalization.leaves import ORDINAL_COLUMN
        from cil_regionalization.pipelines.montecarlo import _ordered
        from cil_regionalization.stats import (
            UnweightedModelWeightsWarning,
            pooled_statistics,
        )

        level_of = {
            member: tuple(
                next(
                    (
                        level
                        for level in cfg.stats.warming_levels.levels
                        if abs(value + 0.5 - level) <= 0.25
                    ),
                    None,
                )
                for value in temperatures
            )
            for member, temperatures in _GWL_TEMPERATURES.items()
        }
        spans = [p.impact_span for p in cfg.stats.warming_levels.periods]
        pieces = []
        for batch, rcp, gcm in itertools.product(*_GWL_LEVELS.values()):
            leaf = pd.read_parquet(
                out / "leaves" / batch / rcp / gcm / "damages.aggregated.parquet"
            ).drop(columns=[ORDINAL_COLUMN])
            for (lo, hi), level in zip(spans, level_of[(gcm, rcp)]):
                if level is None:
                    continue
                inside = leaf[leaf["year"].between(lo, hi)]
                if not cfg.stats.time_reduction == "pooled":
                    inside = (
                        inside.groupby("target_id", sort=False)["value"]
                        .mean()
                        .reset_index()
                    )
                pieces.append(
                    inside.assign(
                        warming_level=level,
                        window=f"{lo}-{hi}",
                        batch=batch,
                        rcp=rcp,
                        gcm=gcm,
                    )
                )
        members = pd.concat(pieces, ignore_index=True)
        years = ["year"] if cfg.stats.time_reduction == "pooled" else []
        members = members[
            ["target_id", "warming_level", "window", "batch", "rcp", "gcm"]
            + years
            + ["value"]
        ]
        with warnings.catch_warnings():
            # the members at a level count equally by design
            warnings.simplefilter("ignore", UnweightedModelWeightsWarning)
            stats = pooled_statistics(
                members,
                sample_dims=["batch", "rcp", "gcm"] + years,
                value_col="value",
                quantiles=cfg.stats.quantiles,
                include_n_members=True,
                quantile_method=cfg.stats.quantile_method,
                summation=cfg.stats.summation,
            )
        keys = ["target_id", "warming_level", "window"]
        counted = members.assign(
            draw=members["batch"] + members["rcp"] + members["gcm"],
            model=members["gcm"] + "/" + members["rcp"],
        ).groupby(keys)
        index = pd.MultiIndex.from_frame(stats[keys])
        if cfg.stats.time_reduction == "pooled":
            stats = stats.rename(columns={"n_members": "n_sample"})
            stats.insert(
                list(stats.columns).index("n_sample"),
                "n_members",
                counted["draw"].nunique().reindex(index).to_numpy(),
            )
        stats["n_bin_members"] = counted["model"].nunique().reindex(index).to_numpy()
        return _ordered(stats, cfg).assign(weight_share=1.0)

    @pytest.mark.parametrize(
        "quantile_method", ["lower_step", "inverted_cdf", "linear"]
    )
    @pytest.mark.parametrize("pooled", [False, True], ids=["means", "years"])
    def test_the_statistics_are_over_the_members_at_each_level(
        self, weights_dir, tmp_path, quantile_method, pooled
    ):
        """Collapsing the window first and pooling its years, each
        under the three quantile definitions."""
        raw, out = self._raw(
            weights_dir,
            tmp_path,
            quantile_method=quantile_method,
            time_reduction="pooled" if pooled else "window_mean",
        )
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        result = run_pipeline(cfg)
        assert list(result.stats.columns) == (
            ["target_id", "warming_level", "window", "statistic", "value", "n_members"]
            + (["n_sample"] if pooled else [])
            + ["n_bin_members", "weight_share"]
        )
        assert result.stats["warming_level"].dtype == "float64"
        _same_frames(result.stats, self._by_hand(cfg, out))
        _same_frames(pd.read_parquet(out / "statistics.parquet"), result.stats)

        counts = (
            result.stats[result.stats["target_id"] == "T1"]
            .drop_duplicates(["warming_level", "window"])
            .set_index(["window", "warming_level"])
        )
        # two batches of each member, and three years of the first period
        assert counts["n_bin_members"].to_dict() == {
            ("2040-2042", 1.5): 3,
            ("2040-2042", 2.0): 2,
            ("2080-2083", 2.0): 1,
            ("2080-2083", 3.0): 3,
        }
        assert (counts["n_members"] == 2 * counts["n_bin_members"]).all()
        if pooled:
            years = counts.index.get_level_values("window").map(
                {"2040-2042": 3, "2080-2083": 4}
            )
            assert (counts["n_sample"] == counts["n_members"] * years).all()

    def test_the_three_quantile_definitions_differ_over_two_models(
        self, weights_dir, tmp_path
    ):
        """Four draws of two models are behind level 2 in the 2040s. The
        95th percentile is the third of the four under the tool's step,
        the fourth under the standard inverse, and between them when
        interpolated."""
        got = {}
        for method in ("lower_step", "inverted_cdf", "linear"):
            raw, _ = self._raw(weights_dir, tmp_path, quantile_method=method)
            raw["output"]["dir"] = str(tmp_path / method)
            stats = run_pipeline(MonteCarloPipelineConfig.model_validate(raw)).stats
            row = stats[
                (stats["target_id"] == "T1")
                & (stats["warming_level"] == 2.0)
                & (stats["window"] == "2040-2042")
                & (stats["statistic"] == "q95")
            ]
            assert row["n_members"].item() == 4 and row["n_bin_members"].item() == 2
            got[method] = row["value"].item()
        assert got["lower_step"] < got["linear"] < got["inverted_cdf"]

    def test_an_excluded_surrogate_is_pooled_at_no_level(
        self, weights_dir, tmp_path, recwarn
    ):
        """Failed with the setting unknown. Its leaves are aggregated and
        go under no level, and the manifest names it."""
        raw, out = self._raw(weights_dir, tmp_path)
        raw["stats"]["warming_levels"]["surrogates"] = "exclude"
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        mid = result.stats[result.stats["window"] == "2040-2042"]
        # without the surrogate 1.5 holds m0/r45 and m1/r45, and 2 holds
        # m0/r85, where with it they hold three and two
        assert mid.groupby("warming_level")["n_bin_members"].max().to_dict() == {
            1.5: 2,
            2.0: 1,
        }
        record = json.loads((out / "statistics.manifest.json").read_text())[
            "warming_levels"
        ]
        assert record["surrogates"] == "exclude"
        assert record["excluded_surrogates"] == [
            {"gcm": "surrogate_m0_06", "rcp": "r45"},
            {"gcm": "surrogate_m0_06", "rcp": "r85"},
        ]

    def test_a_member_of_the_table_no_leaf_belongs_to_is_not_assessed(
        self, weights_dir, tmp_path, recwarn
    ):
        """It has no temperature in any year. Failed with it refused by
        name, although no leaf is of it."""
        raw, out = self._raw(weights_dir, tmp_path)
        gmst = tmp_path / "gmst.csv"
        lines = gmst.read_text().splitlines()
        empty = [f"m9,r45,{year}," for year in [*range(1986, 2006), *_GWL_YEARS]]
        gmst.write_text("\n".join(lines + empty) + "\n")
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        record = json.loads((out / "statistics.manifest.json").read_text())[
            "warming_levels"
        ]
        assert record["unused_members"] == [{"gcm": "m9", "rcp": "r45"}]

    def test_a_declared_member_short_of_a_year_is_refused_by_name(
        self, weights_dir, tmp_path
    ):
        """Only the members a run uses are assessed, and those still have
        to have every year. It guards what was there, so it cannot go red
        by reverting. It went red with the check of the years removed."""
        raw, _ = self._raw(weights_dir, tmp_path)
        gmst = tmp_path / "gmst.csv"
        lines = [
            line for line in gmst.read_text().splitlines() if line != "m0,r45,1990,0.0"
        ]
        gmst.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=r"\{'gcm': 'm0', 'rcp': 'r45'\} has no"):
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

    def test_the_manifest_says_what_decided_the_levels(
        self, weights_dir, tmp_path, recwarn
    ):
        raw, out = self._raw(weights_dir, tmp_path)
        _write_gmst(tmp_path / "gmst.csv", extra=[("m9", "r45")])
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        manifest = json.loads((out / "statistics.manifest.json").read_text())
        assert manifest["stage"] == "all"
        assert manifest["windows"] == ["2040-2042", "2080-2083"]
        assert manifest["n_members_forced"] == [] and manifest["notes"] == []
        assert manifest["model_weights"] is None
        record = manifest["warming_levels"]
        assert record["path"] == str(tmp_path / "gmst.csv")
        assert (
            record["sha256"]
            == hashlib.sha256((tmp_path / "gmst.csv").read_bytes()).hexdigest()
        )
        assert record["member_cols"] == ["gcm", "rcp"]
        assert record["reference_period"] == [1986, 2005]
        assert record["offset"] == 0.5
        assert record["half_width"] == 0.25
        assert record["smoothing"] == "unknown"
        assert record["levels"] == [1.5, 2.0, 3.0, 4.0]
        assert record["periods"] == [
            {
                "window": "2040-2042",
                "gmst_span": [2040, 2043],
                "impact_span": [2040, 2042],
            },
            {
                "window": "2080-2083",
                "gmst_span": [2080, 2083],
                "impact_span": [2080, 2083],
            },
        ]
        # a level nobody is at is recorded and has no row in the file
        assert record["n_members"] == {
            "2040-2042": {"1.5": 3, "2": 2, "3": 0, "4": 0},
            "2080-2083": {"1.5": 0, "2": 1, "3": 3, "4": 0},
        }
        assert record["in_no_bin"] == [{"gcm": "m1", "rcp": "r85"}]
        assert record["unassigned"]["2080-2083"] == [
            {"gcm": "m1", "rcp": "r45"},
            {"gcm": "m1", "rcp": "r85"},
        ]
        assert record["unused_members"] == [{"gcm": "m9", "rcp": "r45"}]
        assert len(record["members"]) == 6
        assert record["members"][0] == {
            "member": {"gcm": "m0", "rcp": "r45"},
            "gmst_mean": {"2040-2042": 1.5, "2080-2083": 2.0},
            "warming_level": {"2040-2042": 1.5, "2080-2083": 2.0},
        }
        stats = pd.read_parquet(out / "statistics.parquet")
        assert sorted(stats["warming_level"].unique()) == [1.5, 2.0, 3.0]
        # the members at a level count equally by design, so the warning
        # for a level named gcm without weights is not for this run
        assert not [
            w for w in recwarn if "weights every climate model" in str(w.message)
        ]

    def test_other_periods_and_the_other_reduction_come_from_the_same_leaves(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """Four impact years where the first run took three, pooled
        where it collapsed, without reading the tree again."""
        from cil_regionalization.pipelines import montecarlo

        raw, out = self._raw(weights_dir, tmp_path)
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

        def read_leaf(*args):
            raise AssertionError("a leaf of the tree was read")

        monkeypatch.setattr(montecarlo, "_read_leaf", read_leaf)
        again = json.loads(json.dumps(raw))
        again["stats"]["time_reduction"] = "pooled"
        again["stats"]["warming_levels"]["periods"] = [
            {"gmst_span": [2040, 2043], "impact_span": [2040, 2043]},
        ]
        again["output"].update(stem="pooled", leaves_stem="statistics")
        cfg = MonteCarloPipelineConfig.model_validate(again)
        got = run_pipeline(cfg, stage="reduce")
        _same_frames(got.stats, self._by_hand(cfg, out))
        assert set(got.stats["window"]) == {"2040-2043"}
        manifest = json.loads((out / "pooled.manifest.json").read_text())
        assert manifest["stage"] == "reduce" and manifest["time_reduction"] == "pooled"
        assert manifest["warming_levels"]["periods"][0]["impact_span"] == [2040, 2043]

    @pytest.mark.parametrize(
        "left_out",
        [
            "reference_period",
            "offset",
            "levels",
            "half_width",
            "smoothing",
            "periods",
            "gmst",
            "member",
        ],
    )
    def test_a_config_that_leaves_out_what_decides_a_level_is_refused(
        self, weights_dir, tmp_path, left_out
    ):
        raw, _ = self._raw(weights_dir, tmp_path)
        del raw["stats"]["warming_levels"][left_out]
        with pytest.raises(ValueError, match=f"warming_levels.{left_out}"):
            MonteCarloPipelineConfig.model_validate(raw)

    def test_configs_that_mix_warming_levels_with_what_they_replace_are_refused(
        self, weights_dir, tmp_path
    ):
        raw, _ = self._raw(weights_dir, tmp_path)

        def refused(message, **changes):
            changed = json.loads(json.dumps(raw))
            for section, values in changes.items():
                if values is None:
                    del changed[section]
                else:
                    changed[section].update(values)
            with pytest.raises(ValueError, match=message):
                MonteCarloPipelineConfig.model_validate(changed)

        refused(
            "stats.windows and \\[stats.warming_levels\\] are both set",
            stats={"windows": [[2040, 2042]]},
        )
        table = tmp_path / "weights.tsv"
        table.write_text("model\tweight\nm0\t1\n")
        refused(
            "stats.model_weights and \\[stats.warming_levels\\] are both set",
            stats={
                "model_weights": str(table),
                "model_weight_level": "gcm",
                "quantile_method": "lower_step",
            },
        )
        refused(
            "member \\['rcp'\\] must be among stats.sample_dims",
            stats={"sample_dims": ["batch", "gcm"]},
        )
        refused("\\[stats.warming_levels\\] needs \\[weights\\]", weights=None)
        strangers = json.loads(json.dumps(raw))
        strangers["stats"]["warming_levels"]["member"] = ["gcm", "ssp"]
        strangers["stats"]["sample_dims"] = ["batch", "rcp", "gcm", "ssp"]
        with pytest.raises(ValueError, match="member \\['ssp'\\] must be tree levels"):
            MonteCarloPipelineConfig.model_validate(strangers)
        neither = json.loads(json.dumps(raw))
        del neither["stats"]["warming_levels"]
        with pytest.raises(ValueError, match="stats.windows must list at least one"):
            MonteCarloPipelineConfig.model_validate(neither)

    def test_what_a_run_needs_is_asked_before_any_leaf_is_read(
        self, weights_dir, tmp_path, monkeypatch
    ):
        from cil_regionalization.pipelines import montecarlo

        def read_leaf(*args):
            raise AssertionError("a leaf was read")

        monkeypatch.setattr(montecarlo, "_read_leaf", read_leaf)
        raw, out = self._raw(weights_dir, tmp_path)

        def refused(message, **changes):
            changed = json.loads(json.dumps(raw))
            for section, values in changes.items():
                changed[section] = {**changed.get(section, {}), **values}
            for dry_run in (True, False):
                with pytest.raises(ValueError, match=message):
                    run_pipeline(
                        MonteCarloPipelineConfig.model_validate(changed),
                        dry_run=dry_run,
                    )
            assert not out.exists()

        refused(
            "\\[stats.warming_levels\\] takes the statistics from the leaf "
            "files, so stage 'all' needs output.write_leaves = true",
            output={"write_leaves": False},
        )
        refused(
            "\\[stats.warming_levels\\] cannot write the reduced file",
            output={"write_reduced": True},
        )
        refused(
            "Set \\[reduce.pandas\\] memory_gb", reduce={"pandas": {"memory_gb": None}}
        )
        # a surrogate is not the model it was made from
        only_m0 = {
            k: v for k, v in _GWL_TEMPERATURES.items() if "surrogate" not in k[0]
        }
        _write_gmst(tmp_path / "gmst.csv", temperatures=only_m0)
        refused(
            "2 of the 6 members the tree declares have no row in the temperature "
            "table .*first: \\{'gcm': 'surrogate_m0_06', 'rcp': 'r45'\\}.*a "
            "surrogate needs rows of its own"
        )
        _write_gmst(
            tmp_path / "gmst.csv",
            temperatures={member: (9.0, 9.0) for member in _GWL_TEMPERATURES},
        )
        refused("none of the 6 members the tree declares is at any of the levels")
        overlapping = json.loads(json.dumps(raw))
        overlapping["stats"]["warming_levels"]["half_width"] = 0.3
        with pytest.raises(ValueError, match="levels 1.5 and 2 overlap"):
            run_pipeline(MonteCarloPipelineConfig.model_validate(overlapping))

    def test_the_temperature_table_is_read_once(
        self, weights_dir, tmp_path, monkeypatch
    ):
        """Before the first pass. A table read again for the second
        could have changed in between."""
        from cil_regionalization.pipelines import montecarlo

        real, reads = montecarlo.read_gmst, []

        def read_gmst(path, **kwargs):
            reads.append(path)
            return real(path, **kwargs)

        monkeypatch.setattr(montecarlo, "read_gmst", read_gmst)
        raw, _ = self._raw(weights_dir, tmp_path)
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
        assert len(reads) == 1

    def test_aggregate_does_not_read_the_temperature_table(self, weights_dir, tmp_path):
        raw, out = self._raw(weights_dir, tmp_path)
        (tmp_path / "gmst.csv").unlink()
        run_pipeline(MonteCarloPipelineConfig.model_validate(raw), stage="aggregate")
        assert _leaves_manifest(out)["complete"] is True
        with pytest.raises(FileNotFoundError):
            run_pipeline(MonteCarloPipelineConfig.model_validate(raw), stage="reduce")

    @pytest.mark.parametrize("year", [2040, 2041, 2042])
    def test_a_leaf_without_a_year_of_an_impact_span_is_refused_by_name(
        self, weights_dir, tmp_path, monkeypatch, year
    ):
        """b1/r45/m0 lacks one year of 2040 to 2042. Its member is at
        1.5 in the 2040s, so its mean there would be over two years of
        three. Refused from the leaves manifest, before any leaf file
        is opened."""
        from cil_regionalization import leaves

        raw, out = self._raw(weights_dir, tmp_path)
        path = tmp_path / "tree" / "b1" / "r45" / "m0" / "damages.parquet"
        frame = pd.read_parquet(path)
        frame[frame["year"] != year].to_parquet(path, index=False)
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        run_pipeline(cfg, stage="aggregate")

        def read_groups(*args, **kwargs):
            raise AssertionError("a leaf file was opened")

        monkeypatch.setattr(leaves, "read_groups", read_groups)
        with pytest.raises(
            ValueError,
            match=rf"leaf b1/r45/m0 has no row for 1 of the 3 years 2040 to 2042 "
            rf"\(first: \[{year}\]\). A mean over fewer years than the span is "
            rf"refused",
        ):
            run_pipeline(cfg, stage="reduce")
        assert not (out / "statistics.parquet").exists()

    def test_a_year_missing_where_the_member_has_no_level_is_let_through(
        self, weights_dir, tmp_path
    ):
        """m1/r85 is at no level in any period, so none of its years is
        averaged."""
        raw, out = self._raw(weights_dir, tmp_path)
        path = tmp_path / "tree" / "b1" / "r85" / "m1" / "damages.parquet"
        frame = pd.read_parquet(path)
        frame[frame["year"] != 2041].to_parquet(path, index=False)
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        stats = run_pipeline(cfg).stats
        assert len(stats) == 2 * 4 * 3

    def test_a_value_missing_in_one_year_of_a_span_is_refused_by_name(
        self, weights_dir, tmp_path
    ):
        """The year has its rows and one of them has no value. Found
        when the leaf is read, and the leaves stay."""
        import pyarrow.parquet as pq

        raw, out = self._raw(weights_dir, tmp_path)
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        run_pipeline(cfg, stage="aggregate")
        entry = next(
            e
            for e in _leaves_manifest(out)["leaves"]
            if e["leaf"] == ["b0", "r45", "m0"]
        )
        path = out / "leaves" / entry["file"]
        table = pq.read_table(path)
        frame = table.to_pandas()
        hole = (frame["target_id"] == "T2") & (frame["year"] == 2081)
        assert hole.sum() == 1
        values = frame["value"].to_numpy().copy()
        values[hole.to_numpy()] = float("nan")
        changed = table.set_column(
            table.schema.get_field_index("value"), "value", [values]
        )
        with pq.ParquetWriter(path, table.schema, compression="snappy") as writer:
            for group in range(2):
                rows = changed.filter(
                    [o == group for o in changed.column("target_ordinal").to_pylist()]
                )
                writer.write_table(rows, row_group_size=max(len(rows), 1))
        # the size in the manifest is that of the file as it was written
        document = _leaves_manifest(out)
        for e in document["leaves"]:
            if e["leaf"] == ["b0", "r45", "m0"]:
                e["size"] = path.stat().st_size
        (out / "leaves" / "statistics.leaves.manifest.json").write_text(
            json.dumps(document)
        )
        with pytest.raises(
            ValueError,
            match=r"leaf b0/r45/m0 has a value for 3 of the 4 years 2080 to 2083 "
            r"at \{'target_id': 'T2'\}",
        ):
            run_pipeline(cfg, stage="reduce")

    def test_the_dry_run_reads_the_table_and_touches_nothing(
        self, weights_dir, tmp_path
    ):
        raw, out = self._raw(weights_dir, tmp_path)
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        lines = run_pipeline(cfg, dry_run=True).summary().splitlines()
        assert lines[-2:] == [
            "  reduce: [stats.warming_levels] takes the statistics from the "
            "leaf files, by blocks, after every leaf is written",
            "  the blocks are sized once the leaves are written",
        ]
        assert not out.exists()
        run_pipeline(cfg)
        lines = run_pipeline(cfg, dry_run=True).summary().splitlines()
        assert "  with the leaves on disk now, which this run writes again:" in lines
        assert any(line.startswith("  reduce: pandas, 1 GB") for line in lines)


class TestOrderedIndex:
    def test_a_frame_whose_index_repeats_keeps_its_rows(self, weights_dir, tmp_path):
        """Two blocks put together without a new index. Failed with the
        rows picked by the labels of the frame as it came, which gave
        every row twice."""
        from cil_regionalization.pipelines.montecarlo import _ordered

        tree = tmp_path / "tree"
        _build_tree(tree, _LEVELS)
        cfg = _pipeline_cfg(weights_dir, tree, tmp_path / "out", levels=_LEVELS)
        stats = run_pipeline(cfg).stats
        halves = pd.concat([stats.iloc[6:], stats.iloc[:6]])
        halves.index = list(range(6)) + list(range(6))
        pd.testing.assert_frame_equal(_ordered(halves, cfg), stats, check_exact=True)
