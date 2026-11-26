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

        # No leaf is kept unless the config asks for it.
        assert not (tmp_path / "out" / "leaves").exists()
        manifest = json.loads((tmp_path / "out" / "statistics.manifest.json").read_text())
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
            "target_id", "statistic", "value", "window", "n_members",
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

        weighted = run_pipeline(self._cfg(
            weights_dir, tree, tmp_path / "weighted", quantiles=[0.5, 0.95],
            quantile_method="lower_step", summation="exact",
            **self._table(tmp_path),
        ))
        manifest = json.loads(Path(weighted.paths["manifest"]).read_text())
        assert manifest["quantile_method"] == "lower_step"
        assert manifest["summation"] == "exact"

        for name in ("linear", "lower_step", "inverted_cdf"):
            named = run_pipeline(self._cfg(
                weights_dir, tree, tmp_path / name, quantile_method=name,
                quantiles=[0.5, 0.95],
            ))
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
        step = run_pipeline(self._cfg(
            weights_dir, tree, tmp_path / "step", quantile_method="lower_step",
            **quantiles,
        ))
        ones = run_pipeline(self._cfg(
            weights_dir, tree, tmp_path / "ones", quantile_method="lower_step",
            **quantiles, **self._table(tmp_path, weights=(1, 1, 1)),
        ))
        pd.testing.assert_frame_equal(step.stats, ones.stats, check_exact=True)
        assert (
            step.stats["value"].to_numpy().tobytes()
            == ones.stats["value"].to_numpy().tobytes()
        )
        linear = run_pipeline(self._cfg(
            weights_dir, tree, tmp_path / "linear", quantile_method="linear",
            **quantiles,
        ))
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
                weights_dir, tmp_path / "t", tmp_path / "o",
                quantile_method="linear", **self._table(tmp_path),
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
        result = run_pipeline(self._cfg(
            weights_dir, tree, out,
            quantile_method="inverted_cdf", summation="exact",
        ))
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
            linear.stats, on=["target_id", "window", "statistic"],
            suffixes=("", "_linear"),
        )
        quantiles = both[both["statistic"] != "mean"]
        assert (quantiles["value"] != quantiles["value_linear"]).all()

    def test_serial_and_parallel_runs_agree_to_the_bit(
        self, weights_dir, tmp_path
    ):
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
        pd.testing.assert_frame_equal(
            serial.stats, parallel.stats, check_exact=True
        )
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
            sample_dims=["kumquat"], windows=self._WINDOWS,
            quantiles=self._QUANTILES, **stats,
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

        monkeypatch.setattr(
            montecarlo, "mp", types.SimpleNamespace(Pool=LastFirstPool)
        )
        tree = tmp_path / "tree"
        _build_tree(tree, self._ORDER_LEVELS)
        raw = self._cfg(
            weights_dir, tree, tmp_path / "out", include_n_members=True, **options
        ).model_dump()
        raw["run"] = {"n_workers": 2}
        raw["output"]["write_reduced"] = True
        result = run_pipeline(MonteCarloPipelineConfig.model_validate(raw))

        windows, targets = ["2080-2081", "2050-2051"], ["T1", "T2"]
        expected = list(itertools.product(
            ["mean", "q95", "q05"], ["z2", "z0", "z1"], windows, targets
        ))
        columns = ["statistic", "zeppelin", "window", "target_id"]
        for frame in (result.stats, pd.read_parquet(result.paths["statistics"])):
            assert list(frame[columns].itertuples(index=False, name=None)) == expected
            assert list(frame.index) == list(range(len(expected)))

        reduced = pd.read_parquet(tmp_path / "out" / "statistics.reduced.parquet")
        expected = list(itertools.product(
            ["k1", "k0"], ["z2", "z0", "z1"], windows, targets
        ))
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
            ).frame
            reduced.append(
                window_means(
                    applied, time_col="year", windows=self._WINDOWS,
                    value_col="value",
                    summation="fast",
                )
            )
        combined = pd.concat(reduced, ignore_index=True)
        assert list(combined.columns) == [
            "target_id", "band", "kumquat", "zeppelin", "window", "value",
        ]
        statistics = _ordered(
            pooled_statistics(
                combined, sample_dims=["zeppelin"], value_col="value",
                quantiles=[0.05, 0.95],
                quantile_method="linear",
                summation="fast",
            ),
            cfg,
        )
        assert list(statistics.columns) == [
            "target_id", "band", "kumquat", "window", "statistic", "value",
        ]

        pd.testing.assert_frame_equal(result.stats, statistics, check_exact=True)
        pd.testing.assert_frame_equal(
            pd.read_parquet(out / "statistics.parquet"),
            _through_parquet(statistics, tmp_path / "expected.parquet"),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(out / "statistics.reduced.parquet"),
            _through_parquet(
                _ordered(combined, cfg), tmp_path / "expected.parquet"
            ),
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
                "unit_id", "band", "year", "value", "kumquat", "zeppelin",
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
            "unit_id", "band", "kumquat", "zeppelin", "window", "value",
        ]

    @pytest.mark.parametrize(
        "name",
        [
            "window", "statistic", "n_members", "model_weight", "target_id",
            "target_ordinal",
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

    def test_a_level_named_like_a_leaf_column_is_refused(
        self, weights_dir, tmp_path
    ):
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
        with pytest.raises(
            ValueError, match="leaf already has a column named 'year'"
        ):
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
quantile_method = "linear"
summation = "fast"
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
quantile_method = "linear"
summation = "fast"
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
            "/" "Volumes/",  # split so test_no_mount_paths.py does not match it
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
        raw["stats"]["quantile_method"] = "lower_step"
        raw["stats"]["model_weight_level"] = "zeppelin"
        cfg = MonteCarloPipelineConfig.model_validate(raw)
        with pytest.raises(ValueError, match="no row in the model"):
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
        cfg = _pipeline_cfg(
            weights_dir, tree, out, levels=_LEVELS, n_workers=n_workers
        )
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
            weights_dir, tree, tmp_path / "out", p,
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
                "target_ordinal", "target_id", "year", "value",
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

        monkeypatch.setattr(
            montecarlo, "mp", types.SimpleNamespace(Pool=LastFirstPool)
        )
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        run_pipeline(_leaves_cfg(weights_dir, tree, out, n_workers=2))
        manifest = _leaves_manifest(out)

        assert sorted(manifest) == [
            "aggregation", "aggregation_digest", "complete", "format_version",
            "group_col", "leaves", "package_versions", "paths", "profiles",
            "rows_per_group", "schema", "stem", "targets", "targets_per_group",
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
                        ("table", "parquet"), ("manifest", "manifest.json"),
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

    def test_the_statistics_manifest_points_at_the_leaves(
        self, weights_dir, tmp_path
    ):
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
        pd.testing.assert_frame_equal(with_leaves.stats, without.stats, check_exact=True)
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

    def test_a_failed_leaf_leaves_the_manifest_incomplete(
        self, weights_dir, tmp_path
    ):
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
        before = {
            p: p.read_bytes() for p in (out / "leaves").rglob("*") if p.is_file()
        }

        with pytest.raises(
            ValueError, match="6 leaf files .* output stem 'first'"
        ):
            run_pipeline(_leaves_cfg(weights_dir, tree, out, stem="second"))
        after = {
            p: p.read_bytes() for p in (out / "leaves").rglob("*") if p.is_file()
        }
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
                    weights_dir, tree, out,
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
            "k0", "k0.leaves.manifest.json", "k1", "k1.leaves.manifest.json",
            k0["targets"]["file"],
        ]

    def test_the_group_size_reaches_the_workers_of_a_pool(
        self, weights_dir, tmp_path
    ):
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
        with pytest.raises(
            ValueError, match="leaf column 'when' has dtype datetime64"
        ):
            run_pipeline(_leaves_cfg(weights_dir, tree, out))
        assert _leaves_manifest(out)["complete"] is False
        assert not (out / "statistics.parquet").exists()

    def test_leaves_with_different_columns_are_refused(self, weights_dir, tmp_path):
        tree, out = tmp_path / "tree", tmp_path / "out"
        _build_tree(tree, _LEVELS)
        path = tree / "k1" / "z0" / "damages.parquet"
        pd.read_parquet(path).astype({"year": "int32"}).to_parquet(path, index=False)
        with pytest.raises(
            ValueError, match="leaf k1/z0 was written with columns .* same columns"
        ):
            run_pipeline(_leaves_cfg(weights_dir, tree, out))
        assert _leaves_manifest(out)["complete"] is False
        assert not (out / "statistics.parquet").exists()

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
            "aggregated_damages.parquet", "damages.aggregated.parquet",
        ]


@pytest.mark.usefixtures("one_target_a_group")
class TestStages:
    """`aggregate` stops after the leaf files."""

    def test_aggregate_writes_the_leaves_and_no_statistics(
        self, weights_dir, tmp_path
    ):
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
        with pytest.raises(ValueError, match="unknown stage 'reduce'"):
            run_pipeline(
                _leaves_cfg(weights_dir, tree, tmp_path / "out"), stage="reduce"
            )

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
