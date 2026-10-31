"""generate_weights: same files as the CLI, and nothing written on failure."""
import json
import shutil

import pandas as pd
import pytest

import cil_regionalization as cilreg
from cil_regionalization.backends.local import LocalBackend
from cil_regionalization.cli import main
from cil_regionalization.validate import SumToOneReport

from tests.integration.test_local_backend import _cfg, _write_toml


def _failing_report() -> SumToOneReport:
    return SumToOneReport(
        weight_names=("pop",),
        n_regions=1,
        failures=pd.DataFrame(
            {"region_id": ["A"], "weight": ["pop"], "sum": [0.5], "deviation": [0.5]}
        ),
        tolerance=1e-6,
    )


@pytest.fixture
def toml_config(three_region_geoparquet, synthetic_raster, tmp_path):
    cfg = _cfg(three_region_geoparquet, synthetic_raster, tmp_path / "out")
    path = tmp_path / "cfg.toml"
    _write_toml(path, cfg)
    return path


@pytest.fixture
def failing_sum(monkeypatch):
    real = LocalBackend.compute

    def compute(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        result.sum_report = _failing_report()
        return result

    monkeypatch.setattr(LocalBackend, "compute", compute)


class TestSameAsCli:
    def test_files_and_manifest_match(self, toml_config, tmp_path):
        assert main(["run", str(toml_config)]) == 0
        out = tmp_path / "out"
        by_cli = tmp_path / "by_cli"
        shutil.move(out, by_cli)

        result = cilreg.generate_weights(toml_config)

        assert (out / "weights.parquet").read_bytes() == (
            by_cli / "weights.parquet"
        ).read_bytes()
        mine = json.loads((out / "weights.manifest.json").read_text())
        theirs = json.loads((by_cli / "weights.manifest.json").read_text())
        for m in (mine, theirs):
            m.pop("timing_seconds")
        assert mine == theirs
        assert result.manifest.outputs == mine["outputs"]
        assert result.sum_report.ok

    def test_config_object_and_path_agree(self, toml_config):
        from_path = cilreg.generate_weights(toml_config)
        from_cfg = cilreg.generate_weights(cilreg.load_config(toml_config))
        pd.testing.assert_frame_equal(from_path.frame, from_cfg.frame)

    def test_result_applies(self, toml_config):
        result = cilreg.generate_weights(toml_config)
        weights = cilreg.WeightsArtifact.from_result(result)
        assert weights.normalization == "per_destination"


class TestMissingInput:
    def test_names_the_missing_raster(
        self, three_region_geoparquet, tmp_path
    ):
        cfg = _cfg(three_region_geoparquet, tmp_path / "gone.tif", tmp_path / "out")
        with pytest.raises(FileNotFoundError, match="gone.tif"):
            cilreg.generate_weights(cfg)
        assert not (tmp_path / "out").exists()


class TestFailedSumToOne:
    def test_raises_with_report_and_writes_nothing(
        self, toml_config, tmp_path, failing_sum
    ):
        with pytest.raises(cilreg.SumToOneError) as err:
            cilreg.generate_weights(toml_config)
        assert not err.value.report.ok
        assert err.value.report.failures["weight"].tolist() == ["pop"]
        assert "FAILED" in str(err.value)
        assert not (tmp_path / "out").exists()

    def test_cli_exits_one_and_writes_nothing(
        self, toml_config, tmp_path, failing_sum, capsys
    ):
        assert main(["run", str(toml_config)]) == 1
        err = capsys.readouterr().err
        assert "nothing written" in err
        assert "FAILED" in err
        assert not (tmp_path / "out").exists()
