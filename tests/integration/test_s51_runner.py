"""s51 runner: cost-gating tests with mocked BigQuery.

The runner is config-driven; no stdin. These tests verify:

- default (confirm_cost=true, no --yes) exits non-zero before issuing the
  combined query, with a message naming the two ways to proceed,
- --yes proceeds past the confirmation gate,
- confirm_cost=false in the config proceeds without --yes,
- a dry-run estimate that exceeds the ceiling aborts regardless of
  confirm_cost or --yes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_RUNNER_PATH = Path(__file__).resolve().parents[2] / "examples" / "s51"


@pytest.fixture
def runner_module(monkeypatch):
    """Import examples/s51/runner.py as a module."""
    monkeypatch.syspath_prepend(str(_RUNNER_PATH))
    if "runner" in sys.modules:
        del sys.modules["runner"]
    import runner

    yield runner
    if "runner" in sys.modules:
        del sys.modules["runner"]


def _write_test_cfg(
    tmp_path: Path, *, confirm_cost: bool, ceiling: int = 10_000_000_000
) -> Path:
    p = tmp_path / "test.toml"
    p.write_text(
        f"""
[project]
name = "test"

[regions]
table = "compute-impactlab.spatial_aggregation.X"
id_fields = ["hierid"]
keep = {{ hierid = ["ABW"] }}
on_null_geometry = "skip"

[grid]
mode = "generate"
resolution = 1.0
offset = "center"
lon_convention = "[-180,180)"

[[weights]]
name = "pop"
table = "compute-impactlab.gpw.pop"

[[weights]]
name = "area"

[backend]
kind = "bigquery"
coverage = "pixel_centroid"

[backend.bigquery]
staging_uri = "gs://example-staging/segment-weights/"
dry_run_byte_ceiling = {ceiling}
cache_temp_tables = true

[output]
dir = "{tmp_path}/out"
format = "parquet"
confirm_cost = {"true" if confirm_cost else "false"}
"""
    )
    return p


def _patch_backend(runner_module, monkeypatch, *, dry_bytes: int):
    """Replace BigQueryBackend with a stub that returns a fixed estimate
    and a usable result on .compute(). Avoids any real BQ calls."""
    # _extend_with_smallest_hierids hits real BQ in test mode; stub it out.
    monkeypatch.setattr(
        runner_module, "_extend_with_smallest_hierids", lambda cfg: cfg
    )

    fake_estimate = {
        "bytes_estimate": dry_bytes,
        "compute_location": "US",
        "ir_location": "us-west1",
        "temp_table": "compute-impactlab.temp_workspace.segweights_geom_xxx",
        "cache_hit": False,
        "n_regions": 1,
        "null_skipped": [],
        "unknown_skipped": [],
    }

    from segment_weights.backends.base import WeightsResult
    from segment_weights.manifest import build_manifest
    from segment_weights.schema import OutputSchema
    from segment_weights.validate import check_sum_to_one
    import pandas as pd

    def _fake_compute(self, regions, grid, weights, cfg):
        schema = OutputSchema(
            id_fields=("hierid",),
            weight_names=tuple(w.name for w in weights),
        )
        df = pd.DataFrame(
            {
                "hierid": ["ABW"],
                "cell_ix": [109],
                "cell_iy": [102],
                "cell_lon": [-70.5],
                "cell_lat": [12.5],
                "popwt": [1.0],
                "pop_raw": [100.0],
                "pop_method": ["native"],
                "areawt": [1.0],
                "area_raw": [1.0],
                "area_method": ["native"],
            }
        )
        manifest = build_manifest(cfg)
        manifest.row_counts["regions"] = 1
        manifest.row_counts["total"] = 1
        manifest.extra["null_geometry_count"] = 0
        manifest.extra["null_geometry_regions"] = []
        manifest.extra["unknown_id_count"] = 0
        manifest.extra["unknown_id_regions"] = []
        report = check_sum_to_one(df, schema)
        return WeightsResult(df, schema, manifest, report)

    fake_backend_cls = MagicMock()
    fake_backend = MagicMock()
    fake_backend.dry_run.return_value = fake_estimate
    fake_backend.compute.side_effect = lambda r, g, w, c: _fake_compute(
        fake_backend, r, g, w, c
    )
    fake_backend_cls.return_value = fake_backend

    monkeypatch.setattr(runner_module, "BigQueryBackend", fake_backend_cls)
    return fake_backend


def test_default_confirm_cost_aborts(runner_module, monkeypatch, tmp_path, capsys):
    """confirm_cost=true and no --yes: print explainer, exit non-zero,
    never call backend.compute()."""
    cfg_path = _write_test_cfg(tmp_path, confirm_cost=True)
    fake = _patch_backend(runner_module, monkeypatch, dry_bytes=1_000_000_000)
    rc = runner_module.main(["--config", str(cfg_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "confirm_cost is true" in captured.err
    assert "--yes" in captured.err
    assert "confirm_cost = false" in captured.err
    fake.compute.assert_not_called()


def test_yes_flag_proceeds(runner_module, monkeypatch, tmp_path, capsys):
    """confirm_cost=true but --yes passed: backend.compute() is called."""
    cfg_path = _write_test_cfg(tmp_path, confirm_cost=True)
    fake = _patch_backend(runner_module, monkeypatch, dry_bytes=1_000_000_000)
    rc = runner_module.main(["--config", str(cfg_path), "--yes"])
    captured = capsys.readouterr()
    assert "confirm_cost is true" not in captured.err
    fake.compute.assert_called_once()


def test_confirm_cost_false_proceeds(runner_module, monkeypatch, tmp_path, capsys):
    """confirm_cost=false in the config: proceed without --yes."""
    cfg_path = _write_test_cfg(tmp_path, confirm_cost=False)
    fake = _patch_backend(runner_module, monkeypatch, dry_bytes=1_000_000_000)
    rc = runner_module.main(["--config", str(cfg_path)])
    captured = capsys.readouterr()
    assert "confirm_cost is true" not in captured.err
    fake.compute.assert_called_once()


def test_dry_run_over_ceiling_always_aborts(
    runner_module, monkeypatch, tmp_path, capsys
):
    """Even with --yes and confirm_cost=false, an estimate over the
    ceiling aborts. The ceiling is not a 'human pause'; it's a hard
    limit on cost."""
    cfg_path = _write_test_cfg(
        tmp_path, confirm_cost=False, ceiling=1_000_000
    )
    fake = _patch_backend(
        runner_module, monkeypatch, dry_bytes=5_000_000  # 5x ceiling
    )
    rc = runner_module.main(["--config", str(cfg_path), "--yes"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "exceeds the configured ceiling" in captured.err
    fake.compute.assert_not_called()


def test_runner_never_reads_stdin(runner_module, monkeypatch, tmp_path):
    """Sanity: closing stdin must not break the runner in any branch.
    Works under nbconvert, cron, and SLURM."""
    import io

    cfg_path = _write_test_cfg(tmp_path, confirm_cost=True)
    _patch_backend(runner_module, monkeypatch, dry_bytes=100)

    closed_stdin = io.StringIO("")
    closed_stdin.close()
    monkeypatch.setattr("sys.stdin", closed_stdin)
    # No path through the runner reads stdin. Even confirm-needed,
    # the runner should exit cleanly with a message, not block.
    rc = runner_module.main(["--config", str(cfg_path)])
    assert rc == 1
