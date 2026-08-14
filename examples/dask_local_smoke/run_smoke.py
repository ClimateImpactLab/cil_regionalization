"""Standalone smoke for the local backend's Dask LocalCluster path.

Runs the synthetic 3-region fixture twice: once with ``backend.local.dask =
"off"`` and once with ``"local"``, then asserts every (region, weight) cell
value matches. Designed to finish in well under a minute on a laptop with
no special hardware; exists so the LocalCluster path can be exercised
without booking SLURM time.

    python examples/dask_local_smoke/run_smoke.py

Exit code 0 on parity, 1 on mismatch. Prints a short summary either way.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from segment_weights.backends.local import LocalBackend  # noqa: E402
from segment_weights.config import Config  # noqa: E402
from segment_weights.grid import GridSpec  # noqa: E402
from segment_weights.regions import RegionSet  # noqa: E402
from segment_weights.weights import from_config_list  # noqa: E402


_VALUE_COLS = [
    "region_id", "cell_ix", "cell_iy", "cell_lon", "cell_lat",
    "popwt", "pop_raw", "pop_method",
    "areawt", "area_raw", "area_method",
]


def _cfg(dask_mode: str, output_dir: Path) -> Config:
    regions = REPO_ROOT / "tests" / "data" / "synthetic" / "regions.parquet"
    raster = REPO_ROOT / "tests" / "data" / "synthetic" / "raster.tif"
    return Config.model_validate(
        {
            "project": {"name": f"dask_local_smoke_{dask_mode}"},
            "regions": {"path": str(regions), "id_fields": ["region_id"]},
            "grid": {
                "mode": "generate",
                "resolution": 1.0,
                "offset": "center",
                "lon_convention": "[-180,180)",
            },
            "weights": [
                {"name": "pop", "raster": str(raster), "fallback": "area"},
                {"name": "area"},
            ],
            "backend": {
                "kind": "local",
                "coverage": "exact_fraction",
                "local": {
                    "dask": dask_mode,
                    "n_workers": 2,
                    "threads_per_worker": 1,
                },
            },
            "output": {"dir": str(output_dir)},
        }
    )


def _run(cfg: Config) -> pd.DataFrame:
    regions = RegionSet.from_config(cfg.regions)
    grid = GridSpec.from_config(cfg.grid)
    specs = from_config_list(cfg.weights)
    result = LocalBackend().compute(regions, grid, specs, cfg)
    return (
        result.frame[_VALUE_COLS]
        .sort_values(["region_id", "cell_ix", "cell_iy"])
        .reset_index(drop=True)
    )


def main() -> int:
    out_dir = REPO_ROOT / "data" / "out" / "dask_local_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] running with dask='off' ...")
    t0 = time.perf_counter()
    serial = _run(_cfg("off", out_dir / "serial"))
    t_serial = time.perf_counter() - t0

    print("[2/2] running with dask='local' (LocalCluster) ...")
    t0 = time.perf_counter()
    dask = _run(_cfg("local", out_dir / "dask"))
    t_dask = time.perf_counter() - t0

    try:
        pd.testing.assert_frame_equal(serial, dask, check_exact=False, rtol=1e-12)
    except AssertionError as exc:
        print("PARITY FAIL: serial and dask outputs differ:")
        print(exc)
        return 1

    print()
    print(f"OK: serial {len(serial)} rows, dask {len(dask)} rows, identical.")
    print(f"    serial wall-clock: {t_serial:.2f}s")
    print(f"    dask   wall-clock: {t_dask:.2f}s")
    print("    (3 regions -> dask is necessarily slower; this exercises the path.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
