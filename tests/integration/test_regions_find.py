"""`cilreg regions find <pattern> <config>` CLI subcommand.

Cheap LIKE search against the configured regions source. Helps users
resolve hierids in this IR vintage's mixed naming patterns (bare ISO3,
single-remainder-only suffixes, admin subdivisions) without guessing.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


def _write_local_cfg(tmp_path: Path) -> Path:
    """A local-backend config pointing at a geoparquet with known ids."""
    gdf = gpd.GeoDataFrame(
        {
            "hierid": [
                "ABW",
                "AND.Ra5cc0db7a54d1bb3",
                "BMU.R676c07148ce9acd1",
                "BHR.5",
                "BHR.Rf0a1304585646a1c",
                "USA.5.123",
                "CHN.34.456",
            ],
            "geometry": [box(i, 0, i + 1, 1) for i in range(7)],
        },
        crs="EPSG:4326",
    )
    regions_path = tmp_path / "regions.parquet"
    gdf.to_parquet(regions_path)
    raster_path = tmp_path / "raster.tif"
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
    ) as ds:
        ds.write(np.ones((4, 4), dtype=np.float32), 1)

    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        f"""
[project]
name = "find_test"
[regions]
path = "{regions_path}"
id_fields = ["hierid"]
[grid]
mode = "generate"
resolution = 1.0
offset = "center"
lon_convention = "[-180,180)"
[[weights]]
name = "pop"
raster = "{raster_path}"
[[weights]]
name = "area"
[backend]
kind = "local"
coverage = "exact_fraction"
[output]
dir = "{tmp_path}/out"
"""
    )
    return cfg_path


class TestRegionsFindLocal:
    def test_exact_match(self, tmp_path, capsys):
        from cil_regionalization.cli import main

        cfg = _write_local_cfg(tmp_path)
        rc = main(["regions", "find", "ABW", str(cfg)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "ABW" in out
        assert "1 matches" in out

    def test_wildcard_finds_remainder_suffix(self, tmp_path, capsys):
        """A bare 'AND' wouldn't match; 'AND%' surfaces the remainder form."""
        from cil_regionalization.cli import main

        cfg = _write_local_cfg(tmp_path)
        rc = main(["regions", "find", "AND%", str(cfg)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "AND.Ra5cc0db7a54d1bb3" in out

    def test_wildcard_finds_multiple_bhr(self, tmp_path, capsys):
        """BHR has both BHR.5 and BHR.R...; the LIKE finds both."""
        from cil_regionalization.cli import main

        cfg = _write_local_cfg(tmp_path)
        rc = main(["regions", "find", "BHR%", str(cfg)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "BHR.5" in out
        assert "BHR.Rf0a1304585646a1c" in out

    def test_no_match_returns_zero_and_prints_message(self, tmp_path, capsys):
        from cil_regionalization.cli import main

        cfg = _write_local_cfg(tmp_path)
        rc = main(["regions", "find", "NOT_A_REGION", str(cfg)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no matches" in out

    def test_limit_caps_output(self, tmp_path, capsys):
        from cil_regionalization.cli import main

        cfg = _write_local_cfg(tmp_path)
        rc = main(["regions", "find", "%", str(cfg), "--limit", "3"])
        out = capsys.readouterr().out
        assert rc == 0
        lines = [l for l in out.splitlines() if l and not l.startswith("searching") and not l.startswith("3 matches")]
        # Three id lines plus the summary line
        assert "3 matches (limit reached)" in out


class TestRegionsFindBigQuery:
    """Mocked test for the BQ branch: verifies the SQL shape, the LIKE
    parameter, and the location lookup."""

    def test_issues_parameterized_like_query(self, tmp_path, monkeypatch, capsys):
        from cil_regionalization import cli

        # Stub the bigquery module import inside _regions_find_bq.
        fake_bq = MagicMock()
        fake_bq.Client = MagicMock()
        client = fake_bq.Client.return_value
        ds = MagicMock()
        ds.location = "us-west1"
        client.get_dataset.return_value = ds
        fake_bq.QueryJobConfig = MagicMock()
        fake_bq.ScalarQueryParameter = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame(
            {"hierid": ["BMU.R676c07148ce9acd1"]}
        )

        # Inject the fake bigquery module so `from google.cloud import bigquery` returns ours.
        import sys, types

        gcloud_pkg = sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
        monkeypatch.setattr(gcloud_pkg, "bigquery", fake_bq, raising=False)
        monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)

        cfg_path = tmp_path / "cfg.toml"
        cfg_path.write_text(
            f"""
[project]
name = "find_bq_test"
[regions]
table = "compute-impactlab.spatial_aggregation.test_table"
id_fields = ["hierid"]
[grid]
mode = "generate"
resolution = 1.0
offset = "center"
lon_convention = "[-180,180)"
[[weights]]
name = "pop"
table = "x.y.pop"
[[weights]]
name = "area"
[backend]
kind = "bigquery"
coverage = "pixel_centroid"
[backend.bigquery]
staging_uri = "gs://example-staging/segment-weights/"
[output]
dir = "{tmp_path}/out"
"""
        )
        rc = cli.main(["regions", "find", "BMU%", str(cfg_path)])
        out = capsys.readouterr().out
        assert rc == 0
        # Verify the SQL contains the LIKE clause and reads from the table.
        sql_arg = client.query.call_args.args[0]
        assert "WHERE hierid LIKE @pat" in sql_arg
        assert "test_table" in sql_arg
        assert "BMU.R676c07148ce9acd1" in out
        # Location matches the dataset's location, not the temp/weight ones.
        assert client.query.call_args.kwargs["location"] == "us-west1"
