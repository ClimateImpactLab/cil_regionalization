"""Manifest building and writing: stable hashing, version capture, JSON sidecar."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from segment_weights.config import Config
from segment_weights.manifest import (
    Manifest,
    build_manifest,
    collect_package_versions,
    hash_config,
    hash_file,
)


def _cfg() -> Config:
    return Config.model_validate(
        {
            "project": {"name": "test"},
            "regions": {"path": "/tmp/r.shp", "id_fields": ["hierid"]},
            "grid": {"mode": "generate", "resolution": 1.0},
            "weights": [
                {"name": "pop", "raster": "/tmp/pop.tif"},
                {"name": "area"},
            ],
            "backend": {"kind": "local"},
            "output": {"dir": "/tmp/out"},
        }
    )


class TestHashing:
    def test_config_hash_is_stable(self):
        cfg = _cfg()
        h1 = hash_config(cfg)
        h2 = hash_config(cfg)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_config_hash_changes_with_content(self):
        cfg = _cfg()
        cfg2 = Config.model_validate(
            {
                **cfg.model_dump(),
                "grid": {"mode": "generate", "resolution": 0.25},
            }
        )
        assert hash_config(cfg) != hash_config(cfg2)

    def test_file_hash_stable(self, tmp_path: Path):
        p = tmp_path / "x.txt"
        p.write_bytes(b"hello world")
        assert hash_file(p) == hash_file(p)
        assert len(hash_file(p)) == 64


class TestVersions:
    def test_known_packages_have_versions(self):
        versions = collect_package_versions(("segment_weights", "pandas", "shapely"))
        assert versions["segment_weights"] == "0.1.0"
        assert versions["pandas"] != "unavailable"
        assert versions["shapely"] != "unavailable"

    def test_unknown_package_marked_unavailable(self):
        versions = collect_package_versions(("does_not_exist_xyz",))
        assert versions["does_not_exist_xyz"] == "unavailable"


class TestBuildManifest:
    def test_required_fields_present(self):
        cfg = _cfg()
        m = build_manifest(cfg)
        assert m.config_hash == hash_config(cfg)
        assert m.backend == "local"
        assert m.coverage == "exact_fraction"
        assert m.lon_convention == "[-180,180)"
        assert m.grid_mode == "generate"
        assert m.grid_resolution == 1.0
        assert "segment_weights" in m.package_versions
        assert m.python_version  # non-empty

    def test_fallback_counts_extensible(self):
        m = build_manifest(_cfg())
        m.fallback_counts["pop"] = {"native": 100, "area_fallback": 5}
        assert m.fallback_counts["pop"]["area_fallback"] == 5


class TestWriteManifest:
    def test_write_produces_valid_json(self, tmp_path: Path):
        m = build_manifest(_cfg())
        m.fallback_counts["pop"] = {"native": 10}
        m.row_counts["total"] = 100
        m.timing_seconds["total"] = 1.234
        p = m.write(tmp_path / "manifest.json")
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert loaded["config_hash"] == m.config_hash
        assert loaded["backend"] == "local"
        assert loaded["fallback_counts"]["pop"]["native"] == 10
        assert loaded["row_counts"]["total"] == 100
        assert loaded["timing_seconds"]["total"] == 1.234

    def test_write_creates_parent_dir(self, tmp_path: Path):
        m = build_manifest(_cfg())
        p = m.write(tmp_path / "nested" / "deeper" / "manifest.json")
        assert p.exists()
