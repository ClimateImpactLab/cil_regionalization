"""Fetching published weight artifacts: verify, cache, and the failure paths.

A local HTTP server stands in for Zenodo, serving the record metadata
shape the fetch layer consumes (files with md5 checksums, sizes, and
download links). The artifact served is a real parquet and manifest pair
with a recorded output checksum, so the content verification path is the
same one a Zenodo download will take. No network access is required.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from cil_regionalization.fetch import (
    ChecksumMismatchError,
    FetchError,
    IncompleteDownloadError,
    UnknownArtifactError,
    clear_cache,
    fetch_weights,
    list_cached,
    load_registry,
)

RECORD_ID = "7654321"


def _artifact_bytes(tmp_path: Path) -> tuple[bytes, bytes]:
    """A real weights parquet plus a manifest whose outputs checksum it."""
    frame = pd.DataFrame(
        {
            "target_id": ["T1", "T1", "T2"],
            "hierid": ["u1", "u2", "u2"],
            "areawt": [1.0, 0.5, 0.5],
        }
    )
    pq = tmp_path / "weights.parquet"
    frame.to_parquet(pq, index=False)
    parquet_bytes = pq.read_bytes()
    manifest = {
        "id_fields": ["target_id"],
        "source_key_columns": ["hierid"],
        "weight_names": ["area"],
        "normalization": "per_source",
        "source_mode": "polygons",
        "regions_version": "targets-v1",
        "source_version": "units-v1",
        "outputs": {
            "weights.parquet": hashlib.sha256(parquet_bytes).hexdigest()
        },
    }
    return parquet_bytes, json.dumps(manifest).encode()


class _Server:
    """Serves /api/records/<id> and file downloads from an in-memory dict."""

    def __init__(self, files: dict[str, bytes], record_id: str = RECORD_ID):
        self.files = files
        self.record_id = record_id
        # Sizes and checksums as declared in record metadata; tests
        # override these to simulate truncation and corruption.
        self.declared_sizes = {k: len(v) for k, v in files.items()}
        self.declared_md5 = {
            k: hashlib.md5(v).hexdigest() for k, v in files.items()
        }
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == f"/api/records/{server.record_id}":
                    body = json.dumps(server._record_json()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                for key in server.files:
                    if self.path == f"/files/{key}":
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(server.files[key])
                        return
                self.send_response(404)
                self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def _record_json(self) -> dict:
        return {
            "id": self.record_id,
            "files": [
                {
                    "key": key,
                    "size": self.declared_sizes[key],
                    "checksum": f"md5:{self.declared_md5[key]}",
                    "links": {"self": f"{self.base_url}/files/{key}"},
                }
                for key in self.files
            ],
        }

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def served(tmp_path):
    parquet, manifest = _artifact_bytes(tmp_path)
    server = _Server(
        {"weights.parquet": parquet, "weights.manifest.json": manifest}
    )
    yield server
    server.stop()


class TestFetchRoundTrip:
    def test_fetch_verify_load(self, served, tmp_path):
        cache = tmp_path / "cache"
        artifact = fetch_weights(
            "test-artifact",
            record=RECORD_ID,
            base_url=served.base_url,
            cache_dir=cache,
        )
        assert artifact.normalization == "per_source"
        assert artifact.source_version == "units-v1"
        assert len(artifact.frame) == 3
        cached = list_cached(cache)
        assert [i["name"] for i in cached] == ["test-artifact"]
        assert cached[0]["record"] == RECORD_ID

    def test_cached_copy_needs_no_network(self, served, tmp_path):
        cache = tmp_path / "cache"
        fetch_weights(
            "test-artifact",
            record=RECORD_ID,
            base_url=served.base_url,
            cache_dir=cache,
        )
        served.stop()
        artifact = fetch_weights(
            "test-artifact",
            record=RECORD_ID,
            base_url=served.base_url,
            cache_dir=cache,
        )
        assert len(artifact.frame) == 3

    def test_doi_resolves_to_record_id(self, served, tmp_path):
        artifact = fetch_weights(
            "test-artifact",
            record=f"10.5281/zenodo.{RECORD_ID}",
            base_url=served.base_url,
            cache_dir=tmp_path / "cache",
        )
        assert len(artifact.frame) == 3

    def test_corrupted_cache_refuses_to_load(self, served, tmp_path):
        cache = tmp_path / "cache"
        fetch_weights(
            "test-artifact",
            record=RECORD_ID,
            base_url=served.base_url,
            cache_dir=cache,
        )
        cached = Path(list_cached(cache)[0]["path"]) / "weights.parquet"
        cached.write_bytes(b"corrupt")
        with pytest.raises(ChecksumMismatchError, match="cached artifact"):
            fetch_weights(
                "test-artifact",
                record=RECORD_ID,
                base_url=served.base_url,
                cache_dir=cache,
            )


class TestFailurePaths:
    def test_unknown_name_lists_known(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CIL_REGIONALIZATION_REGISTRY", raising=False)
        with pytest.raises(UnknownArtifactError, match="registry knows"):
            fetch_weights("no-such-artifact", cache_dir=tmp_path / "cache")

    def test_truncated_download_raises_and_leaves_no_cache(
        self, served, tmp_path
    ):
        served.declared_sizes["weights.parquet"] += 100
        cache = tmp_path / "cache"
        with pytest.raises(IncompleteDownloadError, match="incomplete"):
            fetch_weights(
                "test-artifact",
                record=RECORD_ID,
                base_url=served.base_url,
                cache_dir=cache,
            )
        assert list_cached(cache) == []

    def test_transport_corruption_raises(self, served, tmp_path):
        served.declared_md5["weights.parquet"] = "0" * 32
        cache = tmp_path / "cache"
        with pytest.raises(ChecksumMismatchError, match="md5"):
            fetch_weights(
                "test-artifact",
                record=RECORD_ID,
                base_url=served.base_url,
                cache_dir=cache,
            )
        assert list_cached(cache) == []

    def test_content_corruption_raises(self, served, tmp_path):
        """Server bytes are internally consistent (md5 matches what is
        served) but do not match the manifest's recorded output checksum:
        the case Zenodo's own checksums cannot catch."""
        bad = served.files["weights.parquet"] + b"tail"
        served.files["weights.parquet"] = bad
        served.declared_sizes["weights.parquet"] = len(bad)
        served.declared_md5["weights.parquet"] = hashlib.md5(bad).hexdigest()
        cache = tmp_path / "cache"
        with pytest.raises(ChecksumMismatchError, match="manifest"):
            fetch_weights(
                "test-artifact",
                record=RECORD_ID,
                base_url=served.base_url,
                cache_dir=cache,
            )
        assert list_cached(cache) == []

    def test_manifest_without_outputs_fails_before_parquet(
        self, served, tmp_path
    ):
        doc = json.loads(served.files["weights.manifest.json"])
        del doc["outputs"]
        body = json.dumps(doc).encode()
        served.files["weights.manifest.json"] = body
        served.declared_sizes["weights.manifest.json"] = len(body)
        served.declared_md5["weights.manifest.json"] = hashlib.md5(
            body
        ).hexdigest()
        with pytest.raises(FetchError, match="no output checksum"):
            fetch_weights(
                "test-artifact",
                record=RECORD_ID,
                base_url=served.base_url,
                cache_dir=tmp_path / "cache",
            )

    def test_missing_file_in_record_named(self, served, tmp_path):
        del served.files["weights.manifest.json"]
        with pytest.raises(FetchError, match="has no file"):
            fetch_weights(
                "test-artifact",
                record=RECORD_ID,
                base_url=served.base_url,
                cache_dir=tmp_path / "cache",
            )


class TestRegistry:
    def test_packaged_registry_ships_published_records(self, monkeypatch):
        monkeypatch.delenv("CIL_REGIONALIZATION_REGISTRY", raising=False)
        entries = load_registry()
        for name in (
            "gadm20-adm1-per-source",
            "gadm20-adm1-per-destination",
            "gadm20-adm2-per-source",
            "gadm20-adm2-per-destination",
        ):
            assert entries[name].record == "21934155"
        for name in (
            "gadm41-adm1-per-source",
            "gadm41-adm1-per-destination",
            "gadm41-adm2-per-source",
            "gadm41-adm2-per-destination",
        ):
            assert entries[name].record == "21935431"

    def test_local_registry_resolves_name(self, served, tmp_path):
        reg = tmp_path / "registry.toml"
        reg.write_text(
            "schema = 1\n"
            "[artifacts.my-weights]\n"
            f'record = "{RECORD_ID}"\n'
            f'base_url = "{served.base_url}"\n'
        )
        artifact = fetch_weights(
            "my-weights",
            registry_path=reg,
            cache_dir=tmp_path / "cache",
        )
        assert len(artifact.frame) == 3

    def test_env_registry_is_read(self, served, tmp_path, monkeypatch):
        reg = tmp_path / "registry.toml"
        reg.write_text(
            "[artifacts.env-weights]\n"
            f'record = "{RECORD_ID}"\n'
            f'base_url = "{served.base_url}"\n'
        )
        monkeypatch.setenv("CIL_REGIONALIZATION_REGISTRY", str(reg))
        entries = load_registry()
        assert "env-weights" in entries

    def test_entry_without_record_rejected(self, tmp_path):
        reg = tmp_path / "registry.toml"
        reg.write_text('[artifacts.broken]\nparquet = "weights.parquet"\n')
        with pytest.raises(FetchError, match="no 'record' field"):
            load_registry(reg)

    def test_missing_registry_file_named(self, tmp_path):
        with pytest.raises(FetchError, match="does not exist"):
            load_registry(tmp_path / "absent.toml")


class TestCache:
    def test_clear_by_name_and_all(self, served, tmp_path):
        cache = tmp_path / "cache"
        for name in ("first", "second"):
            fetch_weights(
                name,
                record=RECORD_ID,
                base_url=served.base_url,
                cache_dir=cache,
            )
        assert len(list_cached(cache)) == 2
        assert clear_cache(cache, name="first") == 1
        assert [i["name"] for i in list_cached(cache)] == ["second"]
        assert clear_cache(cache) == 1
        assert list_cached(cache) == []
