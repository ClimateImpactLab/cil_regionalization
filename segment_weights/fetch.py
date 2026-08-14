"""Fetch published weight artifacts from Zenodo, verify, cache, load.

A published artifact is the pair a generation run writes: the weights
parquet and its manifest sidecar. Records live on Zenodo, one record per
target geometry version, with Zenodo's own record versioning carrying
corrections. Which records exist is data, not code: the packaged
``registry.toml`` maps short artifact names to record and file names,
and a caller can extend or replace it with a local registry file (the
``registry_path`` argument or the ``SEGMENT_WEIGHTS_REGISTRY``
environment variable) without any code change. A record identifier can
also be given directly, bypassing the registry entirely.

Integrity has two layers, both required. The transport layer checks
every downloaded file against the md5 that Zenodo's record metadata
declares for it. The content layer checks the weights parquet against
the SHA256 its own manifest recorded at write time (the ``outputs``
field), which ties the downloaded bytes to the generation run rather
than to whatever was uploaded. A failure at either layer deletes the
partial download and raises; nothing unverified ever lands in the cache
under its final name.

The cache lives under ``~/.cache/segment_weights`` (respecting
``XDG_CACHE_HOME``, overridable via the ``cache_dir`` argument or
``SEGMENT_WEIGHTS_CACHE``), one directory per record and artifact,
holding the canonical ``weights.parquet`` and ``weights.manifest.json``
that `WeightsArtifact.load` reads. Cached artifacts are re-verified
against the cached manifest on every use; a corrupt cache entry raises
instead of loading. ``list_cached`` and ``clear_cache`` (or
``segweights cache list|clear``) show and remove cache contents.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

from segment_weights.apply import WeightsArtifact

DEFAULT_BASE_URL = "https://zenodo.org"
_CHUNK = 1 << 20


class FetchError(RuntimeError):
    """Base class for fetch failures."""


class UnknownArtifactError(FetchError):
    """The requested name is in no registry."""


class IncompleteDownloadError(FetchError):
    """The downloaded byte count does not match the record metadata."""


class ChecksumMismatchError(FetchError):
    """A downloaded or cached file fails checksum verification."""


@dataclass(frozen=True)
class RegistryEntry:
    """One published artifact: where it lives and what its files are called."""

    name: str
    record: str
    parquet: str = "weights.parquet"
    manifest: str = "weights.manifest.json"
    base_url: str = DEFAULT_BASE_URL

    @property
    def record_id(self) -> str:
        """Numeric record id, accepting a bare id or a Zenodo DOI."""
        rec = str(self.record)
        if "zenodo." in rec:
            return rec.rsplit("zenodo.", 1)[1]
        return rec


def _packaged_registry_path() -> Path:
    return Path(__file__).parent / "registry.toml"


def load_registry(registry_path: str | Path | None = None) -> dict[str, RegistryEntry]:
    """Load the artifact registry: packaged entries, then overlays.

    Order: the packaged ``registry.toml``, then the file named by the
    ``SEGMENT_WEIGHTS_REGISTRY`` environment variable if set, then
    ``registry_path`` if given. Later entries replace earlier ones of the
    same name, so a local registry can both add artifacts and repoint
    existing names.
    """
    paths: list[Path] = [_packaged_registry_path()]
    env = os.environ.get("SEGMENT_WEIGHTS_REGISTRY")
    if env:
        paths.append(Path(env))
    if registry_path is not None:
        paths.append(Path(registry_path))

    entries: dict[str, RegistryEntry] = {}
    for p in paths:
        if not p.exists():
            if p != _packaged_registry_path():
                raise FetchError(f"registry file does not exist: {p}")
            continue
        with p.open("rb") as f:
            data = tomllib.load(f)
        for name, spec in (data.get("artifacts") or {}).items():
            if "record" not in spec:
                raise FetchError(
                    f"registry entry {name!r} in {p} has no 'record' field"
                )
            entries[name] = RegistryEntry(
                name=name,
                record=str(spec["record"]),
                parquet=spec.get("parquet", "weights.parquet"),
                manifest=spec.get("manifest", "weights.manifest.json"),
                base_url=spec.get("base_url", DEFAULT_BASE_URL),
            )
    return entries


def default_cache_dir() -> Path:
    env = os.environ.get("SEGMENT_WEIGHTS_CACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path("~/.cache").expanduser()
    return base / "segment_weights"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise FetchError(f"GET {url} failed: HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"GET {url} failed: {e.reason}") from e


def _record_files(entry: RegistryEntry) -> dict[str, dict[str, Any]]:
    """The record's file metadata keyed by filename."""
    url = f"{entry.base_url.rstrip('/')}/api/records/{entry.record_id}"
    record = _get_json(url)
    files = {f["key"]: f for f in record.get("files", [])}
    if not files:
        raise FetchError(
            f"record {entry.record_id} at {entry.base_url} lists no files; "
            f"is the record published and open access?"
        )
    return files


def _download(url: str, dest: Path, *, expected_size: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out, _CHUNK)
    except urllib.error.HTTPError as e:
        dest.unlink(missing_ok=True)
        raise FetchError(f"GET {url} failed: HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        dest.unlink(missing_ok=True)
        raise FetchError(f"GET {url} failed: {e.reason}") from e
    if expected_size is not None and dest.stat().st_size != expected_size:
        got = dest.stat().st_size
        dest.unlink(missing_ok=True)
        raise IncompleteDownloadError(
            f"downloaded {got} bytes for {dest.name}, record metadata "
            f"declares {expected_size}; the download is incomplete"
        )


def _verify_transport(path: Path, file_meta: dict[str, Any]) -> None:
    declared = str(file_meta.get("checksum", ""))
    if not declared.startswith("md5:"):
        raise FetchError(
            f"record file {file_meta.get('key')} declares no md5 checksum "
            f"(got {declared!r}); cannot verify the download"
        )
    actual = _md5(path)
    if actual != declared[4:]:
        path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"{path.name}: md5 {actual} does not match the record's "
            f"{declared[4:]}; deleted the download"
        )


def _manifest_sha256(manifest: dict[str, Any], parquet_name: str) -> str:
    outputs = manifest.get("outputs") or {}
    for key in (parquet_name, "weights.parquet"):
        if key in outputs:
            return outputs[key]
    raise FetchError(
        "the artifact's manifest records no output checksum for "
        f"{parquet_name!r} (outputs: {sorted(outputs)}); it was written "
        "before output checksums existed and cannot be fetch-verified"
    )


def _verify_content(parquet: Path, manifest: dict[str, Any], parquet_name: str) -> None:
    expected = _manifest_sha256(manifest, parquet_name)
    actual = _sha256(parquet)
    if actual != expected:
        raise ChecksumMismatchError(
            f"{parquet}: sha256 {actual} does not match the manifest's "
            f"recorded output checksum {expected}"
        )


def _artifact_cache_dir(entry: RegistryEntry, cache_dir: Path) -> Path:
    return cache_dir / f"record-{entry.record_id}" / entry.name


def fetch_weights(
    name: str,
    *,
    record: str | None = None,
    base_url: str | None = None,
    parquet: str | None = None,
    manifest: str | None = None,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> WeightsArtifact:
    """Fetch a published weight artifact and return it ready to apply.

    ``name`` is normally a registry name; with ``record`` given (a Zenodo
    record id or DOI) the registry is bypassed and ``name`` only labels
    the cache entry. The result is a `WeightsArtifact`, directly
    consumable by `apply_weights`. ``refresh`` re-downloads even when a
    verified cached copy exists.
    """
    if record is not None:
        entry = RegistryEntry(
            name=name,
            record=record,
            parquet=parquet or "weights.parquet",
            manifest=manifest or "weights.manifest.json",
            base_url=base_url or DEFAULT_BASE_URL,
        )
    else:
        registry = load_registry(registry_path)
        if name not in registry:
            known = ", ".join(sorted(registry)) or "none"
            raise UnknownArtifactError(
                f"unknown artifact {name!r}; registry knows: {known}. "
                f"Add it to a registry file, or pass record= directly."
            )
        entry = registry[name]
        if base_url is not None:
            entry = RegistryEntry(
                name=entry.name,
                record=entry.record,
                parquet=entry.parquet,
                manifest=entry.manifest,
                base_url=base_url,
            )

    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    target = _artifact_cache_dir(entry, cache)
    parquet_path = target / "weights.parquet"
    manifest_path = target / "weights.manifest.json"

    if not refresh and parquet_path.exists() and manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text())
        expected = _manifest_sha256(cached_manifest, entry.parquet)
        actual = _sha256(parquet_path)
        if actual != expected:
            raise ChecksumMismatchError(
                f"cached artifact at {target} fails verification "
                f"(sha256 {actual}, manifest records {expected}); clear it "
                f"with clear_cache() or 'segweights cache clear' and refetch"
            )
        return WeightsArtifact.load(target)

    files = _record_files(entry)
    for wanted in (entry.parquet, entry.manifest):
        if wanted not in files:
            raise FetchError(
                f"record {entry.record_id} has no file {wanted!r} "
                f"(available: {sorted(files)})"
            )

    manifest_tmp = target / ".partial.manifest.json"
    parquet_tmp = target / ".partial.parquet"
    try:
        meta = files[entry.manifest]
        _download(
            meta["links"]["self"], manifest_tmp, expected_size=meta.get("size")
        )
        _verify_transport(manifest_tmp, meta)
        manifest_doc = json.loads(manifest_tmp.read_text())
        # Fail before the large download if the manifest cannot verify it.
        _manifest_sha256(manifest_doc, entry.parquet)

        meta = files[entry.parquet]
        _download(
            meta["links"]["self"], parquet_tmp, expected_size=meta.get("size")
        )
        _verify_transport(parquet_tmp, meta)
        _verify_content(parquet_tmp, manifest_doc, entry.parquet)

        parquet_tmp.replace(parquet_path)
        manifest_tmp.replace(manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        parquet_tmp.unlink(missing_ok=True)

    return WeightsArtifact.load(target)


def list_cached(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Cached artifacts: name, record, path, and size in bytes."""
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    out: list[dict[str, Any]] = []
    if not cache.exists():
        return out
    for record_dir in sorted(cache.glob("record-*")):
        for artifact_dir in sorted(p for p in record_dir.iterdir() if p.is_dir()):
            parquet = artifact_dir / "weights.parquet"
            if not parquet.exists():
                continue
            out.append(
                {
                    "name": artifact_dir.name,
                    "record": record_dir.name.removeprefix("record-"),
                    "path": str(artifact_dir),
                    "size_bytes": sum(
                        f.stat().st_size for f in artifact_dir.iterdir() if f.is_file()
                    ),
                }
            )
    return out


def clear_cache(
    cache_dir: str | Path | None = None, *, name: str | None = None
) -> int:
    """Remove cached artifacts; all of them, or just ``name``. Returns count."""
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    removed = 0
    for item in list_cached(cache):
        if name is not None and item["name"] != name:
            continue
        shutil.rmtree(item["path"])
        removed += 1
    if not cache.exists():
        return removed
    for record_dir in cache.glob("record-*"):
        if not any(record_dir.iterdir()):
            record_dir.rmdir()
    return removed
