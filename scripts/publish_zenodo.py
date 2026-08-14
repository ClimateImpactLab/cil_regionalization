"""Publish a record to Zenodo: weight files, or any staged file set.

Maintainer tool, not part of the installed package. Plain standard
library against the Zenodo REST deposition API: create a deposition,
upload the files, set metadata, verify checksums, publish. Which files
and which metadata come from a committed record TOML, so publishing a
second record (GADM 4.1 next to GADM 2.0, or the impact region
shapefile) is a second sidecar file and a second invocation, never a
second script.

A record can hold loose files, a zip, or both. The zip is built by
this script at upload time, never staged by hand: for weight records
it contains one folder per artifact, named by the artifact's registry
name so the folder a person browses is the name a fetch_weights call
uses, plus a README from the sidecar. Loose files are what
fetch_weights downloads; the zip is what a person opening the record
page takes. Both are checksum verified after upload.

Usage
-----
    python scripts/publish_zenodo.py --record scripts/zenodo_gadm20.toml --dry-run
    python scripts/publish_zenodo.py --record scripts/zenodo_gadm20.toml --sandbox --draft
    python scripts/publish_zenodo.py --record scripts/zenodo_gadm20.toml --publish
    python scripts/publish_zenodo.py --record scripts/zenodo_gadm20.toml \
        --new-version-of 1234567 --publish

Exactly one of --dry-run, --draft, --publish is required. --dry-run
prints the plan, builds and checksums the zip, and touches nothing
remote. --draft uploads, sets metadata, verifies, and stops for review
in the browser. --publish does all of that and publishes.
--new-version-of takes the id of the latest published version and
opens a new version under the same concept DOI, replacing the
carried-over files; bump the version field in the record TOML first.

The token comes from the environment or a repo-root .env file (never
committed): ZENODO_TOKEN for zenodo.org, ZENODO_SANDBOX_TOKEN with
--sandbox. Sandbox tokens are separate accounts on sandbox.zenodo.org.
The scopes deposit:write (create, edit, upload) and deposit:actions
(publish, new version) are sufficient.

Record TOML fields: files_dir, title, version, keywords, description,
and either [registry_names] (weight records: name = file stem, which
also defines the expected staged files and the printed registry
entries) or files (a plain list of expected file names). Optional:
loose = false to upload only the zip; a [zip] table with name and
readme; [[related_identifiers]] tables with identifier and relation
(entries containing REPLACE are skipped with a warning so a sidecar
can carry a placeholder until a sibling record exists).

Before any upload the script checks each staged parquet against the
SHA256 its own manifest records, so a stale staging directory fails
here. After each upload it compares Zenodo's stored md5 against the
local bytes, so a truncated transfer fails before publication. On
publish it prints the record id, the DOI, and the registry.toml
entries to add with the id filled in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parents[1]
# Fixed timestamp inside built zips so rebuilding from identical inputs
# yields identical bytes.
_ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def _publish_ssl_context() -> ssl.SSLContext:
    """certifi's CA bundle when available, for hosts whose OpenSSL has
    no configured bundle (some cluster environments)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _load_token(sandbox: bool) -> str:
    import os

    name = "ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_TOKEN"
    token = os.environ.get(name)
    if not token:
        env_file = _ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{name}=") and not line.startswith("#"):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        raise SystemExit(
            f"no {name} in the environment or .env; create a token at the "
            f"{'sandbox.' if sandbox else ''}zenodo.org account settings with "
            f"scopes deposit:write and deposit:actions"
        )
    return token


def _request(
    method: str,
    url: str,
    token: str,
    *,
    payload: dict | None = None,
    data: bytes | None = None,
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = data
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    elif data is not None:
        headers["Content-Type"] = "application/octet-stream"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=_publish_ssl_context()) as resp:
            text = resp.read().decode() or "{}"
            return json.loads(text)
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        raise SystemExit(f"{method} {url} failed: HTTP {e.code}\n{detail}") from e


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_record(record_path: Path) -> dict:
    with record_path.open("rb") as f:
        rec = tomllib.load(f)
    for key in ("files_dir", "title", "version", "keywords", "description"):
        if key not in rec:
            raise SystemExit(f"record TOML is missing {key!r}")
    if ("registry_names" in rec) == ("files" in rec):
        raise SystemExit(
            "record TOML needs exactly one of [registry_names] or files"
        )
    return rec


def _staged_files(rec: dict) -> list[Path]:
    files_dir = (_ROOT / rec["files_dir"]).resolve()
    staged = [
        f for f in sorted(files_dir.glob("*"))
        if f.is_file() and not f.name.startswith(".")
    ]
    if not staged:
        raise SystemExit(f"no files staged under {files_dir}")
    if "registry_names" in rec:
        expected = set()
        for stem in rec["registry_names"].values():
            expected.add(f"{stem}.parquet")
            expected.add(f"{stem}.manifest.json")
    else:
        expected = set(rec["files"])
    got = {f.name for f in staged}
    if got != expected:
        raise SystemExit(
            f"staged files do not match the record definition.\n"
            f"  staged but unexpected: {sorted(got - expected)}\n"
            f"  expected but missing:  {sorted(expected - got)}"
        )
    return staged


def _verify_staging(rec: dict, files: list[Path]) -> None:
    """Each staged weights parquet must match the SHA256 its manifest
    records; plain file records have no manifests and skip this."""
    if "registry_names" not in rec:
        print(f"staging present: {len(files)} files (no manifests to check)")
        return
    by_name = {f.name: f for f in files}
    for f in files:
        if not f.name.endswith(".manifest.json"):
            continue
        manifest = json.loads(f.read_text())
        recorded = (manifest.get("outputs") or {}).get("weights.parquet")
        if recorded is None:
            raise SystemExit(f"{f.name} records no output checksum; refusing")
        parquet = by_name[f.name.replace(".manifest.json", ".parquet")]
        actual = _sha256(parquet)
        if actual != recorded:
            raise SystemExit(
                f"{parquet.name}: sha256 {actual} does not match the "
                f"{recorded} its manifest records; the staging directory "
                f"is stale, restage before uploading"
            )
    print(f"staging verified: {len(files)} files, parquets match manifests")


def _build_zip(rec: dict, files: list[Path], workdir: Path) -> Path | None:
    """Build the browsable zip: one folder per artifact named by its
    registry name (so the folder is the fetch name and loads directly
    with WeightsArtifact.load), or the plain files at the root, plus
    the sidecar's README."""
    zcfg = rec.get("zip")
    if not zcfg:
        return None
    zip_name = zcfg["name"]
    root = zip_name.removesuffix(".zip")
    out = workdir / zip_name

    def _add(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
        info = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, data)

    by_name = {f.name: f for f in files}
    with zipfile.ZipFile(out, "w") as zf:
        _add(zf, f"{root}/README.txt", zcfg["readme"].strip().encode() + b"\n")
        if "registry_names" in rec:
            for name, stem in rec["registry_names"].items():
                _add(zf, f"{root}/{name}/weights.parquet",
                     by_name[f"{stem}.parquet"].read_bytes())
                _add(zf, f"{root}/{name}/weights.manifest.json",
                     by_name[f"{stem}.manifest.json"].read_bytes())
        else:
            for f in files:
                _add(zf, f"{root}/{f.name}", f.read_bytes())
    return out


def _metadata(rec: dict) -> dict:
    paragraphs = [p.strip() for p in rec["description"].split("\n\n") if p.strip()]
    html = "\n".join(f"<p>{p}</p>" for p in paragraphs)
    meta = {
        "upload_type": "dataset",
        "title": rec["title"],
        "creators": [{"name": "Climate Impact Lab"}],
        "license": "cc-by-4.0",
        "version": rec["version"],
        "keywords": list(rec["keywords"]),
        "description": html,
    }
    related = []
    for entry in rec.get("related_identifiers", []):
        if "REPLACE" in entry.get("identifier", ""):
            print(f"warning: skipping placeholder related identifier "
                  f"({entry.get('relation')}); fill it in and republish "
                  f"metadata when the sibling record exists")
            continue
        related.append(
            {"identifier": entry["identifier"], "relation": entry["relation"]}
        )
    if related:
        meta["related_identifiers"] = related
    return {"metadata": meta}


def _print_registry(rec: dict, record_id: str) -> None:
    if "registry_names" not in rec:
        return
    print("\nregistry.toml entries to add:\n")
    for name, stem in rec["registry_names"].items():
        print(f"[artifacts.{name}]")
        print(f'record = "{record_id}"')
        print(f'parquet = "{stem}.parquet"')
        print(f'manifest = "{stem}.manifest.json"')
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", required=True, help="record TOML sidecar")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="print the plan and build the zip; no network")
    mode.add_argument("--draft", action="store_true",
                      help="upload and verify, stop before publishing")
    mode.add_argument("--publish", action="store_true",
                      help="upload, verify, and publish")
    parser.add_argument("--sandbox", action="store_true",
                        help="use sandbox.zenodo.org and ZENODO_SANDBOX_TOKEN")
    parser.add_argument("--new-version-of", metavar="ID", default=None,
                        help="open a new version of this published record id")
    args = parser.parse_args(argv)

    rec = _load_record(Path(args.record))
    files = _staged_files(rec)
    upload_loose = rec.get("loose", True)
    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"

    print(f"record:   {rec['title']} (version {rec['version']})")
    print(f"instance: {base}")
    if upload_loose:
        print(f"loose files ({len(files)}):")
        for f in files:
            print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB  md5 {_md5(f)}")
    _verify_staging(rec, files)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = _build_zip(rec, files, Path(tmp))
        if zip_path is not None:
            print(f"zip built: {zip_path.name}  "
                  f"{zip_path.stat().st_size/1e6:.2f} MB  md5 {_md5(zip_path)}")
        uploads = (files if upload_loose else []) + (
            [zip_path] if zip_path is not None else []
        )
        if not uploads:
            raise SystemExit("nothing to upload: loose = false and no [zip]")

        if args.dry_run:
            print(f"\ndry run: {len(uploads)} files would be uploaded; "
                  f"nothing sent")
            return 0

        token = _load_token(args.sandbox)
        api = f"{base}/api/deposit/depositions"

        if args.new_version_of:
            draft = _request(
                "POST", f"{api}/{args.new_version_of}/actions/newversion", token
            )
            draft_url = draft["links"]["latest_draft"]
            dep = _request("GET", draft_url, token)
            # A new version carries the old files over; replace them.
            for old in _request("GET", f"{draft_url}/files", token):
                _request("DELETE", f"{draft_url}/files/{old['id']}", token)
            print(f"opened new version draft {dep['id']} of {args.new_version_of}")
        else:
            dep = _request("POST", api, token, payload={})
            print(f"created deposition {dep['id']}")

        bucket = dep["links"]["bucket"]
        for f in uploads:
            stored = _request("PUT", f"{bucket}/{f.name}", token,
                              data=f.read_bytes())
            remote = str(stored.get("checksum", "")).removeprefix("md5:")
            local = _md5(f)
            if remote != local:
                raise SystemExit(
                    f"{f.name}: Zenodo stored md5 {remote}, local is {local}; "
                    f"the upload is corrupt. Discard the draft and rerun."
                )
            print(f"uploaded {f.name}  md5 verified")

        _request("PUT", f"{api}/{dep['id']}", token, payload=_metadata(rec))
        print("metadata set")

        if args.draft:
            print(f"\ndraft ready for review: {dep['links'].get('html', '')}")
            print("publish from the browser, or rerun with --publish")
            return 0

        published = _request("POST", f"{api}/{dep['id']}/actions/publish", token)
    record_id = str(published["id"])
    doi = published.get("doi") or published.get("metadata", {}).get("doi", "")
    print(f"\npublished: record {record_id}  doi {doi}")
    print(f"url: {published.get('links', {}).get('record_html', '')}")
    _print_registry(rec, record_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
