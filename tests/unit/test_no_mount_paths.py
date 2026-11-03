"""No tracked file names the Mac mount of the cluster volume.

On Midway3 the volume is /project/cil; a Mac that mounts it sees it
under another root. Three examples once carried a helper that swapped
one root for the other when a path was missing, which turns a mount
that dropped out into a read from somewhere else, without a word. A
developer on the Mac who needs a different root passes it as an
argument instead.

Binary files count too: a stored manifest that recorded a Mac path
leaks the same way. The needle is built from two pieces so this file
does not match itself.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import _git_repo_missing

ROOT = Path(__file__).resolve().parents[2]
NEEDLE = "/" + "Volumes"


def _grep(root: Path, *args: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "grep", "-F", "-e", NEEDLE, *args],
        capture_output=True,
    )
    # git grep exits 1 when nothing matches and above 1 on an error
    if out.returncode > 1:
        raise RuntimeError(out.stderr.decode())
    return out.stdout.decode().splitlines()


def files_naming_the_mount(root: Path) -> list[str]:
    """Tracked files that contain the needle, binary ones included."""
    return _grep(root, "-l")


def check(root: Path) -> list[str]:
    files = files_naming_the_mount(root)
    if not files:
        return []
    lines = _grep(root, "-n", "-I")
    binary = sorted(set(files) - {line.split(":", 1)[0] for line in lines})
    shown = lines + [f"{f}: binary file" for f in binary]
    return [
        f"{len(files)} tracked file(s) name {NEEDLE}, the Mac mount of the "
        f"cluster volume. Use the /project/cil path, and take a different "
        f"root as an argument where one is needed:\n"
        + "\n".join(f"  {s}" for s in shown)
    ]


needs_git = pytest.mark.skipif(
    _git_repo_missing(), reason="git or the git checkout is not available"
)


@needs_git
def test_no_tracked_file_names_the_mount():
    problems = check(ROOT)
    assert not problems, "\n\n".join(problems)


def _repo(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)


def _add(root: Path, rel: str, data: bytes) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    subprocess.run(["git", "-C", str(root), "add", rel], check=True)


@needs_git
def test_clean_repo_passes(tmp_path):
    _repo(tmp_path)
    _add(tmp_path, "a.py", b'SOURCE = "/project/cil/gcp/climate"\n')
    assert check(tmp_path) == []


@needs_git
def test_text_and_binary_matches_are_named(tmp_path):
    _repo(tmp_path)
    _add(tmp_path, "a.py", f'alt = "{NEEDLE}/cil"\n'.encode())
    _add(tmp_path, "data/x.parquet", b"\x00\x01" + NEEDLE.encode() + b"/cil\x00")
    (problem,) = check(tmp_path)
    assert "2 tracked file(s)" in problem
    assert "a.py:1:" in problem
    assert "data/x.parquet: binary file" in problem


@needs_git
def test_untracked_files_are_not_checked(tmp_path):
    _repo(tmp_path)
    _add(tmp_path, "a.py", b"x = 1\n")
    (tmp_path / "notes.txt").write_text(f"{NEEDLE}/cil\n")
    assert check(tmp_path) == []
