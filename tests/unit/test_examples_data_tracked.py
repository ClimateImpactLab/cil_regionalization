"""Every file under examples/<name>/data/ is tracked by git, or says why not.

A `*.parquet` rule in .gitignore once kept the data of several examples
out of the repository without a word: the notebooks ran for the person
who had the files and failed in a clone. Asking git for the untracked
files, ignored ones included, catches that whichever rule hides them.

macOS and editor droppings (.DS_Store, ._ files, __pycache__,
.ipynb_checkpoints) are skipped. Files that are staged count as
tracked: git only accepts them past the ignore rules.
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import _git_repo_missing

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = "examples/data_not_committed.txt"
_DATA_FILE = re.compile(r"^examples/[^/]+/data/")
_JUNK = re.compile(r"(^|/)(\.DS_Store|\._[^/]*|__pycache__|\.ipynb_checkpoints)(/|$)")
_MAX_LINES = 30


def untracked_data_files(root: Path) -> list[str]:
    """Untracked files under examples/<name>/data/, ignored ones included."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "-z", "--", "examples"],
        capture_output=True, check=True,
    ).stdout.decode()
    return sorted(
        p for p in out.split("\0")
        if p and _DATA_FILE.match(p) and not _JUNK.search(p)
    )


def read_allowlist(root: Path) -> tuple[list[str], list[str]]:
    """The patterns in the allowlist file, and what is wrong with it."""
    path = root / ALLOWLIST
    if not path.exists():
        return [], []
    patterns, problems = [], []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, _, reason = line.partition("#")
        pattern, reason = pattern.strip(), reason.strip()
        if not _DATA_FILE.match(pattern):
            problems.append(
                f"{ALLOWLIST}:{n}: {pattern!r} is not under "
                f"examples/<name>/data/"
            )
        elif not reason:
            problems.append(
                f"{ALLOWLIST}:{n}: {pattern!r} has no reason; write it as "
                f"'{pattern}  # why it is left out'"
            )
        else:
            patterns.append(pattern)
    return patterns, problems


def check(root: Path) -> list[str]:
    """Problems with the data under examples/, empty when there are none."""
    patterns, problems = read_allowlist(root)
    left = [
        p for p in untracked_data_files(root)
        if not any(fnmatch.fnmatchcase(p, pat) for pat in patterns)
    ]
    if left:
        shown = left[:_MAX_LINES]
        more = (f"\n  ... and {len(left) - len(shown)} more"
                if len(left) > len(shown) else "")
        lines = "\n".join(f"  {p}  # why it is left out" for p in shown) + more
        problems.append(
            f"{len(left)} file(s) under examples/<name>/data/ are not tracked "
            f"by git, so a clone will not have them. Commit them with "
            f"`git add`; if .gitignore refuses, fix the negation there. To "
            f"leave one out on purpose, add a line like these to "
            f"{ALLOWLIST}:\n{lines}"
        )
    return problems


needs_git = pytest.mark.skipif(
    _git_repo_missing(), reason="git or the git checkout is not available"
)


@needs_git
def test_example_data_is_tracked_or_listed():
    problems = check(ROOT)
    assert not problems, "\n\n".join(problems)


class _Repo:
    def __init__(self, root: Path):
        self.root = root
        self.git("init", "-q")
        (root / ".gitignore").write_text("*.parquet\n*.csv\n")

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.root), *args], check=True,
                       capture_output=True)


@pytest.fixture
def repo(tmp_path):
    return _Repo(tmp_path)


def _write(root: Path, rel: str, text: str = "x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@needs_git
class TestChecker:
    def test_a_tracked_file_is_fine(self, repo):
        _write(repo.root, "examples/a/data/t.txt")
        repo.git("add", "examples/a/data/t.txt")
        assert check(repo.root) == []

    def test_a_staged_file_that_gitignore_would_hide_is_fine(self, repo):
        _write(repo.root, "examples/a/data/t.parquet")
        repo.git("add", "-f", "examples/a/data/t.parquet")
        assert check(repo.root) == []

    def test_an_ignored_file_is_reported(self, repo):
        _write(repo.root, "examples/a/data/t.parquet")
        (problem,) = check(repo.root)
        assert "examples/a/data/t.parquet" in problem
        assert "  examples/a/data/t.parquet  # why it is left out" in problem
        assert ALLOWLIST in problem

    def test_an_untracked_file_that_is_not_ignored_is_reported(self, repo):
        _write(repo.root, "examples/a/data/t.nc4")
        assert "examples/a/data/t.nc4" in check(repo.root)[0]

    def test_a_nested_file_is_reported(self, repo):
        _write(repo.root, "examples/a/data/weights/x/weights.parquet")
        assert "weights/x/weights.parquet" in check(repo.root)[0]

    def test_a_listed_file_with_a_reason_is_fine(self, repo):
        _write(repo.root, "examples/a/data/t.parquet")
        _write(repo.root, ALLOWLIST, "examples/a/data/t.parquet  # 9 GB\n")
        assert check(repo.root) == []

    def test_a_glob_covers_several_files(self, repo):
        for i in range(3):
            _write(repo.root, f"examples/a/data/big/{i}.nc4")
        _write(repo.root, ALLOWLIST, "# note\n\nexamples/a/data/big/*.nc4  # too large\n")
        assert check(repo.root) == []

    def test_a_listed_file_without_a_reason_fails(self, repo):
        _write(repo.root, "examples/a/data/t.parquet")
        _write(repo.root, ALLOWLIST, "examples/a/data/t.parquet\n")
        problems = check(repo.root)
        assert any("has no reason" in p and "# why it is left out" in p
                   for p in problems)

    def test_a_listed_path_outside_a_data_directory_fails(self, repo):
        _write(repo.root, ALLOWLIST, "examples/a/other/t.parquet  # nope\n")
        assert any("not under examples/<name>/data/" in p for p in check(repo.root))

    def test_listing_one_file_does_not_cover_another(self, repo):
        _write(repo.root, "examples/a/data/t.parquet")
        _write(repo.root, "examples/a/data/u.parquet")
        _write(repo.root, ALLOWLIST, "examples/a/data/t.parquet  # why\n")
        (problem,) = check(repo.root)
        assert "u.parquet" in problem and "t.parquet" not in problem

    def test_droppings_are_skipped(self, repo):
        for rel in ("examples/a/data/.DS_Store", "examples/a/data/._t.parquet",
                    "examples/a/data/__pycache__/m.pyc",
                    "examples/a/data/.ipynb_checkpoints/n.ipynb"):
            _write(repo.root, rel)
        assert check(repo.root) == []

    def test_files_outside_data_directories_are_not_the_checks_business(self, repo):
        _write(repo.root, "examples/a/notes.csv")
        _write(repo.root, "examples/a/runs/out.log")
        _write(repo.root, "tests/data/x.parquet")
        assert check(repo.root) == []

    def test_a_long_list_is_cut_but_counted(self, repo):
        for i in range(_MAX_LINES + 5):
            _write(repo.root, f"examples/a/data/f{i:03d}.nc4")
        (problem,) = check(repo.root)
        assert f"{_MAX_LINES + 5} file(s)" in problem
        assert "and 5 more" in problem
