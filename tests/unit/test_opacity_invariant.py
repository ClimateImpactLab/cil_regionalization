"""Hierid (and other id_field) opacity invariant.

The library treats hierid as an OPAQUE identifier; a pure join key
whose structure must not be parsed in production code. The usual
``country.adm1.adm2`` shape is a vintage REGULARITY, not a contract;
R-suffixed clustering remainders (e.g. ``AND.Ra5cc0db7a54d1bb3``)
already break clean hierarchy semantics, and any code path that
introduces a structural assumption will silently misbehave on the next
IR vintage.

Allowed exceptions
------------------
- the ``segweights regions find`` CLI (in ``segment_weights/cli.py``)
  does literal pattern matching that the *user* supplies: the user
  passes a SQL ``LIKE`` pattern, the library translates it to a regex
  in the local-source path, but the library itself interprets nothing
  about hierid structure;
- vintage regularities are characterised in prose only: the measured
  world-combo-201710 profile lives in the ``segment_weights/regions.py``
  module docstring, with no conclusions encoded in code.

The guard below greps the package source for the obvious structural-
parsing patterns. False positives can be calibrated with more specific
regexes when they appear; the goal is to make "I quietly introduced
hierid parsing" impossible without explicit intent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import segment_weights


_PACKAGE_DIR = Path(segment_weights.__file__).resolve().parent

# Files allowed to do hierid-pattern operations on USER-SUPPLIED input.
# Keep this list as short as possible; every entry is a request for
# justification in code review.
_ALLOWED = {
    # The regions find CLI is the documented allowed exception. Even
    # there, the only "parsing" is translating the user's SQL LIKE
    # pattern to a regex; the library never interprets hierid structure.
    "cli.py",
}

# Patterns are matched against the source text as raw strings. Each
# row is (regex, human label). False positives produce specific failure
# messages, which is the point.
_FORBIDDEN: list[tuple[str, str]] = [
    (r"REGEXP_EXTRACT\([^)]*hierid", "REGEXP_EXTRACT on hierid in SQL"),
    (r"REGEXP_REPLACE\([^)]*hierid", "REGEXP_REPLACE on hierid in SQL"),
    (r"REGEXP_CONTAINS\([^)]*hierid", "REGEXP_CONTAINS on hierid in SQL"),
    (r"STRPOS\([^)]*hierid", "STRPOS on hierid in SQL"),
    (r"SPLIT\([^)]*hierid", "SPLIT on hierid in SQL"),
    (r"hierid\.split\(", "Python .split on hierid value"),
    (
        r"\[\s*['\"]hierid['\"]\s*\]\.str\.split",
        "Pandas .str.split on hierid column",
    ),
    (
        r"\[\s*['\"]hierid['\"]\s*\]\.str\.extract",
        "Pandas .str.extract on hierid column",
    ),
    (
        r"\[\s*['\"]hierid['\"]\s*\]\.str\.startswith",
        "Pandas .str.startswith on hierid column",
    ),
    (
        r"\[\s*['\"]hierid['\"]\s*\]\.str\.match",
        "Pandas .str.match on hierid column",
    ),
    (
        r"\[\s*['\"]hierid['\"]\s*\]\.str\.contains",
        "Pandas .str.contains on hierid column",
    ),
    # Generic primary_id substring ops in SQL (covers any future
    # id_field column name without grepping for every literal).
    (r"REGEXP_EXTRACT\([^)]*primary_id", "REGEXP_EXTRACT on primary_id"),
    (r"REGEXP_REPLACE\([^)]*primary_id", "REGEXP_REPLACE on primary_id"),
    (r"STRPOS\([^)]*primary_id", "STRPOS on primary_id"),
    (r"SPLIT\([^)]*primary_id", "SPLIT on primary_id"),
]


def test_no_hierid_structural_parsing():
    violations: list[tuple[str, str, str]] = []
    for py_file in _PACKAGE_DIR.rglob("*.py"):
        if py_file.name in _ALLOWED:
            continue
        text = py_file.read_text()
        for pattern, label in _FORBIDDEN:
            if re.search(pattern, text):
                rel = str(py_file.relative_to(_PACKAGE_DIR.parent))
                violations.append((rel, pattern, label))

    if violations:
        msg = ["hierid structural parsing detected in production code:"]
        for path, pattern, label in violations:
            msg.append(f"  {path}: {label}  (matched /{pattern}/)")
        msg.append("")
        msg.append(
            "hierid is opaque. Vintage regularities (country.adm1.adm2, "
            "ISO3 prefixes, etc.) are NOT a contract; the next IR "
            "vintage may break them silently. If profiling a vintage, "
            "record it in prose in the regions.py module docstring. If "
            "letting a user search, do it in the regions find CLI."
        )
        raise AssertionError("\n".join(msg))


def test_allowed_files_have_a_reason():
    """Smoke-test the allowlist; the only allowed exception today is
    the regions find CLI. If a new file is added to ``_ALLOWED`` it
    should have a comment justifying it."""
    assert _ALLOWED == {"cli.py"}, (
        "opacity invariant: the allowlist is structured to require an "
        "explicit code-review request to extend. Update the test "
        "deliberately if you genuinely need another exception."
    )
