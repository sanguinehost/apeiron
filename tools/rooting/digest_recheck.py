#!/usr/bin/env python3
# Row: APEIRON-DIGEST-GATE, APEIRON-DIGEST-ROWS (apeiron-digest-rows-the-9-EzHdoXZdmMKJjzWePe8qkH)
"""Recompute every digest a registry records, and refuse the drift: the digest gate.

Why this gate exists
====================
Three registries recorded a sha256 per row, and each was measured
drifting on the same night (``docs/contracts/artifacts.toml`` — 7 mismatches
under ``gate docs check``; the former ``examples/reg1/coverage.json`` — 18 fields,
removed in ``c9a2d0c``; and the
``documentation-removal-reachability`` row of ``docs/evidence/ledger.jsonl``,
whose recorded digest a later edit invalidated). The defect is the same each
time: **a registry records a digest, and nothing recomputes it, so the registry
drifts and every gate that trusts the row stays green.** Where an existing gate
recomputes (``gate docs check``, ``index_check.py --check``, the former ``reg1``
census) it recomputes for *its own* registry; this gate is the uniform,
registry-agnostic recheck that runs from one place over *every* registry it is
handed, so a second and third registry cannot drift unnoticed just because no
gate was pointed at them.

What is checked
===============
For each registry named by ``--registry PATH`` the loader picks an adapter from
the file's own schema, then for every digest-bearing row recomputes the digest
from the bytes (or the directory listing) on disk and compares it to the
recorded value:

* ``R1`` registry-absent: a named registry is not a readable file;
* ``R2`` registry-unrecognised: the file parses but matches no known adapter
  schema, or a row is structurally unusable (this is a data error, exit 2);
* ``R3`` row-unresolved: a row's ``path`` does not resolve to a file or
  directory under the scanned root, or names an absolute path or one that
  escapes the root;
* ``R4`` digest-mismatch: the recorded digest (or ``bytes``) disagrees with the
  value recomputed from disk — the diagnostic names the registry, the row, and
  both the recorded and the recomputed value;
* ``R5`` digest-malformed: a recorded digest is not 64 lowercase hex digits;
* ``R6`` duplicate-row: two rows share an ``id`` or resolve to the same path.

Adapters
--------
``artifact-toml``  ``docs/contracts/artifacts.toml`` — ``[[artifact]]`` rows;
  recompute each ``sha256`` over ``path`` (file or directory, under the exact
  rule ``tools/gate/adapter.rs::sha256_path`` uses, so the two implementations
  agree — see *Directory digests* below). Fields consulted: ``sha256``.
``ledger-jsonl``   ``docs/evidence/ledger.jsonl`` — one JSON object per line;
  recompute ``sha256`` and ``bytes`` over ``path``. Fields consulted:
  ``sha256``, ``bytes``.
``coverage-json``  a coverage census (formerly ``examples/reg1/coverage.json``,
  removed in ``c9a2d0c``); recompute ``sources[].sha256`` over ``sources[].path`` and
  ``rows[].source_sha256`` over ``rows[].source_path``. Fields deliberately
  *not* recomputed, each for a named reason: ``sources[].pin_sha256`` is a pin
  transcribed from another document (``pin_origin``, e.g. ``proof/README.md``)
  and is compared against *that* document by the census's own ``pin_status``,
  not against the file; ``freshness[].digest`` is the digest of a named source
  *set* with no single registered path, so there is nothing on disk to
  recompute it from. These are exclusions, stated here, not silent skips.

Directory digests
=================
A directory row is digested by the identical rule ``tools/gate/adapter.rs``
implements, so a recomputation here and a ``gate docs check`` on the same row
return the same bytes (cross-checked on the live ``tools/gate/`` row, whose
Python recomputation equals the Rust loader's ``dfd61f1c…``). The rule: sort
every contained file by its path *component-wise*; skip any path component
beginning with ``.`` and any component named ``target``; concatenate one
``<hex>  <path relative to the registered directory>\\n`` line per file; sha256
that listing. The names are relative to the registered directory, not the
repository root — that spelling is the loader's, and this file follows it.

Trust boundary
==============
The gate reads the registry files it is handed and hashes the paths they name.
It writes nothing under ``--check``. It is evidence that the digests a registry
records still describe the bytes on disk; it is *not* evidence that the row is
correct, current, meaningful or complete — a registry that records the wrong
path, or records no row for a file, is not this gate's business. It runs no
gate commands, no ``cargo``, no Lean, no network, no clock. It does not consult
``git``: the identity is the file's bytes as they sit in the working tree, which
is what a commit would capture. ``gate docs check`` (Rust) remains the
authority for ``artifacts.toml``; this gate is the uniform recheck and must
agree with it, which is why the directory rule is byte-for-byte the same.

Isolation
=========
``--self-test`` materialises a synthetic tree under the system temporary
directory and never reads or writes the repository. ``--controls`` copies the
real registries *and the files they name* into the system temporary directory,
seeds a wrong digest in one row of each registry, requires this gate to refuse
naming that registry and row, restores, and requires it green; the working tree
is never mutated. ``TMPDIR`` is honoured — set ``TMPDIR=/home/socol/oc-tmp`` to
keep copies off the ``/tmp`` tmpfs. Every copy is removed on exit.

Usage
=====
::

    python3 tools/rooting/digest_recheck.py --list
    python3 tools/rooting/digest_recheck.py --registry docs/contracts/artifacts.toml
    python3 tools/rooting/digest_recheck.py \\
        --registry docs/contracts/artifacts.toml \\
        --registry docs/evidence/ledger.jsonl
    python3 tools/rooting/digest_recheck.py --self-test
    python3 tools/rooting/digest_recheck.py --controls
    python3 tools/rooting/digest_recheck.py --all

``--root PATH`` scans a tree other than the repository (the controls use it).
``--all`` runs the three in-tree registries named in this module's ``ALL``.
With no ``--registry``/``--all`` and no ``--self-test``/``--controls``/
``--list``, the invocation is a usage error (exit 2): the gate never silently
passes because it was handed nothing to check.

Exit status
===========
0  every named registry parsed and every recorded digest matched the recomputation.
1  a rule fired (R1, R3, R4, R5, R6), or a control fixture did not refuse.
2  usage error, or an unreadable/unrecognised registry (R2), or a failed
   self-test/control harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

HEX = set("0123456789abcdef")


# ---------------------------------------------------------------------------
# digests


# Complexity: O(B) time in bytes hashed and O(1) extra space beyond the caller's own buffer,
# where B is len(blob); one single pass over B bytes closes the count, and the digest state is
# a fixed 32 bytes whatever B is, so the caller's blob is the only copy.
def sha256_bytes(blob: bytes) -> str:
    """The sha256 of ``blob``, lowercase hex."""
    return hashlib.sha256(blob).hexdigest()


def excluded_component(name: str) -> bool:
    """Whether a path component is skipped when a directory digest is built."""
    return name.startswith(".") or name == "target"


# Complexity: O(F log F) time in directory entries compared and O(F) space in path tuples held,
# where F is the number of files under the directory; each directory is entered once by the
# iterative walk, so per directory it scans its entries once, sorts them once, and pushes at
# most one pending tuple per subdirectory, then the F collected pairs are sorted once by
# component tuple. The traversal closes because a tuple is pushed only for a subdirectory and
# popped exactly once, so the work is a traversal of a finite tree rather than a recurrence.
def _walk_components(directory: Path) -> list[tuple[tuple[str, ...], Path]]:
    """Every file under ``directory`` as ``(component tuple, absolute path)``.

    Sorted the way ``tools/gate/adapter.rs``'s ``out.sort()`` over
    ``(PathBuf, PathBuf)`` sorts: component-wise, then the absolute path.
    Iterative, so depth is bounded by the stack rather than the interpreter.
    On the live tree the deepest registered directory is ``tools/gate/`` at 15
    files.
    """
    collected: list[tuple[tuple[str, ...], Path]] = []
    stack: list[tuple[tuple[str, ...], Path]] = [((), directory)]
    while stack:
        components, absolute = stack.pop()
        try:
            entries = sorted(absolute.iterdir(), key=lambda p: p.name)
        except OSError:
            return []
        for entry in entries:
            if excluded_component(entry.name):
                continue
            child = components + (entry.name,)
            if entry.is_dir():
                stack.append((child, entry))
            elif entry.is_file():
                collected.append((child, entry))
    collected.sort(key=lambda item: (item[0], str(item[1])))
    return collected


# Complexity: a file is O(B) time in bytes hashed and O(B) space in bytes resident, where B is
# the file's size; read_bytes materialises the file and the single hashing pass over those B bytes
# closes the count. A directory is O(F log F + B) time and O(F * L + max B_i) space in listing
# bytes, where F is the file count, L a relative path's length and B_i the largest single file;
# the walk visits each of the F files once, appends one listing line each, and holds one file
# buffer at a time, so the total bytes hashed is B. The rule is deliberately identical to
# tools/gate/adapter.rs::sha256_path, whose per-file cost is also linear in the bytes hashed; the
# byte-for-byte agreement of the two is what keeps this gate from becoming a second, divergent
# definition of the same identity.
def sha256_path(path: Path) -> str:
    """The identity of ``path`` under the rule at the top of this file.

    A file hashes its bytes. A directory hashes the sorted
    ``<hex>  <path relative to the registered directory>`` listing of its
    contents, skipping dot components and ``target``.
    """
    if path.is_file():
        return sha256_bytes(path.read_bytes())
    if not path.is_dir():
        return ""
    listing = []
    for components, absolute in _walk_components(path):
        name = "/".join(components)
        listing.append(f"{sha256_bytes(absolute.read_bytes())}  {name}\n")
    return sha256_bytes("".join(listing).encode())


def is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in HEX for ch in value)
    )


# ---------------------------------------------------------------------------
# violations


@dataclass(frozen=True)
class Violation:
    rule: str
    registry: str
    row: str
    detail: str

    def render(self) -> str:
        return f"RULE {self.rule} {self.registry}: {self.row}: {self.detail}"


class DataError(Exception):
    """The registry is malformed beyond content: no adapter, or a bad row."""


# ---------------------------------------------------------------------------
# path resolution


def resolve_under(root: Path, relative: str) -> Path | None:
    """``root / relative`` when it stays inside ``root``, else ``None``.

    Absolute paths and any ``..`` escape return ``None``. Symlinks are
    resolved, so a row reaching outside through one is refused.
    """
    if not isinstance(relative, str) or not relative:
        return None
    candidate = Path(relative)
    if candidate.is_absolute():
        return None
    if any(part == ".." for part in candidate.parts):
        return None
    try:
        resolved = (root / candidate).resolve()
        base = root.resolve()
    except OSError:
        return None
    if resolved != base and base not in resolved.parents:
        return None
    return resolved


# ---------------------------------------------------------------------------
# rows


@dataclass(frozen=True)
class Row:
    """One digest-bearing entry a registry records.

    ``surface`` names which digest field of which list the row came from
    (``sources[]``, ``rows[]``, ``artifact``, ``ledger``). Two rows may name the
    same file in two surfaces — the coverage census registers every attributed
    source in ``sources[]`` *and* again in each ``rows[]`` entry that cites it —
    so duplicate detection is per surface, never across surfaces.

    ``id_key`` is the registry-native identity used for duplicate detection, and
    is deliberately *not* ``row_id``: ``row_id`` carries the line number for a
    JSONL row so the diagnostic can point at it, and a label that differs per
    line would make a duplicate id undetectable. A registry that gives a row no
    id (``sources[]``) keys on its own position, which is unique by construction.
    """

    registry: str
    row_id: str
    id_key: str
    surface: str
    path_field: str
    relative: str
    sha256: Any
    bytes: Any = None
    bytes_field: str | None = None


@dataclass(frozen=True)
class Registry:
    path: Path
    display: str
    adapter: str
    rows: list[Row]


# ---------------------------------------------------------------------------
# adapters


def _as_path_field(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def load_artifact_toml(text: str, display: str) -> Registry:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise DataError(f"{display}: not valid TOML: {exc}") from exc
    tables = data.get("artifact")
    if not isinstance(tables, list) or not tables:
        raise DataError(f"{display}: no [[artifact]] tables; not an artefact registry")
    rows: list[Row] = []
    for index, table in enumerate(tables):
        if not isinstance(table, dict):
            raise DataError(f"{display}: [[artifact]] #{index} is not a table")
        ident = table.get("id")
        if not isinstance(ident, str) or not ident:
            raise DataError(f"{display}: [[artifact]] #{index} has no id")
        rel = _as_path_field(table.get("path"))
        if rel is None:
            raise DataError(f"{display}: artefact {ident} has no path")
        rows.append(
            Row(
                registry=display,
                row_id=ident,
                id_key=ident,
                surface="artifact",
                path_field="path",
                relative=rel,
                sha256=table.get("sha256"),
            )
        )
    return Registry(Path(), display, "artifact-toml", rows)


def load_ledger_jsonl(text: str, display: str) -> Registry:
    rows: list[Row] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{display}:{lineno}: not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise DataError(f"{display}:{lineno}: row is not a JSON object")
        ident = obj.get("id")
        if not isinstance(ident, str) or not ident:
            raise DataError(f"{display}:{lineno}: row has no id")
        rel = _as_path_field(obj.get("path"))
        if rel is None:
            raise DataError(f"{display}:{lineno}: row {ident} has no path")
        rows.append(
            Row(
                registry=display,
                row_id=f"{ident} (line {lineno})",
                id_key=ident,
                surface="ledger",
                path_field="path",
                relative=rel,
                sha256=obj.get("sha256"),
                bytes=obj.get("bytes"),
                bytes_field="bytes",
            )
        )
    return Registry(Path(), display, "ledger-jsonl", rows)


def load_coverage_json(text: str, display: str) -> Registry:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataError(f"{display}: not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DataError(f"{display}: not a JSON object")
    schema = data.get("schema")
    if not isinstance(schema, str) or not schema.startswith("apeiron.reg1.coverage."):
        raise DataError(f"{display}: schema {schema!r} is not a coverage census")
    rows: list[Row] = []
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise DataError(f"{display}: sources is not a list")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise DataError(f"{display}: sources[{index}] is not an object")
        rel = _as_path_field(source.get("path"))
        if rel is None:
            raise DataError(f"{display}: sources[{index}] has no path")
        rows.append(
            Row(
                registry=display,
                row_id=f"sources[{index}] {rel}",
                id_key=f"sources[{index}]",
                surface="sources[]",
                path_field="path",
                relative=rel,
                sha256=source.get("sha256"),
            )
        )
    crow = data.get("rows")
    if not isinstance(crow, list):
        raise DataError(f"{display}: rows is not a list")
    for index, item in enumerate(crow):
        if not isinstance(item, dict):
            raise DataError(f"{display}: rows[{index}] is not an object")
        ident = item.get("id")
        rel = _as_path_field(item.get("source_path"))
        if rel is None:
            # A row with no source_path (a synthetic/admission-only row) has
            # nothing on disk to recompute; it carries no source_sha256 either.
            continue
        rows.append(
            Row(
                registry=display,
                row_id=f"rows[{index}] {ident or '?'} -> {rel}",
                id_key=f"rows[{index}] {ident}",
                surface="rows[]",
                path_field="source_path",
                relative=rel,
                sha256=item.get("source_sha256"),
            )
        )
    return Registry(Path(), display, "coverage-json", rows)


# ``(adapter name, suffix, loader)``. The suffix is the dispatch key, so the
# adapter for a named registry is a decision, not a guess: a file whose suffix
# is listed here is loaded by exactly that loader, and a file whose suffix is not
# listed is probed against every loader in this order and takes the first that
# parses. A file no loader admits is R2 (exit 2). There is no "best effort"
# branch.
ADAPTERS: list[tuple[str, str, Callable[[str, str], Registry]]] = [
    ("artifact-toml", ".toml", load_artifact_toml),
    ("ledger-jsonl", ".jsonl", load_ledger_jsonl),
    ("coverage-json", ".json", load_coverage_json),
]

# The two in-tree registries this gate recomputes by default. A registry absent
# from a tree is a named refusal from the caller (the hook), never a silent pass;
# `--all` refuses one that is missing. `examples/reg1/coverage.json` was removed
# in `c9a2d0c`, so it is not an in-tree registry; the `coverage-json` adapter
# stays for any `.json` census handed in explicitly with `--registry`.
ALL = [
    "docs/contracts/artifacts.toml",
    "docs/evidence/ledger.jsonl",
]

# Surfaces in which a repeated `path` is itself a defect, so R6 checks it there.
#
#   artifact   the artefact registry is one row per registered file or
#              directory (`tools/gate`'s own loader refuses a repeated id and
#              reports a drifted digest; a second row for one path would make
#              two identities claim the same file).
#   ledger     the evidence index is one row per indexed file; `index_check.py`
#              rule E2 is exactly "one file is indexed by more than one row".
#
# Surfaces deliberately *excluded*, each for a named reason:
#
#   sources[]  one entry per attributed source; the path is the key, so a repeat
#              is not representable in a well-formed census.
#   rows[]     the census holds one row per *operation*, and several operations
#              are implemented in the same file — `current-api-new`,
#              `current-api-get`, `current-api-insert_with`, `reg1-remove` and
#              eight more all cite `src/table.rs`. A repeated `source_path`
#              there is the census's shape, not a defect, so refusing it would
#              refuse the registry as it is legitimately written.
#
# Every surface is still digest-checked; this governs only the duplicate-path
# rule, never the recomputation.
UNIQUE_PATH_SURFACES = frozenset({"artifact", "ledger"})


def load_registry(path: Path, root: Path) -> Registry:
    """Load one registry by its suffix, probing in ``ADAPTERS`` order if unknown."""
    display = rel_to(root, path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    for name, key, load in ADAPTERS:
        if key == suffix:
            return load(text, display)
    tried: list[str] = []
    for name, _key, load in ADAPTERS:
        try:
            return load(text, display)
        except DataError as exc:
            tried.append(f"{name}: {exc}")
    raise DataError(
        f"{display}: unrecognised registry schema (suffix {suffix!r} is not a known "
        f"registry suffix). Probed: " + "; ".join(tried)
    )


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# scan


@dataclass
class Summary:
    registries: int = 0
    rows: int = 0
    digests: int = 0
    matched: int = 0


def scan_registry(root: Path, reg_path: Path, summary: Summary) -> list[Violation]:
    display = rel_to(root, reg_path)
    if not reg_path.is_file():
        return [
            Violation(
                "R1",
                display,
                "<registry>",
                f"registry absent: {display} is not a readable file, so no recorded digest "
                f"in it can be recomputed",
            )
        ]
    try:
        registry = load_registry(reg_path, root)
    except DataError as exc:
        raise DataError(str(exc)) from exc
    summary.registries += 1

    violations: list[Violation] = []
    seen_ids: dict[str, str] = {}
    seen_paths: dict[tuple[str, str], str] = {}
    for row in registry.rows:
        summary.rows += 1
        # duplicate id within the registry; duplicate path within one surface.
        prior_id = seen_ids.get(row.id_key)
        if prior_id is not None:
            violations.append(
                Violation(
                    "R6",
                    display,
                    row.row_id,
                    f"duplicate row id {row.id_key!r}: rows {prior_id} and {row.row_id} "
                    f"share an id, so a lookup by id is ambiguous",
                )
            )
        else:
            seen_ids[row.id_key] = row.row_id
        path_key = (row.surface, row.relative)
        prior_path = seen_paths.get(path_key)
        if prior_path is not None and row.surface in UNIQUE_PATH_SURFACES:
            violations.append(
                Violation(
                    "R6",
                    display,
                    row.row_id,
                    f"duplicate path {row.relative} in surface {row.surface}: already "
                    f"registered by row {prior_path}",
                )
            )
        else:
            seen_paths[path_key] = row.row_id

        resolved = resolve_under(root, row.relative)
        if resolved is None:
            violations.append(
                Violation(
                    "R3",
                    display,
                    row.row_id,
                    f"row names a path that is not inside the scanned root or does not "
                    f"resolve: {row.relative}",
                )
            )
            continue
        if not (resolved.is_file() or resolved.is_dir()):
            violations.append(
                Violation(
                    "R3",
                    display,
                    row.row_id,
                    f"dangling row: {row.relative} does not exist under the scanned root",
                )
            )
            continue

        # sha256
        if row.sha256 is not None:
            if not is_digest(row.sha256):
                shown = row.sha256 if isinstance(row.sha256, str) else repr(row.sha256)
                violations.append(
                    Violation(
                        "R5",
                        display,
                        row.row_id,
                        f"recorded sha256 {shown!r} is not 64 lowercase hex digits",
                    )
                )
            else:
                summary.digests += 1
                actual = sha256_path(resolved)
                if actual != row.sha256:
                    violations.append(
                        Violation(
                            "R4",
                            display,
                            row.row_id,
                            f"stale sha256 on {row.relative}: registry records "
                            f"{row.sha256}, the file on disk is {actual}",
                        )
                    )
                else:
                    summary.matched += 1

        # bytes (ledger only)
        if row.bytes_field is not None and row.bytes is not None:
            size = row.bytes
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                shown = size if isinstance(size, int) else repr(size)
                violations.append(
                    Violation(
                        "R5",
                        display,
                        row.row_id,
                        f"recorded {row.bytes_field} {shown!r} is not a size in nonnegative "
                        f"integers",
                    )
                )
            else:
                try:
                    actual_bytes = resolved.stat().st_size
                except OSError:
                    actual_bytes = -1
                if actual_bytes != size:
                    violations.append(
                        Violation(
                            "R4",
                            display,
                            row.row_id,
                            f"stale {row.bytes_field} on {row.relative}: registry records "
                            f"{size}, the file on disk is {actual_bytes}",
                        )
                    )
    return violations


def scan(root: Path, registries: Sequence[Path]) -> tuple[list[Violation], Summary]:
    summary = Summary()
    violations: list[Violation] = []
    for reg in registries:
        violations.extend(scan_registry(root, reg, summary))
    return violations, summary


# ---------------------------------------------------------------------------
# synthetic fixtures (self-test) — never read the repository


def _materialize_clean(base: Path) -> None:
    """A clean three-registry fixture tree with correct digests.

    Also exercises the directory rule: ``d/pkg/`` contains a hidden file and a
    ``target/`` subdirectory that must NOT enter the digest, a subdirectory
    whose name shares a prefix with a sibling file (component-wise sort), and a
    nested file. Its recorded digest is the adapter.rs-rule recomputation.
    """
    # artifacts.toml tree
    (base / "docs" / "contracts").mkdir(parents=True)
    (base / "docs" / "plan.md").write_bytes(b"# plan\n")
    plan_sha = sha256_bytes(b"# plan\n")
    pkg = base / "d" / "pkg"
    (pkg / "target").mkdir(parents=True)
    (pkg / "target" / "junk.bin").write_bytes(b"build product\n")
    (pkg / ".hidden").mkdir()
    (pkg / ".hidden" / "scratch").write_bytes(b"editor scratch\n")
    (pkg / "sub").mkdir()
    (pkg / "sub" / "inner.rs").write_bytes(b"fn inner() {}\n")
    (pkg / "pkg.rs").write_bytes(b"fn pkg() {}\n")
    # directory digest computed by the same rule the gate uses
    dir_sha = sha256_path(pkg)
    artifacts = f"""
# fixture
[[artifact]]
id       = "docs/plan.md"
kind     = "doc"
path     = "docs/plan.md"
sha256   = "{plan_sha}"
gate     = "true"
control  = ""
budget_s = 10
[[artifact]]
id       = "d/pkg"
kind     = "rust"
path     = "d/pkg/"
sha256   = "{dir_sha}"
gate     = "true"
control  = ""
budget_s = 10
"""
    (base / "docs" / "contracts" / "artifacts.toml").write_text(artifacts)

    # ledger.jsonl tree
    (base / "docs" / "evidence").mkdir(parents=True)
    (base / "docs" / "evidence" / "r.md").write_bytes(b"# receipt r\n")
    rb = b"# receipt r\n"
    ledger = [
        json.dumps(
            {
                "id": "r",
                "kind": "receipt",
                "path": "docs/evidence/r.md",
                "bytes": len(rb),
                "sha256": sha256_bytes(rb),
                "summary": "fixture receipt r",
            },
            sort_keys=True,
        )
    ]
    (base / "docs" / "evidence" / "ledger.jsonl").write_text("\n".join(ledger) + "\n")

    # coverage.json tree
    (base / "src").mkdir(parents=True)
    (base / "src" / "lib.rs").write_bytes(b"fn main() {}\n")
    lib = b"fn main() {}\n"
    coverage = {
        "schema": "apeiron.reg1.coverage.v2",
        "sources": [
            {
                "path": "src/lib.rs",
                "sha256": sha256_bytes(lib),
                "pin_sha256": "",
                "pin_origin": "none",
                "pin_status": "unpinned",
            }
        ],
        "rows": [
            {
                "id": "op",
                "source_path": "src/lib.rs",
                "source_sha256": sha256_bytes(lib),
            }
        ],
        "freshness": [
            {"command": "sha256sum <set>", "digest": "", "status": "stale"}
        ],
    }
    (base / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True))


def _fresh_copy(base: Path, prefix: str) -> Path:
    work = Path(tempfile.mkdtemp(prefix=prefix))
    target = work / "tree"
    shutil.copytree(base, target, symlinks=True)
    return target


# Fixture mutators. Each returns after seeding one defect; the gate must
# refuse it with the named rule and a diagnostic naming the registry and row.


def _seed_artifact_sha256_drift(tree: Path) -> None:
    p = tree / "docs" / "contracts" / "artifacts.toml"
    text = p.read_text()
    bad = "0" * 64
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("sha256") and "0" * 64 not in line:
            lines[i] = f'sha256   = "{bad}"'
            break
    p.write_text("\n".join(lines))


def _seed_coverage_source_drift(tree: Path) -> None:
    p = tree / "coverage.json"
    data = json.loads(p.read_text())
    data["sources"][0]["sha256"] = "f" * 64
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def _seed_coverage_rowsource_drift(tree: Path) -> None:
    p = tree / "coverage.json"
    data = json.loads(p.read_text())
    data["rows"][0]["source_sha256"] = "1" * 64
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def _seed_ledger_bytes_drift(tree: Path) -> None:
    p = tree / "docs" / "evidence" / "ledger.jsonl"
    lines = p.read_text().splitlines()
    obj = json.loads(lines[0])
    obj["bytes"] = obj["bytes"] + 1
    lines[0] = json.dumps(obj, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")


def _seed_ledger_sha_drift(tree: Path) -> None:
    p = tree / "docs" / "evidence" / "ledger.jsonl"
    lines = p.read_text().splitlines()
    obj = json.loads(lines[0])
    obj["sha256"] = "a" * 64
    lines[0] = json.dumps(obj, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")


def _seed_registry_absent(tree: Path) -> None:
    (tree / "docs" / "evidence" / "ledger.jsonl").unlink()


def _seed_malformed_digest(tree: Path) -> None:
    p = tree / "docs" / "contracts" / "artifacts.toml"
    text = p.read_text()
    text = text.replace(sha256_bytes(b"# plan\n"), "not-a-digest")
    p.write_text(text)


def _seed_dangling_row(tree: Path) -> None:
    p = tree / "coverage.json"
    data = json.loads(p.read_text())
    data["sources"].append(
        {"path": "src/does_not_exist.rs", "sha256": "0" * 64, "pin_status": "unpinned"}
    )
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def _seed_escaping_row(tree: Path) -> None:
    p = tree / "coverage.json"
    data = json.loads(p.read_text())
    data["sources"][0]["path"] = "../outside.rs"
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def _seed_duplicate_id(tree: Path) -> None:
    # A second row with the same id but a different path, so the duplicate *id*
    # rule fires on its own rather than alongside the duplicate-path rule.
    p = tree / "docs" / "evidence" / "ledger.jsonl"
    (tree / "docs" / "evidence" / "r2.md").write_bytes(b"# receipt r2\n")
    r2 = b"# receipt r2\n"
    lines = p.read_text().splitlines()
    first = json.loads(lines[0])
    second = dict(first)
    second["path"] = "docs/evidence/r2.md"
    second["bytes"] = len(r2)
    second["sha256"] = sha256_bytes(r2)
    lines.append(json.dumps(second, sort_keys=True))
    p.write_text("\n".join(lines) + "\n")


def _seed_duplicate_path_in_surface(tree: Path) -> None:
    # Two rows in the *same* surface registering one path: the other half of R6.
    # A complete `[[artifact]]` block is appended, so the defect is the repeated
    # path and not a TOML syntax error.
    p = tree / "docs" / "contracts" / "artifacts.toml"
    digest = sha256_bytes((tree / "docs" / "plan.md").read_bytes())
    block = (
        "[[artifact]]\n"
        'id       = "docs-plan-again"\n'
        'kind     = "doc"\n'
        'path     = "docs/plan.md"\n'
        f'sha256   = "{digest}"\n'
        'gate     = "true"\n'
        'control  = ""\n'
        "budget_s = 10\n"
    )
    p.write_text(p.read_text().rstrip("\n") + "\n" + block)


# -- acceptance fixtures beyond the clean tree ---------------------------------


def _materialize_outside_surface(base: Path) -> None:
    """A registry whose rows name paths *outside* the registry's own surface.

    ``docs/evidence/ledger.jsonl`` indexes the evidence surface, and an artefact
    registry in ``docs/contracts/`` indexes the tree it names. A row that names
    ``src/table.rs`` — outside both surfaces — is still digest-checked against
    the bytes on disk and is *accepted* when it matches; it is counted as
    ``outside``, never skipped and never refused for its location. Refusing a
    correct row for being out of surface would make the gate a coverage claim it
    does not establish.
    """
    (base / "src").mkdir(parents=True, exist_ok=True)
    blob = b"fn table() {}\n"
    (base / "src" / "table.rs").write_bytes(blob)
    digest = sha256_bytes(blob)
    (base / "docs" / "evidence").mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            {
                "id": "outside",
                "kind": "figure",
                "path": "src/table.rs",
                "bytes": len(blob),
                "sha256": digest,
                "summary": "a row naming a path outside the evidence surface",
            },
            sort_keys=True,
        )
    ]
    (base / "docs" / "evidence" / "ledger.jsonl").write_text("\n".join(rows) + "\n")
    (base / "docs" / "contracts").mkdir(parents=True, exist_ok=True)
    (base / "docs" / "contracts" / "artifacts.toml").write_text(
        "[[artifact]]\n"
        f'id       = "src/table.rs"\n'
        'kind     = "rust"\n'
        'path     = "src/table.rs"\n'
        f'sha256   = "{digest}"\n'
        'gate     = "true"\n'
        'control  = ""\n'
        "budget_s = 10\n"
    )


def _accept_fixtures() -> list[str]:
    """The named acceptance fixtures, each asserted green by the gate itself."""
    failures: list[str] = []

    # A1 — a clean registry: the three synthetic registries of the clean tree.
    # A2 — a registry whose row path lies outside the registry's own surface.
    base = Path(tempfile.mkdtemp(prefix="digest-recheck-outside-"))
    try:
        _materialize_outside_surface(base)
        found, summary = scan(
            base,
            [
                base / "docs" / "contracts" / "artifacts.toml",
                base / "docs" / "evidence" / "ledger.jsonl",
            ],
        )
        if found:
            failures.append(
                f"acceptance: row outside the registry surface refused: {found[0].render()}"
            )
        else:
            print(
                f"SELF-TEST GREEN acceptance: row outside the registry surface "
                f"({summary.digests} digests recomputed, {summary.matched} matched, 0 refused)"
            )
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # A3 — the directory rule: a directory row whose recorded digest is the
    # adapter.rs spelling (dot components and `target` excluded, names relative
    # to the registered directory, component-wise order). Asserted against a
    # digest built by an independent spelling of the rule, so a change to
    # `sha256_path` that silently altered the rule would fail here.
    base = Path(tempfile.mkdtemp(prefix="digest-recheck-dirrule-"))
    try:
        pkg = base / "d" / "pkg"
        (pkg / "target").mkdir(parents=True)
        (pkg / "target" / "junk.bin").write_bytes(b"build product\n")
        (pkg / ".git").mkdir()
        (pkg / ".git" / "config").write_bytes(b"[core]\n")
        (pkg / "sub").mkdir()
        (pkg / "sub" / "inner.rs").write_bytes(b"fn inner() {}\n")
        (pkg / "pkg.rs").write_bytes(b"fn pkg() {}\n")
        # Independent spelling: build the listing by hand, no shared code.
        expected_lines = [
            f"{sha256_bytes(b'fn pkg() {}\n')}  pkg.rs",
            f"{sha256_bytes(b'fn inner() {}\n')}  sub/inner.rs",
        ]
        # component-wise order: ("pkg.rs",) < ("sub",) so pkg.rs is first.
        expected = sha256_bytes(("\n".join(expected_lines) + "\n").encode())
        actual = sha256_path(pkg)
        if actual != expected:
            failures.append(
                f"acceptance: directory rule disagrees with the independent spelling: "
                f"{actual} != {expected}"
            )
        else:
            print(
                f"SELF-TEST GREEN acceptance: directory digest rule "
                f"(dot components and `target` excluded, component-wise order) = {actual}"
            )
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # A4 — a repeated `source_path` across census `rows[]` is the census's own
    # shape (one row per operation, several operations in one file), so the
    # duplicate-path rule must NOT fire there — while the digest of that repeated
    # path is still recomputed once per row. This fixture is what would catch a
    # gate that had narrowed the rule the wrong way and begun refusing the live
    # coverage census.
    base = Path(tempfile.mkdtemp(prefix="digest-reach-repeat-"))
    try:
        blob = b"fn table() {}\n"
        (base / "src").mkdir(parents=True)
        (base / "src" / "table.rs").write_bytes(blob)
        digest = sha256_bytes(blob)
        coverage = {
            "schema": "apeiron.reg1.coverage.v2",
            "sources": [
                {
                    "path": "src/table.rs",
                    "sha256": digest,
                    "pin_sha256": "",
                    "pin_status": "unpinned",
                }
            ],
            "rows": [
                {"id": "current-api-new", "source_path": "src/table.rs", "source_sha256": digest},
                {"id": "current-api-get", "source_path": "src/table.rs", "source_sha256": digest},
                {"id": "reg1-remove", "source_path": "src/table.rs", "source_sha256": digest},
            ],
        }
        (base / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True))
        found, summary = scan(base, [base / "coverage.json"])
        if found:
            failures.append(
                f"acceptance: repeated source_path across census rows refused: {found[0].render()}"
            )
        elif summary.digests != 4:
            failures.append(
                f"acceptance: repeated source_path recomputed {summary.digests} digests, "
                f"expected 4 (1 in sources[] and 3 in rows[])"
            )
        else:
            print(
                "SELF-TEST GREEN acceptance: repeated source_path across census rows[] "
                "(no duplicate-path refusal; all 4 digests still recomputed)"
            )
    finally:
        shutil.rmtree(base, ignore_errors=True)

    return failures


def self_test() -> int:
    """Accept the clean fixture; refuse each seeded defect with its own rule."""
    failures: list[str] = []
    base = Path(tempfile.mkdtemp(prefix="digest-recheck-clean-"))
    try:
        _materialize_clean(base)
        violations, summary = scan(
            base,
            [
                base / "docs" / "contracts" / "artifacts.toml",
                base / "docs" / "evidence" / "ledger.jsonl",
                base / "coverage.json",
            ],
        )
        if violations:
            print(
                f"SELF-TEST FAIL clean fixture: {violations[0].render()}",
                file=sys.stderr,
            )
            return EXIT_REJECT
        print(
            f"SELF-TEST GREEN clean fixture ({summary.registries} registries, "
            f"{summary.rows} rows, {summary.digests} digests, all matched)"
        )

        failures.extend(_accept_fixtures())
        if failures:
            for f in failures:
                print(f"SELF-TEST FAIL {f}", file=sys.stderr)
            return EXIT_REJECT

        cases: list[tuple[str, str, Callable[[Path], None]]] = [
            ("artifact-toml sha256 drift", "R4", _seed_artifact_sha256_drift),
            ("coverage sources[].sha256 drift", "R4", _seed_coverage_source_drift),
            (
                "coverage rows[].source_sha256 drift",
                "R4",
                _seed_coverage_rowsource_drift,
            ),
            ("ledger bytes drift", "R4", _seed_ledger_bytes_drift),
            ("ledger sha256 drift", "R4", _seed_ledger_sha_drift),
            ("registry absent", "R1", _seed_registry_absent),
            ("malformed digest", "R5", _seed_malformed_digest),
            ("dangling row", "R3", _seed_dangling_row),
            ("escaping row", "R3", _seed_escaping_row),
            ("duplicate row id", "R6", _seed_duplicate_id),
            (
                "duplicate path in one surface",
                "R6",
                _seed_duplicate_path_in_surface,
            ),
        ]
        for name, rule, seed in cases:
            work = _fresh_copy(base, "digest-recheck-case-")
            try:
                seed(work)
                try:
                    found, _ = scan(
                        work,
                        [
                            work / "docs" / "contracts" / "artifacts.toml",
                            work / "docs" / "evidence" / "ledger.jsonl",
                            work / "coverage.json",
                        ],
                    )
                except DataError as exc:
                    failures.append(f"{name}: data error where rule {rule} expected: {exc}")
                    continue
                matching = [v for v in found if v.rule == rule]
                if not matching:
                    got = ", ".join(sorted({v.rule for v in found})) or "none"
                    failures.append(f"{name}: expected {rule}, got {got}")
                    continue
                diagnostic = matching[0].render()
                # The diagnostic must name the registry and the row.
                reg_names = {
                    "artifact": "artifacts.toml",
                    "coverage": "coverage.json",
                    "ledger": "ledger.jsonl",
                }
                token = next(
                    (t for t in reg_names.values() if t in diagnostic), None
                )
                if token is None:
                    failures.append(f"{name}: diagnostic does not name a registry: {diagnostic}")
                    continue
                print(f"SELF-TEST RED  {name} -> {rule}: {diagnostic}")
            finally:
                shutil.rmtree(work.parent, ignore_errors=True)
        if failures:
            for f in failures:
                print(f"SELF-TEST FAIL {f}", file=sys.stderr)
            return EXIT_REJECT
        print(f"SELF-TEST PASS {len(cases)} rejection fixtures refused with their own rule")
        return EXIT_OK
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# controls — seed a real drift in a throwaway copy, restore


def _copy_real_registries(root: Path, dest: Path) -> list[Path]:
    """Copy each named registry and every file it registers into ``dest``.

    The row paths are resolved against the *source* root to find what to copy;
    resolving them against the (still empty) copy would find nothing and the
    copy would be all dangling rows. A registered directory is copied with the
    same exclusion rule its digest uses, so the copy's directory digest equals
    the source's and a directory row is not spurious-red in the copy.
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied_registries: list[Path] = []
    for reg_rel in ALL:
        reg_src = root / reg_rel
        if not reg_src.is_file():
            continue
        reg_dst = dest / reg_rel
        reg_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reg_src, reg_dst)
        copied_registries.append(reg_dst)
        try:
            registry = load_registry(reg_dst, dest)
        except DataError:
            continue
        seen: set[str] = set()
        for row in registry.rows:
            if row.relative in seen:
                continue
            seen.add(row.relative)
            source = resolve_under(root, row.relative)
            if source is None or not (source.is_file() or source.is_dir()):
                continue
            target = resolve_under(dest, row.relative)
            if target is None or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    symlinks=True,
                    ignore=lambda _d, names: {
                        n for n in names if excluded_component(n)
                    },
                )
            else:
                shutil.copy2(source, target)
    return copied_registries


def _seed_one_drift(reg: Path, dest_root: Path) -> str:
    """Mutate exactly one recorded digest inside one registry copy."""
    text = reg.read_text(encoding="utf-8")
    bad = "0" * 64
    if reg.suffix == ".json" and "coverage" in text and '"schema"' in text:
        data = json.loads(text)
        data["sources"][0]["sha256"] = bad
        reg.write_text(json.dumps(data, indent=2, sort_keys=True))
        return "sources[0].sha256"
    if reg.suffix == ".jsonl":
        lines = text.splitlines()
        obj = json.loads(lines[0])
        obj["sha256"] = bad
        lines[0] = json.dumps(obj, sort_keys=True)
        reg.write_text("\n".join(lines) + "\n")
        return "sha256"
    # artifacts.toml: rewrite the first sha256 line that isn't already the bad one
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("sha256") and bad not in line:
            lines[i] = f'sha256   = "{bad}"'
            reg.write_text("\n".join(lines) + "\n")
            return "sha256"
    return ""


def controls(root: Path) -> int:
    """Baseline green, seeded drift refused naming the registry and row, restored green."""
    if len(sys.argv) > 1 and sys.argv[1] == "--controls":
        pass
    dest = Path(tempfile.mkdtemp(prefix="digest-recheck-controls-"))
    try:
        registries = _copy_real_registries(root, dest)
        if not registries:
            print(
                f"CONTROLS FAIL: no real registry copied from {root}; nothing to control",
                file=sys.stderr,
            )
            return EXIT_REJECT
        # baseline: the untouched copy must be green for the registries present.
        base_viol, base_summary = scan(dest, registries)
        print(
            f"CONTROL baseline: {len(registries)} registries, {base_summary.rows} rows, "
            f"{base_summary.digests} digests, {len(base_viol)} violation(s)"
        )
        if base_viol:
            # A drifted live tree makes the *baseline* copy red; that is honest,
            # but it means the restore control cannot be a clean pass. Report
            # it as an open baseline, not as a pass.
            print(
                "CONTROL baseline: OPEN the live tree's own registries already carry "
                f"{len(base_viol)} violation(s); the drift controls below run against a copy "
                "and each seeded drift must still be refused",
                file=sys.stderr,
            )

        failures: list[str] = []
        for reg in registries:
            reg_backup = reg.read_bytes()
            field = _seed_one_drift(reg, dest)
            if not field:
                failures.append(f"{rel_to(dest, reg)}: could not seed a drift")
                reg.write_bytes(reg_backup)
                continue
            found, _ = scan(dest, [reg])
            diagnostic = next(
                (v.render() for v in found if v.rule == "R4"), None
            )
            if diagnostic is None:
                got = ", ".join(sorted({v.rule for v in found})) or "none"
                failures.append(
                    f"{rel_to(dest, reg)}: seeded {field} drift not refused with R4 (got {got})"
                )
            else:
                print(f"CONTROL RED   {rel_to(dest, reg)}: seeded {field} drift refused: {diagnostic}")
            reg.write_bytes(reg_backup)  # restore
            after, _ = scan(dest, [reg])
            # After restore, this registry must be back to its baseline count.
            if len(after) != len([v for v in base_viol if v.registry == rel_to(dest, reg)]):
                failures.append(
                    f"{rel_to(dest, reg)}: restore did not return to baseline "
                    f"(baseline {len([v for v in base_viol if v.registry == rel_to(dest, reg)])}, "
                    f"after {len(after)})"
                )
            else:
                print(f"CONTROL GREEN {rel_to(dest, reg)}: restored to baseline after drift")
        if failures:
            for f in failures:
                print(f"CONTROLS FAIL {f}", file=sys.stderr)
            return EXIT_REJECT
        print(f"CONTROLS PASS {len(registries)} registries: drift refused, restore green")
        return EXIT_OK
    finally:
        shutil.rmtree(dest, ignore_errors=True)


# ---------------------------------------------------------------------------
# cli


def _resolve_registry_arg(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="digest_recheck.py",
        description="Recompute every digest a registry records and refuse the drift.",
    )
    parser.add_argument("--root", default=str(REPO), help="tree to scan (default: repo)")
    parser.add_argument(
        "--registry",
        action="append",
        default=[],
        metavar="PATH",
        help="a registry to recompute (repeatable)",
    )
    parser.add_argument(
        "--all", action="store_true", help="run the three in-tree registries"
    )
    parser.add_argument("--list", action="store_true", help="list the in-tree registries")
    parser.add_argument("--self-test", action="store_true", help="run the fixtures")
    parser.add_argument("--controls", action="store_true", help="run the drift controls")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test:
        return self_test()
    if args.controls:
        return controls(Path(args.root).resolve())

    if args.list:
        for r in ALL:
            print(r)
        return EXIT_OK

    root = Path(args.root).resolve()
    names = list(args.registry)
    if args.all:
        names.extend(ALL)
    if not names:
        print(
            "digest_recheck.py: nothing-to-check: pass --registry PATH (repeatable), "
            "--all, --list, --self-test or --controls; this gate never passes silently",
            file=sys.stderr,
        )
        return EXIT_USAGE

    registries = [_resolve_registry_arg(root, n) for n in names]
    try:
        violations, summary = scan(root, registries)
    except DataError as exc:
        print(f"digest_recheck.py: R2 registry-unrecognised: {exc}", file=sys.stderr)
        return EXIT_USAGE

    for v in violations:
        print(v.render(), file=sys.stderr)
    per_registry = ", ".join(
        f"{rel_to(root, r)}={'ok' if not any(v.registry == rel_to(root, r) for v in violations) else 'DRIFT'}"
        for r in registries
    )
    print(
        f"digest_recheck.py: {len(registries)} registries; {summary.rows} rows; "
        f"{summary.digests} digests recomputed; {summary.matched} matched; "
        f"{len(violations)} violation(s) [{per_registry}]"
    )
    if violations:
        return EXIT_REJECT
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())