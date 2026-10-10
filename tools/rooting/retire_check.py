#!/usr/bin/env python3
"""Rejecting gate for the binned Iota derivation route and for removal records.

The binned route is:

* every file under ``proof/corpus/Iota/Derivation/`` (Steps, Decode, Lexing,
  Grammar, Statements, Selection, Validation, Utilities, Composition, Audit);
* every file under ``proof/corpus/Iota/Generated/Trace/``;
* the ``Run``/``Derives`` acceptance in ``tools/iota/trace_check.py``;
* their Lake roots in ``proof/corpus/lakefile.lean`` and their registrations in
  ``tools/iota/zero_axioms.json``.

This runner fails while any part of that route is still present or still
referenced, so a reintroduced file or import cannot pass silently. ``--check``
scans the whole tree; ``--paths`` is the per-batch gate used as the binned files
are removed in bounded batches.

It also refuses an unregistered removal. ``tools/rooting/removals.json`` is the
removal registry; ``AGENTS.md`` section 10 (failure register, consolidation row)
requires that a removal name its replacement or the retained obligation,
inventory its affected consumers and carry reachability evidence, and that a
deletion record carry a discharge receipt rather than being a bare ``rm``. The
registry schema is documented in that file's ``note`` field; each record has
``id``, ``kind``, ``removed``, ``replacement``, ``retained_obligation``,
``consumers``, ``reachability`` (``{cwd, argv}``) and ``receipt``.

What is checked
===============
* ``R1`` binned path present: a file under either binned directory exists;
* ``R2`` Lean import: a ``.lean`` file under ``proof/`` imports
  ``Iota.Derivation.*`` or ``Iota.Generated.Trace.*``;
* ``R3`` Lake root: ``proof/corpus/lakefile.lean`` still lists a binned root;
* ``R4`` registry: ``tools/iota/zero_axioms.json`` still registers a binned
  module or source path;
* ``R5`` acceptance: ``tools/iota/trace_check.py`` still carries the
  ``DecodeTrace.Run``/``Derives`` acceptance (``PHASE_MODULES`` and the
  witness-type check);
* ``R6`` producer: ``tools/iota/export.py`` can resurrect the binned route
  (``Iota/Generated/Trace``, ``Iota.Generated.Trace``, ``Iota.Derivation``)
  or emits an unbounded recursion/heartbeat option (``maxHeartbeats 0``,
  ``maxRecDepth 1000000``);
* ``M1`` removal record names no replacement and no retained obligation;
* ``M2`` removal record's consumer inventory is empty, or a consumer path does
  not resolve under the repository root;
* ``M3`` removal record's reachability evidence names no command, or a command
  whose ``cwd`` is not a directory in the tree, whose ``argv[0]`` resolves
  neither as a path under that ``cwd`` nor on ``PATH``, or whose repo-relative
  file token does not resolve;
* ``M4`` removal record carries no receipt id (a bare deletion record), or a
  receipt that resolves to no ``docs/evidence/ledger.jsonl`` id and to no file
  in the tree;
* ``S`` surface: a reference surface the gate must read is missing.

Structure is strict and loud: a record that is not an object, a duplicate id, a
``kind`` outside ``migration``/``retirement``, or a wrongly typed field is a
data error (exit 2). Content is what ``M1``-``M4`` refuse (exit 1).

Trust boundary
==============
The gate reads the tree and the registry. It resolves the ``receipt`` field and
``argv[0]`` of a reachability command, but it runs no command, so a named
command is evidence of record, not evidence of its exit status; it does not
check that a removed path is now absent (``removed`` is recorded for the
receipt, and ``R1`` is what gates absence for the binned route); and it cannot
see a deletion that was never registered, because there is no record to refuse
and no baseline of the previous tree to diff. That last exclusion is the open
obligation: this gate reads the registry, not history. Resolution is existence,
not correctness: a receipt id is read out of ``docs/evidence/ledger.jsonl`` and
never checked for content, and ``PATH`` is the ambient one, so the recorded
program must be installed for ``--check`` to accept the record. No network, no
clock, no Lean, no Rust.

Isolation
=========
``--self-test`` and ``--controls`` materialise synthetic fixtures under the
system temporary directory and never read or write the repository tree; the
working tree is never mutated. The ``--controls`` removal mutation runs against
an in-memory copy of the real registry, so no file is written either. Both read
``PATH`` from the environment, because M3 resolves ``argv[0]``; a fixture
program is chosen to be absent from it. ``--paths`` and ``--check`` are
read-only.

Usage::

    python3 tools/rooting/retire_check.py --list
    python3 tools/rooting/retire_check.py --self-test
    python3 tools/rooting/retire_check.py --controls
    python3 tools/rooting/retire_check.py --check
    python3 tools/rooting/retire_check.py --paths proof/corpus/Iota/Derivation/Steps.lean

Exit status: 0 when the route is retired, every removal record is admissible
and every control passes; 1 when a binned path or an active reference remains,
a removal record is refused, or a fixture control survives; 2 on a usage or data
error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CASES_PATH = HERE.parent / "retire_cases.json"
REMOVALS_PATH = HERE.parent / "removals.json"

# The evidence ledger a receipt id resolves against, relative to the scanned root.
LEDGER_RELPATH = "docs/evidence/ledger.jsonl"

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

# The one owned path reintroduced by the --controls mutation.
CONTROL_MUTATION = "proof/corpus/Iota/Derivation/Steps.lean"

# The field the --controls removal mutation drops from the first real record.
CONTROL_RECEIPT_FIELD = "receipt"

# A repo-relative file token: no leading dash, no scheme, at least one separator
# or a bare name, and a file extension. Flags, package names and directory
# arguments carry no extension and are not resolved.
_FILE_TOKEN = re.compile(r"^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\.[A-Za-z0-9]+$")


class GateError(Exception):
    """The gate cannot reach its acceptance condition."""


class DataError(GateError):
    """The cases file is missing or malformed."""


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    line: int
    message: str

    def render(self) -> str:
        where = self.path if self.line <= 0 else f"{self.path}:{self.line}"
        return f"RULE {self.rule} {where}: {self.message}"


@dataclass(frozen=True)
class Surface:
    id: str
    rule: str
    title: str
    kind: str  # "file" | "glob"
    path: str
    root: str
    suffixes: tuple[str, ...]
    exclude_dirs: tuple[str, ...]
    tokens: tuple[str, ...]
    missing_is_violation: bool


@dataclass(frozen=True)
class Case:
    name: str
    rule: str
    expect: str
    action: str
    path: str
    text: str


@dataclass(frozen=True)
class Config:
    binned_prefixes: tuple[str, ...]
    binned_files: tuple[tuple[str, tuple[str, ...]], ...]
    surfaces: tuple[Surface, ...]
    rules: tuple[tuple[str, str, str], ...]
    clean_files: tuple[tuple[str, str], ...]
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class Command:
    cwd: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class Removal:
    id: str
    kind: str
    removed: tuple[str, ...]
    replacement: tuple[str, ...]
    retained_obligation: str
    consumers: tuple[str, ...]
    reachability: tuple[Command, ...]
    receipt: str


@dataclass(frozen=True)
class RemovalCase:
    name: str
    rule: str
    expect: str
    entry: str
    drop: tuple[str, ...]
    set_fields: tuple[tuple[str, Any], ...]
    action: str
    path: str
    text: str


@dataclass(frozen=True)
class RemovalConfig:
    rules: tuple[tuple[str, str, str], ...]
    clean_files: tuple[tuple[str, str], ...]
    entries: tuple[Removal, ...]
    cases: tuple[RemovalCase, ...]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def _string(value: object, where: str) -> str:
    _require(isinstance(value, str) and value.strip() != "", f"{where} must be a nonempty string")
    return value  # type: ignore[return-value]


def _string_list(value: object, where: str) -> list[str]:
    _require(isinstance(value, list) and value, f"{where} must be a nonempty list")
    return [_string(entry, f"{where}[]") for entry in value]


def _optional_string_list(value: object, where: str) -> list[str]:
    if value is None:
        return []
    _require(isinstance(value, list), f"{where} must be a list")
    return [_string(entry, f"{where}[]") for entry in value]


def load_config(path: Path = CASES_PATH) -> Config:
    if not path.is_file():
        raise DataError(f"cases file missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"cases file is not valid JSON: {exc}") from exc
    _require(isinstance(data, dict), "cases file must be an object")
    _require(data.get("schema") == 1, "cases schema must be 1")

    prefixes = tuple(_string_list(data.get("binned_prefixes"), "binned_prefixes"))
    files_raw = data.get("binned_files")
    _require(isinstance(files_raw, dict) and files_raw, "binned_files must be a nonempty object")
    binned_files: list[tuple[str, tuple[str, ...]]] = []
    for prefix, names in files_raw.items():
        _require(isinstance(prefix, str) and prefix, "binned_files key must be a nonempty string")
        binned_files.append((prefix, tuple(_string_list(names, f"binned_files[{prefix}]"))))

    surfaces_raw = data.get("reference_surfaces")
    _require(isinstance(surfaces_raw, list) and surfaces_raw, "reference_surfaces must be nonempty")
    surfaces: list[Surface] = []
    for index, raw in enumerate(surfaces_raw):
        where = f"reference_surfaces[{index}]"
        _require(isinstance(raw, dict), f"{where} must be an object")
        kind = _string(raw.get("kind"), f"{where}.kind")
        _require(kind in ("file", "glob"), f"{where}.kind must be file or glob")
        surfaces.append(
            Surface(
                id=_string(raw.get("id"), f"{where}.id"),
                rule=_string(raw.get("rule"), f"{where}.rule"),
                title=_string(raw.get("title"), f"{where}.title"),
                kind=kind,
                path=_string(raw.get("path"), f"{where}.path") if kind == "file" else "",
                root=_string(raw.get("root"), f"{where}.root") if kind == "glob" else "",
                suffixes=tuple(_optional_string_list(raw.get("suffixes"), f"{where}.suffixes")),
                exclude_dirs=tuple(
                    _optional_string_list(raw.get("exclude_dirs"), f"{where}.exclude_dirs")
                ),
                tokens=tuple(_string_list(raw.get("tokens"), f"{where}.tokens")),
                missing_is_violation=bool(raw.get("missing_is_violation", False)),
            )
        )

    rules_raw = data.get("rules")
    _require(isinstance(rules_raw, list) and rules_raw, "rules must be a nonempty list")
    rules: list[tuple[str, str, str]] = []
    for index, raw in enumerate(rules_raw):
        where = f"rules[{index}]"
        _require(isinstance(raw, dict), f"{where} must be an object")
        rules.append(
            (
                _string(raw.get("id"), f"{where}.id"),
                _string(raw.get("title"), f"{where}.title"),
                _string(raw.get("description"), f"{where}.description"),
            )
        )

    self_test = data.get("self_test")
    _require(isinstance(self_test, dict), "self_test must be an object")
    clean = self_test.get("clean")
    _require(isinstance(clean, dict), "self_test.clean must be an object")
    clean_raw = clean.get("files")
    _require(isinstance(clean_raw, dict) and clean_raw, "self_test.clean.files must be nonempty")
    clean_files: list[tuple[str, str]] = []
    for rel, text in clean_raw.items():
        _require(isinstance(rel, str) and rel, "self_test.clean.files key must be a path")
        clean_files.append((rel, _string(text, f"self_test.clean.files[{rel}]")))

    cases_raw = self_test.get("cases")
    _require(isinstance(cases_raw, list) and cases_raw, "self_test.cases must be nonempty")
    cases: list[Case] = []
    for index, raw in enumerate(cases_raw):
        where = f"self_test.cases[{index}]"
        _require(isinstance(raw, dict), f"{where} must be an object")
        expect = _string(raw.get("expect"), f"{where}.expect")
        _require(expect in ("reject", "accept"), f"{where}.expect must be reject or accept")
        action = _string(raw.get("action"), f"{where}.action")
        _require(action in ("write", "append", "delete", "none"), f"{where}.action is invalid")
        cases.append(
            Case(
                name=_string(raw.get("name"), f"{where}.name"),
                rule=str(raw.get("rule", "")),
                expect=expect,
                action=action,
                path=str(raw.get("path", "")),
                text=str(raw.get("text", "")),
            )
        )

    return Config(
        binned_prefixes=prefixes,
        binned_files=tuple(binned_files),
        surfaces=tuple(surfaces),
        rules=tuple(rules),
        clean_files=tuple(clean_files),
        cases=tuple(cases),
    )


# ---------------------------------------------------------------------------
# Removal records
# ---------------------------------------------------------------------------

def _optional_list(value: object, where: str) -> list[object]:
    """A list that may be absent or empty; type errors are still data errors."""
    if value is None:
        return []
    _require(isinstance(value, list), f"{where} must be a list")
    return list(value)


def _optional_text(value: object, where: str) -> str:
    """A string that may be absent or empty; a present value must be a string.

    Empty is content, not structure: a migration names no retained obligation and
    a bare deletion record carries no receipt id, so both reach M1 and M4.
    """
    if value is None:
        return ""
    _require(isinstance(value, str), f"{where} must be a string")
    return value  # type: ignore[return-value]


def _string_tuple(value: object, where: str) -> tuple[str, ...]:
    return tuple(_string(entry, f"{where}[]") for entry in _optional_list(value, where))


def _command(value: object, where: str) -> Command:
    _require(isinstance(value, dict), f"{where} must be an object")
    argv = _optional_list(value.get("argv"), f"{where}.argv")
    for index, token in enumerate(argv):
        _require(isinstance(token, str), f"{where}.argv[{index}] must be a string")
    return Command(cwd=_optional_text(value.get("cwd"), f"{where}.cwd"), argv=tuple(argv))


def removal_from_raw(raw: object, where: str) -> Removal:
    """One removal record. ``id`` and ``kind`` are structure; the rest is content.

    Content fields default when absent so that M1-M4 see the absence instead of
    a load error: the rules, not the loader, refuse an incomplete record.
    """
    _require(isinstance(raw, dict), f"{where} must be an object")
    kind = _string(raw.get("kind"), f"{where}.kind")
    _require(kind in ("migration", "retirement"), f"{where}.kind must be migration or retirement")
    reachability_raw = _optional_list(raw.get("reachability"), f"{where}.reachability")
    return Removal(
        id=_string(raw.get("id"), f"{where}.id"),
        kind=kind,
        removed=_string_tuple(raw.get("removed"), f"{where}.removed"),
        replacement=_string_tuple(raw.get("replacement"), f"{where}.replacement"),
        retained_obligation=_optional_text(
            raw.get("retained_obligation"), f"{where}.retained_obligation"
        ),
        consumers=_string_tuple(raw.get("consumers"), f"{where}.consumers"),
        reachability=tuple(
            _command(entry, f"{where}.reachability[{index}]")
            for index, entry in enumerate(reachability_raw)
        ),
        receipt=_optional_text(raw.get("receipt"), f"{where}.receipt"),
    )


def _unique(removals: Sequence[Removal]) -> None:
    seen: set[str] = set()
    for removal in removals:
        _require(removal.id not in seen, f"duplicate removal id: {removal.id}")
        seen.add(removal.id)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataError(f"file missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path.name} is not valid JSON: {exc}") from exc
    _require(isinstance(data, dict), f"{path.name} must be an object")
    return data


def _rule_triples(data: dict[str, Any], where: str) -> tuple[tuple[str, str, str], ...]:
    rules_raw = data.get("rules")
    _require(isinstance(rules_raw, list) and rules_raw, f"{where}.rules must be nonempty")
    rules: list[tuple[str, str, str]] = []
    for index, raw in enumerate(rules_raw):
        spot = f"{where}.rules[{index}]"
        _require(isinstance(raw, dict), f"{spot} must be an object")
        rules.append(
            (
                _string(raw.get("id"), f"{spot}.id"),
                _string(raw.get("title"), f"{spot}.title"),
                _string(raw.get("description"), f"{spot}.description"),
            )
        )
    return tuple(rules)


def load_removals(path: Path = REMOVALS_PATH) -> RemovalConfig:
    """The removal registry: rules, records and the synthetic self-test."""
    data = _read_json(path)
    _require(data.get("schema") == 1, "removals schema must be 1")

    entries_raw = data.get("removals")
    _require(isinstance(entries_raw, list), "removals must be a list")
    entries = tuple(
        removal_from_raw(raw, f"removals[{index}]") for index, raw in enumerate(entries_raw)
    )
    _unique(entries)

    self_test_raw = data.get("self_test")
    _require(isinstance(self_test_raw, dict), "self_test must be an object")
    clean_raw = self_test_raw.get("clean")
    _require(isinstance(clean_raw, dict), "self_test.clean must be an object")
    files_raw = clean_raw.get("files")
    _require(isinstance(files_raw, dict) and files_raw, "self_test.clean.files must be nonempty")
    clean_files: list[tuple[str, str]] = []
    for rel, text in files_raw.items():
        _require(isinstance(rel, str) and rel, "self_test.clean.files key must be a path")
        clean_files.append((rel, _string(text, f"self_test.clean.files[{rel}]")))

    entries_clean_raw = self_test_raw.get("entries")
    _require(
        isinstance(entries_clean_raw, list) and entries_clean_raw,
        "self_test.entries must be nonempty",
    )
    fixture = tuple(
        removal_from_raw(raw, f"self_test.entries[{index}]")
        for index, raw in enumerate(entries_clean_raw)
    )
    _unique(fixture)

    cases_raw = self_test_raw.get("cases")
    _require(isinstance(cases_raw, list) and cases_raw, "self_test.cases must be nonempty")
    cases: list[RemovalCase] = []
    for index, raw in enumerate(cases_raw):
        spot = f"self_test.cases[{index}]"
        _require(isinstance(raw, dict), f"{spot} must be an object")
        expect = _string(raw.get("expect"), f"{spot}.expect")
        _require(expect in ("reject", "accept"), f"{spot}.expect must be reject or accept")
        action = _string(raw.get("action"), f"{spot}.action")
        _require(action in ("write", "append", "delete", "none"), f"{spot}.action is invalid")
        set_raw = raw.get("set")
        _require(set_raw is None or isinstance(set_raw, dict), f"{spot}.set must be an object")
        set_fields = tuple(sorted((set_raw or {}).items()))
        cases.append(
            RemovalCase(
                name=_string(raw.get("name"), f"{spot}.name"),
                rule=str(raw.get("rule", "")),
                expect=expect,
                entry=str(raw.get("entry", "")),
                drop=_string_tuple(raw.get("drop"), f"{spot}.drop"),
                set_fields=set_fields,
                action=action,
                path=str(raw.get("path", "")),
                text=str(raw.get("text", "")),
            )
        )

    return RemovalConfig(
        rules=_rule_triples(data, "registry"),
        clean_files=tuple(clean_files),
        entries=fixture,
        cases=tuple(cases),
    )


def read_registry(path: Path = REMOVALS_PATH) -> tuple[tuple[Removal, ...], str]:
    """The real records and the source text, for an attributable line number."""
    data = _read_json(path)
    entries_raw = data.get("removals")
    _require(isinstance(entries_raw, list), "removals must be a list")
    entries = tuple(
        removal_from_raw(raw, f"removals[{index}]") for index, raw in enumerate(entries_raw)
    )
    _unique(entries)
    return entries, path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Route predicates and reference surfaces
# ---------------------------------------------------------------------------

def is_binned(cfg: Config, relative: str) -> bool:
    """True when a repo-relative path is part of the binned route."""
    for prefix in cfg.binned_prefixes:
        if relative == prefix.rstrip("/") or relative.startswith(prefix):
            return True
    for prefix, names in cfg.binned_files:
        for name in names:
            if relative == f"{prefix}{name}":
                return True
    return False


def module_name(relative: str) -> str | None:
    """The Lean module name of a repo-relative .lean path under proof/corpus."""
    if not relative.endswith(".lean") or not relative.startswith("proof/corpus/"):
        return None
    inner = relative[len("proof/corpus/") : -len(".lean")]
    if not inner:
        return None
    return inner.replace("/", ".")


def surface_files(cfg: Config, root: Path, surface: Surface) -> list[tuple[str, Path]]:
    """Every existing file a reference surface covers, as (relative, path)."""
    if surface.kind == "file":
        path = root / surface.path
        return [(surface.path, path)] if path.is_file() else []
    base = root / surface.root
    if not base.is_dir():
        return []
    excludes = set(surface.exclude_dirs)
    found: list[tuple[str, Path]] = []
    for current, dirs, names in os.walk(base):
        dirs[:] = [name for name in sorted(dirs) if name not in excludes]
        for name in sorted(names):
            if any(name.endswith(suffix) for suffix in surface.suffixes):
                path = Path(current) / name
                found.append((path.relative_to(root).as_posix(), path))
    return found


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def scan_tree(cfg: Config, root: Path) -> list[Violation]:
    """Every violation of the retirement condition in the tree at ``root``."""
    violations: list[Violation] = []

    for prefix in cfg.binned_prefixes:
        base = root / prefix
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    violations.append(
                        Violation("R1", relative, 0, f"binned path present under {prefix}")
                    )
        elif base.is_file():
            relative = base.relative_to(root).as_posix()
            violations.append(Violation("R1", relative, 0, f"binned path present ({prefix})"))

    for surface in cfg.surfaces:
        files = surface_files(cfg, root, surface)
        if not files:
            if surface.missing_is_violation:
                target = surface.path or surface.root
                violations.append(
                    Violation("S", target, 0, f"reference surface missing ({surface.id})")
                )
            continue
        for relative, path in files:
            lines = _read(path).splitlines()
            for token in surface.tokens:
                hits = [number for number, line in enumerate(lines, 1) if token in line]
                if not hits:
                    continue
                first = hits[0]
                more = "" if len(hits) == 1 else f" (+{len(hits) - 1} more line(s))"
                violations.append(
                    Violation(
                        surface.rule,
                        relative,
                        first,
                        f"active reference to the binned route "
                        f"({surface.id}: {token!r}){more}: {lines[first - 1].strip()[:120]}",
                    )
                )
    return violations


def find_references(cfg: Config, root: Path, relative: str) -> list[tuple[str, str, int]]:
    """Reference hits for one path: (surface_id, file, line) with the module or path."""
    needles = [relative]
    module = module_name(relative)
    if module is not None:
        needles.append(module)
    hits: list[tuple[str, str, int]] = []
    for surface in cfg.surfaces:
        for file_relative, path in surface_files(cfg, root, surface):
            for line_number, line in enumerate(_read(path).splitlines(), 1):
                if any(needle in line for needle in needles):
                    hits.append((surface.id, file_relative, line_number))
    return hits


# ---------------------------------------------------------------------------
# Removal rules
# ---------------------------------------------------------------------------

def resolve_under(root: Path, relative: str) -> Path | None:
    """The absolute path of a repo-relative entry, or None when it escapes root."""
    candidate = Path(relative)
    if candidate.is_absolute():
        return None
    try:
        base = root.resolve(strict=False)
        resolved = (base / candidate).resolve(strict=False)
    except OSError:
        return None
    if resolved == base or base in resolved.parents:
        return resolved
    return None


def entry_line(source: str, entry_id: str) -> int:
    """The 1-based line of the record's id in the registry text, else 0."""
    needle = f'"id": "{entry_id}"'
    for number, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return number
    return 0


def file_tokens(argv: Sequence[str]) -> list[str]:
    """Repo-relative file tokens in an argv; flags, schemes and bare names are not files.

    ``argv[0]`` is excluded: it is the program, resolved by ``resolve_program``.
    """
    return [
        token
        for index, token in enumerate(argv)
        if index
        and not token.startswith("-")
        and "://" not in token
        and _FILE_TOKEN.match(token)
    ]


def _ledger_ids(root: Path) -> frozenset[str] | None:
    """The receipt ids in the evidence ledger under ``root``, or None when absent.

    A ledger line that is not an object, or an object with no ``id``, is a data
    error: the id set cannot be established, so no receipt resolves against it
    and the gate says so instead of guessing. An absent ledger resolves no id,
    which leaves M4 to report the affected record rather than the whole run.
    """
    path = root / LEDGER_RELPATH
    if not path.is_file():
        return None
    ids: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataError(f"{LEDGER_RELPATH}:{number} is not valid JSON: {exc}") from exc
        _require(isinstance(row, dict), f"{LEDGER_RELPATH}:{number} must be an object")
        ident = row.get("id")
        _require(
            isinstance(ident, str) and ident.strip() != "",
            f"{LEDGER_RELPATH}:{number} must carry a nonempty id",
        )
        ids.add(ident)
    return frozenset(ids)


def receipt_resolves(root: Path, receipt: str, ledger: frozenset[str] | None) -> bool:
    """True when a receipt names an evidence-ledger row or a file in the tree."""
    if ledger is not None and receipt in ledger:
        return True
    resolved = resolve_under(root, receipt)
    return resolved is not None and resolved.is_file()


def resolve_program(base: Path, token: str) -> Path | None:
    """Where ``argv[0]`` resolves, or None: a path under the record's cwd, or PATH.

    A token carrying a separator is a path (absolute, or relative to the record's
    ``cwd``) and must be an existing file; a bare name is looked up on the
    ambient ``PATH``. A bare name that happens to exist in ``cwd`` is not
    resolved, so a record must name the installed program or the path to it.
    """
    if "/" in token or "://" in token:
        candidate = Path(token)
        if candidate.is_absolute():
            return candidate if candidate.is_file() else None
        resolved = resolve_under(base, token)
        return resolved if resolved is not None and resolved.is_file() else None
    return shutil.which(token)


def scan_removals(
    entries: Sequence[Removal], root: Path, source: str, text: str
) -> list[Violation]:
    """Every M1-M4 violation in the removal records, each naming its record."""
    violations: list[Violation] = []
    ledger = _ledger_ids(root)
    for removal in entries:
        line = entry_line(text, removal.id)
        where = source

        if not removal.replacement and not removal.retained_obligation:
            violations.append(
                Violation(
                    "M1",
                    where,
                    line,
                    f"removal record {removal.id!r} (kind={removal.kind}) names no replacement "
                    "and no retained obligation: it is neither a migration nor an authorized "
                    "retirement",
                )
            )

        if not removal.consumers:
            violations.append(
                Violation(
                    "M2",
                    where,
                    line,
                    f"removal record {removal.id!r} has an empty consumer inventory",
                )
            )
        else:
            for consumer in removal.consumers:
                resolved = resolve_under(root, consumer)
                if resolved is None or not resolved.exists():
                    violations.append(
                        Violation(
                            "M2",
                            where,
                            line,
                            f"removal record {removal.id!r} names a consumer that does not "
                            f"resolve: {consumer}",
                        )
                    )

        if not removal.reachability:
            violations.append(
                Violation(
                    "M3",
                    where,
                    line,
                    f"removal record {removal.id!r} names no reachability command",
                )
            )
        else:
            for index, command in enumerate(removal.reachability):
                if not command.argv:
                    violations.append(
                        Violation(
                            "M3",
                            where,
                            line,
                            f"removal record {removal.id!r} reachability[{index}] has an "
                            "empty argv",
                        )
                    )
                    continue
                base = (
                    resolve_under(root, command.cwd)
                    if command.cwd
                    else resolve_under(root, ".")
                )
                if base is None or not base.is_dir():
                    violations.append(
                        Violation(
                            "M3",
                            where,
                            line,
                            f"removal record {removal.id!r} reachability[{index}] cwd is not a "
                            f"directory in the tree: {command.cwd or '.'}",
                        )
                    )
                    continue
                program = command.argv[0]
                if resolve_program(base, program) is None:
                    shown = " ".join(command.argv)
                    violations.append(
                        Violation(
                            "M3",
                            where,
                            line,
                            f"removal record {removal.id!r} reachability[{index}] names a "
                            f"program that resolves neither as a path under its cwd nor on "
                            f"PATH: {program} (command: {shown})",
                        )
                    )
                for token in file_tokens(command.argv):
                    resolved = resolve_under(base, token)
                    if resolved is None or not resolved.exists():
                        shown = " ".join(command.argv)
                        violations.append(
                            Violation(
                                "M3",
                                where,
                                line,
                                f"removal record {removal.id!r} reachability[{index}] names a "
                                f"path that does not resolve: {token} (command: {shown})",
                            )
                        )

        if not removal.receipt.strip():
            violations.append(
                Violation(
                    "M4",
                    where,
                    line,
                    f"bare deletion record {removal.id!r} carries no receipt id",
                )
            )
        elif not receipt_resolves(root, removal.receipt.strip(), ledger):
            violations.append(
                Violation(
                    "M4",
                    where,
                    line,
                    f"removal record {removal.id!r} names a receipt that resolves to no "
                    f"{LEDGER_RELPATH} id and to no file in the tree: "
                    f"{removal.receipt.strip()}",
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _materialize(root: Path, files: tuple[tuple[str, str], ...]) -> None:
    for relative, text in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _apply_case(root: Path, case: Case) -> None:
    if case.action == "none":
        return
    path = root / case.path
    if case.action == "write":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(case.text, encoding="utf-8")
    elif case.action == "append":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(case.text)
    elif case.action == "delete":
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# --self-test and --controls
# ---------------------------------------------------------------------------

def self_test(cfg: Config) -> int:
    failures: list[str] = []
    base = Path(tempfile.mkdtemp(prefix="retire-check-clean-"))
    try:
        _materialize(base, cfg.clean_files)
        clean = scan_tree(cfg, base)
        if clean:
            print(f"SELF-TEST FAIL clean fixture: {clean[0].render()}")
            return EXIT_REJECT
        print("SELF-TEST GREEN clean fixture")

        for case in cfg.cases:
            if case.action == "none":
                continue
            work = Path(tempfile.mkdtemp(prefix="retire-check-case-"))
            try:
                shutil.copytree(base, work, dirs_exist_ok=True)
                _apply_case(work, case)
                violations = scan_tree(cfg, work)
                rules = sorted({violation.rule for violation in violations})
                if case.expect == "reject":
                    if not violations:
                        failures.append(f"{case.name}: survived, expected rule {case.rule}")
                        print(f"SELF-TEST FAIL {case.name}: survived")
                    elif case.rule and case.rule not in rules:
                        failures.append(f"{case.name}: rejected by {rules}, expected {case.rule}")
                        print(f"SELF-TEST FAIL {case.name}: rejected by {rules}, expected {case.rule}")
                    else:
                        print(f"SELF-TEST RED  {case.name}: {violations[0].render()}")
                elif violations:
                    failures.append(f"{case.name}: rejected, expected acceptance")
                    print(f"SELF-TEST FAIL {case.name}: {violations[0].render()}")
                else:
                    print(f"SELF-TEST GREEN {case.name}")
            finally:
                shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if failures:
        print(f"SELF-TEST FAIL: {len(failures)} case(s) failed")
        return EXIT_REJECT
    reject_count = sum(1 for case in cfg.cases if case.expect == "reject")
    print(
        f"SELF-TEST PASS: clean fixture accepted, {reject_count} reintroduction "
        "fixture(s) rejected"
    )
    return EXIT_OK


def _apply_removal_case(
    cfg: RemovalConfig, case: RemovalCase
) -> tuple[Removal, ...] | None:
    """The fixture records with one record seeded: fields dropped, then fields set."""
    if not case.entry:
        return cfg.entries
    records: list[Removal] = []
    found = False
    for removal in cfg.entries:
        if removal.id != case.entry:
            records.append(removal)
            continue
        found = True
        raw: dict[str, Any] = {
            "id": removal.id,
            "kind": removal.kind,
            "removed": list(removal.removed),
            "replacement": list(removal.replacement),
            "retained_obligation": removal.retained_obligation,
            "consumers": list(removal.consumers),
            "reachability": [{"cwd": c.cwd, "argv": list(c.argv)} for c in removal.reachability],
            "receipt": removal.receipt,
        }
        for field in case.drop:
            raw.pop(field, None)
        for field, value in case.set_fields:
            raw[field] = value
        records.append(removal_from_raw(raw, f"self_test.cases[{case.name}].set[{field}]"))
    if not found:
        raise DataError(f"case {case.name!r} names an unknown fixture record: {case.entry}")
    return tuple(records)


def removal_self_test(rem: RemovalConfig) -> int:
    """Materialise the synthetic tree and refuse every seeded removal defect."""
    failures: list[str] = []
    names: list[str] = []
    base = Path(tempfile.mkdtemp(prefix="retire-removal-clean-"))
    try:
        _materialize(base, rem.clean_files)
        source = "removals.json#self_test"
        clean = scan_removals(rem.entries, base, source, _registry_text(rem.entries))
        if clean:
            print(f"SELF-TEST FAIL clean removal registry: {clean[0].render()}")
            return EXIT_REJECT
        print("SELF-TEST GREEN clean removal registry")

        for case in rem.cases:
            try:
                entries = _apply_removal_case(rem, case)
            except GateError as exc:
                failures.append(f"{case.name}: {exc}")
                print(f"SELF-TEST FAIL {case.name}: {exc}")
                continue
            root = base
            work: Path | None = None
            if case.action != "none":
                work = Path(tempfile.mkdtemp(prefix="retire-removal-case-"))
                shutil.copytree(base, work, dirs_exist_ok=True)
                _apply_case(work, Case("", "", "none", case.action, case.path, case.text))
                root = work
            records = entries or ()
            try:
                violations = scan_removals(records, root, source, _registry_text(records))
                rules = sorted({violation.rule for violation in violations})
                if case.expect == "reject":
                    if not violations:
                        failures.append(f"{case.name}: survived, expected rule {case.rule}")
                        print(f"SELF-TEST FAIL {case.name}: survived")
                    elif case.rule and case.rule not in rules:
                        failures.append(
                            f"{case.name}: rejected by {rules}, expected {case.rule}"
                        )
                        print(
                            f"SELF-TEST FAIL {case.name}: rejected by {rules}, "
                            f"expected {case.rule}"
                        )
                    else:
                        print(f"SELF-TEST RED  {case.name}: {violations[0].render()}")
                        names.append(f"{case.rule} {case.name}")
                elif violations:
                    failures.append(f"{case.name}: rejected, expected acceptance")
                    print(f"SELF-TEST FAIL {case.name}: {violations[0].render()}")
                else:
                    print(f"SELF-TEST GREEN {case.name}")
            finally:
                if work is not None:
                    shutil.rmtree(work, ignore_errors=True)

        if failures:
            print(f"SELF-TEST FAIL: {len(failures)} removal case(s) failed")
            return EXIT_REJECT
    finally:
        shutil.rmtree(base, ignore_errors=True)

    reject_count = sum(1 for case in rem.cases if case.expect == "reject")
    accept_count = sum(1 for case in rem.cases if case.expect == "accept")
    print(f"SELF-TEST PASS: {reject_count} removal record fixture(s) rejected: {', '.join(names)}")
    print(f"SELF-TEST PASS: {accept_count} removal registry fixture(s) accepted")
    return EXIT_OK


def _registry_text(entries: Sequence[Removal]) -> str:
    return json.dumps(
        [
            {
                "id": removal.id,
                "kind": removal.kind,
                "removed": list(removal.removed),
                "replacement": list(removal.replacement),
                "retained_obligation": removal.retained_obligation,
                "consumers": list(removal.consumers),
                "reachability": [
                    {"cwd": command.cwd, "argv": list(command.argv)}
                    for command in removal.reachability
                ],
                "receipt": removal.receipt,
            }
            for removal in entries
        ],
        indent=2,
    )


def controls(cfg: Config, rem: RemovalConfig) -> int:
    base = Path(tempfile.mkdtemp(prefix="retire-check-base-"))
    work = Path(tempfile.mkdtemp(prefix="retire-check-mutant-"))
    try:
        _materialize(base, cfg.clean_files)
        if scan_tree(cfg, base):
            print(f"CONTROL baseline: RED :: {scan_tree(cfg, base)[0].render()}", file=sys.stderr)
            return EXIT_REJECT
        print("CONTROL baseline: GREEN")

        shutil.copytree(base, work, dirs_exist_ok=True)
        mutation = work / CONTROL_MUTATION
        mutation.parent.mkdir(parents=True, exist_ok=True)
        mutation.write_text("import Iota.Syntax\n", encoding="utf-8")
        mutated = scan_tree(cfg, work)
        if not any(violation.rule == "R1" for violation in mutated):
            print("CONTROL mutation: SURVIVED (R1 not raised)", file=sys.stderr)
            return EXIT_REJECT
        print(f"CONTROL mutation: RED :: {mutated[0].render()}")

        mutation.unlink(missing_ok=True)
        restored = scan_tree(cfg, work)
        if restored:
            print(f"CONTROL restored: RED :: {restored[0].render()}", file=sys.stderr)
            return EXIT_REJECT
        print("CONTROL restored: GREEN")
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)

    # Removal mutation: the real registry, in memory, with the receipt dropped from
    # its first record. A bare deletion record must be refused; no file is written.
    try:
        entries, text = read_registry()
        if not entries:
            print("CONTROL removal baseline: no record in the registry to mutate", file=sys.stderr)
            return EXIT_REJECT
        first = entries[0]
        bare = Removal(
            id=first.id,
            kind=first.kind,
            removed=first.removed,
            replacement=first.replacement,
            retained_obligation=first.retained_obligation,
            consumers=first.consumers,
            reachability=first.reachability,
            receipt="",
        )
        mutated_records = (bare,) + entries[1:]
        mutated = scan_removals(mutated_records, REPO, "tools/rooting/removals.json", text)
        if not any(violation.rule == "M4" for violation in mutated):
            print(
                "CONTROL removal mutation: SURVIVED (M4 not raised)", file=sys.stderr
            )
            return EXIT_REJECT
        print(f"CONTROL removal mutation: RED :: {mutated[0].render()}")
        if scan_removals(entries, REPO, "tools/rooting/removals.json", text):
            print(
                "CONTROL removal restored: RED :: the real registry is not clean",
                file=sys.stderr,
            )
            return EXIT_REJECT
        print("CONTROL removal restored: GREEN")
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print("CONTROLS PASS: baseline green, mutation red, restored green")
    return EXIT_OK


# ---------------------------------------------------------------------------
# --paths and --check
# ---------------------------------------------------------------------------

def normalize_path(raw: str) -> str:
    relative = raw.strip().replace("\\", "/")
    candidate = Path(relative)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(REPO).as_posix()
        except ValueError:
            return relative
    while relative.startswith("./"):
        relative = relative[2:]
    return relative


def check_paths(cfg: Config, root: Path, raw_paths: list[str]) -> int:
    violations = 0
    for raw in raw_paths:
        relative = normalize_path(raw)
        binned = is_binned(cfg, relative)
        present = (root / relative).exists()
        references = find_references(cfg, root, relative)
        print(f"PATH {relative}")
        print(f"  binned: {'yes' if binned else 'no'}")
        print(f"  present: {'yes' if present else 'no'}")
        if references:
            shown = ", ".join(f"{surface} {path}:{line}" for surface, path, line in references[:5])
            more = "" if len(references) <= 5 else f" (+{len(references) - 5} more)"
            print(f"  referenced: yes ({shown}{more})")
        else:
            print("  referenced: no")
        present_violation = binned and present
        reference_violation = bool(references)
        if present_violation or reference_violation:
            violations += 1
            reasons = []
            if present_violation:
                reasons.append("binned path present")
            if reference_violation:
                reasons.append("active reference(s) remain")
            print(f"  VIOLATION: {'; '.join(reasons)}")
        elif binned:
            print("  RETIRED: absent and unreferenced")
        else:
            print("  not part of the binned route (ignored)")
    if violations:
        print(f"PATHS FAIL: {violations} of {len(raw_paths)} path(s) still active")
        return EXIT_REJECT
    print(f"PATHS CLEAN: {len(raw_paths)} path(s) retired")
    return EXIT_OK


def check(cfg: Config, rem: RemovalConfig, root: Path) -> int:
    violations = scan_tree(cfg, root)
    entries, text = read_registry()
    violations = violations + scan_removals(entries, root, "tools/rooting/removals.json", text)
    if not violations:
        print(
            "CHECK CLEAN: the binned Iota derivation route is retired and every "
            f"removal record in tools/rooting/removals.json is admissible "
            f"({len(entries)} record(s))"
        )
        return EXIT_OK
    for violation in violations:
        print(violation.render())
    by_rule: dict[str, int] = {}
    for violation in violations:
        by_rule[violation.rule] = by_rule.get(violation.rule, 0) + 1
    summary = ", ".join(f"{rule}={count}" for rule, count in sorted(by_rule.items()))
    print(f"CHECK FAIL: {len(violations)} violation(s) ({summary})")
    return EXIT_REJECT


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(cfg: Config, rem: RemovalConfig) -> int:
    print("Binned route (must be absent):")
    for prefix in cfg.binned_prefixes:
        names = dict(cfg.binned_files).get(prefix, ())
        print(f"  {prefix}  ({len(names)} known file(s))")
        for name in names:
            print(f"    {prefix}{name}")
    print("Checked reference forms (must not reference the route):")
    for surface in cfg.surfaces:
        target = surface.path if surface.kind == "file" else f"{surface.root}/**"
        missing = "missing=violation" if surface.missing_is_violation else "missing=retired-ok"
        print(f"  {surface.rule} {surface.id}: {target} [{missing}]")
        print(f"    tokens: {list(surface.tokens)}")
    print("Rules:")
    for rule_id, title, description in cfg.rules:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    entries, _ = read_registry()
    print(f"Removal records (tools/rooting/removals.json, {len(entries)} record(s)):")
    for removal in entries:
        disposition = (
            f"migration -> {', '.join(removal.replacement)}"
            if removal.replacement
            else "retirement, retained obligation recorded"
        )
        commands = "; ".join(
            f"cd {command.cwd or '.'} && {' '.join(command.argv)}"
            for command in removal.reachability
        )
        print(f"  {removal.id} [{removal.kind}]: {disposition}")
        print(f"    removed: {', '.join(removal.removed) or '(none)'}")
        print(f"    consumers: {', '.join(removal.consumers)}")
        print(f"    reachability: {commands}")
        print(f"    receipt: {removal.receipt}")
    print("Removal rules:")
    for rule_id, title, description in rem.rules:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="retire_check.py",
        description=(
            "Rejecting gate for the binned Iota derivation route (R1-R6) and for "
            "removal records (M1-M4)."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list the binned route and reference forms")
    mode.add_argument("--self-test", action="store_true", help="exercise the rejection fixtures")
    mode.add_argument("--controls", action="store_true", help="baseline/mutation/restore control")
    mode.add_argument("--check", action="store_true", help="scan the repository tree")
    mode.add_argument("--paths", nargs="+", metavar="PATH", help="per-path retirement check")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        rem = load_removals()
        if args.list:
            return cmd_list(cfg, rem)
        if args.self_test:
            status = self_test(cfg)
            if status != EXIT_OK:
                return status
            return removal_self_test(rem)
        if args.controls:
            return controls(cfg, rem)
        if args.paths:
            return check_paths(cfg, REPO, args.paths)
        return check(cfg, rem, REPO)
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
