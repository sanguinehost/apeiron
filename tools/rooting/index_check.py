#!/usr/bin/env python3
"""Rejecting gate for the evidence index: ``docs/evidence/ledger.jsonl``.

The index sits beside the files it indexes: one JSON object per line in
``docs/evidence/ledger.jsonl``, each naming an artefact under ``docs/evidence/``
and recording its kind, byte size and sha256. ``docs/plan.md`` names the gate
statement for E2.W1 as *every receipt has exactly one index row*, and before this
gate the landed index had no enforcer in the tree: 66 rows shipped with a
duplicate id and a stale digest, and nothing refused them. ``tools/rooting/
retire_check.py`` reads ``ledger.jsonl`` only to resolve a removal record's
receipt id; it does not hold the index to coverage.

What is checked
===============
* ``S`` surface: the evidence directory or the ledger itself is absent;
* ``E1`` unindexed file: a regular file under the evidence root, other than the
  ledger, has no row;
* ``E2`` duplicate row: one file is indexed by more than one row;
* ``E3`` dangling row: a row's path resolves to no file under the scanned root,
  or escapes it;
* ``E4`` duplicate id: two rows share an id;
* ``E5`` stale digest: a row's recorded ``bytes``/``sha256`` disagrees with the
  file on disk;
* ``E6`` row content out of domain: a row omits a required key, or its ``kind``,
  ``bytes``, ``sha256`` or ``summary`` is not the value the schema admits.

Structure is strict and loud: a line that is not JSON, a row that is not an
object, and a row whose ``id`` or ``path`` is absent or not a nonempty string are
data errors (exit 2), because the id set and the coverage map cannot be
established without them. Everything else is content, which ``E1``-``E6`` refuse
(exit 1). The schema, the ``kind`` domain and the fixtures live in
``tools/rooting/index_cases.json`` beside this file, as in ``retire_cases.json``.

Trust boundary
==============
The gate enumerates the evidence tree, reads the ledger and hashes the files it
names; it reads nothing else, and it writes nothing under ``--check``. It is
therefore evidence that the index covers the files that are present, not that the
artefacts the index names are correct, current or worth indexing: an index row
makes a claim, and this gate checks the claim against the bytes on disk only. It
does not check that a row's file is the receipt a document cites, that its
``summary`` describes it, that no row is redundant with another summary, or that
a file outside ``docs/evidence/`` is indexed at all; rows naming a file elsewhere
in the tree are accepted when the file exists and the digest matches, and the
``--check`` summary counts them as ``outside``. Symlinks are followed, so a row
that resolves through one is checked against its target. There is no baseline of
the previous tree, so a file that was deleted *and* had its row deleted in the
same change leaves no trace here. No network, no clock, no Lean, no Rust.

Isolation
=========
``--self-test`` materialises a synthetic tree under the system temporary
directory and never reads or writes the repository. ``--controls`` copies the
real ``docs/evidence/`` subtree into the system temporary directory, runs
baseline / mutation / restore there, and reads the working tree read-only; the
working tree is never mutated. The system temporary directory honours ``TMPDIR``
— set ``TMPDIR=/home/socol/oc-tmp`` to keep the copies off the ``/tmp`` tmpfs,
which has run out of space mid-campaign. No fixture depends on a path that
exists in the real tree, and every copy is removed on exit.

Usage::

    python3 tools/rooting/index_check.py --list
    python3 tools/rooting/index_check.py --self-test
    python3 tools/rooting/index_check.py --controls
    python3 tools/rooting/index_check.py --check

Every mode accepts ``--root PATH`` to scan another tree, which is what the
controls use.

Exit status: 0 when every file under the evidence root has exactly one row, no
row is dangling, no id is duplicated and every recorded digest matches the file;
1 when a rule fires or a fixture control survives; 2 on a usage or data error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CASES_PATH = HERE.parent / "index_cases.json"

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

# The sentinel a fixture ledger row uses for a digest the loader must compute.
AUTO = "auto"

# The one file --controls adds, to seed E1 without touching an indexed artefact.
CONTROL_UNINDEXED = "docs/evidence/index-check-control-unindexed.md"
CONTROL_TEXT = "Seeded by tools/rooting/index_check.py --controls; not an evidence receipt.\n"

# A digest no file has, for the stale-sha256 control.
CONTROL_ZERO_SHA = "0" * 64


class GateError(Exception):
    """The gate cannot reach its acceptance condition."""


class DataError(GateError):
    """The cases file or the ledger is malformed."""


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
class FileOp:
    action: str  # write | append | delete | delete_tree
    path: str
    text: str


@dataclass(frozen=True)
class RowOp:
    op: str  # duplicate | set | drop
    line: int
    field: str
    value: Any


@dataclass(frozen=True)
class Case:
    name: str
    rule: str
    expect: str  # reject | accept | error
    files: tuple[FileOp, ...]
    rows: tuple[RowOp, ...]


@dataclass(frozen=True)
class Config:
    evidence_root: str
    ledger_relpath: str
    kinds: tuple[str, ...]
    required: tuple[str, ...]
    rules: tuple[tuple[str, str, str], ...]
    clean_files: tuple[tuple[str, str], ...]
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class Summary:
    rows: int
    files: int
    kinds: tuple[tuple[str, int], ...]
    with_digest: int
    outside: int
    rules: tuple[str, ...]

    def render(self) -> str:
        kinds = ",".join(f"{kind}={count}" for kind, count in self.kinds) or "none"
        return (
            f"  rows={self.rows} files={self.files} kinds={kinds} "
            f"digests={self.with_digest}/{self.rows} outside={self.outside} "
            f"violations={','.join(self.rules) or 'none'}"
        )


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


def _optional_list(value: object, where: str) -> list[Any]:
    if value is None:
        return []
    _require(isinstance(value, list), f"{where} must be a list")
    return list(value)


def load_config(path: Path = CASES_PATH) -> Config:
    """The index layout, the row schema, the rules and the synthetic fixtures."""
    if not path.is_file():
        raise DataError(f"cases file missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"cases file is not valid JSON: {exc}") from exc
    _require(isinstance(data, dict), "cases file must be an object")
    _require(data.get("schema") == 1, "cases schema must be 1")

    evidence_root = _string(data.get("evidence_root"), "evidence_root").rstrip("/")
    _require(not evidence_root.startswith("/"), "evidence_root must be repo-relative")
    ledger_relpath = _string(data.get("ledger_relpath"), "ledger_relpath")
    _require(
        ledger_relpath.startswith(f"{evidence_root}/"),
        f"ledger_relpath must sit under evidence_root ({evidence_root})",
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
        _require(expect in ("reject", "accept", "error"), f"{where}.expect must be reject/accept/error")
        files: list[FileOp] = []
        for position, entry in enumerate(_optional_list(raw.get("files"), f"{where}.files")):
            spot = f"{where}.files[{position}]"
            _require(isinstance(entry, dict), f"{spot} must be an object")
            action = _string(entry.get("action"), f"{spot}.action")
            _require(
                action in ("write", "append", "delete", "delete_tree"),
                f"{spot}.action must be write, append, delete or delete_tree",
            )
            files.append(
                FileOp(
                    action=action,
                    path=_string(entry.get("path"), f"{spot}.path"),
                    text=entry.get("text") if isinstance(entry.get("text"), str) else "",
                )
            )
        rows: list[RowOp] = []
        for position, entry in enumerate(_optional_list(raw.get("rows"), f"{where}.rows")):
            spot = f"{where}.rows[{position}]"
            _require(isinstance(entry, dict), f"{spot} must be an object")
            op = _string(entry.get("op"), f"{spot}.op")
            _require(op in ("duplicate", "set", "drop"), f"{spot}.op must be duplicate, set or drop")
            line = entry.get("line")
            _require(isinstance(line, int) and line >= 1, f"{spot}.line must be a positive integer")
            field = entry.get("field", "")
            if op in ("set", "drop"):
                _string(field, f"{spot}.field")
            rows.append(
                RowOp(op=op, line=line, field=field if isinstance(field, str) else "", value=entry.get("value"))
            )
        cases.append(
            Case(
                name=_string(raw.get("name"), f"{where}.name"),
                rule=str(raw.get("rule", "")),
                expect=expect,
                files=tuple(files),
                rows=tuple(rows),
            )
        )

    return Config(
        evidence_root=evidence_root,
        ledger_relpath=ledger_relpath,
        kinds=tuple(_string_list(data.get("kinds"), "kinds")),
        required=tuple(_string_list(data.get("required"), "required")),
        rules=tuple(rules),
        clean_files=tuple(clean_files),
        cases=tuple(cases),
    )


# ---------------------------------------------------------------------------
# The ledger
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
    if resolved != base and base in resolved.parents:
        return resolved
    return None


def parse_rows(text: str, where: str) -> list[tuple[int, dict[str, Any]]]:
    """Every ledger row as ``(line, object)``, in file order.

    ``id`` and ``path`` are the two keys a row cannot be without: the id set and
    the coverage map are both derived from them, so their absence is structural
    (a data error) rather than a rule finding.
    """
    rows: list[tuple[int, dict[str, Any]]] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataError(f"{where}:{number} is not valid JSON: {exc}") from exc
        _require(isinstance(row, dict), f"{where}:{number} must be an object")
        for key in ("id", "path"):
            value = row.get(key)
            _require(
                isinstance(value, str) and value.strip() != "",
                f"{where}:{number} must carry a nonempty {key} (got {value!r})",
            )
        rows.append((number, row))
    return rows


def evidence_files(cfg: Config, root: Path) -> list[str]:
    """Every regular file under the evidence root, repo-relative, sorted.

    The ledger is the index, not an indexed artefact, so it is excluded.
    """
    base = root / cfg.evidence_root
    if not base.is_dir():
        return []
    found: list[str] = []
    for current, dirs, names in os.walk(base):
        dirs[:] = [name for name in sorted(dirs) if name != "__pycache__"]
        for name in sorted(names):
            relative = (Path(current) / name).relative_to(root).as_posix()
            if relative == cfg.ledger_relpath:
                continue
            found.append(relative)
    return sorted(found)


def _digest(path: Path) -> tuple[int, str]:
    blob = path.read_bytes()
    return len(blob), hashlib.sha256(blob).hexdigest()


def scan(cfg: Config, root: Path) -> tuple[list[Violation], Summary]:
    """Every violation of the index condition in the tree at ``root``."""
    violations: list[Violation] = []
    ledger_rel = cfg.ledger_relpath
    ledger = root / ledger_rel
    base = root / cfg.evidence_root

    if not base.is_dir():
        violations.append(
            Violation(
                "S",
                cfg.evidence_root,
                0,
                f"evidence directory absent: {cfg.evidence_root} (the index has nothing to sit "
                "beside)",
            )
        )
        return violations, _empty_summary()
    if not ledger.is_file():
        violations.append(
            Violation("S", ledger_rel, 0, f"index absent: {ledger_rel}")
        )
        return violations, _empty_summary()

    rows = parse_rows(ledger.read_text(encoding="utf-8"), ledger_rel)
    seen_ids: dict[str, int] = {}
    by_path: dict[str, list[int]] = {}
    kind_counts: dict[str, int] = {}
    with_digest = 0
    outside = 0

    for line, row in rows:
        ident = str(row["id"]).strip()
        relative = str(row["path"]).strip()

        missing = [key for key in cfg.required if key not in row]
        if missing:
            violations.append(
                Violation(
                    "E6",
                    ledger_rel,
                    line,
                    f"row {line} carries no {', '.join(missing)}: a row must record every key "
                    f"of the schema ({', '.join(cfg.required)})",
                )
            )

        kind = row.get("kind")
        if "kind" in row:
            if isinstance(kind, str):
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            if kind not in cfg.kinds:
                shown = kind if isinstance(kind, str) else repr(kind)
                violations.append(
                    Violation(
                        "E6",
                        ledger_rel,
                        line,
                        f"row {line} records kind {shown!r}, outside "
                        f"{{{', '.join(cfg.kinds)}}}",
                    )
                )

        size = row.get("bytes")
        size_ok = isinstance(size, int) and not isinstance(size, bool) and size >= 0
        if "bytes" in row and not size_ok:
            violations.append(
                Violation(
                    "E6",
                    ledger_rel,
                    line,
                    f"row {line} records bytes {size!r}, which is not a size in "
                    "nonnegative integers",
                )
            )

        digest = row.get("sha256")
        digest_ok = isinstance(digest, str) and len(digest) == 64
        if "sha256" in row and not digest_ok:
            shown = digest if isinstance(digest, str) else repr(digest)
            violations.append(
                Violation(
                    "E6",
                    ledger_rel,
                    line,
                    f"row {line} records sha256 {shown!r}, which is not 64 hex characters",
                )
            )
        if size_ok and digest_ok:
            with_digest += 1

        summary = row.get("summary")
        if "summary" in row and not (isinstance(summary, str) and summary.strip()):
            violations.append(
                Violation(
                    "E6",
                    ledger_rel,
                    line,
                    f"row {line} carries an empty summary: the index row states what the "
                    "artefact is",
                )
            )

        if ident in seen_ids:
            violations.append(
                Violation(
                    "E4",
                    ledger_rel,
                    line,
                    f"duplicate id {ident!r}: rows {seen_ids[ident]} and {line} (an id lookup "
                    "in the index is ambiguous; removal records resolve receipts by id)",
                )
            )
        else:
            seen_ids[ident] = line
        by_path.setdefault(relative, []).append(line)

        resolved = resolve_under(root, relative)
        if resolved is None:
            violations.append(
                Violation(
                    "E3",
                    ledger_rel,
                    line,
                    f"row {line} names a path that resolves outside the scanned root: "
                    f"{relative}",
                )
            )
            continue
        if not relative.startswith(f"{cfg.evidence_root}/"):
            outside += 1
        if not resolved.is_file():
            violations.append(
                Violation(
                    "E3",
                    ledger_rel,
                    line,
                    f"dangling row {line}: {relative} does not exist under the scanned root",
                )
            )
            continue
        actual_bytes, actual_sha = _digest(resolved)
        if size_ok and actual_bytes != size:
            violations.append(
                Violation(
                    "E5",
                    ledger_rel,
                    line,
                    f"stale bytes on {relative}: row {line} records {size}, the file is "
                    f"{actual_bytes}",
                )
            )
        if digest_ok and actual_sha != digest:
            violations.append(
                Violation(
                    "E5",
                    ledger_rel,
                    line,
                    f"stale sha256 on {relative}: row {line} records {digest}, the file is "
                    f"{actual_sha}",
                )
            )

    for relative in evidence_files(cfg, root):
        lines = by_path.get(relative, [])
        if not lines:
            violations.append(
                Violation(
                    "E1",
                    relative,
                    0,
                    f"unindexed: {relative} has no row in {ledger_rel}",
                )
            )
        elif len(lines) > 1:
            shown = ", ".join(str(number) for number in lines)
            violations.append(
                Violation(
                    "E2",
                    relative,
                    0,
                    f"duplicate row: {relative} is indexed by rows {shown} of {ledger_rel}, so "
                    "its recorded size and digest are ambiguous",
                )
            )

    summary = Summary(
        rows=len(rows),
        files=len(evidence_files(cfg, root)),
        kinds=tuple(sorted(kind_counts.items())),
        with_digest=with_digest,
        outside=outside,
        rules=tuple(sorted({violation.rule for violation in violations})),
    )
    return violations, summary


def _empty_summary() -> Summary:
    return Summary(0, 0, (), 0, 0, ())


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _materialize(cfg: Config, root: Path) -> None:
    """Write the clean fixture, resolving each ledger row's ``auto`` digest."""
    ledger_text = ""
    for relative, text in cfg.clean_files:
        if relative == cfg.ledger_relpath:
            ledger_text = text
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    _require(ledger_text != "", f"the clean fixture carries no {cfg.ledger_relpath}")
    resolved: list[str] = []
    for number, line in enumerate(ledger_text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bytes") == AUTO or row.get("sha256") == AUTO:
            target = resolve_under(root, str(row.get("path", "")))
            _require(
                target is not None and target.is_file(),
                f"fixture row {number} names a missing file: {row.get('path')}",
            )
            size, sha = _digest(target)  # type: ignore[arg-type]
            if row.get("bytes") == AUTO:
                row["bytes"] = size
            if row.get("sha256") == AUTO:
                row["sha256"] = sha
        resolved.append(json.dumps(row, sort_keys=True))
    ledger = root / cfg.ledger_relpath
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("".join(f"{line}\n" for line in resolved), encoding="utf-8")


def _apply_file_op(root: Path, op: FileOp) -> None:
    path = root / op.path
    if op.action == "delete_tree":
        shutil.rmtree(path, ignore_errors=True)
        return
    if op.action == "delete":
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if op.action == "append":
        with path.open("a", encoding="utf-8") as handle:
            handle.write(op.text)
        return
    path.write_text(op.text, encoding="utf-8")


def _row_position(lines: Sequence[str], line: int) -> int:
    """The list position of ledger row ``line``, which is the Nth nonempty line.

    Row numbers are file line numbers, and a row is renumbered when one is deleted
    or appended, so a case and a control name a row by that number.
    """
    counter = 0
    for position, text in enumerate(lines):
        if not text.strip():
            continue
        counter += 1
        if counter == line:
            return position
    raise DataError(f"names a ledger row that does not exist: line {line}")


def _apply_row_op(cfg: Config, root: Path, op: RowOp) -> None:
    """Seed one ledger edit: duplicate a row, set a field, or drop a field."""
    ledger = root / cfg.ledger_relpath
    lines = ledger.read_text(encoding="utf-8").splitlines()
    index = _row_position(lines, op.line)
    row = json.loads(lines[index])
    if op.op == "duplicate":
        lines.insert(index + 1, json.dumps(row, sort_keys=True))
    elif op.op in ("set", "drop"):
        _require(op.field in row, f"case {op.op}s a field the row does not carry: {op.field}")
        if op.op == "set":
            row[op.field] = op.value
        else:
            row.pop(op.field)
        lines[index] = json.dumps(row, sort_keys=True)
    else:  # pragma: no cover - load_config admits no other op
        raise DataError(f"unknown row op: {op.op}")
    ledger.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _apply_case(cfg: Config, root: Path, case: Case) -> None:
    for op in case.files:
        _apply_file_op(root, op)
    for op in case.rows:
        _apply_row_op(cfg, root, op)


def _clean_copy(cfg: Config, base: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="index-check-case-"))
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(base, work, dirs_exist_ok=True)
    return work


# ---------------------------------------------------------------------------
# --self-test
# ---------------------------------------------------------------------------

def self_test(cfg: Config) -> int:
    """Accept the clean fixture; refuse every seeded defect with its rule."""
    failures: list[str] = []
    red: list[str] = []
    base = Path(tempfile.mkdtemp(prefix="index-check-clean-"))
    try:
        _materialize(cfg, base)
        clean, _ = scan(cfg, base)
        if clean:
            print(f"SELF-TEST FAIL clean fixture: {clean[0].render()}")
            return EXIT_REJECT
        print("SELF-TEST GREEN clean fixture (3 rows, 3 files, real digests)")

        for case in cfg.cases:
            work = _clean_copy(cfg, base)
            try:
                try:
                    _apply_case(cfg, work, case)
                    violations, _summary = scan(cfg, work)
                except DataError as exc:
                    if case.expect == "error":
                        print(f"SELF-TEST RED  {case.name}: data error: {exc}")
                        red.append(f"{case.name} (structure)")
                        continue
                    failures.append(f"{case.name}: data error where rule {case.rule} was expected: {exc}")
                    print(f"SELF-TEST FAIL {case.name}: data error: {exc}")
                    continue
                rules = sorted({violation.rule for violation in violations})
                if case.expect == "reject":
                    if not violations:
                        failures.append(f"{case.name}: survived, expected rule {case.rule}")
                        print(f"SELF-TEST FAIL {case.name}: survived")
                    elif case.rule and case.rule not in rules:
                        failures.append(f"{case.name}: rejected by {rules}, expected {case.rule}")
                        print(f"SELF-TEST FAIL {case.name}: rejected by {rules}, expected {case.rule}")
                    else:
                        hit = next(
                            (violation for violation in violations if violation.rule == case.rule),
                            violations[0],
                        )
                        print(f"SELF-TEST RED  {case.name}: {hit.render()}")
                        red.append(f"{case.rule} {case.name}")
                elif case.expect == "accept":
                    if violations:
                        failures.append(f"{case.name}: rejected, expected acceptance")
                        print(f"SELF-TEST FAIL {case.name}: {violations[0].render()}")
                    else:
                        print(f"SELF-TEST GREEN {case.name}")
                else:
                    failures.append(f"{case.name}: accepted, expected a data error")
                    print(f"SELF-TEST FAIL {case.name}: accepted, expected a data error")
            finally:
                shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if failures:
        print(f"SELF-TEST FAIL: {len(failures)} case(s) failed")
        return EXIT_REJECT
    reject_count = sum(1 for case in cfg.cases if case.expect == "reject")
    accept_count = sum(1 for case in cfg.cases if case.expect == "accept")
    error_count = sum(1 for case in cfg.cases if case.expect == "error")
    print(
        f"SELF-TEST PASS: clean fixture accepted, {reject_count} defect fixture(s) rejected, "
        f"{error_count} structural fixture(s) raised a data error, {accept_count} accept fixture(s)"
    )
    print(f"SELF-TEST PASS: rejected: {', '.join(red)}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# --controls
# ---------------------------------------------------------------------------

def _load_real_ledger(cfg: Config, root: Path) -> list[tuple[int, dict[str, Any]]]:
    ledger = root / cfg.ledger_relpath
    if not ledger.is_file():
        raise DataError(f"the controls need an index in the tree: {cfg.ledger_relpath}")
    rows = parse_rows(ledger.read_text(encoding="utf-8"), cfg.ledger_relpath)
    if len(rows) < 2:
        raise DataError("the duplicate-id control needs at least two rows in the index")
    return rows


def _rewrite_ledger(root: Path, cfg: Config, transform) -> None:
    ledger = root / cfg.ledger_relpath
    ledger.write_text(transform(ledger.read_text(encoding="utf-8")), encoding="utf-8")


def _set_row_field(text: str, line: int, field: str, value: Any) -> str:
    lines = text.splitlines()
    index = _row_position(lines, line)
    row = json.loads(lines[index])
    row[field] = value
    lines[index] = json.dumps(row, sort_keys=True)
    return "".join(f"{entry}\n" for entry in lines)


def controls(cfg: Config, root: Path) -> int:
    """Baseline green, each rule red under a seeded defect, each restore green.

    The real evidence subtree is copied into the system temporary directory and
    mutated there; the working tree is read-only.
    """
    rows = _load_real_ledger(cfg, root)
    first_line, first = rows[0]
    second = rows[1][1]
    indexed = str(first["path"])
    ledger_rel = cfg.ledger_relpath

    base = Path(tempfile.mkdtemp(prefix="index-check-base-"))
    work = Path(tempfile.mkdtemp(prefix="index-check-mutant-"))
    try:
        shutil.copytree(root / cfg.evidence_root, base / cfg.evidence_root, dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(base, work, dirs_exist_ok=True)

        baseline, _ = scan(cfg, work)
        if baseline:
            print(f"CONTROL baseline: RED :: {baseline[0].render()}", file=sys.stderr)
            return EXIT_REJECT
        print(f"CONTROL baseline: GREEN ({len(rows)} rows, {len(evidence_files(cfg, work))} files)")

        steps: list[tuple[str, str, Any]] = [
            ("S", "ledger-absent", lambda: (work / ledger_rel).unlink()),
            (
                "E1",
                "unindexed-file",
                lambda: (
                    (work / CONTROL_UNINDEXED).write_text(CONTROL_TEXT, encoding="utf-8")
                ),
            ),
            (
                "E2",
                "duplicate-row",
                lambda: _rewrite_ledger(
                    work, cfg, lambda text: text + _line_of(text, first_line)
                ),
            ),
            ("E3", "dangling-row", lambda: (work / indexed).unlink()),
            (
                "E4",
                "duplicate-id",
                lambda: _rewrite_ledger(
                    work, cfg, lambda text: _set_row_field(text, first_line, "id", second["id"])
                ),
            ),
            (
                "E5",
                "stale-sha256",
                lambda: _rewrite_ledger(
                    work, cfg, lambda text: _set_row_field(text, first_line, "sha256", CONTROL_ZERO_SHA)
                ),
            ),
            (
                "E5",
                "stale-bytes",
                lambda: _rewrite_ledger(
                    work,
                    cfg,
                    lambda text: _set_row_field(text, first_line, "bytes", 1 << 20),
                ),
            ),
            (
                "E5",
                "edited-file-behind-the-row",
                lambda: (work / indexed).open("a", encoding="utf-8").write(CONTROL_TEXT),
            ),
            (
                "E6",
                "row-kind-outside-domain",
                lambda: _rewrite_ledger(
                    work, cfg, lambda text: _set_row_field(text, first_line, "kind", "note")
                ),
            ),
        ]

        failures: list[str] = []
        for rule, name, mutate in steps:
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(base, work, dirs_exist_ok=True)
            try:
                mutate()
            except OSError as exc:
                failures.append(f"{name}: mutation failed: {exc}")
                print(f"CONTROL {name}: FAIL :: mutation error: {exc}", file=sys.stderr)
                continue
            try:
                mutated, _ = scan(cfg, work)
            except DataError as exc:
                failures.append(f"{name}: data error where rule {rule} was expected: {exc}")
                print(f"CONTROL {name}: FAIL :: data error: {exc}", file=sys.stderr)
                continue
            if not any(violation.rule == rule for violation in mutated):
                found = sorted({violation.rule for violation in mutated}) or "no rule"
                failures.append(f"{name}: SURVIVED ({found} raised, expected {rule})")
                print(f"CONTROL {name}: SURVIVED :: {found} raised, expected {rule}", file=sys.stderr)
                continue
            hit = next(violation for violation in mutated if violation.rule == rule)
            print(f"CONTROL {name}: RED :: {hit.render()}")

            # Restore leg: put the pristine subtree back and the gate must be green.
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(base, work, dirs_exist_ok=True)
            restored, _ = scan(cfg, work)
            if restored:
                failures.append(f"{name}: restored copy is not green")
                print(f"CONTROL {name} restore: RED :: {restored[0].render()}", file=sys.stderr)
                continue
            print(f"CONTROL {name} restore: GREEN")
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)

    if failures:
        print(f"CONTROLS FAIL: {len(failures)} leg(s) failed", file=sys.stderr)
        return EXIT_REJECT
    print("CONTROLS PASS: baseline green, every seeded defect red with its rule, every restore green")
    return EXIT_OK


def _line_of(text: str, line: int) -> str:
    """The verbatim ledger line of row ``line``, counting nonempty lines."""
    entry = text.splitlines()[_row_position(text.splitlines(), line)]
    return entry if entry.endswith("\n") else entry + "\n"


# ---------------------------------------------------------------------------
# --check and --list
# ---------------------------------------------------------------------------

def check(cfg: Config, root: Path) -> int:
    violations, summary = scan(cfg, root)
    if not violations:
        print(
            f"CHECK CLEAN: {cfg.ledger_relpath} indexes every file under "
            f"{cfg.evidence_root}/ exactly once"
        )
        print(summary.render())
        print(f"  digests: {summary.with_digest}/{summary.rows} row(s) record bytes and sha256")
        return EXIT_OK
    for violation in violations:
        print(violation.render())
    by_rule: dict[str, int] = {}
    for violation in violations:
        by_rule[violation.rule] = by_rule.get(violation.rule, 0) + 1
    detail = ", ".join(f"{rule}={count}" for rule, count in sorted(by_rule.items()))
    print(summary.render())
    print(f"CHECK FAIL: {len(violations)} violation(s) ({detail})")
    return EXIT_REJECT

def cmd_list(cfg: Config, root: Path) -> int:
    print(f"Index: {cfg.ledger_relpath}")
    print(f"Files indexed: every regular file under {cfg.evidence_root}/ except the ledger")
    print(f"Row keys: {', '.join(cfg.required)}")
    print(f"Kinds: {', '.join(cfg.kinds)}")
    print("Rules:")
    for rule_id, title, description in cfg.rules:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    if not (root / cfg.ledger_relpath).is_file():
        print(f"  (no index at {cfg.ledger_relpath} in this tree)")
        return EXIT_OK
    rows = parse_rows((root / cfg.ledger_relpath).read_text(encoding="utf-8"), cfg.ledger_relpath)
    with_digest = sum(
        1
        for _line, row in rows
        if isinstance(row.get("bytes"), int)
        and not isinstance(row.get("bytes"), bool)
        and isinstance(row.get("sha256"), str)
        and len(row["sha256"]) == 64
    )
    print(f"Rows in {cfg.ledger_relpath}: {len(rows)}")
    print(f"Rows recording bytes and sha256: {with_digest}/{len(rows)}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="index_check.py",
        description=(
            "Rejecting gate for the evidence index: every file under docs/evidence has "
            "exactly one ledger row, no row dangles, no id is duplicated and every "
            "recorded digest matches the file (S, E1-E6)."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list the row schema and the rules")
    mode.add_argument("--self-test", action="store_true", help="exercise the seeded-defect fixtures")
    mode.add_argument(
        "--controls",
        action="store_true",
        help="baseline/mutation/restore each rule against a copy of the real evidence tree",
    )
    mode.add_argument("--check", action="store_true", help="scan the repository tree")
    parser.add_argument(
        "--root",
        default=str(REPO),
        metavar="PATH",
        help="tree to scan (default: the repository root)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        root = Path(args.root).resolve()
        if args.list:
            return cmd_list(cfg, root)
        if args.self_test:
            return self_test(cfg)
        if args.controls:
            return controls(cfg, root)
        return check(cfg, root)
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
