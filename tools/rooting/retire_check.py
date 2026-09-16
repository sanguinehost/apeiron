#!/usr/bin/env python3
"""Rejecting retirement gate for the binned Slang derivation route.

The binned route is:

* every file under ``proof/corpus/Slang/Derivation/`` (Steps, Decode, Lexing,
  Grammar, Statements, Selection, Validation, Utilities, Composition, Audit);
* every file under ``proof/corpus/Slang/Generated/Trace/``;
* the ``Run``/``Derives`` acceptance in ``tools/slang/trace_check.py``;
* their Lake roots in ``proof/corpus/lakefile.lean`` and their registrations in
  ``tools/slang/zero_axioms.json``.

This runner fails while any part of that route is still present or still
referenced, so a reintroduced file or import cannot pass silently. ``--check``
scans the whole tree; ``--paths`` is the per-batch gate used as the binned files
are removed in bounded batches.

What is checked
===============
* ``R1`` binned path present: a file under either binned directory exists;
* ``R2`` Lean import: a ``.lean`` file under ``proof/`` imports
  ``Slang.Derivation.*`` or ``Slang.Generated.Trace.*``;
* ``R3`` Lake root: ``proof/corpus/lakefile.lean`` still lists a binned root;
* ``R4`` registry: ``tools/slang/zero_axioms.json`` still registers a binned
  module or source path;
* ``R5`` acceptance: ``tools/slang/trace_check.py`` still carries the
  ``DecodeTrace.Run``/``Derives`` acceptance (``PHASE_MODULES`` and the
  witness-type check);
* ``R6`` producer: ``tools/slang/export.py`` can resurrect the binned route
  (``Slang/Generated/Trace``, ``Slang.Generated.Trace``, ``Slang.Derivation``)
  or emits an unbounded recursion/heartbeat option (``maxHeartbeats 0``,
  ``maxRecDepth 1000000``);
* ``S`` surface: a reference surface the gate must read is missing.

Isolation
=========
``--self-test`` and ``--controls`` materialise synthetic fixtures under the
system temporary directory and never read or write the repository tree; the
working tree is never mutated. ``--paths`` and ``--check`` are read-only. No
Lean, Rust or network is used.

Usage::

    python3 tools/rooting/retire_check.py --list
    python3 tools/rooting/retire_check.py --self-test
    python3 tools/rooting/retire_check.py --controls
    python3 tools/rooting/retire_check.py --check
    python3 tools/rooting/retire_check.py --paths proof/corpus/Slang/Derivation/Steps.lean

Exit status: 0 when the route is retired and every control passes; 1 when a
binned path or an active reference remains, or a fixture control survives; 2 on
a usage or data error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CASES_PATH = HERE.parent / "retire_cases.json"

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

# The one owned path reintroduced by the --controls mutation.
CONTROL_MUTATION = "proof/corpus/Slang/Derivation/Steps.lean"


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


def controls(cfg: Config) -> int:
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
        mutation.write_text("import Slang.Syntax\n", encoding="utf-8")
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


def check(cfg: Config, root: Path) -> int:
    violations = scan_tree(cfg, root)
    if not violations:
        print("CHECK CLEAN: the binned Slang derivation route is retired")
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

def cmd_list(cfg: Config) -> int:
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
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="retire_check.py",
        description="Rejecting retirement gate for the binned Slang derivation route.",
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
        if args.list:
            return cmd_list(cfg)
        if args.self_test:
            return self_test(cfg)
        if args.controls:
            return controls(cfg)
        if args.paths:
            return check_paths(cfg, REPO, args.paths)
        return check(cfg, REPO)
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
