#!/usr/bin/env python3
"""Rejecting gate for the canonical-rooting directive.

Every claim that something refines or executes a canonical object must be rooted
in an executing canonical body: the canonical map semantics
(``Apeiron.Table.{HMap,get,insertWith,build}``), the algebra
(``Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}``), the extracted
``Cell<i64>`` bodies linked by ``word_mul_denote`` / ``word_norm_denote``, and the
fixed-shard ``Table<K,V,H,N>`` with ``Sigma`` in every executing signature and
``Coord`` addressing per face. This runner is the missing command that fails when
a governing document or source path states such a claim without that root, when
it presents a retired route as the active architecture, when a new primitive or
refinement names no canonical link, when a complexity claim omits its qualifier,
or when a task/ownership declaration exceeds three files or lacks a gate or a
negative control.

What is checked
===============
* ``--check`` scans the governing documents and the configured source roots and
  reports every violation of rules A--E; it exits nonzero when any violation is
  reported. The scan is text-only and needs no network, Lean or Rust build.
* ``--self-test`` exercises every rejection path on the synthetic fixtures in
  ``tools/rooting/cases.json`` and accepts a good fixture. It exits 0 only when
  every ``reject`` case is rejected and every ``accept`` case is accepted.
* ``--controls`` runs the baseline/mutation/restore control on those fixtures:
  the good fixture is green, every rejection fixture is red, and the restored
  good fixture is green again.
* ``--list`` prints the rule set and the paths scanned.

The fixture data is ``tools/rooting/cases.json`` (schema 1). The fixtures are
in-repo-independent: they are inline text, not reads of the repository tree, so
``--self-test`` and ``--controls`` are deterministic and tool-free.

Usage::

    python3 tools/rooting/check.py --list
    python3 tools/rooting/check.py --self-test
    python3 tools/rooting/check.py --controls
    python3 tools/rooting/check.py --check

Exit status: 0 when clean or when every self-test/control passes; 1 when a
violation is reported or a fixture control survives; 2 on a usage or data error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CASES_PATH = HERE.parent / "cases.json"

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2


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
class Case:
    name: str
    rule: str
    expect: str
    path: str
    text: str


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


@dataclass(frozen=True)
class RetiredRoute:
    id: str
    title: str
    path_prefixes: tuple[str, ...]
    tokens: tuple[str, ...]
    case_sensitive: bool


@dataclass(frozen=True)
class Config:
    governing_documents: tuple[str, ...]
    source_roots: tuple[str, ...]
    source_excludes: frozenset[str]
    source_file_excludes: frozenset[str]
    text_suffixes: tuple[str, ...]
    canonical_roots: tuple[str, ...]
    claim_terms: tuple[str, ...]
    import_only_markers: tuple[str, ...]
    canonical_reference_markers: tuple[str, ...]
    claim_negation_markers: tuple[str, ...]
    retirement_markers: tuple[str, ...]
    retired_routes: tuple[RetiredRoute, ...]
    complexity_terms: tuple[str, ...]
    complexity_qualifiers: tuple[str, ...]
    declaration_markers: tuple[str, ...]
    task_id_pattern: str
    task_scope_markers: tuple[str, ...]
    owned_files_markers: tuple[str, ...]
    gate_markers: tuple[str, ...]
    control_markers: tuple[str, ...]
    rules: tuple[tuple[str, str, str], ...]
    baseline: Case
    cases: tuple[Case, ...]


def load_config(path: Path = CASES_PATH) -> Config:
    if not path.is_file():
        raise DataError(f"cases file missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"cases file is not valid JSON: {exc}") from exc
    _require(isinstance(data, dict), "cases file must be an object")
    _require(data.get("schema") == 1, "cases schema must be 1")

    routes_raw = data.get("retired_routes")
    _require(isinstance(routes_raw, list) and routes_raw, "retired_routes must be a nonempty list")
    routes: list[RetiredRoute] = []
    for index, raw in enumerate(routes_raw):
        where = f"retired_routes[{index}]"
        _require(isinstance(raw, dict), f"{where} must be an object")
        prefixes = raw.get("path_prefixes", [])
        _require(isinstance(prefixes, list), f"{where}.path_prefixes must be a list")
        tokens = _string_list(raw.get("tokens"), f"{where}.tokens")
        routes.append(
            RetiredRoute(
                id=_string(raw.get("id"), f"{where}.id"),
                title=_string(raw.get("title"), f"{where}.title"),
                path_prefixes=tuple(_string(p, f"{where}.path_prefixes[]") for p in prefixes),
                tokens=tuple(tokens),
                case_sensitive=bool(raw.get("case_sensitive", True)),
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
    baseline_raw = self_test.get("baseline")
    _require(isinstance(baseline_raw, dict), "self_test.baseline must be an object")
    baseline = Case(
        name="baseline",
        rule="",
        expect="accept",
        path=_string(baseline_raw.get("path"), "self_test.baseline.path"),
        text=_string(baseline_raw.get("text"), "self_test.baseline.text"),
    )
    cases_raw = self_test.get("cases")
    _require(isinstance(cases_raw, list) and cases_raw, "self_test.cases must be a nonempty list")
    cases: list[Case] = []
    for index, raw in enumerate(cases_raw):
        where = f"self_test.cases[{index}]"
        _require(isinstance(raw, dict), f"{where} must be an object")
        expect = _string(raw.get("expect"), f"{where}.expect")
        _require(expect in ("reject", "accept"), f"{where}.expect must be reject or accept")
        cases.append(
            Case(
                name=_string(raw.get("name"), f"{where}.name"),
                rule=_string(raw.get("rule"), f"{where}.rule"),
                expect=expect,
                path=_string(raw.get("path"), f"{where}.path"),
                text=_string(raw.get("text"), f"{where}.text"),
            )
        )

    return Config(
        governing_documents=tuple(_string_list(data.get("governing_documents"), "governing_documents")),
        source_roots=tuple(_string_list(data.get("source_roots"), "source_roots")),
        source_excludes=frozenset(_string_list(data.get("source_excludes"), "source_excludes")),
        source_file_excludes=frozenset(
            _optional_string_list(data.get("source_file_excludes"), "source_file_excludes")
        ),
        text_suffixes=tuple(_string_list(data.get("text_suffixes"), "text_suffixes")),
        canonical_roots=tuple(_string_list(data.get("canonical_roots"), "canonical_roots")),
        claim_terms=tuple(_string_list(data.get("claim_terms"), "claim_terms")),
        import_only_markers=tuple(_string_list(data.get("import_only_markers"), "import_only_markers")),
        canonical_reference_markers=tuple(
            _string_list(data.get("canonical_reference_markers"), "canonical_reference_markers")
        ),
        claim_negation_markers=tuple(
            _string_list(data.get("claim_negation_markers"), "claim_negation_markers")
        ),
        retirement_markers=tuple(_string_list(data.get("retirement_markers"), "retirement_markers")),
        retired_routes=tuple(routes),
        complexity_terms=tuple(_string_list(data.get("complexity_terms"), "complexity_terms")),
        complexity_qualifiers=tuple(
            _string_list(data.get("complexity_qualifiers"), "complexity_qualifiers")
        ),
        declaration_markers=tuple(_string_list(data.get("declaration_markers"), "declaration_markers")),
        task_id_pattern=_string(data.get("task_id_pattern"), "task_id_pattern"),
        task_scope_markers=tuple(_string_list(data.get("task_scope_markers"), "task_scope_markers")),
        owned_files_markers=tuple(_string_list(data.get("owned_files_markers"), "owned_files_markers")),
        gate_markers=tuple(_string_list(data.get("gate_markers"), "gate_markers")),
        control_markers=tuple(_string_list(data.get("control_markers"), "control_markers")),
        rules=tuple(rules),
        baseline=baseline,
        cases=tuple(cases),
    )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _term_pattern(term: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive pattern; word boundaries only where the term has word chars."""
    escaped = re.escape(term)
    prefix = r"\b" if re.match(r"\w", term) else ""
    suffix = r"\b" if re.search(r"\w$", term) else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def _any_lower(text: str, terms: tuple[str, ...]) -> bool:
    return any(_term_pattern(term).search(text) for term in terms)


def rule_b_path(path: str, cfg: Config) -> list[Violation]:
    """A retired path existing in the tree is an active retired route."""
    for route in cfg.retired_routes:
        for prefix in route.path_prefixes:
            if path == prefix.rstrip("/") or path.startswith(prefix):
                return [
                    Violation(
                        "B",
                        path,
                        0,
                        f"retired route present in the tree: {route.id} ({prefix})",
                    )
                ]
    return []


def rule_b_token(path: str, text: str, cfg: Config) -> list[Violation]:
    """A retired token presented as active, without a retirement marker."""
    out: list[Violation] = []
    lines = text.splitlines()
    for route in cfg.retired_routes:
        for token in route.tokens:
            for index, line in enumerate(lines, 1):
                haystack = line if route.case_sensitive else line.lower()
                needle = token if route.case_sensitive else token.lower()
                if needle not in haystack:
                    continue
                if _any_lower(line, cfg.retirement_markers):
                    break
                out.append(
                    Violation(
                        "B",
                        path,
                        index,
                        f"active reference to retired route {route.id} "
                        f"({token}): {line.strip()[:140]}",
                    )
                )
                break
    return out


def rule_a(path: str, text: str, cfg: Config) -> list[Violation]:
    """A refinement/execution claim with no executing canonical root."""
    out: list[Violation] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not _any_lower(line, cfg.claim_terms):
            continue
        if _any_lower(line, cfg.claim_negation_markers):
            continue
        window = line + "\n" + (lines[index + 1] if index + 1 < len(lines) else "")
        if any(root in window for root in cfg.canonical_roots):
            continue
        import_only = _any_lower(line, cfg.import_only_markers)
        reference_only = _any_lower(line, cfg.canonical_reference_markers)
        if not (import_only or reference_only):
            continue
        reason = "import-only" if import_only else "canonical-reference-only"
        out.append(
            Violation(
                "A",
                path,
                index + 1,
                f"{reason} refinement/execution claim without an executing canonical "
                f"root: {line.strip()[:140]}",
            )
        )
    return out


def rule_c(path: str, text: str, cfg: Config) -> list[Violation]:
    """A new primitive/refinement declaration with no canonical link."""
    out: list[Violation] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not _any_lower(line, cfg.declaration_markers):
            continue
        window = line + "\n" + (lines[index + 1] if index + 1 < len(lines) else "")
        if any(root in window for root in cfg.canonical_roots):
            continue
        if _any_lower(window, cfg.canonical_reference_markers):
            continue
        out.append(
            Violation(
                "C",
                path,
                index + 1,
                f"new primitive/refinement declaration without a canonical link: "
                f"{line.strip()[:140]}",
            )
        )
    return out


def rule_d(path: str, text: str, cfg: Config) -> list[Violation]:
    """A complexity claim stated without its qualifier."""
    out: list[Violation] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not _any_lower(line, cfg.complexity_terms):
            continue
        if _any_lower(line, cfg.complexity_qualifiers):
            continue
        out.append(
            Violation(
                "D",
                path,
                index,
                f"complexity claim without its qualifier (fixed N, bounded callbacks, "
                f"nonconstant key length, or a no-claim disclaimer): {line.strip()[:140]}",
            )
        )
    return out


def _blocks(text: str) -> list[tuple[int, list[str]]]:
    """Contiguous non-blank line blocks, with their 1-based start line."""
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    for index, line in enumerate(text.splitlines(), 1):
        if line.strip() == "":
            if current:
                blocks.append((start, current))
                current = []
            continue
        if not current:
            start = index
        current.append(line)
    if current:
        blocks.append((start, current))
    return blocks


def rule_e(path: str, text: str, cfg: Config) -> list[Violation]:
    """An ownership declaration over three files, or an ungated task."""
    out: list[Violation] = []
    task_re = re.compile(cfg.task_id_pattern)
    for start, block in _blocks(text):
        joined = "\n".join(block)
        owned_lines = [line for line in block if _any_lower(line, cfg.owned_files_markers)]
        owned_paths: list[str] = []
        for line in owned_lines:
            owned_paths.extend(
                token for token in re.findall(r"`([^`]+)`", line) if "/" in token or "." in token
            )
        has_task = (
            task_re.search(joined) is not None
            and _any_lower(joined, cfg.task_scope_markers)
        )
        if not owned_paths and not has_task:
            continue
        if len(owned_paths) > 3:
            out.append(
                Violation(
                    "E",
                    path,
                    start,
                    f"task/ownership declaration lists {len(owned_paths)} files "
                    f"(maximum 3): {', '.join(owned_paths)}",
                )
            )
        if not any(_any_lower(line, cfg.gate_markers) for line in block):
            out.append(Violation("E", path, start, "task/ownership declaration has no gate"))
        if not any(_any_lower(line, cfg.control_markers) for line in block):
            out.append(
                Violation("E", path, start, "task/ownership declaration has no negative control")
            )
    return out


def evaluate(path: str, text: str, cfg: Config) -> list[Violation]:
    """Every violation of one document under the configured rule set."""
    path_level = rule_b_path(path, cfg)
    if path_level:
        return path_level
    violations: list[Violation] = []
    violations.extend(rule_a(path, text, cfg))
    violations.extend(rule_b_token(path, text, cfg))
    violations.extend(rule_c(path, text, cfg))
    violations.extend(rule_d(path, text, cfg))
    violations.extend(rule_e(path, text, cfg))
    return violations


# ---------------------------------------------------------------------------
# --check scan
# ---------------------------------------------------------------------------

def _is_text(path: Path, suffixes: tuple[str, ...]) -> bool:
    return path.suffix.lower() in suffixes


def _iter_source_files(cfg: Config):
    for root in cfg.source_roots:
        base = REPO / root
        if not base.exists():
            yield root, None
            continue
        if base.is_file():
            yield root, base
            continue
        for current, dirs, files in os.walk(base):
            dirs[:] = [name for name in sorted(dirs) if name not in cfg.source_excludes]
            for name in sorted(files):
                yield f"{root}/{name}", Path(current) / name


def scan(cfg: Config) -> list[Violation]:
    violations: list[Violation] = []
    for relative in cfg.governing_documents:
        path = REPO / relative
        if not path.is_file():
            violations.append(Violation("S", relative, 0, "governing document is missing"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        violations.extend(evaluate(relative, text, cfg))
    for root, path in _iter_source_files(cfg):
        if path is None:
            violations.append(Violation("S", root, 0, "source root is missing"))
            continue
        relative = path.relative_to(REPO).as_posix()
        if relative in cfg.source_file_excludes:
            continue
        path_level = rule_b_path(relative, cfg)
        if path_level:
            violations.extend(path_level)
            continue
        if not _is_text(path, cfg.text_suffixes):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        violations.extend(evaluate(relative, text, cfg))
    return violations


# ---------------------------------------------------------------------------
# --self-test and --controls
# ---------------------------------------------------------------------------

def _report_case(case: Case, cfg: Config) -> list[Violation]:
    return evaluate(case.path, case.text, cfg)


def self_test(cfg: Config) -> int:
    failures: list[str] = []
    for case in cfg.cases:
        violations = _report_case(case, cfg)
        rules = sorted({violation.rule for violation in violations})
        if case.expect == "reject":
            if not violations:
                failures.append(f"{case.name}: survived, expected rejection by rule {case.rule}")
                print(f"SELF-TEST FAIL {case.name}: survived")
                continue
            if case.rule and case.rule not in rules:
                failures.append(
                    f"{case.name}: rejected by {rules} but expected rule {case.rule}"
                )
                print(f"SELF-TEST FAIL {case.name}: rejected by {rules}, expected {case.rule}")
                continue
            print(f"SELF-TEST RED  {case.name}: {case.rule} :: {violations[0].render()}")
        else:
            if violations:
                failures.append(f"{case.name}: rejected but expected acceptance")
                print(f"SELF-TEST FAIL {case.name}: {violations[0].render()}")
                continue
            print(f"SELF-TEST GREEN {case.name}")
    if failures:
        print(f"SELF-TEST FAIL: {len(failures)} case(s) failed")
        return EXIT_REJECT
    print(
        f"SELF-TEST PASS: baseline accepted, "
        f"{sum(1 for c in cfg.cases if c.expect == 'reject')} rejection cases red, "
        f"{sum(1 for c in cfg.cases if c.expect == 'accept')} acceptance cases green"
    )
    return EXIT_OK


def controls(cfg: Config) -> int:
    baseline = cfg.baseline
    baseline_violations = evaluate(baseline.path, baseline.text, cfg)
    if baseline_violations:
        print(f"CONTROL baseline: RED :: {baseline_violations[0].render()}", file=sys.stderr)
        return EXIT_REJECT
    print("CONTROL baseline: GREEN")

    for case in cfg.cases:
        if case.expect != "reject":
            continue
        mutation_path = case.path
        mutation_text = baseline.text + "\n\n" + case.text
        violations = evaluate(mutation_path, mutation_text, cfg)
        if not violations:
            print(f"CONTROL mutation {case.name}: SURVIVED", file=sys.stderr)
            return EXIT_REJECT
        print(f"CONTROL mutation {case.name}: RED :: {violations[0].render()}")

    restored = evaluate(baseline.path, baseline.text, cfg)
    if restored:
        print(f"CONTROL restored: RED :: {restored[0].render()}", file=sys.stderr)
        return EXIT_REJECT
    print("CONTROL restored: GREEN")
    print("CONTROLS PASS: baseline green, every mutation red, restored green")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(cfg: Config) -> int:
    print("Rules:")
    for rule_id, title, description in cfg.rules:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    print("Governing documents scanned by --check:")
    for relative in cfg.governing_documents:
        print(f"  {relative}")
    print("Source roots scanned by --check:")
    for root in cfg.source_roots:
        print(f"  {root}/")
    print(f"Source directories excluded: {', '.join(sorted(cfg.source_excludes))}")
    print("Retired routes rejected when presented as active:")
    for route in cfg.retired_routes:
        prefixes = f" paths={list(route.path_prefixes)}" if route.path_prefixes else ""
        print(f"  {route.id}: {route.title}{prefixes} tokens={list(route.tokens)}")
    print(f"Canonical roots accepted as an executing link: {len(cfg.canonical_roots)}")
    print(
        f"Self-test fixtures: {len(cfg.cases)} cases "
        f"({sum(1 for c in cfg.cases if c.expect == 'reject')} reject, "
        f"{sum(1 for c in cfg.cases if c.expect == 'accept')} accept)"
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check.py",
        description="Rejecting gate for the canonical-rooting directive.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list the rules and scanned paths")
    mode.add_argument("--self-test", action="store_true", help="exercise every rejection path")
    mode.add_argument("--controls", action="store_true", help="baseline/mutation/restore control")
    mode.add_argument("--check", action="store_true", help="scan the governing documents and sources")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        if args.list:
            return cmd_list(cfg)
        if args.self_test:
            return self_test(cfg)
        if args.controls:
            return controls(cfg)
        violations = scan(cfg)
        if not violations:
            print("CHECK CLEAN: no unrooted claim, active retired route, missing canonical link, "
                  "unqualified complexity claim or ungated task declaration")
            return EXIT_OK
        for violation in violations:
            print(violation.render())
        by_rule: dict[str, int] = {}
        for violation in violations:
            by_rule[violation.rule] = by_rule.get(violation.rule, 0) + 1
        summary = ", ".join(f"{rule}={count}" for rule, count in sorted(by_rule.items()))
        print(f"CHECK FAIL: {len(violations)} violation(s) ({summary})")
        return EXIT_REJECT
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
