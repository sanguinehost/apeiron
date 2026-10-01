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
    complexity_claim_pattern: str
    complexity_qualifiers: tuple[str, ...]
    complexity_measure_pattern: str
    complexity_unit_pattern: str
    complexity_unit_head_pattern: str
    complexity_derivation_pattern: str
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
        complexity_claim_pattern=_string(
            data.get("complexity_claim_pattern"), "complexity_claim_pattern"
        ),
        complexity_qualifiers=tuple(
            _string_list(data.get("complexity_qualifiers"), "complexity_qualifiers")
        ),
        complexity_measure_pattern=_string(
            data.get("complexity_measure_pattern"), "complexity_measure_pattern"
        ),
        complexity_unit_pattern=_string(
            data.get("complexity_unit_pattern"), "complexity_unit_pattern"
        ),
        complexity_unit_head_pattern=_string(
            data.get("complexity_unit_head_pattern"), "complexity_unit_head_pattern"
        ),
        complexity_derivation_pattern=_string(
            data.get("complexity_derivation_pattern"), "complexity_derivation_pattern"
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


_LINE_COMMENT_MARKERS = {
    ".rs": "//",
    ".c": "//",
    ".h": "//",
    ".cpp": "//",
    ".hpp": "//",
    ".lean": "--",
    ".hs": "--",
    ".lua": "--",
    ".py": "#",
    ".sh": "#",
    ".bash": "#",
    ".toml": "#",
    ".yaml": "#",
    ".yml": "#",
}
_PROSE_SUFFIXES = frozenset({".md", ".txt", ".tsv"})
_GENERIC_CLAIM = re.compile(r"\bO\(\s*\S|\bconstant[ -]time\b|\bamortiz|\blinear time\b")
_BIG_O = re.compile(r"\bO\([^)]*\)")
# A count expression outside the big-O token itself: the numeral or number word
# that turns "O(1)" into "O(1) in two comparisons". A digit inside an identifier
# (i128, u64, MAX_K) is not a count, and an adverb of exactness is not a count.
#
# A numeral counts only when it is followed by the noun it counts: AGENTS.md §8
# asks for "the measure they are counted in (comparisons, allocations, bytes,
# words, bytes copied)", which is a unit, so "11 cons cells" and "18 layers" are
# measures and a bare digit is not. `32*127` and `Q32.32` are a value bound and a
# format width; they say how large an operand may be, not what is counted, so they
# are not measures. A digit inside an identifier (i128, u64, MAX_K), after a dot
# (`Q32.32`, `0.5`) or beside a non-word character is not a count.
_COUNT = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|fixed|constant|"
    r"single|at most)\b|(?<![A-Za-z0-9_.])\d+\s+[A-Za-z]{2,}",
    re.IGNORECASE,
)
# The unit side of the measure test, mirroring `complexity_unit_pattern` in
# cases.json: AGENTS.md §8 asks for "the measure they are counted in
# (comparisons, allocations, bytes, words, bytes copied)" plus, in this corpus,
# the operation nouns those comments actually use (integer multiply-adds,
# integer operations, payload clones, writes, cons cells, list links, tasks,
# probes). A clause naming a unit for its count is a measure even with no
# numeral: "copies nothing", "`O(M*B*32)` writes", "in byte comparisons".
#
# It is a phrase test, not a word list: a bare `bytes` in "the bytes are read"
# or "only words that could select a slot" is prose, while "source bytes",
# "`O(bytes)`", "cons cells" and "copies nothing" state a unit. So the head
# noun must carry a quantity (`11 cons cells`, `no allocation`) or a non-function
# modifier (`source bytes`, `integer operations`), and must not be an identifier
# (`bytes[l]`, `x.bytes`).
_COUNT_CUE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+|per|each|every|all|"
    r"both|neither|at most|no|none|nothing|exactly|integer|cons|payload|byte|word|list)\s+",
    re.IGNORECASE,
)
_NOT_A_MODIFIER = re.compile(
    r"\b(?:the|a|an|this|that|these|those|its|their|our|your|my|his|her|some|any|not|"
    r"of|in|on|at|by|for|to|from|with|into|onto|up|down|out|over|under|through|across|"
    r"then|so|because|when|while|if|and|or|but|is|are|was|were|be|been|being|has|have|"
    r"had|do|does|did|as|than|which|who|whose|there|here|also|only|just|still|after|"
    r"before|until|unless|whether|since|about|like|returns?|return|reads?|writes?)\s+",
    re.IGNORECASE,
)
_UNIT_HEAD = (
    r"comparisons|compares|operations|ops|multiply-adds|allocations|bytes|words|"
    r"copies|clones|writes|tasks|probes"
)
_ZERO_COUNT = re.compile(
    r"\b(?:cop(?:ies|ied)|clon(?:es|ed)|allocat(?:es|ed|ion)|writ(?:es|ten)|reads?|"
    r"travers(?:es|ed)|touch(?:es|ed)|visit(?:s|ed)|scan(?:s|ned)|probes?|tasks?)\s+"
    r"(?:nothing|none|zero)\b|\bnothing\s+(?:copied|cloned|allocated|written|read|"
    r"traversed|touched|visited|issued|emitted)\b",
    re.IGNORECASE,
)


def _markers_for(path: str) -> str | None:
    """The line-comment marker for a path, or None when the file is prose."""
    suffix = Path(path).suffix.lower()
    if suffix in _PROSE_SUFFIXES or suffix == "":
        return None
    return _LINE_COMMENT_MARKERS.get(suffix)


def _comment_payload(line: str, markers: str | None) -> str | None:
    """The comment payload of a line, or None when it is not a comment line."""
    stripped = line.strip()
    if stripped == "":
        return None
    if markers is None:
        return stripped
    if not stripped.startswith(markers):
        return None
    return stripped[len(markers):].lstrip("/!#* \t").rstrip()


# A block comment whose opener and closer differ from the line marker: Lean's
# `/-- ... -/`. It is read as one block, exactly as Rust's `///` run is, so a
# measure on an adjacent line of the same doc comment is seen.
_BLOCK_COMMENT = {".lean": ("/--", "-/")}


def _block_comment_lines(
    line: str, opening: str, closing: str
) -> list[str] | None:
    """The payloads of a block comment whose text is one line, or None."""
    stripped = line.strip()
    if not stripped.startswith(opening):
        return None
    inner = stripped[len(opening):]
    if closing not in inner:
        return None
    return [inner.split(closing, 1)[0].strip()]


def _comment_blocks(
    text: str, markers: str | None, suffix: str = ""
) -> list[tuple[int, list[str]]]:
    """Contiguous comment lines with their 1-based start line.

    A `/-- ... -/` doc comment contributes one payload entry per source line, so
    `start + offset` still names the line the entry came from and a claim on any
    line of the block sees the whole block.
    """
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    delimiters = _BLOCK_COMMENT.get(suffix)
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        number = index + 1
        if delimiters is not None:
            inline = _block_comment_lines(line, *delimiters)
            if inline is not None:
                if current:
                    blocks.append((start, current))
                    current = []
                start = number
                current.extend(inline)
                index += 1
                continue
            if line.strip().startswith(delimiters[0]):
                if current:
                    blocks.append((start, current))
                    current = []
                start = number
                pending = line.strip()[len(delimiters[0]):]
                while True:
                    if delimiters[1] in pending:
                        current.append(pending.split(delimiters[1], 1)[0].strip())
                        index += 1
                        break
                    current.append(pending.strip())
                    index += 1
                    if index >= len(lines):
                        break
                    pending = lines[index].strip()
                continue
        payload = _comment_payload(line, markers)
        if payload is None:
            if current:
                blocks.append((start, current))
                current = []
            index += 1
            continue
        if not current:
            start = number
        current.append(payload)
        index += 1
    if current:
        blocks.append((start, current))
    return blocks


def _is_complexity_claim(line: str, cfg: Config) -> bool:
    """The line carries a complexity claim, not the bare noun 'complexity'."""
    return _any_lower(line, cfg.complexity_terms) or _term_pattern(
        cfg.complexity_claim_pattern
    ).search(line) is not None


def _is_heading(line: str, markers: str | None) -> bool:
    """A heading that states no claim of its own.

    Structural, not lexical: the test is the line's shape (a `#` heading or a
    bold header) and the absence of a claim token in the generic form, so
    re-adding a noun to `complexity_terms` cannot make a heading fire again.
    """
    payload = _comment_payload(line, markers)
    if payload is None:
        return False
    core = payload.lstrip()
    heading = core.startswith("#") or (
        core.startswith("**") and core.rstrip("*").strip() != ""
    )
    return heading and _GENERIC_CLAIM.search(payload) is None


def _names_unit(block: str, cfg: Config) -> bool:
    """The block names the unit its count is in, with or without a numeral.

    §8 asks for "the measure they are counted in (comparisons, allocations,
    bytes, words, bytes copied)", so a unit is a measure in its own right: a
    clause that says what is counted qualifies, whether or not it also says how
    many. The unit test runs over the whole block, and separately over the
    argument of each big-O token, because `O(bytes)` names bytes as the measure;
    the numeral test does not read the big-O token, because a numeral inside
    `O(1)` is a bound, not a count.
    """
    if _pattern(cfg.complexity_unit_pattern).search(block) is not None:
        return True
    head = _pattern(cfg.complexity_unit_head_pattern)
    for token in _BIG_O.finditer(block):
        argument = token.group(0)[token.group(0).index("(") + 1:-1].strip()
        if argument and head.search(argument) is not None:
            return True
    return False


def _names_measure(block: str, cfg: Config) -> bool:
    """The block counts the cost in something: an operation count or a unit.

    The test is the presence of a count expression outside the big-O token, not a
    word list, so an author is not obliged to use this repository's phrasing. A
    bare numeral is not a count: §8 asks for the measure a cost is counted *in*,
    so the numeral must carry the noun it counts (`11 cons cells`, not `32*127`).
    """
    if _pattern(cfg.complexity_measure_pattern).search(_BIG_O.sub(" ", block)) is not None:
        return True
    return _names_unit(block, cfg)


def _names_derivation(block: str, cfg: Config) -> bool:
    """The block states how the count closes: a loop, pass or per-element bound."""
    return _pattern(cfg.complexity_derivation_pattern).search(block) is not None


@lru_cache(maxsize=None)
def _pattern(source: str) -> re.Pattern[str]:
    return re.compile(source, re.IGNORECASE)


# §8 exempts "trivial O(1) accessors, projections and constructors" from the
# complexity comment. The exemption below is deliberately narrow: the comment
# must claim only O(1), and the flagged item's body must be a projection, a
# `len()`-style read or a constructor application. A loop, any call beyond the
# allowed reads, a conditional other than an `Option`/`Result` match, or
# arithmetic in the body removes it.
_ONLY_O1 = re.compile(r"\bO\(\s*1\s*\)")
_NON_O1_TOKEN = re.compile(r"\bO\([^)]*\)")
_BODY_KEYWORD = re.compile(r"(?<![\w.$])(?:for|while|loop|if|else|unsafe|yield|await)(?![\w$])")
_ALLOWED_READ = frozenset({"len", "copy"})
_BODY_CALL = re.compile(r"(?<![\w.$:])([a-z_]\w*)\s*\(")
_BODY_ARITHMETIC = re.compile(r"\s[+\-*/%]\s|\s[+\-*/%]|[+\-*/%]\s|\+\-")
_MATCH_ARM = re.compile(r"^(?:Some|None|Ok|Err|Self)\b.*$|^[&*]*\(?[A-Za-z_]\w*"
                        r"(?:\.[A-Za-z_]\w*(?:\(\))?)*\)?$")
_ARM_SPLIT = re.compile(r",(?![^(]*\))")


def _match_arms(text: str) -> list[str]:
    """The comma-separated arm results of a `match`, braces dropped."""
    out: list[str] = []
    for part in _ARM_SPLIT.split(text):
        cleaned = part.replace("{", " ").replace("}", " ").strip().strip(",").strip()
        if cleaned:
            out.append(cleaned)
    return out
_LINE_COMMENT_OR_STRING = re.compile(r"//[^\n]*|\"[^\"]*\"|'[^']*'")


def _claims_only_o1(block: str) -> bool:
    """Every big-O token in the block is O(1); no other cost is claimed."""
    tokens = _NON_O1_TOKEN.findall(block)
    if not tokens:
        return False
    return all(_ONLY_O1.fullmatch(token) for token in tokens)


def _strip_body_noise(body: str) -> str:
    return _LINE_COMMENT_OR_STRING.sub(" ", body)


def _rust_body_after(text: str, after_line: int) -> str | None:
    """The body inside the braces of the Rust item following a comment block."""
    lines = text.splitlines()
    index = after_line
    depth = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "" or stripped.startswith("//"):
            index += 1
            continue
        if stripped.startswith("#"):
            depth += sum(stripped.count(char) for char in "([{")
            depth -= sum(stripped.count(char) for char in ")]}")
            index += 1
            if depth > 0:
                continue
            depth = 0
            continue
        break
    if index >= len(lines):
        return None
    signature = lines[index].strip()
    if "fn " not in signature or "(" not in signature or "{" not in signature:
        return None
    item: list[str] = []
    balance = 0
    opened = False
    while index < len(lines):
        line = lines[index]
        item.append(line)
        balance += line.count("{") - line.count("}")
        if "{" in line:
            opened = True
        index += 1
        if opened and balance <= 0:
            break
    if not opened:
        return None
    whole = "\n".join(item)
    opening = whole.index("{")
    depth = 0
    for position in range(opening, len(whole)):
        char = whole[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return whole[opening + 1:position]
    return None


def _trivial_o1_body(body: str) -> bool:
    """The body is a projection, a `len()`-style read or a constructor."""
    text = _strip_body_noise(body)
    if text.strip() == "":
        return False
    if _BODY_KEYWORD.search(text) is not None:
        return False
    for match in _BODY_CALL.finditer(text):
        if match.group(1) not in _ALLOWED_READ:
            return False
    if _BODY_ARITHMETIC.search(text) is not None:
        return False
    if re.search(r"(?<![\w.$])match(?![\w$])", text) is not None:
        arms = [arm for part in re.split(r"=>", text)[1:] for arm in _match_arms(part)]
        if not arms or not all(_MATCH_ARM.match(arm) for arm in arms):
            return False
    return True


def _is_trivial_o1(path: str, text: str, block: str, last_line: int) -> bool:
    """§8's exemption for a trivial O(1) accessor, projection or constructor."""
    if Path(path).suffix.lower() != ".rs":
        return False
    if not _claims_only_o1(block):
        return False
    body = _rust_body_after(text, last_line)
    if body is None:
        return False
    return _trivial_o1_body(body)


def rule_d(path: str, text: str, cfg: Config) -> list[Violation]:
    """A complexity claim whose enclosing comment block adds nothing to it.

    AGENTS.md §8 requires a comment at the definition carrying the complexity,
    the measure counted in, and the derivation. The unit of judgement is the
    enclosing comment block, not the flagged line, because §8 asks for a comment
    at the definition rather than an adjective on one line of it. A block that
    adds a counted measure or a derivation satisfies it; a block that restates
    the claim and stops does not.
    """
    if path in cfg.governing_documents:
        return []
    markers = _markers_for(path)
    blocks = _comment_blocks(text, markers, Path(path).suffix.lower())
    owner: dict[int, int] = {}
    for position, (start, payload) in enumerate(blocks):
        for offset in range(len(payload)):
            owner[start + offset] = position
    out: list[Violation] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not _is_complexity_claim(line, cfg):
            continue
        if _is_heading(line, markers):
            continue
        position = owner.get(index)
        block = blocks[position][1] if position is not None else [line]
        joined = "\n".join(block)
        if _any_lower(joined, cfg.complexity_qualifiers):
            continue
        if _names_measure(joined, cfg) or _names_derivation(joined, cfg):
            continue
        last_line = (
            blocks[position][0] + len(blocks[position][1]) - 1 if position is not None else index
        )
        if _is_trivial_o1(path, text, joined, last_line):
            continue
        out.append(
            Violation(
                "D",
                path,
                index,
                f"complexity claim without its measure and derivation in the enclosing "
                f"comment block (AGENTS.md §8 wants the complexity, the measure counted in, "
                f"and the recurrence): {line.strip()[:140]}",
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
