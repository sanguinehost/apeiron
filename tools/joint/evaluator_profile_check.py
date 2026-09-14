#!/usr/bin/env python3
"""Consistency checker for the W18-T2 joint evaluator profile.

This tool establishes **specification consistency only**. It verifies that
``docs/spec/joint-evaluator.md`` and ``tools/joint/evaluator_profile_rows.json``
are internally complete and that every artifact they name resolves:

* every evaluator profile row carries all eight required fields;
* every named canonical theorem resolves as a declaration in the pinned
  canonical corpus (``../sanguine-apeiron/proof``);
* every named canonical entry point resolves as a declaration, and every named
  Song entry exists as a declaration in its source file;
* every required unmatched blocker, hazard and control is present, and none is
  dropped;
* the resource accounting names distinct kinds and never a generic ``fuel``;
* the representation statement names the terms the row declares.

It does **not** establish semantic correspondence, refinement or proof. A green
run says the specification is well formed and source-connected by name; it says
nothing about whether any row's refinement has been proved.

Usage::

    python3 tools/joint/evaluator_profile_check.py --check
    python3 tools/joint/evaluator_profile_check.py --controls
    python3 tools/joint/evaluator_profile_check.py --self-test

``--controls`` and ``--self-test`` mutate a deep copy of the loaded profile in
memory and require each mutation to be rejected. They never touch the working
tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SPEC_PATH = REPO / "docs" / "spec" / "joint-evaluator.md"
ROWS_PATH = HERE.parent / "evaluator_profile_rows.json"
CORPUS_ROOT = REPO.parent / "sanguine-apeiron" / "proof"
BRIDGE_ROOT = REPO / "proof"

PROFILE_ID = "APEIRON-PROGRAMME-W18-T2"

FIELDS = [
    "canonical",
    "entry_points",
    "representation_invariant",
    "input_grammar",
    "admission",
    "behavior",
    "resources",
    "refinement",
]

REQUIRED_ROWS = ["E1", "E2"]
REQUIRED_BLOCKERS = ["BE1", "BE2", "BE3", "BE4", "BE5"]
REQUIRED_HAZARDS = ["HE1", "HE2", "HE3", "HE4", "HE5"]
REQUIRED_CONTROLS = ["C1", "C2", "C3", "C4", "C5", "C6"]
REQUIRED_SECTIONS = [
    "Required fields",
    "Selected boundary",
    "Rows",
    "Unmatched counterparts",
    "Hazards",
    "Required controls",
    "Integration order",
    "Not covered",
    "Checker",
]

ALLOWED_RESOURCE_KINDS = {
    "reduction_steps",
    "passes",
    "nodes",
    "stack_entries",
    "parser_entries",
    "input_bytes",
    "arith_ops",
    "rows",
    "columns",
    "limbs",
}
REQUIRED_RESOURCE_KINDS = {
    "reduction_steps",
    "passes",
    "nodes",
    "stack_entries",
    "parser_entries",
    "arith_ops",
}
FORBIDDEN_RESOURCE_KINDS = {"fuel"}
BEHAVIOR_KEYS = [
    "success",
    "unsupported_input",
    "resource_exhaustion",
    "raw_return_encoding",
    "post_state",
]
INPUT_GRAMMAR_KEYS = ["grammar", "byte_consumption", "trailing_input_policy"]
ADMISSION_SIDES = ["canonical", "song"]

LEAN_DECL = (
    r"(?m)^\s*(?:noncomputable\s+|private\s+|protected\s+|@\[[^\]]*\]\s*)*"
    r"(?:theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
)
SLANG_DECL = [
    r"(?m)^\s*{n}\s*\(",
    r"(?m)^\s*law\s+{n}\b",
    r"(?m)^\s*carrier\s+{n}\b",
    r"(?m)^\s*spec\s+{n}\b",
]


class CheckError(Exception):
    """A specification consistency failure."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def _nonempty(value: object, where: str) -> None:
    _need(
        isinstance(value, str) and value.strip() != "",
        f"{where} must be a nonempty string",
    )


def _resolve_path(rel: str, where: str) -> Path:
    for base in (REPO, REPO.parent):
        candidate = base / rel
        if candidate.is_file():
            return candidate
    raise CheckError(f"{where}: source file does not resolve: {rel}")


def _check_lean_ref(ref: object, where: str, require_root: str) -> None:
    _need(isinstance(ref, dict), f"{where} entries must be objects")
    name = ref.get("name")
    module = ref.get("module")
    root = ref.get("root")
    _nonempty(name, f"{where}.name")
    _nonempty(module, f"{where}.module")
    _need(root in ("corpus", "bridge"), f"{where}.root must be corpus or bridge")
    _need(root == require_root, f"{where}.root must be {require_root} for {name}")
    base = CORPUS_ROOT if root == "corpus" else BRIDGE_ROOT
    path = base / module
    _need(path.is_file(), f"{where}: module does not resolve: {root}:{module}")
    text = path.read_text(encoding="utf-8")
    segments = name.split(".")
    last = segments[-1]
    _need(
        re.search(LEAN_DECL + re.escape(last) + r"\b", text) is not None,
        f"{where}: declaration {last} not found in {module}",
    )
    ns_lines = re.findall(r"(?m)^\s*namespace\s+([\w.]+)", text)
    tokens: set[str] = set()
    for ns in ns_lines:
        tokens.update(ns.split("."))
    for segment in segments[:-1]:
        _need(
            segment in tokens,
            f"{where}: namespace {segment} of {name} not opened in {module}",
        )


def _check_slang(text: str, symbol: str) -> bool:
    return any(
        re.search(pattern.format(n=re.escape(symbol)), text) is not None
        for pattern in SLANG_DECL
    )


def _check_source_entry(entry: object, where: str) -> None:
    _need(isinstance(entry, dict), f"{where} entries must be objects")
    symbol = entry.get("symbol")
    rel = entry.get("file")
    lang = entry.get("lang")
    _nonempty(symbol, f"{where}.symbol")
    _nonempty(rel, f"{where}.file")
    _need(lang == "slang", f"{where}.lang must be slang, got {lang!r}")
    path = _resolve_path(rel, where)
    text = path.read_text(encoding="utf-8")
    _need(
        _check_slang(text, symbol),
        f"{where}: slang entry {symbol} not found in {rel}",
    )


def _check_modules(modules: object, where: str) -> None:
    _need(
        isinstance(modules, list) and modules,
        f"{where} must be a nonempty list of module paths",
    )
    for module in modules:
        _nonempty(module, f"{where} entry")
        path = CORPUS_ROOT / module
        _need(path.is_file(), f"{where}: module does not resolve: corpus:{module}")


def _validate_row(row: object, seen_ids: set[str], resource_kinds: set[str]) -> None:
    _need(isinstance(row, dict), "each row must be an object")
    rid = row.get("id")
    _nonempty(rid, "row.id")
    _need(rid not in seen_ids, f"duplicate row id {rid}")
    seen_ids.add(rid)
    _nonempty(row.get("title"), f"{rid}.title")
    status = row.get("status")
    _need(
        status in ("selected-open", "blocker", "closed"),
        f"{rid}.status invalid: {status!r}",
    )
    for field in FIELDS:
        _need(field in row, f"{rid} drops required field {field}")
        value = row[field]
        _need(value is not None, f"{rid}.{field} is null")
        if isinstance(value, str):
            _nonempty(value, f"{rid}.{field}")
        elif isinstance(value, (list, dict)):
            _need(len(value) > 0, f"{rid}.{field} is empty")

    canonical = row["canonical"]
    for key in ("object", "observation_relation", "types"):
        _nonempty(canonical.get(key), f"{rid}.canonical.{key}")
    _check_modules(canonical.get("modules"), f"{rid}.canonical.modules")
    theorems = canonical.get("theorems")
    _need(isinstance(theorems, list) and theorems, f"{rid}.canonical.theorems empty")
    for theorem in theorems:
        _check_lean_ref(theorem, f"{rid}.canonical.theorems", require_root="corpus")

    entry_points = row["entry_points"]
    canonical_entries = entry_points.get("canonical")
    _need(
        isinstance(canonical_entries, list) and canonical_entries,
        f"{rid}.entry_points.canonical must be nonempty",
    )
    for entry in canonical_entries:
        _check_lean_ref(entry, f"{rid}.entry_points.canonical", require_root="corpus")
    song_entries = entry_points.get("song")
    _need(
        isinstance(song_entries, list) and song_entries,
        f"{rid}.entry_points.song must be nonempty",
    )
    for entry in song_entries:
        _check_source_entry(entry, f"{rid}.entry_points.song")
    rust_entries = entry_points.get("rust")
    _need(
        isinstance(rust_entries, list),
        f"{rid}.entry_points.rust must be a list (possibly empty)",
    )
    for entry in rust_entries:
        _check_source_entry(entry, f"{rid}.entry_points.rust")
    if not rust_entries:
        _nonempty(
            entry_points.get("rust_absent"), f"{rid}.entry_points.rust_absent"
        )

    invariant = row["representation_invariant"]
    for key in ("constructor", "statement"):
        _nonempty(invariant.get(key), f"{rid}.representation_invariant.{key}")
    terms = invariant.get("representation_terms")
    _need(
        isinstance(terms, list) and terms,
        f"{rid}.representation_invariant.representation_terms must be nonempty",
    )
    statement = invariant["statement"].lower()
    for term in terms:
        _nonempty(term, f"{rid}.representation_invariant.representation_terms entry")
        _need(
            term.lower() in statement,
            f"{rid}.representation_invariant.statement does not name {term!r}",
        )

    grammar = row["input_grammar"]
    for key in INPUT_GRAMMAR_KEYS:
        _nonempty(grammar.get(key), f"{rid}.input_grammar.{key}")

    admission = row["admission"]
    for side in ADMISSION_SIDES:
        sub = admission.get(side)
        _need(isinstance(sub, dict), f"{rid}.admission.{side} must be an object")
        _nonempty(sub.get("predicate"), f"{rid}.admission.{side}.predicate")
        _nonempty(sub.get("source"), f"{rid}.admission.{side}.source")
    rust_admission = admission.get("rust")
    if rust_admission is not None:
        _need(
            isinstance(rust_admission, dict),
            f"{rid}.admission.rust must be an object",
        )
        _nonempty(rust_admission.get("predicate"), f"{rid}.admission.rust.predicate")
        _nonempty(rust_admission.get("source"), f"{rid}.admission.rust.source")

    behavior = row["behavior"]
    for key in BEHAVIOR_KEYS:
        _nonempty(behavior.get(key), f"{rid}.behavior.{key}")

    resources = row["resources"]
    for side in ("canonical", "song"):
        entries = resources.get(side)
        _need(
            isinstance(entries, list) and entries,
            f"{rid}.resources.{side} must be nonempty",
        )
        for entry in entries:
            kind = entry.get("kind") if isinstance(entry, dict) else None
            _need(
                kind not in FORBIDDEN_RESOURCE_KINDS,
                f"{rid}.resources.{side}: forbidden resource kind {kind!r}",
            )
            _need(
                kind in ALLOWED_RESOURCE_KINDS,
                f"{rid}.resources.{side} invalid kind {kind!r}",
            )
            _nonempty(entry.get("accounting"), f"{rid}.resources.{side}[{kind}]")
            resource_kinds.add(kind)
    rust_resources = resources.get("rust")
    _need(
        isinstance(rust_resources, list),
        f"{rid}.resources.rust must be a list (possibly empty)",
    )
    for entry in rust_resources:
        kind = entry.get("kind") if isinstance(entry, dict) else None
        _need(
            kind not in FORBIDDEN_RESOURCE_KINDS,
            f"{rid}.resources.rust: forbidden resource kind {kind!r}",
        )
        _need(kind in ALLOWED_RESOURCE_KINDS, f"{rid}.resources.rust invalid kind {kind!r}")
        _nonempty(entry.get("accounting"), f"{rid}.resources.rust[{kind}]")
        resource_kinds.add(kind)
    if not rust_resources:
        _nonempty(resources.get("rust_absent"), f"{rid}.resources.rust_absent")

    refinement = row["refinement"]
    _nonempty(refinement.get("source_connected"), f"{rid}.refinement.source_connected")
    _nonempty(refinement.get("rejecting_control"), f"{rid}.refinement.rejecting_control")
    artifacts = refinement.get("artifacts")
    _need(
        isinstance(artifacts, list),
        f"{rid}.refinement.artifacts must be a list (possibly empty)",
    )
    if not artifacts:
        _nonempty(refinement.get("artifacts_missing"), f"{rid}.refinement.artifacts_missing")
    for artifact in artifacts:
        _check_lean_ref(artifact, f"{rid}.refinement.artifacts", require_root="bridge")


def _validate_blocker(blocker: object) -> None:
    _need(isinstance(blocker, dict), "each unmatched blocker must be an object")
    bid = blocker.get("id")
    _nonempty(bid, "blocker.id")
    _nonempty(blocker.get("counterpart"), f"{bid}.counterpart")
    _nonempty(blocker.get("reason"), f"{bid}.reason")
    _need(blocker.get("status") == "blocker", f"{bid}.status must be 'blocker'")
    theorem = blocker.get("canonical_theorem")
    if theorem is not None:
        _check_lean_ref(theorem, f"{bid}.canonical_theorem", require_root="corpus")
    theorems = blocker.get("canonical_theorems")
    if theorems is not None:
        _need(isinstance(theorems, list) and theorems, f"{bid}.canonical_theorems empty")
        for item in theorems:
            _check_lean_ref(item, f"{bid}.canonical_theorems", require_root="corpus")
    song_entry = blocker.get("song_entry")
    if song_entry is not None:
        _check_source_entry(song_entry, f"{bid}.song_entry")


def _validate_spec(
    text: object,
    row_ids: list[str],
    blocker_ids: list[str],
    hazard_ids: list[str],
    control_ids: list[str],
) -> None:
    _need(isinstance(text, str) and text.strip(), "spec document is empty")
    headings = [m.group(1).strip() for m in re.finditer(r"(?m)^#{1,6}\s+(.*)$", text)]
    for section in REQUIRED_SECTIONS:
        _need(
            any(section.lower() in heading.lower() for heading in headings),
            f"spec is missing section heading: {section}",
        )
    named = (
        [(token, "row") for token in row_ids]
        + [(token, "blocker") for token in blocker_ids]
        + [(token, "hazard") for token in hazard_ids]
        + [(token, "control") for token in control_ids]
    )
    for token, label in named:
        _need(
            re.search(
                r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])", text
            )
            is not None,
            f"spec does not name required {label} {token}",
        )


def validate(data: object, spec_text: str) -> None:
    """Raise :class:`CheckError` unless the profile is internally consistent."""
    _need(isinstance(data, dict), "profile root must be an object")
    _need(data.get("profile") == PROFILE_ID, f"profile id must be {PROFILE_ID}")
    _need(isinstance(data.get("version"), int), "version must be an integer")

    corpus = data.get("corpus")
    _need(isinstance(corpus, dict), "corpus metadata missing")
    for key in ("root", "git", "mathlib"):
        _nonempty(corpus.get(key), f"corpus.{key}")
    _need(CORPUS_ROOT.is_dir(), f"pinned corpus root does not resolve: {CORPUS_ROOT}")

    fields = data.get("fields")
    _need(
        isinstance(fields, list) and len(fields) == len(FIELDS),
        f"fields must have exactly {len(FIELDS)} entries",
    )
    field_keys = [f.get("key") for f in fields if isinstance(f, dict)]
    _need(field_keys == FIELDS, f"fields keys must be {FIELDS}, got {field_keys}")

    rows = data.get("rows")
    _need(isinstance(rows, list) and rows, "rows must be a nonempty list")
    seen_ids: set[str] = set()
    resource_kinds: set[str] = set()
    for row in rows:
        _validate_row(row, seen_ids, resource_kinds)
    missing_rows = [r for r in REQUIRED_ROWS if r not in seen_ids]
    _need(not missing_rows, f"required selected rows dropped: {missing_rows}")
    missing_kinds = sorted(REQUIRED_RESOURCE_KINDS - resource_kinds)
    _need(
        not missing_kinds,
        f"resource kinds must include {sorted(REQUIRED_RESOURCE_KINDS)}, missing {missing_kinds}",
    )
    _need(
        not (resource_kinds & FORBIDDEN_RESOURCE_KINDS),
        f"forbidden resource kinds present: {sorted(resource_kinds & FORBIDDEN_RESOURCE_KINDS)}",
    )

    blockers = data.get("unmatched_blockers")
    _need(
        isinstance(blockers, list) and blockers,
        "unmatched_blockers must be a nonempty list",
    )
    blocker_ids = [b.get("id") for b in blockers if isinstance(b, dict)]
    _need(len(blocker_ids) == len(set(blocker_ids)), "unmatched blocker ids must be unique")
    missing_blockers = [b for b in REQUIRED_BLOCKERS if b not in blocker_ids]
    _need(not missing_blockers, f"unmatched blockers dropped: {missing_blockers}")
    for blocker in blockers:
        _validate_blocker(blocker)

    hazards = data.get("hazards")
    _need(isinstance(hazards, list) and hazards, "hazards must be a nonempty list")
    hazard_ids = [h.get("id") for h in hazards if isinstance(h, dict)]
    _need(len(hazard_ids) == len(set(hazard_ids)), "hazard ids must be unique")
    missing_hazards = [h for h in REQUIRED_HAZARDS if h not in hazard_ids]
    _need(not missing_hazards, f"hazards dropped: {missing_hazards}")
    for hazard in hazards:
        _need(isinstance(hazard, dict), "each hazard must be an object")
        _nonempty(hazard.get("id"), "hazard.id")
        _nonempty(hazard.get("statement"), f"hazard {hazard.get('id')}.statement")
        _nonempty(hazard.get("consequence"), f"hazard {hazard.get('id')}.consequence")

    controls = data.get("required_controls")
    _need(isinstance(controls, list) and controls, "required_controls must be a nonempty list")
    control_ids = [c.get("id") for c in controls if isinstance(c, dict)]
    _need(len(control_ids) == len(set(control_ids)), "control ids must be unique")
    missing_controls = [c for c in REQUIRED_CONTROLS if c not in control_ids]
    _need(not missing_controls, f"required controls dropped: {missing_controls}")
    for control in controls:
        _need(isinstance(control, dict), "each control must be an object")
        _nonempty(control.get("id"), "control.id")
        _nonempty(control.get("target"), f"control {control.get('id')}.target")
        _nonempty(control.get("description"), f"control {control.get('id')}.description")

    order = data.get("integration_order")
    _need(isinstance(order, list) and order, "integration_order must be a nonempty list")
    _need(
        all(isinstance(item, str) and item.strip() for item in order),
        "integration_order entries must be nonempty strings",
    )
    covered = data.get("not_covered")
    _need(isinstance(covered, list) and covered, "not_covered must be a nonempty list")

    _validate_spec(spec_text, list(seen_ids), blocker_ids, hazard_ids, control_ids)


def _control_cases(data: dict) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []

    mutated = copy.deepcopy(data)
    mutated["rows"][0].pop("canonical", None)
    cases.append(("remove required field canonical from row E1", mutated))

    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"] = [
        b for b in mutated["unmatched_blockers"] if b.get("id") != "BE1"
    ]
    cases.append(("remove unmatched blocker BE1", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"] = mutated["rows"][1:]
    cases.append(("remove selected row E1", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["theorems"][0]["name"] = (
        "Apeiron.Combinator.Comb.does_not_exist"
    )
    cases.append(("rename canonical theorem to a nonexistent name", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["song"][0]["file"] = (
        "song/src/prover/ring/does_not_exist.slang"
    )
    cases.append(("point a song entry at a missing file", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["song"][0]["kind"] = "fuel"
    cases.append(("rename a resource kind to generic fuel", mutated))

    mutated = copy.deepcopy(data)
    mutated["hazards"] = [h for h in mutated["hazards"] if h.get("id") != "HE1"]
    cases.append(("remove hazard HE1", mutated))

    mutated = copy.deepcopy(data)
    mutated["fields"] = mutated["fields"][:-1]
    cases.append(("drop a fields entry", mutated))

    mutated = copy.deepcopy(data)
    mutated["required_controls"] = [
        c for c in mutated["required_controls"] if c.get("id") != "C1"
    ]
    cases.append(("remove required control C1", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["representation_invariant"]["statement"] = "no terms here"
    cases.append(("drop a representation term from the statement", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["input_grammar"].pop("trailing_input_policy", None)
    cases.append(("remove the trailing-input policy", mutated))

    return cases


def run_controls(data: dict, spec_text: str) -> list[tuple[str, bool, str]]:
    """Baseline must pass; every in-memory mutation must be rejected."""
    validate(data, spec_text)
    results: list[tuple[str, bool, str]] = []
    for name, mutated in _control_cases(data):
        try:
            validate(mutated, spec_text)
        except CheckError as exc:
            results.append((name, True, str(exc)))
        else:
            results.append((name, False, "mutation was NOT rejected"))
    return results


def _load() -> tuple[dict, str]:
    _need(ROWS_PATH.is_file(), f"profile rows missing: {ROWS_PATH}")
    _need(SPEC_PATH.is_file(), f"profile spec missing: {SPEC_PATH}")
    data = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    return data, SPEC_PATH.read_text(encoding="utf-8")


def _summary(data: dict) -> str:
    row_ids = [row["id"] for row in data["rows"]]
    blocker_ids = [b["id"] for b in data["unmatched_blockers"]]
    hazard_ids = [h["id"] for h in data["hazards"]]
    control_ids = [c["id"] for c in data["required_controls"]]
    theorems = sum(len(row["canonical"]["theorems"]) for row in data["rows"])
    return (
        f"rows={row_ids} blockers={blocker_ids} hazards={hazard_ids} "
        f"controls={control_ids} canonical_theorems={theorems}"
    )


def _cmd_check(data: dict, spec_text: str) -> int:
    try:
        validate(data, spec_text)
    except CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS (specification consistency only): {_summary(data)}")
    return 0


def _cmd_controls(data: dict, spec_text: str, self_test: bool) -> int:
    try:
        validate(data, spec_text)
    except CheckError as exc:
        print(f"FAIL: baseline profile is inconsistent: {exc}", file=sys.stderr)
        return 1
    results = run_controls(data, spec_text)
    rejected = 0
    for name, ok, detail in results:
        if ok:
            rejected += 1
            print(f"control RED (rejected): {name}")
        else:
            print(f"control GREEN (NOT rejected): {name}: {detail}", file=sys.stderr)
    if self_test:
        mandated = [
            name
            for name, _, _ in results
            if "remove required field" in name or "remove unmatched blocker" in name
        ]
        _need(
            len(mandated) == 2,
            "self-test requires the required-field and unmatched-blocker controls",
        )
    if rejected != len(results):
        print(
            f"FAIL: {len(results) - rejected} of {len(results)} controls were not rejected",
            file=sys.stderr,
        )
        return 1
    label = "self-test" if self_test else "controls"
    print(
        f"PASS ({label}): baseline valid, {rejected}/{len(results)} mutations rejected; "
        "specification consistency only"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate the written spec")
    group.add_argument(
        "--controls", action="store_true", help="validate and require all mutations rejected"
    )
    group.add_argument(
        "--self-test", action="store_true", help="controls plus the mandated mutation assertions"
    )
    args = parser.parse_args(argv)
    try:
        data, spec_text = _load()
    except (CheckError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.check:
        return _cmd_check(data, spec_text)
    return _cmd_controls(data, spec_text, self_test=args.self_test)


if __name__ == "__main__":
    raise SystemExit(main())
