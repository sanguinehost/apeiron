#!/usr/bin/env python3
"""Consistency checker for the W17-T1 joint primitive profile.

This tool establishes **specification consistency only**. It verifies that
``docs/spec/joint-primitives.md`` and ``tools/joint/profile_rows.json`` are
internally complete and that every artifact they name resolves:

* every joint profile row carries all seven required fields;
* every named canonical theorem resolves as a declaration in the pinned
  canonical corpus (``../sanguine-apeiron/proof``), and every named bridge
  refinement resolves in ``proof/``;
* every named Rust and Song entry exists as a declaration in its source file;
* every required unmatched blocker and hazard is present, and none is dropped;
* the resource accounting names the five distinct kinds and no generic fuel.

It does **not** establish semantic correspondence, refinement or proof. A green
run says the specification is well formed and source-connected by name; it says
nothing about whether any row's refinement has been proved.

Usage::

    python3 tools/joint/profile_check.py --check
    python3 tools/joint/profile_check.py --controls
    python3 tools/joint/profile_check.py --self-test

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
SPEC_PATH = REPO / "docs" / "spec" / "joint-primitives.md"
ROWS_PATH = HERE.parent / "profile_rows.json"
CORPUS_ROOT = REPO.parent / "sanguine-apeiron" / "proof"
BRIDGE_ROOT = REPO / "proof"

PROFILE_ID = "APEIRON-PROGRAMME-W17-T1"

SEVEN_FIELDS = [
    "canonical",
    "entry_points",
    "representation_invariant",
    "admission",
    "behavior",
    "resources",
    "refinement",
]

REQUIRED_ROWS = ["R1", "R2"]
REQUIRED_BLOCKERS = ["B1", "B2", "B3", "B4", "B5", "B6"]
REQUIRED_HAZARDS = ["H1", "H2", "H3", "H4", "H5"]
REQUIRED_SECTIONS = [
    "Seven required fields",
    "Selected boundary",
    "Unmatched counterparts",
    "Hazards",
    "Integration order",
    "Not covered",
    "Checker",
]
REQUIRED_RESOURCE_KINDS = {
    "arithmetic_ops",
    "probes",
    "nodes",
    "stack_entries",
    "input_consumption",
}
BEHAVIOR_KEYS = [
    "success",
    "unsupported_input",
    "resource_exhaustion",
    "raw_return_encoding",
    "post_state",
]
ADMISSION_SIDES = ["rust", "song"]

LEAN_DECL = (
    r"(?m)^\s*(?:noncomputable\s+|private\s+|protected\s+|@\[[^\]]*\]\s*)*"
    r"(?:theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
)
RUST_DECL = (
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|unsafe\s+)?"
    r"(?:fn|struct|enum|trait|union|type|mod)\s+"
)
RUST_IMPL = r"(?m)^\s*impl\b[^\n{]*\b"
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


def _check_rust(text: str, symbol: str) -> bool:
    return (
        re.search(RUST_DECL + re.escape(symbol) + r"\b", text) is not None
        or re.search(RUST_IMPL + re.escape(symbol) + r"\b", text) is not None
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
    _need(lang in ("rust", "slang"), f"{where}.lang must be rust or slang")
    path = _resolve_path(rel, where)
    text = path.read_text(encoding="utf-8")
    found = _check_rust(text, symbol) if lang == "rust" else _check_slang(text, symbol)
    _need(found, f"{where}: {lang} entry {symbol} not found in {rel}")


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
    for field in SEVEN_FIELDS:
        _need(field in row, f"{rid} drops required field {field}")
        value = row[field]
        _need(value is not None, f"{rid}.{field} is null")
        if isinstance(value, str):
            _nonempty(value, f"{rid}.{field}")
        elif isinstance(value, (list, dict)):
            _need(len(value) > 0, f"{rid}.{field} is empty")

    canonical = row["canonical"]
    for key in ("object", "observation_relation", "scalar_type", "module"):
        _nonempty(canonical.get(key), f"{rid}.canonical.{key}")
    theorems = canonical.get("theorems")
    _need(isinstance(theorems, list) and theorems, f"{rid}.canonical.theorems empty")
    for theorem in theorems:
        _check_lean_ref(theorem, f"{rid}.canonical.theorems", require_root="corpus")

    entry_points = row["entry_points"]
    for side in ("rust", "song"):
        entries = entry_points.get(side)
        _need(
            isinstance(entries, list) and entries,
            f"{rid}.entry_points.{side} must be nonempty",
        )
        for entry in entries:
            _check_source_entry(entry, f"{rid}.entry_points.{side}")

    invariant = row["representation_invariant"]
    for key in ("constructor", "statement"):
        _nonempty(invariant.get(key), f"{rid}.representation_invariant.{key}")

    admission = row["admission"]
    for side in ADMISSION_SIDES:
        sub = admission.get(side)
        _need(isinstance(sub, dict), f"{rid}.admission.{side} must be an object")
        _nonempty(sub.get("predicate"), f"{rid}.admission.{side}.predicate")
        _nonempty(sub.get("source"), f"{rid}.admission.{side}.source")

    behavior = row["behavior"]
    for key in BEHAVIOR_KEYS:
        _nonempty(behavior.get(key), f"{rid}.behavior.{key}")

    resources = row["resources"]
    for side in ("rust", "song"):
        entries = resources.get(side)
        _need(
            isinstance(entries, list) and entries,
            f"{rid}.resources.{side} must be nonempty",
        )
        for entry in entries:
            kind = entry.get("kind") if isinstance(entry, dict) else None
            _need(
                kind in REQUIRED_RESOURCE_KINDS,
                f"{rid}.resources.{side} invalid kind {kind!r}",
            )
            _nonempty(entry.get("accounting"), f"{rid}.resources.{side}[{kind}]")
            resource_kinds.add(kind)

    refinement = row["refinement"]
    _nonempty(refinement.get("source_connected"), f"{rid}.refinement.source_connected")
    _nonempty(refinement.get("rejecting_control"), f"{rid}.refinement.rejecting_control")
    artifacts = refinement.get("artifacts")
    _need(
        isinstance(artifacts, list) and artifacts,
        f"{rid}.refinement.artifacts must be nonempty",
    )
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
    song_entry = blocker.get("song_entry")
    if song_entry is not None:
        _check_source_entry(song_entry, f"{bid}.song_entry")


def _validate_spec(
    text: object,
    row_ids: list[str],
    blocker_ids: list[str],
    hazard_ids: list[str],
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

    fields = data.get("seven_fields")
    _need(
        isinstance(fields, list) and len(fields) == 7,
        "seven_fields must have exactly seven entries",
    )
    field_keys = [f.get("key") for f in fields if isinstance(f, dict)]
    _need(
        field_keys == SEVEN_FIELDS,
        f"seven_fields keys must be {SEVEN_FIELDS}, got {field_keys}",
    )

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

    order = data.get("integration_order")
    _need(isinstance(order, list) and order, "integration_order must be a nonempty list")
    _need(
        all(isinstance(item, str) and item.strip() for item in order),
        "integration_order entries must be nonempty strings",
    )
    covered = data.get("not_covered")
    _need(isinstance(covered, list) and covered, "not_covered must be a nonempty list")

    _validate_spec(spec_text, list(seen_ids), blocker_ids, hazard_ids)


def _control_cases(data: dict) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []

    mutated = copy.deepcopy(data)
    mutated["rows"][0].pop("canonical", None)
    cases.append(("remove required field canonical from row R1", mutated))

    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"] = [
        b for b in mutated["unmatched_blockers"] if b.get("id") != "B1"
    ]
    cases.append(("remove unmatched blocker B1", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"] = mutated["rows"][1:]
    cases.append(("remove selected row R1", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["theorems"][0]["name"] = (
        "Apeiron.Table.does_not_exist"
    )
    cases.append(("rename canonical theorem to a nonexistent name", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["rust"][0]["file"] = "src/does_not_exist.rs"
    cases.append(("point a source entry at a missing file", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["rust"][0]["kind"] = "fuel"
    cases.append(("rename a resource kind to generic fuel", mutated))

    mutated = copy.deepcopy(data)
    mutated["hazards"] = [h for h in mutated["hazards"] if h.get("id") != "H1"]
    cases.append(("remove hazard H1", mutated))

    mutated = copy.deepcopy(data)
    mutated["seven_fields"] = mutated["seven_fields"][:6]
    cases.append(("drop a seven_fields entry", mutated))

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
    theorems = sum(len(row["canonical"]["theorems"]) for row in data["rows"])
    return (
        f"rows={row_ids} blockers={blocker_ids} hazards={hazard_ids} "
        f"canonical_theorems={theorems}"
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
