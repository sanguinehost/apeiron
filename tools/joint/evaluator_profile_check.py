#!/usr/bin/env python3
"""Consistency checker for the W18-T2 joint evaluator profile.

This tool establishes **specification consistency only**. It verifies that
``docs/contracts/joint-evaluator.md`` and ``tools/joint/evaluator_profile_rows.json``
are internally complete and that every artifact they name resolves:

* every evaluator profile row carries all eight required fields and stays
  ``selected-open`` (a ``closed`` row is rejected);
* every named canonical theorem and canonical entry point resolves as a
  declaration in the pinned canonical corpus (``../sanguine-apeiron/proof``)
  under its exact opened namespace, and every named Song entry exists as a
  concrete executing declaration in its source file (a ``law``/``carrier``/
  ``spec`` header alone is not an entry);
* the corpus revision, Mathlib revision and per-source sha256 digests match the
  resolved checkout;
* every implementation of every row carries its required resource kinds
  separately (a missing account on one side is rejected), and no generic fuel
  kind is admitted;
* every named reference, including blocker ``counterpart``/``reason`` entries
  and control targets, is parsed under the strict grammar documented at
  ``FILE_PATH`` and checked, not only the selected structured references; a
  reference outside the grammar is rejected as malformed, never truncated or
  split;
* every required unmatched blocker, hazard and control is present, and none is
  dropped;
* the spec carries the actual records and fields (row headings and titles,
  required-field records, blocker records, hazard records, control records),
  not incidental mentions.

It does **not** establish semantic correspondence, refinement or proof. A green
run says the specification is well formed and source-connected by name; it says
nothing about whether any row's refinement has been proved.

Usage::

    python3 tools/joint/evaluator_profile_check.py --check
    python3 tools/joint/evaluator_profile_check.py --controls
    python3 tools/joint/evaluator_profile_check.py --self-test

``--controls`` and ``--self-test`` mutate a deep copy of the loaded profile and
the spec text in memory and require each mutation to be rejected. They never
touch the working tree.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import jointlib
import jointspec
from jointlib import (
    CheckError,
    _check_slang,
    _need,
    _nonempty,
    _resolve_path,
)

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SPEC_PATH = REPO / "docs" / "contracts" / "joint-evaluator.md"
ROWS_PATH = HERE.parent / "evaluator_profile_rows.json"

# Named artifacts deliberately absent from the pinned corpus. A reference that
# does not resolve is accepted only when its exact path is listed here; the
# exception is attached to the reference path, not to free text.
MISSING_ARTIFACTS = set()
jointlib.MISSING_ARTIFACTS = MISSING_ARTIFACTS
# This profile resolves no ``.rs`` reference: a ``.rs`` path is scanned as a
# Lean declaration, matching the evaluator profile's grammar.
jointlib._CHECK_RUST = None

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
# Per-implementation resource kinds fixed by the subprofile. Each side must
# independently carry its own required kinds; a kind present on the canonical
# side is never accepted as covering the Song implementation, and vice versa.
REQUIRED_RESOURCE_KINDS_BY_ROW = {
    "E1": {
        "canonical": {"reduction_steps"},
        "song": {
            "passes",
            "nodes",
            "stack_entries",
            "parser_entries",
            "input_bytes",
            "arith_ops",
        },
        "rust": set(),
    },
    "E2": {
        "canonical": {"rows", "arith_ops"},
        "song": {"rows", "columns", "limbs", "arith_ops", "input_bytes"},
        "rust": set(),
    },
}
BEHAVIOR_KEYS = [
    "success",
    "unsupported_input",
    "resource_exhaustion",
    "raw_return_encoding",
    "post_state",
]
INPUT_GRAMMAR_KEYS = ["grammar", "byte_consumption", "trailing_input_policy"]
ADMISSION_SIDES = ["canonical", "song"]


def _check_lean_ref(ref: object, where: str, require_root: str) -> None:
    jointlib.check_lean_ref(ref, where, require_root=require_root)


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
    pin = jointlib._pinned_corpus_rev()
    _need(pin is not None, f"{where}: cannot resolve the corpus dependency revision")
    for module in modules:
        _nonempty(module, f"{where} entry")
        blob = jointlib._repo_blob(jointlib.CORPUS_ROOT, pin, module)
        _need(
            blob is not None,
            f"{where}: module does not resolve at the pin: corpus:{module}",
        )


def _validate_resources(resources: object, rid: str) -> None:
    _need(isinstance(resources, dict), f"{rid}.resources must be an object")
    required_by_side = REQUIRED_RESOURCE_KINDS_BY_ROW.get(rid, {})
    for side in ("canonical", "song", "rust"):
        entries = resources.get(side)
        _need(
            isinstance(entries, list),
            f"{rid}.resources.{side} must be a list (possibly empty)",
        )
        declared: set[str] = set()
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
            _need(
                kind not in declared,
                f"{rid}.resources.{side} duplicates resource kind {kind!r}",
            )
            _nonempty(entry.get("accounting"), f"{rid}.resources.{side}[{kind}]")
            declared.add(kind)
        missing = sorted(required_by_side.get(side, set()) - declared)
        _need(
            not missing,
            f"{rid}.resources.{side} is missing required kinds {missing}",
        )
        if not entries:
            _nonempty(
                resources.get(f"{side}_absent"),
                f"{rid}.resources.{side}_absent",
            )


def _validate_row(row: object, seen_ids: set[str]) -> None:
    _need(isinstance(row, dict), "each row must be an object")
    rid = row.get("id")
    _nonempty(rid, "row.id")
    _need(rid not in seen_ids, f"duplicate row id {rid}")
    seen_ids.add(rid)
    _nonempty(row.get("title"), f"{rid}.title")
    status = row.get("status")
    _need(
        status == "selected-open",
        f"{rid}.status must be selected-open, got {status!r}",
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

    _validate_resources(row["resources"], rid)

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


def _check_blocker_refs(blocker: dict, bid: str) -> None:
    jointlib._check_text_refs(blocker.get("counterpart", ""), f"{bid}.counterpart")
    jointlib._check_text_refs(blocker.get("reason", ""), f"{bid}.reason")


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
    # Every named reference in the free-text counterpart and reason is checked,
    # not only the selected structured references above.
    _check_blocker_refs(blocker, bid)


def _validate_control(control: object, row_ids: set[str]) -> None:
    _need(isinstance(control, dict), "each control must be an object")
    cid = control.get("id")
    _nonempty(cid, "control.id")
    _nonempty(control.get("target"), f"{cid}.target")
    _nonempty(control.get("description"), f"{cid}.description")
    targets = [
        token.strip()
        for token in re.split(r"[,/ ]+", control["target"])
        if token.strip()
    ]
    _need(bool(targets), f"{cid}.target must name at least one row id")
    invalid = [token for token in targets if token not in row_ids]
    _need(not invalid, f"{cid}.target names unknown rows {invalid}")


FIELD_LABELS = {
    "canonical": ("1", "Canonical object and observation relation"),
    "entry_points": ("2", "Entry points and source identity"),
    "representation_invariant": ("3", "Representation invariant"),
    "input_grammar": ("4", "Input grammar, byte consumption and trailing-input policy"),
    "admission": ("5", "Admission predicates"),
    "behavior": ("6", "Behaviour"),
    "resources": ("7", "Resource accounting"),
    "refinement": ("8", "Source-connected refinement and rejecting control"),
}


def _validate_spec(
    text: object,
    row_ids: list[str],
    row_titles: list[str],
    row_statuses: dict[str, str],
    blocker_ids: list[str],
    hazard_ids: list[str],
    control_ids: list[str],
    field_keys: list[str],
) -> None:
    jointspec.validate_spec_numbered(
        text,
        row_ids,
        row_titles,
        row_statuses,
        blocker_ids,
        hazard_ids,
        control_ids,
        field_keys,
        FIELD_LABELS,
        REQUIRED_SECTIONS,
    )


def _verify_corpus(corpus: object) -> None:
    jointspec.verify_corpus(corpus)


def validate(data: object, spec_text: str) -> None:
    """Raise :class:`CheckError` unless the profile is internally consistent."""
    _need(isinstance(data, dict), "profile root must be an object")
    _need(data.get("profile") == PROFILE_ID, f"profile id must be {PROFILE_ID}")
    _need(isinstance(data.get("version"), int), "version must be an integer")

    _verify_corpus(data.get("corpus"))

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
    for row in rows:
        _validate_row(row, seen_ids)
    missing_rows = [r for r in REQUIRED_ROWS if r not in seen_ids]
    _need(not missing_rows, f"required selected rows dropped: {missing_rows}")

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
        _validate_control(control, seen_ids)

    # E2 must record the extended canonical language (residue terminal) and BE4
    # must name the still-missing executing Song refinement; the canonical
    # completeness is not attributed to the Song reader alphabet.
    e2 = next((r for r in rows if isinstance(r, dict) and r.get("id") == "E2"), None)
    _need(e2 is not None, "E2 row missing for the residue-terminal consistency check")
    _need(
        "residue" in e2["canonical"]["types"],
        "E2 canonical types must record the residue terminal",
    )
    _need(
        "residue" in e2["canonical"]["object"],
        "E2 canonical object must record the residue terminal and uncapped completeness",
    )
    he5 = next((h for h in hazards if isinstance(h, dict) and h.get("id") == "HE5"), None)
    _need(
        he5 is not None and "residue" in str(he5.get("statement", "")),
        "HE5 must record the extended-language versus old-reader-alphabet distinction",
    )
    be4 = next((b for b in blockers if isinstance(b, dict) and b.get("id") == "BE4"), None)
    _need(
        be4 is not None and "executing Song" in str(be4.get("counterpart", "")),
        "BE4 must name the missing executing Song refinement",
    )

    order = data.get("integration_order")
    _need(isinstance(order, list) and order, "integration_order must be a nonempty list")
    _need(
        all(isinstance(item, str) and item.strip() for item in order),
        "integration_order entries must be nonempty strings",
    )
    covered = data.get("not_covered")
    _need(isinstance(covered, list) and covered, "not_covered must be a nonempty list")

    row_statuses = {
        row["id"]: row["status"] for row in rows if isinstance(row, dict)
    }
    _validate_spec(
        spec_text,
        list(seen_ids),
        [row["title"] for row in rows],
        row_statuses,
        blocker_ids,
        hazard_ids,
        control_ids,
        list(field_keys),
    )
    jointlib._check_data_text_refs(data, "profile")
    jointlib._check_text_refs(spec_text, "spec")


def _control_cases(data: dict, spec_text: str) -> list[tuple[str, dict, str]]:
    cases: list[tuple[str, dict, str]] = []

    # class 1: resource completeness is per implementation.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["song"] = [
        e
        for e in mutated["rows"][0]["resources"]["song"]
        if e.get("kind") != "arith_ops"
    ]
    cases.append(
        (
            "class 1: drop a required resource account from one implementation of E1",
            mutated,
            spec_text,
        )
    )

    # class 2: corpus pin verification.
    mutated = copy.deepcopy(data)
    mutated["corpus"]["git"] = "0" * 40
    cases.append(
        ("class 2: repin the corpus revision to a wrong value", mutated, spec_text)
    )

    mutated = copy.deepcopy(data)
    mutated["corpus"]["sources"]["song/src/prover/cone/certcheck.slang"] = "0" * 64
    cases.append(
        ("class 2: change a recorded source digest", mutated, spec_text)
    )

    # class 3: exact namespace matching.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["theorems"][0]["name"] = (
        "Comb.Apeiron.Combinator.I"
    )
    cases.append(
        ("class 3: reverse a canonical theorem namespace", mutated, spec_text)
    )

    # class 4: a law/carrier/spec header is not an executing entry.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["song"][0]["symbol"] = "k_select"
    cases.append(
        (
            "class 4: point a song entry at a law header only",
            mutated,
            spec_text,
        )
    )

    # class 5: the spec must carry the records and fields.
    cases.append(
        (
            "class 5: delete the blocker record BE1 from the spec",
            copy.deepcopy(data),
            jointspec.strip_record(spec_text, "BE1"),
        )
    )
    cases.append(
        (
            "class 5: delete the required-field record canonical from the spec",
            copy.deepcopy(data),
            jointspec.strip_field_row(spec_text, "canonical"),
        )
    )

    # class 6: no row closes.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["status"] = "closed"
    cases.append(("class 6: set row E1 status to closed", mutated, spec_text))

    # class 7: every named reference, including blocker entries, is checked.
    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"][0]["counterpart"] = (
        mutated["unmatched_blockers"][0]["counterpart"]
        + "; Rust src/does_not_exist.rs::missing_fn exists."
    )
    cases.append(
        (
            "class 7: name an unresolved file in a blocker counterpart",
            mutated,
            spec_text,
        )
    )

    # Retained structural controls.
    mutated = copy.deepcopy(data)
    mutated["rows"][0].pop("canonical", None)
    cases.append(("remove required field canonical from row E1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"] = [
        b for b in mutated["unmatched_blockers"] if b.get("id") != "BE1"
    ]
    cases.append(("remove unmatched blocker BE1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"] = mutated["rows"][1:]
    cases.append(("remove selected row E1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["theorems"][0]["name"] = (
        "Apeiron.Combinator.Comb.does_not_exist"
    )
    cases.append(("rename canonical theorem to a nonexistent name", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["song"][0]["file"] = (
        "song/src/prover/ring/does_not_exist.slang"
    )
    cases.append(("point a song entry at a missing file", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["song"][0]["kind"] = "fuel"
    cases.append(("rename a resource kind to generic fuel", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["hazards"] = [h for h in mutated["hazards"] if h.get("id") != "HE1"]
    cases.append(("remove hazard HE1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["fields"] = mutated["fields"][:-1]
    cases.append(("drop a fields entry", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["required_controls"] = [
        c for c in mutated["required_controls"] if c.get("id") != "C1"
    ]
    cases.append(("remove required control C1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["representation_invariant"]["statement"] = "no terms here"
    cases.append(("drop a representation term from the statement", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["input_grammar"].pop("trailing_input_policy", None)
    cases.append(("remove the trailing-input policy", mutated, spec_text))

    # class 3: exact fully-qualified declaration name, not a merely opened prefix.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["theorems"][0]["name"] = (
        "Apeiron.Combinator.I"
    )
    cases.append(
        (
            "class 3: rename a canonical theorem missing its Comb namespace",
            mutated,
            spec_text,
        )
    )

    # class 5: Markdown field body, not only its label.
    cases.append(
        (
            "class 5: delete the E1 Markdown representation-invariant field",
            copy.deepcopy(data),
            re.sub(
                r"\*\*3 · Representation invariant\.\*\*.*?(?=\*\*4 · Input grammar)",
                "",
                spec_text,
                count=1,
                flags=re.S,
            ),
        )
    )
    cases.append(
        (
            "class 5: empty the E1 Markdown representation-invariant body",
            copy.deepcopy(data),
            re.sub(
                r"(\*\*3 · Representation invariant\.\*\*).*?(?=\*\*4 · Input grammar)",
                r"\1\n\n",
                spec_text,
                count=1,
                flags=re.S,
            ),
        )
    )

    # class 6: the Markdown row status must agree with the JSON status.
    cases.append(
        (
            "class 6: change the E1 Markdown row status to closed",
            copy.deepcopy(data),
            spec_text.replace("| selected-open |", "| closed |", 1),
        )
    )

    # class 7: a named reference added to a structured prose field.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["refinement"]["source_connected"] = (
        mutated["rows"][0]["refinement"]["source_connected"]
        + " src/does_not_exist.rs::missing_fn"
    )
    cases.append(
        (
            "class 7: add an unresolved named reference to the E1 refinement",
            mutated,
            spec_text,
        )
    )

    # class 7: a qualified reference is consumed in full; the trailing segment
    # names no declared Song function, so the reference is rejected.
    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"][0]["counterpart"] = (
        mutated["unmatched_blockers"][0]["counterpart"]
        + " song/src/prover/cone/certcheck.slang::main::missing"
    )
    cases.append(
        (
            "class 7: reject a qualified reference whose trailing symbol is not "
            "declared in the named Song file",
            mutated,
            spec_text,
        )
    )

    # class 7: the strict reference grammar. A reference outside it is
    # rejected, never truncated or split.
    for label, reference in (
        (
            "class 7: reject a reference with an extra '::' segment",
            "src/table.rs::Table::new::get",
        ),
        (
            "class 7: reject a brace member that is not a single identifier "
            "or a Type::method pair",
            "src/table.rs::{Table::new::get}",
        ),
        (
            "class 7: reject a generic-qualified method bound to the wrong "
            "implementation",
            "src/table.rs::<WordHome as Merge<i64>>::merge",
        ),
        (
            "class 7: reject a generic-qualified method whose trait does not exist",
            "src/scalar.rs::<i64 as NonexistentTrait>::add",
        ),
        (
            "class 7: reject a method attributed to a generic type parameter",
            "src/table.rs::F::home",
        ),
        (
            "class 7: reject a reference with a trailing member access",
            "proof/Denotation.lean::denote.missing",
        ),
    ):
        mutated = copy.deepcopy(data)
        mutated["unmatched_blockers"][0]["counterpart"] = (
            mutated["unmatched_blockers"][0]["counterpart"] + " " + reference
        )
        cases.append((label, mutated, spec_text))

    # class 8: the E2 residue terminal and the BE4 executing-boundary record.
    e2_index = next(
        i for i, r in enumerate(data["rows"]) if r.get("id") == "E2"
    )
    mutated = copy.deepcopy(data)
    mutated["rows"][e2_index]["canonical"]["types"] = mutated["rows"][e2_index][
        "canonical"
    ]["types"].replace(" | residue (cert : ResidueCertificate)", "")
    cases.append(
        (
            "class 8: drop the residue terminal from the E2 canonical types",
            mutated,
            spec_text,
        )
    )
    he5_index = next(
        i for i, h in enumerate(data["hazards"]) if h.get("id") == "HE5"
    )
    mutated = copy.deepcopy(data)
    mutated["hazards"][he5_index]["statement"] = (
        "capped_certificate_complete is a capped-box existence theorem."
    )
    cases.append(
        (
            "class 8: replace HE5 with the old capped-box statement",
            mutated,
            spec_text,
        )
    )
    be4_index = next(
        i for i, b in enumerate(data["unmatched_blockers"]) if b.get("id") == "BE4"
    )
    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"][be4_index]["counterpart"] = (
        "Apeiron/Decider/Stream.lean and the executing stream refinement; both MISSING."
    )
    cases.append(
        (
            "class 8: restore the stale BE4 both-missing counterpart",
            mutated,
            spec_text,
        )
    )
    be4_index = next(
        i for i, b in enumerate(data["unmatched_blockers"]) if b.get("id") == "BE4"
    )
    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"][be4_index]["canonical_theorem"] = {
        "name": "Apeiron.Decider.checkCertStream_sound",
        "module": "Apeiron/Decider/Stream_Gone.lean",
        "root": "corpus",
    }
    cases.append(
        (
            "class 8: point the BE4 canonical theorem at a missing Stream module",
            mutated,
            spec_text,
        )
    )

    return cases


CLASS_MARKERS = [
    "class 1:",
    "class 2:",
    "class 3:",
    "class 4:",
    "class 5:",
    "class 6:",
    "class 7:",
    "class 8:",
]


def run_controls(data: dict, spec_text: str) -> list[tuple[str, bool, str]]:
    """Baseline must pass; every in-memory mutation must be rejected."""
    return jointspec.run_control_cases(validate, data, spec_text, _control_cases)


def _load() -> tuple[dict, str]:
    return jointspec.load_profile(ROWS_PATH, SPEC_PATH)


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


def main(argv: list[str] | None = None) -> int:
    return jointspec.run_cli(
        __doc__,
        _load,
        validate,
        run_controls,
        _summary,
        noun="profile",
        class_markers=tuple(CLASS_MARKERS),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
