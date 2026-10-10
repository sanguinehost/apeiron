#!/usr/bin/env python3
"""Consistency checker for the W18-T3 joint table subprofile.

This tool establishes **specification consistency only**. It verifies that
``docs/contracts/joint-table.md`` and ``tools/joint/table_profile_rows.json`` are
internally complete and that every artifact they name resolves:

* every table subprofile row carries all six required fields and stays
  ``selected-open`` (a ``closed`` row is rejected);
* every named canonical authority target resolves as the named declaration
  (``theorem``/``lemma``, ``def``/``abbrev``, ...) in the pinned canonical
  corpus (``../sanguine-apeiron/proof``) under its exact opened namespace, and
  every named bridge artifact resolves in ``proof/``;
* every named Rust and Song entry exists as a concrete executing declaration
  in its source file (a ``law``/``carrier``/``spec`` header alone is not an
  entry); a Rust method belongs to the named type's ``impl``, a generic type
  parameter is never a self type or a resolvable symbol, a bare symbol resolves
  only to an item the file declares (``fn``/``struct``/``enum``/``trait``/
  ``union``/``type``/``mod``/``const``/``static``) so a word that appears only in
  an ``impl`` header, a generic bound or a ``where`` clause is rejected, and a
  ``<T as Trait>`` qualifier must name a trait the file implements for ``T``;
* the corpus revision, Mathlib revision and per-source sha256 digests match the
  resolved checkout;
* every implementation of every row carries its required resource kinds
  separately (a missing account on one side is rejected), and no generic fuel
  kind is admitted;
* every named reference, including control targets and any file/symbol
  reference in the row text, is parsed under the strict grammar documented at
  ``FILE_PATH`` and resolves; a reference outside the grammar is rejected as
  malformed, never truncated or split;
* every required row, hazard and rejecting control is present, and none is
  dropped;
* the spec carries the actual records and fields (row headings and titles,
  required-field records, hazard records, control records), not incidental
  mentions.

It does **not** establish semantic correspondence, refinement, proof or the
rejecting controls' execution. A green run says the specification is well formed
and source-connected by name; it says nothing about whether any row's refinement
has been proved or whether any control goes red.

Usage::

    python3 tools/joint/table_profile_check.py --check
    python3 tools/joint/table_profile_check.py --controls
    python3 tools/joint/table_profile_check.py --self-test

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
    _check_rust,
    _check_iota,
    _need,
    _nonempty,
    _resolve_path,
)

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SPEC_PATH = REPO / "docs" / "contracts" / "joint-table.md"
ROWS_PATH = HERE.parent / "table_profile_rows.json"

PROFILE_ID = "APEIRON-PROGRAMME-W18-T3"

REQUIRED_FIELDS = [
    "canonical",
    "rust",
    "song",
    "admission",
    "resources",
    "refinement",
]

REQUIRED_ROWS = ["T1", "T2", "T3", "T4", "T5", "T6"]
REQUIRED_HAZARDS = ["H1", "H2", "H3", "H4"]
REQUIRED_CONTROLS = ["C1", "C2", "C3", "C4", "C5", "C6"]
REQUIRED_SECTIONS = [
    "Required fields",
    "Obligation rows",
    "Separate admission",
    "Hazards",
    "Rejecting controls",
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
# Per-implementation resource kinds fixed by the subprofile. Every row's Rust
# and Song account must carry its own required kinds; a kind present in one
# account is never accepted as covering the other. This is deliberately not a
# union across rows or across implementations.
REQUIRED_RESOURCE_KINDS_BY_ROW = {
    "T1": {
        "rust": {"nodes", "stack_entries"},
        "song": {"nodes", "stack_entries"},
    },
    "T2": {
        "rust": {"probes", "arithmetic_ops"},
        "song": {"probes", "arithmetic_ops"},
    },
    "T3": {
        "rust": {"arithmetic_ops", "probes"},
        "song": {"arithmetic_ops", "probes"},
    },
    "T4": {
        "rust": {"probes", "arithmetic_ops"},
        "song": {"probes", "arithmetic_ops"},
    },
    "T5": {
        "rust": {
            "arithmetic_ops",
            "probes",
            "nodes",
            "stack_entries",
            "input_consumption",
        },
        "song": {
            "arithmetic_ops",
            "probes",
            "nodes",
            "stack_entries",
            "input_consumption",
        },
    },
    "T6": {
        "rust": {"input_consumption", "nodes"},
        "song": {"input_consumption", "nodes"},
    },
}
FORBIDDEN_RESOURCE_KINDS = {"fuel"}
REQUIRED_CANONICAL_THEOREMS = {
    "Apeiron.Table.get_insertWith_self",
    "Apeiron.Table.get_insertWith_ne",
    "Apeiron.Table.get_build_acc",
    "Apeiron.Table.build_gauge_invariance",
}
REQUIRED_CANONICAL_DEFS = {
    "Apeiron.Table.HMap",
    "Apeiron.Table.get",
    "Apeiron.Table.insertWith",
    "Apeiron.Table.build",
}
ROW_RUST_KEYS = ["layout", "identity", "merge", "refusal", "bounds"]
ROW_SONG_KEYS = [
    "layout",
    "stride",
    "keys",
    "probes",
    "merge",
    "transactionality",
]
ADMISSION_SIDES = ["rust", "song"]
SEPARATE_ADMISSION_KEYS = ["rust", "song", "joint", "unequal_capacities"]
CONTROL_KEYS = ["name", "target", "mutation", "expected_red", "grade"]


def _check_lean_ref(
    ref: object,
    where: str,
    require_root: str | None = None,
    require_kind: bool = True,
) -> None:
    jointlib.check_lean_ref(
        ref,
        where,
        require_root=require_root,
        require_kind=require_kind,
        gauge_hypotheses=True,
    )


def _check_source_entry(entry: object, where: str, want_lang: str) -> None:
    _need(isinstance(entry, dict), f"{where} entries must be objects")
    symbol = entry.get("symbol")
    rel = entry.get("file")
    lang = entry.get("lang")
    _nonempty(symbol, f"{where}.symbol")
    _nonempty(rel, f"{where}.file")
    _need(lang in ("rust", "iota"), f"{where}.lang must be rust or iota")
    _need(lang == want_lang, f"{where}: expected lang {want_lang}, got {lang}")
    path = _resolve_path(rel, where)
    text = path.read_text(encoding="utf-8")
    found = _check_rust(text, symbol) if lang == "rust" else _check_iota(text, symbol)
    _need(found, f"{where}: {lang} entry {symbol} not found in {rel}")


def _validate_row(
    row: object,
    seen_ids: set[str],
    canonical_theorems: set[str],
    canonical_defs: set[str],
) -> None:
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
    for field in REQUIRED_FIELDS:
        _need(field in row, f"{rid} drops required field {field}")
        value = row[field]
        _need(value is not None, f"{rid}.{field} is null")
        if isinstance(value, str):
            _nonempty(value, f"{rid}.{field}")
        elif isinstance(value, (list, dict)):
            _need(len(value) > 0, f"{rid}.{field} is empty")

    canonical = row["canonical"]
    _need(isinstance(canonical, dict), f"{rid}.canonical must be an object")
    for key in ("object", "module"):
        _nonempty(canonical.get(key), f"{rid}.canonical.{key}")
    targets = canonical.get("targets")
    _need(
        isinstance(targets, list) and targets,
        f"{rid}.canonical.targets must be nonempty",
    )
    for target in targets:
        _check_lean_ref(target, f"{rid}.canonical.targets", require_root="corpus")
        name = target["name"]
        if target.get("kind") == "theorem":
            canonical_theorems.add(name)
        else:
            canonical_defs.add(name)

    rust = row["rust"]
    _need(isinstance(rust, dict), f"{rid}.rust must be an object")
    for key in ROW_RUST_KEYS:
        _nonempty(rust.get(key), f"{rid}.rust.{key}")
    rust_entries = rust.get("entry_points")
    _need(
        isinstance(rust_entries, list) and rust_entries,
        f"{rid}.rust.entry_points must be nonempty",
    )
    for entry in rust_entries:
        _check_source_entry(entry, f"{rid}.rust.entry_points", "rust")

    song = row["song"]
    _need(isinstance(song, dict), f"{rid}.song must be an object")
    for key in ROW_SONG_KEYS:
        _nonempty(song.get(key), f"{rid}.song.{key}")
    song_entries = song.get("entry_points")
    _need(
        isinstance(song_entries, list) and song_entries,
        f"{rid}.song.entry_points must be nonempty",
    )
    for entry in song_entries:
        _check_source_entry(entry, f"{rid}.song.entry_points", "iota")

    admission = row["admission"]
    _need(isinstance(admission, dict), f"{rid}.admission must be an object")
    for side in ADMISSION_SIDES:
        sub = admission.get(side)
        _need(isinstance(sub, dict), f"{rid}.admission.{side} must be an object")
        _nonempty(sub.get("predicate"), f"{rid}.admission.{side}.predicate")
        _nonempty(sub.get("source"), f"{rid}.admission.{side}.source")

    # Per-implementation resource completeness. Each side must independently
    # carry every required kind; a kind declared on the other side, or in
    # another row, does not satisfy it.
    resources = row["resources"]
    _need(isinstance(resources, dict), f"{rid}.resources must be an object")
    required = REQUIRED_RESOURCE_KINDS_BY_ROW.get(rid, {})
    for side in ADMISSION_SIDES:
        entries = resources.get(side)
        _need(
            isinstance(entries, list) and entries,
            f"{rid}.resources.{side} must be nonempty",
        )
        declared: set[str] = set()
        for entry in entries:
            _need(
                isinstance(entry, dict),
                f"{rid}.resources.{side} entries must be objects",
            )
            kind = entry.get("kind")
            _need(
                kind not in FORBIDDEN_RESOURCE_KINDS,
                f"{rid}.resources.{side}: forbidden resource kind {kind!r}",
            )
            _need(
                kind in REQUIRED_RESOURCE_KINDS,
                f"{rid}.resources.{side} invalid kind {kind!r}",
            )
            _need(
                kind not in declared,
                f"{rid}.resources.{side} duplicates resource kind {kind!r}",
            )
            _nonempty(entry.get("accounting"), f"{rid}.resources.{side}[{kind}]")
            declared.add(kind)
        missing = sorted(required.get(side, set()) - declared)
        _need(
            not missing,
            f"{rid}.resources.{side} is missing required kinds {missing}",
        )

    refinement = row["refinement"]
    _need(isinstance(refinement, dict), f"{rid}.refinement must be an object")
    _nonempty(refinement.get("source_connected"), f"{rid}.refinement.source_connected")
    _nonempty(refinement.get("rejecting_control"), f"{rid}.refinement.rejecting_control")
    jointlib._check_text_refs(
        refinement["source_connected"], f"{rid}.refinement.source_connected"
    )
    artifacts = refinement.get("artifacts")
    _need(
        isinstance(artifacts, list) and artifacts,
        f"{rid}.refinement.artifacts must be nonempty",
    )
    for artifact in artifacts:
        _check_lean_ref(
            artifact,
            f"{rid}.refinement.artifacts",
            require_root="bridge",
            require_kind=False,
        )


def _validate_hazard(hazard: object) -> None:
    _need(isinstance(hazard, dict), "each hazard must be an object")
    hid = hazard.get("id")
    _nonempty(hid, "hazard.id")
    _nonempty(hazard.get("statement"), f"{hid}.statement")
    _nonempty(hazard.get("consequence"), f"{hid}.consequence")


def _validate_control(control: object, row_ids: set[str]) -> None:
    _need(isinstance(control, dict), "each control must be an object")
    cid = control.get("id")
    _nonempty(cid, "control.id")
    for key in CONTROL_KEYS:
        _nonempty(control.get(key), f"{cid}.{key}")
    targets = [
        token.strip()
        for token in re.split(r"[,/ ]+", control["target"])
        if token.strip()
    ]
    _need(bool(targets), f"{cid}.target must name at least one row id")
    # Every named target is checked, not only the first valid one.
    invalid = [token for token in targets if token not in row_ids]
    _need(not invalid, f"{cid}.target names unknown rows {invalid}")
    jointlib._check_text_refs(control["expected_red"], f"{cid}.expected_red")
    jointlib._check_text_refs(control["mutation"], f"{cid}.mutation")


ROW_FIELD_LABELS = {
    "canonical": "Canonical",
    "rust": "Rust",
    "song": "Song",
    "admission": "Admission",
    "resources": "Resources",
    "refinement": "Refinement",
}


def _validate_spec(
    text: object,
    row_ids: list[str],
    row_titles: list[str],
    row_statuses: dict[str, str],
    hazard_ids: list[str],
    control_ids: list[str],
    field_keys: list[str],
) -> None:
    jointspec.validate_spec_labeled(
        text,
        row_ids,
        row_titles,
        row_statuses,
        hazard_ids,
        control_ids,
        field_keys,
        ROW_FIELD_LABELS,
        REQUIRED_SECTIONS,
    )


def _verify_corpus(corpus: object) -> None:
    jointspec.verify_corpus(corpus)


def validate(data: object, spec_text: str) -> None:
    """Raise :class:`CheckError` unless the subprofile is internally consistent."""
    _need(isinstance(data, dict), "profile root must be an object")
    _need(data.get("profile") == PROFILE_ID, f"profile id must be {PROFILE_ID}")
    _need(isinstance(data.get("version"), int), "version must be an integer")

    _verify_corpus(data.get("corpus"))

    fields = data.get("fields")
    _need(
        isinstance(fields, list) and len(fields) == len(REQUIRED_FIELDS),
        f"fields must have exactly {len(REQUIRED_FIELDS)} entries",
    )
    field_keys = [f.get("key") for f in fields if isinstance(f, dict)]
    _need(
        field_keys == REQUIRED_FIELDS,
        f"fields keys must be {REQUIRED_FIELDS}, got {field_keys}",
    )

    rows = data.get("rows")
    _need(isinstance(rows, list) and rows, "rows must be a nonempty list")
    seen_ids: set[str] = set()
    canonical_theorems: set[str] = set()
    canonical_defs: set[str] = set()
    for row in rows:
        _validate_row(row, seen_ids, canonical_theorems, canonical_defs)
    missing_rows = [r for r in REQUIRED_ROWS if r not in seen_ids]
    _need(not missing_rows, f"required rows dropped: {missing_rows}")
    missing_theorems = sorted(REQUIRED_CANONICAL_THEOREMS - canonical_theorems)
    _need(
        not missing_theorems,
        f"canonical theorem targets missing: {missing_theorems}",
    )
    missing_defs = sorted(REQUIRED_CANONICAL_DEFS - canonical_defs)
    _need(not missing_defs, f"canonical authority declarations missing: {missing_defs}")

    separate = data.get("separate_admission")
    _need(isinstance(separate, dict), "separate_admission must be an object")
    for key in SEPARATE_ADMISSION_KEYS:
        _nonempty(separate.get(key), f"separate_admission.{key}")

    hazards = data.get("hazards")
    _need(isinstance(hazards, list) and hazards, "hazards must be a nonempty list")
    hazard_ids = [h.get("id") for h in hazards if isinstance(h, dict)]
    _need(len(hazard_ids) == len(set(hazard_ids)), "hazard ids must be unique")
    missing_hazards = [h for h in REQUIRED_HAZARDS if h not in hazard_ids]
    _need(not missing_hazards, f"hazards dropped: {missing_hazards}")
    for hazard in hazards:
        _validate_hazard(hazard)

    controls = data.get("controls")
    _need(isinstance(controls, list) and controls, "controls must be a nonempty list")
    control_ids = [c.get("id") for c in controls if isinstance(c, dict)]
    _need(len(control_ids) == len(set(control_ids)), "control ids must be unique")
    missing_controls = [c for c in REQUIRED_CONTROLS if c not in control_ids]
    _need(not missing_controls, f"rejecting controls dropped: {missing_controls}")
    for control in controls:
        _validate_control(control, seen_ids)

    for row in rows:
        ref = row["refinement"]["rejecting_control"]
        _need(
            ref in control_ids,
            f"{row['id']}.refinement.rejecting_control {ref} is not a named control",
        )

    covered = data.get("not_covered")
    _need(isinstance(covered, list) and covered, "not_covered must be a nonempty list")
    _need(
        all(isinstance(item, str) and item.strip() for item in covered),
        "not_covered entries must be nonempty strings",
    )

    row_statuses = {
        row["id"]: row["status"] for row in rows if isinstance(row, dict)
    }
    _validate_spec(
        spec_text,
        list(seen_ids),
        [row["title"] for row in rows],
        row_statuses,
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
    mutated["rows"][0]["resources"]["rust"] = [
        e
        for e in mutated["rows"][0]["resources"]["rust"]
        if e.get("kind") != "nodes"
    ]
    cases.append(
        (
            "class 1: drop a required resource account from one implementation of T1",
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
    mutated["corpus"]["sources"]["src/table.rs"] = "0" * 64
    cases.append(
        ("class 2: change a recorded source digest", mutated, spec_text)
    )

    # class 3: exact namespace matching.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"].append(
        {
            "name": "Table.Apeiron.HMap",
            "module": "Apeiron/Table/Basic.lean",
            "kind": "abbrev",
        }
    )
    cases.append(
        (
            "class 3: add a canonical target with a reversed namespace",
            mutated,
            spec_text,
        )
    )

    # class 4: a law/carrier/spec header is not an executing entry.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["song"]["entry_points"][0]["symbol"] = "mergeacc_comm"
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
            "class 5: delete the hazard record H1 from the spec",
            copy.deepcopy(data),
            jointspec.strip_record(spec_text, "H1") + "\nThe H1 hazard is discussed above.\n",
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
    cases.append(("class 6: set row T1 status to closed", mutated, spec_text))

    # class 7: every named reference is checked.
    mutated = copy.deepcopy(data)
    mutated["controls"][0]["target"] = "T1, T3, T99"
    cases.append(
        ("class 7: name an unknown row in a control target", mutated, spec_text)
    )

    # Retained structural controls.
    mutated = copy.deepcopy(data)
    mutated["rows"][0].pop("canonical", None)
    cases.append(
        ("remove required field canonical from row T1", mutated, spec_text)
    )

    mutated = copy.deepcopy(data)
    mutated["fields"] = mutated["fields"][:-1]
    cases.append(("drop a required field from the fields list", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"] = [r for r in mutated["rows"] if r.get("id") != "T1"]
    cases.append(("remove required row T1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["hazards"] = [h for h in mutated["hazards"] if h.get("id") != "H1"]
    cases.append(("remove required hazard H1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["controls"] = [c for c in mutated["controls"] if c.get("id") != "C1"]
    cases.append(("remove required control C1", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"][0]["name"] = (
        "Apeiron.Table.does_not_exist"
    )
    cases.append(("rename a canonical target to a nonexistent name", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"][1]["kind"] = "theorem"
    cases.append(("mislabel a canonical def target as a theorem", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["rust"]["entry_points"][0]["file"] = "src/does_not_exist.rs"
    cases.append(("point a rust entry at a missing file", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["song"]["entry_points"][0]["symbol"] = "tableDoesNotExist"
    cases.append(("point a iota entry at a missing symbol", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["rust"][0]["kind"] = "fuel"
    cases.append(("rename a resource kind to generic fuel", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["separate_admission"].pop("joint", None)
    cases.append(("remove the joint separate_admission clause", mutated, spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"] = [
        t for t in mutated["rows"][0]["canonical"]["targets"]
        if t.get("name") != "Apeiron.Table.get_build_acc"
    ]
    mutated["rows"][5]["canonical"]["targets"] = [
        t for t in mutated["rows"][5]["canonical"]["targets"]
        if t.get("name") != "Apeiron.Table.get_build_acc"
    ]
    cases.append(("drop the get_build_acc canonical theorem target", mutated, spec_text))

    # class 3: exact fully-qualified declaration name, not a merely opened prefix.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"].append(
        {
            "name": "Apeiron.HMap",
            "module": "Apeiron/Table/Basic.lean",
            "kind": "abbrev",
        }
    )
    cases.append(
        (
            "class 3: add a canonical target missing its Table namespace",
            mutated,
            spec_text,
        )
    )

    # class 5: Markdown field body, not only its label.
    cases.append(
        (
            "class 5: delete the T1 Markdown Rust field",
            copy.deepcopy(data),
            re.sub(r"\*\*Rust\.\*\*.*?(?=\*\*Song\.\*\*)", "", spec_text, count=1, flags=re.S),
        )
    )
    cases.append(
        (
            "class 5: empty the T1 Markdown Rust field body",
            copy.deepcopy(data),
            re.sub(
                r"(\*\*Rust\.\*\*).*?(?=\*\*Song\.\*\*)",
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
            "class 6: change the T1 Markdown row status to closed",
            copy.deepcopy(data),
            spec_text.replace("| selected-open |", "| closed |", 1),
        )
    )

    # class 7: a named reference added to a structured prose field.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["rust"]["layout"] = (
        mutated["rows"][0]["rust"]["layout"] + " src/does_not_exist.rs::missing_fn"
    )
    cases.append(
        (
            "class 7: add an unresolved named reference to the T1 Rust layout",
            mutated,
            spec_text,
        )
    )

    # class 7: a qualified reference is consumed in full; ``merge`` belongs to
    # ``WordMerge``, not ``WordHome``, so the trailing method is rejected.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["rust"]["layout"] = (
        mutated["rows"][0]["rust"]["layout"] + " src/table.rs::WordHome::merge"
    )
    cases.append(
        (
            "class 7: reject a qualified method bound to the wrong Rust "
            "implementation in the T1 Rust layout",
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
            "class 7: reject a bare symbol that is only an impl-header generic "
            "parameter",
            "src/table.rs::F",
        ),
        (
            "class 7: reject a bare symbol that is only an impl-header bound, "
            "not a declaration",
            "src/table.rs::Fn",
        ),
        (
            "class 7: reject a reference with a trailing member access",
            "proof/Denotation.lean::denote.missing",
        ),
    ):
        mutated = copy.deepcopy(data)
        mutated["rows"][0]["rust"]["layout"] = (
            mutated["rows"][0]["rust"]["layout"] + " " + reference
        )
        cases.append((label, mutated, spec_text))

    return cases


CLASS_MARKERS = [
    "class 1:",
    "class 2:",
    "class 3:",
    "class 4:",
    "class 5:",
    "class 6:",
    "class 7:",
]


def run_controls(data: dict, spec_text: str) -> list[tuple[str, bool, str]]:
    """Baseline must pass; every in-memory mutation must be rejected."""
    return jointspec.run_control_cases(validate, data, spec_text, _control_cases)


def _load() -> tuple[dict, str]:
    return jointspec.load_profile(ROWS_PATH, SPEC_PATH, noun="subprofile")


def _summary(data: dict) -> str:
    row_ids = [row["id"] for row in data["rows"]]
    hazard_ids = [h["id"] for h in data["hazards"]]
    control_ids = [c["id"] for c in data["controls"]]
    targets = sum(len(row["canonical"]["targets"]) for row in data["rows"])
    return (
        f"rows={row_ids} hazards={hazard_ids} controls={control_ids} "
        f"canonical_targets={targets}"
    )


def main(argv: list[str] | None = None) -> int:
    return jointspec.run_cli(
        __doc__,
        _load,
        validate,
        run_controls,
        _summary,
        noun="subprofile",
        class_markers=tuple(CLASS_MARKERS),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
