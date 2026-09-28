#!/usr/bin/env python3
"""Consistency checker for the W18-T1 Slang source-entry fidelity specification.

This tool establishes **specification consistency only**. It verifies that
``docs/contracts/slang-source-entry.md`` and ``tools/joint/source_entry_cases.json``
are internally complete and that every artifact they name resolves:

* every selected entry carries all nine required fields and stays
  ``selected-open`` (a ``closed`` entry is rejected);
* every named Song source exists, its recorded sha256 matches the working-tree
  bytes and its recorded revision matches the resolved Song checkout and the
  blob at that revision;
* every named canonical target resolves as the exact fully-qualified
  declaration (``theorem``/``lemma``, ``def``/``abbrev``, ...) in the pinned
  canonical corpus under its exact opened namespace;
* the corpus revision, Mathlib revision and Apeiron/Song revisions match the
  resolved checkout;
* the explicit non-refinement statements are present: a copied implementation
  model, a differential suite and equal outputs are NOT a source-connected
  refinement, and a body change with an unchanged represented program must be
  rejected;
* the missing-artifact records name the absent checked Slang source-semantics
  link and the absent executing refinement;
* every named reference, including control targets and any file/symbol
  reference in the entry text, is parsed under the strict fail-closed grammar
  documented at ``jointlib.FILE_PATH`` and resolves; a reference outside the
  grammar is rejected as malformed, never truncated or split;
* every required entry and rejecting control is present, and none is dropped;
* the spec carries the actual entry records and fields (headings, titles,
  required-field records, non-refinement records, control records), not
  incidental mentions.

It does **not** establish semantic correspondence, refinement or proof, and it
never claims correspondence. A green run says the specification is well formed
and source-connected by name; it says nothing about whether any entry's
refinement has been proved or whether any control goes red.

The specification states the active source-refinement chain and marks the
retired route; this checker requires those statements and rejects a spec that
drops them. Specification consistency is green. Implementation acceptance is the
``tools/slang/refine.py`` exact-declaration probe on the unconditional
``Slang.SourceAudit.source_binding``, red until that theorem exists.

Usage::

    python3 tools/joint/source_entry_check.py --check
    python3 tools/joint/source_entry_check.py --controls
    python3 tools/joint/source_entry_check.py --self-test

``--controls`` and ``--self-test`` mutate a deep copy of the loaded cases and
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
    _sha256,
)

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SPEC_PATH = REPO / "docs" / "contracts" / "slang-source-entry.md"
CASES_PATH = HERE.parent / "source_entry_cases.json"

PROFILE_ID = "APEIRON-PROGRAMME-W18-T1"

REQUIRED_FIELDS = [
    "song",
    "represented_program",
    "observation_relation",
    "canonical_targets",
    "input_grammar",
    "state_effects",
    "refusal",
    "rejecting_control",
    "missing",
]
TEXT_FIELDS = [
    "represented_program",
    "observation_relation",
    "input_grammar",
    "state_effects",
    "refusal",
]
REQUIRED_ENTRIES = ["SE1", "SE2", "SE3", "SE4", "SE5", "SE6", "SE7", "SE8"]
REQUIRED_CONTROLS = ["SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8"]
REQUIRED_SECTIONS = [
    "Required fields",
    "Active route and acceptance",
    "Selected entries",
    "Non-refinement",
    "Missing artifacts",
    "Rejecting controls",
    "Not covered",
    "Checker",
]
NON_REFINEMENT_KEYS = [
    "copied_model",
    "differential_suite",
    "equal_outputs",
    "body_change_rule",
]
CONTROL_KEYS = ["name", "target", "mutation", "expected_red", "grade"]

# Exact canonical declarations that must appear as targets somewhere in the
# profile. A target is bound to its real fully-qualified namespace, so a merely
# opened prefix, a reversed namespace or a mislabelled kind is rejected.
REQUIRED_CANONICAL_TARGETS = {
    "Apeiron.Table.insertWith",
    "Apeiron.Table.get_build_acc",
    "Apeiron.Table.get",
    "Apeiron.Table.get_insertWith_self",
    "Apeiron.Table.get_insertWith_ne",
    "Apeiron.Table.HMap",
    "Apeiron.Table.build",
    "Apeiron.Combinator.Comb.headReduce",
    "Apeiron.Combinator.Comb.headReduce_sound",
    "Apeiron.Combinator.Comb.normalize",
    "Apeiron.Combinator.Comb.reduces_normalize",
    "Apeiron.Combinator.Comb.hasRedex",
    "Apeiron.Combinator.Comb.normal_of_hasRedex_false",
    "Apeiron.Combinator.Comb.verdict",
    "Apeiron.Combinator.Comb.verdict_decided_exact",
}

# Phrases the non-refinement and missing-artifact records must carry. They are
# matched against a punctuation-stripped, whitespace-collapsed, lowercased copy
# of the spec so a removed or reworded record is rejected.
NON_REFINEMENT_PHRASES = [
    "copied implementation model",
    "differential suite",
    "equal outputs",
    "not a source-connected refinement",
    "represented program",
    "must be rejected",
]
MISSING_PHRASES = [
    "checked slang source-semantics link",
    "executing refinement",
    "missing",
]

# The active source-refinement chain and its canonical roots. These phrases
# must appear in the spec; a spec that drops one is rejected.
ACTIVE_ROUTE_PHRASES = [
    "source arena",
    "checked references",
    "complete memo key",
    "face-correct addressing",
    "memo value",
    "fixed-shard store",
    "ordered fold",
    "slang.sourceaudit.source_binding",
    "apeiron.table.hmap",
    "apeiron.table.get",
    "apeiron.table.insertwith",
    "apeiron.table.build",
    "apeiron.trichotomy.anyring.ringcell",
    "apeiron.trichotomy.anyring.ringmul",
    "apeiron.trichotomy.anyring.ringnormform",
]

# The retired route the spec must mark as retired, never as the active
# mechanism. The tokens are stored lowercase; each line carries a retirement
# marker so the checker never presents a retired route as active.
RETIRED_ROUTE_PHRASES = [
    "retired",  # the retirement marker the spec must carry
    "decodetrace.run",  # retired, not active
    "named proof dag",  # retired, not active
    "segment lemma",  # retired, not active
]

# The acceptance distinction: consistency is green; implementation acceptance is
# the refine.py probe, red until the unconditional theorem exists.
ACCEPTANCE_PHRASES = [
    "specification consistency",
    "implementation acceptance",
    "tools/slang/refine.py",
    "slang.sourceaudit.source_binding",
]

ROW_FIELD_LABELS = {
    "song": "Song",
    "represented_program": "Represented program",
    "observation_relation": "Observation relation",
    "canonical_targets": "Canonical targets",
    "input_grammar": "Input grammar",
    "state_effects": "State and effects",
    "refusal": "Refusal and unsupported input",
    "rejecting_control": "Rejecting control",
    "missing": "Missing",
}


def _plain(text: str) -> str:
    """Lowercase, punctuation-stripped copy for record-presence checks."""
    return re.sub(r"\s+", " ", text.replace("`", "").replace("*", "")).strip().lower()


def _check_lean_ref(ref: object, where: str) -> None:
    jointlib.check_lean_ref(ref, where, require_root="corpus", require_kind=True)


def _check_song_source(source: object, where: str, corpus: dict) -> None:
    """Resolve the named Song source and bind its digest to the recorded revision."""
    _need(isinstance(source, dict), f"{where} must be an object")
    rel = source.get("file")
    symbol = source.get("symbol")
    revision = source.get("revision")
    digest = source.get("sha256")
    _nonempty(rel, f"{where}.file")
    _nonempty(symbol, f"{where}.symbol")
    _nonempty(revision, f"{where}.revision")
    _need(
        isinstance(digest, str) and jointlib.HEX64.fullmatch(digest),
        f"{where}.sha256 must be a sha256 hex digest",
    )
    _need(
        revision == corpus.get("song_git"),
        f"{where}.revision {revision!r} does not match the recorded Song "
        f"revision {corpus.get('song_git')!r}",
    )
    path = _resolve_path(rel, where)
    text = path.read_text(encoding="utf-8")
    _need(
        _check_slang(text, symbol),
        f"{where}: Song entry {symbol} not found in {rel}",
    )
    _need(
        _sha256(path) == digest,
        f"{where}: recorded digest does not match {path}",
    )
    jointspec.bind_source_bytes(
        path,
        rel,
        digest,
        {
            "song": revision,
            "apeiron": corpus.get("apeiron_git"),
            "corpus": corpus.get("git"),
        },
    )


def _validate_entry(
    entry: object,
    corpus: dict,
    seen_ids: set[str],
    canonical_targets: set[str],
) -> None:
    _need(isinstance(entry, dict), "each selected entry must be an object")
    eid = entry.get("id")
    _nonempty(eid, "entry.id")
    _need(eid not in seen_ids, f"duplicate entry id {eid}")
    seen_ids.add(eid)
    _nonempty(entry.get("title"), f"{eid}.title")
    status = entry.get("status")
    _need(
        status == "selected-open",
        f"{eid}.status must be selected-open, got {status!r}",
    )
    for field in REQUIRED_FIELDS:
        _need(field in entry, f"{eid} drops required field {field}")
        value = entry[field]
        _need(value is not None, f"{eid}.{field} is null")
        if isinstance(value, str):
            _nonempty(value, f"{eid}.{field}")
        elif isinstance(value, (list, dict)):
            _need(len(value) > 0, f"{eid}.{field} is empty")

    _check_song_source(entry["song"], f"{eid}.song", corpus)

    targets = entry["canonical_targets"]
    _need(
        isinstance(targets, list) and targets,
        f"{eid}.canonical_targets must be nonempty",
    )
    for target in targets:
        _check_lean_ref(target, f"{eid}.canonical_targets")
        canonical_targets.add(target["name"])

    _nonempty(entry["rejecting_control"], f"{eid}.rejecting_control")
    missing = entry["missing"]
    _need(
        "MISSING" in missing,
        f"{eid}.missing must record the absent artifacts as MISSING",
    )


def _validate_control(control: object, entry_ids: set[str]) -> None:
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
    _need(bool(targets), f"{cid}.target must name at least one entry id")
    invalid = [token for token in targets if token not in entry_ids]
    _need(not invalid, f"{cid}.target names unknown entries {invalid}")
    jointlib._check_text_refs(control["mutation"], f"{cid}.mutation")
    jointlib._check_text_refs(control["expected_red"], f"{cid}.expected_red")


def _validate_spec(
    text: object,
    row_ids: list[str],
    row_titles: list[str],
    row_statuses: dict[str, str],
    control_ids: list[str],
    field_keys: list[str],
) -> None:
    jointspec.validate_spec_labeled(
        text,
        row_ids,
        row_titles,
        row_statuses,
        [],
        control_ids,
        field_keys,
        ROW_FIELD_LABELS,
        REQUIRED_SECTIONS,
    )
    for key in NON_REFINEMENT_KEYS:
        _need(
            re.search(
                r"(?m)^\s*\|\s*`?" + re.escape(key) + r"`?\s*\|", text
            )
            is not None,
            f"spec is missing the non-refinement record {key}",
        )
    plain = _plain(text)
    for phrase in NON_REFINEMENT_PHRASES:
        _need(phrase in plain, f"spec is missing the non-refinement statement: {phrase}")
    for phrase in MISSING_PHRASES:
        _need(phrase in plain, f"spec is missing the missing-artifact statement: {phrase}")
    for phrase in ACTIVE_ROUTE_PHRASES:
        _need(phrase in plain, f"spec is missing the active-route statement: {phrase}")
    for phrase in RETIRED_ROUTE_PHRASES:
        _need(phrase in plain, f"spec is missing the retired-route statement: {phrase}")
    for phrase in ACCEPTANCE_PHRASES:
        _need(phrase in plain, f"spec is missing the acceptance statement: {phrase}")


def validate(data: object, spec_text: str) -> None:
    """Raise :class:`CheckError` unless the specification is internally consistent."""
    _need(isinstance(data, dict), "profile root must be an object")
    _need(data.get("profile") == PROFILE_ID, f"profile id must be {PROFILE_ID}")
    _need(isinstance(data.get("version"), int), "version must be an integer")

    corpus = data.get("corpus")
    jointspec.verify_corpus(corpus)
    _need(isinstance(corpus, dict), "corpus metadata missing")

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

    non_refinement = data.get("non_refinement")
    _need(isinstance(non_refinement, dict), "non_refinement must be an object")
    for key in NON_REFINEMENT_KEYS:
        _nonempty(non_refinement.get(key), f"non_refinement.{key}")
    plain = _plain(" ".join(non_refinement.get(key, "") for key in NON_REFINEMENT_KEYS))
    for phrase in NON_REFINEMENT_PHRASES:
        _need(
            phrase in plain,
            f"non_refinement is missing the statement: {phrase}",
        )

    entries = data.get("entries")
    _need(isinstance(entries, list) and entries, "entries must be a nonempty list")
    seen_ids: set[str] = set()
    canonical_targets: set[str] = set()
    for entry in entries:
        _validate_entry(entry, corpus, seen_ids, canonical_targets)
    missing_entries = [e for e in REQUIRED_ENTRIES if e not in seen_ids]
    _need(not missing_entries, f"required selected entries dropped: {missing_entries}")
    missing_targets = sorted(REQUIRED_CANONICAL_TARGETS - canonical_targets)
    _need(
        not missing_targets,
        f"canonical target declarations missing: {missing_targets}",
    )

    missing_artifacts = data.get("missing_artifacts")
    _need(
        isinstance(missing_artifacts, list) and missing_artifacts,
        "missing_artifacts must be a nonempty list",
    )
    _need(
        all(isinstance(item, str) and item.strip() for item in missing_artifacts),
        "missing_artifacts entries must be nonempty strings",
    )
    missing_plain = _plain(" ".join(missing_artifacts))
    for phrase in MISSING_PHRASES:
        _need(
            phrase in missing_plain,
            f"missing_artifacts is missing the record: {phrase}",
        )

    controls = data.get("controls")
    _need(isinstance(controls, list) and controls, "controls must be a nonempty list")
    control_ids = [c.get("id") for c in controls if isinstance(c, dict)]
    _need(len(control_ids) == len(set(control_ids)), "control ids must be unique")
    missing_controls = [c for c in REQUIRED_CONTROLS if c not in control_ids]
    _need(not missing_controls, f"rejecting controls dropped: {missing_controls}")
    for control in controls:
        _validate_control(control, seen_ids)

    for entry in entries:
        ref = entry["rejecting_control"]
        _need(
            ref in control_ids,
            f"{entry['id']}.rejecting_control {ref} is not a named control",
        )

    covered = data.get("not_covered")
    _need(isinstance(covered, list) and covered, "not_covered must be a nonempty list")
    _need(
        all(isinstance(item, str) and item.strip() for item in covered),
        "not_covered entries must be nonempty strings",
    )

    row_statuses = {
        entry["id"]: entry["status"] for entry in entries if isinstance(entry, dict)
    }
    _validate_spec(
        spec_text,
        list(seen_ids),
        [entry["title"] for entry in entries],
        row_statuses,
        control_ids,
        list(field_keys),
    )
    jointlib._check_data_text_refs(data, "profile")
    jointlib._check_text_refs(spec_text, "spec")


def _drop_target(data: dict, entry_id: str, name: str) -> None:
    for entry in data["entries"]:
        if entry.get("id") == entry_id:
            entry["canonical_targets"] = [
                t for t in entry["canonical_targets"] if t.get("name") != name
            ]
            return
    raise AssertionError(f"control setup: entry {entry_id} not found")


def _entry(data: dict, entry_id: str) -> dict:
    for entry in data["entries"]:
        if entry.get("id") == entry_id:
            return entry
    raise AssertionError(f"control setup: entry {entry_id} not found")


def _control_cases(data: dict, spec_text: str) -> list[tuple[str, dict, str]]:
    cases: list[tuple[str, dict, str]] = []

    def add(name: str, mutated: dict, spec: str | None = None) -> None:
        cases.append((name, mutated, spec if spec is not None else spec_text))

    # class 1: required fields and entries.
    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1").pop("observation_relation", None)
    add("class 1: drop the required field observation_relation from entry SE1", mutated)

    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["represented_program"] = ""
    add("class 1: empty the represented_program of entry SE1", mutated)

    mutated = copy.deepcopy(data)
    mutated["entries"] = [e for e in mutated["entries"] if e.get("id") != "SE1"]
    add("class 1: remove the selected entry SE1", mutated)

    # class 2: digest and revision pins.
    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["song"]["sha256"] = "0" * 64
    add("class 2: change the recorded sha256 of entry SE1", mutated)

    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["song"]["revision"] = "0" * 40
    add("class 2: change the recorded revision of entry SE1", mutated)

    mutated = copy.deepcopy(data)
    mutated["corpus"]["git"] = "0" * 40
    add("class 2: repin the corpus revision to a wrong value", mutated)

    mutated = copy.deepcopy(data)
    mutated["corpus"]["song_git"] = "0" * 40
    add("class 2: repin the Song revision to a wrong value", mutated)

    mutated = copy.deepcopy(data)
    mutated["corpus"]["sources"]["song/src/table/table.slang"] = "0" * 64
    add("class 2: change a recorded Song source digest", mutated)

    # class 3: the named Song source entry.
    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["song"]["symbol"] = "doesNotExist"
    add("class 3: point a Song entry at a missing symbol", mutated)

    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["song"]["symbol"] = "mergeacc_comm"
    add("class 3: point a Song entry at a law header only", mutated)

    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["song"]["file"] = "song/src/table/does_not_exist.slang"
    add("class 3: point a Song entry at a missing file", mutated)

    # class 4: exact fully-qualified canonical target identity.
    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["canonical_targets"][0]["name"] = (
        "Apeiron.Table.does_not_exist"
    )
    add("class 4: rename a canonical target to a nonexistent name", mutated)

    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["canonical_targets"][0]["name"] = (
        "Table.Apeiron.insertWith"
    )
    add("class 4: reverse a canonical namespace", mutated)

    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["canonical_targets"][0]["kind"] = "theorem"
    add("class 4: mislabel a canonical def as a theorem", mutated)

    # class 5: no entry closes.
    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["status"] = "closed"
    add("class 5: set entry SE1 status to closed", mutated)

    # class 6: the spec must carry the records and fields.
    add(
        "class 6: delete the required-field record song from the spec",
        copy.deepcopy(data),
        jointspec.strip_field_row(spec_text, "song"),
    )
    add(
        "class 6: delete the control record SC1 from the spec",
        copy.deepcopy(data),
        jointspec.strip_record(spec_text, "SC1"),
    )
    add(
        "class 6: delete the selected-entry heading SE1 from the spec",
        copy.deepcopy(data),
        re.sub(r"(?m)^###\s+SE1\b.*\n", "", spec_text, count=1),
    )
    add(
        "class 6: empty the SE1 Markdown observation relation body",
        copy.deepcopy(data),
        re.sub(
            r"(\*\*Observation relation\.\*\*).*?(?=\*\*Canonical targets\.\*\*)",
            r"\1\n\n",
            spec_text,
            count=1,
            flags=re.S,
        ),
    )
    add(
        "class 6: change the SE1 Markdown row status to closed",
        copy.deepcopy(data),
        spec_text.replace("| selected-open |", "| closed |", 1),
    )

    # class 7: every named reference is checked under the strict grammar.
    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1")["represented_program"] = (
        _entry(mutated, "SE1")["represented_program"]
        + " src/does_not_exist.rs::missing_fn"
    )
    add("class 7: add an unresolved named reference to a row field", mutated)

    for label, reference in (
        (
            "class 7: reject a reference with an extra '::' segment",
            "song/src/table/table.slang::tableGet::tableFind::tableSpread",
        ),
        (
            "class 7: reject a brace member that is not a single identifier "
            "or a Type::method pair",
            "song/src/table/table.slang::{tableGet::x::y}",
        ),
        (
            "class 7: reject a reference with a trailing member access",
            "proof/Denotation.lean::denote.missing",
        ),
    ):
        mutated = copy.deepcopy(data)
        _entry(mutated, "SE1")["represented_program"] = (
            _entry(mutated, "SE1")["represented_program"] + " " + reference
        )
        add(label, mutated)

    # class 8: non-refinement and missing-artifact records.
    mutated = copy.deepcopy(data)
    mutated["non_refinement"].pop("copied_model", None)
    add("class 8: remove the copied_model non-refinement statement", mutated)

    mutated = copy.deepcopy(data)
    mutated["non_refinement"]["body_change_rule"] = ""
    add("class 8: empty the body_change_rule non-refinement statement", mutated)

    mutated = copy.deepcopy(data)
    _entry(mutated, "SE1").pop("missing", None)
    add("class 8: remove the missing record from entry SE1", mutated)

    mutated = copy.deepcopy(data)
    mutated["missing_artifacts"] = []
    add("class 8: empty the missing_artifacts list", mutated)

    add(
        "class 8: drop the retired segment-lemma statement from the spec",
        copy.deepcopy(data),
        spec_text.replace("segment lemma", "the lemma", 1),
    )

    add(
        "class 8: drop the implementation-acceptance statement from the spec",
        copy.deepcopy(data),
        spec_text.replace("Implementation acceptance", "The result", 1),
    )

    # class 9: structural removals.
    mutated = copy.deepcopy(data)
    mutated["fields"] = mutated["fields"][:-1]
    add("class 9: drop a required field from the fields list", mutated)

    mutated = copy.deepcopy(data)
    mutated["controls"] = [c for c in mutated["controls"] if c.get("id") != "SC1"]
    add("class 9: remove required control SC1", mutated)

    add(
        "class 9: remove the Non-refinement section heading from the spec",
        copy.deepcopy(data),
        re.sub(r"(?m)^##\s+Non-refinement\s*$", "", spec_text, count=1),
    )
    add(
        "class 9: remove the non-refinement record table from the spec",
        copy.deepcopy(data),
        re.sub(
            r"(?m)^\s*\|\s*`?(?:copied_model|differential_suite|equal_outputs|body_change_rule)`?\s*\|.*\n",
            "",
            spec_text,
        ),
    )
    add(
        "class 9: remove the Active route and acceptance section heading from the spec",
        copy.deepcopy(data),
        re.sub(r"(?m)^##\s+Active route and acceptance\s*$", "", spec_text, count=1),
    )
    add(
        "class 9: drop the active-route complete-memo-key statement from the spec",
        copy.deepcopy(data),
        spec_text.replace("complete memo key", "the key", 1),
    )

    # class 10: the required canonical target set.
    mutated = copy.deepcopy(data)
    _drop_target(mutated, "SE2", "Apeiron.Table.get")
    add("class 10: drop the required get canonical target", mutated)

    mutated = copy.deepcopy(data)
    _drop_target(mutated, "SE3", "Apeiron.Table.get_insertWith_ne")
    add("class 10: drop the required get_insertWith_ne canonical target", mutated)

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
    "class 9:",
    "class 10:",
]

MANDATED_CONTROLS = [
    "class 1: drop the required field observation_relation from entry SE1",
    "class 1: remove the selected entry SE1",
    "class 2: change the recorded sha256 of entry SE1",
    "class 2: change a recorded Song source digest",
    "class 3: point a Song entry at a missing symbol",
    "class 3: point a Song entry at a law header only",
    "class 4: rename a canonical target to a nonexistent name",
    "class 4: reverse a canonical namespace",
    "class 4: mislabel a canonical def as a theorem",
    "class 5: set entry SE1 status to closed",
    "class 6: delete the required-field record song from the spec",
    "class 6: delete the control record SC1 from the spec",
    "class 6: delete the selected-entry heading SE1 from the spec",
    "class 6: change the SE1 Markdown row status to closed",
    "class 7: add an unresolved named reference to a row field",
    "class 7: reject a reference with an extra '::' segment",
    "class 7: reject a brace member that is not a single identifier or a "
    "Type::method pair",
    "class 7: reject a reference with a trailing member access",
    "class 8: remove the copied_model non-refinement statement",
    "class 8: empty the body_change_rule non-refinement statement",
    "class 8: remove the missing record from entry SE1",
    "class 8: empty the missing_artifacts list",
    "class 9: drop a required field from the fields list",
    "class 9: remove required control SC1",
    "class 9: remove the Non-refinement section heading from the spec",
    "class 9: remove the Active route and acceptance section heading from the spec",
    "class 9: drop the active-route complete-memo-key statement from the spec",
    "class 8: drop the retired segment-lemma statement from the spec",
    "class 8: drop the implementation-acceptance statement from the spec",
    "class 10: drop the required get canonical target",
    "class 10: drop the required get_insertWith_ne canonical target",
]


def run_controls(data: dict, spec_text: str) -> list[tuple[str, bool, str]]:
    """Baseline must pass; every in-memory mutation must be rejected."""
    return jointspec.run_control_cases(validate, data, spec_text, _control_cases)


def _load() -> tuple[dict, str]:
    return jointspec.load_profile(CASES_PATH, SPEC_PATH, noun="specification")


def _summary(data: dict) -> str:
    entry_ids = [entry["id"] for entry in data["entries"]]
    control_ids = [control["id"] for control in data["controls"]]
    targets = sum(len(entry["canonical_targets"]) for entry in data["entries"])
    return (
        f"entries={entry_ids} controls={control_ids} "
        f"canonical_targets={targets} missing_artifacts={len(data['missing_artifacts'])}"
    )


def main(argv: list[str] | None = None) -> int:
    return jointspec.run_cli(
        __doc__,
        _load,
        validate,
        run_controls,
        _summary,
        noun="specification",
        mandated_controls=tuple(MANDATED_CONTROLS),
        class_markers=tuple(CLASS_MARKERS),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
