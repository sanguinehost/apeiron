#!/usr/bin/env python3
"""Consistency checker for the W19-T2 joint carrier/exchange subprofile.

This tool establishes **specification consistency only**. It verifies that
``docs/contracts/joint-carrier-exchange.md`` and
``tools/joint/carrier_profile_rows.json`` are internally complete and that
every artifact they name resolves:

* every carrier/exchange subprofile row carries all eight required fields and
  stays ``selected-open`` (a ``closed`` row is rejected);
* every named canonical authority target resolves as the named declaration
  (``theorem``/``lemma``, ``def``/``abbrev``, ...) in the pinned canonical
  corpus (``../sanguine-apeiron/proof``) under its exact opened namespace, and
  every named bridge refinement resolves in ``proof/``;
* every named Rust entry exists as a concrete executing declaration bound to
  the named implementation in its source file (a same-named method on another
  type is not an entry), and every named Song entry is a concrete executing
  declaration; a generic type parameter is never a self type or a resolvable
  symbol, a bare symbol resolves only to an item the file declares
  (``fn``/``struct``/``enum``/``trait``/``union``/``type``/``mod``/``const``/
  ``static``) so a word that appears only in an ``impl`` header, a generic bound
  or a ``where`` clause is rejected, and a ``<T as Trait>`` qualifier must name a
  trait the file implements for ``T``;
* the corpus revision, Mathlib revision, Apeiron/Song revisions and per-source
  sha256 digests match the resolved checkout, and each recorded digest is bound
  to the blob at the recorded revision;
* every implementation present in a row carries its required resource kinds
  separately (a kind on one side never covers another), no generic ``fuel`` is
  admitted, and an absent implementation must state its absence;
* every named reference, including blocker ``counterpart``/``reason`` entries
  and control targets, is parsed under the strict grammar documented at
  ``FILE_PATH`` and checked, not only the selected structured references; a
  reference outside the grammar is rejected as malformed, never truncated or
  split;
* every required unmatched blocker, hazard and rejecting control is present,
  and none is dropped;
* the spec carries the actual records and fields (row headings and titles,
  required-field records, blocker records, hazard records, control records),
  not incidental mentions.

It does **not** establish semantic correspondence, refinement or proof, and it
never claims correspondence. A green run says the specification is well formed
and source-connected by name; it says nothing about whether any row's
refinement has been proved or whether any control goes red.

Usage::

    python3 tools/joint/carrier_profile_check.py --check
    python3 tools/joint/carrier_profile_check.py --controls
    python3 tools/joint/carrier_profile_check.py --self-test

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
SPEC_PATH = REPO / "docs" / "contracts" / "joint-carrier-exchange.md"
ROWS_PATH = HERE.parent / "carrier_profile_rows.json"

PROFILE_ID = "APEIRON-PROGRAMME-W19-T2"

FIELDS = [
    "canonical",
    "entry_points",
    "matched_domain",
    "representation_invariant",
    "admission",
    "behavior",
    "resources",
    "refinement",
]

REQUIRED_ROWS = ["K1", "K2", "K3", "K4", "K5", "K6", "K7"]
REQUIRED_BLOCKERS = ["BC1", "BC2", "BC3", "BC4"]
REQUIRED_HAZARDS = ["H1", "H2", "H3", "H4", "H5"]
REQUIRED_CONTROLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
REQUIRED_SECTIONS = [
    "Required fields",
    "Selected boundary",
    "Rows",
    "Unmatched counterparts",
    "Hazards",
    "Rejecting controls",
    "Integration order",
    "Not covered",
    "Checker",
]

# The five distinct implementation resource kinds. They are never converted to
# a common fuel and a kind declared on one implementation never covers another.
REQUIRED_RESOURCE_KINDS = {
    "arithmetic_ops",
    "probes",
    "nodes",
    "stack_entries",
    "input_consumption",
}
FORBIDDEN_RESOURCE_KINDS = {"fuel"}

# Exact canonical declarations that must appear as targets somewhere in the
# profile. A target is bound to its real fully-qualified namespace, so a merely
# opened prefix or a reversed namespace is rejected.
REQUIRED_CANONICAL_THEOREMS = {
    "Apeiron.Trichotomy.AnyRing.ringNormForm_mul",
    "Apeiron.Physics.pointDensity_eq",
    "Apeiron.Physics.pointFlux_eq",
    "Apeiron.Physics.scalarCarrierStep_continuity",
    "Apeiron.Physics.exactHalfFlux_iff",
    "Apeiron.Physics.exactHalfFlux_neg",
    "Apeiron.Physics.ringCarrierMomentumAfter_eq",
    "Apeiron.Physics.ringBodyMomentumAfter_eq",
    "Apeiron.Physics.ringTotalMomentum_conserved",
    "Apeiron.Physics.ringNormFormLedger_conserved",
}
REQUIRED_CANONICAL_DEFS = {
    "Apeiron.Trichotomy.AnyRing.ringMul",
    "Apeiron.Trichotomy.AnyRing.ringNormForm",
    "Apeiron.Physics.pointDensity",
    "Apeiron.Physics.pointFlux",
    "Apeiron.Physics.scalarCarrierStep",
    "Apeiron.Physics.exactHalfFlux",
    "Apeiron.Physics.ringExchangeImpulse",
}
# Exact bridge refinements that must appear as refinement artifacts somewhere in
# the profile. These are the source-connected links the rows claim.
REQUIRED_BRIDGE_ARTIFACTS = {
    "apeiron_word.denote_wrapping_add",
    "apeiron_word.denote_wrapping_sub",
    "apeiron_word.denote_wrapping_mul",
    "apeiron_word.word_mul_denote",
    "apeiron_word.word_norm_denote",
    "apeiron_word.word_norm_mul",
    "apeiron_word.word_point_density_denote",
    "apeiron_word.word_point_flux_denote",
    "apeiron_word.word_conservative_density_step_denote",
    "apeiron_word.word_conservative_density_step_eq_some_iff",
    "apeiron_word.word_diffusive_flux_half_denote",
    "apeiron_word.word_diffusive_flux_half_success",
    "apeiron_word.word_diffusive_line3_step_eq_some_iff",
    "apeiron_word.word_diffusive_line3_step_denote",
    "apeiron_word.word_diffusive_line3_step_total_mass_zero_source",
    "apeiron_word.word_exchange_impulse_denote",
    "apeiron_word.word_elastic_exchange_denote",
    "apeiron_word.word_accept_derived_impulse_spec",
}

BEHAVIOR_KEYS = [
    "success",
    "unsupported_input",
    "resource_exhaustion",
    "raw_return_encoding",
    "post_state",
]
REPRESENTATION_KEYS = ["constructor", "statement", "state_effects"]
ADMISSION_SIDES = ["rust", "song"]
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


def _validate_resources(resources: object, rid: str) -> None:
    """Every present implementation carries all five kinds; absence is stated."""
    _need(isinstance(resources, dict), f"{rid}.resources must be an object")
    for side in ADMISSION_SIDES:
        entries = resources.get(side)
        _need(
            isinstance(entries, list),
            f"{rid}.resources.{side} must be a list (possibly empty)",
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
        if entries:
            missing = sorted(REQUIRED_RESOURCE_KINDS - declared)
            _need(
                not missing,
                f"{rid}.resources.{side} is missing required kinds {missing}",
            )
        else:
            _nonempty(
                resources.get(f"{side}_absent"),
                f"{rid}.resources.{side}_absent",
            )


def _validate_row(
    row: object,
    seen_ids: set[str],
    canonical_theorems: set[str],
    canonical_defs: set[str],
    bridge_artifacts: set[str],
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
    for field in FIELDS:
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

    entry_points = row["entry_points"]
    _need(isinstance(entry_points, dict), f"{rid}.entry_points must be an object")
    rust_entries = entry_points.get("rust")
    _need(
        isinstance(rust_entries, list) and rust_entries,
        f"{rid}.entry_points.rust must be nonempty",
    )
    for entry in rust_entries:
        _check_source_entry(entry, f"{rid}.entry_points.rust", "rust")
    song_entries = entry_points.get("song")
    _need(
        isinstance(song_entries, list),
        f"{rid}.entry_points.song must be a list (possibly empty)",
    )
    for entry in song_entries:
        _check_source_entry(entry, f"{rid}.entry_points.song", "iota")
    if not song_entries:
        _nonempty(
            entry_points.get("song_absent"), f"{rid}.entry_points.song_absent"
        )

    _nonempty(row["matched_domain"], f"{rid}.matched_domain")

    invariant = row["representation_invariant"]
    _need(
        isinstance(invariant, dict),
        f"{rid}.representation_invariant must be an object",
    )
    for key in REPRESENTATION_KEYS:
        _nonempty(invariant.get(key), f"{rid}.representation_invariant.{key}")

    admission = row["admission"]
    _need(isinstance(admission, dict), f"{rid}.admission must be an object")
    for side in ADMISSION_SIDES:
        sub = admission.get(side)
        _need(isinstance(sub, dict), f"{rid}.admission.{side} must be an object")
        _nonempty(sub.get("predicate"), f"{rid}.admission.{side}.predicate")
        _nonempty(sub.get("source"), f"{rid}.admission.{side}.source")
    _nonempty(admission.get("joint"), f"{rid}.admission.joint")

    behavior = row["behavior"]
    _need(isinstance(behavior, dict), f"{rid}.behavior must be an object")
    for key in BEHAVIOR_KEYS:
        _nonempty(behavior.get(key), f"{rid}.behavior.{key}")

    _validate_resources(row["resources"], rid)

    refinement = row["refinement"]
    _need(isinstance(refinement, dict), f"{rid}.refinement must be an object")
    _nonempty(refinement.get("source_connected"), f"{rid}.refinement.source_connected")
    _nonempty(refinement.get("rejecting_control"), f"{rid}.refinement.rejecting_control")
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
        bridge_artifacts.add(artifact["name"])


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
    rust_entry = blocker.get("rust_entry")
    song_entry = blocker.get("song_entry")
    _need(
        theorem is not None or rust_entry is not None or song_entry is not None,
        f"{bid} carries no resolvable named reference",
    )
    if theorem is not None:
        _check_lean_ref(theorem, f"{bid}.canonical_theorem", require_root="corpus")
    if rust_entry is not None:
        _check_source_entry(rust_entry, f"{bid}.rust_entry", "rust")
    if song_entry is not None:
        _check_source_entry(song_entry, f"{bid}.song_entry", "iota")
    _check_blocker_refs(blocker, bid)


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


FIELD_LABELS = {
    "canonical": ("1", "Canonical object and observation relation"),
    "entry_points": ("2", "Entry points and source identity"),
    "matched_domain": ("3", "Matched domain"),
    "representation_invariant": ("4", "Representation invariant"),
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
    """Raise :class:`CheckError` unless the subprofile is internally consistent."""
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
    canonical_theorems: set[str] = set()
    canonical_defs: set[str] = set()
    bridge_artifacts: set[str] = set()
    for row in rows:
        _validate_row(
            row, seen_ids, canonical_theorems, canonical_defs, bridge_artifacts
        )
    missing_rows = [r for r in REQUIRED_ROWS if r not in seen_ids]
    _need(not missing_rows, f"required selected rows dropped: {missing_rows}")
    missing_theorems = sorted(REQUIRED_CANONICAL_THEOREMS - canonical_theorems)
    _need(
        not missing_theorems,
        f"canonical theorem targets missing: {missing_theorems}",
    )
    missing_defs = sorted(REQUIRED_CANONICAL_DEFS - canonical_defs)
    _need(not missing_defs, f"canonical authority declarations missing: {missing_defs}")
    missing_bridge = sorted(REQUIRED_BRIDGE_ARTIFACTS - bridge_artifacts)
    _need(not missing_bridge, f"bridge refinement artifacts missing: {missing_bridge}")

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

    order = data.get("integration_order")
    _need(isinstance(order, list) and order, "integration_order must be a nonempty list")
    _need(
        all(isinstance(item, str) and item.strip() for item in order),
        "integration_order entries must be nonempty strings",
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
        blocker_ids,
        hazard_ids,
        control_ids,
        list(field_keys),
    )
    jointlib._check_data_text_refs(data, "profile")
    jointlib._check_text_refs(spec_text, "spec")


def _drop_row_target(data: dict, row_index: int, name: str) -> None:
    data["rows"][row_index]["canonical"]["targets"] = [
        t for t in data["rows"][row_index]["canonical"]["targets"] if t.get("name") != name
    ]


def _drop_row_artifact(data: dict, row_index: int, name: str) -> None:
    data["rows"][row_index]["refinement"]["artifacts"] = [
        a
        for a in data["rows"][row_index]["refinement"]["artifacts"]
        if a.get("name") != name
    ]


def _control_cases(data: dict, spec_text: str) -> list[tuple[str, dict, str]]:
    cases: list[tuple[str, dict, str]] = []

    def add(name: str, mutated: dict, spec: str | None = None) -> None:
        cases.append((name, mutated, spec if spec is not None else spec_text))

    # class 1: resource completeness is per implementation.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["rust"] = [
        e
        for e in mutated["rows"][0]["resources"]["rust"]
        if e.get("kind") != "arithmetic_ops"
    ]
    add(
        "class 1: drop a required resource account from the K1 Rust implementation",
        mutated,
    )

    # class 2: corpus pin and digest verification.
    mutated = copy.deepcopy(data)
    mutated["corpus"]["git"] = "0" * 40
    add("class 2: repin the corpus revision to a wrong value", mutated)

    mutated = copy.deepcopy(data)
    mutated["corpus"]["sources"]["src/carrier.rs"] = "0" * 64
    add("class 2: change a recorded source digest", mutated)

    # class 3: exact fully-qualified declaration identity.
    mutated = copy.deepcopy(data)
    mutated["rows"][1]["canonical"]["targets"][0]["name"] = (
        "AnyRing.Apeiron.Trichotomy.ringMul"
    )
    add("class 3: reverse a canonical namespace", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][1]["canonical"]["targets"][0]["name"] = (
        "Apeiron.Trichotomy.ringMul"
    )
    add("class 3: drop the AnyRing namespace from a canonical target", mutated)

    # class 4: exact Rust implementation binding; a same-named method on another
    # type does not satisfy the entry.
    mutated = copy.deepcopy(data)
    mutated["rows"][1]["entry_points"]["rust"][0]["symbol"] = "Sigma::mul"
    add("class 4: bind a Rust method to the wrong implementation", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["rust"][0]["symbol"] = "does_not_exist"
    add("class 4: point a Rust entry at a missing symbol", mutated)

    # class 5: the spec must carry the records and fields.
    add(
        "class 5: delete the blocker record BC1 from the spec",
        copy.deepcopy(data),
        jointspec.strip_record(spec_text, "BC1"),
    )
    add(
        "class 5: delete the required-field record canonical from the spec",
        copy.deepcopy(data),
        jointspec.strip_field_row(spec_text, "canonical"),
    )
    add(
        "class 5: delete the hazard record H1 from the spec",
        copy.deepcopy(data),
        jointspec.strip_record(spec_text, "H1"),
    )
    add(
        "class 5: delete the control record C1 from the spec",
        copy.deepcopy(data),
        jointspec.strip_record(spec_text, "C1"),
    )

    # class 6: no row closes.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["status"] = "closed"
    add("class 6: set row K1 status to closed", mutated)

    # class 7: every named reference is checked.
    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"][0]["counterpart"] = (
        mutated["unmatched_blockers"][0]["counterpart"]
        + "; src/does_not_exist.rs::missing_fn exists."
    )
    add("class 7: name an unresolved file in a blocker counterpart", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["refinement"]["source_connected"] = (
        mutated["rows"][0]["refinement"]["source_connected"]
        + " src/does_not_exist.rs::missing_fn"
    )
    add("class 7: add an unresolved named reference to a row field", mutated)

    # class 7: a qualified reference is consumed in full; ``merge`` belongs to
    # ``WordMerge``, not ``WordHome``, so the trailing method is rejected.
    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"][0]["counterpart"] = (
        mutated["unmatched_blockers"][0]["counterpart"]
        + " src/table.rs::WordHome::merge"
    )
    add(
        "class 7: reject a qualified method bound to the wrong Rust "
        "implementation in a blocker counterpart",
        mutated,
    )

    # class 8: one rejecting mutation per row requirement, exercised against
    # the exact-target and exact-artifact checks.
    mutated = copy.deepcopy(data)
    _drop_row_artifact(mutated, 0, "apeiron_word.denote_wrapping_add")
    add(
        "class 8: C1 word arithmetic drops the wrapping-add bridge refinement",
        mutated,
    )

    mutated = copy.deepcopy(data)
    _drop_row_target(mutated, 1, "Apeiron.Trichotomy.AnyRing.ringNormForm_mul")
    add("class 8: C2 cell algebra drops the norm multiplicativity target", mutated)

    mutated = copy.deepcopy(data)
    _drop_row_target(mutated, 2, "Apeiron.Physics.pointFlux_eq")
    add("class 8: C3 point flux drops the pointFlux_eq target", mutated)

    mutated = copy.deepcopy(data)
    _drop_row_target(mutated, 3, "Apeiron.Physics.scalarCarrierStep_continuity")
    add("class 8: C4 conservative step drops the continuity target", mutated)

    mutated = copy.deepcopy(data)
    _drop_row_target(mutated, 4, "Apeiron.Physics.exactHalfFlux_neg")
    add("class 8: C5 exact half-flux drops the exactHalfFlux_neg target", mutated)

    mutated = copy.deepcopy(data)
    _drop_row_artifact(
        mutated, 5, "apeiron_word.word_diffusive_line3_step_total_mass_zero_source"
    )
    add("class 8: C6 three-cell line drops the zero-source conservation artifact", mutated)

    mutated = copy.deepcopy(data)
    _drop_row_artifact(mutated, 6, "apeiron_word.word_accept_derived_impulse_spec")
    add("class 8: C7 exchange drops the checked-claim specification artifact", mutated)

    # Retained structural controls.
    mutated = copy.deepcopy(data)
    mutated["rows"][0].pop("canonical", None)
    add("remove required field canonical from row K1", mutated)

    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"] = [
        b for b in mutated["unmatched_blockers"] if b.get("id") != "BC1"
    ]
    add("remove unmatched blocker BC1", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"] = mutated["rows"][1:]
    add("remove selected row K1", mutated)

    mutated = copy.deepcopy(data)
    mutated["hazards"] = [h for h in mutated["hazards"] if h.get("id") != "H1"]
    add("remove hazard H1", mutated)

    mutated = copy.deepcopy(data)
    mutated["controls"] = [c for c in mutated["controls"] if c.get("id") != "C1"]
    add("remove required control C1", mutated)

    mutated = copy.deepcopy(data)
    mutated["fields"] = mutated["fields"][:-1]
    add("drop a fields entry", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["rust"][0]["kind"] = "fuel"
    add("rename a resource kind to generic fuel", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["rust"][0]["file"] = "src/does_not_exist.rs"
    add("point a Rust entry at a missing file", mutated)

    # Markdown field body, not only its label.
    add(
        "class 5: empty the K1 Markdown canonical body",
        copy.deepcopy(data),
        re.sub(
            r"(\*\*1 · Canonical object and observation relation\.\*\*).*?(?=\*\*2 ·)",
            r"\1\n\n",
            spec_text,
            count=1,
            flags=re.S,
        ),
    )

    # Markdown status must agree with the JSON status.
    add(
        "class 6: change the K1 Markdown row status to closed",
        copy.deepcopy(data),
        spec_text.replace("| selected-open |", "| closed |", 1),
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
        mutated["unmatched_blockers"][0]["counterpart"] = (
            mutated["unmatched_blockers"][0]["counterpart"] + " " + reference
        )
        add(label, mutated)

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

MANDATED_CONTROLS = [
    "class 1: drop a required resource account from the K1 Rust implementation",
    "class 2: repin the corpus revision to a wrong value",
    "class 2: change a recorded source digest",
    "class 3: reverse a canonical namespace",
    "class 3: drop the AnyRing namespace from a canonical target",
    "class 4: bind a Rust method to the wrong implementation",
    "class 5: delete the blocker record BC1 from the spec",
    "class 5: delete the required-field record canonical from the spec",
    "class 5: delete the hazard record H1 from the spec",
    "class 5: delete the control record C1 from the spec",
    "class 6: set row K1 status to closed",
    "class 7: name an unresolved file in a blocker counterpart",
    "class 7: add an unresolved named reference to a row field",
    "class 7: reject a qualified method bound to the wrong Rust implementation "
    "in a blocker counterpart",
    "class 7: reject a reference with an extra '::' segment",
    "class 7: reject a brace member that is not a single identifier or a "
    "Type::method pair",
    "class 7: reject a generic-qualified method bound to the wrong implementation",
    "class 7: reject a generic-qualified method whose trait does not exist",
    "class 7: reject a method attributed to a generic type parameter",
    "class 7: reject a bare symbol that is only an impl-header generic parameter",
    "class 7: reject a bare symbol that is only an impl-header bound, not a "
    "declaration",
    "class 7: reject a reference with a trailing member access",
    "class 8: C1 word arithmetic drops the wrapping-add bridge refinement",
    "class 8: C2 cell algebra drops the norm multiplicativity target",
    "class 8: C3 point flux drops the pointFlux_eq target",
    "class 8: C4 conservative step drops the continuity target",
    "class 8: C5 exact half-flux drops the exactHalfFlux_neg target",
    "class 8: C6 three-cell line drops the zero-source conservation artifact",
    "class 8: C7 exchange drops the checked-claim specification artifact",
]


def run_controls(data: dict, spec_text: str) -> list[tuple[str, bool, str]]:
    """Baseline must pass; every in-memory mutation must be rejected."""
    return jointspec.run_control_cases(validate, data, spec_text, _control_cases)


def _load() -> tuple[dict, str]:
    return jointspec.load_profile(ROWS_PATH, SPEC_PATH, noun="subprofile")


def _summary(data: dict) -> str:
    row_ids = [row["id"] for row in data["rows"]]
    blocker_ids = [b["id"] for b in data["unmatched_blockers"]]
    hazard_ids = [h["id"] for h in data["hazards"]]
    control_ids = [c["id"] for c in data["controls"]]
    targets = sum(len(row["canonical"]["targets"]) for row in data["rows"])
    artifacts = sum(len(row["refinement"]["artifacts"]) for row in data["rows"])
    return (
        f"rows={row_ids} blockers={blocker_ids} hazards={hazard_ids} "
        f"controls={control_ids} canonical_targets={targets} bridge_artifacts={artifacts}"
    )


def main(argv: list[str] | None = None) -> int:
    return jointspec.run_cli(
        __doc__,
        _load,
        validate,
        run_controls,
        _summary,
        noun="subprofile",
        mandated_controls=tuple(MANDATED_CONTROLS),
        class_markers=tuple(CLASS_MARKERS),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
