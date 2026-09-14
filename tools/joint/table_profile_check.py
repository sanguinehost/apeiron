#!/usr/bin/env python3
"""Consistency checker for the W18-T3 joint table subprofile.

This tool establishes **specification consistency only**. It verifies that
``docs/spec/joint-table.md`` and ``tools/joint/table_profile_rows.json`` are
internally complete and that every artifact they name resolves:

* every table subprofile row carries all six required fields;
* every named canonical authority target resolves as the named declaration
  (``theorem``/``lemma``, ``def``/``abbrev``, ...) in the pinned canonical
  corpus (``../sanguine-apeiron/proof``), and every named bridge artifact
  resolves in ``proof/``;
* every named Rust and Song entry exists as a declaration in its source file;
* every required row, hazard and rejecting control is present, and none is
  dropped;
* the resource accounting names the five distinct kinds and no generic fuel;
* the spec names every required row, hazard and control.

It does **not** establish semantic correspondence, refinement, proof or the
rejecting controls' execution. A green run says the specification is well formed
and source-connected by name; it says nothing about whether any row's refinement
has been proved or whether any control goes red.

Usage::

    python3 tools/joint/table_profile_check.py --check
    python3 tools/joint/table_profile_check.py --controls
    python3 tools/joint/table_profile_check.py --self-test

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
SPEC_PATH = REPO / "docs" / "spec" / "joint-table.md"
ROWS_PATH = HERE.parent / "table_profile_rows.json"
CORPUS_ROOT = REPO.parent / "sanguine-apeiron" / "proof"
BRIDGE_ROOT = REPO / "proof"

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

LEAN_DECL_PREFIX = (
    r"(?m)^\s*(?:noncomputable\s+|private\s+|protected\s+|@\[[^\]]*\]\s*)*"
)
LEAN_KEYWORDS = {
    "theorem": r"(?:theorem|lemma)",
    "def": r"(?:def|abbrev)",
    "abbrev": r"abbrev",
    "structure": r"structure",
    "class": r"class",
    "inductive": r"inductive",
    "instance": r"instance",
}
LEAN_ANY_DECL = (
    LEAN_DECL_PREFIX
    + r"(?:theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
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
    r"(?m)^\s*{n}\s*=",
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


def _check_lean_ref(
    ref: object,
    where: str,
    require_root: str | None = None,
    require_kind: bool = True,
) -> None:
    _need(isinstance(ref, dict), f"{where} entries must be objects")
    name = ref.get("name")
    module = ref.get("module")
    root = ref.get("root", require_root)
    kind = ref.get("kind")
    _nonempty(name, f"{where}.name")
    _nonempty(module, f"{where}.module")
    if require_root is not None:
        _need(
            root == require_root,
            f"{where}.root must be {require_root} for {name}",
        )
    _need(root in ("corpus", "bridge"), f"{where}.root must be corpus or bridge")
    base = CORPUS_ROOT if root == "corpus" else BRIDGE_ROOT
    path = base / module
    _need(path.is_file(), f"{where}: module does not resolve: {root}:{module}")
    text = path.read_text(encoding="utf-8")
    segments = name.split(".")
    last = segments[-1]
    if kind is None:
        if require_kind:
            raise CheckError(f"{where}: {name} is missing a kind")
        pattern = LEAN_ANY_DECL + re.escape(last) + r"\b"
    else:
        _need(
            kind in LEAN_KEYWORDS,
            f"{where}: invalid declaration kind {kind!r} for {name}",
        )
        pattern = LEAN_DECL_PREFIX + LEAN_KEYWORDS[kind] + r"\s+" + re.escape(last) + r"\b"
    _need(
        re.search(pattern, text) is not None,
        f"{where}: declaration {last} ({kind or 'any'}) not found in {module}",
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
    if last == "build_gauge_invariance" and kind == "theorem":
        hypotheses = ref.get("hypotheses")
        _nonempty(hypotheses, f"{where}.{name}.hypotheses")
        _need(
            "hcomm" in hypotheses and "hassoc" in hypotheses,
            f"{where}: build_gauge_invariance must carry hcomm and hassoc hypotheses",
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


def _check_source_entry(entry: object, where: str, want_lang: str) -> None:
    _need(isinstance(entry, dict), f"{where} entries must be objects")
    symbol = entry.get("symbol")
    rel = entry.get("file")
    lang = entry.get("lang")
    _nonempty(symbol, f"{where}.symbol")
    _nonempty(rel, f"{where}.file")
    _need(lang in ("rust", "slang"), f"{where}.lang must be rust or slang")
    _need(lang == want_lang, f"{where}: expected lang {want_lang}, got {lang}")
    path = _resolve_path(rel, where)
    text = path.read_text(encoding="utf-8")
    found = _check_rust(text, symbol) if lang == "rust" else _check_slang(text, symbol)
    _need(found, f"{where}: {lang} entry {symbol} not found in {rel}")


def _validate_row(
    row: object,
    seen_ids: set[str],
    resource_kinds: set[str],
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
        status in ("selected-open", "blocker", "closed"),
        f"{rid}.status invalid: {status!r}",
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
        _check_source_entry(entry, f"{rid}.song.entry_points", "slang")

    admission = row["admission"]
    _need(isinstance(admission, dict), f"{rid}.admission must be an object")
    for side in ADMISSION_SIDES:
        sub = admission.get(side)
        _need(isinstance(sub, dict), f"{rid}.admission.{side} must be an object")
        _nonempty(sub.get("predicate"), f"{rid}.admission.{side}.predicate")
        _nonempty(sub.get("source"), f"{rid}.admission.{side}.source")

    resources = row["resources"]
    _need(isinstance(resources, dict), f"{rid}.resources must be an object")
    for side in ADMISSION_SIDES:
        entries = resources.get(side)
        _need(
            isinstance(entries, list) and entries,
            f"{rid}.resources.{side} must be nonempty",
        )
        for entry in entries:
            _need(isinstance(entry, dict), f"{rid}.resources.{side} entries must be objects")
            kind = entry.get("kind")
            _need(
                kind in REQUIRED_RESOURCE_KINDS,
                f"{rid}.resources.{side} invalid kind {kind!r}",
            )
            _nonempty(entry.get("accounting"), f"{rid}.resources.{side}[{kind}]")
            resource_kinds.add(kind)

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
    targets = [token.strip() for token in re.split(r"[,/ ]+", control["target"]) if token.strip()]
    _need(
        any(token in row_ids for token in targets),
        f"{cid}.target must name at least one row id",
    )


def _validate_spec(
    text: object,
    row_ids: list[str],
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
    """Raise :class:`CheckError` unless the subprofile is internally consistent."""
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
    resource_kinds: set[str] = set()
    canonical_theorems: set[str] = set()
    canonical_defs: set[str] = set()
    for row in rows:
        _validate_row(row, seen_ids, resource_kinds, canonical_theorems, canonical_defs)
    missing_rows = [r for r in REQUIRED_ROWS if r not in seen_ids]
    _need(not missing_rows, f"required rows dropped: {missing_rows}")
    missing_kinds = sorted(REQUIRED_RESOURCE_KINDS - resource_kinds)
    _need(
        not missing_kinds,
        f"resource kinds must include {sorted(REQUIRED_RESOURCE_KINDS)}, missing {missing_kinds}",
    )
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

    _validate_spec(spec_text, list(seen_ids), hazard_ids, control_ids)


def _control_cases(data: dict) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []

    mutated = copy.deepcopy(data)
    mutated["rows"][0].pop("canonical", None)
    cases.append(("remove required field canonical from row T1", mutated))

    mutated = copy.deepcopy(data)
    mutated["fields"] = mutated["fields"][:-1]
    cases.append(("drop a required field from the fields list", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"] = [r for r in mutated["rows"] if r.get("id") != "T1"]
    cases.append(("remove required row T1", mutated))

    mutated = copy.deepcopy(data)
    mutated["hazards"] = [h for h in mutated["hazards"] if h.get("id") != "H1"]
    cases.append(("remove required hazard H1", mutated))

    mutated = copy.deepcopy(data)
    mutated["controls"] = [c for c in mutated["controls"] if c.get("id") != "C1"]
    cases.append(("remove required control C1", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"][0]["name"] = (
        "Apeiron.Table.does_not_exist"
    )
    cases.append(("rename a canonical target to a nonexistent name", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"][1]["kind"] = "theorem"
    cases.append(("mislabel a canonical def target as a theorem", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["rust"]["entry_points"][0]["file"] = "src/does_not_exist.rs"
    cases.append(("point a rust entry at a missing file", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["song"]["entry_points"][0]["symbol"] = "tableDoesNotExist"
    cases.append(("point a slang entry at a missing symbol", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["rust"][0]["kind"] = "fuel"
    cases.append(("rename a resource kind to generic fuel", mutated))

    mutated = copy.deepcopy(data)
    mutated["separate_admission"].pop("joint", None)
    cases.append(("remove the joint separate_admission clause", mutated))

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["targets"] = [
        t for t in mutated["rows"][0]["canonical"]["targets"]
        if t.get("name") != "Apeiron.Table.get_build_acc"
    ]
    mutated["rows"][5]["canonical"]["targets"] = [
        t for t in mutated["rows"][5]["canonical"]["targets"]
        if t.get("name") != "Apeiron.Table.get_build_acc"
    ]
    cases.append(("drop the get_build_acc canonical theorem target", mutated))

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
    _need(SPEC_PATH.is_file(), f"subprofile spec missing: {SPEC_PATH}")
    data = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    return data, SPEC_PATH.read_text(encoding="utf-8")


def _summary(data: dict) -> str:
    row_ids = [row["id"] for row in data["rows"]]
    hazard_ids = [h["id"] for h in data["hazards"]]
    control_ids = [c["id"] for c in data["controls"]]
    targets = sum(len(row["canonical"]["targets"]) for row in data["rows"])
    return (
        f"rows={row_ids} hazards={hazard_ids} controls={control_ids} "
        f"canonical_targets={targets}"
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
        print(f"FAIL: baseline subprofile is inconsistent: {exc}", file=sys.stderr)
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
            if "remove required field" in name
            or "remove required row" in name
            or "remove required hazard" in name
        ]
        _need(
            len(mandated) == 3,
            "self-test requires the required-field, required-row and required-hazard controls",
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
