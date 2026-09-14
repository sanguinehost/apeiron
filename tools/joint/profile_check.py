#!/usr/bin/env python3
"""Consistency checker for the W17-T1 joint primitive profile.

This tool establishes **specification consistency only**. It verifies that
``docs/spec/joint-primitives.md`` and ``tools/joint/profile_rows.json`` are
internally complete and that every artifact they name resolves:

* every joint profile row carries all seven populated fields, in both the
  Markdown document and the JSON profile;
* every required blocker and hazard is present as an actual Markdown table
  record and as a JSON record, not merely an incidental ID mention;
* every named canonical theorem resolves as a fully qualified declaration in
  the pinned canonical corpus (``../sanguine-apeiron/proof``), and every named
  bridge refinement resolves in ``proof/``;
* every named Rust and Song entry exists as a concrete declaration in its
  source file, and every Song entry is an executing function rather than a
  ``law``/``carrier``/``spec`` header;
* the frozen selected rows remain ``selected-open`` and no row is ``closed``;
* the resource accounting names all five distinct kinds separately for every
  implementation of every row, and no generic fuel;
* the recorded corpus/Mathlib pins, Apeiron/Song revisions and per-file source
  hashes match the frozen identities and the files on disk;
* every named reference, including blocker entries, resolves.

It does **not** establish semantic correspondence, refinement or proof. A green
run says the specification is well formed and source-connected by name; it says
nothing about whether any row's refinement has been proved.

Usage::

    python3 tools/joint/profile_check.py --check
    python3 tools/joint/profile_check.py --controls
    python3 tools/joint/profile_check.py --self-test

``--controls`` and ``--self-test`` mutate a deep copy of the loaded profile and
the spec text in memory and require each mutation to be rejected. They never
touch the working tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

# Frozen identities. These are the pins the evidence binds to; a recorded value
# that differs from its constant, or a source file whose hash differs from its
# recorded value, is a rejected profile.
EXPECTED_CORPUS_GIT = "50c94192399add3b9f8e64bd14e36d8cba270705"
EXPECTED_MATHLIB = "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
EXPECTED_APEIRON_REVISION = "2b5eda334e262f70f7c10ce567c8641c2396626d"
EXPECTED_SONG_REVISION = "47869e76f982cfa3085a8618a82b80463e22ef46"
REQUIRED_SOURCE_FILES = {
    "src/scalar.rs": "9199e955304cb3d54fe973ac5881f219d4c6f71b4be001a0d32a6fb16ffde92e",
    "src/table.rs": "83597ecf7eb60d28b4545353ec32a296d846bc6fca95b9f4f6dd20eee72b600a",
    "song/src/table/table.slang": "bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d",
}

SEVEN_FIELDS = [
    "canonical",
    "entry_points",
    "representation_invariant",
    "admission",
    "behavior",
    "resources",
    "refinement",
]

# The seven numbered field records each Markdown row must carry.
FIELD_LABELS = [
    ("1", "Canonical object and observation relation"),
    ("2", "Entry points and source identity"),
    ("3", "Representation invariant"),
    ("4", "Admission predicates"),
    ("5", "Behaviour"),
    ("6", "Resource accounting"),
    ("7", "Source-connected refinement and rejecting control"),
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
ALLOWED_STATUSES = ("selected-open", "blocker")

LEAN_DECL = (
    r"(?m)^\s*(?:noncomputable\s+|private\s+|protected\s+|@\[[^\]]*\]\s*)*"
    r"(?:theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
)
RUST_DECL = (
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|unsafe\s+)?"
    r"(?:fn|struct|enum|trait|union|type|mod)\s+"
)
RUST_IMPL = r"(?m)^\s*impl\b[^\n{]*\b"
# A Song entry must be an executing function definition, not a law/carrier/spec
# header. Only ``name(`` at the start of a line is accepted.
SLANG_DECL = r"(?m)^\s*{n}\s*\("


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


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _resolve_path(rel: str, where: str) -> Path:
    for base in (REPO, REPO.parent):
        candidate = base / rel
        if candidate.is_file():
            return candidate
    raise CheckError(f"{where}: source file does not resolve: {rel}")


def _opened_namespaces(text: str) -> set[str]:
    """Return the dotted namespace paths opened in a Lean file, in order.

    ``end`` lines pop the most recent namespace. This rejects a reversed
    qualification such as ``Table.Apeiron.get_build_acc`` even though both
    segments appear somewhere in the file.
    """
    paths: set[str] = set()
    stack: list[str] = []
    pattern = re.compile(r"(?m)^\s*(?:namespace\s+([\w.]+)|end\b[^\n]*)")
    for match in pattern.finditer(text):
        name = match.group(1)
        if name is not None:
            stack.extend(name.split("."))
            paths.add(".".join(stack))
        elif stack:
            stack.pop()
    return paths


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
    prefix = ".".join(segments[:-1])
    if prefix:
        opened = _opened_namespaces(text)
        _need(
            any(path_ == prefix or path_.startswith(prefix + ".") for path_ in opened),
            f"{where}: namespace {prefix} of {name} not opened in {module}",
        )


def _check_rust(text: str, symbol: str) -> bool:
    if "::" in symbol:
        qualifier, method = symbol.rsplit("::", 1)
        method = method.strip()
        if re.fullmatch(r"[A-Za-z_]\w*", method) is None:
            return False
        qualifier = qualifier.strip()
        if qualifier.startswith("<") and qualifier.endswith(">"):
            qualifier = qualifier[1:-1].split(" as ", 1)[0].strip()
        qualifier = re.split(r"[^A-Za-z0-9_]", qualifier)[-1]
        if not qualifier:
            return False
        if re.search(RUST_DECL + re.escape(method) + r"\b", text) is None:
            return False
        return re.search(r"\b" + re.escape(qualifier) + r"\b", text) is not None
    return (
        re.search(RUST_DECL + re.escape(symbol) + r"\b", text) is not None
        or re.search(RUST_IMPL + re.escape(symbol) + r"\b", text) is not None
    )


def _check_slang(text: str, symbol: str) -> bool:
    return re.search(SLANG_DECL.format(n=re.escape(symbol)), text) is not None


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


def _validate_row(row: object, seen_ids: set[str]) -> None:
    _need(isinstance(row, dict), "each row must be an object")
    rid = row.get("id")
    _nonempty(rid, "row.id")
    _need(rid not in seen_ids, f"duplicate row id {rid}")
    seen_ids.add(rid)
    _nonempty(row.get("title"), f"{rid}.title")
    status = row.get("status")
    _need(
        status in ALLOWED_STATUSES,
        f"{rid}.status invalid: {status!r}; a closed row is not permitted",
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
        kinds: list[str] = []
        for entry in entries:
            kind = entry.get("kind") if isinstance(entry, dict) else None
            _need(
                kind in REQUIRED_RESOURCE_KINDS,
                f"{rid}.resources.{side} invalid kind {kind!r}",
            )
            _need(
                kind not in kinds,
                f"{rid}.resources.{side} repeats kind {kind!r}",
            )
            kinds.append(kind)
            _nonempty(entry.get("accounting"), f"{rid}.resources.{side}[{kind}]")
        missing = sorted(REQUIRED_RESOURCE_KINDS - set(kinds))
        _need(
            not missing,
            f"{rid}.resources.{side} must name all five kinds separately; "
            f"missing {missing}",
        )

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
    rust_entry = blocker.get("rust_entry")
    song_entry = blocker.get("song_entry")
    _need(
        theorem is not None or rust_entry is not None or song_entry is not None,
        f"{bid} carries no resolvable named reference",
    )
    if theorem is not None:
        _check_lean_ref(theorem, f"{bid}.canonical_theorem", require_root="corpus")
    if rust_entry is not None:
        _check_source_entry(rust_entry, f"{bid}.rust_entry")
    if song_entry is not None:
        _check_source_entry(song_entry, f"{bid}.song_entry")


def _markdown_table_first_cells(text: str) -> set[str]:
    cells: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\|\s*([^|]+?)\s*\|", line)
        if match is None:
            continue
        cell = match.group(1).strip().strip("`").strip()
        if cell and not set(cell) <= set("-: "):
            cells.add(cell)
    return cells


def _row_section(text: str, row_id: str) -> str:
    heading = re.search(r"(?m)^###\s+" + re.escape(row_id) + r"\b[^\n]*$", text)
    if heading is None:
        return ""
    start = heading.end()
    nxt = re.search(r"(?m)^#{1,3}\s+", text[start:])
    end = start + nxt.start() if nxt else len(text)
    return text[start:end]


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

    cells = _markdown_table_first_cells(text)
    for row_id in row_ids:
        _need(
            row_id in cells,
            f"spec is missing an actual Markdown table record for row {row_id}",
        )
        section = _row_section(text, row_id)
        _need(section.strip() != "", f"spec is missing a section for row {row_id}")
        for number, label in FIELD_LABELS:
            pattern = (
                r"\*\*" + re.escape(number) + r"\s*·\s*" + re.escape(label) + r"\.\*\*"
            )
            _need(
                re.search(pattern, section) is not None,
                f"spec row {row_id} is missing its Markdown field {number} "
                f"record ({label})",
            )
    for field in SEVEN_FIELDS:
        _need(
            field in cells,
            f"spec is missing an actual Markdown record for field {field}",
        )
    for bid in blocker_ids:
        _need(
            bid in cells,
            f"spec is missing an actual Markdown table record for blocker {bid}",
        )
    for hid in hazard_ids:
        _need(
            hid in cells,
            f"spec is missing an actual Markdown table record for hazard {hid}",
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
    _need(
        _is_hex(corpus.get("git"), 40),
        f"corpus.git must be a 40-hex revision, got {corpus.get('git')!r}",
    )
    _need(
        corpus.get("git") == EXPECTED_CORPUS_GIT,
        f"corpus.git pin mismatch: recorded {corpus.get('git')!r}, "
        f"expected {EXPECTED_CORPUS_GIT!r}",
    )
    _need(
        _is_hex(corpus.get("mathlib"), 40),
        f"corpus.mathlib must be a 40-hex revision, got {corpus.get('mathlib')!r}",
    )
    _need(
        corpus.get("mathlib") == EXPECTED_MATHLIB,
        f"corpus.mathlib pin mismatch: recorded {corpus.get('mathlib')!r}, "
        f"expected {EXPECTED_MATHLIB!r}",
    )

    identity = data.get("source_identity")
    _need(isinstance(identity, dict), "source_identity metadata missing")
    _need(
        identity.get("apeiron_revision") == EXPECTED_APEIRON_REVISION,
        f"Apeiron revision mismatch: recorded {identity.get('apeiron_revision')!r}, "
        f"expected {EXPECTED_APEIRON_REVISION!r}",
    )
    _need(
        identity.get("song_revision") == EXPECTED_SONG_REVISION,
        f"Song revision mismatch: recorded {identity.get('song_revision')!r}, "
        f"expected {EXPECTED_SONG_REVISION!r}",
    )
    files = identity.get("files")
    _need(isinstance(files, list) and files, "source_identity.files must be nonempty")
    recorded: dict[str, str] = {}
    for entry in files:
        _need(isinstance(entry, dict), "source_identity.files entries must be objects")
        rel = entry.get("path")
        digest = entry.get("sha256")
        _nonempty(rel, "source_identity.files.path")
        _need(
            _is_hex(digest, 64),
            f"source_identity.files[{rel}].sha256 must be 64-hex, got {digest!r}",
        )
        path = _resolve_path(rel, "source_identity.files")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _need(
            actual == digest,
            f"source hash mismatch for {rel}: recorded {digest}, actual {actual}",
        )
        recorded[rel] = digest
    for rel, expected in REQUIRED_SOURCE_FILES.items():
        _need(
            rel in recorded,
            f"source_identity.files is missing required source {rel}",
        )
        _need(
            recorded[rel] == expected,
            f"source_identity.files[{rel}] pin mismatch: recorded {recorded[rel]}, "
            f"expected {expected}",
        )

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
    for row in rows:
        _validate_row(row, seen_ids)
    missing_rows = [r for r in REQUIRED_ROWS if r not in seen_ids]
    _need(not missing_rows, f"required selected rows dropped: {missing_rows}")
    for row in rows:
        if row.get("id") in REQUIRED_ROWS:
            _need(
                row.get("status") == "selected-open",
                f"frozen selected row {row.get('id')} must remain selected-open, "
                f"got {row.get('status')!r}",
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


def _remove_table_record(spec_text: str, token: str) -> str:
    return re.sub(
        r"(?m)^\|\s*" + re.escape(token) + r"\s*\|[^\n]*\n",
        "",
        spec_text,
    )


def _control_cases(data: dict, spec_text: str) -> list[tuple[str, dict, str]]:
    cases: list[tuple[str, dict, str]] = []

    def add(name: str, mutated: dict, spec: str | None = None) -> None:
        cases.append((name, mutated, spec if spec is not None else spec_text))

    mutated = copy.deepcopy(data)
    mutated["rows"][0].pop("canonical", None)
    add("remove required field canonical from row R1", mutated)

    mutated = copy.deepcopy(data)
    mutated["unmatched_blockers"] = [
        b for b in mutated["unmatched_blockers"] if b.get("id") != "B1"
    ]
    add("remove unmatched blocker B1", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"] = mutated["rows"][1:]
    add("remove selected row R1", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["theorems"][0]["name"] = (
        "Apeiron.Table.does_not_exist"
    )
    add("rename canonical theorem to a nonexistent name", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["rust"][0]["file"] = "src/does_not_exist.rs"
    add("point a source entry at a missing file", mutated)

    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["rust"][0]["kind"] = "fuel"
    add("rename a resource kind to generic fuel", mutated)

    mutated = copy.deepcopy(data)
    mutated["hazards"] = [h for h in mutated["hazards"] if h.get("id") != "H1"]
    add("remove hazard H1", mutated)

    mutated = copy.deepcopy(data)
    mutated["seven_fields"] = mutated["seven_fields"][:6]
    add("drop a seven_fields entry", mutated)

    # Markdown record: remove only R1's representation-invariant field label.
    head, sep, tail = spec_text.partition("### R2 ·")
    mutated_spec = head.replace("**3 · Representation invariant.**", "", 1) + sep + tail
    add(
        "delete markdown R1 representation-invariant record",
        copy.deepcopy(data),
        mutated_spec,
    )

    # Markdown records: delete the B6 blocker and H3 hazard table rows.
    mutated_spec = _remove_table_record(spec_text, "B6")
    mutated_spec = _remove_table_record(mutated_spec, "H3")
    add(
        "delete markdown blocker B6 and hazard H3 records",
        copy.deepcopy(data),
        mutated_spec,
    )

    # Per-implementation resource completeness.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["resources"]["song"] = [
        e for e in mutated["rows"][0]["resources"]["song"] if e.get("kind") != "nodes"
    ]
    add("remove one resource kind from one implementation", mutated)

    # False pins: corpus revision and recorded source hash.
    mutated = copy.deepcopy(data)
    mutated["corpus"]["git"] = "0" * 40
    add("falsify a corpus pin", mutated)

    mutated = copy.deepcopy(data)
    mutated["source_identity"]["files"][0]["sha256"] = "0" * 64
    add("falsify a recorded source hash", mutated)

    # Closed status.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["status"] = "closed"
    add("mark a selected row closed", mutated)

    # Reversed namespace.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["canonical"]["theorems"][0]["name"] = (
        "Table.Apeiron.get_build_acc"
    )
    add("reverse a canonical namespace", mutated)

    # Non-executing Slang entry.
    mutated = copy.deepcopy(data)
    mutated["rows"][0]["entry_points"]["song"][0]["symbol"] = "tableslotkey_addr"
    add(
        "point a Song entry at a non-executing law/spec header",
        mutated,
    )

    # Blocker structured reference.
    mutated = copy.deepcopy(data)
    for blocker in mutated["unmatched_blockers"]:
        if blocker.get("id") == "B6":
            blocker.pop("canonical_theorem", None)
            blocker.pop("rust_entry", None)
    add("remove a blocker's structured reference", mutated)

    return cases


MANDATED_CONTROLS = [
    "remove required field canonical from row R1",
    "remove unmatched blocker B1",
    "delete markdown R1 representation-invariant record",
    "delete markdown blocker B6 and hazard H3 records",
    "remove one resource kind from one implementation",
    "falsify a corpus pin",
    "falsify a recorded source hash",
    "mark a selected row closed",
    "reverse a canonical namespace",
    "point a Song entry at a non-executing law/spec header",
    "remove a blocker's structured reference",
]


def run_controls(data: dict, spec_text: str) -> list[tuple[str, bool, str]]:
    """Baseline must pass; every in-memory mutation must be rejected."""
    validate(data, spec_text)
    results: list[tuple[str, bool, str]] = []
    for name, mutated, mutated_spec in _control_cases(data, spec_text):
        try:
            validate(mutated, mutated_spec)
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
    names: set[str] = set()
    for name, ok, detail in results:
        names.add(name)
        if ok:
            rejected += 1
            print(f"control RED (rejected): {name}")
        else:
            print(f"control GREEN (NOT rejected): {name}: {detail}", file=sys.stderr)
    if self_test:
        missing = [name for name in MANDATED_CONTROLS if name not in names]
        _need(
            not missing,
            f"self-test is missing mandated controls: {missing}",
        )
        for name, ok, _ in results:
            if name in MANDATED_CONTROLS:
                _need(ok, f"self-test mandated control was not rejected: {name}")
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
