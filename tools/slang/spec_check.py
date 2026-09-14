#!/usr/bin/env python3
"""Fail-closed consistency checker for the selected Slang semantics specification."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
SONG = ROOT.parent / "song"
CORPUS = ROOT.parent / "sanguine-apeiron" / "proof"
SPEC_PATH = ROOT / "docs/spec/slang-source-semantics.md"
PLAN_PATH = ROOT / "docs/plan/slang-source-refinement.md"
SONG_REV = "aeaf3baac434b2e3c01154d275bea65fdadc69bf"
CORPUS_REV = "50c94192399add3b9f8e64bd14e36d8cba270705"

OBLIGATIONS = tuple(f"O{i:02d}" for i in range(1, 19))
SECTIONS = (
    "Required obligations",
    "Selected source and identity",
    "Declaration closure",
    "Grammar and source binding",
    "Word and expression semantics",
    "Local state, calls, and control",
    "Memory and extents",
    "Checked execution outcomes and admission",
    "Representation and observation",
    "Required theorem shapes",
    "Termination and source resources",
    "Not covered",
    "Checker",
)
STAGES = tuple(f"W{i}" for i in range(21, 31))

SOURCES = {
    "src/table/table.slang": "bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d",
    "src/kernel/std.slang": "e24b0e360f26794fab0fe1332c32455143d22d2059e871b8c012b4e8a89a6cae",
    "src/kernel/mem/cell.slang": "35e3e46c5b7820c67f9c92f2af07a16bf581964b90acf7c508c5b13570d9b3cc",
    "src/front/parser.slang": "e643bd85ca34b195888a0c64ec3662ce75c0c770d365b360285a9e0af2e4a882",
    "src/front/node.slang": "db096abda00e40ddf6114db7d8da8590b4e03e46390eb4fd38ce0477679b2d6c",
    "src/front/bridge.slang": "6d3aef63833ac4c2eaf99c71b044b7ae22346a7020c414ceb73ad03f5b4466c0",
    "src/emit/isa.slang": "da75ad47bbbbb4ddb8a22c2d5d2729910583f0e37fcec6a198aa7257dd5e79f9",
}
FUNCTIONS = (
    "tableSpread",
    "tableSlotKey",
    "tableSlotVal",
    "tableSlotSet",
    "tableSlotValSet",
    "tableInitGo",
    "tableInit",
    "tableCapacityOk",
    "tableFindSlotGo",
    "tableLookupSlotGo",
    "tableFind",
    "tableGet",
    "tableInsertWith",
)
CONSTANTS = (
    "TABLE_PRESENT",
    "TABLE_EMPTY",
    "TABLE_STRIDE",
    "MERGE_SET",
    "MERGE_MAP",
    "MERGE_ACC",
    "MERGE_GRAPH",
    "TABLE_PHI",
)
CANONICAL = {
    "HMap": "abbrev",
    "get": "def",
    "insertWith": "def",
    "build": "def",
    "get_insertWith_self": "theorem",
    "get_insertWith_ne": "theorem",
}
PHRASES = (
    "64-bit wrapping multiplication",
    "do **not** call `cellGet` or `cellSet`",
    "**NO checked table refusal encoding**",
    "timeout is inconclusive",
    "never `return != -1`",
    "RepSong(memory, base, cap, default, O, h)",
    "observeRead",
    "raw return and resulting state",
    "digest establishes byte equality only",
    "handwritten Lean algorithm",
    "complete brace-delimited bodies",
    "unsupported reachable construct MUST reject",
    "FORBIDDEN: prove all Slang",
    "FORBIDDEN: implement a Rust evaluator",
)


class CheckError(Exception):
    pass


@dataclass(frozen=True)
class Inputs:
    spec: str
    plan: str
    sources: dict[str, str]
    corpus: str


def fail(message: str) -> None:
    raise CheckError(message)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load() -> Inputs:
    try:
        source_text = {name: (SONG / name).read_text(encoding="utf-8") for name in SOURCES}
        return Inputs(
            SPEC_PATH.read_text(encoding="utf-8"),
            PLAN_PATH.read_text(encoding="utf-8"),
            source_text,
            (CORPUS / "Apeiron/Table/Basic.lean").read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeError) as exc:
        fail(f"unresolved file: {exc}")


def git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"repository identity unresolved: {path}: {exc}")
    return result.stdout.strip()


def require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        fail(f"{label}: expected once, found {count}")


def check_markdown(inputs: Inputs) -> None:
    for section in SECTIONS:
        count = len(re.findall(rf"^## {re.escape(section)}$", inputs.spec, re.MULTILINE))
        if count != 1:
            fail(f"section {section}: expected once, found {count}")
    for oid in OBLIGATIONS:
        matches = re.findall(rf"^\| {oid} \|", inputs.spec, re.MULTILINE)
        if len(matches) != 1:
            fail(f"obligation {oid}: expected one table row")
    for phrase in PHRASES[:12]:
        if phrase not in inputs.spec:
            fail(f"required statement missing: {phrase}")
    for phrase in PHRASES[12:]:
        if phrase not in inputs.plan:
            fail(f"required plan boundary missing: {phrase}")
    forbidden_budget = "fu" + "el"
    if re.search(rf"\b{forbidden_budget}\b", inputs.spec + inputs.plan, re.IGNORECASE):
        fail("generic scalar execution budget is forbidden in this specification route")


def check_sources(inputs: Inputs) -> None:
    if git_revision(SONG) != SONG_REV or SONG_REV not in inputs.spec:
        fail("Song repository identity mismatch")
    if git_revision(CORPUS.parent) != CORPUS_REV or CORPUS_REV not in inputs.spec:
        fail("canonical corpus identity mismatch")
    for name, expected in SOURCES.items():
        text = inputs.sources.get(name)
        if text is None:
            fail(f"source missing: song/{name}")
        actual = sha256(text)
        if actual != expected:
            fail(f"source digest mismatch: song/{name}: {actual}")
        row = f"`song/{name}` | `{expected}`"
        if row not in inputs.spec:
            fail(f"source identity row missing: song/{name}")

    table = inputs.sources["src/table/table.slang"]
    std = inputs.sources["src/kernel/std.slang"]
    cell = inputs.sources["src/kernel/mem/cell.slang"]
    if 'use "src/kernel/mem/cell.slang"' not in table:
        fail("table import edge unresolved")
    if 'use "src/kernel/std.slang"' not in cell:
        fail("cell import edge unresolved")
    if not re.search(r"^COORD_CEILING\s*=\s*8796093022208\s*$", std, re.MULTILINE):
        fail("COORD_CEILING definition unresolved")
    for name in CONSTANTS:
        if not re.search(rf"^{re.escape(name)}\s*=", table, re.MULTILINE):
            fail(f"selected constant unresolved: {name}")
        if f"`{name}`" not in inputs.spec:
            fail(f"selected constant absent from spec: {name}")
    for name in FUNCTIONS:
        if not re.search(rf"^{re.escape(name)}\([^\n]*\)\s*\{{", table, re.MULTILINE):
            fail(f"selected function unresolved: {name}")
        if f"`{name}`" not in inputs.spec:
            fail(f"selected function absent from spec: {name}")
    raw = ("return memory[" in table and "memory[base + TABLE_STRIDE * slot] = key;" in table)
    if not raw:
        fail("raw table memory access unresolved")


def check_canonical(inputs: Inputs) -> None:
    if "namespace Apeiron\nnamespace Table" not in inputs.corpus:
        fail("canonical namespace unresolved")
    for name, kind in CANONICAL.items():
        pattern = rf"^{kind}\s+{re.escape(name)}\b"
        if not re.search(pattern, inputs.corpus, re.MULTILINE):
            fail(f"canonical target unresolved: Apeiron.Table.{name} ({kind})")
        require_once(inputs.spec, f"`Apeiron.Table.{name}`", f"canonical target {name}")


def stage_row(plan: str, stage: str) -> str:
    rows = [line for line in plan.splitlines() if line.startswith(f"| {stage} |")]
    if len(rows) != 1:
        fail(f"stage {stage}: expected one task row")
    return rows[0]


def check_plan(inputs: Inputs) -> None:
    for stage in STAGES:
        row = stage_row(inputs.plan, stage)
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if len(cells) != 6:
            fail(f"stage {stage}: expected six columns")
        owned = [item for item in cells[2].split(";") if "`" in item]
        if not 1 <= len(owned) <= 3:
            fail(f"stage {stage}: owned-file count is {len(owned)}, expected 1..3")
        if not cells[3].startswith("`") or not cells[3].endswith("`"):
            fail(f"stage {stage}: gate missing")
        if "MUST" not in cells[4] or "reject" not in cells[4] and "fail" not in cells[4]:
            fail(f"stage {stage}: negative control missing")
        if "Requires" not in cells[5]:
            fail(f"stage {stage}: prerequisite missing")
    if inputs.plan.find("W21") > inputs.plan.find("W30"):
        fail("stage integration order reversed")


def validate(inputs: Inputs) -> None:
    check_markdown(inputs)
    check_sources(inputs)
    check_canonical(inputs)
    check_plan(inputs)


Mutation = tuple[str, Callable[[Inputs], Inputs]]


def sub_spec(old: str, new: str = "") -> Callable[[Inputs], Inputs]:
    return lambda i: replace(i, spec=i.spec.replace(old, new, 1))


def sub_plan(old: str, new: str = "") -> Callable[[Inputs], Inputs]:
    return lambda i: replace(i, plan=i.plan.replace(old, new, 1))


def sub_spec_all(old: str, new: str = "") -> Callable[[Inputs], Inputs]:
    return lambda i: replace(i, spec=i.spec.replace(old, new))


def source_sub(path: str, old: str, new: str) -> Callable[[Inputs], Inputs]:
    def mutate(i: Inputs) -> Inputs:
        changed = dict(i.sources)
        changed[path] = changed[path].replace(old, new, 1)
        return replace(i, sources=changed)
    return mutate


def controls(inputs: Inputs) -> list[Mutation]:
    cases: list[Mutation] = []
    for oid in OBLIGATIONS:
        row = next(line for line in inputs.spec.splitlines() if line.startswith(f"| {oid} |"))
        cases.append((f"remove obligation {oid}", sub_spec(row + "\n")))
    for section in SECTIONS:
        cases.append((f"remove section {section}", sub_spec(f"## {section}", f"### {section}")))
    for stage in STAGES:
        row = stage_row(inputs.plan, stage)
        cases.append((f"remove stage {stage}", sub_plan(row + "\n")))
    for phrase in PHRASES[:12]:
        cases.append((f"remove statement {phrase[:24]}", sub_spec_all(phrase, "REMOVED")))
    for phrase in PHRASES[12:]:
        cases.append((f"remove boundary {phrase}", sub_plan(phrase, "REMOVED")))
    cases.extend(
        [
            ("change selected source digest", sub_spec(next(iter(SOURCES.values())), "0" * 64)),
            ("change Song revision", sub_spec(SONG_REV, "0" * 40)),
            ("change corpus revision", sub_spec(CORPUS_REV, "1" * 40)),
            ("remove table import", source_sub("src/table/table.slang", 'use "src/kernel/mem/cell.slang"', 'use "missing.slang"')),
            ("remove std import", source_sub("src/kernel/mem/cell.slang", 'use "src/kernel/std.slang"', 'use "missing.slang"')),
            ("change COORD_CEILING", source_sub("src/kernel/std.slang", "COORD_CEILING = 8796093022208", "COORD_CEILING = 7")),
            ("remove selected function", source_sub("src/table/table.slang", "tableGet(base, cap, key) {", "tableGone(base, cap, key) {")),
            ("remove selected constant", source_sub("src/table/table.slang", "TABLE_PHI =", "TABLE_GONE =")),
            ("remove canonical theorem", lambda i: replace(i, corpus=i.corpus.replace("theorem get_insertWith_ne", "theorem gone", 1))),
            ("remove canonical namespace", lambda i: replace(i, corpus=i.corpus.replace("namespace Table", "namespace Gone", 1))),
            ("remove W21 gate", sub_plan("`python3 tools/slang/source_semantics_check.py --check harness`", "MISSING")),
            ("remove W27 control", sub_plan("preservation/admission MUST fail", "control omitted")),
            ("introduce forbidden generic budget term", sub_spec("Counts are not converted", ("fu" + "el") + " is allowed. Counts are not converted")),
        ]
    )
    return cases


def run_controls(inputs: Inputs, verbose: bool) -> int:
    validate(inputs)
    cases = controls(inputs)
    survived: list[str] = []
    for index, (name, mutate) in enumerate(cases, 1):
        try:
            validate(mutate(inputs))
        except CheckError as exc:
            if verbose and index == 1:
                print(f"CONTROL RED: {name}: {exc}")
        else:
            survived.append(name)
    if survived:
        fail("controls survived: " + "; ".join(survived))
    print(f"CONTROLS PASS: {len(cases)}/{len(cases)} requirement mutations rejected")
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--controls", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        inputs = load()
        if args.check:
            validate(inputs)
            print(
                f"CHECK PASS: {len(OBLIGATIONS)} obligations, {len(STAGES)} stages, "
                f"{len(SOURCES)} sources, {len(CANONICAL)} canonical targets"
            )
        elif args.controls:
            run_controls(inputs, True)
        else:
            count = run_controls(inputs, True)
            validate(inputs)
            print(f"SELF-TEST PASS: baseline accepted; {count} mutations rejected")
        return 0
    except CheckError as exc:
        print(f"SPEC CHECK FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
