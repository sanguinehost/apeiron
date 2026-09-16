#!/usr/bin/env python3
"""Export the exact Song table import closure as Lean UTF-8 byte strings.

This program is an untrusted renderer.  Lean independently tokenizes and
decodes its output in Slang.SourceAudit.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SONG_ROOT = REPO_ROOT.parent / "song"
DEFAULT_OUTPUT = REPO_ROOT / "proof/corpus/Slang/Generated/Table.lean"
DEFAULT_DERIVATION_OUTPUT = REPO_ROOT / "proof/corpus/Slang/Generated/TableDerivation.lean"
ENTRY = "src/table/table.slang"
USE = re.compile(rb'(?m)^use[ \t]+"([^"\r\n]+)"[ \t]*(?://[^\r\n]*)?$')


def lean_string(text: str) -> str:
    escaped: list[str] = []
    for char in text:
        escaped.append(
            {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(
                char, char
            )
        )
    return '"' + "".join(escaped) + '"'


# A single definition whose term holds more than this many `sourceByte`
# applications exceeds the 10000 recursion budget the Slang modules use, so a
# source longer than this is bound in parts of this many bytes and concatenated:
# each definition stays inside the budget and the final `<stem>Source` denotes
# the same list.
SOURCE_DEF_ELEMENTS = 4000


def source_bytes_literal(data: bytes) -> str:
    """A literal ``SourceBytes`` term built from ``sourceByte`` applications.

    ``(n : Fin 256)`` numerals elaborate through an ``OfNat`` instance whose proof
    depends on ``propext``, which the Slang chain rejects; ``sourceByte`` uses
    ``Nat.mod_lt`` and stays inside the empty axiom cone.  The kernel reduces the
    applications to ``Fin`` values directly, so no ``String.toUTF8`` (stuck in the
    kernel) appears in a proof input.
    """
    values = [f"sourceByte {value}" for value in data]
    width = 8
    rows = [
        "    " + ", ".join(values[index : index + width])
        + ("," if index + width < len(values) else "")
        for index in range(0, len(values), width)
    ]
    if not rows:
        return "  []"
    return "  [\n" + "\n".join(rows) + "\n  ]"


def source_defs(stem: str, data: bytes) -> list[str]:
    """The definitions binding ``{stem}Source`` to ``data``.

    A definition's recursion demand scales with the size of its term, so the
    27 KiB `std` source cannot be one literal under the 10000 budget even when
    the literal is nested.  Sources longer than `SOURCE_DEF_ELEMENTS` are bound
    in ``private`` part definitions and concatenated; ``{stem}Source`` is the
    same list.
    """
    parts = [
        data[start : start + SOURCE_DEF_ELEMENTS]
        for start in range(0, len(data), SOURCE_DEF_ELEMENTS)
    ] or [b""]
    if len(parts) == 1:
        return [
            f"def {stem}Source : Slang.Source.SourceBytes :=",
            source_bytes_literal(data),
        ]
    names = [f"{stem}SourcePart{index}" for index in range(len(parts))]
    lines: list[str] = []
    for name, part in zip(names, parts):
        lines += [
            f"private def {name} : Slang.Source.SourceBytes :=",
            source_bytes_literal(part),
            "",
        ]
    lines += [
        f"def {stem}Source : Slang.Source.SourceBytes :=",
        "  " + " ++ ".join(names),
    ]
    return lines


SOURCE_BYTE_HELPER = (
    "private def sourceByte (value : Nat) : Slang.Source.SourceByte :=",
    "  ⟨value % 256, Nat.mod_lt value (Nat.zero_lt_succ 255)⟩",
    "",
)


def resolve_closure(song_root: Path) -> list[tuple[str, bytes]]:
    root = song_root.resolve()
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[tuple[str, bytes]] = []

    def visit(relative: str) -> None:
        if relative in visiting:
            raise ValueError(f"cyclic import at {relative}")
        if relative in visited:
            return
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"invalid import {relative}")
        data = target.read_bytes()
        data.decode("utf-8")
        visiting.add(relative)
        ordered.append((relative, data))
        for match in USE.finditer(data):
            visit(match.group(1).decode("utf-8"))
        visiting.remove(relative)
        visited.add(relative)

    visit(ENTRY)
    return ordered


def song_revision(song_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=song_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    revision = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Song revision is not a 40-hex Git identity")
    return revision


def render(song_root: Path) -> tuple[str, dict[str, str]]:
    closure = resolve_closure(song_root)
    if [path for path, _ in closure] != [
        "src/table/table.slang",
        "src/kernel/mem/cell.slang",
        "src/kernel/std.slang",
    ]:
        raise ValueError("selected table import closure differs")
    names = {
        "src/table/table.slang": "tableBytes",
        "src/kernel/mem/cell.slang": "cellBytes",
        "src/kernel/std.slang": "stdBytes",
    }
    paths = {names[path][:-5]: path for path, _ in closure}
    lines = [
        "-- Generated by tools/slang/export.py; do not edit.",
        "import Slang.Source.Bytes",
        "namespace Slang.Generated.Table",
        "",
        "-- Literal byte lists of tens of thousands of elements need a deeper elaborator recursion",
        "-- budget; the kernel still reduces each `sourceByte` to its `Fin` value.",
        "set_option maxRecDepth 10000",
        "",
        *SOURCE_BYTE_HELPER,
        f"def songRevision : String := {lean_string(song_revision(song_root))}",
        f"def entryPath : String := {lean_string(ENTRY)}",
        "",
    ]
    for path, data in closure:
        digest = hashlib.sha256(data).hexdigest()
        stem = names[path][:-5]
        lines.extend(
            [
                f"def {stem}Path : String := {lean_string(path)}",
                f"def {stem}Sha256 : String := {lean_string(digest)}",
                *source_defs(stem, data),
                f"def {names[path]} : ByteArray := Slang.Source.encodeBytes {stem}Source",
                "",
            ]
        )
    lines.extend(
        [
            "def closurePaths : List String := [tablePath, cellPath, stdPath]",
            "def closureBytes : List ByteArray := [tableBytes, cellBytes, stdBytes]",
            "",
            "end Slang.Generated.Table",
            "",
        ]
    )
    return "\n".join(lines), paths


def render_derivation(paths: dict[str, str]) -> str:
    entry_path = source_bytes_literal(paths["table"].encode("utf-8"))
    cell_path = source_bytes_literal(paths["cell"].encode("utf-8"))
    std_path = source_bytes_literal(paths["std"].encode("utf-8"))
    return "\n".join(
        [
            "-- Generated by tools/slang/export.py; do not edit.",
            "import Slang.Source.ClosedBinding",
            "import Slang.Generated.Table",
            "",
            "namespace Slang.Generated.TableDerivation",
            "",
            "open Slang Slang.Decode Slang.Source",
            "",
            "set_option autoImplicit false",
            "set_option warningAsError true",
            "",
            *SOURCE_BYTE_HELPER,
            "/-- The exported bytes as source bytes, read by the compiled evaluator. -/",
            "def toSource (packed : ByteArray) : SourceBytes := packed.data.toList.map UInt8.toFin",
            "",
            "def tableSource : SourceBytes := Slang.Generated.Table.tableSource",
            "def cellSource : SourceBytes := Slang.Generated.Table.cellSource",
            "def stdSource : SourceBytes := Slang.Generated.Table.stdSource",
            "",
            "def entryPath : SourceBytes :=",
            entry_path,
            "def cellPath : SourceBytes :=",
            cell_path,
            "def stdPath : SourceBytes :=",
            std_path,
            "",
            "def entryBundle : SourceBundle :=",
            "  { entry := entryPath",
            "    files := [⟨entryPath, tableSource⟩, ⟨cellPath, cellSource⟩, ⟨stdPath, stdSource⟩] }",
            "",
            "def decodeAccepts (source : SourceBytes) : Bool :=",
            "  match Decode.decodeTable (encodeBytes source) with",
            "  | .ok _ => true",
            "  | .error _ => false",
            "",
            "-- Concrete whole-source facts, gated by the compiled evaluator: a false result fails the",
            "-- build, adds no declaration and no axiom. The real entry decodes, is selected-exact, has",
            "-- exactly its one declared import edge, and the bundle resolves cell and std.",
            "#guard importPaths tableSource == [cellPath]",
            "#guard importPaths cellSource == [stdPath]",
            "#guard decodeAccepts tableSource",
            "#guard selectedSourceExact tableSource",
            "#guard (entryBundle.fileBytes? cellPath).isSome",
            "#guard (entryBundle.fileBytes? stdPath).isSome",
            "",
            "end Slang.Generated.TableDerivation",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song-root", type=Path, default=DEFAULT_SONG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--derivation-output", type=Path, default=DEFAULT_DERIVATION_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    song_root = args.song_root.resolve()
    generated, paths = render(song_root)
    derivation = render_derivation(paths)
    outputs: list[tuple[Path, str]] = [
        (args.output, generated),
        (args.derivation_output, derivation),
    ]
    if args.check:
        for path, text in outputs:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise SystemExit(f"generated source binding differs: {path}")
    else:
        for path, text in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
