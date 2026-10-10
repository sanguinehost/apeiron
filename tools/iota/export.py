#!/usr/bin/env python3
"""Export the exact Song table import closure as Lean UTF-8 byte strings.

This program is an untrusted renderer.  Lean independently tokenizes and
decodes its output in Iota.SourceAudit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


if __package__:
    from .export_parts.table import (
        check_output_map, reject_stale_shards,
        render_table_outputs as _render_table_outputs,
    )
else:
    from export_parts.table import (
        check_output_map, reject_stale_shards,
        render_table_outputs as _render_table_outputs,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SONG_ROOT = REPO_ROOT.parent / "song"
DEFAULT_OUTPUT = REPO_ROOT / "proof/corpus/Iota/Generated/Table.lean"
DEFAULT_DERIVATION_OUTPUT = REPO_ROOT / "proof/corpus/Iota/Generated/TableDerivation.lean"
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


# Source parts are also the leaves of the structural UTF-8 witness.  The bound
# is measured in bytes, and a part boundary never splits one UTF-8 code point.
SOURCE_PART_BYTES = 32


def source_bytes_literal(data: bytes) -> str:
    """A literal ``SourceBytes`` term built from ``sourceByte`` applications.

    ``(n : Fin 256)`` numerals elaborate through an ``OfNat`` instance whose proof
    depends on ``propext``, which the Iota chain rejects; ``sourceByte`` uses
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


def source_parts(data: bytes) -> list[bytes]:
    """Split UTF-8 at code-point boundaries into parts of at most 32 bytes.

    Time is linear in decoded code points and space is linear in input bytes:
    each code point is encoded once and each byte is copied into one part.
    """
    parts: list[bytes] = []
    current = bytearray()
    for char in data.decode("utf-8"):
        encoded = char.encode("utf-8")
        if current and len(current) + len(encoded) > SOURCE_PART_BYTES:
            parts.append(bytes(current))
            current.clear()
        current.extend(encoded)
    if current or not parts:
        parts.append(bytes(current))
    return parts


def balanced_expr(names: list[str]) -> str:
    """Build a logarithmic-depth append tree, preserving left-to-right order."""
    level = list(names)
    while len(level) > 1:
        level = [
            (f"({level[index]} ++ {level[index + 1]})"
             if index + 1 < len(level) else level[index])
            for index in range(0, len(level), 2)
        ]
    return level[0]


def source_defs(stem: str, data: bytes) -> list[str]:
    """The definitions binding ``{stem}Source`` to ``data``.

    Every named part is one code-point-aligned UTF-8 witness leaf.  The source
    is a balanced append tree, so structural map/length lemmas traverse a tree
    of logarithmic depth instead of searching a whole literal list.
    """
    parts = source_parts(data)
    names = [f"{stem}Part{index}" for index in range(len(parts))]
    lines: list[str] = []
    for name, part in zip(names, parts):
        lines += [
            f"def {name} : Iota.Source.SourceBytes :=",
            source_bytes_literal(part),
            f"theorem {name}Length : {name}.length = {len(part)} := rfl",
            "",
        ]
    nodes = [
        {"name": name, "length": len(part)}
        for name, part in zip(names, parts)
    ]
    level = 0
    while len(nodes) > 1:
        next_nodes: list[dict[str, object]] = []
        for index in range(0, len(nodes), 2):
            if index + 1 == len(nodes):
                next_nodes.append(nodes[index])
                continue
            left = nodes[index]
            right = nodes[index + 1]
            name = f"{stem}PartNode{level}_{index // 2}"
            length = int(left["length"]) + int(right["length"])
            lines += [
                f"def {name} : Iota.Source.SourceBytes := "
                f"{left['name']} ++ {right['name']}",
                f"theorem {name}Length : {name}.length = {length} := by "
                f"change ({left['name']} ++ {right['name']}).length = {length}; exact "
                f"Eq.trans (@Iota.Generated.Table.sourceLengthAppend _ {left['name']} "
                f"{right['name']}) "
                f"(Eq.trans (congrArg (fun value => value + {right['name']}.length) "
                f"{left['name']}Length) (congrArg (fun value => {left['length']} + value) "
                f"{right['name']}Length))",
                "",
            ]
            next_nodes.append({"name": name, "length": length})
        nodes = next_nodes
        level += 1
    root = str(nodes[0]["name"])
    lines += [
        f"def {stem}Source : Iota.Source.SourceBytes :=",
        f"  {root}",
        f"theorem {stem}SourceLength : {stem}Source.length = {len(data)} := by",
        f"  exact {root}Length",
    ]
    return lines


SOURCE_BYTE_HELPER = (
    "private def sourceByte (value : Nat) : Iota.Source.SourceByte :=",
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
        "-- Generated by tools/iota/export.py; do not edit.",
        "import Iota.Source.Bytes",
        "import Iota.Generated.TableSupport",
        "namespace Iota.Generated.Table",
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
                f"def {names[path]} : ByteArray := Iota.Source.encodeBytes {stem}Source",
                "",
            ]
        )
    lines.extend(
        [
            "def closurePaths : List String := [tablePath, cellPath, stdPath]",
            "def closureBytes : List ByteArray := [tableBytes, cellBytes, stdBytes]",
            "",
            "end Iota.Generated.Table",
            "",
        ]
    )
    return "\n".join(lines), paths


def render_table_outputs(song_root: Path, output: Path = DEFAULT_OUTPUT) -> dict[Path, str]:
    """Render table outputs in O(source/rendered bytes) time and space.

    The legacy renderer traverses source parts once; grouping traverses the
    resulting declarations once more without changing their bodies.
    """
    monolith, _paths = render(song_root)
    return _render_table_outputs(monolith, output, SOURCE_BYTE_HELPER, REPO_ROOT / "proof/corpus")


def render_derivation(paths: dict[str, str]) -> str:
    entry_path = source_bytes_literal(paths["table"].encode("utf-8"))
    cell_path = source_bytes_literal(paths["cell"].encode("utf-8"))
    std_path = source_bytes_literal(paths["std"].encode("utf-8"))
    return "\n".join(
        [
            "-- Generated by tools/iota/export.py; do not edit.",
            "import Iota.Source.ClosedBinding",
            "import Iota.Generated.Table",
            "",
            "namespace Iota.Generated.TableDerivation",
            "",
            "open Iota Iota.Decode Iota.Source",
            "",
            "set_option autoImplicit false",
            "set_option warningAsError true",
            "",
            *SOURCE_BYTE_HELPER,
            "/-- The exported bytes as source bytes, read by the compiled evaluator. -/",
            "def toSource (packed : ByteArray) : SourceBytes := packed.data.toList.map UInt8.toFin",
            "",
            "def tableSource : SourceBytes := Iota.Generated.Table.tableSource",
            "def cellSource : SourceBytes := Iota.Generated.Table.cellSource",
            "def stdSource : SourceBytes := Iota.Generated.Table.stdSource",
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
            "end Iota.Generated.TableDerivation",
            "",
        ]
    )


if __package__:
    from .export_parts.decode import (
        _TOKEN_KINDS,
        _SELECTED_CONSTANT_NAMES,
        _SELECTED_FUNCTION_NAMES,
        _STEM_PATHS,
        _DecodeError,
        _utf8_length,
        _is_space,
        _is_digit,
        _is_alpha,
        _is_alnum,
        lex_chars_sim,
        lex_source,
        _drop_newlines,
        _expect_kind,
        _bin_op,
        _literal,
        _name,
        _call,
        _memory,
        _binary,
        _conditional,
        _bind,
        _store,
        _return_,
        _expression,
        _if_,
        _parse_expr,
        _parse_expr_fuel,
        _parse_primary_fuel,
        _parse_args_fuel,
        _parse_binary_fuel,
        _parse_block_fuel,
        _parse_stmt_fuel,
        _parse_parameters,
        _parse_function,
        _take_line,
        _scan_selected,
        _scan_selected_fuel,
        _distinct,
        _delimiters_balanced,
        _spans_valid,
        _expression_calls_resolve,
        _statements_calls_resolve,
        _resolve_calls,
        _expression_names_resolve,
        _statements_names_resolve,
        _resolve_names,
        _token_to_json,
    )
else:
    from export_parts.decode import (
        _TOKEN_KINDS,
        _SELECTED_CONSTANT_NAMES,
        _SELECTED_FUNCTION_NAMES,
        _STEM_PATHS,
        _DecodeError,
        _utf8_length,
        _is_space,
        _is_digit,
        _is_alpha,
        _is_alnum,
        lex_chars_sim,
        lex_source,
        _drop_newlines,
        _expect_kind,
        _bin_op,
        _literal,
        _name,
        _call,
        _memory,
        _binary,
        _conditional,
        _bind,
        _store,
        _return_,
        _expression,
        _if_,
        _parse_expr,
        _parse_expr_fuel,
        _parse_primary_fuel,
        _parse_args_fuel,
        _parse_binary_fuel,
        _parse_block_fuel,
        _parse_stmt_fuel,
        _parse_parameters,
        _parse_function,
        _take_line,
        _scan_selected,
        _scan_selected_fuel,
        _distinct,
        _delimiters_balanced,
        _spans_valid,
        _expression_calls_resolve,
        _statements_calls_resolve,
        _resolve_calls,
        _expression_names_resolve,
        _statements_names_resolve,
        _resolve_names,
        _token_to_json,
    )


def _source_bytes(song_root: Path) -> dict:
    mapping: dict[str, tuple[str, bytes]] = {}
    for path, data in resolve_closure(song_root):
        stem = _STEM_PATHS.get(path)
        if stem is not None:
            mapping[stem] = (path, data)
    return mapping


def source_data(stem: str, song_root: Path = DEFAULT_SONG_ROOT) -> dict:
    """The decoded data for one generated source, as JSON-serialisable data.

    Mirrors `Decode.lexChars` + `Decode.scanSelected` over the real closure
    bytes: the byte list, code points, tokens, consumed byte count, the decoded
    constants and functions, the selected names in `selectedConstantNames`
    order, and the `std` imported-coordinate data.  Pure data; no Lean emission.
    """
    sources = _source_bytes(song_root)
    if stem not in sources:
        raise ValueError(f"unknown source stem {stem!r}")
    path, data = sources[stem]
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    text = data.decode("utf-8")
    chars = [ord(char) for char in text]
    tokens, consumed, _steps = lex_chars_sim(data)
    for token in tokens:
        if token[0] not in _TOKEN_KINDS:
            raise ValueError(f"token kind {token[0]!r} is not in Decode.TokenKind")
    constants, functions = _scan_selected(
        tokens, _SELECTED_CONSTANT_NAMES, _SELECTED_FUNCTION_NAMES
    )
    selected_constants = [decl["name"] for decl in constants]
    selected_functions = [decl["name"] for decl in functions]
    imported_coord = None
    if stem == "std":
        coord_constants, _ = _scan_selected(tokens, ["COORD_CEILING"], [])
        imported_coord = coord_constants[0] if coord_constants else None
    coord_decl = {"name": "COORD_CEILING", "value": _literal(8796093022208), "span": [0, 1]}
    spans = [decl["span"] for decl in constants] + [decl["span"] for decl in functions]
    constant_names = [decl["name"] for decl in constants]
    function_names = [decl["name"] for decl in functions]
    # The closure constant set is `constants ++ [COORD_CEILING]` on the table
    # path; `std` already supplies it, so append only when absent.
    closure_names = constant_names + (
        [] if "COORD_CEILING" in constant_names else ["COORD_CEILING"]
    )
    checks = {
        "consumedEqualsSize": consumed == len(data),
        "delimitersBalanced": _delimiters_balanced(tokens),
        "constantCount": len(constants),
        "functionCount": len(functions),
        "constantNamesDistinct": _distinct(closure_names),
        "functionNamesDistinct": _distinct(function_names),
        "spansValid": _spans_valid(len(data), spans),
        "resolveCalls": _resolve_calls(functions),
        "resolveNames": _resolve_names(constants + [coord_decl], functions),
        "decodeTableAccepts": (
            consumed == len(data)
            and len(constants) == 8
            and len(functions) == len(_SELECTED_FUNCTION_NAMES)
            and _distinct(closure_names)
            and _distinct(function_names)
            and _spans_valid(len(data), spans)
            and _resolve_calls(functions)
            and _resolve_names(constants + [coord_decl], functions)
        ),
    }
    return {
        "stem": stem,
        "path": path,
        "bytes": list(data),
        "chars": chars,
        "tokens": [_token_to_json(token) for token in tokens],
        "consumed": consumed,
        "decoded": {"constants": constants, "functions": functions},
        "selectedConstantNames": [
            name for name in _SELECTED_CONSTANT_NAMES if name in selected_constants
        ],
        "selectedFunctionNames": [
            name for name in _SELECTED_FUNCTION_NAMES if name in selected_functions
        ],
        "importedCoord": imported_coord,
        "checks": checks,
    }


def _source_data_summary(stem: str, data: dict) -> str:
    return (
        f"{stem}: bytes={len(data['bytes'])} chars={len(data['chars'])} "
        f"tokens={len(data['tokens'])} consumed={data['consumed']} "
        f"constants={data['selectedConstantNames']} "
        f"functions={data['selectedFunctionNames']}"
    )


if __package__:
    from .export_parts.scan import (
        _SCAN_CHUNK,
        _SCAN_BIN_OPS,
        _SCAN_COUNTERS,
        _WITNESS_BODY_LINES,
        _WITNESS_AUDIT_NAMESPACE,
        _WITNESS_AUDIT_PATH,
        _WITNESS_PROGRAMS,
        _scan_str,
        _scan_span,
        _scan_token,
        _scan_expr,
        _scan_stmt,
        _scan_decl,
        _scan_cdecl,
        _StructuralScan,
        _witness_block_lines,
        _pack_witness_blocks,
        _render_witness_module,
        render_source_witness as _render_source_witness,
    )
else:
    from export_parts.scan import (
        _SCAN_CHUNK,
        _SCAN_BIN_OPS,
        _SCAN_COUNTERS,
        _WITNESS_BODY_LINES,
        _WITNESS_AUDIT_NAMESPACE,
        _WITNESS_AUDIT_PATH,
        _WITNESS_PROGRAMS,
        _scan_str,
        _scan_span,
        _scan_token,
        _scan_expr,
        _scan_stmt,
        _scan_decl,
        _scan_cdecl,
        _StructuralScan,
        _witness_block_lines,
        _pack_witness_blocks,
        _render_witness_module,
        render_source_witness as _render_source_witness,
    )


def render_source_witness(song_root: Path) -> tuple[
        dict[Path, str], list[dict[str, str]], list[str]]:
    """Render the structural `ScanSelected` witness module hierarchy.

    Returns the repo-absolute path to text map, the registered declaration
    records (name, expected type, module, path) and the module roots in
    emission order.
    """
    return _render_source_witness(song_root, source_data, REPO_ROOT)


# ---------------------------------------------------------------------------
# Structural whole-source UTF-8 witnesses
# ---------------------------------------------------------------------------

_UTF8_NAMESPACE = "Iota.Generated.SourceWitness.Tokenizer"
_UTF8_AUDIT_NAMESPACE = f"{_UTF8_NAMESPACE}.Audit"
_UTF8_AUDIT_PATH = "proof/corpus/Iota/Generated/SourceWitness/Tokenizer/Audit.lean"

# The generic structural lemmas the whole-source witnesses cite.  They live in
# the `Iota.Generated.Table` namespace because the parts they are applied to do,
# but in their own module so `Generated/Table.lean` keeps only the part
# presentation.  Each is a `List` fact proved without an axiom.
TABLE_SUPPORT_PATH = REPO_ROOT / "proof/corpus/Iota/Generated/TableSupport.lean"


def render_table_support() -> str:
    """Render the append/map/fuel lemmas the part witnesses use.

    These are presentation helpers: `sourceMapAppend` is `List.map_append` with
    the mapped list explicit, `sourceAppendAssoc`/`sourceAppendNil` are the
    `List` associativity and right-identity lemmas, and `sourceFuel` counts the
    code points one codepoint-aligned part adds.  None reduces a whole source
    list; each is a single structural step over `List`.
    """
    return "\n".join([
        "-- Generated by tools/iota/export.py; do not edit.",
        "import Iota.Source.Bytes",
        "",
        "set_option autoImplicit false",
        "set_option warningAsError true",
        "",
        "namespace Iota.Generated.Table",
        "",
        "/-- Fuel one codepoint-aligned part adds: its code-point count after `tail`. -/",
        "def sourceFuel (count tail : Nat) : Nat := tail + count",
        "",
        "theorem sourceFuel_eq (count tail : Nat) : sourceFuel count tail = tail + count :=",
        "  rfl",
        "",
        "/-- `List.map_append` with the mapped list and both operands explicit. -/",
        "theorem sourceMapAppend {α β : Type} (f : α → β) (left right : List α) :",
        "    (left ++ right).map f = left.map f ++ right.map f := by",
        "  induction left with",
        "  | nil => rfl",
        "  | cons head tail ih => exact congrArg (fun rest => f head :: rest) ih",
        "",
        "/-- `List.length_append`, hand-proved: the Init proof uses `simp` and",
        "carries `propext`, which the empty Iota cone rejects. -/",
        "theorem sourceLengthAppend {α : Type} (left right : List α) :",
        "    (left ++ right).length = left.length + right.length := by",
        "  induction left with",
        "  | nil => exact (Nat.zero_add right.length).symm",
        "  | cons head tail ih =>",
        "      exact Eq.trans (congrArg Nat.succ ih)",
        "        (Nat.succ_add tail.length right.length).symm",
        "",
        "/-- `List.append_assoc`, hand-proved to stay inside the empty cone. -/",
        "theorem sourceAppendAssoc {α : Type} (left middle right : List α) :",
        "    (left ++ middle) ++ right = left ++ (middle ++ right) := by",
        "  induction left with",
        "  | nil => rfl",
        "  | cons head tail ih => exact congrArg (fun rest => head :: rest) ih",
        "",
        "/-- `List.append_nil`, hand-proved to stay inside the empty cone. -/",
        "theorem sourceAppendNil {α : Type} (bytes : List α) : bytes ++ [] = bytes := by",
        "  induction bytes with",
        "  | nil => rfl",
        "  | cons head tail ih => exact congrArg (fun rest => head :: rest) ih",
        "",
        "end Iota.Generated.Table",
        "",
    ])


def _uint8(value: int) -> str:
    return f"(UInt8.ofFin ⟨{value}, by decide⟩)"


def _char(value: int) -> str:
    return f"Char.ofNat {value}"


def _utf8_char(char: str) -> str:
    encoded = char.encode("utf-8")
    values = [_uint8(value) for value in encoded]
    if len(values) == 1:
        return f"Char.ofNat {values[0]}.toNat"
    if len(values) == 2:
        return (f"Char.ofNat (({values[0]}.toNat - 0xc0) * 0x40 + "
                f"({values[1]}.toNat - 0x80))")
    if len(values) == 3:
        return (f"Char.ofNat (({values[0]}.toNat - 0xe0) * 0x1000 + "
                f"({values[1]}.toNat - 0x80) * 0x40 + ({values[2]}.toNat - 0x80))")
    return (f"Char.ofNat (({values[0]}.toNat - 0xf0) * 0x40000 + "
            f"({values[1]}.toNat - 0x80) * 0x1000 + "
            f"({values[2]}.toNat - 0x80) * 0x40 + ({values[3]}.toNat - 0x80))")


def _utf8_head(encoded: bytes, codepoint: int) -> str:
    values = " ".join(_uint8(value) for value in encoded)
    if len(encoded) == 1:
        return f"Utf8Head.one {values} _ (by decide)"
    if len(encoded) == 2:
        return f"Utf8Head.two {values} _ (by decide) (by decide) (by decide)"
    if len(encoded) == 3:
        return (f"Utf8Head.three {values} _ (by decide) (by decide) "
                f"(by decide) (by decide)")
    if len(encoded) == 4:
        return (f"Utf8Head.four {values} _ (by decide) (by decide) "
                f"(by decide) (by decide) (by decide)")
    raise ValueError(f"invalid UTF-8 width for U+{codepoint:04X}")


def _utf8_leaf_term(text: str) -> str:
    term = "tail"
    for char in reversed(text):
        term = f"Utf8Scan.next ({_utf8_head(char.encode('utf-8'), ord(char))}) ({term})"
    return term


def _utf8_module(namespace: str, imports: list[str], blocks: list[list[str]]) -> str:
    lines = [
        "-- Generated by tools/iota/export.py; do not edit.",
        *[f"import {name}" for name in imports],
        "",
        "open Iota.Source.Tokenizer",
        "",
        "set_option autoImplicit false",
        "set_option linter.missingDocs false",
        "",
        f"namespace {namespace}",
        "",
    ]
    for position, block in enumerate(blocks):
        if position:
            lines.append("")
        lines.extend(block)
    lines.extend(["", f"end {namespace}", ""])
    return "\n".join(lines)


def _render_utf8_program(stem: str, data: bytes) -> tuple[list[list[str]], str]:
    parts = source_parts(data)
    namespace_stem = stem[0].upper() + stem[1:]
    nodes: list[dict[str, object]] = []
    blocks: list[list[str]] = []
    for index, part in enumerate(parts):
        text = part.decode("utf-8")
        name = f"{stem}Leaf{index}"
        bytes_name = f"{name}Bytes"
        chars_name = f"{name}Chars"
        fuel_name = f"{name}Fuel"
        prefix_name = f"{name}Prefix"
        chars = ", ".join(_utf8_char(char) for char in text)
        blocks.append([
            f"def {bytes_name} (tail : List UInt8) : List UInt8 := "
            f"Iota.Generated.Table.{stem}Part{index}.map UInt8.ofFin ++ tail",
            f"def {chars_name} (tail : List Char) : List Char := [{chars}] ++ tail",
            f"def {fuel_name} (tail : Nat) : Nat := "
            f"Iota.Generated.Table.sourceFuel {len(text)} tail",
            f"theorem {prefix_name} (fuel : Nat) (tailBytes : List UInt8) "
            f"(tailChars : List Char) (tail : Utf8Scan fuel tailBytes tailChars) : "
            f"Utf8Scan ({fuel_name} fuel) ({bytes_name} tailBytes) "
            f"({chars_name} tailChars) := by",
            f"  exact {_utf8_leaf_term(text)}",
            f"theorem {name}FuelEq (tail : Nat) : {fuel_name} tail = "
            f"tail + {len(text)} := Iota.Generated.Table.sourceFuel_eq _ _",
            f"theorem {name}Binding (tail : List UInt8) : "
            f"Iota.Generated.Table.{stem}Part{index}.map UInt8.ofFin ++ tail = "
            f"{bytes_name} tail := rfl",
        ])
        nodes.append({
            "name": name, "bytes": bytes_name, "chars": chars_name,
            "fuel": fuel_name, "prefix": prefix_name,
            "binding": f"{name}Binding",
            "fuel_eq": f"{name}FuelEq",
            "count": len(text),
            "table": f"Iota.Generated.Table.{stem}Part{index}",
        })

    level = 0
    while len(nodes) > 1:
        next_nodes: list[dict[str, object]] = []
        for index in range(0, len(nodes), 2):
            if index + 1 == len(nodes):
                next_nodes.append(nodes[index])
                continue
            left = nodes[index]
            right = nodes[index + 1]
            name = f"{stem}Node{level}_{index // 2}"
            bytes_name = f"{name}Bytes"
            chars_name = f"{name}Chars"
            fuel_name = f"{name}Fuel"
            prefix_name = f"{name}Prefix"
            binding_name = f"{name}Binding"
            fuel_eq_name = f"{name}FuelEq"
            count = int(left["count"]) + int(right["count"])
            table_node = f"Iota.Generated.Table.{stem}PartNode{level}_{index // 2}"
            blocks.append([
                f"def {bytes_name} (tail : List UInt8) : List UInt8 := "
                f"{left['bytes']} ({right['bytes']} tail)",
                f"def {chars_name} (tail : List Char) : List Char := "
                f"{left['chars']} ({right['chars']} tail)",
                f"def {fuel_name} (tail : Nat) : Nat := {left['fuel']} ({right['fuel']} tail)",
                f"theorem {prefix_name} (fuel : Nat) (tailBytes : List UInt8) "
                f"(tailChars : List Char) (tail : Utf8Scan fuel tailBytes tailChars) : "
                f"Utf8Scan ({fuel_name} fuel) ({bytes_name} tailBytes) "
                f"({chars_name} tailChars) := by",
                "  exact Utf8Scan.continue",
                f"    (fuel0 := {fuel_name} fuel) (fuel1 := {right['fuel']} fuel) "
                f"(bytes0 := {bytes_name} tailBytes) (bytes1 := {right['bytes']} tailBytes) "
                f"(chars0 := {chars_name} tailChars) (chars1 := {right['chars']} tailChars)",
                f"    ({left['prefix']} ({right['fuel']} fuel) "
                f"({right['bytes']} tailBytes) ({right['chars']} tailChars))",
                f"    ({right['prefix']} fuel tailBytes tailChars tail)",
                f"theorem {fuel_eq_name} (tail : Nat) : {fuel_name} tail = "
                f"tail + {count} := by change {left['fuel']} ({right['fuel']} tail) = "
                f"tail + {count}; exact Eq.trans (congrArg {left['fuel']} "
                f"({right['fuel_eq']} tail)) (Eq.trans ({left['fuel_eq']} _) "
                f"(Eq.trans (Nat.add_assoc tail {right['count']} {left['count']}) "
                f"(congrArg (Nat.add tail) rfl)))",
                f"theorem {binding_name} (tail : List UInt8) : "
                f"{table_node}.map UInt8.ofFin ++ tail = {bytes_name} tail := by "
                f"exact Eq.trans (congrArg (fun value => value ++ tail) "
                f"(Iota.Generated.Table.sourceMapAppend UInt8.ofFin "
                f"{left['table']} {right['table']})) "
                f"(Eq.trans (Iota.Generated.Table.sourceAppendAssoc _ _ _) (Eq.trans "
                f"(congrArg (fun value => {left['table']}.map UInt8.ofFin ++ value) "
                f"({right['binding']} tail)) ({left['binding']} ({right['bytes']} tail))))",
            ])
            next_nodes.append({
                "name": name, "bytes": bytes_name, "chars": chars_name,
                "fuel": fuel_name, "prefix": prefix_name, "binding": binding_name,
                "fuel_eq": fuel_eq_name, "count": count, "table": table_node,
            })
        nodes = next_nodes
        level += 1

    root = nodes[0]
    source_name = f"Iota.Generated.Table.{stem}Source"
    byte_count = len(data)
    char_count = len(data.decode("utf-8"))
    base_fuel = byte_count - char_count + 1
    chars_name = f"{stem}Chars"
    witness_name = f"{stem}Utf8Witness"
    blocks.append([
        f"def {chars_name} : List Char := {root['chars']} []",
        f"theorem {stem}FuelValue : {root['fuel']} {base_fuel} = "
        f"{byte_count + 1} := Eq.trans ({root['fuel_eq']} {base_fuel}) rfl",
        f"theorem {stem}SourceLength : {source_name}.length = {byte_count} := by",
        f"  exact Iota.Generated.Table.{stem}SourceLength",
        f"theorem {stem}SourceMap : {source_name}.map UInt8.ofFin = {root['bytes']} [] := by",
        f"  change {root['table']}.map UInt8.ofFin = {root['bytes']} []",
        f"  exact Eq.trans (Iota.Generated.Table.sourceAppendNil _).symm "
        f"({root['binding']} [])",
        f"theorem {witness_name} :",
        f"    Utf8Scan ({source_name}.length + 1) ({source_name}.map UInt8.ofFin) "
        f"{chars_name} := by",
        f"  rw [{stem}SourceLength, {stem}SourceMap]",
        f"  have scan := {root['prefix']} {base_fuel} [] [] "
        f"(Utf8Scan.done (n := {base_fuel - 1}))",
        f"  have scan' : Utf8Scan {byte_count + 1} ({root['bytes']} []) "
        f"{chars_name} := {stem}FuelValue ▸ scan",
        "  exact scan'",
    ])
    return blocks, witness_name


def render_utf8_witness(song_root: Path) -> tuple[
        dict[Path, str], list[dict[str, str]], list[str]]:
    """Render balanced source-connected UTF-8 scans for cell, table and std."""
    outputs: dict[Path, str] = {}
    declarations: list[dict[str, str]] = []
    roots: list[str] = []
    audit_imports: list[str] = []
    sources = _source_bytes(song_root)
    for stem in ("cell", "table", "std"):
        blocks, witness = _render_utf8_program(stem, sources[stem][1])
        chunks = _pack_witness_blocks(blocks)
        namespace = f"{_UTF8_NAMESPACE}.{stem[0].upper() + stem[1:]}"
        previous = "Iota.Generated.Table"
        final_path = ""
        for index, chunk in enumerate(chunks, 1):
            module = f"{namespace}.Chunk{index:02d}"
            relative = ("proof/corpus/Iota/Generated/SourceWitness/Tokenizer/"
                        f"{stem[0].upper() + stem[1:]}/Chunk{index:02d}.lean")
            outputs[REPO_ROOT / relative] = _utf8_module(
                namespace,
                [previous, "Iota.Generated.TableSupport", "Iota.Source.Tokenizer.Utf8"],
                [blocks[item] for item in chunk],
            )
            roots.append(module)
            previous = module
            final_path = relative
        audit_imports.append(previous)
        full = f"{namespace}.{witness}"
        declarations.append({
            "name": full,
            "expected_type": f"type_of% @{full}",
            "module": previous,
            "path": final_path,
        })

    audit_lines = [
        "-- Generated by tools/iota/export.py; do not edit.",
        *[f"import {name}" for name in [*audit_imports, "Iota.Audit.Empty"]],
        "",
        "set_option autoImplicit false",
        "set_option linter.missingDocs false",
        "",
        f"namespace {_UTF8_AUDIT_NAMESPACE}",
        "",
        "theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl",
        "",
        f"#audit_empty {_UTF8_AUDIT_NAMESPACE}.nat_add_zero : "
        "type_of% @Iota.Generated.SourceWitness.Tokenizer.Audit.nat_add_zero",
    ]
    for entry in declarations:
        audit_lines.append(f"#audit_empty {entry['name']} : {entry['expected_type']}")
    audit_lines.extend(["", f"end {_UTF8_AUDIT_NAMESPACE}", ""])
    outputs[REPO_ROOT / _UTF8_AUDIT_PATH] = "\n".join(audit_lines)
    roots.append(_UTF8_AUDIT_NAMESPACE)
    anchor = f"{_UTF8_AUDIT_NAMESPACE}.nat_add_zero"
    declarations.insert(0, {
        "name": anchor,
        "expected_type": f"type_of% @{anchor}",
        "module": _UTF8_AUDIT_NAMESPACE,
        "path": _UTF8_AUDIT_PATH,
    })
    return outputs, declarations, roots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song-root", type=Path, default=DEFAULT_SONG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--derivation-output", type=Path, default=DEFAULT_DERIVATION_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--source-data",
        type=Path,
        default=None,
        help="Write the decoded table/cell/std data as JSON and exit; "
        "emits no Lean and writes no generated file.",
    )
    args = parser.parse_args()
    song_root = args.song_root.resolve()
    if args.source_data is not None:
        payload = {stem: source_data(stem, song_root) for stem in ("table", "cell", "std")}
        args.source_data.parent.mkdir(parents=True, exist_ok=True)
        args.source_data.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for stem in ("table", "cell", "std"):
            print(_source_data_summary(stem, payload[stem]))
        return 0
    generated, paths = render(song_root)
    derivation = render_derivation(paths)
    table_outputs = _render_table_outputs(
        generated, args.output, SOURCE_BYTE_HELPER, REPO_ROOT / "proof/corpus"
    )
    outputs: list[tuple[Path, str]] = [
        *table_outputs.items(),
        (args.derivation_output, derivation),
        (TABLE_SUPPORT_PATH, render_table_support()),
    ]
    witness_outputs, _witness_declarations, _witness_roots = render_source_witness(song_root)
    outputs.extend(sorted(witness_outputs.items()))
    utf8_outputs, _utf8_declarations, _utf8_roots = render_utf8_witness(song_root)
    outputs.extend(sorted(utf8_outputs.items()))
    output_map = {path.resolve(): text for path, text in outputs}
    if len(output_map) != len(outputs):
        raise SystemExit("generated output paths overlap")
    try:
        if args.check:
            check_output_map(output_map, args.output)
        else:
            reject_stale_shards(output_map, args.output)
            for path, text in output_map.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
