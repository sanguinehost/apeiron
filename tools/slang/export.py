#!/usr/bin/env python3
"""Export the exact Song table import closure as Lean UTF-8 byte strings.

This program is an untrusted renderer.  Lean independently tokenizes and
decodes its output in Slang.SourceAudit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
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


# Source parts are also the leaves of the structural UTF-8 witness.  The bound
# is measured in bytes, and a part boundary never splits one UTF-8 code point.
SOURCE_PART_BYTES = 32


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
            f"def {name} : Slang.Source.SourceBytes :=",
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
                f"def {name} : Slang.Source.SourceBytes := "
                f"{left['name']} ++ {right['name']}",
                f"theorem {name}Length : {name}.length = {length} := by "
                f"change ({left['name']} ++ {right['name']}).length = {length}; exact "
                f"Eq.trans (@Slang.Generated.Table.sourceLengthAppend _ {left['name']} "
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
        f"def {stem}Source : Slang.Source.SourceBytes :=",
        f"  {root}",
        f"theorem {stem}SourceLength : {stem}Source.length = {len(data)} := by",
        f"  exact {root}Length",
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
        "import Slang.Generated.TableSupport",
        "namespace Slang.Generated.Table",
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


# ---------------------------------------------------------------------------
# Pure-Python mirror of the Slang decoder (proof/corpus/Slang/Decode.lean)
#
# This is the untrusted producer's candidate decode.  It is data only: it
# emits no Lean, and it must agree with the checked decoder on the real
# source bytes.  `lex_chars_sim` is recovered from the pre-44247a8 export.py;
# the parser, `scanSelected`, and the validation relations are new mirrors of
# the corresponding `Decode.lean` definitions.  Every function below restates
# exactly one Lean definition of that file, in the same order.
# ---------------------------------------------------------------------------

# `Decode.TokenKind`'s constructors, in declaration order.
_TOKEN_KINDS = (
    "ident",
    "numeral",
    "lparen",
    "rparen",
    "lbrace",
    "rbrace",
    "lbracket",
    "rbracket",
    "comma",
    "semi",
    "newline",
    "assign",
    "lt",
    "le",
    "gt",
    "ge",
    "other",
)

# `Decode.selectedConstantNames` / `Decode.selectedFunctionNames`.
_SELECTED_CONSTANT_NAMES = [
    "TABLE_PRESENT", "TABLE_EMPTY", "TABLE_STRIDE", "MERGE_SET", "MERGE_MAP",
    "MERGE_ACC", "MERGE_GRAPH", "TABLE_PHI", "COORD_CEILING",
]
_SELECTED_FUNCTION_NAMES = [
    "tableSpread", "tableSlotKey", "tableSlotVal", "tableSlotSet", "tableSlotValSet",
    "tableInitGo", "tableInit", "tableCapacityOk", "tableFindSlotGo",
    "tableLookupSlotGo", "tableFind", "tableGet", "tableInsertWith",
]

_STEM_PATHS = {
    "src/table/table.slang": "table",
    "src/kernel/mem/cell.slang": "cell",
    "src/kernel/std.slang": "std",
}


class _DecodeError(Exception):
    """The Python mirror's `.error`; the message matches `Decode.lean`."""


# --- Decode.utf8ByteLength / isSpace / isDigit / isAlpha / isAlphaNum ---

def _utf8_length(char: str) -> int:
    code = ord(char)
    if code < 0x80:
        return 1
    if code < 0x800:
        return 2
    if code < 0x10000:
        return 3
    return 4


def _is_space(char: str) -> bool:
    return char in (" ", "\t", "\r")


def _is_digit(char: str) -> bool:
    return "0" <= char <= "9"


def _is_alpha(char: str) -> bool:
    return ("a" <= char <= "z") or ("A" <= char <= "Z") or char == "_"


def _is_alnum(char: str) -> bool:
    return _is_alpha(char) or _is_digit(char)


# --- Decode.lexChars (recovered) ---

def lex_chars_sim(data: bytes) -> tuple[list[tuple[str, object, int, int]], int, int]:
    """The real tokenizer's token stream, mirrored from `Decode.lexChars`.

    Returns the tokens, the consumed byte position and the number of recursive
    `lexChars` steps.  A token is `(kind, payload, span_start, span_stop)`:
    `payload` is the identifier `String` for `ident`, the value `Nat` for
    `numeral`, the `Char` for `other`, and `None` for the fixed kinds.
    """
    chars = list(data.decode("utf-8"))
    tokens: list[tuple[str, object, int, int]] = []
    index = 0
    position = 0
    steps = 0
    total = len(chars)
    while index < total:
        char = chars[index]
        width = _utf8_length(char)
        steps += 1
        if _is_space(char):
            index += 1
            position += width
            continue
        if char == "\n":
            tokens.append(("newline", None, position, position + width))
            index += 1
            position += width
            continue
        if char == "/":
            if index + 1 < total and chars[index + 1] == "/":
                scan = index + 2
                offset = position + 2
                while scan < total and chars[scan] != "\n":
                    offset += _utf8_length(chars[scan])
                    scan += 1
                index = scan
                position = offset
                continue
            tokens.append(("other", char, position, position + width))
            index += 1
            position += width
            continue
        if _is_alpha(char):
            scan = index + 1
            while scan < total and _is_alnum(chars[scan]):
                scan += 1
            scanned = chars[index:scan]
            stop = position + sum(_utf8_length(item) for item in scanned)
            tokens.append(("ident", "".join(scanned), position, stop))
            index = scan
            position = stop
            continue
        if _is_digit(char):
            scan = index + 1
            while scan < total and _is_digit(chars[scan]):
                scan += 1
            scanned = chars[index:scan]
            stop = position + sum(_utf8_length(item) for item in scanned)
            tokens.append(("numeral", int("".join(scanned)), position, stop))
            index = scan
            position = stop
            continue
        if char == "<" and index + 1 < total and chars[index + 1] == "=":
            tokens.append(("le", None, position, position + 2))
            index += 2
            position += 2
            continue
        if char == ">" and index + 1 < total and chars[index + 1] == "=":
            tokens.append(("ge", None, position, position + 2))
            index += 2
            position += 2
            continue
        singles = {
            "(": "lparen",
            ")": "rparen",
            "{": "lbrace",
            "}": "rbrace",
            "[": "lbracket",
            "]": "rbracket",
            ",": "comma",
            ";": "semi",
            "=": "assign",
            "<": "lt",
            ">": "gt",
        }
        tokens.append((singles.get(char, "other"), char, position, position + width))
        index += 1
        position += width
    return tokens, position, steps


def lex_source(data: bytes) -> list[tuple[str, object, int, int]]:
    return lex_chars_sim(data)[0]


# --- Decode.dropNewlines / expectKind / binOp ---

def _drop_newlines(tokens, index: int) -> int:
    while index < len(tokens) and tokens[index][0] == "newline":
        index += 1
    return index


def _expect_kind(tokens, index: int, kind: str):
    if index < len(tokens) and tokens[index][0] == kind:
        return tokens[index], index + 1
    if index >= len(tokens):
        raise _DecodeError("expected token at end of input")
    raise _DecodeError("unexpected token kind")


def _bin_op(kind: str, payload) -> "tuple[str, int] | None":
    if kind == "ident" and payload == "or":
        return ("bitOr", 1)
    if kind == "ident" and payload == "xor":
        return ("bitXor", 1)
    if kind == "ident" and payload == "and":
        return ("bitAnd", 2)
    if kind == "ident" and payload == "is":
        return ("eq", 3)
    if kind == "lt":
        return ("lt", 3)
    if kind == "le":
        return ("le", 3)
    if kind == "gt":
        return ("gt", 3)
    if kind == "ge":
        return ("ge", 3)
    if kind == "other" and payload == "+":
        return ("add", 4)
    if kind == "other" and payload == "-":
        return ("sub", 4)
    if kind == "other" and payload == "*":
        return ("mul", 5)
    return None


# --- Decode.Expr / Decode.Stmt constructors as JSON-shaped dicts ---

def _literal(value):
    return {"literal": value}


def _name(value):
    return {"name": value}


def _call(callee, args):
    return {"call": [callee, args]}


def _memory(address):
    return {"memory": address}


def _binary(op, left, right):
    return {"binary": [op, left, right]}


def _conditional(then_value, condition, else_value):
    return {"conditional": [then_value, condition, else_value]}


def _bind(name, value):
    return {"bind": [name, value]}


def _store(address, value):
    return {"store": [address, value]}


def _return_(value):
    return {"return_": value}


def _expression(value):
    return {"expression": value}


def _if_(condition, then_body, else_body):
    return {"if_": [condition, then_body, else_body]}


# --- Decode.parseExprFuel / parsePrimaryFuel / parseArgsFuel / parseBinaryFuel ---

def _parse_expr(tokens, index: int):
    """`Decode.parseExpr`: fuel `|tokens| * 4 + 16` over the suffix at `index`."""
    fuel = (len(tokens) - index) * 4 + 16
    return _parse_expr_fuel(tokens, index, fuel, 0)


def _parse_expr_fuel(tokens, index: int, fuel: int, min_prec: int):
    if fuel == 0:
        raise _DecodeError("expression nesting limit reached")
    inner = fuel - 1
    left, rest = _parse_primary_fuel(tokens, index, inner)
    return _parse_binary_fuel(tokens, rest, inner, min_prec, left)


def _parse_primary_fuel(tokens, index: int, fuel: int):
    if fuel == 0:
        raise _DecodeError("expression nesting limit reached")
    inner = fuel - 1
    index = _drop_newlines(tokens, index)
    if index >= len(tokens):
        raise _DecodeError("expected expression at end of input")
    token = tokens[index]
    rest = index + 1
    kind, payload = token[0], token[1]
    if kind == "numeral":
        return _literal(payload), rest
    if kind == "ident":
        name = payload
        if name == "memory":
            _, rest = _expect_kind(tokens, rest, "lbracket")
            address, rest = _parse_expr_fuel(tokens, rest, inner, 0)
            _, rest = _expect_kind(tokens, rest, "rbracket")
            return _memory(address), rest
        if rest < len(tokens) and tokens[rest][0] == "lparen":
            args, rest = _parse_args_fuel(tokens, rest + 1, inner)
            return _call(name, args), rest
        return _name(name), rest
    if kind == "lparen":
        value, rest = _parse_expr_fuel(tokens, rest, inner, 0)
        _, rest = _expect_kind(tokens, rest, "rparen")
        return value, rest
    raise _DecodeError("unsupported expression token")


def _parse_args_fuel(tokens, index: int, fuel: int):
    if fuel == 0:
        raise _DecodeError("call argument nesting limit reached")
    inner = fuel - 1
    index = _drop_newlines(tokens, index)
    if index >= len(tokens):
        raise _DecodeError("unterminated call arguments")
    if tokens[index][0] == "rparen":
        return [], index + 1
    arg, rest = _parse_expr_fuel(tokens, index, inner, 0)
    after = _drop_newlines(tokens, rest)
    if after >= len(tokens):
        raise _DecodeError("unterminated call arguments")
    if tokens[after][0] == "comma":
        args, tail = _parse_args_fuel(tokens, after + 1, inner)
        return [arg] + args, tail
    if tokens[after][0] == "rparen":
        return [arg], after + 1
    raise _DecodeError("expected comma or closing parenthesis")


def _parse_binary_fuel(tokens, index: int, fuel: int, min_prec: int, left):
    if fuel == 0:
        raise _DecodeError("expression operator limit reached")
    inner = fuel - 1
    start = index
    index = _drop_newlines(tokens, index)
    if index >= len(tokens):
        return left, index
    token = tokens[index]
    rest = index + 1
    kind, payload = token[0], token[1]

    def ordinary():
        op = _bin_op(kind, payload)
        if op is not None:
            name, prec = op
            if prec < min_prec:
                return left, start
            right, after = _parse_expr_fuel(tokens, rest, inner, prec + 1)
            return _parse_binary_fuel(tokens, after, inner, min_prec, _binary(name, left, right))
        if kind == "ident" and payload == "if":
            if 0 < min_prec:
                return left, start
            condition, after = _parse_expr_fuel(tokens, rest, inner, 0)
            after = _drop_newlines(tokens, after)
            if after >= len(tokens):
                raise _DecodeError("unterminated inline conditional")
            if tokens[after][0] == "ident" and tokens[after][1] == "else":
                else_value, tail = _parse_expr_fuel(tokens, after + 1, inner, 0)
                return _conditional(left, condition, else_value), tail
            raise _DecodeError("expected inline else")
        return left, start

    if kind == "ident" and payload == "is" and rest < len(tokens) \
            and tokens[rest][0] == "ident" and tokens[rest][1] == "not":
        if 3 < min_prec:
            return left, start
        right, after = _parse_expr_fuel(tokens, rest + 1, inner, 4)
        return _parse_binary_fuel(tokens, after, inner, min_prec, _binary("ne", left, right))
    return ordinary()


# --- Decode.parseBlockFuel / parseStmtFuel / parseParameters / parseFunction ---

def _parse_block_fuel(tokens, index: int, fuel: int):
    if fuel == 0:
        raise _DecodeError("statement nesting limit reached")
    inner = fuel - 1
    index = _drop_newlines(tokens, index)
    if index >= len(tokens):
        raise _DecodeError("unterminated function body")
    token = tokens[index]
    if token[0] == "rbrace":
        return [], token, index + 1
    stmt, rest = _parse_stmt_fuel(tokens, index, inner)
    body, close, rest = _parse_block_fuel(tokens, rest, inner)
    return [stmt] + body, close, rest


def _parse_stmt_fuel(tokens, index: int, fuel: int):
    if fuel == 0:
        raise _DecodeError("statement nesting limit reached")
    inner = fuel - 1
    index = _drop_newlines(tokens, index)
    if index >= len(tokens):
        value, rest = _parse_expr(tokens, index)
        _, rest = _expect_kind(tokens, _drop_newlines(tokens, rest), "semi")
        return _expression(value), rest
    token = tokens[index]
    rest = index + 1
    kind, payload = token[0], token[1]
    if kind == "ident":
        name = payload
        if name == "return":
            value, rest = _parse_expr(tokens, rest)
            _, rest = _expect_kind(tokens, _drop_newlines(tokens, rest), "semi")
            return _return_(value), rest
        if name == "if":
            condition, rest = _parse_expr(tokens, rest)
            _, rest = _expect_kind(tokens, _drop_newlines(tokens, rest), "lbrace")
            then_body, _, rest = _parse_block_fuel(tokens, rest, inner)
            after = _drop_newlines(tokens, rest)
            if after < len(tokens) and tokens[after][0] == "ident" and tokens[after][1] == "else":
                _, after = _expect_kind(tokens, _drop_newlines(tokens, after + 1), "lbrace")
                else_body, _, tail = _parse_block_fuel(tokens, after, inner)
                return _if_(condition, then_body, else_body), tail
            return _if_(condition, then_body, None), rest
        if name == "memory" and rest < len(tokens) and tokens[rest][0] == "lbracket":
            address, rest = _parse_expr(tokens, rest + 1)
            _, rest = _expect_kind(tokens, rest, "rbracket")
            _, rest = _expect_kind(tokens, _drop_newlines(tokens, rest), "assign")
            value, rest = _parse_expr(tokens, rest)
            _, rest = _expect_kind(tokens, _drop_newlines(tokens, rest), "semi")
            return _store(address, value), rest
        if rest < len(tokens) and tokens[rest][0] == "assign":
            value, rest = _parse_expr(tokens, rest + 1)
            _, rest = _expect_kind(tokens, _drop_newlines(tokens, rest), "semi")
            return _bind(name, value), rest
    value, rest = _parse_expr(tokens, index)
    _, rest = _expect_kind(tokens, _drop_newlines(tokens, rest), "semi")
    return _expression(value), rest


def _parse_parameters(tokens, index: int, fuel: int):
    if fuel == 0:
        raise _DecodeError("parameter limit reached")
    inner = fuel - 1
    if index >= len(tokens):
        raise _DecodeError("expected name at end of input")
    token = tokens[index]
    if token[0] == "rparen":
        return [], index + 1
    if token[0] == "ident":
        name = token[1]
        if index + 1 >= len(tokens):
            raise _DecodeError("unterminated parameter list")
        nxt = tokens[index + 1]
        if nxt[0] == "comma":
            names, rest = _parse_parameters(tokens, index + 2, inner)
            return [name] + names, rest
        if nxt[0] == "rparen":
            return [name], index + 2
        raise _DecodeError("expected ',' or ')' in parameters")
    raise _DecodeError("expected name")


def _parse_function(tokens, index: int, name: str, start: int):
    """`Decode.parseFunction`, with the synthetic open-paren already consumed.

    `index` is the position after the real `(` in `scanSelected`; `Decode.lean`
    prepends a synthetic `lparen` whose span is never read.
    """
    parameters, index = _parse_parameters(tokens, index, (len(tokens) - index) + 1)
    index = _drop_newlines(tokens, index)
    _, index = _expect_kind(tokens, index, "lbrace")
    body, close, index = _parse_block_fuel(tokens, index, (len(tokens) - index) + 16)
    return {
        "name": name,
        "parameters": parameters,
        "body": body,
        "span": [start, close[3]],
    }, index


# --- Decode.takeLine / scanSelected ---

def _take_line(tokens, index: int):
    line = []
    while index < len(tokens) and tokens[index][0] != "newline":
        line.append(tokens[index])
        index += 1
    return line, index


def _scan_selected(tokens, constants, functions):
    return _scan_selected_fuel(
        tokens, 0, len(tokens) + 1, list(constants), list(functions)
    )


def _scan_selected_fuel(tokens, index: int, fuel: int, constants, functions):
    if fuel == 0:
        raise _DecodeError("declaration scan limit reached")
    if index >= len(tokens):
        return [], []
    inner = fuel - 1
    token = tokens[index]
    rest = index + 1
    if token[0] == "ident":
        name = token[1]
        if rest >= len(tokens):
            return _scan_selected_fuel(tokens, rest, inner, constants, functions)
        nxt = tokens[rest]
        tail = rest + 1
        if nxt[0] == "assign":
            if name in constants:
                line, after_line = _take_line(tokens, tail)
                value, leftover = _parse_expr(line, 0)
                if _drop_newlines(line, leftover) != len(line):
                    raise _DecodeError("trailing constant tokens")
                stop = line[-1][3] if line else token[3]
                remaining = list(constants)
                remaining.remove(name)
                decls, funcs = _scan_selected_fuel(tokens, after_line, inner, remaining, functions)
                return [{"name": name, "value": value, "span": [token[2], stop]}] + decls, funcs
            if name in _SELECTED_CONSTANT_NAMES:
                raise _DecodeError("duplicate selected constant")
            return _scan_selected_fuel(tokens, rest, inner, constants, functions)
        if nxt[0] == "lparen":
            if name in functions:
                decl, after_decl = _parse_function(tokens, tail, name, token[2])
                remaining = list(functions)
                remaining.remove(name)
                decls, funcs = _scan_selected_fuel(tokens, after_decl, inner, constants, remaining)
                return decls, [decl] + funcs
            return _scan_selected_fuel(tokens, rest, inner, constants, functions)
        return _scan_selected_fuel(tokens, rest, inner, constants, functions)
    return _scan_selected_fuel(tokens, rest, inner, constants, functions)


# --- Decode.distinct / delimitersBalanced / spansValid ---

def _distinct(names) -> bool:
    return len(list(dict.fromkeys(names))) == len(names)


def _delimiters_balanced(tokens) -> bool:
    stack: list[str] = []
    for kind, _, _, _ in tokens:
        if kind == "lparen":
            stack.append("lparen")
        elif kind == "lbrace":
            stack.append("lbrace")
        elif kind == "lbracket":
            stack.append("lbracket")
        elif kind == "rparen":
            if not stack or stack.pop() != "lparen":
                return False
        elif kind == "rbrace":
            if not stack or stack.pop() != "lbrace":
                return False
        elif kind == "rbracket":
            if not stack or stack.pop() != "lbracket":
                return False
    return not stack


def _spans_valid(size: int, spans) -> bool:
    if not all(start < stop <= size for start, stop in spans):
        return False
    for first in spans:
        for second in spans:
            if first == second:
                continue
            if not (first[1] <= second[0] or second[1] <= first[0]):
                return False
    return True


# --- Decode.expressionCallsResolve / statementsCallsResolve / resolveCalls ---

def _expression_calls_resolve(available, expression) -> bool:
    if "literal" in expression or "name" in expression:
        return True
    if "call" in expression:
        callee, args = expression["call"]
        return callee in available and all(
            _expression_calls_resolve(available, arg) for arg in args
        )
    if "memory" in expression:
        return _expression_calls_resolve(available, expression["memory"])
    if "binary" in expression:
        _, left, right = expression["binary"]
        return _expression_calls_resolve(available, left) and \
            _expression_calls_resolve(available, right)
    then_value, condition, else_value = expression["conditional"]
    return _expression_calls_resolve(available, then_value) and \
        _expression_calls_resolve(available, condition) and \
        _expression_calls_resolve(available, else_value)


def _statements_calls_resolve(available, body) -> bool:
    for stmt in body:
        if "bind" in stmt:
            if not _expression_calls_resolve(available, stmt["bind"][1]):
                return False
        elif "store" in stmt:
            address, value = stmt["store"]
            if not (_expression_calls_resolve(available, address) and
                    _expression_calls_resolve(available, value)):
                return False
        elif "return_" in stmt:
            if not _expression_calls_resolve(available, stmt["return_"]):
                return False
        elif "expression" in stmt:
            if not _expression_calls_resolve(available, stmt["expression"]):
                return False
        else:
            condition, then_body, else_body = stmt["if_"]
            if not _expression_calls_resolve(available, condition):
                return False
            if not _statements_calls_resolve(available, then_body):
                return False
            if else_body is not None and not _statements_calls_resolve(available, else_body):
                return False
    return True


def _resolve_calls(functions) -> bool:
    names = [decl["name"] for decl in functions]
    return all(_statements_calls_resolve(names, decl["body"]) for decl in functions)


# --- Decode.expressionNamesResolve / statementsNamesResolve / resolveNames ---

def _expression_names_resolve(available, expression) -> bool:
    if "literal" in expression:
        return True
    if "name" in expression:
        return expression["name"] in available
    if "call" in expression:
        return all(_expression_names_resolve(available, arg) for arg in expression["call"][1])
    if "memory" in expression:
        return _expression_names_resolve(available, expression["memory"])
    if "binary" in expression:
        _, left, right = expression["binary"]
        return _expression_names_resolve(available, left) and \
            _expression_names_resolve(available, right)
    then_value, condition, else_value = expression["conditional"]
    return _expression_names_resolve(available, then_value) and \
        _expression_names_resolve(available, condition) and \
        _expression_names_resolve(available, else_value)


def _statements_names_resolve(fuel: int, available, body) -> bool:
    if fuel == 0:
        return False
    if not body:
        return True
    stmt = body[0]
    rest = body[1:]
    if "bind" in stmt:
        name, value = stmt["bind"]
        return _expression_names_resolve(available, value) and \
            _statements_names_resolve(fuel - 1, [name] + available, rest)
    if "store" in stmt:
        address, value = stmt["store"]
        return _expression_names_resolve(available, address) and \
            _expression_names_resolve(available, value) and \
            _statements_names_resolve(fuel - 1, available, rest)
    if "return_" in stmt:
        return _expression_names_resolve(available, stmt["return_"]) and \
            _statements_names_resolve(fuel - 1, available, rest)
    if "expression" in stmt:
        return _expression_names_resolve(available, stmt["expression"]) and \
            _statements_names_resolve(fuel - 1, available, rest)
    condition, then_body, else_body = stmt["if_"]
    return _expression_names_resolve(available, condition) and \
        _statements_names_resolve(fuel - 1, available, then_body) and \
        (else_body is None or _statements_names_resolve(fuel - 1, available, else_body)) and \
        _statements_names_resolve(fuel - 1, available, rest)


def _resolve_names(constants, functions) -> bool:
    constant_names = [decl["name"] for decl in constants]
    for decl in functions:
        if not _distinct(decl["parameters"]):
            return False
        if not _statements_names_resolve(
            len(decl["body"]) * 4 + 64,
            list(decl["parameters"]) + constant_names,
            decl["body"],
        ):
            return False
    return True


# --- source_data: the pure-data decode for one generated source ---

def _token_to_json(token) -> dict:
    kind, payload, start, stop = token
    if kind == "other":
        value: object = ord(payload)
    elif kind in ("ident", "numeral"):
        value = payload
    else:
        value = None
    return {"kind": kind, "value": value, "start": start, "stop": stop}


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


# ---------------------------------------------------------------------------
# Structural `ScanSelected` witness emission
#
# `Slang.Source.Parsing.ScanSelected` is the structural counterpart of the
# executable `Decode.scanSelected`; `ScanSelected.sound` is the checked link.
# For each real source program this section renders the token list and the
# structural derivations (`ExprScan`/`PrimaryScan`/`BinaryScan`/`ArgsScan`,
# `StmtScan`/`BlockScan`, `ParameterScan`, `FunctionScan`, `ExprScanTop`,
# `ConstantLineParse`, `SelectedFunctionParse`) that witness its scan, as a
# chain of committed modules.  The `ScanSelected` compositions are chunked at
# delimiter-stack zero with `ScanSelected.continue`, never splitting an
# identifier or a comment.  The emitted terms are data; the Lean kernel checks
# them under the empty axiom cone, and `Slang.Audit.Empty` rejects a nonempty
# set.  Port of the `/tmp/apeiron/gen_structural_scan.py` spike.
# ---------------------------------------------------------------------------

_SCAN_CHUNK = 24
_SCAN_BIN_OPS: dict[str, tuple[str, int]] = {
    "or": ("bitOr", 1), "xor": ("bitXor", 1), "and": ("bitAnd", 2),
    "is": ("eq", 3), "lt": ("lt", 3), "le": ("le", 3),
    "gt": ("gt", 3), "ge": ("ge", 3),
    "+": ("add", 4), "-": ("sub", 4), "*": ("mul", 5),
}
_SCAN_COUNTERS = (
    "ExprScan", "PrimaryScan", "BinaryScan", "ArgsScan", "StmtScan",
    "BlockScan", "ParameterScan", "ExprScanTop", "ConstantLineParse",
    "FunctionScan", "SelectedFunctionParse", "ScanSelected",
)

# A module body is packed to this many physical lines, counting one blank
# separator per block, so a rendered file stays below the 1500-line cap once
# its header and footer are added.
_WITNESS_BODY_LINES = 1480
_WITNESS_AUDIT_NAMESPACE = "Slang.Generated.SourceWitness.Parsing.Audit"
_WITNESS_AUDIT_PATH = "proof/corpus/Slang/Generated/SourceWitness/Parsing/Audit.lean"

_WITNESS_PROGRAMS: tuple[dict[str, object], ...] = (
    {
        "stem": "table",
        "directory": "Table",
        "namespace": "Slang.Generated.SourceWitness.Parsing.Table",
        "constants": list(_SELECTED_CONSTANT_NAMES),
        "functions": list(_SELECTED_FUNCTION_NAMES),
    },
    {
        "stem": "cell",
        "directory": "Cell",
        "namespace": "Slang.Generated.SourceWitness.Parsing.Cell",
        "constants": ["cellTestRegion"],
        "functions": ["cellGet", "cellSet", "cellSetGet", "cellAddGet"],
    },
)


def _scan_str(text: str) -> str:
    return json.dumps(text)


def _scan_span(token: dict) -> str:
    return f'⟨{token["start"]}, {token["stop"]}⟩'


def _scan_token(token: dict) -> str:
    kind = token["kind"]
    value = token["value"]
    if kind == "ident":
        rendered = f".ident {_scan_str(value)}"
    elif kind == "numeral":
        rendered = f".numeral {value}"
    elif kind == "other":
        rendered = f".other (Char.ofNat {value})"
    else:
        rendered = f".{kind}"
    return f'{{ kind := {rendered}, span := {_scan_span(token)} }}'


def _scan_expr(expression: dict) -> str:
    if "literal" in expression:
        return f'.literal {expression["literal"]}'
    if "name" in expression:
        return f'.name {_scan_str(expression["name"])}'
    if "call" in expression:
        name, args = expression["call"]
        return f'.call {_scan_str(name)} [{", ".join(_scan_expr(arg) for arg in args)}]'
    if "memory" in expression:
        return f'.memory ({_scan_expr(expression["memory"])})'
    if "binary" in expression:
        op, left, right = expression["binary"]
        return f'.binary .{op} ({_scan_expr(left)}) ({_scan_expr(right)})'
    then_value, condition, else_value = expression["conditional"]
    return (f'.conditional ({_scan_expr(then_value)}) ({_scan_expr(condition)}) '
            f'({_scan_expr(else_value)})')


def _scan_stmt(statement: dict) -> str:
    if "return" in statement:
        return f'.return_ ({_scan_expr(statement["return"])})'
    if "bind" in statement:
        name, value = statement["bind"]
        return f'.bind {_scan_str(name)} ({_scan_expr(value)})'
    if "store" in statement:
        address, value = statement["store"]
        return f'.store ({_scan_expr(address)}) ({_scan_expr(value)})'
    if "expression" in statement:
        return f'.expression ({_scan_expr(statement["expression"])})'
    condition, then_body, else_body = statement["if"]
    rendered_else = "none" if else_body is None else (
        f'some [{", ".join(_scan_stmt(item) for item in else_body)}]'
    )
    return (f'.if_ ({_scan_expr(condition)}) '
            f'[{", ".join(_scan_stmt(item) for item in then_body)}] ({rendered_else})')


def _scan_decl(declaration: dict) -> str:
    parameters = ", ".join(_scan_str(name) for name in declaration["parameters"])
    body = ", ".join(_scan_stmt(item) for item in declaration["body"])
    return (f'{{ name := {_scan_str(declaration["name"])}, parameters := [{parameters}], '
            f'body := [{body}], span := ⟨{declaration["span"][0]}, {declaration["span"][1]}⟩ }}')


def _scan_cdecl(declaration: dict) -> str:
    return (f'{{ name := {_scan_str(declaration["name"])}, '
            f'value := {_scan_expr(declaration["value"])}, '
            f'span := ⟨{declaration["span"][0]}, {declaration["span"][1]}⟩ }}')


class _StructuralScan:
    """One source program's token data and structural scan derivations."""

    def __init__(self, stem: str, tokens: list[dict], const_names: list[str],
                 func_names: list[str]) -> None:
        self.stem = stem
        self.ts = tokens
        self.n = len(tokens)
        self.cn0 = list(const_names)
        self.fn0 = list(func_names)
        self.blocks: list[list[str]] = []
        self.counts = {name: 0 for name in _SCAN_COUNTERS}
        self.serial = 0
        self.audit: list[dict[str, object]] = []

    def name(self, counter: str) -> str:
        self.serial += 1
        self.counts[counter] += 1
        return f"{self.stem}_{counter.lower()}_{self.serial}"

    def add(self, counter: str, type_: str, body: str, audit: bool = False) -> str:
        name = self.name(counter)
        index = len(self.blocks)
        self.blocks.append([f"theorem {name} : {type_} :=", f"  {body}"])
        if audit:
            self.audit.append({"name": name, "block": index})
        return name

    def suffix(self, index: int) -> str:
        assert 0 <= index <= self.n
        return f"{self.stem}T{index}"

    def emit_data(self) -> None:
        starts = list(range(0, self.n, _SCAN_CHUNK))
        self.blocks.append([f"def {self.stem}T{self.n} : List Token := []"])
        for begin in reversed(starts):
            end = min(begin + _SCAN_CHUNK, self.n)
            tail = f"{self.stem}T{end}"
            for index in range(end - 1, begin - 1, -1):
                body = ("[" + ", ".join(_scan_token(token) for token in self.ts[index:end])
                        + f"] ++ {tail}")
                self.blocks.append([f"def {self.stem}T{index} : List Token := {body}"])
        self.blocks.append([
            f"theorem {self.stem}ConsLength {{α : Type}} {{x : α}} {{xs : List α}} "
            f"{{n : Nat}} (h : xs.length = n) : (x :: xs).length = n + 1 := "
            f"congrArg Nat.succ h",
        ])
        self.blocks.append([
            f"theorem {self.stem}T{self.n}_length : {self.stem}T{self.n}.length = 0 := rfl",
        ])
        for begin in reversed(starts):
            end = min(begin + _SCAN_CHUNK, self.n)
            for index in range(end - 1, begin - 1, -1):
                items = self.ts[index:end]
                lines = [
                    f"theorem {self.stem}T{index}_length : "
                    f"{self.stem}T{index}.length = {self.n - index} := by",
                ]
                cons = (" :: ".join(_scan_token(token) for token in items)
                        + f" :: {self.stem}T{end}")
                lines.append(f"  change ({cons}).length = {self.n - index}")
                proof = f"{self.stem}T{end}_length"
                current = self.n - end
                for token in reversed(items):
                    proof = (f"@{self.stem}ConsLength Token {_scan_token(token)} "
                             f"_ {current} ({proof})")
                    current += 1
                lines.append(f"  exact {proof}")
                self.blocks.append(lines)
        self.blocks.append([f"def {self.stem}Tokens : List Token := {self.stem}T0"])

    def drop_i(self, index: int) -> tuple[str, int]:
        if index == self.n:
            return "DropNewlinesScan.done", index
        if self.ts[index]["kind"] == "newline":
            tail, result = self.drop_i(index + 1)
            return f"DropNewlinesScan.newline (by decide) ({tail})", result
        return "DropNewlinesScan.stop (by decide)", index

    def drop_eq(self, index: int) -> tuple[str, int]:
        pattern, result = self.drop_i(index)
        return f"({pattern}).sound", result

    def binop(self, token: dict) -> tuple[str, int] | None:
        kind = token["kind"]
        value = token["value"]
        key = value if kind == "ident" else (chr(value) if kind == "other" else kind)
        return _SCAN_BIN_OPS.get(key)

    def ordinary(self, index: int) -> str:
        token = self.ts[index]
        if token["kind"] == "ident" and token["value"] == "is":
            if index + 1 == self.n:
                return f"OrdinaryBinaryGuard.isEnd {_scan_span(token)}"
            return "OrdinaryBinaryGuard.isNext (by decide)"
        return "OrdinaryBinaryGuard.notIs (by decide)"

    def parse_expr(self, fuel: int, min_prec: int, index: int):
        primary, left, after_primary = self.parse_primary(fuel - 2, index)
        binary, out, end = self.parse_binary(fuel - 1, min_prec, left, after_primary)
        type_ = (f"ExprScan {fuel} {min_prec} {self.suffix(index)} "
                 f"({_scan_expr(out)}) {self.suffix(end)}")
        node = self.add("ExprScan", type_, f"ExprScan.step {primary} {binary}")
        return node, out, end

    def parse_primary(self, fuel: int, index: int):
        drop, start = self.drop_eq(index)
        token = self.ts[start]
        kind = token["kind"]
        if kind == "numeral":
            out = {"literal": token["value"]}
            end = start + 1
            body = f"PrimaryScan.numeral {drop}"
        elif kind == "ident":
            name = token["value"]
            if name == "memory":
                assert self.ts[start + 1]["kind"] == "lbracket"
                node, address, after = self.parse_expr(fuel, 0, start + 2)
                assert self.ts[after]["kind"] == "rbracket"
                end = after + 1
                out = {"memory": address}
                body = f"PrimaryScan.memory {drop} {node} (by rfl)"
            elif start + 1 < self.n and self.ts[start + 1]["kind"] == "lparen":
                node, args, end = self.parse_args(fuel, start + 2)
                out = {"call": [name, args]}
                body = f"PrimaryScan.call {drop} (by decide) {node}"
            elif start + 1 == self.n:
                out = {"name": name}
                end = start + 1
                body = f"PrimaryScan.nameEnd {drop} (by decide)"
            else:
                out = {"name": name}
                end = start + 1
                body = f"PrimaryScan.nameNext {drop} (by decide) (by decide)"
        elif kind == "lparen":
            node, out, after = self.parse_expr(fuel, 0, start + 1)
            assert self.ts[after]["kind"] == "rparen"
            end = after + 1
            body = f"PrimaryScan.parenthesized {drop} {node} (by rfl)"
        else:
            raise ValueError(("primary", self.stem, index, token))
        type_ = (f"PrimaryScan {fuel + 1} {self.suffix(index)} "
                 f"({_scan_expr(out)}) {self.suffix(end)}")
        node = self.add("PrimaryScan", type_, body)
        return node, out, end

    def parse_args(self, fuel: int, index: int):
        drop, start = self.drop_eq(index)
        if self.ts[start]["kind"] == "rparen":
            args: list[dict] = []
            end = start + 1
            body = f"ArgsScan.done {drop}"
        else:
            node, arg, after = self.parse_expr(fuel - 1, 0, start)
            after_drop, mid = self.drop_eq(after)
            if self.ts[mid]["kind"] == "comma":
                tail, args2, end = self.parse_args(fuel - 1, mid + 1)
                args = [arg] + args2
                body = (f"ArgsScan.comma {drop} (by decide) {node} {after_drop} "
                        f"(by decide) {tail}")
            else:
                assert self.ts[mid]["kind"] == "rparen"
                args = [arg]
                end = mid + 1
                body = (f"ArgsScan.last {drop} (by decide) {node} {after_drop} "
                        f"(by decide) (by decide)")
        type_ = (f"ArgsScan {fuel} {self.suffix(index)} "
                 f"[{', '.join(_scan_expr(item) for item in args)}] {self.suffix(end)}")
        node = self.add("ArgsScan", type_, body)
        return node, args, end

    def parse_binary(self, fuel: int, min_prec: int, left: dict, index: int):
        drop, start = self.drop_eq(index)
        if start == self.n:
            out = left
            end = start
            body = f"BinaryScan.done {drop}"
        elif (self.ts[start]["kind"] == "ident" and self.ts[start]["value"] == "is"
              and start + 1 < self.n and self.ts[start + 1]["kind"] == "ident"
              and self.ts[start + 1]["value"] == "not"):
            if 3 < min_prec:
                out = left
                end = index
                body = f"BinaryScan.isNotStop {drop} (by decide)"
            else:
                node, right, after = self.parse_expr(fuel - 1, 4, start + 2)
                combined = {"binary": ["ne", left, right]}
                tail, out, end = self.parse_binary(fuel - 1, min_prec, combined, after)
                body = f"BinaryScan.isNot {drop} (by decide) {node} {tail}"
        else:
            operator = self.binop(self.ts[start])
            if operator:
                op, prec = operator
                if prec < min_prec:
                    out = left
                    end = index
                    body = (f"BinaryScan.belowPrec {drop} ({self.ordinary(start)}) "
                            f"(by rfl) (by decide)")
                else:
                    node, right, after = self.parse_expr(fuel - 1, prec + 1, start + 1)
                    combined = {"binary": [op, left, right]}
                    tail, out, end = self.parse_binary(fuel - 1, min_prec, combined, after)
                    body = (f"BinaryScan.operatorStep {drop} ({self.ordinary(start)}) "
                            f"(by rfl) (by decide) {node} {tail}")
            elif self.ts[start]["kind"] == "ident" and self.ts[start]["value"] == "if":
                if 0 < min_prec:
                    out = left
                    end = index
                    body = (f"BinaryScan.conditionalStop {drop} "
                            f"(by decide) (by decide) (by decide)")
                else:
                    node, condition, after = self.parse_expr(fuel - 1, 0, start + 1)
                    after_drop, mid = self.drop_eq(after)
                    assert self.ts[mid]["kind"] == "ident" and self.ts[mid]["value"] == "else"
                    tail, else_value, end = self.parse_expr(fuel - 1, 0, mid + 1)
                    out = {"conditional": [left, condition, else_value]}
                    body = (f"BinaryScan.conditional {drop} (by decide) (by decide) "
                            f"(by decide) {node} {after_drop} {tail}")
            else:
                out = left
                end = index
                body = (f"BinaryScan.stop {drop} ({self.ordinary(start)}) "
                        f"(by decide) (by decide)")
        type_ = (f"BinaryScan {fuel} {min_prec} ({_scan_expr(left)}) {self.suffix(index)} "
                 f"({_scan_expr(out)}) {self.suffix(end)}")
        node = self.add("BinaryScan", type_, body)
        return node, out, end

    def expr_top(self, index: int):
        fuel = (self.n - index) * 4 + 16
        node, out, end = self.parse_expr(fuel, 0, index)
        type_ = f"ExprScanTop {self.suffix(index)} ({_scan_expr(out)}) {self.suffix(end)}"
        cast = (f"(Eq.symm (congrArg (fun n => n * 4 + 16) {self.stem}T{index}_length)) "
                f"▸ {node}")
        top = self.add("ExprScanTop", type_, f"ExprScanTop.intro ({cast})", audit=True)
        return top, out, end

    def params(self, fuel: int, index: int):
        token = self.ts[index]
        if token["kind"] == "rparen":
            names: list[str] = []
            end = index + 1
            body = "ParameterScan.done"
        else:
            assert token["kind"] == "ident"
            name = token["value"]
            nxt = self.ts[index + 1]
            if nxt["kind"] == "comma":
                node, names2, end = self.params(fuel - 1, index + 2)
                names = [name] + names2
                body = f"ParameterScan.comma {node}"
            else:
                assert nxt["kind"] == "rparen"
                names = [name]
                end = index + 2
                body = "ParameterScan.last"
        type_ = (f"ParameterScan {fuel} {self.suffix(index)} "
                 f"[{', '.join(_scan_str(item) for item in names)}] {self.suffix(end)}")
        node = self.add("ParameterScan", type_, body)
        return node, names, end

    def parse_stmt(self, fuel: int, index: int):
        drop, start = self.drop_eq(index)
        token = self.ts[start]
        if token["kind"] == "ident" and token["value"] == "return":
            node, value, after = self.expr_top(start + 1)
            after_drop, semi = self.drop_eq(after)
            assert self.ts[semi]["kind"] == "semi"
            end = semi + 1
            out = {"return": value}
            body = f"StmtScan.return_ {drop} {node} {after_drop}"
        elif token["kind"] == "ident" and token["value"] == "if":
            node, condition, after = self.expr_top(start + 1)
            after_drop, brace = self.drop_eq(after)
            assert self.ts[brace]["kind"] == "lbrace"
            block, then_body, _, after_block = self.parse_block(fuel - 1, brace + 1)
            block_drop, mid = self.drop_eq(after_block)
            if (mid < self.n and self.ts[mid]["kind"] == "ident"
                    and self.ts[mid]["value"] == "else"):
                else_drop, brace2 = self.drop_eq(mid + 1)
                assert self.ts[brace2]["kind"] == "lbrace"
                else_node, else_body, _, end = self.parse_block(fuel - 1, brace2 + 1)
                out = {"if": [condition, then_body, else_body]}
                body = (f"StmtScan.ifElse {drop} {node} {after_drop} {block} "
                        f"{block_drop} {else_drop} {else_node}")
            elif mid == self.n:
                end = after_block
                out = {"if": [condition, then_body, None]}
                body = (f"StmtScan.ifNoElseEnd {drop} {node} {after_drop} "
                        f"{block} {block_drop}")
            else:
                end = after_block
                out = {"if": [condition, then_body, None]}
                body = (f"StmtScan.ifNoElseNext {drop} {node} {after_drop} "
                        f"{block} {block_drop} (by decide)")
        elif (token["kind"] == "ident" and token["value"] == "memory"
              and start + 1 < self.n and self.ts[start + 1]["kind"] == "lbracket"):
            node, address, after = self.expr_top(start + 2)
            assert self.ts[after]["kind"] == "rbracket"
            assign_drop, assign = self.drop_eq(after + 1)
            assert self.ts[assign]["kind"] == "assign"
            value_node, value, after_value = self.expr_top(assign + 1)
            semi_drop, semi = self.drop_eq(after_value)
            assert self.ts[semi]["kind"] == "semi"
            end = semi + 1
            out = {"store": [address, value]}
            body = (f"StmtScan.store {drop} {node} (by rfl) {assign_drop} "
                    f"{value_node} {semi_drop}")
        elif (token["kind"] == "ident" and start + 1 < self.n
              and self.ts[start + 1]["kind"] == "assign"):
            node, value, after = self.expr_top(start + 2)
            semi_drop, semi = self.drop_eq(after)
            assert self.ts[semi]["kind"] == "semi"
            end = semi + 1
            out = {"bind": [token["value"], value]}
            body = (f"StmtScan.bind (by decide) (by decide) (by decide) {drop} "
                    f"{node} {semi_drop}")
        else:
            node, value, after = self.expr_top(start)
            semi_drop, semi = self.drop_eq(after)
            assert self.ts[semi]["kind"] == "semi"
            end = semi + 1
            out = {"expression": value}
            if token["kind"] == "ident":
                body = (f"StmtScan.nameExpression (by decide) (by decide) (by decide) "
                        f"{drop} (by decide) {node} {semi_drop}")
            else:
                body = (f"StmtScan.otherExpression ({self.expr_stmt_head(token)}) "
                        f"{drop} {node} {semi_drop}")
        type_ = (f"StmtScan {fuel} {self.suffix(index)} "
                 f"({_scan_stmt(out)}) {self.suffix(end)}")
        node = self.add("StmtScan", type_, body)
        return node, out, end

    def parse_block(self, fuel: int, index: int):
        drop, start = self.drop_eq(index)
        token = self.ts[start]
        if token["kind"] == "rbrace":
            body = f"BlockScan.done {drop} (by decide)"
            out: list[dict] = []
            close = token
            end = start + 1
        else:
            node, statement, after = self.parse_stmt(fuel - 1, start)
            tail, rest, close, end = self.parse_block(fuel - 1, after)
            out = [statement] + rest
            body = f"BlockScan.next {drop} (by decide) {node} {tail}"
        type_ = (f"BlockScan {fuel} {self.suffix(index)} "
                 f"[{', '.join(_scan_stmt(item) for item in out)}] "
                 f"({_scan_token(close)}) {self.suffix(end)}")
        node = self.add("BlockScan", type_, body)
        return node, out, close, end

    def function(self, name: str, start: int, open_index: int):
        assert self.ts[open_index]["kind"] == "lparen"
        parameter_fuel = self.n - (open_index + 1) + 1
        parameter_node, params, after_params = self.params(parameter_fuel, open_index + 1)
        brace_drop, brace = self.drop_eq(after_params)
        assert self.ts[brace]["kind"] == "lbrace"
        body_fuel = self.n - (brace + 1) + 16
        body_node, body, close, end = self.parse_block(body_fuel, brace + 1)
        declaration = {"name": name, "parameters": params, "body": body,
                       "span": [start, close["stop"]]}
        type_ = (f"FunctionScan {_scan_str(name)} {start} {self.suffix(open_index)} "
                 f"({_scan_decl(declaration)}) {self.suffix(end)}")
        fn_body = (
            f"by\n    apply FunctionScan.intro (parameterInput := {self.suffix(open_index + 1)}) "
            f"(afterParameters := {self.suffix(after_params)}) "
            f"(bodyInput := {self.suffix(brace + 1)}) "
            f"(parameters := [{', '.join(_scan_str(item) for item in params)}]) "
            f"(body := [{', '.join(_scan_stmt(item) for item in body)}]) "
            f"(openParenSpan := {_scan_span(self.ts[open_index])}) "
            f"(openBraceSpan := {_scan_span(self.ts[brace])}) (close := {_scan_token(close)})\n"
            f"    · rfl\n"
            f"    · exact (Eq.symm (congrArg (fun n => n + 1) "
            f"{self.stem}T{open_index + 1}_length)) ▸ {parameter_node}\n"
            f"    · exact {brace_drop}\n"
            f"    · exact (Eq.symm (congrArg (fun n => n + 16) "
            f"{self.stem}T{brace + 1}_length)) ▸ {body_node}"
        )
        node = self.add("FunctionScan", type_, fn_body, audit=True)
        selected_type = (f"SelectedFunctionParse {_scan_str(name)} {start} "
                         f"{self.suffix(open_index)} ({_scan_decl(declaration)}) "
                         f"{self.suffix(end)}")
        selected = self.add("SelectedFunctionParse", selected_type,
                            f"SelectedFunctionParse.intro {node}.sound", audit=True)
        return selected, declaration, end

    def expr_stmt_head(self, token: dict) -> str:
        kind = token["kind"]
        if kind == "numeral":
            return f"ExpressionStatementToken.numeral {token['value']} {_scan_span(token)}"
        if kind == "other":
            return (f"ExpressionStatementToken.other (Char.ofNat {token['value']}) "
                    f"{_scan_span(token)}")
        return f"ExpressionStatementToken.{kind} {_scan_span(token)}"

    def nonident(self, token: dict) -> str:
        kind = token["kind"]
        if kind == "numeral":
            return f"NonIdentifierToken.numeral {token['value']} {_scan_span(token)}"
        if kind == "other":
            return (f"NonIdentifierToken.other (Char.ofNat {token['value']}) "
                    f"{_scan_span(token)}")
        return f"NonIdentifierToken.{kind} {_scan_span(token)}"

    def constant(self, name: str, index: int):
        begin = index + 2
        stop = begin
        while stop < self.n and self.ts[stop]["kind"] != "newline":
            stop += 1
        local = _StructuralScan(f"{self.stem}C{index}", self.ts[begin:stop], [], [])
        local.emit_data()
        node, value, end = local.expr_top(0)
        assert end == local.n
        base = len(self.blocks)
        self.blocks += local.blocks
        for counter, count in local.counts.items():
            self.counts[counter] += count
        for entry in local.audit:
            self.audit.append({"name": entry["name"], "block": base + entry["block"]})
        declaration = {"name": name, "value": value,
                       "span": [self.ts[index]["start"], self.ts[stop - 1]["stop"]]}
        take = "TakeLineScan.newline (by decide)"
        for token in reversed(self.ts[begin:stop]):
            take = f"TakeLineScan.next (by decide) ({take})"
        type_ = (f"ConstantLineParse {local.stem}T0 ({_scan_expr(value)}) "
                 f"{local.stem}T{local.n}")
        parse = self.add("ConstantLineParse", type_,
                         f"ConstantLineParse.intro {node}.sound", audit=True)
        return declaration, stop, take, parse

    def wrap(self, action: dict, tail: str) -> str:
        kind = action["kind"]
        if kind == "non":
            return f"ScanSelected.skipNonIdentifier ({action['head']}) ({tail})"
        if kind == "identEnd":
            return f"ScanSelected.identEnd ({tail})"
        if kind == "skipOther":
            return f"ScanSelected.skipOther (by decide) (by decide) ({tail})"
        if kind == "skipConstant":
            return (f"ScanSelected.skipConstant (by decide) (by decide) (by decide) "
                    f"({tail})")
        if kind == "skipFunction":
            return (f"ScanSelected.skipFunction (by decide) (by decide) (by decide) "
                    f"({tail})")
        if kind == "constant":
            return (f"ScanSelected.constant (by decide) (by decide) ({action['take']}) "
                    f"{action['cp']} DropNewlinesScan.done (by rfl) ({tail})")
        if kind == "function":
            return (f"ScanSelected.function (by decide) (by decide) (by decide) "
                    f"{action['fp']} ({tail})")
        raise ValueError(kind)

    def scan(self) -> None:
        names_c = list(self.cn0)
        names_f = list(self.fn0)
        actions: list[dict] = []
        index = 0
        while index < self.n:
            token = self.ts[index]
            if token["kind"] != "ident":
                actions.append({"kind": "non", "i": index, "j": index + 1,
                                "head": self.nonident(token)})
                index += 1
                continue
            name = token["value"]
            if index + 1 == self.n:
                actions.append({"kind": "identEnd", "i": index, "j": index + 1})
                index += 1
                continue
            nxt = self.ts[index + 1]
            if nxt["kind"] == "assign":
                if name in names_c:
                    declaration, end, take, parse = self.constant(name, index)
                    names_c.remove(name)
                    actions.append({"kind": "constant", "i": index, "j": end,
                                    "d": declaration, "take": take, "cp": parse})
                    index = end
                else:
                    actions.append({"kind": "skipConstant", "i": index, "j": index + 1})
                    index += 1
            elif nxt["kind"] == "lparen":
                if name in names_f:
                    selected, declaration, end = self.function(name, token["start"], index + 1)
                    names_f.remove(name)
                    actions.append({"kind": "function", "i": index, "j": end,
                                    "d": declaration, "fp": selected})
                    index = end
                else:
                    actions.append({"kind": "skipFunction", "i": index, "j": index + 1})
                    index += 1
            else:
                actions.append({"kind": "skipOther", "i": index, "j": index + 1})
                index += 1
        if names_f or (self.stem != "table" and names_c):
            raise ValueError(("missing", names_c, names_f))
        constants_state = list(self.cn0)
        functions_state = list(self.fn0)
        states: list[tuple[list[str], list[str]]] = []
        for action in actions:
            states.append((list(constants_state), list(functions_state)))
            if action["kind"] == "constant":
                constants_state.remove(action["d"]["name"])
            if action["kind"] == "function":
                functions_state.remove(action["d"]["name"])
        states.append((constants_state, functions_state))
        out_constants: list[list[dict] | None] = [None] * (len(actions) + 1)
        out_functions: list[list[dict] | None] = [None] * (len(actions) + 1)
        out_constants[-1] = []
        out_functions[-1] = []
        for z in range(len(actions) - 1, -1, -1):
            out_constants[z] = ([actions[z]["d"]] + out_constants[z + 1]) \
                if actions[z]["kind"] == "constant" else out_constants[z + 1]
            out_functions[z] = ([actions[z]["d"]] + out_functions[z + 1]) \
                if actions[z]["kind"] == "function" else out_functions[z + 1]
        # Delimiter-stack-zero action boundaries: a chunk starts and ends at
        # stack depth zero, so no identifier or comment is split.
        stack: list[str] = []
        zero = {0}
        pairs = {"lparen": "rparen", "lbrace": "rbrace", "lbracket": "rbracket"}
        closes = {value: key for key, value in pairs.items()}
        for position, token in enumerate(self.ts):
            kind = token["kind"]
            if kind in pairs:
                stack.append(kind)
            elif kind in closes:
                assert stack and stack[-1] == closes[kind]
                stack.pop()
            if not stack:
                zero.add(position + 1)
        bounds = [0]
        last = 0
        for z, action in enumerate(actions, 1):
            if z - last >= 32 and action["j"] in zero:
                bounds.append(z)
                last = z
        if bounds[-1] != len(actions):
            bounds.append(len(actions))
        z = len(actions)
        fuel = self.n + 1 - z
        constants_names, functions_names = states[z]
        tail = self.add(
            "ScanSelected",
            f"ScanSelected {fuel} {self.suffix(self.n)} "
            f"[{', '.join(_scan_str(item) for item in constants_names)}] "
            f"[{', '.join(_scan_str(item) for item in functions_names)}] [] []",
            "ScanSelected.done",
        )
        for boundary in range(len(bounds) - 2, -1, -1):
            low, high = bounds[boundary], bounds[boundary + 1]
            term = "tail"
            for z in range(high - 1, low - 1, -1):
                term = self.wrap(actions[z], term)
            action = actions[low]
            fuel = self.n + 1 - low
            constants_names, functions_names = states[low]
            type_ = (f"ScanSelected {fuel} {self.suffix(action['i'])} "
                     f"[{', '.join(_scan_str(item) for item in constants_names)}] "
                     f"[{', '.join(_scan_str(item) for item in functions_names)}] "
                     f"[{', '.join(_scan_cdecl(item) for item in out_constants[low])}] "
                     f"[{', '.join(_scan_decl(item) for item in out_functions[low])}]")
            tail = self.add("ScanSelected", type_,
                            f"ScanSelected.continue (fun tail => {term}) {tail}",
                            audit=(low == 0))


def _witness_block_lines(block: list[str]) -> int:
    return sum(len(line.split("\n")) for line in block)


def _pack_witness_blocks(blocks: list[list[str]],
                         body_limit: int = _WITNESS_BODY_LINES) -> list[list[int]]:
    """Pack dependency-ordered blocks into module bodies below the line cap."""
    chunks: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, block in enumerate(blocks):
        cost = _witness_block_lines(block)
        if current:
            cost += 1
        if current and used + cost > body_limit:
            chunks.append(current)
            current = [index]
            used = _witness_block_lines(block)
        else:
            current.append(index)
            used += cost
    if current:
        chunks.append(current)
    return chunks


def _render_witness_module(namespace: str, imports: list[str],
                           blocks: list[list[str]]) -> str:
    lines = [
        "-- Generated by tools/slang/export.py; do not edit.",
        *[f"import {entry}" for entry in imports],
        "",
        "open Slang Slang.Decode Slang.Source Slang.Source.Parsing",
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


def render_source_witness(song_root: Path) -> tuple[
        dict[Path, str], list[dict[str, str]], list[str]]:
    """Render the structural `ScanSelected` witness module hierarchy.

    Returns the repo-absolute path to text map, the registered declaration
    records (name, expected type, module, path) and the module roots in
    emission order.
    """
    outputs: dict[Path, str] = {}
    declarations: list[dict[str, str]] = []
    module_roots: list[str] = []
    audit_imports: list[str] = []
    for program in _WITNESS_PROGRAMS:
        stem = str(program["stem"])
        namespace = str(program["namespace"])
        data = source_data(stem, song_root)
        generator = _StructuralScan(stem, data["tokens"],
                                    list(program["constants"]), list(program["functions"]))
        generator.emit_data()
        generator.scan()
        chunks = _pack_witness_blocks(generator.blocks)
        block_module = [""] * len(generator.blocks)
        block_path = [""] * len(generator.blocks)
        previous = "Slang.Source.Parsing.Tokens"
        for chunk_index, chunk in enumerate(chunks, start=1):
            module_name = f"{namespace}.Chunk{chunk_index:02d}"
            relative = (f"proof/corpus/Slang/Generated/SourceWitness/Parsing/"
                        f"{program['directory']}/Chunk{chunk_index:02d}.lean")
            text = _render_witness_module(
                namespace, [previous], [generator.blocks[index] for index in chunk]
            )
            outputs[REPO_ROOT / relative] = text
            module_roots.append(module_name)
            for index in chunk:
                block_module[index] = module_name
                block_path[index] = relative
            previous = module_name
        audit_imports.append(previous)
        for entry in generator.audit:
            full = f"{namespace}.{entry['name']}"
            block = int(entry["block"])
            declarations.append({
                "name": full,
                "expected_type": f"type_of% @{full}",
                "module": block_module[block],
                "path": block_path[block],
            })
    audit_lines = [
        "-- Generated by tools/slang/export.py; do not edit.",
        *[f"import {entry}" for entry in [*audit_imports, "Slang.Audit.Empty"]],
        "",
        "set_option autoImplicit false",
        "set_option linter.missingDocs false",
        "",
        f"namespace {_WITNESS_AUDIT_NAMESPACE}",
        "",
        "/-- The runner's mutation anchor: `rfl` uses no axiom; the negative control",
        "replaces this proof with a `Classical.choice` term. -/",
        "theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl",
        "",
        f"#audit_empty {_WITNESS_AUDIT_NAMESPACE}.nat_add_zero : (0 : Nat) + 0 = 0",
    ]
    for entry in declarations:
        audit_lines.append(f"#audit_empty {entry['name']} : {entry['expected_type']}")
    audit_lines.extend(["", f"end {_WITNESS_AUDIT_NAMESPACE}", ""])
    outputs[REPO_ROOT / _WITNESS_AUDIT_PATH] = "\n".join(audit_lines)
    module_roots.append(_WITNESS_AUDIT_NAMESPACE)
    declarations.insert(0, {
        "name": f"{_WITNESS_AUDIT_NAMESPACE}.nat_add_zero",
        "expected_type": "(0 : Nat) + 0 = 0",
        "module": _WITNESS_AUDIT_NAMESPACE,
        "path": _WITNESS_AUDIT_PATH,
    })
    return outputs, declarations, module_roots


# ---------------------------------------------------------------------------
# Structural whole-source UTF-8 witnesses
# ---------------------------------------------------------------------------

_UTF8_NAMESPACE = "Slang.Generated.SourceWitness.Tokenizer"
_UTF8_AUDIT_NAMESPACE = f"{_UTF8_NAMESPACE}.Audit"
_UTF8_AUDIT_PATH = "proof/corpus/Slang/Generated/SourceWitness/Tokenizer/Audit.lean"

# The generic structural lemmas the whole-source witnesses cite.  They live in
# the `Slang.Generated.Table` namespace because the parts they are applied to do,
# but in their own module so `Generated/Table.lean` keeps only the part
# presentation.  Each is a `List` fact proved without an axiom.
TABLE_SUPPORT_PATH = REPO_ROOT / "proof/corpus/Slang/Generated/TableSupport.lean"


def render_table_support() -> str:
    """Render the append/map/fuel lemmas the part witnesses use.

    These are presentation helpers: `sourceMapAppend` is `List.map_append` with
    the mapped list explicit, `sourceAppendAssoc`/`sourceAppendNil` are the
    `List` associativity and right-identity lemmas, and `sourceFuel` counts the
    code points one codepoint-aligned part adds.  None reduces a whole source
    list; each is a single structural step over `List`.
    """
    return "\n".join([
        "-- Generated by tools/slang/export.py; do not edit.",
        "import Slang.Source.Bytes",
        "",
        "set_option autoImplicit false",
        "set_option warningAsError true",
        "",
        "namespace Slang.Generated.Table",
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
        "carries `propext`, which the empty Slang cone rejects. -/",
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
        "end Slang.Generated.Table",
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
        "-- Generated by tools/slang/export.py; do not edit.",
        *[f"import {name}" for name in imports],
        "",
        "open Slang.Source.Tokenizer",
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
            f"Slang.Generated.Table.{stem}Part{index}.map UInt8.ofFin ++ tail",
            f"def {chars_name} (tail : List Char) : List Char := [{chars}] ++ tail",
            f"def {fuel_name} (tail : Nat) : Nat := "
            f"Slang.Generated.Table.sourceFuel {len(text)} tail",
            f"theorem {prefix_name} (fuel : Nat) (tailBytes : List UInt8) "
            f"(tailChars : List Char) (tail : Utf8Scan fuel tailBytes tailChars) : "
            f"Utf8Scan ({fuel_name} fuel) ({bytes_name} tailBytes) "
            f"({chars_name} tailChars) := by",
            f"  exact {_utf8_leaf_term(text)}",
            f"theorem {name}FuelEq (tail : Nat) : {fuel_name} tail = "
            f"tail + {len(text)} := Slang.Generated.Table.sourceFuel_eq _ _",
            f"theorem {name}Binding (tail : List UInt8) : "
            f"Slang.Generated.Table.{stem}Part{index}.map UInt8.ofFin ++ tail = "
            f"{bytes_name} tail := rfl",
        ])
        nodes.append({
            "name": name, "bytes": bytes_name, "chars": chars_name,
            "fuel": fuel_name, "prefix": prefix_name,
            "binding": f"{name}Binding",
            "fuel_eq": f"{name}FuelEq",
            "count": len(text),
            "table": f"Slang.Generated.Table.{stem}Part{index}",
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
            table_node = f"Slang.Generated.Table.{stem}PartNode{level}_{index // 2}"
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
                f"(Slang.Generated.Table.sourceMapAppend UInt8.ofFin "
                f"{left['table']} {right['table']})) "
                f"(Eq.trans (Slang.Generated.Table.sourceAppendAssoc _ _ _) (Eq.trans "
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
    source_name = f"Slang.Generated.Table.{stem}Source"
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
        f"  exact Slang.Generated.Table.{stem}SourceLength",
        f"theorem {stem}SourceMap : {source_name}.map UInt8.ofFin = {root['bytes']} [] := by",
        f"  change {root['table']}.map UInt8.ofFin = {root['bytes']} []",
        f"  exact Eq.trans (Slang.Generated.Table.sourceAppendNil _).symm "
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
        previous = "Slang.Generated.Table"
        final_path = ""
        for index, chunk in enumerate(chunks, 1):
            module = f"{namespace}.Chunk{index:02d}"
            relative = ("proof/corpus/Slang/Generated/SourceWitness/Tokenizer/"
                        f"{stem[0].upper() + stem[1:]}/Chunk{index:02d}.lean")
            outputs[REPO_ROOT / relative] = _utf8_module(
                namespace,
                [previous, "Slang.Generated.TableSupport", "Slang.Source.Tokenizer.Utf8"],
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
        "-- Generated by tools/slang/export.py; do not edit.",
        *[f"import {name}" for name in [*audit_imports, "Slang.Audit.Empty"]],
        "",
        "set_option autoImplicit false",
        "set_option linter.missingDocs false",
        "",
        f"namespace {_UTF8_AUDIT_NAMESPACE}",
        "",
        "theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl",
        "",
        f"#audit_empty {_UTF8_AUDIT_NAMESPACE}.nat_add_zero : "
        "type_of% @Slang.Generated.SourceWitness.Tokenizer.Audit.nat_add_zero",
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
    outputs: list[tuple[Path, str]] = [
        (args.output, generated),
        (args.derivation_output, derivation),
        (TABLE_SUPPORT_PATH, render_table_support()),
    ]
    witness_outputs, _witness_declarations, _witness_roots = render_source_witness(song_root)
    outputs.extend(sorted(witness_outputs.items()))
    utf8_outputs, _utf8_declarations, _utf8_roots = render_utf8_witness(song_root)
    outputs.extend(sorted(utf8_outputs.items()))
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
