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
