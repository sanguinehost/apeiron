"""Structural scan witness rendering with explicit source and output roots."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .decode import _SELECTED_CONSTANT_NAMES, _SELECTED_FUNCTION_NAMES

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
# set.  Port of the `/tmp/opencode/gen_structural_scan.py` spike.
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


def render_source_witness(
        song_root: Path, source_provider: Callable[[str, Path], dict], repo_root: Path,
) -> tuple[
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
        data = source_provider(stem, song_root)
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
            outputs[repo_root / relative] = text
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
    outputs[repo_root / _WITNESS_AUDIT_PATH] = "\n".join(audit_lines)
    module_roots.append(_WITNESS_AUDIT_NAMESPACE)
    declarations.insert(0, {
        "name": f"{_WITNESS_AUDIT_NAMESPACE}.nat_add_zero",
        "expected_type": "(0 : Nat) + 0 = 0",
        "module": _WITNESS_AUDIT_NAMESPACE,
        "path": _WITNESS_AUDIT_PATH,
    })
    return outputs, declarations, module_roots


