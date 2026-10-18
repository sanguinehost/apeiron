#!/usr/bin/env python3
"""Shared low-level machinery for the joint profile consistency checkers.

This module holds the primitives that ``profile_check.py``,
``table_profile_check.py``, ``evaluator_profile_check.py`` and
``carrier_profile_check.py`` share: repository paths, the fail-closed check
helpers, Git and manifest identity helpers, Lean declaration parsing, Rust and
Song entry resolution, the strict reference grammar, and Markdown record
helpers. It establishes no profile-specific contract on its own.

The per-checker behaviour that differs is selected through module-level
configuration, set by the owning checker before validation runs:

* ``MISSING_ARTIFACTS`` - the exact paths a checker admits without resolving;
* ``_CHECK_IOTA`` - the Song declaration matcher;
* ``_CHECK_RUST`` - the Rust entry resolver, or ``None`` to treat ``.rs``
  references through the Lean declaration scan.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CORPUS_ROOT = REPO.parent / "sanguine" / "proof"
BRIDGE_ROOT = REPO / "proof"
SONG_ROOT = REPO.parent / "song"
PINNED_CORPUS_REPO = REPO / "proof" / "corpus" / ".lake" / "packages" / "apeiron"
CORPUS_REFERENCE_PREFIX = "../sanguine/"
CORPUS_TEXT_PREFIX = "sanguine/"

# Named artifacts deliberately absent from the pinned corpus. A reference that
# does not resolve is accepted only when its exact path is listed here; the
# exception is attached to the reference path, not to free text.
MISSING_ARTIFACTS: set[str] = set()

# Structured keys whose values have dedicated resolvers, not the text scan.
STRUCTURED_KEYS = {
    "module",
    "name",
    "symbol",
    "file",
    "root",
    "lang",
    "path",
    "sha256",
    "id",
    "title",
    "status",
    "kind",
    "grade",
    "role",
    "hypotheses",
    "representation_terms",
}

_BARE_FILE_INDEX: dict[str, list[Path]] | None = None
# The resolved corpus revision, set by the profile checker's identity step. It
# binds every referenced corpus module's bytes to the checked-out identity.
_RESOLVED_CORPUS_GIT: str | None = None

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

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
LEAN_DECL_LINE = re.compile(
    r"^\s*(?:noncomputable\s+|private\s+|protected\s+|@\[[^\]]*\]\s*)*"
    r"(theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
    r"([A-Za-z_][\w.']*)"
)
# Recorded kind -> the declaration keywords that satisfy it.
KIND_KEYWORDS = {
    "theorem": {"theorem", "lemma"},
    "def": {"def", "abbrev"},
    "abbrev": {"abbrev"},
    "structure": {"structure"},
    "class": {"class"},
    "inductive": {"inductive"},
    "instance": {"instance"},
}
# A declared Rust item name: the identifier bound by a ``fn``/``struct``/
# ``enum``/``trait``/``union``/``type``/``mod``/``const``/``static``
# declaration. A word that appears only in an ``impl`` header, a generic bound
# or a ``where`` clause binds no item and is not captured here.
RUST_ITEM = re.compile(
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|unsafe\s+)?"
    r"(?:fn|struct|enum|trait|union|type|mod|static|const)\s+([A-Za-z_]\w*)"
)
# A Song entry is a concrete executing definition: a function with a body, or a
# constant binding. A bare ``law``/``carrier``/``spec`` header is not an entry.
IOTA_DECL = [
    r"(?m)^\s*{n}\s*\([^\n]*\{",
    r"(?m)^\s*{n}\s*=",
]
# Strict reference grammar. A named reference is a repo-relative source path
# (``.lean``, ``.rs`` or ``.slang``) optionally followed by exactly one of::
#
#     path::symbol
#     path::Type::method
#     path::Type::{method, method, ...}
#     path::{symbol, Type::method, ...}
#     path::<T as Trait>::method
#     path::<T as Trait>::{method, ...}
#
# ``symbol``, ``Type``, ``T``, ``Trait`` and ``method`` are single identifiers;
# a brace member is a single identifier or a ``Type::method`` pair. The frozen
# profiles use the qualified self type ``<i64 as Ring>``, qualified brace
# members (``src/cell.rs::{Cell::mul, ...}``) and a qualified brace group
# (``src/map/table.rs::Table::{new, ...}``), so those forms are admitted. Every
# other form is malformed and rejected, never truncated or split: an extra
# ``::`` segment, a trailing ``.member``, a generic outside the self-type
# position, a brace member that is not a single identifier or a ``Type::method``
# pair, an empty segment, or an unbalanced brace/angle.
FILE_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:lean|rs|slang)")
IDENT = re.compile(r"[A-Za-z_]\w*")


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


def _norm(text: str) -> str:
    """Normalize Markdown prose for record presence checks."""
    return re.sub(r"\s+", " ", text.replace("`", "")).strip()


def _resolve_path(rel: str, where: str) -> Path:
    corpus_rel = None
    if rel.startswith(CORPUS_REFERENCE_PREFIX):
        corpus_rel = rel.removeprefix(CORPUS_REFERENCE_PREFIX)
    elif rel.startswith(CORPUS_TEXT_PREFIX):
        corpus_rel = rel.removeprefix(CORPUS_TEXT_PREFIX)
    if corpus_rel is not None:
        pinned = PINNED_CORPUS_REPO / corpus_rel
        if pinned.is_file():
            return pinned
        raise CheckError(
            f"{where}: source file does not resolve at the pinned corpus dependency: {rel}"
        )
    for base in (REPO, REPO.parent):
        candidate = base / rel
        if candidate.is_file():
            return candidate
    raise CheckError(f"{where}: source file does not resolve: {rel}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None


def _git_rev(path: Path) -> str | None:
    result = _git(["-C", str(path), "rev-parse", "HEAD"])
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_has_commit(path: Path, rev: str) -> bool:
    result = _git(["-C", str(path), "cat-file", "-e", f"{rev}^{{commit}}"])
    return result is not None and result.returncode == 0


def _repo_blob(repo_dir: Path, rev: str, rel_under_dir: str) -> bytes | None:
    """Read ``rel_under_dir`` at ``rev``, relative to the containing Git repo."""
    try:
        prefix = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--show-prefix"],
            capture_output=True,
            check=False,
        )
        if prefix.returncode != 0:
            return None
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(repo_dir),
                "show",
                f"{rev}:{prefix.stdout.decode('utf-8', 'replace').strip()}{rel_under_dir}",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if blob.returncode != 0:
        return None
    return blob.stdout


def _pinned_corpus_rev() -> str | None:
    """The corpus revision Apeiron depends on, from the committed Lake manifest.

    The sibling checkout under ``CORPUS_ROOT`` is a mutable other-stream worktree;
    its live HEAD is not Apeiron's recorded dependency. The pinned revision comes
    from ``proof/corpus/lake-manifest.json`` and its bytes are read from that Git
    revision.
    """
    for manifest in (
        REPO / "proof" / "corpus" / "lake-manifest.json",
        REPO / "proof" / "lake-manifest.json",
    ):
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for package in data.get("packages", []):
            if package.get("name") == "apeiron" and isinstance(
                package.get("rev"), str
            ):
                return package["rev"]
    return None


def _mathlib_rev() -> str | None:
    for manifest in (
        CORPUS_ROOT / "lake-manifest.json",
        CORPUS_ROOT.parent / "lake-manifest.json",
        REPO / "proof" / "corpus" / "lake-manifest.json",
    ):
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for package in data.get("packages", []):
            if package.get("name") == "mathlib" and isinstance(
                package.get("rev"), str
            ):
                return package["rev"]
    return None


def _lean_declarations(text: str) -> dict[str, set[str]]:
    """Map each fully-qualified Lean declaration name to its keywords.

    The namespace stack is tracked exactly, so a declaration binds to its real
    enclosing namespace rather than to any prefix opened in the file.
    """
    declarations: dict[str, set[str]] = {}
    stack: list[str] = []
    for line in text.splitlines():
        match = LEAN_DECL_LINE.match(line)
        if match is not None:
            keyword, name = match.group(1), match.group(2)
            full = ".".join(stack + [name]) if stack else name
            declarations.setdefault(full, set()).add(keyword)
        opened = re.match(r"^\s*namespace\s+([\w.]+)", line)
        if opened is not None:
            stack.extend(opened.group(1).split("."))
            continue
        closed = re.match(r"^\s*end\b\s*([\w.]*)", line)
        if closed is not None:
            name = closed.group(1)
            if name:
                for _ in name.split("."):
                    if stack:
                        stack.pop()
            elif stack:
                stack.pop()
    return declarations


def check_lean_ref(
    ref: object,
    where: str,
    *,
    require_root: str | None = None,
    require_root_strict: bool = False,
    require_kind: bool = False,
    bind_corpus: bool = False,
    gauge_hypotheses: bool = False,
) -> None:
    """Resolve a Lean reference, optionally binding its corpus bytes.

    ``require_root_strict`` keeps the profile checker's message order (validity
    before equality); the other checkers compare equality first. The accept and
    reject decisions are identical.
    """
    _need(isinstance(ref, dict), f"{where} entries must be objects")
    name = ref.get("name")
    module = ref.get("module")
    root = ref.get("root", require_root)
    kind = ref.get("kind")
    _nonempty(name, f"{where}.name")
    _nonempty(module, f"{where}.module")
    if require_root_strict:
        _need(root in ("corpus", "bridge"), f"{where}.root must be corpus or bridge")
        _need(root == require_root, f"{where}.root must be {require_root} for {name}")
    else:
        if require_root is not None:
            _need(
                root == require_root,
                f"{where}.root must be {require_root} for {name}",
            )
        _need(root in ("corpus", "bridge"), f"{where}.root must be corpus or bridge")
    if root == "corpus":
        # Read the canonical module at the pinned dependency revision, not from the
        # mutable sibling worktree: another stream advances that HEAD independently.
        pin = _pinned_corpus_rev()
        _need(
            pin is not None,
            f"{where}: cannot resolve the corpus dependency revision",
        )
        raw = _repo_blob(CORPUS_ROOT, pin, module)
        _need(
            raw is not None,
            f"{where}: module does not resolve at the pin: corpus:{module}",
        )
    else:
        path = BRIDGE_ROOT / module
        _need(path.is_file(), f"{where}: module does not resolve: {root}:{module}")
        raw = path.read_bytes()
        pin = None
    declarations = _lean_declarations(raw.decode("utf-8"))
    _need(
        name in declarations,
        f"{where}: declaration {name} is not the exact fully-qualified name "
        f"declared in {module}",
    )
    if bind_corpus and root == "corpus" and _RESOLVED_CORPUS_GIT is not None:
        resolved = _repo_blob(CORPUS_ROOT, _RESOLVED_CORPUS_GIT, module)
        _need(
            resolved is not None,
            f"{where}: cannot read {module} at corpus revision "
            f"{_RESOLVED_CORPUS_GIT}",
        )
        _need(
            hashlib.sha256(resolved).digest() == hashlib.sha256(raw).digest(),
            f"{where}: {module} bytes do not match the resolved corpus identity "
            f"{_RESOLVED_CORPUS_GIT}",
        )
    if require_kind or kind is not None:
        if kind is None:
            if require_kind:
                raise CheckError(f"{where}: {name} is missing a kind")
        else:
            _need(
                kind in KIND_KEYWORDS,
                f"{where}: invalid declaration kind {kind!r} for {name}",
            )
            _need(
                declarations[name] & KIND_KEYWORDS[kind],
                f"{where}: {name} is declared as {sorted(declarations[name])}, "
                f"not as {kind}",
            )
    if gauge_hypotheses and name.endswith("build_gauge_invariance") and kind == "theorem":
        hypotheses = ref.get("hypotheses")
        _nonempty(hypotheses, f"{where}.{name}.hypotheses")
        _need(
            "hcomm" in hypotheses and "hassoc" in hypotheses,
            f"{where}: build_gauge_invariance must carry hcomm and hassoc hypotheses",
        )


def _strip_leading_generics(header: str) -> str:
    s = header.lstrip()
    if not s.startswith("<"):
        return s
    depth = 0
    i = 0
    while i < len(s):
        char = s[i]
        if char == "-" and i + 1 < len(s) and s[i + 1] == ">":
            i += 2
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return s[i + 1 :]
        i += 1
    return ""


def _impl_generic_params(header: str) -> set[str]:
    """Names bound by an impl header's own leading ``<...>`` parameter list.

    A name in this set is a type parameter of the impl, not a concrete self
    type; a method declared in the impl cannot be attributed to it.
    """
    s = header.lstrip()
    if not s.startswith("<"):
        return set()
    rest = _strip_leading_generics(s)
    if not rest:
        return set()
    body = s[1 : len(s) - len(rest) - 1]
    names: set[str] = set()
    segments: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(body):
        char = body[i]
        if char == "-" and i + 1 < len(body) and body[i + 1] == ">":
            i += 2
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            segments.append(body[start:i])
            start = i + 1
        i += 1
    segments.append(body[start:])
    for segment in segments:
        segment = segment.strip()
        if segment.startswith("const "):
            segment = segment[len("const ") :].lstrip()
        match = re.match(r"([A-Za-z_]\w*)", segment)
        if match is not None:
            names.add(match.group(1))
    return names


def _impl_self_type(header: str) -> str | None:
    s = _strip_leading_generics(header)
    depth = 0
    i = 0
    while i < len(s):
        char = s[i]
        if char == "-" and i + 1 < len(s) and s[i + 1] == ">":
            i += 2
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif depth == 0 and s.startswith(" for ", i):
            s = s[i + 5 :]
            break
        i += 1
    match = re.match(r"\s*([A-Za-z_]\w*)", s)
    if match is None:
        return None
    base = match.group(1)
    # A generic parameter is not a concrete self type. Without this, a method
    # declared in ``impl<F: ...> Trait for F`` would be attributed to the type
    # parameter ``F``, so a reference such as ``F::home`` would resolve.
    if base in _impl_generic_params(header):
        return None
    return base


def _impl_trait(header: str) -> str | None:
    """Return the trait an ``impl Trait for T`` header names, or ``None``."""
    s = _strip_leading_generics(header)
    depth = 0
    i = 0
    while i < len(s):
        char = s[i]
        if char == "-" and i + 1 < len(s) and s[i + 1] == ">":
            i += 2
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif depth == 0 and s.startswith(" for ", i):
            match = re.match(r"\s*([A-Za-z_]\w*)", s[:i])
            return match.group(1) if match is not None else None
        i += 1
    return None


def _impl_blocks(text: str) -> list[tuple[str | None, str, set[str]]]:
    """Return ``(trait, self_type, method_names)`` for every ``impl`` block."""
    blocks: list[tuple[str | None, str, set[str]]] = []
    for match in re.finditer(r"(?m)^[ \t]*impl\b", text):
        start = match.end()
        brace = text.find("{", start)
        if brace < 0:
            continue
        header = text[start:brace]
        base = _impl_self_type(header)
        if base is None:
            continue
        trait = _impl_trait(header)
        depth = 0
        k = brace
        while k < len(text):
            char = text[k]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = text[brace + 1 : k]
        blocks.append((trait, base, set(re.findall(r"\bfn\s+([A-Za-z_]\w*)", body))))
    return blocks


def _declared_rust_names(text: str) -> set[str]:
    """Names bound by a Rust item declaration in ``text``.

    The declaration keyword (``fn``/``struct``/``enum``/``trait``/``union``/
    ``type``/``mod``/``const``/``static``) binds the name. A word that appears
    only in an ``impl`` header, a generic bound or a ``where`` clause is not a
    declaration and is not in the result.
    """
    return {match.group(1) for match in RUST_ITEM.finditer(text)}


def _check_rust(text: str, symbol: str) -> bool:
    if "::" in symbol:
        symbol = symbol.strip()
        trait: str | None = None
        # A leading generic qualifier such as ``<i64 as Ring>::add`` names the
        # implementing type ``i64`` and the trait ``Ring``; both must resolve to
        # an actual ``impl Ring for i64`` block in the file.
        if symbol.startswith("<"):
            rest = _strip_leading_generics(symbol)
            if not rest.startswith("::"):
                return False
            generic = symbol[: len(symbol) - len(rest)][1:-1]
            self_part, sep, trait_part = generic.partition(" as ")
            if not sep:
                return False
            self_match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*", self_part)
            trait_match = re.match(r"\s*([A-Za-z_]\w*)", trait_part)
            if self_match is None or trait_match is None:
                return False
            head = self_match.group(1)
            trait = trait_match.group(1)
            tail = rest[2:]
        else:
            head, _, tail = symbol.partition("::")
        head = head.strip()
        if re.fullmatch(r"[A-Za-z_]\w*", head) is None:
            return False
        members = [segment.strip() for segment in tail.split("::")] if tail else []
        if not members or any(
            re.fullmatch(r"[A-Za-z_]\w*", segment) is None for segment in members
        ):
            return False
        # Every segment after the named type must be declared inside an impl
        # whose self type is that type; when a ``<T as Trait>`` qualifier is
        # present, the impl trait must match it as well. The full qualification
        # is consumed: a trailing method is never silently ignored.
        methods: set[str] = set()
        for impl_trait, base, names in _impl_blocks(text):
            if base != head:
                continue
            if trait is not None and impl_trait != trait:
                continue
            methods |= names
        return all(member in methods for member in members)
    # A bare symbol resolves only to an item this file declares. A word that
    # appears only in an ``impl`` header, a generic bound (``Fn`` in
    # ``impl<K, F: Fn(&K) -> u64> Home<K> for F``) or a ``where`` clause binds
    # no item and is rejected. An impl self type is accepted exactly when the
    # same identifier is a declared item, which this test already covers.
    return symbol in _declared_rust_names(text)


def _check_iota(text: str, symbol: str) -> bool:
    token = re.escape(symbol)
    return any(
        re.search(pattern.replace("{n}", token), text) is not None
        for pattern in IOTA_DECL
    )


# Dispatch selected by the owning checker. ``_CHECK_RUST`` is ``None`` for a
# checker that treats ``.rs`` references through the Lean declaration scan.
_CHECK_IOTA = _check_iota
_CHECK_RUST = _check_rust


def _check_symbol_in_file(path: Path, symbol: str, where: str) -> None:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".slang":
        found = _CHECK_IOTA(text, symbol)
    elif path.suffix == ".rs" and _CHECK_RUST is not None:
        found = _CHECK_RUST(text, symbol)
    else:
        found = re.search(LEAN_ANY_DECL + re.escape(symbol) + r"\b", text) is not None
    _need(found, f"{where}: {symbol} not found in {path}")


def _bare_file_index() -> dict[str, list[Path]]:
    """Index bare source filenames under the local source trees."""
    global _BARE_FILE_INDEX
    if _BARE_FILE_INDEX is not None:
        return _BARE_FILE_INDEX
    index: dict[str, list[Path]] = {}
    skip = {".git", "target", ".lake", "node_modules", "build"}
    for root in (REPO, SONG_ROOT, PINNED_CORPUS_REPO / "proof"):
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in skip]
            for filename in filenames:
                index.setdefault(filename, []).append(Path(dirpath) / filename)
    _BARE_FILE_INDEX = index
    return index


def _resolve_text_ref(rel: str, where: str) -> Path:
    if rel.startswith(CORPUS_TEXT_PREFIX):
        candidate = PINNED_CORPUS_REPO / rel.removeprefix(CORPUS_TEXT_PREFIX)
        if candidate.is_file():
            return candidate
    for base in (
        REPO,
        PINNED_CORPUS_REPO / "proof",
        REPO.parent,
        BRIDGE_ROOT,
        SONG_ROOT,
        SONG_ROOT / "src",
    ):
        candidate = base / rel
        if candidate.is_file():
            return candidate
    if "/" not in rel and not rel.startswith("."):
        matches = _bare_file_index().get(rel, [])
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise CheckError(f"{where}: named reference does not resolve: {rel}")
        raise CheckError(f"{where}: bare reference {rel} is ambiguous: {matches}")
    raise CheckError(f"{where}: named reference does not resolve: {rel}")


def _malformed(where: str, ref: str, reason: str) -> CheckError:
    return CheckError(f"{where}: malformed reference {ref!r}: {reason}")


def _match_angle(text: str, start: int) -> int | None:
    """Return the index just past the ``>`` matching ``text[start] == '<'``."""
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == "-" and i + 1 < len(text) and text[i + 1] == ">":
            i += 2
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _reject_trailing(text: str, pos: int, where: str, ref: str) -> None:
    if text[pos : pos + 2] == "::":
        raise _malformed(where, ref + "::", "unexpected '::' after the reference")
    if (
        pos + 1 < len(text)
        and text[pos] == "."
        and (text[pos + 1].isalnum() or text[pos + 1] == "_")
    ):
        raise _malformed(
            where, ref + text[pos : pos + 2], "member access after the reference"
        )


def _brace_members(body: str, where: str, ref: str) -> list[str]:
    members = [token.strip() for token in body.split(",")]
    if not members or any(member == "" for member in members):
        raise _malformed(where, ref, "empty brace member")
    for member in members:
        parts = member.split("::")
        if len(parts) > 2 or any(IDENT.fullmatch(part) is None for part in parts):
            raise _malformed(
                where,
                ref,
                f"brace member {member!r} is not a single identifier or a "
                "Type::method pair",
            )
    return members


def _parse_reference(text: str, start: int, where: str, rel: str) -> list[str] | None:
    """Parse the ``::`` suffix at ``text[start]`` under the strict grammar.

    Returns the symbols to resolve, or ``None`` when the path is a bare
    mention. Raises :class:`CheckError` for any form outside the grammar.
    """
    if text[start : start + 2] != "::":
        return None
    ref = rel
    i = start + 2
    generic = ""
    if i < len(text) and text[i] == "<":
        close = _match_angle(text, i)
        if close is None:
            raise _malformed(where, ref + text[start:], "unbalanced generic qualifier")
        generic = text[i:close]
        if re.search(r"\S+\s+as\s+\S+", generic[1:-1]) is None:
            raise _malformed(
                where,
                rel + text[start:close],
                "generic qualifier is not a <T as Trait> self type",
            )
        if text[close : close + 2] != "::":
            raise _malformed(
                where,
                ref + text[start:close],
                "generic qualifier is not followed by '::'",
            )
        i = close + 2
        ref = rel + text[start : close + 2]
    if i >= len(text):
        raise _malformed(where, ref, "empty reference segment")
    if text[i] == "{":
        close = text.find("}", i + 1)
        if close < 0:
            raise _malformed(where, ref + text[i:], "unclosed brace group")
        full = rel + text[start : close + 1]
        members = _brace_members(text[i + 1 : close], where, full)
        _reject_trailing(text, close + 1, where, full)
        return [f"{generic}::{member}" if generic else member for member in members]
    match = IDENT.match(text, i)
    if match is None:
        raise _malformed(
            where, ref + text[i:], "expected an identifier, '{' or '<' after '::'"
        )
    segments = [match.group(0)]
    pos = match.end()
    while text[pos : pos + 2] == "::" and text[pos + 2 : pos + 3] != "{":
        following = IDENT.match(text, pos + 2)
        if following is None:
            raise _malformed(
                where,
                ref + text[pos : pos + 2],
                "'::' is not followed by an identifier",
            )
        segments.append(following.group(0))
        pos = following.end()
    if text[pos : pos + 2] == "::" and text[pos + 2 : pos + 3] == "{":
        if len(segments) != 1:
            raise _malformed(
                where,
                ref + "::" + "::".join(segments),
                "more than one '::' segment before a brace group",
            )
        close = text.find("}", pos + 3)
        if close < 0:
            raise _malformed(where, ref + text[start:], "unclosed brace group")
        type_name = segments[0]
        full = rel + text[start : close + 1]
        members = _brace_members(text[pos + 3 : close], where, full)
        _reject_trailing(text, close + 1, where, full)
        prefix = f"{generic}::{type_name}" if generic else type_name
        return [f"{prefix}::{member}" for member in members]
    if len(segments) > 2:
        raise _malformed(
            where,
            ref + "::" + "::".join(segments),
            "more than one '::' segment after the type",
        )
    symbol = "::".join(segments)
    full = rel + text[start : pos]
    _reject_trailing(text, pos, where, full)
    return [f"{generic}::{symbol}" if generic else symbol]


def _check_text_refs(text: str, where: str) -> None:
    """Resolve every named file/symbol reference in one prose string.

    References are parsed under the strict grammar above. A malformed
    reference is rejected rather than truncated or split.
    """
    for match in FILE_PATH.finditer(text):
        rel = match.group(0)
        end = match.end()
        symbols = _parse_reference(text, end, where, rel)
        if symbols is None and (
            end + 1 < len(text)
            and text[end] == "."
            and (text[end + 1].isalnum() or text[end + 1] == "_")
        ):
            raise _malformed(
                where, rel + text[end : end + 2], "member access after a file path"
            )
        try:
            path = _resolve_text_ref(rel, where)
        except CheckError:
            _need(
                rel in MISSING_ARTIFACTS,
                f"{where}: named reference {rel} does not resolve and is not a "
                "declared missing artifact",
            )
            continue
        for symbol in symbols or ():
            _check_symbol_in_file(path, symbol, where)


def _check_data_text_refs(value: object, where: str) -> None:
    if isinstance(value, str):
        _check_text_refs(value, where)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in STRUCTURED_KEYS:
                continue
            _check_data_text_refs(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_data_text_refs(item, f"{where}[{index}]")


def _row_section(text: str, row_id: str) -> str:
    heading = re.search(r"(?m)^###\s+" + re.escape(row_id) + r"\b[^\n]*$", text)
    if heading is None:
        return ""
    start = heading.end()
    nxt = re.search(r"(?m)^#{1,3}\s+", text[start:])
    end = start + nxt.start() if nxt else len(text)
    return text[start:end]


def _markdown_row_status(text: str, row_id: str) -> str | None:
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] != row_id:
            continue
        for cell in reversed(cells):
            if cell in ("selected-open", "blocker", "closed"):
                return cell
    return None
