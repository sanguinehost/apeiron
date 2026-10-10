#!/usr/bin/env python3
"""Resolve every `apeiron:` glossary entry in the book against the source.

Any other backticked `Apeiron.` name in the book prose must resolve the same way.

A glossary entry is a list item that begins `- **apeiron:**`. Each backticked name in it is
resolved as follows.

  * `Apeiron.A.B.name` must be a fully qualified Lean declaration in `proof/canonical/Apeiron/`
    (the namespace stack of the file plus the declared name).
  * `word_*` must be a declaration in the refinement package `proof/corpus/*.lean`.
  * `Type::item` must name an item (`fn`, `struct`, `enum`, `const`, `type`, `trait`) in `src/`.
  * `none` entries must say `anchor` or `model`, so a glossary entry never carries a theorem
    claim without a declaration.

Exit 0 when every entry resolves, 1 otherwise. `--self-test` runs seeded defects.
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECL = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+|partial\s+)*"
    r"(?:theorem|def|abbrev|lemma|structure|inductive|class|instance|opaque|axiom)\s+([^\s:({\[]+)"
)
NS = re.compile(r"^\s*namespace\s+(\S+)")
SECTION = re.compile(r"^\s*section\b")
END = re.compile(r"^\s*end\b(?:\s+(\S+))?")
RUST_ITEM = re.compile(r"\b(?:fn|struct|enum|const|type|trait|static)\s+([A-Za-z_][A-Za-z0-9_]*)")
ENTRY = re.compile(r"^\s*-\s+\*\*apeiron:\*\*\s*(.*)$")
TICKS = re.compile(r"`([^`]+)`")


def lean_names(root):
    names = set()
    base = root / "proof" / "canonical" / "Apeiron"
    for path in sorted(base.rglob("*.lean")):
        stack = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = NS.match(line)
            if m:
                stack.append(m.group(1))
                names.add(".".join(x for x in stack if x))
                continue
            if SECTION.match(line):
                stack.append(None)
                continue
            m = END.match(line)
            if m and stack:
                stack.pop()
                continue
            m = DECL.match(line)
            if m:
                prefix = ".".join(s for s in stack if s)
                name = m.group(1)
                names.add(f"{prefix}.{name}" if prefix else name)
    return names


def bridge_names(root):
    """Short names of declarations in the refinement package `proof/corpus/*.lean`."""
    names = set()
    for path in sorted((root / "proof" / "corpus").glob("*.lean")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = DECL.match(line)
            if m:
                names.add(m.group(1).split(".")[-1])
    return names


WORD = re.compile(r"^word_[A-Za-z0-9_]+$")


def rust_items(root):
    items = set()
    for path in sorted((root / "src").rglob("*.rs")):
        items.update(RUST_ITEM.findall(path.read_text(encoding="utf-8", errors="replace")))
    return items


def entries(root):
    for path in sorted((root / "book" / "src").rglob("*.md")):
        in_fence = False
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if in_fence:
                continue
            m = ENTRY.match(line)
            if m:
                yield path.relative_to(root), number, m.group(1)


def check(root):
    lean = lean_names(root)
    rust = rust_items(root)
    bridge = bridge_names(root)
    findings = []
    count = 0
    for path, number, text in entries(root):
        count += 1
        ticks = TICKS.findall(text)
        named = False
        for token in ticks:
            if token.startswith("Apeiron."):
                named = True
                if token not in lean:
                    findings.append(f"{path}:{number}: unresolved Lean name `{token}`")
            elif WORD.match(token):
                named = True
                if token not in bridge:
                    findings.append(f"{path}:{number}: unresolved bridge theorem `{token}`")
            elif "::" in token:
                named = True
                item = token.split("::")[-1]
                if item not in rust:
                    findings.append(f"{path}:{number}: unresolved Rust item `{token}`")
        if not named and not re.match(r"\s*none\b.*\b(anchor|model)\b", text):
            findings.append(f"{path}:{number}: entry names no declaration and is not `none ... anchor|model`")
    for path in sorted((root / "book" / "src").rglob("*.md")):
        in_fence = False
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if in_fence or ENTRY.match(line):
                continue
            for token in TICKS.findall(line):
                if token.startswith("Apeiron.") and token not in lean:
                    findings.append(f"{path.relative_to(root)}:{number}: unresolved Lean name `{token}`")
                if WORD.match(token) and token not in bridge:
                    findings.append(f"{path.relative_to(root)}:{number}: unresolved bridge theorem `{token}`")
    if count == 0:
        findings.append("no glossary entries found under book/src")
    return count, findings


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "proof/canonical/Apeiron").mkdir(parents=True)
        (root / "proof/canonical/Apeiron/A.lean").write_text(
            "namespace Apeiron\nnamespace T\ntheorem good : True := trivial\nend T\nend Apeiron\n"
        )
        (root / "src").mkdir()
        (root / "src/a.rs").write_text("pub fn mul() {}\n")
        (root / "book/src").mkdir(parents=True)
        page = root / "book/src/p.md"
        cases = [
            ("- **apeiron:** `Apeiron.T.good`; Rust `Cell::mul`.\n", 0),
            ("- **apeiron:** `Apeiron.T.bad`.\n", 1),
            ("- **apeiron:** `Cell::nope`.\n", 1),
            ("- **apeiron:** none. `anchor`.\n", 0),
            ("- **apeiron:** none.\n", 1),
        ]
        ok = True
        for text, want in cases:
            page.write_text(text)
            _, findings = check(root)
            got = 1 if findings else 0
            if got != want:
                ok = False
                print(f"self-test FAIL: {text!r} wanted {want} got {got}: {findings}")
        print("self-test", "OK" if ok else "FAILED", f"({len(cases)} cases)")
        return 0 if ok else 1


def main():
    if "--self-test" in sys.argv:
        return self_test()
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    count, findings = check(root)
    for finding in findings:
        print("FAIL", finding)
    print(f"{count} glossary entries, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
