#!/usr/bin/env python3
"""Fail-closed audit of the generated extraction against the named boundary allowlist.

Three rejections, each independent of the others:

1. `sorry` or `admit` anywhere in the file. Never allowed, listed or not.
2. `axiom` whose name is not in the allowlist file. A new unmodeled external is an unmodeled
   external.
3. An allowlisted name with no declaration in the file. The allowlist is exact in both
   directions: a permission that no longer applies is a stale claim about the cone and is a
   failure until the entry is removed.

`sorryAx` reaches this check as a named `axiom` and not as the token `sorry`, which is why
rule 1 is a token scan and rule 2 is a declaration scan; both are required.

The allowlist is a data file, not a literal here, so the reason each item is admitted is read
next to the name it admits.

Usage: audit-extraction.py GENERATED_LEAN ALLOWLIST_TSV
"""
import os
import re
import sys

# Same token shape the previous shell `audit()` used: a whole-word `axiom`/`sorry`/`admit`
# preceded by start-of-line or whitespace and followed by whitespace or end-of-line.
TOKEN = re.compile(r"(^|\s)(sorry|admit)(\s|$)")
AXIOM_LINE = re.compile(r"^axiom\s+(\S+)")
AXIOM_BARE = re.compile(r"^axiom\s*$")
CONT = re.compile(r"^\s+(\S+)")


def read_allowlist(path):
    """Return {name: reason}, rejecting a malformed line or a duplicate name."""
    allowed = {}
    raw_bytes = open(path, "rb").read()
    # The format is line-oriented, one entry per line. Without a final newline an appended entry
    # concatenates onto the previous reason and parses as a longer reason for a live name, which a
    # name/reason split cannot detect. Require the newline instead of guessing.
    if not raw_bytes.strip():
        sys.exit(f"allowlist {path}: no entries; an empty allowlist is not a policy")
    if not raw_bytes.endswith(b"\n"):
        sys.exit(f"allowlist {path}: missing final newline; the format is one entry per line")
    for lineno, raw in enumerate(raw_bytes.decode().split("\n"), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" not in raw:
            sys.exit(f"allowlist {path}:{lineno}: expected <name><TAB><reason>")
        name, reason = raw.split("\t", 1)
        name, reason = name.strip(), reason.strip()
        if not name or not reason:
            sys.exit(f"allowlist {path}:{lineno}: empty name or empty reason")
        if name in allowed:
            sys.exit(f"allowlist {path}:{lineno}: duplicate entry {name}")
        allowed[name] = reason
    if not allowed:
        sys.exit(f"allowlist {path}: no entries; an empty allowlist is not a policy")
    return allowed


def declared_axioms(lines):
    """Axiom names declared in the file, including the `axiom` / indented-name two-line form."""
    names = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = AXIOM_LINE.match(stripped)
        if m:
            names.append(m.group(1))
            continue
        if AXIOM_BARE.match(stripped) and i + 1 < len(lines):
            n = CONT.match(lines[i + 1])
            if n:
                names.append(n.group(1))
    return names


def main(lean_path, allowlist_path):
    allowed = read_allowlist(allowlist_path)
    text = open(lean_path).read()

    failures = []

    for lineno, line in enumerate(text.split("\n"), start=1):
        m = TOKEN.search(line)
        if m:
            failures.append(f"{lean_path}:{lineno}: `sorry`/`admit` is never admissible: {line.strip()}")

    declared = declared_axioms(text.split("\n"))
    for name in declared:
        if name not in allowed:
            failures.append(f"{lean_path}: unlisted axiom, unmodeled external: {name}")

    present = set(declared)
    for name in sorted(allowed):
        if name not in present:
            failures.append(
                f"{allowlist_path}: admitted boundary axiom absent from the extraction: {name}"
            )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print(
            f"audit-extraction: {lean_path} REJECTED "
            f"({len(declared)} axiom(s) declared, {len(allowed)} boundary item(s) admitted)",
            file=sys.stderr,
        )
        return 1

    admitted = sorted(n for n in declared if n in allowed)
    print(f"audit-extraction: {lean_path}: no sorry/admit; 0 unlisted axioms")
    print(f"audit-extraction: boundary items admitted ({len(admitted)}): {admitted}")
    for name in admitted:
        print(f"audit-extraction:   {name}: {allowed[name]}")
    unused = sorted(set(allowed) - present)
    if unused:
        print(f"audit-extraction: WARNING allowlist entries not declared: {unused}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: audit-extraction.py GENERATED_LEAN ALLOWLIST_TSV", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))