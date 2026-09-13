#!/usr/bin/env python3
"""Gate the CLAIMS.md headline ledger against the committed evidence.

A claimed headline (grade `proved`, `tested`, `measured`, `model` or
`anchor`) must:

  * for `proved`, name declarations that all resolve to a `#audit` line in
    `proof/corpus/AxiomAudit.lean`;
  * name a gate command (`E+B`, `cargo test`, `cargo bench`, `cargo kani`,
    `proof/regen.sh`, `lake build`);
  * record a grade and a revision/evidence identity.

Rows graded `no evidence / open` are declared obligations, not claims. They
are reported and must still carry a `MISSING` marker, but they do not fail
the gate. A row with an unrecognized grade or a duplicate ID fails.

Structural checks only: this does not run Lean or the Rust tests, and
`#audit` presence is not the axiom cone itself (the build checks that).

No external dependencies. Run from the repository root:

    python3 tools/check-claims.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "CLAIMS.md"
AUDIT = ROOT / "proof" / "corpus" / "AxiomAudit.lean"

GRADES = ("proved", "tested", "measured", "model", "anchor")
GRADE_RE = re.compile(r"(" + "|".join(GRADES) + r")\b")
GATE_RE = re.compile(
    r"(\bE\s*\+\s*B\b"
    r"|cargo\s+test\b"
    r"|cargo\s+bench\b"
    r"|cargo\s+kani\b"
    r"|proof/regen\.sh"
    r"|lake\s+build\b)"
)
EB_RE = re.compile(r"\bE\s*\+\s*B\b")
HEX_RE = re.compile(r"\b[0-9a-fA-F]{7,40}\b")
DECL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*$")
AUDIT_RE = re.compile(r"^\s*#audit\s+([A-Za-z0-9_'.]+)", re.MULTILINE)
BACKTICK_RE = re.compile(r"`([^`]+)`")


def read(path):
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"MISSING: cannot read {path}: {exc}")
        sys.exit(2)


def inventory_rows(text):
    """Parse the five-column headline table under '## Headline inventory'."""
    marker = "## Headline inventory"
    if marker not in text:
        print(f"MISSING: {CLAIMS} has no {marker!r} section")
        sys.exit(2)
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        if cells[0] in ("ID", "") or all(c in "-: " for c in cells[0]):
            continue
        rows.append(cells)
    if not rows:
        print(f"MISSING: {CLAIMS} headline table has no rows")
        sys.exit(2)
    return rows


def declarations(cell):
    """Backticked tokens that look like Lean declaration names."""
    out = []
    for token in BACKTICK_RE.findall(cell):
        token = token.strip()
        if DECL_RE.match(token) and ("_" in token or "." in token):
            out.append(token)
    return out


def resolves(decl, audits):
    """A declaration resolves if an #audit line names it, qualified or not."""
    if decl in audits:
        return True
    return any(a.endswith("." + decl) or decl.endswith("." + a) for a in audits)


def gate_named(row):
    return any(GATE_RE.search(cell) for cell in row)


def revision_recorded(row):
    """A revision is a hex commit in the row or a referenced evidence file."""
    for cell in row:
        if HEX_RE.search(cell):
            return True
    for cell in row:
        for token in BACKTICK_RE.findall(cell):
            if "/" in token or token.endswith((".md", ".rs", ".lean", ".sh")):
                if (ROOT / token).exists():
                    return True
    return False


def main():
    text = read(CLAIMS)
    audits = set(AUDIT_RE.findall(read(AUDIT)))
    rows = inventory_rows(text)

    if not audits:
        print(f"MISSING: {AUDIT} defines no #audit lines")
        return 1

    e_defined = "`E` = `proof/regen.sh --check`" in text
    b_defined = "`B` = `lake build`" in text

    failures = []
    warnings = []
    seen = {}
    claimed = 0
    open_rows = 0

    for row in rows:
        rid, decl_cell, grade_cell, gate_cell, evidence_cell = row[:5]
        if rid in seen:
            failures.append(f"{rid}: duplicate headline ID")
        seen[rid] = len(seen) + 1

        grade_lower = grade_cell.lower()
        is_open = "open" in grade_lower or "no evidence" in grade_lower
        if is_open:
            open_rows += 1
            print(f"OPEN  {rid}: {grade_cell}")
            if "MISSING" not in gate_cell and "MISSING" not in evidence_cell:
                failures.append(
                    f"{rid}: open row records neither a MISSING gate nor a MISSING witness"
                )
            continue

        claimed += 1
        before = len(failures)
        match = GRADE_RE.search(grade_lower)
        if not match:
            failures.append(f"{rid}: unrecognized grade {grade_cell!r}")
            grade = "?"
        else:
            grade = match.group(1)

        if grade == "proved":
            decls = declarations(decl_cell)
            if not decls:
                failures.append(f"{rid}: proved headline names no declaration")
            for decl in decls:
                if not resolves(decl, audits):
                    failures.append(
                        f"{rid}: declaration {decl} has no resolving #audit line in "
                        f"{AUDIT.relative_to(ROOT)}"
                    )

        if not gate_named(row):
            failures.append(f"{rid}: no named gate command in the record")
        if EB_RE.search(gate_cell) and not (e_defined and b_defined):
            failures.append(f"{rid}: gate uses E+B but CLAIMS.md does not define E and B")
        if not revision_recorded(row):
            failures.append(f"{rid}: no recorded grade/revision identity")

        if "MISSING" in gate_cell:
            warnings.append(f"{rid}: recorded obligation in witness/audit/gate: {gate_cell}")
        if "MISSING" in evidence_cell:
            warnings.append(f"{rid}: recorded obligation in evidence: {evidence_cell}")

        status = "OK" if len(failures) == before else "FAIL"
        print(f"{status:4}  {rid}: grade={grade} declarations={len(declarations(decl_cell))}")

    for warning in warnings:
        print(f"WARN  {warning}")

    print(
        f"\nCLAIMS.md: {len(rows)} rows, {claimed} claimed, {open_rows} open; "
        f"{len(audits)} #audit lines in {AUDIT.relative_to(ROOT)}"
    )
    if failures:
        print(f"FAIL: {len(failures)} ledger breach(es)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK: every claimed headline resolves an #audit line, names a gate and records a grade/revision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
