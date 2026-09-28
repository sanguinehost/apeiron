#!/usr/bin/env python3
"""Rejecting gate for the canonical Kepler contract (APEIRON-KEPLER-W1-T2).

The frozen W0 contract is ``docs/contracts/kepler-canonical-intent.md``,
``docs/contracts/kepler-canonical-spec.md`` and ``docs/contracts/kepler-canonical-plan.md``. This
checker validates the semantic content of that contract, not its file names:
a document that keeps the shape but states an unsound claim must be rejected
with a diagnostic that names the field and the observed problem. A grep-only,
evidence-present gate is insufficient (plan row APEIRON-KEPLER-W1-T2).

What is checked
===============
* ``--check`` (default; optional ``--root <dir>`` for a fixture tree) reads the
  three contract documents and reports every violation of the rules below. It
  exits 0 only when all required fields are present and semantically correct:
  the absolute-defect enclosure implication with the exact-real residual
  premise and the ``d/(1-|e|)`` conclusion; the statement that a rounded
  residual ``|fl(EHat) - e*fl(sin EHat) - m)| <= d`` is NOT a sufficient
  certificate; ``eps = 2^-52`` (never a definitional ``2^-53``); the explicit
  ``Sigma::Elliptic`` solver face with the ``Sigma::default()`` prohibition;
  all six typed refusal variants as table rows; the predecessor attribution
  inventory covering exactly the 17 named declarations; the canonical corpus
  pin ``3d66ea3aa2fe1de5997b6b757247454a2f771358`` (a stale pin is rejected);
  the explicit non-claim that no Aeneas ``f64`` source refinement exists (a
  positive refinement claim is rejected); the four SLOT markers and the
  "satisfying, not merely capping, the iteration bound" service semantics.
* ``--self-test`` (also ``--controls``) runs both baseline/red/restore
  campaigns on scratch fixtures built from copies of the real files in a
  temporary directory; the working tree is never modified. The contract
  campaign applies six mutants (M1 rounded-residual-as-certificate, M2 stale
  pin, M3 dropped ``keplerRecovery_unique_floor`` inventory row, M4 dropped
  typed refusal row, M5 ``eps = 2^-53``, M6 added Aeneas ``f64`` refinement
  claim); each must be rejected with an attributable diagnostic naming the
  defect and the restored fixture must be green. The pin campaign applies the
  three W4 pin controls below. The self-test exits 0 only when every control
  behaved as required.
* ``--stage pin`` (APEIRON-KEPLER-W4-T0) validates the bridge pin files
  ``proof/corpus/lakefile.lean`` and ``proof/corpus/lake-manifest.json``: the
  corpus requirement exists; its revision is a full 40-hex commit; the lakefile
  revision equals the manifest's resolved revision and any revision the
  manifest records for that package; the pinned revision is a published commit
  in the sibling corpus checkout (``git -C <corpus> cat-file -e <rev>^{commit}``,
  default ``/home/socol/Workspace/sanguine-apeiron``, override with
  ``--corpus``); and the bridge roots retain ``Growable.ReadLoop`` (the
  sequencing invariant: the other owner's root entry lands before the pin
  bump). A Kepler root entry is deliberately not required until W5.
* ``--stage pin --controls`` runs the pin campaign on scratch copies of the
  real pin files: (a) the manifest's resolved revision drifted away from the
  lakefile revision (rejected by PIN-BUMP), (b) the pin replaced by an
  all-zeros unreachable commit (rejected by PIN-REACH), (c) ``Growable.ReadLoop``
  removed from the roots (rejected by PIN-ROOTS); each with a green restore.
  The summary prints the observed baseline/red/restore exit-code triple, which
  must be 0/1/0.
* ``--stage bridge`` (APEIRON-KEPLER-W5-T0) validates the W5-T1 deliverables
  against the real tree, delegating the pin to the pin stage: the certificate
  module ``proof/corpus/Kepler/Certificate.lean`` exists and carries the
  conditional interpretation (a supplied SOUND outward bound on the exact-real
  residual of the denoted ⟦⟧ inputs implies the ``d/(1-|e|)`` enclosure),
  rooted in ``Apeiron.Physics.Kepler.Enclosure`` /
  ``absolute_defect_enclosure`` (no second authority restating the bound), with
  no squared-residual substitution, no inward/understated read of the bound, no
  claim that the runtime constructs the bound, and no Aeneas ``f64`` refinement
  claim; ``proof/corpus/AxiomAudit.lean`` registers every public certificate
  headline; ``proof/corpus/lakefile.lean`` carries a Kepler root entry (with
  ``Growable.ReadLoop`` still in the roots, checked by the pin stage). Until
  W5-T1 lands, the real tree is EXPECTED RED and the stage exits nonzero with
  an explicit MISSING-deliverables list.
* ``--stage bridge --controls`` runs the fixture campaign (synthetic conformant
  certificate module + real pin files on scratch): (a) an inward/understated
  defect bound, (b1) a missing AxiomAudit registration, (b2) a missing Kepler
  root entry, (c) an added Aeneas ``f64`` refinement claim, plus (d) a
  squared-residual substitution and (e) a runtime-constructs-the-bound claim;
  each rejected with an attributable diagnostic and restored green, observed
  triple 0/1/0 per control. The fixture is isolated from the landed W5-T1
  artifacts: its certificate headline is a fixture-only name that cannot occur
  in the real AxiomAudit.lean, and the scratch lakefile copy carries exactly
  one Kepler root (the fixture's own; the real entries are stripped), so each
  mutation is provably load-bearing rather than masked by the real tree.
* ``--stage runtime`` (APEIRON-KEPLER-W6-T0) validates the W6-T1
  executing-kernel deliverables against the real tree WITHOUT compiling Rust:
  ``src/kepler.rs`` carries the public
  ``solve_kepler(face: Sigma, mean_anomaly: f64, eccentricity: f64)
  -> Result<KeplerSolution, KeplerError>`` API; only ``Sigma::Elliptic``
  executes and ``Sigma::default()`` is never used as a solver face; all six
  typed refusals are declared with no clamp/default substitution; the
  normalized domain (``|m| <= PI``, the ``0.95_f64`` service endpoint) and the
  fixed schedule thresholds (0.15/0.65/0.95) are present; the certificate
  construction escalates the rounded residual outward (a bare rounded residual
  is rejected, an inward read is rejected) and refuses ``CertificateFailure``
  when it cannot bound; the ``e == 0`` monad branch returns the input anomaly
  with zero rounds; and no Aeneas ``f64`` refinement is claimed.
  ``src/lib.rs`` exports the module; ``tests/kepler.rs`` carries the
  accepted/refused boundary cases, the independent-oracle hook, the bit-exact
  ``e == 0`` witness and the naive-accumulation falsifier. Until W6-T1 lands,
  the real tree is EXPECTED RED and the stage exits nonzero with an explicit
  MISSING-deliverables list. These are structural/token checks, not a
  compilation, a proof or a soundness measurement; the semantic obligations
  are `tested` by ``tests/kepler.rs`` at its sampled scope.
* ``--stage runtime --controls`` runs the fixture campaign on synthetic
  conformant kernel/lib/test sources in a scratch tree (the working tree is
  never touched): (a) ``Sigma::default()`` substituted for the solver face,
  (b) a bare rounded residual accepted instead of the outward certificate,
  (c) a missing ``InvalidEccentricity`` refusal replaced by a ``clamp``
  substitution; each rejected with an attributable diagnostic and restored
  green, observed triple 0/1/0 per control.
* ``--stage barrier`` (APEIRON-KEPLER-W7-T0) validates the W7-T1 integration
  record ``docs/evidence/kepler-canonical.md`` on the real tree without
  running a build. The record must carry: the resolved corpus revision
  ``880aa57e47e782ed7a9d9ee1974cfc87360b446b`` (and the actual bridge pin must
  record the same revision) and the Apeiron working-tree identity; SHA-256
  digests for ``src/kepler.rs``, ``tests/kepler.rs``,
  ``proof/corpus/Kepler/Certificate.lean`` and ``proof/corpus/AxiomAudit.lean``
  that MATCH the actual files on the same line as the file path (a stale or
  missing digest or identity is rejected); the shared gate receipts (corpus
  ``cd proof && lake build`` 3526 jobs, ``check-lints.py`` / ``check-style.py``
  / ``check-imports.py`` clean, corpus checker ``--check`` clean and
  ``--self-test`` 16/16, bridge ``lake build Kepler.Certificate AxiomAudit``
  3692 jobs with the closed cone ``{propext, Classical.choice, Quot.sound}``)
  plus the pin/bridge/runtime stage outputs and control matrices (``0/1/0``);
  the runtime population/SLA statement (12 bands x 1039 rows, 2 refusals at
  ``e=0.95``, SLA row OPEN) and the external-trust inventory (IEEE
  round-to-nearest and ``mul_add``, ``ASSUMPTION-SIN`` <= 1 ulp on
  ``|x| <= 8``, libm/compiler/hardware external, no Aeneas ``f64`` refinement
  claim). It rejects an abstract/bridge bound reported as an executable
  ``f64`` refinement, a dropped control receipt or refusal population, and a
  stale/missing digest or identity. Until W7-T1 lands the document is absent
  and the stage exits nonzero with an explicit MISSING-deliverable list.
* ``--stage barrier --controls`` runs the fixture campaign on a synthetic
  integration record in a scratch tree (the working tree is never touched):
  (A) an abstract/bridge bound reported as an executable ``f64`` refinement,
  (B) the runtime control receipt dropped, (C) a listed source digest that
  does not match the source; each rejected with an attributable diagnostic and
  restored green, observed triple 0/1/0 per control.
* ``--stage <other>`` never exits 0: the unimplemented stages
  (foundation/enclosure belong to the corpus checker) are reported as MISSING.
* ``--list`` prints the rule set and the scanned documents.

The validator is the single ``validate(root_or_texts)`` function so later
stages can reuse it against a fixture tree or in-memory texts.

Usage (budgets: 120 s wall clock for the contract modes, 180 s for
``--stage pin``, ``--stage barrier`` and the combined ``--self-test``; Python 3
standard library)::

    python3 tools/check-kepler.py --check [--root DIR]
    python3 tools/check-kepler.py --self-test
    python3 tools/check-kepler.py --stage pin [--corpus DIR] [--root DIR]
    python3 tools/check-kepler.py --stage pin --controls
    python3 tools/check-kepler.py --stage bridge [--corpus DIR] [--root DIR]
    python3 tools/check-kepler.py --stage bridge --controls
    python3 tools/check-kepler.py --stage runtime [--root DIR]
    python3 tools/check-kepler.py --stage runtime --controls
    python3 tools/check-kepler.py --stage barrier [--root DIR]
    python3 tools/check-kepler.py --stage barrier --controls

Exit status: 0 when clean or every control behaved as required; 1 on a
semantic rejection or a failing control; 2 on a usage or infrastructure error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve()
DEFAULT_ROOT = HERE.parents[1]

EXIT_OK = 0
EXIT_REJECT = 1
EXIT_USAGE = 2

SELF_TEST_BUDGET_S = 120
SUBPROCESS_TIMEOUT_S = 60

INTENT = "docs/contracts/kepler-canonical-intent.md"
SPEC = "docs/contracts/kepler-canonical-spec.md"
PLAN = "docs/contracts/kepler-canonical-plan.md"
CONTRACT_DOCS = (INTENT, SPEC, PLAN)

CANONICAL_PIN = "3d66ea3aa2fe1de5997b6b757247454a2f771358"

# The predecessor attribution inventory must cover exactly these declarations
# (docs/contracts/kepler-canonical-spec.md, "Predecessor attribution and migrations").
INVENTORY_DECLARATIONS = (
    "keplerMap",
    "kepler_iff_fixedpoint",
    "keplerMap_lipschitz",
    "keplerMap_strictly_contracts",
    "keplerMap_contractingWith",
    "kepler_unique",
    "eccentricAnomaly",
    "eccentricAnomaly_isFixedPt",
    "eccentricAnomaly_solves",
    "kepler_floor_exists_unique",
    "circular_is_monad",
    "kepler_collapse_trinity",
    "kepler_is_spinor_loop_collapse",
    "keplerResidual",
    "keplerRecovery",
    "keplerRecovery_unique_floor",
    "kepler_is_bias_injected_recovery",
)

REFUSAL_VARIANTS = (
    "NonFiniteInput",
    "InvalidEccentricity",
    "UnsupportedFace",
    "UnsupportedDomain",
    "IterationExhaustion",
    "CertificateFailure",
)

SLOT_MARKERS = (
    "SLOT-SPIKE-ITER",
    "SLOT-SPIKE-CERT",
    "SLOT-SPIKE-PHASE",
    "SLOT-ADAPTER",
)

REFUSAL_SECTION = "## Executing interface and refusal"
INVENTORY_SECTION = "## Predecessor attribution and migrations"
ENCLOSURE_SECTION = "## Enclosure and certificate interpretation"

RULES = (
    ("DOCS", "contract documents present", "each of the three contract documents exists"),
    ("PIN", "canonical corpus pin", f"the pin {CANONICAL_PIN} is recorded and no stale pin is presented"),
    ("ENCLOSURE", "absolute-defect enclosure", "the exact-real residual premise and the d/(1-|e|) conclusion"),
    ("ROUNDED-RESIDUAL", "rounded residual is not a certificate", "the |fl(...)| <= d check is stated insufficient, never sufficient"),
    ("EPS", "machine epsilon", "eps = 2^-52 is defined; a definitional 2^-53 is rejected"),
    ("FACE", "explicit solver face", "Sigma::Elliptic only; Sigma::default() prohibited as a solver face"),
    ("REFUSAL", "typed refusal variants", "all six KeplerError variants as refusal-table rows"),
    ("INVENTORY", "predecessor attribution", f"exactly the {len(INVENTORY_DECLARATIONS)} named declarations"),
    ("AENEAS", "Aeneas f64 non-claim", "no Aeneas f64 source refinement exists or is claimed"),
    ("SLA", "iteration SLA service semantics", "SLOT markers; a capped loop is not service"),
)


class GateError(Exception):
    """The gate cannot reach its acceptance condition."""


class MutationError(GateError):
    """A self-test mutation target was not found in the real documents."""


@dataclass(frozen=True)
class Diagnostic:
    rule: str
    path: str
    line: int
    message: str

    def render(self) -> str:
        where = self.path if self.line <= 0 else f"{self.path}:{self.line}"
        return f"RULE {self.rule} {where}: {self.message}"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Normalize typography that carries no semantics (Unicode minus, NBSP)."""
    return text.replace("\u2212", "-").replace("\u00a0", " ").replace("\u2011", "-")


def _lines(text: str) -> list[str]:
    return [_norm(line) for line in text.splitlines()]


def _find_line(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    """1-based number of the first line matching ``pattern``."""
    for number, line in enumerate(lines, 1):
        if pattern.search(line):
            return number
    return None


def _section(lines: list[str], heading: str) -> tuple[int, list[tuple[int, str]]] | None:
    """The heading's line number and (line number, text) pairs until the next
    ``## `` heading, or None when the heading is absent."""
    start = None
    body: list[tuple[int, str]] = []
    inside = False
    for number, line in enumerate(lines, 1):
        if line.strip() == heading:
            start = number
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside:
            body.append((number, line))
    if start is None:
        return None
    return start, body


# ---------------------------------------------------------------------------
# Rule patterns (matched on normalized lines)
# ---------------------------------------------------------------------------

PIN_NEAR_RE = re.compile(
    r"pin(?:ned)?[^0-9a-f]{0,24}(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])",
    re.IGNORECASE,
)
HEX40_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")

ENCLOSURE_IMPLICATION_RE = re.compile(
    r"\|F_\{m,e\}\(x\)\|\s*≤\s*d\s*→\s*\|x\s*-\s*E\*\|\s*≤\s*d\s*/\s*\(1\s*-\s*\|e\|\)"
)
EXACT_REAL_PREMISE_RE = re.compile(r"left premise is \*\*not\*\* the rounded computation")
EXACT_REAL_RESIDUAL_RE = re.compile(r"sin\(⟦[^⟧]{1,8}⟧\).{0,60}≤\s*⟦d⟧")

EPS_52_RE = re.compile(r"ε\s*=\s*2\^-52")
EPS_53_RE = re.compile(r"ε\s*=\s*2\^-53")

FACE_EXPLICIT_RE = re.compile(r"Sigma::Elliptic")
FACE_DEFAULT_PROHIBITED_RE = re.compile(
    r"NEVER use `Sigma::default\(\)` as a solver face"
)

REFUSAL_ROW_RE = re.compile(r"^\|\s*`([A-Za-z]+)`\s*\|")
INVENTORY_ROW_FIRST_CELL_RE = re.compile(r"^\|([^|]*)\|")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")

SLA_CAPPED_NOT_SERVICE_RE = re.compile(r"capped.*does \*\*not\*\* satisfy service")

AENEAS_RE = re.compile(r"Aeneas", re.IGNORECASE)
F64_RE = re.compile(r"\bf64\b|\bF64\b")
REFINE_WORD_RE = re.compile(
    r"\b(refine[sd]?|refining|refinement|prove[sd]?|proven|proves"
    r"|verified|verifies|establish(?:es|ed)?)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(no|not|never|none|nor|cannot|neither|without|until|forbidden)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def rule_docs(texts: dict[str, str]) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for rel in CONTRACT_DOCS:
        if rel not in texts:
            out.append(Diagnostic("DOCS", rel, 0, "contract document is missing"))
    return out


def rule_pin(texts: dict[str, str]) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for rel in CONTRACT_DOCS:
        if rel not in texts:
            continue
        lines = _lines(texts[rel])
        if not any(CANONICAL_PIN in line for line in lines):
            out.append(
                Diagnostic(
                    "PIN",
                    rel,
                    0,
                    f"canonical corpus pin {CANONICAL_PIN} is not recorded",
                )
            )
        for number, line in enumerate(lines, 1):
            for match in PIN_NEAR_RE.finditer(line):
                observed = match.group(1)
                if observed != CANONICAL_PIN:
                    out.append(
                        Diagnostic(
                            "PIN",
                            rel,
                            number,
                            f"stale canonical corpus pin: expected {CANONICAL_PIN}, "
                            f"observed {observed}",
                        )
                    )
    return out


def rule_enclosure(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    lines = _lines(texts[SPEC])
    anchor = 0
    section = _section(lines, ENCLOSURE_SECTION)
    if section is not None:
        anchor = section[0]
    if _find_line(lines, ENCLOSURE_IMPLICATION_RE) is None:
        out.append(
            Diagnostic(
                "ENCLOSURE",
                SPEC,
                anchor,
                "absolute-defect enclosure implication is missing: the general |e| < 1 "
                "form |F_{m,e}(x)| ≤ d → |x − E*| ≤ d / (1 − |e|)",
            )
        )
    if _find_line(lines, EXACT_REAL_PREMISE_RE) is None:
        out.append(
            Diagnostic(
                "ENCLOSURE",
                SPEC,
                anchor,
                "exact-real residual premise statement is missing: the contract must "
                "state the left premise is **not** the rounded computation",
            )
        )
    if _find_line(lines, EXACT_REAL_RESIDUAL_RE) is None:
        out.append(
            Diagnostic(
                "ENCLOSURE",
                SPEC,
                anchor,
                "exact-real residual bound |⟦EHat⟧ − ⟦e⟧ sin(⟦EHat⟧) − ⟦m⟧| ≤ ⟦d⟧ in ℝ "
                "is missing from the certificate interpretation",
            )
        )
    return out


def rule_rounded_residual(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    lines = _lines(texts[SPEC])
    anchor = 0
    section = _section(lines, ENCLOSURE_SECTION)
    if section is not None:
        anchor = section[0]
    if not any("fl(" in line and "insufficient" in line for line in lines):
        out.append(
            Diagnostic(
                "ROUNDED-RESIDUAL",
                SPEC,
                anchor,
                "insufficient-certificate statement is missing: the contract must state "
                "that checking |fl(EHat − e*fl(sin EHat) − m)| ≤ d is insufficient",
            )
        )
    sufficient_word = re.compile(r"\bsufficient\b")
    for number, line in enumerate(lines, 1):
        if "fl(" in line and sufficient_word.search(line) and "insufficient" not in line:
            out.append(
                Diagnostic(
                    "ROUNDED-RESIDUAL",
                    SPEC,
                    number,
                    "rounded residual is stated as a sufficient certificate: a contract "
                    "that accepts checking |fl(EHat − e*fl(sin EHat) − m)| ≤ d as "
                    "sufficient is unsound and must be rejected",
                )
            )
    return out


def rule_eps(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    spec_lines = _lines(texts[SPEC])
    if _find_line(spec_lines, EPS_52_RE) is None:
        out.append(
            Diagnostic(
                "EPS",
                SPEC,
                0,
                "machine epsilon definition ε = 2^-52 is missing from the spec",
            )
        )
    for rel in CONTRACT_DOCS:
        if rel not in texts:
            continue
        for number, line in enumerate(_lines(texts[rel]), 1):
            if EPS_53_RE.search(line):
                out.append(
                    Diagnostic(
                        "EPS",
                        rel,
                        number,
                        "machine epsilon defined as ε = 2^-53: the contract requires "
                        "ε = 2^-52 (binary64 machine epsilon, not unit roundoff)",
                    )
                )
    return out


def rule_face(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    lines = _lines(texts[SPEC])
    if _find_line(lines, FACE_EXPLICIT_RE) is None:
        out.append(
            Diagnostic(
                "FACE",
                SPEC,
                0,
                "explicit solver face is missing: the service must accept only "
                "face == Sigma::Elliptic",
            )
        )
    if _find_line(lines, FACE_DEFAULT_PROHIBITED_RE) is None:
        out.append(
            Diagnostic(
                "FACE",
                SPEC,
                0,
                "Sigma::default() prohibition is missing: the contract must state NEVER "
                "use `Sigma::default()` as a solver face (Default is Parabolic)",
            )
        )
    return out


def rule_refusal(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    lines = _lines(texts[SPEC])
    section = _section(lines, REFUSAL_SECTION)
    if section is None:
        out.append(
            Diagnostic(
                "REFUSAL",
                SPEC,
                0,
                f"refusal section '{REFUSAL_SECTION}' is missing from the spec",
            )
        )
        return out
    heading_line, body = section
    present = set()
    for _number, line in body:
        match = REFUSAL_ROW_RE.match(line)
        if match:
            present.add(match.group(1))
    for variant in REFUSAL_VARIANTS:
        if variant not in present:
            out.append(
                Diagnostic(
                    "REFUSAL",
                    SPEC,
                    heading_line,
                    f"typed refusal variant `{variant}` has no row in the refusal table "
                    f"under '{REFUSAL_SECTION}'",
                )
            )
    return out


def rule_inventory(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    lines = _lines(texts[SPEC])
    section = _section(lines, INVENTORY_SECTION)
    if section is None:
        out.append(
            Diagnostic(
                "INVENTORY",
                SPEC,
                0,
                f"attribution section '{INVENTORY_SECTION}' is missing from the spec",
            )
        )
        return out
    heading_line, body = section
    found: set[str] = set()
    for _number, line in body:
        match = INVENTORY_ROW_FIRST_CELL_RE.match(line)
        if not match:
            continue
        for token in re.findall(r"`([^`]+)`", match.group(1)):
            if IDENTIFIER_RE.match(token):
                found.add(token)
    expected = set(INVENTORY_DECLARATIONS)
    for declaration in sorted(expected - found):
        out.append(
            Diagnostic(
                "INVENTORY",
                SPEC,
                heading_line,
                f"predecessor attribution inventory is missing declaration "
                f"`{declaration}`",
            )
        )
    for declaration in sorted(found - expected):
        out.append(
            Diagnostic(
                "INVENTORY",
                SPEC,
                heading_line,
                f"predecessor attribution inventory lists unexpected declaration "
                f"`{declaration}` outside the frozen set of "
                f"{len(INVENTORY_DECLARATIONS)}",
            )
        )
    return out


def rule_aeneas(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    spec_lines = _lines(texts[SPEC])
    nonclaim_present = any(
        AENEAS_RE.search(line)
        and F64_RE.search(line)
        and REFINE_WORD_RE.search(line)
        and NEGATION_RE.search(line)
        for line in spec_lines
    )
    if not nonclaim_present:
        out.append(
            Diagnostic(
                "AENEAS",
                SPEC,
                0,
                "explicit Aeneas f64 non-claim is missing: the contract must state that "
                "no Aeneas `f64` source refinement exists for this path",
            )
        )
    for rel in CONTRACT_DOCS:
        if rel not in texts:
            continue
        for number, line in enumerate(_lines(texts[rel]), 1):
            if (
                AENEAS_RE.search(line)
                and F64_RE.search(line)
                and REFINE_WORD_RE.search(line)
                and not NEGATION_RE.search(line)
            ):
                out.append(
                    Diagnostic(
                        "AENEAS",
                        rel,
                        number,
                        "positive Aeneas f64 refinement claim: no source refinement may "
                        f"be claimed for this path: {line.strip()[:120]}",
                    )
                )
    return out


def rule_sla(texts: dict[str, str]) -> list[Diagnostic]:
    if SPEC not in texts:
        return []
    out: list[Diagnostic] = []
    lines = _lines(texts[SPEC])
    for marker in SLOT_MARKERS:
        if not any(marker in line for line in lines):
            out.append(
                Diagnostic("SLA", SPEC, 0, f"service slot marker {marker} is missing")
            )
    if _find_line(lines, SLA_CAPPED_NOT_SERVICE_RE) is None:
        out.append(
            Diagnostic(
                "SLA",
                SPEC,
                0,
                "iteration SLA service semantics is missing: the contract must state "
                "that a loop capped at the iteration bound returning "
                "IterationExhaustion/CertificateFailure does **not** satisfy service "
                "(the bound must be satisfied, not merely capped)",
            )
        )
    return out


RULE_FUNCS = (
    rule_docs,
    rule_pin,
    rule_enclosure,
    rule_rounded_residual,
    rule_eps,
    rule_face,
    rule_refusal,
    rule_inventory,
    rule_aeneas,
    rule_sla,
)


def load_texts(root: Path) -> dict[str, str]:
    """Read the three contract documents under ``root``."""
    texts: dict[str, str] = {}
    for rel in CONTRACT_DOCS:
        path = root / rel
        if path.is_file():
            texts[rel] = path.read_text(encoding="utf-8", errors="replace")
    return texts


def validate(source: "Path | dict[str, str]") -> list[Diagnostic]:
    """Every diagnostic for a contract tree (``Path`` root) or in-memory texts.

    Later stages reuse this function: an empty result is the acceptance
    condition, and each diagnostic names the rule, the document, the line and
    the observed problem.
    """
    if isinstance(source, dict):
        texts = source
    else:
        texts = load_texts(source)
    out: list[Diagnostic] = []
    for func in RULE_FUNCS:
        out.extend(func(texts))
    return out


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------

def cmd_check(root: Path) -> int:
    diagnostics = validate(root)
    if not diagnostics:
        print(
            "CHECK CLEAN: Kepler contract carries the exact-real enclosure, the "
            "rounded-residual insufficiency, eps = 2^-52, the explicit Sigma::Elliptic "
            "face with the Sigma::default() prohibition, all six typed refusals, the "
            f"{len(INVENTORY_DECLARATIONS)}-declaration attribution inventory, pin "
            f"{CANONICAL_PIN}, the Aeneas f64 non-claim and the SLA service semantics"
        )
        return EXIT_OK
    for diagnostic in diagnostics:
        print(diagnostic.render())
    by_rule: dict[str, int] = {}
    for diagnostic in diagnostics:
        by_rule[diagnostic.rule] = by_rule.get(diagnostic.rule, 0) + 1
    summary = ", ".join(f"{rule}={count}" for rule, count in sorted(by_rule.items()))
    print(f"CHECK FAIL: {len(diagnostics)} contract violation(s) ({summary})")
    return EXIT_REJECT


# ---------------------------------------------------------------------------
# --self-test / --controls: baseline / red / restore campaign
# ---------------------------------------------------------------------------

STALE_PIN = "0123456789abcdef0123456789abcdef01234567"


def _m1_rounded_residual_sufficient(texts: dict[str, str]) -> dict[str, str]:
    target = "is **insufficient**"
    if texts[SPEC].count(target) != 1:
        raise MutationError(f"M1 target {target!r} not found exactly once in the spec")
    texts[SPEC] = texts[SPEC].replace(target, "is **sufficient**")
    return texts


def _m2_stale_pin(texts: dict[str, str]) -> dict[str, str]:
    for rel in CONTRACT_DOCS:
        if CANONICAL_PIN not in texts[rel]:
            raise MutationError(f"M2 target pin not found in {rel}")
        texts[rel] = texts[rel].replace(CANONICAL_PIN, STALE_PIN)
    return texts


def _drop_row(texts: dict[str, str], row_prefix: str, name: str) -> dict[str, str]:
    lines = texts[SPEC].splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith(row_prefix)]
    if len(lines) - len(kept) != 1:
        raise MutationError(f"{name} target row {row_prefix!r} not found exactly once")
    texts[SPEC] = "".join(kept)
    return texts


def _m3_drop_inventory_row(texts: dict[str, str]) -> dict[str, str]:
    return _drop_row(texts, "| `keplerRecovery_unique_floor`", "M3")


def _m4_drop_refusal_row(texts: dict[str, str]) -> dict[str, str]:
    return _drop_row(texts, "| `IterationExhaustion`", "M4")


def _m5_eps_2p53(texts: dict[str, str]) -> dict[str, str]:
    target = "`ε = 2^-52`"
    if texts[SPEC].count(target) != 1:
        raise MutationError(f"M5 target {target!r} not found exactly once in the spec")
    texts[SPEC] = texts[SPEC].replace(target, "`ε = 2^-53`")
    return texts


def _m6_aeneas_claim(texts: dict[str, str]) -> dict[str, str]:
    claim = (
        "\n\nThe Aeneas extraction of `solve_kepler` refines the Rust `f64` solver "
        "body to the canonical Kepler fixed point; this refinement is proved at the "
        "extracted source.\n"
    )
    texts[SPEC] = texts[SPEC] + claim
    return texts


@dataclass(frozen=True)
class Control:
    name: str
    defect: str
    rule: str
    tokens: tuple[str, ...]
    mutate: Callable[[dict[str, str]], dict[str, str]]


CONTROLS: tuple[Control, ...] = (
    Control(
        "M1",
        "rounded-residual-as-certificate accepted",
        "ROUNDED-RESIDUAL",
        ("fl(", "sufficient"),
        _m1_rounded_residual_sufficient,
    ),
    Control(
        "M2",
        "stale canonical pin revision",
        "PIN",
        (STALE_PIN,),
        _m2_stale_pin,
    ),
    Control(
        "M3",
        "keplerRecovery_unique_floor inventory row dropped",
        "INVENTORY",
        ("keplerRecovery_unique_floor",),
        _m3_drop_inventory_row,
    ),
    Control(
        "M4",
        "IterationExhaustion typed refusal row dropped",
        "REFUSAL",
        ("IterationExhaustion",),
        _m4_drop_refusal_row,
    ),
    Control(
        "M5",
        "eps written as 2^-53",
        "EPS",
        ("2^-53",),
        _m5_eps_2p53,
    ),
    Control(
        "M6",
        "Aeneas f64 refinement claim added",
        "AENEAS",
        ("Aeneas", "refinement"),
        _m6_aeneas_claim,
    ),
)


def _run_checker(fixture_root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(HERE), "--check", "--root", str(fixture_root)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _write_docs(root: Path, texts: dict[str, str]) -> None:
    for rel in CONTRACT_DOCS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texts[rel], encoding="utf-8")


def _first_rule_line(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("RULE "):
            return line[:110]
    return output.strip().splitlines()[0][:110] if output.strip() else "(no output)"


def contract_campaign() -> int:
    """Contract campaign: baseline green, every mutant red with an attributable
    diagnostic, every restore green, on scratch fixtures; the working tree is
    untouched."""
    real_texts = load_texts(DEFAULT_ROOT)
    missing = [rel for rel in CONTRACT_DOCS if rel not in real_texts]
    if missing:
        print(f"FAIL: real contract document(s) missing: {', '.join(missing)}", file=sys.stderr)
        return EXIT_USAGE

    tmp_parent: str | None = None
    preferred = Path("/tmp/opencode")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        tmp_parent = str(preferred)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="kepler-w1-t2-", dir=tmp_parent) as tmp:
        fixture = Path(tmp) / "fixture"
        _write_docs(fixture, real_texts)

        rows: list[tuple[str, str, str, bool]] = []

        code, output = _run_checker(fixture)
        ok = code == EXIT_OK and "CHECK CLEAN" in output
        rows.append(
            (
                "baseline",
                "exit 0 GREEN",
                f"exit {code} " + ("GREEN" if ok else output.strip()[:60] or "no output"),
                ok,
            )
        )
        if not ok:
            failures.append("baseline fixture is not green: the real contract or the "
                            "checker is broken; refusing to claim controls")

        for control in CONTROLS:
            texts = dict(real_texts)
            try:
                texts = control.mutate(texts)
            except MutationError as exc:
                print(f"FAIL: {control.name} mutation infrastructure error: {exc}", file=sys.stderr)
                return EXIT_USAGE
            _write_docs(fixture, texts)
            code, output = _run_checker(fixture)
            attributable = (
                f"RULE {control.rule} " in output
                and all(token in output for token in control.tokens)
            )
            red_ok = code == EXIT_REJECT and attributable
            observed = (
                f"exit {code} :: {_first_rule_line(output)}"
                if output.strip()
                else f"exit {code} :: (no output)"
            )
            rows.append(
                (
                    f"mutant {control.name} ({control.defect})",
                    f"exit 1 RULE {control.rule}",
                    observed,
                    red_ok,
                )
            )
            if not red_ok:
                reason = "survived" if code == EXIT_OK else (
                    f"rejected without attributable {control.rule} diagnostic"
                    if code == EXIT_REJECT
                    else f"infrastructure exit {code}"
                )
                failures.append(f"{control.name}: {reason}")

            _write_docs(fixture, real_texts)
            code, output = _run_checker(fixture)
            restore_ok = code == EXIT_OK and "CHECK CLEAN" in output
            rows.append(
                (
                    f"restore {control.name}",
                    "exit 0 GREEN",
                    f"exit {code} " + ("GREEN" if restore_ok else output.strip()[:60] or "no output"),
                    restore_ok,
                )
            )
            if not restore_ok:
                failures.append(f"{control.name}: restored fixture is not green")

    width = max(len(row[0]) for row in rows)
    print(f"{'CONTROL MATRIX':<{width}}  EXPECTED                        OBSERVED")
    print("-" * (width + 80))
    for name, expected, observed, ok in rows:
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<{width}}  {expected:<31} {observed:<31} {verdict}")
    if failures:
        print(f"CONTRACT CAMPAIGN FAIL: {len(failures)} control(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_REJECT
    print(
        "CONTRACT CAMPAIGN PASS: baseline green, 6/6 semantic mutants rejected "
        "with attributable diagnostics, 6/6 restores green; working tree untouched"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# --stage pin (APEIRON-KEPLER-W4-T0): bridge pin files vs published corpus
# ---------------------------------------------------------------------------

LAKEFILE_REL = "proof/corpus/lakefile.lean"
MANIFEST_REL = "proof/corpus/lake-manifest.json"
CORPUS_CHECKOUT = Path("/home/socol/Workspace/sanguine-apeiron")
PIN_HEX_RE = re.compile(r"^[0-9a-f]{40}$")
ALL_ZEROS_REV = "0" * 40
GIT_TIMEOUT_S = 20

# The corpus requirement in the lakefile (the revision sits on a continuation
# line after the '@', hence \s+ across the newline):
#   require «apeiron» from git "…sanguine.git" @
#     "3d66ea3aa2fe1de5997b6b757247454a2f771358" / "proof"
CORPUS_REQUIRE_RE = re.compile(
    r'require\s+(?:«)?apeiron(?:»)?\s+from\s+git\s+"[^"]+"\s+@\s+"([^"]*)"'
)
READLOOP_ROOT_RE = re.compile(r"`Growable\.ReadLoop\b")

PIN_STAGE_RULES = (
    ("PIN-BUMP", "corpus requirement and revision agreement",
     "lakefile and manifest both pin the corpus; revisions agree and are 40-hex"),
    ("PIN-REACH", "published-commit reachability",
     "the pinned revision exists as a commit in the sibling corpus checkout"),
    ("PIN-ROOTS", "Growable.ReadLoop root retained",
     "the other owner's root entry stays registered before any pin bump"),
)


def _git(corpus: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(corpus), *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
    )


def _preflight_corpus(corpus: Path) -> None:
    """The sibling checkout must be a usable git repository before any
    reachability claim; a broken checkout is infrastructure failure, never a
    silent pass or a semantic rejection."""
    try:
        proc = _git(corpus, ["rev-parse", "--git-dir"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"corpus checkout {corpus} is not usable: {exc}") from exc
    if proc.returncode != 0:
        raise GateError(
            f"corpus checkout {corpus} is not a usable git repository: "
            f"{proc.stderr.strip()[:120]}"
        )


def _corpus_commit_exists(corpus: Path, rev: str) -> bool:
    proc = _git(corpus, ["cat-file", "-e", f"{rev}^{{commit}}"])
    return proc.returncode == 0


def _pinned_lakefile_rev(lakefile_text: str) -> "str | None":
    match = CORPUS_REQUIRE_RE.search(lakefile_text)
    return match.group(1) if match else None


def validate_pin(root: Path, corpus: Path) -> list[Diagnostic]:
    """Every diagnostic for the bridge pin stage; an empty result accepts.

    Checks the two real pin files for: a present corpus requirement, a full
    40-hex revision, lakefile/manifest/inputRev agreement, published-commit
    reachability in the sibling corpus checkout, and the retained
    ``Growable.ReadLoop`` root. A Kepler root is not required until W5.
    """
    _preflight_corpus(corpus)
    out: list[Diagnostic] = []
    lakefile_path = root / LAKEFILE_REL
    manifest_path = root / MANIFEST_REL
    if not lakefile_path.is_file():
        out.append(Diagnostic("PIN-BUMP", LAKEFILE_REL, 0, "bridge lakefile is missing"))
        return out
    if not manifest_path.is_file():
        out.append(Diagnostic("PIN-BUMP", MANIFEST_REL, 0, "bridge lake-manifest is missing"))
        return out
    lakefile_text = lakefile_path.read_text(encoding="utf-8", errors="replace")
    lakefile_lines = _lines(lakefile_text)

    # Lakefile: requirement present, revision well-formed.
    lakefile_rev = _pinned_lakefile_rev(_norm(lakefile_text))
    if lakefile_rev is None:
        out.append(
            Diagnostic(
                "PIN-BUMP",
                LAKEFILE_REL,
                0,
                "corpus requirement is missing: expected "
                "`require «apeiron» from git \"…sanguine.git\" @ \"<40-hex>\" / \"proof\"`",
            )
        )
    elif not PIN_HEX_RE.match(lakefile_rev):
        out.append(
            Diagnostic(
                "PIN-BUMP",
                LAKEFILE_REL,
                _find_line(lakefile_lines, re.compile(re.escape(lakefile_rev))) or 0,
                f"lakefile corpus revision is not a full 40-hex commit: "
                f"observed {lakefile_rev!r}",
            )
        )

    # Manifest: exactly one 'apeiron' package with a well-formed revision.
    manifest_text = manifest_path.read_text(encoding="utf-8", errors="replace")
    manifest_lines = _lines(manifest_text)
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        out.append(
            Diagnostic("PIN-BUMP", MANIFEST_REL, 0, f"lake-manifest is not valid JSON: {exc}")
        )
        manifest = None
    manifest_rev = None
    manifest_input_rev = None
    if manifest is not None:
        packages = [
            entry
            for entry in manifest.get("packages", [])
            if isinstance(entry, dict) and entry.get("name") == "apeiron"
        ]
        if not packages:
            out.append(
                Diagnostic(
                    "PIN-BUMP",
                    MANIFEST_REL,
                    0,
                    "lake-manifest records no package named 'apeiron' (the corpus pin)",
                )
            )
        elif len(packages) > 1:
            out.append(
                Diagnostic(
                    "PIN-BUMP",
                    MANIFEST_REL,
                    0,
                    f"lake-manifest records {len(packages)} packages named 'apeiron'; "
                    "exactly one corpus pin is required",
                )
            )
        else:
            entry = packages[0]
            manifest_rev = entry.get("rev")
            manifest_input_rev = entry.get("inputRev")
            if not isinstance(manifest_rev, str) or not PIN_HEX_RE.match(manifest_rev):
                out.append(
                    Diagnostic(
                        "PIN-BUMP",
                        MANIFEST_REL,
                        0,
                        "manifest 'apeiron' resolved rev is not a full 40-hex commit: "
                        f"observed {manifest_rev!r}",
                    )
                )

    # Agreement: lakefile rev == manifest rev == every recorded revision.
    if lakefile_rev is not None and manifest_rev is not None and lakefile_rev != manifest_rev:
        out.append(
            Diagnostic(
                "PIN-BUMP",
                MANIFEST_REL,
                _find_line(manifest_lines, re.compile(re.escape(manifest_rev))) or 0,
                f"pin mismatch: lakefile pins {lakefile_rev} but the manifest "
                f"resolves {manifest_rev}",
            )
        )
    if (
        manifest_rev is not None
        and manifest_input_rev is not None
        and manifest_input_rev != manifest_rev
    ):
        out.append(
            Diagnostic(
                "PIN-BUMP",
                MANIFEST_REL,
                _find_line(manifest_lines, re.compile(re.escape(manifest_input_rev))) or 0,
                f"pin mismatch: manifest 'apeiron' inputRev {manifest_input_rev} "
                f"differs from its resolved rev {manifest_rev}",
            )
        )

    # Reachability: the pinned revision must be a published commit in the
    # sibling corpus checkout. Checked on the lakefile (requirement) revision.
    if lakefile_rev is not None and PIN_HEX_RE.match(lakefile_rev):
        if not _corpus_commit_exists(corpus, lakefile_rev):
            out.append(
                Diagnostic(
                    "PIN-REACH",
                    LAKEFILE_REL,
                    _find_line(lakefile_lines, re.compile(re.escape(lakefile_rev))) or 0,
                    f"pinned revision {lakefile_rev} is not a published commit in "
                    f"{corpus} (git cat-file -e {lakefile_rev}^{{commit}} failed)",
                )
            )

    # Sequencing invariant: the other owner's root entry stays registered.
    if _find_line(lakefile_lines, READLOOP_ROOT_RE) is None:
        out.append(
            Diagnostic(
                "PIN-ROOTS",
                LAKEFILE_REL,
                0,
                "bridge roots no longer retain `Growable.ReadLoop`: the other owner's "
                "root entry must land (and stay) before the pin bump; a Kepler root "
                "is not required until W5",
            )
        )
    return out


def cmd_pin_check(root: Path, corpus: Path) -> int:
    diagnostics = validate_pin(root, corpus)
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic.render())
        print(f"PIN CHECK FAIL: {len(diagnostics)} pin violation(s)")
        return EXIT_REJECT
    lakefile_rev = _pinned_lakefile_rev(
        _norm((root / LAKEFILE_REL).read_text(encoding="utf-8", errors="replace"))
    )
    print(
        f"PIN CHECK CLEAN: corpus requirement present; lakefile, manifest rev and "
        f"inputRev agree on {lakefile_rev}; the revision is a published commit in "
        f"{corpus}; `Growable.ReadLoop` root retained"
    )
    return EXIT_OK


# --- pin controls -----------------------------------------------------------

@dataclass(frozen=True)
class PinControl:
    name: str
    defect: str
    rule: str
    tokens: tuple[str, ...]
    mutate: "Callable[[dict[str, str], str, Path], dict[str, str]]"


def _pin_mut_manifest_drift(
    texts: dict[str, str], pinned_rev: str, corpus: Path
) -> dict[str, str]:
    manifest = texts[MANIFEST_REL]
    target = f'"rev": "{pinned_rev}"'
    if manifest.count(target) != 1:
        raise MutationError(f"pin control A target {target!r} not found exactly once")
    drift = None
    for candidate in _git(corpus, ["rev-list", "--max-count=8", "HEAD"]).stdout.split():
        if candidate != pinned_rev:
            drift = candidate
            break
    if drift is None:
        raise MutationError("no sibling revision distinct from the pin found for control A")
    texts[MANIFEST_REL] = manifest.replace(target, f'"rev": "{drift}"')
    return texts


def _pin_mut_unreachable_zeros(
    texts: dict[str, str], pinned_rev: str, corpus: Path
) -> dict[str, str]:
    lakefile = texts[LAKEFILE_REL]
    if lakefile.count(f'"{pinned_rev}"') != 1:
        raise MutationError("pin control B target revision not found exactly once in the lakefile")
    if texts[MANIFEST_REL].count(pinned_rev) != 2:
        raise MutationError("pin control B target revision not found twice in the manifest")
    texts[LAKEFILE_REL] = lakefile.replace(f'"{pinned_rev}"', f'"{ALL_ZEROS_REV}"')
    texts[MANIFEST_REL] = texts[MANIFEST_REL].replace(pinned_rev, ALL_ZEROS_REV)
    return texts


def _pin_mut_no_readloop(
    texts: dict[str, str], pinned_rev: str, corpus: Path
) -> dict[str, str]:
    target = "`Growable.ReadLoop, "
    if texts[LAKEFILE_REL].count(target) != 1:
        raise MutationError("pin control C target `Growable.ReadLoop` root not found exactly once")
    texts[LAKEFILE_REL] = texts[LAKEFILE_REL].replace(target, "")
    return texts


PIN_CONTROLS: tuple[PinControl, ...] = (
    PinControl(
        "A",
        "manifest revision drifted away from the lakefile pin",
        "PIN-BUMP",
        ("pin mismatch", "lakefile pins"),
        _pin_mut_manifest_drift,
    ),
    PinControl(
        "B",
        "pin replaced by an all-zeros unreachable commit",
        "PIN-REACH",
        (ALL_ZEROS_REV,),
        _pin_mut_unreachable_zeros,
    ),
    PinControl(
        "C",
        "Growable.ReadLoop removed from the bridge roots",
        "PIN-ROOTS",
        ("Growable.ReadLoop",),
        _pin_mut_no_readloop,
    ),
)


def _run_pin_stage(fixture_root: Path, corpus: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(HERE),
            "--stage",
            "pin",
            "--root",
            str(fixture_root),
            "--corpus",
            str(corpus),
        ],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _write_pin_files(root: Path, texts: dict[str, str]) -> None:
    for rel in (LAKEFILE_REL, MANIFEST_REL):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texts[rel], encoding="utf-8")


def pin_controls(root: Path, corpus: Path) -> int:
    """Baseline/red/restore on scratch copies of the real pin files; the
    observed exit-code triple per control must be 0/1/0 and each rejection
    attributable. The working tree and the corpus checkout are untouched."""
    _preflight_corpus(corpus)
    pristine = {
        LAKEFILE_REL: (root / LAKEFILE_REL).read_text(encoding="utf-8", errors="replace"),
        MANIFEST_REL: (root / MANIFEST_REL).read_text(encoding="utf-8", errors="replace"),
    }
    if validate_pin(root, corpus):
        print(
            "PIN CONTROLS FAIL: the real pin files are not green; a red baseline "
            "cannot establish controls",
            file=sys.stderr,
        )
        return EXIT_REJECT
    pinned_rev = _pinned_lakefile_rev(_norm(pristine[LAKEFILE_REL]))
    if pinned_rev is None or not PIN_HEX_RE.match(pinned_rev):
        print("FAIL: cannot extract the pinned revision from the real lakefile", file=sys.stderr)
        return EXIT_USAGE

    tmp_parent: str | None = None
    preferred = Path("/tmp/opencode")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        tmp_parent = str(preferred)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="kepler-w4-t0-", dir=tmp_parent) as tmp:
        fixture = Path(tmp) / "fixture"
        rows: list[tuple[str, str, str, bool]] = []
        for control in PIN_CONTROLS:
            texts = dict(pristine)
            try:
                texts = control.mutate(texts, pinned_rev, corpus)
            except MutationError as exc:
                print(
                    f"FAIL: pin control {control.name} mutation infrastructure error: {exc}",
                    file=sys.stderr,
                )
                return EXIT_USAGE

            _write_pin_files(fixture, pristine)
            base_code, base_out = _run_pin_stage(fixture, corpus)
            base_ok = base_code == EXIT_OK and "PIN CHECK CLEAN" in base_out

            _write_pin_files(fixture, texts)
            mut_code, mut_out = _run_pin_stage(fixture, corpus)
            attributable = f"RULE {control.rule} " in mut_out and all(
                token in mut_out for token in control.tokens
            )
            red_ok = mut_code == EXIT_REJECT and attributable
            first_rule = _first_rule_line(mut_out) if mut_out.strip() else "(no output)"

            _write_pin_files(fixture, pristine)
            rest_code, rest_out = _run_pin_stage(fixture, corpus)
            rest_ok = rest_code == EXIT_OK and "PIN CHECK CLEAN" in rest_out

            triple_ok = base_ok and red_ok and rest_ok
            observed = f"{base_code}/{mut_code}/{rest_code} :: {first_rule}"
            rows.append(
                (
                    f"pin control {control.name} ({control.defect})",
                    "0/1/0",
                    observed,
                    triple_ok,
                )
            )
            if not triple_ok:
                if not base_ok:
                    failures.append(f"{control.name}: baseline fixture is not green")
                if not red_ok:
                    reason = (
                        "survived"
                        if mut_code == EXIT_OK
                        else (
                            f"rejected without attributable {control.rule} diagnostic"
                            if mut_code == EXIT_REJECT
                            else f"infrastructure exit {mut_code}"
                        )
                    )
                    failures.append(f"{control.name}: {reason}")
                if not rest_ok:
                    failures.append(f"{control.name}: restored fixture is not green")

    width = max(len(row[0]) for row in rows)
    print(f"{'PIN CONTROL MATRIX':<{width}}  EXPECTED  OBSERVED")
    print("-" * (width + 60))
    for name, expected, observed, ok in rows:
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<{width}}  {expected:<9} {observed:<9} {verdict}")
    if failures:
        print(f"PIN CONTROLS FAIL: {len(failures)} control(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_REJECT
    print("PIN CONTROLS PASS: baseline/red/restore = 0/1/0 for 3/3 controls")
    return EXIT_OK


# ---------------------------------------------------------------------------
# --stage bridge (APEIRON-KEPLER-W5-T0): certificate module, audit, roots
# ---------------------------------------------------------------------------

CERTIFICATE_REL = "proof/corpus/Kepler/Certificate.lean"
AXIOMAUDIT_REL = "proof/corpus/AxiomAudit.lean"
CANONICAL_ENCLOSURE_MODULE = "Apeiron.Physics.Kepler.Enclosure"
CANONICAL_ENCLOSURE_THEOREM = "absolute_defect_enclosure"
CANONICAL_ANOMALY_DEF = "eccentricAnomaly"

CERT_DENOTE_RE = re.compile(r"⟦[^⟧]{1,40}⟧")
CERT_CONCLUSION_RE = re.compile(
    r"d\s*/\s*\(\s*1\s*-\s*(?:\|e\||⟦[^⟧]{1,6}⟧|e)\s*\)"
)
CERT_CANON_IMPORT_RE = re.compile(r"import\s+Apeiron\.Physics\.Kepler\.")
CERT_SECOND_AUTHORITY_RE = re.compile(r"\|sin[^|\n]{0,60}≤|abs_sin_sub_sin_le")
CERT_SQUARED_RE = re.compile(
    r"(sin|residual|defect)[^\n]{0,80}\^2|\^2[^\n]{0,80}(residual|defect)",
    re.IGNORECASE,
)
CERT_INWARD_RE = re.compile(
    r"sin[^\n]{0,120}(?:≥|>=)[^\n]{0,40}(?:⟦d⟧|\bd\b)"
    r"|(?:⟦d⟧|\bd\b)[^\n]{0,40}(?:≤|<=)[^\n]{0,120}sin"
)
CERT_RUNTIME_CONSTRUCT_RE = re.compile(
    r"(rust|runtime)[^\n]{0,60}(construct|produc)"
    r"|(construct|produc)[^\n]{0,60}(rust|runtime)",
    re.IGNORECASE,
)
CERT_DECL_RE = re.compile(
    r"^\s*(private\s+)?(?:noncomputable\s+)?(?:theorem|def)\s+([A-Za-z_][A-Za-z0-9_'?]*)"
)
BRIDGE_KEPLER_ROOT_RE = re.compile(r"`Kepler(?:\.[A-Za-z0-9_.']+)?")

BRIDGE_STAGE_RULES = (
    ("BRIDGE", "conditional certificate interpretation",
     "supplied sound outward exact-real residual bound implies the d/(1-|e|) "
     "enclosure, rooted in Apeiron.Physics.Kepler.absolute_defect_enclosure; "
     "no squared substitution, inward read, runtime-construction or Aeneas "
     "f64 claim"),
    ("BRIDGE-AUDIT", "headline registration",
     "every public certificate theorem has a #audit line in AxiomAudit.lean"),
    ("BRIDGE-ROOT", "Kepler root entry",
     "the bridge roots carry a Kepler module entry (Growable.ReadLoop stays)"),
)


def _has_outward_premise(lines: list[str]) -> bool:
    for line in lines:
        if "sin" in line and "≤" in line and re.search(r"⟦d⟧|\bd\b", line):
            return True
    return False


def _public_headlines(cert_lines: list[str]) -> list[str]:
    headlines: list[str] = []
    for line in cert_lines:
        match = CERT_DECL_RE.match(line)
        if match and not match.group(1):
            headlines.append(match.group(2))
    return headlines


def validate_bridge(root: Path, corpus: Path) -> list[Diagnostic]:
    """Every diagnostic for the bridge stage; an empty result accepts.

    Delegates the pin to ``validate_pin`` (a broken pin fails the bridge), then
    checks the W5-T1 deliverables: the certificate module's conditional
    interpretation and its forbidden-substance rules, the AxiomAudit
    registration of every public headline, and the Kepler root entry. On the
    pre-W5 tree this returns MISSING-deliverable diagnostics (expected red).
    """
    out: list[Diagnostic] = list(validate_pin(root, corpus))
    cert_path = root / CERTIFICATE_REL
    audit_path = root / AXIOMAUDIT_REL
    cert_exists = cert_path.is_file()
    headlines: list[str] = []
    if not cert_exists:
        out.append(
            Diagnostic(
                "BRIDGE",
                CERTIFICATE_REL,
                0,
                "MISSING deliverable (W5-T1): certificate-interpretation module "
                "proof/corpus/Kepler/Certificate.lean does not exist",
            )
        )
    else:
        cert_lines = _lines(cert_path.read_text(encoding="utf-8", errors="replace"))
        if not any(CERT_DENOTE_RE.search(line) for line in cert_lines):
            out.append(
                Diagnostic(
                    "BRIDGE",
                    CERTIFICATE_REL,
                    0,
                    "no exact-real denotation relation ⟦⟧ for the binary64 "
                    "input/output bit patterns (signed zero, finite-only admission)",
                )
            )
        if not _has_outward_premise(cert_lines):
            out.append(
                Diagnostic(
                    "BRIDGE",
                    CERTIFICATE_REL,
                    0,
                    "conditional outward bound premise is missing: the module must "
                    "take a supplied SOUND bound |x - e * Real.sin x - m| ≤ d on the "
                    "exact-real residual as a hypothesis",
                )
            )
        if not any(CERT_CONCLUSION_RE.search(line) for line in cert_lines):
            out.append(
                Diagnostic(
                    "BRIDGE",
                    CERTIFICATE_REL,
                    0,
                    "the d/(1-|e|) enclosure conclusion of the certificate relation "
                    "is missing",
                )
            )
        uses_theorem = any(CANONICAL_ENCLOSURE_THEOREM in line for line in cert_lines)
        has_import = any(CERT_CANON_IMPORT_RE.search(line) for line in cert_lines)
        if not (uses_theorem and has_import):
            detail = []
            if not has_import:
                detail.append(f"no `import Apeiron.Physics.Kepler.*` line")
            if not uses_theorem:
                detail.append(
                    f"the canonical theorem `{CANONICAL_ENCLOSURE_THEOREM}` is not "
                    "applied"
                )
            out.append(
                Diagnostic(
                    "BRIDGE",
                    CERTIFICATE_REL,
                    0,
                    "not rooted in the canonical enclosure theorem "
                    f"({CANONICAL_ENCLOSURE_MODULE}.{CANONICAL_ENCLOSURE_THEOREM}): "
                    + "; ".join(detail)
                    + "; no second authority restating the bound is accepted",
                )
            )
        for number, line in enumerate(cert_lines, 1):
            if CERT_SECOND_AUTHORITY_RE.search(line):
                out.append(
                    Diagnostic(
                        "BRIDGE",
                        CERTIFICATE_REL,
                        number,
                        "restates the canonical Lipschitz proof step "
                        "(`|sin x − sin y| ≤ |x−y|` / abs_sin_sub_sin_le) instead of "
                        "applying the canonical enclosure theorem",
                    )
                )
            if CERT_SQUARED_RE.search(line):
                out.append(
                    Diagnostic(
                        "BRIDGE",
                        CERTIFICATE_REL,
                        number,
                        "squared residual substituted for the absolute residual: the "
                        "defect bound premises |F| ≤ d, not F^2",
                    )
                )
            if CERT_INWARD_RE.search(line):
                out.append(
                    Diagnostic(
                        "BRIDGE",
                        CERTIFICATE_REL,
                        number,
                        "inward/understated defect bound: the premise must bound the "
                        "exact-real residual from ABOVE (|F| ≤ d), not from below",
                    )
                )
            if CERT_RUNTIME_CONSTRUCT_RE.search(line) and not NEGATION_RE.search(line):
                out.append(
                    Diagnostic(
                        "BRIDGE",
                        CERTIFICATE_REL,
                        number,
                        "claims the runtime constructs the bound: a conditional "
                        "theorem on a supplied sound bound does not establish that "
                        "Rust constructed it",
                    )
                )
            if (
                AENEAS_RE.search(line)
                and F64_RE.search(line)
                and REFINE_WORD_RE.search(line)
                and not NEGATION_RE.search(line)
            ):
                out.append(
                    Diagnostic(
                        "BRIDGE",
                        CERTIFICATE_REL,
                        number,
                        "Aeneas f64 refinement claim in the bridge module: no source "
                        "refinement exists for this path",
                    )
                )
        headlines = _public_headlines(cert_lines)
        if not headlines:
            out.append(
                Diagnostic(
                    "BRIDGE",
                    CERTIFICATE_REL,
                    0,
                    "MISSING deliverable (W5-T1): the certificate module declares no "
                    "public theorem headline",
                )
            )

    if not audit_path.is_file():
        out.append(
            Diagnostic(
                "BRIDGE-AUDIT",
                AXIOMAUDIT_REL,
                0,
                f"MISSING deliverable: {AXIOMAUDIT_REL} does not exist",
            )
        )
    elif headlines:
        audit_text = audit_path.read_text(encoding="utf-8", errors="replace")
        for name in headlines:
            if name not in audit_text:
                out.append(
                    Diagnostic(
                        "BRIDGE-AUDIT",
                        AXIOMAUDIT_REL,
                        0,
                        f"MISSING registration: AxiomAudit.lean has no `#audit` line "
                        f"for certificate headline `{name}`",
                    )
                )
    elif cert_exists:
        pass  # already diagnosed: the module declares no public headline
    else:
        out.append(
            Diagnostic(
                "BRIDGE-AUDIT",
                AXIOMAUDIT_REL,
                0,
                "MISSING deliverable: no certificate headline registered in "
                f"AxiomAudit.lean ({CERTIFICATE_REL} is absent)",
            )
        )

    lakefile_path = root / LAKEFILE_REL
    if lakefile_path.is_file():
        lakefile_lines = _lines(lakefile_path.read_text(encoding="utf-8", errors="replace"))
        if not any(BRIDGE_KEPLER_ROOT_RE.search(line) for line in lakefile_lines):
            out.append(
                Diagnostic(
                    "BRIDGE-ROOT",
                    LAKEFILE_REL,
                    0,
                    "MISSING deliverable (W5-T1): no Kepler root entry in the bridge "
                    "roots (register after `Growable.ReadLoop`)",
                )
            )
    return out


def cmd_bridge_check(root: Path, corpus: Path) -> int:
    diagnostics = validate_bridge(root, corpus)
    if not diagnostics:
        print(
            "BRIDGE CHECK CLEAN: pin green; certificate module carries the "
            "conditional outward-bound interpretation rooted in "
            f"{CANONICAL_ENCLOSURE_MODULE}.{CANONICAL_ENCLOSURE_THEOREM}; every "
            "public headline registered in AxiomAudit.lean; Kepler root present"
        )
        return EXIT_OK
    for diagnostic in diagnostics:
        print(diagnostic.render())
    missing = [d for d in diagnostics if d.message.startswith("MISSING")]
    if missing:
        print("MISSING deliverables (W5-T1):")
        for diagnostic in missing:
            print(f"  - {diagnostic.path}: {diagnostic.message}")
    print(f"BRIDGE CHECK FAIL: {len(diagnostics)} violation(s)")
    return EXIT_REJECT


# --- bridge controls --------------------------------------------------------

# The fixture owns a headline name that cannot occur in the real tree, and the
# scratch lakefile copy carries exactly one Kepler root (the fixture's own, the
# real entries stripped). Otherwise the landed W5-T1 files mask the mutations:
# a control whose mutant survives because the real artifact satisfies the rule
# is not a control.
FIXTURE_HEADLINE = "kepler_abs_defect_enclosure_bridge_fixture"
REAL_KEPLER_TOKEN_RE = re.compile(r"`Kepler(?:\.[A-Za-z0-9_.']+)?,?\s*")

CERTIFICATE_TEMPLATE = f"""import Aeneas
import {CANONICAL_ENCLOSURE_MODULE}

/-!
Bridge certificate interpretation for the canonical Kepler solve.

Reading of the binary64 bit patterns as exact rationals:
`m = ⟦mean_anomaly⟧`, `e = ⟦eccentricity⟧`, `E = ⟦solution.anomaly⟧`,
`d = ⟦defect bound⟧` (signed zero admitted, finite-only admission).

The headline is CONDITIONAL on a supplied SOUND outward bound on the
exact-real residual of the denoted inputs: it does not establish that Rust
constructed the bound, and no Aeneas `f64` refinement is claimed here.
-/

open Apeiron.Physics.Kepler

/-- Conditional enclosure for the denoted values: a supplied sound outward
defect bound on the exact-real residual implies the absolute-error enclosure. -/
theorem {FIXTURE_HEADLINE}
    (m e x d : ℝ) (he : |e| < 1) (hd : 0 ≤ d)
    (hbound : |x - e * Real.sin x - m| ≤ d) :
    |x - {CANONICAL_ANOMALY_DEF} m e he| ≤ d / (1 - |e|) :=
  {CANONICAL_ENCLOSURE_THEOREM} m e x d he hd hbound
"""

AUDIT_APPEND = (
    "\n-- Kepler certificate headline registration (bridge stage fixture)\n"
    f"#audit Kepler.Certificate.{FIXTURE_HEADLINE}\n"
)
KEPLER_ROOT_TOKEN = "`Kepler.Certificate, "


def _bridge_pristine(root: Path) -> dict[str, str]:
    lakefile = (root / LAKEFILE_REL).read_text(encoding="utf-8", errors="replace")
    if lakefile.count("`Growable.ReadLoop, ") != 1:
        raise MutationError("`Growable.ReadLoop, ` root not found exactly once")
    audit = (root / AXIOMAUDIT_REL).read_text(encoding="utf-8", errors="replace")
    if FIXTURE_HEADLINE in audit:
        raise MutationError(
            "fixture headline collides with a name in the real AxiomAudit.lean; "
            "the fixture must own a registration the real tree cannot supply"
        )
    # Strip the real tree's Kepler root entries so the fixture lakefile owns
    # exactly one Kepler root: removing it must leave none.
    fixture_lakefile = REAL_KEPLER_TOKEN_RE.sub("", lakefile)
    return {
        LAKEFILE_REL: fixture_lakefile.replace(
            "`Growable.ReadLoop, ", "`Growable.ReadLoop, " + KEPLER_ROOT_TOKEN
        ),
        MANIFEST_REL: (root / MANIFEST_REL).read_text(encoding="utf-8", errors="replace"),
        CERTIFICATE_REL: CERTIFICATE_TEMPLATE,
        AXIOMAUDIT_REL: audit + AUDIT_APPEND,
    }


def _brg_mut_inward(texts: dict[str, str], pristine: dict[str, str]) -> dict[str, str]:
    target = "|x - e * Real.sin x - m| ≤ d"
    if texts[CERTIFICATE_REL].count(target) != 1:
        raise MutationError("bridge control A premise not found exactly once")
    texts[CERTIFICATE_REL] = texts[CERTIFICATE_REL].replace(
        target, "|x - e * Real.sin x - m| ≥ d"
    )
    return texts


def _brg_mut_no_audit(texts: dict[str, str], pristine: dict[str, str]) -> dict[str, str]:
    audit = texts[AXIOMAUDIT_REL]
    if audit.count(AUDIT_APPEND) != 1:
        raise MutationError("bridge control B registration block not found")
    texts[AXIOMAUDIT_REL] = audit.replace(AUDIT_APPEND, "\n")
    return texts


def _brg_mut_no_root(texts: dict[str, str], pristine: dict[str, str]) -> dict[str, str]:
    if texts[LAKEFILE_REL].count(KEPLER_ROOT_TOKEN) != 1:
        raise MutationError("bridge control B2 Kepler root token not found")
    texts[LAKEFILE_REL] = texts[LAKEFILE_REL].replace(KEPLER_ROOT_TOKEN, "")
    return texts


def _brg_mut_aeneas(texts: dict[str, str], pristine: dict[str, str]) -> dict[str, str]:
    texts[CERTIFICATE_REL] += (
        "\n-- The Aeneas extraction refines the Rust `f64` solver body to this "
        "certificate theorem.\n"
    )
    return texts


def _brg_mut_squared(texts: dict[str, str], pristine: dict[str, str]) -> dict[str, str]:
    target = "(hbound : |x - e * Real.sin x - m| ≤ d)"
    if texts[CERTIFICATE_REL].count(target) != 1:
        raise MutationError("bridge control D premise not found exactly once")
    texts[CERTIFICATE_REL] = texts[CERTIFICATE_REL].replace(
        target, "(hbound : (x - e * Real.sin x - m)^2 ≤ d)"
    )
    return texts


def _brg_mut_runtime(texts: dict[str, str], pristine: dict[str, str]) -> dict[str, str]:
    texts[CERTIFICATE_REL] += (
        "\n-- This theorem establishes that the Rust runtime constructs the "
        "outward bound d from the returned bit patterns.\n"
    )
    return texts


@dataclass(frozen=True)
class BridgeControl:
    name: str
    defect: str
    rule: str
    tokens: tuple[str, ...]
    mutate: "Callable[[dict[str, str], dict[str, str]], dict[str, str]]"


BRIDGE_CONTROLS: tuple[BridgeControl, ...] = (
    BridgeControl(
        "A",
        "inward/understated defect bound accepted",
        "BRIDGE",
        ("inward",),
        _brg_mut_inward,
    ),
    BridgeControl(
        "B",
        "AxiomAudit registration removed",
        "BRIDGE-AUDIT",
        (FIXTURE_HEADLINE,),
        _brg_mut_no_audit,
    ),
    BridgeControl(
        "B2",
        "Kepler root entry removed",
        "BRIDGE-ROOT",
        ("Kepler root",),
        _brg_mut_no_root,
    ),
    BridgeControl(
        "C",
        "Aeneas f64 refinement claim added",
        "BRIDGE",
        ("Aeneas",),
        _brg_mut_aeneas,
    ),
    BridgeControl(
        "D",
        "squared residual substituted for the absolute residual",
        "BRIDGE",
        ("squared",),
        _brg_mut_squared,
    ),
    BridgeControl(
        "E",
        "runtime-constructs-the-bound claim added",
        "BRIDGE",
        ("construct",),
        _brg_mut_runtime,
    ),
)


def _write_bridge_files(root: Path, texts: dict[str, str]) -> None:
    for rel in (LAKEFILE_REL, MANIFEST_REL, CERTIFICATE_REL, AXIOMAUDIT_REL):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texts[rel], encoding="utf-8")


def _run_bridge_stage(fixture_root: Path, corpus: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(HERE),
            "--stage",
            "bridge",
            "--root",
            str(fixture_root),
            "--corpus",
            str(corpus),
        ],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout + proc.stderr


def bridge_controls(root: Path, corpus: Path) -> int:
    """Baseline/red/restore for the bridge stage on a scratch fixture built
    from the real pin files plus a synthetic conformant certificate module;
    the working tree is untouched. The real tree is expected red until W5-T1
    lands, so the green baseline comes from the fixture, never from the tree."""
    _preflight_corpus(corpus)
    try:
        pristine = _bridge_pristine(root)
    except OSError as exc:
        print(f"FAIL: bridge fixture inputs unreadable: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if validate_pin(root, corpus):
        print(
            "BRIDGE CONTROLS FAIL: the real pin files are not green; the bridge "
            "campaign requires a green pin baseline",
            file=sys.stderr,
        )
        return EXIT_REJECT

    tmp_parent: str | None = None
    preferred = Path("/tmp/opencode")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        tmp_parent = str(preferred)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="kepler-w5-t0-", dir=tmp_parent) as tmp:
        fixture = Path(tmp) / "fixture"
        rows: list[tuple[str, str, str, bool]] = []
        for control in BRIDGE_CONTROLS:
            texts = dict(pristine)
            try:
                texts = control.mutate(texts, pristine)
            except MutationError as exc:
                print(
                    f"FAIL: bridge control {control.name} mutation infrastructure "
                    f"error: {exc}",
                    file=sys.stderr,
                )
                return EXIT_USAGE

            _write_bridge_files(fixture, pristine)
            base_code, base_out = _run_bridge_stage(fixture, corpus)
            base_ok = base_code == EXIT_OK and "BRIDGE CHECK CLEAN" in base_out

            _write_bridge_files(fixture, texts)
            mut_code, mut_out = _run_bridge_stage(fixture, corpus)
            attributable = f"RULE {control.rule} " in mut_out and all(
                token in mut_out for token in control.tokens
            )
            red_ok = mut_code == EXIT_REJECT and attributable
            first_rule = _first_rule_line(mut_out) if mut_out.strip() else "(no output)"

            _write_bridge_files(fixture, pristine)
            rest_code, rest_out = _run_bridge_stage(fixture, corpus)
            rest_ok = rest_code == EXIT_OK and "BRIDGE CHECK CLEAN" in rest_out

            triple_ok = base_ok and red_ok and rest_ok
            observed = f"{base_code}/{mut_code}/{rest_code} :: {first_rule}"
            rows.append(
                (
                    f"bridge control {control.name} ({control.defect})",
                    "0/1/0",
                    observed,
                    triple_ok,
                )
            )
            if not triple_ok:
                if not base_ok:
                    failures.append(f"{control.name}: fixture baseline is not green")
                if not red_ok:
                    reason = (
                        "survived"
                        if mut_code == EXIT_OK
                        else (
                            f"rejected without attributable {control.rule} diagnostic"
                            if mut_code == EXIT_REJECT
                            else f"infrastructure exit {mut_code}"
                        )
                    )
                    failures.append(f"{control.name}: {reason}")
                if not rest_ok:
                    failures.append(f"{control.name}: restored fixture is not green")

    width = max(len(row[0]) for row in rows)
    print(f"{'BRIDGE CONTROL MATRIX':<{width}}  EXPECTED  OBSERVED")
    print("-" * (width + 60))
    for name, expected, observed, ok in rows:
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<{width}}  {expected:<9} {observed:<9} {verdict}")
    if failures:
        print(f"BRIDGE CONTROLS FAIL: {len(failures)} control(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_REJECT
    print(
        f"BRIDGE CONTROLS PASS: baseline/red/restore = 0/1/0 for "
        f"{len(BRIDGE_CONTROLS)}/{len(BRIDGE_CONTROLS)} controls"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# --stage runtime (APEIRON-KEPLER-W6-T0): executing kernel deliverables
# ---------------------------------------------------------------------------

KEPLER_SRC_REL = "src/kepler.rs"
LIB_REL = "src/lib.rs"
KEPLER_TEST_REL = "tests/kepler.rs"

# The public API is matched on whitespace-collapsed source, so a multi-line
# signature is accepted while the contract naming (Sigma / f64 / KeplerSolution
# / KeplerError) is required:
#   pub fn solve_kepler(face: Sigma, mean_anomaly: f64, eccentricity: f64)
#       -> Result<KeplerSolution, KeplerError>
RUNTIME_API_RE = re.compile(
    r"pub\s+fn\s+solve_kepler\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*Sigma\s*,\s*"
    r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*f64\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*f64\s*,?\s*\)\s*"
    r"->\s*Result\s*<\s*KeplerSolution\s*,\s*KeplerError\s*>"
)
RUNTIME_ERROR_ENUM_RE = re.compile(r"pub\s+enum\s+KeplerError\b")
RUNTIME_SOLUTION_DEF_RE = re.compile(r"pub\s+(?:struct|enum)\s+KeplerSolution\b")
RUNTIME_DEFAULT_FACE_RE = re.compile(r"Sigma\s*::\s*default\s*\(\s*\)")
RUNTIME_ELLIPTIC_RE = re.compile(r"Sigma\s*::\s*Elliptic")
RUNTIME_CLAMP_RE = re.compile(r"\.clamp\s*\(")
RUNTIME_DEFAULT_SUBSTITUTE_RE = re.compile(r"unwrap_or_default\s*\(")
RUNTIME_PI_RE = re.compile(r"\bPI\b")
RUNTIME_E95_RE = re.compile(r"0\.95(?:_f64)?")
RUNTIME_SCHEDULE_LO_RE = re.compile(r"\b0\.15\b")
RUNTIME_SCHEDULE_MID_RE = re.compile(r"\b0\.65\b")
# An escalation of the rounded residual: `residual.abs() + <named margin>`,
# `…next_up()` or a `* (1.0 + …)` inflation. A line that accepts the bare
# `residual.abs()` as the certificate does not match.
RUNTIME_ESCALATE_RE = re.compile(
    r"(?i)(?:defect|residual|bound|error)[^\n]{0,60}"
    r"(?:next_up|\+\s*[A-Za-z_][A-Za-z0-9_]*|\*\s*\(\s*1\.0\s*\+)"
)
# A lower bound on the defect is not the certificate: the premise bounds |F|
# from above.
RUNTIME_INWARD_RE = re.compile(
    r"(?i)(?:residual|defect)[^\n]{0,60}(?:≥|>=)"
    r"|(?:≥|>=)[^\n]{0,40}(?:residual|defect)"
)
RUNTIME_ZERO_E_RE = re.compile(
    r"(?i)(?:eccentricity|\be\b)\s*==\s*-?0(?:\.0)?(?:_f64)?"
)
RUNTIME_ZERO_ROUNDS_RE = re.compile(
    r"(?i)\b(?:rounds|iterations|corrections|count)\s*:\s*0\b"
)
RUNTIME_ANOMALY_FIELD_RE = re.compile(r"(?i)\banomaly\s*:")
RUNTIME_EXPORT_RE = re.compile(r"(?m)^\s*pub\s+mod\s+kepler\s*;")

RUNTIME_BOUNDARY_RE = re.compile(r"(?i)(boundary|endpoint|refus|reject)")
RUNTIME_ORACLE_RE = re.compile(r"(?i)(oracle|golden|reference)")
RUNTIME_NAIVE_RE = re.compile(r"(?i)(naive|kahan|accumulat|drift)")
RUNTIME_BITEXACT_RE = re.compile(r"(?i)(to_bits|is_sign_negative|-0\.0|bit[-_ ]?exact)")

RUNTIME_STAGE_RULES = (
    ("RUNTIME-API", "public solve_kepler API",
     "src/kepler.rs declares the frozen signature and the KeplerSolution / "
     "KeplerError types"),
    ("RUNTIME-FACE", "explicit elliptic face",
     "only Sigma::Elliptic executes; Sigma::default() is never a solver face"),
    ("RUNTIME-REFUSAL", "six typed refusals, no substitution",
     "all six KeplerError variants declared; no clamp/default substitution"),
    ("RUNTIME-DOMAIN", "normalized service domain",
     "|m| <= PI over the exact reals and the 0.95_f64 service endpoint"),
    ("RUNTIME-SCHEDULE", "fixed correction schedule",
     "the 0.15/0.65/0.95 schedule thresholds are present"),
    ("RUNTIME-CERT", "outward certificate or CertificateFailure",
     "the certificate escalates the rounded residual outward (never accepts it "
     "bare, never bounds it inward) and refuses CertificateFailure otherwise"),
    ("RUNTIME-MONAD", "e == 0 bit-exact monad",
     "the circular branch returns the input anomaly with zero rounds"),
    ("RUNTIME-NONCLAIM", "no Aeneas f64 refinement claim",
     "no source refinement is claimed for the executing kernel"),
    ("RUNTIME-EXPORT", "lib.rs module export",
     "src/lib.rs declares `pub mod kepler;`"),
    ("RUNTIME-TEST", "kernel integration tests",
     "tests/kepler.rs carries boundary, oracle, bit-exact and "
     "naive-accumulation cases"),
)


def validate_runtime(root: Path) -> list[Diagnostic]:
    """Every diagnostic for the W6 runtime stage; an empty result accepts.

    Structural/token validation of the executing-kernel deliverables against a
    tree (real or fixture). It does not compile Rust and claims no soundness:
    the semantic obligations are `tested` by ``tests/kepler.rs`` at their
    sampled scope, and the pin/bridge remain separately gated.
    """
    out: list[Diagnostic] = []
    src_path = root / KEPLER_SRC_REL
    src_exists = src_path.is_file()
    if not src_exists:
        out.append(
            Diagnostic(
                "RUNTIME-API",
                KEPLER_SRC_REL,
                0,
                "MISSING deliverable (W6-T1): executing kernel src/kepler.rs "
                "does not exist",
            )
        )
    else:
        src_text = src_path.read_text(encoding="utf-8", errors="replace")
        src_lines = _lines(src_text)
        code = _kernel_code_lines(src_lines)
        collapsed = re.sub(r"\s+", " ", _norm(src_text))
        if not RUNTIME_API_RE.search(collapsed):
            out.append(
                Diagnostic(
                    "RUNTIME-API",
                    KEPLER_SRC_REL,
                    0,
                    "public API is missing: expected `pub fn solve_kepler(face: "
                    "Sigma, mean_anomaly: f64, eccentricity: f64) -> "
                    "Result<KeplerSolution, KeplerError>`",
                )
            )
        if not RUNTIME_ERROR_ENUM_RE.search(collapsed):
            out.append(
                Diagnostic(
                    "RUNTIME-API",
                    KEPLER_SRC_REL,
                    0,
                    "typed refusal enum `pub enum KeplerError` is missing",
                )
            )
        if not RUNTIME_SOLUTION_DEF_RE.search(collapsed):
            out.append(
                Diagnostic(
                    "RUNTIME-API",
                    KEPLER_SRC_REL,
                    0,
                    "solution type `pub struct KeplerSolution` is missing",
                )
            )

        for number, line in code:
            if RUNTIME_DEFAULT_FACE_RE.search(line):
                out.append(
                    Diagnostic(
                        "RUNTIME-FACE",
                        KEPLER_SRC_REL,
                        number,
                        "`Sigma::default()` used as the solver face: Default is "
                        "Parabolic and MUST NOT be a solver face",
                    )
                )
        if not any(RUNTIME_ELLIPTIC_RE.search(line) for _n, line in code):
            out.append(
                Diagnostic(
                    "RUNTIME-FACE",
                    KEPLER_SRC_REL,
                    0,
                    "explicit `Sigma::Elliptic` solver face is missing from the "
                    "executing code",
                )
            )

        present = {variant for variant in REFUSAL_VARIANTS if re.search(rf"\b{variant}\b", src_text)}
        for variant in REFUSAL_VARIANTS:
            if variant not in present:
                out.append(
                    Diagnostic(
                        "RUNTIME-REFUSAL",
                        KEPLER_SRC_REL,
                        0,
                        f"typed refusal variant `{variant}` is missing from "
                        f"{KEPLER_SRC_REL}",
                    )
                )
        for number, line in code:
            if RUNTIME_CLAMP_RE.search(line):
                out.append(
                    Diagnostic(
                        "RUNTIME-REFUSAL",
                        KEPLER_SRC_REL,
                        number,
                        "clamp substitution on the refused domain: the contract "
                        "requires a typed refusal, never clamping or defaulting",
                    )
                )
            if RUNTIME_DEFAULT_SUBSTITUTE_RE.search(line):
                out.append(
                    Diagnostic(
                        "RUNTIME-REFUSAL",
                        KEPLER_SRC_REL,
                        number,
                        "`unwrap_or_default` default substitution: refused input "
                        "must not be replaced by a default candidate",
                    )
                )

        if not any(RUNTIME_PI_RE.search(line) for _n, line in code) or not any(
            RUNTIME_E95_RE.search(line) for _n, line in code
        ):
            out.append(
                Diagnostic(
                    "RUNTIME-DOMAIN",
                    KEPLER_SRC_REL,
                    0,
                    "normalized service domain is incomplete: require the exact "
                    "endpoint `PI` and the binary64 endpoint `0.95_f64`",
                )
            )
        if not (
            any(RUNTIME_SCHEDULE_LO_RE.search(line) for _n, line in code)
            and any(RUNTIME_SCHEDULE_MID_RE.search(line) for _n, line in code)
        ):
            out.append(
                Diagnostic(
                    "RUNTIME-SCHEDULE",
                    KEPLER_SRC_REL,
                    0,
                    "fixed correction schedule thresholds are missing: require the "
                    "0.15 and 0.65 schedule constants",
                )
            )

        if not re.search(r"\bCertificateFailure\b", src_text):
            out.append(
                Diagnostic(
                    "RUNTIME-CERT",
                    KEPLER_SRC_REL,
                    0,
                    "no `CertificateFailure` path: a candidate whose outward bound "
                    "cannot be constructed must be refused, never returned",
                )
            )
        if not any(RUNTIME_ESCALATE_RE.search(line) for _n, line in code):
            out.append(
                Diagnostic(
                    "RUNTIME-CERT",
                    KEPLER_SRC_REL,
                    0,
                    "no outward escalation of the exact-real residual: a bare "
                    "rounded `residual.abs()` is not a certificate; the outward "
                    "bound must add a named allowance (or `next_up`)",
                )
            )
        for number, line in code:
            if RUNTIME_INWARD_RE.search(line):
                out.append(
                    Diagnostic(
                        "RUNTIME-CERT",
                        KEPLER_SRC_REL,
                        number,
                        "inward/understated defect bound: the certificate must bound "
                        "the residual from above, never from below",
                    )
                )

        if not any(RUNTIME_ZERO_E_RE.search(line) for _n, line in code) or not any(
            RUNTIME_ZERO_ROUNDS_RE.search(line) for _n, line in code
        ):
            out.append(
                Diagnostic(
                    "RUNTIME-MONAD",
                    KEPLER_SRC_REL,
                    0,
                    "`e == 0` circular monad is missing: require an exact zero "
                    "eccentricity branch returning the input anomaly with zero rounds",
                )
            )
        if not any(RUNTIME_ANOMALY_FIELD_RE.search(line) for _n, line in code):
            out.append(
                Diagnostic(
                    "RUNTIME-MONAD",
                    KEPLER_SRC_REL,
                    0,
                    "the solution's `anomaly` field is not constructed",
                )
            )

        for number, line in enumerate(src_lines, 1):
            if (
                AENEAS_RE.search(line)
                and F64_RE.search(line)
                and REFINE_WORD_RE.search(line)
                and not NEGATION_RE.search(line)
            ):
                out.append(
                    Diagnostic(
                        "RUNTIME-NONCLAIM",
                        KEPLER_SRC_REL,
                        number,
                        "positive Aeneas f64 refinement claim: no source refinement "
                        "exists for this path",
                    )
                )

    lib_path = root / LIB_REL
    if not lib_path.is_file():
        out.append(
            Diagnostic(
                "RUNTIME-EXPORT",
                LIB_REL,
                0,
                "MISSING deliverable (W6-T1): src/lib.rs does not exist",
            )
        )
    elif not RUNTIME_EXPORT_RE.search(
        _norm(lib_path.read_text(encoding="utf-8", errors="replace"))
    ):
        out.append(
            Diagnostic(
                "RUNTIME-EXPORT",
                LIB_REL,
                0,
                "MISSING deliverable (W6-T1): src/lib.rs does not export the module "
                "(no `pub mod kepler;`)",
            )
        )

    test_path = root / KEPLER_TEST_REL
    if not test_path.is_file():
        out.append(
            Diagnostic(
                "RUNTIME-TEST",
                KEPLER_TEST_REL,
                0,
                "MISSING deliverable (W6-T1): tests/kepler.rs does not exist",
            )
        )
    else:
        test_text = test_path.read_text(encoding="utf-8", errors="replace")
        test_lines = _lines(test_text)
        for variant in REFUSAL_VARIANTS:
            if not re.search(rf"\b{variant}\b", test_text):
                out.append(
                    Diagnostic(
                        "RUNTIME-TEST",
                        KEPLER_TEST_REL,
                        0,
                        f"refusal boundary case for `{variant}` is missing from "
                        f"{KEPLER_TEST_REL}",
                    )
                )
        accepted = (
            "Ok(" in test_text or "is_ok(" in test_text or ".expect(" in test_text
        )
        refused = (
            "Err(" in test_text
            or "is_err(" in test_text
            or "unwrap_err(" in test_text
        )
        if not accepted or not refused:
            out.append(
                Diagnostic(
                    "RUNTIME-TEST",
                    KEPLER_TEST_REL,
                    0,
                    "accepted (`Ok(`/`is_ok`) and refused "
                    "(`Err(`/`is_err`) boundary cases are incomplete",
                )
            )
        if not RUNTIME_BOUNDARY_RE.search(test_text):
            out.append(
                Diagnostic(
                    "RUNTIME-TEST",
                    KEPLER_TEST_REL,
                    0,
                    "no endpoint/boundary or refusal case is named in the tests",
                )
            )
        if not RUNTIME_ORACLE_RE.search(test_text):
            out.append(
                Diagnostic(
                    "RUNTIME-TEST",
                    KEPLER_TEST_REL,
                    0,
                    "independent-oracle hook is missing: require an oracle/golden/"
                    "reference value not produced by the kernel under test",
                )
            )
        if not RUNTIME_NAIVE_RE.search(test_text):
            out.append(
                Diagnostic(
                    "RUNTIME-TEST",
                    KEPLER_TEST_REL,
                    0,
                    "naive-accumulation falsifier is missing: require a naive/"
                    "Kahan/accumulation/drift case",
                )
            )
        if not RUNTIME_BITEXACT_RE.search(test_text):
            out.append(
                Diagnostic(
                    "RUNTIME-TEST",
                    KEPLER_TEST_REL,
                    0,
                    "bit-exact `e == 0` witness is missing: require `to_bits`/"
                    "signed-zero comparison of the returned anomaly",
                )
            )
        for number, line in enumerate(test_lines, 1):
            if (
                AENEAS_RE.search(line)
                and F64_RE.search(line)
                and REFINE_WORD_RE.search(line)
                and not NEGATION_RE.search(line)
            ):
                out.append(
                    Diagnostic(
                        "RUNTIME-NONCLAIM",
                        KEPLER_TEST_REL,
                        number,
                        "positive Aeneas f64 refinement claim in the tests",
                    )
                )
    return out


def cmd_runtime_check(root: Path) -> int:
    diagnostics = validate_runtime(root)
    if not diagnostics:
        print(
            "RUNTIME CHECK CLEAN: src/kepler.rs declares the explicit "
            "Sigma::Elliptic solve_kepler API with all six typed refusals, the "
            "normalized domain and fixed schedule, an outward certificate path "
            "(no bare rounded residual, no clamp substitution, no Aeneas f64 "
            "claim); src/lib.rs exports the module; tests/kepler.rs carries the "
            "boundary, oracle, bit-exact and naive-accumulation cases"
        )
        return EXIT_OK
    for diagnostic in diagnostics:
        print(diagnostic.render())
    missing = [d for d in diagnostics if d.message.startswith("MISSING")]
    if missing:
        print("MISSING deliverables (W6-T1):")
        for diagnostic in missing:
            print(f"  - {diagnostic.path}: {diagnostic.message}")
    print(f"RUNTIME CHECK FAIL: {len(diagnostics)} violation(s)")
    return EXIT_REJECT


# --- runtime controls -------------------------------------------------------

# The fixture is fully synthetic: no real tree file is read or written, so the
# campaign establishes the baseline/red/restore triple for the W6 gate while
# the real tree is still EXPECTED RED.
RUNTIME_KEPLER_TEMPLATE = '''\
//! Canonical elliptic Kepler solve (`APEIRON-KEPLER-W6-T1` fixture).
//!
//! This module owns the normalized eccentric anomaly, the outward certified
//! absolute defect bound, the absolute-error enclosure and the correction
//! round count. `Sigma::default()` is Parabolic and is never a solver face.

use crate::cell::Sigma;

/// Typed refusals in the contract's deterministic precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeplerError {
    /// NaN or infinity input.
    NonFiniteInput,
    /// Finite e < 0 or e >= 1.
    InvalidEccentricity,
    /// Parabolic or hyperbolic face.
    UnsupportedFace,
    /// Finite m outside [-pi, pi], or e above the service endpoint.
    UnsupportedDomain,
    /// No candidate within the allocated correction count.
    IterationExhaustion,
    /// The outward certificate could not be constructed.
    CertificateFailure,
}

/// The certified solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerSolution {
    /// The face the host passed.
    pub face: Sigma,
    /// The normalized eccentric anomaly.
    pub anomaly: f64,
    /// An outward absolute defect upper bound.
    pub defect_bound: f64,
    /// An outward absolute-error enclosure.
    pub error_bound: f64,
    /// Complete candidate corrections.
    pub rounds: u8,
}

const SERVICE_ECC: f64 = 0.95_f64;
const ERR_ALLOWANCE: f64 = 4.0 * f64::EPSILON;
const SCHEDULE: [(f64, u8); 3] = [(0.15, 1), (0.65, 2), (0.95, 3)];

/// Solve the normalized Kepler equation on the elliptic face.
pub fn solve_kepler(
    face: Sigma,
    mean_anomaly: f64,
    eccentricity: f64,
) -> Result<KeplerSolution, KeplerError> {
    if !mean_anomaly.is_finite() || !eccentricity.is_finite() {
        return Err(KeplerError::NonFiniteInput);
    }
    if face != Sigma::Elliptic {
        return Err(KeplerError::UnsupportedFace);
    }
    if eccentricity < 0.0 || eccentricity >= 1.0 {
        return Err(KeplerError::InvalidEccentricity);
    }
    if !(-std::f64::consts::PI..=std::f64::consts::PI).contains(&mean_anomaly)
        || eccentricity > SERVICE_ECC
    {
        return Err(KeplerError::UnsupportedDomain);
    }
    if eccentricity == 0.0 {
        return Ok(KeplerSolution {
            face,
            anomaly: mean_anomaly,
            defect_bound: 0.0,
            error_bound: 0.0,
            rounds: 0,
        });
    }
    let rounds = correction_rounds(eccentricity);
    let anomaly = halley_series(mean_anomaly, eccentricity, rounds)?;
    let residual = anomaly - eccentricity * anomaly.sin() - mean_anomaly;
    let defect_bound = certify_outward(residual, ERR_ALLOWANCE)?;
    let error_bound = defect_bound / (1.0 - eccentricity);
    Ok(KeplerSolution { face, anomaly, defect_bound, error_bound, rounds })
}

fn correction_rounds(eccentricity: f64) -> u8 {
    for (limit, rounds) in SCHEDULE {
        if eccentricity <= limit {
            return rounds;
        }
    }
    4
}

fn halley_series(m: f64, e: f64, rounds: u8) -> Result<f64, KeplerError> {
    let mut anomaly = m + e * m.sin();
    for _ in 0..rounds {
        anomaly = anomaly;
    }
    if !anomaly.is_finite() {
        return Err(KeplerError::IterationExhaustion);
    }
    Ok(anomaly)
}

/// Outward-bound the exact-real residual; refuse `CertificateFailure` otherwise.
fn certify_outward(residual: f64, margin: f64) -> Result<f64, KeplerError> {
    if !residual.is_finite() || !margin.is_finite() || margin < 0.0 {
        return Err(KeplerError::CertificateFailure);
    }
    let defect_bound = residual.abs() + margin;
    if !defect_bound.is_finite() {
        return Err(KeplerError::CertificateFailure);
    }
    Ok(defect_bound)
}
'''

RUNTIME_LIB_TEMPLATE = '''\
//! Fixture crate root for the runtime-stage control campaign.

pub mod cell;
pub mod kepler;
'''

RUNTIME_TEST_TEMPLATE = '''\
//! Accepted/refused boundary cases, the independent oracle hook, the bit-exact
//! `e == 0` witness and the naive-accumulation falsifier (`APEIRON-KEPLER-W6-T1`
//! fixture).

use apeiron::cell::Sigma;
use apeiron::kepler::{solve_kepler, KeplerError};

/// Independent oracle hook: high-precision golden values, never the kernel.
fn oracle_anomaly(m: f64, _e: f64) -> f64 {
    m
}

#[test]
fn accepts_and_refuses_at_the_boundary() {
    assert!(solve_kepler(Sigma::Elliptic, 0.0, 0.0).is_ok());
    assert_eq!(
        solve_kepler(Sigma::Elliptic, f64::NAN, 0.0),
        Err(KeplerError::NonFiniteInput)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 0.0, 1.5),
        Err(KeplerError::InvalidEccentricity)
    );
    assert_eq!(
        solve_kepler(Sigma::Parabolic, 0.0, 0.0),
        Err(KeplerError::UnsupportedFace)
    );
    assert_eq!(
        solve_kepler(Sigma::Elliptic, 4.0, 0.1),
        Err(KeplerError::UnsupportedDomain)
    );
    let _ = KeplerError::IterationExhaustion;
    let _ = KeplerError::CertificateFailure;
}

#[test]
fn circular_returns_the_anomaly_bit_exactly() {
    let m = -0.0_f64;
    let solved = solve_kepler(Sigma::Elliptic, m, 0.0).expect("admitted");
    assert_eq!(solved.anomaly.to_bits(), m.to_bits());
    assert_eq!(solved.rounds, 0);
}

#[test]
fn independent_oracle_boundary() {
    let golden = oracle_anomaly(0.5, 0.2);
    assert!(golden.is_finite());
}

#[test]
fn naive_accumulation_falsifier() {
    // Naive `M += n * dt` accumulation drifts; the compensated form does not.
    let mut naive = 0.0_f64;
    for _ in 0..1000 {
        naive += 0.1;
    }
    assert!(naive.is_finite());
}
'''


def _kernel_code_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Code lines of a kernel module: comments removed, everything from the
    first ``#[cfg(test)]`` item (a test may legitimately name the prohibited
    ``Sigma::default()`` face to witness the refusal) excluded."""
    out: list[tuple[int, str]] = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#[cfg(test)]"):
            break
        if (
            not stripped
            or stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
        ):
            continue
        out.append((number, line))
    return out


@dataclass(frozen=True)
class RuntimeControl:
    name: str
    defect: str
    rule: str
    tokens: tuple[str, ...]
    mutate: Callable[[dict[str, str]], dict[str, str]]


def _rt_mut_default_face(texts: dict[str, str]) -> dict[str, str]:
    target = "if face != Sigma::Elliptic {"
    if texts[KEPLER_SRC_REL].count(target) != 1:
        raise MutationError("runtime control A target elliptic face not found once")
    texts[KEPLER_SRC_REL] = texts[KEPLER_SRC_REL].replace(
        target, "if face != Sigma::default() {"
    )
    return texts


def _rt_mut_bare_residual(texts: dict[str, str]) -> dict[str, str]:
    target = "    let defect_bound = residual.abs() + margin;"
    if texts[KEPLER_SRC_REL].count(target) != 1:
        raise MutationError("runtime control B target outward bound not found once")
    texts[KEPLER_SRC_REL] = texts[KEPLER_SRC_REL].replace(
        target, "    let defect_bound = residual.abs();"
    )
    return texts


def _rt_mut_clamp(texts: dict[str, str]) -> dict[str, str]:
    guard = (
        "    if eccentricity < 0.0 || eccentricity >= 1.0 {\n"
        "        return Err(KeplerError::InvalidEccentricity);\n"
        "    }\n"
    )
    if texts[KEPLER_SRC_REL].count(guard) != 1:
        raise MutationError("runtime control C eccentricity guard not found once")
    texts[KEPLER_SRC_REL] = texts[KEPLER_SRC_REL].replace(
        guard, "    let eccentricity = eccentricity.clamp(0.0, SERVICE_ECC);\n"
    )
    variant = "    InvalidEccentricity,\n"
    if texts[KEPLER_SRC_REL].count(variant) != 1:
        raise MutationError("runtime control C InvalidEccentricity variant not found once")
    texts[KEPLER_SRC_REL] = texts[KEPLER_SRC_REL].replace(variant, "")
    return texts


RUNTIME_CONTROLS: tuple[RuntimeControl, ...] = (
    RuntimeControl(
        "A",
        "Sigma::default() substituted for the solver face",
        "RUNTIME-FACE",
        ("Sigma::default()", "solver face"),
        _rt_mut_default_face,
    ),
    RuntimeControl(
        "B",
        "bare rounded residual accepted as the certificate",
        "RUNTIME-CERT",
        ("bare", "outward escalation"),
        _rt_mut_bare_residual,
    ),
    RuntimeControl(
        "C",
        "InvalidEccentricity refusal replaced by a clamp substitution",
        "RUNTIME-REFUSAL",
        ("InvalidEccentricity",),
        _rt_mut_clamp,
    ),
)


def _runtime_pristine() -> dict[str, str]:
    return {
        KEPLER_SRC_REL: RUNTIME_KEPLER_TEMPLATE,
        LIB_REL: RUNTIME_LIB_TEMPLATE,
        KEPLER_TEST_REL: RUNTIME_TEST_TEMPLATE,
    }


def _write_runtime_files(root: Path, texts: dict[str, str]) -> None:
    for rel in (KEPLER_SRC_REL, LIB_REL, KEPLER_TEST_REL):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texts[rel], encoding="utf-8")


def _run_runtime_stage(fixture_root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(HERE), "--stage", "runtime", "--root", str(fixture_root)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout + proc.stderr


def runtime_controls() -> int:
    """Baseline/red/restore for the runtime stage on a scratch tree built from
    synthetic conformant kernel/lib/test sources; the working tree is never
    touched and no real tree file is required, because the real tree is
    EXPECTED RED until W6-T1 lands."""
    pristine = _runtime_pristine()
    tmp_parent: str | None = None
    preferred = Path("/tmp/opencode")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        tmp_parent = str(preferred)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="kepler-w6-t0-", dir=tmp_parent) as tmp:
        fixture = Path(tmp) / "fixture"
        rows: list[tuple[str, str, str, bool]] = []
        for control in RUNTIME_CONTROLS:
            texts = dict(pristine)
            try:
                texts = control.mutate(texts)
            except MutationError as exc:
                print(
                    f"FAIL: runtime control {control.name} mutation infrastructure "
                    f"error: {exc}",
                    file=sys.stderr,
                )
                return EXIT_USAGE

            _write_runtime_files(fixture, pristine)
            base_code, base_out = _run_runtime_stage(fixture)
            base_ok = base_code == EXIT_OK and "RUNTIME CHECK CLEAN" in base_out

            _write_runtime_files(fixture, texts)
            mut_code, mut_out = _run_runtime_stage(fixture)
            attributable = f"RULE {control.rule} " in mut_out and all(
                token in mut_out for token in control.tokens
            )
            red_ok = mut_code == EXIT_REJECT and attributable
            first_rule = _first_rule_line(mut_out) if mut_out.strip() else "(no output)"

            _write_runtime_files(fixture, pristine)
            rest_code, rest_out = _run_runtime_stage(fixture)
            rest_ok = rest_code == EXIT_OK and "RUNTIME CHECK CLEAN" in rest_out

            triple_ok = base_ok and red_ok and rest_ok
            observed = f"{base_code}/{mut_code}/{rest_code} :: {first_rule}"
            rows.append(
                (
                    f"runtime control {control.name} ({control.defect})",
                    "0/1/0",
                    observed,
                    triple_ok,
                )
            )
            if not triple_ok:
                if not base_ok:
                    failures.append(f"{control.name}: fixture baseline is not green")
                if not red_ok:
                    reason = (
                        "survived"
                        if mut_code == EXIT_OK
                        else (
                            f"rejected without attributable {control.rule} diagnostic"
                            if mut_code == EXIT_REJECT
                            else f"infrastructure exit {mut_code}"
                        )
                    )
                    failures.append(f"{control.name}: {reason}")
                if not rest_ok:
                    failures.append(f"{control.name}: restored fixture is not green")

    width = max(len(row[0]) for row in rows)
    print(f"{'RUNTIME CONTROL MATRIX':<{width}}  EXPECTED  OBSERVED")
    print("-" * (width + 60))
    for name, expected, observed, ok in rows:
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<{width}}  {expected:<9} {observed:<9} {verdict}")
    if failures:
        print(f"RUNTIME CONTROLS FAIL: {len(failures)} control(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_REJECT
    print(
        f"RUNTIME CONTROLS PASS: baseline/red/restore = 0/1/0 for "
        f"{len(RUNTIME_CONTROLS)}/{len(RUNTIME_CONTROLS)} controls"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# --stage barrier (APEIRON-KEPLER-W7-T0): integration record and receipts
# ---------------------------------------------------------------------------

# The W7-T1 integration record. The barrier is a documentation/freshness gate:
# it does not compile Rust or run Lean. Freshness is established by matching
# the recorded SHA-256 values against the actual files and the recorded corpus
# revision against the actual bridge pin, so inherited historical green cannot
# be replayed after the sources move.
EVIDENCE_REL = "docs/evidence/kepler-canonical.md"
EXPECTED_CORPUS_REVISION = "880aa57e47e782ed7a9d9ee1974cfc87360b446b"
EXPECTED_APEIRON_BASE = "262bb0a94eff2cc7d5655122a954f44d74de96b6"
BARRIER_DIGEST_FILES = (KEPLER_SRC_REL, KEPLER_TEST_REL, CERTIFICATE_REL, AXIOMAUDIT_REL)
SHA256_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")

# Literal substrings the record must carry for the shared gate receipts. The
# digest table and the three control-matrix rows are checked structurally.
BARRIER_GATE_TOKENS = (
    ("corpus build", "cd proof && lake build"),
    ("corpus build jobs", "3526"),
    ("corpus lints", "check-lints.py"),
    ("corpus style", "check-style.py"),
    ("corpus imports", "check-imports.py"),
    ("corpus checker", "scripts/check-kepler.py"),
    ("corpus checker check flag", "--check"),
    ("corpus self-test flag", "--self-test"),
    ("corpus self-test count", "16/16"),
    ("bridge build", "lake build Kepler.Certificate AxiomAudit"),
    ("bridge build jobs", "3692"),
    ("closed cone", "{propext, Classical.choice, Quot.sound}"),
)
BARRIER_CLEAN_RE = re.compile(r"\bclean\b")

# Each stage's recorded control matrix must sit on a line naming that stage:
# the pin, bridge and runtime campaigns all report the observed triple 0/1/0.
BARRIER_CONTROL_STAGES = ("pin", "bridge", "runtime")
BARRIER_MATRIX_RES = {
    stage: re.compile(rf"(?i){stage}[^\n]{{0,120}}0/1/0")
    for stage in BARRIER_CONTROL_STAGES
}

# Runtime population/SLA row (K5): the recorded failure population and the open
# service row are receipts; dropping them must be rejected.
BARRIER_POPULATION_RE = re.compile(
    r"(?i)(?:12[^\n]{0,12}bands?[^\n]{0,48}1039"
    r"|1039[^\n]{0,12}rows?[^\n]{0,48}12)"
)
BARRIER_REFUSAL_RE = re.compile(r"(?i)\b2\b[^\n]{0,40}refus|refus[^\n]{0,40}\b2\b")
BARRIER_E095_RE = re.compile(r"0\.95")
BARRIER_SLA_OPEN_RE = re.compile(r"(?i)\bSLA\b[^\n]{0,80}\bOPEN\b|\bOPEN\b[^\n]{0,80}\bSLA\b")

# External-trust inventory: the arithmetic, the platform sine assumption and
# the external runtime components, plus the Aeneas f64 non-claim.
BARRIER_TRUST_RES = (
    (
        "IEEE round-to-nearest",
        re.compile(
            r"(?i)IEEE[^\n]{0,40}(?:round-to-nearest|round to nearest|RN)"
            r"|(?:round-to-nearest|round to nearest)[^\n]{0,40}IEEE"
        ),
    ),
    ("mul_add", re.compile(r"mul_add")),
    ("ASSUMPTION-SIN", re.compile(r"ASSUMPTION-SIN")),
    ("sine bound <= 1 ulp", re.compile(r"1\s*ulp")),
    ("sine domain |x| <= 8", re.compile(r"\|x\|\s*(?:≤|<=)\s*8")),
    ("libm external", re.compile(r"libm")),
    ("compiler external", re.compile(r"compiler")),
    ("hardware external", re.compile(r"hardware")),
)

# A line that promotes an abstract/Lean/bridge bound to an executable f64
# refinement. A conditional theorem on a supplied sound bound is not an f64
# source refinement, and this path has no Aeneas extraction.
BARRIER_CLAIM_CONTEXT_RE = re.compile(
    r"(?i)\b(abstract|bridge|lean|canonical|theorem|certificate|bound)\b"
)

BARRIER_STAGE_RULES = (
    ("BARRIER-DOC", "integration record present",
     "docs/evidence/kepler-canonical.md exists"),
    ("BARRIER-IDENTITY", "exact identities",
     "the resolved corpus revision equals the actual bridge pin; the Apeiron "
     "working-tree base is recorded"),
    ("BARRIER-DIGEST", "fresh source digests",
     "SHA-256 for the four sources matches the actual files; stale or missing "
     "digests are rejected"),
    ("BARRIER-GATE", "shared gate receipts",
     "corpus build/lints/style/imports, corpus checker and bridge build with "
     "the closed cone are recorded with commands and results"),
    ("BARRIER-RECEIPT", "stage and failure receipts",
     "pin/bridge/runtime control matrices (0/1/0) and the runtime refusal "
     "population/SLA row are recorded; a dropped receipt is rejected"),
    ("BARRIER-TRUST", "external-trust inventory",
     "IEEE round-to-nearest/mul_add, ASSUMPTION-SIN <= 1 ulp on |x| <= 8, "
     "libm/compiler/hardware external, no Aeneas f64 refinement claim"),
    ("BARRIER-CLAIM", "no abstract-as-f64-refinement",
     "an abstract/bridge bound reported as an executable f64 refinement is "
     "rejected"),
)


def _sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_barrier(root: Path) -> list[Diagnostic]:
    """Every diagnostic for the W7 barrier stage; an empty result accepts.

    Documentation/freshness validation of the W7-T1 integration record against
    ``root``: exact identities, digest freshness against the actual files,
    shared gate receipts, stage control matrices, the runtime population/SLA
    row and the external-trust inventory. It runs no build and claims no
    soundness. On the pre-W7-T1 tree the record is absent and this returns an
    explicit MISSING-deliverable list (expected red).
    """
    out: list[Diagnostic] = []
    evidence_path = root / EVIDENCE_REL
    if not evidence_path.is_file():
        out.append(
            Diagnostic(
                "BARRIER-DOC",
                EVIDENCE_REL,
                0,
                f"MISSING deliverable (W7-T1): integration record {EVIDENCE_REL} "
                "does not exist; the barrier stage cannot accept inherited "
                "historical green without a current record",
            )
        )
        for rel in BARRIER_DIGEST_FILES:
            if not (root / rel).is_file():
                out.append(
                    Diagnostic(
                        "BARRIER-DIGEST",
                        rel,
                        0,
                        f"MISSING deliverable: source {rel} is absent, so no "
                        "recorded digest can be verified",
                    )
                )
        return out

    lines = _lines(evidence_path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(lines)

    # (a) identities: the recorded corpus revision must be the actual pin.
    if not any(EXPECTED_CORPUS_REVISION in line for line in lines):
        out.append(
            Diagnostic(
                "BARRIER-IDENTITY",
                EVIDENCE_REL,
                0,
                f"resolved corpus revision {EXPECTED_CORPUS_REVISION} is not "
                "recorded",
            )
        )
    if not any(EXPECTED_APEIRON_BASE in line for line in lines):
        out.append(
            Diagnostic(
                "BARRIER-IDENTITY",
                EVIDENCE_REL,
                0,
                f"Apeiron working-tree identity {EXPECTED_APEIRON_BASE} is not "
                "recorded",
            )
        )
    lakefile_path = root / LAKEFILE_REL
    if lakefile_path.is_file():
        pinned = _pinned_lakefile_rev(
            _norm(lakefile_path.read_text(encoding="utf-8", errors="replace"))
        )
        if pinned != EXPECTED_CORPUS_REVISION:
            out.append(
                Diagnostic(
                    "BARRIER-IDENTITY",
                    LAKEFILE_REL,
                    0,
                    f"bridge pin records {pinned!r}, not the W7 resolved corpus "
                    f"revision {EXPECTED_CORPUS_REVISION}",
                )
            )
    manifest_path = root / MANIFEST_REL
    if manifest_path.is_file():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8", errors="replace")
            )
        except json.JSONDecodeError:
            manifest = None
        if isinstance(manifest, dict):
            for entry in manifest.get("packages", []):
                if isinstance(entry, dict) and entry.get("name") == "apeiron":
                    if entry.get("rev") != EXPECTED_CORPUS_REVISION:
                        out.append(
                            Diagnostic(
                                "BARRIER-IDENTITY",
                                MANIFEST_REL,
                                0,
                                "manifest 'apeiron' resolved rev "
                                f"{entry.get('rev')!r} is not the W7 resolved "
                                f"corpus revision {EXPECTED_CORPUS_REVISION}",
                            )
                        )

    # (a) digest freshness: each source's SHA-256 must sit on the line naming
    # that source and must equal the actual file digest.
    for rel in BARRIER_DIGEST_FILES:
        path = root / rel
        if not path.is_file():
            out.append(
                Diagnostic(
                    "BARRIER-DIGEST",
                    rel,
                    0,
                    f"MISSING deliverable: source {rel} is absent, so no recorded "
                    "digest can be verified",
                )
            )
            continue
        actual = _sha256_hex(path)
        recorded = [
            (number, match.group(0).lower())
            for number, line in enumerate(lines, 1)
            if rel in line
            for match in SHA256_RE.finditer(line)
        ]
        if not recorded:
            out.append(
                Diagnostic(
                    "BARRIER-DIGEST",
                    EVIDENCE_REL,
                    0,
                    f"source digest for {rel} is missing: no SHA-256 on a line "
                    f"naming {rel}",
                )
            )
            continue
        for number, value in recorded:
            if value != actual:
                out.append(
                    Diagnostic(
                        "BARRIER-DIGEST",
                        EVIDENCE_REL,
                        number,
                        f"stale/mismatched SHA-256 for {rel}: recorded {value}, "
                        f"actual {actual}",
                    )
                )

    # (b) shared gate receipts.
    for title, token in BARRIER_GATE_TOKENS:
        if token not in text:
            out.append(
                Diagnostic(
                    "BARRIER-GATE",
                    EVIDENCE_REL,
                    0,
                    f"shared gate receipt is missing: {title} ({token!r})",
                )
            )
    if BARRIER_CLEAN_RE.search(text) is None:
        out.append(
            Diagnostic(
                "BARRIER-GATE",
                EVIDENCE_REL,
                0,
                "no 'clean' result is recorded for the corpus lint/style/import "
                "or checker gates",
            )
        )

    # (b) stage outputs and control matrices; (c) runtime failure population.
    for stage in BARRIER_CONTROL_STAGES:
        if BARRIER_MATRIX_RES[stage].search(text) is None:
            out.append(
                Diagnostic(
                    "BARRIER-RECEIPT",
                    EVIDENCE_REL,
                    0,
                    f"recorded control receipt for stage `{stage}` (0/1/0) is "
                    "missing",
                )
            )
    if BARRIER_POPULATION_RE.search(text) is None:
        out.append(
            Diagnostic(
                "BARRIER-RECEIPT",
                EVIDENCE_REL,
                0,
                "runtime population is missing: record `12 bands x 1039 rows`",
            )
        )
    if BARRIER_REFUSAL_RE.search(text) is None:
        out.append(
            Diagnostic(
                "BARRIER-RECEIPT",
                EVIDENCE_REL,
                0,
                "runtime failure population is missing: record the `2 refusals` "
                "observed under the strict certificate policy",
            )
        )
    if BARRIER_E095_RE.search(text) is None:
        out.append(
            Diagnostic(
                "BARRIER-RECEIPT",
                EVIDENCE_REL,
                0,
                "the `e=0.95` refusal endpoint is missing from the runtime "
                "population row",
            )
        )
    if BARRIER_SLA_OPEN_RE.search(text) is None:
        out.append(
            Diagnostic(
                "BARRIER-RECEIPT",
                EVIDENCE_REL,
                0,
                "the SLA row must be recorded as OPEN: universal successful "
                "service is not claimed",
            )
        )

    # (c) external-trust inventory and the Aeneas f64 non-claim.
    for title, pattern in BARRIER_TRUST_RES:
        if pattern.search(text) is None:
            out.append(
                Diagnostic(
                    "BARRIER-TRUST",
                    EVIDENCE_REL,
                    0,
                    f"external-trust inventory entry is missing: {title}",
                )
            )
    nonclaim = any(
        AENEAS_RE.search(line)
        and F64_RE.search(line)
        and REFINE_WORD_RE.search(line)
        and NEGATION_RE.search(line)
        for line in lines
    )
    if not nonclaim:
        out.append(
            Diagnostic(
                "BARRIER-TRUST",
                EVIDENCE_REL,
                0,
                "Aeneas f64 non-claim is missing: record that no Aeneas `f64` "
                "source refinement exists for this path",
            )
        )

    # (c) an abstract/bridge bound may not be reported as an executable f64
    # refinement.
    for number, line in enumerate(lines, 1):
        if (
            F64_RE.search(line)
            and REFINE_WORD_RE.search(line)
            and BARRIER_CLAIM_CONTEXT_RE.search(line)
            and not NEGATION_RE.search(line)
        ):
            out.append(
                Diagnostic(
                    "BARRIER-CLAIM",
                    EVIDENCE_REL,
                    number,
                    "abstract/bridge bound reported as an executable f64 "
                    "refinement: the bridge theorem is conditional on a supplied "
                    "sound bound and no executable f64 refinement is "
                    f"established: {line.strip()[:120]}",
                )
            )
    return out


def cmd_barrier_check(root: Path) -> int:
    diagnostics = validate_barrier(root)
    if not diagnostics:
        print(
            "BARRIER CHECK CLEAN: W7-T1 integration record present; corpus "
            f"revision {EXPECTED_CORPUS_REVISION} matches the bridge pin; the four "
            "source digests match the actual files; shared corpus/bridge gate "
            "receipts, the pin/bridge/runtime control matrices, the runtime "
            "refusal population/SLA row and the external-trust inventory are "
            "recorded; no abstract bound is reported as an f64 refinement"
        )
        return EXIT_OK
    for diagnostic in diagnostics:
        print(diagnostic.render())
    missing = [d for d in diagnostics if d.message.startswith("MISSING")]
    if missing:
        print("MISSING deliverables (W7-T1):")
        for diagnostic in missing:
            print(f"  - {diagnostic.path}: {diagnostic.message}")
    print(f"BARRIER CHECK FAIL: {len(diagnostics)} violation(s)")
    return EXIT_REJECT


# --- barrier controls -------------------------------------------------------

# The fixture is synthetic and owns its own four digest sources plus a
# conformant record; the real pin files are copied so the identity check
# (recorded revision == actual bridge pin) is exercised against a real pin.
# The real tree is EXPECTED RED until W7-T1 lands, so the green baseline comes
# from the fixture, never from the tree.
def _barrier_evidence_template(digests: dict[str, str]) -> str:
    return (
        "# Evidence — canonical elliptic Kepler solve (W7 barrier)\n"
        "\n"
        "Integration record for `APEIRON-KEPLER-W7-T1` on the Apeiron working "
        "tree.\n"
        "\n"
        "## Identity\n"
        "\n"
        f"- Apeiron working-tree identity: base HEAD `{EXPECTED_APEIRON_BASE}`, "
        "dirty shared checkout.\n"
        f"- Resolved corpus revision: `{EXPECTED_CORPUS_REVISION}`.\n"
        "- The bridge pin `proof/corpus/lakefile.lean` and "
        "`proof/corpus/lake-manifest.json` records the same revision.\n"
        "\n"
        "## Source digests (SHA-256)\n"
        "\n"
        "| source | sha256 |\n"
        "|---|---|\n"
        + "".join(
            f"| `{rel}` | `{digests[rel]}` |\n" for rel in BARRIER_DIGEST_FILES
        )
        + "\n"
        "## Shared gate receipts\n"
        "\n"
        "| gate | command | result |\n"
        "|---|---|---|\n"
        "| corpus build | `cd proof && lake build` | 3526 jobs |\n"
        "| corpus lints | `python3 scripts/check-lints.py` | clean |\n"
        "| corpus style | `python3 scripts/check-style.py` | clean |\n"
        "| corpus imports | `python3 scripts/check-imports.py` | clean |\n"
        "| corpus checker | `python3 scripts/check-kepler.py --check` | clean |\n"
        "| corpus self-test | `python3 scripts/check-kepler.py --self-test` | "
        "16/16 |\n"
        "| bridge build | `cd proof/corpus && lake build Kepler.Certificate "
        "AxiomAudit` | 3692 jobs |\n"
        "| bridge cone | `#audit` | `{propext, Classical.choice, Quot.sound}` "
        "|\n"
        "| pin controls | `--stage pin --controls` | 0/1/0 x3 |\n"
        "| bridge controls | `--stage bridge --controls` | 0/1/0 x6 |\n"
        "| runtime controls | `--stage runtime --controls` | 0/1/0 x3 |\n"
        "\n"
        "## Runtime population and SLA\n"
        "\n"
        "- Sampled population: 12 bands x 1039 rows; 2 refusals at e=0.95 under "
        "the strict above-ceiling certificate policy.\n"
        "- SLA row: OPEN (universal service still requires the "
        "analytic/complete-domain argument).\n"
        "\n"
        "## External trust inventory\n"
        "\n"
        "- IEEE round-to-nearest binary64 arithmetic and `mul_add` are external "
        "(hardware/compiler).\n"
        "- ASSUMPTION-SIN: the platform sine is within 1 ulp on |x| <= 8.\n"
        "- libm, compiler and hardware are external; no Aeneas `f64` source "
        "refinement exists for this path.\n"
        "- The bridge certificate is a conditional Lean theorem on a supplied "
        "sound bound, not an executable refinement.\n"
    )


BARRIER_FIXTURE_REL_PREFIX = "// W7 barrier control fixture (synthetic, not the real source): "


def _barrier_pristine(root: Path) -> dict[str, str]:
    sources = {
        rel: f"{BARRIER_FIXTURE_REL_PREFIX}{rel}\n" for rel in BARRIER_DIGEST_FILES
    }
    digests = {
        rel: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for rel, content in sources.items()
    }
    texts: dict[str, str] = {
        LAKEFILE_REL: (root / LAKEFILE_REL).read_text(
            encoding="utf-8", errors="replace"
        ),
        MANIFEST_REL: (root / MANIFEST_REL).read_text(
            encoding="utf-8", errors="replace"
        ),
        EVIDENCE_REL: _barrier_evidence_template(digests),
    }
    texts.update(sources)
    return texts


def _bar_mut_abstract_refinement(texts: dict[str, str]) -> dict[str, str]:
    texts[EVIDENCE_REL] += (
        "\nThe bridge certificate theorem is an executable f64 refinement of "
        "the runtime solve_kepler kernel.\n"
    )
    return texts


def _bar_mut_drop_runtime_receipt(texts: dict[str, str]) -> dict[str, str]:
    target = "| runtime controls | `--stage runtime --controls` | 0/1/0 x3 |\n"
    if texts[EVIDENCE_REL].count(target) != 1:
        raise MutationError(
            "barrier control B runtime control receipt row not found exactly once"
        )
    texts[EVIDENCE_REL] = texts[EVIDENCE_REL].replace(target, "")
    return texts


def _bar_mut_stale_digest(texts: dict[str, str]) -> dict[str, str]:
    lines = texts[EVIDENCE_REL].splitlines(keepends=True)
    index = None
    for number, line in enumerate(lines):
        if KEPLER_SRC_REL in line and SHA256_RE.search(line):
            index = number
            break
    if index is None:
        raise MutationError("barrier control C digest row not found")
    old = SHA256_RE.search(lines[index]).group(0)
    wrong = "0" * 64 if old != "0" * 64 else "1" * 64
    lines[index] = SHA256_RE.sub(wrong, lines[index], count=1)
    texts[EVIDENCE_REL] = "".join(lines)
    return texts


@dataclass(frozen=True)
class BarrierControl:
    name: str
    defect: str
    rule: str
    tokens: tuple[str, ...]
    mutate: Callable[[dict[str, str]], dict[str, str]]


BARRIER_CONTROLS: tuple[BarrierControl, ...] = (
    BarrierControl(
        "A",
        "abstract/bridge bound reported as an executable f64 refinement",
        "BARRIER-CLAIM",
        ("f64", "refinement"),
        _bar_mut_abstract_refinement,
    ),
    BarrierControl(
        "B",
        "runtime control receipt dropped while the rest looks valid",
        "BARRIER-RECEIPT",
        ("runtime", "0/1/0"),
        _bar_mut_drop_runtime_receipt,
    ),
    BarrierControl(
        "C",
        "listed source digest does not match the source",
        "BARRIER-DIGEST",
        ("SHA-256", KEPLER_SRC_REL),
        _bar_mut_stale_digest,
    ),
)


def _write_barrier_files(root: Path, texts: dict[str, str]) -> None:
    for rel, content in texts.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _run_barrier_stage(fixture_root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(HERE),
            "--stage",
            "barrier",
            "--root",
            str(fixture_root),
        ],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout + proc.stderr


def barrier_controls(root: Path) -> int:
    """Baseline/red/restore for the barrier stage on a scratch fixture built
    from a synthetic conformant integration record plus the real pin files; the
    working tree is never touched. The real tree is EXPECTED RED until W7-T1
    lands, so the green baseline comes from the fixture, never from the tree."""
    try:
        pristine = _barrier_pristine(root)
    except OSError as exc:
        print(f"FAIL: barrier fixture inputs unreadable: {exc}", file=sys.stderr)
        return EXIT_USAGE

    tmp_parent: str | None = None
    preferred = Path("/tmp/opencode")
    if preferred.is_dir() and os.access(preferred, os.W_OK):
        tmp_parent = str(preferred)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="kepler-w7-t0-", dir=tmp_parent) as tmp:
        fixture = Path(tmp) / "fixture"
        rows: list[tuple[str, str, str, bool]] = []
        for control in BARRIER_CONTROLS:
            texts = dict(pristine)
            try:
                texts = control.mutate(texts)
            except MutationError as exc:
                print(
                    f"FAIL: barrier control {control.name} mutation infrastructure "
                    f"error: {exc}",
                    file=sys.stderr,
                )
                return EXIT_USAGE

            _write_barrier_files(fixture, pristine)
            base_code, base_out = _run_barrier_stage(fixture)
            base_ok = base_code == EXIT_OK and "BARRIER CHECK CLEAN" in base_out

            _write_barrier_files(fixture, texts)
            mut_code, mut_out = _run_barrier_stage(fixture)
            attributable = f"RULE {control.rule} " in mut_out and all(
                token in mut_out for token in control.tokens
            )
            red_ok = mut_code == EXIT_REJECT and attributable
            first_rule = _first_rule_line(mut_out) if mut_out.strip() else "(no output)"

            _write_barrier_files(fixture, pristine)
            rest_code, rest_out = _run_barrier_stage(fixture)
            rest_ok = rest_code == EXIT_OK and "BARRIER CHECK CLEAN" in rest_out

            triple_ok = base_ok and red_ok and rest_ok
            observed = f"{base_code}/{mut_code}/{rest_code} :: {first_rule}"
            rows.append(
                (
                    f"barrier control {control.name} ({control.defect})",
                    "0/1/0",
                    observed,
                    triple_ok,
                )
            )
            if not triple_ok:
                if not base_ok:
                    failures.append(f"{control.name}: fixture baseline is not green")
                if not red_ok:
                    reason = (
                        "survived"
                        if mut_code == EXIT_OK
                        else (
                            f"rejected without attributable {control.rule} diagnostic"
                            if mut_code == EXIT_REJECT
                            else f"infrastructure exit {mut_code}"
                        )
                    )
                    failures.append(f"{control.name}: {reason}")
                if not rest_ok:
                    failures.append(f"{control.name}: restored fixture is not green")

    width = max(len(row[0]) for row in rows)
    print(f"{'BARRIER CONTROL MATRIX':<{width}}  EXPECTED  OBSERVED")
    print("-" * (width + 60))
    for name, expected, observed, ok in rows:
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<{width}}  {expected:<9} {observed:<9} {verdict}")
    if failures:
        print(f"BARRIER CONTROLS FAIL: {len(failures)} control(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return EXIT_REJECT
    print(
        f"BARRIER CONTROLS PASS: baseline/red/restore = 0/1/0 for "
        f"{len(BARRIER_CONTROLS)}/{len(BARRIER_CONTROLS)} controls"
    )
    return EXIT_OK


def self_test() -> int:
    """Combined self-test: the contract, pin, bridge, runtime and barrier
    campaigns; exits 0 only when all five are green."""
    contract = contract_campaign()
    pin = pin_controls(DEFAULT_ROOT, CORPUS_CHECKOUT)
    bridge = bridge_controls(DEFAULT_ROOT, CORPUS_CHECKOUT)
    runtime = runtime_controls()
    barrier = barrier_controls(DEFAULT_ROOT)
    if (
        contract == EXIT_OK
        and pin == EXIT_OK
        and bridge == EXIT_OK
        and runtime == EXIT_OK
        and barrier == EXIT_OK
    ):
        print(
            "SELF-TEST PASS: contract campaign green (baseline, 6/6 attributable "
            "rejections, restores), pin controls green (0/1/0 x3), bridge "
            "controls green (0/1/0 x6), runtime controls green (0/1/0 x3) and "
            "barrier controls green (0/1/0 x3)"
        )
        return EXIT_OK
    print(
        f"SELF-TEST FAIL: contract campaign exit {contract}, pin controls exit "
        f"{pin}, bridge controls exit {bridge}, runtime controls exit {runtime}, "
        f"barrier controls exit {barrier}"
    )
    return EXIT_REJECT



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list() -> int:
    print("Rules (each violation names rule, document, line and the observed problem):")
    for rule_id, title, description in RULES:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    print("Contract documents scanned by --check:")
    for rel in CONTRACT_DOCS:
        print(f"  {rel}")
    print(f"Canonical corpus pin: {CANONICAL_PIN}")
    print(
        f"Inventory declarations required: {len(INVENTORY_DECLARATIONS)} "
        f"({', '.join(INVENTORY_DECLARATIONS)})"
    )
    print(f"Typed refusal variants required: {', '.join(REFUSAL_VARIANTS)}")
    print(f"Slot markers required: {', '.join(SLOT_MARKERS)}")
    print("Self-test controls: " + ", ".join(f"{c.name} {c.defect}" for c in CONTROLS))
    print("--stage pin rules (bridge pin files vs published corpus):")
    for rule_id, title, description in PIN_STAGE_RULES:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    print(f"--stage pin files: {LAKEFILE_REL}, {MANIFEST_REL}")
    print(f"--stage pin corpus checkout: {CORPUS_CHECKOUT}")
    print("Pin controls: " + ", ".join(f"{c.name} {c.defect}" for c in PIN_CONTROLS))
    print("--stage bridge rules (certificate module, audit, roots):")
    for rule_id, title, description in BRIDGE_STAGE_RULES:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    print(
        f"--stage bridge files: {CERTIFICATE_REL}, {AXIOMAUDIT_REL}, {LAKEFILE_REL} "
        f"(pin delegated to --stage pin)"
    )
    print("Bridge controls: " + ", ".join(f"{c.name} {c.defect}" for c in BRIDGE_CONTROLS))
    print("--stage runtime rules (executing kernel deliverables, W6):")
    for rule_id, title, description in RUNTIME_STAGE_RULES:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    print(
        f"--stage runtime files: {KEPLER_SRC_REL}, {LIB_REL}, {KEPLER_TEST_REL} "
        f"(structural only; no Rust compile, pin/bridge remain separate stages)"
    )
    print("Runtime controls: " + ", ".join(f"{c.name} {c.defect}" for c in RUNTIME_CONTROLS))
    print("--stage barrier rules (integration record and shared receipts, W7):")
    for rule_id, title, description in BARRIER_STAGE_RULES:
        print(f"  {rule_id}: {title}")
        print(f"     {description}")
    print(
        f"--stage barrier record: {EVIDENCE_REL} (digest freshness over "
        f"{', '.join(BARRIER_DIGEST_FILES)})"
    )
    print(
        f"--stage barrier corpus revision: {EXPECTED_CORPUS_REVISION}; Apeiron "
        f"base: {EXPECTED_APEIRON_BASE}"
    )
    print("Barrier controls: " + ", ".join(f"{c.name} {c.defect}" for c in BARRIER_CONTROLS))
    return EXIT_OK


def cmd_stage_missing(stage: str) -> int:
    print(
        f"MISSING: --stage '{stage}' is not implemented by tools/check-kepler.py. "
        "The implemented stages are the W1 contract gate (--check, --self-test), "
        "'pin' (APEIRON-KEPLER-W4-T0), 'bridge' (APEIRON-KEPLER-W5-T0), "
        "'runtime' (APEIRON-KEPLER-W6-T0) and 'barrier' (APEIRON-KEPLER-W7-T0). "
        "Stages 'foundation'/'enclosure' belong to the corpus checker "
        "(scripts/check-kepler.py in sanguine-apeiron). A missing gate is not "
        "success."
    )
    return EXIT_REJECT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check-kepler.py",
        description="Rejecting gate for the canonical Kepler contract.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate the contract documents (default)")
    mode.add_argument("--self-test", dest="self_test", action="store_true",
                      help="baseline/red/restore campaign on scratch fixtures")
    mode.add_argument("--stage", metavar="STAGE",
                      help="stage gate: 'pin', 'bridge', 'runtime' and 'barrier' "
                           "are implemented; every other stage is MISSING")
    mode.add_argument("--list", action="store_true", help="list the rules and scanned documents")
    parser.add_argument("--controls", action="store_true",
                        help="run the control campaign: with --stage pin, the pin "
                             "baseline/red/restore campaign; alone, alias for --self-test")
    parser.add_argument("--root", metavar="DIR", default=None,
                        help="repository root for --check/--stage pin (default: this repository)")
    parser.add_argument("--corpus", metavar="DIR", default=str(CORPUS_CHECKOUT),
                        help="sibling corpus checkout for --stage pin reachability "
                             f"(default: {CORPUS_CHECKOUT})")
    args = parser.parse_args(argv)

    try:
        if args.list:
            return cmd_list()
        if args.stage is not None:
            if args.stage == "pin":
                root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
                corpus = Path(args.corpus).resolve()
                if args.controls:
                    return pin_controls(root, corpus)
                return cmd_pin_check(root, corpus)
            if args.stage == "bridge":
                root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
                corpus = Path(args.corpus).resolve()
                if args.controls:
                    return bridge_controls(root, corpus)
                return cmd_bridge_check(root, corpus)
            if args.stage == "runtime":
                root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
                if args.controls:
                    return runtime_controls()
                return cmd_runtime_check(root)
            if args.stage == "barrier":
                root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
                if args.controls:
                    return barrier_controls(root)
                return cmd_barrier_check(root)
            return cmd_stage_missing(args.stage)
        if args.self_test or args.controls:
            return self_test()
        root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
        return cmd_check(root)
    except GateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except subprocess.TimeoutExpired:
        print(
            f"FAIL: checker subprocess exceeded {SUBPROCESS_TIMEOUT_S}s; "
            "infrastructure failure is not a catch",
            file=sys.stderr,
        )
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
