#!/usr/bin/env python3
"""Exact proof-target binding for the Iota source-refinement harness.

The harness must reject a same-named helper or a theorem of ``True`` standing in
for the required proposition. For every proof artifact the artifact names one
exact declaration and its expected type as Lean source; this module builds the
module under test in a throwaway, then runs a generated Lean probe that

* resolves the declaration by its exact fully-qualified name in the elaborated
  environment (so commented text and namespace helpers do not resolve),
* elaborates the expected type in that same environment and requires kernel
  definitional equality with the declaration's type,
* rejects an ``axiom`` target and every nonempty transitive axiom cone, and
* resolves canonical dependencies and admits only the seven Stream 0
  definitions selected by the frozen migration boundary.

The probe is generated text inside ``__ApeironDeclProbe.lean`` and is deleted
after the run. This module owns no stage policy; it is called by ``refine.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROBE_NAME = "__ApeironDeclProbe.lean"
CANONICAL_LINE = "STREAM0_CANONICAL_OK"

ADMISSIBLE_CANONICAL = (
    "Apeiron.Table.HMap",
    "Apeiron.Table.get",
    "Apeiron.Table.insertWith",
    "Apeiron.Table.build",
    "Apeiron.Trichotomy.AnyRing.RingCell",
    "Apeiron.Trichotomy.AnyRing.ringMul",
    "Apeiron.Trichotomy.AnyRing.ringNormForm",
)


class ProofTargetError(Exception):
    """A target is missing, mistyped, axiomatic or outside the Stream 0 boundary."""


def lean_string(value: str) -> str:
    return json.dumps(value)


def probe_text(import_name: str, declaration: str, expected_type: str) -> str:
    """Generate the Lean probe that binds one exact proof target."""
    allowed = "({} : NameSet)"
    for name in ADMISSIBLE_CANONICAL:
        allowed = f"({allowed}).insert `{name}"
    return (
        f"import {import_name}\n"
        "import Lean\n"
        "open Lean Elab Command Meta\n"
        "private def isStream0Canonical (name : Name) : Bool :=\n"
        "  let text := name.toString\n"
        "  text.startsWith \"Apeiron.Table.\" ||\n"
        "    text.startsWith \"Apeiron.Trichotomy.AnyRing.\"\n"
        "private partial def stream0CanonicalDeps\n"
        "    (env : Environment) (pending : List Name)\n"
        "    (seen found : NameSet := {}) : NameSet :=\n"
        "  match pending with\n"
        "  | [] => found\n"
        "  | name :: rest =>\n"
        "      if seen.contains name then\n"
        "        stream0CanonicalDeps env rest seen found\n"
        "      else\n"
        "        let seen := seen.insert name\n"
        "        if isStream0Canonical name then\n"
        "          stream0CanonicalDeps env rest seen (found.insert name)\n"
        "        else\n"
        "          match env.find? name with\n"
        "          | none => stream0CanonicalDeps env rest seen found\n"
        "          | some info =>\n"
        "              stream0CanonicalDeps env\n"
        "                (info.getUsedConstantsAsSet.toList ++ rest) seen found\n"
        "run_cmd do\n"
        "  let env ← getEnv\n"
        f"  let exact := {lean_string(declaration)}\n"
        f"  let expectedSrc := {lean_string(expected_type)}\n"
        f"  let admissible : NameSet := {allowed}\n"
        "  let some info := env.find? exact.toName\n"
        "    | throwError s!\"missing proof target declaration: {exact}\"\n"
        "  if expectedSrc != \"\" then\n"
        "    let stx ← match Lean.Parser.runParserCategory env `term expectedSrc with\n"
        "      | .ok s => pure s\n"
        "      | .error e => throwError s!\"cannot parse expected type: {e}\"\n"
        "    let expected ← liftTermElabM do\n"
        "      let e ← Lean.Elab.Term.elabType stx\n"
        "      Lean.Elab.Term.synthesizeSyntheticMVarsNoPostponing\n"
        "      instantiateMVars e\n"
        "    unless ← liftTermElabM (Meta.isDefEq expected info.type) do\n"
        "      throwError s!\"proof target has the wrong type: {exact}\"\n"
        "  if let .axiomInfo _ := info then\n"
        "    throwError s!\"proof target {exact} is an axiom, not a checked declaration\"\n"
        "  let canonical := stream0CanonicalDeps env [exact.toName]\n"
        "  let forbidden := canonical.toList.filter (fun name => !admissible.contains name)\n"
        "  if !forbidden.isEmpty then\n"
        "    throwError s!\"proof target {exact} names non-admissible Stream 0 canonical declaration(s): {forbidden}\"\n"
        "  for name in canonical.toList do\n"
        f"    logInfo s!\"{CANONICAL_LINE} {{name}} admissible\"\n"
        "  let axioms ← Lean.collectAxioms exact.toName\n"
        "  if !axioms.isEmpty then\n"
        "    throwError s!\"proof target {exact} depends on a nonempty Stream 0 axiom set: {axioms.toList}\"\n"
        "  logInfo \"declaration-present\"\n"
    )


def _tail(text: str, limit: int = 400) -> str:
    return (text or "").strip()[-limit:]


def _inside(candidate: Path, root: Path) -> bool:
    try:
        resolved = candidate.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    try:
        return os.path.commonpath([str(resolved), str(resolved_root)]) == str(
            resolved_root
        )
    except ValueError:
        return False


def _run_once(argv: list[str], cwd: Path, timeout: int):
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProofTargetError(
            f"timeout after {timeout}s running {' '.join(argv)}"
        ) from exc
    except OSError as exc:
        raise ProofTargetError(
            f"subprocess failed to start ({' '.join(argv)}): {exc}"
        ) from exc


def verify(
    package_dir: Path,
    module_rel: str,
    declaration: str,
    expected_type: str,
    resolved_dest: Path,
    timeout: int,
) -> tuple[str, ...]:
    """Build the module and require the exact declaration with the exact type."""
    import_name = module_rel[:-5].replace("/", ".") if module_rel.endswith(
        ".lean"
    ) else module_rel
    probe = package_dir / PROBE_NAME
    if not _inside(probe, resolved_dest):
        raise ProofTargetError(
            f"refusing to write outside the resolved throwaway root: {probe}"
        )
    probe.write_text(probe_text(import_name, declaration, expected_type), "utf-8")
    try:
        build = _run_once(["lake", "build", import_name], package_dir, timeout)
        if build.returncode != 0:
            raise ProofTargetError(
                f"proof dependency build failed: {module_rel}: {_tail(build.stdout)}"
            )
        loaded = _run_once(["lake", "env", "lean", PROBE_NAME], package_dir, timeout)
        if loaded.returncode != 0:
            raise ProofTargetError(
                f"proof target declaration not loaded from real code: "
                f"{module_rel}::{declaration}: {_tail(loaded.stdout)}"
            )
        prefix = CANONICAL_LINE + " "
        return tuple(
            line[len(prefix) :].removesuffix(" admissible")
            for line in loaded.stdout.splitlines()
            if line.startswith(prefix) and line.endswith(" admissible")
        )
    finally:
        probe.unlink(missing_ok=True)
