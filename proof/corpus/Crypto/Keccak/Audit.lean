/-
FIPS-202 Keccak-f[1600] and the SHA-3/SHAKE sponge: axiom-cone audit of the exported headlines.

`#audit` collects each constant's transitive axiom dependencies with `Lean.collectAxioms` — the same
mechanism `#print axioms` uses — and calls `throwError` on anything outside
`{propext, Classical.choice, Quot.sound}`, so a `sorryAx` here is a build failure rather than a line
in a log nobody reads. The `#guard` known-answer checks in `Crypto.Sponge` run the untrusted
evaluator and are not proofs; the audit below gates the constants themselves.
-/

import Crypto.Keccak.Constants
import Crypto.Keccak.Round
import Crypto.Keccak.Certificate
import Crypto.Sponge
import Lean

set_option autoImplicit false
set_option warningAsError true

open Lean Elab Command in
/-- The three axioms every proof in this bridge may depend on. -/
private def auditClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
/-- **The gate.** `#audit foo` collects `foo`'s transitive axiom dependencies with
    `Lean.collectAxioms` and calls `throwError` if any of them falls outside `auditClosedAxioms`.
    Prints the cone on success, so the log carries the same information `#print axioms` did. -/
elab "#audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !auditClosedAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the closed cone " ++
        m!"(propext, Classical.choice, Quot.sound): {bad.toList}"

-- Constants: round constants and rho offsets (FIPS-202 §3.2)
#audit Apeiron.Crypto.Keccak.RC
#audit Apeiron.Crypto.Keccak.ROT

-- Round: the step functions and the 24-round permutation (FIPS-202 §3.2)
#audit Apeiron.Crypto.Keccak.theta
#audit Apeiron.Crypto.Keccak.rho
#audit Apeiron.Crypto.Keccak.pi
#audit Apeiron.Crypto.Keccak.chi
#audit Apeiron.Crypto.Keccak.iota
#audit Apeiron.Crypto.Keccak.round
#audit Apeiron.Crypto.Keccak.permute
#audit Apeiron.Crypto.Keccak.permute_eq_foldl_round

-- Sponge: parameters, byte order, padding, absorb/squeeze and the named functions (FIPS-202 §3-6)
#audit Apeiron.Crypto.Keccak.rate_sha3_256
#audit Apeiron.Crypto.Keccak.rate_sha3_512
#audit Apeiron.Crypto.Keccak.rate_shake128
#audit Apeiron.Crypto.Keccak.rate_shake256
#audit Apeiron.Crypto.Keccak.domain_sha3
#audit Apeiron.Crypto.Keccak.domain_shake
#audit Apeiron.Crypto.Keccak.byteOf
#audit Apeiron.Crypto.Keccak.xorBlock
#audit Apeiron.Crypto.Keccak.pad
#audit Apeiron.Crypto.Keccak.blocks
#audit Apeiron.Crypto.Keccak.absorb
#audit Apeiron.Crypto.Keccak.squeeze
#audit Apeiron.Crypto.Keccak.sponge
#audit Apeiron.Crypto.Keccak.sha3_256
#audit Apeiron.Crypto.Keccak.sha3_512
#audit Apeiron.Crypto.Keccak.shake128
#audit Apeiron.Crypto.Keccak.shake256

-- Refinement: the extracted `word_keccakf` computes `permute` on `view` (bridge corpus)
#audit apeiron_word.RoundStepDenotes
#audit apeiron_word.WordKeccakfDenotes
