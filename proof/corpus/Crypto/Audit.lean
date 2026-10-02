/-
Axiom-cone audit of the crypto headlines outside `Crypto.Keccak.Audit`: the group-law keystream
theorems of `Crypto.Security.Core`, the modular-arithmetic and Legendre/square-root theorems of
`Crypto.Field.ZModPrime`, the `GF(2^n)` arithmetic of `Crypto.Field.GF2n`, and the transcript
determinism and absorb laws of `Crypto.Transcript`.

The `#audit` command below is the same one `Crypto/Keccak/Audit.lean` defines: it collects each
constant's transitive axiom dependencies with `Lean.collectAxioms` — the mechanism
`#print axioms` uses — and calls `throwError` on anything outside
`{propext, Classical.choice, Quot.sound}`. `Crypto.Sponge`, `Crypto.ChaCha`, `Crypto.Sha2`,
`Crypto.Hmac` and `Crypto.Merkle` carry known-answer `#guard` checks over the untrusted evaluator and
no proved headlines, so they register nothing here.
-/

import Crypto.Security.Core
import Crypto.Field.ZModPrime
import Crypto.Field.GF2n
import Crypto.Transcript
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

-- Crypto.Security.Core: keystream laws over an additive commutative group, and the
-- fixed-width `BitVec` instantiations of the same three laws.
#audit Apeiron.Crypto.Security.xorStream_involutive
#audit Apeiron.Crypto.Security.otp_changes_the_wire
#audit Apeiron.Crypto.Security.short_key_passthrough
#audit Apeiron.Crypto.Security.bitVecXorStream_involutive
#audit Apeiron.Crypto.Security.bitVecOtp_changes_the_wire
#audit Apeiron.Crypto.Security.bitVecShortKeyPassthrough

-- Crypto.Field.ZModPrime: inverses and `powZMod`, the Legendre symbol, the square tests,
-- Tonelli–Shanks and the nonresidue existence witness.
#audit Apeiron.Crypto.Field.invZMod_eq_inv
#audit Apeiron.Crypto.Field.mul_invZMod_of_ne_zero
#audit Apeiron.Crypto.Field.invZMod_mul_of_ne_zero
#audit Apeiron.Crypto.Field.invZMod_eq_pow_sub_two
#audit Apeiron.Crypto.Field.powZMod_zero
#audit Apeiron.Crypto.Field.powZMod_succ
#audit Apeiron.Crypto.Field.powZMod_eq
#audit Apeiron.Crypto.Field.legendre_zero
#audit Apeiron.Crypto.Field.legendre_one
#audit Apeiron.Crypto.Field.legendre_mul
#audit Apeiron.Crypto.Field.legendre_eq_zero_iff
#audit Apeiron.Crypto.Field.legendre_eq_neg_one_iff
#audit Apeiron.Crypto.Field.legendre_sq_one
#audit Apeiron.Crypto.Field.legendre_eq_one_or_neg_one
#audit Apeiron.Crypto.Field.legendre_eq_one_iff
#audit Apeiron.Crypto.Field.squareTest_eq_true
#audit Apeiron.Crypto.Field.anySquareTest_eq_true
#audit Apeiron.Crypto.Field.isSquare_iff_exists_natCast
#audit Apeiron.Crypto.Field.is_square_iff_euler
#audit Apeiron.Crypto.Field.is_square_iff_legendre
#audit Apeiron.Crypto.Field.rootUpTo_sound
#audit Apeiron.Crypto.Field.rootUpTo_complete
#audit Apeiron.Crypto.Field.sqrt_mod_sound
#audit Apeiron.Crypto.Field.sqrt_mod_complete
#audit Apeiron.Crypto.Field.sqrt_mod_eq_some_sound
#audit Apeiron.Crypto.Field.sqrt_mod_eq_none_iff
#audit Apeiron.Crypto.Field.sqrt_mod_TS_correct
#audit Apeiron.Crypto.Field.exists_nonresidue

-- Crypto.Field.GF2n: the AES/GCM modulus facts, the additive structure, the carry-less
-- multiplication zero laws, the power laws, and commutativity/associativity/inverse under a
-- stated reduction law.
#audit Crypto.Field.GF2n.modulus_eight
#audit Crypto.Field.GF2n.aesTail_degree
#audit Crypto.Field.GF2n.aesModulus_monic
#audit Crypto.Field.GF2n.aesModulus_natDegree
#audit Crypto.Field.GF2n.aesModulus_ne_zero
#audit Crypto.Field.GF2n.modulus_monic
#audit Crypto.Field.GF2n.modulus_ne_zero
#audit Crypto.Field.GF2n.add_comm
#audit Crypto.Field.GF2n.add_assoc
#audit Crypto.Field.GF2n.add_zero
#audit Crypto.Field.GF2n.zero_add
#audit Crypto.Field.GF2n.add_left_comm
#audit Crypto.Field.GF2n.add_self
#audit Crypto.Field.GF2n.xtime_zero
#audit Crypto.Field.GF2n.toNat_zero
#audit Crypto.Field.GF2n.testBit_of_toNat_zero
#audit Crypto.Field.GF2n.testBit_width_zero
#audit Crypto.Field.GF2n.mulAux_zero_right
#audit Crypto.Field.GF2n.mul_zero_right
#audit Crypto.Field.GF2n.mulAux_zero_left
#audit Crypto.Field.GF2n.mul_zero
#audit Crypto.Field.GF2n.testBit_one
#audit Crypto.Field.GF2n.mulAux_width_zero
#audit Crypto.Field.GF2n.mulAux_one_right
#audit Crypto.Field.GF2n.mul_one
#audit Crypto.Field.GF2n.pow_zero
#audit Crypto.Field.GF2n.mul_pow_succ
#audit Crypto.Field.GF2n.mul_comm
#audit Crypto.Field.GF2n.mul_assoc
#audit Crypto.Field.GF2n.one_ne_zero
#audit Crypto.Field.GF2n.inv_ne_zero

-- Crypto.Transcript: `squeeze` is a function of the absorbed bytes alone, so equal histories give
-- equal challenges at every length; `absorbBytes` appends and leaves the counter alone; absorbing a
-- challenge records it without advancing the counter; and a nonempty absorb changes both the
-- absorbed list and the wire.
#audit Apeiron.Crypto.Transcript.squeeze_eq_shake128
#audit Apeiron.Crypto.Transcript.squeeze_congr_absorbed
#audit Apeiron.Crypto.Transcript.absorbBytes_absorbed
#audit Apeiron.Crypto.Transcript.absorbBytes_counter
#audit Apeiron.Crypto.Transcript.absorbBytes_assoc
#audit Apeiron.Crypto.Transcript.absorbChallenge_absorbed
#audit Apeiron.Crypto.Transcript.absorbChallenge_counter
#audit Apeiron.Crypto.Transcript.changes_the_wire_absorbed
