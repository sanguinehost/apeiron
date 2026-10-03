/-
Axiom-cone audit of the crypto headlines outside `Crypto.Keccak.Audit`: the group-law keystream
theorems of `Crypto.Security.Core`, the modular-arithmetic and Legendre/square-root theorems of
`Crypto.Field.ZModPrime`, the `GF(2^n)` arithmetic of `Crypto.Field.GF2n`, the two prime fields
`GF(2^255 − 19)` and `GF(2^256 − 2^32 − 977)` of `Crypto.Field.Fp25519` and
`Crypto.Field.FpSecp256k1`, and the transcript determinism and absorb laws of
`Crypto.Transcript`.

The `#audit` command below is the same one `Crypto/Keccak/Audit.lean` defines: it collects each
constant's transitive axiom dependencies with `Lean.collectAxioms` — the mechanism
`#print axioms` uses — and calls `throwError` on anything outside
`{propext, Classical.choice, Quot.sound}`. `Crypto.Sponge`, `Crypto.ChaCha`, `Crypto.Sha2`,
`Crypto.Sha512`, `Crypto.Aes`, `Crypto.Hmac`, `Crypto.Blake2` and `Crypto.Blake2s` carry
known-answer `#guard` checks over the untrusted evaluator and no `theorem`s at all (audit sweep
2026-10-03), so they register nothing here; `Crypto.Merkle`'s 21 theorems are registered below.
-/

import Crypto.Security.Core
import Crypto.Field.ZModPrime
import Crypto.Field.GF2n
import Crypto.Field.Fp25519
import Crypto.Field.FpSecp256k1
import Crypto.Transcript
import Crypto.Merkle
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

-- Crypto.Field.Fp25519 (`GF(2^255 − 19)`): the modulus pins and the defining reduction, the Rust
-- limb and word forms of `p`, the ring laws of `ZMod p`, the exponent arithmetic of `pow_p58`,
-- the `p ≡ 5 (mod 8)` square root with its stated partiality, Fermat inversion, and the
-- 32-byte little-endian round trips.
#audit Apeiron.Crypto.Field.Fp25519.p_eq_decimal
#audit Apeiron.Crypto.Field.Fp25519.p_ne_zero
#audit Apeiron.Crypto.Field.Fp25519.two_lt_p
#audit Apeiron.Crypto.Field.Fp25519.p_lt_two_255
#audit Apeiron.Crypto.Field.Fp25519.p_mod_eight
#audit Apeiron.Crypto.Field.Fp25519.div_half
#audit Apeiron.Crypto.Field.Fp25519.two_pow_255_eq_nineteen
#audit Apeiron.Crypto.Field.Fp25519.self_eq_zero
#audit Apeiron.Crypto.Field.Fp25519.expP58_eq_two_pow_252_sub_3
#audit Apeiron.Crypto.Field.Fp25519.pLimbs_sum
#audit Apeiron.Crypto.Field.Fp25519.pWords_sum
#audit Apeiron.Crypto.Field.Fp25519.add_comm
#audit Apeiron.Crypto.Field.Fp25519.add_assoc
#audit Apeiron.Crypto.Field.Fp25519.add_zero
#audit Apeiron.Crypto.Field.Fp25519.zero_add
#audit Apeiron.Crypto.Field.Fp25519.add_neg_cancel
#audit Apeiron.Crypto.Field.Fp25519.mul_comm
#audit Apeiron.Crypto.Field.Fp25519.mul_assoc
#audit Apeiron.Crypto.Field.Fp25519.mul_one
#audit Apeiron.Crypto.Field.Fp25519.one_mul
#audit Apeiron.Crypto.Field.Fp25519.mul_zero
#audit Apeiron.Crypto.Field.Fp25519.add_mul
#audit Apeiron.Crypto.Field.Fp25519.mul_add
#audit Apeiron.Crypto.Field.Fp25519.sub_eq_add_neg'
#audit Apeiron.Crypto.Field.Fp25519.pow_add'
#audit Apeiron.Crypto.Field.Fp25519.pow_mul'
#audit Apeiron.Crypto.Field.Fp25519.powF_eq
#audit Apeiron.Crypto.Field.Fp25519.powF_succ
#audit Apeiron.Crypto.Field.Fp25519.powF_eq_powZMod
#audit Apeiron.Crypto.Field.Fp25519.powP58_eq
#audit Apeiron.Crypto.Field.Fp25519.powP58_mul
#audit Apeiron.Crypto.Field.Fp25519.sqrtCandidate_eq_powP58_mul
#audit Apeiron.Crypto.Field.Fp25519.powP58_sq_pow
#audit Apeiron.Crypto.Field.Fp25519.sqrtCandidate_sq_eq
#audit Apeiron.Crypto.Field.Fp25519.quarter_sq_eq
#audit Apeiron.Crypto.Field.Fp25519.sqrtMinusOne_sq_pow
#audit Apeiron.Crypto.Field.Fp25519.two_ne_zero
#audit Apeiron.Crypto.Field.Fp25519.sqrtMinusOne_sq
#audit Apeiron.Crypto.Field.Fp25519.is_square_iff_euler_25519
#audit Apeiron.Crypto.Field.Fp25519.sqrtWitness
#audit Apeiron.Crypto.Field.Fp25519.sqrt_of_residue
#audit Apeiron.Crypto.Field.Fp25519.sqrt_not_of_nonresidue
#audit Apeiron.Crypto.Field.Fp25519.invert_zero
#audit Apeiron.Crypto.Field.Fp25519.invertOrZero_zero
#audit Apeiron.Crypto.Field.Fp25519.invertOrZero_ne_zero
#audit Apeiron.Crypto.Field.Fp25519.invert_eq_invZMod
#audit Apeiron.Crypto.Field.Fp25519.invert_mul_cancel
#audit Apeiron.Crypto.Field.Fp25519.invertOrZero_mul_cancel
#audit Apeiron.Crypto.Field.Fp25519.bytes32Value_lt_top_bit
#audit Apeiron.Crypto.Field.Fp25519.byteRoundTrip
#audit Apeiron.Crypto.Field.Fp25519.bytes32Value_encode
#audit Apeiron.Crypto.Field.Fp25519.decode_encode
#audit Apeiron.Crypto.Field.Fp25519.encode_decode

-- Crypto.Field.FpSecp256k1 (`GF(2^256 − 2^32 − 977)`): the limb, byte and exponent pins on the
-- Rust constants, the reduction identities, the 32-byte big-endian encode/decode round trips, and
-- the generic-prime inversion, exponentiation and `q ≡ 3 (mod 4)` square-root laws together with
-- their reading at `p` under `secpPrimeCert`.
#audit Apeiron.Crypto.Field.Secp256k1.p_eq_hex
#audit Apeiron.Crypto.Field.Secp256k1.limbsToNat_pLimbs
#audit Apeiron.Crypto.Field.Secp256k1.beNat_pBytes
#audit Apeiron.Crypto.Field.Secp256k1.limbsToNat_eLimbs
#audit Apeiron.Crypto.Field.Secp256k1.two_pow_256_eq_add
#audit Apeiron.Crypto.Field.Secp256k1.two_pow_256_mod_p
#audit Apeiron.Crypto.Field.Secp256k1.p_ne_zero
#audit Apeiron.Crypto.Field.Secp256k1.two_le_p
#audit Apeiron.Crypto.Field.Secp256k1.p_lt_two_pow_256
#audit Apeiron.Crypto.Field.Secp256k1.two_pow_255_le_p
#audit Apeiron.Crypto.Field.Secp256k1.two_pow_256_le_two_mul_p
#audit Apeiron.Crypto.Field.Secp256k1.odd_p
#audit Apeiron.Crypto.Field.Secp256k1.mod_four_p
#audit Apeiron.Crypto.Field.Secp256k1.two_pow_8_succ_mul
#audit Apeiron.Crypto.Field.Secp256k1.beNat_lt_pow_len
#audit Apeiron.Crypto.Field.Secp256k1.beNat_lt_two_pow_256
#audit Apeiron.Crypto.Field.Secp256k1.lt_two_mul_p_of_lt_two_pow_256
#audit Apeiron.Crypto.Field.Secp256k1.beNat_append_singleton
#audit Apeiron.Crypto.Field.Secp256k1.beNat_reverse
#audit Apeiron.Crypto.Field.Secp256k1.leNat_natToBytes
#audit Apeiron.Crypto.Field.Secp256k1.beNat_natToBytes_reverse
#audit Apeiron.Crypto.Field.Secp256k1.natToBytes_length_le
#audit Apeiron.Crypto.Field.Secp256k1.encode_length_le
#audit Apeiron.Crypto.Field.Secp256k1.beNat_encode
#audit Apeiron.Crypto.Field.Secp256k1.decodeBytes_encode
#audit Apeiron.Crypto.Field.Secp256k1.decodeBytes_val_lt
#audit Apeiron.Crypto.Field.Secp256k1.decodeBytes_beNat_mod
#audit Apeiron.Crypto.Field.Secp256k1.decodeBytes_pBytes
#audit Apeiron.Crypto.Field.Secp256k1.curveBField_val
#audit Apeiron.Crypto.Field.Secp256k1.secpInv_zero
#audit Apeiron.Crypto.Field.Secp256k1.secpInv_mul_cancel
#audit Apeiron.Crypto.Field.Secp256k1.secpInv_mul_cancel'
#audit Apeiron.Crypto.Field.Secp256k1.secpPow_eq
#audit Apeiron.Crypto.Field.Secp256k1.secpPow_one
#audit Apeiron.Crypto.Field.Secp256k1.secpPow_add
#audit Apeiron.Crypto.Field.Secp256k1.secpPow_mul
#audit Apeiron.Crypto.Field.Secp256k1.secpPow_sub_two_inv
#audit Apeiron.Crypto.Field.Secp256k1.two_mul_sqrt_exp
#audit Apeiron.Crypto.Field.Secp256k1.four_ne_zero_of_mod_four_three
#audit Apeiron.Crypto.Field.Secp256k1.secpSqrt_sq
#audit Apeiron.Crypto.Field.Secp256k1.secpSqrt_sq_of_isSquare
#audit Apeiron.Crypto.Field.Secp256k1.isSquare_iff_euler
#audit Apeiron.Crypto.Field.Secp256k1.secpSqrt_four_choice
#audit Apeiron.Crypto.Field.Secp256k1.neg_two_eq_natCast_sub_two
#audit Apeiron.Crypto.Field.Secp256k1.secpSqrt_four_choice_val
#audit Apeiron.Crypto.Field.Secp256k1.secpSqrt_four_val
#audit Apeiron.Crypto.Field.Secp256k1.secpInv_zero_p
#audit Apeiron.Crypto.Field.Secp256k1.secpPow_sub_two_inv_p
#audit Apeiron.Crypto.Field.Secp256k1.secpSqrt_sq_p
#audit Apeiron.Crypto.Field.Secp256k1.secpSqrt_four_val_p
#audit Apeiron.Crypto.Field.Secp256k1.secpFieldLawsAtP_holds

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

-- Crypto.Merkle: RFC 6962 §2.1. The height bound on a node, the two `combine` step laws that
-- fix which sibling is combined first, the range and shape laws of `path`, the RFC 6962 §2.1.1
-- path-length bound, the agreement of `path` and `dataAt` on the in-range indices, the
-- reconstruction of a tree's own root from its own path (`rootFn`, `combine_eq_rootFn`,
-- `rootFn_apply_root`, `combine_eq_root`) and the `verify_root` acceptance direction.
#audit Apeiron.Crypto.Merkle.height_node_of_le
#audit Apeiron.Crypto.Merkle.combine_cons_true
#audit Apeiron.Crypto.Merkle.combine_cons_false
#audit Apeiron.Crypto.Merkle.path_leaf_succ
#audit Apeiron.Crypto.Merkle.path_leaf
#audit Apeiron.Crypto.Merkle.size_pos
#audit Apeiron.Crypto.Merkle.size_node_pos
#audit Apeiron.Crypto.Merkle.cons_tag_ne
#audit Apeiron.Crypto.Merkle.path_node
#audit Apeiron.Crypto.Merkle.path_of_node_right
#audit Apeiron.Crypto.Merkle.path_eq_some_iff
#audit Apeiron.Crypto.Merkle.path_length_le_height
#audit Apeiron.Crypto.Merkle.dataAt_none_iff
#audit Apeiron.Crypto.Merkle.path_none_of_dataAt_none
#audit Apeiron.Crypto.Merkle.exists_path_of_dataAt
#audit Apeiron.Crypto.Merkle.exists_dataAt_of_path
#audit Apeiron.Crypto.Merkle.rootFn_iff_path
#audit Apeiron.Crypto.Merkle.combine_eq_rootFn
#audit Apeiron.Crypto.Merkle.rootFn_apply_root
#audit Apeiron.Crypto.Merkle.combine_eq_root
#audit Apeiron.Crypto.Merkle.verify_root
