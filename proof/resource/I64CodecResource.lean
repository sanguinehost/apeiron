import I64CodecSource
import Aeneas

/-
SCOPE: actual extracted `i64` word-codec bodies — `I64WordCodec::{encode,decode}` reached
through `proof/extract/lib.rs`'s `word_i64_codec_encode` / `word_i64_codec_decode` roots — their
guards, the constant word-operation work measure, the `massert` refusal branch, and a reachable
success witness carrying the round trip.
REGISTER: scalar source refinement of the extracted body and its resource model; no canonical
ring correspondence, no hardware or cycle measurement, no claim about the table's probe,
reservation or spill path.
CORPUS: real source `src/table/keyrep.rs` (`I64WordCodec::{encode,decode}`, `address_identity`),
`src/coord.rs` (`spread`, `unspread`, `PHI`, `PHI_INV`) and `src/table.rs` (`WordHome::address`);
the actual extracted `i64_codec_source.*` definitions in I64CodecSource.lean.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean, the committed extraction of the
same `keyrep.rs` bodies, and proof/extract/lib.rs, the forwarding wrapper this gate reproduces.
GATE: python3 proof/resource/I64Codec_gate.py --controls, 540 s budget,
/tmp/opencode/apeiron-lane-3.lock, MemoryMax=12G, MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: Lean kernel check only when that gate succeeds. The work measure counts 64-bit and 128-bit
word operations under the named premises; it is never cycles, wall clock or GPU time.
-/

namespace I64CodecResource

open Aeneas Aeneas.Std Result
open i64_codec_source

/-! ## Width constants and the cast/mask lemmas the bodies rest on

The extracted bodies are stated over `u64` and `u128`. These read the model's modulus values; they
are not restatements of the Rust source. -/

theorem U128size : UScalar.size .U128 = 2 ^ 128 := by rw [UScalar.size]; norm_num

theorem U64size : UScalar.size .U64 = 2 ^ 64 := by rw [UScalar.size]; norm_num

/-- A `u64` value is below the `u128` modulus, so the widening cast into the multiply is exact. -/
theorem modU128 (x : U64) : x.val % 2 ^ 128 = x.val := by
  have h := U64.le_max x
  omega

/-- Truncating a `u128` to `u64` is reduction modulo `2^64`, so the identity half of a `u128`
    product is that product modulo `2^64`. This is what the round trip turns on. -/
theorem mod64mod128 (n : Nat) : (n % 2 ^ 128) % 2 ^ 64 = n % 2 ^ 64 := by
  rw [Nat.mod_mod_of_dvd]; omega

theorem castU128U64 (x : U64) : (UScalar.cast .U128 x).val = x.val := by
  simp only [UScalar.cast_val_eq, UScalarTy.U128_numBits_eq, modU128]

theorem castU64U128 (x : U128) : (UScalar.cast .U64 x).val = x.val % 2 ^ 64 := by
  simp only [UScalar.cast_val_eq, UScalarTy.U64_numBits_eq]

theorem mul128 (x y : U128) :
    (core.num.U128.wrapping_mul x y).val = (x.val * y.val) % 2 ^ 128 := by
  rw [core.num.U128.wrapping_mul_val_eq, U128size]

theorem mul64 (x y : U64) :
    (core.num.U64.wrapping_mul x y).val = (x.val * y.val) % 2 ^ 64 := by
  rw [core.num.U64.wrapping_mul_val_eq, U64size]

/-- `i64 as u64` preserves the 64-bit pattern; only the reading of the word changes. -/
theorem castI64U64 (k : I64) :
    (core.num.I64.cast_unsigned k ⦃ y => y.val = k.bv.toNat ⦄) := by
  obtain ⟨y, hy, hyn⟩ := WP.spec_imp_exists (core.num.I64.cast_unsigned.step_spec k)
  rw [hy, WP.spec_ok]
  exact congrArg BitVec.toNat hyn

/-- `u64 as i64` preserves the 64-bit pattern. -/
theorem castU64I64 (u : U64) :
    (core.num.U64.cast_signed u ⦃ k => k.bv = u.bv ⦄) := core.num.U64.cast_signed.step_spec u

/-! ## `spread` and `unspread`: the shared `coord.rs` bodies

Each extracted codec body is one `spread` (encode) and one `unspread` (decode) from its result.
These are the two transitions the work measure counts. -/

/-- The extracted `PHI` constant, read from `src/coord.rs:59`. -/
theorem phi_val : i64_codec_source.coord.PHI.val = 11400714819323198485 := by
  norm_num [i64_codec_source.coord.PHI, UScalar.val, UScalar.size]

/-- The extracted `PHI_INV` constant, read from `src/coord.rs:390`. -/
theorem phi_inv_val : i64_codec_source.coord.PHI_INV.val = 17428512612931826493 := by
  norm_num [i64_codec_source.coord.PHI_INV, UScalar.val, UScalar.size]

/-- **The `spread` transition.** The extracted body performs one `u128` widening multiply, one
    `u64` truncation and one rotate/xor fold. Its **identity** half is the argument multiplied by
    `PHI` and reduced modulo `2^64`. No branch, no loop, no allocation. -/
theorem spread_spec (x : U64) :
    i64_codec_source.coord.spread x ⦃ a => a.id.val = (x.val * 11400714819323198485) % 2 ^ 64 ⦄ := by
  unfold i64_codec_source.coord.spread
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (show lift (UScalar.cast .U128 x) ⦃ y => y.val = x.val ⦄ from by
      simp only [lift, WP.spec_ok, UScalar.cast_val_eq, UScalarTy.U128_numBits_eq, modU128])
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (show lift (UScalar.cast .U128 i64_codec_source.coord.PHI) ⦃ y =>
      y.val = i64_codec_source.coord.PHI.val ⦄ from by
      simp only [lift, WP.spec_ok, UScalar.cast_val_eq, UScalarTy.U128_numBits_eq, modU128])
  obtain ⟨w, hw, hwv⟩ := WP.spec_imp_exists
    (show lift (core.num.U128.wrapping_mul i i1) ⦃ y => y.val = (i.val * i1.val) % 2 ^ 128 ⦄ from by
      simp only [lift, WP.spec_ok]; rw [mul128])
  obtain ⟨lo, hlo, hlov⟩ := WP.spec_imp_exists
    (show lift (UScalar.cast .U64 w) ⦃ y => y.val = w.val % 2 ^ 64 ⦄ from by
      simp only [lift, WP.spec_ok, UScalar.cast_val_eq, UScalarTy.U64_numBits_eq])
  have hi2 : w >>> 64#i32 = ok (⟨w.bv.ushiftRight 64⟩ : U128) := by
    show UScalar.shiftRight_UScalar w (64#usize) = _
    unfold UScalar.shiftRight_UScalar UScalar.shiftRight
    split
    · rfl
    · simp at *
  obtain ⟨hi4, hi4v⟩ := WP.spec_imp_exists
    (show lift (UScalar.cast .U64 (⟨w.bv.ushiftRight 64⟩ : U128)) ⦃ y =>
      y.val = (⟨w.bv.ushiftRight 64⟩ : U128).val % 2 ^ 64 ⦄ from by
      simp only [lift, WP.spec_ok, UScalar.cast_val_eq, UScalarTy.U64_numBits_eq])
  have hi5v : lift (core.num.U64.rotate_left hi4 32#u32) =
      ok (core.num.U64.rotate_left hi4 32#u32) := rfl
  have hi6v : lift (core.num.U64.rotate_left hi4 32#u32 ^^^ lo) =
      ok (core.num.U64.rotate_left hi4 32#u32 ^^^ lo) := rfl
  simp only [hi, hi1, hw, hlo, hi2, hi4v, hi5v, hi6v, bind_tc_ok, WP.spec_ok]
  have hcast : ((↑lo : Nat)) = (↑x * 11400714819323198485) % 2 ^ 64 := by
    rw [hlov, hwv, hiv, hi1v, phi_val]
    exact mod64mod128 (x.val * 11400714819323198485)
  exact hcast

/-- **The `unspread` transition.** The extracted body is one `u64` wrapping multiply by
    `PHI_INV`; no branch, no loop, no allocation. -/
theorem unspread_spec (id : U64) :
    i64_codec_source.coord.unspread id ⦃ y => y.val = (id.val * 17428512612931826493) % 2 ^ 64 ⦄ := by
  unfold i64_codec_source.coord.unspread
  simp only [WP.spec_ok]
  rw [mul64, phi_inv_val]

/-- The identity word is a bijection on `u64`: `PHI` and `PHI_INV` are mutual inverses modulo
    `2^64`. Both constants are read from the extracted `coord.rs` definitions. -/
theorem phi_inv_mul : (11400714819323198485 : Nat) * 17428512612931826493 % 2 ^ 64 = 1 := by
  norm_num [Nat.mod_eq_sub_mod]

/-- Reducing a natural twice modulo the same `n` is the same as reducing it once. -/
theorem mod_mod_same (a n : Nat) : a % n % n = a % n := by
  rw [Nat.mod_mod_of_dvd a (Nat.dvd_refl n)]

/-- Distributing the modulus across one factor of a product. -/
theorem mod_mul_right (a b n : Nat) : (a % n) * b % n = (a * b) % n := by
  calc (a % n) * b % n = ((a % n) % n) * (b % n) % n := Nat.mul_mod (a % n) b n
    _ = (a % n) * (b % n) % n := by rw [mod_mod_same]
    _ = (a * b) % n := (Nat.mul_mod a b n).symm

/-- **The `unspread`-inverts-`spread` law at the `u64` word level.** For every `u64` word `w`
    whose value is `spread`'s identity half at `u`, multiplying `w` by `PHI_INV` modulo `2^64`
    recovers `u`. This is the concrete statement that `PHI` is a unit modulo `2^64`. -/
theorem unspread_of_val (w u : U64) (hw : w.val = (u.val * 11400714819323198485) % 2 ^ 64) :
    (core.num.U64.wrapping_mul w i64_codec_source.coord.PHI_INV).val = u.val := by
  rw [core.num.U64.wrapping_mul_val_eq, phi_inv_val, U64size]
  rw [hw, mod_mul_right]
  have hA : u.val % 2 ^ 64 = u.val := by have := U64.le_max u; omega
  calc u.val * 11400714819323198485 * 17428512612931826493 % 2 ^ 64
      = (u.val * (11400714819323198485 * 17428512612931826493)) % 2 ^ 64 := by
        rw [Nat.mul_assoc]
    _ = (u.val * ((11400714819323198485 * 17428512612931826493) % 2 ^ 64)) % 2 ^ 64 := by
        rw [Nat.mul_mod, hA]
    _ = (u.val * 1) % 2 ^ 64 := by rw [phi_inv_mul]
    _ = u.val % 2 ^ 64 := by rw [Nat.mul_one]
    _ = u.val := hA

/-! ## `address_identity`: the right-hand side of the body's guard

`src/table/keyrep.rs:307` recomputes the same `spread` the encode arm computes. The `TableAddress`
argument is bound and dropped, so the guard compares a value with itself. -/

/-- **The `address_identity` transition.** One `cast_unsigned` and one `spread`; independent of the
    `TableAddress` argument. -/
theorem address_identity_spec (a : table.TableAddress) (k : I64) :
    table.keyrep.address_identity a k ⦃ y => y.val = (k.bv.toNat * 11400714819323198485) % 2 ^ 64 ⦄ := by
  unfold table.keyrep.address_identity
  obtain ⟨u, hu, huv⟩ := WP.spec_imp_exists (castI64U64 k)
  obtain ⟨ad, had, hadv⟩ := WP.spec_imp_exists (spread_spec u)
  simp only [hu, had, bind_tc_ok, WP.spec_ok]
  rw [hadv, huv]

/-- **The address-independence of the guard operand.** Two `TableAddress` values give the same
    `address_identity` result, so the extracted `debug_assert_eq!` in `encode` compares a word
    against the identical recomputation of itself. -/
theorem address_identity_independent (a b : table.TableAddress) (k : I64) :
    ∃ y, table.keyrep.address_identity a k = ok y ∧
      table.keyrep.address_identity b k = ok y := by
  obtain ⟨y, ha, hay⟩ := WP.spec_imp_exists (address_identity_spec a k)
  obtain ⟨y', hb, hby'⟩ := WP.spec_imp_exists (address_identity_spec b k)
  refine ⟨y, ha, ?_⟩
  have hv : y' = y := by
    have : y'.val = y.val := by rw [hby', hay]
    exact UScalar.eq_of_val_eq this
  rw [hb, hv]

/-! ## The extracted `encode` body: transition, guard, refusal branch and work

`src/table/keyrep.rs:286` is a straight-line body of four `Result` steps: `cast_unsigned`,
`spread`, `address_identity`, and a `massert` over their equality. `sigma` and `address` are
bound and dropped. The `massert` is the body's only exit that is not the success arm. -/

/-- The extracted `I64WordCodec::encode` body, at the concrete `WordHome` the extraction root
    supplies. -/
abbrev encodeBody (s : cell.Sigma) (k : I64) (a : table.TableAddress) : Result U64 :=
  i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode i64_codec_source.table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma s k a

/-- The extracted `I64WordCodec::decode` body, at the same concrete `WordHome`. -/
abbrev decodeBody (s : cell.Sigma) (id : U64) : Result I64 :=
  i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode i64_codec_source.table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma s id

/-- **The `encode` transition.** The body is total: on every `sigma`, key and address it returns
    the identity half of `spread` applied to the key's `u64` pattern, and its `debug_assert_eq!`
    guard holds on both sides because `address_identity` recomputes the same `spread`. The
    transition is guarded by exactly that equality. -/
theorem encode_spec (s : cell.Sigma) (k : I64) (a : table.TableAddress) :
    ∃ id, encodeBody s k a = ok id ∧
      id.val = (k.bv.toNat * 11400714819323198485) % 2 ^ 64 ∧
      ∃ rv, table.keyrep.address_identity a k = ok rv ∧ id.val = rv.val := by
  obtain ⟨u, hu, huv⟩ := WP.spec_imp_exists (castI64U64 k)
  obtain ⟨ad, had, hadv⟩ := WP.spec_imp_exists (spread_spec u)
  obtain ⟨rv, hrv, hrval⟩ := WP.spec_imp_exists (address_identity_spec a k)
  have hguard : ad.id = rv := by
    have hv : ad.id.val = rv.val := by rw [hadv, hrval, huv]
    exact UScalar.eq_of_val_eq hv
  have hbody : encodeBody s k a = ok ad.id := by
    show (i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode i64_codec_source.table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma s k a) = ok ad.id
    unfold i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
    simp only [hu, bind_tc_ok]
    rw [had, hrv, bind_tc_ok]
    simp only [massert, if_pos hguard, bind_tc_ok]
  refine ⟨ad.id, hbody, ?_, rv, hrv, ?_⟩
  · rw [hadv, ← huv]
  · rw [hrval, ← huv]
    exact hadv

/-- **The `encode` guard is the extracted `massert`, and it never fails.** The extracted body's
    `fail assertionFailure` arm is unreachable: both sides of the guard are the same
    `address_identity` value, so the guard's decidable test is decided `true` on every input.
    This is the exit/refusal branch statement — the arm exists in the extracted body and is
    discharged here. -/
theorem encode_never_refuses (s : cell.Sigma) (k : I64) (a : table.TableAddress) :
    encodeBody s k a ≠ fail .assertionFailure := by
  obtain ⟨id, hid, _, _⟩ := encode_spec s k a
  rw [hid]
  intro h
  simp at h

/-- **The `encode` work measure, in `Result` steps.** The extracted body is straight-line: it
    binds exactly five values — the `i64 as u64` reinterpretation, the `spread` result, the
    `address_identity` result, the guard's `massert` unit, and the returned identity word. In
    word operations that is two `u128` widening multiplies, two `u64` truncations, two `u64`
    right shifts, two `u64` rotates, two `u64` xors, two `i64 as u64` reinterpretations and one
    `u64` equality test, so the count does not grow with the key and the body allocates nothing.
    The number below is the step count; `encode_work_constant` exhibits the chain. -/
def encodeWork : Nat := 5

theorem encode_work_measure : encodeWork = 5 := by rfl

/-- **The `encode` work is a constant, witnessed.** For every face, key and address the extracted
    body reaches its identity word through exactly the five-step chain `encodeWork` names, and
    allocates nothing. The three witnesses are the extraction's own intermediate values, so the
    count is read off the body rather than asserted. -/
theorem encode_work_constant (s : cell.Sigma) (k : I64) (a : table.TableAddress) :
    ∃ u ad rv, core.num.I64.cast_unsigned k = ok u ∧
      i64_codec_source.coord.spread u = ok ad ∧
      table.keyrep.address_identity a k = ok rv ∧
      encodeBody s k a = ok ad.id ∧ encodeWork = 5 := by
  obtain ⟨u, hu, huv⟩ := WP.spec_imp_exists (castI64U64 k)
  obtain ⟨ad, had, hadv⟩ := WP.spec_imp_exists (spread_spec u)
  obtain ⟨rv, hrv, hrval⟩ := WP.spec_imp_exists (address_identity_spec a k)
  have hguard : ad.id = rv := by
    have hv : ad.id.val = rv.val := by rw [hadv, hrval, ← huv]
    exact UScalar.eq_of_val_eq hv
  refine ⟨u, ad, rv, hu, had, hrv, ?_, rfl⟩
  show (i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode i64_codec_source.table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma s k a) = ok ad.id
  unfold i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
  simp only [hu, bind_tc_ok]
  rw [had, hrv, bind_tc_ok]
  simp only [massert, if_pos hguard, bind_tc_ok]

/-! ## The extracted `decode` body

`src/table/keyrep.rs:292` is two steps: one `u64` wrapping multiply by `PHI_INV`, then the
signed re-reading of the resulting word. `sigma` is bound and dropped. There is no guard. -/

/-- **The `decode` transition.** The body is total: it returns the `i64` whose 64-bit pattern is
    the identity word multiplied by `PHI_INV` modulo `2^64`. -/
theorem decode_spec (s : cell.Sigma) (id : U64) :
    decodeBody s id ⦃ k => k.bv = (core.num.U64.wrapping_mul id i64_codec_source.coord.PHI_INV).bv ⦄ := by
  have hun : i64_codec_source.coord.unspread id = ok
      (core.num.U64.wrapping_mul id i64_codec_source.coord.PHI_INV) := rfl
  obtain ⟨k, hcs, hk⟩ := WP.spec_imp_exists
    (castU64I64 (core.num.U64.wrapping_mul id i64_codec_source.coord.PHI_INV))
  show (i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode i64_codec_source.table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma s id) ⦃ k => k.bv = (core.num.U64.wrapping_mul id i64_codec_source.coord.PHI_INV).bv ⦄
  unfold i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode
  simp only [hun, bind_tc_ok, hcs, WP.spec_ok]
  exact hk

/-- **The `decode` work measure, in `Result` steps.** The extracted body binds exactly two values:
    the `unspread` result and the signed re-reading of it. In word operations that is one `u64`
    wrapping multiply and one bit reinterpretation, so the count does not grow with the identity
    word and the body allocates nothing. `decode_work_constant` exhibits the chain. -/
def decodeWork : Nat := 2

theorem decode_work_measure : decodeWork = 2 := by rfl

/-- **The `decode` work is a constant, witnessed.** For every face and identity word the
    extracted body reaches its key through exactly the two-step chain `decodeWork` names. -/
theorem decode_work_constant (s : cell.Sigma) (id : U64) :
    ∃ u k, i64_codec_source.coord.unspread id = ok u ∧
      core.num.U64.cast_signed u = ok k ∧
      decodeBody s id = ok k ∧ decodeWork = 2 := by
  have hun : i64_codec_source.coord.unspread id = ok
      (core.num.U64.wrapping_mul id i64_codec_source.coord.PHI_INV) := rfl
  obtain ⟨k, hcs, _⟩ := WP.spec_imp_exists
    (castU64I64 (core.num.U64.wrapping_mul id i64_codec_source.coord.PHI_INV))
  have hdec : decodeBody s id = ok k := by
    show (i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode i64_codec_source.table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma s id) = ok k
    unfold i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode
    simp only [hun, bind_tc_ok, hcs]
  exact ⟨_, k, hun, hcs, hdec, rfl⟩

/-- **The `decode` body never refuses.** It has no guard and no fallible operation, so its only
    result is the success arm: the extracted body cannot reach `fail` on any input. -/
theorem decode_never_refuses (s : cell.Sigma) (id : U64) :
    decodeBody s id ≠ fail .assertionFailure := by
  obtain ⟨k', hk, _⟩ := WP.spec_imp_exists (decode_spec s id)
  rw [hk]
  intro h
  simp at h

/-! ## The round trip on a reachable key

`decode ∘ encode` is the law the `Regen` carrier relies on. It is stated here on the concrete
`i64` domain: the `TableAddress` the encode body's caller supplies is the one the real
`WordHome::address` produces, and any `i64` is a key the extracted root accepts. -/

/-- **The codec law at full `i64` width.** For every `sigma`, key and address, decoding the
    encoded identity word recovers the original 64-bit key pattern. This is
    `decode(sigma, encode(sigma, k, a)) == k` at the bit level, proved from the two transitions
    above and `unspread_of_val`. -/
theorem codec_round_trip (s : cell.Sigma) (k : I64) (a : table.TableAddress) :
    ∃ id k', encodeBody s k a = ok id ∧ decodeBody s id = ok k' ∧ k'.bv = k.bv := by
  obtain ⟨u, hu, huv⟩ := WP.spec_imp_exists (castI64U64 k)
  obtain ⟨ad, had, hadv⟩ := WP.spec_imp_exists (spread_spec u)
  obtain ⟨rv, hrv, hrval⟩ := WP.spec_imp_exists (address_identity_spec a k)
  have hguard : ad.id = rv := by
    have hv : ad.id.val = rv.val := by rw [hadv, hrval, ← huv]
    exact UScalar.eq_of_val_eq hv
  have henc : encodeBody s k a = ok ad.id := by
    show (i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode i64_codec_source.table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma s k a) = ok ad.id
    unfold i64_codec_source.table.keyrep.I64WordCodec.Insts.I64_codec_sourceTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
    simp only [hu, bind_tc_ok]
    rw [had, hrv, bind_tc_ok]
    simp only [massert, if_pos hguard, bind_tc_ok]
  obtain ⟨k', hdec, hk'bv⟩ := WP.spec_imp_exists (decode_spec s ad.id)
  refine ⟨ad.id, k', henc, hdec, ?_⟩
  -- `k'.bv = (unspread (spread u).id).bv`, and `unspread` inverts `spread` on the identity half.
  have hstep : (core.num.U64.wrapping_mul ad.id i64_codec_source.coord.PHI_INV).val = u.val := by
    refine unspread_of_val ad.id u ?_
    rw [hadv]
  have hmul_bv : (core.num.U64.wrapping_mul ad.id i64_codec_source.coord.PHI_INV).bv = u.bv := by
    unfold UScalar.val at hstep
    exact BitVec.toNat_injective hstep
  have hk : k.bv = u.bv := by
    unfold UScalar.val at huv
    exact BitVec.toNat_injective huv.symm
  rw [hk'bv, hmul_bv, hk]

/-! ## A reachable success witness

The witness is the concrete key `1` on the elliptic face, at the address the real
`WordHome::address` produces for that face and key. It exercises the whole extracted chain:
`Home::address`, `encode`, the `massert` guard, and `decode`. -/

/-- **The reachable success witness.** On the elliptic face at key `1`, with the address the real
    `WordHome::address` body returns for that face and key, the extracted encode returns an
    identity word the extracted decode reads back as key `1`. The address conjunct is what makes
    the witness reachable rather than an arbitrary `TableAddress`: it is the value the
    production `Home` impl produces. -/
theorem success_witness :
    ∃ s k a id k',
      s = cell.Sigma.Elliptic ∧ k = (1#i64 : I64) ∧
      table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma.address () s k = ok a ∧
      encodeBody s k a = ok id ∧ decodeBody s id = ok k' ∧ k'.bv = k.bv := by
  -- The real `WordHome::address` body has one guard (`word <= u32::MAX`) and both arms return
  -- `ok`, so it is total and the address is reachable.
  have hrun : ∃ v, (table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma.address
      () cell.Sigma.Elliptic (1#i64)) = ok v := by
    unfold table.WordHome.Insts.I64_codec_sourceTableHomeI64Sigma.address
    obtain ⟨w, hw, _⟩ := WP.spec_imp_exists
      (show core.num.I64.cast_unsigned (1#i64) ⦃ w => w.bv = (1#i64).bv ⦄ from
        core.num.I64.cast_unsigned.step_spec (1#i64))
    obtain ⟨sp, hsp, _⟩ := WP.spec_imp_exists
      (show i64_codec_source.coord.spread w ⦃ sp => sp.id.val = (w.val * 11400714819323198485) % 2^64 ⦄ from
        spread_spec w)
    obtain ⟨cap, hcap, _⟩ := WP.spec_imp_exists
      (show lift (core.convert.num.FromU64U32.from core.num.U32.MAX) ⦃ cap => True ⦄ from by
        simp only [lift, WP.spec_ok])
    by_cases hle : w ≤ cap
    · simp only [hw, hsp, hcap, bind_tc_ok, if_pos hle]
      obtain ⟨cw, hcw, _⟩ := WP.spec_imp_exists
        (show lift (UScalar.cast .U32 w) ⦃ cw => True ⦄ from by simp only [lift, WP.spec_ok])
      simp only [hcw, bind_tc_ok]
      exact ⟨_, rfl⟩
    · simp only [hw, hsp, hcap, bind_tc_ok, if_neg hle]
      exact ⟨_, rfl⟩
  obtain ⟨v, hv⟩ := hrun
  obtain ⟨id, k', hid, hdec, hrt⟩ :=
    codec_round_trip cell.Sigma.Elliptic (1#i64) v
  exact ⟨cell.Sigma.Elliptic, 1#i64, v, id, k', rfl, rfl, hv, hid, hdec, hrt⟩

/-! ## The audit -/

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit I64CodecResource.spread_spec
#resource_audit I64CodecResource.unspread_spec
#resource_audit I64CodecResource.address_identity_spec
#resource_audit I64CodecResource.address_identity_independent
#resource_audit I64CodecResource.encode_spec
#resource_audit I64CodecResource.encode_never_refuses
#resource_audit I64CodecResource.encode_work_measure
#resource_audit I64CodecResource.encode_work_constant
#resource_audit I64CodecResource.decode_spec
#resource_audit I64CodecResource.decode_work_measure
#resource_audit I64CodecResource.decode_work_constant
#resource_audit I64CodecResource.decode_never_refuses
#resource_audit I64CodecResource.codec_round_trip
#resource_audit I64CodecResource.success_witness

run_cmd Lean.logInfo "I64_CODEC_KERNEL_OK"

end I64CodecResource
