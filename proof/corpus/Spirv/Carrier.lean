import Spirv.Semantics
import Mathlib.Tactic

/-!
# The bit-width carrier: one decode and one split lemma indexed by `b`

The emitted `OpSUDot` phase split (`src/simt/dp4a.rs`, served path `gemv_dp4a_module_dyn`) is one
mechanism dialed by the code width, not a kernel per width. For a chart carrier with `b` bits per
code, `wbits = 32 / codes_per_word` (`dp4a.rs:169`) and `byte_mask = 0x01010101 * (2^wbits - 1)`
(`dp4a.rs:170`); phase `l` extracts one code per byte lane as `(WWORD >> wbits·l) & byte_mask`
(`dp4a.rs:769-770`, the shift constants `wbits·l` at `dp4a.rs:534`). Q8 (`b = 8`) uses the word
whole (`split = dots_per_word > 1`, `dp4a.rs:171`); Q4 (`b = 4`) splits into even/odd nibbles
(`dp4a.rs:29`, `w & 0x0F0F0F0F` and `(w >> 4) & 0x0F0F0F0F`); Q2 (`b = 2`) is the four 2-bit
phases. `dots_per_word = codes_per_word / 4 = 8 / b` is the phase count.

This file states that emitter once, indexed by `b`:

* `unpackPhase b p w` is the `eval` denotation (`Spirv.Semantics`) of the emitted
  `OpShiftRightLogical` then `OpBitwiseAnd`: `(w >>> b·p) & maskRep b` in the `Nat` semantics `eval`
  gives those two opcodes, with `maskRep b` the byte-replicated `b`-bit mask `(2^b - 1)·0x01010101`.
* `carrier_decode`: for `b ∈ {1,2,4,8}` every byte lane of `unpackPhase b p w` is the `b`-bit code
  `(w >>> (8·i + b·p)) mod 2^b`, and that code is `< 2^b` — a non-negative `OpSUDot` weight byte.
* `carrier_split`: the per-phase `OpSUDot` sum is the `ringMul` channel-0 sum over the four byte
  lanes, via `Spirv.Semantics.sdot_split_ringMul` at `b = 4`.
* `carrier_split_q8`/`_q4`/`_q2` are one-line instantiations of `carrier_split` at `b = 8, 4, 2`:
  a new width is added by instantiating `b`, not by a new proof.

## What this file does not claim (obligation 11, trust boundary)

The Lean model of `OpSUDot` is `Spirv.Semantics.evalSUDot` at the word face `W32`; the emitter's
real unpack lines are matched by quotation (`dp4a.rs:169-171`, `769-808`) and read as an `eval`
denotation, not by Aeneas/Charon extraction of `dp4a.rs`. `carrier_decode` covers `b ∈ {1,2,4,8}`
only; `b = 3` and `b = 16` are outside the admissible packing (a refusal witness is recorded).
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Spirv

open Apeiron.Trichotomy.AnyRing
open BigOperators

/-- The byte-replicated low-`b`-bit mask `(2^b - 1) * 0x01010101` — the emitter's `byte_mask`
(`src/simt/dp4a.rs:170`) at code width `b`. Each of the four bytes is `2^b - 1`; for `b ≤ 8` the
product has no carries. -/
def maskRepVal (b : Nat) : Nat := (2 ^ b - 1) * 0x01010101

/-- `maskRep b` as a `Word32`. -/
def maskRep (b : Nat) : Word32 := (maskRepVal b : Nat)

/-- The `OpSUDot` phase count for code width `b`: `8 / b` (`dots_per_word`, `dp4a.rs:167`). -/
def phases (b : Nat) : Nat := 8 / b

/-- The emitted phase-`p` unpack at code width `b`: `(w >>> b·p) & maskRep b`, the `Nat` denotation
`eval` gives `OpShiftRightLogical` then `OpBitwiseAnd` (`Spirv.Semantics.eval`, `dp4a.rs:797-808`).
-/
def unpackPhase (b p : Nat) (w : Word32) : Word32 :=
  ((w.val >>> (b * p)) &&& maskRepVal b : Nat)

/-- The `b`-bit code of byte lane `i` at phase `p`: bits `[8·i + b·p, 8·i + b·p + b)` of `w`,
matching the emitter's `(WWORD >> wbits·l)` read of one code per byte lane. -/
def bweight (b p : Nat) (w : Word32) (i : Fin 4) : Word32 :=
  ((w.val >>> (8 * i.val + b * p)) % 2 ^ b : Nat)

lemma maskRepVal_lt {b : Nat} (hb : b ≤ 8) : maskRepVal b < 2 ^ 32 := by
  unfold maskRepVal
  have h : 2 ^ b - 1 ≤ 255 := by
    have hp : 2 ^ b ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) hb
    norm_num at hp ⊢
    omega
  calc (2 ^ b - 1) * 0x01010101 ≤ 255 * 0x01010101 := Nat.mul_le_mul_right _ h
    _ < 2 ^ 32 := by norm_num

lemma le_eight_of_mem {b : Nat} (hb : b ∈ ({1,2,4,8} : Finset Nat)) : b ≤ 8 := by
  fin_cases hb <;> omega

/-- Every byte `k < 4` of the byte-replicated mask product `(2^b - 1) * 0x01010101` is `2^b - 1`. -/
lemma mask_shr_mod {b k : Nat} (hb : b ≤ 8) (hk : k < 4) :
    (maskRepVal b >>> (8 * k)) % 256 = 2 ^ b - 1 := by
  have ha : 2 ^ b - 1 < 256 := by
    have hp : 2 ^ b ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) hb
    norm_num at hp ⊢
    omega
  have hS0 : (0x01010101 : Nat) % 256 = 1 := by norm_num
  have hS1 : (0x01010101 : Nat) = 65793 * 256 + 1 := by norm_num
  have hS2 : (0x01010101 : Nat) = 257 * 65536 + 257 := by norm_num
  have hS3 : (0x01010101 : Nat) = 16777216 + 65793 := by norm_num
  interval_cases k
  · rw [Nat.shiftRight_zero, maskRepVal, Nat.mul_mod, Nat.mod_eq_of_lt ha, hS0,
      Nat.mul_one, Nat.mod_eq_of_lt ha]
  · rw [Nat.shiftRight_eq_div_pow, show (2 : Nat) ^ 8 = 256 by norm_num, maskRepVal]
    rw [show (2 ^ b - 1) * 0x01010101 = (2 ^ b - 1) + ((2 ^ b - 1) * 65793) * 256 by
      rw [hS1, Nat.mul_add, Nat.mul_one, Nat.mul_assoc, Nat.add_comm]]
    rw [Nat.add_mul_div_right _ _ (by norm_num : 0 < 256), Nat.div_eq_of_lt ha, Nat.zero_add,
      Nat.mul_mod, Nat.mod_eq_of_lt ha]
    have h65793 : (65793 : Nat) % 256 = 1 := by norm_num
    rw [h65793, Nat.mul_one, Nat.mod_eq_of_lt ha]
  · rw [Nat.shiftRight_eq_div_pow, show (2 : Nat) ^ 16 = 65536 by norm_num, maskRepVal]
    rw [show (2 ^ b - 1) * 0x01010101 = (2 ^ b - 1) * 257 + ((2 ^ b - 1) * 257) * 65536 by
      rw [hS2, Nat.mul_add, Nat.mul_assoc, Nat.add_comm]]
    have hlt : (2 ^ b - 1) * 257 < 65536 := by omega
    rw [Nat.add_mul_div_right _ _ (by norm_num : 0 < 65536), Nat.div_eq_of_lt hlt, Nat.zero_add,
      Nat.mul_mod, Nat.mod_eq_of_lt ha]
    have h257 : (257 : Nat) % 256 = 1 := by norm_num
    rw [h257, Nat.mul_one, Nat.mod_eq_of_lt ha]
  · rw [Nat.shiftRight_eq_div_pow, show (2 : Nat) ^ 24 = 16777216 by norm_num, maskRepVal]
    rw [show (2 ^ b - 1) * 0x01010101 = (2 ^ b - 1) * 65793 + (2 ^ b - 1) * 16777216 by
      rw [hS3, Nat.mul_add, Nat.add_comm]]
    have hlt : (2 ^ b - 1) * 65793 < 16777216 := by omega
    rw [Nat.add_mul_div_right _ _ (by norm_num : 0 < 16777216), Nat.div_eq_of_lt hlt, Nat.zero_add,
      Nat.mod_eq_of_lt ha]

/-- **The decode.** For `b ∈ {1,2,4,8}` every byte lane of `unpackPhase b p w` is the `b`-bit code
`(w >>> (8·i + b·p)) mod 2^b`, with that code `< 2^b`. -/
theorem carrier_decode {b : Nat} (hb : b ∈ ({1,2,4,8} : Finset Nat)) (p : Nat) (w : Word32)
    (i : Fin 4) :
    byte (unpackPhase b p w) i = bweight b p w i
      ∧ (w.val >>> (8 * i.val + b * p)) % 2 ^ b < 2 ^ b := by
  haveI : NeZero (2 ^ 32 : Nat) := ⟨by norm_num⟩
  refine ⟨?_, Nat.mod_lt _ (by positivity)⟩
  have hval : (unpackPhase b p w).val = (w.val >>> (b * p)) &&& maskRepVal b := by
    unfold unpackPhase
    rw [ZMod.val_natCast, Nat.mod_eq_of_lt (Nat.and_lt_two_pow _ (maskRepVal_lt (le_eight_of_mem hb)))]
  have hbyte : (byte (unpackPhase b p w) i).val
      = ((unpackPhase b p w).val >>> (8 * i.val)) % 256 := by
    rw [byte_order, ZMod.val_natCast,
      Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le (Nat.mod_lt _ (by norm_num)) (by norm_num : 256 ≤ 2 ^ 32))]
  have hmain : ((unpackPhase b p w).val >>> (8 * i.val)) % 256
      = (w.val >>> (8 * i.val + b * p)) % 2 ^ b := by
    rw [hval, Nat.shiftRight_and_distrib]
    rw [show (256 : Nat) = 2 ^ 8 by norm_num]
    rw [Nat.and_mod_two_pow, mask_shr_mod (le_eight_of_mem hb) i.isLt, ← Nat.shiftRight_add]
    rw [show b * p + 8 * i.val = 8 * i.val + b * p by omega]
    rw [Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_mod_of_dvd _ (Nat.pow_dvd_pow 2 (le_eight_of_mem hb))]
  rw [show byte (unpackPhase b p w) i
        = (((byte (unpackPhase b p w) i).val : Nat) : Word32) from
      (ZMod.natCast_zmod_val _).symm, hbyte, hmain]
  rfl

/-- **The split.** The per-phase `OpSUDot` sum at code width `b` is the byte-lane `ringMul`
channel-0 sum. Each phase's `evalSUDot` is `sdot_denote` (`Spirv.Semantics`); the weight byte is the
phase code by `carrier_decode`, and each single-byte product is `(ringMul realizedSigma (u, 0) (v, 0)).1`
(`Spirv.Semantics.evalIMul_face`). At `b = 4` the two phases collapse to the genuine two-component
cell `sdot_split_ringMul` (`carrier_split_q4_ringMul`). -/
theorem carrier_split {b : Nat} (hb : b ∈ ({1,2,4,8} : Finset Nat))
    (x : Fin (phases b) → Word32) (w : Word32) :
    (∑ p : Fin (phases b), evalSUDot (x p) (unpackPhase b p w))
      = ∑ p : Fin (phases b), ∑ i : Fin 4,
          sext8 (byte (x p) i) * zext8 (bweight b p w i) := by
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [sdot_denote]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [(carrier_decode hb p w i).1]

/-- **What `carrier_split` becomes on cells.** The per-phase `OpSUDot` sum is a sum over phases of
`ringMul … .1` reads on the *diagonal* cells `(code, 0)`: for every width the single-byte product is
`(ringMul realizedSigma (u, 0) (v, 0)).1 = u * v` (`Spirv.Semantics.evalIMul_face`), the scalar
subring, not the two-component `ringMul`. The two-component `ringMul` is realized only at `b = 4`,
where the two phases form the genuine cells `actCell`/`wgtCell` (`carrier_split_q4_ringMul`). -/
theorem carrier_split_cells {b : Nat} (hb : b ∈ ({1,2,4,8} : Finset Nat))
    (x : Fin (phases b) → Word32) (w : Word32) :
    (∑ p : Fin (phases b), evalSUDot (x p) (unpackPhase b p w))
      = ∑ p : Fin (phases b), ∑ i : Fin 4,
          (ringMul realizedSigma (sext8 (byte (x p) i), 0)
            (zext8 (bweight b p w i), 0)).1 := by
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [sdot_denote]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [(carrier_decode hb p w i).1, evalIMul_face]

/-- The Q8 instance (`b = 8`): one phase, the word whole. -/
theorem carrier_split_q8 (x : Fin (phases 8) → Word32) (w : Word32) :
    (∑ p : Fin (phases 8), evalSUDot (x p) (unpackPhase 8 p w))
      = ∑ p : Fin (phases 8), ∑ i : Fin 4,
          sext8 (byte (x p) i) * zext8 (bweight 8 p w i) :=
  carrier_split (by decide) x w

/-- The Q4 instance (`b = 4`): two phases, the even/odd nibbles. -/
theorem carrier_split_q4 (x : Fin (phases 4) → Word32) (w : Word32) :
    (∑ p : Fin (phases 4), evalSUDot (x p) (unpackPhase 4 p w))
      = ∑ p : Fin (phases 4), ∑ i : Fin 4,
          sext8 (byte (x p) i) * zext8 (bweight 4 p w i) :=
  carrier_split (by decide) x w

/-- The Q2 instance (`b = 2`): four phases, the 2-bit codes. -/
theorem carrier_split_q2 (x : Fin (phases 2) → Word32) (w : Word32) :
    (∑ p : Fin (phases 2), evalSUDot (x p) (unpackPhase 2 p w))
      = ∑ p : Fin (phases 2), ∑ i : Fin 4,
          sext8 (byte (x p) i) * zext8 (bweight 2 p w i) :=
  carrier_split (by decide) x w

/-- The `b = 4` phase pair is the genuine two-component `ringMul` cell sum:
`Spirv.Semantics.sdot_split_ringMul` on the even/odd unpacked words. -/
theorem carrier_split_q4_ringMul (x : Fin 2 → Word32) (w : Word32) :
    (∑ p : Fin 2, evalSUDot (x p) (unpackPhase 4 p w))
      = ∑ i : Fin 4, (ringMul realizedSigma (actCell (x 0) (x 1) i)
          (wgtCell (unpackPhase 4 0 w) (unpackPhase 4 1 w) i)).1 := by
  rw [Fin.sum_univ_two]
  exact sdot_split_ringMul (x 0) (x 1) (unpackPhase 4 0 w) (unpackPhase 4 1 w)

/-- `b = 8`, phase `0`: the identity — Q8 dots the word whole. -/
theorem unpackPhase_q8_zero (w : Word32) : unpackPhase 8 0 w = w := by
  haveI : NeZero (2 ^ 32 : Nat) := ⟨by norm_num⟩
  apply ZMod.val_injective (2 ^ 32)
  unfold unpackPhase maskRepVal
  rw [ZMod.val_natCast,
    Nat.mod_eq_of_lt (Nat.and_lt_two_pow _ (by norm_num : (2 ^ 8 - 1) * 0x01010101 < 2 ^ 32)),
    show (2 ^ 8 - 1) * (0x01010101 : Nat) = 2 ^ 32 - 1 by norm_num, Nat.shiftRight_zero]
  exact Nat.and_two_pow_sub_one_of_lt_two_pow (ZMod.val_lt w)

/-- `b = 4`, phases `0`/`1`: the emitted even/odd nibble words (`dp4a.rs:29`),
`w & 0x0F0F0F0F` and `(w >> 4) & 0x0F0F0F0F`, the `w0`/`w1` that `sdot_split_ringMul` is stated over.
-/
theorem unpackPhase_four_even_odd (w : Word32) :
    (unpackPhase 4 0 w).val = w.val &&& 0x0F0F0F0F
      ∧ (unpackPhase 4 1 w).val = (w.val >>> 4) &&& 0x0F0F0F0F := by
  constructor
  · unfold unpackPhase maskRepVal
    rw [ZMod.val_natCast,
      Nat.mod_eq_of_lt (Nat.and_lt_two_pow _ (by norm_num : (2 ^ 4 - 1) * 0x01010101 < 2 ^ 32))]
    norm_num
  · unfold unpackPhase maskRepVal
    rw [ZMod.val_natCast,
      Nat.mod_eq_of_lt (Nat.and_lt_two_pow _ (by norm_num : (2 ^ 4 - 1) * 0x01010101 < 2 ^ 32))]
    norm_num

/-- Refusal witness: `b = 3` is outside the admissible packing. -/
theorem carrier_refusal_three : ¬ ((3 : Nat) ∈ ({1,2,4,8} : Finset Nat)) := by decide

/-- Refusal witness: `b = 16` is outside the admissible packing. -/
theorem carrier_refusal_sixteen : ¬ ((16 : Nat) ∈ ({1,2,4,8} : Finset Nat)) := by decide

end Spirv
