import ApeironWord

/-!
# Wrapping words as `ZMod (2^64)`

The denotation that connects the extracted machine-word kernels to the ring the canonical
algebra speaks about. `denote` reads an `i64` as its **bit pattern** (unsigned), so wrapping
arithmetic becomes the corresponding operation in `ZMod (2 ^ 64)`; the signed `i64` view is a
separate interpretation and is not used here.

`denote` is deliberately left opaque to `simp` (no `[simp]` equation unfolding it), so the
wrapping lemmas below fire first and the atomic `denote x` terms reach `ring` as variables.
`WordRefinement` composes these with `Apeiron.Trichotomy.AnyRing`.
-/

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-- The bit pattern of an `I64`, as an element of `ZMod (2 ^ 64)`. -/
def denote (x : I64) : ZMod (2 ^ 64) := (x.bv.toNat : ZMod (2 ^ 64))

theorem denote_def (x : I64) : denote x = (x.bv.toNat : ZMod (2 ^ 64)) := rfl

@[simp] theorem denote_zero : denote (0#i64) = (0 : ZMod (2 ^ 64)) := by simp [denote]

@[simp] theorem denote_one : denote (1#i64) = (1 : ZMod (2 ^ 64)) := by simp [denote]

/-- The extracted integer cast is the identity under `denote`. -/
theorem denote_from_i64 (v : I64) :
    (do let r ← I64.Insts.Apeiron_wordScalarRing.from_i64 v; ok (denote r)) = ok (denote v) := by
  simp [I64.Insts.Apeiron_wordScalarRing.from_i64]

/-- The extracted `i64` addition is ring addition under `denote`. -/
@[simp] theorem denote_wrapping_add (x y : I64) :
    denote (core.num.I64.wrapping_add x y) = denote x + denote y := by
  simp only [denote, core.num.I64.wrapping_add_bv_eq, BitVec.toNat_add, ZMod.natCast_mod,
    Nat.cast_add]

/-- The extracted `i64` subtraction is ring subtraction under `denote`. -/
@[simp] theorem denote_wrapping_sub (x y : I64) :
    denote (core.num.I64.wrapping_sub x y) = denote x - denote y := by
  simp only [denote, core.num.I64.wrapping_sub_bv_eq, BitVec.toNat_sub, ZMod.natCast_mod,
    Nat.cast_add]
  rw [Nat.cast_sub (Nat.le_of_lt y.bv.isLt), ZMod.natCast_self]
  ring

/-- The extracted `i64` multiplication is ring multiplication under `denote`. -/
@[simp] theorem denote_wrapping_mul (x y : I64) :
    denote (core.num.I64.wrapping_mul x y) = denote x * denote y := by
  simp only [denote, core.num.I64.wrapping_mul_bv_eq, BitVec.toNat_mul, ZMod.natCast_mod,
    Nat.cast_mul]

end apeiron_word
