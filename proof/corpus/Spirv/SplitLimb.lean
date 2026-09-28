import Apeiron.Trichotomy.AnyRing
import Mathlib.Data.Int.Basic
import Mathlib.Tactic

/-!
# GPU Foundation: Split-Limb Fixed-Point Arithmetic and Reduction Exactness

Formal verification of GPU integer primitives for Q32.32 fixed-point inference:
1. **Split-Limb Product**:
   Representing 64-bit words as pairs of 32-bit limbs `(hi, lo)`.
   The cross-product `splitMul a b` expands into 4 partial products
   `hi*hi * 2^64 + (hi*lo + lo*hi) * 2^32 + lo*lo`, matching the composition
   of hardware 32-bit multipliers (`OpSMulExtended` / `OpUMulExtended`).
2. **Fixed-Point Rounding**:
   The rounding shift `roundShift32(P) = (P + 2^31) / 2^32` extracts the Q32.32
   fractional quotient.
3. **Exactness of Integer Reduction Trees**:
   Because addition over `ℤ` is strictly associative and commutative,
   parallel tree reductions across warps/workgroups (`OpGroupNonUniformIAdd`)
   are bit-identical to sequential prefix accumulations:
   `(x₀ + x₁) + (x₂ + x₃) = ((x₀ + x₁) + x₂) + x₃`.
   Unlike floating-point summation where order-dependent rounding produces non-deterministic
   drift, integer fixed-point reduction has zero drift (`tree - seq = 0`).
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace Spirv.SplitLimb

/-- Two 32-bit limbs representing a 64-bit integer. -/
structure Limb2 where
  hi : ℤ
  lo : ℤ
deriving Repr, DecidableEq

/-- Value of two limbs: `hi * 2^32 + lo`. -/
def fromLimbs (l : Limb2) : ℤ :=
  l.hi * (2 ^ 32) + l.lo

/-- 128-bit split-limb integer product:
`hi * hi * 2^64 + (hi * lo + lo * hi) * 2^32 + lo * lo`. -/
def splitMul (a b : Limb2) : ℤ :=
  a.hi * b.hi * (2 ^ 64) + (a.hi * b.lo + a.lo * b.hi) * (2 ^ 32) + a.lo * b.lo

/-- Split-limb multiplication exactly matches the product of the combined 64-bit integers. -/
theorem splitMul_eq (a b : Limb2) :
    splitMul a b = fromLimbs a * fromLimbs b := by
  dsimp [splitMul, fromLimbs]
  ring

/-- Round-to-nearest right shift by 32 bits: `(x + 2^31) / 2^32`. -/
def roundShift32 (x : ℤ) : ℤ :=
  (x + 2 ^ 31) / (2 ^ 32)

/-- Bound on rounding perturbation: `|roundShift32(x) * 2^32 - x| ≤ 2^31`. -/
theorem roundShift32_perturbation (x : ℤ) :
    -(2 ^ 31) ≤ roundShift32 x * (2 ^ 32) - x ∧ roundShift32 x * (2 ^ 32) - x ≤ 2 ^ 31 := by
  have hcomm : roundShift32 x * (2 ^ 32) = (2 ^ 32) * roundShift32 x := mul_comm (roundShift32 x) (2 ^ 32)
  rw [hcomm]
  dsimp [roundShift32]
  have hdiv : (0 : ℤ) < 2 ^ 32 := by norm_num
  have hmod : (2 ^ 32) * ((x + 2 ^ 31) / (2 ^ 32)) + (x + 2 ^ 31) % (2 ^ 32) = x + 2 ^ 31 :=
    Int.mul_ediv_add_emod (x + 2 ^ 31) (2 ^ 32)
  have hrem := Int.emod_nonneg (x + 2 ^ 31) (ne_of_gt hdiv)
  have hlt := Int.emod_lt_of_pos (x + 2 ^ 31) hdiv
  norm_num at *
  constructor <;> linarith

/-! ## 2. Reduction Tree Exactness -/

/-- 4-element binary tree reduction (warp shuffle tree structure). -/
def treeReduce4 (x0 x1 x2 x3 : ℤ) : ℤ :=
  (x0 + x1) + (x2 + x3)

/-- 4-element sequential reduction (serial accumulator loop). -/
def seqReduce4 (x0 x1 x2 x3 : ℤ) : ℤ :=
  ((x0 + x1) + x2) + x3

/-- Theorem: Binary tree reduction is bit-identical to sequential reduction over `ℤ`. -/
theorem reduction_tree4_exact (x0 x1 x2 x3 : ℤ) :
    treeReduce4 x0 x1 x2 x3 = seqReduce4 x0 x1 x2 x3 := by
  dsimp [treeReduce4, seqReduce4]
  ring

/-- Zero-drift guarantee: integer parallel tree reduction has identically zero error. -/
theorem reduction_tree4_zero_drift (x0 x1 x2 x3 : ℤ) :
    treeReduce4 x0 x1 x2 x3 - seqReduce4 x0 x1 x2 x3 = 0 := by
  dsimp [treeReduce4, seqReduce4]
  ring

/-- 8-element tree reduction (3-level warp reduction). -/
def treeReduce8 (x : Fin 8 → ℤ) : ℤ :=
  ((x 0 + x 1) + (x 2 + x 3)) + ((x 4 + x 5) + (x 6 + x 7))

/-- 8-element sequential reduction. -/
def seqReduce8 (x : Fin 8 → ℤ) : ℤ :=
  (((((((x 0 + x 1) + x 2) + x 3) + x 4) + x 5) + x 6) + x 7)

/-- Theorem: 8-element warp reduction tree is bit-identical to sequential evaluation. -/
theorem reduction_tree8_exact (x : Fin 8 → ℤ) :
    treeReduce8 x = seqReduce8 x := by
  dsimp [treeReduce8, seqReduce8]
  ring

end Spirv.SplitLimb
