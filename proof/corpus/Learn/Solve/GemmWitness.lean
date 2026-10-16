import ApeironWord
import Learn.Grad.DenseLinearSpec
import Learn.Solve.GemmSpec
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted `src/learn/solve/gemm.rs` products: non-vacuity witness

`Learn.Solve.GemmSpec` states the parameters `BND`, `MAXK`, `withinBound` and the refusal chain
`gemmRefuse` under which the extracted blocked Q32.32 products `gemm_nn`/`gemm_nt` are specified; this
module discharges the non-vacuity obligation (§4, obligation 2): the hypotheses of the accepted path
are **jointly** satisfiable at a concrete instance, and each refusal variant of `gemmRefuse` is
reachable, so neither the entry bound, the contraction bound, the length/`Usize` products, the shape
agreement nor the magnitude flag is an empty domain.

The joint instance is `m = 2`, `k = 3`, `n = 2` on the Q32.32 raw operands
`A = [[1,0,0],[0,1,0]]`, `Bₙₙ = [[1,0],[0,1],[0,0]]`, `Bₙₜ = [[1,0,0],[0,1,0]]`
(`wA`, `wBnn`, `wBnt`, entries `2³²` and `0`); both contractions reproduce `[[1,0],[0,1]]` exactly, so
the model is computed (`decide`, which reduces `rnd32` on concrete integers) rather than assumed. A
mutation of the reference arithmetic reddens the computation; a mutation of the refusal order reddens
the refusal witnesses.

The `Usize` bounds are proved from `Usize.bounds_eq` (the platform is `2³²`- or `2⁶⁴`-wide), so no
witness depends on a compile-time `System.Platform.numBits`.

Trust boundary: rustc/LLVM codegen, the kernel ABI, the `Vec`/`Slice` container layout and the
threaded driver `block_rows_par` are not covered; only the extracted integer body.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## 1. The concrete operands -/

/-- `A = [[1,0,0],[0,1,0]]` in Q32.32 raw units, row-major `2 × 3`. -/
def wA : List ℤ := [(2 : ℤ) ^ 32, 0, 0, 0, (2 : ℤ) ^ 32, 0]

/-- `Bₙₙ = [[1,0],[0,1],[0,0]]` in Q32.32 raw units, row-major `3 × 2`. -/
def wBnn : List ℤ := [(2 : ℤ) ^ 32, 0, 0, (2 : ℤ) ^ 32, 0, 0]

/-- `Bₙₜ = [[1,0,0],[0,1,0]]` in Q32.32 raw units, row-major `2 × 3`. -/
def wBnt : List ℤ := [(2 : ℤ) ^ 32, 0, 0, 0, (2 : ℤ) ^ 32, 0]

/-! ## 2. The `Usize` bounds (platform-independent) -/

/-- `Usize.max ≥ 6`: the width is `32` or `64`, so the maximum is `2³²−1` or `2⁶⁴−1`. -/
theorem witness_usize_max_ge : (6 : ℕ) ≤ Usize.max := by
  rcases Usize.bounds_eq with h | h
  · rw [h, U32.max_def, U32.numBits_def, UScalarTy.U32_numBits_eq]; try norm_num
  · rw [h, U64.max_def, U64.numBits_def, UScalarTy.U64_numBits_eq]; try norm_num

/-- `Usize.max ≤ 2⁶⁴−1`. -/
theorem witness_usize_max_le : Usize.max ≤ 2 ^ 64 - 1 := by
  rcases Usize.bounds_eq with h | h
  · rw [h, U32.max_def, U32.numBits_def, UScalarTy.U32_numBits_eq]; try norm_num
  · rw [h, U64.max_def, U64.numBits_def, UScalarTy.U64_numBits_eq]; try norm_num

/-- The three `2 × 3 × 2` dimension products fit in `Usize` (the negation form `gemmRefuse` tests). -/
theorem witness_dims_ok :
    ¬ ¬ (2 * 3 ≤ Usize.max ∧ 3 * 2 ≤ Usize.max ∧ 2 * 2 ≤ Usize.max) := by
  intro hc
  have h6 := witness_usize_max_ge
  exact hc ⟨by omega, by omega, by omega⟩

/-! ## 3. The joint hypothesis instance -/

theorem witness_withinBound_A : withinBound wA := by
  intro z hz
  simp only [wA, List.mem_cons, List.mem_nil_iff, or_false] at hz
  rcases hz with rfl | rfl | rfl | rfl | rfl | rfl <;> norm_num [BND]

theorem witness_withinBound_Bnn : withinBound wBnn := by
  intro z hz
  simp only [wBnn, List.mem_cons, List.mem_nil_iff, or_false] at hz
  rcases hz with rfl | rfl | rfl | rfl | rfl | rfl <;> norm_num [BND]

theorem witness_withinBound_Bnt : withinBound wBnt := by
  intro z hz
  simp only [wBnt, List.mem_cons, List.mem_nil_iff, or_false] at hz
  rcases hz with rfl | rfl | rfl | rfl | rfl | rfl <;> norm_num [BND]

/-- The accepted `2 × 3 × 2` shape passes every branch of the refusal chain. -/
theorem witness_accepted : gemmRefuse 6 6 2 3 2 true = none := by
  unfold gemmRefuse
  rw [if_neg (by decide), if_neg (by rw [MAXK_eq]; norm_num), if_neg witness_dims_ok,
    if_neg (by simp), if_neg (by simp)]

/-- `gemm_nn` at the joint instance reproduces the `2 × 2` identity in Q32.32 raw units. -/
theorem witness_outNn : outNn wA wBnn 2 3 2 = [(2 : ℤ) ^ 32, 0, 0, (2 : ℤ) ^ 32] := by
  decide

/-- `gemm_nt` at the joint instance reproduces the `2 × 2` identity in Q32.32 raw units. -/
theorem witness_outNt : outNt wA wBnt 2 3 2 = [(2 : ℤ) ^ 32, 0, 0, (2 : ℤ) ^ 32] := by
  decide

/-- The hypotheses of the `nn`/`nt` accepted path hold jointly at one concrete instance: both operands
are within `BND`, the contraction length is below `MAXK`, the refusal chain accepts, and both products
are the computed model. -/
theorem gemm_witness_joint :
    withinBound wA ∧ withinBound wBnn ∧ withinBound wBnt ∧ (3 : ℤ) < MAXK ∧
    gemmRefuse wA.length wBnn.length 2 3 2 true = none ∧
    outNn wA wBnn 2 3 2 = [(2 : ℤ) ^ 32, 0, 0, (2 : ℤ) ^ 32] ∧
    outNt wA wBnt 2 3 2 = [(2 : ℤ) ^ 32, 0, 0, (2 : ℤ) ^ 32] := by
  refine ⟨witness_withinBound_A, witness_withinBound_Bnn, witness_withinBound_Bnt, ?_, ?_,
    witness_outNn, witness_outNt⟩
  · rw [MAXK_eq]; norm_num
  · show gemmRefuse wA.length wBnn.length 2 3 2 true = none
    rw [show wA.length = 6 from rfl, show wBnn.length = 6 from rfl]
    exact witness_accepted

/-! ## 4. One reachable refusal per variant of the chain -/

/-- A zero dimension refuses with `ZeroDim` (the first branch). -/
theorem witness_refuse_zero_dim : gemmRefuse 0 0 0 1 1 true = some .ZeroDim := rfl

/-- A contraction length at the `2⁵⁵` bound refuses with `Overflow` (the second branch). -/
theorem witness_refuse_overflow_k : gemmRefuse 0 0 1 (2 ^ 55) 1 true = some .Overflow := rfl

/-- A dimension product above `Usize.max` refuses with `Overflow` (the third branch): here
`m·k = 2⁸⁰ > 2⁶⁴ − 1 ≥ Usize.max`. -/
theorem witness_refuse_overflow_dims :
    gemmRefuse 4 4 (2 ^ 40) (2 ^ 40) 1 true = some .Overflow := by
  unfold gemmRefuse
  rw [if_neg (by decide), if_neg (by rw [MAXK_eq]; norm_num),
    if_pos (by
      intro hc
      have h1 := witness_usize_max_le
      have hA : (2 : ℕ) ^ 40 * 2 ^ 40 ≤ Usize.max := hc.1
      norm_num at hA h1
      omega)]

/-- A slot-buffer length other than `m·k`/`k·n` refuses with `ShapeMismatch` (the fourth branch). -/
theorem witness_refuse_shape : gemmRefuse 6 7 2 3 2 true = some .ShapeMismatch := by
  unfold gemmRefuse
  rw [if_neg (by decide), if_neg (by rw [MAXK_eq]; norm_num), if_neg witness_dims_ok,
    if_pos (by norm_num)]

/-- A magnitude flag clear refuses with `InputBound` (the fifth branch). -/
theorem witness_refuse_input_bound : gemmRefuse 6 6 2 3 2 false = some .InputBound := by
  unfold gemmRefuse
  rw [if_neg (by decide), if_neg (by rw [MAXK_eq]; norm_num), if_neg witness_dims_ok,
    if_neg (by simp), if_pos (by simp)]

end apeiron_word
