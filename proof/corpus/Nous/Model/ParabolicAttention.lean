import Apeiron.Trichotomy.AnyRing
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.List.Basic
import Mathlib.Tactic

/-!
# Native Parabolic Linear Attention in R[x]/(x^2)

The ring-generic formulation of native parabolic linear attention over the dual numbers
`R[x]/(x^2)` (William Kingdon Clifford, *Preliminary Sketch of Biquaternions*, Proc. London
Math. Soc. 4 (1873), pp. 381–395). In `Nous.Model.Attention`, parabolic attention was treated
as degenerate because the hyperbolic channel split fails when `r = 0`, collapsing channel
weights to the uniform constant `1/2`. Here, the genuine parabolic attention is formulated
as linear attention (Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret,
*Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML 2020)
arising directly from the nilpotent structure `ε^2 = 0` at `σ = 0`.

## Mathematical Setting

1. **Dual Number Algebra (`σ = 0`):**
   Elements of `R[x]/(x^2)` are represented as points `z = (a, b)` in `RingCell R = R × R`,
   corresponding to `a + b·ε` with `ε^2 = 0`. Parabolic multiplication is:
   `ringMul 0 (a, b) (c, d) = (a * c, a * d + b * c)`.
   The parabolic norm form is `ringNormForm 0 (a, b) = a^2`, and the multiplicative identity
   is `ringOne = (1, 0)`.

2. **Parabolic Linear Attention Kernel:**
   For a query `q : R` and key `k : R`, the linear attention score is:
   `K(q, k) = 1 + q * k`.
   Under the dual representations `queryCell q = (1, q) = 1 + q·ε` and
   `keyCell k = (k, 1) = k + 1·ε`, the parabolic ring product is:
   `ringMul 0 (1, q) (k, 1) = (k, 1 + q * k)`.
   The nilpotent `ε`-component (projection `.2`) yields the exact linear attention kernel
   `1 + q * k`.

3. **Two-Pass Streaming State Accumulation:**
   For a sequence of key-value tokens `(k_i, v_i)`:
   - Numerator accumulator `S_N = Σ_i k_i * v_i` and `V_N = Σ_i v_i`.
   - Denominator (normalizer) accumulator `z_N = Σ_i k_i` and `N = Σ_i 1 = l.length`.
   In dual numbers, each token contributes a numerator cell `(k_i * v_i, v_i)` and a
   denominator cell `(k_i, 1)`. The query interaction is computed in one `ringMul 0`:
   `ringMul 0 (1, q) (S_N, V_N) = (S_N, V_N + q * S_N)`.
   The second slot gives `S_N * q + V_N = Σ_i (1 + q * k_i) * v_i`, matching the pairwise
   linear attention sum exactly.

4. **Exact Factorization and Complexity:**
   The pairwise sum `Σ_i (1 + q * k_i) * v_i` has temporal complexity O(N) per query.
   The streaming accumulation computes `(S_N, z_N, V_N, N)` in a single O(N) fold over the
   token sequence, after which any query readout is evaluated in O(1) ring operations:
   `stateOut st q = (st.S * q + st.v) / (st.z * q + st.n)`.

## Trust Boundary

No floating-point model: `R` is any commutative ring for linear accumulation and any field
for normalized division. No external dependencies outside Mathlib and `Apeiron.Trichotomy`.
Does not reach `rustc`/LLVM codegen or hardware-specific instruction scheduling.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

section CommRingSetting

variable {R : Type*} [CommRing R]

/-- The 1D parabolic linear attention score between query `q` and key `k`: `K(q, k) = 1 + q * k`.
Temporal complexity: O(1) operations. -/
def parabolicScore (q k : R) : R :=
  1 + q * k

/-- Dual representation of a query as `(1, q) = 1 + q·ε` in `R[x]/(x^2)`. -/
def queryCell (q : R) : RingCell R :=
  (1, q)

/-- Dual representation of a key as `(k, 1) = k + 1·ε` in `R[x]/(x^2)`. -/
def keyCell (k : R) : RingCell R :=
  (k, 1)

/-- Dual numerator token cell for key `k` and value `v`: `(k * v, v) = (k * v) + v·ε`. -/
def numTokenCell (k v : R) : RingCell R :=
  (k * v, v)

/-- Dual denominator token cell for key `k`: `(k, 1) = k + 1·ε`. -/
def denTokenCell (k : R) : RingCell R :=
  (k, 1)

/-- The parabolic linear attention score is the second slot of the parabolic ring product of
`queryCell q` and `keyCell k` at `σ = 0`. -/
theorem parabolicScore_eq_ringMul (q k : R) :
    (ringMul (0 : R) (queryCell q) (keyCell k)).2 = parabolicScore q k := by
  simp [parabolicScore, queryCell, keyCell]

/-- Commutativity of the parabolic linear attention score under query-key dual multiplication. -/
theorem parabolicScore_eq_ringMul_comm (q k : R) :
    (ringMul (0 : R) (keyCell k) (queryCell q)).2 = parabolicScore q k := by
  simp [parabolicScore, queryCell, keyCell]
  ring

/-- The query cell has unit parabolic norm form: `N(1 + q·ε) = 1^2 = 1`. -/
theorem queryCell_ringNormForm (q : R) :
    ringNormForm (0 : R) (queryCell q) = 1 := by
  simp [queryCell, ringNormForm]

/-- Pairwise unnormalized parabolic linear attention sum: `Σ_i (1 + q * k_i) * v_i`.
Temporal complexity: O(N) operations in sequence length `N = l.length`. -/
def pairwiseNum (q : R) (l : List (R × R)) : R :=
  (l.map (fun p => parabolicScore q p.1 * p.2)).sum

/-- Pairwise parabolic linear attention normalizer: `Σ_i (1 + q * k_i)`.
Temporal complexity: O(N) operations in sequence length `N = l.length`. -/
def pairwiseDen (q : R) (l : List (R × R)) : R :=
  (l.map (fun p => parabolicScore q p.1)).sum

/-! ### Two-Variable Linear Accumulator `(S, z)` -/

/-- Streaming step for the two-variable linear accumulator `(S, z)` (Katharopoulos et al. 2020):
`S` accumulates `k * v` and `z` accumulates `k`.
Temporal complexity: O(1) operations. -/
def linStep (st : R × R) (p : R × R) : R × R :=
  (st.1 + p.1 * p.2, st.2 + p.1)

/-- Streaming fold over tokens `l` with `linStep` from `(0, 0)`.
Temporal complexity: O(N) sequential scan over `N = l.length`. -/
def linScan (l : List (R × R)) : R × R :=
  l.foldl linStep (0, 0)

/-- General accumulator lemma for `linStep`. -/
theorem linScan_acc (l : List (R × R)) (acc : R × R) :
    l.foldl linStep acc =
      (acc.1 + (l.map (fun p => p.1 * p.2)).sum, acc.2 + (l.map (fun p => p.1)).sum) := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    rw [List.foldl_cons, ih, List.map_cons, List.map_cons, List.sum_cons, List.sum_cons]
    refine Prod.ext ?_ ?_
    · show (acc.1 + p.1 * p.2) + (ps.map (fun q => q.1 * q.2)).sum =
        acc.1 + (p.1 * p.2 + (ps.map (fun q => q.1 * q.2)).sum)
      ring
    · show (acc.2 + p.1) + (ps.map (fun q => q.1)).sum =
        acc.2 + (p.1 + (ps.map (fun q => q.1)).sum)
      ring

/-- The scan from zero evaluates to the pair of component sums: `(Σ_i k_i * v_i, Σ_i k_i)`. -/
theorem linScan_eq (l : List (R × R)) :
    linScan l = ((l.map (fun p => p.1 * p.2)).sum, (l.map (fun p => p.1)).sum) := by
  show l.foldl linStep (0, 0) = _
  rw [linScan_acc]
  refine Prod.ext ?_ ?_
  · simp
  · simp

/-- Key theorem: streaming accumulation `S_N * q + Σ_i v_i` equals the pairwise sum
`Σ_i (1 + q * k_i) * v_i`. -/
theorem lin_streaming_num_eq (l : List (R × R)) (q : R) :
    (linScan l).1 * q + (l.map (fun p => p.2)).sum = pairwiseNum q l := by
  rw [linScan_eq]
  show ((l.map (fun p => p.1 * p.2)).sum) * q + (l.map (fun p => p.2)).sum = _
  unfold pairwiseNum parabolicScore
  induction l with
  | nil => simp
  | cons p ps ih =>
    simp only [List.map_cons, List.sum_cons]
    calc
      (p.1 * p.2 + (ps.map (fun p => p.1 * p.2)).sum) * q + (p.2 + (ps.map (fun p => p.2)).sum)
        = (1 + q * p.1) * p.2 + (((ps.map (fun p => p.1 * p.2)).sum) * q + (ps.map (fun p => p.2)).sum) := by ring
      _ = (1 + q * p.1) * p.2 + (ps.map (fun p => (1 + q * p.1) * p.2)).sum := by rw [ih]

/-- Streaming normalizer accumulation `z_N * q + l.length` equals the pairwise sum
`Σ_i (1 + q * k_i)`. -/
theorem lin_streaming_den_eq (l : List (R × R)) (q : R) :
    (linScan l).2 * q + (l.length : R) = pairwiseDen q l := by
  rw [linScan_eq]
  show ((l.map (fun p => p.1)).sum) * q + (l.length : R) = _
  unfold pairwiseDen parabolicScore
  induction l with
  | nil => simp
  | cons p ps ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    calc
      (p.1 + (ps.map (fun p => p.1)).sum) * q + ((ps.length : R) + 1)
        = (1 + q * p.1) + (((ps.map (fun p => p.1)).sum) * q + (ps.length : R)) := by ring
      _ = (1 + q * p.1) + (ps.map (fun p => 1 + q * p.1)).sum := by rw [ih]

/-! ### Four-Component Streaming State `State R` -/

/-- Complete streaming state accumulator for parabolic linear attention:
- `S`: accumulated `k_i * v_i` (linear memory state for query numerator)
- `z`: accumulated `k_i` (linear memory state for query normalizer)
- `v`: accumulated `v_i` (constant term for query numerator)
- `n`: accumulated token count `1` (constant term for normalizer) -/
@[ext]
structure State (R : Type*) where
  S : R
  z : R
  v : R
  n : R
deriving Repr, DecidableEq

/-- One streaming step adding a key-value token `(k, v)` to the running state.
Temporal complexity: O(1) operations. -/
def stateStep (st : State R) (p : R × R) : State R :=
  ⟨st.S + p.1 * p.2, st.z + p.1, st.v + p.2, st.n + 1⟩

/-- The initial zero state. -/
def stateZero : State R :=
  ⟨0, 0, 0, 0⟩

/-- Streaming fold over tokens `l` with `stateStep` from `stateZero`.
Temporal complexity: O(N) sequential scan over `N = l.length`. -/
def stateScan (l : List (R × R)) : State R :=
  l.foldl stateStep stateZero

/-- General accumulator lemma for `stateStep`. -/
theorem stateScan_acc (l : List (R × R)) (st : State R) :
    l.foldl stateStep st =
      ⟨st.S + (l.map (fun p => p.1 * p.2)).sum,
       st.z + (l.map (fun p => p.1)).sum,
       st.v + (l.map (fun p => p.2)).sum,
       st.n + (l.length : R)⟩ := by
  induction l generalizing st with
  | nil => simp
  | cons p ps ih =>
    rw [List.foldl_cons, ih]
    simp only [stateStep, List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    ext <;> ring

/-- The state fold from zero evaluates to the exact quadruplet of component sums. -/
theorem stateScan_eq (l : List (R × R)) :
    stateScan l =
      ⟨(l.map (fun p => p.1 * p.2)).sum,
       (l.map (fun p => p.1)).sum,
       (l.map (fun p => p.2)).sum,
       (l.length : R)⟩ := by
  show l.foldl stateStep stateZero = _
  rw [stateScan_acc]
  simp [stateZero]

/-- Numerator readout from the state against query `q`: `st.S * q + st.v`.
Temporal complexity: O(1) operations. -/
def stateNum (st : State R) (q : R) : R :=
  st.S * q + st.v

/-- Denominator (normalizer) readout from the state against query `q`: `st.z * q + st.n`.
Temporal complexity: O(1) operations. -/
def stateDen (st : State R) (q : R) : R :=
  st.z * q + st.n

/-- Numerator readout equals the pairwise linear attention sum. -/
theorem stateNum_eq_pairwise (l : List (R × R)) (q : R) :
    stateNum (stateScan l) q = pairwiseNum q l := by
  rw [stateScan_eq]
  show ((l.map (fun p => p.1 * p.2)).sum) * q + (l.map (fun p => p.2)).sum = _
  have h := lin_streaming_num_eq l q
  rw [linScan_eq] at h
  exact h

/-- Denominator readout equals the pairwise linear attention normalizer. -/
theorem stateDen_eq_pairwise (l : List (R × R)) (q : R) :
    stateDen (stateScan l) q = pairwiseDen q l := by
  rw [stateScan_eq]
  show ((l.map (fun p => p.1)).sum) * q + (l.length : R) = _
  have h := lin_streaming_den_eq l q
  rw [linScan_eq] at h
  exact h

/-! ### Dual Cell Ring Product Readout -/

/-- Parabolic ring multiplication of `queryCell q` and accumulated numerator cell `(S, V)`
produces `S * q + V` in its second slot. -/
theorem ringMul_query_num (q S V : R) :
    (ringMul (0 : R) (queryCell q) (S, V)).2 = S * q + V := by
  simp [queryCell]
  ring

/-- Parabolic ring multiplication of `queryCell q` and accumulated denominator cell `(z, N)`
produces `z * q + N` in its second slot. -/
theorem ringMul_query_den (q z N : R) :
    (ringMul (0 : R) (queryCell q) (z, N)).2 = z * q + N := by
  simp [queryCell]
  ring

end CommRingSetting

section FieldSetting

variable {R : Type*} [Field R]

/-- Full normalized parabolic linear attention readout from accumulated state.
Temporal complexity: O(1) operations. -/
def stateOut (st : State R) (q : R) : R :=
  stateNum st q / stateDen st q

/-- Pairwise normalized linear attention: `(Σ_i (1 + q * k_i) * v_i) / (Σ_i (1 + q * k_i))`.
Temporal complexity: O(N) operations. -/
def parabolicLinAttn (q : R) (l : List (R × R)) : R :=
  pairwiseNum q l / pairwiseDen q l

/-- Admissible domain for normalized parabolic linear attention: normalizer is nonzero. -/
def ParabolicLinAttnDomain (q : R) (l : List (R × R)) : Prop :=
  stateDen (stateScan l) q ≠ 0

/-- Dividing a list sum by a denominator distributes over each term. -/
theorem list_sum_div_map (l : List R) (d : R) :
    l.sum / d = (l.map (fun x => x / d)).sum := by
  induction l with
  | nil => simp
  | cons x xs ih => rw [List.sum_cons, List.map_cons, List.sum_cons, add_div, ih]

/-- **Exact Parabolic Linear Attention Factorization (Headline Theorem).**
On `ParabolicLinAttnDomain q l`, the streaming state scan and O(1) query evaluation:
1. Matches the pairwise normalized attention output `parabolicLinAttn q l` exactly;
2. Satisfies exact normalization: `stateOut * stateDen = stateNum`;
3. Numerator equals the pairwise sum `Σ_i (1 + q * k_i) * v_i`;
4. Denominator equals the pairwise normalizer `Σ_i (1 + q * k_i)`. -/
theorem parabolic_streaming_exact {q : R} {l : List (R × R)}
    (hdom : ParabolicLinAttnDomain q l) :
    stateOut (stateScan l) q = parabolicLinAttn q l ∧
    stateOut (stateScan l) q * stateDen (stateScan l) q = stateNum (stateScan l) q ∧
    stateNum (stateScan l) q = pairwiseNum q l ∧
    stateDen (stateScan l) q = pairwiseDen q l := by
  have hnum : stateNum (stateScan l) q = pairwiseNum q l := stateNum_eq_pairwise l q
  have hden : stateDen (stateScan l) q = pairwiseDen q l := stateDen_eq_pairwise l q
  have h_out : stateOut (stateScan l) q = parabolicLinAttn q l := by
    unfold stateOut parabolicLinAttn
    rw [hnum, hden]
  have h_norm : stateOut (stateScan l) q * stateDen (stateScan l) q = stateNum (stateScan l) q := by
    unfold stateOut
    exact div_mul_cancel₀ (stateNum (stateScan l) q) hdom
  exact ⟨h_out, h_norm, hnum, hden⟩

/-- The normalized readout expands into per-token normalized weights:
`stateOut (stateScan l) q = Σ_i ((1 + q * k_i) / normalizer) * v_i`. -/
theorem parabolic_lin_attn_expanded (q : R) (l : List (R × R)) :
    stateOut (stateScan l) q =
      (l.map (fun p => (parabolicScore q p.1 / stateDen (stateScan l) q) * p.2)).sum := by
  rw [stateOut, stateNum_eq_pairwise, pairwiseNum]
  show (l.map (fun p => parabolicScore q p.1 * p.2)).sum / stateDen (stateScan l) q = _
  rw [list_sum_div_map, List.map_map]
  refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
  show parabolicScore q p.1 * p.2 / stateDen (stateScan l) q =
    (parabolicScore q p.1 / stateDen (stateScan l) q) * p.2
  exact (div_mul_eq_mul_div (parabolicScore q p.1) _ p.2).symm

/-- Admissibility domain witness: over `ℚ`, query `q = 1` against single token `(1, 1)` yields
normalizer `1 * 1 + 1 = 2 ≠ 0`. -/
theorem parabolic_lin_attn_domain_witness :
    ParabolicLinAttnDomain (1 : ℚ) [((1 : ℚ), (1 : ℚ))] := by
  unfold ParabolicLinAttnDomain stateDen stateScan stateZero stateStep
  norm_num

/-- Refusal side: on an empty token list `[]`, the normalizer vanishes (`0 * q + 0 = 0`),
so normalization is refused. -/
theorem stateDen_nil (q : R) :
    stateDen (stateScan ([] : List (R × R))) q = 0 := by
  simp [stateDen, stateScan, stateZero]

end FieldSetting

end Nous.Model
