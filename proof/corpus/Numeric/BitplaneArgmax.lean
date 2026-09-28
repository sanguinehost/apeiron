/-
Apeiron — two-stage bit-plane exact argmax pruning for Q8_0 LM head evaluation.

## 1. Measured Result (the lmhead-prune-probe note §8)
In greedy decode of Qwen3.5-0.8B (vocabulary size 248,320, hidden dimension 1024,
tied LM head weight in Q8_0 format at 1088 bytes per row):
* Candidate-seeded pruning (seeded by the dense logit of `arg max s_hi`) leaves a median
  of only 246.0 of 248,320 rows surviving (0.099% survival rate; p90 1,383 rows, 0.557%).
* Argmax exactness: 832 of 832 real decode positions verified bit-identical to dense evaluation
  (100.00% exact match across code, prose, and prompt domains).
* Memory bandwidth reduction: head bytes read fall from 270.17 MB (dense) to a median of
  143.16 MB (52.99% of dense baseline, saving ~47% of memory bus traffic).

## 2. Weight Format
The engine targets `Q8_0` quantized rows: each 32-weight block carries a 16-bit float scale
`d` (refined as fixed-point `Q32.32` integer scale in the engine, see `Numeric.DequantQ8_0`)
and 32 signed 8-bit codes `q ∈ [-128, 127]`.

## 3. Mathematical Attribution
* Two-stage bit-plane decomposition: each signed 8-bit code `q` decomposes into an arithmetic
  high nibble `q / 16 ∈ [-8, 7]` and an unsigned low nibble `q % 16 ∈ [0, 15]`.
* Hölder-style interval arithmetic over the discrete hypercube `[0, 15]^32`: because the
  unobserved low nibble varies over `[0, 15]`, the unobserved contribution to block `b`
  is strictly bounded by element-wise sign alignment, achieving the exact attainable infimum
  and supremum without continuous geometric relaxation (e.g. Cauchy-Schwarz or max |h|).

## 4. Trust Boundary and Exclusions
* The conversion from IEEE 754 half-precision (`f16`) scale bytes to fixed-point `Q32.32`
  is established in `Numeric.DequantQ8_0` and `Numeric.DequantRefinement` (`f16_to_fixed_exact`);
  this module operates over exact integer arithmetic (`ℤ`).
* The Rust SIMD/AVX-512 kernel implementation and execution runtime are deferred to later tasks.
* Argmax tying: `PrunedArgmax.lowestMaxOn` formalises the lowest-index maximiser convention
  matching the engine sampler loop (`Numeric.EngineSamplerRefinement`).
-/
import Mathlib
import Numeric.PrunedArgmax

set_option autoImplicit false
set_option warningAsError true

open Classical
open scoped BigOperators

namespace BitplaneArgmax

/-! ## 1. Nibble Decomposition -/

/-- **Headline 1a.** Decomposition of an integer code into an arithmetic quotient
and an unsigned remainder in `[0, 15]` under Euclidean division `Int.ediv` / `Int.emod`. -/
theorem nibble_split (q : ℤ) : q = 16 * (q / 16) + q % 16 ∧ 0 ≤ q % 16 ∧ q % 16 ≤ 15 := by
  omega

/-- **Headline 1b.** The arithmetic high nibble of an 8-bit signed code in `[-128, 127]`
is bounded in `[-8, 7]`. -/
theorem nibble_hi_range (q : ℤ) : -128 ≤ q → q ≤ 127 → -8 ≤ q / 16 ∧ q / 16 ≤ 7 := by
  intro _ _
  omega

/-! ## 2. Row Model and Exact Interval Bounds -/

/-- Lower bound on the unobserved low-nibble dot product of a 32-element block with scale `d`:
the minimum over `l ∈ [0, 15]^32` of `d * ∑_i l_i * x_i`. -/
def blockLoLower (d : ℤ) (x : Fin 32 → ℤ) : ℤ :=
  min (d * 15 * ∑ i : Fin 32, min 0 (x i)) (d * 15 * ∑ i : Fin 32, max 0 (x i))

/-- Upper bound on the unobserved low-nibble dot product of a 32-element block with scale `d`:
the maximum over `l ∈ [0, 15]^32` of `d * ∑_i l_i * x_i`. -/
def blockLoUpper (d : ℤ) (x : Fin 32 → ℤ) : ℤ :=
  max (d * 15 * ∑ i : Fin 32, min 0 (x i)) (d * 15 * ∑ i : Fin 32, max 0 (x i))

lemma sum_min_nonpos (x : Fin 32 → ℤ) : (∑ i : Fin 32, min 0 (x i)) ≤ 0 := by
  have : (∑ i : Fin 32, min 0 (x i)) ≤ ∑ i : Fin 32, (0 : ℤ) :=
    Finset.sum_le_sum fun i _ => min_le_left 0 (x i)
  simpa using this

lemma sum_max_nonneg (x : Fin 32 → ℤ) : 0 ≤ ∑ i : Fin 32, max 0 (x i) := by
  have : (∑ i : Fin 32, (0 : ℤ)) ≤ ∑ i : Fin 32, max 0 (x i) :=
    Finset.sum_le_sum fun i _ => le_max_left 0 (x i)
  simpa using this

/-- When scale `d ≥ 0`, `blockLoLower` selects the negative-activation sum. -/
theorem blockLoLower_of_nonneg {d : ℤ} (hd : 0 ≤ d) (x : Fin 32 → ℤ) :
    blockLoLower d x = d * 15 * ∑ i : Fin 32, min 0 (x i) := by
  unfold blockLoLower
  have h1 := sum_min_nonpos x
  have h2 := sum_max_nonneg x
  have : d * 15 * ∑ i : Fin 32, min 0 (x i) ≤ d * 15 * ∑ i : Fin 32, max 0 (x i) := by
    nlinarith
  exact min_eq_left this

/-- When scale `d < 0`, `blockLoLower` selects the positive-activation sum. -/
theorem blockLoLower_of_neg {d : ℤ} (hd : d < 0) (x : Fin 32 → ℤ) :
    blockLoLower d x = d * 15 * ∑ i : Fin 32, max 0 (x i) := by
  unfold blockLoLower
  have h1 := sum_min_nonpos x
  have h2 := sum_max_nonneg x
  have : d * 15 * ∑ i : Fin 32, max 0 (x i) ≤ d * 15 * ∑ i : Fin 32, min 0 (x i) := by
    nlinarith
  exact min_eq_right this

/-- When scale `d ≥ 0`, `blockLoUpper` selects the positive-activation sum. -/
theorem blockLoUpper_of_nonneg {d : ℤ} (hd : 0 ≤ d) (x : Fin 32 → ℤ) :
    blockLoUpper d x = d * 15 * ∑ i : Fin 32, max 0 (x i) := by
  unfold blockLoUpper
  have h1 := sum_min_nonpos x
  have h2 := sum_max_nonneg x
  have : d * 15 * ∑ i : Fin 32, min 0 (x i) ≤ d * 15 * ∑ i : Fin 32, max 0 (x i) := by
    nlinarith
  exact max_eq_right this

/-- When scale `d < 0`, `blockLoUpper` selects the negative-activation sum. -/
theorem blockLoUpper_of_neg {d : ℤ} (hd : d < 0) (x : Fin 32 → ℤ) :
    blockLoUpper d x = d * 15 * ∑ i : Fin 32, min 0 (x i) := by
  unfold blockLoUpper
  have h1 := sum_min_nonpos x
  have h2 := sum_max_nonneg x
  have : d * 15 * ∑ i : Fin 32, max 0 (x i) ≤ d * 15 * ∑ i : Fin 32, min 0 (x i) := by
    nlinarith
  exact max_eq_left this

lemma term_lower_bound (l_val x_val : ℤ) (hl0 : 0 ≤ l_val) (hl15 : l_val ≤ 15) :
    15 * min 0 x_val ≤ l_val * x_val := by
  by_cases hx : 0 ≤ x_val
  · have : min 0 x_val = 0 := min_eq_left hx
    rw [this, mul_zero]
    nlinarith
  · have : min 0 x_val = x_val := min_eq_right (by omega)
    rw [this]
    nlinarith

lemma term_upper_bound (l_val x_val : ℤ) (hl0 : 0 ≤ l_val) (hl15 : l_val ≤ 15) :
    l_val * x_val ≤ 15 * max 0 x_val := by
  by_cases hx : 0 ≤ x_val
  · have : max 0 x_val = x_val := max_eq_right hx
    rw [this]
    nlinarith
  · have : max 0 x_val = 0 := max_eq_left (by omega)
    rw [this, mul_zero]
    nlinarith

lemma sum_term_lower (l x : Fin 32 → ℤ) (hl : ∀ i, 0 ≤ l i ∧ l i ≤ 15) :
    15 * (∑ i : Fin 32, min 0 (x i)) ≤ ∑ i : Fin 32, l i * x i := by
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => term_lower_bound (l i) (x i) (hl i).1 (hl i).2

lemma sum_term_upper (l x : Fin 32 → ℤ) (hl : ∀ i, 0 ≤ l i ∧ l i ≤ 15) :
    (∑ i : Fin 32, l i * x i) ≤ 15 * (∑ i : Fin 32, max 0 (x i)) := by
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => term_upper_bound (l i) (x i) (hl i).1 (hl i).2

/-- Exact dot product of a row composed of `n` blocks of width 32. -/
def dot (n : ℕ) (d : Fin n → ℤ) (c x : Fin n → Fin 32 → ℤ) : ℤ :=
  ∑ b : Fin n, d b * ∑ i : Fin 32, c b i * x b i

/-- High-nibble partial dot product evaluated in Stage 1. -/
def hiPart (n : ℕ) (d : Fin n → ℤ) (c x : Fin n → Fin 32 → ℤ) : ℤ :=
  ∑ b : Fin n, d b * ∑ i : Fin 32, 16 * (c b i / 16) * x b i

/-- Total lower envelope offset for the unobserved low nibbles. -/
def loLower (n : ℕ) (d : Fin n → ℤ) (x : Fin n → Fin 32 → ℤ) : ℤ :=
  ∑ b : Fin n, blockLoLower (d b) (x b)

/-- Total upper envelope offset for the unobserved low nibbles. -/
def loUpper (n : ℕ) (d : Fin n → ℤ) (x : Fin n → Fin 32 → ℤ) : ℤ :=
  ∑ b : Fin n, blockLoUpper (d b) (x b)

lemma h_split_elem {n : ℕ} (c x : Fin n → Fin 32 → ℤ) (b : Fin n) (i : Fin 32) :
    c b i * x b i = 16 * (c b i / 16) * x b i + (c b i % 16) * x b i := by
  have hs := (nibble_split (c b i)).1
  calc c b i * x b i = (16 * (c b i / 16) + c b i % 16) * x b i := congrArg (· * x b i) hs
    _ = 16 * (c b i / 16) * x b i + (c b i % 16) * x b i := by ring

/-- **Headline 2.** Soundness of the bit-plane interval: the exact dequantized dot product
lies within `[hiPart + loLower, hiPart + loUpper]`. -/
theorem lo_interval_sound (n : ℕ) (d : Fin n → ℤ) (c x : Fin n → Fin 32 → ℤ) :
    hiPart n d c x + loLower n d x ≤ dot n d c x ∧
    dot n d c x ≤ hiPart n d c x + loUpper n d x := by
  have h_block_eq : ∀ b : Fin n,
      d b * ∑ i : Fin 32, c b i * x b i =
        d b * (∑ i : Fin 32, 16 * (c b i / 16) * x b i) +
        d b * (∑ i : Fin 32, (c b i % 16) * x b i) := by
    intro b
    have hsum : (∑ i : Fin 32, c b i * x b i) =
        (∑ i : Fin 32, 16 * (c b i / 16) * x b i) +
        (∑ i : Fin 32, (c b i % 16) * x b i) := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i _ => h_split_elem c x b i
    rw [hsum, mul_add]
  have h_dot_split : dot n d c x =
      hiPart n d c x + ∑ b : Fin n, d b * ∑ i : Fin 32, (c b i % 16) * x b i := by
    unfold dot hiPart
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun b _ => h_block_eq b
  have h_lower_block : ∀ b : Fin n,
      blockLoLower (d b) (x b) ≤ d b * ∑ i : Fin 32, (c b i % 16) * x b i := by
    intro b
    have hrem : ∀ i, 0 ≤ c b i % 16 ∧ c b i % 16 ≤ 15 := fun i => (nibble_split (c b i)).2
    have hmin := sum_term_lower (fun i => c b i % 16) (x b) hrem
    unfold blockLoLower
    by_cases hd : 0 ≤ d b
    · have h1 : d b * 15 * ∑ i : Fin 32, min 0 (x b i) ≤ d b * ∑ i : Fin 32, (c b i % 16) * x b i := by
        calc d b * 15 * ∑ i : Fin 32, min 0 (x b i)
            = d b * (15 * ∑ i : Fin 32, min 0 (x b i)) := by ring
          _ ≤ d b * ∑ i : Fin 32, (c b i % 16) * x b i := mul_le_mul_of_nonneg_left hmin hd
      exact le_trans (min_le_left _ _) h1
    · have hd' : d b ≤ 0 := by omega
      have hmax := sum_term_upper (fun i => c b i % 16) (x b) hrem
      have h1 : d b * 15 * ∑ i : Fin 32, max 0 (x b i) ≤ d b * ∑ i : Fin 32, (c b i % 16) * x b i := by
        calc d b * 15 * ∑ i : Fin 32, max 0 (x b i)
            = d b * (15 * ∑ i : Fin 32, max 0 (x b i)) := by ring
          _ ≤ d b * ∑ i : Fin 32, (c b i % 16) * x b i := mul_le_mul_of_nonpos_left hmax hd'
      exact le_trans (min_le_right _ _) h1
  have h_upper_block : ∀ b : Fin n,
      d b * ∑ i : Fin 32, (c b i % 16) * x b i ≤ blockLoUpper (d b) (x b) := by
    intro b
    have hrem : ∀ i, 0 ≤ c b i % 16 ∧ c b i % 16 ≤ 15 := fun i => (nibble_split (c b i)).2
    have hmax := sum_term_upper (fun i => c b i % 16) (x b) hrem
    unfold blockLoUpper
    by_cases hd : 0 ≤ d b
    · have h2 : d b * ∑ i : Fin 32, (c b i % 16) * x b i ≤ d b * 15 * ∑ i : Fin 32, max 0 (x b i) := by
        calc d b * ∑ i : Fin 32, (c b i % 16) * x b i
            ≤ d b * (15 * ∑ i : Fin 32, max 0 (x b i)) := mul_le_mul_of_nonneg_left hmax hd
          _ = d b * 15 * ∑ i : Fin 32, max 0 (x b i) := by ring
      exact le_trans h2 (le_max_right _ _)
    · have hd' : d b ≤ 0 := by omega
      have hmin := sum_term_lower (fun i => c b i % 16) (x b) hrem
      have h2 : d b * ∑ i : Fin 32, (c b i % 16) * x b i ≤ d b * 15 * ∑ i : Fin 32, min 0 (x b i) := by
        calc d b * ∑ i : Fin 32, (c b i % 16) * x b i
            ≤ d b * (15 * ∑ i : Fin 32, min 0 (x b i)) := mul_le_mul_of_nonpos_left hmin hd'
          _ = d b * 15 * ∑ i : Fin 32, min 0 (x b i) := by ring
      exact le_trans h2 (le_max_left _ _)
  rw [h_dot_split]
  unfold loLower loUpper
  constructor
  · have hsum_le := Finset.sum_le_sum (fun b (_ : b ∈ Finset.univ) => h_lower_block b)
    omega
  · have hsum_le := Finset.sum_le_sum (fun b (_ : b ∈ Finset.univ) => h_upper_block b)
    omega

/-! ## 3. Pruning Soundness and Exact Argmax

Note on `candSet` vs `survivorSet`:
`Numeric.PrunedArgmax.candSet` assumes a symmetric error bound `|L i - A i| ≤ E i`
and a global lower envelope `max_j (A j - E j)`.
In two-stage bit-plane evaluation, the upper bound `hi r` is asymmetric and compared
against a candidate seed logit `m0 = dot k`. We define the asymmetric survivor set
`survivorSet hi m0` directly and prove that the lowest-index maximiser over survivors
equals `PrunedArgmax.lowestMaxOn dotVal Finset.univ`.
-/

/-- The survivor set: rows whose upper bound reaches or exceeds the threshold `m0`. -/
def survivorSet {R : ℕ} (hi : Fin R → ℤ) (m0 : ℤ) : Finset (Fin R) :=
  Finset.univ.filter (fun r => m0 ≤ hi r)

/-- **Headline 3.** For rows `r` with sound upper bounds `dotVal r ≤ hi r`, any row with
`hi r < dotVal k` has strictly lower dot product than seed `k` and cannot be an argmax;
hence every maximiser survives pruning against `m0 = dotVal k`. -/
theorem bitplane_prune_sound {R : ℕ} (dotVal lo hi : Fin R → ℤ)
    (h_sound : ∀ r, lo r ≤ dotVal r ∧ dotVal r ≤ hi r) (k : Fin R) :
    (∀ r, hi r < dotVal k → dotVal r < dotVal k) ∧
    (∀ r, (∀ j, dotVal j ≤ dotVal r) → r ∈ survivorSet hi (dotVal k)) := by
  constructor
  · intro r hr
    have := (h_sound r).2
    omega
  · intro r hr
    rw [survivorSet, Finset.mem_filter]
    refine ⟨Finset.mem_univ r, ?_⟩
    have hrk : dotVal k ≤ dotVal r := hr k
    have := (h_sound r).2
    omega

/-- **Headline 5.** The candidate seed row `k` is always a survivor because its upper bound
contains its exact dot product (`dotVal k ≤ hi k`). -/
theorem survivors_nonempty {R : ℕ} (dotVal lo hi : Fin R → ℤ)
    (h_sound : ∀ r, lo r ≤ dotVal r ∧ dotVal r ≤ hi r) (k : Fin R) :
    (survivorSet hi (dotVal k)).Nonempty := by
  refine ⟨k, ?_⟩
  rw [survivorSet, Finset.mem_filter]
  refine ⟨Finset.mem_univ k, (h_sound k).2⟩

/-- **Headline 4.** The lowest-index argmax over the survivors evaluated exactly equals
the lowest-index argmax over all rows. -/
theorem bitplane_argmax_exact {R : ℕ} (hR : 0 < R) (dotVal lo hi : Fin R → ℤ)
    (h_sound : ∀ r, lo r ≤ dotVal r ∧ dotVal r ≤ hi r) (k : Fin R) :
    (∀ r, (∀ j, dotVal j ≤ dotVal r) → r ∈ survivorSet hi (dotVal k)) ∧
    (∀ (hS : (survivorSet hi (dotVal k)).Nonempty),
      PrunedArgmax.lowestMaxOn dotVal (survivorSet hi (dotVal k)) hS =
      PrunedArgmax.lowestMaxOn dotVal Finset.univ (PrunedArgmax.univ_nonempty_of_pos hR)) := by
  have hmem : ∀ r, (∀ j, dotVal j ≤ dotVal r) → r ∈ survivorSet hi (dotVal k) :=
    (bitplane_prune_sound dotVal lo hi h_sound k).2
  refine ⟨hmem, ?_⟩
  intro hS
  have hglobal : ∀ j, dotVal j ≤ dotVal (PrunedArgmax.lowestMaxOn dotVal Finset.univ (PrunedArgmax.univ_nonempty_of_pos hR)) :=
    fun j => (PrunedArgmax.lowestMaxOn_isMax dotVal Finset.univ (PrunedArgmax.univ_nonempty_of_pos hR)) j (Finset.mem_univ _)
  obtain ⟨g, _, hgmax⟩ :=
    Finset.exists_max_image (Finset.univ : Finset (Fin R)) dotVal (PrunedArgmax.univ_nonempty_of_pos hR)
  have hgmax' : ∀ j, dotVal j ≤ dotVal g := fun j => hgmax j (Finset.mem_univ _)
  have hgS : g ∈ survivorSet hi (dotVal k) := hmem g hgmax'
  apply le_antisymm
  · refine PrunedArgmax.lowestMaxOn_le dotVal (survivorSet hi (dotVal k)) hS _ (hmem _ hglobal) fun j _ => hglobal j
  · refine PrunedArgmax.lowestMaxOn_le dotVal Finset.univ (PrunedArgmax.univ_nonempty_of_pos hR) _ (Finset.mem_univ _) fun j _ => ?_
    exact le_trans (hgmax' j) ((PrunedArgmax.lowestMaxOn_isMax dotVal (survivorSet hi (dotVal k)) hS) g hgS)

/-- Direct equality of the argmax over survivors and the argmax over all rows, using
the automatic nonemptiness proof from `survivors_nonempty`. -/
theorem bitplane_argmax_exact_eq {R : ℕ} (hR : 0 < R) (dotVal lo hi : Fin R → ℤ)
    (h_sound : ∀ r, lo r ≤ dotVal r ∧ dotVal r ≤ hi r) (k : Fin R)
    (hS : (survivorSet hi (dotVal k)).Nonempty := survivors_nonempty dotVal lo hi h_sound k) :
    PrunedArgmax.lowestMaxOn dotVal (survivorSet hi (dotVal k)) hS =
    PrunedArgmax.lowestMaxOn dotVal Finset.univ (PrunedArgmax.univ_nonempty_of_pos hR) :=
  (bitplane_argmax_exact hR dotVal lo hi h_sound k).2 hS

/-! ## 4. Concrete Witness Instance and Seeded Defect -/

/-- Concrete activation vector: activation 1 at index 0, 0 elsewhere. -/
def witness_x : Fin 1 → Fin 32 → ℤ :=
  fun _ i => if i.val = 0 then 1 else 0

/-- Concrete row scales: unit scale `1` for all 3 rows. -/
def witness_d : Fin 3 → Fin 1 → ℤ :=
  fun _ _ => 1

/-- Concrete quantized codes for 3 rows:
* Row 0 (candidate seed): code 32 at element 0 (high nibble 2, low nibble 0).
* Row 1 (true maximiser): code 47 at element 0 (high nibble 2, low nibble 15).
* Row 2 (pruned row):     code 0 at element 0  (high nibble 0, low nibble 0). -/
def witness_c : Fin 3 → Fin 1 → Fin 32 → ℤ :=
  fun r _ i =>
    if i.val = 0 then
      if r.val = 0 then 32
      else if r.val = 1 then 47
      else 0
    else 0

/-- Exact dot product of witness row `r`. -/
def witness_dot (r : Fin 3) : ℤ :=
  dot 1 (witness_d r) (witness_c r) witness_x

/-- Sound upper bound of witness row `r`. -/
def witness_hi (r : Fin 3) : ℤ :=
  hiPart 1 (witness_d r) (witness_c r) witness_x + loUpper 1 (witness_d r) witness_x

/-- The seed row index is Row 0 (`m0 = witness_dot 0 = 32`). -/
def witness_k : Fin 3 := 0

/-- Witness verification: Row 2 is pruned (`witness_hi 2 = 15 < 32 = m0`). -/
theorem witness_row2_pruned : (2 : Fin 3) ∉ survivorSet witness_hi (witness_dot witness_k) := by
  decide

/-- Witness verification: Row 1 survives (`witness_hi 1 = 47 ≥ 32 = m0`). -/
theorem witness_row1_survives : (1 : Fin 3) ∈ survivorSet witness_hi (witness_dot witness_k) := by
  decide

/-- Witness verification: Row 0 survives (`witness_hi 0 = 47 ≥ 32 = m0`). -/
theorem witness_row0_survives : (0 : Fin 3) ∈ survivorSet witness_hi (witness_dot witness_k) := by
  decide

/-- **Headline 6.** Seeded defect: replacing `loUpper` with 0 breaks the upper bound
soundness lemma on Row 1 of the witness instance (`dot = 47`, `hiPart = 32`, so `47 ≤ 32` is false). -/
theorem lo_upper_needed :
    ¬ (dot 1 (witness_d 1) (witness_c 1) witness_x ≤
       hiPart 1 (witness_d 1) (witness_c 1) witness_x + 0) := by
  decide

end BitplaneArgmax
