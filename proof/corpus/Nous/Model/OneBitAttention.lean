import Nous.Model.HyperbolicAttention
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Tactic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# One-bit attention on the hyperbolic face (attention-cut T2)

Part III.

Over real scores no fixed-size carry reproduces `e^{qk}` (`exp_kernel_not_finite_rank`). On one-bit
queries and keys it does: each `x = q_i k_i ∈ {±1}`, and the hyperbolic flow read through that
character is `e^{s x} = cosh s + x sinh s`, the upper channel `ringUpper 1 (flow 1 s)` at `x = 1`
and the lower channel at `x = -1`. The `d`-fold product expands over subsets `S ⊆ [d]` into
`cosh^{d-|S|} s · sinh^{|S|} s · q_S k_S`, so normalized attention over one-bit keys is the read of a
carry indexed by `Finset (Fin d)` (`2^d` entries), independent of the key count. Each key may carry
its own scale `a_n`; the key's features absorb it.

## Cost model (combinatorics over ℕ, not a measurement)

Counted in multiply-adds (MACs) and stored words, real arithmetic at unit cost.

| quantity | carry | scan (KV cache) |
|---|---|---|
| stored words | `2^d (d_v + 1)` | `n (d + d_v)` |
| per-query read | `2^d (d_v + 1)` | `n (d + d_v)` |
| per-key ingest | `2^d (d_v + 1)` | `d + d_v` (append) |
| causal pass over `n` tokens | `n · 2 · 2^d (d_v + 1)` | `(d + d_v) · n (n + 1) / 2` |

`carry_query_le_scan`: the read pays once `2^d ≤ n`. `carry_causal_le_scan`: a causal training or
prefill pass pays once `4 · 2^d ≤ n + 1`. Below those lengths the scan is cheaper; the carry trades a
`2^d` constant for linear time.

Grade: `proved` (Lean kernel; closed axiom cone, audited in `AxiomAudit/Remaining.lean`). Not
covered: floating-point evaluation, any conversion of a real-valued model to signs, hardware cost.
-/

noncomputable section

open Finset Real
open Apeiron.Trichotomy.AnyRing

namespace Nous.Model.OneBit

/-- A one-bit vector: every coordinate is `1` or `-1`. -/
def IsSign {d : ℕ} (x : Fin d → ℝ) : Prop := ∀ i, x i = 1 ∨ x i = -1

theorem isSign_mul {d : ℕ} {q k : Fin d → ℝ} (hq : IsSign q) (hk : IsSign k) :
    IsSign (fun i => q i * k i) := by
  intro i
  rcases hq i with h1 | h1 <;> rcases hk i with h2 | h2 <;> simp [h1, h2]

/-! ## 1. The factor is the hyperbolic flow read through a sign character -/

/-- `e^{s x} = cosh s + x sinh s` on `x ∈ {±1}`. -/
theorem exp_mul_sign (s x : ℝ) (hx : x = 1 ∨ x = -1) :
    Real.exp (s * x) = Real.cosh s + x * Real.sinh s := by
  rcases hx with rfl | rfl
  · rw [mul_one, one_mul, Real.cosh_add_sinh]
  · rw [mul_neg_one, neg_one_mul, ← sub_eq_add_neg, Real.cosh_sub_sinh]

/-- **The one-bit factor is a flow channel.** At `x = 1` it is the upper channel of
`flow 1 s`, at `x = -1` the lower channel. -/
theorem onebit_factor_is_flow (s : ℝ) :
    Real.cosh s + 1 * Real.sinh s = ringUpper (1 : ℝ) (Apeiron.Trichotomy.flow 1 s)
      ∧ Real.cosh s + (-1) * Real.sinh s = ringLower (1 : ℝ) (Apeiron.Trichotomy.flow 1 s) := by
  refine ⟨?_, ?_⟩
  · rw [Nous.Model.ringUpper_flow_one, ← exp_mul_sign s 1 (Or.inl rfl), mul_one]
  · rw [Nous.Model.ringLower_flow_one, ← exp_mul_sign s (-1) (Or.inr rfl), mul_neg_one]

/-! ## 2. Product and subset expansion -/

/-- **Product form (headline).** On one-bit `q, k`, the softmax weight of the score `s·(q·k)`
is the product of `d` flow factors. -/
theorem onebit_exp_product {d : ℕ} (s : ℝ) (q k : Fin d → ℝ) (hq : IsSign q) (hk : IsSign k) :
    Real.exp (s * ∑ i, q i * k i) = ∏ i, (Real.cosh s + q i * k i * Real.sinh s) := by
  rw [Finset.mul_sum, Real.exp_sum]
  exact Finset.prod_congr rfl (fun i _ => exp_mul_sign s _ (isSign_mul hq hk i))

/-- The subset weight `λ_S = cosh^{d-|S|} s · sinh^{|S|} s`. -/
def lam (d : ℕ) (s : ℝ) (S : Finset (Fin d)) : ℝ :=
  Real.cosh s ^ (d - S.card) * Real.sinh s ^ S.card

/-- The character `x_S = Π_{i ∈ S} x_i`. -/
def chi {d : ℕ} (x : Fin d → ℝ) (S : Finset (Fin d)) : ℝ := ∏ i ∈ S, x i

/-- **Subset expansion.** `Π_i (cosh s + x_i sinh s) = Σ_S λ_S x_S`, for any real `x`. -/
theorem prod_flow_expansion {d : ℕ} (s : ℝ) (x : Fin d → ℝ) :
    ∏ i, (Real.cosh s + x i * Real.sinh s) = ∑ S : Finset (Fin d), lam d s S * chi x S := by
  have h := Finset.prod_add (fun i => x i * Real.sinh s) (fun _ => Real.cosh s)
    (Finset.univ : Finset (Fin d))
  rw [Finset.powerset_univ] at h
  calc ∏ i, (Real.cosh s + x i * Real.sinh s)
      = ∏ i, (x i * Real.sinh s + Real.cosh s) := by
        exact Finset.prod_congr rfl (fun i _ => add_comm _ _)
    _ = _ := h
    _ = ∑ S : Finset (Fin d), lam d s S * chi x S := by
        refine Finset.sum_congr rfl (fun S _ => ?_)
        rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.prod_const,
          ← Finset.compl_eq_univ_sdiff, Finset.card_compl, Fintype.card_fin, lam, chi]
        ring

/-- The character of a product is the product of characters. -/
theorem chi_mul {d : ℕ} (q k : Fin d → ℝ) (S : Finset (Fin d)) :
    chi (fun i => q i * k i) S = chi q S * chi k S := by
  simp only [chi, Finset.prod_mul_distrib]

/-- **Exponential expansion (headline).** On one-bit `q, k`:
`e^{s q·k} = Σ_{S ⊆ [d]} cosh^{d-|S|} s · sinh^{|S|} s · q_S k_S`. -/
theorem onebit_exp_expansion {d : ℕ} (s : ℝ) (q k : Fin d → ℝ) (hq : IsSign q) (hk : IsSign k) :
    Real.exp (s * ∑ i, q i * k i) = ∑ S : Finset (Fin d), lam d s S * chi q S * chi k S := by
  rw [onebit_exp_product s q k hq hk]
  rw [prod_flow_expansion s (fun i => q i * k i)]
  refine Finset.sum_congr rfl (fun S _ => ?_)
  rw [chi_mul]
  ring

/-! ## 3. The exact finite carry -/

variable {d N dv : ℕ}

/-- Value carry: entry `(S, j)` holds `Σ_n λ_S(a_n) k_{n,S} v_{n,j}`. Key `n` has its own scale
`a n`; a single head scale is `a = fun _ => s`. -/
def carryNum (a : Fin N → ℝ) (ks : Fin N → Fin d → ℝ) (vs : Fin N → Fin dv → ℝ) :
    Finset (Fin d) → Fin dv → ℝ :=
  fun S j => ∑ n, lam d (a n) S * chi (ks n) S * vs n j

/-- Normalizer carry: entry `S` holds `Σ_n λ_S(a_n) k_{n,S}`. -/
def carryDen (a : Fin N → ℝ) (ks : Fin N → Fin d → ℝ) : Finset (Fin d) → ℝ :=
  fun S => ∑ n, lam d (a n) S * chi (ks n) S

/-- The direct softmax numerator at value coordinate `j`. -/
def attnNum (a : Fin N → ℝ) (q : Fin d → ℝ) (ks : Fin N → Fin d → ℝ) (vs : Fin N → Fin dv → ℝ)
    (j : Fin dv) : ℝ :=
  ∑ n, Real.exp (a n * ∑ i, q i * ks n i) * vs n j

/-- The direct softmax denominator. -/
def attnDen (a : Fin N → ℝ) (q : Fin d → ℝ) (ks : Fin N → Fin d → ℝ) : ℝ :=
  ∑ n, Real.exp (a n * ∑ i, q i * ks n i)

theorem carry_read_num (a : Fin N → ℝ) (q : Fin d → ℝ) (ks : Fin N → Fin d → ℝ)
    (vs : Fin N → Fin dv → ℝ) (hq : IsSign q) (hk : ∀ n, IsSign (ks n)) (j : Fin dv) :
    ∑ S, chi q S * carryNum a ks vs S j = attnNum a q ks vs j := by
  have h : ∀ n, Real.exp (a n * ∑ i, q i * ks n i) * vs n j
      = ∑ S, chi q S * (lam d (a n) S * chi (ks n) S * vs n j) := by
    intro n
    rw [onebit_exp_expansion (a n) q (ks n) hq (hk n), Finset.sum_mul]
    exact Finset.sum_congr rfl (fun S _ => by ring)
  unfold attnNum carryNum
  rw [Finset.sum_congr rfl (fun n _ => h n)]
  simp_rw [Finset.mul_sum]
  exact Finset.sum_comm

theorem carry_read_den (a : Fin N → ℝ) (q : Fin d → ℝ) (ks : Fin N → Fin d → ℝ)
    (hq : IsSign q) (hk : ∀ n, IsSign (ks n)) :
    ∑ S, chi q S * carryDen a ks S = attnDen a q ks := by
  have h : ∀ n, Real.exp (a n * ∑ i, q i * ks n i)
      = ∑ S, chi q S * (lam d (a n) S * chi (ks n) S) := by
    intro n
    rw [onebit_exp_expansion (a n) q (ks n) hq (hk n)]
    exact Finset.sum_congr rfl (fun S _ => by ring)
  unfold attnDen carryDen
  rw [Finset.sum_congr rfl (fun n _ => h n)]
  simp_rw [Finset.mul_sum]
  exact Finset.sum_comm

/-- **One-bit attention is a finite carry (headline).** Normalized softmax attention of a one-bit
query over `N` one-bit keys with per-key scales equals the read of a carry of `2^d · (d_v + 1)`
reals (`carry_card`), whatever `N` is. -/
theorem onebit_attention_is_finite_carry (a : Fin N → ℝ) (q : Fin d → ℝ)
    (ks : Fin N → Fin d → ℝ) (vs : Fin N → Fin dv → ℝ) (hq : IsSign q)
    (hk : ∀ n, IsSign (ks n)) (j : Fin dv) :
    (∑ S, chi q S * carryNum a ks vs S j) / (∑ S, chi q S * carryDen a ks S)
      = attnNum a q ks vs j / attnDen a q ks := by
  rw [carry_read_num a q ks vs hq hk j, carry_read_den a q ks hq hk]

/-- The carry index has `2^d` entries. -/
theorem carry_card (d : ℕ) : Fintype.card (Finset (Fin d)) = 2 ^ d := by
  rw [Fintype.card_finset, Fintype.card_fin]

/-- **Streaming ingest.** Appending key `N` adds its own features to every carry entry; the
carry is a sum over keys, so removing a key subtracts the same term (exact unlearning). -/
theorem carryNum_snoc (a : Fin (N + 1) → ℝ) (ks : Fin (N + 1) → Fin d → ℝ)
    (vs : Fin (N + 1) → Fin dv → ℝ) (S : Finset (Fin d)) (j : Fin dv) :
    carryNum a ks vs S j
      = carryNum (fun n => a n.castSucc) (fun n => ks n.castSucc) (fun n => vs n.castSucc) S j
        + lam d (a (Fin.last N)) S * chi (ks (Fin.last N)) S * vs (Fin.last N) j := by
  simp only [carryNum]
  rw [Fin.sum_univ_castSucc]

theorem carryDen_snoc (a : Fin (N + 1) → ℝ) (ks : Fin (N + 1) → Fin d → ℝ)
    (S : Finset (Fin d)) :
    carryDen a ks S
      = carryDen (fun n => a n.castSucc) (fun n => ks n.castSucc) S
        + lam d (a (Fin.last N)) S * chi (ks (Fin.last N)) S := by
  simp only [carryDen]
  rw [Fin.sum_univ_castSucc]

/-! ## 4. Degree truncation -/

/-- **Truncation bound.** Keeping the subsets with `|S| ≤ r` leaves an error of at most
`Σ_{|S| > r} cosh^{d-|S|} s · |sinh s|^{|S|}` (each `|q_S k_S| = 1`). -/
theorem onebit_truncation_bound (s : ℝ) (q k : Fin d → ℝ) (hq : IsSign q) (hk : IsSign k)
    (r : ℕ) :
    |Real.exp (s * ∑ i, q i * k i)
        - ∑ S ∈ Finset.univ.filter (fun S : Finset (Fin d) => S.card ≤ r),
            lam d s S * chi q S * chi k S|
      ≤ ∑ S ∈ Finset.univ.filter (fun S : Finset (Fin d) => ¬ S.card ≤ r),
          Real.cosh s ^ (d - S.card) * |Real.sinh s| ^ S.card := by
  have habs : ∀ {x : Fin d → ℝ}, IsSign x → ∀ S, |chi x S| = 1 := by
    intro x hx S
    rw [chi, Finset.abs_prod]
    exact Finset.prod_eq_one (fun i _ => by rcases hx i with h | h <;> simp [h])
  rw [onebit_exp_expansion s q k hq hk,
    ← Finset.sum_filter_add_sum_filter_not Finset.univ (fun S : Finset (Fin d) => S.card ≤ r),
    add_sub_cancel_left]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum (fun S _ => le_of_eq ?_))
  rw [lam, abs_mul, abs_mul, abs_mul, habs hq, habs hk, abs_pow, abs_pow,
    abs_of_pos (Real.cosh_pos s)]
  ring

/-! ## 5. Cost model -/

/-- Words stored by the carry, and MACs of one query read: `2^d (d_v + 1)`. -/
def carryWords (d dv : ℕ) : ℕ := 2 ^ d * (dv + 1)

/-- MACs of one query against an `n`-token KV cache: `d` per score and `d_v` per value. -/
def scanQuery (n d dv : ℕ) : ℕ := n * (d + dv)

/-- The read pays once the context holds at least `2^d` tokens. -/
theorem carry_query_le_scan {n d dv : ℕ} (hn : 2 ^ d ≤ n) (hd : 1 ≤ d) :
    carryWords d dv ≤ scanQuery n d dv := by
  unfold carryWords scanQuery
  exact Nat.mul_le_mul hn (by omega)

/-- MACs of a causal pass over `n` tokens through the carry: each token ingests once and reads
once. Linear in `n`. -/
def carryCausal (n d dv : ℕ) : ℕ := n * (2 * carryWords d dv)

/-- MACs of a causal pass through the scan: token `t` (0-based) reads `t + 1` keys. -/
def scanCausal (n d dv : ℕ) : ℕ := ∑ t ∈ Finset.range n, scanQuery (t + 1) d dv

/-- The scan's causal pass is quadratic: `2 · scanCausal = n (n + 1) (d + d_v)`. -/
theorem scanCausal_closed (n d dv : ℕ) : 2 * scanCausal n d dv = n * (n + 1) * (d + dv) := by
  induction n with
  | zero => simp [scanCausal]
  | succ m ih =>
    rw [scanCausal, Finset.sum_range_succ, ← scanCausal, mul_add, ih, scanQuery]
    ring

/-- **A causal pass pays once `4 · 2^d ≤ n + 1` (headline cost theorem).** -/
theorem carry_causal_le_scan {n d dv : ℕ} (hn : 4 * 2 ^ d ≤ n + 1) (hd : 1 ≤ d) :
    carryCausal n d dv ≤ scanCausal n d dv := by
  have h2 := scanCausal_closed n d dv
  have hv : dv + 1 ≤ d + dv := by omega
  have key : 2 * carryCausal n d dv ≤ n * (n + 1) * (d + dv) := by
    unfold carryCausal carryWords
    calc 2 * (n * (2 * (2 ^ d * (dv + 1)))) = n * ((4 * 2 ^ d) * (dv + 1)) := by ring
      _ ≤ n * ((n + 1) * (d + dv)) := Nat.mul_le_mul_left n (Nat.mul_le_mul hn hv)
      _ = n * (n + 1) * (d + dv) := by ring
  omega

/-- The truncated carry keeps `Σ_{j ≤ r} C(d, j)` subsets. -/
theorem truncated_card (d r : ℕ) :
    (Finset.univ.filter (fun S : Finset (Fin d) => S.card ≤ r)).card
      = ∑ j ∈ Finset.range (r + 1), d.choose j := by
  rw [← Finset.powerset_univ]
  have : (Finset.univ : Finset (Fin d)).powerset.filter (fun S => S.card ≤ r)
      = (Finset.range (r + 1)).biUnion
          (fun j => Finset.powersetCard j (Finset.univ : Finset (Fin d))) := by
    ext S
    simp [Finset.mem_powersetCard, Nat.lt_succ_iff]
  rw [this, Finset.card_biUnion]
  · refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin]
  · intro i _ j _ hij
    exact Finset.disjoint_left.mpr (fun S hi hj => hij
      ((Finset.mem_powersetCard.mp hi).2.symm.trans (Finset.mem_powersetCard.mp hj).2))

/-! ## 6. Joint witnesses -/

/-- **Joint witness.** `d = 2`, `q = (1, -1)`, two keys `(1, 1)` and `(-1, -1)`: every sign
hypothesis holds jointly, and at `d = 8, d_v = 64` the carry (16640 words) is cheaper than a
32768-token read (2359296 MACs) and a 32768-token causal pass. -/
theorem onebit_witness :
    IsSign (![1, -1] : Fin 2 → ℝ)
      ∧ (∀ n : Fin 2, IsSign ((![![1, 1], ![-1, -1]] : Fin 2 → Fin 2 → ℝ) n))
      ∧ carryWords 8 64 = 16640
      ∧ carryWords 8 64 ≤ scanQuery 32768 8 64
      ∧ carryCausal 32768 8 64 ≤ scanCausal 32768 8 64 := by
  refine ⟨?_, ?_, by norm_num [carryWords], carry_query_le_scan (by norm_num) (by norm_num),
    carry_causal_le_scan (by norm_num) (by norm_num)⟩
  · intro i; fin_cases i <;> simp
  · intro n i; fin_cases n <;> fin_cases i <;> simp

end Nous.Model.OneBit
