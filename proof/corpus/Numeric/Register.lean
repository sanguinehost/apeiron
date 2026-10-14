/-
Register stages over `R[x]/(x² − σ)` (`QREG-`, epic the quantum-register note).

A register of `2^n` entries is the restriction of `x : ℕ → R` to the indices `i < 2^n`. A stage at
span `2^s` (`s < n`) pairs `i` with `i + 2^s` for every `i` whose bit `s` is `0`
(`i % 2^(s+1) < 2^s`), reads the pair as the point `(x i, x (i + 2^s))` of `R[x]/(x² − σ)`, and
multiplies it by a stage cell `w` with the canonical `Apeiron.Trichotomy.AnyRing.ringMul`:
`(a, b) · (u, v) = (a u + σ b v, a v + b u)`. `cellStage σ w s` is `pairStage s (ringMul σ · w)`;
the Hadamard stage is `pairStage s (channelPair 1)`.

## Landed here

* `register_stage_normForm`: at `σ = −1`, `Σ_{i<2^n} (stage w x i)² = (w₁² + w₂²) · Σ x i²`, from
  the pairwise two-square identity (Brahmagupta, 628; `ringNormForm_mul`), over every `CommRing`.
* `register_stage_conj`: `stage (conj w) ∘ stage w = ringNormForm σ w • id`, any `σ`, any `CommRing`.
* `register_hadamard`: `n` Hadamard stages `s = 0, …, n−1` equal `(sylvesterH n) *ᵥ x` (Sylvester,
  1867) on `i < 2^n`, via `sylvesterH_mulVec_top/bottom`; `register_hadamard_fwht` restates it
  against `fwhtRef`. `register_stage_diagonal`: at `σ = 1` the Hadamard stage diagonalises the cell
  stage: `H ∘ (w ·) = diag(ringUpper 1 w, ringLower 1 w) ∘ H` on every pair, through
  `channelPair_ringMul_left`. The cell `w = (1, 1)` multiplies a pair to `(u + v, u + v)`; the
  butterfly `(u + v, u − v)` is `channelPair 1`, the idempotent-basis change of `ℤ[x]/(x² − 1)`
  (`sylvester_one_channelPair`), not a multiplication by a cell.
* `iwasawa_faces`: every real `2×2` matrix of determinant `1` is `rep (−1) (cos t, sin t) ·
  diag(a, a⁻¹) · (rep 0 (1, v))ᵀ` with `a > 0` (Iwasawa, 1949; `K`, `A`, `N` are the elliptic, the
  hyperbolic and the parabolic face). `rep_parabolic_is_shear`: `rep 0 (1, v) = !![1, 0; v, 1]`.
  `diag_eq_hadamard_conj_rep_hyperbolic`: `diag(eᵗ, e⁻ᵗ) = ½ H · rep 1 (cosh t, sinh t) · H`.
* `register_cost`: `registerMuls n = n·2^(n−1)·4`, `registerAdds n = n·2^(n−1)·2`; `register_stage_pairs`:
  a stage has `2^(n−1)` pairs.
* `reflPair_involutive`, `reflStage_involutive`: at `c² + d² = 1` the reflection pair map
  `(u, v) ↦ (c u + d v, d u − c v)` is an involution, and so is its stage (`pairStage_comp`).
* `reflStage_sum_sq`: a reflection stage with `c² + d² = 1` preserves `Σ_{i<2^n} x i²`.
* `reflStages_reverse_cancel`: for a list of reflection stages each with `c² + d² = 1`, the reversed
  list after the list is the identity (induction on the list, `reflStep_involutive`).
* `refl_quarter`, `quarter_sq`: at `c = d = 1/√2` the reflection is `(1/√2) ·` the butterfly
  `(u, v) ↦ (u + v, u − v)`, and `c² + d² = 1`.
* `cz_involutive`, `cz_sum_sq`: the controlled-`Z` gate `czStage a b` is an involution that preserves
  `Σ x i²`.
* `cz_entangles`: on `(1, 1, 1, 1)` the CZ output `(1, 1, 1, −1)` has `y 0 · y 3 ≠ y 1 · y 2`, while
  every product register `productState u v` satisfies `x 0 · x 3 = x 1 · x 2`
  (`productState_cross`).
* `measProj_idem`, `measProj_add`, `measProj_orth`: the projectors of bit `q` are idempotent, sum to
  the identity and are orthogonal.
* `born_sum`: the two outcome registers of bit `q` carry the total `Σ_{i<2^n} x i²`.

## Complexity

One stage at span `2^s` on `2^n` entries costs `2^(n−1)` cell multiplies, each 4 multiplications
and 2 additions (`σ ∈ {−1, 0, 1}` acts as a sign or a selection, not a multiplication); `n` stages
cost `n·2^(n−1)` cell multiplies. For `n = 7`: `448` cell multiplies, `1792` multiplications,
`896` additions (`register_cost`). `pairStage` evaluates the pair map once per output entry, so the
Lean definition does `2^n` evaluations per stage; the cost model counts each pair once.

A reflection stage costs as a cell stage; `reflStages` over `m` stages costs `m·2^n` pair-map
evaluations (not `O(1)` per entry). `czStage`, `productState` and `measProj` are `O(1)` per entry.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom`, `decide` over an
extracted body, or `maxHeartbeats 0`.
-/
import Numeric.HadamardStage
import Numeric.OneBitUnits
import Apeiron.Physics.Quadratic
import Mathlib.Tactic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Apeiron.Trichotomy.AnyRing
open Nous.Model
open scoped Matrix

namespace Register

/-! ## The pair stage -/

section PairStage

variable {R : Type*} [CommRing R]

/-- One stage at span `2^s`: the pair `(x i, x (i + 2^s))` at every `i` with bit `s` clear is sent
through `f`; entry `i` is the first component of `f` at the pair that holds it as its low element,
or the second component of `f` at the pair that holds it as its high element.
Complexity: `O(1)` pair-map evaluations per entry, `2^n` entries per register, counted in evaluations
of `f`; the cost model counts `2^(n−1)` (`register_stage_pairs`). -/
def pairStage (s : ℕ) (f : R × R → R × R) (x : ℕ → R) : ℕ → R := fun i =>
  if i % 2 ^ (s + 1) < 2 ^ s then (f (x i, x (i + 2 ^ s))).1
  else (f (x (i - 2 ^ s), x i)).2

/-- The cell stage: every pair is multiplied by the cell `w` in `R[x]/(x² − σ)`. -/
def cellStage (σ : R) (w : RingCell R) (s : ℕ) (x : ℕ → R) : ℕ → R :=
  pairStage s (fun z => ringMul σ z w) x

omit [CommRing R] in
theorem pairStage_low {s i : ℕ} (f : R × R → R × R) (x : ℕ → R)
    (hl : i % 2 ^ (s + 1) < 2 ^ s) :
    pairStage s f x i = (f (x i, x (i + 2 ^ s))).1 := by
  simp [pairStage, hl]

omit [CommRing R] in
theorem pairStage_high {s i : ℕ} (f : R × R → R × R) (x : ℕ → R)
    (hh : ¬ i % 2 ^ (s + 1) < 2 ^ s) :
    pairStage s f x i = (f (x (i - 2 ^ s), x i)).2 := by
  simp [pairStage, hh]

theorem mod_add_low {i s : ℕ} (h : i % 2 ^ (s + 1) < 2 ^ s) :
    (i + 2 ^ s) % 2 ^ (s + 1) = i % 2 ^ (s + 1) + 2 ^ s := by
  have hm : 2 ^ (s + 1) = 2 * 2 ^ s := by rw [pow_succ]; ring
  have hi := Nat.mod_add_div i (2 ^ (s + 1))
  have e : i + 2 ^ s = (i % 2 ^ (s + 1) + 2 ^ s) + 2 ^ (s + 1) * (i / 2 ^ (s + 1)) := by omega
  rw [e, Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt (by omega)

theorem mod_sub_high {i s : ℕ} (h : ¬ i % 2 ^ (s + 1) < 2 ^ s) :
    (i - 2 ^ s) % 2 ^ (s + 1) = i % 2 ^ (s + 1) - 2 ^ s := by
  have hm : 2 ^ (s + 1) = 2 * 2 ^ s := by rw [pow_succ]; ring
  have hi := Nat.mod_add_div i (2 ^ (s + 1))
  have hlt : i % 2 ^ (s + 1) < 2 ^ (s + 1) := Nat.mod_lt _ (by positivity)
  have e : i - 2 ^ s = (i % 2 ^ (s + 1) - 2 ^ s) + 2 ^ (s + 1) * (i / 2 ^ (s + 1)) := by omega
  rw [e, Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt (by omega)

omit [CommRing R] in
/-- A stage after a stage at the same span is the stage of the composite pair map. -/
theorem pairStage_comp (s : ℕ) (f g : R × R → R × R) (x : ℕ → R) :
    pairStage s f (pairStage s g x) = pairStage s (f ∘ g) x := by
  funext i
  by_cases hl : i % 2 ^ (s + 1) < 2 ^ s
  · have h2 : ¬ (i + 2 ^ s) % 2 ^ (s + 1) < 2 ^ s := by rw [mod_add_low hl]; omega
    have e1 : pairStage s g x i = (g (x i, x (i + 2 ^ s))).1 := pairStage_low g x hl
    have e2 : pairStage s g x (i + 2 ^ s) = (g (x i, x (i + 2 ^ s))).2 := by
      rw [pairStage_high g x h2, Nat.add_sub_cancel]
    rw [pairStage_low f _ hl, e1, e2, pairStage_low (f ∘ g) x hl]
    rfl
  · have hle : 2 ^ s ≤ i := by
      have := Nat.mod_le i (2 ^ (s + 1)); omega
    have h2 : (i - 2 ^ s) % 2 ^ (s + 1) < 2 ^ s := by
      rw [mod_sub_high hl]
      have hm : 2 ^ (s + 1) = 2 * 2 ^ s := by rw [pow_succ]; ring
      have := Nat.mod_lt i (show 0 < 2 ^ (s + 1) by positivity)
      omega
    have e1 : pairStage s g x i = (g (x (i - 2 ^ s), x i)).2 := pairStage_high g x hl
    have e2 : pairStage s g x (i - 2 ^ s) = (g (x (i - 2 ^ s), x i)).1 := by
      rw [pairStage_low g x h2, Nat.sub_add_cancel hle]
    rw [pairStage_high f _ hl, e1, e2, pairStage_high (f ∘ g) x hl]
    rfl

theorem pairStage_smul (s : ℕ) (c : R) (x : ℕ → R) :
    pairStage s (fun z => (c * z.1, c * z.2)) x = c • x := by
  funext i
  by_cases hl : i % 2 ^ (s + 1) < 2 ^ s
  · rw [pairStage_low _ x hl]; rfl
  · rw [pairStage_high _ x hl]; rfl

/-! ## Summing over the pairs of a register -/

theorem add_lt_pow {a s n : ℕ} (hs : s < n) (ha : a < 2 ^ n)
    (hl : a % 2 ^ (s + 1) < 2 ^ s) : a + 2 ^ s < 2 ^ n := by
  obtain ⟨c, hc⟩ := pow_dvd_pow 2 (show s + 1 ≤ n from hs)
  have hm : 2 ^ (s + 1) = 2 * 2 ^ s := by rw [pow_succ]; ring
  have hpos : 0 < 2 ^ (s + 1) := by positivity
  have hq : a / 2 ^ (s + 1) < c := by
    rw [Nat.div_lt_iff_lt_mul hpos]
    rw [mul_comm, ← hc]; exact ha
  have hi := Nat.mod_add_div a (2 ^ (s + 1))
  have h1 : 2 ^ (s + 1) * (a / 2 ^ (s + 1) + 1) ≤ 2 ^ (s + 1) * c :=
    Nat.mul_le_mul_left _ hq
  have h2 : 2 ^ (s + 1) * (a / 2 ^ (s + 1) + 1)
      = 2 ^ (s + 1) * (a / 2 ^ (s + 1)) + 2 ^ (s + 1) := by ring
  omega

/-- A register sum is the sum over its pairs. -/
theorem sum_pairs {M : Type*} [AddCommMonoid M] {s n : ℕ} (hs : s < n) (F : ℕ → M) :
    ∑ i ∈ Finset.range (2 ^ n), F i =
      ∑ i ∈ (Finset.range (2 ^ n)).filter (fun i => i % 2 ^ (s + 1) < 2 ^ s),
        (F i + F (i + 2 ^ s)) := by
  have hm : 2 ^ (s + 1) = 2 * 2 ^ s := by rw [pow_succ]; ring
  rw [← Finset.sum_filter_add_sum_filter_not (Finset.range (2 ^ n))
    (fun i => i % 2 ^ (s + 1) < 2 ^ s) F, Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_nbij' (fun i => i - 2 ^ s) (fun i => i + 2 ^ s) ?_ ?_ ?_ ?_ ?_
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_range] at ha ⊢
    obtain ⟨ha1, ha2⟩ := ha
    refine ⟨by omega, ?_⟩
    rw [mod_sub_high ha2]
    have := Nat.mod_lt a (show 0 < 2 ^ (s + 1) by positivity)
    omega
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_range] at ha ⊢
    obtain ⟨ha1, ha2⟩ := ha
    refine ⟨add_lt_pow hs ha1 ha2, ?_⟩
    rw [mod_add_low ha2]
    omega
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_range] at ha
    have := Nat.mod_le a (2 ^ (s + 1))
    omega
  · intro a _
    omega
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_range] at ha
    have := Nat.mod_le a (2 ^ (s + 1))
    have e : a - 2 ^ s + 2 ^ s = a := by omega
    rw [e]

/-- A stage has `2^(n−1)` pairs. -/
theorem register_stage_pairs {s n : ℕ} (hs : s < n) :
    ((Finset.range (2 ^ n)).filter (fun i => i % 2 ^ (s + 1) < 2 ^ s)).card = 2 ^ (n - 1) := by
  have h := sum_pairs (M := ℕ) hs (fun _ => 1)
  simp only [Finset.sum_const, Finset.card_range, smul_eq_mul, mul_one] at h
  have hn : 2 ^ n = 2 * 2 ^ (n - 1) := by
    obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
    rw [pow_succ]; simp; ring
  omega

/-- A stage whose pair map scales the sum of squares by `c` scales the register's sum of squares
by `c`. -/
theorem pairStage_sum_sq {s n : ℕ} (hs : s < n) (f : R × R → R × R) (c : R)
    (hf : ∀ z : R × R, (f z).1 ^ 2 + (f z).2 ^ 2 = c * (z.1 ^ 2 + z.2 ^ 2)) (x : ℕ → R) :
    ∑ i ∈ Finset.range (2 ^ n), (pairStage s f x i) ^ 2 =
      c * ∑ i ∈ Finset.range (2 ^ n), (x i) ^ 2 := by
  rw [sum_pairs hs (fun i => (pairStage s f x i) ^ 2), sum_pairs hs (fun i => (x i) ^ 2),
    Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  have hl : i % 2 ^ (s + 1) < 2 ^ s := (Finset.mem_filter.1 hi).2
  have h2 : ¬ (i + 2 ^ s) % 2 ^ (s + 1) < 2 ^ s := by rw [mod_add_low hl]; omega
  rw [pairStage_low f x hl, pairStage_high f x h2, Nat.add_sub_cancel]
  exact hf (x i, x (i + 2 ^ s))

/-! ## Theorem 1: the two-square identity on a register -/

/-- **Elliptic stages scale the squared norm.** At `σ = −1` a cell stage multiplies
`Σ_{i<2^n} x i²` by `ringNormForm (-1) w = w₁² + w₂²`. The sum is over the `2^n` entries of the
register, a stage span `s < n`, over any commutative ring (the identity is polynomial). -/
theorem register_stage_normForm {s n : ℕ} (hs : s < n) (w : RingCell R) (x : ℕ → R) :
    ∑ i ∈ Finset.range (2 ^ n), (cellStage (-1 : R) w s x i) ^ 2 =
      ringNormForm (-1 : R) w * ∑ i ∈ Finset.range (2 ^ n), (x i) ^ 2 := by
  apply pairStage_sum_sq hs
  intro z
  have h := ringNormForm_mul (-1 : R) z w
  simp only [ringNormForm] at h ⊢
  linear_combination h

/-! ## Theorem 2: the conjugate stage inverts up to the norm form -/

/-- **A stage followed by its conjugate stage is the norm form.** For every `σ`: the stage by
`conj w` after the stage by `w` is multiplication of the register by `ringNormForm σ w`. -/
theorem register_stage_conj (σ : R) (w : RingCell R) (s : ℕ) (x : ℕ → R) :
    cellStage σ (ringConj w) s (cellStage σ w s x) = ringNormForm σ w • x := by
  unfold cellStage
  rw [pairStage_comp]
  have e : ((fun z : R × R => ringMul σ z (ringConj w)) ∘ (fun z : R × R => ringMul σ z w))
      = fun z => (ringNormForm σ w * z.1, ringNormForm σ w * z.2) := by
    funext z
    apply Prod.ext <;> simp only [Function.comp, ringMul, ringNormForm] <;> ring
  rw [e]
  exact pairStage_smul s _ x

end PairStage

/-! ## Theorem 3: the Hadamard stages are the Sylvester matrix -/

/-- The Hadamard stage at span `2^s`: `channelPair 1` on every pair, the butterfly
`(u, v) ↦ (u + v, u − v)`. -/
def hadStage (s : ℕ) (x : ℕ → ℤ) : ℕ → ℤ := pairStage s (channelPair (1 : ℤ)) x

/-- The stages `0, …, n−1` in order.
Complexity: `n·2^(n−1)` pair evaluations (`n` stages of `2^(n−1)` pairs). -/
def hadRef : ℕ → (ℕ → ℤ) → (ℕ → ℤ)
  | 0, x => x
  | k + 1, x => hadStage k (hadRef k x)

@[simp] theorem hadRef_zero (x : ℕ → ℤ) : hadRef 0 x = x := rfl

theorem hadRef_succ (k : ℕ) (x : ℕ → ℤ) : hadRef (k + 1) x = hadStage k (hadRef k x) := rfl

/-- Stages below `2^k` do not see a block offset that is a multiple of `2^k`. -/
theorem hadRef_shift (k : ℕ) : ∀ (x : ℕ → ℤ) (c j : ℕ), 2 ^ k ∣ c →
    hadRef k x (c + j) = hadRef k (fun t => x (c + t)) j := by
  induction k with
  | zero => intro x c j _; rfl
  | succ k ih =>
    intro x c j hc
    have hc' : 2 ^ k ∣ c := (pow_dvd_pow 2 (Nat.le_succ k)).trans hc
    have hmod : (c + j) % 2 ^ (k + 1) = j % 2 ^ (k + 1) := by
      obtain ⟨d, rfl⟩ := hc
      rw [Nat.mul_add_mod]
    rw [hadRef_succ, hadRef_succ]
    unfold hadStage
    by_cases hl : j % 2 ^ (k + 1) < 2 ^ k
    · rw [pairStage_low _ _ (by rw [hmod]; exact hl), pairStage_low _ _ hl, add_assoc,
        ih x c j hc', ih x c (j + 2 ^ k) hc']
    · have hle : 2 ^ k ≤ j := by
        have := Nat.mod_le j (2 ^ (k + 1)); omega
      have e : c + j - 2 ^ k = c + (j - 2 ^ k) := by omega
      rw [pairStage_high _ _ (by rw [hmod]; exact hl), pairStage_high _ _ hl, e,
        ih x c (j - 2 ^ k) hc', ih x c j hc']

/-- **The Hadamard stages are the Sylvester matrix.** On `i < 2^n`, the `n` butterfly stages
`s = 0, …, n−1` give `(sylvesterH n) *ᵥ x`. Proved for all `n` by induction on the top stage
(`sylvesterH_mulVec_top/bottom`), with the lower stages acting on the two halves by `hadRef_shift`. -/
theorem register_hadamard (n : ℕ) : ∀ (x : ℕ → ℤ) (i : ℕ) (hi : i < 2 ^ n),
    hadRef n x i = (sylvesterH n).mulVec (fun j : Fin (2 ^ n) => x j.val) ⟨i, hi⟩ := by
  induction n with
  | zero =>
    intro x i hi
    have hi0 : i = 0 := by simpa using hi
    subst hi0
    rw [sylvester_zero, Matrix.one_mulVec]
    rfl
  | succ k ih =>
    intro x i hi
    have hm : 2 ^ (k + 1) = 2 * 2 ^ k := by rw [pow_succ]; ring
    have hmod : i % 2 ^ (k + 1) = i := Nat.mod_eq_of_lt hi
    rw [hadRef_succ]
    unfold hadStage
    by_cases hl : i < 2 ^ k
    · have hl' : i % 2 ^ (k + 1) < 2 ^ k := by rw [hmod]; exact hl
      have htop := sylvesterH_mulVec_top k (fun j : Fin (2 ^ (k + 1)) => x j.val) ⟨i, hl⟩
      have hfin : hTop k ⟨i, hl⟩ = ⟨i, hi⟩ := Fin.ext rfl
      rw [hfin] at htop
      rw [htop, pairStage_low _ _ hl']
      have e1 := ih x i hl
      have e2 := ih (fun t => x (2 ^ k + t)) i hl
      have e3 := hadRef_shift k x (2 ^ k) i dvd_rfl
      have e4 : hadRef k x (i + 2 ^ k) = hadRef k x (2 ^ k + i) := by rw [add_comm]
      simp only [channelPair, ringUpper, one_mul]
      rw [e4, e3, e1, e2]
      rfl
    · have hl' : ¬ i % 2 ^ (k + 1) < 2 ^ k := by rw [hmod]; exact hl
      have hj : i - 2 ^ k < 2 ^ k := by omega
      have hbot := sylvesterH_mulVec_bottom k (fun j : Fin (2 ^ (k + 1)) => x j.val)
        ⟨i - 2 ^ k, hj⟩
      have hfin : hBot k ⟨i - 2 ^ k, hj⟩ = ⟨i, hi⟩ := Fin.ext (by simp only [hBot_val]; omega)
      rw [hfin] at hbot
      rw [hbot, pairStage_high _ _ hl']
      have e1 := ih x (i - 2 ^ k) hj
      have e2 := ih (fun t => x (2 ^ k + t)) (i - 2 ^ k) hj
      have e3 := hadRef_shift k x (2 ^ k) (i - 2 ^ k) dvd_rfl
      have e4 : 2 ^ k + (i - 2 ^ k) = i := by omega
      rw [e4] at e3
      simp only [channelPair, ringLower, one_mul]
      rw [e3, e1, e2]
      rfl

/-- `register_hadamard` against the list reference `fwhtRef`. -/
theorem register_hadamard_fwht (n : ℕ) (x : ℕ → ℤ) (i : ℕ) (hi : i < 2 ^ n) :
    hadRef n x i = (fwhtRef n ((List.range (2 ^ n)).map x))[i]'(by
      rw [fwhtRef_length]; simpa using hi) := by
  have hlen : ((List.range (2 ^ n)).map x).length = 2 ^ n := by simp
  rw [fwhtRef_getElem n _ hlen i (by rw [hlen]; exact hi), register_hadamard n x i hi]
  congr 1
  funext j
  simp

/-- **The Hadamard stage diagonalises the cell stage at `σ = 1`.** On the split face the butterfly
sends multiplication by `w` to the componentwise scaling by `(ringUpper 1 w, ringLower 1 w)`. -/
theorem register_stage_diagonal (w : RingCell ℤ) (s : ℕ) (x : ℕ → ℤ) :
    hadStage s (cellStage (1 : ℤ) w s x) =
      pairStage s (fun p => (ringUpper (1 : ℤ) w * p.1, ringLower (1 : ℤ) w * p.2))
        (hadStage s x) := by
  unfold hadStage cellStage
  rw [pairStage_comp, pairStage_comp]
  congr 1
  funext z
  simp only [Function.comp]
  rw [ringMul_comm, OneBitUnits.channelPair_ringMul_left]

/-! ## Theorem 4: the three faces factor `SL₂(ℝ)` -/

open Apeiron.Physics in
/-- The parabolic cell `(1, v)` at `σ = 0` represents as the shear `!![1, 0; v, 1]`. -/
theorem rep_parabolic_is_shear (v : ℝ) :
    rep 0 ((1 : ℝ), v) = !![(1 : ℝ), 0; v, 1] := by
  simp [rep]

open Apeiron.Physics in
/-- The positive diagonal is the Hadamard conjugate of the hyperbolic cell:
`diag(eᵗ, e⁻ᵗ) = ½ H · rep 1 (cosh t, sinh t) · H`. -/
theorem diag_eq_hadamard_conj_rep_hyperbolic (t : ℝ) :
    !![Real.exp t, 0; 0, Real.exp (-t)] =
      (1 / 2 : ℝ) • (!![(1 : ℝ), 1; 1, -1] * rep 1 (Real.cosh t, Real.sinh t)
        * !![(1 : ℝ), 1; 1, -1]) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rep, Real.cosh_eq, Real.sinh_eq] <;> ring

open Apeiron.Physics in
/-- **Iwasawa decomposition from the three faces.** A real `2×2` matrix of determinant `1` is a
rotation (elliptic cell `(cos t, sin t)`), then a positive diagonal, then the transposed parabolic
cell `(rep 0 (1, v))ᵀ = !![1, v; 0, 1]` (Iwasawa 1949, `SL₂(ℝ) = K A N`). Existence: Gram–Schmidt
on the columns, the angle from `Complex.arg` of the first column. -/
theorem iwasawa_faces (M : Matrix (Fin 2) (Fin 2) ℝ) (hM : M.det = 1) :
    ∃ t a v : ℝ, 0 < a ∧
      M = rep (-1) (Real.cos t, Real.sin t) * !![a, 0; 0, a⁻¹] * (rep 0 ((1 : ℝ), v))ᵀ := by
  obtain ⟨p, q, r, u, rfl⟩ : ∃ p q r u : ℝ, M = !![p, q; r, u] :=
    ⟨M 0 0, M 0 1, M 1 0, M 1 1, by ext i j; fin_cases i <;> fin_cases j <;> rfl⟩
  rw [Matrix.det_fin_two_of] at hM
  have hz0 : (⟨p, r⟩ : ℂ) ≠ 0 := by
    intro h0
    have h1 : p = 0 := congrArg Complex.re h0
    have h2 : r = 0 := congrArg Complex.im h0
    rw [h1, h2] at hM
    norm_num at hM
  have hρ : 0 < ‖(⟨p, r⟩ : ℂ)‖ := norm_pos_iff.2 hz0
  have hρ2 : ‖(⟨p, r⟩ : ℂ)‖ ^ 2 = p ^ 2 + r ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    ring
  have hc : Real.cos (Complex.arg ⟨p, r⟩) = p / ‖(⟨p, r⟩ : ℂ)‖ := Complex.cos_arg hz0
  have hs : Real.sin (Complex.arg ⟨p, r⟩) = r / ‖(⟨p, r⟩ : ℂ)‖ := Complex.sin_arg _
  generalize Complex.arg ⟨p, r⟩ = t at hc hs
  generalize ‖(⟨p, r⟩ : ℂ)‖ = ρ at hρ hρ2 hc hs
  refine ⟨t, ρ, (p * q + r * u) / ρ ^ 2, hρ, ?_⟩
  have hne : ρ ≠ 0 := hρ.ne'
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rep, Matrix.mul_apply, Fin.sum_univ_two, hc, hs, Matrix.transpose_apply]
  · field_simp
  · field_simp
    linear_combination q * hρ2 - r * hM
  · field_simp
  · field_simp
    linear_combination u * hρ2 + p * hM

open Apeiron.Physics in
/-- The same factorisation with the diagonal written as the Hadamard conjugate of the hyperbolic
cell `rep 1 (cosh s, sinh s)`. -/
theorem iwasawa_faces_hadamard (M : Matrix (Fin 2) (Fin 2) ℝ) (hM : M.det = 1) :
    ∃ t s v : ℝ,
      M = rep (-1) (Real.cos t, Real.sin t) *
        ((1 / 2 : ℝ) • (!![(1 : ℝ), 1; 1, -1] * rep 1 (Real.cosh s, Real.sinh s)
          * !![(1 : ℝ), 1; 1, -1])) * (rep 0 ((1 : ℝ), v))ᵀ := by
  obtain ⟨t, a, v, ha, h⟩ := iwasawa_faces M hM
  refine ⟨t, Real.log a, v, ?_⟩
  rw [← diag_eq_hadamard_conj_rep_hyperbolic, Real.exp_log ha, Real.exp_neg, Real.exp_log ha]
  exact h

/-! ## Theorem 5: the cost model -/

/-- Multiplications of `n` stages: `n·2^(n−1)` cell multiplies of 4 products. -/
def registerMuls (n : ℕ) : ℕ := n * 2 ^ (n - 1) * 4

/-- Additions of `n` stages: `n·2^(n−1)` cell multiplies of 2 sums. -/
def registerAdds (n : ℕ) : ℕ := n * 2 ^ (n - 1) * 2

/-- A 128-entry register: `7·2^6 = 448` cell multiplies, `1792` multiplications, `896` additions. -/
theorem register_cost :
    registerMuls 7 = 1792 ∧ registerAdds 7 = 896 ∧ 7 * 2 ^ (7 - 1) = 448 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [registerMuls, registerAdds]

/-- Non-vacuity of `register_stage_normForm`: the `n = 2`, `s = 1` stage over `ℤ` with the cell
`(3, 4)` has norm form `25`, and the sums of squares agree for the register `(1, 2, 3, 4)`. -/
example : ∑ i ∈ Finset.range (2 ^ 2),
    (cellStage (-1 : ℤ) (3, 4) 1 (fun i => (i : ℤ) + 1) i) ^ 2 =
      25 * ∑ i ∈ Finset.range (2 ^ 2), (((i : ℤ) + 1)) ^ 2 := by
  simp [Finset.sum_range_succ, cellStage, pairStage]

/-! ## Reflection stages: the real meridian (`QREG-`, row 13 and row 18) -/

section Reflection

variable {R : Type*} [CommRing R]

/-- The reflection pair map `(u, v) ↦ (c u + d v, d u − c v)`: the butterfly `[[c, d], [d, −c]]`
(`= c·σ_z + d·σ_x`, the pair map of `tools/onebit/kan_probe.py` `register`). At `c² + d² = 1` it is
an involution of `R × R` (`reflPair_involutive`). -/
def reflPair (c d : R) (p : R × R) : R × R := (c * p.1 + d * p.2, d * p.1 - c * p.2)

theorem reflPair_involutive {c d : R} (h : c ^ 2 + d ^ 2 = 1) (p : R × R) :
    reflPair c d (reflPair c d p) = p := by
  obtain ⟨u, v⟩ := p
  simp only [reflPair, Prod.mk.injEq]
  constructor
  · linear_combination u * h
  · linear_combination v * h

omit [CommRing R] in
/-- The identity pair map leaves a register unchanged. -/
theorem pairStage_id (s : ℕ) (x : ℕ → R) : pairStage s id x = x := by
  funext i
  by_cases hl : i % 2 ^ (s + 1) < 2 ^ s
  · rw [pairStage_low _ x hl]; rfl
  · rw [pairStage_high _ x hl]; rfl

/-- A reflection stage at span `2^s` with `c² + d² = 1` is an involution of the register. -/
theorem reflStage_involutive (s : ℕ) {c d : R} (h : c ^ 2 + d ^ 2 = 1) (x : ℕ → R) :
    pairStage s (reflPair c d) (pairStage s (reflPair c d) x) = x := by
  have e : reflPair c d ∘ reflPair c d = id := by
    funext p
    exact reflPair_involutive h p
  rw [pairStage_comp, e, pairStage_id]

/-- A reflection stage with `c² + d² = 1` preserves the register's sum of squares. -/
theorem reflStage_sum_sq {s n : ℕ} (hs : s < n) {c d : R} (h : c ^ 2 + d ^ 2 = 1) (x : ℕ → R) :
    ∑ i ∈ Finset.range (2 ^ n), (pairStage s (reflPair c d) x i) ^ 2 =
      ∑ i ∈ Finset.range (2 ^ n), (x i) ^ 2 := by
  have key := pairStage_sum_sq hs (reflPair c d) 1 (fun z => by
    simp only [reflPair, one_mul]
    linear_combination (z.1 ^ 2 + z.2 ^ 2) * h) x
  rw [key, one_mul]

/-- One reflection stage `(span, c, d)` applied to a register. -/
def reflStep (y : ℕ → R) (e : ℕ × R × R) : ℕ → R := pairStage e.1 (reflPair e.2.1 e.2.2) y

/-- A list of reflection stages applied in list order, the head first.
Complexity: `m` stages of `2^(n−1)` pairs each for a register of `2^n` entries, so `m·2^n`
evaluations of `reflPair` (`pairStage` evaluates the pair map once per entry); not `O(1)` per entry. -/
def reflStages (L : List (ℕ × R × R)) (x : ℕ → R) : ℕ → R := L.foldl reflStep x

theorem reflStep_involutive (e : ℕ × R × R) (he : e.2.1 ^ 2 + e.2.2 ^ 2 = 1) (x : ℕ → R) :
    reflStep (reflStep x e) e = x :=
  reflStage_involutive e.1 he x

theorem reflStages_append (L M : List (ℕ × R × R)) (x : ℕ → R) :
    reflStages (L ++ M) x = reflStages M (reflStages L x) := by
  unfold reflStages
  rw [List.foldl_append]

/-- **Reflection stages invert in reverse.** If every stage of `L` has `c² + d² = 1`, applying
the reversed list after `L` returns the register. Induction on `L`. -/
theorem reflStages_reverse_cancel (L : List (ℕ × R × R)) :
    (∀ e ∈ L, e.2.1 ^ 2 + e.2.2 ^ 2 = 1) →
      ∀ x : ℕ → R, reflStages L.reverse (reflStages L x) = x := by
  induction L with
  | nil =>
    intro _ x
    rfl
  | cons a L ih =>
    intro hL x
    have ha : a.2.1 ^ 2 + a.2.2 ^ 2 = 1 := hL a (by simp)
    have hL' : ∀ e ∈ L, e.2.1 ^ 2 + e.2.2 ^ 2 = 1 := fun e he => hL e (by simp [he])
    have e1 : reflStages (a :: L) x = reflStages L (reflStep x a) := rfl
    rw [List.reverse_cons, reflStages_append, e1, ih hL' (reflStep x a)]
    exact reflStep_involutive a ha x

end Reflection

/-- The real meridian at `c = d = 1/√2`: the reflection is `(1/√2) ·` the butterfly
`(u, v) ↦ (u + v, u − v)`. -/
theorem refl_quarter (u v : ℝ) :
    reflPair ((1 : ℝ) / Real.sqrt 2) ((1 : ℝ) / Real.sqrt 2) (u, v) =
      ((u + v) / Real.sqrt 2, (u - v) / Real.sqrt 2) := by
  simp only [reflPair, Prod.mk.injEq]
  constructor <;> ring

theorem quarter_sq : ((1 : ℝ) / Real.sqrt 2) ^ 2 + ((1 : ℝ) / Real.sqrt 2) ^ 2 = 1 := by
  rw [div_pow, one_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  norm_num

/-! ## Controlled-Z and product states (`QREG-`, row 16) -/

section ControlledZ

variable {R : Type*} [CommRing R]

/-- The controlled-`Z` gate at bits `a` and `b`: entry `i` is negated when bit `a` and bit `b` of
`i` are both `1`, and is fixed otherwise. Complexity: `O(1)` per entry. -/
def czStage (a b : ℕ) (x : ℕ → R) : ℕ → R := fun i =>
  if i / 2 ^ a % 2 = 1 ∧ i / 2 ^ b % 2 = 1 then -x i else x i

/-- The product register `i ↦ u (i % 2) · v (i / 2)` of two single-bit registers. Complexity:
`O(1)` per entry. -/
def productState (u v : ℕ → R) (i : ℕ) : R := u (i % 2) * v (i / 2)

theorem cz_involutive (a b : ℕ) (x : ℕ → R) : czStage a b (czStage a b x) = x := by
  funext i
  by_cases h : i / 2 ^ a % 2 = 1 ∧ i / 2 ^ b % 2 = 1 <;> simp [czStage, h]

theorem cz_sum_sq (a b n : ℕ) (x : ℕ → R) :
    ∑ i ∈ Finset.range (2 ^ n), (czStage a b x i) ^ 2 =
      ∑ i ∈ Finset.range (2 ^ n), (x i) ^ 2 := by
  apply Finset.sum_congr rfl
  intro i _
  unfold czStage
  split_ifs <;> ring

theorem productState_cross (u v : ℕ → R) :
    productState u v 0 * productState u v 3 = productState u v 1 * productState u v 2 := by
  simp only [productState, Nat.reduceMod, Nat.reduceDiv]
  ring

/-- **CZ entangles.** On `x = (1, 1, 1, 1)` (indices `0..3`, bits `a = 0`, `b = 1`) the output is
`(1, 1, 1, −1)`, with `y 0 · y 3 ≠ y 1 · y 2`; every product register satisfies
`x 0 · x 3 = x 1 · x 2`, so the CZ output is not a product state. -/
theorem cz_entangles :
    czStage 0 1 (fun _ => (1 : ℤ)) 0 * czStage 0 1 (fun _ => (1 : ℤ)) 3 ≠
        czStage 0 1 (fun _ => (1 : ℤ)) 1 * czStage 0 1 (fun _ => (1 : ℤ)) 2 ∧
      ∀ u v : ℕ → ℤ,
        productState u v 0 * productState u v 3 = productState u v 1 * productState u v 2 := by
  refine ⟨?_, fun u v => productState_cross u v⟩
  norm_num [czStage]

end ControlledZ

/-! ## Measurement (`QREG-`, row 17) -/

section Measurement

variable {R : Type*} [CommRing R]

/-- The projective measurement of bit `q` with outcome `b`: entry `i` keeps its value when bit `q`
of `i` equals `b` (`decide (i / 2^q % 2 = 1) = b`) and is zeroed otherwise. Complexity: `O(1)` per
entry. -/
def measProj (q : ℕ) (b : Bool) (x : ℕ → R) : ℕ → R := fun i =>
  if decide (i / 2 ^ q % 2 = 1) = b then x i else 0

theorem measProj_idem (q : ℕ) (b : Bool) (x : ℕ → R) :
    measProj q b (measProj q b x) = measProj q b x := by
  funext i
  by_cases h : i / 2 ^ q % 2 = 1 <;> cases b <;> simp [measProj, h]

theorem measProj_add (q : ℕ) (x : ℕ → R) : measProj q false x + measProj q true x = x := by
  funext i
  by_cases h : i / 2 ^ q % 2 = 1 <;> simp [measProj, h]

theorem measProj_orth (q : ℕ) (x : ℕ → R) : measProj q true (measProj q false x) = 0 := by
  funext i
  by_cases h : i / 2 ^ q % 2 = 1 <;> simp [measProj, h]

/-- **Born rule on a register.** The two outcome registers of bit `q` carry the total sum of
squares of `x`. -/
theorem born_sum (q n : ℕ) (x : ℕ → R) :
    ∑ i ∈ Finset.range (2 ^ n), (measProj q false x i) ^ 2 +
      ∑ i ∈ Finset.range (2 ^ n), (measProj q true x i) ^ 2 =
    ∑ i ∈ Finset.range (2 ^ n), (x i) ^ 2 := by
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  by_cases h : i / 2 ^ q % 2 = 1 <;> simp [measProj, h]

end Measurement

end Register

end apeiron_word
