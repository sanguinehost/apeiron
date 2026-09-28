import Nous.Model.State
import Nous.Model.Decomposition

/-!
# Selective Linear Recurrence and Its Associative Scan

A selective linear recurrence over the trichotomy basis carries a state
`h : RingCell R` (`R × R`) and updates it with an input-dependent affine step
`h_t = ringMul σ a_t h_{t-1} + b_t`. The pair `(a_t, b_t)` is one `AffineCell σ`,
the two-slot form of the scalar step `x ↦ a·x + b`.

The step forms a monoid under composition: composing `(a₁, b₁)` then `(a₂, b₂)` gives
`(ringMul σ a₂ a₁, ringMul σ a₂ b₁ + b₂)`, and `(ringOne, 0)` is the identity. Consequently a
sequence of steps composes to one step, so the sequential recurrence equals applying the single
composed element to the initial state. That is the parallel-scan law: a prefix or tree reduction
may compose step ranges in any bracketing, and `scan_split` gives the concatenation case.

## Root

| from | here |
|---|---|
| `ringMul_assoc` | `affComp_assoc` |
| `ringMul_distrib_add` | `affApply_comp` |
| `ringMul_one`, `ringMul_zero` | `affComp_one_right` |
| `ringOne_mul` | `affComp_one_left` |
| `ringAffineFold_apply` (`Trichotomy/AnyRing.lean`) | `recur_eq_scan` |
| `ringAffineFold_append` (`Trichotomy/AnyRing.lean`) | `scan_split` |
| `ringUpper_mul`, `ringLower_mul`, `hyperbolic_decomposition` | `recur_channel_split` |

The canonical `RingAffineStep` is a structure with named fields; this module states the same
monoid on the bare pair `RingCell R × RingCell R` the recurrence consumes, so the state and the
step share one carrier.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedVariables false

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- An affine state step over `R`: a coefficient `a` and an injection `b`. The step sends `h` to
`ringMul σ a h + b`. The parameter `σ` indexes the multiplication; the carrier is the pair
`RingCell R × RingCell R` at every `σ`. -/
abbrev AffineCell (σ : R) : Type _ := RingCell R × RingCell R

/-- The identity step: coefficient the algebra unit `ringOne`, injection the origin. -/
def affId (σ : R) : AffineCell σ := (ringOne, 0)

/-- **Composing two affine steps: apply `q` first, then `p`.** The composed coefficient is
`ringMul σ p.1 q.1` and the composed injection carries `q`'s injection through `p`'s coefficient,
`ringMul σ p.1 q.2 + p.2`. -/
def affComp (σ : R) (p q : AffineCell σ) : AffineCell σ :=
  (ringMul σ p.1 q.1, ringMul σ p.1 q.2 + p.2)

/-- **Applying one affine step:** multiply the state by the coefficient, then add the injection. -/
def affApply (σ : R) (p : AffineCell σ) (h : RingCell R) : RingCell R :=
  ringMul σ p.1 h + p.2

/-- Composition of affine steps is associative. Each slot associates by `ringMul_assoc`; the
injection uses `ringMul_distrib_add` and `add_assoc`. -/
theorem affComp_assoc (σ : R) (a b c : AffineCell σ) :
    affComp σ (affComp σ a b) c = affComp σ a (affComp σ b c) := by
  obtain ⟨a1, a2⟩ := a
  obtain ⟨b1, b2⟩ := b
  obtain ⟨c1, c2⟩ := c
  apply Prod.ext
  · show ringMul σ (ringMul σ a1 b1) c1 = ringMul σ a1 (ringMul σ b1 c1)
    rw [ringMul_assoc]
  · show ringMul σ (ringMul σ a1 b1) c2 + (ringMul σ a1 b2 + a2) =
        ringMul σ a1 (ringMul σ b1 c2 + b2) + a2
    rw [ringMul_assoc, ringMul_distrib_add, add_assoc]

/-- The identity step is a left identity for composition. -/
theorem affComp_one_left (σ : R) (p : AffineCell σ) :
    affComp σ (affId σ) p = p := by
  obtain ⟨a, b⟩ := p
  apply Prod.ext
  · show ringMul σ ringOne a = a
    rw [ringOne_mul]
  · show ringMul σ ringOne b + 0 = b
    rw [ringOne_mul, add_zero]

/-- The identity step is a right identity for composition. -/
theorem affComp_one_right (σ : R) (p : AffineCell σ) :
    affComp σ p (affId σ) = p := by
  obtain ⟨a, b⟩ := p
  apply Prod.ext
  · show ringMul σ a ringOne = a
    rw [ringMul_one]
  · show ringMul σ a 0 + b = b
    rw [ringMul_zero, zero_add]

/-- **Applying a composition is applying the inner step, then the outer step.** The two sides
agree by `ringMul_assoc`, `ringMul_distrib_add` and `add_assoc`. -/
theorem affApply_comp (σ : R) (p q : AffineCell σ) (h : RingCell R) :
    affApply σ (affComp σ p q) h = affApply σ p (affApply σ q h) := by
  obtain ⟨p1, p2⟩ := p
  obtain ⟨q1, q2⟩ := q
  show ringMul σ (ringMul σ p1 q1) h + (ringMul σ p1 q2 + p2) =
      ringMul σ p1 (ringMul σ q1 h + q2) + p2
  rw [ringMul_assoc, ringMul_distrib_add, add_assoc]

/-- The identity step changes nothing. -/
@[simp] theorem affId_apply (σ : R) (h : RingCell R) :
    affApply σ (affId σ) h = h := by
  show ringMul σ ringOne h + 0 = h
  rw [ringOne_mul, add_zero]

/-- The step carrier with `affComp` and `affId` is a monoid, so `p * q = affComp σ p q`. -/
instance instMulAffineCell (σ : R) : Mul (AffineCell σ) := ⟨affComp σ⟩

instance instOneAffineCell (σ : R) : One (AffineCell σ) := ⟨affId σ⟩

instance instMonoidAffineCell (σ : R) : Monoid (AffineCell σ) where
  mul := affComp σ
  one := affId σ
  mul_assoc := affComp_assoc σ
  one_mul := affComp_one_left σ
  mul_one := affComp_one_right σ

/-- **The composed step of a list.** Start from the identity and compose each step on the outside,
so the first entry is applied first. `n` entries cost two `ringMul` and one cell addition each:
`O(n)` cell operations and `O(1)` retained accumulator cells, excluding the input list. -/
def affFold (σ : R) (steps : List (AffineCell σ)) : AffineCell σ :=
  steps.foldl (fun acc s => affComp σ s acc) (affId σ)

/-- **The sequential recurrence.** Fold the affine applies over the steps starting at `h0`, so the
state after the list is one `RingCell R`. Each step costs one `ringMul` and one cell addition; the
retained state is a single cell independent of the list length. -/
def recur (σ : R) (steps : List (AffineCell σ)) (h0 : RingCell R) : RingCell R :=
  steps.foldl (fun acc s => affApply σ s acc) h0

/-- The empty recurrence leaves the initial state. -/
@[simp] theorem recur_nil (σ : R) (h0 : RingCell R) : recur σ [] h0 = h0 := rfl

/-- One step of the recurrence applies its head to the state first. -/
theorem recur_cons (σ : R) (s : AffineCell σ) (steps : List (AffineCell σ)) (h0 : RingCell R) :
    recur σ (s :: steps) h0 = recur σ steps (affApply σ s h0) := rfl

/-- The accumulator form of `recur_eq_scan`: applying a folded step list from an arbitrary
starting step is the fold of the applies from that step applied to the state. -/
private theorem affApply_fold (σ : R) (steps : List (AffineCell σ)) (acc : AffineCell σ)
    (h0 : RingCell R) :
    affApply σ (steps.foldl (fun acc s => affComp σ s acc) acc) h0 =
      steps.foldl (fun z s => affApply σ s z) (affApply σ acc h0) := by
  induction steps generalizing acc h0 with
  | nil => rfl
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (affComp σ s acc) h0, affApply_comp]

/-- **The parallel-scan law.** The sequential recurrence equals applying the single composed
element `affFold σ steps` to the initial state, so the whole scan is one step application. -/
theorem recur_eq_scan (σ : R) (steps : List (AffineCell σ)) (h0 : RingCell R) :
    recur σ steps h0 = affApply σ (affFold σ steps) h0 := by
  rw [recur, affFold, affApply_fold, affId_apply]

/-- The accumulator form of `scan_split`: folding from an arbitrary starting step is the fold of
the list composed on the outside of that step. -/
private theorem affFold_accum (σ : R) (steps : List (AffineCell σ)) (acc : AffineCell σ) :
    steps.foldl (fun acc s => affComp σ s acc) acc =
      affComp σ (steps.foldl (fun acc s => affComp σ s acc) (affId σ)) acc := by
  induction steps generalizing acc with
  | nil =>
    simp only [List.foldl_nil]
    exact (affComp_one_left σ acc).symm
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (affComp σ s acc), ih (affComp σ s (affId σ))]
    rw [affComp_one_right, affComp_assoc]

/-- **Splitting a concatenated scan.** The composed element of `xs ++ ys` is the composition of
the two halves' composed elements, with `ys` on the outside because its steps run later. This is
the associativity a parallel prefix or tree reduction needs: ranges compose in any bracketing. -/
theorem scan_split (σ : R) (xs ys : List (AffineCell σ)) :
    affFold σ (xs ++ ys) = affComp σ (affFold σ ys) (affFold σ xs) := by
  simp only [affFold]
  rw [List.foldl_append, affFold_accum]

/-- The recurrence state carrier: one `RingCell R`, for every step count. -/
abbrev RecurState (σ : R) : Type _ := RingCell R

/-- The recurrence output lies in the fixed carrier `RecurState σ = RingCell R`; the state type
does not depend on the number of steps. -/
theorem recur_state_carrier (σ : R) (steps : List (AffineCell σ)) (h0 : RecurState σ) :
    (recur σ steps h0 : RecurState σ) = recur σ steps h0 := rfl

section Channel

variable {K : Type*} [Field K]

/-- **The channel split at `σ = 1`.** On the hyperbolic face with `2` invertible, `channelPair 1`
carries the `RingCell` recurrence into two independent scalar affine recurrences,
`u_t = ringUpper 1 a_t * u_{t-1} + ringUpper 1 b_t` and
`v_t = ringLower 1 a_t * v_{t-1} + ringLower 1 b_t`. The intertwining facts come from
`hyperbolic_decomposition`, which also makes `channelPair 1` a bijection on the state. -/
theorem recur_channel_split (h2 : (2 : K) ≠ 0) (steps : List (AffineCell (1 : K)))
    (h0 : RingCell K) :
    channelPair (1 : K) (recur (1 : K) steps h0) =
      (steps.foldl (fun u s => ringUpper (1 : K) s.1 * u + ringUpper (1 : K) s.2)
          (ringUpper (1 : K) h0),
       steps.foldl (fun v s => ringLower (1 : K) s.1 * v + ringLower (1 : K) s.2)
          (ringLower (1 : K) h0)) := by
  have hdec := hyperbolic_decomposition (σ := (1 : K)) (r := (1 : K)) (s := (1 : K))
    (h := (2 : K)⁻¹) (by ring) (by ring) (mul_inv_cancel₀ h2)
  obtain ⟨_, hmul, hadd, _, _⟩ := hdec
  have hmulU : ∀ z w : RingCell K,
      ringUpper (1 : K) (ringMul (1 : K) z w) =
        ringUpper (1 : K) z * ringUpper (1 : K) w := fun z w => by
    simpa [channelPair] using congrArg Prod.fst (hmul z w)
  have hmulL : ∀ z w : RingCell K,
      ringLower (1 : K) (ringMul (1 : K) z w) =
        ringLower (1 : K) z * ringLower (1 : K) w := fun z w => by
    simpa [channelPair] using congrArg Prod.snd (hmul z w)
  have haddU : ∀ z w : RingCell K,
      ringUpper (1 : K) (z + w) = ringUpper (1 : K) z + ringUpper (1 : K) w := fun z w => by
    simpa [channelPair] using congrArg Prod.fst (hadd z w)
  have haddL : ∀ z w : RingCell K,
      ringLower (1 : K) (z + w) = ringLower (1 : K) z + ringLower (1 : K) w := fun z w => by
    simpa [channelPair] using congrArg Prod.snd (hadd z w)
  have keyU : ∀ (steps : List (AffineCell (1 : K))) (acc : RingCell K),
      ringUpper (1 : K) (steps.foldl (fun acc s => affApply (1 : K) s acc) acc) =
        steps.foldl (fun u s => ringUpper (1 : K) s.1 * u + ringUpper (1 : K) s.2)
          (ringUpper (1 : K) acc) := by
    intro steps
    induction steps with
    | nil => intro acc; rfl
    | cons s rest ih =>
      intro acc
      simp only [List.foldl_cons]
      rw [ih (affApply (1 : K) s acc)]
      congr 1
      rw [affApply, haddU, hmulU]
  have keyL : ∀ (steps : List (AffineCell (1 : K))) (acc : RingCell K),
      ringLower (1 : K) (steps.foldl (fun acc s => affApply (1 : K) s acc) acc) =
        steps.foldl (fun v s => ringLower (1 : K) s.1 * v + ringLower (1 : K) s.2)
          (ringLower (1 : K) acc) := by
    intro steps
    induction steps with
    | nil => intro acc; rfl
    | cons s rest ih =>
      intro acc
      simp only [List.foldl_cons]
      rw [ih (affApply (1 : K) s acc)]
      congr 1
      rw [affApply, haddL, hmulL]
  apply Prod.ext
  · simpa only [channelPair, recur] using keyU steps h0
  · simpa only [channelPair, recur] using keyL steps h0

end Channel

end Nous.Model
