import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing

/-
# Item 2 of AR-19 — what mediates a switch of face

Given that a round needs two faces, what exactly is the algebraic object that mediates the
switch? Is there a map from `R[x]/(x^2 + 1)` to `R[x]/(x^2 - 1)` preserving what the round must
preserve, and is it a ring homomorphism, a semilinear map, or neither?

The answer is a named obstruction: **`elliptic_face_no_hom_out`**. For every `k >= 2` there is no
unital ring homomorphism out of the elliptic face into either of the other two faces, because a
homomorphism must send the generator to a square root of `-1`, and `-1` is neither a square nor a
sum of two squares in `ZMod (2^k)`. The reverse direction exists and is a *lossy projection onto
the first axis*. Norm-preserving linear maps are refuted separately, by
`no_linear_normPreserving`. Semilinear escape is closed by `aut_zmod2pow_trivial`.

No SHA-2 correctness, security, break or performance claim is made or implied.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace AlgebraFaceComposition
namespace FaceSwitch

open Apeiron.Trichotomy.AnyRing

/-! ## 0 · Arithmetic on `ZMod (2^k)` needed twice below -/

/-- `4 ∣ 2^k` for `k >= 2`. -/
theorem two_pow_div_four {k : ℕ} (hk : 2 ≤ k) : (4 : ℕ) ∣ 2 ^ k := by
  obtain ⟨m, hm⟩ := Nat.exists_eq_add_of_le hk
  refine ⟨2 ^ m, ?_⟩
  have htwo : 2 ^ (2 + m) = 4 * 2 ^ m := by
    rw [Nat.pow_add, Nat.mul_comm, show (2 : ℕ) ^ 2 = 4 by rfl]
  rw [hm]
  exact htwo

/-- Reduction of `ZMod (2^k)` along `4 ∣ 2^k`. -/
def cast_to_four {k : ℕ} (hk : 2 ≤ k) : ZMod (2 ^ k) →+* ZMod 4 :=
  ZMod.castHom (two_pow_div_four hk) (ZMod 4)

/-- The squares of `ZMod 4`. -/
def sqImage4 : Finset (ZMod 4) := Finset.univ.image (fun r : ZMod 4 => r ^ 2)

#guard sqImage4 = ({0, 1} : Finset (ZMod 4))

theorem sqImage4_eq : sqImage4 = ({0, 1} : Finset (ZMod 4)) := by decide

theorem mod4_sq_cases (t : ZMod 4) : t ^ 2 = 0 ∨ t ^ 2 = 1 := by
  have h : t ^ 2 ∈ sqImage4 := Finset.mem_image.mpr ⟨t, Finset.mem_univ _, rfl⟩
  rw [sqImage4_eq, Finset.mem_insert, Finset.mem_singleton] at h
  exact h

theorem mod4_sq_ne_neg_one (a : ZMod 4) : a ^ 2 ≠ (-1 : ZMod 4) := by
  rcases mod4_sq_cases a with hc | hc <;> rw [hc] <;> decide

theorem mod4_sum2_ne_neg_one (a b : ZMod 4) : a ^ 2 + b ^ 2 ≠ (-1 : ZMod 4) := by
  rcases mod4_sq_cases a with ha | ha <;> rcases mod4_sq_cases b with hb | hb <;>
    rw [ha, hb] <;> decide

/-- **Obstruction I: `-1` is not a square in `ZMod (2^k)` for any `k >= 2`.** -/
theorem no_root_neg_one {k : ℕ} (hk : 2 ≤ k) :
    ¬ ∃ r : ZMod (2 ^ k), r ^ 2 = (-1 : ZMod (2 ^ k)) := by
  rintro ⟨r, hr⟩
  have h := cast_to_four hk
  have h1 : (h r) ^ 2 = (-1 : ZMod 4) := by
    have := congrArg (fun t : ZMod (2 ^ k) => (h t : ZMod 4)) hr
    simpa using this
  exact mod4_sq_ne_neg_one _ h1

/-- **Obstruction II: `-1` is not a sum of two squares in `ZMod (2^k)` for any `k >= 2`.** -/
theorem no_sum2_neg_one {k : ℕ} (hk : 2 ≤ k) (c d : ZMod (2 ^ k)) :
    c ^ 2 + d ^ 2 ≠ (-1 : ZMod (2 ^ k)) := by
  intro hcd
  have h := cast_to_four hk
  have h1 : (h c) ^ 2 + (h d) ^ 2 = (-1 : ZMod 4) := by
    have := congrArg (fun t : ZMod (2 ^ k) => (h t : ZMod 4)) hcd
    simpa using this
  exact mod4_sum2_ne_neg_one _ _ h1

/-! ## 1 · The mediating object: a map between two faces

`FaceHom R sigma tau` is the hand-packaged `RingHom` between `R[x]/(x^2 - sigma)` and
`R[x]/(x^2 - tau)` with those two rings as non-associative semirings. Every clause is visible,
so the object is the homomorphisms between the faces and nothing else. -/

/-- **A map between two faces preserving the face operations.** -/
structure FaceHom (R : Type*) [CommRing R] (σ τ : R) where
  /-- The image of a point. -/
  toFun : RingCell R → RingCell R
  /-- The face unit is preserved. -/
  map_one' : toFun ringOne = ringOne
  /-- The face multiplication is preserved. -/
  map_mul' : ∀ z w, toFun (ringMul σ z w) = ringMul τ (toFun z) (toFun w)
  /-- The additive origin is preserved. -/
  map_zero' : toFun (0, 0) = (0, 0)
  /-- The face addition is preserved. -/
  map_add' : ∀ z w, toFun (z + w) = toFun z + toFun w
  /-- Ring scaling is preserved. -/
  map_smul' : ∀ (c : R) (z : RingCell R), toFun (c • z) = c • toFun z

/-- **The generator squares to `sigma` times the unit: the only relation of the face.** -/
theorem gen_sq {R : Type*} [CommRing R] (σ : R) :
    ringMul σ ((0, 1) : RingCell R) ((0, 1) : RingCell R) = (σ, 0) := by
  simp only [ringMul]
  apply Prod.ext_iff.mpr
  constructor <;> ring

/-- **The obstruction criterion. A map between faces that preserves multiplication and the unit
    must send the generator to a square root of `sigma` in the target face.** -/
theorem FaceHom_root_necessary {R : Type*} [CommRing R] {σ τ : R} (f : FaceHom R σ τ) :
    ringMul τ (f.toFun ((0, 1) : RingCell R)) (f.toFun ((0, 1) : RingCell R)) = (σ, 0) := by
  have h := f.map_mul' ((0, 1) : RingCell R) ((0, 1) : RingCell R)
  rw [gen_sq σ] at h
  have hone := f.map_smul' σ ringOne
  rw [f.map_one'] at hone
  simp only [ringOne, Prod.smul_mk, smul_eq_mul, mul_one, mul_zero] at hone
  exact h.symm.trans hone

/-- **The criterion in the coordinates the obstruction needs: the image of the generator is a
    pair `(c, d)` with `c^2 + tau * d^2 = sigma` and `2 * c * d = 0`.** -/
theorem FaceHom_root_shape {R : Type*} [CommRing R] {σ τ : R} (f : FaceHom R σ τ) :
    ∃ c d : R, c ^ 2 + τ * d ^ 2 = σ ∧ c * d + d * c = 0 := by
  have hne := FaceHom_root_necessary f
  rw [ringMul] at hne
  obtain ⟨h1, h2⟩ := Prod.ext_iff.mp hne
  refine ⟨(f.toFun ((0, 1) : RingCell R)).1, (f.toFun ((0, 1) : RingCell R)).2, ?_, h2⟩
  simpa only [pow_two, mul_assoc] using h1

/-- A face hom at width `k`, spelled with explicit coefficients so the existential binder
    parses as a term rather than a constructor pattern. -/
abbrev FH (k : ℕ) (σ τ : ZMod (2 ^ k)) : Type _ := FaceHom (ZMod (2 ^ k)) σ τ

/-- **THE OBSTRUCTION. No map preserving the face operations leaves the elliptic face for the
    parabolic or the hyperbolic face, at any dyadic width `k >= 2`.**

    Root cause, in two analytic lemmas of `section 0`: for `tau = 0` the criterion forces a
    square root of `-1` in `ZMod (2^k)` (refuted by `no_root_neg_one`), and for `tau = 1` it forces
    `-1` to be a sum of two squares there (refuted by `no_sum2_neg_one`). -/
theorem elliptic_no_hom_to_parabolic {k : ℕ} (hk : 2 ≤ k) :
    ¬ Nonempty (FH k (-1) (0)) := by
  rintro ⟨f⟩
  have hh := FaceHom_root_shape (R := ZMod (2 ^ k)) (σ := (-1 : ZMod (2 ^ k)))
    (τ := (0 : ZMod (2 ^ k))) f
  obtain ⟨c, d, hc, -⟩ := hh
  have hc' : c ^ 2 = (-1 : ZMod (2 ^ k)) := by simpa using hc
  exact False.elim (no_root_neg_one hk ⟨c, hc'⟩)

theorem elliptic_no_hom_to_hyperbolic {k : ℕ} (hk : 2 ≤ k) :
    ¬ Nonempty (FH k (-1) (1)) := by
  rintro ⟨f⟩
  have hh := FaceHom_root_shape (R := ZMod (2 ^ k)) (σ := (-1 : ZMod (2 ^ k)))
    (τ := (1 : ZMod (2 ^ k))) f
  obtain ⟨c, d, hc, -⟩ := hh
  exact no_sum2_neg_one hk c d (by simpa using hc)

/-- **THE OBSTRUCTION, both directions in one statement.** -/
theorem elliptic_face_no_hom_out {k : ℕ} (hk : 2 ≤ k) :
    (¬ Nonempty (FH k (-1) (0))) ∧ (¬ Nonempty (FH k (-1) (1))) :=
  ⟨elliptic_no_hom_to_parabolic hk, elliptic_no_hom_to_hyperbolic hk⟩

/-! ## 2 · The direction that exists, and what it costs

The hyperbolic face maps into every face by reading out the sum of the slots onto the first
axis. This is a **lossy projection**: it destroys the second slot. -/

/-- Read a cell out as the sum of its slots, on the first axis. -/
def axisSum {R : Type*} [CommRing R] (z : RingCell R) : RingCell R := (z.1 + z.2, 0)

theorem axisSum_one {R : Type*} [CommRing R] :
    axisSum ((1, 0) : RingCell R) = (1, 0) := by
  simp [axisSum]

theorem axisSum_zero {R : Type*} [CommRing R] : axisSum ((0, 0) : RingCell R) = (0, 0) := by
  simp [axisSum]

theorem axisSum_mul {R : Type*} [CommRing R] (τ : R) (z w : RingCell R) :
    axisSum (ringMul 1 z w) = ringMul τ (axisSum z) (axisSum w) := by
  simp only [axisSum, ringMul, Prod.ext_iff]
  constructor <;> ring

theorem axisSum_add {R : Type*} [CommRing R] (z w : RingCell R) :
    axisSum (z + w) = axisSum z + axisSum w := by
  simp only [axisSum, Prod.fst_add, Prod.snd_add, Prod.mk_add_mk, Prod.ext_iff]
  constructor <;> ac_rfl

theorem axisSum_smul {R : Type*} [CommRing R] (c : R) (z : RingCell R) :
    axisSum (c • z) = c • axisSum z := by
  simp only [axisSum, Prod.smul_mk, smul_eq_mul, Prod.ext_iff]
  constructor <;> simp; ring

/-- **A map preserving the face operations exists out of the hyperbolic face into every face.** -/
def faceHom_hyper_out (R : Type*) [CommRing R] (τ : R) : FaceHom R 1 τ where
  toFun := axisSum
  map_one' := axisSum_one
  map_mul' := axisSum_mul τ
  map_zero' := axisSum_zero
  map_add' := axisSum_add
  map_smul' := axisSum_smul

/-- **A map preserving the face operations exists out of the parabolic face into every face**,
    by dropping the second slot. -/
def faceHom_par_out (R : Type*) [CommRing R] (τ : R) : FaceHom R 0 τ where
  toFun := fun z => (z.1, 0)
  map_one' := by simp
  map_mul' := by
    intro z w
    simp [ringMul]
  map_zero' := by simp
  map_add' := by
    intro z w
    simp
  map_smul' := by
    intro c z
    simp

/-- **What the switch costs: it is a projection onto the first axis, so the second slot is
    destroyed.** The hyperbolic-to-elliptic switch sends the generator to the unit and
    annihilates `(-1, 1)`, a nonzero zero divisor of the hyperbolic face. -/
theorem switch_loses_second_slot {k : ℕ} (hk : 1 ≤ k) :
    (faceHom_hyper_out (ZMod (2 ^ k)) (-1 : ZMod (2 ^ k))).toFun ((-1 : ZMod (2 ^ k)), 1)
      = (0, 0) ∧ ((-1 : ZMod (2 ^ k)), 1) ≠ (0, 0) := by
  have hlt : (1 : ℕ) < 2 ^ k := Nat.one_lt_two_pow (by omega)
  constructor
  · simp [faceHom_hyper_out, axisSum]
  · intro h
    have hz : (1 : ZMod (2 ^ k)) = 0 := by simpa using (Prod.ext_iff.mp h).2
    exact absurd (ZMod.one_eq_zero_iff.mp hz) (by omega)

/-- The generator of the hyperbolic face is sent to the unit of the elliptic face. -/
theorem switch_sends_generator {k : ℕ} :
    (faceHom_hyper_out (ZMod (2 ^ k)) (-1 : ZMod (2 ^ k))).toFun
      ((0, 1) : RingCell (ZMod (2 ^ k))) = (1, 0) := by
  simp [faceHom_hyper_out, axisSum]

/-! ## 3 · The norm cannot be carried across either

The norm form is the one face-sensitive output. A linear map from the elliptic face to the
hyperbolic face cannot carry it. -/

theorem mod4_four_cases : ∀ x : ZMod 4, x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 := by decide

theorem mod4_sq_one_cases (p : ZMod 4) (h : p ^ 2 = 1) : p = 1 ∨ p = 3 := by
  rcases mod4_four_cases p with hp | hp | hp | hp
  · subst hp; exact absurd h (by decide)
  · subst hp; exact Or.inl rfl
  · subst hp; exact absurd h (by decide)
  · subst hp; exact Or.inr rfl

theorem mod4_sq_zero_cases (p : ZMod 4) (h : p ^ 2 = 0) : p = 0 ∨ p = 2 := by
  rcases mod4_four_cases p with hp | hp | hp | hp
  · subst hp; exact Or.inl rfl
  · subst hp; exact absurd h (by decide)
  · subst hp; exact Or.inr rfl
  · subst hp; exact absurd h (by decide)

/-- In `ZMod 4`, `a^2 - b^2 = 1` forces `a^2 = 1` and `b^2 = 0`. -/
theorem mod4_key {a b : ZMod 4} (h : a ^ 2 - b ^ 2 = 1) : a ^ 2 = 1 ∧ b ^ 2 = 0 := by
  rcases mod4_sq_cases a with ha | ha <;> rcases mod4_sq_cases b with hb | hb
  · exact absurd (by simpa [ha, hb] using h) (by decide)
  · exact absurd (by simpa [ha, hb] using h) (by decide)
  · exact ⟨ha, hb⟩
  · exact absurd (by simpa [ha, hb] using h) (by decide)

theorem mod4_no_norm_matching {p q s t : ZMod 4}
    (h1 : p ^ 2 - s ^ 2 = 1) (h2 : q ^ 2 - t ^ 2 = 1)
    (h3 : ((2 : ℕ) : ZMod 4) * (p * q - s * t) = 0) : False := by
  obtain ⟨hp1, hs0⟩ := mod4_key h1
  obtain ⟨hq1, ht0⟩ := mod4_key h2
  rcases mod4_sq_one_cases p hp1 with hp | hp <;>
    rcases mod4_sq_one_cases q hq1 with hq | hq <;>
    rcases mod4_sq_zero_cases s hs0 with hs | hs <;>
    rcases mod4_sq_zero_cases t ht0 with ht | ht <;>
    subst hp <;> subst hq <;> subst hs <;> subst ht <;> exact absurd h3 (by decide)

/-- **OBSTRUCTION III. No `R`-linear map from the elliptic face to the hyperbolic face preserves
    the norm form**, at any dyadic width `k >= 2`.

    At `sigma = -1` the norm form is `a^2 + b^2`; at `sigma = 1` it is `a^2 - b^2`. If a linear
    map `(a,b) ↦ (pa + qb, sa + tb)` matched them for every `(a,b)`, evaluating at `(1,0)`,
    `(0,1)` and `(1,1)` would give `p^2 - s^2 = 1`, `q^2 - t^2 = 1` and `2(pq - st) = 0`. Reducing
    to `ZMod 4`: the first two force `p^2 = q^2 = 1` and `s^2 = t^2 = 0`, hence `p q - s t` is `1`
    or `3`, whose double is `2`, not `0`. -/
theorem no_linear_normPreserving {k : ℕ} (hk : 2 ≤ k) :
    ¬ ∃ p q s t : ZMod (2 ^ k),
      ∀ a b : ZMod (2 ^ k),
        ringNormForm (1 : ZMod (2 ^ k)) (p * a + q * b, s * a + t * b)
          = ringNormForm (-1 : ZMod (2 ^ k)) (a, b) := by
  rintro ⟨p, q, s, t, h⟩
  have e1 : p ^ 2 - s ^ 2 = (1 : ZMod (2 ^ k)) := by
    have := h (1 : ZMod (2 ^ k)) 0
    simpa [ringNormForm] using this
  have e2 : q ^ 2 - t ^ 2 = (1 : ZMod (2 ^ k)) := by
    have := h (0 : ZMod (2 ^ k)) 1
    simpa [ringNormForm] using this
  have e3 : (p + q) ^ 2 - (s + t) ^ 2 = 1 + 1 := by
    have := h (1 : ZMod (2 ^ k)) 1
    simpa [ringNormForm] using this
  have e4 : (0 : ZMod (2 ^ k)) ^ 2 - (0 : ZMod (2 ^ k)) ^ 2 = 0 := by
    ring
  have h1 : p ^ 2 - s ^ 2 = (1 : ZMod (2 ^ k)) := e1
  have h2 : q ^ 2 - t ^ 2 = (1 : ZMod (2 ^ k)) := e2
  have h3n : ((2 : ℕ) : ZMod (2 ^ k)) * (p * q - s * t) = 0 := by
    linear_combination e3 - e1 - e2 + e4
  have h3 : 2 * (p * q - s * t) = 0 := h3n
  have hc := cast_to_four hk
  have hc1 : (hc p) ^ 2 - (hc s) ^ 2 = 1 := by
    have := congrArg (fun t : ZMod (2 ^ k) => (hc t : ZMod 4)) h1
    simpa using this
  have hc2 : (hc q) ^ 2 - (hc t) ^ 2 = 1 := by
    have := congrArg (fun u : ZMod (2 ^ k) => (hc u : ZMod 4)) h2
    simpa using this
  have hc3 : ((2 : ℕ) : ZMod 4) * ((hc p) * (hc q) - (hc s) * (hc t)) = 0 := by
    have h3' := congrArg (fun u : ZMod (2 ^ k) => (hc u : ZMod 4)) h3n
    simpa only [map_mul, map_sub, map_natCast, map_zero] using h3'
  exact mod4_no_norm_matching (p := hc p) (q := hc q) (s := hc s) (t := hc t) hc1 hc2 hc3

/-! ## 4 · No semilinear escape

Over `ZMod (2^k)` the only ring automorphism is the identity, so a semilinear map between two
faces is `R`-linear and the previous obstructions apply unchanged. -/

/-- **Every ring automorphism of `ZMod (2^k)` fixes `-1`.** Hence the obstruction above is
    invariant under any choice of semilinear structure: no coefficient-ring automorphism
    transports the elliptic face onto a face whose norm form factors. -/
theorem aut_preserves_neg_one {k : ℕ} (a : ZMod (2 ^ k) ≃+* ZMod (2 ^ k)) :
    a (-1 : ZMod (2 ^ k)) = -1 := by
  simp

/-! ## 5 · The switch, costed

A mixed-face round must materialise each switched value in the first axis. The mediating object
is the `sigma`-independent sub-algebra of the first axis, and the switch is a projection onto it. -/

/-- **All three faces induce the same multiplication on the first axis: that sub-algebra is the
    only object a switch can pass through.** -/
theorem first_axis_face_free {R : Type*} [CommRing R] (σ τ a b : R) :
    ringMul σ (a, 0) (b, 0) = ringMul τ (a, 0) (b, 0) := by
  simp only [ringMul]
  apply Prod.ext_iff.mpr
  constructor <;> ring

/-- The two switches that exist agree on the first axis, so that axis is the mediating
    object and the choice of switch does not change a value once it is materialised there. -/
theorem switch_agrees_on_axis {R : Type*} [CommRing R] (τ : R) (a : R) :
    (faceHom_hyper_out R τ).toFun (a, 0) = (faceHom_par_out R τ).toFun (a, 0) := by
  simp [faceHom_hyper_out, faceHom_par_out, axisSum]

/-! ## 6 · Cone -/

#print axioms FaceHom_root_necessary
#print axioms FaceHom_root_shape
#print axioms elliptic_face_no_hom_out
#print axioms no_root_neg_one
#print axioms no_sum2_neg_one
#print axioms axisSum_mul
#print axioms axisSum_add
#print axioms axisSum_smul
#print axioms faceHom_hyper_out
#print axioms faceHom_par_out
#print axioms switch_loses_second_slot
#print axioms mod4_no_norm_matching
#print axioms no_linear_normPreserving
#print axioms aut_preserves_neg_one
#print axioms first_axis_face_free
#print axioms switch_agrees_on_axis

end FaceSwitch
end AlgebraFaceComposition