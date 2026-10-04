import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing

/-
# Item 3 of AR-19 — the single-face verdict for one SHA-256 round

Does one SHA-256 round fit inside a single two-slot face, or must it be composed across faces?

The answer proved here is **single face, `sigma = 1`**, and the premise that rotation "wants the
norm-one torus" is **refuted by a named obstruction**: `rotation_not_ringMul`. Every expression
built from one `ringMul` and one channel read is *additive in the word*, at every face, and the
rotation cell is not additive in the word (`rotWrap_not_additive`). The dial is therefore not what
rotation needs; a second slot is.

What a face choice still costs is the readout, and there the additive layer forces `sigma = 1`:
the parabolic root is blind and the elliptic face has no read at any dyadic width. The
chi/Ch/Maj product term is face-independent.

No SHA-2 correctness, security, break or performance claim is made or implied.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace AlgebraFaceComposition
namespace FaceRound

open Apeiron.Trichotomy.AnyRing

/-! ## 0 · The face-independent coordinate facts, re-proved here -/

/-- Multiplication on the first axis is the ring product, at every face. -/
theorem ringMul_first_axis {R : Type*} [CommRing R] (σ a b : R) :
    ringMul σ (a, 0) (b, 0) = (a * b, 0) := by
  simp only [ringMul, Prod.ext_iff]
  constructor <;> ring

/-- A first-axis element times a second-axis element lands in the second axis, at every face. -/
theorem ringMul_second_of_first {R : Type*} [CommRing R] (σ a b : R) :
    ringMul σ (a, 0) (0, b) = (0, a * b) := by
  simp only [ringMul, Prod.ext_iff]
  constructor <;> ring

/-- **The quadratic term of chi, of Ch and of Maj: a product of two first-axis elements, at every
    face.** This is the whole of what the boolean layer's multiplication needs. -/
theorem chi_term_face_free {R : Type*} [CommRing R] (σ x y z : R) :
    (x, 0) + (z, 0) + ringMul σ (y, 0) (z, 0) = (x + z + y * z, 0) := by
  rw [ringMul_first_axis]
  simp

/-! ## 1 · The obstruction for the rotation leg

`ringUpper r (ringMul σ (w, 0) (a, b))` is the whole family of one-`ringMul` readings, and it is
additive in the word at every face. -/

/-- **THE ROTATION OBSTRUCTION, part 1: every one-`ringMul` reading is additive in the word, at
    every face.** Proved from the canonical `ringMul_distrib_add` and `ringUpper_add`. -/
theorem ringMul_family_additive {R : Type*} [CommRing R] (σ r a b x y : R) :
    ringUpper r (ringMul σ ((x, 0) + (y, 0)) (a, b))
      = ringUpper r (ringMul σ (x, 0) (a, b)) + ringUpper r (ringMul σ (y, 0) (a, b)) := by
  rw [ringMul_comm σ ((x, 0) + (y, 0)) (a, b), ringMul_distrib_add σ (a, b) (x, 0) (y, 0),
    ringMul_comm σ (a, b) (x, 0), ringMul_comm σ (a, b) (y, 0), ringUpper_add]

/-- The truncated shift of a rotation by `n`. -/
def rotShift (k n v : ℕ) : ZMod (2 ^ k) := Nat.cast ((v * 2 ^ n) % 2 ^ k)

/-- The bits wrapped around by a rotation by `n`. -/
def rotWrap (k n v : ℕ) : ZMod (2 ^ k) := Nat.cast (v / 2 ^ (k - n))

/-- The two-slot cell a rotation by `n` of a word occupies. -/
def rotCellOf (k n : ℕ) (v : ZMod (2 ^ k)) : RingCell (ZMod (2 ^ k)) :=
  (rotShift k n v.val, rotWrap k n v.val)

/-- `4 ≤ 2^k` for `k >= 2`. -/
theorem two_pow_ge_four {k : ℕ} (hk : 2 ≤ k) : (4 : ℕ) ≤ 2 ^ k := by
  have hh : 2 ^ 2 ≤ 2 ^ k :=
    (Nat.pow_le_pow_iff_right (a := 2) (n := 2) (m := k) (by omega)).2 hk
  simpa using hh

/-- `1 + 1 ≠ 3` in `ZMod (2^k)` for `k >= 2`. -/
theorem one_add_one_ne_three_zmod {k : ℕ} (hk : 2 ≤ k) :
    ((1 : ℕ) : ZMod (2 ^ k)) + ((1 : ℕ) : ZMod (2 ^ k)) ≠ ((3 : ℕ) : ZMod (2 ^ k)) := by
  have h4 : (4 : ℕ) ≤ 2 ^ k := two_pow_ge_four hk
  have hlt1 : (1 : ℕ) < 2 ^ k := by omega
  have hlt3 : (3 : ℕ) < 2 ^ k := by omega
  intro hc
  have hv := congrArg ZMod.val hc
  rw [ZMod.val_add] at hv
  rw [ZMod.val_natCast_of_lt (n := 2 ^ k) (a := 1) hlt1] at hv
  rw [ZMod.val_natCast_of_lt (n := 2 ^ k) (a := 3) hlt3] at hv
  have hA : ((1 : ℕ) + 1) % 2 ^ k = 2 := Nat.mod_eq_of_lt (by omega)
  simp only [hA] at hv
  omega

/-- **THE ROTATION OBSTRUCTION, part 2: the rotation cell is not additive in the word, at every
    `k >= 2`.** Rotation by `k-1`, the word `3`: its wrapped bits are `1`, and `1 + 1 = 2` while
    `⌊6/2⌋ = 3`. -/
theorem rotWrap_not_additive {k : ℕ} (hk : 2 ≤ k) :
    rotWrap k (k - 1) 3 + rotWrap k (k - 1) 3 ≠ rotWrap k (k - 1) 6 := by
  intro h
  have hk' : k - (k - 1) = 1 := by omega
  have h3 : rotWrap k (k - 1) 3 = 1 := by
    unfold rotWrap
    rw [hk']
    rw [(show (3 : ℕ) / 2 ^ 1 = 1 from by decide), Nat.cast_one]
  have h6 : rotWrap k (k - 1) 6 = 3 := by
    unfold rotWrap
    rw [hk']
    rw [(show (6 : ℕ) / 2 ^ 1 = 3 from by decide), Nat.cast_ofNat]
  rw [h3, h6] at h
  exact one_add_one_ne_three_zmod hk (by simpa using h)

/-- **THE ROTATION OBSTRUCTION. No face, elliptic included, computes the wrapped-around high bits
    of a rotation — the exact term the one-product family cannot produce — by one `ringMul` and
    one channel read.**

    The family `v ↦ ringUpper r (ringMul σ (v, 0) (a, b))` is additive in the word at every face
    (`ringMul_family_additive`); the wrapped bits are not (`rotWrap_not_additive`). The `sigma`
    never entered: this holds at all three faces.

    The word is quantified by its representative `v < 2^k`, so no reading of `ZMod.val` is
    needed. The width hypothesis is `3 ≤ k`, which covers the SHA-2 word width `32`; the
    exclusion at `k = 2` is recorded in the receipt. -/
theorem rotation_not_ringMul {k : ℕ} (hk : 3 ≤ k) :
    ¬ ∃ (σ r a b : ZMod (2 ^ k)), ∀ (v : ℕ) (_hv : v < 2 ^ k),
      rotWrap k (k - 1) v
        = ringUpper r (ringMul σ (((v : ℕ) : ZMod (2 ^ k)), 0) (a, b)) := by
  rintro ⟨σ, r, a, b, h⟩
  have hadd (x y : ZMod (2 ^ k)) :
      ringUpper r (ringMul σ ((x, 0) + (y, 0)) (a, b))
        = ringUpper r (ringMul σ (x, 0) (a, b)) + ringUpper r (ringMul σ (y, 0) (a, b)) :=
    ringMul_family_additive σ r a b x y
  have h36 : ((3 : ℕ) : ZMod (2 ^ k)) + ((3 : ℕ) : ZMod (2 ^ k)) = ((6 : ℕ) : ZMod (2 ^ k)) := by
    norm_num
  have hlt3 : (3 : ℕ) < 2 ^ k := by
    have hh := (Nat.pow_le_pow_iff_right (a := 2) (n := 2) (m := k) (by omega)).2 (by omega)
    have : (4 : ℕ) ≤ 2 ^ k := by simpa using hh
    omega
  have hlt6 : (6 : ℕ) < 2 ^ k := by
    have hh := (Nat.pow_le_pow_iff_right (a := 2) (n := 3) (m := k) (by omega)).2 (by omega)
    have : (8 : ℕ) ≤ 2 ^ k := by simpa using hh
    omega
  have hf3 : ringUpper r (ringMul σ (((3 : ℕ) : ZMod (2 ^ k)), 0) (a, b)) = rotWrap k (k - 1) 3 :=
    (h 3 hlt3).symm
  have hf6 : ringUpper r (ringMul σ (((6 : ℕ) : ZMod (2 ^ k)), 0) (a, b)) = rotWrap k (k - 1) 6 :=
    (h 6 hlt6).symm
  have h63 := hadd ((3 : ℕ) : ZMod (2 ^ k)) ((3 : ℕ) : ZMod (2 ^ k))
  rw [Prod.mk_add_mk, zero_add, h36, hf6, hf3] at h63
  exact rotWrap_not_additive (by omega) h63.symm

/-! ## 2 · What the additive layer costs in face choice

The additive layer's readout exists at the parabolic root and is blind there, and does not exist
at all at the elliptic face. So among the three named faces only `sigma = 1` serves it. -/

/-- `1 ≠ 0` in `ZMod (2^k)` for `k >= 1`. -/
theorem one_ne_zero_zmod {k : ℕ} (hk : 1 ≤ k) : (1 : ZMod (2 ^ k)) ≠ 0 := by
  intro hc
  have h1 : 2 ^ k = 1 := ZMod.one_eq_zero_iff.mp hc
  have hlt : (1 : ℕ) < 2 ^ k := by
    have := Nat.pow_lt_pow_right (by norm_num : (1 : ℕ) < 2) hk
    simpa using this
  omega

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

/-- **The elliptic face has no read: `-1` is not a square in `ZMod (2^k)` for any `k >= 2`.** -/
theorem no_root_neg_one {k : ℕ} (hk : 2 ≤ k) :
    ¬ ∃ r : ZMod (2 ^ k), r ^ 2 = (-1 : ZMod (2 ^ k)) := by
  rintro ⟨r, hr⟩
  have h := cast_to_four hk
  have h1 : (h r) ^ 2 = (-1 : ZMod 4) := by
    have := congrArg (fun t : ZMod (2 ^ k) => (h t : ZMod 4)) hr
    simpa using this
  exact mod4_sq_ne_neg_one _ h1

/-- Parabolic read: the first slot alone. -/
theorem read_parabolic {R : Type*} [CommRing R] (z : RingCell R) :
    ringUpper (0 : R) z = z.1 := by
  simp [ringUpper]

/-- **The additive layer's readout: available at the hyperbolic root, blind at the parabolic one,
    absent at the elliptic face.** -/
theorem additive_layer_readout {k : ℕ} (hk : 2 ≤ k) :
    (∃ r : ZMod (2 ^ k), r ^ 2 = 1 ∧ ∀ z : RingCell (ZMod (2 ^ k)), ringUpper r z = z.1 + z.2) ∧
      (∃ r : ZMod (2 ^ k), r ^ 2 = 0 ∧
        ∀ z : RingCell (ZMod (2 ^ k)), ringUpper r z = z.1) ∧
      (¬ ∃ r : ZMod (2 ^ k), r ^ 2 = -1) := by
  exact ⟨⟨1, by ring, fun z => by simp [ringUpper]⟩,
    ⟨0, by ring, fun z => read_parabolic z⟩, no_root_neg_one hk⟩

/-! ## 3 · The verdict

The round's four primitive legs, and what each costs in face choice. -/

/-- **VERDICT. One face serves the whole round: `sigma = 1`.** The rotation leg is not a reason to
    change face, because no face reaches it by multiplication (`rotation_not_ringMul`); the
    additive leg fixes the face at `sigma = 1` (`additive_layer_readout`); and the boolean
    quadratic leg is face-independent (`chi_term_face_free`). So a mixed-face composition buys
    nothing for this round, and the surviving obstruction is `sigma`-free. -/
theorem one_face_serves_the_round {k : ℕ} (hk : 3 ≤ k) :
    (¬ ∃ (σ r a b : ZMod (2 ^ k)), (∀ (v : ℕ) (_hv : v < 2 ^ k),
        rotWrap k (k - 1) v
          = ringUpper r (ringMul σ (((v : ℕ) : ZMod (2 ^ k)), 0) (a, b)))) ∧
      (∃ r : ZMod (2 ^ k), r ^ 2 = 1 ∧ ∀ z : RingCell (ZMod (2 ^ k)), ringUpper r z = z.1 + z.2) ∧
      (∀ x y z : ZMod (2 ^ k), (x, 0) + (z, 0) + ringMul 1 (y, 0) (z, 0) = (x + z + y * z, 0)) :=
  ⟨rotation_not_ringMul hk, (additive_layer_readout (by omega)).1,
    fun x y z => chi_term_face_free 1 x y z⟩

/-! ## 4 · Cone -/

#print axioms ringMul_first_axis
#print axioms ringMul_second_of_first
#print axioms chi_term_face_free
#print axioms ringMul_family_additive
#print axioms rotWrap_not_additive
#print axioms rotation_not_ringMul
#print axioms one_add_one_ne_three_zmod
#print axioms additive_layer_readout
#print axioms no_root_neg_one
#print axioms one_face_serves_the_round

end FaceRound
end AlgebraFaceComposition