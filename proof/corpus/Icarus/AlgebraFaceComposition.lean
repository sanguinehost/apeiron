import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing

/-
# Item 1 of AR-19 — the per-face capability table

Which of SHA-256's primitive operations admit a reading under each of the three faces
`R[x]/(x^2 - sigma)` for `sigma` in `{0, 1, -1}` over `R = ZMod (2^k)`, `k >= 2`.

Every structural fact here is proved at general width `k` from the canonical corpus
(`AnyRing.ringMul`, `AnyRing.ringNormForm`, `ChannelsAnyRing.RingChannel`,
`ChannelsAnyRing.ringUpper`, `ChannelsAnyRing.ringChannel_gen_sq`,
`ChannelsAnyRing.ringNormForm_eq_upper_mul_lower`, `ChannelsAnyRing.ringUpper_channel`).
The finite rows are `#guard`-checked kernel evaluations at `k = 2` and `k = 3`, labelled as such.

No SHA-2 correctness, security, break or performance claim is made or implied.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace AlgebraFaceComposition
namespace FaceTable

open Apeiron.Trichotomy.AnyRing

/-! ## 0 · The three faces

`elliptic` is `sigma = -1`, `parabolic` is `sigma = 0`, `hyperbolic` is `sigma = 1`, the values
`WordRefinement.sigmaI64` gives the three `Sigma` constructors. -/

/-- The three dial values of the corpus, as an inductive type so the table can be indexed. -/
inductive Face where
  /-- `sigma = -1`: the norm form is `a^2 + b^2`. -/
  | elliptic
  /-- `sigma = 0`: the generator is nilpotent, the norm form is `a^2`. -/
  | parabolic
  /-- `sigma = 1`: the norm form is `a^2 - b^2`. -/
  | hyperbolic

/-- The ring element a face dials. -/
def Face.sigma {R : Type*} [CommRing R] : Face → R
  | Face.elliptic => -1
  | Face.parabolic => 0
  | Face.hyperbolic => 1

/-- The three faces, in the order the table lists them. -/
def Face.all : List Face := [Face.elliptic, Face.parabolic, Face.hyperbolic]

@[simp] theorem Face.sigma_elliptic {R : Type*} [CommRing R] :
    Face.sigma (R := R) Face.elliptic = -1 := rfl

@[simp] theorem Face.sigma_parabolic {R : Type*} [CommRing R] :
    Face.sigma (R := R) Face.parabolic = 0 := rfl

@[simp] theorem Face.sigma_hyperbolic {R : Type*} [CommRing R] :
    Face.sigma (R := R) Face.hyperbolic = 1 := rfl

/-! ## 1 · Coordinate facts, all faces

These are the reason most rows of the table are face-independent. -/

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

/-- A second-axis element times a first-axis element, at every face. -/
theorem ringMul_first_of_second {R : Type*} [CommRing R] (σ a b : R) :
    ringMul σ (0, a) (b, 0) = (0, a * b) := by
  simp only [ringMul, Prod.ext_iff]
  constructor <;> ring

/-- Two second-axis elements land in the first axis, scaled by the dial. -/
theorem ringMul_second_second {R : Type*} [CommRing R] (σ a b : R) :
    ringMul σ (0, a) (0, b) = (σ * a * b, 0) := by
  simp only [ringMul, Prod.ext_iff]
  constructor <;> ring

/-- **The dial changes a product only in its first slot, by a `sigma`-multiple of the two second
    slots. This is the single structural fact that governs the whole table.** -/
theorem ringMul_dial_only_in_first_slot {R : Type*} [CommRing R] (σ τ : R) (z w : RingCell R) :
    ringMul τ z w - ringMul σ z w = ((τ - σ) * z.2 * w.2, 0) := by
  simp only [ringMul, Prod.ext_iff, Prod.fst_sub, Prod.snd_sub]
  constructor <;> ring

/-- **The three faces induce the same first-axis subalgebra.** -/
theorem face_agreement_on_first_axis {R : Type*} [CommRing R] (σ τ a b : R) :
    ringMul σ (a, 0) (b, 0) = ringMul τ (a, 0) (b, 0) := by
  rw [ringMul_first_axis, ringMul_first_axis]

/-! ## 2 · The norm form per face -/

/-- Parabolic: `N (a,b) = a^2`. -/
theorem norm_parabolic {R : Type*} [CommRing R] (z : RingCell R) :
    ringNormForm (0 : R) z = z.1 ^ 2 := by
  simp only [ringNormForm]
  ring

/-- Elliptic: `N (a,b) = a^2 + b^2`, the sum of two squares. -/
theorem norm_elliptic {R : Type*} [CommRing R] (z : RingCell R) :
    ringNormForm (-1 : R) z = z.1 ^ 2 + z.2 ^ 2 := by
  simp only [ringNormForm]
  ring

/-- Hyperbolic: `N (a,b) = a^2 - b^2`, the difference of two squares (Euclid II.5 at `sigma = 1`). -/
theorem norm_hyperbolic {R : Type*} [CommRing R] (z : RingCell R) :
    ringNormForm (1 : R) z = z.1 ^ 2 - z.2 ^ 2 := by
  simp only [ringNormForm]
  ring

/-- **The parabolic norm form is completely blind to the second slot.** -/
theorem norm_parabolic_blind {R : Type*} [CommRing R] (a b b' : R) :
    ringNormForm (0 : R) (a, b) = ringNormForm (0 : R) (a, b') := by
  rw [norm_parabolic, norm_parabolic]

/-- At `sigma = 1` the norm form separates second slots whose squares differ. -/
theorem norm_hyperbolic_separates {R : Type*} [CommRing R] (a b b' : R)
    (h : b ^ 2 ≠ b' ^ 2) : ringNormForm (1 : R) (a, b) ≠ ringNormForm (1 : R) (a, b') := by
  have e1 : ringNormForm (1 : R) (a, b) = a ^ 2 - b ^ 2 := norm_hyperbolic (a, b)
  have e2 : ringNormForm (1 : R) (a, b') = a ^ 2 - b' ^ 2 := norm_hyperbolic (a, b')
  rw [e1, e2]
  intro hc
  have h1 : b' ^ 2 - b ^ 2 = 0 := by
    calc b' ^ 2 - b ^ 2 = (a ^ 2 - b ^ 2) - (a ^ 2 - b' ^ 2) := by ring
      _ = 0 := sub_eq_zero_of_eq hc
  exact h (sub_eq_zero.mp h1).symm

/-- At `sigma = -1` the norm form separates second slots whose squares differ. -/
theorem norm_elliptic_separates {R : Type*} [CommRing R] (a b b' : R)
    (h : b ^ 2 ≠ b' ^ 2) : ringNormForm (-1 : R) (a, b) ≠ ringNormForm (-1 : R) (a, b') := by
  have e1 : ringNormForm (-1 : R) (a, b) = a ^ 2 + b ^ 2 := norm_elliptic (a, b)
  have e2 : ringNormForm (-1 : R) (a, b') = a ^ 2 + b' ^ 2 := norm_elliptic (a, b')
  rw [e1, e2]
  intro hc
  have h1 : b ^ 2 - b' ^ 2 = 0 := by
    calc b ^ 2 - b' ^ 2 = (a ^ 2 + b ^ 2) - (a ^ 2 + b' ^ 2) := by ring
      _ = 0 := sub_eq_zero_of_eq hc
  exact h (sub_eq_zero.mp h1)

/-- **The norm form never determines the point, at any face of any commutative ring:** `b` and
    `-b` are distinct second slots with the same square. -/
theorem norm_elliptic_second_sign {R : Type*} [CommRing R] (a : R) :
    ringNormForm (-1 : R) (a, 1) = ringNormForm (-1 : R) (a, -1) := by
  rw [norm_elliptic]
  ring

/-! ## 3 · The channel read per face

`ringUpper r z = z.1 + r * z.2`; the canonical theorem `ChannelsAnyRing.ringUpper_channel`
makes it a `RingChannel sigma` exactly when `r ^ 2 = sigma`, and
`ChannelsAnyRing.ringChannel_gen_sq` forces the converse. -/

/-- Parabolic read: the first slot alone. -/
theorem read_parabolic {R : Type*} [CommRing R] (z : RingCell R) :
    ringUpper (0 : R) z = z.1 := by
  simp [ringUpper]

/-- Hyperbolic read: the sum of the two slots. -/
theorem read_hyperbolic {R : Type*} [CommRing R] (z : RingCell R) :
    ringUpper (1 : R) z = z.1 + z.2 := by
  simp [ringUpper]

/-- **A read of `RingCell R` at `sigma` exists exactly when `sigma` is a square in `R`.** -/
theorem channel_iff_hasRoot {R : Type*} [CommRing R] (σ : R) :
    (∃ φ : RingCell R → R, RingChannel σ φ) ↔ ∃ r : R, r ^ 2 = σ := by
  constructor
  · rintro ⟨φ, hφ⟩
    refine ⟨φ ((0, 1) : RingCell R), ?_⟩
    have h := ringChannel_gen_sq hφ
    rw [← pow_two] at h
    exact h
  · rintro ⟨r, hr⟩
    exact ⟨ringUpper r, ringUpper_channel hr⟩

/-- A read drops the second slot exactly at the root zero. -/
theorem read_is_blind_iff {R : Type*} [CommRing R] (r : R)
    (h : ∀ z : RingCell R, ringUpper r z = z.1) : r = 0 := by
  have hz := h ((0, 1) : RingCell R)
  simp only [ringUpper] at hz
  rw [zero_add, mul_one] at hz
  exact hz

/-! ## 4 · Arithmetic on `ZMod (2^k)` -/

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

/-- **The elliptic face has no read: `-1` is not a square in `ZMod (2^k)` for any `k >= 2`.** -/
theorem no_root_neg_one {k : ℕ} (hk : 2 ≤ k) :
    ¬ ∃ r : ZMod (2 ^ k), r ^ 2 = (-1 : ZMod (2 ^ k)) := by
  rintro ⟨r, hr⟩
  have h := cast_to_four hk
  have h1 : (h r) ^ 2 = (-1 : ZMod 4) := by
    have := congrArg (fun t : ZMod (2 ^ k) => (h t : ZMod 4)) hr
    simpa using this
  exact mod4_sq_ne_neg_one _ h1

/-- **No sum of two squares in `ZMod (2^k)` is `-1`, for any `k >= 2`.** -/
theorem no_sum2_neg_one {k : ℕ} (hk : 2 ≤ k) (c d : ZMod (2 ^ k)) :
    c ^ 2 + d ^ 2 ≠ (-1 : ZMod (2 ^ k)) := by
  intro hcd
  have h := cast_to_four hk
  have h1 : (h c) ^ 2 + (h d) ^ 2 = (-1 : ZMod 4) := by
    have := congrArg (fun t : ZMod (2 ^ k) => (h t : ZMod 4)) hcd
    simpa using this
  exact mod4_sum2_ne_neg_one _ _ h1

/-- `1 ≠ 0` in `ZMod (2^k)` for `k >= 1`. -/
theorem one_ne_zero_zmod {k : ℕ} (hk : 1 ≤ k) : (1 : ZMod (2 ^ k)) ≠ 0 := by
  intro hc
  have h1 : 2 ^ k = 1 := ZMod.one_eq_zero_iff.mp hc
  have hlt : (1 : ℕ) < 2 ^ k := by
    have := Nat.pow_lt_pow_right (by norm_num : (1 : ℕ) < 2) hk
    simpa using this
  omega

/-! ## 5 · The capability predicates

One predicate per column of the table, so each row is a single proposition. -/

/-- The face admits a channel read. -/
def Readable {R : Type*} [CommRing R] (σ : R) : Prop := ∃ r : R, r ^ 2 = σ

/-- The face admits a channel read, and that read is the first slot alone. -/
def ReadBlind {R : Type*} [CommRing R] (σ : R) : Prop :=
  ∃ r : R, r ^ 2 = σ ∧ ∀ z : RingCell R, ringUpper r z = z.1

/-- The face's norm form cannot distinguish two second slots at a fixed first slot. -/
def NormInvisible {R : Type*} [CommRing R] (σ : R) : Prop :=
  ∀ (a b b' : R), ringNormForm σ (a, b) = ringNormForm σ (a, b')

/-- The face's norm form is the product of its two channel coordinates. -/
def NormFactors {R : Type*} [CommRing R] (σ : R) : Prop :=
  ∃ r : R, r ^ 2 = σ ∧ ∀ z : RingCell R, ringNormForm σ z = ringUpper r z * ringLower r z

/-- The face can hold a product of two first-axis elements in its second slot. -/
def CarrySlot {R : Type*} [CommRing R] (σ : R) : Prop :=
  ∀ a b : R, ringMul σ (a, 0) (0, b) = (0, a * b)

/-- The face can hold a product of two first-axis elements in its first slot. -/
def ProdAxis {R : Type*} [CommRing R] (σ : R) : Prop :=
  ∀ a b : R, ringMul σ (a, 0) (b, 0) = (a * b, 0)

/-- **A read is blind exactly at the parabolic face, at any ring.** -/
theorem readBlind_iff {R : Type*} [CommRing R] (σ : R) : ReadBlind σ ↔ σ = 0 := by
  constructor
  · rintro ⟨r, hr, h⟩
    have h0 := read_is_blind_iff r h
    have hz : σ = (0 : R) ^ 2 := hr.symm.trans (congrArg (fun t : R => t ^ 2) h0)
    simpa using hz
  · intro h
    subst h
    exact ⟨0, by simp, fun z => read_parabolic z⟩

/-- **The norm form factors into the two channel coordinates exactly when the face is readable.** -/
theorem normFactors_iff_readable {R : Type*} [CommRing R] (σ : R) :
    NormFactors σ ↔ Readable σ := by
  constructor
  · rintro ⟨r, hr, -⟩; exact ⟨r, hr⟩
  · rintro ⟨r, hr⟩
    exact ⟨r, hr, fun z => ringNormForm_eq_upper_mul_lower hr z⟩

/-- **A product lands in the second slot, and the first slot carries a product, at every face.**
    This is the face-independence of the carry mechanism and of the quadratic term of chi. -/
theorem carrySlot_all {R : Type*} [CommRing R] (σ : R) : CarrySlot σ :=
  fun _ _ => ringMul_second_of_first σ _ _

theorem prodAxis_all {R : Type*} [CommRing R] (σ : R) : ProdAxis σ :=
  fun _ _ => ringMul_first_axis σ _ _

/-- A blind read is in particular a read. -/
theorem readable_of_readBlind {R : Type*} [CommRing R] {σ : R} (h : ReadBlind σ) : Readable σ := by
  rcases h with ⟨r, hr, -⟩
  exact ⟨r, hr⟩

/-- A factoring norm form is in particular a read. -/
theorem readable_of_normFactors {R : Type*} [CommRing R] {σ : R} (h : NormFactors σ) : Readable σ := by
  rcases h with ⟨r, hr, -⟩
  exact ⟨r, hr⟩

theorem root_zero_zmod {k : ℕ} : Readable (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) :=
  ⟨(0 : ZMod (2 ^ k)), by simp⟩

theorem root_one_zmod {k : ℕ} : Readable (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) :=
  ⟨(1 : ZMod (2 ^ k)), by rw [pow_two, mul_one]⟩

/-- `1 ^ 2 ≠ 0 ^ 2` in `ZMod (2^k)` for `k >= 1`. -/
theorem one_sq_ne_zero_sq_zmod {k : ℕ} (hk : 1 ≤ k) :
    (1 : ZMod (2 ^ k)) ^ 2 ≠ (0 : ZMod (2 ^ k)) ^ 2 := by
  intro hc
  have h1 : (1 : ZMod (2 ^ k)) ^ 2 = 0 := by rw [hc]; simp
  rw [pow_two, mul_one] at h1
  exact one_ne_zero_zmod (k := k) hk h1

theorem normInvisible_one_zmod {k : ℕ} (hk : 1 ≤ k) :
    ¬ NormInvisible (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) := by
  intro h
  exact norm_hyperbolic_separates (R := ZMod (2 ^ k)) 1 1 0
    (one_sq_ne_zero_sq_zmod hk) (h 1 1 0)

theorem normInvisible_neg_one_zmod {k : ℕ} (hk : 1 ≤ k) :
    ¬ NormInvisible (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) := by
  intro h
  exact norm_elliptic_separates (R := ZMod (2 ^ k)) 1 1 0
    (one_sq_ne_zero_sq_zmod hk) (h 1 1 0)

theorem readBlind_zero_zmod {k : ℕ} : ReadBlind (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) :=
  ⟨(0 : ZMod (2 ^ k)), by rw [pow_two, zero_mul], fun z => read_parabolic z⟩

theorem not_readBlind_one_zmod {k : ℕ} (hk : 1 ≤ k) :
    ¬ ReadBlind (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) :=
  fun h => one_ne_zero_zmod (k := k) hk
    ((@readBlind_iff (ZMod (2 ^ k)) _ (1 : ZMod (2 ^ k))).mp h)

theorem normFactors_zero_zmod {k : ℕ} : NormFactors (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) :=
  (@normFactors_iff_readable (ZMod (2 ^ k)) _ (0 : ZMod (2 ^ k))).2 root_zero_zmod

theorem normFactors_one_zmod {k : ℕ} : NormFactors (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) :=
  (@normFactors_iff_readable (ZMod (2 ^ k)) _ (1 : ZMod (2 ^ k))).2 root_one_zmod

/-! ## 6 · The capability table, as theorems

Row by row over `R = ZMod (2^k)`, `k >= 2`. -/

/-- **Row `sigma = 0` (parabolic):** a read exists and is blind, the norm form is blind, the norm
    form still factors, and both slots still carry products. -/
theorem row_parabolic {k : ℕ} (_hk : 2 ≤ k) :
    Readable (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) ∧
      ReadBlind (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) ∧
      NormInvisible (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) ∧
      NormFactors (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) ∧
      CarrySlot (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) ∧
      ProdAxis (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) :=
  ⟨root_zero_zmod, readBlind_zero_zmod, by
      intro a b b'; rw [norm_parabolic, norm_parabolic],
    normFactors_zero_zmod, carrySlot_all _, prodAxis_all _⟩

/-- **Row `sigma = -1` (elliptic):** no read at all, and therefore no blind one; the norm form
    separates second slots; the norm form does not factor; both slots still carry products. -/
theorem row_elliptic {k : ℕ} (hk : 2 ≤ k) :
    ¬ Readable (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) ∧
      ¬ ReadBlind (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) ∧
      ¬ NormInvisible (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) ∧
      ¬ NormFactors (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) ∧
      CarrySlot (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) ∧
      ProdAxis (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) := by
  have hnr := no_root_neg_one hk
  exact ⟨hnr, fun h => hnr (readable_of_readBlind h), normInvisible_neg_one_zmod (by omega),
    fun h => hnr (readable_of_normFactors h), carrySlot_all _, prodAxis_all _⟩

/-- **Row `sigma = 1` (hyperbolic):** a read exists and is not blind, the norm form separates
    second slots, the norm form factors, and both slots carry products. -/
theorem row_hyperbolic {k : ℕ} (hk : 2 ≤ k) :
    Readable (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) ∧
      ¬ ReadBlind (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) ∧
      ¬ NormInvisible (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) ∧
      NormFactors (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) ∧
      CarrySlot (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) ∧
      ProdAxis (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) := by
  exact ⟨root_one_zmod, not_readBlind_one_zmod (by omega), normInvisible_one_zmod (by omega),
    normFactors_one_zmod, carrySlot_all _, prodAxis_all _⟩

/-- **The three rows are pairwise distinct on the readout columns**, so a face choice is a real
    decision and not a relabelling: parabolic is the only blind face, elliptic the only one with
    no read, and the only one whose norm form refuses to factor. -/
theorem rows_distinct {k : ℕ} (hk : 2 ≤ k) :
    ReadBlind (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) ∧
      ¬ ReadBlind (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) ∧
      ¬ Readable (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) ∧
      NormFactors (R := ZMod (2 ^ k)) (0 : ZMod (2 ^ k)) ∧
      NormFactors (R := ZMod (2 ^ k)) (1 : ZMod (2 ^ k)) ∧
      ¬ NormFactors (R := ZMod (2 ^ k)) (-1 : ZMod (2 ^ k)) :=
  ⟨readBlind_zero_zmod,
   not_readBlind_one_zmod (by omega),
   no_root_neg_one hk,
   normFactors_zero_zmod, normFactors_one_zmod,
   fun h => no_root_neg_one hk (readable_of_normFactors h)⟩

/-! ## 7 · Finite, `#guard`-checked enumerations at `k = 2` and `k = 3`

Complete enumerations of `ZMod 4` (four elements) and `ZMod 8` (eight elements), run by the
kernel: finite evidence, not a sample. -/

/-- The roots of `sigma` in `ZMod n`. -/
def rootsIn {n : ℕ} [NeZero n] (a : ZMod n) : Finset (ZMod n) :=
  Finset.univ.filter (fun r => r ^ 2 = a)

#guard rootsIn (0 : ZMod 4) = ({0, 2} : Finset (ZMod 4))
#guard rootsIn (1 : ZMod 4) = ({1, 3} : Finset (ZMod 4))
#guard rootsIn (-1 : ZMod 4) = (∅ : Finset (ZMod 4))

#guard rootsIn (0 : ZMod 8) = ({0, 4} : Finset (ZMod 8))
#guard rootsIn (1 : ZMod 8) = ({1, 3, 5, 7} : Finset (ZMod 8))
#guard rootsIn (-1 : ZMod 8) = (∅ : Finset (ZMod 8))

theorem rootsIn4_zero : rootsIn (0 : ZMod 4) = ({0, 2} : Finset (ZMod 4)) := by decide
theorem rootsIn4_one : rootsIn (1 : ZMod 4) = ({1, 3} : Finset (ZMod 4)) := by decide
theorem rootsIn4_neg_one : rootsIn (-1 : ZMod 4) = (∅ : Finset (ZMod 4)) := by decide
theorem rootsIn8_zero : rootsIn (0 : ZMod 8) = ({0, 4} : Finset (ZMod 8)) := by decide
theorem rootsIn8_one : rootsIn (1 : ZMod 8) = ({1, 3, 5, 7} : Finset (ZMod 8)) := by decide
theorem rootsIn8_neg_one : rootsIn (-1 : ZMod 8) = (∅ : Finset (ZMod 8)) := by decide

/-- **The read at the hyperbolic face is not canonical: at `k = 2` two roots of `1` give two
    different reads, at `k = 3` four do.** The root is part of the face specification. -/
theorem roots_not_unique_two : (rootsIn (1 : ZMod 4)).card = 2 := by decide
theorem roots_not_unique_three : (rootsIn (1 : ZMod 8)).card = 4 := by decide
theorem roots_zero_not_unique_two : (rootsIn (0 : ZMod 4)).card = 2 := by decide
theorem roots_zero_not_unique_three : (rootsIn (0 : ZMod 8)).card = 2 := by decide

#guard (1 : ZMod 4) ≠ (-1 : ZMod 4)
#guard ringUpper (1 : ZMod 4) (0, 1) = 1
#guard ringUpper (-1 : ZMod 4) (0, 1) = -1

/-! ## 8 · The readout rows of the round

The carry cell and the rotation cell. `BitVec k` addition is addition in `ZMod (2^k)`. -/

/-- A `k`-bit word read as an element of `ZMod (2^k)`. -/
def bvDen {k : ℕ} (w : BitVec k) : ZMod (2 ^ k) := (w.toNat : ZMod (2 ^ k))

/-- Wrapping addition of `k`-bit words is addition in `ZMod (2^k)`. -/
theorem bvDen_add (k : ℕ) (x y : BitVec k) : bvDen (x + y) = bvDen x + bvDen y := by
  rw [bvDen, BitVec.toNat_add, ZMod.natCast_mod]
  exact Nat.cast_add _ _

/-- **The carry cell: the additive layer's two-slot representative.** Its first slot is the
    exclusive-or and its second slot is the doubled bitwise conjunction. -/
def carryCell {k : ℕ} (_σ : ZMod (2 ^ k)) (x y : BitVec k) : RingCell (ZMod (2 ^ k)) :=
  (bvDen (x ^^^ y), 2 * bvDen (x &&& y))

/-- **The carry cell is one `ringMul` against the unit plus one `ringMul` of the two axes, at
    every face.** -/
theorem carryCell_eq_ringMul_sum {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    carryCell σ x y
      = ringMul σ (bvDen (x ^^^ y), 0) ringOne
        + ringMul σ (2, 0) (0, bvDen (x &&& y)) := by
  rw [carryCell, ringOne, ringMul_first_axis, ringMul_second_of_first]
  apply Prod.ext_iff.mpr
  constructor <;> simp

/-- **The read of the carry cell at an arbitrary root `r`.** -/
theorem read_carryCell_form {k : ℕ} (r : ZMod (2 ^ k)) (x y : BitVec k) :
    ringUpper r (carryCell r x y) = bvDen (x ^^^ y) + r * 2 * bvDen (x &&& y) := by
  simp only [carryCell, ringUpper]
  ring

/-- **At the parabolic root the read is blind to the carry: it returns the exclusive-or only.** -/
theorem read_carryCell_parabolic_blind {k : ℕ} (x y : BitVec k) :
    ringUpper (0 : ZMod (2 ^ k)) (carryCell 0 x y) = bvDen (x ^^^ y) := by
  simp only [read_carryCell_form]
  ring

/-- **At the hyperbolic root `1` the read is the modular sum**, at `k = 2` and at `k = 4`,
    completely: all `2^4` and all `2^8` pairs. -/
theorem read_carryCell_is_sum_two : ∀ (x y : BitVec 2),
    ringUpper (1 : ZMod 4) (carryCell (1 : ZMod 4) x y) = bvDen (x + y) := by
  decide

theorem read_carryCell_is_sum_four : ∀ (x y : BitVec 4),
    ringUpper (1 : ZMod 16) (carryCell (1 : ZMod 16) x y) = bvDen (x + y) := by
  decide

/-- **At every face the two slots of the carry cell recompose the sum at `k = 2` and `k = 4`.** -/
theorem slots_recompose_sum_two : ∀ (x y : BitVec 2),
    (carryCell (0 : ZMod 4) x y).1 + (carryCell (0 : ZMod 4) x y).2 = bvDen (x + y) := by
  decide

theorem slots_recompose_sum_four : ∀ (x y : BitVec 4),
    (carryCell (0 : ZMod 16) x y).1 + (carryCell (0 : ZMod 16) x y).2 = bvDen (x + y) := by
  decide

/-- The truncated shifted word of a rotation by `n`. -/
def rotShift (k n v : ℕ) : ZMod (2 ^ k) := Nat.cast ((v * 2 ^ n) % 2 ^ k)

/-- The bits wrapped around by a rotation by `n`. -/
def rotWrap (k n v : ℕ) : ZMod (2 ^ k) := Nat.cast (v / 2 ^ (k - n))

/-- The rotation cell: first slot the truncated shift, second slot the wrapped-around high bits.
    `FIPS 180-4 §3.2` `ROTL` is the sum of these two in the ring, which is what the hyperbolic
    read delivers. -/
def rotCell (k n v : ℕ) : RingCell (ZMod (2 ^ k)) := (rotShift k n v, rotWrap k n v)

/-- **The hyperbolic read of the rotation cell is the sum of the two terms.** -/
theorem read_rotCell {k : ℕ} (n v : ℕ) :
    ringUpper (1 : ZMod (2 ^ k)) (rotCell k n v) = rotShift k n v + rotWrap k n v := by
  simp only [rotCell, ringUpper, one_mul]

/-- `4 ≤ 2^k` for `k >= 2`. -/
theorem two_pow_ge_four {k : ℕ} (hk : 2 ≤ k) : (4 : ℕ) ≤ 2 ^ k := by
  have hh : 2 ^ 2 ≤ 2 ^ k :=
    (Nat.pow_le_pow_iff_right (a := 2) (n := 2) (m := k) (by omega)).2 hk
  simpa using hh

/-- Natural representatives below the modulus are distinct in `ZMod n`. -/
theorem cast_ne_of_val_ne {n : ℕ} (a b : ℕ) (hab : a < n) (hbb : b < n) (h : a ≠ b) :
    ((a : ℕ) : ZMod n) ≠ ((b : ℕ) : ZMod n) := by
  intro hc
  have hv := congrArg ZMod.val hc
  rw [ZMod.val_natCast, ZMod.val_natCast] at hv
  rw [Nat.mod_eq_of_lt hab, Nat.mod_eq_of_lt hbb] at hv
  exact h hv

/-- `2 ≠ 3` in `ZMod (2^k)` for `k >= 2`. -/
theorem two_ne_three_zmod {k : ℕ} (hk : 2 ≤ k) :
    ((2 : ℕ) : ZMod (2 ^ k)) ≠ ((3 : ℕ) : ZMod (2 ^ k)) := by
  have h4 := two_pow_ge_four hk
  exact cast_ne_of_val_ne 2 3 (by omega) (by omega) (by decide)

/-- `1 + 1 ≠ 3` in `ZMod (2^k)` for `k >= 1`. -/
theorem one_add_one_ne_three_zmod {k : ℕ} (hk : 2 ≤ k) :
    ((1 : ℕ) : ZMod (2 ^ k)) + ((1 : ℕ) : ZMod (2 ^ k)) ≠ ((3 : ℕ) : ZMod (2 ^ k)) := by
  have h4 : (4 : ℕ) ≤ 2 ^ k := two_pow_ge_four (k := k) hk
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

/-- **The rotation cell is not additive in the word, at every `k >= 2`.** A cell built by one
    `ringMul` against the word is additive in it (item 3 of AR-19), so no face reaches this cell
    by multiplication. Witness: rotation by `k-1`, the word `3`, whose second slot is the wrapped
    high bit `1`; `1 + 1 = 2` but `⌊6/2⌋ = 3`. -/
theorem rotCell_not_additive {k : ℕ} (hk : 2 ≤ k) :
    rotCell k (k - 1) 3 + rotCell k (k - 1) 3 ≠ rotCell k (k - 1) 6 := by
  have hk' : k - (k - 1) = 1 := by omega
  have h3 : rotWrap k (k - 1) 3 = 1 := by
    unfold rotWrap; rw [hk']
    rw [(show (3 : ℕ) / 2 ^ 1 = 1 from by norm_num), Nat.cast_one]
  have h6 : rotWrap k (k - 1) 6 = 3 := by
    unfold rotWrap; rw [hk']
    rw [(show (6 : ℕ) / 2 ^ 1 = 3 from by norm_num), Nat.cast_ofNat]
  intro hc
  have hp := Prod.ext_iff.mp hc
  rw [rotCell] at hp
  have h2 : (rotWrap k (k - 1) 3) + (rotWrap k (k - 1) 3) = rotWrap k (k - 1) 6 := by
    simpa only [rotCell, Prod.snd_add] using hp.2
  rw [h3, h6] at h2
  have h2' : ((1 : ℕ) : ZMod (2 ^ k)) + ((1 : ℕ) : ZMod (2 ^ k))
      = ((3 : ℕ) : ZMod (2 ^ k)) := by
    rw [Nat.cast_one, Nat.cast_ofNat]
    exact h2
  exact one_add_one_ne_three_zmod (k := k) hk h2'

/-! ## 9 · Cone -/

#print axioms ringMul_dial_only_in_first_slot
#print axioms channel_iff_hasRoot
#print axioms readBlind_iff
#print axioms normFactors_iff_readable
#print axioms row_parabolic
#print axioms row_elliptic
#print axioms row_hyperbolic
#print axioms rows_distinct
#print axioms rootsIn4_one
#print axioms rootsIn8_one
#print axioms carryCell_eq_ringMul_sum
#print axioms read_carryCell_form
#print axioms read_carryCell_parabolic_blind
#print axioms read_carryCell_is_sum_two
#print axioms read_carryCell_is_sum_four
#print axioms slots_recompose_sum_two
#print axioms slots_recompose_sum_four
#print axioms read_rotCell
#print axioms rotCell_not_additive
#print axioms one_add_one_ne_three_zmod
#print axioms cast_ne_of_val_ne
#print axioms two_pow_ge_four
#print axioms readable_of_readBlind
#print axioms readBlind_zero_zmod
#print axioms no_root_neg_one
#print axioms no_sum2_neg_one

end FaceTable
end AlgebraFaceComposition