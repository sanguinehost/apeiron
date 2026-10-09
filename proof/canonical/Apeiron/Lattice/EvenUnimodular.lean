import Apeiron.Lattice.IntegralForm
import Apeiron.Doubling.Octonion
import Apeiron.Physics.Ladder
import Apeiron.Trichotomy.SplitComplex
import Mathlib.LinearAlgebra.Matrix.Cartan
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.Ring.Star

/-!
# `E₈^e ⊕ U^u`: the family exists, its signature difference is `8e`, and rank 26 is realised

@name EvenUnimodular
@cite bourbaki-1968
@cite mathlib-cartan
@cite milnor-husemoller-1973
@cite vanderblij-1959
@cite watson-1918
@cite conway-sloane-1999
@grade proved

Two generators, both built here rather than posited.

**The rank-two summand `U`.** Its Gram matrix is `!![0, 1; 1, 0]`, and
`hyperbolicPlane_gram_is_the_null_pair` shows that this is the Gram matrix of
`Trichotomy.form 1` on the pair `(1, 1)`, `(1/2, −1/2)`, which is
`Trichotomy.SplitComplex.nullPlus` and `Trichotomy.SplitComplex.eMinus`. Both are isotropic and
they pair to one. `Trichotomy.hasNullPair_iff_disc_pos` says a pair of that kind exists exactly
for `σ > 0`, so the count `u` in `Physics.dimOf e u = 8e + 2u` counts summands carrying the
`σ > 0` null pair, by theorem rather than by naming.

**The rank-eight summand `E₈`.** Its Gram matrix is Mathlib's `CartanMatrix.E₈`, and
`e8_octonion_gram` computes it as the polarisation of `Doubling.normO` on eight octonions written
in Bourbaki's plate VII coordinates. So the definite block is the octonion norm form on a lattice
of octonions, and reaches `Apeiron.Trichotomy` through `Doubling.real_mul` like everything else in
the tower. What that does **not** say is that the `ℤ`-span of those eight octonions is closed
under octonion multiplication: only the form is used, so this is the norm-form half of "`E₈` is
the ring of integral octonions" and not the ring half.

`det E₈ = 1` is a `proof_wanted` in Mathlib. It is proved here twice over: an explicit integer
inverse gives `det = ±1`, and the factorisation `E₈ = B Bᵀ` in octonion coordinates gives
`(det E₈ : ℝ) = (det B)² ≥ 0`, which rules out `−1`.

## What is proved about the family

`evenUnimodular e u = (E₈)^e ⊕ U^u` is even, unimodular, of rank `Physics.dimOf e u = 8e + 2u`,
and carries the signature witness `(8e + u, u)`. Hence `p − q = 8e` on this family
(`signature_difference`) and `8 ∣ p − q` (`eight_dvd_signature_difference`).

That is the **existence** direction and it is stated at exactly that strength. The converse, that
*every* even unimodular form has `p − q ≡ 0 (mod 8)`, is van der Blij's congruence, is not proved
here, and is not used by anything downstream. Three routes to it were checked and each needs one
Mathlib primitive that does not exist: Hasse-Minkowski (van der Blij's own route, through Meyer's
theorem that an indefinite unimodular form in five or more variables represents zero), the sign of
the quadratic Gauss sum (Milgram's route; `Mathlib/NumberTheory/GaussSum.lean` gives
`gaussSum_sq`, the square, and stops there), and lattice Poisson summation (the theta route;
`Mathlib/Analysis/Fourier/PoissonSummation.lean` covers `ℤ ⊂ ℝ` only). The entry in
`theory/DEAD_ENDS.md` records this.

Completeness, "these are all of them", is claimed nowhere and is false in the definite case: `E₈²`
and `D₁₆⁺` are non-isometric even unimodular lattices of signature `(16, 0)`.

## Rank 26

`bosonic_twentysix` exhibits `E₈³ ⊕ U`: even, unimodular, rank 26, signature witness `(25, 1)`.
`cannonball_selects_twentysix` states what picks 26 out of the one-cell column `10, 18, 26`: the
member whose definite block `8e` is a height at which the square pyramid `∑_{k≤n} k²` is a perfect
square. `local_cannonball_uniqueness` proves that over the window `2 ≤ n ≤ 30` the only such
height is `24`, by a kernel-checked square test. That the window can be removed is Watson 1918 and
is cited, not proved: Mathlib has no genus-one machinery to prove it with.

Rank alone does not fix the signature. Rank 26 also admits `E₈² ⊕ U⁵`, of signature `(21, 5)`,
which is even and unimodular by the same construction. Selecting `(25, 1)` needs one timelike
direction as an input, and that input is physics, not a theorem here.

## Root

| from | here |
|---|---|
| `form`, `SplitComplex.nullPlus`, `SplitComplex.eMinus` | `hyperbolicPlane_gram_is_the_null_pair` |
| `hasNullPair_iff_disc_pos` | the reading of `u` as a count of null pairs |
| `Doubling.normO_eq`, through `Doubling.real_mul` | `oForm_self`, `e8_octonion_gram` |
| `e8_octonion_gram` | `e8_gramR_eq`, `E8_det`, `E8_posDef`, `E8_hasSignature` |
| `Physics.dimOf` | `euRank_eq_dimOf`, `cannonball_selects_twentysix` |

The first row is why `u` counts null pairs by theorem: `!![0,1;1,0]` is not a matrix chosen to be
even and unimodular, it is the matrix of `Trichotomy.form 1` on the two isotropic vectors the
`σ > 0` face already carries, and `hasNullPair_iff_disc_pos` says that face is where such a pair
exists. The second row is why the definite block is not an import: `E8_det` and `E8_posDef` both
run through the factorisation in octonion coordinates, so deleting `Doubling.normO` deletes them.
-/

namespace Apeiron
namespace Lattice

open Matrix Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- THE RANK-TWO SUMMAND, AND ITS GRAM MATRIX AS THE NULL PAIR
-- ═══════════════════════════════════════════════════════════════════

/-- The hyperbolic plane `U`: the rank-two even form of determinant minus one.

    hyperbolicPlane                    [the rank-two even form with Gram matrix !![0,1;1,0]]
    IntForm                            [an integral symmetric bilinear form] -/
def hyperbolicPlane : IntForm (Fin 2) where
  gram := !![0, 1; 1, 0]
  gram_symm := by decide

/-- Its Gram matrix, by definition.

    hyperbolicPlane                    [the rank-two even form with Gram matrix !![0,1;1,0]] -/
@[simp] theorem hyperbolicPlane_gram : hyperbolicPlane.gram = !![0, 1; 1, 0] := rfl

/-- The corpus's own null basis at `σ = 1`: the isotropic vector `1 + j` and the idempotent
    `(1 − j)/2`.

    nullBasis                          [the two isotropic vectors of the σ = 1 algebra]
    Cell                               [a point: two real numbers written side by side]
    →                                  [takes the left and gives the right] -/
noncomputable def nullBasis : Fin 2 → Cell := ![SplitComplex.nullPlus, SplitComplex.eMinus]

/-- **The hyperbolic plane's Gram matrix is the Gram matrix of the null pair at `σ = 1`.**

    (i c : Fin 2)                                                    [two basis indices]
    form 1 (nullBasis i) (nullBasis c) = (hyperbolicPlane.gram i c : ℝ)
    form                                [how much two points pull on each other, measured the σ way]
    nullBasis                                       [the two isotropic vectors of the σ = 1 algebra]
    hyperbolicPlane                                                  [the rank-two even form]
    ℝ                                                                [the real numbers]

    So: both basis vectors are isotropic and they pair to one, which is the whole of
    `!![0,1;1,0]`. `Trichotomy.hasNullPair_iff_disc_pos` says a pair of that kind exists exactly
    for `σ > 0`, so the count `u` in `Physics.dimOf e u` counts summands carrying the `σ > 0` null
    pair rather than summands assigned that name. -/
theorem hyperbolicPlane_gram_is_the_null_pair (i c : Fin 2) :
    form 1 (nullBasis i) (nullBasis c) = (hyperbolicPlane.gram i c : ℝ) := by
  fin_cases i <;> fin_cases c <;>
    norm_num [nullBasis, form, SplitComplex.nullPlus, SplitComplex.eMinus]

/-- **The rank-two form is even.**

    hyperbolicPlane                    [the rank-two even form with Gram matrix !![0,1;1,0]]

    So: its diagonal is zero, so every vector pairs with itself to an even integer. -/
theorem hyperbolicPlane_isEven : hyperbolicPlane.IsEven := by
  rw [IntForm.isEven_iff_even_diag]
  intro i
  fin_cases i <;> exact ⟨0, by decide⟩

/-- Its determinant is minus one.

    hyperbolicPlane                    [the rank-two even form with Gram matrix !![0,1;1,0]] -/
@[simp] theorem hyperbolicPlane_det : hyperbolicPlane.gram.det = -1 := by
  rw [hyperbolicPlane_gram, Matrix.det_fin_two]
  decide

/-- **The rank-two form is unimodular.**

    hyperbolicPlane                    [the rank-two even form with Gram matrix !![0,1;1,0]] -/
theorem hyperbolicPlane_isUnimodular : hyperbolicPlane.IsUnimodular := by
  rw [IntForm.IsUnimodular, hyperbolicPlane_det]
  exact isUnit_one.neg

/-- The sign vector of the hyperbolic plane: one plus, one minus.

    hSign                              [the sign vector (1, -1)]
    Fin 2 → IntForm.SignatureSign       [two structural signs indexed by 0 and 1]
    →                                  [takes the left and gives the right]
    IntForm.SignatureSign               [one of the two finite signs] -/
def hSign : Fin 2 → IntForm.SignatureSign := ![.positive, .negative]

/-- It has one plus sign.

    sigPos                             [how many plus ones a sign vector has]
    hSign                              [the sign vector (1, -1)]
    IntForm                            [an integral symmetric bilinear form] -/
@[simp] theorem hSign_sigPos : IntForm.sigPos hSign = 1 := by
  decide

/-- It has one minus sign.

    sigNeg                             [how many entries are not plus one]
    hSign                              [the sign vector (1, -1)]
    IntForm                            [an integral symmetric bilinear form] -/
@[simp] theorem hSign_sigNeg : IntForm.sigNeg hSign = 1 := by
  decide

/-- Its Gram matrix over the reals.

    hyperbolicPlane                    [the rank-two even form with Gram matrix !![0,1;1,0]]
    ℝ                                  [the real numbers] -/
theorem hyperbolicPlane_gramR : hyperbolicPlane.gramR = !![(0 : ℝ), 1; 1, 0] := by
  ext i c
  fin_cases i <;> fin_cases c <;> simp [IntForm.gramR, hyperbolicPlane]

/-- **The hyperbolic plane diagonalises to one plus and one minus.**

    hyperbolicPlane.HasSignature hSign          [the sign vector (1, -1) is a witness]
    hyperbolicPlane                             [the rank-two even form]
    hSign                                       [the sign vector (1, -1)]

    So: the change of basis is the light-cone one, whose columns are the two isotropic
    directions. -/
theorem hyperbolicPlane_hasSignature : hyperbolicPlane.HasSignature hSign := by
  refine ⟨!![1, 1; (1 : ℝ) / 2, -(1 / 2)], ?_, ?_⟩
  · rw [Matrix.det_fin_two]
    norm_num
  · rw [hyperbolicPlane_gramR]
    ext i c
    fin_cases i <;> fin_cases c <;>
      norm_num [Matrix.mul_apply, Fin.sum_univ_two, Matrix.diagonal_apply, hSign,
        IntForm.SignatureSign.toReal]

-- ═══════════════════════════════════════════════════════════════════
-- THE RANK-EIGHT SUMMAND, AS THE OCTONION NORM FORM
-- ═══════════════════════════════════════════════════════════════════

/-- The eight real coordinates of an octonion.

    (x : Doubling.Octonion)            [an octonion]
    oCoord                             [the eight real coordinates of an octonion]
    : Fin 8 → ℝ                        [its eight real coordinates]
    →                                  [takes the left and gives the right]
    ℝ                                  [the real numbers] -/
def oCoord (x : Doubling.Octonion) : Fin 8 → ℝ :=
  ![(Doubling.fst x).re, (Doubling.fst x).imI, (Doubling.fst x).imJ, (Doubling.fst x).imK,
    (Doubling.snd x).re, (Doubling.snd x).imI, (Doubling.snd x).imJ, (Doubling.snd x).imK]

/-- The polarisation of the octonion norm form, in those coordinates.

    (x y : Doubling.Octonion)          [two octonions]
    oForm                              [the polarisation of the octonion norm form]
    : ℝ                                [a real number]
    ℝ                                  [the real numbers] -/
def oForm (x y : Doubling.Octonion) : ℝ := ∑ i, oCoord x i * oCoord y i

/-- **On the diagonal the polarisation is the octonion norm form.**

    (x : Doubling.Octonion)                      [an octonion]
    oForm x x = Doubling.normO x                 [it agrees with the norm form on the diagonal]
    normO                                        [the octonion norm form]

    So: the bilinear form `E₈` is the Gram matrix of is `Doubling.normO`, not a form chosen
    beside it. -/
theorem oForm_self (x : Doubling.Octonion) : oForm x x = Doubling.normO x := by
  simp [oForm, oCoord, Fin.sum_univ_eight, Doubling.normO_eq, Quaternion.normSq_def']
  ring

/-- **The polarisation identity.**

    (x y : Doubling.Octonion)                                        [two octonions]
    oForm                                   [the polarisation of the octonion norm form]
    2 * oForm x y
      = Doubling.normO (x + y) - Doubling.normO x - Doubling.normO y [twice the pairing]
    normO                                                            [the octonion norm form]

    So: the form is determined by the norm form and carries no extra choice. -/
theorem oForm_polarization (x y : Doubling.Octonion) :
    2 * oForm x y = Doubling.normO (x + y) - Doubling.normO x - Doubling.normO y := by
  simp [oForm, oCoord, Fin.sum_univ_eight, Doubling.normO_eq, Quaternion.normSq_def']
  ring

/-- Bourbaki plate VII's eight simple roots of `E₈`, written as octonions in the coordinates
    `oCoord`.

    e8Oct                              [the eight simple roots, as octonions]
    Doubling.Octonion                  [the octonions]
    →                                  [takes the left and gives the right] -/
noncomputable def e8Oct : Fin 8 → Doubling.Octonion :=
  ![Doubling.ofPair (⟨1/2, -(1/2), -(1/2), -(1/2)⟩, ⟨-(1/2), -(1/2), -(1/2), 1/2⟩),
    Doubling.ofPair (⟨1, 1, 0, 0⟩, 0),
    Doubling.ofPair (⟨-1, 1, 0, 0⟩, 0),
    Doubling.ofPair (⟨0, -1, 1, 0⟩, 0),
    Doubling.ofPair (⟨0, 0, -1, 1⟩, 0),
    Doubling.ofPair (⟨0, 0, 0, -1⟩, ⟨1, 0, 0, 0⟩),
    Doubling.ofPair (0, ⟨-1, 1, 0, 0⟩),
    Doubling.ofPair (0, ⟨0, -1, 1, 0⟩)]

/-- The coordinate matrix of the eight roots: row `i` is `oCoord (e8Oct i)`.

    e8B                                [the eight roots written as rows of coordinates]
    Matrix                             [a rectangle of numbers, indexed by row and column]
    ℝ                                  [the real numbers] -/
noncomputable def e8B : Matrix (Fin 8) (Fin 8) ℝ := fun i c => oCoord (e8Oct i) c

open Quaternion in
/-- **The `E₈` Cartan matrix is the Gram matrix of the octonion norm form on those roots.**

    (i c : Fin 8)                                                [two root indices]
    oForm (e8Oct i) (e8Oct c) = (CartanMatrix.E₈ i c : ℝ)        [the Gram matrix is Bourbaki's]
    Matrix                                       [a rectangle of numbers, indexed by row and column]
    ℝ                                                            [the real numbers]

    So: the definite block of the ladder is the octonion norm form on a lattice of octonions,
    computed rather than posited. What this does not say is that the integer span of `e8Oct` is
    closed under octonion multiplication: only the form is used, so this is the norm-form half of
    "`E₈` is the integral octonions" and not the ring half. -/
theorem e8_octonion_gram (i c : Fin 8) :
    oForm (e8Oct i) (e8Oct c) = (CartanMatrix.E₈ i c : ℝ) := by
  fin_cases i <;> fin_cases c <;>
    simp [oForm, oCoord, e8Oct, CartanMatrix.E₈, Fin.sum_univ_eight] <;> norm_num

/-- The rank-eight summand `E₈`, presented by Bourbaki's Cartan matrix.

    E8                                 [the rank-eight definite even form]
    IntForm                            [an integral symmetric bilinear form] -/
def E8 : IntForm (Fin 8) where
  gram := CartanMatrix.E₈
  gram_symm := CartanMatrix.E₈_isSymm

/-- The rank-eight form's Gram matrix is Mathlib's Cartan matrix.

    E8                                 [the rank-eight definite even form] -/
@[simp] theorem E8_gram : E8.gram = CartanMatrix.E₈ := rfl

/-- The entries of `B Bᵀ` are the pairings of the roots.

    (i c : Fin 8)                      [two root indices]
    e8B                                [the eight roots written as rows of coordinates]
    e8Oct                              [the eight simple roots, as octonions]
    oForm                              [the polarisation of the octonion norm form] -/
theorem e8B_mul_transpose (i c : Fin 8) : (e8B * e8Bᵀ) i c = oForm (e8Oct i) (e8Oct c) := by
  simp [Matrix.mul_apply, Matrix.transpose_apply, e8B, oForm]

/-- **`E₈` factors as `B Bᵀ` with `B` the octonion coordinate matrix.**

    E8.gramR = e8B * e8Bᵀ              [the factorisation everything definite rests on]
    E8                                 [the rank-eight definite even form]
    e8B                                [the eight roots written as rows of coordinates] -/
theorem e8_gramR_eq : E8.gramR = e8B * e8Bᵀ := by
  ext i c
  rw [e8B_mul_transpose, e8_octonion_gram]
  rfl

/-- An explicit integer inverse of the `E₈` Cartan matrix.

    e8GramInv                          [the integer inverse of the E8 Cartan matrix]
    Matrix                             [a rectangle of numbers, indexed by row and column] -/
def e8GramInv : Matrix (Fin 8) (Fin 8) ℤ :=
  !![ 4,  5,  7, 10,  8,  6,  4,  2;
      5,  8, 10, 15, 12,  9,  6,  3;
      7, 10, 14, 20, 16, 12,  8,  4;
     10, 15, 20, 30, 24, 18, 12,  6;
      8, 12, 16, 24, 20, 15, 10,  5;
      6,  9, 12, 18, 15, 12,  8,  4;
      4,  6,  8, 12, 10,  8,  6,  3;
      2,  3,  4,  6,  5,  4,  3,  2]

set_option maxRecDepth 8000 in
/-- The exhibited matrix is an inverse, checked in the kernel.

    e8GramInv                          [the integer inverse of the E8 Cartan matrix] -/
theorem e8Gram_mul_inv : CartanMatrix.E₈ * e8GramInv = 1 := by decide

/-- **The rank-eight form is unimodular.**

    E8                                 [the rank-eight definite even form]

    So: from the explicit integer inverse, with no determinant expansion. -/
theorem E8_isUnimodular : E8.IsUnimodular :=
  IntForm.isUnimodular_of_inverse E8 e8GramInv e8Gram_mul_inv

/-- The coordinate matrix is invertible.

    e8B                                [the eight roots written as rows of coordinates]
    ≠                                  [is not equal to] -/
theorem e8B_det_ne_zero : e8B.det ≠ 0 := by
  intro hz
  have hcast : ((E8.gram.det : ℤ) : ℝ) = e8B.det * e8B.det := by
    have hmap : ((E8.gram.det : ℤ) : ℝ) = E8.gramR.det := Int.cast_det E8.gram
    rw [hmap, e8_gramR_eq, Matrix.det_mul, Matrix.det_transpose]
  rw [hz, mul_zero] at hcast
  have hzero : E8.gram.det = 0 := by exact_mod_cast hcast
  rcases (IntForm.isUnimodular_iff E8).mp E8_isUnimodular with h | h <;> rw [h] at hzero <;>
    norm_num at hzero

/-- **`det E₈ = 1`.**

    E8.gram.det = 1                    [the determinant is one]
    E8                                 [the rank-eight definite even form]

    So: Mathlib carries this only as `proof_wanted E₈_det`. The explicit integer inverse gives
    `±1` and the factorisation in octonion coordinates gives `(det E₈ : ℝ) = (det B)² ≥ 0`, which
    rules out the other sign. -/
theorem E8_det : E8.gram.det = 1 :=
  IntForm.det_eq_one_of_gram_eq_mulTranspose e8_gramR_eq E8_isUnimodular

/-- **The rank-eight form is even.**

    E8                                 [the rank-eight definite even form]

    So: every root has norm two, which is the diagonal of the Cartan matrix. -/
theorem E8_isEven : E8.IsEven := by
  rw [IntForm.isEven_iff_even_diag]
  intro i
  rw [E8_gram, CartanMatrix.E₈_diag]
  exact ⟨1, rfl⟩

/-- **`E₈` is positive definite.**

    (x : Fin 8 → ℤ) (hx : x ≠ 0)       [a non-zero integer vector]
    0 < E8.bil x x                     [it pairs positively with itself]
    E8                                 [the rank-eight definite even form]
    ≠                                  [is not equal to]
    <                                  [is smaller than]
    →                                  [takes the left and gives the right] -/
theorem E8_posDef (x : Fin 8 → ℤ) (hx : x ≠ 0) : 0 < E8.bil x x :=
  IntForm.bil_pos_of_gram_eq_mulTranspose e8B_det_ne_zero e8_gramR_eq x hx

/-- **The rank-eight form is definite: its signature witness is all plus.**

    E8                                 [the rank-eight definite even form] -/
theorem E8_hasSignature : E8.HasSignature (fun _ => .positive) :=
  IntForm.hasSignature_one_of_gram_eq_mulTranspose e8B_det_ne_zero e8_gramR_eq

-- ═══════════════════════════════════════════════════════════════════
-- THE FAMILY
-- ═══════════════════════════════════════════════════════════════════

/-- The index type of `E₈^e ⊕ U^u`.

    (e u : ℕ)                          [two counts]
    EUIdx                              [the index set of E8 to the e sum U to the u]
    ×                                  [paired with, so ℝ × ℝ means two real numbers side by side]
    ℕ                                  [the counting numbers 0, 1, 2, 3, and so on] -/
abbrev EUIdx (e u : ℕ) : Type := (Fin 8 × Fin e) ⊕ (Fin 2 × Fin u)

/-- The family `E₈^e ⊕ U^u`.

    (e u : ℕ)                          [two counts]
    evenUnimodular                     [e copies of E8 and u copies of the hyperbolic plane]
    IntForm                            [an integral symmetric bilinear form]
    EUIdx                              [the index set of E8 to the e sum U to the u]
    ℕ                                  [the counting numbers] -/
def evenUnimodular (e u : ℕ) : IntForm (EUIdx e u) :=
  (E8.pow e).dsum (hyperbolicPlane.pow u)

/-- The family's sign vector: `8e` plus ones, then one plus and one minus per hyperbolic plane.

    (e u : ℕ)                          [two counts]
    euSign                             [the sign vector of the family]
    EUIdx                              [the index set of E8 to the e sum U to the u]
    →                                  [takes the left and gives the right]
    ℕ                                  [the counting numbers]
    IntForm.SignatureSign               [one of the two finite signs] -/
def euSign (e u : ℕ) : EUIdx e u → IntForm.SignatureSign :=
  Sum.elim (fun _ => .positive) (fun p => hSign p.1)

/-- **The family is even.**

    (e u : ℕ)                          [two counts]
    evenUnimodular                     [e copies of E8 and u copies of the hyperbolic plane]
    ℕ                                  [the counting numbers 0, 1, 2, 3, and so on] -/
theorem evenUnimodular_isEven (e u : ℕ) : (evenUnimodular e u).IsEven :=
  IntForm.dsum_isEven (IntForm.pow_isEven E8_isEven e)
    (IntForm.pow_isEven hyperbolicPlane_isEven u)

/-- Its determinant is minus one to the number of hyperbolic planes.

    (e u : ℕ)                          [two counts]
    evenUnimodular                     [e copies of E8 and u copies of the hyperbolic plane]
    ℕ                                  [the counting numbers] -/
theorem evenUnimodular_det (e u : ℕ) : (evenUnimodular e u).gram.det = (-1 : ℤ) ^ u := by
  rw [evenUnimodular, IntForm.dsum_det, IntForm.pow_det, IntForm.pow_det, E8_det,
    hyperbolicPlane_det, one_pow, _root_.one_mul]

/-- **The family is unimodular.**

    (e u : ℕ)                          [two counts]
    evenUnimodular                     [e copies of E8 and u copies of the hyperbolic plane]
    ℕ                                  [the counting numbers] -/
theorem evenUnimodular_isUnimodular (e u : ℕ) : (evenUnimodular e u).IsUnimodular :=
  IntForm.dsum_isUnimodular (IntForm.pow_isUnimodular E8_isUnimodular e)
    (IntForm.pow_isUnimodular hyperbolicPlane_isUnimodular u)

/-- **The family carries the signature witness `(8e + u, u)`.**

    (e u : ℕ)                                              [two counts]
    (evenUnimodular e u).HasSignature (euSign e u)         [the sign vector is a witness]
    evenUnimodular                             [e copies of E8 and u copies of the hyperbolic plane]
    euSign                                                 [the sign vector of the family]
    ℕ                                                      [the counting numbers]

    So: the change of basis is block diagonal, definite on each `E₈` block and light-cone on each
    hyperbolic plane. -/
theorem evenUnimodular_hasSignature (e u : ℕ) :
    (evenUnimodular e u).HasSignature (euSign e u) :=
  IntForm.dsum_hasSignature (IntForm.pow_hasSignature E8_hasSignature e)
    (IntForm.pow_hasSignature hyperbolicPlane_hasSignature u)

/-- The family has `8e + u` plus signs.

    (e u : ℕ)                          [two counts]
    sigPos                             [how many plus ones a sign vector has]
    euSign                             [the sign vector of the family]
    IntForm                            [an integral symmetric bilinear form]
    ℕ                                  [the counting numbers] -/
@[simp] theorem euSign_sigPos (e u : ℕ) : IntForm.sigPos (euSign e u) = 8 * e + u := by
  rw [euSign, IntForm.sigPos_sumElim, IntForm.sigPos_const_one, IntForm.sigPos_pow,
    hSign_sigPos, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, mul_one]

/-- The family has `u` minus signs.

    (e u : ℕ)                          [two counts]
    sigNeg                             [how many entries are not plus one]
    euSign                             [the sign vector of the family]
    IntForm                            [an integral symmetric bilinear form]
    ℕ                                  [the counting numbers] -/
@[simp] theorem euSign_sigNeg (e u : ℕ) : IntForm.sigNeg (euSign e u) = u := by
  rw [euSign, IntForm.sigNeg_sumElim, IntForm.sigNeg_const_one, IntForm.sigNeg_pow,
    hSign_sigNeg, mul_one, zero_add]

-- The finite signature evaluator checks the hyperbolic and two rank-twenty-six fixtures.
#guard IntForm.sigPos hSign == 1
#guard IntForm.sigNeg hSign == 1
#guard IntForm.sigPos (euSign 3 1) == 25
#guard IntForm.sigNeg (euSign 3 1) == 1
#guard IntForm.sigPos (euSign 2 5) == 21
#guard IntForm.sigNeg (euSign 2 5) == 5

/-- **On this family the signature difference is `8e`.**

    (e u : ℕ)                                                        [two counts]
    IntForm.sigPos (euSign e u) = IntForm.sigNeg (euSign e u) + 8 * e
    sigPos                                                    [how many plus ones a sign vector has]
    sigNeg                                                       [how many entries are not plus one]
    euSign                                                           [the sign vector of the family]
    IntForm                                                  [an integral symmetric bilinear form]
    ℕ                                                                [the counting numbers]

    So: each hyperbolic plane contributes one plus and one minus and cancels out of the
    difference, leaving the definite blocks eight at a time. This is the existence direction and
    is stated for this family alone. -/
theorem signature_difference (e u : ℕ) :
    IntForm.sigPos (euSign e u) = IntForm.sigNeg (euSign e u) + 8 * e := by
  rw [euSign_sigPos, euSign_sigNeg, Nat.add_comm]

/-- **Eight divides the signature difference on this family.**

    (e u : ℕ)                                                        [two counts]
    (8 : ℤ) ∣ ((IntForm.sigPos (euSign e u) : ℤ) - (IntForm.sigNeg (euSign e u) : ℤ))
    sigPos                                                    [how many plus ones a sign vector has]
    sigNeg                                                       [how many entries are not plus one]
    euSign                                                           [the sign vector of the family]
    IntForm                                                  [an integral symmetric bilinear form]
    ℕ                                                                [the counting numbers]

    So: the converse, that every even unimodular form satisfies this, is van der Blij's
    congruence. It is cited in the header and proved nowhere in this corpus. -/
theorem eight_dvd_signature_difference (e u : ℕ) :
    (8 : ℤ) ∣ ((IntForm.sigPos (euSign e u) : ℤ) - (IntForm.sigNeg (euSign e u) : ℤ)) := by
  rw [euSign_sigPos, euSign_sigNeg]
  exact ⟨e, by push_cast; ring⟩

/-- **The rank is the ladder's dimension count.**

    (e u : ℕ)                                                  [two counts]
    Fintype.card (EUIdx e u) = Physics.dimOf e u               [the rank is 8e + 2u]
    EUIdx                                              [the index set of E8 to the e sum U to the u]
    dimOf                  [how many real directions so many definite blocks and so many cells make]
    ℕ                                                          [the counting numbers]

    So: `Apeiron.Physics.Ladder`'s arithmetic and this lattice are one object counted twice. -/
theorem euRank_eq_dimOf (e u : ℕ) : Fintype.card (EUIdx e u) = Physics.dimOf e u := by
  simp [EUIdx, Physics.dimOf, Fintype.card_sum, Fintype.card_prod]

-- ═══════════════════════════════════════════════════════════════════
-- RANK 26
-- ═══════════════════════════════════════════════════════════════════

/-- **`E₈³ ⊕ U` is even, unimodular, of rank 26, with signature witness `(25, 1)`.**

    (evenUnimodular 3 1).IsEven                                      [even]
    ∧ (evenUnimodular 3 1).IsUnimodular                              [unimodular]
    ∧ (evenUnimodular 3 1).HasSignature (euSign 3 1)                 [with that witness]
    ∧ Fintype.card (EUIdx 3 1) = 26                                  [of rank 26]
    ∧ IntForm.sigPos (euSign 3 1) = 25                               [25 plus signs]
    ∧ IntForm.sigNeg (euSign 3 1) = 1                                [one minus sign]
    evenUnimodular                             [e copies of E8 and u copies of the hyperbolic plane]
    euSign                                                           [the sign vector of the family]
    EUIdx                                                            [the index set]
    sigPos                                                           [how many plus ones]
    sigNeg                                                       [how many entries are not plus one]
    ∧                                                                [both are true]

    So: rank 26 with one timelike direction is realised by a constructed lattice. Rank alone does
    not force that signature: `E₈² ⊕ U⁵` has rank 26 and signature `(21, 5)` by the same
    construction. -/
theorem bosonic_twentysix :
    (evenUnimodular 3 1).IsEven
      ∧ (evenUnimodular 3 1).IsUnimodular
      ∧ (evenUnimodular 3 1).HasSignature (euSign 3 1)
      ∧ Fintype.card (EUIdx 3 1) = 26
      ∧ IntForm.sigPos (euSign 3 1) = 25
      ∧ IntForm.sigNeg (euSign 3 1) = 1 := by
  refine ⟨evenUnimodular_isEven 3 1, evenUnimodular_isUnimodular 3 1,
    evenUnimodular_hasSignature 3 1, ?_, ?_, ?_⟩
  · rw [euRank_eq_dimOf]; decide
  · rw [euSign_sigPos]
  · rw [euSign_sigNeg]

/-- **Rank 26 also carries the signature `(21, 5)`.**

    Fintype.card (EUIdx 2 5) = 26                          [also rank 26]
    ∧ IntForm.sigPos (euSign 2 5) = 21                     [21 plus signs]
    ∧ IntForm.sigNeg (euSign 2 5) = 5                      [five minus signs]
    EUIdx                                                  [the index set]
    euSign                                                 [the sign vector of the family]
    sigPos                                                 [how many plus ones]
    sigNeg                                                 [how many entries are not plus one]
    ∧                                                      [both are true]

    So: the rank fixes the number of hyperbolic planes only modulo four, so selecting one timelike
    direction is an input and not a consequence. This is stated so that no reader takes rank 26 to
    mean signature `(25, 1)`. -/
theorem twentysix_also_carries_five_minus_signs :
    Fintype.card (EUIdx 2 5) = 26
      ∧ IntForm.sigPos (euSign 2 5) = 21
      ∧ IntForm.sigNeg (euSign 2 5) = 5 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [euRank_eq_dimOf]; decide
  · rw [euSign_sigPos]
  · rw [euSign_sigNeg]

/-- **The square pyramid at 24 is a square.**

    ∑ k ∈ Finset.range 25, k ^ 2 = 70 ^ 2       [one squared through 24 squared is 70 squared]
    ∈                                           [is one of the things in] -/
theorem cannonball : ∑ k ∈ Finset.range 25, k ^ 2 = 70 ^ 2 := by
  norm_num [Finset.sum_range_succ]

/-- A kernel-checkable square test: some `m` at most `b` has `m * m = S`.

    (b S : ℕ)                          [a bound and a number]
    isSqUpTo                           [a kernel-checkable square test]
    : Bool                             [true or false, nothing else]
    Bool                               [true or false, nothing else]
    ℕ                                  [the counting numbers 0, 1, 2, 3, and so on] -/
def isSqUpTo (b S : ℕ) : Bool := (List.range (b + 1)).any (fun m => m * m == S)

/-- A bounded non-square certificate.

    {S : ℕ} (hb : S ≤ 9455) (hc : isSqUpTo 97 S = false)             [no root at most 97]
    ¬ IsSquare S                                                     [S is not a square]
    isSqUpTo                                                        [a kernel-checkable square test]
    ¬                                                                [this is not true]
    ≤                                                                [is smaller than or equal to]
    ℕ                                                                [the counting numbers]

    So: `97² = 9409` and `98² = 9604`, so any square at most `9455` has a root at most `97`. The
    test avoids `Nat.sqrt`, which does not reduce in the kernel. -/
theorem not_isSquare_of_check {S : ℕ} (hb : S ≤ 9455) (hc : isSqUpTo 97 S = false) :
    ¬ IsSquare S := by
  rintro ⟨r, rfl⟩
  have hr : r ≤ 97 := by
    by_contra h
    have h98 : 98 ≤ r := by omega
    have hmul : 98 * 98 ≤ r * r := Nat.mul_le_mul h98 h98
    omega
  have hmem : isSqUpTo 97 (r * r) = true := by
    unfold isSqUpTo
    rw [List.any_eq_true]
    exact ⟨r, by rw [List.mem_range]; omega, by simp⟩
  rw [hmem] at hc
  exact Bool.noConfusion hc

/-- **Over `2 ≤ n ≤ 30` the square pyramid is a square only at `n = 24`.**

    ∀ n, 2 ≤ n → n ≤ 30 → IsSquare (∑ i ∈ Finset.range (n + 1), i ^ 2) → n = 24
    ∀                                                                [for every single one]
    ≤                                                                [is smaller than or equal to]
    →                                                           [takes the left and gives the right]
    ∈                                                                [is one of the things in]

    So: the window is checked case by case with the kernel square test. That the window can be
    removed, and 24 is the only solution above one over all `n`, is Watson 1918: it needs the
    rational points of a genus-one curve, and Mathlib carries no machinery for that. -/
theorem local_cannonball_uniqueness :
    ∀ n, 2 ≤ n → n ≤ 30 → IsSquare (∑ i ∈ Finset.range (n + 1), i ^ 2) → n = 24 := by
  intro n h2 h30 hsq
  interval_cases n <;>
    first
      | rfl
      | (exact absurd hsq (not_isSquare_of_check (by norm_num) (by decide)))

/-- **What selects 26 inside the column `10, 18, 26`.**

    (e : ℕ)                                                          [a count of definite blocks]
    (hlow : 2 ≤ Physics.dimOf e 0) (hhigh : Physics.dimOf e 0 ≤ 30)  [the block lies in the window]
    (hsq : IsSquare (∑ i ∈ Finset.range (Physics.dimOf e 0 + 1), i ^ 2))
    Physics.dimOf e 1 = 26                                           [the one-cell member is 26]
    dimOf                  [how many real directions so many definite blocks and so many cells make]
    ∈                                                                [is one of the things in]
    ≤                                                                [is smaller than or equal to]
    ℕ                                                                [the counting numbers]

    So: of the members `10, 18, 26` of the one-cell column, exactly the one whose definite block
    is a height at which the square pyramid is a perfect square, inside the checked window. The
    definite block is then `24` and the dimension is `24 + 2`. Outside the window the selection
    rests on Watson 1918, cited. Nothing here says a physical theory must sit in this column: the
    one timelike direction is an input. -/
theorem cannonball_selects_twentysix (e : ℕ) (hlow : 2 ≤ Physics.dimOf e 0)
    (hhigh : Physics.dimOf e 0 ≤ 30)
    (hsq : IsSquare (∑ i ∈ Finset.range (Physics.dimOf e 0 + 1), i ^ 2)) :
    Physics.dimOf e 1 = 26 := by
  have h24 := local_cannonball_uniqueness (Physics.dimOf e 0) hlow hhigh hsq
  unfold Physics.dimOf at h24 ⊢
  omega

end Lattice
end Apeiron
