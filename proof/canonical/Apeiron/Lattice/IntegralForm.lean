import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Mathlib.Data.Matrix.Block
import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Symmetric

/-!
# Integral symmetric bilinear forms: even, unimodular, signature

@name IntegralForm
@cite milnor-husemoller-1973
@cite conway-sloane-1999
@grade proved

A symmetric matrix `G : Matrix ι ι ℤ` presents an integral symmetric bilinear form
`b(x, y) = xᵀ G y` on the free `ℤ`-module `ι → ℤ`. Three predicates on such a form are needed
downstream:

* **even**: `b(x, x) ∈ 2ℤ` for every `x`. `isEven_iff_even_diag` reduces this to the diagonal,
  since symmetry makes every off-diagonal term appear twice.
* **unimodular**: `det G` is a unit of `ℤ`, so `det G = ±1`.
* **signature `s`**: a real change of basis `P` with `Pᵀ G P = diag s` and every `s i = ±1`.
  This is the existence half of Sylvester's law. `sigPos` and `sigNeg` count the signs of a given
  witness. Uniqueness of the counts, Sylvester's law of inertia proper, is not proved here and no
  statement downstream may read `sigPos` as an invariant of the form.

Two constructions close over all three: the orthogonal direct sum `dsum`, a block-diagonal Gram
matrix on `ι ⊕ κ`, and the `k`-fold power `pow`. Determinants multiply, sign vectors concatenate,
and evenness is inherited from the diagonal.

The positive definite case is reached through the factorisation `G = B Bᵀ` over `ℝ` with
`det B ≠ 0`: then `P = (B⁻¹)ᵀ` gives `Pᵀ G P = 1`, the signature is all plus, `det G = (det B)²`
is positive so a unimodular `G` has `det G = 1`, and `xᵀ G x = ‖Bᵀx‖² > 0` for `x ≠ 0`. Gram
matrices of real bases are exactly the matrices of this shape, so no minor criterion is needed.

## What Mathlib carries

As of the 2026-09-02 sweep, nothing here. `Matrix.IsTotallyUnimodular` is the condition that every
square submatrix has determinant in `{0, ±1}`, which is a combinatorial notion and unrelated.
`Mathlib/Algebra/Module/ZLattice/` carries bases and covolume and no evenness, unimodularity or
signature. `grep -rli "even unimodular" Mathlib/` returns nothing.

## Root

| from | here |
|---|---|
| nothing | nothing |

**This file instantiates no theorem of `Apeiron/Trichotomy`, and that is stated rather than
implied.** Every declaration here is about a symmetric integer matrix and a real change of basis.
σ does not occur in any statement and could not: by `theory/NOVELTY.md` test A1 this is
linear algebra, honest and useful and not a result about `ℝ[x]/(x² − σ)`. It is here because
`Apeiron/Lattice/EvenUnimodular.lean` needs it and Mathlib carries none of it.

The edge is made one file downstream and in two places, both of which would break if this file's
`IntForm` were deleted: `EvenUnimodular.hyperbolicPlane_gram_is_the_null_pair` shows the rank-two
summand's Gram matrix is the matrix of `Trichotomy.form 1` on the corpus's own null basis, and
`EvenUnimodular.e8_octonion_gram` shows the rank-eight summand's is the polarisation of
`Doubling.normO`, which reaches the trichotomy through `Doubling.real_mul`.

The primitive that would let this file be rooted directly is a quadratic form over `Cell σ` of
rank above two, which the corpus does not have: `Trichotomy.normForm` is rank two by
construction and `Trichotomy.classification` is the theorem that says so. Not built. Named.
-/

namespace Apeiron
namespace Lattice

open Matrix

/-- An integral symmetric bilinear form, presented by its Gram matrix on a finite free
    `ℤ`-module with basis indexed by `ι`.

    (ι : Type*) [Fintype ι] [DecidableEq ι]    [a finite index set for the basis]
    IntForm                                    [an integral symmetric bilinear form]
    form                                       [an integral symmetric bilinear form]
    Matrix                                     [a rectangle of numbers, indexed by row and column]
    →                                          [takes the left and gives the right] -/
structure IntForm (ι : Type*) [Fintype ι] [DecidableEq ι] where
  /-- The Gram matrix: entry `i c` is the pairing of basis vector `i` with basis vector `c`. -/
  gram : Matrix ι ι ℤ
  /-- The Gram matrix is symmetric. -/
  gram_symm : gram.IsSymm

namespace IntForm

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ]

omit [DecidableEq ι] in
/-- The bilinear form read off a matrix as a double sum.

    {R : Type*} [CommRing R]                            [any commutative ring]
    (G : Matrix ι ι R) (x y : ι → R)                    [a matrix and two vectors]
    x ⬝ᵥ (G *ᵥ y) = ∑ i, ∑ c, x i * G i c * y c         [the matrix product as a double sum]
    Matrix                                       [a rectangle of numbers, indexed by row and column]
    →                                                   [takes the left and gives the right] -/
theorem dot_mulVec_eq_double_sum {R : Type*} [CommRing R] (G : Matrix ι ι R) (x y : ι → R) :
    x ⬝ᵥ (G *ᵥ y) = ∑ i, ∑ c, x i * G i c * y c := by
  simp only [dotProduct, mulVec, Finset.mul_sum, mul_assoc]

/-- The form itself: `b(x, y) = xᵀ G y`.

    (L : IntForm ι) (x y : ι → ℤ)      [a form and two integer vectors]
    : ℤ                                [an integer]
    IntForm                            [an integral symmetric bilinear form]
    →                                  [takes the left and gives the right] -/
def bil (L : IntForm ι) (x y : ι → ℤ) : ℤ := x ⬝ᵥ (L.gram *ᵥ y)

/-- The form written as a double sum.

    (L : IntForm ι) (x y : ι → ℤ)                               [a form and two vectors]
    L.bil x y = ∑ i, ∑ c, x i * L.gram i c * y c                [the same sum as above]
    IntForm                                                    [an integral symmetric bilinear form]
    →                                                        [takes the left and gives the right] -/
theorem bil_eq_double_sum (L : IntForm ι) (x y : ι → ℤ) :
    L.bil x y = ∑ i, ∑ c, x i * L.gram i c * y c :=
  dot_mulVec_eq_double_sum _ _ _

/-- The Gram matrix with its entries read as real numbers.

    (L : IntForm ι)                    [a form]
    : Matrix ι ι ℝ                     [its Gram matrix over the reals]
    IntForm                            [an integral symmetric bilinear form]
    Matrix                             [a rectangle of numbers, indexed by row and column]
    ℝ                                  [the real numbers: every number on the number line] -/
def gramR (L : IntForm ι) : Matrix ι ι ℝ := L.gram.map (fun z => (z : ℝ))

/-- The real Gram matrix has the cast entries.

    (L : IntForm ι) (i c : ι)          [a form and two indices]
    IntForm                            [an integral symmetric bilinear form]
    ℝ                                  [the real numbers] -/
@[simp] theorem gramR_apply (L : IntForm ι) (i c : ι) : L.gramR i c = (L.gram i c : ℝ) := rfl

/-- The form is even: every vector pairs with itself to an even integer.

    (L : IntForm ι)                    [a form]
    : Prop                             [a statement that is true or false]
    IntForm                            [an integral symmetric bilinear form]
    ∀                                  [for every single one]
    →                                  [takes the left and gives the right] -/
def IsEven (L : IntForm ι) : Prop := ∀ x : ι → ℤ, Even (L.bil x x)

omit [Fintype ι] in
/-- With a symmetric Gram matrix and an even diagonal, the double sum over any finite index set
    is even: the off-diagonal terms pair up, so they contribute in twos.

    (G : Matrix ι ι ℤ)                 [a Gram matrix]
    Matrix                             [a rectangle of numbers, indexed by row and column]
    ∀                                  [for every single one]
    →                                  [takes the left and gives the right]
    ∈                                  [is one of the things in] -/
private theorem even_double_sum (G : Matrix ι ι ℤ) (hsym : G.IsSymm)
    (hdiag : ∀ i, Even (G i i)) (x : ι → ℤ) (s : Finset ι) :
    Even (∑ i ∈ s, ∑ c ∈ s, x i * G i c * x c) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
    have key : ∑ i ∈ insert a s, ∑ c ∈ insert a s, x i * G i c * x c
        = x a * G a a * x a
          + 2 * (∑ c ∈ s, x a * G a c * x c)
          + ∑ i ∈ s, ∑ c ∈ s, x i * G i c * x c := by
      rw [Finset.sum_insert ha]
      simp_rw [Finset.sum_insert ha]
      rw [Finset.sum_add_distrib]
      have hmirror : ∑ i ∈ s, x i * G i a * x a = ∑ c ∈ s, x a * G a c * x c :=
        Finset.sum_congr rfl fun i _ => by rw [hsym.apply a i]; ring
      rw [hmirror]; ring
    rw [key]
    exact (((hdiag a).mul_left (x a)).mul_right (x a)).add ⟨_, by ring⟩ |>.add ih

/-- **A form is even exactly when its diagonal is.**

    (L : IntForm ι)                                   [a form]
    L.IsEven ↔ ∀ i, Even (L.gram i i)                 [even exactly when every diagonal entry is]
    IntForm                                           [an integral symmetric bilinear form]
    ∀                                                 [for every single one]
    ↔                                                 [exactly when, true in both directions]

    So: symmetry pairs the term `xᵢ Gᵢⱼ xⱼ` with `xⱼ Gⱼᵢ xᵢ`, so every off-diagonal contribution
    is doubled and only the diagonal can obstruct evenness. -/
theorem isEven_iff_even_diag (L : IntForm ι) : L.IsEven ↔ ∀ i, Even (L.gram i i) := by
  constructor
  · intro h i
    have hb := h (Pi.single i (1 : ℤ))
    rw [bil_eq_double_sum] at hb
    have hs : ∑ i' : ι, ∑ c : ι,
        (Pi.single i (1 : ℤ) : ι → ℤ) i' * L.gram i' c * (Pi.single i (1 : ℤ) : ι → ℤ) c
          = L.gram i i := by
      simp [Pi.single_apply, ite_mul, mul_ite, Finset.sum_ite_eq']
    rwa [hs] at hb
  · intro h x
    rw [bil_eq_double_sum]
    exact even_double_sum L.gram L.gram_symm h x Finset.univ

/-- The form is unimodular: its Gram determinant is a unit of `ℤ`.

    (L : IntForm ι)                    [a form]
    : Prop                             [a statement that is true or false]
    IntForm                            [an integral symmetric bilinear form] -/
def IsUnimodular (L : IntForm ι) : Prop := IsUnit L.gram.det

/-- Unimodular means determinant plus or minus one.

    (L : IntForm ι)                                                 [a form]
    L.IsUnimodular ↔ L.gram.det = 1 ∨ L.gram.det = -1               [a unit of the integers]
    IntForm                                                    [an integral symmetric bilinear form]
    ↔                                                        [exactly when, true in both directions]
    ∨                                                               [at least one is true] -/
theorem isUnimodular_iff (L : IntForm ι) : L.IsUnimodular ↔ L.gram.det = 1 ∨ L.gram.det = -1 :=
  Int.isUnit_iff

/-- **An explicit integer inverse certifies unimodularity.**

    (L : IntForm ι) (H : Matrix ι ι ℤ) (h : L.gram * H = 1)        [an integer right inverse]
    L.IsUnimodular                                                 [the form is unimodular]
    IntForm                                                    [an integral symmetric bilinear form]
    Matrix                                       [a rectangle of numbers, indexed by row and column]

    So: no determinant expansion is needed. Exhibiting the inverse over `ℤ` gives
    `det G · det H = 1` in `ℤ`. -/
theorem isUnimodular_of_inverse (L : IntForm ι) (H : Matrix ι ι ℤ) (h : L.gram * H = 1) :
    L.IsUnimodular := by
  have hd : L.gram.det * H.det = 1 := by rw [← Matrix.det_mul, h, Matrix.det_one]
  exact ⟨⟨L.gram.det, H.det, hd, by rw [mul_comm]; exact hd⟩, rfl⟩

/-- The two structural signs used by a finite signature witness. -/
inductive SignatureSign where
  | positive
  | negative
  deriving DecidableEq, Repr

/-- The real diagonal entry denoted by a structural sign.

    (s : SignatureSign)                  [a positive or negative structural sign]
    : ℝ                                  [the corresponding real diagonal entry]
    SignatureSign                        [one of the two finite signs]
    →                                    [takes the left and gives the right]
    ℝ                                    [the real numbers] -/
def SignatureSign.toReal : SignatureSign → ℝ
  | .positive => 1
  | .negative => -1

/-- Every structural sign denotes one of the two signature scalars.

    (s : SignatureSign)                  [a positive or negative structural sign]
    s.toReal = 1 ∨ s.toReal = -1         [its real image is plus or minus one]
    SignatureSign                        [one of the two finite signs]
    ∨                                    [at least one is true]
    ℝ                                    [the real numbers] -/
theorem SignatureSign.toReal_eq_one_or_neg_one (s : SignatureSign) :
    s.toReal = 1 ∨ s.toReal = -1 := by
  cases s <;> simp [SignatureSign.toReal]

/-- A signature witness: a real change of basis diagonalising the form to structural signs.

    (L : IntForm ι) (s : ι → SignatureSign) [a form and a finite sign vector]
    : Prop                             [a statement that is true or false]
    IntForm                            [an integral symmetric bilinear form]
    Matrix                             [a rectangle of numbers, indexed by row and column]
    ∃                                  [there is at least one]
    ∧                                  [both are true]
    ≠                                  [is not equal to]
    →                                  [takes the left and gives the right]
    ℝ                                  [the real numbers]

    So: this is existence only. Sylvester's law of inertia, that the counts of plus and minus
    signs do not depend on the witness, is not proved in this corpus, and nothing below reads
    `sigPos` as an invariant of the form. -/
def HasSignature (L : IntForm ι) (s : ι → SignatureSign) : Prop :=
  ∃ P : Matrix ι ι ℝ, P.det ≠ 0 ∧ Pᵀ * L.gramR * P = Matrix.diagonal (fun i => (s i).toReal)

/-- The number of plus ones in a sign vector.

    (s : ι → SignatureSign)            [a finite sign vector]
    : ℕ                                [a counting number]
    sigPos                             [how many plus ones a sign vector has]
    →                                  [takes the left and gives the right]
    ℕ                                  [the counting numbers 0, 1, 2, 3, and so on]
    SignatureSign                      [one of the two finite signs] -/
def sigPos (s : ι → SignatureSign) : ℕ := ∑ i, match s i with | .positive => 1 | .negative => 0

/-- The number of entries that are not plus one.

    (s : ι → SignatureSign)            [a finite sign vector]
    : ℕ                                [a counting number]
    sigNeg                             [how many entries of a sign vector are not plus one]
    →                                  [takes the left and gives the right]
    ℕ                                  [the counting numbers 0, 1, 2, 3, and so on]
    SignatureSign                      [one of the two finite signs] -/
def sigNeg (s : ι → SignatureSign) : ℕ := ∑ i, match s i with | .positive => 0 | .negative => 1

omit [DecidableEq ι] in
/-- The two counts add to the rank.

    (s : ι → SignatureSign)                            [a finite sign vector]
    sigPos s + sigNeg s = Fintype.card ι               [the two counts exhaust the index set]
    sigPos                                             [how many plus ones a sign vector has]
    sigNeg                                             [how many entries are not plus one]
    →                                                  [takes the left and gives the right]
    SignatureSign                                      [one of the two finite signs] -/
theorem sigPos_add_sigNeg (s : ι → SignatureSign) : sigPos s + sigNeg s = Fintype.card ι := by
  rw [sigPos, sigNeg, ← Finset.sum_add_distrib, Fintype.card, Finset.card_eq_sum_ones]
  exact Finset.sum_congr rfl fun i _ => by cases s i <;> rfl

omit [DecidableEq ι] in
/-- **`sigNeg` counts the structural negative entries.**

    (s : ι → SignatureSign)                                  [a finite sign vector]
    sigNeg s = ∑ i, if s i = .negative then 1 else 0         [it counts the negative entries]
    sigNeg                                                  [how many entries are not plus one]
    ∀                                                       [for every single one]
    →                                                       [takes the left and gives the right]
    SignatureSign                                           [one of the two finite signs]

    So: the definition counts everything that is not plus one, whatever it is. It counts minus
    ones by construction. -/
theorem sigNeg_eq_count_neg_one (s : ι → SignatureSign) :
    sigNeg s = ∑ i, if s i = .negative then 1 else 0 := by
  rw [sigNeg]
  exact Finset.sum_congr rfl fun i _ => by cases s i <;> rfl

omit [DecidableEq ι] in
/-- An all-plus sign vector has as many plus ones as there are indices.

    sigPos                             [how many plus ones a sign vector has]
    SignatureSign                      [one of the two finite signs] -/
@[simp] theorem sigPos_const_one : sigPos (fun _ : ι => .positive) = Fintype.card ι := by
  rw [sigPos, Fintype.card, Finset.card_eq_sum_ones]

omit [DecidableEq ι] in
/-- An all-plus sign vector has no other entries.

    sigNeg                             [how many entries are not plus one]
    SignatureSign                      [one of the two finite signs] -/
@[simp] theorem sigNeg_const_one : sigNeg (fun _ : ι => .positive) = 0 := by
  rw [sigNeg]
  exact Finset.sum_eq_zero fun _ _ => rfl

/-- The orthogonal direct sum: a block-diagonal Gram matrix on the disjoint union of the indices.

    (L : IntForm ι) (M : IntForm κ)    [two forms]
    : IntForm (ι ⊕ κ)                  [their orthogonal direct sum]
    IntForm                            [an integral symmetric bilinear form] -/
def dsum (L : IntForm ι) (M : IntForm κ) : IntForm (ι ⊕ κ) where
  gram := Matrix.fromBlocks L.gram 0 0 M.gram
  gram_symm := by
    have hL : L.gramᵀ = L.gram := L.gram_symm
    have hM : M.gramᵀ = M.gram := M.gram_symm
    show (Matrix.fromBlocks L.gram 0 0 M.gram)ᵀ = _
    rw [Matrix.fromBlocks_transpose, hL, hM]
    simp

/-- The Gram matrix of an orthogonal direct sum is block diagonal.

    (L : IntForm ι) (M : IntForm κ)    [two forms]
    IntForm                            [an integral symmetric bilinear form]
    Matrix                             [a rectangle of numbers, indexed by row and column] -/
@[simp] theorem dsum_gram (L : IntForm ι) (M : IntForm κ) :
    (L.dsum M).gram = Matrix.fromBlocks L.gram 0 0 M.gram := rfl

/-- The same over the reals.

    (L : IntForm ι) (M : IntForm κ)    [two forms]
    IntForm                            [an integral symmetric bilinear form]
    Matrix                             [a rectangle of numbers, indexed by row and column] -/
theorem dsum_gramR (L : IntForm ι) (M : IntForm κ) :
    (L.dsum M).gramR = Matrix.fromBlocks L.gramR 0 0 M.gramR := by
  simp [gramR, dsum, Matrix.fromBlocks_map]

/-- **An orthogonal direct sum of even forms is even.**

    {L : IntForm ι} {M : IntForm κ} (hL : L.IsEven) (hM : M.IsEven)   [two even forms]
    (L.dsum M).IsEven                                                 [their sum is even]
    IntForm                                                    [an integral symmetric bilinear form]
    dsum                                                 [the orthogonal direct sum of two forms] -/
theorem dsum_isEven {L : IntForm ι} {M : IntForm κ} (hL : L.IsEven) (hM : M.IsEven) :
    (L.dsum M).IsEven := by
  rw [isEven_iff_even_diag] at hL hM ⊢
  rintro (i | c) <;> simp [hL, hM]

/-- **Determinants multiply across an orthogonal direct sum.**

    (L : IntForm ι) (M : IntForm κ)                                   [two forms]
    (L.dsum M).gram.det = L.gram.det * M.gram.det                     [the determinants multiply]
    IntForm                                                    [an integral symmetric bilinear form]
    dsum                                                 [the orthogonal direct sum of two forms] -/
@[simp] theorem dsum_det (L : IntForm ι) (M : IntForm κ) :
    (L.dsum M).gram.det = L.gram.det * M.gram.det := by
  simp [dsum, Matrix.det_fromBlocks_zero₂₁]

/-- **An orthogonal direct sum of unimodular forms is unimodular.**

    {L : IntForm ι} {M : IntForm κ}                                   [two forms]
    (hL : L.IsUnimodular) (hM : M.IsUnimodular)                       [both unimodular]
    (L.dsum M).IsUnimodular                                           [so is their sum]
    IntForm                                                    [an integral symmetric bilinear form]
    dsum                                                 [the orthogonal direct sum of two forms] -/
theorem dsum_isUnimodular {L : IntForm ι} {M : IntForm κ}
    (hL : L.IsUnimodular) (hM : M.IsUnimodular) : (L.dsum M).IsUnimodular := by
  rw [IsUnimodular, dsum_det]; exact hL.mul hM

/-- **Signature witnesses concatenate across an orthogonal direct sum.**

    {L : IntForm ι} {M : IntForm κ} {s : ι → SignatureSign} {t : κ → SignatureSign}
                                                                  [forms and finite sign vectors]
    (hL : L.HasSignature s) (hM : M.HasSignature t)                  [each diagonalises]
    (L.dsum M).HasSignature (Sum.elim s t)            [so does the sum, with the concatenated signs]
    IntForm                                                    [an integral symmetric bilinear form]
    dsum                                                             [the orthogonal direct sum]
    →                                                           [takes the left and gives the right]
    SignatureSign                                                    [one of the two finite signs]

    So: the change of basis is block-diagonal. -/
theorem dsum_hasSignature {L : IntForm ι} {M : IntForm κ}
    {s : ι → SignatureSign} {t : κ → SignatureSign}
    (hL : L.HasSignature s) (hM : M.HasSignature t) :
    (L.dsum M).HasSignature (Sum.elim s t) := by
  obtain ⟨P, hP, hPeq⟩ := hL
  obtain ⟨Q, hQ, hQeq⟩ := hM
  refine ⟨Matrix.fromBlocks P 0 0 Q, ?_, ?_⟩
  · rw [Matrix.det_fromBlocks_zero₂₁]
    exact mul_ne_zero hP hQ
  · rw [dsum_gramR, Matrix.fromBlocks_transpose]
    simp only [Matrix.transpose_zero, Matrix.fromBlocks_multiply, Matrix.mul_zero,
      Matrix.zero_mul, add_zero, zero_add, hPeq, hQeq]
    rw [Matrix.fromBlocks_diagonal]
    congr 1
    funext i
    cases i <;> rfl

omit [DecidableEq ι] [DecidableEq κ] in
/-- Plus signs add across a concatenation.

    (s : ι → SignatureSign) (t : κ → SignatureSign) [two finite sign vectors]
    sigPos                             [how many plus ones a sign vector has]
    →                                  [takes the left and gives the right]
    SignatureSign                      [one of the two finite signs] -/
@[simp] theorem sigPos_sumElim (s : ι → SignatureSign) (t : κ → SignatureSign) :
    sigPos (Sum.elim s t) = sigPos s + sigPos t := by
  simp only [sigPos, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr]

omit [DecidableEq ι] [DecidableEq κ] in
/-- The other entries add across a concatenation.

    (s : ι → SignatureSign) (t : κ → SignatureSign) [two finite sign vectors]
    sigNeg                             [how many entries are not plus one]
    →                                  [takes the left and gives the right]
    SignatureSign                      [one of the two finite signs] -/
@[simp] theorem sigNeg_sumElim (s : ι → SignatureSign) (t : κ → SignatureSign) :
    sigNeg (Sum.elim s t) = sigNeg s + sigNeg t := by
  simp only [sigNeg, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr]

/-- `k` copies of a form: a block-diagonal Gram matrix indexed by `ι × Fin k`.

    (L : IntForm ι) (k : ℕ)            [a form and a count]
    : IntForm (ι × Fin k)              [k orthogonal copies of it]
    IntForm                            [an integral symmetric bilinear form]
    ×                                  [paired with, so ℝ × ℝ means two real numbers side by side]
    ℕ                                  [the counting numbers 0, 1, 2, 3, and so on] -/
def pow (L : IntForm ι) (k : ℕ) : IntForm (ι × Fin k) where
  gram := Matrix.blockDiagonal fun _ : Fin k => L.gram
  gram_symm := by
    have hL : L.gramᵀ = L.gram := L.gram_symm
    show (Matrix.blockDiagonal fun _ : Fin k => L.gram)ᵀ = _
    rw [Matrix.blockDiagonal_transpose]
    simp only [hL]

/-- The Gram matrix of `k` copies is block diagonal with `k` equal blocks.

    (L : IntForm ι) (k : ℕ)            [a form and a count]
    IntForm                            [an integral symmetric bilinear form]
    Matrix                             [a rectangle of numbers, indexed by row and column]
    ℕ                                  [the counting numbers 0, 1, 2, 3, and so on] -/
@[simp] theorem pow_gram (L : IntForm ι) (k : ℕ) :
    (L.pow k).gram = Matrix.blockDiagonal fun _ : Fin k => L.gram := rfl

/-- The same over the reals.

    (L : IntForm ι) (k : ℕ)            [a form and a count]
    IntForm                            [an integral symmetric bilinear form]
    Matrix                             [a rectangle of numbers, indexed by row and column]
    ℕ                                  [the counting numbers] -/
theorem pow_gramR (L : IntForm ι) (k : ℕ) :
    (L.pow k).gramR = Matrix.blockDiagonal fun _ : Fin k => L.gramR := by
  simp [gramR, pow, Matrix.blockDiagonal_map _ _ (by simp : ((0 : ℤ) : ℝ) = 0)]

/-- **Copies of an even form are even.**

    {L : IntForm ι} (hL : L.IsEven) (k : ℕ)          [an even form and a count]
    (L.pow k).IsEven                                 [the copies are even]
    IntForm                                          [an integral symmetric bilinear form]
    ℕ                                                [the counting numbers] -/
theorem pow_isEven {L : IntForm ι} (hL : L.IsEven) (k : ℕ) : (L.pow k).IsEven := by
  rw [isEven_iff_even_diag] at hL ⊢
  rintro ⟨i, c⟩
  simpa using hL i

/-- The determinant of `k` copies is the `k`-th power.

    (L : IntForm ι) (k : ℕ)            [a form and a count]
    IntForm                            [an integral symmetric bilinear form]
    ℕ                                  [the counting numbers] -/
@[simp] theorem pow_det (L : IntForm ι) (k : ℕ) : (L.pow k).gram.det = L.gram.det ^ k := by
  simp [pow]

/-- **Copies of a unimodular form are unimodular.**

    {L : IntForm ι} (hL : L.IsUnimodular) (k : ℕ)    [a unimodular form and a count]
    (L.pow k).IsUnimodular                           [the copies are unimodular]
    IntForm                                          [an integral symmetric bilinear form]
    ℕ                                                [the counting numbers] -/
theorem pow_isUnimodular {L : IntForm ι} (hL : L.IsUnimodular) (k : ℕ) :
    (L.pow k).IsUnimodular := by
  rw [IsUnimodular, pow_det]; exact hL.pow k

/-- **Copies of a form carry the repeated sign vector.**

    {L : IntForm ι} {s : ι → SignatureSign} (hL : L.HasSignature s) (k : ℕ)
                                                                 [a diagonalised finite signature]
    (L.pow k).HasSignature (fun p : ι × Fin k => s p.1)             [the copies repeat the signs]
    IntForm                                                    [an integral symmetric bilinear form]
    ×                                                               [paired with]
    →                                                           [takes the left and gives the right]
    ℕ                                                               [the counting numbers]
    SignatureSign                                                   [one of the two finite signs] -/
theorem pow_hasSignature {L : IntForm ι} {s : ι → SignatureSign} (hL : L.HasSignature s) (k : ℕ) :
    (L.pow k).HasSignature (fun p : ι × Fin k => s p.1) := by
  obtain ⟨P, hP, hPeq⟩ := hL
  refine ⟨Matrix.blockDiagonal fun _ : Fin k => P, ?_, ?_⟩
  · rw [Matrix.det_blockDiagonal]
    exact Finset.prod_ne_zero_iff.mpr fun _ _ => hP
  · rw [pow_gramR, Matrix.blockDiagonal_transpose,
      ← Matrix.blockDiagonal_mul, ← Matrix.blockDiagonal_mul]
    simp_rw [hPeq]
    rw [Matrix.blockDiagonal_diagonal]

omit [DecidableEq ι] in
/-- Plus signs multiply by the number of copies.

    (s : ι → SignatureSign) (k : ℕ)    [a finite sign vector and a count]
    sigPos                             [how many plus ones a sign vector has]
    ×                                  [paired with]
    →                                  [takes the left and gives the right]
    ℕ                                  [the counting numbers]
    SignatureSign                      [one of the two finite signs] -/
@[simp] theorem sigPos_pow (s : ι → SignatureSign) (k : ℕ) :
    sigPos (fun p : ι × Fin k => s p.1) = k * sigPos s := by
  rw [sigPos, sigPos, Fintype.sum_prod_type, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => by cases h : s i <;> simp [h]

omit [DecidableEq ι] in
/-- The other entries multiply by the number of copies.

    (s : ι → SignatureSign) (k : ℕ)    [a finite sign vector and a count]
    sigNeg                             [how many entries are not plus one]
    ×                                  [paired with]
    →                                  [takes the left and gives the right]
    ℕ                                  [the counting numbers]
    SignatureSign                      [one of the two finite signs] -/
@[simp] theorem sigNeg_pow (s : ι → SignatureSign) (k : ℕ) :
    sigNeg (fun p : ι × Fin k => s p.1) = k * sigNeg s := by
  rw [sigNeg, sigNeg, Fintype.sum_prod_type, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => by cases h : s i <;> simp [h]

variable {L : IntForm ι} {B : Matrix ι ι ℝ}

/-- **A form factoring as `B Bᵀ` with `B` invertible has an all-plus signature.**

    (hB : B.det ≠ 0) (h : L.gramR = B * Bᵀ)          [an invertible factorisation]
    L.HasSignature (fun _ => .positive)              [the signature is all plus]
    IntForm                                          [an integral symmetric bilinear form]
    Matrix                                       [a rectangle of numbers, indexed by row and column]
    ≠                                                [is not equal to]
    ℝ                                                [the real numbers]

    So: the witness is `P = (B⁻¹)ᵀ`, for which `Pᵀ G P = B⁻¹ B Bᵀ (Bᵀ)⁻¹ = 1`. Gram matrices of
    real bases are exactly the matrices `B Bᵀ`, so this is the working criterion for positive
    definiteness and no minor determinants are needed. -/
theorem hasSignature_one_of_gram_eq_mulTranspose (hB : B.det ≠ 0) (h : L.gramR = B * Bᵀ) :
    L.HasSignature (fun _ => .positive) := by
  have hu : IsUnit B.det := isUnit_iff_ne_zero.mpr hB
  have huT : IsUnit (Bᵀ).det := by rwa [Matrix.det_transpose]
  refine ⟨(B⁻¹)ᵀ, ?_, ?_⟩
  · have hinv : IsUnit (B⁻¹).det := Matrix.isUnit_nonsing_inv_det B hu
    rw [Matrix.det_transpose]
    exact isUnit_iff_ne_zero.mp hinv
  · change (B⁻¹) * L.gramR * (B⁻¹)ᵀ = Matrix.diagonal (fun _ => (1 : ℝ))
    rw [h, Matrix.diagonal_one,
      Matrix.mul_assoc, Matrix.mul_assoc, Matrix.transpose_nonsing_inv,
      Matrix.mul_nonsing_inv _ huT, Matrix.mul_one, Matrix.nonsing_inv_mul _ hu]

/-- **A unimodular form factoring as `B Bᵀ` has determinant one.**

    (h : L.gramR = B * Bᵀ) (hu : L.IsUnimodular)     [a factorisation and unimodularity]
    L.gram.det = 1                                   [the determinant is one, not minus one]
    IntForm                                          [an integral symmetric bilinear form]
    Matrix                                       [a rectangle of numbers, indexed by row and column]
    ℝ                                                [the real numbers]

    So: `(det G : ℝ) = (det B)² ≥ 0`, which rules out the other unit. -/
theorem det_eq_one_of_gram_eq_mulTranspose (h : L.gramR = B * Bᵀ) (hu : L.IsUnimodular) :
    L.gram.det = 1 := by
  have hcast : ((L.gram.det : ℤ) : ℝ) = B.det * B.det := by
    have hmap : ((L.gram.det : ℤ) : ℝ) = L.gramR.det := Int.cast_det L.gram
    rw [hmap, h, Matrix.det_mul, Matrix.det_transpose]
  rcases (isUnimodular_iff L).mp hu with h1 | h1
  · exact h1
  · exfalso
    rw [h1] at hcast
    have hnn : (0 : ℝ) ≤ B.det * B.det := mul_self_nonneg _
    rw [← hcast] at hnn
    norm_num at hnn

/-- **A form factoring as `B Bᵀ` with `B` invertible is positive definite over the reals.**

    (hB : B.det ≠ 0) (h : L.gramR = B * Bᵀ)          [an invertible factorisation]
    (v : ι → ℝ) (hv : v ≠ 0)                         [a non-zero real vector]
    0 < v ⬝ᵥ (L.gramR *ᵥ v)                          [it pairs positively with itself]
    IntForm                                          [an integral symmetric bilinear form]
    Matrix                                       [a rectangle of numbers, indexed by row and column]
    ≠                                                [is not equal to]
    <                                                [is smaller than]
    →                                                [takes the left and gives the right]
    ℝ                                                [the real numbers] -/
theorem gramR_pos_of_gram_eq_mulTranspose (hB : B.det ≠ 0) (h : L.gramR = B * Bᵀ)
    (v : ι → ℝ) (hv : v ≠ 0) : 0 < v ⬝ᵥ (L.gramR *ᵥ v) := by
  have hu : IsUnit B.det := isUnit_iff_ne_zero.mpr hB
  have huT : IsUnit (Bᵀ).det := by rwa [Matrix.det_transpose]
  have hkey : v ⬝ᵥ (L.gramR *ᵥ v) = (Bᵀ *ᵥ v) ⬝ᵥ (Bᵀ *ᵥ v) := by
    rw [h, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose]
  rw [hkey]
  have hnn : (0 : ℝ) ≤ (Bᵀ *ᵥ v) ⬝ᵥ (Bᵀ *ᵥ v) :=
    Finset.sum_nonneg fun i _ => mul_self_nonneg _
  rcases lt_or_eq_of_le hnn with hlt | heq
  · exact hlt
  · exfalso
    have hz : Bᵀ *ᵥ v = 0 := _root_.dotProduct_self_eq_zero.mp heq.symm
    have hv0 : v = 0 := by
      have hmap := congrArg (fun w => (Bᵀ)⁻¹ *ᵥ w) hz
      simpa [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ huT] using hmap
    exact hv hv0

/-- **The same form is positive definite on non-zero integer vectors.**

    (hB : B.det ≠ 0) (h : L.gramR = B * Bᵀ)          [an invertible factorisation]
    (x : ι → ℤ) (hx : x ≠ 0)                         [a non-zero integer vector]
    0 < L.bil x x                                    [it pairs positively with itself]
    IntForm                                          [an integral symmetric bilinear form]
    Matrix                                       [a rectangle of numbers, indexed by row and column]
    ≠                                                [is not equal to]
    <                                                [is smaller than]
    →                                                [takes the left and gives the right]
    ℝ                                                [the real numbers] -/
theorem bil_pos_of_gram_eq_mulTranspose (hB : B.det ≠ 0) (h : L.gramR = B * Bᵀ)
    (x : ι → ℤ) (hx : x ≠ 0) : 0 < L.bil x x := by
  have hv : (fun i => (x i : ℝ)) ≠ 0 := by
    intro hc
    apply hx
    funext i
    have hi : ((x i : ℤ) : ℝ) = 0 := by simpa using congrFun hc i
    exact_mod_cast hi
  have hpos := gramR_pos_of_gram_eq_mulTranspose hB h _ hv
  rw [dot_mulVec_eq_double_sum] at hpos
  have hcast : ((L.bil x x : ℤ) : ℝ)
      = ∑ i, ∑ c, (x i : ℝ) * L.gramR i c * (x c : ℝ) := by
    rw [bil_eq_double_sum]
    push_cast [gramR_apply]
    rfl
  rw [← hcast] at hpos
  exact_mod_cast hpos

end IntForm
end Lattice
end Apeiron
