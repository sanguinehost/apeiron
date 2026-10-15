/-
# The cell is a Clifford algebra; two anticommuting cells are a quaternion algebra

. The canonical object is `Apeiron.Trichotomy.Cell = ℝ × ℝ` with
`mul σ z w = (z.1*w.1 + σ*z.2*w.2, z.1*w.2 + z.2*w.1)` — the ring `ℝ[x]/(x² − σ)` in two
coordinates (`Apeiron.Trichotomy.Basic`).

The one-dimensional quadratic form `Q σ t = σ·t²` has Clifford algebra
`CliffordAlgebra (Q σ)`, and `ℝ[x]/(x² − σ)` is exactly that algebra, at every `σ`, with no
face hypothesis: `toCell σ` is an algebra isomorphism carrying the Clifford generator `ι (Q σ) t`
to the cell `(0, t)` and carrying Clifford multiplication to `mul σ`. This is the content of
`Q_apply`, `toAdj`, `ofAdj`, `cliffordEquiv`, `toCell_mul`, `toCell_one`, `toCell_ι` and
`toCell_bijective`.

The three faces are instances. At `σ = −1` the form is Mathlib's negated square
(`Q_neg_one`) and the algebra is `ℂ` (`elliptic_complex`); at `σ = 0` the form is zero
(`Q_zero`) and the algebra is the dual numbers (`parabolic_dual`).

Two anticommuting generators are a quaternion algebra. In `CliffordAlgebra
(CliffordAlgebraQuaternion.Q c₁ c₂)` the two coordinate vectors `e1 = ι (1,0)`, `e2 = ι (0,1)`
square to `c₁`, `c₂` (`e1_sq`, `e2_sq`) and anticommute (`anticommute`); at `(−1,−1)` the
generator `k` squares to `−1` (Hamilton, spin — `k_sq_hamilton`), and at `(+1,+1)` two
hyperbolic generators compose to an elliptic one (`k_sq_split`). The split quaternions
`ℍ[ℝ,1,0,1]` are the real 2×2 matrices: `rebitBasis` with `i = !![1,0;0,-1]`,
`j = !![0,1;1,0]`, `k = i*j`, and `rebit_bijective` says the induced algebra map `rebit` is a
bijection — the rebit algebra. The verdict is `verdict_l1`.

Anchor, graded `anchor` and not proved here: Clifford (1878) for the algebra, Hamilton (1843)
for the quaternions; the rebit reading is McKague–Mosca–Gisin. The point of the file: there is
no fourth face — a spin system is two anticommuting cells.
-/
import Mathlib

import Apeiron.Trichotomy.Basic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable section

namespace Quantum
namespace CellClifford

open Apeiron.Trichotomy (Cell one mul rel root_sq mul_eq_quotient_mul toQuot toQuot_one
  toQuotEquiv toQuotEquiv_apply)
open AdjoinRoot (root)
open CliffordAlgebra
open Polynomial
open scoped Quaternion

/-- The one-dimensional quadratic form `t ↦ σ·t²` on `ℝ`. -/
def Q (σ : ℝ) : QuadraticForm ℝ ℝ := σ • QuadraticMap.sq

/-- `Q σ` evaluated: `Q σ t = σ·(t·t)`. -/
theorem Q_apply (σ t : ℝ) : Q σ t = σ * (t * t) := by
  simp [Q, smul_eq_mul]

/-- `t ↦ t • root (rel σ)`, linear, with square `t² · σ`. -/
def genLinear (σ : ℝ) : ℝ →ₗ[ℝ] AdjoinRoot (rel σ) where
  toFun t := t • root (rel σ)
  map_add' t u := by rw [add_smul]
  map_smul' c t := smul_assoc c t (root (rel σ))

theorem genLinear_apply (σ t : ℝ) : genLinear σ t = t • root (rel σ) := rfl

theorem genLinear_sq (σ t : ℝ) :
    genLinear σ t * genLinear σ t = algebraMap ℝ (AdjoinRoot (rel σ)) (Q σ t) := by
  have hr : root (rel σ) * root (rel σ) = algebraMap ℝ (AdjoinRoot (rel σ)) σ := by
    rw [← pow_two, root_sq]
  rw [genLinear_apply, Algebra.smul_def, Q_apply]
  simp only [map_mul]
  linear_combination (algebraMap ℝ (AdjoinRoot (rel σ)) t *
    algebraMap ℝ (AdjoinRoot (rel σ)) t) * hr

/-- The algebra map `CliffordAlgebra (Q σ) → AdjoinRoot (rel σ)` sending `ι (Q σ) t` to
`t • root (rel σ)`: the Clifford algebra of `σ·t²` is the quotient `ℝ[x]/(x² − σ)`. -/
def toAdj (σ : ℝ) : CliffordAlgebra (Q σ) →ₐ[ℝ] AdjoinRoot (rel σ) :=
  CliffordAlgebra.lift (Q σ) ⟨genLinear σ, genLinear_sq σ⟩

/-- `toAdj` on a generator. -/
theorem toAdj_ι (σ t : ℝ) : toAdj σ (ι (Q σ) t) = t • root (rel σ) := by
  rw [toAdj, CliffordAlgebra.lift_ι_apply, genLinear_apply]

/-- `t • ι (Q σ) 1 = ι (Q σ) t`: `ι` is linear. -/
theorem smul_ι_one (σ t : ℝ) : t • ι (Q σ) (1 : ℝ) = ι (Q σ) t := by
  conv_rhs => rw [show t = t • (1 : ℝ) by rw [smul_eq_mul, mul_one]]
  rw [LinearMap.map_smul]

/-- In the one-dimensional Clifford algebra `CliffordAlgebra (Q σ)` every generator commutes:
the dimension is one, so `ι x * b = b * ι x` for all `b`. This is the commutativity that
`AdjoinRoot.liftHom` needs of its target. -/
theorem ι_mul_eq_mul_ι (σ x : ℝ) (b : CliffordAlgebra (Q σ)) :
    ι (Q σ) x * b = b * ι (Q σ) x := by
  induction b using CliffordAlgebra.induction with
  | algebraMap r => exact (Algebra.commutes r (ι (Q σ) x)).symm
  | ι y =>
      calc ι (Q σ) x * ι (Q σ) y
          = (x • ι (Q σ) 1) * (y • ι (Q σ) 1) := by rw [smul_ι_one, smul_ι_one]
        _ = (x * y) • (ι (Q σ) 1 * ι (Q σ) 1) :=
            smul_mul_smul_comm x (ι (Q σ) 1) y (ι (Q σ) 1)
        _ = (y * x) • (ι (Q σ) 1 * ι (Q σ) 1) := by rw [mul_comm x y]
        _ = (y • ι (Q σ) 1) * (x • ι (Q σ) 1) :=
            (smul_mul_smul_comm y (ι (Q σ) 1) x (ι (Q σ) 1)).symm
        _ = ι (Q σ) y * ι (Q σ) x := by rw [smul_ι_one, smul_ι_one]
  | add a b ha hb => rw [mul_add, add_mul, ha, hb]
  | mul a b ha hb =>
      calc ι (Q σ) x * (a * b) = (ι (Q σ) x * a) * b := (mul_assoc _ _ _).symm
        _ = (a * ι (Q σ) x) * b := by rw [ha]
        _ = a * (ι (Q σ) x * b) := mul_assoc _ _ _
        _ = a * (b * ι (Q σ) x) := by rw [hb]
        _ = (a * b) * ι (Q σ) x := (mul_assoc _ _ _).symm

/-- The one-dimensional Clifford algebra `CliffordAlgebra (Q σ)` is commutative. -/
instance instCommRing (σ : ℝ) : CommRing (CliffordAlgebra (Q σ)) where
  mul_comm := fun x y => by
    induction x using CliffordAlgebra.induction with
    | algebraMap r => exact Algebra.commutes r y
    | ι x => exact ι_mul_eq_mul_ι σ x y
    | add x₁ x₂ hx₁ hx₂ => rw [add_mul, mul_add, hx₁, hx₂]
    | mul x₁ x₂ hx₁ hx₂ =>
        calc (x₁ * x₂) * y = x₁ * (x₂ * y) := mul_assoc _ _ _
          _ = x₁ * (y * x₂) := by rw [hx₂]
          _ = (x₁ * y) * x₂ := (mul_assoc _ _ _).symm
          _ = (y * x₁) * x₂ := by rw [hx₁]
          _ = y * (x₁ * x₂) := mul_assoc _ _ _

/-- The generator `ι (Q σ) 1` is a root of `X² − C σ`. -/
theorem aeval_rel (σ : ℝ) : aeval (ι (Q σ) (1 : ℝ)) (rel σ) = 0 := by
  have hrel : rel σ = X ^ 2 - C σ := rfl
  rw [hrel, map_sub, map_pow, Polynomial.aeval_X, Polynomial.aeval_C, pow_two, ι_sq_scalar,
    Q_apply, one_mul]
  simp

/-- The root of `rel σ` in `CliffordAlgebra (Q σ)`: the inverse direction, from the quotient to
the Clifford algebra. `AdjoinRoot (rel σ)` is `ℝ[X] ⧸ ⟨X² − C σ⟩`, and `IsAdjoinRoot.liftHom`
descends the algebra map `ℝ[X] → CliffordAlgebra (Q σ)`, `X ↦ ι (Q σ) 1`, through that quotient;
it needs the target commutative, which `instCommRing` supplies. -/
def ofAdj (σ : ℝ) : AdjoinRoot (rel σ) →ₐ[ℝ] CliffordAlgebra (Q σ) :=
  (AdjoinRoot.isAdjoinRoot (rel σ)).liftHom (ι (Q σ) (1 : ℝ)) (aeval_rel σ)

/-- `ofAdj` on the generator. -/
theorem ofAdj_root (σ : ℝ) : ofAdj σ (root (rel σ)) = ι (Q σ) (1 : ℝ) := by
  unfold ofAdj
  rw [← AdjoinRoot.isAdjoinRoot_root_eq_root (rel σ), IsAdjoinRoot.liftHom_root]

/-- The two directions compose to the identity on the quotient. -/
theorem toAdj_comp_ofAdj (σ : ℝ) :
    (toAdj σ).comp (ofAdj σ) = AlgHom.id ℝ (AdjoinRoot (rel σ)) := by
  apply AdjoinRoot.algHom_ext
  rw [AlgHom.comp_apply, ofAdj_root, toAdj_ι, one_smul, AlgHom.id_apply]

/-- The two directions compose to the identity on the Clifford algebra. -/
theorem ofAdj_comp_toAdj (σ : ℝ) :
    (ofAdj σ).comp (toAdj σ) = AlgHom.id ℝ (CliffordAlgebra (Q σ)) := by
  apply CliffordAlgebra.hom_ext
  apply LinearMap.ext
  intro t
  change ofAdj σ (toAdj σ (ι (Q σ) t)) = ι (Q σ) t
  rw [toAdj_ι, map_smul, ofAdj_root, smul_ι_one]

/-- `ℝ[x]/(x² − σ) ≅ CliffordAlgebra (Q σ)`: the cell is the Clifford algebra of `σ·t²`. -/
def cliffordEquiv (σ : ℝ) : CliffordAlgebra (Q σ) ≃ₐ[ℝ] AdjoinRoot (rel σ) :=
  AlgEquiv.ofAlgHom (toAdj σ) (ofAdj σ) (toAdj_comp_ofAdj σ) (ofAdj_comp_toAdj σ)

theorem cliffordEquiv_apply (σ : ℝ) (a : CliffordAlgebra (Q σ)) :
    cliffordEquiv σ a = toAdj σ a :=
  rfl

/-- The cell underlying a Clifford algebra element. -/
def toCell (σ : ℝ) (a : CliffordAlgebra (Q σ)) : Cell :=
  (toQuotEquiv σ).symm (cliffordEquiv σ a)

/-- `toQuot` of `toCell` is the Clifford algebra element. -/
theorem toQuot_toCell (σ : ℝ) (a : CliffordAlgebra (Q σ)) :
    toQuot σ (toCell σ a) = cliffordEquiv σ a := by
  rw [← toQuotEquiv_apply, toCell, LinearEquiv.apply_symm_apply]

/-- `toCell` carries Clifford multiplication to `mul σ`. -/
theorem toCell_mul (σ : ℝ) (a b : CliffordAlgebra (Q σ)) :
    toCell σ (a * b) = mul σ (toCell σ a) (toCell σ b) := by
  apply (toQuotEquiv σ).injective
  rw [toQuotEquiv_apply, toQuotEquiv_apply, toQuot_toCell, ← mul_eq_quotient_mul, toQuot_toCell,
    toQuot_toCell]
  exact map_mul (cliffordEquiv σ) a b

/-- `toCell` carries `1` to `one`. -/
theorem toCell_one (σ : ℝ) : toCell σ 1 = one := by
  apply (toQuotEquiv σ).injective
  rw [toQuotEquiv_apply, toQuotEquiv_apply, toQuot_toCell, map_one, toQuot_one]

/-- `toCell` carries the Clifford generator `ι (Q σ) t` to the cell `(0, t)`. -/
theorem toCell_ι (σ t : ℝ) : toCell σ (ι (Q σ) t) = ((0 : ℝ), t) := by
  apply (toQuotEquiv σ).injective
  rw [toQuotEquiv_apply, toQuotEquiv_apply, toQuot_toCell, cliffordEquiv_apply, toAdj_ι]
  simp [toQuot, Algebra.smul_def]

/-- `toCell` is a bijection: the two presentations have the same carrier. -/
theorem toCell_bijective (σ : ℝ) : Function.Bijective (toCell σ) :=
  Function.Bijective.comp (toQuotEquiv σ).symm.bijective (cliffordEquiv σ).bijective

-- `instCommRing` is dropped here: the face theorems below rewrite `Q (-1)` and `Q 0`, and the
-- commutativity instance would make the `QuadraticForm`-transport motive ill-typed.
attribute [-instance] instCommRing

/-- At `σ = −1` the form is Mathlib's `CliffordAlgebraComplex.Q`. -/
theorem Q_neg_one : Q (-1) = CliffordAlgebraComplex.Q := by
  ext v
  rw [Q_apply, CliffordAlgebraComplex.Q_apply]
  ring

/-- At `σ = 0` the form is zero. -/
theorem Q_zero : Q 0 = 0 := by
  ext v
  rw [Q_apply]
  simp

/-- The elliptic face: `CliffordAlgebra (Q (−1)) ≅ ℂ`. -/
theorem elliptic_complex : Nonempty (CliffordAlgebra (Q (-1)) ≃ₐ[ℝ] ℂ) := by
  rw [Q_neg_one]
  exact ⟨CliffordAlgebraComplex.equiv⟩

/-- The parabolic face: `CliffordAlgebra (Q 0) ≅ ℝ[ε]`. -/
theorem parabolic_dual : Nonempty (CliffordAlgebra (Q 0) ≃ₐ[ℝ] DualNumber ℝ) := by
  rw [Q_zero]
  exact ⟨CliffordAlgebraDualNumber.equiv⟩

/-- The first coordinate generator `e1 = ι (1,0)` in `CliffordAlgebra (Q c₁ c₂)`. -/
def e1 (c₁ c₂ : ℝ) : CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂) :=
  ι (CliffordAlgebraQuaternion.Q c₁ c₂) ((1 : ℝ), 0)

/-- The second coordinate generator `e2 = ι (0,1)` in `CliffordAlgebra (Q c₁ c₂)`. -/
def e2 (c₁ c₂ : ℝ) : CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂) :=
  ι (CliffordAlgebraQuaternion.Q c₁ c₂) (0, (1 : ℝ))

/-- `e1` squares to `c₁`. -/
theorem e1_sq (c₁ c₂ : ℝ) :
    e1 c₁ c₂ * e1 c₁ c₂ =
      algebraMap ℝ (CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂)) c₁ := by
  rw [e1, ι_sq_scalar, CliffordAlgebraQuaternion.Q_apply]
  simp

/-- `e2` squares to `c₂`. -/
theorem e2_sq (c₁ c₂ : ℝ) :
    e2 c₁ c₂ * e2 c₁ c₂ =
      algebraMap ℝ (CliffordAlgebra (CliffordAlgebraQuaternion.Q c₁ c₂)) c₂ := by
  rw [e2, ι_sq_scalar, CliffordAlgebraQuaternion.Q_apply]
  simp

/-- `e1` and `e2` anticommute. -/
theorem anticommute (c₁ c₂ : ℝ) :
    e1 c₁ c₂ * e2 c₁ c₂ + e2 c₁ c₂ * e1 c₁ c₂ = 0 := by
  rw [e1, e2, ι_mul_ι_add_swap]
  have h : (CliffordAlgebraQuaternion.Q c₁ c₂).IsOrtho ((1 : ℝ), 0) ((0 : ℝ), 1) := by
    rw [QuadraticMap.isOrtho_def]
    rw [Prod.mk_add_mk, CliffordAlgebraQuaternion.Q_apply, CliffordAlgebraQuaternion.Q_apply,
      CliffordAlgebraQuaternion.Q_apply]
    ring
  rw [h.polar_eq_zero, map_zero]

/-- At `(−1,−1)` the generator `k` is elliptic: `k² = −1` (Hamilton, spin). -/
theorem k_sq_hamilton :
    (⟨0, 0, 0, 1⟩ : ℍ[ℝ,-1,0,-1]) * ⟨0, 0, 0, 1⟩ = -1 := by
  have hk : (⟨0, 0, 0, 1⟩ : ℍ[ℝ,-1,0,-1]) = (QuaternionAlgebra.Basis.self ℝ).k := rfl
  rw [hk, QuaternionAlgebra.Basis.k_mul_k]
  norm_num

/-- At `(+1,+1)` two hyperbolic generators compose to an elliptic one: `k² = −1`. -/
theorem k_sq_split :
    (⟨0, 0, 0, 1⟩ : ℍ[ℝ,1,0,1]) * ⟨0, 0, 0, 1⟩ = -1 := by
  have hk : (⟨0, 0, 0, 1⟩ : ℍ[ℝ,1,0,1]) = (QuaternionAlgebra.Basis.self ℝ).k := rfl
  rw [hk, QuaternionAlgebra.Basis.k_mul_k]
  norm_num

/-- The rebit basis `i = !![1,0;0,-1]`, `j = !![0,1;1,0]`, `k = i*j` of `M₂(ℝ)`. -/
def rebitBasis : QuaternionAlgebra.Basis (R := ℝ) (Matrix (Fin 2) (Fin 2) ℝ) 1 0 1 where
  i := !![1, 0; 0, -1]
  j := !![0, 1; 1, 0]
  k := !![1, 0; 0, -1] * !![0, 1; 1, 0]
  i_mul_i := by
    rw [one_smul, zero_smul, add_zero, Matrix.one_fin_two]
    ext a b
    fin_cases a <;> fin_cases b <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  j_mul_j := by
    rw [one_smul, Matrix.one_fin_two]
    ext a b
    fin_cases a <;> fin_cases b <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  i_mul_j := rfl
  j_mul_i := by
    rw [zero_smul, zero_sub]
    ext a b
    fin_cases a <;> fin_cases b <;> simp [Matrix.mul_apply, Fin.sum_univ_two]

theorem rebitBasis_i : rebitBasis.i = !![1, 0; 0, -1] := rfl

theorem rebitBasis_j : rebitBasis.j = !![0, 1; 1, 0] := rfl

/-- `k` opened out in the rebit basis. -/
theorem rebitBasis_k : rebitBasis.k = !![0, 1; -1, 0] := by
  rw [show rebitBasis.k = !![1, 0; 0, -1] * !![0, 1; 1, 0] from rfl, Matrix.mul_fin_two]
  ext a b
  fin_cases a <;> fin_cases b <;> simp

/-- The algebra map `ℍ[ℝ,1,0,1] → M₂(ℝ)` determined by the rebit basis. -/
def rebit : ℍ[ℝ,1,0,1] →ₐ[ℝ] Matrix (Fin 2) (Fin 2) ℝ :=
  rebitBasis.liftHom

theorem rebit_mk (a b c d : ℝ) :
    rebit (⟨a, b, c, d⟩ : ℍ[ℝ,1,0,1]) = !![a + b, c + d; c - d, a - b] := by
  rw [rebit, QuaternionAlgebra.Basis.liftHom_apply, QuaternionAlgebra.Basis.lift, rebitBasis_i,
    rebitBasis_j, rebitBasis_k, Algebra.algebraMap_eq_smul_one, Matrix.one_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;> simp <;> ring

theorem rebit_apply (x : ℍ[ℝ,1,0,1]) :
    rebit x = !![x.re + x.imI, x.imJ + x.imK; x.imJ - x.imK, x.re - x.imI] := by
  obtain ⟨re, imI, imJ, imK⟩ := x
  exact rebit_mk re imI imJ imK

/-- `ℍ[ℝ,1,0,1] ≅ M₂(ℝ)`: the rebit is a bijection. -/
theorem rebit_bijective : Function.Bijective rebit := by
  constructor
  · intro x y h
    obtain ⟨xre, ximI, ximJ, ximK⟩ := x
    obtain ⟨yre, yimI, yimJ, yimK⟩ := y
    simp only [rebit_mk] at h
    have h00 : xre + ximI = yre + yimI := by
      simpa using congrFun (congrFun h 0) 0
    have h01 : ximJ + ximK = yimJ + yimK := by
      simpa using congrFun (congrFun h 0) 1
    have h10 : ximJ - ximK = yimJ - yimK := by
      simpa using congrFun (congrFun h 1) 0
    have h11 : xre - ximI = yre - yimI := by
      simpa using congrFun (congrFun h 1) 1
    ext <;> linarith
  · intro M
    refine ⟨⟨(M 0 0 + M 1 1) / 2, (M 0 0 - M 1 1) / 2, (M 0 1 + M 1 0) / 2,
      (M 0 1 - M 1 0) / 2⟩, ?_⟩
    rw [rebit_mk]
    ext i j
    fin_cases i <;> fin_cases j <;> simp <;> ring

/-- The unit verdict: the cell is the Clifford algebra of `σ·t²` at every `σ`; the three faces
are instances; two anticommuting generators are a quaternion algebra; and the rebit is a
bijection onto `M₂(ℝ)`. -/
theorem verdict_l1 :
    (∀ (σ : ℝ) (a b : CliffordAlgebra (Q σ)),
        toCell σ (a * b) = mul σ (toCell σ a) (toCell σ b)) ∧
      (∀ σ : ℝ, toCell σ 1 = one) ∧
      (∀ (σ t : ℝ), toCell σ (ι (Q σ) t) = ((0 : ℝ), t)) ∧
      (∀ σ : ℝ, Function.Bijective (toCell σ)) ∧
      Q (-1) = CliffordAlgebraComplex.Q ∧ Q 0 = 0 ∧
      (∀ c₁ c₂ : ℝ, e1 c₁ c₂ * e2 c₁ c₂ + e2 c₁ c₂ * e1 c₁ c₂ = 0) ∧
      (⟨0, 0, 0, 1⟩ : ℍ[ℝ,-1,0,-1]) * ⟨0, 0, 0, 1⟩ = -1 ∧
      (⟨0, 0, 0, 1⟩ : ℍ[ℝ,1,0,1]) * ⟨0, 0, 0, 1⟩ = -1 ∧
      Function.Bijective rebit :=
  ⟨toCell_mul, toCell_one, toCell_ι, toCell_bijective, Q_neg_one, Q_zero, anticommute,
    k_sq_hamilton, k_sq_split, rebit_bijective⟩

end CellClifford
end Quantum
