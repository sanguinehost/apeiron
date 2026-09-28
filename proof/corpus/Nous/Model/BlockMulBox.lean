/-
# A real gate-product stage for the FFN, and a non-degenerate joint block witness (`APEIRON-MODEL-FIX-b`)

`Nous.Model.BlockComposition` carries the FFN elementwise gate/up product as an unrefined `Stage`
hypothesis, `BlockResidual.mul`, whose reference map is pinned to the real elementwise product on its
admissible domain only (`BlockResidual.mul_ref_on`, `BlockComposition.lean:384`). The old
whole-space pin is refuted
(`Nous.Model.BlockGroundedWitness1.pinnedMul_stage_impossible`): the product has no global Lipschitz
constant while `Stage.expansion` is global, so `BlockResidual p h` was empty for every `1 ≤ h`.

This module supplies a *real* gate product with a global expansion factor: the product of
per-coordinate clamps to the box `[-B, B]`. `clampB B` is the projection onto `[-B, B]`; it is
1-Lipschitz (`clampB_lipschitz`) and maps into `[-B, B]` (`abs_clampB_le`), so the clamped product is
globally Lipschitz with sup-norm constant `2 * B` (`mulBoxStage`) and agrees with the exact product
on the box (`mulBoxStage_ref_on`).

The canonical instance is `mulBoxStage_exact`: the executing map is the exact (unclamped) product and
the defect budget is `0`, because on the box the clamps are the identity. No kernel refinement of the
gate product exists at this revision, so `mulBoxStage`'s executing map is a caller-supplied parameter
and `mulBoxStage_exact` is the exact product itself; nothing here claims a Rust body for it.

`blockResidual_nonempty` inhabits `BlockResidual p h` for every `p`, `h` (zero attention and the
boxed product), and `fullBlockGrounded_dom_nonempty_111` places the zero input in
`fullBlockGrounded`'s composed domain at `d = p = h = 1` with the real `F16` projections of
`Nous.Model.BlockGroundedWitness1` (`projF16_dom_zero`, `projF16_exec_zero`,
`rmsVectorStage_exec_zero`), discharging the joint-witness obligation at a real FFN width. The
hidden width is `1` here, unlike the `h = 0` instance of `fullBlockGrounded_dom_nonempty_1`.

Trust boundary: as `Nous.Model.BlockGrounded`, the algebra is checked at the abstract real level
under the closed axiom cone `{propext, Classical.choice, Quot.sound}`; no claim about rustc/LLVM
codegen, the kernel ABI, GPU/SPIR-V, the `unsafe` container, or the dequant-to-ideal-weight reading.
-/
import Nous.Model.BlockGroundedWitness1
import Mathlib.Algebra.Order.Group.MinMax

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open apeiron_word Aeneas Aeneas.Std Result Classical

namespace Nous.Model

noncomputable section

/-! ## The box projection -/

/-- The projection of `x` onto the closed interval `[-B, B]`, as `max (-B) (min B x)`. -/
def clampB (B x : ℝ) : ℝ := max (-B) (min B x)

/-- `clampB B x = min (max (-B) x) B` for `0 ≤ B`: the clamp is the composition of the lower and the
upper cut, which exposes it as a composition of two 1-Lipschitz maps. -/
theorem clampB_eq_min_max (B : ℝ) (hB : 0 ≤ B) (x : ℝ) :
    clampB B x = min (max (-B) x) B := by
  unfold clampB
  rcases le_total x B with h | h
  · rcases le_total (-B) x with h2 | h2
    · rw [min_eq_right h, max_eq_right h2, min_eq_left h]
    · rw [min_eq_right h, max_eq_left h2, min_eq_left (by linarith)]
  · have hx : -B ≤ x := by linarith
    rw [min_eq_left h, max_eq_right (by linarith : -B ≤ B), max_eq_right hx, min_eq_right h]

/-- The clamp fixes every point of the box `[-B, B]`. -/
theorem clampB_eq_self {B x : ℝ} (hx : |x| ≤ B) : clampB B x = x := by
  unfold clampB
  rw [min_eq_right (abs_le.mp hx).2, max_eq_right (abs_le.mp hx).1]

/-- The clamp maps into the box: `|clampB B x| ≤ B` for `0 ≤ B`. -/
theorem abs_clampB_le (B : ℝ) (hB : 0 ≤ B) (x : ℝ) : |clampB B x| ≤ B := by
  unfold clampB
  rw [abs_le]
  exact ⟨le_max_left _ _, max_le (by linarith) (min_le_left _ _)⟩

/-- The clamp is 1-Lipschitz on the line, from its decomposition into the two cuts
(`clampB_eq_min_max`) and the min/max difference bounds of `Algebra.Order.Group.MinMax`. -/
theorem clampB_lipschitz (B : ℝ) (hB : 0 ≤ B) (x y : ℝ) :
    |clampB B x - clampB B y| ≤ |x - y| := by
  rw [clampB_eq_min_max B hB x, clampB_eq_min_max B hB y]
  calc |min (max (-B) x) B - min (max (-B) y) B|
      ≤ max |max (-B) x - max (-B) y| |B - B| := abs_min_sub_min_le_max _ _ _ _
    _ = |max (-B) x - max (-B) y| := by rw [sub_self, abs_zero, max_eq_left (abs_nonneg _)]
    _ ≤ |x - y| := by
        calc |max (-B) x - max (-B) y|
            ≤ max |(-B) - (-B)| |x - y| := abs_max_sub_max_le_max (-B) x (-B) y
          _ = |x - y| := by rw [sub_self, abs_zero, max_eq_right (abs_nonneg _)]

/-- The clamp at the origin is the origin. -/
theorem clampB_zero (B : ℝ) (hB : 0 ≤ B) : clampB B 0 = 0 := by
  rw [clampB_eq_self (by rw [abs_zero]; exact hB)]

/-! ## The boxed gate product as a `Stage` -/

/-- The elementwise gate/up product with both factors clamped to the box `[-B, B]`, packaged as a
`Stage` from the pair of hidden-width vectors to the hidden-width vector. Its domain is the box
`{z | ∀ i, |z.1 i| ≤ B ∧ |z.2 i| ≤ B}`; its reference is `fun z i => clampB B (z.1 i) * clampB B
(z.2 i)`, globally Lipschitz with sup-norm constant `2 * B`; `exec` and its defect budget `delta` on
the box are caller-supplied, so a future kernel refinement of the gate product can replace
`mulBoxStage_exact`'s executing map without changing the reference or the bound. -/
noncomputable def mulBoxStage (h : ℕ) (B : ℝ) (hB : 0 ≤ B) (delta : ℝ) (hdelta : 0 ≤ delta)
    (exec : ((Fin h → ℝ) × (Fin h → ℝ)) → (Fin h → ℝ))
    (hdefect : ∀ z ∈ {z : (Fin h → ℝ) × (Fin h → ℝ) | ∀ i, |z.1 i| ≤ B ∧ |z.2 i| ≤ B},
        ‖exec z - (fun i => clampB B (z.1 i) * clampB B (z.2 i))‖ ≤ delta) :
    Stage ((Fin h → ℝ) × (Fin h → ℝ)) (Fin h → ℝ) where
  ref := fun z i => clampB B (z.1 i) * clampB B (z.2 i)
  exec := exec
  dom := {z | ∀ i, |z.1 i| ≤ B ∧ |z.2 i| ≤ B}
  delta := delta
  L := 2 * B
  delta_nonneg := hdelta
  L_nonneg := by linarith
  defect := hdefect
  expansion := by
    intro z w
    apply (pi_norm_le_iff_of_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ 2 * B)
      (norm_nonneg _))).2
    intro i
    rw [Real.norm_eq_abs]
    show |clampB B (z.1 i) * clampB B (z.2 i) - clampB B (w.1 i) * clampB B (w.2 i)|
        ≤ 2 * B * ‖z - w‖
    have hfst : ‖z.1 - w.1‖ ≤ ‖z - w‖ := by
      rw [show z - w = (z.1 - w.1, z.2 - w.2) from rfl, Prod.norm_mk]
      exact le_max_left _ _
    have hsnd : ‖z.2 - w.2‖ ≤ ‖z - w‖ := by
      rw [show z - w = (z.1 - w.1, z.2 - w.2) from rfl, Prod.norm_mk]
      exact le_max_right _ _
    have hc1 : |z.1 i - w.1 i| ≤ ‖z - w‖ := by
      have h := norm_le_pi_norm (z.1 - w.1) i
      rw [Real.norm_eq_abs, Pi.sub_apply] at h
      exact le_trans h hfst
    have hc2 : |z.2 i - w.2 i| ≤ ‖z - w‖ := by
      have h := norm_le_pi_norm (z.2 - w.2) i
      rw [Real.norm_eq_abs, Pi.sub_apply] at h
      exact le_trans h hsnd
    have h1 : |clampB B (z.1 i) - clampB B (w.1 i)| ≤ ‖z - w‖ :=
      le_trans (clampB_lipschitz B hB _ _) hc1
    have h2 : |clampB B (z.2 i) - clampB B (w.2 i)| ≤ ‖z - w‖ :=
      le_trans (clampB_lipschitz B hB _ _) hc2
    have ha : |clampB B (z.1 i)| ≤ B := abs_clampB_le B hB _
    have hb' : |clampB B (w.2 i)| ≤ B := abs_clampB_le B hB _
    have hmain : |clampB B (z.1 i) * clampB B (z.2 i) - clampB B (w.1 i) * clampB B (w.2 i)|
        ≤ B * |clampB B (z.1 i) - clampB B (w.1 i)|
          + B * |clampB B (z.2 i) - clampB B (w.2 i)| := by
      have hsplit : clampB B (z.1 i) * clampB B (z.2 i)
            - clampB B (w.1 i) * clampB B (w.2 i)
            = clampB B (z.1 i) * (clampB B (z.2 i) - clampB B (w.2 i))
              + (clampB B (z.1 i) - clampB B (w.1 i)) * clampB B (w.2 i) := by ring
      rw [hsplit]
      calc |clampB B (z.1 i) * (clampB B (z.2 i) - clampB B (w.2 i))
            + (clampB B (z.1 i) - clampB B (w.1 i)) * clampB B (w.2 i)|
          ≤ |clampB B (z.1 i) * (clampB B (z.2 i) - clampB B (w.2 i))|
            + |(clampB B (z.1 i) - clampB B (w.1 i)) * clampB B (w.2 i)| := abs_add_le _ _
        _ = |clampB B (z.1 i)| * |clampB B (z.2 i) - clampB B (w.2 i)|
            + |clampB B (z.1 i) - clampB B (w.1 i)| * |clampB B (w.2 i)| := by
            rw [abs_mul, abs_mul]
        _ ≤ B * |clampB B (z.2 i) - clampB B (w.2 i)|
            + |clampB B (z.1 i) - clampB B (w.1 i)| * B :=
            add_le_add (mul_le_mul_of_nonneg_right ha (abs_nonneg _))
              (mul_le_mul_of_nonneg_left hb' (abs_nonneg _))
        _ = B * |clampB B (z.1 i) - clampB B (w.1 i)|
            + B * |clampB B (z.2 i) - clampB B (w.2 i)| := by ring
    calc |clampB B (z.1 i) * clampB B (z.2 i) - clampB B (w.1 i) * clampB B (w.2 i)|
        ≤ B * |clampB B (z.1 i) - clampB B (w.1 i)|
          + B * |clampB B (z.2 i) - clampB B (w.2 i)| := hmain
      _ ≤ B * ‖z - w‖ + B * ‖z - w‖ :=
          add_le_add (mul_le_mul_of_nonneg_left h1 hB) (mul_le_mul_of_nonneg_left h2 hB)
      _ = 2 * B * ‖z - w‖ := by ring

/-- On its domain the boxed reference is the exact elementwise product: the box is fixed by the
clamps (`clampB_eq_self`). This is `BlockResidual.mul_ref_on` for `mulBoxStage`-based residuals. -/
theorem mulBoxStage_ref_on (h : ℕ) (B : ℝ) (hB : 0 ≤ B) (delta : ℝ) (hdelta : 0 ≤ delta)
    (exec : ((Fin h → ℝ) × (Fin h → ℝ)) → (Fin h → ℝ))
    (hdefect : ∀ z ∈ {z : (Fin h → ℝ) × (Fin h → ℝ) | ∀ i, |z.1 i| ≤ B ∧ |z.2 i| ≤ B},
        ‖exec z - (fun i => clampB B (z.1 i) * clampB B (z.2 i))‖ ≤ delta) :
    ∀ z ∈ (mulBoxStage h B hB delta hdelta exec hdefect).dom,
      (mulBoxStage h B hB delta hdelta exec hdefect).ref z = fun i => z.1 i * z.2 i := by
  intro z hz
  funext i
  show clampB B (z.1 i) * clampB B (z.2 i) = z.1 i * z.2 i
  rw [clampB_eq_self (hz i).1, clampB_eq_self (hz i).2]

/-- The canonical `mulBoxStage` instance: the executing map is the exact (unclamped) elementwise
product and the defect budget is `0`. On the box the clamps are the identity, so the exact product
coincides with the boxed reference there. -/
noncomputable def mulBoxStage_exact (h : ℕ) (B : ℝ) (hB : 0 ≤ B) :
    Stage ((Fin h → ℝ) × (Fin h → ℝ)) (Fin h → ℝ) :=
  mulBoxStage h B hB 0 (le_refl 0) (fun z i => z.1 i * z.2 i) (by
    intro z hz
    have hzero : (fun i : Fin h => z.1 i * z.2 i)
        - (fun i => clampB B (z.1 i) * clampB B (z.2 i)) = 0 := by
      funext i
      show z.1 i * z.2 i - clampB B (z.1 i) * clampB B (z.2 i) = 0
      rw [clampB_eq_self (hz i).1, clampB_eq_self (hz i).2, sub_self]
    rw [hzero, norm_zero])

@[simp] theorem mulBoxStage_exact_dom (h : ℕ) (B : ℝ) (hB : 0 ≤ B) :
    (mulBoxStage_exact h B hB).dom = {z : (Fin h → ℝ) × (Fin h → ℝ) |
      ∀ i, |z.1 i| ≤ B ∧ |z.2 i| ≤ B} := rfl

@[simp] theorem mulBoxStage_exact_exec (h : ℕ) (B : ℝ) (hB : 0 ≤ B)
    (z : (Fin h → ℝ) × (Fin h → ℝ)) :
    (mulBoxStage_exact h B hB).exec z = fun i => z.1 i * z.2 i := rfl

/-- `BlockResidual.mul_ref_on` for the canonical exact boxed product. -/
theorem mulBoxStage_exact_ref_on (h : ℕ) (B : ℝ) (hB : 0 ≤ B) :
    ∀ z ∈ (mulBoxStage_exact h B hB).dom,
      (mulBoxStage_exact h B hB).ref z = fun i => z.1 i * z.2 i := by
  intro z hz
  funext i
  show clampB B (z.1 i) * clampB B (z.2 i) = z.1 i * z.2 i
  rw [clampB_eq_self (hz i).1, clampB_eq_self (hz i).2]

/-! ## Inhabiting the residual pair -/

/-- The zero attention stage at projection width `p`: identically zero on the whole space, zero
defect and zero expansion factor. It carries no claim about attention. -/
noncomputable def attnZero (p : ℕ) :
    Stage ((Fin p → ℝ) × (Fin p → ℝ) × (Fin p → ℝ)) (Fin p → ℝ) where
  ref := fun _ => 0
  exec := fun _ => 0
  dom := Set.univ
  delta := 0
  L := 0
  delta_nonneg := le_refl 0
  L_nonneg := le_refl 0
  defect := by intro z _; simp
  expansion := by intro z w; simp

/-- A `BlockResidual` at any `(p, h)`: the zero attention stage and the exact boxed product at cap
`B ≥ 0`. -/
noncomputable def blockResidualZero (p h : ℕ) (B : ℝ) (hB : 0 ≤ B) : BlockResidual p h where
  attn := attnZero p
  mul := mulBoxStage_exact h B hB
  mul_ref_on := mulBoxStage_exact_ref_on h B hB

@[simp] theorem blockResidualZero_attn (p h : ℕ) (B : ℝ) (hB : 0 ≤ B) :
    (blockResidualZero p h B hB).attn = attnZero p := rfl

@[simp] theorem blockResidualZero_mul (p h : ℕ) (B : ℝ) (hB : 0 ≤ B) :
    (blockResidualZero p h B hB).mul = mulBoxStage_exact h B hB := rfl

/-- **Non-vacuity of `BlockResidual`.** For every projection width `p`, hidden width `h` and cap
`B ≥ 0`, `BlockResidual p h` is inhabited: zero attention and the exact boxed product. This is the
consequence the refuted global pin denied at every `1 ≤ h`. -/
theorem blockResidual_nonempty (p h : ℕ) (B : ℝ) (hB : 0 ≤ B) :
    Nonempty (BlockResidual p h) :=
  ⟨blockResidualZero p h B hB⟩

/-! ## The `d = p = h = 1` joint witness -/

/-- The box cap of the `d = p = h = 1` witness: the SiLU stage runs at cap `0`, so its executed value
at `0` is metrically confined to `siluErrTotal 0`. -/
abbrev B111 : ℝ := siluErrTotal 0

theorem B111_nonneg : 0 ≤ B111 := siluErrTotal_nonneg 0

/-- The `d = p = h = 1` witness block: the real `F16` `1 × 1` projection of
`Nous.Model.BlockGroundedWitness1` in all seven slots, every rotation switched off, both vector
norms at the in-window radicand `2^32` (`toReal = 1`), cap `0` and floor `1/2`, the SiLU cap `0`,
and the zero-attention / exact-boxed-product residual. -/
noncomputable def G111 : GroundedBlock 1 1 1 where
  q := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  k := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  v := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  o := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  g := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  u := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  dn := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  qR := rotId 1
  kR := rotId 1
  vR := rotId 1
  oR := rotId 1
  gR := rotId 1
  uR := rotId 1
  dnR := rotId 1
  br := blockResidualZero 1 1 B111 B111_nonneg
  hd := by norm_num
  eps1 := 4294967296#i64
  heps1 := by unfold toReal q32; norm_num
  B1 := 0
  A1 := 1 / 2
  hB1 := by norm_num
  hA1 := by unfold q32; norm_num
  eps2 := 4294967296#i64
  heps2 := by unfold toReal q32; norm_num
  B2 := 0
  A2 := 1 / 2
  hB2 := by norm_num
  hA2 := by unfold q32; norm_num
  Bs := 0
  hBs := le_refl 0

abbrev b111_rmsW1 : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  rmsVectorStage 1 G111.hd G111.eps1 G111.heps1 G111.B1 G111.A1 G111.hB1 G111.hA1

abbrev b111_rmsW2 : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  rmsVectorStage 1 G111.hd G111.eps2 G111.heps2 G111.B2 G111.A2 G111.hB2 G111.hA2

abbrev b111_attnSubW : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  attnSubBlockGrounded G111.q G111.k G111.v G111.o G111.qR G111.kR G111.vR G111.oR G111.br

abbrev b111_gateUp : Stage (Fin 1 → ℝ) ((Fin 1 → ℝ) × (Fin 1 → ℝ)) :=
  gateUpFanoutGrounded G111.g G111.u G111.gR G111.uR G111.Bs G111.hBs

abbrev b111_dnStage : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  rotProj G111.dnR G111.dn.spec

abbrev b111_ffnSubW : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  ffnSubBlockGrounded G111.g G111.u G111.dn G111.gR G111.uR G111.dnR G111.br G111.Bs G111.hBs

/-! ### The two vector norms at the zero row -/

theorem b111_zero_mem_rmsW1 : (0 : Fin 1 → ℝ) ∈ b111_rmsW1.dom := by
  have h : rowVec 1 z1 ∈ b111_rmsW1.dom := z1_mem_rms
  rwa [rowVec_z1] at h

theorem b111_zero_mem_rmsW2 : (0 : Fin 1 → ℝ) ∈ b111_rmsW2.dom := by
  have h : rowVec 1 z1 ∈ b111_rmsW2.dom := z1_mem_rms
  rwa [rowVec_z1] at h

theorem b111_rmsW1_exec_zero : b111_rmsW1.exec (0 : Fin 1 → ℝ) = 0 :=
  rmsVectorStage_exec_zero 1 (by norm_num) (4294967296#i64) (by unfold toReal q32; norm_num)
    (1 / 2) (by unfold q32; norm_num) b111_zero_mem_rmsW1

theorem b111_rmsW2_exec_zero : b111_rmsW2.exec (0 : Fin 1 → ℝ) = 0 :=
  rmsVectorStage_exec_zero 1 (by norm_num) (4294967296#i64) (by unfold toReal q32; norm_num)
    (1 / 2) (by unfold q32; norm_num) b111_zero_mem_rmsW2

/-! ### The real `F16` projections at the zero activation -/

theorem b111_projF16_dom_zero :
    (0 : Fin 1 → ℝ) ∈ G111.q.spec.stage.dom ∧ (0 : Fin 1 → ℝ) ∈ G111.k.spec.stage.dom ∧
      (0 : Fin 1 → ℝ) ∈ G111.v.spec.stage.dom ∧ (0 : Fin 1 → ℝ) ∈ G111.o.spec.stage.dom :=
  ⟨projF16_dom_zero _ _ _ _ _ f16WitP_dom, projF16_dom_zero _ _ _ _ _ f16WitP_dom,
    projF16_dom_zero _ _ _ _ _ f16WitP_dom, projF16_dom_zero _ _ _ _ _ f16WitP_dom⟩

theorem b111_projF16_dom_zero_g : (0 : Fin 1 → ℝ) ∈ G111.g.spec.stage.dom :=
  projF16_dom_zero _ _ _ _ _ f16WitP_dom

theorem b111_projF16_dom_zero_u : (0 : Fin 1 → ℝ) ∈ G111.u.spec.stage.dom :=
  projF16_dom_zero _ _ _ _ _ f16WitP_dom

theorem b111_projF16_dom_zero_dn : (0 : Fin 1 → ℝ) ∈ G111.dn.spec.stage.dom :=
  projF16_dom_zero _ _ _ _ _ f16WitP_dom

theorem b111_projF16_exec_zero_g : G111.g.spec.stage.exec (0 : Fin 1 → ℝ) = 0 :=
  projF16_exec_zero _ _ _ _ _ f16WitP_dom

theorem b111_projF16_exec_zero_u : G111.u.spec.stage.exec (0 : Fin 1 → ℝ) = 0 :=
  projF16_exec_zero _ _ _ _ _ f16WitP_dom

/-! ### The attention sub-block -/

theorem b111_zero_mem_attnSubW : (0 : Fin 1 → ℝ) ∈ b111_attnSubW.dom := by
  obtain ⟨hq, hk, hv, ho⟩ := b111_projF16_dom_zero
  refine ⟨⟨hq, hk, hv⟩, ?_⟩
  exact ⟨trivial, ho⟩

theorem b111_attnSubW_exec_zero : b111_attnSubW.exec (0 : Fin 1 → ℝ) = 0 := by
  have hb : G111.br.attn.exec
      ((qkvFanoutGrounded G111.q G111.k G111.v G111.qR G111.kR G111.vR).exec
        (0 : Fin 1 → ℝ)) = 0 := rfl
  change (rotProj G111.oR G111.o.spec).exec
    (G111.br.attn.exec ((qkvFanoutGrounded G111.q G111.k G111.v G111.qR G111.kR G111.vR).exec
      (0 : Fin 1 → ℝ))) = 0
  rw [hb]
  change G111.o.spec.stage.exec (0 : Fin 1 → ℝ) = 0
  exact projF16_exec_zero _ _ _ _ _ f16WitP_dom

/-! ### The SiLU branch at the origin -/

theorem b111_silu_mem : (0 : ℝ) ∈ (siluStage G111.Bs G111.hBs).dom := by
  have h := siluStage_dom_witness G111.Bs G111.hBs
  simpa [toReal_zero] using h

theorem b111_silu_bound : |(siluStage G111.Bs G111.hBs).exec (0 : ℝ)| ≤ B111 := by
  have h := (siluStage G111.Bs G111.hBs).defect (0 : ℝ) b111_silu_mem
  rw [siluStage_delta] at h
  have href : (siluStage G111.Bs G111.hBs).ref (0 : ℝ) = 0 := by
    show siluReal (0 : ℝ) = 0
    simp [siluReal]
  rw [href, sub_zero, Real.norm_eq_abs] at h
  exact h

theorem b111_gateUp_exec_zero_fst (i : Fin 1) :
    (b111_gateUp.exec (0 : Fin 1 → ℝ)).1 i = (siluStage G111.Bs G111.hBs).exec (0 : ℝ) := by
  show (siluStage G111.Bs G111.hBs).exec ((G111.g.spec.stage.exec (0 : Fin 1 → ℝ)) i)
      = (siluStage G111.Bs G111.hBs).exec (0 : ℝ)
  rw [b111_projF16_exec_zero_g]
  simp

theorem b111_gateUp_exec_zero_snd (i : Fin 1) :
    (b111_gateUp.exec (0 : Fin 1 → ℝ)).2 i = 0 := by
  show (G111.u.spec.stage.exec (0 : Fin 1 → ℝ)) i = 0
  rw [b111_projF16_exec_zero_u]
  simp

/-! ### The residual's product at the origin -/

@[simp] theorem b111_br_mul_exec_apply (z : (Fin 1 → ℝ) × (Fin 1 → ℝ)) (i : Fin 1) :
    G111.br.mul.exec z i = z.1 i * z.2 i := rfl

@[simp] theorem b111_br_mul_dom :
    G111.br.mul.dom = {z : (Fin 1 → ℝ) × (Fin 1 → ℝ) | ∀ i, |z.1 i| ≤ B111 ∧ |z.2 i| ≤ B111} := rfl

/-! ### The FFN sub-block -/

theorem b111_zero_mem_ffnSubW : (0 : Fin 1 → ℝ) ∈ b111_ffnSubW.dom := by
  refine ⟨?_, ?_⟩
  · -- the gate/up fan-out domain: both branches at the origin
    refine ⟨?_, ?_⟩
    · -- gate branch: the projection at `0` then the origin in the SiLU grid
      refine ⟨b111_projF16_dom_zero_g, ?_⟩
      change (G111.g.spec.stage.exec (0 : Fin 1 → ℝ)) ∈ (Stage.pi (siluStage G111.Bs G111.hBs)).dom
      rw [b111_projF16_exec_zero_g]
      intro i
      simpa using b111_silu_mem
    · -- up branch: the projection at `0`
      change (0 : Fin 1 → ℝ) ∈ G111.u.spec.stage.dom
      exact b111_projF16_dom_zero_u
  · -- the gate/up fan-out value lands in the boxed product's composed domain
    refine ⟨?_, ?_⟩
    · rw [b111_br_mul_dom]
      intro i
      constructor
      · rw [b111_gateUp_exec_zero_fst i]; exact b111_silu_bound
      · rw [b111_gateUp_exec_zero_snd i, abs_zero]; exact B111_nonneg
    · have hmul : G111.br.mul.exec (b111_gateUp.exec (0 : Fin 1 → ℝ)) = 0 := by
        funext i
        rw [b111_br_mul_exec_apply, b111_gateUp_exec_zero_snd i, mul_zero]
        simp
      rw [hmul]
      change (0 : Fin 1 → ℝ) ∈ G111.dn.spec.stage.dom
      exact b111_projF16_dom_zero_dn

/-! ### The joint witness -/

/-- **Joint non-vacuity of the grounded block at `d = p = h = 1`.** The zero input is in
`fullBlockGrounded G111`'s composed domain: it is in the first norm's domain, the first norm and the
attention sub-block map it to `0`, so the residual returns it, it is in the second norm's domain, and
the second norm's image `0` lies in the FFN sub-block's domain (the gate branch's executed SiLU value
at `0` is boxed by `siluErrTotal 0`, the up branch and the exact product return `0`). -/
theorem fullBlockGrounded_dom_nonempty_111 :
    (0 : Fin 1 → ℝ) ∈ (fullBlockGrounded G111).dom := by
  have hx1 : (0 : Fin 1 → ℝ) ∈ b111_rmsW1.dom := b111_zero_mem_rmsW1
  have hexec1 : b111_rmsW1.exec (0 : Fin 1 → ℝ) = 0 := b111_rmsW1_exec_zero
  have hattn : (0 : Fin 1 → ℝ) ∈ b111_attnSubW.dom := b111_zero_mem_attnSubW
  have hattnexec : b111_attnSubW.exec (0 : Fin 1 → ℝ) = 0 := b111_attnSubW_exec_zero
  have hx2 : (0 : Fin 1 → ℝ) ∈ b111_rmsW2.dom := b111_zero_mem_rmsW2
  have hexec2 : b111_rmsW2.exec (0 : Fin 1 → ℝ) = 0 := b111_rmsW2_exec_zero
  have hffn : (0 : Fin 1 → ℝ) ∈ b111_ffnSubW.dom := b111_zero_mem_ffnSubW
  have hs1dom : (0 : Fin 1 → ℝ) ∈ (b111_rmsW1.comp b111_attnSubW).dom :=
    ⟨hx1, by rw [hexec1]; exact hattn⟩
  have hs1exec : (b111_rmsW1.comp b111_attnSubW).exec (0 : Fin 1 → ℝ) = 0 := by
    show b111_attnSubW.exec (b111_rmsW1.exec (0 : Fin 1 → ℝ)) = 0
    rw [hexec1]; exact hattnexec
  have hres1 : (Stage.residual (b111_rmsW1.comp b111_attnSubW)).exec (0 : Fin 1 → ℝ) = 0 := by
    show (0 : Fin 1 → ℝ) + (b111_rmsW1.comp b111_attnSubW).exec (0 : Fin 1 → ℝ) = 0
    rw [hs1exec, add_zero]
  have hs2dom : (Stage.residual (b111_rmsW1.comp b111_attnSubW)).exec (0 : Fin 1 → ℝ) ∈
      (Stage.residual (b111_rmsW2.comp b111_ffnSubW)).dom := by
    rw [hres1]
    exact ⟨hx2, by rw [hexec2]; exact hffn⟩
  change (0 : Fin 1 → ℝ) ∈ ((Stage.residual (b111_rmsW1.comp b111_attnSubW)).comp
    (Stage.residual (b111_rmsW2.comp b111_ffnSubW))).dom
  exact ⟨hs1dom, hs2dom⟩

end

end Nous.Model
