/-
# Grid stages with explicit admissible domains: SiLU and scalar RMSNorm

`Nous.Model.BlockComposition` builds `Stage E F` on an explicit domain and proves the composed
domain-propagation obligation, but the two Rust-grounded per-coordinate stages enter through
`Stage.ofLayer` (`BlockComposition.lean:202`) with `dom = Set.univ`. That is sound for
`siluLayer`/`rmsLayer` themselves only because `Layer.ofKernel`
(`Nous/Model/ErrorComposition.lean:352-393`) substitutes `ref` off the kernel's admissible domain:
the resulting `Stage` claims a defect bound at points the extracted body never executes, and
`Stage.comp` therefore never sees the admissible-domain obligation. This module restores it.

## `Stage.ofGrid`

`Stage.ofGrid γ ref g adm delta L` is the `Stage E E` whose `dom` is exactly the image of the
admissible predicate under the grid embedding, `{z | ∃ v, γ v = z ∧ adm v}`, and whose `exec` is the
choice-based `g v` on `dom` and `ref` off it (mirroring `Layer.ofKernel`'s device, but with the
domain kept explicit). `defect` is proved on `dom` from a `hbound` that is assumed there only.

## `siluStage` and `rmsStage`

* `siluStage B hB` uses the **total** SiLU refinement `silu_refines_total`
  (`Numeric/SiluSaturation.lean:150`): every `I64` is covered, so no saturation arm is excluded. Its
  domain is `{|toReal v| ≤ B}`, its budget `siluErrTotal B` (using that `siluErrTotal` is monotone in
  `|x|`), and its expansion factor `3` (`siluReal_lipschitz`).
* `rmsStage a ha B hB` is the scalar RMSNorm stage at the externally fixed radicand `a`: `ref` is
  `rmsRef a`, `exec` is the choice-based `rmsGrid (rmsRoot a ha)`, and `adm` is
  `rmsDom (rmsRoot a ha) B` — the magnitude bound plus the epilogue representability condition, i.e.
  the grid intersected with the domain of `rmsLayer`, **not** `Set.univ`.

## `fullBlock'`

`fullBlock'` replaces every `Stage.pi (Stage.ofLayer siluLayer)` by `Stage.pi (siluStage B hB)` and
each `Stage.pi (Stage.ofLayer (rmsLayer …))` by `Stage.pi (rmsStage …)`. `block_error_bound'` has the
same shape as `block_error_bound` with the new per-stage budgets, now on the composed
`fullBlock'.dom`; the propagation obligation is visible in the type and enters `Stage.comp` rather
than being hidden off-domain.

## Scope

RMSNorm here remains the scalar per-coordinate stage at an externally fixed radicand `a` — the
`a1`/`a2` hypotheses the block already carries. The vector RMSNorm whose `a` is the rounded mean of
the input is the next residual; `Numeric/RmsNormIdeal.lean` states the mean refinement.

## Trust boundary

As in `BlockComposition`: the per-stage bounds are kernel-grounded only through the reused
`silu_refines_total`/`rmsElement_bound` chains. No claim is made about rustc/LLVM codegen, the kernel
ABI, GPU/SPIR-V, the `unsafe` container, or the dequant-to-ideal-weight reading. The grid-stage
algebra here is checked at the abstract real level under the closed axiom cone.
-/
import Nous.Model.BlockComposition
import Numeric.SiluSaturation

namespace Nous.Model

open apeiron_word Aeneas Aeneas.Std Result Classical

noncomputable section

variable {E : Type*} [NormedAddCommGroup E]

namespace Stage

/-- A grid-indexed per-operation refinement realized as a `Stage E E` with an explicit domain. `γ`
embeds the executing grid into the carrier `E`; `g` is the real denotation of the extracted output;
`adm` is the operation's admissible predicate on the grid. The domain is exactly the image
`{z | ∃ v, γ v = z ∧ adm v}`; `exec` is the choice-based `g v` on the domain and `ref` off it, so
`defect` is only ever required where the kernel executes. Unlike `Layer.ofKernel`
(`ErrorComposition.lean:352`), which realizes the same map as a total `Layer`, the admissible domain
is carried in the `Stage` and reaches `Stage.comp` as a propagation obligation. -/
def ofGrid {ι : Type*} (γ : ι → E) (ref : E → E) (g : ι → E) (adm : ι → Prop)
    (delta L : ℝ) (hδ : 0 ≤ delta) (hL : 0 ≤ L)
    (hbound : ∀ v, adm v → ‖g v - ref (γ v)‖ ≤ delta)
    (hlip : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖) : Stage E E :=
  let exec : E → E :=
    fun z => if h : ∃ v, γ v = z ∧ adm v then g (Classical.choose h) else ref z
  { ref := ref
    exec := exec
    dom := {z | ∃ v, γ v = z ∧ adm v}
    delta := delta
    L := L
    delta_nonneg := hδ
    L_nonneg := hL
    defect := by
      intro z hz
      show ‖exec z - ref z‖ ≤ delta
      dsimp only [exec]
      have hz' : ∃ v, γ v = z ∧ adm v := hz
      rw [dif_pos hz']
      obtain ⟨hγ, hd⟩ := Classical.choose_spec hz'
      have hgoal : ‖g (Classical.choose hz') - ref z‖
          = ‖g (Classical.choose hz') - ref (γ (Classical.choose hz'))‖ := by rw [hγ]
      rw [hgoal]
      exact hbound _ hd
    expansion := hlip }

/-- `ofGrid`'s domain is exactly the image of `adm` under `γ`. -/
theorem ofGrid_dom {ι : Type*} (γ : ι → E) (ref : E → E) (g : ι → E) (adm : ι → Prop)
    (delta L : ℝ) (hδ : 0 ≤ delta) (hL : 0 ≤ L)
    (hbound : ∀ v, adm v → ‖g v - ref (γ v)‖ ≤ delta)
    (hlip : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖) :
    (ofGrid γ ref g adm delta L hδ hL hbound hlip).dom = {z | ∃ v, γ v = z ∧ adm v} := rfl

@[simp] theorem ofGrid_delta {ι : Type*} (γ : ι → E) (ref : E → E) (g : ι → E) (adm : ι → Prop)
    (delta L : ℝ) (hδ : 0 ≤ delta) (hL : 0 ≤ L)
    (hbound : ∀ v, adm v → ‖g v - ref (γ v)‖ ≤ delta)
    (hlip : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖) :
    (ofGrid γ ref g adm delta L hδ hL hbound hlip).delta = delta := rfl

@[simp] theorem ofGrid_L {ι : Type*} (γ : ι → E) (ref : E → E) (g : ι → E) (adm : ι → Prop)
    (delta L : ℝ) (hδ : 0 ≤ delta) (hL : 0 ≤ L)
    (hbound : ∀ v, adm v → ‖g v - ref (γ v)‖ ≤ delta)
    (hlip : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖) :
    (ofGrid γ ref g adm delta L hδ hL hbound hlip).L = L := rfl

end Stage

/-! ## The total SiLU ledger -/

/-- The total SiLU error target is nonnegative. -/
theorem siluErrTotal_nonneg (x : ℝ) : 0 ≤ siluErrTotal x := by
  have h1 : (0 : ℝ) ≤ epsExp := epsExp_nonneg
  have h2 : (0 : ℝ) ≤ 4 / q32 := div_nonneg (by norm_num) q32_nonneg
  have h3 : (0 : ℝ) ≤ 1 / (2 * q32) := div_nonneg (by norm_num) (by linarith [q32_pos])
  have hx : (0 : ℝ) ≤ |x| := abs_nonneg x
  have hX : (0 : ℝ) ≤ epsExp * |x| + (4 / q32) * |x| + 1 / (2 * q32) := by
    have ha := mul_nonneg h1 hx
    have hb := mul_nonneg h2 hx
    linarith
  unfold siluErrTotal
  exact le_trans hX (le_max_left _ _)

/-- `siluErrTotal` is monotone in `|x|`: the scaled term only grows with the magnitude and the
saturation constant is independent of the input. -/
theorem siluErrTotal_mono {x B : ℝ} (hB : 0 ≤ B) (hx : |x| ≤ B) :
    siluErrTotal x ≤ siluErrTotal B := by
  have h1 : (0 : ℝ) ≤ epsExp := epsExp_nonneg
  have h2 : (0 : ℝ) ≤ 4 / q32 := div_nonneg (by norm_num) q32_nonneg
  have hp1 : epsExp * |x| ≤ epsExp * B := mul_le_mul_of_nonneg_left hx h1
  have hp2 : (4 / q32) * |x| ≤ (4 / q32) * B := mul_le_mul_of_nonneg_left hx h2
  have hX : epsExp * |x| + (4 / q32) * |x| + 1 / (2 * q32)
      ≤ epsExp * B + (4 / q32) * B + 1 / (2 * q32) := by linarith
  have hBabs : |B| = B := abs_of_nonneg hB
  unfold siluErrTotal
  rw [hBabs]
  exact max_le (le_trans hX (le_max_left _ _)) (le_max_right _ _)

theorem toReal_zero : toReal (0#i64) = 0 := by
  unfold toReal
  norm_num

/-- The total-domain SiLU stage: the choice-based extracted output of `silu_refines_total`, with
admissible domain `{|toReal v| ≤ B}`, defect budget `siluErrTotal B` and expansion factor `3`.
Because `silu_refines_total` covers every `I64`, no saturation arm is excluded: this is the total
refinement, not the window-restricted `siluLayer`. -/
noncomputable def siluStage (B : ℝ) (hB : 0 ≤ B) : Stage ℝ ℝ :=
  Stage.ofGrid (ι := I64) (E := ℝ) toReal siluReal
    (fun v : I64 => toReal (Classical.choose (silu_refines_total v)))
    (fun v : I64 => |toReal v| ≤ B)
    (siluErrTotal B) 3
    (siluErrTotal_nonneg B) (by norm_num)
    (fun v hv => by
      have h := (Classical.choose_spec (silu_refines_total v)).2
      have hmono : siluErrTotal (toReal v) ≤ siluErrTotal B := siluErrTotal_mono hB hv
      rw [Real.norm_eq_abs]
      exact le_trans h hmono)
    siluReal_lipschitz

@[simp] theorem siluStage_delta (B : ℝ) (hB : 0 ≤ B) :
    (siluStage B hB).delta = siluErrTotal B := rfl

@[simp] theorem siluStage_L (B : ℝ) (hB : 0 ≤ B) : (siluStage B hB).L = 3 := rfl

/-- Non-vacuity: the origin of the grid is admissible for every `B ≥ 0`, so `siluStage`'s domain is
never empty. -/
theorem siluStage_dom_witness (B : ℝ) (hB : 0 ≤ B) :
    toReal 0#i64 ∈ (siluStage B hB).dom := by
  refine ⟨0#i64, rfl, ?_⟩
  show |toReal 0#i64| ≤ B
  rw [toReal_zero, abs_zero]
  exact hB

/-! ## The scalar RMSNorm stage -/

/-- `rmsDelta` is nonnegative at every magnitude bound. -/
theorem rmsDelta_nonneg (a : I64) (B : ℝ) (hB : 0 ≤ B) : 0 ≤ rmsDelta a B := by
  unfold rmsDelta
  exact add_nonneg (mul_nonneg (rmsEps_nonneg a) hB) rmsFloor_nonneg

/-- The scalar RMSNorm stage at the externally fixed radicand `a`. `ref` is `rmsRef a`; `exec` is
the choice-based epilogue output `rmsGrid (rmsRoot a ha)`; the admissible domain is
`rmsDom (rmsRoot a ha) B` — the magnitude bound together with the epilogue representability
condition, i.e. the grid intersected with `rmsLayer`'s own domain, not `Set.univ`. Defect budget
`rmsDelta a B`, expansion factor `1/√(toReal a)` (`rmsRef_lipschitz`). -/
noncomputable def rmsStage (a : I64) (ha : rsqrt_window a) (B : ℝ) (hB : 0 ≤ B) :
    Stage ℝ ℝ :=
  Stage.ofGrid (ι := I64) (E := ℝ) toReal (rmsRef a) (rmsGrid (rmsRoot a ha))
    (fun v : I64 => rmsDom (rmsRoot a ha) B v)
    (rmsDelta a B) (1 / Real.sqrt (toReal a))
    (rmsDelta_nonneg a B hB) (div_nonneg (by norm_num) (Real.sqrt_nonneg _))
    (fun x hx => by
      have h := rmsGrid_bound a (rmsRoot a ha) ha rfl B x hx
      have hm : rmsEps a * ‖toReal x‖ + rmsFloor ≤ rmsDelta a B := by
        unfold rmsDelta
        exact add_le_add (mul_le_mul_of_nonneg_left hx.1 (rmsEps_nonneg a)) le_rfl
      exact le_trans h hm)
    (rmsRef_lipschitz a)

@[simp] theorem rmsStage_delta (a : I64) (ha : rsqrt_window a) (B : ℝ) (hB : 0 ≤ B) :
    (rmsStage a ha B hB).delta = rmsDelta a B := rfl

@[simp] theorem rmsStage_L (a : I64) (ha : rsqrt_window a) (B : ℝ) (hB : 0 ≤ B) :
    (rmsStage a ha B hB).L = 1 / Real.sqrt (toReal a) := rfl

/-- The concrete in-window radicand `a = 2^32`, whose Q32.32 value is `1`. -/
def rmsA : I64 := 4294967296#i64

theorem rmsA_window : rsqrt_window rmsA := by
  unfold rmsA rsqrt_window
  norm_num

theorem divRoundNearest_zero : divRoundNearest 0 ((2 : ℤ) ^ (32 : ℕ)) = 0 := by
  unfold divRoundNearest
  norm_num

/-- Non-vacuity: for the concrete in-window radicand `2^32` the origin of the grid is admissible
(magnitude `0 ≤ B`, epilogue `0 ↦ 0`), so `rmsStage`'s domain is never empty. -/
theorem rmsStage_dom_witness (B : ℝ) (hB : 0 ≤ B) :
    toReal 0#i64 ∈ (rmsStage rmsA rmsA_window B hB).dom := by
  refine ⟨0#i64, rfl, ?_⟩
  show rmsDom (rmsRoot rmsA rmsA_window) B 0#i64
  refine ⟨?_, ?_⟩
  · show ‖toReal 0#i64‖ ≤ B
    rw [Real.norm_eq_abs, toReal_zero, abs_zero]
    exact hB
  · refine ⟨0#i64, ?_⟩
    show (0#i64).val
      = divRoundNearest ((0#i64).val * (rmsRoot rmsA rmsA_window).val) ((2 : ℤ) ^ (32 : ℕ))
    have hz : (0#i64).val = 0 := by norm_num
    rw [hz, zero_mul, divRoundNearest_zero]

/-! ## The block with explicit grid domains -/

variable {d p h : ℕ}

/-- The FFN sub-block with the total-domain SiLU stage in place of `Stage.ofLayer siluLayer`: the
`(SiLU(gate), up)` fan-out uses `Stage.pi (siluStage B hB)`, so the SiLU admissible domain is the
`Stage.pi`-lifted `{|toReal v| ≤ B}` rather than `Set.univ`. -/
noncomputable def ffnSubBlock' (gs us : MatvecSpec d h) (dws : MatvecSpec h d)
    (br : BlockResidual p h) (B : ℝ) (hB : 0 ≤ B) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  ((gs.stage.comp (Stage.pi (siluStage B hB))).prod us.stage).comp
    (br.mul.comp dws.stage)

/-- The pre-norm transformer block with explicit grid domains: the same assembly as `fullBlock`,
with `Stage.pi (siluStage Bs hBs)` in the FFN SiLU slot and `Stage.pi (rmsStage …)` at both norm
sites. `a1`/`a2` remain the two externally fixed RMSNorm radicands. -/
noncomputable def fullBlock' (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (gs us : MatvecSpec d h) (dws : MatvecSpec h d) (br : BlockResidual p h)
    (a1 : I64) (ha1 : rsqrt_window a1) (B1 : ℝ) (hB1 : 0 ≤ B1)
    (a2 : I64) (ha2 : rsqrt_window a2) (B2 : ℝ) (hB2 : 0 ≤ B2)
    (Bs : ℝ) (hBs : 0 ≤ Bs) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  (Stage.residual ((Stage.pi (rmsStage a1 ha1 B1 hB1)).comp
      (attnSubBlock qs ks vs os br))).comp
    (Stage.residual ((Stage.pi (rmsStage a2 ha2 B2 hB2)).comp
      (ffnSubBlock' gs us dws br Bs hBs)))

/-- **Block error bound with explicit grid domains.** On `fullBlock'`'s composed domain — the
domain-propagation obligation of every internal `Stage.comp`/`Stage.prod`/`Stage.residual`/
`Stage.pi`, now carrying the SiLU and RMSNorm admissible domains — the executing block differs from
the reference block by at most `fullBlock'`'s `delta`. -/
theorem block_error_bound' (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (gs us : MatvecSpec d h) (dws : MatvecSpec h d) (br : BlockResidual p h)
    (a1 : I64) (ha1 : rsqrt_window a1) (B1 : ℝ) (hB1 : 0 ≤ B1)
    (a2 : I64) (ha2 : rsqrt_window a2) (B2 : ℝ) (hB2 : 0 ≤ B2)
    (Bs : ℝ) (hBs : 0 ≤ Bs) (x : Fin d → ℝ)
    (hx : x ∈ (fullBlock' qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2 Bs hBs).dom) :
    ‖(fullBlock' qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2 Bs hBs).exec x
        - (fullBlock' qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2 Bs hBs).ref x‖
      ≤ (fullBlock' qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2 Bs hBs).delta :=
  (fullBlock' qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2 Bs hBs).defect x hx

/-- One-step unfold of `fullBlock'.delta`, mirroring `fullBlock_delta_eq`: the same
`block2.delta + block2.L * block1.delta` recurrence, now with the `rmsStage` and
`ffnSubBlock'` sub-blocks. -/
theorem fullBlock'_delta_eq (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (gs us : MatvecSpec d h) (dws : MatvecSpec h d) (br : BlockResidual p h)
    (a1 : I64) (ha1 : rsqrt_window a1) (B1 : ℝ) (hB1 : 0 ≤ B1)
    (a2 : I64) (ha2 : rsqrt_window a2) (B2 : ℝ) (hB2 : 0 ≤ B2)
    (Bs : ℝ) (hBs : 0 ≤ Bs) :
    (fullBlock' qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2 Bs hBs).delta =
      (Stage.residual ((Stage.pi (rmsStage a2 ha2 B2 hB2)).comp
          (ffnSubBlock' gs us dws br Bs hBs))).delta
        + (Stage.residual ((Stage.pi (rmsStage a2 ha2 B2 hB2)).comp
              (ffnSubBlock' gs us dws br Bs hBs))).L
          * (Stage.residual ((Stage.pi (rmsStage a1 ha1 B1 hB1)).comp
              (attnSubBlock qs ks vs os br))).delta := rfl

/-- The `(SiLU(gate), up)` fan-out's own `delta` with the grid SiLU stage: `max` of the gate
branch's `siluErrTotal B + 3 * matvecEps` and the up branch's `matvecEps`. -/
theorem ffn_fanout_delta_eq' (gs us : MatvecSpec d h) (B : ℝ) (hB : 0 ≤ B) :
    ((gs.stage.comp (Stage.pi (siluStage B hB))).prod us.stage).delta
      = max (siluErrTotal B + 3 * matvecEps) matvecEps := rfl

/-- One-step unfold of the FFN sub-block's `delta` with the grid SiLU stage, closed in the leaf
quantities, mirroring `ffnSubBlock_delta_eq`. -/
theorem ffnSubBlock_delta_eq' (gs us : MatvecSpec d h) (dws : MatvecSpec h d)
    (br : BlockResidual p h) (B : ℝ) (hB : 0 ≤ B) :
    (ffnSubBlock' gs us dws br B hB).delta =
      (matvecEps + dws.rowBound * br.mul.delta)
        + (dws.rowBound * br.mul.L) * max (siluErrTotal B + 3 * matvecEps) matvecEps := rfl

end

end Nous.Model
