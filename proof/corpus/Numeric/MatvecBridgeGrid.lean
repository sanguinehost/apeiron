/-
# The matvec bridge over every admissible grid input (`APEIRON-MODEL-W4-AI8`)

`Numeric/MatvecBridge.lean` links the extracted fixed-point projection kernel to the abstract
`Stage.matvec` interface on one executed activation: `matvecSpecOf` has `dom = {gridVec x}` for a
single `x : Slice I64`. This module lifts that construction to the whole admissible grid.

* `gridVec_injective` — `gridVec` is injective on slices of length `n`: the Q32.32 reading of the
  full `n`-coordinate window determines the slice, because `toReal` is injective.
* `GridDom`, `gridInv` — the domain `{gridVec x | matvec_domain … x …}` and its total inverse
  reading, with `gridInv_spec` the chosen witness and `gridInv_gridVec` the left-inverse law on the
  domain.
* `matvecSpecGrid` — a `MatvecSpec n_in n_out` with the same `W = deqMatrix deq n_out n_in`, the
  image domain, and `exec` the extracted output of the recovered admissible slice. Its `hdefect` is
  proved for every domain point from `matvec_refines_proved` through `rowDotReal_eq_matvecRef`.
* `matvecSpecGrid_dom_nonempty` — the `F16` witness of `Numeric/DequantRefinement.lean` inhabits it.

Trust boundary: the dequant-to-ideal-weight reading stays the class-(b) `tested` boundary of
`Numeric/EngineMatvecRefinement.lean`, not re-examined here; no claim about rustc/LLVM codegen, the
kernel ABI, GPU/SPIR-V or the `unsafe` container.
-/
import Numeric.MatvecBridge
import Numeric.DequantRefinement

open Aeneas Aeneas.Std Result
open scoped BigOperators

namespace MatvecBridgeGrid

noncomputable section

open Nous.Model
open apeiron_word
open MatvecBridge

/-! ## Injectivity of the grid reading -/

/-- `toReal` is injective: two Q32.32 words with the same real value are equal. -/
theorem toReal_injective : Function.Injective toReal := by
  intro a b h
  unfold toReal at h
  have h2 : (a.val : ℝ) = (b.val : ℝ) := by
    rw [div_eq_div_iff q32_ne_zero q32_ne_zero] at h
    exact mul_right_cancel₀ q32_ne_zero h
  exact IScalar.eq_of_val_eq (Int.cast_injective h2)

/-- `gridVec` is injective on slices of length `n`. -/
theorem gridVec_injective {n : ℕ} {x₁ x₂ : Slice I64}
    (h₁ : x₁.val.length = n) (h₂ : x₂.val.length = n)
    (h : gridVec (n := n) x₁ = gridVec (n := n) x₂) : x₁ = x₂ := by
  apply Slice.ext
  apply List.ext_getElem
  · rw [h₁, h₂]
  · intro i hi1 hi2
    have hi : i < n := by rw [← h₁]; exact hi1
    have hfun := congrFun h (⟨i, hi⟩ : Fin n)
    have e1 : x₁.val.getD i 0#i64 = x₁.val[i]'hi1 :=
      List.getD_eq_getElem (l := x₁.val) (d := 0#i64) (n := i) hi1
    have e2 : x₂.val.getD i 0#i64 = x₂.val[i]'hi2 :=
      List.getD_eq_getElem (l := x₂.val) (d := 0#i64) (n := i) hi2
    simp only [gridVec] at hfun
    rw [e1, e2] at hfun
    exact toReal_injective hfun

/-! ## The grid domain and its inverse -/

/-- The admissible grid domain: the `gridVec` images of the slices satisfying `matvec_domain`. -/
def GridDom (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) : Set (Fin n_in.val → ℝ) :=
  {y | ∃ x : Slice I64,
      matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq ∧
        gridVec (n := n_in.val) x = y}

/-- The total inverse reading: on the grid domain the chosen admissible slice, off it the empty
slice. -/
noncomputable def gridInv (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) :
    (Fin n_in.val → ℝ) → Slice I64 := by
  classical
  exact fun y =>
    if h : y ∈ GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq
    then Classical.choose h
    else Slice.new I64

/-- The chosen slice of `gridInv` is admissible and reads back to the input. -/
theorem gridInv_spec (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (y : Fin n_in.val → ℝ)
    (hy : y ∈ GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq) :
    matvec_domain ggml_type raw n_out n_in
        (gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y)
        qk bb blocks row_bytes need deq ∧
      gridVec (n := n_in.val)
        (gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y) = y := by
  unfold gridInv
  rw [dif_pos hy]
  exact Classical.choose_spec hy

/-- On the grid the inverse is a left inverse: `gridVec` is injective on length-`n` slices, so the
chosen admissible slice with the same image is the original. -/
theorem gridInv_gridVec (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) (x : Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq
        (gridVec (n := n_in.val) x) = x := by
  have hy : gridVec (n := n_in.val) x ∈
      GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq := ⟨x, hdom, rfl⟩
  have hspec := gridInv_spec ggml_type raw n_out n_in qk bb blocks row_bytes need deq
    (gridVec (n := n_in.val) x) hy
  exact gridVec_injective hspec.1.2.1 hdom.2.1 hspec.2

/-! ## The chosen extracted output -/

/-- The output chosen by `matvec_refines_proved` for an admissible slice. -/
noncomputable def gridOut (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) (x : Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    alloc.vec.Vec I64 :=
  Classical.choose
    (matvec_refines_proved ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom)

/-- The chosen output has the `matvec_domain` length, and its `r`-th Q32.32 value is within
`1/(2·q32)` of the real dot of the dequantized row with the slice. -/
theorem gridOut_spec (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) (x : Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    (gridOut ggml_type raw n_out n_in qk bb blocks row_bytes need deq x hdom).val.length =
        n_out.val ∧
      ∀ (r : Nat) (hr : r <
          (gridOut ggml_type raw n_out n_in qk bb blocks row_bytes need deq x hdom).val.length),
        |toReal ((gridOut ggml_type raw n_out n_in qk bb blocks row_bytes need deq x hdom).val[r]'hr)
            - rowDotReal (deq r) x| ≤ 1 / (2 * q32) :=
  (Classical.choose_spec
    (matvec_refines_proved ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom)).2

/-- The executing map on the grid: recover the admissible slice from `y` and take its chosen
extracted output; `0` off the domain. -/
noncomputable def gridExec (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) :
    (Fin n_in.val → ℝ) → (Fin n_out.val → ℝ) := by
  classical
  exact fun y =>
    if hy : y ∈ GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq
    then matvecExecVec
      (gridOut ggml_type raw n_out n_in qk bb blocks row_bytes need deq
        (gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y)
        (gridInv_spec ggml_type raw n_out n_in qk bb blocks row_bytes need deq y hy).1)
      n_out.val
    else 0

/-! ## The spec over the whole grid -/

/-- **The matvec spec over the whole admissible grid.** Same `W` as `matvecSpecOf`, domain the
`gridVec` image of `matvec_domain`, `exec` the recovered slice's chosen extracted output, and a
proved `matvecEps` defect on every domain point. -/
noncomputable def matvecSpecGrid (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) :
    MatvecSpec n_in.val n_out.val :=
  { W := deqMatrix deq n_out.val n_in.val
    exec := gridExec ggml_type raw n_out n_in qk bb blocks row_bytes need deq
    dom := GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq
    rowBound := maxRowBound (deqMatrix deq n_out.val n_in.val)
    rowBound_nonneg := maxRowBound_nonneg _
    hrow := fun i => le_maxRowBound _ i
    hdefect := by
      intro y hy i
      have hmem : y ∈ GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq := hy
      obtain ⟨x, hdom, hgrid⟩ := hy
      have hInv := gridInv_spec ggml_type raw n_out n_in qk bb blocks row_bytes need deq y hmem
      have hxeq : gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y = x :=
        gridVec_injective hInv.1.2.1 hdom.2.1 (by rw [hInv.2, hgrid])
      have hdeq : ∀ i : Fin n_out.val, (deq i.val).val.length = n_in.val :=
        fun i => hInv.1.2.2.2.2.2.2.1 i.val i.isLt
      have hrow : rowDotReal (deq i.val)
          (gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y) =
          Stage.matvecRef (deqMatrix deq n_out.val n_in.val) y i := by
        rw [hxeq]
        rw [rowDotReal_eq_matvecRef deq n_out.val n_in.val x hdeq hdom.2.1 i]
        rw [hgrid]
      have hspec := gridOut_spec ggml_type raw n_out n_in qk bb blocks row_bytes need deq
        (gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y) hInv.1
      have hlen := hspec.1
      have hb := hspec.2 i.val (by rw [hlen]; exact i.isLt)
      have hle : (1 : ℝ) / (2 * q32) ≤ matvecEps := by unfold matvecEps q32; norm_num
      simp only [gridExec]
      rw [dif_pos hmem]
      simp only [matvecExecVec]
      rw [← hrow]
      rw [List.getD_eq_getElem
        (l := (gridOut ggml_type raw n_out n_in qk bb blocks row_bytes need deq
                (gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y) hInv.1).val)
        (d := 0#i64) (n := i.val) (by rw [hlen]; exact i.isLt)]
      exact le_trans hb hle }

@[simp] theorem matvecSpecGrid_W (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) :
    (matvecSpecGrid ggml_type raw n_out n_in qk bb blocks row_bytes need deq).W =
      deqMatrix deq n_out.val n_in.val := rfl

@[simp] theorem matvecSpecGrid_dom (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) :
    (matvecSpecGrid ggml_type raw n_out n_in qk bb blocks row_bytes need deq).dom =
      GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq := rfl

/-- **Grid stage defect.** The `Stage` of `matvecSpecGrid` has defect at most `matvecEps` on its
domain. -/
theorem matvec_stage_grid_proved (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) :
    ∀ z ∈ (matvecSpecGrid ggml_type raw n_out n_in qk bb blocks row_bytes need deq).stage.dom,
      ‖(matvecSpecGrid ggml_type raw n_out n_in qk bb blocks row_bytes need deq).stage.exec z -
        (matvecSpecGrid ggml_type raw n_out n_in qk bb blocks row_bytes need deq).stage.ref z‖ ≤
          matvecEps := by
  intro z hz
  exact (matvecSpecGrid ggml_type raw n_out n_in qk bb blocks row_bytes need deq).stage.defect
    z hz

/-- **Non-vacuity.** The `F16` witness of `Numeric/DequantRefinement.lean` inhabits the grid
domain of `matvecSpecGrid` at `n_out = 1`, `n_in = 1`. -/
theorem matvecSpecGrid_dom_nonempty :
    ∃ (qk bb blocks row_bytes need : Usize),
      (matvecSpecGrid 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
        (fun _ => f16WitVec)).dom.Nonempty := by
  obtain ⟨qk, bb, blocks, row_bytes, need, hdom⟩ := matvec_domain_f16_witness
  exact ⟨qk, bb, blocks, row_bytes, need,
    ⟨gridVec (n := (1#usize).val) f16WitVec, f16WitVec, hdom, rfl⟩⟩

end

end MatvecBridgeGrid
