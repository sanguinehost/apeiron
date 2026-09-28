/-
# The matvec slice ↔ matrix bridge

`Nous/Model/BlockComposition.lean` models the projection defect `matvecEps = 1/(2·2^32)` of the
extracted `word_matvec_fixed` at `model` grade: `Stage.matvec`/`MatvecSpec` take a real matrix
`W : Fin m → Fin n → ℝ` and an `exec` map satisfying a per-coordinate bound, but no Lean term links
the extracted kernel over `Slice I64`/`Nat`-indexed rows to such a matrix. This module supplies the
link.

* `deqMatrix deq m n` reads the `i`-th row of `deq` as a real `Fin n` vector (`getD`, `0` off the
  stored window); `gridVec x` reads the activation `x` as a real `Fin n` vector.
* `rowDotReal_eq_matvecRef` proves that `rowDotReal` (`Numeric/EngineDotRefinement.lean`) and the
  abstract `Stage.matvecRef` (`Nous/Model/BlockComposition.lean`) are the same sum: with both
  windows of length `n`, `rowDotReal (deq i) x = Stage.matvecRef (deqMatrix deq m n) (gridVec x) i`.
* `matvecSpecOf hdom` builds the `MatvecSpec` for a fixed `matvec_domain` instance, with
  `W = deqMatrix …`, `dom = {gridVec x}` (the single executed input), `exec` the real vector of the
  output `matvec_refines_proved` returns (`0` off the singleton domain), and `rowBound` the max row
  `ℓ1` norm of `W`. Its `hdefect` is *proved* from `matvec_refines_proved` through item 1, replacing
  the `model` reading with a kernel-checked term.
* `matvec_stage_proved` exposes the resulting `Stage`: its defect is at most `matvecEps` on its
  domain.

Trust boundary: the dequant-to-ideal-weight reading stays the class-(b) `tested` boundary of
`Numeric/EngineMatvecRefinement.lean`, not re-examined here. No claim is made about rustc/LLVM
codegen, the kernel ABI, GPU/SPIR-V or the `unsafe` container.
-/
import Nous.Model.BlockComposition
import Numeric.EngineMatvecRefinement

open Aeneas Aeneas.Std Result
open scoped BigOperators

namespace MatvecBridge

noncomputable section

open Nous.Model
open apeiron_word

/-! ## The real matrix and activation windows -/

/-- The `i`-th row of `deq` read as a real `Fin n` vector, `0` off the stored window. -/
def deqMatrix (deq : Nat → Slice I64) (m n : ℕ) : Fin m → Fin n → ℝ :=
  fun i j => toReal ((deq i.val).val.getD j.val 0#i64)

/-- The activation `x` read as a real `Fin n` vector, `0` off the stored window. -/
def gridVec {n : ℕ} (x : Slice I64) : Fin n → ℝ :=
  fun j => toReal (x.val.getD j.val 0#i64)

/-- The `zip`/`sum` row dot of `Numeric.EngineDotRefinement.rowDotReal` is the `Fin n` sum of the
`getD` windows when both lists have length `n`. -/
theorem list_dot_bridge (w x : List I64) (n : ℕ) (hw : w.length = n) (hx : x.length = n) :
    ((w.zip x).map (fun p => toReal p.1 * toReal p.2)).sum =
      ∑ j : Fin n, toReal (w.getD j.val 0#i64) * toReal (x.getD j.val 0#i64) := by
  induction n generalizing w x with
  | zero =>
      rw [List.eq_nil_of_length_eq_zero hw, List.eq_nil_of_length_eq_zero hx]
      simp
  | succ k ih =>
      obtain ⟨a, w', rfl⟩ := List.exists_cons_of_length_pos (by omega : 0 < w.length)
      obtain ⟨b, x', rfl⟩ := List.exists_cons_of_length_pos (by omega : 0 < x.length)
      simp only [List.length_cons] at hw hx
      rw [List.zip_cons_cons, List.map_cons, List.sum_cons, Fin.sum_univ_succ]
      rw [ih w' x' (by omega) (by omega)]
      simp

/-- **Item 1.** The `rowDotReal` reference and the abstract `Stage.matvecRef` agree when the row and
the activation both fill the `n` window: the extracted real row dot is the `Fin n` matrix-vector
row. -/
theorem rowDotReal_eq_matvecRef (deq : Nat → Slice I64) (m n : ℕ) (x : Slice I64)
    (hdeq : ∀ i : Fin m, (deq i.val).val.length = n) (hx : x.val.length = n) (i : Fin m) :
    rowDotReal (deq i.val) x = Stage.matvecRef (deqMatrix deq m n) (gridVec (n := n) x) i := by
  unfold rowDotReal Stage.matvecRef deqMatrix gridVec
  exact list_dot_bridge (deq i.val).val x.val n (hdeq i) hx

/-! ## The max row `ℓ1` bound -/

/-- The max row `ℓ1` norm of `W : Fin m → Fin n → ℝ`, `0` for the empty row set. -/
noncomputable def maxRowBound {m n : ℕ} (W : Fin m → Fin n → ℝ) : ℝ :=
  if h : m = 0 then 0
  else Finset.univ.sup' ⟨⟨0, Nat.pos_of_ne_zero h⟩, Finset.mem_univ _⟩
    (fun i => ∑ j, |W i j|)

theorem maxRowBound_nonneg {m n : ℕ} (W : Fin m → Fin n → ℝ) : 0 ≤ maxRowBound W := by
  unfold maxRowBound
  split
  · exact le_refl 0
  · rename_i h
    exact le_trans (Finset.sum_nonneg (fun j _ => abs_nonneg _))
      (Finset.le_sup' (fun i => ∑ j, |W i j|)
        (Finset.mem_univ (⟨0, Nat.pos_of_ne_zero h⟩ : Fin m)))

theorem le_maxRowBound {m n : ℕ} (W : Fin m → Fin n → ℝ) (i : Fin m) :
    ∑ j, |W i j| ≤ maxRowBound W := by
  unfold maxRowBound
  split
  · rename_i h
    exact absurd i.isLt (by omega)
  · exact Finset.le_sup' (fun i => ∑ j, |W i j|) (Finset.mem_univ i)

/-! ## The executed output as a real vector -/

/-- The real vector of the extracted output row window, `0` past the end. -/
noncomputable def matvecExecVec (out : alloc.vec.Vec I64) (m : ℕ) : Fin m → ℝ :=
  fun i => toReal (out.val.getD i.val 0#i64)

/-- The `exec` map of the spec: the real output vector on the singleton domain `{gridVec x}`, `0`
off it. -/
noncomputable def matvecExec (out : alloc.vec.Vec I64) (x : Slice I64) (m n : ℕ) :
    (Fin n → ℝ) → (Fin m → ℝ) :=
  fun y i => if y = gridVec (n := n) x then matvecExecVec out m i else 0

/-! ## Item 2: the `MatvecSpec` for a fixed `matvec_domain` instance -/

/-- **Item 2.** For a fixed `matvec_domain` instance, the `MatvecSpec n_in n_out` whose matrix is the
real reading of `deq`, whose domain is the singleton executed input `{gridVec x}`, whose `exec` is
the real vector of `matvec_refines_proved`'s output, and whose `rowBound` is the max row `ℓ1` norm.
The defect field is proved from `matvec_refines_proved` and `rowDotReal_eq_matvecRef`, not
assumed. -/
noncomputable def matvecSpecOf
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    MatvecSpec n_in.val n_out.val :=
  let h := matvec_refines_proved ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom
  let out := Classical.choose h
  let hspec := Classical.choose_spec h
  { W := deqMatrix deq n_out.val n_in.val
    exec := matvecExec out x n_out.val n_in.val
    dom := {y | y = gridVec (n := n_in.val) x}
    rowBound := maxRowBound (deqMatrix deq n_out.val n_in.val)
    rowBound_nonneg := maxRowBound_nonneg _
    hrow := fun i => le_maxRowBound _ i
    hdefect := by
      intro y hy i
      simp only [Set.mem_setOf_eq] at hy
      subst hy
      have hdeq : ∀ i : Fin n_out.val, (deq i.val).val.length = n_in.val :=
        fun i => hdom.2.2.2.2.2.2.1 i.val i.isLt
      have hxlen : x.val.length = n_in.val := hdom.2.1
      have hlen : out.val.length = n_out.val := hspec.2.1
      have hbound : ∀ (r : Nat) (hr : r < out.val.length),
          |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32) := hspec.2.2
      simp only [matvecExec, matvecExecVec]
      rw [← rowDotReal_eq_matvecRef deq n_out.val n_in.val x hdeq hxlen i]
      have hb := hbound i.val (by rw [hlen]; exact i.isLt)
      have hle : (1 : ℝ) / (2 * q32) ≤ matvecEps := by
        unfold matvecEps q32
        norm_num
      rw [List.getD_eq_getElem (l := out.val) (d := 0#i64) (n := i.val)
        (by rw [hlen]; exact i.isLt)]
      exact le_trans hb hle }

@[simp] theorem matvecSpecOf_W
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    (matvecSpecOf ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom).W =
      deqMatrix deq n_out.val n_in.val := rfl

@[simp] theorem matvecSpecOf_dom
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    (matvecSpecOf ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom).dom =
      {y | y = gridVec (n := n_in.val) x} := rfl

/-! ## Item 3: the resulting `Stage` defect -/

/-- **Item 3.** The `Stage` of the spec built by `matvecSpecOf` has defect at most `matvecEps` on
its domain: kernel-grounded in `MatvecSpec.hdefect` and `Stage.matvec`. -/
theorem matvec_stage_proved
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    ∀ z ∈ (matvecSpecOf ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom).stage.dom,
      ‖(matvecSpecOf ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom).stage.exec z -
        (matvecSpecOf ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom).stage.ref z‖ ≤
          matvecEps := by
  intro z hz
  have h := (matvecSpecOf ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom).stage.defect
    z hz
  simpa using h

end

end MatvecBridge
