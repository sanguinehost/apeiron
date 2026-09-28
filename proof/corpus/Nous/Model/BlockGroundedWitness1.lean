/-
# A non-degenerate grounded-block witness, and the refuted global-product pin (`APEIRON-MODEL-W4-AI17b`)

`Nous.Model.BlockGroundedWitness` (AI-17) exhibits a point of `fullBlockGrounded`'s composed domain,
but only at `d = 2`, `p = h = 0`: the attention and FFN widths are both zero, so the witness is
non-vacuous only at a degenerate shape. Obligation 2 of the working agreement asks for the joint
witness at the claimed types, and this task names `d = p = h = 1` (or the smallest shapes with
`p, h ≥ 1`) with real `F16` projection data and the zero input.

`BlockResidual.mul`'s reference is pinned to the real elementwise product on its domain
(`BlockResidual.mul_ref_on`), because `Stage.expansion` demands a *global* Lipschitz factor:
`∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖`. A pin of the product on the whole space is impossible for
`1 ≤ h`: the product of two `Fin h → ℝ` vectors is not globally Lipschitz under the sup norm, since
on the diagonal `z = c • eᵢ` against `0` the bound forces `c² ≤ L·c`, hence `c ≤ L` for every
`c ≥ 0`. `pinnedMul_stage_impossible` refutes that old contract at the `Stage` level: for `1 ≤ h` no
`Stage` reference is the elementwise product everywhere. The finding is a refutation of the pinned
contract, not of `BlockResidual`, whose domain agreement stands.

What is delivered is the smallest witness: `d = p = 1`, `h = 0`, with the four attention
projections `GridProj 1 1` carrying the real `F16` weight row of
`Numeric/DequantRefinement.lean` (`f16WitRaw` `[0x00, 0x3C]` = `1.0`) and dequantized rows
`f16WitVec` `[1.0]`, whose domain contains the zero activation. The witness input is the zero row:
each grounded `exec` maps `0` to `0`, and the two residual stages are supplied so that they do too.
`fullBlockGrounded_dom_nonempty_f16` states the resulting membership. This de-degenerates the
attention width from `p = 0` to `p = 1` with a genuine quantized-weight domain; the hidden width
stays `0` in this instance, and a hidden-width witness under the relaxed field is not claimed here.

The instance is chosen for the witness; it carries no claim that the block's runtime configuration
looks like it.

Trust boundary: as `Nous.Model.BlockGrounded`, the grid algebra is checked at the abstract real level
under the closed axiom cone `{propext, Classical.choice, Quot.sound}`; no claim about rustc/LLVM
codegen, the kernel ABI, GPU/SPIR-V, the `unsafe` container, or the dequant-to-ideal-weight reading.
-/
import Nous.Model.BlockGroundedWitness
import Numeric.DequantRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open apeiron_word Aeneas Aeneas.Std Result Classical

namespace Nous.Model

noncomputable section

open MatvecBridgeGrid
open MatvecBridge

/-! ## The refuted global-product pin -/

/-- **The global-product pin is impossible.** For `1 ≤ h` no `Stage` from
`(Fin h → ℝ) × (Fin h → ℝ)` to `Fin h → ℝ` has reference the elementwise product on the whole
space: `Stage.expansion` is global, and the product has no global Lipschitz constant — against `0`,
the diagonal input `c • eᵢ` forces `c² ≤ L·c`, i.e. `c ≤ L`, for every `c ≥ 0`. This refutes the old
pinned field `mul_ref : mul.ref = fun z i => z.1 i * z.2 i`; `BlockResidual.mul_ref_on` restricts
the agreement to `mul.dom` instead. -/
theorem pinnedMul_stage_impossible (h : ℕ) (hh : 1 ≤ h) :
    ¬ ∃ s : Stage ((Fin h → ℝ) × (Fin h → ℝ)) (Fin h → ℝ),
      s.ref = fun z i => z.1 i * z.2 i := by
  rintro ⟨s, href⟩
  let i : Fin h := ⟨0, hh⟩
  let c : ℝ := s.L + 1
  have hcpos : 0 < c := by simp only [c]; linarith [s.L_nonneg]
  have hcnonneg : 0 ≤ c := hcpos.le
  let a : Fin h → ℝ := fun j => if j = i then c else 0
  let z : (Fin h → ℝ) × (Fin h → ℝ) := (a, a)
  let w : (Fin h → ℝ) × (Fin h → ℝ) := (0, 0)
  have hale : ‖a‖ ≤ c := by
    rw [pi_norm_le_iff_of_nonneg hcnonneg]
    intro j
    by_cases hj : j = i
    · simp [a, hj, Real.norm_eq_abs, abs_of_pos hcpos]
    · have hz : a j = 0 := by simp [a, hj]
      rw [hz, norm_zero]
      exact hcnonneg
  have hai : a i = c := by simp [a]
  have hznorm : ‖z - w‖ ≤ c := by
    have hzw : z - w = (a, a) := by
      ext j <;> simp [z, w, a]
    rw [hzw, Prod.norm_mk, max_le_iff]
    exact ⟨hale, hale⟩
  have hrefdiff : ((s.ref z - s.ref w) i) = c ^ 2 := by
    rw [Pi.sub_apply, href]
    simp only [z, w, Pi.zero_apply]
    rw [hai]
    ring
  have hlow : c ^ 2 ≤ ‖s.ref z - s.ref w‖ := by
    calc c ^ 2 = |(s.ref z - s.ref w) i| := by
          rw [hrefdiff]; rw [abs_of_nonneg]; positivity
      _ = ‖(s.ref z - s.ref w) i‖ := by rw [Real.norm_eq_abs]
      _ ≤ ‖s.ref z - s.ref w‖ := norm_le_pi_norm _ i
  have hexp := s.expansion z w
  have h1 : c ^ 2 ≤ s.L * c := by
    calc c ^ 2 ≤ ‖s.ref z - s.ref w‖ := hlow
      _ ≤ s.L * ‖z - w‖ := hexp
      _ ≤ s.L * c := mul_le_mul_of_nonneg_left hznorm s.L_nonneg
  have hLle : s.L < c := by simp [c]
  nlinarith [h1, hcpos, hLle]

/-! ## The zero activation at width one -/

/-- The concrete one-element zero activation row. -/
def z1 : Slice I64 := Slice.from [0#i64] (by simp; scalar_tac)

theorem z1_val : z1.val = [0#i64] := Slice.from_val _ _

theorem z1_length : z1.val.length = 1 := by rw [z1_val]; rfl

/-- The zero row reads back as the zero real vector. -/
theorem gridVec_z1 : gridVec (n := 1) z1 = 0 := by
  funext j
  show toReal (z1.val.getD j.val 0#i64) = 0
  fin_cases j
  simp [z1_val, toReal_zero]

/-- The `F16` weight row `[1.0]` against the zero activation stays inside the dot window: every
prefix is `0` and the epilogue rounding of `0` is `0`. -/
theorem row_dot_domain_f16Wit_z1 : row_dot_domain f16WitVec z1 := by
  have hx : f16WitVec.val = [f16WitOne] := f16WitVec_val
  have hz : z1.val = [0#i64] := z1_val
  refine ⟨by simp [hx, hz], ?_, ?_, ?_⟩
  · intro k hk1 hk2
    have hk : k = 1 := by rw [hx] at hk2; simp at hk2; omega
    subst hk
    norm_num [dotPrefix, hx, hz, f16WitOne_val, I128.min_eq, I128.max_eq]
  · norm_num [dotPrefix, hx, hz, f16WitOne_val, divRoundNearest]
  · norm_num [dotPrefix, hx, hz, f16WitOne_val, divRoundNearest]

/-- Replacing the activation by another slice of the same width, when the new activation satisfies
every row-dot window, preserves `matvec_domain`; the row geometry and dequant conjuncts do not
mention the activation. -/
theorem matvec_domain_activation_congr {ggml_type : U32} {raw : Slice U8} {n_out n_in : Usize}
    {qk bb blocks row_bytes need : Usize} {deq : Nat → Slice I64} {x x' : Slice I64}
    (h : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq)
    (hx' : x'.val.length = n_in.val)
    (hrd : ∀ r, r < n_out.val → row_dot_domain (deq r) x') :
    matvec_domain ggml_type raw n_out n_in x' qk bb blocks row_bytes need deq :=
  ⟨h.1, hx', h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, hrd,
    h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

/-- **The `F16` witness domain at the zero activation.** The `matvec_domain` instance of
`matvec_domain_f16_witness`, moved from the `[1.0]` activation to the zero activation `z1`; only the
length and row-dot conjuncts change. -/
theorem matvec_domain_f16_zero :
    ∃ (qk bb blocks row_bytes need : Usize),
      matvec_domain 1#u32 f16WitRaw 1#usize 1#usize z1 qk bb blocks row_bytes need
        (fun _ => f16WitVec) := by
  obtain ⟨qk, bb, blocks, row_bytes, need, hdom⟩ := matvec_domain_f16_witness
  exact ⟨qk, bb, blocks, row_bytes, need,
    matvec_domain_activation_congr hdom z1_length (fun _ _ => row_dot_domain_f16Wit_z1)⟩

/-! ## The concrete projections -/

/-- The `F16` projection at width `1 × 1`, with the witness weight row `[0x00, 0x3C]` (`1.0`) and
the dequantized rows `[1.0]`; the extracted strides are those of `matvec_domain_f16_zero`. -/
def projF16 (qk bb blocks row_bytes need : Usize) : GridProj 1 1 where
  ggml_type := 1#u32
  raw := f16WitRaw
  n_out := 1#usize
  n_in := 1#usize
  h_out := rfl
  h_in := rfl
  qk := qk
  bb := bb
  blocks := blocks
  row_bytes := row_bytes
  need := need
  deq := fun _ => f16WitVec

/-- The width-`1` to width-`0` projection: the output width is zero, so the domain is every
length-1 activation. -/
def proj10 : GridProj 1 0 where
  ggml_type := 1#u32
  raw := Slice.new U8
  n_out := 0#usize
  n_in := 1#usize
  h_out := rfl
  h_in := rfl
  qk := 1#usize
  bb := 2#usize
  blocks := 1#usize
  row_bytes := 2#usize
  need := 0#usize
  deq := fun _ => Slice.new I64

/-- The width-`0` to width-`1` projection: the input width is zero, so the domain is the single
point of `Fin 0 → ℝ`. -/
def proj01 : GridProj 0 1 where
  ggml_type := 1#u32
  raw := Slice.new U8
  n_out := 1#usize
  n_in := 0#usize
  h_out := rfl
  h_in := rfl
  qk := 1#usize
  bb := 2#usize
  blocks := 0#usize
  row_bytes := 0#usize
  need := 0#usize
  deq := fun _ => Slice.new I64

/-- The `F16` witness domain contains the zero activation. -/
theorem projF16_dom_zero (qk bb blocks row_bytes need : Usize)
    (hdom : matvec_domain 1#u32 f16WitRaw 1#usize 1#usize z1 qk bb blocks row_bytes need
      (fun _ => f16WitVec)) :
    (0 : Fin 1 → ℝ) ∈ (projF16 qk bb blocks row_bytes need).spec.stage.dom :=
  ⟨z1, hdom, gridVec_z1⟩

/-- The width-`1`, output-width-`0` domain is the whole grid; `z1` witnesses the zero point. -/
theorem proj10_matvec_domain :
    matvec_domain 1#u32 (Slice.new U8) 0#usize 1#usize z1 1#usize 2#usize 1#usize 2#usize 0#usize
      (fun _ => Slice.new I64) := by
  refine ⟨block_geom_f16, z1_length, div_ceil_one 1#usize, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact checked_mul_lit 1#usize 2#usize 2#usize (by norm_num) (by scalar_tac)
  · exact checked_mul_lit 0#usize 2#usize 0#usize (by simp) (by scalar_tac)
  · simp
  · intro r hr; simp at hr
  · intro r hr; simp at hr
  · intro o ho; simp at ho
  · intro o start i row rb ho; simp at ho

/-- The zero point lies in the width-`1` to width-`0` projection's domain. -/
theorem proj10_dom_zero : (0 : Fin 1 → ℝ) ∈ proj10.spec.stage.dom :=
  ⟨z1, proj10_matvec_domain, gridVec_z1⟩

/-- The width-`0` to width-`1` domain: the empty row and the empty dequantizer row are the single
`Fin 0 → ℝ` point. -/
theorem proj01_matvec_domain :
    matvec_domain 1#u32 (Slice.new U8) 1#usize 0#usize (Slice.new I64) 1#usize 2#usize 0#usize
      0#usize 0#usize (fun _ => Slice.new I64) := by
  refine ⟨block_geom_f16, ?_, div_ceil_one 0#usize, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [Slice.new_val]; rfl
  · exact checked_mul_lit 0#usize 2#usize 0#usize (by simp) (by scalar_tac)
  · exact checked_mul_lit 1#usize 0#usize 0#usize (by simp) (by scalar_tac)
  · rw [Slice.new_val]; simp
  · intro r hr; rw [Slice.new_val]; simp
  · intro r hr
    exact row_dot_domain_nil
  · intro o ho
    obtain ⟨row, hrow⟩ := get_empty_range
    exact ⟨0#usize, 0#usize, row, checked_mul_lit o 0#usize 0#usize (by simp) (by scalar_tac),
      add_zero_zero, hrow⟩
  · intro o start i row rb ho hstart hadd hget hrb0
    obtain ⟨out', hout, hvals⟩ := dequant_f16_row_exact row 0#usize rb
      (by refine ⟨by simp, ?_⟩; intro j hj; simp at hj)
      (by rw [hrb0]; simp)
    refine ⟨out', ?_, ?_⟩
    · rw [dequant_row_fixed_f16]; exact hout
    · have hv : dequantVals out'.val = [] := by
        rw [hvals, hrb0]
        simp [dequantVals, f16RowValues]
      rw [eq_nil_of_dequantVals_nil out' hv, Slice.new_val]

/-- The single `Fin 0 → ℝ` point lies in the width-`0` to width-`1` projection's domain. -/
theorem proj01_dom_zero : (0 : Fin 0 → ℝ) ∈ proj01.spec.stage.dom :=
  ⟨Slice.new I64, proj01_matvec_domain, by funext i; exact i.elim0⟩

/-- The real matrix-vector reference at the origin is zero. -/
theorem matvecRef_zero {n m : ℕ} (W : Fin m → Fin n → ℝ) :
    Stage.matvecRef W (0 : Fin n → ℝ) = 0 := by
  funext i
  simp [Stage.matvecRef]

/-- A matvec spec's stage reference sends the origin to zero. -/
theorem matvecSpec_stage_ref_zero {n m : ℕ} (s : MatvecSpec n m) :
    s.stage.ref (0 : Fin n → ℝ) = 0 :=
  matvecRef_zero s.W

/-- The `F16` projection's executed map sends the zero activation to zero: the defect bound
`1/(2·q32)` leaves only the zero Q32.32 word. -/
theorem projF16_exec_zero (qk bb blocks row_bytes need : Usize)
    (hdom : matvec_domain 1#u32 f16WitRaw 1#usize 1#usize z1 qk bb blocks row_bytes need
      (fun _ => f16WitVec)) :
    (projF16 qk bb blocks row_bytes need).spec.stage.exec (0 : Fin 1 → ℝ) = 0 := by
  have hmem : (0 : Fin 1 → ℝ) ∈ (projF16 qk bb blocks row_bytes need).spec.stage.dom :=
    projF16_dom_zero qk bb blocks row_bytes need hdom
  have hmem' : (0 : Fin 1 → ℝ) ∈
      GridDom 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need (fun _ => f16WitVec) :=
    hmem
  have hmem' : (0 : Fin 1 → ℝ) ∈
      GridDom 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need (fun _ => f16WitVec) :=
    hmem
  have hexec : (projF16 qk bb blocks row_bytes need).spec.stage.exec
      = gridExec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
          (fun _ => f16WitVec) := rfl
  have hzero : gridExec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
      (fun _ => f16WitVec) (0 : Fin 1 → ℝ) = 0 := by
    have href : (matvecSpecGrid 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
        (fun _ => f16WitVec)).stage.ref (0 : Fin 1 → ℝ) = 0 :=
      matvecSpec_stage_ref_zero _
    have hnorm : ‖gridExec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
        (fun _ => f16WitVec) (0 : Fin 1 → ℝ)‖ ≤ 1 / (2 * q32) := by
      have h := (matvecSpecGrid 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
        (fun _ => f16WitVec)).stage.defect (0 : Fin 1 → ℝ) hmem'
      rw [href, sub_zero] at h
      have he : matvecEps = 1 / (2 * q32) := by unfold matvecEps q32; norm_num
      rw [MatvecSpec.stage_delta, he] at h
      exact h
    have hgrid0 : gridExec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
        (fun _ => f16WitVec) (0 : Fin 1 → ℝ)
        = matvecExecVec
            (gridOut 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
              (fun _ => f16WitVec)
              (gridInv 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
                (fun _ => f16WitVec) (0 : Fin 1 → ℝ))
              (gridInv_spec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
                (fun _ => f16WitVec) (0 : Fin 1 → ℝ) hmem').1)
            1 :=
      gridExec_apply_mem 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
        (fun _ => f16WitVec) (0 : Fin 1 → ℝ) hmem'
    funext i
    have hcoord : |toReal ((gridOut 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
          (fun _ => f16WitVec)
          (gridInv 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
            (fun _ => f16WitVec) (0 : Fin 1 → ℝ))
          (gridInv_spec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
            (fun _ => f16WitVec) (0 : Fin 1 → ℝ) hmem').1).val.getD i.val 0#i64)|
        ≤ 1 / (2 * q32) := by
      have h := norm_le_pi_norm (gridExec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes
        need (fun _ => f16WitVec) (0 : Fin 1 → ℝ)) i
      rw [Real.norm_eq_abs] at h
      have h2 : |(gridExec 1#u32 f16WitRaw 1#usize 1#usize qk bb blocks row_bytes need
          (fun _ => f16WitVec) (0 : Fin 1 → ℝ)) i| ≤ 1 / (2 * q32) := le_trans h hnorm
      rw [hgrid0] at h2
      simpa only [matvecExecVec] using h2
    rw [hgrid0]
    simp only [matvecExecVec]
    rw [eq_zero_of_abs_toReal_le _ hcoord]
    simp [toReal_zero]
  rw [hexec]
  exact hzero

/-! ## The vector RMSNorm stage at the zero row -/

/-- The rounded mean of the zero row is `0`. -/
theorem rmsMean_z1 : rmsMean z1 = 0 := by
  unfold rmsMean divRoundNearest
  rw [z1_val]
  norm_num

/-- The zero row is in the RMSNorm domain at `eps = a = 2^32`. -/
theorem z1_rms_domain : rms_norm_domain z1 (4294967296#i64) (4294967296#i64) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [z1_val]; simp
  · rw [rmsMean_z1]; norm_num
  · unfold rsqrt_window; norm_num
  · intro r _ i hi
    have hzi : z1.val[i]'hi = 0#i64 := by
      have hi1 : i < 1 := by rw [z1_length] at hi; exact hi
      interval_cases i; simp [z1_val]
    have h0 : (0#i64).val = 0 := by norm_num
    rw [hzi, h0]
    constructor <;> · unfold divRoundNearest; norm_num
  · unfold rms_arith_window; rw [z1_val]; norm_num
  · rw [z1_val]; norm_num
  · rw [z1_val]; norm_num

/-- **Non-vacuity at width one.** The zero row lies in `rmsVectorStage 1`'s domain at `eps = 2^32`
(`toReal eps = 1`), cap `B = 0` and floor `A0 = 1/2`. -/
theorem z1_mem_rms :
    rowVec 1 z1 ∈ (rmsVectorStage 1 (by norm_num) (4294967296#i64)
      (by unfold toReal q32; norm_num) (0 : ℝ) (1 / 2) (by norm_num)
      (by unfold q32; norm_num)).dom := by
  refine ⟨z1, ?_, rfl⟩
  refine ⟨z1_length, ?_, ?_⟩
  · intro i
    have hzi : z1.val.getD i.val 0#i64 = 0#i64 := by rw [z1_val]; fin_cases i; rfl
    show |toReal (z1.val.getD i.val 0#i64)| ≤ (0 : ℝ)
    rw [hzi, toReal_zero, abs_zero]
  · exact ⟨4294967296#i64, z1_rms_domain, by unfold toReal q32; norm_num⟩

/-- The cap-`0` vector RMSNorm stage's executed map sends the zero vector to zero. The chosen row is
admissible with cap `0`, so it is the zero row; the defect bound `rmsVecDelta 0 A0 = 1/(2·q32)`
leaves only the zero Q32.32 word. -/
theorem rmsVectorStage_exec_zero (n : ℕ) (hn : 1 ≤ n) (eps : I64) (heps : 0 < toReal eps)
    (A0 : ℝ) (hA0 : 1 / (2 * q32) < A0)
    (hz : (0 : Fin n → ℝ) ∈
      (rmsVectorStage n hn eps heps 0 A0 (le_refl 0) hA0).dom) :
    (rmsVectorStage n hn eps heps 0 A0 (le_refl 0) hA0).exec (0 : Fin n → ℝ) = 0 := by
  have hz' : ∃ row : Slice I64, rmsVecAdm n eps 0 A0 row ∧ rowVec n row = (0 : Fin n → ℝ) := hz
  show (if h : ∃ row : Slice I64, rmsVecAdm n eps 0 A0 row ∧ rowVec n row = (0 : Fin n → ℝ)
      then rmsVecOut n eps A0 (Classical.choose h)
      else rmsIdealVec (toReal eps) (0 : Fin n → ℝ)) = 0
  rw [dif_pos hz']
  obtain ⟨hadm, hrow⟩ := Classical.choose_spec hz'
  have hlen : (Classical.choose hz').val.length = n := hadm.1
  have hcap : ∀ i : Fin n, |rowVec n (Classical.choose hz') i| ≤ (0 : ℝ) := hadm.2.1
  have hstrong : ∃ a : I64, rms_norm_domain (Classical.choose hz') eps a ∧ A0 ≤ toReal a :=
    hadm.2.2
  have hb := rmsVecOut_bound eps A0 (0 : ℝ) (Classical.choose hz') (le_refl 0) hlen hcap hA0
    hstrong
  have hid : rmsIdealVec (toReal eps) (rowVec n (Classical.choose hz')) = 0 := by
    rw [hrow]; funext i; simp [rmsIdealVec]
  have hdelta : rmsVecDelta (0 : ℝ) A0 = 1 / (2 * q32) := by unfold rmsVecDelta; ring
  rw [hid, sub_zero, hdelta] at hb
  have hout : ∀ i, (rmsVecOut n eps A0 (Classical.choose hz')) i
      = toReal ((Classical.choose (rms_norm_refines_ideal (Classical.choose hz') eps
          (Classical.choose hstrong) (Classical.choose_spec hstrong).1)).val.getD i.val 0#i64) := by
    intro i
    unfold rmsVecOut
    rw [dif_pos hstrong]
    rfl
  funext i
  have hcoord : |(rmsVecOut n eps A0 (Classical.choose hz')) i| ≤ 1 / (2 * q32) := by
    have h := norm_le_pi_norm (rmsVecOut n eps A0 (Classical.choose hz')) i
    rw [Real.norm_eq_abs] at h
    exact le_trans h hb
  have hcoord' : |toReal ((Classical.choose (rms_norm_refines_ideal (Classical.choose hz') eps
      (Classical.choose hstrong) (Classical.choose_spec hstrong).1)).val.getD i.val 0#i64)|
      ≤ 1 / (2 * q32) := by
    rw [← hout i]; exact hcoord
  rw [hout i, eq_zero_of_abs_toReal_le _ hcoord', toReal_zero]
  rfl

/-- The zero row reads back as the zero real vector. -/
theorem rowVec_z1 : rowVec 1 z1 = 0 := by
  funext j
  show toReal (z1.val.getD j.val 0#i64) = 0
  fin_cases j
  simp [z1_val, toReal_zero]

/-- The tuple form of the `F16` zero-activation domain, for `Classical.choose`. -/
theorem matvec_domain_f16_zero' :
    ∃ p : Usize × Usize × Usize × Usize × Usize,
      matvec_domain 1#u32 f16WitRaw 1#usize 1#usize z1 p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2
        (fun _ => f16WitVec) := by
  obtain ⟨qk, bb, blocks, row_bytes, need, hdom⟩ := matvec_domain_f16_witness
  refine ⟨(qk, bb, blocks, row_bytes, need), ?_⟩
  exact matvec_domain_activation_congr hdom z1_length (fun _ _ => row_dot_domain_f16Wit_z1)

/-- The chosen `F16` strides of the zero-activation domain. -/
noncomputable def f16WitP : Usize × Usize × Usize × Usize × Usize :=
  Classical.choose matvec_domain_f16_zero'

/-- The chosen strides satisfy the zero-activation `F16` domain. -/
theorem f16WitP_dom :
    matvec_domain 1#u32 f16WitRaw 1#usize 1#usize z1 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1
      f16WitP.2.2.2.1 f16WitP.2.2.2.2 (fun _ => f16WitVec) :=
  Classical.choose_spec matvec_domain_f16_zero'

/-- The witness-only residual at `(p, h) = (1, 0)`: the attention and the hidden-width-zero product
are the constant stage at `0` on the full domain. `Fin 0` is a subsingleton, so the product agrees
with the pinned `mul_ref_on`; they carry no claim about the runtime blocks. -/
def br1 : BlockResidual 1 0 where
  attn :=
    { ref := fun _ => 0
      exec := fun _ => 0
      dom := Set.univ
      delta := 0
      L := 0
      delta_nonneg := le_refl 0
      L_nonneg := le_refl 0
      defect := by intro z _; simp
      expansion := by intro z w; simp }
  mul :=
    { ref := fun _ => 0
      exec := fun _ => 0
      dom := Set.univ
      delta := 0
      L := 0
      delta_nonneg := le_refl 0
      L_nonneg := le_refl 0
      defect := by intro z _; simp
      expansion := by intro z w; simp }
  mul_ref_on := by intro z _; funext i; exact i.elim0

/-- The concrete witness instance: `d = p = 1`, `h = 0`. The four attention projections are the
`F16` projection `projF16` at the chosen strides (real quantized-weight domain); the gate, up and
down projections are the width-degenerate `proj10`/`proj01`; every rotation is switched off; both
norms run at the in-window radicand `2^32` (`toReal = 1`), cap `0` and floor `1/2`; the SiLU cap is
`0`. The hidden width is `0` in this instance; the global-product pin that once forced it is refuted
as `pinnedMul_stage_impossible`. -/
noncomputable def G1 : GroundedBlock 1 1 0 where
  q := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  k := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  v := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  o := projF16 f16WitP.1 f16WitP.2.1 f16WitP.2.2.1 f16WitP.2.2.2.1 f16WitP.2.2.2.2
  g := proj10
  u := proj10
  dn := proj01
  qR := rotId 1
  kR := rotId 1
  vR := rotId 1
  oR := rotId 1
  gR := rotId 1
  uR := rotId 1
  dnR := rotId 0
  br := br1
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
  hBs := by norm_num

/-- The first norm of `G1`. -/
abbrev rmsW1 : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  rmsVectorStage 1 G1.hd G1.eps1 G1.heps1 G1.B1 G1.A1 G1.hB1 G1.hA1

/-- The second norm of `G1`. -/
abbrev rmsW2 : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  rmsVectorStage 1 G1.hd G1.eps2 G1.heps2 G1.B2 G1.A2 G1.hB2 G1.hA2

/-- The attention sub-block of `G1`. -/
abbrev attnSubW : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  attnSubBlockGrounded G1.q G1.k G1.v G1.o G1.qR G1.kR G1.vR G1.oR G1.br

/-- The FFN sub-block of `G1`. -/
abbrev ffnSubW : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  ffnSubBlockGrounded G1.g G1.u G1.dn G1.gR G1.uR G1.dnR G1.br G1.Bs G1.hBs

/-- The zero vector lies in the first norm's domain. -/
theorem zero_mem_rmsW1 : (0 : Fin 1 → ℝ) ∈ rmsW1.dom := by
  have h := z1_mem_rms
  rwa [rowVec_z1] at h

/-- The zero vector lies in the second norm's domain. -/
theorem zero_mem_rmsW2 : (0 : Fin 1 → ℝ) ∈ rmsW2.dom := by
  have h := z1_mem_rms
  rwa [rowVec_z1] at h

/-- The first norm's executed map sends the zero vector to zero. -/
theorem rmsW1_exec_zero : rmsW1.exec (0 : Fin 1 → ℝ) = 0 :=
  rmsVectorStage_exec_zero 1 (by norm_num) (4294967296#i64) (by unfold toReal q32; norm_num)
    (1 / 2) (by unfold q32; norm_num) zero_mem_rmsW1

/-- The second norm's executed map sends the zero vector to zero. -/
theorem rmsW2_exec_zero : rmsW2.exec (0 : Fin 1 → ℝ) = 0 :=
  rmsVectorStage_exec_zero 1 (by norm_num) (4294967296#i64) (by unfold toReal q32; norm_num)
    (1 / 2) (by unfold q32; norm_num) zero_mem_rmsW2

/-- The four `F16` projections' domains contain the zero vector. -/
theorem projF16_dom_zero_G1 :
    (0 : Fin 1 → ℝ) ∈ G1.q.spec.stage.dom ∧ (0 : Fin 1 → ℝ) ∈ G1.k.spec.stage.dom ∧
      (0 : Fin 1 → ℝ) ∈ G1.v.spec.stage.dom ∧ (0 : Fin 1 → ℝ) ∈ G1.o.spec.stage.dom :=
  ⟨projF16_dom_zero _ _ _ _ _ f16WitP_dom, projF16_dom_zero _ _ _ _ _ f16WitP_dom,
    projF16_dom_zero _ _ _ _ _ f16WitP_dom, projF16_dom_zero _ _ _ _ _ f16WitP_dom⟩

/-- The zero vector lies in the attention sub-block's domain. -/
theorem zero_mem_attnSubW : (0 : Fin 1 → ℝ) ∈ attnSubW.dom := by
  obtain ⟨hq, hk, hv, ho⟩ := projF16_dom_zero_G1
  refine ⟨⟨hq, hk, hv⟩, ?_⟩
  exact ⟨trivial, ho⟩

/-- The attention sub-block's executed map sends the zero vector to zero: the constant `attn` field
maps every fan-out to `0`, and the `F16` output projection maps `0` to `0`. -/
theorem attnSubW_exec_zero : attnSubW.exec (0 : Fin 1 → ℝ) = 0 := by
  have hb : G1.br.attn.exec
      ((qkvFanoutGrounded G1.q G1.k G1.v G1.qR G1.kR G1.vR).exec (0 : Fin 1 → ℝ)) = 0 := rfl
  change (rotProj G1.oR G1.o.spec).exec
    (G1.br.attn.exec ((qkvFanoutGrounded G1.q G1.k G1.v G1.qR G1.kR G1.vR).exec
      (0 : Fin 1 → ℝ))) = 0
  rw [hb]
  change G1.o.spec.stage.exec (0 : Fin 1 → ℝ) = 0
  exact projF16_exec_zero _ _ _ _ _ f16WitP_dom

/-- The zero vector lies in the FFN sub-block's domain: the gate and up projections' domains, the
vacuous SiLU domain at hidden width `0`, and the down projection's `Fin 0 → ℝ` point. -/
theorem zero_mem_ffnSubW : (0 : Fin 1 → ℝ) ∈ ffnSubW.dom := by
  refine ⟨⟨⟨proj10_dom_zero, ?_⟩, proj10_dom_zero⟩, ?_⟩
  · intro i; exact i.elim0
  · exact ⟨trivial, proj01_dom_zero⟩

/-- **Joint non-vacuity of the grounded block at `d = p = 1`, `h = 0`.** The zero vector is a single
input in `fullBlockGrounded G1`'s composed domain: it is in the first norm's domain, the first norm
and the attention sub-block map it to `0`, so the residual returns it, it is in the second norm's
domain, and the second norm's image `0` lies in the FFN sub-block's domain. -/
theorem fullBlockGrounded_dom_nonempty_f16 :
    (0 : Fin 1 → ℝ) ∈ (fullBlockGrounded G1).dom := by
  have hx1 : (0 : Fin 1 → ℝ) ∈ rmsW1.dom := zero_mem_rmsW1
  have hexec1 : rmsW1.exec (0 : Fin 1 → ℝ) = 0 := rmsW1_exec_zero
  have hattn : (0 : Fin 1 → ℝ) ∈ attnSubW.dom := zero_mem_attnSubW
  have hattnexec : attnSubW.exec (0 : Fin 1 → ℝ) = 0 := attnSubW_exec_zero
  have hx2 : (0 : Fin 1 → ℝ) ∈ rmsW2.dom := zero_mem_rmsW2
  have hexec2 : rmsW2.exec (0 : Fin 1 → ℝ) = 0 := rmsW2_exec_zero
  have hffn : (0 : Fin 1 → ℝ) ∈ ffnSubW.dom := zero_mem_ffnSubW
  have hs1dom : (0 : Fin 1 → ℝ) ∈ (rmsW1.comp attnSubW).dom :=
    ⟨hx1, by rw [hexec1]; exact hattn⟩
  have hs1exec : (rmsW1.comp attnSubW).exec (0 : Fin 1 → ℝ) = 0 := by
    show attnSubW.exec (rmsW1.exec (0 : Fin 1 → ℝ)) = 0
    rw [hexec1]; exact hattnexec
  have hres1 : (Stage.residual (rmsW1.comp attnSubW)).exec (0 : Fin 1 → ℝ) = 0 := by
    show (0 : Fin 1 → ℝ) + (rmsW1.comp attnSubW).exec (0 : Fin 1 → ℝ) = 0
    rw [hs1exec, add_zero]
  have hs2dom : (Stage.residual (rmsW1.comp attnSubW)).exec (0 : Fin 1 → ℝ) ∈
      (Stage.residual (rmsW2.comp ffnSubW)).dom := by
    rw [hres1]
    exact ⟨hx2, by rw [hexec2]; exact hffn⟩
  change (0 : Fin 1 → ℝ) ∈ ((Stage.residual (rmsW1.comp attnSubW)).comp
    (Stage.residual (rmsW2.comp ffnSubW))).dom
  exact ⟨hs1dom, hs2dom⟩

/-- **Joint witness.** `fullBlockGrounded G1`'s composed domain is inhabited at `d = p = 1`,
`h = 0`, with real `F16` projection data. -/
theorem fullBlockGrounded_dom_nonempty_1 :
    ∃ x : Fin 1 → ℝ, x ∈ (fullBlockGrounded G1).dom :=
  ⟨0, fullBlockGrounded_dom_nonempty_f16⟩

end

end Nous.Model
