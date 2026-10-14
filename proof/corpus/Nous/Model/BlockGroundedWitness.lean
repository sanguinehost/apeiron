/-
# A joint witness for the grounded block domain

`Nous.Model.BlockGrounded` proves `block_error_bound_grounded` on the composed domain of
`fullBlockGrounded`, but at this revision only component-level non-vacuity witnesses exist
(`matvecSpecGrid_dom_nonempty`, `rmsVectorStage_dom_witness`, `siluStage_dom_witness`). Obligation 2
of the working agreement requires a *joint* witness at the claimed types: one input `x` that is in
the first norm's domain, whose image under every `exec` is in the next stage's domain, through both
residual branches (the `Stage.comp` propagation obligation).

This module supplies that witness. The instance is `d = 2`, `p = h = 0`:

* Every projection has one of the two shapes `GridProj 2 0` (an input-width-2, output-width-0
  matvec) or `GridProj 0 2` (an input-width-0, output-width-2 matvec). With `n_out = 0` the
  `matvec_domain` row-slice and dequant conjuncts are vacuous, so *every* length-2 activation slice
  is admissible: the projection domain is the whole grid, not a single point. With `n_in = 0` the
  activation is the empty slice, so its domain is the single element of the subsingleton
  `Fin 0 → ℝ`.
* The two `BlockResidual` fields are the constant stage at `0` on the full domain, as the task
  permits for the witness only. `br.attn` therefore maps every `(Q,K,V)` fan-out to `0`, and since
  the output projection's input width is `p = 0` the executed output of `o` is forced to `0` by the
  `matvec_refines_proved` defect bound: `|toReal v| ≤ 1/(2·q32)` on a Q32.32 word leaves only
  `v = 0`. The attention branch's executed update is thus the zero vector.
* The witness input is `rowVec 2 row2`, the `rmsVectorStage_dom_witness` zero row. The first norm's
  executed image is a grid point (the real vector of the chosen `rms_norm_refines_ideal` output),
  which lies in the width-2 projections' `GridDom`; the attention branch adds `0`; the second norm's
  domain contains the same zero row; the FFN fan-out's domains contain the second norm's image and
  the gate/up product's constant stage lands in the down projection's single-point domain.

`fullBlockGrounded_dom_nonempty` states the resulting membership. The instance is chosen for the
witness; it carries no claim that the block's runtime configuration looks like it.

Trust boundary: as `Nous.Model.BlockGrounded`, the grid algebra is checked at the abstract real level
under the closed axiom cone; no claim about rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V, the
`unsafe` container, or the dequant-to-ideal-weight reading.
-/
import Nous.Model.BlockGrounded
import Numeric.Dequant.Refinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open apeiron_word Aeneas Aeneas.Std Result Classical

namespace Nous.Model

noncomputable section

open MatvecBridgeGrid
open MatvecBridge

/-! ## Literal arithmetic on the extracted strides -/

/-- `checked_mul` on literals: the exact product when it fits, stated at the literal. -/
theorem checked_mul_lit (x y z : Usize) (hz : z.val = x.val * y.val)
    (hle : x.val * y.val ≤ Usize.max) : Usize.checked_mul x y = some z := by
  obtain ⟨w, hw, hwv⟩ := dequant_usize_checked_mul_some x y hle
  rw [hw]
  congr 1
  exact UScalar.eq_of_val_eq (by rw [hwv, hz])

/-- `div_ceil n 1 = n`. -/
theorem div_ceil_one (n : Usize) : core.num.Usize.div_ceil n 1#usize = ok n := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := 1#usize) (by decide)
  have hq' : UScalar.div n 1#usize = ok q := hq
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists (UScalar.rem_spec n (y := 1#usize) (by decide))
  have hr' : UScalar.rem n 1#usize = ok r := hr
  have hr0 : r.val = 0 := by rw [hrv]; exact Nat.mod_one _
  have hqn : q = n := UScalar.eq_of_val_eq (by rw [hqv]; simp)
  subst hqn
  unfold core.num.Usize.div_ceil
  simp only [hq', hr', bind_tc_ok, if_pos hr0]

/-- `2 ≤ Usize.max` at either platform width. -/
theorem two_le_usize_max : 2 ≤ Usize.max := by
  have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by simp [Usize.max, Usize.numBits]
  rcases System.Platform.numBits_eq with h | h <;> rw [hmax, h] <;> norm_num

/-- `0 + 0 = 0` on `usize`. -/
theorem add_zero_zero : (0#usize + 0#usize : Result Usize) = ok 0#usize := by
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := 0#usize) (y := 0#usize) (by simp))
  rw [hi]
  congr 1
  exact UScalar.eq_of_val_eq (by rw [hiv]; simp)

/-! ## The two projection shapes -/

/-- A width-2 to width-0 projection: output rows are absent, so every length-2 activation is
admissible. -/
def proj20 (deq : Nat → Slice I64) : GridProj 2 0 :=
  { ggml_type := 1#u32
    raw := Slice.new U8
    n_out := 0#usize
    n_in := 2#usize
    h_out := rfl
    h_in := rfl
    qk := 1#usize
    bb := 2#usize
    blocks := 2#usize
    row_bytes := 4#usize
    need := 0#usize
    deq := deq }

/-- A width-0 to width-2 projection: the activation is the empty slice, its domain the single point
of `Fin 0 → ℝ`. -/
def proj02 (deq : Nat → Slice I64) : GridProj 0 2 :=
  { ggml_type := 1#u32
    raw := Slice.new U8
    n_out := 2#usize
    n_in := 0#usize
    h_out := rfl
    h_in := rfl
    qk := 1#usize
    bb := 2#usize
    blocks := 0#usize
    row_bytes := 0#usize
    need := 0#usize
    deq := deq }

/-- The empty dequantizer row family. -/
def deq0 : Nat → Slice I64 := fun _ => Slice.new I64

/-- A rotation slot that is switched off; `rotProj` therefore reduces to the projection's own
stage. -/
def rotId (n : ℕ) : RotSlot n := { use := false, rot := Stage.id (Fin n → ℝ) }

/-! ## Joint admissibility of the projections -/

/-- Every length-2 activation is admissible for `proj20`: the row-slice and per-row dequant
conjuncts of `matvec_domain` are vacuous at `n_out = 0`. -/
theorem proj20_matvec_domain (deq : Nat → Slice I64) (s : Slice I64) (hs : s.val.length = 2) :
    matvec_domain 1#u32 (Slice.new U8) 0#usize 2#usize s 1#usize 2#usize 2#usize 4#usize 0#usize
      deq := by
  refine ⟨block_geom_f16, hs, div_ceil_one 2#usize, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact checked_mul_lit 2#usize 2#usize 4#usize rfl (by scalar_tac)
  · exact checked_mul_lit 0#usize 4#usize 0#usize rfl (by scalar_tac)
  · simp
  · intro r hr; simp at hr
  · intro r hr; simp at hr
  · intro o ho; simp at ho
  · intro o start i row rb ho; simp at ho

/-- `proj20`'s domain contains every grid point of width 2. -/
theorem proj20_dom_of (deq : Nat → Slice I64) (s : Slice I64) (hs : s.val.length = 2) :
    gridVec s ∈ (proj20 deq).spec.stage.dom :=
  ⟨s, proj20_matvec_domain deq s hs, rfl⟩

/-- `dotPrefix` at the empty slice pair is zero at every index. -/
theorem dotPrefix_nil (k : Nat) : dotPrefix (Slice.new I64) (Slice.new I64) k = 0 := by
  unfold dotPrefix
  rw [Slice.new_val]
  simp

/-- `row_dot_domain` at the empty slice pair: the prefixes are vacuous and the epilogue is `0`. -/
theorem row_dot_domain_nil : row_dot_domain (Slice.new I64) (Slice.new I64) := by
  have hz : (Slice.new I64).val.length = 0 := by rw [Slice.new_val]; rfl
  refine ⟨by simp, ?_, ?_, ?_⟩
  · intro k hk1 hk2
    have hk0 : k ≤ 0 := by simpa [hz] using hk2
    omega
  · rw [dotPrefix_nil, divRoundNearest_zero]; norm_num
  · rw [dotPrefix_nil, divRoundNearest_zero]; norm_num

/-- The empty range of an empty byte slice is returned as an empty slice. -/
theorem get_empty_range :
    ∃ row : Slice U8,
      core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) (Slice.new U8)
        { start := 0#usize, «end» := 0#usize } = ok (some row) := by
  show ∃ row, core.slice.index.SliceIndexRangeUsizeSlice.get
      { start := 0#usize, «end» := 0#usize } (Slice.new U8) = ok (some row)
  unfold core.slice.index.SliceIndexRangeUsizeSlice.get
  rw [if_pos (by simp)]
  exact ⟨_, rfl⟩

/-- `dequantVals l = []` forces `l = []`. -/
theorem eq_nil_of_dequantVals_nil (l : List I64) (h : dequantVals l = []) : l = [] := by
  match l with
  | [] => rfl
  | _ :: _ => simp [dequantVals] at h

/-- Every width-0 activation is admissible for `proj02`; the row slice is the empty range and the
`(n_in = 0)` dequant result is the empty word list. -/
theorem proj02_matvec_domain (deq : Nat → Slice I64) (hdeq : ∀ r, (deq r).val.length = 0) :
    matvec_domain 1#u32 (Slice.new U8) 2#usize 0#usize (Slice.new I64) 1#usize 2#usize 0#usize
      0#usize 0#usize deq := by
  refine ⟨block_geom_f16, ?_, div_ceil_one 0#usize, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [Slice.new_val]; rfl
  · exact checked_mul_lit 0#usize 2#usize 0#usize rfl (by scalar_tac)
  · exact checked_mul_lit 2#usize 0#usize 0#usize rfl (by scalar_tac)
  · rw [Slice.new_val]; simp
  · intro r hr; exact hdeq r
  · intro r hr
    rw [show deq r = Slice.new I64 from by
      apply Slice.ext
      rw [Slice.new_val]
      exact List.eq_nil_of_length_eq_zero (hdeq r)]
    exact row_dot_domain_nil
  · intro o ho
    obtain ⟨row, hrow⟩ := get_empty_range
    exact ⟨0#usize, 0#usize, row,
      checked_mul_lit o 0#usize 0#usize (by simp) (by simp), add_zero_zero, hrow⟩
  · intro o start i row rb ho hstart hadd hget hrb0
    obtain ⟨out', hout, hvals⟩ := dequant_f16_row_exact row 0#usize rb
      (by refine ⟨by simp, ?_⟩; intro j hj; simp at hj)
      (by rw [hrb0]; simp)
    refine ⟨out', ?_, ?_⟩
    · rw [dequant_row_fixed_f16]; exact hout
    · have hv : dequantVals out'.val = [] := by
        rw [hvals, hrb0]
        simp [dequantVals, f16RowValues]
      rw [eq_nil_of_dequantVals_nil out' hv, List.eq_nil_of_length_eq_zero (hdeq o)]

/-- `proj02`'s domain contains the single element of `Fin 0 → ℝ`. -/
theorem proj02_dom_of (deq : Nat → Slice I64) (hdeq : ∀ r, (deq r).val.length = 0) :
    (0 : Fin 0 → ℝ) ∈ (proj02 deq).spec.stage.dom :=
  ⟨Slice.new I64, proj02_matvec_domain deq hdeq, by
    funext i; exact i.elim0⟩

/-! ## The witness-only residual and the concrete block -/

/-- The witness-only residual: attention and the gate/up product are the constant stage at `0` on
the full domain. `Fin 0` is a subsingleton, so both agree with the pinned `mul_ref_on` (the product
on the whole domain, here vacuous) and with any stage a caller might supply; they carry no claim
about the runtime blocks. -/
def br0 : BlockResidual 0 0 where
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

/-- The concrete witness instance: `d = 2`, `p = h = 0`, every projection at one of the two shapes
`GridProj 2 0` / `GridProj 0 2`, every rotation switched off, the in-window radicand `2^32`
(`toReal = 1`), cap `0` and floor `1/2` at both norm sites. -/
def G2 : GroundedBlock 2 0 0 where
  q := proj20 deq0
  k := proj20 deq0
  v := proj20 deq0
  o := proj02 deq0
  g := proj20 deq0
  u := proj20 deq0
  dn := proj02 deq0
  qR := rotId 2
  kR := rotId 2
  vR := rotId 2
  oR := rotId 0
  gR := rotId 2
  uR := rotId 2
  dnR := rotId 0
  br := br0
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

/-- The first norm of `G2`, at the witness parameters. -/
abbrev rms1 : Stage (Fin 2 → ℝ) (Fin 2 → ℝ) :=
  rmsVectorStage 2 G2.hd G2.eps1 G2.heps1 G2.B1 G2.A1 G2.hB1 G2.hA1

/-- The second norm of `G2`, at the witness parameters. -/
abbrev rms2 : Stage (Fin 2 → ℝ) (Fin 2 → ℝ) :=
  rmsVectorStage 2 G2.hd G2.eps2 G2.heps2 G2.B2 G2.A2 G2.hB2 G2.hA2

/-- The attention sub-block of `G2`. -/
abbrev attnSubG : Stage (Fin 2 → ℝ) (Fin 2 → ℝ) :=
  attnSubBlockGrounded G2.q G2.k G2.v G2.o G2.qR G2.kR G2.vR G2.oR G2.br

/-- The FFN sub-block of `G2`. -/
abbrev ffnSubG : Stage (Fin 2 → ℝ) (Fin 2 → ℝ) :=
  ffnSubBlockGrounded G2.g G2.u G2.dn G2.gR G2.uR G2.dnR G2.br G2.Bs G2.hBs

/-- The witness zero row lies in the first norm's domain. -/
theorem row2_mem_rms1 : rowVec 2 row2 ∈ rms1.dom := by
  simpa [rms1, G2] using rmsVectorStage_dom_witness

/-- The witness zero row lies in the second norm's domain (same parameters). -/
theorem row2_mem_rms2 : rowVec 2 row2 ∈ rms2.dom := by
  simpa [rms2, G2] using rmsVectorStage_dom_witness

/-! ## The two executed images as grid points -/

/-- `rowDotReal` against an empty left row is `0`. -/
theorem rowDotReal_eq_zero_of_nil (w x : Slice I64) (hw : w.val = []) : rowDotReal w x = 0 := by
  unfold rowDotReal
  rw [hw]
  simp

/-- A Q32.32 word within `1/(2·q32)` of zero is the zero word: its integer value is a multiple of
`2^-32` and at most `1/2` in magnitude. -/
theorem eq_zero_of_abs_toReal_le (v : I64) (h : |toReal v| ≤ 1 / (2 * q32)) : v = 0#i64 := by
  have hq : (0 : ℝ) < q32 := by unfold q32; norm_num
  have h2 : |((v.val : ℝ))| ≤ 1 / 2 := by
    have h3 : |toReal v| * q32 ≤ (1 / (2 * q32)) * q32 := mul_le_mul_of_nonneg_right h hq.le
    have heq : |toReal v| * q32 = |(v.val : ℝ)| := by
      rw [toReal, abs_div, abs_of_pos hq]
      field_simp
    rw [heq] at h3
    have h4 : (1 / (2 * q32)) * q32 = 1 / 2 := by field_simp
    linarith [h3, h4]
  have h5 : ((|v.val| : ℤ) : ℝ) < 1 := by
    rw [Int.cast_abs]
    linarith
  have h6 : |v.val| < (1 : ℤ) := by exact_mod_cast h5
  exact IScalar.eq_of_val_eq (Int.abs_lt_one_iff.mp h6)

/-- The first norm's executed image of an admissible row is a grid point: it is the real vector of
the chosen `rms_norm_refines_ideal` output, whose length is the row's. -/
theorem rmsVecOut_grid {eps : I64} {B A0 : ℝ} (row : Slice I64)
    (hadm : rmsVecAdm 2 eps B A0 row) :
    ∃ s : Slice I64, s.val.length = 2 ∧ gridVec s = rmsVecOut 2 eps A0 row := by
  have hstrong : ∃ a : I64, rms_norm_domain row eps a ∧ A0 ≤ toReal a := hadm.2.2
  set out := Classical.choose
    (rms_norm_refines_ideal row eps (Classical.choose hstrong) (Classical.choose_spec hstrong).1)
    with hout
  have hspec := Classical.choose_spec
    (rms_norm_refines_ideal row eps (Classical.choose hstrong) (Classical.choose_spec hstrong).1)
  have hlen : out.val.length = 2 := by rw [hspec.2.1, hadm.1]
  refine ⟨Slice.from out.val (by rw [hlen]; exact two_le_usize_max),
    by rw [Slice.from_val]; exact hlen, ?_⟩
  rw [rmsVecOut, dif_pos hstrong]
  funext i
  simp only [gridVec, Slice.from_val, rmsVecOutAux]
  rw [← hout]

/-- The first norm's executed image of an admissible row lies in every width-2 projection's domain:
every length-2 slice is admissible for `proj20`. -/
theorem rmsVecOut_mem_proj20 {eps : I64} {B A0 : ℝ} (row : Slice I64)
    (hadm : rmsVecAdm 2 eps B A0 row) (deq : Nat → Slice I64) :
    rmsVecOut 2 eps A0 row ∈ (proj20 deq).spec.stage.dom := by
  obtain ⟨s, hlen, hgrid⟩ := rmsVecOut_grid row hadm
  rw [← hgrid]
  exact proj20_dom_of deq s hlen

/-! ## The executed attention output is zero -/

/-- `gridExec` at a domain point is the extracted output read as a real vector: the `dif_pos`
evaluation of the choice-based body. -/
theorem gridExec_apply_mem (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) (y : Fin n_in.val → ℝ)
    (hy : y ∈ GridDom ggml_type raw n_out n_in qk bb blocks row_bytes need deq) :
    gridExec ggml_type raw n_out n_in qk bb blocks row_bytes need deq y
      = matvecExecVec
          (gridOut ggml_type raw n_out n_in qk bb blocks row_bytes need deq
            (gridInv ggml_type raw n_out n_in qk bb blocks row_bytes need deq y)
            (gridInv_spec ggml_type raw n_out n_in qk bb blocks row_bytes need deq y hy).1)
          n_out.val := by
  unfold gridExec
  exact dif_pos hy

/-- `gridExec` at the single `Fin 0 → ℝ` input is `0`: the input slice is empty, and the
`MatvecSpec` defect bound `‖exec 0 − ref 0‖ ≤ matvecEps` with `ref 0 = 0` forces every output
coordinate below `1/(2·q32)`, hence the zero word. -/
theorem proj02_exec_zero (deq : Nat → Slice I64) (hdeq : ∀ r, (deq r).val.length = 0) :
    (proj02 deq).spec.stage.exec (0 : Fin 0 → ℝ) = 0 := by
  have hmem := proj02_dom_of deq hdeq
  have hmem' : (0 : Fin 0 → ℝ) ∈
      GridDom 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize 0#usize deq :=
    hmem
  have href : (proj02 deq).spec.stage.ref (0 : Fin 0 → ℝ) = 0 := by
    change Stage.matvecRef (deqMatrix deq 2 0) (0 : Fin 0 → ℝ) = 0
    funext i
    simp [Stage.matvecRef]
  have hnorm : ‖(proj02 deq).spec.stage.exec (0 : Fin 0 → ℝ)‖ ≤ 1 / (2 * q32) := by
    have h := (proj02 deq).spec.stage.defect (0 : Fin 0 → ℝ) hmem
    rw [href, sub_zero] at h
    have hδ : (proj02 deq).spec.stage.delta = matvecEps := rfl
    rw [hδ] at h
    have he : matvecEps = 1 / (2 * q32) := by unfold matvecEps q32; norm_num
    rwa [he] at h
  have hexec : (proj02 deq).spec.stage.exec
      = gridExec 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize 0#usize
          deq := rfl
  have hexec0 : (proj02 deq).spec.stage.exec (0 : Fin 0 → ℝ)
      = matvecExecVec
          (gridOut 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize 0#usize
            deq
            (gridInv 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize 0#usize
              deq (0 : Fin 0 → ℝ))
            (gridInv_spec 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize
              0#usize deq (0 : Fin 0 → ℝ) hmem').1)
          2 := by
    rw [hexec]
    exact gridExec_apply_mem 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize
      0#usize deq (0 : Fin 0 → ℝ) hmem'
  funext i
  have hcoord : |toReal ((gridOut 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize
        0#usize 0#usize deq
        (gridInv 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize 0#usize
          deq (0 : Fin 0 → ℝ))
        (gridInv_spec 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize
          0#usize deq (0 : Fin 0 → ℝ) hmem').1).val.getD i.val 0#i64)| ≤ 1 / (2 * q32) := by
    have h := norm_le_pi_norm ((proj02 deq).spec.stage.exec (0 : Fin 0 → ℝ)) i
    rw [Real.norm_eq_abs] at h
    have h2 : |(gridExec 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize
        0#usize deq (0 : Fin 0 → ℝ)) i| ≤ 1 / (2 * q32) := by
      rw [← hexec]; exact le_trans h hnorm
    rw [gridExec_apply_mem 1#u32 (Slice.new U8) 2#usize 0#usize 1#usize 2#usize 0#usize 0#usize
      0#usize deq (0 : Fin 0 → ℝ) hmem'] at h2
    simpa only [matvecExecVec] using h2
  rw [hexec0]
  simp only [matvecExecVec]
  rw [eq_zero_of_abs_toReal_le _ hcoord]
  simp [toReal_zero]

/-! ## Joint domain membership -/

/-- Membership in the attention sub-block's domain: the three input projections' domains, plus the
output projection's point `0` (the constant attention field supplies it unconditionally). -/
theorem mem_attnSubG_dom {y : Fin 2 → ℝ}
    (hq : y ∈ G2.q.spec.stage.dom) (hk : y ∈ G2.k.spec.stage.dom) (hv : y ∈ G2.v.spec.stage.dom) :
    y ∈ attnSubG.dom := by
  refine ⟨⟨hq, hk, hv⟩, ?_⟩
  exact ⟨trivial, proj02_dom_of deq0 (fun _ => rfl)⟩

/-- Membership in the FFN sub-block's domain: the gate and up projections' domains, plus the down
projection's point `0` (the constant gate/up product supplies it unconditionally). -/
theorem mem_ffnSubG_dom {y : Fin 2 → ℝ}
    (hg : y ∈ G2.g.spec.stage.dom) (hu : y ∈ G2.u.spec.stage.dom) :
    y ∈ ffnSubG.dom := by
  refine ⟨⟨⟨hg, ?_⟩, hu⟩, ?_⟩
  · intro i; exact i.elim0
  · exact ⟨trivial, proj02_dom_of deq0 (fun _ => rfl)⟩

/-- The attention sub-block's executed map is the constant zero: the constant `attn` field maps
every fan-out to `0`, and the `p = 0` output projection maps `0` to `0` (`proj02_exec_zero`). -/
theorem attnSubG_exec_zero (z : Fin 2 → ℝ) : attnSubG.exec z = 0 := by
  have hb : G2.br.attn.exec
      ((qkvFanoutGrounded G2.q G2.k G2.v G2.qR G2.kR G2.vR).exec z) = 0 := rfl
  change (rotProj G2.oR G2.o.spec).exec
    (G2.br.attn.exec ((qkvFanoutGrounded G2.q G2.k G2.v G2.qR G2.kR G2.vR).exec z)) = 0
  rw [hb]
  change G2.o.spec.stage.exec (0 : Fin 0 → ℝ) = 0
  exact proj02_exec_zero deq0 (fun _ => rfl)

/-- **Joint non-vacuity of the grounded block domain.** The zero `rmsVectorStage` witness row is a
single input in `fullBlockGrounded G2`'s composed domain: it is in the first norm's domain, the
attention branch's executed update is the zero vector (`p = 0` forces the output projection to
zero), so that residual returns the same row, which is in the second norm's domain; the second
norm's image lies in the gate and up projections' domains, and the constant gate/up product lands in
the down projection's single-point domain. -/
theorem rowVec_mem_fullBlockGrounded_dom :
    rowVec 2 row2 ∈ (fullBlockGrounded G2).dom := by
  have hx1 : rowVec 2 row2 ∈ rms1.dom := row2_mem_rms1
  have hx1' : ∃ row : Slice I64, rmsVecAdm 2 G2.eps1 G2.B1 G2.A1 row ∧
      rowVec 2 row = rowVec 2 row2 := hx1
  have hx2 : rowVec 2 row2 ∈ rms2.dom := row2_mem_rms2
  have hx2' : ∃ row : Slice I64, rmsVecAdm 2 G2.eps2 G2.B2 G2.A2 row ∧
      rowVec 2 row = rowVec 2 row2 := hx2
  have hrexec : rms1.exec (rowVec 2 row2) = rmsVecOut 2 G2.eps1 G2.A1 (Classical.choose hx1') := by
    change (if h : ∃ row, rmsVecAdm 2 G2.eps1 G2.B1 G2.A1 row ∧ rowVec 2 row = rowVec 2 row2
        then rmsVecOut 2 G2.eps1 G2.A1 (Classical.choose h)
        else rmsIdealVec (toReal G2.eps1) (rowVec 2 row2)) = _
    rw [dif_pos hx1']
  have hrexec2 : rms2.exec (rowVec 2 row2) = rmsVecOut 2 G2.eps2 G2.A2 (Classical.choose hx2') := by
    change (if h : ∃ row, rmsVecAdm 2 G2.eps2 G2.B2 G2.A2 row ∧ rowVec 2 row = rowVec 2 row2
        then rmsVecOut 2 G2.eps2 G2.A2 (Classical.choose h)
        else rmsIdealVec (toReal G2.eps2) (rowVec 2 row2)) = _
    rw [dif_pos hx2']
  have hq : rms1.exec (rowVec 2 row2) ∈ G2.q.spec.stage.dom := by
    rw [hrexec]
    exact rmsVecOut_mem_proj20 (Classical.choose hx1') (Classical.choose_spec hx1').1 deq0
  have hk : rms1.exec (rowVec 2 row2) ∈ G2.k.spec.stage.dom := by
    rw [hrexec]
    exact rmsVecOut_mem_proj20 (Classical.choose hx1') (Classical.choose_spec hx1').1 deq0
  have hv : rms1.exec (rowVec 2 row2) ∈ G2.v.spec.stage.dom := by
    rw [hrexec]
    exact rmsVecOut_mem_proj20 (Classical.choose hx1') (Classical.choose_spec hx1').1 deq0
  have hmem_attn : rms1.exec (rowVec 2 row2) ∈ attnSubG.dom := mem_attnSubG_dom hq hk hv
  have hs1dom : rowVec 2 row2 ∈ (Stage.residual (rms1.comp attnSubG)).dom := ⟨hx1, hmem_attn⟩
  have hs1exec : (Stage.residual (rms1.comp attnSubG)).exec (rowVec 2 row2) = rowVec 2 row2 := by
    show rowVec 2 row2 + (rms1.comp attnSubG).exec (rowVec 2 row2) = rowVec 2 row2
    have hz : (rms1.comp attnSubG).exec (rowVec 2 row2) = 0 := by
      show attnSubG.exec (rms1.exec (rowVec 2 row2)) = 0
      exact attnSubG_exec_zero _
    rw [hz, add_zero]
  have hg : rms2.exec (rowVec 2 row2) ∈ G2.g.spec.stage.dom := by
    rw [hrexec2]
    exact rmsVecOut_mem_proj20 (Classical.choose hx2') (Classical.choose_spec hx2').1 deq0
  have hu : rms2.exec (rowVec 2 row2) ∈ G2.u.spec.stage.dom := by
    rw [hrexec2]
    exact rmsVecOut_mem_proj20 (Classical.choose hx2') (Classical.choose_spec hx2').1 deq0
  have hmem_ffn : rms2.exec (rowVec 2 row2) ∈ ffnSubG.dom := mem_ffnSubG_dom hg hu
  have hs2dom : (Stage.residual (rms1.comp attnSubG)).exec (rowVec 2 row2) ∈
      (Stage.residual (rms2.comp ffnSubG)).dom := by
    rw [hs1exec]
    exact ⟨hx2, hmem_ffn⟩
  change rowVec 2 row2 ∈ ((Stage.residual (rms1.comp attnSubG)).comp
    (Stage.residual (rms2.comp ffnSubG))).dom
  exact ⟨hs1dom, hs2dom⟩

/-- **Joint witness.** `fullBlockGrounded G2`'s composed domain is inhabited by the zero
`rmsVectorStage` witness row. -/
theorem fullBlockGrounded_dom_nonempty :
    ∃ x : Fin 2 → ℝ, x ∈ (fullBlockGrounded G2).dom :=
  ⟨rowVec 2 row2, rowVec_mem_fullBlockGrounded_dom⟩

end
end Nous.Model
