import MatvecSource
import Aeneas

/-
SCOPE: resource theorems for the extracted `word_matvec_fixed` body only: both loop-body
      transitions with their guards, each loop's work measure and iteration bound, the refusal and
      exit branches, and a reachable success witness. Owned file: this one.
REGISTER: local obligation — the work one `word_matvec_fixed` call performs, counted in named units
      (`element` per row-dot iteration, `row` per projection iteration, `append` per output push).
      Excluded: the dequantization and rounding values, the numeric refinement, hardware timing,
      and any canonical ring correspondence.
CORPUS: `src/engine/matvec.rs` lines 86-222 and the extracted `matvec.row_dot_fixed_loop.body`,
      `matvec.row_dot_fixed_loop`, `matvec.row_dot_fixed`, `matvec.matvec_fixed_loop.body`,
      `matvec.matvec_fixed_loop`, `matvec.matvec_fixed`, `word_matvec_fixed`,
      `dequant.dequant_row_fixed`, `quant.block_geom`. `Matvec_gate.py` compares each of these
      declarations byte-for-byte against `proof/ApeironWord.lean` before this file is checked.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean — the canonical-corpus extraction
      of these same bodies (Charon/Aeneas, pins in proof/README.md). No `sanguine`
      theorem is used and none is needed: every statement below is about the extracted Rust body.
GATE: `python3 proof/resource/Matvec_gate.py`, 540 s whole-invocation budget, lane
      `/tmp/opencode/apeiron-lane.lock`, MemoryMax=12G MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: Lean checked (`lake build MatvecResource`, `#resource_audit`) when that gate exits 0.
      Before: no resource theorem for this body. Not claimed: wall-clock or memory measurement,
      dequantization/rounding correctness, or anything about rustc codegen, the heap allocator or
      `alloc::vec` capacity growth.
-/

namespace MatvecResource

open Aeneas Aeneas.Std Result apeiron_word

/-! ## 0. Units and shared scalar facts

A work unit is counted, never a cycle: one `element` is one iteration of
`matvec.row_dot_fixed_loop`, one `row` is one iteration of `matvec.matvec_fixed_loop`, and one
`append` is one `alloc.vec.Vec.push` in the row loop's success branch. -/

/-- Remaining element units of the row-dot loop: the unvisited part of the weight row. -/
def elemRank (w_row : Slice I64) (i : Usize) : Nat := w_row.length - i.val

/-- Remaining row units of the projection loop: the rows not yet visited. -/
def rowRank (n_out o : Usize) : Nat := n_out.val - o.val

/-- Any `usize` value fits `usize`. -/
theorem usize_val_le_max (u : Usize) : u.val ≤ Usize.max := by
  have h := UScalar.hBounds u
  rw [UScalarTy.Usize_numBits_eq] at h
  have hmax : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  omega

/-- Any `usize` addition by one that stays inside the row width stays inside `usize`. -/
theorem add_one_le_max (w_row : Slice I64) (i : Usize) (h : i.val < w_row.length) :
    i.val + (1#usize).val ≤ Usize.max := by
  have hmax : w_row.length ≤ Usize.max := Slice.length_ineq w_row
  have h1 : (1#usize).val = 1 := by norm_num
  rw [h1]
  omega

/-- Any `usize` addition by one that stays inside `n_out.val` stays inside `usize`. -/
theorem add_one_le_max_row (n_out o : Usize) (h : o.val < n_out.val) :
    o.val + (1#usize).val ≤ Usize.max := by
  have hmax : n_out.val ≤ Usize.max := usize_val_le_max n_out
  have h1 : (1#usize).val = 1 := by norm_num
  rw [h1]
  omega

/-! ### The three prologue computations at row width zero

The prologue's block count is `⌈n_in / qk⌉`, its row stride is `blocks · bb` and the requested
payload is `n_out · row_bytes`. At `n_in = n_out = 0` all three are zero, provided the family
block size is nonzero: with `qk = 0` the prologue's own division refuses, which is a different
result and is not claimed here. -/

/-- **Block count at row width zero.** `⌈0 / b⌉` is `0` for a nonzero block size `b`: the quotient
    and the remainder of `0` by `b` are both `0`, and a zero remainder selects the quotient. -/
theorem div_ceil_zero (b : Usize) (hb : b.val ≠ 0) :
    core.num.Usize.div_ceil 0#usize b = ok 0#usize := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec (0#usize : Usize) hb
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists (UScalar.rem_spec (0#usize : Usize) hb)
  have hq' : UScalar.div 0#usize b = ok q := hq
  have hr' : UScalar.rem 0#usize b = ok r := hr
  have hq0 : q = 0#usize := UScalar.eq_of_val_eq (by simp [hqv])
  have hr0 : r.val = 0 := by simp [hrv]
  simp only [core.num.Usize.div_ceil, hq', hr', bind_tc_ok, hq0, hr0]
  rfl

/-- **Stride product at row count zero.** `0 · b` is `0` for every block size `b`, and the product
    cannot overflow because `0` is representable. -/
theorem checked_mul_zero_left (b : Usize) :
    Usize.checked_mul 0#usize b = some 0#usize := by
  have h0 : (0#usize : Usize).val = 0 := by norm_num
  have hz : (0#usize : Usize).val * b.val = 0 := Nat.zero_mul _
  have hb := Usize.checked_mul_bv_spec 0#usize b
  cases hcm : Usize.checked_mul 0#usize b with
  | none =>
      rw [hcm] at hb
      omega
  | some z =>
      rw [hcm] at hb
      exact congrArg some (UScalar.eq_of_val_eq (by omega))

/-! ## 1. The element loop: `matvec.row_dot_fixed_loop` -/

/-- **Guarded element transition.** While the index is inside both slices and the overflow flag
    is clear, one element unit reads `w_row[i]` and `x[i]`, widens them, performs the checked
    `i128` multiply and the checked `i128` add, and continues at `i + 1`. The accumulator takes
    the checked total exactly when the flag stays clear, and is left unchanged on either
    refusal. The remaining element rank drops by exactly one. -/
theorem elem_body_step (w_row x : Slice I64) (acc : I128) (i : Usize)
    (hw : i.val < w_row.length) (hx : i.val < x.length) :
    ∃ acc' : I128, ∃ i' : Usize, ∃ ok' : Bool,
      matvec.row_dot_fixed_loop.body w_row x acc i true =
        ok (ControlFlow.cont (acc', i', ok')) ∧
      i'.val = i.val + 1 ∧ elemRank w_row i' + 1 = elemRank w_row i ∧
      (ok' = true → ∃ a b prod : I128,
        I128.checked_mul a b = some prod ∧ I128.checked_add acc prod = some acc') ∧
      (ok' = false → acc' = acc) := by
  obtain ⟨wv, hwv0⟩ := WP.spec_imp_exists (Slice.index_usize_spec w_row i hw)
  obtain ⟨xv, hxv0⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hx)
  have hwv : Slice.index_usize w_row i = ok wv := hwv0.1
  have hxv : Slice.index_usize x i = ok xv := hxv0.1
  obtain ⟨i', hi, hi'⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) (add_one_le_max w_row i hw))
  have hi0 : i'.val = i.val + 1 := by simpa using hi'
  have hrank : elemRank w_row i' + 1 = elemRank w_row i := by
    simp only [elemRank, hi0]
    omega
  have hguard : i < Slice.len w_row := (UScalar.lt_equiv i (Slice.len w_row)).2 (by
    rw [Slice.len_val]; exact hw)
  cases hm : I128.checked_mul (core.convert.num.FromI128I64.from wv)
      (core.convert.num.FromI128I64.from xv) with
  | none =>
      refine ⟨acc, i', false, ?_, hi0, hrank, ?_, ?_⟩
      · unfold matvec.row_dot_fixed_loop.body
        rw [if_pos hguard, if_pos (rfl : (true : Bool) = true)]
        simp only [hwv, hxv, hm, hi, bind_tc_ok, lift]
        rfl
      · intro hfalse
        exact Bool.noConfusion hfalse
      · intro _
        rfl
  | some prod =>
      cases ha : I128.checked_add acc prod with
      | none =>
          refine ⟨acc, i', false, ?_, hi0, hrank, ?_, ?_⟩
          · unfold matvec.row_dot_fixed_loop.body
            rw [if_pos hguard, if_pos (rfl : (true : Bool) = true)]
            simp only [hwv, hxv, hm, ha, hi, bind_tc_ok, lift]
            rfl
          · intro hfalse
            exact Bool.noConfusion hfalse
          · intro _
            rfl
      | some total =>
          refine ⟨total, i', true, ?_, hi0, hrank, ?_, ?_⟩
          · unfold matvec.row_dot_fixed_loop.body
            rw [if_pos hguard, if_pos (rfl : (true : Bool) = true)]
            simp only [hwv, hxv, hm, ha, hi, bind_tc_ok, lift]
            rfl
          · intro _
            exact ⟨_, _, _, hm, ha⟩
          · intro htrue
            exact Bool.noConfusion htrue

/-- **Element exit on the width.** With the index past the weight row the body exits and returns
    the accumulator and flag unchanged; this is the whole-row exhaustion branch. -/
theorem elem_body_exit_width (w_row x : Slice I64) (acc : I128) (i : Usize) (acc_ok : Bool)
    (hw : ¬ i.val < w_row.length) :
    matvec.row_dot_fixed_loop.body w_row x acc i acc_ok =
      ok (ControlFlow.done (acc, acc_ok)) := by
  unfold matvec.row_dot_fixed_loop.body
  have hnot : ¬ (i < Slice.len w_row) := by
    intro hc
    exact hw (by
      rw [UScalar.lt_equiv] at hc
      rwa [Slice.len_val] at hc)
  rw [if_neg hnot]

/-- **Element exit on the flag.** With the overflow flag dropped the body exits on the next unit,
    keeping the accumulator at its last checked total; this is the `AccumulatorOverflow` branch
    and no further element unit is performed. -/
theorem elem_body_exit_flag (w_row x : Slice I64) (acc : I128) (i : Usize)
    (hw : i.val < w_row.length) :
    matvec.row_dot_fixed_loop.body w_row x acc i false = ok (ControlFlow.done (acc, false)) := by
  unfold matvec.row_dot_fixed_loop.body
  have hnot : ¬ ((false : Bool) = true) := by simp
  rw [if_pos ((UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [Slice.len_val]; exact hw)),
    if_neg hnot]

/-- The row-dot loop's reachable invariant: the index never passes either slice's width, so a
    continuation is always one more element unit of the row rather than a re-read. The flag is
    not part of the invariant: the flag-down branch performs no element work and exits. -/
def elemInv (w_row x : Slice I64) (i : Usize) (_acc_ok : Bool) : Prop :=
  i.val ≤ w_row.length ∧ i.val ≤ x.length

/-- **Element loop: at most the row width in element units.** With the two slice widths equal (the
    guard `row_dot_fixed` checks before the loop), the loop rank `elemRank w_row i` starts at the
    row width (index `0`) and every guarded continuation lowers it by exactly one. So the loop
    terminates within `w_row.length` element units, exiting either by exhausting the row width or
    by dropping the overflow flag. -/
theorem elem_loop_spec (w_row x : Slice I64) (hwidth : x.length = w_row.length) :
    matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true ⦃ _ => True ⦄ := by
  unfold matvec.row_dot_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun st : I128 × Usize × Bool => elemRank w_row st.2.1)
    (inv := fun st : I128 × Usize × Bool => elemInv w_row x st.2.1 st.2.2)
    (post := fun _ => True)
  · intro st hst
    rcases st with ⟨acc, i, acc_ok⟩
    dsimp only at hst ⊢
    rcases hst with ⟨hi, hix⟩
    by_cases hok : acc_ok = true
    · rw [hok]
      by_cases hguard : i.val < w_row.length
      · have hxi : i.val < x.length := by omega
        obtain ⟨acc', i', ok', hbody, hi0, hrank, _, _⟩ :=
          elem_body_step w_row x acc i hguard hxi
        rw [hbody, WP.spec_ok]
        have hi'le : i'.val ≤ w_row.length := by omega
        have hix'le : i'.val ≤ x.length := by omega
        have hrank' : elemRank w_row i' < elemRank w_row i := by
          rw [← hrank]
          exact Nat.lt_succ_self _
        exact ⟨⟨hi'le, hix'le⟩, hrank'⟩
      · rw [elem_body_exit_width w_row x acc i true hguard, WP.spec_ok]
        exact trivial
    · have hfalse : acc_ok = false := by
        cases acc_ok with
        | false => rfl
        | true => exact absurd rfl hok
      rw [hfalse]
      by_cases hguard : i.val < w_row.length
      · rw [elem_body_exit_flag w_row x acc i hguard, WP.spec_ok]
        exact trivial
      · rw [elem_body_exit_width w_row x acc i false hguard, WP.spec_ok]
        exact trivial
  · show elemInv w_row x 0#usize true
    norm_num [elemInv]

/-! ## 2. The row loop: `matvec.matvec_fixed_loop`

One `row` unit is one iteration of `matvec.matvec_fixed_loop` over output rows; the body reads the
row slice at `o·row_bytes`, dequantizes it, reduces it against `x`, and appends one output value
per successful row. -/

/-- **Row exit on the width.** With the row index past `n_out.val` the body exits and returns the
    output vector and error slot unchanged: the whole-matrix completion branch. The exit fires
    whatever the failure flag is, because the width guard is tested first; no row unit is
    performed and no output value is appended. -/
theorem row_body_exit_width (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (x : Slice I64) (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (ok1 : Bool) (err : Option matvec.MatvecError) (ho : ¬ o.val < n_out.val) :
    matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o ok1 err = ok (ControlFlow.done (out, err)) := by
  unfold matvec.matvec_fixed_loop.body
  have hnot : ¬ (o < n_out) := by
    intro hc
    exact ho ((UScalar.lt_equiv o n_out).1 hc)
  rw [if_neg hnot]

/-- **Row exit on the flag.** With the failure flag dropped the body exits on the next unit,
    carrying the recorded error: the first-refusal branch. No further row unit is performed and
    no further output value is appended. -/
theorem row_body_exit_flag (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (x : Slice I64) (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (err : Option matvec.MatvecError) (ho : o.val < n_out.val) :
    matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o false err = ok (ControlFlow.done (out, err)) := by
  unfold matvec.matvec_fixed_loop.body
  have hnot : ¬ ((false : Bool) = true) := by simp
  rw [if_pos ((UScalar.lt_equiv o n_out).2 ho), if_neg hnot]

/-- **Row stride arithmetic.** Under the prologue's checked strides, for every row index `o`
    inside `n_out.val` whose own stride computation succeeded with value `start`, the slice end
    `start + row_bytes` is representable. The derivation is
    `start.val + row_bytes.val = (o.val + 1) · row_bytes.val ≤ n_out.val · row_bytes.val
      = need.val ≤ raw.length ≤ Usize.max`. -/
theorem stride_end_fits (raw : Slice U8) (o n_out row_bytes need : Usize) (start : Usize)
    (hstart : Usize.checked_mul o row_bytes = some start)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.length) (ho : o.val < n_out.val) :
    start.val + row_bytes.val ≤ Usize.max := by
  have hstartv : start.val = o.val * row_bytes.val := by
    have hb := Usize.checked_mul_bv_spec o row_bytes
    rw [hstart] at hb; simp only at hb; exact hb.2.1
  have hneedv : need.val = n_out.val * row_bytes.val := by
    have hb := Usize.checked_mul_bv_spec n_out row_bytes
    rw [hneed] at hb; simp only at hb; exact hb.2.1
  have hraw : raw.length ≤ Usize.max := Slice.length_ineq raw
  have hmul : (o.val + 1) * row_bytes.val ≤ n_out.val * row_bytes.val :=
    Nat.mul_le_mul_right _ (by omega)
  have hexp : (o.val + 1) * row_bytes.val = o.val * row_bytes.val + row_bytes.val := by ring
  rw [hexp] at hmul
  rw [hstartv]
  omega

/-! ## 3. One row unit -/

/-- The row-loop state, as the body's continuation carries it. -/
abbrev RowState :=
  (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I64) ×
    (Std.Usize × Bool × (Option matvec.MatvecError))

/-- The row loop's exit pair: the output vector and the error slot. -/
abbrev RowExit := (alloc.vec.Vec Std.I64) × (Option matvec.MatvecError)

/-- **The row body always continues while the guard holds.** With the flag clear and
    `o.val < n_out.val`, the row-loop body does not exit: it returns a `cont` continuation. Its
    own result is therefore a continuation, which is what the loop's assembly consumes. This
    splits the body's own branches; it assumes nothing about which branch is taken. -/
theorem row_body_continues (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize)
    (x : Slice I64) (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (err : Option matvec.MatvecError)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.length) (houtlen : out.val.length < Usize.max)
    (ho : o.val < n_out.val)
    (hdeqok : ∀ (rb : alloc.vec.Vec I64) (row : Slice U8),
      ∃ r : core.result.Result Unit error.EngineError × alloc.vec.Vec I64,
        dequant.dequant_row_fixed ggml_type row n_in
          { rb with slice := Slice.new Std.I64 } = Result.ok r)
    (hdotok : ∀ (rb : alloc.vec.Vec I64),
      ∃ r : core.result.Result I64 matvec.MatvecError,
        matvec.row_dot_fixed (alloc.vec.Vec.deref rb) x = Result.ok r) :
    ∃ (out' : alloc.vec.Vec I64) (row_buf' : alloc.vec.Vec I64) (o' : Usize) (ok' : Bool)
      (err' : Option matvec.MatvecError),
      matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
          out row_buf o true err = ok (ControlFlow.cont (out', row_buf', o', ok', err')) ∧
      o'.val = o.val + 1 := by
  obtain ⟨o2, ho2, ho2'⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (add_one_le_max_row n_out o ho))
  have hguard : o < n_out := (UScalar.lt_equiv o n_out).2 ho
  have hclear : (alloc.vec.Vec.clear Global row_buf
      : Result (alloc.vec.Vec Std.I64)) = ok { row_buf with slice := Slice.new Std.I64 } := rfl
  have ho20 : o2.val = o.val + 1 := by simpa using ho2'
  by_cases hmul : Usize.checked_mul o row_bytes = none
  · refine ⟨out, row_buf, o2, false, some matvec.MatvecError.LengthOverflow, ?_, ho20⟩
    unfold matvec.matvec_fixed_loop.body
    rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift]
    simp only [hmul, ho2, bind_tc_ok]
    exact rfl
  · match hcm : Usize.checked_mul o row_bytes with
    | none => exact absurd hcm hmul
    | some start =>
        obtain ⟨i, hi, hi'⟩ := WP.spec_imp_exists
          (Usize.add_spec (x := start) (y := row_bytes)
            (stride_end_fits raw o n_out row_bytes need start hcm hneed hsuff ho))
        generalize hgetv : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8)
          raw { start := start, «end» := i } = hgr
        cases hgr with
        | vis e k => exact absurd hgetv (by
            simp only [core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get]
            split <;> simp_all)
        | div => exact absurd hgetv (by
            simp only [core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get]
            split <;> simp_all)
        | ret hopt =>
          have hgetv' : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8)
              raw { start := start, «end» := i } = ok hopt := by
            rw [hgetv]
          cases hopt with
          | none =>
              refine ⟨out, row_buf, o2, false,
                some (matvec.MatvecError.ShortPayload need (Slice.len raw)), ?_, ho20⟩
              unfold matvec.matvec_fixed_loop.body
              rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift]
              simp only [hcm, ho2, hi, hgetv', bind_tc_ok]
              exact rfl
          | some row =>
              obtain ⟨hdqres, hdeq⟩ := hdeqok row_buf row
              obtain ⟨res, rb'⟩ := hdqres
              cases res with
              | Err e =>
                  refine ⟨out, rb', o2, false, some (matvec.MatvecError.Dequant e), ?_, ho20⟩
                  unfold matvec.matvec_fixed_loop.body
                  rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift]
                  simp only [hcm, hclear, ho2, hi, hgetv', hdeq, bind_tc_ok,
                    uncurry_apply_pair]
              | Ok _ =>
                  by_cases hlenrow : alloc.vec.Vec.len rb' = n_in
                  · obtain ⟨hdotres, hdotv⟩ := hdotok rb'
                    cases hdotres with
                    | Err e =>
                        refine ⟨out, rb', o2, false, some e, ?_, ho20⟩
                        unfold matvec.matvec_fixed_loop.body
                        rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift]
                        simp only [hcm, hclear, ho2, hi, hgetv', hdeq, hdotv, bind_tc_ok,
                          uncurry_apply_pair, if_pos hlenrow]
                    | Ok v1 =>
                        obtain ⟨out2, hpush, _⟩ :=
                          WP.spec_imp_exists (alloc.vec.Vec.push_spec out v1 houtlen)
                        refine ⟨out2, rb', o2, true, err, ?_, ho20⟩
                        unfold matvec.matvec_fixed_loop.body
                        rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift]
                        simp only [hcm, hclear, hpush, ho2, hi, hgetv', hdeq, hdotv,
                          bind_tc_ok, uncurry_apply_pair, if_pos hlenrow]
                  · refine ⟨out, rb', o2, false,
                      some (matvec.MatvecError.RowLength o (alloc.vec.Vec.len rb') n_in),
                    ?_, ho20⟩
                    unfold matvec.matvec_fixed_loop.body
                    rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift]
                    simp only [hcm, hclear, ho2, hi, hgetv', hdeq, bind_tc_ok,
                      uncurry_apply_pair, if_neg hlenrow]

/-- **Row-loop transition, one unit.** Under the loop guard `o.val < n_out.val` with the flag
    clear, and given the prologue's checked strides, payload bound and output-capacity bound: if
    the body returns `ok cf` at all, then a `cont` continuation advances the row index by exactly
    one, appends at most one output value, decreases `rowRank n_out` by exactly one, and holds
    the flag and the error slot in agreement — a clear flag continues with the error slot
    unchanged, a dropped flag always carries a recorded refusal. `cf` is whatever the body itself
    returned; the clause covers the body only where it did return `ok`, and no clause assumes any
    dequantizer row or row dot succeeded. The two preconditions are the calls' own `ok`-ness, the
    class-(b) `tested` boundary the resource argument carries; the transition then holds for every
    result they produce. -/
theorem row_body_step (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (err : Option matvec.MatvecError)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.length) (houtlen : out.val.length < Usize.max)
    (ho : o.val < n_out.val)
    (hdeqok : ∀ (rb : alloc.vec.Vec I64) (row : Slice U8),
      ∃ r : core.result.Result Unit error.EngineError × alloc.vec.Vec I64,
        dequant.dequant_row_fixed ggml_type row n_in
          { rb with slice := Slice.new Std.I64 } = ok r)
    (hdotok : ∀ (rb : alloc.vec.Vec I64),
      ∃ r : core.result.Result I64 matvec.MatvecError,
        matvec.row_dot_fixed (alloc.vec.Vec.deref rb) x = ok r)
    {cf : ControlFlow RowState RowExit}
    (hbody : matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o true err = ok cf) :
    ∀ s : RowState, cf = ControlFlow.cont s →
      s.2.2.1.val = o.val + 1 ∧ rowRank n_out s.2.2.1 + 1 = rowRank n_out o ∧
      s.1.val.length ≤ out.val.length + 1 ∧
      (s.2.2.2.1 = true → s.1.val.length = out.val.length + 1 ∧ s.2.2.2.2 = err) ∧
      (s.2.2.2.1 = false → s.2.2.2.2 ≠ none) := by
  obtain ⟨o2, ho2, ho2'⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (add_one_le_max_row n_out o ho))
  have ho20 : o2.val = o.val + 1 := by simpa using ho2'
  have hhrank : rowRank n_out o2 + 1 = rowRank n_out o := by
    simp only [rowRank, ho20]
    omega
  have hguard : o < n_out := (UScalar.lt_equiv o n_out).2 ho
  have hclear : (alloc.vec.Vec.clear Global row_buf
      : Result (alloc.vec.Vec Std.I64)) = ok { row_buf with slice := Slice.new Std.I64 } := rfl
  -- Enumerate the body's own branches. Each names the exact continuation it returns.
  intro s hs
  by_cases hmul : Usize.checked_mul o row_bytes = none
  · -- stride overflow: `LengthOverflow`
    have hcf : cf = .cont (out, row_buf, o2, false, some matvec.MatvecError.LengthOverflow) := by
      unfold matvec.matvec_fixed_loop.body at hbody
      rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift] at hbody
      simp only [hmul, ho2, bind_tc_ok] at hbody
      exact (Result.ok_injective hbody).symm
    subst hcf
    injection hs with h1
    subst s
    simp only
    simp [hhrank, ho20]
  · match hcm : Usize.checked_mul o row_bytes with
    | none => exact absurd hcm hmul
    | some start =>
        obtain ⟨i, hi, hi'⟩ := WP.spec_imp_exists
          (Usize.add_spec (x := start) (y := row_bytes)
            (stride_end_fits raw o n_out row_bytes need start hcm hneed hsuff ho))
        -- the row window read: name the body's own result, split on its three shapes
        generalize hgetv : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8)
          raw { start := start, «end» := i } = hgr
        cases hgr with
        | vis e k => exact absurd hgetv (by
            simp only [core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get]
            split <;> simp_all)
        | div => exact absurd hgetv (by
            simp only [core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get]
            split <;> simp_all)
        | ret hopt =>
          have hgetv' : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8)
              raw { start := start, «end» := i } = ok hopt := by
            rw [hgetv]
          cases hopt with
          | none =>
              -- the row window read yields no slice: `ShortPayload`
              have hcf : cf = .cont (out, row_buf, o2, false,
                  some (matvec.MatvecError.ShortPayload need (Slice.len raw))) := by
                unfold matvec.matvec_fixed_loop.body at hbody
                rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift] at hbody
                simp only [hcm, ho2, hi, hgetv', bind_tc_ok] at hbody
                exact (Result.ok_injective hbody).symm
              subst hcf
              injection hs with h1
              subst s
              simp only
              simp [hhrank, ho20]
          | some row =>
              -- the dequantizer: its own result, split on its two shapes
              obtain ⟨hdqres, hdeq⟩ := hdeqok row_buf row
              obtain ⟨res, rb'⟩ := hdqres
              cases res with
              | Err e =>
                  -- the dequantizer refuses row `o`: `Dequant e`
                  have hcf : cf = .cont (out, rb', o2, false,
                      some (matvec.MatvecError.Dequant e)) := by
                    unfold matvec.matvec_fixed_loop.body at hbody
                    rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift] at hbody
                    simp only [hcm, hclear, ho2, hi, hgetv', hdeq, bind_tc_ok,
                      uncurry_apply_pair] at hbody
                    exact (Result.ok_injective hbody).symm
                  subst hcf
                  injection hs with h1
                  subst s
                  simp only
                  simp [hhrank, ho20]
              | Ok _ =>
                  by_cases hlenrow : alloc.vec.Vec.len rb' = n_in
                  · -- the row dot: its own result, split on its two shapes
                    obtain ⟨hdotres, hdotv⟩ := hdotok rb'
                    cases hdotres with
                    | Err e =>
                        -- the row dot refuses with `e`
                        have hcf : cf = .cont (out, rb', o2, false, some e) := by
                          unfold matvec.matvec_fixed_loop.body at hbody
                          rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift] at hbody
                          simp only [hcm, hclear, ho2, hi, hgetv', hdeq, hdotv, bind_tc_ok,
                            uncurry_apply_pair, if_pos hlenrow] at hbody
                          exact (Result.ok_injective hbody).symm
                        subst hcf
                        injection hs with h1
                        subst s
                        simp only
                        simp [hhrank, ho20]
                    | Ok v1 =>
                        obtain ⟨out2, hpush, hpushv⟩ :=
                          WP.spec_imp_exists (alloc.vec.Vec.push_spec out v1 houtlen)
                        have hlen2 : out2.val.length = out.val.length + 1 := by
                          rw [hpushv]; simp
                        have hcf : cf = .cont (out2, rb', o2, true, err) := by
                          unfold matvec.matvec_fixed_loop.body at hbody
                          rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift] at hbody
                          simp only [hcm, hclear, hpush, ho2, hi, hgetv', hdeq, hdotv,
                            bind_tc_ok, uncurry_apply_pair, if_pos hlenrow] at hbody
                          exact (Result.ok_injective hbody).symm
                        subst hcf
                        injection hs with h1
                        subst s
                        simp only
                        simp [hhrank, ho20, hlen2]
                  · -- a dequantized row whose length differs from `n_in`: `RowLength`
                      have hcf : cf = .cont (out, rb', o2, false,
                          some (matvec.MatvecError.RowLength o (alloc.vec.Vec.len rb') n_in)) := by
                        unfold matvec.matvec_fixed_loop.body at hbody
                        rw [if_pos hguard, if_pos (rfl : (true : Bool) = true), lift] at hbody
                        simp only [hcm, hclear, ho2, hi, hgetv', hdeq, bind_tc_ok,
                          uncurry_apply_pair, if_neg hlenrow] at hbody
                        exact (Result.ok_injective hbody).symm
                      subst hcf
                      injection hs with h1
                      subst s
                      simp only
                      simp [hhrank, ho20]

/-! ## 4. The row loop's measure and unit count -/

/-- The row loop's reachable invariant: at most `n_out.val` rows are left, the output holds no
    more values than rows already visited, and a clear flag means every visited row was appended —
    so the output length is exactly the row index. The third conjunct is what makes the exit count
    exact rather than a bound: while the flag is clear the loop appends one value per visited row,
    so on the success path the output length is `n_out.val` and not merely at most it. -/
def rowInv (n_out : Usize) (out : alloc.vec.Vec I64) (o : Usize) (ok : Bool) : Prop :=
  o.val ≤ n_out.val ∧ out.val.length ≤ o.val ∧ (ok = true → out.val.length = o.val)

/-- The flag-agreement half of the invariant: a clear flag means no refusal was recorded and a
    dropped flag means one was. `row_body_step` proves both directions of this per unit. -/
def rowFlagInv (ok : Bool) (err : Option matvec.MatvecError) : Prop :=
  (ok = true → err = none) ∧ (ok = false → err ≠ none)

/-- From the invariant and the loop guard the output is short enough for the body's own
    `Vec.push`: `out.val.length < Usize.max`. The chain is
    `out.val.length ≤ o.val < n_out.val ≤ Usize.max`. -/
theorem rowInv_push_bound (n_out : Usize) (out : alloc.vec.Vec I64) (o : Usize)
    (hlen : out.val.length ≤ o.val) (ho : o.val < n_out.val) :
    out.val.length < Usize.max := by
  have hmax := usize_val_le_max n_out
  omega

-- The two row-loop preconditions are the calls' own `ok`-ness, the class-(b) `tested` boundary
-- the resource argument carries; `row_body_step` holds for whatever results they produce.

/-- **Row loop: at most `n_out.val` row units and at most `n_out.val` appends.** The loop rank
    `rowRank n_out o` starts at `n_out.val` (row index `0`) and every continuation lowers it by
    exactly one, so the loop terminates within `n_out.val` row units and appends at most
    `n_out.val` output values. It exits either with the flag clear at `o.val = n_out.val` and
    therefore exactly `n_out.val` output values, or with a recorded refusal carrying at most
    `o.val` of them. -/
theorem row_loop_spec (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize) (ok : Bool)
    (err : Option matvec.MatvecError)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.length) (hinv : rowInv n_out out o ok)
    (hflag : rowFlagInv ok err)
    (hdeqok : ∀ (rb : alloc.vec.Vec I64) (row : Slice U8),
      ∃ r : core.result.Result Unit error.EngineError × alloc.vec.Vec I64,
        dequant.dequant_row_fixed ggml_type row n_in
          { rb with slice := Slice.new Std.I64 } = Result.ok r)
    (hdotok : ∀ (rb : alloc.vec.Vec I64),
      ∃ r : core.result.Result I64 matvec.MatvecError,
        matvec.row_dot_fixed (alloc.vec.Vec.deref rb) x = Result.ok r) :
    matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need out row_buf o ok err
      ⦃ b => b.1.val.length ≤ n_out.val ∧ (b.2 = none → b.1.val.length = n_out.val) ⦄ := by
  unfold matvec.matvec_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun s : RowState => rowRank n_out s.2.2.1)
    (inv := fun s : RowState => rowInv n_out s.1 s.2.2.1 s.2.2.2.1 ∧
      rowFlagInv s.2.2.2.1 s.2.2.2.2)
    (post := fun b : RowExit => b.1.val.length ≤ n_out.val ∧
      (b.2 = none → b.1.val.length = n_out.val))
  · intro s hs
    rcases s with ⟨st, row_buf_s, o_s, ok_s, err_s⟩
    dsimp only at hs ⊢
    rcases hs with ⟨⟨ho, hlen, heq⟩, hok⟩
    by_cases hlt : o_s.val < n_out.val
    · by_cases hflag : ok_s = true
      · obtain ⟨cfout, cfbuf, cfo, cfok, cferr, hcf, hcfidx⟩ :=
            row_body_continues ggml_type raw n_out n_in x row_bytes need st row_buf_s o_s err_s
              hneed hsuff (rowInv_push_bound n_out st o_s hlen hlt) hlt hdeqok hdotok
        rw [hflag, hcf, WP.spec_ok]
        show (rowInv n_out cfout cfo cfok ∧ rowFlagInv cfok cferr) ∧
          rowRank n_out cfo < rowRank n_out o_s
        obtain ⟨hidx, hrank, hglen, htrue, hfalse⟩ :=
          row_body_step ggml_type raw n_out n_in x row_bytes need st row_buf_s o_s err_s
            hneed hsuff (rowInv_push_bound n_out st o_s hlen hlt) hlt
            hdeqok hdotok hcf (cfout, cfbuf, cfo, cfok, cferr) rfl
        have hidx' : cfo.val = o_s.val + 1 := hidx
        have hrank' : rowRank n_out cfo + 1 = rowRank n_out o_s := hrank
        have hglen' : cfout.val.length ≤ st.val.length + 1 := hglen
        have htrue' : cfok = true → cfout.val.length = st.val.length + 1 ∧ cferr = err_s := htrue
        have hfalse' : cfok = false → cferr ≠ none := hfalse
        -- the invariant: `cfo ≤ n_out.val`, the output holds at most the rows visited, a clear
        -- flag means one appended value per visited row, and the rank drops by one
        refine ⟨⟨⟨by omega, ?_, ?_⟩, ⟨?_, ?_⟩⟩, ?_⟩
        · have hpushlen : cfout.val.length ≤ st.val.length + 1 := by omega
          omega
        · intro hcfok
          subst hcfok
          rcases htrue' rfl with ⟨hlen'', _⟩
          have hstlen : st.val.length = o_s.val := heq hflag
          omega
        · intro hcfok
          subst hcfok
          rcases htrue' rfl with ⟨_, herr'⟩
          rw [herr']
          exact hok.1 hflag
        · intro hcfok
          exact hfalse' hcfok
        · have : rowRank n_out cfo + 1 = rowRank n_out o_s := hrank'
          omega
      · -- flag down: the body exits on this unit, performing no row unit and no append
        have hfalse : ok_s = false := by
          cases hok_s : ok_s with
          | false => rfl
          | true => exact absurd hok_s hflag
        rw [hfalse, row_body_exit_flag ggml_type raw n_out n_in x row_bytes need st row_buf_s o_s
          err_s hlt, WP.spec_ok]
        refine ⟨?_, ?_⟩
        · show st.val.length ≤ n_out.val
          exact hlen.trans (by omega)
        · intro h
          exact (hok.2 hfalse h).elim
    · -- rows exhausted: the body exits on this unit whatever the flag is, performing no row
      -- unit and no append. With no refusal recorded the loop appended one value per visited row,
      -- so the output holds exactly `n_out.val` values.
      rw [row_body_exit_width ggml_type raw n_out n_in x row_bytes need st row_buf_s o_s
        ok_s err_s hlt, WP.spec_ok]
      refine ⟨?_, ?_⟩
      · show st.val.length ≤ n_out.val
        exact hlen.trans (by omega)
      · intro h
        cases ok_s with
        | true =>
            show st.val.length = n_out.val
            have hstlen : st.val.length = o_s.val := heq rfl
            omega
        | false =>
            exact (hok.2 rfl h).elim
  · exact ⟨hinv, hflag⟩

/-! ## 5. Refusals and a reachable success -/

/-- **Prologue refusal: unsupported quant family.** With `ggml_type` a family `block_geom` does
    not name, the extracted `word_matvec_fixed` refuses with `UnsupportedQuant ggml_type` before
    any row unit is performed: the loop is never entered and the output is empty. -/
theorem matvec_refuses_unsupported_quant (t : U32) (raw : Slice U8) (n_out n_in : Usize)
    (x : Slice I64) (hgeom : quant.block_geom t = ok none) :
    word_matvec_fixed t raw n_out n_in x =
      ok (core.result.Result.Err (matvec.MatvecError.UnsupportedQuant t)) := by
  rw [word_matvec_fixed, matvec.matvec_fixed]
  simp only [hgeom, core.option.Option.ok_or_none,
    core.result.Result.Insts.CoreOpsTry.branch, lift, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl

/-- **Prologue refusal: shape mismatch.** When the activation length differs from the declared row
    width `n_in` the call refuses with `ShapeMismatch`, carrying the activation's own length and
    the declared width; no row unit is performed. -/
theorem matvec_refuses_shape (t : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb : Usize) (hgeom : quant.block_geom t = ok (some (qk, bb)))
    (hlen : x.val.length ≠ n_in.val) :
    word_matvec_fixed t raw n_out n_in x =
      ok (core.result.Result.Err (matvec.MatvecError.ShapeMismatch (Slice.len x) n_in)) := by
  rw [word_matvec_fixed, matvec.matvec_fixed]
  have hne : Slice.len x ≠ n_in := by
    intro hc
    have hv : (Slice.len x).val = (n_in).val := congrArg (fun u : Usize => u.val) hc
    rw [Slice.len_val] at hv
    exact hlen hv
  have hcond : (Slice.len x != n_in) = true := by simpa using hne
  simp only [hgeom, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch, lift, bind_tc_ok, hcond,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl

/-- **Prologue refusal: short payload.** On the prologue's checked strides, when the payload holds
    fewer bytes than `need` the call refuses with `ShortPayload need (Slice.len raw)`: the whole
    requested matrix, so no row unit is performed. -/
theorem matvec_refuses_short_payload (t : U32) (raw : Slice U8) (n_out n_in : Usize)
    (x : Slice I64) (qk bb blocks row_bytes need : Usize)
    (hgeom : quant.block_geom t = ok (some (qk, bb))) (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need) (hshort : raw.val.length < need.val) :
    word_matvec_fixed t raw n_out n_in x =
      ok (core.result.Result.Err (matvec.MatvecError.ShortPayload need (Slice.len raw))) := by
  rw [word_matvec_fixed, matvec.matvec_fixed]
  have hlenx : Slice.len x = n_in := UScalar.eq_of_val_eq (by rw [Slice.len_val]; exact hxlen)
  have hcond : (Slice.len x != n_in) = false := by simp [hlenx]
  have hcondneg : ¬ ((Slice.len x != n_in) = true) := by rw [hcond]; simp
  have hcond2 : Slice.len raw < need := by
    have h1 : raw.val.length < need.val := hshort
    simpa only [Slice.len_val] using ((UScalar.lt_equiv (Slice.len raw) need).2 h1)
  simp only [hgeom, core.option.Option.ok_or_some, bind_tc_ok, lift,
    core.result.Result.Insts.CoreOpsTry.branch]
  simp only [hcondneg, Bool.false_eq_true, ite_false]
  -- The prologue's own `let (qk, bb) := (qk, bb)` destructuring shadows the two block-geometry
  -- values the stride hypotheses are about, so restate the computation over those.
  show (do
      let blocks ← core.num.Usize.div_ceil n_in qk
      let r1 ← core.option.Option.ok_or (blocks.checked_mul bb)
        matvec.MatvecError.LengthOverflow
      let cf1 ← core.result.Result.Insts.CoreOpsTry.branch r1
      match cf1 with
      | .Continue val1 => do
          let r2 ← core.option.Option.ok_or (n_out.checked_mul val1)
            matvec.MatvecError.LengthOverflow
          let cf2 ← core.result.Result.Insts.CoreOpsTry.branch r2
          match cf2 with
          | .Continue val2 =>
              if Slice.len raw < val2 then
                ok (core.result.Result.Err (matvec.MatvecError.ShortPayload val2 (Slice.len raw)))
              else do
                let out := alloc.vec.Vec.with_capacity Std.I64 n_out
                let row_buf := alloc.vec.Vec.with_capacity Std.I64 n_in
                let (out1, err) ← matvec.matvec_fixed_loop t raw n_out n_in x val1 val2 out row_buf
                  0#usize true none
                match err with
                | none => ok (core.result.Result.Ok out1)
                | some e => ok (core.result.Result.Err e)
          | .Break residual =>
              core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                (alloc.vec.Vec Std.I64) (core.convert.FromSame matvec.MatvecError) residual
      | .Break residual =>
          core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
            (alloc.vec.Vec Std.I64) (core.convert.FromSame matvec.MatvecError) residual) =
    ok (core.result.Result.Err (matvec.MatvecError.ShortPayload need (Slice.len raw)))
  simp only [hdiv, hrowb, hneed, bind_tc_ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rw [if_pos hcond2]

/-- **Prologue refusal: row-stride overflow.** On the prologue's checked strides, when
    `n_out·row_bytes` leaves `usize` the call refuses with `LengthOverflow`: no row unit is
    performed. -/
theorem matvec_refuses_length_overflow (t : U32) (raw : Slice U8) (n_out n_in : Usize)
    (x : Slice I64) (qk bb blocks _row_bytes : Usize)
    (hgeom : quant.block_geom t = ok (some (qk, bb))) (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hover : Usize.checked_mul blocks bb = none) :
    word_matvec_fixed t raw n_out n_in x = ok (core.result.Result.Err matvec.MatvecError.LengthOverflow) := by
  rw [word_matvec_fixed, matvec.matvec_fixed]
  have hlenx : Slice.len x = n_in := UScalar.eq_of_val_eq (by rw [Slice.len_val]; exact hxlen)
  have hcond : (Slice.len x != n_in) = false := by simp [hlenx]
  have hcondneg : ¬ ((Slice.len x != n_in) = true) := by rw [hcond]; simp
  simp only [hgeom, core.option.Option.ok_or_some, bind_tc_ok, lift,
    core.result.Result.Insts.CoreOpsTry.branch]
  simp only [hcondneg, Bool.false_eq_true, ite_false]
  -- As in `matvec_refuses_short_payload`, restate the computation over the two block-geometry
  -- values the stride hypotheses are about rather than the destructuring's shadow.
  show (do
      let blocks ← core.num.Usize.div_ceil n_in qk
      let r1 ← core.option.Option.ok_or (blocks.checked_mul bb)
        matvec.MatvecError.LengthOverflow
      let cf1 ← core.result.Result.Insts.CoreOpsTry.branch r1
      match cf1 with
      | .Continue val1 => do
          let r2 ← core.option.Option.ok_or (n_out.checked_mul val1)
            matvec.MatvecError.LengthOverflow
          let cf2 ← core.result.Result.Insts.CoreOpsTry.branch r2
          match cf2 with
          | .Continue val2 =>
              if Slice.len raw < val2 then
                ok (core.result.Result.Err (matvec.MatvecError.ShortPayload val2 (Slice.len raw)))
              else do
                let out := alloc.vec.Vec.with_capacity Std.I64 n_out
                let row_buf := alloc.vec.Vec.with_capacity Std.I64 n_in
                let (out1, err) ← matvec.matvec_fixed_loop t raw n_out n_in x val1 val2 out row_buf
                  0#usize true none
                match err with
                | none => ok (core.result.Result.Ok out1)
                | some e => ok (core.result.Result.Err e)
          | .Break residual =>
              core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                (alloc.vec.Vec Std.I64) (core.convert.FromSame matvec.MatvecError) residual
      | .Break residual =>
          core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
            (alloc.vec.Vec Std.I64) (core.convert.FromSame matvec.MatvecError) residual) =
    ok (core.result.Result.Err matvec.MatvecError.LengthOverflow)
  simp only [hdiv, bind_tc_ok, hover, core.option.Option.ok_or_none,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
  rfl

/-- **Epilogue: a recorded refusal is returned unchanged.** On the prologue's checked strides,
    whatever the row loop returned, the wrapper returns `Ok out` exactly when the error slot is
    `none` and `Err e` exactly when it is `some e`: no value is clamped and no error is rewritten.
    This is the refusal branch of the kernel's exit path, and it decides the whole result. -/
theorem matvec_epilogue (t : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (out1 : alloc.vec.Vec I64)
    (e : matvec.MatvecError)
    (hgeom : quant.block_geom t = ok (some (qk, bb))) (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.val.length)
    (hloop : matvec.matvec_fixed_loop t raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none
      = ok (out1, some e)) :
    word_matvec_fixed t raw n_out n_in x = ok (core.result.Result.Err e) := by
  rw [word_matvec_fixed, matvec.matvec_fixed]
  have hlenx : Slice.len x = n_in := UScalar.eq_of_val_eq (by rw [Slice.len_val]; exact hxlen)
  have hcond : (Slice.len x != n_in) = false := by simp [hlenx]
  have hcond2 : ¬ (Slice.len raw < need) := by
    intro hc
    have h1 := (UScalar.lt_equiv (Slice.len raw) need).1 hc
    simp only [Slice.len_val, Slice.length] at h1
    omega
  simp only [hgeom, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch, lift, bind_tc_ok, hcond, hdiv, hrowb, hneed,
    hcond2, hloop, uncurry_apply_pair]
  rfl

/-- The empty `i64` activation slice, built under the length bound `Slice.from` checks. -/
def emptyX : Slice I64 := Slice.from [] (by scalar_tac)

/-- **Reachable success.** The zero-row case is a real input the kernel accepts: at `n_in = n_out =
    0` the block count and both stride products are zero, so the payload bound is `0 ≤ raw.length`
    and the prologue passes; the row loop then exits on its first unit without appending anything
    (the guard `0 < 0` is false), and the wrapper returns `Ok` of the empty output. The family
    block size is nonzero, as it is for every family `block_geom` names: at `qk = 0` the prologue's
    own division refuses instead, and that result is not claimed here. -/
theorem matvec_empty_rows_success (t : U32) (raw : Slice U8)
    (qk bb : Usize) (hqk : qk.val ≠ 0)
    (hgeom : quant.block_geom t = ok (some (qk, bb))) :
    ∃ out : alloc.vec.Vec I64,
      word_matvec_fixed t raw 0#usize 0#usize emptyX =
        ok (core.result.Result.Ok out) ∧ out.val = [] := by
  refine ⟨alloc.vec.Vec.with_capacity Std.I64 0#usize, ?_, ?_⟩
  · rw [word_matvec_fixed, matvec.matvec_fixed]
    have hlenx : Slice.len emptyX = 0#usize := by
      apply UScalar.eq_of_val_eq
      rw [Slice.len_val]
      simp [emptyX, Slice.from_val]
    have hcond : (Slice.len emptyX != 0#usize) = false := by simp [hlenx]
    have hcondneg : ¬ ((Slice.len emptyX != 0#usize) = true) := by rw [hcond]; simp
    have hblocks : core.num.Usize.div_ceil 0#usize qk = ok 0#usize := div_ceil_zero qk hqk
    have hrowb : Usize.checked_mul 0#usize bb = some 0#usize := checked_mul_zero_left bb
    have hneed : Usize.checked_mul 0#usize 0#usize = some 0#usize := checked_mul_zero_left 0#usize
    have hcond2 : ¬ (Slice.len raw < 0#usize) := by
      intro hc
      have h1 := (UScalar.lt_equiv (Slice.len raw) 0#usize).1 hc
      simp only [Slice.len_val, Slice.length] at h1
      have hzero : (0#usize : Usize).val = 0 := by norm_num
      omega
    simp only [hgeom, core.option.Option.ok_or_some, bind_tc_ok, lift,
      core.result.Result.Insts.CoreOpsTry.branch]
    simp only [hcondneg, Bool.false_eq_true, ite_false]
    -- As in the two prologue refusals, restate the computation over the two block-geometry
    -- values the stride hypotheses are about rather than the destructuring's shadow.
    show (do
        let blocks ← core.num.Usize.div_ceil 0#usize qk
        let r1 ← core.option.Option.ok_or (blocks.checked_mul bb)
          matvec.MatvecError.LengthOverflow
        let cf1 ← core.result.Result.Insts.CoreOpsTry.branch r1
        match cf1 with
        | .Continue val1 => do
            let r2 ← core.option.Option.ok_or (Usize.checked_mul 0#usize val1)
              matvec.MatvecError.LengthOverflow
            let cf2 ← core.result.Result.Insts.CoreOpsTry.branch r2
            match cf2 with
            | .Continue val2 =>
                if Slice.len raw < val2 then
                  ok (core.result.Result.Err (matvec.MatvecError.ShortPayload val2 (Slice.len raw)))
                else do
                  let (out1, err) ← matvec.matvec_fixed_loop t raw 0#usize 0#usize emptyX val1 val2
                      (alloc.vec.Vec.with_capacity Std.I64 0#usize)
                      (alloc.vec.Vec.with_capacity Std.I64 0#usize) 0#usize true none
                  match err with
                  | none => ok (core.result.Result.Ok out1)
                  | some e => ok (core.result.Result.Err e)
            | .Break residual =>
                core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                  (alloc.vec.Vec Std.I64) (core.convert.FromSame matvec.MatvecError) residual
        | .Break residual =>
            core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
              (alloc.vec.Vec Std.I64) (core.convert.FromSame matvec.MatvecError) residual) =
      ok (core.result.Result.Ok (alloc.vec.Vec.with_capacity Std.I64 0#usize))
    simp only [hblocks, hrowb, hneed, bind_tc_ok, core.option.Option.ok_or_some,
      core.result.Result.Insts.CoreOpsTry.branch, hcond2, ite_false,
      core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]
    have hbody : matvec.matvec_fixed_loop.body t raw 0#usize 0#usize emptyX 0#usize 0#usize
        (alloc.vec.Vec.with_capacity Std.I64 0#usize)
        (alloc.vec.Vec.with_capacity Std.I64 0#usize) 0#usize true none
      = ok (ControlFlow.done (alloc.vec.Vec.with_capacity Std.I64 0#usize, none)) :=
      row_body_exit_width t raw 0#usize 0#usize emptyX
        0#usize 0#usize (alloc.vec.Vec.with_capacity Std.I64 0#usize)
        (alloc.vec.Vec.with_capacity Std.I64 0#usize) 0#usize true none (by omega)
    -- `loop` is a partial fixpoint, so the row loop's value comes from its own spec at the
    -- initial state; the invariant is the singleton state whose body exits on the width guard.
    have hspec : matvec.matvec_fixed_loop t raw 0#usize 0#usize emptyX 0#usize 0#usize
        (alloc.vec.Vec.with_capacity Std.I64 0#usize)
        (alloc.vec.Vec.with_capacity Std.I64 0#usize) 0#usize true none
      ⦃ b => b = (alloc.vec.Vec.with_capacity Std.I64 0#usize, none) ⦄ := by
      unfold matvec.matvec_fixed_loop
      apply loop.spec_decr_nat
        (measure := fun _ : RowState => 0)
        (inv := fun s : RowState => s = (alloc.vec.Vec.with_capacity Std.I64 0#usize,
          alloc.vec.Vec.with_capacity Std.I64 0#usize, 0#usize, true, none))
        (post := fun b : RowExit => b = (alloc.vec.Vec.with_capacity Std.I64 0#usize, none))
      · intro s hs
        rcases s with ⟨st, row_buf_s, o_s, ok_s, err_s⟩
        dsimp only at hs ⊢
        simp only [Prod.mk.injEq] at hs
        rcases hs with ⟨hst, hbuf, ho, hok, herr⟩
        subst hst
        subst hbuf
        subst ho
        subst hok
        subst herr
        rw [hbody, WP.spec_ok]
      · rfl
    rw [WP.spec_equiv_exists] at hspec
    obtain ⟨r, hr, hp⟩ := hspec
    rw [hr, hp]
    simp only [bind_tc_ok]
    rfl
  · simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]

end MatvecResource

open Lean Elab Command in
/-- Fail-closed audit: every headline theorem may depend only on `propext`, `Classical.choice`
    and `Quot.sound`. Anything else is an error, not a warning. -/
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit MatvecResource.usize_val_le_max
#resource_audit MatvecResource.add_one_le_max
#resource_audit MatvecResource.add_one_le_max_row
#resource_audit MatvecResource.elem_body_step
#resource_audit MatvecResource.elem_body_exit_width
#resource_audit MatvecResource.elem_body_exit_flag
#resource_audit MatvecResource.elem_loop_spec
#resource_audit MatvecResource.row_body_exit_width
#resource_audit MatvecResource.row_body_exit_flag
#resource_audit MatvecResource.stride_end_fits
#resource_audit MatvecResource.row_body_continues
#resource_audit MatvecResource.row_body_step
#resource_audit MatvecResource.rowInv_push_bound
#resource_audit MatvecResource.row_loop_spec
#resource_audit MatvecResource.matvec_refuses_unsupported_quant
#resource_audit MatvecResource.matvec_refuses_shape
#resource_audit MatvecResource.matvec_refuses_short_payload
#resource_audit MatvecResource.matvec_refuses_length_overflow
#resource_audit MatvecResource.matvec_epilogue
#resource_audit MatvecResource.matvec_empty_rows_success

run_cmd Lean.logInfo "MATVEC_KERNEL_OK"
