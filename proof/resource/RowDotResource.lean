import RowDotSource
import Aeneas

/-
SCOPE: actual extracted fixed-point row-dot body: guarded loop-body transitions, exit branches,
the exact visit and multiply-add counts of the executing loop, a reachable success witness and
every refusal branch as an exact equality.
REGISTER: scalar source-refinement resource model; no hardware timing, no canonical-ring
correspondence, no claim about the dequantizers, the strides or the surrounding projections.
CORPUS: src/engine/matvec.rs `row_dot_fixed` and the actual extracted
`row_dot_source.matvec.row_dot_fixed`, `row_dot_source.matvec.row_dot_fixed_loop`,
`row_dot_source.matvec.row_dot_fixed_loop.body` and `row_dot_source.value.round_shift_32`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean (committed extraction of the same
`matvec.rs` body) and proof/corpus/Numeric/EngineDotRefinement.lean.
GATE: python3 proof/resource/RowDot_gate.py --controls, whole invocation 540 s.
GRADE: Lean checked only when that gate succeeds; the work measure is a visit and multiply-add
count, not a machine-time measurement.
-/

namespace RowDotResource
open Aeneas Aeneas.Std Result
universe u v

/-- The actual extracted loop body, as a function of the loop state. -/
def dotBodyFn (w_row x : Slice I64) : (I128 × Usize × Bool) →
    Result (ControlFlow (I128 × Usize × Bool) (I128 × Bool)) :=
  fun s => row_dot_source.matvec.row_dot_fixed_loop.body w_row x s.1 s.2.1 s.2.2

/-- `Slice.len` reads the same length the indexed reads and the loop guard use. -/
theorem sliceLenVal {α : Type u} (s : Slice α) : (Slice.len s).val = s.val.length := by
  rw [Slice.len_val]

/-! ## The exact prefix and the checked-`i128` witnesses -/

/-- The exact ℤ prefix `Σ_{j<k} w_j·x_j` of the two rows; `0` at `k = 0`. -/
def dotPrefix (w_row x : Slice I64) (k : Nat) : ℤ :=
  (((w_row.val.take k).zip (x.val.take k)).map (fun p => p.1.val * p.2.val)).sum

theorem dotPrefix_zero (w_row x : Slice I64) : dotPrefix w_row x 0 = 0 := by
  unfold dotPrefix
  simp

theorem dotPrefix_succ (w_row x : Slice I64) {i : Nat}
    (hiw : i < w_row.val.length) (hix : i < x.val.length) :
    dotPrefix w_row x (i + 1) =
      dotPrefix w_row x i + w_row.val[i].val * x.val[i].val := by
  unfold dotPrefix
  rw [List.take_add_one, List.take_add_one]
  rw [List.getElem?_eq_getElem hiw, List.getElem?_eq_getElem hix]
  simp only [Option.toList_some]
  have hlen : (w_row.val.take i).length = (x.val.take i).length := by
    rw [List.length_take, List.length_take, Nat.min_eq_left (le_of_lt hiw),
      Nat.min_eq_left (le_of_lt hix)]
  rw [List.zip_append hlen]
  simp only [List.zip_cons_cons, List.zip_nil_right, List.map_append, List.sum_append,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  ring

/-- A product of two `i64`-range integers fits `i128`. -/
theorem i64_prod_range (a b : ℤ)
    (ha : -((2 : ℤ) ^ (63 : ℕ)) ≤ a) (ha' : a ≤ (2 : ℤ) ^ (63 : ℕ) - 1)
    (hb : -((2 : ℤ) ^ (63 : ℕ)) ≤ b) (hb' : b ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    -((2 : ℤ) ^ (127 : ℕ)) ≤ a * b ∧ a * b ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
  have habs : |a * b| ≤ (2 : ℤ) ^ (126 : ℕ) := by
    rw [abs_mul]
    have ha2 : |a| ≤ (2 : ℤ) ^ (63 : ℕ) := abs_le.mpr ⟨by linarith, by linarith⟩
    have hb2 : |b| ≤ (2 : ℤ) ^ (63 : ℕ) := abs_le.mpr ⟨by linarith, by linarith⟩
    calc |a| * |b| ≤ (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (63 : ℕ) :=
          mul_le_mul ha2 hb2 (abs_nonneg _) (by positivity)
      _ = (2 : ℤ) ^ (126 : ℕ) := by norm_num
  have hle := abs_le.mp habs
  constructor <;> linarith [hle.1, hle.2]

/-- The `some` arm of the checked `i128` addition the body performs. -/
theorem I128_checked_add_some (x y : I128)
    (hlo : I128.min ≤ x.val + y.val) (hhi : x.val + y.val ≤ I128.max) :
    ∃ z, I128.checked_add x y = some z ∧ z.val = x.val + y.val := by
  have hspec := I128.checked_add_bv_spec x y
  cases hc : core.num.checked_add_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I128.checked_add] using hc, hspec.2.2.1⟩

/-- The `some` arm of the checked `i128` multiplication the body performs. -/
theorem I128_checked_mul_some (x y : I128)
    (hlo : I128.min ≤ x.val * y.val) (hhi : x.val * y.val ≤ I128.max) :
    ∃ z, I128.checked_mul x y = some z ∧ z.val = x.val * y.val := by
  have hspec := I128.checked_mul_bv_spec x y
  cases hc : core.num.checked_mul_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I128.checked_mul] using hc, hspec.2.2.1⟩

/-- The extracted `x >>> 32#i32` is `Int` floor division by `2^32`. -/
theorem I128_shiftRight_32_val (x : I128) :
    ∃ y : I128, (x >>> 32#i32) = ok y ∧ y.val = x.val / 2^32 := by
  refine ⟨⟨x.bv.sshiftRight 32⟩, ?_, ?_⟩
  · show IScalar.shiftRight x (32#i32).toNat = ok _
    have h32 : (32#i32).toNat = 32 := by decide
    rw [h32]
    unfold IScalar.shiftRight
    rw [if_pos (by norm_num [IScalarTy.numBits, I128.numBits])]
  · show (⟨x.bv.sshiftRight 32⟩ : I128).val = x.val / 2^32
    rw [show (⟨x.bv.sshiftRight 32⟩ : I128).val = (x.bv.sshiftRight 32).toInt from rfl]
    rw [BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]
    norm_num

/-- The extracted epilogue constant `1#i128 <<< 31#i32 = 2^31`. -/
theorem I128_shiftLeft_31 : (1#i128 <<< 31#i32) = ok (2147483648#i128) := by
  have h : (1#i128 <<< 31#i32) ⦃ z => z = (2147483648#i128) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-! ## (a) The guarded loop-body transition -/

/-- **One guarded production visit.** With `i < n` and the failure flag clear the body reads
    `w_row[i]` and `x[i]`, multiplies in `i128`, adds into the accumulator and advances the index
    by exactly one; the checked operations stay on their `some` arms exactly when the next prefix
    sum is inside the `i128` window. This is the transition the work measure counts. -/
theorem body_step (w_row x : Slice I64) (n : Usize) (acc : I128) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = w_row.val.length)
    (hxlen : x.val.length = w_row.val.length)
    (hacc : acc.val = dotPrefix w_row x i.val)
    (hfit : ∀ k, 1 ≤ k → k ≤ w_row.val.length →
        I128.min ≤ dotPrefix w_row x k ∧ dotPrefix w_row x k ≤ I128.max) :
    ∃ total inext,
      row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true =
        ok (ControlFlow.cont (total, inext, true)) ∧
      total.val = dotPrefix w_row x inext.val ∧ inext.val = i.val + 1 := by
  have hbnd : i.val < w_row.val.length := by omega
  have hbndx : i.val < x.val.length := by omega
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨iw, hiw, hiwv⟩ := WP.spec_imp_exists (Slice.index_usize_spec w_row i hbnd)
  have hiw' : iw.val = w_row.val[i.val].val := by rw [hiwv]
  obtain ⟨ix, hix, hixv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hbndx)
  have hix' : ix.val = x.val[i.val].val := by rw [hixv]
  have haw : (core.convert.num.FromI128I64.from iw).val = iw.val :=
    core.convert.num.FromI128I64.from_val_eq iw
  have hbx : (core.convert.num.FromI128I64.from ix).val = ix.val :=
    core.convert.num.FromI128I64.from_val_eq ix
  have hbw := iw.hBounds
  rw [IScalarTy.I64_numBits_eq] at hbw
  norm_num at hbw
  have hbx' := ix.hBounds
  rw [IScalarTy.I64_numBits_eq] at hbx'
  norm_num at hbx'
  have hwb1 : -((2 : ℤ) ^ (63 : ℕ)) ≤ iw.val := by linarith [hbw.1]
  have hwb2 : iw.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by linarith [hbw.2]
  have hxb1 : -((2 : ℤ) ^ (63 : ℕ)) ≤ ix.val := by linarith [hbx'.1]
  have hxb2 : ix.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by linarith [hbx'.2]
  have hprange := i64_prod_range iw.val ix.val hwb1 hwb2 hxb1 hxb2
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from iw).val *
      (core.convert.num.FromI128I64.from ix).val := by
    rw [hI128min, haw, hbx]; exact hprange.1
  have hmulhi : (core.convert.num.FromI128I64.from iw).val *
      (core.convert.num.FromI128I64.from ix).val ≤ I128.max := by
    rw [hI128max, haw, hbx]; exact hprange.2
  obtain ⟨prod, hprod_eq, hprodv⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hprodval : prod.val = w_row.val[i.val].val * x.val[i.val].val := by
    rw [hprodv, haw, hbx, hiw', hix']
  have hsuc : acc.val + prod.val = dotPrefix w_row x (i.val + 1) := by
    rw [hacc, hprodval, dotPrefix_succ w_row x hbnd hbndx]
  have hk1 : 1 ≤ i.val + 1 := by omega
  have hk2 : i.val + 1 ≤ w_row.val.length := by omega
  have hpb := hfit (i.val + 1) hk1 hk2
  have hsumlo : I128.min ≤ acc.val + prod.val := by rw [hsuc]; exact hpb.1
  have hsumhi : acc.val + prod.val ≤ I128.max := by rw [hsuc]; exact hpb.2
  obtain ⟨total, htot_eq, htotv⟩ := I128_checked_add_some acc prod hsumlo hsumhi
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := w_row.property
    rw [h1]; omega
  obtain ⟨inext, hinext, hinextv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  refine ⟨total, inext, ?_, ?_, hinextv⟩
  · rw [row_dot_source.matvec.row_dot_fixed_loop.body, hiw, hix]
    simp only [lift, bind_tc_ok]
    rw [hprod_eq]
    dsimp only
    rw [htot_eq]
    dsimp only
    rw [hinext]
    simp only [bind_tc_ok]
    have hilt' : i < Slice.len w_row :=
      (UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [sliceLenVal]; exact hbnd)
    rw [if_pos hilt', if_pos trivial]
    rfl
  · rw [htotv, hsuc]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hinextv, h1]

/-- **The guarded transition without the window premise.** Each visit performs exactly one checked
    `i128` multiplication and one checked `i128` addition on the pair at `i`, advances the index by
    one, and either commits the sum and keeps the flag clear or leaves the accumulator unchanged
    and sets the flag. No third outcome exists, and the body itself never refuses. -/
theorem body_cont (w_row x : Slice I64) (n : Usize) (acc : I128) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = w_row.val.length)
    (hxlen : x.val.length = w_row.val.length) :
    ∃ iw ix acc' inext acc_ok',
      iw.val = w_row.val[i.val].val ∧ ix.val = x.val[i.val].val ∧
      row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true =
        ok (ControlFlow.cont (acc', inext, acc_ok')) ∧
      inext.val = i.val + 1 ∧
      ((∃ prod total,
          I128.checked_mul (core.convert.num.FromI128I64.from iw)
              (core.convert.num.FromI128I64.from ix) = some prod ∧
          I128.checked_add acc prod = some total ∧ acc' = total ∧ acc_ok' = true) ∨
        (acc' = acc ∧ acc_ok' = false)) := by
  have hbnd : i.val < w_row.val.length := by omega
  have hbndx : i.val < x.val.length := by omega
  obtain ⟨iw, hiw, hiwv⟩ := WP.spec_imp_exists (Slice.index_usize_spec w_row i hbnd)
  obtain ⟨ix, hix, hixv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hbndx)
  have hiwv' : iw.val = w_row.val[i.val].val := by rw [hiwv]
  have hixv' : ix.val = x.val[i.val].val := by rw [hixv]
  have h1 : (1#usize).val = 1 := by norm_num
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have hb := w_row.property
    rw [h1]; omega
  obtain ⟨inext, hinext, hinextv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  have hilt' : i < Slice.len w_row :=
    (UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [sliceLenVal]; exact hbnd)
  cases hmul : (core.convert.num.FromI128I64.from iw).checked_mul
      (core.convert.num.FromI128I64.from ix) with
  | none =>
      refine ⟨iw, ix, acc, inext, false, hiwv', hixv', ?_, hinextv, Or.inr ⟨rfl, rfl⟩⟩
      rw [row_dot_source.matvec.row_dot_fixed_loop.body, hiw, hix]
      simp only [lift, bind_tc_ok]
      rw [hmul]
      dsimp only
      rw [hinext]
      simp only [bind_tc_ok]
      rw [if_pos hilt', if_pos trivial]
      rfl
  | some prod =>
      cases hadd : acc.checked_add prod with
      | none =>
          refine ⟨iw, ix, acc, inext, false, hiwv', hixv', ?_, hinextv, Or.inr ⟨rfl, rfl⟩⟩
          rw [row_dot_source.matvec.row_dot_fixed_loop.body, hiw, hix]
          simp only [lift, bind_tc_ok]
          rw [hmul]
          dsimp only
          rw [hadd]
          dsimp only
          rw [hinext]
          simp only [bind_tc_ok]
          rw [if_pos hilt', if_pos trivial]
          rfl
      | some total =>
          refine ⟨iw, ix, total, inext, true, hiwv', hixv', ?_, hinextv, ?_⟩
          rw [row_dot_source.matvec.row_dot_fixed_loop.body, hiw, hix]
          simp only [lift, bind_tc_ok]
          rw [hmul]
          dsimp only
          rw [hadd]
          dsimp only
          rw [hinext]
          simp only [bind_tc_ok]
          rw [if_pos hilt', if_pos trivial]
          rfl
          exact Or.inl ⟨prod, total, hmul, hadd, rfl, rfl⟩

/-! ## (b) The exit branches and the work measure -/

/-- **Length exit.** With the index at or past the row width the body returns the accumulator and
    the failure flag unchanged, consuming no further product. -/
theorem body_exit_len (w_row x : Slice I64) (acc : I128) (i : Usize) (acc_ok : Bool)
    (hn : ¬ (i < Slice.len w_row)) :
    row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i acc_ok =
      ok (ControlFlow.done (acc, acc_ok)) := by
  rw [row_dot_source.matvec.row_dot_fixed_loop.body, if_neg hn]

/-- **Flag exit.** With the index still inside the row and the failure flag set the body returns
    the accumulator with the flag clear, having consumed no further product. -/
theorem body_exit_flag (w_row x : Slice I64) (acc : I128) (i : Usize)
    (hilt : i.val < (Slice.len w_row).val) :
    row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i false =
      ok (ControlFlow.done (acc, false)) := by
  have hilt' : i < Slice.len w_row :=
    (UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [sliceLenVal]; exact hilt)
  rw [row_dot_source.matvec.row_dot_fixed_loop.body, if_pos hilt', if_neg (by simp)]

/-- The loop's rank measure: the number of row positions not yet accumulated. -/
def rankOf (n : Usize) (s : I128 × Usize × Bool) : Nat := n.val - s.2.1.val

/-- **The measure is strictly decreasing.** Each guarded visit drops the rank by exactly one. This
    is the work measure the loop specification runs on, and each drop is one checked multiply-add. -/
theorem body_rank_drop (w_row x : Slice I64) (n : Usize) (acc : I128) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = w_row.val.length)
    (hxlen : x.val.length = w_row.val.length) :
    ∃ acc' inext acc_ok',
      row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true =
        ok (ControlFlow.cont (acc', inext, acc_ok')) ∧
      inext.val = i.val + 1 ∧
      rankOf n (acc', inext, acc_ok') + 1 = rankOf n (acc, i, true) ∧
      rankOf n (acc', inext, acc_ok') < rankOf n (acc, i, true) := by
  obtain ⟨iw, ix, acc', inext, acc_ok', _, _, hbody, hinext, _⟩ :=
    body_cont w_row x n acc i hilt hn hxlen
  refine ⟨acc', inext, acc_ok', hbody, hinext, ?_, ?_⟩
  · simp only [rankOf]
    omega
  · simp only [rankOf]
    omega

/-- The reachable loop invariant: the flag is clear, the index is inside the row, and the
    accumulator is the exact ℤ prefix at the index. -/
def dotLoopInv (w_row x : Slice I64) (n : Usize) (s : I128 × Usize × Bool) : Prop :=
  s.2.2 = true ∧ s.2.1.val ≤ n.val ∧ s.1.val = dotPrefix w_row x s.2.1.val

/-- **The loop terminates on the decreasing rank.** On equal-length rows whose nonempty prefix
    sums stay inside the `i128` window the loop stops with the flag clear and the exact ℤ prefix at
    the row width. -/
theorem loop_spec (w_row x : Slice I64)
    (hxlen : x.val.length = w_row.val.length)
    (hfit : ∀ k, 1 ≤ k → k ≤ w_row.val.length →
        I128.min ≤ dotPrefix w_row x k ∧ dotPrefix w_row x k ≤ I128.max) :
    row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true ⦃ b =>
      ∃ S : I128, b = (S, true) ∧ S.val = dotPrefix w_row x w_row.val.length ⦄ := by
  unfold row_dot_source.matvec.row_dot_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Usize × Bool => (Slice.len w_row).val - s.2.1.val)
    (inv := dotLoopInv w_row x (Slice.len w_row))
    (post := fun b : I128 × Bool =>
      ∃ S : I128, b = (S, true) ∧ S.val = dotPrefix w_row x w_row.val.length)
  · intro s hs
    rcases s with ⟨acc, i, acc_ok⟩
    dsimp only at hs ⊢
    obtain ⟨hok, hile, hsum⟩ := hs
    dsimp only at hok hile hsum
    rw [hok]
    by_cases hlt : i.val < (Slice.len w_row).val
    · obtain ⟨total, inext, hbody, htot, hinext⟩ :=
        body_step w_row x (Slice.len w_row) acc i hlt
          (by rw [sliceLenVal]) hxlen hsum hfit
      rw [hbody]
      have hinextle : inext.val ≤ (Slice.len w_row).val := by
        rw [hinext]; omega
      refine ⟨⟨rfl, hinextle, htot⟩, ?_⟩
      show (Slice.len w_row).val - inext.val < (Slice.len w_row).val - i.val
      omega
    · have hnot : ¬ (i < Slice.len w_row) := by
        intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len w_row)).1 hc)
      have hbody : row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true
          = ok (ControlFlow.done (acc, true)) :=
        body_exit_len w_row x acc i true hnot
      rw [hbody]
      have hie : i.val = (Slice.len w_row).val := by omega
      exact ⟨acc, rfl, by rw [hsum, hie, sliceLenVal w_row]⟩
  · refine ⟨rfl, ?_, ?_⟩
    · simp
    · show (0#i128).val = dotPrefix w_row x (0#usize).val
      exact dotPrefix_zero w_row x

/-- **A finite certificate of the actual body.** `visits` counts invocations of the extracted loop
    body and `steps` counts the continuations, that is the visits that consume one row position.
    Divergence has no constructor. -/
inductive RowTrace {State : Type u} {Value : Type v}
    (body : State → Result (ControlFlow State Value)) :
    State → Result Value → Nat → Nat → Prop where
  | done {state value} (observed : body state = ok (ControlFlow.done value)) :
      RowTrace body state (ok value) 1 0
  | failure {state error} (observed : body state = fail error) :
      RowTrace body state (fail error) 1 0
  | cont {state next result visits steps} (observed : body state = ok (ControlFlow.cont next))
      (tail : RowTrace body next result visits steps) :
      RowTrace body state result (visits + 1) (steps + 1)

/-- The certificate projects to the actual partial-fixpoint loop, not to an evaluator copy. -/
theorem row_trace_projection {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {state : State} {result : Result Value}
    {visits steps : Nat} (trace : RowTrace body state result visits steps) :
    loop body state = result := by
  induction trace with
  | done observed =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
  | failure observed =>
      unfold loop
      rw [observed]
      simp only [bind_fail]
  | cont observed tail ih =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
      exact ih

/-- Every certificate has exactly one terminating visit, so `visits = steps + 1`: the
    multiply-add count is the visit count minus the terminating visit. -/
theorem row_trace_count {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {state : State} {result : Result Value}
    {visits steps : Nat} (trace : RowTrace body state result visits steps) :
    visits = steps + 1 := by
  induction trace with
  | done _ => rfl
  | failure _ => rfl
  | cont _ tail ih => omega

/-- Two certificates of the same state agree on the outcome and on both counts. -/
theorem row_trace_deterministic {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {state : State}
    {result₁ result₂ : Result Value} {visits₁ steps₁ visits₂ steps₂ : Nat}
    (first : RowTrace body state result₁ visits₁ steps₁)
    (second : RowTrace body state result₂ visits₂ steps₂) :
    result₁ = result₂ ∧ visits₁ = visits₂ ∧ steps₁ = steps₂ := by
  induction first generalizing result₂ visits₂ steps₂ with
  | done observed =>
      cases second <;> simp_all
  | failure observed =>
      cases second <;> simp_all
  | @cont state next result visits steps observed tail ih =>
      cases second with
      | done observed₂ => simp_all
      | failure observed₂ => simp_all
      | @cont state₂ next₂ result₂ visits₂ steps₂ observed₂ tail₂ =>
          have eqn : next = next₂ := by simpa using observed.symm.trans observed₂
          cases eqn
          obtain ⟨hr, hv, hs⟩ := ih tail₂
          exact ⟨hr, by omega, by omega⟩

/-- **The closed-form count.** On the `i128` window, from a state whose index is `i.val ≤ n.val`,
    the actual body is invoked once per remaining row position plus once to terminate:
    `k` checked multiply-adds in `k + 1` invocations, where `k` is the rank the state carries. -/
theorem window_trace_aux (w_row x : Slice I64) (n : Usize)
    (hn : n.val = w_row.val.length) (hxlen : x.val.length = w_row.val.length)
    (hfit : ∀ j, 1 ≤ j → j ≤ w_row.val.length →
        I128.min ≤ dotPrefix w_row x j ∧ dotPrefix w_row x j ≤ I128.max) :
    ∀ (k : Nat) (acc : I128) (i : Usize), n.val - i.val = k →
      acc.val = dotPrefix w_row x i.val →
      ∃ S : I128, RowTrace (dotBodyFn w_row x) (acc, i, true) (ok (S, true)) (k + 1) k := by
  intro k
  induction k using Nat.strong_induction_on with
  | h k ih =>
      intro acc i hrank hacc
      by_cases hlt : i.val < n.val
      · obtain ⟨total, inext, hbody, htot, hinext⟩ :=
          body_step w_row x n acc i hlt hn hxlen hacc hfit
        have hnext : n.val - inext.val = k - 1 := by rw [hinext]; omega
        obtain ⟨S, htail⟩ := ih (k - 1) (by omega) total inext hnext htot
        have hbody' : dotBodyFn w_row x (acc, i, true) =
            ok (ControlFlow.cont (total, inext, true)) := by
          show row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true = _
          exact hbody
        refine ⟨S, ?_⟩
        convert RowTrace.cont hbody' htail using 1 <;> omega
      · have hk0 : k = 0 := by omega
        have hnlt : ¬ (i < Slice.len w_row) := by
          intro hc
          have hc' := (UScalar.lt_equiv i (Slice.len w_row)).1 hc
          rw [sliceLenVal w_row] at hc'
          omega
        have hbody : row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true
            = ok (ControlFlow.done (acc, true)) := body_exit_len w_row x acc i true hnlt
        refine ⟨acc, ?_⟩
        rw [hk0]
        exact RowTrace.done hbody

/-- **The exact work measure of the executing row loop.** From the production entry state
    `acc = 0`, `i = 0`, flag clear, the actual body is invoked `n + 1` times and performs exactly
    `n` checked `i128` multiply-adds, `n` being the row width; every certificate of those counts
    agrees, so the count is the loop's own and not one derivation's. -/
theorem entry_visits_exact (w_row x : Slice I64) (n : Usize)
    (hn : n.val = w_row.val.length) (hxlen : x.val.length = w_row.val.length)
    (hfit : ∀ k, 1 ≤ k → k ≤ w_row.val.length →
        I128.min ≤ dotPrefix w_row x k ∧ dotPrefix w_row x k ≤ I128.max) :
    (∃ S : I128, RowTrace (dotBodyFn w_row x) (0#i128, 0#usize, true) (ok (S, true))
        (n.val + 1) n.val) ∧
      (∀ result (visits steps : Nat),
          RowTrace (dotBodyFn w_row x) (0#i128, 0#usize, true) result visits steps →
            visits = n.val + 1 ∧ steps = n.val) := by
  obtain ⟨S, htrace⟩ := window_trace_aux w_row x n hn hxlen hfit n.val (0#i128) (0#usize) (by simp)
    (by simp [dotPrefix])
  have huniq : ∀ result (visits steps : Nat),
      RowTrace (dotBodyFn w_row x) (0#i128, 0#usize, true) result visits steps →
      visits = n.val + 1 ∧ steps = n.val := by
    intro result visits steps hsecond
    obtain ⟨hr, hv, hs⟩ := row_trace_deterministic htrace hsecond
    exact ⟨by omega, by omega⟩
  exact ⟨⟨S, htrace⟩, huniq⟩

/-- **The refusal-path count.** A state whose failure flag is set leaves the loop after exactly one
    visit and zero multiply-adds, whatever the index inside the row. -/
theorem flag_trace (w_row x : Slice I64) (acc : I128) (i : Usize) :
    RowTrace (dotBodyFn w_row x) (acc, i, false) (ok (acc, false)) 1 0 := by
  have hbody : row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i false
      = ok (ControlFlow.done (acc, false)) := by
    by_cases hlt : i.val < (Slice.len w_row).val
    · rw [body_exit_flag w_row x acc i hlt]
    · exact body_exit_len w_row x acc i false (by simpa [sliceLenVal] using hlt)
  exact RowTrace.done hbody

/-- A flagged state leaves the actual loop immediately, whatever the index. -/
theorem flag_loop (w_row x : Slice I64) (acc : I128) (i : Usize) :
    row_dot_source.matvec.row_dot_fixed_loop w_row x acc i false = ok (acc, false) := by
  show row_dot_source.matvec.row_dot_fixed_loop w_row x acc i false = _
  exact row_trace_projection (flag_trace w_row x acc i)

/-! ## (c) The success branch -/

/-- The extracted entry point after the length test and a successful accumulator loop, as a `do`
    block over the accumulator value. -/
theorem row_dot_fixed_unfold (w_row x : Slice I64) (S : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true)) :
    row_dot_source.matvec.row_dot_fixed w_row x =
      (do
        let rounded ← row_dot_source.value.round_shift_32 S
        let i2 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if rounded > i2
        then ok (core.result.Result.Err row_dot_source.matvec.MatvecError.OutputOverflow)
        else
          let i3 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if rounded < i3
          then ok (core.result.Result.Err row_dot_source.matvec.MatvecError.OutputOverflow)
          else
            let i4 ← lift (IScalar.cast .I64 rounded)
            ok (core.result.Result.Ok i4)) := by
  have hlen' : Slice.len w_row = Slice.len x := by
    apply UScalar.eq_of_val_eq
    rw [sliceLenVal w_row, sliceLenVal x]
    exact hlen
  have hcond : (Slice.len w_row != Slice.len x) = false := by simp [hlen']
  rw [row_dot_source.matvec.row_dot_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **The success branch is exact.** On equal-length rows whose prefix sums stay inside the `i128`
    window, and a rounded accumulator inside the `i64` window, the extracted entry point returns
    `Ok` of exactly that rounded value. -/
theorem row_dot_success (w_row x : Slice I64) (S : I128) (rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hround : row_dot_source.value.round_shift_32 S = ok rounded)
    (hroundb : -((2 : ℤ) ^ (63 : ℕ)) ≤ rounded.val ∧ rounded.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ y : I64, row_dot_source.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Ok y) ∧ y.val = rounded.val := by
  have hmaxv : (IScalar.cast .I128 core.num.I64.MAX).val = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2 : ℤ) ^ (63 : ℕ) - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hminv : (IScalar.cast .I128 core.num.I64.MIN).val = -(2 : ℤ) ^ (63 : ℕ) := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2 : ℤ) ^ (63 : ℕ) := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2 : ℤ) ^ (63 : ℕ))
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hmax : ¬ (rounded > IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv] at hc
    omega
  have hmin : ¬ (rounded < IScalar.cast .I128 core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, hminv] at hc
    omega
  have hcastval : (IScalar.cast .I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  refine ⟨IScalar.cast .I64 rounded, ?_, hcastval⟩
  rw [row_dot_fixed_unfold w_row x S hlen hloop]
  rw [hround]
  simp only [lift, bind_tc_ok]
  rw [if_neg hmax]
  rw [if_neg hmin]

/-- The reachable witness row `[1, 1]` in Q32.32: length two. -/
def witRow : Slice I64 := Slice.from [1#i64, 1#i64] (by simp; scalar_tac)

/-- The reachable witness activation `[1, 1]` in Q32.32: length two. -/
def witVec : Slice I64 := Slice.from [1#i64, 1#i64] (by simp; scalar_tac)

theorem witRow_len : witRow.val.length = 2 := by simp [witRow]

theorem witVec_len : witVec.val.length = 2 := by simp [witVec]

/-- The joint `i128` window for the witness rows: every nonempty prefix sum stays in range. -/
theorem wit_fit : ∀ k, 1 ≤ k → k ≤ witRow.val.length →
    I128.min ≤ dotPrefix witRow witVec k ∧ dotPrefix witRow witVec k ≤ I128.max := by
  intro k hk1 hk2
  rw [witRow_len] at hk2
  have hk : k = 1 ∨ k = 2 := by omega
  rcases hk with hk' | hk'
  · rw [hk']
    norm_num [dotPrefix, witRow, witVec, I128.min, IScalar.min, I128.numBits,
      I128.max, IScalar.max]
  · rw [hk']
    norm_num [dotPrefix, witRow, witVec, I128.min, IScalar.min, I128.numBits,
      I128.max, IScalar.max]

/-- **The reachable cost of the witness.** The two-element witness row needs three invocations of
    the actual loop body and two checked multiply-adds, and accumulates the exact ℤ prefix `2`. -/
theorem wit_visits :
    ∃ S : I128, S.val = 2 ∧
      RowTrace (dotBodyFn witRow witVec) (0#i128, 0#usize, true) (ok (S, true)) 3 2 := by
  obtain ⟨S, htrace⟩ :=
    (entry_visits_exact witRow witVec 2#usize (by norm_num [witRow_len])
      (by rw [witVec_len, witRow_len]) wit_fit).1
  obtain ⟨b, hb, hS, hSval⟩ := WP.spec_imp_exists
    (loop_spec witRow witVec (by rw [witVec_len, witRow_len]) wit_fit)
  have hproj : row_dot_source.matvec.row_dot_fixed_loop witRow witVec 0#i128 0#usize true =
      ok (S, true) := by
    show row_dot_source.matvec.row_dot_fixed_loop witRow witVec 0#i128 0#usize true = _
    exact row_trace_projection htrace
  rw [hproj] at hb
  have hEq : (S, true) = b := Result.ok_injective hb
  rw [← hEq] at hSval
  have hSeq : S = hS := congrArg Prod.fst hSval.1
  refine ⟨S, ?_, htrace⟩
  rw [hSeq, hSval.2, witRow_len]
  norm_num [dotPrefix, witRow, witVec]

/-- **The reachable success witness, at the actual entry point.** The extracted `row_dot_fixed` on
    two length-two rows of `1` returns `Ok 0`: the loop accumulates the exact ℤ prefix `2` in two
    checked multiply-adds, the round-to-nearest `>> 32` epilogue of `2` is `0`, and the `i64`
    narrowing accepts `0`. -/
theorem wit_success :
    row_dot_source.matvec.row_dot_fixed witRow witVec = ok (core.result.Result.Ok 0#i64) := by
  obtain ⟨S, hS, htrace⟩ := wit_visits
  have hloop : row_dot_source.matvec.row_dot_fixed_loop witRow witVec 0#i128 0#usize true =
      ok (S, true) := by
    show row_dot_source.matvec.row_dot_fixed_loop witRow witVec 0#i128 0#usize true = _
    exact row_trace_projection htrace
  have hS2 : S = 2#i128 := IScalar.eq_of_val_eq (by rw [hS]; norm_num)
  have hwin : I128.min ≤ (2 : ℤ) + 2147483648 ∧ (2 : ℤ) + 2147483648 ≤ I128.max := by
    norm_num [I128.min, IScalar.min, I128.numBits, I128.max, IScalar.max]
  have hround : row_dot_source.value.round_shift_32 S = ok (0#i128) := by
    rw [hS2]
    unfold row_dot_source.value.round_shift_32
    rw [I128_shiftLeft_31]
    simp only [bind_tc_ok]
    obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
      (I128.add_spec (x := 2#i128) (y := 2147483648#i128) (by exact hwin.1) (by exact hwin.2))
    rw [hi1]
    simp only [bind_tc_ok]
    obtain ⟨y, hy, hyv⟩ := I128_shiftRight_32_val i1
    have hy0 : y = 0#i128 := IScalar.eq_of_val_eq (by rw [hyv, hi1v]; norm_num)
    rw [hy0] at hy
    exact hy
  have hroundb : -((2 : ℤ) ^ (63 : ℕ)) ≤ (0#i128).val ∧
      (0#i128).val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by norm_num
  obtain ⟨y, hy, hyval⟩ :=
    row_dot_success witRow witVec S (0#i128) (by rw [witRow_len, witVec_len]) hloop hround hroundb
  rw [hy]
  have hy0 : y = 0#i64 := IScalar.eq_of_val_eq (by rw [hyval]; norm_num)
  rw [hy0]

/-! ## (d) The refusal branches, exactly -/

/-- **Length refusal.** Unequal row lengths return `Err (RowLength 0 (Slice.len w_row)
    (Slice.len x))` before any product is read. -/
theorem row_dot_row_length_refuses (w_row x : Slice I64)
    (h : w_row.val.length ≠ x.val.length) :
    row_dot_source.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err
        (row_dot_source.matvec.MatvecError.RowLength 0#usize (Slice.len w_row) (Slice.len x))) := by
  have hlen' : Slice.len w_row ≠ Slice.len x := by
    intro hc
    have hv : (Slice.len w_row).val = (Slice.len x).val :=
      congrArg (fun u : Usize => u.val) hc
    rw [sliceLenVal w_row, sliceLenVal x] at hv
    exact h hv
  have hcond : (Slice.len w_row != Slice.len x) = true := by simpa using hlen'
  rw [row_dot_source.matvec.row_dot_fixed]
  dsimp only
  rw [if_pos hcond]

/-- **Accumulator refusal.** A loop that stops with the failure flag set returns
    `Err AccumulatorOverflow`. -/
theorem row_dot_acc_refuses (w_row x : Slice I64) (S : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true =
      ok (S, false)) :
    row_dot_source.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err row_dot_source.matvec.MatvecError.AccumulatorOverflow) := by
  have hlen' : Slice.len w_row = Slice.len x := by
    apply UScalar.eq_of_val_eq
    rw [sliceLenVal w_row, sliceLenVal x]
    exact hlen
  have hcond : (Slice.len w_row != Slice.len x) = false := by simp [hlen']
  rw [row_dot_source.matvec.row_dot_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [hloop]
  simp

/-- **Upper narrowing refusal.** A rounded value above `i64::MAX` returns `Err OutputOverflow`. -/
theorem row_dot_out_high_refuses (w_row x : Slice I64) (S rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hround : row_dot_source.value.round_shift_32 S = ok rounded)
    (hhigh : rounded > IScalar.cast .I128 core.num.I64.MAX) :
    row_dot_source.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err row_dot_source.matvec.MatvecError.OutputOverflow) := by
  rw [row_dot_fixed_unfold w_row x S hlen hloop]
  rw [hround]
  simp only [lift, bind_tc_ok]
  rw [if_pos hhigh]

/-- **Lower narrowing refusal.** A rounded value below `i64::MIN`, with the upper test already
    passed, returns `Err OutputOverflow`. -/
theorem row_dot_out_low_refuses (w_row x : Slice I64) (S rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hround : row_dot_source.value.round_shift_32 S = ok rounded)
    (hhigh : ¬ (rounded > IScalar.cast .I128 core.num.I64.MAX))
    (hlow : rounded < IScalar.cast .I128 core.num.I64.MIN) :
    row_dot_source.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err row_dot_source.matvec.MatvecError.OutputOverflow) := by
  rw [row_dot_fixed_unfold w_row x S hlen hloop]
  rw [hround]
  simp only [lift, bind_tc_ok]
  rw [if_neg hhigh, if_pos hlow]

end RowDotResource

open Lean Elab Command in
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

#resource_audit RowDotResource.body_step
#resource_audit RowDotResource.body_cont
#resource_audit RowDotResource.body_exit_len
#resource_audit RowDotResource.body_exit_flag
#resource_audit RowDotResource.body_rank_drop
#resource_audit RowDotResource.loop_spec
#resource_audit RowDotResource.row_trace_projection
#resource_audit RowDotResource.row_trace_count
#resource_audit RowDotResource.row_trace_deterministic
#resource_audit RowDotResource.window_trace_aux
#resource_audit RowDotResource.entry_visits_exact
#resource_audit RowDotResource.flag_trace
#resource_audit RowDotResource.flag_loop
#resource_audit RowDotResource.row_dot_success
#resource_audit RowDotResource.wit_visits
#resource_audit RowDotResource.wit_success
#resource_audit RowDotResource.row_dot_row_length_refuses
#resource_audit RowDotResource.row_dot_acc_refuses
#resource_audit RowDotResource.row_dot_out_high_refuses
#resource_audit RowDotResource.row_dot_out_low_refuses

run_cmd Lean.logInfo "ROW_DOT_KERNEL_OK"