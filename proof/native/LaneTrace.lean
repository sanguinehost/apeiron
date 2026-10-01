import ExactDotSource
import LoopTrace
import Apeiron.Trichotomy.AnyRing

/-
SCOPE: proof/native/LaneTrace.lean, proof/native/lane_trace_gate.py.
REGISTER: model lane count and ordered integer sum for extracted inner loop; cost excluded.
CORPUS: LoopTrace.lean, ExactDotSource.lean, Apeiron.Trichotomy.AnyRing.
REFERENCE: target/step-sol-c5hbky9_/aeneas/backends/lean/ExactDotSource.lean,
           Apeiron/Trichotomy/AnyRing.lean, Aeneas/Std/SliceIter.lean.
GATE: python3 proof/native/lane_trace_gate.py --controls; 540s whole invocation.
GRADE: model theorem on extracted body; Rust reachability, GPU, cost excluded.
-/

set_option autoImplicit false
set_option linter.apeiron.notation false

namespace LaneTrace

open Aeneas Aeneas.Std Result ControlFlow apeiron LoopTrace WP

/-- The signed-Int product is the diagonal channel of canonical `ringMul` for any parameter σ. -/
theorem ringMul_int_diag (σ a b : Int) :
    Apeiron.Trichotomy.AnyRing.ringMul σ (a, 0) (b, 0) = (a * b, 0) := by
  simp

/-- Projection to channel 0 preserves exact scalar multiplication. -/
theorem ringMul_channel0 (σ a b : Int) :
    (Apeiron.Trichotomy.AnyRing.ringMul σ (a, 0) (b, 0)).1 = a * b := by
  simp

/-- Modeled byte slice of length 32 with uniform input byte 1. -/
def sampleRow : Slice U8 :=
  Slice.from (List.replicate 32 1#u8) (by simp; scalar_tac)

/-- Modeled QuantAct code vector of length 32 with uniform weight byte 2. -/
def sampleCodes : alloc.vec.Vec I8 := {
  slice := Slice.from (List.replicate 32 2#i8) (by simp; scalar_tac),
  cap := 32#usize
}

/-- Modeled scale vector (not read by the inner loop). -/
def sampleScales : alloc.vec.Vec I64 := {
  slice := Slice.new I64,
  cap := 0#usize
}

/-- Modeled QuantAct structure. -/
def sampleQ : engine.chart.QuantAct := {
  scales := sampleScales,
  codes := sampleCodes
}

/-- The loop state: enumerate iterator over inner byte slice and i32 accumulator. -/
abbrev InnerState := (core.iter.adapters.enumerate.Enumerate (core.slice.iter.Iter U8)) × I32

/-- Extracted inner loop body instantiated with modeled activation and block 0. -/
def innerBody (p : InnerState) : Result (ControlFlow InnerState I32) :=
  engine.kern.gemm.exact_dot_loop0_loop0.body sampleQ 0#usize p.1 p.2

/-- Lane charge: charges 1 per active lane visit, 0 at termination. -/
def innerCharge (s : InnerState) : Nat := if s.1.iter.i < 32 then 1 else 0

/-- Modeled loop state at lane index k with accumulator 2*k. -/
def modelState (k : Nat) (hk : k ≤ 32) : InnerState :=
  ({ iter := ⟨sampleRow, k⟩, count := Usize.ofNatCore k (by scalar_tac) },
   IScalar.ofIntCore (2 * k) (by scalar_tac))

/-- Initial inner iterator starting at index 0 and count 0. -/
def initialIter : core.iter.adapters.enumerate.Enumerate (core.slice.iter.Iter U8) :=
  { iter := ⟨sampleRow, 0⟩, count := 0#usize }

/-- Initial state with dot accumulator 0. -/
def initialState : InnerState := (initialIter, 0#i32)

/-- Usize multiplication by 0 evaluates to 0 without overflow. -/
theorem usize_zero_mul (y : Usize) : (0#usize : Usize) * y = ok 0#usize := by
  have h_spec := Usize.mul_spec (x := 0#usize) (y := y) (by
    have : (0#usize : Usize).val = 0 := rfl
    rw [this, Nat.zero_mul]
    exact Nat.zero_le _)
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists h_spec
  have : (0#usize : Usize).val = 0 := rfl
  rw [this, Nat.zero_mul] at hzv
  have hz_eq : z = 0#usize := UScalar.eq_of_val_eq hzv
  subst hz_eq
  exact hz

/-- Usize addition with 0 evaluates to identity without overflow. -/
theorem usize_zero_add (l : Usize) : (0#usize : Usize) + l = ok l := by
  have h_spec := Usize.add_spec (x := 0#usize) (y := l) (by
    have : (0#usize : Usize).val = 0 := rfl
    rw [this, Nat.zero_add]
    scalar_tac)
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists h_spec
  have : (0#usize : Usize).val = 0 := rfl
  rw [this, Nat.zero_add] at hzv
  have hz_eq : z = l := UScalar.eq_of_val_eq hzv
  subst hz_eq
  exact hz

/-- In-bounds slice lookup via usize index evaluates to list element. -/
theorem slice_index_usize_eq {α : Type} (s : Slice α) (l : Usize) (h : l.val < s.val.length) :
    s.index_usize l = ok s.val[l.val] := by
  have h_spec := Slice.index_usize_spec s l (by
    show l.val < s.length
    exact h)
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists h_spec
  subst hzv
  exact hz

/-- Slice iterator step in active range advances index and yields current byte. -/
theorem slice_iter_next_some {T : Type} (it : core.slice.iter.Iter T) (h : it.i < it.slice.val.length) :
    core.slice.iter.IteratorSliceIter.next it = ok (some (it.slice.val[it.i]), { it with i := it.i + 1 }) := by
  have hlen : it.i < it.slice.len := by
    show it.i < it.slice.len.val
    rw [Slice.len_val]
    exact h
  unfold core.slice.iter.IteratorSliceIter.next
  rw [dif_pos hlen]
  rfl

/-- Concrete step composition of the extracted inner loop body. -/
theorem exact_dot_body_step_eq
    (q : engine.chart.QuantAct) (b : Usize)
    (iter iter_next : core.iter.adapters.enumerate.Enumerate (core.slice.iter.Iter U8))
    (dot dot1 i6 : I32) (l : Usize) (c : U8) (i i4 : I8) (i1 i5 : I32) (s : Slice I8) (i2 i3 : Usize)
    (hn : core.iter.adapters.enumerate.IteratorEnumerate.next
      (core.iter.traits.iterator.IteratorSliceIter U8) iter = ok (some (l, c), iter_next))
    (hc : core.num.U8.cast_signed c = ok i)
    (hl1 : lift (core.convert.num.FromI32I8.from i) = ok i1)
    (hs : engine.chart.QuantAct.impl.codes q = ok s)
    (h2 : b * engine.kern.gemm.ACT_QK = ok i2)
    (h3 : i2 + l = ok i3)
    (h4 : Slice.index_usize s i3 = ok i4)
    (hl2 : lift (core.convert.num.FromI32I8.from i4) = ok i5)
    (h6 : i1 * i5 = ok i6)
    (hd : dot + i6 = ok dot1) :
    engine.kern.gemm.exact_dot_loop0_loop0.body q b iter dot = ok (cont (iter_next, dot1)) := by
  unfold engine.kern.gemm.exact_dot_loop0_loop0.body
  rw [hn]
  simp only [bind_tc_ok]
  show (do
    let i' ← core.num.U8.cast_signed c
    let i1' ← lift (core.convert.num.FromI32I8.from i')
    let s' ← engine.chart.QuantAct.impl.codes q
    let i2' ← b * engine.kern.gemm.ACT_QK
    let i3' ← i2' + l
    let i4' ← s'.index_usize i3'
    let i5' ← lift (core.convert.num.FromI32I8.from i4')
    let i6' ← i1' * i5'
    let dot1' ← dot + i6'
    ok (cont (iter_next, dot1'))) = ok (cont (iter_next, dot1))
  rw [hc]
  simp only [bind_tc_ok]
  rw [hl1]
  simp only [bind_tc_ok]
  rw [hs]
  simp only [bind_tc_ok]
  rw [h2]
  simp only [bind_tc_ok]
  rw [h3]
  simp only [bind_tc_ok]
  rw [h4]
  simp only [bind_tc_ok]
  rw [hl2]
  simp only [bind_tc_ok]
  rw [h6]
  simp only [bind_tc_ok]
  rw [hd]
  simp only [bind_tc_ok]

/-- Each byte of the modeled activation row is 1#u8. -/
theorem sampleRow_get (k : Nat) (hk : k < 32) : sampleRow.val[k] = 1#u8 := by
  have h_eq : sampleRow.val = List.replicate 32 1#u8 := Aeneas.Data.ListN.ListN.from_to_inverse
  have h_len : k < (List.replicate 32 1#u8).length := by simp; omega
  have h_rep := List.getElem_replicate (a := 1#u8) h_len
  have this : sampleRow.val[k] = (List.replicate 32 1#u8)[k] :=
    List.getElem_of_eq (h := h_eq) (i := k) (by simp [sampleRow]; omega)
  exact this.trans h_rep

/-- Each byte of the modeled weight code vector is 2#i8. -/
theorem sampleCodes_get (k : Nat) (hk : k < 32) : sampleCodes.slice.val[k] = 2#i8 := by
  have h_eq : sampleCodes.slice.val = List.replicate 32 2#i8 := Aeneas.Data.ListN.ListN.from_to_inverse
  have h_len : k < (List.replicate 32 2#i8).length := by simp; omega
  have h_rep := List.getElem_replicate (a := 2#i8) h_len
  have this : sampleCodes.slice.val[k] = (List.replicate 32 2#i8)[k] :=
    List.getElem_of_eq (h := h_eq) (i := k) (by simp [sampleCodes]; omega)
  exact this.trans h_rep

/-- At exhausted iterator, the inner loop terminates with done. -/
theorem exact_dot_body_done
    (q : engine.chart.QuantAct)
    (iter : core.iter.adapters.enumerate.Enumerate (core.slice.iter.Iter U8))
    (dot : I32)
    (hdone : ¬ (iter.iter.i < iter.iter.slice.len)) :
    engine.kern.gemm.exact_dot_loop0_loop0.body q 0#usize iter dot = ok (done dot) := by
  have h_proj : (core.iter.traits.iterator.IteratorSliceIter U8).next = core.slice.iter.IteratorSliceIter.next := rfl
  have h_none_spec := core.iter.adapters.enumerate.IteratorEnumerate.next_none_spec
    (core.iter.traits.iterator.IteratorSliceIter U8) iter iter.iter
    (by
      rw [h_proj]
      unfold core.slice.iter.IteratorSliceIter.next
      rw [dif_neg hdone]
      simp [spec_ok])
  obtain ⟨⟨opt, iter'⟩, heq, h1, h2, h3⟩ := WP.spec_imp_exists h_none_spec
  have h_eq : iter' = iter := by cases iter'; cases iter; simp_all
  subst h1; subst h_eq
  unfold engine.kern.gemm.exact_dot_loop0_loop0.body
  rw [heq]
  simp

/-- Terminal observation at lane 32 returns done with accumulator 64#i32. -/
theorem model_done : innerBody (modelState 32 (by decide)) = ok (done 64#i32) := by
  unfold innerBody
  have h_done : ¬ ((modelState 32 (by decide)).1.iter.i < (modelState 32 (by decide)).1.iter.slice.len) := by
    simp [modelState, sampleRow]
  have h_res := exact_dot_body_done sampleQ (modelState 32 (by decide)).1 (modelState 32 (by decide)).2 h_done
  have h64 : (modelState 32 (by decide)).2 = 64#i32 := rfl
  rw [h64] at h_res
  exact h_res

/-- Each lane transition from k to k+1 performs signed product and accumulation. -/
theorem model_step (k : Nat) (hk : k < 32) :
    innerBody (modelState k (by omega)) = ok (cont (modelState (k + 1) (by omega))) := by
  have hk_le : k ≤ 32 := by omega
  have hk1_le : k + 1 ≤ 32 := by omega
  set st_k := modelState k hk_le
  set st_k1 := modelState (k + 1) hk1_le
  have h_proj : (core.iter.traits.iterator.IteratorSliceIter U8).next = core.slice.iter.IteratorSliceIter.next := rfl
  have h_len : k < sampleRow.val.length := by simp [sampleRow]; omega
  have h_some_spec := core.iter.adapters.enumerate.IteratorEnumerate.next_some_spec
    (core.iter.traits.iterator.IteratorSliceIter U8) st_k.1 (sampleRow.val[k])
    { slice := sampleRow, i := k + 1 } (by
      rw [h_proj]
      have := slice_iter_next_some (T := U8) ⟨sampleRow, k⟩ h_len
      have : (core.slice.iter.IteratorSliceIter.next (T := U8) { slice := sampleRow, i := k } ⦃
        fun p => p.1 = some (sampleRow.val[k]) ∧ p.2 = { slice := sampleRow, i := k + 1 } ⦄) := by
        rw [this]
        simp [spec_ok]
      exact this) (by simp [st_k, modelState]; scalar_tac)
  obtain ⟨⟨opt, iter'⟩, heq, h1, h2, h3⟩ := WP.spec_imp_exists h_some_spec
  have h_opt : opt = some (st_k.1.count, sampleRow.val[k]) := h1
  have h_iter' : iter' = st_k1.1 := by
    cases iter'
    rename_i it cnt
    have h_it : it = { slice := sampleRow, i := k + 1 } := h2
    have h_cnt : cnt = st_k1.1.count := by
      apply UScalar.eq_of_val_eq
      have hval1 : cnt.val = st_k.1.count.val + 1 := h3
      have hval2 : st_k.1.count.val = k := by simp [st_k, modelState]
      have hval3 : st_k1.1.count.val = k + 1 := by simp [st_k1, modelState]
      omega
    have : st_k1.1 = { iter := it, count := cnt } := by
      subst h_it; subst h_cnt; rfl
    exact this.symm
  subst h_opt; subst h_iter'
  have h_row_val : sampleRow.val[k] = 1#u8 := sampleRow_get k hk
  have h_cast : core.num.U8.cast_signed (sampleRow.val[k]) = ok 1#i8 := by
    rw [h_row_val]; rfl
  have h_lift1 : lift (core.convert.num.FromI32I8.from 1#i8) = ok 1#i32 := rfl
  have h_codes : engine.chart.QuantAct.impl.codes sampleQ = ok sampleCodes.slice := rfl
  have h_h2 : (0#usize : Usize) * engine.kern.gemm.ACT_QK = ok 0#usize :=
    usize_zero_mul engine.kern.gemm.ACT_QK
  have h_h3 : (0#usize : Usize) + st_k.1.count = ok st_k.1.count :=
    usize_zero_add st_k.1.count
  have h_cnt_k : st_k.1.count.val = k := by simp [st_k, modelState]
  have h_code_len : st_k.1.count.val < sampleCodes.slice.val.length := by
    simp [sampleCodes, h_cnt_k]; omega
  have h_idx_spec := slice_index_usize_eq sampleCodes.slice st_k.1.count h_code_len
  have h_code_val : sampleCodes.slice.val[st_k.1.count.val] = 2#i8 := by
    have h_eq_idx : (sampleCodes.slice.val[st_k.1.count.val] : I8) = sampleCodes.slice.val[k] := by
      congr 1
    exact h_eq_idx.trans (sampleCodes_get k hk)
  have h_h4 : sampleCodes.slice.index_usize st_k.1.count = ok 2#i8 := by
    rw [h_idx_spec, h_code_val]
  have h_lift2 : lift (core.convert.num.FromI32I8.from 2#i8) = ok 2#i32 := rfl
  have h_mul : (1#i32 : I32) * 2#i32 = ok 2#i32 := rfl
  have h_add_spec := I32.add_spec (x := st_k.2) (y := 2#i32) (by
    simp [st_k, modelState, I32.min_eq]; omega) (by
    simp [st_k, modelState, I32.max_eq]; omega)
  obtain ⟨dot', hdot', hdot'v⟩ := WP.spec_imp_exists h_add_spec
  have h_dot'_eq : dot' = st_k1.2 := by
    apply IScalar.eq_of_val_eq
    have h1 : st_k.2.val = 2 * (k : Int) := by simp [st_k, modelState]
    have h2 : (2#i32 : I32).val = 2 := rfl
    have h3 : st_k1.2.val = 2 * ((k : Int) + 1) := by simp [st_k1, modelState]
    rw [hdot'v, h1, h2, h3]
    omega
  subst h_dot'_eq
  unfold innerBody
  exact exact_dot_body_step_eq sampleQ 0#usize st_k.1 st_k1.1 st_k.2 st_k1.2 2#i32
    st_k.1.count (sampleRow.val[k]) 1#i8 2#i8 1#i32 2#i32 sampleCodes.slice 0#usize st_k.1.count
    heq h_cast h_lift1 h_codes h_h2 h_h3 h_h4 h_lift2 h_mul hdot'

/-- Suffix trace by backwards induction on the remaining lane count n = 32 - k. -/
theorem model_trace_suffix (n : Nat) (hn : n ≤ 32) :
    Trace innerBody innerCharge (modelState (32 - n) (by omega)) (ok 64#i32) n := by
  induction n with
  | zero =>
      have hdone := model_done
      have hcharge : innerCharge (modelState 32 (by decide)) = 0 := by
        simp [innerCharge, modelState]
      have ht := Trace.done (body := innerBody) (charge := innerCharge) hdone
      rw [hcharge] at ht
      exact ht
  | succ n ih =>
      have hn_lt : n < 32 := by omega
      have ih' := ih (by omega)
      have hstep := model_step (32 - (n + 1)) (by omega)
      have h_succ : 32 - (n + 1) + 1 = 32 - n := by omega
      have hstep' : innerBody (modelState (32 - (n + 1)) (by omega)) =
        ok (cont (modelState (32 - n) (by omega))) := by
        have := model_step (32 - (n + 1)) (by omega)
        have h_eq : (modelState (32 - (n + 1) + 1) (by omega)) = (modelState (32 - n) (by omega)) := by
          congr 1
        rw [← h_eq]
        exact this
      have hcharge : innerCharge (modelState (32 - (n + 1)) (by omega)) = 1 := by
        simp [innerCharge, modelState]; omega
      have ht := Trace.cont (body := innerBody) (charge := innerCharge) hstep' ih'
      rw [hcharge] at ht
      have h_sum : 1 + n = n + 1 := by omega
      rw [h_sum] at ht
      exact ht

/-- The actual extracted inner loop visits exactly 32 lanes and returns 64#i32. -/
theorem model_lane_trace :
    Trace (fun p => engine.kern.gemm.exact_dot_loop0_loop0.body sampleQ 0#usize p.1 p.2)
      innerCharge initialState (ok 64#i32) 32 := by
  have ht := model_trace_suffix 32 (by decide)
  have h_init : modelState (32 - 32) (by omega) = initialState := rfl
  rw [h_init] at ht
  exact ht

/-- Projection to the partial-fixpoint loop computes the ordered sum 64#i32. -/
theorem model_exact_dot_loop :
    engine.kern.gemm.exact_dot_loop0_loop0 initialIter sampleQ 0#usize 0#i32 = ok 64#i32 :=
  LoopTrace.projection model_lane_trace

/-- Lane count determinism: any finite observation visiting the same initial state visits exactly 32 lanes. -/
theorem model_lane_count_unique (count : Nat)
    (tr : Trace (fun p => engine.kern.gemm.exact_dot_loop0_loop0.body sampleQ 0#usize p.1 p.2)
      innerCharge initialState (ok 64#i32) count) : count = 32 :=
  (LoopTrace.deterministic tr model_lane_trace).2

/-- Finite trace observation excludes loop divergence. -/
theorem model_excludes_divergence : (ok 64#i32 : Result I32) ≠ div :=
  LoopTrace.excludes_divergence model_lane_trace

end LaneTrace

/- Audit every owned declaration under the closed dependency cone. -/
open Lean Elab Command in
elab "#lane_trace_audit" : command => do
  let environment ← getEnv
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let constants := environment.constants.fold (fun acc constant _ =>
    if (`LaneTrace).isPrefixOf constant then constant :: acc else acc) ([] : List Lean.Name)
  if constants.length < 15 then throwError "LANE_TRACE_AUDIT_POPULATION"
  let mut failed := false
  for constant in constants do
    let dependencies ← liftTermElabM <| Lean.collectAxioms constant
    let bad := dependencies.filter (fun value => !allowed.contains value)
    unless bad.isEmpty do
      logError m!"{constant}: forbidden dependencies {bad.toList}"
      failed := true
    logInfo m!"{constant}: dependencies {dependencies.toList}"
  if failed then throwError "LANE_TRACE_CLOSED_CONE_REJECT"
  logInfo m!"LANE_TRACE_KERNEL_OK declarations={constants.length}"

#lane_trace_audit
