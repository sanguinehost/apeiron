import RowDotProposedSource
import RowDotResource
import CostBridge

/-!
SCOPE: proof/resource/RowDotAdmission.lean, new, plus the generated
proof/resource/RowDotProposedSource.lean it imports. The admission of one proposed source change to
`src/engine/matvec.rs`: replacing `match a.checked_mul(b) { … }` by `let prod = a * b;` in
`row_dot_fixed`'s loop, keeping the checked addition and every refusal. The module states the
proposed body, binds the fact that discharges its multiplication to the two extracted bodies,
proves the proposed body returns what the production body returns from every state and at every
refusal, and reads the two bodies' checked-operation counts off their own transitions into the
registered `CostAlgebra.Cost` model. Complete when the module kernel-checks with every headline
audited and both seeded defects reject.
REGISTER: scalar source-refinement resource model. No inference implementation is launched, no
canonical-ring correspondence, no claim about the dequantizers, the strides or the surrounding
projections. **This is a source-cost claim, not a hardware benefit claim.** The measured object is
the number of checked `i128` arithmetic operations the two extracted bodies execute, which is a
count over source text. No machine code, instruction count, latency, throughput, bandwidth or
energy follows from it, and none is asserted. `src/engine/matvec.rs` is unchanged: the proposed
body is not in production, and this module is the evidence that must be accepted before it becomes
the production source.
CORPUS: the production body `row_dot_source.matvec.row_dot_fixed_loop.body` and the proposed body
`row_dot_proposed.matvec.row_dot_fixed_loop.body`, both the Aeneas output of `src/engine/matvec.rs`
(`row_dot_fixed`, lines 86:0-117:1), the proposed one with the multiply change above and nothing
else. Native grade: `proved` for the preservation and cost theorems, checked by the gate's
`#resource_audit` under the closed cone {propext, Classical.choice, Quot.sound}.
REFERENCE: /home/socol/Workspace/apeiron/proof/resource/RowDotSource.lean (the production
extraction and `RowDotResource.i64_prod_range`, `RowDotResource.I128_checked_mul_some`,
`RowDotResource.RowTrace`, `RowDotResource.row_trace_projection`,
`RowDotResource.entry_visits_exact`, `RowDotResource.wit_visits`),
/home/socol/Workspace/apeiron/proof/resource/CostBridge.lean (the `Perf` record, `CostBridge.toCost`
with `ops := mulAdds` and `path := visits`, `CostBridge.DotFit`) and
/home/socol/Workspace/apeiron/proof/resource/CostAlgebra.lean (`Cost`, its componentwise `≤`,
`strictlyCheaper`, `dominates_improves`, `registeredCost`).
GATE: python3 proof/resource/RowDotAdmission_gate.py --check --controls, cwd repository root,
under `flock /tmp/opencode/apeiron-lane.lock` and `systemd-run --user --scope -p MemoryMax=12G
-p MemorySwapMax=0` with LEAN_NUM_THREADS=1, whole invocation 540 s including the extraction of
the proposed body, the fresh build and the two controls with their restorations. Observed red
before this file existed: `test -f proof/resource/RowDotAdmission.lean` exits 1.
GRADE: `proved` for the body agreement, the three refusals, the success value and the strict
`registeredCost` reduction, under the closed cone checked by `#resource_audit`. Grade `model`: the
two per-visit constants `srcOpsPerVisit`/`propOpsPerVisit`/`srcReadsPerVisit`, which the gate
cross-checks against the number of `I128.checked_mul`, `I128.checked_add` and indexed-read
applications in each extracted body's own text, and the critical-path convention `path := visits`
inherited from `CostBridge.toCost`. Unmeasured and unclaimed: machine code, instruction count,
latency, throughput, bandwidth, cache behaviour, compiler or codegen cost, and the cost of any
specific kernel. Not claimed: that the replacement is admitted into production.
-/

namespace RowDotAdmission

open Aeneas Aeneas.Std Result

universe u v

/-! ## (a) The two bodies -/

/-- The production loop body as a function of the loop state: the extracted
`row_dot_source.matvec.row_dot_fixed_loop.body`, which is `RowDotResource.dotBodyFn`. -/
def srcBodyFn (w_row x : Slice I64) : (I128 × Usize × Bool) →
    Result (ControlFlow (I128 × Usize × Bool) (I128 × Bool)) :=
  RowDotResource.dotBodyFn w_row x

/-- **The proposed replacement body.** This is the extracted
`row_dot_proposed.matvec.row_dot_fixed_loop.body`: Aeneas's own output for
`src/engine/matvec.rs` with the loop's `match a.checked_mul(b) { … }` replaced by
`let prod = a * b; match acc.checked_add(prod) { … }` and nothing else changed. It lives in
`proof/resource/RowDotProposedSource.lean`, which the gate regenerates from a scratch copy of the
source and compares byte for byte; the gate also refuses to run unless `src/engine/matvec.rs`
still contains the checked multiply, so the production body cannot drift under this admission. -/
def propBodyFn (w_row x : Slice I64) : (I128 × Usize × Bool) →
    Result (ControlFlow (I128 × Usize × Bool) (I128 × Bool)) :=
  fun s => row_dot_proposed.matvec.row_dot_fixed_loop.body w_row x s.1 s.2.1 s.2.2

theorem srcBodyFn_eq (w_row x : Slice I64) (s : I128 × Usize × Bool) :
    srcBodyFn w_row x s
      = row_dot_source.matvec.row_dot_fixed_loop.body w_row x s.1 s.2.1 s.2.2 := rfl

theorem propBodyFn_eq (w_row x : Slice I64) (s : I128 × Usize × Bool) :
    propBodyFn w_row x s
      = row_dot_proposed.matvec.row_dot_fixed_loop.body w_row x s.1 s.2.1 s.2.2 := rfl

/-! ## (b) The fact that discharges the multiplication -/

/-- Widening an extracted `i64` row read into `i128` is the identity on the value. -/
theorem widening_val (iw : I64) :
    (core.convert.num.FromI128I64.from iw).val = iw.val :=
  core.convert.num.FromI128I64.from_val_eq iw

/-- The `i64` range, stated for the `i64` values the two bodies read. -/
theorem i64_bounds (w : I64) :
    -((2 : ℤ) ^ (63 : ℕ)) ≤ w.val ∧ w.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
  have hb := w.hBounds
  rw [IScalarTy.I64_numBits_eq] at hb
  norm_num at hb
  exact ⟨by linarith [hb.1], by linarith [hb.2]⟩

/-- The `i128` window both bodies operate in, read off the model. -/
theorem i128_window :
    I128.min = -(2 : ℤ) ^ (127 : ℕ) ∧ I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
  constructor
  · norm_num [I128.min, IScalar.min, I128.numBits]
  · norm_num [I128.max, IScalar.max, I128.numBits]

/-- **The product of the two widened operands is inside the `i128` window.** Each operand is an
`i64`, so `|a| ≤ 2^63` and `|b| ≤ 2^63`, hence `|a·b| ≤ 2^126 < 2^127`: the product of two widened
`i64` reads never leaves `i128`. -/
theorem widened_product_window (iw ix : I64) :
    I128.min ≤ (core.convert.num.FromI128I64.from iw).val
        * (core.convert.num.FromI128I64.from ix).val ∧
      (core.convert.num.FromI128I64.from iw).val
        * (core.convert.num.FromI128I64.from ix).val ≤ I128.max := by
  have hbw := i64_bounds iw
  have hbx := i64_bounds ix
  have haw : (core.convert.num.FromI128I64.from iw).val = iw.val := widening_val iw
  have hbx' : (core.convert.num.FromI128I64.from ix).val = ix.val := widening_val ix
  obtain ⟨hmin, hmax⟩ := i128_window
  have hprange := RowDotResource.i64_prod_range iw.val ix.val hbw.1 hbw.2 hbx.1 hbx.2
  constructor
  · rw [hmin, haw, hbx']; exact hprange.1
  · rw [hmax, haw, hbx']; exact hprange.2

/-- **The production body's `I128.checked_mul` never returns `none` on the operands it
multiplies**, so the `none` arm of its extracted `match` is unreachable. The statement is over
`I64`, the type of the two row reads the body performs, and carries no premise: a premise here
would be the uninterpreted claim this module exists to discharge. -/
theorem widened_product_fits (iw ix : I64) :
    ∃ prod, I128.checked_mul (core.convert.num.FromI128I64.from iw)
        (core.convert.num.FromI128I64.from ix) = some prod ∧ prod.val = iw.val * ix.val := by
  obtain ⟨hlo, hhi⟩ := widened_product_window iw ix
  have haw : (core.convert.num.FromI128I64.from iw).val = iw.val := widening_val iw
  have hbx' : (core.convert.num.FromI128I64.from ix).val = ix.val := widening_val ix
  obtain ⟨prod, hprod, hprodv⟩ := RowDotResource.I128_checked_mul_some
    (core.convert.num.FromI128I64.from iw) (core.convert.num.FromI128I64.from ix) hlo hhi
  exact ⟨prod, hprod, by rw [hprodv, haw, hbx']⟩

/-- **The proposed body's plain `*` never fails on the operands it multiplies**, and yields
exactly the product the production body's checked multiply yields. `*` on `I128` is the extracted
`IScalar.mul`, a `Result` that `fail`s on overflow, and `widened_product_window` is what discharges
that `fail`. -/
theorem widened_plain_product_ok (iw ix : I64) :
    ∃ prod, (core.convert.num.FromI128I64.from iw) * (core.convert.num.FromI128I64.from ix)
        = ok prod ∧ prod.val = iw.val * ix.val := by
  obtain ⟨hlo, hhi⟩ := widened_product_window iw ix
  have haw : (core.convert.num.FromI128I64.from iw).val = iw.val := widening_val iw
  have hbx' : (core.convert.num.FromI128I64.from ix).val = ix.val := widening_val ix
  obtain ⟨prod, heq, hval⟩ := WP.spec_imp_exists (I128.mul_bv_spec hlo hhi)
  exact ⟨prod, heq, by rw [hval.1, haw, hbx']⟩

/-- The product the two widened operands denote, in `ℤ`: the widening does not change it. -/
theorem widened_product_val (iw ix : I64) :
    (core.convert.num.FromI128I64.from iw).val * (core.convert.num.FromI128I64.from ix).val
      = iw.val * ix.val := by
  have haw : (core.convert.num.FromI128I64.from iw).val = iw.val := widening_val iw
  have hbx' : (core.convert.num.FromI128I64.from ix).val = ix.val := widening_val ix
  rw [haw, hbx']

/-- **The plain multiply is `ok` of the product the checked multiply produces.** This is the
equation that replaces the `match` on `I128.checked_mul` in the proposed body. -/
theorem widened_plain_product_is_checked (iw ix : I64) (prod : I128)
    (hmul : I128.checked_mul (core.convert.num.FromI128I64.from iw)
        (core.convert.num.FromI128I64.from ix) = some prod) :
    (core.convert.num.FromI128I64.from iw) * (core.convert.num.FromI128I64.from ix) = ok prod := by
  obtain ⟨hlo, hhi⟩ := widened_product_window iw ix
  obtain ⟨hw, hmulw, hmulv⟩ := RowDotResource.I128_checked_mul_some
    (core.convert.num.FromI128I64.from iw) (core.convert.num.FromI128I64.from ix) hlo hhi
  obtain ⟨prod', hplain, hplainv⟩ := widened_plain_product_ok iw ix
  have hhw : hw = prod := Option.some.inj (by rw [← hmul, hmulw])
  have hmulval : hw.val = iw.val * ix.val := by rw [hmulv, widened_product_val iw ix]
  have hval : prod'.val = prod.val := by rw [hplainv, ← hmulval, hhw]
  have hprod' : prod' = prod := IScalar.eq_of_val_eq hval
  rw [hplain, hprod']

/-! ## (c) The transitions of the two bodies -/

/-- **The guarded transition, with both bodies' own equations.** On equal-length rows, from an
index inside the weight row with the failure flag clear, both bodies read `w_row[i]` and `x[i]`,
widen both to `i128`, take the `some` arm of `I128.checked_mul` on the two widened reads
(`widened_product_fits`), add the product into the accumulator with one `I128.checked_add`, and
advance the index by exactly one. The proposed body reaches the same accumulator, index and flag,
because its single multiplication is `ok` of the same product (`widened_plain_product_is_checked`)
and the addition is the same `I128.checked_add`. -/
theorem visit_agree (w_row x : Slice I64) (n : Usize) (acc : I128) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = w_row.val.length)
    (hxlen : x.val.length = w_row.val.length) :
    ∃ iw ix acc' inext acc_ok' prod,
      iw.val = w_row.val[i.val].val ∧ ix.val = x.val[i.val].val ∧
      srcBodyFn w_row x (acc, i, true) = ok (ControlFlow.cont (acc', inext, acc_ok')) ∧
      propBodyFn w_row x (acc, i, true) = ok (ControlFlow.cont (acc', inext, acc_ok')) ∧
      I128.checked_mul (core.convert.num.FromI128I64.from iw)
          (core.convert.num.FromI128I64.from ix) = some prod ∧
      prod.val = iw.val * ix.val ∧
      inext.val = i.val + 1 ∧
      ((∃ total, I128.checked_add acc prod = some total ∧ acc' = total ∧ acc_ok' = true) ∨
        (acc' = acc ∧ acc_ok' = false)) := by
  have hbnd : i.val < w_row.val.length := by omega
  have hbndx : i.val < x.val.length := by omega
  obtain ⟨iw, hiw, hiwv⟩ := WP.spec_imp_exists (Slice.index_usize_spec w_row i hbnd)
  obtain ⟨ix, hix, hixv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hbndx)
  have hiwv' : iw.val = w_row.val[i.val].val := by rw [hiwv]
  have hixv' : ix.val = x.val[i.val].val := by rw [hixv]
  obtain ⟨prod, hmul, hmulv⟩ := widened_product_fits iw ix
  have hplain : (core.convert.num.FromI128I64.from iw) * (core.convert.num.FromI128I64.from ix)
      = ok prod := widened_plain_product_is_checked iw ix prod hmul
  have h1 : (1#usize).val = 1 := by norm_num
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have hb := w_row.property
    rw [h1]; omega
  obtain ⟨inext, hinext, hinextv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  have hilt' : i < Slice.len w_row :=
    (UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [RowDotResource.sliceLenVal]; exact hbnd)
  cases hadd : acc.checked_add prod with
  | none =>
      refine ⟨iw, ix, acc, inext, false, prod, hiwv', hixv', ?_, ?_, hmul, hmulv, hinextv, ?_⟩
      · show row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true = _
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
      · show row_dot_proposed.matvec.row_dot_fixed_loop.body w_row x acc i true = _
        rw [row_dot_proposed.matvec.row_dot_fixed_loop.body, hiw, hix]
        simp only [lift, bind_tc_ok]
        rw [hplain]
        simp only [bind_tc_ok]
        rw [hadd]
        dsimp only
        rw [hinext]
        simp only [bind_tc_ok]
        rw [if_pos hilt', if_pos trivial]
        rfl
      · exact Or.inr ⟨rfl, rfl⟩
  | some total =>
      refine ⟨iw, ix, total, inext, true, prod, hiwv', hixv', ?_, ?_, hmul, hmulv, hinextv, ?_⟩
      · show row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i true = _
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
      · show row_dot_proposed.matvec.row_dot_fixed_loop.body w_row x acc i true = _
        rw [row_dot_proposed.matvec.row_dot_fixed_loop.body, hiw, hix]
        simp only [lift, bind_tc_ok]
        rw [hplain]
        simp only [bind_tc_ok]
        rw [hadd]
        dsimp only
        rw [hinext]
        simp only [bind_tc_ok]
        rw [if_pos hilt', if_pos trivial]
        rfl
      · exact Or.inl ⟨total, hadd, rfl, rfl⟩

/-- **The index exit of both bodies.** At or past the row width both bodies return the accumulator
and the flag unchanged, performing no multiplication and no addition. -/
theorem visit_exit_len_agree (w_row x : Slice I64) (acc : I128) (i : Usize) (acc_ok : Bool)
    (hn : ¬ (i < Slice.len w_row)) :
    srcBodyFn w_row x (acc, i, acc_ok) = ok (ControlFlow.done (acc, acc_ok)) ∧
      propBodyFn w_row x (acc, i, acc_ok) = ok (ControlFlow.done (acc, acc_ok)) := by
  refine ⟨by
    show row_dot_source.matvec.row_dot_fixed_loop.body w_row x acc i acc_ok = _
    exact RowDotResource.body_exit_len w_row x acc i acc_ok hn, ?_⟩
  show row_dot_proposed.matvec.row_dot_fixed_loop.body w_row x acc i acc_ok = _
  rw [row_dot_proposed.matvec.row_dot_fixed_loop.body, if_neg hn]

/-- **The flag exit of both bodies.** Inside the row with the failure flag set, both bodies return
the accumulator with the flag clear, performing no multiplication and no addition. -/
theorem visit_exit_flag_agree (w_row x : Slice I64) (acc : I128) (i : Usize)
    (hilt : i.val < (Slice.len w_row).val) :
    srcBodyFn w_row x (acc, i, false) = ok (ControlFlow.done (acc, false)) ∧
      propBodyFn w_row x (acc, i, false) = ok (ControlFlow.done (acc, false)) := by
  refine ⟨RowDotResource.body_exit_flag w_row x acc i hilt, ?_⟩
  have hilt' : i < Slice.len w_row :=
    (UScalar.lt_equiv i (Slice.len w_row)).2
      (by rw [RowDotResource.sliceLenVal]; exact hilt)
  show row_dot_proposed.matvec.row_dot_fixed_loop.body w_row x acc i false = _
  rw [row_dot_proposed.matvec.row_dot_fixed_loop.body, if_pos hilt', if_neg (by simp)]

/-- **The two bodies return the same outcome from every state.** On equal-length rows, from every
loop state and every flag, the production body and the proposed body return the identical `ok`
value: the same continuation state, or the same exit value. Unequal-length rows never reach a
body, because the entry point refuses before the loop (`preserves_row_length`). -/
theorem body_agree (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    (acc : I128) (i : Usize) (acc_ok : Bool) :
    ∃ r, srcBodyFn w_row x (acc, i, acc_ok) = ok r ∧
         propBodyFn w_row x (acc, i, acc_ok) = ok r := by
  by_cases hlt : i < Slice.len w_row
  · cases acc_ok with
    | false =>
        exact ⟨ControlFlow.done (acc, false), visit_exit_flag_agree w_row x acc i
          ((UScalar.lt_equiv i (Slice.len w_row)).1 hlt)⟩
    | true =>
        obtain ⟨iw, ix, acc', inext, acc_ok', prod, hiw, hix, hsrc, hprop, hmul, hmulv, hinext,
          hadd'⟩ :=
          visit_agree w_row x (Slice.len w_row) acc i
            ((UScalar.lt_equiv i (Slice.len w_row)).1 hlt)
            (by rw [RowDotResource.sliceLenVal]) hxlen
        exact ⟨ControlFlow.cont (acc', inext, acc_ok'), hsrc, hprop⟩
  · exact ⟨ControlFlow.done (acc, acc_ok), visit_exit_len_agree w_row x acc i acc_ok hlt⟩

/-! ## (d) Agreement of the two loops -/

/-- **The two loops return the same outcome from every state.** A strong induction on the number
of row positions the state has left: both bodies take the same transition, and the index advances
by one in both, so the induction runs. -/
theorem loop_agree_aux (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length) :
    ∀ (k : Nat) (acc : I128) (i : Usize) (acc_ok : Bool),
      (Slice.len w_row).val - i.val = k →
      ∃ (b : I128 × Bool) (visits steps : Nat),
        (∃ _c : RowDotResource.RowTrace (RowDotResource.dotBodyFn w_row x) (acc, i, acc_ok)
            (ok b) visits steps, True) ∧
        (∃ _c : RowDotResource.RowTrace (propBodyFn w_row x) (acc, i, acc_ok)
            (ok b) visits steps, True) := by
  intro k
  induction k using Nat.strong_induction_on with
  | h k ih =>
      intro acc i acc_ok hrank
      by_cases hlt : i.val < (Slice.len w_row).val
      · cases acc_ok with
        | false =>
            have hex := visit_exit_flag_agree w_row x acc i hlt
            exact ⟨(acc, false), 1, 0, ⟨RowDotResource.RowTrace.done hex.1, trivial⟩,
              ⟨RowDotResource.RowTrace.done hex.2, trivial⟩⟩
        | true =>
            obtain ⟨iw, ix, acc', inext, acc_ok', prod, hiw, hix, hsrc, hprop, hmul, hmulv,
              hinext, hadd'⟩ :=
              visit_agree w_row x (Slice.len w_row) acc i hlt
                (by rw [RowDotResource.sliceLenVal]) hxlen
            have hrank' : (Slice.len w_row).val - inext.val = k - 1 := by rw [hinext]; omega
            obtain ⟨res, visits, steps, ⟨hsrcTail, _⟩, ⟨hpropTail, _⟩⟩ :=
              ih (k - 1) (by omega) acc' inext acc_ok' hrank'
            exact ⟨res, visits + 1, steps + 1,
              ⟨RowDotResource.RowTrace.cont hsrc hsrcTail, trivial⟩,
              ⟨RowDotResource.RowTrace.cont hprop hpropTail, trivial⟩⟩
      · have hnlt : ¬ (i < Slice.len w_row) := by
          intro hc
          exact hlt ((UScalar.lt_equiv i (Slice.len w_row)).1 hc)
        have hex := visit_exit_len_agree w_row x acc i acc_ok hnlt
        exact ⟨(acc, acc_ok), 1, 0, ⟨RowDotResource.RowTrace.done hex.1, trivial⟩,
          ⟨RowDotResource.RowTrace.done hex.2, trivial⟩⟩

/-- **The two loops return the same outcome.** -/
theorem loop_agree (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    (acc : I128) (i : Usize) (acc_ok : Bool) :
    ∃ b, row_dot_source.matvec.row_dot_fixed_loop w_row x acc i acc_ok = ok b ∧
          row_dot_proposed.matvec.row_dot_fixed_loop w_row x acc i acc_ok = ok b :=
  by
  obtain ⟨b, visits, steps, ⟨hsrc, _⟩, ⟨hprop, _⟩⟩ :=
    loop_agree_aux w_row x hxlen ((Slice.len w_row).val - i.val) acc i acc_ok rfl
  exact ⟨b, RowDotResource.row_trace_projection hsrc,
    RowDotResource.row_trace_projection hprop⟩

/-! ## (e) Agreement of the two entry points -/

/-- **A `do` block that returns `ok` read its first `←` as `ok`.** Any other outcome of the bound
computation leaves the block unable to return `ok`, which is what excludes the `vis` and `div`
outcomes of the two `IScalar` effects the epilogue performs. -/
theorem bind_ok_reads {α : Type u} {x : Result α} {β : Type u} {f : α → Result β} {y : β}
    (h : (do let a ← x; f a) = ok y) : ∃ a, x = ok a ∧ f a = ok y := by
  cases x with
  | ret a => exact ⟨a, rfl, by simpa only [bind_tc_ok] using h⟩
  | vis e k => have h' := h; rw [bind_tc_vis] at h'; exact absurd h' vis_not_ok
  | div => have h' := h; rw [bind_tc_div] at h'; exact absurd h' div_not_ok

/-- The `round_shift_32` epilogue helper agrees: both extractions emit the same term for it. -/
theorem round_shift_agree (v w : I128) (h : row_dot_source.value.round_shift_32 v = ok w) :
    row_dot_proposed.value.round_shift_32 v = ok w := by
  show row_dot_source.value.round_shift_32 v = ok w
  exact h

/-- The proposed entry point after the length test and a successful accumulator loop, as a `do`
block over the accumulator value. -/
theorem prop_fixed_unfold (w_row x : Slice I64) (S : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_proposed.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true =
      ok (S, true)) :
    row_dot_proposed.matvec.row_dot_fixed w_row x =
      (do
        let rounded ← row_dot_proposed.value.round_shift_32 S
        let i2 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if rounded > i2
        then ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.OutputOverflow)
        else
          let i3 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if rounded < i3
          then ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.OutputOverflow)
          else
            let i4 ← lift (IScalar.cast .I64 rounded)
            ok (core.result.Result.Ok i4)) := by
  have hlen' : Slice.len w_row = Slice.len x := by
    apply UScalar.eq_of_val_eq
    rw [RowDotResource.sliceLenVal w_row, RowDotResource.sliceLenVal x]
    exact hlen
  have hcond : (Slice.len w_row != Slice.len x) = false := by simp [hlen']
  rw [row_dot_proposed.matvec.row_dot_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **The length refusal is preserved.** Unequal row lengths return `Err (RowLength 0 len len)` from
both entry points, with the same three fields, before any product is read. -/
theorem preserves_row_length (w_row x : Slice I64)
    (h : w_row.val.length ≠ x.val.length) :
    row_dot_proposed.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err
        (row_dot_proposed.matvec.MatvecError.RowLength 0#usize (Slice.len w_row) (Slice.len x))) := by
  have hlen' : Slice.len w_row ≠ Slice.len x := by
    intro hc
    have hv : (Slice.len w_row).val = (Slice.len x).val :=
      congrArg (fun u : Usize => u.val) hc
    rw [RowDotResource.sliceLenVal w_row, RowDotResource.sliceLenVal x] at hv
    exact h hv
  have hcond : (Slice.len w_row != Slice.len x) = true := by simpa using hlen'
  rw [row_dot_proposed.matvec.row_dot_fixed]
  dsimp only
  rw [if_pos hcond]

/-- **The accumulator refusal is preserved.** A loop that stops with the failure flag set returns
`Err AccumulatorOverflow` from both entry points. The dropped check contributed no refusal here:
the flag is set by the checked addition, which both bodies keep. -/
theorem preserves_acc_overflow (w_row x : Slice I64) (S : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_proposed.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true =
      ok (S, false)) :
    row_dot_proposed.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.AccumulatorOverflow) := by
  have hlen' : Slice.len w_row = Slice.len x := by
    apply UScalar.eq_of_val_eq
    rw [RowDotResource.sliceLenVal w_row, RowDotResource.sliceLenVal x]
    exact hlen
  have hcond : (Slice.len w_row != Slice.len x) = false := by simp [hlen']
  rw [row_dot_proposed.matvec.row_dot_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [hloop]
  simp

/-- **The upper narrowing refusal is preserved.** A rounded value above `i64::MAX` returns
`Err OutputOverflow` from both entry points. -/
theorem preserves_out_high (w_row x : Slice I64) (S rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_proposed.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hround : row_dot_proposed.value.round_shift_32 S = ok rounded)
    (hhigh : rounded > IScalar.cast .I128 core.num.I64.MAX) :
    row_dot_proposed.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.OutputOverflow) := by
  rw [prop_fixed_unfold w_row x S hlen hloop]
  rw [hround]
  simp only [lift, bind_tc_ok]
  rw [if_pos hhigh]

/-- **The lower narrowing refusal is preserved.** A rounded value below `i64::MIN`, with the upper
test already passed, returns `Err OutputOverflow` from both entry points. -/
theorem preserves_out_low (w_row x : Slice I64) (S rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : row_dot_proposed.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hround : row_dot_proposed.value.round_shift_32 S = ok rounded)
    (hhigh : ¬ (rounded > IScalar.cast .I128 core.num.I64.MAX))
    (hlow : rounded < IScalar.cast .I128 core.num.I64.MIN) :
    row_dot_proposed.matvec.row_dot_fixed w_row x =
      ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.OutputOverflow) := by
  rw [prop_fixed_unfold w_row x S hlen hloop]
  rw [hround]
  simp only [lift, bind_tc_ok]
  rw [if_neg hhigh, if_pos hlow]

/-- **Both entry points return the same result of the loop with the flag clear.** With the same
accumulator, the same `rounded` value from the two extractions of the epilogue helper, and the two
narrowing tests stated once, the two entry points agree: either both return `Ok` of the same
narrowed `i64`, or both return `Err OutputOverflow`. -/
theorem preserves_ok (w_row x : Slice I64) (S rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloopSrc : row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hloopProp : row_dot_proposed.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true =
      ok (S, true))
    (hround : row_dot_source.value.round_shift_32 S = ok rounded)
    (hroundProp : row_dot_proposed.value.round_shift_32 S = ok rounded) :
    (row_dot_source.matvec.row_dot_fixed w_row x =
        ok (core.result.Result.Ok (IScalar.cast .I64 rounded)) ∧
      row_dot_proposed.matvec.row_dot_fixed w_row x =
        ok (core.result.Result.Ok (IScalar.cast .I64 rounded))) ∨
    (row_dot_source.matvec.row_dot_fixed w_row x =
        ok (core.result.Result.Err row_dot_source.matvec.MatvecError.OutputOverflow) ∧
      row_dot_proposed.matvec.row_dot_fixed w_row x =
        ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.OutputOverflow)) := by
  by_cases hhigh : rounded > IScalar.cast .I128 core.num.I64.MAX
  · right
    refine ⟨?_, ?_⟩
    · rw [RowDotResource.row_dot_fixed_unfold w_row x S hlen hloopSrc, hround]
      simp only [lift, bind_tc_ok]
      rw [if_pos hhigh]
    · rw [prop_fixed_unfold w_row x S hlen hloopProp, hroundProp]
      simp only [lift, bind_tc_ok]
      rw [if_pos hhigh]
  · by_cases hlow : rounded < IScalar.cast .I128 core.num.I64.MIN
    · right
      refine ⟨?_, ?_⟩
      · rw [RowDotResource.row_dot_fixed_unfold w_row x S hlen hloopSrc, hround]
        simp only [lift, bind_tc_ok]
        rw [if_neg hhigh, if_pos hlow]
      · rw [prop_fixed_unfold w_row x S hlen hloopProp, hroundProp]
        simp only [lift, bind_tc_ok]
        rw [if_neg hhigh, if_pos hlow]
    · left
      refine ⟨?_, ?_⟩
      · rw [RowDotResource.row_dot_fixed_unfold w_row x S hlen hloopSrc, hround]
        simp only [lift, bind_tc_ok]
        rw [if_neg hhigh, if_neg hlow]
      · rw [prop_fixed_unfold w_row x S hlen hloopProp, hroundProp]
        simp only [lift, bind_tc_ok]
        rw [if_neg hhigh, if_neg hlow]

/-- **A success of one is the same success of the other.** Whenever the production entry point
returns `Ok v`, the proposed entry point returns `Ok v` for the very same `v`: the loop agreed, the
epilogue rounded the same accumulator, and the narrowing tests decided the same way. -/
theorem preserves_ok_of (w_row x : Slice I64) (v : I64)
    (hsrc : row_dot_source.matvec.row_dot_fixed w_row x = ok (core.result.Result.Ok v)) :
    row_dot_proposed.matvec.row_dot_fixed w_row x = ok (core.result.Result.Ok v) := by
  have hlen : w_row.val.length = x.val.length := by
    by_contra hne
    have h := RowDotResource.row_dot_row_length_refuses w_row x hne
    rw [hsrc] at h
    injection Result.ok_injective h
  obtain ⟨b, hloopSrc, hloopProp⟩ := loop_agree w_row x hlen.symm 0#i128 0#usize true
  cases b with
  | mk S flag =>
      cases flag with
      | false =>
          have h := RowDotResource.row_dot_acc_refuses w_row x S hlen hloopSrc
          rw [hsrc] at h
          injection Result.ok_injective h
      | true =>
          have hbind : (do
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
                  ok (core.result.Result.Ok i4)) = ok (core.result.Result.Ok v) := by
            rw [RowDotResource.row_dot_fixed_unfold w_row x S hlen hloopSrc] at hsrc
            exact hsrc
          obtain ⟨rounded, hroundSrc, _⟩ := bind_ok_reads hbind
          obtain ⟨hokSrc, hokProp⟩ | ⟨herrSrc, herrProp⟩ :=
            preserves_ok w_row x S rounded hlen hloopSrc hloopProp hroundSrc
              (round_shift_agree S rounded hroundSrc)
          · have he : IScalar.cast .I64 rounded = v := by
              injection Result.ok_injective (hokSrc.symm.trans hsrc)
            rw [he] at hokProp
            exact hokProp
          · rw [hsrc] at herrSrc
            injection Result.ok_injective herrSrc

/-- **Before = after.** For every pair of `i64` slices:
- a success of the production entry point is the same success of the proposed one, the same value;
- unequal lengths are refused by both, with the same three fields;
- on equal lengths, a loop that stops flagged is refused by both with `Err AccumulatorOverflow`, and
  the proposed loop stops on the same value;
- on equal lengths with the flag clear, the two entry points return the same result of the same
  accumulator and the same rounding: either the same `Ok` value, or `Err OutputOverflow` from both.
Every refusal of the production entry point is one of these, so no result of one is unaccounted for
by the other. -/
theorem preserves (w_row x : Slice I64) :
    (∀ v : I64, row_dot_source.matvec.row_dot_fixed w_row x = ok (core.result.Result.Ok v) →
        row_dot_proposed.matvec.row_dot_fixed w_row x = ok (core.result.Result.Ok v)) ∧
    (∀ (_hne : w_row.val.length ≠ x.val.length),
      row_dot_proposed.matvec.row_dot_fixed w_row x =
        ok (core.result.Result.Err
          (row_dot_proposed.matvec.MatvecError.RowLength 0#usize (Slice.len w_row) (Slice.len x))) ∧
      row_dot_source.matvec.row_dot_fixed w_row x =
        ok (core.result.Result.Err
          (row_dot_source.matvec.MatvecError.RowLength 0#usize (Slice.len w_row) (Slice.len x)))) ∧
    (∀ (S : I128) (_hlen : w_row.val.length = x.val.length),
      row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, false) →
      row_dot_proposed.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, false) ∧
        row_dot_source.matvec.row_dot_fixed w_row x =
          ok (core.result.Result.Err row_dot_source.matvec.MatvecError.AccumulatorOverflow) ∧
        row_dot_proposed.matvec.row_dot_fixed w_row x =
          ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.AccumulatorOverflow)) ∧
    (∀ (S rounded : I128) (_hlen : w_row.val.length = x.val.length),
      row_dot_source.matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true) →
      row_dot_source.value.round_shift_32 S = ok rounded →
      (row_dot_source.matvec.row_dot_fixed w_row x =
          ok (core.result.Result.Ok (IScalar.cast .I64 rounded)) ∧
        row_dot_proposed.matvec.row_dot_fixed w_row x =
          ok (core.result.Result.Ok (IScalar.cast .I64 rounded))) ∨
        (row_dot_source.matvec.row_dot_fixed w_row x =
            ok (core.result.Result.Err row_dot_source.matvec.MatvecError.OutputOverflow) ∧
          row_dot_proposed.matvec.row_dot_fixed w_row x =
            ok (core.result.Result.Err row_dot_proposed.matvec.MatvecError.OutputOverflow))) := by
  refine ⟨preserves_ok_of w_row x, ?_, ?_, ?_⟩
  · intro hne
    exact ⟨preserves_row_length w_row x hne,
      RowDotResource.row_dot_row_length_refuses w_row x hne⟩
  · intro S hlen hloop
    obtain ⟨b, hloopSrc, hloopProp⟩ := loop_agree w_row x hlen.symm 0#i128 0#usize true
    have hEq : b = (S, false) := Result.ok_injective (hloopSrc.symm.trans hloop)
    rw [hEq] at hloopProp
    exact ⟨hloopProp, RowDotResource.row_dot_acc_refuses w_row x S hlen hloop,
      preserves_acc_overflow w_row x S hlen hloopProp⟩
  · intro S rounded hlen hloop hround
    obtain ⟨b, hloopSrc, hloopProp⟩ := loop_agree w_row x hlen.symm 0#i128 0#usize true
    have hEq : b = (S, true) := Result.ok_injective (hloopSrc.symm.trans hloop)
    rw [hEq] at hloopProp
    exact preserves_ok w_row x S rounded hlen hloop hloopProp hround
      (round_shift_agree S rounded hround)

/-- **The proposed body's transition is the production body's transition.** From a state the
production body continues from, the proposed body continues to the identical state. -/
theorem prop_of_src_cont (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    {s next : I128 × Usize × Bool} (h : srcBodyFn w_row x s = ok (ControlFlow.cont next)) :
    propBodyFn w_row x s = ok (ControlFlow.cont next) := by
  obtain ⟨r, hsrc, hprop⟩ := body_agree w_row x hxlen s.1 s.2.1 s.2.2
  rw [h] at hsrc
  have hr : r = ControlFlow.cont next := Result.ok_injective hsrc.symm
  rw [hr] at hprop
  exact hprop

/-- **The proposed body's exit is the production body's exit.** From a state the production body
exits from, the proposed body exits with the same value. -/
theorem prop_of_src_done (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    {s : I128 × Usize × Bool} {value : I128 × Bool}
    (h : srcBodyFn w_row x s = ok (ControlFlow.done value)) :
    propBodyFn w_row x s = ok (ControlFlow.done value) := by
  obtain ⟨r, hsrc, hprop⟩ := body_agree w_row x hxlen s.1 s.2.1 s.2.2
  rw [h] at hsrc
  have hr : r = ControlFlow.done value := Result.ok_injective hsrc.symm
  rw [hr] at hprop
  exact hprop

/-- **Neither body fails.** On equal-length rows the production body never returns `fail` at any
state, and neither does the proposed body. -/
theorem src_never_fails (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    {s : I128 × Usize × Bool} {e : Error} (h : srcBodyFn w_row x s = fail e) : False := by
  obtain ⟨r, hsrc, _⟩ := body_agree w_row x hxlen s.1 s.2.1 s.2.2
  rw [h] at hsrc
  exact absurd hsrc fail_not_ok

/-- **The dropped check never fired at any state.** Over every loop state of equal-length rows whose
index is inside the weight row, if the production body continues then it took the `some` arm of its
`I128.checked_mul`, on the two widened row reads `visit_agree` names at that index. The `none` arm
of the extracted `match` has no reachable state. -/
theorem mul_arm_always_some (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    (acc : I128) (i : Usize) (s' : I128 × Usize × Bool) (hbnd : i.val < w_row.val.length)
    (hcont : srcBodyFn w_row x (acc, i, true) = ok (ControlFlow.cont s')) :
    ∃ iw ix prod, I128.checked_mul (core.convert.num.FromI128I64.from iw)
        (core.convert.num.FromI128I64.from ix) = some prod := by
  have hlt : i < Slice.len w_row :=
    (UScalar.lt_equiv i (Slice.len w_row)).mpr hbnd
  obtain ⟨iw, ix, acc', inext, acc_ok', prod, hiw, hix, hsrc, hprop, hmul, hmulv, hinext, hadd'⟩ :=
    visit_agree w_row x (Slice.len w_row) acc i hlt (by rw [RowDotResource.sliceLenVal]) hxlen
  have hne : s' = (acc', inext, acc_ok') := by
    injection Result.ok_injective (hcont.symm.trans hsrc)
  cases hne
  exact ⟨iw, ix, prod, hmul⟩

/-! ## (f) The two bodies' checked-operation counts, read off their own equations -/

/-- The checked `i128` arithmetic operations one visit of the **production** body performs, read off
its own equation: a visit that continues reached the body's `I128.checked_mul` and
`I128.checked_add` matches, so it performed both of them; a visit that exits reached neither. The
gate counts exactly those two applications in the extracted body text. -/
def srcOps (w_row x : Slice I64) (s : I128 × Usize × Bool) : Nat :=
  match (srcBodyFn w_row x s).match with
  | MatchResult.ok (ControlFlow.cont _) => 2
  | _ => 0

/-- The element reads one visit of the production body performs, read off its own equation: the two
row elements at the index, the two `Slice.index_usize` applications `visit_agree` names. -/
def srcReads (w_row x : Slice I64) (s : I128 × Usize × Bool) : Nat :=
  match (srcBodyFn w_row x s).match with
  | MatchResult.ok (ControlFlow.cont _) => 2
  | _ => 0

/-- The checked `i128` arithmetic operations one visit of the **proposed** body performs, read off
its own equation: it reached one `I128.checked_add` and no `I128.checked_mul`, its single
multiplication being the unchecked `*`. This is the count the replacement removes. -/
def propOps (w_row x : Slice I64) (s : I128 × Usize × Bool) : Nat :=
  match (propBodyFn w_row x s).match with
  | MatchResult.ok (ControlFlow.cont _) => 1
  | _ => 0

/-- The element reads one visit of the proposed body performs: the same two row elements, the
replacement reading nothing extra and nothing less. -/
def propReads (w_row x : Slice I64) (s : I128 × Usize × Bool) : Nat :=
  match (propBodyFn w_row x s).match with
  | MatchResult.ok (ControlFlow.cont _) => 2
  | _ => 0

/-- The production body's per-visit checked-operation count, read off its own equation. -/
theorem srcOps_of_cont (w_row x : Slice I64) (s : I128 × Usize × Bool)
    {next : I128 × Usize × Bool} (h : srcBodyFn w_row x s = ok (ControlFlow.cont next)) :
    srcOps w_row x s = 2 := by
  simp [srcOps, h]

/-- The production body's per-visit read count, read off its own equation. -/
theorem srcReads_of_cont (w_row x : Slice I64) (s : I128 × Usize × Bool)
    {next : I128 × Usize × Bool} (h : srcBodyFn w_row x s = ok (ControlFlow.cont next)) :
    srcReads w_row x s = 2 := by
  simp [srcReads, h]

/-- The proposed body's per-visit checked-operation count, read off its own equation. -/
theorem propOps_of_cont (w_row x : Slice I64) (s : I128 × Usize × Bool)
    {next : I128 × Usize × Bool} (h : propBodyFn w_row x s = ok (ControlFlow.cont next)) :
    propOps w_row x s = 1 := by
  simp [propOps, h]

/-- The proposed body's per-visit read count, read off its own equation. -/
theorem propReads_of_cont (w_row x : Slice I64) (s : I128 × Usize × Bool)
    {next : I128 × Usize × Bool} (h : propBodyFn w_row x s = ok (ControlFlow.cont next)) :
    propReads w_row x s = 2 := by
  simp [propReads, h]

/-- **A certificate of a run of a body.** `visits` counts invocations of the body, `ops` counts the
checked `i128` operations those invocations performed and `reads` the element reads they performed,
each read off the visited state's own body equation by `ops` and `reads`. Divergence has no
constructor. -/
inductive Cert {State : Type u} {Value : Type v}
    (body : State → Result (ControlFlow State Value)) (ops reads : State → Nat) :
    State → Result Value → Nat → Nat → Nat → Prop where
  | done {state value} (observed : body state = ok (ControlFlow.done value)) :
      Cert body ops reads state (ok value) 1 0 0
  | failure {state error} (observed : body state = fail error) :
      Cert body ops reads state (fail error) 1 0 0
  | cont {state next result visits o r} (observed : body state = ok (ControlFlow.cont next))
      (tail : Cert body ops reads next result visits o r) :
      Cert body ops reads state result (visits + 1) (o + ops state) (r + reads state)

/-- **The production body's counts.** Every continuing visit performed the two checked operations
and the two reads `visit_agree` names, and no exit performed any, so a certificate with `v` visits
records `2 * (v - 1)` checked operations and `2 * (v - 1)` reads. -/
theorem src_counts {w_row x : Slice I64} {s : I128 × Usize × Bool} {res : Result (I128 × Bool)}
    {v o r : Nat}
    (c : Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x) s res v o r) :
    1 ≤ v ∧ o = 2 * (v - 1) ∧ r = 2 * (v - 1) := by
  induction c with
  | done obs => exact ⟨Nat.le_refl _, rfl, rfl⟩
  | failure obs => exact ⟨Nat.le_refl _, rfl, rfl⟩
  | cont obs tail ih =>
      have hops : srcOps w_row x _ = 2 := srcOps_of_cont w_row x _ obs
      have hreads : srcReads w_row x _ = 2 := srcReads_of_cont w_row x _ obs
      obtain ⟨hv, iho, ihr⟩ := ih
      refine ⟨?_, ?_, ?_⟩
      · omega
      · rw [hops, iho]; omega
      · rw [hreads, ihr]; omega

/-- **The proposed body's counts.** Every continuing visit performed the one checked operation it
reached, so a certificate with `v` visits records `v - 1` checked operations, and the same
`2 * (v - 1)` reads. -/
theorem prop_counts {w_row x : Slice I64} {s : I128 × Usize × Bool} {res : Result (I128 × Bool)}
    {v o r : Nat}
    (c : Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x) s res v o r) :
    1 ≤ v ∧ o = v - 1 ∧ r = 2 * (v - 1) := by
  induction c with
  | done obs => exact ⟨Nat.le_refl _, rfl, rfl⟩
  | failure obs => exact ⟨Nat.le_refl _, rfl, rfl⟩
  | cont obs tail ih =>
      have hops : propOps w_row x _ = 1 := propOps_of_cont w_row x _ obs
      have hreads : propReads w_row x _ = 2 := propReads_of_cont w_row x _ obs
      obtain ⟨hv, iho, ihr⟩ := ih
      refine ⟨?_, ?_, ?_⟩
      · omega
      · rw [hops, iho]
        omega
      · rw [hreads, ihr]
        omega

/-- **The production run's certificate.** A certificate of the actual body in `RowDotResource`'s
visit/step counts is a certificate of the checked operations and the element reads it performed:
`st` multiply-adds, each one a checked multiply and a checked addition, and `2 * st` reads. -/
theorem src_cert_of_trace (w_row x : Slice I64) {s : I128 × Usize × Bool}
    {res : Result (I128 × Bool)} {v st : Nat}
    (t : RowDotResource.RowTrace (srcBodyFn w_row x) s res v st) :
    Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x) s res v (2 * st) (2 * st) := by
  induction t with
  | done obs => exact Cert.done obs
  | failure obs => exact Cert.failure obs
  | cont obs tail ih =>
      have hops : srcOps w_row x _ = 2 := srcOps_of_cont w_row x _ obs
      have hreads : srcReads w_row x _ = 2 := srcReads_of_cont w_row x _ obs
      have h := Cert.cont obs ih
      rw [hops, hreads] at h
      simpa only [Nat.mul_succ] using h

/-- **The proposed run's certificate.** On equal-length rows the two bodies take the same
transition at every state, so a certificate of the production run is also a certificate of the
proposed run, with the same visit count, the same outcome and the same read count. -/
theorem prop_cert_of_trace (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    {s : I128 × Usize × Bool} {res : Result (I128 × Bool)} {v st : Nat}
    (t : RowDotResource.RowTrace (srcBodyFn w_row x) s res v st) :
    Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x) s res v st (2 * st) := by
  induction t with
  | done obs => exact Cert.done (prop_of_src_done w_row x hxlen obs)
  | failure obs => exact absurd (src_never_fails w_row x hxlen obs) (fun h => h)
  | cont obs tail ih =>
      have hops : propOps w_row x _ = 1 :=
        propOps_of_cont w_row x _ (prop_of_src_cont w_row x hxlen obs)
      have hreads : propReads w_row x _ = 2 :=
        propReads_of_cont w_row x _ (prop_of_src_cont w_row x hxlen obs)
      have h := Cert.cont (prop_of_src_cont w_row x hxlen obs) ih
      rw [hops, hreads] at h
      simpa only [Nat.mul_succ] using h

/-! ## (g) The registered cost model -/

/-- **The counts a certificate records, mapped to the `CostBridge.Perf` record the registered cost
map reads.** The record is a function of the certificate's own counts, not a restatement of them:
`CostBridge.toCost` then reads `ops := mulAdds` and `path := visits` from it, and `loads` and
`stores` do not reach the registered order (`CostBridge.loads_invisible_to_order`). Temporal cost of
this projection is O(1); spatial cost is O(1) words. -/
def perfOf {S V : Type u} {body : S → Result (ControlFlow S V)} {opc rdc : S → Nat}
    {s : S} {res : Result V} {v o r : Nat} (_c : Cert body opc rdc s res v o r) :
    CostBridge.Perf := ⟨v, o, r, 0, 0, 0⟩

/-- **The two bodies' counts at the production loop's entry state.** For a row of width `n` whose
nonempty prefix sums stay inside the `i128` window, the production run's certificate records
`n + 1` visits, `2n` checked `i128` operations and `2n` element reads; the proposed run's
certificate records `n + 1` visits, `n` checked `i128` operations and the same `2n` element reads,
and the two runs return the same outcome and the same number of dependent steps. The counts are the
certificates', not a declaration. -/
theorem entry_certs (w_row x : Slice I64) (n : Usize) (hn : n.val = w_row.val.length)
    (hfit : CostBridge.DotFit w_row x) :
    ∃ (res : I128 × Bool)
        (cS : Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x)
            (0#i128, 0#usize, true) (ok res) (n.val + 1) (2 * n.val) (2 * n.val))
        (cP : Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x)
            (0#i128, 0#usize, true) (ok res) (n.val + 1) n.val (2 * n.val)),
      (perfOf cS).visits = (perfOf cP).visits := by
  obtain ⟨⟨S, htrace⟩, _⟩ := RowDotResource.entry_visits_exact w_row x n hn hfit.1 hfit.2
  exact ⟨(S, true), src_cert_of_trace w_row x htrace,
    prop_cert_of_trace w_row x hfit.1 htrace, rfl⟩

/-- **A run that stops on its first visit.** The flag exit is reachable from every state, and both
bodies take it without performing a multiplication or an addition. -/
theorem flag_entry_certs (w_row x : Slice I64) (hxlen : x.val.length = w_row.val.length)
    (acc : I128) (i : Usize) :
    ∃ (res : I128 × Bool)
        (cS : Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x) (acc, i, false)
            (ok res) 1 0 0)
        (cP : Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x) (acc, i, false)
            (ok res) 1 0 0),
      (perfOf cS).visits = (perfOf cP).visits := by
  have htrace := RowDotResource.flag_trace w_row x acc i
  exact ⟨(acc, false), src_cert_of_trace w_row x htrace,
    prop_cert_of_trace w_row x hxlen htrace, rfl⟩

/-- The registered record the proposed run's certificate yields for a width-`n` row. -/
theorem prop_cost_of_entry (w_row x : Slice I64) (n : Usize) {res : I128 × Bool}
    (cP : Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x)
      (0#i128, 0#usize, true) (ok res) (n.val + 1) n.val (2 * n.val)) :
    CostBridge.toCost (perfOf cP) = ⟨n.val, 0, 0, 0, 0, 0, n.val + 1⟩ := rfl

/-- The registered record the production run's certificate yields for a width-`n` row. -/
theorem src_cost_of_entry (w_row x : Slice I64) (n : Usize) {res : I128 × Bool}
    (cS : Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x)
      (0#i128, 0#usize, true) (ok res) (n.val + 1) (2 * n.val) (2 * n.val)) :
    CostBridge.toCost (perfOf cS) = ⟨2 * n.val, 0, 0, 0, 0, 0, n.val + 1⟩ := rfl

/-- **Strict improvement in the registered cost model.** For a row of width `n ≥ 1` whose nonempty
prefix sums stay inside the `i128` window, the proposed run's registered cost is smaller than the
production run's in every field that reaches the order and strictly smaller in the checked
arithmetic count: it performs one checked addition per visited position where the production body
performed a checked addition and a checked multiplication. `CostAlgebra.Cost` carries no field for
element traffic, bytes copied, allocations, transfers, dispatches or peak-live storage, and the
replacement changes none of them. -/
theorem strictly_cheaper_entry (w_row x : Slice I64) (n : Usize) (hpos : 1 ≤ n.val)
    {res : I128 × Bool}
    (cP : Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x)
      (0#i128, 0#usize, true) (ok res) (n.val + 1) n.val (2 * n.val))
    (cS : Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x)
      (0#i128, 0#usize, true) (ok res) (n.val + 1) (2 * n.val) (2 * n.val)) :
    CostAlgebra.strictlyCheaper (CostBridge.toCost (perfOf cP)) (CostBridge.toCost (perfOf cS)) := by
  rw [prop_cost_of_entry w_row x n cP, src_cost_of_entry w_row x n cS]
  have hlt : (n.val : Nat) < 2 * n.val := by omega
  refine ⟨CostAlgebra.Cost.le_of_fields (Nat.le_of_lt hlt) (Nat.le_refl _) (Nat.le_refl _)
    (Nat.zero_le _) (Nat.zero_le _) (Nat.zero_le _) (Nat.le_refl _), ?_⟩
  intro h
  have hm : (n.val : Nat) = 2 * n.val := congrArg CostAlgebra.Cost.ops h
  exact absurd hm (by omega)

/-- **The strict `Nat` cost reduction.** -/
theorem improves_entry (w_row x : Slice I64) (n : Usize) (hpos : 1 ≤ n.val)
    {res : I128 × Bool}
    (cP : Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x)
      (0#i128, 0#usize, true) (ok res) (n.val + 1) n.val (2 * n.val))
    (cS : Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x)
      (0#i128, 0#usize, true) (ok res) (n.val + 1) (2 * n.val) (2 * n.val)) :
    CostAlgebra.registeredCost (CostBridge.toCost (perfOf cP)) <
      CostAlgebra.registeredCost (CostBridge.toCost (perfOf cS)) :=
  CostAlgebra.dominates_improves (strictly_cheaper_entry w_row x n hpos cP cS)

/-- **The reachable witness.** Two length-two rows of `1` need three invocations of each body. The
production body performs four checked `i128` operations over those invocations; the proposed body
performs two, and returns the same `Ok 0`. -/
theorem wit_entry_certs :
    ∃ (res : I128 × Bool)
        (cS : Cert (srcBodyFn RowDotResource.witRow RowDotResource.witVec)
            (srcOps RowDotResource.witRow RowDotResource.witVec)
            (srcReads RowDotResource.witRow RowDotResource.witVec)
            (0#i128, 0#usize, true) (ok res) 3 4 4)
        (cP : Cert (propBodyFn RowDotResource.witRow RowDotResource.witVec)
            (propOps RowDotResource.witRow RowDotResource.witVec)
            (propReads RowDotResource.witRow RowDotResource.witVec)
            (0#i128, 0#usize, true) (ok res) 3 2 4),
      (perfOf cS).loads = (perfOf cP).loads := by
  obtain ⟨S, _, htrace⟩ := RowDotResource.wit_visits
  exact ⟨(S, true), src_cert_of_trace _ _ htrace,
    prop_cert_of_trace _ _ (by rw [RowDotResource.witVec_len, RowDotResource.witRow_len]) htrace,
    rfl⟩

/-- **The witness's registered records.** Two checked operations in three dependent steps map to
`⟨2, 0, 0, 0, 0, 0, 3⟩`; four checked operations map to `⟨4, 0, 0, 0, 0, 0, 3⟩`. -/
theorem wit_costs {res : I128 × Bool}
    (cP : Cert (propBodyFn RowDotResource.witRow RowDotResource.witVec)
      (propOps RowDotResource.witRow RowDotResource.witVec)
      (propReads RowDotResource.witRow RowDotResource.witVec) (0#i128, 0#usize, true) (ok res) 3 2 4)
    (cS : Cert (srcBodyFn RowDotResource.witRow RowDotResource.witVec)
      (srcOps RowDotResource.witRow RowDotResource.witVec)
      (srcReads RowDotResource.witRow RowDotResource.witVec) (0#i128, 0#usize, true) (ok res) 3 4 4) :
    CostBridge.toCost (perfOf cP) = ⟨2, 0, 0, 0, 0, 0, 3⟩ ∧
      CostBridge.toCost (perfOf cS) = ⟨4, 0, 0, 0, 0, 0, 3⟩ := ⟨rfl, rfl⟩

/-- **The witness is a strict improvement.** -/
theorem wit_strictly_cheaper {res : I128 × Bool}
    (cP : Cert (propBodyFn RowDotResource.witRow RowDotResource.witVec)
      (propOps RowDotResource.witRow RowDotResource.witVec)
      (propReads RowDotResource.witRow RowDotResource.witVec) (0#i128, 0#usize, true) (ok res) 3 2 4)
    (cS : Cert (srcBodyFn RowDotResource.witRow RowDotResource.witVec)
      (srcOps RowDotResource.witRow RowDotResource.witVec)
      (srcReads RowDotResource.witRow RowDotResource.witVec) (0#i128, 0#usize, true) (ok res) 3 4 4) :
    CostAlgebra.strictlyCheaper (CostBridge.toCost (perfOf cP)) (CostBridge.toCost (perfOf cS)) := by
  obtain ⟨hP, hS⟩ := wit_costs cP cS
  rw [hP, hS]
  refine ⟨CostAlgebra.Cost.le_of_fields (by decide) (Nat.le_refl _) (Nat.le_refl _)
    (Nat.zero_le _) (Nat.zero_le _) (Nat.zero_le _) (Nat.le_refl _), ?_⟩
  intro h
  have hm : (2 : Nat) = 4 := congrArg CostAlgebra.Cost.ops h
  exact absurd hm (by decide)

/-- **The witness's strict `Nat` cost reduction.** -/
theorem wit_improves {res : I128 × Bool}
    (cP : Cert (propBodyFn RowDotResource.witRow RowDotResource.witVec)
      (propOps RowDotResource.witRow RowDotResource.witVec)
      (propReads RowDotResource.witRow RowDotResource.witVec) (0#i128, 0#usize, true) (ok res) 3 2 4)
    (cS : Cert (srcBodyFn RowDotResource.witRow RowDotResource.witVec)
      (srcOps RowDotResource.witRow RowDotResource.witVec)
      (srcReads RowDotResource.witRow RowDotResource.witVec) (0#i128, 0#usize, true) (ok res) 3 4 4) :
    CostAlgebra.registeredCost (CostBridge.toCost (perfOf cP)) <
      CostAlgebra.registeredCost (CostBridge.toCost (perfOf cS)) :=
  CostAlgebra.dominates_improves (wit_strictly_cheaper cP cS)

/-- **The empty product ties.** A run that stops on its first visit performed no checked operation
in either body, so the two records are equal and the registered model reports no improvement on an
input the loop refuses immediately. This is the boundary of the claim: the strict improvement is on
runs that perform at least one multiply-add. -/
theorem flag_ties (w_row x : Slice I64) (acc : I128) (i : Usize) {res : I128 × Bool}
    (cP : Cert (propBodyFn w_row x) (propOps w_row x) (propReads w_row x) (acc, i, false)
      (ok res) 1 0 0)
    (cS : Cert (srcBodyFn w_row x) (srcOps w_row x) (srcReads w_row x) (acc, i, false)
      (ok res) 1 0 0) :
    CostBridge.toCost (perfOf cP) = CostBridge.toCost (perfOf cS) ∧
      ¬ CostAlgebra.strictlyCheaper (CostBridge.toCost (perfOf cP))
          (CostBridge.toCost (perfOf cS)) :=
  ⟨rfl, CostAlgebra.tie_not_improvement _⟩

end RowDotAdmission

open Lean Elab Command in
/-- `#resource_audit <headline>`: the headline must kernel-check with no logical constant outside
{`propext`, `Classical.choice`, `Quot.sound`}. A rejected headline is reported as an `error:` and
fails the build. `run_cmd Lean.logInfo "ROW_DOT_ADMISSION_KERNEL_OK"` below is a completion
marker, not a cleanliness claim: a bare `throwError` in a command is reported and the command
sequence continues, so the marker follows a rejected headline. RowDotAdmission_gate.py's green path
therefore refuses any build whose output contains an `error:`, which is what makes a rejection
observable as red. -/
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"ROW_DOT_ADMISSION_AXIOM_REJECT {c}: {bad.toList}"
    logInfo m!"ROW_DOT_ADMISSION_AUDIT {c}: axioms {axioms.toList}"

#resource_audit RowDotAdmission.srcBodyFn_eq
#resource_audit RowDotAdmission.propBodyFn_eq
#resource_audit RowDotAdmission.widening_val
#resource_audit RowDotAdmission.i64_bounds
#resource_audit RowDotAdmission.i128_window
#resource_audit RowDotAdmission.widened_product_window
#resource_audit RowDotAdmission.widened_product_fits
#resource_audit RowDotAdmission.widened_plain_product_ok
#resource_audit RowDotAdmission.widened_product_val
#resource_audit RowDotAdmission.widened_plain_product_is_checked
#resource_audit RowDotAdmission.visit_agree
#resource_audit RowDotAdmission.visit_exit_len_agree
#resource_audit RowDotAdmission.visit_exit_flag_agree
#resource_audit RowDotAdmission.body_agree
#resource_audit RowDotAdmission.loop_agree
#resource_audit RowDotAdmission.bind_ok_reads
#resource_audit RowDotAdmission.round_shift_agree
#resource_audit RowDotAdmission.prop_fixed_unfold
#resource_audit RowDotAdmission.preserves_row_length
#resource_audit RowDotAdmission.preserves_acc_overflow
#resource_audit RowDotAdmission.preserves_out_high
#resource_audit RowDotAdmission.preserves_out_low
#resource_audit RowDotAdmission.preserves_ok
#resource_audit RowDotAdmission.preserves_ok_of
#resource_audit RowDotAdmission.preserves
#resource_audit RowDotAdmission.prop_of_src_cont
#resource_audit RowDotAdmission.prop_of_src_done
#resource_audit RowDotAdmission.src_never_fails
#resource_audit RowDotAdmission.mul_arm_always_some
#resource_audit RowDotAdmission.srcOps_of_cont
#resource_audit RowDotAdmission.srcReads_of_cont
#resource_audit RowDotAdmission.propOps_of_cont
#resource_audit RowDotAdmission.propReads_of_cont
#resource_audit RowDotAdmission.src_counts
#resource_audit RowDotAdmission.prop_counts
#resource_audit RowDotAdmission.src_cert_of_trace
#resource_audit RowDotAdmission.prop_cert_of_trace
#resource_audit RowDotAdmission.entry_certs
#resource_audit RowDotAdmission.flag_entry_certs
#resource_audit RowDotAdmission.prop_cost_of_entry
#resource_audit RowDotAdmission.src_cost_of_entry
#resource_audit RowDotAdmission.strictly_cheaper_entry
#resource_audit RowDotAdmission.improves_entry
#resource_audit RowDotAdmission.wit_entry_certs
#resource_audit RowDotAdmission.wit_costs
#resource_audit RowDotAdmission.wit_strictly_cheaper
#resource_audit RowDotAdmission.wit_improves
#resource_audit RowDotAdmission.flag_ties

run_cmd Lean.logInfo "ROW_DOT_ADMISSION_KERNEL_OK"
