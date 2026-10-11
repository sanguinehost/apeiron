import CostAlgebra
import RowDotResource

/-
SCOPE: proof/resource/CostBridge.lean, new. Carries the counts a primitive's own resource
theorem produces (body visits, checked multiply-adds, element reads, element writes, bytes copied,
buffer allocations) into the registered `CostAlgebra.Cost` record, proves the map is monotone in
every count it reads, proves it preserves dependent composition, and exhibits two schedules of one
row-dot computation with equal observations of which one is strictly cheaper.
Complete when the module kernel-checks with every headline audited and the two seeded mapping
defects reject.
REGISTER: resource model; no Rust source, no extraction, no inference implementation, no
canonical ring correspondence, no hardware timing.
CORPUS: no match. grep for `structure Perf`/`structure Count`/`toCost`/`resourceCost` over
/home/socol/Workspace/sanguine/proof/Apeiron/, /home/socol/Workspace/apeiron/proof and
/home/socol/Workspace/apeiron/docs returned no file. Native grade of this module: proved.
REFERENCE: /home/socol/Workspace/apeiron/proof/resource/CostAlgebra.lean (the `Cost` record, its
componentwise `≤`, `costAdd`, `strictlyCheaper`, `dominates_improves`) and
/home/socol/Workspace/apeiron/proof/resource/RowDotResource.lean (the actual extracted
`row_dot_fixed_loop.body`, its `entry_visits_exact`: `n + 1` body visits and `n` checked `i128`
multiply-adds for a width-`n` row, and `body_cont`: two indexed row reads per visit).
GATE: python3 proof/resource/CostBridgeGate.py --check --controls, cwd repository root, under
flock /tmp/opencode/apeiron-lane-3.lock and systemd-run MemoryMax=12G MemorySwapMax=0 with
LEAN_NUM_THREADS=1. Observed red before this file existed: `test -f proof/resource/CostBridge.lean`
exits 1. Budget 540 s for the whole invocation including the two controls.
GRADE: proved for the map, its monotonicity, the composition law and the two schedules, under the
closed logical cone {propext, Classical.choice, Quot.sound} checked by `#resource_audit`. The visit
and multiply-add counts are `RowDotResource`'s closed form for the actual extracted body, at grade
`proved`. Grade `model`: the element-read count `2n`, which is read off `body_cont`'s two indexed
reads rather than proved against a memory model, and the critical-path convention `path := visits`.
Unmeasured and unclaimed: latency, bandwidth, device dispatch cost, cache behaviour, compiler or
codegen cost, the arithmetic of `row_dot_fixed`'s epilogue (excluded from `ops`, which is therefore
a lower bound), and the cost of any specific kernel.

The module-local `#resource_audit` below is this file's own elaborator, carrying the
`COST_BRIDGE_AUDIT` and `COST_BRIDGE_AXIOM_REJECT` markers the gate reads. `CostAlgebra` and
`RowDotResource` each register an elaborator of that name, and one file cannot import both, so
CostBridgeGate.py builds verbatim copies of the two with that single identifier renamed, and refuses
to run unless each copy is byte-equal to the committed file modulo exactly that rename.
-/

namespace CostBridge

open Aeneas Aeneas.Std Result

/-! ## (a) The source-level count record and the map into the registered algebra -/

/-- The source-level counts one primitive's own resource theorem produces. Each count is in its own
unit and is never traded against another:

- `visits`: invocations of the primitive's body.
- `mulAdds`: checked multiply-adds performed.
- `loads`: element reads performed.
- `stores`: element writes performed.
- `bytes`: bytes copied between layouts.
- `allocs`: buffer allocations.
-/
structure Perf where
  visits : Nat
  mulAdds : Nat
  loads : Nat
  stores : Nat
  bytes : Nat
  allocs : Nat
  deriving DecidableEq, Repr

/-- The record of a primitive that does nothing. -/
def Perf.zero : Perf := ⟨0, 0, 0, 0, 0, 0⟩

/-- Dependent composition of two primitives' source-level counts: a primitive run after another
performs both bodies, so every count totals. -/
def Perf.add (a b : Perf) : Perf :=
  ⟨a.visits + b.visits, a.mulAdds + b.mulAdds, a.loads + b.loads,
    a.stores + b.stores, a.bytes + b.bytes, a.allocs + b.allocs⟩

instance : Add Perf where add := Perf.add

/-- The dependent sum, unfolded, so that composition rewrites and its laws close. -/
@[simp] theorem Perf.add_apply (a b : Perf) : a + b = Perf.add a b := rfl

/-- Pointwise order on the source-level counts, in every count. -/
def Perf.costLe (a b : Perf) : Prop :=
  a.visits ≤ b.visits ∧ a.mulAdds ≤ b.mulAdds ∧ a.loads ≤ b.loads ∧ a.stores ≤ b.stores ∧
    a.bytes ≤ b.bytes ∧ a.allocs ≤ b.allocs

instance : LE Perf where le := Perf.costLe

/-- **The map from a primitive's source-level counts to the registered record.**

- `ops := mulAdds`, one unit per checked multiply-add.
- `pack := bytes`, one unit per copied byte.
- `alloc := allocs`, one unit per buffer allocation.
- `xfer := 0`: a scalar source theorem transfers nothing to or from a device, and no premise here
  could license a device count.
- `dispatch := 0`: a scalar primitive dispatches no kernel and drains no queue.
- `peak := 0`: the registered `peak` counts peak-live storage *bytes*, and a source-level count
  carries no buffer width, so no value is asserted for it here.
- `path := visits`: the critical path of one primitive is the length of its own dependent step
  chain, and the extracted row loop executes `n + 1` dependent body steps, the last being the
  terminating guard visit. So `path` is an upper bound on the `n` dependent accumulate steps.
-/
def toCost (p : Perf) : CostAlgebra.Cost :=
  ⟨p.mulAdds, p.bytes, p.allocs, 0, 0, 0, p.visits⟩

/-- Doing nothing maps to the cost of doing nothing. -/
theorem toCost_zero : toCost Perf.zero = CostAlgebra.zero := rfl

/-- **The map is monotone in the visit count.** Every other count of `p` is held fixed, so this
is the order's statement about one count alone. -/
theorem toCost_monotone_visits (p : Perf) (k : Nat) (h : p.visits ≤ k) :
    toCost p ≤ toCost { p with visits := k } :=
  CostAlgebra.Cost.le_of_fields (Nat.le_refl _) (Nat.le_refl _) (Nat.le_refl _) (Nat.zero_le _)
    (Nat.zero_le _) (Nat.zero_le _) h

/-- **The map is monotone in the checked multiply-add count.** -/
theorem toCost_monotone_mulAdds (p : Perf) (k : Nat) (h : p.mulAdds ≤ k) :
    toCost p ≤ toCost { p with mulAdds := k } :=
  CostAlgebra.Cost.le_of_fields h (Nat.le_refl _) (Nat.le_refl _) (Nat.zero_le _) (Nat.zero_le _)
    (Nat.zero_le _) (Nat.le_refl _)

/-- **The map is monotone in the copied-byte count.** -/
theorem toCost_monotone_bytes (p : Perf) (k : Nat) (h : p.bytes ≤ k) :
    toCost p ≤ toCost { p with bytes := k } :=
  CostAlgebra.Cost.le_of_fields (Nat.le_refl _) h (Nat.le_refl _) (Nat.zero_le _) (Nat.zero_le _)
    (Nat.zero_le _) (Nat.le_refl _)

/-- **The map is monotone in the allocation count.** -/
theorem toCost_monotone_allocs (p : Perf) (k : Nat) (h : p.allocs ≤ k) :
    toCost p ≤ toCost { p with allocs := k } :=
  CostAlgebra.Cost.le_of_fields (Nat.le_refl _) (Nat.le_refl _) h (Nat.zero_le _) (Nat.zero_le _)
    (Nat.zero_le _) (Nat.le_refl _)

/-- **The map is monotone in the element-read count, and that monotonicity carries no registered
information.** The registered vocabulary of `CostAlgebra.Cost` has no field for host element
traffic, so `toCost` is unchanged by `loads` and the inequality holds whatever `q` reads. The
consequence is recorded as `loads_invisible_to_order`. -/
theorem toCost_monotone_loads (p q : Perf) (_h : p.loads ≤ q.loads) :
    toCost p ≤ toCost { p with loads := q.loads } :=
  CostAlgebra.Cost.le_of_fields (Nat.le_refl _) (Nat.le_refl _) (Nat.le_refl _) (Nat.le_refl _)
    (Nat.le_refl _) (Nat.le_refl _) (Nat.le_refl _)

/-- **The map is monotone in the element-write count, with the same caveat as for reads.** -/
theorem toCost_monotone_stores (p q : Perf) (_h : p.stores ≤ q.stores) :
    toCost p ≤ toCost { p with stores := q.stores } :=
  CostAlgebra.Cost.le_of_fields (Nat.le_refl _) (Nat.le_refl _) (Nat.le_refl _) (Nat.le_refl _)
    (Nat.le_refl _) (Nat.le_refl _) (Nat.le_refl _)

/-- **Monotone in every count at once.** The two element-traffic counts do not reach the registered
order, as `toCost_monotone_loads` records, so this is monotone in the four registered counts and
holds unconditionally in the other two. -/
theorem toCost_monotone {p q : Perf} (hv : p.visits ≤ q.visits) (hm : p.mulAdds ≤ q.mulAdds)
    (_hl : p.loads ≤ q.loads) (_hs : p.stores ≤ q.stores) (hb : p.bytes ≤ q.bytes)
    (ha : p.allocs ≤ q.allocs) : toCost p ≤ toCost q :=
  CostAlgebra.Cost.le_of_fields hm hb ha (Nat.zero_le _) (Nat.zero_le _) (Nat.zero_le _) hv

/-- **The registered order cannot see element traffic.** Adding `k` element reads and `k` element
writes to a primitive leaves every registered field it does reach unchanged, so the two records tie
in the registered order: the model reports neither an improvement nor a regression from doubled
memory traffic. This is a boundary of the registered vocabulary of `CostAlgebra.Cost`, not a defect
of the map, and it is why `loads` and `stores` are carried but unmapped. -/
theorem loads_invisible_to_order (p : Perf) (k : Nat) :
    toCost { p with loads := p.loads + k, stores := p.stores + k } = toCost p ∧
      ¬ CostAlgebra.strictlyCheaper (toCost p)
          (toCost { p with loads := p.loads + k, stores := p.stores + k }) :=
  ⟨rfl, CostAlgebra.tie_not_improvement _⟩

/-- **The map preserves dependent composition.** Two primitives run after one another perform both
bodies, so their registered costs compose by `CostAlgebra.costAdd`; the map introduces no
combination of counts of its own and loses none under composition. -/
theorem toCost_add (a b : Perf) :
    toCost (Perf.add a b) = CostAlgebra.costAdd (toCost a) (toCost b) := by
  refine CostAlgebra.Cost.eq_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [toCost, Perf.add, CostAlgebra.costAdd]

/-- The composition law in the registered `+` of `CostAlgebra.Cost`. -/
theorem toCost_add_eq (a b : Perf) : toCost (a + b) = toCost a + toCost b :=
  toCost_add a b

/-- Composition of two source-level count records is monotone in each record. -/
theorem Perf.add_monotone {a b c : Perf} (h : a.costLe b) : (a + c).costLe (b + c) :=
  ⟨Nat.add_le_add_right h.1 _, Nat.add_le_add_right h.2.1 _, Nat.add_le_add_right h.2.2.1 _,
    Nat.add_le_add_right h.2.2.2.1 _, Nat.add_le_add_right h.2.2.2.2.1 _,
    Nat.add_le_add_right h.2.2.2.2.2 _⟩

/-! ## (b) The row-dot primitive's counts, read off a certificate of the actual body -/

/-- Reading the counts a certificate of a body records. `visits` counts invocations of the body and
`steps` counts the invocations that consume one row position, which `RowDotResource.row_trace_count`
identifies with the checked multiply-add count; each such visit reads the two row elements at the
index. -/
def tracePerf (visits steps : Nat) : Perf := ⟨visits, steps, 2 * steps, 0, 0, 0⟩

/-- **The source-level count of the row-dot primitive.** `RowDotResource.entry_visits_exact` fixes
the actual loop's counts at `n + 1` body invocations and `n` checked `i128` multiply-adds for a
width-`n` row, and every certificate of those counts agrees. `src/engine/matvec.rs` declares
`row_dot_fixed(w_row: &[i64], x: &[i64]) -> Result<i64, MatvecError>`, so the invocation borrows
both rows, copies no bytes and allocates no buffer. -/
def rowDotPerf (w _x : Slice I64) : Perf :=
  ⟨w.val.length + 1, w.val.length, 2 * w.val.length, 0, 0, 0⟩

/-- The joint premise the row-dot window requires: equal-length rows whose nonempty prefix sums stay
inside the `i128` window. -/
def DotFit (w x : Slice I64) : Prop :=
  x.val.length = w.val.length ∧
    ∀ k, 1 ≤ k → k ≤ w.val.length →
      I128.min ≤ RowDotResource.dotPrefix w x k ∧ RowDotResource.dotPrefix w x k ≤ I128.max

/-- **The map reads the extracted body's own certificate.** On rows satisfying `DotFit`, the reading
of *any* certificate of the entry state is exactly `rowDotPerf`: the visits and checked multiply-adds
the certificate records are the source's counts, not one derivation's. -/
theorem tracePerf_rowDot (w x : Slice I64) (hfit : DotFit w x) :
    ∀ (result : Result (I128 × Bool)) (visits steps : Nat),
      RowDotResource.RowTrace (RowDotResource.dotBodyFn w x) (0#i128, 0#usize, true)
        result visits steps →
      tracePerf visits steps = rowDotPerf w x := by
  intro result visits steps htrace
  have huniq : visits = w.len.val + 1 ∧ steps = w.len.val :=
    (RowDotResource.entry_visits_exact w x w.len (RowDotResource.sliceLenVal w).symm hfit.1
      hfit.2).2 result visits steps htrace
  simp only [tracePerf, rowDotPerf]
  rw [huniq.1, huniq.2, RowDotResource.sliceLenVal w]

/-- **A certificate of those exact counts exists.** The row-dot loop is invoked on the entry state,
and the record it yields is the record the map produces. -/
theorem rowDotPerf_trace (w x : Slice I64) (n : Usize) (hn : n.val = w.val.length)
    (hfit : DotFit w x) :
    ∃ S : I128, tracePerf (n.val + 1) n.val = rowDotPerf w x ∧
      RowDotResource.RowTrace (RowDotResource.dotBodyFn w x) (0#i128, 0#usize, true)
        (ok (S, true)) (n.val + 1) n.val := by
  obtain ⟨⟨S, htrace⟩, _⟩ := RowDotResource.entry_visits_exact w x n hn hfit.1 hfit.2
  refine ⟨S, ?_, htrace⟩
  show tracePerf (n.val + 1) n.val = rowDotPerf w x
  rw [hn]
  rfl

/-! ## (c) Schedules of the row-dot primitive, and the composition they perform -/

/-- What one invocation of the extracted entry point returns: the `Result` effect of the Rust
`Result<i64, MatvecError>`. -/
abbrev RowDotResult := Result (core.result.Result I64 row_dot_source.matvec.MatvecError)

/-- One invocation of the actual `row_dot_fixed` entry point: the borrowed weight row and the
activation row. -/
structure Invocation where
  w : Slice I64
  x : Slice I64

/-- A scheduled run of the row-dot primitive: its invocations, in order. Each invocation is one
dependent stage of the run. Temporal cost is O(n) recursive steps in the number n of invocations;
spatial cost is O(n) words, the mapped cost list. -/
abbrev Schedule := List Invocation

/-- The observable result of a scheduled run: the values the extracted entry point returns, in
order. Two schedules of one computation agree here and may differ only in `runCost`. -/
def run (s : Schedule) : List RowDotResult :=
  s.map (fun i => row_dot_source.matvec.row_dot_fixed i.w i.x)

/-- The dependent composition of the invocations' source-level counts. -/
def schedulePerf (s : Schedule) : Perf :=
  (s.map (fun i => rowDotPerf i.w i.x)).foldr Perf.add Perf.zero

/-- The composed registered cost of a scheduled run. -/
def runCost (s : Schedule) : CostAlgebra.Cost :=
  (s.map (fun i => toCost (rowDotPerf i.w i.x))).foldr (· + ·) CostAlgebra.zero

/-- The composed cost is the map of the composed counts: composition and the map commute, so the map
adds no combination of counts of its own. -/
theorem runCost_eq_toCost : ∀ (s : Schedule), runCost s = toCost (schedulePerf s)
  | [] => by
    show CostAlgebra.zero = toCost Perf.zero
    rw [toCost_zero]
  | i :: s => by
    show CostAlgebra.costAdd (toCost (rowDotPerf i.w i.x)) (runCost s) =
      toCost (Perf.add (rowDotPerf i.w i.x) (schedulePerf s))
    rw [runCost_eq_toCost s, toCost_add]

/-- **Composing two schedules composes their costs.** Temporal cost is one recursive step per
invocation of the first schedule, on top of the cost of computing each append. -/
theorem runCost_append : ∀ (a b : Schedule),
    runCost (a ++ b) = CostAlgebra.costAdd (runCost a) (runCost b)
  | [], b => by
    show runCost b = CostAlgebra.costAdd CostAlgebra.zero (runCost b)
    exact ((CostAlgebra.add_comm CostAlgebra.zero (runCost b)).trans
      (CostAlgebra.add_zero _)).symm
  | i :: a, b => by
    show CostAlgebra.costAdd (toCost (rowDotPerf i.w i.x)) (runCost (a ++ b)) =
      CostAlgebra.costAdd (CostAlgebra.costAdd (toCost (rowDotPerf i.w i.x)) (runCost a))
        (runCost b)
    rw [runCost_append a b]
    simp only [CostAlgebra.costAdd, Nat.add_assoc]

/-- The composition law of the map on schedules: composing two schedules' count records maps to the
registered sum of their composed costs. This is `toCost_add` read at schedule granularity. -/
theorem runCost_append_toCost_add (a b : Schedule) :
    toCost (Perf.add (schedulePerf a) (schedulePerf b)) =
      CostAlgebra.costAdd (runCost a) (runCost b) := by
  rw [toCost_add, runCost_eq_toCost a, runCost_eq_toCost b]

/-- The composed cost of a run of one invocation. -/
theorem runCost_singleton (i : Invocation) : runCost [i] = toCost (rowDotPerf i.w i.x) :=
  CostAlgebra.add_zero _

/-- The composed cost of a run of two invocations, which is `runCost_append` on two singletons. -/
theorem runCost_pair (i j : Invocation) :
    runCost [i, j] = CostAlgebra.costAdd (toCost (rowDotPerf i.w i.x)) (toCost (rowDotPerf j.w j.x)) :=
  congrArg (fun t => CostAlgebra.costAdd (toCost (rowDotPerf i.w i.x)) t)
    (CostAlgebra.add_zero _)

/-- **The composed cost is monotone in every invocation**, so replacing one invocation by a
no-more-expensive one never raises the cost of the run that contains it. -/
theorem runCost_mono : ∀ {a b : Schedule},
    CostAlgebra.Pointwise (fun i j => (rowDotPerf i.w i.x).costLe (rowDotPerf j.w j.x)) a b →
    runCost a ≤ runCost b
  | [], [], _ => CostAlgebra.zero_le _
  | i :: is, j :: js, h => by
    show CostAlgebra.costAdd (toCost (rowDotPerf i.w i.x)) (runCost is) ≤
      CostAlgebra.costAdd (toCost (rowDotPerf j.w j.x)) (runCost js)
    exact Std.IsPreorder.le_trans _ _ _
      (CostAlgebra.add_monotone (c := runCost is)
        (toCost_monotone h.1.1 h.1.2.1 h.1.2.2.1 h.1.2.2.2.1 h.1.2.2.2.2.1 h.1.2.2.2.2.2))
      (CostAlgebra.add_monotone_left (c := toCost (rowDotPerf j.w j.x)) (runCost_mono h.2))

/-! ## (d) Two schedules of one row-dot computation, with equal observations -/

/-- The reachable witness row `[1, 1]` in Q32.32 of `RowDotResource.witRow`, against the same
activation row. -/
abbrev witInvocation : Invocation := ⟨RowDotResource.witRow, RowDotResource.witVec⟩

/-- The same weight row padded to the tile width four with zeros. -/
def padRow : Slice I64 := Slice.from [1#i64, 1#i64, 0#i64, 0#i64] (by simp; scalar_tac)

/-- The same activation row padded to the tile width four with zeros. -/
def padVec : Slice I64 := Slice.from [1#i64, 1#i64, 0#i64, 0#i64] (by simp; scalar_tac)

abbrev padInvocation : Invocation := ⟨padRow, padVec⟩

/-- The candidate: the dot product of the row against the activation, one invocation of the actual
entry point on the exact widths. -/
def exactSchedule : Schedule := [witInvocation]

/-- The baseline: the same dot product, one invocation on the rows padded to the tile width four.
The extra two positions contribute the products `0 · 0`, so the accumulator, and with it the
returned value, is unchanged. -/
def paddedSchedule : Schedule := [padInvocation]

/-- The registered cost of the exact-width run: two checked multiply-adds and three dependent steps,
no packing, allocation, transfer, dispatch or peak. -/
def exactCost : CostAlgebra.Cost := ⟨2, 0, 0, 0, 0, 0, 3⟩

/-- The registered cost of the padded run. -/
def paddedCost : CostAlgebra.Cost := ⟨4, 0, 0, 0, 0, 0, 5⟩

theorem padRow_len : padRow.val.length = 4 := by simp [padRow]

theorem padVec_len : padVec.val.length = 4 := by simp [padVec]

/-- The joint window for the padded rows: every nonempty prefix sum stays inside `i128`. -/
theorem pad_fit : DotFit padRow padVec := by
  refine ⟨by rw [padVec_len, padRow_len], ?_⟩
  intro k hk1 hk2
  rw [padRow_len] at hk2
  have hk : k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 := by omega
  rcases hk with hk' | hk' | hk' | hk'
  · rw [hk']
    norm_num [RowDotResource.dotPrefix, padRow, padVec, I128.min, IScalar.min, I128.numBits,
      I128.max, IScalar.max]
  · rw [hk']
    norm_num [RowDotResource.dotPrefix, padRow, padVec, I128.min, IScalar.min, I128.numBits,
      I128.max, IScalar.max]
  · rw [hk']
    norm_num [RowDotResource.dotPrefix, padRow, padVec, I128.min, IScalar.min, I128.numBits,
      I128.max, IScalar.max]
  · rw [hk']
    norm_num [RowDotResource.dotPrefix, padRow, padVec, I128.min, IScalar.min, I128.numBits,
      I128.max, IScalar.max]

/-- **The padded run is reachable and its source-level count is the closed form.** The actual body
is invoked `5` times on the four-wide rows and performs exactly `4` checked `i128` multiply-adds,
and the loop stops with the exact ℤ prefix `2`. -/
theorem pad_visits : ∃ S : I128, S.val = 2 ∧
    RowDotResource.RowTrace (RowDotResource.dotBodyFn padRow padVec) (0#i128, 0#usize, true)
      (ok (S, true)) 5 4 := by
  obtain ⟨⟨S, htrace⟩, _⟩ := RowDotResource.entry_visits_exact padRow padVec 4#usize
    (by norm_num [padRow_len]) pad_fit.1 pad_fit.2
  obtain ⟨b, hb, hS, hSval⟩ := WP.spec_imp_exists
    (RowDotResource.loop_spec padRow padVec pad_fit.1 pad_fit.2)
  have hproj : row_dot_source.matvec.row_dot_fixed_loop padRow padVec 0#i128 0#usize true =
      ok (S, true) := RowDotResource.row_trace_projection htrace
  rw [hproj] at hb
  have hEq : (S, true) = b := Result.ok_injective hb
  rw [← hEq] at hSval
  have hSeq : S = hS := congrArg Prod.fst hSval.1
  refine ⟨S, ?_, htrace⟩
  rw [hSeq, hSval.2, padRow_len]
  norm_num [RowDotResource.dotPrefix, padRow, padVec]

/-- **The padded run returns the same value as the exact run, at the actual entry point.** The
two invocations reach the same accumulator value, so `row_dot_fixed_unfold` shows their epilogues
and their returned values agree; the extracted `row_dot_fixed` on the four-wide rows of
`[1, 1, 0, 0]` therefore returns `Ok 0`. -/
theorem padded_success :
    row_dot_source.matvec.row_dot_fixed padRow padVec = ok (core.result.Result.Ok 0#i64) := by
  obtain ⟨S, hS, htrace⟩ := pad_visits
  have hloop : row_dot_source.matvec.row_dot_fixed_loop padRow padVec 0#i128 0#usize true =
      ok (S, true) := RowDotResource.row_trace_projection htrace
  obtain ⟨Sw, hSw, hwit⟩ := RowDotResource.wit_visits
  have hwitLoop :
      row_dot_source.matvec.row_dot_fixed_loop RowDotResource.witRow RowDotResource.witVec
        0#i128 0#usize true = ok (Sw, true) := RowDotResource.row_trace_projection hwit
  have hEq : S = Sw := IScalar.eq_of_val_eq (by rw [hS, hSw])
  have hpad : row_dot_source.matvec.row_dot_fixed padRow padVec =
      row_dot_source.matvec.row_dot_fixed RowDotResource.witRow RowDotResource.witVec := by
    rw [RowDotResource.row_dot_fixed_unfold padRow padVec S (by rw [padRow_len, padVec_len]) hloop,
      RowDotResource.row_dot_fixed_unfold RowDotResource.witRow RowDotResource.witVec Sw
        (by rw [RowDotResource.witRow_len, RowDotResource.witVec_len]) hwitLoop]
    rw [hEq]
  rw [hpad]
  exact RowDotResource.wit_success

/-- The exact run returns that value at the actual entry point. -/
theorem exact_run : run exactSchedule = [ok (core.result.Result.Ok 0#i64)] := by
  show [row_dot_source.matvec.row_dot_fixed RowDotResource.witRow RowDotResource.witVec] = _
  rw [RowDotResource.wit_success]

/-- The padded run returns the same value. -/
theorem padded_run : run paddedSchedule = [ok (core.result.Result.Ok 0#i64)] := by
  show [row_dot_source.matvec.row_dot_fixed padRow padVec] = _
  rw [padded_success]

/-- **The two schedules compute the same thing.** Their observations agree at the extracted entry
point, and only their costs differ. -/
theorem observations_equal : run exactSchedule = run paddedSchedule :=
  exact_run.trans padded_run.symm

/-- The exact run maps to two checked multiply-adds, three dependent steps and nothing else. -/
theorem exact_run_cost : runCost exactSchedule = exactCost := by
  show CostAlgebra.costAdd (toCost (rowDotPerf RowDotResource.witRow RowDotResource.witVec))
    CostAlgebra.zero = _
  have hz : CostAlgebra.costAdd
      (toCost (rowDotPerf RowDotResource.witRow RowDotResource.witVec)) CostAlgebra.zero =
      toCost (rowDotPerf RowDotResource.witRow RowDotResource.witVec) := CostAlgebra.add_zero _
  rw [hz, toCost, rowDotPerf, RowDotResource.witRow_len]
  decide

/-- The padded run maps to four checked multiply-adds, five dependent steps and nothing else. -/
theorem padded_run_cost : runCost paddedSchedule = paddedCost := by
  show CostAlgebra.costAdd (toCost (rowDotPerf padRow padVec)) CostAlgebra.zero = _
  have hz : CostAlgebra.costAdd (toCost (rowDotPerf padRow padVec)) CostAlgebra.zero =
      toCost (rowDotPerf padRow padVec) := CostAlgebra.add_zero _
  rw [hz, toCost, rowDotPerf, padRow_len]
  decide

/-- **The exact-width schedule strictly dominates the padded schedule in the registered order.** No
registered resource grows and two of them shrink, so the padding buys nothing and costs work. -/
theorem exact_dominates_padded :
    CostAlgebra.strictlyCheaper (runCost exactSchedule) (runCost paddedSchedule) := by
  show exactCost ≤ paddedCost ∧ exactCost ≠ paddedCost
  decide

/-- Domination yields the strict `Nat` cost reduction the candidate certificate requires, on this
non-empty witness of registered improvement. -/
theorem exact_improves_padded :
    CostAlgebra.registeredCost (runCost exactSchedule) <
      CostAlgebra.registeredCost (runCost paddedSchedule) :=
  CostAlgebra.dominates_improves exact_dominates_padded

/-- The improvement consists of both the multiply-add count and the dependent step count: the padded
run performs twice the checked arithmetic and two more dependent steps for the same value. -/
theorem padding_costs_two_more_mulAdds :
    (runCost paddedSchedule).ops - (runCost exactSchedule).ops = 2 := by
  show paddedCost.ops - exactCost.ops = 2
  decide

theorem padding_costs_two_more_visits :
    (runCost paddedSchedule).path - (runCost exactSchedule).path = 2 := by
  show paddedCost.path - exactCost.path = 2
  decide

/-! ## (e) The same comparison under composition -/

/-- The candidate as two dependent invocations. -/
def exactPair : Schedule := [witInvocation, witInvocation]

/-- The baseline as two dependent invocations on the padded rows. -/
def paddedPair : Schedule := [padInvocation, padInvocation]

/-- The registered cost of the two-invocation exact run. -/
def composedExactCost : CostAlgebra.Cost := ⟨4, 0, 0, 0, 0, 0, 6⟩

/-- The registered cost of the two-invocation padded run. -/
def composedPaddedCost : CostAlgebra.Cost := ⟨8, 0, 0, 0, 0, 0, 10⟩

theorem composed_observations_equal : run exactPair = run paddedPair := by
  show [row_dot_source.matvec.row_dot_fixed RowDotResource.witRow RowDotResource.witVec,
      row_dot_source.matvec.row_dot_fixed RowDotResource.witRow RowDotResource.witVec] =
    [row_dot_source.matvec.row_dot_fixed padRow padVec,
      row_dot_source.matvec.row_dot_fixed padRow padVec]
  rw [RowDotResource.wit_success, padded_success]

theorem composed_exact_cost : runCost exactPair = composedExactCost := by
  show CostAlgebra.costAdd (toCost (rowDotPerf RowDotResource.witRow RowDotResource.witVec))
    (CostAlgebra.costAdd (toCost (rowDotPerf RowDotResource.witRow RowDotResource.witVec))
      CostAlgebra.zero) = _
  have hz : CostAlgebra.costAdd
      (toCost (rowDotPerf RowDotResource.witRow RowDotResource.witVec)) CostAlgebra.zero =
      toCost (rowDotPerf RowDotResource.witRow RowDotResource.witVec) := CostAlgebra.add_zero _
  rw [hz, toCost, rowDotPerf, RowDotResource.witRow_len]
  decide

theorem composed_padded_cost : runCost paddedPair = composedPaddedCost := by
  show CostAlgebra.costAdd (toCost (rowDotPerf padRow padVec))
    (CostAlgebra.costAdd (toCost (rowDotPerf padRow padVec)) CostAlgebra.zero) = _
  have hz : CostAlgebra.costAdd (toCost (rowDotPerf padRow padVec)) CostAlgebra.zero =
      toCost (rowDotPerf padRow padVec) := CostAlgebra.add_zero _
  rw [hz, toCost, rowDotPerf, padRow_len]
  decide

theorem composed_exact_dominates_padded :
    CostAlgebra.strictlyCheaper (runCost exactPair) (runCost paddedPair) := by
  show composedExactCost ≤ composedPaddedCost ∧ composedExactCost ≠ composedPaddedCost
  decide

theorem composed_exact_improves_padded :
    CostAlgebra.registeredCost (runCost exactPair) <
      CostAlgebra.registeredCost (runCost paddedPair) :=
  CostAlgebra.dominates_improves composed_exact_dominates_padded

/-- The two-stage composition is the registered sum of the one-stage costs, which is `toCost_add`
applied once per composition. -/
theorem composed_is_sum_of_singles :
    runCost exactPair = CostAlgebra.costAdd (runCost exactSchedule) (runCost exactSchedule) ∧
      runCost paddedPair = CostAlgebra.costAdd (runCost paddedSchedule) (runCost paddedSchedule) := by
  constructor
  · exact (runCost_append exactSchedule exactSchedule).symm
  · exact (runCost_append paddedSchedule paddedSchedule).symm

end CostBridge

open Lean Elab Command in
/-- `#resource_audit <headline>`: the headline must kernel-check with no logical constant outside
{`propext`, `Classical.choice`, `Quot.sound`}. A rejected headline is reported as an `error:` and
fails the build. `run_cmd Lean.logInfo "COST_BRIDGE_KERNEL_OK"` below is a completion marker, not a
cleanliness claim: a bare `throwError` in a command is reported and the command sequence continues,
so the marker follows a rejected headline. CostBridgeGate.py's green path therefore refuses any
build whose output contains an `error:`, which is what makes a rejection observable as red. -/
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"COST_BRIDGE_AXIOM_REJECT {c}: {bad.toList}"
    logInfo m!"COST_BRIDGE_AUDIT {c}: axioms {axioms.toList}"

#resource_audit CostBridge.Perf.add_apply
#resource_audit CostBridge.toCost_zero
#resource_audit CostBridge.toCost_monotone_visits
#resource_audit CostBridge.toCost_monotone_mulAdds
#resource_audit CostBridge.toCost_monotone_bytes
#resource_audit CostBridge.toCost_monotone_allocs
#resource_audit CostBridge.toCost_monotone_loads
#resource_audit CostBridge.toCost_monotone_stores
#resource_audit CostBridge.toCost_monotone
#resource_audit CostBridge.loads_invisible_to_order
#resource_audit CostBridge.toCost_add
#resource_audit CostBridge.toCost_add_eq
#resource_audit CostBridge.Perf.add_monotone
#resource_audit CostBridge.tracePerf_rowDot
#resource_audit CostBridge.rowDotPerf_trace
#resource_audit CostBridge.runCost_eq_toCost
#resource_audit CostBridge.runCost_append
#resource_audit CostBridge.runCost_append_toCost_add
#resource_audit CostBridge.runCost_singleton
#resource_audit CostBridge.runCost_pair
#resource_audit CostBridge.runCost_mono
#resource_audit CostBridge.pad_fit
#resource_audit CostBridge.pad_visits
#resource_audit CostBridge.padded_success
#resource_audit CostBridge.exact_run
#resource_audit CostBridge.padded_run
#resource_audit CostBridge.observations_equal
#resource_audit CostBridge.exact_run_cost
#resource_audit CostBridge.padded_run_cost
#resource_audit CostBridge.exact_dominates_padded
#resource_audit CostBridge.exact_improves_padded
#resource_audit CostBridge.padding_costs_two_more_mulAdds
#resource_audit CostBridge.padding_costs_two_more_visits
#resource_audit CostBridge.composed_observations_equal
#resource_audit CostBridge.composed_exact_cost
#resource_audit CostBridge.composed_padded_cost
#resource_audit CostBridge.composed_exact_dominates_padded
#resource_audit CostBridge.composed_exact_improves_padded
#resource_audit CostBridge.composed_is_sum_of_singles

run_cmd Lean.logInfo "COST_BRIDGE_KERNEL_OK"