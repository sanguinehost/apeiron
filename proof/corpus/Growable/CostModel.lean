import ApeironWord

/-!
# The cost model of the hot read path

`AGENTS.md` §8 requires every function whose cost is not `O(1)` to name its recurrence and the
measure it is counted in.  This module makes that requirement machine-checked for the read path: it
defines a **cost carrier** (a vector of counted events), counts those events over the *generated*
bodies `table.probe.spill_read`, `table.probe.spill_read_tail_loop` and `table.Region.home`, and
bounds the counts.

The carrier, and the measure each field is counted in:

| field | counted event | measure |
|---|---|---|
| `coordLoads` | loads of a `Region.coords` word | coordinate compares |
| `slotLoads` | loads of a `Region.slots` payload | payload loads |
| `fullCompares` | `table.matches` invocations | `(Sigma, K)` compares |
| `increments` | checked `step + 1` | probe steps |
| `masks` | cycle-mask reductions `&&& mask` | probe steps |
| `homeCalls` | `Region::home` calls | calls |
| `allocations` | allocator calls | calls |
| `bytesCopied` | payload bytes copied | bytes |
| `lines` | distinct 64-byte lines of `Region.coords` opened | cache lines |

Every bound below is read out of the generated body by inversion of that body's own `Result`
equation; nothing is assumed about the body's shape.  Two quantities are *not* properties of the
body and are named as parameters: `LINE_BYTES = 64` is the machine's line width (obligation 11), and
`LANE_BYTES = 8` is `size_of::<u64>()`, the `src/table/probe.rs:186` constant read as an integer.

This module does not bound a control-byte glance: the tree has no control-byte array
(`src/table/` has none, and the extraction has no glance body), so no glance line-count theorem is
stated here.  Whether a glance is the shipped structure is an `anchor`; it is not evidenced here.

**The generated read's control flow (from `ApeironWord.lean`).**  At the head coordinate
`r.coords[r.home(place)]` the body tests `if found = table.COORD_EMPTY then ok none` and returns
`ok none` *directly* (`20528–20529`); that branch never reaches `table.probe.spill_read_tail`.  The
tail call at `20546` is inside the complementary `else` branch, guarded by `b1 :=
core.option.Option.is_none hit` (`20543`): it runs exactly when the head coordinate is live
(`found ≠ COORD_EMPTY`) but `hit = none`, i.e. `found ≠ wanted` or `table.matches` refused.  So an
empty-at-home read costs the head load only, and a live-nonmatching read additionally pays the tail.

Allocation and byte-copy counts are **not** claimed: counting the allocator constants of an
unfolded generated term needs a build-time occurrence counter that is not built here, and the
extraction models `Vec` as an immutable list, so an allocation is not observable in the model at
all.  `Resources.CoreCallTrace` carries a zero-allocation *certificate* for the prepared write path;
that is a premise, not a count over the read body.  See the closing gap list.
-/

namespace Apeiron.GrowableCost

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.unusedVariables false

/-! ## 1 · The cost carrier -/

/-- The counted events of one probe read.  Each field is a `Nat` in the measure named in the
module docstring's table; the carrier counts work performed, not elapsed time. -/
structure Cost where
  /-- Loads of a `Region.coords` coordinate word: one per generated body entry that passed the
  `step < bound` guard. -/
  coordLoads : Nat
  /-- Loads of a `Region.slots` payload: at most one per entry, and only on an entry whose
  coordinate equalled the wanted coordinate. -/
  slotLoads : Nat
  /-- `table.matches` invocations: at most one per payload load. -/
  fullCompares : Nat
  /-- Checked `step + 1` increments: probe steps taken. -/
  increments : Nat
  /-- Cycle-mask reductions `&&& mask`: one per entry that passed the guard. -/
  masks : Nat
  /-- `Region::home` calls. -/
  homeCalls : Nat
  /-- Allocator calls.  Not bounded here; see the module docstring. -/
  allocations : Nat
  /-- Payload bytes copied.  Not bounded here; see the module docstring. -/
  bytesCopied : Nat
  /-- Distinct 64-byte lines of `Region.coords` opened. -/
  lines : Nat
  deriving DecidableEq, Repr

/-- The pointwise order on costs. -/
def Cost.le (a b : Cost) : Prop :=
  a.coordLoads ≤ b.coordLoads ∧ a.slotLoads ≤ b.slotLoads ∧ a.fullCompares ≤ b.fullCompares ∧
  a.increments ≤ b.increments ∧ a.masks ≤ b.masks ∧ a.homeCalls ≤ b.homeCalls ∧
  a.allocations ≤ b.allocations ∧ a.bytesCopied ≤ b.bytesCopied ∧ a.lines ≤ b.lines

/-- A single `coordLoads` event, carrying its one mask and its one line. -/
def Cost.oneCoordLoad : Cost :=
  { coordLoads := 1, slotLoads := 0, fullCompares := 0, increments := 0, masks := 1,
    homeCalls := 0, allocations := 0, bytesCopied := 0, lines := 1 }

/-- The zero carrier: an entry that read nothing and charged nothing. -/
def Cost.zero : Cost :=
  { coordLoads := 0, slotLoads := 0, fullCompares := 0, increments := 0, masks := 0,
    homeCalls := 0, allocations := 0, bytesCopied := 0, lines := 0 }

/-- Pointwise addition of two carriers: counts are summed. -/
def Cost.add (a b : Cost) : Cost :=
  { coordLoads := a.coordLoads + b.coordLoads, slotLoads := a.slotLoads + b.slotLoads,
    fullCompares := a.fullCompares + b.fullCompares, increments := a.increments + b.increments,
    masks := a.masks + b.masks, homeCalls := a.homeCalls + b.homeCalls,
    allocations := a.allocations + b.allocations, bytesCopied := a.bytesCopied + b.bytesCopied,
    lines := a.lines + b.lines }

/-! ## 2 · Line geometry

`Region.coords` is `Vec U64`, one eight-byte coordinate word per slot, and `src/table/probe.rs:186`
fixes `LANES_PER_LINE = 64 / size_of::<u64>()`.  So a `Vec U64` coordinate array of `m` words
occupies at most `⌈m / 8⌉` 64-byte lines. -/

/-- Bytes per `Region.coords` slot: `size_of::<u64>()`, the `src/table/probe.rs:186` constant. -/
def LANE_BYTES : Nat := 8

/-- The machine's cache-line width.  A machine parameter, not a theorem about the generated body. -/
def LINE_BYTES : Nat := 64

/-- Coordinate slots per line: `64 / 8`. -/
def LANES_PER_LINE : Nat := LINE_BYTES / LANE_BYTES

/-- Bytes per lane of a hypothetical glance control array. -/
def CONTROL_BYTES : Nat := 1

/-- Lanes per line of a `u8` control array: `64 / 1`. -/
def LANES_PER_CONTROL_LINE : Nat := LINE_BYTES / CONTROL_BYTES

theorem LANES_PER_LINE_eq : LANES_PER_LINE = 8 := by decide
theorem LANES_PER_CONTROL_LINE_eq : LANES_PER_CONTROL_LINE = 64 := by decide

/-- The line index of coordinate slot `i`. -/
def coordLine (i : Nat) : Nat := i / LANES_PER_LINE

/-- The line index of control slot `i`. -/
def controlLine (i : Nat) : Nat := i / LANES_PER_CONTROL_LINE

/-- `⌈n / m⌉`. -/
def ceilDiv (n m : Nat) : Nat := (n + m - 1) / m

/-- Distinct coordinate lines opened by the consecutive slots `[lo, lo + n]`. -/
def openLinesInterval (lo n : Nat) : Nat :=
  (lo + n + LANES_PER_LINE) / LANES_PER_LINE - lo / LANES_PER_LINE

/-- An interval of `n` consecutive coordinate slots opens at most `⌈n / 8⌉ + 1` lines. -/
theorem openLinesInterval_le (lo n : Nat) :
    openLinesInterval lo n ≤ ceilDiv n LANES_PER_LINE + 1 := by
  unfold openLinesInterval ceilDiv
  norm_num [LANES_PER_LINE, LINE_BYTES, LANE_BYTES]
  have hA := Nat.div_add_mod (lo + n + 8) 8
  have hB := Nat.div_add_mod lo 8
  have hC := Nat.div_add_mod (n + 7) 8
  have hAr : (lo + n + 8) % 8 < 8 := Nat.mod_lt _ (by decide)
  have hBr : lo % 8 < 8 := Nat.mod_lt _ (by decide)
  have hCr : (n + 7) % 8 < 8 := Nat.mod_lt _ (by decide)
  omega

/-! ## 3 · Inverting the generated bodies -/

/-- `bind` inversion for a successful run: a successful `bind` has a successful left operand. -/
theorem bind_eq_ok {α β : Type} (m : Result α) (k : α → Result β) (y : β)
    (h : Std.bind m k = ok y) : ∃ a, m = ok a ∧ k a = ok y := by
  revert h
  refine Result.cases m ?_ ?_ ?_
  · intro a h
    rw [bind_ok] at h
    exact ⟨a, rfl, h⟩
  · intro i k' h
    rw [bind_vis] at h
    exact absurd h.symm (by simp)
  · intro h
    rw [bind_div] at h
    exact absurd h.symm (by simp)

/-- `Usize.max` on this target. -/
theorem usizeMax_eq : Usize.max = 2 ^ System.Platform.numBits - 1 := by
  simp [Usize.max, Usize.numBits]

/-- Every representable `Usize` value is below `2 ^ 64`. -/
theorem usizeMax_lt_pow : Usize.max < 2 ^ System.Platform.numBits := by
  rw [usizeMax_eq]
  have hp : 1 ≤ 2 ^ System.Platform.numBits := Nat.one_le_two_pow
  omega

/-- A `Nat` at or below `Usize.max` is representable as a `Usize`. -/
def uszOf (n : Nat) (h : n ≤ Usize.max) : Std.Usize :=
  ⟨n, by
    have hb : UScalarTy.Usize.numBits = System.Platform.numBits := rfl
    rw [hb]
    exact lt_of_le_of_lt h usizeMax_lt_pow⟩

/-- A `Usize` below `Usize.max + 1` is its own `Nat` value. -/
theorem uszOf_val (n : Nat) (h : n ≤ Usize.max) : (uszOf n h).val = n := rfl

/-! ## 5 · One generated tail-body entry, read off its own `Result` equation

Every fact below is recovered from the entry's own equation
`table.probe.spill_read_tail_loop.body … step = ok result` by `bind` inversion.  Nothing is assumed
about the body's shape. -/

/-- The generated collision tail read one coordinate word at `(home + step) &&& mask` on a
continuing entry: the guard held, so the entry's index sum, mask and coordinate load all succeeded,
its coordinate was live, and its checked increment produced `next`. -/
theorem tail_body_continued {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Std.Usize} {wanted : Std.U64} {step next : Std.Usize}
    (hbody : table.probe.spill_read_tail_loop.body keyRep r sigma i o key home bound mask wanted step
      = ok (ControlFlow.cont next)) :
    step < bound ∧
      ∃ (index slot : Std.Usize) (found : Std.U64),
        home + step = ok index ∧ slot = index &&& mask ∧
        alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords slot
          = ok found ∧
        found ≠ table.COORD_EMPTY ∧ step + 1#usize = ok next ∧ slot.val < r.coords.length := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  split at hbody
  · rename_i hguard
    simp only [Bind.bind] at hbody
    obtain ⟨index, hindex, hbody⟩ := bind_eq_ok (home + step) _ _ hbody
    simp only [lift, bind_ok] at hbody
    obtain ⟨found, hfound, hbody⟩ := bind_eq_ok
      (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords
        (index &&& mask)) _ _ hbody
    have hbounds : (index &&& mask).val < r.coords.length := by
      rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hfound
      cases hread : r.coords[(index &&& mask).val]? with
      | none => rw [hread] at hfound; simp at hfound
      | some payload =>
          have hget : payload = found := by
            rw [hread] at hfound
            simp only [] at hfound
            exact Result.ok.injEq.mp hfound
          subst hget
          exact (List.getElem?_eq_some_iff.mp hread).1
    split at hbody
    · rename_i hvacant
      exact absurd (Result.ok.injEq.mp hbody) (fun h => by cases h)
    · rename_i hlive
      split at hbody
      · rename_i hwanted
        obtain ⟨_, _, hbody⟩ := bind_eq_ok
          (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
            r.slots (index &&& mask)) _ _ hbody
        obtain ⟨_, _, hbody⟩ := bind_eq_ok
          (table.matches keyRep _ sigma key { place := i, direct_word := o }) _ _ hbody
        split at hbody
        · rename_i hmatch
          exact absurd (Result.ok.injEq.mp hbody) (fun h => by cases h)
        · have hinc' : step + 1#usize = ok next := by
            obtain ⟨n, hn, hk⟩ := bind_eq_ok (step + 1#usize) _ _ hbody
            have hn' : n = next := by
              exact ControlFlow.cont.inj (Result.ok.injEq.mp hk)
            rw [hn'] at hn
            exact hn
          refine ⟨hguard, ⟨index, index &&& mask, found, hindex, rfl, hfound, ?_, hinc', hbounds⟩⟩
          exact hlive
      · have hinc' : step + 1#usize = ok next := by
          obtain ⟨n, hn, hk⟩ := bind_eq_ok (step + 1#usize) _ _ hbody
          have hn' : n = next := by
            exact ControlFlow.cont.inj (Result.ok.injEq.mp hk)
          rw [hn'] at hn
          exact hn
        exact ⟨hguard, ⟨index, index &&& mask, found, hindex, rfl, hfound, hlive, hinc', hbounds⟩⟩
  · rename_i hguard
    exact absurd (Result.ok.injEq.mp hbody) (fun h => by cases h)

/-- An entry that returned at the `step < bound` guard read nothing and produced `none`: the
completed-cycle absence certificate, at zero cost. -/
theorem tail_body_done_at_guard {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Std.Usize} {wanted : Std.U64} {step : Std.Usize}
    {y : Option Std.Usize} (hguard : ¬ step < bound)
    (hbody : table.probe.spill_read_tail_loop.body keyRep r sigma i o key home bound mask wanted step
      = ok (ControlFlow.done y)) : y = none := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  rw [if_neg hguard] at hbody
  exact ControlFlow.done.inj (Result.ok.injEq.mp hbody).symm

/-- An entry that returned after passing the guard read exactly one coordinate word at
`(home + step) &&& mask`, and left through one of the body's three exits: the vacancy, a
non-matching coordinate, or the real stored matcher accepting the query. -/
theorem tail_body_done_after_load {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Std.Usize} {wanted : Std.U64} {step : Std.Usize} {y : Option Std.Usize}
    (hguard : step < bound)
    (hbody : table.probe.spill_read_tail_loop.body keyRep r sigma i o key home bound mask wanted step
      = ok (ControlFlow.done y)) :
    ∃ (index slot : Std.Usize) (found : Std.U64),
      home + step = ok index ∧ slot = index &&& mask ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords slot = ok found ∧
      slot.val < r.coords.length ∧
      ((found = table.COORD_EMPTY ∧ y = none) ∨
        (found ≠ table.COORD_EMPTY ∧ found = wanted ∧ slot.val < r.slots.length ∧
          y = slot)) := by
  unfold table.probe.spill_read_tail_loop.body at hbody
  rw [if_pos hguard] at hbody
  simp only [Bind.bind] at hbody
  obtain ⟨index, hindex, hbody⟩ := bind_eq_ok (home + step) _ _ hbody
  simp only [lift, bind_ok] at hbody
  obtain ⟨found, hfound, hbody⟩ := bind_eq_ok
    (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64) r.coords
      (index &&& mask)) _ _ hbody
  have hbounds : (index &&& mask).val < r.coords.length := by
    rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hfound
    cases hread : r.coords[(index &&& mask).val]? with
    | none => rw [hread] at hfound; simp at hfound
    | some payload =>
        have hget : payload = found := by
          rw [hread] at hfound
          simp only [] at hfound
          exact Result.ok.injEq.mp hfound
        subst hget
        exact (List.getElem?_eq_some_iff.mp hread).1
  split at hbody
  · rename_i hvacant
    exact ⟨index, index &&& mask, found, hindex, rfl, hfound, hbounds,
      Or.inl ⟨hvacant, (ControlFlow.done.inj (Result.ok.injEq.mp hbody)).symm⟩⟩
  · rename_i hlive
    split at hbody
    · rename_i hwanted
      obtain ⟨payload, hpayload, hbody⟩ := bind_eq_ok
        (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (table.SpillSlot R V))
          r.slots (index &&& mask)) _ _ hbody
      have hpaylb : (index &&& mask).val < r.slots.length := by
        rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hpayload
        cases hread : r.slots[(index &&& mask).val]? with
        | none => rw [hread] at hpayload; simp at hpayload
        | some p =>
            have hget : p = payload := by
              rw [hread] at hpayload
              simp only [] at hpayload
              exact Result.ok.injEq.mp hpayload
            subst hget
            exact (List.getElem?_eq_some_iff.mp hread).1
      obtain ⟨matched, hmatched, hbody⟩ := bind_eq_ok
        (table.matches keyRep payload sigma key { place := i, direct_word := o }) _ _ hbody
      split at hbody
      · rename_i hyes
        refine ⟨index, index &&& mask, found, hindex, rfl, hfound, hbounds, ?_⟩
        right
        exact ⟨hlive, hwanted, hpaylb, (ControlFlow.done.inj (Result.ok.injEq.mp hbody)).symm⟩
      · rename_i hno
        obtain ⟨n, hn, hb⟩ := bind_eq_ok (step + 1#usize) _ _ hbody
        have hz : ControlFlow.cont n = ControlFlow.done y := Result.ok.injEq.mp hb
        exact absurd hz (fun h => by cases h)
    · rename_i hother
      obtain ⟨n, hn, hb⟩ := bind_eq_ok (step + 1#usize) _ _ hbody
      have hz : ControlFlow.cont n = ControlFlow.done y := Result.ok.injEq.mp hb
      exact absurd hz (fun h => by cases h)

/-! ## 6 · The iteration bound

`table.probe.spill_read_tail_loop` is the generated `loop` over
`table.probe.spill_read_tail_loop.body`.  `LoopRun` is a finite run of such a `loop`: the chain of
`cont` transitions ending in a `done`.  `loop_run` recovers a `LoopRun` from the equation
`loop body x = ok y` under a strictly decreasing `Nat` measure, unfolding the `partial_fixpoint`
equation `loop.eq_def` exactly once per step.

`LoopRun.guarded` counts the trace entries whose `step < bound` guard held.  That is the
coordinate-load count of the read, and it is read off the generated body rather than assumed:
`tail_body_continued` gives every continuing entry exactly one coordinate load and
`tail_body_done_after_load` gives every terminating entry past the guard exactly one, while
`tail_body_done_at_guard` gives a guard-failing entry none.  `guarded_le` is the iteration bound: a
trace from step `x` has at most `bound - x` guarded entries, because `tail_loop_increments` advances
the step by exactly one on every `cont` and an entry at or above `bound` contributes nothing.

The headline is stated on the trace *computed from the run*, not on an existential witness, so the
count is a function of the run equation and the bound is a claim about the generated body:

| theorem | statement | grade |
|---|---|---|
| `tail_loop_measure_decreases` | every `cont` of the generated tail body decreases `bound.val + 1 - step.val` | `proved` |
| `spill_read_tail_cost` | `((tailRun h).tailCost bound).coordLoads + 1 ≤ bound.val` | `proved` |
| `spill_read_absent_home_cost` | an empty head coordinate makes the generated read return `ok none` without entering the tail loop | `proved` |
| `loopRun_of_run` | every continuing entry raises the counter by one and every entry charges ≤ 1 coordinate load ⇒ a loop returning `ok v` peels `budget − counter a` entries charging ≤ that | `proved` |
| `spill_read_tail_loopRun_bound` | that generic bound at the generated tail loop, from `start.val = 1` | `proved` |

`spill_read_tail_cost` bounds the tail loop.  It does not by itself bound `table.probe.spill_read`,
whose head adds one further coordinate load at `home`; see the closing gap list.
-/

/-- A finite run of a generated `loop`: the chain of `cont` transitions from `x` ending in the
`done` that returned `y`.  The proof arguments are the body's own `Result` equations, one per
invocation, so a trace is a witness about the *generated* body. -/
inductive LoopRun {α : Type} {β : Type} (body : α → Result (ControlFlow α β)) : α → β → Type where
  | stop {x : α} {y : β} (h : body x = ok (ControlFlow.done y)) : LoopRun body x y
  | next {x x' : α} {y : β} (h : body x = ok (ControlFlow.cont x')) (hr : LoopRun body x' y) :
      LoopRun body x y

/-- `loop_run`: a run equation for a generated `loop` has a finite trace.  `hdec` is the strictly
decreasing measure, `bound + 1 - s.val` for the generated tail body; the strong recursion on the
measure bounds the number of unfoldings of `loop.eq_def` to the number of executed iterations. -/
noncomputable def loop_run {α β : Type} (body : α → Result (ControlFlow α β)) (measure : α → Nat)
    (hdec : ∀ a a', body a = ok (ControlFlow.cont a') → measure a' < measure a)
    (x : α) (y : β) (h : loop body x = ok y) : LoopRun body x y := by
  have aux : ∀ n, ∀ (x : α) (y : β), measure x ≤ n → loop body x = ok y → LoopRun body x y := by
    intro n
    refine Nat.strongRecOn n ?_
    intro n ih x y hn hloop
    rw [loop.eq_def] at hloop
    cases hB : body x with
    | ret a =>
        rw [hB] at hloop
        simp only [bind_ok] at hloop
        cases a with
        | done z =>
            simp only [] at hloop
            have hzy : z = y := Result.ok_injective hloop
            rw [hzy] at hB
            exact LoopRun.stop hB
        | cont x' =>
            exact LoopRun.next hB
              (ih (measure x') (lt_of_lt_of_le (hdec x x' hB) hn) x' y le_rfl hloop)
    | vis e k =>
        rw [hB, bind_vis] at hloop
        exact absurd hloop (by simp)
    | div =>
        rw [hB, bind_div] at hloop
        exact absurd hloop (by simp)
  exact aux (measure x) x y le_rfl h

namespace LoopRun

/-- The number of `cont` transitions of a trace: the number of checked `step + 1` increments the
trace performed. -/
def contSteps : ∀ {α : Type} {β : Type} {body : α → Result (ControlFlow α β)} {x : α} {y : β},
    LoopRun body x y → Nat
  | _, _, _, _, _, .stop _ => 0
  | _, _, _, _, _, .next _ r => r.contSteps + 1

/-- The number of entries of a trace whose `step < bound` guard held.  For the generated tail body
this is its coordinate-load count: `tail_body_continued` gives every `cont` entry exactly one, and
`tail_body_done_after_load` and `tail_body_done_at_guard` give the terminating entry one iff its
guard held. -/
def guarded (bound : Usize) :
    ∀ {β : Type} {body : Usize → Result (ControlFlow Usize β)} {x : Usize} {y : β},
    LoopRun body x y → Nat
  | _, _, x, _, .stop _ => if x.val < bound.val then 1 else 0
  | _, _, x, _, .next _ r => (if x.val < bound.val then 1 else 0) + r.guarded bound

/-- The cost carrier of a `Usize`-stepped loop run guarded by `step < bound`: one coordinate load,
one mask reduction and one opened line per guarded entry; one checked increment per `cont`
transition; at most one payload load and one `table.matches` invocation per guarded entry, counted
at that upper bound; no `Region::home` call.  Allocations and copied bytes are zero because the
extraction models `Vec` as an immutable list, so neither is observable in the model at all (module
docstring): this is a property of the model, not a claim that the Rust body allocates nothing. -/
def tailCost {β : Type} {body : Usize → Result (ControlFlow Usize β)} {s : Usize} {y : β}
    (run : LoopRun body s y) (bound : Usize) : Cost where
  coordLoads := run.guarded bound
  slotLoads := run.guarded bound
  fullCompares := run.guarded bound
  increments := run.contSteps
  masks := run.guarded bound
  homeCalls := 0
  allocations := 0
  bytesCopied := 0
  lines := run.guarded bound

end LoopRun

/-- The iteration bound, in its sharp form: a trace of a `Usize`-stepped body that advances the step
by exactly one on every `cont` has at most `bound - x` entries that passed the `step < bound` guard.
No assumption is made that a continuing entry passed the guard; an entry at or above `bound`
contributes nothing, and the remaining entries advance one toward `bound`. -/
theorem guarded_le {β : Type} {body : Usize → Result (ControlFlow Usize β)}
    (hinc : ∀ a a', body a = ok (ControlFlow.cont a') → a'.val = a.val + 1)
    {x : Usize} {y : β} (run : LoopRun body x y) (bound : Usize) :
    run.guarded bound ≤ bound.val - x.val := by
  induction run with
  | stop h =>
      simp only [LoopRun.guarded]
      split <;> omega
  | next h r ih =>
      simp only [LoopRun.guarded]
      have hx' := hinc _ _ h
      split <;> omega

/-- A `cont` transition of the generated tail body has already passed the guard, so every increment
is one of the guarded entries: the checked-increment count never exceeds the coordinate-load count. -/
theorem contSteps_le_guarded {β : Type} {body : Usize → Result (ControlFlow Usize β)}
    {bound : Usize} {x : Usize} {y : β} (run : LoopRun body x y)
    (hguard : ∀ a a', body a = ok (ControlFlow.cont a') → a.val < bound.val) :
    run.contSteps ≤ run.guarded bound := by
  induction run with
  | stop h => simp only [LoopRun.contSteps, LoopRun.guarded]; split <;> omega
  | next h r ih =>
      simp only [LoopRun.contSteps, LoopRun.guarded, if_pos (hguard _ _ h)]
      omega

/-- `x + y = ok z` yields `z.val = x.val + y.val`: the checked addition succeeded, so the sum is
exact.  This is the reading of the generated `step + 1#usize`. -/
theorem add_val_of_ok {x y z : Usize} (h : x + y = ok z) : z.val = x.val + y.val := by
  have h' := UScalar.add_equiv (x := x) (y := y)
  rw [h, Result.match.ok] at h'
  exact h'.2.1

/-- The generated tail body, as an explicit function of the probe step. -/
def spillTailBody {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H) (r : table.Region R V)
    (sigma : cell.Sigma) (i : Std.U64) (o : Option Std.U32) (key : K)
    (home bound mask : Usize) (wanted : Std.U64) :
    Usize → Result (ControlFlow Usize (Option Usize)) :=
  fun step => table.probe.spill_read_tail_loop.body keyRep r sigma i o key home bound mask wanted step

/-- A continuing entry of the generated tail body advances the probe step by exactly one.  Read off
`tail_body_continued`, whose inversion of the body's own `Result` equation returns the checked
`step + 1#usize = ok next`. -/
theorem tail_loop_increments {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Usize} {wanted : Std.U64} {a a' : Usize}
    (hbody : spillTailBody keyRep r sigma i o key home bound mask wanted a
      = ok (ControlFlow.cont a')) : a'.val = a.val + 1 := by
  obtain ⟨_, _, _, _, _, _, _, _, hinc, _⟩ := tail_body_continued hbody
  exact add_val_of_ok hinc

/-- A continuing entry of the generated tail body passed its `step < bound` guard. -/
theorem tail_loop_guard_held {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Usize} {wanted : Std.U64} {a a' : Usize}
    (hbody : spillTailBody keyRep r sigma i o key home bound mask wanted a
      = ok (ControlFlow.cont a')) : a.val < bound.val := by
  obtain ⟨hguard, _, _, _, _, _, _, _, _, _⟩ := tail_body_continued hbody
  exact (UScalar.lt_equiv a bound).mp hguard

/-- The strictly decreasing measure for the generated tail loop's `loop_run`: the checked increment
moves the step one past `a`, and `a` was below `bound`, so `bound.val + 1 - a'.val` is smaller. -/
theorem tail_loop_measure_decreases {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Usize} {wanted : Std.U64} {a a' : Usize}
    (hbody : spillTailBody keyRep r sigma i o key home bound mask wanted a
      = ok (ControlFlow.cont a')) :
    (bound.val + 1 - a'.val) < (bound.val + 1 - a.val) := by
  have ha' := tail_loop_increments hbody
  have halt := tail_loop_guard_held hbody
  omega

/-- The trace of a generated `spill_read_tail` run from the shipped start offset `1#usize`.  The
trace is computed from the run equation, so its `guarded` count is a function of the run. -/
noncomputable def tailRun {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Usize} {wanted : Std.U64} {y : Option Usize}
    (h : table.probe.spill_read_tail_loop keyRep r sigma i o key home bound mask wanted 1#usize
      = ok y) : LoopRun (spillTailBody keyRep r sigma i o key home bound mask wanted) 1#usize y :=
  loop_run (spillTailBody keyRep r sigma i o key home bound mask wanted) (fun s =>
      bound.val + 1 - s.val) (fun a a' hb => tail_loop_measure_decreases hb) 1#usize y h

/-- **The iteration bound.**  Every successful run of the generated `spill_read_tail_loop` from the
shipped start offset `1#usize` loads at most `bound.val - 1` coordinate words: the coordinate loads
are exactly the trace entries whose `step < bound` guard held, and a trace from step `1` has at
most `bound - 1` of them.  Equivalently `coordLoads + 1 ≤ bound.val`, so the tail sweep never
performs more than one coordinate load per unit of capacity.  Every other counted field of the same
run is bounded by the same figure except `increments`, which is the exact `cont` count and never
exceeds the guarded count. -/
theorem spill_read_tail_cost {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home bound mask : Usize} {wanted : Std.U64} {y : Option Usize}
    (h : table.probe.spill_read_tail_loop keyRep r sigma i o key home bound mask wanted 1#usize
      = ok y) :
    ((tailRun h).tailCost bound).coordLoads ≤ bound.val ∧
    (bound.val ≠ 0 → ((tailRun h).tailCost bound).coordLoads + 1 ≤ bound.val) ∧
    ((tailRun h).tailCost bound).slotLoads ≤ bound.val ∧
    ((tailRun h).tailCost bound).fullCompares ≤ bound.val ∧
    ((tailRun h).tailCost bound).masks ≤ bound.val ∧
    ((tailRun h).tailCost bound).lines ≤ bound.val ∧
    ((tailRun h).tailCost bound).increments ≤ bound.val ∧
    ((tailRun h).tailCost bound).homeCalls = 0 ∧
    ((tailRun h).tailCost bound).allocations = 0 ∧
    ((tailRun h).tailCost bound).bytesCopied = 0 := by
  have hinc : ∀ a a', spillTailBody keyRep r sigma i o key home bound mask wanted a
      = ok (ControlFlow.cont a') → a'.val = a.val + 1 := fun _ _ hb => tail_loop_increments hb
  have hguard : ∀ a a', spillTailBody keyRep r sigma i o key home bound mask wanted a
      = ok (ControlFlow.cont a') → a.val < bound.val := fun _ _ hb => tail_loop_guard_held hb
  have hg := guarded_le hinc (tailRun h) bound
  have hg1 : (tailRun h).guarded bound ≤ bound.val - 1 := by simpa using hg
  have hcs := contSteps_le_guarded (tailRun h) hguard
  have hcs1 : (tailRun h).contSteps ≤ bound.val - 1 := Nat.le_trans hcs hg1
  have hsub : (tailRun h).guarded bound ≤ bound.val := Nat.le_trans hg1 (Nat.sub_le _ _)
  have hsharp : bound.val ≠ 0 → (tailRun h).guarded bound + 1 ≤ bound.val := by
    intro hnz
    have hpos : 1 ≤ bound.val := by omega
    have hle := Nat.add_le_add_right hg1 1
    rwa [Nat.sub_add_cancel hpos] at hle
  dsimp only [LoopRun.tailCost]
  exact ⟨hsub, hsharp, hsub, hsub, hsub, hsub, Nat.le_trans hcs1 (Nat.sub_le _ _), rfl, rfl, rfl⟩

/-! ## 6b · The generic counted peel (a loop-general second route)

`LoopRun` above counts a trace recovered from a successful run equation.  The `loopRun` below is the
complementary route: a computable peel of the generated `partial_fixpoint` `Std.loop`, charging each
descended entry.  `loopRun_of_run` is the loop-general iteration bound: if every continuing entry
raises a `counter` by one and every entry charges at most one coordinate load, then a loop that
returned `ok v` is reproduced by peeling `budget - counter a` entries, charging at most that many
coordinate loads.  `spill_read_tail_loopRun_bound` applies it to the generated tail loop. -/

/-- The carrier of one tail-loop entry whose `step < bound` guard held: one coordinate load, one
cycle-mask reduction, one checked increment, one line.  Two of those are upper bounds read off
the body: the entry that breaks out performs no increment, and consecutive slots share a line.
Every other field is `0`, because a `Region.slots` load and a `table.matches` call happen only
on the entry whose coordinate equalled the wanted coordinate, and the generated loop calls
neither `Region::home` nor the allocator. -/
def Cost.tailEntry : Cost :=
  { coordLoads := 1, slotLoads := 0, fullCompares := 0, increments := 1, masks := 1,
    homeCalls := 0, allocations := 0, bytesCopied := 0, lines := 1 }

/-- The carrier of the generated tail-loop entry at `step`: `Cost.tailEntry` when the entry
passed its `step < bound` guard and so reached the `Region.coords` load, `Cost.zero` otherwise.
Read off the body by `tail_body_continued`, whose guard hypothesis is this entry's `step < bound`. -/
def tailCharge (cap step : Std.Usize) : Cost :=
  if step.val < cap.val then Cost.tailEntry else Cost.zero

/-- The counted peel of a loop.  `loopRun body charge x n` descends at most `n` entries of
`body` from `x`, charging each entry it descends through, and reports the value the loop
produced together with the carrier it charged.  `n = 0` descends nothing and reports
`Std.loop body x` uncounted; an entry that breaks out reports the loop and charges that entry. -/
def loopRun {α β : Type} (body : α → Result (ControlFlow α β)) (charge : α → Cost) (x : α) :
    Nat → Result β × Cost
  | 0 => (Std.loop body x, Cost.zero)
  | n + 1 =>
      match Result.match (body x) with
      | MatchResult.ok (.cont y) => let (w, c) := loopRun body charge y n
                                    (w, Cost.add (charge x) c)
      | _ => (Std.loop body x, charge x)

/-- **Peeling one entry of a `Std.loop`.**  This is the equation of the `partial_fixpoint`
`Std.loop`, read through `bind_ok`: a breaking entry is the loop's result, and a continuing
entry makes the loop the loop at its continuation. -/
theorem loop_hbreak {α β : Type} (body : α → Result (ControlFlow α β)) (x : α) (v : β)
    (h : body x = ok (ControlFlow.done v)) : Std.loop body x = ok v := by
  rw [Std.loop.eq_def, h, bind_ok]

/-- **Peeling one entry of a `Std.loop`.**  A continuing entry at `x` carrying state `y` leaves
the loop's value equal to the loop's value at `y`. -/
theorem loop_peel {α β : Type} (body : α → Result (ControlFlow α β)) (x y : α)
    (h : body x = ok (ControlFlow.cont y)) : Std.loop body x = Std.loop body y := by
  rw [Std.loop.eq_def, h, bind_ok]

/-- **The counted peel's continuation branch.**  One charged entry at `x`, then the `n`-entry
peel at its continuation. -/
theorem loopRun_cont {α β : Type} (body : α → Result (ControlFlow α β)) (charge : α → Cost)
    (x y : α) (n : Nat) (h : body x = ok (ControlFlow.cont y)) :
    (loopRun body charge x (n + 1)).1 = (loopRun body charge y n).1 ∧
      (loopRun body charge x (n + 1)).2 = Cost.add (charge x) (loopRun body charge y n).2 := by
  simp [loopRun, h]

/-- **The counted peel's exit branch.**  A breaking entry charges itself and reports the loop. -/
theorem loopRun_break {α β : Type} (body : α → Result (ControlFlow α β)) (charge : α → Cost)
    (x : α) (n : Nat) (v : β) (h : body x = ok (ControlFlow.done v)) :
    (loopRun body charge x (n + 1)).1 = Std.loop body x ∧
      (loopRun body charge x (n + 1)).2 = charge x := by
  simp [loopRun, h]

/-- **The generic iteration bound.**  If every continuing entry of a loop raises `counter` by
one and each entry is charged at most one coordinate load, then a loop that returned `ok v` at
`a` is reproduced by peeling `budget - counter a` entries, and that peel charges at most
`budget - counter a` coordinate loads. -/
theorem loopRun_of_run {α β : Type} (body : α → Result (ControlFlow α β)) (charge : α → Cost)
    (counter : α → Nat) (budget : Nat)
    (hinc : ∀ a b, body a = ok (ControlFlow.cont b) → counter b = counter a + 1)
    (hch : ∀ a, (charge a).coordLoads ≤ 1) :
    ∀ (rem : Nat) (a : α), budget - counter a = rem → ∀ (v : β), Std.loop body a = ok v →
      (loopRun body charge a rem).1 = ok v ∧ (loopRun body charge a rem).2.coordLoads ≤ rem := by
  intro rem
  induction rem with
  | zero =>
      intro a _ v h
      show (loopRun body charge a 0).1 = ok v ∧ (loopRun body charge a 0).2.coordLoads ≤ 0
      rw [loopRun]
      exact ⟨h, Nat.zero_le _⟩
  | succ m ih =>
      intro a ha v h
      show (loopRun body charge a (m + 1)).1 = ok v ∧
        (loopRun body charge a (m + 1)).2.coordLoads ≤ m + 1
      cases hB : body a with
      | ret c =>
          cases c with
          | cont y =>
              have hcy : counter y = counter a + 1 := hinc a y hB
              have hy : Std.loop body y = ok v := by
                rw [loop_peel body a y hB] at h
                exact h
              obtain ⟨h1, h2⟩ := ih y (by omega) v hy
              obtain ⟨hcA, hcB⟩ := loopRun_cont body charge a y m hB
              rw [hcA, hcB]
              have hcha := hch a
              refine ⟨h1, ?_⟩
              simp only [Cost.add]
              omega
          | done v' =>
              have hbreak : Std.loop body a = ok v' := loop_hbreak body a v' hB
              have hv : v' = v := by
                rw [hbreak] at h
                exact Result.ok.injEq.mp h
              subst hv
              obtain ⟨hbA, hbB⟩ := loopRun_break body charge a m v' hB
              rw [hbA, hbB]
              have hcha := hch a
              exact ⟨h, by omega⟩
      | vis i k =>
          rw [Std.loop.eq_def] at h
          rw [hB] at h
          simp at h
      | div =>
          rw [Std.loop.eq_def] at h
          rw [hB] at h
          simp at h

/-- **The iteration bound of the generated spill read's collision tail loop.**  A tail loop
entered at `start = 1` that returned `ok y` charged at most `cap.val - 1` coordinate loads, so
`coordLoads + 1 ≤ cap.val`: one per entry that passed the entry's own `step < cap` guard, over a
chain that starts at `1` and therefore ends no later than `cap`.  The entries are counted by
`loopRun` at `tailCharge`, and the chain's length by `loopRun_of_run`. -/
theorem spill_read_tail_loopRun_bound {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {i : Std.U64} {o : Option Std.U32} {key : K}
    {home cap mask : Std.Usize} {wanted : Std.U64} {start : Std.Usize}
    {y : Option Std.Usize} (hcap : 1 ≤ cap.val) (hstart : start.val = 1)
    (hloop : table.probe.spill_read_tail_loop keyRep r sigma i o key home cap mask wanted start
      = ok y) :
    (loopRun (fun s => table.probe.spill_read_tail_loop.body keyRep r sigma i o key home cap mask
        wanted s) (tailCharge cap) start (cap.val - start.val)).2.coordLoads + 1
      ≤ cap.val := by
  have hinc : ∀ (a b : Std.Usize),
      table.probe.spill_read_tail_loop.body keyRep r sigma i o key home cap mask wanted a
        = ok (ControlFlow.cont b) → b.val = a.val + 1 := by
    intro a b hb
    obtain ⟨_, index, slot, found, hindex, hslot, hvindex, hne, hinc, hlb⟩ :=
      tail_body_continued (next := b) hb
    exact add_val_of_ok hinc
  have hch : ∀ a : Std.Usize, (tailCharge cap a).coordLoads ≤ 1 := by
    intro a
    by_cases h : a.val < cap.val
    · simp only [tailCharge, if_pos h, Cost.tailEntry]
      exact Nat.le_refl 1
    · simp only [tailCharge, if_neg h, Cost.zero]
      exact Nat.zero_le _
  have hloop' : Std.loop (fun s => table.probe.spill_read_tail_loop.body keyRep r sigma i o key
      home cap mask wanted s) start = ok y := by
    unfold table.probe.spill_read_tail_loop at hloop
    simpa only [Bind.bind, bind_ok] using hloop
  obtain ⟨_, hle⟩ := loopRun_of_run
    (fun s => table.probe.spill_read_tail_loop.body keyRep r sigma i o key home cap mask wanted s)
    (tailCharge cap) (fun a => a.val) cap.val hinc hch (cap.val - start.val) start
    (by rw [hstart]) y hloop'
  omega

/-- **The absent-at-home branch carrier (`model`).**  The events the module's cost model charges the
`spill_read` branch that finds the wanted coordinate empty at `home`: one coordinate load, one
`Region::home` call and one cache line, and nothing else.  This is a carrier definition, not a count
recovered from the generated body; the body's branch identity is `spill_read_absent_home_cost`. -/
def absentHomeCost : Cost :=
  { coordLoads := 1, slotLoads := 0, fullCompares := 0, increments := 0, masks := 0,
    homeCalls := 1, allocations := 0, bytesCopied := 0, lines := 1 }

/-- **The absent-at-home cost.**  Forward evaluation of the generated read: when the region is
non-empty, `valid_n` passes, `Region::home` yields `home`, `coord_of` yields `wanted` and the head
coordinate probe at `home` reads `COORD_EMPTY`, the generated `table.probe.spill_read` returns
`ok none`.  The evaluation below never reaches the `table.probe.spill_read_tail` call, which is the
argument that no probe iteration happened.  What this *branch* is charged is `absentHomeCost`, whose
`coordLoads` is 1; this is a `model` figure for the branch, not a claim about every absent read.  The
complementary path — the home coordinate live but not matching the wanted key — sets `hit = none`,
`b1 := Option.is_none hit = true`, and does enter `table.probe.spill_read_tail`
(`ApeironWord.lean:20543,20546`), so an absent read on that path pays the tail cost too.

`hne` is the length guard in the body's own `Bool` form
(`Vec.len r.slots != 0#usize`), so the hypothesis is the guard the generated code tests, not a
restatement of it. -/
theorem spill_read_absent_home_cost {K H R V : Type} {keyRep : table.keyrep.KeyRep R K H}
    {r : table.Region R V} {sigma : cell.Sigma} {address : table.TableAddress} {key : K}
    {n home : Usize} {wanted found : Std.U64}
    (hne : (alloc.vec.Vec.len r.slots != 0#usize) = true)
    (hn : table.valid_n n = ok true)
    (hhome : table.Region.home r address.place = ok home)
    (hwanted : table.Region.coord_of R V address.place = ok wanted)
    (hfound : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.U64)
      r.coords home = ok found)
    (hempty : found = table.COORD_EMPTY) :
    table.probe.spill_read keyRep r sigma address key n = ok none := by
  unfold table.probe.spill_read
  dsimp only
  rw [if_pos hne, hn]
  simp only [Bind.bind, bind_ok]
  rw [if_pos (by decide)]
  rw [hhome]
  simp only [bind_ok]
  rw [hwanted]
  simp only [bind_ok]
  rw [hfound]
  simp only [bind_ok]
  rw [if_pos hempty]

/-- **The absent-at-home carrier's figures (`model`).**  A check of the *definition* of
`absentHomeCost`; every equality holds by `rfl`/`decide`.  This fixes the model's carrier for the
branch, it is not a count recovered from the generated body.  The branch's identity — the proved
content — is `spill_read_absent_home_cost`. -/
theorem absentHomeCost_reading :
    absentHomeCost.coordLoads = 1 ∧ absentHomeCost.slotLoads = 0 ∧
    absentHomeCost.fullCompares = 0 ∧ absentHomeCost.increments = 0 ∧
    absentHomeCost.masks = 0 ∧ absentHomeCost.homeCalls = 1 ∧
    absentHomeCost.allocations = 0 ∧ absentHomeCost.bytesCopied = 0 ∧
    absentHomeCost.lines = 1 := by
  decide

end Apeiron.GrowableCost
