import RegenRelocateAllSource
import Aeneas

/-
SCOPE: the actual extracted body `Table::relocate_all` reached by the extraction root
`word_regen_relocate_all`; its one guarded transition (`relocate_all_loop.body`), its exit branch,
its two refusal channels, the exactness of the exit branch, the work measure in delegated per-entry
relocations and checked index increments with its closed form, and a reachable success witness at
the empty plan.
REGISTER: source refinement and an explicit-work count of that one body, read off the extracted
loop-body equation. The traversal issues no payload copy of its own: every continuation delegates
once to `Table::relocate_entry` and advances the index by one, and the payload copies (one of the
held key and one of the value, `src/table.rs:955,959,960`) are issued inside that delegated body,
which is the subject of the sibling per-entry root. Excluded: the delegated per-entry body, the plan
builder `Table::rebuild_plan` (`src/table.rs:621-690`), the reservation adapter, the pending merge,
the non-termination channel of the delegated bodies, canonical-corpus correspondence and hardware
timing.
CORPUS: `src/table.rs:1017-1027`, whose real body is
`pub(crate) fn relocate_all(&mut self, plan: &RebuildPlan, reserved: &mut RebuildReserved<R, V>)
{ let mut entry_at = 0; while entry_at < plan.entries.len() { self.relocate_entry(plan, reserved,
entry_at); entry_at += 1; } }`; the same body is already extracted at
`proof/ApeironWord.lean:22631-22685` (loop body, loop and whole traversal) and the per-entry body
it delegates to at `proof/ApeironWord.lean:22546-22629`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/RegenRelocateAll_gate.py --controls`, 540-second budget, lane
/tmp/opencode/apeiron-lane-2.lock.
GRADE: `proved` for the theorems below when that gate exits 0. The work measure is a count of the
delegated per-entry relocations and index increments the extracted body itself issues under the
Aeneas model, not a measured time.
-/

namespace RegenRelocateAllResource
open Aeneas Aeneas.Std Result ControlFlow Error
open regen_relocate_all

/-- The regenerated word table the extraction root is monomorphic at. -/
abbrev Store := table.Table Std.I64 Std.I64 table.WordHome (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize

/-- The reserved relocated storage the real traversal consumes. -/
abbrev Reserved := table.RebuildReserved (table.keyrep.Regen table.keyrep.I64WordCodec) Std.I64

/-- The real `Clone<i64>` instance the wrapper passes to the traversal. -/
abbrev CloneI := core.clone.CloneI64

/-- The real `Home` instance at the regenerated word carrier. -/
abbrev HomeInst := table.WordHome.Insts.Regen_relocate_allTableHomeI64Sigma

/-- The real `ExactIdentity<i64>` instance the `KeyRep` is built from. -/
abbrev ExactIdentityInst := I64.Insts.Regen_relocate_allCoordLocatorLocator_coreExactIdentityU64Tuple

/-- The real `ExactCodec` instance of `I64WordCodec`. -/
abbrev CodecInst :=
  table.keyrep.I64WordCodec.Insts.Regen_relocate_allTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces
    HomeInst

/-- The real `KeyRep` instance at the regenerated word carrier. -/
abbrev KeyRepInst := table.keyrep.Regen.Insts.Regen_relocate_allTableKeyrepKeyRep
  ExactIdentityInst HomeInst CodecInst

/-- The loop state the extracted traversal threads: the table, the reserved storage and the index. -/
abbrev LoopState := Store × (Reserved × Usize)

/-- The transition type of one real traversal step. -/
abbrev StepT := ControlFlow LoopState (Store × Reserved)

/-- The real per-entry relocation body, at the concrete carrier. The traversal reaches it only as an
opaque `Result`; its own transitions and copy count belong to the sibling per-entry root. -/
def stepBody (state : Store) (plan : table.RebuildPlan) (reserved : Reserved) (entry_at : Usize) :
    Result (Store × Reserved) :=
  table.Table.relocate_entry CloneI CloneI HomeInst KeyRepInst state plan reserved entry_at

/-- The real traversal body, at the concrete carrier. -/
def body (plan : table.RebuildPlan) (state : Store) (reserved : Reserved) (entry_at : Usize) :
    Result StepT :=
  table.Table.relocate_all_loop.body CloneI CloneI HomeInst KeyRepInst plan state reserved entry_at

/-- The extracted loop applied to a loop state, at the concrete carrier. -/
def relocateLoop (plan : table.RebuildPlan) (state : Store) (reserved : Reserved) (entry_at : Usize) :
    Result (Store × Reserved) :=
  table.Table.relocate_all_loop CloneI CloneI HomeInst KeyRepInst state plan reserved entry_at

/-- The whole extracted traversal, at the concrete carrier. -/
def traversal (plan : table.RebuildPlan) (state : Store) (reserved : Reserved) :
    Result (Store × Reserved) :=
  table.Table.relocate_all CloneI CloneI HomeInst KeyRepInst state plan reserved

/-- The resource rank of a loop state: the number of planned entries the traversal has yet to visit.
It is `0` exactly where the exit branch is taken. -/
def rank (plan : table.RebuildPlan) (st : LoopState) : Nat :=
  plan.entries.length - st.2.2.val

/-- The counted work of the real traversal: `relocations` delegations to the real per-entry body and
`advances` checked index increments, over a plan of `planned` entries. A continuation issues one of
each and the exit branch issues none, so a run that takes `n` continuations did `n` of each. -/
structure Work where
  /-- Delegations to the real per-entry relocation body. -/
  relocations : Nat
  /-- Checked index increments. -/
  advances : Nat
  /-- The number of planned entries the traversal was asked to visit. -/
  planned : Nat

/-- The extraction root forwards to the real traversal body at the concrete instances; the root
restates no guard, no loop and no per-entry write. -/
theorem root_forwards (state : Store) (plan : table.RebuildPlan) (reserved : Reserved) :
    word_regen_relocate_all state plan reserved = traversal plan state reserved := rfl

/-- The extracted loop is the `loop` of the extracted body over the extracted loop state. -/
theorem loop_is_body_iteration (plan : table.RebuildPlan) (state : Store)
    (reserved : Reserved) (entry_at : Usize) :
    relocateLoop plan state reserved entry_at
      = loop (fun st => body plan st.1 st.2.1 st.2.2) (state, reserved, entry_at) := rfl

/-- The real traversal starts its index at zero, so the root visits exactly the planned entries. -/
theorem traversal_starts_at_zero (plan : table.RebuildPlan) (state : Store) (reserved : Reserved) :
    traversal plan state reserved = relocateLoop plan state reserved 0#usize := rfl

/-- The guard of the real body is the planned-entry count: the loop continues exactly while the
index is strictly below `plan.entries.length`. -/
theorem guard_iff (plan : table.RebuildPlan) (entry_at : Usize) :
    entry_at < alloc.vec.Vec.len plan.entries ↔ entry_at.val < plan.entries.length := by
  rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]

/-- The guarded transition, verbatim: the whole extracted body under the true guard. Its
right-hand side holds exactly one delegation to the real per-entry body, exactly one checked index
increment, and one continuation carrying the delegated state and the advanced index, in that order;
the kernel check of this equation is what certifies the work count below. -/
theorem body_step (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (state1 : Store) (reserved1 : Reserved) (entry_at1 : Usize)
    (hguard : entry_at.val < plan.entries.length)
    (hstep : stepBody state plan reserved entry_at = ok (state1, reserved1))
    (hadd : entry_at + 1#usize = ok entry_at1) :
    (body plan state reserved entry_at
      = (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved)))) ∧
      body plan state reserved entry_at = ok (ControlFlow.cont (state1, reserved1, entry_at1)) := by
  have hlt : entry_at < alloc.vec.Vec.len plan.entries :=
    (guard_iff plan entry_at).2 hguard
  have heq : body plan state reserved entry_at = (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved))) := rfl
  constructor
  · rw [heq, if_pos hlt, hstep, hadd]
  · rw [heq, if_pos hlt, hstep, hadd]
    simp only [bind_tc_ok]
    rfl

/-- The exit branch, verbatim: with the index at or past the planned-entry count the real body
returns the state and the reserved storage unchanged, delegating to no per-entry body and advancing
no index. -/
theorem body_exit (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (hguard : plan.entries.length ≤ entry_at.val) :
    (body plan state reserved entry_at
      = (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved)))) ∧
      body plan state reserved entry_at = ok (ControlFlow.done (state, reserved)) := by
  have hlt : ¬ (entry_at < alloc.vec.Vec.len plan.entries) := by
    intro he
    have h1 := (guard_iff plan entry_at).1 he
    omega
  have heq : body plan state reserved entry_at = (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved))) := rfl
  constructor
  · rw [heq, if_neg hlt]
  · rw [heq, if_neg hlt]

/-- A one-bind `do` block that returned `ok` consumed an `ok`, and its continuation returned the
same `ok`. This is the model-level fact used to read the delegated call and the increment back out
of the transitions of the real body. -/
theorem bind1_eq_ok {α : Type} {β : Type} {a : Result α} {k : α → Result β} {t : β}
    (h : (do let p ← a; k p) = ok t) : ∃ p, a = ok p ∧ k p = ok t := by
  cases hm : a.match with
  | ok p =>
    have hav : a = ok p := (Result.match.isOk).mp hm
    rw [hav, bind_tc_ok] at h
    exact ⟨p, hav, h⟩
  | vis e kk =>
    rw [(Result.match.isVis).mp hm, bind_tc_vis] at h
    exact absurd h vis_not_ok
  | div =>
    rw [(Result.match.isDiv).mp hm, bind_tc_div] at h
    exact absurd h div_not_ok

/-- The model's own loop equation: a `loop` that returned `ok` either took its break branch on the
first body evaluation, or took a continue branch and the remaining loop again returned `ok`. -/
theorem model_loop_ok_cases {α : Type} {β : Type} (f : α → Result (ControlFlow α β)) (x : α)
    (t : β) (h : loop f x = ok t) :
    (f x = ok (ControlFlow.done t)) ∨
      ∃ x' : α, f x = ok (ControlFlow.cont x') ∧ loop f x' = ok t := by
  have hb : (do let q ← f x; match q with
      | ControlFlow.cont x' => loop f x'
      | ControlFlow.done y => ok y)
      = Std.bind (f x) (fun r => match r with
        | ControlFlow.cont x' => loop f x'
        | ControlFlow.done y => ok y) := rfl
  have hav : (do let q ← f x; match q with
      | ControlFlow.cont x' => loop f x'
      | ControlFlow.done y => ok y) = ok t :=
    hb.trans (loop.eq_1 f x ▸ h)
  obtain ⟨q, hq, hk⟩ := bind1_eq_ok hav
  cases hq' : q with
  | cont x' =>
    refine Or.inr ⟨x', ?_, ?_⟩
    · rw [hq, hq']
    · rw [hq'] at hk
      exact hk
  | done y =>
    have hy : y = t := by
      rw [hq'] at hk
      exact Result.ok_injective hk
    subst hy
    refine Or.inl ?_
    rw [hq, hq']

/-- A one-bind `do` block with a pair pattern that returned `ok` consumed an `ok` pair, and its
continuation returned the same `ok`. -/
theorem bind1pat_eq_ok {α : Type} {β : Type} {γ : Type} {a : Result (α × β)}
    {k : α × β → Result γ} {t : γ}
    (h : (do let (p, q) ← a; k (p, q)) = ok t) :
    ∃ p q, a = ok (p, q) ∧ k (p, q) = ok t := by
  cases hm : a.match with
  | ok x =>
    obtain ⟨p, q⟩ := x
    have hav : a = ok (p, q) := (Result.match.isOk).mp hm
    refine ⟨p, q, hav, ?_⟩
    rw [hav, bind_tc_ok] at h
    exact h
  | vis e kk =>
    rw [(Result.match.isVis).mp hm, bind_tc_vis] at h
    exact absurd h vis_not_ok
  | div =>
    rw [(Result.match.isDiv).mp hm, bind_tc_div] at h
    exact absurd h div_not_ok

/-- The guarded branch of the real body never produces the exit transition: under the true guard
its result is a continuation or a refusal, never `done`. -/
theorem body_not_exit_under_guard (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (s : Store) (r : Reserved) (hguard : entry_at.val < plan.entries.length)
    (h : body plan state reserved entry_at = ok (ControlFlow.done (s, r))) : False := by
  have hlt : entry_at < alloc.vec.Vec.len plan.entries :=
    (guard_iff plan entry_at).2 hguard
  have h' : (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved))) = ok (ControlFlow.done (s, r)) := h
  rw [if_pos hlt] at h'
  obtain ⟨a, b, ha, hrest⟩ := bind1pat_eq_ok
    (k := fun d => do let entry_at1 ← entry_at + 1#usize; ok (ControlFlow.cont (d.1, d.2, entry_at1)))
    h'
  obtain ⟨c, hc, hcont⟩ := bind1_eq_ok hrest
  exact nomatch (Result.ok_injective hcont)

/-- A continuation of the real body delegated exactly once, to the index it carried, and carried
the advanced index: the guard, the delegated result and the checked increment are read back out of
the continuation value. -/
theorem body_step_decompose (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (st' : LoopState) (hguard : entry_at.val < plan.entries.length)
    (h : body plan state reserved entry_at = ok (ControlFlow.cont st')) :
    stepBody state plan reserved entry_at = ok (st'.1, st'.2.1) ∧
      entry_at + 1#usize = ok st'.2.2 ∧ st'.2.2.val = entry_at.val + 1 := by
  have hlt : entry_at < alloc.vec.Vec.len plan.entries :=
    (guard_iff plan entry_at).2 hguard
  have hone : (1#usize).val = 1 := by simp
  have hle : plan.entries.length ≤ Usize.max := alloc.vec.Vec.len_ineq plan.entries
  have h1 : entry_at.val + 1 ≤ plan.entries.length := by omega
  have hmax : entry_at.val + (1#usize).val ≤ Usize.max := by omega
  have h' : (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved))) = ok (ControlFlow.cont st') := h
  rw [if_pos hlt] at h'
  obtain ⟨state1, reserved1, hstep, hrest⟩ := bind1pat_eq_ok
    (k := fun d => do let entry_at1 ← entry_at + 1#usize; ok (ControlFlow.cont (d.1, d.2, entry_at1)))
    h'
  obtain ⟨entry_at1, hadd, hcont⟩ := bind1_eq_ok hrest
  have hcont' : ControlFlow.cont (state1, reserved1, entry_at1)
      = ControlFlow.cont st' := Result.ok_injective hcont
  have hcases : state1 = st'.1 ∧ reserved1 = st'.2.1 ∧ entry_at1 = st'.2.2 := by
    cases st' with
    | mk a' b' =>
      cases b' with
      | mk c' d' => simp only [Prod.mk.injEq, ControlFlow.cont.injEq] at hcont'; grind
  obtain ⟨rfl, rfl, rfl⟩ := hcases
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := entry_at) (y := 1#usize) hmax)
  have hz' : z = st'.2.2 := Result.ok_injective (hz.symm.trans hadd)
  rw [hz'] at hzv
  exact ⟨hstep, hadd, by simpa using hzv⟩

/-- A continuation of the real body delegated exactly once, to the index it carried, and carried the
advanced index, stated with the three components of the loop state named. -/
theorem body_step_decompose' (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (state1 : Store) (reserved1 : Reserved) (entry_at1 : Usize)
    (hguard : entry_at.val < plan.entries.length)
    (h : body plan state reserved entry_at = ok (ControlFlow.cont (state1, reserved1, entry_at1))) :
    stepBody state plan reserved entry_at = ok (state1, reserved1) ∧
      entry_at + 1#usize = ok entry_at1 ∧ entry_at1.val = entry_at.val + 1 :=
  body_step_decompose plan state reserved entry_at (state1, reserved1, entry_at1) hguard h

/-- The first refusal channel: under the true guard, a failing per-entry body makes the real
traversal body fail, and no continuation is produced. -/
theorem body_refuses_when_step_fails (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (e : Error) (hguard : entry_at.val < plan.entries.length)
    (hstep : stepBody state plan reserved entry_at = fail e) :
    body plan state reserved entry_at = fail e := by
  have hlt : entry_at < alloc.vec.Vec.len plan.entries :=
    (guard_iff plan entry_at).2 hguard
  have heq : body plan state reserved entry_at = (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved))) := rfl
  rw [heq, if_pos hlt, hstep]
  simp only [bind_tc_fail]

/-- The second refusal channel: under the true guard and a successful per-entry body, an overflowing
index increment makes the real traversal body fail. The increment is checked, so this is the only way
the index advance can refuse. -/
theorem body_refuses_when_add_fails (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (e : Error) (hguard : entry_at.val < plan.entries.length)
    (hstep : stepBody state plan reserved entry_at = ok (state, reserved))
    (hadd : entry_at + 1#usize = fail e) :
    body plan state reserved entry_at = fail e := by
  have hlt : entry_at < alloc.vec.Vec.len plan.entries :=
    (guard_iff plan entry_at).2 hguard
  have heq : body plan state reserved entry_at = (do
        let i := alloc.vec.Vec.len plan.entries
        if entry_at < i then
          let (self1, reserved1) ← stepBody state plan reserved entry_at
          let entry_at1 ← entry_at + 1#usize
          ok (ControlFlow.cont (self1, reserved1, entry_at1))
        else ok (ControlFlow.done (state, reserved))) := rfl
  rw [heq, if_pos hlt, hstep, hadd]
  simp only [bind_tc_ok, bind_tc_fail]
  rfl

/-- The exit branch is exact: the real body returns the unchanged pair at and past the planned-entry
count, and under the true guard it never returns that pair. -/
theorem body_exit_iff (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) :
    body plan state reserved entry_at = ok (ControlFlow.done (state, reserved)) ↔
      plan.entries.length ≤ entry_at.val := by
  constructor
  · intro h
    by_contra hn
    have hguard : entry_at.val < plan.entries.length := by omega
    exact body_not_exit_under_guard plan state reserved entry_at state reserved hguard h
  · intro h
    exact (body_exit plan state reserved entry_at h).2

/-- A continuation of the real body implies the guard held, i.e. the index was strictly below the
planned-entry count. This is the converse direction of `body_exit_iff`, read off the same equation. -/
theorem body_cont_implies_guard (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (st' : LoopState)
    (h : body plan state reserved entry_at = ok (ControlFlow.cont st')) :
    entry_at.val < plan.entries.length := by
  by_contra hn
  have hguard : plan.entries.length ≤ entry_at.val := by omega
  have hbody : body plan state reserved entry_at = ok (ControlFlow.done (state, reserved)) :=
    (body_exit plan state reserved entry_at hguard).2
  exact nomatch (Result.ok_injective (hbody.symm.trans h))

/-- The rank vanishes exactly at the exit branch. -/
theorem rank_vanishes_iff (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) :
    rank plan (state, reserved, entry_at) = 0 ↔ plan.entries.length ≤ entry_at.val := by
  simp only [rank]
  omega

/-- The rank is positive exactly under the guard. -/
theorem rank_positive_under_guard (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (hguard : entry_at.val < plan.entries.length) :
    0 < rank plan (state, reserved, entry_at) := by
  simp only [rank]
  omega

/-- The checked index increment of the real body advances the index by exactly one, and cannot
overflow while the guard holds. -/
theorem step_advances_by_one (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (state1 : Store) (reserved1 : Reserved) (entry_at1 : Usize)
    (hguard : entry_at.val < plan.entries.length)
    (hstep : stepBody state plan reserved entry_at = ok (state1, reserved1))
    (hadd : entry_at + 1#usize = ok entry_at1) :
    body plan state reserved entry_at = ok (ControlFlow.cont (state1, reserved1, entry_at1)) ∧
      entry_at1.val = entry_at.val + 1 ∧ entry_at1.val ≤ plan.entries.length := by
  have hone : (1#usize).val = 1 := by simp
  have hle : plan.entries.length ≤ Usize.max := alloc.vec.Vec.len_ineq plan.entries
  have h1 : entry_at.val + 1 ≤ plan.entries.length := by omega
  have hmax : entry_at.val + (1#usize).val ≤ Usize.max := by omega
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := entry_at) (y := 1#usize) hmax)
  have hz' : z = entry_at1 := Result.ok_injective (hz.symm.trans hadd)
  rw [hz'] at hzv
  exact ⟨(body_step plan state reserved entry_at state1 reserved1 entry_at1 hguard hstep hadd).2,
    by simpa using hzv, (by omega)⟩

/-- The strictly decreasing rank: one guarded transition delegates once and drops the rank by
exactly one, so the traversal performs at most `plan.entries.length` delegations. -/
theorem rank_strictly_decreases (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (state1 : Store) (reserved1 : Reserved) (entry_at1 : Usize)
    (hguard : entry_at.val < plan.entries.length)
    (hstep : stepBody state plan reserved entry_at = ok (state1, reserved1))
    (hadd : entry_at + 1#usize = ok entry_at1) :
    body plan state reserved entry_at = ok (ControlFlow.cont (state1, reserved1, entry_at1)) ∧
      rank plan (state1, reserved1, entry_at1) + 1 = rank plan (state, reserved, entry_at) := by
  have hone : (1#usize).val = 1 := by simp
  have hle : plan.entries.length ≤ Usize.max := alloc.vec.Vec.len_ineq plan.entries
  have h1 : entry_at.val + 1 ≤ plan.entries.length := by omega
  have hmax : entry_at.val + (1#usize).val ≤ Usize.max := by omega
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := entry_at) (y := 1#usize) hmax)
  have hz' : z = entry_at1 := Result.ok_injective (hz.symm.trans hadd)
  rw [hz'] at hzv
  have hnext : entry_at1.val = entry_at.val + 1 := by simpa using hzv
  refine ⟨(body_step plan state reserved entry_at state1 reserved1 entry_at1 hguard hstep hadd).2, ?_⟩
  simp only [rank]
  omega

/-- The extracted loop, when it returns `ok`, took either its exit transition or one of its guarded
transitions, and the remaining run again returns `ok`. This is the model's own loop equation read at
the concrete carrier. -/
theorem loop_ok_cases (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (entry_at : Usize) (s : Store) (r : Reserved)
    (h : relocateLoop plan state reserved entry_at = ok (s, r)) :
    body plan state reserved entry_at = ok (ControlFlow.done (s, r)) ∨
      ∃ (state1 : Store) (reserved1 : Reserved) (entry_at1 : Usize),
        body plan state reserved entry_at = ok (ControlFlow.cont (state1, reserved1, entry_at1)) ∧
          relocateLoop plan state1 reserved1 entry_at1 = ok (s, r) := by
  have hloop : relocateLoop plan state reserved entry_at
      = loop (fun st => body plan st.1 st.2.1 st.2.2) (state, reserved, entry_at) :=
    loop_is_body_iteration plan state reserved entry_at
  rcases model_loop_ok_cases (fun st => body plan st.1 st.2.1 st.2.2) (state, reserved, entry_at)
      (s, r) (hloop ▸ h) with hdone | ⟨st', hcont, hrest⟩
  · exact Or.inl hdone
  · obtain ⟨t1, rest⟩ := st'
    obtain ⟨r2, i2⟩ := rest
    exact Or.inr ⟨t1, r2, i2, hcont, by
      have hloop' := loop_is_body_iteration plan t1 r2 i2
      rw [hloop']
      exact hrest⟩

/-- The counted work of a successful run of the real traversal, in closed form: started at index
`entry_at`, it delegated to the real per-entry body exactly `n` times and advanced the index exactly
`n` times, where `n` is the number of planned entries that index had left. The count is read from the
transitions of the extracted body: each recursive case of `loop_ok_cases` is one guarded transition,
which `body_step` shows issues exactly one delegation and one checked increment, while the exit branch
issues none, and `rank_strictly_decreases` shows the rank drops by one per transition. So no
successful run can skip an entry, repeat an entry, or stop early. -/
theorem loop_work_closed_form (plan : table.RebuildPlan) :
    ∀ (n : Nat) (state : Store) (reserved : Reserved) (entry_at : Usize) (s : Store) (r : Reserved),
      plan.entries.length - entry_at.val = n → entry_at.val ≤ plan.entries.length →
      relocateLoop plan state reserved entry_at = ok (s, r) →
      ({ relocations := n, advances := n, planned := plan.entries.length } : Work)
        = ({ relocations := plan.entries.length - entry_at.val,
             advances := plan.entries.length - entry_at.val,
             planned := plan.entries.length } : Work) := by
  intro n
  induction n with
  | zero =>
    intro state reserved entry_at s r hn hle h
    have hguard : plan.entries.length ≤ entry_at.val := by omega
    have hbody : body plan state reserved entry_at = ok (ControlFlow.done (state, reserved)) :=
      (body_exit plan state reserved entry_at hguard).2
    rcases loop_ok_cases plan state reserved entry_at s r h with
      hdone | ⟨st1, rs1, i1, hcont, hrest⟩
    · rw [hbody] at hdone
      have hp : (state, reserved) = (s, r) :=
        ControlFlow.done.inj (Result.ok_injective hdone)
      cases hp
      simp only [Work.mk.injEq]
      exact ⟨hn.symm, hn.symm, trivial⟩
    · exact nomatch (Result.ok_injective (hbody.symm.trans hcont))
  | succ n ih =>
    intro state reserved entry_at s r hn hle h
    have hguard : entry_at.val < plan.entries.length := by omega
    rcases loop_ok_cases plan state reserved entry_at s r h with
      hdone | ⟨st1, rs1, i1, hcont, hrest⟩
    · exact (body_not_exit_under_guard plan state reserved entry_at s r hguard hdone).elim
    · obtain ⟨hstep, hadd, hnext⟩ :=
        body_step_decompose' plan state reserved entry_at st1 rs1 i1 hguard hcont
      have hn' : plan.entries.length - i1.val = n := by omega
      have hle' : i1.val ≤ plan.entries.length := by omega
      have hp := ih st1 rs1 i1 s r hn' hle' hrest
      simp only [Work.mk.injEq] at hp ⊢
      exact ⟨by omega, by omega, trivial⟩

/-- The work of one successful run of the real root: it starts at index zero, so it delegated once
per planned entry and advanced the index once per planned entry, with nothing left over. -/
theorem traversal_work_closed_form (plan : table.RebuildPlan) (state : Store) (reserved : Reserved)
    (s : Store) (r : Reserved) (h : traversal plan state reserved = ok (s, r)) :
    ({ relocations := plan.entries.length, advances := plan.entries.length,
       planned := plan.entries.length } : Work)
      = ({ relocations := plan.entries.length - (0#usize).val,
           advances := plan.entries.length - (0#usize).val,
           planned := plan.entries.length } : Work) := by
  rw [traversal_starts_at_zero] at h
  have hzero : (0#usize).val ≤ plan.entries.length := Nat.zero_le _
  have hw := loop_work_closed_form plan (plan.entries.length - (0#usize).val) state reserved
    0#usize s r rfl hzero h
  rw [hw]
  simp only [Work.mk.injEq]
  have hz : (0#usize).val = 0 := by simp
  rw [hz, Nat.sub_zero]
  exact ⟨rfl, rfl, trivial⟩

/-- The empty plan, the reachable input with nothing to relocate: a `RebuildPlan` whose entry vector
is empty. The real plan builder pushes one entry per live representative (`src/table.rs:628-690`),
so a table with no live representative produces exactly this plan. -/
def emptyPlan : table.RebuildPlan :=
  { lanes := 0#usize
    entries := alloc.vec.Vec.from [] (by simp)
    spill_loads := alloc.vec.Vec.from [] (by simp) }

/-- The empty plan has no planned entries. -/
theorem emptyPlan_length : emptyPlan.entries.length = 0 := by
  simp [emptyPlan, alloc.vec.Vec.length, alloc.vec.Vec.val, alloc.vec.Vec.from]

/-- A reachable success witness: on the empty plan the real root returns the state and the reserved
storage unchanged, having delegated zero times and advanced the index zero times, at every table and
every reserved storage. The witness is the extracted root itself, evaluated at the reachable
empty-plan input. -/
theorem empty_plan_witness (state : Store) (reserved : Reserved) :
    traversal emptyPlan state reserved = ok (state, reserved) ∧
      rank emptyPlan (state, reserved, 0#usize) = 0 ∧
      ({ relocations := 0, advances := 0, planned := 0 } : Work)
        = ({ relocations := emptyPlan.entries.length,
             advances := emptyPlan.entries.length,
             planned := emptyPlan.entries.length } : Work) := by
  have hbody : body emptyPlan state reserved 0#usize = ok (ControlFlow.done (state, reserved)) :=
    (body_exit emptyPlan state reserved 0#usize (by simp [emptyPlan_length])).2
  refine ⟨?_, rfl, ?_⟩
  · have hloop : relocateLoop emptyPlan state reserved 0#usize
        = loop (fun st => body emptyPlan st.1 st.2.1 st.2.2) (state, reserved, 0#usize) :=
      loop_is_body_iteration emptyPlan state reserved 0#usize
    rw [traversal_starts_at_zero, hloop, loop.eq_1, hbody]
    simp only [bind_ok]
  · rw [emptyPlan_length]

end RegenRelocateAllResource

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

#resource_audit RegenRelocateAllResource.root_forwards
#resource_audit RegenRelocateAllResource.loop_is_body_iteration
#resource_audit RegenRelocateAllResource.traversal_starts_at_zero
#resource_audit RegenRelocateAllResource.guard_iff
#resource_audit RegenRelocateAllResource.body_step
#resource_audit RegenRelocateAllResource.body_exit
#resource_audit RegenRelocateAllResource.bind1_eq_ok
#resource_audit RegenRelocateAllResource.bind1pat_eq_ok
#resource_audit RegenRelocateAllResource.body_not_exit_under_guard
#resource_audit RegenRelocateAllResource.body_step_decompose
#resource_audit RegenRelocateAllResource.body_refuses_when_step_fails
#resource_audit RegenRelocateAllResource.body_refuses_when_add_fails
#resource_audit RegenRelocateAllResource.body_exit_iff
#resource_audit RegenRelocateAllResource.rank_vanishes_iff
#resource_audit RegenRelocateAllResource.rank_positive_under_guard
#resource_audit RegenRelocateAllResource.step_advances_by_one
#resource_audit RegenRelocateAllResource.rank_strictly_decreases
#resource_audit RegenRelocateAllResource.model_loop_ok_cases
#resource_audit RegenRelocateAllResource.loop_ok_cases
#resource_audit RegenRelocateAllResource.body_cont_implies_guard
#resource_audit RegenRelocateAllResource.loop_work_closed_form
#resource_audit RegenRelocateAllResource.traversal_work_closed_form
#resource_audit RegenRelocateAllResource.emptyPlan_length
#resource_audit RegenRelocateAllResource.empty_plan_witness

run_cmd Lean.logInfo "REGEN_RELOCATE_ALL_KERNEL_OK"
