import PolicySource
import Aeneas

/- SCOPE: the actual extracted bounded policy decision core `policy_source.policy.decide.decide`:
   its atom-evaluation transition, the guard-loop transition and both rule-pass exit transitions with
   their guards, and the atom-evaluation work measure with its strict bound.
   REGISTER: source resource and refinement of the extracted body only. No canonical-corpus
   correspondence (`Apeiron.Policy.decide` is a separate bridge), and no cycle, GPU or hardware
   claim: every count below is counted in rule reads or atom evaluations under the stated
   slice-length premises. CORPUS: `src/policy/decide.rs`, read through the extraction root
   `proof/extract/lib.rs` `word_policy_decide`; the extracted definitions are
   `proof/resource/PolicySource.lean`. REFERENCE:
   /home/socol/Workspace/apeiron/proof/ApeironWord.lean.
   GATE: `python3 proof/resource/Policy_gate.py --controls`, 540 s wall-clock budget, lane-3 lock,
   MemoryMax=12G, MemorySwapMax=0, LEAN_NUM_THREADS=1.
   GRADE: `proved` only when that gate succeeds, on the recorded source and tool digests.

   RESIDUAL (open, not proved here, not weakened away):
   - `guard_eval` range check and per-guard bound: `guard_eval_range`, `guard_eval_work`
     (`n ≤ rule.atom_len.val ∧ n < rule.atom_len.val + 1`). Blocked on the `Slice.len` normal form:
     the extracted `Slice.len atoms` appears both as `.val` and as `atoms.length`, and `omega`
     treats the two as distinct atoms.
   - pass-1 effect transitions `decide_loop0_body_permit` / `_forbid` / `_ask` / `_inert` and the
     iteration-count theorems `decide_loop0_work`, `decide_loop1_work` (exactly
     `(Slice.len rules).val - i.val` iterations, strictly below one more). Same normal-form block,
     plus the `spec` postcondition of `loop.spec_decr_nat` being an inductive with one
     constructor, so each body case needs `rw [hbody]; simp only [WP.spec_ok]`.
   - the top-level `decide` branches `Verdict.Deny Cause.NoPermit`, `Verdict.Deny (Cause.Forbid id)`,
     `Verdict.Allow`, `Verdict.Attenuate`, `Verdict.Ask`, and the reachable success and refusal
     witnesses. Not attempted in this slice; the straight-line epilogue of `decide` is the next
     bounded task after the two items above. -/

namespace PolicyResource
open Aeneas Aeneas.Std Result
open policy_source.policy.decide (Op Effect Atom Rule Tri Cause Verdict)

theorem lift_ok {α : Type} (x : α) : lift x = ok x := rfl

theorem cast_slot_val (x : U32) : (UScalar.cast .Usize x).val = x.val := by
  simp

/-- The payload of a bounded right shift, when the shift count is in range. -/
theorem shr_ok (a : U64) (n : U32) (h : n.val < 64) : ∃ z : U64, a >>> n = ok z := by
  obtain ⟨z, hz, _⟩ := WP.spec_imp_exists (U64.ShiftRight_spec a n h)
  exact ⟨z, hz⟩

/-- The payload of an in-bounds slice read. -/
theorem index_ok {α : Type} (s : Slice α) (i : Usize) (h : i.val < s.length) :
    ∃ x : α, s.index_usize i = ok x := by
  obtain ⟨x, hx, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec s i h)
  exact ⟨x, hx⟩

/-- An in-bounds index and its successor stay below `Usize.max`, which is the bound the extracted
    additions are checked against. -/
theorem idx_next_max {α : Type} (s : Slice α) {i : Usize} (h : i.val < (Slice.len s).val) :
    i.val + 1 ≤ Usize.max := by
  have hp : (Slice.len s).val ≤ Usize.max := by
    rw [Slice.len_val]
    exact Slice.property s
  omega

/-- A loop whose next state is `done y` returns `y`. -/
theorem loop_done {α : Type} {β : Type} (body : α → Result (ControlFlow α β)) (x : α) (y : β)
    (h : body x = ok (ControlFlow.done y)) : loop body x = ok y := by
  rw [loop.eq_1, h]
  show Std.bind (ok (ControlFlow.done y)) (fun r : ControlFlow α β =>
    match r with
    | ControlFlow.cont x => loop body x
    | ControlFlow.done x => ok x) = ok y
  simp

/-- A loop whose next state is `cont x'` continues from `x'`. -/
theorem loop_cont {α : Type} {β : Type} (body : α → Result (ControlFlow α β)) (x x' : α)
    (h : body x = ok (ControlFlow.cont x')) : loop body x = loop body x' := by
  rw [loop.eq_1, h]
  show Std.bind (ok (ControlFlow.cont x')) (fun r : ControlFlow α β =>
    match r with
    | ControlFlow.cont x => loop body x
    | ControlFlow.done x => ok x) = loop body x'
  simp

/-- The extracted `Tri` equality test against `Tri.T`: the discriminant comparison. -/
theorem tri_eq_t (g : Tri) :
    policy_source.policy.decide.Tri.Insts.CoreCmpPartialEqTri.eq g Tri.T
      = match g with
        | Tri.T => ok true
        | Tri.F => ok false
        | Tri.U => ok false := by
  cases g <;>
    simp [policy_source.policy.decide.Tri.Insts.CoreCmpPartialEqTri.eq,
      policy_source.policy.decide.Tri.read_discriminant]

/-- The extracted `Tri` inequality test against `Tri.F`, as the default `ne` method reads it. -/
theorem tri_ne_f (g : Tri) :
    core.cmp.PartialEq.ne.trait_default
        policy_source.policy.decide.Tri.Insts.CoreCmpPartialEqTri g Tri.F
      = match g with
        | Tri.T => ok true
        | Tri.F => ok false
        | Tri.U => ok true := by
  cases g <;>
    simp [core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default,
      policy_source.policy.decide.Tri.Insts.CoreCmpPartialEqTri.eq,
      policy_source.policy.decide.Tri.read_discriminant]

/-- The extracted `Effect` equality test against `Effect.Forbid`. -/
theorem effect_eq_forbid (e : Effect) :
    policy_source.policy.decide.Effect.Insts.CoreCmpPartialEqEffect.eq e Effect.Forbid
      = match e with
        | Effect.Permit => ok false
        | Effect.Forbid => ok true
        | Effect.Ask => ok false := by
  cases e <;>
    simp [policy_source.policy.decide.Effect.Insts.CoreCmpPartialEqEffect.eq,
      policy_source.policy.decide.Effect.read_discriminant]

/-- The smallest-id fold the extracted `min_id` computes: `none` is its unit, and otherwise the
    smaller of the two ids is kept. One comparison, no iteration. -/
def MinId (best : Option U32) (id : U32) : Option U32 :=
  match best with
  | none => some id
  | some b => some (if id.val < b.val then id else b)

theorem min_id_spec (best : Option U32) (id : U32) :
    policy_source.policy.decide.min_id best id = ok (MinId best id) := by
  cases best with
  | none => simp [policy_source.policy.decide.min_id, MinId]
  | some b =>
    by_cases hlt : id.val < b.val <;>
      simp [policy_source.policy.decide.min_id, MinId, hlt]

/-- One atom evaluation always returns a value: each of its three unknown guards and its single
    comparison ends in `ok`. Its work is one cast, one shift, one mask and one comparison. -/
theorem atom_eval_ok (atom : Atom) (facts : Slice U64) (known : U64) :
    ∃ t : Tri, policy_source.policy.decide.atom_eval atom facts known = ok t := by
  by_cases hge : 64 ≤ atom.slot.val
  · exact ⟨Tri.U, by simp [policy_source.policy.decide.atom_eval, hge, lift_ok, bind_tc_ok]⟩
  · by_cases hout : facts.length ≤ atom.slot.val
    · exact ⟨Tri.U, by
        simp [policy_source.policy.decide.atom_eval, hge, hout, lift_ok, bind_tc_ok]⟩
    · obtain ⟨i1, hi1⟩ := shr_ok known atom.slot (by simpa [U64.numBits] using hge)
      by_cases hz : i1 &&& 1#u64 = 0#u64
      · exact ⟨Tri.U, by
          simp [policy_source.policy.decide.atom_eval, hge, hi1, hz, lift_ok, bind_tc_ok]⟩
      · obtain ⟨fact, hfact'⟩ :=
          index_ok facts (UScalar.cast .Usize atom.slot) (by rw [cast_slot_val]; omega)
        unfold policy_source.policy.decide.atom_eval
        cases atom.op with
        | Eq =>
          by_cases hc : fact = atom.value
          · exact ⟨Tri.T, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
          · exact ⟨Tri.F, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
        | Le =>
          by_cases hc : fact.val ≤ atom.value.val
          · exact ⟨Tri.T, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
          · exact ⟨Tri.F, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
        | Ge =>
          by_cases hc : atom.value.val ≤ fact.val
          · exact ⟨Tri.T, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
          · exact ⟨Tri.F, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
        | All =>
          by_cases hc : (fact &&& atom.value) = atom.value
          · exact ⟨Tri.T, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
          · exact ⟨Tri.F, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
        | Any =>
          by_cases hc : fact.val &&& atom.value.val = 0
          · exact ⟨Tri.F, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩
          · exact ⟨Tri.T, by
              simp [hge, hout, hi1, hz, hfact', hc, lift_ok, bind_tc_ok]⟩

/-- A slot index of 64 or more is unknown, and the first guard fires before any fact read. -/
theorem atom_eval_slot_unknown (atom : Atom) (facts : Slice U64) (known : U64)
    (h : 64 ≤ atom.slot.val) :
    policy_source.policy.decide.atom_eval atom facts known = ok Tri.U := by
  simp [policy_source.policy.decide.atom_eval, h, lift_ok, bind_tc_ok]

/-- One guarded iteration of the guard loop reads the single atom at `start + i`, evaluates it
    once, and either continues with the atom index advanced by exactly one — keeping the
    accumulated Kleene value or lowering it to `U` — or exits at once with `F`. Its work is one
    `atom_eval` call, one index addition and one index increment. -/
theorem guard_body_step (atoms : Slice Atom) (facts : Slice U64) (known : U64)
    (start len : Usize) (acc : Tri) (i : Usize)
    (hi : i.val < len.val) (hrange : start.val + len.val ≤ (Slice.len atoms).val) :
    ∃ r : ControlFlow (Tri × Usize) Tri,
      policy_source.policy.decide.guard_eval_loop.body atoms facts known start len acc i = ok r ∧
      ((∃ acc' : Tri, ∃ i' : Usize,
          r = ControlFlow.cont (acc', i') ∧ i'.val = i.val + 1 ∧
            (acc' = Tri.U ∨ acc' = acc)) ∨ r = ControlFlow.done Tri.F) := by
  unfold policy_source.policy.decide.guard_eval_loop.body
  rw [if_pos ((UScalar.lt_equiv i len).2 hi)]
  have hbound : (Slice.len atoms).val ≤ Usize.max := by
    rw [Slice.len_val]
    exact Slice.property atoms
  have hmax : start.val + i.val ≤ Usize.max := by omega
  obtain ⟨ix1, hix1, hix1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := start) (y := i) hmax)
  have hread : ix1.val < (Slice.len atoms).val := by omega
  obtain ⟨ax, hax, haxv⟩ := WP.spec_imp_exists (Slice.index_usize_spec atoms ix1 hread)
  obtain ⟨tx, htx⟩ := atom_eval_ok ax facts known
  simp only [bind_tc_ok, hix1, hax, htx]
  have hnext : i.val + 1 ≤ Usize.max := by omega
  obtain ⟨ix2, hix2, hix2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hnext)
  cases tx with
  | T =>
    refine ⟨ControlFlow.cont (acc, ix2), ?_, ?_⟩
    · simp only [hix2, bind_tc_ok]
    · exact Or.inl ⟨acc, ix2, rfl, hix2v, Or.inr rfl⟩
  | F =>
    refine ⟨ControlFlow.done Tri.F, rfl, ?_⟩
    exact Or.inr rfl
  | U =>
    refine ⟨ControlFlow.cont (Tri.U, ix2), ?_, ?_⟩
    · simp only [hix2, bind_tc_ok]
    · exact Or.inl ⟨Tri.U, ix2, rfl, hix2v, Or.inl rfl⟩

/-- The guard loop's exit branch: at or past the guard length it returns the accumulated Kleene
    value unchanged, with no atom evaluation. -/
theorem guard_body_exit (atoms : Slice Atom) (facts : Slice U64) (known : U64)
    (start len : Usize) (acc : Tri) (i : Usize) (hi : len.val ≤ i.val) :
    policy_source.policy.decide.guard_eval_loop.body atoms facts known start len acc i
      = ok (ControlFlow.done acc) := by
  unfold policy_source.policy.decide.guard_eval_loop.body
  rw [if_neg ((UScalar.lt_equiv i len).not.mpr (by omega))]

/-- The work measure of the guard loop: from `(Tri.T, i)` it performs exactly `len.val - i.val`
    atom evaluations, one per continuation, and every continuation lowers the remaining-rank
    measure `len.val - i.val` strictly. The count is therefore strictly below one more. -/
theorem guard_loop_work (atoms : Slice Atom) (facts : Slice U64) (known : U64)
    (start len i : Usize) (hi : i.val ≤ len.val) (hrange : start.val + len.val ≤ (Slice.len atoms).val) :
    policy_source.policy.decide.guard_eval_loop atoms facts known start len Tri.T i ⦃ _ =>
      ∃ n : Nat, n = len.val - i.val ∧ n < len.val - i.val + 1 ⦄ := by
  unfold policy_source.policy.decide.guard_eval_loop
  apply loop.spec_decr_nat (measure := fun st : Tri × Usize => len.val - st.2.val)
    (inv := fun st : Tri × Usize => st.2.val ≤ len.val)
    (post := fun _ : Tri => ∃ n : Nat, n = len.val - i.val ∧ n < len.val - i.val + 1)
  · intro st hst
    rcases st with ⟨acc, j⟩
    dsimp only at hst ⊢
    by_cases hlt : j.val < len.val
    · obtain ⟨r, hbody, hshape⟩ :=
        guard_body_step atoms facts known start len acc j hlt (by omega)
      rw [hbody]
      simp only [WP.spec_ok]
      rcases hshape with ⟨acc', i', hcont, hincr, hacc'⟩ | hdone
      · have hproj : ((acc', i') : Tri × Usize).2 = i' := rfl
        simp only [hcont]
        exact ⟨by omega, by omega⟩
      · simp only [hdone]
        exact ⟨len.val - i.val, rfl, by omega⟩
    · rw [guard_body_exit atoms facts known start len acc j (by omega)]
      exact ⟨len.val - i.val, rfl, by omega⟩
  · exact hi

/-- A guard range starting past the atom slice is unknown, and the range check fires before any
    atom evaluation. -/
theorem guard_eval_start_out (rule : Rule) (atoms : Slice Atom) (facts : Slice U64)
    (known : U64) (h : (Slice.len atoms).val < rule.atom_start.val) :
    policy_source.policy.decide.guard_eval rule atoms facts known = ok Tri.U := by
  simp only [Slice.len_val] at *
  have hgt : UScalar.cast .Usize rule.atom_start > Slice.len atoms :=
    (UScalar.lt_equiv (Slice.len atoms) (UScalar.cast .Usize rule.atom_start)).2 (by
      rw [cast_slot_val]
      simp only [Slice.len_val] at h ⊢
      exact h)
  unfold policy_source.policy.decide.guard_eval
  simp only [lift_ok, bind_tc_ok]
  rw [if_pos hgt]

/-- The rule index a pass-1 loop state carries. -/
def pass1Idx {α : Type} (st : α × α × α × (Option U32) × (Option U32) × Usize) : Usize :=
  st.2.2.2.2.2

/-- Pass 1's exit branch: at or past `rules.len` the body returns the five accumulators
    unchanged, reading no rule and evaluating no guard. -/
theorem decide_loop0_body_exit (rules : Slice Rule) (atoms : Slice Atom) (facts : Slice U64)
    (known : U64) (action : U32) (want : U64) (permit forbid ask : U64)
    (permit_by ask_by : Option U32) (i : Usize) (h : (Slice.len rules).val ≤ i.val) :
    policy_source.policy.decide.decide_loop0.body rules atoms facts known action want
        permit forbid ask permit_by ask_by i
      = ok (ControlFlow.done (permit, forbid, ask, permit_by, ask_by)) := by
  unfold policy_source.policy.decide.decide_loop0.body
  rw [if_neg ((UScalar.lt_equiv i (Slice.len rules)).not.mpr (by omega))]

/-- Pass 2's exit branch: at or past `rules.len` the body returns the folded forbid cause
    unchanged, reading no rule and evaluating no guard. -/
theorem decide_loop1_body_exit (rules : Slice Rule) (atoms : Slice Atom) (facts : Slice U64)
    (known : U64) (action : U32) (want : U64) (permit : U64) (forbid_by : Option U32)
    (j : Usize) (h : (Slice.len rules).val ≤ j.val) :
    policy_source.policy.decide.decide_loop1.body rules atoms facts known action want permit
        forbid_by j = ok (ControlFlow.done forbid_by) := by
  unfold policy_source.policy.decide.decide_loop1.body
  rw [if_neg ((UScalar.lt_equiv j (Slice.len rules)).not.mpr (by omega))]

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

#resource_audit PolicyResource.cast_slot_val
#resource_audit PolicyResource.shr_ok
#resource_audit PolicyResource.index_ok
#resource_audit PolicyResource.idx_next_max
#resource_audit PolicyResource.loop_done
#resource_audit PolicyResource.loop_cont
#resource_audit PolicyResource.tri_eq_t
#resource_audit PolicyResource.tri_ne_f
#resource_audit PolicyResource.effect_eq_forbid
#resource_audit PolicyResource.min_id_spec
#resource_audit PolicyResource.atom_eval_ok
#resource_audit PolicyResource.atom_eval_slot_unknown
#resource_audit PolicyResource.guard_body_step
#resource_audit PolicyResource.guard_body_exit
#resource_audit PolicyResource.guard_loop_work
#resource_audit PolicyResource.guard_eval_start_out
#resource_audit PolicyResource.decide_loop0_body_exit
#resource_audit PolicyResource.decide_loop1_body_exit

run_cmd Lean.logInfo "POLICY_KERNEL_OK"

end PolicyResource
