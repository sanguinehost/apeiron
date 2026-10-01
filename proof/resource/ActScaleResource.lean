import ActScaleSource
import Aeneas

/-
SCOPE: actual extracted activation-scale loop transitions, exit and resource rank.
REGISTER: scalar source refinement; no hardware or canonical ring correspondence.
CORPUS: src/engine/kern/pow2.rs and actual extracted pow2.act_scale_loop.body.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: lead extraction/proof/mutation/restoration gate, 480-second task budget.
GRADE: Lean checked only when the lead gate succeeds; no hardware measurement.
-/

namespace ActScaleResource
open Aeneas Aeneas.Std Result

/-- One guarded production iteration doubles the scale and increments its counter,
    preserving its power-of-two interpretation and decreasing the remaining rank. -/
theorem body_step (need s : U64) (steps : U32)
    (hscale : s.val = 2 ^ steps.val)
    (hneed : s.val < need.val) (hsteps : steps.val < 62) :
    ∃ s' : U64, ∃ steps' : U32,
      act_scale_source.pow2.act_scale_loop.body need s steps =
        ok (ControlFlow.cont (s', steps')) ∧
      s'.val = 2 * s.val ∧ steps'.val = steps.val + 1 ∧
      s'.val = 2 ^ steps'.val ∧ steps'.val ≤ 62 ∧
      62 - steps'.val < 62 - steps.val := by
  have hpow : (2 : Nat) ^ steps.val ≤ 2 ^ 61 :=
    Nat.pow_le_pow_right (by omega) (by omega)
  have hmul : s.val * (2#u64).val ≤ U64.max := by
    rw [hscale, U64.max_eq]
    norm_num at hpow ⊢
    omega
  have hadd : steps.val + (1#u32).val ≤ U32.max := by
    rw [U32.max_eq]
    norm_num
    omega
  obtain ⟨s', hmul', hs'⟩ := WP.spec_imp_exists
    (U64.mul_spec (x := s) (y := 2#u64) hmul)
  obtain ⟨steps', hadd', hsteps'⟩ := WP.spec_imp_exists
    (U32.add_spec (x := steps) (y := 1#u32) hadd)
  have hs : s'.val = 2 * s.val := by simpa [Nat.mul_comm] using hs'
  have ht : steps'.val = steps.val + 1 := by simpa using hsteps'
  refine ⟨s', steps', ?_, hs, ht, ?_, ?_, ?_⟩
  · unfold act_scale_source.pow2.act_scale_loop.body
    rw [if_pos ((UScalar.lt_equiv s need).2 hneed)]
    have hguard : steps < 62#u32 := (UScalar.lt_equiv steps (62#u32)).2 (by simpa)
    rw [if_pos hguard, hmul', hadd']
    simp only [bind_tc_ok]
  · rw [hs, ht, hscale, pow_succ]
    omega
  · omega
  · omega

/-- The actual exit branch returns the current scale exactly. -/
theorem body_exit (need s : U64) (steps : U32)
    (hexit : need.val ≤ s.val ∨ 62 ≤ steps.val) :
    act_scale_source.pow2.act_scale_loop.body need s steps =
      ok (ControlFlow.done s) := by
  unfold act_scale_source.pow2.act_scale_loop.body
  by_cases hn : s < need
  · have hn' := (UScalar.lt_equiv s need).1 hn
    have ht : ¬ steps < 62#u32 := by
      intro hc
      have hc' : steps.val < 62 := by
        simpa using (UScalar.lt_equiv steps (62#u32)).1 hc
      omega
    rw [if_pos hn, if_neg ht]
  · rw [if_neg hn]

/-- The constructor state of the production loop satisfies the joint invariant. -/
theorem initial_witness :
    (1#u64).val = 2 ^ (0#u32).val ∧ (0#u32).val < 62 := by
  norm_num

/-- The extracted zero branch is an actual successful input witness. -/
theorem zero_witness : act_scale_source.pow2.act_scale 0#u64 = ok (some 0#u64) := by
  simp [act_scale_source.pow2.act_scale]

/-- The executing loop's reachable invariant, including the predecessor bound. -/
def loopInv (need s : U64) (steps : U32) : Prop :=
  s.val = 2 ^ steps.val ∧ steps.val ≤ 58 ∧
    (steps.val = 0 ∨ s.val < 2 * need.val)

/-- Every bounded positive need terminates with the least covering power of two.
    The rank starts at 58 and each continuation lowers it by one. -/
theorem loop_spec (need : U64) (hbound : need.val ≤ 2 ^ 58) :
    act_scale_source.pow2.act_scale_loop need 1#u64 0#u32 ⦃ s =>
      ∃ k : Nat, k ≤ 58 ∧ s.val = 2 ^ k ∧ need.val ≤ s.val ∧
        (k = 0 ∨ s.val < 2 * need.val) ⦄ := by
  unfold act_scale_source.pow2.act_scale_loop
  apply loop.spec_decr_nat
    (measure := fun st : U64 × U32 => 58 - st.2.val)
    (inv := fun st : U64 × U32 => loopInv need st.1 st.2)
    (post := fun s : U64 => ∃ k : Nat, k ≤ 58 ∧ s.val = 2 ^ k ∧
      need.val ≤ s.val ∧ (k = 0 ∨ s.val < 2 * need.val))
  · intro st hst
    rcases st with ⟨s, steps⟩
    dsimp only at hst ⊢
    rcases hst with ⟨hs, ht, hleast⟩
    by_cases hlt : s.val < need.val
    · have h58 : steps.val < 58 := by
        by_contra hc
        have heq : steps.val = 58 := by omega
        rw [heq] at hs
        omega
      obtain ⟨s', steps', hbody, hs', ht', hp', hb', hr'⟩ :=
        body_step need s steps hs hlt (by omega)
      have hnextBound : steps'.val ≤ 58 := by omega
      have hnextLeast : s'.val < 2 * need.val := by rw [hs']; omega
      have hnextRank : 58 - steps'.val < 58 - steps.val := by omega
      have hnext : loopInv need s' steps' := ⟨hp', hnextBound, Or.inr hnextLeast⟩
      rw [hbody]
      exact ⟨hnext, hnextRank⟩
    · rw [body_exit need s steps (Or.inl (by omega))]
      exact ⟨steps.val, ht, hs, by omega, hleast⟩
  · show loopInv need 1#u64 0#u32
    norm_num [loopInv]

/-- A positive production-loop input jointly satisfies the bound and success result. -/
theorem positive_loop_witness :
    ∃ s : U64, act_scale_source.pow2.act_scale_loop 2#u64 1#u64 0#u32 = ok s ∧
      s.val = 2 := by
  obtain ⟨s, hrun, k, hk, hs, hcover, hleast⟩ :=
    WP.spec_imp_exists (loop_spec 2#u64 (by norm_num))
  refine ⟨s, hrun, ?_⟩
  have hlow : 2 ≤ s.val := by simpa using hcover
  rcases hleast with hzero | hsmall
  · rw [hzero] at hs
    norm_num at hs
    omega
  · have hsmall' : s.val < 4 := by simpa using hsmall
    have hkpos : 0 < k := by
      by_contra hc
      have hz : k = 0 := by omega
      rw [hz] at hs
      norm_num at hs
      omega
    have hkone : k = 1 := by
      by_contra hc
      have hpow : (2 : Nat) ^ 2 ≤ 2 ^ k :=
        Nat.pow_le_pow_right (by omega) (by omega)
      norm_num at hpow
      omega
    rw [hkone] at hs
    simpa using hs

/-- The overflow-safe ceiling used by the production scale calculation. -/
def ceilNeed (a : U64) : Nat :=
  a.val / 127 + if a.val % 127 = 0 then 0 else 1

/-- Every positive machine input returns the least covering power of two;
    the actual doubling loop has rank at most 58. -/
theorem scale_positive (a : U64) (ha : a ≠ 0#u64) :
    ∃ s : U64, act_scale_source.pow2.act_scale a = ok (some s) ∧
      ∃ k : Nat, k ≤ 58 ∧ s.val = 2 ^ k ∧ ceilNeed a ≤ s.val ∧
        (k = 0 ∨ s.val < 2 * ceilNeed a) := by
  obtain ⟨q, hq, hqv⟩ := WP.spec_imp_exists
    (U64.div_spec a (y := 127#u64) (by norm_num))
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (U64.rem_spec a (y := 127#u64) (by norm_num))
  have hqv' : q.val = a.val / 127 := by simpa using hqv
  have hrv' : r.val = a.val % 127 := by simpa using hrv
  let b := core.convert.num.FromU64Bool.from (r != 0#u64)
  have hb : b.val = if a.val % 127 = 0 then 0 else 1 := by
    by_cases hz : r = 0#u64
    · have hzero : a.val % 127 = 0 := by rw [← hrv', hz]; simp
      simp [b, core.convert.num.FromU64Bool.from, hz, hzero]
    · have hnonzero : a.val % 127 ≠ 0 := by
        intro hc
        apply hz
        apply UScalar.eq_of_val_eq
        rw [hrv', hc]
        simp
      simp [b, core.convert.num.FromU64Bool.from, hz, hnonzero]
  have hb1 : b.val ≤ 1 := by rw [hb]; split <;> omega
  have hbound : q.val + b.val ≤ 2 ^ 58 := by
    have hmax := U64.le_max a
    rw [hqv']
    norm_num
    omega
  have hadd : q.val + b.val ≤ U64.max := by
    rw [U64.max_eq]
    norm_num at hbound ⊢
    omega
  obtain ⟨need, hneed, hnv⟩ := WP.spec_imp_exists
    (U64.add_spec (x := q) (y := b) hadd)
  have hceil : need.val = ceilNeed a := by rw [hnv, hqv', hb]; rfl
  obtain ⟨s, hloop, k, hk, hscale, hcover, hleast⟩ := WP.spec_imp_exists
    (loop_spec need (by rw [hnv]; exact hbound))
  have hguard : ¬ s < need := by
    intro hc
    have := (UScalar.lt_equiv s need).1 hc
    omega
  refine ⟨s, ?_, k, hk, hscale, ?_, ?_⟩
  · unfold act_scale_source.pow2.act_scale
    rw [if_neg ha, hq, hr]
    simp only [bind_tc_ok, lift]
    change (do let n ← q + b; let z ← act_scale_source.pow2.act_scale_loop n 1#u64 0#u32
               if z < n then ok none else ok (some z)) = ok (some s)
    rw [hneed]
    simp only [bind_tc_ok]
    rw [hloop]
    simp only [bind_tc_ok, if_neg hguard]
  · rwa [hceil] at hcover
  · rwa [hceil] at hleast

/-- No refusal result is reachable for a `u64` production input. -/
theorem scale_never_none (a : U64) : act_scale_source.pow2.act_scale a ≠ ok none := by
  by_cases ha : a = 0#u64
  · subst a
    rw [zero_witness]
    intro hc
    have impossible := Result.ok_injective hc
    cases impossible
  · obtain ⟨s, hs, _⟩ := scale_positive a ha
    rw [hs]
    intro hc
    have impossible := Result.ok_injective hc
    cases impossible

end ActScaleResource

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

#resource_audit ActScaleResource.body_step
#resource_audit ActScaleResource.body_exit
#resource_audit ActScaleResource.initial_witness
#resource_audit ActScaleResource.zero_witness
#resource_audit ActScaleResource.loop_spec
#resource_audit ActScaleResource.positive_loop_witness
#resource_audit ActScaleResource.scale_positive
#resource_audit ActScaleResource.scale_never_none

run_cmd Lean.logInfo "ACT_SCALE_KERNEL_OK"
