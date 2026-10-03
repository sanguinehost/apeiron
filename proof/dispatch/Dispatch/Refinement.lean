import Dispatch.Spec

namespace Dispatch.Refinement
open dispatch_extract Aeneas Aeneas.Std Result

 theorem digest_same (a b : core.Digest) :
    core.Digest.same a b = ok (decide (a = b)) := by
  cases a with | mk a0 a1 a2 a3 =>
    cases b with | mk b0 b1 b2 b3 =>
      by_cases h0 : a0 = b0 <;> by_cases h1 : a1 = b1 <;>
        by_cases h2 : a2 = b2 <;> by_cases h3 : a3 = b3 <;>
        simp [core.Digest.same, core.Digest.mk.injEq, h0, h1, h2, h3]

 theorem start_refines (reg : Option core.Digest) (observed : core.Digest) :
    core.start reg observed = ok (Spec.start reg observed) := by
  cases reg with
  | none => simp [core.start, Spec.start]
  | some identity =>
    by_cases h : identity = observed <;> simp [core.start, Spec.start, digest_same, h]

 theorem advance_refines (s : core.Session) (o : core.Observation) :
    core.advance s o = ok (Spec.advance s o) := by
  cases s with | mk identity stage =>
    cases o with | mk observedStage exit observedIdentity =>
      cases stage <;> cases observedStage <;>
        by_cases he : exit = 0#i32 <;> by_cases hi : identity = observedIdentity <;>
        simp [core.advance, Spec.advance, Spec.Valid, Spec.next, digest_same, he, hi] <;> aesop

 theorem finish_refines (s : core.Session) (observed : core.Digest) :
    core.finish s observed = ok (Spec.finish s observed) := by
  cases s with | mk identity stage =>
    cases stage <;> by_cases hi : identity = observed <;>
      (simp [core.finish, Spec.finish, digest_same, hi]; all_goals aesop)

 theorem complete_refines (reg : Option core.Digest) (initial : core.Digest)
    (a b c d : core.Observation) (fresh : core.Digest) :
    core.complete reg initial a b c d fresh = ok (Spec.complete reg initial a b c d fresh) := by
  rw [core.complete, start_refines]
  by_cases hr : reg = some initial
  · simp only [Spec.start, hr, if_true, bind_tc_ok]
    rw [advance_refines]
    by_cases ha : Spec.Valid a .Extraction initial
    · simp only [Spec.advance, ha, if_true, Spec.next, Option.map_some, bind_tc_ok]
      rw [advance_refines]
      by_cases hb : Spec.Valid b .Dependencies initial
      · simp only [Spec.advance, hb, if_true, Spec.next, Option.map_some, bind_tc_ok]
        rw [advance_refines]
        by_cases hc : Spec.Valid c .Certificate initial
        · simp only [Spec.advance, hc, if_true, Spec.next, Option.map_some, bind_tc_ok]
          rw [advance_refines]
          by_cases hd : Spec.Valid d .Revalidation initial
          · simp only [Spec.advance, hd, if_true, Spec.next, Option.map_some, bind_tc_ok]
            rw [finish_refines]
            simp [Spec.finish, Spec.complete, Spec.Admissible, ha, hb, hc, hd]
          · simp [Spec.advance, hd, Spec.complete, Spec.Admissible, ha, hb, hc]
        · simp [Spec.advance, hc, Spec.complete, Spec.Admissible, ha, hb]
      · simp [Spec.advance, hb, Spec.complete, Spec.Admissible, ha]
    · simp [Spec.advance, ha, Spec.complete, Spec.Admissible]
  · simp [Spec.start, hr, Spec.complete, Spec.Admissible]

 theorem permit_iff (reg : Option core.Digest) (initial : core.Digest)
    (a b c d : core.Observation) (fresh : core.Digest) :
    (∃ p, core.complete reg initial a b c d fresh = ok (some p)) ↔
      Spec.Admissible reg initial a b c d fresh := by
  rw [complete_refines]
  by_cases h : Spec.Admissible reg initial a b c d fresh <;> simp [Spec.complete, h]

 theorem witness_success : core.witness_success = ok (some ⟨core.WITNESS_DIGEST⟩) := by
  simp [core.witness_success, core.witness_obs, complete_refines, Spec.complete,
    Spec.Admissible, Spec.Valid]

 theorem witness_refusal : core.witness_refusal_unregistered = ok none := by
  simp [core.witness_refusal_unregistered, core.witness_obs, complete_refines,
    Spec.complete, Spec.Admissible]

 theorem joint_reachable :
    (∃ p, core.witness_success = ok (some p)) ∧ core.witness_refusal_unregistered = ok none :=
  ⟨⟨_, witness_success⟩, witness_refusal⟩

end Dispatch.Refinement
