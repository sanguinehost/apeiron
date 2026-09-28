import Growable.ReadLoop
import Growable.Scan
import Lean

/-!
# Locate-loop completion facts for `table.probe.spill_locate`

Facts about the extracted spill-locate loop bodies, targeting
`SpillLocateFoundFullMatch` (`Growable/Classifier.lean`): a generated
`ok (table.SpillSite.Found index)` carries a full-key match certificate at `index`,
`∃ payload, region.slots.val[index.val]? = some payload ∧
table.matches keyrep payload sigma key address = ok true`.

Structural facts recorded here:

* the locate inner loop state is `(Bool, Usize, U64)` (matched flag, matched slot,
  remaining mask) instead of `spill_read`'s `(Option Usize, U64)`;
* the locate inner body calls `table.least_set_lane` (the `table.least_set_lane_loop`
  bit scan) instead of `table.scan.least_set_lane` (the SWAR scan whose spec lives in
  `Growable/Scan.lean`). **Scanner-duplication hazard:** two generated least-set-lane
  implementations must agree; `scan_lane_of_loop_lane` transfers the `Scan` facts
  across the coincidence, and a future divergence of either scanner breaks it. The
  shared characterization is the divisibility one from
  `Apeiron.GrowableScan.least_set_lane_selects`;
* a `SpillSite.Found` result forces `may_match = true`: the outer loop's
  `may_match = false` branch scans only vacancies and never sets the match flag `b1`,
  so `Found index` is exactly the outer state's `b1 = true, i2 = index` under
  `may_match = true`. The certificate projects from the loop state; `SpillSite.Found`
  transports only the lane index.

Landed here: the loop-scanner spec (L1), the inner body-step clear (L2), the inner
completion outcome (L3), the outer 6-tuple sweep (`LocateOuterInv`/`LocateOuterOutcome`/
`generated_locate_outer_outcome`) with its geometry discharge
(`locate_outer_step_ready_of_geometry`), and the top-level `generated_spill_locate_unfold`
(L5). Remaining: the final projection to `SpillLocateFoundFullMatch`
(`Growable/Classifier.lean`), which needs run-relative found-extraction lemmas: the
outcome theorems' readiness premises are unsatisfiable at a found step, where the inner
sweep stops at the matching lane and never compares the remaining selected lanes.
-/

namespace Apeiron.GrowableLocateLoop

open Aeneas Aeneas.Std Result
open apeiron_word
open Apeiron.GrowableReadLoop (InnerCompared InnerReady)
open Apeiron.GrowableScan (liveLaneCount)

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-! ## L1: the loop scanner `table.least_set_lane` -/

/-- Clear bits below `l` give the two-adic divisibility `2 ^ l ∣ n`. -/
private theorem two_pow_dvd_of_testBit_false (n l : Nat)
    (h : ∀ j : Nat, j < l → n.testBit j = false) : 2 ^ l ∣ n := by
  induction l generalizing n with
  | zero => exact ⟨n, by ring⟩
  | succ l ih =>
    have h0 : n.testBit 0 = false := h 0 (by omega)
    have heven : 2 ∣ n := by
      by_contra hnot
      have hne1 : n % 2 = 1 := by
        have hlt : n % 2 < 2 := Nat.mod_lt n (by norm_num)
        have hnz : n % 2 ≠ 0 := by
          intro hh
          exact hnot (Nat.dvd_iff_mod_eq_zero.mpr hh)
        omega
      simp only [Nat.testBit_eq_decide_div_mod_eq] at h0
      rw [Nat.pow_zero, Nat.div_one, hne1] at h0
      simp at h0
    obtain ⟨k, hk⟩ := heven
    subst n
    have hshift : ∀ j : Nat, j < l → k.testBit j = false := by
      intro j hj
      have hclear1 : (2 * k).testBit (j + 1) = false := h (j + 1) (by omega)
      simp only [Nat.testBit_eq_decide_div_mod_eq] at hclear1 ⊢
      have hcancel : (2 * k) / 2 ^ (j + 1) = k / 2 ^ j := by
        rw [Nat.pow_succ, Nat.mul_comm (2 ^ j) 2, ← Nat.div_div_eq_div_mul]
        have hhalf : (2 * k) / 2 = k := by omega
        rw [hhalf]
      rw [hcancel] at hclear1
      exact hclear1
    have hkvd : 2 ^ l ∣ k := ih k hshift
    obtain ⟨q, hq⟩ := hkvd
    refine ⟨q, ?_⟩
    rw [Nat.pow_succ, hq]
    ring

/-- A set bit at `l` excludes divisibility by `2 ^ (l + 1)`. -/
private theorem not_two_pow_succ_dvd_of_testBit (n l : Nat)
    (h : n.testBit l = true) : ¬ 2 ^ (l + 1) ∣ n := by
  intro hdvd
  obtain ⟨k, hk⟩ := hdvd
  have hshape : n = 2 ^ l * (2 * k) := by rw [hk, Nat.pow_succ]; ring
  have hdiv : n / 2 ^ l = 2 * k := by
    rw [hshape, Nat.mul_comm (2 ^ l) (2 * k)]
    exact Nat.mul_div_left (2 * k) (pow_pos (by norm_num : (0 : Nat) < 2) l)
  have hfalse : n.testBit l = false := by
    simp only [Nat.testBit_eq_decide_div_mod_eq, hdiv]
    have hmod : (2 * k) % 2 = 0 := by omega
    simp [hmod]
  rw [hfalse] at h
  simp at h

/-- The loop scanner's run form: every lane below the returned one is clear, and a
returned lane below 64 holds the set bit. -/
private theorem least_set_lane_loop_run (bits : U64) (lane : Usize)
    (hbound : lane.val ≤ 64)
    (hclear : ∀ j : Nat, j < lane.val → bits.val.testBit j = false) :
    table.least_set_lane_loop bits lane ⦃ out =>
      out.val ≤ 64 ∧
      (∀ j : Nat, j < out.val → bits.val.testBit j = false) ∧
      (out.val < 64 → bits.val.testBit out.val = true) ⦄ := by
  unfold table.least_set_lane_loop
  apply loop.spec_decr_nat
    (measure := fun l : Usize => 64 - l.val)
    (inv := fun l : Usize =>
      l.val ≤ 64 ∧ ∀ j : Nat, j < l.val → bits.val.testBit j = false)
    (post := fun out : Usize =>
      out.val ≤ 64 ∧
      (∀ j : Nat, j < out.val → bits.val.testBit j = false) ∧
      (out.val < 64 → bits.val.testBit out.val = true))
  · intro st hst
    rcases hst with ⟨hstbound, hstclear⟩
    unfold table.least_set_lane_loop.body
    have hlift : lift (UScalar.cast .Usize core.num.U64.BITS) =
        ok (UScalar.cast .Usize core.num.U64.BITS) := rfl
    rw [hlift, bind_tc_ok]
    have hcast : (UScalar.cast .Usize core.num.U64.BITS).val = 64 := by scalar_tac
    by_cases hcmp : st.val < 64
    · have hlt : st < UScalar.cast .Usize core.num.U64.BITS :=
        (UScalar.lt_equiv _ _).mpr (by simpa [hcast] using hcmp)
      rw [if_pos hlt]
      have hshift : bits >>> st = ok ⟨bits.bv.ushiftRight st.val⟩ := by
        show UScalar.shiftRight bits st.val = ok ⟨bits.bv.ushiftRight st.val⟩
        unfold UScalar.shiftRight
        have hnum : UScalarTy.numBits .U64 = 64 := by decide
        simp only [hnum, if_pos hcmp]
      rw [hshift, bind_tc_ok]
      have hlift2 : lift ((⟨bits.bv.ushiftRight st.val⟩ : U64) &&& 1#u64) =
          ok ((⟨bits.bv.ushiftRight st.val⟩ : U64) &&& 1#u64) := rfl
      rw [hlift2, bind_tc_ok]
      have hone : (((⟨bits.bv.ushiftRight st.val⟩ : U64) &&& 1#u64).val) =
          bits.val / 2 ^ st.val % 2 := by
        rw [UScalar.val_and]
        have hshr : (⟨bits.bv.ushiftRight st.val⟩ : U64).val =
            bits.val / 2 ^ st.val := by
          show (bits.bv >>> st.val).toNat = bits.bv.toNat / 2 ^ st.val
          rw [BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]
        have honev : (1#u64 : U64).val = 1 := by decide
        rw [hshr, honev]
        exact Nat.and_one_is_mod _
      have htest : bits.val.testBit st.val =
          decide (bits.val / 2 ^ st.val % 2 = 1) :=
        Nat.testBit_eq_decide_div_mod_eq
      by_cases hz : (((⟨bits.bv.ushiftRight st.val⟩ : U64) &&& 1#u64) = 0#u64)
      · have hzval : bits.val / 2 ^ st.val % 2 = 0 := by
          have hz0 := congrArg (fun x : U64 => x.val) hz
          have hv : (0#u64 : U64).val = 0 := by decide
          rw [hone, hv] at hz0
          exact hz0
        have hbit : bits.val.testBit st.val = false := by
          rw [htest, hzval]
          decide
        rw [if_pos hz]
        have hcap : st.val + 1 ≤ Usize.max := by
          have hmax : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
            simp [Usize.max, Usize.numBits]
          rcases System.Platform.numBits_eq with hp | hp
          · rw [hmax, hp]; omega
          · rw [hmax, hp]; omega
        obtain ⟨next, hnext, hnextval⟩ :=
          WP.spec_imp_exists (Usize.add_spec (x := st) (y := 1#usize) hcap)
        have hnextValue : next.val = st.val + 1 := hnextval
        rw [hnext, bind_tc_ok, WP.spec_ok]
        refine ⟨⟨by omega, ?_⟩, ?_⟩
        · intro j hj
          by_cases hjl : j < st.val
          · exact hstclear j hjl
          · have heq : j = st.val := by omega
            subst j
            exact hbit
        · simpa using (show 64 - next.val < 64 - st.val by omega)
      · have hnzval : bits.val / 2 ^ st.val % 2 = 1 := by
          have hneval : (((⟨bits.bv.ushiftRight st.val⟩ : U64) &&& 1#u64).val) ≠ 0 := by
            intro hh
            apply hz
            exact UScalar.eq_of_val_eq (by simpa using hh)
          rw [hone] at hneval
          have hbound : bits.val / 2 ^ st.val % 2 < 2 := Nat.mod_lt _ (by norm_num)
          omega
        have hbit : bits.val.testBit st.val = true := by
          rw [htest, hnzval]
          decide
        rw [if_neg hz]
        rw [WP.spec_ok]
        exact ⟨hstbound, hstclear, fun _ => hbit⟩
    · have hnot : ¬ st < UScalar.cast .Usize core.num.U64.BITS := by
        intro hh
        apply hcmp
        have := (UScalar.lt_equiv _ _).mp hh
        simpa [hcast] using this
      rw [if_neg hnot]
      rw [WP.spec_ok]
      refine ⟨hstbound, hstclear, ?_⟩
      intro hh
      exact absurd hh hcmp
  · exact ⟨hbound, hclear⟩

/-- The loop-based `table.least_set_lane` returns the least set lane of a nonzero mask,
in the divisibility characterization shared with the SWAR
`table.scan.least_set_lane` (`Apeiron.GrowableScan.least_set_lane_selects`). -/
theorem least_set_lane_loop_selects (bits : U64) (hne : bits ≠ 0#u64) :
    ∃ lane : Usize, table.least_set_lane bits = ok lane ∧
      lane.val < 64 ∧ 2 ^ lane.val ∣ bits.val ∧ ¬ 2 ^ (lane.val + 1) ∣ bits.val := by
  have hzero : (0#usize).val = 0 := rfl
  have hspec := least_set_lane_loop_run bits 0#usize
    (by rw [hzero]; omega) (by intro j hj; rw [hzero] at hj; omega)
  obtain ⟨out, hrun, hout⟩ := WP.spec_imp_exists hspec
  rcases hout with ⟨houtbound, houtclear, houtset⟩
  have hlt : out.val < 64 := by
    by_contra hge
    have heq : out.val = 64 := by omega
    have hdvd : 2 ^ 64 ∣ bits.val :=
      two_pow_dvd_of_testBit_false bits.val 64
        (fun j hj => houtclear j (by rw [heq]; omega))
    have hbound : bits.val < 2 ^ 64 := bits.bv.isLt
    obtain ⟨k, hk⟩ := hdvd
    have hk0 : k = 0 := by
      by_contra hkne
      have : 2 ^ 64 ≤ bits.val := by
        rw [hk]
        exact Nat.le_mul_of_pos_right (2 ^ 64) (by omega)
      omega
    subst k
    have hbits0 : bits.val = 0 := by rw [hk]; ring
    apply hne
    exact UScalar.eq_of_val_eq (by simpa using hbits0)
  exact ⟨out, hrun, hlt,
    two_pow_dvd_of_testBit_false bits.val out.val houtclear,
    not_two_pow_succ_dvd_of_testBit bits.val out.val (houtset hlt)⟩

/-- The loop scanner's run form: a successful result on a nonzero mask is in range and
selects its least set bit. -/
theorem least_set_lane_loop_in_range (bits : U64) (lane : Usize) (hne : bits ≠ 0#u64)
    (hselect : table.least_set_lane bits = ok lane) :
    lane.val < 64 ∧ 2 ^ lane.val ∣ bits.val ∧ ¬ 2 ^ (lane.val + 1) ∣ bits.val := by
  obtain ⟨lane', hselect', hbound', hdiv', hndiv'⟩ := least_set_lane_loop_selects bits hne
  have heq : lane' = lane := Result.ok_injective (hselect'.symm.trans hselect)
  subst lane'
  exact ⟨hbound', hdiv', hndiv'⟩

/-- The loop scanner selects the set bit at its returned lane. -/
private theorem loop_selected_bit (bits : U64) (lane : Usize)
    (hne : bits ≠ 0#u64)
    (hselect : table.least_set_lane bits = ok lane) :
    bits.bv.getLsbD lane.val = true := by
  obtain ⟨_, hdiv, hnot⟩ := least_set_lane_loop_in_range bits lane hne hselect
  obtain ⟨q, hq⟩ := hdiv
  have hodd : q % 2 = 1 := by
    have hn : ¬ 2 ∣ q := by
      intro hd
      apply hnot
      obtain ⟨k, hk⟩ := hd
      refine ⟨k, ?_⟩
      rw [hq, hk, pow_succ]
      ring
    omega
  rw [← BitVec.testBit_toNat]
  change bits.val.testBit lane.val = true
  rw [hq, Nat.mul_comm, Nat.testBit_mul_two_pow]
  simp [Nat.testBit_eq_decide_div_mod_eq, hodd]

/-- The two generated scanners agree on a nonzero mask: both return the least set lane,
which is unique under the divisibility characterization. -/
theorem scan_lane_of_loop_lane {bits : U64} {lane : Usize}
    (hne : bits ≠ 0#u64)
    (hselect : table.least_set_lane bits = ok lane) :
    table.scan.least_set_lane bits = ok lane := by
  obtain ⟨hbound, hdiv, hndiv⟩ := least_set_lane_loop_in_range bits lane hne hselect
  obtain ⟨lane', hscan, _, hdiv', hndiv'⟩ :=
    Apeiron.GrowableScan.least_set_lane_exact bits hne
  have heq : lane'.val = lane.val := by
    by_contra hneq
    rcases Nat.lt_or_gt_of_ne hneq with hlt | hgt
    · -- lane'.val < lane.val
      have hdvd : (2 : Nat) ^ (lane'.val + 1) ∣ 2 ^ lane.val :=
        (Nat.pow_dvd_pow_iff_le_right (by norm_num)).mpr (by omega)
      exact hndiv' (dvd_trans hdvd hdiv)
    · -- lane.val < lane'.val
      have hdvd : (2 : Nat) ^ (lane.val + 1) ∣ 2 ^ lane'.val :=
        (Nat.pow_dvd_pow_iff_le_right (by norm_num)).mpr (by omega)
      exact hndiv (dvd_trans hdvd hdiv')
  have helim : lane' = lane := UScalar.eq_of_val_eq heq
  rw [helim] at hscan
  exact hscan

/-! ## L2: the locate inner body step -/

/-- A successful generated payload lookup in the *locate* inner body executes the proved
clear after a selected hit. Adapted from
`Apeiron.GrowableScan.generated_spill_read_clears_selected`; the recorded pair is
`(matched, if matched then slot else prior)`. -/
theorem generated_spill_locate_clears_selected {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (v : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (hits : U64) (lane slot : Usize) (entry : cell.Sigma × R × V) (matched : Bool)
    (prior : Usize)
    (hne : hits ≠ 0#u64)
    (hselect : table.least_set_lane hits = ok lane)
    (hslot : base + lane = ok slot)
    (hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) v slot = ok entry)
    (hmatch : table.matches keyRep entry sigma key
      { place := place, direct_word := direct } = ok matched) :
    ∃ next : U64,
      table.probe.spill_locate_loop0_loop0.body keyRep v sigma place direct key base
        false prior hits =
        ok (ControlFlow.cont (matched, if matched then slot else prior, next)) ∧
      (∀ j : Nat, next.bv.getLsbD j =
        if j = lane.val then false else hits.bv.getLsbD j) ∧
      liveLaneCount next + 1 = liveLaneCount hits := by
  have hscan : table.scan.least_set_lane hits = ok lane :=
    scan_lane_of_loop_lane hne hselect
  obtain ⟨next, hclear, hexact, hcount, _⟩ :=
    Apeiron.GrowableScan.generated_clear_hits_live_decrease hits lane hne hscan
  refine ⟨next, ?_, hexact, hcount⟩
  unfold table.probe.spill_locate_loop0_loop0.body
  have hcond : (hits != 0#u64) = true := by simpa using hne
  simp only [hcond, if_true, Bool.false_eq_true, if_false, hselect, hslot,
    hindex, hmatch, bind_tc_ok]
  cases matched <;> simp
  all_goals
    rw [← bind_assoc_eq]
    rw [hclear]
    simp only [bind_tc_ok]

/-! ## L3: the locate inner completion outcome -/

/-- Completion invariant for the locate inner loop: either no match recorded and every
cleared lane compared and missed, or a match recorded whose slot is a certified
full-key comparison at an initially selected lane. -/
def LocateInnerInv {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (initial : U64) (state : Bool × Usize × U64) : Prop :=
  (state.1 = false ∧
    (∀ j, state.2.2.bv.getLsbD j = true → initial.bv.getLsbD j = true) ∧
    (∀ j, initial.bv.getLsbD j = true →
      state.2.2.bv.getLsbD j = true ∨
        ∃ slot, InnerCompared keyRep slots sigma place direct key base j false slot)) ∨
  (state.1 = true ∧
    ∃ j slot, state.2.1 = slot ∧ initial.bv.getLsbD j = true ∧
      InnerCompared keyRep slots sigma place direct key base j true slot)

/-- The locate inner loop's outcome: a recorded match is a certified full-key comparison
at the recorded slot; an unrecorded outcome compared every selected lane and missed. -/
def LocateInnerOutcome {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (initial : U64) (result : Bool × Usize) : Prop :=
  (result.1 = false ∧ ∀ j, initial.bv.getLsbD j = true →
    ∃ slot, InnerCompared keyRep slots sigma place direct key base j false slot) ∨
  (result.1 = true ∧ ∃ j slot, result.2 = slot ∧ initial.bv.getLsbD j = true ∧
    InnerCompared keyRep slots sigma place direct key base j true slot)

/-- The locate inner loop terminates on `liveLaneCount` and lands `LocateInnerOutcome`.
Proof skeleton mirrors `Apeiron.GrowableReadLoop.generated_inner_sweep_outcome`; lane
selection goes through `scan_lane_of_loop_lane` and the body-independent clear
`Apeiron.GrowableScan.generated_clear_hits_live_decrease`. -/
theorem generated_locate_inner_outcome {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (initial : U64) (prior : Usize)
    (hready : ∀ j, j < 64 → initial.bv.getLsbD j = true →
      ∃ slot, InnerCompared keyRep slots sigma place direct key base j false slot ∨
        InnerCompared keyRep slots sigma place direct key base j true slot) :
    table.probe.spill_locate_loop0_loop0 keyRep slots sigma place direct key
      false prior base initial ⦃ result =>
        LocateInnerOutcome keyRep slots sigma place direct key base initial result ⦄ := by
  unfold table.probe.spill_locate_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun state : Bool × Usize × U64 =>
      liveLaneCount state.2.2)
    (inv := LocateInnerInv keyRep slots sigma place direct key base initial)
    (post := LocateInnerOutcome keyRep slots sigma place direct key base initial)
  · intro state hstate
    rcases state with ⟨b, i1, hits⟩
    dsimp only at hstate ⊢
    rcases hstate with ⟨rfl, hsubset, hmiss⟩ | ⟨rfl, j, slot, hseq, horig, hcomp⟩
    · by_cases hne : hits ≠ 0#u64
      · obtain ⟨lane, hselect, hlane, _, _⟩ := least_set_lane_loop_selects hits hne
        have hbit : hits.bv.getLsbD lane.val = true :=
          loop_selected_bit hits lane hne hselect
        have horiginal : initial.bv.getLsbD lane.val = true := hsubset _ hbit
        obtain ⟨nextSlot, hfalse | htrue⟩ := hready lane.val hlane horiginal
        · rcases hfalse with ⟨selected, entry, hval, hadd, hindex, hmatch⟩
          have hselected : selected = lane := UScalar.eq_of_val_eq (by omega)
          subst selected
          obtain ⟨next, hbody, hclear, hcount⟩ :=
            generated_spill_locate_clears_selected
              keyRep slots sigma place direct key base hits lane nextSlot entry false i1
              hne hselect hadd hindex hmatch
          rw [hbody, WP.spec_ok]
          refine ⟨Or.inl ⟨rfl, ?_, ?_⟩, ?_⟩
          · intro k hk
            rw [hclear k] at hk
            split at hk
            · cases hk
            · exact hsubset k hk
          · intro k hk
            rw [hclear k]
            by_cases hkl : k = lane.val
            · simp only [if_pos hkl, Bool.false_eq_true, false_or]
              subst k
              exact ⟨nextSlot, lane, entry, rfl, hadd, hindex, hmatch⟩
            · simpa only [if_neg hkl] using hmiss k hk
          · simpa using (show liveLaneCount next < liveLaneCount hits by omega)
        · rcases htrue with ⟨selected, entry, hval, hadd, hindex, hmatch⟩
          have hselected : selected = lane := UScalar.eq_of_val_eq (by omega)
          subst selected
          obtain ⟨next, hbody, _, _⟩ :=
            generated_spill_locate_clears_selected
              keyRep slots sigma place direct key base hits lane nextSlot entry true i1
              hne hselect hadd hindex hmatch
          rw [hbody, WP.spec_ok]
          refine ⟨Or.inr ⟨rfl, lane.val, nextSlot, rfl, horiginal, ?_⟩, ?_⟩
          · exact ⟨lane, entry, rfl, hadd, hindex, hmatch⟩
          · simpa using (show liveLaneCount next < liveLaneCount hits by omega)
      · have hz : hits = 0#u64 := not_not.mp hne
        subst hits
        unfold table.probe.spill_locate_loop0_loop0.body
        simp only [bne_self_eq_false, Bool.false_eq_true, if_false, WP.spec_ok]
        left
        refine ⟨rfl, ?_⟩
        intro k hk
        rcases hmiss k hk with hbit | hcomp'
        · simp at hbit
        · exact hcomp'
    · -- recorded match carried into the stop: the slot is certified at `j`.
      subst slot
      unfold table.probe.spill_locate_loop0_loop0.body
      by_cases hcond : (hits != 0#u64) = true
      · simp [hcond, WP.spec_ok]
        exact Or.inr ⟨rfl, j, i1, rfl, horig, hcomp⟩
      · simp [hcond, WP.spec_ok]
        exact Or.inr ⟨rfl, j, i1, rfl, horig, hcomp⟩
  · left
    refine ⟨rfl, ?_, ?_⟩
    · intro j hj
      exact hj
    · intro j hj
      exact Or.inl hj

/-! ## L4: the locate outer 6-tuple sweep -/

/-- A recorded match slot: the executed step's inner sweep compared lane `j` at `base`
and recorded the full-key match at `slot`. -/
def LocateMatchWitness {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (slot : Usize) : Prop :=
  ∃ (base : Usize) (j : Nat),
    InnerCompared keyRep slots sigma place direct key base j true slot

/-- A recorded vacancy slot: the executed group's empty scan found the lane empty.
This is execution provenance; emptiness of the payload at `slot` is not claimed here. -/
def LocateVacantWitness (tags : alloc.vec.Vec U8) (n : Usize) (slot : Usize) : Prop :=
  ∃ (base finish : Usize) (line : Slice U8) (empty : U64) (lane : Usize),
    base + n = ok finish ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice U8)
      tags { start := base, «end» := finish } = ok line ∧
    table.scan.match_empty line = ok empty ∧ empty ≠ 0#u64 ∧
    table.least_set_lane empty = ok lane ∧
    base + lane = ok slot

/-- One executed miss step: the group's empty scan found no vacancy, and when the tag
scan ran, the inner sweep compared every selected lane and missed. -/
def LocateOuterStepMiss {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (may_match : Bool) (step : Usize) : Prop :=
  ∃ (group base finish : Usize) (line : Slice U8) (empty : U64),
    table.group place step mask = ok group ∧ group * n = ok base ∧
    base + n = ok finish ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice U8)
      tags { start := base, «end» := finish } = ok line ∧
    table.scan.match_empty line = ok empty ∧ empty = 0#u64 ∧
    (may_match = false ∨
      ∃ (bits : U64) (carried out : Usize),
        table.scan.match_tag line wanted = ok bits ∧
        table.probe.spill_locate_loop0_loop0 keyRep slots sigma place direct key
          false carried base bits = ok (false, out) ∧
        LocateInnerOutcome keyRep slots sigma place direct key base bits
          (false, out))

/-- Checked geometry and both scans for a reachable outer step, with the inner-loop
admission of its tag mask. Mirrors `Apeiron.GrowableReadLoop.OuterStepReady` with the
locate body's vacancy branch added. -/
def LocateOuterStepReady {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (step : Usize) : Prop :=
  ∃ (group base finish : Usize) (line : Slice U8) (bits empty : U64),
    table.group place step mask = ok group ∧ group * n = ok base ∧
    base + n = ok finish ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice U8)
      tags { start := base, «end» := finish } = ok line ∧
    table.scan.match_tag line wanted = ok bits ∧
    InnerReady keyRep slots sigma place direct key base bits ∧
    table.scan.match_empty line = ok empty ∧
    base.val + 64 ≤ Usize.max

/-- Outer completion invariant over the 6-tuple state: `may_match` is loop-invariant;
a recorded match is certified; a recorded vacancy traces to an executed empty scan; and
while neither flag is set, every step below the cursor executed as a miss. -/
def LocateOuterInv {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (may_match : Bool) (limit : Usize)
    (state : Bool × Usize × Bool × Usize × Bool × Usize) : Prop :=
  state.2.2.2.2.1 = may_match ∧
  state.2.2.2.2.2.val ≤ limit.val ∧
  (state.2.2.1 = true →
    LocateMatchWitness keyRep slots sigma place direct key state.2.2.2.1) ∧
  (state.1 = true → LocateVacantWitness tags n state.2.1) ∧
  ((state.1 = false ∧ state.2.2.1 = false) →
    ∀ j : Nat, j < state.2.2.2.2.2.val →
      ∃ cursor : Usize, cursor.val = j ∧
        LocateOuterStepMiss keyRep tags slots sigma place direct key n mask wanted
          may_match cursor)

/-- The locate outer sweep's outcome: a recorded match carries the full-key comparison
certificate, a recorded vacancy carries its empty-scan provenance, and an unflagged
result ran every step of the `WINDOWS + groups` sweep as a miss. -/
def LocateOuterOutcome {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (may_match : Bool) (limit : Usize)
    (result : Bool × Usize × Bool × Usize) : Prop :=
  (result.2.2.1 = true →
    LocateMatchWitness keyRep slots sigma place direct key result.2.2.2) ∧
  (result.1 = true → LocateVacantWitness tags n result.2.1) ∧
  ((result.1 = false ∧ result.2.2.1 = false) →
    ∀ j : Nat, j < limit.val →
      ∃ cursor : Usize, cursor.val = j ∧
        LocateOuterStepMiss keyRep tags slots sigma place direct key n mask wanted
          may_match cursor)

/-- The generated outer step's checked increment: a reachable step below `limit` can
always add one within the `Usize` range. -/
private theorem usize_step_cap {step limit : Usize} (_hbound : step.val ≤ limit.val)
    (hlt : step < limit) : step.val + 1 ≤ Usize.max := by
  have hltVal := (UScalar.lt_equiv step limit).mp hlt
  have hlimitBounds := limit.hBounds
  simp only [UScalarTy.Usize_numBits_eq] at hlimitBounds
  have hlimitCap : limit.val ≤ Usize.max := by
    have heq : Usize.max = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    rw [heq]
    omega
  omega

/-- The real generated outer loop lands `LocateOuterOutcome` from the initial state:
either a certified match slot, a vacancy with empty-scan provenance, or the complete
miss sweep. Per-step readiness is an explicit premise, mirroring
`Apeiron.GrowableReadLoop.generated_outer_sweep_outcome`; it is discharged from
geometry later, as `outer_step_ready_of_geometry` does for `spill_read`. -/
theorem generated_locate_outer_outcome {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8) (limit : Usize) (may_match : Bool)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hready : ∀ step : Usize, step < limit →
      LocateOuterStepReady keyRep tags slots sigma place direct key n mask wanted
        step) :
    table.probe.spill_locate_loop0 keyRep tags slots sigma place direct key n
      groups false 0#usize false 0#usize mask wanted may_match 0#usize ⦃ result =>
        LocateOuterOutcome keyRep tags slots sigma place direct key n mask wanted
          may_match limit result ⦄ := by
  unfold table.probe.spill_locate_loop0
  apply loop.spec_decr_nat
    (measure := fun state : Bool × Usize × Bool × Usize × Bool × Usize =>
      limit.val - state.2.2.2.2.2.val)
    (inv := LocateOuterInv keyRep tags slots sigma place direct key n mask wanted
      may_match limit)
    (post := LocateOuterOutcome keyRep tags slots sigma place direct key n mask
      wanted may_match limit)
  · intro state hstate
    rcases state with ⟨b, i1, b1, i2, may, step⟩
    dsimp only at hstate ⊢
    rcases hstate with ⟨hmay, hbound, hmatchcert, hvaccert, hcovered⟩
    have hbound' : step.val ≤ limit.val := hbound
    unfold table.probe.spill_locate_loop0.body
    simp only [hlimit, bind_tc_ok]
    by_cases hlt : step < limit
    · -- a step is available
      simp only [if_pos hlt]
      by_cases hb1 : b1 = true
      · -- recorded match: stop immediately
        simp only [hb1, if_true, WP.spec_ok]
        refine ⟨fun _ => hmatchcert hb1, hvaccert, ?_⟩
        intro ⟨_, hh2⟩
        exact absurd hh2 (by simp)
      · have hb1f : b1 = false := by cases b1 <;> simp_all
        simp only [hb1f, Bool.false_eq_true, if_false]
        by_cases hm : may = true
        · -- tag scan, inner sweep, vacancy scan
          simp only [hm, if_true]
          rcases hready step hlt with
            ⟨group, base, finish, line, bits, empty, hgroup, hbase, hfinish, hline,
              hbits, hinnerReady, hempty, hbaseCap⟩
          obtain ⟨inner, hinner, houtcome⟩ := WP.spec_imp_exists
            (generated_locate_inner_outcome keyRep slots sigma place direct key
              base bits i2 hinnerReady)
          rcases inner with ⟨b2, i6⟩
          simp only [hgroup, hbase, hfinish, hline, hbits, hinner, bind_tc_ok]
          by_cases hb : b = true
          · -- vacancy already recorded: the pair is kept
            simp only [hb, if_true, bind_tc_ok]
            have hcap : step.val + 1 ≤ Usize.max := usize_step_cap hbound' hlt
            obtain ⟨next, hnext, hnextval⟩ :=
              WP.spec_imp_exists (Usize.add_spec (x := step) (y := 1#usize) hcap)
            have hnextValue : next.val = step.val + 1 := hnextval
            have hnextBound : next.val ≤ limit.val := by
                have hltVal := (UScalar.lt_equiv step limit).mp hlt
                omega
            simp only [hnext, bind_tc_ok]
            refine ⟨⟨hm.symm.trans hmay, hnextBound, ?_, fun _ => hvaccert hb, ?_⟩, ?_⟩
            · intro hb2t
              rcases houtcome with ⟨hmiss, _⟩ | ⟨_, jw, sw, hslot, _, hcomp⟩
              · exact absurd hb2t (by simp_all)
              · have hslot' : i6 = sw := hslot
                subst sw
                exact ⟨base, jw, hcomp⟩
            · intro ⟨hh1, _⟩
              exact absurd hh1 (by simp)
            · simpa using (show limit.val - next.val < limit.val - step.val by omega)
          · have hbf : b = false := by cases b <;> simp_all
            simp only [hbf, Bool.false_eq_true, if_false, hempty, bind_tc_ok]
            by_cases hemp : empty = 0#u64
            · -- miss step: no vacancy
              have hempB : (empty != 0#u64) = false := by simpa using hemp
              simp only [hempB, Bool.false_eq_true, if_false, bind_tc_ok]
              have hcap : step.val + 1 ≤ Usize.max := usize_step_cap hbound' hlt
              obtain ⟨next, hnext, hnextval⟩ :=
                WP.spec_imp_exists (Usize.add_spec (x := step) (y := 1#usize) hcap)
              have hnextValue : next.val = step.val + 1 := hnextval
              have hnextBound : next.val ≤ limit.val := by
                have hltVal := (UScalar.lt_equiv step limit).mp hlt
                omega
              simp only [hnext, bind_tc_ok]
              refine ⟨⟨hm.symm.trans hmay, hnextBound, ?_, ?_, ?_⟩, ?_⟩
              · intro hb2t
                rcases houtcome with ⟨hmiss, _⟩ | ⟨_, jw, sw, hslot, _, hcomp⟩
                · exact absurd hb2t (by simp_all)
                · have hslot' : i6 = sw := hslot
                  subst sw
                  exact ⟨base, jw, hcomp⟩
              · intro hh
                exact absurd hh (by simp)
              · intro hflags j hj
                have hj' : j < next.val := hj
                by_cases hjl : j < step.val
                · exact hcovered ⟨hbf, hb1f⟩ j hjl
                · have heq : j = step.val := by omega
                  subst j
                  have hb2f : b2 = false := hflags.2
                  rw [hb2f] at hinner houtcome
                  exact ⟨step, rfl, group, base, finish, line, empty,
                    hgroup, hbase, hfinish, hline, hempty, hemp,
                    Or.inr ⟨bits, i2, i6, hbits, hinner, houtcome⟩⟩
              · simpa using (show limit.val - next.val < limit.val - step.val by omega)
            · have hempne : (empty != 0#u64) = true := by simpa using hemp
              simp only [if_pos hempne]
              obtain ⟨lane, hlane, hlaneBound, _, _⟩ := least_set_lane_loop_selects empty
                (by simpa using hemp)
              have hcap2 : base.val + lane.val ≤ Usize.max := by
                have hbaseCap' := hbaseCap
                have hlaneB := hlaneBound
                omega
              obtain ⟨slot, hslotAdd, hslotVal⟩ :=
                WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hcap2)
              simp only [hlane, hslotAdd, bind_tc_ok]
              have hcap : step.val + 1 ≤ Usize.max := usize_step_cap hbound' hlt
              obtain ⟨next, hnext, hnextval⟩ :=
                WP.spec_imp_exists (Usize.add_spec (x := step) (y := 1#usize) hcap)
              have hnextValue : next.val = step.val + 1 := hnextval
              have hnextBound : next.val ≤ limit.val := by
                have hltVal := (UScalar.lt_equiv step limit).mp hlt
                omega
              simp only [hnext, bind_tc_ok]
              refine ⟨⟨hm.symm.trans hmay, hnextBound, ?_,
                fun _ => ⟨base, finish, line, empty, lane,
                  hfinish, hline, hempty, (by simpa using hemp), hlane, hslotAdd⟩,
                ?_⟩, ?_⟩
              · intro hb2t
                rcases houtcome with ⟨hmiss, _⟩ | ⟨_, jw, sw, hslot, _, hcomp⟩
                · exact absurd hb2t (by simp_all)
                · have hslot' : i6 = sw := hslot
                  subst sw
                  exact ⟨base, jw, hcomp⟩
              · intro ⟨hh1, _⟩
                exact absurd hh1 (by simp)
              · simpa using (show limit.val - next.val < limit.val - step.val by omega)
        · have hmf : may = false := by cases may <;> simp_all
          simp only [hmf, Bool.false_eq_true, if_false]
          by_cases hb : b = true
          · -- vacancy stop
            simp only [hb, if_true, WP.spec_ok]
            refine ⟨fun hh => absurd hh (by simp), fun _ => hvaccert hb, ?_⟩
            intro ⟨hh1, _⟩
            exact absurd hh1 (by simp)
          · have hbf : b = false := by cases b <;> simp_all
            simp only [hbf, Bool.false_eq_true, if_false]
            rcases hready step hlt with
              ⟨group, base, finish, line, bits, empty, hgroup, hbase, hfinish, hline,
                hbits, hinnerReady, hempty, hbaseCap⟩
            simp only [hgroup, hbase, hfinish, hline, hempty, bind_tc_ok]
            by_cases hemp : empty = 0#u64
            · have hempB : (empty != 0#u64) = false := by simpa using hemp
              simp only [hempB, Bool.false_eq_true, if_false, bind_tc_ok]
              have hcap : step.val + 1 ≤ Usize.max := usize_step_cap hbound' hlt
              obtain ⟨next, hnext, hnextval⟩ :=
                WP.spec_imp_exists (Usize.add_spec (x := step) (y := 1#usize) hcap)
              have hnextValue : next.val = step.val + 1 := hnextval
              have hnextBound : next.val ≤ limit.val := by
                have hltVal := (UScalar.lt_equiv step limit).mp hlt
                omega
              simp only [hnext, bind_tc_ok]
              refine ⟨⟨hmf.symm.trans hmay, hnextBound, ?_, ?_, ?_⟩, ?_⟩
              · intro hh
                exact absurd hh (by simp)
              · intro hh
                exact absurd hh (by simp)
              · intro hflags j hj
                have hj' : j < next.val := hj
                by_cases hjl : j < step.val
                · exact hcovered ⟨hbf, hb1f⟩ j hjl
                · have heq : j = step.val := by omega
                  subst j
                  exact ⟨step, rfl, group, base, finish, line, empty,
                    hgroup, hbase, hfinish, hline, hempty, hemp, Or.inl (hmay.symm.trans hmf)⟩
              · simpa using (show limit.val - next.val < limit.val - step.val by omega)
            · have hempne : (empty != 0#u64) = true := by simpa using hemp
              simp only [if_pos hempne]
              obtain ⟨lane, hlane, hlaneBound, _, _⟩ := least_set_lane_loop_selects empty
                (by simpa using hemp)
              have hcap2 : base.val + lane.val ≤ Usize.max := by
                have hbaseCap' := hbaseCap
                have hlaneB := hlaneBound
                omega
              obtain ⟨slot, hslotAdd, hslotVal⟩ :=
                WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hcap2)
              simp only [hlane, hslotAdd, bind_tc_ok]
              have hcap : step.val + 1 ≤ Usize.max := usize_step_cap hbound' hlt
              obtain ⟨next, hnext, hnextval⟩ :=
                WP.spec_imp_exists (Usize.add_spec (x := step) (y := 1#usize) hcap)
              have hnextValue : next.val = step.val + 1 := hnextval
              have hnextBound : next.val ≤ limit.val := by
                have hltVal := (UScalar.lt_equiv step limit).mp hlt
                omega
              simp only [hnext, bind_tc_ok]
              refine ⟨⟨hmf.symm.trans hmay, hnextBound, ?_,
                fun _ => ⟨base, finish, line, empty, lane,
                  hfinish, hline, hempty, (by simpa using hemp), hlane, hslotAdd⟩,
                ?_⟩, ?_⟩
              · intro hh
                exact absurd hh (by simp)
              · intro ⟨hh1, _⟩
                exact absurd hh1 (by simp)
              · simpa using (show limit.val - next.val < limit.val - step.val by omega)
    · -- bound reached: complete the miss sweep
      have hstepEq : step.val = limit.val := by
        have hnn : ¬ step.val < limit.val := fun hh => hlt ((UScalar.lt_equiv step limit).mpr hh)
        omega
      simp only [if_neg hlt, WP.spec_ok]
      refine ⟨hmatchcert, hvaccert, ?_⟩
      intro hflags j hj
      exact hcovered hflags j (by rw [hstepEq]; omega)
  · refine ⟨rfl, ?_, ?_, ?_, ?_⟩
    · have hstep0 : ((false, 0#usize, false, 0#usize, may_match, 0#usize)).2.2.2.2.2.val = 0 :=
        rfl
      omega
    · intro hh
      exact absurd hh (by simp)
    · intro hh
      exact absurd hh (by simp)
    · intro _ j hj
      have hstep0 : ((false, 0#usize, false, 0#usize, may_match, 0#usize)).2.2.2.2.2.val = 0 :=
        rfl
      exact absurd hj (by omega)

/-! ## L5: the top-level `spill_locate` unfold -/

/-- The generated `spill_locate` reaches its outer sweep exactly when its admitted checks
pass, and a `Found index` result pins the sweep's recorded match slot to `index` and
forces the summary branch (`may_match` is whatever `summary_maybe` returned; a `Found`
can only arise from `may_match = true`, which the sweep itself then witnesses). -/
theorem generated_spill_locate_unfold {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (address : table.TableAddress)
    (key : K) (n groups mask : Usize) (index : Usize)
    (hslots : region.slots.length = groups.val * n.val)
    (hnPos : 0 < n.val)
    (hgroups2 : 2 ≤ groups.val)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hmask : groups - 1#usize = ok mask)
    (hresult : table.probe.spill_locate keyRep region sigma address key n =
      ok (table.SpillSite.Found index)) :
    ∃ wanted : U8, table.tag address.place = ok wanted ∧
      ∃ may_match : Bool, table.Region.summary_maybe region address.place = ok may_match ∧
        ∃ (b : Bool) (i1 : Usize),
          table.probe.spill_locate_loop0 keyRep region.tags region.slots sigma
            address.place address.direct_word key n groups false 0#usize false 0#usize
            mask wanted may_match 0#usize = ok (b, i1, true, index) := by
  have hzeroVal : (0#usize).val = 0 := rfl
  have hneZero : (alloc.vec.Vec.len region.slots) ≠ 0#usize := by
    intro hzero
    have hval := congrArg UScalar.val hzero
    rw [alloc.vec.Vec.len_val, hzeroVal] at hval
    have hpos : 0 < groups.val * n.val :=
      Nat.mul_pos (show 0 < groups.val from by omega) hnPos
    omega
  have hneZeroB : (alloc.vec.Vec.len region.slots != 0#usize) = true := by simpa using hneZero
  have hge : (2#usize : Usize) ≤ groups := (UScalar.le_equiv _ _).mpr (by simpa using hgroups2)
  obtain ⟨q, hqrun, hqval⟩ := WP.spec_imp_exists
    (Usize.div_spec (alloc.vec.Vec.len region.slots) (y := n)
      (by simpa using Nat.pos_iff_ne_zero.mp hnPos))
  have hqval' : q.val = (alloc.vec.Vec.len region.slots).val / n.val := by simpa using hqval
  have hqEq : q = groups := UScalar.eq_of_val_eq (by
    rw [hqval', alloc.vec.Vec.len_val, hslots, Nat.mul_div_cancel groups.val hnPos])
  subst q
  have hcap : groups.val * n.val ≤ Usize.max := by
    have hslotsProp := region.slots.property
    change region.slots.length ≤ Usize.max at hslotsProp
    omega
  obtain ⟨prod, hprodRun, hprodVal⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := groups) (y := n) hcap)
  have hprodVal' : prod.val = groups.val * n.val := by simpa using hprodVal
  have hprodEq : prod = alloc.vec.Vec.len region.slots := UScalar.eq_of_val_eq (by
    rw [hprodVal', alloc.vec.Vec.len_val, hslots])
  have hreduced :
      (do let wanted ← table.tag address.place
          let may_match ← table.Region.summary_maybe region address.place
          let (b1, i4, b2, i5) ← table.probe.spill_locate_loop0 keyRep region.tags
            region.slots sigma address.place address.direct_word key n groups
            false 0#usize false 0#usize mask wanted may_match 0#usize
          if b2
          then ok (table.SpillSite.Found i5)
          else if b1 then ok (table.SpillSite.Vacant i4) else ok table.SpillSite.Grow) =
        ok (table.SpillSite.Found index) := by
    unfold table.probe.spill_locate at hresult
    simp only [hneZeroB, hqrun, ge_iff_le, hge, hprodRun, hprodEq, hpowCheck,
      hmask, bind_tc_ok, ↓reduceIte] at hresult
    exact hresult
  cases htagres : table.tag address.place with
  | vis eff k =>
      rw [htagres] at hreduced
      simp only [bind_tc_vis] at hreduced
      exact (vis_not_ok hreduced).elim
  | div =>
      rw [htagres] at hreduced
      simp only [bind_tc_div] at hreduced
      exact (div_not_ok hreduced).elim
  | ret wanted =>
      rw [htagres, bind_tc_ok] at hreduced
      cases hsumres : table.Region.summary_maybe region address.place with
      | vis eff k =>
          rw [hsumres] at hreduced
          simp only [bind_tc_vis] at hreduced
          exact (vis_not_ok hreduced).elim
      | div =>
          rw [hsumres] at hreduced
          simp only [bind_tc_div] at hreduced
          exact (div_not_ok hreduced).elim
      | ret may_match =>
          rw [hsumres, bind_tc_ok] at hreduced
          cases hloopres : table.probe.spill_locate_loop0 keyRep region.tags
              region.slots sigma address.place address.direct_word key n groups
              false 0#usize false 0#usize mask wanted may_match 0#usize with
          | vis eff k =>
              rw [hloopres] at hreduced
              simp only [bind_tc_vis] at hreduced
              exact (vis_not_ok hreduced).elim
          | div =>
              rw [hloopres] at hreduced
              simp only [bind_tc_div] at hreduced
              exact (div_not_ok hreduced).elim
          | ret tuple =>
              rcases tuple with ⟨b0, i10, b2, i5⟩
              simp only [hloopres, bind_tc_ok] at hreduced
              have hred : (if b2 = true then ok (table.SpillSite.Found i5)
                else if b0 = true then ok (table.SpillSite.Vacant i10)
                else ok table.SpillSite.Grow) = ok (table.SpillSite.Found index) :=
                hreduced
              by_cases hb2 : b2 = true
              · rw [if_pos hb2] at hred
                have hinj : i5 = index := by
                  injection Result.ok_injective hred
                refine ⟨wanted, rfl, may_match, rfl, b0, i10, ?_⟩
                rw [hloopres, hb2, hinj]
              · rw [if_neg hb2] at hred
                by_cases hb : b0 = true
                · rw [if_pos hb] at hred
                  cases Result.ok_injective hred
                · rw [if_neg hb] at hred
                  cases Result.ok_injective hred

/-- Group-major storage with equal tag/payload extents supplies every checked range
read of the locate body, including its vacancy scan. Mirrors
`Apeiron.GrowableReadLoop.outer_step_ready_of_geometry` with two extra premises:
matcher termination stays a callback premise, and `table.scan.match_empty` totality is
stated outright because `Growable/Scan.lean` has no empty-scan spec yet. `hwide`
supplies the word-wide room the vacancy lane add needs until `match_empty` exactness
confines lanes below `n`. -/
theorem locate_outer_step_ready_of_geometry {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8) (limit step : Usize)
    (hwidth : n.val ≤ 64)
    (hwide : 64 ≤ n.val)
    (hshape : tags.length = slots.length ∧ slots.length = groups.val * n.val)
    (hcap : 7 + groups.val ≤ Usize.max)
    (hpower : ∃ exponent, groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hstep : step < limit)
    (hmatcher : ∀ entry : cell.Sigma × R × V,
      ∃ matched : Bool, table.matches keyRep entry sigma key
        { place := place, direct_word := direct } = ok matched)
    (hmatchEmpty : ∀ (line : Slice U8),
      ∃ empty : U64, table.scan.match_empty line = ok empty) :
    LocateOuterStepReady keyRep tags slots sigma place direct key n mask wanted step := by
  obtain ⟨group, hgroup, hgroupBound⟩ :=
    Apeiron.GrowableReadLoop.generated_group_bounded place groups mask limit step
      hcap hpower hmask hlimit hstep
  have hslots := slots.property
  change slots.length ≤ Usize.max at hslots
  rcases hshape with ⟨htagsEq, hslotsEq⟩
  have hgroupNext : group.val + 1 ≤ groups.val := by omega
  have hbaseCap : group.val * n.val ≤ Usize.max := by
    have hmono := Nat.mul_le_mul_right n.val (Nat.le_of_lt hgroupBound)
    omega
  obtain ⟨base, hbase, hbaseVal⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := group) (y := n) hbaseCap)
  have hbaseVal' : base.val = group.val * n.val := by simpa using hbaseVal
  have hfinishCap : base.val + n.val ≤ Usize.max := by
    have hmono := Nat.mul_le_mul_right n.val hgroupNext
    have hnext : (group.val + 1) * n.val = group.val * n.val + n.val := by ring
    rw [hbaseVal']
    omega
  obtain ⟨finish, hfinish, hfinishVal⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := base) (y := n) hfinishCap)
  have hfinishVal' : finish.val = base.val + n.val := by simpa using hfinishVal
  have hfinishBound : finish.val ≤ tags.length := by
    have hmono := Nat.mul_le_mul_right n.val hgroupNext
    have hnext : (group.val + 1) * n.val = group.val * n.val + n.val := by ring
    rw [hfinishVal', hbaseVal']
    omega
  have horder : base ≤ finish := (UScalar.le_equiv base finish).mpr (by omega)
  obtain ⟨line, hline, hlineLength⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_Range_spec tags
      { start := base, «end» := finish } horder hfinishBound)
  have hlineWidth : line.val.length = n.val := by
    have hlen := hlineLength.2
    change line.val.length = finish.val - base.val at hlen
    omega
  have hline64 : line.val.length ≤ 64 := by omega
  obtain ⟨bits, hbits, _⟩ := Apeiron.GrowableScan.match_tag_exact line wanted hline64
  have hreach : base.val + line.val.length ≤ slots.length := by
    have hmono := Nat.mul_le_mul_right n.val hgroupNext
    have hnext : (group.val + 1) * n.val = group.val * n.val + n.val := by ring
    rw [hbaseVal', hlineWidth]
    omega
  obtain ⟨empty, hemp⟩ := hmatchEmpty line
  have hvacRoom : base.val + 64 ≤ Usize.max := by
    have hmono := Nat.mul_le_mul_right n.val hgroupNext
    have hnext : (group.val + 1) * n.val = group.val * n.val + n.val := by ring
    rw [hbaseVal']
    omega
  exact ⟨group, base, finish, line, bits, empty, hgroup, hbase, hfinish, hline, hbits,
    Apeiron.GrowableReadLoop.inner_ready_of_scan_geometry keyRep slots sigma place
      direct key base line wanted bits hline64 hreach hbits hmatcher,
    hemp, hvacRoom⟩

/-! ## L6 support: run-relative found extraction

The outcome theorems above carry per-step readiness (`InnerReady`), which a `Found` run
cannot satisfy: the inner sweep stops at the matching lane and never compares the
remaining selected lanes. The certificates below therefore extract the executed match
step from the run itself, by body inversion plus a `loop.eq_def` induction on
`liveLaneCount` (inner) and on the step cursor (outer). -/

/-- Certify a recorded slot: the payload lookup succeeded at `out` and the *real*
carried-key matcher returned `true` on the returned entry. -/
def LocateSlotCertified {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (out : Usize) : Prop :=
  ∃ entry : cell.Sigma × R × V,
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V))
      slots out = ok entry ∧
    table.matches keyRep entry sigma key { place := place, direct_word := direct } = ok true

/-- Inner-body shape, by inversion: a `done` carries the state unchanged and needs the
state's flag; a `cont` from a non-recorded state either records an executed full-key
match at its slot or keeps the flag clear, and always clears one live lane. -/
theorem locate_inner_body_shape {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (v : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base i1 : Usize) (hits : U64)
    (b : Bool) (r : ControlFlow (Bool × Usize × U64) (Bool × Usize))
    (hrun : table.probe.spill_locate_loop0_loop0.body keyRep v sigma place direct key
      base b i1 hits = ok r) :
    (∃ y, r = ControlFlow.done y ∧ y = (b, i1) ∧ (hits = 0#u64 ∨ b = true)) ∨
    (b = false ∧ hits ≠ 0#u64 ∧
      ∃ (m : Bool) (out : Usize) (next : U64),
        r = ControlFlow.cont (m, out, next) ∧
        (m = true → LocateSlotCertified keyRep v sigma place direct key out) ∧
        liveLaneCount next + 1 = liveLaneCount hits) := by
  unfold table.probe.spill_locate_loop0_loop0.body at hrun
  by_cases hz : hits = 0#u64
  · subst hits
    have hzn : ¬((0#u64 != 0#u64) = true) := by simp
    rw [if_neg hzn] at hrun
    refine Or.inl ⟨(b, i1), ?_, rfl, Or.inl rfl⟩
    exact (Result.ok_injective hrun).symm
  · have hcond : (hits != 0#u64) = true := by simpa using hz
    rw [if_pos hcond] at hrun
    by_cases hb : b = true
    · rw [if_pos hb] at hrun
      refine Or.inl ⟨(true, i1), ?_, ?_, Or.inr hb⟩
      · exact (Result.ok_injective hrun).symm
      · rw [hb]
    · have hbf : b = false := by
        cases b with
        | true => exact absurd rfl hb
        | false => rfl
      rw [if_neg hb] at hrun
      cases hsel : table.least_set_lane hits with
      | vis eff k =>
          rw [hsel] at hrun
          simp only [bind_tc_vis] at hrun
          exact (vis_not_ok hrun).elim
      | div =>
          rw [hsel] at hrun
          simp only [bind_tc_div] at hrun
          exact (div_not_ok hrun).elim
      | ret lane =>
          rw [hsel, bind_tc_ok] at hrun
          cases hadd : base + lane with
          | vis eff k =>
              rw [hadd] at hrun
              simp only [bind_tc_vis] at hrun
              exact (vis_not_ok hrun).elim
          | div =>
              rw [hadd] at hrun
              simp only [bind_tc_div] at hrun
              exact (div_not_ok hrun).elim
          | ret slot' =>
              rw [hadd, bind_tc_ok] at hrun
              cases hidx : alloc.vec.Vec.index
                  (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) v slot' with
              | vis eff k =>
                  rw [hidx] at hrun
                  simp only [bind_tc_vis] at hrun
                  exact (vis_not_ok hrun).elim
              | div =>
                  rw [hidx] at hrun
                  simp only [bind_tc_div] at hrun
                  exact (div_not_ok hrun).elim
              | ret entry =>
                  rw [hidx, bind_tc_ok] at hrun
                  cases hmatch : table.matches keyRep entry sigma key
                      { place := place, direct_word := direct } with
                  | vis eff k =>
                      rw [hmatch] at hrun
                      simp only [bind_tc_vis] at hrun
                      exact (vis_not_ok hrun).elim
                  | div =>
                      rw [hmatch] at hrun
                      simp only [bind_tc_div] at hrun
                      exact (div_not_ok hrun).elim
                  | ret matched =>
                      rw [hmatch, bind_tc_ok] at hrun
                      have hif : (if matched then ok slot' else ok i1) =
                          ok (if matched then slot' else i1) := by
                        cases matched <;> simp
                      rw [hif, bind_tc_ok] at hrun
                      cases hsub : hits - 1#u64 with
                      | vis eff k =>
                          rw [hsub] at hrun
                          simp only [bind_tc_vis] at hrun
                          exact (vis_not_ok hrun).elim
                      | div =>
                          rw [hsub] at hrun
                          simp only [bind_tc_div] at hrun
                          exact (div_not_ok hrun).elim
                      | ret i3 =>
                          rw [hsub, bind_tc_ok] at hrun
                          have hlift : lift (hits &&& i3) = ok (hits &&& i3) := rfl
                          rw [hlift, bind_tc_ok] at hrun
                          have hcount : liveLaneCount (hits &&& i3) + 1 = liveLaneCount hits := by
                            have hscan : table.scan.least_set_lane hits = ok lane :=
                              scan_lane_of_loop_lane hz hsel
                            obtain ⟨next', hclearRun, _, hcount', _⟩ :=
                              Apeiron.GrowableScan.generated_clear_hits_live_decrease
                                hits lane hz hscan
                            rw [hsub, bind_tc_ok] at hclearRun
                            have hlift' : lift (hits &&& i3) = ok (hits &&& i3) := rfl
                            rw [hlift'] at hclearRun
                            have hnext' : next' = hits &&& i3 :=
                              (Result.ok_injective hclearRun).symm
                            rw [hnext'] at hcount'
                            exact hcount'
                          refine Or.inr ⟨hbf, hz, matched, (if matched then slot' else i1),
                            hits &&& i3, ?_, ?_, hcount⟩
                          · exact (Result.ok_injective hrun).symm
                          · intro hm
                            subst matched
                            exact ⟨entry, hidx, hmatch⟩

/-! ## L6 chain: run-relative found extraction (steps 2-5) -/

/-- The generated inner-loop wrapper equals its `loop` presentation with a projection
lambda. This is the bridge that keeps the `loop.eq_def` residue rewritable. -/
theorem wrapper_loop_form {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (b : Bool) (i1 : Usize) (mask : U64) :
    table.probe.spill_locate_loop0_loop0 keyRep slots sigma place direct key
      b i1 base mask =
      loop (fun x => table.probe.spill_locate_loop0_loop0.body keyRep slots sigma
        place direct key base x.1 x.2.1 x.2.2) (b, i1, mask) := by
  unfold table.probe.spill_locate_loop0_loop0
  rfl

/-- Inner-loop found extraction: an inner run that recorded a match certifies its slot.
The induction is over `liveLaneCount`; each `cont` clears exactly one lane. -/
theorem generated_locate_inner_found {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base prior : Usize) (hits : U64)
    (out : Usize)
    (hrun : table.probe.spill_locate_loop0_loop0 keyRep slots sigma place direct key
      false prior base hits = ok (true, out)) :
    LocateSlotCertified keyRep slots sigma place direct key out := by
  have hall : ∀ (k : Nat) (b : Bool) (i1 : Usize) (mask : U64),
      liveLaneCount mask = k →
      (b = true → LocateSlotCertified keyRep slots sigma place direct key i1) →
      ∀ result : Bool × Usize,
        table.probe.spill_locate_loop0_loop0 keyRep slots sigma place direct key
          b i1 base mask = ok result →
        result.1 = true → LocateSlotCertified keyRep slots sigma place direct key result.2 := by
    intro k
    induction k with
    | zero =>
        intro b i1 mask hk hcarry result hres hfnd
        rw [wrapper_loop_form] at hres
        rw [loop.eq_def] at hres
        have hresA : (Std.bind
            (table.probe.spill_locate_loop0_loop0.body keyRep slots sigma place direct
              key base b i1 mask)
            (fun r => match r with
              | ControlFlow.cont x =>
                  table.probe.spill_locate_loop0_loop0 keyRep slots sigma place
                    direct key x.1 x.2.1 base x.2.2
              | ControlFlow.done y => ok y)) = ok result := by
          convert hres using 2
          funext r'
          cases r' with
          | cont st => rcases st with ⟨b3, i3, m3⟩; rfl
          | done y => rfl
        have hres2 : (do
            let y ← table.probe.spill_locate_loop0_loop0.body keyRep slots sigma
              place direct key base b i1 mask
            match y with
            | ControlFlow.cont x =>
                table.probe.spill_locate_loop0_loop0 keyRep slots sigma place
                  direct key x.1 x.2.1 base x.2.2
            | ControlFlow.done y => ok y) = ok result := hresA
        clear hresA
        cases hbody : table.probe.spill_locate_loop0_loop0.body keyRep slots sigma
            place direct key base b i1 mask with
        | vis eff k2 =>
            rw [hbody] at hres2
            rw [bind_tc_vis] at hres2
            exact (vis_not_ok hres2).elim
        | div =>
            rw [hbody] at hres2
            rw [bind_tc_div] at hres2
            exact (div_not_ok hres2).elim
        | ret r =>
            rw [hbody] at hres2
            rw [bind_tc_ok] at hres2
            cases r with
            | done y =>
                have hred : ok y = ok result := hres2
                have hy : y = result := Result.ok_injective hred
                subst result
                obtain ⟨y', hrdone, hyeq, _hflag⟩ | ⟨_hbf, _hne, _m, _outp, _next, hrcont, _hc, _hcount⟩ :=
                  locate_inner_body_shape keyRep slots sigma place direct key base
                    i1 mask b (ControlFlow.done y) hbody
                · have hy' : y = y' := by injection hrdone
                  have hyeq2 : y = (b, i1) := hy'.trans hyeq
                  rw [hyeq2] at hfnd ⊢
                  exact hcarry hfnd
                · cases hrcont
            | cont st =>
                rcases st with ⟨b3, i3, m3⟩
                have hres4 : table.probe.spill_locate_loop0_loop0 keyRep slots sigma
                    place direct key b3 i3 base m3 = ok result := hres2
                rcases locate_inner_body_shape keyRep slots sigma place direct key base
                    i1 mask b (ControlFlow.cont (b3, i3, m3)) hbody with
                  ⟨y, hrdone, _, _⟩ | ⟨_hbf, _hne, m, out', next, hrcont, hcert, hcount⟩
                · cases hrdone
                · have hp : (b3, i3, m3) = (m, out', next) := by
                    cases hrcont
                    rfl
                  have hp1 : b3 = m := congrArg Prod.fst hp
                  have hp2 : i3 = out' := congrArg (fun z => Prod.fst (Prod.snd z)) hp
                  have hp3 : m3 = next := congrArg (fun z => Prod.snd (Prod.snd z)) hp
                  subst m
                  subst out'
                  subst next

                  have hfalse : False := by omega
                  exact hfalse.elim
    | succ k ih =>
        intro b i1 mask hk hcarry result hres hfnd
        rw [wrapper_loop_form] at hres
        rw [loop.eq_def] at hres
        have hresA : (Std.bind
            (table.probe.spill_locate_loop0_loop0.body keyRep slots sigma place direct
              key base b i1 mask)
            (fun r => match r with
              | ControlFlow.cont x =>
                  table.probe.spill_locate_loop0_loop0 keyRep slots sigma place
                    direct key x.1 x.2.1 base x.2.2
              | ControlFlow.done y => ok y)) = ok result := by
          convert hres using 2
          funext r'
          cases r' with
          | cont st => rcases st with ⟨b3, i3, m3⟩; rfl
          | done y => rfl
        have hres2 : (do
            let y ← table.probe.spill_locate_loop0_loop0.body keyRep slots sigma
              place direct key base b i1 mask
            match y with
            | ControlFlow.cont x =>
                table.probe.spill_locate_loop0_loop0 keyRep slots sigma place
                  direct key x.1 x.2.1 base x.2.2
            | ControlFlow.done y => ok y) = ok result := hresA
        clear hresA
        cases hbody : table.probe.spill_locate_loop0_loop0.body keyRep slots sigma
            place direct key base b i1 mask with
        | vis eff k2 =>
            rw [hbody] at hres2
            rw [bind_tc_vis] at hres2
            exact (vis_not_ok hres2).elim
        | div =>
            rw [hbody] at hres2
            rw [bind_tc_div] at hres2
            exact (div_not_ok hres2).elim
        | ret r =>
            rw [hbody] at hres2
            rw [bind_tc_ok] at hres2
            cases r with
            | done y =>
                have hred : ok y = ok result := hres2
                have hy : y = result := Result.ok_injective hred
                subst result
                obtain ⟨y', hrdone, hyeq, _hflag⟩ | ⟨_hbf, _hne, _m, _outp, _next, hrcont, _hc, _hcount⟩ :=
                  locate_inner_body_shape keyRep slots sigma place direct key base
                    i1 mask b (ControlFlow.done y) hbody
                · have hy' : y = y' := by injection hrdone
                  have hyeq2 : y = (b, i1) := hy'.trans hyeq
                  rw [hyeq2] at hfnd ⊢
                  exact hcarry hfnd
                · cases hrcont
            | cont st =>
                rcases st with ⟨b3, i3, m3⟩
                have hres4 : table.probe.spill_locate_loop0_loop0 keyRep slots sigma
                    place direct key b3 i3 base m3 = ok result := hres2
                rcases locate_inner_body_shape keyRep slots sigma place direct key base
                    i1 mask b (ControlFlow.cont (b3, i3, m3)) hbody with
                  ⟨y, hrdone, _, _⟩ | ⟨_hbf, _hne, m, out', next, hrcont, hcert, hcount⟩
                · cases hrdone
                · have hp : (b3, i3, m3) = (m, out', next) := by
                    cases hrcont
                    rfl
                  have hp1 : b3 = m := congrArg Prod.fst hp
                  have hp2 : i3 = out' := congrArg (fun z => Prod.fst (Prod.snd z)) hp
                  have hp3 : m3 = next := congrArg (fun z => Prod.snd (Prod.snd z)) hp
                  subst m
                  subst out'
                  subst next

                  have hk' : liveLaneCount m3 = k := by omega
                  exact ih b3 i3 m3 hk' hcert result hres4 hfnd
  exact hall (liveLaneCount hits) false prior hits rfl
    (fun hh => Bool.noConfusion hh) (true, out) hrun rfl

/-! ## L6 chain: outer body inversion, outer extraction, run-relative unfold -/

/-- Generic bind inversion on the generated `do` form: a successful bind ran its head
to `ok`. This absorbs the `ret/vis/div` boilerplate of every generated case chain. -/
private theorem bind_ret_inv {α β : Type} (x : Result α) (f : α → Result β) (r : β)
    (hrun : (do let y ← x; f y) = ok r) : ∃ y, x = ok y ∧ f y = ok r := by
  cases hx : x with
  | vis eff k =>
      rw [hx] at hrun
      simp only [bind_tc_vis] at hrun
      exact (vis_not_ok hrun).elim
  | div =>
      rw [hx] at hrun
      simp only [bind_tc_div] at hrun
      exact (div_not_ok hrun).elim
  | ret y =>
      rw [hx] at hrun
      simp only [bind_tc_ok] at hrun
      exact ⟨y, rfl, hrun⟩

/-- The generated outer-loop wrapper equals its `loop` presentation with a projection
lambda. -/
theorem outer_wrapper_loop_form {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8)
    (b : Bool) (i1 : Usize) (b1 : Bool) (i2 : Usize) (may : Bool) (step : Usize) :
    table.probe.spill_locate_loop0 keyRep tags slots sigma place direct key n groups
      b i1 b1 i2 mask wanted may step =
      loop (fun x => table.probe.spill_locate_loop0.body keyRep tags slots sigma place
        direct key n groups mask wanted x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1
        x.2.2.2.2.2) (b, i1, b1, i2, may, step) := by
  unfold table.probe.spill_locate_loop0
  rfl

/-- Outer-body shape, by inversion: a `done` carries the state unchanged; a `cont`
advances the cursor by exactly one and, when it records a match, certifies its slot. -/
theorem locate_outer_body_shape {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8)
    (b : Bool) (i1 : Usize) (b1 : Bool) (i2 : Usize) (may : Bool) (step : Usize)
    (r : ControlFlow (Bool × Usize × Bool × Usize × Bool × Usize)
      (Bool × Usize × Bool × Usize))
    (hrun : table.probe.spill_locate_loop0.body keyRep tags slots sigma place direct
      key n groups mask wanted b i1 b1 i2 may step = ok r) :
    (∃ y, r = ControlFlow.done y ∧ y = (b, i1, b1, i2)) ∨
    (∃ (x' : Bool × Usize × Bool × Usize × Bool × Usize),
      r = ControlFlow.cont x' ∧
      x'.2.2.2.2.2.val = step.val + 1 ∧
      (x'.2.2.1 = true →
        LocateSlotCertified keyRep slots sigma place direct key x'.2.2.2.1)) := by
  unfold table.probe.spill_locate_loop0.body at hrun
  obtain ⟨i3, hlim, hrun⟩ := bind_ret_inv _ _ _ hrun
  by_cases hlt : step < i3
  · rw [if_pos hlt] at hrun
    by_cases hb1 : b1 = true
    · rw [if_pos hb1] at hrun
      refine Or.inl ⟨(b, i1, true, i2), (Result.ok_injective hrun).symm, ?_⟩
      rw [hb1]
    · rw [if_neg hb1] at hrun
      by_cases hm : may = true
      · rw [if_pos hm] at hrun
        obtain ⟨i4, hgroup, hrun⟩ := bind_ret_inv _ _ _ hrun
        obtain ⟨base, hmul, hrun⟩ := bind_ret_inv _ _ _ hrun
        obtain ⟨i5, hadd1, hrun⟩ := bind_ret_inv _ _ _ hrun
        obtain ⟨s, hidx, hrun⟩ := bind_ret_inv _ _ _ hrun
        obtain ⟨hits, htag, hrun⟩ := bind_ret_inv _ _ _ hrun
        obtain ⟨p, hinner, hrun⟩ := bind_ret_inv _ _ _ hrun
        rcases p with ⟨b2, i6⟩
        by_cases hb : b = true
        · rw [if_pos hb] at hrun
          obtain ⟨pv, hpv, hrunT⟩ := bind_ret_inv _ _ _ hrun
          rcases pv with ⟨b3, i7⟩
          have hrunU : ((do
              let step1 ← step + 1#usize
              ok (ControlFlow.cont (b3, i7, b2, i6, true, step1))) :
              Result (ControlFlow (Bool × Usize × Bool × Usize × Bool × Usize) (Bool × Usize × Bool × Usize))) = ok r := hrunT
          obtain ⟨s1, hstep, hrun⟩ := bind_ret_inv _ _ _ hrunU
          have hs1 : s1.val = step.val + 1 := by
            have hadd := UScalar.add_equiv step (1#usize)
            rw [hstep] at hadd
            simpa using hadd.2.1
          have hred : ok (ControlFlow.cont (b3, i7, b2, i6, true, s1)) = ok r := hrun
          refine Or.inr ⟨(b3, i7, b2, i6, true, s1),
            (Result.ok_injective hred).symm, by simpa using hs1, ?_⟩
          intro hh
          have hb2 : b2 = true := hh
          rw [hb2] at hinner
          exact generated_locate_inner_found keyRep slots sigma place direct key base
            i2 hits i6 hinner
        · rw [if_neg hb] at hrun
          obtain ⟨pv, hpv, hrunT⟩ := bind_ret_inv _ _ _ hrun
          rcases pv with ⟨b3, i7⟩
          have hrunU : ((do
              let step1 ← step + 1#usize
              ok (ControlFlow.cont (b3, i7, b2, i6, true, step1))) :
              Result (ControlFlow (Bool × Usize × Bool × Usize × Bool × Usize) (Bool × Usize × Bool × Usize))) = ok r := hrunT
          obtain ⟨s1, hstep, hrun⟩ := bind_ret_inv _ _ _ hrunU
          have hs1 : s1.val = step.val + 1 := by
            have hadd := UScalar.add_equiv step (1#usize)
            rw [hstep] at hadd
            simpa using hadd.2.1
          have hred : ok (ControlFlow.cont (b3, i7, b2, i6, true, s1)) = ok r := hrun
          refine Or.inr ⟨(b3, i7, b2, i6, true, s1),
            (Result.ok_injective hred).symm, by simpa using hs1, ?_⟩
          intro hh
          have hb2 : b2 = true := hh
          rw [hb2] at hinner
          exact generated_locate_inner_found keyRep slots sigma place direct key base
            i2 hits i6 hinner
      · rw [if_neg hm] at hrun
        by_cases hb : b = true
        · rw [if_pos hb] at hrun
          refine Or.inl ⟨(true, i1, false, i2), (Result.ok_injective hrun).symm, ?_⟩
          rw [hb]
          have hb1f : b1 = false := by
            cases b1 with
            | true => exact absurd rfl hb1
            | false => rfl
          rw [hb1f]
        · rw [if_neg hb] at hrun
          obtain ⟨i4, hgroup, hrun⟩ := bind_ret_inv _ _ _ hrun
          obtain ⟨base, hmul, hrun⟩ := bind_ret_inv _ _ _ hrun
          obtain ⟨i5, hadd1, hrun⟩ := bind_ret_inv _ _ _ hrun
          obtain ⟨s, hidx, hrun⟩ := bind_ret_inv _ _ _ hrun
          obtain ⟨empty, hemp, hrun⟩ := bind_ret_inv _ _ _ hrun
          obtain ⟨pv, hpv, hrunT⟩ := bind_ret_inv _ _ _ hrun
          rcases pv with ⟨b3, i7⟩
          have hrunU : ((do
              let step1 ← step + 1#usize
              ok (ControlFlow.cont (b3, i7, false, i2, false, step1))) :
              Result (ControlFlow (Bool × Usize × Bool × Usize × Bool × Usize) (Bool × Usize × Bool × Usize))) = ok r := hrunT
          obtain ⟨s1, hstep, hrun⟩ := bind_ret_inv _ _ _ hrunU
          have hs1 : s1.val = step.val + 1 := by
            have hadd := UScalar.add_equiv step (1#usize)
            rw [hstep] at hadd
            simpa using hadd.2.1
          have hred : ok (ControlFlow.cont (b3, i7, false, i2, false, s1)) = ok r := hrun
          refine Or.inr ⟨(b3, i7, false, i2, false, s1),
            (Result.ok_injective hred).symm, by simpa using hs1, ?_⟩
          intro hh
          exact Bool.noConfusion hh
  · rw [if_neg hlt] at hrun
    refine Or.inl ⟨(b, i1, b1, i2), (Result.ok_injective hrun).symm, rfl⟩

/-- Outer found extraction: an outer run whose result records a match certifies the
returned slot. The induction is on `Usize.max - step.val`; each `cont` advances the
cursor by exactly one. -/
theorem generated_locate_outer_found {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8) (may_match : Bool)
    (b : Bool) (i1 : Usize) (index : Usize)
    (hrun : table.probe.spill_locate_loop0 keyRep tags slots sigma place direct key n
      groups false 0#usize false 0#usize mask wanted may_match 0#usize =
      ok (b, i1, true, index)) :
    LocateSlotCertified keyRep slots sigma place direct key index := by
  have hall : ∀ (k : Nat) (b : Bool) (i1 : Usize) (b1 : Bool) (i2 : Usize)
      (may : Bool) (step : Usize),
      Usize.max - step.val = k →
      (b1 = true → LocateSlotCertified keyRep slots sigma place direct key i2) →
      ∀ result : Bool × Usize × Bool × Usize,
        table.probe.spill_locate_loop0 keyRep tags slots sigma place direct key n
          groups b i1 b1 i2 mask wanted may step = ok result →
        result.2.2.1 = true →
          LocateSlotCertified keyRep slots sigma place direct key result.2.2.2 := by
    intro k
    induction k with
    | zero =>
        intro b i1 b1 i2 may step hk hcarry result hres hfnd
        rw [outer_wrapper_loop_form] at hres
        rw [loop.eq_def] at hres
        have hresA : (Std.bind
            (table.probe.spill_locate_loop0.body keyRep tags slots sigma place direct
              key n groups mask wanted b i1 b1 i2 may step)
            (fun r => match r with
              | ControlFlow.cont x =>
                  table.probe.spill_locate_loop0 keyRep tags slots sigma place
                    direct key n groups x.1 x.2.1 x.2.2.1 x.2.2.2.1 mask wanted
                    x.2.2.2.2.1 x.2.2.2.2.2
              | ControlFlow.done x => ok x)) = ok result := by
          convert hres using 2
          funext r'
          cases r' with
          | cont st => rcases st with ⟨b3, i3, b4, i4, b5, i5⟩; rfl
          | done y => rfl
        have hres2 : ((do
            let y ← table.probe.spill_locate_loop0.body keyRep tags slots sigma place
              direct key n groups mask wanted b i1 b1 i2 may step
            match y with
            | ControlFlow.cont x =>
                table.probe.spill_locate_loop0 keyRep tags slots sigma place direct
                  key n groups x.1 x.2.1 x.2.2.1 x.2.2.2.1 mask wanted x.2.2.2.2.1
                  x.2.2.2.2.2
            | ControlFlow.done y => ok y) : Result (Bool × Usize × Bool × Usize)) =
          ok result := hresA
        obtain ⟨y, hbody, hrest⟩ := bind_ret_inv _ _ _ hres2
        cases y with
        | done yv =>
            have hred : ok yv = ok result := hrest
            have hy : yv = result := Result.ok_injective hred
            subst result
            rcases locate_outer_body_shape keyRep tags slots sigma place direct key n
                groups mask wanted b i1 b1 i2 may step (ControlFlow.done yv) hbody with
              ⟨y', hrdone, hyeq⟩ | ⟨x', hrcont, _, _⟩
            · have hy' : yv = y' := by injection hrdone
              have hyeq2 : yv = (b, i1, b1, i2) := hy'.trans hyeq
              rw [hyeq2] at hfnd ⊢
              exact hcarry hfnd
            · cases hrcont
        | cont st =>
            rcases st with ⟨b3, i3, b4, i4, b5, i5⟩
            have hres4 : table.probe.spill_locate_loop0 keyRep tags slots sigma place
                direct key n groups b3 i3 b4 i4 mask wanted b5 i5 = ok result := hrest
            rcases locate_outer_body_shape keyRep tags slots sigma place direct key n
                groups mask wanted b i1 b1 i2 may step
                (ControlFlow.cont (b3, i3, b4, i4, b5, i5)) hbody with
              ⟨y', hrdone, _⟩ | ⟨x', hrcont, hs1, hcert⟩
            · cases hrdone
            · have hp : (b3, i3, b4, i4, b5, i5) = x' := by
                cases hrcont
                rfl
              subst x'
              have hmax : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
                simp [Usize.max, Usize.numBits]
              have hbound' := i5.hBounds
              simp only [UScalarTy.Usize_numBits_eq] at hbound'
              have hfalse : False := by
                have hk2 : (Usize.max : Nat) - step.val = 0 := hk
                have hs2 : i5.val = step.val + 1 := hs1
                have hb2 : i5.val < 2 ^ System.Platform.numBits := by simpa using hbound'
                omega
              exact hfalse.elim
    | succ k ih =>
        intro b i1 b1 i2 may step hk hcarry result hres hfnd
        rw [outer_wrapper_loop_form] at hres
        rw [loop.eq_def] at hres
        have hresA : (Std.bind
            (table.probe.spill_locate_loop0.body keyRep tags slots sigma place direct
              key n groups mask wanted b i1 b1 i2 may step)
            (fun r => match r with
              | ControlFlow.cont x =>
                  table.probe.spill_locate_loop0 keyRep tags slots sigma place
                    direct key n groups x.1 x.2.1 x.2.2.1 x.2.2.2.1 mask wanted
                    x.2.2.2.2.1 x.2.2.2.2.2
              | ControlFlow.done x => ok x)) = ok result := by
          convert hres using 2
          funext r'
          cases r' with
          | cont st => rcases st with ⟨b3, i3, b4, i4, b5, i5⟩; rfl
          | done y => rfl
        have hres2 : ((do
            let y ← table.probe.spill_locate_loop0.body keyRep tags slots sigma place
              direct key n groups mask wanted b i1 b1 i2 may step
            match y with
            | ControlFlow.cont x =>
                table.probe.spill_locate_loop0 keyRep tags slots sigma place direct
                  key n groups x.1 x.2.1 x.2.2.1 x.2.2.2.1 mask wanted x.2.2.2.2.1
                  x.2.2.2.2.2
            | ControlFlow.done y => ok y) : Result (Bool × Usize × Bool × Usize)) =
          ok result := hresA
        obtain ⟨y, hbody, hrest⟩ := bind_ret_inv _ _ _ hres2
        cases y with
        | done yv =>
            have hred : ok yv = ok result := hrest
            have hy : yv = result := Result.ok_injective hred
            subst result
            rcases locate_outer_body_shape keyRep tags slots sigma place direct key n
                groups mask wanted b i1 b1 i2 may step (ControlFlow.done yv) hbody with
              ⟨y', hrdone, hyeq⟩ | ⟨x', hrcont, _, _⟩
            · have hy' : yv = y' := by injection hrdone
              have hyeq2 : yv = (b, i1, b1, i2) := hy'.trans hyeq
              rw [hyeq2] at hfnd ⊢
              exact hcarry hfnd
            · cases hrcont
        | cont st =>
            rcases st with ⟨b3, i3, b4, i4, b5, i5⟩
            have hres4 : table.probe.spill_locate_loop0 keyRep tags slots sigma place
                direct key n groups b3 i3 b4 i4 mask wanted b5 i5 = ok result := hrest
            rcases locate_outer_body_shape keyRep tags slots sigma place direct key n
                groups mask wanted b i1 b1 i2 may step
                (ControlFlow.cont (b3, i3, b4, i4, b5, i5)) hbody with
              ⟨y', hrdone, _⟩ | ⟨x', hrcont, hs1, hcert⟩
            · cases hrdone
            · have hp : (b3, i3, b4, i4, b5, i5) = x' := by
                cases hrcont
                rfl
              subst x'
              have hk' : Usize.max - i5.val = k := by
                have hk2 : (Usize.max : Nat) - step.val = k + 1 := hk
                have hs2 : i5.val = step.val + 1 := hs1
                omega
              exact ih b3 i3 b4 i4 b5 i5 hk' hcert result hres4 hfnd
  have hstart : (false : Bool) = true →
      LocateSlotCertified keyRep slots sigma place direct key 0#usize := by
    intro hh
    exact Bool.noConfusion hh
  exact hall (Usize.max - 0) false 0#usize false 0#usize may_match 0#usize rfl hstart
    (b, i1, true, index) hrun rfl

/-- Run-relative unfold: a `Found` result pins the outer sweep's recorded match slot. -/
theorem generated_spill_locate_unfold_run {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (region : table.Region R V) (sigma : cell.Sigma) (address : table.TableAddress)
    (key : K) (n : Usize) (index : Usize)
    (hresult : table.probe.spill_locate keyRep region sigma address key n =
      ok (table.SpillSite.Found index)) :
    ∃ (wanted : U8) (may_match : Bool) (groups mask : Usize) (b : Bool) (i1 : Usize),
      table.tag address.place = ok wanted ∧
      table.Region.summary_maybe region address.place = ok may_match ∧
      table.probe.spill_locate_loop0 keyRep region.tags region.slots sigma
        address.place address.direct_word key n groups false 0#usize false 0#usize
        mask wanted may_match 0#usize = ok (b, i1, true, index) := by
  unfold table.probe.spill_locate at hresult
  by_cases hc1 : (alloc.vec.Vec.len region.slots != 0#usize) = true
  · simp only [hc1, ↓reduceIte] at hresult
    obtain ⟨groups, hdiv, hresult⟩ := bind_ret_inv _ _ _ hresult
    by_cases hc2 : groups ≥ 2#usize
    · rw [if_pos hc2] at hresult
      obtain ⟨prod, hmul, hresult⟩ := bind_ret_inv _ _ _ hresult
      by_cases hc3 : prod = alloc.vec.Vec.len region.slots
      · rw [if_pos hc3] at hresult
        obtain ⟨pow, hpow, hresult⟩ := bind_ret_inv _ _ _ hresult
        by_cases hc4 : pow = true
        · simp only [hc4, ↓reduceIte] at hresult
          obtain ⟨mask, hmask, hresult⟩ := bind_ret_inv _ _ _ hresult
          obtain ⟨wanted, htag, hresult⟩ := bind_ret_inv _ _ _ hresult
          obtain ⟨may_match, hsum, hresult⟩ := bind_ret_inv _ _ _ hresult
          obtain ⟨p, hloop, hresult⟩ := bind_ret_inv _ _ _ hresult
          rcases p with ⟨b1, i4, b2, i5⟩
          have hred : (if b2 = true then ok (table.SpillSite.Found i5)
            else if b1 = true then ok (table.SpillSite.Vacant i4)
            else ok table.SpillSite.Grow) = ok (table.SpillSite.Found index) := hresult
          by_cases hb2 : b2 = true
          · simp only [hb2, ↓reduceIte] at hred
            have hinj : i5 = index := by injection Result.ok_injective hred
            subst i5
            rw [hb2] at hloop
            exact ⟨wanted, may_match, groups, mask, b1, i4, htag, hsum, hloop⟩
          · rw [if_neg hb2] at hred
            by_cases hb1 : b1 = true
            · simp only [hb1, ↓reduceIte] at hred
              cases Result.ok_injective hred
            · rw [if_neg hb1] at hred
              cases Result.ok_injective hred
        · rw [if_neg hc4] at hresult
          cases Result.ok_injective hresult
      · rw [if_neg hc3] at hresult
        cases Result.ok_injective hresult
    · rw [if_neg hc2] at hresult
      cases Result.ok_injective hresult
  · rw [if_neg hc1] at hresult
    cases Result.ok_injective hresult

end Apeiron.GrowableLocateLoop
