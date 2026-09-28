import Growable.Representation
import Growable.ReadState
import Growable.Resources
import Lean

/-!
# Generated spill-read sweep: bounded step facts

These facts concern the extracted spill-read loop bodies, not `Table.get_at`.
The conditional outer-step sweep and tail surjectivity do not establish
`OuterStepReady` from the table representation or a public read refinement.
-/

namespace Apeiron.GrowableReadLoop

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- A preexisting full match terminates the generated outer body without inspecting another group. -/
theorem generated_outer_stops_on_hit {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (tags : alloc.vec.Vec U8)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (n groups mask : Usize)
    (wanted : U8) (index step : Usize)
    (hbound : ∃ limit, table.WINDOWS + groups = ok limit ∧ step < limit) :
    table.probe.spill_read_loop0.body keyRep tags slots sigma place direct key n
      groups mask wanted (some index) step = ok (ControlFlow.done (some index)) := by
  rcases hbound with ⟨limit, hlimit, hstep⟩
  unfold table.probe.spill_read_loop0.body
  simp only [hlimit, bind_tc_ok, if_pos hstep, core.option.Option.is_none,
    Option.isNone_some, Bool.false_eq_true, if_false]

/-- Exhausting the generated bound returns the accumulated result; no tag hit is synthesized. -/
theorem generated_outer_stops_at_limit {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (tags : alloc.vec.Vec U8)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (n groups mask : Usize)
    (wanted : U8) (hit : Option Usize) (step limit : Usize)
    (hlimit : table.WINDOWS + groups = ok limit) (hstep : ¬ step < limit) :
    table.probe.spill_read_loop0.body keyRep tags slots sigma place direct key n
      groups mask wanted hit step = ok (ControlFlow.done hit) := by
  unfold table.probe.spill_read_loop0.body
  simp only [hlimit, bind_tc_ok, if_neg hstep]

/-- The generated increment succeeds whenever the outer guard holds; the
successful next step stays at or below the exact checked loop bound. -/
theorem generated_outer_increment_checked (step limit : Usize)
    (hstep : step < limit) :
    ∃ next : Usize, step + 1#usize = ok next ∧
      next.val = step.val + 1 ∧ next.val ≤ limit.val := by
  have hlt := (UScalar.lt_equiv step limit).mp hstep
  have hlimitCap : limit.val ≤ Usize.max := by
    have hbound := limit.hBounds
    simp only [UScalarTy.Usize_numBits_eq] at hbound
    have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    rw [hmax]
    omega
  have hone : (1#usize).val = 1 := rfl
  obtain ⟨next, hnext, hnextVal⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := step) (y := 1#usize) (by omega))
  refine ⟨next, hnext, ?_, ?_⟩
  · simpa using hnextVal
  · omega

/-- All tail-step indices, including the last, satisfy the actual generated outer-loop guard.
This establishes eligibility for a step, not execution of its group lookup or absence completeness. -/
theorem generated_tail_indices_eligible (groups : Usize)
    (hcap : 7 + groups.val ≤ Usize.max) :
    ∃ limit : Usize, table.WINDOWS + groups = ok limit ∧
      ∀ j : Nat, j < groups.val →
        ∃ step : Usize, step.val = 7 + j ∧ step < limit :=
  Apeiron.GrowableScan.generated_tail_steps_in_bound groups hcap

/-- Both generated prefix operations of `table.group` complete before either branch. -/
private theorem generated_group_head_ok (place : U64) :
    ∃ (high word : U64), place >>> 8#i32 = ok high ∧
      table.DESCENT_MASK = ok word := by
  obtain ⟨high, hhigh, _⟩ := WP.spec_imp_exists
    (U64.ShiftRight_IScalar_spec place 8#i32 (by norm_num) (by norm_num))
  obtain ⟨word, hword, _⟩ := WP.spec_imp_exists
    (show table.DESCENT_MASK ⦃ _ => True ⦄ from by
      unfold table.DESCENT_MASK
      step
      step)
  exact ⟨high, word, hhigh, hword⟩

/-- The tail of the real generated `table.group` uses a fixed descent seed and
the checked offset `step - WINDOWS + 1`; the final add is a *wrapping* word add. -/
private theorem generated_group_tail_step (place high word : U64) (step mask : Usize)
    (hhigh : place >>> 8#i32 = ok high) (hword : table.DESCENT_MASK = ok word)
    (htail : ¬ step < table.WINDOWS)
    (hroom : step.val - 7 + 1 ≤ Usize.max) :
    ∃ offset : Usize,
      table.group place step mask =
        ok ((core.num.Usize.wrapping_add
          ((UScalar.cast .Usize (high &&& word)) &&& mask) offset) &&& mask) ∧
      offset.val = step.val - 7 + 1 := by
  have hsub : 7 ≤ step.val := by
    have hwindows : table.WINDOWS.val = 7 := by simp [table.WINDOWS]
    have hnot : ¬ step.val < table.WINDOWS.val := by
      intro hlt
      exact htail ((UScalar.lt_equiv step table.WINDOWS).mpr hlt)
    omega
  obtain ⟨difference, hdifference, hdifferenceVal⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := step) (y := table.WINDOWS) (by simpa [table.WINDOWS] using hsub))
  obtain ⟨offset, hoffset, hoffsetVal⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := difference) (y := 1#usize) (by
      rw [hdifferenceVal.1]
      simpa [table.WINDOWS] using hroom))
  refine ⟨offset, ?_, ?_⟩
  · unfold table.group
    simp only [hhigh, hword, bind_tc_ok, lift, if_neg htail,
      hdifference, hoffset]
  · simpa [table.WINDOWS, hdifferenceVal.1] using hoffsetVal

private theorem generated_group_prefix_step (place high word : U64) (step mask : Usize)
    (hhigh : place >>> 8#i32 = ok high) (hword : table.DESCENT_MASK = ok word)
    (hstep : step < table.WINDOWS) :
    ∃ group : Usize, table.group place step mask = ok (group &&& mask) := by
  have hstepVal : step.val < 7 := by
    have hlt := (UScalar.lt_equiv step table.WINDOWS).mp hstep
    simpa [table.WINDOWS] using hlt
  have hmax : 64 ≤ Usize.max := by
    rcases Usize.bounds_eq with h | h <;> rw [h] <;>
      norm_num [U32.max, U64.max, U32.numBits, U64.numBits, UScalarTy.numBits]
  have hone : (1#usize).val = 1 := rfl
  have hseven : (7#usize).val = 7 := rfl
  have hfortynine : (49#usize).val = 49 := rfl
  obtain ⟨next, hnext, hnextVal⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := step) (y := 1#usize) (by omega))
  have hnextBound : next.val ≤ 7 := by omega
  have hmulCap : (7#usize).val * next.val ≤ Usize.max := by
    rw [hseven]
    omega
  obtain ⟨sevenfold, hsevenfold, hsevenfoldVal⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 7#usize) (y := next) hmulCap)
  have hsevenfoldBound : sevenfold.val ≤ 49 := by
    rw [hseven] at hsevenfoldVal
    omega
  obtain ⟨shift, hshift, hshiftVal⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := 49#usize) (y := sevenfold) (by omega))
  have hshiftBound : shift.val < 64 := by omega
  obtain ⟨shifted, hshifted, _⟩ := WP.spec_imp_exists
    (U64.ShiftRight_spec (high &&& word) shift hshiftBound)
  refine ⟨UScalar.cast .Usize (shifted &&& table.WINDOW_MASK), ?_⟩
  unfold table.group
  simp only [hhigh, hword, bind_tc_ok, lift, if_pos hstep,
    hnext, hsevenfold, hshift, hshifted]

private theorem group_mask_mod (groups mask : Usize) (exponent : Nat)
    (hpower : groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask) (x : Usize) :
    (x &&& mask).val = x.val % groups.val := by
  have hpositive : 1 ≤ groups.val := by
    rw [hpower]
    exact Nat.one_le_iff_ne_zero.mpr (by positivity)
  have hsub := Usize.sub_spec (x := groups) (y := 1#usize) (by simpa using hpositive)
  rw [hmask, WP.spec_ok] at hsub
  have hmaskVal : mask.val = groups.val - 1 := by simpa using hsub.1
  rw [UScalar.val_and, hmaskVal, hpower, Nat.and_two_pow_sub_one_eq_mod]

private theorem group_wrapping_mask_mod (groups mask : Usize) (exponent : Nat)
    (hpower : groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask) (x y : Usize) :
    ((core.num.Usize.wrapping_add (x &&& mask) y) &&& mask).val =
      (x.val + y.val) % groups.val := by
  have hwidth : exponent ≤ System.Platform.numBits := by
    by_contra h
    have hlarge := Nat.pow_lt_pow_right (show 1 < (2 : Nat) by omega)
      (show System.Platform.numBits < exponent by omega)
    have hbounds := groups.hBounds
    simp only [UScalarTy.Usize_numBits_eq] at hbounds
    omega
  have hdvd : groups.val ∣ 2 ^ System.Platform.numBits := by
    rw [hpower]
    exact Nat.pow_dvd_pow 2 hwidth
  rw [group_mask_mod groups mask exponent hpower hmask]
  rw [core.num.Usize.wrapping_add_val_eq]
  have hsize : UScalar.size .Usize = 2 ^ System.Platform.numBits := by
    simp [Usize.size, Usize.numBits]
  rw [hsize, Nat.mod_mod_of_dvd _ hdvd]
  rw [group_mask_mod groups mask exponent hpower hmask]
  exact Nat.mod_add_mod x.val groups.val y.val

/-- A machine-representable power-of-two group count leaves room for all seven
window steps on both admitted platform widths. No storage-length bound is used. -/
theorem power_of_two_groups_headroom (groups : Usize)
    (hpower : ∃ exponent : Nat, groups.val = 2 ^ exponent) :
    7 + groups.val ≤ Usize.max := by
  obtain ⟨exponent, hpow⟩ := hpower
  have hword := groups.hBounds
  simp only [UScalarTy.Usize_numBits_eq] at hword
  have hexponent : exponent < System.Platform.numBits := by
    by_contra h
    have hlarge := Nat.pow_le_pow_right (show 0 < (2 : Nat) by omega)
      (show System.Platform.numBits ≤ exponent by omega)
    omega
  have hhalf : groups.val ≤ 2 ^ (System.Platform.numBits - 1) := by
    rw [hpow]
    exact Nat.pow_le_pow_right (show 0 < (2 : Nat) by omega) (by omega)
  have hmargin : 2 ^ (System.Platform.numBits - 1) + 7 ≤ Usize.max := by
    rcases System.Platform.numBits_eq with h | h <;>
      norm_num [Usize.max, Usize.numBits, h]
  omega

/-- Both generated `table.group` branches complete and produce a masked group
within the checked power-of-two extent while the outer guard holds. -/
theorem generated_group_bounded (place : U64) (groups mask limit step : Usize)
    (hcap : 7 + groups.val ≤ Usize.max)
    (hpower : ∃ exponent, groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hstep : step < limit) :
    ∃ group : Usize, table.group place step mask = ok group ∧
      group.val < groups.val := by
  obtain ⟨exponent, hexponent⟩ := hpower
  have hpositive : 0 < groups.val := by rw [hexponent]; positivity
  obtain ⟨high, word, hhigh, hword⟩ := generated_group_head_ok place
  by_cases hprefix : step < table.WINDOWS
  · obtain ⟨unmasked, hgroup⟩ :=
      generated_group_prefix_step place high word step mask hhigh hword hprefix
    refine ⟨unmasked &&& mask, hgroup, ?_⟩
    rw [group_mask_mod groups mask exponent hexponent hmask]
    exact Nat.mod_lt _ hpositive
  · have hsum := Usize.add_spec (x := table.WINDOWS) (y := groups)
        (by simpa [table.WINDOWS] using hcap)
    rw [hlimit, WP.spec_ok] at hsum
    have hlimitVal : limit.val = 7 + groups.val := by simpa [table.WINDOWS] using hsum
    have hstepVal := (UScalar.lt_equiv step limit).mp hstep
    have hroom : step.val - 7 + 1 ≤ Usize.max := by omega
    obtain ⟨offset, hgroup, _⟩ :=
      generated_group_tail_step place high word step mask hhigh hword hprefix hroom
    let candidate : Usize := core.num.Usize.wrapping_add
      ((UScalar.cast .Usize (high &&& word)) &&& mask) offset
    refine ⟨candidate &&& mask, hgroup, ?_⟩
    rw [group_mask_mod groups mask exponent hexponent hmask]
    exact Nat.mod_lt _ hpositive

private theorem cyclic_tail_index (groups start target : Nat)
    (hstart : start < groups)
    (htarget : target < groups) :
    ∃ j, j < groups ∧ (start + j + 1) % groups = target := by
  by_cases hbefore : target < start + 1
  · refine ⟨target + groups - (start + 1), by omega, ?_⟩
    have hsum : start + (target + groups - (start + 1)) + 1 = target + groups := by
      omega
    rw [hsum]
    simpa using (Nat.add_mod_right target groups).trans (Nat.mod_eq_of_lt htarget)
  · refine ⟨target - (start + 1), by omega, ?_⟩
    have hsum : start + (target - (start + 1)) + 1 = target := by omega
    rw [hsum, Nat.mod_eq_of_lt htarget]

/-- The generated tail steps, not merely their guards, visit every group once
under the checked `groups - 1` mask. Works at either platform `usize` width. -/
theorem generated_tail_group_surjective (place : U64) (groups mask : Usize)
    (hcap : 7 + groups.val ≤ Usize.max)
    (hpower : ∃ exponent : Nat, groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask) :
    ∃ limit : Usize, table.WINDOWS + groups = ok limit ∧
      ∀ target : Usize, target.val < groups.val →
        ∃ step : Usize, 7 ≤ step.val ∧ step < limit ∧
          table.group place step mask = ok target := by
  obtain ⟨exponent, hexponent⟩ := hpower
  obtain ⟨limit, hlimit, heligible⟩ := generated_tail_indices_eligible groups hcap
  obtain ⟨high, word, hhigh, hword⟩ := generated_group_head_ok place
  let seed : Usize := UScalar.cast .Usize (high &&& word)
  have hgroups : 0 < groups.val := by rw [hexponent]; positivity
  have hstart : seed.val % groups.val < groups.val := Nat.mod_lt _ hgroups
  refine ⟨limit, hlimit, ?_⟩
  intro target htarget
  obtain ⟨j, hj, hcycle⟩ := cyclic_tail_index groups.val
    (seed.val % groups.val) target.val hstart htarget
  obtain ⟨step, hstep, hlt⟩ := heligible j hj
  have htail : ¬ step < table.WINDOWS := by
    intro h
    have hval := (UScalar.lt_equiv step table.WINDOWS).mp h
    have hwindows : table.WINDOWS.val = 7 := by simp [table.WINDOWS]
    omega
  have hroom : step.val - 7 + 1 ≤ Usize.max := by omega
  obtain ⟨offset, hgroup, hoffset⟩ :=
    generated_group_tail_step place high word step mask hhigh hword htail hroom
  have hvalue :
      ((core.num.Usize.wrapping_add (seed &&& mask) offset) &&& mask).val =
        target.val := by
    rw [group_wrapping_mask_mod groups mask exponent hexponent hmask]
    have hoff : offset.val = j + 1 := by omega
    rw [hoff, ← Nat.mod_add_mod seed.val groups.val (j + 1)]
    simpa only [Nat.add_assoc] using hcycle
  have heq : (core.num.Usize.wrapping_add (seed &&& mask) offset) &&& mask =
      target := UScalar.eq_of_val_eq hvalue
  refine ⟨step, by omega, hlt, ?_⟩
  simpa only [seed, heq] using hgroup

/-- A successful generated payload lookup followed by the *real stored-key matcher* implies
in-bounds, complete `(Sigma,key)` equality. A fingerprint by itself supplies neither fact. -/
theorem stored_candidate_full_match
    (slots : alloc.vec.Vec (cell.Sigma × Apeiron.GrowableIdentity.StoredI64 × I64))
    (index : Usize) (entry : cell.Sigma × Apeiron.GrowableIdentity.StoredI64 × I64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (hslot : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice
        (cell.Sigma × Apeiron.GrowableIdentity.StoredI64 × I64)) slots index = ok entry)
    (hmatch : table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key address =
      ok true) :
    index.val < slots.length ∧
      Apeiron.GrowableIdentity.logicalStored entry.1 entry.2.1 = (sigma, key) := by
  rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hslot
  cases hread : slots.val[index.val]? with
  | none => simp [hread] at hslot
  | some payload =>
      refine ⟨(List.getElem?_eq_some_iff.mp hread).1, ?_⟩
      exact (Apeiron.GrowableIdentity.stored_matches_iff
        entry.1 sigma entry.2.1 entry.2.2 key address).mp hmatch

/-- The generated carried-key matcher terminates for every payload and query;
the generic `KeyRep` trait alone does not supply this law. -/
theorem stored_matcher_total
    (entry : cell.Sigma × Apeiron.GrowableIdentity.StoredI64 × I64)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress) :
    ∃ matched : Bool,
      table.matches Apeiron.GrowableIdentity.storedKeyRep entry sigma key address =
        ok matched := by
  rcases entry with ⟨slotSigma, held, value⟩
  cases slotSigma <;> cases sigma <;> cases held <;>
    simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant,
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches]

/-- One *actual generated publication* establishes tag, payload and summary
agreement at its destination. `RepR.address` ties the logical representative
to the address passed to publication. This is not preservation for every
other live slot or for an arbitrary `RepR` state without a publication trace. -/
theorem repR_stored_published_tag_address
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (decode : cell.Sigma → Apeiron.GrowableIdentity.StoredI64 → Option I64)
    (directIndex : Apeiron.GrowableRepresentation.LogicalKey →
      Option Apeiron.GrowableRepresentation.Location)
    (live : Apeiron.GrowableRepresentation.LogicalKey → Prop)
    (logical : Apeiron.Table.HMap Apeiron.GrowableRepresentation.LogicalKey I64)
    (hrep : Apeiron.GrowableRepresentation.RepR state decode directIndex live logical)
    (shard index : Usize)
    (before region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64)
    (key value : I64) (address : table.TableAddress)
    (hentry : Apeiron.GrowableSearch.entryAt state decode (.spill shard index) =
      some ((sigma, key), value))
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key =
      ok address)
    (hpublish : table.Region.publish_live before index (sigma, held, value)
      address.place = ok region) :
    table.shard address = ok shard ∧
    region.slots.val[index.val]? = some (sigma, held, value) ∧
    ∃ tag : U8, region.tags.val[index.val]? = some tag ∧
      table.tag address.place = ok tag ∧ tag ≠ 0#u8 ∧
      table.Region.summary_maybe region address.place = ok true := by
  have _ : region ∈ state.shards.val := List.mem_of_getElem? hregion
  obtain ⟨actualAddress, hactualAddress, hshard⟩ :=
    hrep.address (.spill shard index) (sigma, key) value hentry
  have heq : actualAddress = address :=
    Result.ok_injective (hactualAddress.symm.trans haddress)
  subst actualAddress
  have hrecorded := Apeiron.GrowableRepresentation.publish_live_recorded
    before region index (sigma, held, value) address.place hpublish
  obtain ⟨tag, htag, htagNonzero, htagAt⟩ :=
    Apeiron.GrowableScan.publish_live_tag_readback
      before region index (sigma, held, value) address.place hpublish
  exact ⟨hshard, hrecorded.payloadAt, tag, htagAt, htag, htagNonzero,
    hrecorded.summary⟩

/-- A successfully compared lane includes the generated checked offset and payload lookup. -/
def InnerCompared {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (j : Nat) (matched : Bool) (slot : Usize) : Prop :=
  ∃ (lane : Usize) (entry : cell.Sigma × R × V),
    lane.val = j ∧ base + lane = ok slot ∧
    alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) slots slot = ok entry ∧
    table.matches keyRep entry sigma key
      { place := place, direct_word := direct } = ok matched

/-- Every initially selected lane is admitted to a checked lookup and a total matcher call.
This is a local group premise, not a claim that the outer loop reaches the group. -/
def InnerReady {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (initial : U64) : Prop :=
  ∀ j, j < 64 → initial.bv.getLsbD j = true →
    ∃ slot, InnerCompared keyRep slots sigma place direct key base j false slot ∨
      InnerCompared keyRep slots sigma place direct key base j true slot

/-- Exact generated tag scanning selects only in-range lanes. The slice's end lies
within payload storage; callback termination is stated separately for the actual
entry returned by the checked lookup. No tag/address equality is needed for
*readiness*: it is needed later to turn a miss into logical absence. -/
theorem inner_ready_of_scan_geometry {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (line : Slice U8) (wanted : U8) (bits : U64)
    (hlen : line.val.length ≤ 64)
    (hreach : base.val + line.val.length ≤ slots.length)
    (hscan : table.scan.match_tag line wanted = ok bits)
    (hmatcher : ∀ entry : cell.Sigma × R × V,
      ∃ matched : Bool, table.matches keyRep entry sigma key
        { place := place, direct_word := direct } = ok matched) :
    InnerReady keyRep slots sigma place direct key base bits := by
  obtain ⟨selected, hselected, hexact⟩ :=
    Apeiron.GrowableScan.match_tag_exact line wanted hlen
  have hsame : selected = bits := Result.ok_injective (hselected.symm.trans hscan)
  subst selected
  intro j hj64 hjbit
  have hjline : j < line.val.length := by
    rw [hexact j] at hjbit
    exact (decide_eq_true_eq.mp hjbit).1
  have hjwidth : j < 2 ^ System.Platform.numBits := by
    have hmin : 64 ≤ 2 ^ System.Platform.numBits := by
      rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num
    omega
  let lane : Usize := ⟨BitVec.ofNat System.Platform.numBits j⟩
  have hlane : lane.val = j := by
    simp [lane, UScalar.val, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hjwidth]
  have hcap : base.val + lane.val ≤ Usize.max := by
    have hslots := slots.property
    change slots.length ≤ Usize.max at hslots
    omega
  obtain ⟨slot, hslot, hslotVal⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hcap)
  have hwithin : slot.val < slots.length := by omega
  obtain ⟨entry, hentry, _⟩ :=
    WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec slots slot hwithin)
  have hindex : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (cell.Sigma × R × V)) slots slot = ok entry := by
    simpa only [alloc.vec.Vec.index_slice_index] using hentry
  obtain ⟨matched, hmatch⟩ := hmatcher entry
  refine ⟨slot, ?_⟩
  cases matched with
  | false => exact Or.inl ⟨lane, entry, hlane, hslot, hindex, hmatch⟩
  | true => exact Or.inr ⟨lane, entry, hlane, hslot, hindex, hmatch⟩

/-- Remaining bits are original bits; every original bit already cleared was compared and missed.
The second branch records the *actual* successful generated payload lookup and full matcher call. -/
def InnerSweepInv {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (initial : U64) (state : Option Usize × U64) : Prop :=
  (state.1 = none ∧
    (∀ j, state.2.bv.getLsbD j = true → initial.bv.getLsbD j = true) ∧
    (∀ j, initial.bv.getLsbD j = true →
      state.2.bv.getLsbD j = true ∨
        ∃ slot, InnerCompared keyRep slots sigma place direct key base j false slot)) ∨
  (∃ j slot, state.1 = some slot ∧ initial.bv.getLsbD j = true ∧
    InnerCompared keyRep slots sigma place direct key base j true slot)

/-- The inner-loop outcome distinguishes a witnessed full match from exhaustion of all
initially selected lanes; this does not prove the outer group's visitation. -/
def InnerSweepOutcome {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (initial : U64) (result : Option Usize) : Prop :=
  (result = none ∧ ∀ j, initial.bv.getLsbD j = true →
    ∃ slot, InnerCompared keyRep slots sigma place direct key base j false slot) ∨
  (∃ j slot, result = some slot ∧ initial.bv.getLsbD j = true ∧
    InnerCompared keyRep slots sigma place direct key base j true slot)

private theorem selected_bit (hits : U64) (lane : Usize)
    (hne : hits ≠ 0#u64)
    (hselect : table.scan.least_set_lane hits = ok lane) :
    hits.bv.getLsbD lane.val = true := by
  obtain ⟨_, hdiv, hnot⟩ :=
    Apeiron.GrowableScan.least_set_lane_in_range hits lane hne hselect
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
  change hits.val.testBit lane.val = true
  rw [hq, Nat.mul_comm, Nat.testBit_mul_two_pow]
  simp [Nat.testBit_eq_decide_div_mod_eq, hodd]

/-- The generated inner loop enumerates its initial finite mask or returns an actual
matching slot. `InnerReady` explicitly excludes checked offset/index and callback failures. -/
theorem generated_inner_sweep_outcome {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K) (base : Usize)
    (initial : U64)
    (hready : InnerReady keyRep slots sigma place direct key base initial) :
    table.probe.spill_read_loop0_loop0 keyRep slots sigma place direct key none base initial
      ⦃ result => InnerSweepOutcome keyRep slots sigma place direct key base initial result ⦄ := by
  unfold table.probe.spill_read_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun state : Option Usize × U64 =>
      Apeiron.GrowableScan.liveLaneCount state.2)
    (inv := InnerSweepInv keyRep slots sigma place direct key base initial)
    (post := InnerSweepOutcome keyRep slots sigma place direct key base initial)
  · intro state hstate
    rcases state with ⟨hit, hits⟩
    dsimp only at hstate ⊢
    rcases hstate with ⟨rfl, hsubset, hmiss⟩ | ⟨j, slot, rfl, horig, hfound⟩
    · by_cases hne : hits ≠ 0#u64
      · obtain ⟨lane, hselect, hlane, _, _⟩ :=
          Apeiron.GrowableScan.least_set_lane_exact hits hne
        have hbit : hits.bv.getLsbD lane.val = true := selected_bit hits lane hne hselect
        have horiginal : initial.bv.getLsbD lane.val = true := hsubset _ hbit
        obtain ⟨nextSlot, hfalse | htrue⟩ := hready lane.val hlane horiginal
        · rcases hfalse with ⟨selected, entry, hval, hadd, hindex, hmatch⟩
          have hselected : selected = lane := UScalar.eq_of_val_eq (by omega)
          subst selected
          obtain ⟨next, hbody, hclear, hcount⟩ :=
            Apeiron.GrowableScan.generated_spill_read_clears_selected
              keyRep slots sigma place direct key base hits lane nextSlot entry false
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
          · simpa using (show Apeiron.GrowableScan.liveLaneCount next <
              Apeiron.GrowableScan.liveLaneCount hits by omega)
        · rcases htrue with ⟨selected, entry, hval, hadd, hindex, hmatch⟩
          have hselected : selected = lane := UScalar.eq_of_val_eq (by omega)
          subst selected
          obtain ⟨next, hbody, _, hcount⟩ :=
            Apeiron.GrowableScan.generated_spill_read_clears_selected
              keyRep slots sigma place direct key base hits lane nextSlot entry true
              hne hselect hadd hindex hmatch
          rw [hbody, WP.spec_ok]
          refine ⟨Or.inr ⟨lane.val, nextSlot, rfl, horiginal, ?_⟩, ?_⟩
          · exact ⟨lane, entry, rfl, hadd, hindex, hmatch⟩
          · simpa using (show Apeiron.GrowableScan.liveLaneCount next <
              Apeiron.GrowableScan.liveLaneCount hits by omega)
      · have hz : hits = 0#u64 := not_not.mp hne
        subst hits
        unfold table.probe.spill_read_loop0_loop0.body
        simp only [bne_self_eq_false, Bool.false_eq_true, if_false, WP.spec_ok]
        left
        refine ⟨rfl, ?_⟩
        intro j hj
        rcases hmiss j hj with hbit | hcomp
        · simp at hbit
        · exact hcomp
    · by_cases hne : hits ≠ 0#u64
      · have hcond : (hits != 0#u64) = true := by simpa using hne
        unfold table.probe.spill_read_loop0_loop0.body
        simp only [hcond, if_true, core.option.Option.is_none, Option.isNone_some,
          Bool.false_eq_true, if_false, WP.spec_ok]
        exact Or.inr ⟨j, slot, rfl, horig, hfound⟩
      · have hz : hits = 0#u64 := not_not.mp hne
        subst hits
        unfold table.probe.spill_read_loop0_loop0.body
        simp only [bne_self_eq_false, Bool.false_eq_true, if_false, WP.spec_ok]
        exact Or.inr ⟨j, slot, rfl, horig, hfound⟩
  · left
    refine ⟨rfl, ?_, ?_⟩
    · intro j hj
      exact hj
    · intro j hj
      exact Or.inl hj

/-- Checked geometry and scan for a reachable outer step, together with the inner-loop
admission needed to turn its selected mask into a successful result. -/
def OuterStepReady {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (step : Usize) : Prop :=
  ∃ (group base finish : Usize) (line : Slice U8) (bits : U64),
    table.group place step mask = ok group ∧ group * n = ok base ∧
    base + n = ok finish ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice U8)
      tags { start := base, «end» := finish } = ok line ∧
    table.scan.match_tag line wanted = ok bits ∧
    InnerReady keyRep slots sigma place direct key base bits

/-- Group-major storage with equal tag/payload extents supplies every checked
range read. `match_tag_exact` supplies a terminating scan and a mask confined to
the slice; matcher termination remains an explicit callback premise. -/
theorem outer_step_ready_of_geometry {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8) (limit step : Usize)
    (hwidth : n.val ≤ 64)
    (hshape : tags.length = slots.length ∧ slots.length = groups.val * n.val)
    (hcap : 7 + groups.val ≤ Usize.max)
    (hpower : ∃ exponent, groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hstep : step < limit)
    (hmatcher : ∀ entry : cell.Sigma × R × V,
      ∃ matched : Bool, table.matches keyRep entry sigma key
        { place := place, direct_word := direct } = ok matched) :
    OuterStepReady keyRep tags slots sigma place direct key n mask wanted step := by
  obtain ⟨group, hgroup, hgroupBound⟩ :=
    generated_group_bounded place groups mask limit step hcap hpower hmask hlimit hstep
  have hslots := slots.property
  change slots.length ≤ Usize.max at hslots
  have hgroupNext : group.val + 1 ≤ groups.val := by omega
  have hbaseCap : group.val * n.val ≤ Usize.max := by
    have hmono := Nat.mul_le_mul_right n.val (Nat.le_of_lt hgroupBound)
    omega
  obtain ⟨base, hbase, hbaseVal⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := group) (y := n) hbaseCap)
  have hfinishCap : base.val + n.val ≤ Usize.max := by
    have hmono := Nat.mul_le_mul_right n.val hgroupNext
    have hnext : (group.val + 1) * n.val = group.val * n.val + n.val := by ring
    omega
  obtain ⟨finish, hfinish, hfinishVal⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := base) (y := n) hfinishCap)
  have hfinishBound : finish.val ≤ tags.length := by
    have hmono := Nat.mul_le_mul_right n.val hgroupNext
    have hnext : (group.val + 1) * n.val = group.val * n.val + n.val := by ring
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
  obtain ⟨bits, hbits, _⟩ :=
    Apeiron.GrowableScan.match_tag_exact line wanted hline64
  have hreach : base.val + line.val.length ≤ slots.length := by
    have hmono := Nat.mul_le_mul_right n.val hgroupNext
    have hnext : (group.val + 1) * n.val = group.val * n.val + n.val := by ring
    omega
  refine ⟨group, base, finish, line, bits, hgroup, hbase, hfinish,
    hline, hbits, ?_⟩
  exact inner_ready_of_scan_geometry keyRep slots sigma place direct key base line
    wanted bits hline64 hreach hbits hmatcher

/-- Admission facts supplied by the generated positive-summary branch and
grouped sizing, not by `RepR` alone. In particular, a recorded summary bit for
a *live entry* does not imply that an arbitrary query has a positive summary. -/
structure GroupedReadAdmission {R : Type} (region : table.Region R I64)
    (place : U64) (n groups mask limit : Usize) : Prop where
  width : n.val ≤ 64
  groupSlots : region.slots.length = groups.val * n.val
  headroom : 7 + groups.val ≤ Usize.max
  power : ∃ exponent, groups.val = 2 ^ exponent
  maskResult : groups - 1#usize = ok mask
  boundResult : table.WINDOWS + groups = ok limit
  summaryPositive : table.Region.summary_maybe region place = ok true

/-- Exact grouped sizing is a generated admission-branch fact, not a consequence
of `RepR.directory`. Power-of-two machine geometry supplies the seven-step
headroom rather than requiring it as a separate allocation promise. -/
theorem grouped_admission_of_checked_branch {R : Type}
    (region : table.Region R I64) (place : U64)
    (n groups mask limit : Usize)
    (hwidth : 0 < n.val ∧ n.val ≤ 64)
    (hslots : region.slots.length = groups.val * n.val)
    (hpower : ∃ exponent : Nat, groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hsummary : table.Region.summary_maybe region place = ok true) :
    GroupedReadAdmission region place n groups mask limit ∧
      region.slots.length / n.val = groups.val := by
  refine ⟨⟨hwidth.2, hslots, power_of_two_groups_headroom groups hpower,
    hpower, hmask, hlimit, hsummary⟩, ?_⟩
  rw [hslots, Nat.mul_div_cancel groups.val hwidth.1]

/-- `RepR.directory` supplies the missing tag/payload length equality for the
selected *actual* shard. Its summary fields are retained without identifying
their existential dictionary with the executing `keyRep`; matcher termination
is required separately for a generic `KeyRep`. -/
theorem repR_region_outer_ready {R : Type}
    (state : Apeiron.GrowableRepresentation.WordTable R)
    (decode : cell.Sigma → R → Option I64)
    (directIndex : Apeiron.GrowableRepresentation.LogicalKey →
      Option Apeiron.GrowableRepresentation.Location)
    (live : Apeiron.GrowableRepresentation.LogicalKey → Prop)
    (logical : Apeiron.Table.HMap Apeiron.GrowableRepresentation.LogicalKey I64)
    (hrep : Apeiron.GrowableRepresentation.RepR state decode directIndex live logical)
    (shard : Usize) (region : table.Region R I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (keyRep : table.keyrep.KeyRep R I64 table.WordHome)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (n groups mask limit : Usize) (wanted : U8)
    (hadmit : GroupedReadAdmission region place n groups mask limit)
    (hmatcher : ∀ entry : cell.Sigma × R × I64,
      ∃ matched : Bool, table.matches keyRep entry sigma key
        { place := place, direct_word := direct } = ok matched) :
    Apeiron.GrowableScan.SpillSummaryShape region ∧
    Apeiron.GrowableScan.SpillFingerprintExact
      (fun sigma key =>
        Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key) ∧
    table.Region.summary_maybe region place = ok true ∧
    (∃ recordedRep : table.keyrep.KeyRep R I64 table.WordHome,
      Apeiron.GrowableRepresentation.CarrierAddressRespecting state decode recordedRep ∧
      Apeiron.GrowableRepresentation.SpillSummaryRecorded state decode recordedRep) ∧
    ∀ step : Usize, step < limit →
      OuterStepReady keyRep region.tags region.slots sigma place direct key
        n mask wanted step := by
  have hmem : region ∈ state.shards.val := List.mem_of_getElem? hregion
  have htagSlots : region.tags.length = region.slots.length := hrep.directory.2.2 region hmem
  refine ⟨hrep.summaryShape shard region hregion,
    hrep.fingerprintExact shard region hregion, hadmit.summaryPositive,
    hrep.summaryRecorded, ?_⟩
  intro step hstep
  exact outer_step_ready_of_geometry keyRep region.tags region.slots sigma place direct
    key n groups mask wanted limit step hadmit.width
    ⟨htagSlots, hadmit.groupSlots⟩ hadmit.headroom hadmit.power
    hadmit.maskResult hadmit.boundResult hstep hmatcher

/-- For the concrete stored-word carrier, the generated matcher discharges the
generic callback-termination premise without using `RepR` as a callback axiom. -/
theorem repR_stored_region_outer_ready
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (decode : cell.Sigma → Apeiron.GrowableIdentity.StoredI64 → Option I64)
    (directIndex : Apeiron.GrowableRepresentation.LogicalKey →
      Option Apeiron.GrowableRepresentation.Location)
    (live : Apeiron.GrowableRepresentation.LogicalKey → Prop)
    (logical : Apeiron.Table.HMap Apeiron.GrowableRepresentation.LogicalKey I64)
    (hrep : Apeiron.GrowableRepresentation.RepR state decode directIndex live logical)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : I64)
    (n groups mask limit : Usize) (wanted : U8)
    (hadmit : GroupedReadAdmission region place n groups mask limit) :
    ∀ step : Usize, step < limit →
      OuterStepReady Apeiron.GrowableIdentity.storedKeyRep
        region.tags region.slots sigma place direct key n mask wanted step := by
  have hready := repR_region_outer_ready state decode directIndex live logical hrep
    shard region hregion Apeiron.GrowableIdentity.storedKeyRep sigma place direct key
    n groups mask limit wanted hadmit (fun entry => stored_matcher_total entry sigma key
      { place := place, direct_word := direct })
  exact hready.2.2.2.2

/-- The generated summary result has two distinct meanings: `true` enters
the scan but need not denote a present key; `false` carries the checked
`RepR` miss certificate for the routed query and selected region. -/
theorem repR_summary_read_branch {R : Type}
    (state : Apeiron.GrowableRepresentation.WordTable R)
    (decode : cell.Sigma → R → Option I64)
    (directIndex : Apeiron.GrowableRepresentation.LogicalKey →
      Option Apeiron.GrowableRepresentation.Location)
    (live : Apeiron.GrowableRepresentation.LogicalKey → Prop)
    (logical : Apeiron.Table.HMap Apeiron.GrowableRepresentation.LogicalKey I64)
    (hrep : Apeiron.GrowableRepresentation.RepR state decode directIndex live logical)
    (query : Apeiron.GrowableRepresentation.LogicalKey)
    (address : table.TableAddress) (shard : Usize) (region : table.Region R I64)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address
      state.home query.1 query.2 = ok address)
    (hshard : table.shard address = ok shard)
    (hregion : state.shards.val[shard.val]? = some region)
    (result : Bool)
    (hread : table.Region.summary_maybe region address.place = ok result) :
    (result = true ∧ table.Region.summary_maybe region address.place = ok true) ∨
    (result = false ∧ Apeiron.GrowableRepresentation.SpillSummaryMissSound
      state decode query shard region) := by
  cases result with
  | false =>
      exact Or.inr ⟨rfl,
        Apeiron.GrowableRepresentation.repR_spillSummaryMissSound hrep query
          address shard region haddress hshard hregion hread⟩
  | true => exact Or.inl ⟨rfl, hread⟩

/-- A genuine live representative forces a successful generated summary query
to be positive. Its converse is false: summaries admit false positives. -/
theorem repR_live_spill_summary_positive {R : Type}
    (state : Apeiron.GrowableRepresentation.WordTable R)
    (decode : cell.Sigma → R → Option I64)
    (directIndex : Apeiron.GrowableRepresentation.LogicalKey →
      Option Apeiron.GrowableRepresentation.Location)
    (live : Apeiron.GrowableRepresentation.LogicalKey → Prop)
    (logical : Apeiron.Table.HMap Apeiron.GrowableRepresentation.LogicalKey I64)
    (hrep : Apeiron.GrowableRepresentation.RepR state decode directIndex live logical)
    (query : Apeiron.GrowableRepresentation.LogicalKey)
    (address : table.TableAddress) (shard index : Usize) (region : table.Region R I64)
    (value : I64)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address
      state.home query.1 query.2 = ok address)
    (hshard : table.shard address = ok shard)
    (hregion : state.shards.val[shard.val]? = some region)
    (hentry : Apeiron.GrowableSearch.entryAt state decode (.spill shard index) =
      some (query, value))
    (result : Bool)
    (hread : table.Region.summary_maybe region address.place = ok result) :
    table.Region.summary_maybe region address.place = ok true := by
  have hbranch := repR_summary_read_branch state decode directIndex live logical
    hrep query address shard region haddress hshard hregion result hread
  rcases hbranch with ⟨_, hpositive⟩ | ⟨_, hmiss⟩
  · exact hpositive
  · exact False.elim ((hmiss.excludes index value) hentry)

/-- One *executed* generated group scan, including its actual inner-loop result. -/
def OuterStepResult {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (step : Usize) (result : Option Usize) : Prop :=
  ∃ (group base finish : Usize) (line : Slice U8) (bits : U64),
    table.group place step mask = ok group ∧ group * n = ok base ∧
    base + n = ok finish ∧
    alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice U8)
      tags { start := base, «end» := finish } = ok line ∧
    table.scan.match_tag line wanted = ok bits ∧
    table.probe.spill_read_loop0_loop0 keyRep slots sigma place direct key
      none base bits = ok result ∧
    InnerSweepOutcome keyRep slots sigma place direct key base bits result

/-- Every earlier step was executed if the accumulated hit is absent. On a hit, the
actual step returning that slot is recorded and every earlier step returned no hit. -/
def OuterSweepInv {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (limit : Usize)
    (state : Option Usize × Usize) : Prop :=
  (state.1 = none ∧ state.2.val ≤ limit.val ∧
    ∀ j, j < state.2.val →
      ∃ cursor : Usize, cursor.val = j ∧
        OuterStepResult keyRep tags slots sigma place direct key n mask wanted cursor none) ∨
  (∃ (j : Nat) (cursor slot : Usize),
    state.1 = some slot ∧ cursor.val = j ∧ j < state.2.val ∧
    state.2.val ≤ limit.val ∧
    OuterStepResult keyRep tags slots sigma place direct key n mask wanted cursor (some slot) ∧
    ∀ k, k < j → ∃ earlier : Usize, earlier.val = k ∧
      OuterStepResult keyRep tags slots sigma place direct key n mask wanted earlier none)

/-- A `none` result has completed every step up to the generated bound, not merely
passed its guard. A hit carries the first step whose inner sweep found that slot. -/
def OuterSweepOutcome {K H R V : Type} (keyRep : table.keyrep.KeyRep R K H)
    (tags : alloc.vec.Vec U8) (slots : alloc.vec.Vec (cell.Sigma × R × V))
    (sigma : cell.Sigma) (place : U64) (direct : Option U32) (key : K)
    (n mask : Usize) (wanted : U8) (limit : Usize) (result : Option Usize) : Prop :=
  (result = none ∧ ∀ j, j < limit.val →
    ∃ cursor : Usize, cursor.val = j ∧
      OuterStepResult keyRep tags slots sigma place direct key n mask wanted cursor none) ∨
  (∃ (j : Nat) (cursor slot : Usize), result = some slot ∧
    cursor.val = j ∧ j < limit.val ∧
    OuterStepResult keyRep tags slots sigma place direct key n mask wanted cursor (some slot) ∧
    ∀ k, k < j → ∃ earlier : Usize, earlier.val = k ∧
      OuterStepResult keyRep tags slots sigma place direct key n mask wanted earlier none)

/-- The real generated outer loop either stops at the first step with a matching
inner result or completes all `WINDOWS + groups` steps. No group-surjectivity is used. -/
theorem generated_outer_sweep_outcome {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (tags : alloc.vec.Vec U8)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8) (limit : Usize)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hready : ∀ step : Usize, step < limit →
      OuterStepReady keyRep tags slots sigma place direct key n mask wanted step) :
    table.probe.spill_read_loop0 keyRep tags slots sigma place direct key n
      none groups mask wanted 0#usize ⦃ result =>
        OuterSweepOutcome keyRep tags slots sigma place direct key n mask wanted limit result ⦄ := by
  unfold table.probe.spill_read_loop0
  apply loop.spec_decr_nat
    (measure := fun state : Option Usize × Usize => limit.val - state.2.val)
    (inv := OuterSweepInv keyRep tags slots sigma place direct key n mask wanted limit)
    (post := OuterSweepOutcome keyRep tags slots sigma place direct key n mask wanted limit)
  · intro state hstate
    rcases state with ⟨hit, step⟩
    rcases hstate with ⟨rfl, hbound, hprior⟩ |
      ⟨j, cursor, slot, rfl, hcursor, hjstep, hbound, hfound, hprior⟩
    · by_cases hlt : step < limit
      · rcases hready step hlt with ⟨group, base, finish, line, bits,
          hgroup, hbase, hfinish, hline, hbits, hinnerReady⟩
        obtain ⟨result, hinner, houtcome⟩ := WP.spec_imp_exists
          (generated_inner_sweep_outcome keyRep slots sigma place direct key base
            bits hinnerReady)
        have hcap : step.val + 1 ≤ Usize.max := by
          have hltVal := (UScalar.lt_equiv step limit).mp hlt
          have hlimitBounds := limit.hBounds
          simp only [UScalarTy.Usize_numBits_eq] at hlimitBounds
          have hlimitCap : limit.val ≤ Usize.max := by
            have heq : Usize.max = 2 ^ System.Platform.numBits - 1 := by
              simp [Usize.max, Usize.numBits]
            rw [heq]
            omega
          omega
        obtain ⟨next, hnext, hnextVal⟩ :=
          WP.spec_imp_exists (Usize.add_spec (x := step) (y := 1#usize) hcap)
        have hnextValue : next.val = step.val + 1 := by simpa using hnextVal
        have hlocal : OuterStepResult keyRep tags slots sigma place direct key
            n mask wanted step result :=
          ⟨group, base, finish, line, bits, hgroup, hbase, hfinish,
            hline, hbits, hinner, houtcome⟩
        unfold table.probe.spill_read_loop0.body
        simp only [hlimit, bind_tc_ok, if_pos hlt, core.option.Option.is_none,
          Option.isNone_none, if_true, hgroup, hbase, hfinish, hline,
          hbits, hinner, hnext, WP.spec_ok]
        cases result with
        | none =>
            refine ⟨Or.inl ⟨rfl, ?_, ?_⟩, ?_⟩
            · have hnextBound := (UScalar.lt_equiv step limit).mp hlt
              change next.val ≤ limit.val
              omega
            · intro k hk
              change k < next.val at hk
              by_cases hbefore : k < step.val
              · exact hprior k hbefore
              · have heq : k = step.val := by omega
                subst k
                exact ⟨step, rfl, hlocal⟩
            · exact Nat.sub_lt_sub_left ((UScalar.lt_equiv step limit).mp hlt) (by omega)
        | some slot =>
            refine ⟨Or.inr ⟨step.val, step, slot, rfl, rfl, ?_, ?_,
              hlocal, hprior⟩, ?_⟩
            · change step.val < next.val
              omega
            · have hnextBound := (UScalar.lt_equiv step limit).mp hlt
              change next.val ≤ limit.val
              omega
            · exact Nat.sub_lt_sub_left ((UScalar.lt_equiv step limit).mp hlt) (by omega)
      · have hstop : step.val = limit.val := by
          have hb : step.val ≤ limit.val := hbound
          have hnot : ¬ step.val < limit.val := by
            intro hval
            exact hlt ((UScalar.lt_equiv step limit).mpr hval)
          omega
        unfold table.probe.spill_read_loop0.body
        simp only [hlimit, bind_tc_ok, if_neg hlt, WP.spec_ok]
        refine Or.inl ⟨rfl, ?_⟩
        intro j hj
        exact hprior j (by simpa only [hstop] using hj)
    · unfold table.probe.spill_read_loop0.body
      by_cases hlt : step < limit
      · simp only [hlimit, bind_tc_ok, if_pos hlt, core.option.Option.is_none,
          Option.isNone_some, Bool.false_eq_true, if_false, WP.spec_ok]
        refine Or.inr ⟨j, cursor, slot, rfl, hcursor, ?_, hfound, hprior⟩
        omega
      · simp only [hlimit, bind_tc_ok, if_neg hlt, WP.spec_ok]
        refine Or.inr ⟨j, cursor, slot, rfl, hcursor, ?_, hfound, hprior⟩
        omega
  · left
    refine ⟨rfl, ?_, ?_⟩
    · exact Nat.zero_le _
    · intro j hj
      simp at hj

/-- Project the complete finite step sweep from a successful `none` result. Each
step's `OuterStepResult` includes the landed inner-loop outcome at that group's mask. -/
theorem generated_outer_none_covers_steps {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (tags : alloc.vec.Vec U8)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8) (limit : Usize)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hready : ∀ step : Usize, step < limit →
      OuterStepReady keyRep tags slots sigma place direct key n mask wanted step)
    (hresult : table.probe.spill_read_loop0 keyRep tags slots sigma place direct key n
      none groups mask wanted 0#usize = ok none) :
    ∀ j, j < limit.val →
      ∃ cursor : Usize, cursor.val = j ∧
        OuterStepResult keyRep tags slots sigma place direct key n mask wanted cursor none := by
  have hpost := generated_outer_sweep_outcome keyRep tags slots sigma place direct key
    n groups mask wanted limit hlimit hready
  rw [hresult, WP.spec_ok] at hpost
  rcases hpost with ⟨_, hsteps⟩ | ⟨_, _, _, hsome, _, _, _, _⟩
  · exact hsteps
  · cases hsome

/-- Under the still-explicit per-step readiness premise, a generated `ok none`
has executed an inner `none` sweep in every group. This does not derive readiness,
tag/payload well-formedness, or the whole-table absence statement. -/
theorem generated_outer_none_covers_groups {K H R V : Type}
    (keyRep : table.keyrep.KeyRep R K H) (tags : alloc.vec.Vec U8)
    (slots : alloc.vec.Vec (cell.Sigma × R × V)) (sigma : cell.Sigma)
    (place : U64) (direct : Option U32) (key : K)
    (n groups mask : Usize) (wanted : U8) (limit : Usize)
    (hcap : 7 + groups.val ≤ Usize.max)
    (hpower : ∃ exponent : Nat, groups.val = 2 ^ exponent)
    (hmask : groups - 1#usize = ok mask)
    (hlimit : table.WINDOWS + groups = ok limit)
    (hready : ∀ step : Usize, step < limit →
      OuterStepReady keyRep tags slots sigma place direct key n mask wanted step)
    (hresult : table.probe.spill_read_loop0 keyRep tags slots sigma place direct key n
      none groups mask wanted 0#usize = ok none) :
    ∀ target : Usize, target.val < groups.val →
      ∃ step : Usize, step < limit ∧ table.group place step mask = ok target ∧
        OuterStepResult keyRep tags slots sigma place direct key n mask wanted step none := by
  obtain ⟨otherLimit, hotherLimit, hsurj⟩ :=
    generated_tail_group_surjective place groups mask hcap hpower hmask
  have hlimits : otherLimit = limit :=
    Result.ok_injective (hotherLimit.symm.trans hlimit)
  subst otherLimit
  intro target htarget
  obtain ⟨step, _, hstep, hgroup⟩ := hsurj target htarget
  have hlt := (UScalar.lt_equiv step limit).mp hstep
  obtain ⟨cursor, hcursor, hdone⟩ :=
    generated_outer_none_covers_steps keyRep tags slots sigma place direct key
      n groups mask wanted limit hlimit hready hresult step.val hlt
  have heq : cursor = step := UScalar.eq_of_val_eq hcursor
  subst cursor
  exact ⟨step, hstep, hgroup, hdone⟩

-- Local fail-closed cone check until the integration owner registers these declarations in
-- AxiomAudit.lean. The shared audit file is outside this task's ownership.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let external : Array Lean.Name := #[``Apeiron.GrowableRepresentation.summary_set_frames,
    ``Apeiron.GrowableRepresentation.publish_live_frames,
    ``Apeiron.GrowableRepresentation.word_stored_vacant_spill_return_shards,
    ``Apeiron.GrowableRepresentation.word_stored_vacant_flat_return_frame]
  let localNames : Array Lean.Name := #[`StoredLiveTagAddress, `StoredTaggedRepR,
    `stored_live_tag_address_of_empty_repR, `word_stored_table_new_tagged_repR,
    `stored_live_tags_initial_reserved, `prepared_stored_live_tags_initial,
    `stored_live_tag_address_of_home_shards_eq, `stored_live_tag_address_replace_spill_value,
    `word_stored_apply_insert_found_spill_keeps_tags, `word_stored_apply_insert_vacant_spill_keeps_tags,
    `word_stored_apply_insert_found_flat_keeps_tags, `word_stored_apply_insert_vacant_flat_keeps_tags,
    `generated_publish_live_frames, `stored_live_tag_address_publish_spill_of_frame,
    `stored_live_tag_address_publish_spill, `InnerCompared, `stored_matcher_total,
    `repR_stored_published_tag_address, `power_of_two_groups_headroom, `InnerReady,
    `InnerSweepInv, `InnerSweepOutcome, `inner_ready_of_scan_geometry,
    `generated_inner_sweep_outcome, `generated_outer_increment_checked, `generated_group_bounded,
    `generated_tail_group_surjective, `OuterStepReady, `OuterStepResult,
    `OuterSweepInv, `OuterSweepOutcome, `outer_step_ready_of_geometry,
    `GroupedReadAdmission, `grouped_admission_of_checked_branch, `repR_region_outer_ready,
    `repR_stored_region_outer_ready, `repR_summary_read_branch, `repR_live_spill_summary_positive,
    `generated_outer_sweep_outcome, `generated_outer_none_covers_steps,
    `generated_outer_none_covers_groups]
  for decl in external ++ localNames.map (Lean.Name.append `Apeiron.GrowableReadLoop) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

/-- An empty-tag group for the last tail step; the tagged payload array has the same extent.
This fixture exercises the generated *outer body*, not the complete spill-read operation. -/
private def emptyTailTags : alloc.vec.Vec U8 :=
  alloc.vec.Vec.from (List.replicate 32 (0#u8)) (by scalar_tac)

private def emptyTailSlots :
    alloc.vec.Vec (cell.Sigma × Apeiron.GrowableIdentity.StoredI64 × I64) :=
  alloc.vec.Vec.from
    (List.replicate 32 (cell.Sigma.Parabolic, { key := 0#i64 }, 0#i64)) (by scalar_tac)

-- The inner sweep must inspect lane zero rather than treating a nonempty mask as exhausted.
#guard match (table.probe.spill_read_loop0_loop0
    Apeiron.GrowableIdentity.storedKeyRep emptyTailSlots cell.Sigma.Parabolic
    0#u64 none 0#i64 none 0#usize 1#u64).match with
  | .ok (some slot) => slot == 0#usize
  | _ => false

#guard match (table.probe.spill_read_loop0_loop0
    Apeiron.GrowableIdentity.storedKeyRep emptyTailSlots cell.Sigma.Parabolic
    0#u64 none 1#i64 none 0#usize 1#u64).match with
  | .ok none => true
  | _ => false

-- At groups=2 the last tail step is 8; reducing the generated bound to WINDOWS+mask skips it.
#guard match (table.probe.spill_read_loop0.body
    Apeiron.GrowableIdentity.storedKeyRep emptyTailTags emptyTailSlots
    cell.Sigma.Parabolic 0#u64 none 0#i64 16#usize 2#usize 1#usize
    128#u8 none 8#usize).match with
  | .ok (.cont (none, step)) => step == 9#usize
  | _ => false

-- The cyclic suffix reaches group seven; truncating its offset to two bits skips it.
#guard match (table.group 0#u64 13#usize 7#usize).match with
  | .ok group => group == 7#usize
  | _ => false

/-- A positive-summary region with vacancies before a full hit at index 31, in group 1 of 2. -/
private def lateHitRegion : table.Region Apeiron.GrowableIdentity.StoredI64 I64 :=
  { tags := alloc.vec.Vec.from (List.replicate 31 (0#u8) ++ [128#u8]) (by scalar_tac)
    slots := emptyTailSlots
    summary := alloc.vec.Vec.new U64 }

#guard match (table.Region.summary_maybe lateHitRegion 0#u64).match with
  | .ok true => true
  | _ => false

-- Actual publication must write the generated tag at the selected previously
-- vacant lane; a summary-only publication is insufficient for this invariant.
#guard match (table.Region.publish_live lateHitRegion 0#usize
    (cell.Sigma.Parabolic, { key := 0#i64 }, 0#i64) 0#u64).match with
  | .ok published =>
      match (table.tag 0#u64).match with
      | .ok expected => published.tags.val[0]? == some expected
      | _ => false
  | _ => false

-- The live fingerprint is the generated tag, not merely a nonzero occupancy byte.
#guard match (table.tag 0#u64).match, lateHitRegion.tags.val[31]? with
  | .ok expected, some stored => stored == expected
  | _, _ => false

-- A positive summary can still be a false positive for a distinct full key.
#guard match (table.Region.summary_maybe lateHitRegion 0#u64).match,
    (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep
      lateHitRegion cell.Sigma.Parabolic
      { place := 0#u64, direct_word := none } 1#i64 16#usize).match with
  | .ok true, .ok none => true
  | _, _ => false

-- Joint concrete sizing, mask, outer bound and positive-summary control at the word types.
-- This fixture is not a witness that a public constructor establishes `RepR`.
#guard match (table.Region.summary_maybe lateHitRegion 0#u64).match,
    (2#usize - 1#usize).match, (table.WINDOWS + 2#usize).match with
  | .ok true, .ok mask, .ok limit =>
      lateHitRegion.tags.length == lateHitRegion.slots.length &&
      lateHitRegion.slots.length == 32 && mask == 1#usize && limit == 9#usize
  | _, _, _ => false

-- The real outer body reads the last lane of group one through the checked
-- payload index; shortening the payload vector makes this step fail.
#guard match (table.probe.spill_read_loop0.body
    Apeiron.GrowableIdentity.storedKeyRep lateHitRegion.tags lateHitRegion.slots
    cell.Sigma.Parabolic 0#u64 none 0#i64 16#usize 2#usize 1#usize
    128#u8 none 7#usize).match with
  | .ok (.cont (some slot, next)) => slot == 31#usize && next == 8#usize
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep
    lateHitRegion cell.Sigma.Parabolic
    { place := 0#u64, direct_word := none } 0#i64 16#usize).match with
  | .ok (some index) => index == 31#usize
  | _ => false

-- A distinct fingerprint visits no candidate; a tag collision with a distinct key also misses.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep
    lateHitRegion cell.Sigma.Parabolic
    { place := 256#u64, direct_word := none } 0#i64 16#usize).match with
  | .ok none => true
  | _ => false

#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep
    lateHitRegion cell.Sigma.Parabolic
    { place := 0#u64, direct_word := none } 1#i64 16#usize).match with
  | .ok none => true
  | _ => false

/-!
## Logical-absence and full-match composition for the generated spill read

The generated `table.probe.spill_read` is composed with the landed sweep facts into the two
read-outcome claims at the stored-word carrier. Both compositions run on the generated body with
its admission checks in Result form; the live tag/address invariant is what turns an executed
`none` sweep into logical absence of a full `(Sigma, key)` match. Nothing here proves
`TableLocator.retarget_held` total: `Apeiron.GrowableRelocation.retarget_held_succeeds_iff` and
`retarget_held_refuses_of_linked_handle_fail` show it can refuse on a linked-handle failure, so a
post-rebuild read carries the rebuild traversal certificate (its exact `hstep` premise) or a
locator well-formedness premise supplying it, as an explicit premise.
-/

/-- A full `(Sigma, key)` match against the stored carrier fixes both components. -/
private theorem stored_full_key_components
    {slotSigma sigma : cell.Sigma} {held : Apeiron.GrowableIdentity.StoredI64} {key : I64}
    (h : Apeiron.GrowableIdentity.logicalStored slotSigma held = (sigma, key)) :
    slotSigma = sigma ∧ held.key = key := by
  have hp : (slotSigma, Apeiron.GrowableIdentity.decodeStored slotSigma held) = (sigma, key) := h
  rcases Prod.mk.inj hp with ⟨hsigma, hkey⟩
  exact ⟨hsigma, hkey⟩

/-- `Nat.div_add_mod` with the product in `x / y * y` order, matching the generated scalar
postconditions `z.val = x.val * y.val`. -/
private theorem div_add_mod_left (x y : Nat) : x / y * y + x % y = x := by
  rw [Nat.mul_comm]
  exact Nat.div_add_mod x y

/-- A successful generated payload lookup reads exactly the stored list entry. -/
private theorem vec_index_readback {T : Type} (slots : alloc.vec.Vec T) (i : Usize) (entry : T)
    (hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice T) slots i = ok entry) :
    slots.val[i.val]? = some entry := by
  rw [alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize] at hindex
  have hindex' : (match slots.val[i.val]? with
      | none => (fail .arrayOutOfBounds : Result T)
      | some x => ok x) = ok entry := hindex
  cases hget : slots.val[i.val]? with
  | none => simp [hget] at hindex'
  | some payload =>
      simp only [hget] at hindex'
      exact congrArg Option.some (Result.ok_injective hindex')

/-- A successful generated tag-slice read returns the sliced bytes with their exact extent. -/
private theorem range_index_readback (tags : alloc.vec.Vec U8) (base finish : Usize)
    (line : Slice U8)
    (horder : base ≤ finish) (hbound : finish.val ≤ tags.length)
    (hline : alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice U8)
      tags { start := base, «end» := finish } = ok line) :
    line.val = tags.val.slice base.val finish.val ∧
      line.val.length = finish.val - base.val := by
  have hspec := alloc.vec.Vec.index_Range_spec tags { start := base, «end» := finish }
    horder hbound
  rw [hline, WP.spec_ok] at hspec
  refine ⟨hspec.1, ?_⟩
  have hlen := hspec.2
  change line.val.length = finish.val - base.val at hlen
  exact hlen

/-- The generated `spill_read` reaches its outer sweep exactly when its admitted checks pass.
Premises: `hslots` is the exact grouped sizing of the payload array; `hnPos` and `hgroups2` are
the numeric halves of the generated `valid_n` and `groups >= 2` checks; `hvalid`, `hpowCheck` and
`hsummary` are the generated `valid_n`, `is_power_of_two` and `summary_maybe` checks in their
Result form; `hmask` pins the checked `groups - 1` mask. The wanted tag is read back from the run
itself, so no tag totality is assumed. -/
theorem generated_spill_read_unfold
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (n groups mask : Usize) (result : Option Usize)
    (hslots : region.slots.length = groups.val * n.val)
    (hnPos : 0 < n.val)
    (hgroups2 : 2 ≤ groups.val)
    (hvalid : table.valid_n n = ok true)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hmask : groups - 1#usize = ok mask)
    (hsummary : table.Region.summary_maybe region address.place = ok true)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok result) :
    ∃ wanted : U8, table.tag address.place = ok wanted ∧
      table.probe.spill_read_loop0 Apeiron.GrowableIdentity.storedKeyRep region.tags region.slots
        sigma address.place address.direct_word key n none groups mask wanted 0#usize = ok result := by
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
          table.probe.spill_read_loop0 Apeiron.GrowableIdentity.storedKeyRep region.tags
            region.slots sigma address.place address.direct_word key n none groups mask wanted
            0#usize) = ok result := by
    unfold table.probe.spill_read at hresult
    simp only [hneZeroB, hvalid, hqrun, ge_iff_le, hge, hprodRun, hprodEq,
      hpowCheck, hsummary, hmask, bind_tc_ok, ↓reduceIte] at hresult
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
      refine ⟨wanted, rfl, ?_⟩
      rw [htagres, bind_tc_ok] at hreduced
      exact hreduced

/-- Deliverable 1: on an admitted region under a positive summary and the live tag/address
invariant, a generated `spill_read` `none` result implies no live slot holds the query under full
`(Sigma, key)` equality. Premises: `hstate`/`hregion` are the live tag/address invariant
(`StoredLiveTagAddress`) and the shard selection of the scanned region; `haddress` is the query
routing equation for the scanned region's address; `hadmit` is the generated grouped admission
(`GroupedReadAdmission`, whose `summaryPositive` field is the positive summary) at `n groups mask
limit`; `htagSlots` is the tag/payload extent equality (from `RepR.directory` in
`repR_region_outer_ready`); `hnPos`, `hgroups2`, `hvalid`, `hpowCheck` are the generated `valid_n`
lower bound, `groups >= 2` and `valid_n`/`is_power_of_two` checks; `hresult` is the generated
`none` run. -/
theorem spill_read_none_implies_no_full_match
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (n groups mask limit : Usize)
    (hadmit : GroupedReadAdmission region address.place n groups mask limit)
    (htagSlots : region.tags.length = region.slots.length)
    (hnPos : 0 < n.val)
    (hgroups2 : 2 ≤ groups.val)
    (hvalid : table.valid_n n = ok true)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none) :
    ∀ (index : Usize) (tag : U8) (slotSigma : cell.Sigma)
      (held : Apeiron.GrowableIdentity.StoredI64) (value : I64),
      region.tags.val[index.val]? = some tag → tag ≠ 0#u8 →
      region.slots.val[index.val]? = some (slotSigma, held, value) →
      Apeiron.GrowableIdentity.logicalStored slotSigma held ≠ (sigma, key) := by
  intro index tag slotSigma held value htag htagNonzero hslot hfull
  obtain ⟨wanted, hwanted, hloop⟩ := generated_spill_read_unfold region sigma address key n
    groups mask none hadmit.groupSlots hnPos hgroups2 hvalid hpowCheck hadmit.maskResult
    hadmit.summaryPositive hresult
  have hready : ∀ step : Usize, step < limit →
      OuterStepReady Apeiron.GrowableIdentity.storedKeyRep region.tags region.slots
        sigma address.place address.direct_word key n mask wanted step := by
    intro step hstep
    exact outer_step_ready_of_geometry Apeiron.GrowableIdentity.storedKeyRep region.tags
      region.slots sigma address.place address.direct_word key n groups mask wanted limit step
      hadmit.width ⟨htagSlots, hadmit.groupSlots⟩ hadmit.headroom hadmit.power
      hadmit.maskResult hadmit.boundResult hstep
      (fun entry => stored_matcher_total entry sigma key
        { place := address.place, direct_word := address.direct_word })
  have hcovers := generated_outer_none_covers_groups
    Apeiron.GrowableIdentity.storedKeyRep region.tags region.slots sigma address.place
    address.direct_word key n groups mask wanted limit hadmit.headroom hadmit.power
    hadmit.maskResult hadmit.boundResult hready hloop
  have ⟨hsigma, hkey⟩ := stored_full_key_components hfull
  obtain ⟨laddr, hladdr, _, htagAt⟩ :=
    hstate shard index region tag slotSigma held value hregion htag htagNonzero hslot
  rw [hsigma, hkey] at hladdr
  have hladdrEq : laddr = address := Result.ok_injective (hladdr.symm.trans haddress)
  subst laddr
  have htagEq : tag = wanted := Result.ok_injective (htagAt.symm.trans hwanted)
  have hslotsLen : index.val < region.slots.length :=
    (List.getElem?_eq_some_iff.mp hslot).1
  have hslotsProp := region.slots.property
  change region.slots.length ≤ Usize.max at hslotsProp
  have hg : index.val / n.val < groups.val := by
    by_contra hnot
    have hge : groups.val ≤ index.val / n.val := by omega
    have hbig : groups.val * n.val ≤ index.val / n.val * n.val :=
      Nat.mul_le_mul_right n.val hge
    have hsplit : index.val / n.val * n.val + index.val % n.val = index.val :=
      div_add_mod_left index.val n.val
    have hslotsEq := hadmit.groupSlots
    omega
  have hj : index.val % n.val < n.val := Nat.mod_lt _ hnPos
  have hgwidth : index.val / n.val < 2 ^ System.Platform.numBits := by
    have hbound := groups.hBounds
    simp only [UScalarTy.Usize_numBits_eq] at hbound
    omega
  let target : Usize := ⟨BitVec.ofNat System.Platform.numBits (index.val / n.val)⟩
  have htarget : target.val = index.val / n.val := by
    have hgen : ∀ k : Nat, k < 2 ^ System.Platform.numBits →
        (⟨BitVec.ofNat System.Platform.numBits k⟩ : Usize).val = k := by
      intro k hk
      simp [UScalar.val, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hk]
    exact hgen (index.val / n.val) hgwidth
  have hgroupNext : target.val + 1 ≤ groups.val := by rw [htarget]; omega
  have hmono : (target.val + 1) * n.val ≤ groups.val * n.val :=
    Nat.mul_le_mul_right n.val hgroupNext
  have hnext : (target.val + 1) * n.val = target.val * n.val + n.val := by ring
  obtain ⟨step, _, hgroupStep, houterNone⟩ := hcovers target (by rwa [htarget])
  obtain ⟨group, base, finish, line, bits, hgroupRun, hbaseRun, hfinishRun, hline, hbits, _,
    houtcome⟩ := houterNone
  have hgroupEq : group = target := Result.ok_injective (hgroupRun.symm.trans hgroupStep)
  subst group
  have hbaseCap : target.val * n.val ≤ Usize.max := by
    have hslotsEq := hadmit.groupSlots
    omega
  obtain ⟨bprod, hbprodRun, hbprodVal⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := target) (y := n) hbaseCap)
  have hbprodEq : bprod = base := Result.ok_injective (hbprodRun.symm.trans hbaseRun)
  subst bprod
  have hbaseVal : base.val = target.val * n.val := by simpa using hbprodVal
  have hfinishCap : base.val + n.val ≤ Usize.max := by
    have hslotsEq := hadmit.groupSlots
    omega
  obtain ⟨fsum, hfsumRun, hfsumVal⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := base) (y := n) hfinishCap)
  have hfsumEq : fsum = finish := Result.ok_injective (hfsumRun.symm.trans hfinishRun)
  subst fsum
  have hfinishVal : finish.val = base.val + n.val := by simpa using hfsumVal
  have horder : base ≤ finish := (UScalar.le_equiv base finish).mpr (by rw [hfinishVal]; omega)
  have hfinishBound : finish.val ≤ region.tags.length := by
    rw [htagSlots]
    have hslotsEq := hadmit.groupSlots
    omega
  obtain ⟨hlineVal, hlineLength⟩ :=
    range_index_readback region.tags base finish line horder hfinishBound hline
  have hlineWidth : line.val.length = n.val := by rw [hlineLength, hfinishVal]; omega
  obtain ⟨bits', hbits', hexact⟩ := Apeiron.GrowableScan.match_tag_exact line wanted (by
    rw [hlineWidth]
    exact hadmit.width)
  have hbitsEq : bits' = bits := Result.ok_injective (hbits'.symm.trans hbits)
  subst bits'
  have hbaseIndex : base.val + index.val % n.val = index.val := by
    rw [hbaseVal, htarget]
    exact div_add_mod_left index.val n.val
  have hslice : line.val[index.val % n.val]? = region.tags.val[base.val + index.val % n.val]? := by
    rw [hlineVal]
    exact List.getElem?_slice base.val finish.val (index.val % n.val) region.tags.val
      ⟨hfinishBound, by omega⟩
  have hbit : bits.bv.getLsbD (index.val % n.val) = true := by
    rw [hexact]
    simp only [decide_eq_true_eq]
    refine ⟨by rw [hlineWidth]; exact hj, ?_⟩
    rw [hslice, hbaseIndex, htag, htagEq]
  rcases houtcome with ⟨_, hcmp⟩ | ⟨_, _, habs, _⟩
  · obtain ⟨hitSlot, hcomp⟩ := hcmp (index.val % n.val) hbit
    obtain ⟨lane, entry, hlane, hadd, hindexE, hmatchFalse⟩ := hcomp
    have haddCap : base.val + lane.val ≤ Usize.max := by rw [hlane, hbaseIndex]; omega
    have haddPost := Usize.add_spec (x := base) (y := lane) haddCap
    rw [hadd, WP.spec_ok] at haddPost
    have hslotVal : hitSlot.val = index.val := by
      have haddVal : hitSlot.val = base.val + lane.val := by simpa using haddPost
      rw [haddVal, hlane, hbaseIndex]
    have hentryEq : region.slots.val[hitSlot.val]? = some entry :=
      vec_index_readback region.slots hitSlot entry hindexE
    have hpayload : entry = (slotSigma, held, value) := by
      have h1 := hentryEq
      rw [hslotVal] at h1
      exact Option.some.inj (h1.symm.trans hslot)
    subst entry
    have hmatchTrue :
        table.matches Apeiron.GrowableIdentity.storedKeyRep (slotSigma, held, value) sigma key
          { place := address.place, direct_word := address.direct_word } = ok true :=
      (Apeiron.GrowableIdentity.stored_matches_iff slotSigma sigma held value key
        { place := address.place, direct_word := address.direct_word }).mpr hfull
    have hcontra : (true : Bool) = false :=
      Result.ok_injective (hmatchTrue.symm.trans hmatchFalse)
    simp at hcontra
  · cases habs

/-- Deliverable 2: on an admitted region, a generated `spill_read` `ok (some slot)` result returns
an in-bounds slot whose payload is a full `(Sigma, key)` match for the query. Premises: `hadmit`
is the generated grouped admission at `n groups mask limit`; `htagSlots` is the tag/payload extent
equality; `hnPos`, `hgroups2`, `hvalid`, `hpowCheck` are the generated `valid_n` lower bound,
`groups >= 2` and `valid_n`/`is_power_of_two` checks; `hresult` is the generated hit run. No state
invariant is assumed. -/
theorem spill_read_some_implies_full_match
    (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (sigma : cell.Sigma) (address : table.TableAddress) (key : I64)
    (n groups mask limit : Usize) (slot : Usize)
    (hadmit : GroupedReadAdmission region address.place n groups mask limit)
    (htagSlots : region.tags.length = region.slots.length)
    (hnPos : 0 < n.val)
    (hgroups2 : 2 ≤ groups.val)
    (hvalid : table.valid_n n = ok true)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok (some slot)) :
    slot.val < region.slots.length ∧
    ∃ entry : cell.Sigma × Apeiron.GrowableIdentity.StoredI64 × I64,
      region.slots.val[slot.val]? = some entry ∧
      Apeiron.GrowableIdentity.logicalStored entry.1 entry.2.1 = (sigma, key) := by
  obtain ⟨wanted, _, hloop⟩ := generated_spill_read_unfold region sigma address key n
    groups mask (some slot) hadmit.groupSlots hnPos hgroups2 hvalid hpowCheck hadmit.maskResult
    hadmit.summaryPositive hresult
  have hready : ∀ step : Usize, step < limit →
      OuterStepReady Apeiron.GrowableIdentity.storedKeyRep region.tags region.slots
        sigma address.place address.direct_word key n mask wanted step := by
    intro step hstep
    exact outer_step_ready_of_geometry Apeiron.GrowableIdentity.storedKeyRep region.tags
      region.slots sigma address.place address.direct_word key n groups mask wanted limit step
      hadmit.width ⟨htagSlots, hadmit.groupSlots⟩ hadmit.headroom hadmit.power
      hadmit.maskResult hadmit.boundResult hstep
      (fun entry => stored_matcher_total entry sigma key
        { place := address.place, direct_word := address.direct_word })
  have hpost := generated_outer_sweep_outcome Apeiron.GrowableIdentity.storedKeyRep region.tags
    region.slots sigma address.place address.direct_word key n groups mask wanted limit
    hadmit.boundResult hready
  rw [hloop, WP.spec_ok] at hpost
  rcases hpost with ⟨hsome, _⟩ | ⟨j, cursor, slot', hsome, _, _, houter, _⟩
  · cases hsome
  · have hslotEq : slot = slot' := Option.some.inj hsome
    subst slot'
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, houtcome⟩ := houter
    rcases houtcome with ⟨hnone, _⟩ | ⟨_, slot2, hsome2, _, hcomp⟩
    · cases hnone
    · have hslotEq2 : slot = slot2 := Option.some.inj hsome2
      subst slot2
      obtain ⟨_, entry, _, _, hindexE, hmatchTrue⟩ := hcomp
      obtain ⟨hinBounds, hlogical⟩ := stored_candidate_full_match region.slots slot entry sigma key
        { place := address.place, direct_word := address.direct_word } hindexE hmatchTrue
      exact ⟨hinBounds, entry,
        vec_index_readback region.slots slot entry hindexE, hlogical⟩

/-- Whole-shard form of the absence composition. Extra premise beyond
`spill_read_none_implies_no_full_match`: `hshard`, the query routing to the scanned shard. -/
theorem spill_read_none_implies_no_shard_full_match
    (state : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (hstate : StoredLiveTagAddress state)
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : state.shards.val[shard.val]? = some region)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address state.home sigma key = ok address)
    (hshard : table.shard address = ok shard)
    (n groups mask limit : Usize)
    (hadmit : GroupedReadAdmission region address.place n groups mask limit)
    (htagSlots : region.tags.length = region.slots.length)
    (hnPos : 0 < n.val)
    (hgroups2 : 2 ≤ groups.val)
    (hvalid : table.valid_n n = ok true)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none) :
    ∀ (shard' : Usize) (region' : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
      (index : Usize) (tag : U8) (slotSigma : cell.Sigma)
      (held : Apeiron.GrowableIdentity.StoredI64) (value : I64),
      state.shards.val[shard'.val]? = some region' →
      region'.tags.val[index.val]? = some tag → tag ≠ 0#u8 →
      region'.slots.val[index.val]? = some (slotSigma, held, value) →
      Apeiron.GrowableIdentity.logicalStored slotSigma held ≠ (sigma, key) := by
  intro shard' region' index tag slotSigma held value hregion' htag htagNonzero hslot hfull
  have habsent := spill_read_none_implies_no_full_match state hstate shard region hregion sigma
    key address haddress n groups mask limit hadmit htagSlots hnPos hgroups2 hvalid hpowCheck
    hresult
  obtain ⟨laddr, hladdr, hshardAt, _⟩ :=
    hstate shard' index region' tag slotSigma held value hregion' htag htagNonzero hslot
  have ⟨hsigma, hkey⟩ := stored_full_key_components hfull
  rw [hsigma, hkey] at hladdr
  have hladdrEq : laddr = address := Result.ok_injective (hladdr.symm.trans haddress)
  subst laddr
  have hshardEq : shard' = shard := Result.ok_injective (hshardAt.symm.trans hshard)
  subst shard'
  have hregionEq : region' = region := Option.some.inj (hregion'.symm.trans hregion)
  subst region'
  exact habsent index tag slotSigma held value htag htagNonzero hslot hfull

/-- Rebuild-certificate form of the absence composition, for a read on the rebuilt view after the
generated stored-carrier relocation traversal. The live tag/address invariant is supplied by
`word_stored_relocate_all_keeps_reserved_tags_from_prepared` under its exact per-step premise
`hstep` (quoted verbatim below); `retarget_held` totality from `sourcePayload` alone is disproved
(`Apeiron.GrowableRelocation.retarget_held_succeeds_iff`,
`retarget_held_refuses_of_linked_handle_fail`), so `hstep` is assumed, not discharged, and no
locator well-formedness premise is invented. Additional premises beyond
`spill_read_none_implies_no_full_match`: `hstep`, the traversal step certificate; `hrelocate`, the
executed traversal run; `haddress` is stated at the rebuilt view's home dictionary. -/
theorem spill_read_after_relocate_all_none_implies_no_full_match
    (certificate : Apeiron.GrowableResources.PreparedResourceClauses
      Apeiron.GrowableIdentity.StoredI64)
    (plan : table.RebuildPlan)
    (hstep : ∀ (s : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
      (current : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64) (entryAt : Usize),
      entryAt.val < plan.entries.length →
      ReservedStoredLiveTagAddress s current →
      word_stored_relocate_entry s plan current entryAt ⦃ result =>
        ReservedStoredLiveTagAddress result.1 result.2 ⦄)
    (nextState : Apeiron.GrowableRepresentation.WordTable Apeiron.GrowableIdentity.StoredI64)
    (nextReserved : table.RebuildReserved Apeiron.GrowableIdentity.StoredI64 I64)
    (hrelocate : word_stored_relocate_all certificate.state plan certificate.storage =
      ok (nextState, nextReserved))
    (shard : Usize) (region : table.Region Apeiron.GrowableIdentity.StoredI64 I64)
    (hregion : (Apeiron.GrowableRefinement.rebuiltView nextState nextReserved).shards.val[shard.val]? =
      some region)
    (sigma : cell.Sigma) (key : I64) (address : table.TableAddress)
    (haddress : Apeiron.GrowableIdentity.wordHomeInst.address
      (Apeiron.GrowableRefinement.rebuiltView nextState nextReserved).home sigma key = ok address)
    (n groups mask limit : Usize)
    (hadmit : GroupedReadAdmission region address.place n groups mask limit)
    (htagSlots : region.tags.length = region.slots.length)
    (hnPos : 0 < n.val)
    (hgroups2 : 2 ≤ groups.val)
    (hvalid : table.valid_n n = ok true)
    (hpowCheck : core.num.Usize.is_power_of_two groups = ok true)
    (hresult : table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep region sigma address
      key n = ok none) :
    ∀ (index : Usize) (tag : U8) (slotSigma : cell.Sigma)
      (held : Apeiron.GrowableIdentity.StoredI64) (value : I64),
      region.tags.val[index.val]? = some tag → tag ≠ 0#u8 →
      region.slots.val[index.val]? = some (slotSigma, held, value) →
      Apeiron.GrowableIdentity.logicalStored slotSigma held ≠ (sigma, key) := by
  have hkeep := word_stored_relocate_all_keeps_reserved_tags_from_prepared certificate plan hstep
  rw [hrelocate, WP.spec_ok] at hkeep
  exact spill_read_none_implies_no_full_match
    (Apeiron.GrowableRefinement.rebuiltView nextState nextReserved) hkeep shard region hregion
    sigma key address haddress n groups mask limit hadmit htagSlots hnPos hgroups2 hvalid hpowCheck
    hresult

-- Local fail-closed cone check for the read-composition block. The shared audit file is outside
-- this task's ownership; the integration owner registers these names in AxiomAudit.lean.
open Lean Elab Command in
run_cmd do
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let external : Array Lean.Name := #[``Apeiron.GrowableScan.match_tag_exact,
    ``Apeiron.GrowableIdentity.stored_matches_iff,
    ``Apeiron.GrowableReadLoop.word_stored_relocate_all_keeps_reserved_tags_from_prepared,
    ``Apeiron.GrowableRepresentation.word_stored_relocate_all_preserves_reserved]
  let localNames : Array Lean.Name := #[`generated_spill_read_unfold,
    `spill_read_none_implies_no_full_match, `spill_read_some_implies_full_match,
    `spill_read_none_implies_no_shard_full_match,
    `spill_read_after_relocate_all_none_implies_no_full_match]
  for decl in external ++ localNames.map (Lean.Name.append `Apeiron.GrowableReadLoop) do
    let axioms ← liftTermElabM <| Lean.collectAxioms decl
    let bad := axioms.filter (fun ax => !allowed.contains ax)
    unless bad.isEmpty do
      throwError m!"'{decl}' depends on axioms outside the closed cone: {bad.toList}"
    logInfo m!"'{decl}' depends on axioms: {axioms.toList}"

-- Deliverable-2 fixture of the composition: a positive generated read returns an in-bounds slot
-- whose held key is a full match under the real generated matcher. A tag/address disagreement at
-- lane 31 (stored tag ≠ generated tag of the held key's home) rejects this fixture.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep
    lateHitRegion cell.Sigma.Parabolic
    { place := 0#u64, direct_word := none } 0#i64 16#usize).match with
  | .ok (some index) =>
      match lateHitRegion.slots.val[index.val]? with
      | some (slotSigma, held, value) =>
          (decide (index.val < 32)) &&
            (match (table.matches Apeiron.GrowableIdentity.storedKeyRep
              (slotSigma, held, value) cell.Sigma.Parabolic 0#i64
              { place := 0#u64, direct_word := none }).match with
             | .ok matched => matched
             | _ => false)
      | none => false
  | _ => false

-- Deliverable-1 fixture of the composition: the `none` read for key 1 has no live lane whose
-- held key is a full match; a summary/live-tag correspondence corruption that surfaces a live
-- full match for key 1 rejects this fixture.
#guard match (table.probe.spill_read Apeiron.GrowableIdentity.storedKeyRep
    lateHitRegion cell.Sigma.Parabolic
    { place := 0#u64, direct_word := none } 1#i64 16#usize).match,
    lateHitRegion.slots.val[31]? with
  | .ok none, some (slotSigma, held, value) =>
      (match (table.matches Apeiron.GrowableIdentity.storedKeyRep
        (slotSigma, held, value) cell.Sigma.Parabolic 1#i64
        { place := 0#u64, direct_word := none }).match with
       | .ok matched => !matched
       | _ => false)
  | _, _ => false

end Apeiron.GrowableReadLoop
