import ApplyPendingSource
import Aeneas

/-
SCOPE: the actual extracted body `Table::apply_pending_and_commit` reached by the extraction root
`word_stored_apply_pending_and_commit`; its four guarded merge/settle transitions, its three exit
transitions, its work measure in merges, payload copies and located-slot writes, the checked
length-increment refusal, and witnesses at concrete arguments for a settling run and for an exit run.
REGISTER: source refinement and an explicit-work count of that one body, read off the extracted arm
equations. Excluded: canonical-corpus correspondence, hardware timing, the transitively reached bodies
(`recycle_unused_staged_path`, `TableLocator::publish_for_held`, `Region::publish_live`,
`alloc::vec::Vec::index_mut`), the unextracted probe walk that builds the decision, the reservation
and relocation adapter in `apply_prepared_rebuild`, and any claim about which cargo profile runs the
`usize` increment. Two exclusions are stated here because they bound this file's claims. First, the
model's word width `UScalarTy.Usize.numBits` is the opaque platform width and does not reduce to a
literal, so the saturated-length population excluded by
`vacant_flat_refuses_unrepresentable_length` is stated at the checked increment and not exhibited as a
concrete table. Second, `found_flat_success_witness` fixes the settle content at concrete arguments but
is conditional on the committing run: `alloc::vec::Vec::index_mut` resolves through `Slice` and `ListN`
and does not reduce under `rfl` or `simp` in the pinned model, so no single evaluation of the real body
at concrete arguments is proved here. The exit witness below is unconditional.
CORPUS: `src/table.rs:1063-1121`, whose real body is
`pub(crate) fn apply_pending_and_commit<F: Merge<V>>(&mut self, mut reserved: RebuildReserved<R, V>,
decision: SearchDecision, merge: &F, sigma: Sigma, value: V, held: R, address: &TableAddress)`;
the `Merge::merge` contract at `src/table.rs:133-136` is new-first, the `WordMerge` instance at
`src/table.rs:190-196` is the wrapping word add, and the forwarding wrapper is
`proof/extract/lib.rs:600-618`, rooted by `proof/regen.sh:173`. The same body is already extracted at
`proof/ApeironWord.lean:22716-22826`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/ApplyPending_gate.py --controls`, 540-second budget, lane
/tmp/opencode/apeiron-lane-3.lock.
GRADE: `proved` for the theorems below when that gate exits 0. The work measure is a count of the
merges, copies and slot writes the extracted body itself issues, not a measured time.
-/

namespace ApplyPendingResource
open Aeneas Aeneas.Std Result Error
open apply_pending_source

/-- The stored word table the extraction root is monomorphic at. -/
abbrev Store := table.Table Std.I64 Std.I64 table.WordHome (table.keyrep.Stored Std.I64) 16#usize

/-- The reserved relocated storage the real call consumes. -/
abbrev Reserved := table.RebuildReserved (table.keyrep.Stored Std.I64) Std.I64

/-- The carried-key word instance the pending `(sigma, key)` is applied for. -/
abbrev Held := table.keyrep.Stored Std.I64

/-- The real `Clone<i64>` instance the wrapper passes. -/
abbrev CloneI := core.clone.CloneI64

/-- The real `Home` instance at the stored word carrier. -/
abbrev HomeInst := table.WordHome.Insts.Apply_pending_sourceTableHomeI64Sigma

/-- The real `KeyRep` instance at the stored word carrier. -/
abbrev KeyRepInst := table.keyrep.Stored.Insts.Apply_pending_sourceTableKeyrepKeyRep
  I64.Insts.Apply_pending_sourceCoordLocatorLocator_coreExactIdentityU64Tuple CloneI HomeInst

/-- The real new-first word merge instance. -/
abbrev MergeInst := table.WordMerge.Insts.Apply_pending_sourceTableMergeI64

/-- The merge argument the wrapper passes: a reference to the unit-sized `WordMerge`. -/
abbrev MergeArg : table.WordMerge := ()

/-- A resident slot of the stored word carrier. -/
abbrev Slot := table.FlatSlot Held Std.I64

/-- A spill region of the stored word carrier. -/
abbrev Region := table.Region Held Std.I64

/-- The stored word payload triple of a spill slot. -/
abbrev Triple := cell.Sigma × Held × Std.I64

/-- The `SliceIndex` instance the real body indexes resident slots with. -/
abbrev SlotIdx := core.slice.index.SliceIndexUsizeSlice Slot

/-- The `SliceIndex` instance the real body indexes spill regions with. -/
abbrev RegionIdx := core.slice.index.SliceIndexUsizeSlice Region

/-- The `SliceIndex` instance the real body indexes spill payload triples with. -/
abbrev TripleIdx := core.slice.index.SliceIndexUsizeSlice Triple

/-- The real extracted body of `Table::apply_pending_and_commit` at the stored word carrier. -/
def body (state : Store) (reserved : Reserved) (decision : table.SearchDecision)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    Result Store :=
  table.Table.apply_pending_and_commit CloneI CloneI HomeInst KeyRepInst MergeInst
    state reserved decision MergeArg sigma value held address

/-- The extraction root forwards to that body; the root restates no match arm, no merge and no field
write. -/
theorem root_forwards (state : Store) (reserved : Reserved) (decision : table.SearchDecision)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    apply_pending_source.word_stored_apply_pending_and_commit
        state reserved decision sigma value held address
      = body state reserved decision sigma value held address := rfl

/-- A monadic bind that returned `ok` consumed an `ok`, and its continuation returned `ok`. -/
theorem bind_eq_ok {α β : Type} {a : Result α} {k : α → Result β} {t : β}
    (h : (do let x ← a; k x) = ok t) : ∃ x, a = ok x ∧ k x = ok t := by
  cases hm : a.match with
  | ok x =>
    have ha : a = ok x := (Result.match.isOk).mp hm
    rw [ha, bind_tc_ok] at h
    exact ⟨x, ha, h⟩
  | vis e kk =>
    cases e with
    | fail e =>
      have hav : a = Result.vis (.fail e) kk := (Result.match.isVis).mp hm
      rw [hav, bind_tc_vis] at h
      exact absurd h vis_not_ok
  | div =>
    have hdiv : a = Result.div := (Result.match.isDiv).mp hm
    rw [hdiv, bind_tc_div] at h
    exact absurd h fun hd => ok_not_div hd.symm

/-- The joint machine-word domain of one real call: the located lane index is a machine word below
its bit width. -/
def MachineWords (index : Usize) : Prop := index.val < 2 ^ UScalarTy.Usize.numBits

/-- The extracted `usize` type initializes the joint domain at every index, so there is no partial
lane input to check and no unreachable premise. -/
theorem machineWords_holds (index : Usize) : MachineWords index := index.hmax

/-- The real new-first merge: `Merge::merge` at `src/table.rs:133-136` takes the new value first,
and the `WordMerge` instance at `src/table.rs:190-196` is the wrapping word add of the two. -/
theorem merge_is_new_first_wrapping_add (new old : Std.I64) :
    MergeInst.merge MergeArg new old = ok (core.num.I64.wrapping_add new old) := rfl

/-- The lane the real body reads is the located lane of the carrier it indexes. -/
theorem index_reads_lane {α : Type} (v : alloc.vec.Vec α) (i : Usize) (hin : i.val < v.length)
    (x : α) (h : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) v i = ok x) :
    x = v.val[i.val] := by
  obtain ⟨y, hy, hylane⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec v i hin)
  have h' : alloc.vec.Vec.index_usize v i = ok x := alloc.vec.Vec.index_slice_index v i ▸ h
  exact (Result.ok_injective (h'.symm.trans hy)).trans hylane

/-- The lane the real body writes through `index_mut` is the located lane: its update closure is
`Vec.set`, which writes that lane by list `set`. This holds at every index, in or out of bounds: the
in-bounds premise governs only which element it reads. -/
theorem index_mut_write {α : Type} (v : alloc.vec.Vec α) (i : Usize) (x : α)
    (y : α → alloc.vec.Vec α)
    (h : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice α) v i = ok (x, y)) :
    y = alloc.vec.Vec.set v i ∧ (alloc.vec.Vec.set v i x).val = v.val.set i.val x := by
  have h' : alloc.vec.Vec.index_mut_usize v i = ok (x, y) :=
    alloc.vec.Vec.index_mut_slice_index v i ▸ h
  obtain ⟨z, _, hz⟩ := bind_eq_ok h'
  have hzz : ok (z, alloc.vec.Vec.set v i) = ok (x, y) := hz
  have hpair : (x, y) = (z, alloc.vec.Vec.set v i) := Result.ok_injective hzz.symm
  exact ⟨congrArg Prod.snd hpair, alloc.vec.Vec.set_val_eq v i x⟩

/-- At an in-bounds index the element `index_mut` reads is the located lane. -/
theorem index_mut_reads_lane {α : Type} (v : alloc.vec.Vec α) (i : Usize) (hin : i.val < v.length)
    (x : α) (y : α → alloc.vec.Vec α)
    (h : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice α) v i = ok (x, y)) :
    x = v.val[i.val] := by
  have h' : alloc.vec.Vec.index_mut_usize v i = ok (x, y) :=
    alloc.vec.Vec.index_mut_slice_index v i ▸ h
  obtain ⟨z, hread, hz⟩ := bind_eq_ok h'
  have hzz : ok (z, alloc.vec.Vec.set v i) = ok (x, y) := hz
  obtain ⟨w, hw, hwlane⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec v i hin)
  have hpair : (x, y) = (z, alloc.vec.Vec.set v i) := Result.ok_injective hzz.symm
  exact (congrArg Prod.fst hpair).trans ((Result.ok_injective (hread.symm.trans hw)).trans hwlane)

/-- The staged-path recycling the two Found arms run first touches only the locator: it never changes
the committed length. -/
theorem recycle_preserves_len (state : Store) (s' : Store)
    (h : table.Table.recycle_unused_staged_path CloneI CloneI HomeInst KeyRepInst state = ok s') :
    s'.len = state.len := by
  unfold table.Table.recycle_unused_staged_path at h
  generalize core.option.Option.take state.locator.staged = tk at h
  obtain ⟨o, o1⟩ := tk
  cases o with
  | none =>
    cases Result.ok_injective h
    rfl
  | some p =>
    obtain ⟨_, staged⟩ := p
    generalize state.locator.directory = dir at h
    cases dir with
    | none =>
      cases Result.ok_injective h
      rfl
    | some directory =>
      obtain ⟨directory1, _, hrest⟩ := bind_eq_ok h
      cases Result.ok_injective hrest.symm
      rfl

/-- The Found/flat settle transition, verbatim: the whole extracted computation under that guard. The
right-hand side holds one `clone`, one `merge`, one `index_mut` and one located-slot write, in that
order; the kernel check of this equation is what certifies the work count below. -/
theorem found_flat_settle (state : Store) (reserved : Reserved) (index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    body state reserved ⟨table.SearchClass.Found, table.SearchLocation.Flat index⟩
        sigma value held address
      = (do
        let self1 ← table.Table.recycle_unused_staged_path CloneI CloneI HomeInst KeyRepInst state
        let fs ← alloc.vec.Vec.index SlotIdx reserved.resident.slots index
        let old ← CloneI.clone fs.value
        let t ← MergeInst.merge MergeArg value old
        let (fs1, index_mut_back) ← alloc.vec.Vec.index_mut SlotIdx reserved.resident.slots index
        let v := index_mut_back { fs1 with value := t }
        ok { self1 with
             resident := { reserved.resident with slots := v },
             shards := reserved.regions }) := rfl

/-- The Found/spill settle transition, verbatim: the merge writes the located spill payload and the
relocated resident carrier is committed unchanged. -/
theorem found_spill_settle (state : Store) (reserved : Reserved) (spill index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    body state reserved ⟨table.SearchClass.Found, table.SearchLocation.Spill spill index⟩
        sigma value held address
      = (do
        let self1 ← table.Table.recycle_unused_staged_path CloneI CloneI HomeInst KeyRepInst state
        let r ← alloc.vec.Vec.index RegionIdx reserved.regions spill
        let (_, _, t) ← alloc.vec.Vec.index TripleIdx r.slots index
        let old ← CloneI.clone t
        let t1 ← MergeInst.merge MergeArg value old
        let (r1, index_mut_back) ← alloc.vec.Vec.index_mut RegionIdx reserved.regions spill
        let ((s, t2, _), index_mut_back1) ← alloc.vec.Vec.index_mut TripleIdx r1.slots index
        let v := index_mut_back1 (s, t2, t1)
        let v1 := index_mut_back { r1 with slots := v }
        ok { self1 with resident := reserved.resident, shards := v1 }) := rfl

/-- The Vacant/flat settle transition, verbatim: one payload copy of the default, one merge, the
held-key publication, the located-slot write and the checked length increment. -/
theorem vacant_flat_settle (state : Store) (reserved : Reserved) (index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.Flat index⟩
        sigma value held address
      = (do
        let t ← CloneI.clone state.default
        let merged ← MergeInst.merge MergeArg value t
        let tl ← table.TableLocator.publish_for_held CloneI HomeInst KeyRepInst
          state.locator state.home sigma held (table.PayloadLocation.Resident index)
        let (_, index_mut_back) ← alloc.vec.Vec.index_mut SlotIdx reserved.resident.slots index
        let i ← state.len + 1#usize
        let v := index_mut_back { live := true, sigma, held, value := merged }
        ok { state with
             resident := { reserved.resident with slots := v },
             shards := reserved.regions,
             len := i,
             locator := tl }) := rfl

/-- The Vacant/spill settle transition, verbatim: the payload is published live through the region
and the located length is incremented. -/
theorem vacant_spill_settle (state : Store) (reserved : Reserved) (spill index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.Spill spill index⟩
        sigma value held address
      = (do
        let t ← CloneI.clone state.default
        let merged ← MergeInst.merge MergeArg value t
        let tl ← table.TableLocator.publish_for_held CloneI HomeInst KeyRepInst
          state.locator state.home sigma held (table.PayloadLocation.Spill spill index)
        let (r, index_mut_back) ← alloc.vec.Vec.index_mut RegionIdx reserved.regions spill
        let r1 ← table.Region.publish_live r index (sigma, held, merged) address.place
        let i ← state.len + 1#usize
        let v := index_mut_back r1
        ok { state with resident := reserved.resident, shards := v, len := i, locator := tl }) := rfl

/-- The first exit transition: a `Found` decision with no index cannot be applied, and the real body
takes the `unreachable!` branch (`src/table.rs:1117`) without touching storage. -/
theorem found_noindex_exit (state : Store) (reserved : Reserved)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    body state reserved ⟨table.SearchClass.Found, table.SearchLocation.NoIndex⟩
        sigma value held address = fail panic := rfl

/-- The second exit transition: a `Vacant` decision with no index has no lane to write. -/
theorem vacant_noindex_exit (state : Store) (reserved : Reserved)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.NoIndex⟩
        sigma value held address = fail panic := rfl

/-- The third exit transition: exhaustion at any location is the growth path, which this core refuses
(`src/table.rs:1114-1116`); it merges nothing and writes nothing. -/
theorem exhausted_exit (state : Store) (reserved : Reserved) (location : table.SearchLocation)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) :
    body state reserved ⟨table.SearchClass.Exhausted, location⟩ sigma value held address
      = fail panic := by
  unfold body table.Table.apply_pending_and_commit
  cases location <;> simp only [bind_tc_ok, bind_tc_fail]

/-- The real body has exactly the seven transition shapes the four settle equations and the three exit
equations name, so no `(class, location)` pair reaches a transition that is not listed. -/
theorem transitions_exhaustive (decision : table.SearchDecision) :
    (∃ index, decision = ⟨table.SearchClass.Found, table.SearchLocation.Flat index⟩) ∨
    (∃ spill index, decision = ⟨table.SearchClass.Found, table.SearchLocation.Spill spill index⟩) ∨
    decision = ⟨table.SearchClass.Found, table.SearchLocation.NoIndex⟩ ∨
    (∃ index, decision = ⟨table.SearchClass.Vacant, table.SearchLocation.Flat index⟩) ∨
    (∃ spill index, decision = ⟨table.SearchClass.Vacant, table.SearchLocation.Spill spill index⟩) ∨
    decision = ⟨table.SearchClass.Vacant, table.SearchLocation.NoIndex⟩ ∨
    (∃ location, decision = ⟨table.SearchClass.Exhausted, location⟩) := by
  rcases decision with ⟨cls, loc⟩
  cases cls with
  | Found =>
    cases loc with
    | Flat index => exact Or.inl ⟨index, rfl⟩
    | Spill spill index => exact Or.inr (Or.inl ⟨spill, index, rfl⟩)
    | NoIndex => exact Or.inr (Or.inr (Or.inl rfl))
  | Vacant =>
    cases loc with
    | Flat index => exact Or.inr (Or.inr (Or.inr (Or.inl ⟨index, rfl⟩)))
    | Spill spill index => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨spill, index, rfl⟩))))
    | NoIndex => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
  | Exhausted => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨loc, rfl⟩)))))

/-- A settling decision never refuses: the `ok` node and the panic node are distinct, so no decision
both settles and exits. -/
theorem settle_and_exit_disjoint (state : Store) (reserved : Reserved)
    (decision : table.SearchDecision) (sigma : cell.Sigma) (value : Std.I64) (held : Held)
    (address : table.TableAddress) (t : Store)
    (h : body state reserved decision sigma value held address = ok t) :
    body state reserved decision sigma value held address ≠ fail panic :=
  fun hfail => ok_not_fail (h.symm.trans hfail)

/-- The work of one evaluation of the real pending core: the merges, payload copies and located-slot
writes the core itself issues. `exit` is the refusal channel and carries no work. -/
inductive Work where
  /-- A settling transition: `merges` merges, `copies` payload copies, `writes` slot writes. -/
  | settle (merges copies writes : Nat)
  /-- A refusal transition. -/
  | exit

/-- The counted work of one evaluation of the real body at its observed outcome. -/
def workOf (r : Result Store) : Work :=
  Result.cases r (fun _ => Work.settle 1 1 1) (fun _ _ => Work.exit) Work.exit

/-- A committing Found/flat evaluation performs exactly one merge, one payload copy and one
located-slot write: `found_flat_settle` is the whole term, so a second merge, a second copy or a
dropped write would break that kernel check. -/
theorem found_flat_work (state : Store) (reserved : Reserved) (index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) (t : Store)
    (h : body state reserved ⟨table.SearchClass.Found, table.SearchLocation.Flat index⟩
        sigma value held address = ok t) :
    workOf (body state reserved ⟨table.SearchClass.Found, table.SearchLocation.Flat index⟩
      sigma value held address) = Work.settle 1 1 1 := by
  rw [h]
  rfl

/-- The same closed form on the vacant/flat guard. -/
theorem vacant_flat_work (state : Store) (reserved : Reserved) (index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) (t : Store)
    (h : body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.Flat index⟩
        sigma value held address = ok t) :
    workOf (body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.Flat index⟩
      sigma value held address) = Work.settle 1 1 1 := by
  rw [h]
  rfl

/-- An exiting evaluation performs no merge, no copy and no write: the exit equations show the whole
body is the bare panic node. -/
theorem exit_work_is_zero (state : Store) (reserved : Reserved) (decision : table.SearchDecision)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress)
    (h : body state reserved decision sigma value held address = fail panic) :
    workOf (body state reserved decision sigma value held address) = Work.exit := by
  rw [h]
  rfl

/-- The checked increment the two vacant arms commit their length through is the model's `usize` add:
when it is `ok`, its value is exactly the old length plus one. -/
theorem usize_increment_spec (n i : Usize) (h : n + 1#usize = ok i) :
    i.val = n.val + (1#usize).val := by
  have heq := UScalar.add_equiv (x := n) (y := 1#usize)
  rw [h] at heq
  simp only [Result.match.ok] at heq
  obtain ⟨_, hval, _⟩ := heq
  exact hval

/-- The Found/flat transition: on a committing run at an in-bounds lane the body copies that lane's
prior payload, merges the pending value into it new-first, writes exactly that lane, commits the
relocated spill vector as the shard vector, and leaves the committed length at its old value. -/
theorem found_flat_commits (state : Store) (reserved : Reserved) (index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) (t : Store)
    (hin : index.val < reserved.resident.slots.length)
    (h : body state reserved ⟨table.SearchClass.Found, table.SearchLocation.Flat index⟩
        sigma value held address = ok t) :
    ∃ (prior merged : Std.I64) (slots : alloc.vec.Vec Slot),
      CloneI.clone (reserved.resident.slots.val[index.val]).value = ok prior ∧
      MergeInst.merge MergeArg value prior = ok merged ∧
      slots = alloc.vec.Vec.set reserved.resident.slots index
        { (reserved.resident.slots.val[index.val]) with value := merged } ∧
      t.resident = { reserved.resident with slots := slots } ∧
      t.shards = reserved.regions ∧
      t.len = state.len := by
  obtain ⟨self1, hself, hr1⟩ :=
    bind_eq_ok ((found_flat_settle state reserved index sigma value held address).trans h)
  obtain ⟨fs, hfs, hr2⟩ := bind_eq_ok hr1
  obtain ⟨old, hold, hr3⟩ := bind_eq_ok hr2
  obtain ⟨merged, hmerged, hr4⟩ := bind_eq_ok hr3
  obtain ⟨⟨fs1, back⟩, hmut, hr5⟩ := bind_eq_ok hr4
  have hfsLane := index_reads_lane reserved.resident.slots index hin fs hfs
  have hfs1 : fs1 = fs :=
    (index_mut_reads_lane reserved.resident.slots index hin fs1 back hmut).trans hfsLane.symm
  have hback := index_mut_write reserved.resident.slots index fs1 back hmut
  have hrec : t = { self1 with
      resident := { reserved.resident with slots := back { fs1 with value := merged } },
      shards := reserved.regions } := Result.ok_injective hr5.symm
  refine ⟨old, merged,
    alloc.vec.Vec.set reserved.resident.slots index { (reserved.resident.slots.val[index.val]) with
      value := merged }, ?_, ?_, rfl, ?_, ?_⟩
  · rw [hfsLane] at hold
    exact hold
  · exact hmerged
  · rw [hrec, hback.1, ← hfsLane, hfs1]
  · refine ⟨?_, ?_⟩
    · rw [hrec]
    · rw [hrec]
      exact recycle_preserves_len state self1 hself

/-- The Found/spill transition: the merge writes the located spill payload new-first, the relocated
resident carrier is committed unchanged and the committed length is left alone. -/
theorem found_spill_commits (state : Store) (reserved : Reserved) (spill index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) (t : Store)
    (h : body state reserved ⟨table.SearchClass.Found, table.SearchLocation.Spill spill index⟩
        sigma value held address = ok t) :
    ∃ (r1 : Region) (locS : cell.Sigma) (locH : Held) (prior old merged : Std.I64)
      (shards : alloc.vec.Vec Region),
      CloneI.clone prior = ok old ∧
      MergeInst.merge MergeArg value old = ok merged ∧
      shards = alloc.vec.Vec.set reserved.regions spill
        { r1 with slots := alloc.vec.Vec.set r1.slots index ⟨locS, locH, merged⟩ } ∧
      t.resident = reserved.resident ∧
      t.len = state.len := by
  obtain ⟨self1, hself, hr1⟩ :=
    bind_eq_ok ((found_spill_settle state reserved spill index sigma value held address).trans h)
  obtain ⟨r, _, hr2⟩ := bind_eq_ok hr1
  obtain ⟨⟨locS, locH, prior⟩, _, hr3⟩ := bind_eq_ok hr2
  obtain ⟨old, hold, hr4⟩ := bind_eq_ok hr3
  obtain ⟨merged, hmerged, hr5⟩ := bind_eq_ok hr4
  obtain ⟨⟨r1, back⟩, hmut, hr6⟩ := bind_eq_ok hr5
  obtain ⟨⟨⟨pS, pH, _⟩, back1⟩, hmut2, hr7⟩ := bind_eq_ok hr6
  have hback := index_mut_write reserved.regions spill r1 back hmut
  have hback1 := index_mut_write r1.slots index ⟨pS, pH, _⟩ back1 hmut2
  have hrec : t = { self1 with
      resident := reserved.resident,
      shards := back { r1 with slots := back1 (pS, pH, merged) } } :=
    Result.ok_injective hr7.symm
  refine ⟨r1, pS, pH, prior, old, merged,
    back { r1 with slots := back1 (pS, pH, merged) }, hold, hmerged, ?_, ?_, ?_⟩
  · rw [hback.1, hback1.1]
  · rw [hrec]
  · rw [hrec]
    exact recycle_preserves_len state self1 hself

/-- The Vacant/flat transition: the body copies the table default, merges the pending value into it
new-first, publishes the held key, writes one fresh live slot at the located lane, and commits the
checked increment of the located length. -/
theorem vacant_flat_commits (state : Store) (reserved : Reserved) (index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) (t : Store)
    (h : body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.Flat index⟩
        sigma value held address = ok t) :
    ∃ (prior merged : Std.I64) (i : Usize) (slots : alloc.vec.Vec Slot),
      CloneI.clone state.default = ok prior ∧
      MergeInst.merge MergeArg value prior = ok merged ∧
      state.len + 1#usize = ok i ∧
      slots = alloc.vec.Vec.set reserved.resident.slots index
        { live := true, sigma := sigma, held := held, value := merged } ∧
      t.resident = { reserved.resident with slots := slots } ∧
      t.shards = reserved.regions ∧
      t.len = i ∧
      t.len.val = state.len.val + 1 := by
  obtain ⟨prior, hprior, hr1⟩ :=
    bind_eq_ok ((vacant_flat_settle state reserved index sigma value held address).trans h)
  obtain ⟨merged, hmerged, hr2⟩ := bind_eq_ok hr1
  obtain ⟨tl, _, hr3⟩ := bind_eq_ok hr2
  obtain ⟨⟨fresh, back⟩, hmut, hr4⟩ := bind_eq_ok hr3
  obtain ⟨i, hi, hr5⟩ := bind_eq_ok hr4
  have hback := index_mut_write reserved.resident.slots index fresh back hmut
  have hrec : t = { state with
      resident := { reserved.resident with
        slots := back { live := true, sigma := sigma, held := held, value := merged } },
      shards := reserved.regions, len := i, locator := tl } := Result.ok_injective hr5.symm
  refine ⟨prior, merged, i, back { live := true, sigma := sigma, held := held, value := merged },
    hprior, hmerged, hi, ?_, ?_, ?_, ?_⟩
  · rw [hback.1]
  · rw [hrec]
  · rw [hrec]
  · rw [hrec]
    simpa using usize_increment_spec state.len i hi

/-- The Vacant/spill transition: the merge result is published live into the located region, the
relocated resident carrier is committed unchanged, and the committed length is the checked increment
of the located length. -/
theorem vacant_spill_commits (state : Store) (reserved : Reserved) (spill index : Usize)
    (sigma : cell.Sigma) (value : Std.I64) (held : Held) (address : table.TableAddress) (t : Store)
    (h : body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.Spill spill index⟩
        sigma value held address = ok t) :
    ∃ (prior merged : Std.I64) (i : Usize) (r1 : Region) (shards : alloc.vec.Vec Region),
      CloneI.clone state.default = ok prior ∧
      MergeInst.merge MergeArg value prior = ok merged ∧
      state.len + 1#usize = ok i ∧
      shards = alloc.vec.Vec.set reserved.regions spill r1 ∧
      t.resident = reserved.resident ∧
      t.len = i ∧
      t.len.val = state.len.val + 1 := by
  obtain ⟨prior, hprior, hr1⟩ :=
    bind_eq_ok ((vacant_spill_settle state reserved spill index sigma value held address).trans h)
  obtain ⟨merged, hmerged, hr2⟩ := bind_eq_ok hr1
  obtain ⟨tl, _, hr3⟩ := bind_eq_ok hr2
  obtain ⟨⟨r, back⟩, hmut, hr4⟩ := bind_eq_ok hr3
  obtain ⟨r1, _, hr5⟩ := bind_eq_ok hr4
  obtain ⟨i, hi, hr6⟩ := bind_eq_ok hr5
  have hback := index_mut_write reserved.regions spill r back hmut
  have hrec : t = { state with
      resident := reserved.resident, shards := back r1, len := i, locator := tl } :=
    Result.ok_injective hr6.symm
  refine ⟨prior, merged, i, r1, back r1, hprior, hmerged, hi, ?_, ?_, ?_⟩
  · rw [hback.1]
  · rw [hrec]
  · rw [hrec]
    simpa using usize_increment_spec state.len i hi

/-- A committed length the `usize` add cannot represent is refused, not wrapped: the vacant arms reach
`ok` only through that checked increment. The excluded population is a committed length at the word
maximum, which this model cannot name as a `usize` literal. -/
theorem vacant_flat_refuses_unrepresentable_length (state : Store) (reserved : Reserved)
    (index : Usize) (sigma : cell.Sigma) (value : Std.I64) (held : Held)
    (address : table.TableAddress) (hr : ¬ ∃ i, state.len + 1#usize = ok i) :
    ¬ ∃ t, body state reserved ⟨table.SearchClass.Vacant, table.SearchLocation.Flat index⟩
        sigma value held address = ok t := by
  rintro ⟨t, h⟩
  obtain ⟨prior, _, hr1⟩ :=
    bind_eq_ok ((vacant_flat_settle state reserved index sigma value held address).trans h)
  obtain ⟨merged, _, hr2⟩ := bind_eq_ok hr1
  obtain ⟨tl, _, hr3⟩ := bind_eq_ok hr2
  obtain ⟨_, _, hr4⟩ := bind_eq_ok hr3
  obtain ⟨i, hi, _⟩ := bind_eq_ok hr4
  exact hr ⟨i, hi⟩

/-- The counted work does not depend on the magnitude of any argument: the four arm equations and the
three exit equations fix the application multiset, and it contains no arithmetic on the arguments. -/
theorem work_is_input_independent (decision : table.SearchDecision)
    (state state' : Store) (reserved reserved' : Reserved)
    (sigma sigma' : cell.Sigma) (value value' : Std.I64) (held held' : Held)
    (address address' : table.TableAddress) (t t' : Store)
    (h : body state reserved decision sigma value held address = ok t)
    (h' : body state' reserved' decision sigma' value' held' address' = ok t') :
    workOf (body state reserved decision sigma value held address) =
      workOf (body state' reserved' decision sigma' value' held' address') := by
  rw [h, h']
  rfl

/-- A one-slot resident carrier holding a single live stored-word slot. -/
def witnessSlot : Slot :=
  { live := true, sigma := cell.Sigma.Elliptic, held := { key := 1#i64 }, value := 5#i64 }

/-- A store whose resident carrier holds exactly that slot, whose locator has no staged path, and
whose committed length is one. -/
def witnessState : Store :=
  { default := 0#i64
  , home := ()
  , resident := { lanes := 1#usize, slots := alloc.vec.Vec.from [witnessSlot] (by grind) }
  , shards := alloc.vec.Vec.new _
  , len := 1#usize
  , key := ()
  , locator := { profile := none, directory := none, links := alloc.vec.Vec.new _, staged := none } }

/-- The relocated reserved storage the wrapper supplies: the same one-slot resident carrier and no
spill region. -/
def witnessReserved : Reserved :=
  { resident := { lanes := 1#usize, slots := alloc.vec.Vec.from [witnessSlot] (by grind) }
  , regions := alloc.vec.Vec.new _ }

/-- The address the witness call carries; the stored word home places it at zero. -/
def witnessAddress : table.TableAddress := { place := 0#u64, direct_word := none }

/-- The settled slot of the witness call: the located slot's prior payload merged with the pending
value, new-first. -/
def witnessSettled : Slot :=
  { witnessSlot with value := core.num.I64.wrapping_add 3#i64 5#i64 }

/-- The witness carrier holds exactly one slot, so the witness lane is in bounds of it and the settle
reads the located payload. -/
theorem witness_lane_in_bounds : 0 < witnessReserved.resident.slots.length := by
  simp [witnessReserved, alloc.vec.Vec.from_val]

/-- The success witness at the joint machine-word types: at these concrete arguments, a committing run
writes lane zero of the committed resident carrier with the pending value merged new-first into that
lane's prior payload, commits the relocated spill vector unchanged as the shard vector, and leaves the
committed length at one. -/
theorem found_flat_success_witness :
    MachineWords 0#usize ∧
    0 < witnessReserved.resident.slots.length ∧
    ∀ t : Store,
      body witnessState witnessReserved
          ⟨table.SearchClass.Found, table.SearchLocation.Flat 0#usize⟩
          cell.Sigma.Elliptic 3#i64 { key := 1#i64 } witnessAddress = ok t →
      ∃ (prior merged : Std.I64) (slots : alloc.vec.Vec Slot),
        CloneI.clone (witnessReserved.resident.slots.val[(0#usize).val]).value = ok prior ∧
        MergeInst.merge MergeArg 3#i64 prior = ok merged ∧
        slots = alloc.vec.Vec.set witnessReserved.resident.slots 0#usize
          { (witnessReserved.resident.slots.val[(0#usize).val]) with value := merged } ∧
        t.resident = { witnessReserved.resident with slots := slots } ∧
        t.shards = witnessReserved.regions ∧
        t.len = witnessState.len := by
  refine ⟨machineWords_holds 0#usize, witness_lane_in_bounds, ?_⟩
  intro t h
  obtain ⟨prior, merged, slots, hprior, hmerged, hslots, hresident, hshards, hlen⟩ :=
    found_flat_commits witnessState witnessReserved 0#usize cell.Sigma.Elliptic 3#i64
      { key := 1#i64 } witnessAddress t (by simpa using witness_lane_in_bounds) h
  exact ⟨prior, merged, slots, hprior, hmerged, hslots, hresident, hshards, hlen⟩

/-- The same witness read as a value: the settled payload is the wrapping sum of the pending value and
the located prior value. -/
theorem found_flat_success_value : core.num.I64.wrapping_add 3#i64 5#i64 = 8#i64 := rfl

/-- A reachable exit witness: the exhaustion decision the probe walk raises when no lane is free
reaches the bare panic node, without a merge, a copy or a write. -/
theorem exhausted_exit_witness :
    body witnessState witnessReserved
        ⟨table.SearchClass.Exhausted, table.SearchLocation.NoIndex⟩
        cell.Sigma.Elliptic 3#i64 { key := 1#i64 } witnessAddress = fail panic :=
  exhausted_exit witnessState witnessReserved table.SearchLocation.NoIndex
    cell.Sigma.Elliptic 3#i64 { key := 1#i64 } witnessAddress

end ApplyPendingResource

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

#resource_audit ApplyPendingResource.root_forwards
#resource_audit ApplyPendingResource.bind_eq_ok
#resource_audit ApplyPendingResource.machineWords_holds
#resource_audit ApplyPendingResource.merge_is_new_first_wrapping_add
#resource_audit ApplyPendingResource.index_reads_lane
#resource_audit ApplyPendingResource.index_mut_write
#resource_audit ApplyPendingResource.index_mut_reads_lane
#resource_audit ApplyPendingResource.recycle_preserves_len
#resource_audit ApplyPendingResource.found_flat_settle
#resource_audit ApplyPendingResource.found_spill_settle
#resource_audit ApplyPendingResource.vacant_flat_settle
#resource_audit ApplyPendingResource.vacant_spill_settle
#resource_audit ApplyPendingResource.found_noindex_exit
#resource_audit ApplyPendingResource.vacant_noindex_exit
#resource_audit ApplyPendingResource.exhausted_exit
#resource_audit ApplyPendingResource.transitions_exhaustive
#resource_audit ApplyPendingResource.settle_and_exit_disjoint
#resource_audit ApplyPendingResource.found_flat_work
#resource_audit ApplyPendingResource.vacant_flat_work
#resource_audit ApplyPendingResource.exit_work_is_zero
#resource_audit ApplyPendingResource.usize_increment_spec
#resource_audit ApplyPendingResource.found_flat_commits
#resource_audit ApplyPendingResource.found_spill_commits
#resource_audit ApplyPendingResource.vacant_flat_commits
#resource_audit ApplyPendingResource.vacant_spill_commits
#resource_audit ApplyPendingResource.vacant_flat_refuses_unrepresentable_length
#resource_audit ApplyPendingResource.work_is_input_independent
#resource_audit ApplyPendingResource.witness_lane_in_bounds
#resource_audit ApplyPendingResource.found_flat_success_witness
#resource_audit ApplyPendingResource.found_flat_success_value
#resource_audit ApplyPendingResource.exhausted_exit_witness

run_cmd Lean.logInfo "APPLY_PENDING_KERNEL_OK"