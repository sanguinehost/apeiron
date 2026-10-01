import ScheduleSource
import Aeneas

/-
SCOPE: the actual extracted body reached by the extraction root `word_schedule_is_complete`, namely
  the real `Schedule::lpt` then the real `Schedule::is_complete` of `src/table/partition.rs`. For
  `is_complete` this file states, for each of the three loops the source contains, the guarded
  transition with the exact mark/flag/position effect, the work measure (a strictly decreasing
  `Nat` rank, one unit per guarded iteration), the two exit branches, and reachable witnesses at
  concrete bins, mark arrays and worker positions. It also pins the root's own decomposition
  (`root_forwards`, `is_complete_decomposes`), which is what the work count is read from:
  `bins.len()` worker steps, one element scan per worker step, then `256` coverage steps.
  Files: proof/resource/ScheduleSource.lean (Aeneas output), ScheduleResource.lean (this file) and
  Schedule_gate.py (lane-2 extraction/mutation gate). No other file is touched.
REGISTER: a local scalar cost-model obligation on the extracted `is_complete` scan: transitions,
  work measure under named premises, exit branches, reachable witnesses. Work is counted in scan
  steps, mark-array reads, mark-array writes and `Usize` comparisons, under the named premise that
  the extracted body is the one in ScheduleSource.lean; it is never a cycle count, never a
  wall-clock and never a GPU number. Excluded: the `lpt` selection sort and worker-bin fill (the
  four `lpt` loops, reached only through `root_forwards`), allocator behaviour, the `Vec`
  index/push adapter, and the canonical-corpus correspondence.
CORPUS: src/table/partition.rs, `Schedule::is_complete` 603:4-629:5 with its three scan loops at
  607:8-620:9, 610:12-618:13 and 622:8-627:9; the extraction root is
  proof/extract/lib.rs:481-486 (`word_schedule_is_complete`), and the same `partition` bodies are
  already extracted at proof/ApeironWord.lean:18118-18304.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: python3 proof/resource/Schedule_gate.py --controls, cwd repository root, on
  /tmp/opencode/apeiron-lane-2.lock, MemoryMax=12G, MemorySwapMax=0, LEAN_NUM_THREADS=1,
  540 s whole-invocation budget.
GRADE: `proved` for every theorem below if and only if that gate exits 0 with
  `SCHEDULE_KERNEL_OK`. The extracted body comes from the real source by `#[path]` with only a
  typed forwarding wrapper added; the shared `proof/extract/lib.rs` unit is not compiled. Nothing
  here is measured.
OPEN: the loop-aggregate spec of the element scan (that a pass reports `false` exactly when a bin
  repeats a slot or carries an index outside `0..255`, and that a bin of distinct in-range slots is
  accepted) and the matching aggregate spec of the worker loop. The exact residual is recorded in
  the handoff for this slice: it needs the four-clause `scanInv` (coverage, pairwise distinctness,
  mark monotonicity, marks-only-from-read-positions) inside `loop.spec_decr_nat`, and the
  `Bool`-valued `decide (256 <= Usize.cMax Usize)` in the model's `OfNat (UScalar Usize) 256`
  instance blocks the `simp`/`show` steps that proof needs. Nothing above is weakened for it: the
  statements proved here are the transition, work, exit and witness obligations only.
-/

namespace ScheduleResource
open Aeneas Aeneas.Std Result

/-! ## Platform facts -/

theorem shift256 : (1#usize <<< 8#u32) = ok (256#usize) := by
  obtain ⟨w, hw, hwv⟩ := WP.spec_imp_exists
    (Aeneas.Std.Usize.ShiftLeft_spec 1#usize 8#u32 (by
      rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num))
  have hwEq : w = 256#usize := by
    apply UScalar.eq_of_val_eq
    rcases System.Platform.numBits_eq with h | h <;>
      norm_num [Usize.size, Usize.numBits, h, ShiftLeft.shiftLeft, Nat.shiftLeft_eq] at hwv ⊢ <;>
      exact hwv.1
  calc
    (1#usize <<< 8#u32) = ok w := hw
    _ = ok (256#usize) := by rw [hwEq]

theorem shards_eq : schedule_source.table.partition.SHARDS = ok (256#usize) := by
  simp only [schedule_source.table.partition.SHARDS,
    schedule_source.table.partition.SHARD_BITS]
  exact shift256

theorem shards_value : schedule_source.table.partition.SHARDS ⦃ n => n.val = 256 ⦄ := by
  rw [shards_eq]
  exact (WP.spec_ok (p := fun n : Usize => n.val = 256) (256#usize)).mpr (by simp)

theorem usize_max_u32 (h32 : System.Platform.numBits = 32) : Usize.max = U32.max := by
  rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq, h32, U32.max_eq]
  norm_num

theorem usize_max_u64 (h64 : System.Platform.numBits = 64) : Usize.max = U64.max := by
  rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq, h64, U64.max_eq]
  norm_num

theorem usize_ge_256 : 256 ≤ Usize.max := by
  rcases System.Platform.numBits_eq with h32 | h64
  · rw [usize_max_u32 h32, U32.max_eq]
    norm_num
  · rw [usize_max_u64 h64, U64.max_eq]
    norm_num

theorem lt256 (s : Usize) : s < (256#usize : Usize) ↔ s.val < 256 := by simp

theorem ge256 (s : Usize) : s >= (256#usize : Usize) ↔ 256 ≤ s.val := by simp

theorem usize_one_val : (1#usize : Usize).val = 1 := by simp
theorem usize_zero_val : (0#usize : Usize).val = 0 := by simp

/-! ## Accessors

`slot`, `bitem`, `wbin` and `batom` are total read accessors for the mark array and for the bin
lists, so each predicate below is stated at concrete `Nat`/`Usize` indices and needs no dependent
proof argument. -/

/-- The mark a `seen` array carries for home-shard slot `i`. -/
def slot (seen : Array Bool 256#usize) (i : Usize) : Bool := seen.val[i.val]!

/-- The element a bin carries at read position `i`. -/
def bitem (bin : alloc.vec.Vec Usize) (i : Nat) : Usize := bin.val[i]!

/-- The bin a schedule carries at worker position `w`. -/
def wbin (bins : alloc.vec.Vec (alloc.vec.Vec Usize)) (w : Nat) : alloc.vec.Vec Usize :=
  bins.val[w]!

/-- The element a schedule's `w`-th bin carries at read position `i`. -/
def batom (bins : alloc.vec.Vec (alloc.vec.Vec Usize)) (w i : Nat) : Usize :=
  (bins.val[w]!)[i]!

theorem slot_set (a : Array Bool 256#usize) (s j : Usize) (hj : j.val < 256) :
    slot (a.set s true) j = if j.val = s.val then true else slot a j := by
  unfold slot
  rw [Array.set_val_eq]
  by_cases h : j.val = s.val
  · rw [if_pos h, List.set_getElem!_eq (l := a.val) (i := s.val) (i' := j.val) (x := true)
        ⟨by simpa [Array.length_eq] using hj, h.symm⟩]
  · rw [if_neg h, List.set_getElem!_ne (l := a.val) (i := s.val) (j := j.val) (x := true)
        (by simp [h])]

theorem slot_index (seen : Array Bool 256#usize) (s : Usize) (h : s.val < 256) (b : Bool)
    (hb : seen.index_usize s = ok b) : b = slot seen s := by
  obtain ⟨x, hx, hxs⟩ := WP.spec_imp_exists (Array.index_usize_spec seen s (by simpa using h))
  have hxb : b = x := Result.ok_injective (by rw [hb] at hx; exact hx)
  have h2 : s.val < seen.val.length := by simpa [Array.length_eq] using h
  calc b = x := hxb
    _ = seen.val[s.val] := hxs
    _ = seen.val[s.val]! := by
        simp [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem h2]

theorem vec_index (bin : alloc.vec.Vec Usize) («at» : Usize) (h : «at».val < bin.length)
    (s : Usize) (hb : alloc.vec.Vec.index_usize bin «at» = ok s) : s = bitem bin «at».val := by
  obtain ⟨x, hx, hxs⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec bin «at» h)
  have hxb : s = x := Result.ok_injective (by rw [hb] at hx; exact hx)
  calc s = x := hxb
    _ = bin.val[«at».val] := hxs
    _ = bin.val[«at».val]! := by
        simp [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem h]

/-- The read position the real scan guard `«at» < bin.len()` tests, in `Nat` form. -/
theorem vec_len_lt {α : Type} («at» : Usize) (bin : alloc.vec.Vec α) :
    «at» < alloc.vec.Vec.len bin ↔ «at».val < bin.length := by
  rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]

theorem vec_len_ge {α : Type} («at» : Usize) (bin : alloc.vec.Vec α) :
    alloc.vec.Vec.len bin ≤ «at» ↔ bin.length ≤ «at».val := by
  rw [UScalar.le_equiv, alloc.vec.Vec.len_val]

/-- **The work measure of the element scan**: the number of read positions the bin still has.
    One guarded iteration spends exactly one unit. -/
def scanRank (bin : alloc.vec.Vec Usize) («at» : Usize) : Nat := bin.length - «at».val

/-- **The work measure of the coverage loop**: the number of home-shard slots still to be read.
    One guarded iteration spends exactly one unit. -/
def coverRank (shard : Usize) : Nat := 256 - shard.val

/-! ## The element scan of `is_complete`: the guarded transition -/

/-- **Guarded transition of the real element scan.** Under the guard `«at» < bin.len()` and with the
    scan's own flag `true`, the extracted body reads the element at position `«at»`, compares it
    against the home-shard bound `SHARDS`, reads the mark that element carries, and advances the
    read position by exactly one. It spends exactly one unit of `scanRank`, and it writes a mark
    slot only when the element read is a real shard slot that was not yet marked. -/
theorem scan_step (bin : alloc.vec.Vec Usize) (seen : Array Bool 256#usize) («at» : Usize)
    (hat : «at».val < bin.length) :
    ∃ w : Array Bool 256#usize × Bool × Usize,
      schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body bin seen true «at»
        = ok (ControlFlow.cont w) ∧
      w.2.2.val = «at».val + 1 ∧
      scanRank bin w.2.2 + 1 = scanRank bin «at» ∧
      ((256 ≤ (bitem bin «at».val).val ∨ slot seen (bitem bin «at».val) = true) →
        w.2.1 = false ∧ w.1 = seen) ∧
      (w.2.1 = true → (bitem bin «at».val).val < 256 ∧
        slot seen (bitem bin «at».val) = false ∧
        w.1 = seen.set (bitem bin «at».val) true) := by
  have hlenmax : bin.length ≤ Usize.max := alloc.vec.Vec.property bin
  have h3 : «at».val < Usize.max := lt_of_lt_of_le hat hlenmax
  have h1 : (1#usize : Usize).val = 1 := usize_one_val
  have hadd : «at».val + (1#usize).val ≤ Usize.max := by
    rw [h1]
    exact Nat.succ_le_of_lt h3
  obtain ⟨«at2», haddrun, hat2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := «at») (y := 1#usize) hadd)
  have hat2v' : «at2».val = «at».val + 1 := by simpa using hat2v
  obtain ⟨i1, hi1run, hi1v⟩ := WP.spec_imp_exists shards_value
  have hgei (x : Usize) : x >= i1 ↔ 256 ≤ x.val := by
    show i1.val ≤ x.val ↔ _
    rw [hi1v]
  obtain ⟨s, hread, hsv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_usize_spec bin «at» hat)
  have hsv' : s = bitem bin «at».val := vec_index bin «at» hat s hread
  obtain ⟨w, hw, hpos, hrank, hlow, hmark⟩ :
      ∃ w : Array Bool 256#usize × Bool × Usize,
        schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body bin seen true «at»
          = ok (ControlFlow.cont w) ∧
        w.2.2.val = «at».val + 1 ∧
        scanRank bin w.2.2 + 1 = scanRank bin «at» ∧
        ((256 ≤ (bitem bin «at».val).val ∨ slot seen (bitem bin «at».val) = true) →
          w.2.1 = false ∧ w.1 = seen) ∧
        (w.2.1 = true → (bitem bin «at».val).val < 256 ∧
          slot seen (bitem bin «at».val) = false ∧
          w.1 = seen.set (bitem bin «at».val) true) := by
    unfold schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body
    rw [if_pos ((vec_len_lt «at» bin).2 hat), if_pos (rfl : (true : Bool) = true)]
    rw [alloc.vec.Vec.index_slice_index, hread, hi1run]
    simp only [bind_tc_ok]
    by_cases hge : 256 ≤ s.val
    · have hge' : s >= i1 := (hgei s).2 hge
      simp only [hge', reduceIte, haddrun, bind_tc_ok]
      refine ⟨⟨seen, false, «at2»⟩, rfl, hat2v', ?_, fun _ => ⟨rfl, rfl⟩, fun hc => by
        simp at hc⟩
      simp only [scanRank]
      rw [hat2v']
      omega
    · have hlt' : s.val < 256 := by omega
      have hge' : ¬ (s >= i1) := fun hv => hge ((hgei s).1 hv)
      simp only [hge', reduceIte]
      obtain ⟨b, hbread, hb⟩ := WP.spec_imp_exists
        (Array.index_usize_spec seen s (by simpa using hlt'))
      have hb' : b = slot seen s := slot_index seen s hlt' b hbread
      simp only [hbread, bind_tc_ok]
      rw [hb']
      cases hseen : slot seen s with
      | true =>
          simp only [reduceIte, haddrun, bind_tc_ok]
          refine ⟨⟨seen, false, «at2»⟩, rfl, hat2v', ?_, fun _ => ⟨rfl, rfl⟩, fun hc => by
            simp at hc⟩
          simp only [scanRank]
          rw [hat2v']
          omega
      | false =>
          obtain ⟨a, haupd, ha⟩ := WP.spec_imp_exists
            (Array.update_spec seen s true (by simpa using hlt'))
          rw [haupd]
          simp only [bind_tc_ok]
          rw [ha]
          simp only [if_neg (by decide : ¬ ((false : Bool) = true)), haddrun, bind_tc_ok]
          refine ⟨⟨seen.set s true, true, «at2»⟩, rfl, hat2v', ?_, ?_, ?_⟩
          · simp only [scanRank]
            rw [hat2v']
            omega
          · intro hf
            rcases hf with hf | hf
            · rw [← hsv'] at hf
              exact absurd hf (Nat.not_le_of_lt hlt')
            · rw [← hsv'] at hf
              rw [hf] at hseen
              exact absurd hseen (by simp)
          · intro _
            refine ⟨?_, ?_, ?_⟩
            · rw [← hsv']
              exact hlt'
            · rw [← hsv']
              exact hseen
            · exact congrArg (fun x : Usize => seen.set x true) hsv'
  exact ⟨w, hw, hpos, hrank, hlow, hmark⟩

/-- **Exit branch of the real element scan, position exhausted.** When the guard
    `«at» < bin.len()` is false the body performs no read, no comparison and no mark write: it
    returns the mark array and the flag unchanged, and the scan's rank is zero. -/
theorem scan_exit_pos (bin : alloc.vec.Vec Usize) (seen : Array Bool 256#usize) («at» : Usize)
    (h : bin.length ≤ «at».val) :
    schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body bin seen true «at»
      = ok (ControlFlow.done (seen, true)) ∧ scanRank bin «at» = 0 := by
  have hlt : alloc.vec.Vec.len bin ≤ «at» := (vec_len_ge «at» bin).2 h
  refine ⟨?_, ?_⟩
  · unfold schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body
    rw [if_neg (not_lt_of_ge hlt)]
  · simp only [scanRank]
    omega

/-- **Exit branch of the real element scan, flag already down.** Once a repeat or an out-of-range
    element has been found, the flag is `false` and every further call of the body returns
    immediately without reading the bin. -/
theorem scan_exit_flag (bin : alloc.vec.Vec Usize) (seen : Array Bool 256#usize) (c : Bool)
    («at» : Usize) (h : c = false) :
    schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body bin seen c «at»
      = ok (ControlFlow.done (seen, c)) := by
  unfold schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body
  rw [h]
  have hn : ¬ (false = true) := by decide
  by_cases hp : «at» < alloc.vec.Vec.len bin
  · rw [if_pos hp, if_neg hn]
  · rw [if_neg hp]

/-- The scan's rank is strictly decreasing, by exactly one unit per guarded iteration. -/
theorem scan_rank_decreases (bin : alloc.vec.Vec Usize) (seen : Array Bool 256#usize)
    («at» : Usize) (hat : «at».val < bin.length) :
    ∃ w : Array Bool 256#usize × Bool × Usize,
      schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body bin seen true «at»
        = ok (ControlFlow.cont w) ∧
      «at».val < bin.length ∧ scanRank bin w.2.2 + 1 = scanRank bin «at» := by
  obtain ⟨w, hw, _, hrank, _, _⟩ := scan_step bin seen «at» hat
  exact ⟨w, hw, hat, hrank⟩

/-! ## The coverage loop of `is_complete` -/

/-- **Guarded transition of the real coverage loop.** Under the guard `shard < SHARDS` and with the
    flag still up, the body reads the mark of slot `shard` and adopts it as the new flag, advancing
    the slot by exactly one. The transition spends exactly one unit of `coverRank` and performs no
    mark write. -/
theorem cover_step (seen : Array Bool 256#usize) (shard : Usize) (hsh : shard.val < 256) :
    ∃ c' : Bool, ∃ shard' : Usize,
      schedule_source.table.partition.Schedule.is_complete_loop1.body seen true shard
        = ok (ControlFlow.cont (c', shard')) ∧
      shard'.val = shard.val + 1 ∧ c' = slot seen shard ∧
      coverRank shard' + 1 = coverRank shard := by
  have hmax : shard.val ≤ Usize.max := by
    have hb := shard.hmax
    rcases System.Platform.numBits_eq with h | h
    · rw [usize_max_u32 h, U32.max_eq]
      norm_num [UScalarTy.Usize_numBits_eq, h] at hb
      omega
    · rw [usize_max_u64 h, U64.max_eq]
      norm_num [UScalarTy.Usize_numBits_eq, h] at hb
      omega
  have h1 : (1#usize : Usize).val = 1 := usize_one_val
  have hadd : shard.val + (1#usize).val ≤ Usize.max := by
    rw [h1]
    have h2 : shard.val + 1 ≤ 256 := by omega
    have h3 : 256 ≤ Usize.max := usize_ge_256
    omega
  obtain ⟨shard2, haddrun, hshard2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := shard) (y := 1#usize) hadd)
  have hshard2v' : shard2.val = shard.val + 1 := by simpa using hshard2v
  obtain ⟨i1, hi1run, hi1v⟩ := WP.spec_imp_exists shards_value
  have hslt : shard < i1 ↔ shard.val < 256 := by
    show shard.val < i1.val ↔ _
    rw [hi1v]
  obtain ⟨b, hbread, hb⟩ := WP.spec_imp_exists
    (Array.index_usize_spec seen shard (by simpa using hsh))
  have hb' : b = slot seen shard := slot_index seen shard hsh b hbread
  have hmain :
      ∃ c' : Bool, ∃ shard' : Usize,
        schedule_source.table.partition.Schedule.is_complete_loop1.body seen true shard
          = ok (ControlFlow.cont (c', shard')) ∧
        shard'.val = shard.val + 1 ∧ c' = slot seen shard ∧
        coverRank shard' + 1 = coverRank shard := by
    unfold schedule_source.table.partition.Schedule.is_complete_loop1.body
    rw [hi1run]
    simp only [bind_tc_ok]
    rw [if_pos ((hslt).2 hsh)]
    simp only [reduceIte]
    rw [hbread]
    simp only [bind_tc_ok]
    rw [hb']
    cases hmark : slot seen shard with
    | true =>
        simp only [reduceIte, haddrun, bind_tc_ok]
        refine ⟨true, shard2, rfl, hshard2v', rfl, ?_⟩
        simp only [coverRank, hshard2v']
        omega
    | false =>
        simp only [if_neg (by decide : ¬ ((false : Bool) = true)), haddrun, bind_tc_ok]
        refine ⟨false, shard2, rfl, hshard2v', rfl, ?_⟩
        simp only [coverRank, hshard2v']
        omega
  obtain ⟨c', shard', hw, hpos, hc', hrank⟩ := hmain
  exact ⟨c', shard', hw, hpos, hc', hrank⟩

/-- **Exit branch of the real coverage loop, slots exhausted.** When the guard `shard < SHARDS` is
    false the body performs no read and returns the flag unchanged, and `coverRank` is zero. -/
theorem cover_exit_pos (seen : Array Bool 256#usize) (shard : Usize) (h : 256 ≤ shard.val) :
    schedule_source.table.partition.Schedule.is_complete_loop1.body seen true shard
      = ok (ControlFlow.done true) ∧ coverRank shard = 0 := by
  obtain ⟨i1, hi1run, hi1v⟩ := WP.spec_imp_exists shards_value
  have hslt : shard < i1 ↔ shard.val < 256 := by
    show shard.val < i1.val ↔ _
    rw [hi1v]
  have hnot : ¬ (shard < i1) := fun hc => (Nat.not_le_of_lt ((hslt).1 hc)) h
  refine ⟨?_, ?_⟩
  · unfold schedule_source.table.partition.Schedule.is_complete_loop1.body
    rw [hi1run]
    simp only [bind_tc_ok]
    rw [if_neg hnot]
  · simp only [coverRank]
    omega

/-- **Exit branch of the real coverage loop, flag already down.** Once a slot was found unmarked the
    loop refuses and every later call of the body returns `done false` without reading. -/
theorem cover_exit_flag (seen : Array Bool 256#usize) (c : Bool) (shard : Usize) (h : c = false) :
    schedule_source.table.partition.Schedule.is_complete_loop1.body seen c shard
      = ok (ControlFlow.done false) := by
  unfold schedule_source.table.partition.Schedule.is_complete_loop1.body
  obtain ⟨i1, hi1run, hi1v⟩ := WP.spec_imp_exists shards_value
  rw [hi1run]
  simp only [bind_tc_ok]
  have hn : ¬ (c = true) := by rw [h]; decide
  by_cases hslt : shard < i1
  · rw [if_pos hslt, if_neg hn]
  · rw [if_neg hslt, h]

/-! ## The worker loop of `is_complete`: the guarded transition -/

/-- **The work measure of the worker loop**: the number of bins the schedule still has to be
    scanned element by element. One guarded iteration spends exactly one unit and runs exactly
    one element scan over the bin at the current worker position. -/
def workerRank (bins : alloc.vec.Vec (alloc.vec.Vec Usize)) (worker : Usize) : Nat :=
  bins.length - worker.val

/-- **Guarded transition of the real worker loop.** Under the guard `worker < bins.len()` and with
    the flag still up, the body reads the bin at that worker position, runs the real element scan
    `is_complete_loop0_loop0` over it from read position `0` with the flag up, adopts that scan's
    mark array and flag, and advances the worker position by exactly one. It spends exactly one
    unit of `workerRank` and performs exactly one element scan. The element scan's own result is
    the premise `hscan`, which is the very call the extracted body makes. -/
theorem worker_step (bins : alloc.vec.Vec (alloc.vec.Vec Usize)) (seen : Array Bool 256#usize)
    (worker : Usize) (hw : worker.val < bins.length)
    (r : Array Bool 256#usize × Bool)
    (hscan : schedule_source.table.partition.Schedule.is_complete_loop0_loop0 seen true
        (wbin bins worker.val) 0#usize = ok r) :
    ∃ worker' : Usize,
      schedule_source.table.partition.Schedule.is_complete_loop0.body { bins := bins } seen true
          worker
        = ok (ControlFlow.cont (r.1, r.2, worker')) ∧
      worker'.val = worker.val + 1 ∧
      workerRank bins worker' + 1 = workerRank bins worker := by
  have hlenmax : bins.length ≤ Usize.max := alloc.vec.Vec.property bins
  have hlt : worker.val < Usize.max := lt_of_lt_of_le hw hlenmax
  have h1 : (1#usize : Usize).val = 1 := usize_one_val
  have hadd : worker.val + (1#usize).val ≤ Usize.max := by
    rw [h1]
    exact Nat.succ_le_of_lt hlt
  obtain ⟨worker2, haddrun, hworker2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := worker) (y := 1#usize) hadd)
  have hworker2v' : worker2.val = worker.val + 1 := by simpa using hworker2v
  obtain ⟨bin, hbread, hbv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_usize_spec bins worker hw)
  have hbv' : bin = wbin bins worker.val := by
    calc bin = bins.val[worker.val] := hbv
      _ = bins.val[worker.val]! := by
          simp [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hw]
      _ = wbin bins worker.val := rfl
  have hbody :
      schedule_source.table.partition.Schedule.is_complete_loop0.body { bins := bins } seen true
          worker
        = ok (ControlFlow.cont (r.1, r.2, worker2)) := by
    unfold schedule_source.table.partition.Schedule.is_complete_loop0.body
    rw [if_pos ((vec_len_lt worker bins).2 hw)]
    simp only [reduceIte]
    rw [alloc.vec.Vec.index_slice_index, hbread]
    simp only [bind_tc_ok]
    rw [hbv', hscan, haddrun]
    simp only [bind_tc_ok]
    rfl
  exact ⟨worker2, hbody, hworker2v', by
    simp only [workerRank, hworker2v']
    omega⟩

/-- **Exit branch of the real worker loop, bins exhausted.** When the guard `worker < bins.len()` is
    false the body performs no bin read and no element scan: it returns the mark array and the flag
    unchanged, and `workerRank` is zero. -/
theorem worker_exit_pos (bins : alloc.vec.Vec (alloc.vec.Vec Usize))
    (seen : Array Bool 256#usize) (worker : Usize) (h : bins.length ≤ worker.val) :
    schedule_source.table.partition.Schedule.is_complete_loop0.body { bins := bins } seen true
        worker
      = ok (ControlFlow.done (seen, true)) ∧ workerRank bins worker = 0 := by
  have hlt : alloc.vec.Vec.len bins ≤ worker := (vec_len_ge worker bins).2 h
  refine ⟨?_, ?_⟩
  · unfold schedule_source.table.partition.Schedule.is_complete_loop0.body
    rw [if_neg (not_lt_of_ge hlt)]
  · simp only [workerRank]
    omega

/-- **Exit branch of the real worker loop, flag already down.** Once the element scan has refused,
    the flag is `false` and every later call of the body returns immediately without reading a bin
    and without running an element scan. -/
theorem worker_exit_flag (bins : alloc.vec.Vec (alloc.vec.Vec Usize))
    (seen : Array Bool 256#usize) (c : Bool) (worker : Usize) (h : c = false) :
    schedule_source.table.partition.Schedule.is_complete_loop0.body { bins := bins } seen c worker
      = ok (ControlFlow.done (seen, c)) := by
  unfold schedule_source.table.partition.Schedule.is_complete_loop0.body
  rw [h]
  have hn : ¬ (false = true) := by decide
  by_cases hp : worker < alloc.vec.Vec.len bins
  · rw [if_pos hp, if_neg hn]
  · rw [if_neg hp]

/-! ## The extracted root and the initial state of the two scans -/

/-- The extraction root adds no work of its own: on the value path it calls the real
    `Schedule::lpt` and then the real `Schedule::is_complete` on the schedule `lpt` returned. -/
theorem root_forwards (loads : Array Usize 256#usize) (workers : Usize) (b : Bool)
    (hb : schedule_source.word_schedule_is_complete loads workers = ok b) :
    ∃ s : schedule_source.table.partition.Schedule,
      schedule_source.table.partition.Schedule.lpt loads workers = ok s ∧
      schedule_source.table.partition.Schedule.is_complete s = ok b := by
  unfold schedule_source.word_schedule_is_complete at hb
  cases hlpt : schedule_source.table.partition.Schedule.lpt loads workers with
  | ret s =>
      rw [hlpt] at hb
      simp only [bind_tc_ok] at hb
      exact ⟨s, rfl, hb⟩
  | vis e f =>
      rw [hlpt] at hb
      exact absurd hb (by simp)
  | div =>
      rw [hlpt] at hb
      exact absurd hb (by simp)

/-- The real `is_complete` is the worker loop over every bin followed by the coverage loop over all
    `256` home-shard slots, entered with the all-unmarked mark array and the flag up. This is the
    decomposition the work measure is read from: `bins.len()` worker steps, one element scan per
    worker step, then `256` coverage steps. -/
theorem is_complete_decomposes (self : schedule_source.table.partition.Schedule) (b : Bool)
    (hb : schedule_source.table.partition.Schedule.is_complete self = ok b) :
      ∃ p : Array Bool 256#usize × Bool,
        schedule_source.table.partition.Schedule.is_complete_loop0 self
            (Array.repeat 256#usize false) true 0#usize
          = ok p ∧
        schedule_source.table.partition.Schedule.is_complete_loop1 p.1 p.2 0#usize = ok b := by
  unfold schedule_source.table.partition.Schedule.is_complete at hb
  dsimp only at hb
  cases hw : schedule_source.table.partition.Schedule.is_complete_loop0 self
      (Array.repeat 256#usize false) true 0#usize with
  | ret p =>
      rw [hw] at hb
      simp only [bind_tc_ok] at hb
      exact ⟨p, rfl, hb⟩
  | vis e f =>
      rw [hw] at hb
      exact absurd hb (by simp)
  | div =>
      rw [hw] at hb
      exact absurd hb (by simp)

/-! ## Reachable witnesses -/

/-- The one-slot bin `[0]`, the smallest bin the real element scan can read. -/
theorem one_le_usize_max : 1 ≤ Usize.max := by have := usize_ge_256; omega

def oneSlot : alloc.vec.Vec Usize :=
  alloc.vec.Vec.from [0#usize] (by
    rw [List.length_cons, List.length_nil]
    exact one_le_usize_max)

theorem oneSlot_length : oneSlot.length = 1 := by
  show oneSlot.val.length = 1
  rw [oneSlot, alloc.vec.Vec.from_val, List.length_cons, List.length_nil]

/-- The one-bin schedule `[ [0] ]`, the smallest schedule the real worker loop can scan. -/
def oneBin : alloc.vec.Vec (alloc.vec.Vec Usize) :=
  alloc.vec.Vec.from [oneSlot] (by simp; exact one_le_usize_max)

theorem oneBin_length : oneBin.length = 1 := by
  show oneBin.val.length = 1
  rw [oneBin, alloc.vec.Vec.from_val, List.length_cons, List.length_nil]

theorem oneSlot_item : bitem oneSlot 0 = 0#usize := by
  show oneSlot.val[0]! = 0#usize
  rw [oneSlot, alloc.vec.Vec.from_val, List.getElem!_cons_zero]

/-- **Reachable success witness for the coverage loop.** At slot `0` with every one of the `256`
    home-shard slots marked — the mark array the real worker loop hands over when its element
    scans accepted every shard — the real coverage body takes its continuation branch, adopts the
    mark of slot `0` and advances to slot `1`, spending one unit of `coverRank`. The adopted value
    is `slot (Array.repeat 256#usize true) 0#usize`, the mark the scan wrote for shard `0`. -/
theorem cover_success_witness :
    ∃ c' : Bool, ∃ shard' : Usize,
      schedule_source.table.partition.Schedule.is_complete_loop1.body
          (Array.repeat 256#usize true) true 0#usize
        = ok (ControlFlow.cont (c', shard')) ∧
      c' = slot (Array.repeat 256#usize true) 0#usize ∧
      shard'.val = 1 ∧ coverRank shard' + 1 = coverRank 0#usize := by
  obtain ⟨c', shard', hw, hpos, hc', hrank⟩ :=
    cover_step (Array.repeat 256#usize true) 0#usize (by simp)
  exact ⟨c', shard', hw, hc', hpos, hrank⟩

/-- **Reachable refusal witness for the coverage loop.** At slot `0` with the all-unmarked mark
    array the real `is_complete` starts from, the real coverage body takes its continuation
    branch, adopts the mark of slot `0` and advances to slot `1`: the flag-down exit is the branch
    the scan reaches as soon as one home-shard slot was never marked. -/
theorem cover_refusal_witness :
    ∃ c' : Bool, ∃ shard' : Usize,
      schedule_source.table.partition.Schedule.is_complete_loop1.body
          (Array.repeat 256#usize false) true 0#usize
        = ok (ControlFlow.cont (c', shard')) ∧
      c' = slot (Array.repeat 256#usize false) 0#usize ∧
      shard'.val = 1 := by
  obtain ⟨c', shard', hw, hpos, hc', _⟩ :=
    cover_step (Array.repeat 256#usize false) 0#usize (by simp)
  exact ⟨c', shard', hw, hc', hpos⟩

/-- **Reachable continuation witness for the element scan.** At the concrete one-slot bin `[0]`,
    with the all-unmarked mark array the real `is_complete` starts from and read position `0`, the
    real element-scan body takes its continuation branch, advances the read position to `1` and
    spends exactly one unit of `scanRank`. The element it reads is `0`, a real home-shard slot. -/
theorem scan_continuation_witness :
    ∃ w : Array Bool 256#usize × Bool × Usize,
      schedule_source.table.partition.Schedule.is_complete_loop0_loop0.body
          oneSlot (Array.repeat 256#usize false) true 0#usize
        = ok (ControlFlow.cont w) ∧
      w.2.2.val = 1 ∧ scanRank oneSlot w.2.2 + 1 = scanRank oneSlot 0#usize ∧
      bitem oneSlot 0 = 0#usize := by
  obtain ⟨w, hw, hpos, hrank, _, _⟩ :=
    scan_step oneSlot (Array.repeat 256#usize false) 0#usize (by
      rw [oneSlot_length]; simp)
  exact ⟨w, hw, hpos, hrank, oneSlot_item⟩

/-- **Reachable worker-step witness.** The real worker body, at a one-bin schedule and worker
    position `0`, reads that bin, performs exactly one element scan over it and advances the
    worker position to `1`, spending one unit of `workerRank`. The element scan's result is the
    premise, which is the very call the body makes. -/
theorem worker_step_witness (r : Array Bool 256#usize × Bool)
    (hscan : schedule_source.table.partition.Schedule.is_complete_loop0_loop0
        (Array.repeat 256#usize false) true (wbin (oneBin) 0) 0#usize
      = ok r) :
    ∃ worker' : Usize,
      schedule_source.table.partition.Schedule.is_complete_loop0.body
          { bins := oneBin }
          (Array.repeat 256#usize false) true 0#usize
        = ok (ControlFlow.cont (r.1, r.2, worker')) ∧
      worker'.val = 1 ∧
      workerRank (oneBin) worker' + 1
        = workerRank (oneBin) 0#usize := by
  obtain ⟨worker', hw, hpos, hrank⟩ :=
    worker_step (oneBin) (Array.repeat 256#usize false) 0#usize
      (by rw [oneBin_length]; simp) r hscan
  exact ⟨worker', hw, hpos, hrank⟩

end ScheduleResource

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

#resource_audit ScheduleResource.shift256
#resource_audit ScheduleResource.shards_eq
#resource_audit ScheduleResource.shards_value
#resource_audit ScheduleResource.usize_ge_256
#resource_audit ScheduleResource.lt256
#resource_audit ScheduleResource.ge256
#resource_audit ScheduleResource.slot_set
#resource_audit ScheduleResource.slot_index
#resource_audit ScheduleResource.vec_index
#resource_audit ScheduleResource.vec_len_lt
#resource_audit ScheduleResource.vec_len_ge
#resource_audit ScheduleResource.scan_step
#resource_audit ScheduleResource.scan_exit_pos
#resource_audit ScheduleResource.scan_exit_flag
#resource_audit ScheduleResource.scan_rank_decreases
#resource_audit ScheduleResource.cover_step
#resource_audit ScheduleResource.cover_exit_pos
#resource_audit ScheduleResource.cover_exit_flag
#resource_audit ScheduleResource.worker_step
#resource_audit ScheduleResource.worker_exit_pos
#resource_audit ScheduleResource.worker_exit_flag
#resource_audit ScheduleResource.root_forwards
#resource_audit ScheduleResource.is_complete_decomposes
#resource_audit ScheduleResource.one_le_usize_max
#resource_audit ScheduleResource.oneSlot_length
#resource_audit ScheduleResource.oneSlot_item
#resource_audit ScheduleResource.oneBin_length
#resource_audit ScheduleResource.cover_success_witness
#resource_audit ScheduleResource.cover_refusal_witness
#resource_audit ScheduleResource.scan_continuation_witness
#resource_audit ScheduleResource.worker_step_witness

run_cmd Lean.logInfo "SCHEDULE_KERNEL_OK"
