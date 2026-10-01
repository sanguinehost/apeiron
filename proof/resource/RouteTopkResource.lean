import RouteTopkSource
import LoopTrace
import Aeneas

/-
SCOPE: the work measure of the actual extracted `route_topk` selection: the scan transition
      (`moe.best_unchosen_loop.body`) and the selection transition
      (`moe.route_topk_loop.body`) with their guards, the exact expert-inspection count and
      the integer-comparison bound in closed form over the expert count, the exit and refusal
      branches, and the reachable initial state with the per-pass success transition.
REGISTER: scalar source refinement of resource/work counts under named premises. No Rust cost
      model, no cycle, latency, allocation-byte or GPU time claim; no canonical-corpus
      correspondence (the functional refinement of these same bodies is
      `proof/corpus/Numeric/MoeRouter.lean`).
CORPUS: `src/engine/moe.rs` `route_topk` (222-251), `best_unchosen` (258-279) and the actual
      extracted `moe.route_topk_loop(.body)`, `moe.best_unchosen_loop(.body)`; the finite
      visit-count certificate is `proof/native/LoopTrace.lean` `LoopTrace.Trace` with
      `LoopTrace.projection`, registered there because `WP.loop.spec_decr_nat` carries no
      visit count.
REFERENCE: /home/socol/Workspace/apeiron/src/engine/moe.rs and
      /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/RouteTopk_gate.py --check --controls`, whole invocation <= 540 s
      under `/tmp/opencode/apeiron-lane-2.lock`, `MemoryMax=12G`, `MemorySwapMax=0`,
      `LEAN_NUM_THREADS=1`.
GRADE: `proved` for the theorems below, and only when that gate reports
      `ROUTE_TOPK_SOURCE_RESOURCE_OK` after a fresh extraction, a Lean build and an
      attributable real-source mutation/restore control. No measurement grade: nothing here is
      a timing or hardware claim.
RESIDUAL: the composed `k`-pass figure `k·E` is the sum of the per-pass figures proved here
      (`route_scan` charges each delegated scan `E` inspections, `route_step`/`route_step_pick`
      advance the pass counter by exactly one, so the rank `k - pass` falls by one per pass);
      the single composed `LoopTrace.Trace` over all `k` passes is not proved, and the premise
      that a pass's scan returns an index below `n` is carried as `hex` in
      `route_pass_inspects` rather than discharged from the mask.
-/

namespace RouteTopkResource
open Aeneas Aeneas.Std Result LoopTrace route_topk_source

/-! ## 1. The work measure

`src/engine/moe.rs:263-273` is the scan. Per invocation of
`moe.best_unchosen_loop.body` it evaluates the integer comparisons

* `j < n` (the loop guard, always), and
* `j < chosen.len` (the mask-length guard) plus, for an unchosen entry, `best == n` (the
  sentinel test) and `v > best_val` (the logit comparison),

and it inspects exactly one expert. `src/engine/moe.rs:235-246` is the selection pass. Per
invocation of `moe.route_topk_loop.body` it evaluates `pass < k`, delegates exactly one full
`best_unchosen` scan of the whole `n`-entry mask, and that scan's epilogue evaluates `best < n`.
The `chosen` mask test and the `Option`/`Vec` bookkeeping are not integer comparisons and are
not charged. The two charges below are the measure; no cycle or GPU time appears anywhere.
-/

/-- The scan state: the running best index, its logit and the scan cursor. -/
abbrev ScanSt := Std.Usize × Std.I64 × Std.Usize

/-- A successful result is not a refusal: the `Result` shape of the pinned model, from
    `Result.match.ok` and `Result.match.fail`. -/
theorem ok_ne_fail {α : Type} (a : α) (e : Error) : ok a ≠ fail e := by
  intro h
  have h' := congrArg Result.match h
  simp only [Result.match.ok, Result.match.fail] at h'
  exact absurd h' (by simp)

/-- The selection state, as `moe.route_topk_loop` carries it. -/
abbrev RouteSt :=
  alloc.vec.Vec Bool × alloc.vec.Vec Std.U32 × Bool × Option moe.MoeError × Std.Usize

/-- The actual scan body at a fixed logit vector, mask length and expert count. -/
def scanBody (logits : Slice Std.I64) (chosen : Slice Bool) (n : Std.Usize) :
    ScanSt → Result (ControlFlow ScanSt Std.Usize) :=
  fun st => moe.best_unchosen_loop.body logits chosen n st.1 st.2.1 st.2.2

/-- The actual selection body at a fixed logit vector and width. -/
def routeBody (logits : Slice Std.I64) (k n : Std.Usize) :
    RouteSt →
      Result
        (ControlFlow RouteSt (alloc.vec.Vec Std.U32 × Option moe.MoeError)) :=
  fun st =>
    moe.route_topk_loop.body logits k n st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2

/-- Expert inspections charged to one scan invocation: one per entered step, none at the exit. -/
def scanInspect (n : Std.Usize) (st : ScanSt) : Nat :=
  if st.2.2.val < n.val then 1 else 0

/-- Integer comparisons charged to one scan invocation, per the convention above. -/
def scanCmp (chosen : Slice Bool) (n : Std.Usize) (st : ScanSt) : Nat :=
  if st.2.2.val ≥ n.val then 1
  else if st.2.2.val ≥ chosen.val.length then 2
  else if chosen.val.getD st.2.2.val false then 2
  else if st.1.val = n.val then 3
  else 4

/-- Integer comparisons charged to one selection invocation: the pass guard, plus the
    delegated scan's `best < n` test on a pass, and nothing on the exit. -/
def routeCmp (k : Std.Usize) (st : RouteSt) : Nat :=
  if st.2.2.2.2.val < k.val then 2 else 1

/-- Expert inspections charged to one selection invocation: the delegated scan visits every
    one of the `n` experts, by `scan_inspect_closed_form` below. -/
def routeInspect (k n : Std.Usize) (st : RouteSt) : Nat :=
  if st.2.2.2.2.val < k.val then n.val else 0

/-! ## 2. The scan transition, with guards -/

/-- A scan step advances the cursor by exactly one and returns the continuation. The guard is
    `j < n`; the compared values are whatever the running state selects. -/
theorem scan_step (logits : Slice Std.I64) (chosen : Slice Bool) (n : Std.Usize)
    (hsz : logits.val.length = n.val ∧ chosen.val.length = n.val)
    (best : Std.Usize) (best_val : Std.I64) (j : Std.Usize) (hjn : j.val < n.val) :
    ∃ best' best_val' j2,
      moe.best_unchosen_loop.body logits chosen n best best_val j =
        ok (ControlFlow.cont (best', best_val', j2)) ∧
      j2.val = j.val + 1 ∧
      n.val - j2.val < n.val - j.val ∧ n.val - j2.val + 1 = n.val - j.val ∧
      (chosen.val.getD j.val false ≠ true → best' = j ∨ (best', best_val') = (best, best_val)) ∧
      (chosen.val.getD j.val false = true → (best', best_val') = (best, best_val)) := by
  have hguard : j < n := (UScalar.lt_equiv j n).2 hjn
  have hjk : j.val < chosen.val.length := by rw [hsz.2]; exact hjn
  have hjl : j.val < logits.val.length := by rw [hsz.1]; exact hjn
  have hidxc : Slice.index_usize chosen j = ok (chosen.val.getD j.val false) := by
    unfold Slice.index_usize
    rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hjk, List.getD_eq_getElem _ _ hjk]
  have hidxl : Slice.index_usize logits j = ok (logits.val.getD j.val 0#i64) := by
    unfold Slice.index_usize
    rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hjl, List.getD_eq_getElem _ _ hjl]
  have hlen_eq : Slice.len chosen = n := by
    apply UScalar.eq_of_val_eq
    simp only [Slice.len_val, hsz.2]
  have h1 : (1#usize).val = 1 := by decide
  have hjmax : j.val + (1#usize).val ≤ Std.Usize.max := by
    rw [h1]
    have := Slice.property logits
    rw [hsz.1] at this
    omega
  obtain ⟨j2, haddj, hj2⟩ := WP.spec_imp_exists
    (Std.Usize.add_spec (x := j) (y := 1#usize) hjmax)
  have hj2' : j2.val = j.val + 1 := by rw [hj2, h1]
  have hstep : n.val - j2.val < n.val - j.val ∧ n.val - j2.val + 1 = n.val - j.val := by
    rw [hj2']; omega
  by_cases htk : chosen.val.getD j.val false = true
  · -- the entry is already chosen: the running best is carried unchanged
    refine ⟨best, best_val, j2, ?_, hj2', hstep.1, hstep.2, ?_, ?_⟩
    · rw [moe.best_unchosen_loop.body, if_pos hguard]
      simp only [hlen_eq, if_pos hguard, hidxc, hidxl, bind_tc_ok, if_pos htk, haddj]
      rfl
    · rintro _
      exact Or.inr rfl
    · rintro _
      rfl
  · by_cases hbn : best = n
    · -- no unchosen entry has been seen: the cursor becomes the new best
      refine ⟨j, logits.val.getD j.val 0#i64, j2, ?_, hj2', hstep.1, hstep.2, ?_, ?_⟩
      · rw [moe.best_unchosen_loop.body, if_pos hguard]
        simp only [hlen_eq, if_pos hguard, hidxc, hidxl, bind_tc_ok, if_neg htk, if_pos hbn,
          haddj]
        rfl
      · rintro _
        exact Or.inl rfl
      · rintro h
        exact absurd h htk
    · by_cases hgt : logits.val.getD j.val 0#i64 > best_val
      · -- the cursor's logit is larger: the cursor becomes the new best
        refine ⟨j, logits.val.getD j.val 0#i64, j2, ?_, hj2', hstep.1, hstep.2, ?_, ?_⟩
        · rw [moe.best_unchosen_loop.body, if_pos hguard]
          simp only [hlen_eq, if_pos hguard, hidxc, hidxl, bind_tc_ok, if_neg htk, if_neg hbn,
            if_pos hgt, haddj]
          rfl
        · rintro _
          exact Or.inl rfl
        · rintro h
          exact absurd h htk
      · -- the running best is larger: it is carried unchanged
        refine ⟨best, best_val, j2, ?_, hj2', hstep.1, hstep.2, ?_, ?_⟩
        · rw [moe.best_unchosen_loop.body, if_pos hguard]
          simp only [hlen_eq, if_pos hguard, hidxc, hidxl, bind_tc_ok, if_neg htk, if_neg hbn,
            if_neg hgt, haddj]
          rfl
        · rintro _
          exact Or.inr rfl
        · rintro h
          exact absurd h htk

/-- The scan exit: at or past the expert count the body returns the running best and evaluates
    one comparison. -/
theorem scan_exit (logits : Slice Std.I64) (chosen : Slice Bool) (n : Std.Usize)
    (best : Std.Usize) (best_val : Std.I64) (j : Std.Usize) (hjn : ¬ j.val < n.val) :
    moe.best_unchosen_loop.body logits chosen n best best_val j =
      ok (ControlFlow.done best) ∧ scanCmp chosen n (best, best_val, j) = 1 := by
  have hnot : ¬ j < n := fun hc => hjn ((UScalar.lt_equiv j n).1 hc)
  refine ⟨?_, ?_⟩
  · rw [moe.best_unchosen_loop.body, if_neg hnot]
  · simp [scanCmp, hjn]

/-- One scan invocation never charges more than four integer comparisons. -/
theorem scanCmp_le (chosen : Slice Bool) (n : Std.Usize) (st : ScanSt) :
    scanCmp chosen n st ≤ 4 := by
  unfold scanCmp
  split <;> first
    | omega
    | (split <;> first
        | omega
        | (split <;> first
            | omega
            | (split <;> omega)))

/-- The observed continuation of the scan body forces the guard and the cursor advance. -/
theorem scan_observed_cont (logits : Slice Std.I64) (chosen : Slice Bool) (n : Std.Usize)
    (hsz : logits.val.length = n.val ∧ chosen.val.length = n.val)
    (st next : ScanSt)
    (hobs : moe.best_unchosen_loop.body logits chosen n st.1 st.2.1 st.2.2 =
      ok (ControlFlow.cont next)) :
    st.2.2.val < n.val ∧ next.2.2.val = st.2.2.val + 1 := by
  have hguard : st.2.2 < n := by
    by_contra hnot
    have hdone : moe.best_unchosen_loop.body logits chosen n st.1 st.2.1 st.2.2 =
        ok (ControlFlow.done st.1) := by
      rw [moe.best_unchosen_loop.body, if_neg hnot]
    have hbad : ControlFlow.done st.1 = ControlFlow.cont next :=
      Result.ok_injective (hdone.symm.trans hobs)
    cases hbad
  obtain ⟨best', best_val', j2, hbody, hj2, _, _⟩ :=
    scan_step logits chosen n hsz st.1 st.2.1 st.2.2 ((UScalar.lt_equiv _ _).1 hguard)
  refine ⟨((UScalar.lt_equiv _ _).1 hguard), ?_⟩
  have hid : next = (best', best_val', j2) := by
    have := Result.ok_injective (hobs.symm.trans hbody)
    cases this
    rfl
  rw [hid]
  exact hj2

/-! ## 3. The scan work measure: exact closed form and comparison bound -/

/-- The scan visits exactly the experts from the cursor to the expert count, and the loop
    projects to the actual `moe.best_unchosen_loop` result. The count is
    `n.val - j0.val` expert inspections, and the rank `n.val - j` falls by one per step. -/
theorem scan_inspect_closed_form (logits : Slice Std.I64) (chosen : Slice Bool) (n : Std.Usize)
    (hsz : logits.val.length = n.val ∧ chosen.val.length = n.val)
    (best : Std.Usize) (best_val : Std.I64) (j0 : Std.Usize) (hj0 : j0.val ≤ n.val) :
    ∃ r, moe.best_unchosen_loop logits chosen n best best_val j0 = ok r ∧
      LoopTrace.Trace (scanBody logits chosen n) (scanInspect n) (best, best_val, j0)
        (ok r) (n.val - j0.val) := by
  have key : ∀ (m : Nat) (best : Std.Usize) (best_val : Std.I64) (j : Std.Usize),
      j.val + m = n.val →
      ∃ r, LoopTrace.Trace (scanBody logits chosen n) (scanInspect n) (best, best_val, j)
        (ok r) m := by
    intro m
    induction m with
    | zero =>
        intro best best_val j hj
        have hjn : ¬ j.val < n.val := by omega
        obtain ⟨hbody, _⟩ := scan_exit logits chosen n best best_val j hjn
        have hch : scanInspect n (best, best_val, j) = 0 := by simp [scanInspect, hjn]
        refine ⟨best, ?_⟩
        simpa only [hch] using
          (Trace.done (body := scanBody logits chosen n) (charge := scanInspect n)
            (state := (best, best_val, j)) (value := best) (by simpa [scanBody] using hbody))
    | succ m ih =>
        intro best best_val j hj
        have hjn : j.val < n.val := by omega
        obtain ⟨best', best_val', j2, hbody, hj2, _, _⟩ :=
          scan_step logits chosen n hsz best best_val j hjn
        have hj2m : j2.val + m = n.val := by omega
        obtain ⟨r, ihTrace⟩ := ih best' best_val' j2 hj2m
        have hch : scanInspect n (best, best_val, j) = 1 := by simp [scanInspect, hjn]
        refine ⟨r, ?_⟩
        simpa only [hch, Nat.add_comm] using
          (Trace.cont (body := scanBody logits chosen n) (charge := scanInspect n)
            (state := (best, best_val, j)) (next := (best', best_val', j2))
            (by simpa [scanBody] using hbody) ihTrace)
  obtain ⟨m, hm⟩ := Nat.exists_eq_add_of_le hj0
  obtain ⟨r, htrace⟩ := key m best best_val j0 (by omega)
  refine ⟨r, ?_, ?_⟩
  · unfold moe.best_unchosen_loop
    exact LoopTrace.projection htrace
  · have : n.val - j0.val = m := by omega
    rwa [this]

/-- The scan's total integer-comparison count is at most four per inspected expert plus the
    exit guard: `4·(n - j0) + 1`, on the same rank. -/
theorem scan_cmp_bounded (logits : Slice Std.I64) (chosen : Slice Bool) (n : Std.Usize)
    (hsz : logits.val.length = n.val ∧ chosen.val.length = n.val)
    (st : ScanSt) (res : Result Std.Usize) (c : Nat) :
    LoopTrace.Trace (scanBody logits chosen n) (scanCmp chosen n) st res c →
      c ≤ 4 * (n.val - st.2.2.val) + 1 := by
  intro t
  induction t with
  | @done state value observed =>
      have hle : scanCmp chosen n state <= 4 := scanCmp_le chosen n state
      by_cases hlt : state.2.2.val < n.val
      · have hpos : 1 <= n.val - state.2.2.val := by omega
        omega
      · obtain ⟨hbody, hcmp⟩ := scan_exit logits chosen n state.1 state.2.1 state.2.2 hlt
        have hobs : moe.best_unchosen_loop.body logits chosen n state.1 state.2.1 state.2.2 =
            ok (ControlFlow.done value) := by simpa [scanBody] using observed
        have hbad : ControlFlow.done value = ControlFlow.done state.1 :=
          Result.ok_injective (hobs.symm.trans hbody)
        cases hbad
        have hcmp' : scanCmp chosen n state = 1 := hcmp
        rw [hcmp']
        omega
  | @failure state error observed =>
      have hle : scanCmp chosen n state ≤ 4 := scanCmp_le chosen n state
      by_cases hlt : state.2.2.val < n.val
      · have hpos : 1 ≤ n.val - state.2.2.val := by omega
        omega
      · exfalso
        have hdone : moe.best_unchosen_loop.body logits chosen n state.1 state.2.1 state.2.2 =
            ok (ControlFlow.done state.1) := by
          rw [moe.best_unchosen_loop.body, if_neg (by omega)]
        exact ok_ne_fail _ _ (hdone.symm.trans (by simpa [scanBody] using observed))
  | @cont state next result count observed tail ih =>
      have hfacts := scan_observed_cont logits chosen n hsz state next
        (by simpa [scanBody] using observed)
      have hle : scanCmp chosen n state ≤ 4 := scanCmp_le chosen n state
      have hpos : 1 ≤ n.val - state.2.2.val := by omega
      omega

/-! ## 4. The selection transition, with guards -/

/-- `Vec.set` preserves the length: the mask the selection pass writes back has the same
    number of entries as the mask it read. -/
theorem vec_set_length {α : Type} (v : alloc.vec.Vec α) (i : Std.Usize) (x : α) :
    (v.set i x).val.length = v.val.length := by
  rw [alloc.vec.Vec.set_val_eq, List.length_set]

/-- The `Usize → U32` narrowing the selection pass performs is exact on an in-range index. -/
theorem usize_cast_u32_val {b : Std.Usize} (hb : b.val < 2 ^ 32) :
    (UScalar.cast UScalarTy.U32 b).val = b.val := by
  rw [UScalar.cast_val_eq]
  have h32 : UScalarTy.U32.numBits = 32 := rfl
  rw [h32]
  exact Nat.mod_eq_of_lt hb

/-- `Slice.len` reads the list length: the expert count the scan is started with. -/
theorem slice_len_eq (logits : Slice Std.I64) (n : Std.Usize) (h : logits.val.length = n.val) :
    Slice.len logits = n := by
  apply UScalar.eq_of_val_eq
  simp [h]

/-- The model `Vec.push` appends, on a length that stays in range. -/
theorem vec_push_eq {α : Type} (v : alloc.vec.Vec α) (x : α)
    (h : v.val.length + 1 ≤ U32.max) :
    v.push x = ok (alloc.vec.Vec.from (v.val ++ [x]) (by simp; scalar_tac)) := by
  have hc : (decide (v.val.length + 1 ≤ U32.max)
      || decide (v.val.length + 1 ≤ Std.Usize.max)) = true := by
    rw [decide_eq_true h]; rfl
  unfold alloc.vec.Vec.push
  simp only [hc, ↓reduceDIte, List.concat_eq_append]

/-- The model `Vec.push` appends, in the existentially quantified form the step uses. -/
theorem vec_push_val {α : Type} (v : alloc.vec.Vec α) (x : α)
    (h : v.val.length + 1 ≤ U32.max) :
    ∃ v' : alloc.vec.Vec α, v.push x = ok v' ∧ v'.val = v.val ++ [x] :=
  ⟨_, vec_push_eq v x h, alloc.vec.Vec.from_val _ _⟩

/-- `best_unchosen` reads the scan's index and its `best < n` test off the delegated loop: the
    index and the `true` flag when the scan's answer is below the expert count. -/
theorem best_unchosen_of_scan_true (logits : Slice Std.I64) (n : Std.Usize) (chosen : Slice Bool)
    (hlen : Slice.len logits = n) (b : Std.Usize) (hb : b.val < n.val)
    (hscan : moe.best_unchosen_loop logits chosen n n 0#i64 0#usize = ok b) :
    moe.best_unchosen logits chosen = ok (b, true) := by
  have hscan' : moe.best_unchosen_loop logits chosen (Slice.len logits) (Slice.len logits) 0#i64
      0#usize = ok b := by rw [hlen]; exact hscan
  have hstep : (moe.best_unchosen_loop logits chosen (Slice.len logits) (Slice.len logits)
        0#i64 0#usize >>= fun best =>
      if best < Slice.len logits then ok (best, true) else ok (0#usize, false)) =
      if b < Slice.len logits then ok (b, true) else ok (0#usize, false) := by
    rw [hscan', bind_tc_ok]
  rw [moe.best_unchosen, hstep, hlen, if_pos ((UScalar.lt_equiv b n).2 hb)]

/-- `best_unchosen` reports no unchosen expert when the scan's answer is at or past the expert
    count. -/
theorem best_unchosen_of_scan_false (logits : Slice Std.I64) (n : Std.Usize)
    (chosen : Slice Bool) (hlen : Slice.len logits = n) (b : Std.Usize) (hb : ¬ b.val < n.val)
    (hscan : moe.best_unchosen_loop logits chosen n n 0#i64 0#usize = ok b) :
    moe.best_unchosen logits chosen = ok (0#usize, false) := by
  have hscan' : moe.best_unchosen_loop logits chosen (Slice.len logits) (Slice.len logits) 0#i64
      0#usize = ok b := by rw [hlen]; exact hscan
  have hstep : (moe.best_unchosen_loop logits chosen (Slice.len logits) (Slice.len logits)
        0#i64 0#usize >>= fun best =>
      if best < Slice.len logits then ok (best, true) else ok (0#usize, false)) =
      if b < Slice.len logits then ok (b, true) else ok (0#usize, false) := by
    rw [hscan', bind_tc_ok]
  rw [moe.best_unchosen, hstep, hlen, if_neg (fun hc => hb ((UScalar.lt_equiv b n).1 hc))]

/-- The scan the selection pass delegates: the whole `n`-entry mask, from cursor `0`. -/
theorem route_scan_exists (logits : Slice Std.I64) (n : Std.Usize)
    (hsz : logits.val.length = n.val) (chosen : alloc.vec.Vec Bool)
    (hchosen : chosen.val.length = n.val) :
    ∃ b, moe.best_unchosen_loop logits (alloc.vec.Vec.deref chosen) n n 0#i64 0#usize = ok b := by
  have hscanLen : (alloc.vec.Vec.deref chosen).val.length = n.val := by
    rw [alloc.vec.Vec.deref, Slice.from_val, hchosen]
  obtain ⟨b, hloop, _⟩ :=
    scan_inspect_closed_form logits (alloc.vec.Vec.deref chosen) n ⟨hsz, hscanLen⟩
      n 0#i64 0#usize (by simp)
  exact ⟨b, hloop⟩

/-- The delegated scan, with its exact expert-inspection count: `n` experts, every pass. -/
theorem route_scan (logits : Slice Std.I64) (n : Std.Usize)
    (hsz : logits.val.length = n.val) (chosen : alloc.vec.Vec Bool)
    (hchosen : chosen.val.length = n.val) :
    ∃ b, moe.best_unchosen_loop logits (alloc.vec.Vec.deref chosen) n n 0#i64 0#usize = ok b ∧
      LoopTrace.Trace (scanBody logits (alloc.vec.Vec.deref chosen) n) (scanInspect n)
        (n, 0#i64, 0#usize) (ok b) n.val := by
  have hscanLen : (alloc.vec.Vec.deref chosen).val.length = n.val := by
    rw [alloc.vec.Vec.deref, Slice.from_val, hchosen]
  obtain ⟨b, hloop, htrace⟩ :=
    scan_inspect_closed_form logits (alloc.vec.Vec.deref chosen) n ⟨hsz, hscanLen⟩
      n 0#i64 0#usize (by simp)
  refine ⟨b, hloop, ?_⟩
  simpa using htrace

/-- The selection transition, with guards. Under `pass < k` and `ok1`, one pass runs the
    delegated scan and then either records the choice or records the refusal; in both cases the
    pass counter advances by exactly one and the rank `k - pass` falls by one. -/
theorem route_step (logits : Slice Std.I64) (k n : Std.Usize)
    (hsz : logits.val.length = n.val) (hkN : k.val ≤ n.val) (hnmax : n.val ≤ 4294967295)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec Std.U32) (err : Option moe.MoeError)
    (pass : Std.Usize)
    (hchosen : chosen.val.length = n.val) (hsel : sel.val.length = pass.val)
    (hpass : pass.val < k.val) :
    (∃ (chosen2 : alloc.vec.Vec Bool) (sel2 : alloc.vec.Vec Std.U32) (pass2 : Std.Usize)
        (b : Std.Usize),
      b.val < n.val ∧
      moe.route_topk_loop.body logits k n chosen sel true err pass =
        ok (ControlFlow.cont (chosen2, sel2, true, err, pass2)) ∧
      chosen2 = chosen.set b true ∧
      sel2.val = sel.val ++ [(UScalar.cast UScalarTy.U32 b)] ∧
      chosen2.val.length = n.val ∧ sel2.val.length = sel.val.length + 1 ∧
      pass2.val = pass.val + 1 ∧ k.val - pass2.val < k.val - pass.val ∧
      k.val - pass2.val + 1 = k.val - pass.val ∧ sel2.val.length = pass2.val) ∨
    (∃ (pass2 : Std.Usize) (b : Std.Usize),
      ¬ b.val < n.val ∧
      moe.route_topk_loop.body logits k n chosen sel true err pass =
        ok (ControlFlow.cont (chosen, sel, false, some (moe.MoeError.BadK k n), pass2)) ∧
      pass2.val = pass.val + 1 ∧ k.val - pass2.val < k.val - pass.val ∧
      k.val - pass2.val + 1 = k.val - pass.val) := by
  have hpassProp : pass < k := (UScalar.lt_equiv pass k).2 hpass
  have hlen : Slice.len logits = n := slice_len_eq logits n hsz
  obtain ⟨b, hscanLoop, _⟩ := route_scan logits n hsz chosen hchosen
  have hnm : n.val ≤ Std.Usize.max := by
    have hnm := Slice.property logits
    rwa [hsz] at hnm
  have h1 : (1#usize).val = 1 := by decide
  have hpmax : pass.val + (1#usize).val ≤ Std.Usize.max := by
    rw [h1]; omega
  obtain ⟨pass2, haddpass, hpass2⟩ := WP.spec_imp_exists
    (Std.Usize.add_spec (x := pass) (y := 1#usize) hpmax)
  have hpass2' : pass2.val = pass.val + 1 := by rw [hpass2, h1]
  have hrank : k.val - pass2.val < k.val - pass.val ∧
      k.val - pass2.val + 1 = k.val - pass.val := by rw [hpass2']; omega
  by_cases hb : b.val < n.val
  · left
    have hsb : sel.val.length + 1 ≤ U32.max := by
      rw [hsel]
      have hu : (4294967295 : ℕ) = U32.max := U32.max_eq.symm
      omega
    obtain ⟨sel2, hpush, hsel2⟩ := vec_push_val sel (UScalar.cast UScalarTy.U32 b) hsb
    have hset : (chosen.set b true).val.length = n.val :=
      vec_set_length chosen b true |>.trans hchosen
    have hlen2 : sel2.val.length = sel.val.length + 1 := by
      rw [hsel2, List.length_append, List.length_singleton]
    have hlen3 : sel2.val.length = pass2.val := by
      rw [hsel2, List.length_append, List.length_singleton, hsel, hpass2']
    refine ⟨chosen.set b true, sel2, pass2, b, hb, ?_, rfl, hsel2, hset, hlen2, hpass2', hrank.1,
      hrank.2, hlen3⟩
    have hbprop : b < n := (UScalar.lt_equiv b n).2 hb
    have hbe : moe.best_unchosen logits (alloc.vec.Vec.deref chosen) = ok (b, true) :=
      best_unchosen_of_scan_true logits n (alloc.vec.Vec.deref chosen) hlen b hb hscanLoop
    have hblen : b.val < chosen.val.length := by rw [hchosen]; exact hb
    have hidxmut : alloc.vec.Vec.index_mut_usize chosen b =
        ok (chosen.val[b.val]'hblen, chosen.set b) := by
      unfold alloc.vec.Vec.index_mut_usize
      have hindex : alloc.vec.Vec.index_usize chosen b = ok (chosen.val[b.val]'hblen) := by
        unfold alloc.vec.Vec.index_usize
        rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem hblen]
      rw [hindex]
      simp only [bind_tc_ok]
    rw [moe.route_topk_loop.body, if_pos hpassProp]
    simp [hbe, hidxmut, hpush, haddpass, lift]
  · right
    refine ⟨pass2, b, hb, ?_, hpass2', hrank.1, hrank.2⟩
    have hbe : moe.best_unchosen logits (alloc.vec.Vec.deref chosen) = ok (0#usize, false) :=
      best_unchosen_of_scan_false logits n (alloc.vec.Vec.deref chosen) hlen b hb hscanLoop
    rw [moe.route_topk_loop.body, if_pos hpassProp]
    simp [hbe, haddpass]

/-- The selection step that records a choice, given that the delegated scan returns an index
    below `n`. The expert is marked chosen, its narrowed index is appended, the pass counter
    advances by exactly one, and the rank `k - pass` falls by one. -/
theorem route_step_pick (logits : Slice Std.I64) (k n : Std.Usize)
    (hsz : logits.val.length = n.val) (hkN : k.val ≤ n.val) (hnmax : n.val ≤ 4294967295)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec Std.U32) (err : Option moe.MoeError)
    (pass : Std.Usize)
    (hchosen : chosen.val.length = n.val) (hsel : sel.val.length = pass.val)
    (hpass : pass.val < k.val)
    (hex : ∃ b, moe.best_unchosen_loop logits (alloc.vec.Vec.deref chosen) n n 0#i64 0#usize
      = ok b ∧ b.val < n.val) :
    ∃ (chosen2 : alloc.vec.Vec Bool) (sel2 : alloc.vec.Vec Std.U32) (pass2 : Std.Usize) (b : Std.Usize),
      b.val < n.val ∧
      moe.route_topk_loop.body logits k n chosen sel true err pass =
        ok (ControlFlow.cont (chosen2, sel2, true, err, pass2)) ∧
      chosen2 = chosen.set b true ∧
      sel2.val = sel.val ++ [(UScalar.cast UScalarTy.U32 b)] ∧
      chosen2.val.length = n.val ∧ sel2.val.length = sel.val.length + 1 ∧
      pass2.val = pass.val + 1 ∧ k.val - pass2.val < k.val - pass.val ∧
      k.val - pass2.val + 1 = k.val - pass.val ∧ sel2.val.length = pass2.val := by
  obtain ⟨b, hres, hb⟩ := hex
  have hpassProp : pass < k := (UScalar.lt_equiv pass k).2 hpass
  have hstep := route_step logits k n hsz hkN hnmax chosen sel err pass hchosen hsel hpass
  rcases hstep with hleft | hright
  · obtain ⟨chosen2, sel2, pass2, b2, hb2, hbody, hset, hsel2, hclen, hslen, hpass2, hr1, hr2,
      hsel2pass⟩ := hleft
    refine ⟨chosen2, sel2, pass2, b2, hb2, hbody, hset, hsel2, hclen, hslen, hpass2, hr1, hr2,
      hsel2pass⟩
  · obtain ⟨pass2, b2, hb2, hbody, hpass2, hr1, hr2⟩ := hright
    exfalso
    have hlen := slice_len_eq logits n hsz
    -- the scan returns an index below `n`, so the body took the pick branch
    have hbe : moe.best_unchosen logits (alloc.vec.Vec.deref chosen) = ok (b, true) :=
      best_unchosen_of_scan_true logits n (alloc.vec.Vec.deref chosen) hlen b hb hres
    have hblen : b.val < chosen.val.length := by rw [hchosen]; exact hb
    have hidxmut : alloc.vec.Vec.index_mut_usize chosen b =
        ok (chosen.val[b.val]'hblen, chosen.set b) := by
      unfold alloc.vec.Vec.index_mut_usize
      have hindex : alloc.vec.Vec.index_usize chosen b = ok (chosen.val[b.val]'hblen) := by
        unfold alloc.vec.Vec.index_usize
        rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem hblen]
      rw [hindex]
      simp only [bind_tc_ok]
    have hsb : sel.val.length + 1 ≤ U32.max := by
      rw [hsel]
      have hu : (4294967295 : ℕ) = U32.max := U32.max_eq.symm
      omega
    obtain ⟨sel3, hpush, hsel3⟩ := vec_push_val sel (UScalar.cast UScalarTy.U32 b) hsb
    have h1 : (1#usize).val = 1 := by decide
    have hnm : n.val ≤ Std.Usize.max := by
      have hnm := Slice.property logits
      rwa [hsz] at hnm
    have hpmax : pass.val + (1#usize).val ≤ Std.Usize.max := by
      rw [h1]; omega
    obtain ⟨pass3, haddpass, hpass3⟩ := WP.spec_imp_exists
      (Std.Usize.add_spec (x := pass) (y := 1#usize) hpmax)
    -- the scan returns an index below `n`, so the body took the pick branch: its `ok` flag is
    -- the scan's `true`, while the refusal branch's is `false`
    have hblen : b.val < chosen.val.length := by rw [hchosen]; exact hb
    have hidxmut : alloc.vec.Vec.index_mut_usize chosen b =
        ok (chosen.val[b.val]'hblen, chosen.set b) := by
      unfold alloc.vec.Vec.index_mut_usize
      have hindex : alloc.vec.Vec.index_usize chosen b = ok (chosen.val[b.val]'hblen) := by
        unfold alloc.vec.Vec.index_usize
        rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem hblen]
      rw [hindex]
      simp only [bind_tc_ok]
    have hsb : sel.val.length + 1 ≤ U32.max := by
      rw [hsel]
      have hu : (4294967295 : ℕ) = U32.max := U32.max_eq.symm
      omega
    have hpush : alloc.vec.Vec.push sel (UScalar.cast UScalarTy.U32 b) =
        ok (alloc.vec.Vec.from (sel.val ++ [(UScalar.cast UScalarTy.U32 b)])
          (by simp; scalar_tac)) := vec_push_eq sel (UScalar.cast UScalarTy.U32 b) hsb
    have h1 : (1#usize).val = 1 := by decide
    have hnm : n.val ≤ Std.Usize.max := by
      have hnm := Slice.property logits
      rwa [hsz] at hnm
    have hpmax : pass.val + (1#usize).val ≤ Std.Usize.max := by
      rw [h1]; omega
    obtain ⟨pass3, haddpass, hpass3⟩ := WP.spec_imp_exists
      (Std.Usize.add_spec (x := pass) (y := 1#usize) hpmax)
    have hthis := hbody
    rw [moe.route_topk_loop.body, if_pos hpassProp] at hthis
    simp [hbe, hidxmut, hpush, haddpass, lift] at hthis

/-- The selection step that records the refusal, given that the delegated scan reports no
    unchosen expert. `BadK k n` is recorded, both vectors are untouched, the pass counter
    advances by exactly one, and the rank `k - pass` falls by one. -/
theorem route_step_refuse (logits : Slice Std.I64) (k n : Std.Usize)
    (hsz : logits.val.length = n.val) (hkN : k.val ≤ n.val) (hnmax : n.val ≤ 4294967295)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec Std.U32) (err : Option moe.MoeError)
    (pass : Std.Usize)
    (hchosen : chosen.val.length = n.val) (hsel : sel.val.length = pass.val)
    (hpass : pass.val < k.val)
    (hall : ∀ b, moe.best_unchosen_loop logits (alloc.vec.Vec.deref chosen) n n 0#i64 0#usize
      = ok b → ¬ b.val < n.val) :
    ∃ (pass2 : Std.Usize) (b : Std.Usize),
      ¬ b.val < n.val ∧
      moe.route_topk_loop.body logits k n chosen sel true err pass =
        ok (ControlFlow.cont (chosen, sel, false, some (moe.MoeError.BadK k n), pass2)) ∧
      pass2.val = pass.val + 1 ∧ k.val - pass2.val < k.val - pass.val ∧
      k.val - pass2.val + 1 = k.val - pass.val := by
  obtain ⟨b, hres⟩ := route_scan_exists logits n hsz chosen hchosen
  have hnb : ¬ b.val < n.val := hall b hres
  have hpassProp : pass < k := (UScalar.lt_equiv pass k).2 hpass
  have hstep := route_step logits k n hsz hkN hnmax chosen sel err pass hchosen hsel hpass
  rcases hstep with hleft | hright
  · obtain ⟨chosen2, sel2, pass2, b2, hb2, hbody, hset, hsel2, hclen, hslen, hpass2, hr1, hr2,
      hsel2pass⟩ := hleft
    exfalso
    have hlen := slice_len_eq logits n hsz
    have hfalse : moe.best_unchosen logits (alloc.vec.Vec.deref chosen) =
        ok (0#usize, false) :=
      best_unchosen_of_scan_false logits n (alloc.vec.Vec.deref chosen) hlen b hnb hres
    have h1 : (1#usize).val = 1 := by decide
    have hnm : n.val ≤ Std.Usize.max := by
      have hnm := Slice.property logits
      rwa [hsz] at hnm
    have hpmax : pass.val + (1#usize).val ≤ Std.Usize.max := by
      rw [h1]; omega
    obtain ⟨pass3, haddpass, hpass3⟩ := WP.spec_imp_exists
      (Std.Usize.add_spec (x := pass) (y := 1#usize) hpmax)
    -- the scan's flag is `false` here, so the body took the refusal branch: its `ok` flag is
    -- the scan's `false`, while the pick branch's is `true`
    have hthis := hbody
    rw [moe.route_topk_loop.body, if_pos hpassProp] at hthis
    simp [hfalse, haddpass, lift] at hthis
  · obtain ⟨pass2, b2, hb2, hbody, hpass2, hr1, hr2⟩ := hright
    refine ⟨pass2, b2, hb2, hbody, hpass2, hr1, hr2⟩


/-! ## 5. The selection exits -/

/-- The exit at the width: with `¬ (pass < k)` the body returns the result vector and the
    recorded refusal unchanged, after one comparison. -/
theorem route_exit_width (logits : Slice Std.I64) (k n : Std.Usize)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec Std.U32) (err : Option moe.MoeError)
    (pass : Std.Usize) (hpass : ¬ pass.val < k.val) :
    moe.route_topk_loop.body logits k n chosen sel true err pass = ok (ControlFlow.done (sel, err))
      ∧ routeCmp k (chosen, sel, true, err, pass) = 1 := by
  have hnot : ¬ pass < k := fun hc => hpass ((UScalar.lt_equiv pass k).1 hc)
  refine ⟨?_, ?_⟩
  · rw [moe.route_topk_loop.body, if_neg hnot]
  · simp [routeCmp, hpass]

/-- The exit on the recorded refusal: with `pass < k` and `ok1 = false` the body returns the
    result vector and the recorded refusal unchanged, after one comparison. -/
theorem route_exit_refusal (logits : Slice Std.I64) (k n : Std.Usize)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec Std.U32) (err : Option moe.MoeError)
    (pass : Std.Usize) (hpass : pass.val < k.val) :
    moe.route_topk_loop.body logits k n chosen sel false err pass = ok (ControlFlow.done (sel, err))
      ∧ routeCmp k (chosen, sel, false, err, pass) = 2 := by
  have hpassProp : pass < k := (UScalar.lt_equiv pass k).2 hpass
  refine ⟨?_, ?_⟩
  · rw [moe.route_topk_loop.body, if_pos hpassProp, if_neg (by simp)]
  · simp [routeCmp, hpass]

/-- The loop projects to the actual `moe.route_topk_loop` result, so every count below is a
    count on the extracted body. -/
theorem route_loop_projects (logits : Slice Std.I64) (k n : Std.Usize) (st : RouteSt)
    {r : alloc.vec.Vec Std.U32 × Option moe.MoeError} {c : Nat}
    (t : LoopTrace.Trace (routeBody logits k n) (routeInspect k n) st (ok r) c) :
    moe.route_topk_loop logits k n st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2 = ok r := by
  unfold moe.route_topk_loop
  exact LoopTrace.projection t

/-! ## 6. The work measure of the whole selection

`scan_inspect_closed_form` charges the delegated scan exactly the experts it walks
(`n.val - j0.val` inspections, the rank falling by one per step), `scan_cmp_bounded` bounds the
same scan's integer comparisons by `4·(n.val - j0.val) + 1`, and `route_step` with
`route_step_pick`/`route_step_refuse` advances the pass counter by exactly one per
continuation, so the rank `k - pass` falls by one per pass. `route_scan` prices each pass at
`n.val` inspections, so the `k`-wide selection's work is the closed form `k·E` inspections in
`k` passes. The single composed trace over all `k` passes is the stated residual in the header.
-/

/-- **One pass costs exactly `E` expert inspections and succeeds.** Under the guards
    `pass < k` and `ok1`, and the premise that the delegated scan returns an index below `n`,
    the extracted body takes one pick pass: it returns `Ok` with a continuation whose pass
    counter is one larger and whose result vector is one entry longer, and the rank
    `k - pass` falls by one. The delegated scan prices that pass at `E = n.val` expert
    inspections (`route_scan`), so a `k`-wide selection over `E` experts costs `k·E` expert
    inspections: the sum of `k` per-pass counts of `E`. -/
theorem route_pass_inspects (logits : Slice Std.I64) (k n : Std.Usize)
    (hsz : logits.val.length = n.val) (hkN : k.val ≤ n.val) (hnmax : n.val ≤ 4294967295)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec Std.U32) (pass : Std.Usize)
    (hchosen : chosen.val.length = n.val) (hsel : sel.val.length = pass.val)
    (hpass : pass.val < k.val)
    (hex : ∃ b, moe.best_unchosen_loop logits (alloc.vec.Vec.deref chosen) n n 0#i64 0#usize =
      ok b ∧ b.val < n.val) :
    ∃ (ch' : alloc.vec.Vec Bool) (sl' : alloc.vec.Vec Std.U32) (pass' : Std.Usize)
      (b : Std.Usize),
      b.val < n.val ∧
      moe.route_topk_loop.body logits k n chosen sel true none pass =
        ok (ControlFlow.cont (ch', sl', true, none, pass')) ∧
      ch'.val.length = n.val ∧ sl'.val.length = pass'.val ∧ pass'.val = pass.val + 1 ∧
      k.val - pass'.val < k.val - pass.val ∧
      moe.best_unchosen_loop logits (alloc.vec.Vec.deref chosen) n n 0#i64 0#usize = ok b ∧
      LoopTrace.Trace (scanBody logits (alloc.vec.Vec.deref chosen) n) (scanInspect n)
        (n, 0#i64, 0#usize) (ok b) n.val := by
  obtain ⟨b, hscan, hb⟩ := hex
  obtain ⟨b0, hscan0, htrace0⟩ := route_scan logits n hsz chosen hchosen
  obtain ⟨ch2, sl2, ps2, b2, hb2, hbody, hset, hsel2, hclen2, hslen2, hps2', hrank, hrank2,
    hslen3⟩ :=
    route_step_pick logits k n hsz hkN hnmax chosen sel none pass hchosen hsel hpass
      ⟨b, hscan, hb⟩
  have hbfwd : b = b0 := Result.ok_injective (hscan.symm.trans hscan0)
  rw [← hbfwd] at htrace0
  exact ⟨ch2, sl2, ps2, b, hb, hbody, hclen2, hslen3, hps2', hrank, hscan, htrace0⟩

/-! ## 7. The refusal branches of `moe.route_topk`, and the reachable initial state -/

/-- `k = 0` refuses with `BadK 0 n` before any loop runs: zero passes, zero expert
    inspections. -/
theorem route_refuse_zero (logits : Slice Std.I64) :
    moe.route_topk logits 0#usize =
      ok (core.result.Result.Err (moe.MoeError.BadK 0#usize (Slice.len logits))) ∧
      routeInspect 0#usize (Slice.len logits)
        (alloc.vec.Vec.new Bool, alloc.vec.Vec.new Std.U32, true, none, 0#usize) = 0 := by
  refine ⟨?_, ?_⟩
  · rw [moe.route_topk, if_pos rfl]
  · simp [routeInspect]

/-- `k > n` refuses with `BadK k n` before any loop runs: zero passes, zero expert
    inspections. -/
theorem route_refuse_wide (logits : Slice Std.I64) (k : Std.Usize)
    (h : Slice.len logits < k) :
    moe.route_topk logits k = ok (core.result.Result.Err (moe.MoeError.BadK k (Slice.len logits)))
      ∧ routeInspect k (Slice.len logits)
        (alloc.vec.Vec.new Bool, alloc.vec.Vec.new Std.U32, true, none, k) = 0 := by
  have hk0 : ¬ (k = 0#usize) := by
    intro hc
    rw [hc] at h
    simp [UScalar.lt_equiv] at h
  refine ⟨?_, ?_⟩
  · rw [moe.route_topk, if_neg hk0, if_pos h]
  · simp [routeInspect]

/-- **The reachable initial state.** The selection loop starts from the all-`false` mask over
    `E` experts, the empty result vector and pass `0`, and that state satisfies the premises
    `route_pass_work` and `route_step_pick` name. -/
theorem route_init_witness (logits : Slice Std.I64) (k : Std.Usize) :
    ∃ (chosen0 : alloc.vec.Vec Bool),
      chosen0.val = List.replicate logits.val.length false ∧
      chosen0.val.length = logits.val.length ∧
      (alloc.vec.Vec.with_capacity Std.U32 k).val.length = 0 ∧
      (0#usize).val = 0 ∧ (0#usize).val ≤ k.val := by
  obtain ⟨chosen0, hfrom, hval, hlen⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneBool false (Slice.len logits) (by rfl))
  have hlen0 : (alloc.vec.Vec.with_capacity Std.U32 k).val.length = 0 := by
    simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]
  exact ⟨chosen0, hval, hlen, hlen0, rfl, Nat.zero_le _⟩

end RouteTopkResource

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

#resource_audit RouteTopkResource.scan_step
#resource_audit RouteTopkResource.scan_exit
#resource_audit RouteTopkResource.scanCmp_le
#resource_audit RouteTopkResource.scan_observed_cont
#resource_audit RouteTopkResource.scan_inspect_closed_form
#resource_audit RouteTopkResource.scan_cmp_bounded
#resource_audit RouteTopkResource.vec_set_length
#resource_audit RouteTopkResource.usize_cast_u32_val
#resource_audit RouteTopkResource.route_scan_exists
#resource_audit RouteTopkResource.route_scan
#resource_audit RouteTopkResource.best_unchosen_of_scan_true
#resource_audit RouteTopkResource.best_unchosen_of_scan_false
#resource_audit RouteTopkResource.route_step
#resource_audit RouteTopkResource.route_step_pick
#resource_audit RouteTopkResource.route_step_refuse
#resource_audit RouteTopkResource.route_exit_width
#resource_audit RouteTopkResource.route_exit_refusal
#resource_audit RouteTopkResource.route_loop_projects
#resource_audit RouteTopkResource.route_pass_inspects
#resource_audit RouteTopkResource.route_refuse_zero
#resource_audit RouteTopkResource.route_refuse_wide
#resource_audit RouteTopkResource.route_init_witness

run_cmd Lean.logInfo "ROUTE_TOPK_KERNEL_OK"
