import ArgmaxSource
import Aeneas

/-
SCOPE: actual extracted argmax scan-loop transition under its guard, the per-element work
  measure, the two exit branches (empty input and index-range refusal), and reachable witnesses.
  Files: proof/resource/ArgmaxSource.lean (Aeneas output), ArgmaxResource.lean (this file) and
  Argmax_gate.py (lane-3 extraction/mutation gate). No other file is touched.
REGISTER: local scalar cost-model obligation on the extracted `sampler.argmax_fixed` body:
  transition, work measure under named premises, exit branches, witnesses. Excluded: argmax
  correctness, the canonical R[x]/(x^2 - sigma) correspondence, cycles and GPU time. Work is
  counted in `i64` comparisons and index steps, never in cycles.
CORPUS: src/engine/sampler.rs 41:0-54:1 (`argmax_fixed`), scan loop 47:4-52:5, narrowing
  53:32-53:63; matches the committed proof/ApeironWord.lean extraction modulo the crate name.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean (`apeiron_word::word_argmax_fixed`,
  `sampler::argmax_fixed_loop`, `sampler::argmax_fixed_loop.body`). No canonical-corpus theorem is
  claimed or refined here.
GATE: python3 proof/resource/Argmax_gate.py --check --controls, cwd repo root, on
  /tmp/opencode/apeiron-lane-3.lock, MemoryMax=12G MemorySwapMax=0, LEAN_NUM_THREADS=1,
  540 s whole-invocation budget.
GRADE: `proved` for every theorem below iff that gate prints ARGMAX_KERNEL_OK; the extracted body
  comes from the real source by `#[path]` with only a typed forwarding wrapper added. Nothing here
  is measured: no cycle count, no wall-clock, no GPU number.
-/

namespace ArgmaxResource
open Aeneas Aeneas.Std Result

/-! ## Platform bounds

The model's `Usize.max` is `2 ^ System.Platform.numBits - 1` with `numBits` either 32 or 64, so
`Usize.max` is `U32.max` or `U64.max`. Both cases are discharged here; the concrete slice witnesses
below are built only from the resulting bounds. -/

theorem usize_max_u32 (h32 : System.Platform.numBits = 32) : Usize.max = U32.max := by
  rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq, h32, U32.max_eq]
  norm_num

theorem usize_max_u64 (h64 : System.Platform.numBits = 64) : Usize.max = U64.max := by
  rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq, h64, U64.max_eq]
  norm_num

theorem usize_ge_two : 2 ≤ Usize.max := by
  rcases System.Platform.numBits_eq with h32 | h64
  · rw [usize_max_u32 h32, U32.max_eq]
    norm_num
  · rw [usize_max_u64 h64, U64.max_eq]
    norm_num

theorem usize_ge_u32max2 (h64 : System.Platform.numBits = 64) : U32.max + 2 ≤ Usize.max := by
  rw [usize_max_u64 h64, U64.max_eq, U32.max_eq]
  norm_num

theorem usize_zero_val : (0#usize : Usize).val = 0 := by simp

theorem usize_one_val : (1#usize : Usize).val = 1 := by simp

/-- The real `is_empty` call returns `true` exactly on a zero-length slice: the model's own
    length test, read through the value the call produced. -/
theorem is_empty_spec_value (logits : Slice I64) (c : Bool)
    (hc : core.slice.Slice.is_empty logits = ok c) :
    (c = true) ↔ logits.length = 0 := by
  obtain ⟨c1, hrun, hspec⟩ := WP.spec_imp_exists (core.slice.Slice.is_empty_spec logits)
  rw [hc] at hrun
  rw [← Result.ok_injective hrun] at hspec
  exact iff_of_eq hspec

theorem is_empty_true (logits : Slice I64) (h : logits.length = 0) (c : Bool)
    (hc : core.slice.Slice.is_empty logits = ok c) : c = true :=
  (is_empty_spec_value logits c hc).mpr h

theorem is_empty_not_true (logits : Slice I64) (hne : 0 < logits.length) (c : Bool)
    (hc : core.slice.Slice.is_empty logits = ok c) : ¬ (c = true) := by
  intro hct
  have hz : logits.length = 0 := (is_empty_spec_value logits c hc).mp hct
  omega

/-! ## The work measure -/

/-- The scan's work measure, in elements: how many logit elements remain to be compared after the
    scan index `i`. One guarded iteration spends exactly one unit and performs exactly one `i64`
    comparison. -/
def rank (logits : Slice I64) (i : Usize) : Nat :=
  logits.length - i.val

/-- The joint invariant of the production scan: `best` is a real index into the input, it never
    runs ahead of the scan index, and the scan index lies in `[1, logits.length]`. -/
def loopInv (logits : Slice I64) (best i : Usize) : Prop :=
  best.val < logits.length ∧ best.val ≤ i.val ∧ 1 ≤ i.val ∧ i.val ≤ logits.length

/-- The joint invariant is a conjunction of the four reachable-state conditions, so one witness for
    those four conditions is a witness for the invariant at the joint type. -/
theorem loopInv_intro (logits : Slice I64) (best i : Usize)
    (hbest : best.val < logits.length) (hble : best.val ≤ i.val)
    (hlo : 1 ≤ i.val) (hhi : i.val ≤ logits.length) : loopInv logits best i :=
  ⟨hbest, hble, hlo, hhi⟩

/-! ## The guarded transition -/

/-- **Guarded transition.** Under the guard `i < logits.length` the extracted body reads
    `logits[i]` and `logits[best]`, performs exactly one `>` comparison on those two `i64` values,
    advances the scan index by exactly one, and spends exactly one unit of `rank`. The comparison
    outcome is reported as the body's own conditional update `if xi > xb then i else best`, so the
    two cases of the theorem are the body's two cases. -/
theorem body_step (logits : Slice I64) (best i : Usize)
    (hbest : best.val < logits.length) (hguard : i.val < logits.length) :
    ∃ best' : Usize, ∃ i' : Usize, ∃ xi : I64, ∃ xb : I64,
      argmax_source.sampler.argmax_fixed_loop.body logits best i =
        ok (ControlFlow.cont (best', i')) ∧
      xi = logits.val[i.val] ∧ xb = logits.val[best.val] ∧
      best' = (if xi > xb then i else best) ∧
      i'.val = i.val + 1 ∧ rank logits i' + 1 = rank logits i := by
  obtain ⟨xi, hxi, hxi'⟩ :=
    WP.spec_imp_exists (Slice.index_usize_spec logits i hguard)
  obtain ⟨xb, hxb, hxb'⟩ :=
    WP.spec_imp_exists (Slice.index_usize_spec logits best hbest)
  have hlen : logits.length ≤ Usize.max := by
    show logits.val.length ≤ Usize.max
    exact logits.property
  have hone : (1#usize : Usize).val = 1 := usize_one_val
  have hadd : i.val + (1#usize).val ≤ Usize.max := by omega
  obtain ⟨i', haddrun, hi'⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hadd)
  have hi'v : i'.val = i.val + 1 := by simpa using hi'
  have hrank : rank logits i' + 1 = rank logits i := by
    simp only [rank, hi'v]
    omega
  have hlt : i < Slice.len logits := by
    refine (UScalar.lt_equiv i (Slice.len logits)).2 ?_
    simpa only [Slice.len_val] using hguard
  by_cases hc : (xi > xb)
  · have hbody : ∃ best1 : Usize,
        argmax_source.sampler.argmax_fixed_loop.body logits best i =
          ok (ControlFlow.cont (best1, i')) ∧ best1 = i := by
      unfold argmax_source.sampler.argmax_fixed_loop.body
      rw [if_pos hlt, hxi, hxb]
      simp only [bind_tc_ok]
      rw [if_pos hc, haddrun]
      simp only [bind_tc_ok]
      exact ⟨_, rfl, rfl⟩
    obtain ⟨best1, hrun, hbestval⟩ := hbody
    refine ⟨best1, i', xi, xb, hrun, hxi', hxb', ?_, hi'v, hrank⟩
    rw [hbestval, if_pos hc]
  · have hbody : ∃ best1 : Usize,
        argmax_source.sampler.argmax_fixed_loop.body logits best i =
          ok (ControlFlow.cont (best1, i')) ∧ best1 = best := by
      unfold argmax_source.sampler.argmax_fixed_loop.body
      rw [if_pos hlt, hxi, hxb]
      simp only [bind_tc_ok]
      rw [if_neg hc, haddrun]
      simp only [bind_tc_ok]
      exact ⟨_, rfl, rfl⟩
    obtain ⟨best1, hrun, hbestval⟩ := hbody
    refine ⟨best1, i', xi, xb, hrun, hxi', hxb', ?_, hi'v, hrank⟩
    rw [hbestval, if_neg hc]

/-- **Exit branch of the scan.** Under the joint invariant, when the guard is false the extracted
    body performs no comparison and returns `done best` unchanged; the guard being false together
    with the invariant forces the scan index to sit at the input length, so `rank` is exhausted. -/
theorem body_exit (logits : Slice I64) (best i : Usize)
    (hinv : loopInv logits best i) (hnlt : ¬ i.val < logits.length) :
    argmax_source.sampler.argmax_fixed_loop.body logits best i = ok (ControlFlow.done best) ∧
      i.val = logits.length ∧ rank logits i = 0 := by
  have hnlt' : ¬ i < Slice.len logits := by
    intro hc
    apply hnlt
    simpa only [Slice.len_val] using (UScalar.lt_equiv i (Slice.len logits)).1 hc
  have hrun : argmax_source.sampler.argmax_fixed_loop.body logits best i =
      ok (ControlFlow.done best) := by
    unfold argmax_source.sampler.argmax_fixed_loop.body
    rw [if_neg hnlt']
  simp only [loopInv] at hinv
  refine ⟨hrun, ?_, ?_⟩
  · omega
  · simp only [rank]
    omega

/-- **Initialization from the actual constructor.** The scan is entered at
    `(best, i) = (0#usize, 1#usize)`, which satisfies the joint invariant for every non-empty
    input, and its starting rank is exactly the number of elements after the first. -/
theorem initial_witness (logits : Slice I64) (hne : 0 < logits.length) :
    loopInv logits 0#usize 1#usize ∧ rank logits (1#usize) = logits.length - 1 ∧
      0 < logits.length := by
  have hlen : logits.length ≤ Usize.max := by
    show logits.val.length ≤ Usize.max
    exact logits.property
  have h0 : (0#usize : Usize).val = 0 := usize_zero_val
  have h1 : (1#usize : Usize).val = 1 := usize_one_val
  have hinv : loopInv logits 0#usize 1#usize := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · omega
    · omega
    · omega
    · omega
  refine ⟨hinv, ?_, hne⟩
  simp only [rank]
  norm_num

/-- **The work measure over the whole scan.** Every reachable state satisfies the joint invariant,
    so the scan is bounded by `logits.length - 1` continuations, each spending exactly one `rank`
    unit on exactly one `i64` comparison, and the returned index is a witness that only elements of
    the input were examined. The measure counts comparisons and index steps, not cycles. -/
theorem loop_work (logits : Slice I64) (hne : 0 < logits.length) :
    argmax_source.sampler.argmax_fixed_loop logits 0#usize 1#usize
      ⦃ best => best.val < logits.length ∧ best.val ≤ logits.length - 1 ⦄ := by
  have hlen : logits.length ≤ Usize.max := by
    show logits.val.length ≤ Usize.max
    exact logits.property
  unfold argmax_source.sampler.argmax_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun st : Usize × Usize => rank logits st.2)
    (inv := fun st : Usize × Usize => loopInv logits st.1 st.2)
    (post := fun best : Usize => best.val < logits.length ∧ best.val ≤ logits.length - 1)
  · intro st hst
    rcases st with ⟨best, i⟩
    dsimp only at hst ⊢
    simp only [loopInv] at hst
    by_cases hlt : i.val < logits.length
    · obtain ⟨best', i', xi, xb, hbody, _, _, hbestval, hi'v, _⟩ :=
        body_step logits best i hst.1 hlt
      have hinv' : loopInv logits best' i' := by
        by_cases hc : (xi > xb)
        · rw [hbestval, if_pos hc]
          refine loopInv_intro logits i i' ?_ ?_ ?_ ?_ <;> omega
        · rw [hbestval, if_neg hc]
          refine loopInv_intro logits best i' ?_ ?_ ?_ ?_ <;> omega
      have hrank' : rank logits i' < rank logits i := by
        simp only [rank, hi'v]
        omega
      rw [hbody]
      exact ⟨hinv', hrank'⟩
    · obtain ⟨hrun, _, _⟩ := body_exit logits best i hst hlt
      rw [hrun]
      exact ⟨hst.1, by omega⟩
  · show loopInv logits 0#usize 1#usize
    exact (initial_witness logits hne).1

/-- The scan's work budget in closed form over the input length: one comparison per element after
    the first, i.e. `logits.length - 1` rank units. Together with `body_step` (one unit and one
    comparison per guarded iteration) and `body_exit` (rank `0` at every exit) this is the exact
    comparison count of the extracted scan. -/
theorem work_budget (logits : Slice I64) (hne : 0 < logits.length) :
    rank logits (1#usize) = logits.length - 1 := (initial_witness logits hne).2.1

/-! ## The extracted root -/

/-- The extracted root forwards to the real body: `word_argmax_fixed` adds no work and no result
    transformation of its own. -/
theorem root_forwards (logits : Slice I64)
    (r : core.result.Result U32 argmax_source.sampler.SamplerError)
    (h : argmax_source.sampler.argmax_fixed logits = ok r) :
    argmax_source.word_argmax_fixed logits = ok r := by
  unfold argmax_source.word_argmax_fixed
  rw [h]

/-! ## Exit branch: empty input -/

/-- **Empty-input exit branch.** An empty input takes the guard's exit and refuses with
    `EmptyLogits` without entering the scan. -/
theorem empty_refusal (logits : Slice I64) (h : logits.length = 0) :
    argmax_source.sampler.argmax_fixed logits =
      ok (core.result.Result.Err argmax_source.sampler.SamplerError.EmptyLogits) := by
  obtain ⟨b, hb, hb'⟩ :=
    WP.spec_imp_exists (core.slice.Slice.is_empty_spec logits)
  unfold argmax_source.sampler.argmax_fixed
  obtain ⟨b, hb, _⟩ :=
    WP.spec_imp_exists (core.slice.Slice.is_empty_spec logits)
  rw [hb]
  simp only [bind_tc_ok, if_pos (is_empty_true logits h b hb)]

/-! ## The two remaining branches of the real body -/

/-- **Success branch.** For a non-empty input whose scan result fits `u32`, the real narrowing step
    returns it unchanged and the real `map_err` leaves `Ok` untouched. -/
theorem success_branch (logits : Slice I64) (best : Usize) (hne : 0 < logits.length)
    (hloop : argmax_source.sampler.argmax_fixed_loop logits 0#usize 1#usize = ok best)
    (hsmall : best.val ≤ U32.max) :
    ∃ u : U32, argmax_source.sampler.argmax_fixed logits = ok (core.result.Result.Ok u) ∧
      u.val = best.val := by
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
    (core.convert.num.ptr_try_from_impls.TryFromU32Usize.try_from.step_spec best)
  unfold argmax_source.sampler.argmax_fixed
  obtain ⟨c, hc, _⟩ :=
    WP.spec_imp_exists (core.slice.Slice.is_empty_spec logits)
  have hc'' : ¬ (c = true) := is_empty_not_true logits hne c hc
  rw [hc]
  simp only [bind_tc_ok, if_neg hc'']
  rw [hloop]
  simp only [bind_tc_ok]
  rw [hr]
  obtain ⟨u, hur, huv⟩ := hpost.1 hsmall
  simp only [bind_tc_ok]
  rw [hur, core.result.Result.map_err_ok]
  exact ⟨u, rfl, huv⟩

/-- **Index-range refusal.** For a non-empty input whose scan result exceeds `u32::MAX`, the real
    `u32::try_from` narrowing refuses and the real `map_err` closure produces `LogitsTooLong`.
    This is the branch statement under a named premise; it is not a claim that some particular
    input takes the branch, which needs the argmax-correctness theorem owned by
    `Numeric.PrunedArgmax` and `Numeric.EngineSamplerRefinement`. -/
theorem too_long_branch (logits : Slice I64) (best : Usize) (hne : 0 < logits.length)
    (hloop : argmax_source.sampler.argmax_fixed_loop logits 0#usize 1#usize = ok best)
    (hbig : U32.max < best.val) :
    argmax_source.sampler.argmax_fixed logits =
      ok (core.result.Result.Err argmax_source.sampler.SamplerError.LogitsTooLong) := by
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
    (core.convert.num.ptr_try_from_impls.TryFromU32Usize.try_from.step_spec best)
  have hr' : r = core.result.Result.Err () := hpost.2 hbig
  unfold argmax_source.sampler.argmax_fixed
  obtain ⟨c, hc, _⟩ :=
    WP.spec_imp_exists (core.slice.Slice.is_empty_spec logits)
  have hc'' : ¬ (c = true) := is_empty_not_true logits hne c hc
  rw [hc]
  simp only [bind_tc_ok, if_neg hc'']
  rw [hloop]
  simp only [bind_tc_ok]
  rw [hr, hr']
  simp only [bind_tc_ok]
  rw [core.result.Result.map_err_err]
  simp only [argmax_source.sampler.argmax_fixed.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorSamplerError.call_once,
    bind_tc_ok]

/-- A non-empty input reaches exactly one of the two remaining branches, and the choice is the
    width of the scan result. This is the branch decomposition of the real body. -/
theorem branch_split (logits : Slice I64) (hne : 0 < logits.length) :
    (∃ b : Usize, argmax_source.sampler.argmax_fixed_loop logits 0#usize 1#usize = ok b ∧
       b.val < logits.length ∧ b.val ≤ U32.max ∧
       ∃ u : U32, argmax_source.sampler.argmax_fixed logits = ok (core.result.Result.Ok u) ∧
         u.val = b.val) ∨
    (∃ b : Usize, argmax_source.sampler.argmax_fixed_loop logits 0#usize 1#usize = ok b ∧
       b.val < logits.length ∧ U32.max < b.val ∧
       argmax_source.sampler.argmax_fixed logits =
         ok (core.result.Result.Err argmax_source.sampler.SamplerError.LogitsTooLong)) := by
  obtain ⟨b, hloop, hbnd⟩ := WP.spec_imp_exists (loop_work logits hne)
  by_cases hsmall : b.val ≤ U32.max
  · obtain ⟨u, hok, huv⟩ := success_branch logits b hne hloop hsmall
    exact Or.inl ⟨b, hloop, hbnd.1, hsmall, u, hok, huv⟩
  · have hbig : U32.max < b.val := by omega
    obtain hok := too_long_branch logits b hne hloop hbig
    exact Or.inr ⟨b, hloop, hbnd.1, hbig, hok⟩

/-- A non-empty input never returns `EmptyLogits`: the branch decomposition leaves only the
    success result and the index-range refusal. -/
theorem nonempty_not_empty (logits : Slice I64) (hne : 0 < logits.length) :
    argmax_source.sampler.argmax_fixed logits ≠
      ok (core.result.Result.Err argmax_source.sampler.SamplerError.EmptyLogits) := by
  intro h
  rcases branch_split logits hne with ⟨b, _, _, _, u, hok, _⟩ | ⟨b, _, _, _, hok⟩
  · rw [hok] at h
    exact absurd h (by simp)
  · rw [hok] at h
    exact absurd (Result.ok_injective h) (by simp)

/-- The `LogitsTooLong` refusal requires an input longer than `u32::MAX`, because the scan's result
    is always a real index into the input. This bounds the domain of the refusal premise. -/
theorem too_long_needs_long_input (logits : Slice I64)
    (h : argmax_source.sampler.argmax_fixed logits =
      ok (core.result.Result.Err argmax_source.sampler.SamplerError.LogitsTooLong)) :
    U32.max < logits.length := by
  by_cases hne : 0 < logits.length
  · rcases branch_split logits hne with ⟨b, _, _, _, u, hok, _⟩ | ⟨b, _, hbnd, hbig, hok⟩
    · rw [hok] at h
      exact absurd h (by simp)
    · rw [hok] at h
      omega
  · have hz : logits.length = 0 := by omega
    rw [empty_refusal logits hz] at h
    exact absurd (Result.ok_injective h) (by simp)

/-- Consequently the index-range refusal is unreachable on a 32-bit `usize` platform, where the
    scan result always fits `u32`. -/
theorem too_long_needs_64bit (logits : Slice I64)
    (h : argmax_source.sampler.argmax_fixed logits =
      ok (core.result.Result.Err argmax_source.sampler.SamplerError.LogitsTooLong)) :
    System.Platform.numBits = 64 := by
  have hbig : U32.max < logits.length := too_long_needs_long_input logits h
  have hlen : logits.length ≤ Usize.max := by
    show logits.val.length ≤ Usize.max
    exact logits.property
  rcases System.Platform.numBits_eq with h32 | h64
  · rw [usize_max_u32 h32] at hlen
    exact (Nat.not_lt_of_ge hlen hbig).elim
  · exact h64

/-- On a 64-bit platform the domain of the refusal premise is satisfiable: an input longer than
    `u32::MAX` exists. This makes `too_long_needs_long_input` non-vacuous as a statement about
    `logits.length`; it does not claim that any such input actually refuses. -/
theorem long_input_witness (h64 : System.Platform.numBits = 64) :
    ∃ logits : Slice I64, U32.max < logits.length := by
  have hrep : (List.replicate (U32.max + 2) (0#i64 : I64)).length = U32.max + 2 := by
    simp only [List.length_replicate]
  have hrep'' : (List.replicate (U32.max + 2) (0#i64 : I64)).length ≤ Usize.max := by
    rw [hrep]
    exact usize_ge_u32max2 h64
  refine ⟨Slice.from (List.replicate (U32.max + 2) (0#i64 : I64)) hrep'', ?_⟩
  have h : (Slice.from (List.replicate (U32.max + 2) (0#i64 : I64)) hrep'').length
      = (List.replicate (U32.max + 2) (0#i64 : I64)).length := by
    simp only [Slice.length, Slice.from_val]
  rw [h, hrep, U32.max_eq]
  omega

/-! ## Reachable witnesses -/

theorem two_logits_bound : [0#i64, 1#i64].length ≤ Usize.max := by
  have h : [0#i64, 1#i64].length = 2 := by simp
  rw [h]
  exact usize_ge_two

/-- A concrete two-element Q32.32 logit vector, built by `Slice.from` under a checked bound. -/
def twoLogits : Slice I64 := Slice.from [0#i64, 1#i64] two_logits_bound

theorem twoLogits_length : twoLogits.length = 2 := by simp [twoLogits]

theorem twoLogits_val : twoLogits.val = [0#i64, 1#i64] := by simp [twoLogits]

/-- On that concrete input the one reachable guarded iteration reads both elements, takes the
    `>` branch of the comparison `1#i64 > 0#i64`, and advances the scan index to `2`: exactly one
    comparison and one rank unit. -/
theorem twoLogits_step :
    argmax_source.sampler.argmax_fixed_loop.body twoLogits 0#usize 1#usize =
      ok (ControlFlow.cont (1#usize, 2#usize)) := by
  have hone : (1#usize : Usize).val = 1 := usize_one_val
  have hone' : (0#usize : Usize).val = 0 := usize_zero_val
  have hlt : (1#usize : Usize) < Slice.len twoLogits := by
    refine (UScalar.lt_equiv (1#usize) (Slice.len twoLogits)).2 ?_
    rw [Slice.len_val, hone, twoLogits_length]
    norm_num
  have hxi : Slice.index_usize twoLogits (1#usize) = ok (1#i64 : I64) := by
    simp [Slice.index_usize, twoLogits]
  have hxb : Slice.index_usize twoLogits (0#usize) = ok (0#i64 : I64) := by
    simp [Slice.index_usize, twoLogits]
  have hgt : (1#i64 : I64) > 0#i64 := by decide
  have hadd : (1#usize : Usize).val + (1#usize).val ≤ Usize.max := by
    rw [hone]
    exact usize_ge_two
  obtain ⟨i', haddrun, hi'⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := 1#usize) (y := 1#usize) hadd)
  have hi' : i' = 2#usize := UScalar.eq_imp _ _ (by simpa using hi')
  unfold argmax_source.sampler.argmax_fixed_loop.body
  rw [if_pos hlt, hxi, hxb]
  simp only [bind_tc_ok]
  rw [if_pos hgt, haddrun, hi']
  simp only [bind_tc_ok]

/-- The same input at its exit state `(1, 2)`: the guard `2 < 2` is false, so the body performs no
    comparison and returns the current index. -/
theorem twoLogits_exit :
    argmax_source.sampler.argmax_fixed_loop.body twoLogits 1#usize 2#usize =
      ok (ControlFlow.done 1#usize) := by
  have h1 : (1#usize : Usize).val = 1 := usize_one_val
  have h2 : (2#usize : Usize).val = 2 := by simp
  have hinv : loopInv twoLogits 1#usize 2#usize := by
    refine loopInv_intro twoLogits 1#usize 2#usize ?_ ?_ ?_ ?_
    · rw [h1, twoLogits_length]
      omega
    · rw [h1, h2]
      omega
    · rw [h2]
      norm_num
    · rw [h2, twoLogits_length]
  have hnlt : ¬ (2#usize : Usize).val < twoLogits.length := by
    rw [h2, twoLogits_length]
    omega
  exact (body_exit twoLogits 1#usize 2#usize hinv hnlt).1

/-- The extracted scan on that concrete input performs exactly one comparison and returns index
    `1`: it is entered at `(0, 1)`, continues once to `(1, 2)`, and exits there. -/
theorem twoLogits_loop_spec :
    argmax_source.sampler.argmax_fixed_loop twoLogits 0#usize 1#usize
      ⦃ b => b.val = 1 ⦄ := by
  unfold argmax_source.sampler.argmax_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun st : Usize × Usize => 2 - st.2.val)
    (inv := fun st : Usize × Usize =>
      (st.1 = 0#usize ∧ st.2 = 1#usize) ∨ (st.1 = 1#usize ∧ st.2 = 2#usize))
    (post := fun b : Usize => b.val = 1)
  · intro st hst
    rcases hst with ⟨hb0, hi1⟩ | ⟨hb1, hi2⟩
    · have hst' : st = (0#usize, 1#usize) := Prod.ext hb0 hi1
      rw [hst']
      dsimp only at ⊢
      rw [twoLogits_step]
      refine ⟨Or.inr ⟨rfl, rfl⟩, ?_⟩
      norm_num
    · have hst' : st = (1#usize, 2#usize) := Prod.ext hb1 hi2
      rw [hst']
      dsimp only at ⊢
      rw [twoLogits_exit]
      exact ⟨by simp⟩
  · exact Or.inl ⟨rfl, rfl⟩

/-- The extracted scan on that concrete input performs exactly one comparison and returns index
    `1`: it is entered at `(0, 1)`, continues once to `(1, 2)`, and exits there. -/
theorem twoLogits_loop :
    argmax_source.sampler.argmax_fixed_loop twoLogits 0#usize 1#usize = ok 1#usize := by
  obtain ⟨b, hrun, hb⟩ := WP.spec_imp_exists twoLogits_loop_spec
  have hb' : b = 1#usize := UScalar.eq_imp _ _ hb
  rw [hrun, hb']

/-- **Reachable success witness at a non-zero index.** The concrete two-element input reaches the
    guarded continuation and returns `Ok 1` through the extracted root `word_argmax_fixed`. -/
theorem success_witness :
    argmax_source.word_argmax_fixed twoLogits = ok (core.result.Result.Ok 1#u32) := by
  have hne : 0 < twoLogits.length := by
    simpa only [twoLogits_length] using (show (0 : Nat) < 2 by omega)
  have hsmall : (1#usize : Usize).val ≤ U32.max := by scalar_tac
  obtain ⟨u, hok, huv⟩ := success_branch twoLogits 1#usize hne twoLogits_loop hsmall
  have hu : u = 1#u32 := UScalar.eq_imp _ _ huv
  exact root_forwards twoLogits (core.result.Result.Ok 1#u32) (by rw [hok, hu])

theorem one_logit_bound : [7#i64].length ≤ Usize.max := by
  have h : [7#i64].length = 1 := by simp
  rw [h]
  rcases System.Platform.numBits_eq with h32 | h64
  · rw [usize_max_u32 h32, U32.max_eq]
    norm_num
  · rw [usize_max_u64 h64, U64.max_eq]
    norm_num

/-- A concrete one-element Q32.32 logit vector. -/
def oneLogit : Slice I64 := Slice.from [7#i64] one_logit_bound

theorem oneLogit_length : oneLogit.length = 1 := by simp [oneLogit]

/-- On that input the scan is entered at `(0, 1)` and exits immediately because `1` equals the
    input length: zero comparisons, zero rank units spent after the entry state. -/
theorem oneLogit_work_spec :
    argmax_source.sampler.argmax_fixed_loop oneLogit 0#usize 1#usize
      ⦃ b => b.val = 0 ⦄ := by
  unfold argmax_source.sampler.argmax_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun _ : Usize × Usize => 0)
    (inv := fun st : Usize × Usize => st.1 = 0#usize ∧ st.2 = 1#usize)
    (post := fun b : Usize => b.val = 0)
  · intro st hst
    rcases hst with ⟨hb0, hi1⟩
    have hst' : st = (0#usize, 1#usize) := Prod.ext hb0 hi1
    rw [hst']
    dsimp only at ⊢
    have hinv : loopInv oneLogit 0#usize 1#usize := by
      refine loopInv_intro oneLogit 0#usize 1#usize ?_ ?_ ?_ ?_
      · rw [usize_zero_val, oneLogit_length]
        omega
      · rw [usize_zero_val, usize_one_val]
        omega
      · rw [usize_one_val]
      · rw [usize_one_val, oneLogit_length]
    have hnlt : ¬ (1#usize : Usize).val < oneLogit.length := by
      rw [usize_one_val, oneLogit_length]
      omega
    obtain ⟨hrun, _, _⟩ := body_exit oneLogit 0#usize 1#usize hinv hnlt
    rw [hrun]
    exact ⟨by simp⟩
  · exact ⟨rfl, rfl⟩

/-- The same input at the level of a plain equation: the extracted scan returns index `0`. -/
theorem oneLogit_work :
    argmax_source.sampler.argmax_fixed_loop oneLogit 0#usize 1#usize = ok 0#usize := by
  obtain ⟨b, hrun, hb⟩ := WP.spec_imp_exists oneLogit_work_spec
  have hb' : b = 0#usize := UScalar.eq_imp _ _ hb
  rw [hrun, hb']

/-- **Reachable success witness at index `0` on a one-element input**, through the extracted root. -/
theorem success_witness_zero :
    argmax_source.word_argmax_fixed oneLogit = ok (core.result.Result.Ok 0#u32) := by
  have hne : 0 < oneLogit.length := by
    simpa only [oneLogit_length] using (show (0 : Nat) < 1 by omega)
  have hsmall : (0#usize : Usize).val ≤ U32.max := by scalar_tac
  obtain ⟨u, hok, huv⟩ := success_branch oneLogit 0#usize hne oneLogit_work hsmall
  have hu : u = 0#u32 := UScalar.eq_imp _ _ huv
  exact root_forwards oneLogit (core.result.Result.Ok 0#u32) (by rw [hok, hu])

/-- **Empty-input refusal, stated on the extracted root.** -/
theorem empty_refusal_root :
    argmax_source.word_argmax_fixed
      (Slice.from [] (Nat.zero_le _))
      = ok (core.result.Result.Err argmax_source.sampler.SamplerError.EmptyLogits) := by
  refine root_forwards _ _ (empty_refusal _ (by simp))

end ArgmaxResource

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

#resource_audit ArgmaxResource.usize_max_u32
#resource_audit ArgmaxResource.usize_max_u64
#resource_audit ArgmaxResource.usize_ge_two
#resource_audit ArgmaxResource.usize_ge_u32max2
#resource_audit ArgmaxResource.usize_zero_val
#resource_audit ArgmaxResource.usize_one_val
#resource_audit ArgmaxResource.is_empty_spec_value
#resource_audit ArgmaxResource.is_empty_true
#resource_audit ArgmaxResource.is_empty_not_true
#resource_audit ArgmaxResource.body_step
#resource_audit ArgmaxResource.body_exit
#resource_audit ArgmaxResource.loopInv_intro
#resource_audit ArgmaxResource.initial_witness
#resource_audit ArgmaxResource.loop_work
#resource_audit ArgmaxResource.work_budget
#resource_audit ArgmaxResource.root_forwards
#resource_audit ArgmaxResource.empty_refusal
#resource_audit ArgmaxResource.success_branch
#resource_audit ArgmaxResource.too_long_branch
#resource_audit ArgmaxResource.branch_split
#resource_audit ArgmaxResource.nonempty_not_empty
#resource_audit ArgmaxResource.too_long_needs_long_input
#resource_audit ArgmaxResource.too_long_needs_64bit
#resource_audit ArgmaxResource.long_input_witness
#resource_audit ArgmaxResource.twoLogits_length
#resource_audit ArgmaxResource.twoLogits_val
#resource_audit ArgmaxResource.twoLogits_step
#resource_audit ArgmaxResource.twoLogits_exit
#resource_audit ArgmaxResource.twoLogits_loop_spec
#resource_audit ArgmaxResource.twoLogits_loop
#resource_audit ArgmaxResource.success_witness
#resource_audit ArgmaxResource.oneLogit_length
#resource_audit ArgmaxResource.oneLogit_work_spec
#resource_audit ArgmaxResource.oneLogit_work
#resource_audit ArgmaxResource.success_witness_zero
#resource_audit ArgmaxResource.empty_refusal_root

run_cmd Lean.logInfo "ARGMAX_KERNEL_OK"
