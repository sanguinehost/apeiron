import Q6KCodesSource
import Aeneas

/-
SCOPE: actual extracted Q6_K six-bit code unpack: the two guarded loop-body transitions, their
strictly decreasing ranks, the closed-form iteration count per 128-weight half and per super-block,
the two loop exit branches, the entry point's length refusal branch, and a reachable success
witness on a concrete 210-byte payload.
REGISTER: scalar source-refinement resource model. Work is counted in loop-body invocations and
array writes under the named premises; it is never cycles, wall-clock or GPU time. The pointwise
code-value refinement is not re-proved here: it is
proof/corpus/Numeric/Packing/Format.lean `word_q6_k_codes_ok`, a separate register, and the
per-code values are not claimed by this file.
CORPUS: src/format/quant.rs `q6_k_codes` and the actual extracted
q6_k_codes_source.quant.q6_k_codes, `.{q6_k_codes_loop0,q6_k_codes_loop0_loop0}.body`,
`quant.{need,Q6_K_BLOCK_BYTES}` and `word_q6_k_codes`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean, the committed extraction of the
same `quant.rs` bodies through `proof/regen.sh --start-from crate::word_q6_k_codes`.
GATE: python3 proof/resource/Q6KCodes_gate.py --controls, whole invocation 540 s on
/tmp/opencode/apeiron-lane-2.lock, systemd-run MemoryMax=12G MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: Lean checked only when that gate succeeds; the work measure is a loop-body invocation and
array-write count under the stated premises, not a machine-time measurement.
TRUST BOUNDARY: excludes rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V and every layer the
extraction does not reach; the cost model is the extracted scalar body's own control flow.
-/

namespace Q6KCodesResource
open Aeneas Aeneas.Std Result

/-- The extracted per-position body of the inner unpack loop. -/
abbrev InnerBody (ql qh : Slice U8) (n qb64 qb32 : Usize) (out : Array I8 256#usize) (l : Usize) :
    Result (ControlFlow ((Array I8 256#usize) × Usize) (Array I8 256#usize)) :=
  q6_k_codes_source.quant.q6_k_codes_loop0_loop0.body ql qh n qb64 qb32 out l

/-- The extracted per-half body of the outer unpack loop. -/
abbrev OuterBody (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize) :
    Result (ControlFlow ((Array I8 256#usize) × Usize) (Array I8 256#usize)) :=
  q6_k_codes_source.quant.q6_k_codes_loop0.body ql qh out n

/-- The extracted inner loop, whose state is the accumulator and the position. -/
abbrev InnerLoop (ql qh : Slice U8) (n qb64 qb32 : Usize)
    (out : Array I8 256#usize) (l : Usize) : Result (Array I8 256#usize) :=
  q6_k_codes_source.quant.q6_k_codes_loop0_loop0 ql qh out n qb64 qb32 l

/-- The extracted outer loop, whose state is the accumulator and the half index. -/
abbrev OuterLoop (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize) :
    Result (Array I8 256#usize) :=
  q6_k_codes_source.quant.q6_k_codes_loop0 ql qh out n

/-- The inner loop as a partial fixpoint on its `(accumulator, position)` state. -/
abbrev InnerFix (ql qh : Slice U8) (n qb64 qb32 : Usize) :
    (Array I8 256#usize × Usize) → Result (Array I8 256#usize) :=
  fun s => InnerLoop ql qh n qb64 qb32 s.1 s.2

/-- The per-position body as a partial function on its `(accumulator, position)` state. -/
abbrev InnerStepFn (ql qh : Slice U8) (n qb64 qb32 : Usize) :
    (Array I8 256#usize × Usize) →
      Result (ControlFlow (Array I8 256#usize × Usize) (Array I8 256#usize)) :=
  fun s => InnerBody ql qh n qb64 qb32 s.1 s.2

/-- The extracted inner loop is the partial fixpoint of its own body on the pair state. -/
theorem inner_fix_eq (ql qh : Slice U8) (n qb64 qb32 : Usize)
    (out : Array I8 256#usize) (l : Usize) :
    InnerLoop ql qh n qb64 qb32 out l = loop (InnerStepFn ql qh n qb64 qb32) (out, l) := by
  unfold InnerLoop InnerStepFn
  rw [q6_k_codes_source.quant.q6_k_codes_loop0_loop0]

/-- The outer loop as a partial fixpoint on its `(accumulator, half)` state. -/
abbrev OuterFix (ql qh : Slice U8) :
    (Array I8 256#usize × Usize) → Result (Array I8 256#usize) :=
  fun s => OuterLoop ql qh s.1 s.2

/-- The per-half body as a partial function on its `(accumulator, half)` state. -/
abbrev OuterStepFn (ql qh : Slice U8) :
    (Array I8 256#usize × Usize) →
      Result (ControlFlow (Array I8 256#usize × Usize) (Array I8 256#usize)) :=
  fun s => OuterBody ql qh s.1 s.2

/-- The extracted per-half loop is the partial fixpoint of its own body on the pair state. -/
theorem outer_fix_eq (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize) :
    OuterLoop ql qh out n = loop (OuterStepFn ql qh) (out, n) := by
  unfold OuterLoop OuterStepFn
  rw [q6_k_codes_source.quant.q6_k_codes_loop0]

/-- The output position of the `c`-th of the four codes one guarded visit writes. -/
def WriteIdx (n l c : Nat) : Nat := 128 * n + l + c

/-- The rank of the per-position loop: the number of code positions of the current half that are
    not yet written. -/
def innerRank (l : Usize) : Nat := 32 - l.val

/-- The list one guarded per-position visit produces: `out` with the four cells at
    `WriteIdx n l` for offsets `0, 32, 64, 96` replaced, in the order the body performs the four
    writes. -/
def FourWrites (out : Array I8 256#usize) (n l : Nat) (w1 w2 w3 w4 : I8) : List I8 :=
  ((((out.val.set (WriteIdx n l 0) w1).set (WriteIdx n l 32) w2).set (WriteIdx n l 64) w3).set (WriteIdx n l 96) w4)

/-- The rank of the per-half loop: the number of 128-weight halves of the super-block that are not
    yet written. -/
def outerRank (n : Usize) : Nat := 2 - n.val

/-- `Q6_K_BLOCK_BYTES` is the 210-byte super-block payload the unpacker reads. -/
theorem block_bytes : q6_k_codes_source.quant.Q6_K_BLOCK_BYTES = ok (210#usize) := by
  have h : q6_k_codes_source.quant.Q6_K_BLOCK_BYTES ⦃ z => z = (210#usize) ⦄ := by
    unfold q6_k_codes_source.quant.Q6_K_BLOCK_BYTES
    simp only [q6_k_codes_source.quant.QK_K]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-! ## (a) Guarded transitions of the actual bodies -/

/-- The `u8` to `i32` widening is the natural inclusion, so a six-bit code stays inside the `i32`
    window after the `-32` offset the body applies. -/
lemma hcast_u8_i32_val (x : U8) : (UScalar.hcast .I32 x).val = (x.val : ℤ) := by
  have hx : (x.val : ℤ) < (2 ^ 32 : ℕ) := by
    have h := x.hBounds
    norm_num [UScalar.size, UScalarTy.numBits] at *
    omega
  have h0 : (0 : ℤ) ≤ (x.val : ℤ) := Int.natCast_nonneg _
  rw [UScalar.hcast_val_eq]
  simp only [IScalarTy.numBits]
  rw [Int.bmod_eq_emod_of_lt (by
    rw [Int.emod_eq_of_lt h0 hx]
    have hb := x.hBounds
    norm_num [UScalar.size, UScalarTy.numBits] at hb
    omega),
    Int.emod_eq_of_lt h0 hx]

lemma i32_window (x : U8) :
    I32.min ≤ (UScalar.hcast .I32 x).val - (32#i32).val ∧
      (UScalar.hcast .I32 x).val - (32#i32).val ≤ I32.max := by
  rw [hcast_u8_i32_val]
  have h0 : (0 : ℤ) ≤ (x.val : ℤ) := Int.natCast_nonneg _
  have h1 : (x.val : ℤ) ≤ 255 := by
    have hb := x.hBounds
    norm_num [UScalar.size, UScalarTy.numBits] at hb
    omega
  constructor <;>
    norm_num [I32.min, IScalar.min, I32.max, IScalar.max, I32.numBits] <;> omega

/-- The `u8` plane read at `64·n + l` is inside a 128-byte low plane. -/
lemma ql_idx_lo (n l L : Nat) (hn : n < 2) (hl : l < 32) (hlen : 128 ≤ L) : 64 * n + l < L := by
  omega

/-- The `u8` plane read at `64·n + l + 32` is inside a 128-byte low plane. -/
lemma ql_idx_hi (n l L : Nat) (hn : n < 2) (hl : l < 32) (hlen : 128 ≤ L) :
    64 * n + l + 32 < L := by omega

/-- The `u8` plane read at `32·n + l` is inside a 64-byte high plane. -/
lemma qh_idx (n l L : Nat) (hn : n < 2) (hl : l < 32) (hlen : 64 ≤ L) : 32 * n + l < L := by
  omega

/-- The `c`-th of the four written output positions is inside the 256-cell array. -/
lemma out_idx (n l c : Nat) (hn : n < 2) (hl : l < 32) (hc : c ≤ 96) : WriteIdx n l c < 256 := by
  unfold WriteIdx
  omega

/-- **The per-position transition.** With the half index `n.val < 2`, the window offsets equal the
    loop's own `64·n`/`32·n`, both planes long enough and the position below `32`, the extracted
    body performs its five checked plane reads and its four checked array writes, returns
    `cont (a, l + 1)`, and the remaining-position rank drops by exactly one. The four written
    positions are `WriteIdx n l` at `c = 0, 32, 64, 96`: strictly increasing, all inside the
    256-cell array. One visit is therefore four array writes, and the loop's write count is four
    times its position count. -/
theorem inner_step (ql qh : Slice U8) (n qb64 qb32 l : Usize) (out : Array I8 256#usize)
    (hn : n.val < 2) (hq64 : qb64.val = 64 * n.val) (hq32 : qb32.val = 32 * n.val)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) (hl : l.val < 32) :
    ∃ a : Array I8 256#usize, ∃ l1 : Usize, ∃ w1 w2 w3 w4 : I8,
      InnerBody ql qh n qb64 qb32 out l = ok (ControlFlow.cont (a, l1)) ∧
      l1.val = l.val + 1 ∧
      a.val = FourWrites out n.val l.val w1 w2 w3 w4 ∧
      innerRank l1 + 1 = innerRank l ∧ innerRank l1 < innerRank l := by
  have hspec : InnerBody ql qh n qb64 qb32 out l
      ⦃ r => ∃ (a : Array I8 256#usize) (l1 : Usize), r = ControlFlow.cont (a, l1) ∧
        l1.val = l.val + 1 ∧
        ∃ w1 w2 w3 w4 : I8, a.val = FourWrites out n.val l.val w1 w2 w3 w4 ⦄ := by
    unfold InnerBody q6_k_codes_source.quant.q6_k_codes_loop0_loop0.body
    rw [if_pos (by scalar_tac : l < 32#usize)]
    step*
    · rw [i8_post]; exact (i32_window i7).1
    · rw [i16_post]; exact (i32_window i15).1
    · rw [i23_post]; exact (i32_window i22).1
    · rw [i31_post]; exact (i32_window i30).1
    · have hi1 : i1.val = 64 * n.val + l.val := by rw [i1_post, hq64]
      have hi9 : i9.val = 64 * n.val + l.val + 32 := by rw [i9_post, hi1]
      have hi24 : i24.val = 64 * n.val + l.val + 32 := by rw [i24_post, hi1]
      have hidx : i.val = 32 * n.val + l.val := by rw [i_post, hq32]
      have hi32 : i32.val = 128 * n.val := by rw [i32_post]
      have hi33 : i33.val = WriteIdx n.val l.val 0 := by rw [i33_post, hi32]; rfl
      have hi36 : i36.val = WriteIdx n.val l.val 32 := by rw [i36_post, i35_post, hi32]; rfl
      have hi39 : i39.val = WriteIdx n.val l.val 64 := by rw [i39_post, i38_post, hi32]; rfl
      have hi42 : i42.val = WriteIdx n.val l.val 96 := by rw [i42_post, i41_post, hi32]; rfl
      refine ⟨a, l1, rfl, l1_post, i34, i37, i40, i43, ?_⟩
      rw [a_post, out3_post, out2_post, out1_post, Array.set_val_eq, Array.set_val_eq,
        Array.set_val_eq, Array.set_val_eq, hi33, hi36, hi39, hi42]
      rfl
  obtain ⟨r, hrun, a, l1, hr, hl1, w1, w2, w3, w4, hset⟩ := WP.spec_imp_exists hspec
  refine ⟨a, l1, w1, w2, w3, w4, ?_, hl1, hset, ?_⟩
  · rw [hrun, hr]
  · simp only [innerRank]
    omega

/-- **The per-position exit branch.** With the position at or past `32` the body returns the
    accumulator unchanged and performs no read, no write and no continuation: zero work. -/
theorem inner_exit (ql qh : Slice U8) (n qb64 qb32 l : Usize) (out : Array I8 256#usize)
    (hl : 32 ≤ l.val) :
    InnerBody ql qh n qb64 qb32 out l = ok (ControlFlow.done out) := by
  have hnot : ¬ (l < 32#usize) := by
    intro hc
    have hc' : l.val < (32#usize).val := (UScalar.lt_equiv l (32#usize)).1 hc
    have h32 : ((32#usize).val : ℤ) = 32 := by norm_num
    omega
  unfold InnerBody q6_k_codes_source.quant.q6_k_codes_loop0_loop0.body
  rw [if_neg hnot]

/-! ## (b) The work measure: a strictly decreasing rank and its closed form -/

/-- A finite certificate of the actual loop body: a finite sequence of observed invocations that
    ends in `done` or in a failure. Divergence has no constructor. -/
inductive Q6Trace {State : Type} {Value : Type}
    (body : State → Result (ControlFlow State Value)) :
    State → Result Value → Type where
  | done {state value} (observed : body state = ok (ControlFlow.done value)) :
      Q6Trace body state (ok value)
  | failure {state error} (observed : body state = fail error) :
      Q6Trace body state (fail error)
  | cont {state next result} (observed : body state = ok (ControlFlow.cont next))
      (tail : Q6Trace body next result) :
      Q6Trace body state result

namespace Q6Trace

variable {State : Type} {Value : Type} {body : State → Result (ControlFlow State Value)}

/-- The number of invocations of the extracted body a certificate records. -/
def visits {state : State} {result : Result Value} (t : Q6Trace body state result) : Nat :=
  match t with
  | Q6Trace.done _ => 1
  | Q6Trace.failure _ => 1
  | Q6Trace.cont _ tail => tail.visits + 1

/-- The number of continuations, that is the invocations that consume one unit of rank. -/
def steps {state : State} {result : Result Value} (t : Q6Trace body state result) : Nat :=
  match t with
  | Q6Trace.done _ => 0
  | Q6Trace.failure _ => 0
  | Q6Trace.cont _ tail => tail.steps + 1

/-- The certificate projects to the actual partial-fixpoint loop, not to an evaluator copy. -/
theorem project {state : State} {result : Result Value} (t : Q6Trace body state result) :
    loop body state = result := by
  induction t with
  | done observed =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
  | failure observed =>
      unfold loop
      rw [observed]
      simp only [bind_fail]
  | cont observed tail ih =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
      exact ih

/-- Every certificate has exactly one terminating visit, so `visits = steps + 1`. -/
theorem count {state : State} {result : Result Value} (t : Q6Trace body state result) :
    t.visits = t.steps + 1 := by
  induction t with
  | done _ => rfl
  | failure _ => rfl
  | cont _ tail ih => simp only [visits, steps]; omega

/-- Two certificates of the same state agree on the outcome and on both counts. -/
theorem deterministic {state : State} {result₁ result₂ : Result Value}
    (first : Q6Trace body state result₁) (second : Q6Trace body state result₂) :
    result₁ = result₂ ∧ first.visits = second.visits ∧ first.steps = second.steps := by
  induction first generalizing result₂ with
  | @done _ v1 observed =>
      cases second with
      | @done _ v2 observed₂ =>
          have hcf := Result.ok_injective (observed.symm.trans observed₂)
          injection hcf with hv
          exact ⟨congrArg ok hv, rfl, rfl⟩
      | failure observed₂ =>
          have h := observed.symm.trans observed₂
          simp at h
      | cont observed₂ tail₂ =>
          have h := observed.symm.trans observed₂
          simp at h
  | @failure _ e1 observed =>
      cases second with
      | done observed₂ =>
          have h := observed.symm.trans observed₂
          simp at h
      | @failure _ e2 observed₂ =>
          exact ⟨congrArg fail (Result.fail_injective (observed.symm.trans observed₂)), rfl, rfl⟩
      | cont observed₂ tail₂ =>
          have h := observed.symm.trans observed₂
          simp at h
  | @cont _ next res observed tail ih =>
      cases second with
      | done observed₂ =>
          have h := observed.symm.trans observed₂
          simp at h
      | failure observed₂ =>
          have h := observed.symm.trans observed₂
          simp at h
      | @cont _ next₂ res₂ observed₂ tail₂ =>
          have eqn : next = next₂ := by simpa using observed.symm.trans observed₂
          cases eqn
          obtain ⟨hr, hv, hs⟩ := ih tail₂
          exact ⟨hr, by simp only [visits]; omega, by simp only [steps]; omega⟩

end Q6Trace

/-- **The per-position closed form, by induction on the rank.** From position `l` the per-position
    loop is invoked `innerRank l + 1` times and consumes `innerRank l` code positions. -/
theorem inner_visits_aux (ql qh : Slice U8) (n qb64 qb32 : Usize) (out : Array I8 256#usize)
    (l : Usize) (k : Nat) (hk : k = innerRank l)
    (hn : n.val < 2) (hq64 : qb64.val = 64 * n.val) (hq32 : qb32.val = 32 * n.val)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    ∃ o : Array I8 256#usize,
      ∃ t : Q6Trace (InnerStepFn ql qh n qb64 qb32) (out, l) (ok o),
        t.visits = k + 1 ∧ t.steps = k := by
  induction k generalizing out l with
  | zero =>
      have hl : 32 ≤ l.val := by simp only [innerRank] at hk; omega
      exact ⟨out, Q6Trace.done (inner_exit ql qh n qb64 qb32 l out hl), rfl, rfl⟩
  | succ k ih =>
      have hl : l.val < 32 := by simp only [innerRank] at *; omega
      obtain ⟨a, l1, w1, w2, w3, w4, hbody, hl1, hset, _⟩ :=
        inner_step ql qh n qb64 qb32 l out hn hq64 hq32 hql hqh hl
      have hk' : k = innerRank l1 := by simp only [innerRank] at *; omega
      obtain ⟨o, htail, hv, hs⟩ := ih a l1 hk'
      refine ⟨o, Q6Trace.cont hbody htail, ?_, ?_⟩
      · simp only [Q6Trace.visits]
        omega
      · simp only [Q6Trace.steps]
        omega

/-- **The per-position closed form.** From any admissible entry state the per-position loop
    terminates and performs exactly `innerRank l` guarded code positions in `innerRank l + 1`
    invocations of its body, each of which writes four cells; every certificate of those counts
    agrees, so the count is the loop's own. -/
theorem inner_visits_exact (ql qh : Slice U8) (n qb64 qb32 : Usize)
    (out : Array I8 256#usize) (l : Usize)
    (hn : n.val < 2) (hq64 : qb64.val = 64 * n.val) (hq32 : qb32.val = 32 * n.val)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    (∃ o : Array I8 256#usize,
      ∃ t : Q6Trace (InnerStepFn ql qh n qb64 qb32) (out, l) (ok o),
        t.visits = innerRank l + 1 ∧ t.steps = innerRank l) ∧
      (∀ (o : Array I8 256#usize) (t : Q6Trace (InnerStepFn ql qh n qb64 qb32) (out, l) (ok o)),
        t.visits = innerRank l + 1 ∧ t.steps = innerRank l) := by
  obtain ⟨o, htrace, hv, hs⟩ := inner_visits_aux ql qh n qb64 qb32 out l (innerRank l) rfl
    hn hq64 hq32 hql hqh
  have huniq : ∀ (o : Array I8 256#usize)
      (t : Q6Trace (InnerStepFn ql qh n qb64 qb32) (out, l) (ok o)),
      t.visits = innerRank l + 1 ∧ t.steps = innerRank l := by
    intro o t
    obtain ⟨hr, hvv, hss⟩ := Q6Trace.deterministic htrace t
    exact ⟨by omega, by omega⟩
  exact ⟨⟨o, htrace, hv, hs⟩, huniq⟩

/-- **The per-position loop terminates.** Under the same premises the extracted per-position loop
    returns `Ok`, so it cannot fail on the index path. -/
theorem inner_loop_ok (ql qh : Slice U8) (n qb64 qb32 : Usize)
    (out : Array I8 256#usize) (l : Usize)
    (hn : n.val < 2) (hq64 : qb64.val = 64 * n.val) (hq32 : qb32.val = 32 * n.val)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    ∃ o : Array I8 256#usize, InnerLoop ql qh n qb64 qb32 out l = ok o := by
  obtain ⟨⟨o, htrace, _, _⟩, _⟩ :=
    inner_visits_exact ql qh n qb64 qb32 out l hn hq64 hq32 hql hqh
  refine ⟨o, ?_⟩
  rw [inner_fix_eq]
  exact Q6Trace.project htrace

/-- **The per-half transition.** With the half index `n.val < 2` the body computes the window
    offsets `64·n` and `32·n`, runs the whole per-position loop once from position `0`, returns
    `cont (out1, n + 1)`, and the remaining-half rank drops by exactly one. The half transition
    therefore costs one inner loop, whose own cost is 32 guarded positions. -/
theorem outer_step (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize) (hn : n.val < 2)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    ∃ out1 : Array I8 256#usize, ∃ n1 : Usize, ∃ qb64 qb32 : Usize,
      OuterBody ql qh out n = ok (ControlFlow.cont (out1, n1)) ∧
      n1.val = n.val + 1 ∧ qb64.val = 64 * n.val ∧ qb32.val = 32 * n.val ∧
      InnerLoop ql qh n qb64 qb32 out 0#usize = ok out1 ∧
      outerRank n1 + 1 = outerRank n ∧ outerRank n1 < outerRank n := by
  have h1 : (64#usize).val = 64 := by norm_num
  have h2 : (32#usize).val = 32 := by norm_num
  have h3 : (1#usize).val = 1 := by norm_num
  obtain ⟨qb64, hqb64, hqb64v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 64#usize) (y := n) (by scalar_tac))
  obtain ⟨qb32, hqb32, hqb32v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 32#usize) (y := n) (by scalar_tac))
  obtain ⟨n1, hn1, hn1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := n) (y := 1#usize) (by scalar_tac))
  have hqv64 : qb64.val = 64 * n.val := by rw [hqb64v, h1]
  have hqv32 : qb32.val = 32 * n.val := by rw [hqb32v, h2]
  have hqv1 : n1.val = n.val + 1 := by rw [hn1v, h3]
  obtain ⟨o, ho⟩ := inner_loop_ok ql qh n qb64 qb32 out 0#usize hn hqv64 hqv32 hql hqh
  refine ⟨o, n1, qb64, qb32, ?_, hqv1, hqv64, hqv32, ho, ?_⟩
  · unfold OuterBody q6_k_codes_source.quant.q6_k_codes_loop0.body
    rw [if_pos (by scalar_tac : n < 2#usize)]
    rw [hqb64, hqb32, hn1]
    simp only [ho, bind_tc_ok]
  · simp only [outerRank]
    omega

/-- **The per-half exit branch.** With the half index at or past `2` the body returns the
    accumulator unchanged, runs no per-position loop and writes nothing: zero work. -/
theorem outer_exit (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize) (hn : 2 ≤ n.val) :
    OuterBody ql qh out n = ok (ControlFlow.done out) := by
  have hnot : ¬ (n < 2#usize) := by
    intro hc
    have hc' : n.val < (2#usize).val := (UScalar.lt_equiv n (2#usize)).1 hc
    have h2 : ((2#usize).val : ℤ) = 2 := by norm_num
    omega
  unfold OuterBody q6_k_codes_source.quant.q6_k_codes_loop0.body
  rw [if_neg hnot]

/-- **The per-half closed form, by induction on the rank.** From half `n` the per-half loop is
    invoked `outerRank n + 1` times and consumes `outerRank n` 128-weight halves, each of which
    runs the per-position loop once from position `0`. -/
theorem outer_visits_aux (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize) (k : Nat)
    (hk : k = outerRank n) (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    ∃ o : Array I8 256#usize,
      ∃ t : Q6Trace (OuterStepFn ql qh) (out, n) (ok o), t.visits = k + 1 ∧ t.steps = k := by
  induction k generalizing out n with
  | zero =>
      have hn : 2 ≤ n.val := by simp only [outerRank] at hk; omega
      exact ⟨out, Q6Trace.done (outer_exit ql qh out n hn), rfl, rfl⟩
  | succ k ih =>
      have hn : n.val < 2 := by simp only [outerRank] at *; omega
      obtain ⟨out1, n1, qb64, qb32, hbody, hn1, h64, h32, hinner, _⟩ :=
        outer_step ql qh out n hn hql hqh
      have hk' : k = outerRank n1 := by simp only [outerRank] at *; omega
      obtain ⟨o, htail, hv, hs⟩ := ih out1 n1 hk'
      refine ⟨o, Q6Trace.cont hbody htail, ?_, ?_⟩
      · simp only [Q6Trace.visits]
        omega
      · simp only [Q6Trace.steps]
        omega

/-- **The per-half closed form.** From any accumulator the per-half loop terminates and performs
    exactly `outerRank n` halves in `outerRank n + 1` invocations of its body; every certificate
    of those counts agrees. -/
theorem outer_visits_exact (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    (∃ o : Array I8 256#usize,
      ∃ t : Q6Trace (OuterStepFn ql qh) (out, n) (ok o),
        t.visits = outerRank n + 1 ∧ t.steps = outerRank n) ∧
      (∀ (o : Array I8 256#usize) (t : Q6Trace (OuterStepFn ql qh) (out, n) (ok o)),
        t.visits = outerRank n + 1 ∧ t.steps = outerRank n) := by
  obtain ⟨o, htrace, hv, hs⟩ := outer_visits_aux ql qh out n (outerRank n) rfl hql hqh
  have huniq : ∀ (o : Array I8 256#usize) (t : Q6Trace (OuterStepFn ql qh) (out, n) (ok o)),
      t.visits = outerRank n + 1 ∧ t.steps = outerRank n := by
    intro o t
    obtain ⟨hr, hvv, hss⟩ := Q6Trace.deterministic htrace t
    exact ⟨by omega, by omega⟩
  exact ⟨⟨o, htrace, hv, hs⟩, huniq⟩

/-- **The entry closed form.** From the production entry state of the per-half loop the body is
    invoked three times and consumes two halves; every certificate of those counts agrees, so
    `3` invocations and `2` halves are the loop's own. -/
theorem entry_visits_exact (ql qh : Slice U8) (out : Array I8 256#usize)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    (∃ o : Array I8 256#usize,
      ∃ t : Q6Trace (OuterStepFn ql qh) (out, 0#usize) (ok o), t.visits = 3 ∧ t.steps = 2) ∧
      (∀ (o : Array I8 256#usize) (t : Q6Trace (OuterStepFn ql qh) (out, 0#usize) (ok o)),
        t.visits = 3 ∧ t.steps = 2) := by
  have hzero : outerRank 0#usize = 2 := by simp only [outerRank]; rfl
  obtain ⟨⟨o, htrace, hv, hs⟩, _⟩ := outer_visits_exact ql qh out 0#usize hql hqh
  rw [hzero] at hv hs
  refine ⟨⟨o, htrace, by omega, by omega⟩, ?_⟩
  intro o' t
  obtain ⟨hv', hs'⟩ := (outer_visits_exact ql qh out 0#usize hql hqh).2 o' t
  rw [hzero] at hv' hs'
  exact ⟨by omega, by omega⟩

/-- **The closed-form work of one super-block.** Two 128-weight halves, 32 code positions per
    half, four array writes per position: `2 · 32 = 64` guarded code positions and
    `2 · 32 · 4 = 256` write operations on the 256-cell output array, spread over `3` per-half body
    invocations each of which runs the per-position loop once in `33` invocations. The counts are
    derived from the two loops' own ranks; they are not cycles, wall-clock or GPU time. -/
theorem codes_work_closed_form :
    outerRank 0#usize = 2 ∧ innerRank 0#usize = 32 ∧ outerRank 0#usize * innerRank 0#usize = 64 ∧
      outerRank 0#usize * innerRank 0#usize * 4 = 256 ∧
      (outerRank 0#usize + 1) * (innerRank 0#usize + 1) = 3 * 33 ∧
      (outerRank 0#usize + 1) + outerRank 0#usize * (innerRank 0#usize + 1) = 69 := by
  simp only [outerRank, innerRank]
  norm_num

/-! ## (c) The entry point: success branch, refusal branch, work and a reachable witness -/

/-- **The per-half loop terminates.** Under the same premises the extracted per-half loop returns
    `Ok`, so it cannot fail on the index path. -/
theorem outer_loop_ok (ql qh : Slice U8) (out : Array I8 256#usize) (n : Usize)
    (hql : 128 ≤ ql.val.length) (hqh : 64 ≤ qh.val.length) :
    ∃ o : Array I8 256#usize, OuterLoop ql qh out n = ok o := by
  obtain ⟨⟨o, htrace, _, _⟩, _⟩ := outer_visits_exact ql qh out n hql hqh
  refine ⟨o, ?_⟩
  rw [outer_fix_eq]
  exact Q6Trace.project htrace

/-- The 128-byte low-nibble plane the entry point builds out of a payload: the `0..128` slice. -/
def Ql (blk : Slice U8) : Slice U8 :=
  .from (blk.val.slice 0 128) (by simp)

/-- The 64-byte high-bit plane the entry point builds out of a payload: the `128..192` slice. -/
def Qh (blk : Slice U8) : Slice U8 :=
  .from (blk.val.slice 128 192) (by
    simp only [List.slice, List.length_take, List.length_drop]
    exact Nat.le_trans (Nat.min_le_left 64 _) (by scalar_tac))

/-- The 256-cell accumulator the entry point starts from. -/
def Init : Array I8 256#usize := Array.repeat 256#usize 0#i8

/-- **The length refusal branch, exactly.** A payload shorter than `Q6_K_BLOCK_BYTES = 210` refuses
    with `Truncated 210 (Slice.len blk)`, carrying the required and the available length; the
    unpacker is never entered, so the work is zero. -/
theorem codes_refuses (blk : Slice U8) (h : Slice.len blk < 210#usize) :
    q6_k_codes_source.word_q6_k_codes blk =
      ok (core.result.Result.Err
        (q6_k_codes_source.quant.QuantError.Truncated 210#usize (Slice.len blk))) := by
  rw [q6_k_codes_source.word_q6_k_codes]
  unfold q6_k_codes_source.quant.q6_k_codes
  rw [block_bytes]
  simp only [bind_tc_ok, q6_k_codes_source.quant.need, Slice.len]
  rw [if_pos h]
  simp [core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]

/-- **The success branch, exactly.** On a payload of at least `Q6_K_BLOCK_BYTES = 210` bytes the
    extracted entry point returns `Ok` of a 256-cell array: the length guard passes, the two planes
    are built, the per-half loop runs to completion and its result is returned. -/
theorem codes_success (blk : Slice U8) (hlen : 210#usize ≤ Slice.len blk) :
    ∃ v : Array I8 256#usize,
      q6_k_codes_source.word_q6_k_codes blk = ok (core.result.Result.Ok v) ∧ v.val.length = 256 := by
  have hspec : q6_k_codes_source.word_q6_k_codes blk
      ⦃ r => ∃ (v : Array I8 256#usize), r = core.result.Result.Ok v ∧ v.val.length = 256 ⦄ := by
    rw [q6_k_codes_source.word_q6_k_codes]
    unfold q6_k_codes_source.quant.q6_k_codes
    rw [block_bytes]
    simp only [bind_tc_ok, q6_k_codes_source.quant.need, Slice.len]
    rw [if_neg (by scalar_tac : ¬ Slice.len blk < 210#usize)]
    simp only [core.result.Result.Insts.CoreOpsTry.branch]
    step*
    · rw [i1_post]
      simp only [q6_k_codes_source.quant.QK_K]
      scalar_tac
    · rw [i3_post, i1_post, i2_post]
      simp only [q6_k_codes_source.quant.QK_K]
      scalar_tac
    · refine WP.spec_bind
        (Pₘ := fun o : Array I8 256#usize => ∃ v : Array I8 256#usize,
          (core.result.Result.Ok o : core.result.Result (Array I8 256#usize)
            q6_k_codes_source.quant.QuantError) = core.result.Result.Ok v ∧
            v.val.length = 256) ?_ ?_
      · have hblk : 210 ≤ blk.val.length := by
          calc 210 = (210#usize).val := by norm_num
            _ ≤ (Slice.len blk).val := by exact_mod_cast hlen
            _ = blk.val.length := Slice.len_val blk
        have hi1 : i1.val = 128 := by
          rw [i1_post]
          simp only [q6_k_codes_source.quant.QK_K]
          norm_num
        have hi2 : i2.val = 64 := by
          rw [i2_post]
          simp only [q6_k_codes_source.quant.QK_K]
          norm_num
        have hi3 : i3.val = 192 := by rw [i3_post, hi1, hi2]
        have hql' : 128 ≤ ql.val.length := by
          rw [ql_post, hi1]
          simp only [List.slice, List.length_take, List.length_drop]
          rw [Nat.min_eq_left (by omega)]
        have hqh' : 64 ≤ qh.val.length := by
          rw [qh_post, hi3, hi1]
          simp only [List.slice, List.length_take, List.length_drop]
          rw [Nat.min_eq_left (by omega)]
        obtain ⟨o, ho⟩ := outer_loop_ok ql qh Init 0#usize hql' hqh'
        have ho' : q6_k_codes_source.quant.q6_k_codes_loop0 ql qh (Array.repeat 256#usize 0#i8)
            0#usize = ok o := ho
        rw [ho']
        exact ⟨⟨o, rfl, o.property⟩⟩
      · intro out1 _
        exact ⟨out1, rfl, out1.property⟩
  obtain ⟨r, hr, v, hv, hlen'⟩ := WP.spec_imp_exists hspec
  exact ⟨v, hr.trans (congrArg ok hv), hlen'⟩

/-- **The work of the entry point's own loops.** On a payload long enough for a whole super-block
    the two planes the entry point builds have the lengths the loops assume, and the per-half loop
    on that plane pair performs exactly two halves in three body invocations. Every certificate of
    those counts agrees, so the counts are the loops' own. -/
theorem entry_work (blk : Slice U8) (hblk : 210 ≤ blk.val.length) :
    128 ≤ (Ql blk).val.length ∧ 64 ≤ (Qh blk).val.length ∧
      (∃ v : Array I8 256#usize,
        ∃ t : Q6Trace (OuterStepFn (Ql blk) (Qh blk)) (Init, 0#usize) (ok v),
          t.visits = 3 ∧ t.steps = 2) ∧
      (∀ (v : Array I8 256#usize)
          (t : Q6Trace (OuterStepFn (Ql blk) (Qh blk)) (Init, 0#usize) (ok v)),
        t.visits = 3 ∧ t.steps = 2) := by
  have hql : 128 ≤ (Ql blk).val.length := by
    have heq : (Ql blk).val = List.slice 0 128 blk.val := Slice.from_val _ _
    rw [heq]
    simp only [List.slice, List.length_take, List.length_drop]
    rw [Nat.min_eq_left (by omega)]
  have hqh : 64 ≤ (Qh blk).val.length := by
    have heq : (Qh blk).val = List.slice 128 192 blk.val := Slice.from_val _ _
    rw [heq]
    simp only [List.slice, List.length_take, List.length_drop]
    rw [Nat.min_eq_left (by omega)]
  obtain ⟨⟨v, htrace, hv, hs⟩, huniq⟩ := entry_visits_exact (Ql blk) (Qh blk) Init hql hqh
  exact ⟨hql, hqh, ⟨v, htrace, hv, hs⟩, huniq⟩

/-- The reachable witness payload: 210 zero bytes, one whole Q6_K super-block. -/
def witBlk : Slice U8 := Slice.from (List.replicate 210 0#u8) (by
  rw [List.length_replicate]
  scalar_tac)

/-- The witness payload carries `Q6_K_BLOCK_BYTES` bytes, so the length guard passes. -/
theorem wit_len : 210#usize ≤ Slice.len witBlk := by
  have h : (Slice.len witBlk).val = 210 := by
    rw [Slice.len_val]
    simp only [Slice.length]
    have heq : witBlk.val = List.replicate 210 0#u8 := Slice.from_val _ _
    rw [heq, List.length_replicate]
  have : (210#usize).val ≤ (Slice.len witBlk).val := by
    rw [show (210#usize).val = 210 from rfl, h]
  exact this

/-- **A reachable success witness with its exact work.** On the concrete 210-byte zero payload the
    extracted entry point takes the success branch and returns a 256-cell array, and the planes it
    builds carry exactly the closed-form work: two halves, `2 · 32 = 64` guarded code positions and
    `256` array writes, in three per-half body invocations. The success branch is reachable, not
    vacuous: no refusal branch is taken and the loops do run. -/
theorem wit_success :
    ∃ (v w : Array I8 256#usize),
      q6_k_codes_source.word_q6_k_codes witBlk = ok (core.result.Result.Ok v) ∧
      v.val.length = 256 ∧
      128 ≤ (Ql witBlk).val.length ∧ 64 ≤ (Qh witBlk).val.length ∧
      (∃ t : Q6Trace (OuterStepFn (Ql witBlk) (Qh witBlk)) (Init, 0#usize) (ok w),
        t.visits = 3 ∧ t.steps = 2) ∧
      (∀ t : Q6Trace (OuterStepFn (Ql witBlk) (Qh witBlk)) (Init, 0#usize) (ok w),
        t.visits = 3 ∧ t.steps = 2) := by
  obtain ⟨v, hok, hlen⟩ := codes_success witBlk wit_len
  have hblk : 210 ≤ witBlk.val.length := by
    have heq : witBlk.val = List.replicate 210 0#u8 := Slice.from_val _ _
    rw [heq, List.length_replicate]
  obtain ⟨hql, hqh, ⟨w, htrace, hv, hs⟩, huniq⟩ := entry_work witBlk hblk
  exact ⟨v, w, hok, hlen, hql, hqh, ⟨htrace, hv, hs⟩, fun t => huniq w t⟩

end Q6KCodesResource

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

#resource_audit Q6KCodesResource.block_bytes
#resource_audit Q6KCodesResource.inner_step
#resource_audit Q6KCodesResource.inner_exit
#resource_audit Q6KCodesResource.inner_visits_aux
#resource_audit Q6KCodesResource.inner_visits_exact
#resource_audit Q6KCodesResource.inner_loop_ok
#resource_audit Q6KCodesResource.outer_step
#resource_audit Q6KCodesResource.outer_exit
#resource_audit Q6KCodesResource.outer_visits_aux
#resource_audit Q6KCodesResource.outer_visits_exact
#resource_audit Q6KCodesResource.outer_loop_ok
#resource_audit Q6KCodesResource.entry_visits_exact
#resource_audit Q6KCodesResource.codes_work_closed_form
#resource_audit Q6KCodesResource.codes_refuses
#resource_audit Q6KCodesResource.codes_success
#resource_audit Q6KCodesResource.entry_work
#resource_audit Q6KCodesResource.wit_len
#resource_audit Q6KCodesResource.wit_success

run_cmd Lean.logInfo "Q6K_CODES_KERNEL_OK"