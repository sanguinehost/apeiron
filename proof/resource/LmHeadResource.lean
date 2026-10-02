import LmHeadSource
import Aeneas

/-
SCOPE: the actual extracted LM-head body `lm_head.lm_head_fixed`: its two refusal guards, its
delegation to the real projection `matvec.matvec_fixed`, the row-loop transitions with both guards,
every row-loop refusal branch as an exact equality, the one row visit *conditional on the row's dot
computation returning `ok`* (`row_body_visit_of_dot_ok`, whose two other arms enumerate the
dequantizer's own `vis`/`div` propagation), and the entry-rank measures of both loops. The
unconditional row-body-success claim and the executed row-visit count are **retired and not
established** by this file (§6); the conditional theorem proves neither.
REGISTER: scalar source-refinement resource model; no hardware timing, no cycle counts, no
canonical-ring correspondence, no claim about the dequantizer bodies, the unextracted read adapter
(`head_payload_len`, `Engine::lm_head_fixed`) or the stored-file path.
CORPUS: src/engine/lm_head.rs `lm_head_fixed` and the actual extracted
`lm_head_source.lm_head.lm_head_fixed`; the composed real projection
`lm_head_source.matvec.matvec_fixed`, `lm_head_source.matvec.matvec_fixed_loop`,
`lm_head_source.matvec.matvec_fixed_loop.body`, and the per-row
`lm_head_source.matvec.row_dot_fixed`, `lm_head_source.matvec.row_dot_fixed_loop{,.body}`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean (the committed shared extraction
unit does not compile the current tree, so this gate never compiles it and extracts through its own
narrow forwarding `#[path]` wrapper) and /home/socol/Workspace/apeiron/proof/native/LoopTrace.lean
for the finite-observation certificate re-derived here as `VisitTrace`.
GATE: python3 proof/resource/LmHead_gate.py --controls, whole invocation 540 s under
/tmp/opencode/apeiron-lane-1.lock with MemoryMax=12G MemorySwapMax=0 and LEAN_NUM_THREADS=1.
GRADE: Lean checked only when that gate succeeds. The work measure is a count of guarded loop
visits, not a machine-time or cycle measurement.
-/

namespace LmHeadResource
open Aeneas Aeneas.Std Result lm_head_source
universe u v
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! ## 0 · The work measure, named as a count

`rowVisits n_out o` counts the vocabulary rows the production row loop has still to visit and is its
strictly decreasing natural rank; `dotVisits w` counts the weights one row dot reduces over. No
measure here is a cycle count. -/

/-- Vocabulary rows the row loop has still to visit. -/
def rowVisits (n_out o : Usize) : Nat := n_out.val - o.val

/-- Weight positions one row dot reduces over, counted from the weight-row length. -/
def dotVisits (n : Nat) : Nat := n

/-- The `Vec` alphabet the projection writes one Q32.32 logit per row. -/
abbrev Logits := alloc.vec.Vec I64

/-- The row-loop state: output logits, row scratch buffer, row index, failure flag, refusal. -/
abbrev RowState := Logits × Logits × Usize × Bool × Option matvec.MatvecError

/-- The dot-loop state: `i128` accumulator, position, failure flag. -/
abbrev DotState := I128 × Usize × Bool

/-- The value the row loop yields. -/
abbrev RowValue := Logits × Option matvec.MatvecError

/-! ## 1 · Model facts about the primitives the two loops use -/

/-- `Slice.len` reads the same length the indexed reads and the loop guards use. -/
theorem sliceLenVal {α : Type} (s : Slice α) : (Slice.len s).val = s.val.length := by
  rw [Slice.len_val]

/-- `Slice.len` is the length the guards compare against. -/
theorem sliceLen_eq {α : Type} (s : Slice α) (n : Usize) (h : s.val.length = n.val) :
    Slice.len s = n := by
  apply UScalar.eq_of_val_eq
  simp [h]

/-- The back direction of `sliceLen_eq`. -/
theorem sliceLen_of_eq {α : Type} (s : Slice α) (n : Usize) (h : Slice.len s = n) :
    s.val.length = n.val :=
  (sliceLenVal s).symm.trans (congrArg (fun z => z.val) h)

/-- The `usize` disequality the LM-head width guard is decided by. -/
theorem usize_bne_false_iff {a b : Usize} (h : (a != b) = false) : a = b := by
  cases hb : (a == b) with
  | true => exact LawfulBEq.eq_of_beq hb
  | false =>
      have hc : (a != b) = true := by
        rw [bne, hb]
        rfl
      exact absurd hc (by rw [h]; exact Bool.false_ne_true)

/-- A positive `usize` disequality is a genuine inequality. -/
theorem usize_ne_of_bne_true {a b : Usize} (h : (a != b) = true) : a ≠ b := by
  intro he
  rw [he] at h
  exact absurd h (by simp [bne])

/-- The `usize` disequality the LM-head width guard is decided by, positively. -/
theorem usize_bne_true_iff {a b : Usize} : (a != b) = true ↔ a ≠ b := by
  constructor
  · exact usize_ne_of_bne_true
  · intro h
    cases hb : (a == b) with
    | true => exact absurd (LawfulBEq.eq_of_beq hb) h
    | false => rw [bne, hb]; rfl

/-- The width guard the LM-head body evaluates. -/
theorem headWidthGuard (x : Slice I64) (n : Usize) (h : x.val.length = n.val) :
    (Slice.len x != n) = false := by
  have hlen : Slice.len x = n := sliceLen_eq x n h
  rw [hlen]
  simp

/-- `1#usize` reads as one. -/
theorem one_usize_val : (1#usize).val = 1 := by norm_num

/-- The `usize` add the row loop performs on the row index and on the row end. -/
theorem usize_add_val (a b : Usize) (h : a.val + b.val ≤ Usize.max) :
    ∃ z : Usize, a + b = ok z ∧ z.val = a.val + b.val := by
  have hspec := Usize.add_spec (x := a) (y := b) h
  rcases (WP.spec_equiv_exists _ _).1 hspec with ⟨z, hz, hzv⟩
  exact ⟨z, hz, hzv⟩

/-- The `Vec.push` the successful row performs appends, on an output length that stays in range. -/
theorem vec_push_val (v : Logits) (x : I64) (h : v.val.length + 1 ≤ Usize.max) :
    ∃ v' : Logits, v.push x = ok v' ∧ v'.val = v.val ++ [x] := by
  have hlt : v.val.length < Usize.max := by omega
  obtain ⟨v', hspec, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec v x hlt)
  exact ⟨v', hspec, hpushv⟩

/-- `Vec.clear` empties the row scratch buffer, so the dequantizer of each row starts at `0`. -/
theorem vec_clear_empty (v : Logits) :
    ∃ v' : Logits, alloc.vec.Vec.clear Global v = ok v' ∧ v'.val = [] := by
  refine ⟨{ v with slice := Slice.new I64 }, rfl, ?_⟩
  simp only [alloc.vec.Vec.val, Slice.new_val]

/-- The row-slice read the row loop performs, as the model's range getter. -/
theorem sliceGetRange_eq {T : Type} (r : core.ops.range.Range Usize) (s : Slice T) :
    core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice T) s r
      = core.slice.index.SliceIndexRangeUsizeSlice.get r s := rfl

/-- A successful range read yields a row slice. The model's range getter is either `ok (some row)`
    or `ok none`, so excluding the latter exhibits the row the row loop proceeds on. -/
theorem rowRead_some {T : Type} (r : core.ops.range.Range Usize) (s : Slice T)
    (h : core.slice.index.SliceIndexRangeUsizeSlice.get r s ≠ ok none) :
    ∃ row : Slice T, core.slice.index.SliceIndexRangeUsizeSlice.get r s = ok (some row) := by
  unfold core.slice.index.SliceIndexRangeUsizeSlice.get at h
  by_cases hc : r.start ≤ r.«end» ∧ r.«end» ≤ s.length
  · rw [if_pos hc] at h
    refine ⟨Slice.from (List.slice r.start.val r.«end».val s.val) (by grind), ?_⟩
    show (if r.start ≤ r.«end» ∧ r.«end» ≤ s.length then
      ok (some (Slice.from (List.slice r.start.val r.«end».val s.val) (by grind))) else ok none)
      = ok (some (Slice.from (List.slice r.start.val r.«end».val s.val) (by grind)))
    rw [if_pos hc]
  · rw [if_neg hc] at h
    exact absurd h (by simp)

/-- The `usize` row stride the row loop multiplies, when it does not overflow `usize`. -/
theorem usize_checked_mul_some (o row_bytes : Usize) (h : o.val * row_bytes.val ≤ Usize.max) :
    ∃ start : Usize, Usize.checked_mul o row_bytes = some start ∧
      start.val = o.val * row_bytes.val := by
  have hspec := Usize.checked_mul_bv_spec o row_bytes
  cases hc : Usize.checked_mul o row_bytes with
  | none =>
      rw [hc] at hspec
      exact absurd h (by omega)
  | some start =>
      rw [hc] at hspec
      obtain ⟨hle, hval, _⟩ := hspec
      exact ⟨start, rfl, hval⟩

/-- A `Result.vis` is never an `ok`. -/
theorem vis_ne_ok {α : Type} (i : RustEffect.Input) (k : RustEffect.Output i → Result α) (a : α) :
    (Result.vis i k : Result α) ≠ Result.ok a := by
  intro h
  have hc := congrArg Result.match h
  simp only [Result.match.vis, Result.match.ok] at hc
  simp at hc

/-- `Result.div` is never an `ok`. -/
theorem div_ne_ok {α : Type} (a : α) : (Result.div : Result α) ≠ Result.ok a := by
  intro h
  have hc := congrArg Result.match h
  simp only [Result.match.div, Result.match.ok] at hc
  simp at hc

/-- A destructuring bind on a visible effect or `div` cannot reach an `ok`: both are distinct
    constructors from the `ret` the row loop needs. -/
theorem pairBind_ne_ok {α β : Type} {i : RustEffect.Input}
    {k : RustEffect.Output i → Result (α × β)} {y : Result (α × β)} {z : β}
    (g : α × β → Result β) (hx : y = Result.vis i k) : ¬ (Std.bind y g = ok z) := by
  rw [hx, bind_vis]
  exact fun h => vis_ne_ok i (fun w => Std.bind (k w) g) z h

/-- The same for `div`. -/
theorem pairBind_div_ne_ok {α β : Type} {y : Result (α × β)} {z : β}
    (g : α × β → Result β) (hx : y = Result.div) : ¬ (Std.bind y g = ok z) := by
  rw [hx, bind_div]
  exact div_ne_ok z

/-- A visible-effect or `div` `Result` cannot equal an `ok`; the row loop's `Vec.push`, row dot
    and dequantizer reach their success arms only through the `ok` constructor. -/
theorem vis_ne_ok' {α : Type} {i : RustEffect.Input} {k : RustEffect.Output i → Result α}
    {x : Result α} (z : α) (hx : x = Result.vis i k) : x ≠ ok z := by
  rw [hx]
  exact vis_ne_ok i k z

theorem div_ne_ok' {α : Type} {x : Result α} (z : α) (hx : x = Result.div) : x ≠ ok z := by
  rw [hx]
  exact div_ne_ok z

/-! ## 2 · A finite-observation certificate for the two actual loops

`VisitTrace` observes each actual body invocation and charges the visit it represents; divergence
has no constructor. It projects to the actual `Std.Primitives.loop`, not to an evaluator copy, and
its count is unique, so a trace fixes the exact visit count of the executing loop. -/

/-- One observed body invocation, charged its visit count; divergence has no constructor. -/
inductive VisitTrace {State : Type u} {Value : Type v}
    (body : State → Result (ControlFlow State Value)) (charge : State → Nat) :
    State → Result Value → Nat → Prop where
  | done {state value} (observed : body state = ok (ControlFlow.done value)) :
      VisitTrace body charge state (ok value) (charge state)
  | failure {state error} (observed : body state = fail error) :
      VisitTrace body charge state (fail error) (charge state)
  | cont {state next result count}
      (observed : body state = ok (ControlFlow.cont next))
      (tail : VisitTrace body charge next result count) :
      VisitTrace body charge state result (charge state + count)

/-- The certificate projects to the actual partial-fixpoint loop. -/
theorem VisitTrace.projection {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {charge : State → Nat}
    {state : State} {result : Result Value} {count : Nat}
    (trace : VisitTrace body charge state result count) : loop body state = result := by
  induction trace with
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

/-- Fixed body and charge observations determine both the outcome and the total visit count. -/
theorem VisitTrace.deterministic {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {charge : State → Nat}
    {state : State} {result₁ result₂ : Result Value} {count₁ count₂ : Nat}
    (first : VisitTrace body charge state result₁ count₁)
    (second : VisitTrace body charge state result₂ count₂) :
    result₁ = result₂ ∧ count₁ = count₂ := by
  induction first generalizing result₂ count₂ with
  | done observed =>
      cases second <;> simp_all
  | failure observed =>
      cases second <;> simp_all
  | @cont state next result count observed tail ih =>
      cases second with
      | done observed₂ => simp_all
      | failure observed₂ => simp_all
      | @cont state₂ next₂ result₂ count₂ observed₂ tail₂ =>
          have eq : next = next₂ := by
            simpa using observed.symm.trans observed₂
          cases eq
          obtain ⟨hr, hc⟩ := ih tail₂
          exact ⟨hr, by omega⟩

/-- A finite certificate cannot project to divergence: the observed loop terminates. -/
theorem VisitTrace.excludes_divergence {State : Type u} {Value : Type v}
    {body : State → Result (ControlFlow State Value)} {charge : State → Nat}
    {state : State} {result : Result Value} {count : Nat}
    (trace : VisitTrace body charge state result count) : result ≠ div := by
  induction trace with
  | done _ => exact ok_not_div
  | failure _ => exact fail_not_div
  | cont _ _ ih => exact ih

/-- The row-loop body, as a function of the row-loop state. -/
def rowBody (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) : RowState → Result (ControlFlow RowState RowValue) :=
  fun s => matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
    s.1 s.2.1 s.2.2.1 s.2.2.2.1 s.2.2.2.2

/-- The row-visit charge: `1` for a visit that can still run, `0` for the terminal visit. -/
def rowCharge (n_out : Usize) (s : RowState) : Nat :=
  if s.2.2.2.1 && s.2.2.1.val < n_out.val then 1 else 0

/-- The dot-loop body, as a function of the dot-loop state. -/
def dotBody (w_row x : Slice I64) : DotState → Result (ControlFlow DotState (I128 × Bool)) :=
  fun s => matvec.row_dot_fixed_loop.body w_row x s.1 s.2.1 s.2.2

/-- The dot-visit charge: `1` for a visit that can still run, `0` for the terminal visit. -/
def dotCharge (w_row : Slice I64) (s : DotState) : Nat :=
  if s.2.2 && s.2.1.val < (Slice.len w_row).val then 1 else 0

/-! ## 3 · The LM-head guards: the two refusals and the delegation -/

/-- The part of the actual body after the width guard: the quant-family test and the delegation.
    It is the `if is_none o = true then … else …` tail of `lm_head.lm_head_fixed`, with the extracted
    quant-geometry lookup already applied to `o`. -/
def headTail (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (o : Option (Usize × Usize)) : Result (core.result.Result Logits lm_head.LmHeadError) :=
  if core.option.Option.is_none o = true then
    ok (core.result.Result.Err (lm_head.LmHeadError.UnsupportedQuant ggml_type))
  else do
    let r ← matvec.matvec_fixed ggml_type raw vocab n_embd x
    core.result.Result.map_err
      P.Insts.CoreOpsFunctionFnOnceTupleMatvecErrorLmHeadError r
      lm_head.LmHeadError.Matvec.constructor

/-- The refusal constructor the head wraps each projection refusal in is the identity on it. -/
theorem matvec_ctor_call (e : matvec.MatvecError) :
    P.Insts.CoreOpsFunctionFnOnceTupleMatvecErrorLmHeadError.call_once
      lm_head.LmHeadError.Matvec.constructor e = ok (lm_head.LmHeadError.Matvec e) := rfl

/-- **The quant-family guard, in the tail.** No quant geometry refuses the head. -/
theorem headTail_none (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64) :
    headTail ggml_type raw vocab n_embd x none =
      ok (core.result.Result.Err (lm_head.LmHeadError.UnsupportedQuant ggml_type)) := by
  simp [headTail, core.option.Option.is_none]

/-- **The delegation, success arm, in the tail.** -/
theorem headTail_some_ok (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (g : Usize × Usize) (v : Logits)
    (hv : matvec.matvec_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Ok v)) :
    headTail ggml_type raw vocab n_embd x (some g) = ok (core.result.Result.Ok v) := by
  show (do
    let r ← matvec.matvec_fixed ggml_type raw vocab n_embd x
    core.result.Result.map_err
      P.Insts.CoreOpsFunctionFnOnceTupleMatvecErrorLmHeadError r
      lm_head.LmHeadError.Matvec.constructor) = _
  rw [hv]
  simp only [bind_tc_ok, core.result.Result.map_err_ok]

/-- **The delegation, refusal arm, in the tail.** Every projection refusal reaches the caller as
    `LmHeadError.Matvec`, carrying the exact refusal. -/
theorem headTail_some_err (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (g : Usize × Usize) (e : matvec.MatvecError)
    (hv : matvec.matvec_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Err e)) :
    headTail ggml_type raw vocab n_embd x (some g) =
      ok (core.result.Result.Err (lm_head.LmHeadError.Matvec e)) := by
  show (do
    let r ← matvec.matvec_fixed ggml_type raw vocab n_embd x
    core.result.Result.map_err
      P.Insts.CoreOpsFunctionFnOnceTupleMatvecErrorLmHeadError r
      lm_head.LmHeadError.Matvec.constructor) = _
  rw [hv]
  simp only [bind_tc_ok, core.result.Result.map_err_err, matvec_ctor_call]

/-- **The width guard.** When the activation length differs from the residual width the actual body
    refuses with `ShapeMismatch` and performs no projection. -/
theorem head_shape_refusal (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (h : x.val.length ≠ n_embd.val) :
    lm_head.lm_head_fixed ggml_type raw vocab n_embd x =
      ok (core.result.Result.Err (lm_head.LmHeadError.ShapeMismatch (Slice.len x) n_embd)) := by
  have hne : (Slice.len x != n_embd) = true :=
    usize_bne_true_iff.mpr (fun hc => h (sliceLen_of_eq x n_embd hc))
  unfold lm_head.lm_head_fixed
  rw [if_pos hne]

/-- **The quant-family guard.** When the quant family has no fixed-point reader the actual body
    refuses with `UnsupportedQuant` and performs no projection. -/
theorem head_quant_refusal (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (hx : x.val.length = n_embd.val) (hq : quant.block_geom ggml_type = ok none) :
    lm_head.lm_head_fixed ggml_type raw vocab n_embd x =
      ok (core.result.Result.Err (lm_head.LmHeadError.UnsupportedQuant ggml_type)) := by
  have hne : (Slice.len x != n_embd) = false := headWidthGuard x n_embd hx
  unfold lm_head.lm_head_fixed
  rw [if_neg (by rw [hne]; simp), hq]
  simp only [bind_tc_ok]
  exact headTail_none ggml_type raw vocab n_embd x

/-- With no quant geometry the projection refuses with `UnsupportedQuant`. -/
theorem matvec_unsupported (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (hx : quant.block_geom ggml_type = ok none) :
    matvec.matvec_fixed ggml_type raw n_out n_in x =
      ok (core.result.Result.Err (matvec.MatvecError.UnsupportedQuant ggml_type)) := by
  unfold matvec.matvec_fixed
  rw [hx]
  simp only [bind_tc_ok, core.option.Option.ok_or_none,
    core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- With a `div` quant-geometry lookup the projection cannot be an `ok`. -/
theorem matvec_div_ne_ok (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (z : core.result.Result Logits matvec.MatvecError)
    (hx : quant.block_geom ggml_type = Result.div) :
    ¬ matvec.matvec_fixed ggml_type raw n_out n_in x = ok z := by
  unfold matvec.matvec_fixed
  rw [hx, bind_tc_div]
  exact div_ne_ok z

/-- With a visible-effect quant-geometry lookup the projection cannot be an `ok`. -/
theorem matvec_vis_ne_ok (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (z : core.result.Result Logits matvec.MatvecError) {i : RustEffect.Input}
    {k : RustEffect.Output i → Result (Option (Usize × Usize))}
    (hx : quant.block_geom ggml_type = Result.vis i k) :
    ¬ matvec.matvec_fixed ggml_type raw n_out n_in x = ok z := by
  unfold matvec.matvec_fixed
  rw [hx, bind_tc_vis]
  exact fun h => vis_ne_ok i _ z h

/-- **The delegation, success arm.** With both guards passed and the quant family resolved, a
    successful projection is the successful LM-head result, unchanged. -/
theorem head_delegates_ok (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (v : Logits) (g : Usize × Usize)
    (hx : x.val.length = n_embd.val)
    (hbg : quant.block_geom ggml_type = ok (some g))
    (hv : matvec.matvec_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Ok v)) :
    lm_head.lm_head_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Ok v) := by
  have hne : (Slice.len x != n_embd) = false := headWidthGuard x n_embd hx
  unfold lm_head.lm_head_fixed
  rw [if_neg (by rw [hne]; simp), hbg]
  simp only [bind_tc_ok]
  exact headTail_some_ok ggml_type raw vocab n_embd x g v hv

/-- **The delegation, refusal arm.** With both guards passed and the quant family resolved, every
    projection refusal reaches the caller as `LmHeadError.Matvec`, carrying the exact refusal. -/
theorem head_delegates_err (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (e : matvec.MatvecError) (g : Usize × Usize)
    (hx : x.val.length = n_embd.val)
    (hbg : quant.block_geom ggml_type = ok (some g))
    (hv : matvec.matvec_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Err e)) :
    lm_head.lm_head_fixed ggml_type raw vocab n_embd x =
      ok (core.result.Result.Err (lm_head.LmHeadError.Matvec e)) := by
  have hne : (Slice.len x != n_embd) = false := headWidthGuard x n_embd hx
  unfold lm_head.lm_head_fixed
  rw [if_neg (by rw [hne]; simp), hbg]
  simp only [bind_tc_ok]
  exact headTail_some_err ggml_type raw vocab n_embd x g e hv

/-- **The quant lookup cannot escape the two head arms.** The tail's result is an `ok`, so the
    quant-geometry lookup was an `ok`, and with the width guard passed the head either refuses on
    the geometry or delegates. A `div` or visible-effect lookup is unreachable at the LM head. -/
theorem head_quant_lookup_ok (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (hx : x.val.length = n_embd.val)
    (z : core.result.Result Logits lm_head.LmHeadError)
    (h : lm_head.lm_head_fixed ggml_type raw vocab n_embd x = ok z) :
    ∃ o, quant.block_geom ggml_type = ok o ∧
      headTail ggml_type raw vocab n_embd x o = ok z := by
  have hne : (Slice.len x != n_embd) = false := headWidthGuard x n_embd hx
  unfold lm_head.lm_head_fixed at h
  rw [if_neg (by rw [hne]; simp)] at h
  cases hb : (quant.block_geom ggml_type).match with
  | ok o =>
      have hbg : quant.block_geom ggml_type = ok o := (Result.match.isOk).1 hb
      rw [hbg, bind_tc_ok] at h
      exact ⟨o, hbg, h⟩
  | div =>
      have hbg : quant.block_geom ggml_type = Result.div := (Result.match.isDiv).1 hb
      rw [hbg, bind_tc_div] at h
      exact (div_ne_ok z h).elim
  | vis i k =>
      have hbg : quant.block_geom ggml_type = Result.vis i k := (Result.match.isVis).1 hb
      rw [hbg, bind_tc_vis] at h
      exact (vis_ne_ok i _ z h).elim
/-! ## 4 · The row loop of the composed projection: transitions, work measure and refusals -/

/-- The row-stride premise the production entry point establishes before entering the row loop: the
    stride of row `o` plus one row width stays in `usize`, and the row index itself stays in
    `usize`. The entry point derives it from `need = n_out · row_bytes ≤ raw.len() ≤ usize::MAX`
    with `o < n_out`. -/
def StrideInRange (o row_bytes : Usize) : Prop :=
  o.val * row_bytes.val + row_bytes.val ≤ Usize.max ∧ o.val + 1 ≤ Usize.max

/-- **The output-capacity premise of a visited row.** The row body is *not* infallible for an
    arbitrary output vector: it appends one `Q32.32` logit per visited row through the real
    `alloc.vec.Vec.push`, and that model fails with `maximumSizeExceeded` — a `Result.vis
    (.fail …)`, not an `ok` — exactly at `out.val.length = Usize.max`
    (`Aeneas/Std/Vec.lean:203`: the push guards `List.length v.val + 1 ≤ U32.max ∨ ≤ Usize.max`).
    `Vec.property` gives only `out.val.length ≤ Usize.max`, which does not exclude the boundary, so
    the `ok (ControlFlow.cont …)` conclusion of `row_body_visit_of_dot_ok` below holds only where
    capacity remains, and it states that premise as `hout` on the pre-state `out` immediately before
    the body invocation.

    `row_body_capacity` below derives `hout` from the row-loop length invariant
    `out.val.length ≤ o.val` together with `o.val < n_out.val` and the `Usize` bound on `n_out`;
    it is not an assumption about the output as a whole. -/
theorem row_body_capacity (out : Logits) (o n_out : Usize)
    (hlen : out.val.length ≤ o.val) (ho : o.val < n_out.val) : out.val.length < Usize.max := by
  have hn : n_out.val ≤ Usize.max := by scalar_tac
  omega

/-- **The row-loop exit: the row budget is spent.** With `o ≥ n_out` the actual body returns
    `done (out, err)` without reading a row, so no further vocabulary row is visited and the row
    loop's output is final. -/
theorem row_body_exit_rows (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : Logits) (o : Usize)
    (err : Option matvec.MatvecError) (ok1 : Bool)
    (ho : n_out.val ≤ o.val) :
    matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need out row_buf o ok1 err =
      ok (ControlFlow.done (out, err)) := by
  unfold matvec.matvec_fixed_loop.body
  rw [if_neg (fun hc => by have := (UScalar.lt_equiv o n_out).1 hc; omega)]

/-- **The row-loop exit: a refusal is recorded.** With the failure flag set the actual body returns
    `done (out, err)` at once, so a refusal stops the row loop at the visit that set it, and the
    refusal already in `err` is the one the caller receives. -/
theorem row_body_exit_flag (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : Logits) (o : Usize)
    (err : Option matvec.MatvecError) (ho : o.val < n_out.val) :
    matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need out row_buf o false err =
      ok (ControlFlow.done (out, err)) := by
  unfold matvec.matvec_fixed_loop.body
  rw [if_pos ((UScalar.lt_equiv o n_out).2 ho), if_neg (by simp)]

/-- **One row visit: the body's outcomes, and the rank drop when it continues.** With a row left
    (`ho`), the row-stride premise in range (`hstride`), output capacity remaining (`hout`, see
    `row_body_capacity`) and the row's own dot computation returning an `ok` (`hdot`), the actual
    row-loop body has exactly the outcomes below, each checked as an exact equality:

    * the row index advances by exactly one (`o + 1#usize = ok onext`, `onext.val = o.val + 1`) and
      the rank `rowVisits n_out o` drops by exactly one, so it is positive whenever a row remains.
      These two facts hold on *every* outcome, because they are about the row index, not the body:
      this is the transition the work measure counts, and the body can only *finish* at a later
      visit, through one of the two exits above.
    * the body returns `ok (ControlFlow.cont …)` — either the row-stride refusal `LengthOverflow`
      (first arm: the untouched `out`, `row_buf`, the row index advanced, that refusal in `err'`), or
      a continuation carrying the appended logit together with the refusals the row's own
      computation recorded. The refusal arms the case split enumerates here are `ShortPayload` on
      the row read, `Dequant` in the dequantizer, `RowLength` on the dequantized width, and the
      dot's own `Err` refusal.
    * the body propagates the dequantizer's own `vis` or `div` unchanged, and the row loop cannot
      continue on that visit. These two arms are *enumerated outcomes*, not assumptions: this file
      makes no assumption about the dequantizer, and it cannot, because the continuation claim is
      false when the dequantizer takes either of these two outcomes (§6).

    **`hdot` is an assumption, not a fact established here.** It says: for every row the dequantizer
    writes into a cleared scratch buffer `cb`, for the resulting dequantized row `rb3`, the row dot
    `matvec.row_dot_fixed` on `rb3` against `x` returns `ok (core.result.Result.Ok w)` for the
    `I64` logit `w` it produces, and `w` is the value the body then appends. The `vis`, `div` and
    `Err` outcomes of the row dot are excluded *by hypothesis*, not by a proof. The unconditional
    claim that the body is an `ok (ControlFlow.cont …)` for every visited row is **retired**; §6
    names the residual obligation, and this theorem does not discharge it. This file also does not
    exhibit an instance of `hdot`, so satisfiability of the premise is open with it: the theorem is
    a conditional transition, not a statement that the premise ever holds.

    `hout` is a domain premise, not a repair of the conclusion: at
    `out.val.length = Usize.max` the body's own `Vec.push` refuses with `vis
    (.fail maximumSizeExceeded)` and the body is not an `ok (ControlFlow.cont …)` at all. The
    production entry `matvec.matvec_fixed` starts its row loop from the empty vector
    `alloc.vec.Vec.with_capacity Std.I64 n_out` at row offset `0#usize` with the flag `true`, so the
    invariant `out.val.length ≤ o.val` holds at entry and each visited row appends at most one
    logit while advancing `o` by one; `row_body_capacity` is the checked form of that derivation.
    This file does not iterate the row loop, so it establishes the invariant's entry value and the
    per-visit append, not the invariant at every active iteration. -/
theorem row_body_visit_of_dot_ok (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : Logits) (o : Usize)
    (err : Option matvec.MatvecError)
    (ho : o.val < n_out.val) (hstride : StrideInRange o row_bytes)
    (hout : out.val.length < Usize.max)
    (hdot : ∀ (cb rb3 : Logits) (row : Slice U8) (u : Unit),
        dequant.dequant_row_fixed ggml_type row n_in cb = ok (core.result.Result.Ok u, rb3) →
          ∃ w : I64, matvec.row_dot_fixed (alloc.vec.Vec.deref rb3) x = ok (core.result.Result.Ok w)) :
    ∃ onext : Usize, o + 1#usize = ok onext ∧ onext.val = o.val + 1 ∧
      (matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need out row_buf o true err
        = ok (ControlFlow.cont (out, row_buf, onext, false, some matvec.MatvecError.LengthOverflow))
        ∨ (∃ out' row_buf' ok' err',
            matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need out row_buf o true err
              = ok (ControlFlow.cont (out', row_buf', onext, ok', err')))
          ∨ ((∃ i : RustEffect.Input, ∃ k : RustEffect.Output i → Result (ControlFlow RowState RowValue),
                matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need out row_buf o true err
                  = Result.vis i k)
              ∨ matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need out row_buf o true err
                  = Result.div)) ∧
      rowVisits n_out onext + 1 = rowVisits n_out o ∧
      1 ≤ rowVisits n_out o := by
  obtain ⟨onext, honext, honextv⟩ := usize_add_val o (1#usize) hstride.2
  have honextv' : (onext : Usize).val = o.val + 1 := by rw [honextv]; norm_num
  have hnext : rowVisits n_out onext + 1 = rowVisits n_out o ∧ 1 ≤ rowVisits n_out o := by
    refine ⟨?_, ?_⟩
    · simp only [rowVisits]; rw [honextv']; omega
    · simp only [rowVisits]; omega
  refine ⟨onext, honext, honextv', ?_⟩
  refine And.intro ?_ (And.intro hnext.1 hnext.2)
  unfold matvec.matvec_fixed_loop.body
  rw [if_pos ((UScalar.lt_equiv o n_out).2 ho)]
  rw [if_pos rfl]
  simp only [lift, bind_tc_ok]
  rw [honext]
  by_cases hmul : Usize.checked_mul o row_bytes = none
  · rw [hmul]
    exact Or.inl (by simp)
  · obtain ⟨start, hmul, hstart⟩ := usize_checked_mul_some o row_bytes (by
      have := hstride.1; omega)
    rw [hmul]
    dsimp only
    obtain ⟨iend, hiend, hiendv⟩ := usize_add_val start row_bytes (by
      rw [hstart]; exact hstride.1)
    rw [hiend]
    simp only [bind_tc_ok, sliceGetRange_eq]
    by_cases hnone : core.slice.index.SliceIndexRangeUsizeSlice.get { start, «end» := iend } raw
        = ok none
    · rw [hnone]
      exact Or.inr (Or.inl ⟨out, row_buf, false,
        some (matvec.MatvecError.ShortPayload need (Slice.len raw)), by simp⟩)
    · obtain ⟨row, hrow⟩ := rowRead_some { start, «end» := iend } raw hnone
      rw [hrow]
      obtain ⟨cleared, hclear, hclearv⟩ := vec_clear_empty row_buf
      rw [hclear]
      simp only [bind_tc_ok]
      cases hdeq' : dequant.dequant_row_fixed ggml_type row n_in cleared with
      | ret p =>
          obtain ⟨r, row_buf3⟩ := p
          cases r with
          | Err e =>
              refine Or.inr (Or.inl ?_)
              refine ⟨out, row_buf3, false, some (matvec.MatvecError.Dequant e), ?_⟩
              simp
          | Ok u =>
              by_cases hlenEq : alloc.vec.Vec.len row_buf3 = n_in
              · simp
                rw [if_pos hlenEq]
                -- The row's dot computation is an `ok` by the premise `hdot`, instantiated at the
                -- dequantized row the body has just produced: `hdeq'` is the dequantizer equation
                -- `hdot` is conditional on. No case split on the dot's outcome remains.
                obtain ⟨w, hdotv⟩ := hdot cleared row_buf3 row u hdeq'
                rw [hdotv]
                obtain ⟨out2, hpush, _⟩ := vec_push_val out w (by
                  have := hout; omega)
                simp
                rw [hpush]
                refine Or.inr (Or.inl ?_)
                refine ⟨out2, row_buf3, Or.inr ⟨err, ?_⟩⟩
                simp
              · simp
                rw [if_neg hlenEq]
                refine Or.inr (Or.inl ?_)
                refine ⟨out, row_buf3, Or.inl ⟨
                  some (matvec.MatvecError.RowLength o (alloc.vec.Vec.len row_buf3) n_in), ?_⟩⟩
                simp
      | vis e k =>
          -- The dequantizer's own visible effect is propagated by the body: the row loop cannot
          -- continue on this visit, and the effect reaches the caller unchanged. This is an outcome
          -- of the body, not a continuation, so the unconditional continuation claim is false here.
          refine Or.inr (Or.inr (Or.inl ?_))
          simp only [bind_tc_vis, bind_vis]
          exact ⟨_, _, rfl⟩
      | div =>
          refine Or.inr (Or.inr (Or.inr ?_))
          simp only [bind_tc_div]

/-- **The entry rank of the row loop, in closed form.** `rowVisits vocab 0` is the number of
    vocabulary rows still to visit before the loop takes its first step, and at the zero offset
    that number is `vocab.val`.

    This is the entry rank only, and it is *not* a count of executed visits. The executed row-visit
    count is **retired** (§6): no theorem here iterates the row-loop step, so none of them counts
    what the loop executes, and the conditional `row_body_visit_of_dot_ok` does not supply that
    iteration. -/
theorem row_visits_closed_form (vocab : Usize) :
    rowVisits vocab (0#usize) = vocab.val := by
  simp only [rowVisits]
  have h0 : (0#usize).val = 0 := by norm_num
  rw [h0]
  omega

/-- **The two entry measures, multiplied.** The entry rank of the row loop times the weight count
    one row dot reduces over is `vocab.val · n_embd.val`. This is a product of the two measures at
    the entry state; it is not a count of executed weight positions, which would need the per-row
    work established by iteration, and the executed row-visit count it multiplies is retired (§6). -/
theorem lm_head_weight_positions (vocab n_embd : Usize)
    (w_row : Slice I64) (hw : w_row.val.length = n_embd.val) :
    rowVisits vocab (0#usize) * dotVisits w_row.val.length = vocab.val * n_embd.val := by
  rw [row_visits_closed_form, dotVisits, hw]

/-! ## 5 · The dot loop of the same projection: transitions, work measure and refusals -/

/-- **The dot-loop transition.** With a position left in the weight row and the failure flag clear
    the actual body advances the position by exactly one, so the rank `len - i` drops by exactly
    one per reduced weight. -/
theorem dot_body_visit (w_row x : Slice I64) (acc : I128) (i : Usize)
    (hi : i.val < w_row.val.length) (hx : x.val.length = w_row.val.length) :
    ∃ acc' i' acc_ok',
      matvec.row_dot_fixed_loop.body w_row x acc i true =
        ok (ControlFlow.cont (acc', i', acc_ok')) ∧
      i'.val = i.val + 1 := by
  obtain ⟨inext, hinext, hinextv⟩ :=
    usize_add_val i (1#usize) (by
      have hb := w_row.property
      have h1 : (1#usize).val = 1 := by norm_num
      rw [h1]; omega)
  unfold matvec.row_dot_fixed_loop.body
  rw [if_pos ((UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [sliceLenVal]; exact hi))]
  rw [if_pos rfl]
  simp only [lift, bind_tc_ok]
  obtain ⟨iw, hiw, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec w_row i hi)
  have hiX : i.val < x.val.length := by rwa [hx]
  obtain ⟨ix, hix, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hiX)
  rw [hiw, hix]
  simp only [bind_tc_ok]
  cases hmul : (core.convert.num.FromI128I64.from iw).checked_mul
      (core.convert.num.FromI128I64.from ix) with
  | none =>
      simp only [bind_tc_ok]
      rw [hinext]
      exact ⟨acc, inext, false, by simp [bind_tc_ok], hinextv⟩
  | some prod =>
      cases hadd : acc.checked_add prod with
      | none =>
          simp only [bind_tc_ok]
          rw [hinext]
          rw [hadd]
          exact ⟨acc, inext, false, by simp [bind_tc_ok], hinextv⟩
      | some total =>
          simp only [bind_tc_ok]
          rw [hinext]
          rw [hadd]
          exact ⟨total, inext, true, by simp [bind_tc_ok], hinextv⟩

/-- **The dot-loop exit: the weight row is exhausted.** With `i ≥ len w_row` the actual body
    returns `done (acc, acc_ok)` without multiplying or adding again. -/
theorem dot_body_exit_end (w_row x : Slice I64) (acc : I128) (i : Usize) (acc_ok : Bool)
    (hi : w_row.val.length ≤ i.val) :
    matvec.row_dot_fixed_loop.body w_row x acc i acc_ok =
      ok (ControlFlow.done (acc, acc_ok)) := by
  unfold matvec.row_dot_fixed_loop.body
  rw [if_neg (fun hc => absurd ((UScalar.lt_equiv i (Slice.len w_row)).1 hc)
    (Nat.not_lt.mpr (by simpa [sliceLenVal] using hi)))]

/-- **The dot-loop exit: a refusal is recorded.** With the failure flag set the actual body returns
    `done (acc, false)` at once, so an `i128` refusal stops the row dot at the visit that set it. -/
theorem dot_body_exit_flag (w_row x : Slice I64) (acc : I128) (i : Usize)
    (hi : i.val < w_row.val.length) :
    matvec.row_dot_fixed_loop.body w_row x acc i false = ok (ControlFlow.done (acc, false)) := by
  unfold matvec.row_dot_fixed_loop.body
  rw [if_pos ((UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [sliceLenVal]; exact hi))]
  rw [if_neg (by simp)]

/-- **The dot-loop exit arms are exact.** With the position inside the row and the flag clear the
    body always continues; with the position past the row, or the flag set, it always finishes.
    These are the two exits, so the body has no third outcome. -/
theorem dot_body_exhaustive (w_row x : Slice I64) (acc : I128) (i : Usize) (acc_ok : Bool) :
    (i.val < w_row.val.length ∧ acc_ok = true) ∨
    w_row.val.length ≤ i.val ∨ acc_ok = false := by
  cases acc_ok with
  | false => exact Or.inr (Or.inr rfl)
  | true =>
      by_cases h1 : i.val < w_row.val.length
      · exact Or.inl ⟨h1, rfl⟩
      · exact Or.inr (Or.inl (by omega))

/-! ## 6 · Retired claims: not established, and not proved by anything above

Two claims of the earlier unconditional form of this file are **retired**. Neither is established
here, no theorem in this file establishes either, and the conditional `row_body_visit_of_dot_ok` is
not to be read as establishing either.

1. **Unconditional row-body success** — that every visited vocabulary row makes the row loop's
   body return `ok (ControlFlow.cont …)`. This claim is not merely unproved: as stated it is
   **false**, for two separate reasons, and `row_body_visit_of_dot_ok` is what remains after both
   are removed.

   * The dequantizer's own `vis` and `div` outcomes are propagated by the body, so on those
     branches the body is not an `ok` at all. The earlier version of this file tried to close those
     two branches with `pairBind_ne_ok … |>.elim`, which asks for a refutation of the goal itself;
     that request cannot be met, because the goal is false there rather than contradictory. The
     two branches were also *never elaborated* — they sit behind the dot branches' tactic errors in
     the `cases` alternatives, so the elaborator never reached them, and their apparent closure was
     an artefact of the abort. The current conclusion **enumerates** those two outcomes instead of
     assuming them away, and makes no assumption about the dequantizer.
   * The dot's `vis`, `div` and `Err` outcomes are excluded by the premise `hdot`, which this file
     does not discharge. Excluding them decomposes into

     - termination of `matvec.row_dot_fixed_loop` — a decreasing measure
       `w_row.val.length - i` driven by the green `dot_body_visit`, `dot_body_exit_end`,
       `dot_body_exit_flag` and `dot_body_exhaustive`, whose one-step unfolding needs
       `loop.eq_1` because the extracted `loop` is a `partial_fixpoint`; and
     - the `acc_ok = true` tail of `matvec.row_dot_fixed`, whose `value.round_shift_32` is three
       `IScalar` operations (`I128` shift, add, shift-right) whose `ok`-ness the model does not
       specify. Landing them means new model-facing specifications, which is a separate gated task
       and not a repair of this file.

     Satisfiability of `hdot` is also open here: no instance of it is exhibited, so the conditional
     theorem is not a statement that the premise ever holds. The dot's two non-`ok` outcomes are in
     fact enumerable in the same way the dequantizer's are; whether to assume them (`hdot`, as here)
     or to enumerate them is a scope decision for the owner, not a proof detail.

2. **The executed row-visit count** — that the executing row loop performs exactly
   `rowVisits vocab (0#usize) = vocab.val` visits, end to end. No theorem here iterates the
   row-loop step, so none of them counts executed visits; `row_visits_closed_form` and
   `lm_head_weight_positions` are statements about the entry rank and about the product of the two
   entry measures, and they survive the retirement unchanged because they never claimed a count.
   `row_body_visit_of_dot_ok` does not establish this claim either: it is a one-visit transition
   under an assumption, not an iteration, and its own doc comment says so.

The conditional replacement, with its two enumerated propagation arms, is what this gate's
row-level resource claim is, and it is audited as such. Anything downstream that needs either
retired claim — an unconditional row success, or an end-to-end row visit count — is **not**
discharged by this file and by this gate's success. -/

end LmHeadResource

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

#resource_audit LmHeadResource.sliceLenVal
#resource_audit LmHeadResource.sliceLen_eq
#resource_audit LmHeadResource.sliceLen_of_eq
#resource_audit LmHeadResource.usize_bne_false_iff
#resource_audit LmHeadResource.usize_ne_of_bne_true
#resource_audit LmHeadResource.usize_bne_true_iff
#resource_audit LmHeadResource.headWidthGuard
#resource_audit LmHeadResource.one_usize_val
#resource_audit LmHeadResource.usize_add_val
#resource_audit LmHeadResource.vec_push_val
#resource_audit LmHeadResource.vec_clear_empty
#resource_audit LmHeadResource.sliceGetRange_eq
#resource_audit LmHeadResource.rowRead_some
#resource_audit LmHeadResource.usize_checked_mul_some
#resource_audit LmHeadResource.vis_ne_ok
#resource_audit LmHeadResource.div_ne_ok
#resource_audit LmHeadResource.pairBind_ne_ok
#resource_audit LmHeadResource.pairBind_div_ne_ok
#resource_audit LmHeadResource.VisitTrace.projection
#resource_audit LmHeadResource.VisitTrace.deterministic
#resource_audit LmHeadResource.VisitTrace.excludes_divergence
#resource_audit LmHeadResource.matvec_ctor_call
#resource_audit LmHeadResource.headTail_none
#resource_audit LmHeadResource.headTail_some_ok
#resource_audit LmHeadResource.headTail_some_err
#resource_audit LmHeadResource.head_shape_refusal
#resource_audit LmHeadResource.head_quant_refusal
#resource_audit LmHeadResource.matvec_unsupported
#resource_audit LmHeadResource.matvec_div_ne_ok
#resource_audit LmHeadResource.matvec_vis_ne_ok
#resource_audit LmHeadResource.head_delegates_ok
#resource_audit LmHeadResource.head_delegates_err
#resource_audit LmHeadResource.head_quant_lookup_ok
#resource_audit LmHeadResource.row_body_exit_rows
#resource_audit LmHeadResource.row_body_exit_flag
#resource_audit LmHeadResource.row_body_capacity
#resource_audit LmHeadResource.row_body_visit_of_dot_ok
#resource_audit LmHeadResource.row_visits_closed_form
#resource_audit LmHeadResource.lm_head_weight_positions
#resource_audit LmHeadResource.dot_body_visit
#resource_audit LmHeadResource.dot_body_exit_end
#resource_audit LmHeadResource.dot_body_exit_flag
#resource_audit LmHeadResource.dot_body_exhaustive

run_cmd Lean.logInfo "LM_HEAD_RETIRED_CLAIMS row_body_success_unconditional row_visit_count_end_to_end"
run_cmd Lean.logInfo "LM_HEAD_KERNEL_OK"
