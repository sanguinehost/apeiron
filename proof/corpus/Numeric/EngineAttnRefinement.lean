/-
Refinement of the extracted single-token attention kernel.

The Rust body is `src/engine/attn.rs` (sha256
`cf6cb15805f79e991da7a37da0020a925c16043478de358b76ff01a489f278f0`, 2055 lines) at the staged
extraction: `AttnError` 36-83, `KvStorage` 87-102, `alloc_f32_storage` 118-123 (admitted as
`attn.alloc_f32_storage`), `KvState` 134-144, `plane_at` 151-155, `score_at` 180-204,
`exp_at` 206-213, `weight_at` 222-231, `recip_u64` 252-258, `weight_at_recip` 279-315,
`KvState::with_carrier` 766-826 (a `def`, not an admitted item), `KvState::append` 912-990, and
`KvState::attn_head` 1027-1237 with its four loops: the score loop at 1090-1120, the softmax loop
at 1132-1149, the weight/value loop at 1178-1216 with its per-component accumulation loop at
1194-1207, and the output epilogue loop at 1223-1232. The `f32` carrier path
`KvState::attn_head_f32` (1263-1398) is admitted as `attn.KvState.attn_head_f32` and is outside
this module. The public entry point is `apeiron_word.word_attn_head`, the extracted wrapper over
`attn.KvState.attn_head`; the extracted kernel is `proof/ApeironWord.lean`, sha256
`c16a68f0232e6b8ae4997e8603e605ed499b4b50b4e211d824f485afbc2fab1c`, 24790 lines, roots
`word_kv_state_new`, `word_kv_append`, `word_attn_head`, `word_argmax_fixed`. Every source line
range above is the `Source:` annotation the extraction itself carries for that item, and both
digests and counts are read from the named artifacts.


## The two discriminants

`KvState` carries a `mixer.Carrier` and an `attn.KvStorage`. `attn_head` reads the carrier first
(`carrier == F32` dispatches to the admitted `attn_head_f32`) and then matches the storage, whose
`Q32` arm carries the two plane tables and whose `F32` arm returns `Err BadGeometry`. Every
theorem below therefore assumes `state.carrier = mixer.Carrier.Q32` and
`state.storage = attn.KvStorage.Q32 ktab vtab` as hypotheses. The relation between the two
discriminants is made by the constructor, whose `Carrier::F32` arm is the admitted item
`attn.alloc_f32_storage`; no theorem here uses the constructor: each proved statement holds for
either carrier or storage given its own hypotheses.

## The exact specification from `attn.rs`

Per query head `q_head` at position `pos`, over the causal prefix `0..=pos` (positions past `pos`
are never read):

* geometry (`attn_head` guards): `n_layer`, `n_head`, `n_head_kv`, `head_dim`, `cap` are nonzero,
  `n_head` is a multiple of `n_head_kv` (GQA), `head_dim` is in the `fixed_rsqrt` window, and the
  load-time score scale is `scale = fixed_rsqrt(head_dim · 2^32)`, computed once at load;
* GQA head mapping and plane addressing: `group = n_head / n_head_kv`, `kvh = q_head / group`, and
  the two plane offsets `stride = n_head_kv · head_dim` and `first = kvh · head_dim` are computed
  with checked `usize` multiplication, each with its own `Overflow` arm. The plane row of
  `(s, kvh)` is the `head_dim` values at `first + s · stride = (s · n_head_kv + kvh) · head_dim`
  (position-major, `plane_at`), and the row is read only when `base ≤ len(plane) - head_dim`
  (saturating), otherwise `Overflow`;
* score (`attn_head_loop0`, `attn.rs:1090-1120`): `raw_j = round(Σ_i q_i·k_{j,i} / 2^32)`
  (`matvec.row_dot_fixed`, one round-to-nearest `>> 32`, ties upward) and
  `score_j = round(raw_j · scale / 2^32)` — one value-path epilogue `value.round_shift_32` against
  the load-time scale. Both reductions are the exact half-biased integer division
  `divRoundNearest`;
* softmax (`attn_head_loop1` → `exp_at`, `attn.rs:1132-1149`): exact maximum shift
  `mx = max_j score_j`; numerator `e_j = fixed_exp(score_j - mx)` with the documented underflow
  saturation `e_j = 0` when `score_j - mx < -16·2^32`; the `fixed_exp` window
  `[-16·2^32, 21·2^32]` is otherwise unchanged and no attended value is clamped; the total `Σ e_j`
  is a checked `i128` accumulation, refused as `ZeroSum` when `total ≤ 0`;
* weights (`recip_u64`, `attn_head_loop2`, `attn.rs:1178-1216`): `recip = recip_u64 total` is the
  `u64` reciprocal of the total when the total is in range and `none` otherwise; each weight is
  `weight_at_recip e_j total recip`, which falls back to the half-biased division
  `weight_at e_j total = divRoundNearest (e_j·2^32) total` narrowed to `i64` when the reciprocal
  path narrows to `none`;
* output (`attn_head_loop2` accumulation at `attn.rs:1194-1207` and `attn_head_loop3` at
  `attn.rs:1223-1232`): the value loop
  accumulates `acc[i] = Σ_j w_j·v_{j,i}` with checked `i128` multiply-adds into a `head_dim`-wide
  accumulator, and the output loop writes `out[i] = round_shift_32 acc[i]` narrowed to `i64` — one
  value-path epilogue per component;
* refusals: `BadGeometry` (the `F32` storage arm), `ScaleOutOfRange`, `LayerOutOfRange`,
  `HeadOutOfRange`, `RowLength`, `PositionOutOfRange`, `CapacityExceeded` (`append`),
  `ExpOutOfRange`, `Overflow`, `ZeroSum`.

`Numeric.EngineDotRefinement` (with `Numeric.EngineNormRefinement`, `Numeric.EngineSiluRefinement`,
`Numeric.Exp.Part4`, `Numeric.RsqrtRefinement` transitively) is imported for the proved
`row_dot_refines_proved`, `round_shift_32_spec`, `div_round_nearest_abs_bound`,
`fixed_exp_refines_proved`, `fixed_rsqrt_refines_proved`, `engine_ONE_eq` and the checked-`i128`
witnesses; those lemmas are reused, not restated.

## Dense softmax reference and the online reading

The real reference of this module is the **dense** softmax attention over the causal prefix,
`Σ_j softmax(score)_j · v_j` (`denseAttnReal`), computed at the exact maximum shift. The shift is
exact, so the reference is shift-invariant (`softmaxWeights_shift_invariant`). The correspondence
to `softmax_attention_is_online_spinor`
(`proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean:462`) is a **reading**, not a
refinement: that theorem states the real online recurrence `num/den = Σ e^sc·v / Σ e^sc` for the
`O(1)`-state `(m, den, num)` scan and identifies the dense form with the recurrence over `ℝ`;
this module does not import it and does not refine the kernel against it.

## Dominant-kernel complexity

`attn_head` (`src/engine/attn.rs:1027`) is `O((pos+1) · head_dim)` time in checked integer
multiply-adds and `fixed_exp` calls — the score stage visits each of the `pos+1` keys over
`head_dim` coordinates (`attn_head_loop0` calls `row_dot_fixed` once per key row) and the value
stage visits each of the `head_dim` components over `pos+1` keys — with `O(pos+1)` scratch for the
scores and exponentials (`Vec::with_capacity slots`, `slots = pos + 1`), a `head_dim`-wide
`Vec<I128>` accumulator (`alloc.vec.from_elem`), and `out` `head_dim` wide. Each loop is a linear
scan closing after `pos+1` (stages 0-2) or `head_dim` (stages 3-4) steps; `fixed_exp`,
`fixed_rsqrt`, `recip_u64` and the epilogues are `O(1)` word operations.

## What is proved here and the residual

Landed here:

* the exact stage specifications (`attnScoreInt`, `numeratorReal`, `softmaxWeights`,
  `denseAttnReal`, `planeRow`, `scoreList`, `valueList`, `idealScoreList`) and the end-to-end
  target (`attn_domain`, `attnErrBound`, `attn_head_refines`);
* the structural unfold past the carrier dispatch, the guard chain and the storage match
  (`word_attn_head_unfold`) and the refusal equalities: `BadGeometry`, `LayerOutOfRange`,
  `HeadOutOfRange`, both `RowLength` arms, `PositionOutOfRange`, the `slots` and both plane-offset
  `Overflow` arms, and the loop-carried refusals as equalities on the loop result
  (`ExpOutOfRange`/`Overflow` through `attn_head_loop1`, `ZeroSum` on `total ≤ 0`, the value-stage
  and output-stage refusals);
* the score/normaliser lemmas: the score-stage `i128` product never overflows
  (`score_epilogue_mul_fits`) and its epilogue bound (`score_epilogue_bound`); the max-shift
  saturation as an exact two-sided threshold with `-16·2^32` pinned (`exp_at_threshold_eq`,
  `exp_at_underflow`, `exp_at_window`, `exp_at_refuses`); the `fixed_exp` passthrough bound
  (`exp_at_real_bound`) and the saturation error (`exp_at_underflow_real`); the fallback weight
  stage identity and normaliser rounding bound (`weight_at_spec`, `weight_rounding_bound`); the
  per-component output epilogue identity (`attn_head_loop3_body_epilogue`) and its rounding bound
  (`value_component_rounding_bound`).

Residual (`attn_head_refines_proved` is not proved; `attn_head_refines` is unchanged as a target):

1. the five loop specifications (`attn_head_loop0..3` and the nested `attn_head_loop2_loop0`) with
   the exact accumulator identities (`scores` = `scoreList`, `mx` = the exact maximum, `exps`/`total`
   = the numerator list and its checked sum, `acc` = `Σ_j w_j·v_{j,i}`, `out` = the epilogue per
   component) and the range-index identity tying `planeRow` to the plane contents
   (`base = (s·n_head_kv + kvh)·head_dim`);
2. the score-stage composition `fixed_rsqrt_refines_proved` + `row_dot_refines_proved` through
   `attn_head_loop0` (needs item 1's `row_dot_fixed` call on the range-indexed key row);
3. the exp→softmax perturbation composition: the per-numerator error `fixed_exp_refines_proved`
   (and the saturation `exp_at_underflow_real`) propagated through the checked `i128` normaliser
   to the dense weights, then through the weight and accumulation stages and `attn_head_loop3` to
   `out`;
4. `attn.recip_u64` and `attn.weight_at_recip`: the reciprocal weight path taken whenever
   `recip = some r` has no specification here, so `weight_at_spec` and `weight_rounding_bound`
   cover only the `recip = none` fallback and item 3 cannot be closed over the executed path.

`attnErrBound` is an `anchor`: its additive decomposition (output epilogue, weight rounding,
`fixed_exp` numerator error through the normaliser, score error through softmax) is documented at
the definition; no theorem claims it.

## Trust boundary

The evidence reaches the extracted kernel `proof/ApeironWord.lean` at the digest above and the
Q32.32 denotation in `Numeric.Scale`. It does not cover the admitted
`attn.KvState.attn_head_f32` and `attn.alloc_f32_storage` (the `f32` carrier path), rustc/LLVM
codegen of
`src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.EngineDotRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Result `Try` reductions -/

/-- `Try::branch` on `Ok` yields `Continue`. -/
lemma result_branch_ok {α ε : Type} (v : α) :
    core.result.Result.Insts.CoreOpsTry.branch (T := α) (E := ε) (core.result.Result.Ok v)
      = ok (core.ops.control_flow.ControlFlow.Continue v) := rfl

/-- `Try::branch` on `Err` yields the `Err` residual. -/
lemma result_branch_err {α ε : Type} (e : ε) :
    core.result.Result.Insts.CoreOpsTry.branch (T := α) (E := ε) (core.result.Result.Err e)
      = ok (core.ops.control_flow.ControlFlow.Break (core.result.Result.Err e)) := rfl

/-- `from_residual` with `FromSame` re-emits the error. -/
lemma result_from_residual_err {T ε : Type} (e : ε) :
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual T
      (core.convert.FromSame ε) (core.result.Result.Err e) = ok (core.result.Result.Err e) := by
  simp only [core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, bind_tc_ok]

/-! ## Exact integer stage specifications -/

/-- The exact score of one causal key: `raw = round(Σ q·k / 2^32)` then
`score = round(raw·scale / 2^32)` (`src/engine/attn.rs:180-204`). Both reductions are the
half-biased integer division `divRoundNearest`; no value is clamped. -/
def attnScoreInt (dotProd scale : ℤ) : ℤ :=
  divRoundNearest (divRoundNearest dotProd ((2 : ℤ) ^ (32 : ℕ)) * scale) ((2 : ℤ) ^ (32 : ℕ))

/-- The real denotation of one max-shifted softmax numerator of `src/engine/attn.rs:206-213`:
`exp` of the Q32.32 shift, or the documented underflow saturation `0` below `-16·2^32`. -/
def numeratorReal (shifted : ℤ) : ℝ :=
  if shifted < -16 * (2 : ℤ) ^ (32 : ℕ) then 0 else Real.exp ((shifted : ℝ) / q32)

/-- The dense real softmax weights of a score list at the maximum shift `m`: one `exp` per score,
one exact normalisation. -/
def softmaxWeights (m : ℝ) (sc : List ℝ) : List ℝ :=
  (sc.map (fun s => Real.exp (s - m))).map
    (fun e => e / (sc.map (fun s => Real.exp (s - m))).sum)

/-- The dense real softmax attention of a causal prefix: `Σ_j softmax(score)_j · v_j` on real
lists. This is the dense reading of `softmax_attention_is_online_spinor` (a pointer, not a
refinement of this kernel). -/
def denseAttnReal (m : ℝ) (sc vs : List ℝ) : ℝ :=
  (((softmaxWeights m sc).zip vs).map (fun p => p.1 * p.2)).sum

/-- The signed Q32.32 denotation of an integer list. -/
def toRealList (l : List ℤ) : List ℝ := l.map (fun x => (x : ℝ) / q32)

/-- The row of the position-major plane of `(s, kvh)`: the `hd` values at `(s · kv + kvh) · hd`
(`src/engine/attn.rs:151-155`). Out-of-range reads default to `0#i64`; the in-domain geometry keeps
every index inside the plane. -/
def planeRow (plane : List I64) (kv hd : ℕ) (kvh s : ℕ) : List I64 :=
  (List.range hd).map (fun i => plane.getD ((s * kv + kvh) * hd + i) 0#i64)

/-- The exact realised score list of the causal prefix `0..=pos`: one `attnScoreInt` per key. -/
def scoreList (plane q : List I64) (kv hd : ℕ) (kvh : ℕ) (scale : ℤ) (pos : ℕ) : List ℤ :=
  (List.range (pos + 1)).map (fun s =>
    attnScoreInt (((planeRow plane kv hd kvh s).zip q).map (fun p => p.1.val * p.2.val)).sum scale)

/-- The exact value column `i` of the causal prefix `0..=pos`, as Q32.32 integers. -/
def valueList (plane : List I64) (kv hd : ℕ) (kvh : ℕ) (pos i : ℕ) : List ℤ :=
  (List.range (pos + 1)).map (fun s => ((planeRow plane kv hd kvh s).getD i 0#i64).val)

/-- The real row dot `<q, k>` of two Q32.32 rows over `2^64`, scaled by `1 / √hdim`: the ideal
score of one causal key, the reference of the score stage. -/
def idealScoreList (kp q : List I64) (kv hd : ℕ) (kvh : ℕ) (hdim : ℝ) (pos : ℕ) : List ℝ :=
  (List.range (pos + 1)).map (fun s =>
    ((((planeRow kp kv hd kvh s).zip q).map (fun p => (p.1.val : ℝ) * (p.2.val : ℝ))).sum / q32^2)
      / Real.sqrt hdim)

/-- `max_j |x_j|` over a real list, `0` on the empty list. -/
def listAbsMax (l : List ℝ) : ℝ := l.foldr (fun x r => max |x| r) 0

/-- The score-stage error of one key against `idealScoreList`, `anchor` derivation: row-dot
rounding `1/(2·2^32)` scaled by `|scale|`, the `fixed_rsqrt` scale error `epsExp·(1/√hdim) +
4/2^32` scaled by the row-dot magnitude `dotAbs`, and the scale epilogue `1/(2·2^32)`. -/
def scoreErr (hdim dotAbs scaleAbs : ℝ) : ℝ :=
  scaleAbs * (1 / (2 * q32))
    + dotAbs * (epsExp * (1 / Real.sqrt hdim) + 4 / q32)
    + 1 / (2 * q32)

/-! ## The in-domain target -/

/-- The documented in-domain geometry of `attn_head` on a built state: the parameters are nonzero,
`n_head` is a multiple of `n_head_kv`, the indices are in range, `q` and `out` are `head_dim`
wide, and `pos` is inside the position capacity. -/
def attn_domain (state : attn.KvState) (layer q_head pos : Usize) (q out : Slice I64) : Prop :=
  0 < state.n_layer.val ∧ 0 < state.n_head.val ∧ 0 < state.n_head_kv.val ∧
  0 < state.head_dim.val ∧ 0 < state.cap.val ∧
  state.n_head.val % state.n_head_kv.val = 0 ∧
  layer.val < state.n_layer.val ∧ q_head.val < state.n_head.val ∧
  q.val.length = state.head_dim.val ∧ out.val.length = state.head_dim.val ∧
  pos.val < state.cap.val

/-- The in-domain error target, `anchor` grade: the additive decomposition of the composed error
against the dense real softmax attention of the ideal scores. Terms, in order: the output epilogue
`round_shift_32` (`1/(2·2^32)`); the weight division `div_round_nearest_abs_bound` over `n` terms
(`n·V/(2·2^32)`); the `fixed_exp` numerator error `epsExp·exp + 4/2^32` through the normaliser at
shifted total `≥ 1` (`V·(2·epsExp + (n+2)·4/2^32)`); the score-stage error `δscore` through softmax
(`2·V·δscore`). `V` is `max_j |v_j|` and `n` is `pos + 1`. -/
def attnErrBound (n : ℕ) (V δscore : ℝ) : ℝ :=
  1 / (2 * q32) + (n : ℝ) * V * (1 / (2 * q32))
    + V * (2 * epsExp + ((n : ℝ) + 2) * (4 / q32)) + 2 * V * δscore

/-- **End-to-end target (residual).** On the `Q32` carrier and `Q32` storage, `word_attn_head`
returns `Ok` and each output component is within `attnErrBound` of the dense real softmax attention
of the ideal scores `<q, k_j> / √head_dim` over the causal prefix. This is the statement to prove;
it is not proved here — see the residual list in the module header. -/
def attn_head_refines : Prop :=
  ∀ (state : attn.KvState) (layer q_head pos : Usize) (q out : Slice I64)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64)) (kplane vplane : alloc.vec.Vec I64)
    (hdim : ℝ),
    attn_domain state layer q_head pos q out →
    state.carrier = mixer.Carrier.Q32 →
    state.storage = attn.KvStorage.Q32 ktab vtab →
    ktab.val[layer.val]? = some kplane →
    vtab.val[layer.val]? = some vplane →
    hdim = (state.head_dim.val : ℝ) →
    ∃ out' : Slice I64,
      word_attn_head state layer q_head q pos out = ok (core.result.Result.Ok (), out') ∧
      out'.val.length = out.val.length ∧
      ∀ i : ℕ, i < state.head_dim.val →
        let kvh := q_head.val / (state.n_head.val / state.n_head_kv.val)
        let sc := idealScoreList kplane.val q.val state.n_head_kv.val state.head_dim.val
          kvh hdim pos.val
        let vs := valueList vplane.val state.n_head_kv.val state.head_dim.val kvh pos.val i
        |toReal (out'.val.getD i 0#i64) - denseAttnReal (listAbsMax sc) sc (toRealList vs)|
          ≤ attnErrBound (pos.val + 1) (listAbsMax (toRealList vs))
              (scoreErr hdim (listAbsMax sc) |toReal state.scale|)

/-! ## Max-shift exactness at the reference -/

/-- The softmax weights do not depend on the shift: `exp(s-m) = exp(s-m')·exp(m'-m)`, the common
factor cancels in the normalisation. -/
theorem softmaxWeights_shift_invariant (m m' : ℝ) (sc : List ℝ) :
    softmaxWeights m sc = softmaxWeights m' sc := by
  have hpt : ∀ (e S c : ℝ), c ≠ 0 → (e * c) / (S * c) = e / S := by
    intro e S c hc
    by_cases hS : S = 0
    · simp [hS]
    · field_simp [hc, hS]
  have hsum : ∀ (l : List ℝ) (c : ℝ), (l.map (fun e => e * c)).sum = l.sum * c := by
    intro l c
    induction l with
    | nil => simp
    | cons a l ih => simp only [List.map_cons, List.sum_cons, ih]; ring
  have hmapEq : ∀ (l : List ℝ) (f g : ℝ → ℝ), (∀ x, f x = g x) → l.map f = l.map g := by
    intro l f g hfg
    induction l with
    | nil => rfl
    | cons a l ih => simp only [List.map_cons, hfg a, ih]
  have hexp : ∀ (s : ℝ), Real.exp (s - m) = Real.exp (s - m') * Real.exp (m' - m) := by
    intro s
    rw [← Real.exp_add]
    congr 1
    ring
  have hmapL : (sc.map (fun s => Real.exp (s - m)))
      = (sc.map (fun s => Real.exp (s - m'))).map (fun e => e * Real.exp (m' - m)) := by
    rw [List.map_map]
    exact hmapEq sc _ _ hexp
  unfold softmaxWeights
  rw [hmapL, hsum, List.map_map]
  apply hmapEq
  intro e
  exact hpt e _ _ (Real.exp_ne_zero _)

/-! ## The carrier dispatch -/

/-- The extracted `==` on the two-constructor `mixer.Carrier` reads the discriminant, so on `Q32`
against `F32` it is `ok false` and `attn_head` takes the Q32 arm. -/
lemma carrier_eq_q32_f32 :
    mixer.Carrier.Insts.CoreCmpPartialEqCarrier.eq mixer.Carrier.Q32 mixer.Carrier.F32 = ok false := by
  simp [mixer.Carrier.Insts.CoreCmpPartialEqCarrier.eq, mixer.Carrier.read_discriminant]

/-- The carrier test on a state whose `carrier` is `Q32`. -/
lemma carrier_eq_of_q32 (state : attn.KvState) (h : state.carrier = mixer.Carrier.Q32) :
    mixer.Carrier.Insts.CoreCmpPartialEqCarrier.eq state.carrier mixer.Carrier.F32 = ok false := by
  rw [h]
  exact carrier_eq_q32_f32

/-! ## Refusal equalities: the guard chain -/

/-- **Layer refusal.** `layer ≥ n_layer` returns `Err LayerOutOfRange` exactly. -/
theorem word_attn_head_layer_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (h : layer ≥ state.n_layer) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err
        (attn.AttnError.LayerOutOfRange layer state.n_layer), out) := by
  rw [word_attn_head, attn.KvState.attn_head]
  simp only [carrier_eq_of_q32 state hcarrier, bind_tc_ok]
  rw [if_neg (by decide)]
  rw [if_pos h]

/-- **Query-head refusal.** `q_head ≥ n_head` returns `Err HeadOutOfRange` exactly. -/
theorem word_attn_head_head_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (h : q_head ≥ state.n_head) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err
        (attn.AttnError.HeadOutOfRange q_head state.n_head), out) := by
  rw [word_attn_head, attn.KvState.attn_head]
  simp only [carrier_eq_of_q32 state hcarrier, bind_tc_ok]
  rw [if_neg (by decide)]
  rw [if_neg hlayer, if_pos h]

/-- **Query width refusal.** A query row not `head_dim` wide returns `Err RowLength` exactly. -/
theorem word_attn_head_q_len_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (h : q.val.length ≠ state.head_dim.val) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err
        (attn.AttnError.RowLength state.head_dim (Slice.len q)), out) := by
  have hne : Slice.len q ≠ state.head_dim := by
    intro hc
    have hv : (Slice.len q).val = state.head_dim.val := congrArg (fun u : Usize => u.val) hc
    rw [Slice.len_val q] at hv
    exact h hv
  have hcond : (Slice.len q != state.head_dim) = true := by simpa using hne
  rw [word_attn_head, attn.KvState.attn_head]
  simp only [carrier_eq_of_q32 state hcarrier, bind_tc_ok]
  rw [if_neg (by decide)]
  rw [if_neg hlayer, if_neg hhead, if_pos hcond]

/-- **Output width refusal.** An output row not `head_dim` wide returns `Err RowLength` exactly. -/
theorem word_attn_head_out_len_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (h : out.val.length ≠ state.head_dim.val) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err
        (attn.AttnError.RowLength state.head_dim (Slice.len out)), out) := by
  have hne : Slice.len out ≠ state.head_dim := by
    intro hc
    have hv : (Slice.len out).val = state.head_dim.val := congrArg (fun u : Usize => u.val) hc
    rw [Slice.len_val out] at hv
    exact h hv
  have hcond : (Slice.len out != state.head_dim) = true := by simpa using hne
  rw [word_attn_head, attn.KvState.attn_head]
  simp only [carrier_eq_of_q32 state hcarrier, bind_tc_ok]
  rw [if_neg (by decide)]
  rw [if_neg hlayer, if_neg hhead, if_neg (by simp [hq]), if_pos hcond]

/-- **Storage refusal.** A state whose `storage` is not the `Q32` arm returns `Err BadGeometry`
exactly, on the Q32 carrier and past the guard chain. -/
theorem word_attn_head_storage_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kf vf : alloc.vec.Vec (alloc.vec.Vec F32))
    (hstorage : state.storage = attn.KvStorage.F32 kf vf) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err attn.AttnError.BadGeometry, out) := by
  rw [word_attn_head, attn.KvState.attn_head]
  simp only [carrier_eq_of_q32 state hcarrier, bind_tc_ok]
  rw [if_neg (by decide)]
  rw [if_neg hlayer, if_neg hhead, if_neg (by simp [hq]), if_neg (by simp [hout])]
  rw [hlen]
  simp only [bind_tc_ok]
  rw [if_neg hpos]
  rw [hstorage]

/-! ## Structural unfold -/

/-- **Structural unfold.** Past the carrier dispatch and the four guard tests the body matches the
storage, reads both planes of the `Q32` arm, maps the GQA head, computes `slots` and the two plane
offsets with checked `usize` arithmetic, then runs the score/softmax/weight/output loops with their
refusal arms. -/
theorem word_attn_head_unfold (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab) :
    word_attn_head state layer q_head q pos out =
      (do
        let len ←
          alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize) state.occupied
            layer
        if pos ≥ len
        then ok (core.result.Result.Err (attn.AttnError.PositionOutOfRange pos len), out)
        else
          match state.storage with
          | attn.KvStorage.Q32 k v =>
            let kplane ←
              alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64))
                k layer
            let vplane ←
              alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64))
                v layer
            let group ← state.n_head / state.n_head_kv
            let kvh ← q_head / group
            let o ← lift (Usize.checked_add pos 1#usize)
            let r ← core.option.Option.ok_or o attn.AttnError.Overflow
            let cf ← core.result.Result.Insts.CoreOpsTry.branch r
            match cf with
            | core.ops.control_flow.ControlFlow.Continue val =>
              let o1 ← lift (Usize.checked_mul state.n_head_kv state.head_dim)
              let r1 ← core.option.Option.ok_or o1 attn.AttnError.Overflow
              let cf1 ← core.result.Result.Insts.CoreOpsTry.branch r1
              match cf1 with
              | core.ops.control_flow.ControlFlow.Continue val1 =>
                let o2 ← lift (Usize.checked_mul kvh state.head_dim)
                let r2 ← core.option.Option.ok_or o2 attn.AttnError.Overflow
                let cf2 ← core.result.Result.Insts.CoreOpsTry.branch r2
                match cf2 with
                | core.ops.control_flow.ControlFlow.Continue val2 =>
                  let scores := alloc.vec.Vec.with_capacity Std.I64 val
                  let (scores1, mx, err) ←
                    attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane val1 val2
                      scores core.num.I64.MIN 0#usize true none
                  match err with
                  | none =>
                    let exps := alloc.vec.Vec.with_capacity Std.I64 val
                    let (exps1, total, err1) ←
                      attn.KvState.attn_head_loop1 pos scores1 mx exps 0#i128 0#usize true none
                    match err1 with
                    | none =>
                      if total ≤ 0#i128
                      then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
                      else
                        let recip ← attn.recip_u64 total
                        let acc ←
                          alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
                        let (acc1, err2) ←
                          attn.KvState.attn_head_loop2 state.head_dim pos vplane val1 val2 exps1
                            total recip acc 0#usize true none
                        match err2 with
                        | none =>
                          let (out1, err3) ←
                            attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
                          match err3 with
                          | none => ok (core.result.Result.Ok (), out1)
                          | some e => ok (core.result.Result.Err e, out1)
                        | some e => ok (core.result.Result.Err e, out)
                    | some e => ok (core.result.Result.Err e, out)
                  | some e => ok (core.result.Result.Err e, out)
                | core.ops.control_flow.ControlFlow.Break residual =>
                  let r3 ←
                    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                      Unit (core.convert.FromSame attn.AttnError) residual
                  ok (r3, out)
              | core.ops.control_flow.ControlFlow.Break residual =>
                let r2 ←
                  core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                    Unit (core.convert.FromSame attn.AttnError) residual
                ok (r2, out)
            | core.ops.control_flow.ControlFlow.Break residual =>
              let r1 ←
                core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual
                  Unit (core.convert.FromSame attn.AttnError) residual
              ok (r1, out)
          | attn.KvStorage.F32 _ _ =>
            ok (core.result.Result.Err attn.AttnError.BadGeometry, out)) := by
  rw [word_attn_head, attn.KvState.attn_head]
  simp only [carrier_eq_of_q32 state hcarrier, bind_tc_ok]
  rw [if_neg (by decide)]
  rw [if_neg hlayer, if_neg hhead, if_neg (by simp [hq]), if_neg (by simp [hout])]
  rw [hstorage]
  rfl

/-! ## Refusal equalities: position, plane offsets and the loop stages -/

/-- **Position refusal.** A position at or past the layer's appended count returns
`Err PositionOutOfRange` exactly. -/
theorem word_attn_head_pos_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : pos ≥ len) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err (attn.AttnError.PositionOutOfRange pos len), out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_pos hpos]

/-- **`slots` overflow refusal.** When `pos + 1` leaves `usize` the body returns `Err Overflow`
exactly. -/
theorem word_attn_head_slots_overflow (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (hadd : Usize.checked_add pos 1#usize = none) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err attn.AttnError.Overflow, out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_none, bind_tc_ok]
  rw [result_branch_err]
  simp only [bind_tc_ok, result_from_residual_err]

/-- **Plane-stride overflow refusal.** When `n_head_kv · head_dim` leaves `usize` the body returns
`Err Overflow` exactly (`src/engine/attn.rs:1078`). -/
theorem word_attn_head_stride_overflow (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (hstride : Usize.checked_mul state.n_head_kv state.head_dim = none) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err attn.AttnError.Overflow, out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_none, bind_tc_ok]
  rw [result_branch_err]
  simp only [bind_tc_ok, result_from_residual_err]

/-- **Plane-first-offset overflow refusal.** When `kvh · head_dim` leaves `usize` the body returns
`Err Overflow` exactly (`src/engine/attn.rs:1079`). -/
theorem word_attn_head_first_overflow (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (stride : Usize) (hstride : Usize.checked_mul state.n_head_kv state.head_dim = some stride)
    (hfirst : Usize.checked_mul kvh state.head_dim = none) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err attn.AttnError.Overflow, out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hfirst]
  simp only [core.option.Option.ok_or_none, bind_tc_ok]
  rw [result_branch_err]
  simp only [bind_tc_ok, result_from_residual_err]

/-- **Score-stage refusal.** A score loop that stops with `err = some e` returns `Err e` exactly;
this covers the plane-range `Overflow` refusal and the `row_dot_fixed` refusals mapped to
`Overflow`. -/
theorem word_attn_head_score_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (stride first : Usize)
    (hstride : Usize.checked_mul state.n_head_kv state.head_dim = some stride)
    (hfirst : Usize.checked_mul kvh state.head_dim = some first)
    (scores : alloc.vec.Vec I64) (mx : I64) (e : attn.AttnError)
    (hloop : attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane stride first
      (alloc.vec.Vec.with_capacity Std.I64 slots) core.num.I64.MIN 0#usize true none
      = ok (scores, mx, some e)) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err e, out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hfirst]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **Softmax-stage refusal.** A softmax loop that stops with `err1 = some e` returns `Err e`
exactly; `e = ExpOutOfRange` is the `fixed_exp` refusal arm. -/
theorem word_attn_head_exp_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (stride first : Usize)
    (hstride : Usize.checked_mul state.n_head_kv state.head_dim = some stride)
    (hfirst : Usize.checked_mul kvh state.head_dim = some first)
    (scores : alloc.vec.Vec I64) (mx : I64)
    (hloop0 : attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane stride first
      (alloc.vec.Vec.with_capacity Std.I64 slots) core.num.I64.MIN 0#usize true none
      = ok (scores, mx, none))
    (exps : alloc.vec.Vec I64) (total : I128) (e : attn.AttnError)
    (hloop1 : attn.KvState.attn_head_loop1 pos scores mx
      (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      = ok (exps, total, some e)) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err e, out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hfirst]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [bind_tc_ok]
  show (do
      let (exps1, total, err1) ←
        attn.KvState.attn_head_loop1 pos scores mx
          (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      match err1 with
      | none =>
        if total ≤ 0#i128
        then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
        else
          let recip ← attn.recip_u64 total
          let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
          let (acc1, err2) ←
            attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps1 total recip
              acc 0#usize true none
          match err2 with
          | none =>
            let (out1, err3) ←
              attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
            match err3 with
            | none => ok (core.result.Result.Ok (), out1)
            | some e => ok (core.result.Result.Err e, out1)
          | some e => ok (core.result.Result.Err e, out)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err e, out)
  rw [hloop1]
  simp only [bind_tc_ok]
  rfl

theorem word_attn_head_zero_sum (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (stride first : Usize)
    (hstride : Usize.checked_mul state.n_head_kv state.head_dim = some stride)
    (hfirst : Usize.checked_mul kvh state.head_dim = some first)
    (scores : alloc.vec.Vec I64) (mx : I64)
    (hloop0 : attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane stride first
      (alloc.vec.Vec.with_capacity Std.I64 slots) core.num.I64.MIN 0#usize true none
      = ok (scores, mx, none))
    (exps : alloc.vec.Vec I64) (total : I128)
    (hloop1 : attn.KvState.attn_head_loop1 pos scores mx
      (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      = ok (exps, total, none))
    (htot : total ≤ 0#i128) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err attn.AttnError.ZeroSum, out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hfirst]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [bind_tc_ok]
  show (do
      let (exps1, total, err1) ←
        attn.KvState.attn_head_loop1 pos scores mx
          (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      match err1 with
      | none =>
        if total ≤ 0#i128
        then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
        else
          let recip ← attn.recip_u64 total
          let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
          let (acc1, err2) ←
            attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps1 total recip
              acc 0#usize true none
          match err2 with
          | none =>
            let (out1, err3) ←
              attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
            match err3 with
            | none => ok (core.result.Result.Ok (), out1)
            | some e => ok (core.result.Result.Err e, out1)
          | some e => ok (core.result.Result.Err e, out)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
  rw [hloop1]
  simp only [bind_tc_ok]
  show (do
      if total ≤ 0#i128
      then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
      else
        let recip ← attn.recip_u64 total
        let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
        let (acc1, err2) ←
          attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip acc
            0#usize true none
        match err2 with
        | none =>
          let (out1, err3) ←
            attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
          match err3 with
          | none => ok (core.result.Result.Ok (), out1)
          | some e => ok (core.result.Result.Err e, out1)
        | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
  rw [if_pos htot]


theorem word_attn_head_weight_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (stride first : Usize)
    (hstride : Usize.checked_mul state.n_head_kv state.head_dim = some stride)
    (hfirst : Usize.checked_mul kvh state.head_dim = some first)
    (scores : alloc.vec.Vec I64) (mx : I64)
    (hloop0 : attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane stride first
      (alloc.vec.Vec.with_capacity Std.I64 slots) core.num.I64.MIN 0#usize true none
      = ok (scores, mx, none))
    (exps : alloc.vec.Vec I64) (total : I128)
    (hloop1 : attn.KvState.attn_head_loop1 pos scores mx
      (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      = ok (exps, total, none))
    (htot : ¬ (total ≤ 0#i128))
    (recip : Option U64) (hrecip : attn.recip_u64 total = ok recip)
    (acc : alloc.vec.Vec I128)
    (hacc : alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim = ok acc)
    (acc1 : alloc.vec.Vec I128) (e : attn.AttnError)
    (hloop2 : attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total
      recip acc 0#usize true none = ok (acc1, some e)) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err e, out) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hfirst]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [bind_tc_ok]
  show (do
      let (exps1, total, err1) ←
        attn.KvState.attn_head_loop1 pos scores mx
          (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      match err1 with
      | none =>
        if total ≤ 0#i128
        then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
        else
          let recip ← attn.recip_u64 total
          let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
          let (acc1, err2) ←
            attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps1 total recip
              acc 0#usize true none
          match err2 with
          | none =>
            let (out1, err3) ←
              attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
            match err3 with
            | none => ok (core.result.Result.Ok (), out1)
            | some e => ok (core.result.Result.Err e, out1)
          | some e => ok (core.result.Result.Err e, out)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err e, out)
  rw [hloop1]
  simp only [bind_tc_ok]
  show (do
      if total ≤ 0#i128
      then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
      else
        let recip ← attn.recip_u64 total
        let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
        let (acc1, err2) ←
          attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip acc
            0#usize true none
        match err2 with
        | none =>
          let (out1, err3) ←
            attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
          match err3 with
          | none => ok (core.result.Result.Ok (), out1)
          | some e => ok (core.result.Result.Err e, out1)
        | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err e, out)
  rw [if_neg htot]
  rw [hrecip]
  simp only [bind_tc_ok]
  rw [hacc]
  simp only [bind_tc_ok]
  show (do
      let (acc1, err2) ←
        attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip acc
          0#usize true none
      match err2 with
      | none =>
        let (out1, err3) ←
          attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
        match err3 with
        | none => ok (core.result.Result.Ok (), out1)
        | some e => ok (core.result.Result.Err e, out1)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err e, out)
  rw [hloop2]
  simp only [bind_tc_ok]
  rfl


theorem word_attn_head_out_refuses (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (stride first : Usize)
    (hstride : Usize.checked_mul state.n_head_kv state.head_dim = some stride)
    (hfirst : Usize.checked_mul kvh state.head_dim = some first)
    (scores : alloc.vec.Vec I64) (mx : I64)
    (hloop0 : attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane stride first
      (alloc.vec.Vec.with_capacity Std.I64 slots) core.num.I64.MIN 0#usize true none
      = ok (scores, mx, none))
    (exps : alloc.vec.Vec I64) (total : I128)
    (hloop1 : attn.KvState.attn_head_loop1 pos scores mx
      (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      = ok (exps, total, none))
    (htot : ¬ (total ≤ 0#i128))
    (recip : Option U64) (hrecip : attn.recip_u64 total = ok recip)
    (acc : alloc.vec.Vec I128)
    (hacc : alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim = ok acc)
    (acc1 : alloc.vec.Vec I128)
    (hloop2 : attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total
      recip acc 0#usize true none = ok (acc1, none))
    (out1 : Slice I64) (e : attn.AttnError)
    (hloop3 : attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
      = ok (out1, some e)) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Err e, out1) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hfirst]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [bind_tc_ok]
  show (do
      let (exps1, total, err1) ←
        attn.KvState.attn_head_loop1 pos scores mx
          (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      match err1 with
      | none =>
        if total ≤ 0#i128
        then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
        else
          let recip ← attn.recip_u64 total
          let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
          let (acc1, err2) ←
            attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps1 total recip
              acc 0#usize true none
          match err2 with
          | none =>
            let (out1, err3) ←
              attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
            match err3 with
            | none => ok (core.result.Result.Ok (), out1)
            | some e => ok (core.result.Result.Err e, out1)
          | some e => ok (core.result.Result.Err e, out)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err e, out1)
  rw [hloop1]
  simp only [bind_tc_ok]
  show (do
      if total ≤ 0#i128
      then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
      else
        let recip ← attn.recip_u64 total
        let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
        let (acc1, err2) ←
          attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip acc
            0#usize true none
        match err2 with
        | none =>
          let (out1, err3) ←
            attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
          match err3 with
          | none => ok (core.result.Result.Ok (), out1)
          | some e => ok (core.result.Result.Err e, out1)
        | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err e, out1)
  rw [if_neg htot]
  rw [hrecip]
  simp only [bind_tc_ok]
  rw [hacc]
  simp only [bind_tc_ok]
  show (do
      let (acc1, err2) ←
        attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip acc
          0#usize true none
      match err2 with
      | none =>
        let (out1, err3) ←
          attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
        match err3 with
        | none => ok (core.result.Result.Ok (), out1)
        | some e => ok (core.result.Result.Err e, out1)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Err e, out1)
  rw [hloop2]
  simp only [bind_tc_ok]
  show (do
      let (out1, err3) ←
        attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
      match err3 with
      | none => ok (core.result.Result.Ok (), out1)
      | some e => ok (core.result.Result.Err e, out1)) = ok (core.result.Result.Err e, out1)
  rw [hloop3]
  simp only [bind_tc_ok]
  rfl


theorem word_attn_head_ok (state : attn.KvState) (layer q_head : Usize)
    (q out : Slice I64) (pos : Usize) (hcarrier : state.carrier = mixer.Carrier.Q32)
    (hlayer : ¬ (layer ≥ state.n_layer)) (hhead : ¬ (q_head ≥ state.n_head))
    (hq : (Slice.len q != state.head_dim) = false)
    (hout : (Slice.len out != state.head_dim) = false)
    (ktab vtab : alloc.vec.Vec (alloc.vec.Vec I64))
    (hstorage : state.storage = attn.KvStorage.Q32 ktab vtab)
    (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len)
    (hpos : ¬ (pos ≥ len))
    (kplane vplane : alloc.vec.Vec I64)
    (hkplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) ktab layer = ok kplane)
    (hvplane : alloc.vec.Vec.index
      (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Std.I64)) vtab layer = ok vplane)
    (group kvh : Usize) (hgroup : state.n_head / state.n_head_kv = ok group)
    (hkvh : q_head / group = ok kvh)
    (slots : Usize) (hadd : Usize.checked_add pos 1#usize = some slots)
    (stride first : Usize)
    (hstride : Usize.checked_mul state.n_head_kv state.head_dim = some stride)
    (hfirst : Usize.checked_mul kvh state.head_dim = some first)
    (scores : alloc.vec.Vec I64) (mx : I64)
    (hloop0 : attn.KvState.attn_head_loop0 state.head_dim state.scale q pos kplane stride first
      (alloc.vec.Vec.with_capacity Std.I64 slots) core.num.I64.MIN 0#usize true none
      = ok (scores, mx, none))
    (exps : alloc.vec.Vec I64) (total : I128)
    (hloop1 : attn.KvState.attn_head_loop1 pos scores mx
      (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      = ok (exps, total, none))
    (htot : ¬ (total ≤ 0#i128))
    (recip : Option U64) (hrecip : attn.recip_u64 total = ok recip)
    (acc : alloc.vec.Vec I128)
    (hacc : alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim = ok acc)
    (acc1 : alloc.vec.Vec I128)
    (hloop2 : attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total
      recip acc 0#usize true none = ok (acc1, none))
    (out1 : Slice I64)
    (hloop3 : attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
      = ok (out1, none)) :
    word_attn_head state layer q_head q pos out =
      ok (core.result.Result.Ok (), out1) := by
  rw [word_attn_head_unfold state layer q_head q out pos hcarrier hlayer hhead hq hout
    ktab vtab hstorage, hstorage]
  rw [hlen]
  simp only [bind_tc_ok]
  try dsimp only
  rw [if_neg hpos]
  rw [hkplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hvplane]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hgroup]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hkvh]
  simp only [bind_tc_ok]
  try dsimp only
  rw [hadd]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hstride]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hfirst]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  rw [hloop0]
  simp only [bind_tc_ok]
  show (do
      let (exps1, total, err1) ←
        attn.KvState.attn_head_loop1 pos scores mx
          (alloc.vec.Vec.with_capacity Std.I64 slots) 0#i128 0#usize true none
      match err1 with
      | none =>
        if total ≤ 0#i128
        then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
        else
          let recip ← attn.recip_u64 total
          let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
          let (acc1, err2) ←
            attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps1 total recip
              acc 0#usize true none
          match err2 with
          | none =>
            let (out1, err3) ←
              attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
            match err3 with
            | none => ok (core.result.Result.Ok (), out1)
            | some e => ok (core.result.Result.Err e, out1)
          | some e => ok (core.result.Result.Err e, out)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Ok (), out1)
  rw [hloop1]
  simp only [bind_tc_ok]
  show (do
      if total ≤ 0#i128
      then ok (core.result.Result.Err attn.AttnError.ZeroSum, out)
      else
        let recip ← attn.recip_u64 total
        let acc ← alloc.vec.from_elem core.clone.CloneI128 0#i128 state.head_dim
        let (acc1, err2) ←
          attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip acc
            0#usize true none
        match err2 with
        | none =>
          let (out1, err3) ←
            attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
          match err3 with
          | none => ok (core.result.Result.Ok (), out1)
          | some e => ok (core.result.Result.Err e, out1)
        | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Ok (), out1)
  rw [if_neg htot]
  rw [hrecip]
  simp only [bind_tc_ok]
  rw [hacc]
  simp only [bind_tc_ok]
  show (do
      let (acc1, err2) ←
        attn.KvState.attn_head_loop2 state.head_dim pos vplane stride first exps total recip acc
          0#usize true none
      match err2 with
      | none =>
        let (out1, err3) ←
          attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
        match err3 with
        | none => ok (core.result.Result.Ok (), out1)
        | some e => ok (core.result.Result.Err e, out1)
      | some e => ok (core.result.Result.Err e, out)) = ok (core.result.Result.Ok (), out1)
  rw [hloop2]
  simp only [bind_tc_ok]
  show (do
      let (out1, err3) ←
        attn.KvState.attn_head_loop3 state.head_dim out acc1 0#usize true none
      match err3 with
      | none => ok (core.result.Result.Ok (), out1)
      | some e => ok (core.result.Result.Err e, out1)) = ok (core.result.Result.Ok (), out1)
  rw [hloop3]
  simp only [bind_tc_ok]
  rfl


/-! ## Score stage: the epilogue and its bound -/

/-- The `i128` product of the score epilogue (`src/engine/attn.rs:193-196`) is always admitted:
the product of two `i64`-range values is at most `2^126` in absolute value, inside `i128`. -/
theorem score_epilogue_mul_fits (raw scale : I64) :
    I128.min ≤ (core.convert.num.FromI128I64.from raw).val *
        (core.convert.num.FromI128I64.from scale).val ∧
      (core.convert.num.FromI128I64.from raw).val *
        (core.convert.num.FromI128I64.from scale).val ≤ I128.max := by
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  have hraw : (core.convert.num.FromI128I64.from raw).val = raw.val :=
    core.convert.num.FromI128I64.from_val_eq raw
  have hscale : (core.convert.num.FromI128I64.from scale).val = scale.val :=
    core.convert.num.FromI128I64.from_val_eq scale
  have hrawb : -(2 : ℤ) ^ (63 : ℕ) ≤ raw.val ∧ raw.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    have hbw := raw.hBounds
    rw [IScalarTy.I64_numBits_eq] at hbw
    norm_num at hbw
    exact ⟨by linarith [hbw.1], by linarith [hbw.2]⟩
  have hscaleb : -(2 : ℤ) ^ (63 : ℕ) ≤ scale.val ∧ scale.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    have hbw := scale.hBounds
    rw [IScalarTy.I64_numBits_eq] at hbw
    norm_num at hbw
    exact ⟨by linarith [hbw.1], by linarith [hbw.2]⟩
  have hprod := i64_prod_range raw.val scale.val hrawb.1 hrawb.2 hscaleb.1 hscaleb.2
  rw [hI128min, hI128max, hraw, hscale]
  exact hprod

/-- **Score epilogue bound.** The single round-to-nearest reduction
`round(raw·scale / 2^32)` is within `1/(2·2^32)` of the real product
`toReal raw · toReal scale = (raw.val·scale.val)/2^64`. -/
theorem score_epilogue_bound (raw scale : I64) (y : I64)
    (hy : y.val = divRoundNearest
      ((core.convert.num.FromI128I64.from raw).val *
        (core.convert.num.FromI128I64.from scale).val) ((2 : ℤ) ^ (32 : ℕ))) :
    |toReal y - toReal raw * toReal scale| ≤ 1 / (2 * q32) := by
  have hraw : (core.convert.num.FromI128I64.from raw).val = raw.val :=
    core.convert.num.FromI128I64.from_val_eq raw
  have hscale : (core.convert.num.FromI128I64.from scale).val = scale.val :=
    core.convert.num.FromI128I64.from_val_eq scale
  have hy' : y.val = divRoundNearest (raw.val * scale.val) ((2 : ℤ) ^ (32 : ℕ)) := by
    rw [hy, hraw, hscale]
  have hmul := toReal_mul_eq raw scale
  have h := dot_rounding_bound (raw.val * scale.val) y hy'
  rw [show toReal y - toReal raw * toReal scale = toReal y - ((raw.val * scale.val : ℤ) : ℝ) / q32^2
      from by rw [hmul]; push_cast; ring]
  exact h

/-! ## Softmax stage: the pinned max-shift threshold -/

/-- The max-shift saturation threshold of `exp_at` (`src/engine/attn.rs:208`) evaluates to
`-16·2^32 = -68719476736`. -/
theorem I64_neg16_mul_ONE :
    ((-16)#i64) * (4294967296#i64) = ok ((-68719476736)#i64) := by
  have h : ((-16)#i64) * (4294967296#i64) ⦃ z => z = ((-68719476736)#i64) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The `exp_at` threshold constant is `-16·2^32`, pinned: the `engine.ONE` load and the
`(-16)·ONE` product. -/
theorem exp_at_threshold_eq :
    (do
      let i ← engine.ONE
      (-16)#i64 * i) = ok ((-68719476736)#i64) := by
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  try dsimp only
  exact I64_neg16_mul_ONE

/-- **Underflow saturation.** With the checked shift subtracted, a shift strictly below
`-16·2^32` returns `Ok 0` exactly (`src/engine/attn.rs:208-209`). -/
theorem exp_at_underflow (score mx shifted : I64)
    (hsub : I64.checked_sub score mx = some shifted)
    (h : shifted.val < (-68719476736 : ℤ)) :
    attn.exp_at score mx = ok (core.result.Result.Ok 0#i64) := by
  rw [attn.exp_at, hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  try dsimp only
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  try dsimp only
  rw [I64_neg16_mul_ONE]
  simp only [bind_tc_ok]
  try dsimp only
  have hlt : shifted < (-68719476736)#i64 := (IScalar.lt_equiv _ _).2 (by norm_num; exact h)
  rw [if_pos hlt]

/-- **Window branch.** With the checked shift subtracted, a shift at or above `-16·2^32` enters
the `fixed_exp` call (`src/engine/attn.rs:211`); the branch condition is the pinned
threshold, so the saturation edge is exact from below. -/
theorem exp_at_window (score mx shifted : I64)
    (hsub : I64.checked_sub score mx = some shifted)
    (h : (-68719476736 : ℤ) ≤ shifted.val) :
    attn.exp_at score mx =
      (do
        let r1 ← fixed_exp.fixed_exp shifted
        core.result.Result.map_err
          attn.exp_at.closure.Insts.CoreOpsFunctionFnOnceTupleExpErrorAttnError r1 ()) := by
  rw [attn.exp_at, hsub]
  simp only [lift, core.option.Option.ok_or_some, bind_tc_ok]
  rw [result_branch_ok]
  simp only [bind_tc_ok]
  try dsimp only
  rw [engine_ONE_eq]
  simp only [bind_tc_ok]
  try dsimp only
  rw [I64_neg16_mul_ONE]
  simp only [bind_tc_ok]
  try dsimp only
  have hnot : ¬ (shifted < (-68719476736)#i64) := by
    intro hc
    have := (IScalar.lt_equiv _ _).1 hc
    norm_num at this
    omega
  rw [if_neg hnot]

/-- **`ExpOutOfRange` refusal.** When the `fixed_exp` call refuses, `exp_at` returns
`Err ExpOutOfRange` exactly. -/
theorem exp_at_refuses (score mx shifted : I64)
    (hsub : I64.checked_sub score mx = some shifted)
    (h : (-68719476736 : ℤ) ≤ shifted.val)
    (he : fixed_exp.fixed_exp shifted
      = ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)) :
    attn.exp_at score mx = ok (core.result.Result.Err attn.AttnError.ExpOutOfRange) := by
  rw [exp_at_window score mx shifted hsub h, he]
  simp only [bind_tc_ok, core.result.Result.map_err_err]
  simp only [attn.exp_at.closure.Insts.CoreOpsFunctionFnOnceTupleExpErrorAttnError.call_once,
    bind_tc_ok]

/-- **In-window numerator bound.** In the `fixed_exp` window the numerator is within the proved
`fixed_exp` error `epsExp·exp + 4/2^32` of the real exponential of the shift. -/
theorem exp_at_real_bound (score mx shifted : I64)
    (hsub : I64.checked_sub score mx = some shifted)
    (h : (-68719476736 : ℤ) ≤ shifted.val)
    (hwin : window shifted) :
    ∃ v : I64,
      attn.exp_at score mx = ok (core.result.Result.Ok v) ∧
        |toReal v - Real.exp (toReal shifted)| ≤
          epsExp * Real.exp (toReal shifted) + 4 / q32 := by
  obtain ⟨v, hv, hbound⟩ := fixed_exp_refines_proved shifted hwin
  refine ⟨v, ?_, hbound⟩
  rw [exp_at_window score mx shifted hsub h]
  rw [show fixed_exp.fixed_exp shifted = word_fixed_exp shifted from rfl, hv]
  simp only [bind_tc_ok, core.result.Result.map_err_ok]

/-- **Saturation error.** In the saturated branch the numerator is `0` and the real exponential of
the shift is at most `exp(-16)`. -/
theorem exp_at_underflow_real (score mx shifted : I64)
    (hsub : I64.checked_sub score mx = some shifted)
    (h : shifted.val < (-68719476736 : ℤ)) :
    attn.exp_at score mx = ok (core.result.Result.Ok 0#i64) ∧
      |(0 : ℝ) - Real.exp (toReal shifted)| ≤ Real.exp ((-16 : ℝ)) := by
  refine ⟨exp_at_underflow score mx shifted hsub h, ?_⟩
  have hlt : toReal shifted < (-16 : ℝ) := by
    unfold toReal q32
    have hc : ((-68719476736 : ℤ) : ℝ) = (-16 : ℝ) * (2 : ℝ) ^ (32 : ℕ) := by norm_num
    have hR : (shifted.val : ℝ) < ((-68719476736 : ℤ) : ℝ) := by exact_mod_cast h
    have hq : (0 : ℝ) < (2 : ℝ) ^ (32 : ℕ) := by positivity
    nlinarith
  rw [zero_sub, abs_neg, abs_of_pos (Real.exp_pos _)]
  exact (Real.exp_le_exp).2 (le_of_lt hlt)

/-! ## Weight stage: identity and normaliser rounding bound -/

/-- **Weight identity.** `weight_at e total` is the half-biased division
`divRoundNearest (e·2^32) total` narrowed to `i64`, on the admitted window
(`src/engine/attn.rs:222-231`). -/
theorem weight_at_spec (e : I64) (total : I128)
    (hnum : 0 ≤ e.val) (htot : 0 < total.val)
    (hadd : e.val * ((2 : ℤ) ^ (32 : ℕ)) + total.val / 2 ≤ I128.max)
    (hfit : -((2 : ℤ) ^ (63 : ℕ)) ≤
        divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val ∧
      divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ w : I64,
      attn.weight_at e total = ok (core.result.Result.Ok w) ∧
        w.val = divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val := by
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  have hone : (core.convert.num.FromI128I64.from (4294967296#i64)).val = 4294967296 := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    norm_num
  have he : (core.convert.num.FromI128I64.from e).val = e.val :=
    core.convert.num.FromI128I64.from_val_eq e
  obtain ⟨num, hnum_eq, hnumv⟩ := I128_checked_mul_some
    (core.convert.num.FromI128I64.from e) (core.convert.num.FromI128I64.from 4294967296#i64)
    (by rw [hI128min, he, hone]; nlinarith [hnum])
    (by
      rw [hI128max, he, hone]
      have hbw := e.hBounds
      rw [IScalarTy.I64_numBits_eq] at hbw
      norm_num at hbw
      nlinarith [hnum, hbw.2])
  have hnumv' : num.val = e.val * ((2 : ℤ) ^ (32 : ℕ)) := by
    rw [hnumv, he, hone]
    norm_num
  obtain ⟨p, hp_eq, hpv⟩ := WP.spec_imp_exists
    (div_round_nearest_spec num total htot
      (by rw [hnumv']; exact mul_nonneg hnum (by norm_num))
      (by rw [hnumv']; exact hadd))
  have hpval : p.val = divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val := by
    rw [hpv, hnumv']
  have hmaxv : (core.convert.num.FromI128I64.from core.num.I64.MAX).val
      = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
  have hminv : (core.convert.num.FromI128I64.from core.num.I64.MIN).val
      = -((2 : ℤ) ^ (63 : ℕ)) := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MIN.val = -((2 : ℤ) ^ (63 : ℕ)) := by decide
    rw [h]
  have hmax : ¬ (p > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv, hpval] at hc
    omega
  have hmin : ¬ (p < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, hminv, hpval] at hc
    omega
  have hcastval : (IScalar.cast .I64 p).val = p.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 p.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  refine ⟨IScalar.cast .I64 p, ?_, ?_⟩
  · rw [attn.weight_at]
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [engine_ONE_eq]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hnum_eq]
    try dsimp only
    simp only [core.option.Option.ok_or_some, bind_tc_ok]
    try dsimp only
    rw [result_branch_ok]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hp_eq]
    simp only [bind_tc_ok]
    try dsimp only
    rw [if_neg hmax]
    rw [if_neg hmin]
  · rw [hcastval, hpval]

/-- **Normaliser rounding bound.** The weight division is within `1/(2·2^32)` of the exact ratio
`e / total` in Q32.32: `|w·total - e·2^32| ≤ total/2` from `div_round_nearest_abs_bound`, divided
by `total·2^32`. -/
theorem weight_rounding_bound (e : I64) (total : I128) (w : I64)
    (htot : 0 < total.val)
    (hw : w.val = divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val) :
    |toReal w - ((e.val : ℝ) / total.val)| ≤ 1 / (2 * q32) := by
  have hZ := div_round_nearest_abs_bound (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val
    (by positivity)
  have hZ'' : |divRoundNearest (e.val * ((2 : ℤ) ^ (32 : ℕ))) total.val * total.val
      - e.val * ((2 : ℤ) ^ (32 : ℕ))| ≤ total.val / 2 := by
    simpa [divRoundNearest] using hZ
  have hZ' : |w.val * total.val - e.val * ((2 : ℤ) ^ (32 : ℕ))| ≤ total.val / 2 := by
    rw [hw]
    exact hZ''
  have hdm : total.val / 2 * 2 ≤ total.val := by omega
  have hnum : |((w.val * total.val - e.val * (2 ^ (32:ℕ)) : ℤ) : ℝ)| ≤
      ((total.val / 2 : ℤ) : ℝ) := by exact_mod_cast hZ'
  have hdmR : ((total.val / 2 : ℤ) : ℝ) ≤ (total.val : ℝ) / 2 := by
    have := hdm
    have h2 : ((total.val / 2 : ℤ) : ℝ) * 2 ≤ (total.val : ℝ) := by exact_mod_cast this
    linarith
  have hcalc : toReal w - ((e.val : ℝ) / total.val)
      = ((w.val * total.val - e.val * (2 ^ (32:ℕ)) : ℤ) : ℝ) / (q32 * total.val) := by
    unfold toReal q32
    push_cast
    have htn : (total.val : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt htot)
    field_simp [htn]
    ring
  rw [hcalc, abs_div]
  have hpos : (0 : ℝ) < q32 * total.val :=
    mul_pos q32_pos (by exact_mod_cast htot)
  rw [abs_of_pos hpos]
  calc |((w.val * total.val - e.val * (2 ^ (32:ℕ)) : ℤ) : ℝ)| / (q32 * total.val)
      ≤ ((total.val / 2 : ℤ) : ℝ) / (q32 * total.val) :=
        div_le_div_of_nonneg_right hnum (by positivity)
    _ ≤ ((total.val : ℝ) / 2) / (q32 * total.val) :=
        div_le_div_of_nonneg_right hdmR (by positivity)
    _ = 1 / (2 * q32) := by
        have htn : (total.val : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt htot)
        field_simp [htn]
        try ring

/-! ## Output stage: the value epilogue and its bound -/

/-- **Output epilogue identity.** At component `i < head_dim`, with the accumulator holding `a` and
the reduction landing inside the `i64` range, one pass of `attn_head_loop3.body`
(`src/engine/attn.rs:1223-1232`) continues at `i + 1` with `out` set at `i` to the single
half-biased reduction `divRoundNearest a.val / 2^32` narrowed to `i64`. -/
theorem attn_head_loop3_body_epilogue (hd : Usize) (acc : alloc.vec.Vec I128) (out : Slice I64)
    (i : Usize) (a : I128)
    (hi : i.val < hd.val) (hiout : i.val < out.val.length)
    (hacc : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.I128) acc i = ok a)
    (hlo : I128.min ≤ a.val + 2 ^ (31 : ℕ)) (hhi : a.val + 2 ^ (31 : ℕ) ≤ I128.max)
    (hfit : -((2 : ℤ) ^ (63 : ℕ)) ≤ divRoundNearest a.val ((2 : ℤ) ^ (32 : ℕ)) ∧
      divRoundNearest a.val ((2 : ℤ) ^ (32 : ℕ)) ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ (rounded : I128) (s : Slice I64) (n : Usize),
      attn.KvState.attn_head_loop3.body hd acc out i true none =
          ok (ControlFlow.cont (s, n, true, none)) ∧
        rounded.val = divRoundNearest a.val ((2 : ℤ) ^ (32 : ℕ)) ∧
        s = out.set i (IScalar.cast .I64 rounded) := by
  obtain ⟨rounded, hround_eq, hroundval⟩ := round_shift_32_spec a hlo hhi
  have hmaxv : (core.convert.num.FromI128I64.from core.num.I64.MAX).val
      = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
  have hminv : (core.convert.num.FromI128I64.from core.num.I64.MIN).val
      = -((2 : ℤ) ^ (63 : ℕ)) := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MIN.val = -((2 : ℤ) ^ (63 : ℕ)) := by decide
    rw [h]
  have hmax : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv, hroundval] at hc
    omega
  have hmin : ¬ (rounded < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, hminv, hroundval] at hc
    omega
  obtain ⟨s, hupd_eq, hupd⟩ :=
    WP.spec_imp_exists (Slice.update_spec out i (IScalar.cast .I64 rounded) hiout)
  have himax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hbw := hd.hBounds
    rw [UScalarTy.Usize_numBits_eq] at hbw
    have hm : (Usize.max : ℕ) = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    omega
  obtain ⟨inext, hadd1, _⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) himax)
  refine ⟨rounded, s, inext, ?_, hroundval, hupd⟩
  rw [attn.KvState.attn_head_loop3.body]
  have hi' : i < hd := (UScalar.lt_equiv i hd).2 hi
  simp only [if_pos hi', if_true, hacc, lift, bind_tc_ok]
  rw [hround_eq]
  simp only [bind_tc_ok]
  rw [if_neg hmax]
  rw [if_neg hmin]
  rw [hupd_eq]
  simp only [bind_tc_ok]
  rw [hadd1]
  simp only [bind_tc_ok]
  rfl

/-- The accumulation `Σ_j w_j·v_{j,i}` over `2^64` is the real sum of products of the Q32.32
denotations: `toReal_mul_eq` over the zipped weight/value rows. -/
theorem value_sum_toReal (ws vs : List I64) :
    ((((ws.zip vs).map (fun p => p.1.val * p.2.val)).sum : ℤ) : ℝ) / q32^2 =
      ((ws.zip vs).map (fun p => toReal p.1 * toReal p.2)).sum :=
  (sum_toReal_zip (ws.zip vs)).symm

/-- **Output epilogue bound.** The output reduction is within `1/(2·2^32)` of the real
accumulation `(Σ_j w_j·v_{j,i}) / 2^64`, the sum of Q32.32 products by `value_sum_toReal`. -/
theorem value_component_rounding_bound (acc : I128) (y : I64)
    (hy : y.val = divRoundNearest acc.val ((2 : ℤ) ^ (32 : ℕ))) :
    |toReal y - ((acc.val : ℝ) / q32^2)| ≤ 1 / (2 * q32) :=
  dot_rounding_bound acc.val y hy

end

end apeiron_word
