/-
Serving-path refinement of the extracted single-token attention head
(`APEIRON-SERVE3-ATTN`; map `escalation/serve3-map.md` stage 6).

The Rust body is `src/engine/attn.rs` (`KvState::attn_head` 1053-1263); the extracted kernel
is `attn.KvState.attn_head` in `proof/ApeironWord.lean:10542`, reached through the public
wrapper `apeiron_word.word_attn_head` (`ApeironWord.lean:102923`).

The plain-mathematical reference, per query head `q_head` at position `pos`, over the causal
prefix `0..=pos` (positions past `pos` are never read):

* score `score_j = round(round(<q, k_j> / 2^32) · scale / 2^32)`, both reductions the
  half-biased integer division `divRoundNearest`;
* numerator `e_j = exp_at(score_j - max)`, the max-shift exact `fixed_exp` with the documented
  underflow saturation `0` below `-16·2^32`;
* weight `w_j = round(e_j · 2^32 / Σ_k e_k)`;
* output `out_i = round(Σ_j w_j · v_{j,i} / 2^32)`.

`word_attn_head_refuses` below lands the exact guard-chain refusal conditions; `word_attn_head_ok`
states the positive refinement. Both the positive refinement and the loop-carried refusals
(`ExpOutOfRange`, the checked `i128` `Overflow`, `ZeroSum`) are open on one residual: the four
`attn_head_loop0..3` loop specifications that state the loop results as `ok` payloads. The extracted
body destructures each loop result with a pattern `let (a, b, err) ← …`; rewriting an `ok`-equality
into that `let`/`match` needs a motive abstraction the extracted literals (`0#usize`, `0#i128`)
refuse at `instances` transparency, so a refusal stated downstream of a loop cannot be composed from
the loop equations alone. Recorded in `Numeric/Engine/Attn.lean:106-123`; this module records where
the statements stand rather than weakening them.

Trust boundary: the evidence reaches the extracted kernel `attn.KvState.attn_head` and the Q32.32
denotation in `Numeric.Scale`. It does not cover rustc/LLVM codegen of `src/engine/attn.rs`, the
kernel ABI, GPU/SPIR-V, or the `unsafe` container. Float-carrier (`F32`) bodies and the AVX2
twins are out of scope.
-/
import Numeric.Attn.SoftmaxPerturb
import Numeric.Attn.ScoreStage

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The plain-mathematical reference

The reference reads the plane contents and the integer arithmetic the documented specification
names; it does not restate the Rust loops. `attnScoreInt`, `planeRow`, `scoreList`, `valueList`,
`idealScoreList`, `denseAttnReal` and `toRealList` are the shared stage references of
`Numeric.Engine.Attn`/`Numeric.Attn.SoftmaxPerturb`, reused, not restated. -/

/-- The exact Q32.32 score of one causal key: the row dot `round(Σ_i q_i·k_i / 2^32)` followed by
the load-time scale epilogue `round(raw·scale / 2^32)` (`src/engine/attn.rs` score stage). -/
def attnHeadScore (krow q : List I64) (scale : ℤ) : ℤ :=
  attnScoreInt (((krow.zip q).map (fun p => p.1.val * p.2.val)).sum) scale

/-- The exact realised score list of the causal prefix `0..=pos` for the GQA head `kvh`: one
`attnHeadScore` per causal key row (`planeRow`). This equals the shared `scoreList`. -/
def attnHeadScores (kplane q : List I64) (kv hd kvh : ℕ) (scale : ℤ) (pos : ℕ) : List ℤ :=
  (List.range (pos + 1)).map (fun s => attnHeadScore (planeRow kplane kv hd kvh s) q scale)

/-- The realised score list is the shared `scoreList`. -/
theorem attnHeadScores_eq_scoreList (kplane q : List I64) (kv hd kvh : ℕ) (scale : ℤ)
    (pos : ℕ) :
    attnHeadScores kplane q kv hd kvh scale pos
      = scoreList kplane q kv hd kvh scale pos := by
  unfold attnHeadScores attnHeadScore scoreList
  rfl

/-- The exact realised weight list: `w_j = round(e_j · 2^32 / Σ_k e_k)`. -/
def attnHeadWeights (exps : List ℤ) (total : ℤ) : List ℤ :=
  exps.map (fun e => divRoundNearest (e * (2 : ℤ) ^ (32 : ℕ)) total)

/-- The exact output component `i`: `out_i = round(Σ_j w_j · v_{j,i} / 2^32)` where `w` is the
weight list and `vcol` the value column (`valueList`). -/
def attnHeadOut (w vcol : List ℤ) : ℤ :=
  divRoundNearest (((w.zip vcol).map (fun p => p.1 * p.2)).sum) ((2 : ℤ) ^ (32 : ℕ))

/-! ## Refusals

The guard chain of `attn_head` refuses with an exact `AttnError` before any stage runs. The four
geometric guards (`layer`, `q_head`, the two `RowLength` arms), the position guard against the
layer's appended count and the `slots = pos + 1` overflow guard are the exhaustive pre-stage
refusals on the `Q32` carrier. The `F32` carrier arm is out of scope (it dispatches to
`attn_head_f32`). -/

/-- **Headline: the exact refusal conditions of `word_attn_head`.** On a `Q32` carrier, past the
`F32` dispatch, each guard of the body returns its exact `AttnError`; the conjunction is the
complete pre-stage refusal surface (the loop-carried refusals — `ExpOutOfRange`, the checked
`i128` `Overflow` and `ZeroSum` — are downstream of these guards). -/
theorem word_attn_head_refuses (state : attn.KvState) (layer q_head pos : Usize)
    (q out : Slice I64) (hcar : carrierQ32 state) (len : Usize)
    (hlen : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.Usize)
      state.occupied layer = ok len) :
    -- `layer ≥ n_layer` refuses `LayerOutOfRange`.
    (layer ≥ state.n_layer →
        word_attn_head state layer q_head q pos out =
          ok (core.result.Result.Err
            (attn.AttnError.LayerOutOfRange layer state.n_layer), out)) ∧
    -- `q_head ≥ n_head` refuses `HeadOutOfRange`.
    (¬ (layer ≥ state.n_layer) → q_head ≥ state.n_head →
        word_attn_head state layer q_head q pos out =
          ok (core.result.Result.Err
            (attn.AttnError.HeadOutOfRange q_head state.n_head), out)) ∧
    -- a query row not `head_dim` wide refuses `RowLength`.
    (¬ (layer ≥ state.n_layer) → ¬ (q_head ≥ state.n_head) →
        q.val.length ≠ state.head_dim.val →
        word_attn_head state layer q_head q pos out =
          ok (core.result.Result.Err
            (attn.AttnError.RowLength state.head_dim (Slice.len q)), out)) ∧
    -- an output row not `head_dim` wide refuses `RowLength`.
    (¬ (layer ≥ state.n_layer) → ¬ (q_head ≥ state.n_head) →
        (Slice.len q != state.head_dim) = false → out.val.length ≠ state.head_dim.val →
        word_attn_head state layer q_head q pos out =
          ok (core.result.Result.Err
            (attn.AttnError.RowLength state.head_dim (Slice.len out)), out)) ∧
    -- a position at or past the layer's appended count refuses `PositionOutOfRange`.
    (¬ (layer ≥ state.n_layer) → ¬ (q_head ≥ state.n_head) →
        (Slice.len q != state.head_dim) = false →
        (Slice.len out != state.head_dim) = false → pos ≥ len →
        word_attn_head state layer q_head q pos out =
          ok (core.result.Result.Err (attn.AttnError.PositionOutOfRange pos len), out)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro h; exact word_attn_head_layer_refuses state layer q_head q out pos hcar h
  · intro hlayer h; exact word_attn_head_head_refuses state layer q_head q out pos hcar hlayer h
  · intro hlayer hhead h
    exact word_attn_head_q_len_refuses state layer q_head q out pos hcar hlayer hhead h
  · intro hlayer hhead hq h
    exact word_attn_head_out_len_refuses state layer q_head q out pos hcar hlayer hhead hq h
  · intro hlayer hhead hq hout h
    exact word_attn_head_pos_refuses state layer q_head q out pos hcar hlayer hhead hq hout len hlen h

/-! ## The positive target -/

/-- **Headline: the positive refinement.** On the in-domain precondition `attn_domain_v2`
(`Numeric.Attn.SoftmaxPerturb`) with the `Q32` carrier guard and `Q32` storage,
`word_attn_head` returns `Ok` and each output component is within `attnErrBoundV2` of the dense
real softmax attention of the ideal scores `<q, k_j>/√head_dim` at score-stage error
`scoreErrV2`. This is the `Numeric.AttnSoftmaxPerturb` target `attn_head_refines_v2`; it is
stated here under its serving-path name. Its proof is the residual — the four extracted loop
specifications — and is not closed here. -/
def word_attn_head_ok : Prop := attn_head_refines_v2

/-- `word_attn_head_ok` names the `attn_head_refines_v2` refinement. -/
theorem word_attn_head_ok_iff :
    word_attn_head_ok ↔ attn_head_refines_v2 := Iff.rfl

/-! ## Non-vacuity

`attnOneState` (`Numeric.Attn.SoftmaxPerturb`) is a one-layer, one-head, one-position `Q32`
state, so the layer guard hypotheses (`1 ≥ n_layer = 1`, `pos = 0`) are jointly inhabited and the
refusal fires with the documented error. The carrier hypothesis `carrierQ32` is carried explicitly
(`Std.Discriminant.read_discriminant` is an opaque instance projection; see
`Numeric/Engine/Attn.lean:237`), so the witness is stated conditional on it. -/

/-- The layer guard refuses on `attnOneState`: layer `1` is out of range (`n_layer = 1`). -/
example (h : carrierQ32 attnOneState) :
    word_attn_head attnOneState (1#usize) (0#usize) attnZeroRow (0#usize) attnZeroRow
      = ok (core.result.Result.Err
          (attn.AttnError.LayerOutOfRange (1#usize) (1#usize)), attnZeroRow) :=
  word_attn_head_layer_refuses attnOneState (1#usize) (0#usize) attnZeroRow attnZeroRow
    (0#usize) h (by simp [attnOneState])

/-- The `attn_domain_v2` precondition of the positive target is inhabited (non-vacuity of
`word_attn_head_ok`'s domain), re-exported from `Numeric.Attn.SoftmaxPerturb`. -/
theorem word_attn_head_ok_domain_witness :
    attn_domain_v2 attnOneState 0#usize 0#usize 0#usize attnZeroRow attnZeroRow
      attnOnePlanes attnOnePlanes attnOnePlane attnOnePlane :=
  attn_domain_v2_witness

end

end apeiron_word

#print axioms apeiron_word.word_attn_head_refuses
#print axioms apeiron_word.word_attn_head_ok_domain_witness
